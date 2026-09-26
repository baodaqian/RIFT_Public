"""Acceptance must fail on missing evidence, failed phases and stalled resume."""
import json
import os
from pathlib import Path
import subprocess

import pytest
import torch

from scripts_pvc import se_pvc_smoke_report as report

ROOT = Path(__file__).resolve().parents[2]


def test_production_gate_rejects_degenerate_initialization(tmp_path):
    path = tmp_path / "gate.json"
    path.write_text(json.dumps({"initialization_audit": {
        "status": "initialization_degenerate", "mean_gradient_norm": 0.,
        "saturated_fraction": 1., "nonzero_spatial_gradients": 0}}))
    assert report.report_gate(path, "production", False) == 1
    assert report.report_gate(path, "reference", True) == 0
    path.write_text('{"initialization_audit": {"status": "initialization_probe_passed"}}')
    assert report.report_gate(path, "production", False) == 0
    assert report.report_gate(path, "reference", True) == 1


def write_run(root, steps=1, status="interrupted"):
    root.mkdir(exist_ok=True)
    torch.save(dict(phase="stage1", stage1_history=[{"data_loss": 1.}] * steps,
                    view_exposures=[steps], sdf_step=0), root / "checkpoint_latest.pt")
    (root / "status.json").write_text('{"phase": "stage1"}')
    log = root / "trainer.log"
    log.write_text('banner\n' + json.dumps(dict(status=status, output=str(root))) + '\n')
    return log


@pytest.mark.parametrize("missing", ["checkpoint_latest.pt", "status.json", "trainer.log"])
def test_missing_smoke_evidence_fails(tmp_path, missing):
    log = write_run(tmp_path)
    (tmp_path / missing).unlink()
    assert report.report_run(tmp_path, [log]) == 1


@pytest.mark.parametrize("status", ["initialization_degenerate", "stage1_unconverged", "surface_unavailable"])
def test_scientific_failure_is_not_smoke_acceptance(tmp_path, status):
    log = write_run(tmp_path, status=status)
    assert report.report_run(tmp_path, [log]) == 1


def test_resume_must_advance_and_have_no_fallback(tmp_path, capsys):
    log = write_run(tmp_path)
    assert report.report_run(tmp_path, [log]) == 0
    previous = tmp_path / "before.json"
    previous.write_text(capsys.readouterr().out)
    assert report.report_run(tmp_path, [log], previous_report=previous) == 1
    log = write_run(tmp_path, steps=2)
    assert report.report_run(tmp_path, [log], previous_report=previous) == 0
    with log.open("a") as handle:
        handle.write("Aten Op fallback from XPU to CPU\n")
    assert report.report_run(tmp_path, [log], previous_report=previous) == 1


@pytest.mark.parametrize("failure", ["none", "fresh_crash", "fresh_timeout", "fresh_complete",
                                     "resume_crash", "resume_timeout", "fresh_report",
                                     "resume_report", "gate_exit", "fallback"])
def test_launcher_propagates_phase_and_acceptance_failures(tmp_path, failure):
    script = (ROOT / "scripts_pvc/smoke_sugavanam_ertin_pvc.sbatch").read_text()
    script = script.replace("source .local-setup/activate-pvc.sh", ":")
    script = script.replace("BASE=/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageC", f"BASE={tmp_path}")
    # Model process outcomes without GPU work. Reporter behavior is tested above.
    stubs = r'''
python() {
    if [[ "$*" == *se_paper_std1.json* ]]; then
        [[ $AUDIT_FAILURE == gate_exit ]] && return 42
        return 2
    fi
    if [[ "$*" == *se_pvc_smoke_report.py* && "$*" == *--run-root* ]]; then
        if [[ "$*" == *--previous-report* ]]; then
            [[ $AUDIT_FAILURE == resume_report ]] && return 1
        else
            [[ $AUDIT_FAILURE == fresh_report ]] && return 1
        fi
    fi
    echo '{}'
    return 0
}
timeout() {
    if [[ "$*" == *--resume* ]]; then
        [[ $AUDIT_FAILURE == resume_crash ]] && return 42
        [[ $AUDIT_FAILURE == resume_timeout ]] && return 124
    else
        [[ $AUDIT_FAILURE == fresh_crash ]] && return 42
        [[ $AUDIT_FAILURE == fresh_timeout ]] && return 124
        [[ $AUDIT_FAILURE == fresh_complete ]] && return 0
    fi
    [[ $AUDIT_FAILURE == fallback ]] && echo 'Aten Op fallback from XPU to CPU'
    return 75
}
'''
    proc = subprocess.run(["bash"], input=stubs + script, capture_output=True, text=True,
                          env=dict(os.environ, SLURM_JOB_ID="test", AUDIT_FAILURE=failure))
    if failure == "none":
        assert proc.returncode == 0, proc.stdout + proc.stderr
    else:
        assert proc.returncode != 0, proc.stdout + proc.stderr
