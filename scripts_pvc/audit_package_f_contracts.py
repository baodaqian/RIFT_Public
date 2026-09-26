#!/usr/bin/env python3
"""Independent negative probes for Package F launchers and completed-head resume.

Uses isolated synthetic files and mocked child exit statuses. No real runs,
checkpoints, dataset responses, or implementation files are modified.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def launcher_probes(root):
    results = {}
    root.mkdir()
    executable = root / "child.py"
    executable.write_text(
        "import os,sys\n"
        "if os.environ.get('AUDIT_FALLBACK') == '1': print('Aten Op fallback from XPU to CPU')\n"
        "sys.exit(int(os.environ['AUDIT_FRESH_EXIT' if '--no-resume' in sys.argv else 'AUDIT_RESUME_EXIT']))\n")
    bin_dir = root / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(
        "#!" + sys.executable + "\nimport json,sys\n"
        "if 'scripts_pvc/plan_radarsplat_pvc.py' in sys.argv:\n"
        " print(json.dumps({'commands': [['true'], [" + repr(sys.executable) + ", "
        + repr(str(executable)) + ", '--no-resume']]}))\n"
        "elif sys.argv[1] == '-c':\n"
        " source=sys.argv[2]; sys.argv=['-c',*sys.argv[3:]]; exec(source)\n"
        "else: raise RuntimeError(sys.argv)\n")
    python.chmod(0o755)
    original = (ROOT / "scripts_pvc/smoke_radarsplat_b787_pvc.sbatch").read_text()
    for name, fresh, resumed, fallback in (
        ("failed_fresh_successful_resume", 17, 0, False),
        ("signals_without_checkpoints", 143, 143, False),
        ("fallback_warning", 0, 0, True),
    ):
        case = root / name
        case.mkdir()
        log = case / "launcher.log"
        # Replace only activation and log destination in the isolated launcher copy.
        text = original.replace("source .local-setup/activate-pvc.sh",
                                "export PATH=" + shlex.quote(str(bin_dir)) + ':"$PATH"')
        text = "\n".join("LOG=" + shlex.quote(str(log)) if line.startswith("LOG=") else line
                         for line in text.splitlines()) + "\n"
        script = case / "launcher.sh"
        script.write_text(text)
        env = {**os.environ, "STAGE": "train", "PVC_OUTPUT_ROOT": str(case / "output"),
               "SLURM_JOB_ID": "synthetic-audit", "SLURM_JOB_NAME": "synthetic-audit",
               "AUDIT_FRESH_EXIT": str(fresh), "AUDIT_RESUME_EXIT": str(resumed),
               "AUDIT_FALLBACK": str(int(fallback)), "RIFT_ACCELERATOR": "cpu"}
        with log.open("w") as stream:
            result = subprocess.run(["bash", str(script)], cwd=ROOT, env=env,
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=90)
        results[name] = {"first_exit": fresh, "resume_exit": resumed,
                         "fallback_injected": fallback, "launcher_exit": result.returncode,
                         "checkpoints_created": len(list(case.rglob("checkpoint_*.pt"))), "log": str(log)}
    return results


def completed_head_probe(root):
    import pytest
    import torch
    from tests.radarsplat_gotcha_fixtures import write_shard, tiny_region
    from tests.test_radarsplat_gotcha import CONFIG, test_shared_source_training_resume_and_gotcha_readout
    from rift.gotcha_dataset import GOTCHADataset
    root.mkdir()
    write_shard(root / "data/New_Transfer/shards/pass1_hh.npz")
    dataset = GOTCHADataset(root / "data", passes=[1], region=tiny_region())
    # Reuse the original lifecycle's synthetic two-step/terminal-state fixture.
    # It tests optimizer/scheduler/recipe validity with tiny synthetic scenes.
    with pytest.MonkeyPatch.context() as mp:
        test_shared_source_training_resume_and_gotcha_readout(dataset, root, mp, "budget48")
        run = root / "run"
        folder = run / "hh/checkpoints"
        latest = torch.load(folder / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
        # The original test deliberately damages final at its end; restore its
        # valid completed state before checking the PVC backend boundary.
        torch.save(latest, folder / "checkpoint_final.pt")
        from rift_pvc import radarsplat_gotcha as pvc
        from rift_pvc.radarsplat_xpu_backend import check_resume_sidecar
        direct_refused = False
        try:
            check_resume_sidecar(folder)
        except RuntimeError:
            direct_refused = True
        before = sum(s.response_reads for s in dataset.shards.values())
        result = pvc.run_gotcha(dataset=dataset, output_dir=run,
                               config={**CONFIG, "fidelity_profile": "budget48"}, device="cpu",
                               resume=run / pvc.CONTROL_FILE)
        after = sum(s.response_reads for s in dataset.shards.values())
        return {"direct_sidecar_gate_refused": direct_refused,
                "pvc_completed_head_status": result["status"],
                "sidecar_exists": (folder / "backend.json").exists(),
                "additional_response_reads": after - before}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    report = {"launcher_probes": launcher_probes(args.output_root / "launchers"),
              "completed_gotcha_head": completed_head_probe(args.output_root / "gotcha")}
    (args.output_root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
