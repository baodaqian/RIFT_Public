#!/usr/bin/env python3
"""Torch-free source and recipe checks for the Adaptive full-scale package."""
from __future__ import annotations

import ast
import io
import json
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "rift" / "b7873200_adaptive_fullscale.py"
DRIVER = ROOT / "rift" / "adaptive_training_workflow.py"
LAUNCHER = ROOT / "slurm" / "train_b7873200_adaptive_fullscale_v1.sbatch"
POSTFLIGHT = ROOT / "scripts" / "postflight_b7873200_adaptive_fullscale.py"


def _read(path: Path) -> str:
    if not path.is_file():
        raise AssertionError(f"missing required source: {path}")
    return path.read_text(encoding="utf-8")


def _parse(path: Path) -> None:
    ast.parse(_read(path), filename=str(path))


def _require(text: str, *needles: str) -> None:
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise AssertionError(f"source is missing required text: {missing}")


def _check_arithmetic() -> None:
    eligible = 48 ** 3
    parents = 0
    for _event in range(15):
        selected = min(eligible, math.ceil(eligible / 512))
        eligible -= selected
        parents += selected
    added = 7 * parents
    final_active = 48 ** 3 + added
    if (parents, added, final_active) != (3203, 22421, 133013):
        raise AssertionError(
            f"capacity arithmetic changed: parents={parents}, added={added}, final={final_active}"
        )


def _check_report_recovery_states() -> None:
    """Exercise the cheap state machine mirrored by the actual report route."""

    def decide(*, final: bool, report: bool, postflight: bool, report_pass: bool | None) -> str:
        if not final:
            raise ValueError("report recovery requires final checkpoint")
        if postflight:
            raise ValueError("report recovery refuses existing postflight")
        if report:
            if report_pass is not True:
                raise ValueError("failed report is terminal")
            return "reuse_passing_report_read_only"
        return "rebuild_missing_report"

    assert decide(final=True, report=False, postflight=False, report_pass=None) == "rebuild_missing_report"
    assert decide(final=True, report=True, postflight=False, report_pass=True) == "reuse_passing_report_read_only"
    for case in (
        dict(final=True, report=True, postflight=False, report_pass=False),
        dict(final=True, report=True, postflight=True, report_pass=True),
        dict(final=True, report=False, postflight=True, report_pass=None),
        dict(final=False, report=False, postflight=False, report_pass=None),
    ):
        try:
            decide(**case)
        except ValueError:
            pass
        else:
            raise AssertionError(f"report recovery negative state was accepted: {case}")


def _check_real_report_reuse_helper(driver_text: str) -> None:
    """Execute the driver's actual JSON reuse helper without importing Torch."""

    tree = ast.parse(driver_text, filename=str(DRIVER))
    target = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_load_passing_report"
    )
    namespace: dict[str, Any] = {
        "Any": Any,
        "Path": Path,
        "json": json,
        "os": os,
        "_resolved": lambda value: os.path.realpath(os.path.abspath(os.fspath(value))),
        "FULLSCALE_SCHEMA": "rift_b7873200_adaptive_fullscale_observer_v2",
    }
    exec(compile(ast.Module(body=[target], type_ignores=[]), str(DRIVER), "exec"), namespace)
    load_passing_report = namespace["_load_passing_report"]
    final = ROOT / ".synthetic-checkpoint_final.pth.tar"

    class MemoryReport:
        def __init__(self, payload: dict[str, Any]) -> None:
            self.payload = payload

        def open(self, *args: Any, **kwargs: Any) -> io.StringIO:
            return io.StringIO(json.dumps(self.payload))

    valid_report = {
        "schema": "rift_b7873200_adaptive_fullscale_observer_v2",
        "engineering_status": "fullscale_candidate_not_production_clearance",
        "pass": True,
        "production_clearance": False,
        "observer_checkpoint_path": str(final),
        "cumulative_peak_process_max_rss_kib": 1,
        "cumulative_peak_torch_allocated_bytes": 1,
        "cumulative_peak_torch_reserved_bytes": 1,
    }
    loaded = load_passing_report(MemoryReport(valid_report), final)
    assert loaded == valid_report
    try:
        load_passing_report(MemoryReport({**valid_report, "pass": False}), final)
    except RuntimeError:
        pass
    else:
        raise AssertionError("real report helper accepted a failed report")


def main() -> int:
    for path in (MODULE, DRIVER, POSTFLIGHT):
        _parse(path)

    module = _read(MODULE)
    driver = _read(DRIVER)
    launcher = _read(LAUNCHER)
    postflight = _read(POSTFLIGHT)

    _require(
        module,
        '"--range-model", "sum2"',
        '"--adam-eps", "1e-8"',
        'FULLSCALE_ANGULAR_FRACTION = 1.0 / 16.0',
        'FULLSCALE_SPATIAL_FLOOR = 0.0',
        'FULLSCALE_ANGULAR_FLOOR = 0.0',
        '"--regularizer-normalization", "fixed_initial"',
        '"--max-points", str(FULLSCALE_MAX_POINTS)',
        '"--adaptive-refine-every", str(FULLSCALE_REFINE_EVERY)',
        '"--adaptive-probe-every", str(FULLSCALE_PROBE_EVERY)',
        '"--point-chunk", "65536"',
        '"--pair-chunk", "64"',
        '"--phase-sign", "-1"',
        '"--seed", "42"',
        "_sample_runtime_peaks",
        "cumulative_peak_torch_allocated_bytes",
        "initialization-inclusive",
    )
    if "reset_peak_memory_stats" in module:
        raise AssertionError("observer must retain initialization-inclusive CUDA peaks")
    _require(
        driver,
        "_validate_parent_header",
        "AdaptiveFullScaleObserver",
        "FULLSCALE_SCHEMA",
        "geometry_readout_status",
        '"reserved_test_materialized": False',
        "--report-only",
        "report_from_checkpoint_state",
        "_quality_metric_artifacts",
        "_load_passing_report",
        "refuses to overwrite existing postflight evidence",
        "if report_path.exists()",
        "checkpoint_final.pth.tar",
    )
    _require(
        launcher,
        "set -euo pipefail",
        'PIPESTATUS[@]',
        "train.py --workflow adaptive-fullscale",
        "RIFT_B7873200_ADAPTIVE_FULLSCALE_OBSERVER_PASS",
        "RIFT_B7873200_ADAPTIVE_FULLSCALE_REPORT_ONLY_PASS",
        '"$1" == "report"',
        "--report-only",
        "report_recovery",
        "! -e \"$POSTFLIGHT\"",
        "scripts/validate_b7873200_adaptive_fullscale_source.py",
        "scripts/postflight_b7873200_adaptive_fullscale.py",
        "gpu-h200",
        "#SBATCH --account=gts-jromberg3-ece",
        "#SBATCH --qos=inferno",
        "#SBATCH --time=12:00:00",
        'SLURM_JOB_QOS:-}" == "inferno"',
        'SLURM_JOB_ACCOUNT:-}" == "gts-jromberg3-ece"',
    )
    if "slurm/train.sbatch" in launcher:
        raise AssertionError("adaptive launcher must not reuse the generic train.sbatch wrapper")
    _require(
        postflight,
        'STATUS = "fullscale_candidate_not_production_clearance"',
        "production_clearance",
        "technical_pass",
        "HEADROOM_FRACTION = 0.80",
        "cumulative_observer_peak_torch_allocated_bytes",
        "measurement_scope",
    )
    for path, text in ((MODULE, module), (DRIVER, driver), (LAUNCHER, launcher), (POSTFLIGHT, postflight)):
        lowered = text.lower()
        if "hashlib" in lowered or "sha256" in lowered:
            raise AssertionError(f"hash validation appeared in adaptive source: {path}")
    _check_arithmetic()
    _check_report_recovery_states()
    _check_real_report_reuse_helper(driver)
    print("RIFT_B7873200_ADAPTIVE_FULLSCALE_SOURCE_PASS")
    print("capacity_arithmetic=parents:3203 added_slots:22421 final_active:133013")
    print("report_recovery_states=missing_report:rebuild passing_report:reuse failed_report_or_postflight:reject")
    print("report_reuse_helper=actual_driver_helper_passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
