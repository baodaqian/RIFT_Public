#!/usr/bin/env python3
"""Validate the full3200 report and resource envelope without opening data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Mapping


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return number


def _check_native(metrics: Mapping[str, object]) -> None:
    for role in ("initial", "final"):
        readout = _mapping(metrics.get(role), f"native metrics {role}")
        for split in ("train", "validation"):
            values = _mapping(readout.get(split), f"native metrics {role}.{split}")
            for key in ("loss", "residual_power", "zero_reference_power", "relative_mse", "relative_l2"):
                _finite(values.get(key), f"native metrics {role}.{split}.{key}")
            if _finite(values["zero_reference_power"], "zero reference", positive=True) <= 0:
                raise ValueError("native zero-reference power must be positive")
            if _finite(values["residual_power"], "residual power") < 0:
                raise ValueError("native residual power must be nonnegative")


def _check_resource(record: Mapping[str, object], *, host_limit_gib: float, time_limit_hours: float) -> dict[str, object]:
    if record.get("schema") != "rift_sugavanam_ertin_b7873200_full_resource_accounting_v2":
        raise ValueError("resource accounting schema changed")
    attempts = record.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("resource accounting lacks per-attempt evidence")
    limit_seconds = time_limit_hours * 3600.0
    declared_limit = _finite(record.get("per_attempt_wall_limit_seconds"), "declared per-attempt wall limit")
    if not math.isclose(declared_limit, limit_seconds, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError("resource accounting limit does not match the reviewed allocation limit")
    cumulative = 0.0
    compute_phase_count = {"stage1": 0, "stage2": 0}
    for index, attempt_value in enumerate(attempts, start=1):
        attempt = _mapping(attempt_value, f"resource attempt {index}")
        if attempt.get("attempt_index") != index:
            raise ValueError("resource attempt indices are not ordered")
        total_wall = _finite(attempt.get("total_wall_seconds"), f"attempt {index} wall time")
        if total_wall > limit_seconds:
            raise ValueError(f"attempt {index} exceeded the per-allocation wall envelope")
        phases = _mapping(attempt.get("phases"), f"resource attempt {index} phases")
        for phase in ("stage1", "stage2"):
            if phase not in phases:
                raise ValueError(f"resource attempt {index} omits explicit {phase} phase evidence")
            phase_record = _mapping(phases[phase], f"attempt {index} {phase}")
            execution = phase_record.get("execution")
            if execution == "not_started":
                continue
            wall = _finite(phase_record.get("wall_seconds"), f"attempt {index} {phase} wall time")
            if wall > limit_seconds:
                raise ValueError(f"attempt {index} {phase} exceeded the per-allocation wall envelope")
            rss = _finite(phase_record.get("process_peak_rss_bytes"), f"attempt {index} {phase} peak RSS", positive=True)
            if rss > host_limit_gib * 1024**3:
                raise ValueError(f"attempt {index} {phase} exceeded the memory envelope")
            for key in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
                gpu_peak = _finite(phase_record.get(key), f"attempt {index} {phase} {key}")
                if gpu_peak < 0:
                    raise ValueError(f"attempt {index} {phase} has a negative GPU peak")
            if execution != "reused_terminal_bundle":
                compute_phase_count[phase] += 1
                cumulative += wall
    recorded_cumulative = _finite(
        record.get("cumulative_work_wall_seconds"), "cumulative work wall time"
    )
    if not math.isclose(recorded_cumulative, cumulative, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError("resource cumulative work does not equal its phase ledger")
    if compute_phase_count["stage1"] == 0 or compute_phase_count["stage2"] == 0:
        raise ValueError("completed experiment lacks Stage-1 or Stage-2 compute evidence")
    return {
        "attempt_count": len(attempts),
        "cumulative_work_wall_seconds": cumulative,
        "stage1_compute_phases": compute_phase_count["stage1"],
        "stage2_compute_phases": compute_phase_count["stage2"],
    }


def _check_time_file(path: Path, *, host_limit_gib: float, time_limit_hours: float) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    result: dict[str, object] = {"path": str(path)}
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if match:
        rss_kib = int(match.group(1))
        result["maximum_rss_kib"] = rss_kib
        if rss_kib * 1024 > host_limit_gib * 1024**3:
            raise ValueError("/usr/bin/time peak RSS exceeded the declared host envelope")
    else:
        raise ValueError("/usr/bin/time output lacks a peak-RSS measurement")
    elapsed = re.search(r"Elapsed \(wall clock\) time .*?:\s*([0-9:.]+)", text)
    if elapsed:
        elapsed_text = elapsed.group(1)
        result["elapsed_text"] = elapsed_text
        pieces = [float(piece) for piece in elapsed_text.split(":")]
        seconds = pieces[0] * 60.0 + pieces[1] if len(pieces) == 2 else pieces[0] * 3600.0 + pieces[1] * 60.0 + pieces[2]
        result["elapsed_seconds"] = seconds
        if seconds > time_limit_hours * 3600.0:
            raise ValueError("/usr/bin/time wall duration exceeded the per-allocation envelope")
    else:
        raise ValueError("/usr/bin/time output lacks a wall-duration measurement")
    return result


def validate(report_path: Path, *, host_limit_gib: float, time_limit_hours: float,
             time_file: Path | None, process_rss_kib: int | None,
             gpu_total_mib: int | None) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != "rift_sugavanam_ertin_b7873200_full_report_v1":
        raise ValueError("full report schema changed")
    status = report.get("status")
    if status not in {"complete", "scientific_negative_topology"}:
        raise ValueError(f"full report is not a completed experiment outcome: {status!r}")
    roles = _mapping(report.get("roles"), "roles")
    expected_roles = {"train_count": 3200, "validation_count": 1000, "reserved_test_count": 1000, "unused_count": 4800}
    for key, expected in expected_roles.items():
        if roles.get(key) != expected:
            raise ValueError(f"role count changed for {key}")
    if roles.get("test_and_unused_response_materialized") is not False:
        raise ValueError("reserved test/unused response payload was materialized")
    normalization = _mapping(report.get("normalization"), "normalization")
    if normalization.get("scope") != "selected_parent_train_only" or normalization.get("source_count") != 3200:
        raise ValueError("normalization is not train-only full scope")
    _finite(normalization.get("raw_complex_rms"), "train-only raw RMS", positive=True)
    _finite(normalization.get("zero_reference_train_mse"), "train-only zero-reference MSE", positive=True)
    native_metrics = _mapping(_mapping(report.get("stage1"), "stage1").get("native_metrics"), "native metrics")
    _check_native(native_metrics)
    if native_metrics.get("original_readout_status") not in {"captured_first_attempt", "preserved_first_attempt"}:
        raise ValueError("native metrics do not preserve the original first-attempt readout")
    handoff = _mapping(report.get("stage1_to_stage2"), "Stage-1 to Stage-2 handoff")
    if handoff.get("stage2_bundle_only_input") is not True or handoff.get("raw_npz_opened_by_stage2") is not False:
        raise ValueError("Stage-2 source boundary is not bundle-only")
    stage2 = _mapping(report.get("stage2"), "stage2")
    recipe = _mapping(stage2.get("recipe"), "stage2 recipe")
    for key, expected in (("steps", 5000), ("init_steps", 1000), ("init_batch", 2048),
                          ("init_log_every", 100), ("n_fourier", 9), ("hidden_dim", 512), ("n_layers", 8)):
        if recipe.get(key) != expected:
            raise ValueError(f"full Stage-2 recipe changed for {key}")
    if float(recipe.get("init_lr")) != 5.0e-4:
        raise ValueError("full Stage-2 initialization learning rate changed")
    resource_summary = _check_resource(
        _mapping(json.loads(Path(report["resource_accounting"]).read_text(encoding="utf-8")), "resource accounting"),
        host_limit_gib=host_limit_gib,
        time_limit_hours=time_limit_hours,
    )
    external_resource: dict[str, object] = {}
    if time_file is not None:
        external_resource["time"] = _check_time_file(time_file, host_limit_gib=host_limit_gib, time_limit_hours=time_limit_hours)
    if process_rss_kib is not None:
        if process_rss_kib < 0 or process_rss_kib * 1024 > host_limit_gib * 1024**3:
            raise ValueError("whole-job process RSS exceeded the declared host envelope")
        external_resource["whole_job_process_rss_kib"] = process_rss_kib
    if gpu_total_mib is not None:
        if gpu_total_mib <= 0:
            raise ValueError("GPU total memory measurement is invalid")
        external_resource["gpu_total_mib"] = gpu_total_mib

    provisional = report.get("provisional_geometry")
    outcome: dict[str, object] = {
        "schema": "rift_sugavanam_ertin_b7873200_full_postflight_v1",
        "status": status,
        "accepted_experiment_outcome": True,
        "production_clearance": False,
        "resource_summary": resource_summary,
        "external_resource": external_resource,
    }
    if status == "scientific_negative_topology":
        if report.get("technical_execution_failure") is not False:
            raise ValueError("scientific-negative report is marked as a technical failure")
        if report.get("complete_package_published") is not False or report.get("production_clearance") is not False:
            raise ValueError("scientific-negative report incorrectly claims a complete or production result")
        provisional = _mapping(provisional, "provisional geometry")
        audit_path = Path(str(provisional.get("directory"))) / "geometry_audit.json"
        audit = _mapping(json.loads(audit_path.read_text(encoding="utf-8")), "provisional geometry audit")
        topology = _mapping(audit.get("topology"), "provisional topology audit")
        if topology.get("passed") is not False:
            raise ValueError("scientific-negative topology outcome lacks a failed topology gate")
        outcome["provisional_geometry"] = str(provisional.get("directory"))
        outcome["topology_gate_passed"] = False
    else:
        if report.get("complete_package_published") is not True:
            raise ValueError("complete report lacks a complete package")
        outcome["topology_gate_passed"] = True
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--host-limit-gib", type=float, default=64.0)
    parser.add_argument("--time-limit-hours", type=float, default=12.0)
    parser.add_argument("--time-file", type=Path, default=None)
    parser.add_argument("--process-rss-kib", type=int, default=None)
    parser.add_argument("--gpu-total-mib", type=int, default=None)
    args = parser.parse_args()
    result = validate(
        args.report,
        host_limit_gib=args.host_limit_gib,
        time_limit_hours=args.time_limit_hours,
        time_file=args.time_file,
        process_rss_kib=args.process_rss_kib,
        gpu_total_mib=args.gpu_total_mib,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("SE_B7873200_FULL_POSTFLIGHT_ACCEPTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
