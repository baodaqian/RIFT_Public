#!/usr/bin/env python3
"""Record the resource acceptance separately from a B787 action-gate result.

This is a deliberately small postflight adapter.  It does not open radar data,
reconstruct a scene, change a checkpoint, or make a production claim.  The
launcher supplies measured process RSS from ``/usr/bin/time`` and GPU capacity
from the allocated RTX6000.  The action-gate report supplies PyTorch's peak
allocator measurements.  Both resource conditions must pass, independently of
whether the engineering action observation passed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


STATUS = "not_production_not_comparison_not_convergence_evidence"
HEADROOM_FRACTION = 0.80


def _positive_int(value: object, label: str) -> int:
    """Accept only a real, positive integer measurement.

    Resource values are release evidence, not convenient defaults.  In
    particular, accepting ``0`` for a missing allocator statistic would turn
    an unavailable measurement into a false headroom pass.  JSON reports and
    the command-line parser both produce builtin ``int`` values, so reject
    floats, booleans, strings, and absent values rather than coercing them.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be a positive integer measurement")
    if value <= 0:
        raise ValueError(f"{label} must be a positive integer measurement")
    return int(value)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _read_report(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("action-gate report must be a JSON object")
    return payload


def build_postflight(
    report: Mapping[str, Any],
    *,
    time_process_rss_kib: int,
    host_limit_gib: int,
    gpu_total_mib: int,
    whole_job_rss_raw: str | None,
) -> dict[str, Any]:
    """Evaluate the fixed resource envelope without changing the action result."""

    if report.get("engineering_status") != STATUS:
        raise ValueError("action-gate report has the wrong engineering status")
    action_pass = report.get("pass") is True
    host_limit_kib = _positive_int(host_limit_gib, "host limit GiB") * 1024 * 1024
    gpu_total_bytes = _positive_int(gpu_total_mib, "GPU total MiB") * 1024 * 1024
    process_rss_kib = _positive_int(time_process_rss_kib, "process RSS KiB")
    report_rss_kib = _positive_int(
        report.get("process_max_rss_kib"), "report process RSS KiB"
    )
    allocated_bytes = _positive_int(
        report.get("peak_torch_allocated_bytes"), "peak Torch allocated bytes"
    )
    reserved_bytes = _positive_int(
        report.get("peak_torch_reserved_bytes"), "peak Torch reserved bytes"
    )
    if reserved_bytes < allocated_bytes:
        raise ValueError("peak Torch reserved bytes cannot be below allocated bytes")
    host_threshold_kib = math.floor(host_limit_kib * HEADROOM_FRACTION)
    gpu_threshold_bytes = math.floor(gpu_total_bytes * HEADROOM_FRACTION)
    process_rss_pass = process_rss_kib < host_threshold_kib
    torch_allocated_pass = allocated_bytes < gpu_threshold_bytes
    memory_acceptance = process_rss_pass and torch_allocated_pass
    return {
        "schema": "rift_b7873200_adaptive_action_gate_postflight_v1",
        "engineering_status": STATUS,
        "resource_envelope": {
            "host_limit_gib": int(host_limit_gib),
            "gpu_total_mib": int(gpu_total_mib),
            "headroom_fraction_strictly_below": HEADROOM_FRACTION,
        },
        "action_gate_pass": action_pass,
        "memory_acceptance": memory_acceptance,
        "overall_release_pass": bool(action_pass and memory_acceptance),
        "measurements": {
            "time_process_max_rss_kib": process_rss_kib,
            "report_process_max_rss_kib": report_rss_kib,
            "whole_job_max_rss_raw": whole_job_rss_raw,
            "peak_torch_allocated_bytes": allocated_bytes,
            "peak_torch_reserved_bytes": reserved_bytes,
            "gpu_total_bytes": gpu_total_bytes,
        },
        "thresholds": {
            "process_rss_kib_strictly_below": host_threshold_kib,
            "torch_allocated_bytes_strictly_below": gpu_threshold_bytes,
        },
        "checks": {
            "action_gate_pass": action_pass,
            "process_rss_headroom": process_rss_pass,
            "torch_allocation_headroom": torch_allocated_pass,
        },
        # These are observed fit diagnostics, not evidence of a selected or
        # converged model.  Keep them plainly visible in the release artifact.
        "observed_fitting_change": {
            "initial_metrics": report.get("initial_metrics"),
            "final_metrics": report.get("final_metrics"),
            "scene_parameter_change_l2": report.get("scene_parameter_change_l2"),
            "gain_parameter_change_l2": report.get("gain_parameter_change_l2"),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--time-process-rss-kib", required=True, type=int)
    parser.add_argument("--host-limit-gib", required=True, type=int)
    parser.add_argument("--gpu-total-mib", required=True, type=int)
    parser.add_argument(
        "--whole-job-rss-raw",
        default=None,
        help="optional scheduler-side MaxRSS observation; retained verbatim when available",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = _read_report(Path(args.report))
    result = build_postflight(
        report,
        time_process_rss_kib=args.time_process_rss_kib,
        host_limit_gib=args.host_limit_gib,
        gpu_total_mib=args.gpu_total_mib,
        whole_job_rss_raw=args.whole_job_rss_raw,
    )
    _atomic_json(Path(args.output), result)
    print(f"RIFT_B7873200_ACTION_GATE_ACTION_PASS={str(result['action_gate_pass']).lower()}")
    print(f"RIFT_B7873200_ACTION_GATE_MEMORY_PASS={str(result['memory_acceptance']).lower()}")
    print(f"RIFT_B7873200_ACTION_GATE_RELEASE_PASS={str(result['overall_release_pass']).lower()}")
    return 0 if result["overall_release_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
