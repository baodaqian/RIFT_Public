#!/usr/bin/env python3
"""Assess bounded SpINR-style small-fit resource evidence after one job.

This postflight deliberately distinguishes a finite engineering lifecycle from
production clearance.  The bounded 16/16, 60-update smoke is never a
production, convergence, reconstruction, NVS, or baseline-comparison pass.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


HEADROOM_FRACTION = 0.80
ENGINEERING_STATUS = (
    "bounded_engineering_smoke_not_production_not_comparison_not_convergence_evidence"
)
REPORT_SCHEMA = "rift_spinr_style_b78716_smallfit_report_v1"
POSTFLIGHT_SCHEMA = "rift_spinr_style_b78716_smallfit_postflight_v1"


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer measurement")
    return int(value)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("small-fit report must be a JSON object")
    return payload


def build_postflight(
    report: Mapping[str, Any],
    *,
    time_process_rss_kib: int,
    host_limit_gib: int,
    gpu_total_mib: int,
    whole_job_rss_raw: str | None,
) -> dict[str, Any]:
    if report.get("schema") != REPORT_SCHEMA or report.get("engineering_status") != ENGINEERING_STATUS:
        raise ValueError("small-fit report has the wrong schema or engineering status")
    if report.get("lifecycle_complete") is not True or report.get("production_clearance") is not False:
        raise ValueError("small-fit report must be a completed engineering lifecycle, never production clearance")
    if report.get("logical_update_count") != 60:
        raise ValueError("small-fit report did not retain exactly sixty logical updates")
    clock = report.get("production_clock")
    if clock != {
        "updates_per_production_epoch": 800,
        "completed_production_epochs": 0,
        "updates_into_current_production_epoch": 60,
    }:
        raise ValueError("small-fit report advanced or lost the production scheduler clock")
    memory = report.get("memory")
    if not isinstance(memory, Mapping):
        raise ValueError("small-fit report lacks memory evidence")
    report_rss_bytes = _positive_int(memory.get("process_max_rss_bytes"), "report process RSS bytes")
    allocated_bytes = _positive_int(
        memory.get("peak_torch_allocated_bytes"), "report Torch allocated bytes"
    )
    reserved_bytes = _positive_int(
        memory.get("peak_torch_reserved_bytes"), "report Torch reserved bytes"
    )
    if reserved_bytes < allocated_bytes:
        raise ValueError("report Torch reserved bytes cannot be below allocated bytes")
    process_rss_kib = _positive_int(time_process_rss_kib, "time process RSS KiB")
    host_limit_kib = _positive_int(host_limit_gib, "host limit GiB") * 1024 * 1024
    gpu_total_bytes = _positive_int(gpu_total_mib, "GPU total MiB") * 1024 * 1024
    host_threshold_kib = math.floor(host_limit_kib * HEADROOM_FRACTION)
    gpu_threshold_bytes = math.floor(gpu_total_bytes * HEADROOM_FRACTION)
    process_rss_pass = process_rss_kib < host_threshold_kib
    torch_allocation_pass = allocated_bytes < gpu_threshold_bytes
    report_rss_pass = report_rss_bytes < host_threshold_kib * 1024
    memory_acceptance = process_rss_pass and report_rss_pass and torch_allocation_pass
    quadrature_records = report.get("quadrature_records")
    quadrature_pass = bool(
        isinstance(quadrature_records, Mapping)
        and isinstance(quadrature_records.get("0"), Mapping)
        and isinstance(quadrature_records.get("60"), Mapping)
        and quadrature_records["0"].get("gate_pass") is True
        and quadrature_records["60"].get("gate_pass") is True
    )
    finite_nonzero_updates = report.get("finite_nonzero_updates")
    if not isinstance(finite_nonzero_updates, bool):
        raise ValueError("small-fit report must state whether every logical update was finite and nonzero")
    lifecycle_acceptance = memory_acceptance and finite_nonzero_updates
    return {
        "schema": POSTFLIGHT_SCHEMA,
        "engineering_status": ENGINEERING_STATUS,
        "lifecycle_complete": True,
        "memory_acceptance": memory_acceptance,
        "overall_lifecycle_pass": lifecycle_acceptance,
        "production_clearance": False,
        "resource_envelope": {
            "host_limit_gib": int(host_limit_gib),
            "gpu_total_mib": int(gpu_total_mib),
            "strict_headroom_fraction": HEADROOM_FRACTION,
        },
        "measurements": {
            "time_process_max_rss_kib": process_rss_kib,
            "report_process_max_rss_bytes": report_rss_bytes,
            "whole_job_max_rss_raw": whole_job_rss_raw,
            "peak_torch_allocated_bytes": allocated_bytes,
            "peak_torch_reserved_bytes": reserved_bytes,
            "gpu_total_bytes": gpu_total_bytes,
        },
        "thresholds": {
            "process_rss_kib_strictly_below": host_threshold_kib,
            "report_rss_bytes_strictly_below": host_threshold_kib * 1024,
            "torch_allocated_bytes_strictly_below": gpu_threshold_bytes,
        },
        "checks": {
            "time_process_rss_headroom": process_rss_pass,
            "report_process_rss_headroom": report_rss_pass,
            "torch_allocation_headroom": torch_allocation_pass,
            "quadrature_gate_pass": quadrature_pass,
            "fixed_train_error_strictly_fell": report.get("fixed_train_error", {}).get("strictly_fell")
            if isinstance(report.get("fixed_train_error"), Mapping)
            else None,
            "finite_nonzero_updates": finite_nonzero_updates,
        },
        "interpretation": (
            "Resource acceptance preserves a completed bounded engineering lifecycle. "
            "It does not make the 16/16, 60-update smoke a production or comparison result."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--time-process-rss-kib", required=True, type=int)
    parser.add_argument("--host-limit-gib", required=True, type=int)
    parser.add_argument("--gpu-total-mib", required=True, type=int)
    parser.add_argument("--whole-job-rss-raw", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_postflight(
        _read_json(Path(args.report)),
        time_process_rss_kib=args.time_process_rss_kib,
        host_limit_gib=args.host_limit_gib,
        gpu_total_mib=args.gpu_total_mib,
        whole_job_rss_raw=args.whole_job_rss_raw,
    )
    _atomic_json(Path(args.output), result)
    print(f"SPINR_STYLE_B78716_SMALLFIT_MEMORY_ACCEPTANCE={str(result['memory_acceptance']).lower()}")
    print(f"SPINR_STYLE_B78716_SMALLFIT_LIFECYCLE_PASS={str(result['overall_lifecycle_pass']).lower()}")
    print("SPINR_STYLE_B78716_SMALLFIT_PRODUCTION_CLEARANCE=false")
    return 0 if result["overall_lifecycle_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
