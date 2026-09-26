#!/usr/bin/env python3
"""Evaluate the measured resource envelope for the adaptive candidate.

This postflight is deliberately separate from the observer result.  It reads
only the observer report and launcher-supplied measurements, writes a compact
technical artifact, and never turns a finite run into production clearance.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


STATUS = "fullscale_candidate_not_production_clearance"
HEADROOM_FRACTION = 0.80


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
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


def build_postflight(
    report: Mapping[str, Any],
    *,
    time_process_rss_kib: int,
    host_limit_gib: int,
    gpu_total_mib: int,
    whole_job_rss_raw: str | None,
    measurement_scope: str,
) -> dict[str, Any]:
    if report.get("engineering_status") != STATUS:
        raise ValueError("adaptive full-scale report has the wrong engineering status")
    observer_pass = report.get("pass") is True
    host_limit_kib = _positive_int(host_limit_gib, "host limit GiB") * 1024 * 1024
    gpu_total_bytes = _positive_int(gpu_total_mib, "GPU total MiB") * 1024 * 1024
    process_rss_kib = _positive_int(time_process_rss_kib, "process RSS KiB")
    cumulative_rss_kib = _positive_int(
        report.get("cumulative_peak_process_max_rss_kib"),
        "cumulative observer process RSS KiB",
    )
    allocated_bytes = _positive_int(
        report.get("cumulative_peak_torch_allocated_bytes"),
        "cumulative observer peak Torch allocated bytes",
    )
    reserved_bytes = _positive_int(
        report.get("cumulative_peak_torch_reserved_bytes"),
        "cumulative observer peak Torch reserved bytes",
    )
    if reserved_bytes < allocated_bytes:
        raise ValueError("peak Torch reserved bytes cannot be below allocated bytes")

    host_threshold_kib = math.floor(host_limit_kib * HEADROOM_FRACTION)
    gpu_threshold_bytes = math.floor(gpu_total_bytes * HEADROOM_FRACTION)
    if measurement_scope not in {"training_attempt", "report_recovery"}:
        raise ValueError(f"unknown resource measurement scope: {measurement_scope}")
    attempt_rss_pass = process_rss_kib < host_threshold_kib
    cumulative_rss_pass = cumulative_rss_kib < host_threshold_kib
    torch_allocated_pass = allocated_bytes < gpu_threshold_bytes
    attempt_measurement_applies = measurement_scope == "training_attempt"
    # GNU time and optional sstat cover only this launcher attempt.  Observer
    # checkpoint maxima are the cumulative acceptance measurements across
    # cleanly resumed attempts.
    memory_acceptance = cumulative_rss_pass and torch_allocated_pass and (
        attempt_rss_pass if attempt_measurement_applies else True
    )
    return {
        "schema": "rift_b7873200_adaptive_fullscale_postflight_v1",
        "engineering_status": STATUS,
        "observer_pass": observer_pass,
        "memory_acceptance": memory_acceptance,
        "technical_pass": bool(observer_pass and memory_acceptance),
        "production_clearance": False,
        "measurement_scope": measurement_scope,
        "resource_envelope": {
            "host_limit_gib": int(host_limit_gib),
            "gpu_total_mib": int(gpu_total_mib),
            "headroom_fraction_strictly_below": HEADROOM_FRACTION,
        },
        "measurements": {
            "attempt_gnu_time_process_max_rss_kib": process_rss_kib,
            "attempt_measurement_applies_to_acceptance": attempt_measurement_applies,
            "cumulative_observer_process_max_rss_kib": cumulative_rss_kib,
            "whole_job_max_rss_raw": whole_job_rss_raw,
            "cumulative_observer_peak_torch_allocated_bytes": allocated_bytes,
            "cumulative_observer_peak_torch_reserved_bytes": reserved_bytes,
            "gpu_total_bytes": gpu_total_bytes,
        },
        "coverage": {
            "observer_cuda_peak_scope": report.get("memory_measurement", {}).get(
                "scope", "not declared"
            ),
            "observer_process_rss_scope": "cumulative maxima retained through observer checkpoints",
            "gnu_time_scope": "current launcher attempt only",
            "whole_job_rss_scope": "current launcher attempt only when sstat supplied it",
            "historical_external_attempt_measurements": (
                "not reconstructed; observer cumulative maxima are the accepted cross-attempt evidence"
            ),
        },
        "thresholds": {
            "process_rss_kib_strictly_below": host_threshold_kib,
            "torch_allocated_bytes_strictly_below": gpu_threshold_bytes,
        },
        "checks": {
            "observer_pass": observer_pass,
            "attempt_gnu_time_rss_headroom": attempt_rss_pass,
            "cumulative_observer_rss_headroom": cumulative_rss_pass,
            "torch_allocation_headroom": torch_allocated_pass,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--time-process-rss-kib", required=True, type=int)
    parser.add_argument("--host-limit-gib", required=True, type=int)
    parser.add_argument("--gpu-total-mib", required=True, type=int)
    parser.add_argument("--whole-job-rss-raw", default=None)
    parser.add_argument(
        "--measurement-scope",
        choices=("training_attempt", "report_recovery"),
        default="training_attempt",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with Path(args.report).open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise ValueError("adaptive full-scale report must be a JSON object")
    result = build_postflight(
        report,
        time_process_rss_kib=args.time_process_rss_kib,
        host_limit_gib=args.host_limit_gib,
        gpu_total_mib=args.gpu_total_mib,
        whole_job_rss_raw=args.whole_job_rss_raw,
        measurement_scope=args.measurement_scope,
    )
    _atomic_json(Path(args.output), result)
    print(f"RIFT_B7873200_ADAPTIVE_FULLSCALE_OBSERVER_PASS={str(result['observer_pass']).lower()}")
    print(f"RIFT_B7873200_ADAPTIVE_FULLSCALE_MEMORY_PASS={str(result['memory_acceptance']).lower()}")
    print(f"RIFT_B7873200_ADAPTIVE_FULLSCALE_TECHNICAL_PASS={str(result['technical_pass']).lower()}")
    return 0 if result["technical_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
