#!/usr/bin/env python3
"""Assess one completed bounded GeRaF B787 16/4 engineering cell.

This postflight never turns the 20-target/32-update lifecycle into production
or comparison evidence.  It only validates the required engineering record
and resource headroom after the one combined allocation finishes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


REPORT_SCHEMA = "rift_geraf_b7873200_engineering_subset16x4_fit_report_v1"
POSTFLIGHT_SCHEMA = "rift_geraf_b7873200_engineering_subset16x4_fit_postflight_v1"
RESOURCE_ENVELOPE_SCHEMA = "rift_geraf_b7873200_engineering_subset16x4_resource_envelope_v1"
HEADROOM_FRACTION = 0.80
SCOPE = "bounded_engineering_smoke_not_production_not_comparison_not_convergence_evidence"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--time-process-rss-kib", type=int, required=True)
    parser.add_argument("--host-limit-gib", type=int, default=32)
    parser.add_argument("--gpu-total-mib", type=int, required=True)
    return parser.parse_args(argv)


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _finite_nonnegative(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("bounded GeRaF report must be a JSON object")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
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
) -> dict[str, Any]:
    if report.get("schema") != REPORT_SCHEMA or report.get("scope") != SCOPE:
        raise ValueError("bounded GeRaF report has the wrong schema or scope")
    if report.get("production_clearance") is not False:
        raise ValueError("bounded GeRaF report must never grant production clearance")
    identity = report.get("run_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("bounded GeRaF report lacks run identity")
    optimization = identity.get("optimization")
    if not isinstance(optimization, Mapping) or optimization.get("stop_updates") != 32:
        raise ValueError("bounded GeRaF report lost its 32-update stop budget")
    if optimization.get("scheduler_horizon_updates") != 50_000:
        raise ValueError("bounded GeRaF report shortened its 50,000-update scheduler horizon")
    subset = identity.get("engineering_subset")
    if not isinstance(subset, Mapping) or subset.get("parent_train_prefix") != 16 or subset.get(
        "parent_validation_prefix"
    ) != 4:
        raise ValueError("bounded GeRaF report lost its exact 16/4 subset identity")
    milestones = report.get("milestones")
    if not isinstance(milestones, list) or [item.get("update") if isinstance(item, Mapping) else None for item in milestones] != [0, 16, 32]:
        raise ValueError("bounded GeRaF report lacks complete 0/16/32 milestone evidence")
    for row in milestones:
        if not isinstance(row, Mapping):
            raise ValueError("bounded GeRaF report milestone is malformed")
        for key, expected_views in (
            ("fixed_train_native_mf", 16),
            ("held_out_validation_native_mf", 4),
        ):
            metrics = row.get(key)
            if not isinstance(metrics, Mapping) or int(metrics.get("views", -1)) != expected_views:
                raise ValueError(f"bounded GeRaF milestone lacks all {expected_views} {key} views")
            _finite_nonnegative(metrics.get("native_mf_magnitude_mse"), f"{key} native MSE")
            _finite_nonnegative(metrics.get("mf_magnitude_relative_mse"), f"{key} relative MSE")
        if row.get("validation_dynamic_mask_changed") is not False:
            raise ValueError("bounded GeRaF validation mutated its training dynamic-mask state")
    visits = report.get("train_view_visit_counts")
    if not isinstance(visits, Mapping) or len(visits) != 16 or any(value != 2 for value in visits.values()):
        raise ValueError("bounded GeRaF report did not use each selected training view exactly twice")
    if report.get("finite_gradients_and_updates") is not True:
        raise ValueError("bounded GeRaF report lacks finite gradient/update evidence")
    parameter_change = report.get("parameter_change")
    if not isinstance(parameter_change, Mapping) or _finite_nonnegative(
        parameter_change.get("parameter_delta_l2"), "parameter delta L2"
    ) <= 0.0:
        raise ValueError("bounded GeRaF report lacks a nonzero parameter change")
    scheduler = report.get("scheduler")
    if not isinstance(scheduler, Mapping) or scheduler != {
        "stop_updates": 32,
        "horizon_updates": 50_000,
        "last_epoch": 32,
    }:
        raise ValueError("bounded GeRaF report has an invalid scheduler clock")
    resources = report.get("resources")
    if not isinstance(resources, Mapping):
        raise ValueError("bounded GeRaF report lacks resource measurements")
    if resources.get("schema") != RESOURCE_ENVELOPE_SCHEMA:
        raise ValueError("bounded GeRaF report lacks cumulative resource-envelope provenance")
    attempt_count = _positive_int(resources.get("attempt_count"), "resource attempt count")
    cumulative_wall_seconds = _finite_nonnegative(
        resources.get("wall_seconds"), "cumulative resource wall seconds"
    )
    current_attempt_wall_seconds = _finite_nonnegative(
        resources.get("current_attempt_wall_seconds"), "current resource attempt wall seconds"
    )
    if current_attempt_wall_seconds > cumulative_wall_seconds:
        raise ValueError("bounded GeRaF resource envelope has a current attempt longer than its cumulative wall time")
    _finite_nonnegative(resources.get("cache_size_bytes"), "prepared cache size bytes")
    _positive_int(resources.get("gpu_total_bytes"), "resource GPU total bytes")
    process_rss_kib = _positive_int(time_process_rss_kib, "time process RSS KiB")
    host_limit_kib = _positive_int(host_limit_gib, "host limit GiB") * 1024 * 1024
    gpu_total_bytes = _positive_int(gpu_total_mib, "GPU total MiB") * 1024 * 1024
    allocated = _positive_int(resources.get("peak_torch_allocated_bytes"), "Torch allocated bytes")
    reserved = _positive_int(resources.get("peak_torch_reserved_bytes"), "Torch reserved bytes")
    if reserved < allocated:
        raise ValueError("bounded GeRaF reserved GPU memory cannot be below allocated memory")
    report_rss = resources.get("process_max_rss_bytes")
    report_rss_pass = report_rss is None or (
        isinstance(report_rss, int) and report_rss > 0 and report_rss < math.floor(host_limit_kib * 1024 * HEADROOM_FRACTION)
    )
    process_rss_pass = process_rss_kib < math.floor(host_limit_kib * HEADROOM_FRACTION)
    torch_allocated_pass = allocated < math.floor(gpu_total_bytes * HEADROOM_FRACTION)
    torch_reserved_pass = reserved < math.floor(gpu_total_bytes * HEADROOM_FRACTION)
    resource_acceptance = process_rss_pass and report_rss_pass and torch_allocated_pass and torch_reserved_pass
    checkpoints = report.get("checkpoints")
    if not isinstance(checkpoints, Mapping) or not all(
        isinstance(checkpoints.get(name), str) and Path(str(checkpoints[name])).is_file()
        for name in ("initial", "best", "final", "latest")
    ):
        raise ValueError("bounded GeRaF report lacks the required durable checkpoint set")
    return {
        "schema": POSTFLIGHT_SCHEMA,
        "engineering_scope": SCOPE,
        "lifecycle_complete": True,
        "production_clearance": False,
        "resource_acceptance": resource_acceptance,
        "resource_measurements": {
            "time_process_rss_kib": process_rss_kib,
            "host_limit_gib": host_limit_gib,
            "gpu_total_mib": gpu_total_mib,
            "torch_allocated_bytes": allocated,
            "torch_reserved_bytes": reserved,
            "resource_attempt_count": attempt_count,
            "cumulative_resource_wall_seconds": cumulative_wall_seconds,
            "current_attempt_wall_seconds": current_attempt_wall_seconds,
        },
        "required_evidence": {
            "prepared_targets": 20,
            "train_views": 16,
            "validation_views": 4,
            "updates": 32,
            "scheduler_horizon_updates": 50_000,
            "milestones": [0, 16, 32],
            "parameter_change": True,
            "finite_updates": True,
        },
        "next_step": (
            "inspect lifecycle and numerical evidence; this bounded engineering cell is not "
            "production or comparative performance evidence"
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    outcome = build_postflight(
        _read_json(Path(args.report)),
        time_process_rss_kib=args.time_process_rss_kib,
        host_limit_gib=args.host_limit_gib,
        gpu_total_mib=args.gpu_total_mib,
    )
    _atomic_json(Path(args.output), outcome)
    if not outcome["resource_acceptance"]:
        raise SystemExit("bounded GeRaF lifecycle completed but resource acceptance did not")
    print("GERAF_B78716_SMALLFIT_POSTFLIGHT_PASS", flush=True)


if __name__ == "__main__":
    main()
