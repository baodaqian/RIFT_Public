#!/usr/bin/env python3
"""Inspect prior RF training resource evidence without changing it."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


def inspect_mapping(record: object, path: Path) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        return {"status": "failed", "path": str(path), "reason": "evidence is not a JSON object"}
    required = (
        "scope",
        "host_peak_rss_bytes",
        "host_budget_bytes",
        "gpu_peak_allocated_bytes",
        "gpu_total_bytes",
        "host_headroom_pass",
        "gpu_headroom_pass",
        "headroom_gate_pass",
    )
    missing = [key for key in required if key not in record]
    if missing:
        return {"status": "failed", "path": str(path), "reason": f"missing fields: {missing}"}
    if record["scope"] != "training Python process only":
        return {"status": "failed", "path": str(path), "reason": "evidence scope is not the training process"}
    numeric = (
        "host_peak_rss_bytes",
        "host_budget_bytes",
        "gpu_peak_allocated_bytes",
        "gpu_total_bytes",
    )
    if any(
        not isinstance(record[key], (int, float))
        or not math.isfinite(float(record[key]))
        or float(record[key]) <= 0
        for key in numeric
    ):
        return {"status": "failed", "path": str(path), "reason": "nonpositive/nonfinite resource field"}
    if record["headroom_gate_pass"] is not True:
        return {"status": "failed", "path": str(path), "reason": "recorded training headroom gate failed"}
    if record["host_headroom_pass"] is not True or record["gpu_headroom_pass"] is not True:
        return {"status": "failed", "path": str(path), "reason": "recorded component headroom gate failed"}
    return {
        "status": "passed",
        "path": str(path),
        "scope": str(record["scope"]),
        "host_peak_rss_bytes": int(record["host_peak_rss_bytes"]),
        "gpu_peak_allocated_bytes": int(record["gpu_peak_allocated_bytes"]),
    }


def inspect_record(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "unavailable",
            "path": str(path),
            "reason": "training_resource.json is absent",
        }
    try:
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, ValueError) as exc:
        return {"status": "failed", "path": str(path), "reason": f"unreadable evidence: {exc}"}
    return inspect_mapping(record, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--reject-failed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = inspect_record(args.path)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    if args.reject_failed and result["status"] == "failed":
        raise SystemExit(96)


if __name__ == "__main__":
    main()
