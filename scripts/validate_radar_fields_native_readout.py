#!/usr/bin/env python3
"""Validate the Torch-free schema and region accounting of an RF readout."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "protocols" / "radar_fields_b7873200_production_v1.json"
CANONICAL_METRIC_DOMAIN = "normalized_dB_range_power_intensity"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, Mapping), f"{path} must contain a JSON object")
    return value


def validate_metric(metric: Mapping[str, Any], label: str) -> int:
    required = ("measurement_count", "mean_abs_error", "rel_mse", "rmse", "psnr_db")
    require(all(key in metric for key in required), f"{label} lacks a required metric")
    count = int(metric["measurement_count"])
    require(count > 0 and count == metric["measurement_count"], f"{label} has an invalid count")
    for key in required[1:]:
        require(math.isfinite(float(metric[key])), f"{label}.{key} is nonfinite")
    return count


def validate(output: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    require(output.get("artifact_schema") == "radar_fields_native_readout_v1", "wrong readout schema")
    require(output.get("status") == "measurement_only", "readout is not measurement-only")
    require(output.get("metric_domain") == CANONICAL_METRIC_DOMAIN, "wrong readout domain")
    require(config.get("readout", {}).get("metric_domain") == CANONICAL_METRIC_DOMAIN, "config readout domain disagrees")
    require(output.get("objective_changed") is False, "readout reports an objective change")
    require(output.get("pair_count") == 256 and output.get("frequency_count") == 600, "wrong native dimensions")
    require(math.isfinite(float(output["zero_rcs_native_output"])), "nonfinite zero-RCS reference")
    resource = output.get("resource_usage")
    require(isinstance(resource, Mapping), "readout lacks resource accounting")
    for key in ("host_peak_rss_bytes", "host_budget_bytes", "gpu_peak_allocated_bytes", "gpu_total_bytes"):
        require(int(resource[key]) > 0, f"readout resource field {key} is not positive")
    require(resource.get("host_headroom_pass") is True, "readout host headroom gate did not pass")
    require(resource.get("gpu_headroom_pass") is True, "readout GPU headroom gate did not pass")
    require(resource.get("headroom_gate_pass") is True, "readout resource gate did not pass")

    expected_roles = list(config["readout"]["roles"])
    roles = output.get("roles")
    require(roles == expected_roles, "readout roles differ from the reviewed config")
    role_results = output.get("role_results")
    require(isinstance(role_results, Mapping), "readout lacks role results")
    for role in expected_roles:
        role_result = role_results.get(role)
        require(isinstance(role_result, Mapping), f"missing readout role {role}")
        regions = role_result.get("regions")
        require(isinstance(regions, Mapping), f"role {role} lacks region results")
        counts = {}
        for region in ("whole_roi", "valid_region", "padded_region"):
            entry = regions.get(region)
            require(isinstance(entry, Mapping), f"role {role} lacks {region}")
            prediction = entry.get("prediction")
            zero = entry.get("zero_reference")
            require(isinstance(prediction, Mapping), f"role {role}/{region} lacks prediction metrics")
            require(isinstance(zero, Mapping), f"role {role}/{region} lacks zero metrics")
            counts[region] = validate_metric(prediction, f"{role}/{region}/prediction")
            require(validate_metric(zero, f"{role}/{region}/zero_reference") == counts[region], f"{role}/{region} count mismatch")
        require(counts["whole_roi"] == counts["valid_region"] + counts["padded_region"], f"{role} ROI partition does not add")
        fraction = float(role_result["padded_measurement_fraction"])
        require(math.isclose(fraction, counts["padded_region"] / counts["whole_roi"], rel_tol=1.0e-12), f"{role} padded fraction mismatch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate(load_json(args.output), load_json(args.config))
    print("Radar Fields native readout schema and ROI partition checks passed")
    print("RF_B7873200_NATIVE_READOUT_SCHEMA_PASS")


if __name__ == "__main__":
    main()
