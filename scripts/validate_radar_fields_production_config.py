#!/usr/bin/env python3
"""Torch-free structural checks for the proposed B787 Radar Fields package."""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "protocols" / "radar_fields_b7873200_production_v1.json"
DEFAULT_LAUNCHER = ROOT / "slurm" / "train_radar_fields_b7873200_production_v1.sbatch"
DEFAULT_READOUT = ROOT / "scripts" / "readout_radar_fields_b7873200_native.py"
DEFAULT_READOUT_VALIDATOR = ROOT / "scripts" / "validate_radar_fields_native_readout.py"
DEFAULT_RESOURCE_INSPECTOR = ROOT / "scripts" / "inspect_radar_fields_training_resource.py"
CANONICAL_METRIC_DOMAIN = "normalized_dB_range_power_intensity"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(left: Any, right: float) -> bool:
    return isinstance(left, (int, float)) and math.isclose(
        float(left), float(right), rel_tol=2.0e-5, abs_tol=2.0e-7
    )


def load_json(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    require(isinstance(value, Mapping), f"{path} must contain a JSON object")
    return value


def validate(
    config: Mapping[str, Any],
    launcher: str,
    readout: str,
    readout_validator: str,
    resource_inspector: str,
) -> None:
    require(config.get("schema_version") == 1, "unsupported config schema")
    require(config.get("name") == "radar_fields_b7873200_production_v1", "wrong config name")
    require(config.get("status") == "review_release", "config must remain an explicit review release")
    require(config.get("production_clearance") is False, "production clearance must remain false")

    data = config["data"]
    require(data["dataset"] == "B787 sphere10k", "wrong dataset")
    require(data["response_shape"] == [10000, 16, 16, 1, 600], "wrong acquisition shape")
    require(data["roles"] == {"train": 3200, "validation": 1000, "test": 1000, "unused": 4800}, "wrong sealed roles")
    require(data["test_and_unused_response_access"] == "sealed_during_training", "test access is not sealed")
    require(data["acquisition"] == {"tx": 16, "rx": 16, "frequencies": 600, "pairs_per_view": 256}, "wrong acquisition contract")

    training = config["training"]
    require(training["sealed_protocol"] is True, "sealed protocol is disabled")
    require(training["num_train"] == 3200 and training["num_val"] == 1000 and training["num_test"] == 1000, "wrong trainer split")
    require(training["steps"] == 8000 and training["steps"] > 1, "production clock is not the proposed 8000 updates")
    require(training["view_batch"] == 1 and training["train_pairs"] == 0 and training["val_pairs"] == 0, "full-pair single-view contract changed")
    require(training["eval_every"] == 2000 and training["checkpoint_every"] == 500, "unexpected evaluation/checkpoint cadence")
    require(training["eval_max_views"] == 0 and training["stats_max_views"] == 0, "a production limiter is enabled")
    require(close(training["lr"], 1.0e-3), "unexpected initial learning rate")
    require(training["range_law"] == "released", "native range law changed")
    require(close(training["intensity_offset"], 0.05) and close(training["intensity_scaler"], 1.0), "native intensity transform changed")
    require(training["granularity"] == 48 and close(training["extent"], 0.15), "support grid changed")

    schedule = config["schedule"]
    require("8000" in schedule["learning_rate"]["formula"], "LR schedule is not tied to the full clock")
    for key, expected in {
        "step_0": 1.0e-3,
        "step_2000": 5.623413e-4,
        "step_4000": 3.162278e-4,
        "step_6000": 1.778279e-4,
        "step_8000": 1.0e-4,
    }.items():
        require(close(schedule["learning_rate"][key], expected), f"LR milestone {key} is inconsistent")
    require("7999" in schedule["feature_mask"]["formula"], "feature mask schedule is compressed or unspecified")
    for key, expected in {
        "step_1_visible_fraction": 0.430118,
        "step_2000_visible_fraction": 0.659637,
        "step_4000_visible_fraction": 0.854306,
        "step_6000_visible_fraction": 0.984362,
        "step_8000_visible_fraction": 1.0,
    }.items():
        require(close(schedule["feature_mask"][key], expected), f"feature-mask milestone {key} is inconsistent")

    objective = config["objective"]
    require(objective["objective_change_for_padding"] is False, "padding was silently turned into an objective change")
    require(objective["metric_domain"] == CANONICAL_METRIC_DOMAIN, "wrong native metric domain")

    resources = config["resources"]
    require(resources["account"] == "gts-jromberg3-ece" and resources["qos"] == "inferno" and resources["partition"] == "gpu-rtx6000", "wrong resource pool")
    require(resources["gres"] == "gpu:rtx_6000:1" and resources["cpus"] == 6 and resources["memory_gb"] == 32, "wrong resource envelope")
    require(resources["walltime_hours"] == 12, "unexpected walltime proposal")

    readout_spec = config["readout"]
    require(readout_spec["roles"] == ["val", "test"] and readout_spec["pair_count"] == 256, "readout roles/pairs changed")
    require(readout_spec["metric_domain"] == CANONICAL_METRIC_DOMAIN, "readout domain disagrees with objective domain")
    require(set(readout_spec["regions"]) == {"whole_roi", "valid_region", "padded_region"}, "readout regions are incomplete")
    require(readout_spec["zero_reference"].startswith("zero RCS passed through"), "zero reference is not native")
    require(readout_spec["raw_coherent_or_power_metrics"] is False, "readout silently claims coherent metrics")

    require("protocols/radar_fields_b7873200_production_v1.json" in launcher, "launcher does not consume the reviewed config")
    require("#SBATCH --account=gts-jromberg3-ece" in launcher and "#SBATCH --qos=inferno" in launcher and "#SBATCH --time=12:00:00" in launcher, "launcher resource directives do not match the Inferno 12-hour package")
    require("readout_radar_fields_b7873200_native.py" in launcher, "launcher lacks the native readout")
    require('"metric_domain": NORMALIZED_DB_INTENSITY_DOMAIN' in readout, "readout emission does not use the canonical domain constant")
    require("RF_B7873200_PRODUCTION_V1_PASS" in launcher, "launcher lacks its final marker")
    require("--readout-only" in launcher, "launcher lacks narrow completed-checkpoint readout recovery")
    require("training_resource.json" in launcher, "launcher lacks measured training resource accounting")
    require("resume_summary.json" in launcher, "launcher lacks resumed-attempt evidence assessment")
    require("srun --kill-on-bad-exit=1 --ntasks=1" in launcher, "launcher lacks explicit single-task Slurm steps")
    require("export SLURM_EXPORT_ENV=ALL" in launcher, "launcher does not restore environment export for srun steps")
    require("inspect_radar_fields_training_resource.py" in launcher, "launcher lacks prior-resource inspection")
    require("--reject-failed" in launcher, "launcher does not reject failed prior training evidence")
    require("readout_resource.json" in readout, "readout lacks measured resource accounting")
    ast.parse(readout, filename="readout_radar_fields_b7873200_native.py")
    ast.parse(readout_validator, filename="validate_radar_fields_native_readout.py")
    ast.parse(resource_inspector, filename="inspect_radar_fields_training_resource.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--launcher", type=Path, default=DEFAULT_LAUNCHER)
    parser.add_argument("--readout", type=Path, default=DEFAULT_READOUT)
    parser.add_argument("--readout-validator", type=Path, default=DEFAULT_READOUT_VALIDATOR)
    parser.add_argument("--resource-inspector", type=Path, default=DEFAULT_RESOURCE_INSPECTOR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    launcher = args.launcher.read_text(encoding="utf-8")
    readout = args.readout.read_text(encoding="utf-8")
    readout_validator = args.readout_validator.read_text(encoding="utf-8")
    resource_inspector = args.resource_inspector.read_text(encoding="utf-8")
    validate(config, launcher, readout, readout_validator, resource_inspector)
    print("Radar Fields production config, launcher, and readout checks passed")
    print("RF_B7873200_PRODUCTION_CONFIG_PASS")


if __name__ == "__main__":
    main()
