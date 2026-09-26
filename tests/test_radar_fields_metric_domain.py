#!/usr/bin/env python3
"""Regression checks for the Torch-free Radar Fields readout domain contract."""

from __future__ import annotations

import importlib.util
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "scripts" / "validate_radar_fields_native_readout.py"
PRODUCER_PATH = ROOT / "rift" / "radar_fields.py"
SPEC = importlib.util.spec_from_file_location("rf_native_readout_validator", VALIDATOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load Radar Fields readout validator")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _metric(count: int) -> dict[str, float | int]:
    return {
        "measurement_count": count,
        "mean_abs_error": 0.1,
        "rel_mse": 0.2,
        "rmse": 0.3,
        "psnr_db": 4.0,
    }


def _fixture(domain: str) -> tuple[dict[str, object], dict[str, object]]:
    regions = {
        "whole_roi": {"prediction": _metric(3), "zero_reference": _metric(3)},
        "valid_region": {"prediction": _metric(2), "zero_reference": _metric(2)},
        "padded_region": {"prediction": _metric(1), "zero_reference": _metric(1)},
    }
    role_results = {
        role: {"regions": regions, "padded_measurement_fraction": 1.0 / 3.0}
        for role in ("val", "test")
    }
    output = {
        "artifact_schema": "radar_fields_native_readout_v1",
        "status": "measurement_only",
        "metric_domain": domain,
        "objective_changed": False,
        "pair_count": 256,
        "frequency_count": 600,
        "zero_rcs_native_output": -1.0,
        "resource_usage": {
            "host_peak_rss_bytes": 1,
            "host_budget_bytes": 2,
            "gpu_peak_allocated_bytes": 1,
            "gpu_total_bytes": 2,
            "host_headroom_pass": True,
            "gpu_headroom_pass": True,
            "headroom_gate_pass": True,
        },
        "roles": ["val", "test"],
        "role_results": role_results,
    }
    config = {"readout": {"roles": ["val", "test"], "metric_domain": domain}}
    return output, config


def test_canonical_domain_is_accepted() -> None:
    output, config = _fixture(MODULE.CANONICAL_METRIC_DOMAIN)
    MODULE.validate(output, config)


def test_producer_constant_agrees_with_validator() -> None:
    tree = ast.parse(PRODUCER_PATH.read_text(encoding="utf-8"), filename=str(PRODUCER_PATH))
    assignments = {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id == "NORMALIZED_DB_INTENSITY_DOMAIN"
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }
    assert assignments.get("NORMALIZED_DB_INTENSITY_DOMAIN") == MODULE.CANONICAL_METRIC_DOMAIN


def test_mismatched_domain_is_rejected() -> None:
    output, config = _fixture(MODULE.CANONICAL_METRIC_DOMAIN)
    output["metric_domain"] = "normalized_dB_range_power"
    try:
        MODULE.validate(output, config)
    except AssertionError:
        return
    raise AssertionError("mismatched Radar Fields metric domain was accepted")


if __name__ == "__main__":
    test_canonical_domain_is_accepted()
    test_producer_constant_agrees_with_validator()
    test_mismatched_domain_is_rejected()
    print("RF_METRIC_DOMAIN_REGRESSION_PASS")
