"""Validate the data-free, structural-only two-target ROI contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "rift" / "gotcha_two_roi_contract.py"
PROTOCOL_PATH = ROOT / "protocols" / "gotcha_step3_two_target_roi_v1.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("gotcha_two_roi_contract_validator", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load two-ROI contract module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CONTRACT = _load_module()


def _load_raw() -> dict:
    with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _expect_rejection(label: str, value: dict) -> None:
    try:
        CONTRACT.validate_declaration(value)
    except ValueError:
        return
    raise AssertionError(f"mutation unexpectedly passed: {label}")


def main() -> int:
    contract = CONTRACT.load_contract(PROTOCOL_PATH)
    report = contract.structural_report()
    if (
        not report["data_free"]
        or report["measured_fit_release"] is not False
        or any(report["extraction_allowed"].values())
        or any(report["fit_allowed"].values())
        or report["test_policy"] != "sealed_no_selection"
        or report["test_sealed"] is not False
        or report["test_state"] != "declared_policy_only_no_future_release_claim"
    ):
        raise AssertionError("structural report is not structural-only and fail-closed")
    if hasattr(CONTRACT, "load_target_after_ready"):
        raise AssertionError("generic callback loader API must not exist in data-free v1")
    if contract.target("tophat").grid.point_count != 40 ** 3 or contract.target("toyota_camry").grid.point_count != 100 ** 3:
        raise AssertionError("unexpected point count")
    if contract.target("tophat").grid.final_sample_m != (1.9, 1.9, 1.9):
        raise AssertionError("Tophat endpoint mismatch")
    if contract.target("toyota_camry").grid.final_sample_m != (4.9, 4.9, 4.9):
        raise AssertionError("Camry endpoint mismatch")

    gate_calls = []
    for target_id in CONTRACT.TARGET_IDS:
        for gate in (CONTRACT.require_target_extraction_ready, CONTRACT.require_target_fit_ready):
            try:
                gate(target_id, contract)
            except RuntimeError as error:
                if "data-free" not in str(error) or "separately reviewed" not in str(error):
                    raise AssertionError(f"gate error is not the required data-free explanation: {error}") from error
            else:
                raise AssertionError(f"data-free target unexpectedly passed gate: {target_id}")
        gate_calls.append(target_id)

    mutations = []
    bad_grid = _load_raw()
    bad_grid["targets"]["tophat"]["grid"]["shape"] = [41, 40, 40]
    mutations.append(("shape", bad_grid))
    bad_split = _load_raw()
    bad_split["role_split"]["test"] = 35
    mutations.append(("split", bad_split))
    bad_af = _load_raw()
    bad_af["targets"]["tophat"]["frozen_phase_r0_af_calibration"]["double_correction"] = True
    mutations.append(("double_correction", bad_af))
    bad_spotlight = _load_raw()
    bad_spotlight["targets"]["tophat"]["complex_spotlight"]["data_prediction_application"] = "magnitude_only"
    mutations.append(("spotlight", bad_spotlight))

    ready_attempts = [
        ("numeric_R", {"R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}),
        ("numeric_t", {"t_m": [1.0, 2.0, 3.0]}),
        ("fake_source", {"source": "fabricated_source"}),
        ("fake_evidence", {"source_evidence": {"verified": True}}),
        ("sourced_placement_status", {"status": "sourced_verified"}),
        ("fake_ids", {"selected_native_observation_ids": {"status": "sourced_verified", "values": [{"pass_id": 1, "polarization": "hh", "sector_id": 1, "pulse_index": 0, "role": "train"}]}}),
        ("role_mismatch", {"selected_native_observation_ids": {"status": "unresolved", "values": [{"role": "validation"}]}}),
        ("extraction_claim", {"extraction_allowed": True}),
        ("fit_claim", {"fit_allowed": True}),
        ("frozen_claim", {"frozen_phase_r0_af_calibration": {"status": "frozen"}}),
        ("spotlight_claim", {"complex_spotlight": {"status": "validated"}}),
        ("padding_claim", {"padding_background": {"status": "validated"}}),
        ("psf_claim", {"psf_leakage": {"status": "validated"}}),
    ]
    for label, patch in ready_attempts:
        value = _load_raw()
        for key, replacement in patch.items():
            if key == "R" or key == "t_m" or key == "source" or key == "status" or key == "fake":
                value["targets"]["tophat"]["p_native"][key] = replacement
            elif key in {"source_evidence"}:
                value["targets"]["tophat"]["p_native"][key] = replacement
            else:
                value["targets"]["tophat"][key] = replacement
        mutations.append((label, value))

    numeric_string = _load_raw()
    numeric_string["targets"]["tophat"]["p_native"]["R"] = "1.0"
    mutations.append(("numeric_string_R", numeric_string))
    for label, value in mutations:
        _expect_rejection(label, value)

    duplicate_rejected = False
    try:
        CONTRACT._duplicate_rejecting_pairs([("target", 1), ("target", 2)])
    except ValueError:
        duplicate_rejected = True
    if not duplicate_rejected:
        raise AssertionError("duplicate JSON key was not rejected")

    print(json.dumps({
        "schema": "rift_gotcha_step3_two_target_roi_validation_v1",
        "status": "PASS",
        "data_free": True,
        "targets": list(CONTRACT.TARGET_IDS),
        "tophat_point_count": contract.target("tophat").grid.point_count,
        "toyota_camry_point_count": contract.target("toyota_camry").grid.point_count,
        "test_policy": report["test_policy"],
        "test_state": report["test_state"],
        "unconditional_gate_block": True,
        "callback_loader_api": False,
        "mutation_rejections": len(mutations),
        "duplicate_key_rejection": duplicate_rejected,
        "gate_targets_checked": gate_calls,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
