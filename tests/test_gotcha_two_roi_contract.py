"""Focused data-free tests for the structural-only two-target ROI contract."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "rift" / "gotcha_two_roi_contract.py"
PROTOCOL_PATH = ROOT / "protocols" / "gotcha_step3_two_target_roi_v1.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("gotcha_two_roi_contract_tests", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load ROI contract module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CONTRACT = _load_module()


def _raw() -> dict:
    with PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class TwoRoiContractTest(unittest.TestCase):
    def test_exact_local_grids_and_structural_only_report(self):
        contract = CONTRACT.load_contract(PROTOCOL_PATH)
        tophat = contract.target("tophat").grid
        camry = contract.target("toyota_camry").grid
        self.assertEqual(tophat.shape, (40, 40, 40))
        self.assertEqual(camry.shape, (100, 100, 100))
        self.assertEqual(tophat.point_count, 40**3)
        self.assertEqual(camry.point_count, 100**3)
        self.assertEqual(tophat.final_sample_m, (1.9, 1.9, 1.9))
        self.assertEqual(camry.final_sample_m, (4.9, 4.9, 4.9))
        self.assertAlmostEqual(tophat.sample(39)[0], 1.9, places=12)
        self.assertAlmostEqual(camry.sample(99)[0], 4.9, places=12)
        report = contract.structural_report()
        self.assertTrue(report["data_free"])
        self.assertFalse(report["measured_fit_release"])
        self.assertEqual(report["extraction_allowed"], {"tophat": False, "toyota_camry": False})
        self.assertEqual(report["fit_allowed"], {"tophat": False, "toyota_camry": False})
        self.assertFalse(report["test_sealed"])
        self.assertEqual(report["test_policy"], "sealed_no_selection")
        self.assertEqual(report["test_state"], "declared_policy_only_no_future_release_claim")

    def test_grid_endpoint_half_voxel_and_shape_mutations_reject(self):
        for mutation in (
            lambda value: value["targets"]["tophat"]["grid"].__setitem__("shape", [41, 40, 40]),
            lambda value: value["targets"]["tophat"]["grid"].__setitem__("endpoint_inclusion", True),
            lambda value: value["targets"]["toyota_camry"]["grid"].__setitem__("half_voxel_shift", True),
            lambda value: value["targets"]["toyota_camry"]["grid"].__setitem__("final_sample_m", [5.0, 4.9, 4.9]),
            lambda value: value["targets"]["tophat"]["grid"].__setitem__("frame", "native_scene_frame"),
        ):
            value = _raw()
            mutation(value)
            with self.assertRaises(ValueError):
                CONTRACT.validate_declaration(value)

    def test_target_set_split_and_test_policy_are_exact(self):
        missing = _raw()
        del missing["targets"]["tophat"]
        with self.assertRaises(ValueError):
            CONTRACT.validate_declaration(missing)
        extra = _raw()
        extra["targets"]["extra"] = copy.deepcopy(extra["targets"]["tophat"])
        with self.assertRaises(ValueError):
            CONTRACT.validate_declaration(extra)
        bad_split = _raw()
        bad_split["role_split"]["validation"] = 35
        with self.assertRaises(ValueError):
            CONTRACT.validate_declaration(bad_split)
        bad_policy = _raw()
        bad_policy["test_policy"] = "open"
        with self.assertRaises(ValueError):
            CONTRACT.validate_declaration(bad_policy)
        with self.assertRaises(ValueError):
            CONTRACT._duplicate_rejecting_pairs([("targets", {}), ("targets", {})])

    def test_all_readiness_gates_unconditionally_reject_and_no_callback_api_exists(self):
        contract = CONTRACT.load_contract(PROTOCOL_PATH)
        self.assertFalse(hasattr(CONTRACT, "load_target_after_ready"))
        for target_id in CONTRACT.TARGET_IDS:
            for gate in (CONTRACT.require_target_extraction_ready, CONTRACT.require_target_fit_ready):
                with self.assertRaisesRegex(RuntimeError, "data-free"):
                    gate(target_id, contract)
                with self.assertRaisesRegex(RuntimeError, "separately reviewed"):
                    gate(target_id, contract)

    def test_every_attempted_readiness_state_rejects(self):
        mutations = []

        def add(label, edit):
            value = _raw()
            edit(value["targets"]["tophat"])
            mutations.append((label, value))

        add("numeric_R", lambda target: target["p_native"].__setitem__("R", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
        add("numeric_t", lambda target: target["p_native"].__setitem__("t_m", [1.0, 2.0, 3.0]))
        add("numeric_string_R", lambda target: target["p_native"].__setitem__("R", "1.0"))
        add("sourced_placement_status", lambda target: target["p_native"].__setitem__("status", "sourced_verified"))
        add("fake_source", lambda target: target["p_native"].__setitem__("source", "fabricated_source"))
        add("fake_evidence", lambda target: target["p_native"].__setitem__("source_evidence", {"verified": True}))
        add("canonical_ids", lambda target: target["selected_native_observation_ids"].update({"status": "sourced_verified", "values": [{"pass_id": 1, "polarization": "hh", "sector_id": 1, "pulse_index": 0, "role": "train"}]}))
        add("role_mismatch", lambda target: target["selected_native_observation_ids"].__setitem__("values", [{"role": "validation"}]))
        add("extraction_claim", lambda target: target.__setitem__("extraction_allowed", True))
        add("fit_claim", lambda target: target.__setitem__("fit_allowed", True))
        add("frozen_claim", lambda target: target["frozen_phase_r0_af_calibration"].__setitem__("status", "frozen"))
        add("spotlight_claim", lambda target: target["complex_spotlight"].__setitem__("status", "validated"))
        add("padding_claim", lambda target: target["padding_background"].__setitem__("status", "validated"))
        add("psf_claim", lambda target: target["psf_leakage"].__setitem__("status", "validated"))

        for label, value in mutations:
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    CONTRACT.validate_declaration(value)

    def test_spotlight_af_padding_and_psf_defaults_are_fail_closed(self):
        mutations = (
            lambda value: value["targets"]["tophat"]["frozen_phase_r0_af_calibration"].__setitem__("af_cross_channel_borrowing", True),
            lambda value: value["targets"]["tophat"]["frozen_phase_r0_af_calibration"].__setitem__("double_correction", True),
            lambda value: value["targets"]["tophat"]["complex_spotlight"].__setitem__("data_prediction_application", "magnitude_only"),
            lambda value: value["targets"]["tophat"]["padding_background"].__setitem__("hidden_padding", True),
            lambda value: value["targets"]["tophat"]["psf_leakage"].__setitem__("evidence", {"kind": "fake"}),
        )
        for mutation in mutations:
            value = _raw()
            mutation(value)
            with self.assertRaises(ValueError):
                CONTRACT.validate_declaration(value)


if __name__ == "__main__":
    unittest.main()
