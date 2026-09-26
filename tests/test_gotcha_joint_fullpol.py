import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np


_MODULE_PATH = Path(__file__).resolve().parents[1] / "rift" / "gotcha_joint_fullpol.py"
_SPEC = importlib.util.spec_from_file_location("gotcha_joint_fullpol_contract", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _manifest():
    split = _MODULE.build_sector_split()
    shards = []
    for pass_id in _MODULE.PASS_IDS:
        for polarization in _MODULE.POLARIZATIONS:
            shard_id = _MODULE.canonical_shard_id(pass_id, polarization)
            shards.append(
                {
                    "shard_id": shard_id,
                    "pass_id": pass_id,
                    "polarization": polarization,
                    "sector_ids": list(_MODULE.SECTOR_IDS),
                    "sector_roles": list(split.role_by_sector),
                    "payload_audited_sector_ids": list(
                        _MODULE.PAYLOAD_AUDITED_SECTOR_IDS
                    ),
                    "payload_audited_sector_count": (
                        _MODULE.PAYLOAD_AUDITED_SECTOR_COUNT
                    ),
                    "sealed_test_sector_ids": list(
                        _MODULE.SEALED_TEST_SECTOR_IDS
                    ),
                    "sealed_test_sector_count": _MODULE.SEALED_TEST_SECTOR_COUNT,
                    "source_files": [
                        f"/raw/pass{pass_id}/{polarization}/az{sector:03d}.mat"
                        for sector in _MODULE.SECTOR_IDS
                    ],
                }
            )
    return {
        "schema": _MODULE.MANIFEST_SCHEMA,
        "scene_id": _MODULE.SCENE_ID,
        "manager_track_id": _MODULE.MANAGER_TRACK_ID,
        "scene_count": _MODULE.SCENE_COUNT,
        "passes": list(_MODULE.PASS_IDS),
        "polarizations": list(_MODULE.POLARIZATIONS),
        "source_file_count": _MODULE.SOURCE_FILE_COUNT,
        "inventoried_source_file_count": _MODULE.INVENTORIED_SOURCE_FILE_COUNT,
        "payload_audited_source_file_count": (
            _MODULE.PAYLOAD_AUDITED_SOURCE_FILE_COUNT
        ),
        "sealed_test_source_file_count": _MODULE.SEALED_TEST_SOURCE_FILE_COUNT,
        "corrections_applied": False,
        "test_opened": False,
        "split": split.as_dict(),
        "support": _MODULE.support_contract(),
        "shards": shards,
    }


def _calibration(bound=0.25):
    split = _MODULE.build_sector_split()
    train_sectors = list(split.train)
    validation_sectors = list(split.validation)
    strata = []
    for pass_id in _MODULE.PASS_IDS:
        for polarization in _MODULE.POLARIZATIONS:
            shard_id = _MODULE.canonical_shard_id(pass_id, polarization)
            if polarization in _MODULE.CO_POLARIZATIONS:
                autofocus = {
                    "validated": True,
                    "official_available": True,
                    "applied": True,
                    "source_shard_id": shard_id,
                    "source_polarization": polarization,
                    "range_field": "af.r_correct",
                    "phase_field": "af.ph_correct",
                    "source_file_count": _MODULE.SECTOR_COUNT,
                    "application_contract": {
                        "range_sign": 1,
                        "phase_sign": -1,
                        "application_order": list(
                            _MODULE.AUTOFOCUS_APPLICATION_ORDER
                        ),
                        "range_formula": _MODULE.AUTOFOCUS_RANGE_FORMULA,
                        "phase_formula": _MODULE.AUTOFOCUS_PHASE_FORMULA,
                        "range_correction_units": "m",
                        "phase_correction_units": "rad",
                    },
                    "heldout_validation": {
                        "role": "validation",
                        "sector_ids": validation_sectors,
                        "metric": _MODULE.AUTOFOCUS_HELDOUT_METRIC,
                        "score_before": 1.1,
                        "score_after": 0.9,
                        "improved": True,
                        "complex_sample_count": 4096,
                    },
                }
            else:
                autofocus = {
                    "validated": True,
                    "official_available": False,
                    "applied": False,
                    "source_shard_id": None,
                    "source_polarization": None,
                    "range_field": None,
                    "phase_field": None,
                    "source_file_count": 0,
                    "application_contract": None,
                    "heldout_validation": None,
                }
            strata.append(
                {
                    "shard_id": shard_id,
                    "pass_id": pass_id,
                    "polarization": polarization,
                    "fit_role": "train",
                    "fit_sector_ids": train_sectors,
                    "frozen": True,
                    "gain_real": 1.0 if pass_id == 2 else 1.0 + 0.01 * pass_id,
                    "gain_imag": (
                        0.0
                        if pass_id == 2
                        else 0.01 * _MODULE.POLARIZATIONS.index(polarization)
                    ),
                    "range_offset_m": 0.0 if pass_id == 2 else 0.01 * (pass_id - 4),
                    "autofocus": autofocus,
                }
            )
    return {
        "schema": _MODULE.CALIBRATION_SCHEMA,
        "scene_id": _MODULE.SCENE_ID,
        "split_seed": _MODULE.SPLIT_SEED,
        "fit_role": "train",
        "frozen": True,
        "test_opened": False,
        "range_offset_bound_m": bound,
        "strata": strata,
    }


def _targets():
    targets = {}
    for index, shard_id in enumerate(_MODULE.SHARD_IDS):
        base = np.arange(1, 9, dtype=np.float64).reshape(2, 4)
        targets[shard_id] = (
            (1.0 + 0.05 * index) * base
            + 1j * (0.25 + 0.01 * index) * base[:, ::-1]
        ).astype(np.complex128)
    return targets


class GotchaJointFullPolTests(unittest.TestCase):
    def test_constants_and_split_are_exact_and_shared(self):
        self.assertEqual(_MODULE.PASS_IDS, tuple(range(1, 9)))
        self.assertEqual(_MODULE.POLARIZATIONS, ("hh", "hv", "vh", "vv"))
        self.assertEqual(_MODULE.SHARD_COUNT, 32)
        self.assertEqual(_MODULE.SOURCE_FILE_COUNT, 11520)
        split = _MODULE.build_sector_split()
        replay = _MODULE.build_sector_split()
        self.assertEqual(split, replay)
        self.assertEqual(
            (len(split.train), len(split.validation), len(split.test)),
            (288, 36, 36),
        )
        self.assertEqual(split.role_for(1), "validation")
        self.assertEqual(split.role_for(6), "test")
        self.assertEqual(split.role_for(2), "train")
        self.assertEqual(len(_MODULE.PAYLOAD_AUDITED_SECTOR_IDS), 324)
        self.assertEqual(len(_MODULE.SEALED_TEST_SECTOR_IDS), 36)
        self.assertEqual(
            set(_MODULE.PAYLOAD_AUDITED_SECTOR_IDS),
            set(split.train) | set(split.validation),
        )
        self.assertEqual(set(_MODULE.SEALED_TEST_SECTOR_IDS), set(split.test))
        self.assertFalse(
            set(_MODULE.PAYLOAD_AUDITED_SECTOR_IDS)
            & set(_MODULE.SEALED_TEST_SECTOR_IDS)
        )
        self.assertEqual(_MODULE.PAYLOAD_AUDITED_SOURCE_FILE_COUNT, 10_368)
        self.assertEqual(_MODULE.SEALED_TEST_SOURCE_FILE_COUNT, 1_152)
        with self.assertRaisesRegex(ValueError, "seals split seed 42"):
            _MODULE.build_sector_split(seed=43)

    def test_support_uses_cell_centres_and_not_one_over_32_m(self):
        contract = _MODULE.support_contract()
        _MODULE.validate_support_contract(contract)
        self.assertEqual(contract["shape"], [1776, 1776, 1])
        self.assertEqual(contract["point_count"], 3_154_176)
        self.assertEqual(_MODULE.SUPPORT_PITCH_M, 100.0 / 1776.0)
        self.assertFalse(
            math.isclose(
                _MODULE.SUPPORT_PITCH_M,
                _MODULE.INCORRECT_ONE_OVER_32_PITCH_M,
            )
        )
        self.assertAlmostEqual(
            contract["first_center_xy_m"][0],
            -50.0 + 0.5 * 100.0 / 1776.0,
            places=14,
        )
        tampered = dict(contract)
        tampered["pitch_xy_m"] = [1.0 / 32.0, 1.0 / 32.0]
        with self.assertRaisesRegex(ValueError, "pitch_xy_m"):
            _MODULE.validate_support_contract(tampered)

    def test_exact_32_shard_manifest_passes(self):
        summary = _MODULE.validate_joint_manifest(_manifest())
        self.assertEqual(summary["shard_count"], 32)
        self.assertEqual(summary["source_file_count"], 11520)
        self.assertEqual(summary["manager_track_id"], _MODULE.MANAGER_TRACK_ID)
        self.assertEqual(summary["scene_count"], 1)
        self.assertEqual(summary["inventoried_source_file_count"], 11_520)
        self.assertEqual(summary["payload_audited_source_file_count"], 10_368)
        self.assertEqual(summary["sealed_test_source_file_count"], 1_152)
        self.assertEqual(
            summary["sector_counts"],
            {"train": 288, "validation": 36, "test": 36},
        )

    def test_manifest_rejects_channel_split_drift_and_missing_shard(self):
        manifest = _manifest()
        manifest["shards"][7]["sector_roles"][0] = "train"
        with self.assertRaisesRegex(ValueError, "canonical sector roles"):
            _MODULE.validate_joint_manifest(manifest)

        manifest = _manifest()
        manifest["shards"].pop()
        with self.assertRaisesRegex(ValueError, "exactly 32 shards"):
            _MODULE.validate_joint_manifest(manifest)

    def test_manifest_rejects_reused_source_file(self):
        manifest = _manifest()
        manifest["shards"][1]["source_files"][0] = manifest["shards"][0][
            "source_files"
        ][0]
        with self.assertRaisesRegex(ValueError, "reused across shards"):
            _MODULE.validate_joint_manifest(manifest)

    def test_manifest_rejects_provenance_and_top_level_count_drift(self):
        cases = (
            ("manager_track_id", "wrong-track", "manager_track_id"),
            ("scene_count", 32, "scene_count"),
            ("source_file_count", 11_519, "source_file_count"),
            (
                "inventoried_source_file_count",
                11_519,
                "inventoried_source_file_count",
            ),
            (
                "payload_audited_source_file_count",
                10_367,
                "payload_audited_source_file_count",
            ),
            (
                "sealed_test_source_file_count",
                1_151,
                "sealed_test_source_file_count",
            ),
            ("corrections_applied", True, "corrections_applied"),
        )
        for key, value, message in cases:
            with self.subTest(key=key):
                manifest = _manifest()
                manifest[key] = value
                with self.assertRaisesRegex(ValueError, message):
                    _MODULE.validate_joint_manifest(manifest)

    def test_manifest_rejects_payload_audit_and_test_seal_drift(self):
        cases = (
            (
                "payload_audited_sector_ids",
                list(_MODULE.PAYLOAD_AUDITED_SECTOR_IDS[:-1]),
                "payload_audited_sector_ids",
            ),
            (
                "payload_audited_sector_count",
                _MODULE.PAYLOAD_AUDITED_SECTOR_COUNT - 1,
                "payload_audited_sector_count",
            ),
            (
                "sealed_test_sector_ids",
                list(_MODULE.SEALED_TEST_SECTOR_IDS[:-1]),
                "sealed_test_sector_ids",
            ),
            (
                "sealed_test_sector_count",
                _MODULE.SEALED_TEST_SECTOR_COUNT - 1,
                "sealed_test_sector_count",
            ),
        )
        for key, value, message in cases:
            with self.subTest(key=key):
                manifest = _manifest()
                manifest["shards"][0][key] = value
                with self.assertRaisesRegex(ValueError, message):
                    _MODULE.validate_joint_manifest(manifest)

    def test_metrics_have_exact_zero_predictor_reference(self):
        result = _MODULE.exact_zero_predictor_gate(_targets())
        self.assertTrue(result["zero_predictor_exact"])
        self.assertEqual(result["stratum_count"], 32)
        for block in (result["macro"], result["energy_pooled"]):
            self.assertEqual(block["coherent_relative_mse"], 1.0)
            self.assertEqual(block["coherent_relative_l2"], 1.0)
            self.assertEqual(block["range_power_relative_mse"], 1.0)
            self.assertEqual(block["range_power_relative_l2"], 1.0)

    def test_metrics_do_not_sum_polarizations_and_match_known_scaling(self):
        targets = _targets()
        predictions = {key: 0.5 * value for key, value in targets.items()}
        result = _MODULE.evaluate_joint_metrics(predictions, targets)
        for block in (result["macro"], result["energy_pooled"]):
            self.assertAlmostEqual(block["coherent_relative_mse"], 0.25, places=14)
            self.assertAlmostEqual(block["coherent_relative_l2"], 0.5, places=14)
            # Halving the complex signal quarters its range power.
            self.assertAlmostEqual(block["range_power_relative_mse"], 0.75**2, places=14)
            self.assertAlmostEqual(block["range_power_relative_l2"], 0.75, places=14)
        perfect = _MODULE.evaluate_joint_metrics(targets, targets)
        self.assertAlmostEqual(perfect["macro"]["coherent_relative_mse"], 0.0)
        self.assertAlmostEqual(perfect["macro"]["range_power_relative_mse"], 0.0)

    def test_metric_reduction_is_recomputable_and_emits_all_group_aggregates(self):
        sums = {}
        for index, shard_id in enumerate(_MODULE.SHARD_IDS):
            if index == 0:
                coherent_error = coherent_energy = 100.0
                range_error = range_energy = 200.0
            else:
                coherent_error = range_error = 0.0
                coherent_energy = 1.0
                range_energy = 2.0
            sums[shard_id] = _MODULE.StratumMetricSums(
                coherent_squared_error=coherent_error,
                coherent_target_energy=coherent_energy,
                range_power_squared_error=range_error,
                range_power_target_energy=range_energy,
                complex_sample_count=3,
            )

        result = _MODULE.reduce_metric_sums(sums)
        self.assertEqual(
            set(result["per_polarization"]), set(_MODULE.POLARIZATIONS)
        )
        self.assertEqual(
            set(result["per_pass"]),
            {f"pass{pass_id}" for pass_id in _MODULE.PASS_IDS},
        )
        self.assertTrue(
            all(
                group["stratum_count"] == 8
                for group in result["per_polarization"].values()
            )
        )
        self.assertTrue(
            all(
                group["stratum_count"] == 4
                for group in result["per_pass"].values()
            )
        )

        first = result["per_stratum"][_MODULE.SHARD_IDS[0]]
        self.assertEqual(first["coherent_squared_error"], 100.0)
        self.assertEqual(first["range_power_squared_error"], 200.0)
        self.assertEqual(
            first["coherent_relative_mse"],
            first["coherent_squared_error"] / first["coherent_target_energy"],
        )
        self.assertEqual(
            first["range_power_relative_mse"],
            first["range_power_squared_error"]
            / first["range_power_target_energy"],
        )

        self.assertAlmostEqual(result["macro"]["coherent_relative_mse"], 1 / 32)
        self.assertAlmostEqual(
            result["energy_pooled"]["coherent_relative_mse"], 100 / 131
        )
        self.assertAlmostEqual(
            result["per_polarization"]["hh"]["macro"][
                "coherent_relative_mse"
            ],
            1 / 8,
        )
        self.assertAlmostEqual(
            result["per_polarization"]["hh"]["energy_pooled"][
                "coherent_relative_mse"
            ],
            100 / 107,
        )
        self.assertAlmostEqual(
            result["per_pass"]["pass1"]["macro"]["coherent_relative_mse"],
            1 / 4,
        )
        self.assertAlmostEqual(
            result["per_pass"]["pass1"]["energy_pooled"][
                "coherent_relative_mse"
            ],
            100 / 103,
        )
        pooled = result["energy_pooled"]
        self.assertEqual(pooled["coherent_squared_error"], 100.0)
        self.assertEqual(pooled["coherent_target_energy"], 131.0)
        self.assertIn("relative-L2", result["aggregation_semantics"]["macro"])
        self.assertIn(
            "sum error numerators",
            result["aggregation_semantics"]["energy_pooled"],
        )

    def test_calibration_schema_is_channel_specific_train_only_and_frozen(self):
        summary = _MODULE.validate_calibration_artifact(_calibration())
        self.assertEqual(summary["stratum_count"], 32)
        self.assertEqual(summary["autofocus_corrected_strata"], 16)
        self.assertEqual(summary["autofocus_absent_strata"], 16)
        self.assertTrue(summary["frozen"])

    def test_calibration_rejects_borrowed_autofocus_and_crosspol_corrections(self):
        artifact = _calibration()
        hh = next(row for row in artifact["strata"] if row["shard_id"] == "pass1_hh")
        hh["autofocus"]["source_polarization"] = "vv"
        with self.assertRaisesRegex(ValueError, "source_polarization"):
            _MODULE.validate_calibration_artifact(artifact)

        artifact = _calibration()
        hv = next(row for row in artifact["strata"] if row["shard_id"] == "pass1_hv")
        hv["autofocus"]["applied"] = True
        with self.assertRaisesRegex(ValueError, "autofocus.applied"):
            _MODULE.validate_calibration_artifact(artifact)

    def test_calibration_requires_exact_formula_and_heldout_improvement(self):
        artifact = _calibration()
        hh = next(row for row in artifact["strata"] if row["shard_id"] == "pass1_hh")
        hh["autofocus"]["application_contract"]["phase_formula"] = "ambiguous"
        with self.assertRaisesRegex(ValueError, "phase_formula"):
            _MODULE.validate_calibration_artifact(artifact)

        artifact = _calibration()
        hh = next(row for row in artifact["strata"] if row["shard_id"] == "pass1_hh")
        hh["autofocus"]["heldout_validation"]["score_after"] = 1.1
        with self.assertRaisesRegex(ValueError, "strictly improve"):
            _MODULE.validate_calibration_artifact(artifact)

        artifact = _calibration()
        hv = next(row for row in artifact["strata"] if row["shard_id"] == "pass1_hv")
        hv["autofocus"]["application_contract"] = {
            "range_sign": 1,
            "phase_sign": 1,
        }
        with self.assertRaisesRegex(ValueError, "application_contract must be null"):
            _MODULE.validate_calibration_artifact(artifact)

    def test_calibration_rejects_leakage_and_unbounded_offsets(self):
        artifact = _calibration(bound=0.1)
        artifact["strata"][0]["fit_sector_ids"][0] = 1  # validation sector
        with self.assertRaisesRegex(ValueError, "exact training role"):
            _MODULE.validate_calibration_artifact(artifact)

        artifact = _calibration(bound=0.1)
        artifact["strata"][0]["range_offset_m"] = 0.1001
        with self.assertRaisesRegex(ValueError, "exceeds the frozen bound"):
            _MODULE.validate_calibration_artifact(artifact)

        artifact = _calibration(bound=1.01)
        with self.assertRaisesRegex(ValueError, "must be in"):
            _MODULE.validate_calibration_artifact(artifact)

    def test_calibration_enforces_pass2_gauge_for_every_polarization(self):
        perturbations = (
            ("gain_real", 1.001),
            ("gain_imag", 0.001),
            ("range_offset_m", 0.001),
        )
        for polarization in _MODULE.POLARIZATIONS:
            for field, value in perturbations:
                with self.subTest(polarization=polarization, field=field):
                    artifact = _calibration()
                    row = next(
                        entry
                        for entry in artifact["strata"]
                        if entry["pass_id"] == 2
                        and entry["polarization"] == polarization
                    )
                    row[field] = value
                    with self.assertRaisesRegex(ValueError, "pass-2 nuisance gauge"):
                        _MODULE.validate_calibration_artifact(artifact)

    @unittest.skipIf(_MODULE.torch is None, "PyTorch is not installed")
    def test_four_head_scene_shares_positions_but_keeps_head_gradients_disjoint(self):
        torch = _MODULE.torch
        scene = _MODULE.PolarimetricFixedPlanarSHScene(
            device="cpu", nx=2, ny=3, extent_m=50.0, init_scale=0.01
        )
        self.assertEqual(tuple(scene.heads), _MODULE.POLARIZATIONS)
        for head in scene.heads.values():
            self.assertEqual(tuple(head.w_re.shape), (6, 16))
            self.assertEqual(tuple(head.w_im.shape), (6, 16))
        positions = scene.position_chunk(0, 6)
        self.assertEqual(tuple(positions.shape), (6, 3))
        self.assertTrue(torch.equal(positions[:, 2], torch.zeros(6)))

        basis = torch.ones(16, 2, dtype=scene.heads["hh"].w_re.dtype)
        hh = scene.view_weight_chunk("hh", basis, 0, 6)
        vv = scene.view_weight_chunk("vv", basis, 0, 6)
        self.assertEqual(tuple(hh.shape), (6, 2))
        self.assertFalse(torch.equal(hh, vv))
        hh.abs().sum().backward()
        self.assertGreater(float(scene.heads["hh"].w_re.grad.abs().sum()), 0.0)
        self.assertGreater(float(scene.heads["hh"].w_im.grad.abs().sum()), 0.0)
        for polarization in ("hv", "vh", "vv"):
            self.assertIsNone(scene.heads[polarization].w_re.grad)
            self.assertIsNone(scene.heads[polarization].w_im.grad)


if __name__ == "__main__":
    unittest.main()
