"""Focused synthetic tests for the GOTCHA Step-2 physical controls."""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ACQ = _load_module("gotcha_acquisition_for_step2_tests", ROOT / "rift" / "gotcha_acquisition.py")
CTL = _load_module("gotcha_step2_controls_tests", ROOT / "rift" / "gotcha_step2_controls.py")


def _ro(value, dtype=None):
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _observation(
    pass_id: int,
    polarization: str,
    sector_id: int,
    pulse_index: int,
    position: np.ndarray,
    r0: float,
    frequencies: np.ndarray,
    response: np.ndarray,
    role: str,
):
    autofocus = ACQ.AutofocusProvenance(
        schema=ACQ.AUTOFOCUS_PROVENANCE_SCHEMA,
        mode=ACQ.AUTOFOCUS_OFFICIAL_ABSENT,
        official_available=False,
        applied=False,
        source_shard_id=f"pass{pass_id}_{polarization}",
        range_field=None,
        phase_field=None,
    )
    return ACQ.NativeObservation(
        identity=ACQ.NativeObservationId(pass_id, polarization, sector_id, pulse_index),
        role=role,
        response=_ro(response, np.complex128),
        frequencies_hz=_ro(frequencies, np.float64),
        position_xyz_m=_ro(position, np.float64),
        r0_m=float(r0),
        th_deg=0.0,
        phi_deg=0.0,
        r_correct_raw=None,
        ph_correct_raw=None,
        phase_reference=ACQ.PhaseReferenceContract(),
        autofocus=autofocus,
    )


def _synthetic_observations():
    points = np.asarray([[0.37, -1.21, 0.70], [-2.40, 1.90, 4.07]], dtype=np.float64)
    amplitudes = {
        "hv": np.asarray([1.0 + 0.3j, 0.2 - 0.1j], dtype=np.complex128),
        "vh": np.asarray([0.7 - 0.2j, -0.3 + 0.4j], dtype=np.complex128),
    }
    rows = (
        (1, "hv", 11, 101, np.asarray([100.0, -20.0, 15.0]), 102.7, np.asarray([9.10e9, 9.17e9, 9.31e9]), "train"),
        (1, "vh", 12, 102, np.asarray([80.0, 25.0, 16.0]), 84.9, np.asarray([9.10e9, 9.19e9, 9.33e9, 9.40e9]), "train"),
        (2, "hv", 13, 203, np.asarray([-90.0, 15.0, 19.0]), 92.8, np.asarray([9.10e9, 9.16e9, 9.29e9, 9.43e9, 9.51e9]), "train"),
        (2, "vh", 14, 204, np.asarray([70.0, -30.0, 18.0]), 77.1, np.asarray([9.10e9, 9.22e9, 9.35e9]), "train"),
        (3, "hv", 15, 305, np.asarray([-65.0, -35.0, 21.0]), 74.6, np.asarray([9.10e9, 9.18e9, 9.37e9, 9.49e9]), "validation"),
        (3, "vh", 16, 306, np.asarray([55.0, 35.0, 23.0]), 65.8, np.asarray([9.10e9, 9.24e9, 9.39e9, 9.55e9, 9.63e9]), "validation"),
    )
    observations = []
    for pass_id, polarization, sector, pulse, position, r0, frequencies, role in rows:
        response = ACQ.direct_point_target_render(
            points,
            amplitudes[polarization],
            tx_positions_m=position[None, :],
            rx_positions_m=position[None, :],
            frequencies_hz=frequencies,
            reference_range_m=np.asarray([r0]),
            convention=ACQ.PUBLISHED_GOTCHA_REFERENCE_CANDIDATE,
        )[0]
        observations.append(
            _observation(pass_id, polarization, sector, pulse, position, r0, frequencies, response, role)
        )
    return points, amplitudes, tuple(observations)


def _selection_shard(pass_id: int, polarization: str, *, include_test: bool = False):
    train_sectors = np.asarray([2, 3, 4, 7, 8, 9], dtype=np.int64)
    sectors = np.concatenate((train_sectors, np.asarray([6], dtype=np.int64))) if include_test else train_sectors
    count = sectors.size
    frequencies = _ro([9.1e9, 9.2e9, 9.3e9], np.float64)
    response = _ro(np.zeros((count, frequencies.size), dtype=np.complex128))
    roles = np.asarray(["train"] * train_sectors.size + (["test"] if include_test else []), dtype="U10")
    autofocus = ACQ.AutofocusProvenance(
        schema=ACQ.AUTOFOCUS_PROVENANCE_SCHEMA,
        mode=ACQ.AUTOFOCUS_OFFICIAL_ABSENT,
        official_available=False,
        applied=False,
        source_shard_id=f"pass{pass_id}_{polarization}",
        range_field=None,
        phase_field=None,
    )
    return ACQ.NativeShard(
        path=Path(f"synthetic_{pass_id}_{polarization}.npz"),
        shard_id=f"pass{pass_id}_{polarization}",
        pass_id=pass_id,
        polarization=polarization,
        response=response,
        frequencies_hz=frequencies,
        x=_ro(np.full(count, 100.0), np.float64),
        y=_ro(np.arange(count, dtype=np.float64)),
        z=_ro(np.full(count, 15.0), np.float64),
        r0=_ro(np.full(count, 100.0), np.float64),
        th=_ro(np.zeros(count), np.float64),
        phi=_ro(np.zeros(count), np.float64),
        sector_id=_ro(sectors, np.int64),
        pulse_index=_ro(np.arange(100, 100 + count), np.int64),
        role=_ro(roles),
        r_correct_raw=_ro(np.empty(0), np.float64),
        ph_correct_raw=_ro(np.empty(0), np.float64),
        metadata={},
        phase_reference=ACQ.PhaseReferenceContract(),
        autofocus=autofocus,
    )


class GotchaStep2ControlsTest(unittest.TestCase):
    def test_nested_selection_is_deterministic_and_rejects_test_before_payload(self):
        shards = tuple(
            _selection_shard(pass_id, polarization)
            for pass_id in (1, 2)
            for polarization in ("hv", "vh")
        )
        first = CTL.select_train_prefixes(shards, counts=(1, 2, 4), seed=17)
        second = CTL.select_train_prefixes(shards, counts=(1, 2, 4), seed=17)
        self.assertEqual(CTL.validate_nested_prefixes(first)["nested"], True)
        self.assertEqual(
            [first[count].as_dict() for count in (1, 2, 4)],
            [second[count].as_dict() for count in (1, 2, 4)],
        )
        self.assertEqual(first[4].counts_by_pass, {1: 8, 2: 8})
        with self.assertRaisesRegex(ValueError, "sealed test"):
            CTL.select_train_prefixes(
                shards + (_selection_shard(1, "hh", include_test=True),),
                counts=(1, 2),
                requested_passes=(1,),
                requested_polarizations=("hh",),
            )

    def test_native_ragged_operator_matches_direct_oracle_and_dot_product(self):
        points, amplitudes, observations = _synthetic_observations()
        observations = observations[0::2]
        operator = CTL.NativeRaggedOperator(observations, point_chunk_size=1)
        coefficients = np.asarray([0.6 - 0.2j, -0.3 + 0.8j], dtype=np.complex128)
        predicted = operator.forward(points, coefficients)
        expected = []
        for observation in observations:
            expected.append(
                ACQ.direct_point_target_render(
                    points,
                    coefficients,
                    tx_positions_m=observation.position_xyz_m[None, :],
                    rx_positions_m=observation.position_xyz_m[None, :],
                    frequencies_hz=observation.frequencies_hz,
                    reference_range_m=np.asarray([observation.r0_m]),
                    convention=ACQ.PUBLISHED_GOTCHA_REFERENCE_CANDIDATE,
                )[0]
            )
        for actual, oracle in zip(predicted.values, expected):
            np.testing.assert_allclose(actual, oracle, rtol=2e-13, atol=2e-13)
        residuals = tuple(
            np.asarray([0.4 + 0.2j] * values.size, dtype=np.complex128)
            for values in predicted.values
        )
        dot_report = operator.dot_test(points, coefficients, residuals)
        self.assertLess(dot_report["relative_error"], 1e-12)
        self.assertEqual(operator.frequency_counts, (3, 5, 4))
        bp = operator.backproject(points, operator.data(), normalization="none")
        self.assertEqual(bp.metadata["normalization"], "none")
        self.assertFalse(bp.metadata["amplitude_weighting"])
        self.assertEqual(bp.metadata["operator"], "direct_native_ragged_adjoint")
        self.assertEqual(bp.metadata["polarization"], "hv")
        self.assertEqual(operator.polarization, "hv")
        with self.assertRaisesRegex(ValueError, "exactly one polarization"):
            CTL.NativeRaggedOperator(_synthetic_observations()[2])
        vh_operator = CTL.NativeRaggedOperator(_synthetic_observations()[2][1::2])
        self.assertNotEqual(vh_operator.polarization, operator.polarization)
        self.assertNotEqual(observations[0].response.tolist(), _synthetic_observations()[2][1].response.tolist())

    def test_height_and_native_r0_are_not_silently_removed(self):
        points, amplitudes, observations = _synthetic_observations()
        observations = observations[0::2]
        operator = CTL.NativeRaggedOperator(observations)
        true_values = operator.forward(points, amplitudes["hv"])
        flattened = points.copy()
        flattened[:, 2] = 0.0
        wrong_height = operator.forward(flattened, amplitudes["hv"])
        error = np.linalg.norm(np.concatenate([a - b for a, b in zip(true_values.values, wrong_height.values)]))
        self.assertGreater(error, 1e-3)

        scalar_reference_observations = tuple(
            dataclasses.replace(observation, r0_m=float(np.linalg.norm(observation.position_xyz_m)))
            for observation in observations
        )
        scalar_operator = CTL.NativeRaggedOperator(scalar_reference_observations)
        scalar_values = scalar_operator.forward(points, amplitudes["hv"])
        reference_error = np.linalg.norm(
            np.concatenate([a - b for a, b in zip(true_values.values, scalar_values.values)])
        )
        self.assertGreater(reference_error, 1e-3)

    def test_ridge_cgls_matches_dense_ridge_oracle_and_reports_controls(self):
        points, amplitudes, all_observations = _synthetic_observations()
        observations = all_observations[0:4:2]
        operator = CTL.NativeRaggedOperator(observations)
        basis = np.eye(points.shape[0], dtype=np.complex128)
        matrix = np.column_stack(
            [np.concatenate(operator.forward(points, column).values) for column in basis.T]
        )
        truth = np.asarray([0.8 - 0.2j, -0.4 + 0.5j], dtype=np.complex128)
        target = operator.forward(points, truth)
        lam = 2e-2
        result = CTL.ridge_cgls(operator, points, target, lambda_ridge=lam, max_iterations=30, rtol=1e-11)
        oracle = np.linalg.solve(matrix.conj().T @ matrix + lam * np.eye(points.shape[0]), matrix.conj().T @ np.concatenate(target.values))
        np.testing.assert_allclose(result.coefficients, oracle, rtol=1e-8, atol=1e-8)
        self.assertIn(result.status, {"converged", "max_iterations"})
        self.assertTrue(result.iterations)
        self.assertIn("data_residual_norm", result.iterations[-1])
        self.assertIn("normal_residual_norm", result.iterations[-1])
        self.assertIn("objective", result.iterations[-1])
        self.assertTrue(all(np.isfinite(row["objective"]) for row in result.iterations))
        self.assertEqual(result.kernel_evaluations_estimate, operator.estimate_cgls_kernel_evaluations(2, 30))
        low_target = operator.forward(points, truth * 1.0e-12)
        low_result = CTL.ridge_cgls(operator, points, low_target, lambda_ridge=lam, max_iterations=5, rtol=1e-6)
        self.assertGreater(low_result.iterations[-1]["iteration"], 0)
        validation_operator = CTL.NativeRaggedOperator(all_observations[0::2])
        with self.assertRaisesRegex(ValueError, "train observations only"):
            CTL.ridge_cgls(
                validation_operator,
                points,
                validation_operator.data(),
                lambda_ridge=lam,
                max_iterations=2,
            )
        with self.assertRaises(RuntimeError):
            CTL.ridge_cgls(
                CTL.NativeRaggedOperator(observations, max_kernel_evaluations=10),
                points,
                target,
                lambda_ridge=lam,
                max_iterations=30,
            )

    def test_train_only_frozen_scale_metrics_and_zero_energy(self):
        points, _, observations = _synthetic_observations()
        observations = observations[0::2]
        actual = CTL.NativePredictions(
            tuple(observation.identity for observation in observations),
            tuple(np.asarray(observation.response, dtype=np.complex128) for observation in observations),
        )
        half = CTL.NativePredictions(actual.ids, tuple(0.5 * value for value in actual.values))
        train = observations[:2]
        train_prediction = CTL.NativePredictions(half.ids[:2], half.values[:2])
        scale = CTL.fit_train_complex_scale(train, train_prediction)
        self.assertAlmostEqual(scale.value.real, 2.0, places=12)
        validation = observations[2:]
        validation_prediction = CTL.NativePredictions(half.ids[2:], half.values[2:])
        metrics = CTL.evaluate_native_metrics(validation, validation_prediction, scale=scale)
        self.assertLess(metrics["scaled"]["relmse"], 1e-24)
        self.assertEqual(metrics["scale"]["fit_role"], "train")
        self.assertEqual(metrics["scale"]["polarization"], "hv")
        vh_observations = _synthetic_observations()[2][1::2]
        vh_predictions = CTL.NativePredictions(
            tuple(observation.identity for observation in vh_observations),
            tuple(0.5 * observation.response for observation in vh_observations),
        )
        with self.assertRaisesRegex(ValueError, "polarization"):
            CTL.evaluate_native_metrics(vh_observations, vh_predictions, scale=scale)
        zero_observation = dataclasses.replace(observations[0], response=_ro(np.zeros_like(observations[0].response)))
        zero_predictions = CTL.NativePredictions((zero_observation.identity,), (np.zeros_like(zero_observation.response),))
        zero_metrics = CTL.evaluate_native_metrics((zero_observation,), zero_predictions)["raw"]
        self.assertIsNone(zero_metrics["relmse"])
        self.assertIsNone(zero_metrics["prediction_energy_over_target"])
        self.assertIsNone(zero_metrics["complex_correlation_abs"])
        with self.assertRaises(ValueError):
            CTL.fit_train_complex_scale(
                _synthetic_observations()[2][:2],
                CTL.NativePredictions(half.ids[:2], half.values[:2]),
            )
        with self.assertRaisesRegex(ValueError, "zero-energy"):
            CTL.fit_train_complex_scale(
                observations[:2],
                CTL.NativePredictions(
                    actual.ids[:2],
                    tuple(np.zeros_like(value) for value in actual.values[:2]),
                ),
            )

    def test_explicit_support_preflight_is_conditional_and_height_capable(self):
        _, _, all_observations = _synthetic_observations()
        observations = all_observations[0::2]
        declaration = {
            "schema": CTL.H0_SUPPORT_SCHEMA,
            "bounds_m": {"x": [-3.0, 3.0], "y": [-3.0, 3.0], "z": [0.0, 4.0]},
            "sampling": {"spacing_m": [1.0, 1.0, 1.0], "shape": [7, 7, 5]},
            "max_kernel_evaluations": 10000,
            "support_mode": "height_capable_volume",
            "frame_contract": "antenna_xyz_unchanged",
            "registration_status": "unresolved",
            "support_status": CTL.SUPPORT_STATUS,
            "hypothesis_label": "conditional_z_volume_not_physical_registration",
            "units": {"x": "m", "y": "m", "z": "m", "r0": "m", "frequency": "Hz"},
            "phase_hypothesis": CTL.PHASE_HYPOTHESIS_NAME,
            "autofocus_status": "raw_unapplied_channel_owned",
        }
        report = CTL.preflight_support_declaration(declaration, observations, point_count=100)
        self.assertFalse(report["full_scene_support_claim"])
        self.assertEqual(report["support_boundary_diagnostic_status"], "conditional_box_diagnostic_not_support_proof")
        self.assertEqual(report["observations"]["native_frequency_spacing_hz"]["uniform_grid_assumed"], False)
        self.assertIn("signed_R_minus_r0_span_m", report)
        self.assertEqual(report["support"]["sampling"]["shape"], [7, 7, 5])
        conditional_bp = CTL.conditional_backproject(declaration, observations)
        self.assertEqual(conditional_bp.values.shape, (7 * 7 * 5,))
        self.assertEqual(conditional_bp.metadata["support_point_count_bound"], 7 * 7 * 5)
        with self.assertRaisesRegex(TypeError, "NativePredictions"):
            CTL.conditional_backproject(
                declaration,
                observations,
                residuals=tuple(CTL.NativeRaggedOperator(observations).data().values),
            )
        conditional_cgls = CTL.conditional_ridge_cgls(
            declaration,
            observations[:2],
            CTL.NativeRaggedOperator(observations[:2]).data(),
            lambda_ridge=1e-2,
            max_iterations=1,
            rtol=1e-6,
        )
        self.assertIsNotNone(conditional_cgls.support_preflight)
        self.assertEqual(conditional_cgls.coefficients.size, 7 * 7 * 5)
        with self.assertRaisesRegex(ValueError, "train observations only"):
            CTL.conditional_ridge_cgls(
                declaration,
                observations,
                CTL.NativeRaggedOperator(observations).data(),
                lambda_ridge=1e-2,
                max_iterations=1,
            )
        with self.assertRaisesRegex(TypeError, "NativePredictions"):
            CTL.conditional_ridge_cgls(
                declaration,
                observations[:2],
                tuple(CTL.NativeRaggedOperator(observations[:2]).data().values),
                lambda_ridge=1e-2,
                max_iterations=1,
            )
        plane = dict(declaration)
        plane["support_mode"] = "plane_no_height"
        plane["bounds_m"] = {"x": [-3.0, 3.0], "y": [-3.0, 3.0], "z": [0.0, 0.0]}
        plane["sampling"] = {"spacing_m": [1.0, 1.0, 1.0], "shape": [7, 7, 1]}
        plane_report = CTL.preflight_support_declaration(plane, observations, point_count=100)
        self.assertFalse(plane_report["height_capable"])
        self.assertTrue(plane_report["plane_only_no_height_claim"])
        direct_plane = CTL.NativeFrameH0Support(
            bounds_m={"x": (-3.0, 3.0), "y": (-3.0, 3.0), "z": (0.0, 0.0)},
            spacing_m=(1.0, 1.0, 1.0),
            grid_shape=(7, 7, 1),
            max_kernel_evaluations=10000,
            support_mode="plane_no_height",
        )
        self.assertEqual(direct_plane.grid_points().shape, (49, 3))
        with self.assertRaisesRegex(ValueError, "nonzero z extent"):
            CTL.NativeFrameH0Support(
                bounds_m={"x": (-3.0, 3.0), "y": (-3.0, 3.0), "z": (0.0, 0.0)},
                spacing_m=(1.0, 1.0, 1.0),
                grid_shape=(7, 7, 1),
                max_kernel_evaluations=10000,
            )
        planar = dict(declaration)
        planar["sampling"] = {"spacing_m": [1.0, 1.0, 4.0], "shape": [7, 7, 1]}
        planar["bounds_m"] = {"x": [-3.0, 3.0], "y": [-3.0, 3.0], "z": [0.0, 4.0]}
        with self.assertRaises(ValueError):
            CTL.preflight_support_declaration(planar, observations, point_count=100)
        with self.assertRaises(ValueError):
            CTL.preflight_support_declaration(None, observations)
        guarded = dict(declaration)
        guarded["max_kernel_evaluations"] = 10
        with self.assertRaises(RuntimeError):
            CTL.preflight_support_declaration(guarded, observations, point_count=100)


if __name__ == "__main__":
    unittest.main()
