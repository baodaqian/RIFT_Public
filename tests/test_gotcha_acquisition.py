"""Dependency-light Batch-A tests for the sealed GOTCHA acquisition adapter."""

from __future__ import annotations

import json
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import contextmanager

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "gotcha_acquisition_batch_a_test", _ROOT / "rift" / "gotcha_acquisition.py"
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("cannot load acquisition module for focused tests")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

AUTOFOCUS_OFFICIAL_ABSENT = _MODULE.AUTOFOCUS_OFFICIAL_ABSENT
AutofocusApplicationContract = _MODULE.AutofocusApplicationContract
HeightSupportCandidate = _MODULE.HeightSupportCandidate
NativeObservationId = _MODULE.NativeObservationId
PairedMonostaticGeometry = _MODULE.PairedMonostaticGeometry
SyntheticReferenceConvention = _MODULE.SyntheticReferenceConvention
apply_published_autofocus = _MODULE.apply_published_autofocus
compare_native_and_endpoint_uniform_frequency = _MODULE.compare_native_and_endpoint_uniform_frequency
direct_point_target_adjoint = _MODULE.direct_point_target_adjoint
direct_point_target_render = _MODULE.direct_point_target_render
forward_adjoint_inner_product_fixture = _MODULE.forward_adjoint_inner_product_fixture
load_native_shard = _MODULE.load_native_shard
preflight_height_support_candidate = _MODULE.preflight_height_support_candidate
PUBLISHED_GOTCHA_REFERENCE_CANDIDATE = _MODULE.PUBLISHED_GOTCHA_REFERENCE_CANDIDATE
forward_vjp_finite_difference_fixture = _MODULE.forward_vjp_finite_difference_fixture
virtual_reference_mapping_regression_fixture = _MODULE.virtual_reference_mapping_regression_fixture


class _Split:
    train = tuple(sector for sector in range(1, 361) if (sector - 1) % 10 not in (0, 5))
    validation = tuple(sector for sector in range(1, 361) if (sector - 1) % 10 == 0)
    test = tuple(sector for sector in range(1, 361) if (sector - 1) % 10 == 5)

    @staticmethod
    def role_for(sector: int) -> str:
        slot = (int(sector) - 1) % 10
        return "validation" if slot == 0 else "test" if slot == 5 else "train"


def build_sector_split() -> _Split:
    return _Split()


def _fixture_arrays(*, polarization: str = "hh", metadata_updates: dict | None = None) -> dict[str, np.ndarray]:
    split = build_sector_split()
    sectors = sorted(set(split.train) | set(split.validation))
    view_count = len(sectors)
    frequencies = np.asarray(
        [9_600_000_000.0, 9_601_000_000.0, 9_602_500_000.0, 9_604_000_000.0, 9_605_000_000.0],
        dtype=np.float32,
    )
    row = np.arange(view_count, dtype=np.float32)
    response = (row[:, None] + 1.0).astype(np.complex64) * np.exp(
        1j * np.asarray([0.0, 0.2, -0.4, 0.7, -0.1], dtype=np.float32)[None, :]
    )
    role = np.asarray([split.role_for(sector) for sector in sectors], dtype="U10")
    if polarization in ("hh", "vv"):
        r_correct = np.linspace(0.01, 0.02, view_count, dtype=np.float32)
        ph_correct = np.linspace(-0.2, 0.2, view_count, dtype=np.float32)
        autofocus_available = np.asarray(True)
        autofocus_state = np.asarray("raw_channel_own_arrays_unapplied", dtype="U36")
    else:
        r_correct = np.empty(0, dtype=np.float64)
        ph_correct = np.empty(0, dtype=np.float64)
        autofocus_available = np.asarray(False)
        autofocus_state = np.asarray(AUTOFOCUS_OFFICIAL_ABSENT, dtype="U36")
    metadata = {
        "schema": "rift_gotcha_joint8_fullpol_native_shard_v1",
        "scene_id": "gotcha_v1_joint8_fullpol",
        "shard_id": f"pass2_{polarization}",
        "pass_id": 2,
        "polarization": polarization,
        "payload_sector_ids": sectors,
        "sealed_test_sector_ids": sorted(split.test),
        "test_opened": False,
        "test_payload_included": False,
        "corrections_applied": False,
        "autofocus_unapplied": True,
        "layout": {
            "native_frequency_preserved": True,
            "resampled": False,
            "padded": False,
            "trimmed": False,
            "autofocus_unapplied": True,
        },
    }
    if metadata_updates:
        metadata.update(metadata_updates)
    return {
        "response": response,
        "frequencies_hz": frequencies,
        "x": (100.0 + row).astype(np.float32),
        "y": (-20.0 + 0.5 * row).astype(np.float32),
        "z": (1.0 + 0.01 * row).astype(np.float32),
        "r0": (120.0 + 0.1 * row).astype(np.float32),
        "th": np.linspace(-2.0, 2.0, view_count, dtype=np.float32),
        "phi": np.linspace(-1.0, 1.0, view_count, dtype=np.float32),
        "sector_id": np.asarray(sectors, dtype=np.int16),
        "pulse_index": np.asarray(sectors, dtype=np.int32) * 100 + 7,
        "pass_id": np.full(view_count, 2, dtype=np.int16),
        "polarization": np.full(view_count, polarization, dtype="U2"),
        "role": role,
        "r_correct_raw": r_correct,
        "ph_correct_raw": ph_correct,
        "autofocus_available": autofocus_available,
        "autofocus_applied": np.asarray(False),
        "autofocus_state": autofocus_state,
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True), dtype="U"),
    }


@contextmanager
def _temporary_npz_path():
    fd, name = tempfile.mkstemp(prefix=".gotcha-acquisition-test-", suffix=".npz")
    os.close(fd)
    path = Path(name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def _write_fixture(path: Path, *, polarization: str = "hh", metadata_updates: dict | None = None, array_updates: dict | None = None) -> Path:
    arrays = _fixture_arrays(polarization=polarization, metadata_updates=metadata_updates)
    if array_updates:
        arrays.update(array_updates)
    with path.open("wb") as handle:
        np.savez(handle, **arrays)
    return path


class GotchaAcquisitionTests(unittest.TestCase):
    def test_native_archive_preserves_exact_frequency_and_identity(self):
        with _temporary_npz_path() as path:
            shard = load_native_shard(_write_fixture(path))
            self.assertEqual(shard.frequencies_hz.dtype, np.dtype("float32"))
            np.testing.assert_array_equal(
                shard.frequencies_hz,
                np.asarray(
                    [9_600_000_000.0, 9_601_000_000.0, 9_602_500_000.0, 9_604_000_000.0, 9_605_000_000.0],
                    dtype=np.float32,
                ),
            )
            self.assertEqual(shard.view_count, 324)
            self.assertEqual(len(shard.identities_for_role("train")), 288)
            self.assertEqual(len(shard.identities_for_role("validation")), 36)
            first = shard.observation_ids[0]
            self.assertEqual(first, NativeObservationId(2, "hh", first.sector_id, first.pulse_index))
            selected = shard.observations((shard.observation_ids[-1], first))
            self.assertEqual(selected[0].identity, shard.observation_ids[-1])
            self.assertNotEqual(selected[0].identity, selected[1].identity)
            self.assertFalse(shard.response.flags.writeable)
            with self.assertRaises(ValueError):
                shard.response[0, 0] = 0

    def test_test_payload_flags_and_rows_are_refused(self):
        with _temporary_npz_path() as test_flag:
            _write_fixture(test_flag, metadata_updates={"test_opened": True})
            with self.assertRaisesRegex(ValueError, "metadata.test_opened"):
                load_native_shard(test_flag)

            arrays = _fixture_arrays()
            arrays["role"] = arrays["role"].copy()
            arrays["role"][0] = "test"
            # The response member is intentionally unreadable with
            # allow_pickle=False; role closure must reject before it is loaded.
            arrays["response"] = np.asarray(["sealed-response"], dtype=object)
            with _temporary_npz_path() as bad_role:
                with bad_role.open("wb") as handle:
                    np.savez(handle, **arrays)
                with self.assertRaisesRegex(ValueError, "test rows|role labels"):
                    load_native_shard(bad_role)

    def test_autofocus_provenance_is_channel_owned_and_application_is_gated(self):
        with _temporary_npz_path() as co_path:
            co_pol = load_native_shard(_write_fixture(co_path, polarization="hh"))
            observation = co_pol.observations((co_pol.observation_ids[0],))[0]
            with self.assertRaisesRegex(RuntimeError, "raw-only"):
                apply_published_autofocus(observation, AutofocusApplicationContract())
            contract = AutofocusApplicationContract(
                validated=True,
                range_sign=1,
                phase_sign=-1,
                range_unit="m",
                phase_unit="rad",
                units_validated=True,
            )
            contract.validate()
            with self.assertRaisesRegex(RuntimeError, "raw-only"):
                apply_published_autofocus(observation, contract)
            np.testing.assert_array_equal(observation.response, co_pol.response[0])

            with _temporary_npz_path() as cross_path:
                cross_pol = load_native_shard(_write_fixture(cross_path, polarization="hv"))
            self.assertEqual(cross_pol.autofocus.mode, AUTOFOCUS_OFFICIAL_ABSENT)
            cross_observation = cross_pol.observations((cross_pol.observation_ids[0],))[0]
            self.assertIsNone(cross_observation.r_correct_raw)
            with self.assertRaisesRegex(RuntimeError, "raw-only"):
                apply_published_autofocus(cross_observation, contract)

            borrowed = _fixture_arrays(polarization="hv")
            borrowed["r_correct_raw"] = np.zeros(324, dtype=np.float32)
            borrowed["ph_correct_raw"] = np.zeros(324, dtype=np.float32)
            with _temporary_npz_path() as borrowed_path:
                with borrowed_path.open("wb") as handle:
                    np.savez(handle, **borrowed)
                with self.assertRaisesRegex(ValueError, "cannot borrow"):
                    load_native_shard(borrowed_path)

    def test_paired_monostatic_geometry_and_cross_pair_rejection(self):
        with _temporary_npz_path() as path:
            shard = load_native_shard(_write_fixture(path))
            geometry = shard.paired_monostatic_geometry((shard.observation_ids[0],))
            self.assertEqual(geometry.tx_xyz_m.shape, (1, 3))
            np.testing.assert_array_equal(geometry.tx_xyz_m, geometry.rx_xyz_m)
            with self.assertRaisesRegex(ValueError, "Tx=Rx"):
                type(geometry)(geometry.observation_ids, geometry.tx_xyz_m, geometry.tx_xyz_m + 1.0)

    def test_direct_forward_adjoint_and_native_frequency_approximation(self):
        tx = np.asarray([[100.0, 0.0, 12.0], [0.0, 100.0, 12.0], [-80.0, 20.0, 12.0]])
        points = np.asarray([[0.5, -1.2, 2.3], [-4.0, 2.0, 5.0]])
        frequencies = np.asarray([9.6e9, 9.601e9, 9.6025e9, 9.604e9, 9.605e9], dtype=np.float32)
        amplitudes = np.asarray([1.0 + 0.5j, -0.2 + 0.9j])
        residual = np.asarray(
            [[0.3 + 0.8j, -0.7 + 0.2j, 0.4 - 0.1j, 0.6 + 0.3j, -0.1 + 0.5j]] * 3,
            dtype=np.complex128,
        )
        report = forward_adjoint_inner_product_fixture(
            points,
            amplitudes,
            residual,
            tx_positions_m=tx,
            frequencies_hz=frequencies,
            reference_range_m=np.asarray([200.0, 200.0, 200.0]),
        )
        self.assertTrue(report["passed"])
        self.assertLess(report["relative_error"], 5.0e-13)
        rendered = direct_point_target_render(
            points,
            amplitudes,
            tx_positions_m=tx,
            frequencies_hz=frequencies,
        )
        adjoint = direct_point_target_adjoint(
            residual,
            points,
            tx_positions_m=tx,
            frequencies_hz=frequencies,
        )
        np.testing.assert_allclose(
            np.vdot(rendered, residual), np.vdot(amplitudes, adjoint), rtol=1e-12, atol=1e-12
        )
        frequency_report = compare_native_and_endpoint_uniform_frequency(
            frequencies,
            tx_positions_m=tx,
            point_positions_m=points,
            amplitudes=amplitudes,
        )
        self.assertTrue(frequency_report["native_retained"])
        self.assertGreater(frequency_report["max_abs_frequency_delta_hz"], 0.0)
        vjp_report = forward_vjp_finite_difference_fixture(
            points,
            amplitudes,
            residual,
            tx_positions_m=tx,
            frequencies_hz=frequencies,
        )
        self.assertTrue(vjp_report["passed"])
        self.assertLess(vjp_report["relative_error"], 5.0e-7)

        candidate = PUBLISHED_GOTCHA_REFERENCE_CANDIDATE
        self.assertEqual(candidate.adjoint_phase_sign, 1)
        candidate_render = direct_point_target_render(
            points[:1],
            np.asarray([1.0 + 0.0j]),
            tx_positions_m=tx[:1],
            frequencies_hz=frequencies,
            reference_range_m=100.0,
            convention=candidate,
        )
        one_way = np.linalg.norm(tx[0] - points[0])
        expected = np.exp(
            -1j
            * (4.0 * np.pi / candidate.speed_of_light_m_s)
            * frequencies.astype(np.float64)
            * (one_way - 100.0)
        )
        np.testing.assert_allclose(candidate_render[0], expected, rtol=1e-12, atol=1e-12)

        mapping = virtual_reference_mapping_regression_fixture(
            observation_positions_m=tx,
            center_xyz_m=np.asarray([3.5, -1.25, 2.0]),
            point_positions_m=np.asarray([[0.37, -1.21, 2.31], [-2.4, 1.9, 4.07]]),
            amplitudes=np.asarray([0.2 + 0.1j, -0.3 + 0.4j]),
            frequencies_hz=np.asarray([9.6e9, 9.601e9, 9.6025e9, 9.604e9], dtype=np.float32),
            r0_offset_m=np.asarray([0.003, -0.003, 0.005]),
            reference_range_m=17.0,
        )
        self.assertTrue(mapping["passed"])
        self.assertTrue(mapping["constant_only_mapping_failed"])
        self.assertLess(mapping["full_q_operator_error"], 2.0e-12)
        self.assertLess(mapping["full_q_signal_error"], 2.0e-12)
        self.assertLess(mapping["full_q_inverse_operator_error"], 2.0e-12)
        self.assertLess(mapping["full_q_adjoint_relative_error"], 2.0e-12)

    def test_height_candidate_is_volume_capable_but_not_real_bounds_evidence(self):
        ids = (NativeObservationId(2, "hh", 1, 7), NativeObservationId(2, "hh", 2, 8))
        positions = np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0]], dtype=np.float64)
        geometry = PairedMonostaticGeometry(ids, positions, positions.copy())
        candidate = HeightSupportCandidate((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
        report = preflight_height_support_candidate(
            candidate,
            geometry,
            reference_range_m=140.0,
            unambiguous_range_m=100.0,
            convention=SyntheticReferenceConvention(),
        )
        self.assertEqual(report["status"], "synthetic_preflight_only_real_bounds_unresolved")
        self.assertFalse(report["real_gotcha_bounds_resolved"])
        self.assertTrue(report["float64_path_calculation"])


if __name__ == "__main__":
    unittest.main()
