"""Data-free tests for the immutable raw/source-AF comparison algebra."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
import sys
from types import SimpleNamespace
import unittest
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "rift" / "gotcha_source_af.py"
DRIVER_PATH = ROOT / "scripts" / "run_gotcha_step2_source_af_compare_v1.py"
DRIVER_PROTOCOL_PATH = ROOT / "protocols" / "gotcha_step2_source_af_hh_sector002_h0_compare_v1.json"
NOMINAL_PSF_DRIVER_PATH = ROOT / "scripts" / "run_gotcha_step2_15tr07_nominal_psf_v1.py"
NOMINAL_PSF_PROTOCOL_PATH = ROOT / "protocols" / "gotcha_step2_15tr07_nominal_psf_p1_hh_train_v1.json"
CORRESPONDENCE_DRIVER_PATH = ROOT / "scripts" / "run_gotcha_step2_15tr07_correspondence_verify_v1.py"
CORRESPONDENCE_PROTOCOL_PATH = ROOT / "protocols" / "gotcha_step2_15tr07_correspondence_verify_p1_hh_v1.json"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SOURCE_AF = _load_module(MODULE_PATH, "gotcha_source_af_tests")
DRIVER = _load_module(DRIVER_PATH, "gotcha_source_af_driver_tests")
NOMINAL_PSF_DRIVER = _load_module(NOMINAL_PSF_DRIVER_PATH, "gotcha_15tr07_nominal_psf_driver_tests")
CORRESPONDENCE_DRIVER = _load_module(CORRESPONDENCE_DRIVER_PATH, "gotcha_15tr07_correspondence_driver_tests")
DRIVER_PROTOCOL = DRIVER._read_protocol(DRIVER_PROTOCOL_PATH)


@contextmanager
def _writable_temp_dir(*, prefix: str = "rift_gotcha_test_"):
    with tempfile.TemporaryDirectory(prefix=prefix) as temp_dir:
        temp_path = Path(temp_dir)
        probe = temp_path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        yield temp_path


@dataclass(frozen=True)
class _ObservationIdentity:
    pass_id: int
    polarization: str
    sector_id: int
    pulse_index: int

    def as_dict(self):
        return {
            "pass_id": int(self.pass_id),
            "polarization": str(self.polarization),
            "sector_id": int(self.sector_id),
            "pulse_index": int(self.pulse_index),
        }


def _observation_identity(index: int, sector_id: int, polarization: str = "hh") -> _ObservationIdentity:
    return _ObservationIdentity(pass_id=1, polarization=polarization, sector_id=sector_id, pulse_index=index)


def _observation(
    pulse_index: int = 0,
    *,
    response: np.ndarray | None = None,
    frequencies: np.ndarray | None = None,
    r0: float = 120.0,
    r_correct: float = 0.01,
    ph_correct: float = 0.2,
    applied: bool = False,
    sector_id: int = 2,
    role: str = "train",
    polarization: str = "hh",
    source_shard_id: str | None = None,
    range_field: str | None = None,
    phase_field: str | None = None,
    frequency_values: str = "native_stored_exact",
):
    if frequencies is None:
        frequencies = np.asarray([9.6e9, 9.601e9, 9.603e9, 9.606e9], dtype=np.float32)
    if response is None:
        response = np.asarray([1.0 + 0.5j, 2.0 - 0.25j, -0.5 + 1.0j, 0.75 + 0.1j], dtype=np.complex64)
    identity = _observation_identity(pulse_index, sector_id, polarization)
    co_polarized = polarization in {"hh", "vv"}
    if source_shard_id is None:
        source_shard_id = f"pass1_{polarization}" if co_polarized else None
    if range_field is None:
        range_field = "r_correct_raw" if co_polarized else None
    if phase_field is None:
        phase_field = "ph_correct_raw" if co_polarized else None
    return SimpleNamespace(
        identity=identity,
        role=role,
        response=response,
        frequencies_hz=frequencies,
        position_xyz_m=np.asarray([100.0, -20.0, 1.0], dtype=np.float32),
        r0_m=r0,
        r_correct_raw=r_correct,
        ph_correct_raw=ph_correct,
        phase_reference=SimpleNamespace(
            reference_range_field="r0",
            geometry_contract="paired_monostatic_tx_equals_rx_same_observation",
            frequency_values=frequency_values,
        ),
        autofocus=SimpleNamespace(
            official_available=co_polarized,
            applied=applied,
            mode="raw_channel_own_arrays_unapplied" if co_polarized else "official_arrays_absent",
            source_shard_id=source_shard_id,
            range_field=range_field,
            phase_field=phase_field,
        ),
    )


def _write_uncompressed_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a real NPZ with metadata_json as the first ZIP member."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, value in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(value), allow_pickle=False)
            archive.writestr(f"{name}.npy", buffer.getvalue(), compress_type=zipfile.ZIP_STORED)


def _build_sealed_synthetic_archive(
    archive_root: Path,
    *,
    frequency_count: int = 424,
    sector_pulse_counts: dict[int, int] | None = None,
    known_point_xyz_m: tuple[float, float, float] | None = None,
    known_point_sectors: tuple[int, ...] = (268, 269, 270),
    validation_point_xyz_m: tuple[float, float, float] | None = None,
    frequency_span_hz: tuple[float, float] = (9.6e9, 9.7e9),
) -> Path:
    """Build the exact Gate-1 payload shape while leaving test sectors sealed."""

    archive_path = archive_root / "shards" / "pass1_hh.npz"
    payload_sectors = [sector for sector in range(1, 361) if (sector - 1) % 10 != 5]
    sealed_test_sectors = [sector for sector in range(1, 361) if (sector - 1) % 10 == 5]
    pulse_counts = {2: 117} if sector_pulse_counts is None else {
        int(sector): int(count) for sector, count in sector_pulse_counts.items()
    }
    rows = [
        (sector, pulse)
        for sector in payload_sectors
        for pulse in range(pulse_counts.get(sector, 1))
    ]
    row_count = len(rows)
    frequencies = np.linspace(float(frequency_span_hz[0]), float(frequency_span_hz[1]), frequency_count, dtype=np.float64)
    sectors = np.asarray([sector for sector, _pulse in rows], dtype=np.int32)
    pulses = np.asarray([pulse for _sector, pulse in rows], dtype=np.int32)
    roles = np.asarray(
        ["validation" if (sector - 1) % 10 == 0 else "train" for sector in sectors],
        dtype="U10",
    )
    angles = (sectors.astype(np.float64) - 1.0) * (2.0 * np.pi / 360.0)
    angles += np.where(sectors == 2, pulses.astype(np.float64) * 0.002, 0.0)
    if known_point_xyz_m is not None:
        point_sectors = {int(value) for value in known_point_sectors}
        point_sector_mask = np.isin(sectors, tuple(point_sectors))
        angles += np.where(point_sector_mask, pulses.astype(np.float64) * 0.002, 0.0)
    radius = 100.0 + 0.05 * np.sin(angles * 3.0)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = 1.0 + 0.01 * np.cos(angles)
    r0 = np.sqrt(x * x + y * y + z * z).astype(np.float64)
    th = np.rad2deg(angles).astype(np.float64)
    phi = np.rad2deg(np.arctan2(z, radius)).astype(np.float64)
    r_correct = np.where(
        sectors == 2,
        0.0005 + 0.00005 * np.sin(pulses.astype(np.float64) * 0.11),
        0.00025,
    ).astype(np.float64)
    ph_correct = np.where(
        sectors == 2,
        0.05 + 0.005 * np.cos(pulses.astype(np.float64) * 0.07),
        0.025,
    ).astype(np.float64)
    frequency_phase = np.arange(frequency_count, dtype=np.float64)[None, :] * 0.0017
    row_phase = sectors.astype(np.float64)[:, None] * 0.013 + pulses.astype(np.float64)[:, None] * 0.021
    amplitude = np.where(sectors == 2, 1.0, 0.25).astype(np.float64)[:, None]
    response = (amplitude * np.exp(1j * (row_phase + frequency_phase))).astype(np.complex64)
    if known_point_xyz_m is not None:
        known_point = np.asarray(known_point_xyz_m, dtype=np.float64)
        if known_point.shape != (3,) or not np.isfinite(known_point).all():
            raise ValueError("known_point_xyz_m must be a finite xyz triplet")
        point_sectors = {int(value) for value in known_point_sectors}
        for row_index, (sector, role) in enumerate(zip(sectors, roles)):
            target = None
            if int(sector) in point_sectors:
                target = known_point
            if str(role) == "validation" and int(sector) == 271 and validation_point_xyz_m is not None:
                target = np.asarray(validation_point_xyz_m, dtype=np.float64)
            if target is None:
                continue
            # Use the public acquisition oracle with its explicitly declared
            # fixed synthetic convention; the fixture does not depend on a
            # private source-AF production helper.
            position = np.asarray([x[row_index], y[row_index], z[row_index]], dtype=np.float64)
            source_response = DRIVER.ACQ.direct_point_target_render(
                target[None, :],
                np.asarray([1.0 + 0.0j], dtype=np.complex128),
                tx_positions_m=position[None, :],
                rx_positions_m=position[None, :],
                frequencies_hz=frequencies,
                reference_range_m=[float(np.float64(r0[row_index]) + np.float64(r_correct[row_index]))],
                convention=DRIVER.ACQ.PUBLISHED_GOTCHA_REFERENCE_CANDIDATE,
            )[0]
            response[row_index] = (
                source_response * np.exp(-1j * np.float64(ph_correct[row_index]))
            ).astype(np.complex64)
    role_counts = {
        "train": int(np.count_nonzero(roles == "train")),
        "validation": int(np.count_nonzero(roles == "validation")),
        "test": 0,
    }
    metadata = {
        "schema": DRIVER.ACQ.NATIVE_ARCHIVE_SCHEMA,
        "pass_id": 1,
        "polarization": "hh",
        "shard_id": "pass1_hh",
        "scene_id": "gotcha_v1_joint8_fullpol",
        "test_opened": False,
        "test_payload_included": False,
        "corrections_applied": False,
        "autofocus_unapplied": True,
        "payload_sector_ids": payload_sectors,
        "sealed_test_sector_ids": sealed_test_sectors,
        "role_counts": role_counts,
        "layout": {
            "native_frequency_preserved": True,
            "resampled": False,
            "padded": False,
            "trimmed": False,
            "autofocus_unapplied": True,
        },
    }
    arrays = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "response": response,
        "frequencies_hz": frequencies,
        "x": x,
        "y": y,
        "z": z,
        "r0": r0,
        "th": th,
        "phi": phi,
        "sector_id": sectors,
        "pulse_index": pulses,
        "pass_id": np.ones(row_count, dtype=np.int16),
        "polarization": np.full(row_count, "hh", dtype="U2"),
        "role": roles,
        "r_correct_raw": r_correct,
        "ph_correct_raw": ph_correct,
        "autofocus_available": np.asarray(True),
        "autofocus_applied": np.asarray(False),
        "autofocus_state": np.asarray(DRIVER.ACQ.AUTOFOCUS_RAW),
    }
    _write_uncompressed_npz(archive_path, arrays)
    return archive_path


def _run_sealed_synthetic_compare_via_cli(
    output_dir: Path, *, stage: str = "compare"
) -> tuple[dict[str, object], str, Path, object]:
    archive_root = output_dir.parent / "archive"
    archive_path = _build_sealed_synthetic_archive(archive_root)
    shard = DRIVER.ACQ.load_native_shard(
        archive_path,
        expected_pass_id=1,
        expected_polarization="hh",
        expected_scene_id="gotcha_v1_joint8_fullpol",
    )
    stream = io.StringIO()
    with redirect_stdout(stream):
        status_code = DRIVER.main(
            [
                "--protocol",
                str(DRIVER_PROTOCOL_PATH),
                "--archive-root",
                str(archive_root),
                "--output-dir",
                str(output_dir),
                "--stage",
                stage,
            ]
        )
    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    if status_code != 0:
        raise RuntimeError(f"CLI returned non-zero status: {status_code}")
    if not lines:
        raise RuntimeError("CLI produced no parseable output")
    return json.loads(lines[-1]), stream.getvalue(), archive_path, shard


def _run_nominal_psf_synthetic_via_cli(
    output_dir: Path,
    *,
    archive_root: Path | None = None,
    sector_pulse_counts: dict[int, int] | None = None,
    known_point_xyz_m: tuple[float, float, float] | None = None,
    validation_point_xyz_m: tuple[float, float, float] | None = None,
    frequency_span_hz: tuple[float, float] = (9.6e9, 9.7e9),
):
    archive_root = output_dir.parent / "archive_nominal_psf" if archive_root is None else archive_root
    archive_path = _build_sealed_synthetic_archive(
        archive_root,
        sector_pulse_counts={268: 2, 269: 2, 270: 2} if sector_pulse_counts is None else sector_pulse_counts,
        known_point_xyz_m=known_point_xyz_m,
        validation_point_xyz_m=validation_point_xyz_m,
        frequency_span_hz=frequency_span_hz,
    )
    shard = NOMINAL_PSF_DRIVER.ACQ.load_native_shard(
        archive_path,
        expected_pass_id=1,
        expected_polarization="hh",
        expected_scene_id="gotcha_v1_joint8_fullpol",
    )
    stream = io.StringIO()
    with redirect_stdout(stream):
        status_code = NOMINAL_PSF_DRIVER.main(
            [
                "--protocol",
                str(NOMINAL_PSF_PROTOCOL_PATH),
                "--archive-root",
                str(archive_root),
                "--output-dir",
                str(output_dir),
                "--stage",
                "diagnostic",
            ]
        )
    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    if status_code != 0:
        raise RuntimeError(f"nominal PSF CLI returned non-zero status: {status_code}")
    if not lines:
        raise RuntimeError("nominal PSF CLI produced no parseable output")
    return json.loads(lines[-1]), stream.getvalue(), archive_path, shard


def _run_correspondence_verify_via_cli(
    output_dir: Path,
    archive_root: Path,
    prior_output_dir: Path,
):
    stream = io.StringIO()
    with redirect_stdout(stream):
        status_code = CORRESPONDENCE_DRIVER.main(
            [
                "--protocol",
                str(CORRESPONDENCE_PROTOCOL_PATH),
                "--archive-root",
                str(archive_root),
                "--prior-output-dir",
                str(prior_output_dir),
                "--output-dir",
                str(output_dir),
                "--stage",
                "verification",
            ]
        )
    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    if status_code != 0:
        raise RuntimeError(f"correspondence verifier returned non-zero status: {status_code}")
    if not lines:
        raise RuntimeError("correspondence verifier produced no parseable output")
    return json.loads(lines[-1]), stream.getvalue()


def _run_known_point_correspondence_fixture(
    workspace: Path,
    *,
    known_point_xyz_m: tuple[float, float, float] = (-5.12, 22.98, -0.05),
    validation_point_xyz_m: tuple[float, float, float] | None = None,
    sector_pulse_counts: dict[int, int] | None = None,
):
    archive_root = workspace / "known_point_archive"
    prior_output_dir = workspace / "prior_nominal"
    effective_validation_point = known_point_xyz_m if validation_point_xyz_m is None else validation_point_xyz_m
    _nominal_payload, _nominal_output, archive_path, _nominal_shard = _run_nominal_psf_synthetic_via_cli(
        prior_output_dir,
        archive_root=archive_root,
        sector_pulse_counts=(
            {268: 12, 269: 12, 270: 12, 271: 12}
            if sector_pulse_counts is None
            else sector_pulse_counts
        ),
        known_point_xyz_m=known_point_xyz_m,
        validation_point_xyz_m=effective_validation_point,
        frequency_span_hz=(9.0e9, 11.0e9),
    )
    verification_output_dir = workspace / "correspondence_verification"
    payload, output = _run_correspondence_verify_via_cli(
        verification_output_dir,
        archive_root,
        prior_output_dir,
    )
    return payload, output, archive_path, archive_root, prior_output_dir, verification_output_dir


class SourceAfTest(unittest.TestCase):
    def test_camry_vv_scope_is_explicit_and_keeps_historic_hh_default(self):
        observation = _observation(polarization="vv", r_correct=0.37, ph_correct=-0.41)
        with self.assertRaisesRegex(ValueError, "HH"):
            SOURCE_AF.build_source_af((observation,), expected_count=1)
        scope = SOURCE_AF.CamryVVTrainSourceAFScope(expected_count=1)
        derived = SOURCE_AF.build_source_af((observation,), scope=scope)[0]
        self.assertEqual(derived.identity.polarization, "vv")
        self.assertEqual(derived.scope.as_dict()["name"], "camry_p1_vv_train_sector002")
        self.assertEqual(derived.provenance["correction_source"]["source_shard_id"], "pass1_vv")
        self.assertEqual(derived.provenance["correction_source"]["range_field"], "r_correct_raw")
        self.assertAlmostEqual(derived.effective_r0_m, 120.37, places=12)
        np.testing.assert_array_equal(
            derived.effective_response,
            observation.response * np.exp(1j * np.float64(-0.41)),
        )
        np.testing.assert_array_equal(observation.response, np.asarray(observation.response))
        with self.assertRaisesRegex(ValueError, "cannot override"):
            SOURCE_AF.build_source_af((observation,), scope=scope, expected_count=2)

    def test_camry_vv_preflight_rejects_provenance_before_response_access(self):
        scope = SOURCE_AF.CamryVVTrainSourceAFScope(expected_count=1)

        class BadResponse:
            identity = _observation_identity(0, 2, "vv")
            role = "train"
            position_xyz_m = np.asarray([100.0, -20.0, 1.0], dtype=np.float32)
            r0_m = 120.0
            r_correct_raw = 0.01
            ph_correct_raw = 0.2
            frequencies_hz = np.asarray([9.6e9, 9.601e9, 9.603e9], dtype=np.float64)
            phase_reference = SimpleNamespace(
                reference_range_field="r0",
                geometry_contract="paired_monostatic_tx_equals_rx_same_observation",
                frequency_values="uniform_grid",
            )
            autofocus = SimpleNamespace(
                official_available=True,
                applied=False,
                mode="raw_channel_own_arrays_unapplied",
                source_shard_id="pass1_hh",
                range_field="r_correct_raw",
                phase_field="ph_correct_raw",
            )

            @property
            def response(self):
                raise AssertionError("VV response must not be accessed before provenance preflight")

        with self.assertRaisesRegex(ValueError, "native_stored_exact"):
            SOURCE_AF.SourceAFObservation.from_observation(BadResponse(), scope=scope)

        for field, value, message in (
            ("source_shard_id", "pass1_hh", "source_shard_id"),
            ("range_field", "r_correct", "range correction field"),
            ("phase_field", "ph_correct", "phase correction field"),
        ):
            bad = _observation(polarization="vv")
            setattr(bad.autofocus, field, value)
            with self.assertRaisesRegex(ValueError, message):
                SOURCE_AF.SourceAFObservation.from_observation(bad, scope=scope)

    def test_camry_vv_scope_rejects_wrong_headers_roles_applied_and_derived(self):
        scope = SOURCE_AF.CamryVVTrainSourceAFScope(expected_count=1)
        cases = (
            (_observation(polarization="hh"), "P1/VV/sector 002"),
            (_observation(polarization="vv", role="validation"), "TRAIN"),
            (_observation(polarization="vv", sector_id=3), "P1/VV/sector 002"),
            (_observation(polarization="vv", applied=True), "originally applied"),
            (_observation(polarization="vv", r_correct=None), "both raw correction"),
        )
        for observation, message in cases:
            with self.assertRaisesRegex(ValueError, message):
                SOURCE_AF.SourceAFObservation.from_observation(observation, scope=scope)
        derived = SOURCE_AF.build_source_af((_observation(polarization="vv"),), scope=scope)[0]
        with self.assertRaisesRegex(TypeError, "already-derived"):
            SOURCE_AF.build_source_af((derived,), scope=scope)

    def test_immutable_explicit_raw_and_effective_fields(self):
        observation = _observation()
        derived = SOURCE_AF.build_source_af((observation,), expected_count=1)[0]
        self.assertEqual(derived.representation_tag, SOURCE_AF.SOURCE_REPRESENTATION)
        self.assertTrue(np.array_equal(derived.response_raw, observation.response))
        self.assertTrue(np.array_equal(derived.raw_payload()[1], observation.response))
        self.assertFalse(np.array_equal(derived.effective_response, derived.response_raw))
        self.assertFalse(derived.response_raw.flags.writeable)
        self.assertFalse(derived.effective_response.flags.writeable)
        self.assertEqual(derived.metadata()["effective_response_state"], "complex128_source_af")
        self.assertAlmostEqual(derived.effective_r0_m, 120.01, places=12)
        with self.assertRaises(ValueError):
            derived.effective_response[0] = 0.0

    def test_header_gate_precedes_response_access_and_rejects_double_application(self):
        class BadHeader:
            identity = SimpleNamespace(pass_id=1, polarization="hh", sector_id=3, pulse_index=0)
            role = "train"
            autofocus = SimpleNamespace(official_available=True, applied=False)
            phase_reference = SimpleNamespace(reference_range_field="r0", geometry_contract="paired_monostatic_tx_equals_rx_same_observation")
            r_correct_raw = 0.01
            ph_correct_raw = 0.2
            frequencies_hz = np.asarray([1.0], dtype=np.float64)

            @property
            def response(self):
                raise AssertionError("response must not be accessed before the header gate")

        with self.assertRaisesRegex(ValueError, "sector 002"):
            SOURCE_AF.SourceAFObservation.from_observation(BadHeader())
        with self.assertRaisesRegex(ValueError, "originally applied"):
            SOURCE_AF.SourceAFObservation.from_observation(_observation(applied=True))

    def test_explicit_multi_sector_scope_is_immutable_and_dynamic(self):
        scope = SOURCE_AF.SourceAFScope(
            sector_ids=(268, 269, 270),
            expected_count=None,
            name="15tr07_test_scope",
        )
        observations = tuple(
            _observation(index, sector_id=sector, response=np.ones(4, dtype=np.complex64))
            for sector, index in ((268, 0), (269, 0), (270, 0))
        )
        records = SOURCE_AF.build_source_af(observations, scope=scope)
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0].scope.as_dict()["sector_ids"], [268, 269, 270])
        self.assertIsInstance(records[0].provenance["scope"], dict)
        with self.assertRaisesRegex(ValueError, r"sectors \[268, 269, 270\]"):
            SOURCE_AF.SourceAFObservation.from_observation(_observation(0, sector_id=271), scope=scope)

    def test_explicit_validation_scope_works_and_test_scope_fails_before_response(self):
        validation_scope = SOURCE_AF.SourceAFScope(
            sector_ids=(271,),
            role="validation",
            expected_count=None,
            name="validation_scope_test",
        )
        validation_record = SOURCE_AF.build_source_af(
            (_observation(0, sector_id=271, role="validation"),),
            scope=validation_scope,
        )[0]
        self.assertEqual(validation_record.scope.role, "validation")
        with self.assertRaisesRegex(ValueError, "train or validation"):
            SOURCE_AF.SourceAFScope(sector_ids=(271,), role="test", expected_count=None)

        class SealedTestRow:
            identity = _observation_identity(0, 271)
            role = "test"
            autofocus = SimpleNamespace(official_available=True, applied=False)
            phase_reference = SimpleNamespace(
                reference_range_field="r0",
                geometry_contract="paired_monostatic_tx_equals_rx_same_observation",
            )
            r_correct_raw = 0.01
            ph_correct_raw = 0.2
            frequencies_hz = np.asarray([1.0], dtype=np.float64)

            @property
            def response(self):
                raise AssertionError("sealed test response must not be accessed")

        with self.assertRaisesRegex(ValueError, "sealed test"):
            SOURCE_AF.SourceAFObservation.from_observation(
                SealedTestRow(),
                scope=validation_scope,
            )

    def test_train_feature_failure_reasons_are_deterministic(self):
        x = np.linspace(-6.42, -3.82, 27, dtype=np.float64)
        y = np.linspace(21.68, 24.28, 27, dtype=np.float64)

        def prior_for(field):
            return {"source_bp": np.asarray(field, dtype=np.complex128), "x": x, "y": y}

        self.assertEqual(
            CORRESPONDENCE_DRIVER._select_train_feature(prior_for(np.zeros(729, dtype=np.complex128)))["reason"],
            "no_train_feature",
        )
        boundary = np.zeros(729, dtype=np.complex128)
        boundary[0] = 1.0
        self.assertEqual(
            CORRESPONDENCE_DRIVER._select_train_feature(prior_for(boundary))["reason"],
            "boundary_or_out_of_support",
        )
        competing = np.zeros(729, dtype=np.complex128)
        competing[13 * 27 + 13] = 1.0
        competing[4 * 27 + 13] = 1.0
        self.assertEqual(
            CORRESPONDENCE_DRIVER._select_train_feature(prior_for(competing))["reason"],
            "ambiguous_train_feature",
        )

    def test_float64_promotion_and_unit_modulus_sample_energy(self):
        observation = _observation(
            r0=np.float32(1_000_000.0),
            r_correct=np.float32(0.01),
            ph_correct=np.float32(0.2),
        )
        derived = SOURCE_AF.build_source_af((observation,), expected_count=1)[0]
        self.assertGreater(derived.effective_r0_m, float(np.float32(1_000_000.0)))
        self.assertEqual(float(np.float32(np.float32(1_000_000.0) + np.float32(0.01))), 1_000_000.0)
        energy = SOURCE_AF.unit_modulus_sample_energy((derived,))
        self.assertEqual(energy["observation_count"], 1)
        self.assertEqual(energy["sample_count"], 4)
        self.assertTrue(energy["passed"])

    def test_panel_compares_contributions_and_reuses_them_for_adjoints(self):
        records = SOURCE_AF.build_source_af(tuple(_observation(index) for index in range(3)), expected_count=3)
        original_direct_adjoint = SOURCE_AF.direct_adjoint
        SOURCE_AF.direct_adjoint = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("panel must reuse contribution sums"))
        try:
            report = SOURCE_AF.panel_equivalence_report((records[0],), relative_tolerance=1e-10, scaled_absolute_tolerance=1e-10)
        finally:
            SOURCE_AF.direct_adjoint = original_direct_adjoint
        self.assertEqual(report["representation_computations"], 2)
        self.assertTrue(report["contribution_metrics"]["passed"])
        self.assertTrue(report["adjoint_metrics"]["passed"])
        with self.assertRaises(ValueError):
            SOURCE_AF.direct_backproject((records[0],), np.zeros((1, 3)), representation=SOURCE_AF.EQUIVALENT_REPRESENTATION)

    def test_panel_adjoint_reference_survives_cancelling_summed_adjoint(self):
        first = _observation(0, response=np.ones(4, dtype=np.complex64))
        second = _observation(1, response=-np.ones(4, dtype=np.complex64))
        records = SOURCE_AF.build_source_af((first, second), expected_count=2)
        report = SOURCE_AF.panel_equivalence_report(records)
        adjoint = report["adjoint_metrics"]
        self.assertLess(adjoint["left_l2"], 1.0e-10)
        self.assertGreater(adjoint["reference_scale"], 1.0)
        self.assertEqual(adjoint["reference_scale_kind"], "already_computed_contribution_vector_l2_floor_1")
        self.assertTrue(adjoint["passed"])

    def test_budget_and_raw_derived_reuse_are_fail_closed(self):
        frequencies = np.linspace(9.6e9, 9.7e9, 424, dtype=np.float32)
        observations = tuple(
            _observation(index, frequencies=frequencies, response=np.ones(424, dtype=np.complex64))
            for index in range(117)
        )
        records = SOURCE_AF.build_source_af(observations, expected_count=117)
        energy = SOURCE_AF.unit_modulus_sample_energy(records)
        self.assertEqual(energy["observation_count"], 117)
        self.assertEqual(energy["sample_count"], 49608)
        budget = SOURCE_AF.budget_report(records, full_point_count=4225, panel_point_count=9)
        self.assertEqual(budget["total_native_frequency_samples"], 49608)
        self.assertEqual(budget["raw_full_branch_kernel_evaluations"], 209593800)
        self.assertEqual(budget["source_full_branch_kernel_evaluations"], 209593800)
        self.assertEqual(budget["panel_two_representation_kernel_evaluations"], 892944)
        self.assertEqual(budget["total_kernel_evaluations"], 420080544)
        with self.assertRaises(TypeError):
            SOURCE_AF.build_source_af((records[0],), expected_count=1)
        with self.assertRaises(TypeError):
            SOURCE_AF.direct_backproject((object(),), np.zeros((1, 3)), representation=SOURCE_AF.RAW_REPRESENTATION)

    def test_write_pair_png_rejects_complex_inputs_and_records_display_metadata(self):
        from PIL import Image, ImageDraw
        from PIL.PngImagePlugin import PngInfo

        support = DRIVER.CTL.NativeFrameH0Support.from_mapping(DRIVER_PROTOCOL["support"])
        with _writable_temp_dir() as workspace:
            output = Path(workspace) / "render.png"
            raw_db = np.full(support.point_count, -40.0, dtype=np.float64)
            source_db = np.full(support.point_count, 0.0, dtype=np.float64)
            with self.assertRaises(TypeError):
                DRIVER._write_pair_png(
                    output,
                    raw_db.astype(np.complex128),
                    source_db,
                    support,
                    Image=Image,
                    ImageDraw=ImageDraw,
                    PngInfo=PngInfo,
                    common_reference=1.0,
                )

            metadata = DRIVER._write_pair_png(
                output,
                raw_db,
                source_db,
                support,
                Image=Image,
                ImageDraw=ImageDraw,
                PngInfo=PngInfo,
                common_reference=1.0,
            )
            self.assertEqual(metadata["display_quantity"], "20log10(abs(BP)/A_ref)")
            self.assertEqual(metadata["db_clip"], [-40.0, 0.0])
            self.assertEqual(metadata["A_ref"], 1.0)
            self.assertTrue(metadata["common_scale"])
            geometry = metadata["geometry"]
            panel_width, panel_height = geometry["panel_size_px"]
            left_x, left_y = geometry["left_origin_px"]
            right_x, right_y = geometry["right_origin_px"]
            left_center = (left_x + panel_width // 2, left_y + panel_height // 2)
            right_center = (right_x + panel_width // 2, right_y + panel_height // 2)
            with Image.open(output) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.getpixel(left_center), (0, 0, 255))
                self.assertEqual(image.getpixel(right_center), (255, 0, 0))
                self.assertIn("Display", image.info)
                draw = ImageDraw.Draw(image)
                for index, line in enumerate(metadata["header_lines"]):
                    self.assertLessEqual(
                        draw.textbbox((6, 4 + index * 16), line)[2],
                        image.width,
                    )

    def test_run_compare_sealed_synthetic_cli_executes_real_full_bps(self):
        from PIL import Image

        with _writable_temp_dir() as workspace:
            output_dir = Path(workspace) / "compare"
            cli_payload, cli_output, archive_path, shard = _run_sealed_synthetic_compare_via_cli(output_dir)
            self.assertEqual(cli_payload["status"], "PASS")
            self.assertIn('"status": "PASS"', cli_output)
            self.assertEqual(shard.view_count, 440)
            self.assertEqual(len(shard.identities_for_role("train")), 404)
            self.assertEqual(len(shard.identities_for_role("validation")), 36)
            self.assertEqual(shard.frequencies_hz.shape, (424,))
            self.assertEqual(shard.response.shape, (440, 424))
            selected_ids = tuple(sorted((identity for identity in shard.observation_ids if identity.sector_id == 2), key=DRIVER._id_key))
            selected_observations = shard.observations(selected_ids)
            self.assertEqual(len(selected_observations), 117)
            self.assertTrue(all(observation.frequencies_hz.shape == (424,) for observation in selected_observations))
            self.assertTrue(all(np.array_equal(observation.frequencies_hz, shard.frequencies_hz) for observation in selected_observations))
            self.assertEqual(shard.metadata["role_counts"], {"train": 404, "validation": 36, "test": 0})
            self.assertEqual(len(shard.metadata["payload_sector_ids"]), 324)
            self.assertEqual(len(shard.metadata["sealed_test_sector_ids"]), 36)
            with zipfile.ZipFile(archive_path, "r") as archive:
                self.assertEqual(archive.infolist()[0].filename, "metadata_json.npy")
                self.assertTrue(all(item.compress_type == zipfile.ZIP_STORED for item in archive.infolist()))
            expected_files = [
                "protocol_echo.json",
                "preflight.json",
                "resource.json",
                "comparison_report.json",
                "status.json",
                "bp_raw_unnormalized.npy",
                "bp_source_unnormalized.npy",
                "bp_raw_mean_native_sample.npy",
                "bp_source_mean_native_sample.npy",
                "x.npy",
                "y.npy",
                "bp_raw_source_compare.png",
            ]
            for name in expected_files:
                self.assertTrue((output_dir / name).is_file(), f"missing {name}")

            preflight = json.loads((output_dir / "preflight.json").read_text(encoding="utf-8"))
            resource = json.loads((output_dir / "resource.json").read_text(encoding="utf-8"))
            comparison = json.loads((output_dir / "comparison_report.json").read_text(encoding="utf-8"))
            raw_bp = np.load(output_dir / "bp_raw_unnormalized.npy")
            source_bp = np.load(output_dir / "bp_source_unnormalized.npy")
            self.assertEqual(preflight["budget"]["total_kernel_evaluations"], 420080544)
            self.assertEqual(preflight["budget"]["raw_full_branch_kernel_evaluations"], 209593800)
            self.assertEqual(preflight["budget"]["source_full_branch_kernel_evaluations"], 209593800)
            self.assertEqual(preflight["budget_before_correction"]["total_native_frequency_samples"], 49608)
            self.assertTrue(preflight["budget_before_correction"]["guard_passed_before_correction_or_bp"])
            self.assertTrue(comparison["panel_equivalence"]["contribution_metrics"]["passed"])
            self.assertTrue(comparison["panel_equivalence"]["adjoint_metrics"]["passed"])
            self.assertEqual(comparison["disclosure"]["test_payload_opened"], False)
            self.assertEqual(comparison["disclosure"]["validation_used_for_bp"], False)
            self.assertEqual(comparison["disclosure"]["selected_response_count"], 117)
            self.assertEqual(comparison["disclosure"]["selected_response_shape"], [117, 424])
            self.assertEqual(comparison["disclosure"]["frequency_policy"], "native_stored_exact")
            self.assertEqual(comparison["disclosure"]["frequency_vector_layout"], "one_shared_exact_vector_per_shard")
            self.assertEqual(comparison["disclosure"]["shared_native_frequency_count"], 424)
            self.assertEqual(comparison["disclosure"]["loaded_response_role_counts"], {"train": 404, "validation": 36})
            self.assertEqual(comparison["disclosure"]["selected_frequency_counts"], [424] * 117)
            self.assertEqual(comparison["disclosure"]["loaded_response_count"], 440)
            self.assertTrue(comparison["disclosure"]["validation_response_payload_materialized"])
            self.assertFalse(comparison["disclosure"]["test_response_payload_materialized"])
            self.assertFalse(comparison["disclosure"]["validation_used_for_panel"])
            self.assertFalse(comparison["disclosure"]["validation_used_for_mean_native_sample_normalization"])
            self.assertFalse(comparison["disclosure"]["validation_used_for_common_reference_or_display"])
            self.assertTrue(comparison["disclosure"]["selected_train_rows_only_for_downstream_operations"])
            self.assertEqual(comparison["disclosure"]["test_sealing_contract"], "archive_metadata_and_row_exclusion_not_independent_historical_exposure_proof")
            self.assertTrue(comparison["disclosure"]["origin_only_diagnostic"])
            self.assertFalse(comparison["disclosure"]["calibration_closure"])
            self.assertEqual(comparison["frequency_policy"], "native_stored_exact")
            self.assertEqual(comparison["frequency_vector_layout"], "one_shared_exact_vector_per_shard")
            self.assertEqual(comparison["shared_native_frequency_count"], 424)
            self.assertEqual(comparison["selected_response_shape"], [117, 424])
            self.assertIn("not a target cube, fitted field, localization, calibration closure, focus/geometry/accuracy result", comparison["diagnostic_statement"])
            self.assertTrue(comparison["native_per_pulse_r0"])
            self.assertEqual(comparison["native_per_pulse_r0"], True)
            self.assertEqual(resource["selected_pulse_count"], 117)
            self.assertEqual(resource["point_chunk_size"], 4096)
            self.assertGreater(resource["elapsed_seconds"], 0.0)
            self.assertEqual(comparison["display"]["db_clip"], [-40.0, 0.0])
            self.assertEqual(comparison["display"]["origin"], "lower")
            self.assertEqual(comparison["display"]["aspect"], "equal")
            self.assertEqual(raw_bp.shape, (4225,))
            self.assertEqual(source_bp.shape, (4225,))
            self.assertEqual(raw_bp.dtype, np.dtype(np.complex128))
            self.assertEqual(source_bp.dtype, np.dtype(np.complex128))
            self.assertGreater(float(np.max(np.abs(raw_bp))), 0.0)
            self.assertGreater(float(np.max(np.abs(source_bp))), 0.0)
            self.assertGreater(float(np.linalg.norm(source_bp - raw_bp)), 0.0)
            with Image.open(output_dir / "bp_raw_source_compare.png") as image:
                self.assertEqual(image.mode, "RGB")
                self.assertIn("Display", image.info)
                self.assertIn("one shared exact native frequency vector per shard", image.info["Description"])
                self.assertIn("selected response shape [117, 424]", image.info["Description"])
                self.assertIn("A_ref=max(abs(raw),abs(source))", image.info["Display"])
                self.assertIn("fixed clip [-40,0] dB", image.info["Display"])
            sentinel_path = output_dir / "sentinel.txt"
            sentinel_path.write_text("still_here", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                DRIVER.run_compare(
                    DRIVER_PROTOCOL_PATH,
                    str(output_dir.parent / "archive"),
                    str(output_dir),
                    stage="compare",
                )
            self.assertEqual(sentinel_path.read_text(encoding="utf-8"), "still_here")

    def test_run_compare_rejects_wrong_shared_frequency_count_before_output(self):
        with _writable_temp_dir() as workspace:
            archive_root = Path(workspace) / "archive"
            archive_path = _build_sealed_synthetic_archive(archive_root, frequency_count=423)
            output_dir = Path(workspace) / "bad_frequency_compare"
            with self.assertRaisesRegex(ValueError, "length 424"):
                DRIVER.run_compare(
                    DRIVER_PROTOCOL_PATH,
                    str(archive_root),
                    str(output_dir),
                    stage="compare",
                )
            self.assertTrue(archive_path.is_file())
            self.assertFalse(output_dir.exists())

    def test_run_nominal_psf_sealed_synthetic_cli_uses_fixed_scope_and_matched_psf(self):
        from PIL import Image, ImageDraw

        with _writable_temp_dir() as workspace:
            output_dir = Path(workspace) / "nominal_psf"
            cli_payload, cli_output, archive_path, shard = _run_nominal_psf_synthetic_via_cli(output_dir)
            self.assertEqual(cli_payload["status"], "PASS")
            self.assertIn('"status": "PASS"', cli_output)
            self.assertEqual(shard.frequencies_hz.shape, (424,))
            self.assertEqual(shard.metadata["role_counts"]["validation"], 36)
            self.assertFalse(shard.metadata["test_opened"])
            selected = tuple(
                identity
                for identity in shard.observation_ids
                if identity.sector_id in (268, 269, 270)
            )
            self.assertEqual(len(selected), 6)
            self.assertTrue(all(str(shard.role[index]).lower() == "train" for index, identity in enumerate(shard.observation_ids) if identity in selected))
            self.assertNotIn(271, {identity.sector_id for identity in selected})
            with zipfile.ZipFile(archive_path, "r") as archive:
                self.assertTrue(all(item.compress_type == zipfile.ZIP_STORED for item in archive.infolist()))

            preflight = json.loads((output_dir / "preflight.json").read_text(encoding="utf-8"))
            resource = json.loads((output_dir / "resource.json").read_text(encoding="utf-8"))
            report = json.loads((output_dir / "comparison_report.json").read_text(encoding="utf-8"))
            self.assertEqual(preflight["selected_record_count"], 6)
            self.assertEqual(preflight["frequency_samples_per_record"], 424)
            self.assertEqual(preflight["total_native_frequency_samples"], 6 * 424)
            self.assertEqual(len(preflight["target_to_antenna_horizontal_bearings_deg"]), 6)
            self.assertEqual(len(preflight["target_to_antenna_elevations_deg"]), 6)
            self.assertGreater(preflight["horizontal_bearing_summary_deg"]["span"], 0.0)
            self.assertEqual(preflight["status"], "inconclusive_non_resolving_metadata_only")
            self.assertTrue(preflight["budget_guard_passed_before_response_conversion_or_bp"])
            budget = preflight["budget"]
            native_samples = 6 * 424
            self.assertEqual(budget["N"], native_samples)
            self.assertEqual(budget["total_kernel_evaluations"], 2918 * native_samples)
            self.assertEqual(
                budget["total_kernel_evaluations"],
                budget["measured_raw_bp_kernel_evaluations"]
                + budget["measured_source_bp_kernel_evaluations"]
                + budget["raw_unit_point_forward_kernel_evaluations"]
                + budget["raw_unit_point_psf_bp_kernel_evaluations"]
                + budget["source_unit_point_forward_kernel_evaluations"]
                + budget["source_unit_point_psf_bp_kernel_evaluations"],
            )
            self.assertEqual(resource["kernel_ledger"]["extra_probe_or_reference_passes"], 0)
            with self.assertRaisesRegex(RuntimeError, "total kernel budget exceeded"):
                SOURCE_AF.matched_unit_point_psf_budget(
                    record_count=6,
                    frequency_count=424,
                    support_point_count=729,
                    max_total_kernel_evaluations=1,
                )

            expected_files = [
                "protocol_echo.json", "preflight.json", "resource.json", "comparison_report.json", "status.json",
                "bp_raw_unnormalized.npy", "bp_source_unnormalized.npy",
                "bp_raw_mean_native_sample.npy", "bp_source_mean_native_sample.npy",
                "psf_raw_unnormalized.npy", "psf_source_unnormalized.npy",
                "psf_raw_mean_native_sample.npy", "psf_source_mean_native_sample.npy",
                "x.npy", "y.npy", "bp_raw_source_compare.png", "psf_raw_source_compare.png",
            ]
            for name in expected_files:
                self.assertTrue((output_dir / name).is_file(), f"missing {name}")
            raw_bp_unnormalized = np.load(output_dir / "bp_raw_unnormalized.npy")
            source_bp_unnormalized = np.load(output_dir / "bp_source_unnormalized.npy")
            raw_psf_unnormalized = np.load(output_dir / "psf_raw_unnormalized.npy")
            source_psf_unnormalized = np.load(output_dir / "psf_source_unnormalized.npy")
            raw_bp = np.load(output_dir / "bp_raw_mean_native_sample.npy")
            source_bp = np.load(output_dir / "bp_source_mean_native_sample.npy")
            raw_psf = np.load(output_dir / "psf_raw_mean_native_sample.npy")
            source_psf = np.load(output_dir / "psf_source_mean_native_sample.npy")
            x_coordinates = np.load(output_dir / "x.npy")
            y_coordinates = np.load(output_dir / "y.npy")
            self.assertEqual(raw_bp.shape, (729,))
            self.assertEqual(source_bp.shape, (729,))
            self.assertEqual(raw_psf.shape, (729,))
            self.assertEqual(source_psf.shape, (729,))
            self.assertEqual(raw_psf.dtype, np.dtype(np.complex128))
            self.assertTrue(np.allclose(x_coordinates[[0, -1]], [-6.42, -3.82], rtol=0.0, atol=1.0e-15))
            self.assertTrue(np.allclose(y_coordinates[[0, -1]], [21.68, 24.28], rtol=0.0, atol=1.0e-15))
            self.assertTrue(
                np.allclose(
                    [x_coordinates[13], y_coordinates[13], -0.05],
                    [-5.12, 22.98, -0.05],
                    rtol=0.0,
                    atol=1.0e-15,
                )
            )
            dynamic_native_sample_count = float(6 * 424)
            self.assertTrue(np.array_equal(raw_bp, raw_bp_unnormalized / dynamic_native_sample_count))
            self.assertTrue(np.array_equal(source_bp, source_bp_unnormalized / dynamic_native_sample_count))
            self.assertTrue(np.array_equal(raw_psf, raw_psf_unnormalized / dynamic_native_sample_count))
            self.assertTrue(np.array_equal(source_psf, source_psf_unnormalized / dynamic_native_sample_count))
            self.assertTrue(np.allclose(raw_psf, source_psf, rtol=5e-12, atol=5e-12))
            center_index = 13 * 27 + 13
            self.assertEqual(center_index, 364)
            self.assertTrue(np.allclose(raw_psf[center_index], 1.0 + 0.0j, rtol=0.0, atol=1.0e-11))
            self.assertTrue(np.allclose(source_psf[center_index], 1.0 + 0.0j, rtol=0.0, atol=1.0e-11))
            self.assertGreaterEqual(
                abs(raw_psf[center_index]),
                float(np.max(np.abs(raw_psf))) * (1.0 - 1.0e-12),
            )
            self.assertTrue(report["disclosure"]["validation_role_unused"])
            self.assertFalse(report["disclosure"]["validation_used_for_bp"])
            self.assertTrue(report["disclosure"]["nominal_point_identity_mapping_unverified"])
            self.assertTrue(report["disclosure"]["heading_convention_unverified"])
            self.assertTrue(report["disclosure"]["support_is_not_a_phase_center_displacement_bound"])
            self.assertTrue(report["disclosure"]["pitch_is_not_coherent_quadrature_claim"])
            self.assertTrue(report["disclosure"]["psf_scale_independent_of_measured_pair"])
            self.assertFalse(report["matched_psf_operator_sanity"]["measured_response_used"])
            with Image.open(output_dir / "psf_raw_source_compare.png") as image:
                self.assertIn("own common A_ref", image.info["Description"])
                draw = ImageDraw.Draw(image)
                for index, line in enumerate(report["display"]["psf_pair"]["header_lines"]):
                    self.assertLessEqual(draw.textbbox((6, 4 + index * 16), line)[2], image.width)

    def test_nominal_psf_selection_rejects_missing_declared_sector(self):
        protocol = NOMINAL_PSF_DRIVER._read_protocol(NOMINAL_PSF_PROTOCOL_PATH)
        fake_shard = SimpleNamespace(
            observation_ids=(
                _observation_identity(0, 268),
                _observation_identity(0, 269),
                _observation_identity(0, 271),
            ),
            role=np.asarray(["train", "train", "validation"], dtype="U10"),
        )
        with self.assertRaisesRegex(ValueError, "requires every declared sector exactly"):
            NOMINAL_PSF_DRIVER._select_metadata_scope(fake_shard, protocol)

    def test_nominal_psf_over_cap_guard_precedes_source_conversion_bp_and_output(self):
        with _writable_temp_dir() as workspace:
            output_dir = Path(workspace) / "over_cap_nominal_psf"
            archive_root = Path(workspace) / "over_cap_archive"
            archive_path = _build_sealed_synthetic_archive(
                archive_root,
                sector_pulse_counts={268: 135, 269: 135, 270: 135},
            )
            copied_protocol = json.loads(NOMINAL_PSF_PROTOCOL_PATH.read_text(encoding="utf-8"))
            derived_native_samples = 405 * 424
            derived_total = 2918 * derived_native_samples
            self.assertGreater(derived_total, copied_protocol["budget"]["max_total_kernel_evaluations"])
            copied_protocol_path = Path(workspace) / "copied_over_cap_protocol.json"
            copied_protocol_path.write_text(json.dumps(copied_protocol, indent=2), encoding="utf-8")

            original_build = NOMINAL_PSF_DRIVER.SOURCE_AF.build_source_af
            original_backproject = NOMINAL_PSF_DRIVER.SOURCE_AF.direct_backproject
            original_psf = NOMINAL_PSF_DRIVER.SOURCE_AF.matched_unit_point_psf

            def fail_if_called(*args, **kwargs):
                raise AssertionError("over-cap guard must precede source conversion/BP")

            NOMINAL_PSF_DRIVER.SOURCE_AF.build_source_af = fail_if_called
            NOMINAL_PSF_DRIVER.SOURCE_AF.direct_backproject = fail_if_called
            NOMINAL_PSF_DRIVER.SOURCE_AF.matched_unit_point_psf = fail_if_called
            try:
                with self.assertRaisesRegex(RuntimeError, "total kernel budget exceeded"):
                    NOMINAL_PSF_DRIVER.run_15tr07(
                        copied_protocol_path,
                        archive_root,
                        output_dir,
                        stage="diagnostic",
                    )
            finally:
                NOMINAL_PSF_DRIVER.SOURCE_AF.build_source_af = original_build
                NOMINAL_PSF_DRIVER.SOURCE_AF.direct_backproject = original_backproject
                NOMINAL_PSF_DRIVER.SOURCE_AF.matched_unit_point_psf = original_psf
            self.assertTrue(archive_path.is_file())
            self.assertFalse(output_dir.exists())

    def test_correspondence_verify_supported_keeps_physical_claims_unresolved(self):
        with _writable_temp_dir() as workspace:
            (
                cli_payload,
                cli_output,
                archive_path,
                archive_root,
                prior_output_dir,
                output_dir,
            ) = _run_known_point_correspondence_fixture(workspace)
            self.assertEqual(cli_payload["status"], "PASS")
            self.assertEqual(cli_payload["technical_execution_status"], "PASS")
            self.assertEqual(cli_payload["evidence_decision"], "supported_conditional_holdout_local_psf_shape_consistency")
            self.assertIn('"status": "PASS"', cli_output)
            self.assertTrue(archive_path.is_file())
            report = json.loads((output_dir / "correspondence_report.json").read_text(encoding="utf-8"))
            preflight = json.loads((output_dir / "preflight.json").read_text(encoding="utf-8"))
            resource = json.loads((output_dir / "resource.json").read_text(encoding="utf-8"))
            feature = json.loads((output_dir / "train_feature.json").read_text(encoding="utf-8"))
            confirmation = json.loads((output_dir / "validation_confirmation.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["technical_execution_status"], "PASS")
            self.assertEqual(report["evidence_decision"], report["decision"])
            self.assertEqual(report["decision"], "supported_conditional_holdout_local_psf_shape_consistency")
            self.assertEqual(feature["conditional_train_grid_feature_xyz_m"], [-5.12, 22.98, -0.05])
            self.assertTrue(feature["candidate_frozen_from_train"])
            self.assertTrue(feature["not_a_registered_offset_or_physical_phase_center"])
            self.assertEqual(confirmation["role"], "validation")
            self.assertFalse(confirmation["used_for_selection"])
            self.assertTrue(confirmation["candidate_frozen_from_train"])
            self.assertEqual(preflight["validation_scope_preflight"]["selected_record_count"], 12)
            self.assertTrue(preflight["validation_scope_preflight"]["loader_materialized_non_test_response_before_new_work_guard"])
            self.assertTrue(preflight["current_archive_train_metadata"]["semantic_match_to_prior_train_provenance"])
            self.assertTrue(preflight["current_archive_train_metadata"]["not_an_archival_identity_proof"])
            self.assertTrue(report["disclosure"]["validation_response_payload_materialized_by_loader"])
            self.assertTrue(report["disclosure"]["selected_validation_observation_objects_materialized_after_guard"])
            self.assertFalse(report["disclosure"]["validation_used_for_selection"])
            self.assertFalse(report["disclosure"]["test_payload_opened"])
            self.assertTrue(report["disclosure"]["supplied_per_row_af_corrections_may_use_validation_responses"])
            self.assertTrue(report["prior_raw_source_diagnostic_context_only"])
            self.assertEqual(report["claims"]["coordinate_alignment"], "not_identifiable_without_independent_landmarks")
            self.assertEqual(report["claims"]["point_c_scattering_center"], "not_identifiable_without_independent_point_c_semantics_or_calibration")
            self.assertEqual(report["claims"]["physical_15tr07_return_association"], "inconclusive_without_external_identity_evidence")
            self.assertEqual(resource["budget"]["N_val"], 12 * 424)
            self.assertEqual(resource["budget"]["new_validation_kernel_evaluations"], (2 * 81 + 1) * 12 * 424)
            self.assertEqual(resource["budget"]["panel_kernel_evaluations"], 0)
            self.assertEqual(resource["budget"]["probe_kernel_evaluations"], 0)
            self.assertEqual(resource["budget"]["fit_kernel_evaluations"], 0)
            validation_bp = np.load(output_dir / "validation_bp_patch_mean_native_sample.npy")
            validation_psf = np.load(output_dir / "validation_psf_patch_mean_native_sample.npy")
            patch_coordinates = np.load(output_dir / "validation_patch_coordinates.npy")
            self.assertEqual(validation_bp.shape, (81,))
            self.assertEqual(validation_psf.shape, (81,))
            self.assertEqual(patch_coordinates.shape, (81, 3))
            self.assertTrue(np.allclose(patch_coordinates[40], [-5.12, 22.98, -0.05], rtol=0.0, atol=1.0e-12))
            self.assertTrue(np.isfinite(validation_bp.real).all())
            self.assertTrue(np.isfinite(validation_psf.real).all())
            shard = DRIVER.ACQ.load_native_shard(
                archive_path,
                expected_pass_id=1,
                expected_polarization="hh",
                expected_scene_id="gotcha_v1_joint8_fullpol",
            )
            first_train_id = next(identity for identity in shard.observation_ids if identity.sector_id == 268 and identity.pulse_index == 0)
            native_observation = shard.observation(first_train_id)
            derived = SOURCE_AF.build_source_af(
                (native_observation,),
                scope=SOURCE_AF.SourceAFScope(
                    pass_id=1,
                    polarization="hh",
                    sector_ids=(268,),
                    role="train",
                    expected_count=1,
                    name="synthetic_oracle_assertion",
                ),
            )[0]
            oracle = DRIVER.ACQ.direct_point_target_render(
                np.asarray([[-5.12, 22.98, -0.05]], dtype=np.float64),
                np.asarray([1.0 + 0.0j], dtype=np.complex128),
                tx_positions_m=native_observation.position_xyz_m[None, :],
                rx_positions_m=native_observation.position_xyz_m[None, :],
                frequencies_hz=native_observation.frequencies_hz,
                reference_range_m=[native_observation.r0_m + native_observation.r_correct_raw],
                convention=DRIVER.ACQ.PUBLISHED_GOTCHA_REFERENCE_CANDIDATE,
            )[0]
            np.testing.assert_allclose(derived.effective_response, oracle, rtol=1.0e-6, atol=1.0e-6)
            self.assertGreater(float(np.linalg.norm(derived.effective_response - derived.response_raw)), 0.0)
            self.assertEqual(report["decision_metric"]["metric"]["validation_train_patch_norm_ratio_threshold"], 0.25)
            self.assertEqual(report["decision_metric"]["metric"]["rho_supported_threshold"], 0.8)
            self.assertEqual(report["decision_metric"]["metric"]["peak_distance_supported_max_cells"], 1)

    def test_correspondence_verify_validation_change_cannot_change_train_candidate(self):
        with _writable_temp_dir() as workspace:
            first = _run_known_point_correspondence_fixture(workspace / "first")
            second = _run_known_point_correspondence_fixture(
                workspace / "second",
                validation_point_xyz_m=(-4.72, 22.98, -0.05),
            )
            first_feature = json.loads((first[5] / "train_feature.json").read_text(encoding="utf-8"))
            second_feature = json.loads((second[5] / "train_feature.json").read_text(encoding="utf-8"))
            self.assertEqual(
                first_feature["conditional_train_grid_feature_xyz_m"],
                second_feature["conditional_train_grid_feature_xyz_m"],
            )
            second_report = json.loads((second[5] / "correspondence_report.json").read_text(encoding="utf-8"))
            self.assertFalse(second_report["validation_used_for_selection"])
            self.assertTrue(second_report["candidate_frozen_from_train"])
            self.assertEqual(second_report["claims"]["physical_15tr07_return_association"], "inconclusive_without_external_identity_evidence")

    def test_correspondence_verify_repeatable_synthetic_clutter_stays_physically_unidentified(self):
        with _writable_temp_dir() as workspace:
            clutter_point = (-5.02, 23.08, -0.05)
            result = _run_known_point_correspondence_fixture(
                workspace,
                known_point_xyz_m=clutter_point,
                validation_point_xyz_m=clutter_point,
            )
            report = json.loads((result[5] / "correspondence_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["decision"], "supported_conditional_holdout_local_psf_shape_consistency")
            self.assertEqual(report["claims"]["physical_15tr07_return_association"], "inconclusive_without_external_identity_evidence")
            self.assertTrue(report["claims"]["conditional_response_result_does_not_change_physical_identity_status"])

    def test_correspondence_verify_shape_mismatch_is_fixed_model_inconsistent_only(self):
        with _writable_temp_dir() as workspace:
            result = _run_known_point_correspondence_fixture(
                workspace,
                validation_point_xyz_m=(-4.72, 22.98, -0.05),
            )
            report = json.loads((result[5] / "correspondence_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["decision"], "inconsistent_with_fixed_single_point_source_af_response_model")
            self.assertEqual(report["claims"]["coordinate_alignment"], "not_identifiable_without_independent_landmarks")
            self.assertEqual(report["claims"]["physical_15tr07_return_association"], "inconclusive_without_external_identity_evidence")

    def test_correspondence_verify_early_train_candidate_failures_stop_before_archive_or_operators(self):
        with _writable_temp_dir() as workspace:
            baseline = _run_known_point_correspondence_fixture(workspace / "baseline")
            prior_source = baseline[4]
            archive_root = baseline[3]
            source_bp_path = prior_source / "bp_source_mean_native_sample.npy"
            source_bp = np.load(source_bp_path, allow_pickle=False)
            cases = {
                "absent": np.zeros_like(source_bp),
                "boundary": np.eye(1, source_bp.size, 0, dtype=np.complex128).reshape(-1),
                "competing": np.zeros_like(source_bp),
            }
            cases["boundary"][0] = 1.0 + 0.0j
            cases["competing"][13 * 27 + 13] = 1.0 + 0.0j
            cases["competing"][4 * 27 + 13] = 1.0 + 0.0j
            original_loader = CORRESPONDENCE_DRIVER.ACQ.load_native_shard
            original_observations = CORRESPONDENCE_DRIVER.ACQ.NativeShard.observations
            original_build = CORRESPONDENCE_DRIVER.SOURCE_AF.build_source_af
            original_backproject = CORRESPONDENCE_DRIVER.SOURCE_AF.direct_backproject
            original_psf = CORRESPONDENCE_DRIVER.SOURCE_AF.matched_unit_point_psf

            def fail_if_called(*args, **kwargs):
                raise AssertionError("early candidate failure must stop before archive/validation/operator work")

            CORRESPONDENCE_DRIVER.ACQ.load_native_shard = fail_if_called
            CORRESPONDENCE_DRIVER.ACQ.NativeShard.observations = fail_if_called
            CORRESPONDENCE_DRIVER.SOURCE_AF.build_source_af = fail_if_called
            CORRESPONDENCE_DRIVER.SOURCE_AF.direct_backproject = fail_if_called
            CORRESPONDENCE_DRIVER.SOURCE_AF.matched_unit_point_psf = fail_if_called
            try:
                for name, mutated_source_bp in cases.items():
                    case_dir = workspace / name
                    case_prior = case_dir / "prior"
                    shutil.copytree(prior_source, case_prior)
                    np.save(case_prior / "bp_source_mean_native_sample.npy", mutated_source_bp)
                    output_dir = case_dir / "verification"
                    result = CORRESPONDENCE_DRIVER.run_verification(
                        CORRESPONDENCE_PROTOCOL_PATH,
                        archive_root,
                        case_prior,
                        output_dir,
                        stage="verification",
                    )
                    self.assertEqual(result["status"], "PASS")
                    self.assertEqual(result["technical_execution_status"], "PASS")
                    self.assertEqual(result["evidence_decision"], "inconclusive_nonresolving_validation_patch")
                    feature = json.loads((output_dir / "train_feature.json").read_text(encoding="utf-8"))
                    self.assertEqual(feature["reason"], {
                        "absent": "no_train_feature",
                        "boundary": "boundary_or_out_of_support",
                        "competing": "ambiguous_train_feature",
                    }[name])
                    report = json.loads((output_dir / "correspondence_report.json").read_text(encoding="utf-8"))
                    resource = json.loads((output_dir / "resource.json").read_text(encoding="utf-8"))
                    self.assertFalse(report["disclosure"]["current_archive_accessed"])
                    self.assertEqual(report["disclosure"]["loaded_response_roles"], [])
                    self.assertEqual(report["disclosure"]["used_response_roles"], [])
                    self.assertFalse(report["disclosure"]["validation_response_payload_materialized_by_loader"])
                    self.assertFalse(resource["loader_materialized_non_test_response_before_new_work_guard"])
                    self.assertFalse(resource["selected_validation_observation_objects_materialized_after_guard"])
                    status = json.loads((output_dir / "status.json").read_text(encoding="utf-8"))
                    self.assertEqual(status["status"], "PASS")
                    self.assertEqual(status["technical_execution_status"], "PASS")
                    self.assertEqual(status["evidence_decision"], "inconclusive_nonresolving_validation_patch")
            finally:
                CORRESPONDENCE_DRIVER.ACQ.load_native_shard = original_loader
                CORRESPONDENCE_DRIVER.ACQ.NativeShard.observations = original_observations
                CORRESPONDENCE_DRIVER.SOURCE_AF.build_source_af = original_build
                CORRESPONDENCE_DRIVER.SOURCE_AF.direct_backproject = original_backproject
                CORRESPONDENCE_DRIVER.SOURCE_AF.matched_unit_point_psf = original_psf

    def test_correspondence_verify_rejects_current_train_metadata_before_observations_or_operators(self):
        with _writable_temp_dir() as workspace:
            baseline = _run_known_point_correspondence_fixture(workspace / "baseline")
            current_archive_root = workspace / "mismatched_current_archive"
            _build_sealed_synthetic_archive(
                current_archive_root,
                sector_pulse_counts={268: 13, 269: 12, 270: 12, 271: 12},
                frequency_span_hz=(9.0e9, 11.0e9),
            )
            output_dir = workspace / "mismatched_verification"
            original_observations = CORRESPONDENCE_DRIVER.ACQ.NativeShard.observations
            original_build = CORRESPONDENCE_DRIVER.SOURCE_AF.build_source_af
            original_backproject = CORRESPONDENCE_DRIVER.SOURCE_AF.direct_backproject
            original_psf = CORRESPONDENCE_DRIVER.SOURCE_AF.matched_unit_point_psf

            def fail_if_called(*args, **kwargs):
                raise AssertionError("train provenance mismatch must stop before validation/operator work")

            CORRESPONDENCE_DRIVER.ACQ.NativeShard.observations = fail_if_called
            CORRESPONDENCE_DRIVER.SOURCE_AF.build_source_af = fail_if_called
            CORRESPONDENCE_DRIVER.SOURCE_AF.direct_backproject = fail_if_called
            CORRESPONDENCE_DRIVER.SOURCE_AF.matched_unit_point_psf = fail_if_called
            try:
                with self.assertRaisesRegex(ValueError, "current train metadata IDs/count/order"):
                    CORRESPONDENCE_DRIVER.run_verification(
                        CORRESPONDENCE_PROTOCOL_PATH,
                        current_archive_root,
                        baseline[4],
                        output_dir,
                        stage="verification",
                    )
            finally:
                CORRESPONDENCE_DRIVER.ACQ.NativeShard.observations = original_observations
                CORRESPONDENCE_DRIVER.SOURCE_AF.build_source_af = original_build
                CORRESPONDENCE_DRIVER.SOURCE_AF.direct_backproject = original_backproject
                CORRESPONDENCE_DRIVER.SOURCE_AF.matched_unit_point_psf = original_psf
            self.assertFalse(output_dir.exists())

    def test_correspondence_verify_over_cap_precedes_validation_observations_and_operators(self):
        with _writable_temp_dir() as workspace:
            prior_output_dir = workspace / "prior_nominal"
            prior_archive_root = workspace / "prior_archive"
            _run_nominal_psf_synthetic_via_cli(
                prior_output_dir,
                archive_root=prior_archive_root,
                sector_pulse_counts={268: 12, 269: 12, 270: 12, 271: 12},
                known_point_xyz_m=(-5.12, 22.98, -0.05),
                validation_point_xyz_m=(-5.12, 22.98, -0.05),
                frequency_span_hz=(9.0e9, 11.0e9),
            )
            over_cap_archive_root = workspace / "over_cap_archive"
            _build_sealed_synthetic_archive(
                over_cap_archive_root,
                sector_pulse_counts={268: 12, 269: 12, 270: 12, 271: 6600},
                frequency_span_hz=(9.0e9, 11.0e9),
            )
            output_dir = workspace / "over_cap_verification"
            original_observations = CORRESPONDENCE_DRIVER.ACQ.NativeShard.observations
            original_build = CORRESPONDENCE_DRIVER.SOURCE_AF.build_source_af
            original_backproject = CORRESPONDENCE_DRIVER.SOURCE_AF.direct_backproject
            original_psf = CORRESPONDENCE_DRIVER.SOURCE_AF.matched_unit_point_psf

            def fail_if_called(*args, **kwargs):
                raise AssertionError("over-cap guard must precede validation materialization/conversion/operator")

            CORRESPONDENCE_DRIVER.ACQ.NativeShard.observations = fail_if_called
            CORRESPONDENCE_DRIVER.SOURCE_AF.build_source_af = fail_if_called
            CORRESPONDENCE_DRIVER.SOURCE_AF.direct_backproject = fail_if_called
            CORRESPONDENCE_DRIVER.SOURCE_AF.matched_unit_point_psf = fail_if_called
            try:
                with self.assertRaisesRegex(RuntimeError, "combined source-AF historical plus validation kernel budget exceeded"):
                    CORRESPONDENCE_DRIVER.run_verification(
                        CORRESPONDENCE_PROTOCOL_PATH,
                        over_cap_archive_root,
                        prior_output_dir,
                        output_dir,
                        stage="verification",
                    )
            finally:
                CORRESPONDENCE_DRIVER.ACQ.NativeShard.observations = original_observations
                CORRESPONDENCE_DRIVER.SOURCE_AF.build_source_af = original_build
                CORRESPONDENCE_DRIVER.SOURCE_AF.direct_backproject = original_backproject
                CORRESPONDENCE_DRIVER.SOURCE_AF.matched_unit_point_psf = original_psf
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
