from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import replace
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import unittest
from unittest import mock
import zipfile

import numpy as np
from scipy.io import savemat


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _PROJECT_ROOT / "scripts" / "materialize_gotcha_joint_fullpol_v1.py"
_SPEC = importlib.util.spec_from_file_location(
    "materialize_gotcha_joint_fullpol_v1", _SCRIPT
)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
_TEMP_COUNTER = itertools.count()


@contextmanager
def _temporary_directory():
    root = _PROJECT_ROOT / ".codex-tmp"
    root.mkdir(parents=True, exist_ok=True)
    directory = root / f"gate1-test-{os.getpid()}-{next(_TEMP_COUNTER)}"
    directory.mkdir()
    try:
        yield str(directory)
    finally:
        shutil.rmtree(directory, ignore_errors=False)


def _gate0_payload() -> dict[str, object]:
    split = _MODULE.build_sector_split()
    shards = []
    total_views = 0
    total_samples = 0
    for pass_id in _MODULE.PASS_IDS:
        for polarization in _MODULE.POLARIZATIONS:
            shard_id = _MODULE.canonical_shard_id(pass_id, polarization)
            frequency_count = 2 + (
                (pass_id + _MODULE.POLARIZATIONS.index(polarization)) % 3
            )
            source_files = [
                f"raw/pass{pass_id}/{polarization}/"
                f"data_3dsar_pass{pass_id}_az{sector_id:03d}_{polarization.upper()}.mat"
                for sector_id in _MODULE.SECTOR_IDS
            ]
            file_records = []
            for sector_id in _MODULE.SECTOR_IDS:
                role = split.role_for(sector_id)
                record = {
                    "sector_id": sector_id,
                    "sector_role": role,
                    "payload_opened": role != "test",
                    "corrections_applied": False,
                }
                if role != "test":
                    record.update(
                        {
                            "frequency_count": frequency_count,
                            "pulse_count": 1,
                            "fp_shape": [frequency_count, 1],
                            "autofocus_present": polarization
                            in _MODULE.CO_POLARIZATIONS,
                        }
                    )
                file_records.append(record)
            payload_count = len(_MODULE.PAYLOAD_SECTOR_IDS)
            total_views += payload_count
            total_samples += payload_count * frequency_count
            official_af = polarization in _MODULE.CO_POLARIZATIONS
            autofocus = {
                "official_available": official_af,
                "present_in_all_payload_audited_source_files": official_af,
                "source_shard_id": shard_id if official_af else None,
                "source_polarization": polarization if official_af else None,
                "range_field": "af.r_correct" if official_af else None,
                "phase_field": "af.ph_correct" if official_af else None,
                "payload_audited_source_file_count": 324,
                "applied": False,
                "policy": "audited_in_native_files_never_applied_by_gate0",
            }
            if official_af:
                r_values = np.asarray(
                    [0.001 * value for value in _MODULE.PAYLOAD_SECTOR_IDS],
                    dtype=np.float32,
                )
                ph_values = np.asarray(
                    [-0.002 * value for value in _MODULE.PAYLOAD_SECTOR_IDS],
                    dtype=np.float32,
                )
                autofocus["r_correct_extrema"] = {
                    "minimum": float(np.min(r_values)),
                    "maximum": float(np.max(r_values)),
                }
                autofocus["ph_correct_extrema"] = {
                    "minimum": float(np.min(ph_values)),
                    "maximum": float(np.max(ph_values)),
                }
            shards.append(
                {
                    "shard_id": shard_id,
                    "pass_id": pass_id,
                    "polarization": polarization,
                    "sector_ids": list(_MODULE.SECTOR_IDS),
                    "sector_roles": list(split.role_by_sector),
                    "source_files": source_files,
                    "source_file_count": 360,
                    "inventoried_source_file_count": 360,
                    "payload_audited_sector_ids": list(
                        _MODULE.PAYLOAD_SECTOR_IDS
                    ),
                    "payload_audited_sector_count": 324,
                    "payload_audited_source_file_count": 324,
                    "sealed_test_sector_ids": list(
                        _MODULE.SEALED_TEST_SECTOR_IDS
                    ),
                    "sealed_test_sector_count": 36,
                    "sealed_test_source_file_count": 36,
                    "file_records": file_records,
                    "native_frequency_hz": np.linspace(
                        9.0e9, 9.2e9, frequency_count, dtype=np.float32
                    ).astype(float).tolist(),
                    "native_frequency_count": frequency_count,
                    "native_frequency_dtype": "float32",
                    "payload_audited_native_pulse_counts_by_sector": [1]
                    * payload_count,
                    "payload_audited_view_count": payload_count,
                    "payload_audited_complex_sample_count": payload_count
                    * frequency_count,
                    "fp_dtype": "complex64",
                    "geometry_dtype": "float32",
                    "corrections_applied": False,
                    "autofocus": autofocus,
                }
            )
    payload = {
        "schema": _MODULE.MANIFEST_SCHEMA,
        "inventory_schema": _MODULE.GATE0_INVENTORY_SCHEMA,
        "gate_id": _MODULE.GATE0_ID,
        "manager_track_id": _MODULE.MANAGER_TRACK_ID,
        "scene_id": _MODULE.SCENE_ID,
        "scene_count": 1,
        "dataset_root": _MODULE.FROZEN_DATASET_ROOT.as_posix(),
        "passes": list(_MODULE.PASS_IDS),
        "polarizations": list(_MODULE.POLARIZATIONS),
        "source_file_count": _MODULE.SOURCE_FILE_COUNT,
        "inventoried_source_file_count": _MODULE.SOURCE_FILE_COUNT,
        "payload_audited_source_file_count": 10_368,
        "sealed_test_source_file_count": 1_152,
        "payload_audited_sector_ids": list(_MODULE.PAYLOAD_SECTOR_IDS),
        "sealed_test_sector_ids": list(_MODULE.SEALED_TEST_SECTOR_IDS),
        "test_opened": False,
        "corrections_applied": False,
        "frequency_policy": "native_per_shard_no_trim_no_padding",
        "row_alignment_policy": "native_observations_no_cross_polarization_row_stacking",
        "split": split.as_dict(),
        "support": _MODULE.support_contract(),
        "shards": shards,
    }
    compatibility = _MODULE.validate_joint_manifest(payload)
    payload["totals"] = {
        "shards": 32,
        "inventoried_source_files": 11_520,
        "payload_audited_source_files": 10_368,
        "sealed_test_source_files": 1_152,
        "payload_audited_native_observations": total_views,
        "payload_audited_native_complex_samples": total_samples,
    }
    payload["contract_compatibility"] = {
        "validator": "rift.gotcha_joint_fullpol.validate_joint_manifest",
        "passed": True,
        "summary": compatibility,
    }
    return payload


def _synthetic_loader(path: Path, record: _MODULE.SourceRecord):
    if _MODULE.build_sector_split().role_for(record.sector_id) == "test":
        raise AssertionError("sealed test payload was opened")
    polarization_index = _MODULE.POLARIZATIONS.index(record.polarization)
    frequency_count = 2 + ((record.pass_id + polarization_index) % 3)
    frequency = np.linspace(9.0e9, 9.2e9, frequency_count, dtype=np.float32)
    value = float(1000 * record.pass_id + 10 * polarization_index + record.sector_id)
    response = np.full(
        (1, frequency_count), value + 1j * (value + 0.5), dtype=np.complex64
    )
    geometry = {
        "x": np.asarray([value], dtype=np.float32),
        "y": np.asarray([value + 1], dtype=np.float32),
        "z": np.asarray([value + 2], dtype=np.float32),
        "r0": np.asarray([value + 3], dtype=np.float32),
        "th": np.asarray([record.sector_id], dtype=np.float32),
        "phi": np.asarray([record.pass_id], dtype=np.float32),
    }
    if record.polarization in _MODULE.CO_POLARIZATIONS:
        r_correct = np.asarray([0.001 * record.sector_id], dtype=np.float32)
        ph_correct = np.asarray([-0.002 * record.sector_id], dtype=np.float32)
    else:
        r_correct = None
        ph_correct = None
    return _MODULE.NativeAcquisition(
        response=response,
        frequencies_hz=frequency,
        x=geometry["x"],
        y=geometry["y"],
        z=geometry["z"],
        r0=geometry["r0"],
        th=geometry["th"],
        phi=geometry["phi"],
        pulse_index=np.asarray([0], dtype=np.int32),
        r_correct_raw=r_correct,
        ph_correct_raw=ph_correct,
        source_response_shape=(frequency_count, 1),
        source_response_layout="frequency_view_transposed",
        pulse_index_source="derived_zero_based_within_sector",
    )


class GotchaJointFullPolGate1Tests(unittest.TestCase):
    def test_gate0_adapter_consumes_the_actual_shard_nested_schema(self):
        payload = _gate0_payload()
        records = _MODULE.adapt_gate0_inventory(payload)
        self.assertEqual(len(records), 11_520)
        self.assertEqual(records[0].shard_id, "pass1_hh")
        self.assertEqual(records[0].sector_id, 1)
        self.assertEqual(records[0].sector_role, "validation")
        self.assertTrue(records[0].payload_opened)
        self.assertEqual(records[0].expected_pulse_count, 1)
        self.assertEqual(records[0].expected_frequency_count, 3)
        self.assertTrue(records[0].source_path.endswith("pass1_az001_HH.mat"))
        self.assertEqual(records[-1].shard_id, "pass8_vv")
        self.assertEqual(records[-1].sector_id, 360)
        facts = _MODULE.adapt_gate0_shard_facts(payload)["pass1_hh"]
        self.assertEqual(facts.pulse_count_for(1), 1)
        self.assertEqual(facts.frequency_dtype, "float32")
        self.assertEqual(facts.response_dtype, "complex64")
        self.assertEqual(facts.geometry_dtype, "float32")
        self.assertEqual(facts.autofocus_dtype, "float32")

        duplicate = copy.deepcopy(payload)
        duplicate["shards"][0]["file_records"][1]["sector_id"] = 1
        with self.assertRaisesRegex(ValueError, "repeats"):
            _MODULE.adapt_gate0_inventory(duplicate)

    def test_materialization_writes_32_native_uncompressed_shards_atomically(self):
        payload = _gate0_payload()
        with _temporary_directory() as directory:
            output = Path(directory) / _MODULE.OUTPUT_ROOT_NAME
            manifest = _MODULE.materialize_inventory(
                payload,
                output,
                mat_loader=_synthetic_loader,
                require_slurm=False,
                gate0_inventory_path="gate0.json",
                enforce_production_paths=False,
            )
            self.assertTrue(output.is_dir())
            self.assertEqual(manifest["archive_count"], 32)
            self.assertEqual(len(manifest["archive_paths"]), 32)
            self.assertEqual(len(list((output / "shards").glob("*.npz"))), 32)
            self.assertTrue(manifest["one_archive_per_pass_polarization"])
            self.assertFalse(manifest["rectangularized_across_shards"])
            self.assertTrue(manifest["autofocus_unapplied"])
            self.assertFalse(manifest["test_opened"])
            self.assertFalse(manifest["corrections_applied"])
            self.assertEqual(manifest["inventoried_source_file_count"], 11_520)
            self.assertEqual(
                manifest["payload_audited_source_file_count"], 10_368
            )
            self.assertEqual(manifest["sealed_test_source_file_count"], 1_152)
            self.assertEqual(
                manifest["gate0_inventory_schema"],
                "rift_gotcha_joint_fullpol_raw_inventory_v1",
            )
            _MODULE.validate_joint_manifest(manifest)
            reloaded = json.loads(
                (output / _MODULE.MANIFEST_FILENAME).read_text(encoding="utf-8")
            )
            _MODULE.validate_joint_manifest(reloaded)

            hh_path = output / "shards" / "pass1_hh.npz"
            hv_path = output / "shards" / "pass1_hv.npz"
            with zipfile.ZipFile(hh_path, "r") as archive:
                self.assertTrue(
                    all(
                        item.compress_type == zipfile.ZIP_STORED
                        for item in archive.infolist()
                    )
                )
            with np.load(hh_path, allow_pickle=False) as hh:
                self.assertEqual(hh["response"].shape[0], 324)
                self.assertEqual(hh["r_correct_raw"].shape, (324,))
                self.assertEqual(hh["ph_correct_raw"].shape, (324,))
                self.assertFalse(bool(hh["autofocus_applied"]))
                self.assertEqual(str(hh["autofocus_state"]), "raw_channel_own_arrays_unapplied")
                self.assertEqual(str(hh["role"][0]), "validation")
                self.assertEqual(str(hh["role"][1]), "train")
                self.assertNotIn("test", set(hh["role"].tolist()))
                self.assertNotIn(6, set(hh["sector_id"].tolist()))
                self.assertTrue(np.all(hh["pass_id"] == 1))
                self.assertTrue(np.all(hh["polarization"] == "hh"))
            with np.load(hv_path, allow_pickle=False) as hv:
                self.assertEqual(hv["r_correct_raw"].size, 0)
                self.assertEqual(hv["ph_correct_raw"].size, 0)
                self.assertFalse(bool(hv["autofocus_available"]))
                self.assertEqual(str(hv["autofocus_state"]), "official_arrays_absent")

            with np.load(hh_path, allow_pickle=False) as loaded:
                tampered_arrays = {name: loaded[name] for name in loaded.files}
            tampered_arrays["sector_id"] = tampered_arrays["sector_id"].copy()
            tampered_arrays["role"] = tampered_arrays["role"].copy()
            tampered_arrays["sector_id"][0] = _MODULE.SEALED_TEST_SECTOR_IDS[0]
            tampered_arrays["role"][0] = "test"
            tampered_path = Path(directory) / "tampered.npz"
            _MODULE._write_uncompressed_npz(tampered_path, tampered_arrays)
            with self.assertRaisesRegex(ValueError, "payload sectors|sealed test"):
                _MODULE.validate_native_archive(tampered_path, 1, "hh")

            hh_frequency_count = manifest["native_layouts"]["pass1_hh"][
                "frequency_shape"
            ][0]
            vv_frequency_count = manifest["native_layouts"]["pass2_vv"][
                "frequency_shape"
            ][0]
            self.assertNotEqual(hh_frequency_count, vv_frequency_count)

            def forbidden_loader(path, record):
                raise AssertionError("idempotent materialization reopened a payload")

            idempotent = _MODULE.materialize_inventory(
                payload,
                output,
                mat_loader=forbidden_loader,
                require_slurm=False,
                gate0_inventory_path="gate0.json",
                enforce_production_paths=False,
            )
            self.assertEqual(idempotent, manifest)

    def test_failed_conversion_reuses_valid_shards_on_resume(self):
        payload = _gate0_payload()

        def failing_loader(path, record):
            if record.shard_id == "pass1_hv" and record.sector_id == 1:
                raise RuntimeError("synthetic loader failure")
            return _synthetic_loader(path, record)

        with _temporary_directory() as directory:
            parent = Path(directory)
            output = parent / _MODULE.OUTPUT_ROOT_NAME
            with self.assertRaisesRegex(RuntimeError, "synthetic loader failure"):
                _MODULE.materialize_inventory(
                    payload,
                    output,
                    mat_loader=failing_loader,
                    require_slurm=False,
                    gate0_inventory_path="gate0.json",
                    enforce_production_paths=False,
                )
            first_shard = output / "shards" / "pass1_hh.npz"
            self.assertTrue(first_shard.is_file())
            self.assertFalse((output / "shards" / "pass1_hv.npz").exists())
            self.assertEqual(list(output.rglob("*.partial")), [])

            with np.load(first_shard, allow_pickle=False) as loaded:
                original_arrays = {name: loaded[name] for name in loaded.files}
            changed_arrays = dict(original_arrays)
            changed_arrays["frequencies_hz"] = (
                changed_arrays["frequencies_hz"] + np.float32(8192.0)
            )
            rewrite = parent / "rewrite.npz"
            _MODULE._write_uncompressed_npz(rewrite, changed_arrays)
            os.replace(rewrite, first_shard)
            forbidden_calls = []

            def forbidden_resume_loader(path, record):
                forbidden_calls.append((record.shard_id, record.sector_id))
                raise AssertionError("resume opened data before validating its shard")

            with self.assertRaisesRegex(ValueError, "frequency values.*Gate 0"):
                _MODULE.materialize_inventory(
                    payload,
                    output,
                    mat_loader=forbidden_resume_loader,
                    require_slurm=False,
                    gate0_inventory_path="gate0.json",
                    enforce_production_paths=False,
                )
            self.assertEqual(forbidden_calls, [])
            restore = parent / "restore.npz"
            _MODULE._write_uncompressed_npz(restore, original_arrays)
            os.replace(restore, first_shard)

            stale_partial = output / "shards" / ".pass1_hv.npz.partial"
            stale_partial.write_bytes(b"interrupted")

            resumed_calls = []

            def resumed_loader(path, record):
                resumed_calls.append((record.shard_id, record.sector_id))
                return _synthetic_loader(path, record)

            manifest = _MODULE.materialize_inventory(
                payload,
                output,
                mat_loader=resumed_loader,
                require_slurm=False,
                gate0_inventory_path="gate0.json",
                enforce_production_paths=False,
            )
            self.assertEqual(manifest["archive_count"], 32)
            self.assertFalse(any(shard == "pass1_hh" for shard, _ in resumed_calls))
            self.assertTrue(resumed_calls)
            self.assertTrue(
                all(
                    sector_id not in _MODULE.SEALED_TEST_SECTOR_IDS
                    for _, sector_id in resumed_calls
                )
            )
            self.assertFalse(stale_partial.exists())

    def test_loaded_payloads_are_bound_to_every_gate0_scientific_fact(self):
        def drift_loader(kind):
            def load(path, record):
                acquisition = _synthetic_loader(path, record)
                target = "pass1_hh" if kind == "af_dtype" else "pass1_vv" if kind == "af_extrema" else "pass1_hv"
                if record.shard_id != target:
                    return acquisition
                if kind == "pulse" and record.sector_id == 1:
                    return replace(
                        acquisition,
                        response=np.repeat(acquisition.response, 2, axis=0),
                        x=np.repeat(acquisition.x, 2),
                        y=np.repeat(acquisition.y, 2),
                        z=np.repeat(acquisition.z, 2),
                        r0=np.repeat(acquisition.r0, 2),
                        th=np.repeat(acquisition.th, 2),
                        phi=np.repeat(acquisition.phi, 2),
                        pulse_index=np.asarray([0, 1], dtype=np.int32),
                        source_response_shape=(acquisition.frequency_count, 2),
                    )
                if kind == "frequency_values":
                    return replace(
                        acquisition,
                        frequencies_hz=acquisition.frequencies_hz
                        + np.float32(8192.0),
                    )
                if kind == "frequency_count":
                    frequency = np.append(
                        acquisition.frequencies_hz,
                        acquisition.frequencies_hz[-1] + np.float32(1.0e6),
                    ).astype(np.float32)
                    response = np.concatenate(
                        [acquisition.response, acquisition.response[:, -1:]],
                        axis=1,
                    )
                    return replace(
                        acquisition,
                        frequencies_hz=frequency,
                        response=response,
                        source_response_shape=(frequency.size, 1),
                    )
                if kind == "frequency_dtype":
                    return replace(
                        acquisition,
                        frequencies_hz=acquisition.frequencies_hz.astype(np.float64),
                    )
                if kind == "response_dtype":
                    return replace(
                        acquisition,
                        response=acquisition.response.astype(np.complex128),
                    )
                if kind == "geometry_dtype":
                    return replace(
                        acquisition, x=acquisition.x.astype(np.float64)
                    )
                if kind == "af_dtype":
                    return replace(
                        acquisition,
                        r_correct_raw=acquisition.r_correct_raw.astype(np.float64),
                        ph_correct_raw=acquisition.ph_correct_raw.astype(np.float64),
                    )
                if kind == "af_extrema" and record.sector_id == 1:
                    return replace(
                        acquisition,
                        ph_correct_raw=np.asarray([1.0], dtype=np.float32),
                    )
                return acquisition

            return load

        cases = (
            "pulse",
            "frequency_values",
            "frequency_count",
            "frequency_dtype",
            "response_dtype",
            "geometry_dtype",
            "af_dtype",
            "af_extrema",
        )
        with _temporary_directory() as directory:
            for index, kind in enumerate(cases):
                with self.subTest(kind=kind):
                    output_parent = Path(directory) / f"drift-{index}"
                    output_parent.mkdir()
                    output = output_parent / _MODULE.OUTPUT_ROOT_NAME
                    with self.assertRaises(ValueError):
                        _MODULE.materialize_inventory(
                            _gate0_payload(),
                            output,
                            mat_loader=drift_loader(kind),
                            require_slurm=False,
                            gate0_inventory_path="gate0.json",
                            enforce_production_paths=False,
                        )
                    self.assertFalse((output / "manifest.json").exists())

    def test_authentic_gate0_is_required_before_output_is_created(self):
        mutations = {
            "schema": lambda value: value.__setitem__("inventory_schema", "wrong"),
            "gate": lambda value: value.__setitem__("gate_id", "wrong"),
            "track": lambda value: value.__setitem__("manager_track_id", "wrong"),
            "scene_count": lambda value: value.__setitem__("scene_count", 2),
            "dataset_root": lambda value: value.__setitem__("dataset_root", "/wrong"),
            "counts": lambda value: value.__setitem__(
                "payload_audited_source_file_count", 10_367
            ),
            "membership": lambda value: value["shards"][0].__setitem__(
                "payload_audited_sector_ids",
                list(_MODULE.PAYLOAD_SECTOR_IDS[:-1]),
            ),
            "corrections": lambda value: value.__setitem__(
                "corrections_applied", True
            ),
        }
        with _temporary_directory() as directory:
            for index, (name, mutate) in enumerate(mutations.items()):
                with self.subTest(name=name):
                    payload = _gate0_payload()
                    mutate(payload)
                    output = (
                        Path(directory)
                        / f"case-{index}"
                        / _MODULE.OUTPUT_ROOT_NAME
                    )
                    output.parent.mkdir()
                    with self.assertRaises(ValueError):
                        _MODULE.materialize_inventory(
                            payload,
                            output,
                            mat_loader=_synthetic_loader,
                            require_slurm=False,
                            gate0_inventory_path="gate0.json",
                            enforce_production_paths=False,
                        )
                    self.assertFalse(output.exists())

    def test_full_entrypoint_requires_slurm_and_exact_output_root_name(self):
        with mock.patch.dict(
            os.environ, {"SLURM_JOB_ID": "", "SLURM_JOBID": ""}, clear=False
        ):
            with self.assertRaisesRegex(RuntimeError, "inside a Slurm allocation"):
                _MODULE._require_slurm_allocation()
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "9876"}, clear=False):
            self.assertEqual(_MODULE._require_slurm_allocation(), "9876")
        with _temporary_directory() as directory:
            with self.assertRaisesRegex(ValueError, _MODULE.OUTPUT_ROOT_NAME):
                _MODULE.materialize_inventory(
                    _gate0_payload(),
                    Path(directory) / "wrong_name",
                    mat_loader=_synthetic_loader,
                    require_slurm=False,
                    gate0_inventory_path="gate0.json",
                    enforce_production_paths=False,
                )
            with self.assertRaisesRegex(ValueError, "frozen Gate-1 root"):
                _MODULE.materialize_inventory(
                    _gate0_payload(),
                    Path(directory) / _MODULE.OUTPUT_ROOT_NAME,
                    mat_loader=_synthetic_loader,
                    require_slurm=False,
                    gate0_inventory_path=_MODULE.FROZEN_GATE0_JSON,
                    enforce_production_paths=True,
                )

    def test_nested_mat_loader_preserves_native_values_and_never_applies_af(self):
        frequency = np.asarray([9.0e9, 9.1e9, 9.2e9], dtype=np.float32)
        fp = np.asarray(
            [[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j], [9 + 10j, 11 + 12j]],
            dtype=np.complex64,
        )
        geometry = {
            "x": np.asarray([1.0, 2.0], dtype=np.float32),
            "y": np.asarray([3.0, 4.0], dtype=np.float32),
            "z": np.asarray([5.0, 6.0], dtype=np.float32),
            "r0": np.asarray([7.0, 8.0], dtype=np.float32),
            "th": np.asarray([9.0, 10.0], dtype=np.float32),
            "phi": np.asarray([11.0, 12.0], dtype=np.float32),
        }
        r_correct = np.asarray([0.1, 0.2], dtype=np.float32)
        ph_correct = np.asarray([-0.3, 0.4], dtype=np.float32)
        with _temporary_directory() as directory:
            hh_path = Path(directory) / "hh.mat"
            savemat(
                hh_path,
                {
                    "data": {
                        "fp": fp,
                        "freq": frequency,
                        **geometry,
                        "af": {
                            "r_correct": r_correct,
                            "ph_correct": ph_correct,
                        },
                    }
                },
                do_compression=False,
            )
            hh_record = _MODULE.SourceRecord(str(hh_path), 1, "hh", 1)
            acquisition = _MODULE.load_native_mat(hh_path, hh_record)
            np.testing.assert_array_equal(acquisition.response, fp.T)
            np.testing.assert_array_equal(acquisition.frequencies_hz, frequency)
            np.testing.assert_array_equal(acquisition.r_correct_raw, r_correct)
            np.testing.assert_array_equal(acquisition.ph_correct_raw, ph_correct)
            self.assertEqual(acquisition.source_response_layout, "frequency_view_transposed")
            self.assertEqual(
                acquisition.pulse_index_source,
                "derived_zero_based_within_sector",
            )

            hv_path = Path(directory) / "hv.mat"
            savemat(
                hv_path,
                {"data": {"fp": fp, "freq": frequency, **geometry}},
                do_compression=False,
            )
            hv_record = _MODULE.SourceRecord(str(hv_path), 1, "hv", 1)
            hv = _MODULE.load_native_mat(hv_path, hv_record)
            self.assertIsNone(hv.r_correct_raw)
            self.assertIsNone(hv.ph_correct_raw)

            borrowed_path = Path(directory) / "borrowed_hv.mat"
            savemat(
                borrowed_path,
                {
                    "data": {
                        "fp": fp,
                        "freq": frequency,
                        **geometry,
                        "af": {
                            "r_correct": r_correct,
                            "ph_correct": ph_correct,
                        },
                    }
                },
                do_compression=False,
            )
            borrowed_record = _MODULE.SourceRecord(
                str(borrowed_path), 1, "hv", 2
            )
            with self.assertRaisesRegex(ValueError, "must not receive"):
                _MODULE.load_native_mat(borrowed_path, borrowed_record)


if __name__ == "__main__":
    unittest.main()
