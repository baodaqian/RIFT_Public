from __future__ import annotations

import copy
from contextlib import contextmanager
import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import unittest
from unittest import mock

import numpy as np
from scipy.io import savemat


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _PROJECT_ROOT / "scripts" / "audit_gotcha_joint_fullpol_v1.py"
_SPEC = importlib.util.spec_from_file_location("audit_gotcha_joint_fullpol_v1", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
_TEMP_COUNTER = itertools.count()


@contextmanager
def _temporary_directory():
    root = _PROJECT_ROOT / ".codex-tmp"
    root.mkdir(parents=True, exist_ok=True)
    directory = root / f"gate0-test-{os.getpid()}-{next(_TEMP_COUNTER)}"
    directory.mkdir()
    try:
        yield str(directory)
    finally:
        shutil.rmtree(directory, ignore_errors=False)


def _source_path(root: Path, pass_id: int, polarization: str, sector_id: int) -> Path:
    directory = _MODULE.raw_shard_directory(root, pass_id, polarization)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / _MODULE.expected_source_name(pass_id, polarization, sector_id)


def _write_mat(
    path: Path,
    *,
    polarization: str,
    overrides: dict[str, object] | None = None,
    include_autofocus: bool | None = None,
) -> None:
    data: dict[str, object] = {
        "fp": np.asarray(
            [[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j], [9 + 10j, 11 + 12j]],
            dtype=np.complex64,
        ),
        "freq": np.asarray([9.0e9, 9.1e9, 9.2e9], dtype=np.float32),
        "x": np.asarray([1.0, 2.0], dtype=np.float32),
        "y": np.asarray([3.0, 4.0], dtype=np.float32),
        "z": np.asarray([5.0, 6.0], dtype=np.float32),
        "r0": np.asarray([7.0, 8.0], dtype=np.float32),
        "th": np.asarray([9.0, 10.0], dtype=np.float32),
        "phi": np.asarray([11.0, 12.0], dtype=np.float32),
    }
    if include_autofocus is None:
        include_autofocus = polarization.lower() in _MODULE.CO_POLARIZATIONS
    if include_autofocus:
        data["af"] = {
            "r_correct": np.asarray([0.1, 0.2], dtype=np.float32),
            "ph_correct": np.asarray([-0.3, 0.4], dtype=np.float32),
        }
    if overrides:
        data.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, {"data": data}, do_compression=False)


def _fake_contract_shards() -> list[dict[str, object]]:
    split = _MODULE.build_sector_split()
    shards: list[dict[str, object]] = []
    for pass_id in _MODULE.PASS_IDS:
        for polarization in _MODULE.POLARIZATIONS:
            shard_id = _MODULE.canonical_shard_id(pass_id, polarization)
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
                            "frequency_count": 3,
                            "pulse_count": 2,
                            "fp_shape": [3, 2],
                            "autofocus_present": polarization
                            in _MODULE.CO_POLARIZATIONS,
                        }
                    )
                file_records.append(record)
            shards.append(
                {
                    "shard_id": shard_id,
                    "pass_id": pass_id,
                    "polarization": polarization,
                    "sector_ids": list(_MODULE.SECTOR_IDS),
                    "sector_roles": list(split.role_by_sector),
                    "source_files": [
                        f"/raw/{shard_id}/{sector_id:03d}.mat"
                        for sector_id in _MODULE.SECTOR_IDS
                    ],
                    "source_file_count": 360,
                    "inventoried_source_file_count": 360,
                    "payload_audited_sector_ids": list(
                        _MODULE.PAYLOAD_AUDITED_SECTOR_IDS
                    ),
                    "payload_audited_sector_count": 324,
                    "payload_audited_source_file_count": 324,
                    "sealed_test_sector_ids": list(_MODULE.SEALED_TEST_SECTOR_IDS),
                    "sealed_test_sector_count": 36,
                    "sealed_test_source_file_count": 36,
                    "native_frequency_hz": [9.0e9, 9.1e9, 9.2e9],
                    "native_frequency_count": 3,
                    "payload_audited_native_pulse_counts_by_sector": [2] * 324,
                    "payload_audited_view_count": 648,
                    "payload_audited_complex_sample_count": 1944,
                    "file_records": file_records,
                    "autofocus": {
                        "official_available": polarization
                        in _MODULE.CO_POLARIZATIONS,
                        "applied": False,
                    },
                    "corrections_applied": False,
                }
            )
    return shards


class GotchaJointFullPolGate0Tests(unittest.TestCase):
    def test_disc_and_source_mapping_are_exact(self):
        root = Path("/dataset")
        self.assertEqual(_MODULE.disc_name_for_pass(1), "GOTCHA-CP_Disc1")
        self.assertEqual(_MODULE.disc_name_for_pass(7), "GOTCHA-CP_Disc1")
        self.assertEqual(_MODULE.disc_name_for_pass(8), "GOTCHA-CP_Disc2")
        self.assertEqual(
            _MODULE.raw_shard_directory(root, 1, "hh").as_posix(),
            "/dataset/extracted/data/GOTCHA/GOTCHA-CP_Disc1/DATA/pass1/HH",
        )
        self.assertEqual(
            _MODULE.raw_shard_directory(root, 8, "vv").as_posix(),
            "/dataset/extracted/data/GOTCHA/GOTCHA-CP_Disc2/DATA/pass8/VV",
        )
        self.assertEqual(
            _MODULE.expected_source_name(8, "vh", 360),
            "data_3dsar_pass8_az360_VH.mat",
        )

    def test_discovery_requires_the_exact_expected_file_set(self):
        with _temporary_directory() as directory:
            root = Path(directory)
            first = _source_path(root, 1, "hh", 1)
            second = _source_path(root, 1, "hh", 2)
            first.write_bytes(b"fixture")
            second.write_bytes(b"fixture")
            found = _MODULE.discover_shard_files(
                root, 1, "hh", expected_sector_ids=(1, 2)
            )
            self.assertEqual(found, (first, second))

            extra = first.parent / "unexpected.mat"
            extra.write_bytes(b"fixture")
            with self.assertRaisesRegex(_MODULE.AuditError, "expected exactly 2"):
                _MODULE.discover_shard_files(
                    root, 1, "hh", expected_sector_ids=(1, 2)
                )

    def test_shard_inventory_never_calls_loader_for_sealed_test_sectors(self):
        calls: list[int] = []

        def fake_auditor(path, *, pass_id, polarization, sector_id):
            calls.append(sector_id)
            return {
                "source_file": Path(path).resolve(strict=True).as_posix(),
                "sector_id": sector_id,
                "frequency_hz": np.asarray(
                    [9.0e9, 9.1e9, 9.2e9], dtype=np.float32
                ),
                "frequency_count": 3,
                "pulse_count": 2,
                "fp_shape": [3, 2],
                "autofocus_present": True,
                "autofocus_stats": {
                    "r_correct": {"minimum": 0.0, "maximum": 0.1},
                    "ph_correct": {"minimum": -0.2, "maximum": 0.2},
                },
                "payload_opened": True,
                "corrections_applied": False,
            }

        with _temporary_directory() as directory:
            root = Path(directory)
            for sector_id in _MODULE.SECTOR_IDS:
                _source_path(root, 1, "hh", sector_id).write_bytes(b"inventory-only")
            shard = _MODULE.audit_shard(
                root, 1, "hh", file_auditor=fake_auditor
            )

        self.assertEqual(calls, list(_MODULE.PAYLOAD_AUDITED_SECTOR_IDS))
        self.assertTrue(set(calls).isdisjoint(_MODULE.SEALED_TEST_SECTOR_IDS))
        self.assertEqual(shard["inventoried_source_file_count"], 360)
        self.assertEqual(shard["payload_audited_source_file_count"], 324)
        self.assertEqual(shard["sealed_test_source_file_count"], 36)
        for record in shard["file_records"]:
            if record["sector_id"] in _MODULE.SEALED_TEST_SECTOR_IDS:
                self.assertEqual(
                    set(record), _MODULE._SEALED_TEST_FILE_RECORD_KEYS
                )
                self.assertFalse(record["payload_opened"])
            else:
                self.assertTrue(record["payload_opened"])

    def test_direct_test_sector_audit_rejects_before_mat_loader(self):
        with _temporary_directory() as directory:
            path = _source_path(Path(directory), 1, "hh", 6)
            path.write_bytes(b"must-not-open")
            with mock.patch.object(_MODULE, "_load_nested_data") as loader:
                with self.assertRaisesRegex(_MODULE.AuditError, "must not be opened"):
                    _MODULE.audit_mat_file(
                        path, pass_id=1, polarization="hh", sector_id=6
                    )
                loader.assert_not_called()

    def test_copol_mat_schema_and_own_autofocus_pass_without_application(self):
        with _temporary_directory() as directory:
            root = Path(directory)
            path = _source_path(root, 1, "hh", 1)
            _write_mat(path, polarization="hh")
            audit = _MODULE.audit_mat_file(
                path, pass_id=1, polarization="hh", sector_id=1
            )

        self.assertEqual(audit["fp_shape"], [3, 2])
        self.assertEqual(audit["fp_dtype"], "complex64")
        self.assertEqual(audit["frequency_dtype"], "float32")
        self.assertEqual(audit["geometry_dtype"], "float32")
        self.assertEqual(audit["pulse_count"], 2)
        self.assertTrue(audit["autofocus_present"])
        self.assertFalse(audit["corrections_applied"])
        self.assertAlmostEqual(
            audit["autofocus_stats"]["r_correct"]["maximum"], 0.2, places=6
        )

    def test_autofocus_presence_is_polarization_specific(self):
        with _temporary_directory() as directory:
            root = Path(directory)
            missing_hh = _source_path(root, 1, "hh", 1)
            _write_mat(
                missing_hh, polarization="hh", include_autofocus=False
            )
            with self.assertRaisesRegex(_MODULE.AuditError, "must contain its own af"):
                _MODULE.audit_mat_file(
                    missing_hh, pass_id=1, polarization="hh", sector_id=1
                )

            borrowed_hv = _source_path(root, 1, "hv", 1)
            _write_mat(
                borrowed_hv, polarization="hv", include_autofocus=True
            )
            with self.assertRaisesRegex(_MODULE.AuditError, "must not contain"):
                _MODULE.audit_mat_file(
                    borrowed_hv, pass_id=1, polarization="hv", sector_id=1
                )

            native_hv = _source_path(root, 1, "hv", 2)
            _write_mat(native_hv, polarization="hv", include_autofocus=False)
            audit = _MODULE.audit_mat_file(
                native_hv, pass_id=1, polarization="hv", sector_id=2
            )
            self.assertFalse(audit["autofocus_present"])
            self.assertFalse(audit["corrections_applied"])

    def test_mat_schema_rejects_wrong_dtype_shape_and_nonfinite_values(self):
        cases = (
            ("dtype", {"x": np.asarray([1.0, 2.0], dtype=np.float64)}, "x dtype"),
            (
                "shape",
                {"fp": np.ones((2, 3), dtype=np.complex64)},
                "fp shape",
            ),
            (
                "nonfinite",
                {
                    "fp": np.asarray(
                        [[np.nan + 0j, 1j], [2j, 3j], [4j, 5j]],
                        dtype=np.complex64,
                    )
                },
                "contains nonfinite",
            ),
        )
        with _temporary_directory() as directory:
            root = Path(directory)
            for sector_id, (label, overrides, message) in enumerate(cases, start=1):
                with self.subTest(label=label):
                    path = _source_path(root, 1, "hh", sector_id)
                    _write_mat(path, polarization="hh", overrides=overrides)
                    with self.assertRaisesRegex(_MODULE.AuditError, message):
                        _MODULE.audit_mat_file(
                            path,
                            pass_id=1,
                            polarization="hh",
                            sector_id=sector_id,
                        )

    def test_manifest_is_compatible_and_preserves_gate0_state(self):
        shards = _fake_contract_shards()
        manifest = _MODULE.build_joint_inventory_manifest("/dataset", shards)
        summary = _MODULE.validate_gate0_inventory_manifest(manifest)
        self.assertEqual(summary["shard_count"], 32)
        self.assertEqual(summary["source_file_count"], 11520)
        self.assertEqual(summary["inventoried_source_file_count"], 11520)
        self.assertEqual(summary["payload_audited_source_file_count"], 10368)
        self.assertEqual(summary["sealed_test_source_file_count"], 1152)
        self.assertEqual(manifest["inventory_schema"], _MODULE.INVENTORY_SCHEMA)
        self.assertEqual(manifest["manager_track_id"], _MODULE.MANAGER_TRACK_ID)
        self.assertEqual(manifest["scene_count"], 1)
        self.assertFalse(manifest["test_opened"])
        self.assertFalse(manifest["corrections_applied"])
        self.assertEqual(
            manifest["totals"]["payload_audited_native_observations"],
            32 * 648,
        )

        duplicate = copy.deepcopy(shards)
        duplicate[1]["source_files"][0] = duplicate[0]["source_files"][0]
        with self.assertRaisesRegex(ValueError, "source file is reused"):
            _MODULE.build_joint_inventory_manifest("/dataset", duplicate)

    def test_manifest_rejects_adversarial_sealed_counts_and_provenance(self):
        manifest = _MODULE.build_joint_inventory_manifest(
            "/dataset", _fake_contract_shards()
        )
        mutations = []

        wrong_top_count = copy.deepcopy(manifest)
        wrong_top_count["payload_audited_source_file_count"] = 10367
        mutations.append((wrong_top_count, "payload_audited_source_file_count"))

        wrong_sector_membership = copy.deepcopy(manifest)
        wrong_sector_membership["shards"][0]["payload_audited_sector_ids"][0] = 6
        mutations.append((wrong_sector_membership, "payload_audited_sector_ids"))

        opened_test = copy.deepcopy(manifest)
        opened_test["shards"][0]["file_records"][5]["payload_opened"] = True
        mutations.append((opened_test, "must remain unopened"))

        leaked_test_value = copy.deepcopy(manifest)
        leaked_test_value["shards"][0]["file_records"][5]["fp_shape"] = [3, 2]
        mutations.append((leaked_test_value, "payload-derived fields"))

        unopened_training = copy.deepcopy(manifest)
        unopened_training["shards"][0]["file_records"][1][
            "payload_opened"
        ] = False
        mutations.append((unopened_training, "was not audited"))

        opened_top_level = copy.deepcopy(manifest)
        opened_top_level["test_opened"] = True
        mutations.append((opened_top_level, "test_opened"))

        for candidate, message in mutations:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    _MODULE.validate_gate0_inventory_manifest(candidate)

    def test_full_scan_defaults_to_slurm_only(self):
        with self.assertRaisesRegex(RuntimeError, "SLURM_JOB_ID is absent"):
            _MODULE.audit_dataset(
                "/path/need/not/exist", allocation_environ={}
            )
        self.assertEqual(
            _MODULE.require_slurm_allocation({"SLURM_JOB_ID": "12345"}), "12345"
        )

    def test_cli_paths_are_frozen_to_production_locations(self):
        dataset_root, output = _MODULE.validate_production_cli_paths(
            _MODULE.DEFAULT_DATASET_ROOT,
            _MODULE.PRODUCTION_OUTPUT_PATH,
        )
        self.assertEqual(dataset_root, _MODULE.DEFAULT_DATASET_ROOT.resolve())
        self.assertEqual(output, _MODULE.PRODUCTION_OUTPUT_PATH.resolve())
        with self.assertRaisesRegex(_MODULE.AuditError, "dataset root must be"):
            _MODULE.validate_production_cli_paths(
                "/wrong/dataset", _MODULE.PRODUCTION_OUTPUT_PATH
            )
        with self.assertRaisesRegex(_MODULE.AuditError, "output path must be"):
            _MODULE.validate_production_cli_paths(
                _MODULE.DEFAULT_DATASET_ROOT, "/wrong/output.json"
            )

    def test_atomic_json_is_deterministic_and_refuses_overwrite(self):
        payload = {"z": 1, "a": [2, 3]}
        with _temporary_directory() as directory:
            output = Path(directory) / "inventory.json"
            real_link = os.link
            with mock.patch.object(
                _MODULE.os, "replace", side_effect=AssertionError("replace is forbidden")
            ), mock.patch.object(_MODULE.os, "link", wraps=real_link) as link:
                _MODULE.atomic_write_json(output, payload)
            link.assert_called_once()
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), payload)
            self.assertTrue(output.read_text(encoding="utf-8").startswith('{\n  "a"'))
            self.assertEqual(list(output.parent.iterdir()), [output])
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                _MODULE.atomic_write_json(output, payload)

    def test_atomic_json_loses_a_publish_race_without_overwriting(self):
        payload = {"writer": "gate0"}
        with _temporary_directory() as directory:
            output = Path(directory) / "inventory.json"

            def concurrent_winner(source, destination):
                Path(destination).write_text(
                    '{"writer":"other"}\n', encoding="utf-8"
                )
                raise FileExistsError(destination)

            with mock.patch.object(_MODULE.os, "link", side_effect=concurrent_winner):
                with self.assertRaisesRegex(FileExistsError, "concurrently created"):
                    _MODULE.atomic_write_json(output, payload)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")), {"writer": "other"}
            )
            self.assertEqual(list(output.parent.iterdir()), [output])


if __name__ == "__main__":
    unittest.main()
