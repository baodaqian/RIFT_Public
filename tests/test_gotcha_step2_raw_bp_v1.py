"""Data-free CLI tests for the reviewed GOTCHA P1 raw-BP profile driver."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
import uuid

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = ROOT / "scripts" / "run_gotcha_step2_raw_bp_v1.py"
PROTOCOL_PATH = ROOT / "protocols" / "gotcha_step2_p1_hh_sector002_h0_raw_bp_v1.json"


def _load_driver():
    spec = importlib.util.spec_from_file_location("gotcha_step2_raw_bp_driver_tests", DRIVER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load P1 driver")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DRIVER = _load_driver()


def _role_for_sector(sector_id: int) -> str:
    slot = (int(sector_id) - 1) % 10
    return "validation" if slot == 0 else "test" if slot == 5 else "train"


def _make_synthetic_archive(
    path: Path,
    *,
    scene_id: str = "gotcha_v1_joint8_fullpol",
    object_response: bool = False,
) -> None:
    sectors = [sector for sector in range(1, 361) if _role_for_sector(sector) != "test"]
    count = len(sectors)
    frequencies = np.asarray([9_600_000_000.0, 9_601_000_000.0, 9_603_000_000.0, 9_606_000_000.0], dtype=np.float32)
    row = np.arange(count, dtype=np.float32)
    response = (row[:, None] + 1.0).astype(np.complex64) * np.exp(
        1j * np.asarray([0.0, 0.2, -0.4, 0.7], dtype=np.float32)[None, :]
    )
    if object_response:
        response_object = np.empty(response.shape, dtype=object)
        response_object[:] = response
        response = response_object
    metadata = {
        "schema": "rift_gotcha_joint8_fullpol_native_shard_v1",
        "scene_id": scene_id,
        "shard_id": "pass1_hh",
        "pass_id": 1,
        "polarization": "hh",
        "payload_sector_ids": sectors,
        "sealed_test_sector_ids": [sector for sector in range(1, 361) if _role_for_sector(sector) == "test"],
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
    arrays = {
        "response": response,
        "frequencies_hz": frequencies,
        "x": (100.0 + row).astype(np.float32),
        "y": (-20.0 + 0.5 * row).astype(np.float32),
        "z": (1.0 + 0.01 * row).astype(np.float32),
        "r0": (120.0 + 0.1 * row).astype(np.float32),
        "th": np.linspace(-2.0, 2.0, count, dtype=np.float32),
        "phi": np.linspace(-1.0, 1.0, count, dtype=np.float32),
        "sector_id": np.asarray(sectors, dtype=np.int16),
        "pulse_index": np.asarray(sectors, dtype=np.int32) * 100 + 7,
        "pass_id": np.full(count, 1, dtype=np.int16),
        "polarization": np.full(count, "hh", dtype="U2"),
        "role": np.asarray([_role_for_sector(value) for value in sectors], dtype="U10"),
        "r_correct_raw": np.linspace(0.01, 0.02, count, dtype=np.float32),
        "ph_correct_raw": np.linspace(-0.2, 0.2, count, dtype=np.float32),
        "autofocus_available": np.asarray(True),
        "autofocus_applied": np.asarray(False),
        "autofocus_state": np.asarray("raw_channel_own_arrays_unapplied", dtype="U36"),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True), dtype="U"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez(handle, **arrays)


class GotchaStep2RawBpV1Test(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ROOT / "batch_a_tmp_test" / f".gotcha-step2-p1-test-{os.getpid()}-{uuid.uuid4().hex}"
        self.root.mkdir()
        (self.root / "shards").mkdir()
        _make_synthetic_archive(self.root / "shards" / "pass1_hh.npz")

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _run(self, output_dir: Path, protocol: Path = PROTOCOL_PATH, stage: str = "profile"):
        return subprocess.run(
            [
                sys.executable,
                str(DRIVER_PATH),
                "--protocol",
                str(protocol),
                "--archive-root",
                str(self.root),
                "--output-dir",
                str(output_dir),
                "--stage",
                stage,
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )

    def test_actual_cli_profile_outputs_and_disclosure(self):
        output = self.root / "out"
        result = self._run(output)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        expected_files = {
            "protocol_echo.json",
            "preflight.json",
            "resource.json",
            "bp_report.json",
            "status.json",
            "bp_complex.npy",
            "x.npy",
            "y.npy",
            "bp_native_xy_z0.png",
        }
        self.assertEqual({path.name for path in output.iterdir()}, expected_files)
        status = json.loads((output / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "PASS")
        preflight = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
        disclosure = preflight["disclosure"]
        self.assertEqual(disclosure["loaded_response_roles"], ["train", "validation"])
        self.assertEqual(disclosure["used_response_roles"], ["train"])
        self.assertEqual(disclosure["selected_response_count"], 1)
        self.assertFalse(disclosure["test_payload_opened"])
        self.assertFalse(disclosure["validation_used_for_bp"])
        self.assertEqual(disclosure["selected_ids"][0]["sector_id"], 2)
        report = json.loads((output / "bp_report.json").read_text(encoding="utf-8"))
        self.assertIn("P1 HH sector2", report["title"])
        self.assertIn("total native samples=4", report["title"])
        self.assertIn("F/observation=4-4", report["title"])
        self.assertIn("raw/unapplied AF", report["title"])
        self.assertIn("registration unresolved", report["title"])
        self.assertEqual(report["backprojection_metadata"]["point_chunk_size"], 4096)
        self.assertEqual(report["backprojection_metadata"]["normalization"], "mean_native_sample")
        self.assertEqual(report["plot"]["transpose_for_display"], True)
        self.assertEqual(report["plot"]["origin"], "lower")
        self.assertEqual(report["plot"]["db_clip"], [-40.0, 0.0])
        self.assertTrue(any("20log10(|BP|/A_ref)" in line for line in report["plot"]["header_lines"]))
        self.assertTrue(any("phase: exp(-i 4*pi*f/c (R-r0))" in line for line in report["plot"]["header_lines"]))
        self.assertTrue(any("native x=[-8,8] m, y=[-8,8] m" in line for line in report["plot"]["header_lines"]))
        self.assertEqual(report["plot"]["native_x_ticks_m"], [-8.0, 0.0, 8.0])
        self.assertEqual(report["plot"]["native_y_ticks_m"], [-8.0, 0.0, 8.0])
        resource = json.loads((output / "resource.json").read_text(encoding="utf-8"))
        self.assertEqual(resource["point_chunk_size"], 4096)
        self.assertTrue(resource["guard_passed_before_bp"])
        bp = np.load(output / "bp_complex.npy")
        self.assertEqual(bp.shape, (65 * 65,))
        self.assertEqual(np.load(output / "x.npy").shape, (65,))
        self.assertEqual(np.load(output / "y.npy").shape, (65,))
        from PIL import Image

        with Image.open(output / "bp_native_xy_z0.png") as image:
            self.assertGreater(image.width, 65)
            self.assertGreater(image.height, 65)

    def test_fresh_output_and_stage_selection_are_fail_closed(self):
        output = self.root / "out"
        self.assertEqual(self._run(output).returncode, 0)
        second = self._run(output)
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("fresh", second.stderr)
        bad_stage = self._run(self.root / "bad-stage", stage="survey")
        self.assertNotEqual(bad_stage.returncode, 0)
        self.assertIn("CLI stage", bad_stage.stderr)

    def test_protocol_role_and_sector_failures_are_rejected(self):
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        for label, mutation, expected_text in (
            ("bad-role", lambda value: value["selection"].__setitem__("role", "validation"), "selection.role"),
            ("bad-sector", lambda value: value["selection"].__setitem__("sector_id", 3), "selection.sector_id"),
        ):
            mutated = copy.deepcopy(protocol)
            mutation(mutated)
            path = self.root / f"{label}.json"
            path.write_text(json.dumps(mutated), encoding="utf-8")
            result = self._run(self.root / label, protocol=path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(expected_text, result.stderr)

    def test_scene_id_and_output_contract_tampering_are_rejected(self):
        _make_synthetic_archive(
            self.root / "shards" / "pass1_hh.npz",
            scene_id="tampered_scene",
            object_response=True,
        )
        with self.assertRaisesRegex(ValueError, "metadata scene_id"):
            DRIVER.ACQ.load_native_shard(
                self.root / "shards" / "pass1_hh.npz",
                expected_pass_id=1,
                expected_polarization="hh",
                expected_scene_id="gotcha_v1_fullpol",
            )
        scene_result = self._run(self.root / "bad-scene")
        self.assertNotEqual(scene_result.returncode, 0)
        self.assertIn("scene_id", scene_result.stderr)

        _make_synthetic_archive(self.root / "shards" / "pass1_hh.npz")
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        mutations = (
            ("bad-output-name", lambda value: value["outputs"].__setitem__("plot", "other.png"), "outputs"),
            ("bad-display-reference", lambda value: value["outputs"]["plot_display"].__setitem__("A_ref", "mean magnitude"), "outputs"),
            ("bad-display-clip", lambda value: value["outputs"]["plot_display"].__setitem__("clip_db", [-30.0, 0.0]), "outputs"),
        )
        for label, mutation, expected_text in mutations:
            mutated = copy.deepcopy(protocol)
            mutation(mutated)
            path = self.root / f"{label}.json"
            path.write_text(json.dumps(mutated), encoding="utf-8")
            result = self._run(self.root / label, protocol=path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(expected_text, result.stderr)

    def test_support_guard_and_chunk_plumbing_before_bp(self):
        support = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))["support"]
        bad_support = copy.deepcopy(support)
        bad_support["max_kernel_evaluations"] = 10
        declaration = dict(bad_support)
        observations = ()
        # The actual preflight guard is exercised with a small valid synthetic
        # observation set through the driver module's loaded controls.
        shard = DRIVER.ACQ.load_native_shard(self.root / "shards" / "pass1_hh.npz", expected_pass_id=1, expected_polarization="hh")
        ids = tuple(identity for identity in shard.observation_ids if int(identity.sector_id) == 2)
        observations = shard.observations(ids)
        with self.assertRaises(RuntimeError):
            DRIVER.CTL.preflight_support_declaration(declaration, observations, point_count=65 * 65)
        self.assertEqual(observations[0].role, "train")
        self.assertEqual(observations[0].identity.sector_id, 2)


if __name__ == "__main__":
    unittest.main()
