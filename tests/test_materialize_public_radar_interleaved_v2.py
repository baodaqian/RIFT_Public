import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "materialize_public_radar_interleaved_v2.py"
)
_SPEC = importlib.util.spec_from_file_location("materialize_public_radar_interleaved_v2", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class MaterializeInterleavedV2Tests(unittest.TestCase):
    def test_camry_materialization_changes_only_split_contract(self):
        groups = np.repeat(np.arange(720, dtype=np.int64), 32)
        within = np.tile(np.arange(32), 720)
        azimuth = (groups * 0.5 + within * (0.5 / 32.0)) % 360.0
        elevation = np.take(np.array([30.0, 40.0, 50.0, 60.0]), within % 4)
        radians = np.deg2rad(azimuth)
        positions = np.column_stack(
            (np.cos(radians), np.sin(radians), np.sin(np.deg2rad(elevation)))
        ).astype(np.float32)
        response = np.arange(groups.size, dtype=np.float32).astype(np.complex64)
        response = response.reshape(-1, 1, 1, 1, 1)
        metadata = {
            "schema": "rift_coherent_radar_v1",
            "split_strategy": "old_contiguous",
            "dataset": "cvdomes",
            "polarization": "hh",
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cvdomes_camry" / "cvdomes_camry.npz"
            source.parent.mkdir()
            output = root / "v2" / "camry.npz"
            np.savez(
                source,
                response=response,
                viewpoint_positions=positions,
                tx_pos=positions[:, None, :],
                rx_pos=positions[:, None, :],
                frequencies_hz=np.array([9.6e9]),
                view_azimuth_deg=azimuth,
                view_elevation_deg=elevation,
                split_group_id=groups,
                train_indices=np.arange(20736, dtype=np.int64),
                validation_indices=np.arange(20736, 23040, dtype=np.int64),
                test_indices=np.empty(0, dtype=np.int64),
                metadata_json=np.asarray(json.dumps(metadata)),
            )
            source_bytes = source.read_bytes()
            summary = _MODULE.materialize(
                "cvdomes_camry", source, output, seed=42
            )
            replay = _MODULE.materialize(
                "cvdomes_camry", source, output, seed=42
            )
            self.assertEqual(summary, replay)
            self.assertEqual(source.read_bytes(), source_bytes)
            self.assertTrue(output.with_suffix(".viewpoints.csv").is_file())
            self.assertTrue(output.with_suffix(".split.json").is_file())
            with np.load(source, allow_pickle=True) as parent, np.load(
                output, allow_pickle=True
            ) as child:
                for key in parent.files:
                    if key in (
                        "train_indices",
                        "validation_indices",
                        "test_indices",
                        "metadata_json",
                    ):
                        continue
                    np.testing.assert_array_equal(parent[key], child[key])
                self.assertEqual(child["train_indices"].size, 18432)
                self.assertEqual(child["validation_indices"].size, 2304)
                self.assertEqual(child["test_indices"].size, 2304)
                child_meta = json.loads(str(child["metadata_json"]))
                self.assertTrue(child_meta["split_interpolation_only"])
                self.assertEqual(child_meta["split_role_slots"]["validation"], 0)
                self.assertEqual(child_meta["split_role_slots"]["test"], 5)

            table = output.with_suffix(".viewpoints.csv")
            table.write_text("truncated\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "viewpoint table"):
                _MODULE.materialize("cvdomes_camry", source, output, seed=42)
            with self.assertRaisesRegex(ValueError, "parent metadata"):
                _MODULE.materialize("gotcha_pass2_hh", source, root / "wrong.npz", seed=42)


if __name__ == "__main__":
    unittest.main()
