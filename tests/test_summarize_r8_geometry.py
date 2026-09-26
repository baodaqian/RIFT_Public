import unittest

from scripts.summarize_r8_geometry import (
    EXPECTED_RUNS,
    EXPECTED_THRESHOLDS,
    summarize_rows,
    validate_native_rows,
)


FIELDS = (
    "run", "epoch", "variant", "thresh", "n_points", "cd_surface", "cd_volume",
    "l2_mm", "hausdorff_mm", "hd95_mm", "iou_solid", "iou_shell",
    "precision", "recall", "f1",
)


def row(run, variant, threshold):
    epoch = 149 if run == "rift_r6_learned_dc" else 150
    return {
        "run": run,
        "epoch": epoch,
        "variant": variant,
        "thresh": threshold,
        "n_points": 2000 if variant == "voxel" else 20000,
        "cd_surface": 1.0e-4 + threshold * 1.0e-6,
        "cd_volume": 2.0e-4 + threshold * 1.0e-6,
        "l2_mm": 5.0 + threshold,
        "hausdorff_mm": 20.0 + threshold,
        "hd95_mm": 10.0 + threshold,
        "iou_solid": 0.1 + threshold * 0.01,
        "iou_shell": 0.2 + threshold * 0.01,
        "precision": 0.3,
        "recall": 0.4,
        "f1": 0.34 + threshold * 0.01,
    }


class GeometrySummaryTest(unittest.TestCase):
    def test_native_and_trilinear_protocols_validate(self):
        native_rows = [
            row(run, "voxel", threshold)
            for run in EXPECTED_RUNS for threshold in EXPECTED_THRESHOLDS
        ]
        tri_rows = [
            row(run, variant, threshold)
            for variant in ("voxel", "mesh")
            for run in EXPECTED_RUNS for threshold in EXPECTED_THRESHOLDS
        ]
        reference = [
            item for item in native_rows
            if item["run"] in {"rift_r6_learned_dc", "rift_r7_target20k_shdeg1em9"}
        ]

        self.assertEqual(set(summarize_rows(native_rows, "native_grid")), {"voxel"})
        self.assertEqual(set(summarize_rows(tri_rows, "trilinear_4x")), {"voxel", "mesh"})
        gate = validate_native_rows(native_rows, reference, "synthetic-reference.csv")
        self.assertEqual(gate["status"], "pass")
        self.assertEqual(gate["rows_compared"], 26)


if __name__ == "__main__":
    unittest.main()
