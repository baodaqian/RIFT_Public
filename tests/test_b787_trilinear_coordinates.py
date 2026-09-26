import unittest
import sys
import types

import numpy as np

# The coordinate helper is dependency-free; keep this local unit runnable in
# the lightweight Codex Python even when plotting/Torch packages are absent.
if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")
if "matplotlib" not in sys.modules:
    matplotlib = types.ModuleType("matplotlib")
    matplotlib.use = lambda *_args, **_kwargs: None
    sys.modules["matplotlib"] = matplotlib
    sys.modules["matplotlib.pyplot"] = types.ModuleType("matplotlib.pyplot")

from scripts.render_b787_vs_stl import sample_indices_to_physical, trilinear_sample_centers


class TrilinearSampleCenterTest(unittest.TestCase):
    def test_native_centers_cover_the_voxel_center_lattice(self):
        centers = trilinear_sample_centers(0.15, 48)
        self.assertEqual(centers.shape, (48,))
        self.assertAlmostEqual(centers[0], -0.146875)
        self.assertAlmostEqual(centers[-1], 0.146875)

    def test_dense_centers_preserve_align_corners_endpoints(self):
        native = trilinear_sample_centers(0.15, 48)
        dense = trilinear_sample_centers(0.15, 48, 192)
        self.assertEqual(dense.shape, (192,))
        self.assertAlmostEqual(dense[0], native[0])
        self.assertAlmostEqual(dense[-1], native[-1])
        self.assertTrue(np.all(np.diff(dense) > 0))

    def test_fractional_sample_indices_use_the_same_physical_axis(self):
        centers = trilinear_sample_centers(0.15, 48, 192)
        indices = np.array([[0.0, 95.5, 191.0]])
        physical = sample_indices_to_physical(indices, centers)
        np.testing.assert_allclose(physical, [[centers[0], 0.0, centers[-1]]])


if __name__ == "__main__":
    unittest.main()
