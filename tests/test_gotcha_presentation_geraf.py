"""GeRaF GOTCHA presentation: the per-sample decomposition re-synthesizes the model's own render."""
from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from rift.geraf_source_ops import NativeAcquisition
from rift_pvc import gotcha_presentation as presentation
from rift_pvc import gotcha_presentation_geraf as geraf
from rift_pvc.geraf_source import build_model, fixed_numpy_seed, predict_native, recipe_from_config, sample_frame

EXTENT = 5.0


def small_model(light_power=3.0):
    recipe = recipe_from_config(dict(sdf_hidden_dim=32, sdf_layers=3, sdf_skip=1, sdf_levels=4, n_aperture=8,
                                     n_samples=8, n_samples_tgt=12, bank_size=1, light_power=light_power), EXTENT)
    torch.manual_seed(0)
    model = build_model(recipe)
    model.eval()
    return model, recipe


def acquisition():
    # Three monostatic pulses ~10 km out at 45 degrees elevation, GOTCHA-like frequencies and 2*r0 reference.
    azimuth = np.radians([30.0, 30.2, 30.4])
    ground = 7_100.0
    points = torch.as_tensor(np.stack([ground * np.cos(azimuth), ground * np.sin(azimuth),
                                       np.full(3, 7_150.0)], -1))
    r0 = points.norm(dim=-1) + torch.tensor([12.6, 12.5, 12.7], dtype=torch.float64)
    frequencies = torch.tensor([9.29e9, 9.3e9, 9.312e9, 9.33e9], dtype=torch.float64)
    return NativeAcquisition(points, points, frequencies, 2 * r0, point_chunk=256, pair_chunk=2)


class GerafPresentationTests(unittest.TestCase):
    @torch.no_grad()
    def test_samples_resynthesize_the_native_render_through_K(self):
        """y_p = sum_s K sqrt(RCS_ps) exp(i phase) / (Rt+Rr)^2 equals predict_native, pulse by pulse."""
        model, recipe = small_model()
        acq = acquisition()
        with fixed_numpy_seed(recipe['seed']):
            frame = sample_frame(acq, recipe, 'train_hh_fixture')
        rendered = predict_native(model, frame, acq)
        points, amplitude, weight, path = geraf.view_scatterers(model, frame, acq)
        phase = acq.phase(points, torch.arange(len(acq.tx)))                  # [P, N, F]
        synthesized = (presentation.K * amplitude[..., None] * phase / path[..., None] ** 2).sum(1)
        self.assertGreater(int((amplitude > 0).sum()), 0)
        np.testing.assert_allclose(synthesized.numpy(), rendered.numpy(), rtol=1e-9, atol=1e-14 * float(rendered.abs().max()))
        # Each sample's sqrt(RCS) is its signal weight times the released specular gate (0..1), over K.
        spec = amplitude * presentation.K / weight.clamp_min(1e-300)
        active = amplitude > 0
        self.assertTrue(bool(((spec[active] > 0) & (spec[active] <= 1 + 1e-9)).all()))

    @torch.no_grad()
    def test_light_power_scales_rcs_as_its_exponential_squared(self):
        acq = acquisition()
        results = []
        for light_power in (0.0, 2.0):
            model, recipe = small_model(light_power)
            with fixed_numpy_seed(recipe['seed']):
                frame = sample_frame(acq, recipe, 'train_hh_fixture')
            points, amplitude, weight, _ = geraf.view_scatterers(model, frame, acq)
            results.append(geraf.view_cells(points, amplitude, weight, EXTENT, 6))
        np.testing.assert_allclose(results[1][1], results[0][1] * math.exp(4.0), rtol=1e-5)
        np.testing.assert_allclose(results[1][0], results[0][0] * math.exp(4.0), rtol=1e-5)

    def test_view_cells_sum_amplitudes_before_squaring(self):
        # CIC on a 2-cell grid: a point at a cell centre deposits into that cell only.
        points = torch.tensor([[2.5, 2.5, 2.5], [2.5, 2.5, 2.5], [-2.5, -2.5, -2.5]], dtype=torch.float64)
        amplitude = torch.tensor([[1.0, 2.0, 5.0], [3.0, 0.0, 1.0]], dtype=torch.float64)
        weight = torch.tensor([0.5, 0.25, 1.0], dtype=torch.float64)
        relative, rcs = geraf.view_cells(points, amplitude, weight, EXTENT, 2)
        self.assertAlmostEqual(rcs[1, 1, 1], (3.0 ** 2 + 3.0 ** 2) / 2)
        self.assertAlmostEqual(rcs[0, 0, 0], (25.0 + 1.0) / 2)
        self.assertAlmostEqual(relative[1, 1, 1], 0.75 ** 2)


if __name__ == '__main__':
    unittest.main()
