"""SpINR GOTCHA presentation: readout-relative and K-calibrated RCS through the native kernel."""
from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from rift.gotcha_dataset import Observation, Region
from rift.spinr_fidelity import field_readout
from rift.spinr_native import NativeKernel
from rift.spinr_style import SpinrStyleINR
from rift_pvc import gotcha_presentation as presentation
from rift_pvc import gotcha_presentation_spinr as spinr
from rift_pvc.spinr_native_batched import BatchedNativeKernel

EXTENT = 5.0
REGION = Region('fixture', 'synthetic', (0., 0., 0.), ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), EXTENT,
                'synthetic unit test')
RECIPE = dict(grid_size=4, nodes_per_cell=1, neural_point_tile=64)


def model(seed=0):
    torch.manual_seed(seed)
    head = SpinrStyleINR(support_m=EXTENT)
    head.eval()
    return head


def observation(position, frequencies, r0, pulse=0):
    return Observation(1, 'hh', 3, pulse, np.asarray(position, dtype=np.float64), np.asarray(frequencies), r0,
                       np.zeros(len(frequencies), dtype=np.complex128), 'own_published_source_af_once')


def checkpoint(head, scale):
    return dict(method_format=spinr.FORMAT, recipe=dict(RECIPE), initial_scales=dict(hh=dict(value=scale)),
                model_state_dict={f'hh.{k}': v for k, v in head.state_dict().items()},
                dataset_contract=dict(region=dict(half_extent_m=EXTENT)), epoch=4, cursor=0, updates=10,
                best_val=None, best_epoch=None)


class SpinrPresentationTests(unittest.TestCase):
    @torch.no_grad()
    def test_rcs_reproduces_the_native_render_through_K(self):
        """Every node's sqrt(RCS) re-synthesizes the native render: y = sum K sqrt(RCS) phase / (2d)^2."""
        head, scale = model(), 3.0e5
        nodes, weights, amplitude = spinr.node_amplitudes(head, scale, RECIPE, EXTENT)
        field = spinr._field(head, nodes)
        frequencies = np.array([9.29e9, 9.3e9, 9.312e9, 9.33e9])        # nonaffine: exact point-kernel path
        antenna = np.array([7000.0, -6800.0, 7250.0])
        obs = observation(antenna, frequencies, 10_120.0)
        render = NativeKernel(obs, REGION).render(nodes, field, weights, scale, selected=False)
        d = torch.linalg.vector_norm(nodes - torch.as_tensor(antenna), dim=-1)
        phase = torch.exp((-4j * math.pi / 299792458.0) * (d - 10_120.0)[:, None] * torch.as_tensor(frequencies))
        synthesized = (presentation.K * amplitude[:, None] * phase / (2 * d[:, None]) ** 2).sum(0)
        np.testing.assert_allclose(render.numpy(), synthesized.numpy(), rtol=1e-10, atol=0)
        # An isolated node: per-sample |y| (2d)^2 / K equals its |sqrt(RCS)| at every frequency.
        i = int(amplitude.abs().argmax())
        single = NativeKernel(obs, REGION).render(nodes[i:i + 1], field[i:i + 1], weights[i:i + 1], scale,
                                                  selected=False)
        np.testing.assert_allclose((single.abs() * (2 * d[i]) ** 2 / presentation.K).numpy(),
                                   np.full(len(frequencies), float(amplitude[i].abs())), rtol=1e-12)
        # The PVC batched kernel the trainer uses renders the same values.
        batched = BatchedNativeKernel([obs, observation(antenna * 1.0001, frequencies, 10_121.0, 1)], REGION)
        np.testing.assert_allclose(batched.render(nodes, field, weights, scale, selected=False)[0].numpy(),
                                   render.numpy(), rtol=1e-10)

    def test_presentation_is_the_readout_and_the_squared_node_sum(self):
        head, scale = model(1), 2.0
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'checkpoint_latest.pt'
            torch.save(checkpoint(head, scale), path)
            result = spinr.spinr_presentation(path, grid=4)
        _, sigma = field_readout(head, grid_size=4, support_m=EXTENT, initial_output_scale=scale,
                                 neural_point_tile=64, device='cpu')
        np.testing.assert_allclose(result.relative, sigma.square().reshape(4, 4, 4).numpy(), rtol=0, atol=0)
        # Cell (i, j, k) is the midpoint at axis[i], axis[j], axis[k] (x outer, z inner).
        axis = (torch.arange(4, dtype=torch.float64) + .5) * 2.5 - EXTENT
        with torch.no_grad():
            value = float(head(torch.stack([axis[3], axis[0], axis[2]])[None]).double()) * scale
        # float32 network: a one-point batch rounds differently from the grid batch.
        self.assertLess(abs(float(result.relative[3, 0, 2]) / value ** 2 - 1), 1e-5)
        self.assertGreater(abs(float(result.relative[0, 3, 2]) / value ** 2 - 1), 1e-3)
        _, weights, amplitude = spinr.node_amplitudes(head, scale, RECIPE, EXTENT)
        np.testing.assert_allclose(result.rcs_m2.reshape(-1), amplitude.square().numpy(), rtol=1e-12)
        self.assertEqual(result.source['half_extent_m'], EXTENT)


if __name__ == '__main__':
    unittest.main()
