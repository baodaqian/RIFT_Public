"""GOTCHA learned-renderer amplitude law: the RIFT-dataset sum2 operator and the legacy unit kernel."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import train_gotcha_dataset as cli
from rift.gotcha_dataset import C, GOTCHADataset, Observation
from rift.gotcha_training import (ChannelField, RangeReadout, native_forward, recipe_from_args,
                                  train)
from rift.range_operator import range_forward_operator
from tests.test_gotcha_dataset import tiny_region, write_shard

ARGV = ['--granularity', '2', '--max-points', '15', '--sh-degree', '1']


def fixture():
    x = torch.tensor([[.01, -.02, .03], [-.03, 0, .01], [.02, .02, -.01]], dtype=torch.float64, requires_grad=True)
    w = torch.tensor([1+.2j, -.1+.3j, .4-.5j], dtype=torch.complex128, requires_grad=True)
    a = torch.tensor([20., -10., 4.], dtype=torch.float64)
    return x, w, a


class AmplitudeLawTests(unittest.TestCase):
    def test_sum2_is_the_rift_operator_amplitude_on_physical_range(self):
        x, w, a = fixture()
        f = torch.tensor([9e9, 9.13e9, 9.6e9], dtype=torch.float64)
        r0 = 21.5
        actual = native_forward(x, w, a, f, r0, point_chunk=2, range_model='sum2')
        r = np.linalg.norm(x.detach().numpy() - a.numpy(), axis=1)
        amplitude = 1 / ((4*np.pi)**2 * ((2*r)**2 + 1e-9))
        kernel = np.exp(-4j*np.pi/C*(r - r0)[:, None]*f.numpy())
        np.testing.assert_allclose(actual.detach().numpy(), (w.detach().numpy()*amplitude) @ kernel,
                                   rtol=2e-10, atol=0)
        self.assertTrue(torch.autograd.gradcheck(
            lambda points, weights: native_forward(points, weights, a, f, r0, point_chunk=2, range_model='sum2')*1e5,
            (x, w), eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_unit_is_the_legacy_kernel_bit_for_bit(self):
        x, w, a = fixture()
        f = torch.tensor([9e9, 9.13e9, 9.6e9], dtype=torch.float64)
        self.assertTrue(torch.equal(native_forward(x, w, a, f, 21.5, range_model='unit'),
                                    native_forward(x, w, a, f, 21.5)))
        with self.assertRaises(ValueError):
            native_forward(x, w, a, f, 21.5, range_model='none')

    def test_sum2_matches_rift_dataset_operator_up_to_the_known_reference_factor(self):
        """range_forward_operator (NUFFT, sum2, phase_sign -1) on the same monostatic pulse."""
        x, w, a = fixture()
        f = torch.linspace(9e9, 10e9, 64, dtype=torch.float64)
        r0 = 21.5
        native = native_forward(x.detach(), w.detach(), a, f, r0, range_model='sum2')
        synthetic = range_forward_operator(f, 2*np.pi*f/C, a[None], a[None], x.detach(), w.detach(),
                                           phase_sign=-1.0, range_model='sum2')[:, 0, 0]
        reference = torch.exp((-4j*np.pi/C)*r0*f)   # conj(D): native reference phase -> absolute path
        torch.testing.assert_close(native*reference, synthetic, rtol=1e-8, atol=1e-8*float(synthetic.abs().max()))


class RecipeTests(unittest.TestCase):
    def test_learned_methods_default_to_sum2_and_mfbp_does_not_declare_it(self):
        args = cli.parse_args(ARGV)
        for method in ('rift', 'rift_grid', 'isotropic'):
            self.assertEqual(recipe_from_args(args, method)['range_model'], 'sum2')
        self.assertNotIn('range_model', recipe_from_args(args, 'mfbp'))
        self.assertEqual(recipe_from_args(cli.parse_args(ARGV + ['--range-model', 'unit']), 'rift')['range_model'], 'unit')

    def test_field_renders_its_declared_law_and_legacy_recipes_render_unit(self):
        f = np.linspace(9e9, 10e9, 32)
        obs = Observation(1, 'hh', 2, 0, np.array([20., 1., 2.]), f, 20., np.ones(32, dtype=complex), 'synthetic')
        r = RangeReadout(tiny_region()).for_observation(obs)
        recipe = recipe_from_args(cli.parse_args(ARGV), 'isotropic')
        legacy = {k: v for k, v in recipe.items() if k != 'range_model'}
        torch.manual_seed(0)
        field = ChannelField('isotropic', tiny_region(), recipe, 'cpu')
        points, weights = field.field.active_scatterers()
        with torch.no_grad():
            y, _ = field(obs, r)
            torch.testing.assert_close(y, field.gain(native_forward(points, weights, r['antenna'], r['frequencies'],
                                                                    20., range_model='sum2')), rtol=0, atol=0)
            field.recipe = legacy
            y, _ = field(obs, r)
            torch.testing.assert_close(y, field.gain(native_forward(points, weights, r['antenna'], r['frequencies'],
                                                                    20.)), rtol=0, atol=0)

    def test_pre_declaration_checkpoint_resumes_only_as_unit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_shard(root/'New_Transfer/shards/pass1_hh.npz')
            ds = GOTCHADataset(root, passes=(1,), region=tiny_region())
            views = ds.viewpoints
            ds.viewpoints = lambda role: views(role)[:2 if role == 'train' else 1]
            args = ['--epochs', '1', '--refine-every', '2', '--probe-every', '1']
            unit = recipe_from_args(cli.parse_args(ARGV + args + ['--range-model', 'unit']), 'rift')
            with redirect_stdout(io.StringIO()):
                train(ds, 'rift', unit, root/'fit', device='cpu')
            checkpoint = torch.load(root/'fit/checkpoint_latest.pt', weights_only=False)
            checkpoint['recipe'].pop('range_model')
            torch.save(checkpoint, root/'legacy.pt')
            with self.assertRaisesRegex(ValueError, 'recipe'):
                train(ds, 'rift', dict(unit, range_model='sum2'), root/'fit', device='cpu', resume=root/'legacy.pt')
            with redirect_stdout(io.StringIO()):
                result = train(ds, 'rift', unit, root/'fit', device='cpu', resume=root/'legacy.pt')
            self.assertEqual(result['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
