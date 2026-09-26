"""GOTCHA adaptive RIFT with the RIFT-dataset start and priors: backprojection, gauge, transferred priors."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import math
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import train_gotcha_dataset as cli
from rift.gotcha_dataset import C, GOTCHADataset
from rift.gotcha_training import (BACKPROJECTION, RIFT_DATASET_PRIORS, RIFT_DATASET_REFERENCE, ChannelField,
                                  RangeReadout, _target, native_adjoint, native_forward, recipe_from_args,
                                  rift_dataset_initialization, rift_dataset_prior, train)
from tests.test_gotcha_dataset import tiny_region, write_shard

ARGV = ['--epochs', '1', '--granularity', '2', '--max-points', '15', '--sh-degree', '1',
        '--refine-every', '100', '--probe-every', '1', '--bp-views', '2']
LEGACY = ['--range-model', 'unit', '--initialization', 'random', '--priors', 'none']


def dataset(root, train_views=2):
    write_shard(root/'New_Transfer/shards/pass1_hh.npz')
    ds = GOTCHADataset(root, passes=(1,), region=tiny_region())
    views = ds.viewpoints
    ds.viewpoints = lambda role: views(role)[:train_views if role == 'train' else 1]
    return ds


def interrupt_after_first_step():
    step = torch.optim.AdamW.step
    def run(optimizer, *args, **kwargs):
        result = step(optimizer, *args, **kwargs)
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        return result
    return patch.object(torch.optim.AdamW, 'step', run)


class AdjointTests(unittest.TestCase):
    def test_native_adjoint_is_the_adjoint_of_native_forward(self):
        g = torch.Generator().manual_seed(0)
        points = (torch.rand(9, 3, generator=g, dtype=torch.float64) - .5) * .06
        x = torch.complex(torch.randn(9, generator=g, dtype=torch.float64), torch.randn(9, generator=g, dtype=torch.float64))
        f = torch.linspace(9e9, 10e9, 12, dtype=torch.float64)
        z = torch.complex(torch.randn(12, generator=g, dtype=torch.float64), torch.randn(12, generator=g, dtype=torch.float64))
        a = torch.tensor([20., 1., 2.], dtype=torch.float64)
        for model in ('sum2', 'unit'):
            forward = torch.vdot(native_forward(points, x, a, f, 20.2, point_chunk=4, range_model=model), z)
            adjoint = torch.vdot(x, native_adjoint(points, z, a, f, 20.2, point_chunk=4, range_model=model))
            torch.testing.assert_close(forward, adjoint, rtol=1e-12, atol=0)


class InitializationTests(unittest.TestCase):
    def test_backprojection_start_gauge_and_prior_strength(self):
        with tempfile.TemporaryDirectory() as temp:
            ds = dataset(Path(temp))
            recipe = recipe_from_args(cli.parse_args(ARGV), 'rift')
            self.assertEqual((recipe['initialization'], recipe['priors']), (BACKPROJECTION, RIFT_DATASET_PRIORS))
            head = ChannelField('rift', ds.region, recipe, 'cpu')
            self.assertEqual(float(head.field.w_re.detach().abs().sum() + head.field.w_im.detach().abs().sum()), 0.0)
            readout = RangeReadout(ds.region)
            views = ds.viewpoints('train')
            with redirect_stdout(io.StringIO()):
                record = rift_dataset_initialization(head, ds, readout, views, 'hh')
            # Independent dense backprojection in the projected loss domain.
            mask = head.field.active_mask
            x = head.field.grid_positions.reshape(-1, 3)[mask].double().numpy()
            b = np.zeros(len(x), dtype=complex)
            for p, sector in views:
                for obs in ds.observations(p, sector, 'hh'):
                    r = readout.for_observation(obs)
                    distance = np.linalg.norm(x - r['antenna'].numpy(), axis=1)
                    amplitude = 1 / ((4*np.pi)**2 * ((2*distance)**2 + 1e-9))
                    matrix = amplitude[None] * np.exp(-4j*np.pi/C*(distance - obs.reference_range_m)[None]*r['frequencies'].numpy()[:, None])
                    projected = readout.lift(readout.project(_target(obs, 'cpu'), r), r).numpy()
                    b += matrix.conj().T @ projected
            y00 = .5 / math.sqrt(math.pi)
            w_re, w_im = head.field.w_re.detach()[mask], head.field.w_im.detach()[mask]
            c00 = torch.complex(w_re[:, 0].double(), w_im[:, 0].double()).numpy()
            self.assertEqual(float(w_re[:, 1:].abs().sum() + w_im[:, 1:].abs().sum()), 0.0)
            alpha = complex(*record['alpha'])
            np.testing.assert_allclose(c00 * y00, record['coefficient_gauge'] * alpha * b, rtol=1e-5)
            # The gauge preserves the prediction and gives B787's starting coefficient size.
            gain = head.gain.gain_value()
            np.testing.assert_allclose(gain * c00 * y00, complex(*record['warm_start_gain']) * alpha * b, rtol=1e-5)
            self.assertAlmostEqual(float(np.abs(c00).mean()) / RIFT_DATASET_REFERENCE['coefficient_norm'], 1.0, places=5)
            # At the start the L1 prior is the RIFT-dataset fraction mu1 of the mean power; SH is free at degree 0.
            _, terms = rift_dataset_prior(head, record)
            self.assertAlmostEqual(float(terms['l1']) / RIFT_DATASET_REFERENCE['mu1'], 1.0, places=5)
            self.assertEqual(float(terms['sh_degree']), 0.0)


class TrainingTests(unittest.TestCase):
    def test_priors_stay_out_of_refinement_statistics_and_epoch_order_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ds = dataset(root)
            saved = {}
            for name, extra in (('priors', []), ('no_priors', ['--priors', 'none']), ('legacy', LEGACY)):
                recipe = recipe_from_args(cli.parse_args(ARGV + extra), 'rift')
                with redirect_stdout(io.StringIO()), interrupt_after_first_step():
                    self.assertEqual(train(ds, 'rift', recipe, root/name, device='cpu')['status'], 'interrupted')
                saved[name] = torch.load(root/name/'checkpoint_latest.pt', weights_only=False)
            self.assertEqual(saved['priors']['order'], saved['legacy']['order'])
            self.assertTrue((root/'priors/initialization.json').is_file())
            self.assertFalse((root/'legacy/initialization.json').exists())
            head = ChannelField('rift', ds.region, saved['priors']['recipe'], 'cpu')
            parameters = {f'hh.{name}' for name, _ in head.named_parameters()}
            a, b = saved['priors']['model_state_dict'], saved['no_priors']['model_state_dict']
            for key in a:
                if key not in parameters:
                    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, msg=key)
            self.assertFalse(torch.equal(a['hh.field.w_re'], b['hh.field.w_re']))   # the prior did act

    def test_completed_fit_records_priors_and_legacy_checkpoint_resumes_only_as_legacy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ds = dataset(root)
            with redirect_stdout(io.StringIO()):
                train(ds, 'rift', recipe_from_args(cli.parse_args(ARGV), 'rift'), root/'fit', device='cpu')
                history = torch.load(root/'fit/checkpoint_final.pt', weights_only=False)['history']
                self.assertEqual(set(history[-1]['priors']['hh']), {'l1', 'sh_degree'})
                legacy = recipe_from_args(cli.parse_args(ARGV + LEGACY), 'rift')
                train(ds, 'rift', legacy, root/'legacy', device='cpu')
            checkpoint = torch.load(root/'legacy/checkpoint_latest.pt', weights_only=False)
            for key in ('range_model', 'initialization', 'priors'):
                checkpoint['recipe'].pop(key)
            torch.save(checkpoint, root/'old.pt')
            for extra in ([], ['--range-model', 'unit', '--initialization', 'random', '--priors', 'none', '--range-model', 'sum2']):
                with self.assertRaisesRegex(ValueError, 'recipe'):
                    train(ds, 'rift', recipe_from_args(cli.parse_args(ARGV + extra), 'rift'), root/'legacy',
                          device='cpu', resume=root/'old.pt')
            with redirect_stdout(io.StringIO()):
                self.assertEqual(train(ds, 'rift', legacy, root/'legacy', device='cpu', resume=root/'old.pt')['status'], 'complete')

    def test_recipe_keys_belong_to_adaptive_rift_only(self):
        args = cli.parse_args(ARGV)
        for method in ('rift_grid', 'isotropic', 'mfbp'):
            recipe = recipe_from_args(args, method)
            self.assertFalse({'initialization', 'priors', 'bp_views', 'rift_dataset_reference'} & set(recipe))
        self.assertEqual(recipe_from_args(args, 'rift')['bp_views'], 2)
        self.assertEqual(recipe_from_args(args, 'rift')['rift_dataset_reference'], RIFT_DATASET_REFERENCE)
        with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), patch('sys.stderr', io.StringIO()):
            cli.parse_args(ARGV + ['--initialization', 'random'])
        self.assertNotIn('bp_views', recipe_from_args(cli.parse_args(ARGV + ['--initialization', 'random', '--priors', 'none']), 'rift'))


if __name__ == '__main__':
    unittest.main()
