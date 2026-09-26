"""GOTCHA adaptive RIFT with the RIFT-dataset B787 optimizer and schedule (default) and the legacy optimizer."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import torch

import train
import train_gotcha_dataset as cli
from rift.b7873200_adaptive_fullscale import fullscale_train_argv
from rift.gotcha_training import (B787_OPTIMIZER, B787_SCHEDULE, LEGACY_OPTIMIZER, fixed_view_order, probe_due,
                                  recipe_from_args, train as fit)
from tests.test_gotcha_rift_dataset_recipe import dataset

ARGV = ['--granularity', '2', '--max-points', '15', '--sh-degree', '1', '--bp-views', '2', '--probe-every', '1']


def b787_command():
    """The production B787 adaptive-RIFT arguments as train.py parses them."""
    return train.parse_args(fullscale_train_argv(npz_path='/nonexistent/b787.npz', manifest_path='/nonexistent/roles.json',
                                                 checkpoint_root='/nonexistent/root'))


def interrupt_after(steps):
    step, calls = torch.optim.AdamW.step, []
    def run(optimizer, *args, **kwargs):
        result = step(optimizer, *args, **kwargs)
        calls.append(1)
        if len(calls) == steps:
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        return result
    return patch.object(torch.optim.AdamW, 'step', run)


class RecipeTests(unittest.TestCase):
    def test_defaults_are_the_b787_production_command(self):
        recipe, b787 = recipe_from_args(cli.parse_args([]), 'rift'), b787_command()
        schedule = recipe['optimizer_schedule']
        self.assertEqual(recipe['optimizer'], B787_OPTIMIZER)
        self.assertEqual((recipe['lr'], recipe['pos_lr'], recipe['adam_eps'], schedule['weight_decay']),
                         (b787.lr, b787.pos_lr, b787.adam_eps, b787.weight_decay))
        self.assertEqual((schedule['t0'], schedule['t_mult']), (b787.t0, b787.t_mult))
        self.assertEqual((recipe['refine_every'], recipe['probe_every'], recipe['max_level'], recipe['max_points'], recipe['epochs']),
                         (b787.adaptive_refine_every, b787.adaptive_probe_every, b787.split_max_level,
                          b787.adaptive_max_active, b787.epochs))
        for key in ('spatial_fraction', 'angular_fraction', 'min_spatial_exposure', 'min_angular_exposure',
                    'spatial_floor', 'angular_floor', 'cooldown_events', 'child_maturity_events'):
            self.assertEqual(schedule[key], getattr(b787, f'adaptive_{key}'), key)
        self.assertEqual(b787.clip_grad_norm, 0.0)   # neither trainer clips
        # sigma^2 = 1: the TRAIN-power normalization is unchanged.
        self.assertEqual(recipe['loss_normalization'], 'train_only_mean_projected_power')

    def test_optimizer_keys_belong_to_adaptive_rift_and_legacy_keeps_old_values(self):
        legacy = recipe_from_args(cli.parse_args(['--optimizer', 'legacy']), 'rift')
        self.assertEqual({k: legacy[k] for k in ('optimizer', 'lr', 'pos_lr', 'adam_eps', 'refine_every', 'probe_every',
                                                 'refine_fraction', 'max_level')},
                         dict(optimizer=LEGACY_OPTIMIZER, lr=.003, pos_lr=1e-4, adam_eps=1e-20, refine_every=100,
                              probe_every=10, refine_fraction=.05, max_level=3))
        self.assertNotIn('optimizer_schedule', legacy)
        for method in ('rift_grid', 'isotropic', 'mfbp'):
            recipe = recipe_from_args(cli.parse_args([]), method)
            self.assertFalse({'optimizer', 'optimizer_schedule'} & set(recipe))
            self.assertEqual((recipe['pos_lr'], recipe['adam_eps'], recipe['refine_every']), (1e-4, 1e-20, 100))
        with self.assertRaisesRegex(ValueError, 'refine-fraction'):
            recipe_from_args(cli.parse_args(['--refine-fraction', '.1']), 'rift')
        self.assertEqual(recipe_from_args(cli.parse_args(['--refine-fraction', '.1', '--optimizer', 'legacy']), 'rift')['refine_fraction'], .1)
        self.assertEqual(recipe_from_args(cli.parse_args(['--lr', '.01']), 'rift')['lr'], .01)   # explicit values are recorded

    def test_probe_stride_rotates_with_the_epoch_as_in_train_py(self):
        recipe, schedule = dict(probe_every=16), B787_SCHEDULE
        for epoch in range(3):
            due = [c for c in range(40) if probe_due(recipe, schedule, updates=None, cursor=c, epoch=epoch)]
            self.assertEqual(due, [c for c in range(40) if (c + epoch) % 16 == 0])
        self.assertTrue(probe_due(recipe, None, updates=32, cursor=5, epoch=1))   # legacy: every probe_every-th update


class TrainingTests(unittest.TestCase):
    def test_learning_rates_follow_train_py_cosine_restarts_and_events_follow_its_epochs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ds = dataset(root)
            recipe = recipe_from_args(cli.parse_args(ARGV + ['--epochs', '12']), 'rift')
            with redirect_stdout(io.StringIO()):
                fit(ds, 'rift', recipe, root/'fit', device='cpu')
            history = torch.load(root/'fit/checkpoint_final.pt', weights_only=False)['history']
            # train.py: CosineAnnealingWarmRestarts(T_0=t0, T_mult=t_mult, eta_min=1e-6), stepped once per epoch.
            b787 = b787_command()
            groups = [torch.nn.Parameter(torch.zeros(1)) for _ in range(2)]
            optimizer = torch.optim.AdamW([{'params': [groups[0]], 'lr': b787.lr}, {'params': [groups[1]], 'lr': b787.pos_lr}])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=b787.t0, T_mult=b787.t_mult, eta_min=1e-6)
            expected = []
            for _ in range(12):
                expected.append([g['lr'] for g in optimizer.param_groups])
                scheduler.step()
            for entry, lrs in zip(history, expected):
                self.assertEqual(entry['optimizer']['learning_rates'], lrs, entry['epoch'])
            self.assertEqual(history[10]['optimizer']['learning_rates'], [b787.lr, b787.pos_lr])   # the restart
            events = [entry['epoch'] for entry in history if 'refinement' in entry['optimizer']]
            self.assertEqual(events, [10])

    def test_fixed_order_and_exact_interrupted_resume_across_epochs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ds = dataset(root)
            recipe = recipe_from_args(cli.parse_args(ARGV + ['--epochs', '3', '--refine-every', '2', '--checkpoint-every', '1']), 'rift')
            with redirect_stdout(io.StringIO()):
                fit(ds, 'rift', recipe, root/'full', device='cpu')
                with interrupt_after(3):   # mid epoch 2, after the epoch-1 scheduler step
                    self.assertEqual(fit(ds, 'rift', recipe, root/'resumed', device='cpu')['status'], 'interrupted')
                latest = torch.load(root/'resumed/checkpoint_latest.pt', weights_only=False)
                self.assertEqual((latest['epoch'], latest['cursor']), (1, 1))
                self.assertEqual(latest['order'], fixed_view_order(42, 2))   # epoch 2 repeats epoch 1's order
                self.assertEqual(latest['scheduler_state_dict']['last_epoch'], 1)
                broken = dict(latest)
                broken.pop('scheduler_state_dict')
                torch.save(broken, root/'broken.pt')
                with self.assertRaisesRegex(ValueError, 'scheduler_state_dict'):
                    fit(ds, 'rift', recipe, root/'resumed', device='cpu', resume=root/'broken.pt')
                self.assertEqual(fit(ds, 'rift', recipe, root/'resumed', device='cpu',
                                     resume=root/'resumed/checkpoint_latest.pt')['status'], 'complete')
            full = torch.load(root/'full/checkpoint_final.pt', weights_only=False)
            resumed = torch.load(root/'resumed/checkpoint_final.pt', weights_only=False)
            self.assertEqual(full['history'], resumed['history'])
            self.assertEqual(full['scheduler_state_dict'], resumed['scheduler_state_dict'])
            for key, value in full['model_state_dict'].items():
                torch.testing.assert_close(resumed['model_state_dict'][key], value, rtol=0, atol=0, msg=key)
            for key, state in full['optimizer_state_dict']['state'].items():
                for name, value in state.items():
                    torch.testing.assert_close(resumed['optimizer_state_dict']['state'][key][name], value, rtol=0, atol=0)
            self.assertEqual(full['optimizer_state_dict']['param_groups'], resumed['optimizer_state_dict']['param_groups'])

    def test_legacy_checkpoint_resumes_only_with_the_legacy_optimizer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ds = dataset(root)
            legacy = recipe_from_args(cli.parse_args(ARGV + ['--epochs', '2', '--optimizer', 'legacy']), 'rift')
            with redirect_stdout(io.StringIO()), interrupt_after(1):
                self.assertEqual(fit(ds, 'rift', legacy, root/'legacy', device='cpu')['status'], 'interrupted')
            checkpoint = torch.load(root/'legacy/checkpoint_latest.pt', weights_only=False)
            self.assertNotIn('scheduler_state_dict', checkpoint)
            self.assertEqual([g['eps'] for g in checkpoint['optimizer_state_dict']['param_groups']], [1e-20, 1e-20])
            self.assertEqual([g['lr'] for g in checkpoint['optimizer_state_dict']['param_groups']], [.003, 1e-4])
            checkpoint['recipe'].pop('optimizer')   # as written before the key existed
            torch.save(checkpoint, root/'old.pt')
            with self.assertRaisesRegex(ValueError, 'recipe'):
                fit(ds, 'rift', recipe_from_args(cli.parse_args(ARGV + ['--epochs', '2']), 'rift'), root/'legacy',
                    device='cpu', resume=root/'old.pt')
            with redirect_stdout(io.StringIO()):
                self.assertEqual(fit(ds, 'rift', legacy, root/'legacy', device='cpu', resume=root/'old.pt')['status'], 'complete')
            final = torch.load(root/'legacy/checkpoint_final.pt', weights_only=False)
            self.assertNotIn('optimizer', final['history'][-1])
            self.assertEqual([g['lr'] for g in final['optimizer_state_dict']['param_groups']], [.003, 1e-4])


if __name__ == '__main__':
    unittest.main()
