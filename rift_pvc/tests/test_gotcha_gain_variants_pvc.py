"""RIFT-on-GOTCHA tuning campaign A62: opt-in gain-initialization variants (the user's experiment, B68).

(A) pooled gain warm start, (B) the gain in its own optimizer group, (D) the curvature rule at the start.
Absent flags keep the recipe, the optimizer layout and the single-pulse warm start.
"""
import json

import pytest
import torch

import train_gotcha_dataset_pvc as cli
from rift.gotcha_training import rift_dataset_initialization
from rift_pvc import gotcha_training as pvc
from rift_pvc.tests.test_gotcha_densify_pvc import ARGV, BOX, root  # noqa: F401  (fixture)
from rift_pvc.tests.test_gotcha_extend_epochs_pvc import dataset

BASE = ['--passes', '1', '2', '--unit-split-stride', '4', '--unit-split-heldout-pass', '2',
        '--densify-epochs', '1', '--densify-max-active', '600', '--densify-lr-target', '1e-3',
        '--densify-curvature-units', '1', '--densify-curvature-iterations', '3', '--train-eval-every', '1',
        '--lr', '3e-4', '--pos-lr', '3e-4', '--loss-domain', 'full_native']


def recipe(extra=(), epochs=2):
    return pvc.recipe_from_args(cli.parse_args(['--epochs', str(epochs)] + ARGV[2:] + BOX + BASE + list(extra)), 'rift')


def test_variants_are_opt_in_and_validated():
    plain = recipe()
    assert not {'gain_optimizer', 'gain_warm_start'} & set(plain) and 'curvature_start' not in plain['densify']
    full = recipe(['--gain-warm-start', 'pooled', '--gain-lr', '1e-3', '--densify-curvature-start'])
    assert full['gain_warm_start']['rule'] == 'pooled' and full['densify']['curvature_start'] is True
    assert full['gain_optimizer']['lr'] == 1e-3 and full['gain_optimizer']['eps'] == 1e-8
    for bad in (['--gain-warm-start', 'pooled', '--initialization', 'random'], ['--gain-lr', '-1'],
                ['--densify-curvature-start', '--densify-lr-rule', 'fixed']):
        with pytest.raises(SystemExit):
            cli.parse_args(ARGV + BOX + BASE + bad)
    with pytest.raises(SystemExit):
        cli.parse_args(ARGV + BOX + ['--loss-domain', 'full_native', '--densify-curvature-start'])


def test_scale_rates_except_gain_keeps_the_gain_group():
    params = [torch.nn.Parameter(torch.zeros(1)) for _ in range(3)]
    optimizer = torch.optim.AdamW([dict(params=[params[0]], lr=1e-3), dict(params=[params[1]], lr=2e-3),
                                   dict(params=[params[2]], lr=5e-2, eps=1e-8, name=pvc.GAIN_GROUP)],
                                  eps=1.5, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    rates = pvc.scale_rates_except_gain(optimizer, scheduler, 0.25)
    assert rates == pytest.approx([2.5e-4, 5e-4, 5e-2])
    assert scheduler.base_lrs == pytest.approx([2.5e-4, 5e-4, 5e-2]) and scheduler.eta_min == pytest.approx(2.5e-7)
    assert pvc.coefficient_base_lr(optimizer, scheduler) == pytest.approx(2.5e-4)


def test_pooled_refit_makes_the_start_least_squares_on_every_start_pulse(root):
    ds = dataset(root)
    rec = recipe(['--gain-warm-start', 'pooled'])
    head = pvc.ChannelField('rift', ds.region, rec, 'cpu')
    readout = pvc.RangeReadout(ds.region, device='cpu')
    views = ds.viewpoints('train')[:3]
    first = rift_dataset_initialization(head, ds, readout, views, 'hh')
    pooled = pvc.pooled_gain_refit(head, ds, readout, views, 'hh', first)
    cross, energy = 0j, 0.0
    with torch.no_grad():
        for view in views:
            for obs in ds.observations(*view, 'hh'):
                predicted = head(obs, readout.for_observation(obs))[0]
                cross += complex((predicted.conj() * pvc._target(obs, 'cpu')).sum().item())
                energy += float(predicted.abs().square().sum())
    assert cross / energy == pytest.approx(1 + 0j, abs=1e-6)   # float32 gain parameters
    assert pooled['warm_start_gain_first_pulse'] == first['warm_start_gain']
    assert pooled['gain_warm_start']['pulses'] == first['pulses']
    g = abs(head.gain.gain_value())
    assert pooled['m1'] == pytest.approx(first['m1'] * g / abs(complex(*pooled['gain_warm_start']['gain_before'])))
    assert pooled['l1_weight'] * pooled['m1'] == pytest.approx(first['l1_weight'] * first['m1'])


def train(root, name, extra, epochs=2):
    out = root / name
    pvc.train(dataset(root), 'rift', recipe(extra, epochs), out, device='cpu')
    return (json.loads((out / 'history.json').read_text()), json.loads((out / 'initialization.json').read_text()),
            torch.load(out / 'checkpoint_final.pt', weights_only=False))


def test_gain_group_keeps_its_rate_through_the_curvature_cut(root):
    history, _, saved = train(root, 'gain', ['--gain-lr', '1e-3'])
    groups = saved['optimizer_state_dict']['param_groups']
    assert len(groups) == 3 and groups[2]['name'] == pvc.GAIN_GROUP and groups[2]['eps'] == 1e-8
    event = history[0]['optimizer']['densify']
    assert event['curvature']['factor'] < 1
    # Only the coefficient and position rates were cut; the gain group follows its own cosine.
    before, after = history[0]['optimizer']['learning_rates'], event['learning_rates_after']
    assert after[0] == pytest.approx(before[0] * event['curvature']['factor'], rel=0.2)
    assert saved['scheduler_state_dict']['base_lrs'][2] == pytest.approx(1e-3)
    assert event['curvature']['base_lr'] == pytest.approx(3e-4)
    assert all(len(h['gain']['hh']) == 2 for h in history)
    plain, _, plain_saved = train(root, 'plain', [])
    assert len(plain_saved['optimizer_state_dict']['param_groups']) == 2 and 'gain' not in plain[0]


def test_curvature_start_sets_the_base_rate_at_the_target(root):
    history, init, _ = train(root, 'start', ['--densify-curvature-start'])
    start = init['hh']['curvature_start']['curvature']
    assert start['allow_raise'] is True and start['stability_after'] == pytest.approx(1e-3)
    assert start['iterations'] == 3 * 3 and all('converged' in u for u in start['units'])
    rates = history[0]['optimizer']['learning_rates']
    assert rates[0] == pytest.approx(3e-4 * start['factor'])
    # A raise (target 4x the measured S) moves the coefficient group only; positions keep pos_lr (B69 review).
    target = 4 * start['stability_before']
    history, init, _ = train(root, 'raise', ['--densify-curvature-start', '--densify-lr-target', repr(target)])
    raised = init['hh']['curvature_start']['curvature']
    assert raised['factor'] == pytest.approx(4.0, rel=1e-6) and raised['raised_groups'] == [0]
    rates = history[0]['optimizer']['learning_rates']
    assert rates[0] == pytest.approx(3e-4 * 4.0, rel=1e-6) and rates[1] == pytest.approx(3e-4)
