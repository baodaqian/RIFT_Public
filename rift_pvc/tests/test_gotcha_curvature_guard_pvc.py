"""RIFT-on-GOTCHA tuning campaign A57: the opt-in curvature guard after SH growth and warm restarts."""
import importlib.util
import json

import pytest
import torch

import train_gotcha_dataset_pvc as cli
from rift_pvc import gotcha_training as pvc
from rift_pvc.tests.test_gotcha_densify_pvc import ARGV, BOX, root  # noqa: F401  (fixture)
from rift_pvc.tests.test_gotcha_extend_epochs_pvc import dataset

spec = importlib.util.spec_from_file_location('extend', 'scripts_pvc/gotcha_extend_epochs.py')
extend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extend)

GUARDED = ['--passes', '1', '2', '--unit-split-stride', '4', '--unit-split-heldout-pass', '2',
           '--densify-epochs', '1', '--densify-max-active', '600', '--densify-lr-target', '1e-3',
           '--densify-curvature-units', '1', '--densify-curvature-iterations', '3', '--train-eval-every', '1',
           '--lr', '3e-4', '--pos-lr', '3e-4', '--loss-domain', 'full_native']


def args(epochs, guard=('restart',)):
    return cli.parse_args(['--epochs', str(epochs)] + ARGV[2:] + BOX + GUARDED
                          + (['--densify-curvature-guard', *guard] if guard else []))


def run(root, name, epochs, resume=None, guard=('restart',)):
    out = root / name / 'region' / 'rift_full_native'
    out.parent.mkdir(parents=True, exist_ok=True)
    (root / name / 'train.log').touch()
    return pvc.train(dataset(root), 'rift', pvc.recipe_from_args(args(epochs, guard), 'rift'), out,
                     device='cpu', resume=resume), out


def test_guard_is_opt_in_and_validated():
    assert 'curvature_guard' not in pvc.recipe_from_args(args(2, guard=()), 'rift')['densify']
    assert pvc.recipe_from_args(args(2, guard=('restart', 'growth', 'restart')), 'rift')['densify'][
        'curvature_guard'] == ['growth', 'restart']
    for bad in (['--densify-curvature-guard', 'growth'],   # no densify
                ['--densify-epochs', '1', '--densify-lr-rule', 'fixed', '--densify-curvature-guard', 'restart']):
        with pytest.raises(SystemExit):
            cli.parse_args(ARGV + BOX + ['--loss-domain', 'full_native'] + bad)


def test_guard_triggers():
    recipe = dict(densify=dict(curvature_guard=['growth', 'restart']))
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([param], lr=1.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
    grown = dict(refinement=dict(hh=dict(split=0, grown=3, active=10)))
    idle = dict(refinement=dict(hh=dict(split=0, grown=0, active=10)))
    optimizer.step()
    scheduler.step()                                   # after epoch 1: mid-cycle
    assert pvc.curvature_guard_due(recipe, grown, scheduler) == ['growth']
    assert pvc.curvature_guard_due(recipe, idle, scheduler) == []
    optimizer.step()
    scheduler.step()                                   # after epoch 2: the warm restart
    assert pvc.curvature_guard_due(recipe, idle, scheduler) == ['restart']
    assert pvc.curvature_guard_due(recipe, grown, scheduler) == ['growth', 'restart']
    assert pvc.curvature_guard_due(dict(densify=dict(curvature_guard=['growth'])), idle, scheduler) == []
    assert pvc.curvature_guard_due(dict(densify={}), grown, scheduler) == []


def test_guard_at_the_restart_and_its_resume_path_agree(root):
    # A declared 11-epoch run meets the epoch-10 restart in the loop. A 10-epoch run extended to 11
    # meets it at resume, where the guard's resume branch must run the same measurement.
    _, full = run(root, 'full', 11)
    history = json.loads((full / 'history.json').read_text())
    guard = history[9]['optimizer']['curvature_guard']
    assert guard['triggers'] == ['restart'] and 'densify' not in history[9]['optimizer']
    curvature = guard['curvature']
    assert curvature['target'] == 1e-3 and curvature['factor'] <= 1.0
    assert curvature['stability_after'] <= 1e-3 * (1 + 1e-9)
    assert all('curvature_guard' not in h['optimizer'] for i, h in enumerate(history) if i != 9)
    _, short = run(root, 'short', 10)
    assert 'curvature_guard' not in json.loads((short / 'history.json').read_text())[9]['optimizer']
    # The epoch-10 SH refinement event grows nothing here (children are not mature), so it is declared.
    extend.main([str(short), str(root / 'extended'), '--epochs', '11', '--past-restart'])
    target = root / 'extended' / 'region' / 'rift_full_native'
    run(root, 'extended', 11, resume=target / 'checkpoint_latest.pt')
    a = torch.load(target / 'checkpoint_final.pt', weights_only=False)
    b = torch.load(full / 'checkpoint_final.pt', weights_only=False)
    for k, v in a['model_state_dict'].items():
        assert torch.equal(v, b['model_state_dict'][k]), k
    assert a['optimizer_state_dict']['param_groups'][0]['lr'] == b['optimizer_state_dict']['param_groups'][0]['lr']
    ha = json.loads((target / 'history.json').read_text())
    ha[9].pop('extension')
    assert ha == history


def test_extension_needs_the_guard_or_a_declaration_past_restarts_and_growth(root):
    _, short = run(root, 'short', 2, guard=())
    with pytest.raises(SystemExit, match='restart after epoch 10'):
        extend.main([str(short), str(root / 'a'), '--epochs', '11'])
    # SH refinement events past the old end need the growth guard (the test recipe has sh_degree 2).
    with pytest.raises(SystemExit, match='refinement event after epoch 10'):
        extend.main([str(short), str(root / 'c'), '--epochs', '11', '--curvature-guard', 'restart'])
    extend.main([str(short), str(root / 'b'), '--epochs', '11', '--curvature-guard', 'restart', 'growth'])
    record = json.loads((root / 'b/region/rift_full_native/extension.json').read_text())
    assert record['curvature_guard_added'] == ['growth', 'restart'] and record['unguarded_acknowledged'] == []
    assert record['restarts_crossed'] == [10] and record['refinement_events_crossed'] == [10]
    saved = torch.load(root / 'b/region/rift_full_native/checkpoint_latest.pt', weights_only=False)
    assert saved['recipe']['densify']['curvature_guard'] == ['growth', 'restart']
