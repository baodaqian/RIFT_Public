"""RIFT-on-GOTCHA tuning campaign (docs/RIFT_GOTCHA_Tune.md section 7): box support and initial SH degree."""
import json

import pytest
import torch

import train_gotcha_dataset_pvc as cli
from rift.gotcha_dataset import GOTCHADataset, load_region
from rift_pvc import gotcha_training as pvc
from rift_pvc.gotcha_isolation import Box
from tests.test_gotcha_dataset import write_shard, tiny_region
from tests.test_gotcha_pulse_sampling import multiple_pulses

ARGV = ['--epochs', '1', '--granularity', '2', '--max-points', '600', '--sh-degree', '3', '--probe-every', '1',
        '--pulses-per-sector', '3', '--point-chunk', '64', '--checkpoint-every', '1']
BOX = ['--support-box', '0.03', '0.015', '0.0125', '--support-pitch', '0.005']


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv('RIFT_ACCELERATOR', 'cpu')
    write_shard(tmp_path / 'New_Transfer/shards/pass1_hh.npz', mutate=multiple_pulses)
    return tmp_path


def dataset(root, monkeypatch):
    ds = GOTCHADataset(root, passes=(1,), region=tiny_region(), pulses_per_sector=3, num_train=2)
    views = ds.viewpoints
    monkeypatch.setattr(ds, 'viewpoints', lambda role: views(role)[:2] if role == 'validation' else views(role))
    return ds


def test_default_recipe_carries_no_new_keys():
    recipe = pvc.recipe_from_args(cli.parse_args(ARGV), 'rift')
    assert 'support' not in recipe and 'sh_init_degree' not in recipe


def test_box_anchors_fill_the_box_at_the_pitch():
    anchors = pvc.box_anchors(dict(kind='box', half_extents=[3.0, 1.5, 1.25], pitch=0.125), 'cpu')
    assert anchors.shape == (48 * 24 * 20, 3)
    assert torch.allclose(anchors.min(0).values, torch.tensor([-2.9375, -1.4375, -1.1875], dtype=torch.float64))
    assert torch.allclose(anchors.max(0).values, torch.tensor([2.9375, 1.4375, 1.1875], dtype=torch.float64))
    with pytest.raises(ValueError):
        pvc.box_anchors(dict(kind='box', half_extents=[3.0, 1.5, 1.2], pitch=0.125), 'cpu')


def test_cli_rejects_bad_options():
    with pytest.raises(SystemExit):
        cli.parse_args(ARGV + ['--sh-init-degree', '4'])
    with pytest.raises(SystemExit):
        cli.parse_args(ARGV + ['--support-box', '0.03', '0.015', '0.0126', '--support-pitch', '0.005'])


def test_box_field_and_initial_degree(root, monkeypatch):
    recipe = pvc.recipe_from_args(cli.parse_args(ARGV + BOX + ['--sh-init-degree', '2']), 'rift')
    assert recipe['support']['half_extents'] == [0.03, 0.015, 0.0125] and recipe['sh_init_degree'] == 2
    head = pvc.ChannelField('rift', tiny_region(), recipe, 'cpu')
    field = head.field
    assert int(field.active_mask.sum()) == 12 * 6 * 5
    assert torch.all(field.order[field.active_mask] == 2)
    positions = field.positions()[field.active_mask].detach().double()
    assert Box((0., 0., 0.), 0., (0.03, 0.015, 0.0125)).contains(positions.numpy(), margin=1e-9).all()


def test_box_support_trains_and_the_start_writes_degree_zero_only(root, monkeypatch):
    ds = dataset(root, monkeypatch)
    recipe = pvc.recipe_from_args(cli.parse_args(ARGV + BOX + ['--sh-init-degree', '2']), 'rift')
    pvc.train(ds, 'rift', recipe, root / 'box', device='cpu')
    saved = torch.load(root / 'box/checkpoint_final.pt', weights_only=False)
    assert saved['recipe']['support'] == recipe['support'] and saved['recipe']['sh_init_degree'] == 2
    state = {k[len('hh.field.'):]: v for k, v in saved['model_state_dict'].items() if k.startswith('hh.field.')}
    assert state['anchors'].shape[0] == 600 and int(state['active_mask'].sum()) >= 360
    # Bands 1..2 were trainable from update one, so they moved away from their zero start.
    active = state['active_mask']
    assert state['w_re'][active][:, 1:9].abs().sum() > 0
    # A checkpoint written with these keys rebuilds from its own recipe, as the held-out evaluator does.
    head = pvc.ChannelField('rift', ds.region, saved['recipe'], 'cpu')
    head.load_state_dict({k[3:]: v for k, v in saved['model_state_dict'].items() if k.startswith('hh.')}, strict=True)
    start = json.loads((root / 'box/initialization.json').read_text()) if (root / 'box/initialization.json').exists() else {}
    assert isinstance(start, dict)


def test_backprojection_start_leaves_higher_bands_zero(root, monkeypatch):
    from rift.gotcha_training import rift_dataset_initialization
    ds = dataset(root, monkeypatch)
    recipe = pvc.recipe_from_args(cli.parse_args(ARGV + BOX + ['--sh-init-degree', '2', '--bp-views', '2']), 'rift')
    head = pvc.ChannelField('rift', ds.region, recipe, 'cpu')
    readout = pvc.RangeReadout(ds.region, device='cpu')
    with torch.no_grad():
        rift_dataset_initialization(head, ds, readout, ds.viewpoints('train')[:2], 'hh')
    field = head.field
    active = field.active_mask
    assert field.w_re[active][:, 0].abs().sum() > 0
    assert field.w_re[active][:, 1:].abs().sum() == 0 and field.w_im[active][:, 1:].abs().sum() == 0
    assert torch.all(field.order[active] == 2)  # bands 1-2 unlocked, at zero


SPARSE = ['--prune-every', '1', '--prune-target-active', '100', '--prune-start-epoch', '1', '--prune-end-epoch', '2']


def test_sparsity_levers_are_opt_in_and_validated():
    recipe = pvc.recipe_from_args(cli.parse_args(ARGV), 'rift')
    assert 'prune' not in recipe and 'mu1_scale' not in recipe
    recipe = pvc.recipe_from_args(cli.parse_args(ARGV + SPARSE + ['--mu1-scale', '10']), 'rift')
    assert recipe['prune']['target_active'] == 100 and recipe['prune']['mode'] == 'target' and recipe['mu1_scale'] == 10.0
    for bad in (['--prune-every', '1'], ['--mu1-scale', '0'], ['--prune-every', '1', '--prune-target-active', '5',
                                                               '--optimizer', 'legacy']):
        with pytest.raises(SystemExit):
            cli.parse_args(ARGV + bad)


def test_pruning_reaches_the_target_and_mu1_scale_scales_l1(root, monkeypatch):
    ds = dataset(root, monkeypatch)
    argv = ARGV + BOX + SPARSE + ['--mu1-scale', '10', '--epochs', '2']
    recipe = pvc.recipe_from_args(cli.parse_args(argv), 'rift')
    pvc.train(ds, 'rift', recipe, root / 'sparse', device='cpu')
    start = json.loads((root / 'sparse/initialization.json').read_text())['hh']
    assert start['prune_ramp_from'] == 12 * 6 * 5 and start['mu1_scale'] == 10.0
    assert start['l1_weight'] == pytest.approx(10 * start['l1_weight_unscaled'])
    history = json.loads((root / 'sparse/history.json').read_text())
    # train.py's cubic ramp from 360 active points: epoch 1 holds the start count, epoch 2 reaches the target.
    assert history[0]['optimizer']['prune']['hh']['target'] == 360
    assert history[1]['optimizer']['prune']['hh'] == dict(target=100, active=100)
    assert history[1]['optimizer']['active_points']['hh'] == 100
    assert 1 <= history[1]['optimizer']['points_99pct_energy']['hh'] <= 100
    saved = torch.load(root / 'sparse/checkpoint_final.pt', weights_only=False)
    head = pvc.ChannelField('rift', ds.region, saved['recipe'], 'cpu')
    head.load_state_dict({k[3:]: v for k, v in saved['model_state_dict'].items() if k.startswith('hh.')}, strict=True)
    assert int(head.field.active_mask.sum()) == 100


def test_step_every_sums_sectors_into_one_step(root, monkeypatch):
    ds = dataset(root, monkeypatch)
    recipe = pvc.recipe_from_args(cli.parse_args(ARGV + BOX + ['--step-every', '2']), 'rift')
    assert recipe['step_every'] == 2 and 'step_every' not in pvc.recipe_from_args(cli.parse_args(ARGV), 'rift')
    steps = []
    original = torch.optim.AdamW.step
    monkeypatch.setattr(torch.optim.AdamW, 'step', lambda self, *a, **k: (steps.append(1), original(self, *a, **k))[1])
    pvc.train(ds, 'rift', recipe, root / 'accumulate', device='cpu')
    history = json.loads((root / 'accumulate/history.json').read_text())
    assert history[-1]['updates'] == len(ds.viewpoints('train')) and len(steps) == -(-len(ds.viewpoints('train')) // 2)
