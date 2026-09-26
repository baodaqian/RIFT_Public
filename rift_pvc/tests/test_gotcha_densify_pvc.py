"""RIFT-on-GOTCHA tuning campaign A41: trilinear densification and the pass-held-out unit split."""
import json

import pytest
import torch

import train_gotcha_dataset_pvc as cli
from rift.gotcha_dataset import GOTCHADataset, sector_split
from rift.sparse_scene import AdaptivePointSHScene
from rift_pvc import gotcha_training as pvc
from rift_pvc.gotcha_densify import scale_learning_rates, trilinear_densify
from rift_pvc.gotcha_unit_split import apply_unit_split, unit_split_ids
from tests.test_gotcha_dataset import tiny_region, write_shard
from tests.test_gotcha_pulse_sampling import multiple_pulses

ARGV = ['--epochs', '2', '--granularity', '2', '--max-points', '600', '--sh-degree', '2', '--probe-every', '1',
        '--pulses-per-sector', '3', '--point-chunk', '64', '--checkpoint-every', '1']
BOX = ['--support-box', '0.03', '0.015', '0.0125', '--support-pitch', '0.005']


def two_point_scene(capacity=32, values=(1.0, 3.0)):
    """Two parents on a 2 x 1 x 1 lattice of pitch 1 filling the box [0, 2] x [0, 1] x [0, 1]."""
    anchors = torch.tensor([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]])
    scene = AdaptivePointSHScene(anchors, 0.5, 'cpu', max_degree=1, init_degree=0, capacity=capacity,
                                 enforce_support_bounds=True, compact_sh_eval=True)
    with torch.no_grad():
        scene.w_re[:2, 0] = torch.tensor(values)
        scene.w_im[:2, 0] = 0.0
        scene.w_re[:2, 1:] = 0.0
    return scene


def child_value(scene, anchor):
    slot = ((scene.anchors - torch.tensor(anchor)).abs().sum(-1) < 1e-6) & scene.active_mask
    assert int(slot.sum()) == 1
    return float(scene.w_re.detach()[slot][0, 0])


def test_children_take_the_trilinear_value_of_the_parent_lattice():
    scene = two_point_scene()
    report = trilinear_densify(scene, max_active=32, weight_scale=0.125, normalize='volume')
    assert report['status'] == 'densified' and report['parents'] == 2 and report['active_after'] == 16
    assert report['level_to'] == 1 and report['pitch_to_m'] == pytest.approx(0.5)
    # Child of parent 0 towards parent 1: 3/4 self + 1/4 neighbour along x; the y and z neighbours are
    # outside the support (empty, zero), so only the 3/4 self weight survives on those axes.
    inner = (0.75 * 1.0 + 0.25 * 3.0) * 0.75 * 0.75
    assert child_value(scene, [0.75, 0.25, 0.25]) == pytest.approx(inner / 8)
    assert child_value(scene, [0.25, 0.25, 0.25]) == pytest.approx(0.75 ** 3 * 1.0 / 8)
    assert child_value(scene, [1.25, 0.75, 0.75]) == pytest.approx((0.75 * 3.0 + 0.25 * 1.0) * 0.75 * 0.75 / 8)
    active = scene.active_mask
    assert torch.allclose(scene.cell_half[active], torch.full((16, 1), 0.25))
    assert torch.all(scene.level[active] == 1) and torch.all(scene.delta_raw[active] == 0)
    # A second event works on the finer lattice.
    report = trilinear_densify(scene, max_active=32, weight_scale=0.125, max_level=1)
    assert report['status'] == 'at_max_level'


def test_energy_normalization_restores_the_parent_coefficient_energy():
    scene = two_point_scene()
    report = trilinear_densify(scene, max_active=32, normalize='energy')
    assert report['normalize'] == 'energy'
    energy = float((scene.w_re.detach().square() + scene.w_im.detach().square())[scene.active_mask].sum())
    assert energy == pytest.approx(1.0 ** 2 + 3.0 ** 2)
    ratio = [child_value(scene, [0.75, 0.25, 0.25]) / child_value(scene, [0.25, 0.25, 0.25])]
    assert ratio[0] == pytest.approx((0.75 * 1.0 + 0.25 * 3.0) / 0.75)   # shape is the trilinear one


def test_inherit_start_preserves_the_coherent_render():
    scene = two_point_scene()
    with torch.no_grad():
        scene.delta_raw[:2] = torch.tensor([[0.3, -0.2, 0.1], [-0.4, 0.5, -0.6]])
        scene.w_im[:2, 0] = torch.tensor([0.5, -1.0])
    frequencies = torch.linspace(9.3e9, 9.9e9, 7)
    antenna = torch.tensor([40.0, 25.0, 30.0])

    def render():
        points, weights = scene.active_scatterers(torch.tensor([[0.8]]), torch.tensor([[0.4]]))
        return pvc.native_forward(points.detach(), weights.detach(), antenna, frequencies, 50.0)

    before = render()
    report = trilinear_densify(scene, max_active=32, init='inherit')
    assert report['init'] == 'inherit' and report['active_after'] == 16
    after = render()
    assert torch.allclose(after, before, rtol=1e-4, atol=1e-6 * float(before.abs().max()))
    assert int((scene.w_re.detach()[scene.active_mask].abs().sum(-1) > 0).sum()) == 2   # one heir per parent


def test_budget_keeps_the_most_energetic_parents_and_the_floor_drops_empty_ones():
    scene = two_point_scene()
    report = trilinear_densify(scene, max_active=8)
    assert report['parents'] == 1 and report['dropped_over_budget'] == 1 and report['active_after'] == 8
    assert torch.all(scene.anchors[scene.active_mask][:, 0] > 1.0)   # children of the stronger parent
    scene = two_point_scene(values=(1e-6, 3.0))
    report = trilinear_densify(scene, max_active=32, energy_floor=1e-3)
    assert report['dropped_below_floor'] == 1 and report['parents'] == 1


def test_mixed_levels_are_refused_and_adam_rows_restart():
    scene = two_point_scene()
    optimizer = torch.optim.AdamW([scene.w_re, scene.w_im, scene.delta_raw], lr=1e-3)
    (scene.w_re.square().sum() + scene.delta_raw.square().sum()).backward()
    optimizer.step()
    trilinear_densify(scene, max_active=32, optimizer=optimizer)
    assert torch.all(optimizer.state[scene.w_re]['exp_avg'][scene.active_mask] == 0)
    with torch.no_grad():
        scene.level[scene.active_mask.nonzero()[0]] = 0
    with pytest.raises(ValueError):
        trilinear_densify(scene, max_active=32)


def test_learning_rate_factor_moves_the_cosine_schedule():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scheduler.step()
    before = optimizer.param_groups[0]['lr']
    scale_learning_rates(optimizer, scheduler, 0.5)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(before / 2)
    scheduler.step()
    reference = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=5e-4)
    twin = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(reference, T_0=10, T_mult=2, eta_min=5e-7)
    twin.step(), twin.step()
    assert optimizer.param_groups[0]['lr'] == pytest.approx(reference.param_groups[0]['lr'])


def test_unit_split_ids_are_every_fourth_unsealed_id_and_never_sealed():
    selected, heldout = unit_split_ids()
    parent = sector_split()
    assert len(selected) == 77 and len(heldout) == 38 and set(heldout) <= set(selected)
    assert not set(selected) & set(parent['test'])
    assert selected == sorted(parent['train'] + parent['validation'])[::4]
    assert unit_split_ids() == (selected, heldout)


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv('RIFT_ACCELERATOR', 'cpu')
    for p in (1, 2):
        write_shard(tmp_path / f'New_Transfer/shards/pass{p}_hh.npz', pass_id=p, mutate=multiple_pulses)
    return tmp_path


def test_unit_split_relabels_rows_and_keeps_test_sealed(root):
    ds = GOTCHADataset(root, passes=(1, 2), region=tiny_region(), pulses_per_sector=3, num_train=2)
    before = ds.identity
    record = apply_unit_split(ds, stride=4, heldout_pass=2, heldout_fraction=0.5)
    selected, heldout = unit_split_ids()
    assert ds.identity != before and ds.contract['split']['schema'] == record['schema']
    assert ds.viewpoints('validation') == [(2, s) for s in heldout]
    assert ds.viewpoints('train') == [(1, s) for s in selected] + [(2, s) for s in selected if s not in heldout]
    shard = ds.shards[2, 'hh']
    test_row = int(shard.sector_rows[sector_split()['test'][0]][0])
    with pytest.raises(PermissionError):
        shard.read(test_row)
    unselected = next(s for s in sector_split()['train'] if s not in selected)
    with pytest.raises(PermissionError):
        shard.read(int(shard.sector_rows[unselected][0]))
    assert shard.read(int(shard.sector_rows[heldout[0]][0])).sector_id == heldout[0]


def test_densify_recipe_is_opt_in_and_validated():
    recipe = pvc.recipe_from_args(cli.parse_args(ARGV), 'rift')
    assert 'densify' not in recipe and 'unit_split' not in recipe
    assert recipe['optimizer_schedule']['spatial_fraction'] > 0
    recipe = pvc.recipe_from_args(cli.parse_args(ARGV + ['--densify-epochs', '1', '--densify-lr-rule', 'fixed',
                                                         '--densify-lr-factor', '0.5']), 'rift')
    assert recipe['densify']['epochs'] == [1] and recipe['densify']['max_active'] == 600
    assert recipe['densify']['lr_rule'] == 'fixed' and recipe['densify']['init'] == 'trilinear'
    assert recipe['optimizer_schedule']['spatial_fraction'] == 0.0
    for bad in (['--densify-epochs', '0'], ['--densify-epochs', '1', '--densify-max-active', '601'],
                ['--densify-epochs', '1', '--optimizer', 'legacy'], ['--densify-epochs', '1', '--densify-lr-factor', '0'],
                ['--unit-split-stride', '4', '--unit-split-heldout-pass', '5', '--passes', '1', '2'],
                ['--densify-epochs', '1']):   # the default curvature rule needs the full-native loss
        with pytest.raises(SystemExit):
            cli.parse_args(ARGV + bad)


def test_densify_trains_end_to_end_on_the_unit_split(root):
    argv = ARGV + BOX + ['--passes', '1', '2', '--unit-split-stride', '4', '--unit-split-heldout-pass', '2',
                         '--densify-epochs', '1', '--densify-max-active', '600', '--densify-lr-rule', 'fixed',
                         '--densify-lr-factor', '0.5',
                         '--train-eval-every', '1', '--lr', '3e-4', '--pos-lr', '3e-4', '--loss-domain', 'full_native']
    args = cli.parse_args(argv)
    ds = GOTCHADataset(root, passes=(1, 2), region=tiny_region(), pulses_per_sector=3, num_train=2)
    apply_unit_split(ds, stride=4, heldout_pass=2, heldout_fraction=0.5)
    recipe = pvc.recipe_from_args(args, 'rift')
    pvc.train(ds, 'rift', recipe, root / 'dense', device='cpu')
    history = json.loads((root / 'dense/history.json').read_text())
    event = history[0]['optimizer']['densify']
    assert event['hh']['status'] == 'densified' and event['hh']['active_before'] == 360
    assert event['hh']['active_after'] == 8 * event['hh']['parents'] <= 600
    assert event['learning_rates_after'][0] == pytest.approx(history[0]['optimizer']['learning_rates'][0] * 0.5, rel=0.2)
    for entry in history:
        fit = entry['train']
        assert fit['units'] == len(ds.viewpoints('train')) and 0 < fit['full_native_rel_mse'] < 10
        assert abs(fit['correlation']) <= 1 and entry['train_running']['units'] == len(ds.viewpoints('train'))
        assert entry['validation_fit']['units'] == len(ds.viewpoints('validation'))
    saved = torch.load(root / 'dense/checkpoint_final.pt', weights_only=False)
    head = pvc.ChannelField('rift', ds.region, saved['recipe'], 'cpu')
    head.load_state_dict({k[3:]: v for k, v in saved['model_state_dict'].items() if k.startswith('hh.')}, strict=True)
    assert torch.all(head.field.level[head.field.active_mask] == 1)


def test_curvature_rule_matches_the_fit_check_and_caps_the_stability_number(root):
    import importlib.util
    spec = importlib.util.spec_from_file_location('fit_check', 'scripts_pvc/gotcha_rift_fit_check.py')
    fit_check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fit_check)
    argv = ARGV + BOX + ['--passes', '1', '2', '--loss-domain', 'full_native', '--densify-epochs', '1',
                         '--densify-max-active', '600', '--densify-lr-target', '1e-3', '--densify-curvature-units', '1',
                         '--densify-curvature-iterations', '6']
    ds = GOTCHADataset(root, passes=(1, 2), region=tiny_region(), pulses_per_sector=3, num_train=2)
    recipe = pvc.recipe_from_args(cli.parse_args(argv), 'rift')
    head = pvc.ChannelField('rift', ds.region, recipe, 'cpu')
    readout = pvc.RangeReadout(ds.region, device='cpu')
    view = ds.viewpoints('train')[0]
    for o in ds.observations(*view, 'hh'):
        head.initialize_scale(o, readout.for_observation(o))
    ours = pvc.update_curvature(head, ds, readout, view, 'hh', 2.5, 'rift', iterations=6)['lambda_max']
    theirs = fit_check.sector_curvature(head, readout, ds, view, 'hh', 2.5, False, 6)['lambda_max']
    assert ours == pytest.approx(theirs, rel=1e-6)
    pvc.train(ds, 'rift', recipe, root / 'curv', device='cpu')
    history = json.loads((root / 'curv/history.json').read_text())
    event = history[0]['optimizer']['densify']['curvature']
    assert event['factor'] < 1 and event['stability_after'] == pytest.approx(1e-3)
    assert history[0]['optimizer']['densify']['learning_rates_after'][0] == pytest.approx(
        history[0]['optimizer']['learning_rates'][0] * event['factor'], rel=0.3)
