"""RIFT-dataset NUFFT evaluation of the GOTCHA kernel and the full-native loss control (PVC lane)."""
import argparse
import math

import numpy as np
import pytest
import torch

from rift.gotcha_dataset import C
from rift.gotcha_training import native_forward
from rift.range_operator import range_forward_operator
from rift_pvc import gotcha_nufft as nufft
from rift_pvc.gotcha_batched import batched_native_forward


def camry_like(pulses=3, points=40, nf=424, seed=0):
    """Camry-scale geometry: ~10 km slant range, a cube 20 m from the phase reference, native-like grid."""
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(points, 3, generator=g, dtype=torch.float64) - .5) * 10.
    angle = torch.linspace(.2, .25, pulses, dtype=torch.float64)
    antennas = torch.stack([7000 * torch.cos(angle), 7000 * torch.sin(angle), torch.full_like(angle, 7300.)], 1)
    scene_centre = torch.tensor([20.66, -18.71, 0.], dtype=torch.float64)
    refs = torch.linalg.vector_norm(antennas - scene_centre, dim=-1)
    weights = torch.complex(torch.randn(pulses, points, generator=g, dtype=torch.float64),
                            torch.randn(pulses, points, generator=g, dtype=torch.float64))
    source = np.linspace(9.28808e9, 9.28808e9 + (nf - 1) * 1.471302e6, nf)
    selected = np.append(np.arange(0, nf, 2), nf - 1) if (nf - 1) % 2 else np.arange(0, nf, 2)
    return x, weights, antennas, refs, source, selected


def direct(x, weights, antennas, refs, frequencies, range_model):
    return torch.stack([native_forward(x, weights[p], antennas[p], frequencies, float(refs[p]),
                                       point_chunk=7, range_model=range_model) for p in range(len(antennas))])


@pytest.mark.parametrize('range_model', ['sum2', 'unit'])
def test_nufft_equals_the_direct_kernel_on_an_exact_linspace(range_model):
    x, w, antennas, refs, source, selected = camry_like()
    grid = nufft.NativeGrid(source, selected, device='cpu')
    dist = torch.linalg.vector_norm(x[None] - antennas[:, None], dim=-1)
    rendered = nufft.nufft_forward(dist, w, refs, grid, point_chunk=16, range_model=range_model)
    reference = direct(x, w, antennas, refs, grid.selected_hz, range_model)
    error = (rendered - reference).abs().square().sum() / reference.abs().square().sum()
    assert float(error) < 1e-18  # relative amplitude error below 1e-9, the operator's validation gate


def test_nufft_matches_range_forward_operator_up_to_the_known_reference_factor():
    """Same primitives as train.py's operator: GOTCHA NUFFT x conj(D) = range_forward_operator(phase_sign=-1)."""
    x, w, antennas, refs, source, selected = camry_like(pulses=1, points=12, nf=64)
    x, antenna = x * 1e-3, torch.tensor([[1.2, .3, .4]], dtype=torch.float64)
    ref = torch.tensor([1.0], dtype=torch.float64)
    grid = nufft.NativeGrid(source, selected, device='cpu')
    dist = torch.linalg.vector_norm(x[None] - antenna[:, None], dim=-1)
    ours = nufft.nufft_forward(dist, w, ref, grid, point_chunk=5, range_model='sum2')[0]
    f = torch.as_tensor(source)
    theirs = range_forward_operator(f, 2 * math.pi * f / C, antenna, antenna, x, w[0], phase_sign=-1.0,
                                    freq_indices=torch.as_tensor(selected), range_model='sum2')[:, 0, 0]
    d = torch.exp((4j * math.pi / C) * grid.selected_hz * float(ref))
    torch.testing.assert_close(ours * d.conj(), theirs, rtol=1e-8, atol=0)


def test_float32_native_frequencies_cost_milliradians_on_the_reference_relative_path():
    x, w, antennas, refs, source, selected = camry_like()
    stored = source.astype(np.float32).astype(np.float64)   # the shards' storage
    grid = nufft.NativeGrid(stored, selected, device='cpu')
    assert 0 < grid.max_linspace_deviation_hz < 1024
    dist = torch.linalg.vector_norm(x[None] - antennas[:, None], dim=-1)
    rendered = nufft.nufft_forward(dist, w, refs, grid, point_chunk=16, range_model='sum2')
    reference = direct(x, w, antennas, refs, grid.selected_hz, 'sum2')
    relative = float(((rendered - reference).abs().square().sum() / reference.abs().square().sum()).sqrt())
    bound = 4 * math.pi / C * 30. * grid.max_linspace_deviation_hz   # |dr| <= ~30 m
    assert relative < bound < 2e-3


def test_adjoint_is_the_adjoint_of_the_forward():
    x, w, antennas, refs, source, selected = camry_like(pulses=4, points=30)
    grid = nufft.NativeGrid(source, selected, device='cpu')
    g = torch.Generator().manual_seed(3)
    v = torch.complex(torch.randn(4, len(selected), generator=g, dtype=torch.float64),
                      torch.randn(4, len(selected), generator=g, dtype=torch.float64))
    common = w[0]
    dist = torch.linalg.vector_norm(x[None] - antennas[:, None], dim=-1)
    forward = nufft.nufft_forward(dist, common[None].expand(4, -1), refs, grid, point_chunk=8, range_model='sum2')
    adjoint = nufft.nufft_adjoint(x, v, antennas, refs, grid, point_chunk=8, range_model='sum2')
    lhs, rhs = (forward.conj() * v).sum(), (common.conj() * adjoint).sum()
    assert abs(complex(lhs - rhs)) <= 1e-12 * abs(complex(lhs))


@pytest.mark.parametrize('range_model', ['sum2', 'unit'])
def test_sector_render_gives_exact_per_pulse_distance_gradients(range_model):
    x0, w0, antennas, refs, source, selected = camry_like(points=25)
    grid = nufft.NativeGrid(source, selected, device='cpu')
    scale = 1e-11 if range_model == 'sum2' else 1.
    g = torch.Generator().manual_seed(5)
    target = scale * torch.complex(torch.randn(3, len(selected), generator=g, dtype=torch.float64),
                                   torch.randn(3, len(selected), generator=g, dtype=torch.float64))
    def loss(out):
        return (out - target).abs().square().sum()
    per_pulse = []
    for p in range(3):
        x = x0.clone().requires_grad_()
        out = nufft.sector_render(x, w0[p:p+1], antennas[p:p+1], refs[p:p+1], grid, point_chunk=9,
                                  range_model=range_model)
        per_pulse.append(torch.autograd.grad((out[0] - target[p]).abs().square().sum(), x)[0])
    x, w = x0.clone().requires_grad_(), w0.clone().requires_grad_()
    pulse_grad_d = torch.zeros(3, 25, dtype=torch.float64)
    loss(nufft.sector_render(x, w, antennas, refs, grid, point_chunk=9, range_model=range_model,
                             pulse_grad_d=pulse_grad_d)).backward()
    unit = x0[None] - antennas[:, None]
    unit = unit / torch.linalg.vector_norm(unit, dim=-1, keepdim=True)
    for p in range(3):
        torch.testing.assert_close(pulse_grad_d[p, :, None] * unit[p], per_pulse[p], rtol=1e-9,
                                   atol=1e-12 * float(per_pulse[p].abs().max()))
    torch.testing.assert_close(x.grad, sum(per_pulse), rtol=1e-9, atol=1e-12 * float(x.grad.abs().max()))
    # Same gradients as the direct kernel's custom backward, to the NUFFT's accuracy.
    xd, wd = x0.clone().requires_grad_(), w0.clone().requires_grad_()
    loss(batched_native_forward(xd, wd, antennas, refs, grid.selected_hz, point_chunk=9,
                                range_model=range_model)).backward()
    torch.testing.assert_close(w.grad, wd.grad, rtol=1e-7, atol=1e-9 * float(wd.grad.abs().max()))
    torch.testing.assert_close(x.grad, xd.grad, rtol=1e-6, atol=1e-8 * float(xd.grad.abs().max()))


def test_grid_gates():
    _, _, _, _, source, selected = camry_like()
    bent = source.copy()
    bent[5] += 0.05 * (source[1] - source[0])
    with pytest.raises(ValueError, match='uniform'):
        nufft.NativeGrid(bent, selected, device='cpu')
    with pytest.raises(ValueError):
        nufft.NativeGrid(source, [0, len(source)], device='cpu')


def test_recipe_keys_are_opt_in_and_adaptive_rift_only():
    default = argparse.Namespace(forward_evaluation='direct', loss_domain='roi_projected')
    control = argparse.Namespace(forward_evaluation='nufft', loss_domain='full_native')
    assert nufft.control_recipe(default, 'rift') == {} == nufft.control_recipe(argparse.Namespace(), 'rift')
    keys = nufft.control_recipe(control, 'rift')
    assert keys['forward_evaluation'] == nufft.NUFFT and keys['loss_domain'] == nufft.FULL
    assert keys['loss_normalization'] == 'train_only_mean_full_native_power'
    assert keys['nufft']['oversample'] == 2 and keys['nufft']['kernel_width'] == 20
    assert nufft.output_suffix(keys) == '_nufft_full_native' and nufft.output_suffix({}) == ''
    only_loss = nufft.control_recipe(argparse.Namespace(forward_evaluation='direct', loss_domain='full_native'), 'rift')
    assert 'forward_evaluation' not in only_loss and nufft.forward_evaluation(only_loss) == nufft.DIRECT
    for method in ('rift_grid', 'isotropic', 'mfbp'):
        with pytest.raises(ValueError, match='adaptive RIFT only'):
            nufft.control_recipe(control, method)


# --- The control inside the PVC trainer (tiny native fixture, CPU) ---------------------------------------------

import signal

import train_gotcha_dataset_pvc as cli
from rift.gotcha_dataset import GOTCHADataset
from rift.gotcha_training import rift_dataset_initialization
from rift_pvc import gotcha_training as pvc
from tests.test_gotcha_dataset import tiny_region, write_shard
from tests.test_gotcha_pulse_sampling import multiple_pulses

ARGV = ['--epochs', '2', '--granularity', '2', '--max-points', '15', '--sh-degree', '1',
        '--pulses-per-sector', '3', '--point-chunk', '3', '--checkpoint-every', '1']
CONTROL = ['--forward-evaluation', 'nufft', '--loss-domain', 'full_native']


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv('RIFT_ACCELERATOR', 'cpu')
    write_shard(tmp_path/'New_Transfer/shards/pass1_hh.npz', mutate=multiple_pulses)
    return tmp_path


def fixture_dataset(root, monkeypatch, validation=2):
    ds = GOTCHADataset(root, passes=(1,), region=tiny_region(), pulses_per_sector=3, num_train=2)
    views = ds.viewpoints
    monkeypatch.setattr(ds, 'viewpoints', lambda role: views(role)[:validation] if role == 'validation' else views(role))
    return ds


def fixture_recipe(extra=(), *, event=False):
    recipe = pvc.recipe_from_args(cli.parse_args(ARGV + list(extra)), 'rift')
    if event:
        # Tiny-fixture thresholds so the epoch-end refinement acts (as test_gotcha_batched_pvc does).
        recipe['optimizer_schedule'] = dict(recipe['optimizer_schedule'], min_spatial_exposure=1,
                                            min_angular_exposure=1, spatial_fraction=.5, angular_fraction=.5)
        recipe['max_level'] = 2
    return recipe


def assert_states_close(a, b, rtol, atol):
    assert a.keys() == b.keys()
    for key, value in a.items():
        if value.is_floating_point() or value.is_complex():
            torch.testing.assert_close(b[key], value, rtol=rtol, atol=atol, msg=key)
        else:
            assert torch.equal(b[key], value), key


def test_default_recipe_carries_no_control_keys(monkeypatch):
    monkeypatch.setenv('RIFT_ACCELERATOR', 'cpu')
    default = fixture_recipe()
    assert not nufft.control_keys(default) and nufft.output_suffix(default) == ''
    assert default['loss_normalization'] == 'train_only_mean_projected_power'
    assert not {'forward_evaluation', 'loss_domain', 'nufft', 'checkpoint_selection'} & set(default)
    control = fixture_recipe(CONTROL)
    assert {k: v for k, v in control.items() if k not in ('forward_evaluation', 'loss_domain', 'nufft',
                                                         'checkpoint_selection', 'loss_normalization')} \
        == {k: v for k, v in default.items() if k != 'loss_normalization'}
    with pytest.raises(SystemExit):
        cli.parse_args(ARGV + ['--loss-domain', 'range_bins'])


def test_control_batched_trainer_matches_its_per_pulse_loop(root, monkeypatch):
    ds = fixture_dataset(root, monkeypatch)
    batched = fixture_recipe(CONTROL + ['--refine-every', '1'], event=True)
    loop = fixture_recipe(CONTROL + ['--refine-every', '1', '--sector-execution', 'loop'], event=True)
    pvc.train(ds, 'rift', loop, root/'loop', device='cpu')
    pvc.train(ds, 'rift', batched, root/'batched', device='cpu')
    a = torch.load(root/'loop/checkpoint_final.pt', weights_only=False)
    b = torch.load(root/'batched/checkpoint_final.pt', weights_only=False)
    for key in ('hh.field.active_mask', 'hh.field.order', 'hh.field.level'):
        assert torch.equal(a['model_state_dict'][key], b['model_state_dict'][key]), key
    assert int(a['model_state_dict']['hh.field.active_mask'].sum()) > 8   # an epoch-end event happened
    assert_states_close(a['model_state_dict'], b['model_state_dict'], rtol=1e-3, atol=1e-5)
    for ea, eb in zip(a['history'], b['history']):
        va, vb = ea['validation'], eb['validation']
        assert va['selection_domain'] == vb['selection_domain'] == nufft.FULL
        assert vb['selection_rel_mse'] == vb['full_native_complex_rel_mse']
        assert vb['selection_rel_mse'] == pytest.approx(va['selection_rel_mse'], rel=1e-4)
        assert vb['pooled_rel_mse'] == pytest.approx(va['pooled_rel_mse'], rel=1e-4)
    assert b['best_val'] == min(e['validation']['selection_rel_mse'] for e in b['history'])
    start = torch.load(root/'batched/checkpoint_final.pt', weights_only=False)['initialization']['hh']
    assert start['forward_evaluation'] == nufft.NUFFT and start['loss_domain'] == nufft.FULL
    assert b['training_statistics']['hh']['loss_domain'] == nufft.FULL


def test_control_start_reduces_to_the_default_start_and_nufft_agrees_with_direct(root, monkeypatch):
    ds = fixture_dataset(root, monkeypatch)
    views = ds.viewpoints('train')
    def start(recipe, fn):
        torch.manual_seed(0)
        heads = torch.nn.ModuleDict({'hh': pvc.ChannelField('rift', ds.region, recipe, 'cpu')})
        nufft.attach_grids(heads, ds, recipe, 'cpu')
        record = fn(heads['hh'], ds, pvc.RangeReadout(ds.region, device='cpu'), views, 'hh')
        return heads['hh'], record
    default = fixture_recipe()
    h0, r0 = start(default, rift_dataset_initialization)
    h1, r1 = start(default, nufft.control_initialization)
    assert_states_close(h0.state_dict(), h1.state_dict(), rtol=1e-10, atol=0)
    for key in ('m1', 'm2', 'l1_weight', 'sh_degree_weight', 'coefficient_gauge'):
        assert r1[key] == pytest.approx(r0[key], rel=1e-10), key
    full_direct = fixture_recipe(['--loss-domain', 'full_native'])
    full_nufft = fixture_recipe(CONTROL)
    h2, r2 = start(full_direct, nufft.control_initialization)
    h3, r3 = start(full_nufft, nufft.control_initialization)
    # The fixture's native bins sit up to 124 Hz off their linspace, which the NUFFT reconstructs (measured 5e-8).
    mask = h2.field.active_mask
    for name in ('w_re', 'w_im'):
        a, b = getattr(h2.field, name)[mask].detach().double(), getattr(h3.field, name)[mask].detach().double()
        assert float((a - b).norm() / a.norm()) < 1e-6, name
    g2, g3 = h2.gain.gain_value(), h3.gain.gain_value()
    assert abs(g3 - g2) < 1e-7 * abs(g2)
    assert r3['m1'] == pytest.approx(r2['m1'], rel=1e-6)


def test_full_native_statistics_and_objective(root, monkeypatch):
    ds = fixture_dataset(root, monkeypatch)
    readout = pvc.RangeReadout(ds.region, device='cpu')
    full, projected = nufft.full_native_statistics(ds, readout)['hh'], pvc.training_statistics(ds, readout)['hh']
    assert full['projected_energy'] == pytest.approx(projected['energy'], rel=1e-12)
    assert full['projected_count'] == projected['count'] and full['range_peak'] == projected['range_peak']
    energy = count = 0
    for p, sector in ds.viewpoints('train'):
        for o in ds.observations(p, sector, 'hh'):
            energy, count = energy + float((abs(o.response) ** 2).sum()), count + o.response.size
    assert full['energy'] == pytest.approx(energy, rel=1e-12) and full['count'] == count
    assert full['mean_power'] == full['energy'] / full['count']
    o = next(iter(ds.observations(*ds.viewpoints('train')[0], 'hh')))
    r = readout.for_observation(o)
    pred = torch.zeros(len(o.frequencies_hz), dtype=torch.complex128)
    loss, _ = pvc.objective('rift', pred, o, r, full)
    assert float(loss) == pytest.approx(float((abs(o.response) ** 2).mean()) / full['mean_power'], rel=1e-12)


def test_control_interrupt_resume_is_exact_and_other_recipes_are_refused(root, monkeypatch):
    ds = fixture_dataset(root, monkeypatch, validation=1)
    control = fixture_recipe(CONTROL)
    pvc.train(ds, 'rift', control, root/'full', device='cpu')
    step = torch.optim.AdamW.step
    def interrupt(opt, *a, **kw):
        result = step(opt, *a, **kw)
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        return result
    with monkeypatch.context() as m:
        m.setattr(torch.optim.AdamW, 'step', interrupt)
        assert pvc.train(ds, 'rift', control, root/'resumed', device='cpu')['status'] == 'interrupted'
    latest = root/'resumed/checkpoint_latest.pt'
    for other in (fixture_recipe(), fixture_recipe(['--loss-domain', 'full_native'])):
        with pytest.raises(ValueError, match='recipe'):
            pvc.train(ds, 'rift', other, root/'resumed', device='cpu', resume=latest)
    pvc.train(ds, 'rift', control, root/'resumed', device='cpu', resume=latest)
    full = torch.load(root/'full/checkpoint_final.pt', weights_only=False)
    resumed = torch.load(root/'resumed/checkpoint_final.pt', weights_only=False)
    assert full['history'] == resumed['history']
    for key, value in full['model_state_dict'].items():
        torch.testing.assert_close(value, resumed['model_state_dict'][key], rtol=0, atol=0)
