"""PVC SpINR batched native kernel and update: equivalence with the per-pulse runtime."""
import copy

import numpy as np
import pytest
import torch

import train_spinr_style as root
from rift import spinr_gotcha_training as original_runtime
from rift.gotcha_dataset import GOTCHADataset
from rift.spinr_native import NativeKernel, loss_and_field_vjp
from rift.spinr_style import gauss_legendre_cell_grid
from rift_pvc import spinr_gotcha_training as runtime
from rift_pvc import spinr_native_batched
from rift_pvc.spinr_native_batched import PULSE_EXECUTION, BatchedNativeKernel, batched_loss_and_field_vjp, shard_groups
from tests.test_gotcha_dataset import write_shard, tiny_region
from tests.test_spinr_gotcha import TinyField, config, native_dataset  # noqa: F401 (fixture)


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv('RIFT_ACCELERATOR', 'cpu')


def affine_frequencies(arrays, _metadata):
    n = len(arrays['frequencies_hz'])
    arrays['frequencies_hz'] = 9e9 + np.arange(n, dtype=np.float64) * 5e6
    arrays['response'] = (np.exp(.3j * np.arange(n)) * (1 + .1 * np.arange(360)[:, None])).astype(np.complex64)


def stride2_like(arrays, _metadata):
    n = len(arrays['frequencies_hz'])
    f = 9e9 + np.arange(n, dtype=np.float64) * 6e6
    f[-1] = f[-2] + 3e6                     # affine prefix plus the retained native endpoint
    arrays['frequencies_hz'] = f


@pytest.fixture(params=['ragged', 'affine', 'endpoint'])
def kernel_case(request, tmp_path):
    mutate = {'affine': affine_frequencies, 'endpoint': stride2_like, 'ragged': None}[request.param]
    write_shard(tmp_path/'New_Transfer/shards/pass1_hh.npz', nf=32, mutate=mutate)
    ds = GOTCHADataset(tmp_path, passes=(1,), region=tiny_region())
    plan = original_runtime.PulsePlan(ds)
    observations = [plan.read(i) for i in range(5)]
    return request.param, ds, observations


def test_batched_kernel_matches_per_pulse_kernel(kernel_case):
    mode, ds, observations = kernel_case
    torch.manual_seed(0)
    points, volumes = gauss_legendre_cell_grid(3, nodes_per_cell=1, support_m=ds.region.half_extent_m, dtype=torch.float64)
    field = torch.randn(len(points), dtype=torch.float64)
    batched = BatchedNativeKernel(observations, ds.region, point_tile=4, element_budget=64)
    assert batched.affine == (mode == 'affine') and batched.point_tile == 1 and batched.pulses == 5
    assert batched.closed_form_endpoint == (mode == 'endpoint')
    singles = [NativeKernel(o, ds.region, point_tile=4) for o in observations]
    for i, single in enumerate(singles):
        assert single.affine == batched.affine
        n = int(batched.counts[i])
        assert torch.equal(batched.bin_ids[i, :n], single.bin_ids) and not bool(batched.valid[i, n:].any())
    full = batched.render(points, field, volumes, 1.7, selected=False)
    bins = batched.render(points, field, volumes, 1.7)
    responses = np.stack([o.response for o in observations])
    targets = batched.target_bins(responses)
    cotangent = torch.randn(bins.shape, dtype=torch.complex128) * batched.valid
    vjp = batched.field_vjp(points, volumes, 1.7, cotangent)
    expected_vjp = torch.zeros_like(vjp)
    for i, single in enumerate(singles):
        n = int(batched.counts[i])
        torch.testing.assert_close(full[i], single.render(points, field, volumes, 1.7, selected=False), rtol=1e-12, atol=1e-18)
        torch.testing.assert_close(bins[i, :n], single.render(points, field, volumes, 1.7), rtol=1e-11, atol=1e-18)
        assert not bool(bins[i, n:].any())
        torch.testing.assert_close(targets[i, :n], single.target_bins(observations[i].response), rtol=1e-12, atol=0)
        expected_vjp += single.field_vjp(points, volumes, 1.7, cotangent[i, :n])
    torch.testing.assert_close(vjp, expected_vjp, rtol=1e-10, atol=1e-18)
    losses, gradient = batched_loss_and_field_vjp(batched, points, field, volumes, 1.7, responses, 1e-3, .25)
    expected_gradient = torch.zeros_like(gradient)
    for i, single in enumerate(singles):
        loss, g = loss_and_field_vjp(single, points, field, volumes, 1.7, observations[i].response, 1e-3)
        assert float(losses[i]) == pytest.approx(loss, rel=1e-10)
        expected_gradient += g * .25
    torch.testing.assert_close(gradient, expected_gradient, rtol=1e-10, atol=1e-18)
    with pytest.raises(ValueError, match='share'):
        replaced = copy.copy(observations[1])
        object.__setattr__(replaced, 'frequencies_hz', observations[1].frequencies_hz * 1.0001)
        BatchedNativeKernel([observations[0], replaced], ds.region)


def _heads_and_optimizer(dataset, seed=1):
    torch.manual_seed(seed)
    heads = torch.nn.ModuleDict({p: TinyField(support_m=dataset.region.half_extent_m) for p in dataset.polarizations})
    with torch.no_grad():
        for h in heads.values():
            h.coefficients.add_(torch.randn(4, dtype=torch.float64) * .1)
    optimizer = torch.optim.Adam(heads.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    return heads, optimizer


def test_batch_update_and_evaluate_match_the_per_pulse_loop(native_dataset):
    ds = native_dataset
    recipe = runtime.recipe_from_config(config(), ds)
    assert recipe['pulse_execution'] == PULSE_EXECUTION
    assert {k: v for k, v in recipe.items() if k != 'pulse_execution'} == original_runtime.recipe_from_config(config(), ds)
    plan = runtime.PulsePlan(ds)
    points, volumes = gauss_legendre_cell_grid(recipe['grid_size'], nodes_per_cell=recipe['nodes_per_cell'],
                                               support_m=ds.region.half_extent_m, dtype=torch.float64)
    stats = runtime.training_statistics(plan)
    ids = plan.order(0, 42)[:len(plan.records)]           # every TRAIN pulse of both passes and both channels
    assert len(shard_groups(plan, ids)) == 4 and sorted(sum(shard_groups(plan, ids), [])) == sorted(int(i) for i in ids)
    results = {}
    for name, update in (('loop', runtime.batch_update_loop), ('batched', runtime.batch_update)):
        heads, optimizer = _heads_and_optimizer(ds)
        scales = runtime.initialize_scales(heads, plan, points, volumes, recipe, 'cpu')
        loss, norm, active = update(heads, optimizer, plan, ids, points, volumes, stats, scales, recipe, 'cpu')
        results[name] = dict(loss=loss, norm=norm, active=active, scales=scales,
                             params={k: v.detach().clone() for k, v in heads.state_dict().items()},
                             validation=runtime.evaluate(heads, ds, points, volumes, stats, scales, recipe, 'cpu'),
                             validation_loop=runtime.evaluate_loop(heads, ds, points, volumes, stats, scales, recipe, 'cpu'))
    a, b = results['loop'], results['batched']
    assert a['active'] == b['active'] and a['scales'] == b['scales']
    assert b['loss'] == pytest.approx(a['loss'], rel=1e-10) and b['norm'] == pytest.approx(a['norm'], rel=1e-9)
    for key, value in a['params'].items():
        torch.testing.assert_close(b['params'][key], value, rtol=1e-9, atol=1e-12)
    for source in (a, b):
        for key in ('pooled_rel_mse', 'full_native_complex_rel_mse', 'native_spectral_objective'):
            assert source['validation'][key] == pytest.approx(source['validation_loop'][key], rel=1e-9)
        for pol in ds.polarizations:
            assert source['validation']['by_polarization'][pol]['pulses'] == source['validation_loop']['by_polarization'][pol]['pulses']
            assert source['validation']['by_polarization'][pol]['samples'] == source['validation_loop']['by_polarization'][pol]['samples']


@pytest.mark.parametrize('stop_after', [1, 2])
def test_batched_run_interrupt_resume_and_recipe_gate(native_dataset, tmp_path, monkeypatch, stop_after):
    monkeypatch.setattr(runtime, 'SpinrStyleINR', TinyField)
    monkeypatch.setattr(original_runtime, 'SpinrStyleINR', TinyField)
    options = dict(config(), cosine_epochs=150)
    runtime.run(native_dataset, tmp_path/'full', options, device='cpu')
    expected = root.load_tensor_checkpoint(tmp_path/'full/checkpoint_latest.pt', map_location='cpu')
    assert expected['recipe']['pulse_execution'] == PULSE_EXECUTION and expected['epoch'] == 2
    counter = {'n': 0}
    update = runtime.batch_update
    def counted(*args):
        counter['n'] += 1
        return update(*args)
    monkeypatch.setattr(runtime, 'batch_update', counted)
    result = runtime.run(native_dataset, tmp_path/'resume', options, device='cpu', should_stop=lambda: counter['n'] >= stop_after)
    assert result['status'] == 'interrupted'
    latest = tmp_path/'resume/checkpoint_latest.pt'
    runtime.run(native_dataset, tmp_path/'resume', options, device='cpu', resume=latest)
    actual = root.load_tensor_checkpoint(latest, map_location='cpu')
    for key in ('model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'history', 'optimization_coverage',
                'head_updates', 'initial_scales', 'training_statistics'):
        assert root._checkpoint_tree_equal(actual[key], expected[key]), key
    # A per-pulse (CUDA-file) checkpoint carries a different recipe and is refused before responses.
    original_runtime.run(native_dataset, tmp_path/'loop', options, device='cpu', should_stop=lambda: True)
    with pytest.raises(ValueError, match='recipe'):
        runtime.run(native_dataset, tmp_path/'loop', options, device='cpu', resume=tmp_path/'loop/checkpoint_latest.pt')
    # Same optimisation trajectory as the per-pulse runtime, to float64 summation order.
    original_runtime.run(native_dataset, tmp_path/'loop_full', options, device='cpu')
    loop = root.load_tensor_checkpoint(tmp_path/'loop_full/checkpoint_latest.pt', map_location='cpu')
    for key, value in loop['model_state_dict'].items():
        torch.testing.assert_close(expected['model_state_dict'][key], value, rtol=1e-8, atol=1e-12)
    for ea, eb in zip(loop['history'], expected['history']):
        assert eb['validation']['pooled_rel_mse'] == pytest.approx(ea['validation']['pooled_rel_mse'], rel=1e-8)
        assert eb['training_objective'] == pytest.approx(ea['training_objective'], rel=1e-8)


def test_closed_form_endpoint_equals_the_point_kernel_dft(tmp_path, monkeypatch):
    write_shard(tmp_path/'New_Transfer/shards/pass1_hh.npz', nf=32, mutate=stride2_like)
    ds = GOTCHADataset(tmp_path, passes=(1,), region=tiny_region())
    plan = original_runtime.PulsePlan(ds)
    observations = [plan.read(i) for i in range(6)]
    torch.manual_seed(2)
    points, volumes = gauss_legendre_cell_grid(3, nodes_per_cell=1, support_m=ds.region.half_extent_m, dtype=torch.float64)
    field = torch.randn(len(points), dtype=torch.float64)
    closed = BatchedNativeKernel(observations, ds.region, point_tile=8)
    monkeypatch.setattr(spinr_native_batched, 'CLOSED_FORM_ENDPOINT', False)
    dft = BatchedNativeKernel(observations, ds.region, point_tile=8)
    assert closed.closed_form_endpoint and not dft.closed_form_endpoint and torch.equal(closed.bin_ids, dft.bin_ids)
    torch.testing.assert_close(closed.render(points, field, volumes, 1.1), dft.render(points, field, volumes, 1.1), rtol=1e-11, atol=1e-18)
    cot = torch.randn(closed.pulses, closed.width, dtype=torch.complex128) * closed.valid
    torch.testing.assert_close(closed.field_vjp(points, volumes, 1.1, cot), dft.field_vjp(points, volumes, 1.1, cot), rtol=1e-11, atol=1e-18)
    # The per-pulse reference kernel (DFT of the exact point kernel) agrees too.
    single = NativeKernel(observations[2], ds.region, point_tile=8)
    assert not single.affine
    n = int(closed.counts[2])
    torch.testing.assert_close(closed.render(points, field, volumes, 1.1)[2, :n], single.render(points, field, volumes, 1.1), rtol=1e-11, atol=1e-18)


def test_dft_matrix_product_equals_the_fft_path_on_ragged_vectors(tmp_path, monkeypatch):
    write_shard(tmp_path/'New_Transfer/shards/pass1_hh.npz', nf=32)      # genuinely nonuniform native grid
    ds = GOTCHADataset(tmp_path, passes=(1,), region=tiny_region())
    plan = original_runtime.PulsePlan(ds)
    observations = [plan.read(i) for i in range(6)]
    torch.manual_seed(4)
    points, volumes = gauss_legendre_cell_grid(3, nodes_per_cell=1, support_m=ds.region.half_extent_m, dtype=torch.float64)
    field = torch.randn(len(points), dtype=torch.float64)
    bmm = BatchedNativeKernel(observations, ds.region, point_tile=8)
    monkeypatch.setattr(spinr_native_batched, 'DFT_MODE', 'fft')
    fft = BatchedNativeKernel(observations, ds.region, point_tile=8)
    assert bmm.dft is not None and bmm.dft.shape == (6, 32, bmm.width) and fft.dft is None and not bmm.affine
    torch.testing.assert_close(bmm.render(points, field, volumes, 1.1), fft.render(points, field, volumes, 1.1), rtol=1e-11, atol=1e-18)
    cot = torch.randn(bmm.pulses, bmm.width, dtype=torch.complex128) * bmm.valid
    torch.testing.assert_close(bmm.field_vjp(points, volumes, 1.1, cot), fft.field_vjp(points, volumes, 1.1, cot), rtol=1e-11, atol=1e-18)
    single = NativeKernel(observations[3], ds.region, point_tile=8)
    n = int(bmm.counts[3])
    torch.testing.assert_close(bmm.render(points, field, volumes, 1.1)[3, :n], single.render(points, field, volumes, 1.1), rtol=1e-11, atol=1e-18)
    assert not bool(bmm.render(points, field, volumes, 1.1)[3, n:].any())
