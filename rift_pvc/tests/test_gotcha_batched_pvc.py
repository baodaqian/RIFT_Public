"""PVC GOTCHA batched sector update: equivalence with the CUDA-file per-pulse loop, resume, disclosure."""
import signal

import pytest
import torch

import train_gotcha_dataset as cuda_cli
import train_gotcha_dataset_pvc as cli
from rift import gotcha_training as original
from rift.gotcha_dataset import GOTCHADataset
from rift_pvc import gotcha_training as pvc
from rift_pvc.gotcha_batched import BATCHED, LOOP, sector_execution_value
from tests.test_gotcha_dataset import write_shard, tiny_region
from tests.test_gotcha_pulse_sampling import multiple_pulses

ARGV = ['--epochs', '2', '--granularity', '2', '--max-points', '15', '--sh-degree', '1', '--probe-every', '1',
        '--pulses-per-sector', '3', '--point-chunk', '3', '--checkpoint-every', '1']


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv('RIFT_ACCELERATOR', 'cpu')
    write_shard(tmp_path/'New_Transfer/shards/pass1_hh.npz', mutate=multiple_pulses)
    return tmp_path


def dataset(root, monkeypatch, validation=2):
    ds = GOTCHADataset(root, passes=(1,), region=tiny_region(), pulses_per_sector=3, num_train=2)
    views = ds.viewpoints
    monkeypatch.setattr(ds, 'viewpoints', lambda role: views(role)[:validation] if role == 'validation' else views(role))
    return ds


def recipes(extra=()):
    loop = original.recipe_from_args(cuda_cli.parse_args(ARGV + list(extra)), 'rift')
    batched = pvc.recipe_from_args(cli.parse_args(ARGV + list(extra)), 'rift')
    assert batched['sector_execution'] == BATCHED
    assert {k: v for k, v in batched.items() if k != 'sector_execution'} == loop
    return loop, batched


def assert_states_close(a, b, rtol=1e-4, atol=1e-6):
    assert a.keys() == b.keys()
    for key, value in a.items():
        other = b[key]
        if value.is_floating_point() or value.is_complex():
            torch.testing.assert_close(other, value, rtol=rtol, atol=atol, msg=key)
        else:
            assert torch.equal(other, value), key


def test_batched_trainer_matches_cuda_file_loop_without_refinement(root, monkeypatch):
    ds = dataset(root, monkeypatch)
    loop, batched = recipes(['--refine-every', '100', '--optimizer', 'legacy'])
    original.train(ds, 'rift', loop, root/'loop', device='cpu')
    pvc.train(ds, 'rift', batched, root/'batched', device='cpu')
    a = torch.load(root/'loop/checkpoint_final.pt', weights_only=False)
    b = torch.load(root/'batched/checkpoint_final.pt', weights_only=False)
    assert a['updates'] == b['updates'] == 4 and b['recipe']['sector_execution'] == BATCHED
    # Parameters, gain and the refinement accumulators (probe every update) agree to float32 order.
    assert_states_close(a['model_state_dict'], b['model_state_dict'])
    for ea, eb in zip(a['history'], b['history']):
        va, vb = ea['validation'], eb['validation']
        # float32 parameters and a 1e3 warm-started gain amplify summation-order noise into the metric.
        assert vb['pooled_rel_mse'] == pytest.approx(va['pooled_rel_mse'], rel=1e-4)
        assert vb['full_native_complex_rel_mse'] == pytest.approx(va['full_native_complex_rel_mse'], rel=1e-4)
        assert vb['by_polarization']['hh']['pulses'] == va['by_polarization']['hh']['pulses'] == 6
        assert vb['viewpoints'] == va['viewpoints'] == 2


def test_batched_trainer_takes_the_same_refinement_decisions(root, monkeypatch):
    ds = dataset(root, monkeypatch)
    loop, batched = recipes(['--refine-every', '2', '--optimizer', 'legacy'])
    original.train(ds, 'rift', loop, root/'loop', device='cpu')
    pvc.train(ds, 'rift', batched, root/'batched', device='cpu')
    a = torch.load(root/'loop/checkpoint_final.pt', weights_only=False)['model_state_dict']
    b = torch.load(root/'batched/checkpoint_final.pt', weights_only=False)['model_state_dict']
    for key in ('hh.field.active_mask', 'hh.field.order', 'hh.field.level'):
        assert torch.equal(a[key], b[key]), key
    assert int(a['hh.field.active_mask'].sum()) > 8   # a refinement event happened
    assert_states_close(a, b, rtol=1e-3, atol=1e-5)


def test_batched_trainer_matches_cuda_file_loop_under_b787_schedule(root, monkeypatch):
    ds = dataset(root, monkeypatch)
    loop, batched = recipes(['--refine-every', '1'])
    # Tiny-fixture thresholds so the epoch-end event acts; the production values cannot be met by 8 points.
    for recipe in (loop, batched):
        recipe['optimizer_schedule'] = dict(recipe['optimizer_schedule'], min_spatial_exposure=1, min_angular_exposure=1,
                                            spatial_fraction=.5, angular_fraction=.5)
        recipe['max_level'] = 2
    original.train(ds, 'rift', loop, root/'loop', device='cpu')
    pvc.train(ds, 'rift', batched, root/'batched', device='cpu')
    a = torch.load(root/'loop/checkpoint_final.pt', weights_only=False)
    b = torch.load(root/'batched/checkpoint_final.pt', weights_only=False)
    for key in ('hh.field.active_mask', 'hh.field.order', 'hh.field.level'):
        assert torch.equal(a['model_state_dict'][key], b['model_state_dict'][key]), key
    assert int(a['model_state_dict']['hh.field.active_mask'].sum()) > 8   # an epoch-end event happened
    # eps 1e-8 passes float32 summation-order noise of the tiny gradients into the steps.
    assert_states_close(a['model_state_dict'], b['model_state_dict'], rtol=1e-3, atol=1e-5)
    assert [e['optimizer'] for e in a['history']] == [e['optimizer'] for e in b['history']]
    assert a['scheduler_state_dict'] == b['scheduler_state_dict']


def test_batched_isotropic_and_grid_methods_match_loop(root, monkeypatch):
    ds = dataset(root, monkeypatch, validation=1)
    for method in ('isotropic', 'rift_grid'):
        loop, batched = recipes(['--refine-every', '100', '--epochs', '1'])
        original.train(ds, method, loop, root/f'loop_{method}', device='cpu')
        pvc.train(ds, method, batched, root/f'batched_{method}', device='cpu')
        a = torch.load(root/f'loop_{method}/checkpoint_final.pt', weights_only=False)
        b = torch.load(root/f'batched_{method}/checkpoint_final.pt', weights_only=False)
        assert_states_close(a['model_state_dict'], b['model_state_dict'])
        assert b['history'][-1]['validation']['pooled_rel_mse'] == pytest.approx(a['history'][-1]['validation']['pooled_rel_mse'], rel=1e-4)


def test_batched_interrupt_resume_is_exact_and_loop_checkpoints_are_refused(root, monkeypatch):
    ds = dataset(root, monkeypatch, validation=1)
    loop, batched = recipes(['--refine-every', '2'])
    pvc.train(ds, 'rift', batched, root/'full', device='cpu')
    original_step = torch.optim.AdamW.step
    def interrupt(opt, *a, **kw):
        result = original_step(opt, *a, **kw)
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        return result
    with monkeypatch.context() as m:
        m.setattr(torch.optim.AdamW, 'step', interrupt)
        assert pvc.train(ds, 'rift', batched, root/'resumed', device='cpu')['status'] == 'interrupted'
    latest = root/'resumed/checkpoint_latest.pt'
    assert torch.load(latest, weights_only=False)['updates'] == 1
    with pytest.raises(ValueError, match='recipe'):
        pvc.train(ds, 'rift', dict(batched, sector_execution=LOOP), root/'resumed', device='cpu', resume=latest)
    pvc.train(ds, 'rift', batched, root/'resumed', device='cpu', resume=latest)
    full = torch.load(root/'full/checkpoint_final.pt', weights_only=False)
    resumed = torch.load(root/'resumed/checkpoint_final.pt', weights_only=False)
    assert full['history'] == resumed['history']
    for key, value in full['model_state_dict'].items():
        torch.testing.assert_close(value, resumed['model_state_dict'][key], rtol=0, atol=0)


def test_frontend_flag_and_recipe_disclosure(monkeypatch):
    monkeypatch.setenv('RIFT_ACCELERATOR', 'cpu')
    assert pvc.recipe_from_args(cli.parse_args(ARGV), 'rift')['sector_execution'] == BATCHED
    assert pvc.recipe_from_args(cli.parse_args(ARGV + ['--sector-execution', 'loop']), 'rift')['sector_execution'] == LOOP
    assert sector_execution_value('batched') == BATCHED and sector_execution_value(LOOP) == LOOP
    with pytest.raises(ValueError):
        sector_execution_value('fast')
    with pytest.raises(SystemExit):
        cli.parse_args(ARGV + ['--sector-execution', 'fast'])
    assert 'sector_execution' not in original.recipe_from_args(cuda_cli.parse_args(ARGV), 'rift')
