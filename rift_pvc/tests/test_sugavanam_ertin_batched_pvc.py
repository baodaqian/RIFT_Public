"""PVC Sugavanam--Ertin batched Stage 1: equivalence with the original objective and workflow."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rift import sugavanam_ertin_paper_workflow as workflow_cuda
from rift.gotcha_dataset import GOTCHADataset
from rift.sugavanam_ertin_acquisition import (CollectionAcquisition, GOTCHAAcquisition, data_objective,
                                              fourier_forward, training_statistics, validation_readout)
from rift.sugavanam_ertin_paper_workflow import grid_points, make_recipe, plan
from rift_pvc import sugavanam_ertin_paper_workflow as workflow_pvc
from rift_pvc.sugavanam_ertin_batched import (EXECUTION, batched_data_objective, batched_validation_readout,
                                              fourier_forward_budget, observation_pairs)
from tests.se_dataset_fixtures import write_shard, tiny_region


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv('RIFT_ACCELERATOR', 'cpu')


def three_pulses(arrays, _metadata):
    for key, value in list(arrays.items()):
        if key != 'frequencies_hz' and len(value) == 360:
            arrays[key] = np.repeat(value, 3, axis=0)
    ids = np.tile(np.arange(3, dtype=np.int32), 360)
    arrays['pulse_index'] = ids
    arrays['z'] = arrays['z'] + ids * .002
    arrays['r0'] = np.sqrt(arrays['x']**2 + arrays['y']**2 + arrays['z']**2)
    arrays['response'] = (arrays['response'] * (1 + .3 * ids)[:, None] * np.exp(.1j * np.arange(arrays['response'].shape[1]))[None]).astype(np.complex64)


@pytest.fixture
def acquisition(tmp_path):
    root = tmp_path/'data'/'New_Transfer'/'shards'
    write_shard(root/'pass1_hh.npz', nf=5, mutate=three_pulses)
    write_shard(root/'pass2_hh.npz', pass_id=2, nf=7, mutate=three_pulses)
    return GOTCHAAcquisition(GOTCHADataset(tmp_path/'data', passes=(1, 2), region=tiny_region()))


def config(**extra):
    base = dict(granularity=2, azimuth_bins=2, elevation_bins=1, stage1_iterations=1, stage2_steps=1,
                hidden_dim=8, n_layers=4, n_fourier=2, batch_on=8, batch_off=8, batch_iso=8, iso_count=8,
                iso_start=1, projection_iterations=4, export_grid=8, checkpoint_every=1, validation_every=1,
                pair_chunk=5)
    base.update(extra)
    return base


def test_batched_objective_and_readout_match_the_originals(acquisition):
    recipe = make_recipe('synthetic', config())
    partition, assignments, planning = plan(acquisition, recipe)
    stats = training_statistics(acquisition)
    points = grid_points(acquisition.extent, recipe['granularity'], 'cpu')
    torch.manual_seed(0)
    x = torch.randn(len(points), dtype=torch.complex128) * 1e-3
    for group in range(len(partition.directions)):
        indices = np.flatnonzero(partition.assignments == group).tolist()
        assert len({o.frequencies_hz.shape for i in indices for o in acquisition.observations(acquisition.keys['train'][i], role='train')}) == 2
        loss_a, grad_a = data_objective(acquisition, points, x, indices, stats, recipe, gradient=True)
        loss_b, grad_b = batched_data_objective(acquisition, points, x, indices, stats, recipe, gradient=True, element_budget=64)
        assert loss_b == pytest.approx(loss_a, rel=1e-11)
        torch.testing.assert_close(grad_b, grad_a, rtol=1e-10, atol=1e-16)
        assert batched_data_objective(acquisition, points, x, indices, stats, recipe) == pytest.approx(loss_a, rel=1e-11)
    fields = torch.randn(len(partition.directions), len(points), dtype=torch.complex128) * 1e-3
    a = validation_readout(acquisition, points, fields, assignments, stats, recipe)
    b = batched_validation_readout(acquisition, points, fields, assignments, stats, recipe, element_budget=64)
    assert a.keys() == b.keys() and b['samples'] == a['samples'] and b['views'] == a['views']
    for key in ('global_complex_rel_mse', 'squared_error', 'target_energy'):
        assert b[key] == pytest.approx(a[key], rel=1e-11)


def test_collection_pairs_reproduce_the_collection_render_and_other_kinds_fall_back():
    torch.manual_seed(1)
    stub = SimpleNamespace(kind='rift_collection', kernel_scale=.37)
    observation = dict(response=torch.randn(6, 2, 3, dtype=torch.complex128).numpy(),
                       tx=np.random.default_rng(2).normal(size=(3, 3)) * 10, rx=np.random.default_rng(3).normal(size=(2, 3)) * 10,
                       freqs=9e9 + np.arange(6) * 5e7)
    points = torch.randn(11, 3, dtype=torch.float64) * .05
    weights = torch.randn(11, dtype=torch.complex128)
    cc, f, d, r, a, t = observation_pairs(stub, observation, 'cpu')
    expected = CollectionAcquisition.render(stub, points, weights, observation, point_chunk=4, pair_chunk=2)
    actual = fourier_forward_budget(points, weights, f, d, r, a, cc=cc, point_chunk=4, pair_chunk=2, element_budget=50)
    torch.testing.assert_close(actual.reshape(expected.shape), expected, rtol=1e-12, atol=1e-16)
    assert t.shape == (6, 6) and torch.equal(t.reshape(6, 2, 3), torch.as_tensor(observation['response']))
    # A synthetic acquisition (no native pairs) is served by the original objective unchanged.
    class Tiny:
        kind = 'synthetic'
        keys = {'train': [0], 'validation': [0]}
        def observations(self, key, *, role): yield dict(response=np.ones(3, dtype=np.complex128))
        def render(self, points, weights, observation, **kw): return weights
    stats = dict(rms=1., per_view_samples=[3])
    recipe = dict(point_chunk=4, pair_chunk=2)
    w = torch.tensor([.5 + .1j, -.2j, .3], dtype=torch.complex128)
    pts = torch.zeros(3, 3, dtype=torch.float64)
    assert batched_data_objective(Tiny(), pts, w, [0], stats, recipe) == data_objective(Tiny(), pts, w, [0], stats, recipe)


def test_pvc_run_matches_the_original_workflow_and_discloses_execution(acquisition, tmp_path):
    recipe = make_recipe('synthetic', config())
    a = workflow_cuda.run(acquisition, recipe, tmp_path/'cuda_file', device='cpu')
    b = workflow_pvc.run(acquisition, recipe, tmp_path/'pvc', device='cpu')
    assert a['status'] == b['status'] == 'stage1_unconverged'
    ca = torch.load(tmp_path/'cuda_file/checkpoint_final.pt', map_location='cpu', weights_only=False)
    cb = torch.load(tmp_path/'pvc/checkpoint_final.pt', map_location='cpu', weights_only=False)
    assert ca['recipe'] == cb['recipe'] and ca['acquisition'] == cb['acquisition'] and ca['partition'] == cb['partition']
    assert cb['view_exposures'] == ca['view_exposures'] and cb['statistics'] == ca['statistics']
    torch.testing.assert_close(cb['fields'], ca['fields'], rtol=1e-9, atol=1e-14)
    assert len(cb['stage1_history']) == len(ca['stage1_history']) == 2
    for ra, rb in zip(ca['stage1_history'], cb['stage1_history']):
        assert rb['execution'] == EXECUTION and 'execution' not in ra
        for key in ('data_loss', 'l1_norm', 'subproblem_duality_gap'):
            assert rb[key] == pytest.approx(ra[key], rel=1e-9, abs=1e-14)
        assert rb['sparse_solver']['converged'] == ra['sparse_solver']['converged']
    assert cb['best']['validation']['global_complex_rel_mse'] == pytest.approx(ca['best']['validation']['global_complex_rel_mse'], rel=1e-9)
    execution = json.loads((tmp_path/'pvc/execution.json').read_text())
    assert execution['execution'] == EXECUTION
    assert not (tmp_path/'cuda_file/execution.json').exists()


def test_pvc_run_interrupts_resumes_exactly_and_resumes_original_checkpoints(acquisition, tmp_path):
    recipe = make_recipe('synthetic', config(stage1_iterations=2))
    workflow_pvc.run(acquisition, recipe, tmp_path/'full', device='cpu')
    full = torch.load(tmp_path/'full/checkpoint_final.pt', map_location='cpu', weights_only=False)
    calls = {'n': 0}
    def stop_after_first():
        calls['n'] += 1
        return calls['n'] >= 1
    result = workflow_pvc.run(acquisition, recipe, tmp_path/'resumed', device='cpu', should_stop=stop_after_first)
    assert result['status'] == 'interrupted'
    partial = torch.load(tmp_path/'resumed/checkpoint_latest.pt', map_location='cpu', weights_only=False)
    assert partial['iteration'] == 0 and partial['group_cursor'] == 1
    workflow_pvc.run(acquisition, recipe, tmp_path/'resumed', device='cpu', resume=tmp_path/'resumed/checkpoint_latest.pt')
    resumed = torch.load(tmp_path/'resumed/checkpoint_final.pt', map_location='cpu', weights_only=False)
    torch.testing.assert_close(resumed['fields'], full['fields'], rtol=0, atol=0)
    assert resumed['stage1_history'] == full['stage1_history'] and resumed['view_exposures'] == full['view_exposures']
    # A checkpoint written by the original (per-observation) workflow resumes into the batched lane.
    calls['n'] = 0
    assert workflow_cuda.run(acquisition, recipe, tmp_path/'cross', device='cpu', should_stop=stop_after_first)['status'] == 'interrupted'
    workflow_pvc.run(acquisition, recipe, tmp_path/'cross', device='cpu', resume=tmp_path/'cross/checkpoint_latest.pt')
    cross = torch.load(tmp_path/'cross/checkpoint_final.pt', map_location='cpu', weights_only=False)
    torch.testing.assert_close(cross['fields'], full['fields'], rtol=1e-9, atol=1e-14)
    assert 'execution' not in cross['stage1_history'][0] and cross['stage1_history'][1]['execution'] == EXECUTION
    with pytest.raises(ValueError, match='original output directory'):
        workflow_pvc.run(acquisition, recipe, tmp_path/'elsewhere', device='cpu', resume=tmp_path/'cross/checkpoint_latest.pt')
