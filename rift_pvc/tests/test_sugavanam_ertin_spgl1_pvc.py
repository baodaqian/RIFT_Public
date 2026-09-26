"""PVC SE SPGL1 lane: the dense Eq. 2 operator equals the batched objective; the Eq. 4 budget maps to sigma."""
import numpy as np
import pytest
import torch

from rift.gotcha_dataset import GOTCHADataset
from rift.sugavanam_ertin_acquisition import GOTCHAAcquisition, training_statistics
from rift.sugavanam_ertin_paper_workflow import grid_points, make_recipe, plan
from rift_pvc import sugavanam_ertin_spgl1 as se_spgl1
from rift_pvc.sugavanam_ertin_batched import batched_data_objective
from rift_pvc.tests.test_sugavanam_ertin_batched_pvc import config, three_pulses
from tests.se_dataset_fixtures import write_shard, tiny_region


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv('RIFT_ACCELERATOR', 'cpu')


class Collection:
    """Stub collection acquisition: two Tx, one Rx, shared frequencies, random responses."""
    kind = 'rift_collection'
    kernel_scale = 2.3e-5

    def __init__(self, views=4, seed=0):
        rng = np.random.default_rng(seed)
        self.keys = {'train': list(range(views)), 'validation': []}
        self._obs = []
        for _ in range(views):
            centre = rng.normal(size=3)
            centre *= 10/np.linalg.norm(centre)
            self._obs.append(dict(tx=centre+rng.normal(size=(2, 3))*.05, rx=centre+rng.normal(size=(1, 3))*.05,
                                  freqs=8.5e9+np.arange(6)*6e8,
                                  response=rng.normal(size=(6, 1, 2))+1j*rng.normal(size=(6, 1, 2))))

    def observations(self, key, *, role):
        yield self._obs[key]


def assert_matches(acquisition, points, indices, stats, recipe):
    A, b = se_spgl1.dense_operator(acquisition, points, indices, stats)
    samples = sum(stats['per_view_samples'][i] for i in indices)
    assert A.shape == (samples, len(points))
    torch.manual_seed(len(indices))
    for scale in (1e-3, 1.):
        x = torch.randn(len(points), dtype=torch.complex128)*scale
        reference, reference_grad = batched_data_objective(acquisition, points, x, indices, stats, recipe, gradient=True)
        residual = A @ x-b
        assert .5*float(residual.abs().square().sum())/samples == pytest.approx(reference, rel=1e-12)
        torch.testing.assert_close((A.T @ residual.conj()).conj()/samples, reference_grad, rtol=1e-11, atol=1e-15)
        op = se_spgl1.linear_operator(A)
        np.testing.assert_allclose(op.rmatvec(residual.numpy()), (A.conj().T @ residual).numpy(), rtol=1e-12, atol=1e-14)


def test_dense_operator_equals_the_batched_collection_objective():
    acquisition = Collection()
    points = grid_points(.15, 3, 'cpu')
    assert_matches(acquisition, points, [0, 1, 3], dict(rms=.7, per_view_samples=[12]*4), dict(point_chunk=16, pair_chunk=3))


def test_dense_operator_equals_the_batched_gotcha_objective(tmp_path):
    root = tmp_path/'data'/'New_Transfer'/'shards'
    write_shard(root/'pass1_hh.npz', nf=5, mutate=three_pulses)
    write_shard(root/'pass2_hh.npz', pass_id=2, nf=7, mutate=three_pulses)
    acquisition = GOTCHAAcquisition(GOTCHADataset(tmp_path/'data', passes=(1, 2), region=tiny_region()))
    recipe = make_recipe('synthetic', config(granularity=3))
    partition, _, _ = plan(acquisition, recipe)
    stats = training_statistics(acquisition)
    points = grid_points(acquisition.extent, recipe['granularity'], 'cpu')
    for group in range(len(partition.directions)):
        assert_matches(acquisition, points, np.flatnonzero(partition.assignments == group).tolist(), stats, recipe)


@pytest.fixture
def gotcha(tmp_path):
    root = tmp_path/'data'/'New_Transfer'/'shards'
    write_shard(root/'pass1_hh.npz', nf=5, mutate=three_pulses)
    write_shard(root/'pass2_hh.npz', pass_id=2, nf=7, mutate=three_pulses)
    return GOTCHAAcquisition(GOTCHADataset(tmp_path/'data', passes=(1, 2), region=tiny_region()))


def spgl1_recipe(**extra):
    # With 27 voxels a 0.3 energy budget ends every fixture sub-aperture at an SPGL1
    # root (0.4-0.8 end suboptimal-BP on this fixture); production keeps 0.01.
    return se_spgl1.make_recipe('synthetic', config(granularity=3, residual_relative_energy=.3, **extra))


def test_spgl1_stage1_feeds_the_original_stage2_unchanged(gotcha, tmp_path):
    from rift_pvc import sugavanam_ertin_paper_workflow as workflow_pvc
    from rift.sugavanam_ertin_paper_workflow import make_recipe as paper_recipe
    recipe = spgl1_recipe()
    assert recipe['stage1_iterations'] == 1 and recipe['stage1_solver'] == se_spgl1.SOLVER
    assert recipe['spgl1_commit'] == se_spgl1.SPGL1_COMMIT
    stats = training_statistics(gotcha)
    partition, _, _ = plan(gotcha, recipe)
    groups = list(range(len(partition.directions)))
    out = tmp_path/'se'
    se_spgl1.solve_groups(gotcha, recipe, stats, out, groups[:1], log=lambda r: None)
    with pytest.raises(FileNotFoundError):
        se_spgl1.assemble(gotcha, recipe, stats, out)
    events = []
    se_spgl1.solve_groups(gotcha, recipe, stats, out, groups, log=events.append)
    assert [e['event'] for e in events] == ['group_kept']+['group_done']*(len(groups)-1)
    assert all(e['operator_check']['gradient_relative_difference'] < 1e-11 for e in events[1:])
    report = se_spgl1.assemble(gotcha, recipe, stats, out)
    assert report['converged'] == len(groups) and report['exits']['root_found'] == len(groups)
    saved = torch.load(out/'checkpoint_latest.pt', map_location='cpu', weights_only=False)
    assert saved['phase'] == 'stage1' and saved['iteration'] == 1 and len(saved['stage1_history']) == len(groups)
    for group, row in enumerate(saved['stage1_history']):
        f, _ = batched_data_objective(gotcha, grid_points(gotcha.extent, 3, 'cpu'), saved['fields'][group],
            np.flatnonzero(partition.assignments == group).tolist(), stats, recipe, gradient=True)
        assert f <= row['target_data_loss']*(1+1e-3)  # on the Eq. 4 budget, original operator
    # The original paper-v1 recipe cannot resume this identity.
    with pytest.raises(ValueError, match='recipe changed'):
        workflow_pvc.run(gotcha, paper_recipe('synthetic', config(granularity=3, residual_relative_energy=.3)),
                         out, device='cpu', resume=out/'checkpoint_latest.pt')
    # Stage-1-only boundary, then the unchanged Stage 2.
    result = workflow_pvc.run(gotcha, recipe, out, device='cpu', resume=out/'checkpoint_latest.pt', stage1_only=True)
    assert result['status'] == 'stage1_complete'
    result = workflow_pvc.run(gotcha, recipe, out, device='cpu', resume=out/'checkpoint_latest.pt')
    assert result['status'] in ('complete', 'surface_unavailable', 'incomplete_iso_supervision')
    final = torch.load(out/'checkpoint_final.pt', map_location='cpu', weights_only=False)
    assert final['sdf_step'] == recipe['stage2_steps'] and final['stage1_source']['iteration'] == 1


def test_an_unconverged_subaperture_keeps_the_original_gate_closed(gotcha, tmp_path, monkeypatch):
    from rift_pvc import sugavanam_ertin_paper_workflow as workflow_pvc
    recipe = spgl1_recipe()
    stats = training_statistics(gotcha)
    partition, _, _ = plan(gotcha, recipe)
    real = se_spgl1.solve
    calls = []
    def one_iteration_limit(A, b, sigma, **log):
        x, info = real(A, b, sigma, **log)
        if not calls:
            info = dict(info, stat=5)
        calls.append(1)
        return x, info
    monkeypatch.setattr(se_spgl1, 'solve', one_iteration_limit)
    out = tmp_path/'se'
    se_spgl1.solve_groups(gotcha, recipe, stats, out, range(len(partition.directions)), log=lambda r: None)
    report = se_spgl1.assemble(gotcha, recipe, stats, out)
    assert report['converged'] == len(partition.directions)-1 and report['exits']['iteration_limit'] == 1
    result = workflow_pvc.run(gotcha, recipe, out, device='cpu', resume=out/'checkpoint_latest.pt')
    assert result['status'] == 'stage1_unconverged'


def test_sigma_is_the_recipe_residual_energy_budget():
    # 0.5 ||r||^2 / N <= target  <=>  ||r|| <= sqrt(2 N target)
    assert se_spgl1.sigma_from_target(.25, 8) == pytest.approx(2.)


def load_driver():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2]/'scripts_pvc'/'se_spgl1_stage1.py'
    spec = importlib.util.spec_from_file_location('se_spgl1_stage1_driver', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_driver_gotcha_source_is_the_frontend_plan(tmp_path, capsys):
    """--gotcha builds acquisition and SE config through train_gotcha_dataset_pvc itself."""
    import json
    import train_gotcha_dataset_pvc as frontend
    root = tmp_path/'data'/'New_Transfer'/'shards'
    write_shard(root/'pass1_hh.npz', nf=5, mutate=three_pulses)
    write_shard(root/'pass2_hh.npz', pass_id=2, nf=7, mutate=three_pulses)
    regions = tmp_path/'regions.json'
    regions.write_text(json.dumps(dict(schema='rift_gotcha_regions_v1', regions=dict(se_fixture=dict(
        target_id='synthetic', translation_m=[0, 0, 0], rotation_local_to_native=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        half_extent_m=.03, placement_provenance='unit test')))))
    config = tmp_path/'se.json'
    config.write_text(json.dumps(dict(granularity=3, residual_relative_energy=.3, initialization_std=.05)))
    gotcha = ['--dataset-root', str(tmp_path/'data'), '--region', 'se_fixture', '--region-config', str(regions),
              '--passes', '1', '2', '--num-train', '4', '--pulses-per-sector', '2', '--method', 'sugavanam_ertin',
              '--config', str(config)]
    dataset, plan_ = frontend.make_plan(frontend.parse_args(gotcha))
    acquisition = GOTCHAAcquisition(dataset)
    recipe = se_spgl1.make_recipe(acquisition.kind, plan_['plans'][0]['config'])
    partition, _, _ = plan(acquisition, recipe)
    groups = len(partition.directions)
    driver = load_driver()
    out = tmp_path/'se'
    assert driver.main(['solve', '--output', str(out), '--groups', f'0-{groups-1}', '--verbosity', '0', '--gotcha', *gotcha]) == 0
    code = driver.main(['assemble', '--output', str(out), '--gotcha', *gotcha])
    assembled = [json.loads(line) for line in capsys.readouterr().out.splitlines() if '"assembled"' in line][-1]
    assert code == (0 if assembled['converged'] == groups else 2) and assembled['groups'] == groups
    saved = torch.load(out/'checkpoint_latest.pt', map_location='cpu', weights_only=False)
    assert saved['acquisition'] == acquisition.identity and saved['recipe']['stage1_solver'] == se_spgl1.SOLVER
    assert saved['recipe']['initialization_std'] == .05 and saved['recipe']['granularity'] == 3
    with pytest.raises(SystemExit):
        driver.main(['solve', '--output', str(out), '--groups', '0', '--npz-path', 'x.npz', '--gotcha', *gotcha])


def test_a_wrong_operator_stops_the_solve(gotcha, tmp_path, monkeypatch):
    recipe = spgl1_recipe()
    stats = training_statistics(gotcha)
    real = se_spgl1.dense_operator
    def swapped_rows(*args):
        A, b = real(*args)
        return A.flip(0), b
    monkeypatch.setattr(se_spgl1, 'dense_operator', swapped_rows)
    with pytest.raises(ValueError, match='differs from the batched objective'):
        se_spgl1.solve_groups(gotcha, recipe, stats, tmp_path/'se', [0], log=lambda r: None)
    assert not (tmp_path/'se'/se_spgl1.GROUPS_DIR/'group_00.pt').exists()


def test_partition_is_pinned_across_nodes_with_last_bit_differences(gotcha, tmp_path, monkeypatch):
    """A node whose numpy/libm changes the mean directions' last bits still binds the pinned partition."""
    recipe = spgl1_recipe()
    stats = training_statistics(gotcha)
    partition, _, _ = plan(gotcha, recipe)
    groups = list(range(len(partition.directions)))
    out = tmp_path/'se'
    se_spgl1.solve_groups(gotcha, recipe, stats, out, groups[:1], log=lambda r: None)
    pinned = torch.load(out/se_spgl1.GROUPS_DIR/'partition.pt', weights_only=False)['record']
    real_plan = se_spgl1.plan
    def other_node(acquisition, recipe_):
        p, v, planning = real_plan(acquisition, recipe_)
        p.directions = p.directions*(1+4e-16)  # last-bit drift, same bins and assignments
        return p, v, planning
    monkeypatch.setattr(se_spgl1, 'plan', other_node)
    assert other_node(gotcha, recipe)[0].record() != pinned
    se_spgl1.solve_groups(gotcha, recipe, stats, out, groups[1:], log=lambda r: None)
    report = se_spgl1.assemble(gotcha, recipe, stats, out)
    assert report['groups'] == len(groups)
    saved = torch.load(out/'checkpoint_latest.pt', map_location='cpu', weights_only=False)
    assert saved['partition'] == pinned
    # A different assignment is not drift: refuse it.
    def other_assignment(acquisition, recipe_):
        p, v, planning = real_plan(acquisition, recipe_)
        p.assignments = p.assignments[::-1].copy()
        return p, v, planning
    monkeypatch.setattr(se_spgl1, 'plan', other_assignment)
    with pytest.raises(ValueError, match='differs from the pinned one'):
        se_spgl1.solve_groups(gotcha, recipe, stats, out, groups[:1], log=lambda r: None)


def test_exclusive_save_keeps_the_first_writer(tmp_path):
    path = tmp_path/'x'/'pinned.pt'
    se_spgl1.exclusive_save(dict(value=1), path)
    se_spgl1.exclusive_save(dict(value=2), path)
    assert torch.load(path, weights_only=False)['value'] == 1
    assert sorted(p.name for p in path.parent.iterdir()) == ['pinned.pt']
