"""Shared fixed acquisition: synthetic multi-pulse sectors, no production fits."""
import copy
import hashlib
import json
from pathlib import Path
import signal

import numpy as np
import pytest
import torch

import train_gotcha_dataset as cli
from rift.gotcha_dataset import GOTCHADataset, NativeShardReader, validate_checkpoint
from rift.gotcha_pulse_sampling import select_rows, validate_pulse_limit, pulse_limit_from_contract
from tests.test_gotcha_dataset import write_shard, tiny_region


METHODS = ['rift', 'spinr', 'radar_fields', 'geraf', 'radarsplat', 'sugavanam_ertin']


def multiple_pulses(arrays, _metadata):
    # Ragged sectors, distinct native pulse IDs, geometry and signal amplitudes.
    counts = 3 + np.arange(360) % 4
    for key, value in list(arrays.items()):
        if key != 'frequencies_hz' and len(value) == 360:
            arrays[key] = np.repeat(value, counts, axis=0)
    ids = np.concatenate([np.arange(n, dtype=np.int32) for n in counts])
    arrays['pulse_index'] = ids
    arrays['z'] += ids * .001
    arrays['r0'] = np.sqrt(sum(arrays[k]**2 for k in ('x', 'y', 'z')))
    arrays['response'] *= (ids + 1)[:, None]


@pytest.fixture
def root(tmp_path):
    write_shard(tmp_path/'New_Transfer/shards/pass1_hh.npz', mutate=multiple_pulses)
    return tmp_path


def dataset(root, cap=2, **kwargs):
    return GOTCHADataset(root, passes=(1,), region=tiny_region(), pulses_per_sector=cap, **kwargs)


@pytest.mark.parametrize('bad', [-1, 1.5, True, None, '2'])
def test_invalid_caps(bad):
    with pytest.raises(ValueError, match='nonnegative integer'):
        validate_pulse_limit(bad)


def test_selection_is_nested_native_order_and_independent_of_global_rng():
    rows = np.arange(10, 30)
    ids = np.arange(40) * 3
    before = np.random.get_state()
    two = select_rows(rows, ids, 2, 3, 157)
    five = select_rows(rows, ids, 5, 3, 157)
    after = np.random.get_state()
    assert set(two) <= set(five) <= set(rows)
    assert np.all(np.diff(five) > 0) and not five.flags.writeable
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    np.testing.assert_array_equal(two, select_rows(rows, ids, 2, 3, 157))
    assert select_rows(rows, ids, 0, 3, 157) is rows
    assert select_rows(rows, ids, 99, 3, 157) is rows


def test_metadata_selection_keeps_source_and_samples_each_role_and_seals_excluded(root, monkeypatch):
    path = root/'New_Transfer/shards/pass1_hh.npz'
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    full = dataset(root, 0, num_train=4)
    with monkeypatch.context() as m:
        m.setattr(np, 'memmap', lambda *a, **kw: pytest.fail('Selection mapped responses'))
        selected = dataset(root, num_train=4)
    shard, source = selected.shards[1, 'hh'], full.shards[1, 'hh']
    assert shard.shape == source.shape and shard.identity == source.identity
    for key in source.arrays:
        np.testing.assert_array_equal(shard.arrays[key], source.arrays[key])
    assert selected.summary()['pulses_by_polarization']['hh']['train'] == 8
    for role in ('train', 'validation', 'test'):
        assert np.count_nonzero(shard.row_roles == role) == 2*len(selected.splits_by_pass[1][role])
        assert set(np.flatnonzero(shard.row_roles == role)) < set(np.flatnonzero(source.row_roles == role))
        for sector in selected.splits_by_pass[1][role]:
            expected = select_rows(source.sector_rows[sector], source.arrays['pulse_index'], 2, 1, sector)
            np.testing.assert_array_equal(shard.sector_rows[sector], expected)
        inventory = selected.training_pulse_selection['inventory']['pass1_hh'][role]
        assert inventory['selected_pulses'] == int(np.count_nonzero(shard.row_roles == role))
        assert inventory['source_pulses'] == int(np.count_nonzero(source.row_roles == role))
    p, sector = selected.viewpoints('train')[0]
    selected_rows = shard.sector_rows[sector]
    excluded = next(row for row in source.sector_rows[sector] if row not in selected_rows)
    test_row = np.flatnonzero(shard.row_roles == 'test')[0]
    with monkeypatch.context() as m:
        m.setattr(np, 'memmap', lambda *a, **kw: pytest.fail('Denied row mapped responses'))
        val_sector = selected.viewpoints('validation')[0][1]
        excluded_val = next(row for row in source.sector_rows[val_sector] if row not in shard.sector_rows[val_sector])
        for row in (excluded, excluded_val, test_row):
            with pytest.raises(PermissionError):
                shard.read(int(row))
    for row, obs in zip(selected_rows, selected.observations(p, sector, 'hh')):
        native = source.read(int(row))
        assert obs.pulse_index == native.pulse_index
        assert obs.reference_range_m == native.reference_range_m
        np.testing.assert_array_equal(obs.position_m, native.position_m)
        np.testing.assert_array_equal(obs.response, native.response)
        np.testing.assert_array_equal(obs.frequencies_hz, native.frequencies_hz)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_hash


def test_polarizations_share_mask_and_caps_bind_identity(root):
    write_shard(root/'New_Transfer/shards/pass1_vv.npz', pol='vv', nf=35, mutate=multiple_pulses)
    ds = dataset(root, polarizations=('hh', 'vv'))
    for role in ('train', 'validation', 'test'):
        for sector in ds.splits_by_pass[1][role]:
            np.testing.assert_array_equal(ds.shards[1, 'hh'].sector_rows[sector],
                                          ds.shards[1, 'vv'].sector_rows[sector])
    full = dataset(root, 0)
    omitted = GOTCHADataset(root, passes=(1,), region=tiny_region())
    assert full.contract == omitted.contract and full.identity == omitted.identity
    assert 'training_pulse_selection' not in full.contract
    assert len({full.identity, dataset(root, 2).identity, dataset(root, 3).identity}) == 3
    assert dataset(root, 2).contract == dataset(root, 2).contract
    assert pulse_limit_from_contract(full.contract) == 0
    assert pulse_limit_from_contract(ds.contract) == 2
    broken = copy.deepcopy(ds.contract)
    broken['training_pulse_selection']['seed'] += 1
    with pytest.raises(ValueError):
        pulse_limit_from_contract(broken)
    broken = copy.deepcopy(ds.contract)
    broken['training_pulse_selection']['schema'] = 'gotcha_fixed_training_pulses_v1'
    with pytest.raises(ValueError, match='matching train/validation/test'):
        pulse_limit_from_contract(broken)


def test_all_six_cli_plans_share_inventory_namespace_and_have_no_response_reads(root, monkeypatch):
    monkeypatch.setattr(cli, 'load_region', lambda *_: tiny_region())
    monkeypatch.setattr(NativeShardReader, 'read', lambda *a: pytest.fail('Plan read response'))
    ds, report = cli.make_plan(cli.parse_args(['--dataset-root', str(root), '--passes', '1',
        '--method', *METHODS, '--pulses-per-sector', '2', '--output-root', str(root/'runs')]))
    assert [p['method'] for p in report['plans']] == METHODS
    assert not ds.summary()['response_payload_read']
    assert len({str(Path(p['output_dir']).parent) for p in report['plans']}) == 1
    for plan in report['plans']:
        assert plan['training_pulse_selection'] == ds.contract['training_pulse_selection']
        assert Path(plan['output_dir']).parent.name == 'pulse_subset2_all_roles_v2'
    for recipe in (report['plans'][0]['config'], report['plans'][1]['native_plan']['recipe']):
        assert recipe['pulse_policy'] != 'all_native'
        assert 'validation' in recipe['pulse_policy']
        assert 'test' in recipe['pulse_policy']
        assert recipe['training_pulse_selection'] == ds.training_pulse_selection
    assert not (root/'runs').exists()


def test_each_baseline_consumes_same_selected_pulses_in_training_and_validation(root, monkeypatch):
    from rift.spinr_gotcha_training import PulsePlan, training_statistics as spinr_stats
    from rift.geraf_source_data import GOTCHASourceData
    from rift import radar_fields_gotcha as rf, radarsplat_gotcha as rs
    from rift.sugavanam_ertin_acquisition import GOTCHAAcquisition
    ds = dataset(root, num_train=10)
    shard = ds.shards[1, 'hh']
    selected_rows = set(np.flatnonzero(shard.row_roles == 'train'))
    plan = PulsePlan(ds)
    assert set(plan.records[:, 1]) == selected_rows and len(plan.records) == 20
    stats = spinr_stats(plan)
    assert stats['hh']['pulses'] == 20 and stats['hh']['samples'] == 20*32
    expected_power = np.mean((shard.arrays['pulse_index'][sorted(selected_rows)] + 1)**2)
    assert stats['hh']['mean_power'] == pytest.approx(expected_power)
    source = GOTCHASourceData(ds)
    rs_cache = rs.GOTCHAPowerCache(ds, 'hh', root/'targets')
    se = GOTCHAAcquisition(ds)
    assert 'train' in rs_cache.recipe['response_selection'].lower() and 'validation' in rs_cache.recipe['response_selection']
    assert rs_cache.recipe['sealed_protocol_identity']['dataset_contract']['training_pulse_selection'] == ds.training_pulse_selection
    assert se.identity['viewpoint'] == 'pass_sector_same_fixed_cap_train_validation_test'
    assert sum(se.train_sample_counts) == 20*32
    for role in ('train', 'validation'):
        view = ds.viewpoints(role)[0]
        rows = shard.sector_rows[view[1]]
        assert len(rows) == 2
        expected_ids = shard.arrays['pulse_index'][rows].tolist()
        index = next(i for i, key in rs_cache.view_keys.items() if key == view)
        assert rs_cache.calibration[index]['native_pulse_count'] == len(rows)
        acquisition = source.acquisition(role, view, 'hh', dict(point_chunk=4, pair_chunk=2), 'cpu')
        response = source.response(role, view, 'hh', 'cpu')
        assert response.shape == (len(rows), 32)
        assert acquisition.tx.shape[0] == len(rows)
        seen = []
        for key in se.keys[role]:
            if key[:2] == view:
                seen.extend(obs.pulse_index for obs in se.observations(key, role=role))
        assert sorted(seen) == sorted(expected_ids)
    recipe = rf.recipe_from_config({}, ds.region.half_extent_m, len(ds.viewpoints('train')))
    rf_stats = rf.training_statistics(ds, recipe)
    assert rf_stats['pulse_counts'] == {'hh': 20}
    # Released RF retains 100 random-with-replacement profiles, drawn ONLY from
    # this shared eligible subset. Observe reads with lightweight geometry.
    view = ds.viewpoints('train')[0]
    seen = []
    original = rf.matched_range_power
    def power(obs, *args, **kwargs):
        seen.append(obs.pulse_index)
        return original(obs, *args, **kwargs)
    monkeypatch.setattr(rf, 'matched_range_power', power)
    monkeypatch.setattr(rf, 'prepare_bistatic_bins', lambda *a, **kw: {})
    frames = rf.prepare_frame(ds, view, 'hh', recipe, rf_stats, 'cpu', training=True)
    assert len(frames) == 100
    assert set(seen) <= set(shard.arrays['pulse_index'][shard.sector_rows[view[1]]])
    validation_frames = rf.prepare_frame(ds, ds.viewpoints('validation')[0], 'hh', recipe, rf_stats, 'cpu')
    assert len(validation_frames) == 2


def test_rift_subset_normalization_refinement_and_exact_resume(root, monkeypatch):
    from rift.gotcha_training import train, recipe_from_args
    ds = dataset(root, num_train=2)
    original_views = ds.viewpoints
    monkeypatch.setattr(ds, 'viewpoints', lambda role: original_views(role)[:1] if role == 'validation' else original_views(role))
    recipe = recipe_from_args(cli.parse_args(['--epochs', '2', '--granularity', '2', '--max-points', '15',
        '--sh-degree', '1', '--refine-every', '2', '--probe-every', '1', '--pulses-per-sector', '2']), 'rift')
    train(ds, 'rift', recipe, root/'full', device='cpu')
    original_step = torch.optim.AdamW.step
    def interrupt(opt, *a, **kw):
        result = original_step(opt, *a, **kw)
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        return result
    with monkeypatch.context() as m:
        m.setattr(torch.optim.AdamW, 'step', interrupt)
        assert train(ds, 'rift', recipe, root/'resumed', device='cpu')['status'] == 'interrupted'
    path = root/'resumed/checkpoint_latest.pt'
    saved = torch.load(path, weights_only=False)
    # Counts are projected range coefficients, not raw frequency bins. These
    # two synthetic sector geometries have 10 and 11 retained coefficients.
    assert saved['training_statistics']['hh']['count'] == 2*(10+11)
    assert saved['cursor'] == 1 and saved['updates'] == 1
    other = dataset(root, 3, num_train=2)
    with monkeypatch.context() as m:
        m.setattr(NativeShardReader, 'read', lambda *a: pytest.fail('Mismatch accessed responses'))
        with pytest.raises(ValueError, match='dataset'):
            train(other, 'rift', recipe, root/'wrong', device='cpu', resume=path)
    assert not (root/'wrong').exists()
    train(ds, 'rift', recipe, root/'resumed', device='cpu', resume=path)
    full = torch.load(root/'full/checkpoint_final.pt', weights_only=False)
    resumed = torch.load(root/'resumed/checkpoint_final.pt', weights_only=False)
    assert full['history'] == resumed['history']
    val_pulses = len(ds.shards[1, 'hh'].sector_rows[ds.viewpoints('validation')[0][1]])
    assert val_pulses == 2
    assert full['history'][-1]['validation']['by_polarization']['hh']['pulses'] == val_pulses
    for key, value in full['model_state_dict'].items():
        torch.testing.assert_close(value, resumed['model_state_dict'][key], rtol=0, atol=0)
    for key, state in full['optimizer_state_dict']['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(value, resumed['optimizer_state_dict']['state'][key][name], rtol=0, atol=0)


def test_spinr_validation_scores_complete_selected_role(root, monkeypatch):
    from rift import spinr_gotcha_training as runtime
    from tests.test_spinr_gotcha import TinyField
    ds = dataset(root, num_train=10)
    views = ds.viewpoints
    monkeypatch.setattr(ds, 'viewpoints', lambda role: views(role)[:1] if role == 'validation' else views(role))
    recipe = runtime.recipe_from_config({}, ds)
    heads = torch.nn.ModuleDict({'hh': TinyField(support_m=ds.region.half_extent_m)})
    points = torch.zeros((1, 3), dtype=torch.float64)
    stats = runtime.training_statistics(runtime.PulsePlan(ds))
    result = runtime.evaluate(heads, ds, points, torch.tensor([.001], dtype=torch.float64),
                              stats, {'hh': {'value': 1.}}, recipe, 'cpu')
    totals = result['by_polarization']['hh']
    assert totals['pulses'] == 2 and totals['samples'] == 64
    shard = ds.shards[1, 'hh']
    rows = shard.sector_rows[ds.viewpoints('validation')[0][1]]
    expected_energy = 32*sum((shard.arrays['pulse_index'][rows]+1)**2)
    assert totals['full_energy'] == pytest.approx(expected_energy)
    assert result['viewpoints'] == 1 and not result['test_accessed']


def test_radarsplat_readout_restores_cap_and_rejects_changed_selection(root, monkeypatch):
    from rift import radarsplat_gotcha as rs
    ds = dataset(root, num_train=4)
    run = root/'run'
    run.mkdir()
    (run/'source.json').write_text(json.dumps(dict(root=str(root), shard_root=str(ds.shard_root),
        passes=list(ds.passes), polarizations=list(ds.polarizations), region=ds.region.as_dict(), num_train=4)))
    state = dict(schema=rs.CONTROL_SCHEMA, identity=rs.planning(ds))
    torch.save(state, run/rs.CONTROL_FILE)
    monkeypatch.setattr(NativeShardReader, 'read', lambda *a: pytest.fail('Readout rebound responses'))
    cache = rs.cache_from_run(run, 'hh')
    assert cache.dataset.identity == ds.identity and cache.dataset.pulses_per_sector == 2
    broken = copy.deepcopy(state)
    broken['identity']['dataset_contract']['training_pulse_selection']['pulses_per_sector'] = 3
    torch.save(broken, run/rs.CONTROL_FILE)
    with pytest.raises(ValueError, match='source/region/recipe changed'):
        rs.cache_from_run(run, 'hh')


def test_baseline_recovery_and_target_caches_reject_another_pulse_subset(root, monkeypatch):
    from rift import radar_fields_gotcha as rf, radarsplat_gotcha as rs
    from rift import geraf_source_training as geraf, sugavanam_ertin_paper_workflow as se
    from rift.geraf_source_data import GOTCHASourceData
    from rift.sugavanam_ertin_acquisition import GOTCHAAcquisition
    small, large = dataset(root, 2, num_train=10), dataset(root, 3, num_train=10)
    checkpoint = dict(schema='rift_gotcha_checkpoint_v1', dataset_contract=small.contract,
                      dataset_identity=small.identity, recipe={})
    monkeypatch.setattr(NativeShardReader, 'read', lambda *a: pytest.fail('Mismatch read a response'))
    # Shared native gate is used by both RIFT and SpINR; RF calls it before
    # model/statistics recovery. Only the pulse selection differs here.
    with pytest.raises(ValueError, match='dataset'):
        validate_checkpoint(checkpoint, large, {})
    with pytest.raises(ValueError, match='dataset'):
        rf.validate_resume(checkpoint, large, {}, 'cpu')
    source_small, source_large = GOTCHASourceData(small), GOTCHASourceData(large)
    with pytest.raises(ValueError, match='source/object/roles/recipe mismatch'):
        geraf.validate_checkpoint(dict(schema=geraf.SCHEMA+'_checkpoint',
            contract=source_small.contract, data_identity=source_small.identity, recipe={}), source_large, {})
    with pytest.raises(ValueError, match='acquisition or recipe changed'):
        se.validate_resume(dict(schema=se.SCHEMA, acquisition=GOTCHAAcquisition(small).identity, recipe={}),
                           GOTCHAAcquisition(large), {}, None)
    target_recipe = {'target_storage': 'lazy_trilinear_accumulated_only_v1'}
    targets = geraf.SourceTargets(root/'geraf_cache', source_small, target_recipe)
    targets.start()
    with pytest.raises(ValueError, match='cache identity mismatch'):
        geraf.SourceTargets(targets.root, source_large, target_recipe).start()
    cache = rs.GOTCHAPowerCache(small, 'hh', root/'rs_cache')
    cache.root.mkdir()
    (cache.root/rs.RECIPE_FILENAME).write_text(json.dumps(cache.recipe))
    with pytest.raises(ValueError, match='recipe mismatch'):
        rs.GOTCHAPowerCache(large, 'hh', cache.root)


def test_geraf_readout_reconstructs_saved_cap_before_validation(root, monkeypatch):
    from scripts import eval_geraf_source as readout
    from rift.geraf_source_data import GOTCHASourceData
    ds = dataset(root, num_train=4)
    checkpoint = root/'geraf.pt'
    torch.save(dict(contract=GOTCHASourceData(ds).contract), checkpoint)
    import rift.gotcha_dataset as ingress
    monkeypatch.setattr(ingress, 'load_region', lambda *_: tiny_region())
    monkeypatch.setattr(NativeShardReader, 'read', lambda *a: pytest.fail('Readout read before identity gate'))
    class Reconstructed(Exception):
        pass
    def inspect(saved, data, device):
        assert data.dataset.contract == ds.contract
        assert data.dataset.identity == ds.identity
        raise Reconstructed
    monkeypatch.setattr(readout, 'load_selected_models', inspect)
    with pytest.raises(Reconstructed):
        readout.main(['--dataset', 'gotcha', '--dataset-root', str(root), '--passes', '1',
            '--num-train', '4', '--checkpoint', str(checkpoint), '--cache-root', str(root/'cache'),
            '--output', str(root/'metrics.json')])
