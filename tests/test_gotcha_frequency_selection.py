"""Native-bin selection, role-specific physics and selected-acquisition recovery."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import signal

import numpy as np
import pytest
import torch

import train_gotcha_dataset as cli
from rift.gotcha_dataset import GOTCHADataset, NativeShardReader, Observation, C, validate_checkpoint
from rift.gotcha_frequency_selection import NativeFrequencySelection, selection_config
from rift.gotcha_training import RangeReadout, native_forward
from tests.test_gotcha_dataset import tiny_region, write_shard


@pytest.mark.parametrize('size', [31, 32, 424, 425, 434])
def test_stride_keeps_native_values_source_indices_and_bandwidth(size):
    f = np.linspace(9e9, 10e9, size)
    f[1::2] += 128
    selector = NativeFrequencySelection(f, 2)
    expected = np.unique(np.r_[np.arange(0, size, 2), size - 1])
    np.testing.assert_array_equal(selector.indices('train'), expected)
    np.testing.assert_array_equal(selector.frequencies('train'), f[expected])
    np.testing.assert_array_equal(selector.frequencies('validation'), f[expected])
    np.testing.assert_array_equal(selector.frequencies('test'), f[expected])
    assert selector.contract['source_indices'] == expected.tolist()
    assert selector.contract['endpoints_hz'] == [f[0], f[-1]]
    assert not selector.train.flags.writeable and not selector.train_indices.flags.writeable
    assert NativeFrequencySelection(f, 1).contract is None
    with pytest.raises(PermissionError):
        selector.frequencies('unused')


@pytest.mark.parametrize('stride', [0, -1, 3, 4, True, 2., '2'])
def test_unsupported_stride_fails_explicitly(stride):
    with pytest.raises(ValueError, match='frequency-stride'):
        selection_config(stride)


def test_projector_rebuilt_from_selected_bins_not_sliced_full_projection():
    f = np.linspace(9e9, 10e9, 64)
    f[1::2] += 128
    selector = NativeFrequencySelection(f, 2)
    obs = Observation(1, 'hh', 2, 0, np.array([20., 1., 2.]), f, 20.,
                      np.exp(.07j * np.arange(len(f))), 'synthetic')
    selected = replace(obs, frequencies_hz=selector.train, response=obs.response[selector.train_indices])
    reader = RangeReadout(tiny_region())
    full = reader.for_observation(obs)
    readout = reader.for_observation(selected)
    assert len(reader._bases) == 2 and readout['q'].shape[0] == len(selector.train)
    q = readout['q']
    torch.testing.assert_close(q.conj().T @ q, torch.eye(q.shape[1], dtype=q.dtype), rtol=1e-12, atol=1e-12)
    # A cropped full basis is not an orthonormal selected-frequency projector.
    cropped = full['q'][selector.train_indices.copy()]
    assert not torch.allclose(cropped.conj().T @ cropped, torch.eye(cropped.shape[1], dtype=cropped.dtype))
    points = torch.tensor([[.01, -.01, .015]], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([1. + .2j], dtype=torch.complex128, requires_grad=True)
    full_prediction = native_forward(points, weights, full['antenna'], full['frequencies'], obs.reference_range_m)
    prediction = native_forward(points, weights, readout['antenna'], readout['frequencies'], selected.reference_range_m)
    torch.testing.assert_close(prediction, full_prediction[selector.train_indices.copy()], atol=1e-12, rtol=1e-12)
    projected = reader.project(prediction, readout)
    projected.abs().square().sum().backward()
    assert torch.isfinite(points.grad).all() and torch.isfinite(weights.grad).all()
    assert float(projected.abs().square().sum() / prediction.abs().square().sum()) > .99


def _multipulse(arrays, metadata):
    for key, value in list(arrays.items()):
        if key not in ('frequencies_hz',) and len(value) == 360:
            arrays[key] = np.repeat(value, 3, axis=0)
    arrays['pulse_index'] = np.tile(np.arange(3, dtype=np.int32), 360)
    arrays['z'] += .001 * arrays['pulse_index']
    arrays['r0'] = np.sqrt(sum(arrays[k]**2 for k in ('x', 'y', 'z')))
    n = len(arrays['frequencies_hz'])
    arrays['response'] *= (1 + .2 * np.arange(n))[None, :]


@pytest.fixture
def native_data(tmp_path):
    for p, nf in ((1, 64), (2, 67)):
        for pol in ('hh', 'hv'):
            write_shard(tmp_path/'New_Transfer'/'shards'/f'pass{p}_{pol}.npz', p, pol, nf, _multipulse)
    return tmp_path


def dataset(root, stride=2, cap=2, **kwargs):
    return GOTCHADataset(root, passes=(1, 2), region=tiny_region(), num_train=10,
                        frequency_stride=stride, pulses_per_sector=cap, **kwargs)


def test_shared_reader_preserves_source_geometry_autofocus_and_sealed_holdouts(native_data, monkeypatch):
    full = dataset(native_data, 1, 0, polarizations=('hh', 'hv'))
    with monkeypatch.context() as m:
        m.setattr(np, 'memmap', lambda *a, **kw: pytest.fail('Preflight mapped responses'))
        selected = dataset(native_data, polarizations=('hh', 'hv'))
    assert selected.frequency_preflight['support_rank_alias_checks'] == 'passed'
    for key, shard in selected.shards.items():
        source = full.shards[key]
        assert shard.shape == source.shape and shard.identity == source.identity
        np.testing.assert_array_equal(shard.frequencies_hz, source.frequencies_hz)
        test_sectors = selected.splits_by_pass[key[0]]['test']
        assert all(len(shard.sector_rows[s]) == 2 for s in test_sectors)
        np.testing.assert_array_equal(shard.frequencies_for_role('test'), shard.frequencies_for_role('train'))
        for role in ('train', 'validation'):
            sector = next(s for p, s in selected.viewpoints(role) if p == key[0])
            rows = shard.sector_rows[sector]
            assert len(rows) == 2
            bins = shard.frequency_indices_for_role(role)
            for row in rows:
                obs, original = shard.read(int(row)), source.read(int(row))
                np.testing.assert_array_equal(obs.response, original.response[bins])
                np.testing.assert_array_equal(obs.frequencies_hz, original.frequencies_hz[bins])
                np.testing.assert_array_equal(obs.position_m, original.position_m)
                assert obs.reference_range_m == original.reference_range_m
                assert obs.autofocus == original.autofocus and obs.pulse_index == original.pulse_index
        with monkeypatch.context() as m:
            m.setattr(np, 'memmap', lambda *a, **kw: pytest.fail('Denied response mapped'))
            for role in ('test', 'excluded'):
                rows = np.flatnonzero(shard.row_roles == role)
                if len(rows):
                    with pytest.raises(PermissionError):
                        shard.read(int(rows[0]))


def test_all_six_plans_share_both_selections_and_default_identity_is_unchanged(native_data, monkeypatch):
    methods = ['rift', 'spinr', 'radar_fields', 'geraf', 'radarsplat', 'sugavanam_ertin']
    monkeypatch.setattr(cli, 'load_region', lambda *_: tiny_region())
    monkeypatch.setattr(NativeShardReader, 'read', lambda *a: pytest.fail('Plan read responses'))
    ds, report = cli.make_plan(cli.parse_args(['--dataset-root', str(native_data), '--passes', '1', '2',
        '--num-train', '10', '--method', *methods, '--pulses-per-sector', '2', '--frequency-stride', '2',
        '--output-root', str(native_data/'runs')]))
    assert [p['method'] for p in report['plans']] == methods
    for plan in report['plans']:
        assert plan['training_frequency_selection'] == ds.frequency_selection
        assert plan['training_pulse_selection'] == ds.training_pulse_selection
        assert Path(plan['output_dir']).parent.name == 'frequency_stride2_all_roles_v2'
        assert Path(plan['output_dir']).parent.parent.name == 'pulse_subset2_all_roles_v2'
    assert 'same_selected_native_bins' in report['plans'][0]['config']['frequency_policy']
    assert 'same_selected_native_bins' in report['plans'][1]['native_plan']['recipe']['frequency_policy']
    assert not (native_data/'runs').exists()
    default = GOTCHADataset(native_data, passes=(1, 2), region=tiny_region(), num_train=10)
    explicit = dataset(native_data, 1, 0)
    assert default.contract == explicit.contract and default.identity == explicit.identity
    assert 'frequency_selection' not in default.contract
    assert len({ds.identity, default.identity, dataset(native_data, 1).identity,
                dataset(native_data, 2, 0).identity}) == 4


def test_baseline_counts_normalization_and_operators_use_role_frequencies(native_data):
    from rift.spinr_gotcha_training import PulsePlan, training_statistics
    from rift.geraf_source_data import GOTCHASourceData
    from rift.sugavanam_ertin_acquisition import GOTCHAAcquisition
    from rift import radar_fields_gotcha as rf, radarsplat_gotcha as rs
    ds = dataset(native_data)
    plan = PulsePlan(ds)
    stats = training_statistics(plan)['hh']
    expected_samples = sum(len(s.sector_rows[v])*len(s.frequencies_for_role('train'))
        for p, v in ds.viewpoints('train') for s in [ds.shards[p, 'hh']])
    expected_energy = sum(np.vdot(plan.read(i).response, plan.read(i).response).real for i in range(len(plan.records)))
    assert stats['pulses'] == 20 and stats['samples'] == expected_samples
    assert stats['mean_power'] == pytest.approx(expected_energy/expected_samples)
    se, geraf = GOTCHAAcquisition(ds), GOTCHASourceData(ds)
    assert sum(se.train_sample_counts) == expected_samples
    points = torch.tensor([[.01, -.01, .015]], dtype=torch.float64)
    weights = torch.tensor([1+.2j], dtype=torch.complex128)
    cache = rs.GOTCHAPowerCache(ds, 'hh', native_data/'targets')
    for role in ('train', 'validation'):
        view = ds.viewpoints(role)[0]
        shard = ds.shards[view[0], 'hh']
        f = shard.frequencies_for_role(role)
        acq = geraf.acquisition(role, view, 'hh', dict(point_chunk=4, pair_chunk=2), 'cpu')
        response = geraf.response(role, view, 'hh', 'cpu')
        torch.testing.assert_close(acq.frequencies, torch.from_numpy(f.copy()))
        assert response.shape == (2, len(f))
        # Preserve GeRaF's released antenna-mean / frequency-SUM normalization.
        expected_mf = (acq.phase(points, torch.arange(len(response))).conj()*response[:, None]).sum((0, 2))/len(response)
        torch.testing.assert_close(acq.matched_filter(response, points), expected_mf)
        obs = next(ds.observations(*view, 'hh'))
        ranges = torch.tensor([obs.reference_range_m, obs.reference_range_m+.001], dtype=torch.float64)
        phase = 4j*torch.pi/C*(ranges[:, None]-obs.reference_range_m)*(torch.from_numpy(f.copy())-f.mean())
        expected_rf = (torch.exp(phase)@torch.from_numpy(obs.response)/len(f)).abs().square()
        torch.testing.assert_close(rf.matched_range_power(obs, ranges), expected_rf)
        se_key = next(k for k in se.keys[role] if k[:2] == view)
        se_obs = next(se.observations(se_key, role=role))
        prediction = se.render(points, weights, se_obs, point_chunk=4, pair_chunk=1)
        assert prediction.shape == (len(f),) and torch.isfinite(prediction).all()
        # Selected SE render is exactly the corresponding full-grid prediction.
        original = dataset(native_data, 1, 0).shards[view[0], 'hh'].read(int(shard.sector_rows[view[1]][0]))
        full = se.render(points, weights, original, point_chunk=4, pair_chunk=1)
        torch.testing.assert_close(prediction, full[shard.frequency_indices_for_role(role).copy()])
        cache_id = next(i for i, key in cache.view_keys.items() if key == view)
        assert cache.calibration[cache_id]['native_pulse_count'] == response.shape[0]
    recipe = rf.recipe_from_config({}, ds.region.half_extent_m, 10)
    rf_stats = rf.training_statistics(ds, recipe)
    expected_peak = max(float(rf.matched_range_power(obs, rf.range_geometry(obs, ds.region,
        recipe['controls']['range_guard_cells'])[1]).max())
        for p, s in ds.viewpoints('train') for obs in ds.observations(p, s, 'hh'))
    assert rf_stats['peak_power']['hh'] == expected_peak and rf_stats['pulse_counts']['hh'] == 20


def test_projector_support_gate_is_preserved_and_saved_contract_is_strict(native_data, monkeypatch):
    from rift.gotcha_frequency_selection import kwargs_from_contract
    monkeypatch.setattr(NativeShardReader, 'read', lambda *a: pytest.fail('Preflight read responses'))
    ds = dataset(native_data)
    assert kwargs_from_contract(ds.contract) == {'frequency_stride': 2}
    assert kwargs_from_contract(dataset(native_data, 1).contract) == {}
    for key in ('range_projector', 'normalization', 'objective', 'schema', 'roles'):
        broken = deepcopy(ds.contract)
        broken['frequency_selection'][key] = 'changed'
        with pytest.raises(ValueError, match='frequency-selection'):
            kwargs_from_contract(broken)
    with pytest.raises(ValueError, match='Frequency selection.*native'):
        GOTCHADataset(native_data, passes=(1, 2), frequency_stride=2,
                      region=replace(tiny_region(), half_extent_m=3.))


@pytest.mark.parametrize('change', ['stride', 'bins', 'source_hash', 'projector', 'normalization'])
def test_all_baseline_checkpoint_and_cache_gates_reject_frequency_changes(native_data, monkeypatch, change):
    from rift import radar_fields_gotcha as rf, radarsplat_gotcha as rs
    from rift import geraf_source_training as geraf, sugavanam_ertin_paper_workflow as se
    from rift.geraf_source_data import GOTCHASourceData
    from rift.sugavanam_ertin_acquisition import GOTCHAAcquisition
    small, other = dataset(native_data), dataset(native_data, 1 if change == 'stride' else 2)
    if change != 'stride':
        selected = other.contract['frequency_selection']
        if change == 'bins': selected['shards']['pass1_hh']['source_indices'][1] = 3
        elif change == 'source_hash': selected['shards']['pass1_hh']['source_frequencies_sha256'] = 'wrong'
        else: selected['range_projector' if change == 'projector' else 'normalization'] = 'wrong'
    checkpoint = dict(schema='rift_gotcha_checkpoint_v1', dataset_contract=small.contract,
                      dataset_identity=small.identity, recipe={})
    monkeypatch.setattr(NativeShardReader, 'read', lambda *a: pytest.fail('Mismatch read responses'))
    with pytest.raises(ValueError, match='dataset'):
        validate_checkpoint(checkpoint, other, {})
    with pytest.raises(ValueError, match='dataset'):
        rf.validate_resume(checkpoint, other, {}, 'cpu')
    source_small, source_other = GOTCHASourceData(small), GOTCHASourceData(other)
    with pytest.raises(ValueError, match='source/object/roles/recipe mismatch'):
        geraf.validate_checkpoint(dict(schema=geraf.SCHEMA+'_checkpoint', contract=source_small.contract,
            data_identity=source_small.identity, recipe={}), source_other, {})
    with pytest.raises(ValueError, match='acquisition or recipe changed'):
        se.validate_resume(dict(schema=se.SCHEMA, acquisition=GOTCHAAcquisition(small).identity, recipe={}),
                           GOTCHAAcquisition(other), {}, None)
    targets = geraf.SourceTargets(native_data/'geraf_cache', source_small,
                                  {'target_storage': 'lazy_trilinear_accumulated_only_v1'})
    targets.start()
    with pytest.raises(ValueError, match='cache identity mismatch'):
        geraf.SourceTargets(targets.root, source_other, targets.recipe).start()
    cache = rs.GOTCHAPowerCache(small, 'hh', native_data/'rs_cache')
    cache.root.mkdir()
    (cache.root/rs.RECIPE_FILENAME).write_text(json.dumps(cache.recipe))
    with pytest.raises(ValueError, match='recipe mismatch'):
        rs.GOTCHAPowerCache(other, 'hh', cache.root)


def test_radarsplat_actual_power_conversion_uses_selected_bins_and_selected_validation(native_data, monkeypatch):
    from rift import radarsplat_gotcha as rs, radarsplat_fidelity
    ds = dataset(native_data)
    xyz = np.array([[.01, 0., .01], [-.01, .01, 0.]])
    monkeypatch.setattr(radarsplat_fidelity, 'polar_world_points', lambda _: xyz)
    for role in ('train', 'validation'):
        view = ds.viewpoints(role)[0]
        obs = list(ds.observations(*view, 'hh'))
        calibration = dict(native_pulse_count=len(obs), elevation_rad=[0.], azimuth_rad=[0.], range_m=[0., 1.])
        actual = rs.sector_power(ds, *view, 'hh', calibration,
                                dict(point_chunk=1, frequency_chunk=7), device='cpu')
        coherent = np.zeros(len(xyz), dtype=np.complex128)
        for o in obs:
            distance = np.linalg.norm(xyz-ds.region.to_local(o.position_m), axis=-1)-o.reference_range_m
            coherent += np.mean(np.exp(4j*np.pi/C*distance[:, None]*o.frequencies_hz)*o.response, axis=1)
        expected = np.abs(coherent/len(obs))**2
        np.testing.assert_allclose(actual[0], expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('method', ['rift', 'spinr'])
def test_combined_selection_exact_interrupted_recovery_and_selected_validation(native_data, monkeypatch, method):
    from rift import gotcha_training as rift, spinr_gotcha_training as spinr
    from tests.test_spinr_gotcha import TinyField, config, assert_tree_equal
    ds = dataset(native_data)
    original_views = ds.viewpoints
    monkeypatch.setattr(ds, 'viewpoints', lambda role: original_views(role)[:1] if role == 'validation' else original_views(role))
    if method == 'spinr':
        monkeypatch.setattr(spinr, 'SpinrStyleINR', TinyField)
        options = dict(config(), pulse_batch_size=6)
        run = lambda path, **kwargs: spinr.run(ds, path, options, device='cpu', **kwargs)
        checkpoint_name = 'checkpoint_latest.pt'
    else:
        recipe = rift.recipe_from_args(cli.parse_args(['--epochs', '2', '--granularity', '2', '--max-points', '15',
            '--sh-degree', '1', '--refine-every', '2', '--probe-every', '1']), 'rift')
        run = lambda path, **kwargs: rift.train(ds, 'rift', recipe, path, device='cpu', **kwargs)
        checkpoint_name = 'checkpoint_final.pt'
    run(native_data/'full')
    original_step = (torch.optim.Adam if method == 'spinr' else torch.optim.AdamW).step
    optimizer_class = torch.optim.Adam if method == 'spinr' else torch.optim.AdamW
    def interrupt(opt, *args, **kwargs):
        result = original_step(opt, *args, **kwargs)
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        return result
    with monkeypatch.context() as m:
        m.setattr(optimizer_class, 'step', interrupt)
        assert run(native_data/'resumed')['status'] == 'interrupted'
    path = native_data/'resumed/checkpoint_latest.pt'
    saved = torch.load(path, weights_only=False)
    assert saved['cursor'] == 1
    assert saved['recipe']['training_frequency_selection'] == ds.frequency_selection
    assert saved['recipe']['training_pulse_selection'] == ds.training_pulse_selection
    if method == 'spinr':
        assert saved['training_statistics']['hh']['samples'] == 10*(33+34)
    else:
        expected = rift.training_statistics(ds, RangeReadout(ds.region))
        assert saved['training_statistics'] == expected
    run(native_data/'resumed', resume=path)
    full = torch.load(native_data/'full'/checkpoint_name, weights_only=False)
    resumed = torch.load(native_data/'resumed'/checkpoint_name, weights_only=False)
    for key in ('model_state_dict', 'optimizer_state_dict', 'history', 'rng_state', 'training_statistics'):
        assert_tree_equal(full[key], resumed[key])
    validation = full['history'][-1]['validation']['by_polarization']['hh']
    expected_energy = sum(float(np.vdot(obs.response, obs.response).real)
                          for p, s in ds.viewpoints('validation') for obs in ds.observations(p, s, 'hh'))
    assert validation['pulses'] == 2
    assert validation['full_energy' if method == 'spinr' else 'full_native_target_energy'] == pytest.approx(expected_energy)
    if method == 'spinr':
        assert validation['samples'] == 2*33
        assert_tree_equal(full['optimization_coverage'], resumed['optimization_coverage'])
        assert_tree_equal(full['scheduler_state_dict'], resumed['scheduler_state_dict'])
    # Changing frequency selection alone is refused before any payload read.
    old = dataset(native_data, 1)
    monkeypatch.setattr(NativeShardReader, 'read', lambda *a: pytest.fail('Recovery mismatch read responses'))
    with pytest.raises(ValueError, match='dataset'):
        if method == 'spinr':
            spinr.run(old, native_data/'resumed', options, device='cpu', resume=path)
        else:
            rift.train(old, 'rift', recipe, native_data/'wrong', device='cpu', resume=path)


@pytest.mark.parametrize('method', ['geraf', 'radarsplat'])
def test_readouts_restore_both_selection_controls_before_identity_gate(native_data, monkeypatch, method):
    from rift import radarsplat_gotcha as rs
    from rift.geraf_source_data import GOTCHASourceData
    ds = dataset(native_data)
    run = native_data/'run'
    run.mkdir()
    monkeypatch.setattr(NativeShardReader, 'read', lambda *a: pytest.fail('Readout read before identity gate'))
    if method == 'radarsplat':
        (run/'source.json').write_text(json.dumps(dict(root=str(native_data), shard_root=str(ds.shard_root),
            passes=list(ds.passes), polarizations=list(ds.polarizations), region=ds.region.as_dict(), num_train=10)))
        state = dict(schema=rs.CONTROL_SCHEMA, identity=rs.planning(ds))
        torch.save(state, run/rs.CONTROL_FILE)
        cache = rs.cache_from_run(run, 'hh')
        assert cache.dataset.identity == ds.identity and cache.dataset.contract == ds.contract
        broken = deepcopy(state)
        broken['identity']['dataset_contract']['frequency_selection']['shards']['pass1_hh']['source_indices'][1] = 3
        torch.save(broken, run/rs.CONTROL_FILE)
        with pytest.raises(ValueError, match='source/region/recipe changed'):
            rs.cache_from_run(run, 'hh')
    else:
        from scripts import eval_geraf_source as readout
        import rift.gotcha_dataset as ingress
        checkpoint = run/'geraf.pt'
        torch.save(dict(contract=GOTCHASourceData(ds).contract), checkpoint)
        monkeypatch.setattr(ingress, 'load_region', lambda *_: tiny_region())
        class Reconstructed(Exception):
            pass
        def inspect(saved, data, device):
            assert data.dataset.contract == ds.contract and data.dataset.identity == ds.identity
            raise Reconstructed
        monkeypatch.setattr(readout, 'load_selected_models', inspect)
        with pytest.raises(Reconstructed):
            readout.main(['--dataset', 'gotcha', '--dataset-root', str(native_data), '--passes', '1', '2',
                '--num-train', '10', '--checkpoint', str(checkpoint), '--cache-root', str(run/'cache'),
                '--output', str(run/'metrics.json')])
