"""PVC twins of the RIFT-dataset evaluators: train2400 range-power stats and Radar Fields geometry contracts."""
import inspect

import pytest

import scripts.eval_b787_geometry_metrics as geometry_original
import scripts.eval_b787_range_power as power_original
import scripts.render_b787_vs_stl as render_original
from scripts_pvc import eval_b787_geometry_metrics_pvc as geometry
from scripts_pvc import eval_b787_range_power_pvc as power

POWER_ORIGINAL_LINES = '''    train_indices = _ordered_role_ids(
        checkpoint, "train", num_views=num_views, expected_count=3200
    )'''
POWER_ADAPTED_LINES = '''    # PVC adaptation: the TRAIN count comes from the checkpoint's own contract (train2400
    # collection runs); the sealed B787 split keeps 3200. The stats' TRAIN IDs must still
    # equal the checkpoint's TRAIN role exactly (checked below).
    train_indices = _ordered_role_ids(
        checkpoint, "train", num_views=num_views, expected_count=expected_train_count(checkpoint)
    )'''
GEOMETRY_ORIGINAL_LINES = '''        validate_checkpoint_object(ck, expected_contract)
        saved_contract = ck.get("sealed_npz_protocol_contract", {})
        if "training_selection" in expected_contract or "training_selection" in saved_contract:
            from train import _validate_saved_sealed_npz_protocol_contract
            _validate_saved_sealed_npz_protocol_contract(saved_contract, expected_contract)
'''
GEOMETRY_ADAPTED_LINES = '''        validate_checkpoint_object(ck, expected_contract)
        if "sealed_npz_protocol_contract" not in ck and isinstance(ck.get("sealed_protocol_contract"), dict):
            # PVC adaptation: Radar Fields keeps its own contract schema
            # (train_radar_fields.sealed_protocol_contract); check its selection in its own terms.
            validate_radar_fields_selection(ck["sealed_protocol_contract"], expected_contract)
        else:
            saved_contract = ck.get("sealed_npz_protocol_contract", {})
            if "training_selection" in expected_contract or "training_selection" in saved_contract:
                from train import _validate_saved_sealed_npz_protocol_contract
                _validate_saved_sealed_npz_protocol_contract(saved_contract, expected_contract)
'''
ANTENNAS = {'schema': 'rift_ordered_source_antennas_v1', 'num_tx': 1, 'num_rx': 1, 'tx_indices': [0], 'rx_indices': [0]}
EXPECTED = {'role_manifest_name': 'rift_dataset_airliner_a320_seed42_train2400_val1000_test1000_v1_1t1r_2010b2bbe725',
            'antenna_selection': ANTENNAS}


def test_copies_differ_from_the_originals_only_by_their_adaptation():
    twin = inspect.getsource(power.validate_normalization_stats)
    assert twin.count(POWER_ADAPTED_LINES) == 1
    assert twin.replace(POWER_ADAPTED_LINES, POWER_ORIGINAL_LINES) == inspect.getsource(
        power_original.validate_normalization_stats)
    twin = inspect.getsource(geometry.load_energy_field)
    assert twin.count(GEOMETRY_ADAPTED_LINES) == 1
    assert twin.replace(GEOMETRY_ADAPTED_LINES, GEOMETRY_ORIGINAL_LINES) == inspect.getsource(
        render_original.load_energy_field)


def test_train_count_comes_from_the_checkpoint_contract():
    collection = {'sealed_npz_protocol_contract': {'training_selection': {'num_train': 2400}}}
    assert power.expected_train_count(collection) == 2400
    assert power.expected_train_count({'sealed_npz_protocol_contract': {'role_ids': {}}}) == 3200
    assert power.expected_train_count({}) == 3200


def test_radar_fields_selection_is_checked_in_its_own_terms():
    saved = {'manifest_name': EXPECTED['role_manifest_name'], 'acquisition_identity': {'antenna_selection': ANTENNAS}}
    geometry.validate_radar_fields_selection(saved, EXPECTED)
    with pytest.raises(ValueError, match='manifest'):
        geometry.validate_radar_fields_selection(dict(saved, manifest_name=EXPECTED['role_manifest_name'].replace(
            'train2400', 'train3200')), EXPECTED)
    with pytest.raises(ValueError, match='antenna'):
        geometry.validate_radar_fields_selection(
            dict(saved, acquisition_identity={'antenna_selection': dict(ANTENNAS, num_tx=16)}), EXPECTED)
    with pytest.raises(ValueError, match='manifest'):
        geometry.validate_radar_fields_selection(saved, dict(EXPECTED, role_manifest_name=None))


def test_wrappers_run_the_original_main_with_the_twin_function(monkeypatch):
    monkeypatch.setattr(power_original, 'validate_normalization_stats', power_original.validate_normalization_stats)
    monkeypatch.setattr(geometry_original, 'load_energy_field', geometry_original.load_energy_field)
    with pytest.raises(SystemExit):
        power.main(['--help'])
    assert power_original.validate_normalization_stats is power.validate_normalization_stats
    with pytest.raises(SystemExit):
        geometry.main(['--help'])
    assert geometry_original.load_energy_field is geometry.load_any_field


# --- held-out scoring harness (scripts_pvc/eval_rift_dataset_heldout_pvc.py) ------------------------------------

from scripts_pvc import eval_rift_dataset_heldout_pvc as heldout

HELDOUT_ARGV = ['--method', 'spinr', '--checkpoint', 'c', '--label', 'l', '--object', 'a320', '--dataset-root', '/nonexistent',
                '--role-manifest', 'm.json', '--stats', 's.json', '--out-dir', 'o']


def test_heldout_test_role_needs_the_explicit_opt_in(monkeypatch):
    monkeypatch.setattr('rift.rift_dataset.resolve_object_inputs', lambda **kw: ('x.npz', 'm.json'))
    with pytest.raises(SystemExit):
        heldout.parse_args(HELDOUT_ARGV + ['--role', 'test'])
    args = heldout.parse_args(HELDOUT_ARGV + ['--role', 'test', '--allow-reserved-test'])
    assert args.role == 'test' and args.allow_reserved_test
    assert heldout.parse_args(HELDOUT_ARGV).role == 'validation'


def test_heldout_adapters_declare_their_prediction_kind():
    assert set(heldout.ADAPTERS) == {'spinr', 'geraf', 'radar_fields', 'sugavanam_ertin', 'radarsplat'}
    assert (heldout.ADAPTERS['spinr'].kind == heldout.ADAPTERS['geraf'].kind
            == heldout.ADAPTERS['sugavanam_ertin'].kind == 'complex')
    assert heldout.ADAPTERS['radar_fields'].kind == 'intensity'
    assert heldout.ADAPTERS['radarsplat'].kind == 'power_profile'


def test_single_pair_profile_map_reproduces_a_matched_filter_image():
    """|A|^2 = N^2 |IFFT|^2 at the bistatic bin index, summed over elevation, for one antenna pair."""
    import numpy as np
    rng = np.random.default_rng(0)
    freqs = np.linspace(9.5e9, 10.5e9, 64, endpoint=False)
    signal = rng.standard_normal(64) + 1j * rng.standard_normal(64)
    tx, rx = np.array([10.0, 0.02, 0.0]), np.array([10.0, -0.02, 0.0])
    ne, npix = 3, 40
    points = np.concatenate([np.stack([np.linspace(0.1, -0.1, npix), np.full(npix, e), np.zeros(npix)], 1)
                             for e in (-0.01, 0.0, 0.01)])
    matrix, nodes, n = heldout.single_pair_profile_map(points, tx, rx, freqs, ne, npix, step=0.02)
    R = np.linalg.norm(points - tx, axis=1) + np.linalg.norm(points - rx, axis=1)
    image = (np.abs(np.exp(2j * np.pi * np.outer(R, freqs) / 299792458.0) @ signal) ** 2).reshape(ne, npix).sum(0)
    bandwidth = (freqs[-1] - freqs[0]) * 64 / 63
    k = np.arange(64)
    profile = np.abs(np.exp(2j * np.pi * np.outer(nodes, k) / 64) @ signal / 64) ** 2     # continuous |IFFT|^2
    assert np.allclose(bandwidth, 1e9)
    assert np.linalg.norm(matrix @ profile - image) / np.linalg.norm(image) < 5e-3     # linear-interpolation error only


def test_power_only_methods_never_report_a_pooled_zero_coherent_score():
    source = inspect.getsource(heldout.main)
    assert "if adapter.kind != 'complex':" in source and "result['coherent_complex_rel_mse'] = None" in source
    assert "the method ROI differs from the evaluator ROI" in source


def test_nonnegative_profile_fit_is_exact_for_consistent_and_bounded_for_inconsistent_images():
    import numpy as np
    rng = np.random.default_rng(1)
    freqs = np.linspace(9.5e9, 10.5e9, 64, endpoint=False)
    signal = rng.standard_normal(64) + 1j * rng.standard_normal(64)
    tx, rx = np.array([10.0, 0.02, 0.0]), np.array([10.0, -0.02, 0.0])
    ne, npix = 3, 40
    points = np.concatenate([np.stack([np.linspace(0.1, -0.1, npix), np.full(npix, e), np.zeros(npix)], 1)
                             for e in (-0.01, 0.0, 0.01)])
    matrix, nodes, n = heldout.single_pair_profile_map(points, tx, rx, freqs, ne, npix, step=0.02)
    k = np.arange(64)
    truth = np.abs(np.exp(2j * np.pi * np.outer(nodes, k) / 64) @ signal / 64) ** 2
    consistent = matrix @ truth
    fitted = heldout.nonnegative_profile_fit(matrix, consistent, 64, ne)
    assert np.linalg.norm(matrix @ fitted - consistent) / np.linalg.norm(consistent) < 1e-6
    inconsistent = rng.random(npix) * consistent.mean()          # no single antenna pair produces this
    fitted = heldout.nonnegative_profile_fit(matrix, inconsistent, 64, ne)
    assert (fitted >= 0).all() and fitted.max() <= 10 * inconsistent.max() / (64 ** 2 * ne)


def _brute_force_floor(values, groups):
    """Best constant per group by direct search over levels, pooled RelMSE."""
    import numpy as np
    error = 0.0
    for group in groups:
        v = values[group]
        levels = np.linspace(v.min(), v.max(), 2001)
        error += min(float(((v - c) ** 2).sum()) for c in levels)
    return error / float((values ** 2).sum())


def test_constant_floors_match_a_direct_search_on_the_cache():
    import numpy as np
    rng = np.random.default_rng(3)
    views, bins, peak, dr = 5, 12, 2.0, 60.0
    t = rng.uniform(0.2, 0.9, (views, bins))
    t[0, :3] = 0.0                                             # clamped bins
    mask = rng.random((views, bins)) < 0.7
    mask[:, 0] = True
    cache = dict(range_power_rel_mse=np.ones(views), target_profile=t.astype(np.float32), roi_mask=mask,
                 count_db=mask.sum(1).astype(float), target_sq_db=np.where(mask, t.astype(np.float32) ** 2, 0).sum(1))
    floors = heldout.constant_floors(cache, peak, dr)
    scored = t.astype(np.float32).astype(np.float64)[mask]
    rows = np.repeat(np.arange(views)[:, None], bins, 1)[mask]
    assert floors['zero_prediction'] == dict(normalized=1.0, linear=1.0)
    assert floors['normalized']['best_constant'] == pytest.approx(_brute_force_floor(scored, [slice(None)]), rel=1e-4)
    assert floors['normalized']['best_per_view_constant'] == pytest.approx(
        _brute_force_floor(scored, [rows == v for v in range(views)]), rel=1e-4)
    linear = np.where(scored > 0, peak * 10 ** (dr * (scored - 1) / 10), 0.0)
    assert floors['linear']['best_constant'] == pytest.approx(_brute_force_floor(linear, [slice(None)]), rel=1e-4)
    cache['target_sq_db'] = cache['target_sq_db'] * 2          # multi-pair cache: the profile no longer suffices
    assert heldout.constant_floors(cache, peak, dr) is None


def test_gotcha_floors_pool_per_polarization_and_sector():
    import numpy as np
    from scripts_pvc import eval_gotcha_heldout_pvc as gotcha
    rng = np.random.default_rng(4)
    sectors = {pol: [rng.uniform(0, 1, n) for n in (7, 9, 4)] for pol in ('hh', 'vv')}
    totals = {}
    for pol, parts in sectors.items():
        values = np.concatenate(parts)
        totals[pol] = dict(db_energy=float((values ** 2).sum()), db_sum=float(values.sum()), bins=len(values),
                           db_sector_spread=sum(float(((p - p.mean()) ** 2).sum()) for p in parts),
                           linear_energy=0.0, linear_sum=0.0, linear_sector_spread=0.0)
    floors = gotcha.constant_floors(totals)
    energy = sum(t['db_energy'] for t in totals.values())
    per_pol = sum(_brute_force_floor(np.concatenate(p), [slice(None)]) * float((np.concatenate(p) ** 2).sum())
                  for p in sectors.values()) / energy
    per_sector = sum(_brute_force_floor(np.concatenate(p), [slice(sum(map(len, p[:i])), sum(map(len, p[:i + 1])))
                                                            for i in range(len(p))]) * float((np.concatenate(p) ** 2).sum())
                     for p in sectors.values()) / energy
    assert floors['normalized']['best_constant'] == pytest.approx(per_pol, rel=1e-4)
    assert floors['normalized']['best_per_sector_constant'] == pytest.approx(per_sector, rel=1e-4)
    assert floors['linear'] is None and floors['zero_prediction'] == dict(normalized=1.0, linear=1.0)
