"""GOTCHA presentation of RadarSplat: relative occupancy union and K-calibrated RCS."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from rift.gotcha_dataset import C, GOTCHADataset, NativeShardReader, Region, SOURCE_SCHEMA, sector_split
from rift.radarsplat_release import REFERENCE_ROOT
from rift_pvc import gotcha_presentation as presentation
from rift_pvc import gotcha_presentation_power as power
from rift_pvc import gotcha_presentation_radarsplat as rs

pytestmark = pytest.mark.skipif(not (REFERENCE_ROOT / 'gsplat/rendering.py').is_file(),
                                reason='pinned RadarSplat source not staged (scripts/fetch_radarsplat_reference.py)')
CONFIG = dict(azimuth_samples=11, elevation_samples=3, point_chunk=2048, frequency_chunk=64)
IDENTITY = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
SH_C0 = 0.28209479177387814


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv('RIFT_ACCELERATOR', 'cpu')
    monkeypatch.setenv('RIFT_PVC_ALLOW_BACKEND', 'cpu')


def write_far_shard(path, radius=10_000., nf=48):
    """Two native pulses per sector on a 10 km, 45-degree ring; unit responses (never read here)."""
    sector = np.repeat(np.arange(1, 361, dtype=np.int16), 2)
    pulse = np.tile(np.arange(2, dtype=np.int32), 360)
    angle = np.radians(sector + .05 * pulse)
    ground = radius / math.sqrt(2)
    xyz = np.stack((ground * np.cos(angle), ground * np.sin(angle), np.full(len(sector), ground)), axis=-1)
    freq = np.linspace(9.288e9, 9.910e9, nf)
    meta = dict(schema=SOURCE_SCHEMA, pass_id=1, polarization='hh', shard_id='pass1_hh', corrections_applied=False,
                native_frequency_preserved=True, all_rows_role='train', source_file_count=360,
                all_available_sectors_used=True, evaluation_holdout=False,
                phase_reference=dict(frequency_unit='Hz', position_unit='m', range_unit='m', reference_range_field='r0',
                                     frequency_values='native_stored_exact',
                                     geometry_contract='paired_monostatic_tx_equals_rx_same_observation'),
                autofocus=dict(applied=False, official_available=True, mode='source_af_unapplied',
                               source_shard_id='pass1_hh'))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, metadata_json=json.dumps(meta), frequencies_hz=freq, x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
             r0=np.linalg.norm(xyz, axis=1), sector_id=sector, pulse_index=pulse,
             pass_id=np.full(len(sector), 1, dtype=np.int16), polarization=np.full(len(sector), 'hh'),
             role=np.full(len(sector), 'train'), r_correct_raw=np.zeros(len(sector)),
             ph_correct_raw=np.zeros(len(sector)), response=np.ones((len(sector), nf), dtype=np.complex64))


@pytest.fixture(scope='module')
def far_look(tmp_path_factory):
    from rift.radarsplat_gotcha import GOTCHAPowerCache
    root = tmp_path_factory.mktemp('far')
    write_far_shard(root / 'New_Transfer/shards/pass1_hh.npz')
    region = Region('present', 'synthetic', (0., 0., 0.), IDENTITY, 5.0, 'presentation test')
    dataset = GOTCHADataset(root, passes=[1], region=region)
    cache = GOTCHAPowerCache(dataset, 'hh', root / 'cache', CONFIG)
    index = cache.train_indices[0]
    row = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cache.calibration[index].items()}
    return dict(dataset=dataset, row=row, view=cache.view_keys[index], renderer=rs.load_renderer(50 / 5.0))


def one_gaussian(position_m, scale=.25, reflectance=.7, opacity=.3, units=10.):
    quat = torch.nn.functional.normalize(torch.tensor([[.9, .2, -.3, .1]]), dim=-1)
    return dict(means=torch.tensor([position_m], dtype=torch.float32) * units, quats=quat,
                scales=torch.log(torch.as_tensor(scale, dtype=torch.float32)).expand(1, 3).clone(),
                opacities=torch.logit(torch.tensor([opacity])), noise_probs=torch.full((1,), -6.),
                sh0=torch.full((1, 1, 3), (reflectance - .5) / SH_C0), shN=torch.zeros(1, 35, 3))


def rendered_power(look, splats, peak, mapping):
    from rift.radarsplat_b7873200 import RadarSplatGrid
    grid = RadarSplatGrid(**look['row']['renderer_grid'])
    pose = torch.as_tensor(np.asarray(look['row']['sensor_to_world']), dtype=torch.float32)
    zeros = torch.zeros(grid.output_azimuth_bins, grid.num_range_bins)
    with torch.no_grad():
        image = look['renderer'](splats, pose, grid, 0, zeros)[0].double()
        empty = look['renderer'](rs._empty_splats(splats), pose, grid, 0, zeros)[0].double()
    return float((rs.native_power(image, peak, mapping) - rs.native_power(empty, peak, mapping)).sum())


@pytest.mark.parametrize('mapping', [rs.LINEAR, rs.LOG])
def test_rendered_gaussian_is_a_point_target_of_the_converted_rcs(far_look, mapping):
    """One Gaussian at 10 km through the method's own renderer and its own target formation.

    sigma * e1 must equal the rendered native image power above the empty scene,
    and a point of that sigma must produce K^2 sigma / (2R)^4 at its own sample.
    """
    look, peak = far_look, 2.0e-7
    splats = one_gaussian([.5, -.3, .2])
    metadata = power.look_metadata(look['dataset'], look['view'], 'hh')
    e1 = rs.point_energy(look['dataset'], metadata, look['row'], 'hh', CONFIG)
    shares, sigma, info = rs.look_rcs(splats, look['renderer'], look['row'], peak, mapping, 0, point_energy=e1)
    rendered = rendered_power(look, splats, peak, mapping)
    assert float(shares.sum()) == pytest.approx(info['rendered_intensity'], rel=1e-4)
    assert rendered > 0 and float(sigma[0]) * e1 == pytest.approx(rendered, rel=1e-4)
    # The same sigma as a point target through sector_power, one elevation sample through the point.
    from rift.radarsplat_gotcha import sector_power
    single = rs.calibration_arrays(look['row'])
    single['elevation_rad'] = np.zeros(1)
    stand_in = rs._PointTargetSector(look['dataset'].region, metadata)
    point = sector_power(stand_in, *look['view'], 'hh', single, CONFIG, device='cpu')
    amplitudes = [power.point_target_observation(n, l, metadata['frequencies'])[0].response[0].real
                  for n, l in zip(metadata['native'], metadata['local'])]
    expected = float(np.mean(amplitudes)) ** 2 * float(sigma[0])
    distance = float(np.linalg.norm(metadata['local'].mean(0)))
    assert float(point.max()) * float(sigma[0]) == pytest.approx(expected, rel=1e-5)
    assert expected == pytest.approx(power.point_target_power(float(sigma[0]), distance), rel=1e-4)


@pytest.mark.parametrize('mapping', [rs.LINEAR, rs.LOG])
def test_overlapping_anisotropic_gaussians_share_the_rendered_power_exactly(far_look, mapping):
    """Every rendered watt above the empty scene is attributed, whatever the mapping's nonlinearity."""
    look, peak = far_look, 1.0
    a = one_gaussian([-.8, .4, -.1], scale=[.05, .6, .3], reflectance=.9, opacity=.6)
    b = one_gaussian([-.75, .45, -.1], scale=[.3, .1, .2], reflectance=.6, opacity=.4)
    both = {k: torch.cat([a[k], b[k]]) for k in a}
    shares, sigma, info = rs.look_rcs(both, look['renderer'], look['row'], peak, mapping, 0, point_energy=1.0)
    assert info['attributed_power'] == pytest.approx(info['rendered_power'], rel=1e-4)
    assert info['attributed_intensity'] == pytest.approx(info['rendered_intensity'], rel=1e-4)
    assert info['rendered_power'] == pytest.approx(rendered_power(look, both, peak, mapping), rel=1e-9)
    alone = rs.look_rcs(a, look['renderer'], look['row'], peak, mapping, 0, point_energy=1.0)[1]
    if mapping == rs.LINEAR:   # additive: a Gaussian's share does not depend on its neighbours
        assert float(sigma[0]) == pytest.approx(float(alone[0]), rel=1e-4)
    assert (sigma > 0).all()


@pytest.mark.parametrize('mapping', [rs.LINEAR, rs.LOG])
def test_saturated_pixels_are_split_by_pre_clamp_contributions(far_look, mapping):
    """Forty bright Gaussians on one spot saturate the renderer's clamps; nothing is lost."""
    look = far_look
    stack = [one_gaussian([.1 * (i % 5), -.05 * (i // 5), 0.], reflectance=.95, opacity=.9) for i in range(40)]
    splats = {k: torch.cat([g[k] for g in stack]) for k in stack[0]}
    shares, sigma, info = rs.look_rcs(splats, look['renderer'], look['row'], 1.0, mapping, 0, point_energy=1.0)
    assert info['kappa'] > 1
    assert info['attributed_power'] == pytest.approx(info['rendered_power'], rel=1e-4)
    assert info['attributed_intensity'] == pytest.approx(info['rendered_intensity'], rel=1e-4)
    assert (sigma > 0).all() and (shares > 0).all()


def test_intensity_mapping_inverse():
    image = torch.tensor([0., 1e-3, .5, 1.], dtype=torch.float64)
    np.testing.assert_allclose(rs.native_power(image, 4., rs.LINEAR).numpy(), [0, 4e-3, 2., 4.])
    log = rs.native_power(image, 4., rs.LOG).numpy()
    assert log[0] == 0 and log[-1] == pytest.approx(4.) and log[2] == pytest.approx(4. * 1e-3)
    assert rs.intensity_mapping({}) == rs.LINEAR and rs.intensity_mapping({'intensity_mapping': rs.LOG}) == rs.LOG
    with pytest.raises(ValueError):
        rs.intensity_mapping({'intensity_mapping': 'other'})


def test_engine_checkpoint_presents_both_directories_without_reading_responses(tmp_path, monkeypatch):
    """A checkpoint in the engine's save format: identity, acquisition record and the released initializer."""
    from rift import radarsplat_release as release
    from rift import radarsplat_release_training as engine
    from rift.radarsplat_gotcha import GOTCHAPowerCache
    # The 10 km fixture: the 20 m RadarSplat fixture's 2 cm aperture makes a 51-degree
    # azimuth beam and a ~17000-row raster, far too slow for a CPU test.
    write_far_shard(tmp_path / 'data/New_Transfer/shards/pass1_hh.npz')
    region = Region('present', 'synthetic', (0., 0., 0.), IDENTITY, 5.0, 'presentation test')
    dataset = GOTCHADataset(tmp_path / 'data', passes=[1], region=region)
    cache = GOTCHAPowerCache(dataset, 'hh', tmp_path / 'cache', CONFIG)
    assert cache.prepare(device='cpu')
    identity = engine.identity_for_cache(cache, 'budget48', release.LOG_INTENSITY)
    splats, _ = release.create_scene(scene_scale=identity['adapter']['initialization_scene_scale'],
                                     scene_center=np.zeros(3), device='cpu', num_points=32)
    path = tmp_path / 'checkpoint_latest.pt'
    torch.save(dict(schema=release.SCHEMA, identity=identity, step=3, splats=splats.state_dict(), validation=None,
                    acquisition_record=cache.acquisition_record), path)
    monkeypatch.setattr(NativeShardReader, 'read', lambda *_: pytest.fail('presentation read a response'))
    shards = tmp_path / 'data/New_Transfer/shards'
    result = rs.radarsplat_presentation(path, shard_root=shards, looks=3, grid=8)
    assert result.source['intensity_mapping'] == rs.LOG and result.source['looks'] == 3
    lo, hi = result.source['attributed_over_rendered']
    assert lo == pytest.approx(1, rel=1e-3) and hi == pytest.approx(1, rel=1e-3)
    assert result.rcs_m2.shape == result.relative.shape == (8, 8, 8) and result.rcs_m2.sum() > 0
    # Relative: the rendered image intensity shares, binned by Gaussian mean (cube-inside Gaussians only).
    shares = [d['attributed_intensity'] for d in result.source['looks_detail']]
    assert result.relative.max() > 0 and result.relative.sum() <= np.mean(shares) * (1 + 1e-9)
    assert result.source['support_proxy']['occupied_cells'] >= 0
    from scripts_pvc.gotcha_presentation import main
    main(['--output-root', str(tmp_path / 'present'), '--run', f'radarsplat={path}', '--shard-root', str(shards)])
    for directory in (presentation.RELATIVE_DIR, presentation.PHYSICAL_DIR):
        assert (tmp_path / 'present' / directory / 'radarsplat_hh.npz').is_file()
