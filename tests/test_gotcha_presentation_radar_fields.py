"""GOTCHA presentation of Radar Fields: relative alpha*rho and K-calibrated RCS."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from rift.gotcha_dataset import GOTCHADataset, NativeShardReader, Region
from rift.radar_fields import radar_fields_intensity
from rift.radar_fields_gotcha import matched_range_power
from rift.radar_fields_native import prepare_bistatic_bins, render_bistatic_batch
from rift_pvc import gotcha_presentation as presentation
from rift_pvc import gotcha_presentation_power as power
from rift_pvc import gotcha_presentation_radar_fields as rf
from tests.test_gotcha_dataset import tiny_region, write_shard

REFERENCE = Path(__file__).resolve().parents[1] / 'external' / 'RadarFields_reference'
needs_reference = pytest.mark.skipif(not REFERENCE.is_dir(), reason='pinned Radar Fields checkout not present')
IDENTITY = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.setenv('RIFT_ACCELERATOR', 'cpu')
    monkeypatch.setenv('RIFT_PVC_ALLOW_BACKEND', 'cpu')
    monkeypatch.setenv('RIFT_PVC_TCNN_SHIM', '1')


class OneCell(torch.nn.Module):
    """alpha*rho = value inside one axis-aligned cell, zero elsewhere (a stand-in field)."""
    def __init__(self, lower, upper, value):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.lower, self.upper, self.value = torch.as_tensor(lower), torch.as_tensor(upper), value

    def query_chunked(self, xyz, view, mask_progress=None, chunk_size=None):
        inside = ((xyz >= self.lower.to(xyz)) & (xyz < self.upper.to(xyz))).all(-1)
        rcs = inside.to(xyz.dtype) * self.value
        return dict(alpha=inside.to(xyz.dtype), reflectance=rcs, rcs=rcs)


def camry_like_look(range_m=10_000.):
    """A GOTCHA-scale look: 10 km slant range at 45 deg, 424 native frequencies, 5 m region."""
    region = Region('present', 'synthetic', (0., 0., 0.), IDENTITY, 5.0, 'presentation test')
    direction = np.array([1., .3, 1.]) / np.linalg.norm([1., .3, 1.])
    position = direction * range_m + np.array([[0., 0., 0.], [0., .2, 0.], [0., -.2, 0.]])
    return dict(view=(1, 1), native=position, local=position, region=region,
                frequencies=np.linspace(9.288e9, 9.910e9, 424))


def recipe(profile='audited-v2'):
    return dict(controls=dict(profile=profile, range_guard_cells=2, query_chunk=32768, dynamic_range_db=60.),
                extent_m=5.0, intensity_offset=1., intensity_scaler=1.)


@needs_reference
@pytest.mark.parametrize('profile', ['audited-v2', 'source-adapted-v3'])
def test_single_cell_render_is_a_point_target_of_the_converted_rcs(profile):
    """Render one occupied G48 cell with the method's own GOTCHA renderer at 10 km.

    The converter's sigma must be the RCS whose point target, formed by the
    method's own matched-range target, carries the rendered native power, and
    whose matched-range power at its own bin is K^2 sigma / (2R)^4.
    """
    # One cell of a 4^3 presentation grid (2.5 m): at 10 km a G48 cell would catch
    # a ray only once per ~5000, so the coarser grid keeps the test fast.
    look, peak, rays = camry_like_look(), 3.5e-4, 4096
    model = OneCell((0., 0., 0.), (2.5, 2.5, 2.5), 50.)
    result = rf.look_power(model, recipe(profile), peak, look, rays=rays, seed=3)
    cells, lost = power.bin_points(result['xyz'], result['power'] / result['e1'], 5.0, 4)
    sigma = cells.sum()
    assert lost == 0 and sigma > 0 and np.count_nonzero(cells) == 1 and cells[2, 2, 2] == sigma
    # Independent render with the same rays: native power above the floor.
    observation, distance = power.point_target_observation(look['native'].mean(0), look['local'].mean(0),
                                                           look['frequencies'])
    from rift.radar_fields_gotcha import range_geometry
    antenna, ranges = range_geometry(observation, look['region'], 2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        geometry = prepare_bistatic_bins(antenna[None], antenna[None], ranges, extent=5.0, ray_samples=rays,
                                         source_sampling=profile == 'source-adapted-v3')
    X = render_bistatic_batch(model, [geometry], query_chunk=32768)[0]['rcs'][0].double()
    I = radar_fields_intensity(X, ranges, offset=1., scaler=1.)
    rendered = torch.where(I > rf.target_floor(recipe(profile)), peak * 10 ** (6 * (I - 1)), torch.zeros_like(I))
    assert float(rendered.sum()) > 0
    point = power.point_target_observation(look['native'].mean(0), look['local'].mean(0), look['frequencies'],
                                           rcs_m2=sigma)[0]
    matched = matched_range_power(point, ranges)
    assert float(matched.sum()) == pytest.approx(float(rendered.sum()), rel=1e-9)
    assert float(matched.max()) == pytest.approx(power.point_target_power(sigma, distance), rel=1e-9)


def test_dB_floor_and_inverse_mapping():
    t = torch.tensor([0., .1, .1525, .2, 1.], dtype=torch.float64)
    source = power.invert_db(t, 2.0, 60., rf.SOURCE_TARGET_FLOOR)
    audited = power.invert_db(t, 2.0, 60., 0.)
    assert source[:3].tolist() == [0., 0., 0.] and audited[0] == 0 and audited[1] > 0
    assert float(audited[-1]) == pytest.approx(2.0) and float(source[3]) == pytest.approx(2.0 * 10 ** (6 * (.2 - 1)))


@needs_reference
def test_real_checkpoint_presents_both_directories_without_reading_responses(tmp_path, monkeypatch):
    from rift_pvc import radar_fields_gotcha as pvc_gotcha
    write_shard(tmp_path / 'New_Transfer/shards/pass1_hh.npz', 1, 'hh', nf=33)
    dataset = GOTCHADataset(tmp_path, passes=(1,), polarizations=('hh',), region=tiny_region())
    config = dict(profile='audited-v2', model_backend=pvc_gotcha.DEFAULTS['model_backend'], steps=2, view_batch=1,
                  seed=7, ray_samples=8, eval_every=2, checkpoint_every=1, hidden_dim=16, feature_dim=4,
                  hash_levels=2, hash_final_resolution=8)
    if 'pose_refinement' in pvc_gotcha.DEFAULTS:
        config['pose_refinement'] = 'disabled'
    assert pvc_gotcha.run_gotcha(dataset=dataset, output_dir=tmp_path / 'run', config=config, device='cpu',
                                 resume=None)['status'] == 'complete'
    checkpoint = tmp_path / 'run/checkpoint_final.pt'
    monkeypatch.setattr(NativeShardReader, 'read', lambda *_: pytest.fail('presentation read a response'))
    result = rf.radar_fields_presentation(checkpoint, shard_root=tmp_path / 'New_Transfer/shards', looks=6,
                                          rays=512, grid=8)
    assert result.relative.shape == result.rcs_m2.shape == (8, 8, 8)
    assert result.source['looks'] == 6 and result.source['train_looks'] == 250
    # Every rendered watt above the floor lands in exactly one cell.
    assert result.source['attributed_rcs_m2'] == pytest.approx(result.source['rendered_rcs_m2'], rel=1e-9)
    assert result.rcs_m2.sum() == pytest.approx(result.source['attributed_rcs_m2'], rel=1e-9)
    # Relative is the field's own alpha*rho at the cell centres, mean over the looks' views.
    saved = torch.load(checkpoint, weights_only=False)
    model, _ = rf.load_head(saved, 'hh')
    looks = [power.look_metadata(power.dataset_from_contract(saved['dataset_contract'], tmp_path / 'New_Transfer/shards'),
                                 v, 'hh') for v in power.select_looks(dataset.viewpoints('train'), 6)]
    centres = power.cell_centres(.03, 8)
    with torch.no_grad():
        expected = sum(model.query_chunked(centres.float(), torch.nn.functional.normalize(
            centres - torch.as_tensor(m['local'].mean(0)), dim=-1).float(), mask_progress=1.)['rcs'].double().clamp_min(0)
            for m in looks) / 6
    np.testing.assert_allclose(result.relative.reshape(-1), expected.numpy(), rtol=1e-6)
    # The CLI writes the same method into both directories.
    from scripts_pvc.gotcha_presentation import main
    main(['--output-root', str(tmp_path / 'present'), '--run', f'radar_fields={checkpoint}',
          '--shard-root', str(tmp_path / 'New_Transfer/shards')])
    for directory in (presentation.RELATIVE_DIR, presentation.PHYSICAL_DIR):
        assert (tmp_path / 'present' / directory / 'radar_fields_hh.npz').is_file()
    manifest = json.loads((tmp_path / 'present/manifest.json').read_text())
    assert manifest['methods']['radar_fields_hh']['model_backend'] == pvc_gotcha.DEFAULTS['model_backend']
