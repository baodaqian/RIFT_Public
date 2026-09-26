"""SE on GOTCHA Camry box-v2 (BASELINES_CAMRY_BOX.md): Stage-1 energy field and Stage-2 surface readouts."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location('se_readout', 'scripts_pvc/se_gotcha_readout_pvc.py')
se_readout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(se_readout)

MESH = Path('data/meshes/camry_xv20_data_frame_box_v2')


def test_field_conserves_the_subaperture_energy_and_keeps_the_region(tmp_path):
    region = dict(name='camry_box_v2', half_extent_m=3.0, translation_m=[1.0, 2.0, 0.0])
    fields = torch.randn(3, 5 ** 3, dtype=torch.complex128)
    ck = dict(acquisition=dict(contract=dict(region=region)), recipe=dict(granularity=5), fields=fields,
              phase='stage1')
    torch.save(ck, tmp_path / 'ck.pt')
    se_readout.field(tmp_path / 'ck.pt', tmp_path / 'field.npz')
    z = np.load(tmp_path / 'field.npz')
    assert z['se_energy'].shape == (se_readout.GRID,) * 3
    assert z['se_energy'].sum() == pytest.approx(float(fields.abs().square().sum()), rel=1e-12)
    meta = json.loads(str(z['meta']))
    assert meta['region'] == region and meta['subapertures'] == 3 and meta['granularity'] == 5


@pytest.mark.skipif(not (MESH / 'camry_xv20_region_local.stl').exists(), reason='registered mesh not present')
def test_surface_scores_a_shifted_copy_of_the_truth_by_its_offset(tmp_path):
    from scripts.render_b787_vs_stl import load_stl_vertices
    tris = load_stl_vertices(MESH / 'camry_xv20_region_local.stl').reshape(-1, 3, 3)[::50]
    vertices = tris.reshape(-1, 3) + np.array([0.0, 0.0, 0.5])
    faces = np.arange(len(vertices)).reshape(-1, 3)
    np.savez(tmp_path / 'surface.npz', vertices=vertices, faces=faces)
    se_readout.surface(tmp_path / 'surface.npz', tmp_path / 'near.json', MESH, 4000, 0.125)
    se_readout.surface(tmp_path / 'surface.npz', tmp_path / 'far.json', MESH, 4000, 0.6)
    near, far = (json.loads((tmp_path / f'{n}.json').read_text()) for n in ('near', 'far'))
    assert near['prf']['f1'] < far['prf']['f1'] and far['prf']['precision'] > 0.9


def test_role_numbers_follow_the_relmse_identity():
    e, rho = 0.55, 0.8
    sums = dict(target=2.0, prediction=2.0 * e, cross=rho * 2.0 * np.sqrt(e), views=3, samples=9)
    sums['error'] = sums['target'] + sums['prediction'] - 2 * sums['cross']
    out = se_readout.role_numbers(sums)
    assert out['energy_ratio'] == pytest.approx(e) and out['correlation'] == pytest.approx(rho)
    assert out['full_native_rel_mse'] == pytest.approx(1 + e - 2 * rho * np.sqrt(e))
