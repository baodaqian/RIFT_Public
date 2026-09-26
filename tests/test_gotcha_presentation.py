"""GOTCHA presentation: relative intensity and K-calibrated RCS directories."""
from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from rift.calibration import GlobalComplexGain
from rift.sparse_scene import AdaptivePointSHScene
from rift_pvc import gotcha_presentation as presentation
from rift_pvc.gotcha_training import native_forward
from scripts.eval_scene_geometry import deposit_points as collection_deposit
from scripts.render_b787_vs_stl import _point_sh_energy_field

REGION = dict(name='fixture', target_id='synthetic', translation_m=[0., 0., 0.],
              rotation_local_to_native=[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], half_extent_m=5.0,
              placement_provenance='synthetic unit test')


def scene(seed=0, granularity=4, degree=3):
    torch.manual_seed(seed)
    field = AdaptivePointSHScene.from_regular_grid(granularity, 5.0, 'cpu', max_degree=degree, init_degree=degree,
                                                   init_scale=1e-3, enforce_support_bounds=True)
    with torch.no_grad():
        field.w_re.normal_()
        field.w_im.normal_()
        field.delta_raw.normal_()
        field.order.copy_(torch.randint(0, degree + 1, field.order.shape))
        field.active_mask[::5] = False
    return field


def checkpoint(field, gain=None, range_model='sum2'):
    gain = gain or GlobalComplexGain()
    state = {f'hh.field.{k}': v for k, v in field.state_dict().items()}
    state.update({f'hh.gain.{k}': v for k, v in gain.state_dict().items()})
    return dict(model_state_dict=state, recipe=dict(method='rift', range_model=range_model),
                dataset_contract=dict(region=REGION, passes=[1],
                                      split=dict(sector_ids=dict(train=[3, 7], validation=[], test=[]))),
                epoch=1, best_val=1.0, updates=2)


class PresentationTests(unittest.TestCase):
    def test_deposit_matches_collection_readout_and_conserves(self):
        positions = torch.rand(500, 3, dtype=torch.float64) * 10 - 5
        values = torch.rand(500, dtype=torch.float64)
        ours = presentation.deposit_points(positions, values, 5.0, 12)
        np.testing.assert_allclose(ours, collection_deposit(positions, values, 5.0, 12).numpy(), rtol=0, atol=0)
        self.assertAlmostEqual(ours.sum(), float(values.sum()), places=10)

    def test_relative_is_the_existing_point_sh_readout(self):
        field = scene()
        positions, coefficients, _ = presentation._point_sh_state(checkpoint(field), 'hh')
        ours = presentation.deposit_points(positions, coefficients.abs().square().sum(-1), 5.0, 8)
        theirs, _ = _point_sh_energy_field(field.state_dict(), 5.0, 8)
        np.testing.assert_allclose(ours, theirs, rtol=1e-12, atol=0)

    def test_aperture_gram_is_the_mean_directional_power(self):
        torch.manual_seed(1)
        d = torch.nn.functional.normalize(torch.randn(40, 3, dtype=torch.float64), dim=1)
        c = torch.complex(torch.randn(16, dtype=torch.float64), torch.randn(16, dtype=torch.float64))
        gram = presentation.aperture_gram(d, 3).to(c.dtype)
        theta, phi = torch.acos(d[:, 2]), torch.atan2(d[:, 1], d[:, 0])
        from rift.spherical_harmonics import real_sh_basis
        direct = torch.stack([(c * real_sh_basis(t, p, 3)).sum().abs().square() for t, p in zip(theta, phi)]).mean()
        self.assertAlmostEqual(float((c @ gram @ c.conj()).real), float(direct), places=10)

    @torch.no_grad()
    def test_rcs_reproduces_the_data_through_K(self):
        """Render each point with the trainer's own forward; data (2R)^2 / K must equal sqrt(RCS)."""
        field = scene(granularity=2)
        gain = GlobalComplexGain()
        with torch.no_grad():
            gain.log_mag.fill_(math.log(3.0e6))
            gain.phase.fill_(0.4)
        torch.manual_seed(2)
        directions = torch.nn.functional.normalize(torch.randn(6, 3, dtype=torch.float64), dim=1)
        R, f = 10_000.0, torch.tensor([9.5e9], dtype=torch.float64)
        for range_model in ('sum2', 'unit'):
            positions, coefficients, degree = presentation._point_sh_state(checkpoint(field, gain, range_model), 'hh')
            predicted = presentation.rift_point_rcs(coefficients, degree, abs(gain.gain_value()), range_model,
                                                    directions.numpy(), np.full(6, R))
            expected = torch.zeros(len(positions), dtype=torch.float64)
            for u in directions:
                theta, phi = torch.acos(u[2]), torch.atan2(u[1], u[0])
                points, weights = field.active_scatterers(theta.float().reshape(1, 1), phi.float().reshape(1, 1))
                for i in range(len(points)):
                    antenna = points[i].double() + R * u      # exactly R from this scatterer
                    y = gain(native_forward(points[i:i + 1].double(), weights[i:i + 1], antenna, f, R,
                                            range_model=range_model))
                    expected[i] += (y.abs()[0] * (2 * R) ** 2 / presentation.K) ** 2 / len(directions)
            np.testing.assert_allclose(predicted.numpy(), expected.numpy(), rtol=2e-6)

    def test_look_geometry_reads_metadata_only(self):
        with tempfile.TemporaryDirectory() as temp:
            rows = np.array([3, 3, 7, 9])
            xyz = np.array([[1000., 0, 1000], [1000., 10, 1000], [0, 2000., 0], [5., 5, 5]])
            # No response member: any response access would fail.
            np.savez(Path(temp) / 'pass1_hh.npz', x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], sector_id=rows)
            geometry = presentation.train_look_geometry(checkpoint(scene())['dataset_contract'], temp)
            self.assertEqual(geometry['sectors'], [(1, 3), (1, 7)])
            np.testing.assert_allclose(geometry['directions'][1], [0, 1, 0], atol=1e-12)
            self.assertAlmostEqual(geometry['ranges_m'][1], 2000.0)

    def test_two_directories_share_file_names_and_one_dbsm_scale(self):
        with tempfile.TemporaryDirectory() as temp:
            for method, level in (('rift', 1.0), ('spinr', 30.0)):
                volume = np.full((4, 4, 4), 1e-6)
                volume[1, 2, 3] = level
                presentation.write_method(temp, presentation.MethodPresentation(
                    method, 'hh', volume * 7, 'native', volume, 'RCS', dict(half_extent_m=5.0)), 5.0)
            manifest = presentation.render(temp)
            names = lambda d: sorted(p.name for p in (Path(temp) / d).iterdir())
            self.assertEqual(names(presentation.RELATIVE_DIR), names(presentation.PHYSICAL_DIR))
            self.assertEqual(sorted(manifest['methods']), ['rift_hh', 'spinr_hh'])
            physical = json.loads((Path(temp) / presentation.PHYSICAL_DIR / 'summary.json').read_text())
            self.assertEqual(physical['colour_scale_dbsm'], [-25.0, 15])
            self.assertAlmostEqual(physical['spinr_hh']['peak_cell_dbsm'], 10 * math.log10(30.0))
            with np.load(Path(temp) / presentation.RELATIVE_DIR / 'rift_hh.npz') as data:
                self.assertEqual(float(data['intensity'].max()), 1.0)


if __name__ == '__main__':
    unittest.main()
