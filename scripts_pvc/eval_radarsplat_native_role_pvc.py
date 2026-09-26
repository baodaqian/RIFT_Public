#!/usr/bin/env python3
"""RadarSplat's native elevation-integrated 2-D power RelMSE on any RIFT-dataset role, reserved TEST included (PVC).

For the manuscript's Table 6 "B787 native reduced-observable diagnostics". The held-out harness
(``scripts_pvc/eval_rift_dataset_heldout_pvc.py``) scores RadarSplat's own metric only on views that have a
cached native target, and the release cache holds TRAIN and VAL only, so its TEST run scored 0 native views. This
readout runs the same harness unchanged, with a subclass of its ``RadarSplatAdapter`` that first materialises each
selected view's native target into this readout's own directory, exactly as
``scripts/prepare_radarsplat_b7873200_targets.py`` builds the cache (chirp-mean response, the cache recipe's polar
grid and matched filter, ``_target_from_response``), and then lets the adapter's unchanged ``predict`` read it. The
release cache itself is never written. On VAL each rebuilt target is also compared with the cached one, and the pooled
metric must reproduce the cached-target readout (0.7317 for B787); on TEST the harness's common-domain score must
reproduce the final evaluation's.

    python scripts_pvc/eval_radarsplat_native_role_pvc.py --target-dir OUT/targets_test -- \\
        <eval_rift_dataset_heldout_pvc.py arguments: --method radarsplat ... --role test --allow-reserved-test>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import scripts_pvc.eval_rift_dataset_heldout_pvc as harness  # noqa: E402

VIEWS = 'radarsplat_b7873200_views'


def adapter_with_targets(target_dir):
    class RadarSplatAnyRoleAdapter(harness.RadarSplatAdapter):
        def __init__(self, checkpoint_path, contract, arrays, device, args):
            super().__init__(checkpoint_path, contract, arrays, device, args)
            from rift.power_baseline_dataset import frequency_grid_hz
            from rift.radarsplat_b7873200_protocol import RECIPE_FILENAME
            self.release_cache_root = self.cache_root
            recipe = json.loads((self.release_cache_root/RECIPE_FILENAME).read_text())
            self.matched_filter = SimpleNamespace(**recipe['target_spec']['matched_filter'])
            self.target_frequencies = torch.as_tensor(frequency_grid_hz(arrays.metadata), dtype=torch.float64)
            self.cache_root = Path(target_dir)             # predict() reads <cache_root>/radarsplat_b7873200_views
            (self.cache_root/VIEWS).mkdir(parents=True, exist_ok=True)
            self.rebuilt = dict(views=0, compared=0, max_power_rel_difference=0.0)
            self.identity = dict(self.identity, native_targets='rebuilt per view by this readout', target_dir=str(target_dir),
                                 release_cache_root=str(self.release_cache_root))

        def _materialise(self, view_index):
            from rift.power_baseline_dataset import build_radarsplat_target_grid
            from scripts.prepare_radarsplat_b7873200_targets import _target_from_response
            path = self.cache_root/VIEWS/f'view_{int(view_index):06d}.npz'
            if path.exists():
                return
            g = self.grid_spec
            tx = torch.as_tensor(self.arrays.tx_pos[view_index], dtype=torch.float32)
            rx = torch.as_tensor(self.arrays.rx_pos[view_index], dtype=torch.float32)
            grid = build_radarsplat_target_grid(
                self.arrays.viewpoint_positions[view_index], tx, rx, scene_center=g['scene_center_m'],
                scene_extent_m=float(g['scene_extent_m']), n_azimuth=int(g['n_azimuth']),
                n_elevation=int(g['n_elevation']), n_range=int(g['n_range']),
                output_azimuth_resolution_deg=float(g['output_azimuth_resolution_deg']),
                elevation_sampling_resolution_deg=float(g['elevation_sampling_resolution_deg']), device='cpu',
                dtype=torch.float32)
            response = np.asarray(self.arrays.response_view(view_index)).mean(axis=2)   # [tx, rx, freq], chirp mean
            power = _target_from_response(response, self.target_frequencies, tx, rx, grid, self.matched_filter)
            power = power.detach().to(torch.float32).cpu().numpy()
            release = self.release_cache_root/VIEWS/path.name
            if release.exists():
                with np.load(release) as saved:
                    cached = saved['radarsplat_mf_power'].astype(np.float64)
                difference = float(np.abs(power - cached).max() / max(np.abs(cached).max(), 1e-30))
                self.rebuilt['compared'] += 1
                self.rebuilt['max_power_rel_difference'] = max(self.rebuilt['max_power_rel_difference'], difference)
            np.savez(path.with_suffix('.tmp.npz'), radarsplat_mf_power=power,
                     **{key: getattr(grid, key).detach().to(torch.float32).cpu().numpy()
                        for key in ('sensor_to_world', 'range_m', 'azimuth_rad', 'elevation_rad')})
            path.with_suffix('.tmp.npz').replace(path)
            self.rebuilt['views'] += 1

        def predict(self, view_index, position, rx_pos, tx_pos, viewpoint):
            self._materialise(view_index)
            return super().predict(view_index, position, rx_pos, tx_pos, viewpoint)

        def extra_summary(self):
            summary = super().extra_summary()
            summary['method_own_metric'] = dict(summary['method_own_metric'], rebuilt_targets=self.rebuilt,
                definition='native_clipped_power_relative_mse on targets rebuilt by eval_radarsplat_native_role_pvc.py')
            return summary
    return RadarSplatAnyRoleAdapter


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index('--') if '--' in argv else len(argv)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--target-dir', type=Path, required=True, help='this readout\'s own native-target directory')
    args = p.parse_args(argv[:split])
    harness_argv = argv[split + 1:]
    if '--method' not in harness_argv or harness_argv[harness_argv.index('--method') + 1] != 'radarsplat':
        raise SystemExit('pass --method radarsplat to the harness')
    harness.ADAPTERS['radarsplat'] = adapter_with_targets(args.target_dir)
    harness.main(harness_argv)


if __name__ == '__main__':
    main()
