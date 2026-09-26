#!/usr/bin/env python3
"""Export adaptive-RIFT GOTCHA checkpoints as compact point clouds for the 3D viewer (tuning campaign A64, CPU).

For each ``LABEL=CHECKPOINT`` (or ``LABEL=field:NPZ:KEY`` for a voxel image such as the measured-data A^H y),
the active points are taken as the CIC readout deposits them (positions with support clamping and
unlocked-band sum |w|^2, ``gotcha_energy_on_car_pvc.point_energy``). All of them are deposited by the geometry
scorer's conservative CIC (``scripts.eval_scene_geometry.deposit_points``) on a finer ``--voxel-grid`` lattice
(96: 6.25 cm over the 6 m cube, against the scorer's 12.5 cm), so a 1M-point scene is shown by its whole energy
and not by its strongest points only. The ``--top`` most energetic voxels are kept with the share of the energy
they hold. Also recorded: the epoch's TRAIN and VAL fit from the checkpoint's own
history, the on-car energy partition (A58), and the best CIC F1 over the threshold sweep found for that checkpoint
path in ``--geometry-dir``. The registered Camry surface is sampled area-weighted (``--mesh-samples``) in the same
region frame, for a viewport placed beside the reconstruction, never overlaid.

    python scripts_pvc/export_gotcha_rift_pointcloud_pvc.py --run T5b_ep10=PATH.pt ... --out-dir DIR
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_scene_geometry import deposit_points  # noqa: E402
from scripts.render_b787_vs_stl import load_stl_vertices  # noqa: E402
from scripts_pvc.gotcha_energy_on_car_pvc import car_geometry, partition, point_energy  # noqa: E402


def best_f1(geometry_dir, checkpoint):
    best = None
    for path in glob.glob(str(Path(geometry_dir) / '*.json')):
        try:
            rows = json.loads(Path(path).read_text()).get('rows') or []
        except (json.JSONDecodeError, OSError, AttributeError):
            continue
        for row in rows:
            if row.get('checkpoint') == checkpoint and 'prf' in row:
                f1 = row['prf']['f1']
                if best is None or f1 > best['f1']:
                    best = dict(f1=f1, threshold=row['threshold'], chamfer_mm=row['surface']['l2_mm'],
                                source=Path(path).name)
    return best


def fit_record(checkpoint):
    history = checkpoint.get('history') or []
    entry = next((h for h in reversed(history) if 'train' in h), None)
    out = dict(epoch=checkpoint.get('epoch'), points_active=None)
    if entry is not None:
        t, v = entry['train'], entry.get('validation_fit') or {}
        out.update(fit_epoch=entry['epoch'], train_rel_mse=t['full_native_rel_mse'], train_rho=t['correlation'],
                   val_rel_mse=v.get('full_native_rel_mse'), val_rho=v.get('correlation'))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', action='append', required=True, help='LABEL=CHECKPOINT or LABEL=field:NPZ:KEY')
    p.add_argument('--mesh-dir', type=Path, default=ROOT / 'data/meshes/camry_xv20_data_frame_box_v2')
    p.add_argument('--geometry-dir', type=Path)
    p.add_argument('--top', type=int, default=40000)
    p.add_argument('--voxel-grid', type=int, default=96)
    p.add_argument('--mesh-samples', type=int, default=40000)
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.mesh_dir / 'manifest.json').read_text())
    extent = float(manifest['region']['half_extent_m'])
    car = car_geometry(args.mesh_dir)
    triangles = load_stl_vertices(args.mesh_dir / 'camry_xv20_region_local.stl').reshape(-1, 3, 3)
    area = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(triangles), size=args.mesh_samples, p=area / area.sum())
    u, v = rng.random((2, args.mesh_samples))
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = triangles[pick]
    surface = t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])
    np.save(args.out_dir / 'mesh_surface.npy', surface.astype(np.float32))
    index = dict(schema='gotcha_rift_pointcloud_export_v1', extent_m=extent, ground_z_m=car['ground'],
                 mesh=dict(dir=str(args.mesh_dir), samples=int(args.mesh_samples), source=manifest.get('source'),
                           stand_in=manifest.get('stand_in')), sets=[])
    for spec in args.run:
        label, source = spec.split('=', 1)
        if source.startswith('field:'):
            _, path, key = source.split(':', 2)
            z = np.load(path)
            energy_grid, axis = z[key], z['axis_m']
            gx, gy, gz = np.meshgrid(axis, axis, axis, indexing='ij')
            positions = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
            energy = energy_grid.ravel().astype(np.float64)
            info = dict(kind='voxel_image', source=path, key=key, voxel_m=float(axis[1] - axis[0]))
        else:
            checkpoint = torch.load(source, map_location='cpu', weights_only=False)
            positions, energy, max_order = point_energy(checkpoint['model_state_dict'], extent)
            info = dict(kind='rift_points', source=source, max_sh_order=max_order, **fit_record(checkpoint))
            info['points_active'] = int(len(energy))
            del checkpoint
            points, point_energy_ = positions, energy
            volume = deposit_points(torch.as_tensor(positions), torch.as_tensor(energy), extent, args.voxel_grid).numpy()
            axis = (np.arange(args.voxel_grid) + 0.5) * (2 * extent / args.voxel_grid) - extent
            gx, gy, gz = np.meshgrid(axis, axis, axis, indexing='ij')
            positions = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
            energy = volume.ravel()
            info.update(voxel_m=float(2 * extent / args.voxel_grid), readout='CIC deposition of every active point')
        order = np.argsort(-energy, kind='stable')[:args.top]
        info.update(on_car=partition(points, point_energy_, car, 0.125) if info['kind'] == 'rift_points' else None,
                    kept=int(len(order)), kept_energy_share=float(energy[order].sum() / energy.sum()),
                    cic_best=best_f1(args.geometry_dir, source) if args.geometry_dir else None, label=label)
        if info['on_car'] is not None:
            info['on_car'] = {k: info['on_car'][k] for k in ('near_share', 'off_share', 'off_by_place',
                                                            'weighted_median_distance_m')}
        np.save(args.out_dir / f'{label}.npy', np.concatenate([positions[order], energy[order, None]], 1).astype(np.float32))
        index['sets'].append(info)
        print(label, {k: info[k] for k in ('kind', 'kept', 'kept_energy_share')}, 'F1', (info['cic_best'] or {}).get('f1'),
              flush=True)
    (args.out_dir / 'index.json').write_text(json.dumps(index, indent=1) + '\n')


if __name__ == '__main__':
    main()
