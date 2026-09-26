#!/usr/bin/env python3
"""GOTCHA Camry geometry under the paper's protocol: data-registered reference, ground slab excluded (CPU).

Same metrics and functions as ``scripts_pvc/eval_gotcha_geometry_pvc.py`` (thresholded 48^3 lattice readout, Chamfer
and Hausdorff against 20,000 surface samples, precision/recall/F1 at tau = one lattice pitch, solid and shell IoU on
cube-side/60 cells), with two protocol settings (user decision 2026-09-24, "protocol C"):

* ``--mesh-dir``: the reference, normally ``data/meshes/camry_xv20_data_registered_box_v2`` (the stand-in mesh moved by
  one rigid transform estimated from the model-free TRAIN backprojection, ``register_gotcha_camry_mesh_to_data_pvc.py``);
* ``--ground-cut``: lattice centres and truth samples (surface and volume) below the reference's ground plane plus one
  lattice pitch are dropped from both sides, because the reference has no ground and car-ground multipath is real
  scattering that would otherwise count as false positives.

Runs are LABEL=npy:PATH (a lattice energy saved by the renderer), LABEL=npz:PATH:KEY, LABEL=rift:CKPT or
LABEL=spinr:CKPT (read as the scorer reads them). Without ``--ground-cut`` and with the original mesh this reproduces
``eval_gotcha_geometry_pvc.py``.

    python scripts_pvc/eval_gotcha_geometry_protocol_pvc.py --mesh-dir data/meshes/camry_xv20_data_registered_box_v2 \\
        --ground-cut --output out.json --run rift=npy:FIELD.npy --run backprojection_full=npz:IMAGE.npz:data_energy
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.eval_b787_geometry_metrics import chamfer, predicted_points, prf, voxel_iou  # noqa: E402
from scripts.render_b787_vs_stl import trilinear_sample_centers  # noqa: E402
from scripts_pvc.eval_gotcha_geometry_pvc import GRID, npz_field, rift_field, spinr_field, truth  # noqa: E402


def lattice_energy(source, extent, region):
    kind, rest = source.split(':', 1)
    if kind == 'npy':
        energy = np.load(rest).astype(np.float64)
        info = dict(readout='renderer lattice energy', path=rest)
    elif kind == 'npz':
        path, key = rest.rsplit(':', 1)
        field_region, energy, info = npz_field(path, key)
        if json.loads(json.dumps(field_region)) != json.loads(json.dumps(region)):
            raise ValueError(f'{path}: field region differs from the reference region')
    elif kind in ('rift', 'spinr'):
        ck, fields = (rift_field if kind == 'rift' else spinr_field)(rest, extent)
        if ck['dataset_contract']['region'] != region:
            raise ValueError(f'{rest}: checkpoint region differs from the reference region')
        (energy, info), = fields.values()
        info = dict(info, checkpoint=rest, epoch=ck.get('epoch'))
    else:
        raise ValueError(f'unknown run kind {kind!r}')
    energy = np.asarray(energy, dtype=np.float64).reshape(GRID, GRID, GRID)
    return energy, info


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mesh-dir', type=Path, required=True)
    p.add_argument('--run', action='append', required=True)
    p.add_argument('--ground-cut', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fixed-threshold', type=float, default=0.2)
    p.add_argument('--thresholds', type=float, nargs='*', default=[])
    p.add_argument('--samples', type=int, default=20000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest, gt_surface, gt_volume = truth(args.mesh_dir, args.samples, args.seed)
    extent = float(manifest['region']['half_extent_m'])
    tau, iou_unit = 2 * extent / GRID, 2 * extent / 60
    ground = float(manifest['pose']['ground_z_m'])
    cut = ground + tau if args.ground_cut else -np.inf
    kept = dict(surface=float((gt_surface[:, 2] >= cut).mean()), volume=float((gt_volume[:, 2] >= cut).mean()))
    gt_surface, gt_volume = gt_surface[gt_surface[:, 2] >= cut], gt_volume[gt_volume[:, 2] >= cut]
    axis = trilinear_sample_centers(extent, GRID, GRID)
    centers = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
    rows = []
    for item in args.run:
        label, source = item.split('=', 1)
        energy, info = lattice_energy(source, extent, manifest['region'])
        magnitude = np.sqrt(np.clip(energy, 0, None))
        normalized = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-30)
        for threshold in [args.fixed_threshold, *args.thresholds]:
            prediction = predicted_points(normalized, centers, threshold)
            above = prediction[prediction[:, 2] >= cut]
            row = dict(method=label, source=source, threshold=threshold,
                       threshold_role='primary' if threshold == args.fixed_threshold else 'sweep',
                       protocol='thresholded_field', points=int(len(above)), points_below_cut=int(len(prediction) - len(above)),
                       readout=info)
            if len(above):
                row.update(surface=chamfer(above, gt_surface), volume=chamfer(above, gt_volume),
                           prf=prf(above, gt_surface, tau),
                           iou_solid=voxel_iou(above, gt_volume, iou_unit, -extent, extent),
                           iou_shell=voxel_iou(above, gt_surface, iou_unit, -extent, extent))
            rows.append(row)
            if threshold == args.fixed_threshold:
                print(f"{label:22s} t={threshold} points {len(above)} (cut {len(prediction) - len(above)}) "
                      f"F1 {row.get('prf', {}).get('f1')} chamfer {row.get('surface', {}).get('cham')}", flush=True)
    result = dict(schema='gotcha_camry_geometry_protocol_v1',
                  truth=dict(mesh_dir=str(args.mesh_dir.resolve()), source=manifest['source'], stand_in=manifest['stand_in'],
                             registration=manifest['pose'], ground_z_m=ground),
                  ground_cut=dict(applied=bool(args.ground_cut), z_min_m=None if not args.ground_cut else cut,
                                  truth_fraction_kept=kept),
                  lattice=dict(grid=GRID, extent_m=extent), tau_m=tau, iou_unit_m=iou_unit, samples=args.samples, rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v))
        handle.write('\n')


if __name__ == '__main__':
    main()
