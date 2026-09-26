#!/usr/bin/env python3
"""GOTCHA geometry for the new vehicles (Sentra, Santa Fe) under the paper's protocol (CPU; reads no radar response).

The Camry's protocol, unchanged, applied to a new vehicle's stand-in mesh:

1. Registration (``register_gotcha_camry_mesh_to_data_pvc.py``'s objective and grids with ``--fix-z``): heading about
   the footprint centre and horizontal shift that maximize the mean min-max normalized magnitude of the model-free
   TRAIN backprojection at mesh surface samples above the ground slab; height fixed at the calibrated ground. A new
   vehicle's box frame comes from a TRAIN-only radar locate that does not know which end is the nose, so its mesh
   exists as ``front_plusx`` and ``front_minusx`` (yaw 180 deg); both are registered and the higher objective wins.
   The heading window is +-15 deg (Camry: +-6 deg): the Camry mesh starts from a survey placement, a new vehicle's
   from the locate estimate (4 deg off on the Camry control). Fits at a grid edge are flagged in the record.
2. Scoring (``eval_gotcha_geometry_protocol_pvc.py --ground-cut``): fixed threshold t on the 48^3 lattice, Chamfer
   and Hausdorff against 20,000 surface samples, F1 at tau = one lattice pitch, solid/shell IoU on cube-side/60
   cells; lattice centres and truth samples below ground + tau dropped on both sides. Truth samples are drawn from
   the unmoved mesh with the scorer's seed and moved by the fitted rigid transform (a rigid move preserves the
   area-uniform and volume-uniform sampling).

Writes <output-dir>/registration.json (both variants' fits; the chosen one also drives the view panels through
``render_gotcha_newcar_panels_pvc.py --registration``) and <output-dir>/geometry.json (rows in the protocol scorer's
format, plus the table's five values per run: chamfer m^2, hausdorff m, hd95 m, iou (solid), f1).

    python scripts_pvc/eval_gotcha_newcar_geometry_pvc.py --target sentra_b15 --mesh-root MESHES \\
        --backprojection BP.npz --run rift=rift:CKPT --run spinr=spinr:CKPT \\
        --run backprojection=npz:BP.npz:data_energy --output-dir OUT
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.eval_b787_geometry_metrics import (chamfer, predicted_points, prf, sample_surface_points,  # noqa: E402
                                                sample_volume_points, voxel_iou)
from scripts.render_b787_vs_stl import load_stl_vertices, trilinear_sample_centers  # noqa: E402
from scripts_pvc.eval_gotcha_geometry_protocol_pvc import lattice_energy  # noqa: E402
from scripts_pvc.eval_gotcha_geometry_pvc import GRID  # noqa: E402
from scripts_pvc.register_gotcha_camry_mesh_to_data_pvc import Objective, load_image, search, transform  # noqa: E402

SCHEMA = 'gotcha_newcar_geometry_protocol_v1'
VARIANTS = ('front_plusx', 'front_minusx')


def variant(mesh_root, target, name):
    root = mesh_root / f'{target}_box_v1' / name
    manifest = json.loads((root / 'manifest.json').read_text())
    return root, manifest, root / f'{target}_region_local.stl', root / f'{target}_solid_region_local.npz'


def register(mesh_root, target, image_spec, samples):
    """The Camry protocol's fix-z registration, on each orientation variant."""
    fits = {}
    for name in VARIANTS:
        _, manifest, stl, _ = variant(mesh_root, target, name)
        extent, ground = float(manifest['region']['half_extent_m']), float(manifest['pose']['ground_local_z'])
        tau = 2 * extent / GRID
        surface = sample_surface_points(load_stl_vertices(stl).reshape(-1, 3, 3), samples, np.random.default_rng(0))
        pivot = 0.5 * (surface[:, :2].min(0) + surface[:, :2].max(0))
        image, info = load_image(image_spec, manifest['region'])
        objective = Objective(image, surface, extent, ground, tau)
        coarse = (np.arange(-15, 15.01, 1.0), np.arange(-0.8, 0.801, 0.1), np.arange(-0.5, 0.501, 0.1), np.zeros(1))
        fine = (np.arange(-1, 1.01, 0.25), np.arange(-0.1, 0.101, 0.025), np.arange(-0.1, 0.101, 0.025), np.zeros(1))
        fit = search(objective, surface, pivot, objective.intensity, (coarse, fine))
        moved = transform(surface, fit['yaw_deg'], np.array(fit['translation_m']), pivot)
        edges = [name_ for name_, value, limit in (('yaw', fit['yaw_deg'], 15.0),
                                                    ('x', fit['translation_m'][0], 0.8),
                                                    ('y', fit['translation_m'][1], 0.5))
                 if abs(value) >= limit - 1e-9]
        fit.update(pivot_xy_m=pivot.tolist(), ground_z_m=ground, source=info, at_grid_edge=edges,
                   f1_at_fit=objective.f1(moved, 0.0), objective_unregistered=objective.intensity(surface, 0.0))
        fits[name] = fit
        print(f"registration {name}: objective {fit['objective']:.5f} yaw {fit['yaw_deg']:+.2f} deg "
              f"shift {np.round(fit['translation_m'], 3).tolist()} m, F1 at fit {fit['f1_at_fit']:.4f}"
              + (f"  AT GRID EDGE: {edges}" if edges else ''), flush=True)
    chosen = max(VARIANTS, key=lambda name: fits[name]['objective'])
    return chosen, fits


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--target', required=True, choices=('sentra_b15', 'santafe_2004'))
    p.add_argument('--mesh-root', type=Path, required=True)
    p.add_argument('--backprojection', type=Path, required=True, help='TRAIN backprojection npz (data_energy)')
    p.add_argument('--run', action='append', required=True, help='LABEL=rift:CKPT | spinr:CKPT | npz:PATH:KEY | npy:PATH')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--fixed-threshold', type=float, default=0.2)
    p.add_argument('--samples', type=int, default=20000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ('registration.json', 'geometry.json'):
        if (args.output_dir / name).exists():
            raise FileExistsError(args.output_dir / name)

    chosen, fits = register(args.mesh_root, args.target, f'{args.backprojection}:data_energy', args.samples)
    root, manifest, stl, solid_path = variant(args.mesh_root, args.target, chosen)
    fit = fits[chosen]
    registration = dict(schema=SCHEMA + '_registration', target=args.target, region=manifest['region']['name'],
                        generated=datetime.datetime.now().astimezone().isoformat(timespec='minutes'),
                        chosen_variant=chosen, mesh_dir=str(root), mesh_stl=str(stl),
                        yaw_deg=fit['yaw_deg'], translation_m=fit['translation_m'], pivot_xy_m=fit['pivot_xy_m'],
                        ground_z_m=fit['ground_z_m'], fits=fits,
                        parameters='yaw about the vertical axis through pivot_xy, then translation; pitch = roll = 0; '
                                   'z fixed at the calibrated ground (Camry protocol --fix-z), both orientation variants')
    (args.output_dir / 'registration.json').write_text(json.dumps(registration, indent=2, default=str) + '\n')

    # truth: the scorer's samples of the chosen variant, moved by the fit
    extent = float(manifest['region']['half_extent_m'])
    tau, iou_unit = 2 * extent / GRID, 2 * extent / 60
    rng = np.random.default_rng(args.seed)
    surface = sample_surface_points(load_stl_vertices(stl).reshape(-1, 3, 3), args.samples, rng)
    solid = np.load(solid_path)
    axis = solid['axis_m']
    volume = sample_volume_points(solid['occupancy'], axis, axis, axis, args.samples, rng)
    pivot, shift = np.array(fit['pivot_xy_m']), np.array(fit['translation_m'])
    gt_surface, gt_volume = (transform(points, fit['yaw_deg'], shift, pivot) for points in (surface, volume))
    cut = fit['ground_z_m'] + tau
    kept = dict(surface=float((gt_surface[:, 2] >= cut).mean()), volume=float((gt_volume[:, 2] >= cut).mean()))
    gt_surface, gt_volume = gt_surface[gt_surface[:, 2] >= cut], gt_volume[gt_volume[:, 2] >= cut]
    lattice = trilinear_sample_centers(extent, GRID, GRID)
    centers = np.stack(np.meshgrid(lattice, lattice, lattice, indexing='ij'), -1).reshape(-1, 3)

    rows, table = [], {}
    for item in args.run:
        label, source = item.split('=', 1)
        energy, info = lattice_energy(source, extent, manifest['region'])
        magnitude = np.sqrt(np.clip(energy, 0, None))
        normalized = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-30)
        prediction = predicted_points(normalized, centers, args.fixed_threshold)
        above = prediction[prediction[:, 2] >= cut]
        row = dict(method=label, source=source, threshold=args.fixed_threshold, threshold_role='primary',
                   protocol='thresholded_field', points=int(len(above)),
                   points_below_cut=int(len(prediction) - len(above)), readout=info)
        if len(above):
            row.update(surface=chamfer(above, gt_surface), volume=chamfer(above, gt_volume),
                       prf=prf(above, gt_surface, tau),
                       iou_solid=voxel_iou(above, gt_volume, iou_unit, -extent, extent),
                       iou_shell=voxel_iou(above, gt_surface, iou_unit, -extent, extent))
            s = row['surface']
            table[label] = dict(chamfer=s['cham'], hausdorff=s['hausdorff_mm'] / 1000, hd95=s['hd95_mm'] / 1000,
                                iou=row['iou_solid'], f1=row['prf']['f1'], points=row['points'],
                                epoch=info.get('epoch'))
        rows.append(row)
        print(f"{label:16s} points {len(above)} (cut {len(prediction) - len(above)}) "
              + json.dumps(table.get(label), default=str), flush=True)
    result = dict(schema=SCHEMA, target=args.target,
                  truth=dict(mesh_dir=str(root), variant=chosen, source=manifest['source'], stand_in=manifest['stand_in'],
                             registration={k: registration[k] for k in ('yaw_deg', 'translation_m', 'pivot_xy_m')},
                             ground_z_m=fit['ground_z_m']),
                  ground_cut=dict(applied=True, z_min_m=cut, truth_fraction_kept=kept),
                  lattice=dict(grid=GRID, extent_m=extent), tau_m=tau, iou_unit_m=iou_unit, samples=args.samples,
                  table=table, rows=rows)
    with (args.output_dir / 'geometry.json').open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v))
        handle.write('\n')


if __name__ == '__main__':
    main()
