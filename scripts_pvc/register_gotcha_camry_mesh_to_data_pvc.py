#!/usr/bin/env python3
"""Rigid registration of the GOTCHA Camry reference mesh to the measured TRAIN data (CPU; no model output is read).

The reference is a same-generation stand-in mesh placed from survey documents (``prepare_gotcha_camry_mesh.py``),
and every reconstruction, including the model-free backprojection of the measured TRAIN data, sits offset from it
(user decision 2026-09-24: adopt a data-registered reference for the paper's GOTCHA geometry). The car rests on the
ground, so its unknown pose is a heading (yaw about the vertical axis through the mesh footprint centre) plus a
translation; pitch and roll are not identifiable from GOTCHA's narrow elevation aperture and stay zero. With
``--fix-z`` (the protocol's setting) the height stays at the calibrated ground: a free vertical shift is biased by the
car-ground multipath line (lowering the car moves the lower body onto it), and the trihedral calibration already
confirms heights to 0.03 m.

Objective (template matching, independent of the evaluation threshold and tau): the mean of the backprojected image's
min-max normalized magnitude, trilinearly sampled at area-uniform mesh surface samples that lie above the moved
ground plane plus one lattice pitch (the ground slab, where car-ground multipath lives, is excluded exactly as in the
ground-cut scorer). Coarse grid, then a fine grid around the optimum. The F1-at-tau optimum on the same image is
reported beside it as a cross-check, and the fit is repeated on a second image (e.g. the 578-unit subset) for
stability.

    python scripts_pvc/register_gotcha_camry_mesh_to_data_pvc.py --output reg.json \\
        --image full=IMAGE.npz:data_energy --image subset=IMAGE578.npz:data_energy [--mesh-dir DIR]
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
from scipy.ndimage import map_coordinates  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from scripts.eval_b787_geometry_metrics import sample_surface_points  # noqa: E402
from scripts.render_b787_vs_stl import load_stl_vertices  # noqa: E402

GRID = 48


def normalized_magnitude(energy):
    magnitude = np.sqrt(np.clip(np.asarray(energy, dtype=np.float64), 0, None))
    return (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())


def load_image(spec, region):
    path, key = spec.rsplit(':', 1)
    data = np.load(path)
    meta = json.loads(str(data['meta'])) if 'meta' in data.files else {}
    if meta.get('region') is not None and json.loads(json.dumps(meta['region'])) != json.loads(json.dumps(region)):
        raise ValueError(f'{path}: image region differs from the mesh region')
    energy = np.asarray(data[key], dtype=np.float64)
    if energy.shape != (GRID,) * 3:
        raise ValueError(f'{path}:{key} is not a {GRID}^3 lattice field')
    return normalized_magnitude(energy), dict(path=path, key=key, readout=meta.get('readout'), label=meta.get('label'))


def transform(points, yaw_deg, shift, pivot):
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    xy = points[:, :2] - pivot
    out = points.copy()
    out[:, 0] = c * xy[:, 0] - s * xy[:, 1] + pivot[0] + shift[0]
    out[:, 1] = s * xy[:, 0] + c * xy[:, 1] + pivot[1] + shift[1]
    out[:, 2] = points[:, 2] + shift[2]
    return out


class Objective:
    def __init__(self, image, surface, extent, ground, tau):
        self.image, self.surface, self.extent, self.ground, self.tau = image, surface, extent, ground, tau
        self.pitch = 2 * extent / GRID
        axis = -extent + self.pitch * (np.arange(GRID) + 0.5)
        self.centres = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)

    def intensity(self, moved, dz):
        keep = moved[:, 2] >= self.ground + dz + self.tau
        index = (moved[keep] + self.extent) / self.pitch - 0.5          # cell-centre index coordinates
        return float(map_coordinates(self.image, index.T, order=1, mode='constant', cval=0.0).mean())

    def f1(self, moved, dz, threshold=0.2):
        cut = self.ground + dz + self.tau
        prediction = self.centres[self.image.reshape(-1) > threshold]
        prediction, truth = prediction[prediction[:, 2] >= cut], moved[moved[:, 2] >= cut]
        d1, _ = cKDTree(truth).query(prediction)
        d2, _ = cKDTree(prediction).query(truth)
        p, r = float((d1 <= self.tau).mean()), float((d2 <= self.tau).mean())
        return 2 * p * r / (p + r) if p + r else 0.0


def search(objective, surface, pivot, score, grids):
    best = (-np.inf, 0.0, np.zeros(3))
    centre_yaw, centre_shift = 0.0, np.zeros(3)
    for yaws, dxs, dys, dzs in grids:
        for yaw in centre_yaw + yaws:
            rotated = transform(surface, yaw, np.zeros(3), pivot)
            for dx in centre_shift[0] + dxs:
                for dy in centre_shift[1] + dys:
                    for dz in centre_shift[2] + dzs:
                        shift = np.array([dx, dy, dz])
                        value = score(rotated + shift, dz)
                        if value > best[0]:
                            best = (value, float(yaw), shift)
        centre_yaw, centre_shift = best[1], best[2]
    return dict(objective=best[0], yaw_deg=best[1], translation_m=best[2].tolist())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mesh-dir', type=Path, default=Path('data/meshes/camry_xv20_data_frame_box_v2'))
    p.add_argument('--image', action='append', required=True, help='LABEL=NPZ:KEY; the first one is the estimate')
    p.add_argument('--samples', type=int, default=20000)
    p.add_argument('--fix-z', action='store_true', help='keep the calibrated height (fit yaw and horizontal shift only)')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    manifest = json.loads((args.mesh_dir/'manifest.json').read_text())
    extent, ground = float(manifest['region']['half_extent_m']), float(manifest['pose']['ground_z_m'])
    tau = 2 * extent / GRID
    triangles = load_stl_vertices(args.mesh_dir/'camry_xv20_region_local.stl').reshape(-1, 3, 3)
    surface = sample_surface_points(triangles, args.samples, np.random.default_rng(0))
    pivot = 0.5 * (surface[:, :2].min(0) + surface[:, :2].max(0))
    coarse = (np.arange(-6, 6.01, 1.0), np.arange(-0.8, 0.801, 0.1), np.arange(-0.5, 0.501, 0.1), np.arange(-0.25, 0.251, 0.05))
    fine = (np.arange(-1, 1.01, 0.25), np.arange(-0.1, 0.101, 0.025), np.arange(-0.1, 0.101, 0.025), np.arange(-0.05, 0.051, 0.0125))
    f1_grid = (np.arange(-2, 2.01, 0.5), np.arange(-0.15, 0.151, 0.05), np.arange(-0.15, 0.151, 0.05), np.arange(-0.0625, 0.0626, 0.03125))
    if args.fix_z:
        coarse, fine, f1_grid = ((*grid[:3], np.zeros(1)) for grid in (coarse, fine, f1_grid))
    results = {}
    for spec in args.image:
        label, source = spec.split('=', 1)
        image, info = load_image(source, manifest['region'])
        objective = Objective(image, surface, extent, ground, tau)
        fit = search(objective, surface, pivot, objective.intensity, (coarse, fine))
        moved = transform(surface, fit['yaw_deg'], np.array(fit['translation_m']), pivot)
        fit.update(source=info, f1_at_fit=objective.f1(moved, fit['translation_m'][2]),
                   objective_unregistered=objective.intensity(surface, 0.0), f1_unregistered=objective.f1(surface, 0.0))
        # cross-check: the F1 optimum near the intensity optimum (metric-in-the-loop; reported, not used)
        cross = search(objective, surface, pivot, objective.f1,
                       ((f1_grid[0] + fit['yaw_deg'], f1_grid[1] + fit['translation_m'][0],
                         f1_grid[2] + fit['translation_m'][1], f1_grid[3] + fit['translation_m'][2]),))
        fit['f1_optimum_crosscheck'] = cross
        results[label] = fit
        print(label, json.dumps({k: v for k, v in fit.items() if k != 'source'}), flush=True)
    estimate = results[args.image[0].split('=', 1)[0]]
    record = dict(schema='gotcha_camry_mesh_data_registration_v1',
                  generated=datetime.datetime.now().astimezone().isoformat(timespec='minutes'),
                  mesh_dir=str(args.mesh_dir), pivot_xy_m=pivot.tolist(), ground_z_m=ground, tau_m=tau,
                  parameters='yaw about the vertical axis through pivot_xy, then translation; pitch = roll = 0'
                             + ('; z fixed at the calibrated ground' if args.fix_z else ''),
                  objective='mean min-max normalized backprojected magnitude on mesh surface samples above ground + tau',
                  estimate=dict(yaw_deg=estimate['yaw_deg'], translation_m=estimate['translation_m'],
                                source=args.image[0]), fits=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1) + '\n')


if __name__ == '__main__':
    main()
