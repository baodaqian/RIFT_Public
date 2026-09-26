#!/usr/bin/env python3
"""Write the data-registered GOTCHA Camry reference mesh (CPU; geometry only).

Applies the rigid pose estimated by ``register_gotcha_camry_mesh_to_data_pvc.py`` (yaw about the vertical axis through
the footprint pivot, then translation) to an existing reference mesh directory and writes a new one with the same
file layout, so ``eval_gotcha_geometry_pvc.py``-style scorers and ``render_gotcha_camry_mip_panels_pvc.py`` take it via
``--mesh-dir``:

* ``camry_xv20_region_local.stl``: the transformed surface (the native-frame STL is not written);
* ``camry_xv20_solid_region_local.npz``: occupancy and surface voxels resampled on the source's own 2 cm axis by
  nearest-neighbour inverse mapping (``axis_m``/``voxel_m`` unchanged);
* ``manifest.json``: the source manifest with ``pose.data_registration`` (the estimate and its provenance) and the
  ground height (unchanged when z is fixed).

    python scripts_pvc/apply_gotcha_camry_mesh_registration_pvc.py --registration reg.json \\
        --source data/meshes/camry_xv20_data_frame_box_v2 --output data/meshes/camry_xv20_data_registered_box_v2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.render_b787_vs_stl import load_stl_vertices  # noqa: E402
from scripts_pvc.prepare_gotcha_camry_mesh import write_binary_stl  # noqa: E402
from scripts_pvc.register_gotcha_camry_mesh_to_data_pvc import transform  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--source', type=Path, default=Path('data/meshes/camry_xv20_data_frame_box_v2'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    registration = json.loads(args.registration.read_text())
    if Path(registration['mesh_dir']).resolve() != args.source.resolve():
        raise ValueError('the registration was estimated against a different mesh directory')
    yaw, shift = float(registration['estimate']['yaw_deg']), np.asarray(registration['estimate']['translation_m'], float)
    pivot = np.asarray(registration['pivot_xy_m'], float)
    manifest = json.loads((args.source/'manifest.json').read_text())
    vertices = load_stl_vertices(args.source/'camry_xv20_region_local.stl').astype(np.float64)
    moved = transform(vertices, yaw, shift, pivot).reshape(-1, 3, 3)
    solid = np.load(args.source/'camry_xv20_solid_region_local.npz')
    axis, voxel = solid['axis_m'], float(solid['voxel_m'])
    grids = {}
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    X, Y = np.meshgrid(axis, axis, indexing='ij')
    # inverse map of every target (x, y) column: undo the translation, then rotate by -yaw about the pivot
    u, v = X - pivot[0] - shift[0], Y - pivot[1] - shift[1]
    sx, sy = c * u + s * v + pivot[0], -s * u + c * v + pivot[1]
    ix, iy = np.rint((sx - axis[0]) / voxel).astype(int), np.rint((sy - axis[0]) / voxel).astype(int)
    iz = np.rint((axis - shift[2] - axis[0]) / voxel).astype(int)
    inside_xy = (ix >= 0) & (ix < len(axis)) & (iy >= 0) & (iy < len(axis))
    inside_z = (iz >= 0) & (iz < len(axis))
    for key in ('occupancy', 'surface'):
        source = solid[key]
        out = np.zeros_like(source)
        columns = source[np.clip(ix, 0, len(axis) - 1), np.clip(iy, 0, len(axis) - 1)]     # (n, n, nz)
        out[:, :, inside_z] = columns[:, :, iz[inside_z]]
        out[~inside_xy] = False
        grids[key] = out
    args.output.mkdir(parents=True)
    write_binary_stl(args.output/'camry_xv20_region_local.stl', moved, header='rift gotcha camry xv20 data-registered')
    np.savez_compressed(args.output/'camry_xv20_solid_region_local.npz', axis_m=axis, voxel_m=np.asarray(voxel),
                        closing_iterations=solid['closing_iterations'], **grids)
    manifest = json.loads(json.dumps(manifest))
    manifest['pose']['ground_z_m'] = float(manifest['pose']['ground_z_m']) + float(shift[2])
    manifest['pose']['data_registration'] = dict(
        registration=str(args.registration.resolve()), source_mesh_dir=str(args.source), yaw_deg=yaw,
        translation_m=shift.tolist(), pivot_xy_m=pivot.tolist(), parameters=registration['parameters'],
        objective=registration['objective'], estimate_source=registration['estimate']['source'],
        fits={k: {kk: vv for kk, vv in f.items() if kk != 'source'} for k, f in registration['fits'].items()},
        decision='user 2026-09-24: adopt the data-registered reference as the primary GOTCHA geometry truth (protocol C)')
    manifest['outputs'] = ['camry_xv20_region_local.stl', 'camry_xv20_solid_region_local.npz']
    manifest['solid']['solid_voxels'] = int(grids['occupancy'].sum())
    manifest['solid']['surface_voxels'] = int(grids['surface'].sum())
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(f"wrote {args.output}: yaw {yaw} deg, shift {shift.tolist()} m; solid voxels "
          f"{int(solid['occupancy'].sum())} -> {int(grids['occupancy'].sum())}")


if __name__ == '__main__':
    main()
