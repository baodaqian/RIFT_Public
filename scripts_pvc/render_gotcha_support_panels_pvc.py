#!/usr/bin/env python3
"""GOTCHA view-figure ground-truth support panels (PVC twin; CPU, reads no radar).

The reference row of the GOTCHA view figures shows where the registered reference surface lies on the evaluator's
48^3 lattice, in one uniform colour outside the paper's palette (user 2026-09-25: surface density does not
correspond to radar reflectivity, so the row carries no magnitude). A lattice cell is support when at least one
area-uniform sample of the mesh surface falls inside it (nearest cell of the cell-centred lattice that
``deposit_points`` uses); the binary field is densified 4x by the same trilinear interpolation as the method
panels and shown where it exceeds 1/2, projected along each camera axis.

Cameras, window, pixel size and background are those of ``render_gotcha_camry_mip_panels_pvc.py``; the pose is the
one the geometry table scores (the Camry's data-registered mesh; for the Sentra and Santa Fe the registration.json
variant and pose, rotated 180 deg about z for display when the nose is at -x, as ``render_gotcha_newcar_panels_pvc``).

Writes <output-dir>/panels/support_<front|side|top>.{png,pdf}, fields/support_g48.npy and manifest.json;
``--paper-dir`` copies the panels to <key>_support_view{1,2,3}.png (view1 front, view2 side, view3 top).

    python scripts_pvc/render_gotcha_support_panels_pvc.py --key camry \\
        --mesh-stl data/meshes/camry_xv20_data_registered_box_v2/camry_xv20_region_local.stl --output-dir OUT
    python scripts_pvc/render_gotcha_support_panels_pvc.py --key sentra --target sentra_b15 --mesh-root MESHES \\
        --registration GEOMETRY_DIR/registration.json --output-dir OUT [--paper-dir manuscripts/iclr27/figures/gotcha_sentra]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

SCHEMA = 'gotcha_support_panels_pvc_v1'
SLOTS = (('front', 'view1'), ('side', 'view2'), ('top', 'view3'))
# Okabe-Ito sky blue: not in inferno (the MIP and error scales), not RIFT's #dd513a, not the baseline gray
SUPPORT_COLOR = '#56B4E9'
GRID = 48


def lattice_support(points, extent, grid=GRID):
    """Cells of the cell-centred [grid]^3 lattice over [-extent, extent] that contain at least one point."""
    pitch = 2.0 * extent / grid
    index = np.clip(np.floor((points + extent) / pitch).astype(int), 0, grid - 1)
    support = np.zeros((grid,) * 3, dtype=bool)
    support[index[:, 0], index[:, 1], index[:, 2]] = True
    return support


def draw_support(axis, dense, view, extent, window, color):
    """The projected support in one colour on the panels' background (the orientation of ``draw_mip``)."""
    from matplotlib.colors import to_rgb
    from scripts_pvc.render_gotcha_camry_mip_panels_pvc import BACKGROUND, finish
    _, forward, _, (ra, _), (ua, _) = view
    projection = dense.max(axis=forward) > 0.5
    remaining = [a for a in range(3) if a != forward]
    mask = projection.T if remaining == [ra, ua] else projection    # rows = up axis, columns = right axis
    image = np.where(mask[..., None], np.array(to_rgb(color)), np.array(to_rgb(BACKGROUND)))
    axis.imshow(image, origin='lower', extent=[-extent, extent] * 2, aspect='equal', interpolation='bilinear')
    finish(axis, view, window)


def reference_triangles(args):
    """Registered reference triangles in the figure frame, and the lattice half-extent."""
    from scripts.render_b787_vs_stl import load_stl_vertices
    if args.key == 'camry':
        manifest = json.loads((args.mesh_stl.parent / 'manifest.json').read_text())
        triangles = load_stl_vertices(args.mesh_stl).reshape(-1, 3, 3).astype(np.float64)
        return triangles, float(manifest['region']['half_extent_m']), dict(mesh_stl=str(args.mesh_stl))
    from scripts_pvc.register_gotcha_camry_mesh_to_data_pvc import transform
    from scripts_pvc.render_gotcha_newcar_panels_pvc import variant_files
    registration = json.loads(args.registration.read_text())
    if registration['target'] != args.target:
        raise ValueError('registration.json belongs to another target')
    root, stl, _ = variant_files(args.mesh_root, args.target, registration['chosen_variant'])
    manifest = json.loads((root / 'manifest.json').read_text())
    triangles = load_stl_vertices(stl).reshape(-1, 3, 3).astype(np.float64)
    triangles = transform(triangles.reshape(-1, 3), registration['yaw_deg'], np.array(registration['translation_m']),
                          np.array(registration['pivot_xy_m'])).reshape(-1, 3, 3)
    flip = registration['chosen_variant'] == 'front_minusx'
    if flip:                                   # rotate 180 deg about z: nose to +x for the figure cameras
        triangles = triangles * np.array([-1.0, -1.0, 1.0])
    return triangles, float(manifest['region']['half_extent_m']), dict(
        mesh_stl=str(stl), registration=str(args.registration), variant=registration['chosen_variant'],
        yaw_deg=registration['yaw_deg'], translation_m=registration['translation_m'], rotated_180_about_z=flip)


def main(argv=None):
    import matplotlib
    matplotlib.use('Agg')
    from scripts.eval_b787_geometry_metrics import sample_surface_points
    from scripts.render_b787_vs_stl import trilinear_upsample
    from scripts_pvc.render_gotcha_camry_mip_panels_pvc import VIEWS, panel_width, save_panel
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--key', required=True, choices=('camry', 'sentra', 'santafe'))
    p.add_argument('--mesh-stl', type=Path, help='camry: the data-registered region-local STL')
    p.add_argument('--target', choices=('sentra_b15', 'santafe_2004'))
    p.add_argument('--mesh-root', type=Path, help='gotcha_new_targets_20260924/meshes')
    p.add_argument('--registration', type=Path, help="the geometry table's registration.json")
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--paper-dir', type=Path)
    p.add_argument('--window', type=float, nargs=3, default=[3.0, 1.5, 1.25])
    p.add_argument('--upsample', type=int, default=4)
    p.add_argument('--surface-samples', type=int, default=2_000_000)
    p.add_argument('--color', default=SUPPORT_COLOR)
    args = p.parse_args(argv)
    if args.key == 'camry' and args.mesh_stl is None:
        p.error('--key camry needs --mesh-stl')
    if args.key != 'camry' and None in (args.target, args.mesh_root, args.registration):
        p.error(f'--key {args.key} needs --target, --mesh-root and --registration')
    if args.output_dir.exists():
        raise FileExistsError(f'fresh output directory required: {args.output_dir}')

    triangles, extent, source = reference_triangles(args)
    points = sample_surface_points(triangles, args.surface_samples, np.random.default_rng(42))
    support = lattice_support(np.asarray(points), extent)
    dense = np.clip(trilinear_upsample(support.astype(np.float32), args.upsample), 0, None)
    (args.output_dir / 'fields').mkdir(parents=True)
    np.save(args.output_dir / 'fields' / 'support_g48.npy', support)

    files = []
    for view in VIEWS:
        files += save_panel(lambda axis: draw_support(axis, dense, view, extent, args.window, args.color),
                            args.output_dir / 'panels' / f'support_{view[0]}', panel_width(view, args.window))
    copied = []
    if args.paper_dir:
        args.paper_dir.mkdir(parents=True, exist_ok=True)
        for view, slot in SLOTS:
            dst = args.paper_dir / f'{args.key}_support_{slot}.png'
            shutil.copy2(args.output_dir / 'panels' / f'support_{view}.png', dst)
            copied.append(str(dst))
    report = dict(schema=SCHEMA, key=args.key, source=source, lattice=dict(grid=GRID, extent_m=extent),
                  support_cells=int(support.sum()), surface_samples=args.surface_samples, color=args.color,
                  window_half_m=args.window, upsample=args.upsample, display_threshold=0.5, files=files,
                  paper_files=copied)
    (args.output_dir / 'manifest.json').write_text(json.dumps(report, indent=1) + '\n')
    print(f'SUPPORT=PASS {args.key}: {int(support.sum())} cells, {len(files)} files, paper {len(copied)}')


if __name__ == '__main__':
    main()
