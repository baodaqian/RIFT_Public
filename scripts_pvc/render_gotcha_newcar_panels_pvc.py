#!/usr/bin/env python3
"""GOTCHA view-figure panels for the new vehicles (Sentra, Santa Fe), as the Camry's (PVC twin; CPU, reads no radar).

Same panels, scale and cameras as ``render_gotcha_camry_mip_panels_pvc.py`` (whose drawing functions this reuses):
one file per row x view, inferno over [t, 1] of per-method min-max sqrt(lattice energy), black below t, the
registered mesh rendered on its own. The Camry tools read the Camry's mesh files by name; this one takes the target.

Front/back orientation and pose. Each new vehicle's stand-in mesh exists in two variants, ``front_plusx`` and
``front_minusx`` (yaw 180 deg), because its box frame's long axis comes from a TRAIN-only radar locate that does
not know which end is the nose. ``--registration`` (``eval_gotcha_newcar_geometry_pvc.py``'s registration.json,
the paper's protocol) gives the variant and the data-registered pose, and the mesh rows show that pose, as the
geometry table scores it. Without it, the variant is chosen by the share of the model-free TRAIN backprojection's
lattice energy within ``--tau`` of each variant's solid (a quick check; both shares are recorded). The figure cameras assume the nose at +x
(front view from +x), so a ``front_minusx`` choice rotates every field and the mesh by 180 deg about z (lattice
indices reversed in x and y; the lattice is symmetric about the region origin). This is a display frame only.

Writes <output-dir>/panels/<row>_mip_<front|side|top>.png (+ .pdf), mesh_<view>.png, the fields, and
manifest.json; ``--paper-dir`` also copies the panels under the manuscript's slot names
<key>_<row>_mip_view{1,2,3}.png and <key>_mesh_view{1,2,3}.png (view1 front, view2 side, view3 top).

    python scripts_pvc/render_gotcha_newcar_panels_pvc.py --target sentra_b15 --key sentra \\
        --mesh-root MESHES --backprojection BP.npz --rift RIFT_CKPT --spinr SPINR_CKPT --output-dir OUT \\
        [--paper-dir manuscripts/iclr27/figures/gotcha_sentra]
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

SCHEMA = 'gotcha_newcar_mip_panels_pvc_v1'
VARIANTS = ('front_plusx', 'front_minusx')
SLOTS = (('front', 'view1'), ('side', 'view2'), ('top', 'view3'))


def variant_files(mesh_root, target, variant):
    root = mesh_root / f'{target}_box_v1' / variant
    return root, root / f'{target}_region_local.stl', root / f'{target}_solid_region_local.npz'


def near_share(centres, energy, solid_path, tau):
    """Share of lattice energy within tau of the solid (nearest-voxel distance lookup, energy-on-car readout)."""
    from scipy import ndimage
    from scripts_pvc.gotcha_energy_on_car_pvc import solid_distance
    solid = np.load(solid_path)
    occupancy, axis, voxel = solid['occupancy'], solid['axis_m'], float(solid['voxel_m'])
    car = dict(axis=axis, voxel=voxel, distance=ndimage.distance_transform_edt(~occupancy, sampling=voxel))
    _, distance = solid_distance(centres, car)
    total = float(energy.sum())
    return {f'within_{d:g}m': float(energy[distance <= d].sum() / total) for d in (tau, 2 * tau)}


def main(argv=None):
    import matplotlib
    matplotlib.use('Agg')
    import torch
    from scripts.eval_b787_geometry_metrics import sample_surface_points
    from scripts.eval_scene_geometry import deposit_points
    from scripts.render_b787_vs_stl import load_stl_vertices
    from scripts_pvc.gotcha_energy_on_car_pvc import lattice_energy
    from scripts_pvc.render_gotcha_camry_mip_panels_pvc import (VIEWS, draw_mesh, draw_mip, mesh_depth,
                                                               normalized_magnitude, normalized_support,
                                                               panel_width, save_panel)
    from scripts_pvc.run_rift_dataset_six_method_postflight_pvc import shade
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--target', required=True, choices=('sentra_b15', 'santafe_2004'))
    p.add_argument('--key', required=True, help='manuscript key: sentra or santafe')
    p.add_argument('--mesh-root', type=Path, required=True, help='gotcha_new_targets_20260924/meshes')
    p.add_argument('--backprojection', type=Path, required=True, help='TRAIN backprojection npz (data_energy)')
    p.add_argument('--rift', required=True, help='RIFT checkpoint (validation-selected)')
    p.add_argument('--spinr', required=True, help='SpINR-style checkpoint (validation-selected)')
    p.add_argument('--orientation', choices=('auto',) + VARIANTS, default='auto')
    p.add_argument('--registration', type=Path, help='registration.json: variant and data-registered pose')
    p.add_argument('--tau', type=float, default=0.125)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--paper-dir', type=Path)
    p.add_argument('--window', type=float, nargs=3, default=[3.0, 1.5, 1.25])
    p.add_argument('--upsample', type=int, default=4)
    p.add_argument('--threshold', type=float, default=0.20)
    p.add_argument('--surface-samples', type=int, default=2_000_000)
    p.add_argument('--mesh-samples', type=int, default=1_500_000)
    p.add_argument('--mesh-px', type=int, default=320)
    args = p.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f'fresh output directory required: {args.output_dir}')

    manifests = {v: json.loads((variant_files(args.mesh_root, args.target, v)[0] / 'manifest.json').read_text())
                 for v in VARIANTS}
    if manifests[VARIANTS[0]]['region'] != manifests[VARIANTS[1]]['region']:
        raise ValueError('the two mesh variants disagree on the region')
    manifest = manifests[VARIANTS[0]]
    extent = float(manifest['region']['half_extent_m'])

    runs = {'backprojection': f'npz:{args.backprojection}:data_energy', 'rift': f'rift:{args.rift}',
            'spinr': f'spinr:{args.spinr}'}
    fields, metas = {}, {}
    for label, spec in runs.items():
        centres, energy, meta = lattice_energy(spec, manifest)
        fields[label], metas[label] = energy.reshape(48, 48, 48), meta
        print(f"{label}: {meta['method']} epoch {meta.get('epoch')} cursor {meta.get('cursor')}", flush=True)

    shares = {v: near_share(centres, fields['backprojection'].reshape(-1),
                            variant_files(args.mesh_root, args.target, v)[2], args.tau) for v in VARIANTS}
    key = f'within_{args.tau:g}m'
    registration = json.loads(args.registration.read_text()) if args.registration else None
    if registration:
        chosen = registration['chosen_variant']
    else:
        chosen = max(VARIANTS, key=lambda v: shares[v][key]) if args.orientation == 'auto' else args.orientation
    flip = chosen == 'front_minusx'
    print(f'orientation: {chosen} ({"registration" if registration else "energy share"}; backprojection energy '
          f'{key}: ' + ', '.join(f'{v} {shares[v][key]:.4f}' for v in VARIANTS) + ')', flush=True)

    _, stl, _ = variant_files(args.mesh_root, args.target, chosen)
    triangles = load_stl_vertices(stl).reshape(-1, 3, 3).astype(np.float64)
    if registration:                           # the data-registered pose the geometry table scores
        from scripts_pvc.register_gotcha_camry_mesh_to_data_pvc import transform
        if registration['target'] != args.target:
            raise ValueError('registration.json belongs to another target')
        triangles = transform(triangles.reshape(-1, 3), registration['yaw_deg'],
                              np.array(registration['translation_m']),
                              np.array(registration['pivot_xy_m'])).reshape(-1, 3, 3)
    if flip:                                   # rotate 180 deg about z: nose to +x for the figure cameras
        triangles = triangles * np.array([-1.0, -1.0, 1.0])
        fields = {k: v[::-1, ::-1, :].copy() for k, v in fields.items()}

    out_panels, out_fields = args.output_dir / 'panels', args.output_dir / 'fields'
    out_fields.mkdir(parents=True)
    surface = deposit_points(torch.as_tensor(sample_surface_points(triangles, args.surface_samples,
                                                                   np.random.default_rng(42))),
                             torch.ones(args.surface_samples, dtype=torch.float64), extent, 48).numpy()
    rows = {'surface': normalized_support(surface, args.upsample)}
    np.save(out_fields / 'surface_energy_g48.npy', surface.astype(np.float32))
    for label in ('rift', 'spinr', 'backprojection'):
        np.save(out_fields / f'{label}_energy_g48.npy', fields[label].astype(np.float32))
        rows[label] = normalized_magnitude(fields[label], args.upsample)

    files = []
    for label, normalized in rows.items():
        for view in VIEWS:
            files += save_panel(lambda axis: draw_mip(axis, normalized, view, extent, args.window, args.threshold),
                                out_panels / f'{label}_mip_{view[0]}', panel_width(view, args.window))
    points = sample_surface_points(triangles, args.mesh_samples, np.random.default_rng(1))
    for view in VIEWS:
        depth, pixel = mesh_depth(points, view, args.window, args.mesh_px)
        shading = shade(depth, pixel)
        files += save_panel(lambda axis: draw_mesh(axis, shading, view, args.window),
                            out_panels / f'mesh_{view[0]}', panel_width(view, args.window))

    report = dict(schema=SCHEMA, target=args.target, key=args.key, region=manifest['region']['name'],
                  orientation=dict(chosen=chosen, requested=args.orientation, rotated_180_about_z=flip,
                                   backprojection_energy_share=shares, tau_m=args.tau,
                                   registration=None if not registration else dict(
                                       path=str(args.registration), yaw_deg=registration['yaw_deg'],
                                       translation_m=registration['translation_m'])),
                  mesh_stl=str(stl), window_half_m=args.window, threshold=args.threshold, upsample=args.upsample,
                  runs={k: dict(source=runs[k], epoch=metas[k].get('epoch'), cursor=metas[k].get('cursor'),
                                method=metas[k]['method'],
                                source_mtime=os.path.getmtime(metas[k]['checkpoint'])
                                if os.path.exists(metas[k]['checkpoint']) else None,
                                visible_fraction=float((rows[k] > args.threshold).mean()))
                        for k in runs},
                  files=files)
    if args.paper_dir:
        args.paper_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        for view, slot in SLOTS:
            for row in ('surface', 'rift', 'spinr', 'backprojection'):
                target = args.paper_dir / f'{args.key}_{row}_mip_{slot}.png'
                shutil.copyfile(out_panels / f'{row}_mip_{view}.png', target)
                copied.append(str(target))
            target = args.paper_dir / f'{args.key}_mesh_{slot}.png'
            shutil.copyfile(out_panels / f'mesh_{view}.png', target)
            copied.append(str(target))
        report['paper_panels'] = copied
    (args.output_dir / 'manifest.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(json.dumps({k: report[k] for k in ('orientation', 'runs')}, indent=2, default=str), flush=True)


if __name__ == '__main__':
    main()
