#!/usr/bin/env python3
"""Standalone max-intensity panels of GOTCHA Camry reconstructions, any method (PVC twin; CPU only, reads no radar data).

The user's figure format (the RIFT-dataset postflight's): one file per method x view, never a composite; one colour
scale for every panel (inferno over [t, 1] of the per-method min-max magnitude, black below t) and one colour bar
exactly the panel height; the registered mesh rendered on its own at the same window and panel height, never
overlaid. Each field is read the way the geometry scorer reads it (``gotcha_energy_on_car_pvc.lattice_energy``:
RIFT the CIC point energy, SpINR sigma^2, an npz such as SE's Stage-1 ``se_energy``) on the scorer's 48^3 lattice,
then shown as sqrt(trilinearly upsampled energy) (display only), min-max normalized per method, so the scale is
shared but brightness is not physically comparable across methods.

Views are proper (unmirrored) orthographic cameras in the box frame (x front, y left, z up), nose to the right in
top and side: top from +z, side from -y (the car's right), front from +x. Each panel shows the box region
(``--window``, half-widths x y z), so panels are rectangular with a common height.

    python scripts_pvc/render_gotcha_camry_mip_panels_pvc.py --output-dir OUT \\
        --run rift_F5full=rift:CKPT --run spinr_B=spinr:CKPT --run se_A=npz:PATH.npz:se_energy
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

SCHEMA = 'gotcha_camry_mip_panels_pvc_v1'
# name, forward axis, camera side (+1: camera on the + end, looking toward -), (right axis, sign), (up axis, sign)
VIEWS = (('top', 2, 1, (0, 1), (1, 1)),
         ('side', 1, -1, (0, 1), (2, 1)),
         ('front', 0, 1, (1, 1), (2, 1)))
PANEL_H_IN, PANEL_DPI, BACKGROUND = 1.6, 200, '#050505'


def normalized_magnitude(energy, upsample):
    from scripts.render_b787_vs_stl import trilinear_upsample
    magnitude = np.sqrt(np.clip(trilinear_upsample(energy, upsample), 0, None))
    lower, upper = float(magnitude.min()), float(magnitude.max())
    if not upper > lower:
        raise ValueError('field cannot be min-max normalized')
    return (magnitude - lower) / (upper - lower)


def mesh_surface_density(mesh_dir, samples):
    """The 3D reference as the scorer lattice sees it: area-uniform samples of the registered mesh surface,
    CIC-deposited on the 48^3 lattice (``deposit_points``, the postflight's SE-surface readout). Shown as native
    support (the density itself is the magnitude), not sqrt."""
    import torch
    from scripts.eval_b787_geometry_metrics import sample_surface_points
    from scripts.eval_scene_geometry import deposit_points
    from scripts.render_b787_vs_stl import load_stl_vertices
    manifest = json.loads((mesh_dir/'manifest.json').read_text())
    extent = float(manifest['region']['half_extent_m'])
    triangles = load_stl_vertices(mesh_dir/'camry_xv20_region_local.stl').reshape(-1, 3, 3)
    points = sample_surface_points(triangles, samples, np.random.default_rng(42))
    return deposit_points(torch.as_tensor(points), torch.ones(len(points), dtype=torch.float64), extent, 48).numpy()


def normalized_support(support, upsample):
    from scripts.render_b787_vs_stl import trilinear_upsample
    dense = np.clip(trilinear_upsample(support, upsample), 0, None)
    return (dense - dense.min()) / (dense.max() - dense.min())


def window_of(view, window):
    _, _, _, (ra, _), (ua, _) = view
    return (-window[ra], window[ra]), (-window[ua], window[ua])


def panel_width(view, window):
    (x0, x1), (y0, y1) = window_of(view, window)
    return PANEL_H_IN * (x1 - x0) / (y1 - y0)


def save_panel(draw, stem, width_in):
    import matplotlib.pyplot as plt
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(width_in, PANEL_H_IN))
    draw(figure.add_axes([0, 0, 1, 1]))
    paths = [stem.with_suffix(f'.{suffix}') for suffix in ('png', 'pdf')]
    for path in paths:
        figure.savefig(path, dpi=PANEL_DPI, facecolor=BACKGROUND)
    plt.close(figure)
    return [str(p) for p in paths]


def finish(axis, view, window):
    (x0, x1), (y0, y1) = window_of(view, window)
    _, _, _, (_, rs), (_, us) = view
    axis.set_xlim((x0, x1) if rs > 0 else (x1, x0))
    axis.set_ylim((y0, y1) if us > 0 else (y1, y0))
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_facecolor(BACKGROUND)
    for spine in axis.spines.values():
        spine.set_color('#252525')


def draw_mip(axis, normalized, view, extent, window, threshold):
    import matplotlib.pyplot as plt
    _, forward, _, (ra, _), (ua, _) = view
    projection = normalized.max(axis=forward)
    remaining = [a for a in range(3) if a != forward]
    image = projection.T if remaining == [ra, ua] else projection    # rows = up axis, columns = right axis
    cmap = plt.get_cmap('inferno').copy()
    cmap.set_bad(BACKGROUND)
    axis.imshow(np.ma.masked_less_equal(image, threshold), origin='lower', extent=[-extent, extent] * 2, cmap=cmap,
                aspect='equal', interpolation='bilinear', vmin=threshold, vmax=1.0)
    finish(axis, view, window)


def mesh_depth(points, view, window, px_h, splat_px=2):
    """Orthographic z-buffer over the view's window (the postflight's ``render_depth`` on a rectangle), each
    surface sample splatted as a disk of ``splat_px`` pixels so the sampled surface renders without pinholes."""
    from scipy.ndimage import minimum_filter
    _, forward, side, (ra, rs), (ua, us) = view
    (x0, x1), (y0, y1) = window_of(view, window)
    pixel = (y1 - y0) / px_h
    px_w = int(round((x1 - x0) / pixel))
    u = np.floor((rs * points[:, ra] - (x0 if rs > 0 else -x1)) / pixel).astype(np.int64)
    v = np.floor((us * points[:, ua] - (y0 if us > 0 else -y1)) / pixel).astype(np.int64)
    ok = (u >= 0) & (u < px_w) & (v >= 0) & (v < px_h)
    zbuf = np.full(px_h * px_w, np.inf)
    np.minimum.at(zbuf, v[ok] * px_w + u[ok], -side * points[ok, forward])
    yy, xx = np.mgrid[-splat_px:splat_px + 1, -splat_px:splat_px + 1]
    zbuf = minimum_filter(zbuf.reshape(px_h, px_w), footprint=(xx * xx + yy * yy) <= (splat_px + .5) ** 2,
                          mode='constant', cval=np.inf)
    return zbuf, pixel


def draw_mesh(axis, shading, view, window):
    import matplotlib.pyplot as plt
    (x0, x1), (y0, y1) = window_of(view, window)
    _, _, _, (_, rs), (_, us) = view
    cmap = plt.get_cmap('Greys_r').copy()
    cmap.set_bad(alpha=0)
    # the z-buffer's pixel (0, 0) is the camera's lower-left corner; ``finish`` then orients the axes
    image = shading[:, ::-1] if rs < 0 else shading
    image = image[::-1] if us < 0 else image
    axis.imshow(image, origin='lower', extent=[x0, x1, y0, y1], cmap=cmap, vmin=-.1, vmax=1.15,
                interpolation='nearest', aspect='equal')
    finish(axis, view, window)


def save_colorbar(stem, threshold):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    stem.parent.mkdir(parents=True, exist_ok=True)
    width, bar = 0.72, 0.16
    figure = plt.figure(figsize=(width, PANEL_H_IN))
    scale = figure.colorbar(ScalarMappable(Normalize(threshold, 1.0), plt.get_cmap('inferno')),
                            cax=figure.add_axes([0, 0, bar / width, 1]))
    ticks = [threshold] + [v for v in (.4, .6, .8) if v > threshold + .05] + [1.0]
    scale.set_ticks(ticks, labels=[f'{v:.1f}' for v in ticks])
    scale.ax.tick_params(labelsize=6.5, length=2.5, width=.6, pad=1.5)
    labels = scale.ax.get_yticklabels()
    labels[0].set_verticalalignment('bottom')
    labels[-1].set_verticalalignment('top')
    scale.outline.set_linewidth(.6)
    scale.set_label('min–max normalized magnitude', fontsize=6.5, labelpad=2)
    paths = [stem.with_suffix(f'.{suffix}') for suffix in ('png', 'pdf')]
    for path in paths:
        figure.savefig(path, dpi=PANEL_DPI, facecolor='white')
    plt.close(figure)
    return [str(p) for p in paths]


def main(argv=None):
    import matplotlib
    matplotlib.use('Agg')
    from scripts.eval_b787_geometry_metrics import sample_surface_points
    from scripts.render_b787_vs_stl import load_stl_vertices
    from scripts_pvc.gotcha_energy_on_car_pvc import lattice_energy
    from scripts_pvc.run_rift_dataset_six_method_postflight_pvc import shade
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', action='append', required=True,
                   help='LABEL=rift:CKPT | LABEL=spinr:CKPT | LABEL=npz:PATH:KEY | LABEL=mesh:SAMPLES (the reference '
                        'surface density on the lattice); LABEL is the file stem')
    p.add_argument('--mesh-dir', type=Path, default=Path('data/meshes/camry_xv20_data_frame_box_v2'))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--window', type=float, nargs=3, default=[3.0, 1.5, 1.25], help='half-widths shown (x, y, z), m')
    p.add_argument('--upsample', type=int, default=4)
    p.add_argument('--threshold', type=float, default=0.20)
    p.add_argument('--mesh-samples', type=int, default=1_500_000)
    p.add_argument('--mesh-px', type=int, default=320, help='mesh panel height in pixels')
    args = p.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f'fresh output directory required: {args.output_dir}')
    manifest = json.loads((args.mesh_dir/'manifest.json').read_text())
    extent = float(manifest['region']['half_extent_m'])
    panels, fields = args.output_dir/'panels', args.output_dir/'fields'
    fields.mkdir(parents=True)
    report = dict(schema=SCHEMA, mesh_dir=str(args.mesh_dir), window_half_m=args.window, threshold=args.threshold,
                  upsample=args.upsample, lattice=dict(grid=48, extent_m=extent),
                  views={v[0]: dict(forward_axis=v[1], camera_side=v[2], right=v[3], up=v[4],
                                    width_in=round(panel_width(v, args.window), 3)) for v in VIEWS},
                  panel_height_in=PANEL_H_IN, dpi=PANEL_DPI,
                  scale='inferno over [t, 1] of per-method min-max sqrt(energy); black below t; not comparable '
                        'in absolute brightness across methods', runs={}, files=[])
    for spec in args.run:
        label, source = spec.split('=', 1)
        if source.startswith('mesh:'):
            energy = mesh_surface_density(args.mesh_dir, int(source.split(':', 1)[1]))
            meta = dict(checkpoint=str(args.mesh_dir/'camry_xv20_region_local.stl'), method='mesh',
                        readout='reference surface density, CIC on the 48^3 lattice, shown as native support')
            normalized = normalized_support(energy, args.upsample)
        else:
            _, energy, meta = lattice_energy(source, manifest)
            energy = energy.reshape(48, 48, 48)
            normalized = normalized_magnitude(energy, args.upsample)
        np.save(fields/f'{label}_energy_g48.npy', energy.astype(np.float32))
        path = meta['checkpoint']
        report['runs'][label] = dict(source=source, epoch=meta.get('epoch'), cursor=meta.get('cursor'),
                                     method=meta['method'], readout=meta['readout'],
                                     source_mtime=os.path.getmtime(path) if os.path.exists(path) else None,
                                     visible_fraction=float((normalized > args.threshold).mean()))
        for view in VIEWS:
            report['files'] += save_panel(
                lambda axis: draw_mip(axis, normalized, view, extent, args.window, args.threshold),
                panels/f'{label}_mip_{view[0]}', panel_width(view, args.window))
        print(f"{label}: {meta['method']} epoch {meta.get('epoch')} cursor {meta.get('cursor')} "
              f"visible {report['runs'][label]['visible_fraction']:.4f}", flush=True)
    triangles = load_stl_vertices(args.mesh_dir/'camry_xv20_region_local.stl').reshape(-1, 3, 3)
    points = sample_surface_points(triangles, args.mesh_samples, np.random.default_rng(1))
    for view in VIEWS:
        depth, pixel = mesh_depth(points, view, args.window, args.mesh_px)
        shading = shade(depth, pixel)
        report['files'] += save_panel(lambda axis: draw_mesh(axis, shading, view, args.window),
                                      panels/f'mesh_{view[0]}', panel_width(view, args.window))
    report['files'] += save_colorbar(panels/'colorbar', args.threshold)
    (args.output_dir/'manifest.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    print(f"wrote {len(report['files'])} files to {panels}")


if __name__ == '__main__':
    main()
