#!/usr/bin/env python3
"""Max-intensity views of an adaptive-RIFT GOTCHA checkpoint over the registered Camry mesh (PVC, CPU).

The scene is read like ``eval_gotcha_geometry_pvc.py`` reads it: the conservative CIC point-SH
energy on a ``--grid``^3 lattice over the region cube (``_point_sh_energy_field``), then upsampled
``--upsample`` x by Plenoxel-style trilinear interpolation of the energy (``trilinear_upsample``,
display only) and shown as sqrt(energy), min-max normalized, above ``--threshold``. Panels:
max-intensity projections along -z (top), -y (side) and -x (front), with the mesh vertices
of ``--mesh-dir`` (region frame) overlaid. Several checkpoints make several rows.

    python scripts_pvc/render_gotcha_rift_views_pvc.py --mesh-dir data/meshes/camry_xv20_data_frame_box_v2 \\
        --run T5b_ep10=PATH.pt --output views.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.render_b787_vs_stl import _point_sh_energy_field, load_stl_vertices, trilinear_upsample  # noqa: E402

PLANES = (('top (x-y)', 2, 0, 1, 'x front [m]', 'y left [m]'),
          ('side (x-z)', 1, 0, 2, 'x front [m]', 'z up [m]'),
          ('front (y-z)', 0, 1, 2, 'y left [m]', 'z up [m]'))


def field(checkpoint, extent, grid, upsample):
    if ':' in checkpoint and checkpoint.split(':', 1)[0].endswith('.npz'):
        # PATH.npz:KEY, a precomputed energy on the scorer's lattice (gotcha_coherent_readout_pvc.py).
        path, key = checkpoint.split(':', 1)
        data = np.load(path)
        meta = json.loads(str(data['meta']))
        energy, ck, active = np.asarray(data[key], dtype=np.float64), dict(epoch=meta.get('epoch')), 0
    else:
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        state = {k[len('hh.field.'):]: v for k, v in ck['model_state_dict'].items() if k.startswith('hh.field.')}
        energy, _ = _point_sh_energy_field(state, extent, grid)
        active = int(state['active_mask'].sum())
    dense = trilinear_upsample(energy, upsample)
    magnitude = np.sqrt(np.clip(dense, 0, None))
    normalized = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-30)
    return ck, normalized, active


def main(argv=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mesh-dir', type=Path, default=Path('data/meshes/camry_xv20_data_frame_box_v2'))
    p.add_argument('--run', action='append', required=True, help='LABEL=CHECKPOINT')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--grid', type=int, default=48)
    p.add_argument('--upsample', type=int, default=4)
    p.add_argument('--threshold', type=float, default=0.2)
    p.add_argument('--crop', type=float, nargs=3, default=[3.0, 1.6, 1.35], help='half-widths shown (x, y, z)')
    args = p.parse_args(argv)
    manifest = json.loads((args.mesh_dir/'manifest.json').read_text())
    extent = float(manifest['region']['half_extent_m'])
    verts = load_stl_vertices(args.mesh_dir/'camry_xv20_region_local.stl').reshape(-1, 3)
    verts = verts[np.random.default_rng(0).choice(len(verts), min(len(verts), 60000), replace=False)]
    rows = [item.split('=', 1) for item in args.run]
    figure, axes = plt.subplots(len(rows), 3, figsize=(15, 3.2 * len(rows)), squeeze=False,
                                gridspec_kw=dict(width_ratios=[2 * args.crop[0], 2 * args.crop[0], 2 * args.crop[1]]))
    cmap = plt.get_cmap('inferno').copy()
    cmap.set_bad('#050505')
    for r, (label, checkpoint) in enumerate(rows):
        ck, normalized, active = field(checkpoint, extent, args.grid, args.upsample)
        n = normalized.shape[0]
        centers = np.linspace(-extent + extent / (args.grid), extent - extent / (args.grid), n)
        for c, (name, mip, h, v, hl, vl) in enumerate(PLANES):
            axis = axes[r][c]
            projection = normalized.max(axis=mip)
            remaining = [i for i in range(3) if i != mip]
            image = projection.T if remaining == [h, v] else projection
            axis.imshow(np.ma.masked_less_equal(image, args.threshold), origin='lower', cmap=cmap,
                        extent=[centers[0], centers[-1]] * 2, vmin=args.threshold, vmax=1.0,
                        interpolation='bilinear', aspect='equal')
            axis.scatter(verts[:, h], verts[:, v], s=0.2, c='#64CCC9', alpha=0.15, linewidths=0, rasterized=True)
            axis.set_xlim(-args.crop[h], args.crop[h])
            axis.set_ylim(-args.crop[v], args.crop[v])
            axis.set_facecolor('#050505')
            axis.set_xlabel(hl, fontsize=8)
            axis.set_ylabel(vl, fontsize=8)
            axis.tick_params(labelsize=7)
            axis.set_title(f'{label}: {name}' + (f'  (epoch {ck.get("epoch")}, {active:,} points)' if c == 0 else ''),
                           fontsize=9)
    figure.suptitle(f'RIFT on GOTCHA Camry HH: max-intensity projections of sqrt(energy) above {args.threshold} '
                    f'({args.grid}^3 CIC readout, {args.upsample}x trilinear for display); teal = Camry mesh '
                    f'({args.mesh_dir.name})', fontsize=9)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=130)
    print(args.output)


if __name__ == '__main__':
    main()
