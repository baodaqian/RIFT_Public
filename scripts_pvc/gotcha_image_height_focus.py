#!/usr/bin/env python3
"""Height focus of a 3D energy image on the geometry scorer's lattice (reviewer check, docs/RIFT_GOTCHA_Tune.md B59).

Inter-pass phase is what resolves height, so a better per-pass phase correction should sharpen each column's vertical
profile. Measured per (x, y) column, for the brightest columns by total energy: the energy-weighted z standard
deviation inside the column, and the share of the column's energy within +-WINDOW of its peak. Both are
column-energy-weighted averages. Also printed for comparison: A56's measure, the energy-weighted z spread of the
brightest 1% of voxels, which mixes in the car's own height across columns.

    python scripts_pvc/gotcha_image_height_focus.py --field raw_all=OUT/raw_all.npz:data_energy \\
        --field carphase_all=OUT/carphase_all.npz:data_energy [--output focus.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.render_b787_vs_stl import trilinear_sample_centers  # noqa: E402


def weighted_std(values, weights):
    mean = (weights * values).sum(-1) / weights.sum(-1)
    return np.sqrt((weights * (values - mean[..., None]) ** 2).sum(-1) / weights.sum(-1))


def focus(energy, extent, fractions, window):
    grid = energy.shape[0]
    z = trilinear_sample_centers(extent, grid, grid)
    flat = energy.ravel()
    top = np.argsort(flat)[::-1][:max(1, int(round(0.01 * flat.size)))]
    zi = np.unravel_index(top, energy.shape)[2]
    row = dict(top1pct_voxel_z_std_m=float(weighted_std(z[zi], flat[top])))
    column = energy.sum(2).ravel()
    order = np.argsort(column)[::-1]
    for fraction in fractions:
        pick = order[:max(1, int(fraction * column.size))]
        profiles = energy.reshape(-1, grid)[pick]
        weight = profiles.sum(1)
        spread = weighted_std(np.broadcast_to(z, profiles.shape), profiles)
        near = np.abs(z[None, :] - z[profiles.argmax(1)][:, None]) <= window
        share = (profiles * near).sum(1) / weight
        key = f'top{fraction * 100:g}pct_columns'
        row[key] = dict(columns=int(len(pick)), z_std_m=float(np.average(spread, weights=weight)),
                        peak_window_share=float(np.average(share, weights=weight)))
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--field', action='append', required=True, help='LABEL=NPZ:KEY')
    p.add_argument('--fractions', type=float, nargs='+', default=[0.01, 0.05, 0.2])
    p.add_argument('--window', type=float, default=0.25, help='half-width (m) of the peak window')
    p.add_argument('--output', type=Path)
    args = p.parse_args(argv)
    rows = {}
    for item in args.field:
        label, spec = item.split('=', 1)
        path, key = spec.rsplit(':', 1)
        data = np.load(path)
        meta = json.loads(str(data['meta']))
        energy = np.asarray(data[key], dtype=np.float64)
        rows[label] = dict(focus(energy, float(meta['region']['half_extent_m']), args.fractions, args.window),
                           source=path, key=key, shard_root=meta.get('shard_root'), pulses=meta.get('pulses'))
        r = rows[label]
        print(f"{label:22s} top-1% voxels z-std {r['top1pct_voxel_z_std_m']:.3f} m | " + ' | '.join(
            f"{k.replace('_columns', '')} cols z-std {v['z_std_m']:.3f} m, peak share {v['peak_window_share']:.3f}"
            for k, v in r.items() if k.endswith('_columns')), flush=True)
    if args.output:
        args.output.write_text(json.dumps(dict(schema='gotcha_image_height_focus_v1', window_m=args.window,
                                               fractions=args.fractions, fields=rows), indent=1))


if __name__ == '__main__':
    main()
