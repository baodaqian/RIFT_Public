#!/usr/bin/env python
"""How much of a B787 reconstruction's angular energy sits ON the airframe?

The standing measurement is that the good (deg-6 baseline) fit keeps ~62% of
its energy OFF the airframe and breaks without it -- see CLAUDE.md's extent
A/B and RIFT_RECYCLING_PLAN.md T2, which argues a scatterer field with no
visibility term can only synthesize a shadowed view by placing canceling
contributions off-target. Round 6 adds the visibility term, so the falsifiable
prediction is that the occlusion arms move energy back onto the airframe.

Metric: per-voxel rotation-invariant energy sum|c_lm|^2 (same quantity every
other eval uses), weighted by whether the voxel center lies within `--tol` of
the STL surface, in the scene frame the synthesis used. Reported as an energy
fraction and as a concentration (energy fraction / volume fraction), so it is
comparable across runs despite the (gain, scene) gauge being free.

    module load anaconda3 && conda activate RIFT
    python scripts/eval_b787_energy_on_airframe.py \
        --checkpoints training_checkpoints/b787_r4_capdeg3 ... --tol 0.005
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.render_b787_vs_stl import (  # noqa: E402
    load_energy_field, load_stl_vertices, stl_into_scene_frame,
)


def voxel_centers(extent, g):
    """Match rift.encoding.generate_dynamic_grid: centers at interval midpoints
    of linspace(-extent, extent, G+1), cartesian order x outer / y mid / z in."""
    edges = np.linspace(-extent, extent, g + 1)
    c = 0.5 * (edges[:-1] + edges[1:])
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="run directories (checkpoint_best.pth.tar is read) or file paths")
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--npz-path", default="data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere2k.npz")
    p.add_argument("--stl", default="data/B787.stl")
    p.add_argument("--extent", type=float, default=0.15)
    p.add_argument("--tol", type=float, default=0.005,
                   help="distance from the STL surface counted as ON the airframe [m]; "
                        "default 5 mm is under one g48 voxel pitch (6.25 mm)")
    args = p.parse_args()

    meta = json.loads(str(np.load(args.npz_path, allow_pickle=True,
                                  mmap_mode="r")["metadata_json"]))
    verts = stl_into_scene_frame(load_stl_vertices(args.stl), meta)
    tree = cKDTree(np.unique(verts, axis=0))

    labels = args.labels or [os.path.basename(c.rstrip("/")) for c in args.checkpoints]
    print(f"{'run':32s} {'ep':>4s} {'E on airframe':>14s} {'vol frac':>9s} {'conc':>7s}")
    for path, label in zip(args.checkpoints, labels):
        if os.path.isdir(path):
            path = os.path.join(path, "checkpoint_best.pth.tar")
        energy, g, epoch = load_energy_field(path)
        c = voxel_centers(args.extent, g)
        pts = np.stack(np.meshgrid(c, c, c, indexing="ij"), axis=-1).reshape(-1, 3)
        d, _ = tree.query(pts, workers=-1)
        on = d <= args.tol
        e = energy.reshape(-1)
        e_frac = e[on].sum() / e.sum()
        v_frac = on.mean()
        print(f"{label:32s} {epoch:4d} {100 * e_frac:13.1f}% {100 * v_frac:8.1f}% "
              f"{e_frac / v_frac:6.1f}x")


if __name__ == "__main__":
    main()
