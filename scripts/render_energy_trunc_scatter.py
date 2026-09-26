#!/usr/bin/env python
"""Energy-truncated 3D scatter of a dense trilinear-interpolated scene.

Reproduces the sphere round's `*_energy_trunc_scatter.png` style (2026-07-14)
for any grid/grid_sh checkpoint: upsample the isotropic |w| field trilinearly
(Plenoxel-style), then show three panels that drop the bottom 0.1% / 1% / 10%
of total |w|^2 energy, with a quarter wedge cut away toward the camera so the
interior is visible (the shells should be hollow). Point color AND alpha both
follow |w|, so dense low-|w| speckle reads as a dark haze while genuine
structure stays saturated.

Usage:
    python scripts/render_energy_trunc_scatter.py \\
        --checkpoint training_checkpoints/pec_cube_grid_g32/checkpoint_final.pth.tar \\
        --extent 2.0 --upsample 4
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from render_dense_scene import load_grid_field, trilinear_upsample  # noqa: E402

DROP_FRACS = (0.001, 0.01, 0.10)


def render(dense, extent, out_path, elev=25, azim=-60, max_pts=3_000_000):
    D = dense.shape[0]
    coords_1d = np.linspace(-extent, extent, D, dtype=np.float32)
    mag = dense.numpy().astype(np.float32).ravel()
    n_total = mag.size

    order = np.argsort(-mag)
    e = mag[order].astype(np.float64) ** 2
    cum = np.cumsum(e)
    cum /= cum[-1]

    vmax = float(mag.max())
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = cm.inferno

    fig = plt.figure(figsize=(21, 6.5))
    for k, drop in enumerate(DROP_FRACS):
        n_keep = int(np.searchsorted(cum, 1.0 - drop)) + 1
        idx = order[:n_keep]
        iz = idx % D
        iy = (idx // D) % D
        ix = idx // (D * D)
        x, y, z = coords_1d[ix], coords_1d[iy], coords_1d[iz]
        v = mag[idx]

        # quarter wedge toward the default camera (azim=-60 looks from +x,-y)
        wedge = (x > 0) & (y < 0)
        x, y, z, v = x[~wedge], y[~wedge], z[~wedge], v[~wedge]
        if v.size > max_pts:  # keep matplotlib tractable at high upsampling
            sub = np.random.default_rng(0).choice(v.size, max_pts, replace=False)
            x, y, z, v = x[sub], y[sub], z[sub], v[sub]

        draw = np.argsort(v)  # brightest last
        x, y, z, v = x[draw], y[draw], z[draw], v[draw]
        rgba = cmap(norm(v))
        rgba[:, 3] = np.clip(v / vmax, 0.03, 0.9)

        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.scatter(x, y, z, c=rgba, s=1.2, linewidths=0, depthshade=False)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(-extent, extent); ax.set_ylim(-extent, extent); ax.set_zlim(-extent, extent)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.set_title(f"drop bottom {100*drop:g}% of $|w|^2$ energy\n"
                     f"keep n={n_keep:,} of {n_total:,} samples ({100*n_keep/n_total:.1f}%); "
                     f"quarter wedge cut away for view", fontsize=10)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, ax=ax, shrink=0.55, label="|w| (trilinear-interp.)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"Saved energy-truncation scatter to {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--upsample", type=int, default=4)
    p.add_argument("--extent", type=float, default=None,
                    help="scene extent override; NOTE checkpoints do not store extent, "
                         "the loader falls back to 1.5 -- pass 2.0 for the cube/tetra runs")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    scene_repr, field, extent = load_grid_field(args.checkpoint)
    if args.extent is not None:
        extent = args.extent
    print(f"Loaded scene_repr={scene_repr!r}, grid {tuple(field.shape)}, extent=+/-{extent}m")
    dense = trilinear_upsample(field, args.upsample)
    print(f"Upsampled to {tuple(dense.shape)}")

    out = args.out or os.path.join(
        "figures", os.path.splitext(os.path.basename(args.checkpoint))[0] + "_energy_trunc_scatter.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    render(dense, extent, out)


if __name__ == "__main__":
    main()
