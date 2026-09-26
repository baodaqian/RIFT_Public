#!/usr/bin/env python
"""Plenoxel-style dense rendering of an already-trained grid_sh checkpoint.

Context: scripts/check_range_resolution.py confirmed the Goal-1a
reconstructions' diffuse radial blob (~0.6-0.7m FWHM around r=1.0m) is
consistent with this dataset's native ~1m range resolution (149.9MHz
bandwidth), not a representation-capacity shortfall. Before training
anything new, this script asks: what does the CURRENT trained checkpoint
look like if we render it as a dense continuous field (trilinear
interpolation of the coarse 24^3 grid, a la Plenoxels) instead of a sparse
scatter of raw voxel centers?

This is post-hoc visualization only -- no retraining, no change to the
learned weights. It upsamples the isotropic (l=0, direction-averaged) SH
coefficient field via torch.nn.functional.interpolate(mode="trilinear"),
then renders orthogonal mid-slices, max-intensity projections, and a dense
point-cloud scatter of the upsampled field.

Usage:
    python scripts/render_dense_scene.py \\
        --checkpoint training_checkpoints/pec_sphere_recon_gridsh6_g24/checkpoint_best.pth.tar \\
        --upsample 4 --sphere-radius 1.0
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from visualize_scene_checkpoint import _infer_granularity_and_degree, _infer_scene_repr  # noqa: E402


def load_grid_field(checkpoint_path, device="cpu"):
    """Loads a grid/grid_sh checkpoint's isotropic magnitude as a dense
    [G,G,G] torch tensor plus the scene's extent. point_sh is not supported
    (no regular grid to interpolate)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    scene_repr = checkpoint.get("scene_repr") or _infer_scene_repr(state_dict)
    if scene_repr not in ("grid", "grid_sh"):
        raise ValueError(f"scene_repr={scene_repr!r} has no regular grid to trilinear-interpolate "
                          f"(point_sh scatterers are unstructured)")

    granularity, _ = _infer_granularity_and_degree(state_dict)
    w_re, w_im = state_dict["w_re"], state_dict["w_im"]
    if w_re.ndim == 4:  # grid_sh: [G,G,G,n_basis] -> isotropic/DC term
        w_re, w_im = w_re[..., 0], w_im[..., 0]
    magnitude = torch.sqrt(w_re ** 2 + w_im ** 2)  # [G,G,G]
    extent = checkpoint.get("extent", 1.5)  # train.py's Goal-1a runs used --extent 1.5
    return scene_repr, magnitude, float(extent)


def trilinear_upsample(field_ggg, factor):
    """[G,G,G] -> [G*factor,G*factor,G*factor] via trilinear interpolation,
    the Plenoxel recipe for turning a coarse voxel grid into a dense
    continuous-looking field."""
    x = field_ggg[None, None]  # [1,1,G,G,G]
    D = field_ggg.shape[0] * factor
    dense = F.interpolate(x, size=(D, D, D), mode="trilinear", align_corners=True)
    return dense[0, 0]


# --- ground-truth overlays -------------------------------------------------
# Tetra vertices recovered from the npz itself (validate_pec_tetrahedron_coherence
# Stage A support-function clustering, 2026-07-16); all |v| = 1.500m.
TETRA_VERTS = np.array([
    [-0.685, +1.225, -0.530],
    [+1.430, -0.022, -0.453],
    [+0.059, -0.028, +1.499],
    [-0.709, -1.212, -0.527],
])
TETRA_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def _cube_verts(half):
    return np.array([[(-1) ** (i & 1) * half, (-1) ** ((i >> 1) & 1) * half,
                      (-1) ** ((i >> 2) & 1) * half] for i in range(8)])


CUBE_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3),
              (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def _poly_cross_section(verts, edges, axis):
    """Closed 2D outline of a convex polyhedron's mid-plane (coord[axis]=0)
    cross-section, in the slice's in-plane coordinates."""
    pts = []
    for i, j in edges:
        a, b = verts[i], verts[j]
        da, db = a[axis], b[axis]
        if da * db < 0:
            pts.append(a + (da / (da - db)) * (b - a))
    if len(pts) < 3:
        return None
    keep = [k for k in range(3) if k != axis]
    p2 = np.array(pts)[:, keep]
    c = p2.mean(axis=0)
    p2 = p2[np.argsort(np.arctan2(p2[:, 1] - c[1], p2[:, 0] - c[0]))]
    return np.vstack([p2, p2[:1]])


def _poly_silhouette(verts, axis):
    """Closed 2D convex-hull outline of the vertices projected along axis."""
    from scipy.spatial import ConvexHull
    keep = [k for k in range(3) if k != axis]
    p2 = verts[:, keep]
    hull = ConvexHull(p2)
    p2 = p2[hull.vertices]
    return np.vstack([p2, p2[:1]])


def make_target(name, sphere_radius=1.0, cube_half=1.0):
    """Ground-truth overlay spec: cross_section(axis)/silhouette(axis) return a
    closed (N,2) outline in the remaining two coords, edges3d() the 3D frame."""
    if name == "sphere":
        t = np.linspace(0, 2 * np.pi, 128)
        circ = sphere_radius * np.stack([np.cos(t), np.sin(t)], axis=-1)
        return {"label": f"r={sphere_radius}m sphere",
                "cross_section": lambda axis: circ,
                "silhouette": lambda axis: circ,
                "verts": None, "edges": None}
    if name == "cube":
        v = _cube_verts(cube_half)
        return {"label": f"{2 * cube_half:g}m cube",
                "cross_section": lambda axis: _poly_cross_section(v, CUBE_EDGES, axis),
                "silhouette": lambda axis: _poly_silhouette(v, axis),
                "verts": v, "edges": CUBE_EDGES}
    if name == "tetra":
        v = TETRA_VERTS
        return {"label": "tetrahedron (data-recovered orientation)",
                "cross_section": lambda axis: _poly_cross_section(v, TETRA_EDGES, axis),
                "silhouette": lambda axis: _poly_silhouette(v, axis),
                "verts": v, "edges": TETRA_EDGES}
    raise ValueError(f"unknown target {name!r}")


def _draw_outline(ax, outline):
    if outline is not None:
        ax.plot(outline[:, 0], outline[:, 1], color="cyan", linestyle="--", linewidth=1.2)


def plot_slices(dense, extent, target, out_path):
    D = dense.shape[0]
    mid = D // 2
    arr = dense.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    slices = [
        ("z=0 (xy plane)", arr[:, :, mid].T, "x (m)", "y (m)", 2),
        ("y=0 (xz plane)", arr[:, mid, :].T, "x (m)", "z (m)", 1),
        ("x=0 (yz plane)", arr[mid, :, :].T, "y (m)", "z (m)", 0),
    ]
    for ax, (title, sl, xl, yl, axis) in zip(axes, slices):
        im = ax.imshow(sl, origin="lower", extent=[-extent, extent, -extent, extent],
                        cmap="inferno", aspect="equal")
        _draw_outline(ax, target["cross_section"](axis))
        ax.set_title(f"{title} (mid-slice)")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        fig.colorbar(im, ax=ax, shrink=0.7, label="|w| (isotropic, trilinear-interp.)")
    fig.suptitle(f"Dense trilinear-interpolated scene, {D}^3 samples "
                 f"(cyan = true {target['label']} cross-section)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved slice plot to {out_path}")


def plot_mip(dense, extent, target, out_path):
    arr = dense.numpy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    mips = [
        ("MIP along z", arr.max(axis=2).T, "x (m)", "y (m)", 2),
        ("MIP along y", arr.max(axis=1).T, "x (m)", "z (m)", 1),
        ("MIP along x", arr.max(axis=0).T, "y (m)", "z (m)", 0),
    ]
    for ax, (title, sl, xl, yl, axis) in zip(axes, mips):
        im = ax.imshow(sl, origin="lower", extent=[-extent, extent, -extent, extent],
                        cmap="inferno", aspect="equal")
        _draw_outline(ax, target["silhouette"](axis))
        ax.set_title(title)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        fig.colorbar(im, ax=ax, shrink=0.7, label="max |w| along ray")
    fig.suptitle(f"Max-intensity projections (cyan = true {target['label']} silhouette)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved MIP plot to {out_path}")


def plot_dense_scatter(dense, extent, target, out_path, top_frac=0.02):
    D = dense.shape[0]
    coords_1d = np.linspace(-extent, extent, D)
    xx, yy, zz = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    mag = dense.numpy().ravel()
    pos = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

    n_top = max(1, int(len(mag) * top_frac))
    top_idx = np.argsort(-mag)[:n_top]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(pos[top_idx, 0], pos[top_idx, 1], pos[top_idx, 2],
                     c=mag[top_idx], cmap="inferno", s=2, alpha=0.4)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="|w| (trilinear-interp.)")

    if target["verts"] is None:  # sphere: wireframe
        r = target["cross_section"](2)[0, 0]
        u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
        ax.plot_wireframe(r * np.cos(u) * np.sin(v), r * np.sin(u) * np.sin(v),
                           r * np.cos(v), color="gray", alpha=0.15, linewidth=0.5)
    else:  # polyhedron: true edge frame
        for i, j in target["edges"]:
            seg = target["verts"][[i, j]]
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="gray", alpha=0.6, linewidth=1.0)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(f"Dense scene: top {top_frac:.0%} of {D}^3 interpolated samples (n={n_top}) "
                 f"vs. true {target['label']}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved dense scatter to {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--upsample", type=int, default=4, help="linear upsampling factor per axis")
    p.add_argument("--target", choices=["sphere", "cube", "tetra"], default="sphere",
                    help="ground-truth overlay: true cross-section/silhouette outlines "
                         "(cube = 2*cube-half axis-aligned; tetra = data-recovered orientation)")
    p.add_argument("--sphere-radius", type=float, default=1.0)
    p.add_argument("--cube-half", type=float, default=1.0, help="cube half-side in meters")
    p.add_argument("--extent", type=float, default=None,
                    help="override scene extent in meters (default: read from checkpoint, "
                         "fallback 1.5 -- the Goal-1a sweep's --extent value)")
    p.add_argument("--out-prefix", default=None,
                    help="output path prefix (default: figures/<checkpoint-stem>_dense)")
    args = p.parse_args()

    scene_repr, field, extent = load_grid_field(args.checkpoint)
    if args.extent is not None:
        extent = args.extent
    print(f"Loaded scene_repr={scene_repr!r}, coarse grid {tuple(field.shape)}, extent=+/-{extent}m")

    dense = trilinear_upsample(field, args.upsample)
    print(f"Trilinear-upsampled to {tuple(dense.shape)} ({args.upsample}x per axis)")

    target = make_target(args.target, args.sphere_radius, args.cube_half)

    out_prefix = args.out_prefix or os.path.join(
        "figures", os.path.splitext(os.path.basename(args.checkpoint))[0] + "_dense")
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    plot_slices(dense, extent, target, out_prefix + "_slices.png")
    plot_mip(dense, extent, target, out_prefix + "_mip.png")
    plot_dense_scatter(dense, extent, target, out_prefix + "_scatter.png")


if __name__ == "__main__":
    main()
