#!/usr/bin/env python
"""Visualize/score a trained scene checkpoint's 3D reconstruction.

For the "does this look like the target?" question (EXPERIMENT_MANAGER_HANDOFF.md,
"New ground-truth-geometry dataset..." Goal 1): loads a train.py checkpoint,
reconstructs the scene module from it, and reports how the scatterer
magnitude is distributed in 3D -- in particular whether it clusters near
the expected target radius, plus a static 3D scatter plot of the top
magnitude entries.

grid/grid_sh/point_sh only (checkpoints record `scene_repr`; mlp has no
explicit positions to plot and is skipped). For SH-based scenes ("grid_sh",
"point_sh"), plots the isotropic (l=0, direction-averaged) coefficient --
the same DC term backprojection_init writes and prune() thresholds on --
NOT a view-dependent render; a true specular target's per-point brightness
still varies by viewing angle, so this is a coarse-but-informative summary,
not the full picture.

Usage:
    python scripts/visualize_scene_checkpoint.py \\
        --checkpoint training_checkpoints/pec_sphere_recon_deg6/checkpoint_best.pth.tar \\
        --sphere-radius 1.0
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rift.sparse_scene import AdaptivePointSHScene, SHVoxelGridScene, VoxelGridScene


def _infer_granularity_and_degree(state_dict):
    w_re = state_dict["w_re"]
    if w_re.ndim == 4:  # grid_sh: [G,G,G,n_basis]
        granularity = w_re.shape[0]
        n_basis = w_re.shape[-1]
    elif w_re.ndim == 3:  # grid: [G,G,G]
        granularity = w_re.shape[0]
        n_basis = None
    else:  # point_sh: [K, n_basis] handled separately
        raise ValueError(f"Unexpected w_re shape {tuple(w_re.shape)}")
    max_degree = int(round(n_basis ** 0.5)) - 1 if n_basis is not None else None
    return granularity, max_degree


def _infer_scene_repr(state_dict):
    """Fallback for checkpoints saved before train.py recorded scene_repr
    explicitly: point_sh has an 'anchors' buffer; grid_sh's w_re has an
    extra (SH basis) dimension; grid's w_re is a plain [G,G,G] tensor."""
    if "anchors" in state_dict:
        return "point_sh"
    if state_dict["w_re"].ndim == 4:
        return "grid_sh"
    if state_dict["w_re"].ndim == 3:
        return "grid"
    raise ValueError(f"Could not infer scene_repr from w_re shape {tuple(state_dict['w_re'].shape)}")


def load_scene(checkpoint_path, device="cpu", scene_repr_override=None):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    scene_repr = scene_repr_override or checkpoint.get("scene_repr") or _infer_scene_repr(state_dict)

    if scene_repr == "point_sh":
        model = AdaptivePointSHScene.from_state(state_dict, device)
        pos = model.positions().detach()
        w_re, w_im = model.w_re[:, 0], model.w_im[:, 0]
    elif scene_repr == "grid_sh":
        granularity, max_degree = _infer_granularity_and_degree(state_dict)
        model = SHVoxelGridScene(granularity, extent=1.0, device=device, max_degree=max_degree)
        model.load_state_dict(state_dict)
        pos = model.grid_positions.reshape(-1, 3)
        w_re, w_im = model.w_re[..., 0].reshape(-1), model.w_im[..., 0].reshape(-1)
    elif scene_repr == "grid":
        granularity, _ = _infer_granularity_and_degree(state_dict)
        model = VoxelGridScene(granularity, extent=1.0, device=device)
        model.load_state_dict(state_dict)
        pos = model.grid_positions.reshape(-1, 3)
        w_re, w_im = model.w_re.reshape(-1), model.w_im.reshape(-1)
    else:
        raise ValueError(f"Unsupported/missing scene_repr={scene_repr!r} in checkpoint "
                          f"(this script handles grid/grid_sh/point_sh only)")

    magnitude = torch.sqrt(w_re ** 2 + w_im ** 2)
    return scene_repr, pos.cpu().numpy(), magnitude.detach().cpu().numpy()


def report_radial_clustering(pos, magnitude, sphere_radius, band_halfwidth):
    r = np.linalg.norm(pos, axis=-1)
    power = magnitude ** 2
    total_power = power.sum()
    in_band = np.abs(r - sphere_radius) <= band_halfwidth
    band_frac = power[in_band].sum() / total_power if total_power > 0 else float("nan")

    weighted_mean_r = (power * r).sum() / total_power if total_power > 0 else float("nan")
    weighted_std_r = np.sqrt((power * (r - weighted_mean_r) ** 2).sum() / total_power) if total_power > 0 else float("nan")

    # null-hypothesis comparison: fraction of the SCENE BOX volume (not power)
    # covered by the same radial band, i.e. what band_frac would be if power
    # were uniformly spread over the box irrespective of the target.
    r_max = r.max()
    volume_frac = (min(sphere_radius + band_halfwidth, r_max) ** 3
                   - max(sphere_radius - band_halfwidth, 0) ** 3) / r_max ** 3

    print(f"Weighted mean radius: {weighted_mean_r:.3f}m (target {sphere_radius}m), "
          f"weighted std: {weighted_std_r:.3f}m")
    print(f"Power fraction within +/-{band_halfwidth}m of r={sphere_radius}m: {band_frac:.1%} "
          f"(vs. {volume_frac:.1%} if power were spread uniformly over the scene box -- "
          f"a reconstruction is only informative if band_frac clearly exceeds this)")
    return weighted_mean_r, weighted_std_r, band_frac


def plot_top_points(pos, magnitude, sphere_radius, out_path, top_frac=0.05):
    n_top = max(1, int(len(magnitude) * top_frac))
    top_idx = np.argsort(-magnitude)[:n_top]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(pos[top_idx, 0], pos[top_idx, 1], pos[top_idx, 2],
                     c=magnitude[top_idx], cmap="inferno", s=8)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="|w| (isotropic/DC term)")

    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    xs = sphere_radius * np.cos(u) * np.sin(v)
    ys = sphere_radius * np.sin(u) * np.sin(v)
    zs = sphere_radius * np.cos(v)
    ax.plot_wireframe(xs, ys, zs, color="gray", alpha=0.15, linewidth=0.5)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(f"Top {top_frac:.0%} magnitude scatterers (n={n_top}) vs. r={sphere_radius}m reference sphere")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sphere-radius", type=float, default=1.0,
                   help="Known/expected target radius in meters, for the reference wireframe and "
                        "radial-clustering metric (pec_sphere npz: 1.0)")
    p.add_argument("--band-halfwidth", type=float, default=0.3,
                   help="Half-width (m) of the radial band around --sphere-radius used for the "
                        "power-fraction clustering metric")
    p.add_argument("--top-frac", type=float, default=0.05,
                   help="Fraction of highest-|w| entries to plot")
    p.add_argument("--output", default=None,
                   help="Output PNG path (default: alongside the checkpoint)")
    p.add_argument("--scene-repr", default=None, choices=["grid", "grid_sh", "point_sh"],
                   help="Override scene_repr for checkpoints saved before train.py recorded it "
                        "(inferred from state_dict shape otherwise)")
    args = p.parse_args()

    scene_repr, pos, magnitude = load_scene(args.checkpoint, scene_repr_override=args.scene_repr)
    print(f"Loaded scene_repr={scene_repr!r}, {pos.shape[0]} scatterers "
          f"(max |w| = {magnitude.max():.3e}, mean |w| = {magnitude.mean():.3e})")

    report_radial_clustering(pos, magnitude, args.sphere_radius, args.band_halfwidth)

    out_path = args.output or os.path.splitext(args.checkpoint)[0] + "_scatter.png"
    plot_top_points(pos, magnitude, args.sphere_radius, out_path, args.top_frac)


if __name__ == "__main__":
    main()
