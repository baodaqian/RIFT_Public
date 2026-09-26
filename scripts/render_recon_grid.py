#!/usr/bin/env python
"""Side-by-side dense (trilinear-upsampled) reconstructions across runs.

One column per checkpoint, one row per projection plane, STL silhouette
overlaid in every panel -- the compact form of render_b787_vs_stl for
comparing a whole sweep on a single slide. Column subtitles carry completion
status and best validation rel-MSE.

    module load anaconda3 && conda activate RIFT
    python scripts/render_recon_grid.py --set ladder --out figures/b787_recon/ladder_grid.png
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.render_b787_vs_stl import (  # noqa: E402
    PLANES, load_energy_field, load_stl_vertices, stl_into_scene_frame, trilinear_upsample,
)

CKPT = "training_checkpoints/{name}/checkpoint_best.pth.tar"

# (checkpoint dir, column title, subtitle)
SETS = {
    "ladder": [
        ("b787_r5_deg0",                  "SH degree 0",           "ep 150/150 (done) · val 28.5%"),
        ("b787_r5_deg1",                  "SH degree 1",           "ep 150/150 (done) · val 26.0%"),
        ("b787_r5_deg2",                  "SH degree 2",           "ep 150/150 (done) · val 25.5%"),
        ("b787_r4_capdeg3",               "SH degree 3 (cap)",     "ep 150/150 (done) · val 26.1%"),
        ("b787_sphere2k_gridsh6_g48_n1800", "SH degree 6 (baseline)", "ep 150/150 (done) · val 38.5%"),
    ],
    "prune": [
        ("b787_r5_prune_target20k", "target 20 000 voxels", "ep 150/150 (done) · val 25.3%"),
        ("b787_r5_prune_target4k",  "target 4 000 voxels",  "best ep 80 · val 27.2% → 49.7% final"),
        ("b787_r5_prune_mass",      "mass 0.2% / check",    "best ep 60 · val 29.1% → 49.7% final"),
        ("b787_r4_deg6_prune",      "degree 6 + prune",     "ep 147/150 (stalled) · val 35.9%"),
    ],
    # Round 6 final audit (2026-08-09): all seven runs completed 150 epochs.
    "round6": [
        ("b787_r4_capdeg3",     "no occlusion (control)",  "complete · val 26.0992%"),
        ("b787_r6_occ_z0p1",    "occ · ζ frozen 0.1",      "complete · val 66.6597%"),
        ("b787_r6_occ_z1p0",    "occ · ζ frozen 1.0",      "complete · val 100.0005%"),
        ("b787_r6_occ_learn",   "occ · learned, energy",   "complete · val 25.2198%"),
        ("b787_r6_occ_dc",      "occ · learned, DC",       "complete · val 25.0540%"),
        ("b787_r6_magmix",      "magnitude λ=2",           "complete · val 32.4232%"),
        ("b787_r6_magwarm",     "magnitude + warm-up",     "complete · val 34.2174%"),
    ],
    "round8": [
        ("b787_r8o_legacy_s42", "legacy", "scene lr 3e−3 · eps 1e−8 · val 29.7036%"),
        ("b787_r8o_e15_lr3em6_s42", "lr 3e−6", "scene eps 1e−15 · val 28.0004%"),
        ("b787_r8o_e15_lr3em5_s42", "lr 3e−5", "scene eps 1e−15 · val 28.4328%"),
        ("b787_r8o_e15_lr3em4_s42", "lr 3e−4", "scene eps 1e−15 · val 28.7061%"),
        ("b787_r8o_e15_lr3em3_s42", "lr 3e−3", "scene eps 1e−15 · val 29.0143%"),
    ],
    "current_best": [
        ("b787_r5_prune_target20k", "Round 5 best · target 20k", "complete · val 25.3216%"),
        ("b787_r6_occ_dc", "Round 6 best · learned DC", "complete · val 25.0540%"),
    ],
    "method_comparison": [
        ("b787_r6_occ_dc", "RIFT · Round 6 best", "SH Plenoxel · trilinear readout"),
        ("b787_r5_deg0", "SpINR-style baseline", "isotropic voxel INR · same readout"),
        ("b787_radar_fields_released", "Radar Fields baseline", "occupancy field · same readout"),
        (None, "Sugavanam–Ertin baseline", "Stage 2 reconstruction unavailable"),
    ],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", choices=sorted(SETS), default="ladder")
    p.add_argument("--npz-path", default="data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere2k.npz")
    p.add_argument("--stl", default="data/B787.stl")
    p.add_argument("--checkpoint-root", default="training_checkpoints",
                   help="root containing one checkpoint directory per selected run")
    p.add_argument("--extent", type=float, default=0.15)
    p.add_argument("--upsample", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--pmin-pct", type=float, default=88.0)
    p.add_argument("--planes", default="0,2", help="comma-separated indices into PLANES")
    p.add_argument("--lim", type=float, default=0.075)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cols = SETS[args.set]
    plane_idx = [int(i) for i in args.planes.split(",")]

    meta = json.loads(str(np.load(args.npz_path, allow_pickle=True, mmap_mode="r")["metadata_json"]))
    verts = stl_into_scene_frame(load_stl_vertices(args.stl), meta)
    idx = np.random.default_rng(0).choice(verts.shape[0], min(12000, verts.shape[0]), replace=False)
    verts = verts[idx]

    n_r, n_c = len(plane_idx), len(cols)
    fig, axes = plt.subplots(n_r, n_c, figsize=(2.85 * n_c, 3.15 * n_r), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")

    for ci, (name, title, subtitle) in enumerate(cols):
        if name is None:
            for ri, pi in enumerate(plane_idx):
                pname, _mip_ax, _ha, _va, _hl, _vl = PLANES[pi]
                ax = axes[ri][ci]
                ax.set_facecolor("#f1f0ed")
                ax.text(0.5, 0.56, "PENDING", ha="center", va="center",
                        transform=ax.transAxes, fontsize=15, fontweight="bold", color="#6f6d68")
                ax.text(0.5, 0.43, "Stage 1 preempted at epoch 54\nStage 2 / 3D field absent",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=9.5, color="#52514e", linespacing=1.35)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color("#cbc8c2")
                if ri == 0:
                    ax.set_title(f"{title}\n{subtitle}", fontsize=10.5, color="#0b0b0b", pad=7)
                if ci == 0:
                    ax.set_ylabel(pname, fontsize=11, color="#52514e")
            continue

        checkpoint = os.path.join(args.checkpoint_root, name, "checkpoint_best.pth.tar")
        energy, G, epoch = load_energy_field(checkpoint)
        energy = trilinear_upsample(energy, args.upsample)
        print(f"{name}: ep{epoch}, {G}^3 -> {energy.shape[0]}^3")
        for ri, pi in enumerate(plane_idx):
            pname, mip_ax, ha, va, hl, vl = PLANES[pi]
            ax = axes[ri][ci]
            mip = energy.max(axis=mip_ax)
            remaining = [a for a in range(3) if a != mip_ax]
            img = mip.T if remaining == [ha, va] else mip
            disp = img ** args.gamma
            e = args.extent
            ax.imshow(disp, origin="lower", extent=[-e, e, -e, e],
                      cmap="inferno", aspect="equal", interpolation="bilinear",
                      vmin=np.percentile(disp, args.pmin_pct), vmax=np.percentile(disp, 99.7))
            ax.scatter(verts[:, ha], verts[:, va], s=0.3, c="cyan", alpha=0.10,
                       linewidths=0, rasterized=True)
            ax.set_xlim(-args.lim, args.lim)
            ax.set_ylim(-args.lim, args.lim)
            ax.set_xticks([]); ax.set_yticks([])
            if ci == 0:
                ax.set_ylabel(pname, fontsize=11, color="#52514e")
            if ri == 0:
                ax.set_title(f"{title}\n{subtitle}", fontsize=10.5, color="#0b0b0b", pad=7)

    single_best_view = args.set == "current_best" and len(plane_idx) == 1
    if not single_best_view:
        if args.set == "current_best":
            suptitle = "Best RIFT B787 reconstructions vs STL truth (cyan)"
        elif args.set == "method_comparison":
            suptitle = "B787 learned 3D fields — common trilinear visualization vs STL truth (cyan)"
        else:
            suptitle = "Dense trilinear-interpolated reconstructions (|w| energy MIP) vs STL truth (cyan)"
        fig.suptitle(suptitle, fontsize=12 if args.set == "current_best" else 13,
                     color="#0b0b0b")
    bottom = 0.035 if args.set == "method_comparison" else 0.0
    if args.set == "method_comparison":
        fig.text(0.5, 0.012,
                 "Visualization only: every learned lattice is trilinearly upsampled; display contrast is normalized per method. Metrics remain native-grid.",
                 ha="center", va="bottom", fontsize=8.5, color="#66635e")
    fig.tight_layout(rect=[0, bottom, 1, 1 if single_best_view else 0.955])
    fig.savefig(args.out, dpi=170, facecolor=fig.get_facecolor())
    print("wrote", args.out)


if __name__ == "__main__":
    main()
