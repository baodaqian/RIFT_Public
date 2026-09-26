#!/usr/bin/env python
"""B787 learned 3D fields across EVERY baseline, on one common visualization.

This extends ``render_recon_grid.py --set method_comparison`` to the baselines
that landed after 2026-08-13, which do not all expose a grid_sh checkpoint:

  * RIFT / SpINR-style   ``grid_sh`` lattice        -> Sigma|c_lm|^2 per voxel
  * Radar Fields         released-reference adapter -> same, when it stores a lattice
  * Matched-filter BP    E28 anchor accumulator     -> |coherent adjoint|^2
  * GeRaF                SDF + reflectivity MLPs    -> surface pdf x reflectivity
  * RadarSplat           explicit 3-D Gaussians     -> binned native scene weight
  * Sugavanam-Ertin      Stage 2 never ran          -> labelled placeholder

Every field is read on the SAME 48^3 lattice over the same +/-0.15 m box and
trilinearly upsampled for display only -- never in any training path
(CLAUDE.md). Display contrast is normalized per method, so this figure ranks
SUPPORT, not amplitude: |w| is a Born density plus speckle, not reflectivity.

    module load anaconda3 && conda activate RIFT
    python scripts/render_b787_baseline_fields.py \
        --out figures/b787_recon/baseline_fields_all.png
"""
import argparse
import json
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.render_b787_vs_stl import (  # noqa: E402
    PLANES, load_energy_field, load_stl_vertices, stl_into_scene_frame,
    trilinear_upsample,
)

POWER_ROOT = "/storage/scratch1/1/dbao31/rift_power_baselines_20260813_v2"
GERAF_IMPL = "/storage/scratch1/1/dbao31/rift_power_baselines_impl_20260813_v2"
ANCHOR_NPZ = os.path.join(POWER_ROOT, "a0_e28_anchor/accumulator_latest.npz")
GERAF_CKPT = os.path.join(POWER_ROOT, "g0_geraf/checkpoints/checkpoint_final.pth.tar")
RADARSPLAT_CKPT = os.path.join(
    POWER_ROOT, "s0_radarsplat/checkpoints/checkpoint_final.pth.tar"
)


def load_anchor_energy(path, granularity=48):
    """E28 matched-filter-backprojection anchor -> |coherent adjoint|^2 on the lattice."""
    field = np.load(path)["complex_adjoint"]
    g = int(round(field.size ** (1.0 / 3.0)))
    if g != granularity:
        raise ValueError(f"anchor accumulator is {g}^3, expected {granularity}^3")
    return (np.abs(field.reshape(g, g, g)) ** 2).astype(np.float64), g


def load_geraf_energy(path, granularity=48, extent=0.15, chunk=65536):
    """GeRaF -> logistic surface pdf x reflectivity, sampled on the same lattice.

    GeRaF carries no voxel lattice: it is an SDF network plus a reflectivity
    network. The readout that corresponds to RIFT's per-voxel scattering energy
    is the NeuS surface density (the logistic pdf of the SDF, which peaks on the
    zero level set) modulated by the learned reflectivity.
    """
    if GERAF_IMPL not in sys.path:
        sys.path.insert(0, GERAF_IMPL)
    from rift.geraf import (  # noqa: E402  (implementation tree, not the main repo)
        GeRaFReflectivityNetwork, GeRaFSDFNetwork, logistic_sdf_pdf,
    )

    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg, sd = ck["model_config"], ck["model_state_dict"]

    sdf_net = GeRaFSDFNetwork(
        extent=cfg["extent"], n_levels=cfg["sdf_levels"], hidden_dim=cfg["sdf_hidden_dim"],
        n_layers=cfg["sdf_layers"], skip_layer=cfg["sdf_skip_layer"],
        hidden_activation=cfg["sdf_hidden_activation"],
        softplus_beta=cfg["sdf_softplus_beta"],
        encoding_include_input=cfg["sdf_encoding_include_input"],
        encoding_coordinate_scale=cfg["sdf_encoding_coordinate_scale"],
    )
    refl_net = GeRaFReflectivityNetwork(
        extent=cfg["extent"], n_levels=cfg["reflectivity_levels"],
        hidden_dim=cfg["reflectivity_hidden_dim"], n_layers=cfg["reflectivity_layers"],
        output_activation=cfg["reflectivity_output_activation"],
        softplus_beta=cfg["reflectivity_softplus_beta"],
        encoding_coordinate_scale=cfg["reflectivity_encoding_coordinate_scale"],
    )
    sdf_net.load_state_dict({k[len("sdf_network."):]: v for k, v in sd.items()
                             if k.startswith("sdf_network.")})
    refl_net.load_state_dict({k[len("reflectivity_network."):]: v for k, v in sd.items()
                              if k.startswith("reflectivity_network.")})
    sdf_net.eval(), refl_net.eval()
    inv_s = float(torch.exp(sd["sdf_sharpness.log_inv_s"]))

    # Voxel-center lattice, identical convention to rift.encoding.generate_dynamic_grid.
    edges = np.linspace(-extent, extent, granularity + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    gx, gy, gz = np.meshgrid(centers, centers, centers, indexing="ij")
    pts = torch.from_numpy(np.stack([gx, gy, gz], -1).reshape(-1, 3)).float()

    out = np.empty(pts.shape[0], dtype=np.float64)
    with torch.no_grad():
        for i in range(0, pts.shape[0], chunk):
            block = pts[i:i + chunk]
            density = logistic_sdf_pdf(sdf_net(block).squeeze(-1), inv_s)
            out[i:i + chunk] = (density * refl_net(block).squeeze(-1)).double().numpy()
    return out.reshape(granularity, granularity, granularity), granularity, inv_s


def load_radarsplat_energy(path, granularity=48, extent=0.15):
    """RadarSplat S0 -> native scene weight binned on the common lattice.

    RadarSplat is an explicit point/Gaussian field rather than a voxel field.
    For this visualization-only common readout, each Gaussian centre is assigned
    to the matching 48^3 voxel and weighted by its native DC scene-power factor:

        min(opacity + learned_noise, 1) * clamp(0.5 + C0 * sh0, 1e-6, 1).

    No STL/geometry truth enters the readout.  Centres outside the common
    +/-0.15 m B787 box are intentionally excluded from the displayed crop.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    means = sd["means"].detach().cpu().numpy().astype(np.float64)
    opacity = torch.sigmoid(sd["opacity_logits"]).detach().cpu().numpy()
    noise = torch.sigmoid(sd["noise_probability_logits"]).detach().cpu().numpy()
    dc_reflectance = np.clip(
        0.2820947917738781 * sd["sh0"][:, 0, 0].detach().cpu().numpy() + 0.5,
        1.0e-6,
        1.0,
    )
    weight = np.minimum(opacity + noise, 1.0) * dc_reflectance

    scaled = (means + extent) * (granularity / (2.0 * extent))
    indices = np.floor(scaled).astype(np.int64)
    inside = np.all((indices >= 0) & (indices < granularity), axis=1)
    field = np.zeros((granularity, granularity, granularity), dtype=np.float64)
    kept = indices[inside]
    np.add.at(field, (kept[:, 0], kept[:, 1], kept[:, 2]), weight[inside])
    kept_fraction = float(weight[inside].sum() / weight.sum())
    return field, granularity, kept_fraction


# (loader-kind, source, column title, subtitle)
COLUMNS = [
    ("rift", "training_checkpoints/b787_r7_target20k_shdeg1em9/checkpoint_final.pth.tar",
     "RIFT · Round 7 (selected)", "coherent 25.3372% · fixed F1 0.5262"),
    ("rift", "training_checkpoints/b787_r5_deg0/checkpoint_best.pth.tar",
     "SpINR-style baseline", "coherent 28.5352% · fixed F1 0.6052"),
    ("rift", "training_checkpoints/b787_radar_fields_released/checkpoint_best.pth.tar",
     "Radar Fields baseline", "power 304.8836% · fixed F1 0.0062"),
    ("anchor", ANCHOR_NPZ,
     "Matched-filter BP (E28)", "model-free anchor · fixed F1 0.2838"),
    ("geraf", GERAF_CKPT,
     "GeRaF baseline", "geraf_3d 27.4973% · no geometry export"),
    ("radarsplat", RADARSPLAT_CKPT,
     "RadarSplat S0", "power 92.7748% · initial, untuned"),
    ("missing", ("Sugavanam–Ertin baseline", "NO RESULT",
                 "Stage 1 preempted\nStage 2 never started\nno surface exported"), "", ""),
]


def draw_placeholder(ax, headline, detail, first_row, first_col, title, plane_name):
    ax.set_facecolor("#f1f0ed")
    ax.text(0.5, 0.60, headline, ha="center", va="center", transform=ax.transAxes,
            fontsize=13, fontweight="bold", color="#6f6d68")
    ax.text(0.5, 0.37, detail, ha="center", va="center", transform=ax.transAxes,
            fontsize=8.5, color="#52514e", linespacing=1.4)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#cbc8c2")
    if first_row:
        ax.set_title(title, fontsize=10, color="#0b0b0b", pad=7)
    if first_col:
        ax.set_ylabel(plane_name, fontsize=11, color="#52514e")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz-path", default="data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere2k.npz")
    p.add_argument("--stl", default="data/B787.stl")
    p.add_argument("--extent", type=float, default=0.15)
    p.add_argument("--upsample", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--pmin-pct", type=float, default=88.0)
    p.add_argument("--planes", default="0,2")
    p.add_argument("--lim", type=float, default=0.075)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    plane_idx = [int(i) for i in args.planes.split(",")]
    meta = json.loads(str(np.load(args.npz_path, allow_pickle=True,
                                  mmap_mode="r")["metadata_json"]))
    verts = stl_into_scene_frame(load_stl_vertices(args.stl), meta)
    verts = verts[np.random.default_rng(0).choice(
        verts.shape[0], min(12000, verts.shape[0]), replace=False)]

    n_r, n_c = len(plane_idx), len(COLUMNS)
    fig, axes = plt.subplots(n_r, n_c, figsize=(2.45 * n_c, 3.05 * n_r), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")

    for ci, (kind, source, title, subtitle) in enumerate(COLUMNS):
        if kind == "missing":
            head_title, headline, detail = source
            for ri, pi in enumerate(plane_idx):
                draw_placeholder(axes[ri][ci], headline, detail, ri == 0, ci == 0,
                                 head_title, PLANES[pi][0])
            continue

        if kind == "rift":
            energy, g, epoch = load_energy_field(source)
            note = f"ep{epoch}"
        elif kind == "anchor":
            energy, g = load_anchor_energy(source)
            note = "coherent adjoint"
        elif kind == "geraf":
            energy, g, inv_s = load_geraf_energy(source, extent=args.extent)
            note = f"inv_s={inv_s:.1f}"
        elif kind == "radarsplat":
            energy, g, kept_fraction = load_radarsplat_energy(
                source, extent=args.extent
            )
            note = f"{100.0 * kept_fraction:.1f}% native weight in crop"
        else:
            raise ValueError(kind)

        print(f"{title}: {note}, {g}^3, energy range "
              f"[{energy.min():.3e}, {energy.max():.3e}]")
        energy = trilinear_upsample(energy, args.upsample)

        for ri, pi in enumerate(plane_idx):
            pname, mip_ax, ha, va, _hl, _vl = PLANES[pi]
            ax = axes[ri][ci]
            mip = energy.max(axis=mip_ax)
            remaining = [a for a in range(3) if a != mip_ax]
            img = mip.T if remaining == [ha, va] else mip
            disp = img ** args.gamma
            e = args.extent
            ax.imshow(disp, origin="lower", extent=[-e, e, -e, e], cmap="inferno",
                      aspect="equal", interpolation="bilinear",
                      vmin=np.percentile(disp, args.pmin_pct),
                      vmax=np.percentile(disp, 99.7))
            ax.scatter(verts[:, ha], verts[:, va], s=0.3, c="cyan", alpha=0.10,
                       linewidths=0, rasterized=True)
            ax.set_xlim(-args.lim, args.lim)
            ax.set_ylim(-args.lim, args.lim)
            ax.set_xticks([]); ax.set_yticks([])
            if ci == 0:
                ax.set_ylabel(pname, fontsize=11, color="#52514e")
            if ri == 0:
                title_lines = "\n".join(textwrap.wrap(
                    title, width=25, break_long_words=False
                ))
                subtitle_lines = "\n".join(textwrap.wrap(
                    subtitle, width=29, break_long_words=False
                ))
                ax.set_title(f"{title_lines}\n{subtitle_lines}", fontsize=8.4,
                             color="#0b0b0b", pad=7, linespacing=1.08)

    fig.suptitle("B787 learned 3D fields — every baseline on a common readout, vs STL truth (cyan)",
                 fontsize=13, color="#0b0b0b")
    fig.text(0.5, 0.012,
             "Visualization only: each field is read on the same 48³ lattice and trilinearly "
             "upsampled; display contrast is normalized per method, so this compares SUPPORT, "
             "not amplitude. Metrics remain native-grid.",
             ha="center", va="bottom", fontsize=8.5, color="#66635e")
    fig.tight_layout(rect=[0, 0.035, 1, 0.955])
    fig.savefig(args.out, dpi=170, facecolor=fig.get_facecolor())
    print("wrote", args.out)


if __name__ == "__main__":
    main()
