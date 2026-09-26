#!/usr/bin/env python
"""R1b/R1c: point-target PSF geometry eval (system impulse-response calibration).

Context (EXPERIMENT_MANAGER_HANDOFF.md, R1b/R1c; DATASET_SPEC_POINT_TARGETS.md):
after training a grid g48 scene on a point-target npz (single 2cm PEC sphere, or
the 6-point constellation), this measures how the system+representation renders
an ideal impulse -- the end-to-end PSF. Truth positions are read from the npz's
`target_positions` key (never filenames). Per target it reports:
  * recovered peak position (argmax and energy-weighted centroid),
  * BIAS VECTOR (recovered - truth) and its radial component -- the sagitta
    hypothesis predicts NO outward radial bias for these OFF-surface points,
    unlike the curved sphere shell (+3.5cm), and predicts the bias (if any)
    tracks local view geometry, not the origin-radial direction,
  * per-axis 3D FWHM of the reconstructed lobe.
For any target pair closer than --pair-thresh (the C5-C6 10cm pair in R1c), it
also samples energy along the pair axis and reports whether two peaks are
resolved or merged -- the first multi-view SYNTHESIZED two-point resolution
measurement (the sphere shell's FWHM always confounded this with target extent).

Energy scalar and checkpoint loading are shared with scripts/eval_scene_geometry.py
(grid -> |w|^2; grid_sh/point_sh -> sum over SH basis; point_sh CIC-deposited to a
dense grid). extent defaults to 1.5 (not stored in checkpoints; the point-target
runs use --extent 1.5).

Usage:
    python scripts/eval_point_targets.py \
        --checkpoint training_checkpoints/pec_pointtarget_bw3ghz_grid_g48/checkpoint_final.pth.tar \
        --npz-path data/pec_pointtarget_fmcw_16t16r_79ghz_bw3ghz_r10m_500.npz
"""
import argparse
import csv
import os

import numpy as np
import torch

from eval_scene_geometry import load_energy_cloud, deposit_points, trilinear_sample  # noqa: E402


def _local_cube(volume, extent, center, half, step):
    """Trilinear-sample `volume` on an axis-aligned cube of half-width `half`
    (m) at `step` (m) around `center`. Returns (cube[n,n,n], axis_coords[n])."""
    n = int(round(2 * half / step)) + 1
    # absolute per-axis coordinate lines centred on the target
    lines = [torch.linspace(center[a] - half, center[a] + half, n, dtype=torch.float64) for a in range(3)]
    gx, gy, gz = torch.meshgrid(lines[0], lines[1], lines[2], indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)
    vals = trilinear_sample(volume, extent, pts).reshape(n, n, n)
    return vals, [ln.numpy() for ln in lines]


def _axis_fwhm(line_coords, profile, i_peak):
    """FWHM of a 1D lobe around index i_peak via half-max crossings."""
    p = np.asarray(profile)
    c = np.asarray(line_coords)
    if p[i_peak] <= 0:
        return float("nan")
    half = p[i_peak] / 2.0
    left = None
    for j in range(i_peak, 0, -1):
        if p[j - 1] < half <= p[j]:
            f = (half - p[j - 1]) / (p[j] - p[j - 1])
            left = c[j - 1] + f * (c[j] - c[j - 1])
            break
    right = None
    for j in range(i_peak, len(p) - 1):
        if p[j + 1] < half <= p[j]:
            f = (half - p[j + 1]) / (p[j] - p[j + 1])
            right = c[j + 1] + f * (c[j] - c[j + 1])
            break
    if left is None or right is None:
        return float("nan")
    return float(right - left)


def analyze_target(volume, extent, truth, half, step, centroid_frac):
    """Localize the reconstructed lobe near `truth` [3]. Returns a dict with
    argmax position, energy-weighted centroid, bias vectors, and per-axis FWHM."""
    cube, lines = _local_cube(volume, extent, truth, half, step)
    arr = cube.numpy()
    imax = np.unravel_index(int(np.argmax(arr)), arr.shape)
    peak_pos = np.array([lines[a][imax[a]] for a in range(3)])

    # centroid over the lobe (voxels >= centroid_frac * local max), sub-voxel
    thr = centroid_frac * arr.max()
    mask = arr >= thr
    gx, gy, gz = np.meshgrid(lines[0], lines[1], lines[2], indexing="ij")
    w = arr[mask]
    centroid = np.array([
        (gx[mask] * w).sum() / w.sum(),
        (gy[mask] * w).sum() / w.sum(),
        (gz[mask] * w).sum() / w.sum(),
    ]) if w.sum() > 0 else peak_pos.copy()

    # per-axis FWHM through the argmax voxel
    fwhm = np.array([
        _axis_fwhm(lines[0], arr[:, imax[1], imax[2]], imax[0]),
        _axis_fwhm(lines[1], arr[imax[0], :, imax[2]], imax[1]),
        _axis_fwhm(lines[2], arr[imax[0], imax[1], :], imax[2]),
    ])

    bias = peak_pos - truth
    r_truth = np.linalg.norm(truth)
    u_r = truth / r_truth if r_truth > 0 else np.zeros(3)
    return dict(
        truth=truth, peak_pos=peak_pos, centroid=centroid,
        bias=bias, bias_norm=float(np.linalg.norm(bias)),
        bias_radial=float(bias @ u_r),                      # + = outward
        bias_radial_centroid=float((centroid - truth) @ u_r),
        fwhm=fwhm, fwhm_mean=float(np.nanmean(fwhm)),
        peak_energy=float(arr.max()),
    )


def analyze_pair(volume, extent, t0, t1, step, oversample=1.6):
    """Sample energy along the axis through t0,t1 (extended by `oversample`)
    and report whether two peaks are resolved. Returns dict."""
    d = np.linalg.norm(t1 - t0)
    u = (t1 - t0) / d
    mid = 0.5 * (t0 + t1)
    s = torch.arange(-oversample * d / 2, oversample * d / 2 + 0.5 * step, step, dtype=torch.float64)
    pts = torch.as_tensor(mid, dtype=torch.float64)[None, :] + s[:, None] * torch.as_tensor(u, dtype=torch.float64)[None, :]
    prof = trilinear_sample(volume, extent, pts).numpy()
    s = s.numpy()

    # local maxima above half of global max
    gmax = prof.max()
    peaks = [i for i in range(1, len(prof) - 1)
             if prof[i] > prof[i - 1] and prof[i] >= prof[i + 1] and prof[i] >= 0.5 * gmax]
    resolved = len(peaks) >= 2
    dip_ratio = float("nan")
    sep = float("nan")
    if resolved:
        i0, i1 = peaks[0], peaks[-1]
        valley = prof[i0:i1 + 1].min()
        dip_ratio = float(valley / min(prof[i0], prof[i1]))  # Rayleigh-style: <~0.81 = resolved
        sep = float(abs(s[i1] - s[i0]))
    return dict(true_sep=float(d), resolved=resolved, n_peaks=len(peaks),
                recovered_sep=sep, dip_ratio=dip_ratio,
                s=s, profile=prof)


def build_volume(checkpoint, extent, deposit_res):
    scene_repr, pos, energy, grid_vol = load_energy_cloud(checkpoint, extent)
    if grid_vol is None:                                    # point_sh
        grid_vol = deposit_points(pos, energy, extent, deposit_res)
    return scene_repr, grid_vol


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--npz-path", required=True, help="point-target npz (reads `target_positions`)")
    p.add_argument("--extent", type=float, default=1.5)
    p.add_argument("--half", type=float, default=0.12, help="half-width (m) of the per-target search cube")
    p.add_argument("--step", type=float, default=0.002, help="sampling step (m)")
    p.add_argument("--centroid-frac", type=float, default=0.25,
                   help="fraction of local max above which voxels enter the centroid")
    p.add_argument("--pair-thresh", type=float, default=0.20,
                   help="targets closer than this (m) are analyzed as a resolution pair")
    p.add_argument("--deposit-res", type=int, default=512, help="point_sh dense-deposit resolution")
    p.add_argument("--out-dir", default="figures/r1bc_point_targets")
    args = p.parse_args()

    d = np.load(args.npz_path, allow_pickle=True)
    truth = np.asarray(d["target_positions"], dtype=np.float64)         # [T,3]
    os.makedirs(args.out_dir, exist_ok=True)
    tag = os.path.basename(os.path.dirname(args.checkpoint)) or "ckpt"

    scene_repr, volume = build_volume(args.checkpoint, args.extent, args.deposit_res)
    print(f"scene_repr={scene_repr}  targets={len(truth)}  volume={tuple(volume.shape)}")

    rows = []
    for i, t in enumerate(truth):
        res = analyze_target(volume, args.extent, t, args.half, args.step, args.centroid_frac)
        rows.append(dict(
            target=i,
            truth_x=t[0], truth_y=t[1], truth_z=t[2], truth_r=float(np.linalg.norm(t)),
            peak_x=res["peak_pos"][0], peak_y=res["peak_pos"][1], peak_z=res["peak_pos"][2],
            cen_x=res["centroid"][0], cen_y=res["centroid"][1], cen_z=res["centroid"][2],
            bias_x=res["bias"][0], bias_y=res["bias"][1], bias_z=res["bias"][2],
            bias_norm=res["bias_norm"], bias_radial=res["bias_radial"],
            bias_radial_centroid=res["bias_radial_centroid"],
            fwhm_x=res["fwhm"][0], fwhm_y=res["fwhm"][1], fwhm_z=res["fwhm"][2], fwhm_mean=res["fwhm_mean"],
        ))
        print(f"  T{i} truth=({t[0]:+.3f},{t[1]:+.3f},{t[2]:+.3f}) r={np.linalg.norm(t):.3f}  "
              f"peak=({res['peak_pos'][0]:+.3f},{res['peak_pos'][1]:+.3f},{res['peak_pos'][2]:+.3f})  "
              f"|bias|={res['bias_norm']*100:.1f}cm  bias_radial(argmax/cen)="
              f"{res['bias_radial']*100:+.1f}/{res['bias_radial_centroid']*100:+.1f}cm  "
              f"FWHM=({res['fwhm'][0]*100:.1f},{res['fwhm'][1]*100:.1f},{res['fwhm'][2]*100:.1f})cm")

    with open(os.path.join(args.out_dir, f"{tag}_targets.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # resolution pairs
    for i in range(len(truth)):
        for j in range(i + 1, len(truth)):
            sep = np.linalg.norm(truth[i] - truth[j])
            if sep <= args.pair_thresh:
                pr = analyze_pair(volume, args.extent, truth[i], truth[j], args.step)
                verdict = "RESOLVED" if pr["resolved"] else "MERGED"
                print(f"  PAIR T{i}-T{j}: true_sep={sep*100:.1f}cm -> {verdict} "
                      f"(n_peaks={pr['n_peaks']}, recovered_sep="
                      f"{pr['recovered_sep']*100 if pr['recovered_sep']==pr['recovered_sep'] else float('nan'):.1f}cm, "
                      f"dip_ratio={pr['dip_ratio']:.2f})")
                np.savez(os.path.join(args.out_dir, f"{tag}_pair_{i}_{j}.npz"),
                         s=pr["s"], profile=pr["profile"], true_sep=pr["true_sep"])

    print(f"\nWrote per-target CSV + pair profiles -> {args.out_dir}/")


if __name__ == "__main__":
    main()
