#!/usr/bin/env python
"""R1a: geometric-fidelity eval of trained scene checkpoints (3D-reconstruction thread).

Context (EXPERIMENT_MANAGER_HANDOFF.md, "PROJECT REFOCUS ... 3D reconstruction
quality", 2026-07-19; project memory rift-reconstruction-focus): the
interpolation/val-rel-MSE thread is CLOSED. The success metric is now the
GEOMETRY of the reconstruction's energy support -- peak radius, per-direction
radial FWHM, shell/interior energy concentration -- NOT val rel-MSE and NEVER
|w| amplitude (|w| is a Born density + speckle; only its spatial support is
meaningful). This script computes those geometry metrics for the sphere-target
checkpoints so the two measured defects can be tracked:
  1. +3.5cm outward radial bias (does the peak sit at ~1.035 vs true 1.000?)
  2. ~2x excess radial FWHM (~20cm observed vs a ~9-10cm physics+grid budget).

Energy scalar (rotation-invariant, per voxel/point):
  grid     -> |w|^2                       (isotropic complex weight)
  grid_sh  -> sum_b (w_re[b]^2 + w_im[b]^2) over ALL SH basis slots
  point_sh -> same per-point sum over SH basis slots
For the real orthonormal SH basis this sum is the direction-integrated angular
power (Parseval), i.e. "how much this voxel/point scatters over all aspects" --
the right support scalar, and used consistently so amplitude scale (which
differs between grid and grid_sh) never enters: every reported number is a
radius, a width, or a normalized fraction, all scale-invariant.

Two families of metric:
  * GLOBAL radial stats -- computed directly from every active scatterer's
    (r=||pos||, energy), no interpolation, no free parameters: energy-weighted
    mean/std radius, radial-histogram peak radius, shell-energy fraction within
    +/-7.5cm of r=1.000 and of r=1.035, interior (r<0.8) energy fraction.
  * PER-RAY radial profiles -- sample the energy field along >=400 Fibonacci
    directions from the origin over r in [0.6,1.4] at <=2mm steps (trilinear),
    then per ray: peak radius and FWHM. grid/grid_sh are sampled from their
    native [G,G,G] energy volume; point_sh is first CIC-splatted onto a dense
    volume (render-agnostic "deposit onto a dense grid" step) then sampled the
    same way. Reports mean+/-sd of per-ray peak radius and median (p25/p75) FWHM.

extent is NOT stored in these checkpoints; every bw3ghz/reg sphere run used
--extent 1.5, which is the default here (override with --extent).

Usage:
    # full R1a sweep (default checkpoint list) -> summary CSV + per-ray CSVs
    python scripts/eval_scene_geometry.py

    # single checkpoint
    python scripts/eval_scene_geometry.py \
        --checkpoints training_checkpoints/pec_sphere_recon_bw3ghz_grid_g48/checkpoint_final.pth.tar
"""
import argparse
import csv
import math
import os

import numpy as np
import torch

# --- energy extraction ------------------------------------------------------


def _infer_scene_repr(state_dict):
    if "anchors" in state_dict:
        return "point_sh"
    if state_dict["w_re"].ndim == 4:
        return "grid_sh"
    if state_dict["w_re"].ndim == 3:
        return "grid"
    raise ValueError(f"cannot infer scene_repr from w_re shape {tuple(state_dict['w_re'].shape)}")


def load_energy_cloud(checkpoint_path, extent, device="cpu"):
    """Returns (scene_repr, positions[N,3], energy[N], grid_volume_or_None).

    Only ACTIVE scatterers are returned (inactive voxels/points contribute
    exactly zero and are dropped). For grid/grid_sh, grid_volume is the dense
    [G,G,G] energy tensor (inactive voxels zeroed) used for per-ray sampling;
    for point_sh it is None (a deposit volume is built later).
    """
    ck = torch.load(checkpoint_path, map_location=device)
    sd = ck["model_state_dict"]
    scene_repr = ck.get("scene_repr") or _infer_scene_repr(sd)

    if scene_repr == "grid":
        w_re, w_im = sd["w_re"].double(), sd["w_im"].double()          # [G,G,G]
        energy_vol = w_re ** 2 + w_im ** 2
        active = sd.get("active_mask", torch.ones_like(energy_vol, dtype=torch.bool))
        energy_vol = energy_vol * active.double()
        G = energy_vol.shape[0]
        pos = _grid_positions(G, extent)                                # [G,G,G,3]
        flat_mask = active.reshape(-1)
        return scene_repr, pos.reshape(-1, 3)[flat_mask], energy_vol.reshape(-1)[flat_mask], energy_vol

    if scene_repr == "grid_sh":
        w_re, w_im = sd["w_re"].double(), sd["w_im"].double()          # [G,G,G,B]
        energy_vol = (w_re ** 2 + w_im ** 2).sum(dim=-1)               # [G,G,G]
        active = sd.get("active_mask", torch.ones_like(energy_vol, dtype=torch.bool))
        energy_vol = energy_vol * active.double()
        G = energy_vol.shape[0]
        pos = _grid_positions(G, extent)
        flat_mask = active.reshape(-1)
        return scene_repr, pos.reshape(-1, 3)[flat_mask], energy_vol.reshape(-1)[flat_mask], energy_vol

    if scene_repr == "point_sh":
        anchors = sd["anchors"].double()
        cell_half = sd["cell_half"].double()                           # [K,1]
        delta_raw = sd["delta_raw"].double()
        pos = anchors + cell_half * torch.tanh(delta_raw)              # [K,3]
        w_re, w_im = sd["w_re"].double(), sd["w_im"].double()          # [K,B]
        energy = (w_re ** 2 + w_im ** 2).sum(dim=-1)                   # [K]
        active = sd.get("active_mask", torch.ones(pos.shape[0], dtype=torch.bool))
        return scene_repr, pos[active], energy[active], None

    raise ValueError(f"unsupported scene_repr={scene_repr!r} (grid/grid_sh/point_sh only)")


def _grid_positions(G, extent):
    """Voxel-center grid over [-extent,extent]^3, matching
    rift.encoding.generate_dynamic_grid(jitter=False): centers at
    intervals midpoints, cartesian_prod (x outer, y mid, z inner)."""
    intervals = torch.linspace(-extent, extent, G + 1, dtype=torch.float64)
    centers = (intervals[:-1] + intervals[1:]) / 2
    xx, yy, zz = torch.meshgrid(centers, centers, centers, indexing="ij")
    return torch.stack([xx, yy, zz], dim=-1)                           # [G,G,G,3]


# --- trilinear volume sampling / deposition ---------------------------------


def trilinear_sample(vol, extent, pts):
    """Sample dense [D,D,D] energy volume `vol` (voxel centers on the same
    convention as _grid_positions, over [-extent,extent]) at query points
    `pts` [M,3], via trilinear interpolation. Out-of-range indices are
    edge-clamped (all radial samples here sit inside the box, so this only
    guards the boundary)."""
    D = vol.shape[0]
    pitch = 2.0 * extent / D
    f = (pts + extent) / pitch - 0.5                                    # [M,3] fractional idx
    i0 = torch.floor(f).long()
    t = f - i0.double()
    out = torch.zeros(pts.shape[0], dtype=torch.float64)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                ix = (i0[:, 0] + dx).clamp(0, D - 1)
                iy = (i0[:, 1] + dy).clamp(0, D - 1)
                iz = (i0[:, 2] + dz).clamp(0, D - 1)
                wx = t[:, 0] if dx else 1.0 - t[:, 0]
                wy = t[:, 1] if dy else 1.0 - t[:, 1]
                wz = t[:, 2] if dz else 1.0 - t[:, 2]
                out += (wx * wy * wz) * vol[ix, iy, iz]
    return out


def deposit_points(pos, energy, extent, D):
    """CIC (trilinear-splat) deposition of point energies onto a dense
    [D,D,D] volume over [-extent,extent] -- the point_sh analogue of the
    grid's native energy volume. Conserves total energy (each point's energy
    is distributed to its 8 surrounding voxels by trilinear weights)."""
    pitch = 2.0 * extent / D
    f = (pos + extent) / pitch - 0.5
    i0 = torch.floor(f).long()
    t = f - i0.double()
    vol = torch.zeros(D * D * D, dtype=torch.float64)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                ix = (i0[:, 0] + dx).clamp(0, D - 1)
                iy = (i0[:, 1] + dy).clamp(0, D - 1)
                iz = (i0[:, 2] + dz).clamp(0, D - 1)
                wx = t[:, 0] if dx else 1.0 - t[:, 0]
                wy = t[:, 1] if dy else 1.0 - t[:, 1]
                wz = t[:, 2] if dz else 1.0 - t[:, 2]
                idx = (ix * D + iy) * D + iz
                vol.index_add_(0, idx, (wx * wy * wz) * energy)
    return vol.view(D, D, D)


# --- radial geometry --------------------------------------------------------


def fibonacci_directions(n):
    i = torch.arange(n, dtype=torch.float64) + 0.5
    phi = torch.arccos(1.0 - 2.0 * i / n)                              # polar
    golden = math.pi * (1.0 + 5.0 ** 0.5)
    theta = golden * i                                                 # azimuth
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    return torch.stack([x, y, z], dim=-1)                             # [n,3] unit


def _cone_dirs(axis, cone_deg, cone_n):
    """`cone_n` unit directions within `cone_deg` of `axis` (a [3] unit
    vector), as a low-discrepancy spherical cap. Averaging a ray's radial
    profile over this cone denoises the profile WITHOUT any radial blur --
    essential for sparse point_sh deposits (a pencil ray hits mostly empty
    voxels), a near-no-op for the dense grid volumes."""
    if cone_n <= 1 or cone_deg <= 0:
        return axis[None, :]
    cos_max = math.cos(math.radians(cone_deg))
    i = torch.arange(cone_n, dtype=torch.float64) + 0.5
    cz = 1.0 - (1.0 - cos_max) * (i / cone_n)                          # ~uniform on cap
    sz = torch.sqrt((1.0 - cz ** 2).clamp_min(0.0))
    az = math.pi * (1.0 + 5.0 ** 0.5) * i
    cap = torch.stack([sz * torch.cos(az), sz * torch.sin(az), cz], dim=-1)  # around +z
    # rotate +z -> axis (Rodrigues); handle the axis == +/-z degenerate cases
    z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    axis = axis / axis.norm()
    v = torch.cross(z, axis, dim=-1)
    c = torch.dot(z, axis)
    if v.norm() < 1e-12:
        return cap if c > 0 else cap * torch.tensor([1.0, 1.0, -1.0], dtype=torch.float64)
    vx = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=torch.float64)
    R = torch.eye(3, dtype=torch.float64) + vx + vx @ vx * (1.0 / (1.0 + c))
    return cap @ R.T


def _fwhm(r, profile):
    """FWHM of a 1D radial profile via linear-interpolated half-max crossings
    on both sides of the global peak. Returns (fwhm, peak_r, truncated) where
    truncated=True (fwhm=nan) if the profile does not fall below half-max on
    one side within the sampled window."""
    r = np.asarray(r)
    p = np.asarray(profile)
    if p.max() <= 0:
        return float("nan"), float("nan"), True
    k = int(np.argmax(p))
    half = p[k] / 2.0
    peak_r = r[k]

    # left crossing
    li = None
    for j in range(k, 0, -1):
        if p[j - 1] < half <= p[j]:
            frac = (half - p[j - 1]) / (p[j] - p[j - 1])
            li = r[j - 1] + frac * (r[j] - r[j - 1])
            break
    # right crossing
    ri = None
    for j in range(k, len(p) - 1):
        if p[j + 1] < half <= p[j]:
            frac = (half - p[j + 1]) / (p[j] - p[j + 1])
            ri = r[j + 1] + frac * (r[j] - r[j + 1])
            break
    if li is None or ri is None:
        return float("nan"), peak_r, True
    return float(ri - li), float(peak_r), False


def global_radial_stats(pos, energy, r_lo, r_hi, dr):
    r = torch.linalg.norm(pos, dim=-1).numpy()
    e = energy.numpy()
    tot = e.sum()
    if tot <= 0:
        return dict(n_active=len(e), total_energy=0.0, wmean_r=float("nan"),
                    wstd_r=float("nan"), hist_peak_r=float("nan"),
                    frac_shell_1000=float("nan"), frac_shell_1035=float("nan"),
                    frac_interior_0p8=float("nan"))
    wmean = float((e * r).sum() / tot)
    wstd = float(np.sqrt((e * (r - wmean) ** 2).sum() / tot))
    edges = np.arange(r_lo, r_hi + dr, dr)
    hist, _ = np.histogram(r, bins=edges, weights=e)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist_peak_r = float(centers[int(np.argmax(hist))]) if hist.max() > 0 else float("nan")
    return dict(
        n_active=int(len(e)),
        total_energy=float(tot),
        wmean_r=wmean,
        wstd_r=wstd,
        hist_peak_r=hist_peak_r,
        frac_shell_1000=float(e[np.abs(r - 1.000) <= 0.075].sum() / tot),
        frac_shell_1035=float(e[np.abs(r - 1.035) <= 0.075].sum() / tot),
        frac_interior_0p8=float(e[r < 0.8].sum() / tot),
    )


def per_ray_profiles(sample_fn, extent, n_dir, r_lo, r_hi, dr, cone_deg=3.0, cone_n=16):
    dirs = fibonacci_directions(n_dir)                                 # [D,3]
    r = torch.arange(r_lo, r_hi + 0.5 * dr, dr, dtype=torch.float64)   # [R]
    n_r = r.shape[0]
    profiles = np.zeros((n_dir, n_r))
    for j in range(n_dir):
        cone = _cone_dirs(dirs[j], cone_deg, cone_n)                   # [C,3]
        # [C,R,3] -> sample all cone*radius points, average over the cone
        pts = (cone[:, None, :] * r[None, :, None]).reshape(-1, 3)
        vals = sample_fn(pts).numpy().reshape(cone.shape[0], n_r)
        profiles[j] = vals.mean(axis=0)
    r_np = r.numpy()
    peaks, fwhms, trunc = [], [], 0
    for j in range(n_dir):
        fw, pk, tr = _fwhm(r_np, profiles[j])
        peaks.append(pk)
        fwhms.append(fw)
        trunc += int(tr)
    peaks = np.array(peaks)
    fwhms = np.array(fwhms)
    valid = ~np.isnan(fwhms)
    mean_profile = profiles.mean(axis=0)
    fw_avg, pk_avg, _ = _fwhm(r_np, mean_profile)
    return dict(
        r=r_np,
        mean_profile=mean_profile,
        ray_peak=peaks,
        ray_fwhm=fwhms,
        ray_peak_mean=float(np.mean(peaks)),
        ray_peak_sd=float(np.std(peaks)),
        fwhm_median=float(np.median(fwhms[valid])) if valid.any() else float("nan"),
        fwhm_p25=float(np.percentile(fwhms[valid], 25)) if valid.any() else float("nan"),
        fwhm_p75=float(np.percentile(fwhms[valid], 75)) if valid.any() else float("nan"),
        n_fwhm_valid=int(valid.sum()),
        n_fwhm_truncated=int(trunc),
        rayavg_peak_r=float(pk_avg),
        rayavg_fwhm=float(fw_avg) if not np.isnan(fw_avg) else float("nan"),
    )


# --- driver -----------------------------------------------------------------

DEFAULT_CHECKPOINTS = [
    # sweep baselines
    "pec_sphere_recon_bw3ghz_grid_g24",
    "pec_sphere_recon_bw3ghz_grid_g48",
    "pec_sphere_recon_bw3ghz_gridsh6_g24",
    "pec_sphere_recon_bw3ghz_gridsh6_g48",
    "pec_sphere_recon_bw3ghz_pointsh6_g48",
    # Stage-1 reg ladder (collapsed l1_xstrong / growonly excluded per spec)
    "pec_sphere_reg_l1_weak",
    "pec_sphere_reg_l1_mid",
    "pec_sphere_reg_l1_strong",
    "pec_sphere_reg_shsm_weak",
    "pec_sphere_reg_shsm_strong",
    "pec_sphere_reg_combo",
    "pec_sphere_reg_combo_strong",
    # Stage 2
    "pec_sphere_reg2_g48",
    "pec_sphere_reg2_shellinit",
    "pec_sphere_reg2_adaptive_deep/checkpoint_ep100_snapshot.pth.tar",
]


def _resolve(entry, ckpt_root):
    """Accept a bare run name, a run/sub.pth.tar, or a full path."""
    if entry.endswith(".pth.tar"):
        cand = entry if os.path.isabs(entry) or os.sep in entry else os.path.join(ckpt_root, entry)
        if os.path.exists(cand):
            return cand
        # run-name/checkpoint given relative to ckpt_root
        cand2 = os.path.join(ckpt_root, entry)
        return cand2
    # bare run name -> prefer final, else best
    for sub in ("checkpoint_final.pth.tar", "checkpoint_best.pth.tar"):
        cand = os.path.join(ckpt_root, entry, sub)
        if os.path.exists(cand):
            return cand
    return os.path.join(ckpt_root, entry, "checkpoint_final.pth.tar")


def eval_one(path, args):
    scene_repr, pos, energy, grid_vol = load_energy_cloud(path, args.extent)
    stats = global_radial_stats(pos, energy, args.r_lo, args.r_hi, args.dr)

    if grid_vol is not None:
        sample_fn = lambda pts: trilinear_sample(grid_vol, args.extent, pts)
        deposit_res = grid_vol.shape[0]
    else:
        vol = deposit_points(pos, energy, args.extent, args.deposit_res)
        sample_fn = lambda pts: trilinear_sample(vol, args.extent, pts)
        deposit_res = args.deposit_res

    rays = per_ray_profiles(sample_fn, args.extent, args.n_dir, args.r_lo, args.r_hi, args.dr,
                            cone_deg=args.cone_deg, cone_n=args.cone_n)
    return scene_repr, deposit_res, stats, rays


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", nargs="*", default=None,
                   help="run names / paths (default: the R1a sweep list)")
    p.add_argument("--checkpoint-root", default="training_checkpoints")
    p.add_argument("--extent", type=float, default=1.5,
                   help="scene box half-width (m); bw3ghz/reg runs used 1.5 (not stored in ckpt)")
    p.add_argument("--n-dir", type=int, default=400, help="number of Fibonacci ray directions")
    p.add_argument("--r-lo", type=float, default=0.6)
    p.add_argument("--r-hi", type=float, default=1.4)
    p.add_argument("--dr", type=float, default=0.002, help="radial step (m), <=2mm")
    p.add_argument("--deposit-res", type=int, default=512,
                   help="point_sh dense-deposit grid resolution per axis (512 over +/-1.5m = 5.9mm; "
                        "adaptive cells are ~3.9mm)")
    p.add_argument("--cone-deg", type=float, default=3.0,
                   help="half-angle (deg) of the per-ray averaging cone -- denoises sparse point_sh "
                        "profiles without radial blur; near-no-op on dense grids (0 = pencil ray)")
    p.add_argument("--cone-n", type=int, default=16, help="sub-directions averaged per ray cone")
    p.add_argument("--out-dir", default="figures/r1a_geometry")
    p.add_argument("--summary-csv", default=None,
                   help="summary CSV path (default: <out-dir>/summary.csv)")
    args = p.parse_args()

    entries = args.checkpoints if args.checkpoints else DEFAULT_CHECKPOINTS
    os.makedirs(args.out_dir, exist_ok=True)
    summary_csv = args.summary_csv or os.path.join(args.out_dir, "summary.csv")

    fields = ["checkpoint", "scene_repr", "n_active", "deposit_res",
              "hist_peak_r", "rayavg_peak_r", "wmean_r", "wstd_r",
              "ray_peak_mean", "ray_peak_sd",
              "fwhm_median", "fwhm_p25", "fwhm_p75", "n_fwhm_valid", "n_fwhm_truncated",
              "rayavg_fwhm", "frac_shell_1000", "frac_shell_1035", "frac_interior_0p8"]
    rows = []

    for entry in entries:
        path = _resolve(entry, args.checkpoint_root)
        name = entry.replace("/", "__").replace(".pth.tar", "")
        if not os.path.exists(path):
            print(f"[skip] {entry}: not found ({path})")
            continue
        scene_repr, deposit_res, stats, rays = eval_one(path, args)
        row = dict(
            checkpoint=name, scene_repr=scene_repr, n_active=stats["n_active"],
            deposit_res=deposit_res,
            hist_peak_r=stats["hist_peak_r"], rayavg_peak_r=rays["rayavg_peak_r"],
            wmean_r=stats["wmean_r"], wstd_r=stats["wstd_r"],
            ray_peak_mean=rays["ray_peak_mean"], ray_peak_sd=rays["ray_peak_sd"],
            fwhm_median=rays["fwhm_median"], fwhm_p25=rays["fwhm_p25"], fwhm_p75=rays["fwhm_p75"],
            n_fwhm_valid=rays["n_fwhm_valid"], n_fwhm_truncated=rays["n_fwhm_truncated"],
            rayavg_fwhm=rays["rayavg_fwhm"],
            frac_shell_1000=stats["frac_shell_1000"], frac_shell_1035=stats["frac_shell_1035"],
            frac_interior_0p8=stats["frac_interior_0p8"],
        )
        rows.append(row)

        # per-ray artifacts
        np.savez(os.path.join(args.out_dir, f"{name}_rays.npz"),
                 r=rays["r"], mean_profile=rays["mean_profile"],
                 ray_peak=rays["ray_peak"], ray_fwhm=rays["ray_fwhm"])
        with open(os.path.join(args.out_dir, f"{name}_mean_profile.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["r_m", "mean_energy"])
            for rr, ee in zip(rays["r"], rays["mean_profile"]):
                w.writerow([f"{rr:.4f}", f"{ee:.6e}"])

        print(f"[ok] {name}: repr={scene_repr} N={stats['n_active']} "
              f"hist_peak={stats['hist_peak_r']:.3f} rayavg_peak={rays['rayavg_peak_r']:.3f} "
              f"ray_peak={rays['ray_peak_mean']:.3f}+/-{rays['ray_peak_sd']:.3f} "
              f"FWHM={rays['fwhm_median']*100:.1f}cm[{rays['fwhm_p25']*100:.1f},{rays['fwhm_p75']*100:.1f}] "
              f"shell1.000={stats['frac_shell_1000']:.1%} shell1.035={stats['frac_shell_1035']:.1%} "
              f"r<0.8={stats['frac_interior_0p8']:.1%}"
              + (f" (trunc {rays['n_fwhm_truncated']})" if rays['n_fwhm_truncated'] else ""))

    with open(summary_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\nWrote {len(rows)} rows -> {summary_csv}")


if __name__ == "__main__":
    main()
