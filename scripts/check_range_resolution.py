#!/usr/bin/env python
"""Measure the native range-domain PSF width of the pec_sphere npz dataset.

Context: the Goal-1a reconstruction sweep (EXPERIMENT_MANAGER_HANDOFF.md,
2026-07-08/09) fits its training data well (grid_sh train rel-MSE 0.32%) but
the learned scatterer field is a broad radial blob (~0.6-0.7m FWHM) centered
near the true r=1.0m target radius, not a thin shell -- true for ALL THREE
representations tried (grid/grid_sh/point_sh), which rules out a
representation-capacity explanation. This dataset's native bandwidth is only
149.9 MHz (30 ADC samples), giving a theoretical range resolution
c/(2B) = 1.00m -- suspiciously close to the observed blob width. This script
checks that directly: build a single-viewpoint matched-filter range profile
(the same model_response() recipe validated in
scripts/validate_pec_sphere_coherence.py, NOT the full 3D forward operator)
and measure its peak width.

Method: for one viewpoint, sample candidate points along the boresight ray
p(r) = u * r (u = unit direction from origin to that viewpoint's array
center, same convention as validate_pec_sphere_coherence.py's specular
point) at fine spacing, score each with the matched-filter statistic
|sum_{tx,rx,f} conj(model_response(p(r))) * data|, and report the FWHM of
the resulting range profile around its peak. Repeated over several
viewpoints for robustness (average FWHM, not just one draw).

Usage:
    python scripts/check_range_resolution.py
    python scripts/check_range_resolution.py --num-views 10 --r-lo 0.3 --r-hi 1.7
"""
import argparse
import json

import numpy as np

CC = 299792458.0


def load_cube(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    resp = d["response"]  # [n_view, Tx, Rx, n_chirp, n_adc] complex64
    vp = d["viewpoint_positions"]  # [n_view, 3]
    tx_pos = d["tx_pos"]
    rx_pos = d["rx_pos"]
    meta = json.loads(str(d["metadata_json"]))
    cube = resp.mean(axis=3).astype(np.complex128)  # average redundant chirps
    return cube, vp, tx_pos, rx_pos, meta


def build_freqs(meta):
    fc = float(meta["radar_fc_hz"])
    bw = float(meta["radar_bandwidth_hz"])
    n_adc = int(meta["num_adc_samples"])
    return (fc - bw / 2) + np.arange(n_adc) * (bw / n_adc)


def model_response(p, txp, rxp, freqs):
    """exp(-j*2*pi*f/c*R_bistatic) for one candidate 3D point p -- same sign
    convention validated in validate_pec_sphere_coherence.py."""
    Rt = np.linalg.norm(txp - p[None, :], axis=-1)
    Rr = np.linalg.norm(rxp - p[None, :], axis=-1)
    Rtr = Rt[:, None] + Rr[None, :]
    phase = -2j * np.pi * freqs[None, None, :] / CC * Rtr[:, :, None]
    return np.exp(phase)


def range_profile(cube_v, txp, rxp, freqs, u, r_grid):
    profile = np.empty(len(r_grid))
    for i, r in enumerate(r_grid):
        p = u * r
        model = model_response(p, txp, rxp, freqs)
        profile[i] = np.abs((np.conj(model) * cube_v).sum())
    return profile


def fwhm(r_grid, profile):
    peak_idx = int(np.argmax(profile))
    peak_val = profile[peak_idx]
    half = peak_val / 2.0
    # walk left/right from the peak to the half-max crossing
    lo = peak_idx
    while lo > 0 and profile[lo] > half:
        lo -= 1
    hi = peak_idx
    while hi < len(profile) - 1 and profile[hi] > half:
        hi += 1

    def _interp_crossing(i0, i1):
        r0, r1 = r_grid[i0], r_grid[i1]
        p0, p1 = profile[i0], profile[i1]
        if p1 == p0:
            return r0
        t = (half - p0) / (p1 - p0)
        return r0 + t * (r1 - r0)

    r_lo = _interp_crossing(lo, lo + 1) if lo < peak_idx else r_grid[lo]
    r_hi = _interp_crossing(hi - 1, hi) if hi > peak_idx else r_grid[hi]
    return r_grid[peak_idx], r_hi - r_lo


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", default="data/pec_sphere_fmcw_16t16r_79ghz_r10m_2k.npz")
    p.add_argument("--num-views", type=int, default=8)
    p.add_argument("--r-lo", type=float, default=0.3)
    p.add_argument("--r-hi", type=float, default=1.7)
    p.add_argument("--r-step", type=float, default=0.002, help="range-grid spacing, meters")
    p.add_argument("--plot", default="figures/range_resolution_check.png")
    args = p.parse_args()

    cube, vp, tx_pos, rx_pos, meta = load_cube(args.npz)
    freqs = build_freqs(meta)
    sphere_r = float(meta["target_radius_m"])
    B = float(meta["radar_bandwidth_hz"])
    theory_res = CC / (2 * B)
    print(f"Loaded {args.npz}: fc={meta['radar_fc_hz']:.4e}Hz, bw={B:.4e}Hz, "
          f"n_adc={meta['num_adc_samples']}, target_radius={sphere_r}m")
    print(f"Theoretical range resolution c/(2B) = {theory_res:.4f}m\n")

    r_grid = np.arange(args.r_lo, args.r_hi, args.r_step)
    fwhms, peaks = [], []
    profiles = []
    for v in range(args.num_views):
        u = vp[v] / np.linalg.norm(vp[v])
        profile = range_profile(cube[v], tx_pos[v], rx_pos[v], freqs, u, r_grid)
        profiles.append(profile)
        peak_r, width = fwhm(r_grid, profile)
        fwhms.append(width)
        peaks.append(peak_r)
        print(f"  view {v:3d}: peak at r={peak_r:.3f}m (target {sphere_r}m), FWHM={width:.3f}m")

    fwhms = np.array(fwhms)
    peaks = np.array(peaks)
    print(f"\nOver {args.num_views} views: mean FWHM={fwhms.mean():.3f}m (std {fwhms.std():.3f}m), "
          f"mean peak r={peaks.mean():.3f}m (target {sphere_r}m)")
    print(f"Ratio mean_FWHM / theoretical_resolution = {fwhms.mean() / theory_res:.2f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for v, profile in enumerate(profiles):
            ax.plot(r_grid, profile / profile.max(), alpha=0.6, label=f"view {v}")
        ax.axvline(sphere_r, color="k", linestyle="--", linewidth=1, label=f"target r={sphere_r}m")
        ax.axvspan(sphere_r - theory_res / 2, sphere_r + theory_res / 2, color="gray", alpha=0.15,
                   label=f"theoretical resolution cell ({theory_res:.2f}m)")
        ax.set_xlabel("range r (m)")
        ax.set_ylabel("normalized matched-filter magnitude")
        ax.set_title("Single-viewpoint range profiles vs. theoretical resolution cell")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"\nSaved plot to {args.plot}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
