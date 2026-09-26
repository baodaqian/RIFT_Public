#!/usr/bin/env python
"""Validate multi-view coherence on the new PEC-sphere FMCW dataset.

Context: the 2026-07-08 pilot diagnosis found that per-view backprojection
was healthy but summing across views on `data/AEDT_Sphere_Repeat_CSV`
collapsed to near-zero for all but the 1-2 angularly-closest view pairs --
i.e. multi-view coordinate incoherence, root cause unresolved (get_array_pos
convention vs. mislabeled data). Daqian re-synthesized sphere data from
scratch with GROUND-TRUTH absolute array geometry per view to isolate it:
`data/pec_sphere_fmcw_16t16r_79ghz_r10m_2k.npz`.

This script does NOT use get_array_pos or forward_operator.py at all -- the
npz already supplies exact per-viewpoint tx_pos/rx_pos/viewpoint_positions,
so there is no analytic-convention assumption left to get wrong. It builds
its own reference matched-filter model directly from those positions.

Data format notes (reverse-engineered, no frequency array is stored):
- `response` is raw FMCW ADC data [n_view, Tx, Rx, n_chirp, n_adc], NOT a
  frequency-domain S-parameter array like the old CSV data.
- The n_chirp axis is redundant repeats for a static target (std ~6e-9 vs.
  signal ~7e-4 at tx0/rx0/view0) -- averaged away here, not Doppler content.
- The n_adc axis behaves like a genuine swept-frequency axis: treating it as
  S(f_i) for f_i = fc - B/2 + i*(B/n_adc) and taking `np.fft.ifft` over it
  (the same recipe as scripts/check_phase_sign.py) puts the range-profile
  peak EXACTLY at bin round((|viewpoint_pos| - target_radius) / (c/2B)) for
  every viewpoint checked -- i.e. the near-surface specular range. This
  fixes both the frequency-grid direction and the sign convention
  (S(f) = exp(-j*2*pi*f/c*R_bistatic) reproduces it) empirically, without
  needing a stored frequency array.

Two stages:
  Stage A (per-view sanity): fit ONE point scatterer at the analytic
  specular point (unit(viewpoint_dir) * sphere_radius) per viewpoint via
  closed-form complex projection; report explained power fraction.
  Gate: mean >= 0.9 over the sampled viewpoints (historical AEDT-data
  baseline was ~0.87 at its best single point; this data should clear it
  easily if geometry is self-consistent).

  Stage B (the actual multi-view coherence test): fix one 3D point (view
  0's specular point) and accumulate the matched-filter response from
  every OTHER viewpoint's own exact array geometry, ordered nearest-
  direction-first. Reports |cumulative sum| at several view counts, plus a
  control using only views >90 degrees from the reference direction (which
  should look like noise, not either explosive growth or collapse-to-zero).
  Gate: the running sum must stay within [0.3x, 3x] of its value at 5
  views, all the way out through all views used -- i.e. it neither
  collapses (the old bug) nor diverges, matching the qualitative behavior
  found interactively (saturates ~11-12 from a single-view baseline ~6,
  control floor ~0.1).

Usage:
    python scripts/validate_pec_sphere_coherence.py
    python scripts/validate_pec_sphere_coherence.py --npz <path> --num-check 40
"""
import argparse

import numpy as np

CC = 299792458.0


def load_cube(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    resp = d["response"]  # [n_view, Tx, Rx, n_chirp, n_adc] complex64
    vp = d["viewpoint_positions"]  # [n_view, 3]
    tx_pos = d["tx_pos"]  # [n_view, Tx, 3]
    rx_pos = d["rx_pos"]  # [n_view, Rx, 3]
    import json

    meta = json.loads(str(d["metadata_json"]))
    cube = resp.mean(axis=3).astype(np.complex128)  # average redundant chirps -> [n_view,Tx,Rx,n_adc]
    return cube, vp, tx_pos, rx_pos, meta


def build_freqs(meta):
    fc = float(meta["radar_fc_hz"])
    bw = float(meta["radar_bandwidth_hz"])
    n_adc = int(meta["num_adc_samples"])
    return (fc - bw / 2) + np.arange(n_adc) * (bw / n_adc)


def model_response(p, txp, rxp, freqs):
    """exp(-j*2*pi*f/c*R_bistatic) for one candidate 3D point p."""
    Rt = np.linalg.norm(txp - p[None, :], axis=-1)  # [Tx]
    Rr = np.linalg.norm(rxp - p[None, :], axis=-1)  # [Rx]
    Rtr = Rt[:, None] + Rr[None, :]  # [Tx,Rx]
    phase = -2j * np.pi * freqs[None, None, :] / CC * Rtr[:, :, None]
    return np.exp(phase)  # [Tx,Rx,nf]


def stage_a(cube, vp, tx_pos, rx_pos, freqs, sphere_r, num_check):
    fracs = []
    for v in range(num_check):
        u = vp[v] / np.linalg.norm(vp[v])
        p = u * sphere_r
        model = model_response(p, tx_pos[v], rx_pos[v], freqs)
        data = cube[v]
        alpha = (np.conj(model) * data).sum() / (np.abs(model) ** 2).sum()
        resid = data - alpha * model
        frac = 1 - (np.abs(resid) ** 2).sum() / (np.abs(data) ** 2).sum()
        fracs.append(frac)
    fracs = np.array(fracs)
    passed = bool(fracs.mean() >= 0.9)
    print(f"Stage A: per-view specular explained-fraction over {num_check} views: "
          f"mean={fracs.mean():.4f}, min={fracs.min():.4f}, max={fracs.max():.4f}  "
          f"-> {'PASS' if passed else 'FAIL'} (gate: mean >= 0.90)")
    return passed


def stage_b(cube, vp, tx_pos, rx_pos, freqs, sphere_r, n_views_list):
    v0 = 0
    u0 = vp[v0] / np.linalg.norm(vp[v0])
    p0 = u0 * sphere_r

    dirs = vp / np.linalg.norm(vp, axis=1, keepdims=True)
    cos_ang = dirs @ u0
    order = np.argsort(-cos_ang)
    ang_deg = np.degrees(np.arccos(np.clip(cos_ang[order], -1, 1)))

    contribs = np.empty(len(order), dtype=np.complex128)
    for i, w in enumerate(order):
        model_w = model_response(p0, tx_pos[w], rx_pos[w], freqs)
        contribs[i] = (np.conj(model_w) * cube[w]).sum()
    cumsum = np.cumsum(contribs)

    print("\nStage B: coherent multi-view accumulation at view-0's specular point")
    mags = {}
    for n in n_views_list:
        if n > len(cumsum):
            continue
        mag = abs(cumsum[n - 1])
        mags[n] = mag
        print(f"  n_views={n:5d} (max angle {ang_deg[n-1]:6.2f} deg): |coherent sum|={mag:.4e}")

    far_mask = ang_deg > 90
    far_idx = order[far_mask][:200]
    far_contribs = []
    for w in far_idx:
        model_w = model_response(p0, tx_pos[w], rx_pos[w], freqs)
        far_contribs.append((np.conj(model_w) * cube[w]).sum())
    far_contribs = np.array(far_contribs)
    print(f"  control: {len(far_contribs)} views >90deg from p0 direction: "
          f"|sum|={np.abs(far_contribs.sum()):.4e}, mean|contrib|={np.abs(far_contribs).mean():.4e}")

    ref_n = 5
    if ref_n not in mags:
        ref_n = min(mags, key=lambda k: abs(k - ref_n))
    ref_mag = mags[ref_n]
    lo, hi = 0.3 * ref_mag, 3.0 * ref_mag
    passed = all(lo <= m <= hi for n, m in mags.items() if n >= ref_n)
    print(f"  -> {'PASS' if passed else 'FAIL'} "
          f"(gate: sum stays within [0.3x,3x] of its n={ref_n} value ({ref_mag:.4e}) "
          f"for all larger n -- neither collapses nor diverges)")
    return passed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", default="data/pec_sphere_fmcw_16t16r_79ghz_r10m_2k.npz")
    p.add_argument("--num-check", type=int, default=40, help="Viewpoints checked in Stage A")
    p.add_argument("--n-views", type=int, nargs="+",
                   default=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
                   help="View-count checkpoints for Stage B")
    args = p.parse_args()

    cube, vp, tx_pos, rx_pos, meta = load_cube(args.npz)
    freqs = build_freqs(meta)
    sphere_r = float(meta["target_radius_m"])
    print(f"Loaded {args.npz}: {cube.shape[0]} views, target_radius={sphere_r}m, "
          f"fc={meta['radar_fc_hz']:.3e}Hz, bw={meta['radar_bandwidth_hz']:.3e}Hz, "
          f"n_adc={meta['num_adc_samples']}")

    ok_a = stage_a(cube, vp, tx_pos, rx_pos, freqs, sphere_r, args.num_check)
    ok_b = stage_b(cube, vp, tx_pos, rx_pos, freqs, sphere_r, args.n_views)

    ok = ok_a and ok_b
    print(f"\nOverall: {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
