#!/usr/bin/env python
"""R1b/R1c acceptance gate for point-target npz datasets (Option A).

Why this exists (EXPERIMENT_MANAGER_HANDOFF.md, 2026-07-20 Option A): the
existing acceptance scripts are sphere/origin-centric and cannot gate a
point-target dataset --
  * scripts/check_range_resolution.py sweeps only the origin->radar boresight
    ray (p=u*r), which never passes through an OFF-CENTER target, and it
    HARDCODES the -1 phase (assumes, never measures phase_sign);
  * scripts/validate_pec_sphere_coherence.py fixes a sphere specular point
    (sphere_r), undefined for a point target.
This script gates the point-target npz directly (operates on the npz data, not
a trained checkpoint), reading ground truth from `target_positions`.

Two checks, per DATASET_SPEC_POINT_TARGETS.md "Acceptance checks":

1. TARGET-AWARE RANGE CHECK + phase_sign MEASUREMENT.
   For each target t and several views, sweep candidate points along the
   radar-center->target ray p(r)=c_v+u*r and score the matched filter
   |sum_{tx,rx,f} conj(model)*data|. The peak must sit at r*=||c_v - t||
   (one-way range; FWHM ~ c/2B = 5-6 cm at 3 GHz). phase_sign is MEASURED, not
   assumed: the explained-power fraction at p=t is computed for BOTH
   conventions exp(phase_sign*i*k*R), and the winner is reported (expected -1).
   For R1c this runs over all six targets -> "all six ranges per view".

2. TARGET-AWARE MULTI-VIEW COHERENCE.
   Fix p at each KNOWN target point and accumulate matched-filter contributions
   across PERMUTED (angular-proximity-ordered) views. NB: unlike a sphere
   front-face (only near-specular views contribute, so its sum saturates and
   the [0.3x,3x] gate applies), an isotropic Born point accumulates COHERENTLY
   across all views -> |sum| GROWS ~linearly. So the gate here is instead:
   coherence efficiency eff=|sum contribs|/sum|contribs| is high at the TRUE
   point and drops at a +30 cm radial DECOY (localization), and the running sum
   never COLLAPSES. This is the point-target analogue of "neither collapses nor
   diverges".

Usage:
    python scripts/check_point_target_acceptance.py \
        --npz data/pec_pointtarget_fmcw_16t16r_79ghz_bw3ghz_r10m_500.npz
    python scripts/check_point_target_acceptance.py \
        --npz data/pec_pointconstellation_fmcw_16t16r_79ghz_bw3ghz_r10m_500.npz
"""
import argparse
import json

import numpy as np

CC = 299792458.0


def load_npz(npz_path):
    d = np.load(npz_path, mmap_mode="r", allow_pickle=True)   # mmap: read views lazily
    meta = json.loads(str(d["metadata_json"]))
    targets = np.asarray(d["target_positions"], dtype=np.float64)   # [T,3]
    return d, meta, targets


def build_freqs(meta):
    fc = float(meta["radar_fc_hz"])
    bw = float(meta["radar_bandwidth_hz"])
    n_adc = int(meta["num_adc_samples"])
    return (fc - bw / 2) + np.arange(n_adc) * (bw / n_adc)


def get_cube_v(d, v):
    """[Tx,Rx,n_adc] complex128 for view v (average the redundant chirp axis)."""
    return np.asarray(d["response"][v]).mean(axis=2).astype(np.complex128)


def model_response(p, txp, rxp, freqs, phase_sign):
    """exp(phase_sign * i * k * R_bistatic) -- same semantics as
    forward_operator_lessparallel's phase_sign (so a returned +1/-1 maps
    straight to the --phase-sign flag). phase_sign=-1 reproduces the recipe
    validated in validate_pec_sphere_coherence.py."""
    Rt = np.linalg.norm(txp - p[None, :], axis=-1)             # [Tx]
    Rr = np.linalg.norm(rxp - p[None, :], axis=-1)             # [Rx]
    Rtr = Rt[:, None] + Rr[None, :]                            # [Tx,Rx]
    phase = phase_sign * 1j * 2 * np.pi * freqs[None, None, :] / CC * Rtr[:, :, None]
    return np.exp(phase)                                       # [Tx,Rx,nf]


def explained_fraction(p, cube_v, txp, rxp, freqs, phase_sign):
    """Fraction of view power explained by a single scatterer at p under the
    given sign (closed-form complex projection, same as Stage A)."""
    model = model_response(p, txp, rxp, freqs, phase_sign)
    alpha = (np.conj(model) * cube_v).sum() / (np.abs(model) ** 2).sum()
    resid = cube_v - alpha * model
    return 1.0 - (np.abs(resid) ** 2).sum() / (np.abs(cube_v) ** 2).sum()


def range_profile(cube_v, txp, rxp, freqs, c_v, u, r_grid, phase_sign):
    prof = np.empty(len(r_grid))
    for i, r in enumerate(r_grid):
        model = model_response(c_v + u * r, txp, rxp, freqs, phase_sign)
        prof[i] = np.abs((np.conj(model) * cube_v).sum())
    return prof


def _fwhm(r_grid, prof):
    k = int(np.argmax(prof))
    half = prof[k] / 2.0
    lo = k
    while lo > 0 and prof[lo] > half:
        lo -= 1
    hi = k
    while hi < len(prof) - 1 and prof[hi] > half:
        hi += 1

    def cross(i0, i1):
        p0, p1 = prof[i0], prof[i1]
        if p1 == p0:
            return r_grid[i0]
        return r_grid[i0] + (half - p0) / (p1 - p0) * (r_grid[i1] - r_grid[i0])

    r_lo = cross(lo, lo + 1) if lo < k else r_grid[lo]
    r_hi = cross(hi - 1, hi) if hi > k else r_grid[hi]
    return r_grid[k], r_hi - r_lo


def measure_sign_and_ranges(d, vp, freqs, targets, view_idx, r_win, r_step, expect_sign):
    print("\n=== Check 1: phase_sign measurement + target-aware range profiles ===")
    # --- sign measurement: mean explained-fraction at each true target, both signs
    means = {}
    for s in (+1.0, -1.0):
        fr = []
        for v in view_idx:
            cube_v = get_cube_v(d, v)
            for t in targets:
                fr.append(explained_fraction(t, cube_v, d["tx_pos"][v], d["rx_pos"][v], freqs, s))
        means[s] = float(np.mean(fr))
    measured = max(means, key=means.get)
    print(f"  explained-fraction at true targets: phase_sign=+1 -> {means[+1.0]:.4f}, "
          f"-1 -> {means[-1.0]:.4f}  => MEASURED phase_sign = {measured:+.0f} "
          f"(expected {expect_sign:+.0f})")
    sign_ok = (measured == expect_sign)

    # --- range profiles (winning sign): peak vs r*=||c_v - t||, FWHM per target
    peak_errs, fwhms, all_ok = [], [], True
    for ti, t in enumerate(targets):
        pe, fw = [], []
        for v in view_idx:
            c_v = np.asarray(vp[v], dtype=np.float64)
            rstar = np.linalg.norm(c_v - t)
            u = (t - c_v) / rstar
            r_grid = np.arange(rstar - r_win, rstar + r_win, r_step)
            cube_v = get_cube_v(d, v)
            prof = range_profile(cube_v, d["tx_pos"][v], d["rx_pos"][v], freqs, c_v, u, r_grid, measured)
            peak_r, width = _fwhm(r_grid, prof)
            pe.append(peak_r - rstar)
            fw.append(width)
        pe, fw = np.array(pe), np.array(fw)
        peak_errs.append(np.abs(pe).mean())
        fwhms.append(fw.mean())
        ok = np.abs(pe).mean() < 0.02 and 0.02 < fw.mean() < 0.10   # <2cm peak err, 2-10cm FWHM
        all_ok &= ok
        print(f"  target {ti} r0={np.linalg.norm(t):.3f}m: mean|peak-r*|={np.abs(pe).mean()*100:.2f}cm, "
              f"mean FWHM={fw.mean()*100:.2f}cm  -> {'ok' if ok else 'CHECK'}")
    return sign_ok and all_ok, measured


def coherence_check(d, vp, freqs, targets, sign, n_list, max_views):
    print("\n=== Check 2: target-aware multi-view coherence (permuted views) ===")
    vp = np.asarray(vp, dtype=np.float64)
    dirs = vp / np.linalg.norm(vp, axis=1, keepdims=True)
    all_ok = True
    for ti, t in enumerate(targets):
        u_t = t / np.linalg.norm(t)
        order = np.argsort(-(dirs @ u_t))[:max_views]           # proximity-ordered = permuted
        decoy = t + 0.30 * u_t                                   # +30cm radial-outward decoy

        contribs = np.empty(len(order), dtype=np.complex128)
        contribs_decoy = np.empty(len(order), dtype=np.complex128)
        for i, v in enumerate(order):
            cube_v = get_cube_v(d, v)
            m = model_response(t, d["tx_pos"][v], d["rx_pos"][v], freqs, sign)
            contribs[i] = (np.conj(m) * cube_v).sum()
            md = model_response(decoy, d["tx_pos"][v], d["rx_pos"][v], freqs, sign)
            contribs_decoy[i] = (np.conj(md) * cube_v).sum()
        cumsum = np.abs(np.cumsum(contribs))
        eff_true = np.abs(contribs.sum()) / np.abs(contribs).sum()
        eff_decoy = np.abs(contribs_decoy.sum()) / np.abs(contribs_decoy).sum()

        traj = {n: cumsum[n - 1] for n in n_list if n <= len(cumsum)}
        ref = traj.get(5, cumsum[min(4, len(cumsum) - 1)])
        no_collapse = cumsum[-1] >= 0.5 * ref
        localizes = eff_true >= 2.0 * eff_decoy and eff_true >= 0.30
        ok = no_collapse and localizes
        all_ok &= ok
        traj_str = " ".join(f"n{n}={traj[n]:.2e}" for n in sorted(traj))
        print(f"  target {ti}: eff_true={eff_true:.3f} vs eff_decoy(+30cm)={eff_decoy:.3f}  "
              f"|sum| {traj_str}  -> {'ok' if ok else 'CHECK'}")
    return all_ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", required=True)
    p.add_argument("--num-views", type=int, default=6, help="views for the range/sign check")
    p.add_argument("--max-coh-views", type=int, default=300, help="views for the coherence accumulation")
    p.add_argument("--r-win", type=float, default=0.20, help="+/- range window around r* (m)")
    p.add_argument("--r-step", type=float, default=0.002)
    p.add_argument("--n-views", type=int, nargs="+", default=[1, 2, 5, 10, 50, 100, 300])
    p.add_argument("--expect-sign", type=float, default=-1.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    d, meta, targets = load_npz(args.npz)
    freqs = build_freqs(meta)
    vp = d["viewpoint_positions"]
    n_view = vp.shape[0]
    B = float(meta["radar_bandwidth_hz"])
    print(f"Loaded {args.npz}: {n_view} views, {len(targets)} targets, "
          f"fc={meta['radar_fc_hz']:.3e}Hz bw={B:.3e}Hz n_adc={meta['num_adc_samples']} "
          f"(range res c/2B = {CC/(2*B)*100:.1f}cm)")

    # permuted view subset for the range/sign check (Fibonacci order is a polar cap)
    rng = np.random.default_rng(args.seed)
    view_idx = np.sort(rng.choice(n_view, size=min(args.num_views, n_view), replace=False))

    ok1, measured = measure_sign_and_ranges(d, vp, freqs, targets, view_idx,
                                             args.r_win, args.r_step, args.expect_sign)
    ok2 = coherence_check(d, vp, freqs, targets, measured, args.n_views,
                          min(args.max_coh_views, n_view))

    ok = ok1 and ok2
    print(f"\nMEASURED phase_sign = {measured:+.0f}  |  Check1={'PASS' if ok1 else 'FAIL'}  "
          f"Check2={'PASS' if ok2 else 'FAIL'}  |  OVERALL {'PASS' if ok else 'FAIL'}")
    if measured != args.expect_sign:
        print(f"  !! phase_sign {measured:+.0f} != expected {args.expect_sign:+.0f} -- the Option-B "
              f"checkpoints (trained --phase-sign {args.expect_sign:+.0f}) would be INVALID.")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
