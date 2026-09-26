#!/usr/bin/env python
"""Acceptance + multi-view coherence validation for the PEC-cube FMCW npz.

Sibling of scripts/validate_pec_{sphere,tetrahedron}_coherence.py. The npz
format is identical (response [n_view,Tx,Rx,n_chirp,n_adc], ground-truth
tx_pos/rx_pos/viewpoint_positions, metadata_json, no stored frequency array --
f_i = fc - B/2 + i*(B/n_adc), empirically validated on the sphere data).
What differs from the sphere is the cube's physics:

- Sphere Stage A fit a single scatterer at unit(view)*radius and gated on
  explained power >= 0.9. For a cube there is no such universally-dominant
  point: face-normal views flash off a whole facet, oblique views are
  edge/corner diffraction, and the DOMINANT return generally sits behind
  the geometrically-nearest point (a weak convex corner). A single-point
  explained-fraction gate would fail on perfectly good data.
- Instead, Stage A gates on causality/geometry: the range-profile peak of
  every checked view must lie on the cube, i.e. its implied
  nearest-scatterer offset h = |vp| - peak_range must satisfy
      -sqrt(3)*a - tol <= h <= a*(|ux|+|uy|+|uz|) + tol
  where a is the half-side and a*|u|_1 is the axis-aligned cube's support
  function. This uses the npz's OWN viewpoint_positions, so it
  simultaneously checks the frequency-grid direction, the sign convention
  S(f)=exp(-j*2*pi*f/c*R) (a flipped sign puts peaks at acausal mirror
  ranges), |vp|, the half-side, and the axis-aligned orientation. Passing
  implies --phase-sign -1.0 in train.py, same as the sphere npz.
- Stage B (the actual multi-view coherence test) accumulates the matched
  filter at a FIXED 3D point across views. The probes are the 6 FACE
  CENTERS a*e_i (not corners): a flat PEC face at normal incidence is a
  strong, stable specular scatterer whose phase center sits at the face
  center, whereas a convex CORNER is a WEAK diffractor. An earlier revision
  of this script probed the best-aligned corner and FAILED on good data:
  corner contributions were near the noise floor (~2e-2 each, mean phase
  resultant 0.45) while facet-flash views entering at the 44-47 deg cone
  edge leaked |c|~87 contributions through matched-filter sidelobes,
  blowing the plateau gate on the divergence side. Same lesson as the
  tetrahedron validator (see its docstring). Gate per face: the running
  coherent sum over a cone about the face normal stays within [0.3x, 3x]
  of its 5-view value (non-collapse -- the old multi-view incoherence bug
  -- and non-divergence), and that plateau sits >20x above the per-view
  noise floor. A >90-degree control should sit near the floor.

Also reports (no gate) the per-view total-power census: a cube's per-view
power spans orders of magnitude (facet flash vs oblique diffraction), which
matters for interpreting power-weighted train/val metrics downstream.

Memory note: reads response.npy via O(1) seeks on the raw npz (ResponseReader
from the tetrahedron validator), so it never loads the 24GB array; runs in
well under a GB and a few minutes.

Usage:
    python scripts/validate_pec_cube_coherence.py \
        --npz data/pec_cube_fmcw_16t16r_79ghz_bw3ghz_r10m_2k.npz
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_pec_tetrahedron_coherence import (  # noqa: E402
    ResponseReader, build_freqs, model_response)

CC = 299792458.0


def resolve_half_side(meta, cli_half_side):
    if cli_half_side is not None:
        return float(cli_half_side), "--half-side CLI"
    for key, scale in (("target_half_side_m", 1.0), ("cube_half_side_m", 1.0),
                       ("target_side_m", 0.5), ("cube_side_m", 0.5),
                       ("target_edge_m", 0.5), ("target_radius_m", 1.0)):
        if key in meta:
            return float(meta[key]) * scale, f"metadata[{key!r}]" + ("*0.5" if scale == 0.5 else "")
    raise SystemExit("No cube size in metadata (tried target_half_side_m/cube_half_side_m/"
                     "target_side_m/cube_side_m/target_edge_m/target_radius_m) -- "
                     "pass --half-side explicitly.")


def stage_a(reader, half_side, meta, num_check, seed):
    bw = float(meta["radar_bandwidth_hz"])
    rres = CC / (2 * bw)
    tol = 1.5 * rres  # peak-bin quantization + a little slack
    # PITFALL (see rift-shell-bias-diagnosis memory): npz views are
    # elevation-ordered; contiguous indices are one polar cap. Sample spread.
    rng = np.random.default_rng(seed)
    checked = rng.permutation(reader.n_view)[:num_check]

    n_bad = 0
    h_all, support_all, power_all, a_lower = [], [], [], 0.0
    for v in checked:
        prof = np.abs(np.fft.ifft(reader.profile_00(v)))
        peak_range = np.argmax(prof) * rres  # sphere-validated bin convention
        r_vp = np.linalg.norm(reader.vp[v])
        u = reader.vp[v] / r_vp
        h = r_vp - peak_range  # implied nearest-scatterer offset from origin
        support = half_side * np.abs(u).sum()  # nearest possible surface point
        ok = (-np.sqrt(3) * half_side - tol) <= h <= (support + tol)
        n_bad += 0 if ok else 1
        h_all.append(h)
        support_all.append(support)
        power_all.append((np.abs(reader.view_avg(v)) ** 2).sum())
        a_lower = max(a_lower, h / np.abs(u).sum())
    h_all, power_all = np.array(h_all), np.array(power_all)

    frac_ok = 1 - n_bad / len(checked)
    passed = frac_ok >= 0.95
    print(f"Stage A: range-peak-on-cube gate over {len(checked)} spread views "
          f"(half-side a={half_side}m, tol={tol*100:.1f}cm):")
    print(f"  peak offsets h=|vp|-peak_range: min={h_all.min():.3f} max={h_all.max():.3f} m "
          f"(support bound a*|u|_1 in [{min(support_all):.3f},{max(support_all):.3f}])")
    print(f"  data-driven half-side lower bound max(h/|u|_1) = {a_lower:.3f} m "
          f"(should be ~a; >a+tol means wrong a/orientation)")
    print(f"  {frac_ok:.1%} of views within bounds -> {'PASS' if passed else 'FAIL'} (gate >= 95%)")
    print(f"  NOTE: causal peaks under f_i=fc-B/2+i*dF + np.fft.ifft confirm "
          f"S(f)=exp(-j*2*pi*f/c*R): use --phase-sign -1.0 in train.py (same as sphere npz).")

    p_sorted = np.sort(power_all)[::-1]
    print(f"  per-view power census ({len(checked)} views): median={np.median(power_all):.3e}, "
          f"max={power_all.max():.3e} ({power_all.max()/np.median(power_all):.0f}x median); "
          f"top-5 views carry {p_sorted[:5].sum()/power_all.sum():.1%} of sampled power "
          f"(facet-flash dynamic range -- expect flashy views to dominate power-weighted metrics)")
    return passed


def _coherent_curve(reader, freqs, probe, order):
    contribs = np.empty(len(order), dtype=np.complex128)
    for i, w in enumerate(order):
        model_w = model_response(probe, reader.tx_pos[w], reader.rx_pos[w], freqs)
        contribs[i] = (np.conj(model_w) * reader.view_avg(w)).sum()
    return contribs


def stage_b(reader, dirs, freqs, half_side, n_views_list, cone_deg,
            max_cone_views, n_control):
    normals = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                        [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    print(f"\nStage B: coherent multi-view accumulation at the 6 FACE CENTERS "
          f"(a={half_side}m; convex corners are weak diffractors on real PEC "
          f"returns, so faces -- not corners -- are the strong stable scatterers).")

    face_pass = []
    for fi, n_hat in enumerate(normals):
        c = half_side * n_hat  # face center
        cos_ang = dirs @ n_hat
        in_cone = np.where(cos_ang >= np.cos(np.radians(cone_deg)))[0]
        order = in_cone[np.argsort(-cos_ang[in_cone])]
        if max_cone_views and len(order) > max_cone_views:
            order = order[:max_cone_views]
        ang_deg = np.degrees(np.arccos(np.clip(cos_ang[order], -1, 1)))
        contribs = _coherent_curve(reader, freqs, c, order)
        cumsum = np.cumsum(contribs)

        mags = {n: abs(cumsum[n - 1]) for n in n_views_list if n <= len(cumsum)}
        print(f"  Face {fi} normal {n_hat.astype(int).tolist()} center "
              f"{c.round(3).tolist()} (best view {order[0]}, {ang_deg[0]:.2f} deg off normal):")
        for n in sorted(mags):
            print(f"    n_views={n:4d} (<= {ang_deg[n-1]:5.2f} deg): "
                  f"|coherent sum|={mags[n]:.4e}")
        # noise floor = median individual contribution beyond the flash cone
        floor = np.median(np.abs(contribs[max(5, len(contribs)//2):])) if len(contribs) > 6 else 0.0
        ref_n = 5 if 5 in mags else max(mags)
        ref_mag = mags[ref_n]
        non_collapse = all(0.3 * ref_mag <= m <= 3.0 * ref_mag
                           for n, m in mags.items() if n >= ref_n)
        above_floor = ref_mag > 20 * floor if floor > 0 else True
        ok = non_collapse and above_floor
        print(f"    plateau {ref_mag:.3e} vs per-view floor {floor:.3e} "
              f"(ratio {ref_mag/floor:.0f}x); non-collapse={non_collapse} "
              f"above-floor={above_floor} -> {'PASS' if ok else 'FAIL'}")
        face_pass.append(ok)

    # control: views >90deg from Face 0's normal, probed at Face 0's center
    n0 = normals[0]
    far = np.where(dirs @ n0 < 0)[0][:n_control]
    far_c = _coherent_curve(reader, freqs, half_side * n0, far)
    print(f"  control: {len(far_c)} views >90deg from Face 0 normal: "
          f"|sum|={np.abs(far_c.sum()):.4e}, mean|contrib|={np.abs(far_c).mean():.4e} "
          f"(should sit near the per-view floor)")

    passed = sum(face_pass) >= 5
    print(f"  -> {'PASS' if passed else 'FAIL'} "
          f"({sum(face_pass)}/6 faces hold their coherent plateau; gate >= 5/6)")
    return passed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", default="data/pec_cube_fmcw_16t16r_79ghz_bw3ghz_r10m_2k.npz")
    p.add_argument("--half-side", type=float, default=None,
                   help="Cube half-side in m; overrides metadata")
    p.add_argument("--num-check", type=int, default=200, help="Viewpoints checked in Stage A")
    p.add_argument("--cone-deg", type=float, default=30.0,
                   help="Stage B uses views within this angle of each face normal")
    p.add_argument("--n-views", type=int, nargs="+", default=[1, 2, 5, 10, 20, 40],
                   help="View-count checkpoints for Stage B")
    p.add_argument("--max-cone-views", type=int, default=40,
                   help="Cap each face's cone to the nearest N views (I/O trim; "
                        "0 = read all cone views). 40 is plenty for non-collapse.")
    p.add_argument("--n-control", type=int, default=40,
                   help="Number of >90deg control views for Stage B")
    p.add_argument("--seed", type=int, default=0, help="Stage A view-sampling seed")
    args = p.parse_args()

    reader = ResponseReader(args.npz)
    meta = reader.meta
    freqs = build_freqs(meta)
    half_side, source = resolve_half_side(meta, args.half_side)
    dirs = reader.vp / np.linalg.norm(reader.vp, axis=1, keepdims=True)
    print(f"Loaded {args.npz}: {reader.n_view} views, fc={meta['radar_fc_hz']:.3e}Hz, "
          f"bw={meta['radar_bandwidth_hz']:.3e}Hz, n_adc={meta['num_adc_samples']}")
    print(f"Cube half-side: {half_side}m (from {source})")
    print(f"metadata_json: {json.dumps(meta)}")

    ok_a = stage_a(reader, half_side, meta, args.num_check, args.seed)
    ok_b = stage_b(reader, dirs, freqs, half_side, args.n_views, args.cone_deg,
                   args.max_cone_views, args.n_control)

    ok = ok_a and ok_b
    print(f"\nOverall: {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
