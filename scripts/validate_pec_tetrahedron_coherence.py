#!/usr/bin/env python
"""Acceptance + multi-view coherence validation for the PEC-tetrahedron FMCW npz.

Sibling of scripts/validate_pec_{sphere,cube}_coherence.py. The npz format is
identical (response [n_view,Tx,Rx,n_chirp,n_adc], ground-truth
tx_pos/rx_pos/viewpoint_positions, metadata_json, no stored frequency array --
f_i = fc - B/2 + i*(B/n_adc), empirically validated on the sphere/cube data).
What is NEW here is the target's lack of symmetry:

- The cube's octahedral symmetry made "axis-aligned, centered at origin" an
  unambiguous orientation, so its validator could HARDCODE the support function
  a*|u|_1. A tetrahedron has no such luck: its orientation in the simulator's
  frame is a genuine unknown (empirically this dataset has one vertex near +z
  and the opposite face down, centroid at origin, but nothing pins that a
  priori). So Stage A here RECOVERS the orientation from the data before gating.

- Recovery: for a convex target the range-profile peak of view u sits at the
  nearest surface point, whose offset from the origin along u is the support
  function h(u) = max_i (v_i . u) over the vertices. The views where h(u) is
  largest (~R_circ) point straight at a vertex. Clustering the highest-h view
  directions therefore recovers the 4 vertex directions; scaling by the
  circumradius R = 3/4 * height (regular-tetra geometry) gives the vertices.
  The recovery is then SELF-CHECKED against known geometry: exactly 4 clusters,
  pairwise angles ~109.47 deg (dots ~ -1/3), and recovered edge == the
  metadata's target_edge_m. This simultaneously validates size, regularity,
  orientation-consistency, |vp|, the frequency-grid direction, and the sign
  convention S(f)=exp(-j*2*pi*f/c*R) (a flipped sign puts peaks at acausal
  mirror ranges). Passing implies --phase-sign -1.0 in train.py (same as
  sphere/cube).

- Stage A's causality gate then requires every checked view's peak offset to
  satisfy  -R - tol <= h <= support(u) + tol, where support(u) is evaluated
  against the RECOVERED vertices -- a peak closer than the support bound is
  impossible, a peak beyond the far silhouette means wrong geometry.

- Stage B (the actual multi-view coherence test) accumulates the matched
  filter at a FIXED 3D point across views, like the sphere/cube. The probe is a
  FACE CENTROID (not a vertex): a flat PEC face at normal incidence is a
  strong, stable specular scatterer whose phase center sits at the centroid,
  whereas a convex VERTEX is a WEAK diffractor (an apex-vertex probe measures
  only sidelobe noise -- verified empirically on this data, and the reason the
  earlier corner-probe design failed on real PEC returns). Each of the 4 face
  centroids c_i = -inradius * (v_i/|v_i|) is probed over a cone about its
  outward normal n_i = -(v_i/|v_i|). Because a facet flash is NARROW (a few
  degrees), the coherent sum builds over the first few views and then must
  PLATEAU as further (non-flashing) cone views are added -- collapse toward
  zero there is the old multi-view incoherence bug. Gate: at least 3 of the 4
  faces keep their running sum within [0.3x, 3x] of its 5-view value for all
  larger n (non-collapse), with that plateau far above the per-view noise
  floor. A >90-degree control per face should sit near that floor.

Memory note: this reads only the tx0/rx0 range profiles (Stage A) and the
cone-view channels (Stage B) by SEEKING into the uncompressed npy inside the
zip, so it never loads the full 24GB `response` into RAM (unlike the earlier
sphere/cube validators). Runs in well under a GB.

Usage:
    python scripts/validate_pec_tetrahedron_coherence.py \
        --npz data/pec_tetrahedron_fmcw_16t16r_79ghz_bw3ghz_r10m_h2m_2k.npz
"""
import argparse
import io
import json
import zipfile

import numpy as np

CC = 299792458.0


class ResponseReader:
    """Seek-based per-view reader for the uncompressed `response.npy` member.

    response is C-contiguous [n_view, Tx, Rx, n_chirp, n_adc] complex64.
    view_view(v) returns the chirp-averaged [Tx, Rx, n_adc] complex128 block;
    profile_00(v) returns just the tx0/rx0 chirp0 range profile magnitude.
    """

    def __init__(self, npz_path):
        self.z = zipfile.ZipFile(npz_path)
        # metadata / geometry members are small -- read them fully
        self.meta = json.loads(str(self._rd("metadata_json.npy")))
        self.vp = self._rd("viewpoint_positions.npy")
        self.tx_pos = self._rd("tx_pos.npy")
        self.rx_pos = self._rd("rx_pos.npy")
        self.Tx, self.Rx, self.n_chirp, self.n_adc = self.meta["frame_shape"]
        self.n_view = self.meta["frame_count"]
        # locate the raw data offset inside the (stored, uncompressed) npy.
        # Seek on the RAW file, not a ZipExtFile: ZipExtFile.seek() on this
        # Python re-reads the stream from the start on backward seeks (and
        # read-discards on forward ones), turning random view access into
        # multi-GB scans. Stored members are byte-identical in the raw zip,
        # so absolute offsets give true O(1) seeks.
        info = self.z.getinfo("response.npy")
        assert info.compress_type == zipfile.ZIP_STORED, "response.npy is compressed"
        raw = open(npz_path, "rb")
        raw.seek(info.header_offset)
        lh = raw.read(30)
        assert lh[:4] == b"PK\x03\x04", "bad zip local header"
        fnlen = int.from_bytes(lh[26:28], "little")
        extralen = int.from_bytes(lh[28:30], "little")
        payload_off = info.header_offset + 30 + fnlen + extralen
        raw.seek(payload_off)
        magic = raw.read(10)
        assert magic[:6] == b"\x93NUMPY", "response.npy is not a .npy stream"
        hlen = int.from_bytes(magic[8:10], "little")
        self.data_off = payload_off + 10 + hlen
        self._f = raw
        self.view_stride = self.Tx * self.Rx * self.n_chirp * self.n_adc * 8

    def _rd(self, name):
        with self.z.open(name) as f:
            return np.load(io.BytesIO(f.read()), allow_pickle=True)

    def profile_00(self, v):
        self._f.seek(self.data_off + v * self.view_stride)
        buf = self._f.read(self.n_adc * 8)  # tx0,rx0,chirp0 sits at the block start
        return np.frombuffer(buf, dtype=np.complex64).astype(np.complex128)

    def view_avg(self, v):
        """Chirp-averaged [Tx, Rx, n_adc] complex128 for one view."""
        self._f.seek(self.data_off + v * self.view_stride)
        buf = self._f.read(self.view_stride)
        block = np.frombuffer(buf, dtype=np.complex64).reshape(
            self.Tx, self.Rx, self.n_chirp, self.n_adc)
        return block.mean(axis=2).astype(np.complex128)


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


def all_peak_offsets(reader, rres):
    """h(v) = |vp_v| - peak_range for every view, from the tx0/rx0 profile."""
    n = reader.n_view
    h = np.empty(n)
    for v in range(n):
        prof = np.abs(np.fft.ifft(reader.profile_00(v)))
        peak_range = np.argmax(prof) * rres
        h[v] = np.linalg.norm(reader.vp[v]) - peak_range
    return h


def recover_vertices(dirs, h, R, h_frac):
    """Greedy-merge the highest-h view directions into 4 vertex directions.

    Returns (verts [<=4?,3] scaled to R, vertex_dirs, n_clusters_found).
    """
    thresh = h_frac * h.max()
    cand = np.where(h >= thresh)[0]
    cand = cand[np.argsort(-h[cand])]
    reps, wts = [], []
    for ci in cand:
        d = dirs[ci]
        for k, r in enumerate(reps):
            if r @ d > 0.8:  # same vertex lobe (vertices are 109 deg apart)
                reps[k] = (r * wts[k] + d)
                reps[k] /= np.linalg.norm(reps[k])
                wts[k] += 1
                break
        else:
            reps.append(d.copy())
            wts.append(1)
    n_found = len(reps)
    order = np.argsort(-np.array(wts))
    vdirs = np.array(reps)[order][:4]
    return R * vdirs, vdirs, n_found


def stage_a(reader, dirs, h, freqs, meta, num_check, seed, h_frac):
    bw = float(meta["radar_bandwidth_hz"])
    rres = CC / (2 * bw)
    tol = 1.5 * rres
    height = float(meta["target_height_m"])
    R = 0.75 * height  # regular-tetra circumradius = 3/4 * height
    edge_meta = float(meta["target_edge_m"])

    verts, vdirs, n_found = recover_vertices(dirs, h, R, h_frac)
    G = vdirs @ vdirs.T
    offdiag = G[np.triu_indices(len(vdirs), 1)]
    angles = np.degrees(np.arccos(np.clip(offdiag, -1, 1)))
    edges = [np.linalg.norm(verts[i] - verts[j])
             for i in range(len(verts)) for j in range(i + 1, len(verts))]
    edge_mean = float(np.mean(edges)) if edges else 0.0

    print(f"Stage A: data-driven orientation recovery + causality gate "
          f"(R_circ={R:.3f}m from height={height}m, tol={tol*100:.1f}cm)")
    print(f"  clusters found among top-h views (>= {h_frac:.0%} of h_max): {n_found} "
          f"(expect exactly 4 vertices)")
    print(f"  recovered vertex directions:")
    for vd in vdirs:
        print(f"    {vd.round(4).tolist()}")
    print(f"  pairwise vertex angles: {angles.round(2).tolist()} deg (regular tetra = 109.47)")
    print(f"  recovered mean edge = {edge_mean:.3f} m vs metadata target_edge_m = {edge_meta:.3f} m")
    apex = verts[np.argmax(verts[:, 2])]
    print(f"  apex vertex (max z): {apex.round(3).tolist()}; base z-values: "
          f"{sorted(np.round(verts[:, 2], 3).tolist())}")

    # Recovery self-checks
    ok_count = (n_found == 4)
    ok_reg = (len(angles) == 6 and np.all(np.abs(angles - 109.47) <= 8.0))
    ok_edge = abs(edge_mean - edge_meta) <= 0.10 * edge_meta

    # Causality gate over spread views (npz views are elevation-ordered --
    # sample a permutation, see rift-shell-bias-diagnosis memory).
    rng = np.random.default_rng(seed)
    checked = rng.permutation(reader.n_view)[:num_check]
    n_bad = 0
    h_chk, sup_chk = [], []
    for v in checked:
        support = float((verts @ dirs[v]).max())  # h_T(u) against recovered verts
        ok = (-R - tol) <= h[v] <= (support + tol)
        n_bad += 0 if ok else 1
        h_chk.append(h[v])
        sup_chk.append(support)
    frac_ok = 1 - n_bad / len(checked)
    ok_causal = frac_ok >= 0.95

    print(f"  causality over {len(checked)} spread views: "
          f"h in [{min(h_chk):.3f},{max(h_chk):.3f}] vs support in "
          f"[{min(sup_chk):.3f},{max(sup_chk):.3f}]; {frac_ok:.1%} within "
          f"[-R-tol, support+tol] (gate >= 95%)")
    print(f"  NOTE: causal peaks under f_i=fc-B/2+i*dF + np.fft.ifft confirm "
          f"S(f)=exp(-j*2*pi*f/c*R): use --phase-sign -1.0 in train.py (same as sphere/cube).")

    passed = ok_count and ok_reg and ok_edge and ok_causal
    print(f"  sub-gates: 4-clusters={ok_count} regularity={ok_reg} "
          f"edge-match={ok_edge} causality={ok_causal} -> {'PASS' if passed else 'FAIL'}")
    return passed, verts


def _coherent_curve(reader, freqs, probe, order):
    contribs = np.empty(len(order), dtype=np.complex128)
    for i, w in enumerate(order):
        model_w = model_response(probe, reader.tx_pos[w], reader.rx_pos[w], freqs)
        contribs[i] = (np.conj(model_w) * reader.view_avg(w)).sum()
    return contribs


def stage_b(reader, dirs, verts, freqs, n_views_list, cone_deg, max_cone_views, n_control):
    R = np.linalg.norm(verts, axis=1).mean()
    inradius = R / 3.0  # regular tetra: inradius = circumradius / 3
    vdirs = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    print(f"\nStage B: coherent multi-view accumulation at the 4 FACE CENTROIDS "
          f"(inradius={inradius:.3f}m; convex vertices are weak diffractors, so "
          f"faces -- not vertices -- are the strong stable scatterers).")

    face_pass = []
    for fi in range(len(vdirs)):
        n_hat = -vdirs[fi]                # outward face normal
        c = -inradius * vdirs[fi]         # face centroid
        cos_ang = dirs @ n_hat
        in_cone = np.where(cos_ang >= np.cos(np.radians(cone_deg)))[0]
        order = in_cone[np.argsort(-cos_ang[in_cone])]
        if max_cone_views and len(order) > max_cone_views:
            order = order[:max_cone_views]
        ang_deg = np.degrees(np.arccos(np.clip(cos_ang[order], -1, 1)))
        contribs = _coherent_curve(reader, freqs, c, order)
        cumsum = np.cumsum(contribs)

        mags = {n: abs(cumsum[n - 1]) for n in n_views_list if n <= len(cumsum)}
        best_view = order[0]
        print(f"  Face {fi} normal {n_hat.round(3).tolist()} centroid "
              f"{c.round(3).tolist()} (best view {best_view}, {ang_deg[0]:.2f} deg off normal):")
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

    # one control per the strongest face: views >90deg from its normal
    n0 = -vdirs[0]
    cos0 = dirs @ n0
    far = np.where(cos0 < 0)[0][:n_control]
    far_c = _coherent_curve(reader, freqs, -inradius * vdirs[0], far)
    print(f"  control: {len(far_c)} views >90deg from Face 0 normal: "
          f"|sum|={np.abs(far_c.sum()):.4e}, mean|contrib|={np.abs(far_c).mean():.4e} "
          f"(should sit near the per-view floor)")

    passed = sum(face_pass) >= 3
    print(f"  -> {'PASS' if passed else 'FAIL'} "
          f"({sum(face_pass)}/4 faces hold their coherent plateau; gate >= 3/4)")
    return passed


def power_census(reader, seed, num_check):
    rng = np.random.default_rng(seed + 1)
    checked = rng.permutation(reader.n_view)[:num_check]
    power = np.array([(np.abs(reader.view_avg(v)) ** 2).sum() for v in checked])
    p_sorted = np.sort(power)[::-1]
    print(f"\nPer-view power census ({num_check} spread views): "
          f"median={np.median(power):.3e}, max={power.max():.3e} "
          f"({power.max()/np.median(power):.0f}x median); top-5 carry "
          f"{p_sorted[:5].sum()/power.sum():.1%} of sampled power "
          f"(facet-flash dynamic range -- expect flashy views to dominate "
          f"power-weighted metrics, same caveat as the cube).")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz",
                   default="data/pec_tetrahedron_fmcw_16t16r_79ghz_bw3ghz_r10m_h2m_2k.npz")
    p.add_argument("--num-check", type=int, default=200,
                   help="Viewpoints checked in Stage A causality gate + power census")
    p.add_argument("--cone-deg", type=float, default=30.0,
                   help="Stage B uses views within this angle of each face normal")
    p.add_argument("--n-views", type=int, nargs="+",
                   default=[1, 2, 5, 10, 20, 40],
                   help="View-count checkpoints for Stage B")
    p.add_argument("--max-cone-views", type=int, default=40,
                   help="Cap each face's cone to the nearest N views (I/O trim; "
                        "0 = read all cone views). 40 is plenty for non-collapse.")
    p.add_argument("--n-control", type=int, default=40,
                   help="Number of >90deg control views for Stage B")
    p.add_argument("--h-frac", type=float, default=0.85,
                   help="Views with h >= h_frac*h_max seed the vertex clustering")
    p.add_argument("--seed", type=int, default=0, help="Stage A view-sampling seed")
    args = p.parse_args()

    reader = ResponseReader(args.npz)
    meta = reader.meta
    freqs = build_freqs(meta)
    rres = CC / (2 * float(meta["radar_bandwidth_hz"]))
    dirs = reader.vp / np.linalg.norm(reader.vp, axis=1, keepdims=True)
    print(f"Loaded {args.npz}: {reader.n_view} views, fc={meta['radar_fc_hz']:.3e}Hz, "
          f"bw={meta['radar_bandwidth_hz']:.3e}Hz, n_adc={meta['num_adc_samples']}")
    print(f"target: {meta['target_type']}, height={meta['target_height_m']}m, "
          f"edge={meta['target_edge_m']:.3f}m")
    print(f"metadata_json: {json.dumps(meta)}")

    h = all_peak_offsets(reader, rres)
    ok_a, verts = stage_a(reader, dirs, h, freqs, meta, args.num_check,
                          args.seed, args.h_frac)
    ok_b = stage_b(reader, dirs, verts, freqs, args.n_views, args.cone_deg,
                   args.max_cone_views, args.n_control)
    power_census(reader, args.seed, min(args.num_check, 80))

    ok = ok_a and ok_b
    print(f"\nOverall: {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
