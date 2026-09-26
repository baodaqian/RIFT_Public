#!/usr/bin/env python
"""Experiment B (shell-misfit + generalization upper bound): how well does an
IDEAL thin uniform shell of isotropic scatterers explain the real bw3ghz
sphere data, as a function of shell radius, on train views AND held-out views?

Two questions, one cheap inference sweep (see EXPERIMENT_MANAGER_HANDOFF.md
2026-07-13 "Point-target PSF calibration + shell-bias attribution", extended
per the 2026-07-13 regularization-round decision):

1. BIAS ATTRIBUTION -- which radius fits the measured data best?
   r=1.00 best -> the trained scenes' +8-10cm outward shell is an
   optimization/init artifact (inherited from backprojection), fixable by
   priors/init. r=1.08 best -> the Born point-scatterer model itself prefers
   an outward-shifted density for this extended specular target (model
   mismatch; calibrate it out instead).

2. GENERALIZATION UPPER BOUND -- the trained scenes are UNCORRELATED with
   held-out views (optimal-gain rel-MSE ~100%). If the ideal shell scores
   well below 100% on the VAL split, a physically-sane scene does generalize
   and regularized training has a concrete, reachable target (its val number
   ~= the best a smooth isotropic scene can do). If even the ideal shell
   sits at ~100% on val, priors cannot fix generalization -- the Born
   isotropic model itself can't predict unseen views of this target.

Shells are CONTINUOUS Fibonacci-lattice point sets at exact radii (not 48^3
voxel shells: at 6.25cm pitch, half-a-cell tolerance makes the r=1.00 and
r=1.04 voxel shells share most of their voxels -- no radius discrimination).
Uniform unit weights; a single least-squares-optimal complex gain per
(radius, split), pooled across that split's views -- the same global-gain
semantics as training's GlobalComplexGain, so rel-MSE numbers are directly
comparable to the training logs.

CAUTION (project memory): npz viewpoint indices are elevation-ordered; always
select views through the seed-42 permutation, never contiguous ranges. This
script draws train views from perm[:750] and val views from perm[750:800] --
train.py's exact split.

Usage:
    python scripts/eval_ideal_shell_misfit.py --device cuda \
        --num-views 25 --out training_checkpoints/ideal_shell_misfit_results.csv
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rift.config import cc
from rift.forward_operator import get_kvector
from rift.range_operator import range_forward_operator

NPZ_DEFAULT = "data/pec_sphere_fmcw_16t16r_79ghz_bw3ghz_r10m_2k.npz"
PHASE_SIGN, SEED = -1.0, 42


def fibonacci_shell(n_points, radius, device):
    i = torch.arange(n_points, dtype=torch.float64, device=device) + 0.5
    z = 1.0 - 2.0 * i / n_points
    az = i * np.pi * (3.0 - np.sqrt(5.0))
    rho = torch.sqrt((1.0 - z ** 2).clamp_min(0.0))
    pos = radius * torch.stack([rho * torch.cos(az), rho * torch.sin(az), z], dim=-1)
    return pos.to(torch.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz-path", default=NPZ_DEFAULT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--radii", type=float, nargs="+",
                   default=[0.92, 0.96, 1.00, 1.04, 1.08, 1.12])
    p.add_argument("--n-points", type=int, default=20000,
                   help="Fibonacci points per shell (~2.5cm spacing at r=1m for 20k -- "
                        "below the 5cm range PSF, i.e. effectively continuous)")
    p.add_argument("--num-views", type=int, default=25, help="views per split")
    p.add_argument("--out", default="training_checkpoints/ideal_shell_misfit_results.csv")
    args = p.parse_args()
    dev = torch.device(args.device)

    print(f"loading {args.npz_path} ...", flush=True)
    d = np.load(args.npz_path, allow_pickle=True)
    meta = json.loads(str(d["metadata_json"]))
    response = d["response"]
    tx_pos_all = d["tx_pos"]; rx_pos_all = d["rx_pos"]
    n_view = response.shape[0]
    fc, bw, n_adc = (float(meta["radar_fc_hz"]), float(meta["radar_bandwidth_hz"]),
                     int(meta["num_adc_samples"]))
    freqs = torch.tensor((fc - bw / 2) + np.arange(n_adc) * (bw / n_adc),
                         dtype=torch.float32, device=dev)
    kvec = get_kvector(freqs, cc)

    perm = np.random.default_rng(SEED).permutation(n_view)
    splits = {
        "train": perm[:750][:args.num_views],
        "val": perm[750:800][:args.num_views],
    }
    for name, idx in splits.items():
        print(f"{name}: {len(idx)} views {idx.tolist()}", flush=True)

    def meas_cube(v):
        cube = response[v].mean(axis=2)  # [Tx, Rx, nf] complex64, chirp-averaged
        return torch.tensor(cube, dtype=torch.complex128, device=dev).permute(1, 0, 2)

    def render_view(v, pos, w):
        rx = torch.tensor(rx_pos_all[v], dtype=torch.float32, device=dev)
        tx = torch.tensor(tx_pos_all[v], dtype=torch.float32, device=dev)
        with torch.no_grad():
            S = range_forward_operator(freqs, kvec, rx, tx, pos, w,
                                       phase_sign=PHASE_SIGN, compute_dtype=torch.float64)
        return S.permute(1, 2, 0)  # [Rx, Tx, nf]

    rows = []
    for radius in args.radii:
        pos = fibonacci_shell(args.n_points, radius, dev)
        w = torch.ones(args.n_points, dtype=torch.complex64, device=dev)
        for split_name, view_idx in splits.items():
            ps = pp = 0.0 + 0.0j
            den = 0.0
            t0 = time.time()
            for v in view_idx:
                S_meas = meas_cube(v)
                S_pred = render_view(v, pos, w)
                ps += (S_pred.conj() * S_meas).sum().item()
                pp += (S_pred.conj() * S_pred).sum().item()
                den += (S_meas.abs() ** 2).sum().item()
            rel_opt = 1.0 - (abs(ps) ** 2 / (pp.real * den))
            g_opt = ps / pp.real
            per_view = (time.time() - t0) / len(view_idx)
            print(f"r={radius:.2f}m {split_name:5s} ({len(view_idx)} views): "
                  f"optimal-gain rel-MSE={rel_opt:.1%}  |g_opt|={abs(g_opt):.4e}  "
                  f"[{per_view:.1f}s/view]", flush=True)
            rows.append({"radius_m": radius, "split": split_name, "views": len(view_idx),
                         "rel_mse_optimal_gain": rel_opt, "g_opt_abs": abs(g_opt)})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nresults written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
