#!/usr/bin/env python
"""Does trilinearly densifying a trained voxel scene reduce HELD-OUT validation error?

Renders the range forward operator from a trained g48 checkpoint's scene as-is
(48^3) and from the same scene trilinearly interpolated to finer grids
(96^3/144^3/192^3), against train.py's exact validation split (seed 42,
views perm[750:800] of the bw3ghz npz). Complex weights are interpolated
re/im separately and divided by the volume ratio (density-preserving);
positions are the fine grid's voxel centers (align_corners=False interpolation
lands exactly on generate_dynamic_grid centers).

Per variant, reports rel-MSE two ways:
  - trained-gain: with the checkpoint's GlobalComplexGain (train.py's metric;
    the 48^3 rows should reproduce the training logs' val rel-MSE)
  - optimal-gain: with the single least-squares-optimal complex gain over the
    evaluated views (scale-free -- isolates waveform correlation from scale)

Usage (GPU strongly recommended; CPU is ~131 s/view at 96^3):
    python scripts/eval_interpolated_scene_val.py --device cuda \
        --num-views 50 --factors 1 2 3 4 \
        --out training_checkpoints/interp_val_eval_results.csv
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rift.config import cc
from rift.encoding import generate_dynamic_grid
from rift.forward_operator import get_kvector
from rift.npz_dataset import _direction_to_theta_phi
from rift.range_operator import range_forward_operator
from rift.sparse_scene import SHVoxelGridScene

NPZ_DEFAULT = "data/pec_sphere_fmcw_16t16r_79ghz_bw3ghz_r10m_2k.npz"
CKPTS = {
    "grid": "training_checkpoints/pec_sphere_recon_bw3ghz_grid_g48/checkpoint_best.pth.tar",
    "grid_sh": "training_checkpoints/pec_sphere_recon_bw3ghz_gridsh6_g48/checkpoint_best.pth.tar",
}
EXTENT, PHASE_SIGN, SEED, G_COARSE = 1.5, -1.0, 42, 48


def trained_gain(ck):
    g = ck["gain_state_dict"]
    return torch.polar(torch.exp(g["log_mag"]), g["phase"]).to(torch.complex128).item()


def upsample_complex(w, gfine):
    """[G,G,G] complex -> [gfine^3] complex at voxel centers, density-preserving."""
    g = w.shape[0]
    re = F.interpolate(w.real[None, None].double(), size=(gfine,) * 3,
                       mode="trilinear", align_corners=False)[0, 0]
    im = F.interpolate(w.imag[None, None].double(), size=(gfine,) * 3,
                       mode="trilinear", align_corners=False)[0, 0]
    return (torch.complex(re, im) / (gfine / g) ** 3).reshape(-1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz-path", default=NPZ_DEFAULT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-views", type=int, default=50,
                   help="how many of the 50 val views to use (first K in split order)")
    p.add_argument("--factors", type=int, nargs="+", default=[1, 2, 3, 4],
                   help="upsampling factors of the 48^3 scene (1 = baseline)")
    p.add_argument("--reprs", nargs="+", default=["grid", "grid_sh"],
                   choices=["grid", "grid_sh"])
    p.add_argument("--out", default="training_checkpoints/interp_val_eval_results.csv")
    args = p.parse_args()
    dev = torch.device(args.device)

    print(f"loading {args.npz_path} ...", flush=True)
    d = np.load(args.npz_path, allow_pickle=True)
    meta = json.loads(str(d["metadata_json"]))
    response = d["response"]; vp = d["viewpoint_positions"]
    tx_pos_all = d["tx_pos"]; rx_pos_all = d["rx_pos"]
    n_view = response.shape[0]
    fc, bw, n_adc = (float(meta["radar_fc_hz"]), float(meta["radar_bandwidth_hz"]),
                     int(meta["num_adc_samples"]))
    freqs = torch.tensor((fc - bw / 2) + np.arange(n_adc) * (bw / n_adc),
                         dtype=torch.float32, device=dev)
    kvec = get_kvector(freqs, cc)
    val_idx = np.random.default_rng(SEED).permutation(n_view)[750:800][:args.num_views]
    print(f"{len(val_idx)} val views: {val_idx.tolist()}", flush=True)

    def meas_cube(v):
        cube = response[v].mean(axis=2)  # [Tx, Rx, nf] complex64
        return torch.tensor(cube, dtype=torch.complex128, device=dev).permute(1, 0, 2)

    def render_view(v, pos, w):
        rx = torch.tensor(rx_pos_all[v], dtype=torch.float32, device=dev)
        tx = torch.tensor(tx_pos_all[v], dtype=torch.float32, device=dev)
        with torch.no_grad():
            S = range_forward_operator(freqs, kvec, rx, tx, pos, w,
                                       phase_sign=PHASE_SIGN, compute_dtype=torch.float64)
        return S.permute(1, 2, 0)  # [Rx, Tx, nf]

    rows = []

    def eval_variant(label, weights_per_view, pos, g_trained):
        num_t = den = 0.0
        ps = pp = 0.0 + 0.0j
        t0 = time.time()
        for v in val_idx:
            S_meas = meas_cube(v)
            S_pred = render_view(v, pos, weights_per_view(v))
            num_t += ((g_trained * S_pred - S_meas).abs() ** 2).sum().item()
            den += (S_meas.abs() ** 2).sum().item()
            ps += (S_pred.conj() * S_meas).sum().item()
            pp += (S_pred.conj() * S_pred).sum().item()
        rel_trained = num_t / den
        rel_opt = 1.0 - (abs(ps) ** 2 / (pp.real * den))
        g_opt = ps / pp.real
        per_view = (time.time() - t0) / len(val_idx)
        print(f"{label} ({len(val_idx)} views): rel-MSE trained-gain={rel_trained:.1%}  "
              f"optimal-gain={rel_opt:.1%}  |g_opt|={abs(g_opt):.4f}  [{per_view:.1f}s/view]",
              flush=True)
        rows.append({"variant": label, "views": len(val_idx),
                     "rel_mse_trained_gain": rel_trained, "rel_mse_optimal_gain": rel_opt,
                     "g_opt_abs": abs(g_opt)})

    for repr_name in args.reprs:
        ck = torch.load(CKPTS[repr_name], map_location="cpu", weights_only=False)
        g_tr = trained_gain(ck)
        print(f"\n== {repr_name} g48 (epoch {ck.get('epoch')}, trained |g|={abs(g_tr):.3f}) ==",
              flush=True)
        if repr_name == "grid":
            sd = ck["model_state_dict"]
            w48 = torch.complex(sd["w_re"].double(), sd["w_im"].double())
            mask = sd.get("active_mask")
            if mask is not None:
                w48 = torch.where(mask.bool(), w48, torch.zeros((), dtype=w48.dtype))
            view_field = lambda v: w48
        else:
            model = SHVoxelGridScene(G_COARSE, EXTENT, torch.device("cpu"),
                                     max_degree=6, init_degree=6)
            model.load_state_dict(ck["model_state_dict"])
            model.eval()

            def view_field(v):
                theta, phi = _direction_to_theta_phi(vp[v])
                with torch.no_grad():
                    _, w = model.active_scatterers(torch.tensor([[theta]]),
                                                   torch.tensor([[phi]]))
                return w.reshape(G_COARSE, G_COARSE, G_COARSE)

        for factor in args.factors:
            gfine = G_COARSE * factor
            pos = generate_dynamic_grid(gfine, EXTENT, dev, jitter=False).reshape(-1, 3)
            if factor == 1:
                wpv = lambda v: view_field(v).reshape(-1).to(dev)
            else:
                wpv = lambda v, gf=gfine: upsample_complex(view_field(v), gf).to(dev)
            eval_variant(f"{repr_name} {gfine}^3", wpv, pos, g_tr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nresults written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
