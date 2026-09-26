#!/usr/bin/env python
"""Val-only evaluation of a trained point_sh (AdaptivePointSHScene) checkpoint.

Purpose (2026-07-17): measure the held-out val rel-MSE of
`pec_sphere_reg2_adaptive_deep`'s epoch-100 checkpoint, whose val was never
recorded (train.py best-checkpointing is train-loss-keyed) and whose resume
loop advances ~1 epoch per 8h allocation. Forward-only over the 50 val views
is ~1/15 of an epoch's work with no backward -- well under 1 GPU-h.

Faithfulness to train.py's evaluate():
  - identical val split: seed-42 permutation, views perm[750:800] of the npz
  - identical measured-data path: chirp-averaged ADC cube -> flat
    [n_adc, Tx*Rx] -> reshape_measured_cubes (Tx-outer/Rx-inner)
  - identical render: scene.active_scatterers(dtheta, dphi) ->
    range_forward_operator -> GlobalComplexGain -> viewpoint_loss
  - the only deliberate difference: uses ALL frequencies by default
    (deterministic) instead of a fresh random 600-subset per view;
    per-view means make the two directly comparable (the predict-zero
    reference 1.912744e-05 was validated against training logs to 5
    digits under exactly this substitution). Pass --num-freq 600 for a
    seeded random subset per view if exact-protocol numbers are wanted.

Reads views via O(1) seeks on the uncompressed response.npy member
(ResponseReader), so host RAM stays ~one view (~8MB), not the full 8GB array.

Reports:
  - val loss (sum over views of per-view mean |dS|^2) -- comparable to the
    training CSVs' "Validation Loss" column
  - trained-gain val rel-MSE (sum|dS|^2 / sum|S|^2, pooled over views)
  - optimal-gain val rel-MSE (single least-squares complex gain over all
    evaluated views, replacing the trained gain; scale-free -- the metric
    the ideal-shell misfit sweep reports, so directly comparable to its
    70.9% val bound at r=1.00)
  - per-view breakdown CSV (view index, |dS|^2, |S|^2, per-view rel-MSE)

Usage (GPU):
    python scripts/eval_pointsh_checkpoint_val.py --device cuda \
        --ckpt training_checkpoints/pec_sphere_reg2_adaptive_deep/checkpoint_ep100_snapshot.pth.tar \
        --out training_checkpoints/pec_sphere_reg2_adaptive_deep/val_eval_ep100.csv
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rift.calibration import GlobalComplexGain
from rift.config import cc
from rift.forward_operator import get_kvector
from rift.npz_dataset import _direction_to_theta_phi, build_freqs
from rift.range_operator import range_forward_operator
from rift.sparse_scene import AdaptivePointSHScene, SHVoxelGridScene
from train import reshape_measured_cubes, viewpoint_loss
from validate_pec_tetrahedron_coherence import ResponseReader

NPZ_DEFAULT = "data/pec_sphere_fmcw_16t16r_79ghz_bw3ghz_r10m_2k.npz"
CKPT_DEFAULT = ("training_checkpoints/pec_sphere_reg2_adaptive_deep/"
                "checkpoint_ep100_snapshot.pth.tar")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--npz", default=NPZ_DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42, help="split seed (must match training)")
    ap.add_argument("--num-train", type=int, default=750, help="split offset (must match training)")
    ap.add_argument("--num-views", type=int, default=50, help="val views to evaluate")
    ap.add_argument("--num-freq", type=int, default=0,
                    help="0 = all frequencies (deterministic); N = seeded random subset per view")
    ap.add_argument("--phase-sign", type=float, default=-1.0)
    ap.add_argument("--compute-dtype", choices=["float32", "float64"], default="float64")
    ap.add_argument("--extent", type=float, default=1.5,
                    help="grid_sh checkpoints only (extent is not stored in checkpoints)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device(args.device)
    compute_dtype = torch.float64 if args.compute_dtype == "float64" else torch.float32

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = ck["model_state_dict"]
    if ck.get("scene_repr") == "grid_sh" or (sd["w_re"].ndim == 4 and "anchors" not in sd):
        gran = sd["w_re"].shape[0]
        n_basis = sd["w_re"].shape[-1]
        max_deg = int(round(n_basis ** 0.5)) - 1
        model = SHVoxelGridScene(gran, args.extent, device, max_degree=max_deg,
                                 init_degree=max_deg, init_scale=0.0).to(device)
        model.load_state_dict(sd)
    else:
        model = AdaptivePointSHScene.from_state(sd, device)
    model.eval()
    n_active = int(model.active_mask.sum().item())
    print(f"Checkpoint: {args.ckpt}")
    print(f"  epoch={ck.get('epoch', '?')}  scene_repr={ck.get('scene_repr', '?')}  "
          f"train loss={ck.get('loss', float('nan')):.6e}")
    print(f"  active points: {n_active} / {model.active_mask.numel()} allocated  "
          f"max_degree={model.max_degree}")

    gain = GlobalComplexGain().to(device)
    gain.load_state_dict(ck["gain_state_dict"])
    g = gain.gain_value()
    print(f"  trained gain: |g|={abs(g):.4e} arg={np.angle(g):.4f} rad")

    reader = ResponseReader(args.npz)
    freqs_np = build_freqs(reader.meta)
    freqs_tensor = torch.tensor(freqs_np, dtype=torch.float32, device=device)
    k_vector_full = get_kvector(freqs_tensor, cc)
    n_freq = freqs_tensor.shape[0]

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(reader.n_view)
    val_idx = perm[args.num_train:args.num_train + args.num_views]
    print(f"Evaluating {len(val_idx)} val views, "
          f"{args.num_freq if args.num_freq > 0 else n_freq}/{n_freq} freqs, "
          f"phase_sign={args.phase_sign}, {args.compute_dtype}, device={device}")

    torch.manual_seed(args.seed)
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    rel_num = 0.0
    rel_den = 0.0
    # pooled optimal-gain accumulators over RAW (pre-gain) predictions:
    # g* = a/b minimizes sum|g*S_raw - S_meas|^2; rel-MSE = (c - |a|^2/b)/c
    acc_a = 0.0 + 0.0j
    acc_b = 0.0
    rows = []
    t0 = time.time()

    with torch.no_grad():
        for i, v in enumerate(val_idx):
            v = int(v)
            cube = reader.view_avg(v)                      # [Tx, Rx, n_adc] complex128
            num_tx, num_rx, _ = cube.shape
            flat = cube.reshape(num_tx * num_rx, -1).T     # [n_adc, Tx*Rx]
            mag = torch.tensor(np.abs(flat), dtype=torch.float32)[None]
            ph = torch.tensor(np.angle(flat), dtype=torch.float32)[None]
            magnitude_cube, phase_cube = reshape_measured_cubes(mag, ph, device, num_tx, num_rx)

            if args.num_freq > 0:
                freq_indices = torch.sort(
                    torch.randperm(n_freq, device=device)[:args.num_freq])[0]
            else:
                freq_indices = torch.arange(n_freq, device=device)
            frame_data_mag = magnitude_cube[:, :, freq_indices]
            frame_data_phase = phase_cube[:, :, freq_indices]

            theta, phi = _direction_to_theta_phi(reader.vp[v])
            dtheta = torch.tensor([[theta]], dtype=torch.float32, device=device)
            dphi = torch.tensor([[phi]], dtype=torch.float32, device=device)
            rx_pos = torch.tensor(reader.rx_pos[v], dtype=torch.float32, device=device)
            tx_pos = torch.tensor(reader.tx_pos[v], dtype=torch.float32, device=device)

            scatterer_pos, scatterer_weights = model.active_scatterers(dtheta, dphi)
            S_raw = range_forward_operator(
                freqs_tensor, k_vector_full, rx_pos, tx_pos,
                scatterer_pos, scatterer_weights,
                phase_sign=args.phase_sign, freq_indices=freq_indices,
                compute_dtype=compute_dtype,
            )
            S_pred = gain(S_raw)

            loss, sq_sum, power = viewpoint_loss(
                S_pred, frame_data_mag, frame_data_phase, criterion, "complex", 1.0, 1000.0)
            total_loss += loss.item()
            rel_num += sq_sum.item()
            rel_den += power.item()

            S_meas = torch.polar(frame_data_mag, frame_data_phase)   # [Rx, Tx, nf]
            S_raw_rt = S_raw.permute(1, 2, 0)                        # match [Rx, Tx, nf]
            acc_a += torch.sum(torch.conj(S_raw_rt) * S_meas).item()
            acc_b += torch.sum(S_raw_rt.real**2 + S_raw_rt.imag**2).item()

            rows.append((v, sq_sum.item(), power.item(), sq_sum.item() / power.item()))
            if (i + 1) % 10 == 0 or i == 0:
                el = time.time() - t0
                print(f"  view {i+1}/{len(val_idx)} (npz idx {v}): per-view rel "
                      f"{rows[-1][3]:.4f}  [{el:.1f}s elapsed, {el/(i+1):.1f}s/view]")

    opt_rel = (rel_den - abs(acc_a) ** 2 / acc_b) / rel_den if acc_b > 0 else float("nan")
    g_opt = acc_a / acc_b if acc_b > 0 else float("nan")
    print()
    print(f"Validation Loss (sum of per-view means): {total_loss:.6e}")
    print(f"Trained-gain val rel-MSE: {rel_num / rel_den:.4%}")
    print(f"Optimal-gain val rel-MSE: {opt_rel:.4%}   (g_opt |g|={abs(g_opt):.4e} "
          f"arg={np.angle(g_opt):.4f} rad)")
    print(f"[reference: predict-zero val floor = 50 x per-view mean power; "
          f"ideal-shell r=1.00 optimal-gain val = 70.9%]")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["view_idx", "sq_err_sum_trained_gain", "power_sum", "rel_mse_view"])
            w.writerows(rows)
            w.writerow([])
            w.writerow(["TOTAL", rel_num, rel_den, rel_num / rel_den])
            w.writerow(["OPTIMAL_GAIN_REL_MSE", opt_rel, "g_opt", str(g_opt)])
            w.writerow(["VAL_LOSS_SUM_PER_VIEW_MEANS", total_loss, "n_views", len(val_idx)])
        print(f"Per-view breakdown written to {args.out}")


if __name__ == "__main__":
    main()
