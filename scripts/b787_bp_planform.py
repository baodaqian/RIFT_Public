#!/usr/bin/env python
"""B787 backprojection planform diagnostic (2D z=0 slice).

Settles the forward-operator phase-sign convention (+1 vs -1) definitively
via multi-viewpoint image coherence -- the wrong sign mirrors the scene
differently per viewpoint, producing viewpoint-inconsistent smear, while
the true sign gives a coherent aircraft planform -- and produces the
project's first B787 image. See EXPERIMENT_MANAGER_HANDOFF.md 2026-07-03,
"New queued diagnostic".

Reuses the project's existing, already-validated building blocks (no new
physics): rift.forward_operator.get_array_pos/get_kvector/
adjoint_operator_lessparallel (same math as train.py's backprojection_init
and Radar_Opt's archive/baseline_bp_B787.py), rift.dataset.CSVSimulationDataset.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from rift.config import cc, spacing as default_spacing
from rift.dataset import CSVSimulationDataset, list_and_select_files
from rift.forward_operator import adjoint_operator_lessparallel, get_array_pos, get_kvector


def reshape_measured_cubes(magnitude_tensor, phase_tensor, device, num_tx, num_rx):
    """Same convention as train.py's reshape_measured_cubes: flat CSV
    channel columns are Tx-outer/Rx-inner; view(-1, num_tx, num_rx) then
    permute(2, 1, 0) -> [Rx, Tx, freq]."""
    magnitude_cube = magnitude_tensor.squeeze(0).to(device).view(-1, num_tx, num_rx).permute(2, 1, 0)
    phase_cube = phase_tensor.squeeze(0).to(device).view(-1, num_tx, num_rx).permute(2, 1, 0)
    return magnitude_cube, phase_cube


def build_z0_grid(extent_m, pitch_m, device):
    n = int(round(2 * extent_m / pitch_m)) + 1
    xs = torch.linspace(-extent_m, extent_m, n)
    ys = torch.linspace(-extent_m, extent_m, n)
    gx, gy = torch.meshgrid(xs, ys, indexing='ij')
    gz = torch.zeros_like(gx)
    grid = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3).to(device)
    return grid, n, xs, ys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dirs", nargs="+", required=True,
                    help="e.g. data/AEDT_B787_Sample/zone1_CSV data/AEDT_B787_Sample/zone3_CSV")
    p.add_argument("--num-viewpoints", type=int, default=14)
    p.add_argument("--num-freq", type=int, default=200)
    p.add_argument("--extent", type=float, default=35.0, help="grid half-extent in x,y (meters)")
    p.add_argument("--pitch", type=float, default=0.015, help="grid pixel pitch (meters)")
    p.add_argument("--chunk", type=int, default=500_000, help="grid points per chunk")
    p.add_argument("--arr-dist", type=float, default=50.0)
    p.add_argument("--num-rx", type=int, default=15)
    p.add_argument("--num-tx", type=int, default=16)
    p.add_argument("--spacing", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="training_checkpoints/b787_bp_planform")
    args = p.parse_args()

    spacing = args.spacing if args.spacing is not None else default_spacing

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # Spread viewpoints across all given zone dirs (angular diversity is
    # what makes the wrong sign smear -- see handoff).
    per_dir = max(1, args.num_viewpoints // len(args.data_dirs))
    files = []
    for d in args.data_dirs:
        files.extend(list_and_select_files(d, num_files=per_dir))
    print(f"Selected {len(files)} viewpoints:", flush=True)
    for f in files:
        print(f"  {f}", flush=True)

    dataset = CSVSimulationDataset(files, device="cpu")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    grid, n, xs, ys = build_z0_grid(args.extent, args.pitch, device)
    print(f"Grid: {n}x{n} = {grid.shape[0]} points, pitch={args.pitch}m, extent=+/-{args.extent}m", flush=True)

    # Cache per-viewpoint (freqs, kvector, rx_pos, tx_pos, S_meas) once --
    # reused for both signs so we only pay the CSV/geometry cost once.
    views = []
    with torch.no_grad():
        for freqs_tensor, dphi_tensor, dtheta_tensor, magnitude_tensor, phase_tensor in loader:
            magnitude_cube, phase_cube = reshape_measured_cubes(
                magnitude_tensor, phase_tensor, device, args.num_tx, args.num_rx
            )
            freqs_full = freqs_tensor.squeeze(0).to(device)
            freq_idx = torch.sort(torch.randperm(freqs_full.shape[0], device=device)[:args.num_freq])[0]
            selected_freqs = freqs_full[freq_idx]
            S_meas = torch.polar(
                magnitude_cube[:, :, freq_idx], phase_cube[:, :, freq_idx]
            ).permute(2, 0, 1).contiguous()  # [nf, Rx, Tx]
            k_vector = get_kvector(selected_freqs, cc)
            rx_pos, tx_pos = get_array_pos(
                dtheta_tensor.to(device), dphi_tensor.to(device),
                args.arr_dist, spacing, args.num_rx, args.num_tx, device
            )
            views.append((selected_freqs, k_vector, rx_pos, tx_pos, S_meas))

    results = {}
    for sign in (1.0, -1.0):
        print(f"--- phase_sign={sign:+.0f} ---", flush=True)
        image = torch.zeros(grid.shape[0], dtype=torch.cfloat, device=device)
        for vi, (freqs, kvector, rx_pos, tx_pos, S_meas) in enumerate(views):
            for start in range(0, grid.shape[0], args.chunk):
                end = min(start + args.chunk, grid.shape[0])
                image[start:end] += adjoint_operator_lessparallel(
                    freqs, kvector, rx_pos, tx_pos, grid[start:end], S_meas, phase_sign=sign
                )
            print(f"  viewpoint {vi+1}/{len(views)} done", flush=True)

        amp = image.abs().reshape(n, n)
        peak = amp.max().item()
        median = amp.median().item()
        ratio = peak / max(median, 1e-30)
        amp_cpu = amp.cpu()
        results[sign] = (amp_cpu, peak, median, ratio)
        print(f"  peak={peak:.4e}  median={median:.4e}  peak/median={ratio:.2f}", flush=True)

        amp_db = 20.0 * torch.log10((amp_cpu / max(peak, 1e-30)).clamp_min(1e-6))
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(amp_db.numpy().T, origin='lower', aspect='equal',
                        extent=[-args.extent, args.extent, -args.extent, args.extent],
                        cmap='viridis', vmin=-40, vmax=0)
        ax.set_title(f"B787 backprojection planform (z=0), phase_sign={sign:+.0f}\n"
                      f"peak/median={ratio:.1f}")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        fig.colorbar(im, ax=ax, label="dB rel. peak")
        sign_tag = "pos1" if sign > 0 else "neg1"
        out_path = os.path.join(args.out_dir, f"planform_sign_{sign_tag}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}", flush=True)

    print("=== SUMMARY ===", flush=True)
    for sign in (1.0, -1.0):
        _, peak, median, ratio = results[sign]
        print(f"phase_sign={sign:+.0f}: peak={peak:.4e} median={median:.4e} peak/median={ratio:.2f}", flush=True)


if __name__ == "__main__":
    main()
