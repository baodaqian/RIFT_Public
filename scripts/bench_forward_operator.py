"""Time + measure the range forward operator on whatever GPU this lands on.

Purpose: decide whether a B787/sphere training run can be moved OFF the
contended H200 partition. The training loop is ~entirely
``range_forward_operator`` forward+backward, so one calibrated measurement
per GPU type gives the wall-clock scaling directly -- no training job needed.

Runs in ~1 minute on one GPU, no data files, no checkpoints. Submit it to
any partition (embers is fine, it is far under the 8 h wall):

    python scripts/bench_forward_operator.py                       # defaults = D1 per-rank shape
    python scripts/bench_forward_operator.py --n-points 110592 --point-chunk 262144   # g48 anchor

Reference anchor to compare against (measured by the experiment manager):
B787 sphere2k, g48 (110,592 voxels), 1x H200, 1800 train views -> ~936 s/epoch,
i.e. ~0.52 s per viewpoint at n_points=110592, point_chunk=262144, pair_chunk=64.

The reported "s/epoch @ 1800 views" is what to compare across GPU types; the
ratio to the H200 anchor is the slowdown factor k, and a scene-sharded run
needs k times as many GPUs to hold the same wall clock.
"""
from __future__ import annotations

import argparse
import time

import torch

from rift.config import cc
from rift.range_operator import range_forward_operator


def build_case(n_points, extent, num_tx, num_rx, nf, standoff, device, dtype):
    """A geometrically realistic case: voxel lattice in a [-extent,extent]^3
    box, array elements on a plane at `standoff` metres, uniform freq grid."""
    g = int(round(n_points ** (1.0 / 3.0)))
    lin = torch.linspace(-extent, extent, max(g, 2), dtype=dtype, device=device)
    pos = torch.cartesian_prod(lin, lin, lin)
    if pos.shape[0] < n_points:  # pad by tiling; only the count matters for timing
        reps = (n_points + pos.shape[0] - 1) // pos.shape[0]
        pos = pos.repeat(reps, 1)
    pos = pos[:n_points].contiguous()

    weights = (0.1 * torch.randn(n_points, dtype=dtype, device=device)
               + 0.1j * torch.randn(n_points, dtype=dtype, device=device))
    weights = weights.to(torch.complex128 if dtype is torch.float64 else torch.complex64)
    weights.requires_grad_(True)

    spacing = 2e-3
    tx = torch.zeros(num_tx, 3, dtype=dtype, device=device)
    tx[:, 0] = (torch.arange(num_tx, dtype=dtype, device=device) - num_tx / 2) * spacing
    tx[:, 2] = standoff
    rx = torch.zeros(num_rx, 3, dtype=dtype, device=device)
    rx[:, 1] = (torch.arange(num_rx, dtype=dtype, device=device) - num_rx / 2) * spacing
    rx[:, 2] = standoff

    fc, bw = 10e9, 3e9
    freqs = torch.linspace(fc - bw / 2, fc + bw / 2, nf, dtype=torch.float64, device=device)
    kvec = 2.0 * torch.pi * freqs / cc
    return pos, weights, tx, rx, freqs, kvec


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-points", type=int, default=110592,
                   help="scatterers rendered by THIS rank (g96/8ranks=110592, g48/1rank=110592, "
                        "g96/1rank=884736)")
    p.add_argument("--point-chunk", type=int, default=16384)
    p.add_argument("--pair-chunk", type=int, default=64)
    p.add_argument("--num-tx", type=int, default=16)
    p.add_argument("--num-rx", type=int, default=16)
    p.add_argument("--nf", type=int, default=600)
    p.add_argument("--extent", type=float, default=0.06)
    p.add_argument("--standoff", type=float, default=10.0)
    p.add_argument("--views", type=int, default=5, help="timed viewpoints (after 2 warmup)")
    p.add_argument("--epoch-views", type=int, default=1800, help="views/epoch for the extrapolation")
    p.add_argument("--compute-dtype", choices=["float64", "float32"], default="float64")
    p.add_argument("--phase-sign", type=float, default=-1.0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if args.compute_dtype == "float64" else torch.float32
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    total_vram = (torch.cuda.get_device_properties(0).total_memory / 1e9
                  if device.type == "cuda" else float("nan"))

    pos, weights, tx, rx, freqs, kvec = build_case(
        args.n_points, args.extent, args.num_tx, args.num_rx, args.nf,
        args.standoff, device, dtype)

    def one_view():
        S = range_forward_operator(
            freqs, kvec, rx, tx, pos, weights,
            phase_sign=args.phase_sign, compute_dtype=dtype,
            point_chunk=args.point_chunk, pair_chunk=args.pair_chunk,
        )
        loss = (S.real ** 2 + S.imag ** 2).sum()
        loss.backward()
        weights.grad = None
        return loss

    for _ in range(2):  # warmup: allocator, deapodization cache, kernel autotune
        one_view()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    for _ in range(args.views):
        one_view()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / args.views

    peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else float("nan")
    reserved = torch.cuda.max_memory_reserved() / 1e9 if device.type == "cuda" else float("nan")

    print(f"device                : {name}  ({total_vram:.0f} GB)")
    print(f"n_points              : {args.n_points:,}   pairs={args.num_tx * args.num_rx}  nf={args.nf}")
    print(f"chunks                : point_chunk={args.point_chunk}  pair_chunk={args.pair_chunk}"
          f"  dtype={args.compute_dtype}")
    print(f"time / viewpoint      : {dt * 1e3:.1f} ms   (fwd+bwd)")
    print(f"s/epoch @ {args.epoch_views} views : {dt * args.epoch_views:.0f} s"
          f"  ({dt * args.epoch_views / 3600:.2f} h)")
    print(f"peak CUDA allocated   : {peak:.2f} GB   (reserved {reserved:.2f} GB)")
    if device.type == "cuda":
        print(f"VRAM headroom         : {total_vram / max(reserved, 1e-9):.1f}x")


if __name__ == "__main__":
    main()
