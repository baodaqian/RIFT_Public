#!/usr/bin/env python
"""Correctness gate for scene-sharded multi-GPU training (rift/distributed.py,
rift/sharded_scene.py).

The claim being tested is the one the whole design rests on: because the
forward operator is a SUM over scatterers, splitting the voxels across R
ranks and all-reducing the partial renders reproduces the single-GPU render
and the single-GPU gradients EXACTLY (to fp64 summation-order noise) -- so a
multi-GPU run is the same experiment, not an approximation of it.

Stages (all run at every world size; run this at 1 and at >=2 processes):
  B1 sum_r operator(shard_r points) == operator(all points)          (the sharding claim, fp64)
  B2 sum_r render(shard_r) == render(full scene)                     (end to end, through the scene)
  C  dL/dw on each shard == the corresponding rows of the full dL/dw (backward)
  D  full_state_dict() round-trips into SHVoxelGridScene             (checkpoint)
  E  prune()/grow() thresholds match the single-scene decisions      (criteria)

B1 vs B2: B1 feeds both sides bit-identical weights and holds to fp64
(~1e-15) -- that is the exact statement that sharding the sum is exact. B2
additionally evaluates the SH weights through each scene's own contraction,
``einsum('xyzc,c->xyz')`` on the [G,G,G,C] grid vs ``einsum('kb,b->k')`` on
the flat shard. Those contract the same numbers in a shape-dependent order
in FLOAT32 (the scene parameters' dtype), so they can disagree by ~1e-7
relative -- fp32 epsilon on the weights themselves, measured at world sizes
that make BLAS pick a different blocking. That is rounding in a quantity
already stored at fp32, not a sharding error, hence the looser fp32-level
tolerance on B2/C.

Usage (CPU/gloo is enough -- this is a math check, not a throughput test):
    python scripts/validate_distributed_scene.py
    torchrun --standalone --nproc_per_node=2 scripts/validate_distributed_scene.py
    torchrun --standalone --nproc_per_node=3 scripts/validate_distributed_scene.py   # uneven split
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rift.distributed import (  # noqa: E402
    all_reduce_sum,
    all_reduce_sum_grad,
    get_rank,
    get_world_size,
    init_distributed,
    shutdown_distributed,
)
from rift.forward_operator import get_kvector  # noqa: E402
from rift.range_operator import range_forward_operator  # noqa: E402
from rift.sharded_scene import ShardedSHVoxelGridScene  # noqa: E402
from rift.sparse_scene import SHVoxelGridScene  # noqa: E402

CC = 299792458.0
FAILURES = []


def check(name, ok, detail=""):
    rank = get_rank()
    status = "PASS" if ok else "FAIL"
    if rank == 0:
        print(f"  [{status}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def rel_err(a, b):
    denom = b.abs().max().clamp_min(1e-300)
    return float((a - b).abs().max() / denom)


def make_geometry(device, dtype=torch.float64, num_tx=4, num_rx=4, nf=16):
    """A miniature viewpoint: two small linear arrays 10 m out, uniform freqs."""
    freqs = torch.linspace(78.5e9, 79.5e9, nf, dtype=dtype, device=device)
    k = get_kvector(freqs, CC)
    off = torch.arange(num_tx, dtype=dtype, device=device)[:, None] * 2e-3
    tx = torch.zeros(num_tx, 3, dtype=dtype, device=device)
    tx[:, 0] = 10.0
    tx[:, 1:2] = off
    rx = torch.zeros(num_rx, 3, dtype=dtype, device=device)
    rx[:, 0] = 10.0
    rx[:, 2:3] = torch.arange(num_rx, dtype=dtype, device=device)[:, None] * 2e-3
    dtheta = torch.tensor([[1.2]], device=device)
    dphi = torch.tensor([[0.3]], device=device)
    return freqs, k, rx, tx, dtheta, dphi


def render(scene, freqs, k, rx, tx, dtheta, dphi, phase_sign=-1.0):
    pos, w = scene.active_scatterers(dtheta, dphi)
    return range_forward_operator(
        freqs, k, rx, tx, pos, w, phase_sign=phase_sign,
        compute_dtype=torch.float64, point_chunk=97, pair_chunk=5,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--granularity", type=int, default=6)
    ap.add_argument("--max-degree", type=int, default=2)
    ap.add_argument("--tol", type=float, default=1e-12,
                    help="fp64 tolerance for the pure operator-sharding identity (B1)")
    ap.add_argument("--tol-fp32", type=float, default=1e-6,
                    help="tolerance for paths that go through the scene's float32 SH contraction "
                         "(B2/C); see the module docstring")
    args = ap.parse_args()

    rank, world_size, device = init_distributed()
    torch.manual_seed(0)
    if rank == 0:
        print(f"validate_distributed_scene: world_size={world_size}, device={device}, "
              f"G={args.granularity} ({args.granularity ** 3} voxels), max_degree={args.max_degree}")

    g, deg = args.granularity, args.max_degree
    extent = 1.5

    # Reference: the ordinary single-GPU scene, identical on every rank
    # (same seed, same construction order, no sharding).
    torch.manual_seed(1234)
    full = SHVoxelGridScene(g, extent, device, max_degree=deg, init_degree=deg, init_scale=0.3)
    ref_state = {kk: v.clone() for kk, v in full.state_dict().items()}

    shard = ShardedSHVoxelGridScene(g, extent, device, max_degree=deg, init_degree=deg, init_scale=0.0)
    shard.load_full_state_dict(ref_state)

    freqs, k, rx, tx, dtheta, dphi = make_geometry(device)

    # ---- B1: the sharding identity, with bit-identical weights -------
    pos_all, w_all = full.active_scatterers(dtheta, dphi)
    s_all = range_forward_operator(
        freqs, k, rx, tx, pos_all, w_all, phase_sign=-1.0,
        compute_dtype=torch.float64, point_chunk=97, pair_chunk=5,
    )
    idx = shard.voxel_index
    s_part = all_reduce_sum(range_forward_operator(
        freqs, k, rx, tx, pos_all[idx], w_all[idx], phase_sign=-1.0,
        compute_dtype=torch.float64, point_chunk=97, pair_chunk=5,
    ))
    err1 = rel_err(s_part, s_all)
    check("B1: sum_r operator(shard_r points) == operator(all points)", err1 < args.tol,
          f"max rel err {err1:.3e}")

    # ---- B2: end to end, each scene evaluating its own SH weights ----
    s_full = render(full, freqs, k, rx, tx, dtheta, dphi)
    s_shard = all_reduce_sum(render(shard, freqs, k, rx, tx, dtheta, dphi))
    err = rel_err(s_shard, s_full)
    check("B2: sum_r render(shard_r) == render(full scene)", err < args.tol_fp32,
          f"max rel err {err:.3e} (fp32 SH-contraction level)")

    # ---- C: backward -------------------------------------------------
    # Same objective shape as training: mean |S_pred - S_meas|^2 with the
    # cross-rank sum in the graph.
    torch.manual_seed(7)
    s_meas = torch.randn_like(s_full)

    def loss_of(pred):
        d = pred - s_meas
        return (d.real ** 2 + d.imag ** 2).mean()

    full.zero_grad(set_to_none=True)
    loss_of(render(full, freqs, k, rx, tx, dtheta, dphi)).backward()
    shard.zero_grad(set_to_none=True)
    loss_of(all_reduce_sum_grad(render(shard, freqs, k, rx, tx, dtheta, dphi))).backward()

    ref_grad = full.w_re.grad.reshape(g ** 3, -1)[shard.voxel_index]
    gerr = rel_err(shard.w_re.grad, ref_grad)
    check("C: dL/dw_re on the shard == the full scene's rows", gerr < args.tol_fp32,
          f"max rel err {gerr:.3e}")
    ref_grad_im = full.w_im.grad.reshape(g ** 3, -1)[shard.voxel_index]
    gerr_im = rel_err(shard.w_im.grad, ref_grad_im)
    check("C: dL/dw_im on the shard == the full scene's rows", gerr_im < args.tol_fp32,
          f"max rel err {gerr_im:.3e}")

    # ---- F: occlusion ------------------------------------------------
    # The extinction field is GLOBAL even though the scene is sharded: a rank
    # must be shadowed by occluders it does not own. opacity_volume()
    # therefore scatters each rank's strided voxels into the full grid and
    # all-reduces (disjoint supports, so backward is the identity, same
    # argument as the S-parameter reduction). If that reduction were missing,
    # every rank would see only 1/R of the scene's opacity and an R-GPU run
    # would silently be a different experiment from a 1-GPU run.
    from rift.occlusion import OcclusionScale, opacity_volume, view_transmittance  # noqa: E402

    origin = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64, device=device)
    zeta = OcclusionScale(0.3, device=device)
    sig_full = opacity_volume(full, zeta)
    sig_shard = opacity_volume(shard, zeta)
    check("F: the sharded extinction field equals the full one",
          rel_err(sig_shard, sig_full) < args.tol_fp32,
          f"max rel err {rel_err(sig_shard, sig_full):.3e}")

    t2_full = view_transmittance(full, zeta, origin)
    t2_shard = view_transmittance(shard, zeta, origin)
    ref_t2 = t2_full[shard.voxel_index[shard.active_mask]] if shard.active_mask.all() \
        else t2_full.reshape(-1)[shard.voxel_index[shard.active_mask]]
    check("F: this rank's T^2 matches the full scene's rows",
          rel_err(t2_shard, ref_t2) < args.tol_fp32,
          f"max rel err {rel_err(t2_shard, ref_t2):.3e}")

    # ---- D: checkpoint round trip ------------------------------------
    gathered = shard.full_state_dict()
    reloaded = SHVoxelGridScene(g, extent, device, max_degree=deg, init_degree=deg, init_scale=0.0)
    reloaded.load_state_dict(gathered)  # canonical layout: strict load must succeed
    same = all(
        torch.equal(reloaded.state_dict()[kk], ref_state[kk])
        for kk in ("w_re", "w_im", "order", "active_mask")
    )
    check("D: full_state_dict() gathers back to the exact single-GPU state", same)

    s_reloaded = render(reloaded, freqs, k, rx, tx, dtheta, dphi)
    check("D: the gathered checkpoint renders identically",
          rel_err(s_reloaded, s_full) == 0.0)

    # ---- E: prune / grow decisions -----------------------------------
    full_p = SHVoxelGridScene(g, extent, device, max_degree=deg, init_degree=0, init_scale=0.3)
    p_state = {kk: v.clone() for kk, v in full_p.state_dict().items()}
    shard_p = ShardedSHVoxelGridScene(g, extent, device, max_degree=deg, init_degree=0, init_scale=0.0)
    shard_p.load_full_state_dict(p_state)

    n_active_full, n_total_full, _ = full_p.prune(threshold_fraction=0.6)
    n_active_shard, n_total_shard, _ = shard_p.prune(threshold_fraction=0.6)
    check("E: prune() keeps the same voxels (global max threshold)",
          n_active_full == n_active_shard and n_total_full == n_total_shard
          and torch.equal(shard_p.active_mask,
                          full_p.active_mask.reshape(-1)[shard_p.voxel_index]),
          f"{n_active_shard}/{n_total_shard} active vs {n_active_full}/{n_total_full}")

    # The 2026-08-03 prune modes resolve their threshold from a GLOBAL order
    # statistic / a GLOBAL energy sum, not just a max, so they have strictly
    # more to get wrong under sharding than relmax did. Same lockstep contract.
    for mode, kw in (("mass", dict(threshold_fraction=0.05)),
                     ("target", dict(target_active=max(4, g ** 3 // 4))),
                     ("relmax+floor", dict(threshold_fraction=0.9,
                                           min_active=max(4, g ** 3 // 8)))):
        f_p = SHVoxelGridScene(g, extent, device, max_degree=deg, init_degree=0, init_scale=0.3)
        st = {kk: v.clone() for kk, v in f_p.state_dict().items()}
        s_p = ShardedSHVoxelGridScene(g, extent, device, max_degree=deg, init_degree=0, init_scale=0.0)
        s_p.load_full_state_dict(st)
        call = dict(kw)
        if mode == "relmax+floor":
            call["mode"] = "relmax"
        else:
            call["mode"] = mode
        n_f, t_f, rep_f = f_p.prune(**call)
        n_s, t_s, rep_s = s_p.prune(**call)
        check(f"E: prune(mode={mode}) keeps the same voxels on 1 vs {get_world_size()} ranks",
              n_f == n_s and t_f == t_s
              and torch.equal(s_p.active_mask, f_p.active_mask.reshape(-1)[s_p.voxel_index]),
              f"{n_s}/{t_s} active vs {n_f}/{t_f}")
        check(f"E: prune(mode={mode}) selectivity report is global, not per-shard",
              rep_f == rep_s, f"\n    full : {rep_f}\n    shard: {rep_s}")

    # grad-ladder growth: same synthetic accumulator on both scenes
    torch.manual_seed(11)
    accum = torch.rand(g ** 3, device=device)
    full_p.grad_accum.copy_(accum.view(g, g, g))
    full_p.grad_accum_count.fill_(3)
    shard_p.grad_accum.copy_(accum[shard_p.voxel_index])
    shard_p.grad_accum_count.fill_(3)
    n_grown_full, _, rep_full = full_p.grow(threshold_fraction=0.5)
    n_grown_shard, _, rep_shard = shard_p.grow(threshold_fraction=0.5)
    check("E: grow() unlocks the same voxels (global max threshold)",
          n_grown_full == n_grown_shard
          and torch.equal(shard_p.order, full_p.order.reshape(-1)[shard_p.voxel_index]),
          f"{n_grown_shard} grown vs {n_grown_full}")
    check("E: grow() selectivity report is global, not per-shard",
          rep_full == rep_shard, f"{rep_shard!r} vs {rep_full!r}")

    # --grow-threshold-mode quantile: the top q of the WHOLE scene, not the top
    # q of every shard (which is what a per-rank quantile would give).
    torch.manual_seed(12)
    accum = torch.rand(g ** 3, device=device)
    for sc, idx in ((full_p, None), (shard_p, shard_p.voxel_index)):
        sc.order.fill_(0)
        sc.grad_accum.copy_(accum.view(g, g, g) if idx is None else accum[idx])
        sc.grad_accum_count.fill_(1)
    # eligible = ACTIVE and not-yet-maxed-out, i.e. what survived the prune above
    n_elig = int(full_p.active_mask.sum())
    n_q_full, _, _ = full_p.grow(threshold_fraction=0.1, mode="quantile")
    n_q_shard, _, _ = shard_p.grow(threshold_fraction=0.1, mode="quantile")
    check("E: grow(mode=quantile) selects the same voxels globally",
          n_q_full == n_q_shard
          and torch.equal(shard_p.order, full_p.order.reshape(-1)[shard_p.voxel_index]),
          f"{n_q_shard} grown vs {n_q_full}")
    check("E: grow(mode=quantile) honours the requested fraction",
          abs(n_q_full / n_elig - 0.1) <= 1.0 / n_elig,
          f"grew {n_q_full}/{n_elig} eligible = {n_q_full / n_elig:.4f}, wanted 0.10")

    # grow_angular's report must also be globally reduced
    torch.manual_seed(13)
    for sc, idx in ((full_p, None), (shard_p, shard_p.voxel_index)):
        sc.order.fill_(1)
    w = torch.rand(g ** 3, full_p.w_re.shape[-1], device=device)
    full_p.w_re.data.copy_(w.view(g, g, g, -1))
    shard_p.w_re.data.copy_(w[shard_p.voxel_index])
    n_a_full, _, rep_a_full = full_p.grow_angular(tail_ratio_threshold=0.05)
    n_a_shard, _, rep_a_shard = shard_p.grow_angular(tail_ratio_threshold=0.05)
    check("E: grow_angular() matches and reports globally",
          n_a_full == n_a_shard and rep_a_full == rep_a_shard,
          f"{n_a_shard}/{rep_a_shard!r} vs {n_a_full}/{rep_a_full!r}")

    if rank == 0:
        if FAILURES:
            print(f"\nFAILED {len(FAILURES)} check(s): {FAILURES}")
        else:
            print(f"\nAll checks passed at world_size={world_size}.")
    shutdown_distributed()
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
