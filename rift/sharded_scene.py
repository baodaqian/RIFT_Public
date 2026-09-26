"""Voxel-sharded twin of SHVoxelGridScene for multi-GPU (scene-parallel)
training -- see rift/distributed.py for why the scene, not the viewpoint
stream, is what gets split.

Same representation, same growth/prune criteria, same checkpoint format as
``rift.sparse_scene.SHVoxelGridScene``; the only difference is that rank r
allocates ONLY its own strided subset of the G^3 voxels, stored flat as
``[n_local, n_basis]`` instead of ``[G,G,G,n_basis]``.  Because the forward
operator sums over scatterers, rendering each rank's subset and all-reducing
the partial S-parameter cubes reproduces the full-scene render exactly.

Checkpoints go through :meth:`full_state_dict`, which gathers the shards
back into the canonical ``[G,G,G,n_basis]`` single-GPU layout -- so
``scripts/visualize_scene_checkpoint.py``, ``eval_scene_geometry.py``,
``render_dense_scene.py`` and every other eval script read a multi-GPU
checkpoint with no changes, and a sharded run can be resumed at a different
GPU count.

Decision rules that reference a GLOBAL quantity (prune()/grow()'s
"fraction of the max", the regularizer's mean over active entries in
train.py) reduce across ranks here; everything else is elementwise and
needs no communication.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from rift.distributed import (
    all_reduce_int,
    all_reduce_max,
    gather_full,
    get_rank,
    get_world_size,
    shard_indices,
)
from rift.encoding import generate_dynamic_grid
from rift.sparse_scene import (
    _tail_ratio_report,
    grow_threshold_and_report,
    prune_threshold_and_report,
)
from rift.spherical_harmonics import basis_degree_index, num_sh_basis, real_sh_basis


class ShardedSHVoxelGridScene(nn.Module):
    """SHVoxelGridScene over voxel indices ``shard_indices(G^3)``.

    Constructing it with world_size=1 gives a scene that is mathematically
    identical to SHVoxelGridScene (verified by
    ``scripts/validate_distributed_scene.py``), just flat-stored, so the
    single- and multi-GPU code paths in train.py are one path.
    """

    def __init__(self, granularity, extent, device, max_degree=10, init_degree=0,
                 init_scale=0.1, rank=None, world_size=None):
        super().__init__()
        if not (0 <= init_degree <= max_degree):
            raise ValueError(f"init_degree ({init_degree}) must be in [0, max_degree={max_degree}]")

        self.granularity = granularity
        self.extent = extent
        self.max_degree = max_degree
        self.rank = get_rank() if rank is None else rank
        self.world_size = get_world_size() if world_size is None else world_size

        n_total = granularity ** 3
        self.n_total = n_total
        idx = shard_indices(n_total, self.rank, self.world_size, device=device)
        n_local = idx.numel()
        if n_local == 0:
            raise ValueError(
                f"rank {self.rank} owns 0 of {n_total} voxels -- world_size "
                f"({self.world_size}) exceeds granularity^3"
            )
        n_basis = num_sh_basis(max_degree)

        self.register_buffer("voxel_index", idx)
        self.register_buffer("basis_degree", basis_degree_index(max_degree, device=device))

        # Build the full lattice and keep only this rank's rows. The full
        # [G^3, 3] lattice is a transient (25 MB even at G=128); the shard is
        # what stays resident.
        full_positions = generate_dynamic_grid(granularity, extent, device, jitter=False).reshape(-1, 3)
        self.register_buffer("grid_positions", full_positions[idx].contiguous())
        del full_positions

        self.w_re = nn.Parameter(init_scale * torch.randn(n_local, n_basis, device=device))
        self.w_im = nn.Parameter(init_scale * torch.randn(n_local, n_basis, device=device))
        with torch.no_grad():
            locked = self.basis_degree > init_degree
            self.w_re[:, locked] = 0.0
            self.w_im[:, locked] = 0.0

        self.register_buffer("order", torch.full((n_local,), init_degree, dtype=torch.int64, device=device))
        self.register_buffer("active_mask", torch.ones(n_local, dtype=torch.bool, device=device))
        self.register_buffer("grad_accum", torch.zeros(n_local, device=device))
        self.register_buffer("grad_accum_count", torch.tensor(0, dtype=torch.int64, device=device))

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def active_scatterers(self, dtheta, dphi):
        """This rank's (positions[n,3], weights[n]) for the given viewpoint.

        Identical to SHVoxelGridScene.active_scatterers apart from the flat
        voxel axis; the caller must all-reduce the rendered S-parameters
        (train.py does, via rift.distributed.all_reduce_sum_grad).
        """
        theta = dtheta.squeeze(0)[0]
        phi = dphi.squeeze(0)[0]
        basis = real_sh_basis(theta, phi, self.max_degree)  # [n_basis]

        order_mask = (self.basis_degree.view(1, -1) <= self.order[:, None]).to(self.w_re.dtype)
        w_re_eff = torch.einsum('kb,b->k', self.w_re * order_mask, basis)
        w_im_eff = torch.einsum('kb,b->k', self.w_im * order_mask, basis)
        weights = torch.complex(w_re_eff, w_im_eff)
        return self.grid_positions[self.active_mask], weights[self.active_mask]

    # ------------------------------------------------------------------
    # adaptive-density mechanics (criteria identical to SHVoxelGridScene,
    # thresholds taken over the GLOBAL scene)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def prune(self, threshold_fraction=0.01, criterion="energy", mode="relmax",
              target_active=None, min_active=0):
        """Sharded SHVoxelGridScene.prune. The criterion is per-voxel and
        therefore purely local; every quantity the THRESHOLD depends on (the
        max, the energy mass, the top-K count) is globally reduced inside
        prune_threshold_and_report, so this stays exact under sharding (see
        scripts/validate_distributed_scene.py).

        Contains collectives on every path -- EVERY rank must call it, even
        one whose local shard is entirely pruned already."""
        if criterion == "dc":
            mag = torch.complex(self.w_re[:, 0], self.w_im[:, 0]).abs()
        elif criterion == "energy":
            unlocked = (self.basis_degree.view(1, -1) <= self.order[:, None])
            sq = (self.w_re ** 2 + self.w_im ** 2) * unlocked.to(self.w_re.dtype)
            mag = sq.sum(dim=-1).sqrt()
        else:
            raise ValueError(f"criterion must be 'energy' or 'dc', got {criterion!r}")
        device = mag.device
        vals = mag[self.active_mask]
        local_max = vals.max() if vals.numel() else torch.zeros((), device=device)
        n_active = all_reduce_int(int(self.active_mask.sum().item()), device=device)
        thresh, report = prune_threshold_and_report(
            vals, all_reduce_max(local_max), threshold_fraction, mode,
            target_active=target_active, min_active=min_active,
            n_total=n_active, reduce=True)
        self.active_mask &= (mag >= thresh)
        self.w_re[~self.active_mask] = 0.0
        self.w_im[~self.active_mask] = 0.0
        return (all_reduce_int(int(self.active_mask.sum().item()), device=device),
                self.n_total, report)

    @torch.no_grad()
    def accumulate_grad_stats(self):
        if self.w_re.grad is None or self.w_im.grad is None:
            return
        per_voxel_grad_sq = (self.w_re.grad ** 2 + self.w_im.grad ** 2).sum(dim=-1)
        self.grad_accum += per_voxel_grad_sq.sqrt()
        self.grad_accum_count += 1

    @torch.no_grad()
    def grow_angular(self, tail_ratio_threshold=0.05):
        """Per-voxel spectral criterion (see SHVoxelGridScene.grow_angular);
        purely local -- the tail ratio is intrinsic to a voxel, so no
        cross-rank reduction is needed beyond the reported counts."""
        device = self.w_re.device
        n_active = all_reduce_int(int(self.active_mask.sum().item()), device=device)
        deg = self.basis_degree.view(1, -1)
        order_e = self.order.unsqueeze(-1)
        sq = self.w_re ** 2 + self.w_im ** 2
        e_keep = (sq * (deg <= order_e).to(sq.dtype)).sum(dim=-1)
        e_top = (sq * (deg == order_e).to(sq.dtype)).sum(dim=-1)
        ratio = e_top / e_keep.clamp_min(torch.finfo(sq.dtype).tiny)
        eligible = self.active_mask & (self.order < self.max_degree) & (e_keep > 0)
        grow_mask = eligible & (ratio >= tail_ratio_threshold)
        self.order[grow_mask] = self.order[grow_mask] + 1
        # _tail_ratio_report contains a collective, so EVERY rank must call it --
        # a rank with no eligible voxels contributes an empty local set. The
        # denominator is the GLOBAL eligible count, not this shard's.
        report = _tail_ratio_report(
            ratio[eligible],
            n_total=all_reduce_int(int(eligible.sum().item()), device=device),
            device=device, reduce=True)
        return all_reduce_int(int(grow_mask.sum().item()), device=device), n_active, report

    @torch.no_grad()
    def grow(self, threshold_fraction=0.1, mode="relmax"):
        """Gradient growth. Both threshold modes reference a GLOBAL quantity --
        the max avg-gradient over the whole scene ("relmax"), or a quantile of
        it ("quantile") -- so both are reduced across ranks here. A per-shard
        threshold would let a dim shard grow its brightest voxel while an
        identical voxel on a bright shard stayed capped, and for "quantile"
        would grow the top q of EVERY shard instead of the top q overall."""
        device = self.w_re.device
        n_active = all_reduce_int(int(self.active_mask.sum().item()), device=device)
        if self.grad_accum_count.item() == 0:
            return 0, n_active, ""

        avg_grad = self.grad_accum / self.grad_accum_count.clamp_min(1)
        eligible = self.active_mask & (self.order < self.max_degree)
        # -inf keeps a rank with no eligible voxels from dragging the max down
        local_max = avg_grad[eligible].max() if bool(eligible.any()) else torch.tensor(
            float("-inf"), device=device, dtype=avg_grad.dtype
        )
        global_max = all_reduce_max(local_max)

        n_grown = 0
        report = ""
        if torch.isfinite(global_max):
            # Collective inside: called on every rank, including ranks whose
            # local eligible set is empty. `eligible.any()` is per-rank and must
            # NOT gate this, or the all-reduce would hang.
            thresh, report = grow_threshold_and_report(
                avg_grad[eligible], global_max, threshold_fraction, mode,
                n_total=all_reduce_int(int(eligible.sum().item()), device=device),
                reduce=True)
            grow_mask = eligible & (avg_grad >= thresh)
            self.order[grow_mask] = self.order[grow_mask] + 1
            n_grown = int(grow_mask.sum().item())

        self.grad_accum.zero_()
        self.grad_accum_count.zero_()
        return all_reduce_int(n_grown, device=device), n_active, report

    # ------------------------------------------------------------------
    # checkpoint interop: canonical SHVoxelGridScene layout
    # ------------------------------------------------------------------
    def full_state_dict(self):
        """The shards gathered into SHVoxelGridScene's exact state-dict
        layout (``w_re``/``w_im`` [G,G,G,n_basis], ``order``/``active_mask``/
        ``grad_accum`` [G,G,G], plus ``basis_degree``/``grid_positions``/
        ``grad_accum_count``), so eval scripts and single-GPU resumes see an
        ordinary checkpoint.  Returned on every rank (all-reduce based);
        callers should still write it from rank 0 only."""
        g = self.granularity
        idx = self.voxel_index
        n = self.n_total
        out = {
            "basis_degree": self.basis_degree.detach().clone(),
            "w_re": gather_full(self.w_re.detach(), idx, n).view(g, g, g, -1),
            "w_im": gather_full(self.w_im.detach(), idx, n).view(g, g, g, -1),
            "order": gather_full(self.order, idx, n).view(g, g, g),
            "active_mask": gather_full(self.active_mask, idx, n).view(g, g, g),
            "grid_positions": gather_full(self.grid_positions, idx, n).view(g, g, g, 3),
            "grad_accum": gather_full(self.grad_accum, idx, n).view(g, g, g),
            "grad_accum_count": self.grad_accum_count.detach().clone(),
        }
        return out

    @torch.no_grad()
    def load_full_state_dict(self, state_dict):
        """Inverse of full_state_dict: take this rank's rows out of a
        canonical (single-GPU or gathered) checkpoint.  The world size at
        save time is irrelevant -- resharding is just a different row
        selection."""
        idx = self.voxel_index
        device = self.w_re.device

        def rows(key):
            t = state_dict[key].to(device)
            return t.reshape(self.n_total, *t.shape[3:])[idx]

        w_re = rows("w_re")
        if w_re.shape[-1] != self.w_re.shape[-1]:
            raise ValueError(
                f"checkpoint has {w_re.shape[-1]} SH coefficients per voxel, this scene has "
                f"{self.w_re.shape[-1]} -- rerun with the checkpoint's --sh-max-degree"
            )
        self.w_re.copy_(w_re)
        self.w_im.copy_(rows("w_im"))
        self.order.copy_(rows("order"))
        self.active_mask.copy_(rows("active_mask"))
        if "grad_accum" in state_dict:
            self.grad_accum.copy_(rows("grad_accum"))
        if "grad_accum_count" in state_dict:
            self.grad_accum_count.copy_(state_dict["grad_accum_count"].to(device))
