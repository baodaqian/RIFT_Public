"""Single-node multi-GPU support for RIFT: SCENE-SHARDED (model-parallel)
training, not DDP-over-viewpoints.

WHY SHARD THE SCENE.  The radar forward operator is a plain SUM over
scatterers,

    S(f)[rx,tx] = sum_n  w_n * G_geom(n,tx,rx) * exp(phase_sign*i*k*R_sum(n)),

so partitioning the scatterer set across R ranks and adding the R partial
renders is EXACT -- it changes neither the objective nor the optimization
trajectory (only floating-point summation order).  Each rank then stores
1/R of the voxels (parameters + AdamW moments) and renders 1/R of the
points, so BOTH the scene state and the forward operator's activation
memory -- the binding constraint on scene density, see
rift/range_operator.py's docstring on point-chunk backward memory -- fall
linearly with the GPU count.  Replicating the scene and splitting
viewpoints instead (DDP) would buy throughput only, and the memory wall is
what currently caps the point cloud.

GRADIENTS.  After the all-reduce every rank holds the same full S and
computes the SAME scalar loss, so dL/dS_local = dL/dS: the differentiable
all-reduce below is an identity in backward and performs NO communication
there.  Scene shards are disjoint parameters, so there is no gradient
all-reduce for the scene at all.  The only replicated parameters are
GlobalComplexGain's two scalars; since every rank evaluates an identical
loss they receive identical gradients and stay in lockstep with no
reduction (they are also initialized identically and stepped by identical
optimizer states).

WHAT MUST STAY IN LOCKSTEP.  Anything that changes the loss: the viewpoint
order (dataloaders are built identically on every rank with batch_size=1
and no shuffling), the random frequency subset (train.py draws it from a
shared CPU generator, not the per-device CUDA RNG, whose stream diverges
once shard shapes differ), and any global reduction used by a decision
rule (prune/grow thresholds, regularization means) -- all of which go
through the helpers here.

Every helper degrades to a no-op when the process group is not
initialized, so the single-GPU path is unchanged.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def rank0_print(*args, **kwargs) -> None:
    """print() on rank 0 only -- every rank runs the same training loop, so
    unguarded prints would produce R interleaved copies of every line."""
    if get_rank() == 0:
        print(*args, **kwargs)


def barrier() -> None:
    if is_dist():
        dist.barrier()


def init_distributed(backend: str | None = None):
    """Join the process group described by torchrun's env vars.

    Returns (rank, world_size, device).  With WORLD_SIZE unset or 1 this
    initializes nothing and returns the plain single-process setup, so
    ``python train.py`` behaves exactly as before.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        n_visible = torch.cuda.device_count()
        if world_size > n_visible:
            raise RuntimeError(
                f"WORLD_SIZE={world_size} but only {n_visible} CUDA device(s) visible -- "
                f"request --gres=gpu:<type>:{world_size} and launch with "
                f"torchrun --nproc_per_node={world_size}"
            )
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    if world_size > 1 and not dist.is_initialized():
        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, world_size=world_size, rank=rank)

    return rank, world_size, device


def shutdown_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def shard_indices(n_total: int, rank: int | None = None, world_size: int | None = None,
                  device=None) -> torch.Tensor:
    """Flat indices owned by this rank: STRIDED (rank, rank+R, rank+2R, ...).

    Strided rather than contiguous so a target occupying a small part of the
    scene box (a B787 inside a 0.3 m cube) still spreads its surviving
    voxels evenly over the ranks after prune() -- a contiguous split maps to
    x-slabs, and an x-aligned target would leave most ranks idle.
    """
    rank = get_rank() if rank is None else rank
    world_size = get_world_size() if world_size is None else world_size
    return torch.arange(rank, n_total, world_size, dtype=torch.long, device=device)


class _AllReduceSum(torch.autograd.Function):
    """Sum a tensor across ranks, differentiably.

    backward is the identity: with L computed from the reduced value on
    every rank, dL/d(local term) = dL/d(sum).  No collective runs in
    backward, so ranks cannot deadlock on mismatched backward orders.
    """

    @staticmethod
    def forward(ctx, x):
        if not is_dist():
            return x
        y = x.detach().clone().contiguous()
        if y.is_complex():
            # NCCL has no complex reduction; the real view is contiguous and
            # aliases the same storage, so this reduces Re/Im in one call.
            dist.all_reduce(torch.view_as_real(y), op=dist.ReduceOp.SUM)
        else:
            dist.all_reduce(y, op=dist.ReduceOp.SUM)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out


def all_reduce_sum_grad(x: torch.Tensor) -> torch.Tensor:
    """Differentiable cross-rank sum (identity when not distributed)."""
    return _AllReduceSum.apply(x)


def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """Non-differentiable cross-rank sum; returns a new tensor."""
    if not is_dist():
        return x
    y = x.detach().clone().contiguous()
    if y.is_complex():
        dist.all_reduce(torch.view_as_real(y), op=dist.ReduceOp.SUM)
    else:
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y


def all_reduce_max(x: torch.Tensor) -> torch.Tensor:
    """Non-differentiable cross-rank max; returns a new tensor.

    Used for the prune/grow relative thresholds, which are fractions of a
    GLOBAL max -- taking each rank's local max would apply a different
    (shard-dependent) threshold on every GPU.
    """
    if not is_dist():
        return x
    y = x.detach().clone().contiguous()
    dist.all_reduce(y, op=dist.ReduceOp.MAX)
    return y


def all_reduce_int(value: int, device=None) -> int:
    """Cross-rank sum of a python int (counts for logging / global means)."""
    if not is_dist():
        return int(value)
    if device is None:
        # NCCL requires the buffer on this rank's GPU; gloo accepts CPU.
        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    t = torch.tensor(int(value), dtype=torch.int64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


def gather_full(local_values: torch.Tensor, index: torch.Tensor, n_total: int) -> torch.Tensor:
    """Reassemble a sharded [n_local, ...] tensor into the full [n_total, ...]
    tensor on every rank, placing each rank's rows at ``index``.

    Implemented as scatter-into-zeros + all-reduce(SUM) because the shards
    are disjoint: simple, dtype-agnostic (bool/int64 are promoted and cast
    back), and correct for the strided index sets shard_indices() produces.
    Transiently allocates one full-size buffer per call, so checkpointing
    walks tensors one at a time rather than gathering the whole state dict
    at once.
    """
    if not is_dist():
        out = torch.zeros((n_total, *local_values.shape[1:]), dtype=local_values.dtype,
                          device=local_values.device)
        out[index] = local_values
        return out

    src_dtype = local_values.dtype
    work_dtype = torch.int64 if src_dtype in (torch.bool, torch.int64, torch.int32) else src_dtype
    buf = torch.zeros((n_total, *local_values.shape[1:]), dtype=work_dtype,
                      device=local_values.device)
    buf[index] = local_values.to(work_dtype)
    if buf.is_complex():
        dist.all_reduce(torch.view_as_real(buf), op=dist.ReduceOp.SUM)
    else:
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    return buf.to(src_dtype)
