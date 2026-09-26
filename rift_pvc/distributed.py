"""Accelerator-agnostic twin of ``rift/distributed.py``.

Only the two functions that touch ``torch.cuda`` are redefined here; every
other helper is re-exported unchanged from ``rift.distributed`` so the sharded
collectives keep their exact semantics. On a CUDA host the redefined functions
behave exactly like the originals.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist

from rift.distributed import (  # noqa: F401  (re-exports)
    _AllReduceSum, all_reduce_max, all_reduce_sum, all_reduce_sum_grad, barrier,
    gather_full, get_rank, get_world_size, is_dist, rank0_print, shard_indices,
    shutdown_distributed,
)
from rift_pvc import accelerator


def init_distributed(backend: str | None = None):
    """Join the process group described by torchrun's env vars.

    Returns (rank, world_size, device). With WORLD_SIZE unset or 1 nothing is
    initialized and the single-process setup is returned: ``(0, 1, "xpu:0")``
    on PVC, ``(0, 1, "cuda:0")`` on CUDA, ``(0, 1, "cpu")`` without a device.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if accelerator.is_available():
        n_visible = accelerator.device_count()
        if world_size > n_visible:
            raise RuntimeError(
                f"WORLD_SIZE={world_size} but only {n_visible} {accelerator.backend().upper()} "
                f"device(s) visible -- request --gres=gpu:<type>:{world_size} and launch with "
                f"torchrun --nproc_per_node={world_size}"
            )
        accelerator.set_device(local_rank)
        device = f"{accelerator.backend()}:{local_rank}"
    else:
        device = "cpu"

    if world_size > 1 and not dist.is_initialized():
        if backend is None:
            backend = accelerator.collective_backend()
        dist.init_process_group(backend=backend, world_size=world_size, rank=rank)

    return rank, world_size, device


def all_reduce_int(value: int, device=None) -> int:
    """Cross-rank sum of a python int (counts for logging / global means)."""
    if not is_dist():
        return int(value)
    if device is None:
        # NCCL/XCCL require the buffer on this rank's device; gloo accepts CPU.
        device = accelerator.device(accelerator.current_device()) if accelerator.is_available() else "cpu"
    t = torch.tensor(int(value), dtype=torch.int64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())
