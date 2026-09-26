"""Seeding and RNG capture/restore twins for the GeRaF PVC entry point.

These replace ``train_geraf.py``'s ``_seed_everything``, ``_capture_rng_state``
and ``_restore_rng_state`` on the ``hardened_v1``/``legacy`` compatibility path.
``train_geraf_pvc.py`` rebinds them onto the imported (unchanged) module.

Payload compatibility, deliberately: ``torch_cuda`` keeps its original key and
its ``None``-when-CUDA-is-absent convention, and an XPU twin ``torch_xpu`` is
added beside it. A checkpoint written on CUDA and resumed on PVC therefore finds
no ``torch_xpu``, restores python/numpy/CPU-torch/view-sampler state and skips
the accelerator payload — exactly the branch the original code already takes on
a host without CUDA. GeRaF draws no random numbers on the device (its ray jitter
comes from ``np.random`` under ``fixed_numpy_seed``), so this payload only
carries accelerator state for completeness; the sampling stream is identical on
both backends.
"""
from __future__ import annotations

import random
from typing import Any, Dict, Mapping

import numpy as np
import torch

from rift_pvc import accelerator


def seed_everything(seed: int) -> None:
    """``train_geraf._seed_everything`` with the accelerator behind the shim."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    accelerator.manual_seed_all(seed)
    # Kept as a call site, guarded by backend: XPU exposes no cudnn benchmark or
    # determinism switch, so there is nothing to set there.
    if accelerator.backend() == "cuda" and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def capture_rng_state(sampler) -> Dict[str, Any]:
    backend = accelerator.backend()
    return {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "torch_xpu": accelerator.get_rng_state_all() if backend == "xpu" else None,
        "accelerator_backend": backend,
        "view_sampler": sampler.state_dict(),
    }


def restore_rng_state(state: Mapping[str, Any], sampler) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    backend = accelerator.backend()
    if backend == "cuda" and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all([tensor.cpu() for tensor in state["torch_cuda"]])
    elif backend == "xpu" and state.get("torch_xpu") is not None:
        accelerator.set_rng_state_all([tensor.cpu() for tensor in state["torch_xpu"]])
    sampler.load_state_dict(state["view_sampler"])
