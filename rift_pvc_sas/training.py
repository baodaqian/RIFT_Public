"""Accelerator twins for the sonar trainer ``train_sas.py`` (Package G).

``train_sas.py`` (RIFT-SAS, adaptive RIFT-SAS and the independent SH-SAS on
sonar caches) touches CUDA in four places: the ``--device`` default (L179),
``seed_all`` (L597-602), the ``cuda_rng_state`` checkpoint key (L847) and the
inline resume restore of that key inside ``main`` (L1011-1012); the read-out
reports ``peak_cuda_memory_bytes`` (L1300). This module provides the twins of
the three module-level functions and the two helpers the copied ``main`` in
``train_sas_pvc.py`` calls. Everything else in ``train_sas`` is imported
unchanged: recipes, cache contract, selected-best resolution, refinement,
renderer and fields.

``install()`` rebinds ``parse_args``, ``seed_all`` and ``checkpoint`` inside
the unchanged ``train_sas`` module (idempotent), so the original ``main`` and
the research diagnostics that drive it also become accelerator-aware.

RNG payload: ``cuda_rng_state`` keeps the original convention (``None`` off
CUDA); ``xpu_rng_state`` (``None`` off XPU) and ``accelerator_backend`` are
added. The sonar trainer draws device RNG only at scene initialization
(``torch.randn`` in ``rift/sparse_scene.py``); ping/bin selection is numpy and
refinement uses a seeded CPU generator, so a checkpoint from the other backend
restores everything but the device payload and continues with the same
trajectory (the original's own tolerance for a missing CUDA payload).
"""
from __future__ import annotations

import os
import random
import sys
from typing import Optional, Sequence

import numpy as np
import torch

import train_sas as _sas
from rift_pvc import accelerator

REBOUND = ("parse_args", "seed_all", "checkpoint")
_ORIGINAL = {name: getattr(_sas, name) for name in REBOUND}
DEVICE_RNG_KEYS = {"cuda": "cuda_rng_state", "xpu": "xpu_rng_state"}


def check_backend() -> str:
    """Refuse to run on a non-XPU backend unless ``RIFT_PVC_ALLOW_BACKEND`` lists it."""
    allowed = {"xpu"} | {b.strip() for b in os.environ.get("RIFT_PVC_ALLOW_BACKEND", "").split(",") if b.strip()}
    backend = accelerator.backend()
    if backend not in allowed:
        raise RuntimeError(
            f"train_sas_pvc.py is the PVC entry point and found backend {backend!r}; "
            "use train_sas.py on CUDA, or set RIFT_PVC_ALLOW_BACKEND=cpu|cuda to override")
    return backend


def _has_device_flag(argv: Sequence[str]) -> bool:
    for token in argv:
        if token == "--":
            return False
        if token == "--device" or token.startswith("--device="):
            return True
    return False


def parse_args(argv: Optional[Sequence[str]] = None):
    """The original parser; ``--device`` defaults to the accelerator device when omitted."""
    raw = list(sys.argv[1:] if argv is None else argv)
    if not _has_device_flag(raw):
        raw = [*raw, "--device", str(accelerator.device())]
    return _ORIGINAL["parse_args"](raw)


parse_args.__rift_pvc_twin__ = True


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if accelerator.backend() == "xpu":
        accelerator.manual_seed_all(seed)


seed_all.__rift_pvc_twin__ = True


def checkpoint(model, calibration, optimizer, step, best_val, rng, history, args, cache):
    """The original payload plus the XPU payload and the backend identity."""
    payload = _ORIGINAL["checkpoint"](model, calibration, optimizer, step, best_val, rng, history, args, cache)
    payload["xpu_rng_state"] = accelerator.get_rng_state_all() if accelerator.backend() == "xpu" else None
    payload["accelerator_backend"] = accelerator.backend()
    return payload


checkpoint.__rift_pvc_twin__ = True


def restore_device_rng_state(state, *, device=None) -> Optional[str]:
    """Restore the active backend's device RNG payload; return the key restored or ``None``.

    On CUDA this is the original's code (``torch.cuda.set_rng_state_all`` when
    available and the key is present). On XPU the ``xpu_rng_state`` payload is
    restored when its device count matches. A checkpoint that carries no payload
    for the active backend (written on the other backend or on CPU) restores
    nothing here and says so; python/numpy/CPU-torch state were already restored
    by the caller and the trajectory is unaffected (module docstring).
    """
    del device  # the backend, not the requested device, owns the payload (as in the original)
    backend = accelerator.backend()
    if backend == "cuda":
        if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
            return "cuda_rng_state"
    elif backend == "xpu":
        payload = state.get("xpu_rng_state")
        if payload is not None:
            if len(payload) != accelerator.device_count():
                print(f"xpu_rng_state carries {len(payload)} device state(s) but {accelerator.device_count()} "
                      "XPU device(s) are visible; device RNG not restored.", flush=True)
                return None
            accelerator.set_rng_state_all(payload)
            return "xpu_rng_state"
    present = [key for key in DEVICE_RNG_KEYS.values() if state.get(key) is not None]
    print(f"Device RNG payload not restored on backend {backend!r}: checkpoint written on "
          f"{state.get('accelerator_backend', 'cuda-or-cpu')!r} carries {present or 'no device payload'}; "
          "the sonar trainer draws device RNG only at scene initialization, so the trajectory is unaffected.",
          flush=True)
    return None


def readout_telemetry(device) -> dict:
    """Backend-labelled peak memory for ``selected_readout.json`` (``peak_cuda_memory_bytes`` is kept)."""
    dev = torch.device(device)
    backend = accelerator.backend()
    peak = int(accelerator.max_memory_allocated(dev)) if dev.type == backend and backend != "cpu" else 0
    return {"peak_accelerator_memory_bytes": peak, "accelerator_backend": backend}


def install():
    """Rebind the audited names in ``train_sas``; idempotent; returns the module."""
    for name in REBOUND:
        twin = globals()[name]
        current = getattr(_sas, name)
        if current is twin:
            continue
        if getattr(current, "__rift_pvc_twin__", False):
            raise RuntimeError(f"train_sas.{name} is bound to a foreign twin {current!r}")
        if current is not _ORIGINAL[name]:
            raise RuntimeError(f"train_sas.{name} was rebound by someone else; refusing to install over it")
        setattr(_sas, name, twin)
    return _sas
