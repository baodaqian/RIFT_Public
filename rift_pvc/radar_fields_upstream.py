"""PVC twin of ``rift/radar_fields_upstream.py``: the torch shim behind the release.

The unchanged module is imported and only ``check_model_backend`` is rebound in
its namespace by :func:`install` (this process only), so the authors'
``RadarField(use_tcnn=True)`` runs byte-identical through
``OriginalRadarFieldsModel`` (ledger D1). The twin

* accepts ``--model-backend upstream-tcnn-torchshim`` on an XPU device (or on
  CPU with ``RIFT_PVC_TCNN_SHIM=1`` for tests) and then aliases
  ``rift_pvc.tcnn_torch`` as ``sys.modules["tinycudann"]`` *before* the first
  ``original_module("radarfields.nn.models")`` call, whose module top executes
  ``import tinycudann as tcnn``;
* keeps the original behaviour for ``upstream-tcnn`` (real tiny-cuda-nn, CUDA
  only) and ``torch`` (the repository's portable backend);
* applies the same fixed hyper-parameter checks as the original.

The identity ``upstream-tcnn-torchshim`` is never presented as
``upstream-tcnn`` (ledger D4).
"""
from __future__ import annotations

import os
import sys

import torch

from rift import radar_fields_upstream as _cuda
from rift.radar_fields_upstream import (  # noqa: F401  (re-exported, unchanged)
    REFERENCE_ROOT, SOURCE_FILES, OriginalRadarFieldsModel, original_module, verify_sources,
)
from rift_pvc import accelerator

REAL_BACKEND = "upstream-tcnn"
TORCHSHIM_BACKEND = "upstream-tcnn-torchshim"
SHIM_ENV = "RIFT_PVC_TCNN_SHIM"
# The original's fixed keys for the upstream model (rift/radar_fields_upstream.py).
FIXED_HYPERPARAMETERS = {"sh_degree": 3, "hash_features": 2, "hash_base_resolution": 16, "hash_log2_size": 19}

_original_check = _cuda.check_model_backend
_installed = False


def shim_forced() -> bool:
    return os.environ.get(SHIM_ENV, "").strip() == "1"


def shim_active() -> bool:
    """True once ``tinycudann`` resolves to the torch shim in this process."""
    from rift_pvc import tcnn_torch
    return sys.modules.get("tinycudann") is tcnn_torch


def install_shim():
    """Alias ``rift_pvc.tcnn_torch`` as ``tinycudann``; refuse to shadow a real one."""
    from rift_pvc import tcnn_torch
    existing = sys.modules.get("tinycudann")
    if existing is tcnn_torch:
        return tcnn_torch
    if existing is not None:
        raise RuntimeError("a real tinycudann module is already imported; the torch shim never replaces it")
    if "radarfields.nn.models" in sys.modules:
        raise RuntimeError("radarfields.nn.models was imported before the tinycudann torch shim was installed")
    sys.modules["tinycudann"] = tcnn_torch
    return tcnn_torch


def check_model_backend(args):
    """Twin of ``rift.radar_fields_upstream.check_model_backend``."""
    verify_sources()
    backend = getattr(args, "model_backend", None)
    if backend != TORCHSHIM_BACKEND:
        return _original_check(args)   # upstream-tcnn: CUDA and real tiny-cuda-nn required; torch: no checks
    device = torch.device(args.device)
    if device.type == "xpu":
        if accelerator.backend() != "xpu":
            raise RuntimeError(f"{TORCHSHIM_BACKEND} requires an allocated XPU device")
    elif device.type == "cpu":
        if not shim_forced():
            raise ValueError(f"{TORCHSHIM_BACKEND} runs on XPU; CPU checks need {SHIM_ENV}=1")
    else:
        raise ValueError(f"{TORCHSHIM_BACKEND} is the PVC backend and does not run on {device.type}; "
                         f"the real tiny-cuda-nn backend is --model-backend {REAL_BACKEND}")
    install_shim()
    for key, expected in FIXED_HYPERPARAMETERS.items():
        if getattr(args, key) != expected:
            raise ValueError(f"{TORCHSHIM_BACKEND} fixes {key}={expected}")


def install():
    """Rebind ``check_model_backend`` inside the unchanged upstream module (idempotent)."""
    global _installed
    if _installed:
        return _cuda
    if not callable(getattr(_cuda, "check_model_backend", None)):
        raise RuntimeError("rift.radar_fields_upstream no longer defines check_model_backend; revisit rift_pvc")
    _cuda.check_model_backend = check_model_backend
    _installed = True
    return _cuda
