"""One accelerator API over ``torch.cuda``, ``torch.xpu`` and CPU.

Every PVC module uses this instead of ``torch.cuda.*`` so the same code runs on
the H100 (CUDA) and PVC (XPU) environments. On a CUDA host every function maps
one-to-one onto ``torch.cuda`` and returns exactly what the original code
obtained from it, including the ``"torch_cuda_all"`` RNG payload key that
``train.py`` checkpoints use.

Backend selection: CUDA if available, else XPU if available, else CPU. The
environment variable ``RIFT_ACCELERATOR`` (``cuda`` | ``xpu`` | ``cpu``) forces
a backend; forcing an unavailable accelerator raises at first use.
"""
from __future__ import annotations

import os
from typing import Sequence

import torch
import torch.distributed as dist

_BACKENDS = ("cuda", "xpu", "cpu")


def _xpu_available() -> bool:
    xpu = getattr(torch, "xpu", None)
    return xpu is not None and bool(xpu.is_available())


def backend() -> str:
    """Return ``"cuda"``, ``"xpu"`` or ``"cpu"``."""
    forced = os.environ.get("RIFT_ACCELERATOR")
    if forced:
        forced = forced.strip().lower()
        if forced not in _BACKENDS:
            raise ValueError(f"RIFT_ACCELERATOR must be one of {_BACKENDS}, got {forced!r}")
        if forced == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("RIFT_ACCELERATOR=cuda but CUDA is not available")
        if forced == "xpu" and not _xpu_available():
            raise RuntimeError("RIFT_ACCELERATOR=xpu but XPU is not available")
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if _xpu_available():
        return "xpu"
    return "cpu"


def module():
    """``torch.cuda``, ``torch.xpu`` or ``None`` for the active backend."""
    b = backend()
    if b == "cuda":
        return torch.cuda
    if b == "xpu":
        return torch.xpu
    return None


def is_available() -> bool:
    return backend() != "cpu"


def is_accelerator(device) -> bool:
    """True when ``device`` (str or torch.device) is a CUDA or XPU device."""
    return torch.device(device).type in ("cuda", "xpu")


def device(index: int | None = None) -> torch.device:
    """The active backend's device, optionally with an index (ignored on CPU)."""
    b = backend()
    if b == "cpu" or index is None:
        return torch.device(b)
    return torch.device(f"{b}:{int(index)}")


def device_count() -> int:
    m = module()
    return int(m.device_count()) if m is not None else 0


def set_device(index: int) -> None:
    m = module()
    if m is not None:
        m.set_device(int(index))


def current_device() -> int:
    m = module()
    return int(m.current_device()) if m is not None else 0


def synchronize(dev=None) -> None:
    m = module()
    if m is not None:
        m.synchronize(dev)


def empty_cache() -> None:
    m = module()
    if m is not None:
        m.empty_cache()


def manual_seed_all(seed: int) -> None:
    m = module()
    if m is not None:
        m.manual_seed_all(int(seed))


def get_rng_state_all() -> list:
    """Per-device RNG byte states (empty list on CPU)."""
    m = module()
    return list(m.get_rng_state_all()) if m is not None else []


def set_rng_state_all(states: Sequence[torch.Tensor]) -> None:
    m = module()
    if m is not None:
        m.set_rng_state_all(list(states))


def rng_state_key() -> str:
    """Checkpoint key for the accelerator RNG payload: ``torch_cuda_all`` on
    CUDA (as train.py writes), ``torch_xpu_all`` on PVC."""
    return f"torch_{backend()}_all"


def max_memory_allocated(dev=None) -> int:
    m = module()
    return int(m.max_memory_allocated(dev)) if m is not None else 0


def max_memory_reserved(dev=None) -> int:
    m = module()
    return int(m.max_memory_reserved(dev)) if m is not None else 0


def memory_allocated(dev=None) -> int:
    m = module()
    return int(m.memory_allocated(dev)) if m is not None else 0


def reset_peak_memory_stats(dev=None) -> None:
    m = module()
    if m is not None:
        m.reset_peak_memory_stats(dev)


def mem_get_info(dev=None) -> tuple[int, int]:
    """(free_bytes, total_bytes); ``(0, 0)`` on CPU."""
    m = module()
    if m is None:
        return (0, 0)
    free, total = m.mem_get_info(dev)
    return int(free), int(total)


def get_device_name(index: int = 0) -> str:
    m = module()
    return str(m.get_device_name(index)) if m is not None else "cpu"


def total_memory(dev=None) -> int:
    m = module()
    if m is None:
        return 0
    return int(m.get_device_properties(dev if dev is not None else current_device()).total_memory)


def collective_backend() -> str:
    """Process-group backend for the active accelerator: ``nccl`` on CUDA,
    ``xccl`` on XPU when torch provides it, otherwise ``gloo``."""
    b = backend()
    if b == "cuda":
        return "nccl"
    if b == "xpu" and bool(getattr(dist, "is_xccl_available", lambda: False)()):
        return "xccl"
    return "gloo"


def autocast(dtype=torch.float16, enabled: bool = True):
    """``torch.autocast`` context for the active backend (``cpu`` when no
    accelerator is present; note CPU autocast supports bfloat16, not float16)."""
    b = backend()
    return torch.autocast(device_type=b, dtype=dtype, enabled=enabled)


def describe() -> dict:
    """Backend facts for logs and execution reports."""
    info = {"backend": backend(), "torch": torch.__version__, "device_count": device_count(),
            "devices": [get_device_name(i) for i in range(device_count())]}
    if info["backend"] == "cuda":
        info["cuda"] = torch.version.cuda
    if info["backend"] == "xpu":
        props = torch.xpu.get_device_properties(0)
        info["driver_version"] = getattr(props, "driver_version", None)
        info["has_fp64"] = bool(getattr(props, "has_fp64", True))
        info["total_memory_bytes"] = int(props.total_memory)
    return info
