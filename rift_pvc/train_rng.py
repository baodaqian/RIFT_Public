"""Accelerator-agnostic twins of ``train.py``'s ``set_seed``,
``capture_rng_state`` and ``restore_rng_state``.

They reuse ``train.py``'s own serialization helpers and its module-level
frequency generator ``_FREQ_RNG`` (imported, not copied), so on a CUDA host the
checkpoint payload is byte-identical to what ``train.py`` writes (key
``torch_cuda_all``). On PVC the device payload is stored under
``torch_xpu_all`` and the originating backend is recorded.

Cross-backend resume: a checkpoint carrying another backend's RNG payload
(a CUDA checkpoint opened on PVC) cannot be restored trajectory-identically.
With ``require_complete=True`` (adaptive-capacity-v2 production resumes) this
is refused, exactly as the original refuses a CUDA topology mismatch; otherwise
the CPU-side generators are restored, a note is printed and ``False`` is
returned, matching the original's contract that ``True`` means exact restore.
"""
from __future__ import annotations

import random

import numpy as np
import torch

import train as _train  # the unchanged trainer module
from rift_pvc import accelerator

_DEVICE_KEYS = ("torch_cuda_all", "torch_xpu_all")


def set_seed(seed_value=42):
    torch.manual_seed(seed_value)
    accelerator.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    _train._FREQ_RNG.manual_seed(seed_value)


def capture_rng_state():
    """Capture every RNG that can alter a resumed training trajectory."""
    state = {
        "version": 2,
        "python": random.getstate(),
        "numpy": _train._serialize_numpy_rng_state(),
        "torch_cpu": torch.get_rng_state(),
        "freq_cpu": _train._FREQ_RNG.get_state(),
    }
    if accelerator.is_available():
        state[accelerator.rng_state_key()] = accelerator.get_rng_state_all()
        state["accelerator_backend"] = accelerator.backend()
    return state


def restore_rng_state(state, *, require_complete=False):
    """Restore :func:`capture_rng_state`; return ``True`` when exact restore succeeded."""
    required = {"python", "numpy", "torch_cpu", "freq_cpu"}
    missing = required - set(state or {})
    if missing:
        if require_complete:
            raise ValueError(
                "adaptive-capacity-v2 resume requires a complete RNG payload; "
                f"checkpoint is missing {sorted(missing)}")
        print("NOTE: checkpoint has no complete RNG payload; legacy resume is state-compatible "
              "but not trajectory-identical.")
        return False
    exact = True
    try:
        python_state = state["python"]
        numpy_state = _train._decode_numpy_rng_state(state["numpy"])
        cpu_state = _train._cpu_rng_tensor(state["torch_cpu"], "torch_cpu")
        freq_state = _train._cpu_rng_tensor(state["freq_cpu"], "freq_cpu")
        key = accelerator.rng_state_key()
        saved_device = state.get(key)
        foreign = [k for k in _DEVICE_KEYS if k != key and state.get(k) is not None]
        if saved_device is not None:
            if not isinstance(saved_device, (list, tuple)):
                raise ValueError(f"{key} must be a list of RNG byte tensors")
            saved_device = [_train._cpu_rng_tensor(value, f"{key}[{index}]")
                            for index, value in enumerate(saved_device)]
            if not accelerator.is_available() or len(saved_device) != accelerator.device_count():
                raise ValueError(
                    f"checkpoint {accelerator.backend().upper()} RNG topology does not match this process "
                    f"(saved {len(saved_device)}, current "
                    f"{accelerator.device_count() if accelerator.is_available() else 0})")
        elif foreign:
            origin = state.get("accelerator_backend") or foreign[0].split("_")[1]
            if require_complete:
                raise ValueError(
                    f"checkpoint carries {origin} RNG state ({foreign}) but this worker runs on "
                    f"{accelerator.backend()}; adaptive-capacity-v2 resume across accelerator "
                    "families is refused (state-compatible, not trajectory-identical)")
            print(f"NOTE: checkpoint RNG state comes from {origin}; restoring CPU generators only, "
                  f"the {accelerator.backend()} stream is not trajectory-identical.")
            exact = False
        elif accelerator.is_available() and require_complete:
            raise ValueError(
                f"adaptive-capacity-v2 resume requires {accelerator.backend().upper()} RNG state "
                f"on a {accelerator.backend().upper()} worker")
        # Validate every payload before mutating any generator, then restore.
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        _train._FREQ_RNG.set_state(freq_state)
        if saved_device is not None:
            accelerator.set_rng_state_all(saved_device)
    except (TypeError, ValueError, RuntimeError) as exc:
        if require_complete:
            raise ValueError(f"adaptive-capacity-v2 RNG restore failed: {exc}") from exc
        print(f"NOTE: checkpoint RNG payload could not be restored exactly: {exc}")
        return False
    return exact
