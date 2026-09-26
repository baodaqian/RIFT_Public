#!/usr/bin/env python
"""PVC (Intel XPU) entry point for the RIFT trainer.

Runs the unchanged ``train`` module. Only its accelerator-specific functions
are rebound at import time to their ``rift_pvc`` twins:

    init_distributed   -> rift_pvc.distributed.init_distributed  (device "xpu:<rank>", backend xccl)
    all_reduce_int     -> rift_pvc.distributed.all_reduce_int
    set_seed           -> rift_pvc.train_rng.set_seed
    capture_rng_state  -> rift_pvc.train_rng.capture_rng_state    (payload key torch_xpu_all)
    restore_rng_state  -> rift_pvc.train_rng.restore_rng_state

Every argument is passed through to ``train.main`` untouched, so the recipe,
sealed roles, checkpoint layout and execution contract are those of train.py.
The rebinding happens in this process only; ``train.py`` on disk and the CUDA
pipeline are not affected.

By default this entry point refuses to run without an XPU device, so it cannot
be started by mistake in the CUDA pipeline; set ``RIFT_PVC_ALLOW_BACKEND=cpu``
(tests) or ``=cuda`` to override.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from rift_pvc import accelerator  # noqa: E402
from rift_pvc import distributed as distributed_pvc  # noqa: E402
from rift_pvc import train_rng  # noqa: E402
import train as _train  # noqa: E402  (the unchanged trainer)

REBIND = {
    "init_distributed": distributed_pvc.init_distributed,
    "all_reduce_int": distributed_pvc.all_reduce_int,
    "set_seed": train_rng.set_seed,
    "capture_rng_state": train_rng.capture_rng_state,
    "restore_rng_state": train_rng.restore_rng_state,
}


def install():
    """Rebind the accelerator-specific names in the ``train`` module namespace."""
    missing = [name for name in REBIND if not callable(getattr(_train, name, None))]
    if missing:
        raise RuntimeError(f"train.py no longer defines {missing}; train_pvc.py must be revisited")
    for name, twin in REBIND.items():
        setattr(_train, name, twin)
    return _train


def check_backend():
    allowed = {"xpu"} | {b.strip() for b in os.environ.get("RIFT_PVC_ALLOW_BACKEND", "").split(",") if b.strip()}
    backend = accelerator.backend()
    if backend not in allowed:
        raise RuntimeError(
            f"train_pvc.py is the PVC entry point and found backend {backend!r}; "
            "use train.py on CUDA, or set RIFT_PVC_ALLOW_BACKEND=cpu|cuda to override")
    return backend


def main(argv=None):
    backend = check_backend()
    install()
    print(f"train_pvc.py: {accelerator.describe()}", flush=True)
    if backend == "xpu":
        print("train_pvc.py: PYTORCH_DEBUG_XPU_FALLBACK="
              f"{os.environ.get('PYTORCH_DEBUG_XPU_FALLBACK', 'unset')} SYCL_CACHE_PERSISTENT="
              f"{os.environ.get('SYCL_CACHE_PERSISTENT', 'unset')}", flush=True)
    return _train.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
