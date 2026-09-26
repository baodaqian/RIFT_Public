#!/usr/bin/env python
"""PVC (Intel XPU) entry point for the Sugavanam--Ertin B787-3200 Stage 2.

Audit C.5. Rebinds only the accelerator-specific names of the unchanged
``train_sugavanam_ertin_stage2`` module, in this process only:

    build_parser      -> --device defaults to the active accelerator
    validate_args     -> device guard against the active backend
    _set_seed         -> accelerator.manual_seed_all
    sample_roi        -> CPU-generator twin (rift_pvc.sugavanam_ertin_stage2_sampling)
    torch             -> CpuGeneratorTorch: CPU generator for XPU, matching draws
    refresh_iso_points_strict -> PVC twin with CPU-generator proposal/subset draws

**Why a CPU generator.** ``torch.Generator(device="xpu")`` never finishes
compiling on PVC (dispatch rule 5), and a CPU generator cannot be passed to a
call that allocates on the device. The draws therefore happen on CPU and the
samples are moved. Recorded consequence: an XPU Stage-2 run is
state-compatible with, but **not trajectory-identical to**, a CUDA run.

**Why no ``torch_xpu`` RNG key.** ``_validate_rng_state`` rejects any payload
key outside ``{python, numpy, torch_cpu, torch_cuda}``. Adding an XPU twin
would make PVC checkpoints unreadable by the unchanged CUDA code. Because the
generator is a CPU generator, the original's own ``include_cuda=False`` branch
(its CPU lane, L405) is taken and the payload stays schema-valid.

**This lane's inputs are not on ACES.** ``B7873200Stage2Contract`` is bound to
Georgia Tech PACE paths (``/storage/scratch1/1/dbao31/...`` for the Stage-1
bundle and output dir, ``/storage/home/hcoda1/...`` for the canonical NPZ).
None of them exist on this cluster, so this entry point cannot be run end to
end here on any backend -- the original refuses first, on its own contract.
The adaptation above is unit-tested in
``rift_pvc/tests/test_sugavanam_ertin_pvc.py`` instead, and the fact is
reported in RIFT_PVC_Adaptation.md rather than hidden behind a passing smoke.
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
from rift_pvc import sugavanam_ertin_stage2_sampling as sampling  # noqa: E402
from rift_pvc.sugavanam_ertin_stage2_refresh import refresh_iso_points_strict  # noqa: E402
import train_sugavanam_ertin_stage2 as _stage2  # noqa: E402  (unchanged)

_ORIGINAL = {name: getattr(_stage2, name, None) for name in
             ("build_parser", "parse_args", "validate_args", "_set_seed", "sample_roi",
              "refresh_iso_points_strict", "torch")}


def build_parser():
    """The original parser with ``--device`` defaulting to the active backend."""
    parser = _ORIGINAL["build_parser"]()
    for action in parser._actions:
        if action.dest == "device":
            action.default = str(accelerator.device())
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def validate_args(args, contract):
    """The original validation, with the CUDA-availability guard generalized."""
    if args.resume is not None and not _stage2._same_path(args.resume, contract.latest_checkpoint):
        raise _stage2.Stage2ContractError("resume must name this B787 Stage-2 latest checkpoint")
    try:
        device = torch.device(args.device)
    except (TypeError, RuntimeError) as exc:
        raise _stage2.Stage2ContractError(f"invalid torch device: {args.device!r}") from exc
    if accelerator.is_accelerator(device) and device.type != accelerator.backend():
        raise _stage2.Stage2ContractError(
            f"{device.type.upper()} was requested but the active backend is {accelerator.backend()}")


def _set_seed(seed: int) -> None:
    """The original seeding with ``torch.cuda`` replaced by the shim."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    accelerator.manual_seed_all(seed)


def install():
    """Rebind the accelerator-specific names in the Stage-2 module namespace."""
    missing = [name for name, value in _ORIGINAL.items() if value is None]
    if missing:
        raise RuntimeError(
            f"train_sugavanam_ertin_stage2.py no longer defines {missing}; "
            "train_sugavanam_ertin_stage2_pvc.py must be revisited")
    _stage2.build_parser = build_parser
    _stage2.parse_args = parse_args
    _stage2.validate_args = validate_args
    _stage2._set_seed = _set_seed
    _stage2.sample_roi = sampling.sample_roi
    _stage2.refresh_iso_points_strict = refresh_iso_points_strict
    # The imported refresh function has its own globals, so it needs its twin too.
    _stage2.torch = sampling.CpuGeneratorTorch(_ORIGINAL["torch"])
    return _stage2


def uninstall():
    """Restore the original names, so a test cannot leak the rebinding."""
    for name, value in _ORIGINAL.items():
        setattr(_stage2, name, value)


def check_backend():
    allowed = {"xpu"} | {b.strip() for b in os.environ.get("RIFT_PVC_ALLOW_BACKEND", "").split(",") if b.strip()}
    backend = accelerator.backend()
    if backend not in allowed:
        raise RuntimeError(
            f"train_sugavanam_ertin_stage2_pvc.py is the PVC entry point and found backend "
            f"{backend!r}; use train_sugavanam_ertin_stage2.py on CUDA, or set "
            "RIFT_PVC_ALLOW_BACKEND=cpu|cuda to override")
    return backend


def main(argv=None):
    check_backend()
    install()
    print(f"train_sugavanam_ertin_stage2_pvc.py: {accelerator.describe()}", flush=True)
    print("train_sugavanam_ertin_stage2_pvc.py: sampling draws on a CPU generator; "
          "this run is state-compatible with, not trajectory-identical to, a CUDA run.",
          flush=True)
    return _stage2.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
