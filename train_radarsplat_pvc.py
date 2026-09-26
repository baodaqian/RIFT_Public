#!/usr/bin/env python
"""PVC (Intel XPU) entry point for the RadarSplat baseline.

Package F of ``RIFT_PVC_Adaptation.md``. Same CLI as ``train_radarsplat.py``, so
the PVC dataset frontends substitute the script name and nothing else:

    train_rift_dataset_pvc.py --method radarsplat  ->  train_radarsplat_pvc.py --cache-root ...
                                                       --checkpoint-dir ... --fidelity-profile budget48
                                                       --no-resume|--resume
    train_gotcha_dataset_pvc.py --method radarsplat -> run_gotcha(dataset=, output_dir=, config=,
                                                                 device=, resume=)

Routing (as in the original):

* ``--fidelity-profile upstream|budget48`` (the production recipe) dispatches
  to the released engine, here ``rift_pvc.radarsplat_release_training.main``:
  the unchanged ``rift/radarsplat_release_training.py`` with the reference
  loader, device default, train/readout twins and the checkpoint sidecar
  rebound. The renderer is the authors' fork on torch mirrors of its five CUDA
  ops (``fork_torch_mirror_xpu_v1``) with the fused-SSIM torch twin
  (``fused_ssim_torch_v1``); every checkpoint directory carries ``backend.json``.
* ``legacy`` / ``audit_v1`` (historical profiles) run the unchanged
  ``train_radarsplat`` module with only its accelerator-specific helpers
  rebound: ``_emit_phase_resource`` (adds ``xpu_max_memory_*``),
  ``_validate_args`` (XPU availability gate) and ``parse_args`` (device
  default). The two inline ``if device.type == "cuda"`` telemetry blocks of that
  path are skipped on XPU.

``train_rift_dataset*.py`` never passes ``--device`` for RadarSplat, so the
device comes from the ``--device`` default, which is ``accelerator.device()``
here. By default this entry point refuses to run without an XPU device; set
``RIFT_PVC_ALLOW_BACKEND=cpu`` (tests) or ``=cuda`` to override.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from rift_pvc import accelerator  # noqa: E402
import train_radarsplat as _trainer  # noqa: E402  (the unchanged trainer)

# Literal capability declaration read by train_gotcha_dataset*.py via ast.literal_eval.
# The required keys equal train_radarsplat.GOTCHA_BACKEND (asserted in the PVC tests);
# the two descriptive fields name the PVC backend instead of the CUDA extension (D5).
GOTCHA_BACKEND = {
    "schema": "rift_gotcha_backend_v1", "method": "radarsplat", "callable": "run_gotcha",
    "selection_unit": "pass_sector", "joint_passes": True,
    "native_frequency_policy": "ragged_exact", "polarizations": ["hh", "hv", "vh", "vv"],
    "metric_domain": "clipped train-normalized native sector MF power",
    "fidelity_status": "released_source_with_native_MF_conversion_pvc_torch_mirror_unvalidated",
    "runtime_requirements": "pinned RadarSplat fork source with rift_pvc torch mirrors of its five CUDA ops "
                            "(fork_torch_mirror_xpu_v1) and fused_ssim_torch_v1; no CPU model fallback",
}

RELEASED_PROFILES = ("upstream", "budget48")


def run_gotcha(*, dataset, output_dir, config, device, resume):
    from rift_pvc.radarsplat_gotcha import run_gotcha as backend
    return backend(dataset=dataset, output_dir=output_dir, config=config, device=device, resume=resume)


def check_backend() -> str:
    allowed = {"xpu"} | {b.strip() for b in os.environ.get("RIFT_PVC_ALLOW_BACKEND", "").split(",") if b.strip()}
    backend = accelerator.backend()
    if backend not in allowed:
        raise RuntimeError(
            f"train_radarsplat_pvc.py is the PVC entry point and found backend {backend!r}; "
            "use train_radarsplat.py on CUDA, or set RIFT_PVC_ALLOW_BACKEND=cpu|cuda to override")
    return backend


def _profile(tokens) -> str:
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--fidelity-profile", default="legacy")
    selected, _ = selector.parse_known_args(tokens)
    return selected.fidelity_profile


def _device_in(argv) -> bool:
    return any(a == "--device" or a.startswith("--device=") for a in argv)


# --- legacy/audit_v1 path: the unchanged trainer with three helpers rebound ---
def _emit_phase_resource(*, device: torch.device, phase: str, optimizer_updates_this_invocation: int) -> None:
    """Twin of train_radarsplat._emit_phase_resource: CUDA fields on CUDA, XPU fields on XPU."""
    import json
    payload: dict[str, object] = {
        "phase": phase,
        "optimizer_updates_this_invocation": int(optimizer_updates_this_invocation),
        "process_peak_rss_kib": _trainer._process_peak_rss_kib(),
    }
    if device.type == "cuda":
        payload["cuda_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        payload["cuda_max_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    elif device.type == "xpu":
        payload["xpu_max_memory_allocated_bytes"] = int(accelerator.max_memory_allocated(device))
        payload["xpu_max_memory_reserved_bytes"] = int(accelerator.max_memory_reserved(device))
    print("RADARSPLAT_B7873200_TRAIN_RESOURCE_JSON=" + json.dumps(payload, sort_keys=True), flush=True)


_ORIGINAL_VALIDATE_ARGS = _trainer._validate_args


def _validate_args(args: argparse.Namespace) -> torch.device:
    device = _ORIGINAL_VALIDATE_ARGS(args)
    if device.type == "xpu" and not accelerator.is_available():
        raise RuntimeError("RadarSplat B7873200 requested XPU but no XPU device is available")
    return device


LEGACY_REBIND = {"_emit_phase_resource": _emit_phase_resource, "_validate_args": _validate_args}


def install_legacy():
    missing = [name for name in LEGACY_REBIND if not callable(getattr(_trainer, name, None))]
    if missing:
        raise RuntimeError(f"train_radarsplat.py no longer defines {missing}; train_radarsplat_pvc.py must be revisited")
    for name, twin in LEGACY_REBIND.items():
        setattr(_trainer, name, twin)
    return _trainer


def parse_args(argv=None):
    """``train_radarsplat.parse_args`` with the PVC engine parser / device default."""
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    if _profile(argv) in RELEASED_PROFILES:
        from rift_pvc.radarsplat_release_training import parse_args as released_args
        return released_args(argv)
    if not _device_in(argv):
        argv = argv + ["--device", str(accelerator.device())]
    return _trainer.parse_args(argv)


def main(argv=None):
    backend = check_backend()
    tokens = list(sys.argv[1:] if argv is None else argv)
    print(f"train_radarsplat_pvc.py: {accelerator.describe()}", flush=True)
    if backend == "xpu":
        print("train_radarsplat_pvc.py: PYTORCH_DEBUG_XPU_FALLBACK="
              f"{os.environ.get('PYTORCH_DEBUG_XPU_FALLBACK', 'unset')} SYCL_CACHE_PERSISTENT="
              f"{os.environ.get('SYCL_CACHE_PERSISTENT', 'unset')}", flush=True)
    if _profile(tokens) in RELEASED_PROFILES:
        from rift_pvc.radarsplat_release_training import main as released_main
        return released_main(tokens)
    install_legacy()
    args = parse_args(tokens)
    _trainer.parse_args = lambda _argv=None: args
    accelerator.reset_peak_memory_stats()
    return _trainer.main(tokens)


if __name__ == "__main__":
    # Same convention as train_radarsplat.py: main() is called for its effects.
    # The released engine's clean interruption is SystemExit(143), which propagates.
    main()
