#!/usr/bin/env python3
"""PVC twin of ``scripts/prepare_radarsplat_b7873200_targets.py`` (first production RadarSplat command).

The unchanged script is imported and only its accelerator-specific names are
rebound in its namespace: ``_emit_prepare_resource`` (adds ``xpu_max_memory_*``
on XPU, keeps the CUDA fields on CUDA), the ``--device`` default (injected as
``--device <accelerator.device()>`` when the caller omits the flag, as the
production planner does) and an XPU availability gate beside the original CUDA
one. The matched-filter conversion, grid policy, cache layout, resume marker
and telemetry are the original's.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from rift_pvc import accelerator  # noqa: E402
import scripts.prepare_radarsplat_b7873200_targets as _prepare  # noqa: E402  (unchanged script)


def _emit_prepare_resource(*, device: torch.device, materialized: int, reused: int, phase: str = "prepare") -> None:
    payload: dict[str, object] = {
        "phase": phase,
        "targets_newly_materialized": int(materialized),
        "targets_reused": int(reused),
        "process_peak_rss_kib": int(_prepare._process_peak_rss_kib() or 0),
    }
    if device.type == "cuda":
        payload["cuda_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        payload["cuda_max_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    elif device.type == "xpu":
        payload["xpu_max_memory_allocated_bytes"] = int(accelerator.max_memory_allocated(device))
        payload["xpu_max_memory_reserved_bytes"] = int(accelerator.max_memory_reserved(device))
    print("RADARSPLAT_B7873200_PREPARE_RESOURCE_JSON=" + json.dumps(payload, sort_keys=True), flush=True)


_ORIGINAL_VALIDATE_ARGS = _prepare._validate_args


def _validate_args(args):
    device = _ORIGINAL_VALIDATE_ARGS(args)
    if device.type == "xpu" and not accelerator.is_available():
        raise RuntimeError("RadarSplat B7873200 target preparation requested XPU, but no XPU device is available")
    return device


def _device_in(argv) -> bool:
    return any(a == "--device" or a.startswith("--device=") for a in argv)


def parse_args(argv=None):
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    if not _device_in(argv):
        argv = argv + ["--device", str(accelerator.device())]
    return _prepare.parse_args(argv)


def install():
    _prepare._emit_prepare_resource = _emit_prepare_resource
    _prepare._validate_args = _validate_args
    return _prepare


def main(argv=None) -> None:
    install()
    args = parse_args(argv)
    _prepare.parse_args = lambda _argv=None: args
    print(f"prepare_radarsplat_b7873200_targets_pvc.py: {accelerator.describe()}", flush=True)
    accelerator.reset_peak_memory_stats()
    _prepare.main(argv)


if __name__ == "__main__":
    main()
