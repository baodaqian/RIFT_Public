"""PVC twin of ``rift/radarsplat_release_training.py`` (the released RadarSplat engine).

The engine is imported unchanged; importing this module rebinds, in the
engine's own namespace, only what the CUDA API forces (contract 1.5, item 4):

    load_cuda_reference  -> rift_pvc.radarsplat_xpu_backend.load_xpu_reference
    parse_args           -> same parser, ``--device`` defaults to the accelerator device
    train                -> sidecar gate (refuse continuing a CUDA-renderer checkpoint),
                            XPU reference injection, then the original ``train``
    readout              -> the original readout plus the backend record (D5)
    ReleasedPreprocessing-> rift_pvc.radarsplat_release.ReleasedPreprocessing (release FFT on the
                            accelerator instead of CUDA, F-dev1)
    release_loss         -> the original loss behind a transparent per-update timer
                            (``RADARSPLAT_PVC_UPDATE_TIMING_JSON=`` lines every
                            ``RIFT_PVC_RADARSPLAT_TIMING_EVERY`` updates, default 10; 0 disables)
    train_radarsplat._atomic_torch_save
                         -> writes ``backend.json`` beside every release-schema checkpoint
                            before the checkpoint lands, so no PVC checkpoint exists without it

Recipes, identity, budgets, sealed roles, recovery schema and every gate of the
original are untouched; ``train(cache, output, *, device, resume, rendering,
fused_ssim, profile, intensity_mapping)`` keeps the engine's signature.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Mapping

import torch

import rift.radarsplat_release_training as _engine
import train_radarsplat as _lifecycle
from rift.radarsplat_release import PROFILES, SCHEMA
from rift.radarsplat_release_training import (  # noqa: F401  (re-exports of device-free engine parts)
    evaluate, export_geometry, identity_for_cache, read_view, renderer_for_cache, restore_state,
)
from rift_pvc import accelerator
from rift_pvc.radarsplat_release import ReleasedPreprocessing
from rift_pvc.radarsplat_xpu_backend import (
    BACKEND_IDENTITY, SIDECAR_NAME, SSIM_IDENTITY, check_resume_sidecar, load_xpu_reference, read_sidecar,
    write_sidecar,
)

_ORIGINALS_ATTRIBUTE = "__rift_pvc_originals__"


def _originals() -> dict:
    stored = getattr(_engine, _ORIGINALS_ATTRIBUTE, None)
    if stored is None:
        stored = {
            "train": _engine.train,
            "readout": _engine.readout,
            "parse_args": _engine.parse_args,
            "load_cuda_reference": _engine.load_cuda_reference,
            "release_loss": _engine.release_loss,
            "ReleasedPreprocessing": _engine.ReleasedPreprocessing,
            "_atomic_torch_save": _lifecycle._atomic_torch_save,
        }
        setattr(_engine, _ORIGINALS_ATTRIBUTE, stored)
    return stored


ORIGINAL = _originals()


class UpdateTimer:
    """Wall time between successive ``release_loss`` calls, i.e. one full update each."""

    def __init__(self) -> None:
        self.count = 0
        self.first = None
        self.last = None

    def reset(self) -> None:
        self.__init__()

    def tick(self) -> None:
        now = time.perf_counter()
        self.count += 1
        if self.first is None:
            self.first = now
        every = int(os.environ.get("RIFT_PVC_RADARSPLAT_TIMING_EVERY", "10") or 0)
        timed = self.count - 1
        if every > 0 and timed > 0 and timed % every == 0:
            elapsed = now - self.first
            print("RADARSPLAT_PVC_UPDATE_TIMING_JSON=" + json.dumps({
                "updates_timed": timed, "seconds": round(elapsed, 3),
                "mean_seconds_per_update": round(elapsed / timed, 4),
                "last_update_seconds": round(now - self.last, 4)}, sort_keys=True), flush=True)
        self.last = now


TIMER = UpdateTimer()


def release_loss(power, occupancy, target, labels, splats, fused_ssim):
    """The original objective; the timer only observes the call."""
    TIMER.tick()
    return ORIGINAL["release_loss"](power, occupancy, target, labels, splats, fused_ssim)


def _atomic_torch_save(path, payload) -> None:
    """Sidecar before checkpoint: a release-schema checkpoint never exists without ``backend.json``."""
    if isinstance(payload, Mapping) and payload.get("schema") == SCHEMA:
        write_sidecar(Path(path).parent)
    return ORIGINAL["_atomic_torch_save"](path, payload)


def train(cache, output, *, device, resume=True, rendering=None, fused_ssim=None, profile="upstream",
          intensity_mapping=None):
    """The original engine loop on the PVC backend."""
    output = Path(output)
    check_resume_sidecar(output)
    if rendering is None or fused_ssim is None:
        rendering, fused_ssim = load_xpu_reference(device=device)
    TIMER.reset()
    result = ORIGINAL["train"](cache, output, device=device, resume=resume, rendering=rendering,
                               fused_ssim=fused_ssim, profile=profile, intensity_mapping=intensity_mapping)
    # Keep the original summary schema (GOTCHA validates it exactly); attach
    # the separate backend record to the returned report instead.
    result["pvc"] = dict(read_sidecar(output))
    return result


def readout(checkpoint, *, checkpoint_path, cache_root, device, role, object_name=None, geometry_path=None, cache=None):
    """The original readout (rendered on the torch mirror) plus the backend record."""
    install()
    result = ORIGINAL["readout"](checkpoint, checkpoint_path=checkpoint_path, cache_root=cache_root, device=device,
                                 role=role, object_name=object_name, geometry_path=geometry_path, cache=cache)
    sidecar = read_sidecar(Path(checkpoint_path).parent)
    result["pvc"] = dict(readout_backend=BACKEND_IDENTITY, readout_ssim=SSIM_IDENTITY,
                         checkpoint_backend=sidecar if sidecar is not None else
                         f"released CUDA renderer (no {SIDECAR_NAME} beside the checkpoint)")
    return result


def parse_args(argv=None):
    """The engine's parser with the accelerator device as the ``--device`` default."""
    parser = argparse.ArgumentParser(description=_engine.__doc__)
    parser.add_argument("--fidelity-profile", choices=tuple(PROFILES), default="upstream")
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--device", default=str(accelerator.device()))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    args.steps, args.init_num_gaussians = 2000, PROFILES[args.fidelity_profile]
    return args


def install() -> None:
    """Rebind the accelerator-specific names in the unchanged engine (idempotent)."""
    _engine.load_cuda_reference = load_xpu_reference
    _engine.parse_args = parse_args
    _engine.train = train
    _engine.readout = readout
    _engine.release_loss = release_loss
    _engine.ReleasedPreprocessing = ReleasedPreprocessing
    _lifecycle._atomic_torch_save = _atomic_torch_save


def uninstall() -> None:
    """Restore the originals (tests)."""
    for name, value in ORIGINAL.items():
        target = _lifecycle if name == "_atomic_torch_save" else _engine
        setattr(target, name, value)


def main(argv=None):
    """The engine's ``main`` on the PVC backend (reference loader, device default, train twin)."""
    install()
    return _engine.main(argv)


install()

__all__ = [
    "SCHEMA", "PROFILES", "evaluate", "export_geometry", "identity_for_cache", "read_view", "renderer_for_cache",
    "restore_state", "train", "readout", "parse_args", "main", "release_loss", "install", "uninstall", "TIMER",
    "ORIGINAL",
]
