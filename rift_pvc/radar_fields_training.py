"""Twins of the accelerator-specific functions of ``train_radar_fields.py`` (Package E).

``install()`` rebinds them, in this process only, inside the unchanged
``train_radar_fields``, ``rift.radar_fields_recipe`` and
``rift.radar_fields_upstream`` namespaces; ``train_radar_fields_pvc.py`` and
``rift_pvc.radar_fields_gotcha`` call it. Each twin exists for one CUDA touch
of the audit table in ``RIFT_PVC_Adaptation.md`` (Package E):

    set_seed                    L88-89   torch.cuda.manual_seed_all -> accelerator.manual_seed_all
    evaluate                    L1165    fork_rng over the XPU generators (the released ray
                                         sampler draws torch.rand on the device, so validation
                                         must not perturb the training stream on PVC either)
    checkpoint_payload          L1229    adds ``xpu_rng_state`` (XPU runs), ``accelerator_backend``
                                         and ``tcnn_shim``; ``cuda_rng_state`` stays as written
    validate_resume_checkpoint  L671-677 strict continuation checks ``xpu_rng_state`` on XPU
                                         exactly as the original checks ``cuda_rng_state``
    restore_device_rng_state    L1679-91 the resume block of ``main()`` per backend (the block
                                         is inline in main(), hence the copied main in the
                                         entry point)
    build_model                 L1273    the torchshim backend id builds the original RadarField
    recipe_contract             recipe   records ``model_backend: upstream-tcnn-torchshim`` and the
                                         shim version/precision (a checkpoint carrying
                                         ``upstream-tcnn`` is refused on resume and vice versa)
    check_model_backend         upstream rift_pvc.radar_fields_upstream twin
"""
from __future__ import annotations

import copy

import numpy as np
import torch

import train_radar_fields as _rf  # the unchanged trainer
from rift import radar_fields_recipe as _recipe
from rift import radar_fields_upstream as _upstream
from rift_pvc import accelerator
from rift_pvc import tcnn_torch
from rift_pvc.radar_fields_upstream import (REAL_BACKEND, TORCHSHIM_BACKEND, check_model_backend,
                                            install as install_upstream)

# Captured once, before any rebinding.
_ORIGINAL = {
    "set_seed": _rf.set_seed,
    "evaluate": _rf.evaluate,
    "checkpoint_payload": _rf.checkpoint_payload,
    "validate_resume_checkpoint": _rf.validate_resume_checkpoint,
    "build_model": _rf.build_model,
    "recipe_contract": _recipe.recipe_contract,
}
RNG_STATE_KEYS = {"cuda": "cuda_rng_state", "xpu": "xpu_rng_state"}
_installed = False


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    accelerator.manual_seed_all(seed)


def normalize_device_rng_state(saved_state, *, expected_device_count=None, require_present=False, backend="xpu"):
    """Backend-neutral twin of ``normalize_cuda_rng_state`` (same checks, same shape)."""
    key = RNG_STATE_KEYS[backend]
    name = backend.upper()
    if saved_state is None:
        if require_present:
            raise ValueError(f"strict {name} resume checkpoint lacks {key}")
        return None
    if not isinstance(saved_state, (list, tuple)):
        raise ValueError(f"{key} must be a list or tuple of RNG tensors")
    if expected_device_count is not None and len(saved_state) != int(expected_device_count):
        raise ValueError(
            f"strict {name} resume checkpoint {name} RNG topology disagrees with the current "
            f"visible device count: checkpoint={len(saved_state)}, current={expected_device_count}")
    normalized = []
    for index, state in enumerate(saved_state):
        if not torch.is_tensor(state) or state.ndim != 1 or state.dtype != torch.uint8:
            raise ValueError(f"{key} entry {index} must be a one-dimensional torch.uint8 RNG tensor")
        normalized.append(state.detach().to(device="cpu", dtype=torch.uint8).contiguous())
    return normalized


def evaluate(model, arrays, view_indices, pair_indices, xyz, ranges, stats, args, device):
    if _rf.recipe_name(args) != _rf.SOURCE_RECIPE:
        return _rf._evaluate_impl(model, arrays, view_indices, pair_indices, xyz, ranges, stats, args, device)
    kind = torch.device(device).type
    if kind == "xpu":
        devices, device_type = list(range(torch.xpu.device_count())), "xpu"
    elif kind == "cuda":
        devices, device_type = list(range(torch.cuda.device_count())), "cuda"
    else:
        devices, device_type = [], "cuda"   # the original's CPU behaviour: CPU generator only
    with torch.random.fork_rng(devices=devices, device_type=device_type):
        torch.manual_seed(0)
        return _rf._evaluate_impl(model, arrays, view_indices, pair_indices, xyz, ranges, stats, args, device)


def shim_identity(args):
    if getattr(args, "model_backend", None) != TORCHSHIM_BACKEND:
        return None
    return tcnn_torch.identity()


def validate_shim_identity(checkpoint, args):
    """Keep the saved arithmetic strategy on resume, including older v2 runs.

    Existing checkpoints already record this outside the scientific recipe.
    Validate that record instead of changing recipe/run-directory identities.
    Parity diagnostics may be updated independently and do not gate recovery.
    """
    current = shim_identity(args)
    if current is None:
        return
    saved = checkpoint.get("tcnn_shim")
    if not isinstance(saved, dict):
        raise ValueError("Radar Fields resume checkpoint lacks tcnn_shim runtime settings")
    for key in ("model_backend", "version", "precision", "weight_grad"):
        if saved.get(key) != current[key]:
            raise ValueError(f"Radar Fields shim resume mismatch: {key}: "
                             f"saved={saved.get(key)!r}, requested={current[key]!r}; "
                             "restore the saved runtime setting or start a fresh run")


def checkpoint_payload(model, optimizer, scheduler, step, best_val, xyz, stats, rng, history, args, **kwargs):
    payload = _ORIGINAL["checkpoint_payload"](model, optimizer, scheduler, step, best_val, xyz, stats, rng,
                                              history, args, **kwargs)
    kind = torch.device(args.device).type
    payload["accelerator_backend"] = kind if kind in RNG_STATE_KEYS else "cpu"
    if kind == "xpu":
        payload["xpu_rng_state"] = torch.xpu.get_rng_state_all()
    payload["tcnn_shim"] = shim_identity(args)
    return payload


def validate_resume_checkpoint(checkpoint, args, stats, current_dataset=None, current_split=None, *,
                               resume_device=None, current_sealed_protocol=None):
    validate_shim_identity(checkpoint, args)
    kind = torch.device(resume_device).type if resume_device is not None else None
    if kind != "xpu":
        return _ORIGINAL["validate_resume_checkpoint"](checkpoint, args, stats, current_dataset, current_split,
                                                       resume_device=resume_device,
                                                       current_sealed_protocol=current_sealed_protocol)
    # The original checks the CUDA payload only for a CUDA resume_device; run it
    # device-agnostically and apply the identical check to the XPU payload.
    result = _ORIGINAL["validate_resume_checkpoint"](checkpoint, args, stats, current_dataset, current_split,
                                                     resume_device=torch.device("cpu"),
                                                     current_sealed_protocol=current_sealed_protocol)
    result["xpu_rng_state_verified"] = False
    if result["strict_resume_contract"]:
        normalize_device_rng_state(checkpoint.get("xpu_rng_state"), expected_device_count=torch.xpu.device_count(),
                                   require_present=True, backend="xpu")
        result["xpu_rng_state_verified"] = True
    return result


def restore_device_rng_state(checkpoint, device, resume_validation) -> None:
    """The device branch of the resume block in ``train_radar_fields.main`` (L1679-1691)."""
    strict = bool(resume_validation["strict_resume_contract"])
    if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
        state = _rf.normalize_cuda_rng_state(
            checkpoint["cuda_rng_state"],
            expected_device_count=torch.cuda.device_count() if strict else None,
            require_present=strict)
        torch.cuda.set_rng_state_all(state)
    elif device.type == "xpu" and checkpoint.get("xpu_rng_state") is not None:
        state = normalize_device_rng_state(
            checkpoint["xpu_rng_state"],
            expected_device_count=torch.xpu.device_count() if strict else None,
            require_present=strict, backend="xpu")
        torch.xpu.set_rng_state_all(state)


def build_model(args, device: torch.device):
    if _rf.native_recipe(args) and args.model_backend in (REAL_BACKEND, TORCHSHIM_BACKEND):
        return _upstream.OriginalRadarFieldsModel(args).to(device)
    return _ORIGINAL["build_model"](args, device)


def recipe_contract(args):
    if getattr(args, "model_backend", None) != TORCHSHIM_BACKEND:
        return _ORIGINAL["recipe_contract"](args)
    mirror = copy.copy(args)
    mirror.model_backend = REAL_BACKEND
    contract = _ORIGINAL["recipe_contract"](mirror)
    contract["model_backend"] = TORCHSHIM_BACKEND
    contract["encoding"] = "original_radarfield_tcnn_torchshim"
    contract["tcnn_shim"] = {"version": tcnn_torch.SHIM_VERSION, "precision": tcnn_torch.precision_mode()}
    return contract


REBIND = {
    _rf: {"set_seed": set_seed, "evaluate": evaluate, "checkpoint_payload": checkpoint_payload,
          "validate_resume_checkpoint": validate_resume_checkpoint, "build_model": build_model,
          "recipe_contract": recipe_contract, "check_model_backend": check_model_backend},
    _recipe: {"recipe_contract": recipe_contract},
}


def install():
    """Rebind the twins in the unchanged modules (idempotent, this process only)."""
    global _installed
    if _installed:
        return
    install_upstream()
    for module, twins in REBIND.items():
        missing = [name for name in twins if not callable(getattr(module, name, None))]
        if missing:
            raise RuntimeError(f"{module.__name__} no longer defines {missing}; rift_pvc.radar_fields_training must be revisited")
        for name, twin in twins.items():
            setattr(module, name, twin)
    _installed = True
