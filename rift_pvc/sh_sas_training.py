"""Package H device/RNG twins; scientific code stays in train_sh_sas."""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, Sequence

import numpy as np
import torch

import train_sh_sas as base
from rift.calibration import GlobalComplexGain
from rift_pvc import accelerator
from rift_pvc.sh_sas import BACKEND, SHSASField

CONTRACT_VERSION = base.CONTRACT_VERSION
original_parse_args = base.parse_args
original_checkpoint_payload = base.checkpoint_payload


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not any(t == "--device" or t.startswith("--device=") for t in argv):
        argv.extend(["--device", str(accelerator.device())])
    return original_parse_args(argv)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    accelerator.manual_seed_all(seed)


def require_backend(device=None):
    backend = accelerator.backend()
    requested = torch.device(device or backend).type
    allowed = {"xpu"} | set(os.environ.get("RIFT_PVC_ALLOW_BACKEND", "").split(","))
    if requested not in allowed or requested != backend:
        raise RuntimeError(
            f"PVC SH-SAS requires an XPU backend (requested {requested}, active {backend}); "
            "for other backends set both RIFT_ACCELERATOR and RIFT_PVC_ALLOW_BACKEND")
    return backend


def restore_device_rng_state(state):
    backend = accelerator.backend()
    if backend == "cpu":
        return
    payload = state.get(backend + "_rng_state")
    if payload is None:
        print(f"Checkpoint has no {backend} RNG state; retaining this device's state.", flush=True)
        return
    # map_location=device moves byte tensors too; RNG setters need CPU.
    accelerator.set_rng_state_all([value.cpu() for value in payload])


def install():
    """Idempotent rebindings in the original trainer's process-local namespace."""
    for name in ("parse_args", "set_seed", "checkpoint_payload", "restore_device_rng_state", "SHSASField"):
        setattr(base, name, globals()[name])
    return base


def checkpoint_payload(
    model: SHSASField,
    gain: GlobalComplexGain,
    optimizer,
    step: int,
    best_val: float,
    rng: np.random.Generator,
    history: Sequence[Dict[str, float]],
    args,
) -> Dict[str, object]:
    return {
        "sh_sas_contract_version": CONTRACT_VERSION,
        "sealed_npz_protocol_contract": getattr(args, "sealed_npz_protocol_contract", None),
        "step": int(step),
        "epoch": int(step),
        "loss": float(best_val),
        "best_val_rel_mse": float(best_val),
        "scene_repr": "hash_sh",
        "extent": float(args.extent),
        "granularity": int(args.granularity),
        "sh_degree": int(args.sh_degree),
        "geometry_truth_used_for_training": False,
        "paper_architecture": model.paper_architecture,
        "paper_terms": {
            "dc_density": True,
            "dc_normals": True,
            "lambertian": bool(args.lambertian),
            "tx_rx_transmittance": bool(args.occlusion),
            "opacity_key": args.opacity_key,
            "opacity_normalize_radar_adaptation": bool(args.opacity_normalize),
        },
        # Existing geometry tooling consumes these dense coefficient tensors.
        "model_state_dict": model.geometry_compatibility_state(args.query_chunk),
        "sh_sas_state_dict": model.state_dict(),
        "gain_state_dict": gain.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": accelerator.get_rng_state_all() if accelerator.backend() == "cuda" else None,
        "xpu_rng_state": accelerator.get_rng_state_all() if accelerator.backend() == "xpu" else None,
        "accelerator_backend": accelerator.backend(),
        "sh_sas_backend": BACKEND if accelerator.backend() == "xpu" else "sh_sas_torch_original",
        "numpy_rng_state_json": json.dumps(rng.bit_generator.state),
        "history": list(history),
        "args": vars(args),
    }
