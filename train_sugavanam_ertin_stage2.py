"""Run the corrected, recoverable Sugavanam--Ertin Stage 2.

The command has no raw-radar, mesh, STL, target-cache, or hyperparameter
arguments.  It consumes only the isolated Stage-1 terminal bundle, derives
the signed-field initialization from that cloud, and publishes a complete
geometry package atomically.  ``--resume`` is accepted solely after a recorded
clean interruption at this identity's ``checkpoint_latest.pth.tar``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from rift.sugavanam_ertin import FourierFeatureSDF, estimate_pca_normals, spatial_gradient
from rift.sugavanam_ertin_a320_stabilized import (
    ProjectionAcceptanceError,
    deterministic_boundary_shell,
    deterministic_inner_anchors,
    orient_normals_outward,
    oriented_normal_loss,
    refresh_iso_points_strict,
    signed_offset_loss,
    signed_offset_samples,
    strict_field_gate,
    topology_contract,
)
from rift.sugavanam_ertin_b7873200_stage2_v1 import (
    ARTIFACT_IDENTITY,
    CAMPAIGN_IDENTITY,
    COMPLETE_DIR,
    IMPLEMENTATION_KIND,
    LATEST_CHECKPOINT,
    METHOD_NAME,
    OUTPUT_DIR,
    POLICY_IDENTITY,
    B7873200Stage2Contract,
    Stage2ContractError,
    default_stage2_recipe,
    lifecycle_contract_record,
    load_validated_stage1_source,
    provenance_output_identity,
    stage2_provenance_record,
    validate_contract,
    validate_stage2_recipe,
)
from rift.sugavanam_ertin_stage2_runtime_v1 import (
    CLEAN_INTERRUPTION_EXIT_CODE,
    LifecyclePhase,
    LifecycleState,
    RuntimeContractError,
    atomic_json_dump,
    atomic_torch_save,
    clean_interruption_record,
    install_stop_handlers,
    load_json_mapping,
    load_torch_mapping,
    new_export_staging,
    prepare_run_root,
    promote_complete_package,
    record_terminal_failure,
    reset_stop_request,
    running_record,
    same_value,
    stop_requested,
    validate_complete_filenames,
    validate_lifecycle_record,
    write_lifecycle,
)
from rift.sugavanam_ertin_validzero import (
    analytic_sphere_sdf,
    closed_anchor_losses,
    closed_field_spec,
    evaluate_field_grid,
    field_validity_from_array,
    ramped_weight,
    sample_roi,
)


CHECKPOINT_SCHEMA = "rift_sugavanam_ertin_b7873200_stage2_checkpoint_v1"
FINAL_CHECKPOINT_NAME = "checkpoint_final.pth.tar"
SURFACE_NAME = "surface_reconstruction.npz"
SUMMARY_NAME = "run_summary.json"
STATUS_NAME = "status.json"
LIFECYCLE_NAME = "lifecycle.json"

_CHECKPOINT_FIELDS = frozenset(
    {
        "schema",
        "contract",
        "phase",
        "checkpoint_role",
        "resume_allowed",
        "step",
        "init_step",
        "max_steps",
        "model_config",
        "model_state_dict",
        "init_optimizer_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "closed_field",
        "normal_orientation_audit",
        "signed_offset_audit",
        "cloud_audit",
        "initialization",
        "best_loss",
        "history",
        "gate_history",
        "iso_refresh_history",
        "iso_points",
        "iso_normals",
        "sample_rng_state",
        "rng_state",
        "surface_audit",
    }
)
_HISTORY_FIELDS = frozenset(
    {
        "step",
        "total",
        "on",
        "normal",
        "signed",
        "off",
        "boundary",
        "inner",
        "iso",
        "iso_normal",
        "eik",
        "effective_lambda_off",
        "iso_count",
        "gate_field_min",
        "gate_boundary_min",
        "gate_shell_min",
        "lr",
        "seconds",
    }
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device only; it does not alter the scientific recipe.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _same_path(left: str | os.PathLike[str], right: str | os.PathLike[str]) -> bool:
    return os.path.realpath(os.path.abspath(os.fspath(left))) == os.path.realpath(
        os.path.abspath(os.fspath(right))
    )


def validate_args(args: argparse.Namespace, contract: B7873200Stage2Contract) -> None:
    if args.resume is not None and not _same_path(args.resume, contract.latest_checkpoint):
        raise Stage2ContractError("resume must name this B787 Stage-2 latest checkpoint")
    try:
        device = torch.device(args.device)
    except (TypeError, RuntimeError) as exc:
        raise Stage2ContractError(f"invalid torch device: {args.device!r}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise Stage2ContractError("CUDA was requested but is unavailable")


def _finite(value: float, label: str, *, positive: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0.0):
        raise RuntimeContractError(f"{label} must be finite" + (" and positive" if positive else ""))
    return value


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_rng(*, include_cuda: bool) -> dict[str, object]:
    """Capture process RNG state for the actual execution device.

    A CPU run must remain resumable on a CPU-only worker even when it happens
    to start on a host where CUDA is visible.  Conversely, a CUDA run must
    retain every visible device's state so a same-topology continuation cannot
    silently change stochastic sampling.
    """

    payload: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if include_cuda:
        if not torch.cuda.is_available():
            raise RuntimeContractError("cannot capture CUDA RNG state when CUDA is unavailable")
        payload["torch_cuda"] = torch.cuda.get_rng_state_all()
    return payload


def _restore_rng(payload: Mapping[str, object], *, require_cuda: bool) -> None:
    if not isinstance(payload, Mapping):
        raise RuntimeContractError("resume checkpoint lacks RNG state")
    required = {"python", "numpy", "torch_cpu"}
    if required.difference(payload):
        raise RuntimeContractError("resume checkpoint has incomplete RNG state")
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    cpu = payload["torch_cpu"]
    if not torch.is_tensor(cpu) or cpu.dtype != torch.uint8 or cpu.ndim != 1:
        raise RuntimeContractError("resume checkpoint CPU RNG state is invalid")
    torch.set_rng_state(cpu.detach().cpu())
    saved_cuda = payload.get("torch_cuda")
    if require_cuda:
        if not torch.cuda.is_available() or not isinstance(saved_cuda, (list, tuple)):
            raise RuntimeContractError("resume CUDA RNG state is required on a CUDA target")
        if len(saved_cuda) != torch.cuda.device_count():
            raise RuntimeContractError("resume CUDA RNG device count changed")
        restored_cuda = []
        for index, tensor in enumerate(saved_cuda):
            if not torch.is_tensor(tensor) or tensor.dtype != torch.uint8 or tensor.ndim != 1 or tensor.numel() == 0:
                raise RuntimeContractError(f"resume CUDA RNG state {index} is invalid")
            restored_cuda.append(tensor.detach().cpu())
        torch.cuda.set_rng_state_all(restored_cuda)


def _stage1_geometry(source: Mapping[str, object], recipe: Mapping[str, object], device: torch.device) -> dict[str, object]:
    points_np = np.asarray(source.get("points"), dtype=np.float32)
    magnitudes = np.asarray(source.get("magnitude"), dtype=np.float32)
    if points_np.ndim != 2 or points_np.shape[1] != 3 or len(points_np) < 3:
        raise Stage2ContractError("validated Stage-1 source has an invalid scattering cloud")
    if magnitudes.shape != (len(points_np),) or not (np.isfinite(points_np).all() and np.isfinite(magnitudes).all()):
        raise Stage2ContractError("validated Stage-1 cloud is non-finite")
    extent = _finite(source.get("extent"), "Stage-1 extent", positive=True)
    granularity = int(source.get("granularity", 0))
    if granularity < 2:
        raise Stage2ContractError("validated Stage-1 granularity is invalid")
    pitch = 2.0 * extent / granularity
    cfg = recipe
    spec = closed_field_spec(
        points_np,
        extent,
        pitch,
        radius_quantile=float(cfg["radius_quantile"]),
        radius_cap_fraction=float(cfg["radius_cap_fraction"]),
    )
    normals_np = estimate_pca_normals(points_np, radius=3.0 * pitch)
    normals_np, normal_audit = orient_normals_outward(points_np, normals_np, spec.center)
    signed_points_np, signed_targets_np, offset_audit = signed_offset_samples(
        points_np,
        normals_np,
        extent,
        float(cfg["signed_offset_pitches"]) * pitch,
    )
    shell = deterministic_boundary_shell(
        extent, pitch, int(cfg["boundary_shell_resolution"]), device
    )
    inner = deterministic_inner_anchors(spec, int(cfg["inner_anchor_count"]), device)
    return {
        "points": torch.as_tensor(points_np, device=device),
        "normals": torch.as_tensor(normals_np, device=device),
        "signed_points": torch.as_tensor(signed_points_np, device=device),
        "signed_targets": torch.as_tensor(signed_targets_np, device=device),
        "shell": shell,
        "inner": inner,
        "extent": extent,
        "granularity": granularity,
        "pitch": pitch,
        "closed_field": spec.as_dict(),
        "spec": spec,
        "normal_audit": normal_audit,
        "offset_audit": offset_audit,
        "cloud_audit": {
            "source_epoch": int(source["source_epoch"]),
            "source_checkpoint": str(source["source_checkpoint"]),
            "count": int(len(points_np)),
            "threshold": float(source["threshold"]),
            "magnitude_min": float(magnitudes.min()),
            "magnitude_max": float(magnitudes.max()),
            "extent": extent,
            "granularity": granularity,
        },
    }


def _model(recipe: Mapping[str, object], extent: float, device: torch.device) -> FourierFeatureSDF:
    return FourierFeatureSDF(
        extent=extent,
        n_fourier=int(recipe["n_fourier"]),
        fourier_scale=float(recipe["fourier_scale"]),
        hidden_dim=int(recipe["hidden_dim"]),
        n_layers=int(recipe["n_layers"]),
        seed=int(recipe["seed"]),
    ).to(device)


def _model_config(recipe: Mapping[str, object], extent: float) -> dict[str, object]:
    return {
        "extent": float(extent),
        "n_fourier": int(recipe["n_fourier"]),
        "fourier_scale": float(recipe["fourier_scale"]),
        "hidden_dim": int(recipe["hidden_dim"]),
        "n_layers": int(recipe["n_layers"]),
        "seed": int(recipe["seed"]),
    }


def _assert_finite_model_and_optimizer(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer | None
) -> None:
    for name, value in model.state_dict().items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise FloatingPointError(f"non-finite model state: {name}")
    if optimizer is None:
        return
    for parameter, state in optimizer.state.items():
        del parameter
        for name, value in state.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise FloatingPointError(f"non-finite optimizer state: {name}")


def _checkpoint_state(
    *,
    contract_record: Mapping[str, object],
    recipe: Mapping[str, object],
    phase: LifecyclePhase,
    step: int,
    init_step: int,
    model: FourierFeatureSDF,
    init_optimizer: torch.optim.Optimizer | None,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    geometry: Mapping[str, object],
    initialization: Mapping[str, object] | None,
    best_loss: float,
    history: list[dict[str, object]],
    gate_history: list[dict[str, object]],
    refresh_history: list[dict[str, object]],
    iso_points: torch.Tensor | None,
    iso_normals: torch.Tensor | None,
    generator: torch.Generator,
    resume_allowed: bool,
    surface_audit: Mapping[str, object] | None = None,
) -> dict[str, object]:
    _assert_finite_model_and_optimizer(model, init_optimizer or optimizer)
    max_steps = int(recipe["steps"])
    if phase is LifecyclePhase.INITIALIZATION and not (0 <= init_step <= int(recipe["init_steps"])):
        raise RuntimeContractError("initialization checkpoint step is invalid")
    if phase is not LifecyclePhase.INITIALIZATION and init_step != int(recipe["init_steps"]):
        raise RuntimeContractError("post-initialization checkpoint must record complete initialization")
    if not (0 <= step <= max_steps):
        raise RuntimeContractError("Stage-2 checkpoint training step is invalid")
    return {
        "schema": CHECKPOINT_SCHEMA,
        "contract": copy.deepcopy(dict(contract_record)),
        "phase": phase.value,
        "checkpoint_role": "latest",
        "resume_allowed": bool(resume_allowed),
        "step": int(step),
        "init_step": int(init_step),
        "max_steps": max_steps,
        "model_config": model.config(),
        "model_state_dict": model.state_dict(),
        "init_optimizer_state_dict": None if init_optimizer is None else init_optimizer.state_dict(),
        "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "closed_field": copy.deepcopy(geometry["closed_field"]),
        "normal_orientation_audit": copy.deepcopy(geometry["normal_audit"]),
        "signed_offset_audit": copy.deepcopy(geometry["offset_audit"]),
        "cloud_audit": copy.deepcopy(geometry["cloud_audit"]),
        "initialization": None if initialization is None else copy.deepcopy(dict(initialization)),
        "best_loss": float(best_loss),
        "history": copy.deepcopy(history),
        "gate_history": copy.deepcopy(gate_history),
        "iso_refresh_history": copy.deepcopy(refresh_history),
        "iso_points": None if iso_points is None else iso_points.detach().cpu(),
        "iso_normals": None if iso_normals is None else iso_normals.detach().cpu(),
        "sample_rng_state": generator.get_state().detach().cpu(),
        "rng_state": _capture_rng(
            include_cuda=getattr(generator, "device", torch.device("cpu")).type == "cuda"
        ),
        "surface_audit": None if surface_audit is None else copy.deepcopy(dict(surface_audit)),
    }


def _strict_int(value: object, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise RuntimeContractError(f"{label} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise RuntimeContractError(f"{label} must be >= {minimum}")
    return result


def _finite_number(value: object, label: str, *, allow_infinity: bool = False) -> float:
    if isinstance(value, bool):
        raise RuntimeContractError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeContractError(f"{label} must be numeric") from exc
    if math.isnan(result) or (not allow_infinity and not math.isfinite(result)):
        raise RuntimeContractError(f"{label} must be finite")
    return result


def _validate_finite_record(value: object, label: str) -> None:
    """Validate a checkpointed audit/history tree without coercing its types."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise RuntimeContractError(f"{label} has a non-string key")
            _validate_finite_record(child, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_finite_record(child, f"{label}[{index}]")
        return
    if value is None or isinstance(value, (str, bool, int, np.integer)):
        return
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise RuntimeContractError(f"{label} contains a non-finite value")
        return
    raise RuntimeContractError(f"{label} contains an unsupported value type")


def _validate_model_state_dict(state: object, model: FourierFeatureSDF) -> None:
    if not isinstance(state, Mapping):
        raise RuntimeContractError("Stage-2 checkpoint lacks model state")
    expected = model.state_dict()
    if set(state) != set(expected):
        raise RuntimeContractError("Stage-2 checkpoint model keys changed")
    for name, reference in expected.items():
        value = state[name]
        if not torch.is_tensor(value):
            raise RuntimeContractError(f"Stage-2 checkpoint model tensor {name} is missing")
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise RuntimeContractError(f"Stage-2 checkpoint model tensor {name} shape/dtype changed")
        if not torch.isfinite(value).all():
            raise RuntimeContractError(f"Stage-2 checkpoint model tensor {name} is non-finite")
        if name == "fourier_bands" and not torch.equal(value.detach().cpu(), reference.detach().cpu()):
            raise RuntimeContractError("Stage-2 checkpoint fixed Fourier bands changed")


def _cosine_learning_rate(recipe: Mapping[str, object], step: int) -> float:
    """Return the post-``scheduler.step()`` learning rate at one-based step."""

    total_steps = max(int(recipe["steps"]), 1)
    if not 0 <= step <= total_steps:
        raise RuntimeContractError("Stage-2 scheduler step is outside its recipe")
    base = float(recipe["lr"])
    eta_min = base * 0.01
    return eta_min + (base - eta_min) * (1.0 + math.cos(math.pi * step / total_steps)) / 2.0


def _validate_adam_state(
    state: object,
    model: FourierFeatureSDF,
    *,
    learning_rate: float,
    completed_steps: int,
    label: str,
) -> None:
    if not isinstance(state, Mapping):
        raise RuntimeContractError(f"{label} optimizer state is missing")
    groups = state.get("param_groups")
    slots = state.get("state")
    parameters = list(model.parameters())
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(groups[0], Mapping):
        raise RuntimeContractError(f"{label} optimizer groups changed")
    group = groups[0]
    ids = group.get("params")
    if not isinstance(ids, list) or len(ids) != len(parameters) or len(set(ids)) != len(ids):
        raise RuntimeContractError(f"{label} optimizer parameter layout changed")
    for key, expected in (("lr", learning_rate), ("eps", 1.0e-8), ("weight_decay", 0.0)):
        actual = _finite_number(group.get(key), f"{label} optimizer {key}")
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-18):
            raise RuntimeContractError(f"{label} optimizer {key} changed")
    betas = group.get("betas")
    if not isinstance(betas, (tuple, list)) or tuple(float(item) for item in betas) != (0.9, 0.999):
        raise RuntimeContractError(f"{label} optimizer betas changed")
    if group.get("amsgrad") is not False:
        raise RuntimeContractError(f"{label} optimizer amsgrad changed")
    if not isinstance(slots, Mapping):
        raise RuntimeContractError(f"{label} optimizer moments are missing")
    if completed_steps == 0:
        if slots:
            raise RuntimeContractError(f"{label} optimizer has moments before its first step")
        return
    if set(slots) != set(ids):
        raise RuntimeContractError(f"{label} optimizer moments do not cover every parameter")
    for parameter_id, parameter in zip(ids, parameters):
        slot = slots[parameter_id]
        if not isinstance(slot, Mapping) or set(slot) != {"step", "exp_avg", "exp_avg_sq"}:
            raise RuntimeContractError(f"{label} optimizer moment schema changed")
        step = slot["step"]
        if torch.is_tensor(step):
            if step.shape != () or not torch.isfinite(step).all():
                raise RuntimeContractError(f"{label} optimizer step is invalid")
            completed = float(step)
        else:
            completed = _finite_number(step, f"{label} optimizer step")
        if not math.isclose(completed, float(completed_steps), rel_tol=0.0, abs_tol=1.0e-6):
            raise RuntimeContractError(f"{label} optimizer step is invalid")
        for key in ("exp_avg", "exp_avg_sq"):
            tensor = slot[key]
            if not torch.is_tensor(tensor) or tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                raise RuntimeContractError(f"{label} optimizer {key} shape/dtype changed")
            if not torch.isfinite(tensor).all():
                raise RuntimeContractError(f"{label} optimizer {key} is non-finite")


def _validate_scheduler_state(state: object, recipe: Mapping[str, object], *, step: int) -> float:
    if not isinstance(state, Mapping):
        raise RuntimeContractError("Stage-2 scheduler state is missing")
    if _strict_int(state.get("T_max"), "Stage-2 scheduler T_max", minimum=1) != max(int(recipe["steps"]), 1):
        raise RuntimeContractError("Stage-2 scheduler T_max changed")
    eta_min = _finite_number(state.get("eta_min"), "Stage-2 scheduler eta_min")
    if not math.isclose(eta_min, float(recipe["lr"]) * 0.01, rel_tol=0.0, abs_tol=1.0e-18):
        raise RuntimeContractError("Stage-2 scheduler eta_min changed")
    if _strict_int(state.get("last_epoch"), "Stage-2 scheduler last_epoch", minimum=0) != step:
        raise RuntimeContractError("Stage-2 scheduler progress changed")
    base_lrs = state.get("base_lrs")
    if not isinstance(base_lrs, list) or len(base_lrs) != 1:
        raise RuntimeContractError("Stage-2 scheduler base_lrs changed")
    if not math.isclose(_finite_number(base_lrs[0], "Stage-2 scheduler base lr"), float(recipe["lr"]), rel_tol=0.0, abs_tol=1.0e-18):
        raise RuntimeContractError("Stage-2 scheduler base lr changed")
    expected_current = _cosine_learning_rate(recipe, step)
    last_lrs = state.get("_last_lr")
    if not isinstance(last_lrs, list) or len(last_lrs) != 1:
        raise RuntimeContractError("Stage-2 scheduler current learning rate is missing")
    if not math.isclose(
        _finite_number(last_lrs[0], "Stage-2 scheduler current lr"),
        expected_current,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeContractError("Stage-2 scheduler current learning rate changed")
    return expected_current


def _validate_rng_state(payload: object, *, require_cuda: bool = False) -> None:
    if not isinstance(payload, Mapping) or set(payload).difference({"python", "numpy", "torch_cpu", "torch_cuda"}):
        raise RuntimeContractError("Stage-2 checkpoint RNG payload changed")
    if not {"python", "numpy", "torch_cpu"}.issubset(payload):
        raise RuntimeContractError("Stage-2 checkpoint RNG payload is incomplete")
    if not isinstance(payload["python"], tuple) or not isinstance(payload["numpy"], tuple):
        raise RuntimeContractError("Stage-2 checkpoint Python/NumPy RNG state is invalid")
    cpu = payload["torch_cpu"]
    if not torch.is_tensor(cpu) or cpu.dtype != torch.uint8 or cpu.ndim != 1 or cpu.numel() == 0:
        raise RuntimeContractError("Stage-2 checkpoint Torch CPU RNG state is invalid")
    cuda = payload.get("torch_cuda")
    if require_cuda and cuda is None:
        raise RuntimeContractError("Stage-2 CUDA RNG state is required on a CUDA target")
    if cuda is not None:
        if not isinstance(cuda, (tuple, list)) or not all(
            torch.is_tensor(item) and item.dtype == torch.uint8 and item.ndim == 1 and item.numel() > 0
            for item in cuda
        ):
            raise RuntimeContractError("Stage-2 checkpoint Torch CUDA RNG state is invalid")
        if require_cuda and (not torch.cuda.is_available() or len(cuda) != torch.cuda.device_count()):
            raise RuntimeContractError("Stage-2 checkpoint CUDA RNG topology changed")
    try:
        random.Random().setstate(payload["python"])
        np.random.RandomState().set_state(payload["numpy"])
        torch.Generator(device="cpu").set_state(cpu.detach().cpu())
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise RuntimeContractError("Stage-2 checkpoint RNG state cannot be restored") from exc


def _validate_iso_state(points: object, normals: object) -> None:
    if points is None and normals is None:
        return
    if not (torch.is_tensor(points) and torch.is_tensor(normals)):
        raise RuntimeContractError("Stage-2 iso state must contain points and normals together")
    if points.ndim != 2 or points.shape[1] != 3 or points.shape != normals.shape:
        raise RuntimeContractError("Stage-2 iso state shape changed")
    if points.dtype != torch.float32 or normals.dtype != torch.float32:
        raise RuntimeContractError("Stage-2 iso state dtype changed")
    if not (torch.isfinite(points).all() and torch.isfinite(normals).all()):
        raise RuntimeContractError("Stage-2 iso state is non-finite")


def _validate_history(
    history: object, *, step: int, recipe: Mapping[str, object] | None = None
) -> list[Mapping[str, object]]:
    if not isinstance(history, list) or len(history) != step:
        raise RuntimeContractError("Stage-2 checkpoint history progression changed")
    validated: list[Mapping[str, object]] = []
    for expected_step, row in enumerate(history, start=1):
        if not isinstance(row, Mapping) or set(row) != _HISTORY_FIELDS:
            raise RuntimeContractError("Stage-2 checkpoint history schema changed")
        if _strict_int(row.get("step"), "Stage-2 history step", minimum=1) != expected_step:
            raise RuntimeContractError("Stage-2 checkpoint history progression changed")
        _strict_int(row.get("iso_count"), "Stage-2 history iso_count", minimum=0)
        for key in _HISTORY_FIELDS.difference({"step", "iso_count"}):
            _finite_number(row.get(key), f"Stage-2 history {key}")
        if recipe is not None and not math.isclose(
            _finite_number(row.get("lr"), "Stage-2 history lr"),
            _cosine_learning_rate(recipe, expected_step),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeContractError("Stage-2 history learning rate changed")
        validated.append(row)
    return validated


def _validate_initialization_audit(
    initialization: object,
    *,
    recipe: Mapping[str, object],
    closed_field: Mapping[str, object],
    completed_steps: int,
    require_gate: bool,
) -> Mapping[str, object]:
    if not isinstance(initialization, Mapping):
        raise RuntimeContractError("Stage-2 initialization checkpoint lacks its audit")
    expected_keys = {
        "policy",
        "steps",
        "completed_steps",
        "learning_rate",
        "batch",
        "final_loss",
        "closed_field",
        "initial_gate",
    }
    if set(initialization) != expected_keys:
        raise RuntimeContractError("Stage-2 initialization audit schema changed")
    if initialization.get("policy") != POLICY_IDENTITY:
        raise RuntimeContractError("Stage-2 initialization policy changed")
    if _strict_int(initialization.get("steps"), "initialization steps", minimum=1) != int(recipe["init_steps"]):
        raise RuntimeContractError("Stage-2 initialization step budget changed")
    if _strict_int(initialization.get("completed_steps"), "initialization completed_steps", minimum=1) != completed_steps:
        raise RuntimeContractError("Stage-2 initialization audit progress changed")
    if not math.isclose(
        _finite_number(initialization.get("learning_rate"), "initialization learning rate"),
        float(recipe["init_lr"]),
        rel_tol=0.0,
        abs_tol=1.0e-18,
    ):
        raise RuntimeContractError("Stage-2 initialization learning rate changed")
    if _strict_int(initialization.get("batch"), "initialization batch", minimum=1) != int(recipe["init_batch"]):
        raise RuntimeContractError("Stage-2 initialization batch changed")
    _finite_number(initialization.get("final_loss"), "Stage-2 initialization final loss")
    if not same_value(initialization.get("closed_field"), closed_field):
        raise RuntimeContractError("Stage-2 initialization closed field changed")
    initial_gate = initialization.get("initial_gate")
    if initial_gate is None:
        if require_gate:
            raise RuntimeContractError("Stage-2 initialization lacks its strict gate")
    else:
        if not isinstance(initial_gate, Mapping) or initial_gate.get("passed") is not True:
            raise RuntimeContractError("Stage-2 initialization strict gate is invalid")
        _validate_finite_record(initial_gate, "Stage-2 initialization strict gate")
    return initialization


def _validate_surface_audit(audit: object, *, required: bool) -> Mapping[str, object] | None:
    if audit is None:
        if required:
            raise RuntimeContractError("Stage-2 surface audit is missing")
        return None
    if not isinstance(audit, Mapping) or set(audit) != {"validity", "topology", "grid_pitch"}:
        raise RuntimeContractError("Stage-2 surface audit schema changed")
    validity = audit.get("validity")
    topology = audit.get("topology")
    if not isinstance(validity, Mapping) or validity.get("passed") is not True:
        raise RuntimeContractError("Stage-2 surface validity audit is invalid")
    if not isinstance(topology, Mapping) or topology.get("passed") is not True:
        raise RuntimeContractError("Stage-2 surface topology audit is invalid")
    _finite_number(audit.get("grid_pitch"), "Stage-2 surface grid pitch")
    _validate_finite_record(audit, "Stage-2 surface audit")
    return audit


def _validate_checkpoint(
    state: Mapping[str, object],
    *,
    contract_record: Mapping[str, object],
    recipe: Mapping[str, object],
    geometry: Mapping[str, object],
    model: FourierFeatureSDF,
    require_resumable: bool,
    require_cuda_rng: bool = False,
) -> LifecyclePhase:
    if not isinstance(state, Mapping) or state.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeContractError("Stage-2 checkpoint schema changed")
    if set(state) != _CHECKPOINT_FIELDS:
        raise RuntimeContractError("Stage-2 checkpoint fields changed")
    if not same_value(state.get("contract"), contract_record):
        raise RuntimeContractError("Stage-2 checkpoint provenance changed")
    try:
        phase = LifecyclePhase(state.get("phase"))
    except ValueError as exc:
        raise RuntimeContractError("Stage-2 checkpoint phase is invalid") from exc
    if state.get("checkpoint_role") != "latest":
        raise RuntimeContractError("only latest checkpoints can resume")
    if not isinstance(state.get("resume_allowed"), bool):
        raise RuntimeContractError("Stage-2 checkpoint resume_allowed must be boolean")
    if state.get("resume_allowed") != require_resumable:
        raise RuntimeContractError("Stage-2 checkpoint resume permission changed")
    if _strict_int(state.get("max_steps"), "Stage-2 checkpoint max_steps", minimum=1) != int(recipe["steps"]):
        raise RuntimeContractError("Stage-2 checkpoint max steps changed")
    step = _strict_int(state.get("step"), "Stage-2 checkpoint step", minimum=0)
    if step > int(recipe["steps"]):
        raise RuntimeContractError("Stage-2 checkpoint step is invalid")
    init_step = _strict_int(state.get("init_step"), "Stage-2 checkpoint init_step", minimum=0)
    if init_step > int(recipe["init_steps"]):
        raise RuntimeContractError("Stage-2 checkpoint initialization step is invalid")
    if state.get("model_config") != _model_config(recipe, geometry["extent"]):
        raise RuntimeContractError("Stage-2 model configuration changed")
    if state.get("closed_field") != geometry["closed_field"]:
        raise RuntimeContractError("Stage-1-derived closed field changed")
    if state.get("normal_orientation_audit") != geometry["normal_audit"]:
        raise RuntimeContractError("Stage-1 normal orientation changed")
    if state.get("signed_offset_audit") != geometry["offset_audit"]:
        raise RuntimeContractError("Stage-1 signed-offset construction changed")
    if state.get("cloud_audit") != geometry["cloud_audit"]:
        raise RuntimeContractError("Stage-1 cloud audit changed")
    best_loss = _finite_number(state.get("best_loss", float("nan")), "Stage-2 checkpoint best loss", allow_infinity=True)
    if best_loss == float("-inf"):
        raise RuntimeContractError("Stage-2 checkpoint best loss is non-finite")
    sample_rng = state.get("sample_rng_state")
    if not torch.is_tensor(sample_rng) or sample_rng.dtype != torch.uint8 or sample_rng.ndim != 1 or sample_rng.numel() == 0:
        raise RuntimeContractError("Stage-2 checkpoint lacks sample-generator state")
    _validate_rng_state(state.get("rng_state"), require_cuda=require_cuda_rng)
    _validate_model_state_dict(state.get("model_state_dict"), model)
    _validate_iso_state(state.get("iso_points"), state.get("iso_normals"))
    history = state.get("history")
    gates = state.get("gate_history")
    refresh = state.get("iso_refresh_history")
    if not isinstance(history, list) or not isinstance(gates, list) or not isinstance(refresh, list):
        raise RuntimeContractError("Stage-2 checkpoint history schema changed")
    _validate_finite_record(gates, "Stage-2 gate history")
    _validate_finite_record(refresh, "Stage-2 refresh history")
    _validate_history(history, step=step, recipe=recipe)
    initialization = state.get("initialization")
    if phase is LifecyclePhase.INITIALIZATION:
        if step != 0 or history or gates or refresh or state.get("iso_points") is not None:
            raise RuntimeContractError("initialization checkpoint carries training progress")
        if state.get("optimizer_state_dict") is not None or state.get("scheduler_state_dict") is not None:
            raise RuntimeContractError("initialization checkpoint carries training optimizer state")
        _validate_adam_state(
            state.get("init_optimizer_state_dict"), model,
            learning_rate=float(recipe["init_lr"]), completed_steps=init_step, label="initialization",
        )
        if init_step == 0:
            if initialization is not None:
                raise RuntimeContractError("zero-step initialization checkpoint carries an audit")
        else:
            _validate_initialization_audit(
                initialization,
                recipe=recipe,
                closed_field=geometry["closed_field"],
                completed_steps=init_step,
                require_gate=False,
            )
        _validate_surface_audit(state.get("surface_audit"), required=False)
    elif phase in (LifecyclePhase.TRAINING, LifecyclePhase.EXPORT_PENDING):
        if init_step != int(recipe["init_steps"]) or not isinstance(initialization, Mapping):
            raise RuntimeContractError("post-initialization checkpoint lacks a complete initialization")
        _validate_initialization_audit(
            initialization,
            recipe=recipe,
            closed_field=geometry["closed_field"],
            completed_steps=init_step,
            require_gate=True,
        )
        if state.get("init_optimizer_state_dict") is not None:
            raise RuntimeContractError("post-initialization checkpoint carries init optimizer state")
        current_learning_rate = _validate_scheduler_state(
            state.get("scheduler_state_dict"), recipe, step=step
        )
        _validate_adam_state(
            state.get("optimizer_state_dict"), model,
            learning_rate=current_learning_rate, completed_steps=step, label="training",
        )
        if len(history) != step or not gates:
            raise RuntimeContractError("post-initialization history/gate progression changed")
        if phase is LifecyclePhase.EXPORT_PENDING:
            if step != int(recipe["steps"]) or state.get("iso_points") is None or not refresh:
                raise RuntimeContractError("export checkpoint is missing terminal training evidence")
        _validate_surface_audit(
            state.get("surface_audit"), required=phase is LifecyclePhase.EXPORT_PENDING and state.get("surface_audit") is not None
        )
    else:
        raise RuntimeContractError("completed checkpoints cannot be used as latest resumes")
    return phase


def _write_latest_and_lifecycle(
    *,
    root: str,
    contract: B7873200Stage2Contract,
    lifecycle_contract: Mapping[str, object],
    recipe: Mapping[str, object],
    phase: LifecyclePhase,
    state: Mapping[str, object],
    clean: bool,
) -> None:
    latest = contract.latest_checkpoint
    atomic_torch_save(state, latest)
    if clean:
        lifecycle = clean_interruption_record(
            phase=phase,
            step=int(state["step"]),
            max_steps=int(recipe["steps"]),
            checkpoint_path=latest,
            expected_contract=lifecycle_contract,
        )
    else:
        lifecycle = running_record(
            phase=phase,
            step=int(state["step"]),
            max_steps=int(recipe["steps"]),
            checkpoint_path=latest,
            expected_contract=lifecycle_contract,
        )
    write_lifecycle(
        contract.lifecycle_path,
        lifecycle,
        expected_contract=lifecycle_contract,
        max_steps=int(recipe["steps"]),
    )


def _fit_initialization(
    *,
    model: FourierFeatureSDF,
    geometry: Mapping[str, object],
    recipe: Mapping[str, object],
    generator: torch.Generator,
    device: torch.device,
    start_step: int,
    optimizer: torch.optim.Optimizer,
    persist_clean: callable,
) -> tuple[int, torch.optim.Optimizer, dict[str, object]]:
    spec = geometry["spec"]
    shell = geometry["shell"]
    inner = geometry["inner"]
    total_steps = int(recipe["init_steps"])
    if not (0 <= start_step < total_steps):
        raise RuntimeContractError("initialization fitting requires an unfinished initialization step")
    final_loss: float | None = None
    model.train()
    for init_step in range(start_step, total_steps):
        completed = init_step + 1
        roi = sample_roi(int(recipe["init_batch"]), spec.extent, generator, device)
        boundary_index = torch.randint(
            len(shell), (max(int(recipe["init_batch"]) // 4, 32),), generator=generator, device=device
        )
        inner_index = torch.randint(
            len(inner), (max(int(recipe["init_batch"]) // 8, 32),), generator=generator, device=device
        )
        samples = torch.cat((roi, shell[boundary_index], inner[inner_index]), dim=0)
        target = analytic_sphere_sdf(samples, spec).clamp(-0.9 * spec.extent, 0.9 * spec.extent)
        prediction = model(samples)
        fit = F.smooth_l1_loss(prediction, target, beta=max(spec.pitch, 1.0e-6))
        boundary_loss, inner_loss = closed_anchor_losses(
            model, shell[boundary_index], inner[inner_index], float(recipe["boundary_margin"])
        )
        total = fit + boundary_loss + inner_loss
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite Stage-2 initialization at step {completed}")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        _assert_finite_model_and_optimizer(model, optimizer)
        final_loss = float(total.detach())
        if completed == 1 or completed % int(recipe["init_log_every"]) == 0 or completed == total_steps:
            print(
                f"SE B7873200 init [{completed}/{recipe['init_steps']}] loss={final_loss:.6g} "
                f"fit={float(fit.detach()):.4g} boundary={float(boundary_loss.detach()):.4g} "
                f"inner={float(inner_loss.detach()):.4g}",
                flush=True,
            )
        if stop_requested():
            persist_clean(completed, optimizer, final_loss)
            raise SystemExit(CLEAN_INTERRUPTION_EXIT_CODE)
    if final_loss is None:
        raise RuntimeContractError("initialization did not execute an optimization step")
    return total_steps, optimizer, _finalize_initialization(
        model=model,
        geometry=geometry,
        recipe=recipe,
        device=device,
        completed_steps=total_steps,
        final_loss=final_loss,
    )


def _finalize_initialization(
    *,
    model: FourierFeatureSDF,
    geometry: Mapping[str, object],
    recipe: Mapping[str, object],
    device: torch.device,
    completed_steps: int,
    final_loss: object,
) -> dict[str, object]:
    """Gate a completed initialization without taking another optimizer step.

    A clean SIGTERM can land immediately after the final initialization update.
    Its checkpoint carries a finite last loss but intentionally has no gate yet.
    On resume, this helper makes the one deterministic gate/transition without
    resetting optimizer state or replaying a training sample.
    """

    if completed_steps != int(recipe["init_steps"]):
        raise RuntimeContractError("cannot finalize an incomplete initialization")
    finite_loss = _finite_number(final_loss, "Stage-2 initialization final loss")
    model.eval()
    gate = strict_field_gate(
        model,
        geometry["extent"],
        int(recipe["gate_grid"]),
        float(recipe["boundary_margin"]),
        geometry["shell"],
        device,
        int(recipe["grid_chunk"]),
    )
    if not gate["passed"]:
        raise RuntimeError(f"Stage-2 closed initialization failed strict gate: {gate}")
    return {
        "policy": POLICY_IDENTITY,
        "steps": int(recipe["init_steps"]),
        "completed_steps": completed_steps,
        "learning_rate": float(recipe["init_lr"]),
        "batch": int(recipe["init_batch"]),
        "final_loss": finite_loss,
        "closed_field": copy.deepcopy(geometry["closed_field"]),
        "initial_gate": gate,
    }


def _append_history_row(root: str, row: Mapping[str, object]) -> None:
    path = Path(root) / "sdf_history.csv"
    columns = list(row)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row[key] for key in columns})


def _atomic_npz(path: str | os.PathLike[str], **arrays: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".npz", dir=destination.parent)
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _export_surface(
    *,
    model: FourierFeatureSDF,
    geometry: Mapping[str, object],
    recipe: Mapping[str, object],
    device: torch.device,
    path: str | os.PathLike[str],
) -> dict[str, object]:
    field, pitch = evaluate_field_grid(
        model, geometry["extent"], int(recipe["mesh_grid"]), device, int(recipe["grid_chunk"])
    )
    validity = field_validity_from_array(field, float(recipe["boundary_margin"]))
    if not validity["passed"]:
        raise RuntimeError(f"Stage-2 export field validity failed: {validity}")
    try:
        from skimage.measure import marching_cubes
    except ImportError as exc:
        raise RuntimeError("Stage-2 surface export requires scikit-image") from exc
    vertices, faces, _normals, _values = marching_cubes(field, level=0.0, spacing=(pitch, pitch, pitch))
    vertices = vertices.astype(np.float32) - float(geometry["extent"])
    faces = faces.astype(np.int32)
    topology = topology_contract(
        vertices,
        faces,
        float(geometry["extent"]),
        protected_shell_depth=2.0 * float(geometry["pitch"]),
    )
    if not topology["passed"]:
        raise RuntimeError(f"Stage-2 export topology failed: {topology}")
    audit = {"validity": validity, "topology": topology, "grid_pitch": float(pitch)}
    _atomic_npz(
        path,
        vertices=vertices,
        faces=faces,
        field=field.astype(np.float32),
        extent=np.asarray(float(geometry["extent"]), dtype=np.float64),
        validity_json=np.asarray(json.dumps(validity, sort_keys=True, allow_nan=False)),
        topology_json=np.asarray(json.dumps(topology, sort_keys=True, allow_nan=False)),
    )
    return audit


def _load_json_scalar(value: object, label: str) -> dict[str, object]:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise RuntimeContractError(f"{label} must be a scalar JSON string")
    raw = array.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise RuntimeContractError(f"{label} must be a JSON string")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeContractError(f"{label} is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeContractError(f"{label} must decode to an object")
    _validate_finite_record(payload, label)
    return payload


def _validate_surface_artifact(
    path: Path,
    *,
    model: FourierFeatureSDF,
    geometry: Mapping[str, object],
    recipe: Mapping[str, object],
    device: torch.device,
) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as surface:
        required = {"vertices", "faces", "field", "extent", "validity_json", "topology_json"}
        if set(surface.files) != required:
            raise RuntimeContractError("completion surface artifact fields changed")
        vertices = np.asarray(surface["vertices"])
        faces = np.asarray(surface["faces"])
        field = np.asarray(surface["field"])
        extent = np.asarray(surface["extent"])
        if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.dtype.kind != "f":
            raise RuntimeContractError("completion surface vertices have invalid shape or dtype")
        if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in {"i", "u"}:
            raise RuntimeContractError("completion surface faces have invalid shape or dtype")
        mesh_grid = int(recipe["mesh_grid"])
        if field.shape != (mesh_grid, mesh_grid, mesh_grid) or field.dtype.kind != "f":
            raise RuntimeContractError("completion surface field has invalid shape or dtype")
        if extent.shape != () or extent.dtype.kind != "f":
            raise RuntimeContractError("completion surface extent has invalid shape or dtype")
        saved_extent = _finite(extent.item(), "completion surface extent", positive=True)
        if not math.isclose(saved_extent, float(geometry["extent"]), rel_tol=0.0, abs_tol=1.0e-12):
            raise RuntimeContractError("completion surface extent changed")
        if not (np.isfinite(vertices).all() and np.isfinite(field).all()):
            raise RuntimeContractError("completion surface arrays are non-finite")
        validity = _load_json_scalar(surface["validity_json"], "completion surface validity")
        topology = _load_json_scalar(surface["topology_json"], "completion surface topology")

    expected_validity = field_validity_from_array(field, float(recipe["boundary_margin"]))
    if not same_value(validity, expected_validity) or validity.get("passed") is not True:
        raise RuntimeContractError("completion surface validity audit disagrees with its field")
    expected_topology = topology_contract(
        vertices,
        faces,
        float(geometry["extent"]),
        protected_shell_depth=2.0 * float(geometry["pitch"]),
    )
    if not same_value(topology, expected_topology) or topology.get("passed") is not True:
        raise RuntimeContractError("completion surface topology audit disagrees with its mesh")
    regenerated, regenerated_pitch = evaluate_field_grid(
        model,
        float(geometry["extent"]),
        mesh_grid,
        device,
        int(recipe["grid_chunk"]),
    )
    if not math.isclose(
        float(regenerated_pitch),
        2.0 * float(geometry["extent"]) / float(mesh_grid - 1),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeContractError("completion surface renderer pitch changed")
    if not np.allclose(field, regenerated, rtol=1.0e-5, atol=2.0e-6, equal_nan=False):
        raise RuntimeContractError("completion surface field does not match the final checkpoint")
    return {
        "validity": validity,
        "topology": topology,
        "grid_pitch": float(regenerated_pitch),
    }


def _validate_complete_package(
    directory: Path,
    *,
    checkpoint_contract: Mapping[str, object],
    lifecycle_contract: Mapping[str, object],
    recipe: Mapping[str, object],
    geometry: Mapping[str, object],
    device: torch.device,
) -> None:
    identity = provenance_output_identity(checkpoint_contract)
    files = validate_complete_filenames(directory)
    final = load_torch_mapping(files[FINAL_CHECKPOINT_NAME], "final checkpoint")
    if set(final) != _CHECKPOINT_FIELDS:
        raise RuntimeContractError("completion package final checkpoint fields changed")
    if final.get("schema") != CHECKPOINT_SCHEMA or final.get("phase") != LifecyclePhase.COMPLETE.value:
        raise RuntimeContractError("completion package final checkpoint is not complete")
    if final.get("checkpoint_role") != "final" or final.get("resume_allowed") is not False:
        raise RuntimeContractError("completion package final checkpoint has invalid role")
    if not same_value(final.get("contract"), checkpoint_contract):
        raise RuntimeContractError("completion package final provenance changed")
    if _strict_int(final.get("step"), "completion final step", minimum=0) != int(recipe["steps"]):
        raise RuntimeContractError("completion package final step changed")
    if _strict_int(final.get("init_step"), "completion final initialization step", minimum=0) != int(recipe["init_steps"]):
        raise RuntimeContractError("completion package final initialization changed")
    if _strict_int(final.get("max_steps"), "completion final max steps", minimum=1) != int(recipe["steps"]):
        raise RuntimeContractError("completion package final max steps changed")
    if final.get("model_config") != _model_config(recipe, geometry["extent"]):
        raise RuntimeContractError("completion package model configuration changed")
    if not same_value(final.get("closed_field"), geometry["closed_field"]):
        raise RuntimeContractError("completion package closed field changed")
    for key, expected in (
        ("normal_orientation_audit", geometry["normal_audit"]),
        ("signed_offset_audit", geometry["offset_audit"]),
        ("cloud_audit", geometry["cloud_audit"]),
    ):
        if not same_value(final.get(key), expected):
            raise RuntimeContractError(f"completion package {key} changed")
    model = _model(recipe, float(geometry["extent"]), device)
    _validate_model_state_dict(final.get("model_state_dict"), model)
    model.load_state_dict(final["model_state_dict"])
    _validate_rng_state(final.get("rng_state"), require_cuda=device.type == "cuda")
    sample_rng = final.get("sample_rng_state")
    if not torch.is_tensor(sample_rng) or sample_rng.dtype != torch.uint8 or sample_rng.ndim != 1:
        raise RuntimeContractError("completion package sample-generator state is invalid")
    _validate_iso_state(final.get("iso_points"), final.get("iso_normals"))
    history = _validate_history(final.get("history"), step=int(recipe["steps"]), recipe=recipe)
    gates = final.get("gate_history")
    refresh = final.get("iso_refresh_history")
    if not isinstance(gates, list) or not gates or not isinstance(refresh, list) or not refresh:
        raise RuntimeContractError("completion package lacks gate or iso-refresh evidence")
    _validate_finite_record(gates, "completion gate history")
    _validate_finite_record(refresh, "completion iso-refresh history")
    last_gate = gates[-1]
    if not isinstance(last_gate, Mapping) or last_gate.get("phase") != "pre_export" or last_gate.get("passed") is not True:
        raise RuntimeContractError("completion package final field gate is invalid")
    if final.get("init_optimizer_state_dict") is not None:
        raise RuntimeContractError("completion package retains an initialization optimizer")
    current_learning_rate = _validate_scheduler_state(
        final.get("scheduler_state_dict"), recipe, step=int(recipe["steps"])
    )
    _validate_adam_state(
        final.get("optimizer_state_dict"), model,
        learning_rate=current_learning_rate, completed_steps=int(recipe["steps"]), label="completion",
    )
    _validate_initialization_audit(
        final.get("initialization"),
        recipe=recipe,
        closed_field=geometry["closed_field"],
        completed_steps=int(recipe["init_steps"]),
        require_gate=True,
    )
    best_loss = _finite_number(final.get("best_loss"), "completion best loss")
    if best_loss < 0.0:
        raise RuntimeContractError("completion best loss is invalid")
    expected_audit = _validate_surface_artifact(
        files[SURFACE_NAME], model=model, geometry=geometry, recipe=recipe, device=device
    )
    final_audit = _validate_surface_audit(final.get("surface_audit"), required=True)
    if not same_value(final_audit, expected_audit):
        raise RuntimeContractError("completion final checkpoint surface audit disagrees with its files")

    summary = load_json_mapping(files[SUMMARY_NAME], "completion summary")
    expected_summary_fields = {
        "method", "campaign_identity", "artifact_identity", "policy_identity", "implementation_kind",
        "contract", "steps", "best_loss", "final_gate", "last_iso_refresh", "surface_audit",
        "ground_truth_geometry_used", "novel_view_signal_supported",
    }
    if set(summary) != expected_summary_fields:
        raise RuntimeContractError("completion summary fields changed")
    if summary.get("contract") != lifecycle_contract:
        raise RuntimeContractError("completion summary provenance changed")
    if summary.get("method") != identity["method"] or summary.get("campaign_identity") != identity["campaign_identity"]:
        raise RuntimeContractError("completion summary method identity changed")
    if summary.get("artifact_identity") != identity["artifact_identity"] or summary.get("policy_identity") != identity["policy_identity"]:
        raise RuntimeContractError("completion summary artifact identity changed")
    if summary.get("implementation_kind") != identity["implementation_kind"]:
        raise RuntimeContractError("completion summary implementation kind changed")
    if summary.get("ground_truth_geometry_used") is not False or summary.get("novel_view_signal_supported") is not False:
        raise RuntimeContractError("completion summary capability flags changed")
    if _strict_int(summary.get("steps"), "completion summary steps", minimum=1) != int(recipe["steps"]):
        raise RuntimeContractError("completion summary steps changed")
    if not math.isclose(_finite_number(summary.get("best_loss"), "completion summary best loss"), best_loss, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeContractError("completion summary best loss changed")
    if not same_value(summary.get("final_gate"), last_gate):
        raise RuntimeContractError("completion summary final gate changed")
    if not same_value(summary.get("last_iso_refresh"), refresh[-1]):
        raise RuntimeContractError("completion summary iso-refresh audit changed")
    if not same_value(summary.get("surface_audit"), expected_audit):
        raise RuntimeContractError("completion summary surface audit changed")

    status = load_json_mapping(files[STATUS_NAME], "completion status")
    expected_status_fields = {"schema", "phase", "state", "step", "max_steps", "resume_allowed", "contract"}
    if set(status) != expected_status_fields or status.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeContractError("completion status fields changed")
    if status.get("contract") != lifecycle_contract:
        raise RuntimeContractError("completion status provenance changed")
    if status.get("phase") != LifecyclePhase.COMPLETE.value or status.get("state") != LifecycleState.COMPLETE.value:
        raise RuntimeContractError("completion status is not terminal complete")
    if status.get("resume_allowed") is not False:
        raise RuntimeContractError("completion status resume contract changed")
    if _strict_int(status.get("step"), "completion status step", minimum=0) != int(recipe["steps"]):
        raise RuntimeContractError("completion status step changed")
    if _strict_int(status.get("max_steps"), "completion status max steps", minimum=1) != int(recipe["steps"]):
        raise RuntimeContractError("completion status max steps changed")


def _completion_checkpoint(state: Mapping[str, object], surface_audit: Mapping[str, object]) -> dict[str, object]:
    final = copy.deepcopy(dict(state))
    final["phase"] = LifecyclePhase.COMPLETE.value
    final["checkpoint_role"] = "final"
    final["resume_allowed"] = False
    final["surface_audit"] = copy.deepcopy(dict(surface_audit))
    return final


def _run(
    args: argparse.Namespace,
    *,
    contract: B7873200Stage2Contract,
    recipe: Mapping[str, object],
    source: Mapping[str, object] | None = None,
    provenance: Mapping[str, object] | None = None,
    engineering_override: bool = False,
    preserve_existing_stop: bool = False,
) -> int:
    """Run the frozen lane, or a fully explicit engineering smoke variant.

    The public ``main`` path keeps the frozen contract and recipe checks.  A
    bounded real-data smoke may inject its already validated Stage-1 source,
    provenance, and separate output contract through the private seam below;
    this avoids weakening the production bundle validator or making the
    production names mean two different things.
    """

    if engineering_override:
        if source is None or provenance is None:
            raise Stage2ContractError(
                "engineering Stage-2 runs require an explicit validated source and provenance"
            )
        if not isinstance(contract, Mapping) and not all(
            hasattr(contract, attribute)
            for attribute in ("output_dir", "latest_checkpoint", "complete_dir", "lifecycle_path")
        ):
            raise Stage2ContractError("engineering Stage-2 contract lacks its output paths")
        recipe = copy.deepcopy(dict(recipe))
    else:
        contract = validate_contract(contract)
        recipe = validate_stage2_recipe(recipe)
    validate_args(args, contract)
    device = torch.device(args.device)
    if source is None:
        source = load_validated_stage1_source(contract)
    if provenance is None:
        provenance = stage2_provenance_record(contract, source["stage1_record"], recipe)
    lifecycle_contract = lifecycle_contract_record(provenance)
    output_identity = provenance_output_identity(provenance)
    mode = prepare_run_root(
        contract.output_dir,
        resume=args.resume,
        expected_contract=lifecycle_contract,
        max_steps=int(recipe["steps"]),
    )
    if mode == "complete":
        complete_geometry = _stage1_geometry(source, recipe, device)
        _validate_complete_package(
            Path(contract.complete_dir),
            checkpoint_contract=provenance,
            lifecycle_contract=lifecycle_contract,
            recipe=recipe,
            geometry=complete_geometry,
            device=device,
        )
        print("B7873200 Stage-2 completion package already validated; no-op.", flush=True)
        return 0

    if preserve_existing_stop and args.resume is not None and stop_requested():
        # The full-package driver owns the signal handler and has already
        # validated this resumable Stage-2 latest checkpoint.  Preserve a
        # stop that arrived in the narrow handoff window so a repeated
        # interruption remains a clean lifecycle outcome.
        raise SystemExit(CLEAN_INTERRUPTION_EXIT_CODE)
    if not preserve_existing_stop:
        reset_stop_request()
        install_stop_handlers()
    _set_seed(int(recipe["seed"]))
    geometry = _stage1_geometry(source, recipe, device)
    model = _model(recipe, geometry["extent"], device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(recipe["seed"]))
    initialization: dict[str, object] | None = None
    history: list[dict[str, object]] = []
    gate_history: list[dict[str, object]] = []
    refresh_history: list[dict[str, object]] = []
    iso_points: torch.Tensor | None = None
    iso_normals: torch.Tensor | None = None
    best_loss = float("inf")
    start_step = 0
    init_step = 0
    init_optimizer: torch.optim.Optimizer | None = torch.optim.Adam(model.parameters(), lr=float(recipe["init_lr"]))
    optimizer: torch.optim.Optimizer | None = None
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    phase = LifecyclePhase.INITIALIZATION
    resumed_export_pending: dict[str, object] | None = None

    if mode == "resume":
        state = load_torch_mapping(contract.latest_checkpoint, "latest checkpoint")
        phase = _validate_checkpoint(
            state,
            contract_record=provenance,
            recipe=recipe,
            geometry=geometry,
            model=model,
            require_resumable=True,
            require_cuda_rng=device.type == "cuda",
        )
        model.load_state_dict(state["model_state_dict"])
        _restore_rng(state["rng_state"], require_cuda=device.type == "cuda")
        generator.set_state(state["sample_rng_state"].detach().to(device="cpu"))
        init_step = int(state["init_step"])
        start_step = int(state["step"])
        initialization = state.get("initialization")
        history = list(state.get("history", []))
        gate_history = list(state.get("gate_history", []))
        refresh_history = list(state.get("iso_refresh_history", []))
        iso_points = None if state.get("iso_points") is None else state["iso_points"].to(device)
        iso_normals = None if state.get("iso_normals") is None else state["iso_normals"].to(device)
        best_loss = float(state.get("best_loss", float("inf")))
        if phase is LifecyclePhase.INITIALIZATION:
            if state.get("init_optimizer_state_dict") is None or state.get("optimizer_state_dict") is not None:
                raise RuntimeContractError("initialization resume optimizer state is invalid")
            init_optimizer.load_state_dict(state["init_optimizer_state_dict"])
        else:
            init_optimizer = None
            optimizer = torch.optim.Adam(model.parameters(), lr=float(recipe["lr"]))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(int(recipe["steps"]), 1), eta_min=float(recipe["lr"]) * 0.01
            )
            if state.get("optimizer_state_dict") is None or state.get("scheduler_state_dict") is None:
                raise RuntimeContractError("training/export resume lacks optimizer/scheduler state")
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            if phase is LifecyclePhase.EXPORT_PENDING:
                # This is a completed trajectory awaiting only atomic export.
                # Preserve its validated checkpoint byte-for-byte in memory:
                # it contains the final gate, optimizer/scheduler/RNG state,
                # and histories that a resumed export must not rewrite.
                resumed_export_pending = copy.deepcopy(state)
        print(f"Resuming B7873200 Stage-2 at {phase.value}, step={start_step}", flush=True)

    def checkpoint(
        phase_value: LifecyclePhase,
        *,
        clean: bool,
        current_init_step: int,
        current_step: int,
        surface_audit: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        state = _checkpoint_state(
            contract_record=provenance,
            recipe=recipe,
            phase=phase_value,
            step=current_step,
            init_step=current_init_step,
            model=model,
            init_optimizer=init_optimizer if phase_value is LifecyclePhase.INITIALIZATION else None,
            optimizer=optimizer if phase_value is not LifecyclePhase.INITIALIZATION else None,
            scheduler=scheduler if phase_value is not LifecyclePhase.INITIALIZATION else None,
            geometry=geometry,
            initialization=initialization,
            best_loss=best_loss,
            history=history,
            gate_history=gate_history,
            refresh_history=refresh_history,
            iso_points=iso_points,
            iso_normals=iso_normals,
            generator=generator,
            resume_allowed=clean,
            surface_audit=surface_audit,
        )
        _write_latest_and_lifecycle(
            root=contract.output_dir,
            contract=contract,
            lifecycle_contract=lifecycle_contract,
            recipe=recipe,
            phase=phase_value,
            state=state,
            clean=clean,
        )
        return state

    try:
        if phase is LifecyclePhase.INITIALIZATION:
            assert init_optimizer is not None

            def persist_initialization_clean(completed: int, active_optimizer: torch.optim.Optimizer, last_loss: float) -> None:
                nonlocal initialization, init_step
                init_step = completed
                initialization = {
                    "policy": POLICY_IDENTITY,
                    "steps": int(recipe["init_steps"]),
                    "completed_steps": completed,
                    "learning_rate": float(recipe["init_lr"]),
                    "batch": int(recipe["init_batch"]),
                    "final_loss": last_loss,
                    "closed_field": copy.deepcopy(geometry["closed_field"]),
                    "initial_gate": None,
                }
                checkpoint(LifecyclePhase.INITIALIZATION, clean=True, current_init_step=completed, current_step=0)

            if init_step == int(recipe["init_steps"]):
                # A signal can arrive immediately after the final optimizer
                # update and before the first strict gate.  The validated
                # checkpoint already contains that finite loss and optimizer
                # state.  Gate it once without replaying a sample or update.
                _validate_initialization_audit(
                    initialization,
                    recipe=recipe,
                    closed_field=geometry["closed_field"],
                    completed_steps=init_step,
                    require_gate=False,
                )
                initialization = _finalize_initialization(
                    model=model,
                    geometry=geometry,
                    recipe=recipe,
                    device=device,
                    completed_steps=init_step,
                    final_loss=initialization["final_loss"],
                )
            else:
                init_step, init_optimizer, initialization = _fit_initialization(
                    model=model,
                    geometry=geometry,
                    recipe=recipe,
                    generator=generator,
                    device=device,
                    start_step=init_step,
                    optimizer=init_optimizer,
                    persist_clean=persist_initialization_clean,
                )
            init_optimizer = None
            phase = LifecyclePhase.TRAINING
            optimizer = torch.optim.Adam(model.parameters(), lr=float(recipe["lr"]))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(int(recipe["steps"]), 1), eta_min=float(recipe["lr"]) * 0.01
            )
            initial_gate = dict(initialization["initial_gate"])
            initial_gate.update({"step": 0, "phase": "post_initialization"})
            gate_history.append(initial_gate)
            checkpoint(LifecyclePhase.TRAINING, clean=False, current_init_step=init_step, current_step=start_step)

        assert optimizer is not None and scheduler is not None and initialization is not None
        started = time.time()
        last_gate: dict[str, object] = dict(gate_history[-1]) if gate_history else {}
        for step0 in range(start_step, int(recipe["steps"])):
            step = step0 + 1
            refresh_due = step >= int(recipe["iso_start"]) and (
                iso_points is None or (step - int(recipe["iso_start"])) % int(recipe["iso_refresh"]) == 0
            )
            if refresh_due:
                model.eval()
                pre_gate = strict_field_gate(
                    model, geometry["extent"], int(recipe["gate_grid"]), float(recipe["boundary_margin"]),
                    geometry["shell"], device, int(recipe["grid_chunk"]),
                )
                pre_gate.update({"step": step, "phase": "pre_iso_refresh"})
                gate_history.append(pre_gate)
                if not pre_gate["passed"]:
                    raise RuntimeError(f"Stage-2 pre-refresh gate failed: {pre_gate}")
                try:
                    iso_points, refresh = refresh_iso_points_strict(
                        model, geometry["points"], geometry["extent"], geometry["pitch"],
                        int(recipe["n_iso"]), generator, oversample=float(recipe["projection_oversample"]),
                    )
                except ProjectionAcceptanceError as exc:
                    failure = {"step": step, "pre_gate": pre_gate, **exc.audit}
                    refresh_history.append(failure)
                    raise RuntimeError(f"Stage-2 strict iso refresh failed: {failure}") from exc
                iso_normals_np = estimate_pca_normals(
                    iso_points.detach().cpu().numpy(), radius=3.0 * geometry["pitch"]
                )
                iso_normals_np, iso_normal_audit = orient_normals_outward(
                    iso_points.detach().cpu().numpy(), iso_normals_np, geometry["spec"].center
                )
                iso_normals = torch.as_tensor(iso_normals_np, device=device)
                post_gate = strict_field_gate(
                    model, geometry["extent"], int(recipe["gate_grid"]), float(recipe["boundary_margin"]),
                    geometry["shell"], device, int(recipe["grid_chunk"]),
                )
                post_gate.update({"step": step, "phase": "post_iso_refresh"})
                gate_history.append(post_gate)
                if not post_gate["passed"]:
                    raise RuntimeError(f"Stage-2 post-refresh gate failed: {post_gate}")
                refresh.update({"step": step, "pre_gate": pre_gate, "post_gate": post_gate, "iso_normal_orientation": iso_normal_audit})
                refresh_history.append(refresh)
                model.train()

            on_index = torch.randint(len(geometry["points"]), (int(recipe["batch_on"]),), generator=generator, device=device)
            signed_index = torch.randint(len(geometry["signed_points"]), (int(recipe["batch_signed"]),), generator=generator, device=device)
            boundary_index = torch.randint(len(geometry["shell"]), (int(recipe["batch_boundary"]),), generator=generator, device=device)
            inner_index = torch.randint(len(geometry["inner"]), (int(recipe["batch_inner"]),), generator=generator, device=device)
            p_on = geometry["points"][on_index].clone()
            n_on = geometry["normals"][on_index]
            p_off = sample_roi(int(recipe["batch_off"]), geometry["extent"], generator, device)
            f_on, g_on = spatial_gradient(model, p_on, create_graph=True)
            f_off, g_off = spatial_gradient(model, p_off, create_graph=True)
            boundary_loss, inner_loss = closed_anchor_losses(
                model, geometry["shell"][boundary_index], geometry["inner"][inner_index], float(recipe["boundary_margin"])
            )
            losses: dict[str, torch.Tensor] = {
                "on": f_on.abs().mean(),
                "normal": oriented_normal_loss(g_on, n_on),
                "signed": signed_offset_loss(
                    model,
                    geometry["signed_points"][signed_index],
                    geometry["signed_targets"][signed_index],
                    beta=max(geometry["pitch"], 1.0e-6),
                ),
                "off": torch.exp(-float(recipe["alpha_off"]) * f_off.abs()).mean(),
                "boundary": boundary_loss,
                "inner": inner_loss,
            }
            eikonal = [(1.0 - g_off.norm(dim=-1)).abs()]
            if iso_points is not None and iso_normals is not None:
                iso_index = torch.randint(len(iso_points), (int(recipe["batch_iso"]),), generator=generator, device=device)
                f_iso, g_iso = spatial_gradient(model, iso_points[iso_index].clone(), create_graph=True)
                losses["iso"] = f_iso.abs().mean()
                losses["iso_normal"] = oriented_normal_loss(g_iso, iso_normals[iso_index])
                eikonal.append((1.0 - g_iso.norm(dim=-1)).abs())
            else:
                zero = f_on.new_zeros(())
                losses["iso"] = zero
                losses["iso_normal"] = zero
            losses["eik"] = torch.cat(eikonal).mean()
            effective_off = ramped_weight(step, float(recipe["lambda_off"]), int(recipe["off_warmup"]), int(recipe["off_ramp"]))
            total = (
                float(recipe["lambda_on"]) * losses["on"]
                + float(recipe["lambda_normal"]) * losses["normal"]
                + float(recipe["lambda_signed"]) * losses["signed"]
                + effective_off * losses["off"]
                + float(recipe["lambda_eik"]) * losses["eik"]
                + float(recipe["lambda_iso"]) * losses["iso"]
                + float(recipe["lambda_iso_normal"]) * losses["iso_normal"]
                + float(recipe["lambda_boundary"]) * losses["boundary"]
                + float(recipe["lambda_inner"]) * losses["inner"]
            )
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite Stage-2 training loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            scheduler.step()
            _assert_finite_model_and_optimizer(model, optimizer)

            gate_ran = step % int(recipe["gate_every"]) == 0 or step == int(recipe["steps"])
            if gate_ran:
                model.eval()
                last_gate = strict_field_gate(
                    model, geometry["extent"], int(recipe["gate_grid"]), float(recipe["boundary_margin"]),
                    geometry["shell"], device, int(recipe["grid_chunk"]),
                )
                last_gate.update({"step": step, "phase": "post_optimizer"})
                gate_history.append(last_gate)
                model.train()
                if not last_gate["passed"]:
                    raise RuntimeError(f"Stage-2 periodic strict gate failed: {last_gate}")
            row = {
                "step": step,
                "total": float(total.detach()),
                **{key: float(value.detach()) for key, value in losses.items()},
                "effective_lambda_off": float(effective_off),
                "iso_count": 0 if iso_points is None else int(len(iso_points)),
                "gate_field_min": float(last_gate.get("field_min", float("nan"))),
                "gate_boundary_min": float(last_gate.get("boundary_min", float("nan"))),
                "gate_shell_min": float(last_gate.get("protected_shell", {}).get("minimum", float("nan"))),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "seconds": float(time.time() - started),
            }
            if not all(math.isfinite(float(row[key])) for key in ("total", "on", "normal", "signed", "off", "eik", "boundary", "inner", "lr", "seconds")):
                raise FloatingPointError("Stage-2 history contains a non-finite required metric")
            history.append(row)
            eligible_best = bool(gate_ran and iso_points is not None and row["total"] < best_loss)
            if eligible_best:
                best_loss = row["total"]
            if step == 1 or step % 5 == 0 or step == int(recipe["steps"]):
                _append_history_row(contract.output_dir, row)
                print(
                    f"SE B7873200 [{step}/{recipe['steps']}] loss={row['total']:.6g} "
                    f"on={row['on']:.4g} signed={row['signed']:.4g} off={row['off']:.4g} "
                    f"eik={row['eik']:.4g} iso={row['iso']:.4g}",
                    flush=True,
                )
            if eligible_best:
                best_state = _checkpoint_state(
                    contract_record=provenance, recipe=recipe, phase=LifecyclePhase.TRAINING,
                    step=step, init_step=init_step, model=model, init_optimizer=None, optimizer=optimizer,
                    scheduler=scheduler, geometry=geometry, initialization=initialization, best_loss=best_loss,
                    history=history, gate_history=gate_history, refresh_history=refresh_history,
                    iso_points=iso_points, iso_normals=iso_normals, generator=generator, resume_allowed=False,
                )
                best_state["checkpoint_role"] = "best"
                atomic_torch_save(best_state, Path(contract.output_dir) / "checkpoint_best.pth.tar")
            if step % int(recipe["save_every"]) == 0:
                checkpoint(LifecyclePhase.TRAINING, clean=False, current_init_step=init_step, current_step=step)
            if stop_requested():
                checkpoint(LifecyclePhase.TRAINING, clean=True, current_init_step=init_step, current_step=step)
                print("B7873200 Stage-2 clean interruption checkpointed.", flush=True)
                return CLEAN_INTERRUPTION_EXIT_CODE

        if iso_points is None or not refresh_history:
            raise RuntimeError("Stage-2 finished without an accepted strict iso refresh")
        if phase is LifecyclePhase.EXPORT_PENDING:
            if resumed_export_pending is None:
                raise RuntimeContractError("export-pending resume lost its validated checkpoint")
            if not gate_history:
                raise RuntimeContractError("export-pending resume lacks its terminal gate")
            final_gate = gate_history[-1]
            if (
                not isinstance(final_gate, Mapping)
                or final_gate.get("phase") != "pre_export"
                or int(final_gate.get("step", -1)) != int(recipe["steps"])
                or final_gate.get("passed") is not True
            ):
                raise RuntimeContractError("export-pending resume terminal gate changed")
            pending = resumed_export_pending
        else:
            model.eval()
            final_gate = strict_field_gate(
                model, geometry["extent"], int(recipe["gate_grid"]), float(recipe["boundary_margin"]),
                geometry["shell"], device, int(recipe["grid_chunk"]),
            )
            final_gate.update({"step": int(recipe["steps"]), "phase": "pre_export"})
            gate_history.append(final_gate)
            if not final_gate["passed"]:
                raise RuntimeError(f"Stage-2 final strict gate failed: {final_gate}")
            phase = LifecyclePhase.EXPORT_PENDING
            pending = checkpoint(phase, clean=False, current_init_step=init_step, current_step=int(recipe["steps"]))
        if stop_requested():
            checkpoint(phase, clean=True, current_init_step=init_step, current_step=int(recipe["steps"]))
            return CLEAN_INTERRUPTION_EXIT_CODE

        staging = new_export_staging(contract.output_dir)
        surface_path = staging / SURFACE_NAME
        surface_audit = _export_surface(
            model=model, geometry=geometry, recipe=recipe, device=device, path=surface_path
        )
        final = _completion_checkpoint(pending, surface_audit)
        atomic_torch_save(final, staging / FINAL_CHECKPOINT_NAME)
        summary = {
            "method": output_identity["method"],
            "campaign_identity": output_identity["campaign_identity"],
            "artifact_identity": output_identity["artifact_identity"],
            "policy_identity": output_identity["policy_identity"],
            "implementation_kind": output_identity["implementation_kind"],
            "contract": lifecycle_contract,
            "steps": int(recipe["steps"]),
            "best_loss": float(best_loss),
            "final_gate": final_gate,
            "last_iso_refresh": refresh_history[-1],
            "surface_audit": surface_audit,
            "ground_truth_geometry_used": False,
            "novel_view_signal_supported": False,
        }
        status = {
            "schema": CHECKPOINT_SCHEMA,
            "phase": LifecyclePhase.COMPLETE.value,
            "state": LifecycleState.COMPLETE.value,
            "step": int(recipe["steps"]),
            "max_steps": int(recipe["steps"]),
            "resume_allowed": False,
            "contract": lifecycle_contract,
        }
        atomic_json_dump(summary, staging / SUMMARY_NAME)
        atomic_json_dump(status, staging / STATUS_NAME)
        if stop_requested():
            checkpoint(phase, clean=True, current_init_step=init_step, current_step=int(recipe["steps"]), surface_audit=surface_audit)
            return CLEAN_INTERRUPTION_EXIT_CODE
        promoted = promote_complete_package(
            staging,
            contract.complete_dir,
            validate=lambda directory: _validate_complete_package(
                directory,
                checkpoint_contract=provenance,
                lifecycle_contract=lifecycle_contract,
                recipe=recipe,
                geometry=geometry,
                device=device,
            ),
        )
        complete_lifecycle = {
            "schema": "rift_sugavanam_ertin_stage2_runtime_v1",
            "phase": LifecyclePhase.COMPLETE.value,
            "state": LifecycleState.COMPLETE.value,
            "step": int(recipe["steps"]),
            "max_steps": int(recipe["steps"]),
            "resume_allowed": False,
            "checkpoint_path": str(promoted / FINAL_CHECKPOINT_NAME),
            "exit_code": None,
            "contract": lifecycle_contract,
        }
        write_lifecycle(
            contract.lifecycle_path,
            complete_lifecycle,
            expected_contract=lifecycle_contract,
            max_steps=int(recipe["steps"]),
        )
        print(f"B7873200 Stage-2 complete: {promoted}", flush=True)
        return 0
    except SystemExit as exc:
        if exc.code == CLEAN_INTERRUPTION_EXIT_CODE:
            return CLEAN_INTERRUPTION_EXIT_CODE
        raise
    except BaseException as exc:
        record_terminal_failure(
            contract.output_dir,
            phase=phase,
            step=init_step if phase is LifecyclePhase.INITIALIZATION else int(len(history)),
            max_steps=int(recipe["steps"]),
            expected_contract=lifecycle_contract,
            error=exc,
        )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    return _run(
        parse_args(argv),
        contract=B7873200Stage2Contract(),
        recipe=default_stage2_recipe(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
