#!/usr/bin/env python3
"""Run a Sugavanam--Ertin engineering smoke for one RIFT dataset object.

Use --object a320 (or any registered object) for the shared two-stage recipe.
The former A320-only SDF workflow remains available explicitly through
--recipe legacy-a320-stabilized; it is not interchangeable with this recipe.
Without --object, historical explicit B787 archive/manifest commands still work.

The driver performs a header-only parent preflight, derives a sealed 16/16
child manifest, computes a train-only reporting scale, runs a bounded real
Stage-1 fit, packages that terminal state, and immediately feeds only the
validated smoke bundle into the existing strict Stage-2 lifecycle.  It is not
the frozen 3,200/1,000 baseline and never produces a production result.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import numpy as np
import torch

import train as rift_train
from rift.sugavanam_ertin_b7873200_real_smoke import (
    B787_CANONICAL_NPZ,
    B787_PARENT_MANIFEST,
    SMOKE_REPORT_NAME,
    SMOKE_RUN_NAME,
    SMOKE_STAGE1_BUNDLE_FIELD,
    SMOKE_STAGE1_BUNDLE_FILENAME,
    SMOKE_STAGE1_CHECKPOINT_NAME,
    SMOKE_STAGE2_ARTIFACT,
    SMOKE_STAGE2_CAMPAIGN,
    SMOKE_STAGE1_LABEL,
    assert_restricted_roles,
    build_child_manifest,
    compute_train_only_signal_normalization,
    analytic_sdf_shell_diagnostic,
    load_canonical_parent,
    load_collection_smoke_inputs,
    read_json_mapping,
    resolved,
    smoke_cloud_from_final,
    smoke_stage1_argv,
    smoke_stage1_checkpoint_dir,
    smoke_stage1_record,
    smoke_stage1_recipe,
    validate_child_manifest,
    validate_generic_stage1_final,
    write_child_manifest,
)
from rift.sugavanam_ertin_stage2_runtime_v1 import atomic_json_dump
import train_sugavanam_ertin_stage2 as stage2
from rift.rift_dataset import (
    DEFAULT_ROOT, collection_manifest, object_identity, object_spec,
    resolve_object_inputs, validate_checkpoint_object,
)


class RealSmokeContractError(ValueError):
    """Raised when the engineering-only smoke identity is not unambiguous."""


SMOKE_WALL_LIMIT_SECONDS = 3_600.0
SMOKE_MEMORY_LIMIT_BYTES = 32 * 1024**3


@dataclass(frozen=True)
class SmokeStage2Contract:
    """Minimal output contract accepted by the private Stage-2 smoke seam."""

    output_dir: str

    @property
    def latest_checkpoint(self) -> str:
        return str(Path(self.output_dir) / "checkpoint_latest.pth.tar")

    @property
    def complete_dir(self) -> str:
        return str(Path(self.output_dir) / "complete")

    @property
    def lifecycle_path(self) -> str:
        return str(Path(self.output_dir) / "lifecycle.json")


class Stage1EngineeringObserver:
    """Read-only fit evidence attached to the existing generic trainer."""

    def __init__(self, **kwargs: object) -> None:
        self.train_loader = kwargs["train_loader"]
        self.validation_loader = kwargs["validation_loader"]
        self.criterion = kwargs["criterion"]
        self.device = kwargs["device"]
        self.num_freq_selected = int(kwargs["num_freq_selected"])
        self.phase_sign = float(kwargs["phase_sign"])
        self.forward_operator_name = str(kwargs["forward_operator_name"])
        self.compute_dtype = kwargs["compute_dtype"]
        self.data_format = str(kwargs["data_format"])
        self.op_kwargs = dict(kwargs["op_kwargs"])
        self.occlusion = kwargs["occlusion"]
        self.arr_dist = float(kwargs["arr_dist"])
        self.spacing = float(kwargs["spacing"])
        self.num_rx = int(kwargs["num_rx"])
        self.num_tx = int(kwargs["num_tx"])
        self.fp_grid = kwargs["fp_grid"]
        self.w_1 = float(kwargs["w_1"])
        self.w_2 = float(kwargs["w_2"])
        self.initial_readout: dict[str, object] | None = None
        self.final_readout: dict[str, object] | None = None
        self._gain: object | None = None
        self.start_epoch: int | None = None
        self.logical_updates_at_start: int | None = None
        self._before_parameters: dict[str, object] | None = None
        self._before_gradients: dict[str, object] | None = None
        self.steps: list[dict[str, object]] = []

    @staticmethod
    def _parameters(model: object, gain: object | None) -> dict[str, object]:
        parameters: dict[str, object] = {}
        for name, parameter in model.named_parameters():
            parameters[f"model.{name}"] = parameter
        if gain is not None:
            for name, parameter in gain.named_parameters():
                parameters[f"gain.{name}"] = parameter
        return parameters

    def _readout(self, model: object, gain: object | None) -> dict[str, object]:
        common = {
            "criterion": self.criterion,
            "device": self.device,
            "num_freq_selected": self.num_freq_selected,
            "fp_grid": self.fp_grid,
            "w_1": self.w_1,
            "w_2": self.w_2,
            "arr_dist": self.arr_dist,
            "spacing": self.spacing,
            "num_rx": self.num_rx,
            "num_tx": self.num_tx,
            "loss_mode": "complex",
            "gain": gain,
            "phase_sign": self.phase_sign,
            "forward_operator_name": self.forward_operator_name,
            "compute_dtype": self.compute_dtype,
            "data_format": self.data_format,
            "op_kwargs": self.op_kwargs,
            "occlusion": self.occlusion,
            "mag_weight": 0.0,
            "return_metrics": True,
        }
        train_metrics = rift_train.evaluate(model, self.train_loader, **common)
        validation_metrics = rift_train.evaluate(model, self.validation_loader, **common)
        result = {"train": dict(train_metrics), "validation": dict(validation_metrics)}
        for role, metrics in result.items():
            if set(metrics) != {
                "loss", "residual_power", "zero_reference_power", "relative_mse", "relative_l2"
            }:
                raise RealSmokeContractError(f"Stage-1 {role} readout schema changed")
            for key, value in metrics.items():
                if not math.isfinite(float(value)):
                    raise RealSmokeContractError(f"Stage-1 {role} readout is non-finite: {key}")
            if float(metrics["zero_reference_power"]) <= 0 or float(metrics["residual_power"]) < 0:
                raise RealSmokeContractError(f"Stage-1 {role} readout has invalid coherent power")
            expected_mse = float(metrics["residual_power"]) / float(metrics["zero_reference_power"])
            if not math.isclose(float(metrics["relative_mse"]), expected_mse, rel_tol=1.0e-12, abs_tol=1.0e-15):
                raise RealSmokeContractError(f"Stage-1 {role} relative MSE denominator changed")
            if not math.isclose(float(metrics["relative_l2"]), math.sqrt(expected_mse), rel_tol=1.0e-12, abs_tol=1.0e-15):
                raise RealSmokeContractError(f"Stage-1 {role} relative L2 denominator changed")
        return result

    def on_training_start(
        self, *, model: object, optimizer: object, gain: object | None,
        start_epoch: int, logical_optimizer_updates: int,
    ) -> None:
        del optimizer
        self._gain = gain
        self.start_epoch = int(start_epoch)
        self.logical_updates_at_start = int(logical_optimizer_updates)
        self.initial_readout = self._readout(model, gain)

    def before_optimizer_step(self, *, model: object, optimizer: object) -> None:
        del optimizer
        parameters = self._parameters(model, self._gain)
        if not parameters:
            raise RealSmokeContractError("Stage-1 observer found no trainable parameters")
        self._before_parameters = {
            name: parameter.detach().cpu().clone() for name, parameter in parameters.items()
        }
        self._before_gradients = {
            name: (None if parameter.grad is None else parameter.grad.detach().cpu().clone())
            for name, parameter in parameters.items()
        }

    def on_optimizer_step(
        self, *, model: object, optimizer: object, epoch: int,
        logical_optimizer_updates: int, grad_norm: float,
    ) -> None:
        del optimizer
        if self._before_parameters is None or self._before_gradients is None:
            raise RealSmokeContractError("Stage-1 observer missed the pre-step snapshot")
        parameters = self._parameters(model, self._gain)
        gradient_power = 0.0
        finite_gradients = True
        nonzero_gradient_tensors = 0
        for name, gradient in self._before_gradients.items():
            if gradient is None:
                continue
            values = gradient.numpy()
            finite_gradients = finite_gradients and bool(np.isfinite(values).all())
            gradient_power += float(np.square(values.astype(np.float64, copy=False)).sum())
            if bool(np.any(values != 0)):
                nonzero_gradient_tensors += 1
        delta_power = 0.0
        max_delta = 0.0
        finite_delta = True
        for name, before in self._before_parameters.items():
            after = parameters[name].detach().cpu()
            delta = (after - before).numpy()
            finite_delta = finite_delta and bool(np.isfinite(delta).all())
            delta_power += float(np.square(delta.astype(np.float64, copy=False)).sum())
            if delta.size:
                max_delta = max(max_delta, float(np.max(np.abs(delta))))
        if not finite_gradients or not finite_delta:
            raise RealSmokeContractError("Stage-1 observer saw a non-finite gradient or parameter change")
        self.steps.append({
            "epoch": int(epoch),
            "logical_optimizer_updates": int(logical_optimizer_updates),
            "reported_grad_norm": float(grad_norm),
            "observed_gradient_l2": math.sqrt(gradient_power),
            "finite_gradients": finite_gradients,
            "nonzero_gradient_tensors": int(nonzero_gradient_tensors),
            "parameter_delta_l2": math.sqrt(delta_power),
            "max_abs_parameter_delta": max_delta,
            "finite_parameter_delta": finite_delta,
            "nonzero_parameter_update": bool(delta_power > 0.0),
        })
        self._before_parameters = None
        self._before_gradients = None

    def on_epoch_end(
        self, *, model: object, optimizer: object, epoch: int,
        num_epochs: int, val_metrics: Mapping[str, object] | None,
    ) -> None:
        del optimizer, val_metrics
        if int(epoch) == int(num_epochs):
            self.final_readout = self._readout(model, self._gain)

    def result(self) -> dict[str, object]:
        if self.initial_readout is None or self.final_readout is None:
            raise RealSmokeContractError("Stage-1 observer lacks initial/final readouts")
        if self.start_epoch is None or self.logical_updates_at_start is None:
            raise RealSmokeContractError("Stage-1 observer lacks its starting update count")
        nonzero_updates = sum(1 for row in self.steps if row["nonzero_parameter_update"])
        if not self.steps or nonzero_updates <= 0:
            raise RealSmokeContractError("Stage-1 observed no finite nonzero parameter updates")
        if any(row["observed_gradient_l2"] <= 0 or row["parameter_delta_l2"] <= 0 for row in self.steps):
            raise RealSmokeContractError("Stage-1 observed a zero gradient or parameter update")
        return {
            "schema": "rift_sugavanam_ertin_stage1_engineering_observation_v1",
            "start_epoch": int(self.start_epoch),
            "logical_optimizer_updates_at_start": int(self.logical_updates_at_start),
            "observed_optimizer_steps": int(len(self.steps)),
            "observed_nonzero_parameter_updates": int(nonzero_updates),
            "logical_optimizer_updates_final": int(self.steps[-1]["logical_optimizer_updates"]),
            "steps": copy.deepcopy(self.steps),
            "initial": copy.deepcopy(self.initial_readout),
            "final": copy.deepcopy(self.final_readout),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=("two-stage", "legacy-a320-stabilized"), default="two-stage",
                        help="legacy mode has a separate fixed-checkpoint CLI; use --recipe legacy-a320-stabilized --help")
    parser.add_argument("--object", default=None, help="RIFT dataset object or alias")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--npz-path", default=None)
    parser.add_argument("--parent-role-manifest", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.recipe != "two-stage":
        raise RealSmokeContractError("Use dispatch_main for the explicit legacy stabilized-SDF recipe")
    if args.object is not None:
        args.object = object_spec(args.object)["object_id"]
        npz, manifest = resolve_object_inputs(
            object_name=args.object, dataset_root=args.dataset_root,
            npz_path=args.npz_path, role_manifest_path=args.parent_role_manifest)
        args.npz_path, args.parent_role_manifest = str(npz), str(manifest)
    else:
        args.npz_path = args.npz_path or B787_CANONICAL_NPZ
        args.parent_role_manifest = args.parent_role_manifest or B787_PARENT_MANIFEST
    return args


def dispatch_main(argv: Sequence[str] | None = None) -> int:
    """One CLI, with an explicit compatibility path for the distinct old recipe."""
    argv = list(sys.argv[1:] if argv is None else argv)
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--recipe", choices=("two-stage", "legacy-a320-stabilized"), default="two-stage")
    selector.add_argument("--object")
    selected, remainder = selector.parse_known_args(argv)
    if selected.recipe == "legacy-a320-stabilized":
        if selected.object is not None and object_spec(selected.object)["object_id"] != "airliner_a320":
            raise RealSmokeContractError("Legacy stabilized-SDF mode is bound to its historical A320 checkpoint")
        from rift import sugavanam_ertin_stabilized_smoke_runtime as legacy
        legacy.main(remainder)
        return 0
    return main(argv)


def _absolute_unresolved(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _reject_symlink(path: Path, label: str) -> None:
    if path.is_symlink():
        raise RealSmokeContractError(f"{label} must not be a symbolic link")


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows source checks
        return None
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _resource_snapshot() -> dict[str, object]:
    snapshot: dict[str, object] = {"process_peak_rss_bytes": _peak_rss_bytes()}
    if torch.cuda.is_available():
        snapshot.update(
            {
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
        )
    else:
        snapshot.update({"cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None})
    return snapshot


def _reset_phase_peaks(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _validate_resource_snapshot(snapshot: Mapping[str, object], *, label: str) -> None:
    wall = float(snapshot.get("wall_seconds", float("nan")))
    rss = snapshot.get("process_peak_rss_bytes")
    if not math.isfinite(wall) or wall < 0 or wall > SMOKE_WALL_LIMIT_SECONDS:
        raise RealSmokeContractError(f"{label} exceeded the one-hour wall-time envelope")
    if rss is None or not math.isfinite(float(rss)) or float(rss) <= 0:
        raise RealSmokeContractError(f"{label} lacks a finite process peak-RSS measurement")
    if float(rss) > SMOKE_MEMORY_LIMIT_BYTES:
        raise RealSmokeContractError(f"{label} exceeded the declared 32 GiB memory envelope")
    for key in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
        value = snapshot.get(key)
        if value is not None and (not math.isfinite(float(value)) or float(value) < 0):
            raise RealSmokeContractError(f"{label} has an invalid CUDA resource measurement")


def _write_resource_accounting(
    path: Path,
    *,
    run_start: float,
    status: str,
    phase: str,
    stage1: Mapping[str, object] | None = None,
    stage2: Mapping[str, object] | None = None,
    error: BaseException | None = None,
) -> None:
    """Persist timing/peak evidence before success or after an exception."""

    payload: dict[str, object] = {
        "schema": "rift_sugavanam_ertin_b7873200_resource_accounting_v1",
        "status": status,
        "phase": phase,
        "declared": {
            "wall_seconds": SMOKE_WALL_LIMIT_SECONDS,
            "memory_bytes": SMOKE_MEMORY_LIMIT_BYTES,
            "cuda_devices": 1,
            "cpus": 6,
            "temporary_storage_gib": 12,
        },
        "total_wall_seconds": float(time.monotonic() - run_start),
        "current_peak": _resource_snapshot(),
        "stage1": None if stage1 is None else dict(stage1),
        "stage2": None if stage2 is None else dict(stage2),
    }
    if error is not None:
        payload["error"] = {"type": type(error).__name__, "message": str(error)}
    atomic_json_dump(payload, path)


_RESOURCE_CONTEXT: dict[str, object] | None = None


def _resource_failure_guard(function):
    """Persist the latest resource state for every post-preflight failure."""

    def guarded(argv: Sequence[str] | None = None) -> int:
        global _RESOURCE_CONTEXT
        try:
            return function(argv)
        except BaseException as exc:
            context = _RESOURCE_CONTEXT
            if context is not None:
                stage1 = context.get("stage1")
                stage2 = context.get("stage2")
                try:
                    _write_resource_accounting(
                        context["path"],
                        run_start=float(context["run_start"]),
                        status="failed",
                        phase=str(context["phase"]),
                        stage1=stage1 if isinstance(stage1, Mapping) else None,
                        stage2=stage2 if isinstance(stage2, Mapping) else None,
                        error=exc,
                    )
                except BaseException:
                    # Never replace the original training, validation, or export error.
                    pass
            raise
        finally:
            _RESOURCE_CONTEXT = None

    return guarded


def _smoke_stage2_recipe() -> dict[str, object]:
    """Return a bounded, explicitly non-production Stage-2 recipe."""

    return {
        "schema": "rift_sugavanam_ertin_b7873200_stage2_engineering_smoke_recipe_v3",
        "recipe_revision": 3,
        "steps": 20,
        "init_steps": 1000,
        "init_lr": 5.0e-4,
        "init_batch": 2048,
        "init_log_every": 100,
        "batch_on": 128,
        "batch_off": 128,
        "batch_iso": 64,
        "batch_signed": 128,
        "batch_boundary": 64,
        "batch_inner": 32,
        "n_iso": 64,
        "iso_start": 2,
        "iso_refresh": 5,
        "scatter_threshold": 0.15,
        "max_scatter_points": 20_000,
        "normal_radius_policy": "three_stage1_voxel_pitches",
        "signed_offset_pitches": 1.0,
        "n_fourier": 5,
        "fourier_scale": 2.0,
        "hidden_dim": 128,
        "sdf_skip_layer_index": 4,
        "sdf_min_layers": 5,
        "n_layers": 5,
        "lr": 1.0e-4,
        "alpha_off": 100.0,
        "lambda_on": 1.0,
        "lambda_normal": 1.0,
        "lambda_signed": 1.0,
        "lambda_off": 1.0,
        "lambda_eik": 1.0,
        "lambda_iso": 1.0,
        "lambda_iso_normal": 1.0,
        "lambda_boundary": 1.0,
        "lambda_inner": 1.0,
        "off_warmup": 2,
        "off_ramp": 4,
        "radius_quantile": 0.5,
        "radius_cap_fraction": 0.65,
        "boundary_margin": 1.0e-4,
        "boundary_shell_resolution": 16,
        "inner_anchor_count": 256,
        "gate_every": 5,
        "gate_grid": 24,
        "grid_chunk": 8192,
        "projection_oversample": 2.0,
        "mesh_grid": 32,
        "mesh_points": 1000,
        "save_every": 5,
        "seed": 42,
    }


def _stage2_provenance(
    *,
    stage1_record: Mapping[str, object],
    recipe: Mapping[str, object],
    output_dir: Path,
    npz_path: str = B787_CANONICAL_NPZ,
    dataset_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "contract": {
            "campaign_identity": SMOKE_STAGE2_CAMPAIGN,
            "artifact_identity": SMOKE_STAGE2_ARTIFACT,
            "method": ("Sugavanam--Ertin RIFT dataset engineering smoke" if dataset_identity
                       else "Sugavanam--Ertin B787 real-data engineering smoke"),
            "policy_identity": stage2.POLICY_IDENTITY,
            "implementation_kind": "stabilized derivative; altered engineering budget; not production",
            "canonical_npz_path": npz_path,
            **({"dataset_identity": dict(dataset_identity)} if dataset_identity else {}),
            "stage1_final_bundle_filename": SMOKE_STAGE1_BUNDLE_FILENAME,
            "output_dir": str(output_dir),
            "ground_truth_geometry_used": False,
            "novel_view_signal_supported": False,
        },
        "stage1_record": dict(stage1_record),
        "stage2_recipe": dict(recipe),
        "ground_truth_geometry_used": False,
        "novel_view_signal_supported": False,
    }


def _validate_existing_bundle(
    bundle_path: Path,
    *,
    manifest_contract: Mapping[str, object],
    normalization: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    if not bundle_path.is_file():
        raise RealSmokeContractError(f"missing Stage-1 smoke bundle: {bundle_path}")
    bundle = rift_train.load_tensor_checkpoint(str(bundle_path), map_location="cpu")
    record = bundle.get(SMOKE_STAGE1_BUNDLE_FIELD)
    state = bundle.get("generic_final_state")
    if not isinstance(record, Mapping) or not isinstance(state, Mapping):
        raise RealSmokeContractError("Stage-1 smoke bundle is incomplete")
    if record.get("recipe_id") != SMOKE_STAGE1_LABEL or record.get("role") != "checkpoint_final":
        raise RealSmokeContractError("Stage-1 smoke bundle identity changed")
    if record.get("sealed_protocol_identity") != dict(manifest_contract):
        raise RealSmokeContractError("Stage-1 smoke bundle sealed child contract changed")
    recipe = record.get("stage1_recipe")
    saved_norm = recipe.get("normalization") if isinstance(recipe, Mapping) else None
    expected_norm = {
        "scope": "selected_parent_train_only",
        "source_count": 16,
        "raw_complex_rms": float(normalization["raw_complex_rms"]),
        "zero_reference_train_mse": float(normalization["zero_reference_train_mse"]),
        "used_for": "reported same-domain normalization only; no validation/test leakage",
    }
    if not isinstance(saved_norm, Mapping) or dict(saved_norm) != expected_norm:
        raise RealSmokeContractError("Stage-1 smoke bundle train-only normalization changed")
    observation = record.get("engineering_observation")
    validate_generic_stage1_final(
        state,
        manifest_contract=manifest_contract,
        observation=observation if isinstance(observation, Mapping) else None,
    )
    return state, record


def _write_stage1_bundle(
    bundle_path: Path,
    *,
    state: Mapping[str, object],
    record: Mapping[str, object],
) -> None:
    if bundle_path.exists():
        raise RealSmokeContractError(f"refusing to overwrite terminal smoke bundle: {bundle_path}")
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    rift_train._atomic_torch_save(
        {"generic_final_state": dict(state), SMOKE_STAGE1_BUNDLE_FIELD: dict(record)},
        str(bundle_path),
    )


def _validate_cli(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path, Path]:
    args.dataset_identity = None
    if collection_manifest(args.parent_role_manifest):
        _parent, contract = load_canonical_parent(args.npz_path, args.parent_role_manifest)
        args.dataset_identity = contract["dataset_identity"]
        if getattr(args, "object", None) is not None and args.dataset_identity != object_identity(args.object):
            raise RealSmokeContractError("Selected object does not match the collection files")
    else:
        if getattr(args, "object", None) is not None:
            raise RealSmokeContractError("Named RIFT dataset objects require their collection manifest")
        if resolved(args.npz_path) != resolved(B787_CANONICAL_NPZ):
            raise RealSmokeContractError(f"real smoke accepts only {B787_CANONICAL_NPZ} or a RIFT dataset object")
        if resolved(args.parent_role_manifest) != resolved(B787_PARENT_MANIFEST):
            raise RealSmokeContractError(f"real smoke accepts only {B787_PARENT_MANIFEST} or a RIFT dataset manifest")
    try:
        device = torch.device(args.device)
    except (TypeError, RuntimeError) as exc:
        raise RealSmokeContractError(f"invalid device: {args.device!r}") from exc
    if device.type != "cuda" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RealSmokeContractError("real B787 Sugavanam--Ertin smoke requires one allocated CUDA device")
    run_name = (f"rift_dataset_{args.dataset_identity['object_id']}_se_smoke_v1"
                if args.dataset_identity else SMOKE_RUN_NAME)
    root = _absolute_unresolved(args.checkpoint_root) / run_name
    stage1_root = smoke_stage1_checkpoint_dir(root)
    stage1_final = stage1_root / "checkpoint_final.pth.tar"
    stage1_latest = stage1_root / "checkpoint_latest.pth.tar"
    bundle_path = root / SMOKE_STAGE1_BUNDLE_FILENAME
    stage2_root = root / "stage2"
    stage2_latest = stage2_root / "checkpoint_latest.pth.tar"
    report_path = root / SMOKE_REPORT_NAME
    manifest_path = root / "derived_roles.json"
    for path, label in (
        (root, "smoke root"),
        (stage1_root, "Stage-1 smoke root"),
        (stage2_root, "Stage-2 smoke root"),
        (manifest_path, "derived smoke manifest"),
        (bundle_path, "Stage-1 smoke bundle"),
        (report_path, "smoke report"),
    ):
        _reject_symlink(path, label)
    resume = None if args.resume is None else _absolute_unresolved(args.resume)
    if resume is None:
        if root.exists() and any(root.iterdir()):
            raise RealSmokeContractError("fresh real smoke requires a new empty identity root")
    elif resume not in (stage1_latest, stage2_latest) or not resume.is_file():
        raise RealSmokeContractError("resume must be this smoke's Stage-1 or Stage-2 latest checkpoint")
    if stage2_root.joinpath("complete").exists() and resume == stage2_latest:
        raise RealSmokeContractError("completed Stage-2 smoke is terminal and cannot resume")
    return root, stage1_final, stage1_latest, bundle_path, stage2_root, report_path


@_resource_failure_guard
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_resource_start = time.monotonic()
    root, stage1_final, stage1_latest, bundle_path, stage2_root, report_path = _validate_cli(args)
    device = torch.device(args.device)
    resume_path = None if args.resume is None else _absolute_unresolved(args.resume)
    root.mkdir(parents=True, exist_ok=True)
    resource_accounting_path = root / "resource_accounting.json"
    _write_resource_accounting(
        resource_accounting_path,
        run_start=run_resource_start,
        status="running",
        phase="preparation",
    )
    global _RESOURCE_CONTEXT
    _RESOURCE_CONTEXT = {
        "path": resource_accounting_path,
        "run_start": run_resource_start,
        "phase": "preparation",
        "stage1": None,
        "stage2": None,
    }

    parent, _parent_contract = load_canonical_parent(args.npz_path, args.parent_role_manifest)
    child = build_child_manifest(parent)
    validate_child_manifest(parent, child)
    manifest_path = write_child_manifest(root / "derived_roles.json", child)

    if args.dataset_identity:
        arrays, manifest_contract = load_collection_smoke_inputs(args.npz_path, manifest_path)
        # Check object binding before any train normalization or response scan.
        for candidate in (resume_path, bundle_path if bundle_path.exists() else None):
            if candidate is not None:
                saved = rift_train.load_tensor_checkpoint(str(candidate), map_location="cpu")
                validate_checkpoint_object(saved, _parent_contract)
    else:
        from train import _load_sealed_npz_protocol_contract
        arrays, manifest_contract = _load_sealed_npz_protocol_contract(
            args.npz_path, str(manifest_path), num_train=16, num_val=16, num_test=1_000)
    assert_restricted_roles(
        arrays,
        {
            "train": child["split"]["train_indices"],
            "validation": child["split"]["validation_indices"],
            "test": child["split"]["test_indices"],
            "unused": child["split"]["unused_indices"],
        },
    )
    normalization = compute_train_only_signal_normalization(
        arrays, child["split"]["train_indices"]
    )
    stage1_semantic_recipe = smoke_stage1_recipe(
        normalized_train_rms=float(normalization["raw_complex_rms"]),
        zero_reference_train_mse=float(normalization["zero_reference_train_mse"]),
    )

    # Stage-1 accounting intentionally includes archive/manifest preparation,
    # sealing checks, and train-only normalization before the fit begins.
    stage1_resource_start = run_resource_start
    _reset_phase_peaks(device)
    if resume_path == stage1_latest:
        stage1_resume = str(stage1_latest)
    else:
        stage1_resume = None
    observer_box: dict[str, Stage1EngineeringObserver] = {}

    def engineering_observer_factory(**kwargs: object) -> Stage1EngineeringObserver:
        observer = Stage1EngineeringObserver(**kwargs)
        observer_box["observer"] = observer
        return observer

    _RESOURCE_CONTEXT["phase"] = "stage1"
    if stage1_final.exists():
        if not bundle_path.exists():
            raise RealSmokeContractError("Stage-1 final exists without its sealed smoke bundle")
        state, record = _validate_existing_bundle(
            bundle_path,
            manifest_contract=manifest_contract,
            normalization=normalization,
        )
        stage1_observation = record.get("engineering_observation")
        if not isinstance(stage1_observation, Mapping):
            raise RealSmokeContractError("existing Stage-1 bundle lacks observed fit evidence")
        stage1_audit = validate_generic_stage1_final(
            state,
            manifest_contract=manifest_contract,
            observation=stage1_observation,
        )
    else:
        rift_train.main(
            smoke_stage1_argv(
                npz_path=args.npz_path,
                manifest_path=manifest_path,
                checkpoint_root=root,
                resume=stage1_resume,
            ),
            engineering_observer_factory=engineering_observer_factory,
        )
        if not stage1_final.is_file():
            raise RealSmokeContractError("Stage-1 smoke returned without a terminal checkpoint")
        state = rift_train.load_tensor_checkpoint(str(stage1_final), map_location="cpu")
        observer = observer_box.get("observer")
        if observer is None:
            raise RealSmokeContractError("Stage-1 fit did not construct its engineering observer")
        stage1_observation = observer.result()
        stage1_audit = validate_generic_stage1_final(
            state,
            manifest_contract=manifest_contract,
            observation=stage1_observation,
        )
        record = smoke_stage1_record(
            state=state,
            manifest_contract=manifest_contract,
            normalization=normalization,
            final_path=str(bundle_path),
            observation=stage1_observation,
        )
        # Ensure the stored recipe carries exactly the same normalization used by
        # the response preflight; this is reporting-only and never uses validation.
        if record["stage1_recipe"]["normalization"] != {
            "scope": "selected_parent_train_only",
            "source_count": 16,
            "raw_complex_rms": float(normalization["raw_complex_rms"]),
            "zero_reference_train_mse": float(normalization["zero_reference_train_mse"]),
            "used_for": "reported same-domain normalization only; no validation/test leakage",
        }:
            raise RealSmokeContractError("Stage-1 smoke normalization record changed")
        _write_stage1_bundle(bundle_path, state=state, record=record)
    stage1_resource = {
        "wall_seconds": float(time.monotonic() - stage1_resource_start),
        **_resource_snapshot(),
    }
    _RESOURCE_CONTEXT["stage1"] = stage1_resource
    _validate_resource_snapshot(stage1_resource, label="Stage-1 preparation and fit")
    _RESOURCE_CONTEXT["phase"] = "stage2"
    _write_resource_accounting(
        resource_accounting_path,
        run_start=run_resource_start,
        status="running",
        phase="stage2",
        stage1=stage1_resource,
    )

    source = smoke_cloud_from_final(
        state,
        stage1_record=record,
        checkpoint_path=str(bundle_path),
    )
    recipe = _smoke_stage2_recipe()
    geometry_feasibility = analytic_sdf_shell_diagnostic(
        source["points"],
        extent=float(source["extent"]),
        granularity=int(source["granularity"]),
        radius_quantile=float(recipe["radius_quantile"]),
        radius_cap_fraction=float(recipe["radius_cap_fraction"]),
    )
    if geometry_feasibility["feasible"] is not True:
        raise RealSmokeContractError(
            "Stage-2 analytic SDF shell preflight is infeasible: "
            f"center={geometry_feasibility['center']} "
            f"radius={geometry_feasibility['radius']:.9g} "
            f"boundary_clearance={geometry_feasibility['boundary_clearance']:.9g} "
            f"protected_shell_depth={geometry_feasibility['protected_shell_depth']:.9g} "
            f"shell_clearance={geometry_feasibility['shell_clearance']:.9g}; "
            "increase Stage-1 granularity or use a fresh tuned identity"
        )
    print(
        "SE_B7873200_REAL_SMOKE_ANALYTIC_SDF_SHELL="
        + json.dumps(geometry_feasibility, sort_keys=True),
        flush=True,
    )
    provenance = _stage2_provenance(
        stage1_record=record,
        recipe=recipe,
        output_dir=stage2_root,
        npz_path=args.npz_path,
        dataset_identity=args.dataset_identity,
    )
    stage2_contract = SmokeStage2Contract(str(stage2_root))
    stage2_args = argparse.Namespace(
        resume=(
            str(resume_path)
            if resume_path is not None
            and resume_path == _absolute_unresolved(stage2_contract.latest_checkpoint)
            else None
        ),
        device=str(args.device),
    )
    stage2_resource_start = time.monotonic()
    _reset_phase_peaks(device)
    result = stage2._run(
        stage2_args,
        contract=stage2_contract,
        recipe=recipe,
        source=source,
        provenance=provenance,
        engineering_override=True,
    )
    stage2_resource = {
        "wall_seconds": float(time.monotonic() - stage2_resource_start),
        **_resource_snapshot(),
    }
    _RESOURCE_CONTEXT["stage2"] = stage2_resource
    _validate_resource_snapshot(stage2_resource, label="Stage-2 geometry and export")
    if result != 0:
        _write_resource_accounting(
            resource_accounting_path,
            run_start=run_resource_start,
            status="interrupted" if int(result) == stage2.CLEAN_INTERRUPTION_EXIT_CODE else "failed",
            phase="stage2",
            stage1=stage1_resource,
            stage2=stage2_resource,
        )
        return int(result)
    _RESOURCE_CONTEXT["phase"] = "export"
    if not Path(stage2_contract.complete_dir).is_dir():
        raise RealSmokeContractError("Stage-2 smoke returned without a complete package")
    stage2_summary = read_json_mapping(
        Path(stage2_contract.complete_dir) / stage2.SUMMARY_NAME,
        label="Stage-2 smoke summary",
    )
    total_resource = {
        "wall_seconds": float(time.monotonic() - run_resource_start),
        **_resource_snapshot(),
    }
    _validate_resource_snapshot(total_resource, label="combined smoke")
    _write_resource_accounting(
        resource_accounting_path,
        run_start=run_resource_start,
        status="complete",
        phase="complete",
        stage1=stage1_resource,
        stage2=stage2_resource,
    )
    report = {
        "schema": "rift_sugavanam_ertin_b7873200_real_smoke_report_v3",
        "run_name": root.name,
        **({"dataset_identity": args.dataset_identity} if args.dataset_identity else {}),
        "scope": "bounded_engineering_smoke_not_production_not_comparison_not_convergence_evidence",
        "input_archive": resolved(args.npz_path),
        "parent_manifest": resolved(args.parent_role_manifest),
        "derived_manifest": str(manifest_path),
        "roles": {
            "train_count": 16,
            "validation_count": 16,
            "reserved_test_count": 1_000,
            "unused_count": 8_968,
            "test_and_unused_response_materialized": False,
        },
        "normalization": normalization,
        "stage1": {
            "audit": stage1_audit,
            "fit_evidence": stage1_audit["observed_fit"],
            "recipe": stage1_semantic_recipe,
            "bundle": str(bundle_path),
            "resource": stage1_resource,
        },
        "stage1_to_stage2_handoff": {
            "source": str(bundle_path),
            "source_role": "checkpoint_final",
            "validated_before_geometry": True,
            "ground_truth_geometry_used": False,
            "analytic_sdf_shell": geometry_feasibility,
        },
        "stage2": {
            "campaign_identity": SMOKE_STAGE2_CAMPAIGN,
            "artifact_identity": SMOKE_STAGE2_ARTIFACT,
            "recipe": recipe,
            "summary": stage2_summary,
            "resource": stage2_resource,
        },
        "resource_accounting": {
            "path": str(resource_accounting_path),
            "declared_wall_seconds": SMOKE_WALL_LIMIT_SECONDS,
            "declared_memory_bytes": SMOKE_MEMORY_LIMIT_BYTES,
            "combined": total_resource,
        },
        "cross_attempt_resume_validated": False,
        "production_clearance": False,
        "historical_baseline_unchanged": True,
    }
    atomic_json_dump(report, report_path)
    print(f"SE_B7873200_REAL_SMOKE_REPORT: {report_path}", flush=True)
    print("SE_B7873200_REAL_SMOKE_LIFECYCLE_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(dispatch_main())
