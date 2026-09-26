#!/usr/bin/env python3
"""Prepare then fit the bounded real-B787 GeRaF 16/4 engineering cell.

This is not a replacement for the sealed 3,200/1,000 GeRaF runner.  It first
materializes and validates exactly twenty native ``|MF|`` leaves, then runs
two shuffled 16-view cycles (32 optimizer updates) on the existing GeRaF
model, renderer, objective, optimizer, sampler, and dynamic-mask bank.  The
scientific cosine horizon remains 50,000 updates; 32 is only this driver's
engineering stop budget.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

try:  # Windows has no ``resource`` module; PACE Linux does.
    import resource
except ImportError:  # pragma: no cover - exercised only on non-POSIX hosts.
    resource = None  # type: ignore[assignment]

import train_geraf as trainer
from rift.config import cc
from rift.forward_operator import get_kvector
from rift.geraf_b78716_smallfit import (
    CACHE_MANIFEST_FILENAME,
    CACHE_PROTOCOL_FILENAME,
    CACHE_RECIPE_FILENAME,
    CACHE_SCHEMA,
    CACHE_STATS_FILENAME,
    FIT_ID,
    MAX_UPDATES,
    MILESTONE_UPDATES,
    NUM_TRAIN,
    NUM_VALIDATION,
    PREPARATION_ID,
    SCHEDULER_HORIZON_UPDATES,
    TARGET_GRID_SHAPE,
    PreparedB78716SmallfitCache,
    bounded_worklists,
    load_complete_subset_cache,
    prepare_complete_subset_cache,
    training_args,
)
from rift.geraf_b7873200_acquisition import (
    acquisition_records_equal,
    validate_b7873200_operator_frequency_grid,
)
from rift.geraf_b7873200_protocol import (
    B787_3200_CANONICAL_MANIFEST_PATH,
    B787_3200_CANONICAL_NPZ_PATH,
)
from rift.power_baseline_dataset import atomic_write_json, frequency_grid_hz


CHECKPOINT_FORMAT = "rift_geraf_b7873200_engineering_subset16x4_fit_checkpoint_v1"
REPORT_SCHEMA = "rift_geraf_b7873200_engineering_subset16x4_fit_report_v1"
RESOURCE_ENVELOPE_SCHEMA = "rift_geraf_b7873200_engineering_subset16x4_resource_envelope_v1"
RUN_NAME = FIT_ID
SCOPE = "bounded_engineering_smoke_not_production_not_comparison_not_convergence_evidence"
_CHECKPOINT_INITIAL = "checkpoint_initial.pth.tar"
_CHECKPOINT_LATEST = "checkpoint_latest.pth.tar"
_CHECKPOINT_BEST = "checkpoint_best.pth.tar"
_CHECKPOINT_FINAL = "checkpoint_final.pth.tar"
_REPORT = "smallfit_report.json"
_POSTFLIGHT = "smallfit_postflight.json"

# A clean fitting interruption is intentionally distinguishable from a
# terminal failure. A stop before the fit has a complete cache but no
# checkpoint, so it must not masquerade as a resumable interruption.
CLEAN_STOP_EXIT_CODE = 143
PRE_FIT_STOP_EXIT_CODE = 75

_STOP_REQUESTED = False
_STOP_SIGNAL: Optional[int] = None
_TIMED_WRAPPER_READY_ENV = "GERAF_TIMED_WRAPPER_READY_FILE"
_TIMED_WRAPPER_READY_MARKER = "GERAF_TIMED_WRAPPER_CHILD_READY"


@dataclass(frozen=True)
class _CleanStop:
    checkpoint: Path
    signal: int | None


@dataclass(frozen=True)
class _PreFitStop:
    signal: int | None


def _request_stop(signum: int, _frame: object) -> None:
    global _STOP_REQUESTED, _STOP_SIGNAL
    _STOP_REQUESTED = True
    _STOP_SIGNAL = int(signum)
    print(
        f"Received signal {signum}; will finish the current bounded GeRaF operation and "
        "write a clean resumable checkpoint before returning.",
        flush=True,
    )


def _publish_timed_wrapper_ready() -> None:
    """Publish readiness only for the v2 timing wrapper's opt-in signal path."""

    raw_path = os.environ.get(_TIMED_WRAPPER_READY_ENV)
    if raw_path is None:
        return
    ready_path = Path(raw_path)
    if not ready_path.parent.is_dir():
        raise RuntimeError(f"timed-wrapper ready-marker parent is missing: {ready_path.parent}")
    try:
        with ready_path.open("x", encoding="utf-8") as handle:
            handle.write(_TIMED_WRAPPER_READY_MARKER + "\n")
    except FileExistsError as exc:
        raise RuntimeError(f"timed-wrapper ready marker already exists: {ready_path}") from exc
    print(f"GERAF_B78716_SMALLFIT_TIMED_WRAPPER_READY path={ready_path}", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, help="separate 16/4 prepared-target root")
    parser.add_argument("--checkpoint-dir", required=True, help="separate bounded-fit output root")
    parser.add_argument("--npz-path", default=B787_3200_CANONICAL_NPZ_PATH)
    parser.add_argument("--role-manifest", default=B787_3200_CANONICAL_MANIFEST_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resume",
        default=None,
        help="exact clean-interruption latest checkpoint for this identity; omitted for a fresh attempt",
    )
    parser.add_argument("--host-rss-limit-gib", type=float, default=32.0)
    return parser.parse_args(argv)


def _normalized_path(value: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(os.fspath(value))))


def _validate_cli(args: argparse.Namespace) -> None:
    if _normalized_path(args.npz_path) != _normalized_path(B787_3200_CANONICAL_NPZ_PATH):
        raise ValueError(
            "bounded GeRaF requires the canonical B787 archive under /storage/home: "
            f"{B787_3200_CANONICAL_NPZ_PATH}"
        )
    if _normalized_path(args.role_manifest) != _normalized_path(B787_3200_CANONICAL_MANIFEST_PATH):
        raise ValueError("bounded GeRaF requires the frozen canonical B787 role manifest")
    if not math.isfinite(float(args.host_rss_limit_gib)) or float(args.host_rss_limit_gib) != 32.0:
        raise ValueError("bounded GeRaF host RSS gate is frozen at 32 GiB")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("bounded real-B787 GeRaF requires one allocated CUDA device")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("bounded real-B787 GeRaF requires exactly one visible CUDA device")
    if args.resume is not None and not str(args.resume).strip():
        raise ValueError("--resume must name an exact checkpoint path")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _peak_host_rss_bytes() -> int | None:
    if resource is None:
        return None
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, OSError, ValueError):
        return None
    # Linux reports KiB; the bounded PACE allocation uses Linux.  Retain a
    # non-negative raw conversion instead of guessing a platform-specific unit.
    return None if value < 0 else value * 1024


def _cache_size_bytes(cache_root: Path) -> int:
    total = 0
    for path in cache_root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return int(total)


def _finite_metric_mapping(metrics: Mapping[str, object], label: str) -> None:
    for name, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            if not math.isfinite(float(value)):
                raise FloatingPointError(f"bounded GeRaF {label}.{name} is non-finite")


def _native_metrics(metrics: Mapping[str, float], peak: float) -> dict[str, object]:
    """Name both normalized-loss values and native ``|MF|`` units explicitly."""

    result = {str(name): float(value) for name, value in metrics.items()}
    normalized_mse = float(result["mf_magnitude_mse"])
    result["normalized_mf_magnitude_mse"] = normalized_mse
    result["normalized_mf_magnitude_rmse"] = float(result["mf_magnitude_rmse"])
    result["native_mf_magnitude_mse"] = normalized_mse * float(peak) ** 2
    result["native_mf_magnitude_rmse"] = float(result["mf_magnitude_rmse"]) * float(peak)
    result["native_readout"] = "complex magnitude |MF| (not squared power)"
    _finite_metric_mapping(result, "metrics")
    return result


def _zero_reference(
    *,
    cache: PreparedB78716SmallfitCache,
    role: str,
    indices: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    """Calculate same-domain zero-predictor error using cached targets only."""

    target_energy = 0.0
    count = 0
    for index in indices:
        view = trainer.load_training_view(cache, int(index), role, args, device)
        target_energy += float(view.target_normalized.square().sum().detach().cpu())
        count += int(view.target_normalized.numel())
        del view
    if count <= 0:
        raise ValueError("bounded GeRaF zero reference has no target values")
    normalized_mse = target_energy / float(count)
    if not math.isfinite(normalized_mse) or normalized_mse < 0.0:
        raise FloatingPointError("bounded GeRaF zero reference is non-finite")
    relative_defined = target_energy > 0.0
    result: dict[str, object] = {
        "views": int(len(indices)),
        "voxels": int(count),
        "predictor": "zero_native_mf_magnitude",
        "target_energy_normalized": target_energy,
        "normalized_mf_magnitude_mse": normalized_mse,
        "normalized_mf_magnitude_rmse": math.sqrt(normalized_mse),
        "native_mf_magnitude_mse": normalized_mse * cache.geraf_mf_magnitude_peak**2,
        "native_mf_magnitude_rmse": math.sqrt(normalized_mse) * cache.geraf_mf_magnitude_peak,
        "relative_mse_defined": relative_defined,
        "mf_magnitude_relative_mse": 1.0 if relative_defined else None,
        "mf_magnitude_relative_l2": 1.0 if relative_defined else None,
    }
    _finite_metric_mapping(result, "zero reference")
    return result


def _mask_snapshot(mask_bank: trainer.PerViewDynamicMaskBank) -> dict[str, object]:
    state = mask_bank.state_dict()
    histories = state.get("histories", {})
    if not isinstance(histories, Mapping):
        raise ValueError("bounded GeRaF dynamic mask state is malformed")
    return {
        "shape": tuple(state["shape"]),
        "high_threshold": float(state["high_threshold"]),
        "low_ratio": float(state["low_ratio"]),
        "low_threshold": float(state["low_threshold"]),
        "histories": {int(key): torch.as_tensor(value).cpu().clone() for key, value in histories.items()},
    }


def _same_mask_snapshot(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    for key in ("shape", "high_threshold", "low_ratio", "low_threshold"):
        if left.get(key) != right.get(key):
            return False
    lhs = left.get("histories")
    rhs = right.get("histories")
    if not isinstance(lhs, Mapping) or not isinstance(rhs, Mapping) or set(lhs) != set(rhs):
        return False
    return all(torch.equal(torch.as_tensor(lhs[key]), torch.as_tensor(rhs[key])) for key in lhs)


def _evaluate_role(
    *,
    model: torch.nn.Module,
    cache: PreparedB78716SmallfitCache,
    frequencies: torch.Tensor,
    kvector: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    indices: Sequence[int],
    role: str,
    mask_bank: trainer.PerViewDynamicMaskBank,
) -> dict[str, object]:
    before = _mask_snapshot(mask_bank)
    metrics, complete = trainer.evaluate_indices(
        model,
        cache,
        frequencies,
        kvector,
        args,
        device,
        indices=indices,
        role=role,
    )
    if not complete:
        raise RuntimeError("bounded GeRaF evaluation ended incomplete")
    after = _mask_snapshot(mask_bank)
    if not _same_mask_snapshot(before, after):
        raise RuntimeError("bounded GeRaF evaluation changed the training dynamic-mask state")
    return _native_metrics(metrics, cache.geraf_mf_magnitude_peak)


def _milestone(
    *,
    update: int,
    model: torch.nn.Module,
    cache: PreparedB78716SmallfitCache,
    frequencies: torch.Tensor,
    kvector: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    mask_bank: trainer.PerViewDynamicMaskBank,
    elapsed_seconds: float,
) -> dict[str, object]:
    fixed_train = _evaluate_role(
        model=model,
        cache=cache,
        frequencies=frequencies,
        kvector=kvector,
        args=args,
        device=device,
        indices=cache.train_indices,
        role="train",
        mask_bank=mask_bank,
    )
    held_out_validation = _evaluate_role(
        model=model,
        cache=cache,
        frequencies=frequencies,
        kvector=kvector,
        args=args,
        device=device,
        indices=cache.validation_indices,
        role="validation",
        mask_bank=mask_bank,
    )
    result: dict[str, object] = {
        "update": int(update),
        "elapsed_seconds": float(elapsed_seconds),
        "fixed_train_native_mf": fixed_train,
        "held_out_validation_native_mf": held_out_validation,
        "same_domain_zero_reference": {
            "fixed_train_native_mf": _zero_reference(
                cache=cache,
                role="train",
                indices=cache.train_indices,
                args=args,
                device=device,
            ),
            "held_out_validation_native_mf": _zero_reference(
                cache=cache,
                role="validation",
                indices=cache.validation_indices,
                args=args,
                device=device,
            ),
        },
        "validation_dynamic_mask_changed": False,
    }
    return result


def _run_identity(
    cache: PreparedB78716SmallfitCache, args: argparse.Namespace
) -> dict[str, object]:
    worklists = bounded_worklists(cache.parent_sealed_identity)
    return {
        "format": CHECKPOINT_FORMAT,
        "version": 1,
        "run_name": RUN_NAME,
        "scope": SCOPE,
        "preparation_id": PREPARATION_ID,
        "fit_id": FIT_ID,
        "cache_schema": CACHE_SCHEMA,
        "cache_recipe": cache.recipe,
        "target_manifest": cache.target_manifest,
        "target_stats": cache.stats,
        "parent_sealed_protocol_identity": cache.parent_sealed_identity,
        "engineering_subset": worklists.as_dict(),
        "model": trainer.model_config_from_args(args),
        "operator": {
            "native_readout": "complex magnitude |MF| (not squared power)",
            "phase_sign": -1.0,
            "backend": "range_nufft",
            "compute_dtype": "float64",
            "full_tx_rx_pairs": 16 * 16,
            "full_frequency_bins": 600,
            "kernel_width": 20,
            "oversample": 2,
            "pair_chunk": 32,
            "point_chunk": 512,
        },
        "optimization": {
            "optimizer": "AdamW",
            "stop_updates": MAX_UPDATES,
            "scheduler": "CosineAnnealingLR",
            "scheduler_horizon_updates": SCHEDULER_HORIZON_UPDATES,
            "milestone_updates": list(MILESTONE_UPDATES),
            "checkpoint_every_updates": 1,
            "validation_dynamic_mask": False,
        },
    }


def _checkpoint_payload(
    *,
    identity: Mapping[str, object],
    cache: PreparedB78716SmallfitCache,
    phase: str,
    completed_updates: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    sampler: trainer.DeterministicViewSampler,
    mask_bank: trainer.PerViewDynamicMaskBank,
    visit_counts: Mapping[int, int],
    milestones: Sequence[Mapping[str, object]],
    last_train: Mapping[str, object] | None,
    started_unix_time: float,
    wall_seconds: float,
    resource_envelope: Mapping[str, object] | None = None,
    terminal_report: Mapping[str, object] | None = None,
) -> dict[str, object]:
    trainer._require_finite_optimization_state(model, optimizer, scheduler)
    return {
        "format": CHECKPOINT_FORMAT,
        "version": 1,
        "run_identity": dict(identity),
        "phase": str(phase),
        "completed_updates": int(completed_updates),
        "scheduler_horizon_updates": SCHEDULER_HORIZON_UPDATES,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "acquisition_record": cache.acquisition_record,
        "rng_state": trainer._capture_rng_state(sampler),
        "dynamic_mask_bank": mask_bank.state_dict(),
        "visit_counts": {str(int(key)): int(value) for key, value in visit_counts.items()},
        "milestones": [copy.deepcopy(dict(row)) for row in milestones],
        "last_train": None if last_train is None else dict(last_train),
        "started_unix_time": float(started_unix_time),
        "wall_seconds": float(wall_seconds),
        "saved_unix_time": time.time(),
        "resource_envelope": None
        if resource_envelope is None
        else copy.deepcopy(dict(resource_envelope)),
        "terminal_report": None
        if terminal_report is None
        else copy.deepcopy(dict(terminal_report)),
    }


def _write_checkpoint(path: Path, **kwargs: Any) -> Path:
    trainer._atomic_torch_save(_checkpoint_payload(**kwargs), path)
    return path


def _integer(value: object, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"bounded GeRaF checkpoint {label} must be an integer")
    result = int(value)
    if result < minimum or result > maximum:
        raise ValueError(f"bounded GeRaF checkpoint {label} is outside [{minimum},{maximum}]")
    return result


def _nonnegative_int(value: object, label: str, *, positive: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"bounded GeRaF resource envelope {label} must be an integer")
    result = int(value)
    if result < 0 or (positive and result == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"bounded GeRaF resource envelope {label} must be {qualifier}")
    return result


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"bounded GeRaF resource envelope {label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"bounded GeRaF resource envelope {label} must be finite and nonnegative")
    return result


def _validate_resource_envelope(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or value.get("schema") != RESOURCE_ENVELOPE_SCHEMA:
        raise ValueError(f"bounded GeRaF {label} lacks the cumulative resource-envelope schema")
    result = {
        "schema": RESOURCE_ENVELOPE_SCHEMA,
        "attempt_count": _nonnegative_int(value.get("attempt_count"), "attempt_count", positive=True),
        "wall_seconds": _finite_nonnegative(value.get("wall_seconds"), "wall_seconds"),
        "current_attempt_wall_seconds": _finite_nonnegative(
            value.get("current_attempt_wall_seconds"), "current_attempt_wall_seconds"
        ),
        "process_max_rss_bytes": None,
        "peak_torch_allocated_bytes": _nonnegative_int(
            value.get("peak_torch_allocated_bytes"), "peak_torch_allocated_bytes"
        ),
        "peak_torch_reserved_bytes": _nonnegative_int(
            value.get("peak_torch_reserved_bytes"), "peak_torch_reserved_bytes"
        ),
        "gpu_total_bytes": _nonnegative_int(value.get("gpu_total_bytes"), "gpu_total_bytes", positive=True),
        "cache_size_bytes": _nonnegative_int(value.get("cache_size_bytes"), "cache_size_bytes"),
    }
    host_rss = value.get("process_max_rss_bytes")
    if host_rss is not None:
        result["process_max_rss_bytes"] = _nonnegative_int(host_rss, "process_max_rss_bytes")
    if int(result["peak_torch_reserved_bytes"]) < int(result["peak_torch_allocated_bytes"]):
        raise ValueError("bounded GeRaF resource envelope reserved GPU memory is below allocated memory")
    return result


def _merge_resource_envelope(
    prior: Mapping[str, object] | None, current_snapshot: Mapping[str, object]
) -> dict[str, object]:
    """Merge one invocation's monotone resource peaks with prior clean-stop evidence."""

    current = _validate_resource_envelope(
        {
            "schema": RESOURCE_ENVELOPE_SCHEMA,
            "attempt_count": 1,
            "wall_seconds": current_snapshot.get("wall_seconds"),
            "current_attempt_wall_seconds": current_snapshot.get("wall_seconds"),
            "process_max_rss_bytes": current_snapshot.get("process_max_rss_bytes"),
            "peak_torch_allocated_bytes": current_snapshot.get("peak_torch_allocated_bytes"),
            "peak_torch_reserved_bytes": current_snapshot.get("peak_torch_reserved_bytes"),
            "gpu_total_bytes": current_snapshot.get("gpu_total_bytes"),
            "cache_size_bytes": current_snapshot.get("cache_size_bytes"),
        },
        "current resource snapshot",
    )
    if prior is None:
        return current
    previous = _validate_resource_envelope(prior, "clean checkpoint resource evidence")
    if previous["gpu_total_bytes"] != current["gpu_total_bytes"]:
        raise ValueError("bounded GeRaF clean continuation changed the allocated GPU-memory capacity")
    previous_rss = previous["process_max_rss_bytes"]
    current_rss = current["process_max_rss_bytes"]
    known_rss = [value for value in (previous_rss, current_rss) if value is not None]
    return {
        "schema": RESOURCE_ENVELOPE_SCHEMA,
        "attempt_count": int(previous["attempt_count"]) + 1,
        "wall_seconds": float(previous["wall_seconds"]) + float(current["wall_seconds"]),
        "current_attempt_wall_seconds": float(current["wall_seconds"]),
        "process_max_rss_bytes": None if not known_rss else max(int(value) for value in known_rss),
        "peak_torch_allocated_bytes": max(
            int(previous["peak_torch_allocated_bytes"]), int(current["peak_torch_allocated_bytes"])
        ),
        "peak_torch_reserved_bytes": max(
            int(previous["peak_torch_reserved_bytes"]), int(current["peak_torch_reserved_bytes"])
        ),
        "gpu_total_bytes": int(current["gpu_total_bytes"]),
        "cache_size_bytes": max(int(previous["cache_size_bytes"]), int(current["cache_size_bytes"])),
    }


def _validate_checkpoint(
    payload: object,
    *,
    identity: Mapping[str, object],
    cache: PreparedB78716SmallfitCache,
    required_phase: str = "interrupted_clean",
) -> tuple[int, dict[int, int], list[dict[str, object]], Mapping[str, object] | None, float]:
    if not isinstance(payload, Mapping):
        raise ValueError("bounded GeRaF checkpoint must be a mapping")
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("version") != 1:
        raise ValueError("bounded GeRaF checkpoint format/version mismatch")
    if payload.get("run_identity") != dict(identity):
        raise ValueError("bounded GeRaF checkpoint belongs to a different cache/model/recipe identity")
    saved_acquisition = payload.get("acquisition_record")
    if not isinstance(saved_acquisition, Mapping) or not acquisition_records_equal(
        saved_acquisition, cache.acquisition_record
    ):
        raise ValueError(
            "bounded GeRaF checkpoint was made with different calibrated poses, metadata, or frequency grid"
        )
    if required_phase not in {"interrupted_clean", "complete"}:
        raise ValueError("bounded GeRaF checkpoint validator received an unsupported required phase")
    if payload.get("phase") != required_phase:
        if required_phase == "interrupted_clean":
            raise ValueError("only a clean-interruption bounded GeRaF checkpoint may resume")
        raise ValueError(f"bounded GeRaF checkpoint must have phase {required_phase!r}")
    step = _integer(payload.get("completed_updates"), "completed_updates", 0, MAX_UPDATES)
    if payload.get("scheduler_horizon_updates") != SCHEDULER_HORIZON_UPDATES:
        raise ValueError("bounded GeRaF checkpoint changed the preserved scheduler horizon")
    for name in (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "rng_state",
        "dynamic_mask_bank",
        "visit_counts",
        "milestones",
    ):
        if name not in payload:
            raise ValueError(f"bounded GeRaF checkpoint lacks {name!r}")
    if not isinstance(payload["model_state_dict"], Mapping) or not isinstance(
        payload["optimizer_state_dict"], Mapping
    ) or not isinstance(payload["scheduler_state_dict"], Mapping):
        raise ValueError("bounded GeRaF checkpoint model/optimizer/scheduler state is malformed")
    scheduler_state = payload["scheduler_state_dict"]
    if int(scheduler_state.get("T_max", -1)) != SCHEDULER_HORIZON_UPDATES:
        raise ValueError("bounded GeRaF checkpoint scheduler has a shortened or changed horizon")
    if int(scheduler_state.get("last_epoch", -1)) != step:
        raise ValueError("bounded GeRaF checkpoint scheduler clock disagrees with completed updates")
    counts_raw = payload["visit_counts"]
    if not isinstance(counts_raw, Mapping):
        raise ValueError("bounded GeRaF checkpoint visit counts are malformed")
    expected_ids = set(cache.train_indices)
    if {int(key) for key in counts_raw} != expected_ids:
        raise ValueError("bounded GeRaF checkpoint visit-count IDs differ from its 16 train views")
    counts = {
        int(key): _integer(value, f"visit_counts[{key}]", 0, 2)
        for key, value in counts_raw.items()
    }
    if sum(counts.values()) != step:
        raise ValueError("bounded GeRaF checkpoint visit counts do not sum to completed updates")
    milestones_raw = payload["milestones"]
    if not isinstance(milestones_raw, list):
        raise ValueError("bounded GeRaF checkpoint milestones must be a list")
    milestones = [dict(row) for row in milestones_raw if isinstance(row, Mapping)]
    if len(milestones) != len(milestones_raw):
        raise ValueError("bounded GeRaF checkpoint contains a malformed milestone")
    expected_milestones = [value for value in MILESTONE_UPDATES if value <= step]
    if [int(row.get("update", -1)) for row in milestones] != expected_milestones:
        raise ValueError("bounded GeRaF checkpoint milestone history is incomplete or out of order")
    last_train = payload.get("last_train")
    if last_train is not None and not isinstance(last_train, Mapping):
        raise ValueError("bounded GeRaF checkpoint last_train must be an object or null")
    if step and last_train is None:
        raise ValueError("bounded GeRaF checkpoint after an update lacks last_train")
    if last_train is not None:
        last_step = _integer(last_train.get("step"), "last_train.step", 1, MAX_UPDATES)
        if last_step != step:
            raise ValueError("bounded GeRaF checkpoint last_train step disagrees with completed updates")
        last_view = _integer(last_train.get("view_index"), "last_train.view_index", 0, 9_999)
        if last_view not in expected_ids:
            raise ValueError("bounded GeRaF checkpoint last_train names a non-selected training view")
    try:
        started = float(payload.get("started_unix_time"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("bounded GeRaF checkpoint started time is invalid") from exc
    if not math.isfinite(started):
        raise ValueError("bounded GeRaF checkpoint started time is non-finite")
    return step, counts, milestones, last_train, started


def _load_initial_state(checkpoint_dir: Path, identity: Mapping[str, object]) -> Mapping[str, torch.Tensor]:
    initial_path = checkpoint_dir / _CHECKPOINT_INITIAL
    if not initial_path.is_file():
        raise FileNotFoundError("bounded GeRaF lacks checkpoint_initial for parameter-change evidence")
    payload = torch.load(initial_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("run_identity") != dict(identity):
        raise ValueError("bounded GeRaF checkpoint_initial belongs to a different identity")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("bounded GeRaF checkpoint_initial lacks a model state")
    return {str(key): torch.as_tensor(value).detach().cpu().clone() for key, value in state.items()}


def _parameter_change(
    model: torch.nn.Module, initial_state: Mapping[str, torch.Tensor]
) -> dict[str, float]:
    total_squared = 0.0
    maximum = 0.0
    changed_tensors = 0
    state = model.state_dict()
    if set(state) != set(initial_state):
        raise ValueError("bounded GeRaF final model state differs from checkpoint_initial keys")
    for name, current in state.items():
        delta = current.detach().cpu().to(torch.float64) - initial_state[name].to(torch.float64)
        squared = float(delta.square().sum())
        maximum = max(maximum, float(delta.abs().max()) if delta.numel() else 0.0)
        total_squared += squared
        changed_tensors += int(bool(delta.numel()) and bool((delta != 0).any()))
    l2 = math.sqrt(total_squared)
    if not math.isfinite(l2) or not math.isfinite(maximum) or l2 <= 0.0 or changed_tensors <= 0:
        raise RuntimeError("bounded GeRaF did not produce a finite nonzero parameter update")
    return {
        "parameter_delta_l2": l2,
        "parameter_delta_max_abs": maximum,
        "changed_tensor_count": float(changed_tensors),
    }


def _resource_snapshot(started: float, cache_root: Path) -> dict[str, object]:
    allocated = int(torch.cuda.max_memory_allocated())
    reserved = int(torch.cuda.max_memory_reserved())
    total = int(torch.cuda.get_device_properties(0).total_memory)
    return {
        "wall_seconds": time.monotonic() - started,
        "process_max_rss_bytes": _peak_host_rss_bytes(),
        "peak_torch_allocated_bytes": allocated,
        "peak_torch_reserved_bytes": reserved,
        "gpu_total_bytes": total,
        "cache_size_bytes": _cache_size_bytes(cache_root),
    }


def _write_report(path: Path, payload: Mapping[str, object]) -> Path:
    atomic_write_json(path, _json_safe(dict(payload)))
    return path


@dataclass(frozen=True)
class _StartPaths:
    clean_resume_checkpoint: Path | None
    terminal_finalize_checkpoint: Path | None


def _fresh_paths(checkpoint_dir: Path, resume: str | None) -> _StartPaths:
    paths = [
        checkpoint_dir / _CHECKPOINT_INITIAL,
        checkpoint_dir / _CHECKPOINT_LATEST,
        checkpoint_dir / _CHECKPOINT_BEST,
        checkpoint_dir / _CHECKPOINT_FINAL,
        checkpoint_dir / _REPORT,
        checkpoint_dir / _POSTFLIGHT,
        checkpoint_dir / "run_config.json",
    ]
    if resume is None:
        existing = [path for path in paths if path.exists()]
        if existing:
            raise FileExistsError(
                "fresh bounded GeRaF fit refuses existing evidence; use an exact clean --resume "
                f"or a new checkpoint directory (found {existing[0]})"
            )
        return _StartPaths(clean_resume_checkpoint=None, terminal_finalize_checkpoint=None)
    path = Path(resume).expanduser().resolve()
    if path != (checkpoint_dir / _CHECKPOINT_LATEST).resolve():
        raise ValueError("bounded GeRaF --resume must name this run's exact checkpoint_latest")
    final_path = checkpoint_dir / _CHECKPOINT_FINAL
    report_path = checkpoint_dir / _REPORT
    if report_path.exists() or (checkpoint_dir / _POSTFLIGHT).exists():
        raise ValueError("bounded GeRaF terminal evidence already exists and must not resume")
    if final_path.is_file():
        # The final checkpoint embeds the terminal report before publication.
        # An explicit resume can therefore repair only the missing report,
        # without running another optimizer update or reopening raw responses.
        return _StartPaths(clean_resume_checkpoint=None, terminal_finalize_checkpoint=final_path)
    if not path.is_file():
        raise FileNotFoundError(f"bounded GeRaF resume checkpoint is missing: {path}")
    return _StartPaths(clean_resume_checkpoint=path, terminal_finalize_checkpoint=None)


def _recover_terminal_report(
    *,
    checkpoint_dir: Path,
    final_checkpoint: Path,
    payload: object,
    identity: Mapping[str, object],
    cache: PreparedB78716SmallfitCache,
) -> dict[str, object]:
    """Publish a report embedded in a valid complete checkpoint, without fitting.

    Final checkpoint publication is atomic but report publication is a separate
    file.  This explicit recovery path closes that unavoidable two-file crash
    window without replaying an optimizer update or reopening raw responses.
    """

    step, visit_counts, milestones, _last_train, _started = _validate_checkpoint(
        payload, identity=identity, cache=cache, required_phase="complete"
    )
    if step != MAX_UPDATES or any(value != 2 for value in visit_counts.values()):
        raise ValueError("bounded GeRaF terminal checkpoint does not contain the completed two-cycle fit")
    if [int(row.get("update", -1)) for row in milestones] != list(MILESTONE_UPDATES):
        raise ValueError("bounded GeRaF terminal checkpoint lacks the required milestone sequence")
    if not isinstance(payload, Mapping):  # Kept explicit for the following direct read.
        raise ValueError("bounded GeRaF terminal checkpoint must be a mapping")
    embedded = payload.get("terminal_report")
    if not isinstance(embedded, Mapping):
        raise ValueError("bounded GeRaF complete checkpoint lacks recoverable terminal report evidence")
    report = dict(embedded)
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("scope") != SCOPE
        or report.get("run_identity") != dict(identity)
    ):
        raise ValueError("bounded GeRaF embedded terminal report has a different identity or scope")
    terminal_resources = _validate_resource_envelope(
        payload.get("resource_envelope"), "terminal checkpoint"
    )
    report_resources = report.get("resources")
    if (
        not isinstance(report_resources, Mapping)
        or _validate_resource_envelope(
            report_resources, "embedded terminal report resource evidence"
        )
        != terminal_resources
    ):
        raise ValueError(
            "bounded GeRaF embedded terminal report resource evidence differs from its complete checkpoint"
        )
    checkpoints = report.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        raise ValueError("bounded GeRaF embedded terminal report lacks checkpoint paths")
    expected_final = str(final_checkpoint.resolve())
    expected_latest = str((checkpoint_dir / _CHECKPOINT_LATEST).resolve())
    if checkpoints.get("final") != expected_final or checkpoints.get("latest") != expected_latest:
        raise ValueError("bounded GeRaF embedded terminal report names different terminal checkpoints")
    for name in (_CHECKPOINT_INITIAL, _CHECKPOINT_BEST):
        if not (checkpoint_dir / name).is_file():
            raise FileNotFoundError(f"bounded GeRaF terminal recovery lacks {name}")

    # A crash may have occurred before the terminal latest checkpoint was
    # published.  The valid final payload is authoritative and can restore it
    # atomically before the report becomes visible.
    trainer._atomic_torch_save(dict(payload), checkpoint_dir / _CHECKPOINT_LATEST)
    report_path = _write_report(checkpoint_dir / _REPORT, report)
    print(f"GERAF_B78716_SMALLFIT_TERMINAL_REPORT_RECOVERED report={report_path}", flush=True)
    return report


def _pre_fit_stop_outcome(fresh_fit: bool) -> _PreFitStop | None:
    """Only a fresh preparation may stop before a resumable fit checkpoint exists."""

    if fresh_fit and _STOP_REQUESTED:
        print("GERAF_B78716_SMALLFIT_STOPPED_BEFORE_FIT_NO_CHECKPOINT", flush=True)
        return _PreFitStop(signal=_STOP_SIGNAL)
    return None


def run(args: argparse.Namespace) -> dict[str, object] | _CleanStop | _PreFitStop:
    """Run preparation and bounded fitting once; returns a terminal report if complete."""

    global _STOP_REQUESTED, _STOP_SIGNAL
    _STOP_REQUESTED = False
    _STOP_SIGNAL = None
    _validate_cli(args)
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    _publish_timed_wrapper_ready()
    device = torch.device(args.device)
    started_monotonic = time.monotonic()
    started_unix = time.time()
    torch.cuda.reset_peak_memory_stats(device)

    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    start_paths = _fresh_paths(checkpoint_dir, args.resume)
    fresh_fit = (
        start_paths.clean_resume_checkpoint is None
        and start_paths.terminal_finalize_checkpoint is None
    )
    if fresh_fit:
        # One fresh invocation owns the dependency: fit cannot begin until the
        # preparer returns a completed, metadata-validated 20-target cache.
        prepare_complete_subset_cache(
            cache_root=args.cache_root,
            npz_path=args.npz_path,
            role_manifest=args.role_manifest,
            device=str(device),
        )
        stop_outcome = _pre_fit_stop_outcome(fresh_fit=True)
        if stop_outcome is not None:
            return stop_outcome
    # Both clean continuation and terminal report recovery use only the
    # metadata-only cache reader.  They never reopen the raw response archive.
    cache = load_complete_subset_cache(
        cache_root=args.cache_root,
        npz_path=args.npz_path,
        role_manifest=args.role_manifest,
        device=str(device),
    )
    stop_outcome = _pre_fit_stop_outcome(fresh_fit=fresh_fit)
    if stop_outcome is not None:
        return stop_outcome
    training = training_args(device=str(device))
    trainer._validate_cli(training)
    if int(training.steps) != SCHEDULER_HORIZON_UPDATES:
        raise AssertionError("bounded GeRaF adapter unexpectedly changed the production scheduler horizon")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    identity = _run_identity(cache, training)
    run_config = checkpoint_dir / "run_config.json"
    if run_config.exists():
        with run_config.open("r", encoding="utf-8") as handle:
            prior = json.load(handle)
        if not isinstance(prior, Mapping) or prior.get("run_identity") != identity:
            raise ValueError("bounded GeRaF run configuration belongs to a different identity")
    elif not fresh_fit:
        raise FileNotFoundError("bounded GeRaF resume/recovery lacks its original run configuration")
    else:
        atomic_write_json(run_config, {"run_identity": identity, "scope": SCOPE})

    if start_paths.terminal_finalize_checkpoint is not None:
        terminal_payload = torch.load(
            start_paths.terminal_finalize_checkpoint, map_location="cpu", weights_only=False
        )
        return _recover_terminal_report(
            checkpoint_dir=checkpoint_dir,
            final_checkpoint=start_paths.terminal_finalize_checkpoint,
            payload=terminal_payload,
            identity=identity,
            cache=cache,
        )

    resume_payload: Mapping[str, object] | None = None
    prior_resource_envelope: Mapping[str, object] | None = None
    resume_state: tuple[
        int,
        dict[int, int],
        list[dict[str, object]],
        Mapping[str, object] | None,
        float,
    ] | None = None
    if start_paths.clean_resume_checkpoint is not None:
        loaded_payload = torch.load(
            start_paths.clean_resume_checkpoint, map_location="cpu", weights_only=False
        )
        resume_state = _validate_checkpoint(loaded_payload, identity=identity, cache=cache)
        if not isinstance(loaded_payload, Mapping):
            raise AssertionError("bounded GeRaF checkpoint validator accepted a non-mapping payload")
        resume_payload = loaded_payload
        prior_resource_envelope = _validate_resource_envelope(
            loaded_payload.get("resource_envelope"), "clean checkpoint"
        )

    frequencies = torch.as_tensor(
        validate_b7873200_operator_frequency_grid(
            cache.acquisition_record, frequency_grid_hz(cache.source.arrays.metadata)
        ),
        dtype=torch.float64,
        device=device,
    )
    kvector = get_kvector(frequencies, cc)
    trainer._seed_everything(training.seed)
    model = trainer.GeRaFModel(**trainer.model_config_from_args(training)).to(
        device=device, dtype=torch.float32
    )
    optimizer = trainer.build_geraf_optimizer(model, training)
    scheduler = trainer.build_geraf_scheduler(
        optimizer, training, horizon_steps=SCHEDULER_HORIZON_UPDATES
    )
    sampler = trainer.DeterministicViewSampler(cache.train_indices, training.seed)
    mask_bank = trainer.PerViewDynamicMaskBank(
        cache.train_indices,
        cache.grid_shape,
        high_threshold=training.mask_high_threshold,
        low_ratio=training.mask_low_ratio,
        low_threshold=training.mask_low_threshold,
        device=device,
    )
    visit_counts = {int(index): 0 for index in cache.train_indices}
    milestones: list[dict[str, object]] = []
    last_train: Mapping[str, object] | None = None
    step = 0

    if resume_state is not None:
        if resume_payload is None:
            raise AssertionError("bounded GeRaF has resume state without its checkpoint payload")
        step, visit_counts, milestones, last_train, started_unix = resume_state
        model.load_state_dict(resume_payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        trainer._optimizer_to(optimizer, device)
        scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        sampler.load_state_dict(resume_payload["rng_state"]["view_sampler"])
        mask_bank.load_state_dict(resume_payload["dynamic_mask_bank"])
        trainer._restore_rng_state(resume_payload["rng_state"], sampler)
        trainer._require_finite_optimization_state(model, optimizer, scheduler)
        print(f"Resumed bounded GeRaF clean checkpoint at update {step}/{MAX_UPDATES}.", flush=True)
    else:
        milestones.append(
            _milestone(
                update=0,
                model=model,
                cache=cache,
                frequencies=frequencies,
                kvector=kvector,
                args=training,
                device=device,
                mask_bank=mask_bank,
                elapsed_seconds=time.monotonic() - started_monotonic,
            )
        )
        initial_kwargs = dict(
            identity=identity,
            cache=cache,
            phase="initialized",
            completed_updates=0,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            mask_bank=mask_bank,
            visit_counts=visit_counts,
            milestones=milestones,
            last_train=last_train,
            started_unix_time=started_unix,
            wall_seconds=time.monotonic() - started_monotonic,
        )
        _write_checkpoint(checkpoint_dir / _CHECKPOINT_INITIAL, **initial_kwargs)
        _write_checkpoint(checkpoint_dir / _CHECKPOINT_LATEST, **initial_kwargs)

    best_validation_mse = math.inf
    for row in milestones:
        validation = row["held_out_validation_native_mf"]
        if isinstance(validation, Mapping):
            best_validation_mse = min(best_validation_mse, float(validation["mf_magnitude_mse"]))
    if not math.isfinite(best_validation_mse):
        raise RuntimeError("bounded GeRaF lacks a finite update-0 validation metric")

    cache_root_path = Path(args.cache_root).expanduser().resolve()

    def cumulative_resources() -> dict[str, object]:
        return _merge_resource_envelope(
            prior_resource_envelope,
            _resource_snapshot(started_monotonic, cache_root_path),
        )

    def save_latest(phase: str) -> Path:
        resource_envelope = cumulative_resources()
        return _write_checkpoint(
            checkpoint_dir / _CHECKPOINT_LATEST,
            identity=identity,
            cache=cache,
            phase=phase,
            completed_updates=step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            mask_bank=mask_bank,
            visit_counts=visit_counts,
            milestones=milestones,
            last_train=last_train,
            started_unix_time=started_unix,
            wall_seconds=float(resource_envelope["wall_seconds"]),
            resource_envelope=resource_envelope,
        )

    # Update 0 is a real held-out evaluation and therefore establishes the
    # first best checkpoint even if later milestones are worse.
    if not (checkpoint_dir / _CHECKPOINT_BEST).exists():
        _write_checkpoint(
            checkpoint_dir / _CHECKPOINT_BEST,
            identity=identity,
            cache=cache,
            phase="initialized" if step == 0 else "running",
            completed_updates=step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            mask_bank=mask_bank,
            visit_counts=visit_counts,
            milestones=milestones,
            last_train=last_train,
            started_unix_time=started_unix,
            wall_seconds=time.monotonic() - started_monotonic,
        )

    if _STOP_REQUESTED:
        latest = save_latest("interrupted_clean")
        print(f"GERAF_B78716_SMALLFIT_STOPPED_CLEANLY checkpoint={latest}", flush=True)
        return _CleanStop(checkpoint=latest, signal=_STOP_SIGNAL)

    while step < MAX_UPDATES:
        last_train = trainer.train_one_view_update(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            mask_bank=mask_bank,
            cache=cache,
            frequencies=frequencies,
            kvector=kvector,
            args=training,
            device=device,
            completed_step=step,
        )
        step = int(last_train["step"])
        view_index = int(last_train["view_index"])
        if view_index not in visit_counts:
            raise RuntimeError("bounded GeRaF sampler emitted a non-selected training ID")
        visit_counts[view_index] += 1
        if visit_counts[view_index] > 2:
            raise RuntimeError("bounded GeRaF used a selected training view more than twice")
        print(
            f"bounded GeRaF update={step}/{MAX_UPDATES} view={view_index} "
            f"loss={float(last_train['loss']):.7e} "
            f"lr=({float(last_train['sdf_lr']):.3e},{float(last_train['other_lr']):.3e})",
            flush=True,
        )
        if step in MILESTONE_UPDATES:
            milestones.append(
                _milestone(
                    update=step,
                    model=model,
                    cache=cache,
                    frequencies=frequencies,
                    kvector=kvector,
                    args=training,
                    device=device,
                    mask_bank=mask_bank,
                    elapsed_seconds=time.monotonic() - started_monotonic,
                )
            )
            if step == NUM_TRAIN and any(count != 1 for count in visit_counts.values()):
                raise RuntimeError(
                    "bounded GeRaF first 16-update cycle did not use every selected training view exactly once"
                )
            validation = milestones[-1]["held_out_validation_native_mf"]
            if not isinstance(validation, Mapping):
                raise AssertionError("bounded GeRaF milestone validation is malformed")
            mse = float(validation["mf_magnitude_mse"])
            if mse < best_validation_mse:
                best_validation_mse = mse
                _write_checkpoint(
                    checkpoint_dir / _CHECKPOINT_BEST,
                    identity=identity,
                    cache=cache,
                    phase="running",
                    completed_updates=step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=sampler,
                    mask_bank=mask_bank,
                    visit_counts=visit_counts,
                    milestones=milestones,
                    last_train=last_train,
                    started_unix_time=started_unix,
                    wall_seconds=time.monotonic() - started_monotonic,
                )
        if _STOP_REQUESTED:
            latest = save_latest("interrupted_clean")
            print(f"GERAF_B78716_SMALLFIT_STOPPED_CLEANLY checkpoint={latest}", flush=True)
            return _CleanStop(checkpoint=latest, signal=_STOP_SIGNAL)
        save_latest("running")

    if any(count != 2 for count in visit_counts.values()):
        raise RuntimeError("bounded GeRaF did not use every selected training view exactly twice")
    if [int(row["update"]) for row in milestones] != list(MILESTONE_UPDATES):
        raise RuntimeError("bounded GeRaF did not record exactly the required 0/16/32 milestones")
    if int(scheduler.state_dict().get("last_epoch", -1)) != MAX_UPDATES:
        raise RuntimeError("bounded GeRaF scheduler clock does not equal the 32 completed updates")
    if int(scheduler.state_dict().get("T_max", -1)) != SCHEDULER_HORIZON_UPDATES:
        raise RuntimeError("bounded GeRaF scheduler horizon was shortened from 50,000 updates")
    parameter_change = _parameter_change(model, _load_initial_state(checkpoint_dir, identity))
    resources = cumulative_resources()
    host_rss = resources["process_max_rss_bytes"]
    host_limit = int(float(args.host_rss_limit_gib) * (1024**3))
    if host_rss is not None and int(host_rss) >= int(0.8 * host_limit):
        raise RuntimeError("bounded GeRaF process RSS reached the 80% engineering gate")
    if int(resources["peak_torch_allocated_bytes"]) >= int(0.8 * int(resources["gpu_total_bytes"])):
        raise RuntimeError("bounded GeRaF allocated GPU memory reached the 80% engineering gate")
    if int(resources["peak_torch_reserved_bytes"]) >= int(0.8 * int(resources["gpu_total_bytes"])):
        raise RuntimeError("bounded GeRaF reserved GPU memory reached the 80% engineering gate")

    final_path = checkpoint_dir / _CHECKPOINT_FINAL
    latest_path = checkpoint_dir / _CHECKPOINT_LATEST
    report = {
        "schema": REPORT_SCHEMA,
        "version": 1,
        "run_name": RUN_NAME,
        "scope": SCOPE,
        "production_clearance": False,
        "reason_not_production": "16/4 engineering cache and 32 updates are lifecycle evidence only",
        "run_identity": identity,
        "cache": {
            "root": str(Path(args.cache_root).expanduser().resolve()),
            "recipe_filename": CACHE_RECIPE_FILENAME,
            "manifest_filename": CACHE_MANIFEST_FILENAME,
            "stats_filename": CACHE_STATS_FILENAME,
            "protocol_filename": CACHE_PROTOCOL_FILENAME,
            "target_grid_shape": list(TARGET_GRID_SHAPE),
            "normalization": "train-only maximum native |MF| over exactly 16 targets",
        },
        "milestones": milestones,
        "train_view_visit_counts": {str(index): count for index, count in visit_counts.items()},
        "finite_gradients_and_updates": True,
        "parameter_change": parameter_change,
        "scheduler": {
            "stop_updates": MAX_UPDATES,
            "horizon_updates": SCHEDULER_HORIZON_UPDATES,
            "last_epoch": int(scheduler.state_dict()["last_epoch"]),
        },
        "resources": resources,
        "checkpoints": {
            "initial": str(checkpoint_dir / _CHECKPOINT_INITIAL),
            "best": str(checkpoint_dir / _CHECKPOINT_BEST),
            "final": str(final_path),
            "latest": str(latest_path),
            "clean_checkpoint_recovery_supported": True,
        },
    }
    # Embed the complete report in both terminal checkpoints before publishing
    # its JSON sidecar. An explicit metadata-only recovery can close the
    # two-file publication gap without another training update.
    terminal_kwargs = dict(
        identity=identity,
        cache=cache,
        phase="complete",
        completed_updates=step,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        mask_bank=mask_bank,
        visit_counts=visit_counts,
        milestones=milestones,
        last_train=last_train,
        started_unix_time=started_unix,
        resource_envelope=resources,
        terminal_report=report,
    )
    final = _write_checkpoint(
        final_path,
        wall_seconds=float(resources["wall_seconds"]),
        **terminal_kwargs,
    )
    latest = _write_checkpoint(
        latest_path,
        wall_seconds=float(resources["wall_seconds"]),
        **terminal_kwargs,
    )
    report_path = _write_report(checkpoint_dir / _REPORT, report)
    print(f"GERAF_B78716_SMALLFIT_REPORT_WRITTEN report={report_path}", flush=True)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    result = run(parse_args(argv))
    if isinstance(result, _CleanStop):
        if not result.checkpoint.is_file():
            raise RuntimeError("bounded GeRaF clean stop did not leave a usable latest checkpoint")
        print("GERAF_B78716_SMALLFIT_CLEAN_STOP_RESUMABLE", flush=True)
        raise SystemExit(CLEAN_STOP_EXIT_CODE)
    if isinstance(result, _PreFitStop):
        raise SystemExit(PRE_FIT_STOP_EXIT_CODE)
    print("GERAF_B78716_SMALLFIT_LIFECYCLE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
