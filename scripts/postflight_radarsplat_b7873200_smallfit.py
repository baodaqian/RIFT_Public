#!/usr/bin/env python
"""Audit one completed native-power RadarSplat B787 8/4 engineering fit.

This reader never opens the raw B787 archive.  It reuses the cache-only
trainer and adapter to measure the fixed selected train and validation roles,
then records the bounded run's lifecycle, update, resource, and Gaussian
occupancy evidence.  It is deliberately not an NVS, coherent-signal, mesh,
reconstruction, convergence, or baseline-comparison evaluator.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import sys
from typing import Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_radarsplat as trainer
from rift.radarsplat_b7873200 import RadarSplatEffects, RadarSplatModel
from rift.radarsplat_b7873200_adapter import model_from_checkpoint_state
from rift.radarsplat_b7873200_protocol import atomic_write_json, load_cache


EXPECTED_TRAIN_VIEWS = 8
EXPECTED_VALIDATION_VIEWS = 4
EXPECTED_STEPS = 20
EXPECTED_INITIAL_GAUSSIANS = 64
EXPECTED_GAUSSIAN_CHUNK_SIZE = 64
EXPECTED_MAX_RASTER_CANDIDATE_PAIRS = 2_000_000
PREPARE_RESOURCE_PREFIX = "RADARSPLAT_B7873200_PREPARE_RESOURCE_JSON="
TRAIN_RESOURCE_PREFIX = "RADARSPLAT_B7873200_TRAIN_RESOURCE_JSON="
COMPLETION_RESUME_MARKER = "RadarSplat B7873200 completion artifacts are present; no update executed."
POSTFLIGHT_CLEAN_STOP_MARKER = "RADARSPLAT_B7873200_POSTFLIGHT_CLEAN_STOP"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--launcher-log-root", required=True)
    parser.add_argument("--completion-resume-log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gaussian-chunk-size", type=int, default=EXPECTED_GAUSSIAN_CHUNK_SIZE)
    parser.add_argument("--max-raster-candidate-pairs", type=int, default=EXPECTED_MAX_RASTER_CANDIDATE_PAIRS)
    parser.add_argument("--host-limit-gib", type=int, default=32)
    parser.add_argument("--gpu-total-mib", type=int, required=True)
    parser.add_argument("--whole-job-rss-raw", default=None)
    return parser.parse_args(argv)


def _finite_number(value: object, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"RadarSplat B787 small-fit {label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"RadarSplat B787 small-fit {label} must be finite" + (" and non-negative" if nonnegative else ""))
    return result


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"RadarSplat B787 small-fit {label} must be a mapping")
    return value


def _positive_int(value: object, label: str) -> int:
    numeric = _finite_number(value, label, nonnegative=True)
    if numeric < 1.0 or not numeric.is_integer():
        raise ValueError(f"RadarSplat B787 small-fit {label} must be a positive integer")
    return int(numeric)


def _nonnegative_int(value: object, label: str) -> int:
    numeric = _finite_number(value, label, nonnegative=True)
    if not numeric.is_integer():
        raise ValueError(f"RadarSplat B787 small-fit {label} must be an integer")
    return int(numeric)


def _read_json(path: Path, label: str) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"RadarSplat B787 small-fit {label} is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return _mapping(payload, label)


def _request_postflight_stop(_signum: int, _frame: object) -> None:
    """Leave explicit safe-stop evidence before the read-only postflight exits."""

    print(POSTFLIGHT_CLEAN_STOP_MARKER, flush=True)
    raise SystemExit(143)


def _require_exact_smallfit_identity(cache, checkpoint: Mapping[str, object]) -> Mapping[str, object]:
    if not cache.is_development_subset:
        raise ValueError("RadarSplat B787 small-fit cache must be explicitly marked as an engineering subset")
    if len(cache.train_indices) != EXPECTED_TRAIN_VIEWS or len(cache.validation_indices) != EXPECTED_VALIDATION_VIEWS:
        raise ValueError("RadarSplat B787 small-fit must expose exactly the approved 8 train and 4 validation views")
    sealed_roles = _mapping(cache.identity.get("role_ids"), "sealed role IDs")
    expected_train = tuple(int(value) for value in sealed_roles["train"][:EXPECTED_TRAIN_VIEWS])
    expected_validation = tuple(int(value) for value in sealed_roles["validation"][:EXPECTED_VALIDATION_VIEWS])
    if cache.train_indices != expected_train or cache.validation_indices != expected_validation:
        raise ValueError("RadarSplat B787 small-fit cache is not the ordered canonical train[:8]/validation[:4] subset")
    identity = _mapping(checkpoint.get("run_identity"), "checkpoint run identity")
    if identity.get("method") != "radarsplat_b7873200_native_power_v1":
        raise ValueError("RadarSplat B787 small-fit checkpoint has the wrong native-power method")
    if identity.get("sealed_protocol_identity") != cache.identity or identity.get("target_recipe") != cache.recipe:
        raise ValueError("RadarSplat B787 small-fit checkpoint does not match the prepared cache")
    model = _mapping(identity.get("model"), "model identity")
    optimization = _mapping(identity.get("optimization"), "optimization identity")
    expected_model = {
        "init_num_gaussians": EXPECTED_INITIAL_GAUSSIANS,
        "init_extent_m": 0.3,
        "init_scale_m": 0.003,
        "init_opacity": 0.1,
        "init_noise_probability": 0.1,
        "sh_degree": 3,
        "sh_degree_interval": 600,
        "planar_initialization": True,
    }
    if model != expected_model:
        raise ValueError("RadarSplat B787 small-fit changed the declared native model recipe")
    expected_optimization = {
        "steps": EXPECTED_STEPS,
        "validation_every": 5,
        "learning_rates": {
            "means": 4.8e-05,
            "log_scales": 0.005,
            "quaternions": 0.001,
            "opacity_logits": 0.05,
            "noise_probability_logits": 0.05,
            "sh0": 0.0025,
            "shN": 0.0025,
        },
        "betas": [0.9, 0.999],
        "eps": 1.0e-15,
        "seed": 42,
    }
    if optimization != expected_optimization:
        raise ValueError("RadarSplat B787 small-fit changed the declared native optimizer recipe")
    objective = _mapping(identity.get("objective"), "objective identity")
    if objective != {
        "weights": {"ssim": 0.2, "occupancy": 5.0, "max_size": 100.0, "opacity_noise": 100.0},
        "occupancy_threshold": 0.001,
        "occupancy_balance": "equal positive/background class means when both exist",
        "max_scale_m": 0.006,
    }:
        raise ValueError("RadarSplat B787 small-fit changed the declared native objective")
    strategy = _mapping(identity.get("strategy"), "strategy identity")
    if strategy != {
        "prune_opacity": 0.0005,
        "strict_prune_comparison": "opacity < prune_opacity, with float32 sigmoid/logit boundary roundoff only",
        "prune_every": 5,
        "all_rows_prune_guard": "retain maximum-opacity row",
    }:
        raise ValueError("RadarSplat B787 small-fit changed the declared native pruning policy")
    renderer = _mapping(identity.get("renderer"), "renderer identity")
    expected_renderer = trainer._renderer_identity(
        trainer.load_native_view(cache, cache.train_indices[0], "train", "cpu").grid,
        RadarSplatEffects.b787_clean(),
    )
    if renderer != expected_renderer:
        raise ValueError("RadarSplat B787 small-fit changed its native renderer semantics")
    return identity


def _evaluation_args(args: argparse.Namespace) -> argparse.Namespace:
    """Use the trainer's declared evaluation knobs without a raw-data route."""

    return trainer.parse_args(
        [
            "--cache-root", args.cache_root,
            "--checkpoint-dir", args.checkpoint_dir,
            "--device", args.device,
            "--gaussian-chunk-size", str(args.gaussian_chunk_size),
            "--max-raster-candidate-pairs", str(args.max_raster_candidate_pairs),
        ]
    )


def _model_from_initial_identity(identity: Mapping[str, object], device: torch.device) -> RadarSplatModel:
    model = _mapping(identity["model"], "model identity")
    optimization = _mapping(identity["optimization"], "optimization identity")
    return RadarSplatModel.random_scene(
        num_gaussians=int(model["init_num_gaussians"]),
        extent=float(model["init_extent_m"]),
        planar_initialization=True,
        seed=int(optimization["seed"]),
        initial_scale=float(model["init_scale_m"]),
        initial_opacity=float(model["init_opacity"]),
        initial_noise_probability=float(model["init_noise_probability"]),
        sh_degree=int(model["sh_degree"]),
    ).to(device)


def _active_sh_degree(identity: Mapping[str, object], step: int) -> int:
    model = _mapping(identity["model"], "model identity")
    interval = int(model["sh_degree_interval"])
    maximum = int(model["sh_degree"])
    if interval < 1 or maximum < 0 or step < 1:
        raise ValueError("RadarSplat B787 small-fit has an invalid SH schedule")
    return min((step - 1) // interval, maximum)


def _fixed_metrics(
    model: RadarSplatModel,
    cache,
    effects: RadarSplatEffects,
    evaluation_args: argparse.Namespace,
    device: torch.device,
    active_sh_degree: int,
) -> dict[str, dict[str, float | int]]:
    return {
        role: dict(
            trainer.evaluate_native_power(
                model,
                cache,
                effects,
                evaluation_args,
                device,
                active_sh_degree,
                role=role,
            )
        )
        for role in ("train", "validation")
    }


def _zero_reference(metrics: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for role, row in metrics.items():
        target_energy = _finite_number(row.get("target_energy_native_power"), f"{role} target energy", nonnegative=True)
        if target_energy <= 0.0:
            raise ValueError(f"RadarSplat B787 small-fit {role} has a zero native-power denominator")
        result[role] = {
            "relative_mse_native_power": 1.0,
            "relative_l2_native_power": 1.0,
            "target_energy_native_power": target_energy,
            "bins": int(row["bins"]),
            "views": int(row["views"]),
        }
    return result


def _parse_training_trace(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"RadarSplat B787 small-fit train log is missing: {path}")
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            continue
        try:
            candidate = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, Mapping) and "step" in candidate and "gradient_l1" in candidate and "update_l1" in candidate:
            records.append(dict(candidate))
    expected_steps = list(range(1, EXPECTED_STEPS + 1))
    observed_steps = [record.get("step") for record in records]
    if observed_steps != expected_steps:
        raise ValueError("RadarSplat B787 small-fit train log lacks exactly one complete 20-step trace")
    for record in records:
        for name in ("gradient_l1", "update_l1"):
            values = _mapping(record[name], f"step {record['step']} {name}")
            numeric = [
                _finite_number(value, f"step {record['step']} {name}.{parameter}", nonnegative=True)
                for parameter, value in values.items()
            ]
            if not numeric or not any(value > 0.0 for value in numeric):
                raise ValueError(f"RadarSplat B787 small-fit step {record['step']} has no non-zero {name}")
        _finite_number(record.get("seconds"), f"step {record['step']} seconds", nonnegative=True)
        _positive_int(record.get("process_peak_rss_kib"), f"step {record['step']} process peak RSS")
        _positive_int(
            record.get("cuda_max_memory_allocated_bytes"),
            f"step {record['step']} CUDA allocated peak",
        )
        _positive_int(
            record.get("cuda_max_memory_reserved_bytes"),
            f"step {record['step']} CUDA reserved peak",
        )
    return records


def _state_change(initial: Mapping[str, torch.Tensor], final: Mapping[str, torch.Tensor]) -> dict[str, object]:
    if set(initial) != set(final):
        raise ValueError("RadarSplat B787 small-fit initial and final parameter names differ")
    per_parameter: dict[str, dict[str, object]] = {}
    changed = False
    for name, before in initial.items():
        after = final[name]
        same_shape = before.shape == after.shape and before.dtype == after.dtype
        equal = same_shape and torch.equal(before.detach().cpu(), after.detach().cpu())
        changed = changed or not equal
        row: dict[str, object] = {
            "initial_shape": list(before.shape),
            "final_shape": list(after.shape),
            "dtype": str(after.dtype),
            "changed": not equal,
        }
        if same_shape:
            row["l1_change"] = float((after.detach().cpu() - before.detach().cpu()).abs().sum())
        per_parameter[name] = row
    if not changed:
        raise ValueError("RadarSplat B787 small-fit final model is identical to its deterministic initialization")
    return {"any_parameter_changed": changed, "parameters": per_parameter}


def _read_resource_markers(
    launcher_log_root: Path,
    *,
    filename: str,
    prefix: str,
    label: str,
) -> list[dict[str, object]]:
    if not launcher_log_root.is_dir():
        raise FileNotFoundError(f"RadarSplat B787 small-fit launcher-log root is missing: {launcher_log_root}")
    records: list[dict[str, object]] = []
    for path in sorted(launcher_log_root.glob(f"*/{filename}")):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith(prefix):
                continue
            try:
                payload = json.loads(line[len(prefix):])
            except json.JSONDecodeError as exc:
                raise ValueError(f"RadarSplat B787 small-fit {label} resource record is not JSON: {path}") from exc
            records.append({"source": str(path), "payload": dict(_mapping(payload, f"{label} resource payload"))})
    if not records:
        raise ValueError(f"RadarSplat B787 small-fit lacks a durable {label} resource record")
    return records


def _read_resource_markers_from_log(
    path: Path,
    *,
    prefix: str,
    label: str,
) -> list[dict[str, object]]:
    """Read a single durable launcher log without accepting an arbitrary path."""

    if not path.is_file():
        raise FileNotFoundError(f"RadarSplat B787 small-fit {label} log is missing: {path}")
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(prefix):
            continue
        try:
            payload = json.loads(line[len(prefix):])
        except json.JSONDecodeError as exc:
            raise ValueError(f"RadarSplat B787 small-fit {label} resource record is not JSON: {path}") from exc
        records.append({"source": str(path), "payload": dict(_mapping(payload, f"{label} resource payload"))})
    if not records:
        raise ValueError(f"RadarSplat B787 small-fit lacks a durable {label} resource record")
    return records


def _completion_resume_log_path(launcher_log_root: Path, completion_resume_log: Path) -> Path:
    """Bind no-update evidence to one immutable launcher attempt directory."""

    root = launcher_log_root.resolve()
    log = completion_resume_log.resolve()
    if log.name != "completion_resume.log" or log.parent.parent != root:
        raise ValueError(
            "RadarSplat B787 small-fit completion-resume evidence must be its launcher_logs/<jobid>/completion_resume.log"
        )
    return log


def _phase_peak(label: str, records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not records:
        raise ValueError(f"RadarSplat B787 small-fit lacks {label} resource evidence")
    process_values: list[int] = []
    allocated_values: list[int] = []
    reserved_values: list[int] = []
    sources: list[str] = []
    for record in records:
        payload = _mapping(record.get("payload"), f"{label} resource payload")
        phase = payload.get("phase")
        if not isinstance(phase, str) or not phase:
            raise ValueError(f"RadarSplat B787 small-fit {label} resource record lacks a phase")
        process_values.append(_positive_int(payload.get("process_peak_rss_kib"), f"{label} process peak RSS"))
        allocated_values.append(
            _positive_int(payload.get("cuda_max_memory_allocated_bytes"), f"{label} CUDA allocated peak")
        )
        reserved_values.append(
            _positive_int(payload.get("cuda_max_memory_reserved_bytes"), f"{label} CUDA reserved peak")
        )
        source = record.get("source")
        if not isinstance(source, str):
            raise ValueError(f"RadarSplat B787 small-fit {label} resource record lacks its source")
        sources.append(source)
    return {
        "resource_records": len(records),
        "sources": sources,
        "process_peak_rss_kib": max(process_values),
        "torch_peak_allocated_bytes": max(allocated_values),
        "torch_peak_reserved_bytes": max(reserved_values),
    }


def _phase_within_limits(
    label: str,
    phase: Mapping[str, object],
    *,
    host_limit_kib: int,
    gpu_limit_bytes: int,
) -> dict[str, object]:
    process_peak = _positive_int(phase.get("process_peak_rss_kib"), f"{label} process peak RSS")
    allocated_peak = _positive_int(phase.get("torch_peak_allocated_bytes"), f"{label} CUDA allocated peak")
    reserved_peak = _positive_int(phase.get("torch_peak_reserved_bytes"), f"{label} CUDA reserved peak")
    if process_peak >= int(0.80 * host_limit_kib):
        raise RuntimeError(f"RadarSplat B787 small-fit {label} process RSS reached the 80% host-memory gate")
    if allocated_peak >= int(0.80 * gpu_limit_bytes) or reserved_peak >= int(0.80 * gpu_limit_bytes):
        raise RuntimeError(f"RadarSplat B787 small-fit {label} Torch GPU memory reached the 80% allocation gate")
    return {
        **dict(phase),
        "process_rss_fraction": float(process_peak) / float(host_limit_kib),
        "torch_peak_allocated_fraction": float(allocated_peak) / float(gpu_limit_bytes),
        "torch_peak_reserved_fraction": float(reserved_peak) / float(gpu_limit_bytes),
    }


def _resource_report(args: argparse.Namespace, trace: Sequence[Mapping[str, object]], device: torch.device) -> dict[str, object]:
    if args.host_limit_gib < 1 or args.gpu_total_mib < 1:
        raise ValueError("RadarSplat B787 small-fit resource inputs must be positive")
    if device.type != "cuda":
        raise RuntimeError("RadarSplat B787 small-fit postflight requires the allocated CUDA device")
    launcher_log_root = Path(args.launcher_log_root).resolve()
    prepare_records = _read_resource_markers(
        launcher_log_root,
        filename="prepare.log",
        prefix=PREPARE_RESOURCE_PREFIX,
        label="preparation",
    )
    for record in prepare_records:
        payload = _mapping(record["payload"], "preparation resource payload")
        if payload.get("phase") != "prepare":
            raise ValueError("RadarSplat B787 small-fit preparation resource record has the wrong phase")
        _nonnegative_int(payload.get("targets_newly_materialized"), "preparation newly materialized target count")
        _nonnegative_int(payload.get("targets_reused"), "preparation reused target count")
    if not any(
        _nonnegative_int(
            _mapping(record["payload"], "preparation resource payload").get("targets_newly_materialized"),
            "preparation newly materialized target count",
        )
        == EXPECTED_TRAIN_VIEWS + EXPECTED_VALIDATION_VIEWS
        for record in prepare_records
    ):
        raise ValueError("RadarSplat B787 small-fit lacks resource evidence for materializing all 12 approved targets")
    train_phase_records = _read_resource_markers(
        launcher_log_root,
        filename="fit_stdout.log",
        prefix=TRAIN_RESOURCE_PREFIX,
        label="training lifecycle",
    )
    completion_resume_log = _completion_resume_log_path(
        launcher_log_root, Path(args.completion_resume_log)
    )
    completion_resume_records = _read_resource_markers_from_log(
        completion_resume_log,
        prefix=TRAIN_RESOURCE_PREFIX,
        label="completion resume",
    )
    if len(completion_resume_records) != 1:
        raise ValueError("RadarSplat B787 small-fit completion resume must emit exactly one resource record")
    completion_payload = _mapping(completion_resume_records[0]["payload"], "completion resume resource payload")
    if completion_payload.get("phase") != "completion_resume" or _nonnegative_int(
        completion_payload.get("optimizer_updates_this_invocation"),
        "completion resume optimizer updates",
    ) != 0:
        raise ValueError("RadarSplat B787 small-fit completion resume did not prove its no-update resource path")

    actual_updates = 0
    saw_finalization = False
    actual_update_records: list[dict[str, object]] = []
    fit_stdout_completion_records: list[dict[str, object]] = []
    for record in train_phase_records:
        payload = _mapping(record["payload"], "training lifecycle resource payload")
        phase = payload.get("phase")
        updates = _nonnegative_int(
            payload.get("optimizer_updates_this_invocation"),
            "training lifecycle optimizer updates",
        )
        if phase == "clean_interruption":
            if updates < 1:
                raise ValueError("RadarSplat B787 small-fit clean interruption has no completed optimizer update")
            actual_updates += updates
            actual_update_records.append(record)
        elif phase == "fit_finalization":
            saw_finalization = True
            actual_updates += updates
            actual_update_records.append(record)
        elif phase == "completion_resume":
            if updates != 0:
                raise ValueError("RadarSplat B787 small-fit completion resume executed an optimizer update")
            fit_stdout_completion_records.append(record)
        else:
            raise ValueError("RadarSplat B787 small-fit has an unexpected training lifecycle resource phase")
    if not saw_finalization or actual_updates != EXPECTED_STEPS:
        raise ValueError(
            "RadarSplat B787 small-fit lacks exact phase-complete resource evidence for all 20 optimizer updates"
        )

    trace_phase = {
        "resource_records": len(trace),
        "sources": [str(Path(args.train_log))],
        "process_peak_rss_kib": max(
            _positive_int(record.get("process_peak_rss_kib"), "trace process peak RSS") for record in trace
        ),
        "torch_peak_allocated_bytes": max(
            _positive_int(record.get("cuda_max_memory_allocated_bytes"), "trace CUDA allocated peak") for record in trace
        ),
        "torch_peak_reserved_bytes": max(
            _positive_int(record.get("cuda_max_memory_reserved_bytes"), "trace CUDA reserved peak") for record in trace
        ),
        "source_note": "maximum across every durable actual optimizer-update record, including earlier clean-resume attempts",
    }
    postflight_phase = {
        "resource_records": 1,
        "sources": ["current cache-only postflight process"],
        "process_peak_rss_kib": _positive_int(trainer._process_peak_rss_kib(), "postflight process peak RSS"),
        "torch_peak_allocated_bytes": _positive_int(
            torch.cuda.max_memory_allocated(device), "postflight CUDA allocated peak"
        ),
        "torch_peak_reserved_bytes": _positive_int(
            torch.cuda.max_memory_reserved(device), "postflight CUDA reserved peak"
        ),
    }
    host_limit_kib = int(args.host_limit_gib) * 1024 * 1024
    gpu_limit_bytes = int(args.gpu_total_mib) * 2**20
    phase_peaks = {
        "target_preparation": _phase_peak("target preparation", prepare_records),
        "actual_optimizer_updates": trace_phase,
        "training_finalization": {
            **_phase_peak("training lifecycle", actual_update_records),
            "optimizer_updates_across_clean_interruptions_and_finalization": actual_updates,
        },
        "no_update_completion_resume": _phase_peak(
            "completion resume", [*fit_stdout_completion_records, *completion_resume_records]
        ),
        "cache_only_postflight": postflight_phase,
    }
    checked_phases = {
        label: _phase_within_limits(
            label,
            phase,
            host_limit_kib=host_limit_kib,
            gpu_limit_bytes=gpu_limit_bytes,
        )
        for label, phase in phase_peaks.items()
    }
    overall = {
        "process_peak_rss_kib": max(
            _positive_int(phase["process_peak_rss_kib"], f"{label} process peak RSS")
            for label, phase in checked_phases.items()
        ),
        "torch_peak_allocated_bytes": max(
            _positive_int(phase["torch_peak_allocated_bytes"], f"{label} CUDA allocated peak")
            for label, phase in checked_phases.items()
        ),
        "torch_peak_reserved_bytes": max(
            _positive_int(phase["torch_peak_reserved_bytes"], f"{label} CUDA reserved peak")
            for label, phase in checked_phases.items()
        ),
    }
    overall = _phase_within_limits(
        "all completed phases",
        overall,
        host_limit_kib=host_limit_kib,
        gpu_limit_bytes=gpu_limit_bytes,
    )
    return {
        "host_limit_gib": int(args.host_limit_gib),
        "gpu_total_mib": int(args.gpu_total_mib),
        "whole_job_rss_raw": args.whole_job_rss_raw,
        "phase_peaks": checked_phases,
        "overall_peak": overall,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.gaussian_chunk_size != EXPECTED_GAUSSIAN_CHUNK_SIZE:
        raise ValueError("RadarSplat B787 small-fit postflight requires the approved 64-Gaussian renderer chunk")
    if args.max_raster_candidate_pairs != EXPECTED_MAX_RASTER_CANDIDATE_PAIRS:
        raise ValueError("RadarSplat B787 small-fit postflight requires the approved 2,000,000 raster-pair cap")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"RadarSplat B787 small-fit postflight refuses to overwrite: {output}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("RadarSplat B787 small-fit postflight requires CUDA")
    torch.cuda.reset_peak_memory_stats(device)
    signal.signal(signal.SIGTERM, _request_postflight_stop)
    signal.signal(signal.SIGINT, _request_postflight_stop)
    checkpoint_dir = Path(args.checkpoint_dir)
    expected_train_log = (checkpoint_dir / "training_trace.jsonl").resolve()
    if Path(args.train_log).resolve() != expected_train_log:
        raise ValueError("RadarSplat B787 small-fit postflight must read its own checkpoint training_trace.jsonl")
    completion_resume_log = _completion_resume_log_path(
        Path(args.launcher_log_root), Path(args.completion_resume_log)
    )
    if not completion_resume_log.is_file() or COMPLETION_RESUME_MARKER not in completion_resume_log.read_text(encoding="utf-8").splitlines():
        raise ValueError("RadarSplat B787 small-fit postflight lacks the verified same-identity completion-resume marker")
    cache = load_cache(args.cache_root)
    final_path = checkpoint_dir / "checkpoint_final.pt"
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    geometry_path = checkpoint_dir / "gaussian_occupancy_geometry.npz"
    summary_path = checkpoint_dir / "summary.json"
    final_checkpoint = trainer._load_checkpoint(final_path, device)
    if final_checkpoint.get("complete") is not True or final_checkpoint.get("finalization_pending") is not False:
        raise ValueError("RadarSplat B787 small-fit final checkpoint is not complete")
    identity = _require_exact_smallfit_identity(cache, final_checkpoint)
    latest_checkpoint = trainer._load_checkpoint(latest_path, device)
    if not trainer._directly_equal(final_checkpoint, latest_checkpoint):
        raise ValueError("RadarSplat B787 small-fit completion resume changed the latest checkpoint")
    step = int(final_checkpoint.get("step", -1))
    if step != EXPECTED_STEPS:
        raise ValueError("RadarSplat B787 small-fit final checkpoint has the wrong update count")
    history = final_checkpoint.get("history")
    if not isinstance(history, list) or [row.get("step") for row in history if isinstance(row, Mapping)] != [5, 10, 15, 20]:
        raise ValueError("RadarSplat B787 small-fit lacks the expected validation milestones")
    initial_model = _model_from_initial_identity(identity, device)
    state = _mapping(final_checkpoint.get("model_state_dict"), "final model state")
    final_model = model_from_checkpoint_state(state, device=device, seed=int(_mapping(identity["optimization"], "optimization identity")["seed"]))
    active_sh_degree = _active_sh_degree(identity, step)
    evaluation_args = _evaluation_args(args)
    effects = RadarSplatEffects.b787_clean()
    initial_metrics = _fixed_metrics(initial_model, cache, effects, evaluation_args, device, active_sh_degree)
    final_metrics = _fixed_metrics(final_model, cache, effects, evaluation_args, device, active_sh_degree)
    zero_metrics = _zero_reference(initial_metrics)
    trace = _parse_training_trace(expected_train_log)
    final_last_train = _mapping(final_checkpoint.get("last_train"), "final checkpoint last_train")
    if not trainer._directly_equal(trace[-1], final_last_train):
        raise ValueError("RadarSplat B787 small-fit trace final update does not match checkpoint_final.pt")
    trainer._validate_existing_geometry(
        geometry_path,
        model=final_model,
        active_sh_degree=active_sh_degree,
        train_peak_power=cache.train_peak_power,
    )
    trainer._validate_existing_summary(
        summary_path,
        best_validation=float(final_checkpoint["best_validation_rel_mse"]),
        history=history,
        geometry_path=geometry_path,
    )
    with np.load(geometry_path, allow_pickle=False) as geometry:
        occupancy = np.asarray(geometry["occupancy"], dtype=np.float64)
        means = np.asarray(geometry["means"], dtype=np.float64)
    if means.ndim != 2 or means.shape[1] != 3 or occupancy.shape != (means.shape[0],):
        raise ValueError("RadarSplat B787 small-fit Gaussian geometry export is malformed")
    if not np.isfinite(means).all() or not np.isfinite(occupancy).all() or np.any(occupancy < 0.0) or np.any(occupancy > 1.0):
        raise ValueError("RadarSplat B787 small-fit Gaussian occupancy readout is invalid")
    report = {
        "schema": "rift_radarsplat_b7873200_smallfit_postflight_v1",
        "scope": "bounded engineering lifecycle only; no coherent signal, mesh, NVS, reconstruction, convergence, or baseline-comparison claim",
        "production_clearance": False,
        "observable": "native real polar power: sum_elevation(abs(matched_filter_complex)**2)",
        "roles": {
            "train_indices": list(cache.train_indices),
            "validation_indices": list(cache.validation_indices),
            "selected_counts": {"train": len(cache.train_indices), "validation": len(cache.validation_indices)},
            "all_coordinates": "16 Tx x 16 Rx x 600 frequencies per selected view",
            "test_or_parent_unused_access": False,
        },
        "fit": {
            "updates": step,
            "initial_gaussians": EXPECTED_INITIAL_GAUSSIANS,
            "final_gaussians": int(means.shape[0]),
            "active_sh_degree_at_final": active_sh_degree,
            "fixed_native_power_relative_errors": {
                "initial": initial_metrics,
                "final": final_metrics,
                "zero_reference": zero_metrics,
            },
            "fixed_train_relative_mse_delta": float(
                final_metrics["train"]["relative_mse_native_power"]
                - initial_metrics["train"]["relative_mse_native_power"]
            ),
            "fixed_train_relative_mse_decreased": bool(
                final_metrics["train"]["relative_mse_native_power"]
                < initial_metrics["train"]["relative_mse_native_power"]
            ),
            "train_trace_seconds": [float(record["seconds"]) for record in trace],
            "train_trace_total_seconds": float(sum(float(record["seconds"]) for record in trace)),
            "evaluation_renderer_controls": {
                "gaussian_chunk_size": EXPECTED_GAUSSIAN_CHUNK_SIZE,
                "max_raster_candidate_pairs": EXPECTED_MAX_RASTER_CANDIDATE_PAIRS,
            },
            "gradient_and_update_trace_complete": True,
            "parameter_change": _state_change(initial_model.state_dict(), final_model.state_dict()),
        },
        "checkpoint_recovery": {
            "final_checkpoint": str(final_path),
            "latest_checkpoint": str(latest_path),
            "complete": True,
            "completion_resume_kept_latest_equal_to_final": True,
            "completion_resume_log": str(completion_resume_log),
            "completion_resume_marker": COMPLETION_RESUME_MARKER,
        },
        "native_geometry_readout": {
            "path": str(geometry_path),
            "representation": "Gaussian locations/scales/orientations with occupancy; not a mesh",
            "gaussian_count": int(means.shape[0]),
            "occupancy_min": float(np.min(occupancy)),
            "occupancy_mean": float(np.mean(occupancy)),
            "occupancy_max": float(np.max(occupancy)),
        },
        "resources": _resource_report(args, trace, device),
    }
    atomic_write_json(output, report)
    print("RADARSPLAT_B7873200_SMALLFIT_POSTFLIGHT_PASS", flush=True)


if __name__ == "__main__":
    main()
