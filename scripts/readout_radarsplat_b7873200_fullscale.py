#!/usr/bin/env python
"""Read out one completed full B787 RadarSplat native-power fit.

This is a cache-only reader.  It evaluates the complete train and validation
roles at the selected best checkpoint, emits same-domain zero-reference
metrics, and retains final/latest completion state, terminal metrics, and
terminal Gaussian export as diagnostics.  The reserved test role is reported
as sealed and is never opened by this program.
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_radarsplat as trainer
from rift.radarsplat_b7873200 import RadarSplatEffects
from rift.radarsplat_b7873200_adapter import (
    export_gaussian_occupancy_geometry,
    model_from_checkpoint_state,
)
from rift.radarsplat_b7873200_protocol import (
    B787_3200_CANONICAL_MANIFEST_PATH,
    B787_3200_CANONICAL_NPZ_PATH,
    atomic_write_json,
    load_cache,
)


PREPARE_PREFIX = "RADARSPLAT_B7873200_PREPARE_RESOURCE_JSON="
TRAIN_PREFIX = "RADARSPLAT_B7873200_TRAIN_RESOURCE_JSON="
READOUT_PREFIX = "RADARSPLAT_B7873200_READOUT_RESOURCE_JSON="
CLEAN_STOP_MARKER = "RADARSPLAT_B7873200_READOUT_CLEAN_STOP"
_READOUT_DEVICE: torch.device | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", help="optional RIFT dataset object; otherwise use the bound cache object")
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parents[1] / "data/RIFT_dataset")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--launcher-log-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=480000)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--validation-every", type=int, default=600)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--init-num-gaussians", type=int, default=2048)
    parser.add_argument("--init-extent-m", type=float, default=0.3)
    parser.add_argument("--init-scale-m", type=float, default=0.003)
    parser.add_argument("--init-opacity", type=float, default=0.1)
    parser.add_argument("--init-noise-probability", type=float, default=0.1)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--sh-degree-interval", type=int, default=600)
    parser.add_argument("--means-lr-base", type=float, default=1.6e-4)
    parser.add_argument("--scales-lr", type=float, default=5.0e-3)
    parser.add_argument("--quaternions-lr", type=float, default=1.0e-3)
    parser.add_argument("--opacity-lr", type=float, default=5.0e-2)
    parser.add_argument("--noise-probability-lr", type=float, default=5.0e-2)
    parser.add_argument("--sh0-lr", type=float, default=2.5e-3)
    parser.add_argument("--shn-lr", type=float, default=2.5e-3)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1.0e-15)
    parser.add_argument("--ssim-weight", type=float, default=0.2)
    parser.add_argument("--occupancy-weight", type=float, default=5.0)
    parser.add_argument("--max-size-weight", type=float, default=100.0)
    parser.add_argument("--opacity-noise-weight", type=float, default=100.0)
    parser.add_argument("--max-scale-m", type=float, default=0.006)
    parser.add_argument("--occupancy-threshold", type=float, default=0.001)
    parser.add_argument("--prune-opacity", type=float, default=0.0005)
    parser.add_argument("--prune-every", type=int, default=100)
    parser.add_argument("--gaussian-chunk-size", type=int, default=64)
    parser.add_argument("--max-raster-candidate-pairs", type=int, default=2_000_000)
    parser.add_argument("--host-limit-gib", type=int, default=32)
    parser.add_argument("--gpu-total-mib", type=int, required=True)
    return parser.parse_args(argv)


def _emit_readout_resource(phase: str) -> None:
    device = _READOUT_DEVICE
    payload: dict[str, object] = {
        "phase": phase,
        "process_peak_rss_kib": int(trainer._process_peak_rss_kib() or 0),
        "cuda_max_memory_allocated_bytes": 0,
        "cuda_max_memory_reserved_bytes": 0,
    }
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        payload["cuda_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        payload["cuda_max_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    print(READOUT_PREFIX + json.dumps(payload, sort_keys=True), flush=True)


def _stop(_signum: int, _frame: object) -> None:
    _emit_readout_resource("readout_clean_interruption")
    print(CLEAN_STOP_MARKER, flush=True)
    raise SystemExit(143)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"RadarSplat full-scale {label} must be a mapping")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"RadarSplat full-scale {label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 1.0 or not number.is_integer():
        raise ValueError(f"RadarSplat full-scale {label} must be a positive integer")
    return int(number)


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"RadarSplat full-scale {label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or not number.is_integer():
        raise ValueError(f"RadarSplat full-scale {label} must be a finite non-negative integer")
    return int(number)


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"RadarSplat full-scale {label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"RadarSplat full-scale {label} must be finite and non-negative")
    return number


def _read_resource_markers(root: Path, filename: str, prefix: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(root.glob(f"*/{filename}")):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith(prefix):
                continue
            payload = json.loads(line[len(prefix):])
            records.append({"source": str(path), "payload": dict(_mapping(payload, "resource payload"))})
    return records


def _resource_payload(record: Mapping[str, object], label: str) -> Mapping[str, object]:
    payload = _mapping(record.get("payload"), label)
    _nonnegative_int(payload.get("process_peak_rss_kib"), f"{label} process peak RSS")
    _nonnegative_int(payload.get("cuda_max_memory_allocated_bytes"), f"{label} Torch allocated peak")
    _nonnegative_int(payload.get("cuda_max_memory_reserved_bytes"), f"{label} Torch reserved peak")
    return payload


def _trace(path: Path, steps: int) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"RadarSplat full-scale training trace is missing: {path}")
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text.startswith("{"):
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping) and "step" in value and "gradient_l1" in value and "update_l1" in value:
            rows.append(dict(value))
    if [row.get("step") for row in rows] != list(range(1, steps + 1)):
        raise ValueError("RadarSplat full-scale trace must contain exactly one durable row for every optimizer update")
    for row in rows:
        _finite(row.get("seconds"), f"step {row['step']} seconds")
        _positive_int(row.get("process_peak_rss_kib"), f"step {row['step']} process RSS")
        _positive_int(row.get("cuda_max_memory_allocated_bytes"), f"step {row['step']} Torch allocation")
        _positive_int(row.get("cuda_max_memory_reserved_bytes"), f"step {row['step']} Torch reservation")
        for name in ("gradient_l1", "update_l1"):
            values = _mapping(row.get(name), f"step {row['step']} {name}")
            if not values or not any(_finite(value, f"step {row['step']} {name}") > 0.0 for value in values.values()):
                raise ValueError(f"RadarSplat full-scale step {row['step']} has no nonzero {name}")
    return rows


def _expected_validation_steps(steps: int, validation_every: int) -> list[int]:
    if steps < 1 or validation_every < 1:
        raise ValueError("RadarSplat full-scale validation cadence must be positive")
    milestones = list(range(validation_every, steps + 1, validation_every))
    if not milestones or milestones[-1] != steps:
        milestones.append(steps)
    return milestones


def _active_sh_degree(step: int, interval: int, maximum: int) -> int:
    """Derive the active SH degree using the trainer's completed-step rule."""

    step_value = _positive_int(step, "checkpoint step")
    interval_value = _positive_int(interval, "SH activation interval")
    maximum_value = _nonnegative_int(maximum, "maximum SH degree")
    return min((max(step_value, 1) - 1) // interval_value, maximum_value)


def _validation_row(row: object, label: str) -> tuple[int, float, int]:
    value = _mapping(row, label)
    step = _positive_int(value.get("step"), f"{label} step")
    metric = _finite(value.get("relative_mse_native_power"), f"{label} relative MSE")
    active_sh_degree = _nonnegative_int(value.get("active_sh_degree"), f"{label} active SH degree")
    return step, metric, active_sh_degree


def _select_best_validation_row(
    history: object,
) -> tuple[int, Mapping[str, object], float, int]:
    """Select the earliest exact minimum from the completed validation history."""

    if not isinstance(history, list) or not history:
        raise ValueError("RadarSplat full-scale validation history must be a nonempty list")
    selected_index: int | None = None
    selected_row: Mapping[str, object] | None = None
    selected_metric: float | None = None
    selected_step = 0
    for index, row in enumerate(history):
        step, metric, _active_sh_degree = _validation_row(row, f"validation history row {index}")
        if selected_metric is None or metric < selected_metric:
            selected_index = index
            selected_row = _mapping(row, f"validation history row {index}")
            selected_metric = metric
            selected_step = step
    assert selected_index is not None and selected_row is not None and selected_metric is not None
    return selected_index, selected_row, selected_metric, selected_step


def _evaluate_role(
    model: torch.nn.Module,
    cache,
    effects: RadarSplatEffects,
    evaluation_args: argparse.Namespace,
    device: torch.device,
    active_sh_degree: int,
    *,
    role: str,
    expected_views: int,
) -> dict[str, object]:
    metrics = dict(
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
    views = _positive_int(metrics.get("views"), f"{role} metric views")
    if views != expected_views:
        raise ValueError(
            f"RadarSplat full-scale {role} metrics cover {views} views rather than {expected_views}"
        )
    metrics["views_complete"] = views
    metrics["views_total"] = expected_views
    return metrics


def _running_best_at_step(history: object, endpoint: int) -> dict[str, object]:
    """Return the running-best validation metric through one scheduled endpoint."""

    endpoint_value = _positive_int(endpoint, "budget endpoint")
    if not isinstance(history, list):
        raise ValueError("RadarSplat full-scale validation history must be a list")
    prefix: list[Mapping[str, object]] = []
    endpoint_seen = False
    for index, row in enumerate(history):
        step, _metric, _active_sh_degree = _validation_row(row, f"validation history row {index}")
        if step <= endpoint_value:
            prefix.append(_mapping(row, f"validation history row {index}"))
        if step == endpoint_value:
            endpoint_seen = True
    if not endpoint_seen:
        raise ValueError(f"RadarSplat full-scale history lacks required budget endpoint {endpoint_value}")
    _selected_index, _selected_row, selected_metric, selected_step = _select_best_validation_row(prefix)
    return {
        "step": endpoint_value,
        "running_best_validation_relative_mse_native_power": selected_metric,
        "selected_validation_step": selected_step,
    }


def _learning_curve_budget_status(history: object) -> dict[str, object]:
    at_384000 = _running_best_at_step(history, 384000)
    at_480000 = _running_best_at_step(history, 480000)
    best_384000 = float(at_384000["running_best_validation_relative_mse_native_power"])
    best_480000 = float(at_480000["running_best_validation_relative_mse_native_power"])
    if best_384000 > 0.0:
        relative_improvement = (best_384000 - best_480000) / best_384000
    elif best_480000 == 0.0:
        relative_improvement = 0.0
    else:
        relative_improvement = 0.0
    if not math.isfinite(relative_improvement):
        raise ValueError("RadarSplat full-scale budget improvement is non-finite")
    threshold = 0.01
    return {
        "step_384000": at_384000,
        "step_480000": at_480000,
        "relative_improvement": relative_improvement,
        "threshold": threshold,
        "label": (
            "still improving at budget"
            if relative_improvement >= threshold
            else "<1% best-validation improvement over the last 30 passes"
        ),
        "descriptive_only": True,
        "automatic_extension": False,
    }


def _trainer_args(args: argparse.Namespace) -> argparse.Namespace:
    values = [
        "--cache-root", args.cache_root, "--checkpoint-dir", args.checkpoint_dir,
        "--device", args.device, "--seed", str(args.seed), "--steps", str(args.steps),
        "--validation-every", str(args.validation_every), "--checkpoint-every", str(args.checkpoint_every),
        "--log-every", str(args.log_every), "--resume",
        "--init-num-gaussians", str(args.init_num_gaussians), "--init-extent-m", str(args.init_extent_m),
        "--init-scale-m", str(args.init_scale_m), "--init-opacity", str(args.init_opacity),
        "--init-noise-probability", str(args.init_noise_probability), "--sh-degree", str(args.sh_degree),
        "--sh-degree-interval", str(args.sh_degree_interval), "--means-lr-base", str(args.means_lr_base),
        "--scales-lr", str(args.scales_lr), "--quaternions-lr", str(args.quaternions_lr),
        "--opacity-lr", str(args.opacity_lr), "--noise-probability-lr", str(args.noise_probability_lr),
        "--sh0-lr", str(args.sh0_lr), "--shn-lr", str(args.shn_lr), "--adam-beta1", str(args.adam_beta1),
        "--adam-beta2", str(args.adam_beta2), "--adam-eps", str(args.adam_eps),
        "--ssim-weight", str(args.ssim_weight), "--occupancy-weight", str(args.occupancy_weight),
        "--max-size-weight", str(args.max_size_weight), "--opacity-noise-weight", str(args.opacity_noise_weight),
        "--max-scale-m", str(args.max_scale_m), "--occupancy-threshold", str(args.occupancy_threshold),
        "--prune-opacity", str(args.prune_opacity), "--prune-every", str(args.prune_every),
        "--gaussian-chunk-size", str(args.gaussian_chunk_size),
        "--max-raster-candidate-pairs", str(args.max_raster_candidate_pairs),
    ]
    return trainer.parse_args(values)


def _zero_reference(metrics: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for role, row in metrics.items():
        energy = _finite(row.get("target_energy_native_power"), f"{role} target energy")
        if energy <= 0.0:
            raise ValueError(f"RadarSplat full-scale {role} has no native-power energy")
        result[role] = {
            "relative_mse_native_power": 1.0,
            "relative_l2_native_power": 1.0,
            "target_energy_native_power": energy,
            "bins": _positive_int(row.get("bins"), f"{role} bins"),
            "views": _positive_int(row.get("views"), f"{role} views"),
        }
    return result


def _phase_peak(
    records: Sequence[Mapping[str, object]], label: str, *, require_process_peak: bool = True
) -> dict[str, object]:
    if not records:
        raise ValueError(f"RadarSplat full-scale lacks {label} resource evidence")
    payloads = [_resource_payload(record, label) for record in records]
    result = {
        "records": len(payloads),
        "sources": [str(record["source"]) for record in records],
        "process_peak_rss_kib": max(_nonnegative_int(p["process_peak_rss_kib"], label) for p in payloads),
        "torch_peak_allocated_bytes": max(_nonnegative_int(p["cuda_max_memory_allocated_bytes"], label) for p in payloads),
        "torch_peak_reserved_bytes": max(_nonnegative_int(p["cuda_max_memory_reserved_bytes"], label) for p in payloads),
    }
    if require_process_peak and result["process_peak_rss_kib"] <= 0:
        raise ValueError(f"RadarSplat full-scale {label} lacks actual process resource evidence")
    return result


def _check_limits(phases: Mapping[str, Mapping[str, object]], args: argparse.Namespace) -> dict[str, object]:
    host_limit = args.host_limit_gib * 1024 * 1024
    gpu_limit = args.gpu_total_mib * 2**20
    checked: dict[str, object] = {}
    for label, phase in phases.items():
        process = _nonnegative_int(phase["process_peak_rss_kib"], f"{label} process")
        allocated = _nonnegative_int(phase["torch_peak_allocated_bytes"], f"{label} allocation")
        reserved = _nonnegative_int(phase["torch_peak_reserved_bytes"], f"{label} reservation")
        if process >= int(0.8 * host_limit) or allocated >= int(0.8 * gpu_limit) or reserved >= int(0.8 * gpu_limit):
            raise RuntimeError(f"RadarSplat full-scale {label} exceeded the 80% resource gate")
        checked[label] = {
            **dict(phase),
            "process_rss_fraction": process / host_limit,
            "torch_peak_allocated_fraction": allocated / gpu_limit,
            "torch_peak_reserved_fraction": reserved / gpu_limit,
        }
    return {
        "host_limit_gib": args.host_limit_gib,
        "gpu_total_mib": args.gpu_total_mib,
        "phase_peaks": checked,
    }


def collection_readout_provenance(args):
    """Read identity metadata before load_cache can scan response-derived targets."""
    from rift.rift_dataset import (collection_contract, resolve_object_inputs,
                                   load_object_contract, validate_checkpoint_object, object_identity)
    from rift.radarsplat_b7873200_protocol import RECIPE_FILENAME, STATS_FILENAME
    with (Path(args.cache_root) / RECIPE_FILENAME).open() as handle:
        recipe = json.load(handle)
    identity = collection_contract(recipe.get("sealed_protocol_identity", {}))
    if identity is None:
        if args.object is not None:
            raise ValueError("A named collection object cannot use a legacy B787 cache")
        return {"dataset_npz_path": B787_3200_CANONICAL_NPZ_PATH,
                "role_manifest_path": B787_3200_CANONICAL_MANIFEST_PATH}
    name = args.object or identity["dataset_identity"]["object_id"]
    npz_path, manifest_path = resolve_object_inputs(object_name=name, dataset_root=args.dataset_root)
    _, expected = load_object_contract(npz_path, manifest_path)
    validate_checkpoint_object(object_identity(name), expected)
    validate_checkpoint_object(recipe, expected)
    with (Path(args.cache_root) / STATS_FILENAME).open() as handle:
        validate_checkpoint_object(json.load(handle), expected)
    for name in ("checkpoint_final.pt", "checkpoint_best.pt", "checkpoint_latest.pt"):
        saved = torch.load(Path(args.checkpoint_dir) / name, map_location="cpu", weights_only=False)
        validate_checkpoint_object(saved, expected)
        del saved
    return {"dataset_npz_path": str(npz_path), "role_manifest_path": str(manifest_path),
            "dataset_identity": expected["dataset_identity"]}


def main(argv: Sequence[str] | None = None) -> None:
    global _READOUT_DEVICE
    args = parse_args(argv)
    if (
        args.epochs != 150
        or args.init_num_gaussians != 2048
        or args.validation_every != 600
        or args.sh_degree_interval != 600
    ):
        raise ValueError("RadarSplat full-scale readout received a non-candidate recipe")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("RadarSplat full-scale readout requires the allocated CUDA device")
    _READOUT_DEVICE = torch.device(args.device)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"RadarSplat full-scale readout refuses to overwrite: {output}")
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    torch.cuda.reset_peak_memory_stats()

    source_provenance = collection_readout_provenance(args)
    cache = load_cache(args.cache_root)
    if cache.is_development_subset or len(cache.train_indices) != 3200 or len(cache.validation_indices) != 1000:
        raise ValueError("RadarSplat full-scale readout requires the complete 3200/1000 cache")
    role_manifest_name = cache.identity.get("role_manifest_name")
    if not isinstance(role_manifest_name, str) or not role_manifest_name:
        raise ValueError("RadarSplat full-scale cache identity lacks role_manifest_name")
    target_spec = _mapping(cache.recipe.get("target_spec"), "cache target_spec")
    target_grid = _mapping(target_spec.get("grid"), "cache target grid")
    target_matched_filter = _mapping(target_spec.get("matched_filter"), "cache matched_filter")
    target_normalization = _mapping(target_spec.get("normalization"), "cache normalization")
    if target_spec.get("projection") != "sum_elevation(abs(matched_filter_complex)**2) -> [azimuth,range]":
        raise ValueError("RadarSplat full-scale cache target projection is not the canonical native projection")
    if (
        target_normalization.get("mode") != "linear_peak"
        or target_normalization.get("fit_split") != "train"
        or target_normalization.get("clip") is not False
    ):
        raise ValueError("RadarSplat full-scale cache normalization is not the canonical train-only linear peak")
    train_peak_power = float(cache.train_peak_power)
    if not math.isfinite(train_peak_power) or train_peak_power <= 0.0:
        raise ValueError("RadarSplat full-scale cache train_peak_power must be finite and positive")
    expected_steps = args.epochs * len(cache.train_indices)
    if args.steps != expected_steps:
        raise ValueError(
            f"RadarSplat full-scale readout requires {args.epochs} complete passes, "
            f"which is {expected_steps} updates for this cache; got {args.steps}"
        )
    final_path = Path(args.checkpoint_dir) / "checkpoint_final.pt"
    best_path = Path(args.checkpoint_dir) / "checkpoint_best.pt"
    latest_path = Path(args.checkpoint_dir) / "checkpoint_latest.pt"
    geometry_path = Path(args.checkpoint_dir) / "gaussian_occupancy_geometry.npz"
    selected_geometry_path = Path(args.checkpoint_dir) / "gaussian_occupancy_geometry_selected.npz"
    summary_path = Path(args.checkpoint_dir) / "summary.json"
    final = trainer._load_checkpoint(final_path, torch.device(args.device))
    best = trainer._load_checkpoint(best_path, torch.device(args.device))
    latest = trainer._load_checkpoint(latest_path, torch.device(args.device))
    if final.get("complete") is not True or final.get("finalization_pending") is not False:
        raise ValueError("RadarSplat full-scale final checkpoint is not complete")
    if not trainer._directly_equal(final, latest):
        raise ValueError("RadarSplat full-scale latest checkpoint is not equal to final")
    if final.get("step") != args.steps:
        raise ValueError("RadarSplat full-scale checkpoint has the wrong update count")
    history = final.get("history")
    expected_validation_steps = _expected_validation_steps(args.steps, args.validation_every)
    if not isinstance(history, list) or [row.get("step") for row in history if isinstance(row, Mapping)] != expected_validation_steps:
        raise ValueError("RadarSplat full-scale checkpoint lacks the scheduled SH-boundary and terminal validation milestones")
    selected_index, selected_row, selected_metric, selected_step = _select_best_validation_row(history)
    _selected_history_step, _selected_history_metric, selected_history_degree = _validation_row(
        selected_row, "selected validation history row"
    )
    if selected_step not in expected_validation_steps:
        raise ValueError("RadarSplat full-scale selected validation step is not scheduled")

    device = torch.device(args.device)
    effects = RadarSplatEffects.b787_clean()
    evaluation_args = _trainer_args(args)
    expected_identity = trainer.run_identity(evaluation_args, cache, effects)
    if final.get("run_identity") != expected_identity:
        raise ValueError("RadarSplat full-scale checkpoint identity disagrees with the reviewed candidate")
    if best.get("checkpoint_version") != trainer.CHECKPOINT_VERSION:
        raise ValueError("RadarSplat full-scale best checkpoint version is unsupported")
    if best.get("run_identity") != expected_identity:
        raise ValueError("RadarSplat full-scale best checkpoint identity disagrees with the reviewed candidate")
    if best.get("complete") is not False or best.get("finalization_pending") is not False:
        raise ValueError("RadarSplat full-scale best checkpoint must be a nonterminal validation state")
    best_step = _positive_int(best.get("step"), "best checkpoint step")
    if best_step != selected_step or best_step not in expected_validation_steps:
        raise ValueError("RadarSplat full-scale best checkpoint is not the selected validation milestone")
    best_metric = _finite(best.get("best_validation_rel_mse"), "best checkpoint validation RelMSE")
    final_best_metric = _finite(final.get("best_validation_rel_mse"), "final checkpoint validation RelMSE")
    if best_metric != selected_metric or final_best_metric != selected_metric:
        raise ValueError("RadarSplat full-scale best validation RelMSE disagrees with the selected history minimum")
    best_history = best.get("history")
    if not isinstance(best_history, list) or not trainer._directly_equal(
        best_history, history[: selected_index + 1]
    ):
        raise ValueError("RadarSplat full-scale best checkpoint history is not the selected final-history prefix")
    best_last_train = _mapping(best.get("last_train"), "best last_train")
    if _positive_int(best_last_train.get("step"), "best last_train step") != best_step:
        raise ValueError("RadarSplat full-scale best last_train does not match the selected checkpoint step")
    selected_active_sh_degree = _active_sh_degree(best_step, args.sh_degree_interval, args.sh_degree)
    if selected_active_sh_degree != selected_history_degree or _nonnegative_int(
        best_last_train.get("active_sh_degree"), "best last_train active SH degree"
    ) != selected_active_sh_degree:
        raise ValueError("RadarSplat full-scale selected active SH degree disagrees with trainer state")
    final_step = _positive_int(final.get("step"), "final checkpoint step")
    final_active_sh_degree = _active_sh_degree(final_step, args.sh_degree_interval, args.sh_degree)
    _terminal_history_step, _terminal_history_metric, terminal_history_degree = _validation_row(
        history[-1], "terminal validation history row"
    )
    final_last_train = _mapping(final.get("last_train"), "final last_train")
    if (
        terminal_history_degree != final_active_sh_degree
        or _nonnegative_int(final_last_train.get("active_sh_degree"), "final last_train active SH degree")
        != final_active_sh_degree
    ):
        raise ValueError("RadarSplat full-scale terminal active SH degree disagrees with trainer state")

    selected_state = _mapping(best.get("model_state_dict"), "selected model state")
    final_state = _mapping(final.get("model_state_dict"), "final model state")
    selected_model = model_from_checkpoint_state(selected_state, device=device, seed=args.seed)
    final_model = model_from_checkpoint_state(final_state, device=device, seed=args.seed)
    selected_metrics = {
        role: _evaluate_role(
            selected_model,
            cache,
            effects,
            evaluation_args,
            device,
            selected_active_sh_degree,
            role=role,
            expected_views=len(cache.train_indices if role == "train" else cache.validation_indices),
        )
        for role in ("train", "validation")
    }
    final_diagnostic_metrics = {
        role: _evaluate_role(
            final_model,
            cache,
            effects,
            evaluation_args,
            device,
            final_active_sh_degree,
            role=role,
            expected_views=len(cache.train_indices if role == "train" else cache.validation_indices),
        )
        for role in ("train", "validation")
    }
    zero = _zero_reference(selected_metrics)
    learning_curve_budget_status = _learning_curve_budget_status(history)
    trace = _trace(Path(args.checkpoint_dir) / "training_trace.jsonl", args.steps)
    if not trainer._directly_equal(trace[-1], final_last_train):
        raise ValueError("RadarSplat full-scale trace does not terminate at checkpoint_final.pt")
    trainer._validate_existing_geometry(
        geometry_path,
        model=final_model,
        active_sh_degree=final_active_sh_degree,
        train_peak_power=cache.train_peak_power,
    )
    trainer._validate_existing_summary(
        summary_path, best_validation=float(final["best_validation_rel_mse"]), history=history, geometry_path=geometry_path
    )
    if selected_geometry_path.exists():
        trainer._validate_existing_geometry(
            selected_geometry_path,
            model=selected_model,
            active_sh_degree=selected_active_sh_degree,
            train_peak_power=cache.train_peak_power,
        )
    else:
        export_gaussian_occupancy_geometry(
            selected_geometry_path,
            selected_model,
            active_sh_degree=selected_active_sh_degree,
            train_peak_power=cache.train_peak_power,
        )
        trainer._validate_existing_geometry(
            selected_geometry_path,
            model=selected_model,
            active_sh_degree=selected_active_sh_degree,
            train_peak_power=cache.train_peak_power,
        )
    with np.load(selected_geometry_path, allow_pickle=False) as selected_geometry:
        selected_means = np.asarray(selected_geometry["means"])
        selected_occupancy = np.asarray(selected_geometry["occupancy"])
    with np.load(geometry_path, allow_pickle=False) as geometry:
        final_means = np.asarray(geometry["means"])
        final_occupancy = np.asarray(geometry["occupancy"])
    for label, means, occupancy in (
        ("selected", selected_means, selected_occupancy),
        ("final", final_means, final_occupancy),
    ):
        if means.ndim != 2 or means.shape[1] != 3 or occupancy.shape != (means.shape[0],):
            raise ValueError(f"RadarSplat full-scale {label} Gaussian export has invalid shapes")
        if not np.isfinite(means).all() or not np.isfinite(occupancy).all() or np.any(occupancy < 0.0) or np.any(occupancy > 1.0):
            raise ValueError(f"RadarSplat full-scale {label} Gaussian export is non-finite or out of range")

    launcher_root = Path(args.launcher_log_root).resolve()
    prepare_records = _read_resource_markers(launcher_root, "prepare.log", PREPARE_PREFIX)
    if not prepare_records:
        raise ValueError("RadarSplat full-scale lacks preparation resource evidence")
    for record in prepare_records:
        _resource_payload(record, "preparation")
    if not any(
        _nonnegative_int(_mapping(record["payload"], "prepare resource").get("targets_newly_materialized"), "new targets")
        + _nonnegative_int(_mapping(record["payload"], "prepare resource").get("targets_reused"), "reused targets")
        == 4200
        for record in prepare_records
    ):
        raise ValueError("RadarSplat full-scale preparation lacks a complete 4200-target resource row")
    train_records = _read_resource_markers(launcher_root, "fit_stdout.log", TRAIN_PREFIX)
    if not train_records:
        raise ValueError("RadarSplat full-scale lacks fit resource evidence")
    fit_lifecycle_records: list[dict[str, object]] = []
    completion_records = _read_resource_markers(launcher_root, "completion_resume.log", TRAIN_PREFIX)
    updates = 0
    for record in train_records:
        payload = _resource_payload(record, "training")
        phase = payload.get("phase")
        if phase in {"clean_interruption", "fit_finalization"}:
            fit_lifecycle_records.append(record)
            updates += _nonnegative_int(payload.get("optimizer_updates_this_invocation"), "fit updates")
        elif phase == "completion_resume":
            completion_records.append(record)
        else:
            raise ValueError(f"RadarSplat full-scale has an unknown training resource phase: {phase!r}")
    if not fit_lifecycle_records:
        raise ValueError("RadarSplat full-scale lacks fit lifecycle resource evidence")
    if not any(
        _mapping(record["payload"], "completion resource").get("phase") == "completion_resume"
        and _nonnegative_int(
            _mapping(record["payload"], "completion resource").get("optimizer_updates_this_invocation"),
            "completion updates",
        ) == 0
        for record in completion_records
    ):
        raise ValueError("RadarSplat full-scale lacks a verified zero-update completion-resume resource row")
    if updates != args.steps:
        raise ValueError(f"RadarSplat full-scale resource rows account for {updates} rather than {args.steps} updates")
    readout_records = _read_resource_markers(launcher_root, "readout.log", READOUT_PREFIX)
    current_readout_record = {
        "source": "current full-scale readout process",
        "payload": {
            "phase": "readout",
            "process_peak_rss_kib": int(trainer._process_peak_rss_kib() or 0),
            "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "cuda_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
    }
    phases = {
        "target_preparation": _phase_peak(prepare_records, "preparation"),
        "optimizer_updates": {
            "records": len(trace),
            "sources": [str(Path(args.checkpoint_dir) / "training_trace.jsonl")],
            "process_peak_rss_kib": max(_positive_int(row["process_peak_rss_kib"], "trace RSS") for row in trace),
            "torch_peak_allocated_bytes": max(_positive_int(row["cuda_max_memory_allocated_bytes"], "trace allocation") for row in trace),
            "torch_peak_reserved_bytes": max(_positive_int(row["cuda_max_memory_reserved_bytes"], "trace reservation") for row in trace),
        },
        "fit_lifecycle": _phase_peak(fit_lifecycle_records, "fit lifecycle"),
        "completion_resume": _phase_peak(completion_records, "completion resume"),
        "readout": _phase_peak([*readout_records, current_readout_record], "readout"),
    }
    resources = _check_limits(phases, args)
    resources["optimizer_updates_accounted"] = updates

    report = {
        "schema": "rift_radarsplat_b7873200_fullscale_readout_v1",
        "production_clearance": False,
        "scope": "full 3200-view native-power fit and 1000-view validation readout; no coherent phase, mesh, NVS, or reserved-test claim",
        "observable": "native real local-polar power: sum_elevation(abs(matched_filter_complex)**2)",
        "roles": {
            "train_count": len(cache.train_indices),
            "validation_count": len(cache.validation_indices),
            "reserved_test_count": 1000,
            "reserved_test_materialized": False,
            "reserved_test_accessed": False,
            "parent_unused_accessed": False,
            "coordinates_per_view": "16 Tx x 16 Rx x 600 frequency samples",
        },
        "target_provenance": {
            "cache_root": str(Path(args.cache_root).resolve()),
            **source_provenance,
            "role_manifest_name": role_manifest_name,
            "train_view_ids": [int(value) for value in cache.train_indices],
            "validation_view_ids": [int(value) for value in cache.validation_indices],
            "target_axes": ["azimuth", "range"],
            "target_spec": dict(target_spec),
            "grid": dict(target_grid),
            "grid_crop": {
                "azimuth_axis": "azimuth",
                "range_axis": "range",
                "n_azimuth": int(target_grid["n_azimuth"]),
                "n_range": int(target_grid["n_range"]),
                "azimuth_center_deg": float(target_grid["azimuth_center_deg"]),
                "output_azimuth_resolution_deg": float(target_grid["output_azimuth_resolution_deg"]),
                "intermediate_azimuth_resolution_deg": float(target_grid["intermediate_azimuth_resolution_deg"]),
            },
            "projection": target_spec["projection"],
            "matched_filter": dict(target_matched_filter),
            "normalization": {
                "mode": target_normalization["mode"],
                "fit_split": target_normalization["fit_split"],
                "clip": target_normalization["clip"],
                "train_peak_power": train_peak_power,
            },
        },
        "fit": {
            "updates": args.steps,
            "initial_gaussians": args.init_num_gaussians,
            "final_gaussians": int(final_means.shape[0]),
            "selected_gaussians": int(selected_means.shape[0]),
            "active_sh_degree_at_final": final_active_sh_degree,
            "active_sh_degree_at_selected": selected_active_sh_degree,
            "metrics_native_power": {
                "selected": selected_metrics,
                "zero_reference": zero,
                "final_diagnostic": final_diagnostic_metrics,
                "aggregation": {
                    "relative_mse_formula": "sum squared_error_native_power / sum target_energy_native_power over all evaluated views and bins",
                    "zero_reference_relative_mse": 1.0,
                },
            },
            "epochs": args.epochs,
            "updates_per_epoch": len(cache.train_indices),
            "validation_steps": expected_validation_steps,
            "densification": "disabled",
            "prune_every": args.prune_every,
            "trace_seconds_total": float(sum(float(row["seconds"]) for row in trace)),
        },
        "learning_curve_budget_status": learning_curve_budget_status,
        "checkpoint_selection": {
            "rule": "minimum validation global native-power RelMSE",
            "tie_break": "earliest exact tie",
            "selected_checkpoint": str(best_path),
            "selected_step": selected_step,
            "selected_active_sh_degree": selected_active_sh_degree,
            "selected_validation_relative_mse_native_power": selected_metric,
            "terminal_checkpoint": str(final_path),
            "terminal_step": final_step,
            "terminal_active_sh_degree": final_active_sh_degree,
            "same_checkpoint_pairing": "selected checkpoint supplies the benchmark train/validation metrics and selected native occupancy; terminal final/latest state and geometry are diagnostics",
        },
        "checkpoint_recovery": {
            "best_checkpoint": str(best_path),
            "final_checkpoint": str(final_path),
            "latest_checkpoint": str(latest_path),
            "latest_equals_final": True,
            "best_is_nonterminal_validation_state": True,
            "complete": True,
        },
        "native_geometry_readout": {
            "path": str(selected_geometry_path),
            "checkpoint": str(best_path),
            "step": selected_step,
            "active_sh_degree": selected_active_sh_degree,
            "representation": "Gaussian locations/scales/orientations with occupancy; not a mesh",
            "gaussian_count": int(selected_means.shape[0]),
            "occupancy_min": float(np.min(selected_occupancy)),
            "occupancy_mean": float(np.mean(selected_occupancy)),
            "occupancy_max": float(np.max(selected_occupancy)),
            "metric_geometry": "N/A absent registered fixed-threshold evaluation",
        },
        "terminal_geometry_diagnostic": {
            "path": str(geometry_path),
            "checkpoint": str(final_path),
            "step": final_step,
            "active_sh_degree": final_active_sh_degree,
            "representation": "Gaussian locations/scales/orientations with occupancy; not a mesh",
            "gaussian_count": int(final_means.shape[0]),
            "occupancy_min": float(np.min(final_occupancy)),
            "occupancy_mean": float(np.mean(final_occupancy)),
            "occupancy_max": float(np.max(final_occupancy)),
        },
        "resources": resources,
    }
    atomic_write_json(output, report)
    print(READOUT_PREFIX + json.dumps(resources["phase_peaks"]["readout"], sort_keys=True), flush=True)
    print("RADARSPLAT_B7873200_FULLSCALE_READOUT_PASS", flush=True)


if __name__ == "__main__":
    main()
