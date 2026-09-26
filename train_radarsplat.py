#!/usr/bin/env python
"""Train the object-bound native-power RadarSplat baseline.

This is deliberately separate from the historical sphere2k trainer.  It can
read only a prepared B7873200 local-polar cache; there is no raw NPZ argument
or raw-response accessor in this program.  Consequently, a development run
can consume exactly the sealed train and validation target roles and cannot
accidentally evaluate the reserved test set.

The current defaults form an engineering small-fit lane.  They preserve the
current occupancy/pruning policy (0.001 / 0.0005) and native Gaussian power
renderer, but they are not a claim that a 64-Gaussian CPU reference fit is the
final 20k-Gaussian paper-scale experiment.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rift.radarsplat_b7873200 import RadarSplatEffects, RadarSplatGrid, RadarSplatModel
from rift.radarsplat_b7873200_adapter import (
    OCCUPANCY_THRESHOLD,
    PRUNE_OPACITY,
    ObjectiveWeights,
    GEOMETRY_SCHEMA,
    create_optimizers,
    export_gaussian_occupancy_geometry,
    load_native_view,
    model_from_checkpoint_state,
    native_power_objective,
    prune_low_opacity,
    target_concentration,
)
from rift.radarsplat_b7873200_protocol import RECIPE_FILENAME, atomic_write_json, load_cache
from rift.radarsplat_b7873200_acquisition import acquisition_records_equal


GOTCHA_BACKEND = {
    "schema": "rift_gotcha_backend_v1", "method": "radarsplat", "callable": "run_gotcha",
    "selection_unit": "pass_sector", "joint_passes": True,
    "native_frequency_policy": "ragged_exact", "polarizations": ["hh", "hv", "vh", "vv"],
    "metric_domain": "clipped train-normalized native sector MF power",
    "fidelity_status": "released_source_with_native_MF_conversion_cuda_unvalidated",
    "runtime_requirements": "original pinned RadarSplat CUDA and fused-SSIM; no CPU model fallback",
}


def run_gotcha(*, dataset, output_dir, config, device, resume):
    from rift.radarsplat_gotcha import run_gotcha as backend
    return backend(dataset=dataset, output_dir=output_dir, config=config, device=device, resume=resume)


CHECKPOINT_VERSION = 2
_STOP_REQUESTED = False
_STOP_SIGNAL: int | None = None


def _request_stop(signum, _frame) -> None:
    global _STOP_REQUESTED, _STOP_SIGNAL
    _STOP_REQUESTED = True
    _STOP_SIGNAL = int(signum)
    print("RadarSplat B7873200 received a stop signal; saving after the current update.", flush=True)


def _process_peak_rss_kib() -> int | None:
    """Read the Linux process high-water mark for bounded-run telemetry.

    This is deliberately a readout rather than a training control. The project
    runs this lane on Linux; returning ``None`` keeps local source inspection
    portable, while allocated-node postflight requires the per-update records.
    """

    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) == 3 and fields[0] == "VmHWM:" and fields[2] == "kB":
                value = int(fields[1])
                return value if value > 0 else None
    except (OSError, ValueError):
        return None
    return None


def _emit_phase_resource(*, device: torch.device, phase: str, optimizer_updates_this_invocation: int) -> None:
    """Emit phase-complete resource telemetry without changing fit state."""

    payload: dict[str, object] = {
        "phase": phase,
        "optimizer_updates_this_invocation": int(optimizer_updates_this_invocation),
        "process_peak_rss_kib": _process_peak_rss_kib(),
    }
    if device.type == "cuda":
        payload["cuda_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        payload["cuda_max_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    print("RADARSPLAT_B7873200_TRAIN_RESOURCE_JSON=" + json.dumps(payload, sort_keys=True), flush=True)


class DeterministicViewSampler:
    """A checkpointable permutation cycle over the sealed training IDs."""

    def __init__(self, indices: Sequence[int], seed: int) -> None:
        if not indices:
            raise ValueError("RadarSplat B7873200 requires at least one training view")
        self.indices = tuple(int(index) for index in indices)
        self.rng = np.random.Generator(np.random.PCG64(int(seed)))
        self.order: tuple[int, ...] = ()
        self.cursor = 0

    def next(self) -> int:
        if self.cursor >= len(self.order):
            self.order = tuple(int(value) for value in self.rng.permutation(self.indices))
            self.cursor = 0
        index = self.order[self.cursor]
        self.cursor += 1
        return index

    def state_dict(self) -> dict[str, object]:
        return {
            "indices": list(self.indices),
            "order": list(self.order),
            "cursor": int(self.cursor),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state.get("indices") != list(self.indices):
            raise ValueError("RadarSplat B7873200 checkpoint sampler belongs to different training IDs")
        order = state.get("order")
        cursor = state.get("cursor")
        rng_state = state.get("rng_state")
        if not isinstance(order, list) or not isinstance(cursor, int) or not isinstance(rng_state, Mapping):
            raise ValueError("RadarSplat B7873200 checkpoint sampler state is malformed")
        normalized = tuple(int(value) for value in order)
        if normalized and (len(normalized) != len(self.indices) or set(normalized) != set(self.indices)):
            raise ValueError("RadarSplat B7873200 checkpoint sampler order is invalid")
        if cursor < 0 or cursor > len(normalized):
            raise ValueError("RadarSplat B7873200 checkpoint sampler cursor is invalid")
        self.order = normalized
        self.cursor = cursor
        self.rng.bit_generator.state = dict(rng_state)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    tokens = list(sys.argv[1:] if argv is None else argv)
    profile_parser = argparse.ArgumentParser(add_help=False)
    profile_parser.add_argument("--fidelity-profile", default="legacy")
    profile, _ = profile_parser.parse_known_args(tokens)
    if profile.fidelity_profile in ("upstream", "budget48"):
        from rift.radarsplat_release_training import parse_args as released_args
        return released_args(tokens)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-development-subset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="explicitly permit an engineering-only cache subset; never use for the 3200-view comparison",
    )
    parser.add_argument("--init-num-gaussians", type=int, default=64)
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
    parser.add_argument("--occupancy-threshold", type=float, default=OCCUPANCY_THRESHOLD)
    parser.add_argument("--prune-opacity", type=float, default=PRUNE_OPACITY)
    parser.add_argument("--prune-every", type=int, default=5)
    parser.add_argument("--gaussian-chunk-size", type=int, default=64)
    parser.add_argument("--max-raster-candidate-pairs", type=int, default=2_000_000)
    parser.add_argument("--fidelity-profile", choices=("legacy", "audit_v1", "upstream", "budget48"), default="legacy",
                        help="budget48 uses source CUDA with 112000 Gaussians; upstream retains 20000; audit_v1 is historical")
    parser.add_argument("--occupancy-window", type=int, default=10)
    parser.add_argument("--map-power-threshold", type=float, default=0.15)
    parser.add_argument("--raster-alpha-cutoff", type=float, default=1.0/255.0,
                        help="release default; alternatives are recorded numerical ablations")
    parser.add_argument("--use-noise-probability", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--antenna-gain", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> torch.device:
    from rift.radarsplat_fidelity import OccupancyRecipe
    OccupancyRecipe(window_views=args.occupancy_window, power_threshold=args.map_power_threshold)
    if not math.isfinite(args.raster_alpha_cutoff) or not 0 <= args.raster_alpha_cutoff < 1:
        raise ValueError("invalid raster cutoff")
    if args.fidelity_profile == "legacy" and (
        args.occupancy_window != 10 or args.map_power_threshold != 0.15 or
        args.raster_alpha_cutoff != 1.0/255.0 or not args.use_noise_probability or not args.antenna_gain
    ):
        raise ValueError("RadarSplat ablations require a separately identified audit_v1 recipe")
    if min(args.steps, args.validation_every, args.checkpoint_every, args.log_every, args.init_num_gaussians, args.sh_degree_interval, args.gaussian_chunk_size, args.max_raster_candidate_pairs) < 1:
        raise ValueError("RadarSplat B7873200 positive count settings are required")
    if not (0.0 < args.init_extent_m and 0.0 < args.init_scale_m and 0.0 < args.max_scale_m):
        raise ValueError("RadarSplat B7873200 extent and scales must be positive")
    if not (0.0 < args.init_opacity < 1.0 and 0.0 < args.init_noise_probability < 1.0):
        raise ValueError("RadarSplat B7873200 initial probabilities must lie strictly in (0,1)")
    if not 0 <= args.sh_degree <= 4:
        raise ValueError("RadarSplat B7873200 supports SH degree 0 through 4")
    if not 0.0 < args.occupancy_threshold <= 1.0:
        raise ValueError("RadarSplat B7873200 occupancy threshold must lie in (0,1]")
    if not math.isclose(args.occupancy_threshold, OCCUPANCY_THRESHOLD, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("RadarSplat B7873200 fixes occupancy threshold at the current 0.001 policy")
    if not 0.0 <= args.prune_opacity < 1.0 or args.prune_every < 1:
        raise ValueError("RadarSplat B7873200 prune settings are invalid")
    if not math.isclose(args.prune_opacity, PRUNE_OPACITY, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("RadarSplat B7873200 fixes prune opacity at the current 0.0005 policy")
    if not (0.0 <= args.ssim_weight <= 1.0):
        raise ValueError("RadarSplat B7873200 SSIM weight must lie in [0,1]")
    if min(args.occupancy_weight, args.max_size_weight, args.opacity_noise_weight) < 0.0:
        raise ValueError("RadarSplat B7873200 objective weights must be non-negative")
    if not (0.0 <= args.adam_beta1 < 1.0 and 0.0 <= args.adam_beta2 < 1.0 and args.adam_eps > 0.0):
        raise ValueError("RadarSplat B7873200 Adam settings are invalid")
    learning_rate_values = (
        args.means_lr_base,
        args.scales_lr,
        args.quaternions_lr,
        args.opacity_lr,
        args.noise_probability_lr,
        args.sh0_lr,
        args.shn_lr,
    )
    if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in learning_rate_values):
        raise ValueError("RadarSplat B7873200 learning rates must be finite and positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("RadarSplat B7873200 requested CUDA but CUDA is unavailable")
    return device


def _learning_rates(args: argparse.Namespace) -> dict[str, float]:
    return {
        "means": float(args.means_lr_base) * float(args.init_extent_m),
        "log_scales": float(args.scales_lr),
        "quaternions": float(args.quaternions_lr),
        "opacity_logits": float(args.opacity_lr),
        "noise_probability_logits": float(args.noise_probability_lr),
        "sh0": float(args.sh0_lr),
        "shN": float(args.shn_lr),
    }


def _model_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "init_num_gaussians": int(args.init_num_gaussians),
        "init_extent_m": float(args.init_extent_m),
        "init_scale_m": float(args.init_scale_m),
        "init_opacity": float(args.init_opacity),
        "init_noise_probability": float(args.init_noise_probability),
        "sh_degree": int(args.sh_degree),
        "sh_degree_interval": int(args.sh_degree_interval),
        "planar_initialization": getattr(args, "fidelity_profile", "legacy") == "legacy",
    }


def effects_for_args(args: argparse.Namespace) -> RadarSplatEffects:
    effects = RadarSplatEffects.b787_clean()
    if getattr(args, "fidelity_profile", "legacy") == "audit_v1":
        effects = replace(effects, probability_ceiling=1.0,
                          raster_alpha_cutoff=args.raster_alpha_cutoff,
                          use_noise_probability=args.use_noise_probability,
                          use_azimuth_antenna_gain=args.antenna_gain)
    return effects


def _renderer_identity(grid: RadarSplatGrid, effects: RadarSplatEffects) -> dict[str, object]:
    """Serialize scientific renderer settings without source/build identity pins."""

    serialized_effects = asdict(effects)
    # Preserve the exact historical identity for existing checkpoints.
    if effects.probability_ceiling is None and effects.raster_alpha_cutoff == 1.0/255.0:
        serialized_effects.pop("probability_ceiling")
        serialized_effects.pop("raster_alpha_cutoff")
    return {
        "grid": asdict(grid),
        "effects": serialized_effects,
        "native_observable": "additive local-polar Gaussian power",
        "elevation_gain": "unity release-faithful hook; no measured profile supplied",
        "range_power_exponent": 0.0,
        "local_crop_filtering": "native renderer halo/filter/crop path",
        "b787_coordinate_adaptations": {
            "sensor_frame_covariance_rotation": True,
            "cartesian_sh_direction": True,
        },
    }


def run_identity(args: argparse.Namespace, cache, effects: RadarSplatEffects) -> dict[str, object]:
    """Directly compare scientific semantics on resume; paths are provenance only."""

    identity = {
        "method": "radarsplat_b7873200_native_power_v1",
        "sealed_protocol_identity": dict(cache.identity),
        "target_recipe": dict(cache.recipe),
        "train_peak_power": float(cache.train_peak_power),
        "model": _model_config(args),
        "renderer": _renderer_identity(
            load_native_view(cache, cache.train_indices[0], "train", "cpu").grid, effects
        ),
        "objective": {
            "weights": asdict(
                ObjectiveWeights(
                    ssim=args.ssim_weight,
                    occupancy=args.occupancy_weight,
                    max_size=args.max_size_weight,
                    opacity_noise=args.opacity_noise_weight,
                )
            ),
            "occupancy_threshold": float(args.occupancy_threshold),
            "occupancy_balance": "equal positive/background class means when both exist",
            "max_scale_m": float(args.max_scale_m),
        },
        "strategy": {
            "prune_opacity": float(args.prune_opacity),
            "strict_prune_comparison": "opacity < prune_opacity, with float32 sigmoid/logit boundary roundoff only",
            "prune_every": int(args.prune_every),
            "all_rows_prune_guard": "retain maximum-opacity row",
        },
        "optimization": {
            "steps": int(args.steps),
            "validation_every": int(args.validation_every),
            "learning_rates": _learning_rates(args),
            "betas": [float(args.adam_beta1), float(args.adam_beta2)],
            "eps": float(args.adam_eps),
            "seed": int(args.seed),
        },
        "observable": "native polar real power and occupancy; no coherent phase prediction",
    }
    if getattr(args, "fidelity_profile", "legacy") == "audit_v1":
        from rift.radarsplat_fidelity import OccupancyRecipe
        identity["method"] = "radarsplat_native_power_audit_v1"
        identity["fidelity_profile"] = "audit_v1"
        identity["objective"].update(
            occupancy_balance="image-wide L1 mean",
            occupancy_target=OccupancyRecipe(window_views=args.occupancy_window,
                                            power_threshold=args.map_power_threshold).identity(),
            ssim="Gaussian 11x11 sigma1.5 valid; fixed C1=0.0001 C2=0.0009",
        )
    return identity


def _atomic_torch_save(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".pt", dir=path.parent, prefix=path.stem + ".tmp.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_checkpoint(path: Path, device: torch.device) -> Mapping[str, object]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, Mapping):
        raise ValueError("RadarSplat B7873200 checkpoint must be a mapping")
    return payload


def _artifact_paths(checkpoint_dir: Path) -> dict[str, Path]:
    return {
        "run_config": checkpoint_dir / "run_config.json",
        "latest": checkpoint_dir / "checkpoint_latest.pt",
        "best": checkpoint_dir / "checkpoint_best.pt",
        "final": checkpoint_dir / "checkpoint_final.pt",
        "geometry": checkpoint_dir / "gaussian_occupancy_geometry.npz",
        "summary": checkpoint_dir / "summary.json",
    }


def _assert_output_policy(paths: Mapping[str, Path], *, resume: bool) -> Path | None:
    """Reject ambiguous occupied output directories before any file is written."""

    occupied = {name for name, path in paths.items() if path.exists()}
    if not resume:
        if occupied:
            raise FileExistsError(
                "RadarSplat B7873200 --no-resume refuses an occupied output directory: "
                + ", ".join(sorted(occupied))
            )
        return None
    latest = paths["latest"]
    final = paths["final"]
    if occupied and not latest.is_file() and not final.is_file():
        raise FileExistsError(
            "RadarSplat B7873200 output has artifacts but no durable checkpoint; "
            "refusing to overwrite ambiguous evidence"
        )
    if latest.is_file():
        return latest
    if final.is_file():
        return final
    return None


def _write_or_validate_run_config(path: Path, identity: Mapping[str, object], cache_root: str) -> None:
    """Write provenance once, or compare its scientific identity without overwrite."""

    desired = {"run_identity": dict(identity), "cache_root": str(Path(cache_root).resolve())}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            observed = json.load(handle)
        if not isinstance(observed, Mapping) or observed.get("run_identity") != desired["run_identity"]:
            raise ValueError("RadarSplat B7873200 existing run_config disagrees with this scientific run")
        return
    atomic_write_json(path, desired)


def _assert_finite_value(value: object, label: str) -> None:
    """Reject non-finite serialized optimization state without a digest proxy."""

    if torch.is_tensor(value):
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise ValueError(f"RadarSplat B7873200 {label} is non-finite")
        return
    if isinstance(value, np.ndarray):
        if (np.issubdtype(value.dtype, np.floating) or np.issubdtype(value.dtype, np.complexfloating)) and not np.isfinite(value).all():
            raise ValueError(f"RadarSplat B7873200 {label} is non-finite")
        return
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError(f"RadarSplat B7873200 {label} is non-finite")
        return
    if isinstance(value, Mapping):
        for name, nested in value.items():
            _assert_finite_value(nested, f"{label}.{name}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _assert_finite_value(nested, f"{label}[{index}]")


def _is_strict_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _finite_metric(value: object, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"RadarSplat B7873200 {label} must be a numeric scalar")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(
            f"RadarSplat B7873200 {label} must be finite"
            + (" and non-negative" if nonnegative else "")
        )
    return result


def _adam_step_number(
    value: object,
    *,
    label: str,
    maximum_step: int | None,
) -> int:
    """Read an Adam step without accepting fractional or reset state."""

    if torch.is_tensor(value):
        if (
            value.layout != torch.strided
            or value.numel() != 1
            or torch.is_complex(value)
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"RadarSplat B7873200 {label} Adam step is invalid")
        numeric = float(value.detach().cpu())
    elif _is_strict_int(value) or isinstance(value, (float, np.floating)) and not isinstance(value, (bool, np.bool_)):
        numeric = float(value)
    else:
        raise ValueError(f"RadarSplat B7873200 {label} Adam step is invalid")
    if not math.isfinite(numeric) or numeric < 1.0 or not numeric.is_integer():
        raise ValueError(f"RadarSplat B7873200 {label} Adam step is invalid")
    result = int(numeric)
    if maximum_step is not None and result > maximum_step:
        raise ValueError(f"RadarSplat B7873200 {label} Adam step exceeds checkpoint progress")
    return result


def _validate_history_and_best(
    history: object,
    best_validation_rel_mse: object,
    *,
    checkpoint_step: int,
) -> tuple[list[dict[str, object]], float]:
    """Reject metric records that could make a saved run non-resumable."""

    if not isinstance(history, list):
        raise ValueError("RadarSplat B7873200 checkpoint history must be a list")
    rows: list[dict[str, object]] = []
    metric_values: list[float] = []
    previous_step = 0
    for position, row in enumerate(history):
        if not isinstance(row, Mapping):
            raise ValueError(f"RadarSplat B7873200 history row {position} is malformed")
        row_step = row.get("step")
        if not _is_strict_int(row_step) or int(row_step) <= previous_step or int(row_step) > checkpoint_step:
            raise ValueError(f"RadarSplat B7873200 history row {position} has an invalid validation step")
        metric = _finite_metric(
            row.get("relative_mse_native_power"),
            f"history row {position} relative_mse_native_power",
            nonnegative=True,
        )
        if "relative_l2_native_power" in row:
            relative_l2 = _finite_metric(
                row["relative_l2_native_power"],
                f"history row {position} relative_l2_native_power",
                nonnegative=True,
            )
            if not math.isclose(relative_l2 * relative_l2, metric, rel_tol=1.0e-6, abs_tol=1.0e-12):
                raise ValueError(
                    f"RadarSplat B7873200 history row {position} has inconsistent relative-L2 and relative-MSE"
                )
        for field in ("validation_bins", "validation_views", "active_sh_degree"):
            if field in row and (
                not _is_strict_int(row[field])
                or int(row[field]) < (1 if field != "active_sh_degree" else 0)
            ):
                raise ValueError(f"RadarSplat B7873200 history row {position} has invalid {field}")
        if "mean_positive_fraction" in row:
            fraction = _finite_metric(
                row["mean_positive_fraction"],
                f"history row {position} mean_positive_fraction",
                nonnegative=True,
            )
            if fraction > 1.0:
                raise ValueError(
                    f"RadarSplat B7873200 history row {position} mean_positive_fraction exceeds one"
                )
        _assert_finite_value(row, f"history row {position}")
        rows.append(dict(row))
        metric_values.append(metric)
        previous_step = int(row_step)
    if isinstance(best_validation_rel_mse, (bool, np.bool_)) or not isinstance(
        best_validation_rel_mse, (int, float, np.integer, np.floating)
    ):
        raise ValueError("RadarSplat B7873200 checkpoint best validation metric is malformed")
    best = float(best_validation_rel_mse)
    if math.isnan(best) or best == -math.inf or best < 0.0:
        raise ValueError("RadarSplat B7873200 checkpoint best validation metric is invalid")
    if not rows:
        if best != math.inf:
            raise ValueError("RadarSplat B7873200 checkpoint has a finite best metric before validation")
    else:
        expected_best = min(metric_values)
        if not math.isfinite(best) or best != expected_best:
            raise ValueError("RadarSplat B7873200 checkpoint best validation metric disagrees with history")
    return rows, best


def _checkpoint_control_state(
    *,
    identity: Mapping[str, object],
    step: object,
    complete: object,
    finalization_pending: object,
    history: object,
    best_validation_rel_mse: object,
    last_train: object,
) -> tuple[int, list[dict[str, object]], float]:
    """Validate save/restore control state before it can alter a run."""

    optimization = identity.get("optimization")
    if not isinstance(optimization, Mapping) or not _is_strict_int(optimization.get("steps")):
        raise ValueError("RadarSplat B7873200 run identity lacks a valid final step")
    configured_steps = int(optimization["steps"])
    if configured_steps < 1 or not _is_strict_int(step) or int(step) < 0 or int(step) > configured_steps:
        raise ValueError("RadarSplat B7873200 checkpoint has invalid progress state")
    if not isinstance(complete, bool) or not isinstance(finalization_pending, bool) or (complete and finalization_pending):
        raise ValueError("RadarSplat B7873200 checkpoint has invalid completion state")
    step_value = int(step)
    rows, best = _validate_history_and_best(
        history,
        best_validation_rel_mse,
        checkpoint_step=step_value,
    )
    if step_value == 0:
        if last_train is not None:
            raise ValueError("RadarSplat B7873200 checkpoint has training diagnostics before its first update")
    else:
        if not isinstance(last_train, Mapping) or last_train.get("step") != step_value:
            raise ValueError("RadarSplat B7873200 checkpoint last_train disagrees with progress")
        _assert_finite_value(last_train, "checkpoint last_train")
    if step_value == configured_steps or complete or finalization_pending:
        if step_value != configured_steps or not rows or rows[-1].get("step") != step_value:
            raise ValueError("RadarSplat B7873200 terminal checkpoint is not at the declared final validation step")
        if not math.isfinite(best):
            raise ValueError("RadarSplat B7873200 terminal checkpoint lacks a finite best validation metric")
    return step_value, rows, best


def _validate_model_state_for_restore(state: Mapping[str, object], model: RadarSplatModel) -> None:
    """Reject incompatible raw parameter tensors before PyTorch can cast them."""

    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("RadarSplat B7873200 checkpoint model state has unexpected parameters")
    for name, expected_tensor in expected.items():
        observed = state[name]
        if (
            not torch.is_tensor(observed)
            or observed.layout != torch.strided
            or observed.dtype != expected_tensor.dtype
            or observed.shape != expected_tensor.shape
            or not bool(torch.isfinite(observed).all())
        ):
            raise ValueError(f"RadarSplat B7873200 checkpoint model parameter {name} is incompatible")


def _validate_serialized_optimizer_states(
    optimizer_states: Mapping[str, object],
    optimizers: Mapping[str, torch.optim.Optimizer],
    *,
    checkpoint_step: int,
) -> None:
    """Check raw Adam state before load_state_dict can coerce it in place."""

    if set(optimizer_states) != set(optimizers):
        raise ValueError("RadarSplat B7873200 checkpoint optimizer groups are incompatible")
    any_initialized_state = False
    for name, optimizer in optimizers.items():
        serialized = optimizer_states[name]
        expected_serialized = optimizer.state_dict()
        if not isinstance(serialized, Mapping) or set(serialized) != {"state", "param_groups"}:
            raise ValueError(f"RadarSplat B7873200 checkpoint optimizer {name} is malformed")
        raw_groups = serialized["param_groups"]
        expected_groups = expected_serialized["param_groups"]
        if not _directly_equal(raw_groups, expected_groups):
            raise ValueError(
                f"RadarSplat B7873200 checkpoint optimizer {name} changes the declared native Adam recipe"
            )
        assert isinstance(raw_groups, list) and len(raw_groups) == 1
        parameter_ids = raw_groups[0].get("params")
        if not isinstance(parameter_ids, list) or len(parameter_ids) != 1 or not _is_strict_int(parameter_ids[0]):
            raise ValueError(f"RadarSplat B7873200 checkpoint optimizer {name} has invalid parameter binding")
        parameter_id = int(parameter_ids[0])
        raw_state = serialized["state"]
        if not isinstance(raw_state, Mapping):
            raise ValueError(f"RadarSplat B7873200 checkpoint optimizer {name} state is malformed")
        if not raw_state:
            continue
        if set(raw_state) != {parameter_id}:
            raise ValueError(f"RadarSplat B7873200 checkpoint optimizer {name} state has invalid parameter IDs")
        entry = raw_state[parameter_id]
        if not isinstance(entry, Mapping):
            raise ValueError(f"RadarSplat B7873200 checkpoint optimizer {name} state is malformed")
        # A row-prune preserves an explicitly empty state for a parameter that
        # has not yet received a gradient.  It is a valid uninitialized Adam
        # slot, provided another parameter records the positive-step update.
        if not entry:
            continue
        expected_fields = {"step", "exp_avg", "exp_avg_sq"}
        if set(entry) != expected_fields:
            raise ValueError(f"RadarSplat B7873200 checkpoint optimizer {name} Adam state is malformed")
        parameter = optimizer.param_groups[0]["params"][0]
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = entry[moment_name]
            if (
                not torch.is_tensor(moment)
                or moment.layout != torch.strided
                or moment.dtype != parameter.dtype
                or moment.shape != parameter.shape
                or not bool(torch.isfinite(moment).all())
            ):
                raise ValueError(
                    f"RadarSplat B7873200 checkpoint optimizer {name} {moment_name} is incompatible"
                )
        if bool((entry["exp_avg_sq"] < 0.0).any()):
            raise ValueError(
                f"RadarSplat B7873200 checkpoint optimizer {name} exp_avg_sq is negative"
            )
        _adam_step_number(
            entry["step"],
            label=f"checkpoint optimizer {name}",
            maximum_step=checkpoint_step,
        )
        any_initialized_state = True
    if checkpoint_step > 0 and not any_initialized_state:
        raise ValueError("RadarSplat B7873200 positive-step checkpoint has no native Adam state")


@torch.no_grad()
def _assert_finite_training_state(
    model: RadarSplatModel,
    optimizers: Mapping[str, torch.optim.Optimizer],
    *,
    label: str,
    maximum_adam_step: int | None = None,
    require_initialized_adam: bool = False,
) -> None:
    """Require a usable model and one compatible finite native Adam state per row tensor."""

    parameters = dict(model.named_parameters())
    if set(optimizers) != set(parameters):
        raise ValueError(f"RadarSplat B7873200 {label} optimizer groups do not match model parameters")
    any_initialized_state = False
    for name, parameter in parameters.items():
        if not bool(torch.isfinite(parameter).all()):
            raise ValueError(f"RadarSplat B7873200 {label} model parameter {name} is non-finite")
        optimizer = optimizers[name]
        if not isinstance(optimizer, torch.optim.Adam) or len(optimizer.param_groups) != 1:
            raise ValueError(f"RadarSplat B7873200 {label} optimizer {name} is not the native single-group Adam")
        group = optimizer.param_groups[0]
        if group.get("name") != name or len(group.get("params", ())) != 1 or group["params"][0] is not parameter:
            raise ValueError(f"RadarSplat B7873200 {label} optimizer {name} does not bind its model parameter")
        try:
            learning_rate = float(group["lr"])
            epsilon = float(group["eps"])
            beta1, beta2 = (float(value) for value in group["betas"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"RadarSplat B7873200 {label} optimizer {name} settings are malformed") from error
        if not (
            math.isfinite(learning_rate)
            and learning_rate > 0.0
            and math.isfinite(epsilon)
            and epsilon > 0.0
            and math.isfinite(beta1)
            and math.isfinite(beta2)
            and 0.0 <= beta1 < 1.0
            and 0.0 <= beta2 < 1.0
        ):
            raise ValueError(f"RadarSplat B7873200 {label} optimizer {name} settings are invalid")
        if not isinstance(group.get("amsgrad"), bool) or bool(group["amsgrad"]):
            raise ValueError(f"RadarSplat B7873200 {label} optimizer {name} has unsupported Adam semantics")
        for state_parameter, state in optimizer.state.items():
            if state_parameter is not parameter or not isinstance(state, Mapping):
                raise ValueError(f"RadarSplat B7873200 {label} optimizer {name} state is incompatible")
            if state:
                any_initialized_state = True
                expected_state = {"step", "exp_avg", "exp_avg_sq"}
                if set(state) != expected_state:
                    raise ValueError(f"RadarSplat B7873200 {label} optimizer {name} Adam state is malformed")
                for moment_name in ("exp_avg", "exp_avg_sq"):
                    moment = state[moment_name]
                    if (
                        not torch.is_tensor(moment)
                        or moment.layout != torch.strided
                        or moment.shape != parameter.shape
                        or moment.dtype != parameter.dtype
                        or not bool(torch.isfinite(moment).all())
                    ):
                        raise ValueError(f"RadarSplat B7873200 {label} optimizer {name} {moment_name} is incompatible")
                if bool((state["exp_avg_sq"] < 0.0).any()):
                    raise ValueError(f"RadarSplat B7873200 {label} optimizer {name} exp_avg_sq is negative")
                _adam_step_number(
                    state["step"],
                    label=f"{label} optimizer {name}",
                    maximum_step=maximum_adam_step,
                )
                _assert_finite_value(state, f"{label} optimizer {name} state")
    if require_initialized_adam and not any_initialized_state:
        raise ValueError(f"RadarSplat B7873200 {label} has no initialized native Adam state")
    quaternion_norm = torch.linalg.vector_norm(model.quaternions, dim=-1)
    derived = {
        "scales": model.scales,
        "occupancy": model.opacity,
        "noise_probability": model.noise_probability,
        "normalized_quaternions": torch.nn.functional.normalize(model.quaternions, p=2, dim=-1),
    }
    if not bool(torch.isfinite(quaternion_norm).all()) or not bool((quaternion_norm > 0.0).all()):
        raise ValueError(f"RadarSplat B7873200 {label} has invalid quaternion geometry")
    for name, value in derived.items():
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"RadarSplat B7873200 {label} derived {name} is non-finite")
    if not bool((derived["scales"] > 0.0).all()):
        raise ValueError(f"RadarSplat B7873200 {label} derived scales underflowed or became non-positive")


def _checkpoint_payload(
    *,
    identity: Mapping[str, object],
    model: RadarSplatModel,
    optimizers: Mapping[str, torch.optim.Optimizer],
    sampler: DeterministicViewSampler,
    step: int,
    history: list[dict[str, object]],
    best_validation_rel_mse: float,
    complete: bool,
    last_train: Mapping[str, object] | None,
    acquisition_record: Mapping[str, np.ndarray],
    finalization_pending: bool = False,
) -> dict[str, object]:
    step_value, validated_history, best_value = _checkpoint_control_state(
        identity=identity,
        step=step,
        complete=complete,
        finalization_pending=finalization_pending,
        history=history,
        best_validation_rel_mse=best_validation_rel_mse,
        last_train=last_train,
    )
    _assert_finite_training_state(
        model,
        optimizers,
        label="checkpoint save",
        maximum_adam_step=step_value,
        require_initialized_adam=step_value > 0,
    )
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "run_identity": dict(identity),
        "step": step_value,
        "complete": bool(complete),
        "finalization_pending": bool(finalization_pending),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dicts": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
        "sampler_state": sampler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "history": validated_history,
        "best_validation_rel_mse": best_value,
        "last_train": None if last_train is None else dict(last_train),
        "acquisition_record": {name: np.asarray(value).copy() for name, value in acquisition_record.items()},
    }


def _restore_checkpoint(
    checkpoint: Mapping[str, object],
    identity: Mapping[str, object],
    model: RadarSplatModel,
    optimizers: Mapping[str, torch.optim.Optimizer],
    sampler: DeterministicViewSampler,
    acquisition_record: Mapping[str, np.ndarray],
) -> tuple[int, list[dict[str, object]], float, Mapping[str, object] | None, bool, bool]:
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("RadarSplat B7873200 checkpoint version is unsupported")
    if checkpoint.get("run_identity") != identity:
        raise ValueError("RadarSplat B7873200 resume would change roles, target, renderer, objective, or strategy")
    state = checkpoint.get("model_state_dict")
    optimizer_states = checkpoint.get("optimizer_state_dicts")
    sampler_state = checkpoint.get("sampler_state")
    history = checkpoint.get("history")
    checkpoint_acquisition = checkpoint.get("acquisition_record")
    if not isinstance(state, Mapping) or not isinstance(optimizer_states, Mapping) or not isinstance(sampler_state, Mapping):
        raise ValueError("RadarSplat B7873200 checkpoint is structurally incomplete")
    step = checkpoint.get("step")
    complete = checkpoint.get("complete")
    finalization_pending = checkpoint.get("finalization_pending")
    best = checkpoint.get("best_validation_rel_mse")
    last_train = checkpoint.get("last_train")
    step_value, validated_history, best_value = _checkpoint_control_state(
        identity=identity,
        step=step,
        complete=complete,
        finalization_pending=finalization_pending,
        history=history,
        best_validation_rel_mse=best,
        last_train=last_train,
    )
    if not isinstance(checkpoint_acquisition, Mapping) or not acquisition_records_equal(
        checkpoint_acquisition, acquisition_record
    ):
        raise ValueError("RadarSplat B7873200 checkpoint direct acquisition calibration disagrees with this cache")
    _validate_model_state_for_restore(state, model)
    _validate_serialized_optimizer_states(
        optimizer_states,
        optimizers,
        checkpoint_step=step_value,
    )
    model.load_state_dict(state)
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(optimizer_states[name])
    _assert_finite_training_state(
        model,
        optimizers,
        label="checkpoint restore",
        maximum_adam_step=step_value,
        require_initialized_adam=step_value > 0,
    )
    sampler.load_state_dict(sampler_state)
    torch_state = checkpoint.get("torch_rng_state")
    python_state = checkpoint.get("python_rng_state")
    if not torch.is_tensor(torch_state) or python_state is None:
        raise ValueError("RadarSplat B7873200 checkpoint is missing RNG state")
    torch.set_rng_state(torch_state.cpu())
    random.setstate(python_state)
    assert isinstance(last_train, Mapping) or last_train is None
    assert isinstance(complete, bool) and isinstance(finalization_pending, bool)
    return step_value, validated_history, best_value, last_train, complete, finalization_pending


@torch.inference_mode()
def evaluate_native_power(
    model: RadarSplatModel,
    cache,
    effects: RadarSplatEffects,
    args: argparse.Namespace,
    device: torch.device,
    active_sh_degree: int,
    *,
    role: str,
) -> dict[str, float | int]:
    """Evaluate one sealed cache role with the unchanged native renderer.

    The training loop keeps its historical validation record shape through
    :func:`validate`.  This role-explicit helper is for bounded engineering
    readout only: it permits the small-fit postflight to report fixed training
    and validation metrics without opening a response or changing optimization.
    """

    if role == "train":
        indices = cache.train_indices
    elif role == "validation":
        indices = cache.validation_indices
    else:
        raise ValueError("RadarSplat B7873200 evaluation role must be train or validation")
    squared_error = 0.0
    target_energy = 0.0
    count = 0
    concentration: list[dict[str, float | int]] = []
    model.eval()
    for index in indices:
        view = load_native_view(cache, index, role, device)
        rendered = model.render(
            view.sensor_to_world,
            view.grid,
            effects,
            active_sh_degree=active_sh_degree,
            gaussian_chunk_size=args.gaussian_chunk_size,
            max_candidate_pairs=args.max_raster_candidate_pairs,
            raster_backend="torch_sparse_reference",
        )
        final_power = rendered.get("final_power")
        if (
            not torch.is_tensor(final_power)
            or final_power.ndim != 3
            or final_power.shape[0] != 1
            or final_power.shape[1:] != view.target_power.shape
            or not bool(torch.isfinite(final_power).all())
        ):
            raise RuntimeError(f"RadarSplat B7873200 {role} renderer returned invalid native power")
        if not bool(torch.isfinite(view.target_power).all()):
            raise RuntimeError(f"RadarSplat B7873200 {role} target is non-finite")
        difference = final_power[0] - view.target_power
        if not bool(torch.isfinite(difference).all()):
            raise RuntimeError(f"RadarSplat B7873200 {role} residual is non-finite")
        squared_error += float(difference.square().sum().cpu())
        target_energy += float(view.target_power.square().sum().cpu())
        concentration.append(target_concentration(view.target_power, args.occupancy_threshold))
        count += int(view.target_power.numel())
    rel_mse = squared_error / max(target_energy, 1.0e-30)
    relative_l2 = math.sqrt(rel_mse)
    mean_positive_fraction = float(np.mean([float(row["positive_fraction"]) for row in concentration]))
    if not all(math.isfinite(value) and value >= 0.0 for value in (squared_error, target_energy, rel_mse, relative_l2)):
        raise RuntimeError(f"RadarSplat B7873200 {role} metrics are non-finite")
    if not math.isfinite(mean_positive_fraction) or not 0.0 <= mean_positive_fraction <= 1.0:
        raise RuntimeError(f"RadarSplat B7873200 {role} concentration metric is invalid")
    return {
        "relative_mse_native_power": rel_mse,
        "relative_l2_native_power": relative_l2,
        "squared_error_native_power": squared_error,
        "target_energy_native_power": target_energy,
        "bins": count,
        "views": len(indices),
        "mean_positive_fraction": mean_positive_fraction,
    }


def validate(model: RadarSplatModel, cache, effects: RadarSplatEffects, args: argparse.Namespace, device: torch.device, active_sh_degree: int) -> dict[str, float | int]:
    """Preserve the validation-history schema used by the checkpoint protocol."""

    evaluated = evaluate_native_power(
        model,
        cache,
        effects,
        args,
        device,
        active_sh_degree,
        role="validation",
    )
    return {
        "relative_mse_native_power": float(evaluated["relative_mse_native_power"]),
        "relative_l2_native_power": float(evaluated["relative_l2_native_power"]),
        "validation_bins": int(evaluated["bins"]),
        "validation_views": int(evaluated["views"]),
        "mean_positive_fraction": float(evaluated["mean_positive_fraction"]),
    }


def _final_summary(
    *,
    best_validation: float,
    history: Sequence[Mapping[str, object]],
    geometry_path: Path,
) -> dict[str, object]:
    final_metric = None
    if history:
        final_metric = history[-1].get("relative_mse_native_power")
    return {
        "method": "RadarSplat B7873200 native-power development fit",
        "complete": True,
        "best_validation_relative_mse_native_power": float(best_validation),
        "final_validation_relative_mse_native_power": final_metric,
        "history": [dict(row) for row in history],
        "geometry_export": str(geometry_path),
        "geometry_checkpoint": "checkpoint_final.pt",
        "best_checkpoint": "checkpoint_best.pt when present; it is not the geometry export",
        "observable": "native polar real power; no coherent phase output",
        "geometry": "native Gaussian occupancy rows; not a mesh",
        "target_resolution_note": "concentration diagnostics are descriptive and do not alter the 32x32 target grid",
    }


def _directly_equal(left: object, right: object) -> bool:
    """Compare continuation state directly, without a digest surrogate."""

    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            return False
        return left.shape == right.shape and left.dtype == right.dtype and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)):
            return False
        return left.shape == right.shape and left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not (isinstance(left, Mapping) and isinstance(right, Mapping)):
            return False
        return set(left) == set(right) and all(_directly_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right):
            return False
        return len(left) == len(right) and all(_directly_equal(a, b) for a, b in zip(left, right))
    if type(left) is not type(right):
        return False
    try:
        result = left == right
    except (TypeError, ValueError):
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _validate_existing_completed_checkpoint(
    path: Path,
    *,
    expected: Mapping[str, object],
    label: str,
    device: torch.device,
) -> None:
    existing = _load_checkpoint(path, device)
    if not _directly_equal(existing, expected):
        raise ValueError(
            f"RadarSplat B7873200 existing {label} checkpoint is not the current completed state"
        )


def _validate_existing_geometry(
    path: Path,
    *,
    model: RadarSplatModel,
    active_sh_degree: int,
    train_peak_power: float,
) -> None:
    expected = {
        "schema": np.asarray(GEOMETRY_SCHEMA),
        "active_sh_degree": np.asarray(int(active_sh_degree), dtype=np.int64),
        "train_peak_power": np.asarray(float(train_peak_power), dtype=np.float64),
        "observable": np.asarray("native Gaussian occupancy geometry; no coherent phase field"),
        "means": model.means.detach().cpu().numpy().astype(np.float32, copy=False),
        "scales": model.scales.detach().cpu().numpy().astype(np.float32, copy=False),
        "quaternions": torch.nn.functional.normalize(model.quaternions.detach(), p=2, dim=-1).cpu().numpy().astype(np.float32, copy=False),
        "occupancy": model.opacity.detach().cpu().numpy().astype(np.float32, copy=False),
        "noise_probability": model.noise_probability.detach().cpu().numpy().astype(np.float32, copy=False),
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise ValueError("RadarSplat B7873200 existing geometry export has unexpected fields")
        observed = {name: np.asarray(archive[name]) for name in expected}
    exact_fields = {"schema", "active_sh_degree", "train_peak_power", "observable", "means"}
    derived_float32_fields = {"scales", "quaternions", "occupancy", "noise_probability"}
    for name in exact_fields:
        if observed[name].dtype != expected[name].dtype or not np.array_equal(observed[name], expected[name]):
            raise ValueError("RadarSplat B7873200 existing geometry export is not the current final model")
    for name in derived_float32_fields:
        expected_value = expected[name]
        observed_value = observed[name]
        if observed_value.shape != expected_value.shape or observed_value.dtype != expected_value.dtype:
            raise ValueError("RadarSplat B7873200 existing geometry export is not the current final model")
        # `exp`, sigmoid, and quaternion normalization are renderer readouts,
        # not persisted raw model rows.  A resumed export on a different CUDA
        # generation may round a float32 result by a few local ULPs.  This
        # tolerance is deliberately per-value and does not mask a model change.
        tolerance = 8.0 * np.maximum(
            np.abs(np.spacing(expected_value.astype(np.float32, copy=False))),
            np.finfo(np.float32).tiny,
        )
        if not np.all(np.abs(observed_value - expected_value) <= tolerance):
            raise ValueError("RadarSplat B7873200 existing geometry export is not the current final model")


def _validate_existing_summary(
    path: Path,
    *,
    best_validation: float,
    history: Sequence[Mapping[str, object]],
    geometry_path: Path,
) -> None:
    with path.open("r", encoding="utf-8") as handle:
        observed = json.load(handle)
    expected = _final_summary(
        best_validation=best_validation, history=history, geometry_path=geometry_path
    )
    if not _directly_equal(observed, expected):
        raise ValueError("RadarSplat B7873200 existing summary is not the current final model summary")


def _finalize_completed_artifacts(
    *,
    paths: Mapping[str, Path],
    identity: Mapping[str, object],
    model: RadarSplatModel,
    optimizers: Mapping[str, torch.optim.Optimizer],
    sampler: DeterministicViewSampler,
    step: int,
    history: list[dict[str, object]],
    best_validation: float,
    last_train: Mapping[str, object] | None,
    cache,
    args: argparse.Namespace,
) -> None:
    """Fill only missing final artifacts, then durably mark completion.

    A signal after the last optimizer update but before export leaves an
    explicit finalization-pending checkpoint. On resume this routine completes
    that export without re-running training or validation.
    """

    geometry_path = paths["geometry"]
    active_sh_degree = min((max(int(step), 1) - 1) // args.sh_degree_interval, args.sh_degree)
    complete_payload = _checkpoint_payload(
        identity=identity,
        model=model,
        optimizers=optimizers,
        sampler=sampler,
        step=step,
        history=history,
        best_validation_rel_mse=best_validation,
        complete=True,
        last_train=last_train,
        acquisition_record=cache.acquisition_record,
    )
    if paths["final"].exists():
        _validate_existing_completed_checkpoint(
            paths["final"],
            expected=complete_payload,
            label="final",
            device=next(model.parameters()).device,
        )
    if not geometry_path.exists():
        export_gaussian_occupancy_geometry(
            geometry_path,
            model,
            active_sh_degree=active_sh_degree,
            train_peak_power=cache.train_peak_power,
        )
    else:
        _validate_existing_geometry(
            geometry_path,
            model=model,
            active_sh_degree=active_sh_degree,
            train_peak_power=cache.train_peak_power,
        )
    if not paths["summary"].exists():
        atomic_write_json(
            paths["summary"],
            _final_summary(
                best_validation=best_validation,
                history=history,
                geometry_path=geometry_path,
            ),
        )
    else:
        _validate_existing_summary(
            paths["summary"],
            best_validation=best_validation,
            history=history,
            geometry_path=geometry_path,
        )
    if not paths["final"].exists():
        _atomic_torch_save(paths["final"], complete_payload)
    # A pending latest checkpoint must transition only after derived artifacts
    # exist. Replacing it here is an idempotent completion marker, not a rerun.
    _atomic_torch_save(paths["latest"], complete_payload)


def main(argv: Sequence[str] | None = None) -> None:
    tokens = list(sys.argv[1:] if argv is None else argv)
    profile_parser = argparse.ArgumentParser(add_help=False)
    profile_parser.add_argument("--fidelity-profile", default="legacy")
    profile, _ = profile_parser.parse_known_args(tokens)
    if profile.fidelity_profile in ("upstream", "budget48"):
        from rift.radarsplat_release_training import main as released_main
        return released_main(tokens)
    global _STOP_REQUESTED, _STOP_SIGNAL
    _STOP_REQUESTED = False
    _STOP_SIGNAL = None
    args = parse_args(argv)
    device = _validate_args(args)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    checkpoint_dir = Path(args.checkpoint_dir)
    paths = _artifact_paths(checkpoint_dir)
    checkpoint_source = _assert_output_policy(paths, resume=args.resume)
    checkpoint = None
    if checkpoint_source is not None:
        checkpoint = _load_checkpoint(checkpoint_source, device)
        saved_identity = checkpoint.get("run_identity", {})
        with (Path(args.cache_root) / RECIPE_FILENAME).open(encoding="utf-8") as handle:
            target_recipe = json.load(handle)
        expected_method = ("radarsplat_native_power_audit_v1" if args.fidelity_profile == "audit_v1"
                           else "radarsplat_b7873200_native_power_v1")
        # Cache validation reads every target image. Gate source/object/roles,
        # target recipe and profile using metadata before any of those reads.
        if (checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION
                or not isinstance(saved_identity, Mapping)
                or saved_identity.get("method") != expected_method
                or saved_identity.get("target_recipe") != target_recipe
                or saved_identity.get("sealed_protocol_identity") != target_recipe.get("sealed_protocol_identity")):
            raise ValueError("RadarSplat B7873200 checkpoint disagrees with this scientific run")
    cache = load_cache(args.cache_root)
    if cache.is_development_subset and not args.allow_development_subset:
        raise ValueError(
            "RadarSplat B7873200 cache is an explicit engineering subset; "
            "pass --allow-development-subset only for a small-fit gate"
        )
    if not math.isclose(cache.occupancy_threshold, args.occupancy_threshold, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("RadarSplat B7873200 trainer cutoff must match the prepared cache cutoff")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    effects = effects_for_args(args)
    if args.fidelity_profile == "audit_v1" and min(int(cache.grid[k]) for k in ("n_azimuth", "n_range")) < 11:
        raise ValueError("audit_v1 requires a target image of at least 11x11 for release SSIM")
    identity = run_identity(args, cache, effects)
    sampler = DeterministicViewSampler(cache.train_indices, args.seed)
    step = 0
    history: list[dict[str, object]] = []
    best_validation = math.inf
    last_train: Mapping[str, object] | None = None
    complete = False
    finalization_pending = False
    if checkpoint_source is not None:
        checkpoint_identity = checkpoint.get("run_identity")
        checkpoint_state = checkpoint.get("model_state_dict")
        if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION or checkpoint_identity != identity:
            raise ValueError("RadarSplat B7873200 checkpoint disagrees with this scientific run")
        if not isinstance(checkpoint_state, Mapping):
            raise ValueError("RadarSplat B7873200 checkpoint is missing its model state")
        model = model_from_checkpoint_state(checkpoint_state, device=device, seed=args.seed)
    else:
        model = RadarSplatModel.random_scene(
            num_gaussians=args.init_num_gaussians,
            extent=args.init_extent_m,
            planar_initialization=_model_config(args)["planar_initialization"],
            seed=args.seed,
            initial_scale=args.init_scale_m,
            initial_opacity=args.init_opacity,
            initial_noise_probability=args.init_noise_probability,
            sh_degree=args.sh_degree,
        ).to(device)
    optimizers = create_optimizers(
        model,
        _learning_rates(args),
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
    )
    if checkpoint_source is not None:
        step, history, best_validation, last_train, complete, finalization_pending = _restore_checkpoint(
            checkpoint, identity, model, optimizers, sampler, cache.acquisition_record
        )
    starting_step = step
    # A final checkpoint is durable evidence that this directory has already
    # completed.  Never let an unrelated/stale `latest` checkpoint train past
    # it: both files must describe the exact same completed control and model
    # state before this process writes a config or takes an optimizer step.
    if paths["final"].exists():
        expected_completed = _checkpoint_payload(
            identity=identity,
            model=model,
            optimizers=optimizers,
            sampler=sampler,
            step=step,
            history=history,
            best_validation_rel_mse=best_validation,
            complete=True,
            last_train=last_train,
            acquisition_record=cache.acquisition_record,
        )
        _validate_existing_completed_checkpoint(
            paths["final"],
            expected=expected_completed,
            label="final",
            device=device,
        )
        if paths["latest"].exists():
            if complete:
                expected_latest = expected_completed
                latest_label = "latest alongside final"
            elif finalization_pending:
                expected_latest = _checkpoint_payload(
                    identity=identity,
                    model=model,
                    optimizers=optimizers,
                    sampler=sampler,
                    step=step,
                    history=history,
                    best_validation_rel_mse=best_validation,
                    complete=False,
                    finalization_pending=True,
                    last_train=last_train,
                    acquisition_record=cache.acquisition_record,
                )
                latest_label = "finalization-pending latest alongside final"
            else:
                raise ValueError(
                    "RadarSplat B7873200 final checkpoint coexists with a nonterminal latest checkpoint"
                )
            _validate_existing_completed_checkpoint(
                paths["latest"],
                expected=expected_latest,
                label=latest_label,
                device=device,
            )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_or_validate_run_config(paths["run_config"], identity, args.cache_root)
    if complete or finalization_pending:
        _finalize_completed_artifacts(
            paths=paths,
            identity=identity,
            model=model,
            optimizers=optimizers,
            sampler=sampler,
            step=step,
            history=history,
            best_validation=best_validation,
            last_train=last_train,
            cache=cache,
            args=args,
        )
        _emit_phase_resource(
            device=device,
            phase="completion_resume",
            optimizer_updates_this_invocation=0,
        )
        print("RadarSplat B7873200 completion artifacts are present; no update executed.", flush=True)
        return
    original_handlers = {kind: signal.getsignal(kind) for kind in (signal.SIGTERM, signal.SIGINT)}
    for kind in original_handlers:
        signal.signal(kind, _request_stop)
    weights = ObjectiveWeights(
        ssim=args.ssim_weight,
        occupancy=args.occupancy_weight,
        max_size=args.max_size_weight,
        opacity_noise=args.opacity_noise_weight,
    )
    occupancy_provider = None
    if args.fidelity_profile == "audit_v1":
        from rift.radarsplat_fidelity import OccupancyRecipe, TrainingOccupancy
        occupancy_provider = TrainingOccupancy(cache, OccupancyRecipe(
            window_views=args.occupancy_window, power_threshold=args.map_power_threshold))
    try:
        for current_step in range(step + 1, args.steps + 1):
            started = time.perf_counter()
            model.train()
            view_index = sampler.next()
            view = load_native_view(cache, view_index, "train", device)
            active_sh_degree = min((current_step - 1) // args.sh_degree_interval, args.sh_degree)
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
            rendered = model.render(
                view.sensor_to_world,
                view.grid,
                effects,
                active_sh_degree=active_sh_degree,
                gaussian_chunk_size=args.gaussian_chunk_size,
                max_candidate_pairs=args.max_raster_candidate_pairs,
                raster_backend="torch_sparse_reference",
            )
            occupancy_target = None
            occupancy_diagnostics = None
            if occupancy_provider is not None:
                mask, occupancy_diagnostics = occupancy_provider.label(view_index)
                occupancy_target = torch.tensor(mask, device=device)
            losses = native_power_objective(
                rendered,
                view.target_power,
                model,
                max_scale=args.max_scale_m,
                occupancy_threshold_value=args.occupancy_threshold,
                weights=weights,
                target_occupancy=occupancy_target,
                fidelity_profile=args.fidelity_profile,
            )
            total = losses["total"]
            assert torch.is_tensor(total)
            if not bool(torch.isfinite(total)):
                raise RuntimeError("RadarSplat B7873200 native power objective is non-finite")
            separate_gradients = None
            if occupancy_provider is not None and (current_step == 1 or current_step % args.checkpoint_every == 0):
                from rift.radarsplat_fidelity import loss_branch_gradients
                separate_gradients = loss_branch_gradients(losses, model)
            total.backward()
            nonzero_gradient = False
            gradient_l1: dict[str, float] = {}
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    gradient_l1[name] = 0.0
                    continue
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError(f"RadarSplat B7873200 has non-finite gradient for {name}")
                value = float(parameter.grad.abs().sum().detach().cpu())
                gradient_l1[name] = value
                nonzero_gradient = nonzero_gradient or value > 0.0
            if not nonzero_gradient:
                raise RuntimeError("RadarSplat B7873200 native renderer produced zero gradient for every parameter")
            parameters_before_update = {
                name: parameter.detach().clone() for name, parameter in model.named_parameters()
            }
            for optimizer in optimizers.values():
                optimizer.step()
            update_l1: dict[str, float] = {}
            nonzero_update = False
            for name, parameter in model.named_parameters():
                value = float(
                    (parameter.detach() - parameters_before_update[name]).abs().sum().cpu()
                )
                if not math.isfinite(value):
                    raise RuntimeError(f"RadarSplat B7873200 has a non-finite update for {name}")
                update_l1[name] = value
                nonzero_update = nonzero_update or value > 0.0
            if not nonzero_update:
                raise RuntimeError("RadarSplat B7873200 native Adam produced zero update for every parameter")
            prune_info: dict[str, int | bool] | None = None
            if current_step % args.prune_every == 0:
                prune_info = prune_low_opacity(model, optimizers, args.prune_opacity)
            _assert_finite_training_state(
                model,
                optimizers,
                label=f"post-update step {current_step}",
            )
            last_train = {
                "step": current_step,
                "view_index": view_index,
                "active_sh_degree": active_sh_degree,
                "total": float(total.detach().cpu()),
                "power_l1": float(torch.as_tensor(losses["power_l1"]).detach().cpu()),
                "occupancy_l1": float(torch.as_tensor(losses["occupancy_l1"]).detach().cpu()),
                "occupancy_balance_mode": str(losses["occupancy_balance_mode"]),
                "gradient_l1": gradient_l1,
                "update_l1": update_l1,
                "prune": prune_info,
                "seconds": time.perf_counter() - started,
            }
            if occupancy_provider is not None:
                from rift.radarsplat_fidelity import branch_diagnostics
                last_train["occupancy_map"] = occupancy_diagnostics
                last_train["branches"] = branch_diagnostics(rendered, effects.raster_alpha_cutoff)
                if separate_gradients is not None:
                    last_train["branch_gradient_l1"] = separate_gradients
            if device.type == "cuda":
                last_train["cuda_max_memory_allocated_bytes"] = int(
                    torch.cuda.max_memory_allocated(device)
                )
                last_train["cuda_max_memory_reserved_bytes"] = int(
                    torch.cuda.max_memory_reserved(device)
                )
            process_peak_rss_kib = _process_peak_rss_kib()
            if process_peak_rss_kib is not None:
                last_train["process_peak_rss_kib"] = process_peak_rss_kib
            if current_step % args.log_every == 0:
                print(json.dumps(last_train, sort_keys=True), flush=True)
            due_validation = current_step % args.validation_every == 0 or current_step == args.steps
            if due_validation:
                metrics = validate(model, cache, effects, args, device, active_sh_degree)
                metrics["step"] = current_step
                metrics["active_sh_degree"] = active_sh_degree
                history.append(metrics)
                if metrics["relative_mse_native_power"] < best_validation:
                    best_validation = float(metrics["relative_mse_native_power"])
                    _atomic_torch_save(paths["best"], _checkpoint_payload(
                        identity=identity, model=model, optimizers=optimizers, sampler=sampler,
                        step=current_step, history=history, best_validation_rel_mse=best_validation,
                        complete=False, last_train=last_train,
                        acquisition_record=cache.acquisition_record,
                    ))
            should_save = current_step % args.checkpoint_every == 0 or due_validation or _STOP_REQUESTED
            if should_save:
                _atomic_torch_save(paths["latest"], _checkpoint_payload(
                    identity=identity, model=model, optimizers=optimizers, sampler=sampler,
                    step=current_step, history=history, best_validation_rel_mse=best_validation,
                    complete=False, last_train=last_train,
                    acquisition_record=cache.acquisition_record,
                ))
            if _STOP_REQUESTED:
                # Preserve phase-wide resource evidence for every optimizer
                # update that completed before this clean, resumable exit.
                # This includes the narrow but important case where TERM
                # arrives after update ``args.steps`` and before finalization:
                # a later no-update resume must not erase those 20 updates
                # from the postflight accounting.
                _emit_phase_resource(
                    device=device,
                    phase="clean_interruption",
                    optimizer_updates_this_invocation=current_step - starting_step,
                )
                print("RadarSplat B7873200 clean interruption saved checkpoint_latest.pt", flush=True)
                raise SystemExit(143)
        pending_payload = _checkpoint_payload(
            identity=identity, model=model, optimizers=optimizers, sampler=sampler,
            step=args.steps, history=history, best_validation_rel_mse=best_validation,
            complete=False, finalization_pending=True, last_train=last_train,
            acquisition_record=cache.acquisition_record,
        )
        _atomic_torch_save(paths["latest"], pending_payload)
        _finalize_completed_artifacts(
            paths=paths,
            identity=identity,
            model=model,
            optimizers=optimizers,
            sampler=sampler,
            step=args.steps,
            history=history,
            best_validation=best_validation,
            last_train=last_train,
            cache=cache,
            args=args,
        )
        _emit_phase_resource(
            device=device,
            phase="fit_finalization",
            optimizer_updates_this_invocation=args.steps - starting_step,
        )
        print("RADARSPLAT_B7873200_NATIVE_TRAINING_COMPLETE", flush=True)
    finally:
        for kind, handler in original_handlers.items():
            signal.signal(kind, handler)


if __name__ == "__main__":
    main()
