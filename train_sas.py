#!/usr/bin/env python
"""Train legacy RIFT-SAS, adaptive RIFT-SAS, or independent SH-SAS.

The adaptive and SH-SAS comparison identities require a measured cache with
explicit disjoint train/validation/test roles.  ``rift_sas`` retains the
historical unscoped-cache fallback for old synthetic experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import operator
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from rift.rift_sas import (
    AdaptiveRIFTSASField,
    ComplexSHSonarField,
    RIFTSASGrid,
    RIFTSASRectangularGrid,
)
from rift.calibration import GlobalComplexGain
from rift.sas_dataset import atomic_json, load_sas_cache
from rift.sas_operator import render_sas_bins
from rift.sparse_scene import AdaptivePointSHScene
from rift.sh_sas import (
    PAPER_HASH_BASE_RESOLUTION,
    PAPER_HASH_FINAL_RESOLUTION,
    PAPER_HASH_LEVELS,
    PAPER_MLP_WIDTH,
    PAPER_SH_DEGREE,
    SHSASField,
)


CONTRACT_VERSION = 1
STOP_REQUESTED = False


def request_stop(signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"Received signal {signum}; checkpointing after this step.", flush=True)


class ComplexCalibration(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.real = torch.nn.Parameter(torch.tensor(1.0))
        self.imag = torch.nn.Parameter(torch.tensor(0.0))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def maybe_initialize(self, predicted: torch.Tensor, target: torch.Tensor) -> None:
        if bool(self.initialized):
            return
        denominator = predicted.abs().square().sum().clamp_min(1e-20)
        gain = (target * predicted.conj()).sum() / denominator
        if torch.isfinite(gain.real) and torch.isfinite(gain.imag):
            self.real.copy_(gain.real.float())
            self.imag.copy_(gain.imag.float())
        self.initialized.fill_(True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.complex(self.real, self.imag).to(value.dtype)


class LogPolarCalibration(GlobalComplexGain):
    def __init__(self, corr_threshold: float = 0.05) -> None:
        super().__init__()
        self.corr_threshold = float(corr_threshold)

    @torch.no_grad()
    def maybe_initialize(self, predicted: torch.Tensor, target: torch.Tensor) -> None:
        """One-time warm start of the gain from the first rendered viewpoint.

        Reimplements GlobalComplexGain.maybe_init_scale (rift/calibration.py)
        with a parameterized correlation threshold in place of its hard-coded
        0.05: that module is shared with the frozen radar path and must not
        change for this sonar-only flag. Arithmetic and prints are otherwise
        identical; with the default 0.05 the behavior is identical to before.
        """
        if bool(self.initialized):
            return
        meas_norm = torch.linalg.vector_norm(target)
        pred_norm = torch.linalg.vector_norm(predicted)
        if pred_norm < 1e-20:
            return
        if predicted.shape != target.shape:
            raise ValueError(f"maybe_initialize needs identically-laid-out tensors, got "
                              f"{tuple(predicted.shape)} vs {tuple(target.shape)}")
        proj = (predicted.conj() * target).sum() / (pred_norm ** 2).clamp_min(1e-30)
        norm_ratio = (meas_norm / pred_norm.clamp_min(1e-30)).clamp_min(1e-30)
        if proj.abs() > self.corr_threshold * norm_ratio:
            self.log_mag.fill_(torch.log(proj.abs().clamp_min(1e-30)).item())
            self.phase.fill_(torch.angle(proj).item())
            how = "projection <S_pred,S_meas>/||S_pred||^2"
        else:
            self.log_mag.fill_(torch.log(norm_ratio).item())
            how = "power ratio ||S_meas||/||S_pred|| (prediction uncorrelated with data)"
        self.initialized.fill_(True)
        g = torch.polar(torch.exp(self.log_mag), self.phase)
        print(f"GlobalComplexGain: warm-started g = {complex(g.item()):.4e} via {how}", flush=True)


def resolve_calibration_mode(
    model_kind: str, requested_mode: str, state: Optional[dict] = None
) -> str:
    valid_modes = ("auto", "log_polar", "legacy_cartesian")
    if requested_mode not in valid_modes:
        raise ValueError(f"unsupported calibration mode {requested_mode!r}; expected one of {valid_modes}")
    if state is None:
        if requested_mode != "auto":
            return requested_mode
        return "log_polar" if model_kind == "adaptive_rift_sas" else "legacy_cartesian"

    calibration_state = state.get("calibration_state_dict")
    if not isinstance(calibration_state, dict):
        raise ValueError("checkpoint is missing calibration_state_dict")
    state_keys = set(calibration_state.keys())
    legacy_keys = {"real", "imag", "initialized"}
    log_polar_keys = {"log_mag", "phase", "initialized"}
    if state_keys == legacy_keys:
        inferred_mode = "legacy_cartesian"
    elif state_keys == log_polar_keys:
        inferred_mode = "log_polar"
    else:
        raise ValueError(f"unsupported calibration state keys: {sorted(state_keys)}")
    if "calibration_mode" in state:
        metadata_mode = state["calibration_mode"]
    else:
        metadata_mode = None
    if "calibration_mode" in state and metadata_mode != inferred_mode:
        raise ValueError(
            f"calibration metadata/state disagreement: metadata={metadata_mode!r}, state={inferred_mode!r}"
        )
    if requested_mode != "auto" and requested_mode != inferred_mode:
        raise ValueError(
            "a resumed checkpoint must retain its calibration parameterization: "
            f"requested {requested_mode!r}, checkpoint {inferred_mode!r}"
        )
    return inferred_mode


def build_calibration(mode: str, device: torch.device, corr_threshold: float = 0.05) -> torch.nn.Module:
    if mode == "log_polar":
        return LogPolarCalibration(corr_threshold=corr_threshold).to(device)
    if mode == "legacy_cartesian":
        return ComplexCalibration().to(device)
    raise ValueError(f"unsupported calibration mode {mode!r}")


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--model", choices=("rift_sas", "adaptive_rift_sas", "sh_sas"), required=True)
    parser.add_argument(
        "--calibration-mode",
        choices=("auto", "log_polar", "legacy_cartesian"),
        default="auto",
        help="Global gain parameterization; auto preserves checkpoints and uses log-polar for new adaptive sonar runs.",
    )
    parser.add_argument("--checkpoint-root", default="training_checkpoints")
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--evaluation-role", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--profile", choices=("smoke", "full"), default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--granularity", type=int, default=64)
    parser.add_argument("--initial-granularity", type=int, default=16)
    parser.add_argument("--adaptive-capacity", type=int, default=65536)
    parser.add_argument("--max-active", type=int, default=65536)
    parser.add_argument("--split-max-level", type=int, default=3)
    parser.add_argument("--refine-every", type=int, default=1000)
    parser.add_argument("--probe-every", type=int, default=16)
    parser.add_argument("--spatial-fraction", type=float, default=0.05)
    parser.add_argument("--angular-fraction", type=float, default=0.10)
    parser.add_argument("--cooldown-events", type=int, default=1)
    parser.add_argument("--child-maturity-events", type=int, default=1)
    parser.add_argument("--position-lr", type=float, default=1.0e-4)
    parser.add_argument("--coefficient-lr", type=float, default=1.0e-3)
    parser.add_argument("--sh-degree", type=int, default=PAPER_SH_DEGREE)
    parser.add_argument("--init-scale", type=float, default=1e-2)
    parser.add_argument("--hash-levels", type=int, default=PAPER_HASH_LEVELS)
    parser.add_argument("--hash-features", type=int, default=2)
    parser.add_argument("--hash-base-resolution", type=int, default=PAPER_HASH_BASE_RESOLUTION)
    parser.add_argument("--hash-final-resolution", type=int, default=PAPER_HASH_FINAL_RESOLUTION)
    parser.add_argument("--hash-log2-size", type=int, default=19)
    parser.add_argument("--hidden-dim", type=int, default=PAPER_MLP_WIDTH)
    parser.add_argument("--query-chunk", type=int, default=65536)
    parser.add_argument("--grid-shape", type=int, nargs=3, default=None)

    parser.add_argument("--num-rays", type=int, default=4900)
    parser.add_argument("--max-bins", type=int, default=110)
    parser.add_argument("--opacity-scale", type=float, default=500.0)
    parser.add_argument("--lambertian-ratio", type=float, default=0.0)
    parser.add_argument("--normal-step", type=float, default=0.0032)
    parser.add_argument("--signal-scale", type=float, default=10.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-pings", type=int, default=8)
    parser.add_argument("--eval-bins", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-pings", type=int, default=0)
    parser.add_argument("--require-explicit-splits", action="store_true")
    parser.add_argument("--sh-direction", choices=("rx_to_point", "tx_to_point"), default="rx_to_point")
    parser.add_argument("--beamwidth-deg", type=float, default=None)
    parser.add_argument("--opacity-normalize", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ray-chunk", type=int, default=0)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--pings-per-step", type=int, default=1)
    parser.add_argument("--gain-init-corr-threshold", type=float, default=0.05)
    return parser.parse_args(argv)


SAVED_SCIENTIFIC_FIELDS = (
    "model", "seed", "lr", "granularity", "grid_shape", "initial_granularity", "adaptive_capacity",
    "max_active", "split_max_level", "refine_every", "probe_every", "spatial_fraction",
    "angular_fraction", "cooldown_events", "child_maturity_events", "position_lr",
    "coefficient_lr", "sh_degree", "init_scale", "hash_levels", "hash_features",
    "hash_base_resolution", "hash_final_resolution", "hash_log2_size", "hidden_dim",
    "num_rays", "max_bins", "opacity_scale", "lambertian_ratio", "normal_step",
    "signal_scale", "grad_clip", "max_pings", "require_explicit_splits", "sh_direction",
    "beamwidth_deg", "opacity_normalize", "adam_eps", "pings_per_step", "gain_init_corr_threshold",
)
SAVED_EVALUATION_FIELDS = ("eval_every", "eval_pings", "eval_bins")


def _explicit_cli_fields(raw_argv: Sequence[str]) -> set[str]:
    fields: set[str] = set()
    known = set(SAVED_SCIENTIFIC_FIELDS) | set(SAVED_EVALUATION_FIELDS) | {
        "profile", "steps", "query_chunk", "evaluation_role", "calibration_mode",
    }
    for token in raw_argv:
        if token == "--":
            break
        if not token.startswith("--"):
            continue
        name = token[2:].split("=", 1)[0].replace("-", "_")
        if name.startswith("no_"):
            name = name[3:]
        if name in known:
            fields.add(name)
    return fields


def _apply_profile(args, explicit_fields: set[str]) -> set[str]:
    if args.profile != "full" or args.eval_only:
        return set()
    values = {
        "steps": 26000,
        "num_rays": 4900,
        "max_bins": 110,
        "eval_every": 1000,
        "eval_pings": 64,
        "eval_bins": 0,
    }
    applied: set[str] = set()
    for field, value in values.items():
        if field not in explicit_fields:
            setattr(args, field, value)
            applied.add(field)
    return applied


def _recipe_value(field: str, value: Any) -> Any:
    if field == "opacity_normalize":
        return False if value is None else bool(value)
    if field == "grid_shape":
        if value is None:
            return None
        try:
            normalized = tuple(operator.index(item) for item in value)
        except (AttributeError, TypeError, ValueError):
            raise ValueError("grid_shape must contain exactly three integers >= 2") from None
        if len(normalized) != 3 or any(item < 2 for item in normalized):
            raise ValueError("grid_shape must contain exactly three integers >= 2")
        return normalized
    return value


def _recipe_values_equal(field: str, first: Any, second: Any) -> bool:
    first = _recipe_value(field, first)
    second = _recipe_value(field, second)
    if first is None or second is None:
        return first is None and second is None
    if isinstance(first, (float, np.floating)) or isinstance(second, (float, np.floating)):
        try:
            return bool(np.isclose(float(first), float(second), rtol=0.0, atol=1.0e-12, equal_nan=True))
        except (TypeError, ValueError):
            return False
    return first == second


def _saved_args_or_fail(state: Mapping[str, Any], *, require_evaluation: bool = True) -> Mapping[str, Any]:
    saved_args = state.get("args")
    if not isinstance(saved_args, Mapping):
        raise ValueError("checkpoint is missing saved args; cannot restore its scientific recipe")
    # Bounded migrations are intentional: checkpoints written before an option
    # existed used its listed historical default, so a missing field resolves
    # to that default rather than failing the checkpoint. grid_shape=None is
    # the legacy cubic field; adam_eps/pings_per_step/gain_init_corr_threshold
    # are the P1 flags' defaults, which is what every historical trajectory
    # actually used.
    saved_args = dict(saved_args)
    if "grid_shape" not in saved_args:
        saved_args["grid_shape"] = None
    if "adam_eps" not in saved_args:
        saved_args["adam_eps"] = 1e-8
    if "pings_per_step" not in saved_args:
        saved_args["pings_per_step"] = 1
    if "gain_init_corr_threshold" not in saved_args:
        saved_args["gain_init_corr_threshold"] = 0.05
    required = [field for field in SAVED_SCIENTIFIC_FIELDS if field != "grid_shape"]
    if require_evaluation:
        required.extend(SAVED_EVALUATION_FIELDS)
    missing = [field for field in required if field not in saved_args]
    if missing:
        raise ValueError(
            "checkpoint saved args are missing required fields: "
            f"{missing}; saved keys are {sorted(saved_args)}"
        )
    return saved_args


def _reconcile_saved_recipe(
    args,
    state: Mapping[str, Any],
    explicit_fields: set[str],
    profile_applied_fields: set[str],
    *,
    eval_only: bool,
) -> None:
    saved_args = _saved_args_or_fail(state)
    for field in SAVED_SCIENTIFIC_FIELDS:
        requested = field in explicit_fields or field in profile_applied_fields
        saved_value = saved_args[field]
        current_value = getattr(args, field)
        if requested:
            if not _recipe_values_equal(field, current_value, saved_value):
                raise ValueError(
                    f"checkpoint recipe mismatch for {field}: "
                    f"saved={_recipe_value(field, saved_value)!r}, "
                    f"requested={_recipe_value(field, current_value)!r}"
                )
        else:
            setattr(args, field, _recipe_value(field, saved_value))

    for field in SAVED_EVALUATION_FIELDS:
        saved_value = saved_args[field]
        if eval_only and field in explicit_fields:
            continue
        requested = field in explicit_fields or field in profile_applied_fields
        if requested and not _recipe_values_equal(field, getattr(args, field), saved_value):
            raise ValueError(
                f"checkpoint evaluation recipe mismatch for {field}: "
                f"saved={saved_value!r}, requested={getattr(args, field)!r}"
            )
        if not requested:
            setattr(args, field, saved_value)


_CACHE_SCIENTIFIC_PATHS = (
    ("scene",),
    ("dataset_identity",),
    ("frontend_asset_identity",),
    ("bandwidth_khz",),
    ("num_pings",),
    ("num_bins",),
    ("original_num_bins",),
    ("ring_size",),
    ("num_rings",),
    ("ring_protocol",),
    ("held_out_protocol",),
    ("split_contract", "explicit"),
    ("split_contract", "train_indices"),
    ("split_contract", "validation_indices"),
    ("split_contract", "test_indices"),
    ("split_contract", "train_ring_residue"),
    ("split_contract", "validation_ring_residue"),
    ("split_contract", "test_ring_residue"),
    ("split_contract", "counts"),
    ("frontend_normalization", "mode"),
    ("frontend_normalization", "strict_train_only"),
    ("frontend_normalization", "supplied_scale"),
    ("frontend_normalization", "max_transmissions"),
    ("aggressive_crop", "original_min_dist"),
    ("aggressive_crop", "original_max_dist"),
    ("aggressive_crop", "new_min_sample"),
    ("aggressive_crop", "new_max_sample_exclusive"),
    ("aggressive_crop", "cropped_min_dist"),
    ("aggressive_crop", "cropped_max_dist"),
    ("crop", "min_sample"),
    ("crop", "min_dist"),
    ("crop", "max_dist"),
    ("crop", "num_samples"),
    ("sound_speed_mps",),
    ("sample_rate_hz",),
    ("geometry_grid_shape",),
)


def _manifest_value(manifest: Mapping[str, Any], path: tuple[str, ...], source: str) -> Any:
    value: Any = manifest
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            dotted = ".".join(path)
            raise ValueError(f"{source} cache manifest is missing scientific field {dotted!r}")
        value = value[key]
    if isinstance(value, Mapping):
        return {key: _manifest_value(value, (key,), source) for key in value}
    if isinstance(value, list):
        return [_manifest_value({"value": item}, ("value",), source) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _scientific_cache_contract(manifest: Mapping[str, Any], source: str) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise ValueError(f"{source} cache manifest is not a mapping")
    return {
        ".".join(path): _manifest_value(manifest, path, source)
        for path in _CACHE_SCIENTIFIC_PATHS
    }


def _cache_contract_equal(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    for key in first:
        left, right = first[key], second[key]
        if isinstance(left, (float, int)) and isinstance(right, (float, int)):
            if not bool(np.isclose(float(left), float(right), rtol=0.0, atol=1.0e-12, equal_nan=True)):
                return False
        elif left != right:
            return False
    return True


def _validate_cache_contract(state: Mapping[str, Any], cache) -> None:
    saved_manifest = state.get("cache_manifest")
    current_manifest = getattr(cache, "manifest", None)
    saved_contract = _scientific_cache_contract(saved_manifest, "checkpoint")
    current_contract = _scientific_cache_contract(current_manifest, "current")
    if not _cache_contract_equal(saved_contract, current_contract):
        mismatches = [
            key for key in saved_contract
            if not _cache_contract_equal({key: saved_contract[key]}, {key: current_contract[key]})
        ]
        raise ValueError(f"scientific cache contract mismatch in {mismatches}")

    split = current_manifest["split_contract"]
    for role, attribute in (
        ("train_indices", "train_indices"),
        ("validation_indices", "validation_indices"),
        ("test_indices", "test_indices"),
    ):
        actual = np.asarray(getattr(cache, attribute), dtype=np.int64)
        declared = np.asarray(split[role], dtype=np.int64)
        if not np.array_equal(actual, declared):
            raise ValueError(f"current cache {attribute} disagrees with manifest split_contract.{role}")
    num_pings = int(getattr(cache, "num_pings"))
    num_bins = int(getattr(cache, "num_bins"))
    if num_pings != int(saved_contract["num_pings"]):
        raise ValueError(f"current cache num_pings={num_pings} disagrees with saved {saved_contract['num_pings']}")
    if num_bins != int(saved_contract["num_bins"]):
        raise ValueError(f"current cache num_bins={num_bins} disagrees with saved {saved_contract['num_bins']}")


def _validate_saved_model_box(state: Mapping[str, Any], cache) -> None:
    model_state = state.get("model_state_dict")
    if not isinstance(model_state, Mapping):
        raise ValueError("checkpoint is missing model_state_dict needed for scene-box validation")
    for key in ("scene_center", "scene_half_extent"):
        if key not in model_state:
            raise ValueError(f"checkpoint model_state_dict is missing scene box buffer {key!r}")
    corners = torch.as_tensor(np.asarray(cache.corners), dtype=torch.float32)
    if corners.ndim != 2 or corners.shape[-1] != 3:
        raise ValueError("current cache corners must have shape [N,3] for scene-box validation")
    expected_center = (corners.amin(dim=0) + corners.amax(dim=0)) / 2.0
    expected_half = (corners.amax(dim=0) - corners.amin(dim=0)) / 2.0
    if not torch.allclose(torch.as_tensor(model_state["scene_center"]).cpu(), expected_center, atol=1.0e-6, rtol=0.0):
        raise ValueError("current cache box disagrees with saved model scene_center")
    if not torch.allclose(torch.as_tensor(model_state["scene_half_extent"]).cpu(), expected_half, atol=1.0e-6, rtol=0.0):
        raise ValueError("current cache box disagrees with saved model scene_half_extent")


def _state_recipe_contract(state: Mapping[str, Any]) -> dict[str, Any]:
    saved_args = _saved_args_or_fail(state)
    return {
        field: _recipe_value(field, saved_args[field])
        for field in (*SAVED_SCIENTIFIC_FIELDS, *SAVED_EVALUATION_FIELDS)
    }


def _historical_best(state: Mapping[str, Any]) -> tuple[int, float]:
    history = state.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("resume checkpoint lacks validation history needed to resolve selected best")
    rows = []
    for row in history:
        if not isinstance(row, Mapping) or "step" not in row or "val_rel_mse" not in row:
            continue
        metric = float(row["val_rel_mse"])
        if math.isfinite(metric):
            rows.append((int(row["step"]), metric))
    if not rows:
        raise ValueError("resume checkpoint has no finite validation history for selected-best resolution")
    best_step, best_metric = min(rows, key=lambda item: item[1])
    saved_best = float(state.get("best_val_rel_mse", float("nan")))
    if not math.isfinite(saved_best) or not math.isclose(saved_best, best_metric, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(
            "resume checkpoint best_val_rel_mse disagrees with validation history: "
            f"saved={saved_best!r}, history={best_metric!r}"
        )
    return best_step, best_metric


def _validate_best_candidate(
    candidate: Mapping[str, Any],
    resume_state: Mapping[str, Any],
    cache,
    expected_step: int,
    expected_metric: float,
) -> None:
    if candidate.get("model_kind") != resume_state.get("model_kind"):
        raise ValueError("selected-best checkpoint model kind disagrees with resume checkpoint")
    if candidate.get("calibration_mode") != resume_state.get("calibration_mode"):
        raise ValueError("selected-best checkpoint calibration mode disagrees with resume checkpoint")
    if _state_recipe_contract(candidate) != _state_recipe_contract(resume_state):
        raise ValueError("selected-best checkpoint scientific recipe disagrees with resume checkpoint")
    saved_manifest = resume_state.get("cache_manifest")
    candidate_manifest = candidate.get("cache_manifest")
    if not _cache_contract_equal(
        _scientific_cache_contract(saved_manifest, "resume"),
        _scientific_cache_contract(candidate_manifest, "selected-best"),
    ):
        raise ValueError("selected-best checkpoint cache contract disagrees with resume checkpoint")
    if int(candidate.get("step", -1)) != expected_step:
        raise ValueError(
            f"selected-best checkpoint step mismatch: expected {expected_step}, got {candidate.get('step')!r}"
        )
    candidate_metric = float(candidate.get("best_val_rel_mse", float("nan")))
    if not math.isclose(candidate_metric, expected_metric, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(
            "selected-best checkpoint metric mismatch: "
            f"expected {expected_metric!r}, got {candidate_metric!r}"
        )
    _validate_cache_contract(candidate, cache)
    _validate_saved_model_box(candidate, cache)


def _resolve_selected_best(resume_path: Path, state: Mapping[str, Any], cache) -> dict[str, Any] | None:
    history = state.get("history")
    if (not isinstance(history, list) or not history) and not math.isfinite(
        float(state.get("best_val_rel_mse", float("nan")))
    ):
        # A checkpoint with no validation history and no finite best is a
        # continuation before the first selection event, not a claim about a
        # historical best model. The final export still requires a real best.
        return None
    best_step, best_metric = _historical_best(state)
    resume_step = int(state.get("step", -1))
    if best_step == resume_step:
        candidate = state
        source = resume_path
    else:
        source = resume_path.parent / "checkpoint_best.pt"
        if not source.exists():
            raise FileNotFoundError(
                "historical best predates the resume checkpoint, but its adjacent "
                f"checkpoint_best.pt is missing: {source}"
            )
        candidate = torch.load(source, map_location="cpu", weights_only=False)
    _validate_best_candidate(candidate, state, cache, best_step, best_metric)
    return {"payload": candidate, "source": source, "step": best_step, "metric": best_metric}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(args, cache, device: torch.device) -> ComplexSHSonarField:
    corners = torch.as_tensor(cache.corners, dtype=torch.float32)
    scene_min, scene_max = corners.amin(dim=0), corners.amax(dim=0)
    if args.model == "adaptive_rift_sas":
        scene = AdaptivePointSHScene.from_regular_grid(
            args.initial_granularity,
            1.0,
            device,
            max_degree=args.sh_degree,
            init_degree=0,
            init_scale=args.init_scale,
            capacity=args.adaptive_capacity,
            compact_sh_eval=True,
        )
        coefficients = AdaptiveRIFTSASField(
            scene,
            raster_granularity=args.granularity,
            extent=1.0,
            query_chunk=args.query_chunk,
        ).to(device)
    elif args.model == "rift_sas":
        if getattr(args, "grid_shape", None) is not None:
            coefficients = RIFTSASRectangularGrid(
                args.grid_shape,
                1.0,
                device,
                max_degree=args.sh_degree,
                init_scale=args.init_scale,
            )
        else:
            coefficients = RIFTSASGrid(
                args.granularity,
                1.0,
                device,
                max_degree=args.sh_degree,
                init_degree=args.sh_degree,
                init_scale=args.init_scale,
            )
    else:
        coefficients = SHSASField(
            extent=1.0,
            granularity=max(2, args.granularity),
            sh_degree=args.sh_degree,
            hidden_dim=args.hidden_dim,
            hash_levels=args.hash_levels,
            hash_features=args.hash_features,
            hash_base_resolution=args.hash_base_resolution,
            hash_final_resolution=args.hash_final_resolution,
            hash_log2_size=args.hash_log2_size,
            device=device,
        ).to(device)
    return ComplexSHSonarField(
        coefficients, scene_min, scene_max, args.sh_degree, query_chunk=args.query_chunk
    ).to(device)


def select_bins(rng: np.random.Generator, target: np.ndarray, count: int) -> np.ndarray:
    num_bins = target.shape[0]
    if count <= 0 or count >= num_bins:
        return np.arange(num_bins, dtype=np.int64)
    return np.sort(rng.choice(num_bins, size=count, replace=False))


def metric_record(predicted: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    diff = predicted - target
    complex_error_sum = diff.abs().square().sum()
    target_power = target.abs().square().sum()
    target_power_for_ratio = target_power.clamp_min(1.0e-20)
    amplitude_diff = predicted.abs() - target.abs()
    return {
        "rel_mse": complex_error_sum / target_power_for_ratio,
        "l1_real_sum": diff.real.abs().sum(),
        "l1_imag_sum": diff.imag.abs().sum(),
        "l1_mag_sum": amplitude_diff.abs().sum(),
        "mse_real_sum": diff.real.square().sum(),
        "mse_imag_sum": diff.imag.square().sum(),
        "mse_mag_sum": amplitude_diff.square().sum(),
        "complex_error_sum": complex_error_sum,
        "count": torch.tensor(float(diff.numel()), device=diff.device),
        "target_power": target_power,
    }


def select_eval_indices(view_indices, limit: int) -> np.ndarray:
    values = np.asarray(list(view_indices), dtype=np.int64)
    if limit > 0 and values.size > limit:
        selected = np.unique(np.linspace(0, values.size - 1, limit).round().astype(np.int64))
        values = values[selected]
    return values


def select_eval_bins(cache, args) -> np.ndarray:
    if args.eval_bins <= 0:
        return np.arange(cache.num_bins, dtype=np.int64)
    return np.unique(
        np.linspace(0, cache.num_bins - 1, min(args.eval_bins, cache.num_bins))
        .round()
        .astype(np.int64)
    )


def render_one(
    model, calibration, cache, ping: int, bins: np.ndarray, args, device,
    *, allow_calibration_init: bool = False, probe_next_band: bool = False,
):
    bin_tensor = torch.as_tensor(bins, dtype=torch.long, device=device)
    radii = torch.as_tensor(cache.radii, dtype=torch.float32, device=device)
    tx = torch.as_tensor(cache.tx_coords[ping], dtype=torch.float32, device=device)
    rx = torch.as_tensor(cache.rx_coords[ping], dtype=torch.float32, device=device)
    corners = torch.as_tensor(cache.corners, dtype=torch.float32, device=device)
    tx_direction = None
    if cache.tx_vecs is not None:
        tx_direction = torch.as_tensor(cache.tx_vecs[ping], dtype=torch.float32, device=device)
    target = torch.as_tensor(np.asarray(cache.weights[ping, bins]), dtype=torch.complex64, device=device)
    predicted_raw, aux = render_sas_bins(
        model,
        radii,
        tx,
        rx,
        corners,
        num_rays=args.num_rays,
        opacity_scale=args.opacity_scale,
        lambertian_ratio=args.lambertian_ratio,
        normal_step=args.normal_step,
        tx_direction=tx_direction,
        beamwidth_deg=args.beamwidth_deg,
        point_at_center=True,
        transmit_from_tx=True,
        output_bin_indices=bin_tensor,
        mean_normalize_opacity=bool(args.opacity_normalize),
        sh_direction=args.sh_direction,
        probe_next_band=probe_next_band,
        ray_chunk=getattr(args, "ray_chunk", 0),
    )
    predicted_raw = predicted_raw * args.signal_scale
    if allow_calibration_init:
        calibration.maybe_initialize(predicted_raw, target)
    predicted = calibration(predicted_raw)
    loss = torch.nn.functional.mse_loss(predicted.real, target.real) + torch.nn.functional.mse_loss(
        predicted.imag, target.imag
    )
    aux["calibration_raw"] = predicted_raw.detach()
    aux["calibration_target"] = target.detach()
    aux["calibration_predicted"] = predicted.detach()
    return loss, metric_record(predicted, target), aux


@torch.no_grad()
def calibration_diagnostics(
    aux: Dict[str, torch.Tensor], calibration: torch.nn.Module
) -> Dict[str, float]:
    raw = aux["calibration_raw"]
    target = aux["calibration_target"]
    predicted = aux["calibration_predicted"]

    def rms(value: torch.Tensor) -> float:
        return float(value.abs().square().mean().sqrt().item())

    raw_norm = torch.linalg.vector_norm(raw)
    target_norm = torch.linalg.vector_norm(target)
    denominator = raw_norm * target_norm
    if bool(denominator == 0):
        raw_corr_abs = float("nan")
    else:
        raw_corr_abs = float(((raw.conj() * target).sum().abs() / denominator).item())
    one = torch.ones((), dtype=torch.complex64, device=raw.device)
    gain = calibration(one)
    transmittance = aux["transmittance"].detach()
    lambertian = aux["lambertian"].detach()
    return {
        "raw_rms": rms(raw),
        "target_rms": rms(target),
        "pred_rms": rms(predicted),
        "raw_corr_abs": raw_corr_abs,
        "gain_abs": float(gain.abs().item()),
        "gain_phase": float(torch.angle(gain).item()),
        "T_all_min": float(transmittance.min().item()),
        "T_all_mean": float(transmittance.mean().item()),
        "T_all_lt_1e3": float((transmittance < 1.0e-3).float().mean().item()),
        "lambert_positive": float((lambertian > 0).float().mean().item()),
    }


@torch.no_grad()
def evaluate(model, calibration, cache, view_indices, args, device) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    calibration.eval()
    pings = select_eval_indices(view_indices, args.eval_pings)
    expected_views = len(pings)
    bins = select_eval_bins(cache, args)
    rel_numerator = 0.0
    denominator = 0.0
    l1_totals = {key: 0.0 for key in ("l1_real", "l1_imag", "l1_mag")}
    mse_totals = {key: 0.0 for key in ("mse_real", "mse_imag", "mse_mag")}
    count_total = 0.0
    for ping in pings:
        _loss, metrics, _ = render_one(model, calibration, cache, int(ping), bins, args, device)
        rel_numerator += float(metrics["complex_error_sum"])
        denominator += float(metrics["target_power"])
        count = float(metrics["count"])
        for key, source in (("l1_real", "l1_real_sum"), ("l1_imag", "l1_imag_sum"), ("l1_mag", "l1_mag_sum")):
            l1_totals[key] += float(metrics[source])
        for key, source in (("mse_real", "mse_real_sum"), ("mse_imag", "mse_imag_sum"), ("mse_mag", "mse_mag_sum")):
            mse_totals[key] += float(metrics[source])
        count_total += count
    if was_training:
        model.train()
        calibration.train()
    if denominator <= 0.0:
        raise ValueError("relative MSE is undefined: the evaluated cohort has zero target power")
    totals = {
        "rel_mse": rel_numerator / denominator,
        "metric_convention": "rel_mse_is_complex_residual; magnitude_fields_are_amplitude_residuals",
    }
    totals.update({key: value / max(count_total, 1.0) for key, value in l1_totals.items()})
    totals.update({key: value / max(count_total, 1.0) for key, value in mse_totals.items()})
    totals["complete"] = float(len(pings) == expected_views)
    totals["views"] = float(len(pings))
    return totals


def atomic_torch_save(payload: Dict[str, object], path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint(model, calibration, optimizer, step, best_val, rng, history, args, cache):
    scene = _adaptive_scene(model)
    return {
        "sas_contract_version": CONTRACT_VERSION,
        "step": int(step),
        "epoch": int(step),
        "best_val_rel_mse": float(best_val),
        "model_kind": args.model,
        "calibration_mode": args.calibration_mode,
        "model_state_dict": model.state_dict(),
        "calibration_state_dict": calibration.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "history": history,
        "args": vars(args),
        "cache_manifest": cache.manifest,
        "geometry_truth_used_for_training": False,
        "shared_operator": "reed_ellipsoidal_cleanroom_v1",
        "parameter_counts": {
            "allocated_scalars": sum(parameter.numel() for parameter in model.parameters()),
            "active_scalars": scene.active_parameter_count() if scene is not None else None,
            "active_points": int(scene.active_mask.sum().item()) if scene is not None else None,
        },
    }


def save_history(history, path: Path) -> None:
    if not history:
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    os.replace(temporary, path)


def _adaptive_scene(model):
    field = getattr(model, "coefficient_field", None)
    if isinstance(field, AdaptiveRIFTSASField):
        return field.underlying_scene
    return None


def _optimizer_for_model(model, calibration, args):
    scene = _adaptive_scene(model)
    adam_eps = getattr(args, "adam_eps", 1e-8)
    if scene is None:
        return torch.optim.Adam(
            list(model.parameters()) + list(calibration.parameters()), lr=args.lr, eps=adam_eps
        )
    return torch.optim.Adam(
        [
            {"params": [scene.w_re, scene.w_im], "lr": args.coefficient_lr},
            {"params": [scene.delta_raw], "lr": args.position_lr},
            {"params": list(calibration.parameters()), "lr": args.coefficient_lr},
        ],
        eps=adam_eps,
    )


def _model_parameters(model, calibration):
    return list(model.parameters()) + list(calibration.parameters())


def _diagnostic_hook(observer: object | None, name: str, **context: Any) -> None:
    """Call one opt-in research observer hook without touching the default path."""

    if observer is None:
        return
    hook = getattr(observer, name, None)
    if hook is not None:
        hook(**context)


def main(
    argv: Optional[Sequence[str]] = None, *, diagnostic_observer: object | None = None
) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    explicit_fields = _explicit_cli_fields(raw_argv)
    args = parse_args(raw_argv)
    profile_applied_fields = _apply_profile(args, explicit_fields)
    if args.opacity_normalize is None:
        args.opacity_normalize = False
    output = Path(args.checkpoint_root) / args.checkpoint_name
    output.mkdir(parents=True, exist_ok=True)
    resume = Path(args.resume) if args.resume else output / "checkpoint_latest.pt"
    if args.resume is not None and not resume.exists():
        raise FileNotFoundError(f"explicit --resume checkpoint does not exist: {resume}")
    if args.eval_only and not resume.exists():
        raise ValueError("--eval-only requires an existing --resume checkpoint")
    state = (
        torch.load(resume, map_location="cpu", weights_only=False)
        if resume.exists() else None
    )
    if state is not None and state.get("model_kind") != args.model:
        raise ValueError("checkpoint model kind does not match --model")
    if args.query_chunk <= 0:
        raise ValueError("--query-chunk must be positive")
    if state is not None:
        _reconcile_saved_recipe(
            args,
            state,
            explicit_fields,
            profile_applied_fields,
            eval_only=bool(args.eval_only),
        )
        if args.steps < int(state["step"]):
            raise ValueError(
                f"terminal --steps={args.steps} is below resumed checkpoint step {int(state['step'])}"
            )
    if args.grid_shape is not None and args.model != "rift_sas":
        raise ValueError("--grid-shape is allowed only with --model rift_sas")
    if args.ray_chunk < 0:
        raise ValueError("--ray-chunk must be nonnegative")
    if args.pings_per_step < 1:
        raise ValueError("--pings-per-step must be >= 1")
    if not 0.0 <= args.gain_init_corr_threshold <= 1.0:
        raise ValueError("--gain-init-corr-threshold must be in [0, 1]")
    if args.model in ("adaptive_rift_sas", "sh_sas"):
        if state is None:
            args.require_explicit_splits = True
        if not args.require_explicit_splits:
            raise ValueError("this comparison requires an AirSAS cache with explicit train/validation/test splits")
    if args.model == "adaptive_rift_sas":
        if args.sh_degree != 3:
            raise ValueError("adaptive_rift_sas comparison is fixed to SH degree 3")
        if args.probe_every <= 0 or args.refine_every <= 0:
            raise ValueError("adaptive probe/refinement intervals must be positive")
    args.calibration_mode = resolve_calibration_mode(args.model, args.calibration_mode, state)
    seed_all(args.seed)
    device = torch.device(args.device)
    cache = load_sas_cache(args.cache)
    if state is not None:
        _validate_cache_contract(state, cache)
        _validate_saved_model_box(state, cache)
    if args.require_explicit_splits and not cache.has_explicit_splits:
        raise ValueError("this comparison requires an AirSAS cache with explicit train/validation/test splits")
    if cache.has_explicit_splits:
        if args.max_pings > 0:
            raise ValueError("--max-pings cannot truncate a cache with explicit train/validation/test splits")
        train_indices = cache.train_indices
        validation_indices = cache.validation_indices
        test_indices = cache.test_indices
    else:
        if args.model in ("adaptive_rift_sas", "sh_sas"):
            raise ValueError("this comparison requires an AirSAS cache with explicit train/validation/test splits")
        num_pings = cache.num_pings if args.max_pings <= 0 else min(args.max_pings, cache.num_pings)
        train_indices = np.arange(num_pings, dtype=np.int64)
        validation_indices = np.arange(num_pings, dtype=np.int64)
        test_indices = np.empty(0, dtype=np.int64)
    if train_indices.size == 0 or validation_indices.size == 0:
        raise ValueError("training and validation cohorts must be non-empty")

    if state is not None and not args.eval_only and diagnostic_observer is None:
        selected_best = _resolve_selected_best(resume, state, cache)
        if selected_best is not None:
            selected_destination = output / "checkpoint_best.pt"
            selected_source = Path(selected_best["source"])
            if selected_source.resolve() != selected_destination.resolve():
                atomic_torch_save(selected_best["payload"], selected_destination)

    model = build_model(args, cache, device)
    calibration = build_calibration(args.calibration_mode, device, args.gain_init_corr_threshold)
    optimizer = _optimizer_for_model(model, calibration, args)
    rng = np.random.default_rng(args.seed)
    start, best_val, history = 0, float("inf"), []
    if state is not None:
        model.load_state_dict(state["model_state_dict"])
        calibration.load_state_dict(state["calibration_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start = int(state["step"])
        best_val = float(state["best_val_rel_mse"])
        history = list(state.get("history", []))
        rng.bit_generator.state = state["rng_state"]
        torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        print(f"Resumed {args.model} at step {start}", flush=True)
    print(f"calibration_mode={args.calibration_mode}", flush=True)
    _diagnostic_hook(
        diagnostic_observer,
        "on_restore",
        model=model,
        calibration=calibration,
        optimizer=optimizer,
        rng=rng,
        cache=cache,
        args=args,
        device=device,
        state=state,
        start=start,
        best_val=best_val,
        history=history,
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
    )

    if args.eval_only:
        role_indices = validation_indices if args.evaluation_role == "validation" else test_indices
        if role_indices.size == 0:
            raise ValueError(f"evaluation role {args.evaluation_role!r} is empty")
        metrics = evaluate(model, calibration, cache, role_indices, args, device)
        selected_eval_indices = select_eval_indices(role_indices, args.eval_pings)
        selected_eval_bins = select_eval_bins(cache, args)
        atomic_json(
            {
                **metrics,
                "evaluation_role": args.evaluation_role,
                "source_ids": cache.source_ids[selected_eval_indices].tolist(),
                "selected_source_ids": cache.source_ids[selected_eval_indices].tolist(),
                "selected_bin_ids": selected_eval_bins.tolist(),
                "checkpoint_step": start,
            },
            output / f"{args.evaluation_role}_eval.json",
        )
        print(f"{args.evaluation_role} evaluation rel_mse={metrics['rel_mse']:.6e}", flush=True)
        return

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    model.train()
    started = time.time()
    for step in range(start, args.steps):
        current = step + 1
        optimizer.zero_grad(set_to_none=True)
        scene = _adaptive_scene(model)
        prior_delta_grad = torch.zeros_like(scene.delta_raw) if scene is not None else None
        for _ping_index in range(args.pings_per_step):
            ping = int(rng.choice(train_indices))
            target_np = np.asarray(cache.weights[ping])
            bins = select_bins(rng, target_np, args.max_bins)
            loss, metrics, aux = render_one(
                model, calibration, cache, ping, bins, args, device,
                allow_calibration_init=True,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {current}")
            (loss / args.pings_per_step).backward()
            if scene is not None:
                # Incremental, not cumulative: accumulate_refinement_data_stats
                # wants the gradient contributed by *this* ping alone (its own
                # docstring warns against "dependence on optimizer
                # accumulation"), but pings within a step share one
                # zero_grad(), so delta_raw.grad keeps growing across them.
                current_delta_grad = (
                    scene.delta_raw.grad.detach().clone()
                    if scene.delta_raw.grad is not None else torch.zeros_like(scene.delta_raw)
                )
                data_delta = current_delta_grad - prior_delta_grad
                prior_delta_grad = current_delta_grad
                next_re = next_im = None
                if current % args.probe_every == 0 and (scene.active_mask & scene.order.lt(scene.max_degree)).any():
                    probe_loss, _probe_metrics, _probe_aux = render_one(
                        model, calibration, cache, ping, bins, args, device,
                        allow_calibration_init=False,
                        probe_next_band=True,
                    )
                    next_re, next_im = torch.autograd.grad(
                        probe_loss, (scene.w_re, scene.w_im), allow_unused=True
                    )
                    if next_re is None:
                        next_re = torch.zeros_like(scene.w_re)
                    if next_im is None:
                        next_im = torch.zeros_like(scene.w_im)
                scene.accumulate_refinement_data_stats(data_delta, next_re, next_im)
        log_now = current == 1 or current % args.log_every == 0
        diagnostics = calibration_diagnostics(aux, calibration) if log_now else None
        grad_norm = torch.nn.utils.clip_grad_norm_(
            _model_parameters(model, calibration), args.grad_clip
        )
        will_refine = scene is not None and current % args.refine_every == 0
        _diagnostic_hook(
            diagnostic_observer,
            "on_before_optimizer",
            model=model,
            calibration=calibration,
            optimizer=optimizer,
            cache=cache,
            args=args,
            device=device,
            scene=scene,
            step=current,
            ping=ping,
            bins=bins,
            loss=loss,
            metrics=metrics,
            aux=aux,
            grad_norm=grad_norm,
            will_refine=will_refine,
        )
        optimizer.step()
        _diagnostic_hook(
            diagnostic_observer,
            "on_after_optimizer",
            model=model,
            calibration=calibration,
            optimizer=optimizer,
            cache=cache,
            args=args,
            device=device,
            scene=scene,
            step=current,
            ping=ping,
            bins=bins,
            loss=loss,
            metrics=metrics,
            aux=aux,
            grad_norm=grad_norm,
            will_refine=will_refine,
        )
        if log_now:
            print(
                f"step {current}/{args.steps} loss={float(loss):.6e} rel_mse={float(metrics['rel_mse']):.6e} "
                f"grad={float(grad_norm):.3e} rays={int(aux['actual_rays'])}",
                flush=True,
            )
            diagnostic_text = " ".join(
                f"{key}={diagnostics[key]:.6e}" for key in (
                    "raw_rms", "target_rms", "pred_rms", "raw_corr_abs", "gain_abs",
                    "gain_phase", "T_all_min", "T_all_mean", "T_all_lt_1e3", "lambert_positive",
                )
            )
            print(f"sonar_diagnostics {diagnostic_text}", flush=True)
        if scene is not None and current % args.refine_every == 0:
            snapshot = scene.refinement_snapshot(
                max_level=args.split_max_level,
                min_spatial_exposure=1,
                min_angular_exposure=1,
                cooldown_events=args.cooldown_events,
                child_maturity_events=args.child_maturity_events,
            )
            _diagnostic_hook(
                diagnostic_observer,
                "on_before_refinement",
                model=model,
                calibration=calibration,
                optimizer=optimizer,
                cache=cache,
                args=args,
                device=device,
                scene=scene,
                step=current,
                snapshot=snapshot,
            )
            _n_split, _n_angular, _active, report = scene.apply_refinement_snapshot(
                snapshot,
                spatial_fraction=args.spatial_fraction,
                angular_fraction=args.angular_fraction,
                max_level=args.split_max_level,
                optimizer=optimizer,
                max_active=args.max_active,
            )
            _diagnostic_hook(
                diagnostic_observer,
                "on_after_refinement",
                model=model,
                calibration=calibration,
                optimizer=optimizer,
                cache=cache,
                args=args,
                device=device,
                scene=scene,
                step=current,
                snapshot=snapshot,
                n_split=_n_split,
                n_angular=_n_angular,
                active=_active,
                report=report,
            )
            print(report, flush=True)
        if current % args.eval_every == 0 or current == args.steps:
            val = evaluate(model, calibration, cache, validation_indices, args, device)
            history.append(
                {
                    "step": current,
                    "train_loss": float(loss.detach()),
                    "train_rel_mse": float(metrics["rel_mse"].detach()),
                    "val_rel_mse": val["rel_mse"],
                    "val_l1_real": val["l1_real"],
                    "val_l1_imag": val["l1_imag"],
                    "val_l1_mag": val["l1_mag"],
                    "elapsed_seconds": time.time() - started,
                }
            )
            print(f"validation step={current} rel_mse={val['rel_mse']:.6e} views={int(val['views'])}", flush=True)
            if val["complete"] and val["rel_mse"] < best_val:
                best_val = val["rel_mse"]
                atomic_torch_save(
                    checkpoint(model, calibration, optimizer, current, best_val, rng, history, args, cache),
                    output / "checkpoint_best.pt",
                )
            save_history(history, output / "history.csv")
        if current % args.checkpoint_every == 0 or current == args.steps or STOP_REQUESTED:
            atomic_torch_save(
                checkpoint(model, calibration, optimizer, current, best_val, rng, history, args, cache),
                output / "checkpoint_latest.pt",
            )
        if STOP_REQUESTED:
            atomic_json(
                {"done": False, "step": current, "reason": "signal", "model": args.model},
                output / "status.json",
            )
            return

    if diagnostic_observer is not None:
        _diagnostic_hook(
            diagnostic_observer,
            "on_finish",
            model=model,
            calibration=calibration,
            optimizer=optimizer,
            rng=rng,
            cache=cache,
            args=args,
            device=device,
            step=args.steps,
            best_val=best_val,
            history=history,
            checkpoint_writer=checkpoint,
            atomic_save=atomic_torch_save,
        )
        return

    final = checkpoint(model, calibration, optimizer, args.steps, best_val, rng, history, args, cache)
    atomic_torch_save(final, output / "checkpoint_final.pt")
    selected_path = output / "checkpoint_best.pt"
    if not selected_path.exists():
        raise RuntimeError(
            "no validated checkpoint_best.pt is available for final selection; "
            "the final model cannot be cited under a historical best metric"
        )
    selected = torch.load(selected_path, map_location=device, weights_only=False)
    selected_step, selected_metric = _historical_best(final)
    _validate_best_candidate(selected, final, cache, selected_step, selected_metric)
    model.load_state_dict(selected["model_state_dict"])
    calibration.load_state_dict(selected["calibration_state_dict"])
    model.eval()
    calibration.eval()
    voxels = torch.as_tensor(cache.voxels, dtype=torch.float32, device=device)
    density = model.dense_density(voxels).numpy().astype(np.float32)
    np.save(output / "density.npy", density)
    if test_indices.size:
        test_metrics = evaluate(model, calibration, cache, test_indices, args, device)
        atomic_json(test_metrics, output / "test_metrics.json")
    atomic_json(
        {
            "selected_checkpoint": str(selected_path.name if selected_path.exists() else "checkpoint_final.pt"),
            "selected_checkpoint_step": selected_step,
            "selected_checkpoint_rel_mse": selected_metric,
            "best_val_rel_mse": selected_metric,
            "test_role_used_after_selection": bool(test_indices.size),
            "selected_validation_source_ids": cache.source_ids[
                select_eval_indices(validation_indices, args.eval_pings)
            ].tolist(),
            "selected_test_source_ids": cache.source_ids[
                select_eval_indices(test_indices, args.eval_pings)
            ].tolist() if test_indices.size else [],
            "allocated_parameter_scalars": sum(parameter.numel() for parameter in model.parameters()),
            "active_parameter_scalars": (
                _adaptive_scene(model).active_parameter_count()
                if _adaptive_scene(model) is not None else None
            ),
            "elapsed_seconds": time.time() - started,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        },
        output / "selected_readout.json",
    )
    atomic_json(
        {"done": True, "step": args.steps, "best_val_rel_mse": best_val, "model": args.model},
        output / "status.json",
    )


if __name__ == "__main__":
    main()
