#!/usr/bin/env python
"""Evaluate the selected GeRaF B7873200 checkpoint in both common domains.

This is the production counterpart of the older PACE-only GeRaF complex
response evaluator.  It deliberately binds one frozen checkpoint to the
sealed sphere10k validation role and reports two pooled metrics from the same
rendered complex response:

* native complex held-out-signal relative MSE, and
* Radar Fields' fixed 60 dB normalized matched-range-power relative MSE.

The GeRaF renderer must run under ``torch.enable_grad()`` because its SDF
normal calculation differentiates through sampled points even during eval.
The response is detached only after rendering, before metric aggregation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_geraf as tg  # noqa: E402
from rift.config import cc  # noqa: E402
from rift.forward_operator import get_kvector  # noqa: E402
from rift.geraf import GeRaFModel, PrimaryRaySamples  # noqa: E402
from rift.geraf_b7873200_acquisition import (  # noqa: E402
    b7873200_operator_frequency_grid_hz,
    acquisition_records_equal,
    load_b7873200_acquisition_record,
    validate_b7873200_acquisition_record,
    validate_b7873200_operator_frequency_grid,
)
from rift.geraf_b7873200_protocol import (  # noqa: E402
    B787_3200_ACQUISITION_SCHEMA,
    B787_3200_CANONICAL_MANIFEST_PATH,
    B787_3200_CANONICAL_NPZ_PATH,
    B787_3200_CACHE_ACQUISITION_FILENAME,
    B787_3200_CACHE_RECIPE_FILENAME,
    B787_3200_CACHE_STATS_FILENAME,
    B787_3200_MANIFEST_NAME,
    B787_3200_NUM_TEST,
    B787_3200_NUM_TRAIN,
    B787_3200_NUM_UNUSED,
    B787_3200_NUM_VALIDATION,
    B787_3200_NUM_VIEWS,
)
from rift.geraf_b7873200_source import (  # noqa: E402
    load_b7873200_metadata_source,
)
from rift.geraf_signal_operator import (  # noqa: E402
    bistatic_pair_positions,
    matched_filter_from_response_range,
    pairwise_range_forward_operator,
)
from rift.radar_fields_dataset import (  # noqa: E402
    load_radar_fields_npz,
    normalize_power_db,
    range_bin_centers,
    response_view_to_range_power,
    restrict_radar_fields_response_views,
    scene_range_mask,
)
from scripts.b787_common_eval_metrics import (  # noqa: E402
    RF_GRID_EXTENT_M,
    RF_RANGE_MARGIN_M,
    diagnostic_relative_mse,
    make_rf_midpoint_grid,
    partition_squared_error,
    pooled_relative_mse,
    rf_supported_mask,
)
from scripts.eval_b787_range_power import (  # noqa: E402
    _sealed_role_indices,
    validate_normalization_stats,
)
from scripts.prepare_geraf_b7873200_targets import _matched_filter_amplitude  # noqa: E402


DEFAULT_CHECKPOINT = (
    "/storage/scratch1/1/dbao31/rift_b7873200_geraf_fullscale_v1/fit/"
    "checkpoint_best.pth.tar"
)
DEFAULT_NPZ = (
    "/storage/project/r-jromberg3-0/dbao31/RIFT/data/"
    "b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz"
)
DEFAULT_ROLE_MANIFEST = "/storage/scratch1/1/dbao31/rift_round8b_impl_20260810/splits/round8b/b78710k_interp_seed42_train3200_val1000_test1000_v1.json"
DEFAULT_CACHE_ROOT = "/storage/scratch1/1/dbao31/rift_b7873200_geraf_fullscale_v1/prepared_targets"
DEFAULT_POWER_STATS = "/storage/scratch1/1/dbao31/rift_b7873200_radar_fields_production_v1/attempt-12941397/rf_b7873200_production_v1/radar_fields_power_stats.json"
DEFAULT_OUT_DIR = "/storage/scratch1/1/dbao31/rift_b7873200_geraf_fullscale_v1/common_signal_evaluation_v1"

SELECTED_CHECKPOINT_STEP = 37_000
EXPECTED_SELECTED_MF_MSE = 0.0011093310088487512
EXPECTED_SELECTED_NATIVE_REL_MSE = 0.16445661583652937
EXPECTED_POWER_PEAK = 1.1800956040705975e-07
EXPECTED_POWER_DYNAMIC_RANGE_DB = 60.0
EXPECTED_VALIDATION_SAMPLES = 1000 * 16 * 16 * 600
EXPECTED_POWER_ELEMENT_COUNT = 3_328_000
EXPECTED_POWER_TARGET_SQUARED_NORM = 411925.32444763184
EXPECTED_TEST_POWER_TARGET_SQUARED_NORM = 419975.4931983948
EXPECTED_NATIVE_MF_SAMPLES = 1000 * 32 * 32 * 32
RESPONSE_SHAPE = (B787_3200_NUM_VIEWS, 16, 16, 1, 600)
VIEW_RESPONSE_SHAPE = RESPONSE_SHAPE[1:]
EPS = 1.0e-30


def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **dict(arrays))
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _finite_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _require_close(label: str, observed: object, expected: float, *, abs_tol: float = 1.0e-15) -> None:
    value = _finite_float(observed, label)
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=abs_tol):
        raise ValueError(f"{label}={value!r} disagrees with frozen value {expected!r}")


def _ordered_role_ids(identity: Mapping[str, object], role: str, expected_count: int) -> tuple[int, ...]:
    roles = identity.get("role_ids")
    if not isinstance(roles, Mapping):
        raise ValueError("B7873200 sealed identity lacks role IDs")
    raw = roles.get(role)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"B7873200 sealed identity role {role!r} is not an integer sequence")
    values = tuple(int(value) for value in raw)
    if len(values) != expected_count or len(set(values)) != expected_count:
        raise ValueError(f"B7873200 sealed identity role {role!r} has invalid cardinality")
    if any(value < 0 or value >= B787_3200_NUM_VIEWS for value in values):
        raise ValueError(f"B7873200 sealed identity role {role!r} contains an out-of-range ID")
    return values


def _sealed_contract(identity: Mapping[str, object]) -> dict[str, object]:
    """Adapt the GeRaF identity to the common stats validator's contract."""
    from rift.rift_dataset import collection_contract
    collection = collection_contract(identity)
    if collection is not None:
        return collection
    return {
        "response_shape": list(RESPONSE_SHAPE),
        "response_dtype": "complex64",
        "role_ids": {
            "train": list(_ordered_role_ids(identity, "train", B787_3200_NUM_TRAIN)),
            "validation": list(_ordered_role_ids(identity, "validation", B787_3200_NUM_VALIDATION)),
            "reserved_test": list(_ordered_role_ids(identity, "reserved_test", B787_3200_NUM_TEST)),
            "unused": list(_ordered_role_ids(identity, "unused", B787_3200_NUM_UNUSED)),
        },
    }


def _validate_checkpoint_selection(
    checkpoint: Mapping[str, object], cache: Any, args: argparse.Namespace
) -> Mapping[str, object]:
    from rift.rift_dataset import collection_contract, validate_checkpoint_object
    collection = collection_contract(cache.sealed_identity)
    selected_step = SELECTED_CHECKPOINT_STEP
    selected_mse = EXPECTED_SELECTED_MF_MSE
    if collection is not None:
        validate_checkpoint_object(checkpoint, collection)
        selected_step = int(checkpoint.get("step", -1))
        selected_mse = _finite_float(checkpoint.get("best_val_mse"), "checkpoint best validation MSE")
        if selected_step <= 0 or selected_step > 50000 or selected_step % 1000 or selected_mse < 0:
            raise ValueError("Collection checkpoint must be a validation-selected checkpoint on the frozen cadence")
    if int(checkpoint.get("step", -1)) != selected_step:
        raise ValueError("the common evaluation requires the frozen checkpoint_best step 37000")
    if checkpoint.get("pending_validation_step") is not None:
        raise ValueError("selected checkpoint retains a pending validation obligation")
    history = checkpoint.get("history")
    if not isinstance(history, list):
        raise ValueError("selected checkpoint lacks validation history")
    if collection is None:
        expected_steps = list(range(1_000, SELECTED_CHECKPOINT_STEP + 1, 1_000))
    else:
        expected_steps = list(range(1_000, selected_step + 1, 1_000))
    observed_steps: list[int] = []
    expected_voxels = B787_3200_NUM_VALIDATION * 32 * 32 * 32
    for number, raw_row in enumerate(history):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"checkpoint history row {number} is not an object")
        row_step = int(raw_row.get("step", -1))
        if row_step <= 0 or row_step % 1_000 != 0:
            raise ValueError("checkpoint validation history is not on the 1000-step cadence")
        if int(float(raw_row.get("views", -1))) != B787_3200_NUM_VALIDATION:
            raise ValueError("every checkpoint history row must cover all 1000 validation views")
        if int(float(raw_row.get("voxels", -1))) != expected_voxels:
            raise ValueError("every checkpoint history row must cover 32,768,000 validation voxels")
        observed_steps.append(row_step)
    if observed_steps != expected_steps:
        raise ValueError(f"checkpoint validation history must cover every 1000-step row through step {selected_step}")
    rows = [row for row in history if isinstance(row, Mapping) and int(row.get("step", -1)) == selected_step]
    if len(rows) != 1:
        raise ValueError(f"selected checkpoint lacks exactly one step-{selected_step} validation row")
    row = rows[0]
    if int(float(row.get("views", -1))) != B787_3200_NUM_VALIDATION:
        raise ValueError("selected checkpoint best row does not cover all 1000 validation views")
    _require_close("checkpoint best validation MSE", checkpoint.get("best_val_mse"), selected_mse)
    _require_close("selected checkpoint MF magnitude MSE", row.get("mf_magnitude_mse"), selected_mse)
    if collection is not None and _finite_float(row.get("mf_magnitude_relative_mse"), "selected native MF relative MSE") < 0:
        raise ValueError("Selected native MF relative MSE must be nonnegative")
    _require_close(
        "selected checkpoint native MF magnitude RelMSE",
        row.get("mf_magnitude_relative_mse"),
        (_finite_float(row.get("mf_magnitude_relative_mse"), "selected native MF relative MSE")
         if collection is not None else EXPECTED_SELECTED_NATIVE_REL_MSE),
    )
    observed_mses = [_finite_float(item.get("mf_magnitude_mse"), "checkpoint history MF magnitude MSE") for item in history if isinstance(item, Mapping)]
    if not observed_mses or not math.isclose(min(observed_mses), selected_mse, rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError(f"step {selected_step} is not proven to be the best MF-magnitude history row")

    identity = checkpoint.get("run_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("selected checkpoint lacks run_identity provenance")
    if not isinstance(checkpoint.get("model_config"), Mapping) or not isinstance(checkpoint.get("model_state_dict"), Mapping):
        raise ValueError("selected checkpoint lacks a loadable GeRaF model")
    if checkpoint["model_config"] != tg.model_config_from_args(args):
        raise ValueError("checkpoint model configuration disagrees with its declared recipe")
    if not acquisition_records_equal(checkpoint.get("acquisition_record", {}), cache.acquisition_record):
        raise ValueError("checkpoint acquisition record disagrees with the prepared cache")
    expected_identity = tg.run_identity(args=args, cache=cache, model_config=checkpoint["model_config"])
    if dict(identity) != expected_identity:
        raise ValueError("checkpoint run_identity disagrees with the current trainer/cache/model contract")
    if int(identity.get("steps", -1)) != 50_000 or int(identity.get("seed", -1)) != 42:
        raise ValueError("selected checkpoint is not the frozen 50000-step seed-42 run")
    validation = identity.get("validation")
    if not isinstance(validation, Mapping) or int(validation.get("every_optimizer_steps", -1)) != 1000:
        raise ValueError("selected checkpoint has the wrong validation cadence")
    if validation.get("selection_metric") != "mf_magnitude_mse":
        raise ValueError("selected checkpoint has the wrong selection metric")
    grid = identity.get("grid")
    if not isinstance(grid, Mapping):
        raise ValueError("selected checkpoint lacks the frozen scene grid contract")
    for label, value, expected in (
        ("scene_extent", grid.get("scene_extent"), 0.15),
        ("aperture_scale", grid.get("aperture_scale"), 1.0),
    ):
        _require_close(f"checkpoint grid {label}", value, expected)
    for label in ("n_azimuth", "n_elevation", "n_depth"):
        if int(grid.get(label, -1)) != 32:
            raise ValueError(f"checkpoint grid {label} is not 32")
    render = identity.get("render")
    if not isinstance(render, Mapping):
        raise ValueError("selected checkpoint lacks the renderer contract")
    if render.get("lensless_correction") is not True or render.get("detach_start_cdf") is not True:
        raise ValueError("GeRaF renderer contract must use lensless=true and detach_start_cdf=true")
    _require_close("checkpoint directional exponent", render.get("directional_exponent"), 1.0)
    _require_close("checkpoint min distance", render.get("min_distance"), 1.0e-6, abs_tol=1.0e-18)
    operator = identity.get("operator")
    if not isinstance(operator, Mapping):
        raise ValueError("selected checkpoint lacks the signal-operator contract")
    if operator.get("backend") != "range_nufft" or operator.get("compute_dtype") != "float64":
        raise ValueError("selected checkpoint operator is not float64 range-NUFFT")
    if operator.get("phase_sign") != -1.0 or operator.get("oversample") != 2 or operator.get("kernel_width") != 20:
        raise ValueError("selected checkpoint has the wrong phase/range-NUFFT contract")
    if operator.get("pair_chunk") != 32 or operator.get("point_chunk") != 4096:
        raise ValueError("selected checkpoint has the wrong signal chunk contract")
    if checkpoint.get("gain_state_dict") is not None:
        raise ValueError("GeRaF common evaluation forbids an extra fitted complex gain")

    if identity.get("sealed_protocol_identity") != cache.sealed_identity:
        raise ValueError("checkpoint sealed-role provenance disagrees with the prepared cache")
    if checkpoint.get("cache_recipe") != cache.recipe:
        raise ValueError("checkpoint cache recipe disagrees with the prepared target cache")
    if checkpoint.get("target_manifest") != cache.target_manifest:
        raise ValueError("checkpoint target manifest disagrees with the prepared target cache")
    if checkpoint.get("target_stats") != cache.stats:
        raise ValueError("checkpoint target stats disagree with the prepared target cache")
    del args
    return row


def flatten_ge_ra_f_response(response: torch.Tensor) -> torch.Tensor:
    """Align GeRaF ``[F,Rx,Tx]`` with source ``[Tx,Rx,F]`` order."""

    if response.ndim != 3:
        raise ValueError(f"GeRaF response must have shape [F,Rx,Tx], got {tuple(response.shape)}")
    if tuple(response.shape) != (600, 16, 16):
        raise ValueError(f"GeRaF response must have shape [600,16,16], got {tuple(response.shape)}")
    return response.permute(2, 1, 0).contiguous().reshape(16 * 16, 600)


def render_complex_response(
    model: GeRaFModel,
    view: Any,
    frequencies: torch.Tensor,
    kvector: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Render only GeRaF's pre-matched-filter complex response."""

    tx_positions = view.tx_positions.detach()
    rx_positions = view.rx_positions.detach()
    integration_samples = tg.render_samples_for_view(view, args)
    fixed_samples = PrimaryRaySamples(
        ray_origins=integration_samples.ray_origins.detach(),
        primary_direction=integration_samples.primary_direction.detach(),
        depths=integration_samples.depths.detach(),
        depth_deltas=integration_samples.depth_deltas.detach(),
        points=integration_samples.points.detach(),
        depth_edges=integration_samples.depth_edges,
    )
    tx_pair, rx_pair = bistatic_pair_positions(tx_positions, rx_positions)
    with torch.enable_grad():
        volume = model.render_volume(
            fixed_samples,
            tx_pair,
            rx_pair,
            lensless_correction=args.lensless_correction,
            detach_start_cdf=args.detach_start_cdf,
            directional_exponent=args.directional_exponent,
            create_graph=False,
            min_distance=args.min_distance,
        )
    flat_points = volume.points.reshape(-1, 3).detach()
    pair_amplitudes = volume.amplitudes.reshape(volume.amplitudes.shape[0], -1).detach()
    with torch.no_grad():
        response = pairwise_range_forward_operator(
            frequencies.detach(),
            kvector.detach(),
            tx_positions,
            rx_positions,
            flat_points,
            pair_amplitudes,
            phase_sign=args.phase_sign,
            oversample=args.oversample,
            kernel_width=args.kernel_width,
            pair_chunk=args.pair_chunk,
            point_chunk=args.point_chunk,
            compute_dtype=tg._compute_dtype(args.compute_dtype),
        )
    if not bool(torch.isfinite(response.real).all().item()) or not bool(torch.isfinite(response.imag).all().item()):
        raise FloatingPointError(f"view {view.view_index} predicted non-finite response")
    return response.detach()


def _enabled_metrics(*, rf_supported: bool, native_mf: bool) -> tuple[str, ...]:
    metrics = ["coherent", "range_power"]
    if rf_supported:
        metrics.append("rf_supported")
    if native_mf:
        metrics.append("native_mf")
    return tuple(metrics)


def _empty_cache(
    view_indices: np.ndarray,
    positions: np.ndarray,
    *,
    role: str = "validation",
    rf_supported: bool = False,
    native_mf: bool = False,
) -> dict[str, np.ndarray]:
    n = len(view_indices)
    nan = lambda: np.full(n, np.nan, dtype=np.float64)
    cache = {
        "view_indices": np.asarray(view_indices, dtype=np.int64),
        "viewpoint_positions": np.asarray(positions, dtype=np.float64),
        "selected_role": np.asarray(role),
        "enabled_metrics": np.asarray(",".join(_enabled_metrics(rf_supported=rf_supported, native_mf=native_mf))),
        "coherent_squared_error": nan(),
        "coherent_target_energy": nan(),
        "coherent_prediction_energy": nan(),
        "coherent_sample_count": nan(),
        "coherent_rel_mse": nan(),
        "range_power_squared_error": nan(),
        "range_power_target_squared_norm": nan(),
        "range_power_element_count": nan(),
        "range_power_rel_mse": nan(),
        "elapsed_seconds": nan(),
    }
    if rf_supported:
        cache.update(
            {
                "rf_supported_squared_error": nan(),
                "rf_supported_target_squared_norm": nan(),
                "rf_supported_element_count": nan(),
                "rf_supported_rel_mse": nan(),
                "rf_padded_squared_error": nan(),
                "rf_padded_target_squared_norm": nan(),
                "rf_padded_element_count": nan(),
                "rf_padded_rel_mse": nan(),
            }
        )
    if native_mf:
        cache.update(
            {
                "native_mf_squared_error": nan(),
                "native_mf_target_squared_norm": nan(),
                "native_mf_element_count": nan(),
                "native_mf_rel_mse": nan(),
            }
        )
    return cache


def _resume_identity(
    *, checkpoint_path: str, cache_root: str, npz_path: str, role_manifest: str,
    power_stats: str, view_indices: np.ndarray, role: str, rf_supported: bool, native_mf: bool,
) -> dict[str, object]:
    return {
        "schema": "rift_geraf_b7873200_common_eval_cache_v2",
        "checkpoint": os.path.abspath(checkpoint_path),
        "checkpoint_step": SELECTED_CHECKPOINT_STEP,
        "cache_root": os.path.abspath(cache_root),
        "npz_path": os.path.abspath(npz_path),
        "role_manifest": os.path.abspath(role_manifest),
        "power_stats": os.path.abspath(power_stats),
        "selected_role": role,
        "view_indices": [int(value) for value in view_indices],
        "enabled_metrics": list(_enabled_metrics(rf_supported=rf_supported, native_mf=native_mf)),
        "metric_contract": {
            "coherent_shape": [16, 16, 600],
            "power_dynamic_range_db": EXPECTED_POWER_DYNAMIC_RANGE_DB,
            "calibration": "no extra gain",
            "rf_supported": bool(rf_supported),
            "native_mf": bool(native_mf),
        },
    }


def _cache_identity_path(path: Path) -> Path:
    return path.with_name(path.stem + "_identity.json")


def _load_cache(
    path: Path,
    view_indices: np.ndarray,
    positions: np.ndarray,
    identity: Mapping[str, object],
    resume: bool,
    *,
    role: str,
    rf_supported: bool,
    native_mf: bool,
) -> dict[str, np.ndarray]:
    if not resume or not path.is_file():
        return _empty_cache(
            view_indices,
            positions,
            role=role,
            rf_supported=rf_supported,
            native_mf=native_mf,
        )
    with np.load(path, allow_pickle=False) as archive:
        cache = {name: np.asarray(archive[name]) for name in archive.files}
    if not np.array_equal(cache.get("view_indices"), view_indices):
        raise ValueError("resume cache source IDs do not match the current ordered role")
    required = set(
        _empty_cache(
            view_indices,
            np.zeros((len(view_indices), 3)),
            role=role,
            rf_supported=rf_supported,
            native_mf=native_mf,
        ).keys()
    )
    if not required.issubset(cache):
        raise ValueError("resume cache lacks the current metric fields")
    saved_role = str(np.asarray(cache["selected_role"]).reshape(()))
    saved_metrics = str(np.asarray(cache["enabled_metrics"]).reshape(()))
    expected_metrics = ",".join(_enabled_metrics(rf_supported=rf_supported, native_mf=native_mf))
    if saved_role != role or saved_metrics != expected_metrics:
        raise ValueError("resume cache role or enabled metrics disagree with the current request")
    metadata_path = _cache_identity_path(path)
    if not metadata_path.is_file():
        raise ValueError("resume cache lacks its identity sidecar")
    with metadata_path.open("r", encoding="utf-8") as handle:
        saved = json.load(handle)
    if saved.get("resume_identity") != dict(identity):
        raise ValueError("resume cache identity disagrees with the current checkpoint/role/stats contract")
    return cache


def _complete_mask(
    cache: Mapping[str, np.ndarray], *, rf_supported: bool = False, native_mf: bool = False
) -> np.ndarray:
    complete = np.isfinite(cache["coherent_rel_mse"]) & np.isfinite(cache["range_power_rel_mse"])
    if rf_supported:
        complete &= np.isfinite(cache["rf_supported_rel_mse"])
    if native_mf:
        complete &= np.isfinite(cache["native_mf_rel_mse"])
    return complete


def _aggregate(
    cache: Mapping[str, np.ndarray], *, rf_supported: bool = False, native_mf: bool = False
) -> dict[str, float | int]:
    complete = _complete_mask(cache, rf_supported=rf_supported, native_mf=native_mf)
    coherent_error = float(np.nansum(cache["coherent_squared_error"]))
    coherent_target = float(np.nansum(cache["coherent_target_energy"]))
    coherent_pred = float(np.nansum(cache["coherent_prediction_energy"]))
    coherent_count = int(round(float(np.nansum(cache["coherent_sample_count"]))))
    power_error = float(np.nansum(cache["range_power_squared_error"]))
    power_target = float(np.nansum(cache["range_power_target_squared_norm"]))
    power_count = int(round(float(np.nansum(cache["range_power_element_count"]))))
    if coherent_target <= 0.0 or power_target <= 0.0:
        raise ValueError("metric aggregation encountered a zero target norm")
    result: dict[str, float | int] = {
        "views_complete": int(complete.sum()),
        "views_total": int(len(complete)),
        "coherent_complex_rel_mse": coherent_error / coherent_target,
        "coherent_complex_percent": 100.0 * coherent_error / coherent_target,
        "coherent_squared_error_sum": coherent_error,
        "coherent_target_energy_sum": coherent_target,
        "coherent_prediction_energy_sum": coherent_pred,
        "coherent_sample_count": coherent_count,
        "normalized_range_power_rel_mse": power_error / power_target,
        "normalized_range_power_percent": 100.0 * power_error / power_target,
        "normalized_range_power_squared_error_sum": power_error,
        "normalized_range_power_target_squared_norm": power_target,
        "normalized_range_power_element_count": power_count,
        "elapsed_view_seconds_sum": float(np.nansum(cache["elapsed_seconds"])),
    }
    if rf_supported:
        supported_error = float(np.nansum(cache["rf_supported_squared_error"]))
        supported_target = float(np.nansum(cache["rf_supported_target_squared_norm"]))
        supported_count = int(round(float(np.nansum(cache["rf_supported_element_count"]))))
        padded_error = float(np.nansum(cache["rf_padded_squared_error"]))
        padded_target = float(np.nansum(cache["rf_padded_target_squared_norm"]))
        padded_count = int(round(float(np.nansum(cache["rf_padded_element_count"]))))
        padded_target_array = cache["rf_padded_target_squared_norm"]
        padded_count_array = cache["rf_padded_element_count"]
        padded_zero_target_view_count = int(
            (complete & np.isfinite(padded_count_array) & np.isfinite(padded_target_array) & (padded_target_array <= 0.0)).sum()
        )
        padded_empty_view_count = int(
            (complete & np.isfinite(padded_count_array) & (padded_count_array == 0.0)).sum()
        )
        result.update(
            {
                "rf_supported_squared_error_sum": supported_error,
                "rf_supported_target_squared_norm": supported_target,
                "rf_supported_element_count": supported_count,
                "rf_supported_rel_mse": pooled_relative_mse(supported_error, supported_target),
                "rf_supported_percent": 100.0 * supported_error / supported_target,
                "rf_padded_squared_error_sum": padded_error,
                "rf_padded_target_squared_norm": padded_target,
                "rf_padded_element_count": padded_count,
                "rf_padded_rel_mse": pooled_relative_mse(padded_error, padded_target),
                "rf_padded_percent": 100.0 * padded_error / padded_target,
                "rf_roi_element_count": supported_count + padded_count,
                "rf_padded_zero_target_view_count": padded_zero_target_view_count,
                "rf_padded_empty_view_count": padded_empty_view_count,
            }
        )
    if native_mf:
        native_error = float(np.nansum(cache["native_mf_squared_error"]))
        native_target = float(np.nansum(cache["native_mf_target_squared_norm"]))
        native_count = int(round(float(np.nansum(cache["native_mf_element_count"]))))
        native_rel = pooled_relative_mse(native_error, native_target)
        result.update(
            {
                "native_mf_squared_error_sum": native_error,
                "native_mf_target_squared_norm": native_target,
                "native_mf_element_count": native_count,
                "native_mf_rel_mse": native_rel,
                "native_mf_percent": 100.0 * native_rel,
                "native_mf_mse": native_error / native_count,
                "native_mf_rmse": math.sqrt(native_error / native_count),
            }
        )
    return result


def validate_production_aggregate(
    summary: Mapping[str, object], *, role: str = "validation", rf_supported: bool = False, native_mf: bool = False,
    collection_identity: Mapping[str, object] | None = None,
) -> None:
    if role not in ("validation", "test"):
        raise ValueError(f"unsupported production role {role!r}")
    if int(summary.get("views_complete", -1)) != B787_3200_NUM_VALIDATION or int(summary.get("views_total", -1)) != B787_3200_NUM_VALIDATION:
        raise ValueError(f"production common evaluation must cover all 1000 {role} views")
    if int(summary.get("coherent_sample_count", -1)) != EXPECTED_VALIDATION_SAMPLES:
        raise ValueError("coherent metric sample count does not match 1000x16x16x600")
    if int(summary.get("normalized_range_power_element_count", -1)) != EXPECTED_POWER_ELEMENT_COUNT:
        raise ValueError("normalized range-power ROI element count disagrees with the frozen protocol")
    expected_power_target = (
        EXPECTED_POWER_TARGET_SQUARED_NORM
        if role == "validation"
        else EXPECTED_TEST_POWER_TARGET_SQUARED_NORM
    )
    if collection_identity is None:
        _require_close("normalized range-power target squared norm", summary.get("normalized_range_power_target_squared_norm"), expected_power_target, abs_tol=1.0e-4)
    elif _finite_float(summary.get("normalized_range_power_target_squared_norm"), "target squared norm") <= 0:
        raise ValueError("Collection power metric requires a positive measured target norm")
    for key in ("coherent_complex_rel_mse", "normalized_range_power_rel_mse"):
        value = _finite_float(summary.get(key), key)
        if value < 0.0:
            raise ValueError(f"{key} must be nonnegative")
    if rf_supported:
        for key in ("rf_supported_rel_mse", "rf_padded_rel_mse"):
            value = _finite_float(summary.get(key), key)
            if value < 0.0:
                raise ValueError(f"{key} must be nonnegative")
        if int(summary.get("rf_supported_element_count", 0)) <= 0 or int(summary.get("rf_padded_element_count", 0)) <= 0:
            raise ValueError("RF-supported production aggregate has an empty partition")
    if native_mf:
        if int(summary.get("native_mf_element_count", -1)) != EXPECTED_NATIVE_MF_SAMPLES:
            raise ValueError("native GeRaF MF sample count does not match the 1000x32x32x32 grid")
        if _finite_float(summary.get("native_mf_rel_mse"), "native_mf_rel_mse") < 0.0:
            raise ValueError("native_mf_rel_mse must be nonnegative")


def _summary(
    aggregate: Mapping[str, object], *, resume_identity: Mapping[str, object], checkpoint_row: Mapping[str, object],
    cache: Any, source: Any, response_source: Any, stats: Mapping[str, object], peak_power: float,
    dynamic_range_db: float, selected_indices: np.ndarray, role: str, rf_supported: bool, native_mf: bool,
    production: bool,
) -> dict[str, object]:
    payload = dict(aggregate)
    payload.update(
        {
            "schema": "rift_geraf_b7873200_common_eval_v1",
            "status": "production_complete" if production else "incomplete_or_smoke",
            "method": (f"GeRaF ({resume_identity['dataset_identity']['object_id']})"
                       if "dataset_identity" in resume_identity else "GeRaF v1 (sealed B7873200)"),
            "metric_domains": {
                "coherent_complex": "raw complex frequency response before matched filtering",
                "normalized_range_power": "60 dB normalized |IFFT(mean_chirp(response))|^2 with scene-range ROI",
            },
            "checkpoint_selection": {
                "path": resume_identity["checkpoint"],
                "step": int(checkpoint_row["step"]),
                "best_history_row": dict(checkpoint_row),
                "best_mf_magnitude_mse": float(checkpoint_row["mf_magnitude_mse"]),
                "best_native_mf_magnitude_rel_mse": float(checkpoint_row["mf_magnitude_relative_mse"]),
                "validation_views": B787_3200_NUM_VALIDATION,
            },
            "renderer_contract": {
                "scene_grid": [32, 32, 32],
                "scene_extent_m": 0.15,
                "lensless_correction": True,
                "detach_start_cdf": True,
                "directional_exponent": 1.0,
                "min_distance": 1.0e-6,
                "phase_sign": -1.0,
                "range_backend": "range_nufft",
                "oversample": 2,
                "kernel_width": 20,
                "compute_dtype": "float64",
                "channel_order": "Tx-outer/Rx-inner after [F,Rx,Tx] -> permute(2,1,0)",
            },
            "calibration": "no extra gain",
            "normalization": {
                "stats_path": resume_identity["power_stats"],
                "peak_power": peak_power,
                "dynamic_range_db": dynamic_range_db,
                "normalization_domain": stats.get("normalization_domain", "normalized_dB_range_power"),
                "normalization_scan_role": stats.get("normalization_provenance", {}).get("normalization_scan_role") if isinstance(stats.get("normalization_provenance"), Mapping) else None,
                "train_view_count": stats.get("train_view_count"),
            },
            "selected_role": role,
            "ordered_source_view_ids": [int(value) for value in selected_indices],
            "authorized_response_view_ids": [int(value) for value in selected_indices],
            "authorized_response_view_count": int(len(selected_indices)),
            "enabled_metrics": list(_enabled_metrics(rf_supported=rf_supported, native_mf=native_mf)),
            "provenance_paths": {
                "npz": resume_identity["npz_path"],
                "role_manifest": resume_identity["role_manifest"],
                "cache_root": resume_identity["cache_root"],
                "acquisition_record": str(Path(str(resume_identity["cache_root"])) / B787_3200_CACHE_ACQUISITION_FILENAME),
                "cache_recipe": str(Path(str(resume_identity["cache_root"])) / B787_3200_CACHE_RECIPE_FILENAME),
                "cache_stats": str(Path(str(resume_identity["cache_root"])) / B787_3200_CACHE_STATS_FILENAME),
            },
            "response_header": list(RESPONSE_SHAPE),
            "authorized_response_roles": ["reserved_test" if role == "test" else "validation"],
            "reserved_test_accessed": role == "test",
            "unused_views_accessed": False,
            "resume_identity": dict(resume_identity),
            "source_response_access_restricted": bool(getattr(response_source, "response_access_is_restricted", False)),
            "source_authorized_response_view_count": len(getattr(response_source, "allowed_response_view_indices", ())),
            "native_target_provenance": (
                "prepared validation target normalized by TRAIN-only peak"
                if role == "validation"
                else "fresh reserved_test phase-only native matched-filter amplitude, float32, divided by frozen TRAIN-only peak"
            ) if native_mf else None,
            "rf_supported_metrics_contract": (
                {
                    "mask_definition": "cell_mass[:, roi] > 0",
                    "grid": "generate_dynamic_grid(48, 0.15, jitter=False) midpoint grid",
                    "grid_dtype": "float32",
                    "pair_count": 256,
                    "range_bins": 600,
                    "pair_chunk": 8,
                    "extent_m": RF_GRID_EXTENT_M,
                    "range_margin_m": RF_RANGE_MARGIN_M,
                    "learned_support_used": False,
                    "target_dependent": False,
                }
                if rf_supported
                else None
            ),
            "target_cache_schema": cache.recipe.get("schema"),
        }
    )
    if role == "validation":
        payload["ordered_validation_view_ids"] = [int(value) for value in selected_indices]
    else:
        payload.pop("ordered_validation_view_ids", None)
    return payload


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object")
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parents[1] / "data/RIFT_dataset")
    parser.add_argument("--checkpoint")
    parser.add_argument("--npz-path")
    parser.add_argument("--role-manifest")
    parser.add_argument("--cache-root")
    parser.add_argument("--power-stats")
    parser.add_argument("--out-dir")
    parser.add_argument("--allow-reserved-test", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--max-views", type=int, default=0, help="debug cap; production requires 0")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--role", choices=("validation", "val", "test", "reserved_test"), default="validation")
    parser.add_argument("--rf-supported-metrics", action="store_true")
    parser.add_argument("--native-mf-metrics", action="store_true")
    args = parser.parse_args(argv)
    from rift.rift_dataset import collection_manifest, resolve_object_inputs
    args.role = {"val": "validation", "reserved_test": "test"}.get(args.role, args.role)
    args.collection_mode = args.object is not None or (
        args.role_manifest is not None and collection_manifest(args.role_manifest))
    if args.collection_mode:
        if any(value is None for value in (args.checkpoint, args.cache_root, args.power_stats)):
            parser.error("collection evaluation requires explicit --checkpoint, --cache-root, and --power-stats")
        if args.role == "test" and not args.allow_reserved_test:
            parser.error("collection test evaluation requires --allow-reserved-test")
        args.npz_path, args.role_manifest = map(str, resolve_object_inputs(
            object_name=args.object, dataset_root=args.dataset_root,
            npz_path=args.npz_path, role_manifest_path=args.role_manifest))
        args.out_dir = args.out_dir or str(Path(args.cache_root) / "collection_complex_readout")
    else:
        for name, default in (("checkpoint", DEFAULT_CHECKPOINT), ("npz_path", DEFAULT_NPZ),
                ("role_manifest", DEFAULT_ROLE_MANIFEST), ("cache_root", DEFAULT_CACHE_ROOT),
                ("power_stats", DEFAULT_POWER_STATS), ("out_dir", DEFAULT_OUT_DIR)):
            setattr(args, name, getattr(args, name) or default)
    return args


def collection_evaluation_inputs(args, checkpoint):
    """Reject foreign checkpoint, cache and stats before cached/raw target reads."""
    from rift.rift_dataset import (load_object_contract, validate_checkpoint_object, object_identity)
    _, contract = load_object_contract(args.npz_path, args.role_manifest,
        response_roles=(args.role,), allow_reserved_test=args.allow_reserved_test)
    if args.object is not None:
        validate_checkpoint_object(object_identity(args.object), contract)
    validate_checkpoint_object(checkpoint, contract)
    for filename in (B787_3200_CACHE_RECIPE_FILENAME, B787_3200_CACHE_STATS_FILENAME):
        with (Path(args.cache_root) / filename).open() as handle:
            validate_checkpoint_object(json.load(handle), contract)
    with open(args.power_stats) as handle:
        stats = json.load(handle)
    validate_checkpoint_object(stats, contract)
    return contract, stats


def _test_view(
    *, source_arrays: Any, view_index: int, train_args: argparse.Namespace, device: torch.device
) -> Any:
    """Build the TEST view from compact frozen geometry without opening a target file."""

    index = int(view_index)
    viewpoint = np.asarray(source_arrays.viewpoint_positions[index], dtype=np.float32)
    tx_np = np.asarray(source_arrays.tx_pos[index], dtype=np.float32)
    rx_np = np.asarray(source_arrays.rx_pos[index], dtype=np.float32)
    tx = torch.as_tensor(tx_np, dtype=torch.float32, device=device)
    rx = torch.as_tensor(rx_np, dtype=torch.float32, device=device)
    geometry = tg._expected_cached_geometry(viewpoint, tx_np, rx_np, train_args)
    geometry["viewpoint_position"] = viewpoint
    samples = tg._samples_from_cached_geometry(geometry, device)
    return SimpleNamespace(
        view_index=index,
        target_normalized=None,
        samples=samples,
        tx_positions=tx,
        rx_positions=rx,
        viewpoint_position=viewpoint,
    )


def _target_operator_args(cache: Any) -> argparse.Namespace:
    """Decode the already verified native target operator recipe."""

    target_spec = cache.recipe.get("target_spec")
    operator = target_spec.get("operator") if isinstance(target_spec, Mapping) else None
    if not isinstance(target_spec, Mapping) or not isinstance(operator, Mapping):
        raise ValueError("prepared GeRaF cache lacks a target_spec operator")
    if target_spec.get("backend") != "range" or target_spec.get("phase_sign") != -1.0:
        raise ValueError("prepared GeRaF target recipe is not the frozen phase -1 range backend")
    expected = {
        "range_model": "none",
        "include_four_pi": False,
        "kernel_width": 20,
        "oversample": 2,
        "pair_chunk": 32,
        "point_chunk": 4096,
    }
    for name, value in expected.items():
        if operator.get(name) != value:
            raise ValueError(f"prepared GeRaF target recipe {name} disagrees with frozen value {value!r}")
    if target_spec.get("compute_dtype") != "float64":
        raise ValueError("prepared GeRaF target recipe must use float64 computation")
    return argparse.Namespace(
        backend="range",
        phase_sign=-1.0,
        compute_dtype="float64",
        kernel_width=20,
        oversample=2,
        pair_chunk=32,
        point_chunk=4096,
    )


def _native_target(
    *,
    role: str,
    view: Any,
    raw_measured: np.ndarray,
    frequencies: torch.Tensor,
    cache: Any,
    target_args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    if role == "validation":
        target = view.target_normalized
        if not isinstance(target, torch.Tensor):
            raise ValueError("validation GeRaF view lacks its cached native target")
        return target
    if role != "test":
        raise ValueError(f"unsupported native target role {role!r}")
    response_tx_rx_freq = np.asarray(raw_measured.mean(axis=2))
    with torch.no_grad():
        amplitude = _matched_filter_amplitude(
            response_tx_rx_freq,
            frequencies,
            view.tx_positions,
            view.rx_positions,
            view.samples.points.reshape(-1, 3),
            target_args,
        )
        target = amplitude.abs().reshape(cache.grid_shape).to(torch.float32)
    return target / float(cache.geraf_mf_magnitude_peak)


def main(argv=None) -> None:
    args = _parse_args(argv)
    if args.max_views < 0 or args.max_views > B787_3200_NUM_VALIDATION or args.save_every <= 0:
        raise ValueError("max-views must lie in [0,1000] and save-every must be positive")
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    checkpoint_path = os.path.abspath(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("GeRaF checkpoint must be a mapping")
    collection = None
    if args.collection_mode:
        collection, power_stats = collection_evaluation_inputs(args, checkpoint)
    cli_args = checkpoint.get("cli_args")
    if not isinstance(cli_args, Mapping):
        raise ValueError("GeRaF checkpoint lacks its CLI contract")
    train_args = argparse.Namespace(**dict(cli_args))
    train_args.npz_path = os.path.abspath(args.npz_path)
    train_args.role_manifest = os.path.abspath(args.role_manifest)
    train_args.cache_root = os.path.abspath(args.cache_root)
    train_args.device = str(device)

    cache = tg.verify_prepared_b7873200_cache(train_args)
    checkpoint_row = _validate_checkpoint_selection(checkpoint, cache, train_args)
    source = load_b7873200_metadata_source(args.npz_path, args.role_manifest)
    if source.identity != cache.sealed_identity:
        raise ValueError("metadata source and target cache do not share sealed role identity")

    acquisition = validate_b7873200_acquisition_record(args.cache_root, source.arrays)
    stored_acquisition = load_b7873200_acquisition_record(Path(args.cache_root) / B787_3200_CACHE_ACQUISITION_FILENAME)
    if not acquisition_records_equal(acquisition, stored_acquisition):
        raise ValueError("acquisition record changed between validation reads")
    operator_frequency_hz = b7873200_operator_frequency_grid_hz(source.arrays.metadata)
    frequencies = torch.as_tensor(
        validate_b7873200_operator_frequency_grid(acquisition, operator_frequency_hz),
        dtype=torch.float64,
        device=device,
    )
    kvector = get_kvector(frequencies, cc)

    if collection is None:
        if "dataset_identity" in cache.sealed_identity:
            raise ValueError("Collection GeRaF evaluation requires --object or its role manifest")
        with open(args.power_stats, "r", encoding="utf-8") as handle:
            power_stats = json.load(handle)
    peak_power, dynamic_range_db = validate_normalization_stats(
        power_stats,
        {"sealed_npz_protocol_contract": _sealed_contract(cache.sealed_identity)},
        num_views=B787_3200_NUM_VIEWS,
    )
    if collection is None:
        _require_close("power normalization peak", peak_power, EXPECTED_POWER_PEAK, abs_tol=1.0e-22)
    _require_close("power normalization dynamic range", dynamic_range_db, EXPECTED_POWER_DYNAMIC_RANGE_DB)
    if power_stats.get("normalization_domain") not in (None, "normalized_dB_range_power"):
        raise ValueError("power stats are not in the normalized range-power domain")

    response_source = load_radar_fields_npz(args.npz_path, load_response=False)
    if tuple(int(value) for value in (response_source.response_shape or ())) != RESPONSE_SHAPE:
        raise ValueError("evaluator response header disagrees with the canonical B7873200 shape")
    if np.dtype(response_source.response_dtype) != np.dtype(np.complex64):
        raise ValueError("evaluator response header is not complex64")
    for name in ("viewpoint_positions", "tx_pos", "rx_pos"):
        if not np.array_equal(getattr(response_source, name), getattr(source.arrays, name)):
            raise ValueError(f"evaluator metadata field {name} disagrees with the acquisition source")
    role_contract = {"sealed_npz_protocol_contract": _sealed_contract(cache.sealed_identity)}
    role_args = argparse.Namespace(
        num_train=B787_3200_NUM_TRAIN,
        num_val=B787_3200_NUM_VALIDATION,
        seed=42,
        max_views=args.max_views,
    )
    if collection is not None:
        from rift.rift_dataset import evaluation_role_indices
        selected_indices = evaluation_role_indices(collection, args.role, allow_reserved_test=args.allow_reserved_test)
        if args.max_views:
            selected_indices = selected_indices[:args.max_views]
    else:
        selected_indices = _sealed_role_indices(role_contract, response_source, role_args, args.role)
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    expected_role = np.asarray(
        _ordered_role_ids(
            cache.sealed_identity,
            "reserved_test" if args.role == "test" else "validation",
            B787_3200_NUM_TEST if args.role == "test" else B787_3200_NUM_VALIDATION,
        ),
        dtype=np.int64,
    )
    expected_role = expected_role[: args.max_views] if args.max_views else expected_role
    if not np.array_equal(selected_indices, expected_role):
        raise ValueError(f"selected {args.role} IDs disagree with the sealed cache identity")
    response_source = restrict_radar_fields_response_views(response_source, selected_indices)
    positions = np.asarray(response_source.viewpoint_positions[selected_indices], dtype=np.float64)
    identity = _resume_identity(
        checkpoint_path=checkpoint_path,
        cache_root=args.cache_root,
        npz_path=args.npz_path,
        role_manifest=args.role_manifest,
        power_stats=args.power_stats,
        view_indices=selected_indices,
        role=args.role,
        rf_supported=args.rf_supported_metrics,
        native_mf=args.native_mf_metrics,
    )
    out_dir = Path(args.out_dir)
    if collection is not None:
        identity.update(dataset_identity=collection["dataset_identity"],
                        checkpoint_step=int(checkpoint["step"]), power_normalization=power_stats)
    metric_suffix = "common" if args.role == "validation" and not (args.rf_supported_metrics or args.native_mf_metrics) else args.role
    if args.rf_supported_metrics:
        metric_suffix += "_rf"
    if args.native_mf_metrics:
        metric_suffix += "_native"
    prefix = f"geraf_{collection['dataset_identity']['object_id']}" if collection else "geraf_b7873200"
    cache_path = out_dir / f"{prefix}_{metric_suffix}_per_view.npz"
    summary_path = out_dir / f"{prefix}_{metric_suffix}_metrics.json"
    metric_cache = _load_cache(
        cache_path,
        selected_indices,
        positions,
        identity,
        args.resume,
        role=args.role,
        rf_supported=args.rf_supported_metrics,
        native_mf=args.native_mf_metrics,
    )
    if not np.array_equal(metric_cache["viewpoint_positions"], positions):
        raise ValueError("metric cache viewpoint positions disagree with the acquisition record")

    model = GeRaFModel(**dict(checkpoint["model_config"])).to(device=device, dtype=torch.float32)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    ranges = range_bin_centers(source.arrays.metadata, device=device, dtype=torch.float32)
    if args.rf_supported_metrics:
        if (response_source.num_tx, response_source.num_rx, response_source.num_freq) != (16, 16, 600):
            raise ValueError("RF-supported GeRaF metrics require the canonical 16x16x600 acquisition")
        if not math.isclose(float(train_args.scene_extent), RF_GRID_EXTENT_M, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("RF-supported GeRaF metrics require scene extent 0.15 m")
        rf_grid = make_rf_midpoint_grid(device)
        pair_indices = torch.arange(256, dtype=torch.long, device=device)
    else:
        rf_grid = None
        pair_indices = None
    target_operator_args = _target_operator_args(cache) if args.native_mf_metrics else None
    complete = _complete_mask(
        metric_cache,
        rf_supported=args.rf_supported_metrics,
        native_mf=args.native_mf_metrics,
    )
    pending = np.flatnonzero(~complete)
    started = time.perf_counter()
    print(f"device={device}; role={args.role}; views={len(selected_indices)}; pending={len(pending)}; checkpoint_step={checkpoint['step']}", flush=True)

    for ordinal, slot in enumerate(pending, start=1):
        view_index = int(selected_indices[slot])
        view_started = time.perf_counter()
        view = (
            tg.load_training_view(cache, view_index, "validation", train_args, device)
            if args.role == "validation"
            else _test_view(
                source_arrays=response_source,
                view_index=view_index,
                train_args=train_args,
                device=device,
            )
        )
        predicted_response = render_complex_response(model, view, frequencies, kvector, train_args)
        model.zero_grad(set_to_none=True)
        pred_flat = flatten_ge_ra_f_response(predicted_response)
        raw_measured = response_source.response_view(view_index)
        if raw_measured.shape != VIEW_RESPONSE_SHAPE or np.dtype(raw_measured.dtype) != np.dtype(np.complex64):
            raise ValueError(f"view {view_index} measured response header disagrees with {VIEW_RESPONSE_SHAPE}/complex64")
        if not np.isfinite(raw_measured.real).all() or not np.isfinite(raw_measured.imag).all():
            raise FloatingPointError(f"view {view_index} measured response is non-finite")
        measured_flat_np = raw_measured.mean(axis=2).reshape(16 * 16, 600)
        measured = torch.as_tensor(measured_flat_np, dtype=torch.complex128, device=device)
        if pred_flat.shape != measured.shape:
            raise ValueError(f"view {view_index} prediction/measurement shapes disagree")

        residual = pred_flat - measured
        coherent_error = float(residual.abs().square().sum().cpu())
        coherent_target = float(measured.abs().square().sum().cpu())
        coherent_prediction = float(pred_flat.abs().square().sum().cpu())
        coherent_count = int(residual.numel())
        if coherent_target <= 0.0 or not all(math.isfinite(value) for value in (coherent_error, coherent_prediction)):
            raise FloatingPointError(f"view {view_index} coherent metric has an invalid norm")

        pred_power = torch.fft.ifft(pred_flat, dim=-1).abs().square()
        target_power = response_view_to_range_power(raw_measured, device=device)
        pred_intensity = normalize_power_db(pred_power, peak_power, dynamic_range_db)
        target_intensity = normalize_power_db(target_power, peak_power, dynamic_range_db)
        viewpoint = torch.as_tensor(source.arrays.viewpoint_positions[view_index], dtype=torch.float32, device=device)
        roi = scene_range_mask(ranges, viewpoint, float(train_args.scene_extent), margin=0.05)
        pred_roi = pred_intensity[:, roi]
        target_roi = target_intensity[:, roi]
        power_error = float((pred_roi - target_roi).square().sum().cpu())
        power_target = float(target_roi.square().sum().cpu())
        power_count = int(target_roi.numel())
        if power_target <= 0.0 or power_count <= 0 or not math.isfinite(power_error):
            raise FloatingPointError(f"view {view_index} range-power metric has an invalid norm")

        rf_partition = None
        if args.rf_supported_metrics:
            if rf_grid is None or pair_indices is None:
                raise RuntimeError("RF-supported metrics were enabled without mask inputs")
            supported_mask, _ = rf_supported_mask(
                rf_grid,
                ranges=ranges,
                viewpoint=viewpoint,
                tx_pos=view.tx_positions,
                rx_pos=view.rx_positions,
                pair_indices=pair_indices,
                metadata=source.arrays.metadata,
                extent_m=float(train_args.scene_extent),
                range_margin_m=RF_RANGE_MARGIN_M,
                roi=roi,
            )
            rf_partition = partition_squared_error(pred_roi, target_roi, supported_mask)

        native_prediction = None
        native_target = None
        if args.native_mf_metrics:
            native_amplitude = matched_filter_from_response_range(
                predicted_response,
                frequencies,
                kvector,
                view.tx_positions,
                view.rx_positions,
                view.samples.points.reshape(-1, 3),
                phase_sign=float(train_args.phase_sign),
                oversample=int(train_args.oversample),
                kernel_width=int(train_args.kernel_width),
                pair_chunk=int(train_args.pair_chunk),
                point_chunk=int(train_args.point_chunk),
                compute_dtype=tg._compute_dtype(train_args.compute_dtype),
            )
            native_prediction = native_amplitude.abs().reshape(cache.grid_shape) / float(cache.geraf_mf_magnitude_peak)
            native_target = _native_target(
                role=args.role,
                view=view,
                raw_measured=raw_measured,
                frequencies=frequencies,
                cache=cache,
                target_args=target_operator_args,
                device=device,
            )
            if native_prediction.shape != native_target.shape or not bool(torch.isfinite(native_prediction).all().item()) or not bool(torch.isfinite(native_target).all().item()):
                raise FloatingPointError(f"view {view_index} native MF metric has invalid values")
            native_error = float((native_prediction - native_target).square().sum().cpu())
            native_target_norm = float(native_target.square().sum().cpu())
            native_count = int(native_target.numel())
            if native_target_norm <= 0.0 or native_count <= 0 or not math.isfinite(native_error):
                raise FloatingPointError(f"view {view_index} native MF metric has an invalid norm")

        metric_cache["coherent_squared_error"][slot] = coherent_error
        metric_cache["coherent_target_energy"][slot] = coherent_target
        metric_cache["coherent_prediction_energy"][slot] = coherent_prediction
        metric_cache["coherent_sample_count"][slot] = coherent_count
        metric_cache["coherent_rel_mse"][slot] = coherent_error / coherent_target
        metric_cache["range_power_squared_error"][slot] = power_error
        metric_cache["range_power_target_squared_norm"][slot] = power_target
        metric_cache["range_power_element_count"][slot] = power_count
        metric_cache["range_power_rel_mse"][slot] = power_error / power_target
        if rf_partition is not None:
            metric_cache["rf_supported_squared_error"][slot] = rf_partition["supported_squared_error"]
            metric_cache["rf_supported_target_squared_norm"][slot] = rf_partition["supported_target_squared_norm"]
            metric_cache["rf_supported_element_count"][slot] = rf_partition["supported_element_count"]
            metric_cache["rf_supported_rel_mse"][slot] = pooled_relative_mse(
                rf_partition["supported_squared_error"], rf_partition["supported_target_squared_norm"]
            )
            metric_cache["rf_padded_squared_error"][slot] = rf_partition["padded_squared_error"]
            metric_cache["rf_padded_target_squared_norm"][slot] = rf_partition["padded_target_squared_norm"]
            metric_cache["rf_padded_element_count"][slot] = rf_partition["padded_element_count"]
            metric_cache["rf_padded_rel_mse"][slot] = diagnostic_relative_mse(
                rf_partition["padded_squared_error"], rf_partition["padded_target_squared_norm"]
            )
        if native_prediction is not None and native_target is not None:
            metric_cache["native_mf_squared_error"][slot] = native_error
            metric_cache["native_mf_target_squared_norm"][slot] = native_target_norm
            metric_cache["native_mf_element_count"][slot] = native_count
            metric_cache["native_mf_rel_mse"][slot] = native_error / native_target_norm
        metric_cache["elapsed_seconds"][slot] = time.perf_counter() - view_started
        del view, predicted_response, pred_flat, measured, residual, pred_power, target_power

        done = int(_complete_mask(metric_cache, rf_supported=args.rf_supported_metrics, native_mf=args.native_mf_metrics).sum())
        detail = f"coherent={100.0 * metric_cache['coherent_rel_mse'][slot]:9.4f}% power={100.0 * metric_cache['range_power_rel_mse'][slot]:9.4f}%"
        if args.native_mf_metrics:
            detail += f" native={100.0 * metric_cache['native_mf_rel_mse'][slot]:9.4f}%"
        print(f"[{done:4d}/{len(selected_indices)}] view {view_index:4d}: {detail}", flush=True)
        if done % args.save_every == 0 or done == len(selected_indices):
            _atomic_save_npz(cache_path, metric_cache)
            _atomic_write_json(_cache_identity_path(cache_path), {"resume_identity": identity})
            aggregate = _aggregate(metric_cache, rf_supported=args.rf_supported_metrics, native_mf=args.native_mf_metrics)
            _atomic_write_json(
                summary_path,
                _summary(
                    aggregate,
                    resume_identity=identity,
                    checkpoint_row=checkpoint_row,
                    cache=cache,
                    source=source,
                    response_source=response_source,
                    stats=power_stats,
                    peak_power=peak_power,
                    dynamic_range_db=dynamic_range_db,
                    selected_indices=selected_indices,
                    role=args.role,
                    rf_supported=args.rf_supported_metrics,
                    native_mf=args.native_mf_metrics,
                    production=False,
                ),
            )

    _atomic_save_npz(cache_path, metric_cache)
    _atomic_write_json(_cache_identity_path(cache_path), {"resume_identity": identity})
    aggregate = _aggregate(metric_cache, rf_supported=args.rf_supported_metrics, native_mf=args.native_mf_metrics)
    production = args.max_views == 0
    if production:
        validate_production_aggregate(
            aggregate,
            role=args.role,
            rf_supported=args.rf_supported_metrics,
            native_mf=args.native_mf_metrics,
            collection_identity=collection,
        )
    final_summary = _summary(
        aggregate,
        resume_identity=identity,
        checkpoint_row=checkpoint_row,
        cache=cache,
        source=source,
        response_source=response_source,
        stats=power_stats,
        peak_power=peak_power,
        dynamic_range_db=dynamic_range_db,
        selected_indices=selected_indices,
        role=args.role,
        rf_supported=args.rf_supported_metrics,
        native_mf=args.native_mf_metrics,
        production=production,
    )
    final_summary["elapsed_seconds_this_invocation"] = time.perf_counter() - started
    _atomic_write_json(summary_path, final_summary)
    if production:
        print("GERAF_B7873200_COMMON_EVAL_PASS", flush=True)
    else:
        print("GERAF_B7873200_COMMON_EVAL_SMOKE_COMPLETE", flush=True)
    print(f"cache: {cache_path}\nsummary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
