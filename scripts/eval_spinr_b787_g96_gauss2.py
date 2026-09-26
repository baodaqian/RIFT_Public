#!/usr/bin/env python3
"""Evaluate the bounded SpINR-style G96/Gauss2 B787 engineering smoke.

The checkpoint was fitted on 16 train and 16 validation views for 60 updates.
This evaluator applies that frozen field to the canonical 1,000-view validation
or reserved-test cohort without changing the model, scale, quadrature, or
physics operator.  The 1,000-view readout is a smoke-trained-model evaluation,
not evidence of a production 3,200-view training run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rift.config import cc  # noqa: E402
from rift.forward_operator import get_kvector  # noqa: E402
from rift.range_operator import range_forward_operator  # noqa: E402
from rift.radar_fields_dataset import (  # noqa: E402
    load_radar_fields_npz,
    normalize_power_db,
    range_bin_centers,
    response_view_to_range_power,
    restrict_radar_fields_response_views,
    scene_range_mask,
)
from rift.spinr_style import (  # noqa: E402
    SPINR_STYLE_HIDDEN_LAYERS,
    SPINR_STYLE_HIDDEN_WIDTH,
    SPINR_STYLE_INPUT_FEATURES,
    SPINR_STYLE_PARAMETER_COUNT,
    SPINR_STYLE_PHASE_SIGN,
    SPINR_STYLE_RANGE_MODEL,
    SPINR_STYLE_SUPPORT_M,
    SpinrStyleINR,
    build_spinr_style_acquisition_identity,
    gauss_legendre_cell_grid,
    metadata_frequency_grid,
    scale_field_to_renderer_weights,
    validate_spinr_style_acquisition_identity,
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
from train_spinr_style import evaluate_neural_field_tiled  # noqa: E402


DEFAULT_CHECKPOINT = (
    "/storage/scratch1/1/dbao31/rift_b7873200_spinr_style_smallfit_g96_gauss2_v1/"
    "b78710k_spinr_style_inr_pm_g96_gauss2_smallfit16_v1/checkpoint_final.pth.tar"
)
DEFAULT_NPZ = (
    "/storage/project/r-jromberg3-0/dbao31/RIFT/data/"
    "b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz"
)
DEFAULT_ROLE_MANIFEST = (
    "/storage/scratch1/1/dbao31/rift_round8b_impl_20260810/splits/round8b/"
    "b78710k_interp_seed42_train3200_val1000_test1000_v1.json"
)
DEFAULT_STATS = (
    "/storage/scratch1/1/dbao31/rift_b7873200_radar_fields_production_v1/"
    "attempt-12941397/rf_b7873200_production_v1/radar_fields_power_stats.json"
)
DEFAULT_OUT_DIR = "/storage/scratch1/1/dbao31/rift_b7873200_spinr_g96_gauss2_eval_v1"

CHECKPOINT_FORMAT = "rift_spinr_style_b78716_smallfit_g96_gauss2_v1"
CHECKPOINT_UPDATES = 60
CHECKPOINT_FIT_TRAIN_COUNT = 16
CHECKPOINT_FIT_VALIDATION_COUNT = 16
PARENT_TRAIN_COUNT = 3200
ROLE_COUNT = 1000
UNUSED_COUNT = 4800
RESPONSE_SHAPE = (10_000, 16, 16, 1, 600)
VIEW_RESPONSE_SHAPE = RESPONSE_SHAPE[1:]
QUADRATURE_PARENT_GRID = 96
QUADRATURE_NODES_PER_CELL = 2
QUADRATURE_POINT_COUNT = 7_077_888
QUADRATURE_SUPPORT_M = 0.15
NEURAL_POINT_TILE = 4096
RENDERER_POINT_TILE = 65536
PAIR_TILE = 16
EXPECTED_POWER_PEAK = 1.1800956040705975e-07
EXPECTED_POWER_DYNAMIC_RANGE_DB = 60.0
EXPECTED_VALIDATION_SAMPLES = 153_600_000
EXPECTED_POWER_ELEMENT_COUNT = 3_328_000
EXPECTED_VALIDATION_POWER_TARGET = 411925.32444763184
EXPECTED_TEST_POWER_TARGET = 419975.4931983948


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


def _finite_positive(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return number


def _require_close(label: str, observed: object, expected: float, *, abs_tol: float = 1.0e-12) -> None:
    value = float(observed)
    if not math.isfinite(value) or not math.isclose(value, expected, rel_tol=0.0, abs_tol=abs_tol):
        raise ValueError(f"{label}={value!r} disagrees with frozen value {expected!r}")


def _enabled_metrics() -> tuple[str, ...]:
    return ("coherent", "linear_power", "range_power", "rf_supported", "rf_padded")


def _empty_cache(view_indices: np.ndarray, positions: np.ndarray, *, role: str) -> dict[str, np.ndarray]:
    count = len(view_indices)
    nan = lambda: np.full(count, np.nan, dtype=np.float64)
    return {
        "view_indices": np.asarray(view_indices, dtype=np.int64),
        "viewpoint_positions": np.asarray(positions, dtype=np.float64),
        "selected_role": np.asarray(role),
        "enabled_metrics": np.asarray(",".join(_enabled_metrics())),
        "coherent_squared_error": nan(),
        "coherent_target_energy": nan(),
        "coherent_prediction_energy": nan(),
        "coherent_sample_count": nan(),
        "coherent_rel_mse": nan(),
        "linear_power_squared_error": nan(),
        "linear_power_target_squared_norm": nan(),
        "linear_power_element_count": nan(),
        "linear_power_rel_mse": nan(),
        "range_power_squared_error": nan(),
        "range_power_target_squared_norm": nan(),
        "range_power_element_count": nan(),
        "range_power_rel_mse": nan(),
        "rf_supported_squared_error": nan(),
        "rf_supported_target_squared_norm": nan(),
        "rf_supported_element_count": nan(),
        "rf_supported_rel_mse": nan(),
        "rf_padded_squared_error": nan(),
        "rf_padded_target_squared_norm": nan(),
        "rf_padded_element_count": nan(),
        "rf_padded_rel_mse": nan(),
        "elapsed_seconds": nan(),
    }


def _complete_mask(cache: Mapping[str, np.ndarray]) -> np.ndarray:
    complete = np.ones(len(cache["view_indices"]), dtype=bool)
    for name in (
        "coherent_rel_mse",
        "linear_power_rel_mse",
        "range_power_rel_mse",
        "rf_supported_rel_mse",
    ):
        complete &= np.isfinite(cache[name])
    return complete


def _load_cache(
    path: Path,
    view_indices: np.ndarray,
    positions: np.ndarray,
    *,
    role: str,
    resume: bool,
    identity: Mapping[str, object],
) -> dict[str, np.ndarray]:
    if not resume or not path.is_file():
        return _empty_cache(view_indices, positions, role=role)
    with np.load(path, allow_pickle=False) as archive:
        cache = {name: np.asarray(archive[name]) for name in archive.files}
    required = set(_empty_cache(view_indices, positions, role=role))
    if not required.issubset(cache):
        raise ValueError("resume cache lacks the complete common/RF metric contract")
    if not np.array_equal(cache["view_indices"], view_indices):
        raise ValueError("resume cache IDs disagree with the selected role")
    if str(np.asarray(cache["selected_role"]).reshape(())) != role:
        raise ValueError("resume cache role disagrees with the selected role")
    if str(np.asarray(cache["enabled_metrics"]).reshape(())) != ",".join(_enabled_metrics()):
        raise ValueError("resume cache metric contract disagrees with the evaluator")
    identity_path = path.with_name(path.stem + "_identity.json")
    if not identity_path.is_file():
        raise ValueError("resume cache lacks its identity sidecar")
    with identity_path.open("r", encoding="utf-8") as handle:
        saved_identity = json.load(handle)
    if saved_identity.get("resume_identity") != dict(identity):
        raise ValueError("resume cache identity disagrees with the selected role/checkpoint contract")
    return cache


def _aggregate(cache: Mapping[str, np.ndarray]) -> dict[str, float | int]:
    complete = _complete_mask(cache)
    result: dict[str, float | int] = {
        "views_complete": int(complete.sum()),
        "views_total": int(len(complete)),
    }
    for prefix in ("coherent", "linear_power", "range_power", "rf_supported", "rf_padded"):
        error = float(np.nansum(cache[f"{prefix}_squared_error"]))
        target_key = "coherent_target_energy" if prefix == "coherent" else f"{prefix}_target_squared_norm"
        count_key = "coherent_sample_count" if prefix == "coherent" else f"{prefix}_element_count"
        target = float(np.nansum(cache[target_key]))
        count = int(round(float(np.nansum(cache[count_key]))))
        if target <= 0.0 or count <= 0:
            raise ValueError(f"{prefix} aggregation encountered a non-positive norm/count")
        result[f"{prefix}_squared_error_sum"] = error
        result["coherent_target_energy_sum" if prefix == "coherent" else f"{prefix}_target_squared_norm"] = target
        result[f"{prefix}_sample_count" if prefix == "coherent" else f"{prefix}_element_count"] = count
        result[f"{prefix}_rel_mse"] = pooled_relative_mse(error, target)
        result[f"{prefix}_percent"] = 100.0 * error / target
    padded_target_array = cache["rf_padded_target_squared_norm"]
    padded_count_array = cache["rf_padded_element_count"]
    result["rf_padded_zero_target_view_count"] = int(
        (complete & np.isfinite(padded_count_array) & np.isfinite(padded_target_array) & (padded_target_array <= 0.0)).sum()
    )
    result["rf_padded_empty_view_count"] = int(
        (complete & np.isfinite(padded_count_array) & (padded_count_array == 0.0)).sum()
    )
    result["elapsed_seconds_sum"] = float(np.nansum(cache["elapsed_seconds"]))
    return result


def _validate_production_aggregate(summary: Mapping[str, object], *, role: str) -> None:
    if int(summary.get("views_complete", -1)) != ROLE_COUNT or int(summary.get("views_total", -1)) != ROLE_COUNT:
        raise ValueError(f"complete SpINR smoke evaluation must cover all 1000 {role} views")
    if int(summary.get("coherent_sample_count", -1)) != EXPECTED_VALIDATION_SAMPLES:
        raise ValueError("coherent sample count disagrees with 1000x16x16x600")
    if int(summary.get("range_power_element_count", -1)) != EXPECTED_POWER_ELEMENT_COUNT:
        raise ValueError("range-power ROI cell count disagrees with the frozen contract")
    expected_target = EXPECTED_VALIDATION_POWER_TARGET if role == "validation" else EXPECTED_TEST_POWER_TARGET
    _require_close("range-power target squared norm", summary.get("range_power_target_squared_norm"), expected_target, abs_tol=1.0e-4)
    for prefix in ("coherent", "linear_power", "range_power", "rf_supported", "rf_padded"):
        if float(summary.get(f"{prefix}_rel_mse", -1.0)) < 0.0:
            raise ValueError(f"{prefix} relative MSE is invalid")


def _validate_checkpoint(checkpoint: Mapping[str, object]) -> tuple[Mapping[str, object], Mapping[str, object], float, float]:
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("selected checkpoint has the wrong G96/Gauss2 small-fit format")
    execution = checkpoint.get("execution")
    completed = execution.get("completed_updates") if isinstance(execution, Mapping) else checkpoint.get("completed_updates")
    if isinstance(completed, bool) or int(completed) != CHECKPOINT_UPDATES:
        raise ValueError("selected SpINR smoke checkpoint must contain exactly 60 completed updates")
    recipe = checkpoint.get("recipe")
    if not isinstance(recipe, Mapping):
        raise ValueError("selected checkpoint must use the bounded recipe metadata key")
    network = recipe.get("network")
    operator = recipe.get("operator")
    if not isinstance(network, Mapping) or not isinstance(operator, Mapping):
        raise ValueError("selected G96 checkpoint lacks network/operator recipe metadata")
    expected_network = {
        "input_features": SPINR_STYLE_INPUT_FEATURES,
        "hidden_layers": SPINR_STYLE_HIDDEN_LAYERS,
        "hidden_width": SPINR_STYLE_HIDDEN_WIDTH,
        "signed_real_output": True,
        "learned_gain": False,
        "support_m": SPINR_STYLE_SUPPORT_M,
    }
    for key, expected in expected_network.items():
        if network.get(key) != expected:
            raise ValueError(f"checkpoint network recipe {key} disagrees with the frozen smoke")
    expected_operator = {
        "parent_cell_grid_size": QUADRATURE_PARENT_GRID,
        "physical_volume_weights": True,
        "phase_sign": SPINR_STYLE_PHASE_SIGN,
        "range_model": SPINR_STYLE_RANGE_MODEL,
        "physics_dtype": "float64_complex128",
        "network_dtype": "float32",
        "full_frequency_bins": 600,
        "full_pairs": "16x16",
        "training_quadrature": "tensor_gauss_legendre_2_nodes_per_axis_on_96_cubed_cells",
    }
    for key, expected in expected_operator.items():
        if operator.get(key) != expected:
            raise ValueError(f"checkpoint operator recipe {key} disagrees with the frozen G96 contract")
    worklists = checkpoint.get("worklists")
    if not isinstance(worklists, Mapping):
        raise ValueError("selected checkpoint lacks bounded worklists")
    if len(worklists.get("fit_training_ids", ())) != CHECKPOINT_FIT_TRAIN_COUNT or len(worklists.get("validation_ids", ())) != CHECKPOINT_FIT_VALIDATION_COUNT:
        raise ValueError("selected checkpoint is not the 16-view fit/validation smoke")
    if len(worklists.get("normalization_training_ids", ())) != PARENT_TRAIN_COUNT or len(worklists.get("initialization_training_ids", ())) != 32:
        raise ValueError("selected checkpoint lacks the parent-3200/first-32 normalization worklists")
    for name, expected_count in (
        ("fit_training_ids", CHECKPOINT_FIT_TRAIN_COUNT),
        ("validation_ids", CHECKPOINT_FIT_VALIDATION_COUNT),
        ("normalization_training_ids", PARENT_TRAIN_COUNT),
        ("initialization_training_ids", 32),
    ):
        values = tuple(int(item) for item in worklists[name])
        if len(set(values)) != expected_count or any(value < 0 or value >= RESPONSE_SHAPE[0] for value in values):
            raise ValueError(f"checkpoint worklist {name} is not a distinct in-range source-ID list")
    normalization = checkpoint.get("normalization")
    if not isinstance(normalization, Mapping):
        raise ValueError("selected checkpoint lacks saved normalization metadata")
    training_mean = _finite_positive(normalization.get("training_mean_raw_power"), "training mean raw power")
    initial_scale = _finite_positive(normalization.get("initial_output_scale"), "initial output scale")
    _require_close("saved initial output scale", initial_scale, 3540.655842132721, abs_tol=1.0e-9)
    _require_close("saved training mean raw power", training_mean, 6.667741086506592e-09, abs_tol=1.0e-20)
    state = checkpoint.get("model_state_dict")
    contract = checkpoint.get("sealed_npz_protocol_contract")
    acquisition = checkpoint.get("acquisition_identity")
    if not isinstance(state, Mapping) or not isinstance(contract, Mapping) or not isinstance(acquisition, Mapping):
        raise ValueError("selected checkpoint lacks model/sealed acquisition metadata")
    return state, contract, initial_scale, training_mean


def _make_cache_identity(*, checkpoint: str, npz: str, manifest: str, stats: str, role: str, ids: np.ndarray) -> dict[str, object]:
    return {
        "schema": "rift_spinr_b787_g96_gauss2_eval_v1",
        "checkpoint": os.path.abspath(checkpoint),
        "npz_path": os.path.abspath(npz),
        "role_manifest": os.path.abspath(manifest),
        "stats_path": os.path.abspath(stats),
        "selected_role": role,
        "ordered_source_view_ids": [int(value) for value in ids],
        "enabled_metrics": list(_enabled_metrics()),
        "checkpoint_format": CHECKPOINT_FORMAT,
        "completed_updates": CHECKPOINT_UPDATES,
        "quadrature": {
            "parent_cell_grid_size": QUADRATURE_PARENT_GRID,
            "nodes_per_cell": QUADRATURE_NODES_PER_CELL,
            "point_count": QUADRATURE_POINT_COUNT,
            "physical_volume_weights": True,
            "support_m": QUADRATURE_SUPPORT_M,
        },
    }


def _render_from_frozen_field(
    *,
    points_m: torch.Tensor,
    weights: torch.Tensor,
    frequencies_hz: torch.Tensor,
    kvector: torch.Tensor,
    rx_pos: torch.Tensor,
    tx_pos: torch.Tensor,
) -> torch.Tensor:
    response = range_forward_operator(
        frequencies_hz,
        kvector,
        rx_pos,
        tx_pos,
        points_m,
        weights,
        phase_sign=SPINR_STYLE_PHASE_SIGN,
        pair_chunk=PAIR_TILE,
        point_chunk=RENDERER_POINT_TILE,
        compute_dtype=torch.float64,
        range_model=SPINR_STYLE_RANGE_MODEL,
    )
    if tuple(response.shape) != (600, 16, 16) or response.dtype != torch.complex128:
        raise ValueError("SpINR G96 renderer must return [600,16,16] complex128")
    return response


def _flatten_measured_response(raw: np.ndarray, *, device: torch.device) -> torch.Tensor:
    """Average only chirps and retain Tx-outer/Rx-inner/frequency order."""

    if raw.shape != VIEW_RESPONSE_SHAPE or np.dtype(raw.dtype) != np.dtype(np.complex64):
        raise ValueError(f"raw response must have shape {VIEW_RESPONSE_SHAPE} and dtype complex64")
    averaged = np.asarray(raw.mean(axis=2), dtype=np.complex64)
    return torch.as_tensor(averaged, dtype=torch.complex128, device=device).reshape(-1, 600)


def _summary(
    aggregate: Mapping[str, object], *, identity: Mapping[str, object], role: str,
    response_source: Any, fit_worklists: Mapping[str, object], production: bool,
    peak_power: float, dynamic_range_db: float, training_mean_raw_power: float,
    initial_output_scale: float,
) -> dict[str, object]:
    result = dict(aggregate)
    result.update(
        {
            "schema": "rift_spinr_b787_g96_gauss2_eval_v1",
            "status": "complete_1000_evaluation_of_smoke_trained_model" if production else "incomplete_prefix",
            "method": "SpINR-style neural baseline (G96/Gauss2 bounded smoke)",
            "selected_role": role,
            "ordered_source_view_ids": list(identity["ordered_source_view_ids"]),
            "authorized_response_view_count": int(len(identity["ordered_source_view_ids"])),
            "reserved_test_accessed": role == "test",
            "checkpoint": {
                "path": identity["checkpoint"],
                "format": CHECKPOINT_FORMAT,
                "completed_updates": CHECKPOINT_UPDATES,
                "fit_training_view_count": CHECKPOINT_FIT_TRAIN_COUNT,
                "fit_validation_view_count": CHECKPOINT_FIT_VALIDATION_COUNT,
                "worklists": dict(fit_worklists),
            },
            "normalization": {
                "source": "streamed_all_3200_parent_train_rows",
                "training_mean_raw_power": training_mean_raw_power,
                "initial_output_scale": initial_output_scale,
                "peak_power": peak_power,
                "dynamic_range_db": dynamic_range_db,
            },
            "quadrature": dict(identity["quadrature"]),
            "renderer_contract": {
                "field_evaluated_once_and_reused_across_views": True,
                "neural_point_tile": NEURAL_POINT_TILE,
                "renderer_point_tile": RENDERER_POINT_TILE,
                "pair_chunk": PAIR_TILE,
                "phase_sign": SPINR_STYLE_PHASE_SIGN,
                "range_model": SPINR_STYLE_RANGE_MODEL,
                "physics_dtype": "float64_complex128",
                "network_dtype": "float32",
                "physical_volume_weights": True,
            },
            "response_header": list(RESPONSE_SHAPE),
            "source_response_access_restricted": bool(response_source.response_access_is_restricted),
            "source_authorized_response_view_count": len(response_source.allowed_response_view_indices or ()),
            "native_target_provenance": "raw measured complex response; no learned gain or TEST adjustment",
            "provenance": {
                "npz": identity["npz_path"],
                "role_manifest": identity["role_manifest"],
                "stats": identity["stats_path"],
            },
        }
    )
    if role == "validation":
        result["ordered_validation_view_ids"] = list(identity["ordered_source_view_ids"])
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--npz", default=DEFAULT_NPZ)
    parser.add_argument("--stats", default=DEFAULT_STATS)
    parser.add_argument("--role-manifest", default=DEFAULT_ROLE_MANIFEST)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--label", default="spinr_b787_g96_gauss2")
    parser.add_argument("--role", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--max-views", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.max_views < 0 or args.max_views > ROLE_COUNT or args.save_every <= 0:
        raise ValueError("max-views must lie in [0,1000] and save-every must be positive")
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    checkpoint_path = os.path.abspath(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("SpINR checkpoint must be a mapping")
    state, sealed_contract, initial_output_scale, training_mean_raw_power = _validate_checkpoint(checkpoint)
    roles = sealed_contract.get("role_ids")
    worklists = checkpoint.get("worklists")
    if not isinstance(roles, Mapping) or not isinstance(worklists, Mapping):
        raise ValueError("checkpoint roles/worklists are malformed")
    train_ids = tuple(int(item) for item in roles.get("train", ()))
    validation_ids = tuple(int(item) for item in roles.get("validation", ()))
    if tuple(int(item) for item in worklists["normalization_training_ids"]) != train_ids:
        raise ValueError("checkpoint normalization worklist is not exactly the canonical parent TRAIN role")
    if not set(int(item) for item in worklists["fit_training_ids"]).issubset(set(train_ids)):
        raise ValueError("checkpoint fit worklist is outside the canonical parent TRAIN role")
    if not set(int(item) for item in worklists["validation_ids"]).issubset(set(validation_ids)):
        raise ValueError("checkpoint small-fit validation worklist is outside the canonical validation role")
    if tuple(int(item) for item in worklists["initialization_training_ids"]) != train_ids[:32]:
        raise ValueError("checkpoint initialization worklist is not parent TRAIN[:32]")

    arrays = load_radar_fields_npz(args.npz, load_response=False)
    if tuple(int(value) for value in (arrays.response_shape or ())) != RESPONSE_SHAPE or np.dtype(arrays.response_dtype) != np.dtype(np.complex64):
        raise ValueError("SpINR evaluator requires the canonical sphere10k complex64 response header")
    if not isinstance(sealed_contract, Mapping):
        raise ValueError("checkpoint sealed contract is malformed")
    acquisition_expected = build_spinr_style_acquisition_identity(
        {
            "response": None,
            "meta": arrays.metadata,
            "rx_pos": arrays.rx_pos,
            "tx_pos": arrays.tx_pos,
        },
        sealed_contract,
    )
    validate_spinr_style_acquisition_identity(checkpoint["acquisition_identity"], acquisition_expected)
    role_args = SimpleNamespace(num_train=PARENT_TRAIN_COUNT, num_val=ROLE_COUNT, seed=42, max_views=args.max_views)
    selected_indices = np.asarray(_sealed_role_indices(checkpoint, arrays, role_args, args.role), dtype=np.int64)
    if len(selected_indices) == 0:
        raise ValueError(f"evaluation selected no {args.role} views")
    arrays = restrict_radar_fields_response_views(arrays, selected_indices)

    with open(args.stats, "r", encoding="utf-8") as handle:
        power_stats = json.load(handle)
    peak_power, dynamic_range_db = validate_normalization_stats(
        power_stats,
        checkpoint,
        num_views=arrays.num_views,
    )
    _require_close("power normalization peak", peak_power, EXPECTED_POWER_PEAK, abs_tol=1.0e-22)
    _require_close("power normalization dynamic range", dynamic_range_db, EXPECTED_POWER_DYNAMIC_RANGE_DB)
    frequencies_hz = metadata_frequency_grid(arrays.metadata, dtype=torch.float64).to(device=device)
    kvector = get_kvector(frequencies_hz, cc).to(device=device, dtype=torch.float64)
    ranges = range_bin_centers(arrays.metadata, device=device, dtype=torch.float32)
    rf_grid = make_rf_midpoint_grid(device)
    pair_indices = torch.arange(256, dtype=torch.long, device=device)

    model = SpinrStyleINR().to(device=device, dtype=torch.float32)
    if model.trainable_parameter_count() != SPINR_STYLE_PARAMETER_COUNT:
        raise ValueError("SpINR model parameter count disagrees with the frozen checkpoint recipe")
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    points_m, quadrature_weights = gauss_legendre_cell_grid(
        QUADRATURE_PARENT_GRID,
        nodes_per_cell=QUADRATURE_NODES_PER_CELL,
        support_m=QUADRATURE_SUPPORT_M,
        device=device,
        dtype=torch.float64,
    )
    if points_m.shape != (QUADRATURE_POINT_COUNT, 3) or quadrature_weights.shape != (QUADRATURE_POINT_COUNT,):
        raise ValueError("G96/Gauss2 quadrature shape disagrees with the frozen 7,077,888-point contract")
    field = evaluate_neural_field_tiled(model, points_m, neural_point_tile=NEURAL_POINT_TILE)
    renderer_weights = scale_field_to_renderer_weights(
        field,
        cell_volume_m3=quadrature_weights,
        initial_output_scale=initial_output_scale,
    )
    if not bool(torch.isfinite(renderer_weights).all().item()) or not bool((quadrature_weights > 0).all().item()):
        raise FloatingPointError("SpINR G96 field or physical quadrature weights are invalid")

    identity = _make_cache_identity(
        checkpoint=checkpoint_path,
        npz=args.npz,
        manifest=args.role_manifest,
        stats=args.stats,
        role=args.role,
        ids=selected_indices,
    )
    out_dir = Path(args.out_dir)
    cache_path = out_dir / f"{args.label}_{args.role}_per_view.npz"
    summary_path = out_dir / f"{args.label}_{args.role}_metrics.json"
    identity_path = cache_path.with_name(cache_path.stem + "_identity.json")
    positions = np.asarray(arrays.viewpoint_positions[selected_indices], dtype=np.float64)
    cache = _load_cache(
        cache_path,
        selected_indices,
        positions,
        role=args.role,
        resume=args.resume,
        identity=identity,
    )
    if not np.array_equal(cache["viewpoint_positions"], positions):
        raise ValueError("resume cache viewpoint positions disagree with the selected source IDs")
    complete = _complete_mask(cache)
    pending = np.flatnonzero(~complete)
    started = time.perf_counter()
    print(f"device={device}; role={args.role}; views={len(selected_indices)}; pending={len(pending)}; checkpoint_updates={CHECKPOINT_UPDATES}", flush=True)

    for slot in pending:
        view_index = int(selected_indices[slot])
        view_started = time.perf_counter()
        tx_pos = torch.as_tensor(arrays.tx_pos[view_index], dtype=torch.float64, device=device)
        rx_pos = torch.as_tensor(arrays.rx_pos[view_index], dtype=torch.float64, device=device)
        predicted = _render_from_frozen_field(
            points_m=points_m,
            weights=renderer_weights,
            frequencies_hz=frequencies_hz,
            kvector=kvector,
            rx_pos=rx_pos,
            tx_pos=tx_pos,
        )
        raw = arrays.response_view(view_index)
        if raw.shape != VIEW_RESPONSE_SHAPE or np.dtype(raw.dtype) != np.dtype(np.complex64):
            raise ValueError(f"view {view_index} response disagrees with {VIEW_RESPONSE_SHAPE}/complex64")
        measured_flat = _flatten_measured_response(raw, device=device)
        pred_flat = predicted.permute(2, 1, 0).contiguous().reshape(-1, 600)
        residual = pred_flat - measured_flat
        coherent_error = float(residual.abs().square().sum().item())
        coherent_target = float(measured_flat.abs().square().sum().item())
        coherent_pred = float(pred_flat.abs().square().sum().item())
        coherent_count = int(residual.numel())
        pred_power = torch.fft.ifft(pred_flat, dim=-1).abs().square()
        target_power = response_view_to_range_power(raw, device=device)
        pred_intensity = normalize_power_db(pred_power, peak_power, dynamic_range_db)
        target_intensity = normalize_power_db(target_power, peak_power, dynamic_range_db)
        viewpoint = torch.as_tensor(arrays.viewpoint_positions[view_index], dtype=torch.float32, device=device)
        roi = scene_range_mask(ranges, viewpoint, QUADRATURE_SUPPORT_M, margin=RF_RANGE_MARGIN_M)
        pred_roi = pred_intensity[:, roi]
        target_roi = target_intensity[:, roi]
        range_error = float((pred_roi - target_roi).square().sum().item())
        range_target = float(target_roi.square().sum().item())
        range_count = int(target_roi.numel())
        linear_pred_roi = pred_power[:, roi]
        linear_target_roi = target_power[:, roi]
        linear_error = float((linear_pred_roi - linear_target_roi).square().sum().item())
        linear_target = float(linear_target_roi.square().sum().item())
        rf_supported, _ = rf_supported_mask(
            rf_grid,
            ranges=ranges,
            viewpoint=viewpoint,
            tx_pos=tx_pos.to(dtype=torch.float32),
            rx_pos=rx_pos.to(dtype=torch.float32),
            pair_indices=pair_indices,
            metadata=arrays.metadata,
            extent_m=QUADRATURE_SUPPORT_M,
            range_margin_m=RF_RANGE_MARGIN_M,
            roi=roi,
        )
        rf = partition_squared_error(pred_roi, target_roi, rf_supported)
        values = {
            "coherent_squared_error": coherent_error,
            "coherent_target_energy": coherent_target,
            "coherent_prediction_energy": coherent_pred,
            "coherent_sample_count": coherent_count,
            "coherent_rel_mse": coherent_error / coherent_target,
            "linear_power_squared_error": linear_error,
            "linear_power_target_squared_norm": linear_target,
            "linear_power_element_count": int(linear_target_roi.numel()),
            "linear_power_rel_mse": linear_error / linear_target,
            "range_power_squared_error": range_error,
            "range_power_target_squared_norm": range_target,
            "range_power_element_count": range_count,
            "range_power_rel_mse": range_error / range_target,
            "rf_supported_squared_error": rf["supported_squared_error"],
            "rf_supported_target_squared_norm": rf["supported_target_squared_norm"],
            "rf_supported_element_count": rf["supported_element_count"],
            "rf_supported_rel_mse": pooled_relative_mse(rf["supported_squared_error"], rf["supported_target_squared_norm"]),
            "rf_padded_squared_error": rf["padded_squared_error"],
            "rf_padded_target_squared_norm": rf["padded_target_squared_norm"],
            "rf_padded_element_count": rf["padded_element_count"],
            "rf_padded_rel_mse": diagnostic_relative_mse(rf["padded_squared_error"], rf["padded_target_squared_norm"]),
            "elapsed_seconds": time.perf_counter() - view_started,
        }
        if coherent_target <= 0.0 or range_target <= 0.0 or linear_target <= 0.0 or not all(
            math.isfinite(float(v)) for k, v in values.items() if k != "rf_padded_rel_mse"
        ):
            raise FloatingPointError(f"view {view_index} produced an invalid SpINR metric")
        for key, value in values.items():
            cache[key][slot] = value
        done = int(_complete_mask(cache).sum())
        print(f"[{done:4d}/{len(selected_indices)}] view {view_index:4d}: coherent={100.0 * values['coherent_rel_mse']:9.4f}% power={100.0 * values['range_power_rel_mse']:9.4f}% rf_supported={100.0 * values['rf_supported_rel_mse']:9.4f}%", flush=True)
        if done % args.save_every == 0 or done == len(selected_indices):
            _atomic_save_npz(cache_path, cache)
            _atomic_write_json(identity_path, {"resume_identity": identity})
            _atomic_write_json(summary_path, _summary(_aggregate(cache), identity=identity, role=args.role, response_source=arrays, fit_worklists=checkpoint["worklists"], production=False, peak_power=peak_power, dynamic_range_db=dynamic_range_db, training_mean_raw_power=training_mean_raw_power, initial_output_scale=initial_output_scale))

    _atomic_save_npz(cache_path, cache)
    _atomic_write_json(identity_path, {"resume_identity": identity})
    aggregate = _aggregate(cache)
    production = args.max_views == 0
    if production:
        _validate_production_aggregate(aggregate, role=args.role)
    final = _summary(aggregate, identity=identity, role=args.role, response_source=arrays, fit_worklists=checkpoint["worklists"], production=production, peak_power=peak_power, dynamic_range_db=dynamic_range_db, training_mean_raw_power=training_mean_raw_power, initial_output_scale=initial_output_scale)
    final["elapsed_seconds_this_invocation"] = time.perf_counter() - started
    _atomic_write_json(summary_path, final)
    print("SPINR_B787_G96_GAUSS2_EVAL_PASS" if production else "SPINR_B787_G96_GAUSS2_EVAL_SMOKE_COMPLETE", flush=True)
    print(f"cache: {cache_path}\nsummary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
