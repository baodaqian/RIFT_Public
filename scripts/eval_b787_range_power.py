#!/usr/bin/env python
"""Evaluate coherent checkpoints in Radar Fields' range-power domain.

Radar Fields supervises a 60 dB-normalized range image made by coherently
averaging chirps, IFFT-ing the 600 swept-frequency samples, and squaring the
magnitude.  RIFT and the isotropic SpINR-style checkpoint emit coherent
frequency responses, so exactly the same observable is one IFFT away.

This script renders the configured B787 evaluation role from existing
checkpoints only (it never trains), applies the Radar Fields normalization and
scene-range ROI, and reports the same global relative MSE, RMSE, and PSNR.  A
sealed sphere10k checkpoint binds the ordered 1,000-view role IDs; the legacy
default remains available for unsealed validation checkpoints.  The script also
retains linear-power and coherent-complex diagnostics and a compact per-view
cache used by ``plot_b787_nvs_sphere.py``.

The cache is written every few views and can be resumed after interruption.
Collection evaluation accepts --object/--dataset-root or an explicit
--role-manifest, requires object-bound --stats, and defaults to validation.
Reserved-test evaluation additionally requires --allow-reserved-test.

Example
-------
python scripts/eval_b787_range_power.py \
  --checkpoint training_checkpoints/b787_r6_occ_dc/checkpoint_best.pth.tar \
  --label rift_r6_occ_dc --resume
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rift.calibration import GlobalComplexGain  # noqa: E402
from rift.config import cc  # noqa: E402
from rift.forward_operator import get_kvector  # noqa: E402
from rift.occlusion import (  # noqa: E402
    OcclusionScale,
    array_phase_centre,
    view_transmittance,
)
from rift.radar_fields_dataset import (  # noqa: E402
    build_frequency_grid,
    load_radar_fields_npz,
    normalize_power_db,
    range_bin_centers,
    response_view_to_range_power,
    restrict_radar_fields_response_views,
    scene_range_mask,
    split_view_indices,
    _sealed_stats_cache_matches_train_role,
)
from rift.range_operator import range_forward_operator  # noqa: E402
from rift.sparse_scene import (  # noqa: E402
    AdaptivePointSHScene,
    SHVoxelGridScene,
    VoxelGridScene,
)
from scripts.b787_common_eval_metrics import (  # noqa: E402
    RF_GRID_EXTENT_M,
    RF_RANGE_MARGIN_M,
    make_rf_midpoint_grid,
    partition_squared_error,
    pooled_relative_mse,
    rf_supported_mask,
)


DEFAULT_NPZ = "data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere2k.npz"
DEFAULT_STATS = "training_checkpoints/b787_radar_fields_released/radar_fields_power_stats.json"


def theta_phi(position: np.ndarray) -> tuple[float, float]:
    radius = float(np.linalg.norm(position))
    theta = float(np.arccos(np.clip(position[2] / radius, -1.0, 1.0)))
    phi = float(np.arctan2(position[1], position[0]))
    return theta, phi


def resolve_scene_extent(checkpoint: Mapping[str, object]) -> float:
    """Resolve the fixed scene half-width from checkpoint metadata.

    Point-scene checkpoints do not expose a Python ``model.extent`` attribute,
    so the training loop records a top-level ``None`` for them.  Their
    execution contract still records the parsed scene extent.  Do not infer a
    scoring ROI from learned point support or silently substitute a default
    when both metadata locations are unusable.
    """

    candidates = [("checkpoint.extent", checkpoint.get("extent"))]
    execution_contract = checkpoint.get("execution_contract")
    scene = execution_contract.get("scene") if isinstance(execution_contract, Mapping) else None
    candidates.append(
        (
            "checkpoint.execution_contract.scene.extent_m",
            scene.get("extent_m") if isinstance(scene, Mapping) else None,
        )
    )
    for label, raw in candidates:
        if isinstance(raw, (bool, np.bool_)):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            return value
    raise ValueError(
        "checkpoint lacks a finite positive scene extent; expected either "
        "checkpoint.extent or execution_contract.scene.extent_m"
    )


def _ordered_role_ids(
    checkpoint: Mapping[str, object],
    role: str,
    *,
    num_views: int,
    expected_count: int | None = None,
) -> np.ndarray:
    """Decode one ordered role from a sealed checkpoint contract."""

    sealed = checkpoint.get("sealed_npz_protocol_contract")
    roles = sealed.get("role_ids") if isinstance(sealed, Mapping) else None
    if not isinstance(roles, Mapping) or role not in roles:
        raise ValueError(
            "sealed checkpoint lacks ordered role IDs for "
            f"{role!r}"
        )
    raw = roles[role]
    if isinstance(raw, (str, bytes)):
        raise ValueError(f"sealed checkpoint role {role!r} must be an integer-ID sequence")
    try:
        values = tuple(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"sealed checkpoint role {role!r} must be an integer-ID sequence"
        ) from exc
    decoded = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"sealed checkpoint role {role!r} contains a boolean ID")
        try:
            integer = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"sealed checkpoint role {role!r} contains a non-integer ID"
            ) from exc
        if integer != value or not 0 <= integer < int(num_views):
            raise ValueError(
                f"sealed checkpoint role {role!r} contains invalid source-view ID {value!r}"
            )
        decoded.append(integer)
    if len(set(decoded)) != len(decoded):
        raise ValueError(f"sealed checkpoint role {role!r} contains duplicate source-view IDs")
    if expected_count is not None and len(decoded) != int(expected_count):
        raise ValueError(
            f"sealed checkpoint role {role!r} has {len(decoded)} IDs; "
            f"expected {expected_count}"
        )
    return np.asarray(decoded, dtype=np.int64)


def _sealed_role_indices(
    checkpoint: Mapping[str, object],
    arrays,
    args: argparse.Namespace,
    role: str,
) -> np.ndarray:
    """Bind evaluation to one ordered canonical sealed role."""

    if role not in ("validation", "test"):
        raise ValueError(f"unsupported B787 evaluation role {role!r}")

    sealed = checkpoint.get("sealed_npz_protocol_contract")
    if sealed is None:
        if role == "test":
            raise ValueError(
                "TEST evaluation requires a sealed checkpoint role contract; "
                "refusing to infer a historical test split"
            )
        _, val_indices, _ = split_view_indices(
            arrays.num_views,
            args.num_train,
            args.num_val,
            0,
            args.seed,
            val_from_tail=True,
        )
        return val_indices[: args.max_views] if args.max_views > 0 else val_indices

    if (args.num_train, args.num_val, args.seed) != (3200, 1000, 42):
        raise ValueError(
            "sealed B787 evaluation requires --num-train 3200, --num-val 1000, --seed 42"
        )
    raw_shape = sealed.get("response_shape")
    if isinstance(raw_shape, (str, bytes)):
        raise ValueError("sealed checkpoint response_shape must be a five-dimensional integer sequence")
    try:
        sealed_shape = tuple(int(value) for value in raw_shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "sealed checkpoint response_shape must be a five-dimensional integer sequence"
        ) from exc
    actual_shape = tuple(int(value) for value in (arrays.response_shape or ()))
    if sealed_shape != actual_shape or sealed_shape != (10_000, 16, 16, 1, 600):
        raise ValueError(
            "sealed B787 checkpoint/NPZ response headers must match "
            "sphere10k [10000,16,16,1,600]"
        )
    sealed_dtype = sealed.get("response_dtype")
    if sealed_dtype is not None and str(sealed_dtype) != str(arrays.response_dtype):
        raise ValueError(
            "sealed checkpoint response_dtype disagrees with the NPZ response header"
        )

    train_indices = _ordered_role_ids(
        checkpoint, "train", num_views=arrays.num_views, expected_count=3200
    )
    validation_indices = _ordered_role_ids(
        checkpoint, "validation", num_views=arrays.num_views, expected_count=1000
    )
    test_indices = _ordered_role_ids(
        checkpoint, "reserved_test", num_views=arrays.num_views, expected_count=1000
    )
    unused_indices = _ordered_role_ids(
        checkpoint, "unused", num_views=arrays.num_views, expected_count=4800
    )
    all_roles = np.concatenate(
        (train_indices, validation_indices, test_indices, unused_indices)
    )
    if not np.array_equal(np.sort(all_roles), np.arange(arrays.num_views, dtype=np.int64)):
        raise ValueError("sealed checkpoint roles do not form a complete source-view partition")

    _, derived_val, _ = split_view_indices(
        arrays.num_views,
        args.num_train,
        args.num_val,
        0,
        args.seed,
        val_from_tail=True,
    )
    if not np.array_equal(validation_indices, derived_val):
        raise ValueError(
            "sealed checkpoint validation IDs disagree with the requested seed-42 fixed-tail split"
        )
    selected_role = validation_indices if role == "validation" else test_indices
    selected = selected_role[: args.max_views] if args.max_views > 0 else selected_role
    if len(selected) == 0:
        raise ValueError(f"evaluation selected no {role} views")
    return selected


def _sealed_validation_indices(
    checkpoint: Mapping[str, object],
    arrays,
    args: argparse.Namespace,
) -> np.ndarray:
    """Backward-compatible wrapper for the canonical validation role."""

    return _sealed_role_indices(checkpoint, arrays, args, "validation")


def _selected_role_indices(
    checkpoint: Mapping[str, object],
    arrays,
    args: argparse.Namespace,
) -> np.ndarray:
    """Resolve the explicit evaluation role without guessing a split."""

    return _sealed_role_indices(checkpoint, arrays, args, str(args.role))


def validate_normalization_stats(
    stats: Mapping[str, object],
    checkpoint: Mapping[str, object],
    *,
    num_views: int,
) -> tuple[float, float]:
    """Validate fixed dB normalization before any metric computation."""

    try:
        peak_power = float(stats["peak_power"])
        dynamic_range_db = float(stats["dynamic_range_db"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("normalization stats must contain numeric peak_power and dynamic_range_db") from exc
    if not math.isfinite(peak_power) or peak_power <= 0.0:
        raise ValueError(f"normalization stats peak_power must be finite and positive, got {peak_power!r}")
    if not math.isfinite(dynamic_range_db) or dynamic_range_db <= 0.0:
        raise ValueError(
            "normalization stats dynamic_range_db must be finite and positive, "
            f"got {dynamic_range_db!r}"
        )

    sealed = checkpoint.get("sealed_npz_protocol_contract")
    if isinstance(sealed, Mapping) and "dataset_identity" in sealed:
        from rift.rift_dataset import validate_checkpoint_object
        validate_checkpoint_object(stats, sealed)
    if sealed is None:
        return peak_power, dynamic_range_db

    train_indices = _ordered_role_ids(
        checkpoint, "train", num_views=num_views, expected_count=3200
    )
    expected_train = train_indices.tolist()
    raw_stats_train = stats.get("train_view_indices")
    raw_stats_scan = stats.get("normalization_scan_view_indices")
    provenance = stats.get("normalization_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("sealed normalization stats lack normalization_provenance")
    if provenance.get("normalization_scan_role") != "train":
        raise ValueError("sealed normalization scan is not declared as a train-role scan")

    def decode_stats_ids(raw: object, label: str) -> list[int]:
        if isinstance(raw, (str, bytes)):
            raise ValueError(f"sealed normalization stats {label} must be an integer-ID sequence")
        try:
            values = tuple(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"sealed normalization stats {label} must be an integer-ID sequence"
            ) from exc
        decoded = []
        for value in values:
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"sealed normalization stats {label} contains a boolean ID")
            try:
                integer = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"sealed normalization stats {label} contains a non-integer ID") from exc
            if integer != value or not 0 <= integer < int(num_views):
                raise ValueError(
                    f"sealed normalization stats {label} contains invalid source-view ID {value!r}"
                )
            decoded.append(integer)
        if len(set(decoded)) != len(decoded):
            raise ValueError(f"sealed normalization stats {label} contains duplicate source-view IDs")
        return decoded

    stats_train = decode_stats_ids(raw_stats_train, "train_view_indices")
    stats_scan = decode_stats_ids(raw_stats_scan, "normalization_scan_view_indices")
    provenance_train = decode_stats_ids(
        provenance.get("train_view_indices"), "provenance.train_view_indices"
    )
    provenance_scan = decode_stats_ids(
        provenance.get("normalization_scan_view_indices"),
        "provenance.normalization_scan_view_indices",
    )
    if stats_train != expected_train or provenance_train != expected_train:
        raise ValueError("normalization stats train IDs disagree with the checkpoint TRAIN role")
    if stats_scan != provenance_scan:
        raise ValueError("normalization stats top-level and provenance scan IDs disagree")
    if not stats_scan or not set(stats_scan).issubset(set(expected_train)):
        raise ValueError("normalization scan IDs must be a nonempty subset of the TRAIN role")
    if not _sealed_stats_cache_matches_train_role(
        stats,
        requested_train_indices=expected_train,
        requested_scan_indices=stats_scan,
        num_views=num_views,
    ):
        raise ValueError(
            "normalization stats lack the existing sealed train-only provenance contract"
        )
    return peak_power, dynamic_range_db


def load_scene(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"]
    if checkpoint.get("scene_repr") == "point_sh" or "anchors" in state:
        model = AdaptivePointSHScene.from_state(state, device).to(device)
        model.eval()
        gain = None
        if checkpoint.get("gain_state_dict") is not None:
            gain = GlobalComplexGain().to(device)
            gain.load_state_dict(checkpoint["gain_state_dict"])
            gain.eval()
        return checkpoint, model, gain, None

    if checkpoint.get("scene_repr") == "grid":
        w_re = state.get("w_re")
        w_im = state.get("w_im")
        if not isinstance(w_re, torch.Tensor) or not isinstance(w_im, torch.Tensor):
            raise ValueError("scalar grid checkpoint must contain tensor w_re and w_im")
        if w_re.ndim != 3 or w_im.shape != w_re.shape:
            raise ValueError(
                "scalar grid checkpoint must contain matching three-dimensional w_re/w_im"
            )
        raw_granularity = checkpoint.get("granularity")
        if raw_granularity is None:
            raise ValueError("scalar grid checkpoint lacks granularity metadata")
        granularity = int(raw_granularity)
        if tuple(int(value) for value in w_re.shape) != (granularity,) * 3:
            raise ValueError(
                "scalar grid checkpoint weight shape disagrees with granularity metadata"
            )
        extent = resolve_scene_extent(checkpoint)
        model = VoxelGridScene(
            granularity,
            extent,
            device,
            init_scale=0.0,
        ).to(device)
        state_on_device = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in state.items()
        }
        model.load_state_dict(state_on_device, strict=True)
        model.eval()
        gain = None
        if checkpoint.get("gain_state_dict") is not None:
            gain = GlobalComplexGain().to(device)
            gain.load_state_dict(checkpoint["gain_state_dict"])
            gain.eval()
        return checkpoint, model, gain, None

    w_re = state["w_re"]
    if w_re.ndim != 4:
        raise ValueError(f"expected grid-SH weights [G,G,G,C], got {tuple(w_re.shape)}")
    granularity = int(w_re.shape[0])
    extent = resolve_scene_extent(checkpoint)
    n_basis = int(w_re.shape[-1])
    max_degree = int(round(math.sqrt(n_basis) - 1))
    if (max_degree + 1) ** 2 != n_basis:
        raise ValueError(f"checkpoint has {n_basis} coefficients, not a complete SH basis")

    model = SHVoxelGridScene(
        granularity,
        extent,
        device,
        max_degree=max_degree,
        init_degree=0,
        init_scale=0.0,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    gain = None
    if checkpoint.get("gain_state_dict") is not None:
        gain = GlobalComplexGain().to(device)
        gain.load_state_dict(checkpoint["gain_state_dict"])
        gain.eval()

    occlusion = None
    if checkpoint.get("occlusion_state_dict") is not None:
        scale = OcclusionScale(0.1, learnable=False, device=device)
        scale.load_state_dict(checkpoint["occlusion_state_dict"])
        occlusion = {
            "scale": scale,
            "key": checkpoint.get("occlusion_key", "energy"),
            "n_steps": None,
            "step_frac": 0.5,
            "point_chunk": 16384,
        }

    return checkpoint, model, gain, occlusion


def active_scatterers_for_view(
    model: torch.nn.Module,
    dtheta: torch.Tensor,
    dphi: torch.Tensor,
):
    """Dispatch isotropic scalar grids without introducing an SH Y00 factor."""

    if isinstance(model, VoxelGridScene):
        return model.active_scatterers()
    return model.active_scatterers(dtheta, dphi)


def empty_cache(
    views: np.ndarray,
    positions: np.ndarray,
    num_freq: int,
    *,
    role: str = "validation",
    rf_supported_metrics: bool = False,
) -> dict[str, np.ndarray]:
    n = len(views)
    nan = lambda: np.full(n, np.nan, dtype=np.float64)
    cache = {
        "view_indices": views.astype(np.int64),
        "viewpoint_positions": positions.astype(np.float64),
        "selected_role": np.asarray([str(role)]),
        "target_signal": nan(),
        "pred_signal": nan(),
        "range_power_rel_mse": nan(),
        "linear_power_rel_mse": nan(),
        "coherent_rel_mse": nan(),
        "sq_error_db": nan(),
        "target_sq_db": nan(),
        "count_db": nan(),
        "sq_error_linear": nan(),
        "target_sq_linear": nan(),
        "sq_error_complex": nan(),
        "target_sq_complex": nan(),
        "target_profile": np.full((n, num_freq), np.nan, dtype=np.float32),
        "pred_profile": np.full((n, num_freq), np.nan, dtype=np.float32),
        "roi_mask": np.zeros((n, num_freq), dtype=bool),
    }
    if rf_supported_metrics:
        for name in (
            "rf_supported_squared_error_db",
            "rf_supported_target_sq_db",
            "rf_padded_squared_error_db",
            "rf_padded_target_sq_db",
        ):
            cache[name] = nan()
        for name in ("rf_supported_count_db", "rf_padded_count_db"):
            cache[name] = nan()
    return cache


def load_or_initialize_cache(
    cache_path: Path,
    views: np.ndarray,
    positions: np.ndarray,
    num_freq: int,
    resume: bool,
    *,
    role: str = "validation",
    rf_supported_metrics: bool = False,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, np.ndarray]:
    identity_text = json.dumps(provenance, sort_keys=True) if provenance is not None else None
    if resume and cache_path.exists():
        with np.load(cache_path) as saved:
            cache = {key: saved[key] for key in saved.files}
        if identity_text is not None and (
            "collection_provenance" not in cache
            or str(cache["collection_provenance"].item()) != identity_text
        ):
            raise ValueError("Cached evaluation belongs to a different object/checkpoint/normalization")
        if not np.array_equal(cache["view_indices"], views):
            raise ValueError(f"cached {role} split in {cache_path} does not match")
        if "selected_role" in cache:
            cached_roles = tuple(str(value) for value in cache["selected_role"].reshape(-1))
            if cached_roles != (str(role),):
                raise ValueError(
                    f"cached evaluation role in {cache_path} does not match {role!r}"
                )
        elif role == "test" or rf_supported_metrics:
            raise ValueError(
                f"cached evaluation in {cache_path} lacks explicit role/mask metadata; "
                "start a fresh cache"
            )
        if rf_supported_metrics:
            required = {
                "rf_supported_squared_error_db",
                "rf_supported_target_sq_db",
                "rf_supported_count_db",
                "rf_padded_squared_error_db",
                "rf_padded_target_sq_db",
                "rf_padded_count_db",
            }
            missing = sorted(required.difference(cache))
            if missing:
                raise ValueError(
                    f"cached evaluation in {cache_path} lacks RF-supported statistics: {missing}"
                )
        return cache
    cache = empty_cache(
        views,
        positions,
        num_freq,
        role=role,
        rf_supported_metrics=rf_supported_metrics,
    )
    if identity_text is not None:
        cache["collection_provenance"] = np.asarray(identity_text)
    return cache


def save_cache(path: Path, cache: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **cache)
    os.replace(temporary, path)


def sum_finite(values: np.ndarray) -> float:
    return float(np.nansum(values))


def aggregate(cache: dict[str, np.ndarray]) -> dict[str, float | int]:
    complete = np.isfinite(cache["range_power_rel_mse"])
    sq_db = sum_finite(cache["sq_error_db"])
    target_db = sum_finite(cache["target_sq_db"])
    count_db = sum_finite(cache["count_db"])
    sq_linear = sum_finite(cache["sq_error_linear"])
    target_linear = sum_finite(cache["target_sq_linear"])
    sq_complex = sum_finite(cache["sq_error_complex"])
    target_complex = sum_finite(cache["target_sq_complex"])
    mse = sq_db / max(count_db, 1.0)
    result: dict[str, float | int] = {
        "views_complete": int(complete.sum()),
        "views_total": int(len(complete)),
        "normalized_range_power_rel_mse": sq_db / max(target_db, 1.0e-30),
        "normalized_range_power_rmse": math.sqrt(mse),
        "normalized_range_power_psnr_db": -10.0 * math.log10(max(mse, 1.0e-30)),
        "normalized_range_power_squared_error": sq_db,
        "normalized_range_power_target_squared_norm": target_db,
        "normalized_range_power_element_count": int(count_db),
        "linear_range_power_rel_mse": sq_linear / max(target_linear, 1.0e-30),
        "coherent_complex_rel_mse": sq_complex / max(target_complex, 1.0e-30),
    }
    if "rf_supported_squared_error_db" in cache:
        supported_error = sum_finite(cache["rf_supported_squared_error_db"])
        supported_target = sum_finite(cache["rf_supported_target_sq_db"])
        supported_count = sum_finite(cache["rf_supported_count_db"])
        padded_error = sum_finite(cache["rf_padded_squared_error_db"])
        padded_target = sum_finite(cache["rf_padded_target_sq_db"])
        padded_count = sum_finite(cache["rf_padded_count_db"])
        result.update(
            {
                "rf_supported_normalized_range_power_rel_mse": pooled_relative_mse(
                    supported_error, supported_target
                ),
                "rf_supported_normalized_range_power_squared_error": supported_error,
                "rf_supported_normalized_range_power_target_squared_norm": supported_target,
                "rf_supported_normalized_range_power_element_count": int(supported_count),
                "rf_padded_normalized_range_power_rel_mse": (
                    pooled_relative_mse(padded_error, padded_target)
                    if padded_target > 0.0 and padded_count > 0.0
                    else None
                ),
                "rf_padded_normalized_range_power_squared_error": padded_error,
                "rf_padded_normalized_range_power_target_squared_norm": padded_target,
                "rf_padded_normalized_range_power_element_count": int(padded_count),
                "rf_supported_mask_definition": "cell_mass[:, roi] > 0 from RF 48^3 midpoint grid and bistatic range-cell interpolation",
            }
        )
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--role",
        choices=("train", "validation", "val", "test", "reserved_test"),
        default="validation",
        help="sealed source-view role to evaluate (default: validation)",
    )
    parser.add_argument("--object")
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parents[1] / "data/RIFT_dataset")
    parser.add_argument("--role-manifest")
    parser.add_argument("--allow-reserved-test", action="store_true")
    parser.add_argument("--npz-path")
    parser.add_argument("--stats")
    parser.add_argument("--out-dir")
    parser.add_argument("--num-train", type=int, default=1800)
    parser.add_argument("--num-val", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-views", type=int, default=0,
                        help="debug/benchmark cap; 0 evaluates all selected role views")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--pair-chunk", type=int, default=64)
    parser.add_argument("--point-chunk", type=int, default=262144)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument(
        "--rf-supported-metrics",
        action="store_true",
        help="also partition normalized range-power error by the deterministic RF support mask",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="evaluation device; use cuda for large point-SH checkpoints",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.max_views < 0 or args.save_every <= 0:
        parser.error("--max-views must be nonnegative and --save-every positive")
    args.role = {"val": "validation", "reserved_test": "test"}.get(args.role, args.role)
    args.collection_mode = args.object is not None or args.role_manifest is not None
    if args.collection_mode:
        from rift.rift_dataset import resolve_object_inputs
        if args.stats is None:
            parser.error("collection evaluation requires explicit --stats for the selected object")
        if args.role == "test" and not args.allow_reserved_test:
            parser.error("collection test evaluation requires --allow-reserved-test")
        args.npz_path, args.role_manifest = map(str, resolve_object_inputs(
            object_name=args.object, dataset_root=args.dataset_root,
            npz_path=args.npz_path, role_manifest_path=args.role_manifest))
    else:
        args.npz_path = args.npz_path or DEFAULT_NPZ
        args.stats = args.stats or DEFAULT_STATS
        if args.role == "train":
            parser.error("train-role evaluation requires an object-bound collection manifest")
    args.out_dir = args.out_dir or (str(Path(args.checkpoint).parent / "range_power_readout")
                                   if args.collection_mode else "figures/b787_range_power")
    return args


def collection_evaluation_inputs(args, checkpoint, stats):
    """Validate object/checkpoint/stats before exposing any response row."""
    from rift.rift_dataset import (load_object_contract, evaluation_role_indices,
                                   validate_checkpoint_object, object_identity, collection_contract)
    public, contract = load_object_contract(args.npz_path, args.role_manifest,
        response_roles=(args.role,), allow_reserved_test=args.allow_reserved_test)
    if args.object is not None:
        validate_checkpoint_object(object_identity(args.object), contract)
    validate_checkpoint_object(checkpoint, contract)
    if collection_contract(checkpoint.get("sealed_npz_protocol_contract", {})) != collection_contract(contract):
        raise ValueError("Checkpoint acquisition/role contract disagrees with the registered object")
    validate_checkpoint_object(stats, contract)
    from rift.radar_fields_dataset import from_collection_arrays, validate_power_stats_acquisition
    args.collection_arrays = from_collection_arrays(public, contract)
    validate_power_stats_acquisition(stats, args.collection_arrays.acquisition_identity)
    selected = evaluation_role_indices(contract, args.role,
        allow_reserved_test=args.allow_reserved_test)
    return contract, selected[:args.max_views] if args.max_views else selected


@torch.inference_mode()
def main(argv=None) -> None:
    args = parse_args(argv)

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    print(f"device={device}; torch threads={torch.get_num_threads()}", flush=True)

    checkpoint, model, gain, occlusion = load_scene(args.checkpoint, device)
    extent = resolve_scene_extent(checkpoint)
    with open(args.stats, "r", encoding="utf-8") as handle:
        stats = json.load(handle)
    collection_contract = None
    if args.collection_mode:
        collection_contract, selected_indices = collection_evaluation_inputs(args, checkpoint, stats)
        args.num_train = len(collection_contract["role_ids"]["train"])
        args.num_val = len(collection_contract["role_ids"]["validation"])
    elif "dataset_identity" in checkpoint.get("sealed_npz_protocol_contract", {}):
        raise ValueError("A collection checkpoint requires --object or --role-manifest")
    arrays = (args.collection_arrays if collection_contract is not None
              else load_radar_fields_npz(args.npz_path, load_response=False))
    if collection_contract is None:
        selected_indices = _selected_role_indices(checkpoint, arrays, args)
    arrays = restrict_radar_fields_response_views(arrays, selected_indices)
    peak_power, dynamic_range_db = validate_normalization_stats(
        stats, checkpoint, num_views=arrays.num_views
    )
    range_model = checkpoint.get("range_model", "sum2")
    freqs = torch.as_tensor(
        build_frequency_grid(arrays.metadata), dtype=torch.float32, device=device
    )
    kvector = get_kvector(freqs, cc)
    ranges = range_bin_centers(arrays.metadata, device=device, dtype=torch.float32)
    if args.rf_supported_metrics and (
        arrays.num_tx,
        arrays.num_rx,
        arrays.num_freq,
    ) != (16, 16, 600):
        raise ValueError(
            "RF-supported B787 metrics require the canonical 16x16x600 acquisition"
        )
    if args.rf_supported_metrics and not math.isclose(
        extent, RF_GRID_EXTENT_M, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError(
            "RF-supported B787 metrics require checkpoint scene extent 0.15 m"
        )
    rf_grid = make_rf_midpoint_grid(device) if args.rf_supported_metrics else None
    pair_indices = (
        torch.arange(256, dtype=torch.long, device=device)
        if args.rf_supported_metrics
        else None
    )

    out_dir = Path(args.out_dir)
    cache_path = out_dir / f"{args.label}_per_view.npz"
    summary_path = out_dir / f"{args.label}_metrics.json"
    cache = load_or_initialize_cache(
        cache_path,
        selected_indices,
        arrays.viewpoint_positions[selected_indices],
        arrays.num_freq,
        args.resume,
        role=args.role,
        rf_supported_metrics=args.rf_supported_metrics,
        provenance=({"contract": collection_contract, "checkpoint": str(Path(args.checkpoint).resolve()),
                     "stats": stats} if collection_contract is not None else None),
    )

    active = int(model.active_mask.sum().item())
    gain_value = gain.gain_value() if gain is not None else complex(1.0)
    occlusion_value = occlusion["scale"].value if occlusion is not None else None
    scene_repr = str(checkpoint.get("scene_repr", "grid_sh"))
    spatial_summary = (
        f"G={model.granularity}"
        if hasattr(model, "granularity")
        else f"allocated={model.active_mask.numel():,}"
    )
    degree_summary = getattr(model, "max_degree", "scalar")
    print(
        f"{args.label}: epoch={checkpoint.get('epoch')} repr={scene_repr} {spatial_summary} "
        f"degree={degree_summary} active={active:,} gain={gain_value:.5g} "
        f"occlusion={occlusion_value} range_model={range_model}",
        flush=True,
    )

    pending_slots = np.flatnonzero(~np.isfinite(cache["range_power_rel_mse"]))
    pending_view_ids = selected_indices[pending_slots]
    slot_by_view = {
        int(view_index): int(slot)
        for slot, view_index in zip(pending_slots, pending_view_ids)
    }
    started = time.perf_counter()

    def current_summary() -> dict[str, object]:
        result = aggregate(cache)
        result.update(
            {
                "status": (
                    "production_complete"
                    if args.max_views == 0
                    and result["views_complete"] == result["views_total"] == (
                        len(collection_contract["role_ids"]["reserved_test" if args.role == "test" else args.role])
                        if collection_contract else 1000)
                    else "incomplete_or_smoke"
                ),
                "label": args.label,
                "checkpoint": os.path.abspath(args.checkpoint),
                "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
                "checkpoint_best_epoch": (
                    int(checkpoint["best_epoch"])
                    if checkpoint.get("best_epoch") is not None
                    else None
                ),
                "checkpoint_val_rel_mse": (
                    float(checkpoint["val_rel_mse"])
                    if checkpoint.get("val_rel_mse") is not None
                    else None
                ),
                "npz_path": os.path.abspath(args.npz_path),
                "stats_path": os.path.abspath(args.stats),
                "dataset_identity": collection_contract["dataset_identity"] if collection_contract else None,
                "selected_role": args.role,
                "reserved_test_accessed": args.role == "test",
                "ordered_source_view_ids": selected_indices.tolist(),
                "validation_split": (
                    "registered object-bound role" if collection_contract else (
                        "seed-42 fixed permutation tail" if args.role == "validation"
                        else "sealed checkpoint reserved_test role")
                ),
                "num_train": args.num_train,
                "num_val": args.num_val,
                "num_selected": int(len(selected_indices)),
                "peak_power": peak_power,
                "dynamic_range_db": dynamic_range_db,
                "target_range_power_dtype": "complex64 (Radar Fields contract)",
                "range_margin_m": RF_RANGE_MARGIN_M,
                "active_scatterers": active,
                "scene_repr": scene_repr,
                "device": str(device),
                "rf_supported_metrics_enabled": bool(args.rf_supported_metrics),
                "rf_supported_metrics_contract": (
                    {
                        "mask_definition": "cell_mass[:, roi] > 0",
                        "grid": "generate_dynamic_grid(48, 0.15, jitter=False) midpoint grid",
                        "grid_dtype": "float32",
                        "pair_chunk": 8,
                        "extent_m": 0.15,
                        "range_margin_m": 0.05,
                        "learned_support_used": False,
                        "target_dependent": False,
                    }
                    if args.rf_supported_metrics
                    else None
                ),
                "elapsed_seconds_this_invocation": time.perf_counter() - started,
            }
        )
        return result

    for pending_number, (view_index, response_view) in enumerate(
        arrays.iter_response_views(pending_view_ids), start=1
    ):
        view_index = int(view_index)
        slot = slot_by_view[view_index]
        view_started = time.perf_counter()
        viewpoint_np = arrays.viewpoint_positions[view_index]
        theta, phi = theta_phi(viewpoint_np)
        dtheta = torch.tensor([[theta]], dtype=torch.float32, device=device)
        dphi = torch.tensor([[phi]], dtype=torch.float32, device=device)
        rx_pos = torch.as_tensor(
            arrays.rx_pos[view_index], dtype=torch.float32, device=device
        )
        tx_pos = torch.as_tensor(
            arrays.tx_pos[view_index], dtype=torch.float32, device=device
        )

        positions, weights = active_scatterers_for_view(model, dtheta, dphi)
        # At the learned Round-6 solution zeta is ~5e-16.  fp32 marching then
        # returns exp(-tau) == 1 exactly; skip millions of identical lookups.
        if occlusion is not None and occlusion_value >= 1.0e-7:
            transmittance = view_transmittance(
                model,
                occlusion["scale"],
                array_phase_centre(rx_pos, tx_pos),
                key=occlusion["key"],
                n_steps=occlusion["n_steps"],
                step_frac=occlusion["step_frac"],
                point_chunk=occlusion["point_chunk"],
            )
            weights = weights * transmittance.to(weights.dtype)

        prediction = range_forward_operator(
            freqs,
            kvector,
            rx_pos,
            tx_pos,
            positions,
            weights,
            phase_sign=-1.0,
            compute_dtype=torch.float64,
            pair_chunk=args.pair_chunk,
            point_chunk=args.point_chunk,
            range_model=range_model,
        )
        if gain is not None:
            prediction = gain(prediction)
        pred_flat = prediction.permute(2, 1, 0).reshape(-1, arrays.num_freq)

        measured_np = response_view.mean(axis=2).reshape(-1, arrays.num_freq)
        measured = torch.as_tensor(measured_np, dtype=torch.complex128, device=device)
        pred_power = torch.fft.ifft(pred_flat, dim=-1).abs().square()
        # Reuse Radar Fields' complex64 target conversion verbatim.  The
        # coherent diagnostic below remains complex128, but the primary
        # normalized range-power score now has bitwise-identical target
        # preprocessing to the power-only baseline.
        target_power = response_view_to_range_power(
            response_view, device=device
        )
        pred_intensity = normalize_power_db(pred_power, peak_power, dynamic_range_db)
        target_intensity = normalize_power_db(target_power, peak_power, dynamic_range_db)

        viewpoint = torch.as_tensor(viewpoint_np, dtype=torch.float32, device=device)
        roi = scene_range_mask(ranges, viewpoint, extent, margin=0.05)
        pred_roi = pred_intensity[:, roi]
        target_roi = target_intensity[:, roi]
        pred_linear_roi = pred_power[:, roi]
        target_linear_roi = target_power[:, roi]

        rf_partition = None
        if args.rf_supported_metrics:
            if rf_grid is None or pair_indices is None:
                raise RuntimeError("RF-supported metrics were enabled without mask inputs")
            supported_mask, _ = rf_supported_mask(
                rf_grid,
                ranges=ranges,
                viewpoint=viewpoint,
                tx_pos=tx_pos,
                rx_pos=rx_pos,
                pair_indices=pair_indices,
                metadata=arrays.metadata,
                extent_m=extent,
                range_margin_m=RF_RANGE_MARGIN_M,
                roi=roi,
            )
            rf_partition = partition_squared_error(
                pred_roi,
                target_roi,
                supported_mask,
            )

        sq_db = float((pred_roi - target_roi).square().sum())
        target_sq_db = float(target_roi.square().sum())
        sq_linear = float((pred_linear_roi - target_linear_roi).square().sum())
        target_sq_linear = float(target_linear_roi.square().sum())
        sq_complex = float((pred_flat - measured).abs().square().sum())
        target_sq_complex = float(measured.abs().square().sum())

        cache["target_signal"][slot] = float(target_roi.mean())
        cache["pred_signal"][slot] = float(pred_roi.mean())
        cache["range_power_rel_mse"][slot] = sq_db / max(target_sq_db, 1.0e-30)
        cache["linear_power_rel_mse"][slot] = sq_linear / max(target_sq_linear, 1.0e-30)
        cache["coherent_rel_mse"][slot] = sq_complex / max(target_sq_complex, 1.0e-30)
        cache["sq_error_db"][slot] = sq_db
        cache["target_sq_db"][slot] = target_sq_db
        cache["count_db"][slot] = int(pred_roi.numel())
        cache["sq_error_linear"][slot] = sq_linear
        cache["target_sq_linear"][slot] = target_sq_linear
        cache["sq_error_complex"][slot] = sq_complex
        cache["target_sq_complex"][slot] = target_sq_complex
        if rf_partition is not None:
            cache["rf_supported_squared_error_db"][slot] = rf_partition[
                "supported_squared_error"
            ]
            cache["rf_supported_target_sq_db"][slot] = rf_partition[
                "supported_target_squared_norm"
            ]
            cache["rf_supported_count_db"][slot] = rf_partition["supported_element_count"]
            cache["rf_padded_squared_error_db"][slot] = rf_partition[
                "padded_squared_error"
            ]
            cache["rf_padded_target_sq_db"][slot] = rf_partition[
                "padded_target_squared_norm"
            ]
            cache["rf_padded_count_db"][slot] = rf_partition["padded_element_count"]
        cache["target_profile"][slot] = (
            target_intensity.mean(dim=0).float().cpu().numpy()
        )
        cache["pred_profile"][slot] = (
            pred_intensity.mean(dim=0).float().cpu().numpy()
        )
        cache["roi_mask"][slot] = roi.cpu().numpy()

        elapsed_view = time.perf_counter() - view_started
        done_total = int(np.isfinite(cache["range_power_rel_mse"]).sum())
        print(
            f"[{done_total:3d}/{len(selected_indices)}] view {view_index:4d}  "
            f"power={100 * cache['range_power_rel_mse'][slot]:8.3f}%  "
            f"complex={100 * cache['coherent_rel_mse'][slot]:7.3f}%  "
            f"{elapsed_view:6.2f}s",
            flush=True,
        )

        if pending_number % args.save_every == 0 or pending_number == len(pending_slots):
            save_cache(cache_path, cache)
            result = current_summary()
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, sort_keys=True)
                handle.write("\n")

    save_cache(cache_path, cache)
    result = current_summary()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"cache: {cache_path}\nsummary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
