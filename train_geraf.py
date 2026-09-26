#!/usr/bin/env python
"""Train the object-bound GeRaF baseline.

The default ``source_v1`` recipe calls the released GeRaFStage1 loss with
explicit v1 paper settings. ``hardened_v1`` and ``legacy`` are compatibility
recipes, not source reproductions. See docs/GERAF_V1_HARDENING.md.
The companion target preparer may materialize only the frozen 3,200 training and 1,000
validation response rows; reserved-test and unused rows have no path through
this program.

The native supervision is GeRaF's full 3-D coherent matched-filter magnitude
``|MF|``.  Despite the paper's occasional use of the word "power," this is
not squared magnitude and is not a direct complex-signal objective.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rift.config import cc  # noqa: E402
from rift.forward_operator import get_kvector  # noqa: E402
from rift.geraf import (  # noqa: E402
    DynamicLossMask,
    GeRaFModel,
    PrimaryRaySamples,
    masked_magnitude_l2,
)
from rift.geraf_b7873200_protocol import (  # noqa: E402
    B787_3200_ACQUISITION_SCHEMA,
    B787_3200_CACHE_ACQUISITION_FILENAME,
    B787_3200_CACHE_MANIFEST_FILENAME,
    B787_3200_CACHE_RECIPE_FILENAME,
    B787_3200_CACHE_SCHEMA,
    B787_3200_CACHE_STATS_FILENAME,
    B787_3200_CACHE_VIEW_DIRECTORY,
    B787_3200_CANONICAL_MANIFEST_PATH,
    B787_3200_CANONICAL_NPZ_PATH,
    B787_3200_NUM_TRAIN,
    B787_3200_NUM_VALIDATION,
    B787_3200_SEED,
    B787_3200_TARGET_SCHEMA,
    validate_b7873200_target_cache,
)
from rift.geraf_b7873200_acquisition import (  # noqa: E402
    acquisition_records_equal,
    validate_b7873200_acquisition_record,
    validate_b7873200_operator_frequency_grid,
)
from rift.geraf_b7873200_source import (  # noqa: E402
    B7873200MetadataArrays,
    B7873200DevelopmentSource,
    load_b7873200_metadata_source,
)
from rift.geraf_signal_operator import (  # noqa: E402
    bistatic_pair_positions,
    matched_filter_from_response_range,
    pairwise_range_forward_operator,
)
from rift.power_baseline_dataset import (  # noqa: E402
    atomic_write_json,
    build_lensless_grid,
    frequency_grid_hz,
)


PAPER_ID = "arXiv:2605.29097v2"
METHOD_NAME = "GeRaF v1 (sealed B7873200)"
CHECKPOINT_VERSION = 2
RUN_IDENTITY_SCHEMA = "rift_geraf_b7873200_run_v1"
EPS = 1.0e-30
CLEAN_STOP_EXIT_CODE = 143
_STOP_REQUESTED = False
_STOP_SIGNAL: Optional[int] = None
_TIMED_WRAPPER_READY_ENV = "GERAF_TIMED_WRAPPER_READY_FILE"
_TIMED_WRAPPER_READY_MARKER = "GERAF_TIMED_WRAPPER_CHILD_READY"


def _request_stop(signum: int, _frame: object) -> None:
    """Ask the main loop to save a resumable checkpoint after one view."""

    global _STOP_REQUESTED, _STOP_SIGNAL
    _STOP_REQUESTED = True
    _STOP_SIGNAL = int(signum)
    print(
        f"Received signal {signum}; will atomically save checkpoint_latest.pth.tar "
        "after the current view.",
        flush=True,
    )


def _publish_timed_wrapper_ready() -> None:
    """Publish readiness for the allocation's optional TERM-forwarding wrapper."""

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
    print(f"GERAF_B7873200_TIMED_WRAPPER_READY path={ready_path}", flush=True)


GOTCHA_BACKEND = {
    "schema": "rift_gotcha_backend_v1", "method": "geraf", "callable": "run_gotcha",
    "selection_unit": "pass_sector", "joint_passes": True,
    "native_frequency_policy": "ragged_exact", "polarizations": ["hh", "hv", "vh", "vv"],
    "metric_domain": "native_coherent_mf_magnitude",
    "implementation": "source_v1",
}


def run_gotcha(*, dataset, output_dir, config, device, resume):
    from rift.geraf_gotcha import run_gotcha as backend
    return backend(dataset=dataset, output_dir=output_dir, config=config, device=device, resume=resume)


def parse_args(argv=None) -> argparse.Namespace:
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--implementation", choices=("source_v1", "hardened_v1", "legacy"), default="source_v1")
    selected, _ = selector.parse_known_args(argv)
    if selected.implementation == "source_v1":
        from rift.geraf_source_cli import parse_args as source_args
        return source_args(argv)
    return parse_compatibility_args(argv)


def parse_compatibility_args(argv=None) -> argparse.Namespace:
    from rift.rift_dataset import DEFAULT_ROOT, resolve_object_inputs
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_argument_group("sealed B7873200 data contract")
    data.add_argument("--object", help="Registered RIFT collection object or alias")
    data.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    data.add_argument(
        "--npz-path",
        default=None,
        help="Registered collection archive, or the canonical historical B787 archive",
    )
    data.add_argument(
        "--role-manifest",
        default=None,
        help="frozen 3200/1000/1000 interpolation split manifest",
    )
    data.add_argument("--cache-root", required=True, help="complete versioned prepared cache")
    data.add_argument("--checkpoint-dir", required=True)
    data.add_argument("--seed", type=int, default=B787_3200_SEED)
    data.add_argument("--scene-extent", type=float, default=0.15)
    data.add_argument("--n-azimuth", type=int, default=32)
    data.add_argument("--n-elevation", type=int, default=32)
    data.add_argument("--n-depth", type=int, default=32)
    data.add_argument("--aperture-scale", type=float, default=1.0)
    data.add_argument("--phase-sign", type=float, choices=(-1.0, 1.0), default=-1.0)

    model = parser.add_argument_group("GeRaF v1 model")
    model.add_argument("--implementation", choices=("hardened_v1", "legacy"), default="hardened_v1",
                       help="Versioned GeRaF v1 hardening; old checkpoints require --implementation legacy")
    model.add_argument("--sdf-levels", type=int, default=10)
    model.add_argument("--sdf-hidden-dim", type=int, default=256)
    model.add_argument("--sdf-layers", type=int, default=8)
    model.add_argument("--sdf-skip-layer", type=int, default=4)
    model.add_argument("--sdf-softplus-beta", type=float, default=100.0)
    model.add_argument("--reflectivity-levels", type=int, default=0)
    model.add_argument("--reflectivity-hidden-dim", type=int, default=256)
    model.add_argument("--reflectivity-layers", type=int, default=4)
    model.add_argument(
        "--reflectivity-output-activation",
        choices=("softplus", "sigmoid", "none"),
        default="softplus",
    )
    model.add_argument("--reflectivity-softplus-beta", type=float, default=1.0)
    model.add_argument("--init-tx-amplitude", type=float, default=1.0)
    model.add_argument("--init-inv-s", type=float, default=64.0)
    model.add_argument(
        "--learnable-inv-s", action=argparse.BooleanOptionalAction, default=True
    )
    model.add_argument(
        "--lensless-correction", action=argparse.BooleanOptionalAction, default=True
    )
    model.add_argument(
        "--detach-start-cdf", action=argparse.BooleanOptionalAction, default=True
    )
    model.add_argument("--directional-exponent", type=float, default=1.0)
    model.add_argument("--min-distance", type=float, default=1.0e-6)

    optimization = parser.add_argument_group("paper optimization protocol")
    optimization.add_argument("--steps", type=int, default=50_000)
    optimization.add_argument("--sdf-lr", type=float, default=1.0e-4)
    optimization.add_argument("--other-lr", type=float, default=1.0e-3)
    optimization.add_argument("--weight-decay", type=float, default=1.0e-2)
    optimization.add_argument("--adam-beta1", type=float, default=0.9)
    optimization.add_argument("--adam-beta2", type=float, default=0.999)
    optimization.add_argument("--adam-eps", type=float, default=1.0e-8)
    optimization.add_argument("--cosine-min-lr", type=float, default=0.0)
    optimization.add_argument("--gradient-clip-norm", type=float, default=0.0)

    mask = parser.add_argument_group("dynamic-mask adaptation thresholds")
    mask.add_argument("--mask-high-threshold", type=float, default=0.1)
    mask.add_argument("--mask-low-ratio", type=float, default=0.1)
    mask.add_argument("--mask-low-threshold", type=float, default=0.0)
    mask.add_argument("--measured-mask-current-fraction", type=float, default=0.05)
    mask.add_argument("--measured-mask-accumulated-fraction", type=float, default=0.1)
    mask.add_argument("--measured-mask-grid", type=int, default=48)

    operator = parser.add_argument_group("differentiable range-NUFFT")
    operator.add_argument("--compute-dtype", choices=("float32", "float64"), default="float64")
    operator.add_argument("--oversample", type=int, default=2)
    operator.add_argument("--kernel-width", type=int, default=20)
    operator.add_argument("--pair-chunk", type=int, default=32)
    operator.add_argument("--point-chunk", type=int, default=4096)

    runtime = parser.add_argument_group("runtime and checkpointing")
    runtime.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    runtime.add_argument("--validation-every", type=int, default=1000)
    runtime.add_argument("--checkpoint-every", type=int, default=100)
    runtime.add_argument("--checkpoint-seconds", type=float, default=600.0)
    runtime.add_argument("--log-every", type=int, default=10)
    runtime.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    runtime.add_argument("--resume-path", default=None)
    args = parser.parse_args(argv)
    if args.object is not None:
        args.npz_path, args.role_manifest = map(str, resolve_object_inputs(
            object_name=args.object, dataset_root=args.dataset_root,
            npz_path=args.npz_path, role_manifest_path=args.role_manifest))
    else:
        args.npz_path = args.npz_path or B787_3200_CANONICAL_NPZ_PATH
        args.role_manifest = args.role_manifest or B787_3200_CANONICAL_MANIFEST_PATH
    return args


def _read_json(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"B7873200 {label} is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"B7873200 {label} must contain a JSON object")
    return payload


def _require_equal(name: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(f"B7873200 contract mismatch for {name}: {observed!r} != {expected!r}")


def _require_close(name: str, observed: object, expected: object, atol: float = 1.0e-12) -> None:
    if not math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=atol):
        raise ValueError(f"B7873200 contract mismatch for {name}: {observed!r} != {expected!r}")


def _target_path(root: Path, view_index: int) -> Path:
    return root / B787_3200_CACHE_VIEW_DIRECTORY / f"view_{int(view_index):06d}.npz"


def _target_spec_text(target_spec: Mapping[str, Any]) -> str:
    return json.dumps(dict(target_spec), sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class PreparedB7873200Cache:
    source: B7873200DevelopmentSource
    root: Path
    recipe: Dict[str, Any]
    stats: Dict[str, Any]
    target_manifest: Dict[str, Any]
    acquisition_record: Dict[str, object]
    sealed_identity: Dict[str, object]
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    grid_shape: tuple[int, int, int]
    geraf_mf_magnitude_peak: float
    effective_pairs_per_plane: int


@dataclass(frozen=True)
class TrainingView:
    view_index: int
    target_normalized: torch.Tensor
    samples: PrimaryRaySamples
    tx_positions: torch.Tensor
    rx_positions: torch.Tensor


@dataclass(frozen=True)
class GeRaFSignalPrediction:
    normalized_magnitude: torch.Tensor
    complex_response: torch.Tensor
    diagnostics: Optional[Dict[str, Any]] = None


def _validate_cli(args: argparse.Namespace) -> None:
    if getattr(args, "implementation", "legacy") == "hardened_v1":
        if args.compute_dtype != "float64":
            raise ValueError("hardened GeRaF requires float64 propagation")
        for name in ("measured_mask_current_fraction", "measured_mask_accumulated_fraction"):
            value = getattr(args, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite in [0,1]")
        if args.measured_mask_grid < 2:
            raise ValueError("measured mask grid must be at least 2")
    if args.seed != B787_3200_SEED:
        raise ValueError(f"B7873200 comparison is frozen at seed {B787_3200_SEED}")
    floating_options = (
        "phase_sign",
        "scene_extent",
        "aperture_scale",
        "sdf_softplus_beta",
        "reflectivity_softplus_beta",
        "init_tx_amplitude",
        "init_inv_s",
        "directional_exponent",
        "min_distance",
        "sdf_lr",
        "other_lr",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "adam_eps",
        "cosine_min_lr",
        "gradient_clip_norm",
        "mask_high_threshold",
        "mask_low_ratio",
        "mask_low_threshold",
        "checkpoint_seconds",
    )
    nonfinite = []
    for name in floating_options:
        try:
            finite = math.isfinite(float(getattr(args, name)))
        except (AttributeError, TypeError, ValueError, OverflowError):
            finite = False
        if not finite:
            nonfinite.append(name)
    if nonfinite:
        raise ValueError(f"floating-point arguments must be finite: {nonfinite}")
    if float(args.phase_sign) != -1.0:
        raise ValueError("B7873200 GeRaF requires the established phase sign -1")
    positive = {
        "steps": args.steps,
        "scene_extent": args.scene_extent,
        "n_azimuth": args.n_azimuth,
        "n_elevation": args.n_elevation,
        "n_depth": args.n_depth,
        "aperture_scale": args.aperture_scale,
        "sdf_lr": args.sdf_lr,
        "other_lr": args.other_lr,
        "adam_eps": args.adam_eps,
        "validation_every": args.validation_every,
        "checkpoint_every": args.checkpoint_every,
        "checkpoint_seconds": args.checkpoint_seconds,
        "log_every": args.log_every,
        "oversample": args.oversample,
        "kernel_width": args.kernel_width,
        "pair_chunk": args.pair_chunk,
        "point_chunk": args.point_chunk,
        "sdf_softplus_beta": args.sdf_softplus_beta,
        "reflectivity_softplus_beta": args.reflectivity_softplus_beta,
        "directional_exponent": args.directional_exponent,
        "min_distance": args.min_distance,
    }
    invalid = [name for name, value in positive.items() if float(value) <= 0.0]
    if invalid:
        raise ValueError(f"arguments must be positive: {invalid}")
    if args.oversample < 2 or args.kernel_width < 4:
        raise ValueError("range-NUFFT requires oversample>=2 and kernel_width>=4")
    if args.sdf_levels != 10:
        raise ValueError("reportable GeRaF requires exactly 10 positional-encoding levels")
    if args.weight_decay < 0 or args.cosine_min_lr < 0 or args.gradient_clip_norm < 0:
        raise ValueError("weight decay, cosine minimum, and gradient clipping must be non-negative")
    if args.cosine_min_lr > min(args.sdf_lr, args.other_lr):
        raise ValueError("cosine-min-lr cannot exceed either initial learning rate")
    if not (0.0 <= args.adam_beta1 < 1.0 and 0.0 <= args.adam_beta2 < 1.0):
        raise ValueError("Adam betas must lie in [0,1)")
    if args.mask_high_threshold < 0 or args.mask_low_threshold < 0:
        raise ValueError("dynamic-mask thresholds must be non-negative")
    if not 0.0 <= args.mask_low_ratio <= 1.0:
        raise ValueError("mask-low-ratio must lie in [0,1]")


def _target_spec_from_recipe(
    recipe: Mapping[str, Any],
    *,
    accepted_cache_schemas: Iterable[str] = (B787_3200_CACHE_SCHEMA,),
) -> Mapping[str, Any]:
    """Return a target recipe for the production cache or an explicit adapter.

    Production callers retain the one-element default.  A bounded engineering
    adapter can opt in to its *own* cache schema without weakening the full
    3,200/1,000 cache verifier or changing the on-disk target payload schema.
    """

    schemas = frozenset(str(value) for value in accepted_cache_schemas)
    if not schemas or recipe.get("schema") not in schemas or recipe.get("version") != 1:
        raise ValueError("B7873200 recipe schema/version mismatch")
    target_spec = recipe.get("target_spec")
    if not isinstance(target_spec, Mapping):
        raise ValueError("B7873200 recipe lacks a target_spec object")
    return target_spec


def _require_metadata_only_arrays(arrays: object) -> B7873200MetadataArrays:
    """Fail closed if training is ever handed a response-capable source."""

    if not isinstance(arrays, B7873200MetadataArrays):
        raise TypeError(
            "B7873200 trainer requires B7873200MetadataArrays; "
            "response-capable development sources are preparation-only"
        )
    if arrays.response_payload_materialized is not False:
        raise RuntimeError("B7873200 metadata-only trainer source unexpectedly exposes response payloads")
    return arrays


def verify_prepared_b7873200_cache(args: argparse.Namespace) -> PreparedB7873200Cache:
    """Bind the complete direct-equality cache before constructing a model.

    This creates a metadata-only B787 source after manifest/header preflight.
    Training consumes only cached ``|MF|`` targets and calibrated geometry; it
    has no raw response accessor to call.
    """

    source = load_b7873200_metadata_source(args.npz_path, args.role_manifest)
    arrays = _require_metadata_only_arrays(source.arrays)
    identity = dict(source.identity)
    root = Path(args.cache_root).expanduser().resolve()
    validate_b7873200_target_cache(root, identity)
    recipe = _read_json(root / B787_3200_CACHE_RECIPE_FILENAME, "cache recipe")
    target_manifest = _read_json(root / B787_3200_CACHE_MANIFEST_FILENAME, "target manifest")
    stats = _read_json(root / B787_3200_CACHE_STATS_FILENAME, "target stats")
    acquisition_record = validate_b7873200_acquisition_record(root, source.arrays)
    target_spec = _target_spec_from_recipe(recipe)
    grid = target_spec.get("grid")
    if not isinstance(grid, Mapping):
        raise ValueError("B7873200 target_spec.grid must be an object")
    for option, recipe_key in (
        ("scene_extent", "scene_extent_m"),
        ("n_azimuth", "n_azimuth"),
        ("n_elevation", "n_elevation"),
        ("n_depth", "n_depth"),
        ("aperture_scale", "aperture_scale"),
    ):
        observed = getattr(args, option)
        expected = grid.get(recipe_key)
        if option.startswith("n_"):
            _require_equal(f"target grid {recipe_key}", int(observed), int(expected))
        else:
            _require_close(f"target grid {recipe_key}", observed, expected)
    _require_equal("target native readout", target_spec.get("native_readout"), "complex magnitude |MF|")
    _require_close("target phase sign", target_spec.get("phase_sign"), -1.0)
    _require_equal("target backend", target_spec.get("backend"), "range")
    _require_equal("target compute dtype", target_spec.get("compute_dtype"), args.compute_dtype)
    operator_spec = target_spec.get("operator")
    if not isinstance(operator_spec, Mapping):
        raise ValueError("B7873200 target_spec.operator must be an object")
    _require_equal("target operator implementation", operator_spec.get("implementation"), "range_nufft")
    _require_equal("target range model", operator_spec.get("range_model"), "none")
    _require_equal("target include-four-pi", operator_spec.get("include_four_pi"), False)
    _require_equal("target direct frequency chunk", operator_spec.get("freq_chunk"), None)
    _require_equal("target kernel width", operator_spec.get("kernel_width"), int(args.kernel_width))
    _require_equal("target oversample", operator_spec.get("oversample"), int(args.oversample))
    _require_equal("target point chunk", operator_spec.get("point_chunk"), int(args.point_chunk))
    _require_equal("target pair chunk", operator_spec.get("pair_chunk"), int(args.pair_chunk))

    if arrays.num_views != 10_000 or arrays.num_tx != 16 or arrays.num_rx != 16:
        raise ValueError(
            "B7873200 requires the calibrated 10,000-view, 16 Tx x 16 Rx sphere10k archive"
        )
    roles = identity.get("role_ids")
    if not isinstance(roles, Mapping):
        raise AssertionError("B7873200 source identity unexpectedly lacks roles")
    train = tuple(int(value) for value in roles["train"])
    validation = tuple(int(value) for value in roles["validation"])
    if len(train) != B787_3200_NUM_TRAIN or len(validation) != B787_3200_NUM_VALIDATION:
        raise ValueError("B7873200 cache has an invalid training or validation role count")
    expected_roles = {"train": list(train), "validation": list(validation)}
    _require_equal("target manifest schema", target_manifest.get("schema"), B787_3200_CACHE_SCHEMA)
    _require_equal("target manifest version", target_manifest.get("version"), 1)
    _require_equal("target manifest roles", target_manifest.get("roles"), expected_roles)
    _require_equal("target stats schema", stats.get("schema"), B787_3200_CACHE_SCHEMA)
    _require_equal("target stats version", stats.get("version"), 1)
    _require_equal("target stats fit split", stats.get("fit_split"), "train")
    _require_equal("target stats clip", stats.get("clip"), False)
    peak = float(stats.get("geraf_mf_magnitude_peak", 0.0))
    if not math.isfinite(peak) or peak <= 0.0:
        raise ValueError("B7873200 target cache has an invalid train-only GeRaF normalizer")
    _validate_complete_b7873200_target_contents(
        root=root,
        recipe=recipe,
        stats=stats,
        arrays=arrays,
        train=train,
        validation=validation,
        args=args,
    )
    return PreparedB7873200Cache(
        source=source,
        root=root,
        recipe=recipe,
        stats=stats,
        target_manifest=target_manifest,
        acquisition_record=acquisition_record,
        sealed_identity=identity,
        train_indices=train,
        validation_indices=validation,
        grid_shape=(int(args.n_elevation), int(args.n_azimuth), int(args.n_depth)),
        geraf_mf_magnitude_peak=peak,
        effective_pairs_per_plane=int(arrays.num_tx * arrays.num_rx),
    )


def _allclose_cached(name: str, observed: np.ndarray, expected: torch.Tensor) -> None:
    expected_np = expected.detach().cpu().numpy()
    if observed.shape != expected_np.shape or not np.allclose(
        observed, expected_np, rtol=1.0e-6, atol=1.0e-6
    ):
        raise ValueError(f"B7873200 cached target metadata drift for {name}")


def _depth_deltas(depth: torch.Tensor, num_rays: int) -> torch.Tensor:
    if depth.numel() == 1:
        delta = depth.new_full((1,), 1.0)
    else:
        spacing = depth[1:] - depth[:-1]
        if bool((spacing <= 0).any().item()):
            raise ValueError("B7873200 lensless depths must be strictly increasing")
        delta = torch.cat((spacing, spacing[-1:]), dim=0)
    return delta.view(1, -1).expand(num_rays, -1)


def _samples_from_cached_geometry(
    target: Mapping[str, np.ndarray], device: torch.device
) -> PrimaryRaySamples:
    names = (
        "viewpoint_position",
        "primary_direction",
        "azimuth_axis",
        "elevation_axis",
        "depth_m",
        "azimuth_offsets_m",
        "elevation_offsets_m",
    )
    for name in names:
        values = np.asarray(target[name])
        if values.dtype != np.dtype(np.float32):
            raise ValueError(f"B7873200 cached geometry {name} must be float32")
    viewpoint = torch.as_tensor(target["viewpoint_position"], device=device)
    primary = torch.as_tensor(target["primary_direction"], device=device)
    azimuth_axis = torch.as_tensor(target["azimuth_axis"], device=device)
    elevation_axis = torch.as_tensor(target["elevation_axis"], device=device)
    depth = torch.as_tensor(target["depth_m"], device=device)
    azimuth_offsets = torch.as_tensor(target["azimuth_offsets_m"], device=device)
    elevation_offsets = torch.as_tensor(target["elevation_offsets_m"], device=device)
    if viewpoint.shape != (3,) or primary.shape != (3,):
        raise ValueError("B7873200 cached viewpoint/primary direction must each have shape [3]")
    if azimuth_axis.shape != (3,) or elevation_axis.shape != (3,):
        raise ValueError("B7873200 cached aperture axes must each have shape [3]")
    if depth.ndim != 1 or azimuth_offsets.ndim != 1 or elevation_offsets.ndim != 1:
        raise ValueError("B7873200 cached depth and aperture offsets must be one-dimensional")
    elevation_grid, azimuth_grid = torch.meshgrid(elevation_offsets, azimuth_offsets, indexing="ij")
    origins_grid = (
        viewpoint
        + azimuth_grid[..., None] * azimuth_axis
        + elevation_grid[..., None] * elevation_axis
    )
    points_grid = origins_grid[..., None, :] + depth[None, None, :, None] * primary
    num_rays = int(azimuth_offsets.numel() * elevation_offsets.numel())
    return PrimaryRaySamples(
        ray_origins=origins_grid.reshape(num_rays, 3),
        primary_direction=primary,
        depths=depth.view(1, -1).expand(num_rays, -1),
        depth_deltas=_depth_deltas(depth, num_rays),
        points=points_grid.reshape(num_rays, depth.numel(), 3),
    )


def _load_target(index: int, role: str, cache: PreparedB7873200Cache) -> Dict[str, np.ndarray]:
    path = _target_path(cache.root, index)
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "schema",
            "target_spec_json",
            "view_index",
            "role",
            "geraf_mf_magnitude",
            "viewpoint_position",
            "primary_direction",
            "azimuth_axis",
            "elevation_axis",
            "depth_m",
            "azimuth_offsets_m",
            "elevation_offsets_m",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"B7873200 target {path} is missing {missing}")
        target = {name: np.asarray(archive[name]) for name in required}
    schema = str(np.asarray(target["schema"]).reshape(()))
    target_spec_text = str(np.asarray(target["target_spec_json"]).reshape(()))
    observed_index = int(np.asarray(target["view_index"]).reshape(()))
    observed_role = str(np.asarray(target["role"]).reshape(()))
    if schema != B787_3200_TARGET_SCHEMA:
        raise ValueError(f"B7873200 target {path} has an unexpected schema")
    accepted_cache_schemas = getattr(
        cache, "target_recipe_schemas", (B787_3200_CACHE_SCHEMA,)
    )
    expected_target_spec = _target_spec_from_recipe(
        cache.recipe, accepted_cache_schemas=accepted_cache_schemas
    )
    if target_spec_text != _target_spec_text(expected_target_spec):
        raise ValueError(f"B7873200 target {path} has a different native target recipe")
    if observed_index != int(index):
        raise ValueError(f"B7873200 target filename/index mismatch for view {index}")
    if observed_role != role:
        raise ValueError(f"B7873200 target {index} is labelled {observed_role!r}, expected {role!r}")
    return target


def _compact_unit(vector: torch.Tensor, name: str) -> torch.Tensor:
    """Match the historical lensless-frame normalization without allocating voxels."""

    norm = torch.linalg.vector_norm(vector)
    if not torch.isfinite(norm) or float(norm) < 1.0e-12:
        raise ValueError(f"cannot construct B7873200 frozen geometry: degenerate {name}")
    return vector / norm


def _compact_fallback_tangent(primary: torch.Tensor) -> torch.Tensor:
    basis = torch.eye(3, dtype=primary.dtype, device=primary.device)
    candidate = basis[torch.argmin(primary.abs())]
    return _compact_unit(candidate - torch.dot(candidate, primary) * primary, "fallback tangent")


def _expected_cached_geometry(
    viewpoint_position: np.ndarray,
    tx_positions: np.ndarray,
    rx_positions: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    """Recompute a target's compact frozen frame without materializing its 3-D grid.

    Target preparation used :func:`build_lensless_grid`, which also constructs
    ``n_elevation*n_azimuth*n_depth`` points.  Cache preflight needs to check
    all 4,200 targets before model construction, but only the compact axes and
    one-dimensional coordinates are persisted.  This mirrors that frame math
    in float32 and deliberately avoids both raw response access and a voxel
    allocation for every target.
    """

    dtype = torch.float32
    vp = torch.as_tensor(np.asarray(viewpoint_position, dtype=np.float64), dtype=dtype).reshape(3)
    tx = torch.as_tensor(np.asarray(tx_positions), dtype=dtype).reshape(-1, 3)
    rx = torch.as_tensor(np.asarray(rx_positions), dtype=dtype).reshape(-1, 3)
    center = torch.zeros(3, dtype=dtype)
    primary = _compact_unit(center - vp, "primary direction")

    tx_span = tx[-1] - tx[0] if len(tx) > 1 else torch.zeros_like(primary)
    tx_tangent = tx_span - torch.dot(tx_span, primary) * primary
    azimuth = (
        _compact_fallback_tangent(primary)
        if float(torch.linalg.vector_norm(tx_tangent)) < 1.0e-12
        else _compact_unit(tx_tangent, "Tx tangent")
    )

    rx_span = rx[-1] - rx[0] if len(rx) > 1 else torch.zeros_like(primary)
    rx_tangent = rx_span - torch.dot(rx_span, primary) * primary
    rx_tangent = rx_tangent - torch.dot(rx_tangent, azimuth) * azimuth
    if float(torch.linalg.vector_norm(rx_tangent)) < 1.0e-12:
        elevation = _compact_unit(torch.linalg.cross(primary, azimuth), "elevation tangent")
    else:
        elevation = _compact_unit(rx_tangent, "Rx tangent")
    if torch.dot(torch.linalg.cross(azimuth, elevation), primary) < 0:
        elevation = -elevation

    virtual = 0.5 * (tx[:, None, :] + rx[None, :, :])
    virtual_offset = virtual.reshape(-1, 3) - vp
    az_projection = virtual_offset @ azimuth
    el_projection = virtual_offset @ elevation
    az_lo = float(args.aperture_scale) * az_projection.min()
    az_hi = float(args.aperture_scale) * az_projection.max()
    el_lo = float(args.aperture_scale) * el_projection.min()
    el_hi = float(args.aperture_scale) * el_projection.max()
    if float((az_hi - az_lo).abs()) < 1.0e-9:
        az_lo = az_projection.new_tensor(-float(args.scene_extent))
        az_hi = az_projection.new_tensor(float(args.scene_extent))
    if float((el_hi - el_lo).abs()) < 1.0e-9:
        el_lo = el_projection.new_tensor(-float(args.scene_extent))
        el_hi = el_projection.new_tensor(float(args.scene_extent))
    center_range = torch.linalg.vector_norm(center - vp)

    return {
        "primary_direction": primary.numpy(),
        "azimuth_axis": azimuth.numpy(),
        "elevation_axis": elevation.numpy(),
        "depth_m": torch.linspace(
            center_range - float(args.scene_extent),
            center_range + float(args.scene_extent),
            int(args.n_depth),
            dtype=dtype,
        ).numpy(),
        "azimuth_offsets_m": torch.linspace(
            az_lo, az_hi, int(args.n_azimuth), dtype=dtype
        ).numpy(),
        "elevation_offsets_m": torch.linspace(
            el_lo, el_hi, int(args.n_elevation), dtype=dtype
        ).numpy(),
    }


def _require_frozen_geometry(
    *,
    target: Mapping[str, np.ndarray],
    view_index: int,
    arrays: B7873200MetadataArrays,
    args: argparse.Namespace,
) -> None:
    """Validate one cached target's shape, finite values, and calibrated frame."""

    expected_shape = (int(args.n_elevation), int(args.n_azimuth), int(args.n_depth))
    magnitude = np.asarray(target["geraf_mf_magnitude"])
    if magnitude.dtype != np.dtype(np.float32) or magnitude.shape != expected_shape:
        raise ValueError(
            f"B7873200 cached target {view_index} has invalid |MF| dtype/shape; "
            f"expected float32 {expected_shape}"
        )
    if not np.isfinite(magnitude).all() or bool(np.any(magnitude < 0.0)):
        raise ValueError(f"B7873200 cached target {view_index} has non-finite or negative |MF|")

    geometry_shapes = {
        "viewpoint_position": (3,),
        "primary_direction": (3,),
        "azimuth_axis": (3,),
        "elevation_axis": (3,),
        "depth_m": (expected_shape[2],),
        "azimuth_offsets_m": (expected_shape[1],),
        "elevation_offsets_m": (expected_shape[0],),
    }
    for field, expected in geometry_shapes.items():
        values = np.asarray(target[field])
        if values.dtype != np.dtype(np.float32) or values.shape != expected:
            raise ValueError(
                f"B7873200 cached target {view_index} has invalid frozen {field} dtype/shape"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"B7873200 cached target {view_index} has non-finite frozen {field}")

    expected_viewpoint = np.asarray(arrays.viewpoint_positions[view_index], dtype=np.float32)
    if not np.array_equal(np.asarray(target["viewpoint_position"]), expected_viewpoint):
        raise ValueError(
            f"B7873200 cached target {view_index} has stale calibrated viewpoint geometry"
        )
    expected_geometry = _expected_cached_geometry(
        arrays.viewpoint_positions[view_index],
        arrays.tx_pos[view_index],
        arrays.rx_pos[view_index],
        args,
    )
    for field, expected in expected_geometry.items():
        observed = np.asarray(target[field])
        if not np.allclose(observed, expected, rtol=1.0e-6, atol=1.0e-6):
            raise ValueError(
                f"B7873200 cached target {view_index} frozen {field} differs from calibrated geometry"
            )


def _validate_complete_b7873200_target_contents(
    *,
    root: Path,
    recipe: Mapping[str, Any],
    stats: Mapping[str, Any],
    arrays: B7873200MetadataArrays,
    train: Sequence[int],
    validation: Sequence[int],
    args: argparse.Namespace,
) -> None:
    """Preflight every authorized cache entry without opening radar responses.

    The protocol sidecar proves only the role policy.  This engineering gate
    additionally rejects malformed, stale, or non-finite target entries and
    recomputes the training-only normalizer from the cached native targets.
    It performs no source-response read and intentionally uses no digest or
    integrity-pin mechanism.
    """

    cache = SimpleNamespace(root=root, recipe=dict(recipe))
    train_peak = 0.0
    for role, indices in (("train", train), ("validation", validation)):
        for index in indices:
            target = _load_target(int(index), role, cache)
            _require_frozen_geometry(
                target=target,
                view_index=int(index),
                arrays=arrays,
                args=args,
            )
            if role == "train":
                train_peak = max(train_peak, float(np.max(np.asarray(target["geraf_mf_magnitude"]))))
    if not math.isfinite(train_peak) or train_peak <= 0.0:
        raise ValueError("B7873200 complete target cache lacks a finite positive training |MF| peak")
    observed_peak = stats.get("geraf_mf_magnitude_peak")
    if isinstance(observed_peak, bool):
        raise ValueError("B7873200 target stats have an invalid training |MF| peak")
    try:
        expected_peak = float(observed_peak)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("B7873200 target stats have an invalid training |MF| peak") from exc
    if not math.isfinite(expected_peak) or expected_peak <= 0.0 or expected_peak != train_peak:
        raise ValueError(
            "B7873200 target stats training |MF| peak disagrees with the complete cached targets"
        )


def load_training_view(
    cache: PreparedB7873200Cache,
    view_index: int,
    role: str,
    args: argparse.Namespace,
    device: torch.device,
) -> TrainingView:
    index = int(view_index)
    if role == "train":
        allowed = cache.train_indices
    elif role == "validation":
        allowed = cache.validation_indices
    else:
        raise ValueError(f"B7873200 does not expose target role {role!r}")
    if index not in allowed:
        raise ValueError(f"B7873200 view {index} is not authorized for target role {role!r}")
    target = _load_target(index, role, cache)
    arrays = cache.source.arrays
    tx = torch.as_tensor(arrays.tx_pos[index], dtype=torch.float32, device=device)
    rx = torch.as_tensor(arrays.rx_pos[index], dtype=torch.float32, device=device)
    viewpoint = np.asarray(arrays.viewpoint_positions[index], dtype=np.float64)
    cached_viewpoint = np.asarray(target["viewpoint_position"])
    expected_cached_viewpoint = np.asarray(viewpoint, dtype=np.float32)
    if cached_viewpoint.dtype != np.dtype(np.float32) or not np.array_equal(
        cached_viewpoint, expected_cached_viewpoint
    ):
        raise ValueError(f"B7873200 target {index} has stale calibrated viewpoint geometry")
    grid = build_lensless_grid(
        viewpoint,
        tx,
        rx,
        scene_center=(0.0, 0.0, 0.0),
        scene_extent_m=args.scene_extent,
        n_azimuth=args.n_azimuth,
        n_elevation=args.n_elevation,
        n_depth=args.n_depth,
        aperture_scale=args.aperture_scale,
        device=device,
        dtype=torch.float32,
    )
    _allclose_cached("primary_direction", target["primary_direction"], grid.primary_direction)
    _allclose_cached("azimuth_axis", target["azimuth_axis"], grid.azimuth_axis)
    _allclose_cached("elevation_axis", target["elevation_axis"], grid.elevation_axis)
    _allclose_cached("depth_m", target["depth_m"], grid.depth_m)
    _allclose_cached("azimuth_offsets_m", target["azimuth_offsets_m"], grid.azimuth_offsets_m)
    _allclose_cached("elevation_offsets_m", target["elevation_offsets_m"], grid.elevation_offsets_m)
    target_magnitude = torch.as_tensor(
        target["geraf_mf_magnitude"], dtype=torch.float32, device=device
    )
    if tuple(target_magnitude.shape) != cache.grid_shape:
        raise ValueError(
            f"B7873200 view {index} matched-filter target shape {tuple(target_magnitude.shape)} "
            f"!= {cache.grid_shape}"
        )
    if not bool(torch.isfinite(target_magnitude).all().item()) or bool(
        (target_magnitude < 0).any().item()
    ):
        raise ValueError(f"B7873200 view {index} target MF magnitude is invalid")
    return TrainingView(
        index,
        target_magnitude / cache.geraf_mf_magnitude_peak,
        _samples_from_cached_geometry(target, device),
        tx,
        rx,
    )


class DeterministicViewSampler:
    """Seeded permutation cycles with serializable order and state."""

    def __init__(self, indices: Sequence[int], seed: int) -> None:
        if not indices:
            raise ValueError("B7873200 training indices cannot be empty")
        self.indices = np.asarray(indices, dtype=np.int64)
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.order = np.empty(0, dtype=np.int64)
        self.cursor = 0
        self.exposures = {int(index): 0 for index in self.indices}

    def record_update(self, index):
        if int(index) not in self.exposures:
            raise ValueError("optimizer exposure is outside the training role")
        self.exposures[int(index)] += 1

    def coverage(self):
        counts = list(self.exposures.values())
        return {"optimizer_updates": sum(counts), "unique_training_views": sum(n > 0 for n in counts),
                "registered_training_views": len(counts), "minimum_exposures": min(counts),
                "maximum_exposures": max(counts), "per_view_exposures": dict(self.exposures)}

    def next(self) -> int:
        if self.cursor >= self.order.size:
            self.order = self.rng.permutation(self.indices)
            self.cursor = 0
        result = int(self.order[self.cursor])
        self.cursor += 1
        return result

    def state_dict(self) -> Dict[str, Any]:
        return {
            "indices": self.indices.copy(),
            "order": self.order.copy(),
            "cursor": int(self.cursor),
            "bit_generator_state": self.rng.bit_generator.state,
            "exposures": dict(self.exposures),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not np.array_equal(np.asarray(state["indices"], dtype=np.int64), self.indices):
            raise ValueError("B7873200 checkpoint sampler belongs to a different training role")
        order = np.asarray(state["order"], dtype=np.int64)
        cursor = int(state["cursor"])
        if cursor < 0 or cursor > order.size:
            raise ValueError("B7873200 checkpoint has an invalid sampler cursor")
        if order.size and (order.size != self.indices.size or set(order.tolist()) != set(self.indices.tolist())):
            raise ValueError("B7873200 checkpoint has an invalid sampler permutation")
        self.order = order.copy()
        self.cursor = cursor
        self.rng.bit_generator.state = state["bit_generator_state"]
        if "exposures" in state:
            counts = state["exposures"]
            if set(counts) != set(self.exposures) or any(type(n) is not int or n < 0 for n in counts.values()):
                raise ValueError("invalid checkpoint training exposure counts")
            self.exposures = dict(counts)


class PerViewDynamicMaskBank:
    """Independent historical DynamicLossMask state for training views only."""

    def __init__(
        self,
        train_indices: Iterable[int],
        shape: Sequence[int],
        *,
        high_threshold: float,
        low_ratio: float,
        low_threshold: float,
        device: torch.device,
    ) -> None:
        self.allowed = frozenset(int(value) for value in train_indices)
        self.shape = tuple(int(value) for value in shape)
        self.high_threshold = float(high_threshold)
        self.low_ratio = float(low_ratio)
        self.low_threshold = float(low_threshold)
        self.histories: Dict[int, torch.Tensor] = {}
        self.mask = DynamicLossMask(
            self.high_threshold,
            low_ratio=self.low_ratio,
            low_threshold=self.low_threshold,
            shape=self.shape,
        ).to(device)

    def valid_mask(self, view_index: int, current_power: torch.Tensor) -> torch.Tensor:
        index = int(view_index)
        if index not in self.allowed:
            raise ValueError("B7873200 dynamic mask may only be used for a training view")
        history = self.histories.get(index)
        with torch.no_grad():
            self.mask.historical_max.zero_()
            if history is not None:
                self.mask.historical_max.copy_(history.to(self.mask.historical_max))
        valid = self.mask(current_power, update=False)
        with torch.no_grad():
            self.mask.historical_max.copy_(
                torch.maximum(
                    self.mask.historical_max,
                    current_power.detach().to(self.mask.historical_max),
                )
            )
        self.histories[index] = self.mask.historical_max.detach().cpu().clone()
        return valid

    def state_dict(self) -> Dict[str, Any]:
        return {
            "shape": self.shape,
            "high_threshold": self.high_threshold,
            "low_ratio": self.low_ratio,
            "low_threshold": self.low_threshold,
            "histories": {int(key): value.clone() for key, value in self.histories.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        _require_equal("mask-bank shape", tuple(state["shape"]), self.shape)
        _require_close("mask-bank high_threshold", state["high_threshold"], self.high_threshold)
        _require_close("mask-bank low_ratio", state["low_ratio"], self.low_ratio)
        _require_close("mask-bank low_threshold", state["low_threshold"], self.low_threshold)
        restored: Dict[int, torch.Tensor] = {}
        for key, value in state.get("histories", {}).items():
            index = int(key)
            tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu")
            if index not in self.allowed or tuple(tensor.shape) != self.shape:
                raise ValueError(f"invalid B7873200 dynamic-mask history for view {index}")
            if not bool(torch.isfinite(tensor).all().item()) or bool((tensor < 0).any().item()):
                raise ValueError(f"invalid B7873200 dynamic-mask values for view {index}")
            restored[index] = tensor.clone()
        self.histories = restored


def _compute_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def render_samples_for_view(view: TrainingView, args: argparse.Namespace) -> PrimaryRaySamples:
    """Separate full-scene integration cells from the frozen MF query lattice."""
    if getattr(args, "implementation", "legacy") != "hardened_v1":
        return view.samples
    from rift.geraf_v1 import sample_scene_rays
    origins = view.samples.ray_origins.reshape(args.n_elevation, args.n_azimuth, 3).double()
    azimuth = origins[0, -1] - origins[0, 0] if args.n_azimuth > 1 else None
    elevation = origins[-1, 0] - origins[0, 0] if args.n_elevation > 1 else None
    center = origins.reshape(-1, 3).mean(0)
    # Project the scene origin onto the aperture plane: small measured aperture
    # offsets must not shift the full-scene sampling footprint.
    primary = view.samples.primary_direction.double()
    primary = primary / primary.norm()
    center = primary * torch.dot(center, primary)
    return sample_scene_rays(center, primary, extent=args.scene_extent,
                             n_azimuth=args.n_azimuth, n_elevation=args.n_elevation,
                             n_depth=args.n_depth, azimuth_axis=azimuth, elevation_axis=elevation)


def _measured_mask_bank(cache, args):
    from rift.geraf_v1 import MeasuredViewMaskBank
    acquisition_hash = hashlib.sha256()
    for key, value in sorted(cache.acquisition_record.items()):
        acquisition_hash.update(key.encode())
        if isinstance(value, np.ndarray):
            acquisition_hash.update(str((value.shape, value.dtype.str)).encode())
            acquisition_hash.update(value.tobytes())
        else:
            acquisition_hash.update(json.dumps(value, sort_keys=True).encode())
    return MeasuredViewMaskBank(
        cache.train_indices, cache.grid_shape, extent=args.scene_extent,
        size=args.measured_mask_grid,
        identity={"sealed_identity": cache.sealed_identity, "cache_recipe": cache.recipe,
                  "acquisition_sha256": acquisition_hash.hexdigest(), "target_stats": cache.stats},
        current_fraction=args.measured_mask_current_fraction,
        accumulated_fraction=args.measured_mask_accumulated_fraction,
    )


def _prepare_measured_mask_reference(bank, cache):
    """Manager-time preparation from existing training targets; no raw reads."""
    for index in cache.train_indices:
        if _STOP_REQUESTED:
            raise InterruptedError("stopped during train-only measured mask preparation")
        target = _load_target(index, "train", cache)
        samples = _samples_from_cached_geometry(target, torch.device("cpu"))
        magnitude = torch.as_tensor(target["geraf_mf_magnitude"]) / cache.geraf_mf_magnitude_peak
        bank.reference.add(index, samples.points, magnitude)
    bank.reference.finalize()


def predict_normalized_magnitude_with_response(
    model: GeRaFModel,
    view: TrainingView,
    frequencies: torch.Tensor,
    kvector: torch.Tensor,
    cache: PreparedB7873200Cache,
    args: argparse.Namespace,
    *,
    create_graph: bool,
) -> GeRaFSignalPrediction:
    """Render a calibrated view through GeRaF and the native full 3-D MF."""

    tx_positions = view.tx_positions.detach()
    rx_positions = view.rx_positions.detach()
    integration_samples = render_samples_for_view(view, args)
    fixed_samples = PrimaryRaySamples(
        ray_origins=integration_samples.ray_origins.detach(),
        primary_direction=integration_samples.primary_direction.detach(),
        depths=integration_samples.depths.detach(),
        depth_deltas=integration_samples.depth_deltas.detach(),
        points=integration_samples.points.detach(),
        depth_edges=integration_samples.depth_edges,
    )
    fixed_frequencies = frequencies.detach()
    fixed_kvector = kvector.detach()
    tx_pair, rx_pair = bistatic_pair_positions(tx_positions, rx_positions)
    volume = model.render_volume(
        fixed_samples,
        tx_pair,
        rx_pair,
        lensless_correction=args.lensless_correction,
        detach_start_cdf=args.detach_start_cdf,
        directional_exponent=args.directional_exponent,
        create_graph=create_graph,
        min_distance=args.min_distance,
    )
    flat_points = volume.points.reshape(-1, 3).detach()
    pair_amplitudes = volume.amplitudes.reshape(volume.amplitudes.shape[0], -1)
    if not create_graph:
        flat_points = flat_points.detach()
        pair_amplitudes = pair_amplitudes.detach()
    signal_context = contextlib.nullcontext() if create_graph else torch.no_grad()
    with signal_context:
        response = pairwise_range_forward_operator(
            fixed_frequencies,
            fixed_kvector,
            tx_positions,
            rx_positions,
            flat_points,
            pair_amplitudes,
            phase_sign=args.phase_sign,
            oversample=args.oversample,
            kernel_width=args.kernel_width,
            pair_chunk=args.pair_chunk,
            point_chunk=args.point_chunk,
            compute_dtype=_compute_dtype(args.compute_dtype),
        )
        mf_amplitude = matched_filter_from_response_range(
            response,
            fixed_frequencies,
            fixed_kvector,
            tx_positions,
            rx_positions,
            view.samples.points.reshape(-1, 3).detach(),
            phase_sign=args.phase_sign,
            oversample=args.oversample,
            kernel_width=args.kernel_width,
            pair_chunk=args.pair_chunk,
            point_chunk=args.point_chunk,
            compute_dtype=_compute_dtype(args.compute_dtype),
        )
        magnitude = mf_amplitude.abs().reshape(cache.grid_shape)
    if not bool(torch.isfinite(magnitude).all().item()):
        raise FloatingPointError(f"B7873200 view {view.view_index} predicted non-finite MF magnitude")
    return GeRaFSignalPrediction(
        normalized_magnitude=magnitude / cache.geraf_mf_magnitude_peak,
        complex_response=response,
        diagnostics=volume.diagnostics,
    )


def predict_normalized_magnitude(
    model: GeRaFModel,
    view: TrainingView,
    frequencies: torch.Tensor,
    kvector: torch.Tensor,
    cache: PreparedB7873200Cache,
    args: argparse.Namespace,
    *,
    create_graph: bool,
) -> torch.Tensor:
    return predict_normalized_magnitude_with_response(
        model, view, frequencies, kvector, cache, args, create_graph=create_graph
    ).normalized_magnitude


def build_geraf_optimizer(
    model: GeRaFModel, args: argparse.Namespace
) -> torch.optim.Optimizer:
    """Build the unchanged paper optimizer grouping for an opt-in caller."""

    sdf_parameters = list(model.sdf_network.parameters())
    sdf_ids = {id(parameter) for parameter in sdf_parameters}
    other_parameters = [parameter for parameter in model.parameters() if id(parameter) not in sdf_ids]
    if not sdf_parameters or not other_parameters:
        raise RuntimeError("failed to form SDF and non-SDF optimizer parameter groups")
    return torch.optim.AdamW(
        [
            {"params": sdf_parameters, "lr": args.sdf_lr, "name": "sdf"},
            {"params": other_parameters, "lr": args.other_lr, "name": "other"},
        ],
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )


def build_geraf_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    *,
    horizon_steps: Optional[int] = None,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """Keep production's ``args.steps`` horizon unless an adapter names one."""

    horizon = int(args.steps) if horizon_steps is None else int(horizon_steps)
    if horizon <= 0:
        raise ValueError("GeRaF scheduler horizon must be positive")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=horizon, eta_min=args.cosine_min_lr
    )


def train_one_view_update(
    *,
    model: GeRaFModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    sampler: DeterministicViewSampler,
    mask_bank: PerViewDynamicMaskBank,
    cache: PreparedB7873200Cache,
    frequencies: torch.Tensor,
    kvector: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    completed_step: int,
) -> Dict[str, Any]:
    """Run one recipe-selected GeRaF update for a compatible cache facade.

    This is the narrow seam used by the bounded 16/4 engineering driver.  The
    production loop below calls the same function, so the adapter does not
    maintain a copied training loop.
    """

    started = time.perf_counter()
    model.train()
    view_index = sampler.next()
    view = load_training_view(cache, view_index, "train", args, device)
    optimizer.zero_grad(set_to_none=True)
    hardened = getattr(args, "implementation", "legacy") == "hardened_v1"
    diagnostics = None
    if hardened:
        prediction = predict_normalized_magnitude_with_response(
            model, view, frequencies, kvector, cache, args, create_graph=True)
        predicted, diagnostics = prediction.normalized_magnitude, prediction.diagnostics
        valid_mask = mask_bank.valid_mask(view_index, view.target_normalized,
                                          points=view.samples.points)
    else:
        predicted = predict_normalized_magnitude(
            model, view, frequencies, kvector, cache, args, create_graph=True
        )
        valid_mask = mask_bank.valid_mask(view_index, predicted)
    loss = masked_magnitude_l2(predicted, view.target_normalized, valid_mask)
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError(
            f"non-finite B7873200 loss at step {int(completed_step) + 1}, view {view_index}"
        )
    loss.backward()
    _require_finite_gradients(model)
    if args.gradient_clip_norm > 0:
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
        )
        if not math.isfinite(gradient_norm):
            raise FloatingPointError(
                "B7873200 gradient norm overflowed despite finite elementwise gradients; "
                "refusing optimizer update"
            )
        _require_finite_gradients(model)
    else:
        gradient_norm = math.nan
    optimizer.step()
    _require_finite_optimization_state(model, optimizer)
    scheduler.step()
    _require_finite_optimization_state(model, optimizer, scheduler)
    sampler.record_update(view_index)
    result = {
        "step": int(completed_step) + 1,
        "view_index": int(view_index),
        "loss": float(loss.detach().cpu()),
        "valid_fraction": float(valid_mask.float().mean().detach().cpu()),
        "gradient_norm_before_clip": gradient_norm,
        "sdf_lr": float(optimizer.param_groups[0]["lr"]),
        "other_lr": float(optimizer.param_groups[1]["lr"]),
        "seconds": time.perf_counter() - started,
    }
    if hardened:
        target_energy = view.target_normalized.detach().square()
        result.update({
            "unmasked_mf_magnitude_mse": float((predicted.detach() - view.target_normalized).square().mean()),
            "excluded_target_energy_fraction": float(target_energy[~valid_mask].sum() / target_energy.sum().clamp_min(EPS)),
            "render_diagnostics": diagnostics, "training_coverage": sampler.coverage(),
        })
    del view, predicted, valid_mask, loss
    return result


def model_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    config = {
        "extent": float(args.scene_extent),
        "sdf_levels": int(args.sdf_levels),
        "sdf_hidden_dim": int(args.sdf_hidden_dim),
        "sdf_layers": int(args.sdf_layers),
        "sdf_skip_layer": None if args.sdf_skip_layer < 0 else int(args.sdf_skip_layer),
        "sdf_hidden_activation": "softplus",
        "sdf_softplus_beta": float(args.sdf_softplus_beta),
        "sdf_encoding_include_input": False,
        "sdf_encoding_coordinate_scale": 1.0,
        "reflectivity_levels": int(args.reflectivity_levels),
        "reflectivity_hidden_dim": int(args.reflectivity_hidden_dim),
        "reflectivity_layers": int(args.reflectivity_layers),
        "reflectivity_output_activation": args.reflectivity_output_activation,
        "reflectivity_softplus_beta": float(args.reflectivity_softplus_beta),
        "reflectivity_encoding_coordinate_scale": 1.0,
        "init_tx_amplitude": float(args.init_tx_amplitude),
        "init_inv_s": float(args.init_inv_s),
        "learnable_inv_s": bool(args.learnable_inv_s),
    }
    if getattr(args, "implementation", "legacy") == "hardened_v1":
        config.update(implementation="hardened_v1", sdf_initialization="upstream_geometric",
                      sdf_encoding_include_input=True,
                      sdf_encoding_coordinate_scale=1 / args.scene_extent,
                      sdf_output_scale=args.scene_extent)
    return config


def run_identity(
    args: argparse.Namespace, cache: PreparedB7873200Cache, model_config: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return direct semantic identity for this new cache/training lane."""

    identity = {
        "schema": RUN_IDENTITY_SCHEMA,
        "version": CHECKPOINT_VERSION,
        "method": METHOD_NAME,
        "paper_id": PAPER_ID,
        "sealed_protocol_identity": cache.sealed_identity,
        "cache_recipe": cache.recipe,
        "acquisition_record": {
            "schema": B787_3200_ACQUISITION_SCHEMA,
            "filename": B787_3200_CACHE_ACQUISITION_FILENAME,
        },
        "model": dict(model_config),
        "steps": int(args.steps),
        "seed": int(args.seed),
        "validation": {
            "every_optimizer_steps": int(args.validation_every),
            "dynamic_mask": False,
            "selection_metric": "mf_magnitude_mse",
        },
        "grid": {
            "scene_extent": float(args.scene_extent),
            "n_azimuth": int(args.n_azimuth),
            "n_elevation": int(args.n_elevation),
            "n_depth": int(args.n_depth),
            "aperture_scale": float(args.aperture_scale),
        },
        "render": {
            "lensless_correction": bool(args.lensless_correction),
            "detach_start_cdf": bool(args.detach_start_cdf),
            "directional_exponent": float(args.directional_exponent),
            "min_distance": float(args.min_distance),
        },
        "operator": {
            "backend": "range_nufft",
            "phase_sign": float(args.phase_sign),
            "compute_dtype": args.compute_dtype,
            "oversample": int(args.oversample),
            "kernel_width": int(args.kernel_width),
            "pair_chunk": int(args.pair_chunk),
            "point_chunk": int(args.point_chunk),
            "pair_sampling": "all calibrated Tx/Rx pairs from one view/plane",
            "effective_pairs_per_plane": int(cache.effective_pairs_per_plane),
            "paper_signal_tracing_bank_implemented": False,
        },
        "optimizer": {
            "name": "AdamW",
            "sdf_lr": float(args.sdf_lr),
            "other_lr": float(args.other_lr),
            "weight_decay": float(args.weight_decay),
            "betas": [float(args.adam_beta1), float(args.adam_beta2)],
            "eps": float(args.adam_eps),
            "scheduler": "CosineAnnealingLR",
            "cosine_min_lr": float(args.cosine_min_lr),
            "gradient_clip_norm": float(args.gradient_clip_norm),
        },
        "loss": {
            "name": "full_3d_normalized_geraf_eq3_mf_magnitude_l2",
            "readout": "|complex matched-filter amplitude|; not squared",
            "normalization_peak": float(cache.geraf_mf_magnitude_peak),
            "clip": False,
            "dynamic_mask_scope": "independent historical bank per training view",
            "mask_high_threshold": float(args.mask_high_threshold),
            "mask_low_ratio": float(args.mask_low_ratio),
            "mask_low_threshold": float(args.mask_low_threshold),
            "validation_dynamic_mask": False,
        },
    }
    if getattr(args, "implementation", "legacy") == "hardened_v1":
        from rift.geraf_v1 import UPSTREAM_COMMIT
        identity["implementation"] = "hardened_v1"
        identity["upstream_reference_commit"] = UPSTREAM_COMMIT
        identity["render"].update(
            opacity="neus_gradient_cell_endpoints_v1", boundary="scene_aabb_array_mean_tx_rx_v1",
            sampling="full_scene_aabb_midpoint_cells_v1", target_queries="unchanged_native_cache_grid",
            eikonal_weight=0.0,
        )
        identity["loss"].update(dynamic_mask_scope="measured_train_only_world_reference_ray_mask_v1",
                               measured_mask_application="loss_queries_only_full_scene_signal",
                               measured_current_fraction=args.measured_mask_current_fraction,
                               measured_accumulated_fraction=args.measured_mask_accumulated_fraction,
                               measured_reference_grid=args.measured_mask_grid)
    return identity


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.cpu() if key == "step" and value.numel() == 1 else value.to(device)


def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _capture_rng_state(sampler: DeterministicViewSampler) -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "view_sampler": sampler.state_dict(),
    }


def _restore_rng_state(state: Mapping[str, Any], sampler: DeterministicViewSampler) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all([tensor.cpu() for tensor in state["torch_cuda"]])
    sampler.load_state_dict(state["view_sampler"])


def _require_finite_gradients(model: torch.nn.Module) -> None:
    """Reject a finite scalar loss whose backward pass produced invalid gradients."""

    invalid = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item())
    ]
    if invalid:
        raise FloatingPointError(
            "B7873200 loss backward produced non-finite gradients; refusing optimizer update: "
            + ", ".join(invalid[:8])
        )


def _collect_nonfinite_numeric_state(value: Any, label: str, invalid: list[str]) -> None:
    """Append direct paths to non-finite numeric state without coercing objects."""

    if torch.is_tensor(value):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all().item()
        ):
            invalid.append(label)
        return
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "fc" and not bool(np.isfinite(value).all()):
            invalid.append(label)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _collect_nonfinite_numeric_state(item, f"{label}.{key}", invalid)
        return
    if isinstance(value, (tuple, list)):
        for number, item in enumerate(value):
            _collect_nonfinite_numeric_state(item, f"{label}[{number}]", invalid)
        return
    if isinstance(value, (float, complex, np.floating, np.complexfloating)) and not bool(
        np.isfinite(value)
    ):
        invalid.append(label)


def _require_finite_optimization_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
) -> None:
    """Reject non-finite post-update state before it can become resumable."""

    invalid_model = [
        name
        for name, parameter in model.named_parameters()
        if not bool(torch.isfinite(parameter.detach()).all().item())
    ]
    invalid_model.extend(
        name
        for name, buffer in model.named_buffers()
        if (buffer.is_floating_point() or buffer.is_complex())
        and not bool(torch.isfinite(buffer.detach()).all().item())
    )
    if invalid_model:
        raise FloatingPointError(
            "B7873200 optimizer update produced non-finite model parameters or buffers: "
            + ", ".join(invalid_model[:8])
        )
    invalid_state: list[str] = []
    for group_number, group in enumerate(optimizer.param_groups):
        for key, value in group.items():
            if key != "params":
                _collect_nonfinite_numeric_state(
                    value, f"optimizer.param_groups[{group_number}].{key}", invalid_state
                )
        for parameter_number, parameter in enumerate(group["params"]):
            for key, value in optimizer.state.get(parameter, {}).items():
                _collect_nonfinite_numeric_state(
                    value, f"optimizer.state[{group_number}][{parameter_number}].{key}", invalid_state
                )
    if scheduler is not None:
        _collect_nonfinite_numeric_state(scheduler.state_dict(), "scheduler", invalid_state)
    if invalid_state:
        raise FloatingPointError(
            "B7873200 optimizer or scheduler state is non-finite: "
            + ", ".join(invalid_state[:8])
        )


def make_checkpoint(
    *,
    args: argparse.Namespace,
    cache: PreparedB7873200Cache,
    model: GeRaFModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    sampler: DeterministicViewSampler,
    mask_bank: PerViewDynamicMaskBank,
    step: int,
    best_val_mse: float,
    history: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
    last_train: Optional[Mapping[str, Any]],
    pending_validation_step: Optional[int],
    stop_reason: Optional[str],
    complete: bool,
) -> Dict[str, Any]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "method": METHOD_NAME,
        "paper_id": PAPER_ID,
        "step": int(step),
        "complete": bool(complete),
        "stop_reason": stop_reason,
        "best_val_mse": float(best_val_mse),
        "model_config": dict(identity["model"]),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": _capture_rng_state(sampler),
        "dynamic_mask_bank": mask_bank.state_dict(),
        **({"training_coverage": sampler.coverage()}
           if getattr(args, "implementation", "legacy") == "hardened_v1" else {}),
        "history": [dict(row) for row in history],
        "last_train": None if last_train is None else dict(last_train),
        "pending_validation_step": (
            None if pending_validation_step is None else int(pending_validation_step)
        ),
        "run_identity": dict(identity),
        "cache_recipe": cache.recipe,
        "acquisition_record": cache.acquisition_record,
        "target_stats": cache.stats,
        "target_manifest": cache.target_manifest,
        "dataset_provenance": {"canonical_path": str(Path(args.npz_path).resolve())},
        "cli_args": vars(args).copy(),
        "saved_unix_time": time.time(),
    }


def write_checkpoint(name: str, checkpoint_dir: Path, **checkpoint_kwargs: Any) -> Path:
    destination = checkpoint_dir / name
    _atomic_torch_save(make_checkpoint(**checkpoint_kwargs), destination)
    return destination


def evaluate_indices(
    model: GeRaFModel,
    cache: PreparedB7873200Cache,
    frequencies: torch.Tensor,
    kvector: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    *,
    indices: Sequence[int],
    role: str,
) -> tuple[Dict[str, float], bool]:
    """Evaluate named target IDs without reading or updating dynamic masks.

    The full trainer calls this only for its validation role.  The bounded
    engineering adapter also uses it for a frozen training-set measurement;
    neither path touches ``PerViewDynamicMaskBank``.
    """

    if role not in {"train", "validation"}:
        raise ValueError(f"B7873200 evaluator does not expose role {role!r}")
    requested = tuple(int(index) for index in indices)
    allowed = cache.train_indices if role == "train" else cache.validation_indices
    if not requested or any(index not in allowed for index in requested):
        raise ValueError(f"B7873200 evaluator received an unauthorized {role} target")

    model.eval()
    squared_error = 0.0
    target_energy = 0.0
    count = 0
    completed_views = 0
    with torch.enable_grad():
        for index in requested:
            if _STOP_REQUESTED:
                break
            view = load_training_view(cache, index, role, args, device)
            predicted = predict_normalized_magnitude(
                model, view, frequencies, kvector, cache, args, create_graph=False
            ).detach()
            residual = predicted - view.target_normalized
            squared_error += float(residual.square().sum().cpu())
            target_energy += float(view.target_normalized.square().sum().cpu())
            count += residual.numel()
            completed_views += 1
            del view, predicted, residual
    if completed_views != len(requested):
        return {"views": float(completed_views), "voxels": float(count)}, False
    mse = squared_error / max(count, 1)
    return {
        "views": float(completed_views),
        "voxels": float(count),
        "mf_magnitude_mse": mse,
        "mf_magnitude_rmse": math.sqrt(mse),
        "mf_magnitude_relative_mse": squared_error / max(target_energy, EPS),
        "mf_magnitude_psnr_db": -10.0 * math.log10(max(mse, EPS)),
    }, True


def validate(
    model: GeRaFModel,
    cache: PreparedB7873200Cache,
    frequencies: torch.Tensor,
    kvector: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Dict[str, float], bool]:
    """Evaluate validation targets without reading/updating dynamic masks."""

    return evaluate_indices(
        model,
        cache,
        frequencies,
        kvector,
        args,
        device,
        indices=cache.validation_indices,
        role="validation",
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _resume_path(args: argparse.Namespace, checkpoint_dir: Path) -> Optional[Path]:
    if args.resume_path is not None:
        path = Path(args.resume_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"explicit resume checkpoint is missing: {path}")
        return path
    if args.resume:
        final_path = checkpoint_dir / "checkpoint_final.pth.tar"
        latest_path = checkpoint_dir / "checkpoint_latest.pth.tar"
        return final_path if final_path.is_file() else latest_path if latest_path.is_file() else None
    stale = [
        path
        for path in (
            checkpoint_dir / "checkpoint_latest.pth.tar",
            checkpoint_dir / "checkpoint_best.pth.tar",
            checkpoint_dir / "checkpoint_final.pth.tar",
        )
        if path.exists()
    ]
    if stale:
        raise FileExistsError(
            "--no-resume refuses to overwrite an existing sealed B7873200 identity; "
            f"use a new --checkpoint-dir (found {stale[0]})"
        )
    return None


def _begin_pending_validation(step: int, pending_validation_step: Optional[int]) -> int:
    """Mark a completed update whose validation must finish before another update."""

    if pending_validation_step is not None:
        raise ValueError(
            f"B7873200 cannot schedule validation at step {step} while step "
            f"{pending_validation_step} remains pending"
        )
    return int(step)


def _commit_pending_validation(
    *,
    step: int,
    pending_validation_step: Optional[int],
    metrics: Mapping[str, float],
    history: list[Dict[str, Any]],
    best_val_mse: float,
) -> tuple[Optional[int], float, bool]:
    """Commit one complete validation exactly once and choose a new best row."""

    if pending_validation_step != int(step):
        raise ValueError(
            f"B7873200 attempted to commit validation for step {step} without matching pending state"
        )
    if history and int(history[-1].get("step", -1)) >= int(step):
        raise ValueError(f"B7873200 validation history already contains step {step}")
    row = {"step": int(step), **dict(metrics)}
    history.append(row)
    mse = float(metrics["mf_magnitude_mse"])
    if not math.isfinite(mse):
        raise FloatingPointError("B7873200 validation produced a non-finite MSE")
    improved = mse < float(best_val_mse)
    return None, mse if improved else float(best_val_mse), improved


def _checkpoint_integer(name: str, value: object, *, minimum: int, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"B7873200 checkpoint {name} must be an integer")
    result = int(value)
    if result < minimum or result > maximum:
        raise ValueError(
            f"B7873200 checkpoint {name}={result} is outside [{minimum}, {maximum}]"
        )
    return result


def _checkpoint_finite_float(
    name: str, value: object, *, minimum: Optional[float] = None
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"B7873200 checkpoint {name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"B7873200 checkpoint {name} must be numeric") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        qualifier = "finite" if minimum is None else f"finite and >= {minimum:g}"
        raise ValueError(f"B7873200 checkpoint {name} must be {qualifier}")
    return result


def _require_checkpoint_close(name: str, observed: object, expected: object) -> None:
    try:
        observed_float = _checkpoint_finite_float(name, observed)
    except ValueError:
        raise
    if not math.isclose(observed_float, float(expected), rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError(
            f"B7873200 checkpoint {name}={observed_float!r} differs from this sealed run"
        )


def _validate_checkpoint_optimizer_scheduler(
    checkpoint: Mapping[str, Any], args: argparse.Namespace, step: int
) -> None:
    """Reject a resume payload whose saved update semantics differ from this run."""

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if not isinstance(optimizer_state, Mapping):
        raise ValueError("B7873200 checkpoint lacks an optimizer state mapping")
    groups = optimizer_state.get("param_groups")
    if not isinstance(groups, list) or len(groups) != 2:
        raise ValueError("B7873200 checkpoint must contain exactly SDF and non-SDF optimizer groups")
    expected_groups = (
        ("sdf", float(args.sdf_lr)),
        ("other", float(args.other_lr)),
    )
    for number, (group, (name, initial_lr)) in enumerate(zip(groups, expected_groups)):
        if not isinstance(group, Mapping):
            raise ValueError(f"B7873200 checkpoint optimizer group {number} is malformed")
        if group.get("name") != name:
            raise ValueError(f"B7873200 checkpoint optimizer group {number} is not {name!r}")
        _require_checkpoint_close(f"optimizer group {name} betas[0]", group.get("betas", (None,))[0] if isinstance(group.get("betas"), (tuple, list)) and len(group.get("betas")) == 2 else None, args.adam_beta1)
        _require_checkpoint_close(f"optimizer group {name} betas[1]", group.get("betas", (None, None))[1] if isinstance(group.get("betas"), (tuple, list)) and len(group.get("betas")) == 2 else None, args.adam_beta2)
        _require_checkpoint_close(f"optimizer group {name} eps", group.get("eps"), args.adam_eps)
        _require_checkpoint_close(
            f"optimizer group {name} weight_decay", group.get("weight_decay"), args.weight_decay
        )
        _require_checkpoint_close(f"optimizer group {name} initial_lr", group.get("initial_lr"), initial_lr)
        current_lr = _checkpoint_finite_float(
            f"optimizer group {name} lr", group.get("lr"), minimum=0.0
        )
        if current_lr > initial_lr + 1.0e-15:
            raise ValueError(f"B7873200 checkpoint optimizer group {name} lr exceeds its initial lr")
        params = group.get("params")
        if not isinstance(params, list) or not params:
            raise ValueError(f"B7873200 checkpoint optimizer group {name} lacks parameters")

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if not isinstance(scheduler_state, Mapping):
        raise ValueError("B7873200 checkpoint lacks a scheduler state mapping")
    _checkpoint_integer("scheduler T_max", scheduler_state.get("T_max"), minimum=1, maximum=args.steps)
    if int(scheduler_state.get("T_max")) != int(args.steps):
        raise ValueError("B7873200 checkpoint scheduler T_max differs from this sealed run")
    _require_checkpoint_close("scheduler eta_min", scheduler_state.get("eta_min"), args.cosine_min_lr)
    base_lrs = scheduler_state.get("base_lrs")
    if not isinstance(base_lrs, list) or len(base_lrs) != 2:
        raise ValueError("B7873200 checkpoint scheduler base_lrs must have two entries")
    for number, expected in enumerate((args.sdf_lr, args.other_lr)):
        _require_checkpoint_close(f"scheduler base_lrs[{number}]", base_lrs[number], expected)
    last_epoch = _checkpoint_integer(
        "scheduler last_epoch", scheduler_state.get("last_epoch"), minimum=0, maximum=args.steps
    )
    if last_epoch != step:
        raise ValueError("B7873200 checkpoint scheduler progress does not match optimizer step")
    saved_lrs = scheduler_state.get("_last_lr")
    if not isinstance(saved_lrs, list) or len(saved_lrs) != 2:
        raise ValueError("B7873200 checkpoint scheduler _last_lr must have two entries")
    for number, value in enumerate(saved_lrs):
        saved_lr = _checkpoint_finite_float(f"scheduler _last_lr[{number}]", value, minimum=0.0)
        group_lr = _checkpoint_finite_float(
            f"optimizer group {expected_groups[number][0]} lr", groups[number].get("lr"), minimum=0.0
        )
        if saved_lr != group_lr:
            raise ValueError("B7873200 checkpoint scheduler lr differs from its optimizer group")


def _validate_checkpoint_history(
    checkpoint: Mapping[str, Any], args: argparse.Namespace, cache: PreparedB7873200Cache, step: int
) -> tuple[list[Dict[str, Any]], float, Optional[int], bool]:
    """Validate pending-validation and best-selection state before restoring it."""

    raw_history = checkpoint.get("history", [])
    if not isinstance(raw_history, list):
        raise ValueError("B7873200 checkpoint history must be a list")
    history: list[Dict[str, Any]] = []
    previous_step = 0
    expected_views = float(len(cache.validation_indices))
    expected_voxels = float(len(cache.validation_indices) * int(np.prod(cache.grid_shape)))
    required_metric_minima = {
        "views": 0.0,
        "voxels": 0.0,
        "mf_magnitude_mse": 0.0,
        "mf_magnitude_rmse": 0.0,
        "mf_magnitude_relative_mse": 0.0,
        "mf_magnitude_psnr_db": None,
    }
    for number, raw_row in enumerate(raw_history):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"B7873200 checkpoint history row {number} is not an object")
        row = dict(raw_row)
        row_step = _checkpoint_integer(
            f"history row {number}.step", row.get("step"), minimum=1, maximum=step
        )
        if row_step <= previous_step:
            raise ValueError("B7873200 checkpoint history steps must be strictly increasing")
        if row_step % int(args.validation_every) != 0 and row_step != int(args.steps):
            raise ValueError("B7873200 checkpoint history contains an unscheduled validation step")
        previous_step = row_step
        for metric, minimum in required_metric_minima.items():
            _checkpoint_finite_float(f"history row {number}.{metric}", row.get(metric), minimum=minimum)
        if float(row["views"]) != expected_views or float(row["voxels"]) != expected_voxels:
            raise ValueError(
                f"B7873200 checkpoint history row {number} does not cover the complete validation role"
            )
        history.append(row)

    raw_pending = checkpoint.get("pending_validation_step")
    pending = None
    if raw_pending is not None:
        pending = _checkpoint_integer(
            "pending validation step", raw_pending, minimum=1, maximum=step
        )
        if pending != step:
            raise ValueError("B7873200 checkpoint pending validation must belong to its optimizer step")
        if pending % int(args.validation_every) != 0 and pending != int(args.steps):
            raise ValueError("B7873200 checkpoint pending validation is not scheduled")
        if any(int(row["step"]) == pending for row in history):
            raise ValueError("B7873200 checkpoint cannot both commit and pend the same validation step")

    scheduled_steps = list(range(int(args.validation_every), step + 1, int(args.validation_every)))
    if step == int(args.steps) and step not in scheduled_steps:
        scheduled_steps.append(step)
    expected_history_steps = [candidate for candidate in scheduled_steps if candidate != pending]
    observed_history_steps = [int(row["step"]) for row in history]
    if observed_history_steps != expected_history_steps:
        raise ValueError(
            "B7873200 checkpoint validation history must cover every due step before resume"
        )

    raw_best = checkpoint.get("best_val_mse")
    if not history:
        try:
            best = float(raw_best)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("B7873200 checkpoint best validation MSE is invalid") from exc
        if best != math.inf:
            raise ValueError("B7873200 checkpoint without validation history must retain +inf best MSE")
    else:
        best = _checkpoint_finite_float("best validation MSE", raw_best, minimum=0.0)
        minimum_mse = min(float(row["mf_magnitude_mse"]) for row in history)
        if best != minimum_mse:
            raise ValueError("B7873200 checkpoint best validation MSE disagrees with validation history")

    complete = checkpoint.get("complete")
    if not isinstance(complete, bool):
        raise ValueError("B7873200 checkpoint complete flag must be boolean")
    terminal = (
        step == int(args.steps)
        and pending is None
        and bool(history)
        and int(history[-1]["step"]) == step
    )
    if complete != terminal:
        raise ValueError(
            "B7873200 checkpoint complete flag must match completed optimizer and final-validation state"
        )
    return history, best, pending, complete


def _validate_resume_checkpoint_payload(
    checkpoint: object,
    args: argparse.Namespace,
    cache: PreparedB7873200Cache,
) -> tuple[int, list[Dict[str, Any]], float, Optional[int], bool]:
    """Validate every saved control state before restoring model or optimizer state."""

    if not isinstance(checkpoint, Mapping):
        raise ValueError("B7873200 checkpoint payload must be a mapping")
    step = _checkpoint_integer("step", checkpoint.get("step"), minimum=0, maximum=int(args.steps))
    history, best, pending, complete = _validate_checkpoint_history(checkpoint, args, cache, step)
    _validate_checkpoint_optimizer_scheduler(checkpoint, args, step)
    for field in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "rng_state", "dynamic_mask_bank"):
        if not isinstance(checkpoint.get(field), Mapping):
            raise ValueError(f"B7873200 checkpoint lacks mapping {field!r}")
    rng_state = checkpoint["rng_state"]
    if not all(key in rng_state for key in ("python", "numpy_global", "torch_cpu", "view_sampler")):
        raise ValueError("B7873200 checkpoint RNG state is incomplete")
    mask_state = checkpoint["dynamic_mask_bank"]
    if not all(key in mask_state for key in ("shape", "high_threshold", "low_ratio", "low_threshold", "histories")):
        raise ValueError("B7873200 checkpoint dynamic-mask state is incomplete")
    if getattr(args, "implementation", "legacy") == "hardened_v1":
        sampler_state = rng_state["view_sampler"]
        counts = sampler_state.get("exposures")
        if (not isinstance(counts, Mapping) or set(counts) != set(cache.train_indices)
                or any(type(n) is not int or n < 0 for n in counts.values())
                or sum(counts.values()) != step):
            raise ValueError("hardened GeRaF checkpoint training coverage disagrees with optimizer step")
        order = np.asarray(sampler_state.get("order", ()), dtype=np.int64)
        cursor = _checkpoint_integer("view_sampler.cursor", sampler_state.get("cursor"),
                                     minimum=0, maximum=len(cache.train_indices))
        expected_cursor = ((step - 1) % len(cache.train_indices) + 1) if step else 0
        if (cursor != expected_cursor or (step and (order.shape != (len(cache.train_indices),)
                or set(order.tolist()) != set(cache.train_indices)))):
            raise ValueError("hardened GeRaF sampler cycle disagrees with optimizer step")
        cycles = (step - cursor) // len(cache.train_indices)
        prefix = set(order[:cursor].tolist())
        if any(counts[index] != cycles + int(index in prefix) for index in cache.train_indices):
            raise ValueError("hardened GeRaF exposure ledger disagrees with its sampled permutation")
        if mask_state.get("schema") != "rift_geraf_measured_mask_bank_v1":
            raise ValueError("hardened GeRaF requires measured-data mask state")
    last_train = checkpoint.get("last_train")
    if last_train is not None:
        if not isinstance(last_train, Mapping):
            raise ValueError("B7873200 checkpoint last_train must be an object or null")
        if _checkpoint_integer("last_train.step", last_train.get("step"), minimum=1, maximum=step) != step:
            raise ValueError("B7873200 checkpoint last_train step disagrees with optimizer step")
        view_index = _checkpoint_integer(
            "last_train.view_index", last_train.get("view_index"), minimum=0, maximum=9_999
        )
        if view_index not in cache.train_indices:
            raise ValueError("B7873200 checkpoint last_train refers to a non-training view")
    elif step != 0:
        raise ValueError("B7873200 checkpoint after an optimizer step lacks last_train")
    return step, history, best, pending, complete


def _record_validation(
    checkpoint_dir: Path,
    step: int,
    metrics: Mapping[str, float],
    history: list[Dict[str, Any]],
) -> None:
    if not history or int(history[-1].get("step", -1)) != int(step):
        raise ValueError("B7873200 validation must be committed before it is recorded")
    atomic_write_json(checkpoint_dir / "history.json", {"validation": history})
    print(
        f"validation step={step}: mse={metrics['mf_magnitude_mse']:.7e} "
        f"rel_mse={metrics['mf_magnitude_relative_mse']:.7e} "
        f"psnr={metrics['mf_magnitude_psnr_db']:.3f} dB (dynamic mask disabled)",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if getattr(args, "implementation", "legacy") == "source_v1":
        from rift.geraf_source_cli import run
        return run(args)
    _validate_cli(args)
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    _publish_timed_wrapper_ready()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    _seed_everything(args.seed)
    cache = verify_prepared_b7873200_cache(args)
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_config = model_config_from_args(args)
    identity = run_identity(args, cache, model_config)
    run_config_path = checkpoint_dir / "run_config.json"
    if run_config_path.is_file():
        prior = _read_json(run_config_path, "run configuration")
        _require_equal("existing run identity", prior.get("run_identity"), identity)
    atomic_write_json(
        run_config_path,
        {
            "run_identity": identity,
            "cache_recipe": cache.recipe,
            "acquisition_record": {
                "schema": B787_3200_ACQUISITION_SCHEMA,
                "filename": B787_3200_CACHE_ACQUISITION_FILENAME,
            },
            "target_stats": cache.stats,
            "target_manifest": cache.target_manifest,
            "dataset_provenance": {"canonical_path": str(Path(args.npz_path).resolve())},
            "cli_args": vars(args),
        },
    )
    model = GeRaFModel(**model_config).to(device=device, dtype=torch.float32)
    optimizer = build_geraf_optimizer(model, args)
    scheduler = build_geraf_scheduler(optimizer, args)
    sampler = DeterministicViewSampler(cache.train_indices, args.seed)
    if getattr(args, "implementation", "legacy") == "hardened_v1":
        mask_bank = _measured_mask_bank(cache, args)
    else:
        mask_bank = PerViewDynamicMaskBank(
            cache.train_indices,
            cache.grid_shape,
            high_threshold=args.mask_high_threshold,
            low_ratio=args.mask_low_ratio,
            low_threshold=args.mask_low_threshold,
            device=device,
        )
    operator_frequency_hz = frequency_grid_hz(cache.source.arrays.metadata)
    frequencies = torch.as_tensor(
        validate_b7873200_operator_frequency_grid(cache.acquisition_record, operator_frequency_hz),
        dtype=torch.float64,
        device=device,
    )
    kvector = get_kvector(frequencies, cc)
    step = 0
    best_val_mse = math.inf
    history: list[Dict[str, Any]] = []
    last_train: Optional[Dict[str, Any]] = None
    pending_validation_step: Optional[int] = None
    checkpoint_complete = False
    resume_path = _resume_path(args, checkpoint_dir)
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("B7873200 checkpoint payload must be a mapping")
        _require_equal("checkpoint version", checkpoint.get("checkpoint_version"), CHECKPOINT_VERSION)
        _require_equal("checkpoint identity", checkpoint.get("run_identity"), identity)
        saved_acquisition = checkpoint.get("acquisition_record")
        if not isinstance(saved_acquisition, Mapping) or not acquisition_records_equal(
            saved_acquisition, cache.acquisition_record
        ):
            raise ValueError(
                "B7873200 checkpoint was made with different calibrated poses, metadata, or frequency grid"
            )
        step, history, best_val_mse, pending_validation_step, checkpoint_complete = (
            _validate_resume_checkpoint_payload(checkpoint, args, cache)
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _optimizer_to(optimizer, device)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        sampler.load_state_dict(checkpoint["rng_state"]["view_sampler"])
        mask_bank.load_state_dict(checkpoint["dynamic_mask_bank"])
        _restore_rng_state(checkpoint["rng_state"], sampler)
        last_train = checkpoint.get("last_train")
        _require_finite_optimization_state(model, optimizer, scheduler)
        mask_description = ("restored measured training reference"
                            if getattr(args, "implementation", "legacy") == "hardened_v1"
                            else f"restored {len(mask_bank.histories)} per-view mask histories")
        print(f"Resumed {resume_path} at completed step {step}; {mask_description}.", flush=True)
    if getattr(args, "implementation", "legacy") == "hardened_v1" and resume_path is None:
        print("Preparing measured mask reference from all training targets (validation excluded).", flush=True)
        _prepare_measured_mask_reference(mask_bank, cache)

    checkpoint_common = {
        "args": args,
        "cache": cache,
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "sampler": sampler,
        "mask_bank": mask_bank,
        "identity": identity,
    }

    def terminal_checkpoint_state() -> bool:
        return (
            step == int(args.steps)
            and pending_validation_step is None
            and bool(history)
            and int(history[-1].get("step", -1)) == step
        )

    def save_checkpoint(
        name: str, *, stop_reason: Optional[str], complete: bool
    ) -> Path:
        _require_finite_optimization_state(model, optimizer, scheduler)
        terminal = terminal_checkpoint_state()
        if complete and not terminal:
            raise ValueError("cannot mark B7873200 checkpoint complete before final validation")
        return write_checkpoint(
            name,
            checkpoint_dir,
            step=step,
            best_val_mse=best_val_mse,
            history=history,
            last_train=last_train,
            pending_validation_step=pending_validation_step,
            stop_reason=stop_reason,
            complete=terminal,
            **checkpoint_common,
        )

    def materialize_complete_artifacts() -> tuple[Path, Path]:
        """Atomically write the durable terminal pair without changing training state."""

        if not terminal_checkpoint_state():
            raise ValueError("cannot materialize B7873200 completion before final validation")
        final_path = save_checkpoint("checkpoint_final.pth.tar", stop_reason=None, complete=True)
        latest_path = save_checkpoint("checkpoint_latest.pth.tar", stop_reason=None, complete=True)
        return final_path, latest_path

    if checkpoint_complete:
        final_path, latest_path = materialize_complete_artifacts()
        print(
            f"GeRaF sealed B7873200 run is already complete at step {step}; "
            f"ensured {final_path.name} and {latest_path.name}.",
            flush=True,
        )
        return

    def finish_pending_validation() -> bool:
        """Finish the saved validation obligation before another optimizer step."""

        nonlocal pending_validation_step, best_val_mse
        if pending_validation_step is None:
            return True
        if pending_validation_step != step:
            raise ValueError("B7873200 pending validation does not match the current optimizer state")
        metrics, complete = validate(model, cache, frequencies, kvector, args, device)
        if not complete:
            return False
        pending_validation_step, best_val_mse, improved = _commit_pending_validation(
            step=step,
            pending_validation_step=pending_validation_step,
            metrics=metrics,
            history=history,
            best_val_mse=best_val_mse,
        )
        _record_validation(checkpoint_dir, step, metrics, history)
        if improved:
            best_path = save_checkpoint(
                "checkpoint_best.pth.tar", stop_reason=None, complete=step >= args.steps
            )
            print(f"Saved new best checkpoint: {best_path}", flush=True)
        return True

    print(
        f"Training {METHOD_NAME} ({PAPER_ID}) on {device}: steps={args.steps}, "
        f"SDF lr={args.sdf_lr:g}, other lr={args.other_lr:g}, "
        f"rays={args.n_azimuth * args.n_elevation}x{args.n_depth}",
        flush=True,
    )
    last_checkpoint_time = time.monotonic()
    signal_checkpoint_written = False
    if pending_validation_step is not None and not _STOP_REQUESTED:
        if not finish_pending_validation() and not _STOP_REQUESTED:
            raise RuntimeError("B7873200 validation ended incomplete without a stop request")
    if _STOP_REQUESTED:
        if terminal_checkpoint_state():
            final_path, latest_path = materialize_complete_artifacts()
            print(
                f"Training complete after signal at step {step}: {final_path}; "
                f"ensured {latest_path.name}.",
                flush=True,
            )
            return
        latest_path = save_checkpoint(
            "checkpoint_latest.pth.tar", stop_reason=f"signal:{_STOP_SIGNAL}", complete=False
        )
        print(f"Safe stop at step {step}; resume from {latest_path}", flush=True)
        raise SystemExit(CLEAN_STOP_EXIT_CODE)

    while step < args.steps and not _STOP_REQUESTED:
        if pending_validation_step is not None:
            if not finish_pending_validation() and not _STOP_REQUESTED:
                raise RuntimeError("B7873200 validation ended incomplete without a stop request")
            if _STOP_REQUESTED:
                break
        last_train = train_one_view_update(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            mask_bank=mask_bank,
            cache=cache,
            frequencies=frequencies,
            kvector=kvector,
            args=args,
            device=device,
            completed_step=step,
        )
        step = int(last_train["step"])
        if step == 1 or step % args.log_every == 0:
            print(
                f"step={step}/{args.steps} view={last_train['view_index']} "
                f"loss={last_train['loss']:.7e} valid={last_train['valid_fraction']:.4f} "
                f"lr=({last_train['sdf_lr']:.3e},{last_train['other_lr']:.3e}) "
                f"seconds={last_train['seconds']:.2f}",
                flush=True,
            )
            if getattr(args, "implementation", "legacy") == "hardened_v1":
                diagnostics = {key: last_train[key] for key in (
                    "step", "view_index", "unmasked_mf_magnitude_mse",
                    "excluded_target_energy_fraction", "render_diagnostics")}
                diagnostics["training_coverage"] = {
                    key: value for key, value in last_train["training_coverage"].items()
                    if key != "per_view_exposures"}
                print("GERAF_V1_DIAGNOSTICS " + json.dumps(diagnostics, allow_nan=False), flush=True)
        if step % args.validation_every == 0 or step == args.steps:
            pending_validation_step = _begin_pending_validation(step, pending_validation_step)
            if not _STOP_REQUESTED:
                if not finish_pending_validation() and not _STOP_REQUESTED:
                    raise RuntimeError("B7873200 validation ended incomplete without a stop request")
        due_by_step = step % args.checkpoint_every == 0
        due_by_time = time.monotonic() - last_checkpoint_time >= args.checkpoint_seconds
        if due_by_step or due_by_time or _STOP_REQUESTED:
            latest_path = save_checkpoint(
                "checkpoint_latest.pth.tar",
                stop_reason=(f"signal:{_STOP_SIGNAL}" if _STOP_REQUESTED else None),
                complete=False,
            )
            last_checkpoint_time = time.monotonic()
            signal_checkpoint_written = signal_checkpoint_written or _STOP_REQUESTED
            print(f"Saved latest checkpoint: {latest_path}", flush=True)

    if _STOP_REQUESTED:
        if terminal_checkpoint_state():
            final_path, latest_path = materialize_complete_artifacts()
            print(
                f"Training complete after signal at step {step}: {final_path}; "
                f"ensured {latest_path.name}.",
                flush=True,
            )
            return
        if not signal_checkpoint_written:
            latest_path = save_checkpoint(
                "checkpoint_latest.pth.tar", stop_reason=f"signal:{_STOP_SIGNAL}", complete=False
            )
        else:
            latest_path = checkpoint_dir / "checkpoint_latest.pth.tar"
        print(f"Safe stop at step {step}; resume from {latest_path}", flush=True)
        raise SystemExit(CLEAN_STOP_EXIT_CODE)

    if pending_validation_step is None and (
        not history or int(history[-1].get("step", -1)) != step
    ):
        pending_validation_step = _begin_pending_validation(step, pending_validation_step)
    if pending_validation_step is not None:
        if not finish_pending_validation():
            latest_path = save_checkpoint(
                "checkpoint_latest.pth.tar", stop_reason=f"signal:{_STOP_SIGNAL}", complete=False
            )
            print(f"Safe stop during final validation; resume from {latest_path}", flush=True)
            raise SystemExit(CLEAN_STOP_EXIT_CODE)
    final_path, _ = materialize_complete_artifacts()
    print(f"Training complete: {final_path}", flush=True)


if __name__ == "__main__":
    main()
