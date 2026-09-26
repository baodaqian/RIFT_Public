#!/usr/bin/env python
"""Prepare the isolated sealed B7873200 native GeRaF ``|MF|`` cache.

This is deliberately a new preparation lane.  It leaves the historical
``prepare_b787_power_targets.py`` cache, hashes, and resume contract intact.
The legacy default uses the canonical B787 archive; --object/--dataset-root
select an object-bound RIFT collection manifest. The sealed source adapter
validates the manifest and response header before it can expose a response
row, and its guarded accessor permits only the train and validation roles.

The target remains GeRaF's native three-dimensional complex matched-filter
magnitude (not squared power) on the existing calibrated primary-ray grid.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rift.geraf_b7873200_protocol import (  # noqa: E402
    B787_3200_ACQUISITION_SCHEMA,
    B787_3200_CACHE_ACQUISITION_FILENAME,
    B787_3200_CANONICAL_MANIFEST_PATH,
    B787_3200_CANONICAL_NPZ_PATH,
    write_b7873200_target_cache_protocol,
)
from rift.geraf_b7873200_acquisition import (  # noqa: E402
    validate_b7873200_operator_frequency_grid,
    validate_b7873200_acquisition_record,
    write_or_validate_b7873200_acquisition_record,
)
from rift.geraf_b7873200_source import load_b7873200_development_source  # noqa: E402
from rift.geraf_b7873200_preparation_state import (  # noqa: E402
    CLEAN_STOP_EXIT_CODE,
    PREPARATION_STATE_FILENAME,
    PREPARATION_STATE_SCHEMA,
    build_preparation_state,
    complete_preparation_phase,
)
from rift.matched_filter_power import matched_filter_complex  # noqa: E402
from rift.power_baseline_dataset import (  # noqa: E402
    atomic_save_target,
    atomic_write_json,
    build_lensless_grid,
    frequency_grid_hz,
)


CACHE_SCHEMA = "rift_geraf_b7873200_native_mf_cache_v1"
TARGET_SCHEMA = "rift_geraf_b7873200_native_mf_target_v1"
RECIPE_FILENAME = "geraf_b7873200_recipe.json"
MANIFEST_FILENAME = "geraf_b7873200_manifest.json"
STATS_FILENAME = "geraf_b7873200_stats.json"
VIEWS_DIRECTORY = "geraf_b7873200_views"
_STOP_REQUESTED = False
_STOP_SIGNAL: int | None = None
_TIMED_WRAPPER_READY_ENV = "GERAF_TIMED_WRAPPER_READY_FILE"
_TIMED_WRAPPER_READY_MARKER = "GERAF_TIMED_WRAPPER_CHILD_READY"


def _request_stop(signum: int, _frame: object) -> None:
    global _STOP_REQUESTED, _STOP_SIGNAL
    _STOP_REQUESTED = True
    _STOP_SIGNAL = int(signum)
    print(
        f"Received signal {signum}; will finish the current target and retain a clean partial cache.",
        flush=True,
    )


def _publish_timed_wrapper_ready() -> None:
    """Publish readiness for the allocation's TERM-forwarding wrapper."""

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
    print(f"GERAF_B7873200_PREPARER_TIMED_WRAPPER_READY path={ready_path}", flush=True)


def _write_preparation_state(
    cache_root: Path,
    *,
    status: str,
    completed_count: int,
    total_count: int,
    last_completed_view: int | None,
) -> None:
    """Record phase state so only a real clean preparation stop can resume."""

    atomic_write_json(
        cache_root / PREPARATION_STATE_FILENAME,
        build_preparation_state(
            status=status,
            completed_count=int(completed_count),
            total_count=int(total_count),
            last_completed_view=last_completed_view,
            stop_signal=_STOP_SIGNAL,
        ),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", help="RIFT dataset object or registered alias")
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "data/RIFT_dataset")
    parser.add_argument("--npz-path")
    parser.add_argument("--role-manifest")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--scene-extent", type=float, default=0.15)
    parser.add_argument("--n-azimuth", type=int, default=32)
    parser.add_argument("--n-elevation", type=int, default=32)
    parser.add_argument("--n-depth", type=int, default=32)
    parser.add_argument("--aperture-scale", type=float, default=1.0)
    parser.add_argument("--phase-sign", type=float, choices=(-1.0, 1.0), default=-1.0)
    parser.add_argument("--backend", choices=("range",), default="range")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compute-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--point-chunk", type=int, default=4096)
    parser.add_argument("--pair-chunk", type=int, default=32)
    parser.add_argument("--freq-chunk", type=int, default=75)
    parser.add_argument("--kernel-width", type=int, default=20)
    parser.add_argument("--oversample", type=int, default=2)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--max-views",
        type=int,
        default=0,
        help="smoke-only cap over the ordered train then validation roles; zero means all 4,200",
    )
    args = parser.parse_args(argv)
    if args.object is not None:
        from rift.rift_dataset import resolve_object_inputs
        args.npz_path, args.role_manifest = map(str, resolve_object_inputs(
            object_name=args.object, dataset_root=args.dataset_root,
            npz_path=args.npz_path, role_manifest_path=args.role_manifest))
    else:
        args.npz_path = args.npz_path or B787_3200_CANONICAL_NPZ_PATH
        args.role_manifest = args.role_manifest or B787_3200_CANONICAL_MANIFEST_PATH
    return args


def _normalized_path(value: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(os.fspath(value))))


def _validate_args(args: argparse.Namespace) -> None:
    from rift.rift_dataset import collection_manifest
    if (_normalized_path(args.npz_path) != _normalized_path(B787_3200_CANONICAL_NPZ_PATH)
            and not collection_manifest(args.role_manifest)):
        raise ValueError(
            "the corrected GeRaF B7873200 preparer accepts only the canonical "
            f"/storage/home archive: {B787_3200_CANONICAL_NPZ_PATH}"
        )
    finite_positive_float = {
        "scene_extent": args.scene_extent,
        "aperture_scale": args.aperture_scale,
    }
    invalid_float = [
        name
        for name, value in finite_positive_float.items()
        if not math.isfinite(float(value)) or float(value) <= 0.0
    ]
    if invalid_float:
        raise ValueError(f"floating arguments must be finite and positive: {invalid_float}")
    positive_integer = {
        "n_azimuth": args.n_azimuth,
        "n_elevation": args.n_elevation,
        "n_depth": args.n_depth,
        "point_chunk": args.point_chunk,
        "pair_chunk": args.pair_chunk,
        "freq_chunk": args.freq_chunk,
        "kernel_width": args.kernel_width,
        "oversample": args.oversample,
    }
    invalid_integer = [
        name
        for name, value in positive_integer.items()
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0
    ]
    if invalid_integer:
        raise ValueError(f"integer arguments must be positive integers: {invalid_integer}")
    if args.max_views < 0:
        raise ValueError("max_views must be non-negative")
    if args.kernel_width < 4 or args.oversample < 2:
        raise ValueError("range backend requires kernel_width>=4 and oversample>=2")
    if not math.isfinite(float(args.phase_sign)) or float(args.phase_sign) != -1.0:
        raise ValueError("the corrected B7873200 GeRaF recipe requires phase_sign=-1")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA target preparation requested but CUDA is unavailable")


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _write_or_require_equal(path: Path, payload: Mapping[str, Any], label: str) -> None:
    expected = dict(payload)
    if path.exists():
        observed = _read_json_object(path, label)
        if observed != expected:
            raise ValueError(f"{label} disagrees with this B7873200 recipe; use a new cache root")
        return
    atomic_write_json(path, expected)


def _view_path(cache_root: Path, view_index: int) -> Path:
    return cache_root / VIEWS_DIRECTORY / f"view_{int(view_index):06d}.npz"


def _identity_roles(identity: Mapping[str, Any], role: str) -> tuple[int, ...]:
    role_ids = identity.get("role_ids")
    if not isinstance(role_ids, Mapping):
        raise ValueError("sealed B7873200 source identity lacks role_ids")
    values = role_ids.get(role)
    if not isinstance(values, list) or not all(isinstance(value, int) for value in values):
        raise ValueError(f"sealed B7873200 source identity role {role!r} is malformed")
    return tuple(int(value) for value in values)


def _target_spec(args: argparse.Namespace) -> dict[str, Any]:
    """The direct, hash-free semantics which each selected target repeats."""

    return {
        "native_readout": "complex magnitude |MF|",
        "native_readout_detail": "GeRaF Eq. 3 / Algorithm 1; not squared",
        "phase_sign": float(args.phase_sign),
        "backend": str(args.backend),
        "compute_dtype": str(args.compute_dtype),
        "grid": {
            "kind": "calibrated_parallel_primary_rays",
            "scene_center_m": [0.0, 0.0, 0.0],
            "scene_extent_m": float(args.scene_extent),
            "n_azimuth": int(args.n_azimuth),
            "n_elevation": int(args.n_elevation),
            "n_depth": int(args.n_depth),
            "aperture": "calibrated virtual MIMO phase centres (tx+rx)/2",
            "aperture_scale": float(args.aperture_scale),
            "axis_order_3d": ["elevation aperture offset", "azimuth aperture offset", "depth"],
            "geometry_dtype": "float32",
        },
        "operator": {
            "implementation": "range_nufft" if args.backend == "range" else "direct",
            "range_model": "none",
            "include_four_pi": False,
            "kernel_width": int(args.kernel_width) if args.backend == "range" else None,
            "oversample": int(args.oversample) if args.backend == "range" else None,
            "point_chunk": int(args.point_chunk),
            "pair_chunk": int(args.pair_chunk),
            "freq_chunk": int(args.freq_chunk) if args.backend == "exact" else None,
        },
    }


def _recipe(identity: Mapping[str, Any], target_spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": CACHE_SCHEMA,
        "version": 1,
        "sealed_protocol_identity": dict(identity),
        "response_roles_materialized": ["train", "validation"],
        "acquisition_record": {
            "schema": B787_3200_ACQUISITION_SCHEMA,
            "filename": B787_3200_CACHE_ACQUISITION_FILENAME,
        },
        "target_spec": dict(target_spec),
    }


def _scalar_text(values: np.ndarray) -> str:
    result = np.asarray(values).reshape(())
    return str(result.item())


def _target_spec_text(target_spec: Mapping[str, Any]) -> str:
    return json.dumps(dict(target_spec), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_target(
    path: Path,
    *,
    index: int,
    role: str,
    target_spec: Mapping[str, Any],
    expected_shape: tuple[int, int, int],
) -> np.ndarray:
    """Open one authorized cache entry; never hash it or touch raw radar rows."""

    if not path.is_file():
        raise FileNotFoundError(f"missing B7873200 GeRaF target {path}")
    with np.load(path, allow_pickle=False) as target:
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
        absent = required.difference(target.files)
        if absent:
            raise ValueError(f"target {path} is missing {sorted(absent)}")
        if _scalar_text(target["schema"]) != TARGET_SCHEMA:
            raise ValueError(f"target {path} has an incompatible schema")
        if _scalar_text(target["target_spec_json"]) != _target_spec_text(target_spec):
            raise ValueError(f"target {path} has a different native target recipe")
        if int(np.asarray(target["view_index"]).reshape(())) != int(index):
            raise ValueError(f"target filename/index mismatch for {path}")
        if _scalar_text(target["role"]) != role:
            raise ValueError(f"target {path} has an incompatible role")
        magnitude = np.asarray(target["geraf_mf_magnitude"])
        if magnitude.shape != expected_shape or magnitude.dtype != np.dtype(np.float32):
            raise ValueError(f"target {path} has invalid |MF| shape or dtype")
        if not np.isfinite(magnitude).all() or np.any(magnitude < 0):
            raise ValueError(f"target {path} has invalid |MF| values")
        geometry_shapes = {
            "viewpoint_position": (3,),
            "primary_direction": (3,),
            "azimuth_axis": (3,),
            "elevation_axis": (3,),
            "depth_m": (expected_shape[2],),
            "azimuth_offsets_m": (expected_shape[1],),
            "elevation_offsets_m": (expected_shape[0],),
        }
        for name, shape in geometry_shapes.items():
            values = np.asarray(target[name])
            if values.shape != shape or values.dtype != np.dtype(np.float32):
                raise ValueError(f"target {path} has invalid frozen geometry {name}")
            if not np.isfinite(values).all():
                raise ValueError(f"target {path} has non-finite frozen geometry {name}")
        return magnitude.copy()


def _matched_filter_amplitude(
    response_tx_rx_freq: np.ndarray,
    frequencies_hz: torch.Tensor,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    query_points: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Historical native GeRaF Eq. 3 target: coherent complex MF then magnitude."""

    common = {
        "phase_sign": args.phase_sign,
        "response_layout": "tx_rx_freq",
        "range_model": "none",
        "include_four_pi": False,
        "backend": "range_nufft" if args.backend == "range" else "direct",
        "point_chunk": args.point_chunk,
        "pair_chunk": args.pair_chunk,
        "compute_dtype": torch.float64 if args.compute_dtype == "float64" else torch.float32,
    }
    if args.backend == "range":
        common.update(
            nufft_kernel_width=args.kernel_width,
            nufft_oversample=args.oversample,
        )
    else:
        common.update(freq_chunk=args.freq_chunk)
    return matched_filter_complex(
        torch.as_tensor(response_tx_rx_freq, device=query_points.device),
        tx_positions,
        rx_positions,
        frequencies_hz,
        query_points,
        **common,
    )


def _reject_legacy_or_sealed_targets(cache_root: Path, identity: Mapping[str, Any]) -> None:
    """Keep the corrected lane separate from legacy roots and sealed response roles."""

    legacy = [
        cache_root / "target_contract.json",
        cache_root / "power_stats.json",
        cache_root / "target_manifest.json",
        cache_root / "views",
    ]
    present_legacy = [path for path in legacy if path.exists()]
    if present_legacy:
        raise ValueError(
            "refusing to mutate a historical GeRaF/RadarSplat cache root; "
            f"found {present_legacy[0]}"
        )
    for role in ("reserved_test", "unused"):
        for index in _identity_roles(identity, role):
            if _view_path(cache_root, index).exists():
                raise ValueError(
                    f"corrected cache contains a forbidden {role} response-derived target: "
                    f"{_view_path(cache_root, index)}"
                )


def _prepare_one(
    source: Any,
    index: int,
    role: str,
    *,
    cache_root: Path,
    target_spec: Mapping[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    frequencies: torch.Tensor,
) -> None:
    arrays = source.arrays
    tx = torch.as_tensor(arrays.tx_pos[index], dtype=torch.float32, device=device)
    rx = torch.as_tensor(arrays.rx_pos[index], dtype=torch.float32, device=device)
    grid = build_lensless_grid(
        arrays.viewpoint_positions[index],
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
    # This is the only raw-response read in this script.  The adapter validates
    # the role before reading and is restricted to train plus validation.
    response = source.response_view(index)
    if np.asarray(response).shape != (16, 16, 600):
        raise ValueError(
            "sealed B7873200 response adapter must return one chirp-averaged "
            "[Tx,Rx,F] = [16,16,600] view"
        )
    with torch.inference_mode():
        amplitude = _matched_filter_amplitude(
            response,
            frequencies,
            tx,
            rx,
            grid.flat_points,
            args,
        )
        magnitude = amplitude.abs().reshape(grid.shape)
    if (
        not bool(torch.isfinite(magnitude).all())
        or float(magnitude.min()) < 0.0
    ):
        raise RuntimeError(f"view {index} produced invalid native GeRaF |MF|")
    atomic_save_target(
        _view_path(cache_root, index),
        schema=np.asarray(TARGET_SCHEMA),
        target_spec_json=np.asarray(_target_spec_text(target_spec)),
        view_index=np.asarray(index, dtype=np.int64),
        role=np.asarray(role),
        geraf_mf_magnitude=magnitude.detach().to(torch.float32).cpu().numpy(),
        viewpoint_position=np.asarray(arrays.viewpoint_positions[index], dtype=np.float32),
        primary_direction=grid.primary_direction.detach().cpu().numpy(),
        azimuth_axis=grid.azimuth_axis.detach().cpu().numpy(),
        elevation_axis=grid.elevation_axis.detach().cpu().numpy(),
        depth_m=grid.depth_m.detach().cpu().numpy(),
        azimuth_offsets_m=grid.azimuth_offsets_m.detach().cpu().numpy(),
        elevation_offsets_m=grid.elevation_offsets_m.detach().cpu().numpy(),
    )


def _finalize(
    cache_root: Path,
    *,
    identity: Mapping[str, Any],
    target_spec: Mapping[str, Any],
    train: Sequence[int],
    validation: Sequence[int],
    expected_shape: tuple[int, int, int],
) -> None:
    """Write publication-free cache metadata only after all allowed targets validate."""

    peak = 0.0
    for role, indices in (("train", train), ("validation", validation)):
        for index in indices:
            magnitude = _validate_target(
                _view_path(cache_root, index),
                index=index,
                role=role,
                target_spec=target_spec,
                expected_shape=expected_shape,
            )
            if role == "train":
                peak = max(peak, float(np.max(magnitude)))
    if not math.isfinite(peak) or peak <= 0.0:
        raise RuntimeError("training-role native GeRaF |MF| peak must be finite and positive")
    manifest = {
        "schema": CACHE_SCHEMA,
        "version": 1,
        "kind": "native_mf_targets",
        "roles": {
            "train": [int(index) for index in train],
            "validation": [int(index) for index in validation],
        },
    }
    stats = {
        "schema": CACHE_SCHEMA,
        "version": 1,
        "kind": "train_normalization",
        "fit_split": "train",
        "geraf_mf_magnitude_peak": peak,
        "clip": False,
    }
    if "dataset_identity" in identity:
        stats["dataset_identity"] = dict(identity["dataset_identity"])
    _write_or_require_equal(cache_root / MANIFEST_FILENAME, manifest, "corrected target manifest")
    _write_or_require_equal(cache_root / STATS_FILENAME, stats, "corrected target statistics")
    if (
        _read_json_object(cache_root / MANIFEST_FILENAME, "corrected target manifest") != manifest
        or _read_json_object(cache_root / STATS_FILENAME, "corrected target statistics") != stats
    ):
        raise AssertionError("corrected cache finalization did not persist its direct metadata")
    # The sidecar represents only the sealed role policy.  It must not exist
    # for a partial cache, because a partial cache is not trainable.
    write_b7873200_target_cache_protocol(cache_root, identity)


def main(argv: Sequence[str] | None = None) -> None:
    global _STOP_REQUESTED, _STOP_SIGNAL
    _STOP_REQUESTED = False
    _STOP_SIGNAL = None
    args = parse_args(argv)
    _validate_args(args)
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    _publish_timed_wrapper_ready()
    device = torch.device(args.device)

    # The compatibility factory has a hard ordering guarantee: role-manifest
    # and NPZ metadata/header validation happen before this script receives a
    # source object capable of returning a radar response row.
    from rift.rift_dataset import (collection_manifest, load_object_contract,
                                   object_identity, validate_checkpoint_object)
    if args.object is not None or collection_manifest(args.role_manifest):
        _, contract = load_object_contract(args.npz_path, args.role_manifest)
        if args.object is not None:
            validate_checkpoint_object(object_identity(args.object), contract)
    source = load_b7873200_development_source(args.npz_path, args.role_manifest)
    identity = source.identity
    train = _identity_roles(identity, "train")
    validation = _identity_roles(identity, "validation")
    authorized = train + validation
    if not train or not validation or len(authorized) != len(set(authorized)):
        raise ValueError("sealed B7873200 source has invalid development roles")

    arrays = source.arrays
    if tuple(identity.get("response_shape", ())) != (10_000, 16, 16, 1, 600):
        raise ValueError("sealed B7873200 source adapter exposed an unexpected response header")
    cache_root = Path(args.cache_root).expanduser().resolve()
    if not args.resume and cache_root.exists() and any(cache_root.iterdir()):
        raise FileExistsError("--no-resume requires a new empty corrected cache root")
    cache_root.mkdir(parents=True, exist_ok=True)
    _reject_legacy_or_sealed_targets(cache_root, identity)

    target_spec = _target_spec(args)
    recipe = _recipe(identity, target_spec)
    _write_or_require_equal(cache_root / RECIPE_FILENAME, recipe, "corrected target recipe")
    acquisition_record = write_or_validate_b7873200_acquisition_record(cache_root, arrays)
    operator_frequency_hz = frequency_grid_hz(arrays.metadata)
    frequencies = torch.as_tensor(
        validate_b7873200_operator_frequency_grid(acquisition_record, operator_frequency_hz),
        device=device,
    )
    expected_shape = (args.n_elevation, args.n_azimuth, args.n_depth)
    role_for_index = {index: "train" for index in train}
    role_for_index.update({index: "validation" for index in validation})
    selected = list(authorized)
    if args.max_views:
        selected = selected[: args.max_views]
    completed_count = 0
    last_completed_view: int | None = None
    _write_preparation_state(
        cache_root,
        status="running",
        completed_count=completed_count,
        total_count=len(authorized),
        last_completed_view=None,
    )
    print(
        f"Preparing {len(selected)} sealed development targets on {device}: "
        f"GeRaF={args.n_elevation}x{args.n_azimuth}x{args.n_depth}, backend={args.backend}",
        flush=True,
    )
    for number, index in enumerate(selected, start=1):
        if _STOP_REQUESTED:
            break
        destination = _view_path(cache_root, index)
        if destination.exists():
            if not args.resume:
                raise FileExistsError(f"--no-resume refuses existing target {destination}")
            _validate_target(
                destination,
                index=index,
                role=role_for_index[index],
                target_spec=target_spec,
                expected_shape=expected_shape,
            )
            completed_count += 1
            last_completed_view = int(index)
            _write_preparation_state(
                cache_root,
                status="running",
                completed_count=completed_count,
                total_count=len(authorized),
                last_completed_view=last_completed_view,
            )
            print(f"[{number}/{len(selected)}] view {index}: cached", flush=True)
            continue
        started = time.perf_counter()
        _prepare_one(
            source,
            index,
            role_for_index[index],
            cache_root=cache_root,
            target_spec=target_spec,
            args=args,
            device=device,
            frequencies=frequencies,
        )
        completed_count += 1
        last_completed_view = int(index)
        _write_preparation_state(
            cache_root,
            status="running",
            completed_count=completed_count,
            total_count=len(authorized),
            last_completed_view=last_completed_view,
        )
        print(f"[{number}/{len(selected)}] view {index}: {time.perf_counter() - started:.2f}s", flush=True)

    missing = [index for index in authorized if not _view_path(cache_root, index).is_file()]
    if _STOP_REQUESTED:
        if missing:
            _write_preparation_state(
                cache_root,
                status="partial_clean_stop",
                completed_count=len(authorized) - len(missing),
                total_count=len(authorized),
                last_completed_view=last_completed_view,
            )
            print(
                f"Clean preparation stop with {len(missing)} targets remaining; "
                "resume the explicit preparation phase before fitting.",
                flush=True,
            )
            raise SystemExit(CLEAN_STOP_EXIT_CODE)
        _reject_legacy_or_sealed_targets(cache_root, identity)
        validate_b7873200_acquisition_record(cache_root, arrays)
        _finalize(
            cache_root,
            identity=identity,
            target_spec=target_spec,
            train=train,
            validation=validation,
            expected_shape=expected_shape,
        )
        _write_preparation_state(
            cache_root,
            status="complete_clean_stop_before_fit",
            completed_count=len(authorized),
            total_count=len(authorized),
            last_completed_view=last_completed_view,
        )
        print("Clean preparation stop after complete cache finalization; fit was not started.", flush=True)
        raise SystemExit(CLEAN_STOP_EXIT_CODE)
    if missing:
        _write_preparation_state(
            cache_root,
            status="partial",
            completed_count=len(authorized) - len(missing),
            total_count=len(authorized),
            last_completed_view=last_completed_view,
        )
        print(
            f"Corrected train/validation cache is partial ({len(missing)} targets missing); "
            "manifest, statistics, and sealed sidecar were not written.",
            flush=True,
        )
        return
    _reject_legacy_or_sealed_targets(cache_root, identity)
    validate_b7873200_acquisition_record(cache_root, arrays)
    completion_status, completion_exit = complete_preparation_phase(
        lambda: _finalize(
            cache_root,
            identity=identity,
            target_spec=target_spec,
            train=train,
            validation=validation,
            expected_shape=expected_shape,
        ),
        lambda: _STOP_REQUESTED,
    )
    _write_preparation_state(
        cache_root,
        status=completion_status,
        completed_count=len(authorized),
        total_count=len(authorized),
        last_completed_view=last_completed_view,
    )
    if completion_exit is not None:
        print("Clean preparation stop after complete cache finalization; fit was not started.", flush=True)
        raise SystemExit(completion_exit)
    print("Completed sealed B7873200 native GeRaF target cache.", flush=True)


if __name__ == "__main__":
    main()
