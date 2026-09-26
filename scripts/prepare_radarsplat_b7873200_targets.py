#!/usr/bin/env python
"""Prepare sealed train/validation native-power targets for RadarSplat B7873200.

Only this preparation entrypoint opens raw B787 responses.  It binds the
canonical sphere10k archive to its frozen interpolation manifest *before*
reading a response, then exposes exactly the 3,200 training and 1,000
validation rows.  It never opens a reserved-test or unused response.

The target is intentionally the RadarSplat-native observable,
``sum_elevation(abs(matched_filter_complex)**2)``, on a local polar grid.  It
does not write, infer, or retain a coherent phase target.
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

from rift.matched_filter_power import matched_filter_complex
from rift.power_baseline_dataset import (
    build_radarsplat_target_grid,
    frequency_grid_hz,
    load_b787_power_arrays,
)
from rift.radarsplat_b7873200_acquisition import acquisition_payload, write_acquisition_record
from rift.radarsplat_b7873200_adapter import (
    OCCUPANCY_THRESHOLD,
    native_power_from_matched_filter,
    normalize_power,
    target_concentration,
)
from rift.radarsplat_b7873200_protocol import (
    B787_3200_CANONICAL_MANIFEST_PATH,
    B787_3200_CANONICAL_NPZ_PATH,
    CACHE_SCHEMA,
    CURRENT_OCCUPANCY_THRESHOLD,
    MANIFEST_FILENAME,
    RECIPE_FILENAME,
    STATS_FILENAME,
    TARGET_SCHEMA,
    atomic_save_npz,
    atomic_write_json,
    expected_cache_recipe,
    load_b7873200_sealed_identity,
    load_target,
    target_path,
)


_PREPARE_DEVICE: torch.device | None = None
_PREPARE_MATERIALIZED = 0
_PREPARE_REUSED = 0


def _process_peak_rss_kib() -> int | None:
    """Return Linux process high-water RSS for allocated-node resource evidence."""

    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) == 3 and fields[0] == "VmHWM:" and fields[2] == "kB":
                value = int(fields[1])
                return value if value > 0 else None
    except (OSError, ValueError):
        return None
    return None


def _emit_prepare_resource(
    *, device: torch.device, materialized: int, reused: int, phase: str = "prepare"
) -> None:
    """Emit post-success phase telemetry; it does not alter cache semantics."""

    payload: dict[str, object] = {
        "phase": phase,
        "targets_newly_materialized": int(materialized),
        "targets_reused": int(reused),
        "process_peak_rss_kib": int(_process_peak_rss_kib() or 0),
    }
    if device.type == "cuda":
        payload["cuda_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        payload["cuda_max_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    print("RADARSPLAT_B7873200_PREPARE_RESOURCE_JSON=" + json.dumps(payload, sort_keys=True), flush=True)


def _request_prepare_stop(signum: int, _frame: object) -> None:
    """Record an atomic-cache clean stop so an explicit resume is auditable."""

    device = _PREPARE_DEVICE or torch.device("cpu")
    _emit_prepare_resource(
        device=device,
        materialized=_PREPARE_MATERIALIZED,
        reused=_PREPARE_REUSED,
        phase="prepare_clean_interruption",
    )
    print(
        "RADARSPLAT_B7873200_PREPARE_CLEAN_STOP_RETAINED "
        f"signal={int(signum)} materialized={_PREPARE_MATERIALIZED} reused={_PREPARE_REUSED}",
        flush=True,
    )
    raise SystemExit(143)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", help="RIFT dataset object or registered alias")
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parents[1] / "data/RIFT_dataset")
    parser.add_argument("--npz-path")
    parser.add_argument("--role-manifest")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backend", choices=("direct", "range_nufft"), default="range_nufft")
    parser.add_argument("--compute-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--point-chunk", type=int, default=1024)
    parser.add_argument("--pair-chunk", type=int, default=64)
    parser.add_argument("--freq-chunk", type=int, default=64)
    parser.add_argument("--nufft-oversample", type=int, default=2)
    parser.add_argument("--nufft-kernel-width", type=int, default=20)
    parser.add_argument("--scene-extent-m", type=float, default=0.15)
    parser.add_argument("--n-azimuth", type=int, default=32)
    parser.add_argument("--n-elevation", type=int, default=32)
    parser.add_argument("--n-range", type=int, default=32)
    parser.add_argument("--output-azimuth-resolution-deg", type=float, default=0.9)
    parser.add_argument("--elevation-sampling-resolution-deg", type=float, default=0.9)
    parser.add_argument("--intermediate-azimuth-resolution-deg", type=float, default=0.09)
    parser.add_argument("--grid-policy", choices=("legacy", "scene_support"), default="legacy",
                        help="scene_support derives angular spacing from the scene cube and calibrated standoff; Q=10")
    parser.add_argument("--occupancy-threshold", type=float, default=OCCUPANCY_THRESHOLD)
    parser.add_argument(
        "--max-train",
        type=int,
        default=0,
        help="optional frozen-prefix engineering subset; zero prepares all 3,200 training targets",
    )
    parser.add_argument(
        "--max-validation",
        type=int,
        default=0,
        help="optional frozen-prefix engineering subset; zero prepares all 1,000 validation targets",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
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


def _validate_args(args: argparse.Namespace) -> torch.device:
    if min(args.point_chunk, args.pair_chunk, args.freq_chunk, args.nufft_oversample, args.nufft_kernel_width) < 1:
        raise ValueError("RadarSplat B7873200 matched-filter chunk and NUFFT settings must be positive")
    if min(args.n_azimuth, args.n_elevation, args.n_range) < 2:
        raise ValueError("RadarSplat B7873200 target dimensions must each be at least two")
    if min(
        args.scene_extent_m,
        args.output_azimuth_resolution_deg,
        args.elevation_sampling_resolution_deg,
        args.intermediate_azimuth_resolution_deg,
    ) <= 0.0:
        raise ValueError("RadarSplat B7873200 target dimensions and resolutions must be positive")
    ratio = args.output_azimuth_resolution_deg / args.intermediate_azimuth_resolution_deg
    if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("RadarSplat B7873200 output/intermediate azimuth ratio must be integral")
    if not 0.0 < args.occupancy_threshold <= 1.0:
        raise ValueError("RadarSplat B7873200 occupancy threshold must lie in (0,1]")
    if not math.isclose(args.occupancy_threshold, CURRENT_OCCUPANCY_THRESHOLD, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("RadarSplat B7873200 target preparation fixes occupancy threshold at 0.001")
    if args.max_train < 0 or args.max_validation < 0:
        raise ValueError("RadarSplat B7873200 subset limits must be non-negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("RadarSplat B7873200 target preparation requested CUDA, but CUDA is unavailable")
    return device


def _target_spec(args: argparse.Namespace, beamwidth_deg: float, leakage_width_m: float) -> dict[str, object]:
    return {
        "grid": {
            "scene_extent_m": float(args.scene_extent_m),
            "scene_center_m": [0.0, 0.0, 0.0],
            "azimuth_center_deg": 0.0,
            "n_azimuth": int(args.n_azimuth),
            "n_elevation": int(args.n_elevation),
            "n_range": int(args.n_range),
            "output_azimuth_resolution_deg": float(args.output_azimuth_resolution_deg),
            "elevation_sampling_resolution_deg": float(args.elevation_sampling_resolution_deg),
            "intermediate_azimuth_resolution_deg": float(args.intermediate_azimuth_resolution_deg),
            "azimuth_beamwidth_deg": float(beamwidth_deg),
            "spectral_leakage_width_m": float(leakage_width_m),
        },
        "matched_filter": {
            "phase_sign": -1.0,
            "response_layout": "tx_rx_freq",
            "range_model": "none",
            "include_four_pi": False,
            "backend": str(args.backend),
            "compute_dtype": str(args.compute_dtype),
            "point_chunk": int(args.point_chunk),
            "pair_chunk": int(args.pair_chunk),
            "freq_chunk": int(args.freq_chunk),
            "nufft_oversample": int(args.nufft_oversample),
            "nufft_kernel_width": int(args.nufft_kernel_width),
        },
        "occupancy_threshold": float(args.occupancy_threshold),
    }


def _write_or_compare_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            observed = json.load(handle)
        if observed != payload:
            raise ValueError(f"RadarSplat B7873200 existing metadata disagrees with requested semantics: {path}")
        return
    atomic_write_json(path, payload)


def _authorized_roles(materialized: Mapping[str, object]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    roles = materialized
    return (
        ("train", tuple(int(value) for value in roles["train"])),
        ("validation", tuple(int(value) for value in roles["validation"])),
    )


def _target_from_response(
    response: np.ndarray,
    frequencies: torch.Tensor,
    tx: torch.Tensor,
    rx: torch.Tensor,
    grid,
    args: argparse.Namespace,
) -> torch.Tensor:
    compute_dtype = torch.float64 if args.compute_dtype == "float64" else torch.float32
    amplitude = matched_filter_complex(
        torch.as_tensor(response, device=grid.flat_points.device),
        tx,
        rx,
        frequencies,
        grid.flat_points,
        phase_sign=-1.0,
        response_layout="tx_rx_freq",
        range_model="none",
        include_four_pi=False,
        backend=args.backend,
        point_chunk=args.point_chunk,
        pair_chunk=args.pair_chunk,
        freq_chunk=args.freq_chunk,
        compute_dtype=compute_dtype,
        nufft_oversample=args.nufft_oversample,
        nufft_kernel_width=args.nufft_kernel_width,
    )
    return native_power_from_matched_filter(
        amplitude,
        n_elevation=grid.shape[0],
        n_azimuth=grid.shape[1],
        n_range=grid.shape[2],
    )


def _stats_from_complete_cache(cache_root: Path, recipe: Mapping[str, object], materialized: Mapping[str, object], threshold: float) -> dict[str, object]:
    roles = _authorized_roles(materialized)
    target_spec = recipe["target_spec"]
    assert isinstance(target_spec, Mapping)
    grid = target_spec["grid"]
    assert isinstance(grid, Mapping)
    train_peak = 0.0
    for role, indices in roles:
        for index in indices:
            arrays = load_target(cache_root, index, role, expected_grid=grid)
            if role == "train":
                train_peak = max(train_peak, float(np.max(arrays["radarsplat_mf_power"])))
    if not math.isfinite(train_peak) or train_peak <= 0.0:
        raise RuntimeError("RadarSplat B7873200 training targets have no positive native-power peak")
    concentration_rows: list[dict[str, float | int]] = []
    for _role, indices in roles:
        for index in indices:
            arrays = load_target(cache_root, index, _role, expected_grid=grid)
            normalized = normalize_power(
                torch.as_tensor(arrays["radarsplat_mf_power"], dtype=torch.float64), train_peak
            )
            concentration_rows.append(target_concentration(normalized, threshold))
    keys = ("positive_fraction", "nonzero_fraction", "effective_support_bins", "peak_to_mean", "azimuth_std_bins", "range_std_bins")
    summary = {
        key: {
            "mean": float(np.mean([float(row[key]) for row in concentration_rows])),
            "minimum": float(np.min([float(row[key]) for row in concentration_rows])),
            "maximum": float(np.max([float(row[key]) for row in concentration_rows])),
        }
        for key in keys
    }
    return {
        "schema": CACHE_SCHEMA,
        "version": 1,
        "fit_split": "train",
        **({"dataset_identity": dict(recipe["sealed_protocol_identity"]["dataset_identity"])}
           if "dataset_identity" in recipe.get("sealed_protocol_identity", {}) else {}),
        **({"acquisition_identity": {key: recipe["sealed_protocol_identity"][key] for key in
             ("antenna_selection", "source_geometry_sha256", "source_response_shape")}}
           if recipe["sealed_protocol_identity"].get("antenna_selection") else {}),
        "normalization": "linear_peak",
        "clip": False,
        "train_peak_power": train_peak,
        "occupancy_threshold": float(threshold),
        "target_concentration_note": "descriptive only; it does not change the 32x32 target resolution",
        "target_concentration": summary,
    }


def main(argv: Sequence[str] | None = None) -> None:
    global _PREPARE_DEVICE, _PREPARE_MATERIALIZED, _PREPARE_REUSED
    args = parse_args(argv)
    device = _validate_args(args)
    _PREPARE_DEVICE = device
    _PREPARE_MATERIALIZED = 0
    _PREPARE_REUSED = 0
    signal.signal(signal.SIGTERM, _request_prepare_stop)
    signal.signal(signal.SIGINT, _request_prepare_stop)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    from rift.rift_dataset import (collection_manifest, load_object_contract,
                                   object_identity, validate_checkpoint_object)
    if args.object is not None or collection_manifest(args.role_manifest):
        _, contract = load_object_contract(args.npz_path, args.role_manifest)
        if args.object is not None:
            validate_checkpoint_object(object_identity(args.object), contract)
    identity = load_b7873200_sealed_identity(args.npz_path, args.role_manifest)
    if "dataset_identity" in identity:
        from rift.geraf_b7873200_adapter import load_b7873200_sealed_power_arrays
        arrays, bound_identity = load_b7873200_sealed_power_arrays(args.npz_path, args.role_manifest)
        if bound_identity != identity:
            raise ValueError("RadarSplat RIFT dataset source and target identities disagree")
    else:
        arrays = load_b787_power_arrays(args.npz_path)
    if arrays.response.shape != tuple(identity["response_shape"]) or arrays.response.dtype != np.dtype("complex64"):
        raise ValueError("RadarSplat B7873200 archive does not match the complete canonical acquisition")
    canonical_roles = identity["role_ids"]
    assert isinstance(canonical_roles, Mapping)
    if args.max_train > len(canonical_roles["train"]) or args.max_validation > len(canonical_roles["validation"]):
        raise ValueError("RadarSplat B7873200 subset limit exceeds the corresponding sealed development role")
    materialized = {
        "train": list(canonical_roles["train"][: args.max_train or None]),
        "validation": list(canonical_roles["validation"][: args.max_validation or None]),
    }
    authorized_indices = np.asarray(
        materialized["train"] + materialized["validation"], dtype=np.int64
    )
    frequencies_np = frequency_grid_hz(arrays.metadata)
    wavelength = 299_792_458.0 / float(arrays.metadata["radar_fc_hz"])
    authorized_tx = arrays.tx_pos[authorized_indices]
    authorized_rx = arrays.rx_pos[authorized_indices]
    authorized_viewpoints = arrays.viewpoint_positions[authorized_indices]
    if args.grid_policy == "scene_support":
        from rift.radarsplat_fidelity import scene_support_angular_sampling
        (args.output_azimuth_resolution_deg, args.elevation_sampling_resolution_deg,
         args.intermediate_azimuth_resolution_deg) = scene_support_angular_sampling(
            authorized_viewpoints, half_extent_m=args.scene_extent_m,
            n_azimuth=args.n_azimuth, n_elevation=args.n_elevation)
    if identity.get('antenna_selection'):
        # Explicit source indices may be reordered; aperture is independent of order.
        tx_apertures = np.linalg.norm(authorized_tx[:, :, None] - authorized_tx[:, None, :], axis=-1).max(axis=(1,2))
        rx_apertures = np.linalg.norm(authorized_rx[:, :, None] - authorized_rx[:, None, :], axis=-1).max(axis=(1,2))
    else:
        tx_apertures = np.linalg.norm(authorized_tx[:, -1] - authorized_tx[:, 0], axis=1)
        rx_apertures = np.linalg.norm(authorized_rx[:, -1] - authorized_rx[:, 0], axis=1)
    tx_aperture = float(tx_apertures[0])
    rx_aperture = float(rx_apertures[0])
    aperture = max(tx_aperture, rx_aperture)
    per_view_aperture = np.maximum(tx_apertures, rx_apertures)
    if not np.allclose(per_view_aperture, aperture, rtol=2.0e-5, atol=2.0e-7):
        raise ValueError(
            "RadarSplat B7873200 calibrated array aperture changes across views; "
            "a single native azimuth beamwidth would be invalid"
        )
    single_pair = bool(identity.get('antenna_selection')) and arrays.num_tx == arrays.num_rx == 1
    if single_pair:
        # The array-aperture approximation is undefined with one phase center.
        # Use the collection simulator's documented element power HPBW as an
        # explicit sensor-filter approximation, never an invented array aperture.
        beamwidth_deg = 10.0
    else:
        if not math.isfinite(aperture) or aperture <= 0.0:
            raise ValueError("RadarSplat B7873200 cannot infer a beamwidth from a degenerate calibrated array")
        beamwidth_deg = math.degrees(0.886 * wavelength / aperture)
    leakage_width_m = 2.0 * 299_792_458.0 / float(arrays.metadata["radar_bandwidth_hz"])
    recipe = expected_cache_recipe(
        identity,
        _target_spec(args, beamwidth_deg, leakage_width_m),
        materialized_roles=materialized,
    )
    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    _write_or_compare_json(cache_root / RECIPE_FILENAME, recipe)
    write_acquisition_record(
        cache_root,
        **acquisition_payload(
            view_indices=authorized_indices,
            frequency_hz=frequencies_np,
            viewpoint_positions=authorized_viewpoints,
            tx_pos=authorized_tx,
            rx_pos=authorized_rx,
            scene_center_m=np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
            metadata={**arrays.metadata, **({"rift_antenna_selection": identity["antenna_selection"]}
                      if identity.get("antenna_selection") else {})},
            response_shape=tuple(int(value) for value in arrays.response.shape),
            response_dtype=str(arrays.response.dtype),
        ),
    )
    frequencies = torch.as_tensor(frequencies_np, dtype=torch.float64, device=device)
    target_spec = recipe["target_spec"]
    assert isinstance(target_spec, Mapping)
    grid_spec = target_spec["grid"]
    assert isinstance(grid_spec, Mapping)
    recipe_materialized = recipe["materialized_role_ids"]
    assert isinstance(recipe_materialized, Mapping)
    targets_newly_materialized = 0
    targets_reused = 0
    for role, indices in _authorized_roles(recipe_materialized):
        for position, index in enumerate(indices, start=1):
            destination = target_path(cache_root, index)
            if args.resume and destination.exists():
                load_target(cache_root, index, role, expected_grid=grid_spec)
                targets_reused += 1
                _PREPARE_REUSED = targets_reused
                continue
            # This loop is the only raw response path.  ``indices`` came from
            # the sealed identity and has no test/unused member.
            tx = torch.as_tensor(arrays.tx_pos[index], dtype=torch.float32, device=device)
            rx = torch.as_tensor(arrays.rx_pos[index], dtype=torch.float32, device=device)
            polar_grid = build_radarsplat_target_grid(
                arrays.viewpoint_positions[index],
                tx,
                rx,
                scene_center=grid_spec["scene_center_m"],
                scene_extent_m=float(grid_spec["scene_extent_m"]),
                n_azimuth=int(grid_spec["n_azimuth"]),
                n_elevation=int(grid_spec["n_elevation"]),
                n_range=int(grid_spec["n_range"]),
                output_azimuth_resolution_deg=float(grid_spec["output_azimuth_resolution_deg"]),
                elevation_sampling_resolution_deg=float(grid_spec["elevation_sampling_resolution_deg"]),
                device=device,
                dtype=torch.float32,
            )
            power = _target_from_response(
                arrays.response_view(index), frequencies, tx, rx, polar_grid, args
            )
            atomic_save_npz(
                destination,
                schema=np.asarray(TARGET_SCHEMA),
                view_index=np.asarray(index, dtype=np.int64),
                role=np.asarray(role),
                radarsplat_mf_power=power.detach().to(torch.float32).cpu().numpy(),
                sensor_to_world=polar_grid.sensor_to_world.detach().to(torch.float32).cpu().numpy(),
                range_m=polar_grid.range_m.detach().to(torch.float32).cpu().numpy(),
                azimuth_rad=polar_grid.azimuth_rad.detach().to(torch.float32).cpu().numpy(),
                elevation_rad=polar_grid.elevation_rad.detach().to(torch.float32).cpu().numpy(),
            )
            targets_newly_materialized += 1
            _PREPARE_MATERIALIZED = targets_newly_materialized
            print(f"prepared {role} target {position}/{len(indices)}: view={index}", flush=True)
    roles = recipe_materialized
    manifest = {
        "schema": CACHE_SCHEMA,
        "version": 1,
        "roles": {"train": roles["train"], "validation": roles["validation"]},
    }
    _write_or_compare_json(cache_root / MANIFEST_FILENAME, manifest)
    stats = _stats_from_complete_cache(cache_root, recipe, roles, args.occupancy_threshold)
    _write_or_compare_json(cache_root / STATS_FILENAME, stats)
    print(json.dumps(stats, indent=2, sort_keys=True), flush=True)
    _emit_prepare_resource(
        device=device,
        materialized=targets_newly_materialized,
        reused=targets_reused,
    )
    print("RADARSPLAT_B7873200_TARGET_CACHE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
