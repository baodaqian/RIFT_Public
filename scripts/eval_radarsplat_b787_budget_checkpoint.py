#!/usr/bin/env python
"""Evaluate the accepted-budget RadarSplat B787 checkpoint.

This is an evaluation-only reader for the nonterminal checkpoint produced by
the user-accepted budget stop.  It intentionally does not resume training or
open the shared native target cache for the reserved TEST role.  TEST targets
are constructed once per selected response view and are never persisted.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_radarsplat as trainer
from rift.power_baseline_dataset import build_radarsplat_target_grid, frequency_grid_hz
from rift.radar_fields_dataset import (
    load_radar_fields_npz,
    restrict_radar_fields_response_views,
)
from rift.radarsplat_b7873200 import RadarSplatEffects
from rift.radarsplat_b7873200_acquisition import acquisition_records_equal
from rift.radarsplat_b7873200_adapter import (
    NativeRadarSplatView,
    export_gaussian_occupancy_geometry,
    load_native_view,
    model_from_checkpoint_state,
    normalize_power,
    target_grid_from_arrays,
)
from rift.radarsplat_b7873200_protocol import load_cache
from scripts.eval_b787_range_power import _sealed_role_indices
from scripts.prepare_radarsplat_b7873200_targets import _target_from_response


CHECKPOINT_VERSION = 2
CHECKPOINT_STEP = 15_000
SH_DEGREE_INTERVAL = 600
MAX_SH_DEGREE = 3
GAUSSIAN_CHUNK_SIZE = 64
MAX_RASTER_CANDIDATE_PAIRS = 2_000_000
NATIVE_OBSERVABLE = "sum_elevation(abs(matched_filter_complex)**2) -> [azimuth,range]"
EVALUATION_CONTEXT = "accepted 24h budget checkpoint; nonterminal training state"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--role-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--role", choices=("validation", "test"), required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--max-views", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 1.0 or not number.is_integer():
        raise ValueError(f"{label} must be a positive integer")
    return int(number)


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return number


def _active_sh_degree(step: int) -> int:
    step_value = _positive_int(step, "checkpoint step")
    return min((max(step_value, 1) - 1) // SH_DEGREE_INTERVAL, MAX_SH_DEGREE)


def _candidate_trainer_args(args: argparse.Namespace) -> argparse.Namespace:
    """Recreate only the reviewed full-scale identity, without training setup."""

    values = [
        "--cache-root", str(args.cache_root),
        "--checkpoint-dir", str(Path(args.checkpoint).resolve().parent),
        "--device", str(args.device),
        "--seed", "42", "--steps", "480000", "--validation-every", "600",
        "--checkpoint-every", "100", "--log-every", "1",
        "--init-num-gaussians", "2048", "--init-extent-m", "0.3",
        "--init-scale-m", "0.003", "--init-opacity", "0.1",
        "--init-noise-probability", "0.1", "--sh-degree", "3",
        "--sh-degree-interval", "600", "--means-lr-base", "1.6e-4",
        "--scales-lr", "5.0e-3", "--quaternions-lr", "1.0e-3",
        "--opacity-lr", "5.0e-2", "--noise-probability-lr", "5.0e-2",
        "--sh0-lr", "2.5e-3", "--shn-lr", "2.5e-3",
        "--adam-beta1", "0.9", "--adam-beta2", "0.999", "--adam-eps", "1.0e-15",
        "--ssim-weight", "0.2", "--occupancy-weight", "5.0",
        "--max-size-weight", "100.0", "--opacity-noise-weight", "100.0",
        "--max-scale-m", "0.006", "--occupancy-threshold", "0.001",
        "--prune-opacity", "0.0005", "--prune-every", "100",
        "--gaussian-chunk-size", str(GAUSSIAN_CHUNK_SIZE),
        "--max-raster-candidate-pairs", str(MAX_RASTER_CANDIDATE_PAIRS),
    ]
    return trainer.parse_args(values)


def _verify_checkpoint(
    checkpoint: Mapping[str, object],
    cache,
    args: argparse.Namespace,
) -> tuple[Mapping[str, object], torch.nn.Module, int, Mapping[str, object]]:
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("RadarSplat budget checkpoint version is not v2")
    if checkpoint.get("step") != CHECKPOINT_STEP:
        raise ValueError("RadarSplat budget checkpoint is not the selected step 15000")
    if checkpoint.get("complete") is not False or checkpoint.get("finalization_pending") is not False:
        raise ValueError("RadarSplat budget checkpoint must be explicitly nonterminal")
    saved_identity = checkpoint.get("run_identity")
    if not isinstance(saved_identity, Mapping):
        raise ValueError("RadarSplat budget checkpoint lacks run_identity")
    expected_identity = trainer.run_identity(
        _candidate_trainer_args(args), cache, RadarSplatEffects.b787_clean()
    )
    if saved_identity != expected_identity:
        raise ValueError("RadarSplat budget checkpoint identity disagrees with the reviewed full-scale recipe")
    if saved_identity.get("sealed_protocol_identity") != dict(cache.identity):
        raise ValueError("checkpoint sealed protocol identity disagrees with cache")
    if saved_identity.get("target_recipe") != dict(cache.recipe):
        raise ValueError("checkpoint target recipe disagrees with cache")
    saved_peak = _finite(saved_identity.get("train_peak_power"), "checkpoint train peak", positive=True)
    if not math.isclose(saved_peak, cache.train_peak_power, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("checkpoint TRAIN peak disagrees with cache")
    checkpoint_acquisition = checkpoint.get("acquisition_record")
    if not isinstance(checkpoint_acquisition, Mapping) or not acquisition_records_equal(
        checkpoint_acquisition, cache.acquisition_record
    ):
        raise ValueError("checkpoint acquisition record disagrees with cache")
    renderer = saved_identity.get("renderer")
    if not isinstance(renderer, Mapping):
        raise ValueError("checkpoint identity lacks renderer settings")
    effects = renderer.get("effects")
    if effects != asdict(RadarSplatEffects.b787_clean()):
        raise ValueError("checkpoint renderer effects disagree with the clean B787 preset")
    if renderer.get("range_power_exponent") != 0.0:
        raise ValueError("checkpoint renderer range exponent is not zero")
    if renderer.get("native_observable") != "additive local-polar Gaussian power":
        raise ValueError("checkpoint renderer observable is not native local-polar power")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("RadarSplat budget checkpoint lacks model_state_dict")
    model = model_from_checkpoint_state(state, device=torch.device(args.device), seed=42)
    if model.num_gaussians != 2046 or model.sh_degree != 3:
        raise ValueError("RadarSplat budget checkpoint topology is not the reviewed 2046-row SH-degree-3 state")
    active_sh_degree = _active_sh_degree(int(checkpoint["step"]))
    if active_sh_degree != 3:
        raise ValueError("selected checkpoint does not derive active SH degree 3")
    model.eval()
    return checkpoint, model, active_sh_degree, saved_identity


def _verify_source_metadata(arrays, cache) -> None:
    expected_shape = tuple(int(value) for value in np.asarray(cache.acquisition_record["response_shape"]).tolist())
    if arrays.response_shape != expected_shape:
        raise ValueError("TEST source response header disagrees with cache acquisition record")
    if str(arrays.response_dtype) != str(np.asarray(cache.acquisition_record["response_dtype"]).reshape(()).item()):
        raise ValueError("TEST source response dtype disagrees with cache acquisition record")
    expected_metadata = json.loads(str(np.asarray(cache.acquisition_record["metadata_json"]).reshape(()).item()))
    if arrays.metadata != expected_metadata:
        raise ValueError("TEST source metadata disagrees with cache acquisition record")
    frequencies = frequency_grid_hz(arrays.metadata)
    if not np.array_equal(frequencies, np.asarray(cache.acquisition_record["frequency_hz"], dtype=np.float64)):
        raise ValueError("TEST source frequency geometry disagrees with cache acquisition record")
    record_ids = np.asarray(cache.acquisition_record["view_indices"], dtype=np.int64)
    for position, index in enumerate(record_ids.tolist()):
        if not np.array_equal(np.asarray(arrays.viewpoint_positions[index]), cache.acquisition_record["viewpoint_positions"][position]):
            raise ValueError("source viewpoint calibration disagrees with cache acquisition record")
        if not np.array_equal(np.asarray(arrays.tx_pos[index]), cache.acquisition_record["tx_pos"][position]):
            raise ValueError("source Tx calibration disagrees with cache acquisition record")
        if not np.array_equal(np.asarray(arrays.rx_pos[index]), cache.acquisition_record["rx_pos"][position]):
            raise ValueError("source Rx calibration disagrees with cache acquisition record")


def _target_args(cache) -> argparse.Namespace:
    matched_filter = cache.recipe["target_spec"]["matched_filter"]
    return argparse.Namespace(
        backend=str(matched_filter["backend"]),
        compute_dtype=str(matched_filter["compute_dtype"]),
        point_chunk=int(matched_filter["point_chunk"]),
        pair_chunk=int(matched_filter["pair_chunk"]),
        freq_chunk=int(matched_filter["freq_chunk"]),
        nufft_oversample=int(matched_filter["nufft_oversample"]),
        nufft_kernel_width=int(matched_filter["nufft_kernel_width"]),
    )


def _test_view(arrays, cache, index: int, device: torch.device) -> NativeRadarSplatView:
    grid_spec = cache.grid
    tx = torch.as_tensor(arrays.tx_pos[index], dtype=torch.float32, device=device)
    rx = torch.as_tensor(arrays.rx_pos[index], dtype=torch.float32, device=device)
    viewpoint = np.asarray(arrays.viewpoint_positions[index], dtype=np.float32)
    polar_grid = build_radarsplat_target_grid(
        viewpoint,
        tx,
        rx,
        scene_center=np.asarray(grid_spec["scene_center_m"], dtype=np.float32),
        scene_extent_m=float(grid_spec["scene_extent_m"]),
        n_azimuth=int(grid_spec["n_azimuth"]),
        n_elevation=int(grid_spec["n_elevation"]),
        n_range=int(grid_spec["n_range"]),
        output_azimuth_resolution_deg=float(grid_spec["output_azimuth_resolution_deg"]),
        elevation_sampling_resolution_deg=float(grid_spec["elevation_sampling_resolution_deg"]),
        device=device,
        dtype=torch.float32,
    )
    raw = arrays.response_view(index)
    if raw.ndim != 4:
        raise ValueError("TEST response view must have [Tx,Rx,chirp,freq] shape")
    chirp_averaged = raw.mean(axis=2)
    frequencies = torch.as_tensor(frequency_grid_hz(arrays.metadata), dtype=torch.float64, device=device)
    power = _target_from_response(
        chirp_averaged, frequencies, tx, rx, polar_grid, _target_args(cache)
    )
    target_power = normalize_power(
        power.detach().to(device=device, dtype=torch.float32), cache.train_peak_power
    )
    pose = polar_grid.sensor_to_world.detach().to(device=device, dtype=torch.float32)
    stored_range = polar_grid.range_m.detach().to(dtype=torch.float32).cpu().numpy()
    stored_azimuth = polar_grid.azimuth_rad.detach().to(dtype=torch.float32).cpu().numpy()
    renderer_grid = target_grid_from_arrays(
        range_m=stored_range,
        azimuth_rad=stored_azimuth,
        expected_grid=grid_spec,
    )
    return NativeRadarSplatView(
        view_index=int(index),
        role="test",
        target_power=target_power,
        sensor_to_world=pose,
        grid=renderer_grid,
        elevation_count=int(polar_grid.elevation_rad.numel()),
    )


@torch.inference_mode()
def _render_view(model, view: NativeRadarSplatView, effects: RadarSplatEffects, active_sh_degree: int, device: torch.device) -> tuple[float, float, int]:
    rendered = model.render(
        view.sensor_to_world,
        view.grid,
        effects,
        active_sh_degree=active_sh_degree,
        gaussian_chunk_size=GAUSSIAN_CHUNK_SIZE,
        max_candidate_pairs=MAX_RASTER_CANDIDATE_PAIRS,
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
        raise RuntimeError("RadarSplat renderer returned invalid native power")
    if not bool(torch.isfinite(view.target_power).all()):
        raise RuntimeError("RadarSplat target is non-finite")
    difference = final_power[0] - view.target_power
    if not bool(torch.isfinite(difference).all()):
        raise RuntimeError("RadarSplat residual is non-finite")
    return (
        float(difference.square().sum().detach().cpu()),
        float(view.target_power.square().sum().detach().cpu()),
        int(view.target_power.numel()),
    )


def _aggregate(squared_error: np.ndarray, target_energy: np.ndarray, element_count: np.ndarray, status: np.ndarray) -> dict[str, object]:
    complete = status.astype(bool)
    error = float(np.sum(squared_error[complete], dtype=np.float64))
    energy = float(np.sum(target_energy[complete], dtype=np.float64))
    count = int(np.sum(element_count[complete], dtype=np.int64))
    ratio = error / max(energy, 1.0e-30) if complete.any() else None
    return {
        "views_complete": int(np.count_nonzero(complete)),
        "views_incomplete": int(np.count_nonzero(~complete)),
        "squared_error_native_power": error,
        "target_energy_native_power": energy,
        "native_power_bins": count,
        "relative_mse_native_power": ratio,
        "relative_mse_native_power_percent": None if ratio is None else 100.0 * ratio,
    }


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".tmp.", delete=False) as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _save_progress(path: Path, *, identity: Mapping[str, object], view_indices: np.ndarray, squared_error: np.ndarray, target_energy: np.ndarray, element_count: np.ndarray, status: np.ndarray) -> None:
    identity_path = path.with_name("progress_identity.json")
    _atomic_json(identity_path, identity)
    with tempfile.NamedTemporaryFile(suffix=".npz", dir=path.parent, prefix=path.stem + ".tmp.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez(
            temporary,
            view_indices=view_indices.astype(np.int64, copy=False),
            squared_error=squared_error.astype(np.float64, copy=False),
            target_energy=target_energy.astype(np.float64, copy=False),
            element_count=element_count.astype(np.int64, copy=False),
            status=status.astype(np.uint8, copy=False),
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_progress_identity(observed_identity: Mapping[str, object], identity: Mapping[str, object]) -> None:
    if dict(observed_identity) != dict(identity):
        raise ValueError("existing RadarSplat evaluation progress identity disagrees with this role")


def _load_progress(path: Path, identity: Mapping[str, object], view_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    identity_path = path.with_name("progress_identity.json")
    if not identity_path.is_file() or not path.is_file():
        return (
            np.zeros(view_indices.size, dtype=np.float64),
            np.zeros(view_indices.size, dtype=np.float64),
            np.zeros(view_indices.size, dtype=np.int64),
            np.zeros(view_indices.size, dtype=np.uint8),
        )
    observed_identity = json.loads(identity_path.read_text(encoding="utf-8"))
    _validate_progress_identity(observed_identity, identity)
    with np.load(path, allow_pickle=False) as archive:
        required = {"view_indices", "squared_error", "target_energy", "element_count", "status"}
        if set(archive.files) != required:
            raise ValueError("existing RadarSplat evaluation progress has unexpected fields")
        observed_ids = np.asarray(archive["view_indices"], dtype=np.int64)
        if not np.array_equal(observed_ids, view_indices):
            raise ValueError("existing RadarSplat evaluation progress has different ordered role IDs")
        arrays = (
            np.asarray(archive["squared_error"], dtype=np.float64),
            np.asarray(archive["target_energy"], dtype=np.float64),
            np.asarray(archive["element_count"], dtype=np.int64),
            np.asarray(archive["status"], dtype=np.uint8),
        )
    if any(value.shape != view_indices.shape for value in arrays):
        raise ValueError("existing RadarSplat evaluation progress has invalid array lengths")
    if not np.isfinite(arrays[0]).all() or not np.isfinite(arrays[1]).all() or np.any(arrays[0] < 0) or np.any(arrays[1] < 0):
        raise ValueError("existing RadarSplat evaluation progress has invalid statistics")
    if np.any((arrays[3] != 0) & (arrays[3] != 1)):
        raise ValueError("existing RadarSplat evaluation progress has invalid status values")
    return arrays


def _role_indices(args: argparse.Namespace, cache) -> np.ndarray:
    base = cache.validation_indices if args.role == "validation" else cache.identity["role_ids"]["reserved_test"]
    selected = np.asarray(tuple(int(value) for value in base), dtype=np.int64)
    if args.max_views > 0:
        selected = selected[: int(args.max_views)]
    if selected.size == 0:
        raise ValueError("selected role contains no views")
    return selected


def _progress_identity(args: argparse.Namespace, cache, checkpoint_path: Path, selected_indices: np.ndarray) -> dict[str, object]:
    return {
        "schema": "rift_radarsplat_b787_budget_evaluation_v1",
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_step": CHECKPOINT_STEP,
        "checkpoint_version": CHECKPOINT_VERSION,
        "role": args.role,
        "npz_path": str(Path(args.npz_path).resolve()),
        "role_manifest": str(Path(args.role_manifest).resolve()),
        "cache_root": str(Path(args.cache_root).resolve()),
        "ordered_view_indices": selected_indices.tolist(),
        "observable": NATIVE_OBSERVABLE,
        "train_peak_power": float(cache.train_peak_power),
        "training_complete": False,
    }


def _evaluation_status(aggregate: Mapping[str, object], requested_views: int) -> tuple[str, bool]:
    complete_rows = int(aggregate["views_complete"]) == int(requested_views)
    full = (
        complete_rows
        and int(requested_views) == 1_000
        and int(aggregate["native_power_bins"]) == 1_024_000
        and math.isfinite(float(aggregate["target_energy_native_power"]))
        and float(aggregate["target_energy_native_power"]) > 0.0
    )
    if full:
        return "complete", True
    if complete_rows:
        return "complete_prefix", False
    return "incomplete", False


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.threads < 0 or args.max_views < 0 or args.save_every < 1:
        raise ValueError("threads, max_views, and save_every must be non-negative/positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    if args.threads > 0:
        torch.set_num_threads(int(args.threads))
    cache = load_cache(args.cache_root)
    if cache.is_development_subset or len(cache.train_indices) != 3200 or len(cache.validation_indices) != 1000:
        raise ValueError("accepted-budget readout requires the complete 3200/1000 native cache")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = trainer._load_checkpoint(checkpoint_path, torch.device("cpu"))
    _checkpoint, model, active_sh_degree, saved_identity = _verify_checkpoint(checkpoint, cache, args)
    effects = RadarSplatEffects.b787_clean()
    selected_indices = _role_indices(args, cache)

    arrays = None
    if args.role == "test":
        arrays = load_radar_fields_npz(args.npz_path, load_response=False)
        _verify_source_metadata(arrays, cache)
        role_args = argparse.Namespace(num_train=3200, num_val=1000, seed=42, max_views=int(args.max_views))
        role_checkpoint = {"sealed_npz_protocol_contract": dict(cache.identity)}
        resolved = _sealed_role_indices(role_checkpoint, arrays, role_args, "test")
        if not np.array_equal(resolved, selected_indices):
            raise ValueError("TEST role resolver disagrees with the verified checkpoint/cache identity")
        arrays = restrict_radar_fields_response_views(arrays, resolved)

    out_dir = Path(args.out_dir)
    if out_dir.exists() and not args.resume and any(out_dir.iterdir()):
        raise FileExistsError("--no-resume refuses to overwrite an occupied RadarSplat evaluation directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "per_view_progress.npz"
    identity = _progress_identity(args, cache, checkpoint_path, selected_indices)
    squared_error, target_energy, element_count, status = _load_progress(
        progress_path if args.resume else out_dir / "missing-progress.npz", identity, selected_indices
    )
    geometry_path = out_dir / "selected_checkpoint_gaussian_occupancy_geometry.npz"
    if not geometry_path.exists():
        export_gaussian_occupancy_geometry(
            geometry_path,
            model,
            active_sh_degree=active_sh_degree,
            train_peak_power=cache.train_peak_power,
        )

    started = time.perf_counter()
    for position, index in enumerate(selected_indices.tolist()):
        if status[position] == 1:
            continue
        if args.role == "validation":
            view = load_native_view(cache, int(index), "validation", device)
        else:
            assert arrays is not None
            view = _test_view(arrays, cache, int(index), device)
        error, energy, count = _render_view(model, view, effects, active_sh_degree, device)
        squared_error[position] = error
        target_energy[position] = energy
        element_count[position] = count
        status[position] = 1
        if int(np.count_nonzero(status)) % int(args.save_every) == 0:
            _save_progress(
                progress_path,
                identity=identity,
                view_indices=selected_indices,
                squared_error=squared_error,
                target_energy=target_energy,
                element_count=element_count,
                status=status,
            )
    _save_progress(
        progress_path,
        identity=identity,
        view_indices=selected_indices,
        squared_error=squared_error,
        target_energy=target_energy,
        element_count=element_count,
        status=status,
    )
    aggregate = _aggregate(squared_error, target_energy, element_count, status)
    status_label, full_evaluation = _evaluation_status(aggregate, int(selected_indices.size))
    summary = {
        "schema": "rift_radarsplat_b787_budget_evaluation_v1",
        "status": status_label,
        "evaluation_complete": full_evaluation,
        "evaluation_context": EVALUATION_CONTEXT,
        "training_complete": False,
        "selected_checkpoint": {
            "path": str(checkpoint_path),
            "step": CHECKPOINT_STEP,
            "checkpoint_version": CHECKPOINT_VERSION,
            "complete": False,
            "finalization_pending": False,
            "active_sh_degree": active_sh_degree,
            "gaussian_rows": int(model.num_gaussians),
            "selected_historical_validation_rel_mse": checkpoint.get("best_validation_rel_mse"),
        },
        "role": args.role,
        "ordered_view_indices": selected_indices.tolist(),
        "observable": NATIVE_OBSERVABLE,
        "metric_name": "native_power_rel_mse",
        "metric_domain_note": "RadarSplat native elevation-integrated 2-D power; not a common RIFT/RF per-channel metric",
        "normalization": {"mode": "linear_peak", "fit_split": "train", "train_peak_power": float(cache.train_peak_power)},
        "renderer_effects": asdict(effects),
        "target_recipe": cache.recipe["target_spec"],
        "geometry_export": {"path": str(geometry_path), "label": "selected nonterminal checkpoint Gaussian occupancy; not a watertight surface"},
        "views_requested": int(selected_indices.size),
        "elapsed_seconds": time.perf_counter() - started,
        **aggregate,
    }
    _atomic_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
