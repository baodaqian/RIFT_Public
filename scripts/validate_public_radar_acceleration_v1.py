#!/usr/bin/env python
"""Synthetic CPU release gate for PublicRadar cache and view batching.

The gate dynamically loads both integrated trainers, uses one tiny immutable
directional-SH cache shared by both tracks, and exercises only synthetic 1x1
monostatic data and a 5 x 4 fixed planar scene.  It never opens a dataset,
checkpoint, experiment run, or training artifact.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rift.public_radar_acceleration_validation_v1"
CACHE_DATASET_NAME = "synthetic_public_radar_acceleration_v1"
BATCH_SIZES = (1, 2, 4)
DEGREES = (0, 3)
POINT_CHUNK = 3
PREDICTION_RELATIVE_L2_TOLERANCE = 2.0e-6
GRADIENT_RELATIVE_L2_TOLERANCE = 2.0e-6
CANONICAL_ORDER = np.asarray([5, 1, 5, 3], dtype=np.int64)
TRACKS = {
    "camry": {
        "trainer_file": "public_radar_interpolation_dense_v2.py",
        "scene_key": "camry",
        "cell": "camry_bp400_deg0",
        "propagation_model": "monostatic_far_field_reference",
    },
    "gotcha": {
        "trainer_file": "public_radar_gotcha_full_domain_v1_partial.py",
        "scene_key": "gotcha_p2_full_domain",
        "cell": "gotcha_p2_full_domain_bp400_deg0",
        "propagation_model": "monostatic_near_field_reference",
    },
}


class GateRecorder:
    def __init__(self):
        self.rows = []

    def check(self, name, passed, **details):
        row = {"name": str(name), "passed": bool(passed), **details}
        self.rows.append(row)
        if not row["passed"]:
            raise AssertionError(name)
        print(f"PASS: {name}", flush=True)


def atomic_json(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite validation result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = os.environ.get("SLURM_JOB_ID", "local")
    temporary = path.with_suffix(
        path.suffix + f".{identity}.{os.getpid()}.tmp"
    )
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.link(temporary, path)
    temporary.unlink()


def load_trainer(track_name):
    path = PROJECT_ROOT / "scripts" / TRACKS[track_name]["trainer_file"]
    module_name = f"rift_public_radar_acceleration_gate_{track_name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load trainer spec from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def synthetic_positions():
    return np.asarray(
        [
            [7.0, 4.0, 3.0],
            [7.3, 3.7, 3.2],
            [6.8, 4.4, 2.7],
            [-6.5, 4.6, 3.4],
            [-7.1, -3.8, 2.9],
            [5.9, -5.0, 3.6],
        ],
        dtype=np.float64,
    )


def synthetic_arrays(positions):
    frequency_count = 17
    frequency_step_hz = 12.5e6
    frequency_start_hz = 9.0e9
    frequencies = frequency_start_hz + frequency_step_hz * np.arange(
        frequency_count, dtype=np.float64
    )
    response = np.empty(
        (positions.shape[0], 1, 1, 1, frequency_count), dtype=np.complex64
    )
    frequency_coordinate = np.linspace(-1.0, 1.0, frequency_count)
    for view_index in range(positions.shape[0]):
        magnitude = (
            0.75
            + 0.04 * view_index
            + 0.08 * np.square(frequency_coordinate)
        )
        phase = (
            -0.35
            + 0.11 * view_index
            + (0.45 + 0.03 * view_index) * frequency_coordinate
        )
        response[view_index, 0, 0, 0, :] = (
            magnitude * np.exp(1j * phase)
        ).astype(np.complex64)
    bandwidth_hz = frequency_count * frequency_step_hz
    metadata = {
        "radar_fc_hz": frequency_start_hz + 0.5 * bandwidth_hz,
        "radar_bandwidth_hz": bandwidth_hz,
        "num_adc_samples": frequency_count,
    }
    phase_centers = positions[:, None, :].copy()
    return {
        "meta": metadata,
        "frequencies_hz": frequencies,
        "response": response,
        "viewpoint_positions": positions.copy(),
        "tx_pos": phase_centers.copy(),
        "rx_pos": phase_centers.copy(),
    }


def relative_l2(actual, expected):
    actual = torch.as_tensor(actual).detach().cpu()
    expected = torch.as_tensor(expected).detach().cpu()
    if actual.shape != expected.shape:
        return math.inf
    difference_power = float(
        (actual - expected).abs().to(torch.float64).square().sum()
    )
    expected_power = float(expected.abs().to(torch.float64).square().sum())
    return math.sqrt(difference_power) / max(math.sqrt(expected_power), 1.0e-30)


def finite_json_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def render_gradient_snapshot(model, renderer, items):
    model.zero_grad(set_to_none=True)
    predicted, measured = renderer.raw_prediction_and_measurement_batch(
        model, items
    )
    if (
        predicted.ndim != 4
        or measured.shape != predicted.shape
        or predicted.shape[0] != len(items)
    ):
        raise RuntimeError("aligned trainer renderer returned an invalid shape")
    loss = (predicted - measured).abs().square().mean()
    loss.backward()
    gradients = []
    for parameter in model.parameters():
        if parameter.grad is None:
            raise RuntimeError("scene parameter is missing its full-scene gradient")
        gradients.append(parameter.grad.detach().cpu().clone())
    return {
        "prediction": predicted.detach().cpu(),
        "measurement": measured.detach().cpu(),
        "loss": float(loss.detach()),
        "gradients": gradients,
    }


def render_legacy_scalar_gradient_snapshot(model, renderer, items):
    """Stream the pre-acceleration scalar path over one or more views."""

    items = list(items)
    if not items:
        raise ValueError("legacy scalar renderer requires at least one item")
    model.zero_grad(set_to_none=True)
    predicted_rows = []
    measured_rows = []
    view_losses = []
    for item in items:
        predicted, measured = renderer.raw_prediction_and_measurement(model, item)
        if predicted.ndim != 3 or measured.shape != predicted.shape:
            raise RuntimeError("legacy scalar renderer returned an invalid shape")
        view_loss = (predicted - measured).abs().square().mean()
        (view_loss / len(items)).backward()
        predicted_rows.append(predicted.detach().cpu().unsqueeze(0))
        measured_rows.append(measured.detach().cpu().unsqueeze(0))
        view_losses.append(float(view_loss.detach()))
    gradients = []
    for parameter in model.parameters():
        if parameter.grad is None:
            raise RuntimeError("legacy scalar scene parameter is missing its gradient")
        gradients.append(parameter.grad.detach().cpu().clone())
    return {
        "prediction": torch.cat(predicted_rows, dim=0),
        "measurement": torch.cat(measured_rows, dim=0),
        "loss": sum(view_losses) / len(view_losses),
        "gradients": gradients,
    }


def gradient_relative_l2(actual, expected):
    if len(actual) != len(expected):
        return math.inf
    difference_power = 0.0
    expected_power = 0.0
    for actual_gradient, expected_gradient in zip(actual, expected):
        if actual_gradient.shape != expected_gradient.shape:
            return math.inf
        difference = actual_gradient.to(torch.float64) - expected_gradient.to(
            torch.float64
        )
        difference_power += float(difference.square().sum())
        expected_power += float(expected_gradient.to(torch.float64).square().sum())
    return math.sqrt(difference_power) / max(math.sqrt(expected_power), 1.0e-30)


@contextmanager
def replaced_argv(values):
    original = sys.argv
    sys.argv = list(values)
    try:
        yield
    finally:
        sys.argv = original


class ForbiddenAccess:
    """Sentinel proving a run-level guard returns before experiment inputs."""

    def __getattr__(self, name):
        raise AssertionError(f"required-cache run guard touched stub attribute {name}")

    def __getitem__(self, key):
        raise AssertionError(f"required-cache run guard touched stub key {key!r}")


def parse_and_run_guard_gates(track_name, trainer, temporary_root, recorder):
    config = TRACKS[track_name]
    legacy_namespace = SimpleNamespace()
    if track_name == "camry":
        legacy_acceleration_options = trainer.acceleration_options(
            legacy_namespace
        )
    else:
        legacy_acceleration_options = trainer.validate_acceleration_args(
            legacy_namespace
        )
    recorder.check(
        f"{track_name} old programmatic Namespace gets scalar defaults",
        legacy_acceleration_options == (1, None, False),
        acceleration_options=list(legacy_acceleration_options),
    )
    parse_root = temporary_root / f"{track_name}_parse_default_run_root"
    default_argv = [
        str(PROJECT_ROOT / "scripts" / config["trainer_file"]),
        config["cell"],
        "fresh",
        "--npz-path",
        "unused-synthetic.npz",
        "--run-root",
        str(parse_root),
    ]
    with replaced_argv(default_argv):
        defaults = trainer.parse_args()
    recorder.check(
        f"{track_name} legacy acceleration defaults remain scalar production",
        defaults.view_batch_size == 1
        and defaults.sh_basis_cache is None
        and defaults.require_sh_basis_cache is False
        and defaults.point_chunk == 32768
        and defaults.pair_chunk == 1,
    )

    run_guard_root = temporary_root / f"{track_name}_required_cache_run_guard"
    guard_args = SimpleNamespace(
        require_sh_basis_cache=True,
        sh_basis_cache="declared-but-deliberately-not-loaded",
        view_batch_size=2,
        run_root=str(run_guard_root),
        cell=config["cell"],
        launch_mode="fresh",
    )
    forbidden = ForbiddenAccess()
    try:
        if track_name == "camry":
            trainer.run(
                guard_args,
                config["scene_key"],
                400,
                0,
                forbidden,
                forbidden,
                forbidden,
                forbidden,
                sh_basis_cache=None,
            )
        else:
            trainer.run(
                guard_args,
                config["scene_key"],
                400,
                0,
                forbidden,
                forbidden,
                forbidden,
                forbidden,
                forbidden,
                sh_basis_cache=None,
            )
    except ValueError as exc:
        run_guard_error = str(exc)
    else:
        run_guard_error = None
    recorder.check(
        f"{track_name} required cache fails at run entry before any write",
        run_guard_error == "required SH basis cache was not loaded"
        and not run_guard_root.exists(),
        error=run_guard_error,
    )

    supported_chunks = tuple(trainer.SUPPORTED_POINT_CHUNKS)
    recorder.check(
        f"{track_name} validated point chunks include legacy, 64k, and 128k",
        supported_chunks == (32768, 65536, 131072)
        and all(
            trainer.validate_renderer_chunk_options(
                SimpleNamespace(point_chunk=point_chunk, pair_chunk=1)
            )
            == (point_chunk, 1)
            for point_chunk in supported_chunks
        ),
        supported_point_chunks=list(supported_chunks),
    )

    point_guard_root = temporary_root / f"{track_name}_point_guard"
    invalid_point_argv = default_argv[:-1] + [
        str(point_guard_root),
        "--point-chunk",
        "262144",
    ]
    with replaced_argv(invalid_point_argv), mock.patch.object(
        trainer.V1, "assert_public_source_contract", return_value=None
    ), mock.patch.object(
        trainer.torch.cuda, "is_available", return_value=True
    ), mock.patch.object(
        trainer.torch.cuda, "device_count", return_value=1
    ):
        try:
            trainer.main()
        except ValueError as exc:
            point_guard_error = str(exc)
        else:
            point_guard_error = None
    recorder.check(
        f"{track_name} unsupported point chunk still fails before any write",
        point_guard_error is not None
        and "legacy default is 32768" in point_guard_error
        and not point_guard_root.exists(),
        error=point_guard_error,
    )

    try:
        trainer.validate_renderer_chunk_options(
            SimpleNamespace(point_chunk=32768, pair_chunk=2)
        )
    except ValueError as exc:
        pair_guard_error = str(exc)
    else:
        pair_guard_error = None
    recorder.check(
        f"{track_name} pair-chunk legacy contract remains sealed to one",
        pair_guard_error == "pair_chunk must remain 1",
        error=pair_guard_error,
    )

    try:
        trainer.validate_renderer_chunk_options(
            SimpleNamespace(point_chunk=32768.0, pair_chunk=1.0)
        )
    except ValueError as exc:
        type_guard_error = str(exc)
    else:
        type_guard_error = None
    recorder.check(
        f"{track_name} programmatic chunk values do not coerce floats",
        type_guard_error == "point_chunk and pair_chunk must be integers",
        error=type_guard_error,
    )


def track_cache_and_renderer_gates(
    track_name,
    trainer,
    arrays,
    cache_dir,
    recorder,
):
    config = TRACKS[track_name]
    device = torch.device("cpu")
    cache = trainer.SH_CACHE.PublicRadarSHBasisCache(
        cache_dir,
        arrays["viewpoint_positions"],
        expected_dataset_name=CACHE_DATASET_NAME,
    )
    base_dataset = trainer.PecSphereNPZDataset(arrays, CANONICAL_ORDER)
    indexed_dataset = trainer.SH_CACHE.IndexedPublicRadarDataset(
        base_dataset,
        CANONICAL_ORDER,
        arrays["viewpoint_positions"],
    )
    live_items = [base_dataset[index] for index in range(len(base_dataset))]
    cached_items = [
        indexed_dataset[index] for index in range(len(indexed_dataset))
    ]
    recovered_order = trainer.SH_CACHE.item_view_indices(cached_items)
    repeated_basis = cache.basis_rows(
        CANONICAL_ORDER, 6, device, dtype=torch.float32
    )
    recorder.check(
        f"{track_name} cache items preserve canonical order and repeats",
        np.array_equal(recovered_order, CANONICAL_ORDER)
        and int(recovered_order[0]) == int(recovered_order[2])
        and torch.equal(repeated_basis[:, 0], repeated_basis[:, 2]),
    )

    metadata = {
        "propagation_model": config["propagation_model"],
        "reference_range_m": float(
            trainer.SCENES[config["scene_key"]]["reference_range_m"]
        ),
        "scene_center_m": [0.0, 0.0, 0.0],
    }
    live_renderer = trainer.SerializedRenderer(
        arrays,
        metadata,
        device,
        POINT_CHUNK,
        pair_chunk=1,
    )
    cached_renderer = trainer.SerializedRenderer(
        arrays,
        metadata,
        device,
        POINT_CHUNK,
        pair_chunk=1,
        sh_basis_cache=cache,
    )
    rows = []
    legacy_rows = []
    recorder.check(
        f"{track_name} old-style renderer construction defaults to live scalar SH",
        getattr(live_renderer, "sh_basis_cache", None) is None,
    )
    recorder.check(
        f"{track_name} aligned scene chunk API remains a lazy generator",
        inspect.isgeneratorfunction(
            trainer.PLANAR.FixedPlanarSHScene.scatterer_view_chunks
        ),
    )
    for degree in DEGREES:
        torch.manual_seed(7100 + 100 * int(degree) + (0 if track_name == "camry" else 1))
        model = trainer.PLANAR.FixedPlanarSHScene(
            5,
            4,
            0.4,
            degree,
            device,
            init_scale=0.02,
        )
        with torch.no_grad():
            scalar_prediction, scalar_measurement = (
                cached_renderer.raw_prediction_and_measurement(
                    model, cached_items[0]
                )
            )
        expected_scalar_shape = (arrays["frequencies_hz"].size, 1, 1)
        recorder.check(
            f"{track_name} degree-{degree} cached scalar B=1 shape",
            tuple(scalar_prediction.shape) == expected_scalar_shape
            and tuple(scalar_measurement.shape) == expected_scalar_shape,
        )

        for batch_size in BATCH_SIZES:
            legacy = render_legacy_scalar_gradient_snapshot(
                model, live_renderer, live_items[:batch_size]
            )
            live = render_gradient_snapshot(
                model, live_renderer, live_items[:batch_size]
            )
            cached = render_gradient_snapshot(
                model, cached_renderer, cached_items[:batch_size]
            )
            prediction_error = relative_l2(
                cached["prediction"], live["prediction"]
            )
            gradient_error = gradient_relative_l2(
                cached["gradients"], live["gradients"]
            )
            measurement_equal = torch.equal(
                cached["measurement"], live["measurement"]
            )
            loss_relative_error = abs(cached["loss"] - live["loss"]) / max(
                abs(live["loss"]), 1.0e-30
            )
            legacy_live_prediction_error = relative_l2(
                live["prediction"], legacy["prediction"]
            )
            legacy_live_gradient_error = gradient_relative_l2(
                live["gradients"], legacy["gradients"]
            )
            legacy_live_loss_relative_error = abs(
                live["loss"] - legacy["loss"]
            ) / max(abs(legacy["loss"]), 1.0e-30)
            legacy_live_measurement_equal = torch.equal(
                live["measurement"], legacy["measurement"]
            )
            legacy_cached_prediction_error = relative_l2(
                cached["prediction"], legacy["prediction"]
            )
            legacy_cached_gradient_error = gradient_relative_l2(
                cached["gradients"], legacy["gradients"]
            )
            legacy_cached_loss_relative_error = abs(
                cached["loss"] - legacy["loss"]
            ) / max(abs(legacy["loss"]), 1.0e-30)
            legacy_cached_measurement_equal = torch.equal(
                cached["measurement"], legacy["measurement"]
            )
            recorder.check(
                (
                    f"{track_name} degree-{degree} B={batch_size} legacy scalar "
                    "stream and live aligned forward, measurement, loss, and "
                    "full gradients"
                ),
                legacy_live_measurement_equal
                and math.isfinite(legacy_live_prediction_error)
                and legacy_live_prediction_error
                <= PREDICTION_RELATIVE_L2_TOLERANCE
                and math.isfinite(legacy_live_loss_relative_error)
                and legacy_live_loss_relative_error
                <= PREDICTION_RELATIVE_L2_TOLERANCE
                and math.isfinite(legacy_live_gradient_error)
                and legacy_live_gradient_error
                <= GRADIENT_RELATIVE_L2_TOLERANCE,
                prediction_relative_l2=finite_json_float(
                    legacy_live_prediction_error
                ),
                loss_relative_error=finite_json_float(
                    legacy_live_loss_relative_error
                ),
                gradient_relative_l2=finite_json_float(
                    legacy_live_gradient_error
                ),
            )
            recorder.check(
                (
                    f"{track_name} degree-{degree} B={batch_size} legacy scalar "
                    "stream and cached aligned forward, measurement, loss, and "
                    "full gradients"
                ),
                legacy_cached_measurement_equal
                and math.isfinite(legacy_cached_prediction_error)
                and legacy_cached_prediction_error
                <= PREDICTION_RELATIVE_L2_TOLERANCE
                and math.isfinite(legacy_cached_loss_relative_error)
                and legacy_cached_loss_relative_error
                <= PREDICTION_RELATIVE_L2_TOLERANCE
                and math.isfinite(legacy_cached_gradient_error)
                and legacy_cached_gradient_error
                <= GRADIENT_RELATIVE_L2_TOLERANCE,
                prediction_relative_l2=finite_json_float(
                    legacy_cached_prediction_error
                ),
                loss_relative_error=finite_json_float(
                    legacy_cached_loss_relative_error
                ),
                gradient_relative_l2=finite_json_float(
                    legacy_cached_gradient_error
                ),
            )
            legacy_rows.append(
                {
                    "degree": int(degree),
                    "batch_size": int(batch_size),
                    "live_aligned": {
                        "prediction_relative_l2": finite_json_float(
                            legacy_live_prediction_error
                        ),
                        "loss_relative_error": finite_json_float(
                            legacy_live_loss_relative_error
                        ),
                        "gradient_relative_l2": finite_json_float(
                            legacy_live_gradient_error
                        ),
                        "measurement_exact": bool(
                            legacy_live_measurement_equal
                        ),
                    },
                    "cached_aligned": {
                        "prediction_relative_l2": finite_json_float(
                            legacy_cached_prediction_error
                        ),
                        "loss_relative_error": finite_json_float(
                            legacy_cached_loss_relative_error
                        ),
                        "gradient_relative_l2": finite_json_float(
                            legacy_cached_gradient_error
                        ),
                        "measurement_exact": bool(
                            legacy_cached_measurement_equal
                        ),
                    },
                }
            )
            recorder.check(
                (
                    f"{track_name} degree-{degree} B={batch_size} cached/live "
                    "aligned forward, measurement, and full gradients"
                ),
                measurement_equal
                and math.isfinite(prediction_error)
                and prediction_error <= PREDICTION_RELATIVE_L2_TOLERANCE
                and math.isfinite(loss_relative_error)
                and loss_relative_error <= PREDICTION_RELATIVE_L2_TOLERANCE
                and math.isfinite(gradient_error)
                and gradient_error <= GRADIENT_RELATIVE_L2_TOLERANCE,
                prediction_relative_l2=finite_json_float(prediction_error),
                loss_relative_error=finite_json_float(loss_relative_error),
                gradient_relative_l2=finite_json_float(gradient_error),
            )
            rows.append(
                {
                    "degree": int(degree),
                    "batch_size": int(batch_size),
                    "prediction_relative_l2": finite_json_float(
                        prediction_error
                    ),
                    "loss_relative_error": finite_json_float(
                        loss_relative_error
                    ),
                    "gradient_relative_l2": finite_json_float(gradient_error),
                    "measurement_exact": bool(measurement_equal),
                }
            )
        model.zero_grad(set_to_none=True)
    return {
        "propagation_model": config["propagation_model"],
        "reference_range_m": metadata["reference_range_m"],
        "cache_contract": cache.contract(),
        "canonical_order_with_repeat": CANONICAL_ORDER.tolist(),
        "legacy_scalar_stream_vs_aligned": legacy_rows,
        "comparisons": rows,
    }


def run_validation(recorder):
    trainers = {name: load_trainer(name) for name in TRACKS}
    positions = synthetic_positions()
    arrays = synthetic_arrays(positions)
    with tempfile.TemporaryDirectory(
        prefix="rift_public_radar_acceleration_v1_"
    ) as directory:
        temporary_root = Path(directory)
        cache_dir = temporary_root / "shared_directional_sh_cache"
        manifest = trainers["camry"].SH_CACHE.materialize_cache(
            cache_dir,
            positions,
            CACHE_DATASET_NAME,
        )
        recorder.check(
            "tiny immutable shared cache materializes through degree 6",
            manifest.get("schema")
            == trainers["camry"].SH_CACHE.SCHEMA
            and int(manifest.get("view_count", -1)) == positions.shape[0]
            and int(manifest.get("max_degree", -1)) == 6,
        )
        for track_name, trainer in trainers.items():
            parse_and_run_guard_gates(
                track_name, trainer, temporary_root, recorder
            )
        tracks = {
            track_name: track_cache_and_renderer_gates(
                track_name,
                trainer,
                arrays,
                cache_dir,
                recorder,
            )
            for track_name, trainer in trainers.items()
        }
    return {
        "cache_manifest": {
            "schema": manifest["schema"],
            "schema_version": manifest["schema_version"],
            "dataset_name": manifest["dataset_name"],
            "view_count": manifest["view_count"],
            "max_degree": manifest["max_degree"],
            "basis_count": manifest["basis_count"],
            "basis_dtype": manifest["basis_dtype"],
            "view_index_space": manifest["view_index_space"],
        },
        "tracks": tracks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-path", required=True)
    args = parser.parse_args()
    result_path = Path(args.result_path).resolve()
    if result_path.exists():
        raise FileExistsError(
            f"refusing to overwrite validation result: {result_path}"
        )

    recorder = GateRecorder()
    started = time.time()
    try:
        details = run_validation(recorder)
    except Exception as exc:
        payload = {
            "schema": SCHEMA,
            "schema_version": 1,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "gates": recorder.rows,
            "runtime_seconds": time.time() - started,
            "completed_unix": time.time(),
        }
        atomic_json(result_path, payload)
        print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)
        raise

    payload = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "pass",
        "gates": recorder.rows,
        "gate_count": len(recorder.rows),
        "runtime_seconds": time.time() - started,
        "completed_unix": time.time(),
        "torch_version": str(torch.__version__),
        **details,
    }
    atomic_json(result_path, payload)
    print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)
    print("PUBLIC_RADAR_ACCELERATION_V1_VALIDATION_OK", flush=True)


if __name__ == "__main__":
    main()
