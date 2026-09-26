#!/usr/bin/env python
"""Benchmark production PublicRadar rendering with larger point tiles.

Each invocation owns one immutable scene/degree/view-batch/tile result.  It
uses real interleaved PublicRadar views, the validated directional-SH cache,
and the production aligned renderer.  No experiment orchestration or model
checkpoint path is entered.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import statistics
import sys
import time

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rift.public_radar.point_tile_benchmark_v1"
REFERENCE_POINT_CHUNK = 32768
POINT_CHUNKS = (32768, 65536, 131072)
VIEW_BATCH_SIZES = (1, 2, 4, 8)
MEASURED_STEPS = 3
UPDATES_PER_EPOCH = 1800
CUDA_CAP_BYTES = 12 * 1024**3
NOISY_CV_LIMIT = 0.10
# Point tiling changes FP32 scene-weight and grid summation order.  Prediction
# and loss should agree to tens of ppm; accumulated parameter gradients get a
# modestly wider allowance while remaining far below an optimization step.
PREDICTION_RELATIVE_L2_TOLERANCE = 5.0e-5
LOSS_RELATIVE_L2_TOLERANCE = 5.0e-5
GRADIENT_RELATIVE_L2_TOLERANCE = 2.0e-4
FINITE_CHECK_CHUNK = 1_048_576
SCENES = {
    "camry": {
        "trainer": "public_radar_interpolation_dense_v2.py",
        "trainer_scene": "camry",
        "host_cap_kib": 12 * 1024**2,
    },
    "gotcha_full": {
        "trainer": "public_radar_gotcha_full_domain_v1_partial.py",
        "trainer_scene": "gotcha_p2_full_domain",
        "host_cap_kib": 24 * 1024**2,
    },
}


def set_stage(progress, stage):
    progress["stage"] = str(stage)


def update_whole_process_cuda_peak(progress, device):
    progress["cuda_peak_allocated_bytes"] = max(
        int(progress.get("cuda_peak_allocated_bytes", 0)),
        int(torch.cuda.max_memory_allocated(device)),
    )
    progress["cuda_peak_reserved_bytes"] = max(
        int(progress.get("cuda_peak_reserved_bytes", 0)),
        int(torch.cuda.max_memory_reserved(device)),
    )


def load_trainer(scene):
    config = SCENES[scene]
    path = PROJECT_ROOT / "scripts" / config["trainer"]
    name = f"rift_public_radar_point_tiles_{scene}_trainer"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, path


def process_memory_kib():
    values = {}
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        with status_path.open(encoding="utf-8") as handle:
            for line in handle:
                name, _, value = line.partition(":")
                if name in ("VmRSS", "VmHWM"):
                    values[name] = int(value.strip().split()[0])
    values["ru_maxrss"] = int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    )
    if "VmHWM" not in values:
        values["VmHWM"] = values["ru_maxrss"]
    return values


def atomic_json(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing benchmark: {path}")
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


def tensor_all_finite(tensor):
    flat = tensor.detach().reshape(-1)
    for start in range(0, flat.numel(), FINITE_CHECK_CHUNK):
        if not bool(
            torch.isfinite(flat[start : start + FINITE_CHECK_CHUNK]).all()
        ):
            return False
    return True


def model_gradients_finite(model):
    return all(
        parameter.grad is not None and tensor_all_finite(parameter.grad)
        for parameter in model.parameters()
    )


def model_parameters_finite(model):
    return all(tensor_all_finite(parameter) for parameter in model.parameters())


def optimizer_state_finite(optimizer):
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                if not tensor_all_finite(value):
                    return False
            elif isinstance(value, (float, np.floating)):
                if not math.isfinite(float(value)):
                    return False
    return True


def finite_relative_l2(actual, expected):
    actual = torch.as_tensor(actual).detach().cpu()
    expected = torch.as_tensor(expected).detach().cpu()
    if actual.shape != expected.shape:
        return None
    if not bool(torch.isfinite(actual).all()) or not bool(
        torch.isfinite(expected).all()
    ):
        return None
    denominator = float(torch.linalg.vector_norm(expected))
    numerator = float(torch.linalg.vector_norm(actual - expected))
    return numerator / max(denominator, 1.0e-30)


def finite_json_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def normalized_view_losses(predicted, measured, target_mean_power):
    if (
        predicted.ndim != 4
        or measured.shape != predicted.shape
        or predicted.shape[0] == 0
    ):
        raise RuntimeError("aligned renderer returned an unexpected batch shape")
    return (
        (predicted - measured)
        .abs()
        .square()
        .reshape(predicted.shape[0], -1)
        .mean(dim=1)
        / target_mean_power
    )


def item_microbatches(items, view_batch_size):
    for start in range(0, len(items), view_batch_size):
        yield items[start : start + view_batch_size]


def render_backward_window(
    model,
    renderer,
    items,
    target_mean_power,
    view_batch_size,
    progress,
    *,
    stage_prefix,
    capture_gradients,
):
    model.zero_grad(set_to_none=True)
    window_size = len(items)
    predictions = []
    measurements = []
    losses = []
    blocks = list(item_microbatches(items, view_batch_size))
    for block_index, block in enumerate(blocks):
        set_stage(
            progress,
            f"{stage_prefix}_microbatch_{block_index + 1}_of_{len(blocks)}",
        )
        predicted, measured = renderer.raw_prediction_and_measurement_batch(
            model, block
        )
        view_losses = normalized_view_losses(
            predicted, measured, target_mean_power
        )
        (view_losses.sum() / window_size).backward()
        predictions.append(predicted.detach().cpu())
        measurements.append(measured.detach().cpu())
        losses.append(view_losses.detach().cpu())
    prediction = torch.cat(predictions, dim=0)
    measurement = torch.cat(measurements, dim=0)
    view_losses = torch.cat(losses, dim=0)
    gradients = None
    if capture_gradients:
        gradients = [
            parameter.grad.detach().cpu().clone()
            if parameter.grad is not None
            else None
            for parameter in model.parameters()
        ]
    return {
        "prediction": prediction,
        "measurement": measurement,
        "view_losses": view_losses,
        "loss": float(view_losses.mean()),
        "gradients": gradients,
        "prediction_finite": tensor_all_finite(prediction),
        "measurement_finite": tensor_all_finite(measurement),
        "loss_finite": math.isfinite(float(view_losses.mean())),
        "gradients_finite": model_gradients_finite(model),
        "window_views": window_size,
        "microbatches": len(blocks),
        "optimizer_steps": 0,
    }


def compare_parameter_gradients(model, expected_gradients):
    if len(expected_gradients) != len(list(model.parameters())):
        return {
            "all_present": False,
            "all_finite": False,
            "relative_l2_by_parameter": None,
            "maximum_relative_l2": None,
            "maximum_absolute_error": None,
        }
    relative_errors = []
    maximum_absolute_error = 0.0
    all_finite = True
    for parameter, expected in zip(model.parameters(), expected_gradients):
        if (
            parameter.grad is None
            or expected is None
            or parameter.grad.shape != expected.shape
        ):
            return {
                "all_present": False,
                "all_finite": False,
                "relative_l2_by_parameter": None,
                "maximum_relative_l2": None,
                "maximum_absolute_error": None,
            }
        current = parameter.grad.detach().reshape(-1)
        expected = expected.reshape(-1)
        difference_squared = 0.0
        expected_squared = 0.0
        for start in range(0, current.numel(), FINITE_CHECK_CHUNK):
            stop = min(start + FINITE_CHECK_CHUNK, current.numel())
            current_chunk = current[start:stop].cpu()
            expected_chunk = expected[start:stop]
            if not bool(torch.isfinite(current_chunk).all()) or not bool(
                torch.isfinite(expected_chunk).all()
            ):
                all_finite = False
                continue
            difference = current_chunk.to(torch.float64) - expected_chunk.to(
                torch.float64
            )
            difference_squared += float(difference.square().sum())
            expected_squared += float(
                expected_chunk.to(torch.float64).square().sum()
            )
            maximum_absolute_error = max(
                maximum_absolute_error,
                float(difference.abs().max()) if difference.numel() else 0.0,
            )
        relative_errors.append(
            math.sqrt(difference_squared)
            / max(math.sqrt(expected_squared), 1.0e-30)
        )
    return {
        "all_present": True,
        "all_finite": bool(all_finite),
        "relative_l2_by_parameter": relative_errors,
        "maximum_relative_l2": max(relative_errors),
        "maximum_absolute_error": maximum_absolute_error,
    }


def point_tile_correctness_gate(
    trainer,
    arrays,
    metadata,
    cache,
    model,
    items,
    target_mean_power,
    view_batch_size,
    candidate_point_chunk,
    device,
    progress,
):
    reference_renderer = trainer.SerializedRenderer(
        arrays,
        metadata,
        device,
        REFERENCE_POINT_CHUNK,
        pair_chunk=1,
        sh_basis_cache=cache,
    )
    candidate_renderer = trainer.SerializedRenderer(
        arrays,
        metadata,
        device,
        candidate_point_chunk,
        pair_chunk=1,
        sh_basis_cache=cache,
    )
    reference = render_backward_window(
        model,
        reference_renderer,
        items,
        target_mean_power,
        view_batch_size,
        progress,
        stage_prefix="correctness_reference_32k",
        capture_gradients=True,
    )
    expected_gradients = reference.pop("gradients")
    candidate = render_backward_window(
        model,
        candidate_renderer,
        items,
        target_mean_power,
        view_batch_size,
        progress,
        stage_prefix=f"correctness_candidate_{candidate_point_chunk}",
        capture_gradients=False,
    )
    candidate.pop("gradients")
    gradient_comparison = compare_parameter_gradients(
        model, expected_gradients
    )
    prediction_relative_l2 = finite_relative_l2(
        candidate["prediction"], reference["prediction"]
    )
    view_loss_relative_l2 = finite_relative_l2(
        candidate["view_losses"], reference["view_losses"]
    )
    measurement_equal = torch.equal(
        candidate["measurement"], reference["measurement"]
    )
    loss_relative_error = None
    if math.isfinite(reference["loss"]) and math.isfinite(candidate["loss"]):
        loss_relative_error = abs(candidate["loss"] - reference["loss"]) / max(
            abs(reference["loss"]), 1.0e-30
        )
    all_finite = bool(
        reference["prediction_finite"]
        and reference["measurement_finite"]
        and reference["loss_finite"]
        and reference["gradients_finite"]
        and candidate["prediction_finite"]
        and candidate["measurement_finite"]
        and candidate["loss_finite"]
        and candidate["gradients_finite"]
        and gradient_comparison["all_finite"]
    )
    passed = bool(
        all_finite
        and measurement_equal
        and prediction_relative_l2 is not None
        and prediction_relative_l2
        <= PREDICTION_RELATIVE_L2_TOLERANCE
        and view_loss_relative_l2 is not None
        and view_loss_relative_l2 <= LOSS_RELATIVE_L2_TOLERANCE
        and loss_relative_error is not None
        and loss_relative_error <= LOSS_RELATIVE_L2_TOLERANCE
        and gradient_comparison["maximum_relative_l2"] is not None
        and gradient_comparison["maximum_relative_l2"]
        <= GRADIENT_RELATIVE_L2_TOLERANCE
    )
    payload = {
        "reference_point_chunk": REFERENCE_POINT_CHUNK,
        "candidate_point_chunk": int(candidate_point_chunk),
        "window_views": len(items),
        "view_batch_size": int(view_batch_size),
        "microbatches_per_window": int(
            math.ceil(len(items) / view_batch_size)
        ),
        "same_measurement_exact": bool(measurement_equal),
        "all_finite": all_finite,
        "prediction_relative_l2": prediction_relative_l2,
        "view_loss_relative_l2": view_loss_relative_l2,
        "mean_loss_relative_error": loss_relative_error,
        "parameter_order": [name for name, _ in model.named_parameters()],
        "parameter_gradients": gradient_comparison,
        "tolerances": {
            "prediction_relative_l2": PREDICTION_RELATIVE_L2_TOLERANCE,
            "loss_relative_l2": LOSS_RELATIVE_L2_TOLERANCE,
            "parameter_gradient_relative_l2": (
                GRADIENT_RELATIVE_L2_TOLERANCE
            ),
        },
        "passed": passed,
    }
    model.zero_grad(set_to_none=True)
    del reference, candidate, expected_gradients
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    update_whole_process_cuda_peak(progress, device)
    return payload


def timed_training_window(
    model,
    optimizer,
    renderer,
    items,
    target_mean_power,
    view_batch_size,
    progress,
    stage_prefix,
):
    torch.cuda.synchronize()
    total_start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    window_size = len(items)
    blocks = list(item_microbatches(items, view_batch_size))
    forward_seconds = 0.0
    backward_seconds = 0.0
    loss_values = []
    for block_index, block in enumerate(blocks):
        set_stage(
            progress,
            f"{stage_prefix}_microbatch_{block_index + 1}_of_{len(blocks)}",
        )
        forward_start = time.perf_counter()
        predicted, measured = renderer.raw_prediction_and_measurement_batch(
            model, block
        )
        view_losses = normalized_view_losses(
            predicted, measured, target_mean_power
        )
        torch.cuda.synchronize()
        forward_seconds += time.perf_counter() - forward_start

        backward_start = time.perf_counter()
        (view_losses.sum() / window_size).backward()
        torch.cuda.synchronize()
        backward_seconds += time.perf_counter() - backward_start
        loss_values.extend(
            finite_json_float(value) for value in view_losses.detach().cpu()
        )

    set_stage(progress, f"{stage_prefix}_optimizer_step")
    optimizer_start = time.perf_counter()
    optimizer.step()
    torch.cuda.synchronize()
    optimizer_seconds = time.perf_counter() - optimizer_start
    total_seconds = time.perf_counter() - total_start

    return {
        "loss": (
            sum(value for value in loss_values if value is not None)
            / window_size
            if all(value is not None for value in loss_values)
            else None
        ),
        "view_losses": loss_values,
        "loss_finite": all(value is not None for value in loss_values),
        "window_views": window_size,
        "microbatches": len(blocks),
        "optimizer_steps": 1,
        "forward_seconds": float(forward_seconds),
        "backward_seconds": float(backward_seconds),
        "optimizer_seconds": float(optimizer_seconds),
        "total_seconds": float(total_seconds),
    }


def timing_summary(steps):
    individual = {}
    medians = {}
    coefficients_of_variation = {}
    for component in ("forward", "backward", "optimizer", "total"):
        key = f"{component}_seconds"
        values = [float(step[key]) for step in steps]
        mean = statistics.fmean(values)
        individual[component] = values
        medians[component] = float(statistics.median(values))
        coefficients_of_variation[component] = (
            float(statistics.pstdev(values) / mean) if mean > 0.0 else None
        )
    return {
        "individual_seconds": individual,
        "median_seconds": medians,
        "coefficient_of_variation": coefficients_of_variation,
        "noisy_cv_limit": NOISY_CV_LIMIT,
    }


def initialize_nonzero_scene(model):
    generator = torch.Generator(device=model.w_re.device)
    generator.manual_seed(420_091 + int(model.max_degree))
    coefficient_std = 1.0e-3 / math.sqrt(2.0 * model.n_basis)
    with torch.no_grad():
        model.w_re.normal_(0.0, coefficient_std, generator=generator)
        model.w_im.normal_(0.0, coefficient_std, generator=generator)
    return coefficient_std


def load_real_problem(args, device, progress):
    set_stage(progress, "load_integrated_trainer")
    trainer, trainer_path = load_trainer(args.scene)
    trainer_scene = SCENES[args.scene]["trainer_scene"]
    set_stage(progress, "validate_real_interleaved_dataset")
    loaded = trainer.load_dataset_contract(args.npz_path, trainer_scene)
    arrays, metadata, partition = loaded[:3]
    support = loaded[3] if len(loaded) > 3 else None
    train_indices = np.asarray(arrays["train_indices"], dtype=np.int64)
    set_stage(progress, "materialize_real_training_dataset")
    base_dataset = trainer.PecSphereNPZDataset(arrays, train_indices)
    set_stage(progress, "validate_immutable_sh_basis_cache")
    cache = trainer.SH_CACHE.PublicRadarSHBasisCache(
        args.sh_basis_cache,
        arrays["viewpoint_positions"],
        expected_dataset_name=trainer.SCENES[trainer_scene]["dataset_name"],
    )
    set_stage(progress, "validate_canonical_view_identity")
    dataset = trainer.SH_CACHE.IndexedPublicRadarDataset(
        base_dataset,
        train_indices,
        arrays["viewpoint_positions"],
    )
    window_size = int(math.ceil(int(partition["train"]) / UPDATES_PER_EPOCH))
    required_views = 3 * window_size
    if len(dataset) < required_views:
        raise ValueError("training split is too small for three benchmark windows")
    local_indices = np.linspace(
        0,
        len(dataset) - 1,
        num=required_views,
        dtype=np.int64,
    )
    if np.unique(local_indices).size != required_views:
        raise ValueError("benchmark view selection contains duplicate local indices")
    item_windows = []
    for role_index in range(3):
        window = local_indices[role_index::3]
        item_windows.append([dataset[int(index)] for index in window])
    if len(item_windows) != 3 or any(
        len(window) != window_size for window in item_windows
    ):
        raise RuntimeError("benchmark windows do not match production sizing")
    set_stage(progress, "compute_fixed_training_target_power")
    target_mean_power, target_power_sum, target_sample_count = (
        trainer.V1.mean_train_target_power(arrays, train_indices)
    )
    set_stage(progress, "allocate_production_scene")
    model = trainer.make_scene(trainer_scene, args.degree, device)
    coefficient_std = initialize_nonzero_scene(model)
    update_whole_process_cuda_peak(progress, device)
    return {
        "trainer": trainer,
        "trainer_path": trainer_path,
        "trainer_scene": trainer_scene,
        "arrays": arrays,
        "metadata": metadata,
        "partition": partition,
        "support": support,
        "cache": cache,
        "dataset": dataset,
        "item_windows": item_windows,
        "window_size": window_size,
        "target_mean_power": float(target_mean_power),
        "target_power_sum": float(target_power_sum),
        "target_sample_count": int(target_sample_count),
        "model": model,
        "coefficient_std": coefficient_std,
    }


def timing_is_valid(
    steps,
    summary,
    expected_window_views=None,
    expected_microbatches=None,
):
    if len(steps) != MEASURED_STEPS:
        return False
    for step in steps:
        if step["optimizer_steps"] != 1:
            return False
        if (
            expected_window_views is not None
            and step["window_views"] != expected_window_views
        ):
            return False
        if (
            expected_microbatches is not None
            and step["microbatches"] != expected_microbatches
        ):
            return False
        for component in ("forward", "backward", "optimizer", "total"):
            value = float(step[f"{component}_seconds"])
            if not math.isfinite(value) or value <= 0.0:
                return False
        component_sum = sum(
            float(step[f"{component}_seconds"])
            for component in ("forward", "backward", "optimizer")
        )
        if float(step["total_seconds"]) + 1.0e-6 < component_sum:
            return False
    return all(
        summary["coefficient_of_variation"][component] is not None
        and math.isfinite(summary["median_seconds"][component])
        and summary["median_seconds"][component] > 0.0
        for component in ("forward", "backward", "optimizer", "total")
    )


def measure_renderer_phase(
    *,
    label,
    point_chunk,
    trainer,
    arrays,
    metadata,
    cache,
    model,
    warmup_items,
    measured_items,
    target_mean_power,
    view_batch_size,
    device,
    progress,
):
    set_stage(progress, f"{label}_reset_scene")
    coefficient_std = initialize_nonzero_scene(model)
    renderer = trainer.SerializedRenderer(
        arrays,
        metadata,
        device,
        point_chunk,
        pair_chunk=1,
        sh_basis_cache=cache,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-5,
        eps=1.0e-15,
        weight_decay=0.0,
    )
    warmup = timed_training_window(
        model,
        optimizer,
        renderer,
        warmup_items,
        target_mean_power,
        view_batch_size,
        progress,
        f"{label}_warmup",
    )
    update_whole_process_cuda_peak(progress, device)
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    free_before, total_before = torch.cuda.mem_get_info(device)
    baseline_allocated = int(torch.cuda.memory_allocated(device))
    baseline_reserved = int(torch.cuda.memory_reserved(device))
    host_before = process_memory_kib()
    torch.cuda.reset_peak_memory_stats()
    steps = []
    for repetition in range(MEASURED_STEPS):
        steps.append(
            timed_training_window(
                model,
                optimizer,
                renderer,
                measured_items,
                target_mean_power,
                view_batch_size,
                progress,
                f"{label}_measured_window_{repetition + 1}",
            )
        )
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    update_whole_process_cuda_peak(progress, device)
    free_after, total_after = torch.cuda.mem_get_info(device)
    host_after = process_memory_kib()
    set_stage(progress, f"{label}_timing_and_finite_checks")
    timing = timing_summary(steps)
    gradients_finite = model_gradients_finite(model)
    parameters_finite = model_parameters_finite(model)
    adam_state_finite = optimizer_state_finite(optimizer)
    losses_finite = bool(
        warmup["loss_finite"]
        and all(step["loss_finite"] for step in steps)
    )
    timing_valid = timing_is_valid(
        steps,
        timing,
        expected_window_views=len(measured_items),
        expected_microbatches=math.ceil(
            len(measured_items) / view_batch_size
        ),
    )
    result = {
        "label": label,
        "point_chunk": int(point_chunk),
        "initial_coefficient_std": coefficient_std,
        "warmup_window": warmup,
        "measured_windows": steps,
        "timing": timing,
        "timing_valid": timing_valid,
        "finite_checks": {
            "losses": losses_finite,
            "gradients": bool(gradients_finite),
            "model_parameters": bool(parameters_finite),
            "adam_state": bool(adam_state_finite),
        },
        "numerical_finite": bool(
            losses_finite
            and gradients_finite
            and parameters_finite
            and adam_state_finite
        ),
        "cuda_peak_scope": (
            f"three_repetitions_of_one_production_window_after_{label}_warmup"
        ),
        "cuda_baseline_allocated_bytes": baseline_allocated,
        "cuda_baseline_reserved_bytes": baseline_reserved,
        "cuda_peak_allocated_bytes": peak_allocated,
        "cuda_peak_reserved_bytes": peak_reserved,
        "cuda_incremental_peak_allocated_bytes": (
            peak_allocated - baseline_allocated
        ),
        "cuda_incremental_peak_reserved_bytes": (
            peak_reserved - baseline_reserved
        ),
        "cuda_free_before_bytes": int(free_before),
        "cuda_free_after_bytes": int(free_after),
        "cuda_mem_get_info_total_before_bytes": int(total_before),
        "cuda_mem_get_info_total_after_bytes": int(total_after),
        "host_before_kib": host_before,
        "host_after_kib": host_after,
        "host_rss_before_kib": int(host_before.get("VmRSS", 0)),
        "host_rss_after_kib": int(host_after.get("VmRSS", 0)),
        "host_rss_delta_kib": int(
            host_after.get("VmRSS", 0) - host_before.get("VmRSS", 0)
        ),
    }
    set_stage(progress, f"{label}_cleanup")
    optimizer.zero_grad(set_to_none=True)
    del optimizer, renderer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return result


def paired_speedups(reference, candidate):
    median_speedup = {}
    paired_values = {}
    paired_medians = {}
    for component in ("forward", "backward", "optimizer", "total"):
        reference_values = reference["timing"]["individual_seconds"][component]
        candidate_values = candidate["timing"]["individual_seconds"][component]
        values = [
            reference_value / candidate_value
            for reference_value, candidate_value in zip(
                reference_values, candidate_values
            )
        ]
        paired_values[component] = values
        paired_medians[component] = float(statistics.median(values))
        median_speedup[component] = (
            reference["timing"]["median_seconds"][component]
            / candidate["timing"]["median_seconds"][component]
        )
    return {
        "definition": "reference_32k_seconds_divided_by_candidate_seconds",
        "median_time_speedup": median_speedup,
        "paired_repetition_speedups": paired_values,
        "median_paired_repetition_speedup": paired_medians,
    }


def gotcha_support_preflight_identity(support):
    if support is None:
        return None
    preflight = support["preflight"]
    return {
        "support_schema": support["schema"],
        "mask_realization": preflight["mask_realization"],
        "view_count": int(preflight["view_count"]),
        "minimum_lower_window_margin_m": float(
            preflight["minimum_lower_window_margin_m"]
        ),
        "minimum_upper_window_margin_m": float(
            preflight["minimum_upper_window_margin_m"]
        ),
    }


def benchmark(args, device, progress):
    problem = load_real_problem(args, device, progress)
    trainer = problem["trainer"]
    model = problem["model"]
    cache = problem["cache"]
    item_windows = problem["item_windows"]
    target_mean_power = problem["target_mean_power"]
    canonical_windows = [
        trainer.SH_CACHE.item_view_indices(items).tolist()
        for items in item_windows
    ]
    scene_config = trainer.SCENES[problem["trainer_scene"]]
    pitch_x, pitch_y = model.pitch_xy
    common = {
        "schema": SCHEMA,
        "schema_version": 1,
        "scene": args.scene,
        "trainer_scene": problem["trainer_scene"],
        "trainer_source": str(problem["trainer_path"].resolve()),
        "dataset": str(Path(args.npz_path).resolve()),
        "degree": int(args.degree),
        "view_batch_size": int(args.view_batch_size),
        "point_chunk": int(args.point_chunk),
        "reference_point_chunk": REFERENCE_POINT_CHUNK,
        "production_updates_per_epoch": UPDATES_PER_EPOCH,
        "production_window_views": int(problem["window_size"]),
        "microbatches_per_window": int(
            math.ceil(problem["window_size"] / args.view_batch_size)
        ),
        "window_loss_reduction": (
            "sum_per_view_normalized_losses_divided_by_full_window_size"
        ),
        "points": int(model.n_points),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "parameter_dtype": str(model.w_re.dtype),
        "parameter_dtypes": {
            "w_re": str(model.w_re.dtype),
            "w_im": str(model.w_im.dtype),
        },
        "scene_geometry": {
            "shape": list(model.shape),
            "extent_xy_m": float(scene_config["extent"]),
            "x_bounds_m": [
                -float(scene_config["extent"]),
                float(scene_config["extent"]),
            ],
            "y_bounds_m": [
                -float(scene_config["extent"]),
                float(scene_config["extent"]),
            ],
            "pitch_x_m": float(pitch_x),
            "pitch_y_m": float(pitch_y),
            "z_m": 0.0,
        },
        "physics": {
            "range_model": "none",
            "propagation_model": str(problem["metadata"]["propagation_model"]),
            "reference_range_m": float(
                problem["metadata"]["reference_range_m"]
            ),
            "scene_center_m": [
                float(value)
                for value in problem["metadata"]["scene_center_m"]
            ],
            "phase_sign": -1.0,
        },
        "gotcha_support_preflight": gotcha_support_preflight_identity(
            problem["support"]
        ),
        "frequencies": int(problem["arrays"]["frequencies_hz"].size),
        "train_views": int(problem["partition"]["train"]),
        "target_mean_power": target_mean_power,
        "target_power_sum": problem["target_power_sum"],
        "target_sample_count": problem["target_sample_count"],
        "initial_coefficient_std": problem["coefficient_std"],
        "sh_basis_cache": cache.contract(),
        "canonical_view_windows": {
            "correctness": canonical_windows[0],
            "warmup": canonical_windows[1],
            "measured_repeated_three_times": canonical_windows[2],
        },
        "device": torch.cuda.get_device_name(device),
        "cuda_total_bytes": int(
            torch.cuda.get_device_properties(device).total_memory
        ),
        "cuda_cap_bytes": CUDA_CAP_BYTES,
        "host_cap_kib": int(SCENES[args.scene]["host_cap_kib"]),
        "host_cap_scope": (
            "whole_process_VmHWM_including_dataset_cache_correctness_and_"
            "both_paired_timing_phases"
        ),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
    }

    correctness = point_tile_correctness_gate(
        trainer,
        problem["arrays"],
        problem["metadata"],
        cache,
        model,
        item_windows[0],
        target_mean_power,
        args.view_batch_size,
        args.point_chunk,
        device,
        progress,
    )
    if not correctness["all_finite"]:
        set_stage(progress, "correctness_nonfinite")
        update_whole_process_cuda_peak(progress, device)
        host_after = process_memory_kib()
        return {
            **common,
            "status": "nonfinite",
            "stage": progress["stage"],
            "measurement_completed": False,
            "memory_safe": None,
            "correctness_gate": correctness,
            "unsafe_reasons": [],
            "cuda_whole_process_peak_scope": (
                "maximum_across_all_stage_isolated_peak_counters"
            ),
            "cuda_whole_process_peak_allocated_bytes": progress[
                "cuda_peak_allocated_bytes"
            ],
            "cuda_whole_process_peak_reserved_bytes": progress[
                "cuda_peak_reserved_bytes"
            ],
            "host_after_kib": host_after,
            "host_VmHWM_kib": int(host_after["VmHWM"]),
            "host_VmHWM_scope": common["host_cap_scope"],
            "completed_unix": time.time(),
        }
    if not correctness["passed"]:
        set_stage(progress, "correctness_tolerance_failure")
        update_whole_process_cuda_peak(progress, device)
        host_after = process_memory_kib()
        return {
            **common,
            "status": "unsafe",
            "stage": progress["stage"],
            "measurement_completed": False,
            "memory_safe": None,
            "correctness_gate": correctness,
            "unsafe_reasons": ["point_tile_correctness_gate"],
            "cuda_whole_process_peak_scope": (
                "maximum_across_all_stage_isolated_peak_counters"
            ),
            "cuda_whole_process_peak_allocated_bytes": progress[
                "cuda_peak_allocated_bytes"
            ],
            "cuda_whole_process_peak_reserved_bytes": progress[
                "cuda_peak_reserved_bytes"
            ],
            "host_after_kib": host_after,
            "host_VmHWM_kib": int(host_after["VmHWM"]),
            "host_VmHWM_scope": common["host_cap_scope"],
            "completed_unix": time.time(),
        }

    reference = measure_renderer_phase(
        label="reference_32k",
        point_chunk=REFERENCE_POINT_CHUNK,
        trainer=trainer,
        arrays=problem["arrays"],
        metadata=problem["metadata"],
        cache=cache,
        model=model,
        warmup_items=item_windows[1],
        measured_items=item_windows[2],
        target_mean_power=target_mean_power,
        view_batch_size=args.view_batch_size,
        device=device,
        progress=progress,
    )
    candidate = measure_renderer_phase(
        label=f"candidate_{args.point_chunk}",
        point_chunk=args.point_chunk,
        trainer=trainer,
        arrays=problem["arrays"],
        metadata=problem["metadata"],
        cache=cache,
        model=model,
        warmup_items=item_windows[1],
        measured_items=item_windows[2],
        target_mean_power=target_mean_power,
        view_batch_size=args.view_batch_size,
        device=device,
        progress=progress,
    )
    set_stage(progress, "final_engineering_checks")
    timing_valid = bool(
        reference["timing_valid"] and candidate["timing_valid"]
    )
    speedup = paired_speedups(reference, candidate) if timing_valid else None
    update_whole_process_cuda_peak(progress, device)
    host_after = process_memory_kib()
    host_peak_kib = int(host_after["VmHWM"])
    numerical_finite = bool(
        correctness["all_finite"]
        and reference["numerical_finite"]
        and candidate["numerical_finite"]
    )
    memory_safe = bool(
        progress["cuda_peak_allocated_bytes"] <= CUDA_CAP_BYTES
        and progress["cuda_peak_reserved_bytes"] <= CUDA_CAP_BYTES
        and reference["cuda_peak_allocated_bytes"] <= CUDA_CAP_BYTES
        and reference["cuda_peak_reserved_bytes"] <= CUDA_CAP_BYTES
        and candidate["cuda_peak_allocated_bytes"] <= CUDA_CAP_BYTES
        and candidate["cuda_peak_reserved_bytes"] <= CUDA_CAP_BYTES
        and host_peak_kib <= SCENES[args.scene]["host_cap_kib"]
    )
    unsafe_reasons = []
    if not timing_valid:
        unsafe_reasons.append("timing_invalid")
    if progress["cuda_peak_allocated_bytes"] > CUDA_CAP_BYTES:
        unsafe_reasons.append("cuda_whole_process_peak_allocated")
    if progress["cuda_peak_reserved_bytes"] > CUDA_CAP_BYTES:
        unsafe_reasons.append("cuda_whole_process_peak_reserved")
    if reference["cuda_peak_allocated_bytes"] > CUDA_CAP_BYTES:
        unsafe_reasons.append("reference_cuda_peak_allocated")
    if reference["cuda_peak_reserved_bytes"] > CUDA_CAP_BYTES:
        unsafe_reasons.append("reference_cuda_peak_reserved")
    if candidate["cuda_peak_allocated_bytes"] > CUDA_CAP_BYTES:
        unsafe_reasons.append("candidate_cuda_peak_allocated")
    if candidate["cuda_peak_reserved_bytes"] > CUDA_CAP_BYTES:
        unsafe_reasons.append("candidate_cuda_peak_reserved")
    if host_peak_kib > SCENES[args.scene]["host_cap_kib"]:
        unsafe_reasons.append("host_whole_process_VmHWM")
    timing_components = ("forward", "backward", "total")
    noisy_components = []
    for phase_name, phase in (("reference", reference), ("candidate", candidate)):
        for component in timing_components:
            coefficient = phase["timing"]["coefficient_of_variation"][component]
            if coefficient is not None and coefficient > NOISY_CV_LIMIT:
                noisy_components.append(f"{phase_name}.{component}")
    noisy = bool(noisy_components)
    if not numerical_finite:
        status = "nonfinite"
    elif unsafe_reasons or not memory_safe:
        status = "unsafe"
    elif noisy:
        status = "noisy"
    else:
        status = "pass"
    set_stage(progress, "complete")
    return {
        **common,
        "status": status,
        "stage": progress["stage"],
        "measurement_completed": True,
        "correctness_gate": correctness,
        "reference_32k": reference,
        "candidate": candidate,
        "paired_speedup": speedup,
        "candidate_host_rss_before_kib": candidate["host_rss_before_kib"],
        "candidate_host_rss_after_kib": candidate["host_rss_after_kib"],
        "candidate_host_rss_delta_kib": candidate["host_rss_delta_kib"],
        "noisy_components": noisy_components,
        "numerical_finite": numerical_finite,
        "timing_valid": timing_valid,
        "memory_safe": memory_safe,
        "unsafe_reasons": unsafe_reasons,
        "cuda_whole_process_peak_scope": (
            "maximum_across_all_stage_isolated_peak_counters"
        ),
        "cuda_whole_process_peak_allocated_bytes": progress[
            "cuda_peak_allocated_bytes"
        ],
        "cuda_whole_process_peak_reserved_bytes": progress[
            "cuda_peak_reserved_bytes"
        ],
        "host_after_kib": host_after,
        "host_VmHWM_kib": host_peak_kib,
        "host_VmHWM_scope": common["host_cap_scope"],
        "completed_unix": time.time(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, choices=tuple(SCENES))
    parser.add_argument("--degree", required=True, type=int, choices=(0, 3))
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--view-batch-size",
        required=True,
        type=int,
        choices=VIEW_BATCH_SIZES,
    )
    parser.add_argument(
        "--point-chunk",
        required=True,
        type=int,
        choices=POINT_CHUNKS,
    )
    parser.add_argument("--sh-basis-cache", required=True)
    args = parser.parse_args()
    if not str(args.sh_basis_cache).strip():
        raise ValueError("--sh-basis-cache must be a nonempty path")
    result_path = (
        Path(args.run_root).resolve()
        / (
            f"point_tiles_{args.scene}_deg{args.degree}_"
            f"b{args.view_batch_size}_tile{args.point_chunk}.json"
        )
    )
    if result_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing benchmark: {result_path}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("point-tile benchmark requires exactly one CUDA device")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    progress = {
        "stage": "benchmark_start",
        "cuda_peak_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }
    common = {
        "schema": SCHEMA,
        "schema_version": 1,
        "scene": args.scene,
        "degree": int(args.degree),
        "view_batch_size": int(args.view_batch_size),
        "point_chunk": int(args.point_chunk),
        "cuda_cap_bytes": CUDA_CAP_BYTES,
        "host_cap_kib": int(SCENES[args.scene]["host_cap_kib"]),
        "host_cap_scope": (
            "whole_process_VmHWM_including_dataset_cache_correctness_and_"
            "both_paired_timing_phases"
        ),
    }
    unexpected_error = None
    try:
        payload = benchmark(args, device, progress)
    except torch.cuda.OutOfMemoryError as exc:
        update_whole_process_cuda_peak(progress, device)
        host_after = process_memory_kib()
        payload = {
            **common,
            "status": "oom",
            "stage": progress["stage"],
            "measurement_completed": False,
            "memory_safe": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "cuda_whole_process_peak_scope": (
                "maximum_across_all_stage_isolated_peak_counters"
            ),
            "cuda_whole_process_peak_allocated_bytes": progress[
                "cuda_peak_allocated_bytes"
            ],
            "cuda_whole_process_peak_reserved_bytes": progress[
                "cuda_peak_reserved_bytes"
            ],
            "host_after_kib": host_after,
            "host_VmHWM_kib": int(host_after["VmHWM"]),
            "host_VmHWM_scope": common["host_cap_scope"],
            "completed_unix": time.time(),
        }
        torch.cuda.empty_cache()
    except (RuntimeError, MemoryError) as exc:
        update_whole_process_cuda_peak(progress, device)
        host_after = process_memory_kib()
        payload = {
            **common,
            "status": "unsafe",
            "stage": progress["stage"],
            "measurement_completed": False,
            "memory_safe": None,
            "unsafe_reasons": ["unexpected_runtime_or_memory_error"],
            "error_type": type(exc).__name__,
            "error": str(exc),
            "cuda_whole_process_peak_scope": (
                "maximum_across_all_stage_isolated_peak_counters"
            ),
            "cuda_whole_process_peak_allocated_bytes": progress[
                "cuda_peak_allocated_bytes"
            ],
            "cuda_whole_process_peak_reserved_bytes": progress[
                "cuda_peak_reserved_bytes"
            ],
            "host_after_kib": host_after,
            "host_VmHWM_kib": int(host_after["VmHWM"]),
            "host_VmHWM_scope": common["host_cap_scope"],
            "completed_unix": time.time(),
        }
        unexpected_error = exc
    except Exception as exc:
        update_whole_process_cuda_peak(progress, device)
        host_after = process_memory_kib()
        payload = {
            **common,
            "status": "unsafe",
            "stage": progress["stage"],
            "measurement_completed": False,
            "memory_safe": None,
            "unsafe_reasons": ["unexpected_exception"],
            "error_type": type(exc).__name__,
            "error": str(exc),
            "cuda_whole_process_peak_scope": (
                "maximum_across_all_stage_isolated_peak_counters"
            ),
            "cuda_whole_process_peak_allocated_bytes": progress[
                "cuda_peak_allocated_bytes"
            ],
            "cuda_whole_process_peak_reserved_bytes": progress[
                "cuda_peak_reserved_bytes"
            ],
            "host_after_kib": host_after,
            "host_VmHWM_kib": int(host_after["VmHWM"]),
            "host_VmHWM_scope": common["host_cap_scope"],
            "completed_unix": time.time(),
        }
        unexpected_error = exc
    atomic_json(result_path, payload)
    print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)
    if unexpected_error is not None:
        raise RuntimeError(
            "point-tile benchmark failed after publishing its unsafe artifact"
        ) from unexpected_error


if __name__ == "__main__":
    main()
