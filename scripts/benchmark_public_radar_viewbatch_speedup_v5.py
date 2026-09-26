#!/usr/bin/env python
"""Paired GOTCHA timing gate for scalar and aligned viewpoint rendering.

This is an isolated benchmark, not a training entry point.  One invocation
owns one immutable degree/B result.  It uses the production full-domain
GOTCHA scene, the production 19-view accumulation-window size, live SH
evaluation on both paths, and no checkpoint or experiment run directory.
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
SCHEMA = "rift.public_radar.viewbatch_speedup_v5_inferno"
SCHEMA_VERSION = 5
SCENE_KEY = "gotcha_p2_full_domain"
POINT_CHUNK = 32768
PAIR_CHUNK = 1
VIEW_BATCH_SIZES = (1, 2, 4, 8)
UPDATES_PER_EPOCH = 1800
MEASURED_WINDOWS = 3
CUDA_CAP_BYTES = 12 * 1024**3
HOST_CAP_KIB = 24 * 1024**2
NOISY_CV_LIMIT = 0.10
FINITE_CHECK_CHUNK = 1_048_576
PREDICTION_RELATIVE_L2_TOLERANCE = 2.0e-6
LOSS_RELATIVE_ERROR_TOLERANCE = 2.0e-6
GRADIENT_RELATIVE_L2_TOLERANCE = 5.0e-6
# The tiny gate is a cheap early check, not a stricter numerical contract than
# the full production-window gate below.  CUDA reduction order can differ at
# low single-ppm scale, so use the same forward/loss allowance here and let the
# full 19-view gradient comparison remain authoritative.
TINY_FORWARD_RELATIVE_L2_TOLERANCE = PREDICTION_RELATIVE_L2_TOLERANCE
TINY_LOSS_RELATIVE_ERROR_TOLERANCE = LOSS_RELATIVE_ERROR_TOLERANCE
TINY_GRADIENT_RELATIVE_L2_TOLERANCE = 5.0e-6
INITIALIZATION_SEED = 420_091


def set_stage(progress, stage):
    progress["stage"] = str(stage)


def update_cuda_peak(progress, device):
    progress["cuda_peak_allocated_bytes"] = max(
        int(progress.get("cuda_peak_allocated_bytes", 0)),
        int(torch.cuda.max_memory_allocated(device)),
    )
    progress["cuda_peak_reserved_bytes"] = max(
        int(progress.get("cuda_peak_reserved_bytes", 0)),
        int(torch.cuda.max_memory_reserved(device)),
    )


def load_trainer():
    path = PROJECT_ROOT / "scripts" / "public_radar_gotcha_full_domain_v1_partial.py"
    spec = importlib.util.spec_from_file_location(
        "rift_public_radar_viewbatch_speedup_v5_trainer", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
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
    values["ru_maxrss"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
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
        if not bool(torch.isfinite(flat[start : start + FINITE_CHECK_CHUNK]).all()):
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
            if isinstance(value, torch.Tensor) and not tensor_all_finite(value):
                return False
            if isinstance(value, (float, np.floating)) and not math.isfinite(
                float(value)
            ):
                return False
    return True


@torch.no_grad()
def reset_allocated_adamw_state(optimizer):
    """Restore an allocated AdamW optimizer to its exact fresh zero state.

    The unmeasured warmup below must allocate Adam's large moment buffers so
    their one-time creation does not contaminate the first timed window.  All
    state values are then reset to zero, which is the state immediately before
    the first production optimizer step.
    """

    if not optimizer.state:
        raise RuntimeError("AdamW warmup did not allocate optimizer state")
    state_schema = []
    for state in optimizer.state.values():
        keys = sorted(str(key) for key in state)
        state_schema.append(keys)
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                value.zero_()
            elif isinstance(value, (int, float, np.number)):
                state[key] = type(value)(0)
            else:
                raise RuntimeError(
                    f"unsupported AdamW state value for {key!r}: {type(value)!r}"
                )
    if not optimizer_state_finite(optimizer):
        raise RuntimeError("reset AdamW state is non-finite")
    if any(
        bool(torch.count_nonzero(value))
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    ):
        raise RuntimeError("AdamW state reset left a nonzero tensor")
    return state_schema


def finite_relative_l2(actual, expected):
    actual = torch.as_tensor(actual).detach().cpu()
    expected = torch.as_tensor(expected).detach().cpu()
    if actual.shape != expected.shape:
        return None
    if not tensor_all_finite(actual) or not tensor_all_finite(expected):
        return None
    numerator = float(torch.linalg.vector_norm(actual - expected))
    denominator = float(torch.linalg.vector_norm(expected))
    return numerator / max(denominator, 1.0e-30)


def finite_json_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def initialize_nonzero_scene(model):
    generator = torch.Generator(device=model.w_re.device)
    generator.manual_seed(INITIALIZATION_SEED + int(model.max_degree))
    coefficient_std = 1.0e-3 / math.sqrt(2.0 * model.n_basis)
    with torch.no_grad():
        model.w_re.normal_(0.0, coefficient_std, generator=generator)
        model.w_im.normal_(0.0, coefficient_std, generator=generator)
    model.zero_grad(set_to_none=True)
    signature = {
        "seed": INITIALIZATION_SEED + int(model.max_degree),
        "coefficient_std": float(coefficient_std),
        "w_re_head": [
            float(value)
            for value in model.w_re.detach().reshape(-1)[:8].cpu().tolist()
        ],
        "w_im_head": [
            float(value)
            for value in model.w_im.detach().reshape(-1)[:8].cpu().tolist()
        ],
    }
    return signature


def normalized_scalar_loss(predicted, measured, target_mean_power):
    return (predicted - measured).abs().square().mean() / target_mean_power


def normalized_batch_losses(predicted, measured, target_mean_power):
    if predicted.ndim != 4 or predicted.shape != measured.shape:
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
    *,
    mode,
    view_batch_size,
    progress,
    stage_prefix,
    capture_gradients,
):
    """Render one complete accumulation window without an optimizer step."""

    model.zero_grad(set_to_none=True)
    predictions = []
    measurements = []
    losses = []
    window_size = len(items)
    if mode == "former_scalar":
        blocks = [[item] for item in items]
    elif mode == "aligned":
        blocks = list(item_microbatches(items, view_batch_size))
    else:
        raise ValueError(f"unknown renderer mode {mode!r}")

    for block_index, block in enumerate(blocks):
        set_stage(
            progress,
            f"{stage_prefix}_microbatch_{block_index + 1}_of_{len(blocks)}",
        )
        if mode == "former_scalar":
            predicted, measured = renderer.raw_prediction_and_measurement(
                model, block[0]
            )
            predicted = predicted.unsqueeze(0)
            measured = measured.unsqueeze(0)
            view_losses = normalized_batch_losses(
                predicted, measured, target_mean_power
            )
        else:
            predicted, measured = renderer.raw_prediction_and_measurement_batch(
                model, block
            )
            view_losses = normalized_batch_losses(
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
        "window_views": int(window_size),
        "microbatches": int(len(blocks)),
        "optimizer_steps": 0,
    }


def compare_parameter_gradients(model, expected_gradients):
    named_parameters = list(model.named_parameters())
    if len(expected_gradients) != len(named_parameters):
        return {
            "all_present": False,
            "all_finite": False,
            "relative_l2_by_parameter": None,
            "maximum_relative_l2": None,
            "maximum_absolute_error": None,
        }
    relative = {}
    maximum_absolute_error = 0.0
    all_finite = True
    for (name, parameter), expected in zip(named_parameters, expected_gradients):
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
            if not tensor_all_finite(current_chunk) or not tensor_all_finite(
                expected_chunk
            ):
                all_finite = False
                continue
            difference = current_chunk.to(torch.float64) - expected_chunk.to(
                torch.float64
            )
            difference_squared += float(difference.square().sum())
            expected_squared += float(expected_chunk.to(torch.float64).square().sum())
            if difference.numel():
                maximum_absolute_error = max(
                    maximum_absolute_error, float(difference.abs().max())
                )
        relative[name] = math.sqrt(difference_squared) / max(
            math.sqrt(expected_squared), 1.0e-30
        )
    return {
        "all_present": True,
        "all_finite": bool(all_finite),
        "relative_l2_by_parameter": relative,
        "maximum_relative_l2": max(relative.values()),
        "maximum_absolute_error": maximum_absolute_error,
    }


def tiny_cuda_equivalence_gate(trainer, device, degree, batch_size, progress):
    """Retain a cheap isolated scalar-vs-aligned CUDA physics gate."""

    set_stage(progress, "tiny_cuda_equivalence")
    torch.manual_seed(2900 + batch_size)
    model = trainer.PLANAR.FixedPlanarSHScene(
        5, 4, 0.4, degree, device, init_scale=0.02
    )
    dtheta = torch.linspace(0.25, 0.75, batch_size, device=device)
    dphi = torch.linspace(0.4, 1.6, batch_size, device=device)
    offsets = torch.linspace(
        -0.3, 0.3, batch_size, dtype=torch.float64, device=device
    )
    platforms = torch.stack(
        (7.0 + offsets, 4.0 - 0.5 * offsets, 3.0 + 0.25 * offsets), dim=-1
    )
    frequencies = torch.linspace(
        9.0e9, 9.2e9, 17, dtype=torch.float64, device=device
    )
    kvector = trainer.get_kvector(frequencies, trainer.cc)
    kwargs = {
        "phase_sign": -1.0,
        "freq_indices": torch.arange(frequencies.numel(), device=device),
        "compute_dtype": torch.float64,
        "range_model": "none",
        "propagation_model": "monostatic_near_field_reference",
        "reference_range_m": 7.5,
        "scene_center_m": (0.0, 0.0, 0.0),
    }

    def scalar_prediction():
        values = []
        for index in range(batch_size):
            platform = platforms[index].reshape(1, 3)
            theta = dtheta[index].reshape(1, 1)
            phi = dphi[index].reshape(1, 1)
            value = trainer.SERIAL.range_forward_operator_chunks(
                frequencies,
                kvector,
                platform,
                platform,
                lambda theta=theta, phi=phi: model.scatterer_chunks(
                    theta, phi, POINT_CHUNK
                ),
                pair_chunk=1,
                **kwargs,
            )
            values.append(value[:, 0, 0])
        return torch.stack(values)

    reference = scalar_prediction()
    batched = trainer.SERIAL.range_forward_operator_aligned_view_chunks(
        frequencies,
        kvector,
        platforms,
        platforms,
        lambda: model.scatterer_view_chunks(dtheta, dphi, POINT_CHUNK),
        **kwargs,
    )
    prediction_relative_l2 = finite_relative_l2(batched, reference)
    target = torch.complex(
        torch.randn_like(reference.real), torch.randn_like(reference.real)
    )
    model.zero_grad(set_to_none=True)
    reference_loss = (reference - target).abs().square().mean()
    reference_loss.backward()
    expected_gradients = [
        parameter.grad.detach().cpu().clone() for parameter in model.parameters()
    ]
    model.zero_grad(set_to_none=True)
    batched = trainer.SERIAL.range_forward_operator_aligned_view_chunks(
        frequencies,
        kvector,
        platforms,
        platforms,
        lambda: model.scatterer_view_chunks(dtheta, dphi, POINT_CHUNK),
        **kwargs,
    )
    batched_loss = (batched - target).abs().square().mean()
    batched_loss.backward()
    gradient_comparison = compare_parameter_gradients(model, expected_gradients)
    loss_relative_error = abs(float(batched_loss - reference_loss)) / max(
        abs(float(reference_loss)), 1.0e-30
    )
    all_finite = bool(
        prediction_relative_l2 is not None
        and math.isfinite(loss_relative_error)
        and gradient_comparison["all_finite"]
    )
    passed = bool(
        all_finite
        and prediction_relative_l2 <= TINY_FORWARD_RELATIVE_L2_TOLERANCE
        and loss_relative_error <= TINY_LOSS_RELATIVE_ERROR_TOLERANCE
        and gradient_comparison["maximum_relative_l2"]
        <= TINY_GRADIENT_RELATIVE_L2_TOLERANCE
    )
    result = {
        "batch_size": int(batch_size),
        "points": int(model.n_points),
        "frequencies": int(frequencies.numel()),
        "prediction_relative_l2": prediction_relative_l2,
        "loss_relative_error": loss_relative_error,
        "parameter_gradients": gradient_comparison,
        "all_finite": all_finite,
        "tolerances": {
            "prediction_relative_l2": TINY_FORWARD_RELATIVE_L2_TOLERANCE,
            "loss_relative_error": TINY_LOSS_RELATIVE_ERROR_TOLERANCE,
            "parameter_gradient_relative_l2": (
                TINY_GRADIENT_RELATIVE_L2_TOLERANCE
            ),
        },
        "passed": passed,
    }
    model.zero_grad(set_to_none=True)
    # Keep the closure inputs alive until return; deleting captured names here
    # makes static closure analysis ambiguous.  The tiny tensors are released
    # immediately when this function returns.
    del reference, batched, target, reference_loss, batched_loss, expected_gradients
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    update_cuda_peak(progress, device)
    return result


def select_windows(train_indices, dataset, window_size):
    required = (1 + MEASURED_WINDOWS) * window_size
    if len(dataset) < required:
        raise ValueError("GOTCHA training split is too small for benchmark windows")
    selected_local = np.linspace(
        0, len(dataset) - 1, num=required, dtype=np.int64
    )
    if np.unique(selected_local).size != required:
        raise ValueError("benchmark view selection contains duplicate local indices")
    local_windows = [selected_local[offset::4] for offset in range(4)]
    if any(window.size != window_size for window in local_windows):
        raise RuntimeError("benchmark window sizing changed")
    item_windows = [
        [dataset[int(local_index)] for local_index in window]
        for window in local_windows
    ]
    global_windows = [
        np.asarray(train_indices[window], dtype=np.int64) for window in local_windows
    ]
    flat_global = np.concatenate(global_windows)
    if np.unique(flat_global).size != required:
        raise RuntimeError("benchmark windows are not globally disjoint")
    return item_windows, local_windows, global_windows


def gotcha_support_identity(support):
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


def load_real_problem(args, device, progress):
    set_stage(progress, "load_integrated_trainer")
    trainer, trainer_path = load_trainer()
    set_stage(progress, "validate_full_domain_gotcha_contract")
    loaded = trainer.load_dataset_contract(args.npz_path, SCENE_KEY)
    arrays, metadata, partition = loaded[:3]
    support = loaded[3] if len(loaded) > 3 else None
    train_indices = np.asarray(arrays["train_indices"], dtype=np.int64)
    set_stage(progress, "materialize_real_training_dataset")
    dataset = trainer.PecSphereNPZDataset(arrays, train_indices)
    window_size = int(math.ceil(int(partition["train"]) / UPDATES_PER_EPOCH))
    if window_size != 19:
        raise ValueError(f"sealed GOTCHA production window changed: {window_size}")
    item_windows, local_windows, global_windows = select_windows(
        train_indices, dataset, window_size
    )
    set_stage(progress, "compute_fixed_training_target_power")
    target_mean_power, target_power_sum, target_sample_count = (
        trainer.V1.mean_train_target_power(arrays, train_indices)
    )
    set_stage(progress, f"allocate_production_degree{args.degree}_scene")
    model = trainer.make_scene(SCENE_KEY, args.degree, device)
    update_cuda_peak(progress, device)
    return {
        "trainer": trainer,
        "trainer_path": trainer_path,
        "arrays": arrays,
        "metadata": metadata,
        "partition": partition,
        "support": support,
        "train_indices": train_indices,
        "dataset": dataset,
        "window_size": window_size,
        "item_windows": item_windows,
        "local_windows": local_windows,
        "global_windows": global_windows,
        "target_mean_power": float(target_mean_power),
        "target_power_sum": float(target_power_sum),
        "target_sample_count": int(target_sample_count),
        "model": model,
    }


def full_window_equivalence_gate(problem, args, device, progress):
    """Production-size correctness gate; also warms both implementation paths."""

    trainer = problem["trainer"]
    model = problem["model"]
    renderer = trainer.SerializedRenderer(
        problem["arrays"],
        problem["metadata"],
        device,
        POINT_CHUNK,
        pair_chunk=PAIR_CHUNK,
    )
    if getattr(renderer, "sh_basis_cache", None) is not None:
        raise RuntimeError("speed benchmark must use live SH on both paths")
    warmup_items = problem["item_windows"][0]
    set_stage(progress, "full_window_former_scalar_correctness_warmup")
    reference_signature = initialize_nonzero_scene(model)
    reference = render_backward_window(
        model,
        renderer,
        warmup_items,
        problem["target_mean_power"],
        mode="former_scalar",
        view_batch_size=1,
        progress=progress,
        stage_prefix="full_window_former_scalar",
        capture_gradients=True,
    )
    expected_gradients = reference.pop("gradients")
    set_stage(progress, "full_window_aligned_correctness_warmup")
    candidate_signature = initialize_nonzero_scene(model)
    if candidate_signature != reference_signature:
        raise RuntimeError("former and aligned initial scenes are not identical")
    candidate = render_backward_window(
        model,
        renderer,
        warmup_items,
        problem["target_mean_power"],
        mode="aligned",
        view_batch_size=args.batch_size,
        progress=progress,
        stage_prefix="full_window_aligned",
        capture_gradients=False,
    )
    candidate.pop("gradients")
    gradient_comparison = compare_parameter_gradients(model, expected_gradients)
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
        and prediction_relative_l2 <= PREDICTION_RELATIVE_L2_TOLERANCE
        and view_loss_relative_l2 is not None
        and view_loss_relative_l2 <= LOSS_RELATIVE_ERROR_TOLERANCE
        and loss_relative_error is not None
        and loss_relative_error <= LOSS_RELATIVE_ERROR_TOLERANCE
        and gradient_comparison["maximum_relative_l2"] is not None
        and gradient_comparison["maximum_relative_l2"]
        <= GRADIENT_RELATIVE_L2_TOLERANCE
    )
    result = {
        "role": "production_correctness_and_disjoint_warmup_window",
        "former_path": "raw_prediction_and_measurement_per_view_live_sh",
        "new_path": "raw_prediction_and_measurement_batch_aligned_live_sh",
        "window_views": int(len(warmup_items)),
        "former_microbatches": int(reference["microbatches"]),
        "new_microbatches": int(candidate["microbatches"]),
        "same_initial_scene": True,
        "initialization_signature": reference_signature,
        "same_measurement_exact": bool(measurement_equal),
        "prediction_relative_l2": prediction_relative_l2,
        "view_loss_relative_l2": view_loss_relative_l2,
        "mean_loss_relative_error": loss_relative_error,
        "parameter_gradients": gradient_comparison,
        "all_finite": all_finite,
        "tolerances": {
            "prediction_relative_l2": PREDICTION_RELATIVE_L2_TOLERANCE,
            "loss_relative_error": LOSS_RELATIVE_ERROR_TOLERANCE,
            "parameter_gradient_relative_l2": GRADIENT_RELATIVE_L2_TOLERANCE,
        },
        "passed": passed,
    }
    model.zero_grad(set_to_none=True)
    del renderer, reference, candidate, expected_gradients
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    update_cuda_peak(progress, device)
    return result


def timed_training_window(
    model,
    optimizer,
    renderer,
    items,
    target_mean_power,
    *,
    mode,
    view_batch_size,
    progress,
    stage_prefix,
):
    """Time one exact 19-view optimizer update."""

    torch.cuda.synchronize()
    total_start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    window_size = len(items)
    if mode == "former_scalar":
        blocks = [[item] for item in items]
    elif mode == "aligned":
        blocks = list(item_microbatches(items, view_batch_size))
    else:
        raise ValueError(f"unknown renderer mode {mode!r}")
    forward_seconds = 0.0
    backward_seconds = 0.0
    view_losses_values = []
    for block_index, block in enumerate(blocks):
        set_stage(
            progress,
            f"{stage_prefix}_microbatch_{block_index + 1}_of_{len(blocks)}",
        )
        forward_start = time.perf_counter()
        if mode == "former_scalar":
            predicted, measured = renderer.raw_prediction_and_measurement(
                model, block[0]
            )
            view_losses = normalized_scalar_loss(
                predicted, measured, target_mean_power
            ).reshape(1)
        else:
            predicted, measured = renderer.raw_prediction_and_measurement_batch(
                model, block
            )
            view_losses = normalized_batch_losses(
                predicted, measured, target_mean_power
            )
        torch.cuda.synchronize()
        forward_seconds += time.perf_counter() - forward_start

        backward_start = time.perf_counter()
        (view_losses.sum() / window_size).backward()
        torch.cuda.synchronize()
        backward_seconds += time.perf_counter() - backward_start
        view_losses_values.extend(
            finite_json_float(value) for value in view_losses.detach().cpu()
        )

    set_stage(progress, f"{stage_prefix}_optimizer_step")
    optimizer_start = time.perf_counter()
    optimizer.step()
    torch.cuda.synchronize()
    optimizer_seconds = time.perf_counter() - optimizer_start
    total_seconds = time.perf_counter() - total_start
    return {
        "window_views": int(window_size),
        "microbatches": int(len(blocks)),
        "optimizer_steps": 1,
        "loss": (
            sum(value for value in view_losses_values if value is not None)
            / window_size
            if all(value is not None for value in view_losses_values)
            else None
        ),
        "view_losses": view_losses_values,
        "loss_finite": all(value is not None for value in view_losses_values),
        "forward_seconds": float(forward_seconds),
        "backward_seconds": float(backward_seconds),
        "optimizer_seconds": float(optimizer_seconds),
        "total_seconds": float(total_seconds),
        "per_view_forward_seconds": float(forward_seconds / window_size),
        "per_view_backward_seconds": float(backward_seconds / window_size),
        "per_view_total_seconds": float(total_seconds / window_size),
    }


def timing_summary(steps):
    individual = {}
    medians = {}
    coefficients_of_variation = {}
    per_view_medians = {}
    for component in ("forward", "backward", "optimizer", "total"):
        key = f"{component}_seconds"
        values = [float(step[key]) for step in steps]
        mean = statistics.fmean(values)
        individual[component] = values
        medians[component] = float(statistics.median(values))
        coefficients_of_variation[component] = (
            float(statistics.pstdev(values) / mean) if mean > 0.0 else None
        )
        if component != "optimizer":
            per_view_medians[component] = float(
                statistics.median(
                    step[f"per_view_{component}_seconds"] for step in steps
                )
            )
    return {
        "individual_window_seconds": individual,
        "median_window_seconds": medians,
        "median_per_view_seconds": per_view_medians,
        "coefficient_of_variation": coefficients_of_variation,
        "noisy_cv_limit": NOISY_CV_LIMIT,
    }


def timing_valid(steps, summary, expected_microbatches):
    if len(steps) != MEASURED_WINDOWS:
        return False
    for step in steps:
        if (
            step["window_views"] != 19
            or step["microbatches"] != expected_microbatches
            or step["optimizer_steps"] != 1
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
        and math.isfinite(summary["median_window_seconds"][component])
        and summary["median_window_seconds"][component] > 0.0
        for component in ("forward", "backward", "optimizer", "total")
    )


def measure_phase(
    *,
    label,
    mode,
    view_batch_size,
    problem,
    device,
    progress,
):
    """Reset model/Adam identically, then time three disjoint windows."""

    trainer = problem["trainer"]
    model = problem["model"]
    set_stage(progress, f"{label}_identical_model_reset")
    initialization_signature = initialize_nonzero_scene(model)
    renderer = trainer.SerializedRenderer(
        problem["arrays"],
        problem["metadata"],
        device,
        POINT_CHUNK,
        pair_chunk=PAIR_CHUNK,
    )
    if getattr(renderer, "sh_basis_cache", None) is not None:
        raise RuntimeError("speed benchmark must use live SH on both paths")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3.0e-5, eps=1.0e-15, weight_decay=0.0
    )
    if optimizer.state:
        raise RuntimeError("new AdamW optimizer unexpectedly has state")

    set_stage(progress, f"{label}_unmeasured_state_allocation_warmup")
    warmup = timed_training_window(
        model,
        optimizer,
        renderer,
        problem["item_windows"][0],
        problem["target_mean_power"],
        mode=mode,
        view_batch_size=view_batch_size,
        progress=progress,
        stage_prefix=f"{label}_unmeasured_warmup",
    )
    if not warmup["loss_finite"] or not model_parameters_finite(model):
        raise RuntimeError(f"{label} warmup produced a non-finite value")
    update_cuda_peak(progress, device)
    replay_signature = initialize_nonzero_scene(model)
    if replay_signature != initialization_signature:
        raise RuntimeError(f"{label} model reset after warmup is not exact")
    adam_state_schema = reset_allocated_adamw_state(optimizer)
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    free_before, total_before = torch.cuda.mem_get_info(device)
    baseline_allocated = int(torch.cuda.memory_allocated(device))
    baseline_reserved = int(torch.cuda.memory_reserved(device))
    host_before = process_memory_kib()
    # The frozen PublicRadar PyTorch 2.6 build rejects a ``torch.device``
    # argument for this helper.  Exactly one CUDA device is required above, so
    # the argument-free form is unambiguous and keeps the memory gate intact.
    torch.cuda.reset_peak_memory_stats()
    steps = []
    for repetition, items in enumerate(problem["item_windows"][1:], start=1):
        steps.append(
            timed_training_window(
                model,
                optimizer,
                renderer,
                items,
                problem["target_mean_power"],
                mode=mode,
                view_batch_size=view_batch_size,
                progress=progress,
                stage_prefix=f"{label}_measured_window_{repetition}",
            )
        )
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    update_cuda_peak(progress, device)
    free_after, total_after = torch.cuda.mem_get_info(device)
    host_after = process_memory_kib()
    set_stage(progress, f"{label}_finite_and_timing_checks")
    timing = timing_summary(steps)
    expected_microbatches = (
        19 if mode == "former_scalar" else math.ceil(19 / view_batch_size)
    )
    valid = timing_valid(steps, timing, expected_microbatches)
    finite_checks = {
        "losses": bool(all(step["loss_finite"] for step in steps)),
        "gradients": bool(model_gradients_finite(model)),
        "model_parameters": bool(model_parameters_finite(model)),
        "adam_state": bool(optimizer_state_finite(optimizer)),
    }
    result = {
        "label": label,
        "implementation": mode,
        "live_sh": True,
        "sh_basis_cache": None,
        "view_batch_size": int(view_batch_size),
        "point_chunk": POINT_CHUNK,
        "initialization_signature": initialization_signature,
        "unmeasured_state_allocation_warmup": warmup,
        "optimizer_state_entries_before_measurement": int(len(optimizer.state)),
        "optimizer_state_schema": adam_state_schema,
        "optimizer_state_reset_to_fresh_zero_before_measurement": True,
        "measured_windows": steps,
        "timing": timing,
        "timing_valid": bool(valid),
        "finite_checks": finite_checks,
        "numerical_finite": bool(all(finite_checks.values())),
        "cuda_baseline_allocated_bytes": baseline_allocated,
        "cuda_baseline_reserved_bytes": baseline_reserved,
        "cuda_peak_allocated_bytes": peak_allocated,
        "cuda_peak_reserved_bytes": peak_reserved,
        "cuda_incremental_peak_allocated_bytes": (
            peak_allocated - baseline_allocated
        ),
        "cuda_incremental_peak_reserved_bytes": peak_reserved - baseline_reserved,
        "cuda_free_before_bytes": int(free_before),
        "cuda_free_after_bytes": int(free_after),
        "cuda_mem_get_info_total_before_bytes": int(total_before),
        "cuda_mem_get_info_total_after_bytes": int(total_after),
        "host_before_kib": host_before,
        "host_after_kib": host_after,
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


def paired_speedups(former, aligned):
    paired = {}
    paired_medians = {}
    median_time = {}
    for component in ("forward", "backward", "optimizer", "total"):
        former_values = former["timing"]["individual_window_seconds"][component]
        aligned_values = aligned["timing"]["individual_window_seconds"][component]
        values = [
            old / new for old, new in zip(former_values, aligned_values)
        ]
        paired[component] = values
        paired_medians[component] = float(statistics.median(values))
        median_time[component] = float(
            former["timing"]["median_window_seconds"][component]
            / aligned["timing"]["median_window_seconds"][component]
        )
    return {
        "definition": "former_scalar_seconds_divided_by_aligned_seconds",
        "paired_same_view_window_speedups": paired,
        "median_paired_same_view_window_speedup": paired_medians,
        "ratio_of_median_window_times": median_time,
    }


def epoch_projection(phase, train_views):
    medians = phase["timing"]["median_window_seconds"]
    window_views = 19
    unassigned_overhead = max(
        0.0,
        medians["total"]
        - medians["forward"]
        - medians["backward"]
        - medians["optimizer"],
    )
    view_scaled_epoch_seconds = (
        (medians["forward"] + medians["backward"])
        / window_views
        * train_views
        + (medians["optimizer"] + unassigned_overhead) * UPDATES_PER_EPOCH
    )
    worst_case_epoch_seconds = medians["total"] * UPDATES_PER_EPOCH
    return {
        "basis": (
            "training_only_median_compute_scaled_to_all_train_views_plus_"
            "1800_optimizer_steps; excludes_BP_evaluation_checkpoint_and_queue_time"
        ),
        "train_views": int(train_views),
        "updates_per_epoch": UPDATES_PER_EPOCH,
        "average_views_per_update": float(train_views / UPDATES_PER_EPOCH),
        "benchmark_window_views": window_views,
        "median_seconds_per_19_view_update": float(medians["total"]),
        "estimated_epoch_seconds_view_scaled": float(view_scaled_epoch_seconds),
        "estimated_epoch_hours_view_scaled": float(view_scaled_epoch_seconds / 3600),
        "estimated_15_epoch_hours_view_scaled": float(
            view_scaled_epoch_seconds * 15 / 3600
        ),
        "worst_case_19_view_updates_epoch_seconds": float(
            worst_case_epoch_seconds
        ),
        "worst_case_19_view_updates_epoch_hours": float(
            worst_case_epoch_seconds / 3600
        ),
    }


def benchmark(args, device, progress, started):
    problem = load_real_problem(args, device, progress)
    trainer = problem["trainer"]
    model = problem["model"]
    tiny = tiny_cuda_equivalence_gate(
        trainer, device, args.degree, args.batch_size, progress
    )

    scene_config = trainer.SCENES[SCENE_KEY]
    pitch_x, pitch_y = model.pitch_xy
    common = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "scene": SCENE_KEY,
        "dataset_name": scene_config["dataset_name"],
        "dataset": str(Path(args.npz_path).resolve()),
        "trainer_source": str(problem["trainer_path"].resolve()),
        "degree": int(args.degree),
        "batch_memory_safety_selector": bool(args.degree == 3),
        "batch_memory_safety_policy": (
            "degree3_is_the_worst_case_selector; degree0_is_timing_evidence_only"
        ),
        "point_chunk": POINT_CHUNK,
        "requested_aligned_batch_size": int(args.batch_size),
        "former_batch_size": 1,
        "sh_basis_source": "live_real_sh_basis_on_both_paths",
        "sh_basis_cache": None,
        "production_updates_per_epoch": UPDATES_PER_EPOCH,
        "production_window_views": int(problem["window_size"]),
        "measured_disjoint_windows": MEASURED_WINDOWS,
        "window_loss_reduction": (
            "sum_per_view_normalized_losses_divided_by_full_19_view_window"
        ),
        "frozen_gain": [1.0, 0.0],
        "points": int(model.n_points),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "parameter_dtype": str(model.w_re.dtype),
        "scene_geometry": {
            "shape": list(model.shape),
            "extent_xy_m": float(scene_config["extent"]),
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
                float(value) for value in problem["metadata"]["scene_center_m"]
            ],
            "phase_sign": -1.0,
        },
        "gotcha_support_preflight": gotcha_support_identity(problem["support"]),
        "frequencies": int(problem["arrays"]["frequencies_hz"].size),
        "train_views": int(problem["partition"]["train"]),
        "target_mean_power": problem["target_mean_power"],
        "target_power_sum": problem["target_power_sum"],
        "target_sample_count": problem["target_sample_count"],
        "view_windows": {
            "correctness_and_warmup": {
                "local_training_indices": problem["local_windows"][0].tolist(),
                "canonical_global_view_ids": problem["global_windows"][0].tolist(),
            },
            "measured_disjoint": [
                {
                    "repetition": index,
                    "local_training_indices": local.tolist(),
                    "canonical_global_view_ids": global_ids.tolist(),
                }
                for index, (local, global_ids) in enumerate(
                    zip(
                        problem["local_windows"][1:],
                        problem["global_windows"][1:],
                    ),
                    start=1,
                )
            ],
        },
        "device": torch.cuda.get_device_name(device),
        "cuda_total_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        "cuda_cap_bytes": CUDA_CAP_BYTES,
        "host_cap_kib": HOST_CAP_KIB,
        "host_cap_scope": (
            "whole_process_VmHWM_including_dataset_correctness_warmup_and_"
            "both_timing_phases"
        ),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "tiny_cuda_equivalence_gate": tiny,
    }

    if not tiny["passed"]:
        set_stage(progress, "tiny_cuda_equivalence_failure")
        update_cuda_peak(progress, device)
        host_after = process_memory_kib()
        runtime_seconds = float(time.perf_counter() - started)
        return {
            **common,
            "status": "unsafe" if tiny["all_finite"] else "nonfinite",
            "stage": progress["stage"],
            "measurement_completed": False,
            "runtime_seconds": runtime_seconds,
            "runtime_nonzero": bool(runtime_seconds > 0.0),
            "memory_safe": None,
            "unsafe_reasons": ["tiny_cuda_equivalence_gate"],
            "cuda_whole_process_peak_allocated_bytes": progress[
                "cuda_peak_allocated_bytes"
            ],
            "cuda_whole_process_peak_reserved_bytes": progress[
                "cuda_peak_reserved_bytes"
            ],
            "host_after_kib": host_after,
            "host_VmHWM_kib": int(host_after["VmHWM"]),
            "completed_unix": time.time(),
        }

    correctness = full_window_equivalence_gate(
        problem, args, device, progress
    )
    if not correctness["passed"]:
        set_stage(progress, "full_window_equivalence_failure")
        update_cuda_peak(progress, device)
        host_after = process_memory_kib()
        return {
            **common,
            "status": "unsafe" if correctness["all_finite"] else "nonfinite",
            "stage": progress["stage"],
            "measurement_completed": False,
            "full_window_equivalence_gate": correctness,
            "runtime_seconds": float(time.perf_counter() - started),
            "runtime_nonzero": bool(time.perf_counter() - started > 0.0),
            "cuda_whole_process_peak_allocated_bytes": progress[
                "cuda_peak_allocated_bytes"
            ],
            "cuda_whole_process_peak_reserved_bytes": progress[
                "cuda_peak_reserved_bytes"
            ],
            "host_after_kib": host_after,
            "host_VmHWM_kib": int(host_after["VmHWM"]),
            "completed_unix": time.time(),
        }

    former = measure_phase(
        label="former_scalar_live_sh",
        mode="former_scalar",
        view_batch_size=1,
        problem=problem,
        device=device,
        progress=progress,
    )
    aligned = measure_phase(
        label=f"aligned_b{args.batch_size}_live_sh",
        mode="aligned",
        view_batch_size=args.batch_size,
        problem=problem,
        device=device,
        progress=progress,
    )
    if former["initialization_signature"] != aligned["initialization_signature"]:
        raise RuntimeError("timed former and aligned phases did not start identically")

    set_stage(progress, "final_engineering_checks")
    update_cuda_peak(progress, device)
    host_after = process_memory_kib()
    host_peak_kib = int(host_after["VmHWM"])
    timing_ok = bool(former["timing_valid"] and aligned["timing_valid"])
    numerical_finite = bool(
        correctness["all_finite"]
        and former["numerical_finite"]
        and aligned["numerical_finite"]
    )
    memory_safe = bool(
        progress["cuda_peak_allocated_bytes"] <= CUDA_CAP_BYTES
        and progress["cuda_peak_reserved_bytes"] <= CUDA_CAP_BYTES
        and former["cuda_peak_allocated_bytes"] <= CUDA_CAP_BYTES
        and former["cuda_peak_reserved_bytes"] <= CUDA_CAP_BYTES
        and aligned["cuda_peak_allocated_bytes"] <= CUDA_CAP_BYTES
        and aligned["cuda_peak_reserved_bytes"] <= CUDA_CAP_BYTES
        and host_peak_kib <= HOST_CAP_KIB
    )
    unsafe_reasons = []
    if not timing_ok:
        unsafe_reasons.append("timing_invalid")
    if not correctness["passed"]:
        unsafe_reasons.append("full_window_equivalence")
    if progress["cuda_peak_allocated_bytes"] > CUDA_CAP_BYTES:
        unsafe_reasons.append("cuda_whole_process_peak_allocated")
    if progress["cuda_peak_reserved_bytes"] > CUDA_CAP_BYTES:
        unsafe_reasons.append("cuda_whole_process_peak_reserved")
    if former["cuda_peak_allocated_bytes"] > CUDA_CAP_BYTES:
        unsafe_reasons.append("former_cuda_peak_allocated")
    if former["cuda_peak_reserved_bytes"] > CUDA_CAP_BYTES:
        unsafe_reasons.append("former_cuda_peak_reserved")
    if aligned["cuda_peak_allocated_bytes"] > CUDA_CAP_BYTES:
        unsafe_reasons.append("aligned_cuda_peak_allocated")
    if aligned["cuda_peak_reserved_bytes"] > CUDA_CAP_BYTES:
        unsafe_reasons.append("aligned_cuda_peak_reserved")
    if host_peak_kib > HOST_CAP_KIB:
        unsafe_reasons.append("host_whole_process_VmHWM")
    noisy_components = []
    for phase_name, phase in (("former", former), ("aligned", aligned)):
        for component in ("forward", "backward", "total"):
            coefficient = phase["timing"]["coefficient_of_variation"][component]
            if coefficient is not None and coefficient > NOISY_CV_LIMIT:
                noisy_components.append(f"{phase_name}.{component}")
    speedup = paired_speedups(former, aligned) if timing_ok else None
    projections = None
    if timing_ok:
        projections = {
            "former_scalar": epoch_projection(
                former, int(problem["partition"]["train"])
            ),
            "aligned": epoch_projection(
                aligned, int(problem["partition"]["train"])
            ),
            "scope_warning": (
                f"compute-only degree-{args.degree} estimate; excludes queue, BP initialization, "
                "checkpoint I/O, train/validation evaluation, and interruption"
            ),
        }
    runtime_seconds = float(time.perf_counter() - started)
    if not numerical_finite:
        status = "nonfinite"
    elif unsafe_reasons or not memory_safe:
        status = "unsafe"
    elif noisy_components:
        status = "noisy"
    else:
        status = "pass"
    set_stage(progress, "complete")
    return {
        **common,
        "status": status,
        "stage": progress["stage"],
        "measurement_completed": True,
        "runtime_seconds": runtime_seconds,
        "runtime_nonzero": bool(runtime_seconds > 0.0),
        "full_window_equivalence_gate": correctness,
        "former_scalar": former,
        "aligned": aligned,
        "paired_speedup": speedup,
        "training_time_projection": projections,
        "timing_valid": timing_ok,
        "noisy_components": noisy_components,
        "numerical_finite": numerical_finite,
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
        "completed_unix": time.time(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--degree", required=True, type=int, choices=(0, 3))
    parser.add_argument(
        "--batch-size", required=True, type=int, choices=VIEW_BATCH_SIZES
    )
    args = parser.parse_args()
    result_path = (
        Path(args.run_root).resolve()
        / f"gotcha_full_deg{args.degree}_b{args.batch_size}.json"
    )
    if result_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing benchmark: {result_path}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("view-batch speed benchmark requires exactly one CUDA device")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    progress = {
        "stage": "benchmark_start",
        "cuda_peak_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }
    common = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "scene": SCENE_KEY,
        "degree": int(args.degree),
        "point_chunk": POINT_CHUNK,
        "requested_aligned_batch_size": int(args.batch_size),
        "cuda_cap_bytes": CUDA_CAP_BYTES,
        "host_cap_kib": HOST_CAP_KIB,
    }
    unexpected_error = None
    try:
        payload = benchmark(args, device, progress, started)
    except torch.cuda.OutOfMemoryError as exc:
        update_cuda_peak(progress, device)
        host_after = process_memory_kib()
        runtime_seconds = float(time.perf_counter() - started)
        payload = {
            **common,
            "status": "oom",
            "stage": progress["stage"],
            "measurement_completed": False,
            "runtime_seconds": runtime_seconds,
            "runtime_nonzero": bool(runtime_seconds > 0.0),
            "memory_safe": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "cuda_whole_process_peak_allocated_bytes": progress[
                "cuda_peak_allocated_bytes"
            ],
            "cuda_whole_process_peak_reserved_bytes": progress[
                "cuda_peak_reserved_bytes"
            ],
            "host_after_kib": host_after,
            "host_VmHWM_kib": int(host_after["VmHWM"]),
            "completed_unix": time.time(),
        }
        torch.cuda.empty_cache()
    except Exception as exc:
        update_cuda_peak(progress, device)
        host_after = process_memory_kib()
        runtime_seconds = float(time.perf_counter() - started)
        payload = {
            **common,
            "status": "error",
            "stage": progress["stage"],
            "measurement_completed": False,
            "runtime_seconds": runtime_seconds,
            "runtime_nonzero": bool(runtime_seconds > 0.0),
            "memory_safe": None,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "cuda_whole_process_peak_allocated_bytes": progress[
                "cuda_peak_allocated_bytes"
            ],
            "cuda_whole_process_peak_reserved_bytes": progress[
                "cuda_peak_reserved_bytes"
            ],
            "host_after_kib": host_after,
            "host_VmHWM_kib": int(host_after["VmHWM"]),
            "completed_unix": time.time(),
        }
        unexpected_error = exc
    atomic_json(result_path, payload)
    print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)
    if unexpected_error is not None:
        raise RuntimeError(
            "view-batch speed benchmark failed after publishing its error artifact"
        ) from unexpected_error


if __name__ == "__main__":
    main()
