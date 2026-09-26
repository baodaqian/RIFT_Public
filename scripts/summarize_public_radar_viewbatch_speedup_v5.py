#!/usr/bin/env python
"""Fail-closed audit of the sealed PublicRadar V5 viewpoint-batch matrix.

The input directory must contain exactly the eight immutable GOTCHA V5
artifacts (degree 0/3 crossed with B=1/2/4/8).  This reader recomputes the
engineering gates, paired speedups, and compute-only 15-epoch projections;
it does not trust the benchmark's summary fields and never modifies a run
artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import statistics
import uuid


SOURCE_SCHEMA = "rift.public_radar.viewbatch_speedup_v5_inferno"
SOURCE_SCHEMA_VERSION = 5
SUMMARY_SCHEMA = "rift.public_radar.viewbatch_speedup_v5_summary"
SUMMARY_SCHEMA_VERSION = 1
SCENE = "gotcha_p2_full_domain"
DATASET_NAME = "gotcha_pass2_hh"
DEGREES = (0, 3)
BATCH_SIZES = (1, 2, 4, 8)
POINT_CHUNK = 32768
WINDOW_VIEWS = 19
MEASURED_WINDOWS = 3
UPDATES_PER_EPOCH = 1800
CUDA_CAP_BYTES = 12 * 1024**3
HOST_CAP_KIB = 24 * 1024**2
NOISY_CV_LIMIT = 0.10
TINY_PREDICTION_TOLERANCE = 2.0e-6
TINY_LOSS_TOLERANCE = 2.0e-6
TINY_GRADIENT_TOLERANCE = 5.0e-6
FULL_PREDICTION_TOLERANCE = 2.0e-6
FULL_LOSS_TOLERANCE = 2.0e-6
FULL_GRADIENT_TOLERANCE = 5.0e-6
INITIALIZATION_SEED = 420_091
COMPONENTS = ("forward", "backward", "optimizer", "total")
ARTIFACT_RE = re.compile(r"^gotcha_full_deg(0|3)_b(1|2|4|8)\.json$")


class AuditError(ValueError):
    """The V5 matrix is incomplete, inconsistent, or does not pass."""


def fail(message):
    raise AuditError(message)


def require_mapping(mapping, key, where):
    value = mapping.get(key)
    if not isinstance(value, dict):
        fail(f"{where}.{key} must be an object")
    return value


def require_list(mapping, key, where, *, length=None):
    value = mapping.get(key)
    if not isinstance(value, list):
        fail(f"{where}.{key} must be an array")
    if length is not None and len(value) != length:
        fail(f"{where}.{key} must contain exactly {length} entries")
    return value


def exact_bool(mapping, key, expected, where):
    value = mapping.get(key)
    if type(value) is not bool or value is not expected:
        fail(f"{where}.{key} must be exactly {expected!r}")
    return value


def exact_int(mapping, key, expected, where):
    value = mapping.get(key)
    if type(value) is not int or value != expected:
        fail(f"{where}.{key} must be exactly {expected!r}")
    return value


def positive_int(mapping, key, where):
    value = mapping.get(key)
    if type(value) is not int or value <= 0:
        fail(f"{where}.{key} must be a positive integer")
    return value


def nonnegative_int(mapping, key, where):
    value = mapping.get(key)
    if type(value) is not int or value < 0:
        fail(f"{where}.{key} must be a nonnegative integer")
    return value


def finite_number(value, where, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{where} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        fail(f"{where} must be finite")
    if positive and value <= 0.0:
        fail(f"{where} must be positive")
    if nonnegative and value < 0.0:
        fail(f"{where} must be nonnegative")
    return value


def close_number(actual, expected, where):
    actual = finite_number(actual, where)
    expected = finite_number(expected, f"{where} (recomputed)")
    if not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
        fail(f"{where} disagrees with recomputed evidence")


def canonical_json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        fail(f"identity is not canonical JSON: {exc}")


def load_json(path):
    def reject_constant(value):
        fail(f"{path} contains non-standard JSON constant {value!r}")

    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle, parse_constant=reject_constant)
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read benchmark artifact {path}: {exc}")
    if not isinstance(payload, dict):
        fail(f"benchmark artifact {path} must contain one JSON object")
    return payload


def discover_artifacts(run_root):
    root = Path(run_root).resolve()
    if not root.is_dir():
        fail(f"benchmark run root is not a directory: {root}")
    json_paths = sorted(path for path in root.iterdir() if path.is_file() and path.suffix == ".json")
    found = {}
    for path in json_paths:
        match = ARTIFACT_RE.fullmatch(path.name)
        if match is None:
            fail(f"unexpected JSON artifact in sealed V5 run root: {path.name}")
        key = (int(match.group(1)), int(match.group(2)))
        if key in found:
            fail(f"duplicate V5 artifact identity {key}")
        found[key] = path
    expected = {(degree, batch) for degree in DEGREES for batch in BATCH_SIZES}
    if set(found) != expected or len(json_paths) != 8:
        fail(
            "V5 run root must contain exactly eight canonical artifacts; "
            f"missing={sorted(expected - set(found))}, "
            f"unexpected={sorted(set(found) - expected)}, found={len(json_paths)}"
        )
    return root, found


def validate_gradient_comparison(mapping, where, tolerance):
    exact_bool(mapping, "all_present", True, where)
    exact_bool(mapping, "all_finite", True, where)
    relative = require_mapping(mapping, "relative_l2_by_parameter", where)
    if set(relative) != {"w_re", "w_im"}:
        fail(f"{where}.relative_l2_by_parameter must contain w_re and w_im")
    values = [
        finite_number(relative[name], f"{where}.relative_l2_by_parameter.{name}", nonnegative=True)
        for name in ("w_re", "w_im")
    ]
    maximum = finite_number(
        mapping.get("maximum_relative_l2"),
        f"{where}.maximum_relative_l2",
        nonnegative=True,
    )
    close_number(maximum, max(values), f"{where}.maximum_relative_l2")
    if maximum > tolerance:
        fail(f"{where}.maximum_relative_l2 exceeds {tolerance}")
    finite_number(
        mapping.get("maximum_absolute_error"),
        f"{where}.maximum_absolute_error",
        nonnegative=True,
    )


def validate_tolerances(mapping, where, prediction, loss, gradient):
    close_number(mapping.get("prediction_relative_l2"), prediction, f"{where}.prediction_relative_l2")
    close_number(mapping.get("loss_relative_error"), loss, f"{where}.loss_relative_error")
    close_number(
        mapping.get("parameter_gradient_relative_l2"),
        gradient,
        f"{where}.parameter_gradient_relative_l2",
    )


def validate_tiny_gate(payload, where, batch_size):
    gate = require_mapping(payload, "tiny_cuda_equivalence_gate", where)
    gate_where = f"{where}.tiny_cuda_equivalence_gate"
    exact_int(gate, "batch_size", batch_size, gate_where)
    exact_int(gate, "points", 20, gate_where)
    exact_int(gate, "frequencies", 17, gate_where)
    tolerances = require_mapping(gate, "tolerances", gate_where)
    validate_tolerances(
        tolerances,
        f"{gate_where}.tolerances",
        TINY_PREDICTION_TOLERANCE,
        TINY_LOSS_TOLERANCE,
        TINY_GRADIENT_TOLERANCE,
    )
    prediction = finite_number(
        gate.get("prediction_relative_l2"),
        f"{gate_where}.prediction_relative_l2",
        nonnegative=True,
    )
    loss = finite_number(
        gate.get("loss_relative_error"),
        f"{gate_where}.loss_relative_error",
        nonnegative=True,
    )
    if prediction > TINY_PREDICTION_TOLERANCE or loss > TINY_LOSS_TOLERANCE:
        fail(f"{gate_where} exceeds its forward/loss tolerance")
    validate_gradient_comparison(
        require_mapping(gate, "parameter_gradients", gate_where),
        f"{gate_where}.parameter_gradients",
        TINY_GRADIENT_TOLERANCE,
    )
    exact_bool(gate, "all_finite", True, gate_where)
    exact_bool(gate, "passed", True, gate_where)


def validate_initialization_signature(mapping, where, degree):
    exact_int(mapping, "seed", INITIALIZATION_SEED + degree, where)
    expected_std = 1.0e-3 / math.sqrt(2.0 * (degree + 1) ** 2)
    close_number(mapping.get("coefficient_std"), expected_std, f"{where}.coefficient_std")
    for key in ("w_re_head", "w_im_head"):
        values = require_list(mapping, key, where, length=8)
        for index, value in enumerate(values):
            finite_number(value, f"{where}.{key}[{index}]")
    return canonical_json(mapping)


def validate_full_gate(payload, where, degree, batch_size):
    gate = require_mapping(payload, "full_window_equivalence_gate", where)
    gate_where = f"{where}.full_window_equivalence_gate"
    if gate.get("role") != "production_correctness_and_disjoint_warmup_window":
        fail(f"{gate_where}.role is invalid")
    if gate.get("former_path") != "raw_prediction_and_measurement_per_view_live_sh":
        fail(f"{gate_where}.former_path is invalid")
    if gate.get("new_path") != "raw_prediction_and_measurement_batch_aligned_live_sh":
        fail(f"{gate_where}.new_path is invalid")
    exact_int(gate, "window_views", WINDOW_VIEWS, gate_where)
    exact_int(gate, "former_microbatches", WINDOW_VIEWS, gate_where)
    exact_int(gate, "new_microbatches", math.ceil(WINDOW_VIEWS / batch_size), gate_where)
    exact_bool(gate, "same_initial_scene", True, gate_where)
    signature = validate_initialization_signature(
        require_mapping(gate, "initialization_signature", gate_where),
        f"{gate_where}.initialization_signature",
        degree,
    )
    exact_bool(gate, "same_measurement_exact", True, gate_where)
    tolerances = require_mapping(gate, "tolerances", gate_where)
    validate_tolerances(
        tolerances,
        f"{gate_where}.tolerances",
        FULL_PREDICTION_TOLERANCE,
        FULL_LOSS_TOLERANCE,
        FULL_GRADIENT_TOLERANCE,
    )
    for key, tolerance in (
        ("prediction_relative_l2", FULL_PREDICTION_TOLERANCE),
        ("view_loss_relative_l2", FULL_LOSS_TOLERANCE),
        ("mean_loss_relative_error", FULL_LOSS_TOLERANCE),
    ):
        value = finite_number(gate.get(key), f"{gate_where}.{key}", nonnegative=True)
        if value > tolerance:
            fail(f"{gate_where}.{key} exceeds {tolerance}")
    validate_gradient_comparison(
        require_mapping(gate, "parameter_gradients", gate_where),
        f"{gate_where}.parameter_gradients",
        FULL_GRADIENT_TOLERANCE,
    )
    exact_bool(gate, "all_finite", True, gate_where)
    exact_bool(gate, "passed", True, gate_where)
    return signature


def validate_step(step, where, microbatches):
    if not isinstance(step, dict):
        fail(f"{where} must be an object")
    exact_int(step, "window_views", WINDOW_VIEWS, where)
    exact_int(step, "microbatches", microbatches, where)
    exact_int(step, "optimizer_steps", 1, where)
    exact_bool(step, "loss_finite", True, where)
    losses = require_list(step, "view_losses", where, length=WINDOW_VIEWS)
    losses = [
        finite_number(value, f"{where}.view_losses[{index}]")
        for index, value in enumerate(losses)
    ]
    loss = finite_number(step.get("loss"), f"{where}.loss")
    close_number(loss, statistics.fmean(losses), f"{where}.loss")
    components = {}
    for component in COMPONENTS:
        components[component] = finite_number(
            step.get(f"{component}_seconds"),
            f"{where}.{component}_seconds",
            positive=True,
        )
    if components["total"] + 1.0e-6 < sum(
        components[name] for name in ("forward", "backward", "optimizer")
    ):
        fail(f"{where}.total_seconds is shorter than its components")
    for component in ("forward", "backward", "total"):
        close_number(
            step.get(f"per_view_{component}_seconds"),
            components[component] / WINDOW_VIEWS,
            f"{where}.per_view_{component}_seconds",
        )
    return components


def validate_process_memory(mapping, where):
    values = {}
    for key in ("VmRSS", "VmHWM", "ru_maxrss"):
        values[key] = nonnegative_int(mapping, key, where)
    if values["VmHWM"] < values["VmRSS"]:
        fail(f"{where}.VmHWM is smaller than VmRSS")
    return values


def validate_phase(payload, where, *, label, implementation, batch_size, degree, cuda_total):
    if not isinstance(payload, dict):
        fail(f"{where} must be an object")
    if payload.get("label") != label or payload.get("implementation") != implementation:
        fail(f"{where} has the wrong implementation identity")
    exact_bool(payload, "live_sh", True, where)
    if payload.get("sh_basis_cache") is not None:
        fail(f"{where}.sh_basis_cache must be null")
    exact_int(payload, "view_batch_size", batch_size, where)
    exact_int(payload, "point_chunk", POINT_CHUNK, where)
    signature = validate_initialization_signature(
        require_mapping(payload, "initialization_signature", where),
        f"{where}.initialization_signature",
        degree,
    )
    validate_step(
        require_mapping(payload, "unmeasured_state_allocation_warmup", where),
        f"{where}.unmeasured_state_allocation_warmup",
        math.ceil(WINDOW_VIEWS / batch_size),
    )
    entries = positive_int(payload, "optimizer_state_entries_before_measurement", where)
    schema = require_list(payload, "optimizer_state_schema", where, length=entries)
    for index, keys in enumerate(schema):
        if keys != ["exp_avg", "exp_avg_sq", "step"]:
            fail(f"{where}.optimizer_state_schema[{index}] is not AdamW state")
    exact_bool(payload, "optimizer_state_reset_to_fresh_zero_before_measurement", True, where)
    measured = require_list(payload, "measured_windows", where, length=MEASURED_WINDOWS)
    rows = [
        validate_step(step, f"{where}.measured_windows[{index}]", math.ceil(WINDOW_VIEWS / batch_size))
        for index, step in enumerate(measured)
    ]
    timing = require_mapping(payload, "timing", where)
    individual = require_mapping(timing, "individual_window_seconds", f"{where}.timing")
    medians = require_mapping(timing, "median_window_seconds", f"{where}.timing")
    per_view = require_mapping(timing, "median_per_view_seconds", f"{where}.timing")
    cvs = require_mapping(timing, "coefficient_of_variation", f"{where}.timing")
    close_number(timing.get("noisy_cv_limit"), NOISY_CV_LIMIT, f"{where}.timing.noisy_cv_limit")
    result_values = {}
    result_medians = {}
    result_cvs = {}
    for component in COMPONENTS:
        stored = require_list(
            individual,
            component,
            f"{where}.timing.individual_window_seconds",
            length=MEASURED_WINDOWS,
        )
        values = [
            finite_number(value, f"{where}.timing.individual_window_seconds.{component}[{index}]", positive=True)
            for index, value in enumerate(stored)
        ]
        for index, (actual, expected) in enumerate(zip(values, rows)):
            close_number(actual, expected[component], f"{where}.timing.individual_window_seconds.{component}[{index}]")
        median = statistics.median(values)
        close_number(medians.get(component), median, f"{where}.timing.median_window_seconds.{component}")
        mean = statistics.fmean(values)
        cv = statistics.pstdev(values) / mean
        close_number(cvs.get(component), cv, f"{where}.timing.coefficient_of_variation.{component}")
        if component != "optimizer":
            expected_per_view = statistics.median(
                step[f"per_view_{component}_seconds"] for step in measured
            )
            close_number(per_view.get(component), expected_per_view, f"{where}.timing.median_per_view_seconds.{component}")
        result_values[component] = values
        result_medians[component] = float(median)
        result_cvs[component] = float(cv)
    exact_bool(payload, "timing_valid", True, where)
    finite_checks = require_mapping(payload, "finite_checks", where)
    for key in ("losses", "gradients", "model_parameters", "adam_state"):
        exact_bool(finite_checks, key, True, f"{where}.finite_checks")
    exact_bool(payload, "numerical_finite", True, where)

    baseline_allocated = nonnegative_int(payload, "cuda_baseline_allocated_bytes", where)
    baseline_reserved = nonnegative_int(payload, "cuda_baseline_reserved_bytes", where)
    peak_allocated = nonnegative_int(payload, "cuda_peak_allocated_bytes", where)
    peak_reserved = nonnegative_int(payload, "cuda_peak_reserved_bytes", where)
    if peak_allocated < baseline_allocated or peak_reserved < baseline_reserved:
        fail(f"{where} reports a peak below its baseline")
    if baseline_allocated > baseline_reserved or peak_allocated > peak_reserved:
        fail(f"{where} reports allocated CUDA memory above reserved memory")
    exact_int(
        payload,
        "cuda_incremental_peak_allocated_bytes",
        peak_allocated - baseline_allocated,
        where,
    )
    exact_int(
        payload,
        "cuda_incremental_peak_reserved_bytes",
        peak_reserved - baseline_reserved,
        where,
    )
    for key in ("cuda_free_before_bytes", "cuda_free_after_bytes"):
        value = nonnegative_int(payload, key, where)
        if value > cuda_total:
            fail(f"{where}.{key} exceeds physical CUDA capacity")
    exact_int(payload, "cuda_mem_get_info_total_before_bytes", cuda_total, where)
    exact_int(payload, "cuda_mem_get_info_total_after_bytes", cuda_total, where)
    host_before = validate_process_memory(
        require_mapping(payload, "host_before_kib", where), f"{where}.host_before_kib"
    )
    host_after = validate_process_memory(
        require_mapping(payload, "host_after_kib", where), f"{where}.host_after_kib"
    )
    exact_int(payload, "host_rss_delta_kib", host_after["VmRSS"] - host_before["VmRSS"], where)
    return {
        "signature": signature,
        "values": result_values,
        "medians": result_medians,
        "cvs": result_cvs,
        "cuda_peak_allocated_bytes": peak_allocated,
        "cuda_peak_reserved_bytes": peak_reserved,
    }


def validate_paired_speedup(payload, where, former, aligned):
    speedup = require_mapping(payload, "paired_speedup", where)
    speed_where = f"{where}.paired_speedup"
    if speedup.get("definition") != "former_scalar_seconds_divided_by_aligned_seconds":
        fail(f"{speed_where}.definition is invalid")
    paired = require_mapping(speedup, "paired_same_view_window_speedups", speed_where)
    paired_median = require_mapping(speedup, "median_paired_same_view_window_speedup", speed_where)
    ratio_median = require_mapping(speedup, "ratio_of_median_window_times", speed_where)
    result = {}
    for component in COMPONENTS:
        expected = [
            old / new
            for old, new in zip(former["values"][component], aligned["values"][component])
        ]
        stored = require_list(
            paired,
            component,
            f"{speed_where}.paired_same_view_window_speedups",
            length=MEASURED_WINDOWS,
        )
        for index, (actual, recomputed) in enumerate(zip(stored, expected)):
            close_number(actual, recomputed, f"{speed_where}.paired_same_view_window_speedups.{component}[{index}]")
        expected_paired_median = statistics.median(expected)
        expected_ratio_median = former["medians"][component] / aligned["medians"][component]
        close_number(
            paired_median.get(component),
            expected_paired_median,
            f"{speed_where}.median_paired_same_view_window_speedup.{component}",
        )
        close_number(
            ratio_median.get(component),
            expected_ratio_median,
            f"{speed_where}.ratio_of_median_window_times.{component}",
        )
        result[component] = {
            "paired_repetition_speedups": [float(value) for value in expected],
            "median_paired_speedup": float(expected_paired_median),
            "ratio_of_medians": float(expected_ratio_median),
        }
    return result


def compute_projection(medians, train_views):
    unassigned = max(
        0.0,
        medians["total"] - medians["forward"] - medians["backward"] - medians["optimizer"],
    )
    epoch_seconds = (
        (medians["forward"] + medians["backward"]) / WINDOW_VIEWS * train_views
        + (medians["optimizer"] + unassigned) * UPDATES_PER_EPOCH
    )
    worst_case = medians["total"] * UPDATES_PER_EPOCH
    return {
        "basis": (
            "training_only_median_compute_scaled_to_all_train_views_plus_"
            "1800_optimizer_steps; excludes_BP_evaluation_checkpoint_and_queue_time"
        ),
        "train_views": train_views,
        "updates_per_epoch": UPDATES_PER_EPOCH,
        "average_views_per_update": train_views / UPDATES_PER_EPOCH,
        "benchmark_window_views": WINDOW_VIEWS,
        "median_seconds_per_19_view_update": medians["total"],
        "estimated_epoch_seconds_view_scaled": epoch_seconds,
        "estimated_epoch_hours_view_scaled": epoch_seconds / 3600.0,
        "estimated_15_epoch_hours_view_scaled": epoch_seconds * 15.0 / 3600.0,
        "worst_case_19_view_updates_epoch_seconds": worst_case,
        "worst_case_19_view_updates_epoch_hours": worst_case / 3600.0,
    }


def validate_projection(payload, where, degree, former, aligned, train_views):
    stored = require_mapping(payload, "training_time_projection", where)
    scope = (
        f"compute-only degree-{degree} estimate; excludes queue, BP initialization, "
        "checkpoint I/O, train/validation evaluation, and interruption"
    )
    if stored.get("scope_warning") != scope:
        fail(f"{where}.training_time_projection.scope_warning is invalid")
    recomputed = {
        "former_scalar": compute_projection(former["medians"], train_views),
        "aligned": compute_projection(aligned["medians"], train_views),
    }
    for phase_name, expected in recomputed.items():
        actual = require_mapping(stored, phase_name, f"{where}.training_time_projection")
        if actual.get("basis") != expected["basis"]:
            fail(f"{where}.training_time_projection.{phase_name}.basis is invalid")
        for key in ("train_views", "updates_per_epoch", "benchmark_window_views"):
            exact_int(actual, key, expected[key], f"{where}.training_time_projection.{phase_name}")
        for key in (
            "average_views_per_update",
            "median_seconds_per_19_view_update",
            "estimated_epoch_seconds_view_scaled",
            "estimated_epoch_hours_view_scaled",
            "estimated_15_epoch_hours_view_scaled",
            "worst_case_19_view_updates_epoch_seconds",
            "worst_case_19_view_updates_epoch_hours",
        ):
            close_number(
                actual.get(key),
                expected[key],
                f"{where}.training_time_projection.{phase_name}.{key}",
            )
    return recomputed


def validate_view_windows(payload, where):
    windows = require_mapping(payload, "view_windows", where)
    correctness = require_mapping(windows, "correctness_and_warmup", f"{where}.view_windows")
    measured = require_list(windows, "measured_disjoint", f"{where}.view_windows", length=MEASURED_WINDOWS)
    all_local = []
    all_global = []

    def validate_indices(mapping, index_where):
        local = require_list(mapping, "local_training_indices", index_where, length=WINDOW_VIEWS)
        global_ids = require_list(mapping, "canonical_global_view_ids", index_where, length=WINDOW_VIEWS)
        for name, values in (("local_training_indices", local), ("canonical_global_view_ids", global_ids)):
            if any(type(value) is not int or value < 0 for value in values):
                fail(f"{index_where}.{name} must contain nonnegative integers")
            if len(set(values)) != len(values):
                fail(f"{index_where}.{name} contains duplicates")
        all_local.extend(local)
        all_global.extend(global_ids)

    validate_indices(correctness, f"{where}.view_windows.correctness_and_warmup")
    for index, window in enumerate(measured, start=1):
        if not isinstance(window, dict):
            fail(f"{where}.view_windows.measured_disjoint[{index - 1}] must be an object")
        exact_int(window, "repetition", index, f"{where}.view_windows.measured_disjoint[{index - 1}]")
        validate_indices(window, f"{where}.view_windows.measured_disjoint[{index - 1}]")
    if len(set(all_local)) != len(all_local) or len(set(all_global)) != len(all_global):
        fail(f"{where}.view_windows are not disjoint")
    return canonical_json(windows)


def validate_artifact(path, payload, degree, batch_size):
    where = str(path)
    if payload.get("schema") != SOURCE_SCHEMA:
        fail(f"{where}.schema must be exactly {SOURCE_SCHEMA!r}")
    exact_int(payload, "schema_version", SOURCE_SCHEMA_VERSION, where)
    if payload.get("scene") != SCENE or payload.get("dataset_name") != DATASET_NAME:
        fail(f"{where} has the wrong scene/dataset identity")
    exact_int(payload, "degree", degree, where)
    exact_int(payload, "requested_aligned_batch_size", batch_size, where)
    exact_int(payload, "point_chunk", POINT_CHUNK, where)
    exact_int(payload, "former_batch_size", 1, where)
    exact_bool(payload, "batch_memory_safety_selector", degree == 3, where)
    if payload.get("batch_memory_safety_policy") != (
        "degree3_is_the_worst_case_selector; degree0_is_timing_evidence_only"
    ):
        fail(f"{where}.batch_memory_safety_policy is invalid")
    if payload.get("sh_basis_source") != "live_real_sh_basis_on_both_paths":
        fail(f"{where}.sh_basis_source is invalid")
    if payload.get("sh_basis_cache") is not None:
        fail(f"{where}.sh_basis_cache must be null")
    exact_int(payload, "production_updates_per_epoch", UPDATES_PER_EPOCH, where)
    exact_int(payload, "production_window_views", WINDOW_VIEWS, where)
    exact_int(payload, "measured_disjoint_windows", MEASURED_WINDOWS, where)
    if payload.get("window_loss_reduction") != (
        "sum_per_view_normalized_losses_divided_by_full_19_view_window"
    ):
        fail(f"{where}.window_loss_reduction is invalid")
    if payload.get("frozen_gain") != [1.0, 0.0]:
        fail(f"{where}.frozen_gain must be [1.0, 0.0]")

    points = positive_int(payload, "points", where)
    parameters = positive_int(payload, "parameters", where)
    if payload.get("parameter_dtype") != "torch.float32":
        fail(f"{where}.parameter_dtype must be torch.float32")
    if parameters != 2 * points * (degree + 1) ** 2:
        fail(f"{where}.parameters disagrees with points and SH degree")
    geometry = require_mapping(payload, "scene_geometry", where)
    shape = require_list(geometry, "shape", f"{where}.scene_geometry", length=3)
    if (
        any(type(value) is not int or value <= 0 for value in shape)
        or shape[2] != 1
        or math.prod(shape) != points
    ):
        fail(f"{where}.scene_geometry.shape disagrees with points")
    finite_number(geometry.get("extent_xy_m"), f"{where}.scene_geometry.extent_xy_m", positive=True)
    finite_number(geometry.get("pitch_x_m"), f"{where}.scene_geometry.pitch_x_m", positive=True)
    finite_number(geometry.get("pitch_y_m"), f"{where}.scene_geometry.pitch_y_m", positive=True)
    close_number(geometry.get("z_m"), 0.0, f"{where}.scene_geometry.z_m")
    physics = require_mapping(payload, "physics", where)
    if physics.get("range_model") != "none" or physics.get("propagation_model") != "monostatic_near_field_reference":
        fail(f"{where}.physics has the wrong propagation contract")
    finite_number(physics.get("reference_range_m"), f"{where}.physics.reference_range_m", positive=True)
    close_number(physics.get("phase_sign"), -1.0, f"{where}.physics.phase_sign")
    center = require_list(physics, "scene_center_m", f"{where}.physics", length=3)
    for index, value in enumerate(center):
        finite_number(value, f"{where}.physics.scene_center_m[{index}]")
    support = require_mapping(payload, "gotcha_support_preflight", where)
    for key in ("support_schema", "mask_realization"):
        if not isinstance(support.get(key), str) or not support[key]:
            fail(f"{where}.gotcha_support_preflight.{key} is invalid")
    positive_int(support, "view_count", f"{where}.gotcha_support_preflight")
    finite_number(
        support.get("minimum_lower_window_margin_m"),
        f"{where}.gotcha_support_preflight.minimum_lower_window_margin_m",
        positive=True,
    )
    finite_number(
        support.get("minimum_upper_window_margin_m"),
        f"{where}.gotcha_support_preflight.minimum_upper_window_margin_m",
        positive=True,
    )
    exact_int(payload, "frequencies", 426, where)
    train_views = positive_int(payload, "train_views", where)
    target_mean = finite_number(payload.get("target_mean_power"), f"{where}.target_mean_power", positive=True)
    target_sum = finite_number(payload.get("target_power_sum"), f"{where}.target_power_sum", positive=True)
    target_count = positive_int(payload, "target_sample_count", where)
    close_number(target_mean, target_sum / target_count, f"{where}.target_mean_power")
    windows_identity = validate_view_windows(payload, where)
    for key in ("dataset", "trainer_source", "device", "torch_version", "cuda_version"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            fail(f"{where}.{key} must be a nonempty string")
    if not payload["dataset"].replace("\\", "/").endswith("/gotcha_pass2_hh/gotcha_pass2_hh.npz"):
        fail(f"{where}.dataset has the wrong GOTCHA source")
    if not payload["trainer_source"].replace("\\", "/").endswith(
        "/scripts/public_radar_gotcha_full_domain_v1_partial.py"
    ):
        fail(f"{where}.trainer_source is invalid")
    if "V100" not in payload["device"]:
        fail(f"{where}.device is not a V100")
    if payload["torch_version"] != "2.6.0+cu124" or payload["cuda_version"] != "12.4":
        fail(f"{where} was not measured in the sealed Torch/CUDA environment")

    if payload.get("status") != "pass" or payload.get("stage") != "complete":
        fail(f"{where} is not a completed passing benchmark")
    for key in ("measurement_completed", "runtime_nonzero", "timing_valid", "numerical_finite", "memory_safe"):
        exact_bool(payload, key, True, where)
    runtime = finite_number(payload.get("runtime_seconds"), f"{where}.runtime_seconds", positive=True)
    finite_number(payload.get("completed_unix"), f"{where}.completed_unix", positive=True)
    if require_list(payload, "unsafe_reasons", where) != []:
        fail(f"{where}.unsafe_reasons must be empty")
    if require_list(payload, "noisy_components", where) != []:
        fail(f"{where}.noisy_components must be empty")
    if "error" in payload or "error_type" in payload:
        fail(f"{where} contains an error record")
    exact_int(payload, "cuda_cap_bytes", CUDA_CAP_BYTES, where)
    exact_int(payload, "host_cap_kib", HOST_CAP_KIB, where)
    if payload.get("host_cap_scope") != (
        "whole_process_VmHWM_including_dataset_correctness_warmup_and_both_timing_phases"
    ):
        fail(f"{where}.host_cap_scope is invalid")
    cuda_total = positive_int(payload, "cuda_total_bytes", where)
    if cuda_total < CUDA_CAP_BYTES:
        fail(f"{where}.cuda_total_bytes is below the engineering cap")

    validate_tiny_gate(payload, where, batch_size)
    full_signature = validate_full_gate(payload, where, degree, batch_size)
    former = validate_phase(
        require_mapping(payload, "former_scalar", where),
        f"{where}.former_scalar",
        label="former_scalar_live_sh",
        implementation="former_scalar",
        batch_size=1,
        degree=degree,
        cuda_total=cuda_total,
    )
    aligned = validate_phase(
        require_mapping(payload, "aligned", where),
        f"{where}.aligned",
        label=f"aligned_b{batch_size}_live_sh",
        implementation="aligned",
        batch_size=batch_size,
        degree=degree,
        cuda_total=cuda_total,
    )
    if not (former["signature"] == aligned["signature"] == full_signature):
        fail(f"{where} did not use the same initialized scene in all paths")
    noisy = [
        f"{phase_name}.{component}"
        for phase_name, phase in (("former", former), ("aligned", aligned))
        for component in ("forward", "backward", "total")
        if phase["cvs"][component] > NOISY_CV_LIMIT
    ]
    if noisy:
        fail(f"{where} has noisy timed components: {noisy}")
    paired = validate_paired_speedup(payload, where, former, aligned)
    projection = validate_projection(payload, where, degree, former, aligned, train_views)

    if payload.get("cuda_whole_process_peak_scope") != "maximum_across_all_stage_isolated_peak_counters":
        fail(f"{where}.cuda_whole_process_peak_scope is invalid")
    whole_allocated = nonnegative_int(payload, "cuda_whole_process_peak_allocated_bytes", where)
    whole_reserved = nonnegative_int(payload, "cuda_whole_process_peak_reserved_bytes", where)
    if whole_allocated > whole_reserved:
        fail(f"{where} reports whole-process allocated memory above reserved memory")
    if whole_allocated > CUDA_CAP_BYTES or whole_reserved > CUDA_CAP_BYTES:
        fail(f"{where} exceeds the CUDA memory cap")
    if whole_allocated < max(former["cuda_peak_allocated_bytes"], aligned["cuda_peak_allocated_bytes"]):
        fail(f"{where} whole-process allocated peak is below a phase peak")
    if whole_reserved < max(former["cuda_peak_reserved_bytes"], aligned["cuda_peak_reserved_bytes"]):
        fail(f"{where} whole-process reserved peak is below a phase peak")
    host_after = validate_process_memory(
        require_mapping(payload, "host_after_kib", where), f"{where}.host_after_kib"
    )
    host_peak = positive_int(payload, "host_VmHWM_kib", where)
    exact_int(payload, "host_VmHWM_kib", host_after["VmHWM"], where)
    if host_peak > HOST_CAP_KIB:
        fail(f"{where}.host_VmHWM_kib exceeds the host cap")

    return {
        "degree": degree,
        "view_batch_size": batch_size,
        "path": str(path.resolve()),
        "status": "pass",
        "runtime_seconds": runtime,
        "paired_speedup": paired,
        "compute_only_projection": projection,
        "cuda_whole_process_peak_allocated_bytes": whole_allocated,
        "cuda_whole_process_peak_reserved_bytes": whole_reserved,
        "host_VmHWM_kib": host_peak,
        "scientific_identity": {
            "windows": windows_identity,
            "initialization": full_signature,
        },
    }


IDENTITY_FIELDS = (
    "scene",
    "dataset_name",
    "dataset",
    "trainer_source",
    "point_chunk",
    "former_batch_size",
    "sh_basis_source",
    "sh_basis_cache",
    "production_updates_per_epoch",
    "production_window_views",
    "measured_disjoint_windows",
    "window_loss_reduction",
    "frozen_gain",
    "points",
    "parameter_dtype",
    "scene_geometry",
    "physics",
    "gotcha_support_preflight",
    "frequencies",
    "train_views",
    "target_mean_power",
    "target_power_sum",
    "target_sample_count",
    "view_windows",
    "cuda_cap_bytes",
    "host_cap_kib",
    "host_cap_scope",
    "torch_version",
    "cuda_version",
)


def validate_cross_cell_identity(payloads):
    first_key = (0, 1)
    first = payloads[first_key]
    missing = [field for field in IDENTITY_FIELDS if field not in first]
    if missing:
        fail(f"degree 0/B1 is missing cross-cell identity fields: {missing}")
    expected = canonical_json({field: first[field] for field in IDENTITY_FIELDS})
    for key, payload in sorted(payloads.items()):
        missing = [field for field in IDENTITY_FIELDS if field not in payload]
        if missing:
            fail(f"degree {key[0]}/B{key[1]} is missing identity fields: {missing}")
        actual = canonical_json({field: payload[field] for field in IDENTITY_FIELDS})
        if actual != expected:
            fail(f"scientific or viewpoint identity changes at degree {key[0]}/B{key[1]}")
    for degree in DEGREES:
        reference_parameters = payloads[(degree, 1)]["parameters"]
        reference_signature = payloads[(degree, 1)]["full_window_equivalence_gate"][
            "initialization_signature"
        ]
        for batch in BATCH_SIZES[1:]:
            payload = payloads[(degree, batch)]
            if payload["parameters"] != reference_parameters:
                fail(f"parameter count changes across B for degree {degree}")
            if canonical_json(payload["full_window_equivalence_gate"]["initialization_signature"]) != canonical_json(reference_signature):
                fail(f"initial scene changes across B for degree {degree}")


def audit(run_root):
    root, paths = discover_artifacts(run_root)
    payloads = {key: load_json(path) for key, path in paths.items()}
    validate_cross_cell_identity(payloads)
    rows = {
        key: validate_artifact(paths[key], payloads[key], *key)
        for key in sorted(paths)
    }
    passing_degree3 = [
        batch for batch in BATCH_SIZES if rows[(3, batch)]["status"] == "pass"
    ]
    if not passing_degree3:
        fail("no degree-3 viewpoint batch passed the complete engineering gate")
    selected = max(passing_degree3)
    row_list = [rows[key] for key in sorted(rows)]
    return {
        "schema": SUMMARY_SCHEMA,
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "source_schema": SOURCE_SCHEMA,
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "run_root": str(root),
        "artifact_count": len(row_list),
        "audit_passed": True,
        "selection": {
            "degree": 3,
            "eligible_batch_sizes": passing_degree3,
            "largest_fully_passing_batch_size": selected,
            "policy": "largest batch whose degree-3 row passes every equivalence, finite, timing, CV, and memory gate",
        },
        "eta_scope": (
            "compute-only 15-epoch estimates recomputed from median 19-view timings; "
            "excludes queue, backprojection, evaluation, checkpoints, and interruptions"
        ),
        "rows": row_list,
    }


def write_summary(path, payload):
    destination = Path(path).resolve()
    if destination.exists():
        fail(f"refusing to overwrite existing summary: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        destination.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = None
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    except FileExistsError:
        fail(f"refusing to overwrite existing summary: {destination}")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Audit exactly eight PublicRadar V5 viewpoint-batch artifacts, "
            "select the largest passing degree-3 batch, and recompute 15-epoch ETAs."
        )
    )
    parser.add_argument("--run-root", required=True, help="directory containing the sealed eight-cell V5 matrix")
    parser.add_argument("--output", help="optional new summary JSON path (never overwritten)")
    args = parser.parse_args(argv)
    try:
        summary = audit(args.run_root)
        if args.output:
            written = write_summary(args.output, summary)
            summary = {**summary, "written_to": str(written)}
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    except AuditError as exc:
        parser.exit(2, f"V5 viewpoint-batch audit failed: {exc}\n")


if __name__ == "__main__":
    main()
