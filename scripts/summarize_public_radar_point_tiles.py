#!/usr/bin/env python
"""Fail-closed audit and selection for PublicRadar point-tile benchmarks.

The benchmark emits one immutable artifact for every scene, SH degree, and
point-tile size.  This reader accepts exactly one complete 12-row matrix at a
single viewpoint batch size, validates the engineering evidence in every row,
and applies the point-tile promotion policy without modifying run artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import statistics


SOURCE_SCHEMA = "rift.public_radar.point_tile_benchmark_v1"
SUMMARY_SCHEMA = "rift.public_radar.point_tile_summary_v1"
SCHEMA_VERSION = 1
REFERENCE_POINT_CHUNK = 32768
POINT_CHUNKS = (32768, 65536, 131072)
CANDIDATE_POINT_CHUNKS = (65536, 131072)
SCENES = ("camry", "gotcha_full")
DEGREES = (0, 3)
VIEW_BATCH_SIZES = (1, 2, 4, 8)
MEASURED_STEPS = 3
MINIMUM_FORWARD_BACKWARD_SPEEDUP = 1.10
MAXIMUM_REGRESSION_FRACTION = 0.05
NEAR_TIE_FRACTION = 0.05
NOISY_CV_LIMIT = 0.10
EXPECTED_TRAINER_SCENE = {
    "camry": "camry",
    "gotcha_full": "gotcha_p2_full_domain",
}
EXPECTED_CACHE_DATASET = {
    "camry": "cvdomes_camry",
    "gotcha_full": "gotcha_pass2_hh",
}
SH_CACHE_SCHEMA = "rift.public_radar_directional_sh_cache_v1"
SH_CACHE_SEMANTICS = {
    "basis_order": "degree_major_then_m_negative_to_positive",
    "angle_source": "viewpoint_positions_float64_then_dataset_float32",
    "theta_convention": "acos(z / norm(position))",
    "phi_convention": "atan2(y, x)",
    "view_index_space": "canonical_npz_row_before_role_selection",
}
CUDA_CAP_BYTES = 12 * 1024**3
EXPECTED_HOST_CAP_KIB = {
    "camry": 12 * 1024**2,
    "gotcha_full": 24 * 1024**2,
}
ARTIFACT_RE = re.compile(
    r"^point_tiles_(camry|gotcha_full)_deg(0|3)_b(1|2|4|8)_"
    r"tile(32768|65536|131072)\.json$"
)


class AuditError(ValueError):
    """The benchmark matrix is incomplete, inconsistent, or unsafe."""


def fail(message):
    raise AuditError(message)


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


def close_number(actual, expected, where):
    actual = finite_number(actual, where)
    expected = finite_number(expected, f"{where} (recomputed)")
    if not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
        fail(f"{where} disagrees with the measured-window data")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


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


def discover_artifacts(run_root, view_batch_size):
    run_root = Path(run_root).resolve()
    if not run_root.is_dir():
        fail(f"benchmark run root is not a directory: {run_root}")
    found = {}
    paths = sorted(run_root.rglob("point_tiles_*.json"))
    if not paths:
        fail(f"no point-tile artifacts found under {run_root}")
    for path in paths:
        match = ARTIFACT_RE.fullmatch(path.name)
        if match is None:
            fail(f"non-canonical point-tile artifact name: {path}")
        scene, degree, batch_size, point_chunk = match.groups()
        batch_size = int(batch_size)
        if batch_size != view_batch_size:
            fail(
                "benchmark root mixes viewpoint batch sizes: "
                f"expected B={view_batch_size}, found B={batch_size} at {path}"
            )
        key = (scene, int(degree), int(point_chunk))
        if key in found:
            fail(f"duplicate point-tile artifact identity {key}: {found[key]} and {path}")
        found[key] = path
    expected = {
        (scene, degree, point_chunk)
        for scene in SCENES
        for degree in DEGREES
        for point_chunk in POINT_CHUNKS
    }
    missing = sorted(expected - set(found))
    unexpected = sorted(set(found) - expected)
    if missing or unexpected or len(found) != len(expected):
        fail(
            "point-tile artifact matrix must contain exactly 12 rows; "
            f"missing={missing}, unexpected={unexpected}, found={len(found)}"
        )
    return run_root, found


def validate_step(step, where, window_views, microbatches):
    if not isinstance(step, dict):
        fail(f"{where} must be an object")
    exact_bool(step, "loss_finite", True, where)
    exact_int(step, "window_views", window_views, where)
    exact_int(step, "microbatches", microbatches, where)
    exact_int(step, "optimizer_steps", 1, where)
    loss = finite_number(step.get("loss"), f"{where}.loss")
    view_losses = require_list(step, "view_losses", where, length=window_views)
    finite_view_losses = [
        finite_number(value, f"{where}.view_losses[{index}]")
        for index, value in enumerate(view_losses)
    ]
    close_number(loss, statistics.fmean(finite_view_losses), f"{where}.loss")
    components = {}
    for component in ("forward", "backward", "optimizer", "total"):
        components[component] = finite_number(
            step.get(f"{component}_seconds"),
            f"{where}.{component}_seconds",
            positive=True,
        )
    if components["total"] + 1.0e-6 < sum(
        components[name] for name in ("forward", "backward", "optimizer")
    ):
        fail(f"{where}.total_seconds is shorter than its timed components")
    return loss, components


def validate_phase(
    phase,
    where,
    *,
    label,
    point_chunk,
    window_views,
    microbatches,
    cuda_cap_bytes,
):
    if not isinstance(phase, dict):
        fail(f"{where} must be an object")
    if phase.get("label") != label:
        fail(f"{where}.label must be exactly {label!r}")
    exact_int(phase, "point_chunk", point_chunk, where)
    finite_number(
        phase.get("initial_coefficient_std"),
        f"{where}.initial_coefficient_std",
        positive=True,
    )
    validate_step(
        require_mapping(phase, "warmup_window", where),
        f"{where}.warmup_window",
        window_views,
        microbatches,
    )
    measured = require_list(phase, "measured_windows", where, length=MEASURED_STEPS)
    measured_components = []
    for index, step in enumerate(measured):
        _, components = validate_step(
            step,
            f"{where}.measured_windows[{index}]",
            window_views,
            microbatches,
        )
        measured_components.append(components)

    timing = require_mapping(phase, "timing", where)
    individual = require_mapping(timing, "individual_seconds", f"{where}.timing")
    medians = require_mapping(timing, "median_seconds", f"{where}.timing")
    cvs = require_mapping(timing, "coefficient_of_variation", f"{where}.timing")
    close_number(
        timing.get("noisy_cv_limit"),
        NOISY_CV_LIMIT,
        f"{where}.timing.noisy_cv_limit",
    )
    values_by_component = {}
    for component in ("forward", "backward", "optimizer", "total"):
        stored = require_list(
            individual,
            component,
            f"{where}.timing.individual_seconds",
            length=MEASURED_STEPS,
        )
        values = [
            finite_number(
                value,
                f"{where}.timing.individual_seconds.{component}[{index}]",
                positive=True,
            )
            for index, value in enumerate(stored)
        ]
        expected_values = [row[component] for row in measured_components]
        for index, (value, expected) in enumerate(zip(values, expected_values)):
            close_number(
                value,
                expected,
                f"{where}.timing.individual_seconds.{component}[{index}]",
            )
        expected_median = statistics.median(values)
        close_number(
            medians.get(component),
            expected_median,
            f"{where}.timing.median_seconds.{component}",
        )
        mean = statistics.fmean(values)
        expected_cv = statistics.pstdev(values) / mean
        close_number(
            cvs.get(component),
            expected_cv,
            f"{where}.timing.coefficient_of_variation.{component}",
        )
        if component in ("forward", "backward", "total") and expected_cv > NOISY_CV_LIMIT:
            fail(f"{where} is noisy in {component} (CV={expected_cv:.6g})")
        values_by_component[component] = values

    exact_bool(phase, "timing_valid", True, where)
    exact_bool(phase, "numerical_finite", True, where)
    finite_checks = require_mapping(phase, "finite_checks", where)
    for key in ("losses", "gradients", "model_parameters", "adam_state"):
        exact_bool(finite_checks, key, True, f"{where}.finite_checks")
    for key in (
        "cuda_baseline_allocated_bytes",
        "cuda_baseline_reserved_bytes",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
        "cuda_incremental_peak_allocated_bytes",
        "cuda_incremental_peak_reserved_bytes",
        "cuda_free_before_bytes",
        "cuda_free_after_bytes",
        "cuda_mem_get_info_total_before_bytes",
        "cuda_mem_get_info_total_after_bytes",
        "host_rss_before_kib",
        "host_rss_after_kib",
    ):
        finite_number(phase.get(key), f"{where}.{key}", nonnegative=True)
    for key in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
        if int(phase[key]) > cuda_cap_bytes:
            fail(f"{where}.{key} exceeds the declared CUDA cap")
    return values_by_component


def validate_correctness(
    payload,
    where,
    point_chunk,
    view_batch_size,
    window_views,
    microbatches,
):
    gate = require_mapping(payload, "correctness_gate", where)
    exact_int(gate, "reference_point_chunk", REFERENCE_POINT_CHUNK, f"{where}.correctness_gate")
    exact_int(gate, "candidate_point_chunk", point_chunk, f"{where}.correctness_gate")
    exact_int(gate, "view_batch_size", view_batch_size, f"{where}.correctness_gate")
    exact_int(gate, "window_views", window_views, f"{where}.correctness_gate")
    exact_int(
        gate,
        "microbatches_per_window",
        microbatches,
        f"{where}.correctness_gate",
    )
    exact_bool(gate, "same_measurement_exact", True, f"{where}.correctness_gate")
    exact_bool(gate, "all_finite", True, f"{where}.correctness_gate")
    exact_bool(gate, "passed", True, f"{where}.correctness_gate")
    tolerances = require_mapping(gate, "tolerances", f"{where}.correctness_gate")
    prediction_tolerance = finite_number(
        tolerances.get("prediction_relative_l2"),
        f"{where}.correctness_gate.tolerances.prediction_relative_l2",
        positive=True,
    )
    loss_tolerance = finite_number(
        tolerances.get("loss_relative_l2"),
        f"{where}.correctness_gate.tolerances.loss_relative_l2",
        positive=True,
    )
    gradient_tolerance = finite_number(
        tolerances.get("parameter_gradient_relative_l2"),
        f"{where}.correctness_gate.tolerances.parameter_gradient_relative_l2",
        positive=True,
    )
    close_number(
        prediction_tolerance,
        5.0e-5,
        f"{where}.correctness_gate.tolerances.prediction_relative_l2",
    )
    close_number(
        loss_tolerance,
        5.0e-5,
        f"{where}.correctness_gate.tolerances.loss_relative_l2",
    )
    close_number(
        gradient_tolerance,
        2.0e-4,
        f"{where}.correctness_gate.tolerances.parameter_gradient_relative_l2",
    )
    for key in ("prediction_relative_l2", "view_loss_relative_l2", "mean_loss_relative_error"):
        value = finite_number(gate.get(key), f"{where}.correctness_gate.{key}", nonnegative=True)
        tolerance = prediction_tolerance if key == "prediction_relative_l2" else loss_tolerance
        if value > tolerance:
            fail(f"{where}.correctness_gate.{key} exceeds its tolerance")
    gradients = require_mapping(gate, "parameter_gradients", f"{where}.correctness_gate")
    exact_bool(gradients, "all_present", True, f"{where}.correctness_gate.parameter_gradients")
    exact_bool(gradients, "all_finite", True, f"{where}.correctness_gate.parameter_gradients")
    maximum_gradient = finite_number(
        gradients.get("maximum_relative_l2"),
        f"{where}.correctness_gate.parameter_gradients.maximum_relative_l2",
        nonnegative=True,
    )
    if maximum_gradient > gradient_tolerance:
        fail(f"{where} parameter-gradient mismatch exceeds its tolerance")
    relative_by_parameter = require_list(
        gradients,
        "relative_l2_by_parameter",
        f"{where}.correctness_gate.parameter_gradients",
    )
    parameter_order = require_list(gate, "parameter_order", f"{where}.correctness_gate")
    if not parameter_order or len(parameter_order) != len(relative_by_parameter):
        fail(f"{where} parameter-gradient evidence has inconsistent parameter order")
    relative_values = [
        finite_number(
            value,
            f"{where}.correctness_gate.parameter_gradients.relative_l2_by_parameter[{index}]",
            nonnegative=True,
        )
        for index, value in enumerate(relative_by_parameter)
    ]
    close_number(
        maximum_gradient,
        max(relative_values),
        f"{where}.correctness_gate.parameter_gradients.maximum_relative_l2",
    )
    finite_number(
        gradients.get("maximum_absolute_error"),
        f"{where}.correctness_gate.parameter_gradients.maximum_absolute_error",
        nonnegative=True,
    )


def recompute_speedups(payload, where, reference_values, candidate_values):
    stored = require_mapping(payload, "paired_speedup", where)
    if stored.get("definition") != "reference_32k_seconds_divided_by_candidate_seconds":
        fail(f"{where}.paired_speedup.definition is invalid")
    stored_median_times = require_mapping(stored, "median_time_speedup", f"{where}.paired_speedup")
    stored_values = require_mapping(stored, "paired_repetition_speedups", f"{where}.paired_speedup")
    stored_paired_medians = require_mapping(
        stored,
        "median_paired_repetition_speedup",
        f"{where}.paired_speedup",
    )
    paired = {}
    for component in ("forward", "backward", "optimizer", "total"):
        ratios = [
            reference / candidate
            for reference, candidate in zip(
                reference_values[component], candidate_values[component]
            )
        ]
        saved = require_list(
            stored_values,
            component,
            f"{where}.paired_speedup.paired_repetition_speedups",
            length=MEASURED_STEPS,
        )
        for index, (actual, expected) in enumerate(zip(saved, ratios)):
            close_number(
                actual,
                expected,
                f"{where}.paired_speedup.paired_repetition_speedups.{component}[{index}]",
            )
        median_ratio = statistics.median(ratios)
        close_number(
            stored_paired_medians.get(component),
            median_ratio,
            f"{where}.paired_speedup.median_paired_repetition_speedup.{component}",
        )
        ratio_of_medians = (
            statistics.median(reference_values[component])
            / statistics.median(candidate_values[component])
        )
        close_number(
            stored_median_times.get(component),
            ratio_of_medians,
            f"{where}.paired_speedup.median_time_speedup.{component}",
        )
        paired[component] = ratios
    forward_backward = [
        (reference_values["forward"][index] + reference_values["backward"][index])
        / (candidate_values["forward"][index] + candidate_values["backward"][index])
        for index in range(MEASURED_STEPS)
    ]
    return {
        "total": float(statistics.median(paired["total"])),
        "forward_backward": float(statistics.median(forward_backward)),
        "total_by_repetition": paired["total"],
        "forward_backward_by_repetition": forward_backward,
    }


def validate_artifact(path, payload, expected_key, view_batch_size):
    scene, degree, point_chunk = expected_key
    where = str(path)
    if payload.get("schema") != SOURCE_SCHEMA:
        fail(f"{where}.schema must be exactly {SOURCE_SCHEMA!r}")
    exact_int(payload, "schema_version", SCHEMA_VERSION, where)
    if payload.get("scene") != scene:
        fail(f"{where}.scene disagrees with its artifact identity")
    if payload.get("trainer_scene") != EXPECTED_TRAINER_SCENE[scene]:
        fail(f"{where}.trainer_scene is invalid for {scene}")
    exact_int(payload, "degree", degree, where)
    exact_int(payload, "view_batch_size", view_batch_size, where)
    exact_int(payload, "point_chunk", point_chunk, where)
    exact_int(payload, "reference_point_chunk", REFERENCE_POINT_CHUNK, where)
    exact_int(payload, "production_updates_per_epoch", 1800, where)
    window_views = positive_int(payload, "production_window_views", where)
    microbatches = positive_int(payload, "microbatches_per_window", where)
    expected_microbatches = math.ceil(window_views / view_batch_size)
    if microbatches != expected_microbatches:
        fail(f"{where}.microbatches_per_window does not match B={view_batch_size}")
    if payload.get("window_loss_reduction") != (
        "sum_per_view_normalized_losses_divided_by_full_window_size"
    ):
        fail(f"{where}.window_loss_reduction is invalid")
    positive_int(payload, "points", where)
    positive_int(payload, "parameters", where)
    positive_int(payload, "frequencies", where)
    positive_int(payload, "train_views", where)
    finite_number(payload.get("target_mean_power"), f"{where}.target_mean_power", positive=True)
    finite_number(payload.get("target_power_sum"), f"{where}.target_power_sum", positive=True)
    positive_int(payload, "target_sample_count", where)
    finite_number(
        payload.get("initial_coefficient_std"),
        f"{where}.initial_coefficient_std",
        positive=True,
    )
    cache = require_mapping(payload, "sh_basis_cache", where)
    if cache.get("schema") != SH_CACHE_SCHEMA:
        fail(f"{where}.sh_basis_cache.schema is invalid")
    exact_int(cache, "schema_version", 1, f"{where}.sh_basis_cache")
    if cache.get("dataset_name") != EXPECTED_CACHE_DATASET[scene]:
        fail(f"{where}.sh_basis_cache.dataset_name is invalid for {scene}")
    positive_int(cache, "view_count", f"{where}.sh_basis_cache")
    exact_int(cache, "max_degree", 6, f"{where}.sh_basis_cache")
    exact_int(cache, "basis_count", 49, f"{where}.sh_basis_cache")
    if cache.get("basis_dtype") != "float32":
        fail(f"{where}.sh_basis_cache.basis_dtype must be 'float32'")
    if degree > cache["max_degree"]:
        fail(f"{where} requests an SH degree beyond the cache contract")
    for key, expected in SH_CACHE_SEMANTICS.items():
        if cache.get(key) != expected:
            fail(f"{where}.sh_basis_cache.{key} is invalid")
    source_dtype = cache.get("viewpoint_positions_source_dtype")
    if not isinstance(source_dtype, str) or not source_dtype:
        fail(f"{where}.sh_basis_cache.viewpoint_positions_source_dtype is invalid")
    cache_dir = cache.get("cache_dir")
    if not isinstance(cache_dir, str) or not cache_dir:
        fail(f"{where}.sh_basis_cache.cache_dir is invalid")
    for key in ("trainer_source", "dataset", "device", "torch_version", "cuda_version"):
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            fail(f"{where}.{key} must be a nonempty string")
    if payload.get("parameter_dtype") != "torch.float32":
        fail(f"{where}.parameter_dtype must be 'torch.float32'")
    parameter_dtypes = require_mapping(payload, "parameter_dtypes", where)
    if parameter_dtypes != {"w_re": "torch.float32", "w_im": "torch.float32"}:
        fail(f"{where}.parameter_dtypes is invalid")
    if not isinstance(payload.get("scene_geometry"), dict) or not payload["scene_geometry"]:
        fail(f"{where}.scene_geometry must be a nonempty object")
    if not isinstance(payload.get("physics"), dict) or not payload["physics"]:
        fail(f"{where}.physics must be a nonempty object")
    support = payload.get("gotcha_support_preflight")
    if scene == "camry" and support is not None:
        fail(f"{where}.gotcha_support_preflight must be null for Camry")
    if scene == "gotcha_full" and (not isinstance(support, dict) or not support):
        fail(f"{where}.gotcha_support_preflight must be present for GOTCHA")
    if not isinstance(payload.get("canonical_view_windows"), dict):
        fail(f"{where}.canonical_view_windows must be an object")
    for role in ("correctness", "warmup", "measured_repeated_three_times"):
        values = require_list(
            payload["canonical_view_windows"], role, f"{where}.canonical_view_windows"
        )
        if len(values) != window_views or any(type(value) is not int or value < 0 for value in values):
            fail(f"{where}.canonical_view_windows.{role} is invalid")
    if payload.get("status") != "pass" or payload.get("stage") != "complete":
        fail(f"{where} is not a completed passing benchmark")
    exact_bool(payload, "measurement_completed", True, where)
    exact_bool(payload, "numerical_finite", True, where)
    exact_bool(payload, "timing_valid", True, where)
    exact_bool(payload, "memory_safe", True, where)
    if require_list(payload, "unsafe_reasons", where) != []:
        fail(f"{where}.unsafe_reasons must be empty")
    if require_list(payload, "noisy_components", where) != []:
        fail(f"{where}.noisy_components must be empty")
    if "error" in payload or "error_type" in payload:
        fail(f"{where} contains an error record")
    finite_number(payload.get("completed_unix"), f"{where}.completed_unix", positive=True)
    cuda_cap = positive_int(payload, "cuda_cap_bytes", where)
    if cuda_cap != CUDA_CAP_BYTES:
        fail(f"{where}.cuda_cap_bytes disagrees with the benchmark-v1 contract")
    host_cap = positive_int(payload, "host_cap_kib", where)
    if host_cap != EXPECTED_HOST_CAP_KIB[scene]:
        fail(f"{where}.host_cap_kib disagrees with the {scene} contract")
    cuda_total = positive_int(payload, "cuda_total_bytes", where)
    if cuda_total < cuda_cap:
        fail(f"{where}.cuda_total_bytes is smaller than the benchmark cap")
    for key in (
        "cuda_whole_process_peak_allocated_bytes",
        "cuda_whole_process_peak_reserved_bytes",
    ):
        value = finite_number(payload.get(key), f"{where}.{key}", nonnegative=True)
        if value > cuda_cap:
            fail(f"{where}.{key} exceeds the declared CUDA cap")
    host_peak = finite_number(payload.get("host_VmHWM_kib"), f"{where}.host_VmHWM_kib", positive=True)
    if host_peak > host_cap:
        fail(f"{where}.host_VmHWM_kib exceeds the declared host cap")
    validate_correctness(
        payload,
        where,
        point_chunk,
        view_batch_size,
        window_views,
        microbatches,
    )
    reference_values = validate_phase(
        require_mapping(payload, "reference_32k", where),
        f"{where}.reference_32k",
        label="reference_32k",
        point_chunk=REFERENCE_POINT_CHUNK,
        window_views=window_views,
        microbatches=microbatches,
        cuda_cap_bytes=cuda_cap,
    )
    candidate_values = validate_phase(
        require_mapping(payload, "candidate", where),
        f"{where}.candidate",
        label=f"candidate_{point_chunk}",
        point_chunk=point_chunk,
        window_views=window_views,
        microbatches=microbatches,
        cuda_cap_bytes=cuda_cap,
    )
    speedups = recompute_speedups(payload, where, reference_values, candidate_values)
    return {
        "scene": scene,
        "degree": degree,
        "point_chunk": point_chunk,
        "path": str(path.resolve()),
        "paired_median_speedup": {
            "total": speedups["total"],
            "forward_backward": speedups["forward_backward"],
        },
        "paired_repetition_speedup": {
            "total": speedups["total_by_repetition"],
            "forward_backward": speedups["forward_backward_by_repetition"],
        },
        "candidate_median_seconds": {
            "total": float(statistics.median(candidate_values["total"])),
            "forward_backward": float(
                statistics.median(
                    [
                        forward + backward
                        for forward, backward in zip(
                            candidate_values["forward"], candidate_values["backward"]
                        )
                    ]
                )
            ),
        },
        "cuda_peak_allocated_bytes": int(payload["cuda_whole_process_peak_allocated_bytes"]),
        "cuda_peak_reserved_bytes": int(payload["cuda_whole_process_peak_reserved_bytes"]),
        "host_VmHWM_kib": int(payload["host_VmHWM_kib"]),
        "completed_unix": float(payload["completed_unix"]),
    }


# Device name and physical CUDA capacity are deliberately per-row checks: the
# PACE V100 pool mixes 16 GB and 32 GB cards.  Every row is still bounded by
# the same 12 GiB benchmark cap, so hardware capacity is not scientific identity.
TILE_INVARIANT_FIELDS = (
    "schema",
    "schema_version",
    "scene",
    "trainer_scene",
    "trainer_source",
    "dataset",
    "degree",
    "view_batch_size",
    "reference_point_chunk",
    "production_updates_per_epoch",
    "production_window_views",
    "microbatches_per_window",
    "window_loss_reduction",
    "points",
    "parameters",
    "parameter_dtype",
    "parameter_dtypes",
    "scene_geometry",
    "physics",
    "gotcha_support_preflight",
    "frequencies",
    "train_views",
    "target_mean_power",
    "target_power_sum",
    "target_sample_count",
    "initial_coefficient_std",
    "sh_basis_cache",
    "canonical_view_windows",
    "cuda_cap_bytes",
    "host_cap_kib",
    "host_cap_scope",
    "torch_version",
    "cuda_version",
)

DEGREE_INVARIANT_FIELDS = tuple(
    field
    for field in TILE_INVARIANT_FIELDS
    if field not in ("degree", "parameters", "initial_coefficient_std")
)


def invariant(mapping, fields, where):
    missing = [field for field in fields if field not in mapping]
    if missing:
        fail(f"{where} is missing identity fields {missing}")
    return canonical_json({field: mapping[field] for field in fields})


def validate_cross_row_identity(payloads):
    for scene in SCENES:
        degree_identities = []
        for degree in DEGREES:
            rows = [payloads[(scene, degree, chunk)] for chunk in POINT_CHUNKS]
            expected = invariant(rows[0], TILE_INVARIANT_FIELDS, f"{scene}/deg{degree}/32k")
            for point_chunk, row in zip(POINT_CHUNKS[1:], rows[1:]):
                actual = invariant(row, TILE_INVARIANT_FIELDS, f"{scene}/deg{degree}/{point_chunk}")
                if actual != expected:
                    fail(f"{scene} degree {degree} changes scientific identity across tile sizes")
            degree_identities.append(
                invariant(rows[0], DEGREE_INVARIANT_FIELDS, f"{scene}/deg{degree}")
            )
        if degree_identities[0] != degree_identities[1]:
            fail(f"{scene} changes dataset or engineering identity across SH degrees")


def candidate_policy(rows, point_chunk):
    speedups = [
        rows[(scene, degree, point_chunk)]["paired_median_speedup"]["forward_backward"]
        for scene in SCENES
        for degree in DEGREES
    ]
    totals = [
        rows[(scene, degree, point_chunk)]["paired_median_speedup"]["total"]
        for scene in SCENES
        for degree in DEGREES
    ]
    median_speedup = float(statistics.median(speedups))
    minimum_speedup = min(speedups)
    return {
        "point_chunk": point_chunk,
        "median_forward_backward_speedup": median_speedup,
        "median_total_speedup": float(statistics.median(totals)),
        "minimum_scene_degree_forward_backward_speedup": minimum_speedup,
        "median_improvement_gate_passed": median_speedup >= MINIMUM_FORWARD_BACKWARD_SPEEDUP,
        "no_scene_degree_regression_over_5pct": minimum_speedup >= 1.0 - MAXIMUM_REGRESSION_FRACTION,
        "eligible": bool(
            median_speedup >= MINIMUM_FORWARD_BACKWARD_SPEEDUP
            and minimum_speedup >= 1.0 - MAXIMUM_REGRESSION_FRACTION
        ),
    }


def scene_candidate_policy(rows, scene, point_chunk):
    speedups = [
        rows[(scene, degree, point_chunk)]["paired_median_speedup"]["forward_backward"]
        for degree in DEGREES
    ]
    totals = [
        rows[(scene, degree, point_chunk)]["paired_median_speedup"]["total"]
        for degree in DEGREES
    ]
    median_speedup = float(statistics.median(speedups))
    minimum_speedup = min(speedups)
    return {
        "point_chunk": point_chunk,
        "median_forward_backward_speedup": median_speedup,
        "median_total_speedup": float(statistics.median(totals)),
        "minimum_degree_forward_backward_speedup": minimum_speedup,
        "median_improvement_gate_passed": median_speedup >= MINIMUM_FORWARD_BACKWARD_SPEEDUP,
        "no_degree_regression_over_5pct": minimum_speedup >= 1.0 - MAXIMUM_REGRESSION_FRACTION,
        "eligible": bool(
            median_speedup >= MINIMUM_FORWARD_BACKWARD_SPEEDUP
            and minimum_speedup >= 1.0 - MAXIMUM_REGRESSION_FRACTION
        ),
    }


def choose_candidate(candidates):
    eligible = {item["point_chunk"]: item for item in candidates if item["eligible"]}
    if not eligible:
        return REFERENCE_POINT_CHUNK, "no larger tile passed both promotion gates"
    if len(eligible) == 1:
        point_chunk = next(iter(eligible))
        return point_chunk, "only one larger tile passed both promotion gates"
    small = eligible[65536]["median_forward_backward_speedup"]
    large = eligible[131072]["median_forward_backward_speedup"]
    relative_gap = abs(small - large) / max(small, large)
    if relative_gap <= NEAR_TIE_FRACTION:
        return 65536, "64k and 128k were within 5%; selected 64k"
    if large > small:
        return 131072, "128k exceeded 64k by more than 5%"
    return 65536, "64k exceeded 128k by more than 5%"


def build_selection(rows):
    global_candidates = [
        candidate_policy(rows, point_chunk) for point_chunk in CANDIDATE_POINT_CHUNKS
    ]
    global_chunk, global_reason = choose_candidate(global_candidates)
    if global_chunk != REFERENCE_POINT_CHUNK:
        return {
            "scope": "global",
            "point_chunk": global_chunk,
            "reason": global_reason,
            "global_candidates": global_candidates,
            "per_scene": None,
        }
    per_scene = {}
    for scene in SCENES:
        candidates = [
            scene_candidate_policy(rows, scene, point_chunk)
            for point_chunk in CANDIDATE_POINT_CHUNKS
        ]
        point_chunk, reason = choose_candidate(candidates)
        per_scene[scene] = {
            "point_chunk": point_chunk,
            "reason": reason,
            "candidates": candidates,
        }
    return {
        "scope": "per_scene",
        "point_chunk": None,
        "reason": "no larger tile passed the global promotion gates",
        "global_candidates": global_candidates,
        "per_scene": per_scene,
    }


def audit(run_root, view_batch_size):
    if view_batch_size not in VIEW_BATCH_SIZES:
        fail(f"unsupported viewpoint batch size: {view_batch_size}")
    resolved_root, paths = discover_artifacts(run_root, view_batch_size)
    payloads = {key: load_json(path) for key, path in paths.items()}
    validate_cross_row_identity(payloads)
    rows = {
        key: validate_artifact(paths[key], payloads[key], key, view_batch_size)
        for key in sorted(paths)
    }
    selection = build_selection(rows)
    row_list = [rows[key] for key in sorted(rows)]
    for row in row_list:
        if row["point_chunk"] in CANDIDATE_POINT_CHUNKS:
            speedup = row["paired_median_speedup"]["forward_backward"]
            row["row_forward_backward_improvement_gate_passed"] = (
                speedup >= MINIMUM_FORWARD_BACKWARD_SPEEDUP
            )
    return {
        "schema": SUMMARY_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "source_schema": SOURCE_SCHEMA,
        "run_root": str(resolved_root),
        "view_batch_size": view_batch_size,
        "artifact_count": len(row_list),
        "audit_passed": True,
        "policy": {
            "candidate_median_forward_backward_speedup_minimum": MINIMUM_FORWARD_BACKWARD_SPEEDUP,
            "maximum_scene_degree_regression_fraction": MAXIMUM_REGRESSION_FRACTION,
            "prefer_64k_within_fraction": NEAR_TIE_FRACTION,
            "speedup_definition": "same_job_reference_32k_seconds_divided_by_candidate_seconds",
            "selection_statistic": "median_of_three_paired_repetition_speedups",
        },
        "rows": row_list,
        "selection": selection,
    }


def write_summary(path, payload):
    path = Path(path).resolve()
    if path.exists():
        fail(f"refusing to overwrite existing summary: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
    except FileExistsError:
        fail(f"refusing to overwrite existing summary: {path}")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--view-batch-size",
        required=True,
        type=int,
        choices=VIEW_BATCH_SIZES,
    )
    parser.add_argument(
        "--output",
        help="optional new JSON path; existing paths are never overwritten",
    )
    args = parser.parse_args(argv)
    try:
        summary = audit(args.run_root, args.view_batch_size)
        if args.output:
            output = write_summary(args.output, summary)
            summary = {**summary, "written_to": str(output)}
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    except AuditError as exc:
        parser.exit(2, f"point-tile audit failed: {exc}\n")


if __name__ == "__main__":
    main()
