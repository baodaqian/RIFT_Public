"""Run the bounded local GOTCHA raw-versus-source-AF Step-2 comparison.

This is a compare-only driver. It loads the expected Gate-1 shard, selects the
canonical training sector-002 identities, checks the complete finite kernel
budget, then computes exactly two full BPs and the declared 9-point
source/equivalent panel. It does not apply autofocus through Batch A, fit a
model, choose a target ROI, or touch manager/PACE state.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ACQ = _load_module("gotcha_acquisition_for_source_af_compare_v1", PROJECT_ROOT / "rift" / "gotcha_acquisition.py")
CTL = _load_module("gotcha_step2_controls_for_source_af_compare_v1", PROJECT_ROOT / "rift" / "gotcha_step2_controls.py")
SOURCE_AF = _load_module("gotcha_source_af_for_compare_v1", PROJECT_ROOT / "rift" / "gotcha_source_af.py")


PROTOCOL_SCHEMA = "rift_gotcha_step2_source_af_hh_sector002_h0_compare_protocol_v1"
DRIVER_SCHEMA = "rift_gotcha_step2_source_af_hh_sector002_h0_compare_driver_v1"
EXPECTED_DISCLOSURE = {
    "loaded_response_roles": ["train", "validation"],
    "used_response_roles": ["train"],
    "frequency_policy": "native_stored_exact",
    "frequency_vector_layout": "one_shared_exact_vector_per_shard",
    "shared_native_frequency_count": 424,
    "selected_response_shape": [117, 424],
    "validation_transformed": False,
    "validation_used_for_bp": False,
    "validation_used_for_scale_energy_peak_or_refinement": False,
    "validation_response_payload_materialized": True,
    "validation_used_for_panel": False,
    "validation_used_for_mean_native_sample_normalization": False,
    "validation_used_for_common_reference_or_display": False,
    "selected_train_rows_only_for_downstream_operations": True,
    "test_payload_opened": False,
    "test_response_payload_materialized": False,
    "test_sealing_contract": "archive_metadata_and_row_exclusion_not_independent_historical_exposure_proof",
    "target_cube_or_localization": False,
    "model_fit": False,
    "accuracy_focus_geometry_scores": False,
    "raw_source_change_diagnostics_only": True,
    "origin_only_diagnostic": True,
    "calibration_closure": False,
}
def _json_ready(value):
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_ready(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _read_protocol(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("protocol must be a JSON object")
    return value


def _assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise ValueError(f"protocol {label} must be {expected!r}; got {actual!r}")


def _validate_protocol(protocol: Mapping[str, object], stage: str) -> None:
    _assert_equal(stage, "compare", "CLI stage")
    _assert_equal(protocol.get("schema"), PROTOCOL_SCHEMA, "schema")
    _assert_equal(protocol.get("stage"), "compare", "stage")

    source = protocol.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("protocol source must be an object")
    for key, expected in {
        "archive_root_contract": "converted_v3_joint8_fullpol",
        "relative_path": "shards/pass1_hh.npz",
        "scene_id": "gotcha_v1_joint8_fullpol",
        "pass_id": 1,
        "polarization": "hh",
    }.items():
        _assert_equal(source.get(key), expected, f"source.{key}")

    selection = protocol.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("protocol selection must be an object")
    for key, expected in {
        "role": "train",
        "sector_id": 2,
        "canonical_identity": "(pass, polarization, sector, pulse)",
        "order": "ascending canonical identity",
        "all_native_pulses_in_sector": True,
        "expected_pulse_count": 117,
        "expected_native_frequency_count_per_pulse": 424,
        "loaded_roles": ["train", "validation"],
        "used_roles": ["train"],
        "test_payload_opened": False,
    }.items():
        _assert_equal(selection.get(key), expected, f"selection.{key}")

    native = protocol.get("native_contract")
    if not isinstance(native, Mapping):
        raise ValueError("protocol native_contract must be an object")
    for key, expected in {
        "frequency_policy": "native_stored_exact",
        "frequency_vector_layout": "one_shared_exact_vector_per_shard",
        "r0_field": "r0",
        "r0_unit": "m",
        "geometry": "paired_monostatic_tx_equals_rx_same_observation",
        "phase_hypothesis": CTL.PHASE_HYPOTHESIS_NAME,
        "phase_forward": CTL.PHASE_HYPOTHESIS_FORWARD,
        "phase_adjoint": CTL.PHASE_HYPOTHESIS_ADJOINT,
        "phase_status": CTL.PHASE_HYPOTHESIS_STATUS,
        "autofocus": "raw_unapplied_channel_owned",
        "autofocus_application": "source_only_downstream_representation; Gate-1 raw remains unchanged",
        "source_form": SOURCE_AF.SOURCE_AF_FORMULA,
        "fixed_r0_equivalent_form": SOURCE_AF.FIXED_R0_EQUIVALENT_FORMULA,
        "fixed_r0_equivalent_scope": "exact predeclared 9-point panel only",
        "combined_corrected_r0_response_compensation": False,
        "double_correction": False,
        "amplitude_weighting": False,
        "normalization": "mean_native_sample",
        "float64_geometry_r0_frequency_corrections": True,
        "complex128_response_and_accumulation": True,
        "raw_arrays_retained_unchanged": True,
        "raw_native_api_mutation": False,
    }.items():
        _assert_equal(native.get(key), expected, f"native_contract.{key}")

    support = protocol.get("support")
    if not isinstance(support, Mapping):
        raise ValueError("protocol support must be an object")
    _assert_equal(support.get("schema"), CTL.H0_SUPPORT_SCHEMA, "support.schema")
    _assert_equal(support.get("support_mode"), "plane_no_height", "support.support_mode")
    _assert_equal(support.get("bounds_m"), {"x": [-8.0, 8.0], "y": [-8.0, 8.0], "z": [0.0, 0.0]}, "support.bounds_m")
    _assert_equal(support.get("sampling"), {"spacing_m": [0.25, 0.25, 1.0], "shape": [65, 65, 1]}, "support.sampling")
    for key, expected in {
        "frame_contract": "antenna_xyz_unchanged",
        "registration_status": "unresolved",
        "support_status": CTL.SUPPORT_STATUS,
        "runtime_profile_origin_only": True,
        "target_roi_or_localization_claim": False,
        "max_kernel_evaluations": 250000000,
        "phase_hypothesis": CTL.PHASE_HYPOTHESIS_NAME,
        "autofocus_status": "raw_unapplied_channel_owned",
        "point_chunk_size": 4096,
    }.items():
        _assert_equal(support.get(key), expected, f"support.{key}")
    _assert_equal(support.get("units"), {"x": "m", "y": "m", "z": "m", "r0": "m", "frequency": "Hz"}, "support.units")

    panel = protocol.get("panel")
    if not isinstance(panel, Mapping):
        raise ValueError("protocol panel must be an object")
    for key, expected in {
        "coordinates_m": [-8.0, 0.0, 8.0],
        "z_m": 0.0,
        "point_count": 9,
        "representations": ["source_af", "fixed_r0_equivalent_panel_only"],
        "no_full_equivalent_bp": True,
        "comparison_quantity": "complex weighted contributions response*conjugate(kernel), then direct adjoint sums",
        "response_equality_comparison": False,
        "relative_tolerance": SOURCE_AF.DEFAULT_RELATIVE_TOLERANCE,
        "scaled_absolute_tolerance": SOURCE_AF.DEFAULT_SCALED_ABSOLUTE_TOLERANCE,
    }.items():
        _assert_equal(panel.get(key), expected, f"panel.{key}")

    budget = protocol.get("budget")
    if not isinstance(budget, Mapping):
        raise ValueError("protocol budget must be an object")
    for key, expected in {
        "expected_selected_pulse_count": 117,
        "expected_total_native_frequency_samples": 49608,
        "full_point_count": 4225,
        "raw_full_branch_kernel_evaluations": 209593800,
        "source_full_branch_kernel_evaluations": 209593800,
        "panel_two_representation_kernel_evaluations": 892944,
        "total_kernel_evaluations": 420080544,
        "max_full_branch_kernel_evaluations": 250000000,
        "max_total_kernel_evaluations": 500000000,
        "count_all_direct_kernels": True,
        "guard_before_correction_or_bp": True,
    }.items():
        _assert_equal(budget.get(key), expected, f"budget.{key}")
    if int(budget["raw_full_branch_kernel_evaluations"]) + int(budget["source_full_branch_kernel_evaluations"]) + int(budget["panel_two_representation_kernel_evaluations"]) != int(budget["total_kernel_evaluations"]):
        raise ValueError("budget total does not equal all direct kernels")

    outputs = protocol.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("protocol outputs must be an object")
    expected_outputs = {
        "raw_bp_unnormalized": "bp_raw_unnormalized.npy",
        "source_bp_unnormalized": "bp_source_unnormalized.npy",
        "raw_bp_mean_native_sample": "bp_raw_mean_native_sample.npy",
        "source_bp_mean_native_sample": "bp_source_mean_native_sample.npy",
        "x_coordinates": "x.npy",
        "y_coordinates": "y.npy",
        "plot": "bp_raw_source_compare.png",
        "plot_display": {
            "quantity": "20log10(abs(BP)/A_ref)",
            "A_ref": "max(abs(raw),abs(source))",
            "clip_db": [-40.0, 0.0],
            "common_scale": True,
            "all_zero_behavior": "A_ref=0 and both display fields are constant -40 dB; no convention is selected by appearance",
            "x_horizontal": True,
            "y_vertical": True,
            "origin": "lower",
            "aspect": "equal",
        },
    }
    _assert_equal(outputs, expected_outputs, "outputs")
    _assert_equal(protocol.get("disclosure"), EXPECTED_DISCLOSURE, "disclosure")


def _id_key(identity) -> tuple[int, str, int, int]:
    return (
        int(identity.pass_id),
        str(identity.polarization).lower(),
        int(identity.sector_id),
        int(identity.pulse_index),
    )


def _select_train_sector_observations(shard, protocol: Mapping[str, object]):
    selection = protocol["selection"]
    identities = tuple(shard.observation_ids)
    roles = tuple(str(value).lower() for value in np.asarray(shard.role).tolist())
    if len(identities) != len(roles):
        raise ValueError("native identity and role headers have different lengths")
    loaded_roles = sorted(set(roles))
    if loaded_roles != ["train", "validation"]:
        raise ValueError(f"loaded response roles must be train+validation; got {loaded_roles}")
    if any(role == "test" for role in roles):
        raise ValueError("test payload rows must remain sealed")
    role_counts = {role: int(sum(item == role for item in roles)) for role in loaded_roles}
    role_by_id = dict(zip(identities, roles))
    selected = tuple(sorted((identity for identity in identities if role_by_id[identity] == "train" and int(identity.sector_id) == 2), key=_id_key))
    if len(selected) != int(selection["expected_pulse_count"]):
        raise ValueError(f"sector 002 requires exactly {selection['expected_pulse_count']} train pulses; got {len(selected)}")
    if tuple(sorted(selected, key=_id_key)) != selected or len(set(selected)) != len(selected):
        raise ValueError("selected identities are not unique canonical sorted IDs")
    for identity in selected:
        if _id_key(identity)[:3] != (1, "hh", 2):
            raise ValueError(f"selected identity violates protocol: {identity}")
    observations = tuple(shard.observations(selected))
    if tuple(observation.identity for observation in observations) != selected:
        raise ValueError("materialized observations do not preserve selected canonical IDs")
    for observation in observations:
        if str(observation.role).lower() != "train" or _id_key(observation.identity)[:3] != (1, "hh", 2):
            raise ValueError("selected observation role/pass/polarization/sector mismatch")
    return selected, observations, loaded_roles, role_counts


def _validate_native_contract(shard, observations, protocol: Mapping[str, object]) -> None:
    native = protocol["native_contract"]
    expected_frequency_count = int(protocol["selection"]["expected_native_frequency_count_per_pulse"])
    if shard.phase_reference.frequency_values != native["frequency_policy"]:
        raise ValueError("native frequency provenance does not preserve the declared frequency policy")
    shard_frequencies = np.asarray(shard.frequencies_hz)
    if shard_frequencies.ndim != 1 or shard_frequencies.size != expected_frequency_count:
        raise ValueError(
            f"shared native frequency vector must have length {expected_frequency_count}"
        )
    shard_response = np.asarray(shard.response)
    if shard_response.shape != (int(shard.view_count), expected_frequency_count):
        raise ValueError(
            f"native response matrix must have shape [{int(shard.view_count)}, {expected_frequency_count}]"
        )
    if shard.phase_reference.reference_range_field != native["r0_field"]:
        raise ValueError("native r0 field does not match protocol")
    if shard.phase_reference.geometry_contract != native["geometry"]:
        raise ValueError("native geometry contract does not match protocol")
    if shard.autofocus.mode != ACQ.AUTOFOCUS_RAW or shard.autofocus.applied:
        raise ValueError("source-AF compare requires raw/unapplied autofocus provenance")
    if shard.autofocus.official_available is not True:
        raise ValueError("source-AF compare requires HH channel-owned raw autofocus arrays")
    for observation in observations:
        if observation.autofocus.applied:
            raise ValueError("selected observation has originally applied autofocus")
        if observation.autofocus.official_available is not True:
            raise ValueError("selected observation lacks channel-owned HH correction arrays")
        observation_frequencies = np.asarray(observation.frequencies_hz)
        if observation_frequencies.shape != (expected_frequency_count,):
            raise ValueError(
                f"selected observation frequency vector must have length {expected_frequency_count}"
            )
        if np.asarray(observation.response).shape != (expected_frequency_count,):
            raise ValueError(
                f"selected observation response must have shape [{expected_frequency_count}]"
            )
        if not np.array_equal(observation_frequencies, shard_frequencies):
            raise ValueError("selected observation frequency vector was not preserved exactly")
        if observation.r_correct_raw is None or observation.ph_correct_raw is None:
            raise ValueError("selected observation lacks raw HH correction values")


def _budget_before_correction(observations, protocol: Mapping[str, object]) -> dict[str, int | bool]:
    budget = protocol["budget"]
    count = int(len(observations))
    expected_frequency_count = int(protocol["selection"]["expected_native_frequency_count_per_pulse"])
    frequency_counts = [int(np.asarray(observation.frequencies_hz).size) for observation in observations]
    if any(value != expected_frequency_count for value in frequency_counts):
        raise ValueError(
            f"selected per-pulse frequency counts must all equal {expected_frequency_count}; got {frequency_counts}"
        )
    total_samples = int(sum(frequency_counts))
    if count != int(budget["expected_selected_pulse_count"]):
        raise ValueError("selected pulse count does not match budget")
    if total_samples != int(budget["expected_total_native_frequency_samples"]):
        raise ValueError("selected native frequency sample count does not match frozen budget")
    raw_full = int(budget["full_point_count"]) * total_samples
    panel_one = 9 * total_samples
    total = raw_full + raw_full + 2 * panel_one
    expected = {
        "raw_full_branch_kernel_evaluations": raw_full,
        "source_full_branch_kernel_evaluations": raw_full,
        "panel_two_representation_kernel_evaluations": 2 * panel_one,
        "total_kernel_evaluations": total,
    }
    for key, value in expected.items():
        if int(budget[key]) != value:
            raise ValueError(f"budget.{key} is inconsistent with selected native sample count")
    if raw_full > int(budget["max_full_branch_kernel_evaluations"]) or total > int(budget["max_total_kernel_evaluations"]):
        raise RuntimeError("source-AF kernel budget exceeded before correction/BP/panel computation")
    return {**expected, "total_native_frequency_samples": total_samples, "guard_passed_before_correction_or_bp": True}


def _loader_use_disclosure(
    shard,
    loaded_roles: list[str],
    loaded_role_counts: Mapping[str, int],
    selected_records,
) -> dict[str, object]:
    loaded_response = np.asarray(shard.response)
    loaded_roles_array = np.asarray(shard.role).astype(str)
    validation_materialized = bool(
        loaded_role_counts.get("validation", 0) > 0
        and loaded_response.ndim == 2
        and loaded_response.shape[0] == int(shard.view_count)
    )
    test_materialized = bool(np.any(loaded_roles_array == "test"))
    downstream_roles = {str(record.role).lower() for record in selected_records}
    return {
        "validation_response_payload_materialized": validation_materialized,
        "test_response_payload_materialized": test_materialized,
        "validation_used_for_panel": "validation" in downstream_roles,
        "validation_used_for_mean_native_sample_normalization": "validation" in downstream_roles,
        "validation_used_for_common_reference_or_display": "validation" in downstream_roles,
        "selected_train_rows_only_for_downstream_operations": downstream_roles == {"train"},
        "test_sealing_contract": EXPECTED_DISCLOSURE["test_sealing_contract"],
    }


def _require_pillow():
    try:
        from PIL import Image, ImageDraw
        from PIL.PngImagePlugin import PngInfo
    except Exception as error:
        raise RuntimeError("Pillow is required before the source-AF comparison archive is created") from error
    return Image, ImageDraw, PngInfo


def _write_pair_png(
    path: Path,
    raw_db: np.ndarray,
    source_db: np.ndarray,
    support,
    *,
    Image,
    ImageDraw,
    PngInfo,
    common_reference: float,
    header_lines: list[str] | tuple[str, ...] | None = None,
    panel_labels: tuple[str, str] | None = None,
    x_tick_values: tuple[float, float, float] = (-8.0, 0.0, 8.0),
    y_tick_values: tuple[float, float, float] = (-8.0, 0.0, 8.0),
    x_axis_label: str = "x (m)",
    y_axis_label: str = "y (m)",
) -> dict[str, object]:
    raw_db = np.asarray(raw_db)
    source_db = np.asarray(source_db)
    if raw_db.shape != source_db.shape:
        raise ValueError("complexity-free dB inputs for render must be same shape")
    if np.iscomplexobj(raw_db) or np.iscomplexobj(source_db):
        raise TypeError("_write_pair_png expects precomputed real-valued dB fields")
    raw_db = np.asarray(raw_db, dtype=np.float64)
    source_db = np.asarray(source_db, dtype=np.float64)
    if not np.isfinite(common_reference):
        raise ValueError("common_reference must be a finite scalar")
    if not np.all(np.isfinite(raw_db)):
        raise ValueError("raw_dB_db contains non-finite values")
    if not np.all(np.isfinite(source_db)):
        raise ValueError("source_dB_db contains non-finite values")
    raw_db = raw_db.reshape(tuple(support.grid_shape), order="C")[:, :, 0]
    source_db = source_db.reshape(tuple(support.grid_shape), order="C")[:, :, 0]
    all_zero = bool(np.all(raw_db == -40.0) and np.all(source_db == -40.0))

    def render(field: np.ndarray) -> np.ndarray:
        display = ((field + 40.0) / 40.0).T[::-1, :]
        color = np.empty((*display.shape, 3), dtype=np.uint8)
        color[:, :, 0] = np.asarray(np.clip(255.0 * display, 0.0, 255.0), dtype=np.uint8)
        color[:, :, 1] = np.asarray(np.clip(255.0 * (1.0 - np.abs(2.0 * display - 1.0)), 0.0, 255.0), dtype=np.uint8)
        color[:, :, 2] = np.asarray(np.clip(255.0 * (1.0 - display), 0.0, 255.0), dtype=np.uint8)
        return color

    scale = 4
    left = Image.fromarray(render(raw_db), mode="RGB").resize((raw_db.shape[0] * scale, raw_db.shape[1] * scale), resample=Image.Resampling.NEAREST)
    right = Image.fromarray(render(source_db), mode="RGB").resize((source_db.shape[0] * scale, source_db.shape[1] * scale), resample=Image.Resampling.NEAREST)
    if header_lines is None:
        header_lines = [
            "GOTCHA Step-2 compare | pass 1 | HH | train sector 002 | 117 pulses | 49,608 native samples | selected response shape [117, 424]",
            "native antenna frame | x/y coordinates (m) | z=0 m | per-pulse native r0 | one shared exact native frequency vector per shard | 424 frequencies",
            "phase: exp(-i 4*pi*f/c*(R-r0)) | source-AF: r0_src=r0_raw+r_correct; fp_src=fp_raw exp(+i ph_correct)",
            "common A_ref=max(abs(raw),abs(source)) | fixed display clip [-40,0] dB | no appearance-selected convention",
            "conditional diagnostic only | registration unresolved | not accuracy/focus/geometry/ROI/model-fit evidence",
        ]
    header_lines = [str(line) for line in header_lines]
    if panel_labels is not None and len(panel_labels) != 2:
        raise ValueError("pair renderer requires two panel labels")
    if len(x_tick_values) != 3 or len(y_tick_values) != 3:
        raise ValueError("pair renderer requires two panel labels and three x/y tick values")
    top = 16 + 16 * len(header_lines)
    bottom = 68
    colorbar_width = 64
    gap = 34
    measure = Image.new("RGB", (1, 1), "white")
    measure_draw = ImageDraw.Draw(measure)
    header_text_right = max(
        (int(measure_draw.textbbox((6, 4 + index * 16), line)[2]) for index, line in enumerate(header_lines)),
        default=6,
    )
    base_width = left.width + right.width + gap + colorbar_width + 12
    canvas_width = max(base_width, header_text_right + 12)
    canvas = Image.new("RGB", (canvas_width, top + left.height + bottom), "white")
    left_x = 6
    right_x = left_x + left.width + gap
    image_y = top
    canvas.paste(left, (left_x, image_y))
    canvas.paste(right, (right_x, image_y))
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(header_lines):
        draw.text((6, 4 + index * 16), line, fill="black")
    if panel_labels is None:
        # Preserve the historic default pixel/layout placement exactly.
        draw.text((left_x + left.width // 2 - 42, image_y - 14), "raw/unapplied AF", fill="black")
        draw.text((right_x + right.width // 2 - 48, image_y - 14), "source-AF downstream", fill="black")
    else:
        for origin_x, label in ((left_x, panel_labels[0]), (right_x, panel_labels[1])):
            label_box = draw.textbbox((0, 0), str(label))
            label_width = label_box[2] - label_box[0]
            draw.text((origin_x + (left.width - label_width) // 2, image_y - 14), str(label), fill="black")
    draw.rectangle((left_x, image_y, left_x + left.width - 1, image_y + left.height - 1), outline="black")
    draw.rectangle((right_x, image_y, right_x + right.width - 1, image_y + right.height - 1), outline="black")

    def draw_axes(origin_x: int) -> None:
        for fraction, value in zip((0.0, 0.5, 1.0), x_tick_values):
            x_tick = int(round(origin_x + fraction * (left.width - 1)))
            draw.line((x_tick, image_y + left.height, x_tick, image_y + left.height + 5), fill="black")
            draw.text((x_tick - 10, image_y + left.height + 8), f"{value:g}", fill="black")
        for fraction, value in zip((1.0, 0.5, 0.0), y_tick_values):
            y_tick = int(round(image_y + fraction * (left.height - 1)))
            draw.line((origin_x - 5, y_tick, origin_x, y_tick), fill="black")
            draw.text((origin_x - 29, y_tick - 5), f"{value:g}", fill="black")
        draw.text((origin_x + left.width // 2 - 18, image_y + left.height + 28), x_axis_label, fill="black")
        draw.text((origin_x - 28, image_y + left.height // 2), y_axis_label, fill="black")

    draw_axes(left_x)
    draw_axes(right_x)
    bar_x = right_x + right.width + 16
    bar_y = image_y
    bar_h = left.height
    bar_w = 16
    bar = np.linspace(1.0, 0.0, bar_h)[:, None]
    bar_rgb = np.empty((bar_h, bar_w, 3), dtype=np.uint8)
    bar_rgb[:, :, 0] = np.asarray(255.0 * bar, dtype=np.uint8)
    bar_rgb[:, :, 1] = np.asarray(255.0 * (1.0 - np.abs(2.0 * bar - 1.0)), dtype=np.uint8)
    bar_rgb[:, :, 2] = np.asarray(255.0 * (1.0 - bar), dtype=np.uint8)
    canvas.paste(Image.fromarray(bar_rgb, mode="RGB"), (bar_x, bar_y))
    draw.text((bar_x - 2, bar_y - 14), "dB", fill="black")
    for fraction, label in ((0.0, "0"), (0.5, "-20"), (1.0, "-40")):
        y_tick = int(round(bar_y + fraction * (bar_h - 1)))
        draw.line((bar_x + bar_w, y_tick, bar_x + bar_w + 5, y_tick), fill="black")
        draw.text((bar_x + bar_w + 8, y_tick - 5), label, fill="black")
    info = PngInfo()
    info.add_text("Title", header_lines[0])
    info.add_text("Description", "\n".join(header_lines))
    if x_tick_values == (-8.0, 0.0, 8.0) and y_tick_values == (-8.0, 0.0, 8.0) and x_axis_label == "x (m)" and y_axis_label == "y (m)":
        axes_description = "native antenna x/y coordinates in metres; z=0; ticks=-8,0,8 m; origin=lower; aspect=equal"
    else:
        axes_description = (
            "native antenna x/y coordinates in metres; "
            f"x_ticks={tuple(float(value) for value in x_tick_values)}; "
            f"y_ticks={tuple(float(value) for value in y_tick_values)}; origin=lower; aspect=equal"
        )
    info.add_text("Axes", axes_description)
    info.add_text("Display", "20log10(abs(BP)/A_ref), common A_ref=max(abs(raw),abs(source)), fixed clip [-40,0] dB")
    info.add_text("AllZeroBehavior", "A_ref=0 yields constant -40 dB in both panels" if all_zero else "not all zero")
    canvas.save(path, format="PNG", pnginfo=info)
    geometry = {
        "header_line_count": len(header_lines),
        "header_line_height_px": 16,
        "header_top_padding_px": 16,
        "header_text_right_px": header_text_right,
        "canvas_width_px": canvas_width,
        "top_px": top,
        "bottom_px": bottom,
        "scale": scale,
        "grid_shape": list(support.grid_shape),
        "panel_size_px": [left.width, left.height],
        "left_origin_px": [left_x, image_y],
        "right_origin_px": [right_x, image_y],
    }
    return {
        "backend": "Pillow_dependency_light",
        "display_quantity": "20log10(abs(BP)/A_ref)",
        "A_ref": float(common_reference),
        "db_clip": [-40.0, 0.0],
        "common_scale": True,
        "all_zero_behavior": "constant -40 dB in both panels" if all_zero else "not all zero",
        "origin": "lower",
        "aspect": "equal",
        "x_horizontal": True,
        "y_vertical": True,
        "header_lines": header_lines,
        "geometry": geometry,
        "visible_x_axis_ticks_m": [float(value) for value in x_tick_values],
        "visible_y_axis_ticks_m": [float(value) for value in y_tick_values],
        "visible_axis_ticks_m": [float(value) for value in x_tick_values] if x_tick_values == y_tick_values else None,
        "visible_db_legend": [0.0, -20.0, -40.0],
    }


def run_compare(protocol_path: str | Path, archive_root: str | Path, output_dir: str | Path, stage: str = "compare") -> dict[str, object]:
    protocol = _read_protocol(Path(protocol_path))
    _validate_protocol(protocol, stage)
    Image, ImageDraw, PngInfo = _require_pillow()
    archive_root = Path(archive_root)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory must be fresh: {output_dir}")
    relative = Path(str(protocol["source"]["relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("protocol source path must be a safe relative path")
    archive_path = archive_root / relative
    if not archive_path.is_file():
        raise FileNotFoundError(f"protocol archive is missing: {archive_path}")

    shard = ACQ.load_native_shard(
        archive_path,
        expected_pass_id=1,
        expected_polarization="hh",
        expected_scene_id=str(protocol["source"]["scene_id"]),
    )
    selected_ids, observations, loaded_roles, loaded_role_counts = _select_train_sector_observations(shard, protocol)
    _validate_native_contract(shard, observations, protocol)
    if shard.metadata.get("test_opened") is not False or shard.metadata.get("test_payload_included") is not False:
        raise ValueError("Gate-1 metadata does not prove that test payload remained sealed")

    support_decl = dict(protocol["support"])
    support = CTL.NativeFrameH0Support.from_mapping(support_decl)
    if support.point_count != 4225:
        raise ValueError("source-AF compare support must contain exactly 4225 full-grid points")
    preflight = CTL.preflight_support_declaration(support_decl, observations, point_count=support.point_count)
    budget_before = _budget_before_correction(observations, protocol)

    # Only after selection, native validation, support preflight, and the total
    # direct-kernel guard do we construct the downstream corrected view.
    source_records = SOURCE_AF.build_source_af(observations, expected_count=117)
    budget = SOURCE_AF.budget_report(
        source_records,
        full_point_count=support.point_count,
        panel_point_count=9,
        max_total_kernel_evaluations=int(protocol["budget"]["max_total_kernel_evaluations"]),
        max_branch_kernel_evaluations=int(protocol["budget"]["max_full_branch_kernel_evaluations"]),
    )
    if budget != {**budget_before, "max_full_branch_kernel_evaluations": 250000000, "max_total_kernel_evaluations": 500000000}:
        raise ValueError("post-construction budget does not match the pre-correction guard")

    points = support.grid_points()
    started = time.perf_counter()
    raw_unnormalized = SOURCE_AF.direct_backproject(
        source_records,
        points,
        representation=SOURCE_AF.RAW_REPRESENTATION,
        normalization="none",
        point_chunk_size=int(protocol["support"]["point_chunk_size"]),
        max_kernel_evaluations=int(protocol["budget"]["max_full_branch_kernel_evaluations"]),
    )
    source_unnormalized = SOURCE_AF.direct_backproject(
        source_records,
        points,
        representation=SOURCE_AF.SOURCE_REPRESENTATION,
        normalization="none",
        point_chunk_size=int(protocol["support"]["point_chunk_size"]),
        max_kernel_evaluations=int(protocol["budget"]["max_full_branch_kernel_evaluations"]),
    )
    panel = SOURCE_AF.panel_equivalence_report(
        source_records,
        relative_tolerance=float(protocol["panel"]["relative_tolerance"]),
        scaled_absolute_tolerance=float(protocol["panel"]["scaled_absolute_tolerance"]),
    )
    elapsed = time.perf_counter() - started
    if not panel["contribution_metrics"]["passed"] or not panel["adjoint_metrics"]["passed"]:
        raise RuntimeError("source-AF fixed-r0 panel equivalence failed")

    total_samples = int(budget["total_native_frequency_samples"])
    raw_reference = float(np.max(np.abs(raw_unnormalized))) if raw_unnormalized.size else 0.0
    source_reference = float(np.max(np.abs(source_unnormalized))) if source_unnormalized.size else 0.0
    common_reference = float(max(raw_reference, source_reference))
    if common_reference == 0.0:
        raw_db = np.full(raw_unnormalized.shape, -40.0, dtype=np.float64)
        source_db = np.full(source_unnormalized.shape, -40.0, dtype=np.float64)
    else:
        floor = 10.0 ** (-40.0 / 20.0)
        raw_db = np.clip(
            20.0 * np.log10(np.maximum(np.abs(raw_unnormalized) / common_reference, floor)),
            -40.0,
            0.0,
        )
        source_db = np.clip(
            20.0 * np.log10(np.maximum(np.abs(source_unnormalized) / common_reference, floor)),
            -40.0,
            0.0,
        )
    raw_mean = np.asarray(raw_unnormalized, dtype=np.complex128) / float(total_samples)
    source_mean = np.asarray(source_unnormalized, dtype=np.complex128) / float(total_samples)
    x = np.linspace(-8.0, 8.0, 65, dtype=np.float64)
    y = np.linspace(-8.0, 8.0, 65, dtype=np.float64)
    output_dir.mkdir(parents=True, exist_ok=False)
    display_metadata = _write_pair_png(
        output_dir / "bp_raw_source_compare.png",
        raw_db,
        source_db,
        support,
        Image=Image,
        ImageDraw=ImageDraw,
        PngInfo=PngInfo,
        common_reference=common_reference,
    )
    np.save(output_dir / "bp_raw_unnormalized.npy", np.asarray(raw_unnormalized, dtype=np.complex128))
    np.save(output_dir / "bp_source_unnormalized.npy", np.asarray(source_unnormalized, dtype=np.complex128))
    np.save(output_dir / "bp_raw_mean_native_sample.npy", raw_mean)
    np.save(output_dir / "bp_source_mean_native_sample.npy", source_mean)
    np.save(output_dir / "x.npy", x)
    np.save(output_dir / "y.npy", y)

    disclosure = {
        **EXPECTED_DISCLOSURE,
        "loaded_response_roles": loaded_roles,
        "loaded_response_role_counts": loaded_role_counts,
        "loaded_response_count": int(shard.view_count),
        "used_response_roles": ["train"],
        "selected_response_count": len(observations),
        "selected_response_shape": [
            len(observations),
            int(protocol["selection"]["expected_native_frequency_count_per_pulse"]),
        ],
        "selected_ids": [identity.as_dict() for identity in selected_ids],
        "selected_r0_m": [
            {"id": observation.identity.as_dict(), "r0_m": float(observation.r0_m)}
            for observation in observations
        ],
        "selected_frequency_counts": [int(record.frequencies_hz.size) for record in source_records],
        **_loader_use_disclosure(shard, loaded_roles, loaded_role_counts, source_records),
        "archive_units_disclosure": protocol["source"]["archive_units_disclosure"],
    }
    _write_json(output_dir / "protocol_echo.json", protocol)
    _write_json(
        output_dir / "preflight.json",
        {"support_preflight": preflight, "budget": budget, "budget_before_correction": budget_before, "disclosure": disclosure},
    )
    _write_json(
        output_dir / "resource.json",
        {
            "schema": "rift_gotcha_step2_source_af_compare_resource_v1",
            "budget": budget,
            "point_chunk_size": int(protocol["support"]["point_chunk_size"]),
            "point_chunks_per_observation": int((support.point_count + int(protocol["support"]["point_chunk_size"]) - 1) // int(protocol["support"]["point_chunk_size"])),
            "selected_pulse_count": len(observations),
            "elapsed_seconds": elapsed,
            "guard_passed_before_bp": True,
        },
    )
    _write_json(
        output_dir / "comparison_report.json",
        {
            "schema": "rift_gotcha_step2_source_af_compare_report_v1",
            "status": "PASS",
            "diagnostic_statement": "origin-only raw/source change diagnostic and panel algebra equivalence only—not a target cube, fitted field, localization, calibration closure, focus/geometry/accuracy result, registration result, target-ROI, or model-fit evidence",
            "selected_pulse_count": len(observations),
            "selected_frequency_samples": total_samples,
            "frequency_policy": protocol["native_contract"]["frequency_policy"],
            "frequency_vector_layout": protocol["native_contract"]["frequency_vector_layout"],
            "shared_native_frequency_count": int(protocol["selection"]["expected_native_frequency_count_per_pulse"]),
            "selected_response_shape": [
                len(observations),
                int(protocol["selection"]["expected_native_frequency_count_per_pulse"]),
            ],
            "native_per_pulse_r0": True,
            "autofocus_raw_input": "raw/unapplied channel-owned HH",
            "source_representation": SOURCE_AF.SOURCE_AF_FORMULA,
            "fixed_r0_equivalent_representation": SOURCE_AF.FIXED_R0_EQUIVALENT_FORMULA,
            "normalization": "mean_native_sample",
            "unnormalized_arrays_preserved": True,
            "support": support.as_dict(),
            "correction_stats": SOURCE_AF.correction_stats(source_records),
            "unit_modulus_sample_energy": SOURCE_AF.unit_modulus_sample_energy(source_records),
            "panel_equivalence": panel,
            "change_diagnostics": SOURCE_AF.change_diagnostics(raw_unnormalized, source_unnormalized),
            "display": display_metadata,
            "disclosure": disclosure,
        },
    )
    status = {
        "schema": DRIVER_SCHEMA,
        "status": "PASS",
        "stage": "compare",
        "output_dir": str(output_dir),
        "files": [
            "protocol_echo.json",
            "preflight.json",
            "resource.json",
            "comparison_report.json",
            "status.json",
            "bp_raw_unnormalized.npy",
            "bp_source_unnormalized.npy",
            "bp_raw_mean_native_sample.npy",
            "bp_source_mean_native_sample.npy",
            "x.npy",
            "y.npy",
            "bp_raw_source_compare.png",
        ],
        "test_payload_opened": False,
        "validation_used_for_bp": False,
        "future_auto_chain": False,
    }
    _write_json(output_dir / "status.json", status)
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args(argv)
    run_compare(args.protocol, args.archive_root, args.output_dir, args.stage)
    print(json.dumps({"status": "PASS", "output_dir": str(Path(args.output_dir))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
