"""Run the bounded local 15TR-07 nominal-point raw/source-AF diagnostic.

This driver is conditional and metadata-led.  It selects only the fixed P1/HH
train sectors 268--270, derives the native geometry/frequency preflight, then
computes four full BPs and two matched synthetic unit-point PSFs.  It never
searches for a response condition, localizes a peak, fits a model, or opens a
measured GOTCHA payload in this test-owned workflow.
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
HISTORIC_DRIVER = None


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Reuse the existing acquisition/operator imports and the parameterized
# renderer; this package deliberately has no second renderer implementation.
HISTORIC_DRIVER = _load_module(
    "gotcha_source_af_compare_driver_for_15tr07_v1",
    PROJECT_ROOT / "scripts" / "run_gotcha_step2_source_af_compare_v1.py",
)
ACQ = HISTORIC_DRIVER.ACQ
CTL = HISTORIC_DRIVER.CTL
SOURCE_AF = HISTORIC_DRIVER.SOURCE_AF


PROTOCOL_SCHEMA = "rift_gotcha_step2_15tr07_nominal_psf_p1_hh_train_protocol_v1"
DRIVER_SCHEMA = "rift_gotcha_step2_15tr07_nominal_psf_driver_v1"
EXPECTED_SECTORS = (268, 269, 270)
EXPECTED_VALIDATION_SECTORS = (271,)
NOMINAL_POINT_C_M = np.asarray((-5.12, 22.98, -0.05), dtype=np.float64)
SUPPORT_X_TICKS = (-6.42, -5.12, -3.82)
SUPPORT_Y_TICKS = (21.68, 22.98, 24.28)


def _json_ready(value):
    return HISTORIC_DRIVER._json_ready(value)


def _write_json(path: Path, value) -> None:
    HISTORIC_DRIVER._write_json(path, value)


def _assert_equal(actual, expected, label: str) -> None:
    HISTORIC_DRIVER._assert_equal(actual, expected, label)


def _read_protocol(path: Path) -> dict:
    return HISTORIC_DRIVER._read_protocol(path)


def _validate_protocol(protocol: Mapping[str, object], stage: str) -> None:
    _assert_equal(stage, "diagnostic", "CLI stage")
    _assert_equal(protocol.get("schema"), PROTOCOL_SCHEMA, "schema")
    _assert_equal(protocol.get("stage"), "diagnostic", "stage")

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
        "sector_ids": list(EXPECTED_SECTORS),
        "contiguous_train_sector_aperture": True,
        "working_aperture_degrees": [267.0, 270.0],
        "validation_sector_ids": list(EXPECTED_VALIDATION_SECTORS),
        "validation_role_unused": True,
        "canonical_identity": "(pass, polarization, sector, pulse)",
        "order": "ascending canonical identity",
        "expected_pulse_count": None,
        "expected_native_frequency_count_per_pulse": 424,
        "loaded_roles": ["train", "validation"],
        "used_roles": ["train"],
        "test_payload_opened": False,
    }.items():
        _assert_equal(selection.get(key), expected, f"selection.{key}")

    nominal = protocol.get("nominal_point_c")
    if not isinstance(nominal, Mapping):
        raise ValueError("protocol nominal_point_c must be an object")
    _assert_equal(nominal.get("xyz_m"), [-5.12, 22.98, -0.05], "nominal_point_c.xyz_m")
    _assert_equal(nominal.get("z_plane_m"), -0.05, "nominal_point_c.z_plane_m")
    _assert_equal(nominal.get("heading_degrees"), 270.0, "nominal_point_c.heading_degrees")
    _assert_equal(nominal.get("truth_or_phase_center_claim"), False, "nominal_point_c.truth_or_phase_center_claim")
    if nominal.get("identity_mapping_status") != "unverified working hypothesis":
        raise ValueError("nominal Point C identity mapping must remain unverified")
    if nominal.get("heading_convention_status") != "unverified working hypothesis; sector index near 270 is not verified alignment":
        raise ValueError("nominal heading convention must remain unverified")

    native = protocol.get("native_contract")
    if not isinstance(native, Mapping):
        raise ValueError("protocol native_contract must be an object")
    for key, expected in {
        "frequency_policy": "native_stored_exact",
        "frequency_vector_layout": "one_shared_exact_vector_per_shard",
        "r0_field": "r0",
        "geometry": "paired_monostatic_tx_equals_rx_same_observation",
        "phase_hypothesis": CTL.PHASE_HYPOTHESIS_NAME,
        "phase_forward": CTL.PHASE_HYPOTHESIS_FORWARD,
        "phase_adjoint": CTL.PHASE_HYPOTHESIS_ADJOINT,
        "phase_status": CTL.PHASE_HYPOTHESIS_STATUS,
        "autofocus": "raw_unapplied_channel_owned",
        "source_form": SOURCE_AF.SOURCE_AF_FORMULA,
        "interpolation": False,
        "regridding": False,
        "windowing": False,
        "unit_weights": True,
        "per_pulse_r0_exact_stored": True,
        "raw_arrays_retained_unchanged": True,
        "raw_native_api_mutation": False,
    }.items():
        _assert_equal(native.get(key), expected, f"native_contract.{key}")

    support = protocol.get("support")
    if not isinstance(support, Mapping):
        raise ValueError("protocol support must be an object")
    _assert_equal(support.get("schema"), CTL.H0_SUPPORT_SCHEMA, "support.schema")
    _assert_equal(support.get("support_mode"), "plane_no_height", "support.support_mode")
    _assert_equal(
        support.get("bounds_m"),
        {"x": [-6.42, -3.82], "y": [21.68, 24.28], "z": [-0.05, -0.05]},
        "support.bounds_m",
    )
    _assert_equal(
        support.get("sampling"),
        {"spacing_m": [0.1, 0.1, 1.0], "shape": [27, 27, 1]},
        "support.sampling",
    )
    for key, expected in {
        "frame_contract": "antenna_xyz_unchanged",
        "registration_status": "unresolved",
        "support_status": CTL.SUPPORT_STATUS,
        "runtime_profile_origin_only": False,
        "target_roi_or_localization_claim": False,
        "max_kernel_evaluations": 250000000,
        "phase_hypothesis": CTL.PHASE_HYPOTHESIS_NAME,
        "autofocus_status": "raw_unapplied_channel_owned",
        "point_chunk_size": 4096,
    }.items():
        _assert_equal(support.get(key), expected, f"support.{key}")
    _assert_equal(support.get("units"), {"x": "m", "y": "m", "z": "m", "r0": "m", "frequency": "Hz"}, "support.units")
    if "support_interpretation" not in support or "not a survey-to-phase-center displacement bound" not in str(support["support_interpretation"]):
        raise ValueError("support must disclose its bounded-diagnostic interpretation")
    if "pitch_interpretation" not in support or "not a coherent field-fitting quadrature claim" not in str(support["pitch_interpretation"]):
        raise ValueError("support must disclose the non-quadrature pitch interpretation")

    preflight = protocol.get("preflight")
    if not isinstance(preflight, Mapping):
        raise ValueError("protocol preflight must be an object")
    for key, expected in {
        "selection_basis": "metadata identity/role and fixed protocol scope only",
        "response_condition_search": False,
        "brightness_selection": False,
        "sampling_proxy_status": "metadata-only proxy; inconclusive/non-resolving",
        "result_status_when_weak": "inconclusive_non_resolving",
    }.items():
        _assert_equal(preflight.get(key), expected, f"preflight.{key}")

    budget = protocol.get("budget")
    if not isinstance(budget, Mapping):
        raise ValueError("protocol budget must be an object")
    for key, expected in {
        "full_point_count": 729,
        "total_formula": "4*S*N+2*N",
        "expanded_formula": "2918*N",
        "max_total_kernel_evaluations": 500000000,
        "max_branch_kernel_evaluations": 250000000,
        "max_N_under_total_cap": 171350,
        "max_records_at_424_bins_under_total_cap": 404,
        "panel_kernel_evaluations": 0,
        "probe_kernel_evaluations": 0,
        "fit_kernel_evaluations": 0,
        "count_all_direct_kernels": True,
        "guard_before_response_conversion_or_bp": True,
    }.items():
        _assert_equal(budget.get(key), expected, f"budget.{key}")

    outputs = protocol.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("protocol outputs must be an object")
    for key in (
        "raw_bp_unnormalized", "source_bp_unnormalized", "raw_bp_mean_native_sample",
        "source_bp_mean_native_sample", "raw_psf_unnormalized", "source_psf_unnormalized",
        "raw_psf_mean_native_sample", "source_psf_mean_native_sample", "x_coordinates",
        "y_coordinates", "bp_plot", "psf_plot",
    ):
        if not isinstance(outputs.get(key), str) or not outputs[key]:
            raise ValueError(f"outputs.{key} must be a nonempty filename")

    disclosure = protocol.get("disclosure")
    if not isinstance(disclosure, Mapping):
        raise ValueError("protocol disclosure must be an object")
    for key, expected in {
        "loaded_response_roles": ["train", "validation"],
        "used_response_roles": ["train"],
        "validation_sector_ids": [271],
        "validation_used_for_bp": False,
        "selected_train_rows_only_for_downstream_operations": True,
        "test_payload_opened": False,
        "test_response_payload_materialized": False,
        "nominal_point_identity_mapping_unverified": True,
        "heading_convention_unverified": True,
        "nominal_point_is_not_truth_or_phase_center": True,
        "support_is_not_a_phase_center_displacement_bound": True,
        "pitch_is_not_coherent_quadrature_claim": True,
        "raw_source_change_diagnostics_only": True,
        "matched_psf_operator_sanity_only": True,
        "target_cube_or_localization": False,
        "peak_offset_or_accuracy_output": False,
        "registration_or_calibration_closure": False,
        "model_fit": False,
        "conditional_result": True,
    }.items():
        _assert_equal(disclosure.get(key), expected, f"disclosure.{key}")


def _id_key(identity) -> tuple[int, str, int, int]:
    return HISTORIC_DRIVER._id_key(identity)


def _select_metadata_scope(shard, protocol: Mapping[str, object]):
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
    selected_sector_ids = tuple(int(value) for value in selection["sector_ids"])
    selected = tuple(
        sorted(
            (
                identity
                for identity, role in zip(identities, roles)
                if role == "train" and int(identity.sector_id) in selected_sector_ids
            ),
            key=_id_key,
        )
    )
    if not selected:
        raise ValueError("fixed 268--270 train scope selected no observations")
    selected_sector_set = {int(identity.sector_id) for identity in selected}
    if selected_sector_set != set(EXPECTED_SECTORS):
        raise ValueError(
            "fixed 268--270 train scope requires every declared sector exactly once in the selected set; "
            f"got {sorted(selected_sector_set)}"
        )
    if any(int(identity.sector_id) not in selected_sector_ids for identity in selected):
        raise ValueError("selected identity escaped the immutable sector scope")
    if any(int(identity.sector_id) in EXPECTED_VALIDATION_SECTORS for identity in selected):
        raise ValueError("validation sector 271 must remain unused")
    if tuple(sorted(selected, key=_id_key)) != selected or len(set(selected)) != len(selected):
        raise ValueError("selected identities are not unique canonical sorted IDs")
    for identity in selected:
        if _id_key(identity)[:3] != (1, "hh", int(identity.sector_id)):
            raise ValueError(f"selected identity violates pass/polarization scope: {identity}")
    if not any(int(identity.sector_id) in EXPECTED_VALIDATION_SECTORS and role == "validation" for identity, role in zip(identities, roles)):
        raise ValueError("validation sector 271 must be present in the loaded metadata scope")
    return selected, loaded_roles, role_counts


def _metadata_preflight(shard, selected_ids, protocol: Mapping[str, object], support) -> dict[str, object]:
    index_by_id = {identity: index for index, identity in enumerate(shard.observation_ids)}
    indices = np.asarray([index_by_id[identity] for identity in selected_ids], dtype=np.int64)
    positions = np.column_stack((shard.x[indices], shard.y[indices], shard.z[indices])).astype(np.float64)
    target = NOMINAL_POINT_C_M
    relative = positions - target[None, :]
    horizontal_distance = np.linalg.norm(relative[:, :2], axis=1)
    ranges = np.linalg.norm(relative, axis=1)
    bearings = np.mod(np.rad2deg(np.arctan2(relative[:, 1], relative[:, 0])), 360.0)
    elevations = np.rad2deg(np.arctan2(relative[:, 2], np.maximum(horizontal_distance, np.finfo(np.float64).tiny)))
    bearing_reference = float(np.rad2deg(np.arctan2(np.mean(relative[:, 1]), np.mean(relative[:, 0]))) % 360.0)
    unwrapped_bearings = bearing_reference + ((bearings - bearing_reference + 180.0) % 360.0 - 180.0)
    frequencies = np.asarray(shard.frequencies_hz, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size != 424:
        raise ValueError("metadata preflight requires the exact 424-sample native frequency vector")
    bandwidth = float(frequencies[-1] - frequencies[0])
    center_frequency = float(np.mean(frequencies))
    horizontal_span_deg = float(np.max(unwrapped_bearings) - np.min(unwrapped_bearings))
    range_proxy = float(ACQ.SPEED_OF_LIGHT_M_S / (2.0 * bandwidth)) if bandwidth > 0 else None
    if horizontal_span_deg > 0.0:
        half_angle = np.deg2rad(horizontal_span_deg / 2.0)
        cross_range_proxy = float((ACQ.SPEED_OF_LIGHT_M_S / center_frequency) / (4.0 * np.sin(half_angle)))
    else:
        cross_range_proxy = None
    native_th = np.asarray(shard.th[indices], dtype=np.float64)
    native_phi = np.asarray(shard.phi[indices], dtype=np.float64)
    return {
        "support_preflight": {
            "support": support.as_dict(),
            "support_fixed_before_response_conversion": True,
            "point_count": support.point_count,
            "bounded_support_interpretation": str(protocol["support"]["support_interpretation"]),
            "pitch_interpretation": str(protocol["support"]["pitch_interpretation"]),
        },
        "selected_record_count": int(len(selected_ids)),
        "selected_ids": [identity.as_dict() for identity in selected_ids],
        "frequency_samples_per_record": int(frequencies.size),
        "total_native_frequency_samples": int(len(selected_ids) * frequencies.size),
        "frequency_vector_policy": "native_stored_exact",
        "frequency_vector_layout": "one_shared_exact_vector_per_shard",
        "actual_frequency_span_hz": [float(frequencies[0]), float(frequencies[-1])],
        "actual_frequency_center_hz": center_frequency,
        "actual_frequency_count": int(frequencies.size),
        "nominal_point_c_xyz_m": target.tolist(),
        "nominal_identity_mapping": "workbook coordinates mapped directly to native antenna xyz frame",
        "nominal_identity_mapping_status": "unverified working hypothesis",
        "target_to_antenna_horizontal_bearings_deg": bearings.tolist(),
        "target_to_antenna_elevations_deg": elevations.tolist(),
        "horizontal_bearing_summary_deg": {
            "minimum_unwrapped": float(np.min(unwrapped_bearings)),
            "maximum_unwrapped": float(np.max(unwrapped_bearings)),
            "span": horizontal_span_deg,
            "reference": bearing_reference,
        },
        "elevation_summary_deg": {
            "minimum": float(np.min(elevations)),
            "maximum": float(np.max(elevations)),
            "span": float(np.max(elevations) - np.min(elevations)),
        },
        "range_to_nominal_c_summary_m": {
            "minimum": float(np.min(ranges)),
            "maximum": float(np.max(ranges)),
            "span": float(np.max(ranges) - np.min(ranges)),
        },
        "native_look_angle_fields_deg": {
            "th_minimum": float(np.min(native_th)),
            "th_maximum": float(np.max(native_th)),
            "th_span": float(np.max(native_th) - np.min(native_th)),
            "phi_minimum": float(np.min(native_phi)),
            "phi_maximum": float(np.max(native_phi)),
            "phi_span": float(np.max(native_phi) - np.min(native_phi)),
        },
        "metadata_only_sampling_proxy": {
            "range_proxy_m": range_proxy,
            "cross_range_proxy_m": cross_range_proxy,
            "bandwidth_hz": bandwidth,
            "center_frequency_hz": center_frequency,
            "formula_status": "metadata-only proxy; not a resolving-power or focus result",
        },
        "response_condition_search": False,
        "brightness_selection": False,
        "heading_convention": "270-degree workbook heading is an unverified working hypothesis; sector index near 270 is not verified alignment",
        "status": "inconclusive_non_resolving_metadata_only",
    }


def _validate_native_contract(shard, observations, protocol: Mapping[str, object]) -> None:
    expected_frequency_count = int(protocol["selection"]["expected_native_frequency_count_per_pulse"])
    native = protocol["native_contract"]
    if shard.phase_reference.frequency_values != native["frequency_policy"]:
        raise ValueError("native frequency provenance does not preserve the declared exact policy")
    shard_frequencies = np.asarray(shard.frequencies_hz)
    if shard_frequencies.shape != (expected_frequency_count,):
        raise ValueError(f"shared native frequency vector must have length {expected_frequency_count}")
    if np.asarray(shard.response).shape != (int(shard.view_count), expected_frequency_count):
        raise ValueError("native response matrix shape does not match the exact shared frequency vector")
    if shard.phase_reference.reference_range_field != native["r0_field"]:
        raise ValueError("native r0 field does not match protocol")
    if shard.phase_reference.geometry_contract != native["geometry"]:
        raise ValueError("native geometry contract does not match protocol")
    if shard.autofocus.mode != ACQ.AUTOFOCUS_RAW or shard.autofocus.applied or shard.autofocus.official_available is not True:
        raise ValueError("source-AF diagnostic requires raw/unapplied HH correction provenance")
    for observation in observations:
        if str(observation.role).lower() != "train" or int(observation.identity.sector_id) not in EXPECTED_SECTORS:
            raise ValueError("materialized observation escaped the fixed train scope")
        if observation.autofocus.applied or observation.autofocus.official_available is not True:
            raise ValueError("selected observation lacks raw/unapplied HH correction provenance")
        frequencies = np.asarray(observation.frequencies_hz)
        if frequencies.shape != (expected_frequency_count,) or not np.array_equal(frequencies, shard_frequencies):
            raise ValueError("selected observation did not preserve the exact shared native frequency vector")
        if np.asarray(observation.response).shape != (expected_frequency_count,):
            raise ValueError("selected response shape does not match the native frequency vector")
        if observation.r_correct_raw is None or observation.ph_correct_raw is None:
            raise ValueError("selected HH observation lacks channel-owned raw correction values")


def _db_pair(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    left = np.asarray(left, dtype=np.complex128)
    right = np.asarray(right, dtype=np.complex128)
    reference = float(max(np.max(np.abs(left)) if left.size else 0.0, np.max(np.abs(right)) if right.size else 0.0))
    if reference == 0.0:
        return (
            np.full(left.shape, -40.0, dtype=np.float64),
            np.full(right.shape, -40.0, dtype=np.float64),
            reference,
        )
    floor = 10.0 ** (-40.0 / 20.0)
    return (
        np.clip(20.0 * np.log10(np.maximum(np.abs(left) / reference, floor)), -40.0, 0.0),
        np.clip(20.0 * np.log10(np.maximum(np.abs(right) / reference, floor)), -40.0, 0.0),
        reference,
    )


def _pair_difference(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    difference = np.asarray(left, dtype=np.complex128) - np.asarray(right, dtype=np.complex128)
    return {
        "max_abs_difference": float(np.max(np.abs(difference))) if difference.size else 0.0,
        "l2_difference": float(np.linalg.norm(difference)),
        "left_l2": float(np.linalg.norm(left)),
        "right_l2": float(np.linalg.norm(right)),
    }


def run_15tr07(protocol_path: str | Path, archive_root: str | Path, output_dir: str | Path, stage: str = "diagnostic") -> dict[str, object]:
    protocol = _read_protocol(Path(protocol_path))
    _validate_protocol(protocol, stage)
    Image, ImageDraw, PngInfo = HISTORIC_DRIVER._require_pillow()
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
    selected_ids, loaded_roles, loaded_role_counts = _select_metadata_scope(shard, protocol)
    support = CTL.NativeFrameH0Support.from_mapping(protocol["support"])
    if support.point_count != 729:
        raise ValueError("15TR-07 nominal PSF support must contain exactly 729 points")

    # This is the complete pre-response guard.  The source conversion and all
    # operator calls occur only after this fixed metadata-led budget passes.
    frequency_count = int(np.asarray(shard.frequencies_hz).size)
    preflight = _metadata_preflight(shard, selected_ids, protocol, support)
    budget = SOURCE_AF.matched_unit_point_psf_budget(
        record_count=len(selected_ids),
        frequency_count=frequency_count,
        support_point_count=support.point_count,
        max_total_kernel_evaluations=int(protocol["budget"]["max_total_kernel_evaluations"]),
        max_branch_kernel_evaluations=int(protocol["budget"]["max_branch_kernel_evaluations"]),
    )
    expected_budget = {
        "full_point_count": support.point_count,
        "max_total_kernel_evaluations": int(protocol["budget"]["max_total_kernel_evaluations"]),
        "max_branch_kernel_evaluations": int(protocol["budget"]["max_branch_kernel_evaluations"]),
    }
    for key, expected in expected_budget.items():
        if key == "full_point_count":
            continue
        if budget[key] != expected:
            raise ValueError(f"budget.{key} disagrees with the protocol cap")
    preflight["budget"] = budget
    preflight["budget_guard_passed_before_response_conversion_or_bp"] = True

    observations = tuple(shard.observations(selected_ids))
    _validate_native_contract(shard, observations, protocol)
    scope = SOURCE_AF.SourceAFScope(
        pass_id=1,
        polarization="hh",
        sector_ids=EXPECTED_SECTORS,
        role="train",
        expected_count=None,
        name="15tr07_nominal_train_sectors_268_270",
    )
    source_records = SOURCE_AF.build_source_af(observations, scope=scope)
    if len(source_records) != len(selected_ids):
        raise ValueError("source-AF conversion changed the selected record count")
    points = support.grid_points()
    started = time.perf_counter()
    raw_unnormalized = SOURCE_AF.direct_backproject(
        source_records,
        points,
        representation=SOURCE_AF.RAW_REPRESENTATION,
        normalization="none",
        point_chunk_size=int(protocol["support"]["point_chunk_size"]),
        max_kernel_evaluations=int(protocol["budget"]["max_branch_kernel_evaluations"]),
    )
    source_unnormalized = SOURCE_AF.direct_backproject(
        source_records,
        points,
        representation=SOURCE_AF.SOURCE_REPRESENTATION,
        normalization="none",
        point_chunk_size=int(protocol["support"]["point_chunk_size"]),
        max_kernel_evaluations=int(protocol["budget"]["max_branch_kernel_evaluations"]),
    )
    raw_psf_unnormalized = SOURCE_AF.matched_unit_point_psf(
        source_records,
        points,
        NOMINAL_POINT_C_M,
        representation=SOURCE_AF.RAW_REPRESENTATION,
        normalization="none",
        point_chunk_size=int(protocol["support"]["point_chunk_size"]),
        max_kernel_evaluations=int(protocol["budget"]["max_branch_kernel_evaluations"]),
    )
    source_psf_unnormalized = SOURCE_AF.matched_unit_point_psf(
        source_records,
        points,
        NOMINAL_POINT_C_M,
        representation=SOURCE_AF.SOURCE_REPRESENTATION,
        normalization="none",
        point_chunk_size=int(protocol["support"]["point_chunk_size"]),
        max_kernel_evaluations=int(protocol["budget"]["max_branch_kernel_evaluations"]),
    )
    elapsed = time.perf_counter() - started

    total_samples = int(budget["total_native_frequency_samples"])
    raw_mean = np.asarray(raw_unnormalized, dtype=np.complex128) / float(total_samples)
    source_mean = np.asarray(source_unnormalized, dtype=np.complex128) / float(total_samples)
    raw_psf_mean = np.asarray(raw_psf_unnormalized, dtype=np.complex128) / float(total_samples)
    source_psf_mean = np.asarray(source_psf_unnormalized, dtype=np.complex128) / float(total_samples)
    raw_db, source_db, measured_reference = _db_pair(raw_unnormalized, source_unnormalized)
    raw_psf_db, source_psf_db, psf_reference = _db_pair(raw_psf_unnormalized, source_psf_unnormalized)
    pulse_count = len(source_records)
    shape_text = f"[{pulse_count}, {frequency_count}]"
    common_headers = [
        f"15TR-07 nominal PSF diagnostic | pass 1 | HH | train sectors 268,269,270 | {pulse_count} pulses | {total_samples:,} native samples | response shape {shape_text}",
        "Point C=(-5.12,22.98,-0.05) m | workbook-to-native identity map | z=-0.05 plane | 270-degree heading: unverified working hypotheses",
        "selected native trajectories | exact stored frequency vector | unit weights | per-pulse native r0 | no interpolation/regridding/windowing",
        "bounded 2.6 m diagnostic support | 27x27 at 0.10 m | not a survey-to-phase-center displacement bound | pitch is not coherent quadrature",
        "conditional diagnostic only | no brightness selection/search | no registration/localization/accuracy/focus/geometry/model-fit evidence",
    ]
    measured_headers = common_headers + [
        "measured raw/source pair | common A_ref=max(abs(raw),abs(source)) | fixed display clip [-40,0] dB",
    ]
    psf_headers = common_headers + [
        "matched synthetic unit-point raw/source PSF pair | own common A_ref=max(abs(raw_psf),abs(source_psf)) | fixed display clip [-40,0] dB",
    ]

    output_dir.mkdir(parents=True, exist_ok=False)
    bp_display = HISTORIC_DRIVER._write_pair_png(
        output_dir / str(protocol["outputs"]["bp_plot"]),
        raw_db,
        source_db,
        support,
        Image=Image,
        ImageDraw=ImageDraw,
        PngInfo=PngInfo,
        common_reference=measured_reference,
        header_lines=measured_headers,
        panel_labels=("measured raw/unapplied AF", "measured source-AF"),
        x_tick_values=SUPPORT_X_TICKS,
        y_tick_values=SUPPORT_Y_TICKS,
    )
    psf_display = HISTORIC_DRIVER._write_pair_png(
        output_dir / str(protocol["outputs"]["psf_plot"]),
        raw_psf_db,
        source_psf_db,
        support,
        Image=Image,
        ImageDraw=ImageDraw,
        PngInfo=PngInfo,
        common_reference=psf_reference,
        header_lines=psf_headers,
        panel_labels=("matched raw unit-PSF", "matched source-AF unit-PSF"),
        x_tick_values=SUPPORT_X_TICKS,
        y_tick_values=SUPPORT_Y_TICKS,
    )
    outputs = protocol["outputs"]
    np.save(output_dir / str(outputs["raw_bp_unnormalized"]), np.asarray(raw_unnormalized, dtype=np.complex128))
    np.save(output_dir / str(outputs["source_bp_unnormalized"]), np.asarray(source_unnormalized, dtype=np.complex128))
    np.save(output_dir / str(outputs["raw_bp_mean_native_sample"]), raw_mean)
    np.save(output_dir / str(outputs["source_bp_mean_native_sample"]), source_mean)
    np.save(output_dir / str(outputs["raw_psf_unnormalized"]), np.asarray(raw_psf_unnormalized, dtype=np.complex128))
    np.save(output_dir / str(outputs["source_psf_unnormalized"]), np.asarray(source_psf_unnormalized, dtype=np.complex128))
    np.save(output_dir / str(outputs["raw_psf_mean_native_sample"]), raw_psf_mean)
    np.save(output_dir / str(outputs["source_psf_mean_native_sample"]), source_psf_mean)
    np.save(output_dir / str(outputs["x_coordinates"]), np.linspace(-6.42, -3.82, 27, dtype=np.float64))
    np.save(output_dir / str(outputs["y_coordinates"]), np.linspace(21.68, 24.28, 27, dtype=np.float64))

    disclosure = {
        "loaded_response_roles": loaded_roles,
        "loaded_response_role_counts": loaded_role_counts,
        "loaded_response_count": int(shard.view_count),
        "used_response_roles": ["train"],
        "selected_response_count": pulse_count,
        "selected_response_shape": [pulse_count, frequency_count],
        "selected_sector_ids": list(EXPECTED_SECTORS),
        "validation_sector_ids": list(EXPECTED_VALIDATION_SECTORS),
        "validation_role_unused": True,
        "validation_response_payload_materialized": bool(loaded_role_counts.get("validation", 0) > 0),
        "validation_used_for_bp": False,
        "validation_used_for_scale_energy_peak_or_refinement": False,
        "test_payload_opened": False,
        "test_response_payload_materialized": False,
        "test_sealing_contract": "archive_metadata_and_row_exclusion_not_independent_historical_exposure_proof",
        "selected_train_rows_only_for_downstream_operations": True,
        "frequency_policy": "native_stored_exact",
        "frequency_vector_layout": "one_shared_exact_vector_per_shard",
        "shared_native_frequency_count": frequency_count,
        "nominal_point_identity_mapping_unverified": True,
        "heading_convention_unverified": True,
        "nominal_point_is_not_truth_or_phase_center": True,
        "support_is_not_a_phase_center_displacement_bound": True,
        "pitch_is_not_coherent_quadrature_claim": True,
        "measured_raw_source_pair_common_scale": True,
        "psf_raw_source_pair_common_scale": True,
        "psf_scale_independent_of_measured_pair": True,
        "raw_source_change_diagnostics_only": True,
        "matched_psf_operator_sanity_only": True,
        "target_cube_or_localization": False,
        "peak_offset_or_accuracy_output": False,
        "registration_or_calibration_closure": False,
        "model_fit": False,
        "conditional_result": True,
        "archive_units_disclosure": protocol["source"]["archive_units_disclosure"],
    }
    _write_json(output_dir / "protocol_echo.json", protocol)
    _write_json(output_dir / "preflight.json", {**preflight, "disclosure": disclosure})
    _write_json(
        output_dir / "resource.json",
        {
            "schema": "rift_gotcha_step2_15tr07_nominal_psf_resource_v1",
            "budget": budget,
            "selected_pulse_count": pulse_count,
            "point_count": support.point_count,
            "point_chunk_size": int(protocol["support"]["point_chunk_size"]),
            "elapsed_seconds": elapsed,
            "kernel_ledger": {
                "measured_raw_bp": "one full BP",
                "measured_source_bp": "one full BP",
                "raw_unit_psf": "one unit-point forward plus one full BP",
                "source_unit_psf": "one unit-point forward plus one full BP",
                "extra_probe_or_reference_passes": 0,
            },
            "guard_passed_before_bp": True,
        },
    )
    _write_json(
        output_dir / "comparison_report.json",
        {
            "schema": "rift_gotcha_step2_15tr07_nominal_psf_report_v1",
            "status": "PASS",
            "diagnostic_statement": "conditional raw/source change diagnostic with matched synthetic unit-point operator sanity only—not a target cube, scattering truth, phase-center claim, localization, accuracy/focus/geometry score, registration, calibration closure, or model-fit result",
            "result_status": preflight["status"],
            "selected_pulse_count": pulse_count,
            "selected_frequency_samples": total_samples,
            "selected_response_shape": [pulse_count, frequency_count],
            "nominal_point_c": {
                "xyz_m": NOMINAL_POINT_C_M.tolist(),
                "identity_mapping_status": "unverified working hypothesis",
                "heading_convention_status": "unverified working hypothesis; sector index near 270 is not verified alignment",
                "truth_or_phase_center_claim": False,
            },
            "support": support.as_dict(),
            "budget": budget,
            "raw_source_arrays_emitted": True,
            "matched_psf_operator_sanity": {
                "raw_source_agreement": _pair_difference(raw_psf_mean, source_psf_mean),
                "measured_response_used": False,
                "no_peak_or_offset_reported": True,
            },
            "display": {
                "measured_pair": bp_display,
                "psf_pair": psf_display,
                "measured_pair_common_scale": True,
                "psf_pair_common_scale": True,
                "psf_scale_independent_of_measured_pair": True,
            },
            "disclosure": disclosure,
        },
    )
    files = [
        "protocol_echo.json", "preflight.json", "resource.json", "comparison_report.json", "status.json",
        str(outputs["raw_bp_unnormalized"]), str(outputs["source_bp_unnormalized"]),
        str(outputs["raw_bp_mean_native_sample"]), str(outputs["source_bp_mean_native_sample"]),
        str(outputs["raw_psf_unnormalized"]), str(outputs["source_psf_unnormalized"]),
        str(outputs["raw_psf_mean_native_sample"]), str(outputs["source_psf_mean_native_sample"]),
        str(outputs["x_coordinates"]), str(outputs["y_coordinates"]),
        str(outputs["bp_plot"]), str(outputs["psf_plot"]),
    ]
    status = {
        "schema": DRIVER_SCHEMA,
        "status": "PASS",
        "stage": "diagnostic",
        "output_dir": str(output_dir),
        "files": files,
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
    run_15tr07(args.protocol, args.archive_root, args.output_dir, args.stage)
    print(json.dumps({"status": "PASS", "output_dir": str(Path(args.output_dir))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
