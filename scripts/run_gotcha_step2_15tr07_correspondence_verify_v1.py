"""Verify conditional held-out source-AF local point-response consistency.

This post-processor freezes one deterministic TRAIN feature from the accepted
15TR-07 source-AF BP artifact, then evaluates only a fixed 9x9 patch on the
adjacent validation sector 271.  The result is a conditional shape-consistency
statement, never a registered coordinate error, a phase-center offset, or a
physical 15TR-07 identity claim.
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


# Reuse the established acquisition/operator implementation and JSON helpers.
BASE = _load_module(
    "gotcha_source_af_compare_driver_for_correspondence_verify_v1",
    PROJECT_ROOT / "scripts" / "run_gotcha_step2_source_af_compare_v1.py",
)
ACQ = BASE.ACQ
CTL = BASE.CTL
SOURCE_AF = BASE.SOURCE_AF


PROTOCOL_SCHEMA = "rift_gotcha_step2_15tr07_correspondence_verify_p1_hh_protocol_v1"
DRIVER_SCHEMA = "rift_gotcha_step2_15tr07_correspondence_verify_driver_v1"
NOMINAL_PROTOCOL_SCHEMA = "rift_gotcha_step2_15tr07_nominal_psf_p1_hh_train_protocol_v1"
TRAIN_SECTORS = (268, 269, 270)
VALIDATION_SECTORS = (271,)
FREQUENCY_COUNT = 424
GRID_SHAPE = (27, 27)
SUPPORT_POINT_COUNT = 729
PATCH_RADIUS_CELLS = 4
PATCH_SIDE = 2 * PATCH_RADIUS_CELLS + 1
PATCH_POINT_COUNT = PATCH_SIDE * PATCH_SIDE
MAX_BRANCH_KERNEL_EVALUATIONS = 250_000_000
MAX_COMBINED_KERNEL_EVALUATIONS = 500_000_000
NOMINAL_POINT_C_M = np.asarray((-5.12, 22.98, -0.05), dtype=np.float64)
SOURCE_FORM = "r0_src=float64(r0_raw)+float64(r_correct_raw); fp_src=complex128(fp_raw)*exp(+i*float64(ph_correct_raw))"
DECISION_SUPPORTED = "supported_conditional_holdout_local_psf_shape_consistency"
DECISION_INCONSISTENT = "inconsistent_with_fixed_single_point_source_af_response_model"
DECISION_INCONCLUSIVE = "inconclusive_nonresolving_validation_patch"
REASON_NO_TRAIN_FEATURE = "no_train_feature"
REASON_BOUNDARY = "boundary_or_out_of_support"
REASON_AMBIGUOUS = "ambiguous_train_feature"


def _json_ready(value):
    return BASE._json_ready(value)


def _write_json(path: Path, value) -> None:
    BASE._write_json(path, value)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _assert_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise ValueError(f"protocol {label} must be {expected!r}; got {actual!r}")


def _finite_complex(value: np.ndarray, label: str) -> None:
    if not np.iscomplexobj(value) or not np.isfinite(value.real).all() or not np.isfinite(value.imag).all():
        raise ValueError(f"{label} must be finite complex data")


def _required_file(directory: Path, name: str) -> Path:
    path = directory / name
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"required artifact is missing or not a regular file: {path}")
    return path


def _validate_protocol(protocol: Mapping[str, object], stage: str) -> None:
    _assert_equal(stage, "verification", "CLI stage")
    _assert_equal(protocol.get("schema"), PROTOCOL_SCHEMA, "schema")
    _assert_equal(protocol.get("stage"), "verification", "stage")

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

    prior = protocol.get("prior_artifact")
    if not isinstance(prior, Mapping):
        raise ValueError("protocol prior_artifact must be an object")
    for key, expected in {
        "required_status": "PASS",
        "required_stage": "diagnostic",
        "required_driver_schema": "rift_gotcha_step2_15tr07_nominal_psf_driver_v1",
        "required_selection_train_sector_ids": list(TRAIN_SECTORS),
        "required_validation_sector_ids": list(VALIDATION_SECTORS),
        "required_source_bp_array": "bp_source_mean_native_sample.npy",
        "required_source_psf_array": "psf_source_mean_native_sample.npy",
        "png_provenance_required": True,
    }.items():
        _assert_equal(prior.get(key), expected, f"prior_artifact.{key}")

    selection = protocol.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("protocol selection must be an object")
    for key, expected in {
        "train_role": "train",
        "train_sector_ids": list(TRAIN_SECTORS),
        "validation_role": "validation",
        "validation_sector_ids": list(VALIDATION_SECTORS),
        "candidate_source": "prior source-AF mean BP only",
        "candidate_field": "conditional_train_grid_feature_xyz_m",
        "tie_break": "lowest flat (ix,iy) index",
        "interior_margin_cells": PATCH_RADIUS_CELLS,
        "unique_peak_gap_db": 3.0,
        "validation_used_for_selection": False,
        "raw_bp_used_for_selection": False,
    }.items():
        _assert_equal(selection.get(key), expected, f"selection.{key}")

    nominal = protocol.get("nominal_point_c")
    if not isinstance(nominal, Mapping):
        raise ValueError("protocol nominal_point_c must be an object")
    _assert_equal(nominal.get("xyz_m"), [-5.12, 22.98, -0.05], "nominal_point_c.xyz_m")
    _assert_equal(nominal.get("heading_degrees"), 270.0, "nominal_point_c.heading_degrees")
    _assert_equal(nominal.get("truth_or_phase_center_claim"), False, "nominal_point_c.truth_or_phase_center_claim")
    if nominal.get("identity_mapping_status") != "unverified working hypothesis":
        raise ValueError("nominal Point C identity mapping must remain unverified")

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
        "max_kernel_evaluations": MAX_BRANCH_KERNEL_EVALUATIONS,
        "runtime_profile_origin_only": False,
        "target_roi_or_localization_claim": False,
        "phase_hypothesis": CTL.PHASE_HYPOTHESIS_NAME,
        "autofocus_status": "raw_unapplied_channel_owned",
        "point_chunk_size": 4096,
    }.items():
        _assert_equal(support.get(key), expected, f"support.{key}")
    _assert_equal(support.get("units"), {"x": "m", "y": "m", "z": "m", "r0": "m", "frequency": "Hz"}, "support.units")
    _assert_equal(support.get("support_interpretation"), "bounded 2.6 m neighborhood diagnostic support, not a survey-to-phase-center displacement bound", "support.support_interpretation")
    _assert_equal(support.get("pitch_interpretation"), "0.10 m pitch is a display/sampling choice, not a coherent field-fitting quadrature claim", "support.pitch_interpretation")

    patch = protocol.get("patch")
    if not isinstance(patch, Mapping):
        raise ValueError("protocol patch must be an object")
    for key, expected in {
        "radius_cells": PATCH_RADIUS_CELLS,
        "shape": [9, 9, 1],
        "point_count": PATCH_POINT_COUNT,
        "center": "frozen train candidate grid point",
        "no_recentering": True,
        "no_transform": True,
    }.items():
        _assert_equal(patch.get(key), expected, f"patch.{key}")

    native = protocol.get("native_contract")
    if not isinstance(native, Mapping):
        raise ValueError("protocol native_contract must be an object")
    for key, expected in {
        "frequency_policy": "native_stored_exact",
        "frequency_vector_layout": "one_shared_exact_vector_per_shard",
        "expected_native_frequency_count_per_pulse": FREQUENCY_COUNT,
        "r0_field": "r0",
        "geometry": "paired_monostatic_tx_equals_rx_same_observation",
        "phase_forward": CTL.PHASE_HYPOTHESIS_FORWARD,
        "phase_adjoint": CTL.PHASE_HYPOTHESIS_ADJOINT,
        "source_form": SOURCE_FORM,
        "interpolation": False,
        "regridding": False,
        "windowing": False,
        "unit_weights": True,
        "per_pulse_r0_exact_stored": True,
    }.items():
        _assert_equal(native.get(key), expected, f"native_contract.{key}")

    search_controls = protocol.get("search_controls")
    if not isinstance(search_controls, Mapping):
        raise ValueError("protocol search_controls must be an object")
    for key in (
        "response_condition_search", "continuous_candidate_search", "validation_selection", "phase_sign_search",
        "gain_search", "aperture_search", "support_search", "refinement", "registration", "fit",
        "validation_phase_or_gain_optimization",
    ):
        _assert_equal(search_controls.get(key), False, f"search_controls.{key}")
    _assert_equal(search_controls.get("train_only_discrete_grid_argmax_selection"), True, "search_controls.train_only_discrete_grid_argmax_selection")

    metric = protocol.get("decision_metric")
    if not isinstance(metric, Mapping):
        raise ValueError("protocol decision_metric must be an object")
    for key, expected in {
        "name": "fixed_validation_train_patch_shape_consistency",
        "formula": "rho=abs(vdot(psf_patch,validation_bp_patch))/(norm(psf_patch)*norm(validation_bp_patch))",
        "zero_norm_result": DECISION_INCONCLUSIVE,
        "validation_train_patch_norm_ratio_threshold": 0.25,
        "rho_supported_threshold": 0.8,
        "rho_inconsistent_threshold": 0.4,
        "peak_distance_supported_max_cells": 1,
        "peak_distance_inconsistent_min_exclusive_cells": 1,
        "no_optimizing_phase_or_gain": True,
    }.items():
        _assert_equal(metric.get(key), expected, f"decision_metric.{key}")

    claims = protocol.get("claims")
    if not isinstance(claims, Mapping):
        raise ValueError("protocol claims must be an object")
    for key, expected in {
        "coordinate_alignment": "not_identifiable_without_independent_landmarks",
        "point_c_scattering_center": "not_identifiable_without_independent_point_c_semantics_or_calibration",
        "physical_15tr07_return_association": "inconclusive_without_external_identity_evidence",
        "physical_identity_status_independent_of_conditional_response": True,
    }.items():
        _assert_equal(claims.get(key), expected, f"claims.{key}")

    budget = protocol.get("budget")
    if not isinstance(budget, Mapping):
        raise ValueError("protocol budget must be an object")
    for key, expected in {
        "K": PATCH_POINT_COUNT,
        "N_val_definition": "validation_record_count multiplied by exact native frequency samples per record",
        "new_formula": "(2*K+1)*N_val",
        "max_branch_kernel_evaluations": MAX_BRANCH_KERNEL_EVALUATIONS,
        "max_combined_kernel_evaluations": MAX_COMBINED_KERNEL_EVALUATIONS,
        "extra_probe_kernel_evaluations": 0,
        "extra_fit_kernel_evaluations": 0,
        "guard_before_validation_observations_source_conversion_or_bp": True,
        "cumulative_cap_is_scientific_provenance_guard_not_new_job_resource_consumption": True,
    }.items():
        _assert_equal(budget.get(key), expected, f"budget.{key}")
    _assert_equal(
        budget.get("new_line_items"),
        {
            "validation_source_bp": "K*N_val",
            "validation_unit_point_forward": "N_val",
            "validation_source_psf_bp": "K*N_val",
        },
        "budget.new_line_items",
    )

    outputs = protocol.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("protocol outputs must be an object")
    expected_output_names = {
        "protocol_echo": "protocol_echo.json",
        "preflight": "preflight.json",
        "resource": "resource.json",
        "train_feature": "train_feature.json",
        "validation_confirmation": "validation_confirmation.json",
        "correspondence_report": "correspondence_report.json",
        "status": "status.json",
        "validation_bp_patch_mean_native_sample": "validation_bp_patch_mean_native_sample.npy",
        "validation_psf_patch_mean_native_sample": "validation_psf_patch_mean_native_sample.npy",
        "patch_coordinates": "validation_patch_coordinates.npy",
    }
    _assert_equal(outputs, expected_output_names, "outputs")

    disclosure = protocol.get("disclosure")
    if not isinstance(disclosure, Mapping):
        raise ValueError("protocol disclosure must be an object")
    for key, expected in {
        "test_payload_opened": False,
        "test_response_payload_materialized": False,
        "validation_response_payload_materialized_by_loader": True,
        "validation_used_for_selection": False,
        "candidate_frozen_from_train": True,
        "validation_used_for_model_fitting": False,
        "supplied_per_row_af_corrections_may_use_validation_responses": True,
        "held_out_from_candidate_selection_and_model_fitting_not_strict_unseen_response_calibration": True,
        "validation_sector_is_adjacent_aperture_slice": True,
        "raw_source_imagery_context_only": True,
        "no_registered_transform_or_offset": True,
        "no_fitted_gain_or_phase": True,
        "source_af_conditionality": True,
        "current_train_metadata_semantic_match_required": True,
        "current_train_metadata_match_not_archival_identity_proof": True,
        "non_test_response_matrix_materialized_by_loader_before_new_work_guard": True,
        "train_observation_objects_materialized_for_downstream": False,
    }.items():
        _assert_equal(disclosure.get(key), expected, f"disclosure.{key}")


def _support(protocol: Mapping[str, object]):
    support = CTL.NativeFrameH0Support.from_mapping(protocol["support"])
    if support.point_count != SUPPORT_POINT_COUNT:
        raise ValueError("verification support must contain exactly 729 points")
    return support


def _validate_prior_artifact(prior_output_dir: Path, protocol: Mapping[str, object]) -> dict[str, object]:
    status = _read_json(_required_file(prior_output_dir, "status.json"))
    prior_protocol = _read_json(_required_file(prior_output_dir, "protocol_echo.json"))
    prior_preflight = _read_json(_required_file(prior_output_dir, "preflight.json"))
    prior_resource = _read_json(_required_file(prior_output_dir, "resource.json"))
    prior_report = _read_json(_required_file(prior_output_dir, "comparison_report.json"))
    if status.get("status") != "PASS" or status.get("stage") != "diagnostic":
        raise ValueError("prior nominal artifact must have PASS diagnostic status")
    if status.get("schema") != "rift_gotcha_step2_15tr07_nominal_psf_driver_v1":
        raise ValueError("prior nominal artifact has the wrong driver schema")
    if prior_protocol.get("schema") != NOMINAL_PROTOCOL_SCHEMA or prior_protocol.get("stage") != "diagnostic":
        raise ValueError("prior protocol is not the accepted nominal 15TR-07 protocol")
    if prior_protocol.get("source") != protocol.get("source"):
        raise ValueError("prior source contract does not match the verification source")
    prior_selection = prior_protocol.get("selection")
    if not isinstance(prior_selection, Mapping):
        raise ValueError("prior selection is missing")
    for key, expected in {
        "role": "train",
        "sector_ids": list(TRAIN_SECTORS),
        "validation_sector_ids": list(VALIDATION_SECTORS),
        "expected_native_frequency_count_per_pulse": FREQUENCY_COUNT,
        "loaded_roles": ["train", "validation"],
        "used_roles": ["train"],
        "test_payload_opened": False,
    }.items():
        _assert_equal(prior_selection.get(key), expected, f"prior selection.{key}")
    prior_native = prior_protocol.get("native_contract")
    if not isinstance(prior_native, Mapping):
        raise ValueError("prior native contract is missing")
    for key, expected in {
        "frequency_policy": "native_stored_exact",
        "frequency_vector_layout": "one_shared_exact_vector_per_shard",
        "source_form": SOURCE_FORM,
        "interpolation": False,
        "regridding": False,
        "windowing": False,
        "unit_weights": True,
        "per_pulse_r0_exact_stored": True,
    }.items():
        _assert_equal(prior_native.get(key), expected, f"prior native_contract.{key}")
    if prior_protocol.get("support") != protocol.get("support"):
        raise ValueError("prior support declaration does not match the fixed verification support")
    prior_disclosure = prior_report.get("disclosure")
    if not isinstance(prior_disclosure, Mapping):
        raise ValueError("prior report disclosure is missing")
    for key, expected in {
        "validation_used_for_bp": False,
        "validation_used_for_scale_energy_peak_or_refinement": False,
        "test_payload_opened": False,
        "test_response_payload_materialized": False,
        "selected_train_rows_only_for_downstream_operations": True,
        "peak_offset_or_accuracy_output": False,
        "registration_or_calibration_closure": False,
        "matched_psf_operator_sanity_only": True,
    }.items():
        _assert_equal(prior_disclosure.get(key), expected, f"prior disclosure.{key}")
    if prior_report.get("matched_psf_operator_sanity", {}).get("measured_response_used") is not False:
        raise ValueError("prior matched PSF is not admitted as a measured-response-free operator sanity check")
    if prior_report.get("matched_psf_operator_sanity", {}).get("no_peak_or_offset_reported") is not True:
        raise ValueError("prior matched PSF artifact does not preserve the no-offset disclosure")
    if prior_report.get("status") != "PASS":
        raise ValueError("prior comparison report must have PASS status")
    if "raw_source_change_diagnostics" in prior_report:
        raise ValueError("prior nominal artifact contains an unapproved raw/source peak diagnostic")

    prior_support = _support(prior_protocol)
    x = np.asarray(np.load(_required_file(prior_output_dir, "x.npy"), allow_pickle=False), dtype=np.float64)
    y = np.asarray(np.load(_required_file(prior_output_dir, "y.npy"), allow_pickle=False), dtype=np.float64)
    if x.shape != (27,) or y.shape != (27,) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("prior support coordinate arrays must be finite 27-vectors")
    if not np.allclose(x, np.linspace(-6.42, -3.82, 27, dtype=np.float64), rtol=0.0, atol=1.0e-12):
        raise ValueError("prior x coordinates do not match the fixed native support")
    if not np.allclose(y, np.linspace(21.68, 24.28, 27, dtype=np.float64), rtol=0.0, atol=1.0e-12):
        raise ValueError("prior y coordinates do not match the fixed native support")
    source_bp = np.asarray(np.load(_required_file(prior_output_dir, "bp_source_mean_native_sample.npy"), allow_pickle=False))
    raw_psf = np.asarray(np.load(_required_file(prior_output_dir, "psf_raw_mean_native_sample.npy"), allow_pickle=False))
    source_psf = np.asarray(np.load(_required_file(prior_output_dir, "psf_source_mean_native_sample.npy"), allow_pickle=False))
    if source_bp.shape != (SUPPORT_POINT_COUNT,) or raw_psf.shape != (SUPPORT_POINT_COUNT,) or source_psf.shape != (SUPPORT_POINT_COUNT,):
        raise ValueError("prior source BP/PSF arrays must contain exactly 729 points")
    _finite_complex(source_bp, "prior source BP")
    _finite_complex(raw_psf, "prior raw PSF")
    _finite_complex(source_psf, "prior source PSF")
    if not np.allclose(raw_psf, source_psf, rtol=5.0e-12, atol=5.0e-12):
        raise ValueError("prior raw/source matched PSFs do not satisfy the admitted numerical sanity check")
    prior_psf_agreement = prior_report.get("matched_psf_operator_sanity", {}).get("raw_source_agreement")
    if not isinstance(prior_psf_agreement, Mapping):
        raise ValueError("prior matched PSF numerical agreement is missing")
    max_psf_difference = float(prior_psf_agreement.get("max_abs_difference", float("inf")))
    if not np.isfinite(max_psf_difference) or max_psf_difference > 5.0e-12:
        raise ValueError("prior matched PSF report exceeds the admitted numerical sanity tolerance")
    prior_budget = prior_report.get("budget")
    prior_preflight_budget = prior_preflight.get("budget")
    prior_resource_budget = prior_resource.get("budget")
    if not isinstance(prior_budget, Mapping) or not isinstance(prior_preflight_budget, Mapping) or not isinstance(prior_resource_budget, Mapping):
        raise ValueError("prior historical budget artifacts are incomplete")
    prior_total = int(prior_budget.get("total_kernel_evaluations", -1))
    if prior_total <= 0 or prior_total != int(prior_preflight_budget.get("total_kernel_evaluations", -2)) or prior_total != int(prior_resource_budget.get("total_kernel_evaluations", -3)):
        raise ValueError("prior historical budget is not semantically consistent")
    for key in ("measured_raw_bp_kernel_evaluations", "measured_source_bp_kernel_evaluations", "raw_unit_point_psf_bp_kernel_evaluations", "source_unit_point_psf_bp_kernel_evaluations"):
        if int(prior_budget.get(key, MAX_BRANCH_KERNEL_EVALUATIONS + 1)) > MAX_BRANCH_KERNEL_EVALUATIONS:
            raise ValueError("prior historical direct-BP branch exceeds the declared cap")
    prior_output_names = prior_protocol.get("outputs")
    if not isinstance(prior_output_names, Mapping):
        raise ValueError("prior output declaration is missing")
    for key in ("bp_plot", "psf_plot"):
        name = prior_output_names.get(key)
        if not isinstance(name, str) or not name:
            raise ValueError(f"prior output {key} is missing")
        _required_file(prior_output_dir, name)
    return {
        "status": status,
        "protocol": prior_protocol,
        "preflight": prior_preflight,
        "resource": prior_resource,
        "report": prior_report,
        "source_bp": source_bp,
        "source_psf": source_psf,
        "x": x,
        "y": y,
        "support": prior_support,
        "historical_budget": dict(prior_budget),
        "historical_total_kernel_evaluations": prior_total,
        "prior_pngs": {
            "bp": str(prior_output_dir / str(prior_output_names["bp_plot"])),
            "psf": str(prior_output_dir / str(prior_output_names["psf_plot"])),
        },
    }


def _validate_current_train_metadata(shard, prior: Mapping[str, object]) -> tuple[tuple[object, ...], dict[str, object]]:
    """Bind the current archive's TRAIN metadata to the prior nominal provenance.

    This is a semantic metadata check only.  It is deliberately not an archival
    identity proof and it never materializes TRAIN observation objects.
    """

    identities = tuple(shard.observation_ids)
    roles = tuple(str(value).lower() for value in np.asarray(shard.role).tolist())
    if len(identities) != len(roles):
        raise ValueError("current train metadata identity and role headers have different lengths")
    loaded_roles = sorted(set(roles))
    if loaded_roles != ["train", "validation"]:
        raise ValueError(f"current archive roles must be train+validation; got {loaded_roles}")
    if any(role == "test" for role in roles):
        raise ValueError("current archive test payload rows must remain sealed")
    selected = tuple(
        sorted(
            (
                identity
                for identity, role in zip(identities, roles)
                if role == "train" and int(identity.sector_id) in TRAIN_SECTORS
            ),
            key=BASE._id_key,
        )
    )
    if not selected:
        raise ValueError("current archive has no canonical P1/HH TRAIN 268--270 metadata")
    if any(
        int(identity.pass_id) != 1
        or str(identity.polarization).lower() != "hh"
        or int(identity.sector_id) not in TRAIN_SECTORS
        for identity in selected
    ):
        raise ValueError("current train metadata escaped the immutable P1/HH 268--270 scope")
    if tuple(sorted(selected, key=BASE._id_key)) != selected or len(set(selected)) != len(selected):
        raise ValueError("current train metadata IDs are not unique canonical sorted IDs")

    prior_preflight = prior.get("preflight")
    if not isinstance(prior_preflight, Mapping):
        raise ValueError("prior nominal preflight is missing for current train metadata binding")
    prior_ids = prior_preflight.get("selected_ids")
    if not isinstance(prior_ids, list):
        raise ValueError("prior nominal preflight selected train IDs are missing")
    current_ids = [identity.as_dict() for identity in selected]
    if current_ids != prior_ids:
        raise ValueError("current train metadata IDs/count/order do not match prior nominal preflight")
    prior_count = int(prior_preflight.get("selected_record_count", -1))
    if prior_count != len(selected):
        raise ValueError("current train metadata count does not match prior nominal preflight")

    frequencies = np.asarray(shard.frequencies_hz, dtype=np.float64)
    if frequencies.shape != (FREQUENCY_COUNT,) or not np.isfinite(frequencies).all():
        raise ValueError("current archive lacks the exact 424-sample native frequency vector")
    prior_frequency_count = int(prior_preflight.get("frequency_samples_per_record", -1))
    if prior_frequency_count != FREQUENCY_COUNT:
        raise ValueError("prior nominal preflight does not declare the exact 424-sample frequency vector")
    current_total_samples = int(len(selected) * frequencies.size)
    prior_total_samples = int(prior_preflight.get("total_native_frequency_samples", -1))
    if prior_total_samples != int(prior_count * prior_frequency_count) or current_total_samples != prior_total_samples:
        raise ValueError("current train native sample count does not match prior nominal preflight")
    if prior_preflight.get("frequency_vector_policy") != "native_stored_exact":
        raise ValueError("prior nominal preflight does not declare native exact frequency provenance")
    if prior_preflight.get("frequency_vector_layout") != "one_shared_exact_vector_per_shard":
        raise ValueError("prior nominal preflight does not declare one shared exact frequency vector")
    if shard.phase_reference.frequency_values != "native_stored_exact":
        raise ValueError("current archive frequency provenance is not native exact")
    if shard.phase_reference.reference_range_field != "r0":
        raise ValueError("current archive reference range field is not r0")
    if shard.phase_reference.geometry_contract != "paired_monostatic_tx_equals_rx_same_observation":
        raise ValueError("current archive geometry contract is not paired monostatic")
    if shard.autofocus.mode != ACQ.AUTOFOCUS_RAW or shard.autofocus.applied or shard.autofocus.official_available is not True:
        raise ValueError("current archive does not preserve raw/unapplied autofocus provenance")
    if shard.autofocus.range_field != "af.r_correct" or shard.autofocus.phase_field != "af.ph_correct":
        raise ValueError("current archive raw/unapplied autofocus header fields do not match the native contract")
    if np.asarray(shard.response).shape != (int(shard.view_count), FREQUENCY_COUNT):
        raise ValueError("current archive response matrix does not match the exact native frequency vector")
    if np.asarray(shard.r_correct_raw).shape != (int(shard.view_count),) or np.asarray(shard.ph_correct_raw).shape != (int(shard.view_count),):
        raise ValueError("current archive raw/unapplied autofocus header arrays do not match the response rows")

    current_span = [float(frequencies[0]), float(frequencies[-1])]
    current_center = float(np.mean(frequencies))
    prior_span = prior_preflight.get("actual_frequency_span_hz")
    prior_center = prior_preflight.get("actual_frequency_center_hz")
    if not isinstance(prior_span, list) or len(prior_span) != 2 or prior_center is None:
        raise ValueError("prior nominal preflight lacks declared frequency endpoints/span/center")
    if not np.allclose(current_span, np.asarray(prior_span, dtype=np.float64), rtol=0.0, atol=1.0e-6):
        raise ValueError("current archive frequency endpoints/span do not match prior nominal preflight")
    if not np.isclose(current_center, float(prior_center), rtol=0.0, atol=1.0e-6):
        raise ValueError("current archive frequency center does not match prior nominal preflight")

    role_counts = {role: int(sum(item == role for item in roles)) for role in loaded_roles}
    provenance = {
        "status": "PASS",
        "semantic_match_to_prior_train_provenance": True,
        "not_an_archival_identity_proof": True,
        "selected_ids": current_ids,
        "selected_record_count": len(selected),
        "total_native_frequency_samples": current_total_samples,
        "loaded_roles": loaded_roles,
        "loaded_response_role_counts": role_counts,
        "frequency_vector_policy": "native_stored_exact",
        "frequency_vector_layout": "one_shared_exact_vector_per_shard",
        "frequency_samples_per_record": int(frequencies.size),
        "actual_frequency_span_hz": current_span,
        "actual_frequency_center_hz": current_center,
        "raw_unapplied_autofocus_header": {
            "mode": ACQ.AUTOFOCUS_RAW,
            "applied": False,
            "official_available": True,
            "range_field": "af.r_correct",
            "phase_field": "af.ph_correct",
        },
        "train_observation_objects_materialized": False,
        "test_payload_opened": False,
    }
    return selected, provenance


def _patch_flat_indices(ix: int, iy: int) -> np.ndarray:
    if ix < PATCH_RADIUS_CELLS or ix >= GRID_SHAPE[0] - PATCH_RADIUS_CELLS or iy < PATCH_RADIUS_CELLS or iy >= GRID_SHAPE[1] - PATCH_RADIUS_CELLS:
        raise ValueError("candidate does not have the required fixed patch interior margin")
    return np.asarray(
        [
            (x_index * GRID_SHAPE[1] + y_index)
            for x_index in range(ix - PATCH_RADIUS_CELLS, ix + PATCH_RADIUS_CELLS + 1)
            for y_index in range(iy - PATCH_RADIUS_CELLS, iy + PATCH_RADIUS_CELLS + 1)
        ],
        dtype=np.int64,
    )


def _select_train_feature(prior: Mapping[str, object]) -> dict[str, object]:
    source_bp = np.asarray(prior["source_bp"], dtype=np.complex128)
    magnitude = np.abs(source_bp)
    if not magnitude.size or float(np.max(magnitude)) <= 0.0:
        return {"proceed": False, "reason": REASON_NO_TRAIN_FEATURE, "candidate_frozen_from_train": False}
    flat_index = int(np.argmax(magnitude))
    ix, iy = divmod(flat_index, GRID_SHAPE[1])
    candidate = {
        "conditional_train_grid_feature_xyz_m": [float(prior["x"][ix]), float(prior["y"][iy]), -0.05],
        "grid_index_ix_iy": [ix, iy],
        "flat_index": flat_index,
        "source_bp_mean_abs": float(magnitude[flat_index]),
        "not_a_registered_offset_or_physical_phase_center": True,
        "source_representation_only": True,
    }
    if ix < PATCH_RADIUS_CELLS or ix >= GRID_SHAPE[0] - PATCH_RADIUS_CELLS or iy < PATCH_RADIUS_CELLS or iy >= GRID_SHAPE[1] - PATCH_RADIUS_CELLS:
        return {
            "proceed": False,
            "reason": REASON_BOUNDARY,
            "candidate_frozen_from_train": False,
            **candidate,
        }
    patch_indices = _patch_flat_indices(ix, iy)
    patch_mask = np.ones(GRID_SHAPE[0] * GRID_SHAPE[1], dtype=bool)
    patch_mask[patch_indices] = False
    outside = magnitude[patch_mask]
    outside_max = float(np.max(outside)) if outside.size else 0.0
    if outside_max <= 0.0:
        gap_db = float("inf")
    else:
        gap_db = float(20.0 * np.log10(max(float(magnitude[flat_index]) / outside_max, np.finfo(np.float64).tiny)))
    candidate["strongest_outside_patch_abs"] = outside_max
    candidate["unique_peak_gap_db"] = gap_db
    if gap_db < 3.0:
        return {
            "proceed": False,
            "reason": REASON_AMBIGUOUS,
            "candidate_frozen_from_train": False,
            **candidate,
        }
    train_patch = source_bp[patch_indices]
    train_patch_norm = float(np.linalg.norm(train_patch))
    if train_patch_norm <= 0.0:
        return {
            "proceed": False,
            "reason": REASON_NO_TRAIN_FEATURE,
            "candidate_frozen_from_train": False,
            **candidate,
        }
    return {
        "proceed": True,
        "reason": None,
        "candidate_frozen_from_train": True,
        **candidate,
        "patch_flat_indices": patch_indices.tolist(),
        "train_source_bp_patch_norm": train_patch_norm,
        "interior_margin_cells": PATCH_RADIUS_CELLS,
        "candidate_selection_basis": "deterministic argmax of prior source-AF mean BP magnitude; validation and raw BP not used",
    }


def _select_validation_metadata(shard) -> tuple[tuple[object, ...], list[str], dict[str, int]]:
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
    selected = tuple(
        sorted(
            (
                identity
                for identity, role in zip(identities, roles)
                if role == "validation" and int(identity.sector_id) == VALIDATION_SECTORS[0]
            ),
            key=BASE._id_key,
        )
    )
    if not selected:
        raise ValueError("validation sector 271 contains no rows")
    if any(int(identity.pass_id) != 1 or str(identity.polarization).lower() != "hh" or int(identity.sector_id) != 271 for identity in selected):
        raise ValueError("validation selection escaped the fixed P1/HH sector-271 scope")
    if tuple(sorted(selected, key=BASE._id_key)) != selected or len(set(selected)) != len(selected):
        raise ValueError("validation identities are not unique canonical sorted IDs")
    return selected, loaded_roles, role_counts


def _validate_validation_contract(shard, observations) -> None:
    if shard.phase_reference.frequency_values != "native_stored_exact":
        raise ValueError("validation frequency provenance is not native exact")
    frequencies = np.asarray(shard.frequencies_hz)
    if frequencies.shape != (FREQUENCY_COUNT,):
        raise ValueError("validation requires the exact 424-sample native vector")
    if np.asarray(shard.response).shape != (int(shard.view_count), FREQUENCY_COUNT):
        raise ValueError("native response matrix does not match the exact frequency vector")
    if shard.phase_reference.reference_range_field != "r0" or shard.phase_reference.geometry_contract != "paired_monostatic_tx_equals_rx_same_observation":
        raise ValueError("validation native geometry/r0 contract does not match protocol")
    if shard.autofocus.mode != ACQ.AUTOFOCUS_RAW or shard.autofocus.applied or shard.autofocus.official_available is not True:
        raise ValueError("validation requires raw/unapplied HH correction provenance")
    for observation in observations:
        if int(observation.identity.sector_id) != 271 or str(observation.role).lower() != "validation":
            raise ValueError("materialized validation observation escaped sector 271")
        if observation.autofocus.applied or observation.autofocus.official_available is not True:
            raise ValueError("validation observation lacks raw/unapplied HH correction provenance")
        observation_frequencies = np.asarray(observation.frequencies_hz)
        if observation_frequencies.shape != (FREQUENCY_COUNT,) or not np.array_equal(observation_frequencies, frequencies):
            raise ValueError("validation observation did not preserve the exact shared native frequency vector")
        if np.asarray(observation.response).shape != (FREQUENCY_COUNT,):
            raise ValueError("validation response shape does not match the native frequency vector")
        if observation.r_correct_raw is None or observation.ph_correct_raw is None:
            raise ValueError("validation HH observation lacks supplied raw correction arrays")


def _new_validation_budget(validation_count: int, prior_total: int) -> dict[str, int | bool | str]:
    if validation_count <= 0:
        raise ValueError("validation_count must be positive")
    n_val = int(validation_count * FREQUENCY_COUNT)
    source_bp = int(PATCH_POINT_COUNT * n_val)
    unit_forward = n_val
    source_psf_bp = source_bp
    new_total = source_bp + unit_forward + source_psf_bp
    combined = int(prior_total + new_total)
    if source_bp > MAX_BRANCH_KERNEL_EVALUATIONS or source_psf_bp > MAX_BRANCH_KERNEL_EVALUATIONS:
        raise RuntimeError("combined source-AF validation branch kernel budget exceeded")
    if combined > MAX_COMBINED_KERNEL_EVALUATIONS:
        raise RuntimeError(
            "combined source-AF historical plus validation kernel budget exceeded: "
            f"estimate={combined} max={MAX_COMBINED_KERNEL_EVALUATIONS}"
        )
    return {
        "K": PATCH_POINT_COUNT,
        "N_val": n_val,
        "validation_record_count": int(validation_count),
        "frequency_count_per_validation_record": FREQUENCY_COUNT,
        "validation_source_bp_kernel_evaluations": source_bp,
        "validation_unit_point_forward_kernel_evaluations": unit_forward,
        "validation_source_psf_bp_kernel_evaluations": source_psf_bp,
        "new_validation_kernel_evaluations": new_total,
        "prior_historical_kernel_evaluations": int(prior_total),
        "combined_historical_plus_validation_kernel_evaluations": combined,
        "panel_kernel_evaluations": 0,
        "probe_kernel_evaluations": 0,
        "fit_kernel_evaluations": 0,
        "new_formula": "(2*K+1)*N_val",
        "max_branch_kernel_evaluations": MAX_BRANCH_KERNEL_EVALUATIONS,
        "max_combined_kernel_evaluations": MAX_COMBINED_KERNEL_EVALUATIONS,
        "guard_passed_before_validation_observations_source_conversion_or_bp": True,
        "cumulative_cap_is_scientific_provenance_guard_not_new_job_resource_consumption": True,
    }


def _claims(protocol: Mapping[str, object]) -> dict[str, object]:
    claims = dict(protocol["claims"])
    claims["conditional_response_result_does_not_change_physical_identity_status"] = True
    claims["external_evidence_needed"] = {
        "alignment": ">=3 independently surveyed non-collinear native/workbook anchors plus a held-out anchor",
        "point_c": "independent Point-C/scatterer definition or controlled calibration",
        "identity": "independent target-specific structure or signature to distinguish clutter",
    }
    return claims


def _base_disclosure(protocol: Mapping[str, object]) -> dict[str, object]:
    return {
        **dict(protocol["disclosure"]),
        "loaded_response_roles": ["train", "validation"],
        "used_response_roles": ["validation"],
        "source_af_formula": SOURCE_FORM,
        "conditional_response_failure_is_not_absence_of_reflector": True,
        "prior_result_status_not_used_for_classification": True,
    }


def _write_inconclusive(
    protocol: Mapping[str, object],
    output_dir: Path,
    prior: Mapping[str, object],
    feature: Mapping[str, object],
    train_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    disclosure = {
        **_base_disclosure(protocol),
        "current_archive_accessed": bool(train_provenance),
        "non_test_response_matrix_materialized_by_loader_before_new_work_guard": bool(train_provenance),
        "train_observation_objects_materialized_for_downstream": False,
        "loaded_response_roles": [],
        "used_response_roles": [],
        "loaded_response_role_counts": {},
        "validation_used_for_selection": False,
        "candidate_frozen_from_train": False,
        "test_payload_opened": False,
        "validation_response_payload_materialized_by_loader": False,
        "current_archive_train_metadata_semantically_matched": bool(
            train_provenance and train_provenance.get("semantic_match_to_prior_train_provenance") is True
        ),
    }
    reason = str(feature["reason"])
    preflight = {
        "prior_artifact_validated": True,
        "prior_historical_kernel_evaluations": int(prior["historical_total_kernel_evaluations"]),
        "current_archive_train_metadata": dict(train_provenance or {}),
        "candidate_preflight": feature,
        "validation_not_opened_for_selection_or_rescue": True,
        "response_condition_search": False,
        "status": DECISION_INCONCLUSIVE,
        "technical_execution_status": "PASS",
        "evidence_decision": DECISION_INCONCLUSIVE,
        "inconclusive_reason": reason,
        "disclosure": disclosure,
    }
    resource = {
        "schema": "rift_gotcha_step2_15tr07_correspondence_verify_resource_v1",
        "prior_historical_kernel_evaluations": int(prior["historical_total_kernel_evaluations"]),
        "new_validation_kernel_evaluations": 0,
        "combined_historical_plus_validation_kernel_evaluations": int(prior["historical_total_kernel_evaluations"]),
        "selected_validation_observation_objects_materialized_after_guard": False,
        "loader_materialized_non_test_response_before_new_work_guard": bool(train_provenance),
        "cumulative_cap_is_scientific_provenance_guard_not_new_job_resource_consumption": True,
        "extra_probe_kernel_evaluations": 0,
        "extra_fit_kernel_evaluations": 0,
    }
    train_feature = {**feature, "status": "INCONCLUSIVE", "used_for_validation": False}
    validation_confirmation = {
        "status": "NOT_RUN",
        "reason": reason,
        "used_for_selection": False,
        "candidate_frozen_from_train": False,
        "test_payload_opened": False,
    }
    report = {
        "schema": "rift_gotcha_step2_15tr07_correspondence_verify_report_v1",
        "status": "PASS",
        "technical_execution_status": "PASS",
        "evidence_decision": DECISION_INCONCLUSIVE,
        "decision": DECISION_INCONCLUSIVE,
        "inconclusive_reason": reason,
        "claims": _claims(protocol),
        "train_feature": train_feature,
        "validation_confirmation": validation_confirmation,
        "validation_used_for_selection": False,
        "candidate_frozen_from_train": False,
        "raw_source_imagery_context_only": True,
        "disclosure": disclosure,
    }
    files = [
        "protocol_echo.json", "preflight.json", "resource.json", "train_feature.json",
        "validation_confirmation.json", "correspondence_report.json", "status.json",
    ]
    _write_json(output_dir / "protocol_echo.json", protocol)
    _write_json(output_dir / "preflight.json", preflight)
    _write_json(output_dir / "resource.json", resource)
    _write_json(output_dir / "train_feature.json", train_feature)
    _write_json(output_dir / "validation_confirmation.json", validation_confirmation)
    _write_json(output_dir / "correspondence_report.json", report)
    status = {
        "schema": DRIVER_SCHEMA,
        "status": "PASS",
        "technical_execution_status": "PASS",
        "stage": "verification",
        "decision": DECISION_INCONCLUSIVE,
        "evidence_decision": DECISION_INCONCLUSIVE,
        "output_dir": str(output_dir),
        "files": files,
        "test_payload_opened": False,
        "validation_used_for_selection": False,
    }
    _write_json(output_dir / "status.json", status)
    return status


def _decide(
    validation_bp: np.ndarray,
    validation_psf: np.ndarray,
    train_patch_norm: float,
    protocol: Mapping[str, object],
) -> dict[str, object]:
    metric = protocol["decision_metric"]
    validation_norm = float(np.linalg.norm(validation_bp))
    psf_norm = float(np.linalg.norm(validation_psf))
    ratio = float(validation_norm / train_patch_norm) if train_patch_norm > 0.0 else None
    rho = float(abs(np.vdot(validation_psf, validation_bp)) / (psf_norm * validation_norm)) if psf_norm > 0.0 and validation_norm > 0.0 else None
    expected_index = int(np.argmax(np.abs(validation_psf))) if validation_psf.size else None
    validation_index = int(np.argmax(np.abs(validation_bp))) if validation_bp.size else None
    peak_distance = (
        int(max(abs(expected_index // PATCH_SIDE - validation_index // PATCH_SIDE), abs(expected_index % PATCH_SIDE - validation_index % PATCH_SIDE)))
        if expected_index is not None and validation_index is not None
        else None
    )
    if ratio is None or rho is None or peak_distance is None:
        decision = DECISION_INCONCLUSIVE
    elif ratio >= float(metric["validation_train_patch_norm_ratio_threshold"]) and rho >= float(metric["rho_supported_threshold"]) and peak_distance <= int(metric["peak_distance_supported_max_cells"]):
        decision = DECISION_SUPPORTED
    elif ratio >= float(metric["validation_train_patch_norm_ratio_threshold"]) and (rho <= float(metric["rho_inconsistent_threshold"]) or peak_distance > int(metric["peak_distance_inconsistent_min_exclusive_cells"])):
        decision = DECISION_INCONSISTENT
    else:
        decision = DECISION_INCONCLUSIVE
    return {
        "decision": decision,
        "validation_patch_norm": validation_norm,
        "train_patch_norm": float(train_patch_norm),
        "validation_train_patch_norm_ratio": ratio,
        "rho": rho,
        "psf_patch_norm": psf_norm,
        "psf_expected_local_index": expected_index,
        "validation_local_max_index": validation_index,
        "validation_peak_distance_cells": peak_distance,
        "metric": {
            "formula": metric["formula"],
            "validation_train_patch_norm_ratio_threshold": metric["validation_train_patch_norm_ratio_threshold"],
            "rho_supported_threshold": metric["rho_supported_threshold"],
            "rho_inconsistent_threshold": metric["rho_inconsistent_threshold"],
            "peak_distance_supported_max_cells": metric["peak_distance_supported_max_cells"],
            "peak_distance_inconsistent_min_exclusive_cells": metric["peak_distance_inconsistent_min_exclusive_cells"],
            "no_optimizing_phase_or_gain": metric["no_optimizing_phase_or_gain"],
        },
        "interpretation": "fixed phase- and amplitude-invariant descriptive shape consistency; not native-complex held-out prediction",
    }


def run_verification(
    protocol_path: str | Path,
    archive_root: str | Path,
    prior_output_dir: str | Path,
    output_dir: str | Path,
    stage: str = "verification",
) -> dict[str, object]:
    protocol = _read_json(Path(protocol_path))
    _validate_protocol(protocol, stage)
    prior_output_dir = Path(prior_output_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory must be fresh: {output_dir}")
    prior = _validate_prior_artifact(prior_output_dir, protocol)
    candidate_preflight = _select_train_feature(prior)
    if not bool(candidate_preflight.get("proceed")):
        return _write_inconclusive(protocol, output_dir, prior, candidate_preflight)

    relative = Path(str(protocol["source"]["relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("protocol source path must be a safe relative path")
    archive_path = Path(archive_root) / relative
    if not archive_path.is_file():
        raise FileNotFoundError(f"protocol archive is missing: {archive_path}")
    shard = ACQ.load_native_shard(
        archive_path,
        expected_pass_id=1,
        expected_polarization="hh",
        expected_scene_id=str(protocol["source"]["scene_id"]),
    )
    _train_ids, train_provenance = _validate_current_train_metadata(shard, prior)
    feature = _select_train_feature(prior)
    if not bool(feature.get("proceed")):
        return _write_inconclusive(protocol, output_dir, prior, feature, train_provenance)

    validation_ids, loaded_roles, loaded_role_counts = _select_validation_metadata(shard)
    # ACQ.load_native_shard has already materialized the non-test response
    # matrix after its own metadata gate. This is the sole new-work guard
    # before selected validation observations, source conversion, or operators.
    budget = _new_validation_budget(len(validation_ids), int(prior["historical_total_kernel_evaluations"]))
    preflight = {
        "prior_artifact_validated": True,
        "prior_historical_budget": prior["historical_budget"],
        "prior_renderer_png_provenance": prior["prior_pngs"],
        "current_archive_train_metadata": train_provenance,
        "candidate_preflight": feature,
        "validation_scope_preflight": {
            "sector_ids": list(VALIDATION_SECTORS),
            "role": "validation",
            "selected_record_count": len(validation_ids),
            "selected_ids": [identity.as_dict() for identity in validation_ids],
            "loaded_roles": loaded_roles,
            "loaded_response_role_counts": loaded_role_counts,
            "loader_materialized_non_test_response_before_new_work_guard": True,
            "test_payload_opened": False,
        },
        "budget": budget,
        "guard_passed_before_validation_observations_source_conversion_or_bp": True,
        "train_only_discrete_grid_argmax_selection": True,
        "response_condition_search": False,
            "validation_selection": False,
        "prior_result_status_not_used_for_classification": True,
    }
    observations = tuple(shard.observations(validation_ids))
    validation_scope = SOURCE_AF.SourceAFScope(
        pass_id=1,
        polarization="hh",
        sector_ids=VALIDATION_SECTORS,
        role="validation",
        expected_count=None,
        name="15tr07_correspondence_validation_sector_271",
    )
    _validate_validation_contract(shard, observations)
    source_records = SOURCE_AF.build_source_af(observations, scope=validation_scope)
    ix, iy = (int(feature["grid_index_ix_iy"][0]), int(feature["grid_index_ix_iy"][1]))
    patch_indices = _patch_flat_indices(ix, iy)
    x = np.asarray(prior["x"], dtype=np.float64)
    y = np.asarray(prior["y"], dtype=np.float64)
    patch_coordinates = np.asarray(
        [[x[x_index], y[y_index], -0.05] for x_index in range(ix - PATCH_RADIUS_CELLS, ix + PATCH_RADIUS_CELLS + 1) for y_index in range(iy - PATCH_RADIUS_CELLS, iy + PATCH_RADIUS_CELLS + 1)],
        dtype=np.float64,
    )
    if not np.array_equal(patch_indices, np.asarray(feature["patch_flat_indices"], dtype=np.int64)):
        raise ValueError("frozen train patch indices changed before validation")
    started = time.perf_counter()
    validation_bp_unnormalized = SOURCE_AF.direct_backproject(
        source_records,
        patch_coordinates,
        representation=SOURCE_AF.SOURCE_REPRESENTATION,
        normalization="none",
        point_chunk_size=int(protocol["support"]["point_chunk_size"]),
        max_kernel_evaluations=MAX_BRANCH_KERNEL_EVALUATIONS,
    )
    validation_psf_unnormalized = SOURCE_AF.matched_unit_point_psf(
        source_records,
        patch_coordinates,
        np.asarray(feature["conditional_train_grid_feature_xyz_m"], dtype=np.float64),
        representation=SOURCE_AF.SOURCE_REPRESENTATION,
        normalization="none",
        point_chunk_size=int(protocol["support"]["point_chunk_size"]),
        max_kernel_evaluations=MAX_BRANCH_KERNEL_EVALUATIONS,
    )
    elapsed = time.perf_counter() - started
    n_val = int(budget["N_val"])
    validation_bp = np.asarray(validation_bp_unnormalized, dtype=np.complex128) / float(n_val)
    validation_psf = np.asarray(validation_psf_unnormalized, dtype=np.complex128) / float(n_val)
    _finite_complex(validation_bp, "validation BP patch")
    _finite_complex(validation_psf, "validation PSF patch")
    decision = _decide(validation_bp, validation_psf, float(feature["train_source_bp_patch_norm"]), protocol)
    disclosure = {
        **_base_disclosure(protocol),
        "current_archive_accessed": True,
        "non_test_response_matrix_materialized_by_loader_before_new_work_guard": True,
        "train_observation_objects_materialized_for_downstream": False,
        "loaded_response_roles": loaded_roles,
        "loaded_response_role_counts": loaded_role_counts,
        "loaded_response_count": int(shard.view_count),
        "used_response_roles": ["validation"],
        "selected_validation_response_count": len(validation_ids),
        "selected_validation_response_shape": [len(validation_ids), FREQUENCY_COUNT],
        "validation_used_for_selection": False,
        "candidate_frozen_from_train": True,
        "validation_used_for_model_fitting": False,
        "test_payload_opened": False,
        "test_response_payload_materialized": False,
        "validation_response_payload_materialized_by_loader": True,
        "selected_validation_observation_objects_materialized_after_guard": True,
        "validation_source_representation": SOURCE_FORM,
        "validation_supplied_af_provenance": "channel-owned per-row corrections supplied in the native archive; corrections may have used validation responses",
        "validation_aperture_relation": "adjacent aperture slice sector 271; not independently surveyed correspondence",
    }
    validation_confirmation = {
        "status": "PASS",
        "selected_ids": [identity.as_dict() for identity in validation_ids],
        "selected_record_count": len(validation_ids),
        "selected_response_shape": [len(validation_ids), FREQUENCY_COUNT],
        "sector_ids": list(VALIDATION_SECTORS),
        "role": "validation",
        "used_for_selection": False,
        "candidate_frozen_from_train": True,
        "validation_used_for_model_fitting": False,
        "test_payload_opened": False,
        "source_scope": validation_scope.as_dict(),
        "decision_metric": decision,
    }
    train_feature = {
        **feature,
        "status": "FROZEN",
        "used_for_validation": True,
        "validation_used_for_selection": False,
    }
    report = {
        "schema": "rift_gotcha_step2_15tr07_correspondence_verify_report_v1",
        "status": "PASS",
        "technical_execution_status": "PASS",
        "evidence_decision": decision["decision"],
        "decision": decision["decision"],
        "diagnostic_statement": "conditional held-out source-AF local point-response shape consistency only; not registered coordinate error, Point-C/scattering-center offset, physical 15TR-07 identity, native-complex held-out prediction, transform, gain/phase fit, localization, accuracy, focus, geometry, registration, calibration closure, or model fit",
        "result_failure_scope": "a failure is only a failure of the fixed supplied-AF single-point response model under adjacent-sector validation, never absence of a reflector",
        "claims": _claims(protocol),
        "prior_raw_source_diagnostic_context_only": True,
        "prior_historical_kernel_evaluations": int(prior["historical_total_kernel_evaluations"]),
        "new_validation_budget": budget,
        "train_feature": train_feature,
        "validation_confirmation": validation_confirmation,
        "validation_used_for_selection": False,
        "candidate_frozen_from_train": True,
        "validation_used_for_model_fitting": False,
        "decision_metric": decision,
        "no_recentering_or_transform": True,
        "no_validation_phase_or_gain_optimization": True,
        "disclosure": disclosure,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    np.save(output_dir / "validation_bp_patch_mean_native_sample.npy", validation_bp)
    np.save(output_dir / "validation_psf_patch_mean_native_sample.npy", validation_psf)
    np.save(output_dir / "validation_patch_coordinates.npy", patch_coordinates)
    _write_json(output_dir / "protocol_echo.json", protocol)
    _write_json(output_dir / "preflight.json", preflight)
    _write_json(
        output_dir / "resource.json",
        {
            "schema": "rift_gotcha_step2_15tr07_correspondence_verify_resource_v1",
            "budget": budget,
            "elapsed_seconds": elapsed,
            "point_count": PATCH_POINT_COUNT,
            "point_chunk_size": int(protocol["support"]["point_chunk_size"]),
            "kernel_ledger": {
                "validation_source_bp": "one source-AF measured validation BP",
                "validation_unit_point_forward": "one matched source-AF unit-point forward",
                "validation_source_psf_bp": "one matched source-AF unit-point PSF BP",
                "extra_probe_or_reference_passes": 0,
                "extra_fit_kernel_evaluations": 0,
            },
            "prior_historical_kernel_evaluations": int(prior["historical_total_kernel_evaluations"]),
            "new_validation_kernel_evaluations": int(budget["new_validation_kernel_evaluations"]),
            "combined_historical_plus_validation_kernel_evaluations": int(budget["combined_historical_plus_validation_kernel_evaluations"]),
            "selected_validation_observation_objects_materialized_after_guard": True,
            "loader_materialized_non_test_response_before_new_work_guard": True,
            "cumulative_cap_is_scientific_provenance_guard_not_new_job_resource_consumption": True,
            "guard_passed_before_bp": True,
        },
    )
    _write_json(output_dir / "train_feature.json", train_feature)
    _write_json(output_dir / "validation_confirmation.json", validation_confirmation)
    _write_json(output_dir / "correspondence_report.json", report)
    files = [
        "protocol_echo.json", "preflight.json", "resource.json", "train_feature.json",
        "validation_confirmation.json", "correspondence_report.json", "status.json",
        "validation_bp_patch_mean_native_sample.npy",
        "validation_psf_patch_mean_native_sample.npy",
        "validation_patch_coordinates.npy",
    ]
    status = {
        "schema": DRIVER_SCHEMA,
        "status": "PASS",
        "technical_execution_status": "PASS",
        "stage": "verification",
        "decision": decision["decision"],
        "evidence_decision": decision["decision"],
        "output_dir": str(output_dir),
        "files": files,
        "test_payload_opened": False,
        "validation_used_for_selection": False,
        "candidate_frozen_from_train": True,
    }
    _write_json(output_dir / "status.json", status)
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--prior-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args(argv)
    result = run_verification(args.protocol, args.archive_root, args.prior_output_dir, args.output_dir, args.stage)
    print(
        json.dumps(
            {
                "status": result["status"],
                "technical_execution_status": result.get("technical_execution_status", "PASS"),
                "decision": result.get("decision"),
                "evidence_decision": result.get("evidence_decision", result.get("decision")),
                "output_dir": str(Path(args.output_dir)),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
