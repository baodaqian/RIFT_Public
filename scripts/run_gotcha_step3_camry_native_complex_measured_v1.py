"""Run one bounded measured Camry source-AF native-complex readiness diagnostic.

The driver is intentionally PACE-safe: the caller supplies an archive root and a
fresh output directory, and all selected identities are read from the trusted
Gate-1 loader at runtime.  It performs no TEST/validation selection, registration
fit, autofocus search, target crop, or production reconstruction claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Use the trusted source modules under their direct-loader names so this script
# remains usable from the bundled Python environment, where importing rift.__init__
# may require optional training dependencies.
ACQ = _load_module("gotcha_acquisition", PROJECT_ROOT / "rift" / "gotcha_acquisition.py")
SOURCE_AF = _load_module("gotcha_source_af", PROJECT_ROOT / "rift" / "gotcha_source_af.py")
STEP2 = _load_module("gotcha_step2_controls", PROJECT_ROOT / "rift" / "gotcha_step2_controls.py")
V2 = _load_module("gotcha_step3_native_complex", PROJECT_ROOT / "rift" / "gotcha_step3_native_complex.py")


PROTOCOL_SCHEMA = "rift_gotcha_step3_camry_native_complex_measured_protocol_v1"
DRIVER_SCHEMA = "rift_gotcha_step3_camry_native_complex_measured_driver_v1"
# NumPy and Torch both use float64 at the source-AF boundary, but their
# independent norm/phase/complex-exp implementations need not produce bitwise
# identical values at kilometre-scale ranges and 9--10 GHz phase arguments.
# These are deliberately narrow, dimensionless acceptance limits; the
# geometry-derived roundoff budget below is reported alongside them.
BRIDGE_RELATIVE_TOLERANCE = 2.0e-9
BRIDGE_RANGE_ERROR_EPSILON_FACTOR = 1.0
BRIDGE_R0_ERROR_EPSILON_FACTOR = 1.0
BRIDGE_PHASE_REDUCTION_EPSILON_FACTOR = 1.0
BRIDGE_EXP_ERROR_EPSILON_FACTOR = 1.0
EXPECTED_OUTPUTS = {
    "protocol_echo": "protocol_echo.json",
    "selection": "selection.json",
    "resource": "resource.json",
    "native_complex_report": "native_complex_report.json",
    "ridge_checkpoint": "ridge_checkpoint.npz",
    "status": "status.json",
}


def _json_ready(value: Any) -> Any:
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


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_ready(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_measured_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle, object_pairs_hook=_reject_duplicate_json_pairs)
    if not isinstance(payload, dict):
        raise ValueError("measured protocol must be a JSON object")
    validate_measured_protocol(payload)
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_measured_protocol(protocol: Mapping[str, Any]) -> None:
    """Validate the fixed measured-readiness contract before archive access."""

    _require(protocol.get("schema") == PROTOCOL_SCHEMA, "unexpected measured Camry protocol schema")
    _require(protocol.get("stage") == "measured_readiness", "measured protocol stage must be measured_readiness")
    _require(protocol.get("data_free") is False, "measured protocol must be data-bearing and explicitly labeled")
    _require(protocol.get("measured_fit_release") is False, "measured readiness must not release a fit")
    _require(
        protocol.get("run_kind") == "one_bounded_p1_hh_train_sector002_source_af_native_complex",
        "measured protocol run kind is not the bounded Camry package",
    )

    scope = protocol.get("scope")
    _require(isinstance(scope, Mapping), "measured protocol scope is missing")
    for key, expected in {
        "pass_id": 1,
        "polarization": "hh",
        "role": "train",
        "sector_id": 2,
        "canonical_identity": "(pass_id, polarization, sector_id, pulse_index)",
        "identity_order": "ascending canonical identity",
        "all_native_pulses_in_sector": True,
        "expected_pulse_count": 117,
        "expected_native_frequency_count_per_pulse": 424,
        "validation_selection": False,
        "validation_records_materialized": False,
        "validation_response_payload_materialized": True,
        "test_payload_opened": False,
        "test_payload_materialized": False,
    }.items():
        _require(scope.get(key) == expected, f"scope.{key} does not preserve the measured contract")
    _require(scope.get("loaded_roles") == ["train", "validation"], "scope.loaded_roles must disclose both loader roles")
    _require(scope.get("materialized_roles") == ["train"], "scope.materialized_roles must be TRAIN-only")
    _require(scope.get("used_roles") == ["train"], "scope.used_roles must be TRAIN-only")

    source = protocol.get("source")
    _require(isinstance(source, Mapping), "measured protocol source is missing")
    for key, expected in {
        "archive_root_contract": "converted_v3_joint8_fullpol",
        "relative_path": "shards/pass1_hh.npz",
        "scene_id": "gotcha_v1_joint8_fullpol",
        "loader": "rift.gotcha_acquisition.load_native_shard",
    }.items():
        _require(source.get(key) == expected, f"source.{key} does not preserve the trusted archive contract")
    expectations = source.get("loader_expectations")
    _require(
        expectations == {
            "expected_pass_id": 1,
            "expected_polarization": "hh",
            "expected_scene_id": "gotcha_v1_joint8_fullpol",
        },
        "source.loader_expectations changed",
    )
    _require(
        source.get("selection_source") == "actual_loaded_archive_records; IDs are emitted at runtime and are not embedded here",
        "source.selection_source must require runtime archive identities",
    )

    camry = protocol.get("camry")
    _require(isinstance(camry, Mapping), "measured protocol Camry declaration is missing")
    _require(camry.get("target_id") == "toyota_camry", "measured package is limited to Toyota Camry")
    _require(camry.get("placement_equation") == "p_native = R @ p_local + t", "Camry placement equation changed")
    _require(np.array_equal(np.asarray(camry.get("R"), dtype=np.float64), V2.CAMRY_VEHICLE_AXIS_ROTATION), "Camry R changed")
    _require(np.array_equal(np.asarray(camry.get("t_m"), dtype=np.float64), V2.CAMRY_TRANSLATION_M), "Camry t changed")
    _require(
        camry.get("placement_qualification") == "footprint_derived_vehicle_axis_inference_not_independently_registered",
        "Camry placement qualification must remain conditional",
    )
    _require(camry.get("physical_registration_claim") is False, "physical registration must not be claimed")
    _require(camry.get("support_count") == 4, "measured support must contain exactly four points")
    _require(camry.get("support_semantics") == "direct_complex_point_weights", "support must be direct complex point weights")
    bridge_probe = np.asarray(camry.get("bridge_probe_coefficients_real_imag"), dtype=np.float64)
    _require(bridge_probe.shape == (4, 2), "bridge probe must contain four real/imag pairs")
    _require(np.isfinite(bridge_probe).all() and np.any(bridge_probe != 0.0), "bridge probe must be finite and nonzero")
    _require(camry.get("support_half_open_cube") == "[-5,5)^3", "Camry support cube changed")
    _require(camry.get("reference_grid_quadrature") is False, "reference-grid quadrature is forbidden")
    _require(camry.get("spherical_harmonic_Y00") is False, "Y00 quadrature is forbidden")
    _require(camry.get("global_complex_gain") == {"value": "1+0j", "trainable": False}, "global gain must remain frozen")
    support_local = np.asarray(camry.get("support_local_points_m"), dtype=np.float64)
    _require(support_local.shape == (4, 3), "measured support_local_points_m must be [4,3]")
    _require(np.array_equal(support_local, V2.smoke_support_local()), "measured support points changed from the reviewed four-point support")
    V2.transform_smoke_support("toyota_camry", support_local)
    footprint = V2.camry_footprint_consistency()
    _require(footprint["all_corners_within_absolute_tolerance"], "Camry footprint corner tolerance failed")
    _require(footprint["inverse_footprint_contained_in_camry_grid"], "Camry inverse footprint containment failed")

    native_contract = protocol.get("native_contract")
    _require(isinstance(native_contract, Mapping), "native contract is missing")
    for key, expected in {
        "frequency_policy": "native_stored_exact",
        "frequency_vector_layout": "one_shared_exact_vector_per_shard",
        "r0_field": "r0",
        "geometry": "paired_monostatic_tx_equals_rx_same_observation",
        "source_af_formula": SOURCE_AF.SOURCE_AF_FORMULA,
        "phase_hypothesis_name": "demanet_2012_gotcha_phase_candidate",
        "phase_forward": "exp(-i * 4*pi*f/c * (R-r0))",
        "phase_adjoint": "conjugate(exp(-i * 4*pi*f/c * (R-r0)))",
        "phase_status": "named_candidate_unverified_for_real_archive",
        "identity_transform": "no spatial spotlight; T=identity",
        "data_prediction_application": "same exact complex T before residual, loss, and energy",
        "no_target_crop_subtraction_padding": True,
        "magnitude_conversion": False,
    }.items():
        _require(native_contract.get(key) == expected, f"native_contract.{key} changed")
    bridge_tolerance = native_contract.get("bridge_tolerance")
    _require(isinstance(bridge_tolerance, Mapping), "native_contract.bridge_tolerance is missing")
    for key, expected in {
        "relative_l2": BRIDGE_RELATIVE_TOLERANCE,
        "range_error_epsilon_factor": BRIDGE_RANGE_ERROR_EPSILON_FACTOR,
        "r0_error_epsilon_factor": BRIDGE_R0_ERROR_EPSILON_FACTOR,
        "phase_reduction_epsilon_factor": BRIDGE_PHASE_REDUCTION_EPSILON_FACTOR,
        "exp_error_epsilon_factor": BRIDGE_EXP_ERROR_EPSILON_FACTOR,
    }.items():
        _require(bridge_tolerance.get(key) == expected, f"native_contract.bridge_tolerance.{key} changed")
    _require(
        bridge_tolerance.get("criterion")
        == "per-record scaled L2; scaled maximum absolute error reported diagnostically",
        "native_contract.bridge_tolerance.criterion changed",
    )
    _require(
        bridge_tolerance.get("acceptance_basis")
        == "empirical fixed-four-point nonzero-probe bridge envelope",
        "native_contract.bridge_tolerance.acceptance_basis changed",
    )

    optimizer = protocol.get("optimizer")
    _require(isinstance(optimizer, Mapping), "optimizer contract is missing")
    for key, expected in {
        "initialization": "zero real/imag coefficients",
        "ridge": 0.001,
        "max_iterations": 2,
        "max_kernel_evaluations": 2_000_000,
        "step_rule": "0.5 / (point_count * sum_abs_multiplier_squared / native_sample_count + ridge)",
        "cumulative_kernel_formula": "(1+2*max_iterations)*point_count*native_sample_count",
        "preflight_rejects_before_expensive_calculation": True,
        "checkpoint_prediction_parity_forward_calls": 2,
    }.items():
        _require(optimizer.get(key) == expected, f"optimizer.{key} changed")
    _require(
        optimizer.get("lipschitz_scope") == "unit-modulus kernel entries and no quadrature amplitude weights",
        "optimizer Lipschitz scope changed",
    )
    declared_ledger = optimizer.get("resource_ledger")
    _require(isinstance(declared_ledger, Mapping), "optimizer.resource_ledger is missing")
    for key, expected in {
        "selected_pulses": 117,
        "native_frequency_samples": 49_608,
        "point_count": 4,
        "K_point_times_native_samples": 198_432,
        "numpy_torch_bridge": "2K",
        "optimizer_trajectory": "(1+2I)K = 5K",
        "final_reloaded_prediction_parity": "2K",
        "total": "9K = 1785888",
    }.items():
        _require(declared_ledger.get(key) == expected, f"optimizer.resource_ledger.{key} changed")
    _require(protocol.get("outputs") == EXPECTED_OUTPUTS, "measured output names changed")
    disclosure = protocol.get("release_disclosure")
    _require(isinstance(disclosure, Mapping), "release disclosure is missing")
    for key in ("not_recovery", "not_production_fit", "not_isolated_target_claim", "not_physical_registration_claim", "not_accuracy_focus_geometry_evidence", "decreasing_loss_is_not_recovery"):
        _require(disclosure.get(key) is True, f"release disclosure.{key} must remain true")


def _identity_key(identity: Any) -> tuple[int, str, int, int]:
    try:
        return (
            int(identity.pass_id),
            str(identity.polarization).lower(),
            int(identity.sector_id),
            int(identity.pulse_index),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError("archive identities must expose pass/polarization/sector/pulse") from exc


def _identity_dict(identity: Any) -> dict[str, Any]:
    if hasattr(identity, "as_dict"):
        return dict(identity.as_dict())
    key = _identity_key(identity)
    return {"pass_id": key[0], "polarization": key[1], "sector_id": key[2], "pulse_index": key[3]}


def select_train_sector002(shard: Any, protocol: Mapping[str, Any]) -> tuple[Any, ...]:
    """Seal the split and return sorted IDs without materializing observations."""

    scope = protocol["scope"]
    ids = tuple(shard.observation_ids)
    roles = tuple(str(value).lower() for value in np.asarray(shard.role).tolist())
    if len(ids) != len(roles) or not ids:
        raise ValueError("archive identity and role vectors must be nonempty and aligned")
    identity_keys = tuple(_identity_key(identity) for identity in ids)
    if len(set(identity_keys)) != len(identity_keys):
        raise ValueError("archive contains duplicate canonical observation identities")
    if any(key[0] != 1 or key[1] != "hh" for key in identity_keys):
        raise ValueError("archive identity scope differs from P1 HH")
    if any(role == "test" for role in roles):
        raise ValueError("sealed TEST rows cannot enter measured readiness selection")
    if set(roles) != {"train", "validation"}:
        raise ValueError("archive must expose exactly the trusted train/validation roles")
    metadata = getattr(shard, "metadata", {})
    for key, expected in (("test_opened", False), ("test_payload_included", False), ("corrections_applied", False), ("autofocus_unapplied", True)):
        if metadata.get(key) != expected:
            raise ValueError(f"archive metadata.{key} must be {expected!r}")
    selected = tuple(
        identity
        for identity, role in zip(ids, roles)
        if role == str(scope["role"]).lower() and int(identity.sector_id) == int(scope["sector_id"])
    )
    selected = tuple(sorted(selected, key=_identity_key))
    if len(selected) != int(scope["expected_pulse_count"]):
        raise ValueError(f"sector 002 selected {len(selected)} pulses, expected {scope['expected_pulse_count']}")
    if tuple(_identity_key(identity) for identity in selected) != tuple(sorted(_identity_key(identity) for identity in selected)):
        raise ValueError("selected archive IDs are not in canonical order")
    if any(_identity_key(identity)[2] != 2 for identity in selected):
        raise ValueError("selected archive IDs are not sector 002")
    return selected


def validate_selected_records(
    records: Sequence[Any],
    selected_ids: Sequence[Any],
    *,
    expected_frequencies: np.ndarray,
    protocol: Mapping[str, Any],
) -> None:
    """Validate source-AF prerequisites after the identity preflight."""

    records = tuple(records)
    selected_ids = tuple(selected_ids)
    if tuple(getattr(record, "identity", None) for record in records) != selected_ids:
        raise ValueError("materialized records do not preserve archive-selected identity order")
    if len(records) != int(protocol["scope"]["expected_pulse_count"]):
        raise ValueError("materialized record count differs from measured scope")
    frequencies = np.asarray(expected_frequencies)
    if frequencies.ndim != 1 or frequencies.size != 424:
        raise ValueError("trusted native frequency vector must have 424 entries")
    for record in records:
        if str(record.role).lower() != "train":
            raise ValueError("measured source-AF materialization is TRAIN-only")
        record_frequencies = np.asarray(record.frequencies_hz)
        if not np.isrealobj(record_frequencies) or record_frequencies.ndim != 1 or not np.array_equal(record_frequencies, frequencies):
            raise ValueError("record frequency vector is not the exact shared native vector")
        response = np.asarray(record.response)
        if not np.iscomplexobj(response) or response.shape != frequencies.shape:
            raise ValueError("record response must be complex and aligned to native frequencies")
        position = np.asarray(record.position_xyz_m)
        if not np.isrealobj(position) or position.shape != (3,):
            raise ValueError("record native coordinates must have shape [3]")
        for value, label in (
            (response.real, "response.real"),
            (response.imag, "response.imag"),
            (position, "position_xyz_m"),
            (frequencies, "frequencies_hz"),
        ):
            if not np.isfinite(value).all():
                raise ValueError(f"{label} contains non-finite values")
        if record.r_correct_raw is None or record.ph_correct_raw is None:
            raise ValueError("measured HH source-AF records must retain both raw corrections")
        if not np.isfinite(float(record.r_correct_raw)) or not np.isfinite(float(record.ph_correct_raw)):
            raise ValueError("measured HH raw corrections must be finite")
        autofocus = record.autofocus
        if not autofocus.official_available or autofocus.applied:
            raise ValueError("measured source-AF requires official raw unapplied autofocus provenance")
        if autofocus.mode != ACQ.AUTOFOCUS_RAW:
            raise ValueError("measured source-AF requires the channel-owned raw autofocus mode")
        phase_reference = record.phase_reference
        if phase_reference.reference_range_field != "r0" or phase_reference.geometry_contract != "paired_monostatic_tx_equals_rx_same_observation":
            raise ValueError("measured source-AF phase/reference provenance is not qualified")


def resource_ledger(
    selected_count: int,
    frequency_counts: Sequence[int],
    point_count: int,
    max_iterations: int,
    max_kernel_evaluations: int,
) -> dict[str, int | str]:
    """Compute and cap the complete measured trajectory before kernel work."""

    selected_count = int(selected_count)
    point_count = int(point_count)
    max_iterations = int(max_iterations)
    cap = int(max_kernel_evaluations)
    counts = tuple(int(value) for value in frequency_counts)
    if selected_count <= 0 or len(counts) != selected_count or any(value <= 0 for value in counts):
        raise ValueError("resource ledger frequency counts must align to positive selected records")
    if point_count <= 0 or max_iterations <= 0 or cap <= 0:
        raise ValueError("resource ledger bounds must be positive")
    native_samples = int(sum(counts))
    k = int(point_count * native_samples)
    bridge = int(2 * k)
    optimizer = int((1 + 2 * max_iterations) * k)
    parity = int(2 * k)
    total = int(bridge + optimizer + parity)
    ledger: dict[str, int | str] = {
        "selected_pulses": selected_count,
        "native_frequency_samples": native_samples,
        "point_count": point_count,
        "K_point_times_native_samples": k,
        "numpy_torch_bridge_kernel_evaluations": bridge,
        "optimizer_kernel_evaluations": optimizer,
        "final_reloaded_prediction_parity_kernel_evaluations": parity,
        "total_kernel_evaluations": total,
        "max_kernel_evaluations": cap,
        "optimizer_cumulative_formula": "(1+2*I)*K",
        "preflight_status": "PASS",
    }
    if total > cap:
        ledger["preflight_status"] = "REJECT_CAP_EXCEEDED"
        raise RuntimeError(f"measured kernel-evaluation cap exceeded before expensive calculation: {total}>{cap}")
    return ledger


def _relative_error(actual: Any, expected: Any) -> float:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    return float(np.linalg.norm(actual_array - expected_array) / max(1.0, float(np.linalg.norm(expected_array))))


def bridge_parity_report(
    actual: Sequence[Any],
    expected: Sequence[Any],
    *,
    points_xyz_m: Any,
    records: Sequence[Any],
    coefficients: Any,
) -> dict[str, Any]:
    """Apply the empirical per-record NumPy/Torch bridge criterion.

    ``actual`` and ``expected`` are already routed through the declared
    measurement transform.  The acceptance check intentionally remains
    per-record: averaging 117 pulses could hide one bad observation.  The
    scale-aware maximum error is retained as a diagnostic alongside the L2
    result, not as a second redundant gate.

    The arithmetic scale context is not a theorem or acceptance gate.  It
    combines first-order float64 distance/r0 roundoff, phase sensitivity at the
    largest native frequency and phase argument, the unit-modulus complex
    exponential scale, and the L1 weight contribution over native frequencies.
    """

    actual_values = tuple(actual)
    expected_values = tuple(expected)
    records_tuple = tuple(records)
    if len(actual_values) != len(expected_values) or len(actual_values) != len(records_tuple):
        raise AssertionError("bridge predictions and source-AF records must have matching lengths")
    if not records_tuple:
        raise AssertionError("bridge parity requires at least one record")
    points = np.asarray(points_xyz_m, dtype=np.float64)
    coefficients_array = np.asarray(coefficients, dtype=np.complex128)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("bridge points must have shape [point,3]")
    if coefficients_array.shape != (points.shape[0],):
        raise ValueError("bridge coefficients must have one value per support point")
    if not np.isfinite(points).all() or not np.isfinite(coefficients_array.real).all() or not np.isfinite(coefficients_array.imag).all():
        raise ValueError("bridge points and coefficients must be finite")

    epsilon = float(np.finfo(np.float64).eps)
    coefficient_l1 = float(np.sum(np.abs(coefficients_array)))
    per_record: list[dict[str, Any]] = []
    max_coordinate_scale = 1.0
    max_frequency_hz = 0.0
    max_abs_differential_range_m = 0.0
    max_phase_argument_rad = 0.0
    max_arithmetic_l2_scale = 0.0
    max_relative_l2 = 0.0
    max_scaled_max_absolute = 0.0
    for index, (actual_value, expected_value, record) in enumerate(
        zip(actual_values, expected_values, records_tuple)
    ):
        actual_array = np.asarray(actual_value, dtype=np.complex128)
        expected_array = np.asarray(expected_value, dtype=np.complex128)
        frequencies = np.asarray(record.frequencies_hz, dtype=np.float64)
        position = np.asarray(record.position_xyz_m, dtype=np.float64)
        if actual_array.shape != expected_array.shape or actual_array.shape != frequencies.shape:
            raise AssertionError(f"bridge shape mismatch at record {index}")
        if position.shape != (3,) or not np.isfinite(position).all() or not np.isfinite(frequencies).all():
            raise ValueError(f"bridge geometry/frequency inputs are invalid at record {index}")
        if not np.isfinite(actual_array.real).all() or not np.isfinite(actual_array.imag).all():
            raise ValueError(f"bridge actual prediction is non-finite at record {index}")
        if not np.isfinite(expected_array.real).all() or not np.isfinite(expected_array.imag).all():
            raise ValueError(f"bridge expected prediction is non-finite at record {index}")
        delta = actual_array - expected_array
        actual_l2 = float(np.linalg.norm(actual_array))
        expected_l2 = float(np.linalg.norm(expected_array))
        scale_l2 = max(1.0, actual_l2, expected_l2)
        delta_l2 = float(np.linalg.norm(delta))
        relative_l2 = delta_l2 / scale_l2
        actual_max = float(np.max(np.abs(actual_array))) if actual_array.size else 0.0
        expected_max = float(np.max(np.abs(expected_array))) if expected_array.size else 0.0
        scale_inf = max(1.0, actual_max, expected_max)
        delta_max = float(np.max(np.abs(delta))) if delta.size else 0.0
        scaled_max_absolute = delta_max / scale_inf

        distance = np.linalg.norm(points - position[None, :], axis=1)
        differential_range = distance - float(record.effective_r0_m)
        frequency_max = float(np.max(np.abs(frequencies)))
        phase_sensitivity = (4.0 * math.pi / V2.SPEED_OF_LIGHT_M_S) * frequency_max
        phase_argument_max = float(np.max(np.abs(phase_sensitivity * differential_range)))
        coordinate_scale = max(
            1.0,
            float(np.max(np.abs(points))),
            float(np.max(np.abs(position))),
            abs(float(record.effective_r0_m)),
        )
        range_roundoff_m = BRIDGE_RANGE_ERROR_EPSILON_FACTOR * epsilon * coordinate_scale
        r0_roundoff_m = BRIDGE_R0_ERROR_EPSILON_FACTOR * epsilon * coordinate_scale
        phase_reduction_scale_rad = (
            BRIDGE_PHASE_REDUCTION_EPSILON_FACTOR * epsilon * max(1.0, phase_argument_max)
        )
        phase_roundoff_rad = phase_sensitivity * (range_roundoff_m + r0_roundoff_m) + phase_reduction_scale_rad
        unit_kernel_roundoff = phase_roundoff_rad + BRIDGE_EXP_ERROR_EPSILON_FACTOR * epsilon
        arithmetic_l2_scale = math.sqrt(float(frequencies.size)) * coefficient_l1 * unit_kernel_roundoff

        max_coordinate_scale = max(max_coordinate_scale, coordinate_scale)
        max_frequency_hz = max(max_frequency_hz, frequency_max)
        max_abs_differential_range_m = max(max_abs_differential_range_m, float(np.max(np.abs(differential_range))))
        max_phase_argument_rad = max(max_phase_argument_rad, phase_argument_max)
        max_arithmetic_l2_scale = max(max_arithmetic_l2_scale, arithmetic_l2_scale)
        max_relative_l2 = max(max_relative_l2, relative_l2)
        max_scaled_max_absolute = max(max_scaled_max_absolute, scaled_max_absolute)
        relative_pass = relative_l2 <= BRIDGE_RELATIVE_TOLERANCE
        per_record.append(
            {
                "index": index,
                "native_frequency_count": int(frequencies.size),
                "actual_l2": actual_l2,
                "expected_l2": expected_l2,
                "scale_l2": scale_l2,
                "delta_l2": delta_l2,
                "relative_l2": relative_l2,
                "scale_inf": scale_inf,
                "delta_max_absolute": delta_max,
                "scaled_max_absolute": scaled_max_absolute,
                "coordinate_scale_m": coordinate_scale,
                "frequency_max_hz": frequency_max,
                "max_abs_differential_range_m": float(np.max(np.abs(differential_range))),
                "phase_argument_max_rad": phase_argument_max,
                "phase_sensitivity_rad_per_m": phase_sensitivity,
                "range_roundoff_bound_m": range_roundoff_m,
                "r0_roundoff_bound_m": r0_roundoff_m,
                "phase_roundoff_bound_rad": phase_roundoff_rad,
                "phase_reduction_scale_rad": phase_reduction_scale_rad,
                "unit_kernel_roundoff_bound": unit_kernel_roundoff,
                "arithmetic_l2_scale_context": arithmetic_l2_scale,
                "relative_l2_pass": relative_pass,
                "passed": bool(relative_pass),
            }
        )
    return {
        "criterion": "per-record scaled L2; scaled maximum absolute error reported diagnostically",
        "acceptance_basis": "empirical fixed-four-point nonzero-probe bridge envelope",
        "relative_l2_tolerance": BRIDGE_RELATIVE_TOLERANCE,
        "epsilon_float64": epsilon,
        "range_error_epsilon_factor": BRIDGE_RANGE_ERROR_EPSILON_FACTOR,
        "r0_error_epsilon_factor": BRIDGE_R0_ERROR_EPSILON_FACTOR,
        "phase_reduction_epsilon_factor": BRIDGE_PHASE_REDUCTION_EPSILON_FACTOR,
        "exp_error_epsilon_factor": BRIDGE_EXP_ERROR_EPSILON_FACTOR,
        "coefficient_l1": coefficient_l1,
        "max_coordinate_scale_m": max_coordinate_scale,
        "max_frequency_hz": max_frequency_hz,
        "max_abs_differential_range_m": max_abs_differential_range_m,
        "max_phase_argument_rad": max_phase_argument_rad,
        "max_arithmetic_l2_scale_context": max_arithmetic_l2_scale,
        "max_relative_l2": max_relative_l2,
        "max_scaled_max_absolute": max_scaled_max_absolute,
        "passed": bool(all(item["passed"] for item in per_record)),
        "per_record": per_record,
    }


def _prediction_error(actual: Sequence[Any], expected: Sequence[Any]) -> float:
    if len(actual) != len(expected):
        raise AssertionError("ragged prediction lengths differ")
    return max((_relative_error(a, b) for a, b in zip(actual, expected)), default=0.0)


def _energy(values: Sequence[Any]) -> float:
    return float(sum(float(np.vdot(np.asarray(value), np.asarray(value)).real) for value in values))


def _torch_arrays(values: Sequence[Any]) -> tuple[np.ndarray, ...]:
    return tuple(value.detach().cpu().numpy() for value in values)


CHECKPOINT_SCHEMA = "rift_gotcha_step3_camry_native_complex_measured_checkpoint_v1"
CHECKPOINT_KEYS = frozenset(
    {
        "schema",
        "target_id",
        "transform_name",
        "global_gain_real",
        "global_gain_imag",
        "pass_id",
        "polarization",
        "sector_id",
        "pulse_index",
        "frequency_counts",
        "support_local_m",
        "support_native_m",
        "coefficients_real",
        "coefficients_imag",
    }
)


def _checkpoint_scalar_text(loaded: Any, name: str) -> str:
    value = np.asarray(loaded[name])
    if value.ndim != 0 or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"checkpoint {name} must be a primitive string scalar")
    return str(value.item())


def validate_checkpoint_state(
    loaded: Any,
    *,
    selected_ids: Sequence[Any],
    frequency_counts: Sequence[int],
    support_local: np.ndarray,
    support_native: np.ndarray,
    point_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate every primitive checkpoint field before reloaded prediction."""

    if set(loaded.files) != CHECKPOINT_KEYS:
        raise ValueError(
            "checkpoint key set mismatch: "
            f"expected={sorted(CHECKPOINT_KEYS)} actual={sorted(loaded.files)}"
        )
    if _checkpoint_scalar_text(loaded, "schema") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema changed")
    if _checkpoint_scalar_text(loaded, "target_id") != "toyota_camry":
        raise ValueError("checkpoint target binding changed")
    if _checkpoint_scalar_text(loaded, "transform_name") != "identity_native_complex_measurement_transform":
        raise ValueError("checkpoint transform binding changed")
    gain_real = np.asarray(loaded["global_gain_real"])
    gain_imag = np.asarray(loaded["global_gain_imag"])
    if gain_real.dtype != np.dtype(np.float64) or gain_imag.dtype != np.dtype(np.float64) or gain_real.ndim != 0 or gain_imag.ndim != 0:
        raise ValueError("checkpoint global gain must be float64 scalar values")
    if float(gain_real) != 1.0 or float(gain_imag) != 0.0:
        raise ValueError("checkpoint global complex gain is not the frozen 1+0j gain")

    expected_keys = np.asarray([_identity_key(identity) for identity in selected_ids], dtype=object)
    expected_pass = np.asarray([key[0] for key in expected_keys], dtype=np.int64)
    expected_polarization = np.asarray([key[1] for key in expected_keys], dtype="U2")
    expected_sector = np.asarray([key[2] for key in expected_keys], dtype=np.int64)
    expected_pulse = np.asarray([key[3] for key in expected_keys], dtype=np.int64)
    for name, expected in (
        ("pass_id", expected_pass),
        ("polarization", expected_polarization),
        ("sector_id", expected_sector),
        ("pulse_index", expected_pulse),
        ("frequency_counts", np.asarray(tuple(int(value) for value in frequency_counts), dtype=np.int64)),
    ):
        actual = np.asarray(loaded[name])
        if actual.dtype != expected.dtype or actual.shape != expected.shape or not np.array_equal(actual, expected):
            raise ValueError(f"checkpoint {name} does not bind the selected canonical IDs/counts")

    def exact_float_array(name: str, expected: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        actual = np.asarray(loaded[name])
        if actual.dtype != np.dtype(np.float64) or actual.shape != shape or not np.isfinite(actual).all():
            raise ValueError(f"checkpoint {name} must be finite float64 with shape {shape}")
        if not np.array_equal(actual, expected):
            raise ValueError(f"checkpoint {name} does not match the declared diagnostic state")
        return actual

    exact_float_array("support_local_m", np.asarray(support_local, dtype=np.float64), tuple(support_local.shape))
    reloaded_native = exact_float_array("support_native_m", np.asarray(support_native, dtype=np.float64), tuple(support_native.shape))
    real = np.asarray(loaded["coefficients_real"])
    imag = np.asarray(loaded["coefficients_imag"])
    expected_coeff_shape = (int(point_count),)
    for name, actual in (("coefficients_real", real), ("coefficients_imag", imag)):
        if actual.dtype != np.dtype(np.float64) or actual.shape != expected_coeff_shape or not np.isfinite(actual).all():
            raise ValueError(f"checkpoint {name} must be finite float64 with shape {expected_coeff_shape}")
    return reloaded_native, real, imag


def _safe_archive_path(archive_root: str | Path, protocol: Mapping[str, Any]) -> Path:
    root = Path(archive_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"archive root must be an existing regular directory: {root}")
    relative = Path(str(protocol["source"]["relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("protocol archive relative_path must remain inside archive_root")
    root_resolved = root.resolve()
    archive_path = (root_resolved / relative).resolve()
    try:
        archive_path.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError("resolved native archive escaped archive_root") from exc
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ValueError(f"native archive must be a fresh regular file under archive_root: {archive_path}")
    return archive_path


def run_measured(
    protocol_path: str | Path,
    archive_root: str | Path,
    output_dir: str | Path,
    stage: str = "measured_readiness",
) -> dict[str, Any]:
    protocol = load_measured_protocol(protocol_path)
    if stage != protocol["stage"]:
        raise ValueError(f"requested stage {stage!r} differs from protocol stage {protocol['stage']!r}")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"output directory must be fresh and not already exist: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"output parent must already exist: {output.parent}")
    if not V2.TORCH_AVAILABLE:
        raise RuntimeError(f"measured source-AF run is fail-closed without Torch: {V2._TORCH_IMPORT_ERROR}")

    archive_path = _safe_archive_path(archive_root, protocol)
    shard = ACQ.load_native_shard(
        archive_path,
        expected_pass_id=1,
        expected_polarization="hh",
        expected_scene_id="gotcha_v1_joint8_fullpol",
    )
    selected_ids = select_train_sector002(shard, protocol)
    loaded_roles = tuple(str(value).lower() for value in np.asarray(shard.role).tolist())
    loaded_role_counts = {role: int(loaded_roles.count(role)) for role in ("train", "validation")}
    native_frequencies = np.asarray(shard.frequencies_hz)
    if native_frequencies.size != int(protocol["scope"]["expected_native_frequency_count_per_pulse"]):
        raise ValueError("loaded archive native frequency vector differs from measured 424-frequency contract")
    if not np.isfinite(native_frequencies).all() or (native_frequencies.size > 1 and not np.all(np.diff(native_frequencies) > 0.0)):
        raise ValueError("loaded native frequency vector is not finite and strictly increasing")
    frequency_counts = tuple(int(native_frequencies.size) for _ in selected_ids)
    ledger = resource_ledger(
        len(selected_ids),
        frequency_counts,
        int(protocol["camry"]["support_count"]),
        int(protocol["optimizer"]["max_iterations"]),
        int(protocol["optimizer"]["max_kernel_evaluations"]),
    )
    # The count/cap preflight above precedes observation materialization and all
    # source-AF/Torch kernel work.
    observations = tuple(shard.observations(selected_ids))
    validate_selected_records(
        observations,
        selected_ids,
        expected_frequencies=native_frequencies,
        protocol=protocol,
    )
    scope = SOURCE_AF.SourceAFScope(
        pass_id=1,
        polarization="hh",
        sector_ids=(2,),
        role="train",
        expected_count=len(selected_ids),
        name="camry_measured_sector002_train",
    )
    source_records = SOURCE_AF.build_source_af(observations, scope=scope, expected_count=len(selected_ids))
    if tuple(record.identity for record in source_records) != selected_ids:
        raise ValueError("source-AF conversion changed selected identity order")
    for record in source_records:
        if np.asarray(record.frequencies_hz).dtype != np.dtype(np.float64):
            raise AssertionError("source-AF boundary did not promote frequencies to float64")
        if np.asarray(record.position_xyz_m).dtype != np.dtype(np.float64):
            raise AssertionError("source-AF boundary did not promote geometry to float64")
        if np.asarray(record.response_raw).dtype != np.dtype(np.complex128):
            raise AssertionError("source-AF boundary did not promote raw response to complex128")
        if np.asarray(record.effective_response).dtype != np.dtype(np.complex128):
            raise AssertionError("source-AF effective response is not complex128")
        if not isinstance(record.effective_r0_m, float) or not np.isfinite(record.effective_r0_m):
            raise AssertionError("source-AF effective r0 is not finite float64")
    if any(not np.isfinite(record.effective_response).all() or not np.isfinite(record.effective_r0_m) for record in source_records):
        raise ValueError("source-AF effective r0/response contains non-finite values")

    support_local = np.asarray(protocol["camry"]["support_local_points_m"], dtype=np.float64)
    support_native = V2.transform_smoke_support("toyota_camry", support_local)
    transform = V2.ComplexFrequencyTransform.identity(selected_ids, frequency_counts)
    if protocol["native_contract"]["data_prediction_application"] != "same exact complex T before residual, loss, and energy":
        raise AssertionError("measured protocol does not declare common data/prediction transform routing")
    if transform.name != "identity_native_complex_measurement_transform":
        raise AssertionError("measured readiness requires the declared identity transform")
    source = V2.SourceAFNativeComplexTorchOperator(
        source_records,
        transform=transform,
        max_kernel_evaluations=int(protocol["optimizer"]["max_kernel_evaluations"]),
    )
    binding = V2.MeasuredCamryReadinessBinding(
        archive_path=archive_path,
        selected_ids=selected_ids,
        expected_count=len(selected_ids),
    )
    probe_pairs = np.asarray(protocol["camry"]["bridge_probe_coefficients_real_imag"], dtype=np.float64)
    bridge_coefficients = np.asarray(probe_pairs[:, 0] + 1j * probe_pairs[:, 1], dtype=np.complex128)
    bridge_numpy = source.forward_numpy(support_native, bridge_coefficients)
    import torch

    bridge_torch = source.forward_torch(
        support_native,
        torch.as_tensor(bridge_coefficients.real, dtype=torch.float64),
        torch.as_tensor(bridge_coefficients.imag, dtype=torch.float64),
    )
    bridge_expected = transform.apply_values(bridge_numpy.values, ids=selected_ids)
    bridge_report = bridge_parity_report(
        bridge_torch,
        bridge_expected,
        points_xyz_m=support_native,
        records=source_records,
        coefficients=bridge_coefficients,
    )
    if not bridge_report["passed"]:
        raise AssertionError(
            "NumPy/Torch nonzero-probe bridge parity failed: "
            f"max_relative_l2={bridge_report['max_relative_l2']:.16g} "
            f"max_scaled_max_absolute={bridge_report['max_scaled_max_absolute']:.16g}"
        )
    bridge_error = float(bridge_report["max_relative_l2"])

    fit = source.bounded_torch_ridge_fit(
        support_native,
        ridge=float(protocol["optimizer"]["ridge"]),
        max_iterations=int(protocol["optimizer"]["max_iterations"]),
        measured_binding=binding,
    )
    if fit["fit_materialization"] != "bounded_train_only_measured_camry_readiness_diagnostic":
        raise AssertionError("measured fit returned a synthetic materialization label")
    if int(fit["kernel_evaluation_estimate"]) != int(ledger["optimizer_kernel_evaluations"]):
        raise AssertionError("Torch fit kernel estimate differs from the measured ledger")

    objective_history = tuple(float(value) for value in fit["objective_history"])
    measurement_history = tuple(float(value) for value in fit["measurement_objective_history"])
    ridge_penalty_history = tuple(float(value) for value in fit["ridge_penalty_history"])
    gradient_norm_history = tuple(float(value) for value in fit["gradient_norm_history"])
    if len(objective_history) != 3 or len(measurement_history) != 3 or len(ridge_penalty_history) != 3 or len(gradient_norm_history) != 2:
        raise AssertionError("measured two-update objective/gradient history has the wrong length")
    if not all(np.isfinite(value) for value in (*objective_history, *measurement_history, *ridge_penalty_history, *gradient_norm_history)):
        raise AssertionError("measured objective/gradient history is non-finite")
    if any(not np.isclose(objective, measurement + ridge, rtol=0.0, atol=1.0e-14) for objective, measurement, ridge in zip(objective_history, measurement_history, ridge_penalty_history)):
        raise AssertionError("measured objective history does not equal measurement plus ridge penalty")
    if any(objective_history[index + 1] > objective_history[index] + 1.0e-12 for index in range(2)):
        raise AssertionError("measured readiness objective increased")
    if measurement_history[0] <= 0.0:
        raise AssertionError("measured zero-reference objective must be positive")
    measurement_relmse_vs_zero = tuple(float(value / measurement_history[0]) for value in measurement_history)
    if not np.isclose(measurement_relmse_vs_zero[0], 1.0, rtol=0.0, atol=1.0e-14):
        raise AssertionError("initial measured native-complex RelMSE versus zero is not one")

    final_real = np.asarray(fit["coefficients_real"], dtype=np.float64)
    final_imag = np.asarray(fit["coefficients_imag"], dtype=np.float64)
    prediction_before = source.forward_torch(support_native, final_real, final_imag)
    checkpoint_path = output / str(protocol["outputs"]["ridge_checkpoint"])
    output.mkdir(parents=False, exist_ok=False)
    np.savez(
        checkpoint_path,
        schema=np.asarray(CHECKPOINT_SCHEMA),
        target_id=np.asarray("toyota_camry"),
        transform_name=np.asarray(transform.name),
        global_gain_real=np.asarray(1.0, dtype=np.float64),
        global_gain_imag=np.asarray(0.0, dtype=np.float64),
        pass_id=np.asarray([int(_identity_key(identity)[0]) for identity in selected_ids], dtype=np.int64),
        polarization=np.asarray([_identity_key(identity)[1] for identity in selected_ids], dtype="U2"),
        sector_id=np.asarray([int(_identity_key(identity)[2]) for identity in selected_ids], dtype=np.int64),
        pulse_index=np.asarray([int(_identity_key(identity)[3]) for identity in selected_ids], dtype=np.int64),
        frequency_counts=np.asarray(frequency_counts, dtype=np.int64),
        support_local_m=np.asarray(support_local, dtype=np.float64),
        support_native_m=np.asarray(support_native, dtype=np.float64),
        coefficients_real=final_real,
        coefficients_imag=final_imag,
    )
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        reloaded_support_native, reload_real, reload_imag = validate_checkpoint_state(
            checkpoint,
            selected_ids=selected_ids,
            frequency_counts=frequency_counts,
            support_local=support_local,
            support_native=support_native,
            point_count=support_local.shape[0],
        )
    prediction_after = source.forward_torch(reloaded_support_native, reload_real, reload_imag)
    reload_error = _prediction_error(prediction_after, prediction_before)
    if reload_error > 1.0e-12:
        raise AssertionError(f"save/reload prediction parity failed: {reload_error}")

    observed_raw_values = tuple(record.effective_response for record in source_records)
    observed_values = transform.apply_values(observed_raw_values, ids=selected_ids)
    predicted_values = _torch_arrays(prediction_before)
    observed_energy = _energy(observed_values)
    predicted_energy = _energy(predicted_values)
    native_samples = int(ledger["native_frequency_samples"])
    selection_report = {
        "schema": "rift_gotcha_step3_camry_native_complex_measured_selection_v1",
        "archive_path": str(archive_path),
        "shard_id": shard.shard_id,
        "pass_id": int(shard.pass_id),
        "polarization": str(shard.polarization),
        "loaded_role_counts": loaded_role_counts,
        "materialized_role_counts": {"train": len(observations)},
        "used_role_counts": {"train": len(source_records)},
        "selected_canonical_ids": [_identity_dict(identity) for identity in selected_ids],
        "selected_pulse_count": len(selected_ids),
        "native_frequency_vector_hz": native_frequencies,
        "native_frequency_count_per_pulse": int(native_frequencies.size),
        "native_frequency_counts": list(frequency_counts),
        "native_sample_count": native_samples,
        "validation_selection": False,
        "validation_records_materialized": False,
        "validation_response_payload_materialized": True,
        "test_payload_opened": False,
        "test_payload_materialized": False,
        "selection_identity_source": "actual_loaded_archive_records",
    }
    resource_report = {
        "schema": "rift_gotcha_step3_camry_native_complex_measured_resource_v1",
        "ledger": ledger,
        "fit_kernel_evaluation_estimate": int(fit["kernel_evaluation_estimate"]),
        "fit_max_kernel_evaluations": int(fit["max_kernel_evaluations"]),
        "fit_kernel_guard": fit["kernel_guard"],
        "cap_rejected_before_expensive_calculation": False,
        "all_declared_passes": {
            "numpy_torch_bridge": int(ledger["numpy_torch_bridge_kernel_evaluations"]),
            "optimizer_trajectory": int(ledger["optimizer_kernel_evaluations"]),
            "final_reloaded_prediction_parity": int(ledger["final_reloaded_prediction_parity_kernel_evaluations"]),
        },
    }
    native_report = {
        "schema": DRIVER_SCHEMA,
        "status": "PASS",
        "stage": stage,
        "run_kind": protocol["run_kind"],
        "target": "toyota_camry",
        "source_af_representation": {
            "formula": SOURCE_AF.SOURCE_AF_FORMULA,
            "effective_r0_finite": True,
            "effective_response_finite": True,
            "autofocus_application": "none; raw channel-owned corrections converted once to downstream source-AF representation",
        },
        "camry": {
            "placement": {
                "equation": "p_native = R @ p_local + t",
                "R": V2.CAMRY_VEHICLE_AXIS_ROTATION,
                "t_m": V2.CAMRY_TRANSLATION_M,
                "qualification": protocol["camry"]["placement_qualification"],
                "physical_registration_claim": False,
                "footprint_consistency": V2.camry_footprint_consistency(),
            },
            "support_local_m": support_local,
            "support_native_m": support_native,
            "support_count": int(support_local.shape[0]),
            "support_semantics": "direct_complex_point_weights",
            "reference_grid_quadrature": False,
            "global_complex_gain": "1+0j_frozen",
            "bridge_probe_coefficients_real_imag": probe_pairs,
        },
        "native_contract": {
            "frequency_policy": "native_stored_exact",
            "phase_hypothesis": "demanet_2012_gotcha_phase_candidate",
            "phase_status": "named_candidate_unverified_for_real_archive",
            "identity_transform": "T=identity applied identically to data and prediction",
            "data_prediction_application": "NumPy raw prediction passed through transform.apply_values; Torch prediction passed through transform.apply_torch",
            "observed_energy_source": "transform.apply_values(effective source-AF response) before energy calculation",
            "exterior_native_residual_retained": True,
            "target_crop_subtraction_padding": False,
        "validation_selection": False,
        "validation_response_payload_materialized": True,
        "test_payload_opened": False,
        },
        "global_complex_gain": {"value": "1+0j", "trainable": False},
        "native_complex_metrics": {
            "initial_native_complex_measurement_objective": measurement_history[0],
            "final_native_complex_measurement_objective": measurement_history[-1],
            "relmse_vs_zero_history": measurement_relmse_vs_zero,
            "initial_relmse_vs_zero": measurement_relmse_vs_zero[0],
            "final_relmse_vs_zero": measurement_relmse_vs_zero[-1],
            "objective_history_including_ridge": objective_history,
            "ridge_penalty_history": ridge_penalty_history,
            "objective_equals_measurement_plus_ridge": True,
            "gradient_norm_history": gradient_norm_history,
            "objective_monotone_nonincreasing": True,
            "observed_energy_sum_abs2": observed_energy,
            "predicted_energy_sum_abs2": predicted_energy,
            "observed_energy_mean_per_native_sample": observed_energy / native_samples,
            "predicted_energy_mean_per_native_sample": predicted_energy / native_samples,
            "numpy_torch_bridge_prediction_relative_error": bridge_error,
            "numpy_torch_bridge_parity": bridge_report,
            "save_reload_prediction_relative_error": reload_error,
        },
        "optimizer": {
            "initialization": "zero real/imag coefficients",
            "ridge": float(protocol["optimizer"]["ridge"]),
            "max_iterations": int(protocol["optimizer"]["max_iterations"]),
            "fit_materialization": fit["fit_materialization"],
            "lipschitz_upper_bound": float(fit["lipschitz_upper_bound"]),
            "step_size": float(fit["step_size"]),
            "step_rule": fit["step_rule"],
            "dtype": fit["dtype"],
        },
        "resource_ledger": resource_report,
        "checkpoint": {
            "path": str(checkpoint_path),
            "schema": CHECKPOINT_SCHEMA,
            "state_fields_validated": sorted(CHECKPOINT_KEYS),
            "save_reload_prediction_parity": "PASS",
            "checkpoint_reload_forward_calls": 2,
        },
        "release_disclosure": {
            "decreasing_loss_is_not_recovery": True,
            "not_target_recovery": True,
            "not_production_fit": True,
            "not_isolated_target_claim": True,
            "not_physical_registration_claim": True,
            "not_accuracy_focus_geometry_evidence": True,
            "four_point_support_inadequate_for_full_vehicle": True,
        },
    }
    status_report = {
        "schema": "rift_gotcha_step3_camry_native_complex_measured_status_v1",
        "status": "PASS",
        "stage": stage,
        "target": "toyota_camry",
        "measured_fit_release": False,
        "test_payload_opened": False,
        "validation_selection": False,
        "validation_response_payload_materialized": True,
        "readiness_statement": "bounded actual TRAIN-only Camry source-AF native-complex optimizer diagnostic; not recovery or production fitting",
        "files": list(EXPECTED_OUTPUTS.values()),
    }
    _write_json(output / EXPECTED_OUTPUTS["protocol_echo"], protocol)
    _write_json(output / EXPECTED_OUTPUTS["selection"], selection_report)
    _write_json(output / EXPECTED_OUTPUTS["resource"], resource_report)
    _write_json(output / EXPECTED_OUTPUTS["native_complex_report"], native_report)
    _write_json(output / EXPECTED_OUTPUTS["status"], status_report)
    return native_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", default="measured_readiness")
    args = parser.parse_args(argv)
    try:
        report = run_measured(args.protocol, args.archive_root, args.output_dir, args.stage)
    except Exception as exc:
        print(json.dumps({"schema": DRIVER_SCHEMA, "status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps({"schema": DRIVER_SCHEMA, "status": report["status"], "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
