"""Data-free synthetic validator for the GOTCHA Step-2 controls."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ACQ = _load("gotcha_acquisition_for_step2_validator", ROOT / "rift" / "gotcha_acquisition.py")
CTL = _load("gotcha_step2_controls_validator", ROOT / "rift" / "gotcha_step2_controls.py")


def _ro(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _observation(pass_id, polarization, sector, pulse, position, r0, frequencies, response, role):
    autofocus = ACQ.AutofocusProvenance(
        schema=ACQ.AUTOFOCUS_PROVENANCE_SCHEMA,
        mode=ACQ.AUTOFOCUS_OFFICIAL_ABSENT,
        official_available=False,
        applied=False,
        source_shard_id=f"pass{pass_id}_{polarization}",
        range_field=None,
        phase_field=None,
    )
    return ACQ.NativeObservation(
        identity=ACQ.NativeObservationId(pass_id, polarization, sector, pulse),
        role=role,
        response=_ro(response, np.complex128),
        frequencies_hz=_ro(frequencies, np.float64),
        position_xyz_m=_ro(position, np.float64),
        r0_m=float(r0),
        th_deg=0.0,
        phi_deg=0.0,
        r_correct_raw=None,
        ph_correct_raw=None,
        phase_reference=ACQ.PhaseReferenceContract(),
        autofocus=autofocus,
    )


def make_observations():
    points = np.asarray([[0.37, -1.21, 0.70], [-2.40, 1.90, 4.07]], dtype=np.float64)
    amplitudes = {
        "hv": np.asarray([1.0 + 0.3j, 0.2 - 0.1j], dtype=np.complex128),
        "vh": np.asarray([0.7 - 0.2j, -0.3 + 0.4j], dtype=np.complex128),
    }
    rows = (
        (1, "hv", 11, 101, [100.0, -20.0, 15.0], 102.7, [9.10e9, 9.17e9, 9.31e9], "train"),
        (1, "vh", 12, 102, [80.0, 25.0, 16.0], 84.9, [9.10e9, 9.19e9, 9.33e9, 9.40e9], "train"),
        (2, "hv", 13, 203, [-90.0, 15.0, 19.0], 92.8, [9.10e9, 9.16e9, 9.29e9, 9.43e9, 9.51e9], "train"),
        (2, "vh", 14, 204, [70.0, -30.0, 18.0], 77.1, [9.10e9, 9.22e9, 9.35e9], "train"),
        (3, "hv", 15, 305, [-65.0, -35.0, 21.0], 74.6, [9.10e9, 9.18e9, 9.37e9, 9.49e9], "validation"),
        (3, "vh", 16, 306, [55.0, 35.0, 23.0], 65.8, [9.10e9, 9.24e9, 9.39e9, 9.55e9, 9.63e9], "validation"),
    )
    observations = []
    for row in rows:
        pass_id, polarization, sector, pulse, position, r0, frequencies, role = row
        position = np.asarray(position, dtype=np.float64)
        frequencies = np.asarray(frequencies, dtype=np.float64)
        response = ACQ.direct_point_target_render(
            points,
            amplitudes[polarization],
            tx_positions_m=position[None, :],
            rx_positions_m=position[None, :],
            frequencies_hz=frequencies,
            reference_range_m=[r0],
            convention=ACQ.PUBLISHED_GOTCHA_REFERENCE_CANDIDATE,
        )[0]
        observations.append(_observation(pass_id, polarization, sector, pulse, position, r0, frequencies, response, role))
    return points, amplitudes, tuple(observations)


def make_shard(pass_id: int, polarization: str, *, include_test: bool = False):
    train_sectors = np.asarray([2, 3, 4, 7, 8, 9], dtype=np.int64)
    sectors = np.concatenate((train_sectors, [6])) if include_test else train_sectors
    count = sectors.size
    autofocus = ACQ.AutofocusProvenance(
        schema=ACQ.AUTOFOCUS_PROVENANCE_SCHEMA,
        mode=ACQ.AUTOFOCUS_OFFICIAL_ABSENT,
        official_available=False,
        applied=False,
        source_shard_id=f"pass{pass_id}_{polarization}",
        range_field=None,
        phase_field=None,
    )
    return ACQ.NativeShard(
        path=Path(f"synthetic_{pass_id}_{polarization}.npz"),
        shard_id=f"pass{pass_id}_{polarization}",
        pass_id=pass_id,
        polarization=polarization,
        response=_ro(np.zeros((count, 3), dtype=np.complex128)),
        frequencies_hz=_ro([9.1e9, 9.2e9, 9.3e9], np.float64),
        x=_ro(np.full(count, 100.0), np.float64),
        y=_ro(np.arange(count), np.float64),
        z=_ro(np.full(count, 15.0), np.float64),
        r0=_ro(np.full(count, 100.0), np.float64),
        th=_ro(np.zeros(count), np.float64),
        phi=_ro(np.zeros(count), np.float64),
        sector_id=_ro(sectors, np.int64),
        pulse_index=_ro(np.arange(100, 100 + count), np.int64),
        role=_ro(np.asarray(["train"] * 6 + (["test"] if include_test else []), dtype="U10")),
        r_correct_raw=_ro([]),
        ph_correct_raw=_ro([]),
        metadata={},
        phase_reference=ACQ.PhaseReferenceContract(),
        autofocus=autofocus,
    )


def _support_declaration(max_kernel_evaluations=10000):
    return {
        "schema": CTL.H0_SUPPORT_SCHEMA,
        "bounds_m": {"x": [-3.0, 3.0], "y": [-3.0, 3.0], "z": [0.0, 4.0]},
        "sampling": {"spacing_m": [1.0, 1.0, 1.0], "shape": [7, 7, 5]},
        "max_kernel_evaluations": max_kernel_evaluations,
        "support_mode": "height_capable_volume",
        "frame_contract": "antenna_xyz_unchanged",
        "registration_status": "unresolved",
        "support_status": CTL.SUPPORT_STATUS,
        "hypothesis_label": "conditional_z_volume_not_physical_registration",
        "units": {"x": "m", "y": "m", "z": "m", "r0": "m", "frequency": "Hz"},
        "phase_hypothesis": CTL.PHASE_HYPOTHESIS_NAME,
        "autofocus_status": "raw_unapplied_channel_owned",
    }


def main() -> int:
    points, amplitudes, all_observations = make_observations()
    observations = all_observations[0::2]
    cross_pol_observations = all_observations[1::2]
    operator = CTL.NativeRaggedOperator(observations, point_chunk_size=1)
    coefficients = np.asarray([0.6 - 0.2j, -0.3 + 0.8j], dtype=np.complex128)
    predicted = operator.forward(points, coefficients)
    direct_oracle = []
    for observation in observations:
        direct_oracle.append(
            ACQ.direct_point_target_render(
                points,
                coefficients,
                tx_positions_m=observation.position_xyz_m[None, :],
                rx_positions_m=observation.position_xyz_m[None, :],
                frequencies_hz=observation.frequencies_hz,
                reference_range_m=[observation.r0_m],
                convention=ACQ.PUBLISHED_GOTCHA_REFERENCE_CANDIDATE,
            )[0]
        )
    direct_error = max(float(np.max(np.abs(a - b))) for a, b in zip(predicted.values, direct_oracle))
    dot_report = operator.dot_test(
        points,
        coefficients,
        tuple(np.full(value.size, 0.4 + 0.2j, dtype=np.complex128) for value in predicted.values),
    )

    basis = np.eye(points.shape[0], dtype=np.complex128)
    fit_observations = observations[:2]
    fit_operator = CTL.NativeRaggedOperator(fit_observations)
    dense_matrix = np.column_stack([np.concatenate(fit_operator.forward(points, column).values) for column in basis.T])
    truth = np.asarray([0.8 - 0.2j, -0.4 + 0.5j], dtype=np.complex128)
    target = fit_operator.forward(points, truth)
    lam = 2e-2
    cgls = CTL.ridge_cgls(fit_operator, points, target, lambda_ridge=lam, max_iterations=30, rtol=1e-11)
    dense_oracle = np.linalg.solve(
        dense_matrix.conj().T @ dense_matrix + lam * np.eye(points.shape[0]),
        dense_matrix.conj().T @ np.concatenate(target.values),
    )
    cgls_error = float(np.max(np.abs(cgls.coefficients - dense_oracle)))
    if direct_error > 2e-12 or dot_report["relative_error"] > 2e-12 or cgls_error > 1e-8:
        raise AssertionError("direct, adjoint, or ridge-CGLS oracle check failed")
    low_result = CTL.ridge_cgls(
        fit_operator,
        points,
        fit_operator.forward(points, truth * 1.0e-12),
        lambda_ridge=lam,
        max_iterations=5,
        rtol=1e-6,
    )
    if low_result.iterations[-1]["iteration"] == 0:
        raise AssertionError("relative CGLS stopping rule used an unintended absolute floor")
    validation_fit_operator = CTL.NativeRaggedOperator(observations)
    try:
        CTL.ridge_cgls(validation_fit_operator, points, validation_fit_operator.data(), max_iterations=2)
    except ValueError:
        validation_rejection = True
    else:
        validation_rejection = False
    if not validation_rejection:
        raise AssertionError("validation observations were accepted by inverse fitting")

    wrong_height = operator.forward(points * np.asarray([1.0, 1.0, 0.0]), amplitudes["hv"])
    height_error = float(np.linalg.norm(np.concatenate([a - b for a, b in zip(predicted.values, wrong_height.values)])))
    scalar_observations = tuple(dataclasses.replace(observation, r0_m=float(np.linalg.norm(observation.position_xyz_m))) for observation in observations)
    scalar_values = CTL.NativeRaggedOperator(scalar_observations).forward(points, amplitudes["hv"])
    scalar_r0_error = float(np.linalg.norm(np.concatenate([a - b for a, b in zip(predicted.values, scalar_values.values)])))
    if height_error <= 1e-3 or scalar_r0_error <= 1e-3:
        raise AssertionError("height or per-pulse native r0 was silently removed")

    actual = CTL.NativePredictions(
        tuple(observation.identity for observation in observations),
        tuple(np.asarray(observation.response, dtype=np.complex128) for observation in observations),
    )
    half = CTL.NativePredictions(actual.ids, tuple(0.5 * value for value in actual.values))
    scale = CTL.fit_train_complex_scale(observations[:2], CTL.NativePredictions(half.ids[:2], half.values[:2]))
    metrics = CTL.evaluate_native_metrics(observations[2:], CTL.NativePredictions(half.ids[2:], half.values[2:]), scale=scale)
    if abs(scale.value - 2.0) > 1e-12 or metrics["scaled"]["relmse"] > 1e-24:
        raise AssertionError("train-only frozen complex scale check failed")
    try:
        CTL.fit_train_complex_scale(
            observations[:2],
            CTL.NativePredictions(actual.ids[:2], tuple(np.zeros_like(value) for value in actual.values[:2])),
        )
    except ValueError:
        zero_scale_rejection = True
    else:
        zero_scale_rejection = False
    if not zero_scale_rejection:
        raise AssertionError("zero-energy train scale was silently frozen")

    prefixes = CTL.select_train_prefixes(
        tuple(make_shard(pass_id, polarization) for pass_id in (1, 2) for polarization in ("hv", "vh")),
        counts=(1, 2, 4),
        seed=17,
    )
    prefix_report = CTL.validate_nested_prefixes(prefixes)
    try:
        CTL.select_train_prefixes(
            (make_shard(1, "hh", include_test=True),),
            counts=(1,),
            requested_passes=(1,),
            requested_polarizations=("hh",),
        )
    except ValueError as error:
        test_rejection = "sealed test" in str(error)
    else:
        test_rejection = False
    if not test_rejection:
        raise AssertionError("test rows were not rejected before selection")

    support_report = CTL.preflight_support_declaration(_support_declaration(), observations, point_count=100)
    if support_report["full_scene_support_claim"] or support_report["support_boundary_diagnostic_status"] != "conditional_box_diagnostic_not_support_proof":
        raise AssertionError("support preflight made an unconditional physical claim")
    try:
        CTL.conditional_backproject(
            _support_declaration(),
            observations,
            residuals=tuple(operator.data().values),
        )
    except TypeError:
        bare_residual_rejection = True
    else:
        bare_residual_rejection = False
    if not bare_residual_rejection:
        raise AssertionError("conditional backprojection accepted bare residuals")
    conditional_fit = CTL.conditional_ridge_cgls(
        _support_declaration(),
        fit_observations,
        fit_operator.data(),
        lambda_ridge=lam,
        max_iterations=1,
        rtol=1e-6,
    )
    try:
        CTL.conditional_ridge_cgls(
            _support_declaration(),
            fit_observations,
            tuple(fit_operator.data().values),
            lambda_ridge=lam,
            max_iterations=1,
            rtol=1e-6,
        )
    except TypeError:
        bare_target_rejection = True
    else:
        bare_target_rejection = False
    if not bare_target_rejection:
        raise AssertionError("conditional inverse accepted a bare target sequence")
    try:
        CTL.preflight_support_declaration(_support_declaration(10), observations, point_count=100)
    except RuntimeError:
        guard_rejection = True
    else:
        guard_rejection = False
    if not guard_rejection:
        raise AssertionError("kernel guard did not reject oversized support")
    try:
        CTL.NativeRaggedOperator(observations + cross_pol_observations)
    except ValueError:
        mixed_polarization_rejection = True
    else:
        mixed_polarization_rejection = False
    if not mixed_polarization_rejection:
        raise AssertionError("mixed-polarization operator was accepted")

    print(json.dumps({
        "schema": "rift_gotcha_step2_controls_validation_v1",
        "status": "PASS",
        "observations": len(observations),
        "native_frequency_counts": list(operator.frequency_counts),
        "direct_oracle_max_abs_error": direct_error,
        "dot_product_relative_error": dot_report["relative_error"],
        "cgls_dense_oracle_max_abs_error": cgls_error,
        "cgls_status": cgls.status,
        "cgls_iterations": len(cgls.iterations),
        "cgls_kernel_evaluations_estimate": cgls.kernel_evaluations_estimate,
        "height_sensitivity_norm": height_error,
        "native_r0_scalar_shortcut_rejection_norm": scalar_r0_error,
        "train_frozen_scale": {"real": float(scale.value.real), "imag": float(scale.value.imag)},
        "train_frozen_scale_polarization": scale.polarization,
        "scaled_validation_relmse": metrics["scaled"]["relmse"],
        "low_amplitude_cgls_iterations": len(low_result.iterations),
        "nested_prefixes": prefix_report,
        "test_rejection": test_rejection,
        "zero_scale_rejection": zero_scale_rejection,
        "mixed_polarization_rejection": mixed_polarization_rejection,
        "validation_inverse_rejection": validation_rejection,
        "bare_target_rejection": bare_target_rejection,
        "bare_residual_rejection": bare_residual_rejection,
        "conditional_fit_kernel_evaluations_estimate": conditional_fit.kernel_evaluations_estimate,
        "support_status": support_report["status"],
        "support_boundary_diagnostic_status": support_report["support_boundary_diagnostic_status"],
        "kernel_guard_rejection": guard_rejection,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"schema": "rift_gotcha_step2_controls_validation_v1", "status": "FAIL", "error": str(error)}, separators=(",", ":")))
        raise
