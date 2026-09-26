"""Run the data-free validator for the measured Camry readiness package.

This validator intentionally does not accept an archive argument.  It checks the
measured contract, geometry, exact resource ledger, identity transform, and the
existing Torch bridge on tiny synthetic records.  It exits with status 2 when
Torch is unavailable, so a missing numerical dependency cannot look like a pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from run_gotcha_step3_camry_native_complex_measured_v1 import (
    DRIVER_SCHEMA,
    PROJECT_ROOT,
    V2,
    bridge_parity_report,
    load_measured_protocol,
    resource_ledger,
)


PROTOCOL_PATH = PROJECT_ROOT / "protocols" / "gotcha_step3_camry_native_complex_measured_v1.json"
VALIDATOR_SCHEMA = "rift_gotcha_step3_camry_native_complex_measured_validator_v1"


def _relative_error(actual: Any, expected: Any) -> float:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    return float(np.linalg.norm(actual_array - expected_array) / max(1.0, float(np.linalg.norm(expected_array))))


def run(protocol_path: str | Path = PROTOCOL_PATH) -> dict[str, Any]:
    protocol = load_measured_protocol(protocol_path)
    if not V2.TORCH_AVAILABLE:
        return {
            "schema": VALIDATOR_SCHEMA,
            "status": "BLOCKED_TORCH_UNAVAILABLE",
            "protocol_schema": protocol["schema"],
            "data_free": True,
            "archive_accessed": False,
            "torch_error": str(V2._TORCH_IMPORT_ERROR),
        }

    import torch

    support_local = np.asarray(protocol["camry"]["support_local_points_m"], dtype=np.float64)
    support_native = V2.transform_smoke_support("toyota_camry", support_local)
    footprint = V2.camry_footprint_consistency()
    if not footprint["all_corners_within_absolute_tolerance"] or not footprint["inverse_footprint_contained_in_camry_grid"]:
        raise AssertionError("Camry footprint geometry check failed")
    ledger = resource_ledger(117, (424,) * 117, 4, 2, 2_000_000)
    expected_ledger = {
        "native_frequency_samples": 49_608,
        "K_point_times_native_samples": 198_432,
        "numpy_torch_bridge_kernel_evaluations": 396_864,
        "optimizer_kernel_evaluations": 992_160,
        "final_reloaded_prediction_parity_kernel_evaluations": 396_864,
        "total_kernel_evaluations": 1_785_888,
    }
    if any(ledger[key] != value for key, value in expected_ledger.items()):
        raise AssertionError(f"measured resource ledger changed: {ledger}")

    records = V2.source_af_records_from_synthetic()
    source = V2.SourceAFNativeComplexTorchOperator(records, max_kernel_evaluations=2_000_000)
    probe_pairs = np.asarray(protocol["camry"]["bridge_probe_coefficients_real_imag"], dtype=np.float64)
    probe = np.asarray(probe_pairs[:, 0] + 1j * probe_pairs[:, 1], dtype=np.complex128)
    numpy_prediction = source.forward_numpy(support_native, probe)
    torch_prediction = source.forward_torch(
        support_native,
        torch.as_tensor(probe.real, dtype=torch.float64),
        torch.as_tensor(probe.imag, dtype=torch.float64),
    )
    bridge_report = bridge_parity_report(
        tuple(actual.detach().cpu().numpy() for actual in torch_prediction),
        numpy_prediction.values,
        points_xyz_m=support_native,
        records=records,
        coefficients=probe,
    )
    if not bridge_report["passed"]:
        raise AssertionError(
            "synthetic NumPy/Torch bridge parity failed: "
            f"max_relative_l2={bridge_report['max_relative_l2']:.16g} "
            f"max_scaled_max_absolute={bridge_report['max_scaled_max_absolute']:.16g}"
        )
    fit = source.bounded_torch_ridge_fit(support_native, ridge=1.0e-3, max_iterations=2)
    if fit["fit_materialization"] != "bounded_synthetic_source_af_adapter_only":
        raise AssertionError("default synthetic fit was silently relabeled")
    if len(fit["objective_history"]) != 3 or len(fit["measurement_objective_history"]) != 3:
        raise AssertionError("synthetic objective history length changed")
    if any(not np.isfinite(value) for value in fit["objective_history"]):
        raise AssertionError("synthetic objective history is non-finite")
    false_binding = V2.MeasuredCamryReadinessBinding(
        archive_path=Path(__file__),
        selected_ids=tuple(record.identity for record in records),
        expected_count=len(records),
    )
    try:
        source.bounded_torch_ridge_fit(
            support_native,
            ridge=1.0e-3,
            max_iterations=2,
            measured_binding=false_binding,
        )
        fit_with_false_label = True
    except ValueError:
        fit_with_false_label = None
    if fit_with_false_label is not None:
        raise AssertionError("synthetic source-AF records were silently relabeled as measured")

    return {
        "schema": VALIDATOR_SCHEMA,
        "status": "PASS",
        "protocol_schema": protocol["schema"],
        "data_free": True,
        "archive_accessed": False,
        "geometry": {
            "camry_footprint_all_corners_within_tolerance": footprint["all_corners_within_absolute_tolerance"],
            "camry_inverse_footprint_contained": footprint["inverse_footprint_contained_in_camry_grid"],
            "support_local_shape": list(support_local.shape),
            "support_native_shape": list(support_native.shape),
        },
        "identity_transform": "T=identity; data and prediction use the same exact path",
        "numpy_torch_bridge_relative_error": bridge_report["max_relative_l2"],
        "numpy_torch_bridge_parity": bridge_report,
        "bridge_probe_coefficients_real_imag": probe_pairs,
        "synthetic_fit_materialization": fit["fit_materialization"],
        "synthetic_fit_objective_history": list(fit["objective_history"]),
        "measured_relabel_rejection": True,
        "resource_ledger": ledger,
        "release_disclosure": {
            "not_recovery": True,
            "not_production_fit": True,
            "not_isolated_target_claim": True,
            "not_physical_registration_claim": True,
            "archive_accessed": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    args = parser.parse_args()
    try:
        report = run(args.protocol)
    except Exception as exc:
        print(json.dumps({"schema": VALIDATOR_SCHEMA, "status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(report, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else str(value), sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
