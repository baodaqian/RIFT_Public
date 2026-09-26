"""Run Batch-A GOTCHA acquisition/operator checks on a synthetic NPZ fixture.

The default invocation creates one temporary pass-2/HH Gate-1-shaped archive,
loads it through the sealed adapter, runs direct float64 synthetic checks, and
prints compact JSON.  It never reads raw GOTCHA files or designated test
payloads.  Real-data ``fp``/``r0`` and autofocus sign/order correctness remain
open contract questions in the report.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_ACQUISITION_PATH = PROJECT_ROOT / "rift" / "gotcha_acquisition.py"
_SPEC = importlib.util.spec_from_file_location("gotcha_acquisition_batch_a", _ACQUISITION_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load acquisition module from {_ACQUISITION_PATH}")
_ACQUISITION = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _ACQUISITION
_SPEC.loader.exec_module(_ACQUISITION)

AUTOFOCUS_RAW = _ACQUISITION.AUTOFOCUS_RAW
HeightSupportCandidate = _ACQUISITION.HeightSupportCandidate
PairedMonostaticGeometry = _ACQUISITION.PairedMonostaticGeometry
PUBLISHED_GOTCHA_REFERENCE_CANDIDATE = _ACQUISITION.PUBLISHED_GOTCHA_REFERENCE_CANDIDATE
SyntheticReferenceConvention = _ACQUISITION.SyntheticReferenceConvention
compare_native_and_endpoint_uniform_frequency = _ACQUISITION.compare_native_and_endpoint_uniform_frequency
forward_adjoint_inner_product_fixture = _ACQUISITION.forward_adjoint_inner_product_fixture
forward_vjp_finite_difference_fixture = _ACQUISITION.forward_vjp_finite_difference_fixture
load_native_shard = _ACQUISITION.load_native_shard
preflight_height_support_candidate = _ACQUISITION.preflight_height_support_candidate
virtual_reference_mapping_regression_fixture = _ACQUISITION.virtual_reference_mapping_regression_fixture


def _role_for_sector(sector_id: int) -> str:
    slot = (int(sector_id) - 1) % 10
    return "validation" if slot == 0 else "test" if slot == 5 else "train"


def _synthetic_archive(path: Path) -> None:
    sectors = [sector for sector in range(1, 361) if _role_for_sector(sector) != "test"]
    count = len(sectors)
    frequencies = np.asarray(
        [9_600_000_000.0, 9_601_000_000.0, 9_602_500_000.0, 9_604_000_000.0, 9_605_000_000.0],
        dtype=np.float32,
    )
    row = np.arange(count, dtype=np.float32)
    response = (row[:, None] + 1.0).astype(np.complex64) * np.exp(
        1j * np.asarray([0.0, 0.2, -0.4, 0.7, -0.1], dtype=np.float32)[None, :]
    )
    metadata = {
        "schema": "rift_gotcha_joint8_fullpol_native_shard_v1",
        "scene_id": "gotcha_v1_joint8_fullpol",
        "shard_id": "pass2_hh",
        "pass_id": 2,
        "polarization": "hh",
        "payload_sector_ids": sectors,
        "sealed_test_sector_ids": [sector for sector in range(1, 361) if _role_for_sector(sector) == "test"],
        "test_opened": False,
        "test_payload_included": False,
        "corrections_applied": False,
        "autofocus_unapplied": True,
        "layout": {
            "native_frequency_preserved": True,
            "resampled": False,
            "padded": False,
            "trimmed": False,
            "autofocus_unapplied": True,
        },
    }
    arrays = {
        "response": response,
        "frequencies_hz": frequencies,
        "x": (100.0 + row).astype(np.float32),
        "y": (-20.0 + 0.5 * row).astype(np.float32),
        "z": (1.0 + 0.01 * row).astype(np.float32),
        "r0": (120.0 + 0.1 * row).astype(np.float32),
        "th": np.linspace(-2.0, 2.0, count, dtype=np.float32),
        "phi": np.linspace(-1.0, 1.0, count, dtype=np.float32),
        "sector_id": np.asarray(sectors, dtype=np.int16),
        "pulse_index": np.asarray(sectors, dtype=np.int32) * 100 + 7,
        "pass_id": np.full(count, 2, dtype=np.int16),
        "polarization": np.full(count, "hh", dtype="U2"),
        "role": np.asarray([_role_for_sector(value) for value in sectors], dtype="U10"),
        "r_correct_raw": np.linspace(0.01, 0.02, count, dtype=np.float32),
        "ph_correct_raw": np.linspace(-0.2, 0.2, count, dtype=np.float32),
        "autofocus_available": np.asarray(True),
        "autofocus_applied": np.asarray(False),
        "autofocus_state": np.asarray(AUTOFOCUS_RAW, dtype="U36"),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True), dtype="U"),
    }
    with path.open("wb") as handle:
        np.savez(handle, **arrays)


def run_validation() -> dict[str, object]:
    fd, name = tempfile.mkstemp(
        prefix=".gotcha-acquisition-fixture-", suffix=".npz"
    )
    os.close(fd)
    archive = Path(name)
    try:
        _synthetic_archive(archive)
        shard = load_native_shard(archive, expected_pass_id=2, expected_polarization="hh")
        ids = shard.observation_ids[:3]
        geometry = shard.paired_monostatic_geometry(ids)
        points = np.asarray([[0.5, -1.2, 2.3], [-4.0, 2.0, 5.0]], dtype=np.float64)
        amplitudes = np.asarray([1.0 + 0.5j, -0.2 + 0.9j], dtype=np.complex128)
        frequencies = shard.frequencies_hz
        residual = np.asarray(
            [[0.3 + 0.8j, -0.7 + 0.2j, 0.4 - 0.1j, 0.6 + 0.3j, -0.1 + 0.5j]] * 3,
            dtype=np.complex128,
        )
        convention = SyntheticReferenceConvention()
        adjoint = forward_adjoint_inner_product_fixture(
            points,
            amplitudes,
            residual,
            tx_positions_m=geometry.tx_xyz_m,
            rx_positions_m=geometry.rx_xyz_m,
            frequencies_hz=frequencies,
            reference_range_m=np.full(3, 200.0),
            convention=convention,
        )
        vjp = forward_vjp_finite_difference_fixture(
            points,
            amplitudes,
            residual,
            tx_positions_m=geometry.tx_xyz_m,
            rx_positions_m=geometry.rx_xyz_m,
            frequencies_hz=frequencies,
            reference_range_m=np.full(3, 200.0),
            convention=convention,
        )
        frequency = compare_native_and_endpoint_uniform_frequency(
            frequencies,
            tx_positions_m=geometry.tx_xyz_m,
            rx_positions_m=geometry.rx_xyz_m,
            point_positions_m=points,
            amplitudes=amplitudes,
            convention=convention,
        )
        mapping = virtual_reference_mapping_regression_fixture(
            observation_positions_m=geometry.tx_xyz_m,
            center_xyz_m=np.asarray([3.5, -1.25, 2.0]),
            point_positions_m=points,
            amplitudes=amplitudes,
            frequencies_hz=np.asarray([9.6e9, 9.601e9, 9.6025e9, 9.604e9], dtype=np.float32),
            r0_offset_m=np.asarray([0.003, -0.003, 0.005]),
            reference_range_m=17.0,
        )
        candidate = HeightSupportCandidate((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
        support = preflight_height_support_candidate(
            candidate,
            geometry,
            reference_range_m=200.0,
            unambiguous_range_m=200.0,
            convention=convention,
        )
        return {
            "schema": "rift_gotcha_acquisition_batch_a_report_v1",
            "synthetic_only": True,
            "test_payload_opened": False,
            "archive": {
                "shard_id": shard.shard_id,
                "view_count": shard.view_count,
                "frequency_count": shard.frequency_count,
                "frequency_dtype": str(shard.frequencies_hz.dtype),
                "native_frequency_retained": bool(
                    shard.phase_reference.frequency_values == "native_stored_exact"
                ),
                "train_observations": len(shard.identities_for_role("train")),
                "validation_observations": len(shard.identities_for_role("validation")),
                "sealed_test_observations": 0,
                "autofocus_mode": shard.autofocus.mode,
                "autofocus_applied": shard.autofocus.applied,
                "r0_definition": shard.phase_reference.r0_interpretation,
                "angle_definition": {
                    "theta_label": shard.phase_reference.theta_label,
                    "phi_label": shard.phase_reference.phi_label,
                    "theta_zero": shard.phase_reference.theta_zero_direction,
                    "phi_zero_plane": shard.phase_reference.phi_zero_plane,
                },
                "autofocus_units": {
                    "range": shard.autofocus.range_unit,
                    "phase": shard.autofocus.phase_unit,
                },
            },
            "paired_monostatic": {
                "count": geometry.count,
                "tx_equals_rx": bool(np.array_equal(geometry.tx_xyz_m, geometry.rx_xyz_m)),
                "cross_product_used": False,
            },
            "direct_float64_fixture": {
                "forward_adjoint": adjoint,
                "forward_vjp_finite_difference": vjp,
                "phase_reference_schema": convention.schema,
                "published_reference_candidate": {
                    "name": PUBLISHED_GOTCHA_REFERENCE_CANDIDATE.named_candidate,
                    "path_formula": PUBLISHED_GOTCHA_REFERENCE_CANDIDATE.path_formula,
                    "reference_formula": PUBLISHED_GOTCHA_REFERENCE_CANDIDATE.reference_formula,
                    "reference_range_factor": PUBLISHED_GOTCHA_REFERENCE_CANDIDATE.reference_range_factor,
                    "forward_phase_sign": PUBLISHED_GOTCHA_REFERENCE_CANDIDATE.phase_sign,
                    "adjoint_phase_sign": -PUBLISHED_GOTCHA_REFERENCE_CANDIDATE.phase_sign,
                    "frequency_usage": "absolute_native_hz_carrier_inclusive_no_centering",
                    "status": PUBLISHED_GOTCHA_REFERENCE_CANDIDATE.real_data_status,
                },
                "accelerated_path": "not_implemented_in_batch_a",
            },
            "native_vs_endpoint_uniform_diagnostic": frequency,
            "virtual_reference_mapping_regression": mapping,
            "height_support_candidate": support,
            "open_real_data_questions": [
                "real GOTCHA fp/r0 carrier-or-baseband phase convention remains unresolved",
                "HH/VV autofocus sign, application order, and carrier/baseband placement remain unverified",
                "candidate xyz bounds are not registered real-data support evidence",
            ],
        }
    finally:
        archive.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    report = run_validation()
    print(json.dumps(report, indent=2, sort_keys=True))
    checks = [
        report["synthetic_only"],
        not report["test_payload_opened"],
        report["paired_monostatic"]["tx_equals_rx"],
        report["direct_float64_fixture"]["forward_adjoint"]["passed"],
        report["direct_float64_fixture"]["forward_vjp_finite_difference"]["passed"],
        report["native_vs_endpoint_uniform_diagnostic"]["native_retained"],
        report["virtual_reference_mapping_regression"]["passed"],
    ]
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
