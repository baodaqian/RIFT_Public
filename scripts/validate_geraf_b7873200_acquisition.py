#!/usr/bin/env python
"""Data-free regression checks for B7873200 acquisition compatibility.

The new GeRaF cache stores the scientific inputs which define every native
matched-filter target: the response header, decoded acquisition metadata,
frequency grid, and every calibrated viewpoint/Tx/Rx position.  This test uses
only a synthetic header and geometry; it never opens the B787 archive or a
radar response payload.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _validation_tmp_parent() -> Path:
    """Use the allocated launcher scratch when supplied, else preserve local behavior."""

    configured = os.environ.get("RIFT_VALIDATION_TMPDIR")
    parent = ROOT if configured is None else Path(configured).expanduser().resolve()
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        raise RuntimeError(f"no usable validation temporary parent: {parent}")
    return parent


def _load_isolated_modules() -> object:
    """Load the data-only modules without importing the ordinary RIFT package."""

    package = types.ModuleType("rift")
    package.__path__ = [str(ROOT / "rift")]  # type: ignore[attr-defined]
    sys.modules["rift"] = package
    protocol_spec = importlib.util.spec_from_file_location(
        "rift.geraf_b7873200_protocol", ROOT / "rift" / "geraf_b7873200_protocol.py"
    )
    if protocol_spec is None or protocol_spec.loader is None:
        raise RuntimeError("cannot load the isolated B7873200 protocol module")
    protocol = importlib.util.module_from_spec(protocol_spec)
    sys.modules[protocol_spec.name] = protocol
    protocol_spec.loader.exec_module(protocol)
    acquisition_spec = importlib.util.spec_from_file_location(
        "rift.geraf_b7873200_acquisition_under_test",
        ROOT / "rift" / "geraf_b7873200_acquisition.py",
    )
    if acquisition_spec is None or acquisition_spec.loader is None:
        raise RuntimeError("cannot load the isolated B7873200 acquisition module")
    acquisition = importlib.util.module_from_spec(acquisition_spec)
    sys.modules[acquisition_spec.name] = acquisition
    acquisition_spec.loader.exec_module(acquisition)
    return acquisition


acquisition = _load_isolated_modules()


class Gates:
    def __init__(self) -> None:
        self.count = 0

    def check(self, condition: bool, detail: str) -> None:
        if not condition:
            raise AssertionError(detail)
        self.count += 1
        print(f"PASS {self.count:02d}: {detail}", flush=True)


def _expect_rejection(gates: Gates, detail: str, action: object) -> None:
    try:
        action()  # type: ignore[operator]
    except ValueError:
        gates.check(True, detail)
    else:
        raise AssertionError(detail)


class HeaderOnlyResponse:
    """A response surface which supports metadata but deliberately no payload read."""

    shape = (10_000, 16, 16, 1, 600)
    dtype = np.dtype(np.complex64)

    def __getitem__(self, _index: object) -> np.ndarray:
        raise AssertionError("acquisition compatibility must not index response payload")


@dataclass
class SyntheticArrays:
    response: HeaderOnlyResponse
    metadata: dict[str, object]
    viewpoint_positions: np.ndarray
    tx_pos: np.ndarray
    rx_pos: np.ndarray


def _arrays() -> SyntheticArrays:
    viewpoints = np.arange(10_000 * 3, dtype=np.float64).reshape(10_000, 3) / 100.0
    tx = np.arange(10_000 * 16 * 3, dtype=np.float64).reshape(10_000, 16, 3) / 1000.0
    rx = -np.arange(10_000 * 16 * 3, dtype=np.float64).reshape(10_000, 16, 3) / 1500.0
    return SyntheticArrays(
        response=HeaderOnlyResponse(),
        metadata={
            "radar_fc_hz": 10.0e9,
            "radar_bandwidth_hz": 3.0e9,
            "num_adc_samples": 600,
            "num_chirps_cpi": 1,
            "calibration_label": "synthetic-regression-fixture",
        },
        viewpoint_positions=viewpoints,
        tx_pos=tx,
        rx_pos=rx,
    )


def _copy_arrays(source: SyntheticArrays) -> SyntheticArrays:
    return SyntheticArrays(
        response=source.response,
        metadata=dict(source.metadata),
        viewpoint_positions=source.viewpoint_positions.copy(),
        tx_pos=source.tx_pos.copy(),
        rx_pos=source.rx_pos.copy(),
    )


def main() -> None:
    gates = Gates()
    source = _arrays()
    record = acquisition.build_b7873200_acquisition_record(source)
    gates.check(
        tuple(record["response_shape"].tolist()) == (10_000, 16, 16, 1, 600)
        and record["response_dtype"] == "complex64"
        and record["frequency_hz"].shape == (600,),
        "the direct record preserves the canonical header and all 600 frequency samples",
    )
    gates.check(
        record["viewpoint_positions"].dtype == np.float64
        and record["tx_pos"].shape == (10_000, 16, 3)
        and record["rx_pos"].shape == (10_000, 16, 3),
        "the direct record contains every calibrated viewpoint, Tx, and Rx position",
    )
    gates.check(
        acquisition.acquisition_records_equal(record, acquisition.build_b7873200_acquisition_record(source)),
        "identical header-only sources compare equal without response-payload access",
    )
    operator_grid = acquisition.b7873200_operator_frequency_grid_hz(source.metadata)
    gates.check(
        np.array_equal(
            acquisition.validate_b7873200_operator_frequency_grid(record, operator_grid),
            record["frequency_hz"],
        ),
        "the persisted acquisition grid exactly equals the grid passed to the matched-filter operator",
    )
    shifted_operator_grid = operator_grid.copy()
    shifted_operator_grid[271] += 1.0
    _expect_rejection(
        gates,
        "a shifted matched-filter operator grid is rejected before use",
        lambda: acquisition.validate_b7873200_operator_frequency_grid(record, shifted_operator_grid),
    )
    for detail, key, value in (
        ("zero bandwidth is rejected before acquisition caching", "radar_bandwidth_hz", 0.0),
        ("negative bandwidth is rejected before acquisition caching", "radar_bandwidth_hz", -1.0),
        ("fractional ADC count is rejected before acquisition caching", "num_adc_samples", 600.5),
    ):
        invalid_metadata = _copy_arrays(source)
        invalid_metadata.metadata[key] = value
        _expect_rejection(
            gates,
            detail,
            lambda invalid_metadata=invalid_metadata: acquisition.build_b7873200_acquisition_record(
                invalid_metadata
            ),
        )
    # A launcher supplies a job-owned node-local parent; locally we preserve a
    # workspace-local fixture for desktop sandboxes that restrict OS temp.
    with tempfile.TemporaryDirectory(
        prefix=".geraf_b7873200_acquisition_", dir=_validation_tmp_parent()
    ) as temporary:
        root = Path(temporary)
        stored = acquisition.write_or_validate_b7873200_acquisition_record(root, source)
        loaded = acquisition.validate_b7873200_acquisition_record(root, source)
        gates.check(
            acquisition.acquisition_records_equal(stored, loaded),
            "a stored acquisition record round-trips and validates against the current source",
        )

        changed_tx = _copy_arrays(source)
        changed_tx.tx_pos[4_321, 7, 1] += 0.125
        _expect_rejection(
            gates,
            "an interior calibrated Tx-position change rejects cache reuse",
            lambda: acquisition.validate_b7873200_acquisition_record(root, changed_tx),
        )
        changed_rx = _copy_arrays(source)
        changed_rx.rx_pos[3_210, 5, 2] -= 0.25
        _expect_rejection(
            gates,
            "an interior calibrated Rx-position change rejects cache reuse",
            lambda: acquisition.validate_b7873200_acquisition_record(root, changed_rx),
        )
        changed_viewpoint = _copy_arrays(source)
        changed_viewpoint.viewpoint_positions[1_234, 0] += 0.5
        _expect_rejection(
            gates,
            "an interior calibrated viewpoint change rejects cache reuse",
            lambda: acquisition.validate_b7873200_acquisition_record(root, changed_viewpoint),
        )
        changed_frequency = _copy_arrays(source)
        changed_frequency.metadata["radar_fc_hz"] = 10.0e9 + 1.0
        _expect_rejection(
            gates,
            "a frequency-grid change rejects cache reuse",
            lambda: acquisition.validate_b7873200_acquisition_record(root, changed_frequency),
        )
        changed_metadata = _copy_arrays(source)
        changed_metadata.metadata["calibration_label"] = "other-fixture"
        _expect_rejection(
            gates,
            "a relevant decoded-metadata change rejects cache reuse",
            lambda: acquisition.validate_b7873200_acquisition_record(root, changed_metadata),
        )
    print(f"GeRaF B7873200 acquisition compatibility passed: {gates.count} checks.", flush=True)


if __name__ == "__main__":
    main()
