"""Direct scientific acquisition compatibility for sealed GeRaF B7873200.

The cached matched-filter targets depend on all calibrated Tx/Rx locations and
the exact frequency grid, not only the derived primary-ray geometry.  This
module persists those scientific inputs directly beside the new cache and
compares them directly before reuse or checkpoint recovery.  It intentionally
does not read a radar response and does not use checksums or content hashes.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rift.geraf_b7873200_protocol import (
    B787_3200_ACQUISITION_SCHEMA,
    B787_3200_CACHE_ACQUISITION_FILENAME,
)


_RECORD_VERSION = 1
_ARRAY_FIELDS = ("frequency_hz", "viewpoint_positions", "tx_pos", "rx_pos", "response_shape")
_TEXT_FIELDS = ("schema", "response_dtype", "metadata_json")


def _scalar_text(value: object, label: str) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"B7873200 acquisition {label} must be scalar")
    return str(array.item())


def _metadata_text(metadata: Mapping[str, object]) -> str:
    try:
        return json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("B7873200 acquisition metadata must be JSON-serializable") from exc


def _finite_positive_metadata_float(metadata: Mapping[str, object], name: str) -> float:
    try:
        value = float(metadata[name])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"B7873200 acquisition metadata {name!r} must be numeric") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"B7873200 acquisition metadata {name!r} must be finite and positive")
    return value


def _exact_metadata_integer(
    metadata: Mapping[str, object], name: str, expected: int
) -> int:
    raw = metadata.get(name)
    if isinstance(raw, (bool, np.bool_)):
        raise ValueError(f"B7873200 acquisition metadata {name!r} must be an integer")
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"B7873200 acquisition metadata {name!r} must be an integer") from exc
    if not np.isfinite(value) or not value.is_integer() or int(value) != int(expected):
        raise ValueError(f"B7873200 acquisition metadata {name!r} must equal {int(expected)}")
    return int(value)


def b7873200_operator_frequency_grid_hz(metadata: Mapping[str, object]) -> np.ndarray:
    """Return the exact float64 grid passed to this wrapper's MF operators."""

    center = _finite_positive_metadata_float(metadata, "radar_fc_hz")
    bandwidth = _finite_positive_metadata_float(metadata, "radar_bandwidth_hz")
    count = _exact_metadata_integer(metadata, "num_adc_samples", 600)
    _exact_metadata_integer(metadata, "num_chirps_cpi", 1)
    grid = (center - bandwidth / 2.0) + np.arange(count, dtype=np.float64) * bandwidth / count
    if grid.shape != (600,) or not np.isfinite(grid).all() or not bool(np.all(np.diff(grid) > 0.0)):
        raise ValueError("B7873200 acquisition frequency grid must be finite and strictly increasing")
    return grid


def build_b7873200_acquisition_record(arrays: object) -> dict[str, object]:
    """Extract the target-defining physics inputs without accessing response data."""

    response = getattr(arrays, "response", None)
    response_shape = tuple(int(value) for value in getattr(response, "shape", ()))
    response_dtype = str(getattr(response, "dtype", ""))
    if response_shape != (10_000, 16, 16, 1, 600) or response_dtype != "complex64":
        raise ValueError("B7873200 acquisition record requires the canonical response header")
    metadata = getattr(arrays, "metadata", None)
    if not isinstance(metadata, Mapping):
        raise ValueError("B7873200 acquisition record requires decoded metadata")
    frequency_hz = b7873200_operator_frequency_grid_hz(metadata)
    result: dict[str, object] = {
        "schema": B787_3200_ACQUISITION_SCHEMA,
        "version": _RECORD_VERSION,
        "response_shape": np.asarray(response_shape, dtype=np.int64),
        "response_dtype": response_dtype,
        "metadata_json": _metadata_text(metadata),
        "frequency_hz": np.asarray(frequency_hz, dtype=np.float64),
        "viewpoint_positions": np.asarray(getattr(arrays, "viewpoint_positions"), dtype=np.float64),
        "tx_pos": np.asarray(getattr(arrays, "tx_pos"), dtype=np.float64),
        "rx_pos": np.asarray(getattr(arrays, "rx_pos"), dtype=np.float64),
    }
    expected_shapes = {
        "viewpoint_positions": (10_000, 3),
        "tx_pos": (10_000, 16, 3),
        "rx_pos": (10_000, 16, 3),
    }
    for field, expected_shape in expected_shapes.items():
        values = np.asarray(result[field])
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise ValueError(f"B7873200 acquisition {field} has invalid shape or values")
    return result


def _validate_record_mapping(record: Mapping[str, object], label: str) -> dict[str, object]:
    if _scalar_text(record.get("schema"), f"{label}.schema") != B787_3200_ACQUISITION_SCHEMA:
        raise ValueError(f"B7873200 {label} has an unexpected schema")
    version = np.asarray(record.get("version"))
    if version.shape != () or int(version.item()) != _RECORD_VERSION:
        raise ValueError(f"B7873200 {label} has an unexpected version")
    validated: dict[str, object] = {
        "schema": B787_3200_ACQUISITION_SCHEMA,
        "version": _RECORD_VERSION,
    }
    for field in _TEXT_FIELDS[1:]:
        validated[field] = _scalar_text(record.get(field), f"{label}.{field}")
    if validated["response_dtype"] != "complex64":
        raise ValueError(f"B7873200 {label} has an unexpected response dtype")
    try:
        decoded_metadata = json.loads(str(validated["metadata_json"]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"B7873200 {label} contains invalid metadata JSON") from exc
    if not isinstance(decoded_metadata, Mapping):
        raise ValueError(f"B7873200 {label} metadata JSON must contain an object")
    expected_shapes = {
        "response_shape": (5,),
        "frequency_hz": (600,),
        "viewpoint_positions": (10_000, 3),
        "tx_pos": (10_000, 16, 3),
        "rx_pos": (10_000, 16, 3),
    }
    for field in _ARRAY_FIELDS:
        values = np.asarray(record.get(field))
        if values.shape != expected_shapes[field]:
            raise ValueError(f"B7873200 {label} has invalid {field} shape")
        if field == "response_shape":
            if values.dtype.kind not in "iu" or tuple(int(value) for value in values) != (10_000, 16, 16, 1, 600):
                raise ValueError(f"B7873200 {label} has an unexpected response header")
            validated[field] = np.asarray(values, dtype=np.int64).copy()
        else:
            if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
                raise ValueError(f"B7873200 {label} has invalid {field} values")
            validated[field] = np.asarray(values, dtype=np.float64).copy()
    if not bool(np.all(np.diff(np.asarray(validated["frequency_hz"])) > 0.0)):
        raise ValueError(f"B7873200 {label} frequency grid must be strictly increasing")
    return validated


def acquisition_records_equal(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    """Compare target-defining acquisition state directly, with no digest proxy."""

    try:
        lhs = _validate_record_mapping(left, "left acquisition record")
        rhs = _validate_record_mapping(right, "right acquisition record")
    except (TypeError, ValueError):
        return False
    if lhs["schema"] != rhs["schema"] or lhs["version"] != rhs["version"]:
        return False
    for field in _TEXT_FIELDS[1:]:
        if lhs[field] != rhs[field]:
            return False
    return all(np.array_equal(np.asarray(lhs[field]), np.asarray(rhs[field])) for field in _ARRAY_FIELDS)


def _atomic_save_record(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b", dir=path.parent, prefix=path.name + ".tmp.", suffix=".npz", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez(handle, **dict(record))
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_b7873200_acquisition_record(path: str | os.PathLike[str]) -> dict[str, object]:
    location = Path(path)
    if not location.is_file():
        raise FileNotFoundError(f"B7873200 acquisition record is missing: {location}")
    with np.load(location, allow_pickle=False) as archive:
        required = {"schema", "version", "response_shape", "response_dtype", "metadata_json", *_ARRAY_FIELDS}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"B7873200 acquisition record is missing {missing}")
        record = {field: np.asarray(archive[field]).copy() for field in required}
    return _validate_record_mapping(record, "stored acquisition record")


def write_or_validate_b7873200_acquisition_record(
    cache_root: str | os.PathLike[str], arrays: object
) -> dict[str, object]:
    """Persist or directly compare the cache's complete scientific acquisition state."""

    expected = build_b7873200_acquisition_record(arrays)
    location = Path(cache_root) / B787_3200_CACHE_ACQUISITION_FILENAME
    if location.exists():
        observed = load_b7873200_acquisition_record(location)
        if not acquisition_records_equal(observed, expected):
            raise ValueError(
                "B7873200 cache acquisition differs from the current calibrated poses or frequency grid; "
                "use a new cache root"
            )
        return observed
    _atomic_save_record(location, expected)
    return _validate_record_mapping(expected, "new acquisition record")


def validate_b7873200_acquisition_record(
    cache_root: str | os.PathLike[str], arrays: object
) -> dict[str, object]:
    """Load the cached record and compare it directly with the current source."""

    expected = build_b7873200_acquisition_record(arrays)
    observed = load_b7873200_acquisition_record(
        Path(cache_root) / B787_3200_CACHE_ACQUISITION_FILENAME
    )
    if not acquisition_records_equal(observed, expected):
        raise ValueError(
            "B7873200 prepared targets were made with different calibrated poses, metadata, or frequency grid"
        )
    return observed


def validate_b7873200_operator_frequency_grid(
    record: Mapping[str, object], operator_frequency_hz: object
) -> np.ndarray:
    """Require the persisted acquisition grid to equal the actual MF operator grid."""

    validated = _validate_record_mapping(record, "acquisition record")
    observed = np.asarray(operator_frequency_hz)
    if observed.dtype != np.dtype(np.float64) or observed.shape != (600,):
        raise ValueError("B7873200 operator frequency grid must be float64 with 600 samples")
    if not np.isfinite(observed).all() or not bool(np.all(np.diff(observed) > 0.0)):
        raise ValueError("B7873200 operator frequency grid must be finite and strictly increasing")
    if not np.array_equal(np.asarray(validated["frequency_hz"]), observed):
        raise ValueError(
            "B7873200 matched-filter operator frequency grid differs from the prepared acquisition record"
        )
    return observed.copy()
