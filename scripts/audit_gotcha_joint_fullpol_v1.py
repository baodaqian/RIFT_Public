#!/usr/bin/env python3
"""Gate 0 inventory audit for the GOTCHA Joint-8 full-pol experiment.

The audit inventories all 11,520 raw MATLAB paths, but opens payloads only for
the 10,368 train/validation files.  The 1,152 sealed-test payloads remain
unopened.  For opened files it records native frequency grids and pulse counts
without ever applying the supplied autofocus corrections.  The complete audit
therefore refuses to run outside a Slurm allocation.

The emitted JSON is both a raw-data inventory and a valid
``rift.gotcha_joint_fullpol`` v1 joint manifest.  Extra inventory fields are
additive; the common contract validator ignores them while enforcing the exact
32-shard/360-sector layout and shared split.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence
import uuid

import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rift.gotcha_joint_fullpol import (  # noqa: E402
    CO_POLARIZATIONS,
    INVENTORIED_SOURCE_FILE_COUNT,
    MANIFEST_SCHEMA,
    MANAGER_TRACK_ID,
    PASS_IDS,
    PAYLOAD_AUDITED_SECTOR_COUNT,
    PAYLOAD_AUDITED_SECTOR_IDS,
    PAYLOAD_AUDITED_SOURCE_FILE_COUNT,
    POLARIZATIONS,
    SCENE_ID,
    SCENE_COUNT,
    SEALED_TEST_SECTOR_COUNT,
    SEALED_TEST_SECTOR_IDS,
    SEALED_TEST_SOURCE_FILE_COUNT,
    SECTOR_IDS,
    SOURCE_FILE_COUNT,
    build_sector_split,
    canonical_shard_id,
    support_contract,
    validate_joint_manifest,
)


DEFAULT_DATASET_ROOT = Path(
    "/storage/scratch1/1/dbao31/datasets/rift_public_radar/"
    "afrl_sdms_20260814"
)
PRODUCTION_OUTPUT_PATH = Path(
    "/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/"
    "public_radar_tuning/gotcha_joint8_fullpol_v1/"
    "joint8_fullpol_dataaudit_v1.json"
)
INVENTORY_SCHEMA = "rift_gotcha_joint_fullpol_raw_inventory_v1"
GATE_ID = "joint8_fullpol_dataaudit_v1"
EXPECTED_FLOAT_DTYPE = np.dtype(np.float32)
EXPECTED_COMPLEX_DTYPE = np.dtype(np.complex64)
GEOMETRY_FIELDS = ("x", "y", "z", "r0", "th", "phi")
AUTOFOCUS_FIELDS = ("r_correct", "ph_correct")
_MISSING = object()

_SEALED_SPLIT = build_sector_split()
if PAYLOAD_AUDITED_SECTOR_COUNT != 324:
    raise AssertionError("Gate 0 must audit exactly 324 payload sectors per shard")
if SEALED_TEST_SECTOR_COUNT != 36:
    raise AssertionError("Gate 0 must seal exactly 36 test sectors per shard")
if PAYLOAD_AUDITED_SOURCE_FILE_COUNT != 10368:
    raise AssertionError("Gate 0 must audit exactly 10,368 payload files")
if SEALED_TEST_SOURCE_FILE_COUNT != 1152:
    raise AssertionError("Gate 0 must leave exactly 1,152 test payloads unopened")


class AuditError(ValueError):
    """A deterministic raw-data contract violation."""


def disc_name_for_pass(pass_id: int) -> str:
    """Return the exact AFRL disc directory that owns ``pass_id``."""

    pass_id = int(pass_id)
    if pass_id not in PASS_IDS:
        raise AuditError(f"pass_id must be one of {PASS_IDS}, got {pass_id}")
    return "GOTCHA-CP_Disc1" if pass_id <= 7 else "GOTCHA-CP_Disc2"


def raw_shard_directory(
    dataset_root: str | os.PathLike[str],
    pass_id: int,
    polarization: str,
) -> Path:
    """Resolve the official Disc1/Disc2 directory for one native shard."""

    polarization = str(polarization).lower()
    canonical_shard_id(pass_id, polarization)  # Validate both identifiers.
    disc = disc_name_for_pass(pass_id)
    return (
        Path(dataset_root)
        / "extracted"
        / "data"
        / "GOTCHA"
        / disc
        / "DATA"
        / f"pass{int(pass_id)}"
        / polarization.upper()
    )


def expected_source_name(pass_id: int, polarization: str, sector_id: int) -> str:
    """Return the one legal filename for a pass/polarization/sector."""

    polarization = str(polarization).lower()
    canonical_shard_id(pass_id, polarization)
    sector_id = int(sector_id)
    if sector_id not in SECTOR_IDS:
        raise AuditError(f"sector_id must be in 1..360, got {sector_id}")
    return (
        f"data_3dsar_pass{int(pass_id)}_az{sector_id:03d}_"
        f"{polarization.upper()}.mat"
    )


def discover_shard_files(
    dataset_root: str | os.PathLike[str],
    pass_id: int,
    polarization: str,
    *,
    expected_sector_ids: Sequence[int] = SECTOR_IDS,
) -> tuple[Path, ...]:
    """Fail closed unless a shard has exactly its expected MAT filenames.

    ``expected_sector_ids`` exists so the pure discovery helper can be tested
    on a tiny fixture.  Production calls always use the sealed 1..360 default.
    """

    sector_ids = tuple(int(value) for value in expected_sector_ids)
    if not sector_ids or len(set(sector_ids)) != len(sector_ids):
        raise AuditError("expected_sector_ids must be nonempty and unique")
    if any(value not in SECTOR_IDS for value in sector_ids):
        raise AuditError("expected_sector_ids must be contained in 1..360")

    directory = raw_shard_directory(dataset_root, pass_id, polarization)
    if not directory.is_dir():
        raise AuditError(f"missing raw shard directory: {directory}")

    expected_names = tuple(
        expected_source_name(pass_id, polarization, sector_id)
        for sector_id in sector_ids
    )
    found_paths = tuple(
        sorted(
            (
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() == ".mat"
            ),
            key=lambda path: path.name,
        )
    )
    found_names = tuple(path.name for path in found_paths)
    if len(found_names) != len(expected_names) or set(found_names) != set(expected_names):
        missing = sorted(set(expected_names) - set(found_names))
        extra = sorted(set(found_names) - set(expected_names))
        raise AuditError(
            f"{directory}: expected exactly {len(expected_names)} MAT files; "
            f"found={len(found_names)}, missing={missing[:8]}, extra={extra[:8]}"
        )

    by_name = {path.name: path for path in found_paths}
    return tuple(by_name[name] for name in expected_names)


def _unwrap_singleton(value: Any) -> Any:
    while isinstance(value, np.ndarray) and value.size == 1:
        if value.dtype.names is not None:
            return value.reshape(-1)[0]
        if value.dtype == object:
            value = value.reshape(-1)[0]
            continue
        break
    return value


def _field(container: Any, name: str, *, required: bool = True) -> Any:
    container = _unwrap_singleton(container)
    value = _MISSING
    if isinstance(container, Mapping):
        value = container.get(name, _MISSING)
    elif isinstance(container, np.void) and container.dtype.names:
        if name in container.dtype.names:
            value = container[name]
    elif hasattr(container, name):
        value = getattr(container, name)

    if value is _MISSING and required:
        raise AuditError(f"missing MATLAB field {name!r}")
    return value


def _load_nested_data(path: Path) -> Any:
    try:
        payload = loadmat(
            path,
            squeeze_me=True,
            struct_as_record=False,
            verify_compressed_data_integrity=True,
        )
    except NotImplementedError as exc:
        raise AuditError(
            f"{path}: MATLAB v7.3/HDF5 is not supported by scipy.io.loadmat"
        ) from exc
    except Exception as exc:
        raise AuditError(f"{path}: scipy.io.loadmat failed: {exc}") from exc

    public = {key: value for key, value in payload.items() if not key.startswith("__")}
    if set(public) != {"data"}:
        raise AuditError(
            f"{path}: expected exactly one public top-level variable named 'data'; "
            f"found {sorted(public)}"
        )
    return _unwrap_singleton(public["data"])


def _array(
    value: Any,
    *,
    path: Path,
    field_name: str,
    expected_dtype: np.dtype,
    ndim: int,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim:
        raise AuditError(
            f"{path}:{field_name} must have ndim={ndim}, got shape={array.shape}"
        )
    if array.dtype.kind != expected_dtype.kind or array.dtype.itemsize != expected_dtype.itemsize:
        raise AuditError(
            f"{path}:{field_name} dtype must be {expected_dtype.name}, "
            f"got {array.dtype}"
        )
    if not np.isfinite(array).all():
        raise AuditError(f"{path}:{field_name} contains nonfinite values")
    return array


def audit_mat_file(
    path: str | os.PathLike[str],
    *,
    pass_id: int,
    polarization: str,
    sector_id: int,
) -> dict[str, Any]:
    """Open and validate one official nested GOTCHA MAT file.

    Returned frequency values are native values used by the shard auditor to
    prove a common grid.  No response or autofocus value is transformed.
    """

    path = Path(path)
    polarization = str(polarization).lower()
    shard_id = canonical_shard_id(pass_id, polarization)
    expected_name = expected_source_name(pass_id, polarization, sector_id)
    if path.name != expected_name:
        raise AuditError(f"expected {expected_name}, got {path.name}")
    if int(sector_id) in SEALED_TEST_SECTOR_IDS:
        raise AuditError(
            f"{path}: sealed test-sector payload must not be opened during Gate 0"
        )

    data = _load_nested_data(path)
    fp = _array(
        _field(data, "fp"),
        path=path,
        field_name="fp",
        expected_dtype=EXPECTED_COMPLEX_DTYPE,
        ndim=2,
    )
    freq = _array(
        _field(data, "freq"),
        path=path,
        field_name="freq",
        expected_dtype=EXPECTED_FLOAT_DTYPE,
        ndim=1,
    )
    if freq.size == 0 or not np.all(np.diff(freq.astype(np.float64)) > 0.0):
        raise AuditError(f"{path}:freq must be a nonempty strictly increasing vector")

    geometry: dict[str, np.ndarray] = {}
    for name in GEOMETRY_FIELDS:
        geometry[name] = _array(
            _field(data, name),
            path=path,
            field_name=name,
            expected_dtype=EXPECTED_FLOAT_DTYPE,
            ndim=1,
        )

    pulse_count = int(geometry["x"].size)
    if pulse_count <= 0:
        raise AuditError(f"{path}: native pulse count must be positive")
    for name, values in geometry.items():
        if values.size != pulse_count:
            raise AuditError(
                f"{path}:{name} has {values.size} rows, expected {pulse_count}"
            )
    expected_fp_shape = (int(freq.size), pulse_count)
    if fp.shape != expected_fp_shape:
        raise AuditError(
            f"{path}:fp shape must be [frequency,pulse]={expected_fp_shape}, "
            f"got {fp.shape}"
        )

    autofocus = _field(data, "af", required=False)
    autofocus_stats: dict[str, dict[str, float]] | None = None
    if polarization in CO_POLARIZATIONS:
        if autofocus is _MISSING:
            raise AuditError(
                f"{path}: {polarization.upper()} must contain its own af structure"
            )
        autofocus_stats = {}
        for name in AUTOFOCUS_FIELDS:
            values = _array(
                _field(autofocus, name),
                path=path,
                field_name=f"af.{name}",
                expected_dtype=EXPECTED_FLOAT_DTYPE,
                ndim=1,
            )
            if values.size != pulse_count:
                raise AuditError(
                    f"{path}:af.{name} has {values.size} rows, expected {pulse_count}"
                )
            autofocus_stats[name] = {
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
    elif autofocus is not _MISSING:
        raise AuditError(
            f"{path}: {polarization.upper()} must not contain official autofocus fields"
        )

    return {
        "shard_id": shard_id,
        "pass_id": int(pass_id),
        "polarization": polarization,
        "sector_id": int(sector_id),
        "source_file": path.resolve(strict=True).as_posix(),
        "frequency_hz": freq.copy(),
        "frequency_count": int(freq.size),
        "pulse_count": pulse_count,
        "fp_shape": [int(fp.shape[0]), int(fp.shape[1])],
        "fp_dtype": EXPECTED_COMPLEX_DTYPE.name,
        "frequency_dtype": EXPECTED_FLOAT_DTYPE.name,
        "geometry_dtype": EXPECTED_FLOAT_DTYPE.name,
        "autofocus_present": autofocus_stats is not None,
        "autofocus_stats": autofocus_stats,
        "payload_opened": True,
        "corrections_applied": False,
    }


def _combined_extrema(
    records: Sequence[dict[str, Any]], field_name: str
) -> dict[str, float] | None:
    stats = [
        record["autofocus_stats"][field_name]
        for record in records
        if record["autofocus_stats"] is not None
    ]
    if not stats:
        return None
    return {
        "minimum": min(float(item["minimum"]) for item in stats),
        "maximum": max(float(item["maximum"]) for item in stats),
    }


def audit_shard(
    dataset_root: str | os.PathLike[str],
    pass_id: int,
    polarization: str,
    *,
    file_auditor: Callable[..., dict[str, Any]] = audit_mat_file,
) -> dict[str, Any]:
    """Inventory 360 paths while opening only train/validation payloads."""

    polarization = str(polarization).lower()
    shard_id = canonical_shard_id(pass_id, polarization)
    paths = discover_shard_files(dataset_root, pass_id, polarization)
    split = build_sector_split()
    source_files: list[str] = []
    payload_records: list[dict[str, Any]] = []
    file_records: list[dict[str, Any]] = []
    official_autofocus = polarization in CO_POLARIZATIONS
    for sector_id, path in zip(SECTOR_IDS, paths):
        source_file = path.resolve(strict=True).as_posix()
        source_files.append(source_file)
        sector_role = split.role_for(sector_id)
        if sector_role == "test":
            file_records.append(
                {
                    "sector_id": int(sector_id),
                    "sector_role": sector_role,
                    "payload_opened": False,
                    "corrections_applied": False,
                }
            )
            continue

        record = file_auditor(
            path,
            pass_id=pass_id,
            polarization=polarization,
            sector_id=sector_id,
        )
        if record.get("source_file") != source_file:
            raise AuditError(f"{path}: payload audit returned the wrong source identity")
        if int(record.get("sector_id", -1)) != sector_id:
            raise AuditError(f"{path}: payload audit returned the wrong sector identity")
        if record.get("payload_opened") is not True:
            raise AuditError(f"{path}: payload audit did not prove payload_opened=true")
        if record.get("corrections_applied") is not False:
            raise AuditError(f"{path}: Gate 0 must never apply autofocus corrections")
        if bool(record.get("autofocus_present")) != official_autofocus:
            raise AuditError(f"{path}: autofocus presence disagrees with polarization")
        payload_records.append(record)
        file_records.append(
            {
                "sector_id": int(sector_id),
                "sector_role": sector_role,
                "payload_opened": True,
                "frequency_count": int(record["frequency_count"]),
                "pulse_count": int(record["pulse_count"]),
                "fp_shape": list(record["fp_shape"]),
                "autofocus_present": bool(record["autofocus_present"]),
                "corrections_applied": False,
            }
        )

    if [record["sector_id"] for record in payload_records] != list(
        PAYLOAD_AUDITED_SECTOR_IDS
    ):
        raise AuditError(f"{shard_id}: payload audit sector coverage is not exact")
    frequency_hz = payload_records[0]["frequency_hz"]
    for record in payload_records[1:]:
        if not np.array_equal(record["frequency_hz"], frequency_hz):
            raise AuditError(
                f"{record['source_file']}: native frequency grid differs within {shard_id}"
            )

    pulse_counts = [int(record["pulse_count"]) for record in payload_records]
    autofocus: dict[str, Any] = {
        "official_available": official_autofocus,
        "present_in_all_payload_audited_source_files": official_autofocus,
        "source_shard_id": shard_id if official_autofocus else None,
        "source_polarization": polarization if official_autofocus else None,
        "range_field": "af.r_correct" if official_autofocus else None,
        "phase_field": "af.ph_correct" if official_autofocus else None,
        "payload_audited_source_file_count": len(PAYLOAD_AUDITED_SECTOR_IDS),
        "applied": False,
        "policy": "audited_in_native_files_never_applied_by_gate0",
    }
    if official_autofocus:
        autofocus["r_correct_extrema"] = _combined_extrema(
            payload_records, "r_correct"
        )
        autofocus["ph_correct_extrema"] = _combined_extrema(
            payload_records, "ph_correct"
        )

    return {
        "shard_id": shard_id,
        "pass_id": int(pass_id),
        "polarization": polarization,
        "disc": disc_name_for_pass(pass_id),
        "raw_directory": raw_shard_directory(
            dataset_root, pass_id, polarization
        ).resolve(strict=True).as_posix(),
        "sector_ids": list(SECTOR_IDS),
        "sector_roles": list(split.role_by_sector),
        "source_files": source_files,
        "source_file_count": len(SECTOR_IDS),
        "inventoried_source_file_count": len(SECTOR_IDS),
        "payload_audited_sector_ids": list(PAYLOAD_AUDITED_SECTOR_IDS),
        "payload_audited_sector_count": PAYLOAD_AUDITED_SECTOR_COUNT,
        "payload_audited_source_file_count": len(PAYLOAD_AUDITED_SECTOR_IDS),
        "sealed_test_sector_ids": list(SEALED_TEST_SECTOR_IDS),
        "sealed_test_sector_count": SEALED_TEST_SECTOR_COUNT,
        "sealed_test_source_file_count": len(SEALED_TEST_SECTOR_IDS),
        "native_frequency_hz": [float(value) for value in frequency_hz],
        "native_frequency_count": int(frequency_hz.size),
        "native_frequency_dtype": EXPECTED_FLOAT_DTYPE.name,
        "payload_audited_native_pulse_counts_by_sector": pulse_counts,
        "payload_audited_view_count": int(sum(pulse_counts)),
        "payload_audited_complex_sample_count": int(
            frequency_hz.size * sum(pulse_counts)
        ),
        "fp_dtype": EXPECTED_COMPLEX_DTYPE.name,
        "geometry_dtype": EXPECTED_FLOAT_DTYPE.name,
        "validated_fields": ["fp", "freq", *GEOMETRY_FIELDS],
        "file_records": file_records,
        "autofocus": autofocus,
        "corrections_applied": False,
    }


_SEALED_TEST_FILE_RECORD_KEYS = {
    "sector_id",
    "sector_role",
    "payload_opened",
    "corrections_applied",
}
_PAYLOAD_FILE_RECORD_KEYS = _SEALED_TEST_FILE_RECORD_KEYS | {
    "frequency_count",
    "pulse_count",
    "fp_shape",
    "autofocus_present",
}


def validate_gate0_inventory_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_dataset_root: str | os.PathLike[str] | None = None,
    require_source_files_exist: bool = False,
) -> dict[str, Any]:
    """Validate the sealed-payload boundary in an emitted Gate-0 manifest."""

    if not isinstance(manifest, Mapping):
        raise AuditError("Gate-0 manifest must be a mapping")
    core_summary = validate_joint_manifest(manifest)
    expected_top = {
        "inventory_schema": INVENTORY_SCHEMA,
        "gate_id": GATE_ID,
        "manager_track_id": MANAGER_TRACK_ID,
        "scene_count": SCENE_COUNT,
        "source_file_count": INVENTORIED_SOURCE_FILE_COUNT,
        "inventoried_source_file_count": INVENTORIED_SOURCE_FILE_COUNT,
        "payload_audited_source_file_count": PAYLOAD_AUDITED_SOURCE_FILE_COUNT,
        "sealed_test_source_file_count": SEALED_TEST_SOURCE_FILE_COUNT,
        "test_opened": False,
        "corrections_applied": False,
    }
    for key, expected in expected_top.items():
        if manifest.get(key) != expected:
            raise AuditError(
                f"Gate-0 {key} must be {expected!r}, got {manifest.get(key)!r}"
            )
    if expected_dataset_root is not None:
        actual_root = Path(str(manifest.get("dataset_root", ""))).resolve()
        if actual_root != Path(expected_dataset_root).resolve():
            raise AuditError("Gate-0 manifest names the wrong dataset root")

    shards = manifest.get("shards")
    if not isinstance(shards, Sequence) or isinstance(shards, (str, bytes)):
        raise AuditError("Gate-0 shards must be a sequence")
    payload_count = 0
    sealed_count = 0
    inventoried_count = 0
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise AuditError("every Gate-0 shard must be a mapping")
        shard_id = str(shard.get("shard_id", "<unknown>"))
        polarization = str(shard.get("polarization", "")).lower()
        expected_shard = {
            "source_file_count": len(SECTOR_IDS),
            "inventoried_source_file_count": len(SECTOR_IDS),
            "payload_audited_sector_count": PAYLOAD_AUDITED_SECTOR_COUNT,
            "payload_audited_source_file_count": len(PAYLOAD_AUDITED_SECTOR_IDS),
            "sealed_test_sector_count": SEALED_TEST_SECTOR_COUNT,
            "sealed_test_source_file_count": len(SEALED_TEST_SECTOR_IDS),
            "corrections_applied": False,
        }
        for key, expected in expected_shard.items():
            if shard.get(key) != expected:
                raise AuditError(
                    f"{shard_id}.{key} must be {expected!r}, got {shard.get(key)!r}"
                )
        if shard.get("payload_audited_sector_ids") != list(
            PAYLOAD_AUDITED_SECTOR_IDS
        ):
            raise AuditError(f"{shard_id} payload_audited_sector_ids are not exact")
        if shard.get("sealed_test_sector_ids") != list(SEALED_TEST_SECTOR_IDS):
            raise AuditError(f"{shard_id} sealed_test_sector_ids are not exact")

        source_files = shard.get("source_files")
        file_records = shard.get("file_records")
        if not isinstance(source_files, Sequence) or isinstance(
            source_files, (str, bytes)
        ):
            raise AuditError(f"{shard_id}.source_files must be a sequence")
        if not isinstance(file_records, Sequence) or isinstance(
            file_records, (str, bytes)
        ):
            raise AuditError(f"{shard_id}.file_records must be a sequence")
        if len(source_files) != len(SECTOR_IDS) or len(file_records) != len(SECTOR_IDS):
            raise AuditError(f"{shard_id} must inventory exactly 360 source records")

        shard_payload_count = 0
        shard_sealed_count = 0
        for sector_id, source_file, record in zip(
            SECTOR_IDS, source_files, file_records
        ):
            if not isinstance(record, Mapping):
                raise AuditError(f"{shard_id} sector {sector_id} record is not a mapping")
            expected_role = _SEALED_SPLIT.role_for(sector_id)
            if record.get("sector_id") != sector_id:
                raise AuditError(f"{shard_id} file-record sector order drifted")
            if record.get("sector_role") != expected_role:
                raise AuditError(f"{shard_id} sector {sector_id} role drifted")
            if record.get("corrections_applied") is not False:
                raise AuditError(
                    f"{shard_id} sector {sector_id} applies a correction in Gate 0"
                )
            if expected_role == "test":
                if set(record) != _SEALED_TEST_FILE_RECORD_KEYS:
                    raise AuditError(
                        f"{shard_id} sealed sector {sector_id} carries payload-derived fields"
                    )
                if record.get("payload_opened") is not False:
                    raise AuditError(
                        f"{shard_id} sealed sector {sector_id} must remain unopened"
                    )
                shard_sealed_count += 1
            else:
                if set(record) != _PAYLOAD_FILE_RECORD_KEYS:
                    raise AuditError(
                        f"{shard_id} payload sector {sector_id} has incomplete provenance"
                    )
                if record.get("payload_opened") is not True:
                    raise AuditError(
                        f"{shard_id} payload sector {sector_id} was not audited"
                    )
                if bool(record.get("autofocus_present")) != (
                    polarization in CO_POLARIZATIONS
                ):
                    raise AuditError(
                        f"{shard_id} sector {sector_id} autofocus provenance is wrong"
                    )
                if int(record.get("frequency_count", 0)) <= 0:
                    raise AuditError(f"{shard_id} has an invalid frequency count")
                if int(record.get("pulse_count", 0)) <= 0:
                    raise AuditError(f"{shard_id} has an invalid pulse count")
                if record.get("fp_shape") != [
                    int(record["frequency_count"]),
                    int(record["pulse_count"]),
                ]:
                    raise AuditError(f"{shard_id} has an invalid fp shape record")
                shard_payload_count += 1
            if require_source_files_exist:
                source = Path(str(source_file))
                if source.is_symlink() or not source.is_file():
                    raise AuditError(
                        f"{shard_id} source is missing or not regular: {source}"
                    )

        if shard_payload_count != len(PAYLOAD_AUDITED_SECTOR_IDS):
            raise AuditError(f"{shard_id} payload-opened count is not 324")
        if shard_sealed_count != len(SEALED_TEST_SECTOR_IDS):
            raise AuditError(f"{shard_id} sealed-test count is not 36")
        pulse_counts = shard.get("payload_audited_native_pulse_counts_by_sector")
        if not isinstance(pulse_counts, Sequence) or isinstance(
            pulse_counts, (str, bytes)
        ):
            raise AuditError(f"{shard_id} payload pulse counts must be a sequence")
        if len(pulse_counts) != len(PAYLOAD_AUDITED_SECTOR_IDS):
            raise AuditError(f"{shard_id} payload pulse-count coverage is not 324")
        if any(int(value) <= 0 for value in pulse_counts):
            raise AuditError(f"{shard_id} payload pulse counts must be positive")
        frequency = shard.get("native_frequency_hz")
        if not isinstance(frequency, Sequence) or isinstance(frequency, (str, bytes)):
            raise AuditError(f"{shard_id} native frequency must be a sequence")
        frequency_values = np.asarray(frequency, dtype=np.float64)
        if (
            frequency_values.ndim != 1
            or frequency_values.size != int(shard.get("native_frequency_count", -1))
            or not np.isfinite(frequency_values).all()
            or not np.all(np.diff(frequency_values) > 0.0)
        ):
            raise AuditError(f"{shard_id} native frequency record is invalid")
        if int(shard.get("payload_audited_view_count", -1)) != sum(
            int(value) for value in pulse_counts
        ):
            raise AuditError(f"{shard_id} payload view count is inconsistent")
        if int(shard.get("payload_audited_complex_sample_count", -1)) != int(
            frequency_values.size * sum(int(value) for value in pulse_counts)
        ):
            raise AuditError(f"{shard_id} payload complex-sample count is inconsistent")
        autofocus = shard.get("autofocus")
        if not isinstance(autofocus, Mapping) or autofocus.get("applied") is not False:
            raise AuditError(f"{shard_id} autofocus policy is invalid")

        inventoried_count += len(source_files)
        payload_count += shard_payload_count
        sealed_count += shard_sealed_count

    if inventoried_count != INVENTORIED_SOURCE_FILE_COUNT:
        raise AuditError("Gate-0 inventoried source count is not 11,520")
    if payload_count != PAYLOAD_AUDITED_SOURCE_FILE_COUNT:
        raise AuditError("Gate-0 payload-audited source count is not 10,368")
    if sealed_count != SEALED_TEST_SOURCE_FILE_COUNT:
        raise AuditError("Gate-0 sealed-test source count is not 1,152")
    totals = manifest.get("totals")
    if not isinstance(totals, Mapping):
        raise AuditError("Gate-0 totals must be a mapping")
    expected_totals = {
        "shards": len(PASS_IDS) * len(POLARIZATIONS),
        "inventoried_source_files": INVENTORIED_SOURCE_FILE_COUNT,
        "payload_audited_source_files": PAYLOAD_AUDITED_SOURCE_FILE_COUNT,
        "sealed_test_source_files": SEALED_TEST_SOURCE_FILE_COUNT,
    }
    for key, expected in expected_totals.items():
        if totals.get(key) != expected:
            raise AuditError(f"Gate-0 totals.{key} must be {expected}")
    return {
        **core_summary,
        "inventoried_source_file_count": inventoried_count,
        "payload_audited_source_file_count": payload_count,
        "sealed_test_source_file_count": sealed_count,
        "test_opened": False,
        "corrections_applied": False,
    }


def build_joint_inventory_manifest(
    dataset_root: str | os.PathLike[str],
    shards: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble and validate the additive Gate 0 joint manifest."""

    split = build_sector_split()
    serialized_shards = [dict(shard) for shard in shards]
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "inventory_schema": INVENTORY_SCHEMA,
        "gate_id": GATE_ID,
        "manager_track_id": MANAGER_TRACK_ID,
        "scene_id": SCENE_ID,
        "scene_count": SCENE_COUNT,
        "dataset_root": Path(dataset_root).resolve().as_posix(),
        "passes": list(PASS_IDS),
        "polarizations": list(POLARIZATIONS),
        "source_file_count": SOURCE_FILE_COUNT,
        "inventoried_source_file_count": INVENTORIED_SOURCE_FILE_COUNT,
        "payload_audited_source_file_count": PAYLOAD_AUDITED_SOURCE_FILE_COUNT,
        "sealed_test_source_file_count": SEALED_TEST_SOURCE_FILE_COUNT,
        "payload_audited_sector_ids": list(PAYLOAD_AUDITED_SECTOR_IDS),
        "sealed_test_sector_ids": list(SEALED_TEST_SECTOR_IDS),
        "test_opened": False,
        "corrections_applied": False,
        "frequency_policy": "native_per_shard_no_trim_no_padding",
        "row_alignment_policy": "native_observations_no_cross_polarization_row_stacking",
        "split": split.as_dict(),
        "support": support_contract(),
        "shards": serialized_shards,
    }
    compatibility = validate_joint_manifest(manifest)

    total_views = sum(
        int(shard.get("payload_audited_view_count", 0)) for shard in shards
    )
    total_complex_samples = sum(
        int(shard.get("payload_audited_complex_sample_count", 0))
        for shard in shards
    )
    manifest["totals"] = {
        "shards": len(serialized_shards),
        "inventoried_source_files": INVENTORIED_SOURCE_FILE_COUNT,
        "payload_audited_source_files": PAYLOAD_AUDITED_SOURCE_FILE_COUNT,
        "sealed_test_source_files": SEALED_TEST_SOURCE_FILE_COUNT,
        "payload_audited_native_observations": total_views,
        "payload_audited_native_complex_samples": total_complex_samples,
    }
    manifest["contract_compatibility"] = {
        "validator": "rift.gotcha_joint_fullpol.validate_joint_manifest",
        "passed": True,
        "summary": compatibility,
    }
    validate_gate0_inventory_manifest(manifest)
    return manifest


def require_slurm_allocation(environ: Mapping[str, str] | None = None) -> str:
    """Return the Slurm job ID or reject a login/local full scan."""

    environ = os.environ if environ is None else environ
    job_id = str(environ.get("SLURM_JOB_ID", "")).strip()
    if not job_id:
        raise RuntimeError(
            "A complete 11,520-file GOTCHA audit must run inside a Slurm "
            "allocation (SLURM_JOB_ID is absent)."
        )
    return job_id


def audit_dataset(
    dataset_root: str | os.PathLike[str],
    *,
    require_allocation: bool = True,
    allocation_environ: Mapping[str, str] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the complete sealed 8-pass × 4-pol × 360-file Gate 0 audit."""

    if require_allocation:
        require_slurm_allocation(allocation_environ)
    root = Path(dataset_root).resolve(strict=True)

    shards = []
    resolved_sources: set[str] = set()
    for pass_id in PASS_IDS:
        for polarization in POLARIZATIONS:
            shard = audit_shard(root, pass_id, polarization)
            for source_file in shard["source_files"]:
                resolved = Path(source_file).resolve(strict=True).as_posix()
                if resolved in resolved_sources:
                    raise AuditError(f"raw source file is reused: {resolved}")
                resolved_sources.add(resolved)
            shards.append(shard)
            if progress is not None:
                progress(
                    f"audited {shard['shard_id']}: "
                    f"inventoried_files={len(shard['source_files'])}, "
                    f"payload_audited_files={shard['payload_audited_source_file_count']}, "
                    f"sealed_test_files={shard['sealed_test_source_file_count']}, "
                    f"payload_audited_views={shard['payload_audited_view_count']}, "
                    f"frequencies={shard['native_frequency_count']}"
                )

    if len(resolved_sources) != SOURCE_FILE_COUNT:
        raise AuditError(
            f"expected {SOURCE_FILE_COUNT} globally unique files, "
            f"found {len(resolved_sources)}"
        )
    return build_joint_inventory_manifest(root, shards)


def atomic_write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    """Atomically publish JSON without any overwrite race.

    The flushed temporary file and destination are siblings.  ``os.link`` is
    atomic and fails if the destination exists, unlike a precheck followed by
    ``os.replace``.  Removing the temporary name leaves the published hard link
    intact.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing artifact: {destination}")

    temporary_path = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
                stream,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite concurrently created artifact: {destination}"
            ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def validate_production_cli_paths(
    dataset_root: str | os.PathLike[str],
    output: str | os.PathLike[str],
) -> tuple[Path, Path]:
    """Reject CLI attempts outside the two frozen production paths."""

    dataset_root = Path(dataset_root).resolve()
    output = Path(output).resolve()
    expected_dataset_root = DEFAULT_DATASET_ROOT.resolve()
    expected_output = PRODUCTION_OUTPUT_PATH.resolve()
    if dataset_root != expected_dataset_root:
        raise AuditError(
            f"production dataset root must be {DEFAULT_DATASET_ROOT}, got {dataset_root}"
        )
    if output != expected_output:
        raise AuditError(
            f"production output path must be {PRODUCTION_OUTPUT_PATH}, got {output}"
        )
    return dataset_root, output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="AFRL SDMS root containing extracted/data/GOTCHA (PACE default)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new JSON inventory path; an existing file is never overwritten",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root, output = validate_production_cli_paths(
        args.dataset_root, args.output
    )
    manifest = audit_dataset(
        dataset_root, progress=lambda message: print(message, flush=True)
    )
    atomic_write_json(output, manifest)
    print(
        json.dumps(
            {
                "gate_id": GATE_ID,
                "output": output.as_posix(),
                "inventoried_source_file_count": manifest[
                    "inventoried_source_file_count"
                ],
                "payload_audited_source_file_count": manifest[
                    "payload_audited_source_file_count"
                ],
                "sealed_test_source_file_count": manifest[
                    "sealed_test_source_file_count"
                ],
                "shards": len(manifest["shards"]),
                "test_opened": manifest["test_opened"],
                "corrections_applied": manifest["corrections_applied"],
                "contract_compatible": manifest["contract_compatibility"]["passed"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
