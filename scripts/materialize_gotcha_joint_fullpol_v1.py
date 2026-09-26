#!/usr/bin/env python3
"""Gate 1 materializer for the GOTCHA Joint-8 full-polarization dataset.

The materialized dataset is intentionally a directory of 32 independent,
uncompressed NPZ archives.  It never trims, pads, resamples, or stacks the
native frequency grids across pass/polarization shards.  HH/VV autofocus
arrays are copied as raw per-view values but are not applied; HV/VH carry an
explicit no-official-autofocus state.

The command-line entrypoint is a full conversion and therefore refuses to run
outside a Slurm allocation.  Unit tests call the pure materialization helpers
with synthetic records and an injected loader; that does not relax the CLI.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence
import zipfile

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rift.gotcha_joint_fullpol import (  # noqa: E402
    CO_POLARIZATIONS,
    MANIFEST_SCHEMA,
    PASS_IDS,
    POLARIZATIONS,
    SCENE_ID,
    SECTOR_COUNT,
    SECTOR_IDS,
    SHARD_COUNT,
    SHARD_IDS,
    SOURCE_FILE_COUNT,
    SPLIT_SEED,
    build_sector_split,
    canonical_shard_id,
    support_contract,
    validate_joint_manifest,
)


OUTPUT_ROOT_NAME = "converted_v3_joint8_fullpol"
MANIFEST_FILENAME = "manifest.json"
MATERIALIZATION_SCHEMA = "rift_gotcha_joint8_fullpol_materialization_v1"
ARCHIVE_SCHEMA = "rift_gotcha_joint8_fullpol_native_shard_v1"
MANAGER_TRACK_ID = "rift_publicradar_gotcha_joint_v1"
GATE0_INVENTORY_SCHEMA = "rift_gotcha_joint_fullpol_raw_inventory_v1"
GATE0_ID = "joint8_fullpol_dataaudit_v1"
FROZEN_DATASET_ROOT = Path(
    "/storage/scratch1/1/dbao31/datasets/rift_public_radar/afrl_sdms_20260814"
)
FROZEN_GATE0_JSON = Path(
    "/storage/project/r-jromberg3-0/dbao31/RIFT/experiment_state/"
    "public_radar_tuning/gotcha_joint8_fullpol_v1/"
    "joint8_fullpol_dataaudit_v1.json"
)
FROZEN_OUTPUT_ROOT = FROZEN_DATASET_ROOT / OUTPUT_ROOT_NAME

_SEALED_SPLIT = build_sector_split()
PAYLOAD_SECTOR_IDS = tuple(
    sector_id
    for sector_id in SECTOR_IDS
    if _SEALED_SPLIT.role_for(sector_id) != "test"
)
SEALED_TEST_SECTOR_IDS = tuple(
    sector_id
    for sector_id in SECTOR_IDS
    if _SEALED_SPLIT.role_for(sector_id) == "test"
)
PAYLOAD_SECTOR_COUNT = len(PAYLOAD_SECTOR_IDS)
SEALED_TEST_SECTOR_COUNT = len(SEALED_TEST_SECTOR_IDS)
PAYLOAD_AUDITED_SOURCE_FILE_COUNT = SHARD_COUNT * PAYLOAD_SECTOR_COUNT
SEALED_TEST_SOURCE_FILE_COUNT = SHARD_COUNT * SEALED_TEST_SECTOR_COUNT

if (
    PAYLOAD_SECTOR_COUNT != 324
    or SEALED_TEST_SECTOR_COUNT != 36
    or PAYLOAD_AUDITED_SOURCE_FILE_COUNT != 10_368
    or SEALED_TEST_SOURCE_FILE_COUNT != 1_152
):
    raise AssertionError("sealed Gate-1 source counts changed")

_NO_DEFAULT = object()
_MISSING = object()
_PATH_KEYS = ("source_path", "path", "mat_path", "file_path", "source_file")
_PASS_KEYS = ("pass_id", "pass", "pass_number")
_POLARIZATION_KEYS = ("polarization", "pol", "channel")
_SECTOR_KEYS = ("sector_id", "sector", "azimuth_sector", "az_sector")
_RESPONSE_LAYOUT_KEYS = ("response_layout", "fp_layout", "matrix_layout")

_ARCHIVE_ARRAY_KEYS = {
    "response",
    "frequencies_hz",
    "x",
    "y",
    "z",
    "r0",
    "th",
    "phi",
    "sector_id",
    "pulse_index",
    "pass_id",
    "polarization",
    "role",
    "r_correct_raw",
    "ph_correct_raw",
    "autofocus_available",
    "autofocus_applied",
    "autofocus_state",
    "metadata_json",
}


@dataclass(frozen=True)
class SourceRecord:
    """One audited raw MAT source, keyed by pass/polarization/sector."""

    source_path: str
    pass_id: int
    polarization: str
    sector_id: int
    response_layout: str | None = None
    sector_role: str | None = None
    payload_opened: bool | None = None
    expected_pulse_count: int | None = None
    expected_frequency_count: int | None = None

    @property
    def shard_id(self) -> str:
        return canonical_shard_id(self.pass_id, self.polarization)


@dataclass(frozen=True)
class NativeAcquisition:
    """One sector's native phase history normalized only to ``[view, F]``."""

    response: np.ndarray
    frequencies_hz: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    r0: np.ndarray
    th: np.ndarray
    phi: np.ndarray
    pulse_index: np.ndarray
    r_correct_raw: np.ndarray | None
    ph_correct_raw: np.ndarray | None
    source_response_shape: tuple[int, ...]
    source_response_layout: str
    pulse_index_source: str

    @property
    def view_count(self) -> int:
        return int(self.response.shape[0])

    @property
    def frequency_count(self) -> int:
        return int(self.response.shape[1])


@dataclass(frozen=True)
class Gate0ShardFacts:
    """Scientific facts audited by Gate 0 for one native shard."""

    shard_id: str
    pass_id: int
    polarization: str
    payload_sector_ids: tuple[int, ...]
    pulse_counts: tuple[int, ...]
    frequencies_hz: tuple[float, ...]
    frequency_dtype: str
    response_dtype: str
    geometry_dtype: str
    autofocus_dtype: str | None
    r_correct_extrema: tuple[float, float] | None
    ph_correct_extrema: tuple[float, float] | None

    def pulse_count_for(self, sector_id: int) -> int:
        try:
            index = self.payload_sector_ids.index(int(sector_id))
        except ValueError as exc:
            raise ValueError(
                f"{self.shard_id} sector {sector_id} is not payload-audited"
            ) from exc
        return int(self.pulse_counts[index])

    def as_dict(self) -> dict[str, Any]:
        def extrema(value: tuple[float, float] | None) -> dict[str, float] | None:
            if value is None:
                return None
            return {"minimum": value[0], "maximum": value[1]}

        return {
            "shard_id": self.shard_id,
            "pass_id": self.pass_id,
            "polarization": self.polarization,
            "payload_sector_ids": list(self.payload_sector_ids),
            "pulse_counts": list(self.pulse_counts),
            "frequencies_hz": list(self.frequencies_hz),
            "frequency_count": len(self.frequencies_hz),
            "frequency_dtype": self.frequency_dtype,
            "response_dtype": self.response_dtype,
            "geometry_dtype": self.geometry_dtype,
            "autofocus_dtype": self.autofocus_dtype,
            "r_correct_extrema": extrema(self.r_correct_extrema),
            "ph_correct_extrema": extrema(self.ph_correct_extrema),
        }


def _first(mapping: Mapping[str, Any], keys: Sequence[str], default: Any = _NO_DEFAULT) -> Any:
    lowered = {str(key).lower(): value for key, value in mapping.items()}
    for key in keys:
        if key.lower() in lowered:
            return lowered[key.lower()]
    if default is _NO_DEFAULT:
        raise KeyError(f"none of {tuple(keys)} is present")
    return default


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    """Convert common SciPy MATLAB struct representations into mappings."""

    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    if isinstance(value, Mapping):
        return value
    field_names = getattr(value, "_fieldnames", None)
    if field_names:
        return {name: getattr(value, name) for name in field_names}
    if isinstance(value, np.void) and value.dtype.names:
        return {name: value[name] for name in value.dtype.names}
    return None


def _record_containers(payload: Any) -> Iterable[tuple[Any, Mapping[str, Any]]]:
    """Yield record payloads plus shard-level inherited values.

    Gate 0 may publish either a flat ``records/files/entries`` sequence or a
    ``shards`` sequence whose children own those records.  Supporting only
    these two small shapes keeps the adapter tolerant without guessing at
    scientific content.
    """

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        for item in payload:
            yield item, {}
        return
    if not isinstance(payload, Mapping):
        raise ValueError("Gate-0 inventory must be a mapping or record sequence")

    for key in ("records", "files", "entries"):
        value = payload.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                yield item, {}
            return

    inventory = payload.get("inventory")
    if inventory is not None:
        yield from _record_containers(inventory)
        return

    shards = payload.get("shards")
    if isinstance(shards, Sequence) and not isinstance(shards, (str, bytes)):
        for shard in shards:
            if not isinstance(shard, Mapping):
                raise ValueError("every Gate-0 shard must be a mapping")
            inherited = {
                key: value
                for key, value in shard.items()
                if str(key).lower()
                in set(_PASS_KEYS + _POLARIZATION_KEYS + _RESPONSE_LAYOUT_KEYS)
            }
            children = _first(shard, ("records", "files", "entries"), default=None)
            source_files = shard.get("source_files")
            file_records = shard.get("file_records")
            if children is None and isinstance(source_files, Sequence) and not isinstance(
                source_files, (str, bytes)
            ):
                if file_records is None:
                    children = list(source_files)
                elif isinstance(file_records, Sequence) and not isinstance(
                    file_records, (str, bytes)
                ) and len(file_records) == len(source_files):
                    children = []
                    for source_file, file_record in zip(source_files, file_records):
                        if not isinstance(file_record, Mapping):
                            raise ValueError("Gate-0 file_records entries must be mappings")
                        combined = dict(file_record)
                        combined["source_file"] = str(source_file)
                        children.append(combined)
                else:
                    raise ValueError(
                        "Gate-0 source_files and file_records must have equal lengths"
                    )
            if not isinstance(children, Sequence) or isinstance(children, (str, bytes)):
                raise ValueError(
                    "every Gate-0 shard must contain records/files/entries or source_files"
                )
            for item in children:
                yield item, inherited
        return
    raise ValueError("Gate-0 inventory has no records/files/entries or shards")


def _identifier_from_path(path: str, kind: str) -> Any:
    normalized = str(path).replace("\\", "/")
    if kind == "pass":
        match = re.search(r"(?:^|[^a-z0-9])pass[_-]?0*([1-8])(?:[^0-9]|$)", normalized, re.I)
        return int(match.group(1)) if match else None
    if kind == "polarization":
        matches = re.findall(r"(?:^|[^a-z])(hh|hv|vh|vv)(?:[^a-z]|$)", normalized, re.I)
        return matches[-1].lower() if matches else None
    if kind == "sector":
        match = re.search(r"(?:az|sector)[_-]?0*(\d{1,3})(?:[^0-9]|$)", normalized, re.I)
        return int(match.group(1)) if match else None
    raise AssertionError(f"unknown identifier kind {kind}")


def _coerce_source_record(item: Any, inherited: Mapping[str, Any]) -> SourceRecord:
    if isinstance(item, (str, os.PathLike)):
        values: Mapping[str, Any] = {"path": str(item)}
    elif isinstance(item, Mapping):
        values = item
    else:
        raise ValueError("each Gate-0 source record must be a path or mapping")

    try:
        path = str(_first(values, _PATH_KEYS))
    except KeyError as exc:
        raise ValueError("Gate-0 source record lacks a path") from exc
    if not path.strip():
        raise ValueError("Gate-0 source path must not be empty")

    pass_value = _first(values, _PASS_KEYS, default=_MISSING)
    if pass_value is _MISSING:
        pass_value = _first(inherited, _PASS_KEYS, default=None)
    if pass_value is None:
        pass_value = _identifier_from_path(path, "pass")

    polarization_value = _first(values, _POLARIZATION_KEYS, default=_MISSING)
    if polarization_value is _MISSING:
        polarization_value = _first(inherited, _POLARIZATION_KEYS, default=None)
    if polarization_value is None:
        polarization_value = _identifier_from_path(path, "polarization")

    sector_value = _first(values, _SECTOR_KEYS, default=None)
    if sector_value is None:
        sector_value = _identifier_from_path(path, "sector")

    try:
        pass_id = int(pass_value)
        polarization = str(polarization_value).lower()
        sector_id = int(sector_value)
        canonical_shard_id(pass_id, polarization)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cannot identify pass/polarization/sector for {path}") from exc
    if sector_id not in SECTOR_IDS:
        raise ValueError(f"sector for {path} must be in 1..360")

    layout = _first(values, _RESPONSE_LAYOUT_KEYS, default=_MISSING)
    if layout is _MISSING:
        layout = _first(inherited, _RESPONSE_LAYOUT_KEYS, default=None)
    if isinstance(layout, Mapping):
        layout = _first(layout, ("response_layout", "fp_layout", "orientation"), default=None)
    sector_role = _first(values, ("sector_role", "role"), default=None)
    payload_opened = _first(values, ("payload_opened",), default=None)
    pulse_count = _first(values, ("pulse_count",), default=None)
    frequency_count = _first(values, ("frequency_count",), default=None)
    return SourceRecord(
        source_path=path,
        pass_id=pass_id,
        polarization=polarization,
        sector_id=sector_id,
        response_layout=None if layout is None else str(layout),
        sector_role=None if sector_role is None else str(sector_role),
        payload_opened=None if payload_opened is None else bool(payload_opened),
        expected_pulse_count=None if pulse_count is None else int(pulse_count),
        expected_frequency_count=None
        if frequency_count is None
        else int(frequency_count),
    )


def adapt_gate0_inventory(payload: Any, *, require_full: bool = True) -> list[SourceRecord]:
    """Adapt the small accepted Gate-0 JSON variants into canonical records."""

    if isinstance(payload, Mapping):
        for flag in ("complete", "audit_complete"):
            if flag in payload and payload[flag] is not True:
                raise ValueError(f"Gate-0 inventory has {flag}=false")
        state = str(payload.get("state", payload.get("status", ""))).lower()
        if any(token in state for token in ("fail", "incomplete", "pending", "running")):
            raise ValueError(f"Gate-0 inventory is not complete: {state!r}")

    records = [
        _coerce_source_record(item, inherited)
        for item, inherited in _record_containers(payload)
    ]
    seen_keys: dict[tuple[int, str, int], str] = {}
    seen_paths: set[str] = set()
    for record in records:
        key = (record.pass_id, record.polarization, record.sector_id)
        if key in seen_keys:
            raise ValueError(
                f"Gate-0 inventory repeats {key}: {seen_keys[key]} and {record.source_path}"
            )
        if record.source_path in seen_paths:
            raise ValueError(f"Gate-0 inventory reuses source path {record.source_path}")
        seen_keys[key] = record.source_path
        seen_paths.add(record.source_path)

    if require_full:
        expected = {
            (pass_id, polarization, sector_id)
            for pass_id in PASS_IDS
            for polarization in POLARIZATIONS
            for sector_id in SECTOR_IDS
        }
        actual = set(seen_keys)
        if len(records) != SOURCE_FILE_COUNT or actual != expected:
            missing = sorted(expected - actual)[:8]
            extra = sorted(actual - expected)[:8]
            raise ValueError(
                "Gate-0 inventory must contain exactly 11,520 canonical records; "
                f"count={len(records)}, missing_head={missing}, extra_head={extra}"
            )
        if isinstance(payload, Mapping):
            declared = payload.get("source_file_count", payload.get("file_count"))
            if declared is not None and int(declared) != SOURCE_FILE_COUNT:
                raise ValueError("Gate-0 declared source-file count is not 11,520")
    return sorted(records, key=lambda row: (row.pass_id, POLARIZATIONS.index(row.polarization), row.sector_id))


def _gate0_extrema(
    autofocus: Mapping[str, Any], key: str, shard_id: str
) -> tuple[float, float]:
    value = autofocus.get(key)
    if not isinstance(value, Mapping) or set(value) != {"minimum", "maximum"}:
        raise ValueError(f"{shard_id} autofocus.{key} is not an exact extrema record")
    minimum = float(value["minimum"])
    maximum = float(value["maximum"])
    if not np.isfinite((minimum, maximum)).all() or minimum > maximum:
        raise ValueError(f"{shard_id} autofocus.{key} is invalid")
    return minimum, maximum


def adapt_gate0_shard_facts(payload: Mapping[str, Any]) -> dict[str, Gate0ShardFacts]:
    """Retain every Gate-0 payload fact needed to bind Gate-1 bytes."""

    shards = payload.get("shards")
    if not isinstance(shards, Sequence) or isinstance(shards, (str, bytes)):
        raise ValueError("Gate-0 shards must be a sequence")
    result: dict[str, Gate0ShardFacts] = {}
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise ValueError("Gate-0 shard facts must be mappings")
        pass_id = int(shard.get("pass_id", -1))
        polarization = str(shard.get("polarization", "")).lower()
        shard_id = canonical_shard_id(pass_id, polarization)
        if shard.get("shard_id") != shard_id or shard_id in result:
            raise ValueError("Gate-0 shard fact identity is invalid or repeated")
        pulse_counts = tuple(
            int(value)
            for value in shard.get(
                "payload_audited_native_pulse_counts_by_sector", ()
            )
        )
        frequencies = tuple(float(value) for value in shard.get("native_frequency_hz", ()))
        frequency_dtype = str(shard.get("native_frequency_dtype", ""))
        response_dtype = str(shard.get("fp_dtype", ""))
        geometry_dtype = str(shard.get("geometry_dtype", ""))
        autofocus = shard.get("autofocus")
        if not isinstance(autofocus, Mapping):
            raise ValueError(f"{shard_id} lacks Gate-0 autofocus facts")
        if polarization in CO_POLARIZATIONS:
            # Gate 0's MAT auditor enforces EXPECTED_FLOAT_DTYPE for both raw
            # autofocus arrays before recording their extrema.
            autofocus_dtype = "float32"
            r_extrema = _gate0_extrema(autofocus, "r_correct_extrema", shard_id)
            ph_extrema = _gate0_extrema(autofocus, "ph_correct_extrema", shard_id)
        else:
            autofocus_dtype = None
            r_extrema = None
            ph_extrema = None
            if (
                "r_correct_extrema" in autofocus
                or "ph_correct_extrema" in autofocus
            ):
                raise ValueError(f"{shard_id} cross-pol must not carry autofocus extrema")
        result[shard_id] = Gate0ShardFacts(
            shard_id=shard_id,
            pass_id=pass_id,
            polarization=polarization,
            payload_sector_ids=PAYLOAD_SECTOR_IDS,
            pulse_counts=pulse_counts,
            frequencies_hz=frequencies,
            frequency_dtype=frequency_dtype,
            response_dtype=response_dtype,
            geometry_dtype=geometry_dtype,
            autofocus_dtype=autofocus_dtype,
            r_correct_extrema=r_extrema,
            ph_correct_extrema=ph_extrema,
        )
    if set(result) != set(SHARD_IDS):
        raise ValueError("Gate-0 shard facts do not cover the exact 32 shards")
    return result


def _expected_source_path(
    pass_id: int, polarization: str, sector_id: int
) -> Path:
    disc = "GOTCHA-CP_Disc1" if int(pass_id) <= 7 else "GOTCHA-CP_Disc2"
    return (
        FROZEN_DATASET_ROOT
        / "extracted"
        / "data"
        / "GOTCHA"
        / disc
        / "DATA"
        / f"pass{int(pass_id)}"
        / str(polarization).upper()
        / (
            f"data_3dsar_pass{int(pass_id)}_az{int(sector_id):03d}_"
            f"{str(polarization).upper()}.mat"
        )
    )


def validate_authentic_gate0(
    payload: Any,
    *,
    gate0_inventory_path: os.PathLike[str] | str | None,
    enforce_production_paths: bool,
) -> dict[str, Any]:
    """Validate the exact Gate-0 producer contract before any output write.

    This is intentionally called by :func:`materialize_inventory` itself.
    The Slurm wrapper repeats some checks as operational defense in depth, but
    invoking this module directly cannot bypass the scientific prerequisite.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("Gate-0 artifact must be a JSON mapping")
    compatibility = validate_joint_manifest(payload)
    required = {
        "schema": MANIFEST_SCHEMA,
        "inventory_schema": GATE0_INVENTORY_SCHEMA,
        "gate_id": GATE0_ID,
        "scene_id": SCENE_ID,
        "manager_track_id": MANAGER_TRACK_ID,
        "scene_count": 1,
        "source_file_count": SOURCE_FILE_COUNT,
        "inventoried_source_file_count": SOURCE_FILE_COUNT,
        "payload_audited_source_file_count": PAYLOAD_AUDITED_SOURCE_FILE_COUNT,
        "sealed_test_source_file_count": SEALED_TEST_SOURCE_FILE_COUNT,
        "test_opened": False,
        "corrections_applied": False,
        "frequency_policy": "native_per_shard_no_trim_no_padding",
        "row_alignment_policy": (
            "native_observations_no_cross_polarization_row_stacking"
        ),
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"Gate-0 authentic field {key} must be {expected!r}, "
                f"got {payload.get(key)!r}"
            )
    if str(payload.get("dataset_root")) != FROZEN_DATASET_ROOT.as_posix():
        raise ValueError("Gate-0 dataset_root is not the frozen AFRL SDMS root")
    if list(payload.get("passes", ())) != list(PASS_IDS):
        raise ValueError("Gate-0 passes are not exactly 1..8")
    if list(payload.get("polarizations", ())) != list(POLARIZATIONS):
        raise ValueError("Gate-0 polarization vocabulary/order changed")

    if list(payload.get("payload_audited_sector_ids", ())) != list(
        PAYLOAD_SECTOR_IDS
    ):
        raise ValueError("Gate-0 top-level payload sector membership is not exact")
    if list(payload.get("sealed_test_sector_ids", ())) != list(
        SEALED_TEST_SECTOR_IDS
    ):
        raise ValueError("Gate-0 top-level sealed-test membership is not exact")

    totals = payload.get("totals")
    if not isinstance(totals, Mapping):
        raise ValueError("Gate-0 totals must be present")
    expected_totals = {
        "shards": SHARD_COUNT,
        "inventoried_source_files": SOURCE_FILE_COUNT,
        "payload_audited_source_files": PAYLOAD_AUDITED_SOURCE_FILE_COUNT,
        "sealed_test_source_files": SEALED_TEST_SOURCE_FILE_COUNT,
    }
    for key, expected in expected_totals.items():
        if int(totals.get(key, -1)) != expected:
            raise ValueError(f"Gate-0 totals.{key} must be {expected}")
    contract = payload.get("contract_compatibility")
    if not isinstance(contract, Mapping):
        raise ValueError("Gate-0 contract_compatibility must be present")
    if (
        contract.get("validator")
        != "rift.gotcha_joint_fullpol.validate_joint_manifest"
        or contract.get("passed") is not True
    ):
        raise ValueError("Gate-0 contract compatibility is not authentic/passing")
    if contract.get("summary") != compatibility:
        raise ValueError("Gate-0 contract compatibility summary is stale")

    shards = payload.get("shards")
    if not isinstance(shards, Sequence) or isinstance(shards, (str, bytes)):
        raise ValueError("Gate-0 shards must be a sequence")
    if len(shards) != SHARD_COUNT:
        raise ValueError("Gate-0 must contain exactly 32 shards")
    expected_roles = list(_SEALED_SPLIT.role_by_sector)
    total_views = 0
    total_complex_samples = 0
    for expected_shard_id, shard in zip(SHARD_IDS, shards):
        if not isinstance(shard, Mapping):
            raise ValueError("Gate-0 shard must be a mapping")
        pass_id = int(shard.get("pass_id", -1))
        polarization = str(shard.get("polarization", "")).lower()
        shard_id = canonical_shard_id(pass_id, polarization)
        if shard_id != expected_shard_id or shard.get("shard_id") != shard_id:
            raise ValueError("Gate-0 shards are not in exact canonical order")
        if list(shard.get("sector_ids", ())) != list(SECTOR_IDS):
            raise ValueError(f"{shard_id} lacks exact 1..360 sector membership")
        if list(shard.get("sector_roles", ())) != expected_roles:
            raise ValueError(f"{shard_id} sector roles differ from the sealed split")
        shard_membership = {
            "source_file_count": SECTOR_COUNT,
            "inventoried_source_file_count": SECTOR_COUNT,
            "payload_audited_sector_count": PAYLOAD_SECTOR_COUNT,
            "payload_audited_source_file_count": PAYLOAD_SECTOR_COUNT,
            "sealed_test_sector_count": SEALED_TEST_SECTOR_COUNT,
            "sealed_test_source_file_count": SEALED_TEST_SECTOR_COUNT,
        }
        for key, expected in shard_membership.items():
            if int(shard.get(key, -1)) != expected:
                raise ValueError(f"{shard_id}.{key} must be {expected}")
        if list(shard.get("payload_audited_sector_ids", ())) != list(
            PAYLOAD_SECTOR_IDS
        ):
            raise ValueError(f"{shard_id} payload membership is not exact")
        if list(shard.get("sealed_test_sector_ids", ())) != list(
            SEALED_TEST_SECTOR_IDS
        ):
            raise ValueError(f"{shard_id} sealed-test membership is not exact")
        if shard.get("corrections_applied") is not False:
            raise ValueError(f"{shard_id} must record corrections_applied=false")

        source_files = shard.get("source_files")
        file_records = shard.get("file_records")
        pulse_counts = shard.get(
            "payload_audited_native_pulse_counts_by_sector"
        )
        if (
            not isinstance(source_files, Sequence)
            or isinstance(source_files, (str, bytes))
            or len(source_files) != SECTOR_COUNT
        ):
            raise ValueError(f"{shard_id} must inventory exactly 360 source paths")
        if (
            not isinstance(file_records, Sequence)
            or isinstance(file_records, (str, bytes))
            or len(file_records) != SECTOR_COUNT
        ):
            raise ValueError(f"{shard_id} must inventory exactly 360 file records")
        if (
            not isinstance(pulse_counts, Sequence)
            or isinstance(pulse_counts, (str, bytes))
            or len(pulse_counts) != PAYLOAD_SECTOR_COUNT
        ):
            raise ValueError(f"{shard_id} must audit exactly 324 pulse counts")

        frequency_count = int(shard.get("native_frequency_count", -1))
        frequency_values = shard.get("native_frequency_hz")
        if frequency_count <= 0 or not isinstance(frequency_values, Sequence):
            raise ValueError(f"{shard_id} native frequency inventory is invalid")
        frequency = np.asarray(frequency_values, dtype=np.float64)
        if (
            frequency.shape != (frequency_count,)
            or not np.isfinite(frequency).all()
            or not np.all(np.diff(frequency) > 0)
        ):
            raise ValueError(f"{shard_id} native frequency values are invalid")
        dtype_required = {
            "native_frequency_dtype": "float32",
            "fp_dtype": "complex64",
            "geometry_dtype": "float32",
        }
        for key, expected in dtype_required.items():
            if shard.get(key) != expected:
                raise ValueError(f"{shard_id}.{key} must be {expected}")

        normalized_pulse_counts = []
        official_autofocus = polarization in CO_POLARIZATIONS
        payload_pulse_by_sector = dict(zip(PAYLOAD_SECTOR_IDS, pulse_counts))
        for sector_id, source_file, file_record in zip(
            SECTOR_IDS, source_files, file_records
        ):
            if not isinstance(file_record, Mapping):
                raise ValueError(f"{shard_id} file record is not a mapping")
            role = _SEALED_SPLIT.role_for(sector_id)
            expected_record = {
                "sector_id": sector_id,
                "sector_role": role,
                "payload_opened": role != "test",
                "corrections_applied": False,
            }
            for key, expected in expected_record.items():
                if file_record.get(key) != expected:
                    raise ValueError(
                        f"{shard_id} sector {sector_id} has invalid {key}"
                    )
            if role == "test":
                if set(file_record) != {
                    "sector_id",
                    "sector_role",
                    "payload_opened",
                    "corrections_applied",
                }:
                    raise ValueError(
                        f"{shard_id} sealed sector {sector_id} leaks payload fields"
                    )
            else:
                pulse_count = int(payload_pulse_by_sector[sector_id])
                normalized_pulse_counts.append(pulse_count)
                if pulse_count <= 0:
                    raise ValueError(f"{shard_id} has a nonpositive pulse count")
                payload_fields = {
                    "frequency_count": frequency_count,
                    "pulse_count": pulse_count,
                    "fp_shape": [frequency_count, pulse_count],
                    "autofocus_present": official_autofocus,
                }
                for key, expected in payload_fields.items():
                    if file_record.get(key) != expected:
                        raise ValueError(
                            f"{shard_id} sector {sector_id} has invalid {key}"
                        )
                if set(file_record) != {
                    "sector_id",
                    "sector_role",
                    "payload_opened",
                    "frequency_count",
                    "pulse_count",
                    "fp_shape",
                    "autofocus_present",
                    "corrections_applied",
                }:
                    raise ValueError(
                        f"{shard_id} sector {sector_id} payload schema changed"
                    )
            if enforce_production_paths:
                expected_path = _expected_source_path(
                    pass_id, polarization, sector_id
                ).as_posix()
                if str(source_file) != expected_path:
                    raise ValueError(
                        f"{shard_id} sector {sector_id} source path is not authentic"
                    )

        native_views = sum(normalized_pulse_counts)
        native_samples = native_views * frequency_count
        if int(shard.get("payload_audited_view_count", -1)) != native_views:
            raise ValueError(f"{shard_id} payload view count is inconsistent")
        if int(
            shard.get("payload_audited_complex_sample_count", -1)
        ) != native_samples:
            raise ValueError(
                f"{shard_id} payload complex-sample count is inconsistent"
            )
        total_views += native_views
        total_complex_samples += native_samples

        autofocus = shard.get("autofocus")
        if not isinstance(autofocus, Mapping):
            raise ValueError(f"{shard_id} autofocus inventory is missing")
        autofocus_required = {
            "official_available": official_autofocus,
            "present_in_all_payload_audited_source_files": official_autofocus,
            "source_shard_id": shard_id if official_autofocus else None,
            "source_polarization": polarization if official_autofocus else None,
            "range_field": "af.r_correct" if official_autofocus else None,
            "phase_field": "af.ph_correct" if official_autofocus else None,
            "payload_audited_source_file_count": PAYLOAD_SECTOR_COUNT,
            "applied": False,
            "policy": "audited_in_native_files_never_applied_by_gate0",
        }
        for key, expected in autofocus_required.items():
            if autofocus.get(key) != expected:
                raise ValueError(f"{shard_id} autofocus.{key} is not authentic")
        if official_autofocus:
            _gate0_extrema(autofocus, "r_correct_extrema", shard_id)
            _gate0_extrema(autofocus, "ph_correct_extrema", shard_id)
        elif (
            "r_correct_extrema" in autofocus
            or "ph_correct_extrema" in autofocus
        ):
            raise ValueError(f"{shard_id} cross-pol must not carry AF extrema")

    if int(totals.get("payload_audited_native_observations", -1)) != total_views:
        raise ValueError("Gate-0 total payload observations are inconsistent")
    if int(
        totals.get("payload_audited_native_complex_samples", -1)
    ) != total_complex_samples:
        raise ValueError("Gate-0 total payload complex samples are inconsistent")

    if enforce_production_paths:
        if gate0_inventory_path is None:
            raise ValueError("production materialization requires Gate-0 artifact path")
        if Path(gate0_inventory_path).as_posix() != FROZEN_GATE0_JSON.as_posix():
            raise ValueError("Gate-0 JSON path is not the frozen production path")
    return compatibility


def _source_archive_summary(payload: Any) -> list[dict[str, Any]]:
    """Carry Gate-0 archive paths/counts, deliberately excluding hashes."""

    if not isinstance(payload, Mapping):
        return []
    candidates = payload.get("source_archives", payload.get("archives", []))
    if not candidates and isinstance(payload.get("inventory"), Mapping):
        return _source_archive_summary(payload["inventory"])
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return []
    result = []
    for candidate in candidates:
        if isinstance(candidate, (str, os.PathLike)):
            result.append({"path": str(candidate), "file_count": None})
            continue
        if not isinstance(candidate, Mapping):
            continue
        path = _first(candidate, ("path", "archive_path", "source_path"), default=None)
        if path is None:
            continue
        count = _first(candidate, ("file_count", "member_count", "count"), default=None)
        result.append(
            {
                "path": str(path),
                "file_count": None if count is None else int(count),
            }
        )
    return result


def load_gate0_json(path: os.PathLike[str] | str) -> tuple[Any, list[SourceRecord]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload, adapt_gate0_inventory(payload, require_full=True)


def _select_mat_root(loaded: Mapping[str, Any]) -> Mapping[str, Any]:
    public = {key: value for key, value in loaded.items() if not str(key).startswith("__")}
    if _first(public, ("fp", "response"), default=_MISSING) is not _MISSING:
        return public
    candidates = []
    for value in public.values():
        mapping = _as_mapping(value)
        if mapping is not None and _first(mapping, ("fp", "response"), default=_MISSING) is not _MISSING:
            candidates.append(mapping)
    if len(candidates) != 1:
        raise ValueError(
            "MAT file must expose exactly one struct containing fp/response"
        )
    return candidates[0]


def _finite_vector(value: Any, name: str, length: int | None = None) -> np.ndarray:
    array = np.asarray(value).squeeze()
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1 or array.dtype == object or not np.isrealobj(array):
        raise ValueError(f"{name} must be a real one-dimensional vector")
    if length is not None and array.size != int(length):
        raise ValueError(f"{name} has {array.size} values for {length} views")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a nonfinite value")
    return np.array(array, copy=True)


def _response_view_frequency(
    fp: Any,
    frequency_count: int,
    layout_hint: str | None,
) -> tuple[np.ndarray, tuple[int, ...], str]:
    source = np.asarray(fp)
    source_shape = tuple(int(value) for value in source.shape)
    response = source.squeeze()
    if response.dtype == object or not np.iscomplexobj(response):
        raise ValueError("fp/response must be a numeric complex array")
    if response.ndim == 1:
        if response.size != frequency_count:
            raise ValueError("one-dimensional fp length disagrees with frequency")
        return np.array(response.reshape(1, -1), copy=True), source_shape, "single_view_frequency"
    if response.ndim != 2:
        raise ValueError("fp/response must squeeze to one or two dimensions")

    hint = "" if layout_hint is None else layout_hint.lower().replace(" ", "")
    view_frequency_hints = {
        "view,frequency",
        "view_by_frequency",
        "[view,f]",
        "[views,frequency]",
        "vf",
    }
    frequency_view_hints = {
        "frequency,view",
        "frequency_by_view",
        "[f,view]",
        "[frequency,views]",
        "fv",
    }
    if hint in view_frequency_hints:
        if response.shape[1] != frequency_count:
            raise ValueError("declared view-by-frequency fp layout is inconsistent")
        orientation = "view_frequency"
    elif hint in frequency_view_hints:
        if response.shape[0] != frequency_count:
            raise ValueError("declared frequency-by-view fp layout is inconsistent")
        response = response.T
        orientation = "frequency_view_transposed"
    elif response.shape[1] == frequency_count and response.shape[0] != frequency_count:
        orientation = "view_frequency"
    elif response.shape[0] == frequency_count and response.shape[1] != frequency_count:
        response = response.T
        orientation = "frequency_view_transposed"
    else:
        raise ValueError(
            "fp orientation is ambiguous or disagrees with the native frequency count"
        )
    if not np.isfinite(response.real).all() or not np.isfinite(response.imag).all():
        raise ValueError("fp/response contains a nonfinite value")
    return np.array(response, copy=True), source_shape, orientation


def load_native_mat(path: os.PathLike[str] | str, record: SourceRecord) -> NativeAcquisition:
    """Load one audited GOTCHA MAT file without applying any correction."""

    try:
        from scipy.io import loadmat
    except ModuleNotFoundError as exc:  # pragma: no cover - PACE/runtime diagnostic.
        raise ModuleNotFoundError("Gate-1 MAT conversion requires scipy") from exc

    try:
        loaded = loadmat(path, simplify_cells=True)
    except TypeError:  # Older SciPy fallback.
        loaded = loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError as exc:
        raise ValueError("MATLAB v7.3/HDF5 input requires a separately audited loader") from exc
    root = _select_mat_root(loaded)

    try:
        frequencies = _finite_vector(
            _first(root, ("freq", "frequency", "frequencies", "frequencies_hz")),
            "frequency",
        )
        response, source_shape, orientation = _response_view_frequency(
            _first(root, ("fp", "response")),
            int(frequencies.size),
            record.response_layout,
        )
    except KeyError as exc:
        raise ValueError(f"{record.source_path} lacks fp or frequency") from exc
    if frequencies.size == 0 or not np.all(np.diff(frequencies.astype(np.float64)) > 0):
        raise ValueError("frequency must be nonempty and strictly increasing")
    view_count = int(response.shape[0])

    geometry = {}
    for name in ("x", "y", "z", "r0", "th", "phi"):
        try:
            geometry[name] = _finite_vector(_first(root, (name,)), name, view_count)
        except KeyError as exc:
            raise ValueError(f"{record.source_path} lacks {name}") from exc

    pulse_value = _first(root, ("pulse_index", "pulse_indices", "pulse"), default=None)
    if pulse_value is None:
        pulse_index = np.arange(view_count, dtype=np.int32)
        pulse_index_source = "derived_zero_based_within_sector"
    else:
        pulse_raw = _finite_vector(pulse_value, "pulse_index", view_count)
        if not np.all(np.equal(pulse_raw, np.floor(pulse_raw))):
            raise ValueError("pulse_index values must be integers")
        pulse_index = pulse_raw.astype(np.int64, copy=False)
        pulse_index_source = "raw_mat_field"
    if np.unique(pulse_index).size != view_count:
        raise ValueError("pulse_index repeats within a sector")

    af_value = _first(root, ("af", "autofocus"), default=None)
    af = _as_mapping(af_value) if af_value is not None else None
    correction_root = af if af is not None else root
    r_value = _first(correction_root, ("r_correct",), default=None)
    ph_value = _first(correction_root, ("ph_correct",), default=None)
    if record.polarization in CO_POLARIZATIONS:
        if r_value is None or ph_value is None:
            raise ValueError(f"{record.shard_id} requires its own raw autofocus arrays")
        r_correct = _finite_vector(r_value, "r_correct", view_count)
        ph_correct = _finite_vector(ph_value, "ph_correct", view_count)
    else:
        for name, value in (("r_correct", r_value), ("ph_correct", ph_value)):
            if value is not None and np.asarray(value).size:
                raise ValueError(
                    f"{record.shard_id} must not receive official or borrowed {name}"
                )
        r_correct = None
        ph_correct = None

    return NativeAcquisition(
        response=response,
        frequencies_hz=frequencies,
        x=geometry["x"],
        y=geometry["y"],
        z=geometry["z"],
        r0=geometry["r0"],
        th=geometry["th"],
        phi=geometry["phi"],
        pulse_index=pulse_index,
        r_correct_raw=r_correct,
        ph_correct_raw=ph_correct,
        source_response_shape=source_shape,
        source_response_layout=orientation,
        pulse_index_source=pulse_index_source,
    )


def _validate_acquisition(acquisition: NativeAcquisition, record: SourceRecord) -> None:
    if not isinstance(acquisition, NativeAcquisition):
        raise ValueError("MAT loader must return NativeAcquisition")
    response = np.asarray(acquisition.response)
    frequency = np.asarray(acquisition.frequencies_hz)
    if response.ndim != 2 or response.shape[0] <= 0 or response.shape[1] <= 0:
        raise ValueError(f"{record.source_path} response must have shape [view, F]")
    if not np.iscomplexobj(response) or not np.isfinite(response.real).all() or not np.isfinite(response.imag).all():
        raise ValueError(f"{record.source_path} response must be finite complex")
    if frequency.ndim != 1 or frequency.shape[0] != response.shape[1]:
        raise ValueError(f"{record.source_path} frequency shape disagrees with response")
    if not np.isfinite(frequency).all() or not np.all(np.diff(frequency.astype(np.float64)) > 0):
        raise ValueError(f"{record.source_path} frequency must be finite and increasing")
    for name in ("x", "y", "z", "r0", "th", "phi", "pulse_index"):
        array = np.asarray(getattr(acquisition, name))
        if array.ndim != 1 or array.size != response.shape[0]:
            raise ValueError(f"{record.source_path} {name} is not view-aligned")
        if not np.isfinite(array).all():
            raise ValueError(f"{record.source_path} {name} is nonfinite")
    if np.unique(acquisition.pulse_index).size != response.shape[0]:
        raise ValueError(f"{record.source_path} pulse_index repeats")
    if record.polarization in CO_POLARIZATIONS:
        for name in ("r_correct_raw", "ph_correct_raw"):
            array = np.asarray(getattr(acquisition, name))
            if array.ndim != 1 or array.size != response.shape[0] or not np.isfinite(array).all():
                raise ValueError(f"{record.source_path} {name} is not view-aligned")
    elif acquisition.r_correct_raw is not None or acquisition.ph_correct_raw is not None:
        raise ValueError(f"{record.shard_id} cross-pol autofocus must be absent")


def _validate_acquisition_against_gate0(
    acquisition: NativeAcquisition,
    record: SourceRecord,
    facts: Gate0ShardFacts,
) -> None:
    """Bind one loaded acquisition to its Gate-0 audited scientific facts."""

    _validate_acquisition(acquisition, record)
    expected_role = _SEALED_SPLIT.role_for(record.sector_id)
    if expected_role == "test":
        raise ValueError(f"{record.shard_id} attempted to bind a sealed test payload")
    if record.sector_role != expected_role or record.payload_opened is not True:
        raise ValueError(f"{record.shard_id} sector {record.sector_id} Gate-0 role drifted")
    expected_pulse_count = facts.pulse_count_for(record.sector_id)
    if record.expected_pulse_count != expected_pulse_count:
        raise ValueError(
            f"{record.shard_id} sector {record.sector_id} Gate-0 pulse records disagree"
        )
    if acquisition.view_count != expected_pulse_count:
        raise ValueError(
            f"{record.shard_id} sector {record.sector_id} view count "
            f"{acquisition.view_count} differs from Gate-0 pulse count "
            f"{expected_pulse_count}"
        )

    expected_frequency = np.asarray(
        facts.frequencies_hz, dtype=np.dtype(facts.frequency_dtype)
    )
    if record.expected_frequency_count != expected_frequency.size:
        raise ValueError(
            f"{record.shard_id} sector {record.sector_id} Gate-0 frequency counts disagree"
        )
    actual_frequency = np.asarray(acquisition.frequencies_hz)
    if actual_frequency.dtype != np.dtype(facts.frequency_dtype):
        raise ValueError(
            f"{record.shard_id} frequency dtype differs from Gate 0"
        )
    if actual_frequency.shape != expected_frequency.shape:
        raise ValueError(
            f"{record.shard_id} frequency count differs from Gate 0"
        )
    if not np.array_equal(actual_frequency, expected_frequency):
        raise ValueError(
            f"{record.shard_id} frequency values differ from Gate 0"
        )
    if np.asarray(acquisition.response).dtype != np.dtype(facts.response_dtype):
        raise ValueError(f"{record.shard_id} response dtype differs from Gate 0")
    for name in ("x", "y", "z", "r0", "th", "phi"):
        if np.asarray(getattr(acquisition, name)).dtype != np.dtype(
            facts.geometry_dtype
        ):
            raise ValueError(
                f"{record.shard_id} {name} dtype differs from Gate 0"
            )
    if facts.autofocus_dtype is not None:
        for name in ("r_correct_raw", "ph_correct_raw"):
            if np.asarray(getattr(acquisition, name)).dtype != np.dtype(
                facts.autofocus_dtype
            ):
                raise ValueError(
                    f"{record.shard_id} {name} dtype differs from Gate 0"
                )


def _validate_autofocus_extrema_against_gate0(
    acquisitions: Sequence[NativeAcquisition], facts: Gate0ShardFacts
) -> None:
    if facts.polarization not in CO_POLARIZATIONS:
        return
    expected = {
        "r_correct_raw": facts.r_correct_extrema,
        "ph_correct_raw": facts.ph_correct_extrema,
    }
    for name, expected_extrema in expected.items():
        if expected_extrema is None:
            raise ValueError(f"{facts.shard_id} lacks Gate-0 {name} extrema")
        values = [np.asarray(getattr(item, name)) for item in acquisitions]
        observed = (
            float(min(float(np.min(value)) for value in values)),
            float(max(float(np.max(value)) for value in values)),
        )
        if observed != expected_extrema:
            raise ValueError(
                f"{facts.shard_id} {name} extrema {observed} differ from "
                f"Gate 0 {expected_extrema}"
            )


def _same_native_dtype(acquisitions: Sequence[NativeAcquisition], name: str) -> np.dtype:
    dtypes = {np.asarray(getattr(acquisition, name)).dtype.str for acquisition in acquisitions}
    if len(dtypes) != 1:
        raise ValueError(f"native dtype for {name} changes within one shard: {sorted(dtypes)}")
    return np.asarray(getattr(acquisitions[0], name)).dtype


def _build_shard_arrays(
    records: Sequence[SourceRecord],
    acquisitions: Sequence[NativeAcquisition],
    *,
    inventoried_records: Sequence[SourceRecord],
    gate0_facts: Gate0ShardFacts,
    gate0_inventory_path: str | None,
    materialization_slurm_job_id: str | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if (
        len(records) != PAYLOAD_SECTOR_COUNT
        or len(acquisitions) != PAYLOAD_SECTOR_COUNT
    ):
        raise ValueError("one shard payload must contain exactly 324 sectors")
    if len(inventoried_records) != SECTOR_COUNT:
        raise ValueError("one shard inventory must preserve exactly 360 sectors")
    first_record = records[0]
    shard_id = first_record.shard_id
    if [record.sector_id for record in records] != list(PAYLOAD_SECTOR_IDS):
        raise ValueError(f"{shard_id} payload must be the exact 324 train/validation sectors")
    if [record.sector_id for record in inventoried_records] != list(SECTOR_IDS):
        raise ValueError(f"{shard_id} inventory must remain ordered sectors 1..360")
    if any(record.shard_id != shard_id for record in (*records, *inventoried_records)):
        raise ValueError("records from different shards cannot share an NPZ")
    if gate0_facts.shard_id != shard_id:
        raise ValueError("Gate-0 shard facts do not match the payload shard")

    for record, acquisition in zip(records, acquisitions):
        _validate_acquisition_against_gate0(acquisition, record, gate0_facts)
    _validate_autofocus_extrema_against_gate0(acquisitions, gate0_facts)
    frequency = np.asarray(acquisitions[0].frequencies_hz)
    for acquisition in acquisitions[1:]:
        candidate = np.asarray(acquisition.frequencies_hz)
        if candidate.dtype != frequency.dtype or not np.array_equal(candidate, frequency):
            raise ValueError(f"{shard_id} native frequency grid changes across sectors")

    concatenated_names = ("response", "x", "y", "z", "r0", "th", "phi", "pulse_index")
    for name in concatenated_names:
        _same_native_dtype(acquisitions, name)
    arrays = {
        name: np.concatenate([np.asarray(getattr(value, name)) for value in acquisitions], axis=0)
        for name in concatenated_names
    }
    view_count = int(arrays["response"].shape[0])
    split = build_sector_split()
    sector_id = np.concatenate(
        [np.full(value.view_count, record.sector_id, dtype=np.int16) for record, value in zip(records, acquisitions)]
    )
    role = np.asarray([split.role_for(int(value)) for value in sector_id], dtype="U10")
    if np.any(role == "test"):
        raise ValueError(f"{shard_id} payload illegally includes a sealed test row")
    arrays.update(
        {
            "frequencies_hz": np.array(frequency, copy=True),
            "sector_id": sector_id,
            "pass_id": np.full(view_count, first_record.pass_id, dtype=np.int16),
            "polarization": np.full(view_count, first_record.polarization, dtype="U2"),
            "role": role,
            "autofocus_available": np.asarray(
                first_record.polarization in CO_POLARIZATIONS, dtype=np.bool_
            ),
            "autofocus_applied": np.asarray(False, dtype=np.bool_),
            "autofocus_state": np.asarray(
                "raw_channel_own_arrays_unapplied"
                if first_record.polarization in CO_POLARIZATIONS
                else "official_arrays_absent",
                dtype="U36",
            ),
        }
    )
    if first_record.polarization in CO_POLARIZATIONS:
        _same_native_dtype(acquisitions, "r_correct_raw")
        _same_native_dtype(acquisitions, "ph_correct_raw")
        arrays["r_correct_raw"] = np.concatenate(
            [np.asarray(value.r_correct_raw) for value in acquisitions]
        )
        arrays["ph_correct_raw"] = np.concatenate(
            [np.asarray(value.ph_correct_raw) for value in acquisitions]
        )
    else:
        arrays["r_correct_raw"] = np.empty(0, dtype=np.float64)
        arrays["ph_correct_raw"] = np.empty(0, dtype=np.float64)

    source_shapes = sorted({tuple(value.source_response_shape) for value in acquisitions})
    source_layouts = sorted({value.source_response_layout for value in acquisitions})
    pulse_sources = sorted({value.pulse_index_source for value in acquisitions})
    views_per_sector = [value.view_count for value in acquisitions]
    layout = {
        "response_shape": [view_count, int(frequency.size)],
        "response_dtype": str(arrays["response"].dtype),
        "frequency_shape": [int(frequency.size)],
        "frequency_dtype": str(frequency.dtype),
        "geometry_shapes": {name: [view_count] for name in ("x", "y", "z", "r0", "th", "phi")},
        "geometry_dtypes": {name: str(arrays[name].dtype) for name in ("x", "y", "z", "r0", "th", "phi")},
        "source_response_shapes": [list(shape) for shape in source_shapes],
        "source_response_layouts": source_layouts,
        "pulse_index_sources": pulse_sources,
        "views_per_sector_min": int(min(views_per_sector)),
        "views_per_sector_max": int(max(views_per_sector)),
        "inventoried_sector_count": SECTOR_COUNT,
        "payload_sector_count": PAYLOAD_SECTOR_COUNT,
        "sealed_test_sector_count": SEALED_TEST_SECTOR_COUNT,
        "native_frequency_preserved": True,
        "resampled": False,
        "padded": False,
        "trimmed": False,
        "autofocus_unapplied": True,
    }
    metadata = {
        "schema": ARCHIVE_SCHEMA,
        "scene_id": SCENE_ID,
        "shard_id": shard_id,
        "pass_id": first_record.pass_id,
        "polarization": first_record.polarization,
        "inventoried_sector_count": SECTOR_COUNT,
        "payload_sector_count": PAYLOAD_SECTOR_COUNT,
        "sealed_test_sector_count": SEALED_TEST_SECTOR_COUNT,
        "inventoried_sector_ids": list(SECTOR_IDS),
        "payload_sector_ids": list(PAYLOAD_SECTOR_IDS),
        "sealed_test_sector_ids": list(SEALED_TEST_SECTOR_IDS),
        "inventoried_source_files": [
            record.source_path for record in inventoried_records
        ],
        "payload_source_files": [record.source_path for record in records],
        "sealed_test_source_files": [
            record.source_path
            for record in inventoried_records
            if record.sector_id in SEALED_TEST_SECTOR_IDS
        ],
        "view_count": view_count,
        "frequency_count": int(frequency.size),
        "split_seed": SPLIT_SEED,
        "test_payload_included": False,
        "test_opened": False,
        "corrections_applied": False,
        "autofocus_unapplied": True,
        "autofocus_state": str(arrays["autofocus_state"]),
        "gate0_inventory_path": gate0_inventory_path,
        "gate0_audited_facts": gate0_facts.as_dict(),
        "materialization_slurm_job_id": materialization_slurm_job_id,
        "layout": layout,
    }
    arrays["metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")), dtype="U"
    )
    return arrays, layout


def _write_uncompressed_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite archive {path}")
    with path.open("xb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    with zipfile.ZipFile(path, "r") as archive:
        if not archive.infolist() or any(
            item.compress_type != zipfile.ZIP_STORED for item in archive.infolist()
        ):
            raise ValueError(f"{path} is not a native uncompressed NPZ")


def validate_native_archive(
    path: os.PathLike[str] | str,
    pass_id: int,
    polarization: str,
    *,
    expected_inventory: Sequence[SourceRecord] | None = None,
    expected_gate0_facts: Gate0ShardFacts | None = None,
) -> dict[str, Any]:
    """Reopen and validate one published-format native shard."""

    path = Path(path)
    canonical_shard_id(pass_id, polarization)
    with zipfile.ZipFile(path, "r") as archive:
        if any(item.compress_type != zipfile.ZIP_STORED for item in archive.infolist()):
            raise ValueError(f"{path} contains compressed NPZ members")
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != _ARCHIVE_ARRAY_KEYS:
            raise ValueError(f"{path} has the wrong archive keys: {sorted(loaded.files)}")
        arrays = {key: loaded[key] for key in loaded.files}
    if any(array.dtype == object for array in arrays.values()):
        raise ValueError(f"{path} contains an object array")
    response = arrays["response"]
    frequency = arrays["frequencies_hz"]
    if response.ndim != 2 or not np.iscomplexobj(response) or response.size == 0:
        raise ValueError(f"{path} response is not nonempty [view, F] complex data")
    view_count, frequency_count = response.shape
    if frequency.shape != (frequency_count,) or not np.all(np.diff(frequency.astype(np.float64)) > 0):
        raise ValueError(f"{path} frequency is not native one-dimensional increasing data")
    if not np.isfinite(response.real).all() or not np.isfinite(response.imag).all() or not np.isfinite(frequency).all():
        raise ValueError(f"{path} response/frequency is nonfinite")

    row_names = ("x", "y", "z", "r0", "th", "phi", "sector_id", "pulse_index", "pass_id", "polarization", "role")
    for name in row_names:
        if arrays[name].shape != (view_count,):
            raise ValueError(f"{path} {name} is not aligned to the view axis")
    for name in ("x", "y", "z", "r0", "th", "phi"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{path} {name} is nonfinite")
    actual_sectors = set(int(value) for value in np.unique(arrays["sector_id"]))
    if actual_sectors != set(PAYLOAD_SECTOR_IDS):
        raise ValueError(f"{path} does not cover the exact 324 payload sectors")
    split = build_sector_split()
    expected_role = np.asarray(
        [split.role_for(int(sector)) for sector in arrays["sector_id"]], dtype="U10"
    )
    if not np.array_equal(arrays["role"], expected_role):
        raise ValueError(f"{path} contains sector-role drift")
    if np.any(arrays["role"] == "test"):
        raise ValueError(f"{path} contains a sealed test row")
    if not np.all(arrays["pass_id"] == int(pass_id)):
        raise ValueError(f"{path} contains the wrong pass identity")
    if not np.all(arrays["polarization"] == str(polarization).lower()):
        raise ValueError(f"{path} contains the wrong polarization identity")
    for sector_id in PAYLOAD_SECTOR_IDS:
        pulse = arrays["pulse_index"][arrays["sector_id"] == sector_id]
        if pulse.size == 0 or np.unique(pulse).size != pulse.size:
            raise ValueError(f"{path} sector {sector_id} has invalid pulse identities")

    if expected_gate0_facts is not None:
        facts = expected_gate0_facts
        if facts.shard_id != canonical_shard_id(pass_id, polarization):
            raise ValueError(f"{path} was given the wrong Gate-0 shard facts")
        for sector_id in PAYLOAD_SECTOR_IDS:
            observed_count = int(np.count_nonzero(arrays["sector_id"] == sector_id))
            expected_count = facts.pulse_count_for(sector_id)
            if observed_count != expected_count:
                raise ValueError(
                    f"{path} sector {sector_id} view count {observed_count} "
                    f"differs from Gate-0 pulse count {expected_count}"
                )
        expected_frequency = np.asarray(
            facts.frequencies_hz, dtype=np.dtype(facts.frequency_dtype)
        )
        if frequency.dtype != np.dtype(facts.frequency_dtype):
            raise ValueError(f"{path} frequency dtype differs from Gate 0")
        if frequency.shape != expected_frequency.shape:
            raise ValueError(f"{path} frequency count differs from Gate 0")
        if not np.array_equal(frequency, expected_frequency):
            raise ValueError(f"{path} frequency values differ from Gate 0")
        if response.dtype != np.dtype(facts.response_dtype):
            raise ValueError(f"{path} response dtype differs from Gate 0")
        for name in ("x", "y", "z", "r0", "th", "phi"):
            if arrays[name].dtype != np.dtype(facts.geometry_dtype):
                raise ValueError(f"{path} {name} dtype differs from Gate 0")

    autofocus_available = bool(arrays["autofocus_available"])
    autofocus_applied = bool(arrays["autofocus_applied"])
    autofocus_state = str(arrays["autofocus_state"])
    if autofocus_applied:
        raise ValueError(f"{path} illegally applies autofocus during Gate 1")
    if polarization in CO_POLARIZATIONS:
        if not autofocus_available or autofocus_state != "raw_channel_own_arrays_unapplied":
            raise ValueError(f"{path} lacks raw channel-own co-pol autofocus provenance")
        for name in ("r_correct_raw", "ph_correct_raw"):
            if arrays[name].shape != (view_count,) or not np.isfinite(arrays[name]).all():
                raise ValueError(f"{path} {name} is not view-aligned")
            if (
                expected_gate0_facts is not None
                and arrays[name].dtype
                != np.dtype(expected_gate0_facts.autofocus_dtype)
            ):
                raise ValueError(f"{path} {name} dtype differs from Gate 0")
        if expected_gate0_facts is not None:
            expected_extrema = {
                "r_correct_raw": expected_gate0_facts.r_correct_extrema,
                "ph_correct_raw": expected_gate0_facts.ph_correct_extrema,
            }
            for name, extrema in expected_extrema.items():
                observed = (
                    float(np.min(arrays[name])),
                    float(np.max(arrays[name])),
                )
                if extrema is None or observed != extrema:
                    raise ValueError(
                        f"{path} {name} extrema differ from Gate 0"
                    )
    else:
        if autofocus_available or autofocus_state != "official_arrays_absent":
            raise ValueError(f"{path} cross-pol autofocus absence is not explicit")
        if arrays["r_correct_raw"].size or arrays["ph_correct_raw"].size:
            raise ValueError(f"{path} cross-pol correction arrays must be empty")

    metadata = json.loads(str(arrays["metadata_json"]))
    metadata_required = {
        "schema": ARCHIVE_SCHEMA,
        "scene_id": SCENE_ID,
        "shard_id": canonical_shard_id(pass_id, polarization),
        "pass_id": int(pass_id),
        "polarization": str(polarization).lower(),
        "inventoried_sector_count": SECTOR_COUNT,
        "payload_sector_count": PAYLOAD_SECTOR_COUNT,
        "sealed_test_sector_count": SEALED_TEST_SECTOR_COUNT,
        "inventoried_sector_ids": list(SECTOR_IDS),
        "payload_sector_ids": list(PAYLOAD_SECTOR_IDS),
        "sealed_test_sector_ids": list(SEALED_TEST_SECTOR_IDS),
        "split_seed": SPLIT_SEED,
        "test_payload_included": False,
        "test_opened": False,
        "corrections_applied": False,
        "autofocus_unapplied": True,
    }
    for key, expected in metadata_required.items():
        if metadata.get(key) != expected:
            raise ValueError(f"{path} metadata.{key} is invalid")
    if int(metadata.get("view_count", -1)) != view_count:
        raise ValueError(f"{path} metadata view count is invalid")
    if int(metadata.get("frequency_count", -1)) != frequency_count:
        raise ValueError(f"{path} metadata frequency count is invalid")
    if not isinstance(metadata.get("layout"), Mapping):
        raise ValueError(f"{path} metadata layout is missing")
    if expected_gate0_facts is not None and metadata.get(
        "gate0_audited_facts"
    ) != expected_gate0_facts.as_dict():
        raise ValueError(f"{path} metadata Gate-0 scientific facts changed")

    inventoried_sources = metadata.get("inventoried_source_files")
    payload_sources = metadata.get("payload_source_files")
    sealed_sources = metadata.get("sealed_test_source_files")
    if (
        not isinstance(inventoried_sources, list)
        or len(inventoried_sources) != SECTOR_COUNT
        or not isinstance(payload_sources, list)
        or len(payload_sources) != PAYLOAD_SECTOR_COUNT
        or not isinstance(sealed_sources, list)
        or len(sealed_sources) != SEALED_TEST_SECTOR_COUNT
    ):
        raise ValueError(f"{path} metadata source membership counts are invalid")
    if expected_inventory is not None:
        ordered_inventory = sorted(expected_inventory, key=lambda row: row.sector_id)
        if len(ordered_inventory) != SECTOR_COUNT:
            raise ValueError("expected archive inventory must contain 360 sectors")
        expected_inventoried_sources = [
            record.source_path for record in ordered_inventory
        ]
        expected_payload_sources = [
            record.source_path
            for record in ordered_inventory
            if record.sector_id in PAYLOAD_SECTOR_IDS
        ]
        expected_sealed_sources = [
            record.source_path
            for record in ordered_inventory
            if record.sector_id in SEALED_TEST_SECTOR_IDS
        ]
        if inventoried_sources != expected_inventoried_sources:
            raise ValueError(f"{path} inventoried source provenance changed")
        if payload_sources != expected_payload_sources:
            raise ValueError(f"{path} payload source provenance changed")
        if sealed_sources != expected_sealed_sources:
            raise ValueError(f"{path} sealed-test source provenance changed")
    return {
        "view_count": int(view_count),
        "frequency_count": int(frequency_count),
        "response_dtype": str(response.dtype),
        "autofocus_unapplied": True,
        "layout": dict(metadata["layout"]),
        "metadata": metadata,
    }


def _serialized_path(path: os.PathLike[str] | str | None) -> str | None:
    return None if path is None else Path(path).as_posix()


def _clean_exact_partial(path: Path) -> None:
    """Remove one known unpublished regular file, never a broad path/glob."""

    if path.is_symlink():
        raise ValueError(f"refusing to clean symlink partial {path}")
    if path.exists():
        if not path.is_file():
            raise ValueError(f"partial path is not a regular file: {path}")
        path.unlink()


def _publish_no_replace(partial_path: Path, canonical_path: Path) -> None:
    """Atomically publish a same-directory file without replacement.

    A hard-link publication is atomic and fails if the canonical name already
    exists.  The temporary name is then removed.  This is not a lock or claim:
    the manager remains single-writer and a preemption simply leaves either a
    reusable canonical file or one strictly named unpublished partial.
    """

    if canonical_path.exists() or canonical_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {canonical_path}")
    os.link(partial_path, canonical_path)
    partial_path.unlink()


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite JSON partial {path}")
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _records_by_shard(records: Sequence[SourceRecord]) -> dict[str, list[SourceRecord]]:
    grouped = {shard_id: [] for shard_id in SHARD_IDS}
    for record in records:
        grouped[record.shard_id].append(record)
    for shard_id in SHARD_IDS:
        grouped[shard_id].sort(key=lambda row: row.sector_id)
        if [row.sector_id for row in grouped[shard_id]] != list(SECTOR_IDS):
            raise ValueError(f"{shard_id} inventory is not exact sectors 1..360")
    return grouped


def _prepare_output_root(output_root: Path) -> tuple[Path, Path]:
    """Create or reopen the exact resumable output tree, fail-closed."""

    if output_root.is_symlink():
        raise ValueError(f"output root must not be a symlink: {output_root}")
    if output_root.exists() and not output_root.is_dir():
        raise ValueError(f"output root must be a directory: {output_root}")
    if not output_root.exists():
        output_root.mkdir()

    shard_root = output_root / "shards"
    if shard_root.is_symlink():
        raise ValueError(f"shard root must not be a symlink: {shard_root}")
    if shard_root.exists() and not shard_root.is_dir():
        raise ValueError(f"shard root must be a directory: {shard_root}")
    if not shard_root.exists():
        shard_root.mkdir()

    final_manifest = output_root / MANIFEST_FILENAME
    partial_manifest = output_root / f".{MANIFEST_FILENAME}.partial"
    allowed_root = {"shards", MANIFEST_FILENAME, partial_manifest.name}
    root_entries = {entry.name for entry in output_root.iterdir()}
    unexpected_root = root_entries - allowed_root
    if unexpected_root:
        raise ValueError(
            f"Gate-1 output has extraneous root entries: {sorted(unexpected_root)}"
        )

    allowed_shards = {f"{shard_id}.npz" for shard_id in SHARD_IDS}
    allowed_partials = {f".{shard_id}.npz.partial" for shard_id in SHARD_IDS}
    shard_entries = {entry.name for entry in shard_root.iterdir()}
    unexpected_shards = shard_entries - allowed_shards - allowed_partials
    if unexpected_shards:
        raise ValueError(
            "Gate-1 shard directory has extraneous entries: "
            f"{sorted(unexpected_shards)}"
        )

    _clean_exact_partial(partial_manifest)
    for shard_id in SHARD_IDS:
        _clean_exact_partial(shard_root / f".{shard_id}.npz.partial")
    return shard_root, final_manifest


def _validate_materialization_manifest(
    manifest: Mapping[str, Any],
    *,
    output_root: Path,
    records: Sequence[SourceRecord],
    gate0_payload: Mapping[str, Any],
    gate0_inventory_path: os.PathLike[str] | str,
    require_published_manifest: bool,
) -> dict[str, Any]:
    """Validate a complete Gate-1 artifact, including every archive."""

    summary = validate_joint_manifest(manifest)
    gate0_path = _serialized_path(gate0_inventory_path)
    expected_top = {
        "materialization_schema": MATERIALIZATION_SCHEMA,
        "scene_id": SCENE_ID,
        "manager_track_id": MANAGER_TRACK_ID,
        "scene_count": 1,
        "source_file_count": SOURCE_FILE_COUNT,
        "inventoried_source_file_count": SOURCE_FILE_COUNT,
        "payload_audited_source_file_count": PAYLOAD_AUDITED_SOURCE_FILE_COUNT,
        "sealed_test_source_file_count": SEALED_TEST_SOURCE_FILE_COUNT,
        "archive_count": SHARD_COUNT,
        "converted_root_name": OUTPUT_ROOT_NAME,
        "archive_format": "npz_uncompressed_native",
        "one_archive_per_pass_polarization": True,
        "rectangularized_across_shards": False,
        "autofocus_unapplied": True,
        "test_payload_included": False,
        "test_opened": False,
        "corrections_applied": False,
        "gate0_inventory_path": gate0_path,
        "gate0_inventory_schema": GATE0_INVENTORY_SCHEMA,
        "gate0_gate_id": GATE0_ID,
        "gate0_dataset_root": FROZEN_DATASET_ROOT.as_posix(),
    }
    for key, expected in expected_top.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Gate-1 manifest {key} must be {expected!r}")

    expected_archives = [f"shards/{shard_id}.npz" for shard_id in SHARD_IDS]
    if list(manifest.get("archive_paths", ())) != expected_archives:
        raise ValueError("Gate-1 archive paths are not exact/canonical")
    if manifest.get("source_archives") != _source_archive_summary(gate0_payload):
        raise ValueError("Gate-1 source-archive provenance changed")
    if int(manifest.get("source_archive_count", -1)) != len(
        _source_archive_summary(gate0_payload)
    ):
        raise ValueError("Gate-1 source-archive count changed")

    grouped = _records_by_shard(records)
    gate0_facts_by_shard = adapt_gate0_shard_facts(gate0_payload)
    declared_shards = manifest.get("shards")
    if not isinstance(declared_shards, Sequence) or isinstance(
        declared_shards, (str, bytes)
    ):
        raise ValueError("Gate-1 shards must be a sequence")
    if len(declared_shards) != SHARD_COUNT:
        raise ValueError("Gate-1 must declare exactly 32 shards")
    native_layouts = manifest.get("native_layouts")
    if not isinstance(native_layouts, Mapping) or set(native_layouts) != set(
        SHARD_IDS
    ):
        raise ValueError("Gate-1 native layouts do not cover exact shards")

    shard_root = output_root / "shards"
    expected_shard_entries = {f"{shard_id}.npz" for shard_id in SHARD_IDS}
    actual_shard_entries = {entry.name for entry in shard_root.iterdir()}
    if actual_shard_entries != expected_shard_entries:
        raise ValueError(
            "Gate-1 archive directory is incomplete or extraneous: "
            f"{sorted(actual_shard_entries ^ expected_shard_entries)}"
        )

    for expected_shard_id, declared in zip(SHARD_IDS, declared_shards):
        if not isinstance(declared, Mapping):
            raise ValueError("Gate-1 shard declaration must be a mapping")
        inventory = grouped[expected_shard_id]
        gate0_facts = gate0_facts_by_shard[expected_shard_id]
        pass_id = inventory[0].pass_id
        polarization = inventory[0].polarization
        payload_records = [
            row for row in inventory if row.sector_id in PAYLOAD_SECTOR_IDS
        ]
        sealed_records = [
            row for row in inventory if row.sector_id in SEALED_TEST_SECTOR_IDS
        ]
        relative_archive = f"shards/{expected_shard_id}.npz"
        expected_shard = {
            "shard_id": expected_shard_id,
            "pass_id": pass_id,
            "polarization": polarization,
            "sector_ids": list(SECTOR_IDS),
            "sector_roles": list(_SEALED_SPLIT.role_by_sector),
            "source_files": [row.source_path for row in inventory],
            "source_file_count": SECTOR_COUNT,
            "inventoried_source_file_count": SECTOR_COUNT,
            "payload_audited_sector_ids": list(PAYLOAD_SECTOR_IDS),
            "payload_audited_sector_count": PAYLOAD_SECTOR_COUNT,
            "payload_audited_source_files": [
                row.source_path for row in payload_records
            ],
            "payload_audited_source_file_count": PAYLOAD_SECTOR_COUNT,
            "sealed_test_sector_ids": list(SEALED_TEST_SECTOR_IDS),
            "sealed_test_sector_count": SEALED_TEST_SECTOR_COUNT,
            "sealed_test_source_files": [
                row.source_path for row in sealed_records
            ],
            "sealed_test_source_file_count": SEALED_TEST_SECTOR_COUNT,
            "archive_path": relative_archive,
            "archive_format": "npz_uncompressed_native",
            "autofocus_unapplied": True,
            "test_payload_included": False,
            "test_opened": False,
            "corrections_applied": False,
            "gate0_audited_facts": gate0_facts.as_dict(),
        }
        for key, expected in expected_shard.items():
            if declared.get(key) != expected:
                raise ValueError(
                    f"Gate-1 {expected_shard_id}.{key} must be exact"
                )

        archive_path = output_root / relative_archive
        if archive_path.is_symlink() or not archive_path.is_file():
            raise ValueError(f"Gate-1 archive is not a regular file: {archive_path}")
        archive = validate_native_archive(
            archive_path,
            pass_id,
            polarization,
            expected_inventory=inventory,
            expected_gate0_facts=gate0_facts,
        )
        if int(declared.get("archive_view_count", -1)) != archive["view_count"]:
            raise ValueError(f"{expected_shard_id} archive view count changed")
        if int(declared.get("archive_frequency_count", -1)) != archive[
            "frequency_count"
        ]:
            raise ValueError(f"{expected_shard_id} archive frequency count changed")
        if native_layouts[expected_shard_id] != archive["layout"]:
            raise ValueError(f"{expected_shard_id} native layout changed")
        if archive["metadata"].get("gate0_inventory_path") != gate0_path:
            raise ValueError(f"{expected_shard_id} Gate-0 provenance changed")

    if require_published_manifest:
        expected_root_entries = {MANIFEST_FILENAME, "shards"}
        actual_root_entries = {entry.name for entry in output_root.iterdir()}
        if actual_root_entries != expected_root_entries:
            raise ValueError("published Gate-1 root is incomplete or extraneous")
        manifest_path = output_root / MANIFEST_FILENAME
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("published Gate-1 manifest is not a regular file")
        with manifest_path.open("r", encoding="utf-8") as handle:
            persisted = json.load(handle)
        if persisted != manifest:
            raise ValueError("published Gate-1 manifest content changed")
    return summary


def _resolve_source_path(record: SourceRecord, raw_root: Path | None) -> Path:
    source = Path(record.source_path)
    if not source.is_absolute() and raw_root is not None:
        source = raw_root / source
    return source


def _require_slurm_allocation() -> str:
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    if not job_id:
        raise RuntimeError("full GOTCHA materialization must run inside a Slurm allocation")
    return str(job_id)


def materialize_inventory(
    gate0_payload: Any,
    output_root: os.PathLike[str] | str,
    *,
    raw_root: os.PathLike[str] | str | None = None,
    mat_loader: Callable[[Path, SourceRecord], NativeAcquisition] = load_native_mat,
    require_slurm: bool = True,
    gate0_inventory_path: os.PathLike[str] | str | None = None,
    enforce_production_paths: bool = True,
) -> dict[str, Any]:
    """Materialize or safely resume the complete sealed-test Gate-1 dataset."""

    slurm_job_id = _require_slurm_allocation() if require_slurm else None
    output_root = Path(output_root)
    if output_root.name != OUTPUT_ROOT_NAME:
        raise ValueError(f"output root must be named {OUTPUT_ROOT_NAME}")
    if gate0_inventory_path is None:
        raise ValueError("Gate-1 requires the exact Gate-0 artifact path")
    if enforce_production_paths:
        if output_root.as_posix() != FROZEN_OUTPUT_ROOT.as_posix():
            raise ValueError("production output path is not the frozen Gate-1 root")
        if raw_root is not None:
            raise ValueError("production conversion does not accept a raw-root override")
    validate_authentic_gate0(
        gate0_payload,
        gate0_inventory_path=gate0_inventory_path,
        enforce_production_paths=enforce_production_paths,
    )
    records = adapt_gate0_inventory(gate0_payload, require_full=True)
    if not isinstance(gate0_payload, Mapping):  # Proven by authentic validator.
        raise AssertionError("validated Gate-0 payload unexpectedly is not a mapping")
    raw_root_path = None if raw_root is None else Path(raw_root)
    by_shard = _records_by_shard(records)
    gate0_facts_by_shard = adapt_gate0_shard_facts(gate0_payload)

    if output_root.parent.is_symlink() or not output_root.parent.is_dir():
        raise ValueError("Gate-1 parent must be an existing regular directory")
    shard_directory, manifest_path = _prepare_output_root(output_root)

    # A fully valid final artifact is an idempotent no-op.  Validation includes
    # all 32 archives and their exact Gate-0 source membership; no loader runs.
    if manifest_path.exists() or manifest_path.is_symlink():
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("existing Gate-1 manifest is not a regular file")
        with manifest_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        _validate_materialization_manifest(
            existing,
            output_root=output_root,
            records=records,
            gate0_payload=gate0_payload,
            gate0_inventory_path=gate0_inventory_path,
            require_published_manifest=True,
        )
        return dict(existing)

    split = build_sector_split()
    manifest_shards = []
    native_layouts = {}
    archive_paths = []
    gate0_path = _serialized_path(gate0_inventory_path)
    for pass_id in PASS_IDS:
        for polarization in POLARIZATIONS:
            shard_id = canonical_shard_id(pass_id, polarization)
            inventory_records = by_shard[shard_id]
            gate0_facts = gate0_facts_by_shard[shard_id]
            payload_records = [
                record
                for record in inventory_records
                if record.sector_id in PAYLOAD_SECTOR_IDS
            ]
            sealed_records = [
                record
                for record in inventory_records
                if record.sector_id in SEALED_TEST_SECTOR_IDS
            ]
            relative_archive = Path("shards") / f"{shard_id}.npz"
            archive_path = output_root / relative_archive
            partial_path = shard_directory / f".{shard_id}.npz.partial"

            if archive_path.exists() or archive_path.is_symlink():
                if archive_path.is_symlink() or not archive_path.is_file():
                    raise ValueError(
                        f"existing Gate-1 shard is not a regular file: {archive_path}"
                    )
                archive_summary = validate_native_archive(
                    archive_path,
                    pass_id,
                    polarization,
                    expected_inventory=inventory_records,
                    expected_gate0_facts=gate0_facts,
                )
                if archive_summary["metadata"].get("gate0_inventory_path") != gate0_path:
                    raise ValueError(f"{shard_id} belongs to a different Gate-0 artifact")
            else:
                # This is the sealed boundary: the loader is never invoked for
                # any of the exact 36 test sectors in any shard.
                acquisitions = [
                    mat_loader(_resolve_source_path(record, raw_root_path), record)
                    for record in payload_records
                ]
                arrays, _ = _build_shard_arrays(
                    payload_records,
                    acquisitions,
                    inventoried_records=inventory_records,
                    gate0_facts=gate0_facts,
                    gate0_inventory_path=gate0_path,
                    materialization_slurm_job_id=slurm_job_id,
                )
                _write_uncompressed_npz(partial_path, arrays)
                archive_summary = validate_native_archive(
                    partial_path,
                    pass_id,
                    polarization,
                    expected_inventory=inventory_records,
                    expected_gate0_facts=gate0_facts,
                )
                _publish_no_replace(partial_path, archive_path)

            archive_paths.append(relative_archive.as_posix())
            native_layouts[shard_id] = archive_summary["layout"]
            manifest_shards.append(
                {
                    "shard_id": shard_id,
                    "pass_id": pass_id,
                    "polarization": polarization,
                    "sector_ids": list(SECTOR_IDS),
                    "sector_roles": list(split.role_by_sector),
                    "source_files": [
                        record.source_path for record in inventory_records
                    ],
                    "source_file_count": SECTOR_COUNT,
                    "inventoried_source_file_count": SECTOR_COUNT,
                    "payload_audited_sector_ids": list(PAYLOAD_SECTOR_IDS),
                    "payload_audited_sector_count": PAYLOAD_SECTOR_COUNT,
                    "payload_audited_source_files": [
                        record.source_path for record in payload_records
                    ],
                    "payload_audited_source_file_count": PAYLOAD_SECTOR_COUNT,
                    "sealed_test_sector_ids": list(SEALED_TEST_SECTOR_IDS),
                    "sealed_test_sector_count": SEALED_TEST_SECTOR_COUNT,
                    "sealed_test_source_files": [
                        record.source_path for record in sealed_records
                    ],
                    "sealed_test_source_file_count": SEALED_TEST_SECTOR_COUNT,
                    "archive_path": relative_archive.as_posix(),
                    "archive_view_count": archive_summary["view_count"],
                    "archive_frequency_count": archive_summary["frequency_count"],
                    "archive_format": "npz_uncompressed_native",
                    "autofocus_unapplied": True,
                    "test_payload_included": False,
                    "test_opened": False,
                    "corrections_applied": False,
                    "gate0_audited_facts": gate0_facts.as_dict(),
                }
            )

    source_archives = _source_archive_summary(gate0_payload)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "materialization_schema": MATERIALIZATION_SCHEMA,
        "scene_id": SCENE_ID,
        "manager_track_id": MANAGER_TRACK_ID,
        "scene_count": 1,
        "passes": list(PASS_IDS),
        "polarizations": list(POLARIZATIONS),
        "source_file_count": SOURCE_FILE_COUNT,
        "inventoried_source_file_count": SOURCE_FILE_COUNT,
        "payload_audited_source_file_count": PAYLOAD_AUDITED_SOURCE_FILE_COUNT,
        "sealed_test_source_file_count": SEALED_TEST_SOURCE_FILE_COUNT,
        "source_archives": source_archives,
        "source_archive_count": len(source_archives),
        "converted_root_name": OUTPUT_ROOT_NAME,
        "archive_count": SHARD_COUNT,
        "archive_paths": archive_paths,
        "archive_format": "npz_uncompressed_native",
        "one_archive_per_pass_polarization": True,
        "rectangularized_across_shards": False,
        "native_layouts": native_layouts,
        "autofocus_unapplied": True,
        "test_payload_included": False,
        "test_opened": False,
        "corrections_applied": False,
        "split": split.as_dict(),
        "support": support_contract(),
        "shards": manifest_shards,
        "gate0_inventory_path": gate0_path,
        "gate0_inventory_schema": gate0_payload.get("inventory_schema"),
        "gate0_gate_id": gate0_payload.get("gate_id"),
        "gate0_dataset_root": gate0_payload.get("dataset_root"),
        "materialization_slurm_job_id": slurm_job_id,
    }
    _validate_materialization_manifest(
        manifest,
        output_root=output_root,
        records=records,
        gate0_payload=gate0_payload,
        gate0_inventory_path=gate0_inventory_path,
        require_published_manifest=False,
    )

    partial_manifest = output_root / f".{MANIFEST_FILENAME}.partial"
    _write_json_new(partial_manifest, manifest)
    with partial_manifest.open("r", encoding="utf-8") as handle:
        reloaded = json.load(handle)
    if reloaded != manifest:
        raise ValueError("Gate-1 manifest JSON round trip changed content")
    _publish_no_replace(partial_manifest, manifest_path)
    _validate_materialization_manifest(
        manifest,
        output_root=output_root,
        records=records,
        gate0_payload=gate0_payload,
        gate0_inventory_path=gate0_inventory_path,
        require_published_manifest=True,
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate0-json", required=True, help="Completed Gate-0 inventory JSON")
    parser.add_argument(
        "--output-root",
        required=True,
        help=f"New output directory; basename must be {OUTPUT_ROOT_NAME}",
    )
    parser.add_argument(
        "--raw-root",
        default=None,
        help="Optional root prepended to relative Gate-0 source paths",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _require_slurm_allocation()
    payload, _ = load_gate0_json(args.gate0_json)
    manifest = materialize_inventory(
        payload,
        args.output_root,
        raw_root=args.raw_root,
        require_slurm=True,
        gate0_inventory_path=args.gate0_json,
    )
    print(
        json.dumps(
            {
                "schema": manifest["materialization_schema"],
                "scene_id": manifest["scene_id"],
                "output_root": str(Path(args.output_root)),
                "archive_count": manifest["archive_count"],
                "source_file_count": manifest["source_file_count"],
                "autofocus_unapplied": manifest["autofocus_unapplied"],
                "test_opened": manifest["test_opened"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
