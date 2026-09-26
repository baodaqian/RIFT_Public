"""Core Camry full-polarization L=3 RIFT model and training primitives.

This module is deliberately an additive core layer.  It does not load an
archive, open TEST, submit a job, or define a CLI.  The caller supplies the
already loaded native shards and selects the exact TRAIN panel through
``select_camry_train_panel``.

The implementation keeps the GOTCHA native operator in NumPy for the bounded
DC CGLS initializer and uses float64/complex128 Torch rendering for learned
training.  The spatial candidate lattice is implicit: only integer lattice
indices and their selected float64 coordinates are materialized.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from rift.calibration import GlobalComplexGain
from rift.gotcha_native_complex_roi_fit import (
    CAMRY_PLACEMENT,
    RaggedComplexValues,
    RANGE_SUBSPACE_GOLDEN_ITERATIONS,
    RANGE_SUBSPACE_GUARD_CELLS,
    RANGE_SUBSPACE_HARMONIC_TOLERANCE,
    RANGE_SUBSPACE_RETENTION_THRESHOLD,
    RANGE_SUBSPACE_SAMPLES,
    RANGE_SUBSPACE_SCHEMA,
    SUPPORTS,
    _range_box_bounds,
    _range_delay_samples,
    _range_group_records,
)
from rift.gotcha_source_af import (
    CamryVVTrainSourceAFScope,
    SourceAFObservation,
    SourceAFScope,
    build_source_af,
)
from rift.spherical_harmonics import real_sh_basis, num_sh_basis


SCHEMA = "rift_gotcha_camry_fullpol_l3_core_v1"
CHECKPOINT_SCHEMA = f"{SCHEMA}.checkpoint"
CGLS_SCHEMA = f"{SCHEMA}.dc_cgls24"
SPEED_OF_LIGHT_M_S = 299_792_458.0
POLARIZATIONS = ("hh", "hv", "vh", "vv")
SH_DEGREE = 3
SH_BASIS_COUNT = num_sh_basis(SH_DEGREE)
Y00 = 1.0 / math.sqrt(4.0 * math.pi)
MAX_ACTIVE_SITES = 8_192
RESERVED_SITE_CAPACITY = MAX_ACTIVE_SITES
MAX_CGLS_ITERATIONS = 24
CGLS_DIAGNOSTIC_STEPS = (0, 8, 16, 24)
GROUP_EPS = 1.0e-12


def _readonly(value: Any, dtype: Any) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _finite(value: Any, label: str) -> None:
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")


def _identity_key(identity: Any) -> tuple[int, str, int, int]:
    try:
        return (
            int(identity.pass_id),
            str(identity.polarization).lower(),
            int(identity.sector_id),
            int(identity.pulse_index),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError("identity must expose pass_id/polarization/sector_id/pulse_index") from exc


def _as_tuple_records(records: Iterable[Any]) -> tuple[Any, ...]:
    values = tuple(records)
    if not values:
        raise ValueError("record set must be nonempty")
    keys = tuple(_identity_key(record.identity) for record in values)
    if len(set(keys)) != len(keys) or keys != tuple(sorted(keys)):
        raise ValueError("records must have unique canonical sorted identities")
    if any(str(getattr(record, "role", "")).lower() == "test" for record in values):
        raise ValueError("TEST observations are sealed")
    return values


@dataclass(frozen=True)
class CamryTrainingRecord:
    """Selected measurement view with explicit channel-owned AF semantics."""

    identity: Any
    role: str
    polarization: str
    position_xyz_m: np.ndarray
    frequencies_hz: np.ndarray
    r0_raw_m: float
    response_raw: np.ndarray
    r_correct_raw_m: float | None
    ph_correct_raw_rad: float | None
    r0_selected_m: float
    response_selected: np.ndarray
    representation_tag: str
    source_af_applied: bool
    source_record: Any = None

    def __post_init__(self) -> None:
        identity_pol = str(getattr(self.identity, "polarization", "")).lower()
        polarization = str(self.polarization).lower()
        if identity_pol != polarization or polarization not in POLARIZATIONS:
            raise ValueError("record polarization does not match canonical identity")
        if str(self.role).lower() != "train":
            raise ValueError("Camry full-pol core accepts TRAIN records only")
        key = _identity_key(self.identity)
        if key[0] != 1 or key[2] != 2:
            raise ValueError("Camry panel requires pass 1 sector 002")
        position = _readonly(self.position_xyz_m, np.float64)
        frequencies = _readonly(self.frequencies_hz, np.float64)
        response_raw = _readonly(self.response_raw, np.complex128)
        response_selected = _readonly(self.response_selected, np.complex128)
        if position.shape != (3,):
            raise ValueError("position_xyz_m must have shape [3]")
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError("frequencies_hz must be a nonempty vector")
        if frequencies.size > 1 and not np.all(np.diff(frequencies) > 0.0):
            raise ValueError("frequencies_hz must be strictly increasing")
        if response_raw.shape != frequencies.shape or response_selected.shape != frequencies.shape:
            raise ValueError("responses must match the native frequency vector")
        _finite(position, "position_xyz_m")
        _finite(frequencies, "frequencies_hz")
        _finite(response_raw.real, "response_raw.real")
        _finite(response_raw.imag, "response_raw.imag")
        _finite(response_selected.real, "response_selected.real")
        _finite(response_selected.imag, "response_selected.imag")
        if not np.isfinite(float(self.r0_raw_m)) or not np.isfinite(float(self.r0_selected_m)):
            raise ValueError("range references must be finite")
        if polarization in {"hh", "vv"}:
            if not self.source_af_applied or self.representation_tag != "source_af":
                raise ValueError("HH/VV must use their own source-AF representation")
            if self.r_correct_raw_m is None or self.ph_correct_raw_rad is None:
                raise ValueError("HH/VV source-AF records require both raw corrections")
            if not isinstance(self.source_record, SourceAFObservation):
                raise TypeError("HH/VV core records must retain a scoped SourceAFObservation view")
            if _identity_key(self.source_record.identity) != key:
                raise ValueError("source-AF provenance identity does not match the core record")
        else:
            if self.source_af_applied or self.representation_tag != "raw_unapplied":
                raise ValueError("HV/VH must retain raw response and raw r0")
            if float(self.r0_selected_m) != float(self.r0_raw_m) or not np.array_equal(response_selected, response_raw):
                raise ValueError("HV/VH selected values must be raw")
            if _identity_key(getattr(self.source_record, "identity", None)) != key:
                raise ValueError("raw provenance identity does not match the core record")
        object.__setattr__(self, "role", "train")
        object.__setattr__(self, "polarization", polarization)
        object.__setattr__(self, "position_xyz_m", position)
        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "response_raw", response_raw)
        object.__setattr__(self, "response_selected", response_selected)

    @classmethod
    def from_source_af_observation(cls, observation: SourceAFObservation) -> "CamryTrainingRecord":
        """Wrap an immutable, explicitly scoped co-polar SourceAFObservation."""

        if not isinstance(observation, SourceAFObservation):
            raise TypeError("co-polar core routing requires SourceAFObservation")
        identity = observation.identity
        key = _identity_key(identity)
        return cls(
            identity=identity,
            role=str(observation.role).lower(),
            polarization=key[1],
            position_xyz_m=observation.position_xyz_m,
            frequencies_hz=observation.frequencies_hz,
            r0_raw_m=observation.r0_raw_m,
            response_raw=observation.response_raw,
            r_correct_raw_m=observation.r_correct_raw_m,
            ph_correct_raw_rad=observation.ph_correct_raw_rad,
            r0_selected_m=observation.effective_r0_m,
            response_selected=observation.effective_response,
            representation_tag="source_af",
            source_af_applied=True,
            source_record=observation,
        )

    @classmethod
    def from_native_observation(cls, observation: Any) -> "CamryTrainingRecord":
        """Route co-polar records through scoped AF and cross-pol records raw."""

        identity = getattr(observation, "identity", None)
        key = _identity_key(identity)
        polarization = key[1]
        role = str(getattr(observation, "role", "")).lower()
        if polarization in {"hh", "vv"}:
            if polarization == "hh":
                scope = SourceAFScope(expected_count=1, name="camry_p1_hh_train_sector002")
            else:
                scope = CamryVVTrainSourceAFScope(expected_count=1)
            return cls.from_source_af_observation(build_source_af((observation,), scope=scope)[0])
        elif polarization in {"hv", "vh"}:
            raw_response = np.asarray(getattr(observation, "response"), dtype=np.complex128)
            frequencies = np.asarray(getattr(observation, "frequencies_hz"), dtype=np.float64)
            raw_r0 = float(np.float64(getattr(observation, "r0_m")))
            r_correct = getattr(observation, "r_correct_raw", None)
            ph_correct = getattr(observation, "ph_correct_raw", None)
            selected_r0 = raw_r0
            selected_response = raw_response
            tag = "raw_unapplied"
            applied = False
        else:
            raise ValueError(f"unsupported Camry polarization {polarization!r}")
        return cls(
            identity=identity,
            role=role,
            polarization=polarization,
            position_xyz_m=np.asarray(getattr(observation, "position_xyz_m"), dtype=np.float64),
            frequencies_hz=frequencies,
            r0_raw_m=raw_r0,
            response_raw=raw_response,
            r_correct_raw_m=None if r_correct is None else float(np.float64(r_correct)),
            ph_correct_raw_rad=None if ph_correct is None else float(np.float64(ph_correct)),
            r0_selected_m=selected_r0,
            response_selected=selected_response,
            representation_tag=tag,
            source_af_applied=applied,
            source_record=observation,
        )

    @property
    def r0_source_m(self) -> float:
        return float(self.r0_selected_m)

    @property
    def response_source(self) -> np.ndarray:
        return self.response_selected

    def metadata(self) -> dict[str, Any]:
        return {
            "identity": list(_identity_key(self.identity)),
            "role": self.role,
            "polarization": self.polarization,
            "frequency_count": int(self.frequencies_hz.size),
            "representation_tag": self.representation_tag,
            "source_af_applied": bool(self.source_af_applied),
            "r0_selected_m": float(self.r0_selected_m),
        }


@dataclass(frozen=True)
class CamryTrainingPanel:
    records_by_polarization: Mapping[str, tuple[CamryTrainingRecord, ...]]

    def __post_init__(self) -> None:
        normalized: dict[str, tuple[CamryTrainingRecord, ...]] = {}
        for polarization in POLARIZATIONS:
            if polarization not in self.records_by_polarization:
                raise ValueError(f"missing Camry polarization {polarization}")
            records = _as_tuple_records(self.records_by_polarization[polarization])
            if any(record.polarization != polarization for record in records):
                raise ValueError("panel mapping key does not match record polarization")
            normalized[polarization] = records
        object.__setattr__(self, "records_by_polarization", normalized)

    @property
    def fmax_hz(self) -> float:
        return float(max(np.max(record.frequencies_hz) for records in self.records_by_polarization.values() for record in records))

    @property
    def record_counts(self) -> dict[str, int]:
        return {polarization: len(self.records_by_polarization[polarization]) for polarization in POLARIZATIONS}

    @property
    def frequency_sample_counts(self) -> dict[str, int]:
        return {
            polarization: int(sum(record.frequencies_hz.size for record in self.records_by_polarization[polarization]))
            for polarization in POLARIZATIONS
        }

    @property
    def logical_batch_count(self) -> int:
        return max(self.record_counts.values())

    def records(self, polarization: str) -> tuple[CamryTrainingRecord, ...]:
        key = str(polarization).lower()
        if key not in POLARIZATIONS:
            raise ValueError(f"unsupported polarization {polarization!r}")
        return self.records_by_polarization[key]

    def metadata(self) -> dict[str, Any]:
        return {
            "polarizations": list(POLARIZATIONS),
            "record_counts": self.record_counts,
            "frequency_sample_counts": self.frequency_sample_counts,
            "fmax_hz": self.fmax_hz,
            "test_opened": False,
            "records": {polarization: [record.metadata() for record in self.records(polarization)] for polarization in POLARIZATIONS},
        }


def select_camry_train_panel(shards: Mapping[str, Any] | Sequence[Any]) -> CamryTrainingPanel:
    """Select all canonical P1/sector-002 TRAIN pulses independently per channel.

    The shard metadata/identity gate intentionally runs for every supplied shard
    before the first ``NativeShard.observations`` call.  This keeps a malformed
    later row (or an extra/wrong channel shard) from being discovered only after
    an earlier co-polar response has already been materialized.
    """

    if isinstance(shards, Mapping):
        values = tuple(shards.values())
    else:
        values = tuple(shards)
    if len(values) != len(POLARIZATIONS):
        raise ValueError(
            f"Camry full-pol selection requires exactly {len(POLARIZATIONS)} P1 channel shards"
        )
    by_pol: dict[str, list[Any]] = {polarization: [] for polarization in POLARIZATIONS}
    selected_ids: dict[str, tuple[Any, ...]] = {}
    seen_polarizations: set[str] = set()
    for shard in values:
        shard_pol = str(getattr(shard, "polarization", "")).lower()
        if shard_pol not in POLARIZATIONS or int(getattr(shard, "pass_id", -1)) != 1:
            raise ValueError("Camry full-pol selection accepts only P1 HH/HV/VH/VV shards")
        if shard_pol in seen_polarizations:
            raise ValueError(f"duplicate Camry polarization shard: {shard_pol}")
        seen_polarizations.add(shard_pol)
        identities = tuple(getattr(shard, "observation_ids", ()))
        roles = tuple(str(value).lower() for value in np.asarray(getattr(shard, "role", ())).tolist())
        if len(identities) == 0 or len(identities) != len(roles):
            raise ValueError(f"{shard_pol.upper()} shard identity/role vectors are incomplete")
        if len(set(_identity_key(identity) for identity in identities)) != len(identities):
            raise ValueError(f"{shard_pol.upper()} shard contains duplicate native identities")
        # The loader already seals TEST, but retain the explicit row-level gate
        # here so custom NativeShard-like inputs fail closed as well.
        for identity, role in zip(identities, roles):
            if int(getattr(identity, "pass_id", -1)) != 1 or str(getattr(identity, "polarization", "")).lower() != shard_pol:
                raise ValueError("Camry shard identity does not match its P1/channel header")
            if role == "test":
                raise ValueError("Camry full-pol selection rejects sealed TEST rows")
            if role not in {"train", "validation"}:
                raise ValueError("Camry shard role labels must be TRAIN or validation")
        frequencies = np.asarray(getattr(shard, "frequencies_hz"), dtype=np.float64)
        if frequencies.ndim != 1 or frequencies.size == 0 or not np.isfinite(frequencies).all():
            raise ValueError(f"{shard_pol.upper()} shard has an invalid native frequency vector")
        if frequencies.size > 1 and not np.all(np.diff(frequencies) > 0.0):
            raise ValueError(f"{shard_pol.upper()} native frequencies are not strictly increasing")
        selected = tuple(
            sorted(
                (identity for identity, role in zip(identities, roles) if role == "train" and int(identity.sector_id) == 2),
                key=_identity_key,
            )
        )
        if not selected:
            raise ValueError(f"no P1 sector-002 TRAIN records found for {shard_pol.upper()}")
        # Inspect complete selected co-polar provenance and correction vectors
        # using shard metadata/arrays before touching the response member.
        autofocus = getattr(shard, "autofocus", None)
        phase_reference = getattr(shard, "phase_reference", None)
        if shard_pol in {"hh", "vv"}:
            if autofocus is None or getattr(autofocus, "official_available", False) is not True or bool(getattr(autofocus, "applied", True)):
                raise ValueError(f"{shard_pol.upper()} shard lacks unapplied channel-owned source-AF provenance")
            if getattr(phase_reference, "reference_range_field", None) != "r0" or getattr(phase_reference, "geometry_contract", None) != "paired_monostatic_tx_equals_rx_same_observation":
                raise ValueError(f"{shard_pol.upper()} shard lacks native paired-monostatic r0 provenance")
            if shard_pol == "vv" and (
                getattr(phase_reference, "frequency_values", None) != "native_stored_exact"
                or getattr(autofocus, "mode", None) != "raw_channel_own_arrays_unapplied"
                or getattr(autofocus, "source_shard_id", None) != "pass1_vv"
                or getattr(autofocus, "range_field", None) not in {"r_correct_raw", "af.r_correct"}
                or getattr(autofocus, "phase_field", None) not in {"ph_correct_raw", "af.ph_correct"}
            ):
                raise ValueError("VV shard source-AF provenance is not the native pass1_vv representation")
            correction_range = np.asarray(getattr(shard, "r_correct_raw"), dtype=np.float64)
            correction_phase = np.asarray(getattr(shard, "ph_correct_raw"), dtype=np.float64)
            row_by_identity = {
                _identity_key(identity): index for index, identity in enumerate(identities)
            }
            if correction_range.shape != (len(identities),) or correction_phase.shape != (len(identities),):
                raise ValueError(f"{shard_pol.upper()} correction vectors are not aligned to the complete shard")
            correction_rows = np.asarray(
                [
                    (
                        correction_range[row_by_identity[_identity_key(identity)]],
                        correction_phase[row_by_identity[_identity_key(identity)]],
                    )
                    for identity in selected
                ],
                dtype=np.float64,
            )
            if correction_rows.shape != (len(selected), 2) or not np.isfinite(correction_rows).all():
                raise ValueError(f"{shard_pol.upper()} selected source-AF corrections are incomplete/non-finite")
        else:
            if autofocus is None or getattr(autofocus, "official_available", True) is not False or bool(getattr(autofocus, "applied", True)):
                raise ValueError(f"{shard_pol.upper()} shard must retain raw no-source-AF provenance")
        selected_ids[shard_pol] = selected
    if seen_polarizations != set(POLARIZATIONS):
        raise ValueError("Camry full-pol selection must provide one shard for each HH/HV/VH/VV channel")
    # All four shard-level gates have passed; only now access the selected
    # native responses and route co-pol records through their immutable scopes.
    for shard in values:
        shard_pol = str(getattr(shard, "polarization", "")).lower()
        by_pol[shard_pol].extend(shard.observations(selected_ids[shard_pol]))
    records: dict[str, tuple[CamryTrainingRecord, ...]] = {}
    for polarization in POLARIZATIONS:
        selected = tuple(sorted(by_pol[polarization], key=lambda value: _identity_key(value.identity)))
        if not selected:
            raise ValueError(f"no P1 sector-002 TRAIN records found for {polarization.upper()}")
        if polarization == "hh":
            scope = SourceAFScope(expected_count=len(selected), name="camry_p1_hh_train_sector002")
            derived = build_source_af(selected, scope=scope)
            records[polarization] = tuple(CamryTrainingRecord.from_source_af_observation(value) for value in derived)
        elif polarization == "vv":
            scope = CamryVVTrainSourceAFScope(expected_count=len(selected))
            derived = build_source_af(selected, scope=scope)
            records[polarization] = tuple(CamryTrainingRecord.from_source_af_observation(value) for value in derived)
        else:
            records[polarization] = tuple(CamryTrainingRecord.from_native_observation(value) for value in selected)
    return CamryTrainingPanel(records)


@dataclass(frozen=True)
class CamryCandidateLattice:
    """Implicit centered Camry lattice; no N^3 array is allocated."""

    fmax_hz: float
    h_space_m: float
    N: int
    edge_margin_m: float

    def __post_init__(self) -> None:
        fmax = float(self.fmax_hz)
        h = float(self.h_space_m)
        N = int(self.N)
        if not np.isfinite(fmax) or fmax <= 0.0:
            raise ValueError("fmax_hz must be positive and finite")
        expected_h = SPEED_OF_LIGHT_M_S / (2.0 * fmax)
        if h != expected_h:
            raise ValueError("h_space_m must equal c/(2*fmax) in float64 arithmetic")
        expected_N = int(math.floor(10.0 / h)) + 1
        if N != expected_N or N < 2:
            raise ValueError("N must equal floor(10/h_space)+1")
        edge = (10.0 - (N - 1) * h) / 2.0
        if not (0.0 <= edge < h / 2.0 + 1.0e-15):
            raise ValueError("centered lattice edge margin is outside the declared interval")
        if abs(float(self.edge_margin_m) - edge) > 1.0e-14 * max(1.0, abs(edge)):
            raise ValueError("edge_margin_m does not match fmax-derived centered lattice")

    @classmethod
    def from_fmax(cls, fmax_hz: float) -> "CamryCandidateLattice":
        fmax = float(np.float64(fmax_hz))
        h = float(np.float64(SPEED_OF_LIGHT_M_S) / np.float64(2.0 * fmax))
        N = int(math.floor(10.0 / h)) + 1
        return cls(fmax, h, N, (10.0 - (N - 1) * h) / 2.0)

    @classmethod
    def from_panel(cls, panel: CamryTrainingPanel) -> "CamryCandidateLattice":
        return cls.from_fmax(panel.fmax_hz)

    @property
    def candidate_count(self) -> int:
        return int(self.N) ** 3

    @property
    def local_origin_m(self) -> float:
        return -0.5 * float(self.N - 1) * float(self.h_space_m)

    def validate_ijk(self, indices_ijk: Any) -> np.ndarray:
        indices = np.asarray(indices_ijk, dtype=np.int64)
        if indices.ndim != 2 or indices.shape[1] != 3 or indices.shape[0] == 0:
            raise ValueError("lattice indices must have shape [K,3]")
        if np.any(indices < 0) or np.any(indices >= int(self.N)):
            raise ValueError("lattice index lies outside the implicit candidate lattice")
        if len({tuple(row) for row in indices.tolist()}) != indices.shape[0]:
            raise ValueError("shared support lattice indices must be distinct")
        return indices

    def points_local(self, indices_ijk: Any) -> np.ndarray:
        indices = self.validate_ijk(indices_ijk)
        points = (indices.astype(np.float64) - 0.5 * float(self.N - 1)) * float(self.h_space_m)
        return _readonly(points, np.float64)

    def points_native(self, indices_ijk: Any) -> np.ndarray:
        return CAMRY_PLACEMENT.apply(self.points_local(indices_ijk))

    def metadata(self) -> dict[str, Any]:
        return {
            "fmax_hz": float(self.fmax_hz),
            "h_space_m": float(self.h_space_m),
            "N": int(self.N),
            "candidate_count": self.candidate_count,
            "local_origin_m": float(self.local_origin_m),
            "edge_margin_m": float(self.edge_margin_m),
            "centered_points": "(i-(N-1)/2)*h_space",
            "implicit": True,
        }


def rank_candidate_sites(indices_ijk: Any, scores: Any, *, max_active: int = MAX_ACTIVE_SITES) -> np.ndarray:
    """Select a deterministic shared support from explicitly scored sites."""

    indices = np.asarray(indices_ijk, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if indices.ndim != 2 or indices.shape[1] != 3 or values.shape != (indices.shape[0],):
        raise ValueError("candidate indices/scores have incompatible shapes")
    if not 0 < int(max_active) <= MAX_ACTIVE_SITES:
        raise ValueError(f"max_active must be in 1..{MAX_ACTIVE_SITES}")
    _finite(values, "candidate scores")
    if np.any(values < 0.0):
        raise ValueError("candidate scores must be nonnegative")
    if len({tuple(row) for row in indices.tolist()}) != indices.shape[0]:
        raise ValueError("candidate sites must be distinct")
    order = sorted(range(indices.shape[0]), key=lambda index: (-float(values[index]), tuple(int(v) for v in indices[index])))
    selected = indices[np.asarray(order[: int(max_active)], dtype=np.int64)]
    return _readonly(selected, np.int64)


def _source_r0(record: CamryTrainingRecord) -> float:
    return float(record.r0_selected_m)


class CamryL3SparseOperator:
    """Matrix-free native operator for four independent L=3 heads."""

    def __init__(self, points_native_m: Any, *, point_chunk_size: int = 256) -> None:
        points = np.asarray(points_native_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError("points_native_m must have shape [K,3]")
        if points.shape[0] > MAX_ACTIVE_SITES:
            raise ValueError("active support exceeds K<=8192")
        _finite(points, "points_native_m")
        if int(point_chunk_size) <= 0:
            raise ValueError("point_chunk_size must be positive")
        self.points_native_m = _readonly(points, np.float64)
        self.point_chunk_size = int(point_chunk_size)
        self._geometry_cache: dict[tuple[int, str, int, int], tuple[np.ndarray, np.ndarray]] = {}

    @property
    def point_count(self) -> int:
        return int(self.points_native_m.shape[0])

    @staticmethod
    def _basis_and_distance(record: CamryTrainingRecord, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        antenna = np.asarray(record.position_xyz_m, dtype=np.float64)
        delta = antenna[None, :] - points
        distance = np.linalg.norm(delta, axis=1)
        if np.any(distance <= 0.0) or not np.isfinite(distance).all():
            raise ValueError("antenna cannot coincide with a support point")
        unit_local = (delta / distance[:, None]) @ CAMRY_PLACEMENT.rotation
        theta = np.arccos(np.clip(unit_local[:, 2], -1.0, 1.0))
        phi = np.arctan2(unit_local[:, 1], unit_local[:, 0])
        with torch.no_grad():
            basis = real_sh_basis(
                torch.as_tensor(theta, dtype=torch.float64),
                torch.as_tensor(phi, dtype=torch.float64),
                SH_DEGREE,
            ).detach().cpu().numpy().T
        return basis.astype(np.float64, copy=False), distance.astype(np.float64, copy=False)

    def _geometry(self, record: CamryTrainingRecord) -> tuple[np.ndarray, np.ndarray]:
        key = _identity_key(record.identity)
        cached = self._geometry_cache.get(key)
        if cached is None:
            cached = self._basis_and_distance(record, self.points_native_m)
            self._geometry_cache[key] = cached
        return cached

    def _validate_coefficients(self, coefficients: Any, *, dc: bool = False) -> np.ndarray:
        values = np.asarray(coefficients, dtype=np.complex128)
        expected = (self.point_count,) if dc else (self.point_count, SH_BASIS_COUNT)
        if values.shape != expected:
            raise ValueError(f"coefficients must have shape {expected}")
        _finite(values.real, "coefficients.real")
        _finite(values.imag, "coefficients.imag")
        return values

    def forward(self, records: Sequence[CamryTrainingRecord], coefficients: Any) -> RaggedComplexValues:
        values = _as_tuple_records(records)
        coeff = self._validate_coefficients(coefficients)
        predictions: list[np.ndarray] = []
        for record in values:
            basis, distance = self._geometry(record)
            frequencies = np.asarray(record.frequencies_hz, dtype=np.float64)
            output = np.zeros(frequencies.size, dtype=np.complex128)
            for start in range(0, self.point_count, self.point_chunk_size):
                stop = min(start + self.point_chunk_size, self.point_count)
                field = np.sum(basis[start:stop, :] * coeff[start:stop, :], axis=1)
                phase = -(4.0 * np.pi / SPEED_OF_LIGHT_M_S) * (
                    distance[start:stop, None] - _source_r0(record)
                ) * frequencies[None, :]
                output += np.sum(np.exp(1j * phase) * field[:, None], axis=0)
            predictions.append(output)
        return RaggedComplexValues(tuple(record.identity for record in values), tuple(predictions))

    def adjoint(self, records: Sequence[CamryTrainingRecord], residuals: RaggedComplexValues | Sequence[Any]) -> np.ndarray:
        values = _as_tuple_records(records)
        if isinstance(residuals, RaggedComplexValues):
            if tuple(residuals.ids) != tuple(record.identity for record in values):
                raise ValueError("adjoint residual identities do not match records")
            residual_values = residuals.values
        else:
            residual_values = tuple(np.asarray(value, dtype=np.complex128) for value in residuals)
        if len(residual_values) != len(values):
            raise ValueError("adjoint residual count does not match records")
        result = np.zeros((self.point_count, SH_BASIS_COUNT), dtype=np.complex128)
        for record, residual in zip(values, residual_values):
            frequencies = np.asarray(record.frequencies_hz, dtype=np.float64)
            residual = np.asarray(residual, dtype=np.complex128)
            if residual.shape != frequencies.shape:
                raise ValueError("adjoint residual shape does not match native frequencies")
            _finite(residual.real, "adjoint residual.real")
            _finite(residual.imag, "adjoint residual.imag")
            basis, distance = self._geometry(record)
            for start in range(0, self.point_count, self.point_chunk_size):
                stop = min(start + self.point_chunk_size, self.point_count)
                phase = -(4.0 * np.pi / SPEED_OF_LIGHT_M_S) * (
                    distance[start:stop, None] - _source_r0(record)
                ) * frequencies[None, :]
                site = np.sum(np.exp(-1j * phase) * residual[None, :], axis=1)
                result[start:stop, :] += basis[start:stop, :].conj() * site[:, None]
        return result

    def forward_dc(self, records: Sequence[CamryTrainingRecord], coefficients_dc: Any) -> RaggedComplexValues:
        values = _as_tuple_records(records)
        coeff = self._validate_coefficients(coefficients_dc, dc=True)
        predictions: list[np.ndarray] = []
        for record in values:
            _, distance = self._geometry(record)
            frequencies = np.asarray(record.frequencies_hz, dtype=np.float64)
            output = np.zeros(frequencies.size, dtype=np.complex128)
            for start in range(0, self.point_count, self.point_chunk_size):
                stop = min(start + self.point_chunk_size, self.point_count)
                phase = -(4.0 * np.pi / SPEED_OF_LIGHT_M_S) * (
                    distance[start:stop, None] - _source_r0(record)
                ) * frequencies[None, :]
                output += np.sum(np.exp(1j * phase) * (Y00 * coeff[start:stop])[:, None], axis=0)
            predictions.append(output)
        return RaggedComplexValues(tuple(record.identity for record in values), tuple(predictions))

    def adjoint_dc(self, records: Sequence[CamryTrainingRecord], residuals: RaggedComplexValues | Sequence[Any]) -> np.ndarray:
        values = _as_tuple_records(records)
        if isinstance(residuals, RaggedComplexValues):
            if tuple(residuals.ids) != tuple(record.identity for record in values):
                raise ValueError("DC adjoint residual identities do not match records")
            residual_values = residuals.values
        else:
            residual_values = tuple(np.asarray(value, dtype=np.complex128) for value in residuals)
        if len(residual_values) != len(values):
            raise ValueError("DC adjoint residual count does not match records")
        result = np.zeros(self.point_count, dtype=np.complex128)
        for record, residual in zip(values, residual_values):
            frequencies = np.asarray(record.frequencies_hz, dtype=np.float64)
            residual = np.asarray(residual, dtype=np.complex128)
            if residual.shape != frequencies.shape:
                raise ValueError("DC adjoint residual shape does not match frequencies")
            _, distance = self._geometry(record)
            for start in range(0, self.point_count, self.point_chunk_size):
                stop = min(start + self.point_chunk_size, self.point_count)
                phase = -(4.0 * np.pi / SPEED_OF_LIGHT_M_S) * (
                    distance[start:stop, None] - _source_r0(record)
                ) * frequencies[None, :]
                result[start:stop] += Y00 * np.sum(np.exp(-1j * phase) * residual[None, :], axis=1)
        return result


@dataclass(frozen=True)
class _CamryRangeGeometryModel:
    """Minimal geometry carrier for the reused native range-analysis helpers."""

    support: Any = SUPPORTS["toyota_camry"]
    placement: Any = CAMRY_PLACEMENT
    target_id: str = "toyota_camry"


@dataclass(frozen=True)
class _CamryRangeEntry:
    group_id: int
    M: int
    q_centered: np.ndarray
    q_full: np.ndarray
    rank: int
    condition_number: float


class CamryRangeProjector:
    """Fixed B/BH projector accepting source-AF and raw channel views."""

    def __init__(self, records: Sequence[CamryTrainingRecord], *, require_retention: bool = True) -> None:
        self.records = _as_tuple_records(records)
        self.model = _CamryRangeGeometryModel()
        self.groups, group_ids = _range_group_records(self.records)
        self._group_for_identity = {
            _identity_key(record.identity): int(group_id)
            for record, group_id in zip(self.records, group_ids)
        }
        if len(self._group_for_identity) != len(self.records):
            raise ValueError("range projector records must have unique canonical identities")
        self._q_cache: dict[tuple[int, int], _CamryRangeEntry] = {}
        self._record_geometry: dict[tuple[int, str, int, int], dict[str, Any]] = {}
        self._retention: dict[tuple[int, str, int, int], dict[str, Any]] = {}
        for record, group_id in zip(self.records, group_ids):
            identity_key = _identity_key(record.identity)
            group = self.groups[group_id]
            antenna = np.asarray(record.position_xyz_m, dtype=np.float64)
            r_min, r_max, c_native, r_c = _range_box_bounds(self.model, antenna)
            q_minus = float(r_min - r_c - RANGE_SUBSPACE_GUARD_CELLS * group.g_m)
            q_plus = float(r_max - r_c + RANGE_SUBSPACE_GUARD_CELLS * group.g_m)
            ambiguity_period = float(SPEED_OF_LIGHT_M_S / (2.0 * group.max_df_hz))
            principal_lower = float(-0.5 * ambiguity_period)
            principal_upper = float(0.5 * ambiguity_period)
            if q_minus < principal_lower or q_plus > principal_upper:
                raise ValueError(
                    f"range projector guarded interval [{q_minus},{q_plus}] escapes nominal principal cell "
                    f"[{principal_lower},{principal_upper}]"
                )
            unguarded_lower = float(r_min - r_c)
            unguarded_upper = float(r_max - r_c)
            alias_candidates = [
                unguarded_lower - ambiguity_period,
                unguarded_upper + ambiguity_period,
                q_minus - ambiguity_period,
                q_plus + ambiguity_period,
            ]
            same_delay_captures = []
            for candidate in alias_candidates:
                wrapped = float(((candidate + 0.5 * ambiguity_period) % ambiguity_period) - 0.5 * ambiguity_period)
                if unguarded_lower <= wrapped <= unguarded_upper:
                    same_delay_captures.append(
                        {
                            "exterior_q_m": float(candidate),
                            "wrapped_q_m": wrapped,
                            "within_unguarded_in_cube_interval": True,
                        }
                    )
            span = float(q_plus - q_minus)
            M = int(1 + np.ceil(span / group.g_m))
            if M >= int(group.frequency_count):
                raise ValueError(f"range projector requires M<K; got M={M}, K={group.frequency_count} for {identity_key}")
            entry = self._q_entry(group_id, M)
            if entry.rank != M or not np.isfinite(entry.condition_number) or entry.condition_number > 10.0:
                raise ValueError(
                    f"range projector failed rank(E)=M/conditioning guard for {identity_key}: "
                    f"M={M}, rank={entry.rank}, condition={entry.condition_number}"
                )
            q_last = float(q_minus + (M - 1) * group.g_m)
            q_overshoot = float(q_last - q_plus)
            if not (-np.finfo(np.float64).eps * max(1.0, abs(q_plus)) <= q_overshoot < group.g_m):
                raise ValueError("range projector final grid overshoot is outside [0,g)")
            in_cube_q = _range_delay_samples(unguarded_lower, unguarded_upper, group.g_m / 8.0)
            in_cube_values = self._retention_values(group_id, M, in_cube_q, q_offset_m=q_minus)
            retention_min = float(np.min(in_cube_values))
            retention_max = float(np.max(in_cube_values))
            if require_retention and retention_min < RANGE_SUBSPACE_RETENTION_THRESHOLD:
                raise ValueError(
                    f"range projector fixed-2g sampled in-cube retention failed: {retention_min:.9g} < "
                    f"{RANGE_SUBSPACE_RETENTION_THRESHOLD:.9g}"
                )
            exterior_q = np.concatenate(
                (
                    np.linspace(unguarded_lower - 4.0 * group.g_m, unguarded_lower - group.g_m, 4),
                    np.linspace(unguarded_lower, unguarded_upper, 9),
                    np.linspace(unguarded_upper + group.g_m, unguarded_upper + 4.0 * group.g_m, 4),
                    np.asarray(alias_candidates, dtype=np.float64),
                )
            )
            exterior_values = self._retention_values(group_id, M, exterior_q, q_offset_m=q_minus)
            geometry = {
                "identity_key": identity_key,
                "group_id": int(group_id),
                "r0_source_m": float(record.r0_selected_m),
                "antenna_native_m": antenna,
                "c_native_m": c_native,
                "r_c_m": r_c,
                "s_c_m": float(r_c - record.r0_selected_m),
                "frequency_count": int(group.frequency_count),
                "r_min_m": r_min,
                "r_max_m": r_max,
                "ambiguity_period_m": ambiguity_period,
                "principal_cell_lower_m": principal_lower,
                "principal_cell_upper_m": principal_upper,
                "principal_cell_guard_pass": True,
                "same_delay_exterior_capture_nominal": {
                    "period_U_m": ambiguity_period,
                    "principal_cell_m": [principal_lower, principal_upper],
                    "unguarded_in_cube_interval_m": [unguarded_lower, unguarded_upper],
                    "guarded_interval_m": [q_minus, q_plus],
                    "candidate_exterior_delays_m": alias_candidates,
                    "captured_exterior_delays": same_delay_captures,
                    "same_delay_exterior_capture_possible": bool(same_delay_captures),
                    "interpretation": "nominal frequency-alias diagnostic only; no physical exterior returns were separated",
                },
                "q_minus_m": q_minus,
                "q_plus_m": q_plus,
                "q_last_m": q_last,
                "q_overshoot_m": q_overshoot,
                "M": M,
                "rank": entry.rank,
                "condition_number": entry.condition_number,
                "sampled_in_cube_retention_min": retention_min,
                "sampled_in_cube_retention_max": retention_max,
                "sampled_in_cube_sample_count": int(in_cube_q.size),
                "exterior_leakage_curve": {
                    "q_m": exterior_q.tolist(),
                    "projected_energy_fraction": exterior_values.tolist(),
                    "outside_q_m": {"lower": unguarded_lower, "upper": unguarded_upper},
                    "guarded_q_m": {"lower": q_minus, "upper": q_plus},
                },
            }
            self._record_geometry[identity_key] = geometry
            self._retention[identity_key] = {
                "q_m": in_cube_q.tolist(),
                "projected_energy_fraction": in_cube_values.tolist(),
                "minimum": retention_min,
                "maximum": retention_max,
                "threshold": RANGE_SUBSPACE_RETENTION_THRESHOLD,
            }
        self._gauge_report = self._check_frequency_gauges()

    @property
    def h_m(self) -> None:
        return None

    def _q_entry(self, group_id: int, M: int) -> _CamryRangeEntry:
        key = (int(group_id), int(M))
        cached = self._q_cache.get(key)
        if cached is not None:
            return cached
        group = self.groups[int(group_id)]
        columns = np.arange(int(M), dtype=np.float64) * float(group.g_m)
        kappa = 4.0 * np.pi / SPEED_OF_LIGHT_M_S
        centered = np.exp(-1j * kappa * group.centered_frequencies_hz[:, None] * columns[None, :])
        full = np.exp(-1j * kappa * group.frequencies_hz[:, None] * columns[None, :])
        centered /= np.sqrt(float(group.frequency_count))
        full /= np.sqrt(float(group.frequency_count))
        q_centered, r_centered = np.linalg.qr(centered, mode="reduced")
        q_full, _ = np.linalg.qr(full, mode="reduced")
        if not np.isfinite(q_centered.real).all() or not np.isfinite(q_centered.imag).all():
            raise ValueError("centered-frequency QR basis is non-finite")
        entry = _CamryRangeEntry(
            group_id=int(group_id),
            M=int(M),
            q_centered=_readonly(q_centered, np.complex128),
            q_full=_readonly(q_full, np.complex128),
            rank=int(np.linalg.matrix_rank(r_centered, tol=1.0e-12)),
            condition_number=float(np.linalg.cond(r_centered)),
        )
        self._q_cache[key] = entry
        return entry

    def _entry_for(self, record: CamryTrainingRecord) -> tuple[Any, dict[str, Any], _CamryRangeEntry]:
        key = _identity_key(record.identity)
        try:
            geometry = self._record_geometry[key]
        except KeyError as exc:
            raise ValueError("record is not part of this range projector plan") from exc
        group = self.groups[int(geometry["group_id"])]
        return group, geometry, self._q_entry(group.group_id, int(geometry["M"]))

    def _retention_values(self, group_id: int, M: int, q_values_m: Any, *, q_offset_m: float) -> np.ndarray:
        group = self.groups[int(group_id)]
        entry = self._q_entry(group_id, M)
        kappa = 4.0 * np.pi / SPEED_OF_LIGHT_M_S
        values = []
        for q_value in np.asarray(q_values_m, dtype=np.float64):
            probe = np.exp(-1j * kappa * group.centered_frequencies_hz * float(q_value - q_offset_m))
            transformed = entry.q_centered.conj().T @ probe
            values.append(float(np.vdot(transformed, transformed).real / float(group.frequency_count)))
        result = np.asarray(values, dtype=np.float64)
        if not np.isfinite(result).all():
            raise ValueError("sampled range retention is non-finite")
        return result

    def _validate_native_values(self, values: RaggedComplexValues) -> None:
        if tuple(values.ids) != tuple(record.identity for record in self.records):
            raise ValueError("range values do not match projector identities")

    def _validate_projected_values(self, values: RaggedComplexValues) -> None:
        self._validate_native_values(values)
        for record, value in zip(self.records, values.values):
            _, _, entry = self._entry_for(record)
            if np.asarray(value).shape != (entry.M,):
                raise ValueError("range adjoint values must have M projected samples")

    def _resolve_torch_records(self, records: Sequence[CamryTrainingRecord] | None) -> tuple[CamryTrainingRecord, ...]:
        """Resolve a full panel or an identity-keyed subset without positional remapping."""

        if records is None:
            return self.records
        selected = tuple(records)
        if not selected:
            raise ValueError("Torch range record subset must be nonempty")
        keys = tuple(_identity_key(record.identity) for record in selected)
        if len(set(keys)) != len(keys):
            raise ValueError("Torch range record subset contains duplicate identities")
        available = set(self._record_geometry)
        missing = [key for key in keys if key not in available]
        if missing:
            raise ValueError(f"Torch range record subset contains identities outside this projector: {missing[0]}")
        if any(str(getattr(record, "role", "")).lower() == "test" for record in selected):
            raise ValueError("TEST observations are sealed")
        return selected

    def apply_forward(self, values: RaggedComplexValues) -> RaggedComplexValues:
        self._validate_native_values(values)
        output = []
        kappa = 4.0 * np.pi / SPEED_OF_LIGHT_M_S
        for record, value in zip(self.records, values.values):
            group, geometry, entry = self._entry_for(record)
            if np.asarray(value).shape != (group.frequency_count,):
                raise ValueError("range forward values must have native K frequency samples")
            phase = np.exp(1j * kappa * group.centered_frequencies_hz * (geometry["s_c_m"] + geometry["q_minus_m"]))
            output.append(entry.q_centered.conj().T @ (phase * np.asarray(value, dtype=np.complex128)))
        return RaggedComplexValues(values.ids, tuple(output))

    def apply_adjoint(self, values: RaggedComplexValues) -> RaggedComplexValues:
        self._validate_projected_values(values)
        output = []
        kappa = 4.0 * np.pi / SPEED_OF_LIGHT_M_S
        for record, value in zip(self.records, values.values):
            group, geometry, entry = self._entry_for(record)
            phase = np.exp(-1j * kappa * group.centered_frequencies_hz * (geometry["s_c_m"] + geometry["q_minus_m"]))
            output.append(phase * (entry.q_centered @ np.asarray(value, dtype=np.complex128)))
        return RaggedComplexValues(values.ids, tuple(output))

    def target(self) -> RaggedComplexValues:
        return self.apply_forward(
            RaggedComplexValues(
                tuple(record.identity for record in self.records),
                tuple(record.response_selected for record in self.records),
            )
        )

    def retained_energy_fraction(self) -> float:
        """Return the full-cube range projection fraction ||B y||² / ||y||²."""

        native = RaggedComplexValues(
            tuple(record.identity for record in self.records),
            tuple(record.response_selected for record in self.records),
        )
        projected = self.apply_forward(native)
        retained = self.apply_adjoint(projected)
        denominator = float(sum(np.vdot(value, value).real for value in native.values))
        numerator = float(sum(np.vdot(value, value).real for value in retained.values))
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("range projector native TRAIN energy is non-positive/non-finite")
        if not np.isfinite(numerator) or numerator < 0.0:
            raise ValueError("range projector retained TRAIN energy is non-finite/negative")
        fraction = numerator / denominator
        if not np.isfinite(fraction) or fraction < 0.0:
            raise ValueError("range projector retained-energy fraction is non-finite/negative")
        return float(fraction)

    def forward_torch(
        self,
        values: Sequence[torch.Tensor],
        *,
        records: Sequence[CamryTrainingRecord] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        selected_records = self._resolve_torch_records(records)
        if len(values) != len(selected_records):
            raise ValueError("Torch range values do not match the identity-keyed projector records")
        output = []
        for record, value in zip(selected_records, values):
            group, geometry, entry = self._entry_for(record)
            if tuple(value.shape) != (group.frequency_count,):
                raise ValueError("Torch native value shape does not match frequency vector")
            phase = np.exp(1j * (4.0 * np.pi / SPEED_OF_LIGHT_M_S) * group.centered_frequencies_hz * (geometry["s_c_m"] + geometry["q_minus_m"]))
            qh = torch.as_tensor(entry.q_centered.conj().T, dtype=torch.complex128, device=value.device)
            phase_t = torch.as_tensor(phase, dtype=torch.complex128, device=value.device)
            output.append(qh @ (phase_t * value))
        return tuple(output)

    def adjoint_torch(
        self,
        values: Sequence[torch.Tensor],
        *,
        records: Sequence[CamryTrainingRecord] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        selected_records = self._resolve_torch_records(records)
        if len(values) != len(selected_records):
            raise ValueError("Torch range values do not match the identity-keyed projector records")
        output = []
        for record, value in zip(selected_records, values):
            group, geometry, entry = self._entry_for(record)
            if tuple(value.shape) != (entry.M,):
                raise ValueError("Torch adjoint values must have M projected samples")
            phase = np.exp(-1j * (4.0 * np.pi / SPEED_OF_LIGHT_M_S) * group.centered_frequencies_hz * (geometry["s_c_m"] + geometry["q_minus_m"]))
            q = torch.as_tensor(entry.q_centered, dtype=torch.complex128, device=value.device)
            phase_t = torch.as_tensor(phase, dtype=torch.complex128, device=value.device)
            output.append(phase_t * (q @ value))
        return tuple(output)

    def normalization(self) -> int:
        return int(sum(int(self._record_geometry[_identity_key(record.identity)]["M"]) for record in self.records))

    def _check_frequency_gauges(self) -> dict[str, Any]:
        kappa = 4.0 * np.pi / SPEED_OF_LIGHT_M_S
        maximum = 0.0
        comparisons = []
        for record in self.records:
            group, geometry, entry = self._entry_for(record)
            delay = geometry["s_c_m"] + 0.37 * group.g_m
            centered_probe = np.exp(-1j * kappa * group.centered_frequencies_hz * delay)
            full_probe = np.exp(-1j * kappa * group.frequencies_hz * delay)
            centered_z = entry.q_centered.conj().T @ (
                np.exp(1j * kappa * group.centered_frequencies_hz * (geometry["s_c_m"] + geometry["q_minus_m"])) * centered_probe
            )
            full_z = entry.q_full.conj().T @ (
                np.exp(1j * kappa * group.frequencies_hz * (geometry["s_c_m"] + geometry["q_minus_m"])) * full_probe
            )
            centered_projection = np.exp(-1j * kappa * group.centered_frequencies_hz * (geometry["s_c_m"] + geometry["q_minus_m"])) * (entry.q_centered @ centered_z)
            full_projection = np.exp(-1j * kappa * group.frequencies_hz * (geometry["s_c_m"] + geometry["q_minus_m"])) * (entry.q_full @ full_z)
            full_projection *= np.exp(1j * kappa * float(np.mean(group.frequencies_hz)) * delay)
            denominator = max(float(np.linalg.norm(centered_projection)), np.finfo(np.float64).tiny)
            relative = float(np.linalg.norm(centered_projection - full_projection) / denominator)
            maximum = max(maximum, relative)
            comparisons.append({"identity_key": list(_identity_key(record.identity)), "relative_projection_error": relative})
        if maximum > RANGE_SUBSPACE_HARMONIC_TOLERANCE:
            raise ValueError(f"centered/full-frequency range gauges disagree: {maximum:.9g}")
        return {
            "centered_frequency_definition": "nu=f-mean(f)",
            "full_frequency_definition": "f=mean(f)+nu",
            "maximum_relative_projection_error": maximum,
            "tolerance": RANGE_SUBSPACE_HARMONIC_TOLERANCE,
            "comparisons": comparisons,
        }

    @property
    def gauge_report(self) -> Mapping[str, Any]:
        return self._gauge_report

    def _plan_metadata(self) -> dict[str, Any]:
        group_records = []
        for group in self.groups:
            group_records.append(
                {
                    "group_id": int(group.group_id),
                    "record_identity_keys": [list(_identity_key(record.identity)) for record in self.records if self._group_for_identity[_identity_key(record.identity)] == group.group_id],
                    "K": int(group.frequency_count),
                    "first_hz": float(group.first_hz),
                    "last_hz": float(group.last_hz),
                    "min_df_hz": float(group.min_df_hz),
                    "median_df_hz": float(group.median_df_hz),
                    "max_df_hz": float(group.max_df_hz),
                    "max_endpoint_affine_deviation_hz": float(group.endpoint_affine_deviation_hz),
                    "rRayleigh_m": float(group.r_rayleigh_m),
                    "g_m": float(group.g_m),
                    "H_first_minimum": float(group.first_minimum_h),
                    "first_minimum_sample_index": int(group.first_minimum_sample_index),
                    "first_minimum_bracket_m": list(group.first_minimum_bracket_m),
                    "point_response_samples": RANGE_SUBSPACE_SAMPLES,
                    "golden_section_iterations": RANGE_SUBSPACE_GOLDEN_ITERATIONS,
                }
            )
        records = []
        for record in self.records:
            geometry = dict(self._record_geometry[_identity_key(record.identity)])
            geometry["antenna_native_m"] = geometry["antenna_native_m"].tolist()
            geometry["c_native_m"] = geometry["c_native_m"].tolist()
            records.append(geometry)
        return {
            "schema": RANGE_SUBSPACE_SCHEMA,
            "target_id": self.model.target_id,
            "frequency_grouping": "direct in-memory float64 np.array_equal to representative; no hashes",
            "groups": group_records,
            "records": records,
            "q_guard_cells": RANGE_SUBSPACE_GUARD_CELLS,
            "q_guard_definition": "qminus=Rmin-Rc-2g; qplus=Rmax-Rc+2g",
            "range_grid_definition": "q_m=qminus+m*g; M=1+ceil((qplus-qminus)/g)",
            "require_M_lt_K": True,
            "rank_requirement": "rank=M",
            "condition_requirement": "finite cond(E)<=10",
            "E_definition": "K^-1/2 exp(-1j*kappa*nu*m*g)",
            "h": "g exactly",
            "dense_P_or_T_materialized": False,
            "centered_full_frequency_gauge": dict(self.gauge_report),
            "sampled_in_cube_retention_threshold": RANGE_SUBSPACE_RETENTION_THRESHOLD,
            "sampled_in_cube_retention": {
                "records": {str(list(key)): dict(value) for key, value in self._retention.items()},
                "minimum": float(min(value["minimum"] for value in self._retention.values())),
                "maximum": float(max(value["maximum"] for value in self._retention.values())),
            },
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": f"{SCHEMA}.range_projector",
            "h_m": None,
            "h_m_semantics": "RangeSubspacePlan range-analysis spacing; default h_m==g_m",
            "support_bounds": {
                "lower_edge_m": list(SUPPORTS["toyota_camry"].lower_edge_m),
                "upper_edge_exclusive_m": list(SUPPORTS["toyota_camry"].upper_edge_exclusive_m),
            },
            "record_count": len(self.records),
            "N_T": self.normalization(),
            "plan": self._plan_metadata(),
        }


class _CamryL3Head(nn.Module):
    """One independently optimized complex L=3 field head."""

    def __init__(self, point_count: int) -> None:
        super().__init__()
        self.w_re = nn.Parameter(torch.zeros((point_count, SH_BASIS_COUNT), dtype=torch.float64))
        self.w_im = nn.Parameter(torch.zeros((point_count, SH_BASIS_COUNT), dtype=torch.float64))


class CamryL3FullPolModel(nn.Module):
    """Four trainable complex degree-3 heads on one fixed shared support."""

    def __init__(
        self,
        points_local_m: Any,
        *,
        point_chunk_size: int = 256,
        placement: Any = CAMRY_PLACEMENT,
        lattice: CamryCandidateLattice | None = None,
        site_indices_ijk: Any | None = None,
    ) -> None:
        super().__init__()
        local = np.asarray(points_local_m, dtype=np.float64)
        if local.ndim != 2 or local.shape[1] != 3 or local.shape[0] == 0:
            raise ValueError("points_local_m must have shape [K,3]")
        if local.shape[0] > MAX_ACTIVE_SITES:
            raise ValueError("active support exceeds K<=8192")
        _finite(local, "points_local_m")
        if not np.array_equal(np.asarray(placement.rotation, dtype=np.float64), np.asarray(CAMRY_PLACEMENT.rotation, dtype=np.float64)) or not np.array_equal(np.asarray(placement.translation_m, dtype=np.float64), np.asarray(CAMRY_PLACEMENT.translation_m, dtype=np.float64)):
            raise ValueError("Camry full-pol model must use the fixed Camry placement")
        if site_indices_ijk is not None:
            if lattice is None:
                raise ValueError("site_indices_ijk requires its frozen candidate lattice")
            indices = lattice.validate_ijk(site_indices_ijk)
            lattice_points = lattice.points_local(indices)
            if indices.shape[0] != local.shape[0] or not np.array_equal(local, lattice_points):
                raise ValueError("model support points must equal the selected frozen lattice sites")
        else:
            indices = None
        if int(point_chunk_size) <= 0:
            raise ValueError("point_chunk_size must be positive")
        self.point_chunk_size = int(point_chunk_size)
        self.degree = SH_DEGREE
        self.polarizations = POLARIZATIONS
        self.register_buffer("points_local_m", torch.as_tensor(local, dtype=torch.float64))
        self.register_buffer("points_native_m", torch.as_tensor(placement.apply(local), dtype=torch.float64))
        self.register_buffer("rotation", torch.as_tensor(placement.rotation, dtype=torch.float64))
        self.register_buffer("translation_m", torch.as_tensor(placement.translation_m, dtype=torch.float64))
        self.register_buffer("site_indices_ijk", None if indices is None else torch.as_tensor(indices, dtype=torch.int64))
        self.register_buffer(
            "basis_degree",
            torch.as_tensor([degree for degree in range(SH_DEGREE + 1) for _ in range(2 * degree + 1)], dtype=torch.int64),
        )
        K = local.shape[0]
        self.heads = nn.ModuleDict({polarization: _CamryL3Head(K) for polarization in POLARIZATIONS})
        self.gains = nn.ModuleDict({polarization: GlobalComplexGain().double() for polarization in POLARIZATIONS})

    @property
    def point_count(self) -> int:
        return int(self.points_local_m.shape[0])

    @property
    def coefficient_shape(self) -> tuple[int, int, int]:
        return (4, self.point_count, SH_BASIS_COUNT)

    @property
    def w_re(self) -> torch.Tensor:
        return torch.stack(tuple(self.heads[polarization].w_re for polarization in POLARIZATIONS), dim=0)

    @property
    def w_im(self) -> torch.Tensor:
        return torch.stack(tuple(self.heads[polarization].w_im for polarization in POLARIZATIONS), dim=0)

    def _pol_index(self, polarization: str) -> int:
        key = str(polarization).lower()
        if key not in POLARIZATIONS:
            raise ValueError(f"unsupported polarization {polarization!r}")
        return POLARIZATIONS.index(key)

    def coefficients(self, polarization: str) -> torch.Tensor:
        head = self.heads[str(polarization).lower()]
        return torch.complex(head.w_re, head.w_im)

    def gain(self, polarization: str) -> torch.Tensor:
        gain = self.gains[str(polarization).lower()]
        return torch.polar(torch.exp(gain.log_mag), gain.phase)

    def _record_prediction(self, record: CamryTrainingRecord, coefficients: torch.Tensor) -> torch.Tensor:
        device = coefficients.device
        points = self.points_native_m.to(device=device)
        position = torch.as_tensor(record.position_xyz_m, dtype=torch.float64, device=device)
        frequencies = torch.as_tensor(record.frequencies_hz, dtype=torch.float64, device=device)
        output = torch.zeros((frequencies.numel(),), dtype=torch.complex128, device=device)
        kappa = 4.0 * math.pi / SPEED_OF_LIGHT_M_S
        for start in range(0, self.point_count, self.point_chunk_size):
            stop = min(start + self.point_chunk_size, self.point_count)
            delta = position[None, :] - points[start:stop]
            distance = torch.linalg.vector_norm(delta, dim=1)
            unit_local = (delta / distance[:, None]) @ self.rotation.to(device=device)
            theta = torch.arccos(torch.clamp(unit_local[:, 2], -1.0, 1.0))
            phi = torch.atan2(unit_local[:, 1], unit_local[:, 0])
            basis = real_sh_basis(theta, phi, SH_DEGREE).transpose(0, 1)
            field = torch.sum(basis * coefficients[start:stop], dim=1)
            phase = -kappa * (distance[:, None] - float(record.r0_selected_m)) * frequencies[None, :]
            kernel = torch.exp(torch.complex(torch.zeros_like(phase), phase))
            output = output + torch.sum(kernel * field[:, None], dim=0)
        return output

    def forward_record(self, record: CamryTrainingRecord, polarization: str, *, apply_gain: bool = True) -> torch.Tensor:
        if record.polarization != str(polarization).lower():
            raise ValueError("record polarization does not match requested head")
        result = self._record_prediction(record, self.coefficients(polarization))
        return self.gains[str(polarization).lower()](result) if apply_gain else result

    def forward_channel(self, polarization: str, records: Sequence[CamryTrainingRecord], *, apply_gain: bool = True) -> tuple[torch.Tensor, ...]:
        return tuple(self.forward_record(record, polarization, apply_gain=apply_gain) for record in _as_tuple_records(records))

    def set_l3_coefficients(self, coefficients_by_polarization: Mapping[str, Any]) -> None:
        with torch.no_grad():
            for polarization in POLARIZATIONS:
                values = np.asarray(coefficients_by_polarization[polarization], dtype=np.complex128)
                if values.shape != (self.point_count, SH_BASIS_COUNT):
                    raise ValueError("L3 coefficient shape does not match model")
                head = self.heads[polarization]
                head.w_re.copy_(torch.as_tensor(values.real, dtype=torch.float64, device=head.w_re.device))
                head.w_im.copy_(torch.as_tensor(values.imag, dtype=torch.float64, device=head.w_im.device))

    def set_gains_identity(self) -> None:
        with torch.no_grad():
            for gain in self.gains.values():
                gain.log_mag.zero_()
                gain.phase.zero_()
                gain.initialized.fill_(False)

    @torch.no_grad()
    def normalize_rms_one(self, target_rms: float = 1.0) -> dict[str, float]:
        if not np.isfinite(float(target_rms)) or float(target_rms) <= 0.0:
            raise ValueError("target_rms must be positive and finite")
        scales: dict[str, float] = {}
        for polarization in POLARIZATIONS:
            head = self.heads[polarization]
            rms = torch.sqrt(torch.mean(head.w_re ** 2 + head.w_im ** 2))
            value = float(rms.item())
            if value <= 0.0 or not np.isfinite(value):
                scales[polarization] = 1.0
                continue
            scale = float(target_rms) / value
            head.w_re.mul_(scale)
            head.w_im.mul_(scale)
            self.gains[polarization].log_mag.sub_(math.log(scale))
            scales[polarization] = scale
        return scales

    def effective_coefficients(self, polarization: str) -> torch.Tensor:
        return self.gain(polarization) * self.coefficients(polarization)

    def gain_values(self) -> dict[str, complex]:
        return {polarization: self.gains[polarization].gain_value() for polarization in POLARIZATIONS}


@dataclass(frozen=True)
class DCCGLSResult:
    coefficients_dc: Mapping[str, np.ndarray]
    diagnostics: Mapping[str, tuple[Mapping[str, Any], ...]]
    termination: Mapping[str, str]
    max_iterations: int


def _ragged_subtract(left: RaggedComplexValues, right: RaggedComplexValues) -> RaggedComplexValues:
    if tuple(left.ids) != tuple(right.ids):
        raise ValueError("ragged IDs do not match")
    return RaggedComplexValues(left.ids, tuple(a - b for a, b in zip(left.values, right.values)))


def _ragged_norm2(values: RaggedComplexValues) -> float:
    return float(sum(np.vdot(value, value).real for value in values.values))


def _cpu_rng_tensor(value: Any, label: str) -> torch.Tensor:
    """Validate and normalize a saved Torch RNG byte state to CPU."""

    if not torch.is_tensor(value) or value.ndim != 1 or value.dtype != torch.uint8:
        raise ValueError(f"{label} must be a one-dimensional torch.uint8 RNG tensor")
    return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def _checkpoint_rng_states(payload: Mapping[str, Any]) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
    """Validate checkpoint RNG topology before mutating any generator."""

    if "rng_state" not in payload:
        raise ValueError("Camry checkpoint is missing its CPU RNG state")
    cpu_state = _cpu_rng_tensor(payload["rng_state"], "rng_state")
    saved_cuda = payload.get("cuda_rng_state_all")
    if saved_cuda is None:
        if torch.cuda.is_available():
            raise ValueError("Camry CUDA checkpoint is missing CUDA RNG state")
        return cpu_state, None
    if not isinstance(saved_cuda, (list, tuple)):
        raise ValueError("cuda_rng_state_all must be a list or tuple of RNG byte tensors")
    visible_devices = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if len(saved_cuda) != visible_devices:
        raise ValueError(
            "Camry checkpoint CUDA RNG topology does not match this process "
            f"(saved {len(saved_cuda)}, visible {visible_devices})"
        )
    cuda_states = tuple(_cpu_rng_tensor(value, f"cuda_rng_state_all[{index}]") for index, value in enumerate(saved_cuda))
    return cpu_state, cuda_states


def run_dc_cgls24(
    panel: CamryTrainingPanel,
    operator: CamryL3SparseOperator,
    projectors: Mapping[str, CamryRangeProjector],
    *,
    max_iterations: int = MAX_CGLS_ITERATIONS,
) -> DCCGLSResult:
    """Run zero-start CGLS only in the final renderer's DC/Y00 subspace."""

    iterations = int(max_iterations)
    if not 0 < iterations <= MAX_CGLS_ITERATIONS:
        raise ValueError("DC CGLS iterations must lie in 1..24")
    coeffs: dict[str, np.ndarray] = {}
    all_diagnostics: dict[str, tuple[Mapping[str, Any], ...]] = {}
    terminations: dict[str, str] = {}
    for polarization in POLARIZATIONS:
        records = panel.records(polarization)
        projector = projectors[polarization]
        target = projector.target()

        def D(x: np.ndarray) -> RaggedComplexValues:
            return projector.apply_forward(operator.forward_dc(records, x))

        def DH(values: RaggedComplexValues) -> np.ndarray:
            return operator.adjoint_dc(records, projector.apply_adjoint(values))

        x = np.zeros(operator.point_count, dtype=np.complex128)
        residual = target
        search = DH(residual)
        gamma = float(np.vdot(search, search).real)
        direction = np.array(search, copy=True)
        diagnostics: list[Mapping[str, Any]] = []
        diagnostics.append({"iteration": 0, "residual_norm2": _ragged_norm2(residual), "gradient_norm2": gamma})
        termination = "max_iterations_reached"
        if not np.isfinite(gamma):
            raise FloatingPointError(f"DC CGLS {polarization} initial gradient is non-finite")
        if gamma <= 0.0:
            termination = "normal_equation_stationarity"
        completed = 0
        for iteration in range(1, iterations + 1):
            if termination != "max_iterations_reached" or gamma <= 0.0:
                break
            projected_direction = D(direction)
            denominator = _ragged_norm2(projected_direction)
            if not np.isfinite(denominator):
                raise FloatingPointError(f"DC CGLS {polarization} denominator is non-finite")
            if denominator <= 0.0:
                termination = "nonpositive_search_denominator_breakdown"
                diagnostics.append(
                    {
                        "iteration": int(completed),
                        "residual_norm2": _ragged_norm2(residual),
                        "gradient_norm2": float(gamma),
                        "termination": termination,
                    }
                )
                break
            alpha = gamma / denominator
            if not np.isfinite(alpha):
                raise FloatingPointError(f"DC CGLS {polarization} alpha is non-finite")
            x = x + alpha * direction
            residual = _ragged_subtract(residual, RaggedComplexValues(projected_direction.ids, tuple(alpha * value for value in projected_direction.values)))
            new_search = DH(residual)
            new_gamma = float(np.vdot(new_search, new_search).real)
            if not np.isfinite(new_gamma) or new_gamma < 0.0:
                raise FloatingPointError(f"DC CGLS {polarization} refreshed gradient is non-finite")
            completed = iteration
            if iteration in CGLS_DIAGNOSTIC_STEPS or iteration == iterations or new_gamma <= 0.0:
                diagnostics.append({"iteration": iteration, "residual_norm2": _ragged_norm2(residual), "gradient_norm2": new_gamma, "alpha": float(alpha)})
            if new_gamma <= 0.0:
                termination = "normal_equation_stationarity"
                diagnostics[-1] = {**diagnostics[-1], "termination": termination}
                break
            beta = new_gamma / gamma
            if not np.isfinite(beta):
                raise FloatingPointError(f"DC CGLS {polarization} beta is non-finite")
            direction = new_search + beta * direction
            gamma = new_gamma
        coeffs[polarization] = _readonly(x, np.complex128)
        all_diagnostics[polarization] = tuple(diagnostics)
        if termination == "max_iterations_reached" and completed >= iterations:
            terminations[polarization] = "max_iterations_reached"
        else:
            terminations[polarization] = termination
    return DCCGLSResult(coeffs, all_diagnostics, terminations, iterations)


def embed_dc_coefficients(coefficients_dc: Mapping[str, Any], point_count: int) -> dict[str, np.ndarray]:
    """Embed DC CGLS coefficients into L=3 at index 0, preserving predictions."""

    output: dict[str, np.ndarray] = {}
    for polarization in POLARIZATIONS:
        values = np.asarray(coefficients_dc[polarization], dtype=np.complex128)
        if values.shape != (int(point_count),):
            raise ValueError("DC coefficient shape does not match point count")
        embedded = np.zeros((int(point_count), SH_BASIS_COUNT), dtype=np.complex128)
        embedded[:, 0] = values
        output[polarization] = _readonly(embedded, np.complex128)
    return output


def save_dc_cgls_checkpoint(
    path: str | Path,
    panel: CamryTrainingPanel,
    lattice: CamryCandidateLattice,
    result: DCCGLSResult,
) -> Path:
    """Persist the separate DC initializer without pretending it is L=3 training."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CGLS_SCHEMA,
        "degree": 0,
        "embedded_model_degree": SH_DEGREE,
        "polarizations": list(POLARIZATIONS),
        "lattice": lattice.metadata(),
        "panel": panel.metadata(),
        "max_iterations": int(result.max_iterations),
        "coefficients_dc": {polarization: np.asarray(result.coefficients_dc[polarization], dtype=np.complex128) for polarization in POLARIZATIONS},
        "diagnostics": {polarization: tuple(result.diagnostics[polarization]) for polarization in POLARIZATIONS},
        "termination": dict(result.termination),
        "test_opened": False,
    }
    temporary = target.with_name(target.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)
    return target


def load_dc_cgls_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load and validate the distinct DC-only CGLS checkpoint schema."""

    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(Path(path), map_location="cpu")
    if payload.get("schema") != CGLS_SCHEMA or int(payload.get("degree", -1)) != 0:
        raise ValueError("not a Camry DC-CGLS checkpoint")
    if int(payload.get("embedded_model_degree", -1)) != SH_DEGREE:
        raise ValueError("DC checkpoint does not target the degree-3 model")
    if tuple(payload.get("polarizations", ())) != POLARIZATIONS:
        raise ValueError("DC checkpoint polarization vocabulary does not match")
    if payload.get("test_opened", True):
        raise ValueError("DC checkpoint reports opened TEST data")
    return payload


def build_camry_candidate_lattice(panel: CamryTrainingPanel) -> CamryCandidateLattice:
    """Named construction helper for downstream CLI/validator integration."""

    return CamryCandidateLattice.from_panel(panel)


class CamryFullPolTrainer:
    """Full-panel range objective, priors, AdamW/cosine lifecycle, and checkpoints."""

    def __init__(
        self,
        panel: CamryTrainingPanel,
        model: CamryL3FullPolModel,
        *,
        require_retention: bool = True,
    ) -> None:
        self.panel = panel
        self.model = model
        self.lattice = CamryCandidateLattice.from_panel(panel)
        if model.point_count > MAX_ACTIVE_SITES:
            raise ValueError("model support exceeds K<=8192")
        if model.site_indices_ijk is None:
            raise ValueError("Camry trainer requires support bound to selected frozen lattice indices")
        self.lattice.validate_ijk(model.site_indices_ijk.detach().cpu().numpy())
        if not np.array_equal(self.lattice.points_local(model.site_indices_ijk.detach().cpu().numpy()), model.points_local_m.detach().cpu().numpy()):
            raise ValueError("Camry trainer support points do not match selected lattice indices")
        self.operator = CamryL3SparseOperator(model.points_native_m.detach().cpu().numpy())
        self.projectors = {
            polarization: CamryRangeProjector(panel.records(polarization), require_retention=require_retention)
            for polarization in POLARIZATIONS
        }
        self.targets = {polarization: self.projectors[polarization].target() for polarization in POLARIZATIONS}
        self.target_energy = {polarization: _ragged_norm2(self.targets[polarization]) for polarization in POLARIZATIONS}
        if any(value <= 0.0 or not np.isfinite(value) for value in self.target_energy.values()):
            raise ValueError("every polarization must have positive finite projected TRAIN energy")
        self.Q0: float | None = None
        self.S_group: float | None = None
        self.S_energy: float | None = None
        self.optimizer: AdamW | None = None
        self.scheduler: CosineAnnealingWarmRestarts | None = None
        self.history: list[dict[str, Any]] = []
        self.optimizer_diagnostics: list[dict[str, Any]] = []
        self.optimizer_updates = 0
        self.current_epoch = 0
        self.current_batch_index = 0
        self.best_epoch: int | None = None
        self.best_q: float | None = None

    def _torch_targets(
        self,
        polarization: str,
        device: torch.device,
        records: Sequence[CamryTrainingRecord] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        target = self.targets[polarization]
        if records is None:
            values = target.values
        else:
            by_id = {_identity_key(identity): value for identity, value in zip(target.ids, target.values)}
            values = tuple(by_id[_identity_key(record.identity)] for record in records)
        return tuple(torch.as_tensor(value, dtype=torch.complex128, device=device) for value in values)

    def _projected_residuals(self, polarization: str, records: Sequence[CamryTrainingRecord], predictions: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        projected = self.projectors[polarization].forward_torch(predictions, records=records)
        target_values = self._torch_targets(polarization, projected[0].device, records)
        return tuple(predicted - target for predicted, target in zip(projected, target_values))

    def data_loss_torch(self, *, batch_index: int | None = None) -> torch.Tensor:
        U = self.panel.logical_batch_count
        total: torch.Tensor | None = None
        for polarization in POLARIZATIONS:
            records = self.panel.records(polarization)
            if batch_index is None:
                selected = records
                scale = 1.0 / 4.0
            else:
                if int(batch_index) >= len(records):
                    continue
                selected = (records[int(batch_index)],)
                scale = float(U) / 4.0
            predictions = self.model.forward_channel(polarization, selected, apply_gain=True)
            residuals = self._projected_residuals(polarization, selected, predictions)
            value = scale * sum(torch.sum(torch.abs(residual) ** 2) for residual in residuals) / float(self.target_energy[polarization])
            total = value if total is None else total + value
        if total is None:
            raise ValueError("logical batch contains no observations")
        return total

    def _effective_coefficients_stack(self) -> torch.Tensor:
        return torch.stack([self.model.effective_coefficients(polarization) for polarization in POLARIZATIONS], dim=0)

    def _normalized_coefficient_magnitude_sq(self) -> torch.Tensor:
        """Return |g_p w_p|^2 / E_p for every channel/site/SH coefficient."""

        coefficients = self._effective_coefficients_stack()
        energies = torch.as_tensor([self.target_energy[p] for p in POLARIZATIONS], dtype=torch.float64, device=coefficients.device)
        normalized = coefficients / torch.sqrt(energies)[:, None, None]
        return normalized.real ** 2 + normalized.imag ** 2

    def _normalized_energy_scale(self) -> float:
        """Freeze the exact gain- and target-energy-normalized SH scale."""

        magnitude_sq = self._normalized_coefficient_magnitude_sq()
        scale = float(torch.sum(magnitude_sq).item()) / float(self.model.point_count)
        if not np.isfinite(scale) or scale <= 0.0:
            raise FloatingPointError("Camry normalized SH energy scale is non-positive/non-finite")
        return scale

    def prior_terms_torch(self) -> tuple[torch.Tensor, torch.Tensor]:
        coeff = self._effective_coefficients_stack()
        magnitude_sq = self._normalized_coefficient_magnitude_sq()
        group = torch.sqrt(torch.sum(magnitude_sq, dim=(0, 2)) + GROUP_EPS ** 2) - GROUP_EPS
        group_raw = torch.sum(group) / float(self.model.point_count)
        degree = self.model.basis_degree.to(device=coeff.device, dtype=torch.float64)
        sh_raw = torch.sum(magnitude_sq * (degree * (degree + 1.0))[None, None, :]) / float(self.model.point_count)
        return group_raw, sh_raw

    def prior_loss_torch(self) -> torch.Tensor:
        group_raw, sh_raw = self.prior_terms_torch()
        if self.Q0 is None or self.S_group is None or self.S_energy is None:
            return torch.zeros((), dtype=torch.float64, device=group_raw.device)
        return (1.0e-3 * float(self.Q0) * group_raw / float(self.S_group)) + (1.0e-4 * float(self.Q0) * sh_raw / float(self.S_energy))

    def loss_torch(self, *, batch_index: int | None = None) -> torch.Tensor:
        return self.data_loss_torch(batch_index=batch_index) + self.prior_loss_torch()

    @torch.no_grad()
    def metrics(self) -> dict[str, Any]:
        q_by_pol: dict[str, float] = {}
        native_by_pol: dict[str, float] = {}
        native_raw_by_pol: dict[str, float] = {}
        retained_by_pol: dict[str, float] = {}
        pooled_num = 0.0
        pooled_den = 0.0
        retained_pooled_num = 0.0
        retained_pooled_den = 0.0
        for polarization in POLARIZATIONS:
            records = self.panel.records(polarization)
            retained_by_pol[polarization] = self.projectors[polarization].retained_energy_fraction()
            native_target_energy = float(sum(np.vdot(record.response_selected, record.response_selected).real for record in records))
            retained_pooled_num += retained_by_pol[polarization] * native_target_energy
            retained_pooled_den += native_target_energy
            predictions = self.model.forward_channel(polarization, records, apply_gain=True)
            projected = self.projectors[polarization].forward_torch(predictions)
            target = self._torch_targets(polarization, projected[0].device)
            residual = tuple(predicted - observed for predicted, observed in zip(projected, target))
            numerator = float(sum(torch.sum(torch.abs(value) ** 2).item() for value in residual))
            denominator = float(self.target_energy[polarization])
            q_by_pol[polarization] = numerator / denominator
            native_num = 0.0
            native_den = 0.0
            native_raw_num = 0.0
            native_raw_den = 0.0
            for record, prediction in zip(records, predictions):
                pred_np = prediction.detach().cpu().numpy()
                native_num += float(np.vdot(pred_np - record.response_selected, pred_np - record.response_selected).real)
                native_den += float(np.vdot(record.response_selected, record.response_selected).real)
                if record.source_af_applied:
                    pred_raw = pred_np * np.exp(-1j * np.float64(record.ph_correct_raw_rad))
                else:
                    pred_raw = pred_np
                native_raw_num += float(np.vdot(pred_raw - record.response_raw, pred_raw - record.response_raw).real)
                native_raw_den += float(np.vdot(record.response_raw, record.response_raw).real)
            native_by_pol[polarization] = native_num / native_den if native_den > 0.0 else float("nan")
            native_raw_by_pol[polarization] = native_raw_num / native_raw_den if native_raw_den > 0.0 else float("nan")
            pooled_num += numerator
            pooled_den += denominator
        return {
            "range_relmse_by_polarization": q_by_pol,
            "range_macro_relmse": float(np.mean(tuple(q_by_pol.values()))),
            "range_energy_pooled_relmse": pooled_num / pooled_den,
            "native_relmse_by_polarization": native_by_pol,
            "native_raw_relmse_by_polarization": native_raw_by_pol,
            "record_counts": self.panel.record_counts,
            "frequency_sample_counts": self.panel.frequency_sample_counts,
            "N_T_by_polarization": {polarization: self.projectors[polarization].normalization() for polarization in POLARIZATIONS},
            "target_energy_by_polarization": dict(self.target_energy),
            "range_projector_retained_energy_fraction_by_polarization": retained_by_pol,
            "range_projector_retained_energy_fraction_pooled": retained_pooled_num / retained_pooled_den,
            "test_opened": False,
        }

    @torch.no_grad()
    def initialize_from_dc_cgls(self, result: DCCGLSResult) -> dict[str, Any]:
        embedded = embed_dc_coefficients(result.coefficients_dc, self.model.point_count)
        self.model.set_l3_coefficients(embedded)
        self.model.set_gains_identity()
        gain_init: dict[str, Any] = {}
        for polarization in POLARIZATIONS:
            records = self.panel.records(polarization)
            raw_predictions = self.model.forward_channel(polarization, records, apply_gain=False)
            projected = self.projectors[polarization].forward_torch(raw_predictions)
            target = self._torch_targets(polarization, projected[0].device)
            pred_flat = torch.cat(projected)
            target_flat = torch.cat(target)
            gain = self.model.gains[polarization]
            gain.maybe_init_scale(pred_flat, target_flat)
            if not bool(gain.initialized) or not bool(torch.isfinite(gain.log_mag)) or not bool(torch.isfinite(gain.phase)):
                raise FloatingPointError(f"Camry {polarization.upper()} gain warm start is pending or non-finite")
            gain_init[polarization] = gain.gain_value()
        scales = self.model.normalize_rms_one()
        metrics = self.metrics()
        self.Q0 = float(metrics["range_macro_relmse"])
        group_raw, sh_raw = self.prior_terms_torch()
        self.S_group = max(float(group_raw.item()), GROUP_EPS)
        self.S_energy = self._normalized_energy_scale()
        return {"cgls": result, "gain_init": gain_init, "rms_scales": scales, "initial_metrics": metrics, "Q0": self.Q0, "S_group": self.S_group, "S_energy": self.S_energy}

    def build_optimizer(self, *, scene_lr: float = 3.0e-3, gain_lr: float = 3.0e-3) -> tuple[AdamW, CosineAnnealingWarmRestarts]:
        if scene_lr <= 0.0 or gain_lr <= 0.0 or not np.isfinite(scene_lr + gain_lr):
            raise ValueError("learning rates must be positive and finite")
        parameter_groups = []
        for polarization in POLARIZATIONS:
            head = self.model.heads[polarization]
            parameter_groups.append(
                {"params": [head.w_re, head.w_im], "lr": float(scene_lr), "weight_decay": 0.0, "eps": 1.0e-8, "name": f"head_{polarization}"}
            )
            parameter_groups.append(
                {"params": list(self.model.gains[polarization].parameters()), "lr": float(gain_lr), "weight_decay": 0.0, "eps": 1.0e-8, "name": f"gain_{polarization}"}
            )
        self.optimizer = AdamW(
            parameter_groups,
            betas=(0.9, 0.999),
            weight_decay=0.0,
            eps=1.0e-8,
        )
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=1.0e-6)
        return self.optimizer, self.scheduler

    def _checkpoint_payload(self, *, epoch: int, best_epoch: int | None, best_q: float | None, phase: str) -> dict[str, Any]:
        return {
            "schema": CHECKPOINT_SCHEMA,
            "model_schema": SCHEMA,
            "degree": SH_DEGREE,
            "sh_degree": SH_DEGREE,
            "basis_count": SH_BASIS_COUNT,
            "basis_kind": "real_sh",
            "sh_order": "degree_major_m_minus_l_to_l",
            "coefficients_are_complex": True,
            "polarizations": list(POLARIZATIONS),
            "gain_keys": list(POLARIZATIONS),
            "point_count": self.model.point_count,
            "selected_site_indices": None if self.model.site_indices_ijk is None else self.model.site_indices_ijk.detach().cpu().tolist(),
            "selected_site_points_local_m": self.model.points_local_m.detach().cpu().tolist(),
            "placement": {
                "name": CAMRY_PLACEMENT.name,
                "rotation": np.asarray(CAMRY_PLACEMENT.rotation, dtype=np.float64).tolist(),
                "translation_m": np.asarray(CAMRY_PLACEMENT.translation_m, dtype=np.float64).tolist(),
            },
            "lattice": self.lattice.metadata(),
            "panel": self.panel.metadata(),
            "range_projectors": {polarization: self.projectors[polarization].metadata() for polarization in POLARIZATIONS},
            "test_opened": False,
            "epoch": int(epoch),
            "batch_index": int(self.current_batch_index),
            "optimizer_updates": int(self.optimizer_updates),
            "best_epoch": None if best_epoch is None else int(best_epoch),
            "best_q": None if best_q is None else float(best_q),
            "phase": str(phase),
            "Q0": self.Q0,
            "S_group": self.S_group,
            "S_energy": self.S_energy,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": None if self.optimizer is None else self.optimizer.state_dict(),
            "scheduler_state_dict": None if self.scheduler is None else self.scheduler.state_dict(),
            "rng_state": _cpu_rng_tensor(torch.get_rng_state(), "rng_state"),
            "cuda_rng_state_all": None if not torch.cuda.is_available() else tuple(
                _cpu_rng_tensor(value, f"cuda_rng_state_all[{index}]")
                for index, value in enumerate(torch.cuda.get_rng_state_all())
            ),
            "history": tuple(self.history),
            "optimizer_diagnostics": tuple(self.optimizer_diagnostics),
        }

    @staticmethod
    def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        torch.save(dict(payload), temporary)
        temporary.replace(path)

    def _verify_optimizer_state_devices(self) -> None:
        if self.optimizer is None:
            return
        parameter_groups = {
            id(parameter): group
            for group in self.optimizer.param_groups
            for parameter in group.get("params", ())
        }
        optimizer_defaults = getattr(self.optimizer, "defaults", {})
        for parameter, state in self.optimizer.state.items():
            group = parameter_groups.get(id(parameter))
            if group is None:
                raise RuntimeError("restored optimizer state contains an unregistered parameter")
            capturable = bool(group.get("capturable", optimizer_defaults.get("capturable", False)))
            fused = bool(group.get("fused", optimizer_defaults.get("fused", False)))
            expected_step_device = parameter.device if capturable or fused else torch.device("cpu")
            for key, value in tuple(state.items()):
                if not torch.is_tensor(value):
                    continue
                expected_device = expected_step_device if key == "step" else parameter.device
                if value.device != expected_device:
                    raise RuntimeError(
                        f"optimizer state {key!r} is on {value.device}, expected {expected_device} "
                        f"for parameter device {parameter.device}"
                    )

    def _optimizer_diagnostic_groups(self) -> tuple[tuple[str, str, tuple[torch.Tensor, ...], Mapping[str, Any]], ...]:
        """Return the named scene-head and learned-gain groups in optimizer order."""

        if self.optimizer is None or not hasattr(self.optimizer, "param_groups"):
            return ()
        groups: list[tuple[str, str, tuple[torch.Tensor, ...], Mapping[str, Any]]] = []
        seen: set[str] = set()
        for group in self.optimizer.param_groups:
            name = group.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("every Camry optimizer parameter group must have a stable name")
            if name in seen:
                raise ValueError(f"duplicate Camry optimizer parameter-group name: {name}")
            seen.add(name)
            if name.startswith("head_"):
                kind = "scene"
            elif name.startswith("gain_"):
                kind = "gain"
            else:
                raise ValueError(f"unexpected Camry optimizer parameter-group name: {name}")
            parameters = tuple(parameter for parameter in group.get("params", ()) if torch.is_tensor(parameter))
            if not parameters:
                raise ValueError(f"Camry optimizer parameter group {name} is empty")
            groups.append((name, kind, parameters, group))
        return tuple(groups)

    @staticmethod
    def _diagnostic_rms(sum_square: float, count: int, label: str) -> float:
        if count <= 0 or not np.isfinite(sum_square) or sum_square < 0.0:
            raise FloatingPointError(f"Camry optimizer diagnostic {label} is invalid")
        value = math.sqrt(sum_square / float(count))
        if not np.isfinite(value):
            raise FloatingPointError(f"Camry optimizer diagnostic {label} is non-finite")
        return float(value)

    @staticmethod
    def _diagnostic_tensor_sum_square(value: torch.Tensor, label: str) -> float:
        detached = value.detach()
        if not bool(torch.isfinite(detached).all()):
            raise FloatingPointError(f"Camry optimizer diagnostic {label} is non-finite")
        result = float(torch.sum(torch.abs(detached).to(dtype=torch.float64) ** 2).item())
        if not np.isfinite(result):
            raise FloatingPointError(f"Camry optimizer diagnostic {label} is non-finite")
        return result

    def _optimizer_step_with_diagnostics(self, *, epoch: int) -> dict[str, Any] | None:
        """Measure one optimizer step without changing the optimizer behavior."""

        assert self.optimizer is not None
        groups = self._optimizer_diagnostic_groups()
        if not groups:
            self.optimizer.step()
            return None
        before: dict[str, tuple[torch.Tensor, ...]] = {}
        measurements: dict[str, dict[str, Any]] = {}
        for name, kind, parameters, group in groups:
            snapshots = tuple(parameter.detach().clone() for parameter in parameters)
            before[name] = snapshots
            parameter_sum_square = sum(
                self._diagnostic_tensor_sum_square(parameter, f"{name}.parameter") for parameter in parameters
            )
            parameter_count = sum(int(parameter.numel()) for parameter in parameters)
            gradient_sum_square = 0.0
            gradient_count = parameter_count
            for parameter in parameters:
                if parameter.grad is None:
                    continue
                if tuple(parameter.grad.shape) != tuple(parameter.shape):
                    raise FloatingPointError(f"Camry optimizer gradient shape mismatch in {name}")
                gradient_sum_square += self._diagnostic_tensor_sum_square(parameter.grad, f"{name}.gradient")
            optimizer_state = getattr(self.optimizer, "state", {})
            optimizer_step_values = []
            for parameter in parameters:
                state = optimizer_state.get(parameter, {}) if hasattr(optimizer_state, "get") else {}
                if "step" not in state:
                    optimizer_step_values.append(0.0)
                    continue
                step_value = state["step"]
                if torch.is_tensor(step_value):
                    if step_value.numel() != 1 or not bool(torch.isfinite(step_value).all()):
                        raise FloatingPointError(f"Camry optimizer step state for {name} is invalid")
                    optimizer_step_values.append(float(step_value.detach().item()))
                else:
                    step_float = float(step_value)
                    if not np.isfinite(step_float):
                        raise FloatingPointError(f"Camry optimizer step state for {name} is non-finite")
                    optimizer_step_values.append(step_float)
            lr = float(group.get("lr", float("nan")))
            if not np.isfinite(lr):
                raise FloatingPointError(f"Camry optimizer learning rate for {name} is non-finite")
            measurements[name] = {
                "kind": kind,
                "gradient_rms": self._diagnostic_rms(gradient_sum_square, gradient_count, f"{name}.gradient_rms"),
                "parameter_rms_before": self._diagnostic_rms(parameter_sum_square, parameter_count, f"{name}.parameter_rms_before"),
                "parameter_norm_before": math.sqrt(parameter_sum_square),
                "optimizer_step_before": float(max(optimizer_step_values)),
                "lr_used": lr,
            }
        self.optimizer.step()
        for name, _kind, parameters, _group in groups:
            update_sum_square = 0.0
            parameter_count = 0
            for parameter, snapshot in zip(parameters, before[name]):
                update = parameter.detach() - snapshot
                update_sum_square += self._diagnostic_tensor_sum_square(update, f"{name}.update")
                parameter_count += int(parameter.numel())
            update_rms = self._diagnostic_rms(update_sum_square, parameter_count, f"{name}.update_rms")
            relative_step = math.sqrt(update_sum_square) / max(
                float(measurements[name]["parameter_norm_before"]), np.finfo(np.float64).tiny
            )
            if not np.isfinite(relative_step):
                raise FloatingPointError(f"Camry optimizer relative step for {name} is non-finite")
            measurements[name]["update_rms"] = update_rms
            measurements[name]["relative_step"] = float(relative_step)
            del measurements[name]["parameter_norm_before"]
        return {
            "epoch": int(epoch),
            "optimizer_update": int(self.optimizer_updates + 1),
            "groups": measurements,
        }

    @staticmethod
    def _aggregate_diagnostic_values(values: Sequence[float]) -> dict[str, float]:
        if not values or not all(np.isfinite(value) for value in values):
            raise FloatingPointError("Camry optimizer diagnostic aggregate is empty or non-finite")
        return {"mean": float(np.mean(values)), "max": float(np.max(values)), "last": float(values[-1])}

    def _epoch_optimizer_diagnostics(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        next_epoch_lr_by_group: Mapping[str, float],
    ) -> dict[str, Any] | None:
        if not records:
            return None
        group_records: dict[str, list[Mapping[str, Any]]] = {}
        for record in records:
            for name, values in record["groups"].items():
                group_records.setdefault(name, []).append(values)

        def aggregate(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            return {
                "kind": values[-1]["kind"],
                "gradient_rms": self._aggregate_diagnostic_values([float(value["gradient_rms"]) for value in values]),
                "parameter_rms_before": self._aggregate_diagnostic_values([float(value["parameter_rms_before"]) for value in values]),
                "optimizer_step_before": self._aggregate_diagnostic_values([float(value["optimizer_step_before"]) for value in values]),
                "lr_used": self._aggregate_diagnostic_values([float(value["lr_used"]) for value in values]),
                "update_rms": self._aggregate_diagnostic_values([float(value["update_rms"]) for value in values]),
                "relative_step": self._aggregate_diagnostic_values([float(value["relative_step"]) for value in values]),
            }

        groups = {name: aggregate(values) for name, values in group_records.items()}

        def aggregate_kind(kind: str) -> dict[str, Any]:
            values = [
                group_values
                for name, group_values in group_records.items()
                if groups[name]["kind"] == kind
                for group_values in group_values
            ]
            return {
                "group_count": sum(1 for name in group_records if groups[name]["kind"] == kind),
                "gradient_rms": self._aggregate_diagnostic_values([float(value["gradient_rms"]) for value in values]),
                "parameter_rms_before": self._aggregate_diagnostic_values([float(value["parameter_rms_before"]) for value in values]),
                "optimizer_step_before": self._aggregate_diagnostic_values([float(value["optimizer_step_before"]) for value in values]),
                "lr_used": self._aggregate_diagnostic_values([float(value["lr_used"]) for value in values]),
                "update_rms": self._aggregate_diagnostic_values([float(value["update_rms"]) for value in values]),
                "relative_step": self._aggregate_diagnostic_values([float(value["relative_step"]) for value in values]),
            }

        return {
            "logical_updates": int(len(records)),
            "groups": groups,
            "scene": aggregate_kind("scene"),
            "gain": aggregate_kind("gain"),
            "lr_used_by_group": {
                name: {"first": float(values[0]["lr_used"]), "last": float(values[-1]["lr_used"])}
                for name, values in group_records.items()
            },
            "next_epoch_lr_by_group": {name: float(value) for name, value in next_epoch_lr_by_group.items()},
        }

    def save_checkpoint(self, path: str | Path, *, epoch: int, best_epoch: int | None, best_q: float | None, phase: str) -> Path:
        target = Path(path)
        self._atomic_torch_save(self._checkpoint_payload(epoch=epoch, best_epoch=best_epoch, best_q=best_q, phase=phase), target)
        return target

    def fit(
        self,
        *,
        epochs: int = 150,
        seed: int = 42,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        total_epochs = int(epochs)
        if total_epochs != 150:
            raise ValueError("Camry full-pol training is frozen at exactly 150 epochs")
        if self.optimizer is None or self.scheduler is None:
            self.build_optimizer()
        if self.optimizer_updates == 0 and not self.history:
            torch.manual_seed(int(seed))
        U = self.panel.logical_batch_count
        best_q = self.best_q
        best_epoch = self.best_epoch
        output = None if output_dir is None else Path(output_dir)
        for epoch in range(self.current_epoch + 1, total_epochs + 1):
            self.model.train()
            epoch_optimizer_diagnostics: list[dict[str, Any]] = []
            for batch_index in range(U):
                assert self.optimizer is not None
                self.current_batch_index = batch_index
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.loss_torch(batch_index=batch_index)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Camry full-pol training loss is non-finite")
                loss.backward()
                diagnostic = self._optimizer_step_with_diagnostics(epoch=epoch)
                self.optimizer_updates += 1
                if diagnostic is not None:
                    diagnostic["optimizer_update"] = int(self.optimizer_updates)
                    self.optimizer_diagnostics.append(diagnostic)
                    epoch_optimizer_diagnostics.append(diagnostic)
                self.current_batch_index = batch_index + 1
            assert self.scheduler is not None
            self.scheduler.step()
            next_epoch_lr_by_group = {}
            if hasattr(self.optimizer, "param_groups"):
                next_epoch_lr_by_group = {
                    str(group["name"]): float(group["lr"])
                    for group in self.optimizer.param_groups
                    if isinstance(group.get("name"), str)
                }
            self.model.eval()
            metrics = self.metrics()
            q = float(metrics["range_macro_relmse"])
            if not np.isfinite(q) or any(
                not np.isfinite(float(metrics["range_relmse_by_polarization"][polarization]))
                for polarization in POLARIZATIONS
            ):
                raise FloatingPointError("Camry full-pol TRAIN metric is non-finite")
            row = {"epoch": epoch, "optimizer_updates": self.optimizer_updates, **metrics, "loss": q}
            optimizer_diagnostics = self._epoch_optimizer_diagnostics(
                epoch_optimizer_diagnostics,
                next_epoch_lr_by_group=next_epoch_lr_by_group,
            )
            if optimizer_diagnostics is not None:
                row["optimizer_diagnostics"] = optimizer_diagnostics
            self.history.append(row)
            self.current_epoch = epoch
            self.current_batch_index = 0
            if best_q is None or q < best_q:
                best_q = q
                best_epoch = epoch
                if output is not None:
                    self.save_checkpoint(output / "checkpoint_best.pt", epoch=epoch, best_epoch=best_epoch, best_q=best_q, phase="best")
            if output is not None:
                self.save_checkpoint(output / "checkpoint_latest.pt", epoch=epoch, best_epoch=best_epoch, best_q=best_q, phase="latest")
        self.best_q = best_q
        self.best_epoch = best_epoch
        if output is not None:
            self.save_checkpoint(output / "checkpoint_final.pt", epoch=self.current_epoch, best_epoch=best_epoch, best_q=best_q, phase="final")
        return {
            "epochs": total_epochs,
            "optimizer_updates": self.optimizer_updates,
            "best_epoch": best_epoch,
            "best_q": best_q,
            "history": tuple(self.history),
            "optimizer_diagnostics": tuple(self.optimizer_diagnostics),
            "final_metrics": self.metrics(),
        }

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        try:
            payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        except TypeError:  # Older Torch releases do not expose weights_only.
            payload = torch.load(Path(path), map_location="cpu")
        if payload.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("checkpoint schema is not the Camry full-pol L3 schema")
        cpu_rng_state, cuda_rng_states = _checkpoint_rng_states(payload)
        raw_optimizer_diagnostics = payload.get("optimizer_diagnostics", ())
        if not isinstance(raw_optimizer_diagnostics, (list, tuple)) or any(
            not isinstance(record, Mapping) or not isinstance(record.get("groups"), Mapping)
            for record in raw_optimizer_diagnostics
        ):
            raise ValueError("Camry checkpoint optimizer diagnostics are malformed")
        if int(payload.get("degree", -1)) != SH_DEGREE or int(payload.get("sh_degree", -1)) != SH_DEGREE or int(payload.get("basis_count", -1)) != SH_BASIS_COUNT:
            raise ValueError("checkpoint is not degree-3 with the full 16-band layout")
        if payload.get("basis_kind") != "real_sh" or payload.get("sh_order") != "degree_major_m_minus_l_to_l" or payload.get("coefficients_are_complex") is not True:
            raise ValueError("checkpoint SH basis metadata does not match the Camry L=3 contract")
        if tuple(payload.get("polarizations", ())) != POLARIZATIONS:
            raise ValueError("checkpoint polarization vocabulary does not match")
        if tuple(payload.get("gain_keys", ())) != POLARIZATIONS:
            raise ValueError("checkpoint gain vocabulary does not match")
        if int(payload.get("point_count", -1)) != self.model.point_count:
            raise ValueError("checkpoint point count does not match fixed support")
        placement = payload.get("placement", {})
        if placement.get("name") != CAMRY_PLACEMENT.name or not np.array_equal(np.asarray(placement.get("rotation"), dtype=np.float64), np.asarray(CAMRY_PLACEMENT.rotation, dtype=np.float64)) or not np.array_equal(np.asarray(placement.get("translation_m"), dtype=np.float64), np.asarray(CAMRY_PLACEMENT.translation_m, dtype=np.float64)):
            raise ValueError("checkpoint placement does not match the fixed Camry placement")
        checkpoint_points = np.asarray(payload.get("selected_site_points_local_m"), dtype=np.float64)
        if checkpoint_points.shape != tuple(self.model.points_local_m.shape) or not np.array_equal(checkpoint_points, self.model.points_local_m.detach().cpu().numpy()):
            raise ValueError("checkpoint selected support points do not match the model")
        checkpoint_indices = payload.get("selected_site_indices")
        if checkpoint_indices is not None:
            if self.model.site_indices_ijk is None or not np.array_equal(np.asarray(checkpoint_indices, dtype=np.int64), self.model.site_indices_ijk.detach().cpu().numpy()):
                raise ValueError("checkpoint selected lattice indices do not match the model")
        if payload.get("test_opened", True):
            raise ValueError("Camry training checkpoint reports opened TEST data")
        self.model.load_state_dict(payload["model_state_dict"])
        if self.optimizer is not None and payload.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
            self._verify_optimizer_state_devices()
        if self.scheduler is not None and payload.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        self.Q0 = payload.get("Q0")
        self.S_group = payload.get("S_group")
        self.S_energy = payload.get("S_energy")
        self.history = [dict(row) for row in payload.get("history", ())]
        self.optimizer_updates = int(payload.get("optimizer_updates", 0))
        self.optimizer_diagnostics = [dict(record) for record in raw_optimizer_diagnostics]
        self.current_epoch = int(payload.get("epoch", len(self.history)))
        self.current_batch_index = int(payload.get("batch_index", 0))
        if not 0 <= self.current_epoch <= 150 or self.current_batch_index != 0:
            raise ValueError("Camry checkpoints must represent an epoch boundary in 0..150")
        expected_updates = int(self.current_epoch * self.panel.logical_batch_count)
        if self.optimizer_updates != expected_updates:
            raise ValueError("Camry checkpoint optimizer update count is not 150*U-consistent")
        if len(self.optimizer_diagnostics) != self.optimizer_updates:
            raise ValueError("Camry checkpoint optimizer diagnostics are not update-continuous")
        self.best_epoch = None if payload.get("best_epoch") is None else int(payload["best_epoch"])
        self.best_q = None if payload.get("best_q") is None else float(payload["best_q"])
        return payload


__all__ = [
    "SCHEMA",
    "CHECKPOINT_SCHEMA",
    "CGLS_SCHEMA",
    "POLARIZATIONS",
    "SH_DEGREE",
    "SH_BASIS_COUNT",
    "MAX_ACTIVE_SITES",
    "RESERVED_SITE_CAPACITY",
    "CamryTrainingRecord",
    "CamryTrainingPanel",
    "select_camry_train_panel",
    "CamryCandidateLattice",
    "rank_candidate_sites",
    "CamryL3SparseOperator",
    "CamryRangeProjector",
    "CamryL3FullPolModel",
    "DCCGLSResult",
    "run_dc_cgls24",
    "embed_dc_coefficients",
    "save_dc_cgls_checkpoint",
    "load_dc_cgls_checkpoint",
    "build_camry_candidate_lattice",
    "CamryFullPolTrainer",
]
