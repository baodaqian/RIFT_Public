"""Sealed native-complex directional-SH ROI control for GOTCHA Step 3.

This module is intentionally separate from the bounded Torch readiness smoke.
It provides a small, NumPy-first control implementation for the next measured
fit package: every fixed-support voxel is active, each voxel carries four real
degree-1 spherical-harmonic coefficients with complex real/imaginary field
values, and the coherent field is accumulated over spatial chunks before the
residual is formed.  The local control case uses only in-memory records.

The module does not load an archive, inspect PACE, submit jobs, fit gains, or
perform autofocus/calibration.  ``SourceAFObservation`` instances from the
existing source-AF module are accepted by duck-typed contract checks so this
file can also be loaded directly by the local no-Torch validators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA = "rift_gotcha_step3_native_complex_directional_sh_roi_fit_v1"
CHECKPOINT_SCHEMA = f"{SCHEMA}.checkpoint"
READOUT_LABEL = "coarse-support readout—not .1m fitted resolution"
SOURCE_AF_REPRESENTATION = "source_af"
SOURCE_AF_FORMULA = "r0_src=float64(r0_raw)+float64(r_correct_raw); response_src=complex128(response_raw)*exp(+i*float64(ph_correct_raw))"
SH_BASIS_CONVENTION = "[Y00=1/sqrt(4pi), Y1x=sqrt(3/(4pi))*ux, Y1y=sqrt(3/(4pi))*uy, Y1z=sqrt(3/(4pi))*uz]"
SPEED_OF_LIGHT_M_S = 299_792_458.0
GLOBAL_COMPLEX_GAIN = 1.0 + 0.0j
TRAIN_SECTOR_ID = 2
VALIDATION_SECTOR_ID = 1
TOPHAT_TRAIN_PASS_IDS = (1, 7)
TOPHAT_TRAIN_SECTOR_IDS = (2, 92, 182, 272)
TOPHAT_VALIDATION_SECTOR_IDS = (1, 91, 181, 271)
DEFAULT_SPATIAL_CHUNK_SIZE = 128
DEFAULT_FREQUENCY_CHUNK_SIZE = 64
DEFAULT_UPDATES = 12
DEFAULT_CHECKPOINT_STEPS = (0, 4, 8, 12)
MAX_CGLS_ITERATIONS = 24
CGLS_DIAGNOSTIC_STEPS = (0, 8, 16, 24)
RANGE_SUBSPACE_SAMPLES = 8193
RANGE_SUBSPACE_GOLDEN_ITERATIONS = 64
RANGE_SUBSPACE_GUARD_CELLS = 2
RANGE_SUBSPACE_RETENTION_THRESHOLD = 0.95
RANGE_SUBSPACE_HARMONIC_TOLERANCE = 2.0e-11
RANGE_SUBSPACE_SCHEMA = f"{SCHEMA}.range_subspace"
RANGE_COMPARISON_SCHEMA = f"{SCHEMA}.range_comparison"
DEFAULT_FAKE_PROVENANCE = {
    "mode": "local_fake",
    "source_kind": "in_memory_fake_only",
    "archive_opened": False,
    "test_payload_opened": False,
}


def _readonly(value: Any, dtype: Any) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _finite(value: Any, label: str) -> None:
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")


def _id_key(identity: Any) -> tuple[int, str, int, int]:
    try:
        return (
            int(identity.pass_id),
            str(identity.polarization).lower(),
            int(identity.sector_id),
            int(identity.pulse_index),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError("canonical identity must expose pass/polarization/sector/pulse") from exc


def source_af_values(
    r0_raw_m: float,
    response_raw: Any,
    r_correct_raw_m: float,
    ph_correct_raw_rad: float,
) -> tuple[float, np.ndarray]:
    """Apply the one allowed source-AF conversion exactly once."""

    r0 = float(np.float64(r0_raw_m) + np.float64(r_correct_raw_m))
    response = np.asarray(response_raw, dtype=np.complex128) * np.exp(
        1j * np.float64(ph_correct_raw_rad)
    )
    _finite(response.real, "source-AF response.real")
    _finite(response.imag, "source-AF response.imag")
    return r0, _readonly(response, np.complex128)


@dataclass(frozen=True)
class NativeIdentity:
    """Small canonical identity used by the in-memory control records."""

    pass_id: int
    polarization: str
    sector_id: int
    pulse_index: int

    def __post_init__(self) -> None:
        if int(self.pass_id) not in {1, 7}:
            raise ValueError("the local control identity contract is limited to pass 1 or pass 7")
        if str(self.polarization).lower() != "hh":
            raise ValueError("the Camry control contract is limited to HH")

    def as_dict(self) -> dict[str, Any]:
        return {
            "pass": int(self.pass_id),
            "polarization": str(self.polarization).lower(),
            "sector": int(self.sector_id),
            "pulse": int(self.pulse_index),
        }


@dataclass(frozen=True)
class InMemorySourceAFRecord:
    """A source-AF-shaped fake record for local tests and control runs only."""

    identity: Any
    role: str
    position_xyz_m: np.ndarray
    frequencies_hz: np.ndarray
    r0_raw_m: float
    response_raw: np.ndarray
    r_correct_raw_m: float
    ph_correct_raw_rad: float
    effective_r0_m: float
    effective_response: np.ndarray
    representation_tag: str = SOURCE_AF_REPRESENTATION
    scope_name: str = "local_fake_source_af_control"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        position = _readonly(self.position_xyz_m, np.float64)
        frequencies = _readonly(self.frequencies_hz, np.float64)
        raw = _readonly(self.response_raw, np.complex128)
        effective = _readonly(self.effective_response, np.complex128)
        if position.shape != (3,):
            raise ValueError("position_xyz_m must have shape [3]")
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError("frequencies_hz must be a nonempty vector")
        if raw.shape != frequencies.shape or effective.shape != frequencies.shape:
            raise ValueError("raw/effective responses must match frequencies_hz")
        if str(self.representation_tag) != SOURCE_AF_REPRESENTATION:
            raise ValueError("fake records must be explicitly tagged source_af")
        for value, label in (
            (position, "position_xyz_m"),
            (frequencies, "frequencies_hz"),
            (raw.real, "response_raw.real"),
            (raw.imag, "response_raw.imag"),
            (effective.real, "effective_response.real"),
            (effective.imag, "effective_response.imag"),
        ):
            _finite(value, label)
        if frequencies.size > 1 and not np.all(np.diff(frequencies) > 0.0):
            raise ValueError("frequencies_hz must be strictly increasing")
        expected_r0, expected_response = source_af_values(
            self.r0_raw_m, raw, self.r_correct_raw_m, self.ph_correct_raw_rad
        )
        if float(self.effective_r0_m) != expected_r0:
            raise ValueError("effective_r0_m does not use the declared float64 source-AF formula")
        if not np.array_equal(effective, expected_response):
            raise ValueError("effective_response does not use the declared source-AF formula")
        object.__setattr__(self, "position_xyz_m", position)
        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "response_raw", raw)
        object.__setattr__(self, "effective_response", effective)
        object.__setattr__(self, "role", str(self.role).lower())
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def r0_source_m(self) -> float:
        return float(self.effective_r0_m)

    @property
    def response_source(self) -> np.ndarray:
        return self.effective_response

    def with_effective_response(self, response: Any) -> "InMemorySourceAFRecord":
        effective = np.asarray(response, dtype=np.complex128)
        if effective.shape != self.frequencies_hz.shape:
            raise ValueError("replacement response shape does not match frequencies_hz")
        raw = effective * np.exp(-1j * np.float64(self.ph_correct_raw_rad))
        _, canonical_effective = source_af_values(
            self.r0_raw_m, raw, self.r_correct_raw_m, self.ph_correct_raw_rad
        )
        return InMemorySourceAFRecord(
            identity=self.identity,
            role=self.role,
            position_xyz_m=self.position_xyz_m,
            frequencies_hz=self.frequencies_hz,
            r0_raw_m=self.r0_raw_m,
            response_raw=raw,
            r_correct_raw_m=self.r_correct_raw_m,
            ph_correct_raw_rad=self.ph_correct_raw_rad,
            effective_r0_m=self.effective_r0_m,
            effective_response=canonical_effective,
            representation_tag=self.representation_tag,
            scope_name=self.scope_name,
            provenance=self.provenance,
        )


@dataclass(frozen=True)
class NativePlacement:
    """Validated local-to-native placement ``p_native = R @ p_local + t``."""

    rotation: np.ndarray
    translation_m: np.ndarray
    name: str = "local_fake_camry_placement"

    def __post_init__(self) -> None:
        rotation = _readonly(self.rotation, np.float64)
        translation = _readonly(self.translation_m, np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("placement requires R[3,3] and t[3]")
        _finite(rotation, "placement rotation")
        _finite(translation, "placement translation")
        if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=2e-12):
            raise ValueError("placement rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=2e-12):
            raise ValueError("placement rotation must be proper")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation_m", translation)

    def apply(self, points_local_m: Any) -> np.ndarray:
        points = np.asarray(points_local_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_local_m must have shape [point,3]")
        return _readonly(points @ self.rotation.T + self.translation_m[None, :], np.float64)

    def inverse_directions(self, unit_native: Any) -> np.ndarray:
        values = np.asarray(unit_native, dtype=np.float64)
        return values @ self.rotation


CAMRY_PLACEMENT = NativePlacement(
    rotation=np.asarray(
        [
            [-0.05442654550121675, 0.9985177770800097, 0.0],
            [-0.9985177770800097, -0.05442654550121675, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    translation_m=np.asarray([20.66, -18.71, 0.02], dtype=np.float64),
    name="camry_point_c_declared_working_native_frame_convention",
)


TOPHAT_PROVISIONAL_PLACEMENT = NativePlacement(
    rotation=np.eye(3, dtype=np.float64),
    translation_m=np.asarray([-17.0244, 20.8791, 1.0], dtype=np.float64),
    name="tophat_screen_consistent_provisional_operational_roi",
)


def placement_contract(
    placement: NativePlacement,
    *,
    target_id: str,
) -> dict[str, Any]:
    """Return a self-contained placement disclosure for reports/protocols."""

    target_id = str(target_id)
    if target_id == "tophat":
        status = "screen_consistent_provisional_operational_roi"
    elif target_id == "toyota_camry":
        status = "declared_working_native_frame_convention"
    else:
        raise ValueError(f"unsupported placement target_id: {target_id}")
    return {
        "target_id": target_id,
        "equation": "p_native = R @ p_local + t",
        "R": placement.rotation.tolist(),
        "t_m": placement.translation_m.tolist(),
        "name": placement.name,
        "status": status,
        "claim_scope": (
            "operational placement convention only; no target recovery, survey registration, "
            "autofocus validation, physical geometry, or height claim"
        ),
        "no_recovery_claim": True,
        "no_survey_registration_claim": True,
        "no_physical_geometry_claim": True,
    }


@dataclass(frozen=True)
class SupportSpec:
    target_id: str
    lower_edge_m: tuple[float, float, float]
    upper_edge_exclusive_m: tuple[float, float, float]
    cell_size_m: float
    shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        if self.target_id not in {"tophat", "toyota_camry"}:
            raise ValueError(f"unsupported target_id: {self.target_id}")
        lower = np.asarray(self.lower_edge_m, dtype=np.float64)
        upper = np.asarray(self.upper_edge_exclusive_m, dtype=np.float64)
        if lower.shape != (3,) or upper.shape != (3,):
            raise ValueError("support bounds must be length-3")
        if not np.all(upper > lower) or not np.isfinite(np.r_[lower, upper]).all():
            raise ValueError("support bounds must be finite and increasing")
        if float(self.cell_size_m) <= 0.0 or not np.isfinite(float(self.cell_size_m)):
            raise ValueError("cell_size_m must be positive and finite")
        if len(self.shape) != 3 or any(int(n) <= 0 for n in self.shape):
            raise ValueError("support shape must be positive length three")
        expected = lower + float(self.cell_size_m) * np.asarray(self.shape, dtype=np.float64)
        if not np.allclose(expected, upper, rtol=0.0, atol=1e-12):
            raise ValueError("support upper edge must equal lower + cell_size*shape")

    @property
    def point_count(self) -> int:
        return int(np.prod(np.asarray(self.shape, dtype=np.int64)))

    @property
    def delta_volume_m3(self) -> float:
        return float(self.cell_size_m) ** 3

    def local_midpoints(self) -> np.ndarray:
        lower = np.asarray(self.lower_edge_m, dtype=np.float64)
        axes = [lower[i] + (np.arange(int(self.shape[i]), dtype=np.float64) + 0.5) * float(self.cell_size_m) for i in range(3)]
        mesh = np.meshgrid(*axes, indexing="ij")
        return _readonly(np.stack(mesh, axis=-1).reshape(-1, 3), np.float64)


SUPPORTS = {
    "tophat": SupportSpec("tophat", (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0), 0.5, (8, 8, 8)),
    "toyota_camry": SupportSpec("toyota_camry", (-5.0, -5.0, -5.0), (5.0, 5.0, 5.0), 1.0, (10, 10, 10)),
}


@dataclass(frozen=True)
class ExactReadoutGrid:
    target_id: str
    lower_edge_m: tuple[float, float, float]
    upper_edge_exclusive_m: tuple[float, float, float]
    spacing_m: float
    shape: tuple[int, int, int]

    @property
    def point_count(self) -> int:
        return int(np.prod(np.asarray(self.shape, dtype=np.int64)))

    def local_points(self) -> np.ndarray:
        lower = np.asarray(self.lower_edge_m, dtype=np.float64)
        axes = [lower[i] + np.arange(int(self.shape[i]), dtype=np.float64) * float(self.spacing_m) for i in range(3)]
        mesh = np.meshgrid(*axes, indexing="ij")
        return _readonly(np.stack(mesh, axis=-1).reshape(-1, 3), np.float64)


READOUT_GRIDS = {
    "tophat": ExactReadoutGrid("tophat", (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0), 0.1, (40, 40, 40)),
    "toyota_camry": ExactReadoutGrid("toyota_camry", (-5.0, -5.0, -5.0), (5.0, 5.0, 5.0), 0.1, (100, 100, 100)),
}


def support_for_target(target_id: str) -> SupportSpec:
    try:
        return SUPPORTS[str(target_id)]
    except KeyError as exc:
        raise ValueError(f"unsupported target_id: {target_id}") from exc


def exact_readout_grid(target_id: str) -> ExactReadoutGrid:
    try:
        return READOUT_GRIDS[str(target_id)]
    except KeyError as exc:
        raise ValueError(f"unsupported target_id: {target_id}") from exc


def degree1_real_sh_basis(unit_local: Any) -> np.ndarray:
    """Return real degree-1 basis [Y00, Y1x, Y1y, Y1z]."""

    directions = np.asarray(unit_local, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("unit_local must have shape [point,3]")
    _finite(directions, "unit_local")
    scale0 = 1.0 / np.sqrt(4.0 * np.pi)
    scale1 = np.sqrt(3.0 / (4.0 * np.pi))
    return np.column_stack(
        [
            np.full(directions.shape[0], scale0, dtype=np.float64),
            scale1 * directions[:, 0],
            scale1 * directions[:, 1],
            scale1 * directions[:, 2],
        ]
    )


def l0_scalar_coefficients(weights: Any, delta_volume_m3: float) -> np.ndarray:
    """Convert scalar voxel weights to c00 so Y00*c00*ΔV equals the weight."""

    if float(delta_volume_m3) <= 0.0:
        raise ValueError("delta_volume_m3 must be positive")
    return np.sqrt(4.0 * np.pi) * np.asarray(weights, dtype=np.float64) / float(delta_volume_m3)


@dataclass(frozen=True)
class RaggedComplexValues:
    ids: tuple[Any, ...]
    values: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if len(self.ids) != len(self.values):
            raise ValueError("ragged IDs and values must have equal lengths")
        keys = tuple(_id_key(identity) for identity in self.ids)
        if len(set(keys)) != len(keys):
            raise ValueError("ragged IDs must be unique")
        frozen = []
        for value in self.values:
            array = np.asarray(value, dtype=np.complex128)
            if array.ndim != 1:
                raise ValueError("ragged values must be one-dimensional")
            _finite(array.real, "ragged values.real")
            _finite(array.imag, "ragged values.imag")
            frozen.append(_readonly(array, np.complex128))
        object.__setattr__(self, "values", tuple(frozen))


def _record_response(record: Any) -> np.ndarray:
    if str(getattr(record, "role", "")).lower() == "test":
        raise ValueError("TEST observations are sealed out of native-complex ROI forward/render paths")
    if getattr(record, "representation_tag", None) != SOURCE_AF_REPRESENTATION:
        raise TypeError("directional-SH ROI fitting requires explicit source_af records")
    value = np.asarray(getattr(record, "effective_response", getattr(record, "response_source", None)), dtype=np.complex128)
    frequencies = np.asarray(getattr(record, "frequencies_hz"), dtype=np.float64)
    if value.shape != frequencies.shape:
        raise ValueError("source-AF response does not match frequencies_hz")
    _finite(value.real, "source-AF response.real")
    _finite(value.imag, "source-AF response.imag")
    return value


def _record_source_r0(record: Any) -> float:
    value = getattr(record, "r0_source_m", None)
    if value is None:
        value = getattr(record, "effective_r0_m", None)
    if value is None:
        raise ValueError("source-AF record must expose effective source r0")
    return float(np.float64(value))


def validate_record_set(
    records: Sequence[Any],
    *,
    split: str,
    expected_count: int | None = None,
    target_id: str = "toyota_camry",
) -> tuple[Any, ...]:
    """Seal the Camry train/held-out validation selectors before payload use."""

    values = tuple(records)
    if not values:
        raise ValueError(f"{split} record set must be nonempty")
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    target_id = str(target_id)
    if target_id not in {"toyota_camry", "tophat"}:
        raise ValueError(f"unsupported target_id: {target_id}")
    if expected_count is not None and len(values) != int(expected_count):
        raise ValueError(f"{split} record set requires exactly {expected_count} records; got {len(values)}")
    keys = []
    for record in values:
        identity = getattr(record, "identity", None)
        key = _id_key(identity)
        if target_id == "toyota_camry":
            required_sector = TRAIN_SECTOR_ID if split == "train" else VALIDATION_SECTOR_ID
            selector_ok = key[0] == 1 and key[1] == "hh" and key[2] == required_sector
            selector_text = f"P1/HH/sector {required_sector:03d}"
        else:
            required_sectors = TOPHAT_TRAIN_SECTOR_IDS if split == "train" else TOPHAT_VALIDATION_SECTOR_IDS
            selector_ok = (
                key[0] in TOPHAT_TRAIN_PASS_IDS
                and key[1] == "hh"
                and key[2] in required_sectors
            )
            selector_text = (
                "P1/P7 HH TRAIN sectors 002/092/182/272"
                if split == "train"
                else "P1/P7 HH validation sectors 001/091/181/271"
            )
        if not selector_ok:
            raise ValueError(f"{split} selector requires {selector_text}; got {key}")
        role = str(getattr(record, "role", "")).lower()
        if role != split:
            raise ValueError(f"{split} selector requires role={split}")
        if role == "test":
            raise ValueError("TEST observations are sealed out of the ROI fit")
        _record_response(record)
        position = np.asarray(getattr(record, "position_xyz_m"), dtype=np.float64)
        frequencies = np.asarray(getattr(record, "frequencies_hz"), dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("record position must be finite [3]")
        if frequencies.ndim != 1 or frequencies.size == 0 or not np.isfinite(frequencies).all():
            raise ValueError("record frequencies must be finite and nonempty")
        if frequencies.size > 1 and not np.all(np.diff(frequencies) > 0.0):
            raise ValueError("record frequencies must be strictly increasing")
        if getattr(record, "r0_source_m", getattr(record, "effective_r0_m", None)) is None:
            raise ValueError("source-AF record must expose effective source r0")
        keys.append(key)
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate {split} observation IDs are not allowed")
    if tuple(keys) != tuple(sorted(keys)):
        raise ValueError(f"{split} observation IDs must be canonically sorted")
    return values


class NativeComplexDirectionalSHROIModel:
    """All-active fixed-support real degree-1 directional-SH complex field."""

    def __init__(
        self,
        target_id: str = "toyota_camry",
        *,
        placement: NativePlacement | None = None,
        spatial_chunk_size: int = DEFAULT_SPATIAL_CHUNK_SIZE,
        frequency_chunk_size: int = DEFAULT_FREQUENCY_CHUNK_SIZE,
    ) -> None:
        self.target_id = str(target_id)
        self.support = support_for_target(self.target_id)
        if self.target_id == "tophat" and placement is None:
            raise ValueError(
                "TopHat ROI fitting is fail-closed until the measured screen supplies a non-inconclusive placement"
            )
        self.placement = CAMRY_PLACEMENT if placement is None else placement
        self.spatial_chunk_size = int(spatial_chunk_size)
        self.frequency_chunk_size = int(frequency_chunk_size)
        if self.spatial_chunk_size <= 0 or self.frequency_chunk_size <= 0:
            raise ValueError("chunk sizes must be positive")
        self.local_points_m = self.support.local_midpoints()
        self.native_points_m = self.placement.apply(self.local_points_m)
        self.global_complex_gain = GLOBAL_COMPLEX_GAIN

    @property
    def point_count(self) -> int:
        return self.support.point_count

    @property
    def coefficient_shape(self) -> tuple[int, int]:
        return (self.point_count, 4)

    def initial_coefficients(self) -> np.ndarray:
        return np.zeros(self.coefficient_shape, dtype=np.complex128)

    def _validate_coefficients(self, coefficients: Any) -> np.ndarray:
        values = np.asarray(coefficients, dtype=np.complex128)
        if values.shape != self.coefficient_shape:
            raise ValueError(f"coefficients must have shape {self.coefficient_shape}")
        _finite(values.real, "coefficients.real")
        _finite(values.imag, "coefficients.imag")
        return values

    def direction_geometry(self, antenna_native_m: Any, points_native_m: Any | None = None) -> dict[str, np.ndarray]:
        antenna = np.asarray(antenna_native_m, dtype=np.float64)
        points = self.native_points_m if points_native_m is None else np.asarray(points_native_m, dtype=np.float64)
        if antenna.shape != (3,) or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("antenna must be [3] and points must be [point,3]")
        delta = antenna[None, :] - points
        distance = np.linalg.norm(delta, axis=1)
        if np.any(distance <= 0.0) or not np.isfinite(distance).all():
            raise ValueError("antenna cannot coincide with a support point")
        unit_native = delta / distance[:, None]
        unit_local = self.placement.inverse_directions(unit_native)
        unit_local /= np.linalg.norm(unit_local, axis=1, keepdims=True)
        theta = np.arccos(np.clip(unit_local[:, 2], -1.0, 1.0))
        phi = np.arctan2(unit_local[:, 1], unit_local[:, 0])
        return {
            "distance_m": distance,
            "unit_native": unit_native,
            "unit_local": unit_local,
            "theta_rad": theta,
            "phi_rad": phi,
        }

    def _record_prediction(self, record: Any, coefficients: np.ndarray) -> np.ndarray:
        frequencies = np.asarray(record.frequencies_hz, dtype=np.float64)
        antenna = np.asarray(record.position_xyz_m, dtype=np.float64)
        result = np.zeros(frequencies.shape, dtype=np.complex128)
        for start in range(0, self.point_count, self.spatial_chunk_size):
            stop = min(start + self.spatial_chunk_size, self.point_count)
            points = self.native_points_m[start:stop]
            geometry = self.direction_geometry(antenna, points)
            basis = degree1_real_sh_basis(geometry["unit_local"])
            field = np.sum(basis * coefficients[start:stop, :], axis=1)
            distance = geometry["distance_m"]
            for fstart in range(0, frequencies.size, self.frequency_chunk_size):
                fstop = min(fstart + self.frequency_chunk_size, frequencies.size)
                phase = -(4.0 * np.pi / SPEED_OF_LIGHT_M_S) * (
                    distance[:, None] - _record_source_r0(record)
                ) * frequencies[None, fstart:fstop]
                kernel = np.exp(1j * phase)
                result[fstart:fstop] += (
                    np.sum(field[:, None] * kernel, axis=0) * self.support.delta_volume_m3
                )
        return result

    def forward(
        self,
        records: Sequence[Any],
        coefficients: Any | None = None,
        *,
        gain: Any | None = None,
        record_gains: Any | None = None,
    ) -> RaggedComplexValues:
        if gain is not None or record_gains is not None:
            raise ValueError("directional-SH ROI fitting fixes one global gain at 1+0j; per-record gains are forbidden")
        values = tuple(records)
        if not values:
            raise ValueError("forward requires at least one source-AF record")
        if any(str(getattr(record, "role", "")).lower() == "test" for record in values):
            raise ValueError("TEST observations are sealed out of native-complex ROI forward/render paths")
        coeff = self.initial_coefficients() if coefficients is None else self._validate_coefficients(coefficients)
        predictions = tuple(self._record_prediction(record, coeff) for record in values)
        return RaggedComplexValues(tuple(record.identity for record in values), predictions)

    def loss_and_gradient(
        self,
        records: Sequence[Any],
        coefficients: Any,
        *,
        targets: RaggedComplexValues | None = None,
    ) -> tuple[float, np.ndarray, RaggedComplexValues]:
        values = tuple(records)
        coeff = self._validate_coefficients(coefficients)
        prediction = self.forward(values, coeff)
        if targets is None:
            target = RaggedComplexValues(
                tuple(record.identity for record in values),
                tuple(_record_response(record) for record in values),
            )
        else:
            target = targets
            if tuple(target.ids) != tuple(prediction.ids):
                raise ValueError("target identities do not match prediction identities")
        residuals = tuple(p - y for p, y in zip(prediction.values, target.values))
        total_frequencies = sum(int(value.size) for value in residuals)
        if total_frequencies <= 0:
            raise ValueError("loss requires at least one frequency sample")
        loss = 0.5 * sum(float(np.vdot(value, value).real) for value in residuals) / float(total_frequencies)
        gradient = self.gradient_from_native_residuals(
            values,
            RaggedComplexValues(prediction.ids, residuals),
            normalization=float(total_frequencies),
        )
        return float(loss), gradient, prediction

    def gradient_from_native_residuals(
        self,
        records: Sequence[Any],
        residuals: RaggedComplexValues,
        *,
        normalization: float | None = None,
    ) -> np.ndarray:
        """Apply the native renderer adjoint to supplied residuals.

        The residuals are native-frequency values.  Keeping this operation
        separate lets the range projector form ``B^H B(Ac-y)`` without ever
        materializing a dense native or range normal matrix.
        """

        values = tuple(records)
        if tuple(residuals.ids) != tuple(record.identity for record in values):
            raise ValueError("native residual identities do not match records")
        residual_values = tuple(np.asarray(value, dtype=np.complex128) for value in residuals.values)
        total_frequencies = sum(int(value.size) for value in residual_values)
        if total_frequencies <= 0:
            raise ValueError("gradient requires at least one frequency sample")
        denominator = float(total_frequencies if normalization is None else normalization)
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("gradient normalization must be positive and finite")
        gradient = np.zeros(self.coefficient_shape, dtype=np.complex128)
        # This is a second streamed pass and therefore cannot form a hidden
        # dense H cache.
        for record, residual in zip(values, residual_values):
            frequencies = np.asarray(record.frequencies_hz, dtype=np.float64)
            antenna = np.asarray(record.position_xyz_m, dtype=np.float64)
            source_r0 = _record_source_r0(record)
            if residual.ndim != 1 or residual.shape != frequencies.shape:
                raise ValueError("native residual shape does not match record frequencies")
            _finite(residual.real, "native residual.real")
            _finite(residual.imag, "native residual.imag")
            for start in range(0, self.point_count, self.spatial_chunk_size):
                stop = min(start + self.spatial_chunk_size, self.point_count)
                geometry = self.direction_geometry(antenna, self.native_points_m[start:stop])
                basis = degree1_real_sh_basis(geometry["unit_local"])
                chunk_gradient = np.zeros((stop - start, 4), dtype=np.complex128)
                for fstart in range(0, frequencies.size, self.frequency_chunk_size):
                    fstop = min(fstart + self.frequency_chunk_size, frequencies.size)
                    phase = -(4.0 * np.pi / SPEED_OF_LIGHT_M_S) * (
                        geometry["distance_m"][:, None] - source_r0
                    ) * frequencies[None, fstart:fstop]
                    kernel = np.exp(1j * phase)
                    weighted_residual = np.conjugate(kernel) * residual[None, fstart:fstop]
                    chunk_gradient += np.sum(
                        np.conjugate(basis)[:, :, None] * weighted_residual[:, None, :], axis=2
                    ) * self.support.delta_volume_m3
                gradient[start:stop, :] += chunk_gradient
        gradient /= denominator
        return gradient

    def adjoint_seed(self, records: Sequence[Any]) -> np.ndarray:
        """Deterministic zero-state adjoint seed, with no validation peeking."""

        zero = self.initial_coefficients()
        _, gradient, _ = self.loss_and_gradient(records, zero)
        return _readonly(-gradient, np.complex128)

    def readout(self, coefficients: Any) -> dict[str, Any]:
        """Hold each fitted coarse cell over the exact .1m reporting grid."""

        coeff = self._validate_coefficients(coefficients)
        support = self.support
        grid = exact_readout_grid(self.target_id)
        points = grid.local_points()
        lower = np.asarray(support.lower_edge_m, dtype=np.float64)
        index = np.floor((points - lower[None, :]) / float(support.cell_size_m)).astype(np.int64)
        index = np.clip(index, 0, np.asarray(support.shape, dtype=np.int64)[None, :] - 1)
        flat = np.ravel_multi_index(tuple(index[:, axis] for axis in range(3)), support.shape)
        held = coeff[flat]
        energy = np.sum(np.abs(held) ** 2, axis=1).reshape(grid.shape)
        l0 = held[:, 0].reshape(grid.shape)
        projections = np.stack(
            [held[:, 0].real, held[:, 1].real, held[:, 2].real, held[:, 3].real], axis=1
        ).reshape((*grid.shape, 4))
        z_mid_index = int(round((0.0 - float(grid.lower_edge_m[2])) / float(grid.spacing_m)))
        if not 0 <= z_mid_index < int(grid.shape[2]):
            raise AssertionError("exact readout grid does not contain the declared local z=0 slice")
        return {
            "schema": f"{SCHEMA}.readout",
            "target_id": self.target_id,
            "label": READOUT_LABEL,
            "frame": "local_target_frame",
            "endpoint_inclusion": False,
            "half_voxel_shift": False,
            "lower_edge_m": list(grid.lower_edge_m),
            "upper_edge_exclusive_m": list(grid.upper_edge_exclusive_m),
            "spacing_m": float(grid.spacing_m),
            "shape": list(grid.shape),
            "point_count": grid.point_count,
            "energy": energy,
            "l0_real": l0.real,
            "l0_imag": l0.imag,
            "real_coefficient_projections": projections,
            "mid_z_index_for_local_zero": z_mid_index,
        }

    def diagnostics(self, coefficients: Any) -> dict[str, Any]:
        coeff = self._validate_coefficients(coefficients)
        energy_by_voxel = np.sum(np.abs(coeff) ** 2, axis=1)
        energy_by_coefficient = np.sum(np.abs(coeff) ** 2, axis=0)
        readout = self.readout(coeff)
        mid_z_index = int(readout["mid_z_index_for_local_zero"])
        return {
            "schema": f"{SCHEMA}.diagnostics",
            "target_id": self.target_id,
            "all_voxels_active": True,
            "prune_grow_adaptive_support": False,
            "energy_by_voxel": energy_by_voxel,
            "energy_by_coefficient": energy_by_coefficient,
            "coefficient_l0_projection": coeff[:, 0],
            "readout_label": READOUT_LABEL,
            "readout_energy_mid_z_index": mid_z_index,
            "readout_energy_mid_z_slice": readout["energy"][:, :, mid_z_index],
        }


@dataclass(frozen=True)
class FitConfig:
    updates: int = DEFAULT_UPDATES
    microbatch_size: int = 2
    # Fixed conservative control-step scale for the 1m^3 Camry support.  It is
    # deliberately a declared schedule constant, not a line search or a
    # validation-derived adaptive rule.
    step_size: float = 1.0e-3
    checkpoint_steps: tuple[int, ...] = DEFAULT_CHECKPOINT_STEPS
    validation_steps: tuple[int, ...] = DEFAULT_CHECKPOINT_STEPS
    control_kind: str = "fake_cyclic_control"
    initialization: str = "adjoint_seed"
    solver: str = "gd"

    def __post_init__(self) -> None:
        solver = str(self.solver).lower()
        if solver not in {"gd", "cgls"}:
            raise ValueError("solver must be gd or cgls")
        if solver == "cgls" and str(self.control_kind) != "actual_full_aperture":
            raise ValueError("cgls is available only for actual_full_aperture fitting")
        if solver == "cgls" and int(self.updates) > MAX_CGLS_ITERATIONS:
            raise ValueError(f"cgls updates cannot exceed {MAX_CGLS_ITERATIONS}")
        if str(self.control_kind) == "fake_cyclic_control" and int(self.updates) < DEFAULT_UPDATES:
            raise ValueError("non-smoke directional-SH control requires at least 12 updates")
        if int(self.updates) <= 0:
            raise ValueError("updates must be positive")
        if str(self.control_kind) == "actual_full_aperture" and str(self.initialization) != "zero":
            raise ValueError("actual full-aperture control requires zero initialization")
        if str(self.control_kind) == "fake_cyclic_control" and str(self.initialization) != "adjoint_seed":
            raise ValueError("fake cyclic control requires the declared adjoint seed")
        if int(self.microbatch_size) <= 0:
            raise ValueError("microbatch_size must be positive")
        if not np.isfinite(float(self.step_size)) or float(self.step_size) <= 0.0:
            raise ValueError("step_size must be positive and finite")
        if tuple(sorted(set(int(v) for v in self.checkpoint_steps))) != tuple(self.checkpoint_steps):
            raise ValueError("checkpoint_steps must be sorted and unique")
        if int(self.checkpoint_steps[0]) != 0 or int(self.checkpoint_steps[-1]) != int(self.updates):
            raise ValueError("checkpoint_steps must include fixed step 0 and final update")
        if any(v < 0 or v > int(self.updates) for v in self.validation_steps):
            raise ValueError("validation_steps must lie within the fixed update schedule")


def conservative_full_aperture_step_size(model: NativeComplexDirectionalSHROIModel) -> float:
    """Return the declared target-specific eta=0.5/L control scale."""

    support_count = float(model.point_count)
    delta_volume = float(model.support.delta_volume_m3)
    return float(0.5 * np.pi / (support_count * delta_volume * delta_volume))


def actual_fit_config(
    model: NativeComplexDirectionalSHROIModel,
    *,
    updates: int | None = None,
    microbatch_size: int = 2,
    solver: str = "gd",
) -> FitConfig:
    """Build an explicit bounded full-aperture config; updates are not scientific constants."""

    solver = str(solver).lower()
    if updates is None:
        updates = MAX_CGLS_ITERATIONS if solver == "cgls" else DEFAULT_UPDATES
    updates = int(updates)
    if updates <= 0:
        raise ValueError("actual updates must be positive")
    if solver == "cgls":
        if updates > MAX_CGLS_ITERATIONS:
            raise ValueError(f"cgls updates cannot exceed {MAX_CGLS_ITERATIONS}")
        checkpoint_steps = tuple(step for step in CGLS_DIAGNOSTIC_STEPS if step <= updates)
        if checkpoint_steps[-1] != updates:
            checkpoint_steps = (*checkpoint_steps, updates)
    else:
        checkpoint_steps = tuple(sorted({0, updates // 3, (2 * updates) // 3, updates}))
    return FitConfig(
        updates=updates,
        microbatch_size=int(microbatch_size),
        step_size=conservative_full_aperture_step_size(model),
        checkpoint_steps=checkpoint_steps,
        validation_steps=checkpoint_steps,
        control_kind="actual_full_aperture",
        initialization="zero",
        solver=solver,
    )


@dataclass(frozen=True)
class FitResult:
    target_id: str
    coefficients_latest: np.ndarray
    coefficients_selected: np.ndarray
    coefficients_final: np.ndarray
    history: tuple[Mapping[str, Any], ...]
    checkpoint_steps: tuple[int, ...]
    validation_steps: tuple[int, ...]
    checkpoint_paths: Mapping[str, str] = field(default_factory=dict)
    solver: str = "gd"
    termination_reason: str = "MAX_ITERATIONS"
    max_iterations: int | None = None
    executed_iterations: int | None = None
    selected_final_step: int | None = None
    diagnostic_steps: tuple[int, ...] = ()


def _ragged_sidecar_payload(values: RaggedComplexValues) -> dict[str, np.ndarray]:
    counts = np.asarray([value.size for value in values.values], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    concatenated = np.concatenate(values.values) if offsets[-1] else np.empty(0, dtype=np.complex128)
    ids = np.asarray([json.dumps(_id_key(identity), separators=(",", ":")) for identity in values.ids], dtype=str)
    return {"values": concatenated, "offsets": offsets, "identity_keys": ids}


def _save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def _checkpoint_metadata(
    *,
    model: NativeComplexDirectionalSHROIModel,
    checkpoint_name: str,
    step: int,
    history: Sequence[Mapping[str, Any]],
    prediction_ids: Sequence[Any],
    train_records: Sequence[Any] | None,
    validation_records: Sequence[Any] | None,
    fit_config: FitConfig | None,
    artifacts_generated_after_checkpoint_reload: bool,
    gradient_record_scope: str,
    run_provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    train_ids = tuple(record.identity for record in train_records) if train_records is not None else tuple(prediction_ids)
    validation_ids = tuple(record.identity for record in validation_records) if validation_records is not None else tuple()
    schedule = FitConfig() if fit_config is None else fit_config
    provenance = dict(DEFAULT_FAKE_PROVENANCE if run_provenance is None else run_provenance)
    metadata = {
        "schema": CHECKPOINT_SCHEMA,
        "target_id": model.target_id,
        "checkpoint_name": checkpoint_name,
        "step": int(step),
        "support": {
            "shape": list(model.support.shape),
            "cell_size_m": model.support.cell_size_m,
            "point_count": model.point_count,
        },
        "all_voxels_active": True,
        "global_complex_gain": [1.0, 0.0],
        "source_af": {
            "representation_tag": SOURCE_AF_REPRESENTATION,
            "formula": SOURCE_AF_FORMULA,
            "conversion_count": "exactly_once_after_metadata_header_gate",
            "archive_opened": bool(provenance.get("archive_opened", False)),
            "test_payload_opened": bool(provenance.get("test_payload_opened", False)),
        },
        "run_provenance": provenance,
        "canonical_selected_ids": {
            "train": [list(_id_key(identity)) for identity in train_ids],
            "validation": [list(_id_key(identity)) for identity in validation_ids],
            "identity_order": "(pass,polarization,sector,pulse)",
        },
        "placement": {
            "equation": "p_native = R @ p_local + t",
            "R": model.placement.rotation.tolist(),
            "t_m": model.placement.translation_m.tolist(),
            "name": model.placement.name,
        },
        "sh_basis": {
            "degree": 1,
            "basis_order": ["Y00", "Y1x", "Y1y", "Y1z"],
            "convention": SH_BASIS_CONVENTION,
            "coefficient_dtype": "complex128 (real/imag field)",
        },
        "execution_contract": {
            "native_geometry_dtype": "float64",
            "measurement_dtype": "complex128",
            "spatial_chunk_size": model.spatial_chunk_size,
            "frequency_chunk_size": model.frequency_chunk_size,
            "coherent_spatial_sum_before_residual": True,
            "adaptive_support": False,
            "per_record_gain": False,
        },
        "schedule": {
            "updates": int(schedule.updates),
            "microbatch_size": int(schedule.microbatch_size),
            "record_microbatch_size_used": gradient_record_scope != "all_selected_train_records",
            "record_microbatching": (
                "cyclic fake control batches"
                if gradient_record_scope != "all_selected_train_records"
                else "ignored in actual full-aperture mode; all selected TRAIN records are used per update"
            ),
            "array_memory_chunks": {
                "spatial_chunk_size": model.spatial_chunk_size,
                "frequency_chunk_size": model.frequency_chunk_size,
                "used_for_actual_array_memory": gradient_record_scope == "all_selected_train_records",
            },
            "step_size": float(schedule.step_size),
            "checkpoint_steps": [int(value) for value in schedule.checkpoint_steps],
            "validation_steps": [int(value) for value in schedule.validation_steps],
            "validation_use": "diagnostic_only_fixed_points",
            "selected_model": f"update_{int(schedule.updates)}_final_for_train_and_validation",
            "persistence_points_only": [int(value) for value in schedule.checkpoint_steps],
            "atomic_per_update": False,
            "resumable_per_update": False,
            "gradient_record_scope": gradient_record_scope,
            "record_chunks_are_memory_scheduling_only": gradient_record_scope == "all_selected_train_records",
            "control_kind": str(schedule.control_kind),
            "initialization": str(schedule.initialization),
        },
        "history": [dict(entry) for entry in history],
        "readout_label": READOUT_LABEL,
        "artifacts_generated_after_checkpoint_reload": bool(artifacts_generated_after_checkpoint_reload),
    }
    if str(schedule.solver).lower() == "cgls":
        metadata["schedule"].update(
            {
                "solver": "cgls",
                "max_iterations": int(provenance.get("max_iterations", schedule.updates)),
                "executed_iterations": int(provenance.get("executed_iterations", step)),
                "selected_final_step": int(provenance.get("selected_final_step", step)),
                "termination_reason": str(provenance.get("termination_reason", "not_yet_terminated")),
                "selected_model": (
                    f"cgls_terminal_step_{int(provenance['selected_final_step'])}_final_for_train_and_validation"
                    if "selected_final_step" in provenance
                    else f"cgls_diagnostic_step_{int(step)}"
                ),
                "persistence_points_only": [
                    int(value)
                    for value in provenance.get("diagnostic_steps", schedule.checkpoint_steps)
                ],
                "cgls_passes_per_iteration": 3,
            }
        )
    return metadata


def save_checkpoint_state(
    output_dir: str | Path,
    *,
    model: NativeComplexDirectionalSHROIModel,
    checkpoint_name: str,
    step: int,
    coefficients: Any,
    history: Sequence[Mapping[str, Any]],
    prediction_ids: Sequence[Any],
    train_records: Sequence[Any] | None = None,
    validation_records: Sequence[Any] | None = None,
    fit_config: FitConfig | None = None,
    artifacts_generated_after_checkpoint_reload: bool = False,
    gradient_record_scope: str = "cyclic_fake_control_only",
    run_provenance: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Persist one full state/contract file at an explicitly fixed point."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    coeff = model._validate_coefficients(coefficients)
    checkpoint_path = root / f"checkpoint_{checkpoint_name}.npz"
    _save_npz(checkpoint_path, coefficients=coeff, step=np.asarray([int(step)], dtype=np.int64))
    metadata_path = root / f"checkpoint_{checkpoint_name}.json"
    metadata = _checkpoint_metadata(
        model=model,
        checkpoint_name=checkpoint_name,
        step=step,
        history=history,
        prediction_ids=prediction_ids,
        train_records=train_records,
        validation_records=validation_records,
        fit_config=fit_config,
        artifacts_generated_after_checkpoint_reload=artifacts_generated_after_checkpoint_reload,
        gradient_record_scope=gradient_record_scope,
        run_provenance=run_provenance,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {"checkpoint": str(checkpoint_path), "metadata": str(metadata_path)}


def save_prediction_artifacts(
    output_dir: str | Path,
    *,
    model: NativeComplexDirectionalSHROIModel,
    artifact_name: str,
    source_checkpoint: str,
    step: int,
    coefficients: Any,
    history: Sequence[Mapping[str, Any]],
    predictions: RaggedComplexValues,
    residuals: RaggedComplexValues,
    train_records: Sequence[Any] | None = None,
    validation_records: Sequence[Any] | None = None,
    fit_config: FitConfig | None = None,
    artifacts_generated_after_checkpoint_reload: bool = False,
    gradient_record_scope: str = "cyclic_fake_control_only",
    run_provenance: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Persist sidecars/readout without creating a second selected checkpoint."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    coeff = model._validate_coefficients(coefficients)
    prediction_path = root / f"predictions_{artifact_name}.npz"
    residual_path = root / f"residuals_{artifact_name}.npz"
    _save_npz(prediction_path, **_ragged_sidecar_payload(predictions))
    _save_npz(residual_path, **_ragged_sidecar_payload(residuals))
    diagnostics = model.diagnostics(coeff)
    readout = model.readout(coeff)
    diagnostics_path = root / f"diagnostics_{artifact_name}.npz"
    readout_path = root / f"readout_{artifact_name}.npz"
    _save_npz(
        diagnostics_path,
        energy_by_voxel=diagnostics["energy_by_voxel"],
        energy_by_coefficient=diagnostics["energy_by_coefficient"],
        coefficient_l0_projection=diagnostics["coefficient_l0_projection"],
        readout_energy_mid_z_index=np.asarray([diagnostics["readout_energy_mid_z_index"]], dtype=np.int64),
        readout_energy_mid_z_slice=diagnostics["readout_energy_mid_z_slice"],
    )
    _save_npz(
        readout_path,
        energy=readout["energy"],
        l0_real=readout["l0_real"],
        l0_imag=readout["l0_imag"],
        real_coefficient_projections=readout["real_coefficient_projections"],
    )
    metadata_path = root / f"artifacts_{artifact_name}.json"
    metadata = _checkpoint_metadata(
        model=model,
        checkpoint_name=artifact_name,
        step=step,
        history=history,
        prediction_ids=predictions.ids,
        train_records=train_records,
        validation_records=validation_records,
        fit_config=fit_config,
        artifacts_generated_after_checkpoint_reload=bool(artifacts_generated_after_checkpoint_reload),
        gradient_record_scope=gradient_record_scope,
        run_provenance=run_provenance,
    )
    metadata["artifact_only"] = True
    metadata["source_checkpoint"] = str(source_checkpoint)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "predictions": str(prediction_path),
        "residuals": str(residual_path),
        "diagnostics": str(diagnostics_path),
        "readout": str(readout_path),
        "metadata": str(metadata_path),
    }


def save_checkpoint_bundle(
    output_dir: str | Path,
    *,
    model: NativeComplexDirectionalSHROIModel,
    checkpoint_name: str,
    step: int,
    coefficients: Any,
    history: Sequence[Mapping[str, Any]],
    predictions: RaggedComplexValues,
    residuals: RaggedComplexValues,
    train_records: Sequence[Any] | None = None,
    validation_records: Sequence[Any] | None = None,
    fit_config: FitConfig | None = None,
    artifacts_generated_after_checkpoint_reload: bool = False,
    gradient_record_scope: str = "cyclic_fake_control_only",
    run_provenance: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Persist one state plus its sidecars at a fixed checkpoint point."""

    state_paths = save_checkpoint_state(
        output_dir,
        model=model,
        checkpoint_name=checkpoint_name,
        step=step,
        coefficients=coefficients,
        history=history,
        prediction_ids=predictions.ids,
        train_records=train_records,
        validation_records=validation_records,
        fit_config=fit_config,
        artifacts_generated_after_checkpoint_reload=bool(artifacts_generated_after_checkpoint_reload),
        gradient_record_scope=gradient_record_scope,
        run_provenance=run_provenance,
    )
    artifact_paths = save_prediction_artifacts(
        output_dir,
        model=model,
        artifact_name=checkpoint_name,
        source_checkpoint=state_paths["checkpoint"],
        step=step,
        coefficients=coefficients,
        history=history,
        predictions=predictions,
        residuals=residuals,
        train_records=train_records,
        validation_records=validation_records,
        fit_config=fit_config,
        artifacts_generated_after_checkpoint_reload=bool(artifacts_generated_after_checkpoint_reload),
        gradient_record_scope=gradient_record_scope,
        run_provenance=run_provenance,
    )
    return {**state_paths, **artifact_paths}


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        return {
            "coefficients": np.asarray(payload["coefficients"], dtype=np.complex128),
            "step": int(np.asarray(payload["step"], dtype=np.int64).reshape(-1)[0]),
        }


def residual_metrics(
    model: NativeComplexDirectionalSHROIModel,
    records: Sequence[Any] | None,
    coefficients: Any,
    *,
    split: str,
) -> dict[str, Any] | None:
    """Report native-complex residual metrics without changing the selected state."""

    if records is None:
        return None
    values = tuple(records)
    if not values:
        return None
    prediction = model.forward(values, coefficients)
    targets = tuple(_record_response(record) for record in values)
    residuals = tuple(p - target for p, target in zip(prediction.values, targets))
    target_energy = sum(float(np.vdot(target, target).real) for target in targets)
    residual_energy = sum(float(np.vdot(residual, residual).real) for residual in residuals)
    return {
        "split": str(split),
        "record_count": len(values),
        "frequency_sample_count": int(sum(target.size for target in targets)),
        "native_complex_relmse": None if target_energy == 0.0 else float(residual_energy / target_energy),
        "native_complex_residual_energy": float(residual_energy),
        "native_complex_target_energy": float(target_energy),
        "max_residual_abs": float(max(np.max(np.abs(value)) for value in residuals)),
    }


@dataclass(frozen=True)
class RangeFrequencyGroup:
    """One exact-equality native frequency-vector group."""

    group_id: int
    frequencies_hz: np.ndarray
    centered_frequencies_hz: np.ndarray
    frequency_count: int
    first_hz: float
    last_hz: float
    min_df_hz: float
    median_df_hz: float
    max_df_hz: float
    endpoint_affine_deviation_hz: float
    r_rayleigh_m: float
    g_m: float
    first_minimum_sample_index: int
    first_minimum_bracket_m: tuple[float, float]
    first_minimum_h: float

    def __post_init__(self) -> None:
        frequencies = _readonly(self.frequencies_hz, np.float64)
        centered = _readonly(self.centered_frequencies_hz, np.float64)
        if frequencies.ndim != 1 or frequencies.size != int(self.frequency_count):
            raise ValueError("frequency group frequencies must match frequency_count")
        if centered.shape != frequencies.shape:
            raise ValueError("centered frequency group shape mismatch")
        _finite(frequencies, "frequency group frequencies")
        _finite(centered, "frequency group centered frequencies")
        if frequencies.size < 3 or not np.all(np.diff(frequencies) > 0.0):
            raise ValueError("frequency groups require at least three strictly increasing native frequencies")
        if not np.isfinite(float(self.g_m)) or float(self.g_m) <= 0.0:
            raise ValueError("frequency group g must be positive and finite")
        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "centered_frequencies_hz", centered)


@dataclass(frozen=True)
class RangeRecordGeometry:
    """Geometry-only range bounds and per-record projector coordinates."""

    identity_key: tuple[int, str, int, int]
    group_id: int
    r0_source_m: float
    antenna_native_m: np.ndarray
    c_native_m: np.ndarray
    r_c_m: float
    s_c_m: float
    frequency_count: int
    r_min_m: float
    r_max_m: float
    ambiguity_period_m: float
    principal_cell_lower_m: float
    principal_cell_upper_m: float
    principal_cell_guard_pass: bool
    same_delay_exterior_capture_nominal: Mapping[str, Any]
    q_minus_m: float
    q_plus_m: float
    q_last_m: float
    q_overshoot_m: float
    M: int
    rank: int
    condition_number: float
    sampled_in_cube_retention_min: float
    sampled_in_cube_retention_max: float
    sampled_in_cube_sample_count: int
    exterior_leakage_curve: Mapping[str, Any]

    def __post_init__(self) -> None:
        antenna = _readonly(self.antenna_native_m, np.float64)
        center = _readonly(self.c_native_m, np.float64)
        if antenna.shape != (3,) or center.shape != (3,):
            raise ValueError("record geometry vectors must have shape [3]")
        _finite(antenna, "record antenna")
        _finite(center, "record local-origin placement")
        for value, label in (
            (self.r0_source_m, "record source r0"),
            (self.r_c_m, "record Rc"),
            (self.s_c_m, "record s_c"),
            (self.r_min_m, "record Rmin"),
            (self.r_max_m, "record Rmax"),
            (self.ambiguity_period_m, "record nominal ambiguity period"),
            (self.principal_cell_lower_m, "record principal-cell lower bound"),
            (self.principal_cell_upper_m, "record principal-cell upper bound"),
            (self.q_minus_m, "record qminus"),
            (self.q_plus_m, "record qplus"),
            (self.q_last_m, "record qlast"),
            (self.q_overshoot_m, "record q overshoot"),
            (self.condition_number, "record projector condition"),
        ):
            if not np.isfinite(float(value)):
                raise ValueError(f"{label} must be finite")
        if int(self.M) <= 0 or int(self.rank) <= 0:
            raise ValueError("record projector M and rank must be positive")
        object.__setattr__(self, "antenna_native_m", antenna)
        object.__setattr__(self, "c_native_m", center)

    def as_dict(self) -> dict[str, Any]:
        return {
            "identity_key": list(self.identity_key),
            "frequency_group": int(self.group_id),
            "r0_source_m": float(self.r0_source_m),
            "antenna_native_m": self.antenna_native_m.tolist(),
            "c_native_m": self.c_native_m.tolist(),
            "Rc_m": float(self.r_c_m),
            "s_c_m": float(self.s_c_m),
            "Rmin_m": float(self.r_min_m),
            "Rmax_m": float(self.r_max_m),
            "nominal_ambiguity_period_U_m": float(self.ambiguity_period_m),
            "nominal_principal_cell_m": [float(self.principal_cell_lower_m), float(self.principal_cell_upper_m)],
            "nominal_principal_cell_guard_pass": bool(self.principal_cell_guard_pass),
            "same_delay_exterior_capture_nominal": dict(self.same_delay_exterior_capture_nominal),
            "qminus_m": float(self.q_minus_m),
            "qplus_m": float(self.q_plus_m),
            "q_last_m": float(self.q_last_m),
            "q_overshoot_m": float(self.q_overshoot_m),
            "K": int(self.frequency_count),
            "M": int(self.M),
            "rank": int(self.rank),
            "condition_number": float(self.condition_number),
            "sampled_in_cube_retention_min": float(self.sampled_in_cube_retention_min),
            "sampled_in_cube_retention_max": float(self.sampled_in_cube_retention_max),
            "sampled_in_cube_sample_count": int(self.sampled_in_cube_sample_count),
            "exterior_leakage_curve": dict(self.exterior_leakage_curve),
        }


@dataclass(frozen=True)
class _RangeQEntry:
    group_id: int
    M: int
    Q_centered: np.ndarray
    Q_full: np.ndarray
    R_centered: np.ndarray
    rank_E: int
    condition_E: float
    matrix_entries: int


def _range_h_value(centered_frequencies_hz: np.ndarray, q_m: float) -> float:
    values = np.exp(-1j * (4.0 * np.pi / SPEED_OF_LIGHT_M_S) * centered_frequencies_hz * float(q_m))
    return float(np.abs(np.mean(values)) ** 2)


def _refine_range_minimum(
    frequencies_hz: np.ndarray,
    left_m: float,
    right_m: float,
) -> float:
    """Refine one sampled local-minimum bracket using exactly 64 iterations."""

    centered = frequencies_hz - np.mean(frequencies_hz, dtype=np.float64)
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    left = float(left_m)
    right = float(right_m)
    c = right - ratio * (right - left)
    d = left + ratio * (right - left)
    fc = _range_h_value(centered, c)
    fd = _range_h_value(centered, d)
    for _ in range(RANGE_SUBSPACE_GOLDEN_ITERATIONS):
        if fc < fd:
            right, d, fd = d, c, fc
            c = right - ratio * (right - left)
            fc = _range_h_value(centered, c)
        else:
            left, c, fc = c, d, fd
            d = left + ratio * (right - left)
            fd = _range_h_value(centered, d)
    return float(0.5 * (left + right))


def analyze_range_frequency_group(
    frequencies_hz: Any,
    *,
    group_id: int = 0,
) -> RangeFrequencyGroup:
    """Analyze one stored native float64 frequency vector without regridding."""

    original = np.asarray(frequencies_hz)
    if original.dtype != np.dtype(np.float64):
        raise TypeError("range projector requires the stored native frequency vector to be float64")
    frequencies = np.asarray(original, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size < 3:
        raise ValueError("range projector requires at least three native frequencies")
    if not np.isfinite(frequencies).all() or not np.all(np.diff(frequencies) > 0.0):
        raise ValueError("native frequencies must be finite and strictly increasing")
    bandwidth = float(np.float64(frequencies[-1]) - np.float64(frequencies[0]))
    r_rayleigh = float(SPEED_OF_LIGHT_M_S / (2.0 * bandwidth))
    centered = frequencies - np.mean(frequencies, dtype=np.float64)
    sample_grid = np.linspace(0.0, 2.0 * r_rayleigh, RANGE_SUBSPACE_SAMPLES, dtype=np.float64)
    values = np.asarray([_range_h_value(centered, value) for value in sample_grid], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("sampled native-band point response is non-finite")
    minima = np.flatnonzero(
        (values[1:-1] < values[:-2]) & (values[1:-1] < values[2:])
    ) + 1
    if minima.size == 0:
        raise ValueError("native-band point response has no first positive strict bracketed local minimum")
    first_index = int(minima[0])
    bracket = (float(sample_grid[first_index - 1]), float(sample_grid[first_index + 1]))
    g = _refine_range_minimum(frequencies, bracket[0], bracket[1])
    if not (0.5 * r_rayleigh <= g <= 1.5 * r_rayleigh):
        raise ValueError(
            f"native-band point-response minimum g={g} lies outside the fixed [0.5,1.5]*rRayleigh guard"
        )
    df = np.diff(frequencies)
    affine = np.linspace(frequencies[0], frequencies[-1], frequencies.size, dtype=np.float64)
    endpoint_deviation = float(np.max(np.abs(frequencies - affine)))
    return RangeFrequencyGroup(
        group_id=int(group_id),
        frequencies_hz=frequencies,
        centered_frequencies_hz=centered,
        frequency_count=int(frequencies.size),
        first_hz=float(frequencies[0]),
        last_hz=float(frequencies[-1]),
        min_df_hz=float(np.min(df)),
        median_df_hz=float(np.median(df)),
        max_df_hz=float(np.max(df)),
        endpoint_affine_deviation_hz=endpoint_deviation,
        r_rayleigh_m=r_rayleigh,
        g_m=g,
        first_minimum_sample_index=first_index,
        first_minimum_bracket_m=bracket,
        first_minimum_h=float(values[first_index]),
    )


def _range_group_records(records: Sequence[Any]) -> tuple[tuple[RangeFrequencyGroup, ...], tuple[int, ...]]:
    groups: list[RangeFrequencyGroup] = []
    group_ids: list[int] = []
    for record in records:
        frequencies = np.asarray(getattr(record, "frequencies_hz"))
        if frequencies.dtype != np.dtype(np.float64):
            raise TypeError("range projector requires every stored native frequency vector to be float64")
        assigned = None
        for group in groups:
            if np.array_equal(frequencies, group.frequencies_hz):
                assigned = int(group.group_id)
                break
        if assigned is None:
            assigned = len(groups)
            groups.append(analyze_range_frequency_group(frequencies, group_id=assigned))
        group_ids.append(assigned)
    if not groups:
        raise ValueError("range projector requires at least one record")
    return tuple(groups), tuple(group_ids)


def _range_box_bounds(
    model: NativeComplexDirectionalSHROIModel,
    antenna_native_m: np.ndarray,
) -> tuple[float, float, np.ndarray, float]:
    lower = np.asarray(model.support.lower_edge_m, dtype=np.float64)
    upper = np.asarray(model.support.upper_edge_exclusive_m, dtype=np.float64)
    c_native = model.placement.apply(np.zeros((1, 3), dtype=np.float64))[0]
    antenna_local = (antenna_native_m - model.placement.translation_m) @ model.placement.rotation
    closest_local = np.clip(antenna_local, lower, upper)
    closest_native = model.placement.apply(closest_local[None, :])[0]
    r_min = float(np.linalg.norm(antenna_native_m - closest_native))
    corners = np.asarray(
        [
            [x, y, z]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ],
        dtype=np.float64,
    )
    r_max = float(np.max(np.linalg.norm(model.placement.apply(corners) - antenna_native_m[None, :], axis=1)))
    r_c = float(np.linalg.norm(antenna_native_m - c_native))
    return r_min, r_max, c_native, r_c


def _range_delay_samples(q_low: float, q_high: float, step: float) -> np.ndarray:
    if q_high < q_low:
        raise ValueError("range delay interval is reversed")
    values = np.arange(float(q_low), float(q_high), float(step), dtype=np.float64)
    values = values[values <= float(q_high)]
    values = np.concatenate((values, np.asarray([float(q_low), float(q_high)], dtype=np.float64)))
    return np.unique(values)


class RangeSubspacePlan:
    """Exact-native, matrix-free range projector plan for selected records."""

    def __init__(
        self,
        model: NativeComplexDirectionalSHROIModel,
        records: Sequence[Any],
        *,
        require_retention: bool = True,
        h_m: float | None = None,
    ) -> None:
        self.model = model
        self.records = tuple(records)
        self.h_m = None if h_m is None else float(h_m)
        if self.h_m is not None and (not np.isfinite(self.h_m) or self.h_m <= 0.0):
            raise ValueError("range projector h must be positive and finite")
        if not self.records:
            raise ValueError("range projector requires selected records")
        if any(str(getattr(record, "role", "")).lower() == "test" for record in self.records):
            raise ValueError("TEST observations are sealed out of range-projector paths")
        for record in self.records:
            _record_response(record)
        self.groups, group_ids = _range_group_records(self.records)
        if self.h_m is not None and any(self.h_m != float(group.g_m) for group in self.groups):
            raise ValueError("range projector scored mode requires h=g exactly; h=g/2 is rejected")
        self._group_for_identity = {
            _id_key(record.identity): int(group_id)
            for record, group_id in zip(self.records, group_ids)
        }
        if len(self._group_for_identity) != len(self.records):
            raise ValueError("range projector records must have unique canonical identities")
        self._q_cache: dict[tuple[int, int], _RangeQEntry] = {}
        self._record_geometry: dict[tuple[int, str, int, int], RangeRecordGeometry] = {}
        self._retention: dict[tuple[int, str, int, int], dict[str, Any]] = {}
        for record, group_id in zip(self.records, group_ids):
            identity_key = _id_key(record.identity)
            frequencies = np.asarray(record.frequencies_hz)
            antenna = np.asarray(record.position_xyz_m, dtype=np.float64)
            source_r0 = _record_source_r0(record)
            r_min, r_max, c_native, r_c = _range_box_bounds(model, antenna)
            group = self.groups[group_id]
            q_minus = float(r_min - r_c - RANGE_SUBSPACE_GUARD_CELLS * group.g_m)
            q_plus = float(r_max - r_c + RANGE_SUBSPACE_GUARD_CELLS * group.g_m)
            ambiguity_period = float(SPEED_OF_LIGHT_M_S / (2.0 * group.max_df_hz))
            principal_lower = float(-0.5 * ambiguity_period)
            principal_upper = float(0.5 * ambiguity_period)
            principal_guard_pass = bool(q_minus >= principal_lower and q_plus <= principal_upper)
            if not principal_guard_pass:
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
            same_delay_diagnostic = {
                "period_U_m": ambiguity_period,
                "principal_cell_m": [principal_lower, principal_upper],
                "unguarded_in_cube_interval_m": [unguarded_lower, unguarded_upper],
                "guarded_interval_m": [q_minus, q_plus],
                "candidate_exterior_delays_m": alias_candidates,
                "captured_exterior_delays": same_delay_captures,
                "same_delay_exterior_capture_possible": bool(same_delay_captures),
                "interpretation": "nominal frequency-alias diagnostic only; no physical exterior returns were separated",
            }
            span = float(q_plus - q_minus)
            M = int(1 + np.ceil(span / group.g_m))
            if M >= int(group.frequency_count):
                raise ValueError(
                    f"range projector requires M<K; got M={M}, K={group.frequency_count} for {identity_key}"
                )
            entry = self._q_entry(group_id, M)
            rank = int(entry.rank_E)
            condition = float(entry.condition_E)
            if rank != M or not np.isfinite(condition) or condition > 10.0:
                raise ValueError(
                    f"range projector failed rank(E)=M/conditioning guard for {identity_key}: "
                    f"M={M}, rank={rank}, condition={condition}"
                )
            q_last = float(q_minus + (M - 1) * group.g_m)
            q_overshoot = float(q_last - q_plus)
            if not (-np.finfo(np.float64).eps * max(1.0, abs(q_plus)) <= q_overshoot < group.g_m):
                raise ValueError("range projector final grid overshoot is outside [0,g)")
            in_cube_q = _range_delay_samples(r_min - r_c, r_max - r_c, group.g_m / 8.0)
            in_cube_values = self._retention_values(record, group_id, M, in_cube_q, q_offset_m=q_minus)
            retention_min = float(np.min(in_cube_values))
            retention_max = float(np.max(in_cube_values))
            if require_retention and retention_min < RANGE_SUBSPACE_RETENTION_THRESHOLD:
                raise ValueError(
                    f"range projector fixed-2g sampled in-cube retention failed: {retention_min:.9g} < "
                    f"{RANGE_SUBSPACE_RETENTION_THRESHOLD:.9g}"
                )
            exterior_q = np.concatenate(
                (
                    np.linspace(r_min - r_c - 4.0 * group.g_m, r_min - r_c - group.g_m, 4),
                    np.linspace(r_min - r_c, r_max - r_c, 9),
                    np.linspace(r_max - r_c + group.g_m, r_max - r_c + 4.0 * group.g_m, 4),
                    np.asarray(alias_candidates, dtype=np.float64),
                )
            )
            exterior_values = self._retention_values(record, group_id, M, exterior_q, q_offset_m=q_minus)
            leakage = {
                "q_m": exterior_q.tolist(),
                "projected_energy_fraction": exterior_values.tolist(),
                "outside_q_m": {
                    "lower": float(r_min - r_c),
                    "upper": float(r_max - r_c),
                },
                "guarded_q_m": {"lower": q_minus, "upper": q_plus},
            }
            geometry = RangeRecordGeometry(
                identity_key=identity_key,
                group_id=int(group_id),
                r0_source_m=source_r0,
                antenna_native_m=antenna,
                c_native_m=c_native,
                r_c_m=r_c,
                s_c_m=float(r_c - source_r0),
                frequency_count=int(group.frequency_count),
                r_min_m=r_min,
                r_max_m=r_max,
                ambiguity_period_m=ambiguity_period,
                principal_cell_lower_m=principal_lower,
                principal_cell_upper_m=principal_upper,
                principal_cell_guard_pass=principal_guard_pass,
                same_delay_exterior_capture_nominal=same_delay_diagnostic,
                q_minus_m=q_minus,
                q_plus_m=q_plus,
                q_last_m=q_last,
                q_overshoot_m=q_overshoot,
                M=M,
                rank=rank,
                condition_number=condition,
                sampled_in_cube_retention_min=retention_min,
                sampled_in_cube_retention_max=retention_max,
                sampled_in_cube_sample_count=int(in_cube_q.size),
                exterior_leakage_curve=leakage,
            )
            self._record_geometry[identity_key] = geometry
            self._retention[identity_key] = {
                "q_m": in_cube_q.tolist(),
                "projected_energy_fraction": in_cube_values.tolist(),
                "minimum": retention_min,
                "maximum": retention_max,
                "threshold": RANGE_SUBSPACE_RETENTION_THRESHOLD,
            }
        self._gauge_report = self._check_frequency_gauges()

    def _q_entry(self, group_id: int, M: int) -> _RangeQEntry:
        key = (int(group_id), int(M))
        cached = self._q_cache.get(key)
        if cached is not None:
            return cached
        group = self.groups[int(group_id)]
        columns = np.arange(int(M), dtype=np.float64) * float(group.g_m)
        kappa = 4.0 * np.pi / SPEED_OF_LIGHT_M_S
        centered_e = np.exp(-1j * kappa * group.centered_frequencies_hz[:, None] * columns[None, :])
        full_e = np.exp(-1j * kappa * group.frequencies_hz[:, None] * columns[None, :])
        centered_e /= np.sqrt(float(group.frequency_count))
        full_e /= np.sqrt(float(group.frequency_count))
        q_centered, r_centered = np.linalg.qr(centered_e, mode="reduced")
        q_full, _ = np.linalg.qr(full_e, mode="reduced")
        if not np.isfinite(q_centered.real).all() or not np.isfinite(q_centered.imag).all():
            raise ValueError("centered-frequency QR basis is non-finite")
        entry = _RangeQEntry(
            group_id=int(group_id),
            M=int(M),
            Q_centered=_readonly(q_centered, np.complex128),
            Q_full=_readonly(q_full, np.complex128),
            R_centered=_readonly(r_centered, np.complex128),
            rank_E=int(np.linalg.matrix_rank(r_centered, tol=1.0e-12)),
            condition_E=float(np.linalg.cond(r_centered)),
            matrix_entries=int(group.frequency_count * M),
        )
        self._q_cache[key] = entry
        return entry

    def _geometry_for(self, record: Any) -> RangeRecordGeometry:
        try:
            return self._record_geometry[_id_key(record.identity)]
        except KeyError as exc:
            raise ValueError("record is not part of this range projector plan") from exc

    def _q_for(self, record: Any) -> tuple[RangeFrequencyGroup, RangeRecordGeometry, _RangeQEntry]:
        geometry = self._geometry_for(record)
        group = self.groups[geometry.group_id]
        return group, geometry, self._q_entry(geometry.group_id, geometry.M)

    def _retention_values(
        self,
        record: Any,
        group_id: int,
        M: int,
        q_values_m: Any,
        *,
        q_offset_m: float,
    ) -> np.ndarray:
        group = self.groups[int(group_id)]
        entry = self._q_entry(group_id, M)
        q_values = np.asarray(q_values_m, dtype=np.float64)
        kappa = 4.0 * np.pi / SPEED_OF_LIGHT_M_S
        values = []
        # A pure centered-frequency delay probe is sufficient for projector
        # retention; the disclosed s_c gauge is applied in the actual operator.
        for q_value in q_values:
            probe = np.exp(-1j * kappa * group.centered_frequencies_hz * float(q_value - q_offset_m))
            transformed = entry.Q_centered.conj().T @ probe
            values.append(float(np.vdot(transformed, transformed).real / float(group.frequency_count)))
        result = np.asarray(values, dtype=np.float64)
        if not np.isfinite(result).all():
            raise ValueError("sampled range retention is non-finite")
        return result

    @staticmethod
    def _validate_values(records: Sequence[Any], values: RaggedComplexValues) -> tuple[Any, ...]:
        records = tuple(records)
        if tuple(values.ids) != tuple(record.identity for record in records):
            raise ValueError("range values and records have different canonical IDs")
        return records

    def apply_forward(self, records: Sequence[Any], values: RaggedComplexValues) -> RaggedComplexValues:
        records = self._validate_values(records, values)
        kappa = 4.0 * np.pi / SPEED_OF_LIGHT_M_S
        transformed = []
        for record, value in zip(records, values.values):
            group, geometry, entry = self._q_for(record)
            if np.asarray(value).shape != (group.frequency_count,):
                raise ValueError("range forward values must have native K frequency samples")
            phase = np.exp(1j * kappa * group.centered_frequencies_hz * (geometry.s_c_m + geometry.q_minus_m))
            transformed.append(entry.Q_centered.conj().T @ (phase * np.asarray(value, dtype=np.complex128)))
        return RaggedComplexValues(values.ids, tuple(transformed))

    def apply_adjoint(self, records: Sequence[Any], values: RaggedComplexValues) -> RaggedComplexValues:
        records = tuple(records)
        if tuple(values.ids) != tuple(record.identity for record in records):
            raise ValueError("range values and records have different canonical IDs")
        kappa = 4.0 * np.pi / SPEED_OF_LIGHT_M_S
        native = []
        for record, value in zip(records, values.values):
            group, geometry, entry = self._q_for(record)
            if np.asarray(value).shape != (entry.M,):
                raise ValueError("range adjoint values must have M projected samples")
            phase = np.exp(-1j * kappa * group.centered_frequencies_hz * (geometry.s_c_m + geometry.q_minus_m))
            native.append(phase * (entry.Q_centered @ np.asarray(value, dtype=np.complex128)))
        return RaggedComplexValues(values.ids, tuple(native))

    def apply_normal(self, records: Sequence[Any], values: RaggedComplexValues) -> RaggedComplexValues:
        return self.apply_adjoint(records, self.apply_forward(records, values))

    def transform_targets(self, records: Sequence[Any]) -> RaggedComplexValues:
        records = tuple(records)
        return self.apply_forward(
            records,
            RaggedComplexValues(
                tuple(record.identity for record in records),
                tuple(_record_response(record) for record in records),
            ),
        )

    def normalization(self, records: Sequence[Any]) -> int:
        # The rank(E)=M gate has already passed for every record; keep the
        # declared range-observable normalization in terms of M, not native K.
        return int(sum(self._geometry_for(record).M for record in records))

    def _check_frequency_gauges(self) -> dict[str, Any]:
        kappa = 4.0 * np.pi / SPEED_OF_LIGHT_M_S
        maximum = 0.0
        comparisons = []
        for record in self.records:
            group, geometry, entry = self._q_for(record)
            delay = geometry.s_c_m + 0.37 * group.g_m
            centered_probe = np.exp(-1j * kappa * group.centered_frequencies_hz * delay)
            full_probe = np.exp(-1j * kappa * group.frequencies_hz * delay)
            centered_z = entry.Q_centered.conj().T @ (
                np.exp(1j * kappa * group.centered_frequencies_hz * (geometry.s_c_m + geometry.q_minus_m))
                * centered_probe
            )
            full_z = entry.Q_full.conj().T @ (
                np.exp(1j * kappa * group.frequencies_hz * (geometry.s_c_m + geometry.q_minus_m))
                * full_probe
            )
            centered_projection = np.exp(-1j * kappa * group.centered_frequencies_hz * (geometry.s_c_m + geometry.q_minus_m)) * (
                entry.Q_centered @ centered_z
            )
            full_projection = np.exp(-1j * kappa * group.frequencies_hz * (geometry.s_c_m + geometry.q_minus_m)) * (
                entry.Q_full @ full_z
            )
            # The full-frequency probe differs from its centered-frequency
            # gauge only by this common phase.  Remove it before comparing the
            # two native projected vectors.
            full_projection *= np.exp(1j * kappa * float(np.mean(group.frequencies_hz)) * delay)
            denominator = max(float(np.linalg.norm(centered_projection)), np.finfo(np.float64).tiny)
            relative = float(np.linalg.norm(centered_projection - full_projection) / denominator)
            maximum = max(maximum, relative)
            comparisons.append({"identity_key": list(_id_key(record.identity)), "relative_projection_error": relative})
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

    def metadata(self) -> dict[str, Any]:
        group_records = []
        for group in self.groups:
            members = [
                list(_id_key(record.identity))
                for record in self.records
                if self._group_for_identity[_id_key(record.identity)] == group.group_id
            ]
            group_records.append(
                {
                    "group_id": int(group.group_id),
                    "record_identity_keys": members,
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
        records = [self._record_geometry[_id_key(record.identity)].as_dict() for record in self.records]
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
            "Q_cache_key": "(frequency_group,M)",
            "dense_P_or_T_materialized": False,
            "rank_condition_source": "centered QR R factor from original E; rank(E)=M and cond(E)=cond(R) are gated",
            "centered_full_frequency_gauge": dict(self.gauge_report),
            "sampled_in_cube_retention_threshold": RANGE_SUBSPACE_RETENTION_THRESHOLD,
            "nominal_ambiguity_diagnostic": {
                "period_definition": "U=c/(2*max_df)",
                "principal_cell": "[-U/2,U/2] containing q=0",
                "guarded_interval_must_be_inside_principal_cell": True,
                "same_delay_exterior_capture_is_nominal_only": True,
                "off_grid_and_exterior_leakage_are_disclosed_not_selector_tuned": True,
            },
            "sampled_in_cube_retention": {
                "records": {
                    str(list(key)): dict(value) for key, value in self._retention.items()
                },
                "minimum": float(min(value["minimum"] for value in self._retention.values())),
                "maximum": float(max(value["maximum"] for value in self._retention.values())),
            },
            "exterior_leakage": {
                "definition": "projected energy fraction along deterministic four-point exterior/interior curve",
                "records": {str(list(key)): dict(self._record_geometry[key].exterior_leakage_curve) for key in self._record_geometry},
            },
        }


def build_range_subspace_plan(
    model: NativeComplexDirectionalSHROIModel,
    records: Sequence[Any],
    *,
    require_retention: bool = True,
    h_m: float | None = None,
) -> RangeSubspacePlan:
    """Build and guard the exact-native range plan before fitting or conversion."""

    return RangeSubspacePlan(model, records, require_retention=require_retention, h_m=h_m)


def _range_targets_and_prediction(
    model: NativeComplexDirectionalSHROIModel,
    records: Sequence[Any],
    coefficients: Any,
    plan: RangeSubspacePlan,
) -> tuple[RaggedComplexValues, RaggedComplexValues, RaggedComplexValues]:
    records = tuple(records)
    native_prediction = model.forward(records, coefficients)
    transformed_prediction = plan.apply_forward(records, native_prediction)
    transformed_target = plan.transform_targets(records)
    return native_prediction, transformed_prediction, transformed_target


def range_loss_and_gradient(
    model: NativeComplexDirectionalSHROIModel,
    records: Sequence[Any],
    coefficients: Any,
    plan: RangeSubspacePlan,
    *,
    transformed_targets: RaggedComplexValues | None = None,
) -> tuple[float, np.ndarray, RaggedComplexValues]:
    """Return the T-aware loss/gradient using only streamed B and B^H."""

    records = tuple(records)
    if not records:
        raise ValueError("range loss requires at least one record")
    native_prediction = model.forward(records, coefficients)
    transformed_prediction = plan.apply_forward(records, native_prediction)
    target = plan.transform_targets(records) if transformed_targets is None else transformed_targets
    if tuple(target.ids) != tuple(transformed_prediction.ids):
        raise ValueError("transformed target identities do not match predictions")
    residual = RaggedComplexValues(
        transformed_prediction.ids,
        tuple(p - y for p, y in zip(transformed_prediction.values, target.values)),
    )
    n_t = plan.normalization(records)
    if n_t <= 0:
        raise ValueError("range loss requires positive N_T normalization")
    loss = 0.5 * sum(float(np.vdot(value, value).real) for value in residual.values) / float(n_t)
    native_residual = plan.apply_adjoint(records, residual)
    gradient = model.gradient_from_native_residuals(records, native_residual, normalization=float(n_t))
    return float(loss), gradient, transformed_prediction


def range_metrics(
    model: NativeComplexDirectionalSHROIModel,
    records: Sequence[Any] | None,
    coefficients: Any,
    plan: RangeSubspacePlan,
    *,
    split: str,
) -> dict[str, Any] | None:
    """Report transformed-domain metrics plus the separately scoped native residual."""

    if records is None or not tuple(records):
        return None
    records = tuple(records)
    native_prediction, transformed_prediction, target = _range_targets_and_prediction(
        model, records, coefficients, plan
    )
    return range_metrics_from_predictions(
        model,
        records,
        native_prediction,
        transformed_prediction,
        target,
        plan,
        split=split,
    )


def range_metrics_from_predictions(
    model: NativeComplexDirectionalSHROIModel,
    records: Sequence[Any],
    native_prediction: RaggedComplexValues,
    transformed_prediction: RaggedComplexValues,
    target: RaggedComplexValues,
    plan: RangeSubspacePlan,
    *,
    split: str,
) -> dict[str, Any]:
    """Score supplied native/transformed predictions without an extra render."""

    records = tuple(records)
    if tuple(native_prediction.ids) != tuple(record.identity for record in records):
        raise ValueError("native prediction IDs do not match records")
    if tuple(transformed_prediction.ids) != tuple(record.identity for record in records):
        raise ValueError("transformed prediction IDs do not match records")
    if tuple(target.ids) != tuple(record.identity for record in records):
        raise ValueError("transformed target IDs do not match records")
    residual = tuple(p - y for p, y in zip(transformed_prediction.values, target.values))
    transformed_prediction_energy = float(sum(np.vdot(value, value).real for value in transformed_prediction.values))
    target_energy = float(sum(np.vdot(value, value).real for value in target.values))
    residual_energy = float(sum(np.vdot(value, value).real for value in residual))
    native_target = tuple(_record_response(record) for record in records)
    native_target_energy = float(sum(np.vdot(value, value).real for value in native_target))
    native_residual = tuple(
        prediction - target_value for prediction, target_value in zip(native_prediction.values, native_target)
    )
    native_residual_energy = float(sum(np.vdot(value, value).real for value in native_residual))
    correlation_denominator = np.sqrt(max(target_energy * transformed_prediction_energy, 0.0))
    complex_correlation = (
        None
        if correlation_denominator == 0.0
        else abs(sum(np.vdot(y, p) for y, p in zip(target.values, transformed_prediction.values))) / correlation_denominator
    )
    return {
        "split": str(split),
        "record_count": len(records),
        "native_frequency_sample_count": int(sum(value.size for value in native_target)),
        "N_T": int(plan.normalization(records)),
        "range_focused_complex_RelMSE": None if target_energy == 0.0 else residual_energy / target_energy,
        "range_prediction_energy": transformed_prediction_energy,
        "range_target_energy": target_energy,
        "range_residual_energy": residual_energy,
        "range_complex_correlation": complex_correlation,
        "raw_energy_fraction_retained_by_B": None if native_target_energy == 0.0 else target_energy / native_target_energy,
        "raw_native_complex_relmse_scoped_diagnostic": None if native_target_energy == 0.0 else native_residual_energy / native_target_energy,
        "native_complex_residual_energy_scoped_diagnostic": native_residual_energy,
        "native_complex_target_energy_scoped_diagnostic": native_target_energy,
    }


def range_isotropic_bp(
    model: NativeComplexDirectionalSHROIModel,
    train_records: Sequence[Any],
    validation_records: Sequence[Any] | None,
    plan: RangeSubspacePlan,
) -> dict[str, Any]:
    """Compute the matched complex isotropic BP arm with TRAIN-only scaling."""

    train = tuple(train_records)
    n_t = plan.normalization(train)
    if n_t <= 0:
        raise ValueError("isotropic BP requires positive TRAIN N_T")
    transformed_target = plan.transform_targets(train)
    native_target = plan.apply_adjoint(train, transformed_target)
    b = model.gradient_from_native_residuals(train, native_target, normalization=float(n_t))
    b = np.asarray(b, dtype=np.complex128)
    b[:, 1:] = 0.0 + 0.0j
    native_bp_train = model.forward(train, b)
    transformed_bp_train = plan.apply_forward(train, native_bp_train)
    numerator = sum(np.vdot(p, y) for p, y in zip(transformed_bp_train.values, transformed_target.values))
    denominator = float(sum(np.vdot(p, p).real for p in transformed_bp_train.values))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise FloatingPointError("range isotropic BP has a nonpositive or non-finite TRAIN denominator")
    alpha = numerator / denominator
    if not np.isfinite(alpha.real) or not np.isfinite(alpha.imag):
        raise FloatingPointError("range isotropic BP alpha is non-finite")
    coefficients = np.asarray(alpha * b, dtype=np.complex128)
    return {
        "coefficients": coefficients,
        "unscaled_isotropic_coefficients": b,
        "alpha": complex(alpha),
        "train_denominator": denominator,
        "train_numerator": complex(numerator),
        "train_N_T": int(n_t),
        "validation_N_T": None if validation_records is None else int(plan.normalization(validation_records)),
        "basis_channels_retained": ["Y00"],
        "validation_did_not_enter_scaling": True,
    }


def range_comparison_resource_ledger(
    model: NativeComplexDirectionalSHROIModel,
    train_frequency_counts: Sequence[int],
    validation_frequency_counts: Sequence[int],
    plan: RangeSubspacePlan,
    *,
    saved_native_reuse: bool,
    saved_native_fallback: bool,
    actual_iterations: int = MAX_CGLS_ITERATIONS,
    diagnostic_steps: Sequence[int] = CGLS_DIAGNOSTIC_STEPS,
    cgls_breakdown: bool = False,
) -> dict[str, Any]:
    """Separate direct field-kernel work from projector/setup work."""

    train_samples = int(sum(int(value) for value in train_frequency_counts))
    validation_samples = int(sum(int(value) for value in validation_frequency_counts))
    K = int(model.point_count)
    diagnostic_steps = tuple(sorted(set(int(value) for value in diagnostic_steps)))
    diagnostics = int(len(diagnostic_steps))
    saved_terms = int(K * (train_samples + validation_samples)) if saved_native_fallback else 0
    bp_adjoint_terms = int(K * train_samples)
    # The BP primitive renders the unscaled b on TRAIN, then the selected
    # alpha*b is reloaded and rendered on both splits for final artifacts.
    bp_forward_terms = int(K * (train_samples + train_samples + validation_samples))
    actual_iterations = int(actual_iterations)
    if not 0 <= actual_iterations <= MAX_CGLS_ITERATIONS:
        raise ValueError("actual_iterations must lie within the CGLS maximum")
    cgls_diagnostic_terms = int(diagnostics * K * (3 * train_samples + 2 * validation_samples))
    cgls_failed_search_terms = int(K * train_samples) if cgls_breakdown else 0
    cgls_terms = int(
        2 * K * train_samples
        + 3 * actual_iterations * K * train_samples
        + cgls_diagnostic_terms
        + K * (train_samples + validation_samples)
        + cgls_failed_search_terms
    )
    max_cgls_terms = int(
        2 * K * train_samples
        + 3 * MAX_CGLS_ITERATIONS * K * train_samples
        + 4 * K * (3 * train_samples + 2 * validation_samples)
        + K * (train_samples + validation_samples)
    )
    entries = tuple(plan._q_cache.values())
    qr_entries = int(sum(entry.matrix_entries for entry in entries))
    probe_count = int(sum(geometry.sampled_in_cube_sample_count for geometry in plan._record_geometry.values()))
    probe_terms = int(
        sum(
            geometry.sampled_in_cube_sample_count
            * plan.groups[geometry.group_id].frequency_count
            * geometry.M
            for geometry in plan._record_geometry.values()
        )
    )
    return {
        "schema": f"{RANGE_COMPARISON_SCHEMA}.resource_ledger",
        "resource_estimate": {
            "class": "CPU-only NumPy exact-native range comparison; manager runtime preflight required",
            "gpu": False,
            "direct_term_definition": "support-point/native-frequency field-kernel terms; not FLOPs",
        },
        "train_frequency_sample_count": train_samples,
        "validation_frequency_sample_count": validation_samples,
        "direct_field_kernel_terms": {
            "saved_native_objective_cgls24_sidecar_reuse": 0,
            "saved_native_objective_cgls24_checkpoint_forward_fallback": saved_terms,
            "range_focused_isotropic_bp_adjoint_train": bp_adjoint_terms,
            "range_focused_isotropic_bp_forward_train_twice_validation_once": bp_forward_terms,
            "range_focused_isotropic_bp_forward_train_validation": bp_forward_terms,
            "range_focused_degree1_cgls24": cgls_terms,
        },
        "saved_native_sidecar_reused": bool(saved_native_reuse),
        "saved_native_checkpoint_forward_fallback": bool(saved_native_fallback),
        "saved_native_additional_direct_term_forecast": saved_terms,
        "total_direct_field_kernel_terms": int(saved_terms + bp_adjoint_terms + bp_forward_terms + cgls_terms),
        "total_direct_field_kernel_terms_max24_forecast": int(saved_terms + bp_adjoint_terms + bp_forward_terms + max_cgls_terms),
        "actual_executed_cgls_iterations": actual_iterations,
        "actual_cgls_diagnostic_steps": list(diagnostic_steps),
        "actual_cgls_breakdown_failed_search_forward_terms": cgls_failed_search_terms,
        "actual_direct_field_kernel_terms_are_exact_for_completed_path": True,
        "range_focused_degree1_cgls24_formula": (
            "2*K*F_train + 3*k*K*F_train + D*(3*K*F_train+2*K*F_validation) "
            "+ K*(F_train+F_validation)"
        ),
        "range_focused_degree1_cgls24_completed_path_terms": {
            "initial_gradient": int(2 * K * train_samples),
            "iterations": int(3 * actual_iterations * K * train_samples),
            "diagnostics_at_actual_steps": cgls_diagnostic_terms,
            "final_reload_artifacts": int(K * (train_samples + validation_samples)),
            "failed_search_forward": cgls_failed_search_terms,
            "total": cgls_terms,
            "max24_forecast_total": max_cgls_terms,
        },
        "qr_projection_setup": {
            "cache_key": "(frequency_group,M)",
            "cache_entry_count": len(entries),
            "centered_QR_matrix_entries": qr_entries,
            "full_frequency_gauge_QR_matrix_entries": qr_entries,
            "E_and_full_gauge_are_setup_work_not_direct_field_terms": True,
            "dense_P_or_T_materialized": False,
        },
        "sampled_synthetic_setup": {
            "probe_count": probe_count,
            "projection_frequency_matrix_terms": probe_terms,
            "retention_threshold": RANGE_SUBSPACE_RETENTION_THRESHOLD,
            "included_in_direct_field_kernel_total": False,
        },
        "operator_frequency_group_count": len(plan.groups),
        "operator_frequency_groups": [
            {
                "group_id": int(group.group_id),
                "K": int(group.frequency_count),
                "M_values": sorted(
                    {
                        int(geometry.M)
                        for geometry in plan._record_geometry.values()
                        if geometry.group_id == group.group_id
                    }
                ),
            }
            for group in plan.groups
        ],
    }


def _range_native_residuals(
    records: Sequence[Any],
    predictions: RaggedComplexValues,
) -> RaggedComplexValues:
    return RaggedComplexValues(
        predictions.ids,
        tuple(
            prediction - _record_response(record)
            for record, prediction in zip(records, predictions.values)
        ),
    )


def _save_range_sidecar(
    path: str | Path,
    values: RaggedComplexValues,
    *,
    domain: str,
    source_checkpoint: str,
    step: int,
) -> str:
    output = Path(path)
    payload = _ragged_sidecar_payload(values)
    payload["domain"] = np.asarray([str(domain)])
    payload["source_checkpoint"] = np.asarray([str(source_checkpoint)])
    payload["step"] = np.asarray([int(step)], dtype=np.int64)
    _save_npz(output, **payload)
    return str(output)


def fit_range_focused_cgls24(
    model: NativeComplexDirectionalSHROIModel,
    train_records: Sequence[Any],
    validation_records: Sequence[Any] | None,
    plan: RangeSubspacePlan,
    *,
    output_dir: str | Path | None = None,
    run_provenance: Mapping[str, Any] | None = None,
) -> FitResult:
    """Fit degree-1 coefficients with the fixed 24-step T-aware CGLS arm."""

    train = tuple(train_records)
    validation = tuple() if validation_records is None else tuple(validation_records)
    if int(plan.normalization(train)) <= 0:
        raise ValueError("range CGLS requires positive TRAIN N_T")
    coefficients = model.initial_coefficients()
    target_train = plan.transform_targets(train)
    target_validation = plan.transform_targets(validation) if validation else None
    _, gradient, _ = range_loss_and_gradient(
        model, train, coefficients, plan, transformed_targets=target_train
    )
    search = np.asarray(-gradient, dtype=np.complex128)
    direction = np.array(search, copy=True)
    gamma = float(np.vdot(search, search).real)
    if not np.isfinite(gamma) or gamma < 0.0:
        raise FloatingPointError("range CGLS normal-residual gamma is non-finite or negative")
    max_iterations = MAX_CGLS_ITERATIONS
    fixed_steps = CGLS_DIAGNOSTIC_STEPS
    history: list[dict[str, Any]] = []
    evaluated_steps: set[int] = set()
    checkpoint_paths: dict[str, str] = {}
    active_provenance = dict(DEFAULT_FAKE_PROVENANCE if run_provenance is None else run_provenance)
    active_provenance.update(
        {
            "solver": "range_focused_degree1_cgls24",
            "max_iterations": max_iterations,
            "range_operator": plan.metadata(),
        }
    )
    config = actual_fit_config(model, updates=max_iterations, solver="cgls")

    def evaluate(step: int, *, save_latest: bool) -> None:
        nonlocal coefficients
        train_loss, train_gradient, train_prediction = range_loss_and_gradient(
            model, train, coefficients, plan, transformed_targets=target_train
        )
        validation_loss = None
        validation_prediction = None
        if validation:
            validation_prediction = plan.apply_forward(validation, model.forward(validation, coefficients))
            validation_residual = RaggedComplexValues(
                validation_prediction.ids,
                tuple(p - y for p, y in zip(validation_prediction.values, target_validation.values)),
            )
            validation_loss = 0.5 * sum(float(np.vdot(value, value).real) for value in validation_residual.values) / float(
                plan.normalization(validation)
            )
        history.append(
            {
                "step": int(step),
                "train_loss": float(train_loss),
                "validation_loss": None if validation_loss is None else float(validation_loss),
                "range_train_loss": float(train_loss),
                "range_validation_loss": None if validation_loss is None else float(validation_loss),
                "validation_use": "diagnostic_only_fixed_point",
                "selected_checkpoint": False,
                "solver": "range_focused_degree1_cgls24",
                "cgls_iteration": int(step),
                "N_T_train": int(plan.normalization(train)),
                "N_T_validation": None if not validation else int(plan.normalization(validation)),
            }
        )
        evaluated_steps.add(int(step))
        if output_dir is not None and save_latest:
            active_provenance["diagnostic_steps"] = sorted(evaluated_steps)
            active_provenance["executed_iterations"] = int(step)
            active_provenance["selected_final_step"] = int(step)
            active_provenance["termination_reason"] = "not_yet_terminated"
            train_native = model.forward(train, coefficients)
            train_residual = _range_native_residuals(train, train_native)
            paths = save_checkpoint_bundle(
                output_dir,
                model=model,
                checkpoint_name="latest",
                step=step,
                coefficients=coefficients,
                history=history,
                predictions=train_native,
                residuals=train_residual,
                train_records=train,
                validation_records=validation,
                fit_config=config,
                gradient_record_scope="all_selected_train_records_range_projected",
                run_provenance=active_provenance,
            )
            checkpoint_paths.update({f"latest_{key}": value for key, value in paths.items()})
            checkpoint_paths["latest_range_predictions_train"] = _save_range_sidecar(
                Path(output_dir) / "range_predictions_latest_train.npz",
                train_prediction,
                domain="B_native_train",
                source_checkpoint=paths["checkpoint"],
                step=step,
            )
            if validation and validation_prediction is not None:
                val_native = model.forward(validation, coefficients)
                val_residual = _range_native_residuals(validation, val_native)
                val_paths = save_prediction_artifacts(
                    output_dir,
                    model=model,
                    artifact_name="latest_validation",
                    source_checkpoint=paths["checkpoint"],
                    step=step,
                    coefficients=coefficients,
                    history=history,
                    predictions=val_native,
                    residuals=val_residual,
                    train_records=train,
                    validation_records=validation,
                    fit_config=config,
                    gradient_record_scope="all_selected_train_records_range_projected",
                    run_provenance=active_provenance,
                )
                checkpoint_paths.update({f"latest_validation_{key}": value for key, value in val_paths.items()})
                checkpoint_paths["latest_range_predictions_validation"] = _save_range_sidecar(
                    Path(output_dir) / "range_predictions_latest_validation.npz",
                    validation_prediction,
                    domain="B_native_validation",
                    source_checkpoint=paths["checkpoint"],
                    step=step,
                )

    termination_reason: str | None = None
    completed_iterations = 0
    evaluate(0, save_latest=True)
    if gamma <= 0.0:
        termination_reason = "normal_equation_stationarity"
    if termination_reason is None:
        for iteration in range(1, max_iterations + 1):
            q = range_forward(model, train, direction, plan)
            denominator = float(sum(np.vdot(value, value).real for value in q.values)) / float(plan.normalization(train))
            if not np.isfinite(denominator):
                raise FloatingPointError("range CGLS search denominator is non-finite")
            if denominator <= 0.0:
                termination_reason = "nonpositive_search_denominator_breakdown"
                break
            alpha = float(gamma / denominator)
            if not np.isfinite(alpha):
                raise FloatingPointError("range CGLS alpha is non-finite")
            coefficients = np.asarray(coefficients + alpha * direction, dtype=np.complex128)
            _, gradient_new, _ = range_loss_and_gradient(
                model, train, coefficients, plan, transformed_targets=target_train
            )
            search_new = np.asarray(-gradient_new, dtype=np.complex128)
            gamma_new = float(np.vdot(search_new, search_new).real)
            if not np.isfinite(gamma_new) or gamma_new < 0.0:
                raise FloatingPointError("range CGLS refreshed gamma is non-finite or negative")
            completed_iterations = int(iteration)
            if iteration in fixed_steps:
                evaluate(iteration, save_latest=True)
            if gamma_new <= 0.0:
                termination_reason = "normal_equation_stationarity"
                if iteration not in evaluated_steps:
                    evaluate(iteration, save_latest=True)
                break
            if iteration == max_iterations:
                termination_reason = "max_iterations_reached"
                break
            beta = float(gamma_new / gamma)
            if not np.isfinite(beta):
                raise FloatingPointError("range CGLS beta is non-finite")
            direction = np.asarray(search_new + beta * direction, dtype=np.complex128)
            gamma = gamma_new
    if termination_reason is None:
        raise AssertionError("range CGLS termination reason was not resolved")
    if completed_iterations not in evaluated_steps:
        evaluate(completed_iterations, save_latest=True)
    active_provenance.update(
        {
            "executed_iterations": int(completed_iterations),
            "selected_final_step": int(completed_iterations),
            "termination_reason": str(termination_reason),
            "diagnostic_steps": sorted(evaluated_steps),
        }
    )
    if output_dir is not None:
        final_state = save_checkpoint_state(
            output_dir,
            model=model,
            checkpoint_name="final",
            step=completed_iterations,
            coefficients=coefficients,
            history=history,
            prediction_ids=tuple(record.identity for record in train),
            train_records=train,
            validation_records=validation,
            fit_config=config,
            artifacts_generated_after_checkpoint_reload=True,
            gradient_record_scope="all_selected_train_records_range_projected",
            run_provenance=active_provenance,
        )
        checkpoint_paths.update({f"final_{key}": value for key, value in final_state.items()})
        restored = load_checkpoint(final_state["checkpoint"])
        if restored["step"] != int(completed_iterations):
            raise AssertionError("range CGLS final checkpoint reload returned the wrong selected step")
        coefficients = model._validate_coefficients(restored["coefficients"])
        native_train = model.forward(train, coefficients)
        train_residual = _range_native_residuals(train, native_train)
        final_paths = save_prediction_artifacts(
            output_dir,
            model=model,
            artifact_name="final",
            source_checkpoint=final_state["checkpoint"],
            step=completed_iterations,
            coefficients=coefficients,
            history=history,
            predictions=native_train,
            residuals=train_residual,
            train_records=train,
            validation_records=validation,
            fit_config=config,
            artifacts_generated_after_checkpoint_reload=True,
            gradient_record_scope="all_selected_train_records_range_projected",
            run_provenance=active_provenance,
        )
        checkpoint_paths.update({f"final_{key}": value for key, value in final_paths.items()})
        checkpoint_paths["final_range_predictions_train"] = _save_range_sidecar(
            Path(output_dir) / "range_predictions_final_train.npz",
            plan.apply_forward(train, native_train),
            domain="B_native_train",
            source_checkpoint=final_state["checkpoint"],
            step=completed_iterations,
        )
        if validation:
            native_validation = model.forward(validation, coefficients)
            validation_residual = _range_native_residuals(validation, native_validation)
            validation_paths = save_prediction_artifacts(
                output_dir,
                model=model,
                artifact_name="final_validation",
                source_checkpoint=final_state["checkpoint"],
                step=completed_iterations,
                coefficients=coefficients,
                history=history,
                predictions=native_validation,
                residuals=validation_residual,
                train_records=train,
                validation_records=validation,
                fit_config=config,
                artifacts_generated_after_checkpoint_reload=True,
                gradient_record_scope="all_selected_train_records_range_projected",
                run_provenance=active_provenance,
            )
            checkpoint_paths.update({f"final_validation_{key}": value for key, value in validation_paths.items()})
            checkpoint_paths["final_range_predictions_validation"] = _save_range_sidecar(
                Path(output_dir) / "range_predictions_final_validation.npz",
                plan.apply_forward(validation, native_validation),
                domain="B_native_validation",
                source_checkpoint=final_state["checkpoint"],
                step=completed_iterations,
            )
    return FitResult(
        target_id=model.target_id,
        coefficients_latest=np.asarray(coefficients, dtype=np.complex128),
        coefficients_selected=np.asarray(coefficients, dtype=np.complex128),
        coefficients_final=np.asarray(coefficients, dtype=np.complex128),
        history=tuple(history),
        checkpoint_steps=tuple(sorted(evaluated_steps)),
        validation_steps=tuple(sorted(evaluated_steps)),
        checkpoint_paths=checkpoint_paths,
        solver="range_focused_degree1_cgls24",
        termination_reason=str(termination_reason),
        max_iterations=max_iterations,
        executed_iterations=int(completed_iterations),
        selected_final_step=int(completed_iterations),
        diagnostic_steps=tuple(sorted(evaluated_steps)),
    )


def range_forward(
    model: NativeComplexDirectionalSHROIModel,
    records: Sequence[Any],
    coefficients: Any,
    plan: RangeSubspacePlan,
) -> RaggedComplexValues:
    """Render A then apply B without materializing a normal matrix."""

    return plan.apply_forward(records, model.forward(records, coefficients))


def _render_projection_png(
    path: Path,
    projection: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    x_axis_label: str,
    y_axis_label: str,
    title: str,
    vmax: float,
) -> None:
    """Render one small labelled projection with Pillow when Matplotlib is absent."""

    try:
        from PIL import Image, ImageDraw, ImageFont
        from PIL.PngImagePlugin import PngInfo
    except ImportError as exc:  # pragma: no cover - depends on runtime extras.
        raise RuntimeError("selected-final figure output requires Pillow or matplotlib") from exc
    field = np.asarray(projection, dtype=np.float64)
    if field.ndim != 2 or not np.isfinite(field).all():
        raise ValueError("projection must be a finite two-dimensional energy field")
    normalized = np.clip(field.T[::-1, :] / float(vmax), 0.0, 1.0)
    color = np.empty((*normalized.shape, 3), dtype=np.uint8)
    color[:, :, 0] = np.asarray(np.clip(255.0 * normalized, 0.0, 255.0), dtype=np.uint8)
    color[:, :, 1] = np.asarray(
        np.clip(255.0 * (1.0 - np.abs(2.0 * normalized - 1.0)), 0.0, 255.0), dtype=np.uint8
    )
    color[:, :, 2] = np.asarray(np.clip(255.0 * (1.0 - normalized), 0.0, 255.0), dtype=np.uint8)
    scale = max(3, min(6, 600 // max(color.shape)))
    raster = Image.fromarray(color, mode="RGB").resize(
        (color.shape[1] * scale, color.shape[0] * scale),
        resample=Image.Resampling.NEAREST,
    )
    font = ImageFont.load_default()
    line_height = 14
    header_lines = [
        str(title),
        "coarse-support readout - not .1m fitted resolution",
        f"linear coefficient energy; shared scale [0,{float(vmax):.6g}]",
        f"local metre axes: x=[{lower[0]:g},{upper[0]:g}), {y_axis_label}=[{lower[2] if y_axis_label.startswith('local z') else lower[1]:g},{upper[2] if y_axis_label.startswith('local z') else upper[1]:g})",
    ]
    left, top, right, bottom = 82, line_height * len(header_lines) + 10, 124, 78
    image_width, image_height = raster.size
    canvas = Image.new("RGB", (left + image_width + right, top + image_height + bottom), "white")
    canvas.paste(raster, (left, top))
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(header_lines):
        draw.text((5, 5 + index * line_height), line, fill="black", font=font)
    draw.rectangle((left, top, left + image_width - 1, top + image_height - 1), outline="black")
    draw.text((left + image_width // 2 - 22, top + image_height + 32), x_axis_label, fill="black", font=font)
    draw.text((5, top + image_height // 2), y_axis_label, fill="black", font=font)
    x_ticks = (float(lower[0]), float((lower[0] + upper[0]) / 2.0), float(upper[0]))
    y_lower = float(lower[2] if y_axis_label.startswith("local z") else lower[1])
    y_upper = float(upper[2] if y_axis_label.startswith("local z") else upper[1])
    y_ticks = (y_lower, float((y_lower + y_upper) / 2.0), y_upper)
    for fraction, value in zip((0.0, 0.5, 1.0), x_ticks):
        x = int(round(left + fraction * (image_width - 1)))
        draw.line((x, top + image_height, x, top + image_height + 5), fill="black")
        draw.text((x - 12, top + image_height + 8), f"{value:g}", fill="black", font=font)
    for fraction, value in zip((1.0, 0.5, 0.0), y_ticks):
        y = int(round(top + fraction * (image_height - 1)))
        draw.line((left - 5, y, left, y), fill="black")
        draw.text((5, y - 4), f"{value:g}", fill="black", font=font)
    bar_x, bar_y, bar_w, bar_h = left + image_width + 18, top, 16, image_height
    bar_values = np.linspace(1.0, 0.0, bar_h)[:, None]
    bar_rgb = np.empty((bar_h, bar_w, 3), dtype=np.uint8)
    bar_rgb[:, :, 0] = np.asarray(255.0 * bar_values, dtype=np.uint8)
    bar_rgb[:, :, 1] = np.asarray(
        255.0 * (1.0 - np.abs(2.0 * bar_values - 1.0)), dtype=np.uint8
    )
    bar_rgb[:, :, 2] = np.asarray(255.0 * (1.0 - bar_values), dtype=np.uint8)
    canvas.paste(Image.fromarray(bar_rgb, mode="RGB"), (bar_x, bar_y))
    for fraction, label in ((0.0, f"{float(vmax):.4g}"), (1.0, "0")):
        y = int(round(bar_y + fraction * (bar_h - 1)))
        draw.line((bar_x + bar_w, y, bar_x + bar_w + 4, y), fill="black")
        draw.text((bar_x + bar_w + 7, y - 4), label, fill="black", font=font)
    draw.text((bar_x - 1, bar_y + bar_h + 7), "energy", fill="black", font=font)
    png_info = PngInfo()
    png_info.add_text("Title", str(title))
    png_info.add_text("Description", "\n".join(header_lines))
    png_info.add_text("Axes", "local metre axes; origin lower; equal aspect; max projection")
    canvas.save(path, format="PNG", pnginfo=png_info)


def _render_fit_figures_pillow(
    model: NativeComplexDirectionalSHROIModel,
    energy: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    output: Path,
    *,
    prefix: str,
    vmax: float,
) -> dict[str, str]:
    xy_path = output / f"{prefix}_xy_energy.png"
    xz_path = output / f"{prefix}_xz_energy.png"
    _render_projection_png(
        xy_path,
        np.max(energy, axis=2),
        lower=lower,
        upper=upper,
        x_axis_label="local x (m)",
        y_axis_label="local y (m)",
        title=f"{model.target_id} selected-final XY max-projection",
        vmax=vmax,
    )
    _render_projection_png(
        xz_path,
        np.max(energy, axis=1),
        lower=lower,
        upper=upper,
        x_axis_label="local x (m)",
        y_axis_label="local z (m)",
        title=f"{model.target_id} selected-final XZ max-projection",
        vmax=vmax,
    )
    return {"xy_figure": str(xy_path), "xz_figure": str(xz_path)}


def render_fit_figures(
    model: NativeComplexDirectionalSHROIModel,
    coefficients: Any,
    output_dir: str | Path,
    *,
    prefix: str = "selected_final",
) -> dict[str, str]:
    """Render disclosed XY and vertical energy views from one selected state."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    readout = model.readout(coefficients)
    energy = np.asarray(readout["energy"], dtype=np.float64)
    xy = np.max(energy, axis=2)
    xz = np.max(energy, axis=1)
    lower = np.asarray(readout["lower_edge_m"], dtype=np.float64)
    upper = np.asarray(readout["upper_edge_exclusive_m"], dtype=np.float64)
    xy_path = output / f"{prefix}_xy_energy.png"
    xz_path = output / f"{prefix}_xz_energy.png"
    vmin = 0.0
    vmax = max(float(np.max(xy)), float(np.max(xz)), np.finfo(np.float64).tiny)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return _render_fit_figures_pillow(
            model, energy, lower, upper, output, prefix=prefix, vmax=vmax
        )
    figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    image = axis.imshow(
        xy.T,
        origin="lower",
        aspect="equal",
        extent=(lower[0], upper[0], lower[1], upper[1]),
        vmin=vmin,
        vmax=vmax,
    )
    axis.set_title(f"{model.target_id} selected-final XY max-projection\n{READOUT_LABEL}")
    axis.set_xlabel("local x (m)")
    axis.set_ylabel("local y (m)")
    figure.colorbar(image, ax=axis, label="coefficient energy")
    figure.savefig(xy_path, dpi=140)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    image = axis.imshow(
        xz.T,
        origin="lower",
        aspect="equal",
        extent=(lower[0], upper[0], lower[2], upper[2]),
        vmin=vmin,
        vmax=vmax,
    )
    axis.set_title(f"{model.target_id} selected-final XZ max-projection\n{READOUT_LABEL}")
    axis.set_xlabel("local x (m)")
    axis.set_ylabel("local z (m)")
    figure.colorbar(image, ax=axis, label="coefficient energy; shared scale")
    figure.savefig(xz_path, dpi=140)
    plt.close(figure)
    return {"xy_figure": str(xy_path), "xz_figure": str(xz_path)}


def direct_term_resource_ledger(
    model: NativeComplexDirectionalSHROIModel,
    train_frequency_counts: Sequence[int],
    validation_frequency_counts: Sequence[int] | None = None,
    *,
    updates: int = DEFAULT_UPDATES,
    validation_steps: Sequence[int] = DEFAULT_CHECKPOINT_STEPS,
    solver: str = "gd",
    actual_iterations: int | None = None,
    termination_reason: str | None = None,
) -> dict[str, Any]:
    """Compute the direct-kernel term envelope for the selected solver."""

    solver = str(solver).lower()
    if solver not in {"gd", "cgls"}:
        raise ValueError("solver must be gd or cgls")
    updates = int(updates)
    if updates <= 0:
        raise ValueError("resource ledger updates must be positive")
    train_counts = tuple(int(value) for value in train_frequency_counts)
    validation_counts = tuple(int(value) for value in (validation_frequency_counts or ()))
    if not train_counts or any(value <= 0 for value in train_counts + validation_counts):
        raise ValueError("resource ledger requires positive frequency counts")
    train_samples = int(sum(train_counts))
    validation_samples = int(sum(validation_counts))
    support_points = int(model.point_count)
    validation_steps = tuple(int(value) for value in validation_steps)
    initial_terms = 0
    if solver == "cgls":
        cgls_iterations = updates if actual_iterations is None else int(actual_iterations)
        if not 0 <= cgls_iterations <= updates:
            raise ValueError("actual_iterations must lie between zero and the CGLS maximum")
        initial_normal_residual_terms = 2 * support_points * train_samples
        update_terms = 3 * cgls_iterations * support_points * train_samples
        failed_search_terms = (
            support_points * train_samples
            if str(termination_reason) == "nonpositive_search_denominator_breakdown"
            else 0
        )
    else:
        cgls_iterations = None
        initial_normal_residual_terms = 0
        update_terms = updates * 2 * support_points * train_samples
        failed_search_terms = 0
    diagnostic_samples = train_samples + validation_samples
    diagnostic_terms = len(validation_steps) * 2 * support_points * diagnostic_samples
    final_prediction_terms = support_points * (train_samples + validation_samples)
    report_metric_terms = support_points * (train_samples + validation_samples)
    total_terms = initial_terms + update_terms + diagnostic_terms + final_prediction_terms + report_metric_terms
    ledger = {
        "sample_term_definition": "one support-point/frequency direct-kernel term; not exact FLOPs",
        "support_points": support_points,
        "train_record_count": len(train_counts),
        "train_frequency_sample_count": train_samples,
        "validation_record_count": len(validation_counts),
        "validation_frequency_sample_count": validation_samples,
        "updates": updates,
        "validation_steps": [int(value) for value in validation_steps],
        "initial_adjoint_seed_terms": int(initial_terms),
        "zero_initialization_terms": 0,
        "full_aperture_update_terms": int(update_terms),
        "fixed_diagnostic_terms": int(diagnostic_terms),
        "selected_final_prediction_terms": int(final_prediction_terms),
        "postfit_metric_prediction_terms": int(report_metric_terms),
        "total_direct_kernel_sample_terms": int(total_terms),
        "spatial_chunk_size": model.spatial_chunk_size,
        "frequency_chunk_size": model.frequency_chunk_size,
        "full_aperture_gradient": True,
        "record_chunks_are_memory_scheduling_only": True,
        "resource_estimate": {
            "class": "CPU-only NumPy implementation; adequate headroom requires manager runtime preflight",
            "cpu": "not fixed before manager preflight",
            "memory": "not fixed before manager preflight",
            "tmp": "not fixed before manager preflight",
            "wall_time": "not fixed before manager preflight",
            "gpu": False,
            "gpu_backend": "none",
        },
        "excluded_from_direct_term_total": [
            "ACQ.load_native_shard eager native-shard materialization",
            "SH-channel arithmetic outside direct-kernel support-point/frequency terms",
            "artifact I/O and PNG/NPZ serialization",
        ],
        "archive_loader_materialization_included": False,
    }
    if solver == "cgls":
        ledger.update(
            {
                "solver": "cgls",
                "cgls_max_iterations": updates,
                "cgls_iterations": int(cgls_iterations),
                "cgls_passes_per_iteration": 3,
                "cgls_pass_definition": "one model.forward(train,p) plus one model.loss_and_gradient(train,c) with two streamed direct passes",
                "cgls_initial_normal_residual_passes": 2,
                "cgls_initial_normal_residual_terms": int(initial_normal_residual_terms),
                "cgls_iteration_terms": int(update_terms),
                "cgls_failed_search_forward_terms": int(failed_search_terms),
                "cgls_attempted_search_forward_count": int(
                    cgls_iterations
                    + (1 if failed_search_terms else 0)
                ),
                "fixed_diagnostic_passes_per_split": 2,
                "cgls_diagnostic_evaluation_count": len(validation_steps),
                "cgls_diagnostic_steps": [int(value) for value in validation_steps],
                "selected_final_forward_passes": 1,
                "postfit_report_forward_passes": 1,
                "cgls_selected_final_and_report_terms": int(
                    final_prediction_terms + report_metric_terms
                ),
                "cgls_breakdown_search_forward_included": bool(failed_search_terms),
                "cgls_term_accounting": "initial g=2*F_train; each CGLS iteration=3*F_train; diagnostics/final/report forwards are added separately",
            }
        )
        ledger["total_direct_kernel_sample_terms"] = int(
            initial_normal_residual_terms
            + update_terms
            + failed_search_terms
            + diagnostic_terms
            + final_prediction_terms
            + report_metric_terms
        )
    return ledger


def _fit_fixed_schedule_impl(
    model: NativeComplexDirectionalSHROIModel,
    train_records: Sequence[Any],
    validation_records: Sequence[Any] | None,
    *,
    config: FitConfig | None = None,
    output_dir: str | Path | None = None,
    full_aperture_gradient: bool = False,
    run_provenance: Mapping[str, Any] | None = None,
) -> FitResult:
    """Run the deterministic fixed-support local control schedule."""

    config = FitConfig() if config is None else config
    train = validate_record_set(train_records, split="train", target_id=model.target_id)
    validation = (
        tuple()
        if validation_records is None
        else validate_record_set(validation_records, split="validation", target_id=model.target_id)
    )
    if full_aperture_gradient:
        if str(config.control_kind) != "actual_full_aperture" or str(config.initialization) != "zero":
            raise ValueError("full-aperture fitting requires actual_full_aperture config with zero initialization")
        coefficients = model.initial_coefficients()
    else:
        coefficients = model.adjoint_seed(train)
    history: list[dict[str, Any]] = []
    checkpoint_paths: dict[str, str] = {}
    batches = tuple(
        train[start : start + int(config.microbatch_size)]
        for start in range(0, len(train), int(config.microbatch_size))
    )
    fixed_steps = tuple(sorted(set(int(v) for v in config.validation_steps)))
    gradient_record_scope = (
        "all_selected_train_records" if full_aperture_gradient else "cyclic_fake_control_only"
    )

    def evaluate(step: int, *, checkpoint_name: str | None = None) -> None:
        train_loss, _, train_prediction = model.loss_and_gradient(train, coefficients)
        if validation:
            validation_loss, _, validation_prediction = model.loss_and_gradient(validation, coefficients)
        else:
            validation_loss, validation_prediction = None, None
        history.append(
            {
                "step": int(step),
                "train_loss": float(train_loss),
                "validation_loss": None if validation_loss is None else float(validation_loss),
                "validation_use": "diagnostic_only_fixed_point",
                "selected_checkpoint": False,
            }
        )
        if output_dir is not None and checkpoint_name is not None:
            train_target = RaggedComplexValues(tuple(record.identity for record in train), tuple(_record_response(record) for record in train))
            train_residual = RaggedComplexValues(train_prediction.ids, tuple(p - y for p, y in zip(train_prediction.values, train_target.values)))
            latest_paths = save_checkpoint_bundle(
                output_dir,
                model=model,
                checkpoint_name=checkpoint_name,
                step=step,
                coefficients=coefficients,
                history=history,
                predictions=train_prediction,
                residuals=train_residual,
                train_records=train,
                validation_records=validation,
                fit_config=config,
                artifacts_generated_after_checkpoint_reload=False,
                gradient_record_scope=gradient_record_scope,
                run_provenance=run_provenance,
            )
            checkpoint_paths.update(
                {f"{checkpoint_name}_{key}": value for key, value in latest_paths.items()}
            )
            if validation:
                validation_target = RaggedComplexValues(tuple(record.identity for record in validation), tuple(_record_response(record) for record in validation))
                validation_residual = RaggedComplexValues(
                    validation_prediction.ids,
                    tuple(p - y for p, y in zip(validation_prediction.values, validation_target.values)),
                )
                checkpoint_paths.update(
                    {f"{checkpoint_name}_validation_{key}": value for key, value in save_prediction_artifacts(
                        output_dir,
                        model=model,
                        artifact_name=f"{checkpoint_name}_validation",
                        source_checkpoint=latest_paths["checkpoint"],
                        step=step,
                        coefficients=coefficients,
                        history=history,
                        predictions=validation_prediction,
                        residuals=validation_residual,
                        train_records=train,
                        validation_records=validation,
                        fit_config=config,
                        artifacts_generated_after_checkpoint_reload=False,
                        gradient_record_scope=gradient_record_scope,
                        run_provenance=run_provenance,
                    ).items()}
                )

    if 0 in fixed_steps:
        evaluate(0, checkpoint_name="latest" if 0 in config.checkpoint_steps else None)
    for step in range(1, int(config.updates) + 1):
        # Fake control keeps its explicitly declared cyclic microbatch
        # schedule.  Actual archive fitting calls this implementation with
        # full_aperture_gradient=True, so chunks are memory scheduling only and
        # every update sees every frozen TRAIN record.
        batch = train if full_aperture_gradient else batches[(step - 1) % len(batches)]
        _, gradient, _ = model.loss_and_gradient(batch, coefficients)
        coefficients = np.asarray(coefficients - float(config.step_size) * gradient, dtype=np.complex128)
        if step in fixed_steps:
            checkpoint_name = "latest" if step in config.checkpoint_steps else None
            evaluate(step, checkpoint_name=checkpoint_name)

    coefficients_final = np.array(coefficients, copy=True)
    if output_dir is not None:
        # Final update-12 coefficients are the sole selected state.  Validation
        # values above remain diagnostic-only and can never select/relabel a
        # checkpoint.
        final_state_paths = save_checkpoint_state(
            output_dir,
            model=model,
            checkpoint_name="final",
            step=config.updates,
            coefficients=coefficients_final,
            history=history,
            prediction_ids=tuple(record.identity for record in train),
            train_records=train,
            validation_records=validation,
            fit_config=config,
            artifacts_generated_after_checkpoint_reload=True,
            gradient_record_scope=gradient_record_scope,
            run_provenance=run_provenance,
        )
        checkpoint_paths.update({f"final_{key}": value for key, value in final_state_paths.items()})
        restored_final = load_checkpoint(final_state_paths["checkpoint"])
        if restored_final["step"] != int(config.updates):
            raise AssertionError("final checkpoint reload returned the wrong fixed update")
        coefficients_final = model._validate_coefficients(restored_final["coefficients"])
        final_prediction = model.forward(train, coefficients_final)
        train_target = RaggedComplexValues(tuple(record.identity for record in train), tuple(_record_response(record) for record in train))
        final_residual = RaggedComplexValues(final_prediction.ids, tuple(p - y for p, y in zip(final_prediction.values, train_target.values)))
        checkpoint_paths.update(
            {f"final_{key}": value for key, value in save_prediction_artifacts(
                output_dir,
                model=model,
                artifact_name="final",
                source_checkpoint=final_state_paths["checkpoint"],
                step=config.updates,
                coefficients=coefficients_final,
                history=history,
                predictions=final_prediction,
                residuals=final_residual,
                train_records=train,
                validation_records=validation,
                fit_config=config,
                artifacts_generated_after_checkpoint_reload=True,
                gradient_record_scope=gradient_record_scope,
                run_provenance=run_provenance,
            ).items()}
        )
        if validation:
            final_validation_prediction = model.forward(validation, coefficients_final)
            validation_target = RaggedComplexValues(tuple(record.identity for record in validation), tuple(_record_response(record) for record in validation))
            final_validation_residual = RaggedComplexValues(
                final_validation_prediction.ids,
                tuple(p - y for p, y in zip(final_validation_prediction.values, validation_target.values)),
            )
            checkpoint_paths.update(
                {f"final_validation_{key}": value for key, value in save_prediction_artifacts(
                    output_dir,
                    model=model,
                    artifact_name="final_validation",
                    source_checkpoint=final_state_paths["checkpoint"],
                    step=config.updates,
                    coefficients=coefficients_final,
                    history=history,
                    predictions=final_validation_prediction,
                    residuals=final_validation_residual,
                    train_records=train,
                    validation_records=validation,
                    fit_config=config,
                    artifacts_generated_after_checkpoint_reload=True,
                    gradient_record_scope=gradient_record_scope,
                    run_provenance=run_provenance,
                ).items()}
            )
    return FitResult(
        target_id=model.target_id,
        coefficients_latest=np.asarray(coefficients, dtype=np.complex128),
        coefficients_selected=np.asarray(coefficients_final, dtype=np.complex128),
        coefficients_final=np.asarray(coefficients_final, dtype=np.complex128),
        history=tuple(history),
        checkpoint_steps=tuple(config.checkpoint_steps),
        validation_steps=tuple(config.validation_steps),
        checkpoint_paths=checkpoint_paths,
    )


def fit_fixed_schedule(
    model: NativeComplexDirectionalSHROIModel,
    train_records: Sequence[Any],
    validation_records: Sequence[Any] | None,
    *,
    config: FitConfig | None = None,
    output_dir: str | Path | None = None,
    run_provenance: Mapping[str, Any] | None = None,
) -> FitResult:
    """Run the deterministic local fake-control cyclic microbatch schedule."""

    return _fit_fixed_schedule_impl(
        model,
        train_records,
        validation_records,
        config=config,
        output_dir=output_dir,
        full_aperture_gradient=False,
        run_provenance=run_provenance,
    )


def _fit_cgls_schedule_impl(
    model: NativeComplexDirectionalSHROIModel,
    train_records: Sequence[Any],
    validation_records: Sequence[Any] | None,
    *,
    config: FitConfig,
    output_dir: str | Path | None = None,
    run_provenance: Mapping[str, Any] | None = None,
) -> FitResult:
    """Run the bounded matrix-free complex CGLS schedule for actual mode."""

    train = validate_record_set(train_records, split="train", target_id=model.target_id)
    validation = (
        tuple()
        if validation_records is None
        else validate_record_set(validation_records, split="validation", target_id=model.target_id)
    )
    if str(config.solver).lower() != "cgls" or str(config.control_kind) != "actual_full_aperture":
        raise ValueError("CGLS requires solver=cgls and control_kind=actual_full_aperture")
    if str(config.initialization) != "zero":
        raise ValueError("CGLS requires zero initialization")

    coefficients = model.initial_coefficients()
    _, gradient, _ = model.loss_and_gradient(train, coefficients)
    search = np.asarray(-gradient, dtype=np.complex128)
    direction = np.array(search, copy=True)
    gamma = float(np.vdot(search, search).real)
    if not np.isfinite(gamma) or gamma < 0.0:
        raise FloatingPointError("CGLS normal-residual gamma is non-finite or negative")

    history: list[dict[str, Any]] = []
    checkpoint_paths: dict[str, str] = {}
    fixed_steps = tuple(sorted(set(int(value) for value in config.checkpoint_steps)))
    fixed_step_set = set(fixed_steps)
    evaluated_steps: set[int] = set()
    active_provenance = dict(DEFAULT_FAKE_PROVENANCE if run_provenance is None else run_provenance)
    active_provenance.update(
        {
            "solver": "cgls",
            "max_iterations": int(config.updates),
        }
    )
    gradient_record_scope = "all_selected_train_records"

    def evaluate(step: int, *, checkpoint_name: str | None = None) -> None:
        train_loss, _, train_prediction = model.loss_and_gradient(train, coefficients)
        if validation:
            validation_loss, _, validation_prediction = model.loss_and_gradient(validation, coefficients)
        else:
            validation_loss, validation_prediction = None, None
        history.append(
            {
                "step": int(step),
                "train_loss": float(train_loss),
                "validation_loss": None if validation_loss is None else float(validation_loss),
                "validation_use": "diagnostic_only_fixed_point",
                "selected_checkpoint": False,
                "solver": "cgls",
                "cgls_iteration": int(step),
            }
        )
        evaluated_steps.add(int(step))
        if output_dir is not None and checkpoint_name is not None:
            train_target = RaggedComplexValues(
                tuple(record.identity for record in train),
                tuple(_record_response(record) for record in train),
            )
            train_residual = RaggedComplexValues(
                train_prediction.ids,
                tuple(p - y for p, y in zip(train_prediction.values, train_target.values)),
            )
            latest_paths = save_checkpoint_bundle(
                output_dir,
                model=model,
                checkpoint_name=checkpoint_name,
                step=step,
                coefficients=coefficients,
                history=history,
                predictions=train_prediction,
                residuals=train_residual,
                train_records=train,
                validation_records=validation,
                fit_config=config,
                artifacts_generated_after_checkpoint_reload=False,
                gradient_record_scope=gradient_record_scope,
                run_provenance=active_provenance,
            )
            checkpoint_paths.update({f"{checkpoint_name}_{key}": value for key, value in latest_paths.items()})
            if validation:
                validation_target = RaggedComplexValues(
                    tuple(record.identity for record in validation),
                    tuple(_record_response(record) for record in validation),
                )
                validation_residual = RaggedComplexValues(
                    validation_prediction.ids,
                    tuple(p - y for p, y in zip(validation_prediction.values, validation_target.values)),
                )
                validation_paths = save_prediction_artifacts(
                    output_dir,
                    model=model,
                    artifact_name=f"{checkpoint_name}_validation",
                    source_checkpoint=latest_paths["checkpoint"],
                    step=step,
                    coefficients=coefficients,
                    history=history,
                    predictions=validation_prediction,
                    residuals=validation_residual,
                    train_records=train,
                    validation_records=validation,
                    fit_config=config,
                    artifacts_generated_after_checkpoint_reload=False,
                    gradient_record_scope=gradient_record_scope,
                    run_provenance=active_provenance,
                )
                checkpoint_paths.update(
                    {f"{checkpoint_name}_validation_{key}": value for key, value in validation_paths.items()}
                )

    def mark_terminal(reason: str, step: int, *, diagnostic_step_pending: bool = False) -> None:
        diagnostic_steps = set(evaluated_steps)
        if diagnostic_step_pending:
            diagnostic_steps.add(int(step))
        active_provenance.update(
            {
                "executed_iterations": int(step),
                "selected_final_step": int(step),
                "termination_reason": str(reason),
                "diagnostic_steps": sorted(diagnostic_steps),
            }
        )

    def refresh_latest_metadata() -> None:
        """Refresh terminal provenance without adding a duplicate numerical pass."""

        if output_dir is None:
            return
        for name in ("checkpoint_latest.json", "artifacts_latest.json", "artifacts_latest_validation.json"):
            path = Path(output_dir) / name
            if not path.exists():
                continue
            metadata = json.loads(path.read_text(encoding="utf-8"))
            metadata["run_provenance"] = dict(active_provenance)
            schedule = metadata.setdefault("schedule", {})
            schedule.update(
                {
                    "solver": "cgls",
                    "max_iterations": int(config.updates),
                    "executed_iterations": int(active_provenance["executed_iterations"]),
                    "selected_final_step": int(active_provenance["selected_final_step"]),
                    "termination_reason": str(active_provenance["termination_reason"]),
                    "selected_model": f"cgls_terminal_step_{int(active_provenance['selected_final_step'])}_final_for_train_and_validation",
                    "persistence_points_only": list(active_provenance["diagnostic_steps"]),
                    "cgls_passes_per_iteration": 3,
                }
            )
            path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    termination_reason: str | None = None
    if gamma <= 0.0:
        termination_reason = "normal_equation_stationarity"
        mark_terminal(termination_reason, 0, diagnostic_step_pending=True)
    evaluate(0, checkpoint_name="latest" if 0 in fixed_step_set else None)
    completed_iterations = 0
    if termination_reason is None:
        for iteration in range(1, int(config.updates) + 1):
            q = model.forward(train, direction)
            denominator = float(sum(np.vdot(value, value).real for value in q.values)) / float(
                sum(value.size for value in q.values)
            )
            if not np.isfinite(denominator):
                raise FloatingPointError("CGLS search denominator is non-finite")
            if denominator <= 0.0:
                termination_reason = "nonpositive_search_denominator_breakdown"
                break
            alpha = float(gamma / denominator)
            if not np.isfinite(alpha):
                raise FloatingPointError("CGLS alpha is non-finite")
            coefficients = np.asarray(coefficients + alpha * direction, dtype=np.complex128)
            _, gradient_new, _ = model.loss_and_gradient(train, coefficients)
            search_new = np.asarray(-gradient_new, dtype=np.complex128)
            gamma_new = float(np.vdot(search_new, search_new).real)
            if not np.isfinite(gamma_new) or gamma_new < 0.0:
                raise FloatingPointError("CGLS refreshed gamma is non-finite or negative")
            completed_iterations = int(iteration)
            if gamma_new <= 0.0:
                termination_reason = "normal_equation_stationarity"
                mark_terminal(termination_reason, iteration, diagnostic_step_pending=True)
                evaluate(iteration, checkpoint_name="latest")
                break
            if iteration == int(config.updates):
                termination_reason = "max_iterations_reached"
                mark_terminal(termination_reason, iteration, diagnostic_step_pending=iteration in fixed_step_set)
                if iteration in fixed_step_set:
                    evaluate(iteration, checkpoint_name="latest")
                break
            if iteration in fixed_step_set:
                evaluate(iteration, checkpoint_name="latest")
            beta = float(gamma_new / gamma)
            if not np.isfinite(beta):
                raise FloatingPointError("CGLS beta is non-finite")
            direction = np.asarray(search_new + beta * direction, dtype=np.complex128)
            search = search_new
            gamma = gamma_new
    if termination_reason is None:  # pragma: no cover - defensive invariant.
        raise AssertionError("CGLS termination reason was not resolved")
    if completed_iterations not in evaluated_steps:
        mark_terminal(termination_reason, completed_iterations, diagnostic_step_pending=True)
        evaluate(completed_iterations, checkpoint_name="latest")
    else:
        mark_terminal(termination_reason, completed_iterations)
        refresh_latest_metadata()

    selected_final_step = int(completed_iterations)
    coefficients_final = np.array(coefficients, copy=True)
    if output_dir is not None:
        final_state_paths = save_checkpoint_state(
            output_dir,
            model=model,
            checkpoint_name="final",
            step=selected_final_step,
            coefficients=coefficients_final,
            history=history,
            prediction_ids=tuple(record.identity for record in train),
            train_records=train,
            validation_records=validation,
            fit_config=config,
            artifacts_generated_after_checkpoint_reload=True,
            gradient_record_scope=gradient_record_scope,
            run_provenance=active_provenance,
        )
        checkpoint_paths.update({f"final_{key}": value for key, value in final_state_paths.items()})
        restored_final = load_checkpoint(final_state_paths["checkpoint"])
        if restored_final["step"] != selected_final_step:
            raise AssertionError("CGLS final checkpoint reload returned the wrong selected step")
        coefficients_final = model._validate_coefficients(restored_final["coefficients"])
        final_prediction = model.forward(train, coefficients_final)
        train_target = RaggedComplexValues(
            tuple(record.identity for record in train),
            tuple(_record_response(record) for record in train),
        )
        final_residual = RaggedComplexValues(
            final_prediction.ids,
            tuple(p - y for p, y in zip(final_prediction.values, train_target.values)),
        )
        final_paths = save_prediction_artifacts(
            output_dir,
            model=model,
            artifact_name="final",
            source_checkpoint=final_state_paths["checkpoint"],
            step=selected_final_step,
            coefficients=coefficients_final,
            history=history,
            predictions=final_prediction,
            residuals=final_residual,
            train_records=train,
            validation_records=validation,
            fit_config=config,
            artifacts_generated_after_checkpoint_reload=True,
            gradient_record_scope=gradient_record_scope,
            run_provenance=active_provenance,
        )
        checkpoint_paths.update({f"final_{key}": value for key, value in final_paths.items()})
        if validation:
            final_validation_prediction = model.forward(validation, coefficients_final)
            validation_target = RaggedComplexValues(
                tuple(record.identity for record in validation),
                tuple(_record_response(record) for record in validation),
            )
            final_validation_residual = RaggedComplexValues(
                final_validation_prediction.ids,
                tuple(p - y for p, y in zip(final_validation_prediction.values, validation_target.values)),
            )
            final_validation_paths = save_prediction_artifacts(
                output_dir,
                model=model,
                artifact_name="final_validation",
                source_checkpoint=final_state_paths["checkpoint"],
                step=selected_final_step,
                coefficients=coefficients_final,
                history=history,
                predictions=final_validation_prediction,
                residuals=final_validation_residual,
                train_records=train,
                validation_records=validation,
                fit_config=config,
                artifacts_generated_after_checkpoint_reload=True,
                gradient_record_scope=gradient_record_scope,
                run_provenance=active_provenance,
            )
            checkpoint_paths.update(
                {f"final_validation_{key}": value for key, value in final_validation_paths.items()}
            )

    return FitResult(
        target_id=model.target_id,
        coefficients_latest=np.asarray(coefficients, dtype=np.complex128),
        coefficients_selected=np.asarray(coefficients_final, dtype=np.complex128),
        coefficients_final=np.asarray(coefficients_final, dtype=np.complex128),
        history=tuple(history),
        checkpoint_steps=tuple(config.checkpoint_steps),
        validation_steps=tuple(config.validation_steps),
        checkpoint_paths=checkpoint_paths,
        solver="cgls",
        termination_reason=termination_reason,
        max_iterations=int(config.updates),
        executed_iterations=selected_final_step,
        selected_final_step=selected_final_step,
        diagnostic_steps=tuple(sorted(evaluated_steps)),
    )


def fit_full_aperture_schedule(
    model: NativeComplexDirectionalSHROIModel,
    train_records: Sequence[Any],
    validation_records: Sequence[Any] | None,
    *,
    config: FitConfig | None = None,
    output_dir: str | Path | None = None,
    run_provenance: Mapping[str, Any] | None = None,
    solver: str | None = None,
) -> FitResult:
    """Fit actual data with the selected normalized GD or matrix-free CGLS solver."""

    if config is None:
        config = actual_fit_config(model, solver="gd" if solver is None else solver)
    elif solver is not None and str(config.solver).lower() != str(solver).lower():
        raise ValueError("solver argument does not match FitConfig.solver")
    if str(config.control_kind) != "actual_full_aperture":
        raise ValueError("actual fitting requires control_kind=actual_full_aperture")
    if str(config.solver).lower() == "cgls":
        return _fit_cgls_schedule_impl(
            model,
            train_records,
            validation_records,
            config=config,
            output_dir=output_dir,
            run_provenance=run_provenance,
        )
    return _fit_fixed_schedule_impl(
        model,
        train_records,
        validation_records,
        config=config,
        output_dir=output_dir,
        full_aperture_gradient=True,
        run_provenance=run_provenance,
    )


def make_fake_source_af_records(
    *,
    role: str,
    count: int,
    frequency_count: int = 8,
) -> tuple[InMemorySourceAFRecord, ...]:
    """Make sorted P1/HH/sector-002 train or sector-001 validation records."""

    role = str(role).lower()
    if role not in {"train", "validation"}:
        raise ValueError("fake source-AF role must be train or validation")
    if int(count) <= 0 or int(frequency_count) <= 0:
        raise ValueError("fake record and frequency counts must be positive")
    sector = TRAIN_SECTOR_ID if role == "train" else VALIDATION_SECTOR_ID
    records = []
    frequencies = 9.1e9 + np.arange(int(frequency_count), dtype=np.float64) * 7.0e6
    for index in range(int(count)):
        position = np.asarray(
            [34.0 + 0.37 * index, -22.0 + 0.23 * index, 6.0 + 0.11 * index], dtype=np.float64
        )
        raw = (0.2 + 0.1j) * np.exp(1j * np.arange(frequencies.size) * (0.11 + 0.01 * index))
        correction_r = 0.001 + 0.0001 * index
        correction_ph = 0.03 + 0.002 * index
        effective_r0, effective_response = source_af_values(
            float(np.linalg.norm(position)), raw, correction_r, correction_ph
        )
        records.append(
            InMemorySourceAFRecord(
                identity=NativeIdentity(1, "hh", sector, index),
                role=role,
                position_xyz_m=position,
                frequencies_hz=frequencies,
                r0_raw_m=float(np.linalg.norm(position)),
                response_raw=raw,
                r_correct_raw_m=correction_r,
                ph_correct_raw_rad=correction_ph,
                effective_r0_m=effective_r0,
                effective_response=effective_response,
                provenance={"source": "in_memory_fake_only", "archive_opened": False, "test_opened": False},
            )
        )
    return tuple(records)


def make_fake_tophat_source_af_records(
    *,
    role: str = "train",
    records_per_panel: int = 1,
    frequency_count: int = 8,
) -> tuple[InMemorySourceAFRecord, ...]:
    """Make independent fake records for the eight P1/P7 HH panels."""

    role = str(role).lower()
    if role not in {"train", "validation"}:
        raise ValueError("TopHat fake role must be train or validation")
    if int(records_per_panel) <= 0 or int(frequency_count) <= 0:
        raise ValueError("records_per_panel and frequency_count must be positive")
    frequencies = 9.2e9 + np.arange(int(frequency_count), dtype=np.float64) * 5.0e6
    sector_ids = TOPHAT_TRAIN_SECTOR_IDS if role == "train" else TOPHAT_VALIDATION_SECTOR_IDS
    records = []
    pulse = 0
    for pass_id in TOPHAT_TRAIN_PASS_IDS:
        for sector_id in sector_ids:
            for panel_index in range(int(records_per_panel)):
                position = np.asarray(
                    [34.0 + 0.18 * pulse, -21.0 + 0.13 * pulse, 5.0 + 0.07 * panel_index],
                    dtype=np.float64,
                )
                raw = (0.16 + 0.08j) * np.exp(1j * np.arange(frequencies.size) * (0.09 + 0.002 * pulse))
                correction_r = 0.0008 + 0.00001 * pulse
                correction_ph = 0.02 + 0.0007 * pulse
                effective_r0, effective_response = source_af_values(
                    float(np.linalg.norm(position)), raw, correction_r, correction_ph
                )
                records.append(
                    InMemorySourceAFRecord(
                        identity=NativeIdentity(pass_id, "hh", sector_id, panel_index),
                        role=role,
                        position_xyz_m=position,
                        frequencies_hz=frequencies,
                        r0_raw_m=float(np.linalg.norm(position)),
                        response_raw=raw,
                        r_correct_raw_m=correction_r,
                        ph_correct_raw_rad=correction_ph,
                        effective_r0_m=effective_r0,
                        effective_response=effective_response,
                        provenance={
                            "source": "in_memory_fake_only",
                            "panel": f"P{pass_id}/HH/{sector_id:03d}",
                            "archive_opened": False,
                            "test_opened": False,
                        },
                    )
                )
                pulse += 1
    return tuple(records)


def make_fake_tophat_control_case(
    *,
    records_per_panel: int = 1,
    frequency_count: int = 8,
) -> tuple[NativeComplexDirectionalSHROIModel, tuple[InMemorySourceAFRecord, ...], np.ndarray]:
    """Return a provisional-placement fake TopHat TRAIN-response control."""

    model = NativeComplexDirectionalSHROIModel(
        "tophat", placement=TOPHAT_PROVISIONAL_PLACEMENT
    )
    train_base = make_fake_tophat_source_af_records(
        records_per_panel=records_per_panel, frequency_count=frequency_count
    )
    planted = model.initial_coefficients()
    planted[77, 0] = 0.5 + 0.07j
    planted[321, 1] = -0.22 + 0.09j
    planted[444, 3] = 0.18 - 0.05j
    target = model.forward(train_base, planted)
    train = tuple(record.with_effective_response(value) for record, value in zip(train_base, target.values))
    return model, train, planted


def make_fake_camry_control_case(
    *,
    train_count: int = 4,
    validation_count: int = 2,
    frequency_count: int = 8,
) -> tuple[NativeComplexDirectionalSHROIModel, tuple[InMemorySourceAFRecord, ...], tuple[InMemorySourceAFRecord, ...], np.ndarray]:
    """Return a deterministic fake Camry case with a planted complex field."""

    model = NativeComplexDirectionalSHROIModel("toyota_camry")
    train_base = make_fake_source_af_records(role="train", count=train_count, frequency_count=frequency_count)
    validation_base = make_fake_source_af_records(role="validation", count=validation_count, frequency_count=frequency_count)
    planted = model.initial_coefficients()
    planted[123, 0] = 0.8 + 0.15j
    planted[456, 1] = -0.35 + 0.2j
    planted[789, 3] = 0.25 - 0.1j
    train_target = model.forward(train_base, planted)
    validation_target = model.forward(validation_base, planted)
    train = tuple(record.with_effective_response(value) for record, value in zip(train_base, train_target.values))
    validation = tuple(record.with_effective_response(value) for record, value in zip(validation_base, validation_target.values))
    return model, train, validation, planted


def protocol_payload() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "target_ids": ["toyota_camry", "tophat"],
        "target_id": "toyota_camry",
        "train_selector": "P1/HH/sector002/TRAIN, runtime-preflighted IDs (Camry)",
        "validation_selector": "P1/HH/sector001/validation, held out after architecture/support/updates are frozen (Camry)",
        "tophat_train_selector": "P1/P7 HH canonical TRAIN sectors 002/092/182/272, independent ragged panels",
        "tophat_validation_selector": "P1/P7 HH validation sectors 001/091/181/271, diagnostic-only fixed selector",
        "tophat_panel_wise_image_pooling": False,
        "actual_mode": {
            "requires_archive_root": True,
            "archive_root_contract": "archive parent or exact converted_v3_joint8_fullpol/shards directory",
            "requires_fresh_output_dir": True,
            "fake_fallback": False,
            "solver_default": "gd",
            "solver_choices": ["gd", "cgls"],
            "cgls_max_iterations": MAX_CGLS_ITERATIONS,
            "cgls_diagnostic_steps": list(CGLS_DIAGNOSTIC_STEPS),
            "cgls_termination_reasons": [
                "normal_equation_stationarity",
                "nonpositive_search_denominator_breakdown",
                "max_iterations_reached",
            ],
            "record_microbatch_argument": "accepted for interface compatibility; ignored in actual full-aperture mode",
            "actual_array_memory_chunks": "spatial_chunk_size=128 and frequency_chunk_size=64",
            "range_comparison_mode": "actual_range_subspace_comparison",
            "range_comparison_requires_saved_native_root": True,
            "camry_loader_sequence": "ACQ.load_native_shard -> header/role/identity seal -> shard.observations -> SOURCE_AF.build_source_af(train then validation)",
            "tophat_loader_sequence": "ACQ.load_native_shard(P1/P7) -> LOC.prepare_tophat_measured_response_screen_dataset(train once) -> scoped validation conversion",
            "validation_selector_must_be_explicit": True,
            "test_payload_opened": False,
        },
        "count_contracts": {
            "toyota_camry": {
                "train_record_count": 117,
                "train_frequency_count_per_record": 424,
                "train_frequency_sample_count": 49_608,
                "validation_record_count": 117,
                "validation_frequency_count_per_record": 424,
                "validation_frequency_sample_count": 49_608,
            },
            "tophat": {
                "train_record_count": 949,
                "train_frequency_sample_count": 407_176,
                "validation_record_count": 947,
                "validation_frequency_sample_count": 406_308,
                "frequency_counts_by_pass": {"1": 424, "7": 434},
                "train_record_counts_by_panel": {
                    "P1/HH/002": 117, "P1/HH/092": 117, "P1/HH/182": 117, "P1/HH/272": 118,
                    "P7/HH/002": 120, "P7/HH/092": 120, "P7/HH/182": 120, "P7/HH/272": 120,
                },
                "validation_record_counts_by_panel": {
                    "P1/HH/001": 117, "P1/HH/091": 117, "P1/HH/181": 118, "P1/HH/271": 117,
                    "P7/HH/001": 119, "P7/HH/091": 119, "P7/HH/181": 120, "P7/HH/271": 120,
                },
            },
        },
        "tophat_placement": {
            "status": "screen_consistent_provisional_operational_roi",
            "equation": "p_native = I @ p_local + t",
            "R": TOPHAT_PROVISIONAL_PLACEMENT.rotation.tolist(),
            "t_m": TOPHAT_PROVISIONAL_PLACEMENT.translation_m.tolist(),
            "not_a_passed_localizer_survey_registration_physical_containment_or_height_result": True,
        },
        "placements": {
            "toyota_camry": placement_contract(CAMRY_PLACEMENT, target_id="toyota_camry"),
            "tophat": placement_contract(TOPHAT_PROVISIONAL_PLACEMENT, target_id="tophat"),
        },
        "test_payload_opened": False,
        "actual_archive_opened": False,
        "pace_action": False,
        "manager_touched": False,
        "deployment": "none",
        "new_job_launch": False,
        "source_af_formula": "r0_src=float64(r0_raw)+float64(r_correct_raw); response_src=complex128(response_raw)*exp(+i*float64(ph_correct_raw))",
        "global_complex_gain": "1+0j fixed; no per-record gains",
        "field_model": {
            "basis": "real degree-1 SH [Y00,Y1x,Y1y,Y1z]",
            "coefficients": "four real basis coefficients per voxel, complex128 real/imag field",
            "all_voxels_active": True,
            "adaptive_prune_grow_support": False,
            "spatial_chunk_size": DEFAULT_SPATIAL_CHUNK_SIZE,
            "frequency_chunk_size": DEFAULT_FREQUENCY_CHUNK_SIZE,
            "coherent_spatial_sum_before_residual": True,
        },
        "supports": {
            target: {
                "lower_edge_m": list(spec.lower_edge_m),
                "upper_edge_exclusive_m": list(spec.upper_edge_exclusive_m),
                "cell_size_m": spec.cell_size_m,
                "shape": list(spec.shape),
                "point_count": spec.point_count,
            }
            for target, spec in SUPPORTS.items()
        },
        "readout": {target: {"shape": list(grid.shape), "spacing_m": grid.spacing_m, "label": READOUT_LABEL} for target, grid in READOUT_GRIDS.items()},
        "schedule": {
            "minimum_actual_optimizer_updates": DEFAULT_UPDATES,
            "deterministic_adjoint_initial_seed": True,
            "fake_control_schedule": "fixed cyclic record microbatches",
            "actual_optimizer_schedule": "full-aperture gradient over all selected TRAIN records; record microbatch argument is ignored; spatial/frequency chunks are array-memory scheduling only",
            "fixed_checkpoint_and_validation_points": list(DEFAULT_CHECKPOINT_STEPS),
            "validation_use": "diagnostic_only_fixed_points",
            "selected_model": "update_12_final_for_train_and_validation",
            "persistence_points_only": list(DEFAULT_CHECKPOINT_STEPS),
            "atomic_per_update": False,
            "resumable_per_update": False,
            "validation_support_selection_or_early_stop": False,
        },
        "cgls_contract": {
            "available_only_in_actual_mode": True,
            "actual_optimizer_schedule": "matrix-free complex CGLS over all selected TRAIN records; zero initialization; validation is diagnostic-only",
            "initialization": "zero",
            "train_records_only": True,
            "normalization": "model.loss_and_gradient returns A^H(Ac-y)/N; alpha uses vdot(s,s).real / (vdot(q,q).real/N)",
            "no_extra_factor_two": True,
            "no_denominator_floor": True,
            "stationarity_test": "gamma == 0 indicates normal-equation stationarity; this is not a zero-residual test",
            "per_iteration_passes": "one model.forward(train,p) plus one model.loss_and_gradient(train,c), whose prediction and gradient are two streamed passes",
            "max_iterations": MAX_CGLS_ITERATIONS,
            "diagnostic_steps": list(CGLS_DIAGNOSTIC_STEPS),
        },
        "range_subspace_contract": {
            "available_only_in_actual_range_comparison": True,
            "mode": "actual_range_subspace_comparison",
            "frequency_grouping": "direct in-memory float64 array equality to a representative; no hashes",
            "point_response": {
                "kappa": "4*pi/c",
                "samples": RANGE_SUBSPACE_SAMPLES,
                "interval": "[0,2*rRayleigh]",
                "first_positive_strict_bracketed_local_minimum": True,
                "golden_section_iterations": RANGE_SUBSPACE_GOLDEN_ITERATIONS,
                "g_guard": "0.5*rRayleigh <= g <= 1.5*rRayleigh",
            },
            "native_geometry": {
                "frequencies": "stored native float64 vector exactly; no endpoint regrid",
                "r0": "effective source r0 only",
                "c_native": "placement(local origin)",
                "Rmin": "antenna distance to closed transformed local box",
                "Rmax": "maximum antenna distance to transformed cube corners",
            },
            "range_grid": {
                "guard": "fixed 2g only",
                "qminus": "Rmin-Rc-2g",
                "qplus": "Rmax-Rc+2g",
                "grid": "q_m=qminus+m*g",
                "M": "1+ceil((qplus-qminus)/g)",
                "require_M_lt_K": True,
                "require_full_rank": True,
                "condition_bound": 10.0,
            },
            "E": "K^-1/2 exp(-1j*kappa*nu[m]*m*g)",
            "Q_cache_key": "(frequency_group,M)",
            "B": "Q^H * exp(+1j*kappa*nu*(s_c+qminus)) * y",
            "BH": "exp(-1j*kappa*nu*(s_c+qminus)) * Qz",
            "dense_T_or_P_materialized": False,
            "h": "g exactly; g/2 rejected for scored mode",
            "N_T": "sum M over the split after rank(E)=M; distinct from native K",
            "range_loss": "||B(Ac-y)||^2/(2*N_T_train)",
            "range_gradient": "A^H B^H B(Ac-y)/N_T_train",
            "retention_guard": "fixed-2g sampled retention >= 0.95 on unguarded Rmin-Rc..Rmax-Rc at g/8 plus endpoints",
            "gauge_check": "centered-frequency and full-frequency gauges agree numerically",
            "nominal_ambiguity_diagnostic": {
                "period_definition": "U=c/(2*max_df)",
                "principal_cell": "[-U/2,U/2] containing q=0",
                "guarded_interval_must_be_inside_principal_cell": True,
                "same_delay_exterior_capture_is_nominal_only": True,
                "off_grid_and_exterior_leakage_are_disclosed_not_selector_tuned": True,
            },
            "arms": [
                "saved_native_objective_cgls24",
                "range_focused_isotropic_bp",
                "range_focused_degree1_cgls24",
            ],
            "saved_arm_fallback": "missing/incompatible sidecars require an independently compatible frozen checkpoint before a fresh forward; no hashes",
            "bp": "A0 retains Y00 and zeros channels 1:; TRAIN-only alpha=vdot(p,By)/vdot(p,p); no denominator floor",
            "metrics": [
                "raw_native_complex_relmse_scoped_diagnostic",
                "range_focused_complex_RelMSE",
                "B_domain_prediction_target_energy",
                "range_complex_correlation",
                "raw_energy_fraction_retained_by_B",
                "sampled_in_cube_retention",
                "exterior_leakage_curve",
            ],
        },
        "persistence": {
            "full_state": ["checkpoint_latest", "checkpoint_final"],
            "canonical_ragged_sidecars": ["predictions", "residuals"],
            "diagnostics": ["coefficient_energy", "projections", "slices", "exact_grid_energy_readout"],
            "reload_same_checkpoint_parity": True,
        },
        "actual_resource_policy": {
            "class": "CPU-only NumPy implementation; adequate headroom requires manager runtime preflight",
            "gpu_backend": False,
            "no_inherited_20_minute_mapper_limit": True,
        },
        "fit_release_status": "LOCAL_FAKE_TWO_TARGET_FIXED_SUPPORT_CONTROL_ONLY",
    }


__all__ = [
    "CAMRY_PLACEMENT",
    "CHECKPOINT_SCHEMA",
    "CGLS_DIAGNOSTIC_STEPS",
    "RANGE_SUBSPACE_GOLDEN_ITERATIONS",
    "RANGE_SUBSPACE_RETENTION_THRESHOLD",
    "RANGE_SUBSPACE_SAMPLES",
    "RANGE_SUBSPACE_SCHEMA",
    "RANGE_COMPARISON_SCHEMA",
    "DEFAULT_CHECKPOINT_STEPS",
    "DEFAULT_FREQUENCY_CHUNK_SIZE",
    "DEFAULT_SPATIAL_CHUNK_SIZE",
    "DEFAULT_UPDATES",
    "ExactReadoutGrid",
    "FitConfig",
    "FitResult",
    "GLOBAL_COMPLEX_GAIN",
    "MAX_CGLS_ITERATIONS",
    "InMemorySourceAFRecord",
    "NativeComplexDirectionalSHROIModel",
    "NativeIdentity",
    "NativePlacement",
    "RaggedComplexValues",
    "RangeFrequencyGroup",
    "RangeRecordGeometry",
    "RangeSubspacePlan",
    "READOUT_GRIDS",
    "READOUT_LABEL",
    "SCHEMA",
    "SOURCE_AF_REPRESENTATION",
    "SUPPORTS",
    "SupportSpec",
    "SH_BASIS_CONVENTION",
    "SOURCE_AF_FORMULA",
    "TOPHAT_PROVISIONAL_PLACEMENT",
    "TOPHAT_TRAIN_PASS_IDS",
    "TOPHAT_TRAIN_SECTOR_IDS",
    "TOPHAT_VALIDATION_SECTOR_IDS",
    "actual_fit_config",
    "conservative_full_aperture_step_size",
    "degree1_real_sh_basis",
    "direct_term_resource_ledger",
    "analyze_range_frequency_group",
    "build_range_subspace_plan",
    "exact_readout_grid",
    "fit_fixed_schedule",
    "fit_full_aperture_schedule",
    "l0_scalar_coefficients",
    "load_checkpoint",
    "make_fake_camry_control_case",
    "make_fake_source_af_records",
    "make_fake_tophat_control_case",
    "make_fake_tophat_source_af_records",
    "placement_contract",
    "protocol_payload",
    "render_fit_figures",
    "range_comparison_resource_ledger",
    "range_forward",
    "range_isotropic_bp",
    "range_loss_and_gradient",
    "range_metrics",
    "range_metrics_from_predictions",
    "fit_range_focused_cgls24",
    "residual_metrics",
    "save_checkpoint_bundle",
    "source_af_values",
    "support_for_target",
    "validate_record_set",
]
