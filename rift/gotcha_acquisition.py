"""Read-only GOTCHA acquisition contracts and small synthetic operator fixtures.

This module is deliberately additive to the historical GOTCHA planar code.  It
reads one Gate-1 pass/polarization NPZ at a time, preserves the native arrays,
and exposes observations by their native identity ``(sector_id, pulse_index)``.
It does not infer the meaning of the public ``fp``/``r0`` convention, apply
autofocus, open test payloads, or provide an accelerated renderer.

The direct point-target renderer is a deterministic correctness fixture.  Its
reference convention is explicit and synthetic: the phase path is
``||tx-point|| + ||rx-point|| - reference_range_m`` and the default monostatic
geometry sets ``tx == rx`` for the same observation.  This convention must not
be read as a conclusion about real GOTCHA phase metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np


# Keep Batch A NumPy-only.  The historical contract module imports the full
# RIFT package, whose eager model import is not needed for a sealed acquisition
# adapter and is unavailable in small CPU-only validation runtimes.  These
# constants mirror the frozen Gate-1 identity and are cross-checked by the
# focused tests when the legacy contract is importable.
PASS_IDS = tuple(range(1, 9))
POLARIZATIONS = ("hh", "hv", "vh", "vv")
CO_POLARIZATIONS = ("hh", "vv")
SECTOR_IDS = tuple(range(1, 361))


def canonical_shard_id(pass_id: int, polarization: str) -> str:
    pass_id = int(pass_id)
    polarization = str(polarization).lower()
    if pass_id not in PASS_IDS:
        raise ValueError(f"pass_id must be one of {PASS_IDS}")
    if polarization not in POLARIZATIONS:
        raise ValueError(f"polarization must be one of {POLARIZATIONS}")
    return f"pass{pass_id}_{polarization}"


def _role_for_sector(sector_id: int) -> str:
    sector_id = int(sector_id)
    if sector_id not in SECTOR_IDS:
        raise ValueError("sector_id must be in 1..360")
    slot = (sector_id - 1) % 10
    if slot == 0:
        return "validation"
    if slot == 5:
        return "test"
    return "train"


def _payload_sector_ids() -> tuple[int, ...]:
    return tuple(sector_id for sector_id in SECTOR_IDS if _role_for_sector(sector_id) != "test")


def _sealed_test_sector_ids() -> tuple[int, ...]:
    return tuple(sector_id for sector_id in SECTOR_IDS if _role_for_sector(sector_id) == "test")


SPEED_OF_LIGHT_M_S = 299_792_458.0
NATIVE_ARCHIVE_SCHEMA = "rift_gotcha_joint8_fullpol_native_shard_v1"
PHASE_REFERENCE_SCHEMA = "rift_gotcha_native_phase_reference_v1"
AUTOFOCUS_PROVENANCE_SCHEMA = "rift_gotcha_autofocus_provenance_v1"
SYNTHETIC_REFERENCE_SCHEMA = "rift_gotcha_synthetic_reference_v1"

AUTOFOCUS_RAW = "raw_channel_own_arrays_unapplied"
# Kept as a private compatibility spelling while the public name above is used
# in new code.  The value, unlike the identifier spelling, is part of Gate-1.
AUTOFOKUS_RAW = AUTOFOCUS_RAW
AUTOFOCUS_PUBLISHED = "published_response_derived"
AUTOFOCUS_TRAIN_NUISANCE = "train_only_nuisance"
AUTOFOCUS_OFFICIAL_ABSENT = "official_arrays_absent"

_ARCHIVE_KEYS = frozenset(
    {
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
)
_ROLE_NAMES = ("train", "validation", "test")
_PAYLOAD_ROLE_NAMES = ("train", "validation")


def _readonly_array(value: Any, *, dtype: np.dtype | str | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _scalar(value: Any, *, name: str) -> Any:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must be a scalar")
    return array.reshape(-1)[0].item()


def _scalar_text(value: Any, *, name: str) -> str:
    return str(_scalar(value, name=name))


def _scalar_bool(value: Any, *, name: str) -> bool:
    scalar = _scalar(value, name=name)
    if isinstance(scalar, (bool, np.bool_)):
        return bool(scalar)
    if isinstance(scalar, (int, np.integer)) and int(scalar) in (0, 1):
        return bool(scalar)
    raise ValueError(f"{name} must be a boolean scalar")


def _require_vector(arrays: Mapping[str, np.ndarray], name: str, count: int) -> np.ndarray:
    value = np.asarray(arrays[name])
    if value.shape != (count,):
        raise ValueError(f"{name} must be aligned to the view axis with shape ({count},)")
    return value


def _require_finite(value: np.ndarray, *, name: str) -> None:
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


@dataclass(frozen=True)
class NativeObservationId:
    """Stable identity for one native GOTCHA pulse observation."""

    pass_id: int
    polarization: str
    sector_id: int
    pulse_index: int

    def __post_init__(self) -> None:
        if int(self.pass_id) not in PASS_IDS:
            raise ValueError(f"pass_id must be one of {PASS_IDS}")
        polarization = str(self.polarization).lower()
        if polarization not in POLARIZATIONS:
            raise ValueError(f"polarization must be one of {POLARIZATIONS}")
        if int(self.sector_id) not in SECTOR_IDS:
            raise ValueError("sector_id must be in 1..360")
        object.__setattr__(self, "pass_id", int(self.pass_id))
        object.__setattr__(self, "polarization", polarization)
        object.__setattr__(self, "sector_id", int(self.sector_id))
        object.__setattr__(self, "pulse_index", int(self.pulse_index))

    @property
    def shard_id(self) -> str:
        return canonical_shard_id(self.pass_id, self.polarization)

    def as_dict(self) -> dict[str, Any]:
        return {
            "pass_id": self.pass_id,
            "polarization": self.polarization,
            "sector_id": self.sector_id,
            "pulse_index": self.pulse_index,
        }


@dataclass(frozen=True)
class PhaseReferenceContract:
    """Units and provenance for native phase metadata.

    ``r0_interpretation`` intentionally remains unresolved for real GOTCHA
    until a later calibration gate.  Keeping that uncertainty in the contract
    prevents the adapter from silently treating a source reference as the
    synthetic path reference used by the fixture renderer.
    """

    schema: str = PHASE_REFERENCE_SCHEMA
    frequency_unit: str = "Hz"
    position_unit: str = "m"
    range_unit: str = "m"
    angle_unit: str = "deg"
    phase_unit: str = "rad"
    frequency_values: str = "native_stored_exact"
    reference_range_field: str = "r0"
    r0_interpretation: str = "one_way_antenna_to_scene_center_range_per_packaged_metadata"
    autofocus_range_interpretation: str = "correction_for_r0_per_packaged_metadata"
    autofocus_phase_interpretation: str = "phase_correction_per_packaged_metadata"
    theta_label: str = "azimuth"
    phi_label: str = "elevation"
    theta_zero_direction: str = "+x"
    phi_zero_plane: str = "xy"
    correction_application_convention: str = "unverified_no_default"
    geometry_contract: str = "paired_monostatic_tx_equals_rx_same_observation"

    def __post_init__(self) -> None:
        if self.schema != PHASE_REFERENCE_SCHEMA:
            raise ValueError("unexpected phase-reference contract schema")
        if self.frequency_values != "native_stored_exact":
            raise ValueError("native frequency policy must preserve stored values")
        if self.reference_range_field != "r0":
            raise ValueError("native reference field must be r0")
        if self.correction_application_convention != "unverified_no_default":
            raise ValueError("real-data correction signs/order must remain unverified in Batch A")
        if self.geometry_contract != "paired_monostatic_tx_equals_rx_same_observation":
            raise ValueError("native geometry must be paired monostatic")


@dataclass(frozen=True)
class AutofocusProvenance:
    """Explicit correction provenance without silently applying a correction."""

    schema: str
    mode: str
    official_available: bool
    applied: bool
    source_shard_id: str
    range_field: str | None
    phase_field: str | None
    range_unit: str = "unknown"
    phase_unit: str = "unknown"

    def __post_init__(self) -> None:
        if self.schema != AUTOFOCUS_PROVENANCE_SCHEMA:
            raise ValueError("unexpected autofocus provenance schema")
        valid_modes = {
            AUTOFOKUS_RAW,
            AUTOFOCUS_PUBLISHED,
            AUTOFOCUS_TRAIN_NUISANCE,
            AUTOFOCUS_OFFICIAL_ABSENT,
        }
        if self.mode not in valid_modes:
            raise ValueError(f"unknown autofocus mode {self.mode!r}")
        if self.applied:
            raise ValueError(
                "a loaded native shard cannot claim applied autofocus; application needs a declared contract"
            )
        if self.official_available:
            if not self.range_field or not self.phase_field:
                raise ValueError("available autofocus must name both source fields")
        elif self.range_field is not None or self.phase_field is not None:
            raise ValueError("officially absent autofocus cannot name source fields")


@dataclass(frozen=True)
class AutofocusApplicationContract:
    """A separately validated recipe required before applying published AF."""

    mode: str = AUTOFOCUS_PUBLISHED
    validated: bool = False
    range_sign: int | None = None
    phase_sign: int | None = None
    order: tuple[str, ...] = ("range", "phase")
    range_unit: str = "unverified"
    phase_unit: str = "unverified"
    units_validated: bool = False
    order_status: str = "candidate_order_unverified"
    range_formula: str = "r0_after = r0_before + range_sign * r_correct_raw"
    phase_formula: str = (
        "response_after = response_before * exp(1j * phase_sign * ph_correct_raw)"
    )

    def validate(self) -> None:
        if self.mode != AUTOFOCUS_PUBLISHED:
            raise ValueError("only published-response-derived autofocus is applicable here")
        if not self.validated:
            raise ValueError("autofocus sign/order must be validated before application")
        if self.range_sign not in (-1, 1) or self.phase_sign not in (-1, 1):
            raise ValueError("autofocus range_sign and phase_sign must be +1 or -1")
        if set(self.order) != {"range", "phase"} or len(self.order) != 2:
            raise ValueError("autofocus order must name range and phase exactly once")
        if self.order_status != "candidate_order_unverified":
            raise ValueError("real-data autofocus order remains unverified in Batch A")
        if not self.units_validated or self.range_unit != "m" or self.phase_unit != "rad":
            raise ValueError(
                "autofocus application requires an explicit metre/radian candidate declaration"
            )


@dataclass(frozen=True)
class TrainNuisanceCalibrationContract:
    """Metadata-only contract for the later train-only nuisance fit."""

    mode: str = AUTOFOCUS_TRAIN_NUISANCE
    fit_role: str = "train"
    frozen: bool = False
    test_opened: bool = False
    gauge_reference_pass: int = 2
    range_offset_bound_m: float | None = None

    def validate(self) -> None:
        if self.mode != AUTOFOCUS_TRAIN_NUISANCE or self.fit_role != "train":
            raise ValueError("nuisance calibration must be explicitly train-only")
        if self.test_opened:
            raise ValueError("train-only nuisance calibration cannot open test data")
        if self.gauge_reference_pass != 2:
            raise ValueError("the declared nuisance gauge uses pass 2")
        if self.frozen and (
            self.range_offset_bound_m is None
            or not np.isfinite(float(self.range_offset_bound_m))
            or float(self.range_offset_bound_m) <= 0
        ):
            raise ValueError("a frozen nuisance fit must declare a positive range bound")


@dataclass(frozen=True)
class PairedMonostaticGeometry:
    """Tx/Rx positions paired one-to-one by native observation identity."""

    observation_ids: tuple[NativeObservationId, ...]
    tx_xyz_m: np.ndarray
    rx_xyz_m: np.ndarray

    def __post_init__(self) -> None:
        tx = np.asarray(self.tx_xyz_m)
        rx = np.asarray(self.rx_xyz_m)
        if tx.shape != (len(self.observation_ids), 3) or rx.shape != tx.shape:
            raise ValueError("paired Tx/Rx positions must have shape [observation, 3]")
        if not np.array_equal(tx, rx):
            raise ValueError("GOTCHA validation geometry requires Tx=Rx per observation")
        _require_finite(tx, name="paired positions")

    @property
    def count(self) -> int:
        return len(self.observation_ids)


@dataclass(frozen=True)
class HeightSupportCandidate:
    """Candidate xyz bounds for a later volume preflight.

    These are engineering inputs, not recovered GOTCHA bounds.  In particular,
    this object never treats the historical z=0 plane as a volume proof.
    """

    x_bounds_m: tuple[float, float]
    y_bounds_m: tuple[float, float]
    z_bounds_m: tuple[float, float]
    schema: str = "rift_gotcha_height_support_candidate_v1"
    status: str = "candidate_bounds_only_real_bounds_unresolved"

    def __post_init__(self) -> None:
        if self.schema != "rift_gotcha_height_support_candidate_v1":
            raise ValueError("unexpected height-support candidate schema")
        for name, bounds in (
            ("x_bounds_m", self.x_bounds_m),
            ("y_bounds_m", self.y_bounds_m),
            ("z_bounds_m", self.z_bounds_m),
        ):
            if len(bounds) != 2 or not np.isfinite(bounds).all() or not bounds[0] < bounds[1]:
                raise ValueError(f"{name} must be finite increasing bounds")
        if self.z_bounds_m[0] == self.z_bounds_m[1]:
            raise ValueError("height-capable support requires nonzero z extent")

    def corners_xyz_m(self) -> np.ndarray:
        corners = np.asarray(
            [
                [x, y, z]
                for x in self.x_bounds_m
                for y in self.y_bounds_m
                for z in self.z_bounds_m
            ],
            dtype=np.float64,
        )
        return _readonly_array(corners)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "x_bounds_m": list(self.x_bounds_m),
            "y_bounds_m": list(self.y_bounds_m),
            "z_bounds_m": list(self.z_bounds_m),
            "status": self.status,
        }


def preflight_height_support_candidate(
    candidate: HeightSupportCandidate,
    geometry: PairedMonostaticGeometry,
    *,
    reference_range_m: float,
    unambiguous_range_m: float,
    convention: Any = None,
) -> dict[str, Any]:
    """Check a candidate volume under the explicitly synthetic path convention.

    The check is a corner bound used for a small correctness fixture.  It is
    not a real-data registration or support result; the returned status says
    so explicitly.  A later GOTCHA gate must replace the convention and bounds
    only after resolving the source phase/reference contract.
    """

    if convention is None:
        convention = DEFAULT_SYNTHETIC_REFERENCE
    convention.validate()
    if not np.isfinite(reference_range_m) or not reference_range_m >= 0:
        raise ValueError("reference_range_m must be finite and nonnegative")
    if not np.isfinite(unambiguous_range_m) or not unambiguous_range_m > 0:
        raise ValueError("unambiguous_range_m must be finite and positive")
    corners = candidate.corners_xyz_m()
    tx = np.asarray(geometry.tx_xyz_m, dtype=np.float64)
    rx = np.asarray(geometry.rx_xyz_m, dtype=np.float64)
    path = (
        np.linalg.norm(tx[:, None, :] - corners[None, :, :], axis=-1)
        + np.linalg.norm(rx[:, None, :] - corners[None, :, :], axis=-1)
    )
    delta = path - float(convention.reference_range_factor) * float(reference_range_m)
    lower_margin = float(np.min(delta))
    upper_margin = float(unambiguous_range_m - np.max(delta))
    return {
        "schema": "rift_gotcha_height_support_preflight_v1",
        "candidate": candidate.as_dict(),
        "geometry_count": geometry.count,
        "reference_convention": convention.schema,
        "float64_path_calculation": True,
        "minimum_delta_path_m": float(np.min(delta)),
        "maximum_delta_path_m": float(np.max(delta)),
        "minimum_lower_margin_m": lower_margin,
        "minimum_upper_margin_m": upper_margin,
        "strict_nonwrapping_candidate": bool(lower_margin > 0 and upper_margin > 0),
        "real_gotcha_bounds_resolved": False,
        "status": "synthetic_preflight_only_real_bounds_unresolved",
    }


@dataclass(frozen=True)
class NativeObservation:
    """One row from a validated native shard."""

    identity: NativeObservationId
    role: str
    response: np.ndarray
    frequencies_hz: np.ndarray
    position_xyz_m: np.ndarray
    r0_m: float
    th_deg: float
    phi_deg: float
    r_correct_raw: float | None
    ph_correct_raw: float | None
    phase_reference: PhaseReferenceContract
    autofocus: AutofocusProvenance

    def __post_init__(self) -> None:
        if self.role not in _PAYLOAD_ROLE_NAMES:
            raise ValueError("native observations exposed by this adapter cannot be test rows")
        for name in ("response", "frequencies_hz", "position_xyz_m"):
            array = np.asarray(getattr(self, name))
            if array.flags.writeable:
                raise ValueError(f"{name} must be read-only")
        if self.autofocus.official_available:
            if self.r_correct_raw is None or self.ph_correct_raw is None:
                raise ValueError("co-polarized observations must retain raw autofocus values")
        elif self.r_correct_raw is not None or self.ph_correct_raw is not None:
            raise ValueError("cross-polarized observations cannot borrow autofocus values")


@dataclass(frozen=True)
class SyntheticReferenceConvention:
    """Declared convention used only by the direct synthetic fixture."""

    schema: str = SYNTHETIC_REFERENCE_SCHEMA
    speed_of_light_m_s: float = SPEED_OF_LIGHT_M_S
    path_formula: str = "norm(tx-point) + norm(rx-point)"
    reference_formula: str = "path_minus_reference_range"
    frequency_unit: str = "Hz"
    position_unit: str = "m"
    reference_range_unit: str = "m"
    phase_unit: str = "rad"
    phase_sign: int = 1
    reference_range_factor: float = 1.0
    named_candidate: str = "synthetic_two_way_path_reference"
    real_data_status: str = "synthetic_fixture_only"

    def validate(self) -> None:
        if self.schema != SYNTHETIC_REFERENCE_SCHEMA:
            raise ValueError("unexpected synthetic reference schema")
        if self.phase_sign not in (-1, 1):
            raise ValueError("synthetic phase_sign must be +1 or -1")
        if not np.isfinite(self.reference_range_factor) or self.reference_range_factor <= 0:
            raise ValueError("reference_range_factor must be positive and finite")
        if self.speed_of_light_m_s <= 0 or not np.isfinite(self.speed_of_light_m_s):
            raise ValueError("synthetic speed of light must be positive and finite")

    @property
    def adjoint_phase_sign(self) -> int:
        """The direct adjoint uses the conjugate kernel sign."""

        return -int(self.phase_sign)


DEFAULT_SYNTHETIC_REFERENCE = SyntheticReferenceConvention()


PUBLISHED_GOTCHA_REFERENCE_CANDIDATE = SyntheticReferenceConvention(
    reference_formula="2 * (one_way_range_minus_r0)",
    reference_range_factor=2.0,
    phase_sign=-1,
    named_candidate="demanet_2012_gotcha_phase_candidate",
    real_data_status="published_candidate_unverified_for_this_archive",
)


@dataclass(frozen=True)
class NativeShard:
    """Immutable, one-pass/one-polarization view of a Gate-1 NPZ archive."""

    path: Path
    shard_id: str
    pass_id: int
    polarization: str
    response: np.ndarray
    frequencies_hz: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    r0: np.ndarray
    th: np.ndarray
    phi: np.ndarray
    sector_id: np.ndarray
    pulse_index: np.ndarray
    role: np.ndarray
    r_correct_raw: np.ndarray
    ph_correct_raw: np.ndarray
    metadata: Mapping[str, Any]
    phase_reference: PhaseReferenceContract
    autofocus: AutofocusProvenance

    def __post_init__(self) -> None:
        for name in (
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
            "role",
            "r_correct_raw",
            "ph_correct_raw",
        ):
            if np.asarray(getattr(self, name)).flags.writeable:
                raise ValueError(f"NativeShard.{name} must be read-only")
        expected = canonical_shard_id(self.pass_id, self.polarization)
        if self.shard_id != expected:
            raise ValueError("shard identity does not match pass/polarization")

    @property
    def view_count(self) -> int:
        return int(self.response.shape[0])

    @property
    def frequency_count(self) -> int:
        return int(self.response.shape[1])

    @property
    def observation_ids(self) -> tuple[NativeObservationId, ...]:
        return tuple(
            NativeObservationId(
                self.pass_id,
                self.polarization,
                int(sector),
                int(pulse),
            )
            for sector, pulse in zip(self.sector_id, self.pulse_index)
        )

    def _indices_for_ids(
        self, observation_ids: Iterable[NativeObservationId]
    ) -> np.ndarray:
        index_by_id = {identity: index for index, identity in enumerate(self.observation_ids)}
        requested = tuple(observation_ids)
        if any(not isinstance(identity, NativeObservationId) for identity in requested):
            raise TypeError("observation selection must use NativeObservationId values")
        missing = [identity for identity in requested if identity not in index_by_id]
        if missing:
            raise KeyError(f"observation identity is not present in {self.shard_id}: {missing[0]}")
        return np.asarray([index_by_id[identity] for identity in requested], dtype=np.int64)

    def identities_for_role(self, role: str) -> tuple[NativeObservationId, ...]:
        role = str(role).lower()
        if role not in _PAYLOAD_ROLE_NAMES:
            raise ValueError("role selection is limited to train or validation; test remains sealed")
        return tuple(
            identity
            for identity, observed_role in zip(self.observation_ids, self.role)
            if str(observed_role) == role
        )

    def observations(
        self, observation_ids: Iterable[NativeObservationId] | None = None
    ) -> tuple[NativeObservation, ...]:
        if observation_ids is None:
            indices = np.arange(self.view_count, dtype=np.int64)
        else:
            indices = self._indices_for_ids(observation_ids)
        identities = self.observation_ids
        output = []
        for index in indices:
            i = int(index)
            identity = identities[i]
            output.append(
                NativeObservation(
                    identity=identity,
                    role=str(self.role[i]),
                    response=_readonly_array(self.response[i]),
                    frequencies_hz=_readonly_array(self.frequencies_hz),
                    position_xyz_m=_readonly_array(
                        [self.x[i], self.y[i], self.z[i]], dtype=np.float64
                    ),
                    r0_m=float(self.r0[i]),
                    th_deg=float(self.th[i]),
                    phi_deg=float(self.phi[i]),
                    r_correct_raw=(
                        float(self.r_correct_raw[i])
                        if self.autofocus.official_available
                        else None
                    ),
                    ph_correct_raw=(
                        float(self.ph_correct_raw[i])
                        if self.autofocus.official_available
                        else None
                    ),
                    phase_reference=self.phase_reference,
                    autofocus=self.autofocus,
                )
            )
        return tuple(output)

    def observation(self, observation_id: NativeObservationId) -> NativeObservation:
        return self.observations((observation_id,))[0]

    def paired_monostatic_geometry(
        self, observation_ids: Iterable[NativeObservationId] | None = None
    ) -> PairedMonostaticGeometry:
        if observation_ids is None:
            indices = np.arange(self.view_count, dtype=np.int64)
            identities = self.observation_ids
        else:
            requested = tuple(observation_ids)
            indices = self._indices_for_ids(requested)
            identities = requested
        positions = _readonly_array(
            np.column_stack((self.x[indices], self.y[indices], self.z[indices])),
            dtype=np.float64,
        )
        return PairedMonostaticGeometry(identities, positions, _readonly_array(positions))

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_pass_id: int | None = None,
        expected_polarization: str | None = None,
        expected_scene_id: str | None = None,
    ) -> "NativeShard":
        path = Path(path)
        arrays, metadata = _load_and_validate_archive(
            path,
            expected_pass_id=expected_pass_id,
            expected_polarization=expected_polarization,
            expected_scene_id=expected_scene_id,
        )
        return _native_shard_from_arrays(path, arrays, metadata)


def _read_metadata_before_response(
    path: Path,
    *,
    expected_pass_id: int | None,
    expected_polarization: str | None,
    expected_scene_id: str | None,
) -> Mapping[str, Any]:
    """Read only the NPZ metadata member before touching phase history."""

    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != _ARCHIVE_KEYS:
            raise ValueError(
                f"{path} has the wrong Gate-1 keys: {sorted(set(loaded.files) ^ _ARCHIVE_KEYS)}"
            )
        metadata_text = _scalar_text(loaded["metadata_json"], name="metadata_json")
    try:
        metadata = json.loads(metadata_text)
    except json.JSONDecodeError as exc:
        raise ValueError("metadata_json is not valid JSON") from exc
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata_json must encode a mapping")
    for key, expected in {
        "schema": NATIVE_ARCHIVE_SCHEMA,
        "test_opened": False,
        "test_payload_included": False,
        "corrections_applied": False,
        "autofocus_unapplied": True,
    }.items():
        if metadata.get(key) != expected:
            raise ValueError(f"metadata.{key} must be {expected!r}")
    pass_id = metadata.get("pass_id")
    polarization = str(metadata.get("polarization", "")).lower()
    if pass_id is None or int(pass_id) not in PASS_IDS:
        raise ValueError("metadata.pass_id must be one constant valid pass identity")
    if polarization not in POLARIZATIONS:
        raise ValueError("metadata.polarization must be one constant valid channel identity")
    if expected_pass_id is not None and int(pass_id) != int(expected_pass_id):
        raise ValueError(f"metadata pass_id {pass_id} differs from expected {expected_pass_id}")
    if expected_polarization is not None and polarization != str(expected_polarization).lower():
        raise ValueError(
            f"metadata polarization {polarization} differs from expected {expected_polarization}"
        )
    if expected_scene_id is not None and metadata.get("scene_id") != str(expected_scene_id):
        raise ValueError(
            f"metadata scene_id {metadata.get('scene_id')!r} differs from expected {expected_scene_id!r}"
        )
    if metadata.get("shard_id") != canonical_shard_id(int(pass_id), polarization):
        raise ValueError("metadata.shard_id does not match native pass/polarization identity")
    return metadata


def _load_and_validate_archive(
    path: Path,
    *,
    expected_pass_id: int | None,
    expected_polarization: str | None,
    expected_scene_id: str | None,
) -> tuple[dict[str, np.ndarray], Mapping[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"native shard must be a regular file: {path}")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            if any(item.compress_type != zipfile.ZIP_STORED for item in archive.infolist()):
                raise ValueError(f"{path} contains compressed NPZ members")
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{path} is not a valid NPZ archive") from exc

    # Gate-1 metadata closure is checked while only the metadata member is
    # read.  A test flag therefore fails before response/frequency payloads are
    # materialized by this adapter.
    _read_metadata_before_response(
        path,
        expected_pass_id=expected_pass_id,
        expected_polarization=expected_polarization,
        expected_scene_id=expected_scene_id,
    )
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {
            key: _readonly_array(loaded[key])
            for key in loaded.files
            if key != "response"
        }
    if any(value.dtype == object for value in arrays.values()):
        raise ValueError(f"{path} contains an object array")

    frequencies = arrays["frequencies_hz"]
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies_hz must retain the native one-dimensional grid")
    _require_finite(frequencies, name="frequencies_hz")
    frequency_count = int(frequencies.size)
    if frequency_count > 1 and not np.all(np.diff(frequencies.astype(np.float64)) > 0):
        raise ValueError("frequencies_hz must be strictly increasing in native order")

    sector_values = np.asarray(arrays["sector_id"])
    if sector_values.ndim != 1 or sector_values.size == 0:
        raise ValueError("sector_id must be a nonempty one-dimensional native identity")
    view_count = int(sector_values.size)
    for name in ("x", "y", "z", "r0", "th", "phi", "sector_id", "pulse_index", "pass_id", "polarization", "role"):
        _require_vector(arrays, name, view_count)
    for name in ("x", "y", "z", "r0", "th", "phi"):
        _require_finite(arrays[name], name=name)

    pass_id_values = np.unique(arrays["pass_id"])
    if pass_id_values.size != 1 or int(pass_id_values[0]) not in PASS_IDS:
        raise ValueError("pass_id must be one constant valid pass identity")
    pass_id = int(pass_id_values[0])
    polarization_values = {str(value).lower() for value in np.unique(arrays["polarization"])}
    if len(polarization_values) != 1 or next(iter(polarization_values)) not in POLARIZATIONS:
        raise ValueError("polarization must be one constant valid channel identity")
    polarization = next(iter(polarization_values))
    if expected_pass_id is not None and pass_id != int(expected_pass_id):
        raise ValueError(f"archive pass_id {pass_id} differs from expected {expected_pass_id}")
    if expected_polarization is not None and polarization != str(expected_polarization).lower():
        raise ValueError(
            f"archive polarization {polarization} differs from expected {expected_polarization}"
        )

    payload_sectors = set(_payload_sector_ids())
    sectors = arrays["sector_id"].astype(np.int64, copy=False)
    if set(np.unique(sectors).tolist()) != payload_sectors:
        raise ValueError("archive must contain exactly the 288 train and 36 validation sectors")
    expected_roles = np.asarray([_role_for_sector(int(value)) for value in sectors], dtype="U10")
    observed_roles = np.asarray([str(value).lower() for value in arrays["role"]], dtype="U10")
    if np.any(observed_roles == "test"):
        raise ValueError("sealed test rows are not permitted in the acquisition interface")
    if not np.array_equal(observed_roles, expected_roles):
        raise ValueError("archive role labels do not match the shared 288/36/36 split")
    arrays["role"] = _readonly_array(observed_roles)

    if not np.issubdtype(arrays["pulse_index"].dtype, np.integer):
        raise ValueError("pulse_index must be an integer native identity")
    identities = list(zip(sectors.tolist(), arrays["pulse_index"].astype(np.int64).tolist()))
    if len(set(identities)) != view_count:
        raise ValueError("(sector_id, pulse_index) must identify every native observation uniquely")
    for sector_id in payload_sectors:
        pulse_values = arrays["pulse_index"][sectors == sector_id]
        if pulse_values.size == 0 or np.unique(pulse_values).size != pulse_values.size:
            raise ValueError(f"sector {sector_id} has invalid native pulse identities")

    metadata_text = _scalar_text(arrays["metadata_json"], name="metadata_json")
    try:
        metadata = json.loads(metadata_text)
    except json.JSONDecodeError as exc:
        raise ValueError("metadata_json is not valid JSON") from exc
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata_json must encode a mapping")
    required_metadata = {
        "schema": NATIVE_ARCHIVE_SCHEMA,
        "pass_id": pass_id,
        "polarization": polarization,
        "test_opened": False,
        "test_payload_included": False,
        "corrections_applied": False,
        "autofocus_unapplied": True,
    }
    if expected_scene_id is not None:
        required_metadata["scene_id"] = str(expected_scene_id)
    for key, expected in required_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"metadata.{key} must be {expected!r}")
    if metadata.get("shard_id") != canonical_shard_id(pass_id, polarization):
        raise ValueError("metadata.shard_id does not match native pass/polarization identity")
    if metadata.get("payload_sector_ids") != sorted(payload_sectors):
        raise ValueError("metadata.payload_sector_ids does not preserve the shared payload split")
    if metadata.get("sealed_test_sector_ids") != list(_sealed_test_sector_ids()):
        raise ValueError("metadata.sealed_test_sector_ids does not preserve the sealed split")
    layout = metadata.get("layout")
    if not isinstance(layout, Mapping):
        raise ValueError("metadata.layout is required")
    for key, expected in {
        "native_frequency_preserved": True,
        "resampled": False,
        "padded": False,
        "trimmed": False,
        "autofocus_unapplied": True,
    }.items():
        if layout.get(key) != expected:
            raise ValueError(f"metadata.layout.{key} must be {expected!r}")

    autofocus_available = _scalar_bool(arrays["autofocus_available"], name="autofocus_available")
    autofocus_applied = _scalar_bool(arrays["autofocus_applied"], name="autofocus_applied")
    if autofocus_applied:
        raise ValueError("native Gate-1 interface refuses archives with applied autofocus")
    autofocus_state = _scalar_text(arrays["autofocus_state"], name="autofocus_state")
    if polarization in CO_POLARIZATIONS:
        if not autofocus_available or autofocus_state != AUTOFOCUS_RAW:
            raise ValueError("HH/VV must retain channel-own raw autofocus provenance")
        for name in ("r_correct_raw", "ph_correct_raw"):
            if arrays[name].shape != (view_count,):
                raise ValueError(f"{name} must be view-aligned for HH/VV")
            _require_finite(arrays[name], name=name)
    else:
        if autofocus_available or autofocus_state != AUTOFOCUS_OFFICIAL_ABSENT:
            raise ValueError("HV/VH must explicitly declare official autofocus absence")
        if arrays["r_correct_raw"].size or arrays["ph_correct_raw"].size:
            raise ValueError("HV/VH cannot borrow or carry correction arrays")

    # Only after metadata, row identity, split closure, and autofocus ownership
    # have passed do we materialize the potentially large phase-history member.
    with np.load(path, allow_pickle=False) as loaded:
        response = _readonly_array(loaded["response"])
    if response.ndim != 2 or not np.iscomplexobj(response) or response.size == 0:
        raise ValueError("response must be nonempty complex data with shape [view, frequency]")
    if response.shape != (view_count, frequency_count):
        raise ValueError("response shape does not match validated native row/frequency axes")
    _require_finite(response.real, name="response.real")
    _require_finite(response.imag, name="response.imag")
    if response.dtype == object:
        raise ValueError("response contains an object array")
    arrays["response"] = response

    return arrays, _freeze_json(metadata)


def _native_shard_from_arrays(
    path: Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]
) -> NativeShard:
    pass_id = int(np.unique(arrays["pass_id"])[0])
    polarization = str(np.unique(arrays["polarization"])[0]).lower()
    autofocus_available = _scalar_bool(arrays["autofocus_available"], name="autofocus_available")
    autofocus = AutofocusProvenance(
        schema=AUTOFOCUS_PROVENANCE_SCHEMA,
        mode=AUTOFOCUS_RAW if autofocus_available else AUTOFOCUS_OFFICIAL_ABSENT,
        official_available=autofocus_available,
        applied=False,
        source_shard_id=canonical_shard_id(pass_id, polarization),
        range_field="af.r_correct" if autofocus_available else None,
        phase_field="af.ph_correct" if autofocus_available else None,
    )
    return NativeShard(
        path=path,
        shard_id=canonical_shard_id(pass_id, polarization),
        pass_id=pass_id,
        polarization=polarization,
        response=arrays["response"],
        frequencies_hz=arrays["frequencies_hz"],
        x=arrays["x"],
        y=arrays["y"],
        z=arrays["z"],
        r0=arrays["r0"],
        th=arrays["th"],
        phi=arrays["phi"],
        sector_id=arrays["sector_id"],
        pulse_index=arrays["pulse_index"],
        role=arrays["role"],
        r_correct_raw=arrays["r_correct_raw"],
        ph_correct_raw=arrays["ph_correct_raw"],
        metadata=metadata,
        phase_reference=PhaseReferenceContract(),
        autofocus=autofocus,
    )


def load_native_shard(
    path: str | Path,
    *,
    expected_pass_id: int | None = None,
    expected_polarization: str | None = None,
    expected_scene_id: str | None = None,
) -> NativeShard:
    """Load and validate one sealed Gate-1 pass/polarization archive."""

    return NativeShard.load(
        path,
        expected_pass_id=expected_pass_id,
        expected_polarization=expected_polarization,
        expected_scene_id=expected_scene_id,
    )


def apply_published_autofocus(
    observation: NativeObservation,
    contract: AutofocusApplicationContract,
) -> None:
    """Reject native autofocus application in the raw-only Batch-A adapter.

    ``AutofocusApplicationContract`` is retained as a metadata candidate for a
    later frozen calibration gate.  Keeping this function as an unconditional
    guard makes it impossible for a caller to turn unknown real-data units,
    signs, or order into an executed correction by assertion alone.
    """

    del observation, contract
    raise RuntimeError(
        "Batch A is raw-only: first or double published-autofocus application is deferred until the frozen real-data gate"
    )


def _validate_geometry_and_frequencies(
    tx_positions_m: Any,
    rx_positions_m: Any | None,
    frequencies_hz: Any,
    reference_range_m: Any,
    convention: SyntheticReferenceConvention,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    convention.validate()
    tx = np.asarray(tx_positions_m, dtype=np.float64)
    if tx.ndim != 2 or tx.shape[1] != 3 or tx.shape[0] == 0:
        raise ValueError("tx_positions_m must have shape [observation, 3]")
    rx = tx if rx_positions_m is None else np.asarray(rx_positions_m, dtype=np.float64)
    if rx.shape != tx.shape:
        raise ValueError("paired monostatic tx/rx positions must have equal shapes")
    if not np.array_equal(tx, rx):
        raise ValueError("cross-product or bistatic Tx/Rx geometry is not allowed")
    frequencies = np.asarray(frequencies_hz)
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies_hz must be a nonempty one-dimensional native vector")
    frequencies_float = frequencies.astype(np.float64, copy=False)
    _require_finite(frequencies_float, name="frequencies_hz")
    if frequencies.size > 1 and not np.all(np.diff(frequencies_float) > 0):
        raise ValueError("frequencies_hz must be strictly increasing")
    reference = np.asarray(reference_range_m, dtype=np.float64)
    if reference.ndim == 0:
        reference = np.full(tx.shape[0], float(reference), dtype=np.float64)
    if reference.shape != (tx.shape[0],):
        raise ValueError("reference_range_m must be scalar or one value per observation")
    _require_finite(tx, name="tx_positions_m")
    _require_finite(reference, name="reference_range_m")
    return tx, rx, frequencies_float, reference


def direct_point_target_render(
    point_positions_m: Any,
    amplitudes: Any,
    *,
    tx_positions_m: Any,
    frequencies_hz: Any,
    rx_positions_m: Any | None = None,
    reference_range_m: Any = 0.0,
    convention: SyntheticReferenceConvention = DEFAULT_SYNTHETIC_REFERENCE,
) -> np.ndarray:
    """Render complex point targets with the explicit synthetic convention.

    The output has shape ``[observation, frequency]``.  The implementation is
    deliberately direct (no NUFFT, interpolation, or uniform-grid shortcut).
    """

    points = np.asarray(point_positions_m, dtype=np.float64)
    if points.ndim == 1:
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("point_positions_m must have shape [point, 3]")
    _require_finite(points, name="point_positions_m")
    coeffs = np.asarray(amplitudes, dtype=np.complex128)
    if coeffs.ndim == 0:
        coeffs = coeffs.reshape(1)
    if coeffs.shape != (points.shape[0],):
        raise ValueError("amplitudes must have one complex value per point")
    _require_finite(coeffs.real, name="amplitudes.real")
    _require_finite(coeffs.imag, name="amplitudes.imag")
    tx, rx, frequencies, reference = _validate_geometry_and_frequencies(
        tx_positions_m, rx_positions_m, frequencies_hz, reference_range_m, convention
    )
    path = (
        np.linalg.norm(tx[:, None, :] - points[None, :, :], axis=-1)
        + np.linalg.norm(rx[:, None, :] - points[None, :, :], axis=-1)
    )
    delta_path = path - float(convention.reference_range_factor) * reference[:, None]
    phase = (
        float(convention.phase_sign)
        * (2.0 * np.pi / float(convention.speed_of_light_m_s))
        * delta_path[:, :, None]
        * frequencies[None, None, :]
    )
    kernel = np.exp(1j * phase)
    return np.einsum("vpf,p->vf", kernel, coeffs, optimize=True)


def direct_point_target_adjoint(
    residual: Any,
    point_positions_m: Any,
    *,
    tx_positions_m: Any,
    frequencies_hz: Any,
    rx_positions_m: Any | None = None,
    reference_range_m: Any = 0.0,
    convention: SyntheticReferenceConvention = DEFAULT_SYNTHETIC_REFERENCE,
) -> np.ndarray:
    """Apply the Hermitian adjoint of :func:`direct_point_target_render`."""

    values = np.asarray(residual, dtype=np.complex128)
    frequencies = np.asarray(frequencies_hz)
    if values.shape != (np.asarray(tx_positions_m).shape[0], frequencies.size):
        raise ValueError("residual must have shape [observation, frequency]")
    points = np.asarray(point_positions_m, dtype=np.float64)
    if points.ndim == 1:
        points = points[None, :]
    basis = np.eye(points.shape[0], dtype=np.complex128)
    columns = [
        direct_point_target_render(
            points,
            basis[index],
            tx_positions_m=tx_positions_m,
            rx_positions_m=rx_positions_m,
            frequencies_hz=frequencies_hz,
            reference_range_m=reference_range_m,
            convention=convention,
        )
        for index in range(points.shape[0])
    ]
    kernel = np.stack(columns, axis=0)
    return np.einsum("pvf,vf->p", np.conjugate(kernel), values, optimize=True)


def forward_adjoint_inner_product_fixture(
    point_positions_m: Any,
    amplitudes: Any,
    residual: Any,
    *,
    tx_positions_m: Any,
    frequencies_hz: Any,
    rx_positions_m: Any | None = None,
    reference_range_m: Any = 0.0,
    convention: SyntheticReferenceConvention = DEFAULT_SYNTHETIC_REFERENCE,
) -> dict[str, float | bool]:
    """Return a deterministic ``<Ax,y>`` versus ``<x,Aᴴy>`` check."""

    forward = direct_point_target_render(
        point_positions_m,
        amplitudes,
        tx_positions_m=tx_positions_m,
        rx_positions_m=rx_positions_m,
        frequencies_hz=frequencies_hz,
        reference_range_m=reference_range_m,
        convention=convention,
    )
    adjoint = direct_point_target_adjoint(
        residual,
        point_positions_m,
        tx_positions_m=tx_positions_m,
        rx_positions_m=rx_positions_m,
        frequencies_hz=frequencies_hz,
        reference_range_m=reference_range_m,
        convention=convention,
    )
    lhs = np.vdot(forward, np.asarray(residual, dtype=np.complex128))
    rhs = np.vdot(np.asarray(amplitudes, dtype=np.complex128), adjoint)
    scale = max(1.0, abs(lhs), abs(rhs))
    error = float(abs(lhs - rhs) / scale)
    return {
        "lhs_real": float(lhs.real),
        "lhs_imag": float(lhs.imag),
        "rhs_real": float(rhs.real),
        "rhs_imag": float(rhs.imag),
        "relative_error": error,
        "passed": bool(error <= 5.0e-13),
    }


def forward_vjp_finite_difference_fixture(
    point_positions_m: Any,
    amplitudes: Any,
    residual: Any,
    *,
    tx_positions_m: Any,
    frequencies_hz: Any,
    rx_positions_m: Any | None = None,
    reference_range_m: Any = 0.0,
    convention: SyntheticReferenceConvention = DEFAULT_SYNTHETIC_REFERENCE,
    epsilon: float = 1.0e-6,
) -> dict[str, float | bool]:
    """Check the amplitude VJP with a central finite difference.

    The checked scalar is ``0.5 * ||A x - target||²`` with
    ``target = A x - residual``.  The complex VJP is therefore ``Aᴴ residual``;
    no measured response or independently rephased target is involved.
    """

    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive and finite")
    coeffs = np.asarray(amplitudes, dtype=np.complex128)
    forward = direct_point_target_render(
        point_positions_m,
        coeffs,
        tx_positions_m=tx_positions_m,
        rx_positions_m=rx_positions_m,
        frequencies_hz=frequencies_hz,
        reference_range_m=reference_range_m,
        convention=convention,
    )
    values = np.asarray(residual, dtype=np.complex128)
    target = forward - values
    analytic = direct_point_target_adjoint(
        values,
        point_positions_m,
        tx_positions_m=tx_positions_m,
        rx_positions_m=rx_positions_m,
        frequencies_hz=frequencies_hz,
        reference_range_m=reference_range_m,
        convention=convention,
    )

    def loss(candidate: np.ndarray) -> float:
        difference = direct_point_target_render(
            point_positions_m,
            candidate,
            tx_positions_m=tx_positions_m,
            rx_positions_m=rx_positions_m,
            frequencies_hz=frequencies_hz,
            reference_range_m=reference_range_m,
            convention=convention,
        ) - target
        return float(0.5 * np.vdot(difference, difference).real)

    numeric = np.empty_like(coeffs)
    for index in range(coeffs.size):
        plus_real = coeffs.copy()
        minus_real = coeffs.copy()
        plus_real[index] += epsilon
        minus_real[index] -= epsilon
        plus_imag = coeffs.copy()
        minus_imag = coeffs.copy()
        plus_imag[index] += 1j * epsilon
        minus_imag[index] -= 1j * epsilon
        numeric[index] = (
            (loss(plus_real) - loss(minus_real)) / (2.0 * epsilon)
            + 1j * (loss(plus_imag) - loss(minus_imag)) / (2.0 * epsilon)
        )
    difference = numeric - analytic
    scale = max(1.0, float(np.linalg.norm(analytic)))
    error = float(np.linalg.norm(difference) / scale)
    return {
        "epsilon": float(epsilon),
        "max_abs_error": float(np.max(np.abs(difference))),
        "relative_error": error,
        "passed": bool(error <= 5.0e-7),
    }


def endpoint_uniform_frequency_grid(frequencies_hz: Any) -> np.ndarray:
    """Construct a diagnostic endpoint-affine grid without changing native data."""

    native = np.asarray(frequencies_hz)
    if native.ndim != 1 or native.size == 0:
        raise ValueError("frequencies_hz must be a nonempty one-dimensional vector")
    if native.size == 1:
        return _readonly_array(native)
    return _readonly_array(
        np.linspace(float(native[0]), float(native[-1]), native.size, dtype=native.dtype)
    )


def compare_native_and_endpoint_uniform_frequency(
    frequencies_hz: Any,
    *,
    tx_positions_m: Any,
    point_positions_m: Any,
    amplitudes: Any,
    rx_positions_m: Any | None = None,
    reference_range_m: Any = 0.0,
    convention: SyntheticReferenceConvention = DEFAULT_SYNTHETIC_REFERENCE,
) -> dict[str, float | bool | str]:
    """Measure, but do not adopt, an endpoint-uniform approximation."""

    native = np.asarray(frequencies_hz)
    uniform = endpoint_uniform_frequency_grid(native)
    exact = direct_point_target_render(
        point_positions_m,
        amplitudes,
        tx_positions_m=tx_positions_m,
        rx_positions_m=rx_positions_m,
        frequencies_hz=native,
        reference_range_m=reference_range_m,
        convention=convention,
    )
    approximate = direct_point_target_render(
        point_positions_m,
        amplitudes,
        tx_positions_m=tx_positions_m,
        rx_positions_m=rx_positions_m,
        frequencies_hz=uniform,
        reference_range_m=reference_range_m,
        convention=convention,
    )
    error = approximate - exact
    return {
        "native_dtype": str(native.dtype),
        "uniform_dtype": str(uniform.dtype),
        "max_abs_frequency_delta_hz": float(np.max(np.abs(uniform.astype(np.float64) - native.astype(np.float64)))),
        "max_abs_render_delta": float(np.max(np.abs(error))),
        "relative_render_delta": float(np.linalg.norm(error) / max(1.0, np.linalg.norm(exact))),
        "native_retained": True,
    }


def virtual_reference_mapping_regression_fixture(
    *,
    observation_positions_m: Any,
    center_xyz_m: Any,
    point_positions_m: Any,
    amplitudes: Any,
    frequencies_hz: Any,
    r0_offset_m: Any,
    reference_range_m: float,
) -> dict[str, float | bool | int | str]:
    """Exercise per-pulse virtual-reference mapping with a direct SAR fixture.

    For ``kappa = 4*pi*f/c`` this uses the named candidate
    ``q = exp(-1j*kappa*(Rref + r0 - d))``.  We construct paired monostatic
    observations, derive ``d = ||position-center||`` and ``r0=d+offset``, then
    render the native and old direct operators with the named candidate.  The
    check proves full-q operator round-trip and direct SAR adjoint equivalence
    while showing that a constant-only ``Rref`` factor fails when ``r0-d``
    varies.  It is synthetic algebra, not measured phase or autofocus evidence.
    """

    positions = np.asarray(observation_positions_m, dtype=np.float64)
    center = np.asarray(center_xyz_m, dtype=np.float64)
    points = np.asarray(point_positions_m, dtype=np.float64)
    coeffs = np.asarray(amplitudes, dtype=np.complex128)
    frequencies = np.asarray(frequencies_hz)
    offsets = np.asarray(r0_offset_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        raise ValueError("observation_positions_m must have shape [observation, 3]")
    if frequencies.ndim != 1 or frequencies.size == 0:
        raise ValueError("frequencies_hz must be a nonempty vector")
    if center.shape != (3,) or not np.isfinite(center).all() or np.all(center == 0):
        raise ValueError("center_xyz_m must be a finite nonzero center")
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("point_positions_m must have shape [point, 3]")
    if coeffs.shape != (points.shape[0],):
        raise ValueError("amplitudes must have one complex value per point")
    if offsets.shape != (positions.shape[0],):
        raise ValueError("r0_offset_m must have one value per observation")
    _require_finite(positions, name="observation_positions_m")
    _require_finite(points, name="point_positions_m")
    _require_finite(coeffs.real, name="amplitudes.real")
    _require_finite(coeffs.imag, name="amplitudes.imag")
    _require_finite(frequencies.astype(np.float64), name="frequencies_hz")
    _require_finite(offsets, name="r0_offset_m")
    if not np.isfinite(reference_range_m):
        raise ValueError("reference_range_m must be finite")
    d = np.linalg.norm(positions - center[None, :], axis=1)
    r0 = d + offsets
    kappa = 4.0 * np.pi * frequencies.astype(np.float64) / SPEED_OF_LIGHT_M_S
    q = np.exp(-1j * kappa[None, :] * (float(reference_range_m) + r0[:, None] - d[:, None]))
    scalar_q = np.exp(-1j * kappa * float(reference_range_m))[None, :]
    candidate = PUBLISHED_GOTCHA_REFERENCE_CANDIDATE

    def operator(reference: np.ndarray) -> np.ndarray:
        basis = np.eye(points.shape[0], dtype=np.complex128)
        return np.stack(
            [
                direct_point_target_render(
                    points,
                    basis[index],
                    tx_positions_m=positions,
                    rx_positions_m=positions,
                    frequencies_hz=frequencies,
                    reference_range_m=reference,
                    convention=candidate,
                )
                for index in range(points.shape[0])
            ],
            axis=0,
        )

    native_operator = operator(r0)
    old_operator = operator(d - float(reference_range_m))
    native_signal = direct_point_target_render(
        points,
        coeffs,
        tx_positions_m=positions,
        rx_positions_m=positions,
        frequencies_hz=frequencies,
        reference_range_m=r0,
        convention=candidate,
    )
    old_signal = direct_point_target_render(
        points,
        coeffs,
        tx_positions_m=positions,
        rx_positions_m=positions,
        frequencies_hz=frequencies,
        reference_range_m=d - float(reference_range_m),
        convention=candidate,
    )
    full_q_operator_error = float(
        np.linalg.norm(old_operator - q[None, :, :] * native_operator)
        / max(1.0, np.linalg.norm(old_operator))
    )
    full_q_signal_error = float(
        np.linalg.norm(old_signal - q * native_signal) / max(1.0, np.linalg.norm(old_signal))
    )
    inverse_operator_error = float(
        np.linalg.norm(np.conjugate(q)[None, :, :] * old_operator - native_operator)
        / max(1.0, np.linalg.norm(native_operator))
    )
    y = old_signal + np.asarray(
        [[0.25 + 0.5j] * frequencies.size] * positions.shape[0], dtype=np.complex128
    )
    old_adjoint = direct_point_target_adjoint(
        y,
        points,
        tx_positions_m=positions,
        rx_positions_m=positions,
        frequencies_hz=frequencies,
        reference_range_m=d - float(reference_range_m),
        convention=candidate,
    )
    native_adjoint_from_q = direct_point_target_adjoint(
        np.conjugate(q) * y,
        points,
        tx_positions_m=positions,
        rx_positions_m=positions,
        frequencies_hz=frequencies,
        reference_range_m=r0,
        convention=candidate,
    )
    adjoint_error = float(
        np.linalg.norm(old_adjoint - native_adjoint_from_q)
        / max(1.0, np.linalg.norm(old_adjoint))
    )
    scalar_only_error = float(
        np.linalg.norm(old_operator - scalar_q[None, :, :] * native_operator)
        / max(1.0, np.linalg.norm(old_operator))
    )
    return {
        "schema": "rift_gotcha_virtual_reference_mapping_regression_v1",
        "mapping": "q=exp(-1j*4*pi*f/c*(Rref+r0-d))",
        "forward_phase_sign": -1,
        "adjoint_phase_sign": 1,
        "frequency_dtype": str(frequencies.dtype),
        "r0_minus_d_span_m": float(np.ptp(r0 - d)),
        "observation_count": int(positions.shape[0]),
        "point_count": int(points.shape[0]),
        "center_xyz_m_nonzero": True,
        "off_grid_scatterer_count": int(points.shape[0]),
        "full_q_operator_error": full_q_operator_error,
        "full_q_signal_error": full_q_signal_error,
        "full_q_inverse_operator_error": inverse_operator_error,
        "full_q_adjoint_relative_error": adjoint_error,
        "constant_only_mapping_relative_error": scalar_only_error,
        "constant_only_mapping_failed": bool(scalar_only_error > 1.0e-6),
        "passed": bool(
            full_q_operator_error <= 2.0e-12
            and full_q_signal_error <= 2.0e-12
            and inverse_operator_error <= 2.0e-12
            and adjoint_error <= 2.0e-12
            and scalar_only_error > 1.0e-6
        ),
        "real_data_status": "synthetic_algebra_only_no_measured_inference",
    }


__all__ = [
    "AUTOFOCUS_OFFICIAL_ABSENT",
    "AUTOFOCUS_PUBLISHED",
    "AUTOFOCUS_RAW",
    "AUTOFOCUS_TRAIN_NUISANCE",
    "AutofocusApplicationContract",
    "AutofocusProvenance",
    "DEFAULT_SYNTHETIC_REFERENCE",
    "HeightSupportCandidate",
    "NativeObservation",
    "NativeObservationId",
    "NativeShard",
    "PairedMonostaticGeometry",
    "PUBLISHED_GOTCHA_REFERENCE_CANDIDATE",
    "PhaseReferenceContract",
    "SPEED_OF_LIGHT_M_S",
    "SyntheticReferenceConvention",
    "TrainNuisanceCalibrationContract",
    "apply_published_autofocus",
    "compare_native_and_endpoint_uniform_frequency",
    "direct_point_target_adjoint",
    "direct_point_target_render",
    "endpoint_uniform_frequency_grid",
    "forward_adjoint_inner_product_fixture",
    "forward_vjp_finite_difference_fixture",
    "load_native_shard",
    "preflight_height_support_candidate",
    "virtual_reference_mapping_regression_fixture",
]
