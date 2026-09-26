"""Immutable downstream source-AF algebra for the bounded Step-2 compare.

This module consumes already validated, raw/unapplied HH ``NativeObservation``
records but never mutates them and never changes the Gate-1 acquisition API.
It creates an explicitly labeled downstream representation containing both raw
provenance and the effective source-AF ``r0``/response pair.  A raw payload is
never serialized or returned under a corrected label, and a full fixed-r0
equivalent BP is rejected: that representation is legal only on the declared
9-point panel.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


SPEED_OF_LIGHT_M_S = 299_792_458.0
SOURCE_AF_SCHEMA = "rift_gotcha_step2_source_af_downstream_representation_v1"
SOURCE_AF_FORMULA = "r0_src=float64(r0_raw)+float64(r_correct_raw); fp_src=complex128(fp_raw)*exp(+i*float64(ph_correct_raw))"
FIXED_R0_EQUIVALENT_FORMULA = "fp_equiv=fp_raw*exp(i*ph_correct-i*(4*pi*f/c)*r_correct), raw r0"
RAW_REPRESENTATION = "raw_unapplied"
SOURCE_REPRESENTATION = "source_af"
EQUIVALENT_REPRESENTATION = "fixed_r0_equivalent_panel_only"
PANEL_COORDINATES_M = (-8.0, 0.0, 8.0)
DEFAULT_POINT_CHUNK_SIZE = 4096
DEFAULT_MAX_KERNEL_EVALUATIONS = 250_000_000
DEFAULT_RELATIVE_TOLERANCE = 2.0e-11
DEFAULT_SCALED_ABSOLUTE_TOLERANCE = 2.0e-11


def _readonly(value: Any, *, dtype: Any) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _finite(value: np.ndarray, label: str) -> None:
    if not np.isfinite(value).all():
        raise ValueError(f"{label} contains non-finite values")


def _id_key(identity: Any) -> tuple[int, str, int, int]:
    return (
        int(identity.pass_id),
        str(identity.polarization).lower(),
        int(identity.sector_id),
        int(identity.pulse_index),
    )


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): value[key] for key in value})


@dataclass(frozen=True)
class SourceAFScope:
    """Immutable header scope for train or explicitly selected validation conversion.

    The historic default remains P1/HH/train sector 002 with 117 records;
    validation scopes must be explicit and never include sealed test rows.
    """

    pass_id: int = 1
    polarization: str = "hh"
    sector_ids: tuple[int, ...] = (2,)
    role: str = "train"
    expected_count: int | None = 117
    name: str = "historic_sector002"

    def __post_init__(self) -> None:
        pass_id = int(self.pass_id)
        polarization = str(self.polarization).lower()
        sector_ids = tuple(sorted({int(value) for value in self.sector_ids}))
        role = str(self.role).lower()
        if pass_id != 1:
            raise ValueError("source-AF scopes are limited to pass 1")
        if polarization != "hh":
            raise ValueError("source-AF scopes are limited to HH")
        if not sector_ids or any(value < 1 or value > 360 for value in sector_ids):
            raise ValueError("source-AF scope requires one or more sector IDs in 1..360")
        if role not in {"train", "validation"}:
            raise ValueError("source-AF scopes are limited to train or validation observations")
        if self.expected_count is not None:
            expected_count = int(self.expected_count)
            if expected_count <= 0:
                raise ValueError("source-AF expected_count must be positive or None")
        else:
            expected_count = None
        name = str(self.name)
        if not name:
            raise ValueError("source-AF scope name must be nonempty")
        object.__setattr__(self, "pass_id", pass_id)
        object.__setattr__(self, "polarization", polarization)
        object.__setattr__(self, "sector_ids", sector_ids)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "expected_count", expected_count)
        object.__setattr__(self, "name", name)

    @property
    def sector_label(self) -> str:
        if self.sector_ids == (2,):
            return "sector 002"
        return "sectors [" + ", ".join(str(value) for value in self.sector_ids) + "]"

    def validate_header(self, identity: Any, role: Any) -> None:
        if (
            int(getattr(identity, "pass_id", -1)) != self.pass_id
            or str(getattr(identity, "polarization", "")).lower() != self.polarization
            or int(getattr(identity, "sector_id", -1)) not in self.sector_ids
        ):
            raise ValueError(
                f"source-AF header gate requires pass {self.pass_id} {self.polarization.upper()} {self.sector_label}"
            )
        if str(role).lower() == "test":
            raise ValueError("source-AF conversion rejects sealed test observations")
        if str(role).lower() != self.role:
            raise ValueError(f"source-AF header gate requires a {self.role} observation")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pass_id": self.pass_id,
            "polarization": self.polarization,
            "sector_ids": list(self.sector_ids),
            "role": self.role,
            "expected_count": self.expected_count,
        }


HISTORIC_SOURCE_AF_SCOPE = SourceAFScope()


@dataclass(frozen=True)
class CamryVVTrainSourceAFScope:
    """Explicit same-channel P1/VV/TRAIN/sector-002 source-AF scope."""

    pass_id: int = 1
    polarization: str = "vv"
    sector_ids: tuple[int, ...] = (2,)
    role: str = "train"
    expected_count: int | None = None
    name: str = "camry_p1_vv_train_sector002"

    def __post_init__(self) -> None:
        if int(self.pass_id) != 1:
            raise ValueError("Camry VV source-AF scope requires pass 1")
        if str(self.polarization).lower() != "vv":
            raise ValueError("Camry VV source-AF scope requires polarization VV")
        if tuple(sorted({int(value) for value in self.sector_ids})) != (2,):
            raise ValueError("Camry VV source-AF scope requires sector 002")
        if str(self.role).lower() != "train":
            raise ValueError("Camry VV source-AF scope requires TRAIN observations")
        if self.expected_count is None or int(self.expected_count) <= 0:
            raise ValueError("Camry VV source-AF scope requires an explicit positive expected_count")
        if str(self.name) != "camry_p1_vv_train_sector002":
            raise ValueError("Camry VV source-AF scope name must be camry_p1_vv_train_sector002")
        object.__setattr__(self, "pass_id", 1)
        object.__setattr__(self, "polarization", "vv")
        object.__setattr__(self, "sector_ids", (2,))
        object.__setattr__(self, "role", "train")
        object.__setattr__(self, "expected_count", int(self.expected_count))
        object.__setattr__(self, "name", str(self.name))

    @property
    def sector_label(self) -> str:
        return "sector 002"

    def validate_header(self, identity: Any, role: Any) -> None:
        if (
            int(getattr(identity, "pass_id", -1)) != 1
            or str(getattr(identity, "polarization", "")).lower() != "vv"
            or int(getattr(identity, "sector_id", -1)) != 2
        ):
            raise ValueError("Camry VV source-AF header gate requires P1/VV/sector 002")
        if str(role).lower() == "test":
            raise ValueError("source-AF conversion rejects sealed test observations")
        if str(role).lower() != "train":
            raise ValueError("Camry VV source-AF conversion requires TRAIN observations")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pass_id": self.pass_id,
            "polarization": self.polarization,
            "sector_ids": list(self.sector_ids),
            "role": self.role,
            "expected_count": self.expected_count,
        }


@dataclass(frozen=True)
class MultipassHHTrainSourceAFScope:
    """Explicit multipass HH/TRAIN source-AF header scope.

    This adapter is intentionally separate from :class:`SourceAFScope` so the
    historic P1/HH/sector-002 contract remains unchanged.  It is limited to
    channel-owned HH corrections and TRAIN rows; it does not infer a source-AF
    convention for VV/HV/VH or open sealed TEST rows.
    """

    pass_ids: tuple[int, ...] = tuple(range(1, 9))
    polarization: str = "hh"
    sector_ids: tuple[int, ...] = (2, 92, 182, 272)
    role: str = "train"
    expected_count: int | None = None
    name: str = "multipass_panel_hh_train"

    def __post_init__(self) -> None:
        pass_ids = tuple(sorted({int(value) for value in self.pass_ids}))
        polarization = str(self.polarization).lower()
        sector_ids = tuple(sorted({int(value) for value in self.sector_ids}))
        role = str(self.role).lower()
        if pass_ids != tuple(range(1, 9)):
            raise ValueError("multipass TopHat source-AF scope requires exactly passes 1..8")
        if polarization != "hh":
            raise ValueError("multipass source-AF scope is limited to HH")
        if sector_ids != (2, 92, 182, 272):
            raise ValueError("multipass TopHat source-AF scope requires exactly sectors 002/092/182/272")
        if role != "train":
            raise ValueError("multipass source-AF scope is limited to TRAIN observations")
        if self.expected_count is not None and int(self.expected_count) <= 0:
            raise ValueError("multipass source-AF expected_count must be positive or None")
        name = str(self.name)
        if not name:
            raise ValueError("multipass source-AF scope name must be nonempty")
        object.__setattr__(self, "pass_ids", pass_ids)
        object.__setattr__(self, "polarization", polarization)
        object.__setattr__(self, "sector_ids", sector_ids)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "expected_count", None if self.expected_count is None else int(self.expected_count))
        object.__setattr__(self, "name", name)

    @property
    def sector_label(self) -> str:
        return "sectors [" + ", ".join(str(value) for value in self.sector_ids) + "]"

    def validate_header(self, identity: Any, role: Any) -> None:
        if (
            int(getattr(identity, "pass_id", -1)) not in self.pass_ids
            or str(getattr(identity, "polarization", "")).lower() != self.polarization
            or int(getattr(identity, "sector_id", -1)) not in self.sector_ids
        ):
            raise ValueError(
                "multipass source-AF header gate requires one of "
                f"passes {list(self.pass_ids)} {self.polarization.upper()} {self.sector_label}"
            )
        if str(role).lower() != self.role:
            raise ValueError("multipass source-AF conversion requires TRAIN observations")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pass_ids": list(self.pass_ids),
            "polarization": self.polarization,
            "sector_ids": list(self.sector_ids),
            "role": self.role,
            "expected_count": self.expected_count,
        }


@dataclass(frozen=True)
class TophatHHTrainValidationSourceAFScope:
    """Explicit P1/P7 HH TRAIN plus non-TEST validation source-AF scope."""

    pass_ids: tuple[int, ...] = (1, 7)
    polarization: str = "hh"
    train_sector_ids: tuple[int, ...] = (2, 92, 182, 272)
    validation_sector_ids: tuple[int, ...] = (1, 91, 181, 271)
    expected_train_count: int | None = 949
    expected_validation_count: int | None = None
    name: str = "tophat_p1_p7_hh_train_validation"

    def __post_init__(self) -> None:
        pass_ids = tuple(sorted({int(value) for value in self.pass_ids}))
        polarization = str(self.polarization).lower()
        train_sector_ids = tuple(sorted({int(value) for value in self.train_sector_ids}))
        validation_sector_ids = tuple(sorted({int(value) for value in self.validation_sector_ids}))
        if pass_ids != (1, 7):
            raise ValueError("TopHat source-AF scope requires exactly passes 1 and 7")
        if polarization != "hh":
            raise ValueError("TopHat source-AF scope is limited to HH")
        if train_sector_ids != (2, 92, 182, 272):
            raise ValueError("TopHat TRAIN source-AF scope requires sectors 002/092/182/272")
        if validation_sector_ids != (1, 91, 181, 271):
            raise ValueError("TopHat validation source-AF scope requires sectors 001/091/181/271")
        for value, label in (
            (self.expected_train_count, "expected_train_count"),
            (self.expected_validation_count, "expected_validation_count"),
        ):
            if value is not None and int(value) <= 0:
                raise ValueError(f"{label} must be positive or None")
        if not str(self.name):
            raise ValueError("TopHat source-AF scope name must be nonempty")
        object.__setattr__(self, "pass_ids", pass_ids)
        object.__setattr__(self, "polarization", polarization)
        object.__setattr__(self, "train_sector_ids", train_sector_ids)
        object.__setattr__(self, "validation_sector_ids", validation_sector_ids)
        object.__setattr__(self, "expected_train_count", None if self.expected_train_count is None else int(self.expected_train_count))
        object.__setattr__(self, "expected_validation_count", None if self.expected_validation_count is None else int(self.expected_validation_count))
        object.__setattr__(self, "name", str(self.name))

    def validate_header(self, identity: Any, role: Any) -> None:
        pass_id = int(getattr(identity, "pass_id", -1))
        polarization = str(getattr(identity, "polarization", "")).lower()
        sector_id = int(getattr(identity, "sector_id", -1))
        normalized_role = str(role).lower()
        if pass_id not in self.pass_ids or polarization != self.polarization:
            raise ValueError("TopHat source-AF header gate requires P1/P7 HH")
        if normalized_role == "test":
            raise ValueError("TopHat source-AF conversion rejects sealed TEST observations")
        if normalized_role == "train" and sector_id not in self.train_sector_ids:
            raise ValueError("TopHat source-AF TRAIN gate requires sectors 002/092/182/272")
        if normalized_role == "validation" and sector_id not in self.validation_sector_ids:
            raise ValueError("TopHat source-AF validation gate requires sectors 001/091/181/271")
        if normalized_role not in {"train", "validation"}:
            raise ValueError("TopHat source-AF conversion is limited to TRAIN/validation")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pass_ids": list(self.pass_ids),
            "polarization": self.polarization,
            "train_sector_ids": list(self.train_sector_ids),
            "validation_sector_ids": list(self.validation_sector_ids),
            "expected_train_count": self.expected_train_count,
            "expected_validation_count": self.expected_validation_count,
        }


SourceAFScopeLike = SourceAFScope | CamryVVTrainSourceAFScope | MultipassHHTrainSourceAFScope | TophatHHTrainValidationSourceAFScope


def _validate_source_af_metadata(observation: Any, scope: SourceAFScopeLike) -> None:
    """Validate every non-response source-AF field before payload access."""

    identity = getattr(observation, "identity", None)
    if identity is None:
        raise TypeError("source-AF conversion requires NativeObservation.identity")
    scope.validate_header(identity, getattr(observation, "role", ""))
    position = np.asarray(getattr(observation, "position_xyz_m", None), dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("source-AF metadata requires finite position_xyz_m[3]")
    r0 = float(getattr(observation, "r0_m"))
    if not np.isfinite(r0):
        raise ValueError("source-AF metadata requires finite native r0")
    autofocus = getattr(observation, "autofocus", None)
    if autofocus is None or getattr(autofocus, "official_available", False) is not True:
        raise ValueError("HH source-AF conversion requires channel-owned raw correction arrays")
    if bool(getattr(autofocus, "applied", True)):
        raise ValueError("source-AF conversion rejects observations with originally applied autofocus")
    phase_reference = getattr(observation, "phase_reference", None)
    if phase_reference is None or getattr(phase_reference, "reference_range_field", None) != "r0":
        raise ValueError("source-AF header gate requires native per-pulse r0 provenance")
    if getattr(phase_reference, "geometry_contract", None) != "paired_monostatic_tx_equals_rx_same_observation":
        raise ValueError("source-AF header gate requires paired monostatic geometry")
    r_correct = getattr(observation, "r_correct_raw", None)
    ph_correct = getattr(observation, "ph_correct_raw", None)
    if r_correct is None or ph_correct is None:
        raise ValueError("source-AF conversion requires both raw correction values")
    if not np.isfinite(float(r_correct)) or not np.isfinite(float(ph_correct)):
        raise ValueError("source-AF correction values must be finite")
    frequencies = np.asarray(getattr(observation, "frequencies_hz"), dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0 or not np.isfinite(frequencies).all():
        raise ValueError("source-AF metadata requires a finite native frequency vector")
    if frequencies.size > 1 and not np.all(np.diff(frequencies) > 0):
        raise ValueError("source-AF native frequencies must be strictly increasing")
    if isinstance(scope, CamryVVTrainSourceAFScope):
        if getattr(phase_reference, "frequency_values", None) != "native_stored_exact":
            raise ValueError("Camry VV source-AF requires phase_reference.frequency_values=native_stored_exact")
        if getattr(autofocus, "mode", None) != "raw_channel_own_arrays_unapplied":
            raise ValueError("Camry VV source-AF requires raw channel-owned autofocus mode")
        if getattr(autofocus, "source_shard_id", None) != "pass1_vv":
            raise ValueError("Camry VV source-AF requires source_shard_id=pass1_vv")
        if getattr(autofocus, "range_field", None) not in {"r_correct_raw", "af.r_correct"}:
            raise ValueError("Camry VV source-AF requires the native raw range correction field")
        if getattr(autofocus, "phase_field", None) not in {"ph_correct_raw", "af.ph_correct"}:
            raise ValueError("Camry VV source-AF requires the native raw phase correction field")


@dataclass(frozen=True)
class SourceAFObservation:
    """One immutable raw/source-AF pair with explicit provenance labels."""

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
    provenance: Mapping[str, Any]
    representation_tag: str
    scope: SourceAFScopeLike

    def __post_init__(self) -> None:
        identity = self.identity
        if not isinstance(self.scope, (SourceAFScope, CamryVVTrainSourceAFScope, MultipassHHTrainSourceAFScope, TophatHHTrainValidationSourceAFScope)):
            raise TypeError("source-AF records require an immutable SourceAFScope")
        self.scope.validate_header(identity, self.role)

        position = _readonly(self.position_xyz_m, dtype=np.float64)
        frequencies = _readonly(self.frequencies_hz, dtype=np.float64)
        response_raw = _readonly(self.response_raw, dtype=np.complex128)
        effective_response = _readonly(self.effective_response, dtype=np.complex128)
        if position.shape != (3,):
            raise ValueError("position_xyz_m must have shape [3]")
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError("frequencies_hz must be a nonempty native vector")
        if response_raw.shape != frequencies.shape or effective_response.shape != frequencies.shape:
            raise ValueError("raw/effective responses must match the native frequency vector")
        if self.representation_tag != SOURCE_REPRESENTATION:
            raise ValueError("derived source-AF records require the explicit source_af representation tag")
        for value, label in (
            (position, "position_xyz_m"),
            (frequencies, "frequencies_hz"),
            (response_raw.real, "response_raw.real"),
            (response_raw.imag, "response_raw.imag"),
            (effective_response.real, "effective_response.real"),
            (effective_response.imag, "effective_response.imag"),
        ):
            _finite(value, label)
        for value, label in (
            (self.r0_raw_m, "r0_raw_m"),
            (self.r_correct_raw_m, "r_correct_raw_m"),
            (self.ph_correct_raw_rad, "ph_correct_raw_rad"),
            (self.effective_r0_m, "effective_r0_m"),
        ):
            if not np.isfinite(float(value)):
                raise ValueError(f"{label} must be finite")
        expected_r0 = float(np.float64(self.r0_raw_m) + np.float64(self.r_correct_raw_m))
        expected_response = np.asarray(self.response_raw, dtype=np.complex128) * np.exp(
            1j * np.float64(self.ph_correct_raw_rad)
        )
        if not np.isclose(float(self.effective_r0_m), expected_r0, rtol=0.0, atol=0.0):
            raise ValueError("effective_r0_m does not use float64 promotion before addition")
        if not np.array_equal(np.asarray(effective_response), expected_response):
            raise ValueError("effective_response does not use the declared complex128 phase correction")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("provenance must be a mapping")
        object.__setattr__(self, "position_xyz_m", position)
        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "response_raw", response_raw)
        object.__setattr__(self, "effective_response", effective_response)
        object.__setattr__(self, "provenance", _freeze_mapping(dict(self.provenance)))

    @property
    def r0_source_m(self) -> float:
        """Compatibility label for the explicitly effective source r0."""

        return self.effective_r0_m

    @property
    def response_source(self) -> np.ndarray:
        """Compatibility label for the explicitly effective response."""

        return self.effective_response

    @classmethod
    def from_observation(
        cls,
        observation: Any,
        *,
        scope: SourceAFScopeLike | None = None,
    ) -> "SourceAFObservation":
        if isinstance(observation, cls):
            raise TypeError("source-AF conversion rejects an already-derived view")
        scope = HISTORIC_SOURCE_AF_SCOPE if scope is None else scope
        if not isinstance(scope, (SourceAFScope, CamryVVTrainSourceAFScope, MultipassHHTrainSourceAFScope, TophatHHTrainValidationSourceAFScope)):
            raise TypeError("source-AF conversion requires an immutable SourceAFScope")
        # The immutable identity/role gate deliberately precedes response access.
        identity = getattr(observation, "identity", None)
        _validate_source_af_metadata(observation, scope)
        autofocus = getattr(observation, "autofocus")
        phase_reference = getattr(observation, "phase_reference")
        r_correct = getattr(observation, "r_correct_raw")
        ph_correct = getattr(observation, "ph_correct_raw")
        frequencies = np.asarray(getattr(observation, "frequencies_hz"), dtype=np.float64)
        _finite(frequencies, "frequencies_hz")
        r0_raw = float(np.float64(getattr(observation, "r0_m")))
        r_correct_f64 = float(np.float64(r_correct))
        ph_correct_f64 = float(np.float64(ph_correct))
        # Response access occurs only after the identity/role/provenance gate.
        raw_response = np.asarray(getattr(observation, "response"), dtype=np.complex128)
        return cls(
            identity=identity,
            role=str(getattr(observation, "role")),
            position_xyz_m=np.asarray(getattr(observation, "position_xyz_m"), dtype=np.float64),
            frequencies_hz=frequencies,
            r0_raw_m=r0_raw,
            response_raw=raw_response,
            r_correct_raw_m=r_correct_f64,
            ph_correct_raw_rad=ph_correct_f64,
            effective_r0_m=float(np.float64(r0_raw) + np.float64(r_correct_f64)),
            effective_response=raw_response * np.exp(1j * np.float64(ph_correct_f64)),
            provenance={
                "schema": SOURCE_AF_SCHEMA,
                "raw_response_state": "retained_unmodified_from_native_observation",
                "raw_r0_state": "retained_unmodified_from_native_observation",
                "raw_autofocus_state": "channel_owned_raw_unapplied",
                "source_formula": SOURCE_AF_FORMULA,
                "correction_representation": "one_source_af_representation_only",
                "correction_source": {
                    "source_shard_id": getattr(autofocus, "source_shard_id", None),
                    "range_field": getattr(autofocus, "range_field", None),
                    "phase_field": getattr(autofocus, "phase_field", None),
                    "autofocus_mode": getattr(autofocus, "mode", None),
                    "frequency_values": getattr(phase_reference, "frequency_values", None),
                },
                "scope": scope.as_dict(),
            },
            representation_tag=SOURCE_REPRESENTATION,
            scope=scope,
        )

    def raw_payload(self) -> tuple[float, np.ndarray]:
        """Return only the explicitly raw/unapplied payload."""

        return self.r0_raw_m, self.response_raw

    def source_payload(self) -> tuple[float, np.ndarray]:
        """Return only the explicitly source-AF payload."""

        return self.effective_r0_m, self.effective_response

    def metadata(self) -> dict[str, Any]:
        return {
            "identity": self.identity.as_dict() if hasattr(self.identity, "as_dict") else str(self.identity),
            "role": self.role,
            "raw_r0_m": self.r0_raw_m,
            "effective_r0_m": self.effective_r0_m,
            "effective_response_state": "complex128_source_af",
            "r_correct_raw_m": self.r_correct_raw_m,
            "ph_correct_raw_rad": self.ph_correct_raw_rad,
            "frequency_count": int(self.frequencies_hz.size),
            "scope": self.scope.as_dict(),
            "provenance": dict(self.provenance),
        }


def build_source_af(
    observations: Sequence[Any],
    *,
    expected_count: int | None = None,
    scope: SourceAFScopeLike | None = None,
) -> tuple[SourceAFObservation, ...]:
    records = tuple(observations)
    if scope is None:
        scope = HISTORIC_SOURCE_AF_SCOPE if expected_count is None else SourceAFScope(expected_count=int(expected_count))
    if not isinstance(scope, (SourceAFScope, CamryVVTrainSourceAFScope, MultipassHHTrainSourceAFScope, TophatHHTrainValidationSourceAFScope)):
        raise TypeError("source-AF build requires an immutable SourceAFScope")
    if isinstance(scope, MultipassHHTrainSourceAFScope):
        raise TypeError("use build_multipass_source_af for the explicit TopHat multipass scope")
    if isinstance(scope, TophatHHTrainValidationSourceAFScope):
        raise TypeError("use build_tophat_train_validation_source_af for the explicit P1/P7 scope")
    if expected_count is None:
        expected_count = scope.expected_count
    elif isinstance(scope, CamryVVTrainSourceAFScope) and int(expected_count) != int(scope.expected_count):
        raise ValueError("Camry VV source-AF expected_count cannot override the scoped selected-header count")
    if expected_count is not None and len(records) != int(expected_count):
        raise ValueError(f"source-AF scope requires exactly {expected_count} observations; got {len(records)}")
    identities = tuple(getattr(record, "identity", None) for record in records)
    if any(identity is None for identity in identities):
        raise TypeError("source-AF scope requires canonical observation identities")
    keys = tuple(_id_key(identity) for identity in identities)
    if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
        raise ValueError("source-AF identities must be unique and canonically sorted")
    for record in records:
        if isinstance(record, SourceAFObservation):
            raise TypeError("source-AF conversion rejects an already-derived view")
        _validate_source_af_metadata(record, scope)
    converted = tuple(SourceAFObservation.from_observation(record, scope=scope) for record in records)
    return converted


def build_multipass_source_af(
    observations: Sequence[Any],
    *,
    expected_count: int | None = None,
    scope: MultipassHHTrainSourceAFScope | None = None,
) -> tuple[SourceAFObservation, ...]:
    """Convert an explicit multipass HH/TRAIN panel after a complete header gate.

    All identities and roles are checked before the first response is accessed.
    The conversion itself still uses the single float64/complex128 source-AF
    formula implemented by :meth:`SourceAFObservation.from_observation`.
    """

    records = tuple(observations)
    if scope is None:
        scope = MultipassHHTrainSourceAFScope(
            expected_count=None if expected_count is None else int(expected_count)
        )
    if not isinstance(scope, MultipassHHTrainSourceAFScope):
        raise TypeError("multipass source-AF build requires MultipassHHTrainSourceAFScope")
    if expected_count is None:
        expected_count = scope.expected_count
    if expected_count is not None and len(records) != int(expected_count):
        raise ValueError(f"multipass source-AF scope requires exactly {expected_count} observations; got {len(records)}")
    identities = tuple(getattr(record, "identity", None) for record in records)
    if any(identity is None for identity in identities):
        raise TypeError("multipass source-AF scope requires canonical observation identities")
    keys = tuple(_id_key(identity) for identity in identities)
    if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
        raise ValueError("multipass source-AF identities must be unique and canonically sorted")
    # Complete metadata gate before any SourceAFObservation can read response.
    for record, identity in zip(records, identities):
        _validate_source_af_metadata(record, scope)
    return tuple(SourceAFObservation.from_observation(record, scope=scope) for record in records)


def build_tophat_train_validation_source_af(
    observations: Sequence[Any],
    *,
    expected_count: int | None = None,
    scope: TophatHHTrainValidationSourceAFScope | None = None,
) -> tuple[SourceAFObservation, ...]:
    """Convert an explicitly selected P1/P7 TRAIN or validation subset once."""

    records = tuple(observations)
    if not records:
        raise ValueError("TopHat source-AF conversion requires nonempty selected records")
    scope = TophatHHTrainValidationSourceAFScope() if scope is None else scope
    if not isinstance(scope, TophatHHTrainValidationSourceAFScope):
        raise TypeError("TopHat source-AF conversion requires TophatHHTrainValidationSourceAFScope")
    identities = tuple(getattr(record, "identity", None) for record in records)
    if any(identity is None for identity in identities):
        raise TypeError("TopHat source-AF scope requires canonical observation identities")
    keys = tuple(_id_key(identity) for identity in identities)
    if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
        raise ValueError("TopHat source-AF identities must be unique and canonically sorted")
    roles = tuple(str(getattr(record, "role", "")).lower() for record in records)
    if len(set(roles)) != 1:
        raise ValueError("TopHat source-AF conversion requires one uniform TRAIN or validation role per batch")
    for record in records:
        scope.validate_header(getattr(record, "identity"), getattr(record, "role", ""))
        _validate_source_af_metadata(record, scope)
    if expected_count is not None and len(records) != int(expected_count):
        raise ValueError(f"TopHat source-AF conversion requires exactly {expected_count} observations; got {len(records)}")
    role = roles[0]
    expected_role_count = scope.expected_train_count if role == "train" else scope.expected_validation_count
    if expected_role_count is not None and len(records) != int(expected_role_count):
        raise ValueError(f"TopHat source-AF scope requires exactly {expected_role_count} {role} observations; got {len(records)}")
    return tuple(SourceAFObservation.from_observation(record, scope=scope) for record in records)


def _require_derived_records(records: Sequence[SourceAFObservation]) -> tuple[SourceAFObservation, ...]:
    result = tuple(records)
    if not result or any(not isinstance(record, SourceAFObservation) for record in result):
        raise TypeError("source-AF kernels require immutable SourceAFObservation derived views")
    if any(record.representation_tag != SOURCE_REPRESENTATION for record in result):
        raise ValueError("source-AF kernels require the explicit source_af representation tag")
    return result


def panel_points() -> np.ndarray:
    return np.asarray(
        [[x, y, 0.0] for x in PANEL_COORDINATES_M for y in PANEL_COORDINATES_M],
        dtype=np.float64,
    )


def _validate_points(points_xyz_m: Any) -> np.ndarray:
    points = np.asarray(points_xyz_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("points_xyz_m must have shape [point, 3]")
    _finite(points, "points_xyz_m")
    return points


def _is_panel(points: np.ndarray) -> bool:
    expected = panel_points()
    return points.shape == expected.shape and np.array_equal(points, expected)


def _kernel(points: np.ndarray, record: SourceAFObservation, r0_m: float) -> np.ndarray:
    one_way_range = np.linalg.norm(points - record.position_xyz_m[None, :], axis=1)
    phase = (
        -1j
        * (4.0 * np.pi / SPEED_OF_LIGHT_M_S)
        * (one_way_range[:, None] - np.float64(r0_m))
        * record.frequencies_hz[None, :]
    )
    return np.exp(phase).astype(np.complex128, copy=False)


def fixed_r0_equivalent_response(record: SourceAFObservation) -> np.ndarray:
    """Return the panel-only raw-r0 response equivalent to source-AF."""

    phase = (
        1j * np.float64(record.ph_correct_raw_rad)
        - 1j
        * (4.0 * np.pi / SPEED_OF_LIGHT_M_S)
        * record.frequencies_hz
        * np.float64(record.r_correct_raw_m)
    )
    result = record.response_raw * np.exp(phase)
    result.setflags(write=False)
    return result


def _representation_payload(record: SourceAFObservation, representation: str) -> tuple[float, np.ndarray]:
    if representation == RAW_REPRESENTATION:
        return record.raw_payload()
    if representation == SOURCE_REPRESENTATION:
        return record.source_payload()
    if representation == EQUIVALENT_REPRESENTATION:
        return record.r0_raw_m, fixed_r0_equivalent_response(record)
    raise ValueError(f"unknown correction representation: {representation}")


def weighted_contributions(
    record: SourceAFObservation,
    points_xyz_m: Any,
    *,
    representation: str,
) -> np.ndarray:
    if not isinstance(record, SourceAFObservation):
        raise TypeError("weighted contributions reject raw-labeled or duck-typed corrected observations")
    points = _validate_points(points_xyz_m)
    if representation == EQUIVALENT_REPRESENTATION and not _is_panel(points):
        raise ValueError("fixed-r0 equivalent representation is legal only on the exact 9-point panel")
    r0_m, response = _representation_payload(record, representation)
    return np.asarray(response[None, :] * np.conjugate(_kernel(points, record, r0_m)), dtype=np.complex128)


def direct_adjoint(
    records: Sequence[SourceAFObservation],
    points_xyz_m: Any,
    *,
    representation: str,
) -> np.ndarray:
    records = _require_derived_records(records)
    points = _validate_points(points_xyz_m)
    if representation == EQUIVALENT_REPRESENTATION and not _is_panel(points):
        raise ValueError("fixed-r0 equivalent adjoint is legal only on the exact 9-point panel")
    result = np.zeros(points.shape[0], dtype=np.complex128)
    for record in records:
        result += np.sum(weighted_contributions(record, points, representation=representation), axis=1)
    result.setflags(write=False)
    return result


def direct_backproject(
    records: Sequence[SourceAFObservation],
    points_xyz_m: Any,
    *,
    representation: str,
    normalization: str = "mean_native_sample",
    point_chunk_size: int = DEFAULT_POINT_CHUNK_SIZE,
    max_kernel_evaluations: int = DEFAULT_MAX_KERNEL_EVALUATIONS,
) -> np.ndarray:
    """Compute only raw or source-AF full BP; equivalent full BP is rejected."""

    if representation == EQUIVALENT_REPRESENTATION:
        raise ValueError("fixed-r0 equivalent representation cannot form a full BP/image")
    if representation not in {RAW_REPRESENTATION, SOURCE_REPRESENTATION}:
        raise ValueError(f"unknown correction representation: {representation}")
    points = _validate_points(points_xyz_m)
    records = _require_derived_records(records)
    if not records:
        raise ValueError("at least one source-AF record is required")
    if int(point_chunk_size) <= 0 or int(point_chunk_size) != point_chunk_size:
        raise ValueError("point_chunk_size must be a positive integer")
    total_samples = int(sum(record.frequencies_hz.size for record in records))
    estimate = int(points.shape[0] * total_samples)
    if estimate > int(max_kernel_evaluations):
        raise RuntimeError(f"source-AF kernel budget exceeded: estimate={estimate} max={max_kernel_evaluations}")
    result = np.zeros(points.shape[0], dtype=np.complex128)
    for start in range(0, points.shape[0], int(point_chunk_size)):
        stop = min(start + int(point_chunk_size), points.shape[0])
        chunk = points[start:stop]
        for record in records:
            result[start:stop] += np.sum(weighted_contributions(record, chunk, representation=representation), axis=1)
    if normalization == "mean_native_sample":
        result /= float(total_samples)
    elif normalization != "none":
        raise ValueError("normalization must be 'none' or 'mean_native_sample'")
    result.setflags(write=False)
    return result


def matched_unit_point_psf_budget(
    *,
    record_count: int,
    frequency_count: int,
    support_point_count: int,
    max_total_kernel_evaluations: int = 500_000_000,
    max_branch_kernel_evaluations: int = DEFAULT_MAX_KERNEL_EVALUATIONS,
) -> dict[str, Any]:
    """Ledger the matched raw/source BP and unit-point PSF kernels.

    ``N`` is the total native frequency-sample count across the selected
    records.  The ledger intentionally contains no panel, probe, fit, or
    reference kernel pass.
    """

    record_count = int(record_count)
    frequency_count = int(frequency_count)
    support_point_count = int(support_point_count)
    max_total_kernel_evaluations = int(max_total_kernel_evaluations)
    max_branch_kernel_evaluations = int(max_branch_kernel_evaluations)
    if record_count <= 0 or frequency_count <= 0 or support_point_count <= 0:
        raise ValueError("record_count, frequency_count, and support_point_count must be positive")
    if max_total_kernel_evaluations <= 0 or max_branch_kernel_evaluations <= 0:
        raise ValueError("kernel-evaluation caps must be positive")
    total_samples = record_count * frequency_count
    full_bp = support_point_count * total_samples
    unit_forward = total_samples
    unit_psf_bp = full_bp
    total = 2 * full_bp + 2 * unit_forward + 2 * unit_psf_bp
    if full_bp > max_branch_kernel_evaluations or unit_psf_bp > max_branch_kernel_evaluations:
        raise RuntimeError("source-AF full/PSF branch kernel budget exceeded")
    if total > max_total_kernel_evaluations:
        raise RuntimeError(
            "source-AF total kernel budget exceeded: "
            f"estimate={total} max={max_total_kernel_evaluations}"
        )
    return {
        "S": support_point_count,
        "N": total_samples,
        "selected_record_count": record_count,
        "frequency_count_per_record": frequency_count,
        "total_native_frequency_samples": total_samples,
        "measured_raw_bp_kernel_evaluations": full_bp,
        "measured_source_bp_kernel_evaluations": full_bp,
        "raw_unit_point_forward_kernel_evaluations": unit_forward,
        "raw_unit_point_psf_bp_kernel_evaluations": unit_psf_bp,
        "source_unit_point_forward_kernel_evaluations": unit_forward,
        "source_unit_point_psf_bp_kernel_evaluations": unit_psf_bp,
        "panel_kernel_evaluations": 0,
        "probe_kernel_evaluations": 0,
        "fit_kernel_evaluations": 0,
        "total_kernel_evaluations": total,
        "formula": "4*S*N+2*N",
        "N_definition": "selected_record_count multiplied by exact native frequency samples per record",
        "max_branch_kernel_evaluations": max_branch_kernel_evaluations,
        "max_total_kernel_evaluations": max_total_kernel_evaluations,
        "guard_passed_before_response_conversion_or_bp": True,
    }


def _unit_point_r0(record: SourceAFObservation, representation: str) -> float:
    if representation == RAW_REPRESENTATION:
        return float(record.r0_raw_m)
    if representation == SOURCE_REPRESENTATION:
        return float(record.effective_r0_m)
    raise ValueError("matched unit-point PSF accepts only raw or source-AF representations")


def matched_unit_point_psf(
    records: Sequence[SourceAFObservation],
    points_xyz_m: Any,
    unit_point_xyz_m: Any,
    *,
    representation: str,
    normalization: str = "mean_native_sample",
    point_chunk_size: int = DEFAULT_POINT_CHUNK_SIZE,
    max_kernel_evaluations: int = DEFAULT_MAX_KERNEL_EVALUATIONS,
) -> np.ndarray:
    """Compute a matched unit-point PSF using the same native operator.

    The synthetic unit response is generated from each selected record's
    position, exact frequencies, and representation-specific native ``r0``.
    No measured response is read or used by this helper.
    """

    if representation not in {RAW_REPRESENTATION, SOURCE_REPRESENTATION}:
        raise ValueError("matched unit-point PSF accepts only raw or source-AF representations")
    records = _require_derived_records(records)
    points = _validate_points(points_xyz_m)
    unit_point = np.asarray(unit_point_xyz_m, dtype=np.float64)
    if unit_point.shape != (3,):
        raise ValueError("unit_point_xyz_m must have shape [3]")
    _finite(unit_point, "unit_point_xyz_m")
    if int(point_chunk_size) <= 0 or int(point_chunk_size) != point_chunk_size:
        raise ValueError("point_chunk_size must be a positive integer")
    total_samples = int(sum(record.frequencies_hz.size for record in records))
    bp_estimate = int(points.shape[0] * total_samples)
    if bp_estimate > int(max_kernel_evaluations):
        raise RuntimeError(
            "source-AF matched unit-point PSF kernel budget exceeded: "
            f"estimate={bp_estimate} max={max_kernel_evaluations}"
        )
    result = np.zeros(points.shape[0], dtype=np.complex128)
    target = unit_point[None, :]
    for record in records:
        r0_m = _unit_point_r0(record, representation)
        # One exact forward kernel per selected record; this is the only
        # synthetic response and is deliberately independent of measured fp.
        unit_response = _kernel(target, record, r0_m)[0]
        for start in range(0, points.shape[0], int(point_chunk_size)):
            stop = min(start + int(point_chunk_size), points.shape[0])
            result[start:stop] += np.sum(
                unit_response[None, :] * np.conjugate(_kernel(points[start:stop], record, r0_m)),
                axis=1,
            )
    if normalization == "mean_native_sample":
        result /= float(total_samples)
    elif normalization != "none":
        raise ValueError("normalization must be 'none' or 'mean_native_sample'")
    result.setflags(write=False)
    return result


def estimate_kernel_evaluations(records: Sequence[SourceAFObservation], point_count: int) -> int:
    return int(int(point_count) * sum(record.frequencies_hz.size for record in records))


def budget_report(
    records: Sequence[SourceAFObservation],
    *,
    full_point_count: int,
    panel_point_count: int = 9,
    max_total_kernel_evaluations: int = 500_000_000,
    max_branch_kernel_evaluations: int = DEFAULT_MAX_KERNEL_EVALUATIONS,
) -> dict[str, Any]:
    total_samples = int(sum(record.frequencies_hz.size for record in records))
    raw_full = int(full_point_count) * total_samples
    source_full = raw_full
    panel_one = int(panel_point_count) * total_samples
    panel_two = 2 * panel_one
    total = raw_full + source_full + panel_two
    if raw_full > max_branch_kernel_evaluations or source_full > max_branch_kernel_evaluations:
        raise RuntimeError("source-AF full-branch kernel budget exceeded")
    if total > max_total_kernel_evaluations:
        raise RuntimeError("source-AF total kernel budget exceeded")
    return {
        "total_native_frequency_samples": total_samples,
        "raw_full_branch_kernel_evaluations": raw_full,
        "source_full_branch_kernel_evaluations": source_full,
        "panel_two_representation_kernel_evaluations": panel_two,
        "total_kernel_evaluations": total,
        "max_full_branch_kernel_evaluations": int(max_branch_kernel_evaluations),
        "max_total_kernel_evaluations": int(max_total_kernel_evaluations),
        "guard_passed_before_correction_or_bp": True,
    }


def unit_modulus_sample_energy(records: Sequence[SourceAFObservation]) -> dict[str, Any]:
    records = _require_derived_records(records)
    factors = np.asarray([np.exp(1j * record.ph_correct_raw_rad) for record in records], dtype=np.complex128)
    raw_energy = float(sum(np.vdot(record.response_raw, record.response_raw).real for record in records))
    source_energy = float(sum(np.vdot(record.response_source, record.response_source).real for record in records))
    relative_energy_delta = abs(source_energy - raw_energy) / max(raw_energy, np.finfo(np.float64).tiny)
    max_unit_modulus_error = float(np.max(np.abs(np.abs(factors) - 1.0)))
    return {
        "observation_count": int(len(records)),
        "sample_count": int(sum(record.response_raw.size for record in records)),
        "max_unit_modulus_abs_error": max_unit_modulus_error,
        "raw_response_energy": raw_energy,
        "source_response_energy": source_energy,
        "relative_sample_energy_delta": float(relative_energy_delta),
        "passed": bool(max_unit_modulus_error <= 2.0e-15 and relative_energy_delta <= 2.0e-14),
    }


def correction_stats(records: Sequence[SourceAFObservation]) -> dict[str, Any]:
    records = _require_derived_records(records)
    r_values = np.asarray([record.r_correct_raw_m for record in records], dtype=np.float64)
    ph_values = np.asarray([record.ph_correct_raw_rad for record in records], dtype=np.float64)
    _finite(r_values, "r_correct_raw_m")
    _finite(ph_values, "ph_correct_raw_rad")
    return {
        "count": int(len(records)),
        "r_correct_raw_m": {"minimum": float(np.min(r_values)), "maximum": float(np.max(r_values)), "mean": float(np.mean(r_values))},
        "ph_correct_raw_rad": {"minimum": float(np.min(ph_values)), "maximum": float(np.max(ph_values)), "mean": float(np.mean(ph_values))},
        "source_formula": SOURCE_AF_FORMULA,
        "fixed_r0_equivalent_formula": FIXED_R0_EQUIVALENT_FORMULA,
    }


def _error_metrics(
    left: np.ndarray,
    right: np.ndarray,
    *,
    relative_tolerance: float,
    scaled_absolute_tolerance: float,
    reference_scale: float | None = None,
    reference_scale_kind: str = "pair_vector_l2_floor_1",
) -> dict[str, Any]:
    difference = np.asarray(left, dtype=np.complex128) - np.asarray(right, dtype=np.complex128)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    scale = max(
        float(reference_scale) if reference_scale is not None else left_norm,
        float(reference_scale) if reference_scale is not None else right_norm,
        1.0,
    )
    relative_l2 = float(np.linalg.norm(difference) / scale)
    absolute_max = float(np.max(np.abs(difference))) if difference.size else 0.0
    amplitude_scale = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1.0)
    scaled_absolute_max = absolute_max / amplitude_scale
    return {
        "left_l2": left_norm,
        "right_l2": right_norm,
        "reference_scale": float(scale),
        "reference_scale_kind": reference_scale_kind,
        "relative_l2_error": relative_l2,
        "absolute_max_error": absolute_max,
        "scaled_absolute_max_error": float(scaled_absolute_max),
        "relative_tolerance": float(relative_tolerance),
        "scaled_absolute_tolerance": float(scaled_absolute_tolerance),
        "passed": bool(relative_l2 <= relative_tolerance and scaled_absolute_max <= scaled_absolute_tolerance),
    }


def panel_equivalence_report(
    records: Sequence[SourceAFObservation],
    *,
    relative_tolerance: float = DEFAULT_RELATIVE_TOLERANCE,
    scaled_absolute_tolerance: float = DEFAULT_SCALED_ABSOLUTE_TOLERANCE,
) -> dict[str, Any]:
    records = _require_derived_records(records)
    points = panel_points()
    source_parts = []
    equivalent_parts = []
    for record in records:
        source_parts.append(weighted_contributions(record, points, representation=SOURCE_REPRESENTATION))
        equivalent_parts.append(weighted_contributions(record, points, representation=EQUIVALENT_REPRESENTATION))
    source_contributions = np.concatenate([part.reshape(-1) for part in source_parts])
    equivalent_contributions = np.concatenate([part.reshape(-1) for part in equivalent_parts])
    contribution_reference_scale = max(
        float(np.linalg.norm(source_contributions)),
        float(np.linalg.norm(equivalent_contributions)),
        1.0,
    )
    # Sum the already-computed contribution arrays. This is the corresponding
    # direct adjoint without a second panel kernel pass.
    source_adjoint = np.zeros(points.shape[0], dtype=np.complex128)
    equivalent_adjoint = np.zeros(points.shape[0], dtype=np.complex128)
    for source_part, equivalent_part in zip(source_parts, equivalent_parts):
        source_adjoint += np.sum(source_part, axis=1)
        equivalent_adjoint += np.sum(equivalent_part, axis=1)
    return {
        "panel_point_count": int(points.shape[0]),
        "panel_points_m": points.tolist(),
        "representation_computations": 2,
        "comparison_quantity": "complex weighted contributions response*conjugate(kernel), then direct adjoint sums",
        "response_equality_comparison": False,
        "contribution_reference_scale": contribution_reference_scale,
        "contribution_metrics": _error_metrics(
            source_contributions,
            equivalent_contributions,
            relative_tolerance=relative_tolerance,
            scaled_absolute_tolerance=scaled_absolute_tolerance,
            reference_scale=contribution_reference_scale,
            reference_scale_kind="contribution_vector_l2_floor_1",
        ),
        "adjoint_metrics": _error_metrics(
            source_adjoint,
            equivalent_adjoint,
            relative_tolerance=relative_tolerance,
            scaled_absolute_tolerance=scaled_absolute_tolerance,
            reference_scale=contribution_reference_scale,
            reference_scale_kind="already_computed_contribution_vector_l2_floor_1",
        ),
        "fixture_justification": "predeclared float64 geometry/r0/frequency and complex128 correction/accumulation; tolerances are local to this equivalence check and are not borrowed from an unrelated dot test",
    }


def change_diagnostics(raw_bp: np.ndarray, source_bp: np.ndarray) -> dict[str, Any]:
    raw = np.asarray(raw_bp, dtype=np.complex128)
    source = np.asarray(source_bp, dtype=np.complex128)
    if raw.shape != source.shape:
        raise ValueError("raw/source BP shapes must match")
    delta = source - raw
    raw_energy = float(np.vdot(raw, raw).real)
    source_energy = float(np.vdot(source, source).real)
    correlation = np.vdot(raw, source) / max(np.sqrt(raw_energy * source_energy), np.finfo(np.float64).tiny)
    return {
        "interpretation": "change diagnostics only; not accuracy, focus, geometry, or registration scores",
        "raw_l2": float(np.linalg.norm(raw)),
        "source_l2": float(np.linalg.norm(source)),
        "delta_l2": float(np.linalg.norm(delta)),
        "relative_delta_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(raw), np.finfo(np.float64).tiny)),
        "raw_energy": raw_energy,
        "source_energy": source_energy,
        "complex_correlation_abs": float(abs(correlation)) if raw_energy and source_energy else None,
        "raw_peak_abs": float(np.max(np.abs(raw))) if raw.size else 0.0,
        "source_peak_abs": float(np.max(np.abs(source))) if source.size else 0.0,
    }


__all__ = [
    "DEFAULT_MAX_KERNEL_EVALUATIONS",
    "DEFAULT_POINT_CHUNK_SIZE",
    "DEFAULT_RELATIVE_TOLERANCE",
    "DEFAULT_SCALED_ABSOLUTE_TOLERANCE",
    "EQUIVALENT_REPRESENTATION",
    "FIXED_R0_EQUIVALENT_FORMULA",
    "HISTORIC_SOURCE_AF_SCOPE",
    "CamryVVTrainSourceAFScope",
    "MultipassHHTrainSourceAFScope",
    "TophatHHTrainValidationSourceAFScope",
    "PANEL_COORDINATES_M",
    "RAW_REPRESENTATION",
    "SOURCE_AF_FORMULA",
    "SOURCE_AF_SCHEMA",
    "SOURCE_REPRESENTATION",
    "SourceAFScope",
    "SourceAFScopeLike",
    "SourceAFObservation",
    "build_source_af",
    "build_multipass_source_af",
    "build_tophat_train_validation_source_af",
    "budget_report",
    "change_diagnostics",
    "correction_stats",
    "direct_adjoint",
    "direct_backproject",
    "estimate_kernel_evaluations",
    "fixed_r0_equivalent_response",
    "panel_equivalence_report",
    "panel_points",
    "matched_unit_point_psf",
    "matched_unit_point_psf_budget",
    "unit_modulus_sample_energy",
    "weighted_contributions",
]
