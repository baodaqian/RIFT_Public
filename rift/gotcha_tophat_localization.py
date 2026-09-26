"""Synthetic-first, fail-closed TopHat multipass localization readiness.

This module is intentionally separate from the completed Camry readiness path
and from the data-free two-cube extraction contract.  It provides only bounded
native-coordinate mechanics: a qualified diagram XY seed, a predeclared HH
TRAIN panel, aspect/elevation geometry diagnostics, a conditional native-Z
search, and synthetic cylinder/ring fixtures.

The diagram seed is not a surveyed native cube center.  A conditional native-Z
minimum is not an absolute height above ground.  The supplied HH source-AF
correction is treated as fixed input; this module never refits per-record range
corrections, frequency gains, or unconstrained 3-D scatterers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType, SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from .gotcha_acquisition import (
        AUTOFOCUS_PUBLISHED,
        AUTOFOCUS_PROVENANCE_SCHEMA,
        NativeObservation,
        NativeObservationId,
        PhaseReferenceContract,
        AutofocusProvenance,
        SPEED_OF_LIGHT_M_S,
    )
    from .gotcha_source_af import (
        MultipassHHTrainSourceAFScope,
        SourceAFObservation,
        _kernel as _source_af_kernel,
        build_multipass_source_af,
    )
except ImportError:  # Direct loading used by local validators.
    from gotcha_acquisition import (  # type: ignore
        AUTOFOCUS_PUBLISHED,
        AUTOFOCUS_PROVENANCE_SCHEMA,
        NativeObservation,
        NativeObservationId,
        PhaseReferenceContract,
        AutofocusProvenance,
        SPEED_OF_LIGHT_M_S,
    )
    from gotcha_source_af import (  # type: ignore
        MultipassHHTrainSourceAFScope,
        SourceAFObservation,
        _kernel as _source_af_kernel,
        build_multipass_source_af,
    )


TOPHAT_LOCALIZATION_SCHEMA = "rift_gotcha_step3_tophat_train_localization_readiness_v1"
TOPHAT_TARGET_ID = "tophat"
TOPHAT_PANEL_PASSES = tuple(range(1, 9))
TOPHAT_PANEL_POLARIZATION = "hh"
TOPHAT_PANEL_SECTORS = (2, 92, 182, 272)
TOPHAT_DIAGRAM_LABEL = "LTH1"
TOPHAT_DIAGRAM_FEET_TO_M = 0.3048
TOPHAT_DIAGRAM_ROTATION_DEG = 8.83045
TOPHAT_DIAGRAM_TRANSLATION_XY_M = (-30.61589, 17.02769)
TOPHAT_DIAGRAM_XY_M = (-16.0244, 22.3791)
TOPHAT_DIAGRAM_ANCHOR_COUNT = 7
TOPHAT_DIAGRAM_ANCHOR_RMSE_M = 1.778
TOPHAT_DIAGRAM_MAX_ANCHOR_RESIDUAL_M = 2.635
TOPHAT_DIAGRAM_LOO_X_RANGE_M = (-16.670, -15.722)
TOPHAT_DIAGRAM_LOO_Y_RANGE_M = (22.159, 22.511)

BLOCKED_MISSING_DECLARED_NATIVE_Z_SEARCH_INTERVAL = "BLOCKED_MISSING_DECLARED_NATIVE_Z_SEARCH_INTERVAL"
BLOCKED_MISSING_NATIVE_Z_DATUM = "BLOCKED_MISSING_NATIVE_Z_DATUM"
BLOCKED_RESOURCE_CAP_EXCEEDED = "BLOCKED_RESOURCE_CAP_EXCEEDED"
BLOCKED_NATIVE_CUBE_PLACEMENT_UNRESOLVED = "BLOCKED_NATIVE_CUBE_PLACEMENT_UNRESOLVED"
INCONCLUSIVE_INSUFFICIENT_ELEVATION_GEOMETRY = "INCONCLUSIVE_INSUFFICIENT_ELEVATION_GEOMETRY"
INCONCLUSIVE_ASPECT_DEPENDENT_EVIDENCE = "INCONCLUSIVE_ASPECT_DEPENDENT_EVIDENCE"
INCONCLUSIVE_NO_RESPONSE_LOCALIZATION = "INCONCLUSIVE_NO_RESPONSE_LOCALIZATION"
INCONCLUSIVE_ZERO_RECEIVED_ENERGY = "INCONCLUSIVE_ZERO_RECEIVED_ENERGY"
INCONCLUSIVE_FLAT_EVIDENCE = "INCONCLUSIVE_FLAT_EVIDENCE"
INCONCLUSIVE_BOUNDARY_CANDIDATE = "INCONCLUSIVE_BOUNDARY_CANDIDATE"
INCONCLUSIVE_MODEL_MISMATCH = "INCONCLUSIVE_MODEL_MISMATCH"
SYNTHETIC_STENCIL_DISCRIMINATIVE_PROFILE = "SYNTHETIC_STENCIL_DISCRIMINATIVE_PROFILE"
CONDITIONAL_NATIVE_Z_UNREGISTERED = "CONDITIONAL_NATIVE_Z_UNREGISTERED"
CONDITIONAL_XY_FEASIBLE_SET_UNREGISTERED = "CONDITIONAL_XY_FEASIBLE_SET_UNREGISTERED"
ABSOLUTE_HEIGHT_STATUS_MISSING_DATUM = BLOCKED_MISSING_NATIVE_Z_DATUM
ABSOLUTE_HEIGHT_STATUS_SOURCE_DATUM_ONLY = "SOURCE_BACKED_DATUM_AVAILABLE_REGISTRATION_REVIEW_REQUIRED"

MAX_CONDITIONAL_POINT_COUNT = 4096
MAX_CONDITIONAL_Z_SAMPLES = 81
DEFAULT_MAX_KERNEL_EVALUATIONS = 20_000_000
DEFAULT_MIN_ELEVATION_GROUPS = 2
DEFAULT_MIN_ELEVATION_SPAN_DEG = 0.5
DEFAULT_MIN_ANTENNA_VERTICAL_SPAN_M = 0.5
DEFAULT_MAX_GEOMETRY_CONDITION = 1.0e8
PROVISIONAL_NATIVE_Z_SEARCH_INTERVAL_M = (-2.0, 3.0)
PROVISIONAL_NATIVE_Z_SEARCH_INTERVAL_STATUS = "DECLARED_SEARCH_ASSUMPTION_NOT_SURVEYED_DATUM"
SYNTHETIC_RADIAL_BOUNDS_M = (0.75, 1.25)
SYNTHETIC_ALPHA_BOUNDS_DEG = (-30.0, 30.0)
SYNTHETIC_SHARED_ZETA_BOUNDS_M = (-0.5, 0.5)
SYNTHETIC_ROBUST_QUORUM_FRACTION = 0.75
SYNTHETIC_MIN_ROBUST_QUORUM = 3
SYNTHETIC_RESPONSE_FREQUENCY_COUNTS_BY_PASS = {
    1: 424,
    2: 426,
    3: 428,
    4: 428,
    5: 428,
    6: 430,
    7: 434,
    8: 432,
}
SYNTHETIC_RESPONSE_FREQUENCY_ENDPOINTS_BY_PASS_HZ = {
    1: (9.288080384e9, 9.910440960e9),
    2: (9.288080384e9, 9.910448128e9),
    3: (9.288080384e9, 9.910455296e9),
    4: (9.288080384e9, 9.910455296e9),
    5: (9.288080384e9, 9.910455296e9),
    6: (9.288080384e9, 9.910461440e9),
    7: (9.288080384e9, 9.910474752e9),
    8: (9.288080384e9, 9.910468608e9),
}
SYNTHETIC_RESPONSE_FREQUENCY_ENDPOINTS_HZ = SYNTHETIC_RESPONSE_FREQUENCY_ENDPOINTS_BY_PASS_HZ[1]
SYNTHETIC_RESPONSE_SHARD_COUNTS_BY_PASS_SECTOR = {
    1: {2: 117, 92: 117, 182: 117, 272: 118},
    2: {2: 118, 92: 118, 182: 118, 272: 118},
    3: {2: 118, 92: 118, 182: 118, 272: 118},
    4: {2: 118, 92: 118, 182: 118, 272: 119},
    5: {2: 119, 92: 118, 182: 118, 272: 118},
    6: {2: 119, 92: 119, 182: 118, 272: 119},
    7: {2: 120, 92: 120, 182: 120, 272: 120},
    8: {2: 120, 92: 119, 182: 119, 272: 120},
}
SYNTHETIC_RESPONSE_STENCIL_CENTER_XYZ_M = (TOPHAT_DIAGRAM_XY_M[0], TOPHAT_DIAGRAM_XY_M[1], 0.5)
SYNTHETIC_RESPONSE_STENCIL_OFFSETS_XY_M = (-3.0, -1.5, 0.0, 1.5, 3.0)
SYNTHETIC_RESPONSE_STENCIL_Z_M = tuple(float(value) for value in np.linspace(-2.0, 3.0, 7))
SYNTHETIC_RESPONSE_STENCIL_OFFSETS_M = (-0.20, 0.0, 0.20)
SYNTHETIC_RESPONSE_CANDIDATE_COUNT = 175
SYNTHETIC_RESPONSE_MAX_MODEL_MISMATCH_LOSS = 1.0e-3
SYNTHETIC_RESPONSE_MIN_DISCRIMINATION_RELATIVE_GAP = 1.0e-3
SYNTHETIC_RESPONSE_MIN_RECEIVED_ENERGY = 1.0e-20
SYNTHETIC_RESPONSE_WORK_CAP = 65_000_000

# The measured-response screen is a future archive binding.  These constants
# are deliberately separate from the synthetic response package above: no
# synthetic rho/alpha/zeta model, truth generator, or H-cache is reused here.
MEASURED_RESPONSE_SCREEN_SCHEMA = "rift_gotcha_step3_tophat_measured_response_screen_v1"
MEASURED_RESPONSE_SCREEN_ARCHIVE_SOURCE = "converted_v3_joint8_fullpol/shards/pass{1..8}_hh.npz"
MEASURED_RESPONSE_SCREEN_SCENE = "gotcha_v1_joint8_fullpol"
MEASURED_RESPONSE_SCREEN_PASSES = (1, 7)
MEASURED_RESPONSE_SCREEN_ALL_HH_PASSES = tuple(range(1, 9))
MEASURED_RESPONSE_SCREEN_SECTORS = TOPHAT_PANEL_SECTORS
MEASURED_RESPONSE_SCREEN_POLARIZATION = "hh"
MEASURED_RESPONSE_SCREEN_ROLE = "train"
MEASURED_RESPONSE_SCREEN_COUNT_MATRIX = {
    1: {2: 117, 92: 117, 182: 117, 272: 118},
    7: {2: 120, 92: 120, 182: 120, 272: 120},
}
MEASURED_RESPONSE_SCREEN_FREQUENCY_COUNTS_BY_PASS = {1: 424, 7: 434}
MEASURED_RESPONSE_SCREEN_PASS_ROW_COUNTS = {1: 469, 7: 480}
MEASURED_RESPONSE_SCREEN_TOTAL_RECORD_COUNT = 949
MEASURED_RESPONSE_SCREEN_TOTAL_FREQUENCY_SAMPLES = 407_176
MEASURED_RESPONSE_SCREEN_XY_OFFSETS_M = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
MEASURED_RESPONSE_SCREEN_Z_SAMPLES_M = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
MEASURED_RESPONSE_SCREEN_CANDIDATE_COUNT = 294
MEASURED_RESPONSE_SCREEN_FINE_CANDIDATE_COUNT = 512
MEASURED_RESPONSE_SCREEN_CUBE_XY_OFFSETS_M = (-1.0, 0.0, 1.0)
MEASURED_RESPONSE_SCREEN_CUBE_Z_OFFSETS_M = (0.0, 1.0)
MEASURED_RESPONSE_SCREEN_CUBE_SIDE_M = 4.0
MEASURED_RESPONSE_SCREEN_PANEL_COUNT = 8
MEASURED_RESPONSE_SCREEN_MAP_KERNEL_TERMS = 119_709_744
MEASURED_RESPONSE_SCREEN_STAGE_A_MAP_TERMS = 119_709_744
MEASURED_RESPONSE_SCREEN_STAGE_B_FINE_MAP_TERMS = 208_474_112
MEASURED_RESPONSE_SCREEN_UNION_FORWARD_PSF_TERMS = 328_591_032
MEASURED_RESPONSE_SCREEN_OFFGRID_REFERENCE_TERMS = 91_207_424
MEASURED_RESPONSE_SCREEN_RAW_SOURCE_BRIDGE_TERMS = 7_329_168
MEASURED_RESPONSE_SCREEN_TOTAL_DIRECT_KERNEL_TERMS = 755_311_480
MEASURED_RESPONSE_SCREEN_WORK_CAP = 800_000_000
MEASURED_RESPONSE_SCREEN_STATUS_PENDING_GEOMETRY_RULE = "LOCAL_PREPARATION_READY_FOR_MEASURED_STAGE_A_B"
INCONCLUSIVE_MEASURED_SCREEN_HEADER_INPUT = "INCONCLUSIVE_MEASURED_SCREEN_HEADER_INPUT"
INCONCLUSIVE_MEASURED_SCREEN_NO_COHERENT_SUPPORT = "INCONCLUSIVE_MEASURED_SCREEN_NO_COHERENT_SUPPORT"
INCONCLUSIVE_MEASURED_SCREEN_BOUNDARY_OR_COVERAGE = "INCONCLUSIVE_MEASURED_SCREEN_BOUNDARY_OR_COVERAGE"
INCONCLUSIVE_MEASURED_SCREEN_SCENE_OR_ALIAS_AMBIGUITY = "INCONCLUSIVE_MEASURED_SCREEN_SCENE_OR_ALIAS_AMBIGUITY"
INCONCLUSIVE_MEASURED_SCREEN_ASPECT_SUPPORT = "INCONCLUSIVE_MEASURED_SCREEN_ASPECT_DEPENDENT_SUPPORT"
INCONCLUSIVE_MEASURED_SCREEN_OPERATOR_LIMITATION = "INCONCLUSIVE_MEASURED_SCREEN_OPERATOR_RESOLUTION_LIMITATION"
QUALIFIED_CONDITIONAL_4M_WORKING_CUBE_UNREGISTERED = "QUALIFIED_CONDITIONAL_4M_WORKING_CUBE_UNREGISTERED"


class TophatLocalizationBlocked(RuntimeError):
    """A deliberate fail-closed readiness decision."""

    def __init__(self, status: str, message: str | None = None) -> None:
        self.status = str(status)
        super().__init__(message or self.status)


def _readonly(value: Any, dtype: Any = np.float64) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _finite(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return array


def _freeze(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): value[key] for key in value})


def _identity_key(identity: Any) -> tuple[int, str, int, int]:
    try:
        return (
            int(identity.pass_id),
            str(identity.polarization).lower(),
            int(identity.sector_id),
            int(identity.pulse_index),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError("canonical identity must expose pass/polarization/sector/pulse") from exc


@dataclass(frozen=True)
class TophatDiagramXYSeed:
    """Qualified LTH1 diagram-derived XY candidate, with no Z or R/t."""

    xy_m: np.ndarray = field(default_factory=lambda: _readonly(TOPHAT_DIAGRAM_XY_M))
    label: str = TOPHAT_DIAGRAM_LABEL
    source: str = "GOTCHA_OFFICIAL_DATA_DESCRIPTION_20260909_embedded_calibration_targets_diagram"
    feet_to_m: float = TOPHAT_DIAGRAM_FEET_TO_M
    rigid_rotation_deg: float = TOPHAT_DIAGRAM_ROTATION_DEG
    rigid_translation_xy_m: np.ndarray = field(default_factory=lambda: _readonly(TOPHAT_DIAGRAM_TRANSLATION_XY_M))
    anchor_count: int = TOPHAT_DIAGRAM_ANCHOR_COUNT
    anchor_rmse_m: float = TOPHAT_DIAGRAM_ANCHOR_RMSE_M
    max_anchor_residual_m: float = TOPHAT_DIAGRAM_MAX_ANCHOR_RESIDUAL_M
    loo_x_range_m: tuple[float, float] = TOPHAT_DIAGRAM_LOO_X_RANGE_M
    loo_y_range_m: tuple[float, float] = TOPHAT_DIAGRAM_LOO_Y_RANGE_M
    loo_interpretation: str = "sensitivity_not_confidence_interval"
    native_placement_status: str = "missing_numeric_native_R_t"
    z_status: str = "unresolved_no_native_height_datum"

    def __post_init__(self) -> None:
        xy = _readonly(self.xy_m)
        translation = _readonly(self.rigid_translation_xy_m)
        if xy.shape != (2,):
            raise ValueError("TopHat diagram XY seed must have shape [2]")
        if translation.shape != (2,):
            raise ValueError("diagram rigid translation must have shape [2]")
        _finite(xy, "diagram XY seed")
        _finite(translation, "diagram translation")
        if self.label != TOPHAT_DIAGRAM_LABEL:
            raise ValueError("only the qualified LTH1 TopHat diagram seed is permitted")
        if not str(self.source) or self.feet_to_m != TOPHAT_DIAGRAM_FEET_TO_M:
            raise ValueError("diagram seed provenance must retain the 0.3048 m/ft map")
        if not np.isfinite(self.rigid_rotation_deg) or not np.isfinite(self.anchor_rmse_m):
            raise ValueError("diagram seed provenance must be finite")
        if self.anchor_count != TOPHAT_DIAGRAM_ANCHOR_COUNT:
            raise ValueError("diagram seed must retain all seven surveyed anchors")
        if self.loo_interpretation != "sensitivity_not_confidence_interval":
            raise ValueError("leave-one-out ranges are sensitivity, not confidence intervals")
        if self.native_placement_status != "missing_numeric_native_R_t":
            raise ValueError("diagram seed cannot declare a native rigid placement")
        if self.z_status != "unresolved_no_native_height_datum":
            raise ValueError("diagram seed cannot declare a numeric native Z")
        object.__setattr__(self, "xy_m", xy)
        object.__setattr__(self, "rigid_translation_xy_m", translation)

    @property
    def loo_envelope_xy_m(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (tuple(self.loo_x_range_m), tuple(self.loo_y_range_m))

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "xy_m": self.xy_m.tolist(),
            "source": self.source,
            "feet_to_m": self.feet_to_m,
            "rigid_rotation_deg": self.rigid_rotation_deg,
            "rigid_translation_xy_m": self.rigid_translation_xy_m.tolist(),
            "anchor_count": self.anchor_count,
            "anchor_rmse_m": self.anchor_rmse_m,
            "max_anchor_residual_m": self.max_anchor_residual_m,
            "loo_x_range_m": list(self.loo_x_range_m),
            "loo_y_range_m": list(self.loo_y_range_m),
            "loo_interpretation": self.loo_interpretation,
            "native_placement_status": self.native_placement_status,
            "z_status": self.z_status,
        }


DEFAULT_TOPHAT_DIAGRAM_XY_SEED = TophatDiagramXYSeed()


@dataclass(frozen=True)
class TophatNativeHeightDatum:
    """Optional source-backed datum for absolute-height claims only."""

    native_z_m: float | None = None
    source: str | None = None
    source_backed: bool = False
    status: str = "missing_source_backed_native_z_datum"

    def __post_init__(self) -> None:
        if self.native_z_m is not None and not np.isfinite(float(self.native_z_m)):
            raise ValueError("native height datum must be finite when supplied")
        if self.source_backed and (self.native_z_m is None or not str(self.source)):
            raise ValueError("a source-backed height datum requires numeric native Z and provenance")
        if not self.source_backed and self.status != "missing_source_backed_native_z_datum":
            raise ValueError("an absent height datum must retain its blocked status")
        if self.source_backed and self.status == "missing_source_backed_native_z_datum":
            object.__setattr__(self, "status", "source_backed_native_z_datum_available")

    @classmethod
    def missing(cls) -> "TophatNativeHeightDatum":
        return cls()

    @property
    def available(self) -> bool:
        return bool(self.source_backed and self.native_z_m is not None)

    def require_absolute_height(self) -> float:
        if not self.available:
            raise TophatLocalizationBlocked(
                BLOCKED_MISSING_NATIVE_Z_DATUM,
                "absolute height/registration is blocked without a source-backed native Z datum",
            )
        return float(self.native_z_m)

    def as_dict(self) -> dict[str, Any]:
        return {
            "native_z_m": None if self.native_z_m is None else float(self.native_z_m),
            "source": self.source,
            "source_backed": self.source_backed,
            "status": self.status,
        }


@dataclass(frozen=True)
class TophatNativeZSearchInterval:
    """Finite native-Z interval for conditional, unregistered search."""

    lower_m: float
    upper_m: float
    spacing_m: float
    name: str = "injected_synthetic_native_z_interval"

    def __post_init__(self) -> None:
        lower = float(self.lower_m)
        upper = float(self.upper_m)
        spacing = float(self.spacing_m)
        if not np.isfinite([lower, upper, spacing]).all() or not lower < upper or spacing <= 0:
            raise ValueError("native-Z search interval must be finite with lower < upper and positive spacing")
        count = int(np.ceil((upper - lower) / spacing - 1.0e-12)) + 1
        if count > MAX_CONDITIONAL_Z_SAMPLES:
            raise ValueError("native-Z search interval exceeds the bounded sample cap")
        if not str(self.name):
            raise ValueError("native-Z search interval name must be nonempty")
        object.__setattr__(self, "lower_m", lower)
        object.__setattr__(self, "upper_m", upper)
        object.__setattr__(self, "spacing_m", spacing)

    @property
    def values_m(self) -> np.ndarray:
        regular_count = int(np.floor((self.upper_m - self.lower_m) / self.spacing_m + 1.0e-12)) + 1
        values = self.lower_m + self.spacing_m * np.arange(regular_count, dtype=np.float64)
        if values[-1] < self.upper_m - 1.0e-12:
            values = np.append(values, self.upper_m)
        return _readonly(values)

    @property
    def sample_count(self) -> int:
        return int(self.values_m.size)

    def as_dict(self) -> dict[str, Any]:
        return {
            "lower_m": self.lower_m,
            "upper_m": self.upper_m,
            "spacing_m": self.spacing_m,
            "sample_count": self.sample_count,
            "name": self.name,
            "conditional_unregistered_only": True,
        }


def provisional_tophat_native_z_search_interval() -> TophatNativeZSearchInterval:
    """Return the prospective [-2,+3] m breadth assumption, never a datum."""

    return TophatNativeZSearchInterval(
        -2.0,
        3.0,
        0.1,
        name=PROVISIONAL_NATIVE_Z_SEARCH_INTERVAL_STATUS,
    )


@dataclass(frozen=True)
class TophatMultipassPanel:
    """Frozen future geometry-panel input; it makes no data-existence claim."""

    pass_ids: tuple[int, ...] = TOPHAT_PANEL_PASSES
    polarization: str = TOPHAT_PANEL_POLARIZATION
    sector_ids: tuple[int, ...] = TOPHAT_PANEL_SECTORS
    role: str = "train"

    def __post_init__(self) -> None:
        pass_ids = tuple(sorted({int(value) for value in self.pass_ids}))
        sectors = tuple(sorted({int(value) for value in self.sector_ids}))
        if pass_ids != TOPHAT_PANEL_PASSES:
            raise ValueError("TopHat panel design must declare passes 1..8")
        if str(self.polarization).lower() != TOPHAT_PANEL_POLARIZATION:
            raise ValueError("TopHat panel design is explicitly HH/TRAIN")
        if sectors != TOPHAT_PANEL_SECTORS:
            raise ValueError("TopHat panel design must declare sectors 002/092/182/272")
        if str(self.role).lower() != "train":
            raise ValueError("TopHat panel design is TRAIN-only")
        object.__setattr__(self, "pass_ids", pass_ids)
        object.__setattr__(self, "polarization", TOPHAT_PANEL_POLARIZATION)
        object.__setattr__(self, "sector_ids", sectors)
        object.__setattr__(self, "role", "train")

    def accepts(self, identity: Any) -> bool:
        key = _identity_key(identity)
        return key[0] in self.pass_ids and key[1] == self.polarization and key[2] in self.sector_ids

    def as_dict(self) -> dict[str, Any]:
        return {
            "pass_ids": list(self.pass_ids),
            "polarization": self.polarization,
            "sector_ids": list(self.sector_ids),
            "role": self.role,
            "identity_key": "(pass_id, polarization, sector_id, pulse_index)",
            "data_existence_claim": False,
            "validation_and_test": "not selected; TEST remains sealed",
        }


DEFAULT_TOPHAT_MULTIPASS_PANEL = TophatMultipassPanel()


@dataclass(frozen=True)
class TwoCubeMeasurementExteriorPolicy:
    """Shared native-complex/exterior policy for the two-cube readiness stage."""

    target_ids: tuple[str, str] = ("tophat", "toyota_camry")
    representation: str = "native_complex"
    transform_name: str = "identity"
    apply_to_data: bool = True
    apply_to_predictions: bool = True
    crop: bool = False
    subtraction: bool = False
    padding: bool = False
    magnitude_conversion: bool = False
    exterior_scene_retained: bool = True
    cube_isolation: bool = False
    future_spotlight_requires_complex_linear_adjoint: bool = True

    def validate(self) -> None:
        if self.target_ids != ("tophat", "toyota_camry"):
            raise ValueError("measurement policy must keep TopHat and Camry separate")
        if self.representation != "native_complex" or self.transform_name != "identity":
            raise ValueError("readiness policy requires native complex identity transform")
        if not self.apply_to_data or not self.apply_to_predictions:
            raise ValueError("identity transform must be applied consistently to data and predictions")
        if any((self.crop, self.subtraction, self.padding, self.magnitude_conversion)):
            raise ValueError("crop/subtraction/padding/magnitude conversion are forbidden")
        if not self.exterior_scene_retained or self.cube_isolation:
            raise ValueError("exterior scene must remain in the residual/denominator")
        if not self.future_spotlight_requires_complex_linear_adjoint:
            raise ValueError("future spatial spotlight must declare a matching complex-linear adjoint")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "target_ids": list(self.target_ids),
            "representation": self.representation,
            "transform_name": self.transform_name,
            "apply_to_data": self.apply_to_data,
            "apply_to_predictions": self.apply_to_predictions,
            "crop": self.crop,
            "subtraction": self.subtraction,
            "padding": self.padding,
            "magnitude_conversion": self.magnitude_conversion,
            "exterior_scene_retained": self.exterior_scene_retained,
            "cube_isolation": self.cube_isolation,
            "future_spotlight_requires_complex_linear_adjoint": self.future_spotlight_requires_complex_linear_adjoint,
            "interpretation": "not cube isolation; exterior returns retained in native-complex residual and denominator",
        }


DEFAULT_TWO_CUBE_MEASUREMENT_POLICY = TwoCubeMeasurementExteriorPolicy()


@dataclass(frozen=True)
class TophatTrainPanelSelection:
    """Canonical IDs selected from existing shards without touching response."""

    ids: tuple[Any, ...]
    available_shards: tuple[str, ...]
    missing_shards: tuple[str, ...]
    counts_by_shard: Mapping[str, int]
    panel: TophatMultipassPanel = DEFAULT_TOPHAT_MULTIPASS_PANEL

    def __post_init__(self) -> None:
        keys = tuple(_identity_key(identity) for identity in self.ids)
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise ValueError("panel selection IDs must be unique and canonically sorted")
        object.__setattr__(self, "counts_by_shard", _freeze(dict(self.counts_by_shard)))

    @property
    def count(self) -> int:
        return len(self.ids)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ids": [identity.as_dict() if hasattr(identity, "as_dict") else str(identity) for identity in self.ids],
            "count": self.count,
            "available_shards": list(self.available_shards),
            "missing_shards": list(self.missing_shards),
            "counts_by_shard": dict(self.counts_by_shard),
            "panel": self.panel.as_dict(),
        }


def select_tophat_train_ids(
    shards: Sequence[Any], *, panel: TophatMultipassPanel = DEFAULT_TOPHAT_MULTIPASS_PANEL
) -> TophatTrainPanelSelection:
    """Select permitted TRAIN identities from metadata-only shard APIs.

    The function calls only ``identities_for_role('train')`` on matching shards;
    it never reads ``response`` and it never asks a shard for all observations.
    Missing pass/polarization shards are reported because the panel is a design
    input, not a claim that the corresponding archive exists.
    """

    panel = panel
    seen_shards: set[str] = set()
    selected: list[Any] = []
    available: list[str] = []
    for shard in tuple(shards):
        pass_id = int(getattr(shard, "pass_id"))
        polarization = str(getattr(shard, "polarization")).lower()
        shard_id = str(getattr(shard, "shard_id", f"pass{pass_id}_{polarization}"))
        if pass_id not in panel.pass_ids or polarization != panel.polarization:
            continue
        if shard_id in seen_shards:
            raise ValueError(f"duplicate TopHat panel shard: {shard_id}")
        seen_shards.add(shard_id)
        available.append(shard_id)
        # Header/role selection precedes any observation or response materialization.
        for identity in tuple(shard.identities_for_role("train")):
            key = _identity_key(identity)
            if key[0] != pass_id or key[1] != polarization:
                raise ValueError("shard returned an identity inconsistent with its pass/polarization header")
            # A real shard contains all TRAIN sectors.  Only the four frozen
            # panel sectors are selected; off-panel TRAIN rows are not errors.
            if panel.accepts(identity):
                selected.append(identity)
    keys = [_identity_key(identity) for identity in selected]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate canonical TopHat TRAIN identities")
    selected.sort(key=_identity_key)
    expected = {f"pass{pass_id}_{panel.polarization}" for pass_id in panel.pass_ids}
    return TophatTrainPanelSelection(
        ids=tuple(selected),
        available_shards=tuple(sorted(available)),
        missing_shards=tuple(sorted(expected - set(available))),
        counts_by_shard={shard_id: sum(1 for identity in selected if identity.shard_id == shard_id) for shard_id in sorted(available)},
        panel=panel,
    )


def validate_tophat_train_headers(
    observations: Sequence[Any], *, panel: TophatMultipassPanel = DEFAULT_TOPHAT_MULTIPASS_PANEL
) -> tuple[Any, ...]:
    """Validate identity/role/geometry headers without accessing response payloads."""

    records = tuple(observations)
    if not records:
        raise ValueError("TopHat panel requires at least one TRAIN observation")
    seen: set[tuple[int, str, int, int]] = set()
    for record in records:
        identity = getattr(record, "identity", None)
        if identity is None or not panel.accepts(identity):
            raise ValueError("TopHat panel header is outside the explicit HH/TRAIN geometry panel")
        role = str(getattr(record, "role", "")).lower()
        if role != "train":
            raise ValueError("TopHat localization readiness accepts TRAIN only; validation/TEST are closed")
        key = _identity_key(identity)
        if key in seen:
            raise ValueError(f"duplicate canonical TopHat TRAIN identity: {key}")
        seen.add(key)
        position = np.asarray(getattr(record, "position_xyz_m", None), dtype=np.float64)
        frequencies = np.asarray(getattr(record, "frequencies_hz", None), dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("TopHat panel headers require finite position_xyz_m[3]")
        if frequencies.ndim != 1 or frequencies.size == 0 or not np.isfinite(frequencies).all():
            raise ValueError("TopHat panel headers require finite native ragged frequencies")
        if frequencies.size > 1 and not np.all(np.diff(frequencies) > 0):
            raise ValueError("native frequencies must be strictly increasing")
        for name in ("r0_m", "th_deg", "phi_deg"):
            value = float(getattr(record, name))
            if not np.isfinite(value):
                raise ValueError(f"TopHat panel header {name} must be finite")
        phase_reference = getattr(record, "phase_reference", None)
        if phase_reference is None or getattr(phase_reference, "reference_range_field", None) != "r0":
            raise ValueError("TopHat panel headers require native per-pulse r0 provenance")
        if getattr(phase_reference, "geometry_contract", None) != "paired_monostatic_tx_equals_rx_same_observation":
            raise ValueError("TopHat panel headers require paired monostatic geometry")
    keys = tuple(_identity_key(record.identity) for record in records)
    if keys != tuple(sorted(keys)):
        raise ValueError("TopHat TRAIN observations must already be in canonical identity order")
    return records


def convert_tophat_train_source_af(
    observations: Sequence[Any],
    *,
    panel: TophatMultipassPanel = DEFAULT_TOPHAT_MULTIPASS_PANEL,
) -> tuple[SourceAFObservation, ...]:
    """Complete TopHat header preflight followed by one explicit source-AF conversion."""

    if panel != DEFAULT_TOPHAT_MULTIPASS_PANEL:
        raise ValueError("reviewed TopHat source-AF route requires the exact frozen panel")
    records = validate_tophat_train_headers(observations, panel=panel)
    scope = MultipassHHTrainSourceAFScope()
    return build_multipass_source_af(records, scope=scope)


class TophatMeasuredResponseScreenInputError(TophatLocalizationBlocked):
    """Fail-closed metadata/input error for the future measured screen."""


@dataclass(frozen=True)
class TophatMeasuredResponseScreenSelection:
    """Metadata-only full TRAIN selection for measured P1/P7 HH cells."""

    ids: tuple[Any, ...]
    available_hh_shards: tuple[str, ...]
    selected_shards: tuple[str, ...]
    missing_selected_shards: tuple[str, ...]
    counts_by_cell: Mapping[str, int]
    loaded_hh_passes: tuple[int, ...]

    def __post_init__(self) -> None:
        keys = tuple(_identity_key(identity) for identity in self.ids)
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise ValueError("measured TopHat selection IDs must be unique and canonically sorted")
        object.__setattr__(self, "counts_by_cell", _freeze(dict(self.counts_by_cell)))
        object.__setattr__(self, "available_hh_shards", tuple(sorted(str(value) for value in self.available_hh_shards)))
        object.__setattr__(self, "selected_shards", tuple(sorted(str(value) for value in self.selected_shards)))
        object.__setattr__(self, "missing_selected_shards", tuple(sorted(str(value) for value in self.missing_selected_shards)))
        object.__setattr__(self, "loaded_hh_passes", tuple(sorted(int(value) for value in self.loaded_hh_passes)))

    @property
    def count(self) -> int:
        return len(self.ids)

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "available_hh_shards": list(self.available_hh_shards),
            "selected_shards": list(self.selected_shards),
            "missing_selected_shards": list(self.missing_selected_shards),
            "loaded_hh_passes": list(self.loaded_hh_passes),
            "counts_by_cell": dict(self.counts_by_cell),
            "identity_rule": "all metadata-derived TRAIN identities in selected pass/sector cells; canonical sort",
            "response_accessed": False,
            "test_accessed": False,
        }


def _screen_cell_key(pass_id: int, sector_id: int) -> str:
    return f"pass{int(pass_id)}_sector{int(sector_id):03d}"


def _measured_screen_input_error(message: str) -> TophatMeasuredResponseScreenInputError:
    return TophatMeasuredResponseScreenInputError(INCONCLUSIVE_MEASURED_SCREEN_HEADER_INPUT, message)


def select_tophat_measured_response_screen_ids(
    shards: Sequence[Any],
) -> TophatMeasuredResponseScreenSelection:
    """Select all metadata-derived TRAIN pulses from P1/P7 HH target cells.

    The metadata API is the only shard API used here.  P2--P6/P8 HH shards can
    be present in the metadata-first load, but are intentionally not selected
    for the eight full-aperture maps.  No response payload is touched.
    """

    available: list[str] = []
    loaded_passes: list[int] = []
    selected_shards: list[str] = []
    selected: list[Any] = []
    seen_shards: set[str] = set()
    for shard in tuple(shards):
        pass_id = int(getattr(shard, "pass_id"))
        polarization = str(getattr(shard, "polarization")).lower()
        if polarization != MEASURED_RESPONSE_SCREEN_POLARIZATION:
            continue
        shard_id = str(getattr(shard, "shard_id", f"pass{pass_id}_{polarization}"))
        if shard_id in seen_shards:
            raise _measured_screen_input_error(f"duplicate HH metadata shard: {shard_id}")
        seen_shards.add(shard_id)
        available.append(shard_id)
        loaded_passes.append(pass_id)
        if pass_id not in MEASURED_RESPONSE_SCREEN_PASSES:
            continue
        selected_shards.append(shard_id)
        # This is deliberately the role-filtered metadata path.  It must not
        # be replaced by observations() or a response-backed convenience API.
        for identity in tuple(shard.identities_for_role(MEASURED_RESPONSE_SCREEN_ROLE)):
            key = _identity_key(identity)
            if key[0] != pass_id or key[1] != polarization:
                raise _measured_screen_input_error(
                    f"metadata identity is inconsistent with {shard_id}: {key}"
                )
            if key[2] in MEASURED_RESPONSE_SCREEN_SECTORS:
                selected.append(identity)

    if any(f"pass{pass_id}_{MEASURED_RESPONSE_SCREEN_POLARIZATION}" not in seen_shards for pass_id in MEASURED_RESPONSE_SCREEN_PASSES):
        missing = tuple(
            f"pass{pass_id}_{MEASURED_RESPONSE_SCREEN_POLARIZATION}"
            for pass_id in MEASURED_RESPONSE_SCREEN_PASSES
            if f"pass{pass_id}_{MEASURED_RESPONSE_SCREEN_POLARIZATION}" not in seen_shards
        )
        raise _measured_screen_input_error(f"missing selected HH metadata shard(s): {', '.join(missing)}")

    keys = [_identity_key(identity) for identity in selected]
    if len(set(keys)) != len(keys):
        raise _measured_screen_input_error("duplicate measured TopHat TRAIN identities")
    selected.sort(key=_identity_key)
    counts = {
        _screen_cell_key(pass_id, sector_id): sum(
            1 for identity in selected
            if int(identity.pass_id) == pass_id and int(identity.sector_id) == sector_id
        )
        for pass_id in MEASURED_RESPONSE_SCREEN_PASSES
        for sector_id in MEASURED_RESPONSE_SCREEN_SECTORS
    }
    expected_counts = {
        _screen_cell_key(pass_id, sector_id): int(count)
        for pass_id, by_sector in MEASURED_RESPONSE_SCREEN_COUNT_MATRIX.items()
        for sector_id, count in by_sector.items()
    }
    if counts != expected_counts:
        raise _measured_screen_input_error(
            f"measured P1/P7 TRAIN count matrix drifted: observed={counts} expected={expected_counts}"
        )
    if len(selected) != MEASURED_RESPONSE_SCREEN_TOTAL_RECORD_COUNT:
        raise _measured_screen_input_error("measured P1/P7 selection does not contain exactly 949 TRAIN rows")
    return TophatMeasuredResponseScreenSelection(
        ids=tuple(selected),
        available_hh_shards=tuple(sorted(available)),
        selected_shards=tuple(sorted(selected_shards)),
        missing_selected_shards=(),
        counts_by_cell=counts,
        loaded_hh_passes=tuple(sorted(set(loaded_passes))),
    )


def _metadata_header_records_for_ids(
    shards: Sequence[Any], ids: Sequence[Any]
) -> tuple[Any, ...]:
    """Build response-free header records from NativeShard metadata arrays."""

    by_shard: dict[str, list[Any]] = {}
    for identity in ids:
        by_shard.setdefault(str(identity.shard_id), []).append(identity)
    headers: list[Any] = []
    shard_by_id = {str(getattr(shard, "shard_id")): shard for shard in tuple(shards)}
    for shard_id, selected_ids in by_shard.items():
        if shard_id not in shard_by_id:
            raise _measured_screen_input_error(f"selected identity has no metadata shard: {shard_id}")
        shard = shard_by_id[shard_id]
        # observation_ids, unlike observations(), is constructed solely from
        # sector/pulse metadata and is safe before the response gate.
        all_ids = tuple(shard.observation_ids)
        index_by_id = {identity: index for index, identity in enumerate(all_ids)}
        for identity in selected_ids:
            if identity not in index_by_id:
                raise _measured_screen_input_error(f"selected identity is absent from metadata: {identity}")
            i = int(index_by_id[identity])
            headers.append(
                SimpleNamespace(
                    identity=identity,
                    role=str(np.asarray(shard.role)[i]),
                    frequencies_hz=np.asarray(shard.frequencies_hz, dtype=np.float64),
                    position_xyz_m=np.asarray(
                        [np.asarray(shard.x)[i], np.asarray(shard.y)[i], np.asarray(shard.z)[i]],
                        dtype=np.float64,
                    ),
                    r0_m=float(np.asarray(shard.r0)[i]),
                    th_deg=float(np.asarray(shard.th)[i]),
                    phi_deg=float(np.asarray(shard.phi)[i]),
                    r_correct_raw=float(np.asarray(shard.r_correct_raw)[i]),
                    ph_correct_raw=float(np.asarray(shard.ph_correct_raw)[i]),
                    phase_reference=shard.phase_reference,
                    autofocus=shard.autofocus,
                )
            )
    headers.sort(key=lambda record: _identity_key(record.identity))
    return tuple(headers)


def preflight_tophat_measured_response_screen_headers(
    headers: Sequence[Any],
) -> dict[str, Any]:
    """Complete measured-screen header gate, still before response access."""

    records = validate_tophat_train_headers(headers, panel=DEFAULT_TOPHAT_MULTIPASS_PANEL)
    if len(records) != MEASURED_RESPONSE_SCREEN_TOTAL_RECORD_COUNT:
        raise _measured_screen_input_error("measured screen header gate requires exactly 949 rows")
    counts = {
        _screen_cell_key(pass_id, sector_id): sum(
            1 for record in records
            if int(record.identity.pass_id) == pass_id and int(record.identity.sector_id) == sector_id
        )
        for pass_id in MEASURED_RESPONSE_SCREEN_PASSES
        for sector_id in MEASURED_RESPONSE_SCREEN_SECTORS
    }
    expected_counts = {
        _screen_cell_key(pass_id, sector_id): int(count)
        for pass_id, by_sector in MEASURED_RESPONSE_SCREEN_COUNT_MATRIX.items()
        for sector_id, count in by_sector.items()
    }
    if counts != expected_counts:
        raise _measured_screen_input_error("measured screen header count matrix drifted")
    frequency_counts = {
        int(pass_id): tuple(sorted({int(np.asarray(record.frequencies_hz).size) for record in records if int(record.identity.pass_id) == pass_id}))
        for pass_id in MEASURED_RESPONSE_SCREEN_PASSES
    }
    expected_frequency_counts = {
        int(pass_id): (int(frequency_count),)
        for pass_id, frequency_count in MEASURED_RESPONSE_SCREEN_FREQUENCY_COUNTS_BY_PASS.items()
    }
    if frequency_counts != expected_frequency_counts:
        raise _measured_screen_input_error(
            f"measured screen ragged frequency widths drifted: {frequency_counts}"
        )
    total_frequency_samples = int(sum(np.asarray(record.frequencies_hz).size for record in records))
    if total_frequency_samples != MEASURED_RESPONSE_SCREEN_TOTAL_FREQUENCY_SAMPLES:
        raise _measured_screen_input_error("measured screen native frequency sample count drifted")
    return {
        "record_count": len(records),
        "frequency_sample_count": total_frequency_samples,
        "frequency_counts_by_pass": {str(key): list(value) for key, value in frequency_counts.items()},
        "counts_by_cell": counts,
        "role": MEASURED_RESPONSE_SCREEN_ROLE,
        "polarization": MEASURED_RESPONSE_SCREEN_POLARIZATION,
        "response_accessed": False,
        "header_gate_complete": True,
    }


def convert_tophat_measured_response_screen_source_af(
    observations: Sequence[Any],
) -> tuple[SourceAFObservation, ...]:
    """Convert the complete P1/P7 TRAIN screen exactly once to source-AF."""

    records = validate_tophat_train_headers(observations, panel=DEFAULT_TOPHAT_MULTIPASS_PANEL)
    if len(records) != MEASURED_RESPONSE_SCREEN_TOTAL_RECORD_COUNT:
        raise _measured_screen_input_error("source-AF screen conversion requires exactly 949 rows")
    # The existing multipass scope is intentionally used unchanged: it
    # validates HH/TRAIN source-AF provenance while allowing this selected
    # P1/P7 subset.  No alternate sign/range convention is introduced.
    scope = MultipassHHTrainSourceAFScope(expected_count=len(records), name="measured_screen_p1_p7_hh_train")
    return build_multipass_source_af(records, scope=scope)


@dataclass(frozen=True)
class TophatMeasuredResponseScreenPanel:
    """One separate full-aperture pass/sector map ownership region."""

    panel_id: str
    pass_id: int
    sector_id: int
    record_indices: tuple[int, ...]
    frequency_count: int
    frequency_sample_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "panel_id": self.panel_id,
            "pass_id": self.pass_id,
            "sector_id": self.sector_id,
            "record_count": len(self.record_indices),
            "frequency_count": self.frequency_count,
            "frequency_sample_count": self.frequency_sample_count,
            "map_mode": "separate full-aperture native-complex map; no coherent sum across panels",
        }


def build_tophat_measured_response_screen_panels(
    source_records: Sequence[SourceAFObservation],
) -> tuple[TophatMeasuredResponseScreenPanel, ...]:
    records = tuple(source_records)
    if len(records) != MEASURED_RESPONSE_SCREEN_TOTAL_RECORD_COUNT:
        raise _measured_screen_input_error("measured screen panels require the complete 949-row source-AF set")
    keys = tuple(_identity_key(record.identity) for record in records)
    if keys != tuple(sorted(keys)):
        raise _measured_screen_input_error("source-AF records must be canonically sorted before panel partition")
    panels: list[TophatMeasuredResponseScreenPanel] = []
    for pass_id in MEASURED_RESPONSE_SCREEN_PASSES:
        for sector_id in MEASURED_RESPONSE_SCREEN_SECTORS:
            indices = tuple(
                index for index, record in enumerate(records)
                if int(record.identity.pass_id) == pass_id and int(record.identity.sector_id) == sector_id
            )
            expected_rows = int(MEASURED_RESPONSE_SCREEN_COUNT_MATRIX[pass_id][sector_id])
            expected_frequencies = int(MEASURED_RESPONSE_SCREEN_FREQUENCY_COUNTS_BY_PASS[pass_id])
            if len(indices) != expected_rows:
                raise _measured_screen_input_error("panel partition row count drifted")
            widths = {int(records[index].frequencies_hz.size) for index in indices}
            if widths != {expected_frequencies}:
                raise _measured_screen_input_error("panel partition frequency width drifted")
            panels.append(
                TophatMeasuredResponseScreenPanel(
                    panel_id=_screen_cell_key(pass_id, sector_id),
                    pass_id=pass_id,
                    sector_id=sector_id,
                    record_indices=indices,
                    frequency_count=expected_frequencies,
                    frequency_sample_count=expected_rows * expected_frequencies,
                )
            )
    if len(panels) != MEASURED_RESPONSE_SCREEN_PANEL_COUNT:
        raise _measured_screen_input_error("measured screen requires exactly eight separate panels")
    return tuple(panels)


@dataclass(frozen=True)
class TophatMeasuredResponseScreenDataset:
    """Bound measured records after metadata gate and one source-AF conversion."""

    source_records: tuple[SourceAFObservation, ...]
    selection: TophatMeasuredResponseScreenSelection
    header_preflight: Mapping[str, Any]
    panels: tuple[TophatMeasuredResponseScreenPanel, ...]
    source_af_conversion_count: int = 1
    archive_source: str = MEASURED_RESPONSE_SCREEN_ARCHIVE_SOURCE
    scene_name: str = MEASURED_RESPONSE_SCREEN_SCENE

    def __post_init__(self) -> None:
        records = tuple(self.source_records)
        if len(records) != MEASURED_RESPONSE_SCREEN_TOTAL_RECORD_COUNT:
            raise ValueError("measured screen dataset requires exactly 949 source-AF records")
        if self.source_af_conversion_count != 1:
            raise ValueError("measured screen source-AF conversion must occur exactly once")
        if int(self.header_preflight.get("record_count", -1)) != len(records):
            raise ValueError("header preflight and source-AF record counts differ")
        object.__setattr__(self, "source_records", records)
        object.__setattr__(self, "header_preflight", _freeze(dict(self.header_preflight)))

    @property
    def frequency_sample_count(self) -> int:
        return int(sum(record.frequencies_hz.size for record in self.source_records))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": MEASURED_RESPONSE_SCREEN_SCHEMA,
            "archive_source": self.archive_source,
            "scene_name": self.scene_name,
            "selected_record_count": len(self.source_records),
            "native_frequency_sample_count": self.frequency_sample_count,
            "selection": self.selection.as_dict(),
            "header_preflight": dict(self.header_preflight),
            "panels": [panel.as_dict() for panel in self.panels],
            "source_af_conversion_count": self.source_af_conversion_count,
            "representation": "native_complex_source_af_identity",
            "exterior_scene_retained": True,
            "cube_isolation": False,
            "response_payload_serialized": False,
        }


def prepare_tophat_measured_response_screen_dataset(
    shards: Sequence[Any],
) -> TophatMeasuredResponseScreenDataset:
    """Metadata-first bind of future measured shards; no archive I/O is done here."""

    selection = select_tophat_measured_response_screen_ids(shards)
    headers = _metadata_header_records_for_ids(shards, selection.ids)
    header_preflight = preflight_tophat_measured_response_screen_headers(headers)
    # Response materialization is intentionally the first payload operation,
    # after the complete identity/role/geometry/frequency gate above.
    by_shard = {str(getattr(shard, "shard_id")): shard for shard in tuple(shards)}
    observations: list[Any] = []
    for shard_id in selection.selected_shards:
        shard = by_shard[shard_id]
        shard_ids = tuple(identity for identity in selection.ids if str(identity.shard_id) == shard_id)
        observations.extend(tuple(shard.observations(shard_ids)))
    observations.sort(key=lambda record: _identity_key(record.identity))
    source_records = convert_tophat_measured_response_screen_source_af(tuple(observations))
    panels = build_tophat_measured_response_screen_panels(source_records)
    return TophatMeasuredResponseScreenDataset(
        source_records=source_records,
        selection=selection,
        header_preflight=header_preflight,
        panels=panels,
    )


@dataclass(frozen=True)
class TophatMeasuredResponseScreenGrid:
    """Finite q grid and discrete working-box candidates; q means scatterer."""

    points_xyz_m: np.ndarray
    cube_centers_xyz_m: np.ndarray
    xy_seed_m: np.ndarray
    xy_offsets_m: tuple[float, ...] = MEASURED_RESPONSE_SCREEN_XY_OFFSETS_M
    z_samples_m: tuple[float, ...] = MEASURED_RESPONSE_SCREEN_Z_SAMPLES_M
    cube_xy_offsets_m: tuple[float, ...] = MEASURED_RESPONSE_SCREEN_CUBE_XY_OFFSETS_M
    cube_z_offsets_m: tuple[float, ...] = MEASURED_RESPONSE_SCREEN_CUBE_Z_OFFSETS_M
    cube_side_m: float = MEASURED_RESPONSE_SCREEN_CUBE_SIDE_M
    stage: str = "coarse"

    def __post_init__(self) -> None:
        points = _readonly(self.points_xyz_m, dtype=np.float64)
        centers = _readonly(self.cube_centers_xyz_m, dtype=np.float64)
        seed = _readonly(self.xy_seed_m, dtype=np.float64)
        expected_points = (
            MEASURED_RESPONSE_SCREEN_CANDIDATE_COUNT
            if self.stage == "coarse"
            else MEASURED_RESPONSE_SCREEN_FINE_CANDIDATE_COUNT
            if self.stage == "fine"
            else -1
        )
        if points.shape != (expected_points, 3):
            raise ValueError("measured screen q grid has an unsupported stage/count")
        expected_centers = (18, 3) if self.stage == "coarse" else (1, 3)
        if centers.shape != expected_centers or seed.shape != (2,):
            raise ValueError("measured screen grid/cube candidate shapes drifted")
        _finite(points, "measured screen q grid")
        _finite(centers, "measured screen working cubes")
        _finite(seed, "measured screen XY seed")
        if self.cube_side_m != 4.0:
            raise ValueError("measured screen working cubes must be 4 m")
        object.__setattr__(self, "points_xyz_m", points)
        object.__setattr__(self, "cube_centers_xyz_m", centers)
        object.__setattr__(self, "xy_seed_m", seed)
        if self.stage not in {"coarse", "fine"}:
            raise ValueError("measured screen grid stage must be coarse or fine")

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "candidate_count": int(self.points_xyz_m.shape[0]),
            "xy_seed_m": self.xy_seed_m.tolist(),
            "xy_offsets_m": list(self.xy_offsets_m),
            "z_samples_m": list(self.z_samples_m),
            "working_cube_candidate_count": int(self.cube_centers_xyz_m.shape[0]),
            "cube_xy_offsets_m": list(self.cube_xy_offsets_m),
            "cube_z_offsets_m": list(self.cube_z_offsets_m),
            "cube_side_m": self.cube_side_m,
            "q_semantics": "scattering-location image pixel; not a TopHat center or surveyed physical registration",
        }


def build_tophat_measured_response_screen_grid(
    seed: TophatDiagramXYSeed = DEFAULT_TOPHAT_DIAGRAM_XY_SEED,
) -> TophatMeasuredResponseScreenGrid:
    points = np.asarray(
        [
            [seed.xy_m[0] + dx, seed.xy_m[1] + dy, z]
            for dx in MEASURED_RESPONSE_SCREEN_XY_OFFSETS_M
            for dy in MEASURED_RESPONSE_SCREEN_XY_OFFSETS_M
            for z in MEASURED_RESPONSE_SCREEN_Z_SAMPLES_M
        ],
        dtype=np.float64,
    )
    centers = np.asarray(
        [
            [seed.xy_m[0] + dx, seed.xy_m[1] + dy, z]
            for dx in MEASURED_RESPONSE_SCREEN_CUBE_XY_OFFSETS_M
            for dy in MEASURED_RESPONSE_SCREEN_CUBE_XY_OFFSETS_M
            for z in MEASURED_RESPONSE_SCREEN_CUBE_Z_OFFSETS_M
        ],
        dtype=np.float64,
    )
    return TophatMeasuredResponseScreenGrid(points, centers, seed.xy_m, stage="coarse")


def build_tophat_measured_response_fine_grid(
    cube_center_xyz_m: Sequence[float],
) -> TophatMeasuredResponseScreenGrid:
    """Build the one finite 0.5 m half-open refinement inside a Stage-A cube."""

    center = np.asarray(cube_center_xyz_m, dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("fine measured screen grid requires one finite cube center")
    offsets = tuple(-2.0 + 0.5 * index for index in range(8))
    points = np.asarray(
        [[center[0] + dx, center[1] + dy, center[2] + dz] for dx in offsets for dy in offsets for dz in offsets],
        dtype=np.float64,
    )
    return TophatMeasuredResponseScreenGrid(
        points,
        np.asarray([center], dtype=np.float64),
        center[:2],
        xy_offsets_m=offsets,
        z_samples_m=offsets,
        cube_xy_offsets_m=(),
        cube_z_offsets_m=(),
        stage="fine",
    )


@dataclass(frozen=True)
class TophatMeasuredResponseScreenMapSet:
    """Eight streamed maps; only per-panel accumulators are retained."""

    matched_complex_by_panel: np.ndarray
    normalized_power_by_panel: np.ndarray
    kernel_energy_by_panel: np.ndarray
    response_energy_by_panel: np.ndarray
    panels: tuple[TophatMeasuredResponseScreenPanel, ...]
    candidate_count: int
    kernel_sample_terms: int
    map_rerender_count: int = 0

    def __post_init__(self) -> None:
        matched = _readonly(self.matched_complex_by_panel, dtype=np.complex128)
        power = _readonly(self.normalized_power_by_panel, dtype=np.float64)
        kernel_energy = _readonly(self.kernel_energy_by_panel, dtype=np.float64)
        energy = _readonly(self.response_energy_by_panel, dtype=np.float64)
        expected = (MEASURED_RESPONSE_SCREEN_PANEL_COUNT, int(self.candidate_count))
        if matched.shape != expected or power.shape != expected or kernel_energy.shape != expected or energy.shape != (MEASURED_RESPONSE_SCREEN_PANEL_COUNT,):
            raise ValueError("measured screen streamed map shapes drifted")
        if self.map_rerender_count != 0:
            raise ValueError("measured screen maps cannot rerender kernels")
        object.__setattr__(self, "matched_complex_by_panel", matched)
        object.__setattr__(self, "normalized_power_by_panel", power)
        object.__setattr__(self, "kernel_energy_by_panel", kernel_energy)
        object.__setattr__(self, "response_energy_by_panel", energy)

    @property
    def noncoherent_rank_summary(self) -> np.ndarray:
        ranks = np.argsort(np.argsort(-self.normalized_power_by_panel, axis=1), axis=1)
        if self.candidate_count <= 1:
            return _readonly(np.ones(1, dtype=np.float64))
        rank_score = 1.0 - (ranks.astype(np.float64) / float(self.candidate_count - 1))
        return _readonly(np.median(rank_score, axis=0), dtype=np.float64)

    def as_dict(self) -> dict[str, Any]:
        return {
            "panel_count": len(self.panels),
            "candidate_count": self.candidate_count,
            "panels": [panel.as_dict() for panel in self.panels],
            "kernel_sample_terms": self.kernel_sample_terms,
            "map_rerender_count": self.map_rerender_count,
            "map_storage": "eight complex/native-complex panel maps; no dense H[C,F] cache",
            "noncoherent_summary": "median rank display only; no coherent sum and no peak-equality rule",
            "noncoherent_rank_summary_artifact": "stage_a_maps.npz::noncoherent_rank_summary (or stage_b_maps.npz for fine maps)",
            "noncoherent_rank_summary_persisted": True,
            "response_energy_status": [
                "nonzero" if float(value) > 0.0 else INCONCLUSIVE_MEASURED_SCREEN_NO_COHERENT_SUPPORT
                for value in self.response_energy_by_panel
            ],
        }


def stream_tophat_measured_response_screen_maps(
    dataset: TophatMeasuredResponseScreenDataset,
    grid: TophatMeasuredResponseScreenGrid,
) -> TophatMeasuredResponseScreenMapSet:
    """Stream records/candidate kernels into eight owning panel accumulators."""

    records = dataset.source_records
    points = np.asarray(grid.points_xyz_m, dtype=np.float64)
    if points.shape not in {
        (MEASURED_RESPONSE_SCREEN_CANDIDATE_COUNT, 3),
        (MEASURED_RESPONSE_SCREEN_FINE_CANDIDATE_COUNT, 3),
    }:
        raise ValueError("measured screen map grid must contain 294 coarse or 512 fine q points")
    matched = np.zeros((MEASURED_RESPONSE_SCREEN_PANEL_COUNT, points.shape[0]), dtype=np.complex128)
    kernel_energy = np.zeros((MEASURED_RESPONSE_SCREEN_PANEL_COUNT, points.shape[0]), dtype=np.float64)
    energies = np.zeros(MEASURED_RESPONSE_SCREEN_PANEL_COUNT, dtype=np.float64)
    record_to_panel: dict[int, int] = {}
    for panel_index, panel in enumerate(dataset.panels):
        for record_index in panel.record_indices:
            if record_index in record_to_panel:
                raise ValueError("measured screen panel partition overlaps")
            record_to_panel[record_index] = panel_index
    kernel_terms = 0
    for record_index, record in enumerate(records):
        panel_index = record_to_panel.get(record_index)
        if panel_index is None:
            raise ValueError("measured screen panel partition has a gap")
        response = np.asarray(record.effective_response, dtype=np.complex128)
        kernel = _source_af_kernel(points, record, record.effective_r0_m)
        matched[panel_index] += np.conjugate(kernel) @ response
        kernel_energy[panel_index] += np.sum(np.abs(kernel) ** 2, axis=1)
        energies[panel_index] += float(np.vdot(response, response).real)
        kernel_terms += int(points.shape[0] * response.size)
    expected_kernel_terms = int(points.shape[0] * dataset.frequency_sample_count)
    if kernel_terms != expected_kernel_terms:
        raise _measured_screen_input_error(
            f"measured screen streamed map kernel terms drifted: {kernel_terms}"
        )
    power = np.zeros_like(matched.real, dtype=np.float64)
    for panel_index, panel in enumerate(dataset.panels):
        if energies[panel_index] <= 0.0 or not np.isfinite(energies[panel_index]):
            continue
        denominator = kernel_energy[panel_index] * energies[panel_index]
        nonzero = denominator > 0.0
        power[panel_index, nonzero] = np.abs(matched[panel_index, nonzero]) ** 2 / denominator[nonzero]
    return TophatMeasuredResponseScreenMapSet(
        matched_complex_by_panel=matched,
        normalized_power_by_panel=power,
        kernel_energy_by_panel=kernel_energy,
        response_energy_by_panel=energies,
        panels=dataset.panels,
        candidate_count=points.shape[0],
        kernel_sample_terms=kernel_terms,
    )


@dataclass(frozen=True)
class TophatMeasuredResponseSupportComponent:
    """Sampled half-power component with the declared half-cell dilation."""

    panel_id: str
    point_indices: tuple[int, ...]
    lower_xyz_m: np.ndarray
    upper_xyz_m: np.ndarray
    touches_outer_boundary: bool
    peak_power: float

    def __post_init__(self) -> None:
        lower = _readonly(self.lower_xyz_m, dtype=np.float64)
        upper = _readonly(self.upper_xyz_m, dtype=np.float64)
        if lower.shape != (3,) or upper.shape != (3,) or np.any(lower > upper):
            raise ValueError("support component bounds are invalid")
        object.__setattr__(self, "lower_xyz_m", lower)
        object.__setattr__(self, "upper_xyz_m", upper)
        object.__setattr__(self, "point_indices", tuple(sorted(int(value) for value in self.point_indices)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "panel_id": self.panel_id,
            "point_count": len(self.point_indices),
            "lower_xyz_m": self.lower_xyz_m.tolist(),
            "upper_xyz_m": self.upper_xyz_m.tolist(),
            "touches_outer_boundary": self.touches_outer_boundary,
            "peak_power": self.peak_power,
            "dilation": "half one-metre coarse cell = 0.5 m per side; sampled uncertainty convention",
        }


def _coarse_grid_index_neighbors(index: int) -> tuple[int, ...]:
    ix, remainder = divmod(int(index), 7 * 6)
    iy, iz = divmod(remainder, 6)
    result: list[int] = []
    for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        nx, ny, nz = ix + dx, iy + dy, iz + dz
        if 0 <= nx < 7 and 0 <= ny < 7 and 0 <= nz < 6:
            result.append(nx * 7 * 6 + ny * 6 + nz)
    return tuple(result)


def measured_tophat_half_power_components(
    grid: TophatMeasuredResponseScreenGrid,
    maps: TophatMeasuredResponseScreenMapSet,
) -> tuple[tuple[TophatMeasuredResponseSupportComponent, ...], ...]:
    """Find coarse 6-connected half-power supports without sub-grid fitting."""

    if grid.stage != "coarse" or maps.candidate_count != MEASURED_RESPONSE_SCREEN_CANDIDATE_COUNT:
        raise ValueError("half-power support components require Stage-A coarse maps")
    points = np.asarray(grid.points_xyz_m, dtype=np.float64)
    result: list[tuple[TophatMeasuredResponseSupportComponent, ...]] = []
    for panel_index, panel in enumerate(maps.panels):
        values = np.asarray(maps.normalized_power_by_panel[panel_index], dtype=np.float64)
        maximum = float(np.max(values)) if values.size else 0.0
        if not np.isfinite(maximum) or maximum <= 0.0:
            result.append(())
            continue
        active = set(np.flatnonzero(values >= 0.5 * maximum).tolist())
        components: list[TophatMeasuredResponseSupportComponent] = []
        while active:
            seed_index = min(active)
            stack = [seed_index]
            active.remove(seed_index)
            component: list[int] = []
            while stack:
                index = stack.pop()
                component.append(index)
                for neighbor in _coarse_grid_index_neighbors(index):
                    if neighbor in active:
                        active.remove(neighbor)
                        stack.append(neighbor)
            component.sort()
            component_points = points[np.asarray(component, dtype=np.int64)]
            lower = np.min(component_points, axis=0) - 0.5
            upper = np.max(component_points, axis=0) + 0.5
            touches = any(
                (index // (7 * 6) in {0, 6})
                or ((index % (7 * 6)) // 6 in {0, 6})
                or (index % 6 in {0, 5})
                for index in component
            )
            components.append(
                TophatMeasuredResponseSupportComponent(
                    panel_id=panel.panel_id,
                    point_indices=tuple(component),
                    lower_xyz_m=lower,
                    upper_xyz_m=upper,
                    touches_outer_boundary=touches,
                    peak_power=maximum,
                )
            )
        components.sort(key=lambda item: (-item.peak_power, item.lower_xyz_m.tolist(), item.point_indices))
        result.append(tuple(components))
    return tuple(result)


@dataclass(frozen=True)
class TophatMeasuredResponseCubeAssessment:
    """One of the 18 coarse working-cube candidates and all support margins."""

    cube_index: int
    center_xyz_m: np.ndarray
    lower_xyz_m: np.ndarray
    upper_xyz_m: np.ndarray
    discovery_fits: bool
    confirmation_fits: bool
    common_family_member: bool
    minimum_containment_margin_m: float | None
    touches_outer_boundary: bool
    discovery_response_ratio: float | None = None
    confirmation_response_ratio: float | None = None
    all_response_ratio: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "center_xyz_m", _readonly(self.center_xyz_m, dtype=np.float64))
        object.__setattr__(self, "lower_xyz_m", _readonly(self.lower_xyz_m, dtype=np.float64))
        object.__setattr__(self, "upper_xyz_m", _readonly(self.upper_xyz_m, dtype=np.float64))

    def as_dict(self) -> dict[str, Any]:
        return {
            "cube_index": self.cube_index,
            "center_xyz_m": self.center_xyz_m.tolist(),
            "lower_xyz_m": self.lower_xyz_m.tolist(),
            "upper_xyz_m": self.upper_xyz_m.tolist(),
            "discovery_fits": self.discovery_fits,
            "confirmation_fits": self.confirmation_fits,
            "common_family_member": self.common_family_member,
            "minimum_containment_margin_m": self.minimum_containment_margin_m,
            "touches_outer_boundary": self.touches_outer_boundary,
            "discovery_response_ratio": self.discovery_response_ratio,
            "confirmation_response_ratio": self.confirmation_response_ratio,
            "all_response_ratio": self.all_response_ratio,
        }


@dataclass(frozen=True)
class TophatMeasuredResponseStageAResult:
    """Stage-A support decision; Stage B is conditional on selected_cube."""

    status: str
    components_by_panel: tuple[tuple[TophatMeasuredResponseSupportComponent, ...], ...]
    cube_assessments: tuple[TophatMeasuredResponseCubeAssessment, ...]
    common_cube_indices: tuple[int, ...]
    selected_cube_index: int | None
    selected_cube_center_xyz_m: np.ndarray | None
    inspection_cube_index: int | None
    inspection_cube_center_xyz_m: np.ndarray | None
    inspection_label: str
    inside_outside_ratios_by_panel: Mapping[str, float | None]
    context_dominated: bool
    accepted_physical_R_t: None = None
    ground_height_status: str = BLOCKED_MISSING_NATIVE_Z_DATUM

    def __post_init__(self) -> None:
        center = None if self.selected_cube_center_xyz_m is None else _readonly(self.selected_cube_center_xyz_m, dtype=np.float64)
        if center is not None and center.shape != (3,):
            raise ValueError("selected Stage-A cube center must have shape [3]")
        inspection = None if self.inspection_cube_center_xyz_m is None else _readonly(self.inspection_cube_center_xyz_m, dtype=np.float64)
        if inspection is not None and inspection.shape != (3,):
            raise ValueError("inspection cube center must have shape [3]")
        object.__setattr__(self, "selected_cube_center_xyz_m", center)
        object.__setattr__(self, "inspection_cube_center_xyz_m", inspection)
        object.__setattr__(self, "inside_outside_ratios_by_panel", _freeze(dict(self.inside_outside_ratios_by_panel)))
        if self.accepted_physical_R_t is not None:
            raise ValueError("measured TopHat preparation cannot accept physical R/t")

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "component_count_by_panel": [len(components) for components in self.components_by_panel],
            "cube_assessments": [assessment.as_dict() for assessment in self.cube_assessments],
            "common_cube_indices": list(self.common_cube_indices),
            "selected_cube_index": self.selected_cube_index,
            "selected_cube_center_xyz_m": None if self.selected_cube_center_xyz_m is None else self.selected_cube_center_xyz_m.tolist(),
            "inspection_cube_index": self.inspection_cube_index,
            "inspection_cube_center_xyz_m": None if self.inspection_cube_center_xyz_m is None else self.inspection_cube_center_xyz_m.tolist(),
            "inspection_label": self.inspection_label,
            "inspection_center": None if self.inspection_cube_center_xyz_m is None else self.inspection_cube_center_xyz_m.tolist(),
            "working_transform": None if self.selected_cube_center_xyz_m is None else {
                "R": "I",
                "t_working": self.selected_cube_center_xyz_m.tolist(),
                "semantics": "conditional response-informed operational ROI only",
            },
            "t_working": None if self.selected_cube_center_xyz_m is None else self.selected_cube_center_xyz_m.tolist(),
            "inside_outside_ratios_by_panel": dict(self.inside_outside_ratios_by_panel),
            "context_dominated": self.context_dominated,
            "context_dominated_rule": "descriptive flag iff all 8 panels have outside_max/inside_max >= 1; not a qualification gate",
            "accepted_physical_R_t": None,
            "ground_height_status": self.ground_height_status,
            "q_interpretation": "conditional response-informed operational ROI; not surveyed physical placement",
            "alternate_boxes_retained": True,
        }


def evaluate_tophat_measured_response_stage_a(
    dataset: TophatMeasuredResponseScreenDataset,
    grid: TophatMeasuredResponseScreenGrid,
    maps: TophatMeasuredResponseScreenMapSet,
) -> TophatMeasuredResponseStageAResult:
    """Apply the finite Stage-A support rule plus the fixed fallback inspection rule."""

    components = measured_tophat_half_power_components(grid, maps)
    if any(float(value) <= 0.0 for value in maps.response_energy_by_panel):
        return TophatMeasuredResponseStageAResult(
            status=INCONCLUSIVE_MEASURED_SCREEN_NO_COHERENT_SUPPORT,
            components_by_panel=components,
            cube_assessments=(),
            common_cube_indices=(),
            selected_cube_index=None,
            selected_cube_center_xyz_m=None,
            inspection_cube_index=None,
            inspection_cube_center_xyz_m=None,
            inspection_label="no_coherent_support",
            inside_outside_ratios_by_panel={},
            context_dominated=False,
        )
    assessments: list[TophatMeasuredResponseCubeAssessment] = []
    points = np.asarray(grid.points_xyz_m, dtype=np.float64)
    discovery_indices = tuple(index for index, panel in enumerate(dataset.panels) if panel.sector_id in {2, 182})
    confirmation_indices = tuple(index for index, panel in enumerate(dataset.panels) if panel.sector_id in {92, 272})
    all_boundary = any(component.touches_outer_boundary for group in components for component in group)
    for cube_index, center in enumerate(np.asarray(grid.cube_centers_xyz_m, dtype=np.float64)):
        lower = center - 2.0
        upper = center + 2.0
        margins: list[float] = []
        panel_fits: list[bool] = []
        for group in components:
            if not group:
                panel_fits.append(False)
                continue
            fits = True
            for component in group:
                margin = float(np.min(np.minimum(component.lower_xyz_m - lower, upper - component.upper_xyz_m)))
                margins.append(margin)
                fits = fits and bool(np.all(component.lower_xyz_m >= lower) and np.all(component.upper_xyz_m <= upper))
            panel_fits.append(fits)
        discovery_fits = bool(discovery_indices) and all(panel_fits[index] for index in discovery_indices)
        confirmation_fits = bool(confirmation_indices) and all(panel_fits[index] for index in confirmation_indices)
        cube_inside = np.all((points >= lower) & (points < upper), axis=1)
        ratios = []
        for values in np.asarray(maps.normalized_power_by_panel, dtype=np.float64):
            finite_values = np.where(np.isfinite(values), values, 0.0)
            global_max = float(np.max(finite_values))
            inside_max = float(np.max(finite_values[cube_inside])) if np.any(cube_inside) else 0.0
            ratios.append(0.0 if global_max <= 0.0 else inside_max / global_max)
        discovery_ratio = float(np.median([ratios[index] for index in discovery_indices]))
        confirmation_ratio = float(np.median([ratios[index] for index in confirmation_indices]))
        assessments.append(
            TophatMeasuredResponseCubeAssessment(
                cube_index=cube_index,
                center_xyz_m=center,
                lower_xyz_m=lower,
                upper_xyz_m=upper,
                discovery_fits=discovery_fits,
                confirmation_fits=confirmation_fits,
                common_family_member=bool(discovery_fits and confirmation_fits and not all_boundary),
                minimum_containment_margin_m=min(margins) if margins else None,
                touches_outer_boundary=all_boundary,
                discovery_response_ratio=discovery_ratio,
                confirmation_response_ratio=confirmation_ratio,
                all_response_ratio=float(np.median(ratios)),
            )
        )
    common_sorted = tuple(
        assessment.cube_index
        for assessment in sorted(
            (assessment for assessment in assessments if assessment.common_family_member),
            key=lambda item: (-float(item.minimum_containment_margin_m), tuple(float(value) for value in item.center_xyz_m)),
        )
    )
    selected = common_sorted[0] if common_sorted else None
    if selected is not None:
        inspection = selected
        inspection_label = "qualified_candidate_for_inspection"
    else:
        # Fixed fallback: rank all 18 boxes by (min(D,C), A), then lexicographic
        # center.  It provides an inspection center only; it never creates t_working.
        inspection = min(
            range(len(assessments)),
            key=lambda index: (
                -min(float(assessments[index].discovery_response_ratio), float(assessments[index].confirmation_response_ratio)),
                -float(assessments[index].all_response_ratio),
                tuple(float(value) for value in assessments[index].center_xyz_m),
            ),
        )
        inspection_label = "heuristic_response_ranked_inspection_only"
    inspection_assessment = assessments[inspection]
    inspection_inside = np.all(
        (points >= inspection_assessment.lower_xyz_m) & (points < inspection_assessment.upper_xyz_m), axis=1
    )
    ratios_by_panel: dict[str, float | None] = {}
    for panel_index, panel in enumerate(dataset.panels):
        values = np.where(np.isfinite(maps.normalized_power_by_panel[panel_index]), maps.normalized_power_by_panel[panel_index], 0.0)
        inside_max = float(np.max(values[inspection_inside])) if np.any(inspection_inside) else 0.0
        outside_max = float(np.max(values[~inspection_inside])) if np.any(~inspection_inside) else 0.0
        ratios_by_panel[panel.panel_id] = None if inside_max <= 0.0 else outside_max / inside_max
    context_dominated = bool(
        all(ratios_by_panel.get(panel.panel_id) is not None and ratios_by_panel[panel.panel_id] >= 1.0 for panel in dataset.panels)
    )
    if all_boundary:
        status = INCONCLUSIVE_MEASURED_SCREEN_BOUNDARY_OR_COVERAGE
    elif selected is None:
        status = INCONCLUSIVE_MEASURED_SCREEN_ASPECT_SUPPORT
    else:
        status = "STAGE_A_CONDITIONAL_4M_WORKING_CUBE_SELECTED"
    selected_center = None if selected is None else assessments[selected].center_xyz_m
    return TophatMeasuredResponseStageAResult(
        status=status,
        components_by_panel=components,
        cube_assessments=tuple(assessments),
        common_cube_indices=common_sorted,
        selected_cube_index=selected,
        selected_cube_center_xyz_m=selected_center,
        inspection_cube_index=inspection,
        inspection_cube_center_xyz_m=inspection_assessment.center_xyz_m,
        inspection_label=inspection_label,
        inside_outside_ratios_by_panel=ratios_by_panel,
        context_dominated=context_dominated,
    )


@dataclass(frozen=True)
class TophatMeasuredResponseStageBResult:
    """One finite 0.5 m refinement around a qualified or provisional inspection cube."""

    status: str
    grid: TophatMeasuredResponseScreenGrid | None
    maps: TophatMeasuredResponseScreenMapSet | None
    stopped_before_fine_stage: bool
    qualified_selected_cube_index: int | None
    inspection_cube_index: int | None
    inspection_label: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "grid": None if self.grid is None else self.grid.as_dict(),
            "maps": None if self.maps is None else self.maps.as_dict(),
            "stopped_before_fine_stage": self.stopped_before_fine_stage,
            "qualified_selected_cube_index": self.qualified_selected_cube_index,
            "inspection_cube_index": self.inspection_cube_index,
            "inspection_label": self.inspection_label,
            "interpretation": "single finite half-open [-2,2)^3 0.5 m refinement; not continuous 0.1 m localization",
        }


def run_tophat_measured_response_stage_b(
    dataset: TophatMeasuredResponseScreenDataset,
    stage_a: TophatMeasuredResponseStageAResult,
) -> TophatMeasuredResponseStageBResult:
    """Run Stage B only when Stage A selected a deterministic working cube."""

    if stage_a.inspection_cube_index is None or stage_a.inspection_cube_center_xyz_m is None:
        return TophatMeasuredResponseStageBResult(
            status="STAGE_B_STOPPED_BEFORE_FINE_GRID_NO_STAGE_A_CUBE",
            grid=None,
            maps=None,
            stopped_before_fine_stage=True,
            qualified_selected_cube_index=None,
            inspection_cube_index=None,
            inspection_label=stage_a.inspection_label,
        )
    fine_grid = build_tophat_measured_response_fine_grid(stage_a.inspection_cube_center_xyz_m)
    fine_maps = stream_tophat_measured_response_screen_maps(dataset, fine_grid)
    return TophatMeasuredResponseStageBResult(
        status="STAGE_B_FINITE_REFINEMENT_READY_FOR_REVIEW",
        grid=fine_grid,
        maps=fine_maps,
        stopped_before_fine_stage=False,
        qualified_selected_cube_index=stage_a.selected_cube_index,
        inspection_cube_index=stage_a.inspection_cube_index,
        inspection_label=stage_a.inspection_label,
    )


def choose_tophat_measured_screen_reference(
    grid: TophatMeasuredResponseScreenGrid,
    maps: TophatMeasuredResponseScreenMapSet,
) -> tuple[int, np.ndarray]:
    """Choose one deterministic discovery-rank reference; never refine adaptively."""

    if grid.stage != "coarse" or maps.candidate_count != MEASURED_RESPONSE_SCREEN_CANDIDATE_COUNT:
        raise ValueError("map-derived reference requires Stage-A coarse maps")
    indices = [index for index, panel in enumerate(maps.panels) if panel.sector_id in {2, 182}]
    if not indices:
        raise ValueError("discovery panels are required for the deterministic reference")
    values = maps.normalized_power_by_panel[np.asarray(indices, dtype=np.int64)]
    ranks = np.argsort(np.argsort(-values, axis=1), axis=1)
    summary = np.median(1.0 - ranks.astype(np.float64) / float(maps.candidate_count - 1), axis=0)
    maximum = float(np.max(summary))
    candidates = np.flatnonzero(np.isclose(summary, maximum, rtol=0.0, atol=0.0))
    points = np.asarray(grid.points_xyz_m, dtype=np.float64)
    selected = min((int(index) for index in candidates), key=lambda index: tuple(float(value) for value in points[index]))
    return selected, _readonly(points[selected], dtype=np.float64)


@dataclass(frozen=True)
class TophatMeasuredResponsePSFResult:
    """Separate panel PSFs and noncoherent discovery/confirmation summaries."""

    reference_xyz_m: np.ndarray
    union_point_count: int
    normalized_power_by_panel: np.ndarray
    discovery_rank_summary: np.ndarray
    confirmation_rank_summary: np.ndarray
    kernel_sample_terms: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "reference_xyz_m", _readonly(self.reference_xyz_m, dtype=np.float64))
        object.__setattr__(self, "normalized_power_by_panel", _readonly(self.normalized_power_by_panel, dtype=np.float64))
        object.__setattr__(self, "discovery_rank_summary", _readonly(self.discovery_rank_summary, dtype=np.float64))
        object.__setattr__(self, "confirmation_rank_summary", _readonly(self.confirmation_rank_summary, dtype=np.float64))

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference_xyz_m": self.reference_xyz_m.tolist(),
            "union_point_count": self.union_point_count,
            "panel_count": int(self.normalized_power_by_panel.shape[0]),
            "kernel_sample_terms": self.kernel_sample_terms,
            "group_aggregation": "noncoherent median/rank summaries over four panels; no coherent sum",
            "reference_rule": "discovery rank maximum, then lexicographically lowest q; one diagnostic reference, no refinement loop",
            "group_summary_artifacts": {
                "discovery_rank_summary": "psf_diagnostics.npz::discovery_rank_summary",
                "confirmation_rank_summary": "psf_diagnostics.npz::confirmation_rank_summary",
            },
        }


def stream_tophat_measured_response_psfs(
    dataset: TophatMeasuredResponseScreenDataset,
    coarse_grid: TophatMeasuredResponseScreenGrid,
    fine_grid: TophatMeasuredResponseScreenGrid,
    reference_xyz_m: Sequence[float],
) -> TophatMeasuredResponsePSFResult:
    """Stream separate panel PSFs over the declared coarse+fine trajectory union."""

    if coarse_grid.stage != "coarse" or fine_grid.stage != "fine":
        raise ValueError("PSF requires one coarse and one fine grid")
    reference = np.asarray(reference_xyz_m, dtype=np.float64)
    if reference.shape != (3,) or not np.isfinite(reference).all():
        raise ValueError("PSF reference must be one finite q point")
    union_points = np.vstack((coarse_grid.points_xyz_m, fine_grid.points_xyz_m))
    C_union = int(union_points.shape[0])
    accum = np.zeros((MEASURED_RESPONSE_SCREEN_PANEL_COUNT, C_union), dtype=np.complex128)
    kernel_energy = np.zeros((MEASURED_RESPONSE_SCREEN_PANEL_COUNT, C_union), dtype=np.float64)
    reference_energy = np.zeros(MEASURED_RESPONSE_SCREEN_PANEL_COUNT, dtype=np.float64)
    record_to_panel = {
        record_index: panel_index
        for panel_index, panel in enumerate(dataset.panels)
        for record_index in panel.record_indices
    }
    kernel_terms = 0
    for record_index, record in enumerate(dataset.source_records):
        panel_index = record_to_panel[record_index]
        reference_kernel = _source_af_kernel(reference[None, :], record, record.effective_r0_m)[0]
        union_kernel = _source_af_kernel(union_points, record, record.effective_r0_m)
        accum[panel_index] += np.conjugate(union_kernel) @ reference_kernel
        kernel_energy[panel_index] += np.sum(np.abs(union_kernel) ** 2, axis=1)
        reference_energy[panel_index] += float(np.vdot(reference_kernel, reference_kernel).real)
        kernel_terms += int((C_union + 1) * record.frequencies_hz.size)
    expected = int((C_union + 1) * dataset.frequency_sample_count)
    if kernel_terms != expected or expected != MEASURED_RESPONSE_SCREEN_UNION_FORWARD_PSF_TERMS:
        raise _measured_screen_input_error("union PSF direct-kernel accounting drifted")
    power = np.zeros_like(kernel_energy)
    for panel_index in range(MEASURED_RESPONSE_SCREEN_PANEL_COUNT):
        denominator = kernel_energy[panel_index] * reference_energy[panel_index]
        good = denominator > 0.0
        power[panel_index, good] = np.abs(accum[panel_index, good]) ** 2 / denominator[good]
    discovery = np.median(power[np.asarray([0, 2, 4, 6])], axis=0)
    confirmation = np.median(power[np.asarray([1, 3, 5, 7])], axis=0)
    return TophatMeasuredResponsePSFResult(
        reference_xyz_m=reference,
        union_point_count=C_union,
        normalized_power_by_panel=power,
        discovery_rank_summary=discovery,
        confirmation_rank_summary=confirmation,
        kernel_sample_terms=kernel_terms,
    )


@dataclass(frozen=True)
class TophatMeasuredResponseOffgridSensitivityResult:
    """Eight half-cell reference checks, explicitly sampling sensitivity only."""

    reference_points_xyz_m: np.ndarray
    panel_scores: np.ndarray
    normalized_power_by_panel_corner_candidate: np.ndarray
    kernel_sample_terms: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "reference_points_xyz_m", _readonly(self.reference_points_xyz_m, dtype=np.float64))
        object.__setattr__(self, "panel_scores", _readonly(self.panel_scores, dtype=np.float64))
        object.__setattr__(self, "normalized_power_by_panel_corner_candidate", _readonly(self.normalized_power_by_panel_corner_candidate, dtype=np.float64))

    def as_dict(self) -> dict[str, Any]:
        return {
            "reference_count": int(self.reference_points_xyz_m.shape[0]),
            "panel_count": int(self.panel_scores.shape[0]),
            "kernel_sample_terms": self.kernel_sample_terms,
            "sensitivity_shape": list(self.normalized_power_by_panel_corner_candidate.shape),
            "local_grid": "bounded 3x3x3 fine-neighbor BP around each half-cell corner",
            "interpretation": "local off-grid sampling sensitivity only; not coverage or reflector identity proof",
        }


def stream_tophat_measured_response_offgrid_sensitivity(
    dataset: TophatMeasuredResponseScreenDataset,
    reference_xyz_m: Sequence[float],
) -> TophatMeasuredResponseOffgridSensitivityResult:
    """Run the fixed eight-corner, 1-forward+27-local-BP sensitivity control."""

    reference = np.asarray(reference_xyz_m, dtype=np.float64)
    if reference.shape != (3,) or not np.isfinite(reference).all():
        raise ValueError("off-grid sensitivity requires one finite reference q")
    local_offsets = np.asarray(
        [(dx, dy, dz) for dx in (-0.5, 0.0, 0.5) for dy in (-0.5, 0.0, 0.5) for dz in (-0.5, 0.0, 0.5)],
        dtype=np.float64,
    )
    # The candidate lattice is fixed before the eight off-grid truth corners;
    # therefore no corner is itself a candidate point.
    local_points = reference[None, :] + local_offsets
    corners = np.asarray(
        [reference + np.asarray((dx, dy, dz), dtype=np.float64) for dx in (-0.25, 0.25) for dy in (-0.25, 0.25) for dz in (-0.25, 0.25)],
        dtype=np.float64,
    )
    accumulated = np.zeros((MEASURED_RESPONSE_SCREEN_PANEL_COUNT, corners.shape[0], local_points.shape[0]), dtype=np.complex128)
    local_energy = np.zeros((MEASURED_RESPONSE_SCREEN_PANEL_COUNT, local_points.shape[0]), dtype=np.float64)
    forward_energy = np.zeros((MEASURED_RESPONSE_SCREEN_PANEL_COUNT, corners.shape[0]), dtype=np.float64)
    record_to_panel = {
        record_index: panel_index
        for panel_index, panel in enumerate(dataset.panels)
        for record_index in panel.record_indices
    }
    kernel_terms = 0
    for corner_index, corner in enumerate(corners):
        for record_index, record in enumerate(dataset.source_records):
            panel_index = record_to_panel[record_index]
            forward = _source_af_kernel(corner[None, :], record, record.effective_r0_m)[0]
            local_kernel = _source_af_kernel(reference[None, :] + local_offsets, record, record.effective_r0_m)
            local_bp = np.conjugate(local_kernel) @ forward
            accumulated[panel_index, corner_index] += local_bp
            if corner_index == 0:
                local_energy[panel_index] += np.sum(np.abs(local_kernel) ** 2, axis=1)
            forward_energy[panel_index, corner_index] += float(np.vdot(forward, forward).real)
            kernel_terms += int((1 + 27) * record.frequencies_hz.size)
    expected = int(8 * 28 * dataset.frequency_sample_count)
    if kernel_terms != expected or expected != MEASURED_RESPONSE_SCREEN_OFFGRID_REFERENCE_TERMS:
        raise _measured_screen_input_error("off-grid reference direct-kernel accounting drifted")
    normalized = np.zeros_like(accumulated.real, dtype=np.float64)
    for panel_index in range(MEASURED_RESPONSE_SCREEN_PANEL_COUNT):
        denominator = local_energy[panel_index][None, :] * forward_energy[panel_index][:, None]
        good = denominator > 0.0
        normalized[panel_index, good] = np.abs(accumulated[panel_index][good]) ** 2 / denominator[good]
    panel_max = np.max(normalized, axis=(1, 2), keepdims=True)
    panel_scores = np.divide(
        np.max(normalized, axis=2), panel_max[:, 0, 0][:, None],
        out=np.zeros((MEASURED_RESPONSE_SCREEN_PANEL_COUNT, corners.shape[0]), dtype=np.float64),
        where=panel_max[:, 0, 0][:, None] > 0.0,
    )
    return TophatMeasuredResponseOffgridSensitivityResult(corners, panel_scores, normalized, kernel_terms)


@dataclass(frozen=True)
class TophatMeasuredResponseRawSourceBridgeResult:
    """Nine-point raw/source bridge arithmetic with no tuning authority."""

    point_count: int
    raw_equivalent_scores: np.ndarray
    source_scores: np.ndarray
    max_absolute_difference: float
    kernel_sample_terms: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_equivalent_scores", _readonly(self.raw_equivalent_scores, dtype=np.complex128))
        object.__setattr__(self, "source_scores", _readonly(self.source_scores, dtype=np.complex128))

    def as_dict(self) -> dict[str, Any]:
        return {
            "point_count": self.point_count,
            "kernel_sample_terms": self.kernel_sample_terms,
            "max_absolute_difference": self.max_absolute_difference,
            "interpretation": "fixed representation bridge check only; does not prove AF correctness or reflector identity",
        }


def stream_tophat_measured_response_raw_source_bridge(
    dataset: TophatMeasuredResponseScreenDataset,
) -> TophatMeasuredResponseRawSourceBridgeResult:
    """Compare raw and source-AF identity maps on the fixed 3x3 seed panel."""

    seed = DEFAULT_TOPHAT_DIAGRAM_XY_SEED.xy_m
    points = np.asarray(
        [[seed[0] + dx, seed[1] + dy, 0.0] for dx in (-1.0, 0.0, 1.0) for dy in (-1.0, 0.0, 1.0)],
        dtype=np.float64,
    )
    raw_equivalent_scores = np.zeros(points.shape[0], dtype=np.complex128)
    source_scores = np.zeros(points.shape[0], dtype=np.complex128)
    kernel_terms = 0
    for record in dataset.source_records:
        raw_r0, raw_response = record.raw_payload()
        raw_kernel = _source_af_kernel(points, record, raw_r0)
        source_kernel = _source_af_kernel(points, record, record.effective_r0_m)
        k_r_correct = (4.0 * np.pi / SPEED_OF_LIGHT_M_S) * record.frequencies_hz * np.float64(record.r_correct_raw_m)
        raw_equivalent_response = raw_response * np.exp(
            1j * np.float64(record.ph_correct_raw_rad) - 1j * k_r_correct
        )
        raw_equivalent_scores += np.conjugate(raw_kernel) @ raw_equivalent_response
        source_scores += np.conjugate(source_kernel) @ record.effective_response
        kernel_terms += int(2 * points.shape[0] * record.frequencies_hz.size)
    expected = int(18 * dataset.frequency_sample_count)
    if kernel_terms != expected or expected != MEASURED_RESPONSE_SCREEN_RAW_SOURCE_BRIDGE_TERMS:
        raise _measured_screen_input_error("raw/source bridge direct-kernel accounting drifted")
    return TophatMeasuredResponseRawSourceBridgeResult(
        point_count=9,
        raw_equivalent_scores=raw_equivalent_scores,
        source_scores=source_scores,
        max_absolute_difference=float(np.max(np.abs(raw_equivalent_scores - source_scores))),
        kernel_sample_terms=kernel_terms,
    )


@dataclass(frozen=True)
class TophatMeasuredResponseScreenWorkLedger:
    """Future-run direct-kernel accounting; entries are sample terms, not FLOPs."""

    record_count: int = MEASURED_RESPONSE_SCREEN_TOTAL_RECORD_COUNT
    frequency_sample_count: int = MEASURED_RESPONSE_SCREEN_TOTAL_FREQUENCY_SAMPLES
    candidate_count: int = MEASURED_RESPONSE_SCREEN_CANDIDATE_COUNT
    panel_count: int = MEASURED_RESPONSE_SCREEN_PANEL_COUNT
    source_af_sample_terms: int = MEASURED_RESPONSE_SCREEN_TOTAL_FREQUENCY_SAMPLES
    coarse_map_kernel_sample_terms: int = MEASURED_RESPONSE_SCREEN_STAGE_A_MAP_TERMS
    fine_map_kernel_sample_terms: int = MEASURED_RESPONSE_SCREEN_STAGE_B_FINE_MAP_TERMS
    union_forward_psf_kernel_sample_terms: int = MEASURED_RESPONSE_SCREEN_UNION_FORWARD_PSF_TERMS
    offgrid_reference_kernel_sample_terms: int = MEASURED_RESPONSE_SCREEN_OFFGRID_REFERENCE_TERMS
    raw_source_bridge_kernel_sample_terms: int = MEASURED_RESPONSE_SCREEN_RAW_SOURCE_BRIDGE_TERMS
    total_direct_kernel_sample_terms: int = MEASURED_RESPONSE_SCREEN_TOTAL_DIRECT_KERNEL_TERMS
    max_total_kernel_sample_terms: int = MEASURED_RESPONSE_SCREEN_WORK_CAP
    adaptive_reruns: int = 0
    dense_h_cache_allocated: bool = False

    def __post_init__(self) -> None:
        values = (
            self.record_count,
            self.frequency_sample_count,
            self.candidate_count,
            self.panel_count,
            self.source_af_sample_terms,
            self.coarse_map_kernel_sample_terms,
            self.fine_map_kernel_sample_terms,
            self.union_forward_psf_kernel_sample_terms,
            self.offgrid_reference_kernel_sample_terms,
            self.raw_source_bridge_kernel_sample_terms,
            self.total_direct_kernel_sample_terms,
            self.max_total_kernel_sample_terms,
            self.adaptive_reruns,
        )
        if any(int(value) < 0 for value in values):
            raise ValueError("measured screen work ledger values must be nonnegative")
        expected_total = int(
            self.coarse_map_kernel_sample_terms
            + self.fine_map_kernel_sample_terms
            + self.union_forward_psf_kernel_sample_terms
            + self.offgrid_reference_kernel_sample_terms
            + self.raw_source_bridge_kernel_sample_terms
        )
        if expected_total != int(self.total_direct_kernel_sample_terms):
            raise ValueError("measured screen direct-kernel ledger does not reconcile")
        if self.total_direct_kernel_sample_terms > self.max_total_kernel_sample_terms:
            raise TophatLocalizationBlocked(
                BLOCKED_RESOURCE_CAP_EXCEEDED,
                "measured screen direct-kernel estimate exceeds the provisional 800M cap",
            )
        if self.dense_h_cache_allocated or self.adaptive_reruns:
            raise ValueError("measured screen requires streaming and no adaptive reruns")

    @property
    def under_cap(self) -> bool:
        return self.total_direct_kernel_sample_terms <= self.max_total_kernel_sample_terms

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_term_definition": "direct kernel/sample terms only; not exact FLOPs",
            "record_count": self.record_count,
            "frequency_sample_count": self.frequency_sample_count,
            "candidate_count": self.candidate_count,
            "panel_count": self.panel_count,
            "source_af_sample_terms": self.source_af_sample_terms,
            "coarse_map_kernel_sample_terms": self.coarse_map_kernel_sample_terms,
            "fine_map_kernel_sample_terms": self.fine_map_kernel_sample_terms,
            "union_forward_psf_kernel_sample_terms": self.union_forward_psf_kernel_sample_terms,
            "offgrid_reference_kernel_sample_terms": self.offgrid_reference_kernel_sample_terms,
            "raw_source_bridge_kernel_sample_terms": self.raw_source_bridge_kernel_sample_terms,
            "total_direct_kernel_sample_terms": self.total_direct_kernel_sample_terms,
            "formula": "1855 * F = 755311480; coarse maps + fine maps + union forward/PSF + bridge + off-grid references",
            "max_total_kernel_sample_terms": self.max_total_kernel_sample_terms,
            "under_cap": self.under_cap,
            "streaming_required": True,
            "dense_h_cache_allocated": self.dense_h_cache_allocated,
            "adaptive_reruns": self.adaptive_reruns,
            "resource_plan": {
                "cpu": 1,
                "ram_gib": 4,
                "tmp_gib": 4,
                "wall_minutes": 20,
                "gpu": False,
                "status": "PROVISIONAL_REQUIRES_MANAGER_PREFLIGHT_AND_LOCAL_TIMING_REVIEW",
                "memory_note": "streamed 512x434 complex128 kernel is about 3.6 MB; loaded P1/P7 response copies dominate peak RAM",
            },
            "time_basis": "conservative CPU extrapolation for NumPy streaming; not a measured runtime proof or manager-approved envelope",
        }


@dataclass(frozen=True)
class AspectElevationGeometryReport:
    """Header-only elevation/aspect diversity and trajectory identifiability."""

    observation_count: int
    pass_ids: tuple[int, ...]
    sector_ids: tuple[int, ...]
    elevation_group_count: int
    elevation_span_deg: float
    antenna_vertical_span_m: float
    aspect_group_count: int
    aspect_span_deg: float
    trajectory_rank: int
    horizontal_rank: int
    condition_number: float
    sufficient_elevation_geometry: bool
    status: str
    groups_by_pass: Mapping[str, Mapping[str, Any]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups_by_pass", _freeze({str(k): dict(v) for k, v in self.groups_by_pass.items()}))

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation_count": self.observation_count,
            "pass_ids": list(self.pass_ids),
            "sector_ids": list(self.sector_ids),
            "elevation_group_count": self.elevation_group_count,
            "elevation_span_deg": self.elevation_span_deg,
            "antenna_vertical_span_m": self.antenna_vertical_span_m,
            "aspect_group_count": self.aspect_group_count,
            "aspect_span_deg": self.aspect_span_deg,
            "trajectory_rank": self.trajectory_rank,
            "horizontal_rank": self.horizontal_rank,
            "condition_number": self.condition_number,
            "sufficient_elevation_geometry": self.sufficient_elevation_geometry,
            "status": self.status,
            "groups_by_pass": {key: dict(value) for key, value in self.groups_by_pass.items()},
            "interpretation": "trajectory geometry only; it does not establish native Z datum or physical registration",
        }


def assess_aspect_elevation_geometry(
    observations: Sequence[Any],
    *,
    panel: TophatMultipassPanel = DEFAULT_TOPHAT_MULTIPASS_PANEL,
    min_elevation_groups: int = DEFAULT_MIN_ELEVATION_GROUPS,
    min_elevation_span_deg: float = DEFAULT_MIN_ELEVATION_SPAN_DEG,
    min_antenna_vertical_span_m: float = DEFAULT_MIN_ANTENNA_VERTICAL_SPAN_M,
    max_condition_number: float = DEFAULT_MAX_GEOMETRY_CONDITION,
) -> AspectElevationGeometryReport:
    records = validate_tophat_train_headers(observations, panel=panel)
    passes = tuple(sorted({int(record.identity.pass_id) for record in records}))
    sectors = tuple(sorted({int(record.identity.sector_id) for record in records}))
    phi = np.asarray([float(record.phi_deg) for record in records], dtype=np.float64)
    theta = np.asarray([float(record.th_deg) for record in records], dtype=np.float64)
    positions = np.asarray([record.position_xyz_m for record in records], dtype=np.float64)
    groups: dict[str, dict[str, Any]] = {}
    for pass_id in passes:
        values = phi[np.asarray([int(record.identity.pass_id) == pass_id for record in records])]
        groups[str(pass_id)] = {
            "count": int(values.size),
            "phi_min_deg": float(np.min(values)),
            "phi_max_deg": float(np.max(values)),
            "phi_median_deg": float(np.median(values)),
        }
    centered = positions - np.mean(positions, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False) if len(records) > 1 else np.zeros(3)
    trajectory_rank = int(np.linalg.matrix_rank(centered, tol=1.0e-10))
    horizontal_rank = int(np.linalg.matrix_rank(centered[:, :2], tol=1.0e-10))
    nonzero_singular = singular[singular > 1.0e-10]
    condition_number = float(nonzero_singular[0] / nonzero_singular[-1]) if nonzero_singular.size else float("inf")
    elevation_span = float(np.max(phi) - np.min(phi)) if phi.size else 0.0
    antenna_vertical_span = float(np.max(positions[:, 2]) - np.min(positions[:, 2])) if positions.size else 0.0
    aspect_span = float(np.max(theta) - np.min(theta)) if theta.size else 0.0
    sufficient = bool(
        len(records) >= 2
        and len(passes) >= int(min_elevation_groups)
        and elevation_span >= float(min_elevation_span_deg)
        and antenna_vertical_span >= float(min_antenna_vertical_span_m)
        and horizontal_rank >= 1
        and trajectory_rank >= 2
        and condition_number <= float(max_condition_number)
    )
    return AspectElevationGeometryReport(
        observation_count=len(records),
        pass_ids=passes,
        sector_ids=sectors,
        elevation_group_count=len(passes),
        elevation_span_deg=elevation_span,
        antenna_vertical_span_m=antenna_vertical_span,
        aspect_group_count=len(sectors),
        aspect_span_deg=aspect_span,
        trajectory_rank=trajectory_rank,
        horizontal_rank=horizontal_rank,
        condition_number=condition_number,
        sufficient_elevation_geometry=sufficient,
        status="CONDITIONAL_ELEVATION_GEOMETRY_AVAILABLE" if sufficient else INCONCLUSIVE_INSUFFICIENT_ELEVATION_GEOMETRY,
        groups_by_pass=groups,
    )


@dataclass(frozen=True)
class ResourcePreflight:
    observation_count: int
    native_frequency_samples: int
    point_count: int
    estimated_kernel_evaluations: int
    max_kernel_evaluations: int
    dense_reference_grid_forbidden: bool = True
    status: str = "READY_BOUNDED_CONDITIONAL_SUPPORT"

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation_count": self.observation_count,
            "native_frequency_samples": self.native_frequency_samples,
            "point_count": self.point_count,
            "estimated_kernel_evaluations": self.estimated_kernel_evaluations,
            "max_kernel_evaluations": self.max_kernel_evaluations,
            "dense_reference_grid_forbidden": self.dense_reference_grid_forbidden,
            "status": self.status,
        }


def preflight_resource_budget(
    observations: Sequence[Any],
    *,
    point_count: int,
    max_kernel_evaluations: int = DEFAULT_MAX_KERNEL_EVALUATIONS,
    panel: TophatMultipassPanel = DEFAULT_TOPHAT_MULTIPASS_PANEL,
) -> ResourcePreflight:
    records = validate_tophat_train_headers(observations, panel=panel)
    point_count = int(point_count)
    max_kernel_evaluations = int(max_kernel_evaluations)
    if point_count <= 0 or point_count > MAX_CONDITIONAL_POINT_COUNT or point_count == 40 * 40 * 40:
        raise TophatLocalizationBlocked(
            BLOCKED_RESOURCE_CAP_EXCEEDED,
            "conditional readiness rejects dense 40^3 cube search and oversized supports before kernels",
        )
    if max_kernel_evaluations <= 0:
        raise ValueError("max_kernel_evaluations must be positive")
    frequency_samples = int(sum(np.asarray(record.frequencies_hz).size for record in records))
    estimate = int(point_count * frequency_samples)
    if estimate > max_kernel_evaluations:
        raise TophatLocalizationBlocked(
            BLOCKED_RESOURCE_CAP_EXCEEDED,
            f"bounded conditional kernel estimate {estimate} exceeds cap {max_kernel_evaluations}",
        )
    return ResourcePreflight(len(records), frequency_samples, point_count, estimate, max_kernel_evaluations)


@dataclass(frozen=True)
class ConditionalNativeZSearchResult:
    status: str
    conditional_geometric_center_z_m: float | None
    near_minimum_geometric_center_z_samples_m: tuple[float, ...]
    sampled_near_minimum_z_envelope_m: tuple[float, float] | None
    residuals_by_z: Mapping[str, float]
    z_interval: TophatNativeZSearchInterval
    geometry: AspectElevationGeometryReport
    absolute_height_status: str
    native_scattering_point_z_status: str = "aspect_dependent_scattering_z_not_fixed"
    native_geometric_center_z_status: str = "conditional_unregistered_native_coordinate"
    response_localization_used: bool = False

    def __post_init__(self) -> None:
        if self.conditional_geometric_center_z_m is not None:
            raise ValueError("bounded conditional native-Z results cannot claim a point recovery")
        samples = tuple(float(value) for value in self.near_minimum_geometric_center_z_samples_m)
        if tuple(sorted(set(samples))) != samples:
            raise ValueError("near-minimum native-Z samples must be sorted and unique")
        if self.sampled_near_minimum_z_envelope_m is not None:
            low, high = self.sampled_near_minimum_z_envelope_m
            if not np.isfinite([low, high]).all() or low > high:
                raise ValueError("sampled near-minimum native-Z envelope is invalid")
            if samples and (float(low), float(high)) != (samples[0], samples[-1]):
                raise ValueError("sampled near-minimum envelope must match retained samples")
        object.__setattr__(self, "near_minimum_geometric_center_z_samples_m", samples)
        object.__setattr__(self, "residuals_by_z", _freeze(dict(self.residuals_by_z)))

    def require_absolute_height(self, datum: TophatNativeHeightDatum) -> float:
        datum.require_absolute_height()
        raise TophatLocalizationBlocked(
            BLOCKED_MISSING_NATIVE_Z_DATUM,
            "a source-backed datum is present, but absolute registration conversion requires separate reviewed mapping",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "conditional_geometric_center_z_m": self.conditional_geometric_center_z_m,
            "near_minimum_geometric_center_z_samples_m": list(self.near_minimum_geometric_center_z_samples_m),
            "sampled_near_minimum_z_envelope_m": None if self.sampled_near_minimum_z_envelope_m is None else list(self.sampled_near_minimum_z_envelope_m),
            "near_minimum_interpretation": "discrete retained grid samples; gaps are not bridged and the envelope is not continuous feasibility",
            "residuals_by_z": dict(self.residuals_by_z),
            "z_interval": self.z_interval.as_dict(),
            "geometry": self.geometry.as_dict(),
            "absolute_height_status": self.absolute_height_status,
            "native_scattering_point_z_status": self.native_scattering_point_z_status,
            "native_geometric_center_z_status": self.native_geometric_center_z_status,
            "response_localization_used": self.response_localization_used,
        }


def conditional_native_z_search(
    observations: Sequence[Any],
    *,
    z_interval: TophatNativeZSearchInterval | None,
    residuals_by_z: Mapping[float, float] | None = None,
    aspect_evidence_ambiguous: bool = False,
    panel: TophatMultipassPanel = DEFAULT_TOPHAT_MULTIPASS_PANEL,
    height_datum: TophatNativeHeightDatum | None = None,
    feasible_residual_tolerance: float = 0.07,
) -> ConditionalNativeZSearchResult:
    """Return only a bounded conditional native-Z result.

    ``residuals_by_z`` is intentionally explicit and is the only route to a
    bounded conditional interval.  It cannot create a point recovery: the
    only point-recovery path is the separately named full-fixed synthetic rule.
    A real driver must compute residuals from a reviewed complex-linear response
    path; this module does not convert or fit payloads.
    Header validation and the declared interval gate happen first.
    """

    validate_tophat_train_headers(observations, panel=panel)
    if z_interval is None:
        raise TophatLocalizationBlocked(
            BLOCKED_MISSING_DECLARED_NATIVE_Z_SEARCH_INTERVAL,
            "conditional native-Z work requires an explicit finite bounded search interval",
        )
    geometry = assess_aspect_elevation_geometry(observations, panel=panel)
    datum = TophatNativeHeightDatum.missing() if height_datum is None else height_datum
    if not geometry.sufficient_elevation_geometry:
        return ConditionalNativeZSearchResult(
            status=INCONCLUSIVE_INSUFFICIENT_ELEVATION_GEOMETRY,
            conditional_geometric_center_z_m=None,
            near_minimum_geometric_center_z_samples_m=(),
            sampled_near_minimum_z_envelope_m=None,
            residuals_by_z={},
            z_interval=z_interval,
            geometry=geometry,
            absolute_height_status=ABSOLUTE_HEIGHT_STATUS_MISSING_DATUM if not datum.available else ABSOLUTE_HEIGHT_STATUS_SOURCE_DATUM_ONLY,
        )
    if residuals_by_z is None:
        return ConditionalNativeZSearchResult(
            status=INCONCLUSIVE_NO_RESPONSE_LOCALIZATION,
            conditional_geometric_center_z_m=None,
            near_minimum_geometric_center_z_samples_m=(),
            sampled_near_minimum_z_envelope_m=None,
            residuals_by_z={},
            z_interval=z_interval,
            geometry=geometry,
            absolute_height_status=ABSOLUTE_HEIGHT_STATUS_MISSING_DATUM if not datum.available else ABSOLUTE_HEIGHT_STATUS_SOURCE_DATUM_ONLY,
        )
    allowed = tuple(float(value) for value in z_interval.values_m)
    normalized = {float(key): float(value) for key, value in residuals_by_z.items()}
    if set(normalized) != set(allowed):
        raise ValueError("conditional native-Z residuals must cover exactly the declared interval samples")
    if any(not np.isfinite(value) or value < 0 for value in normalized.values()):
        raise ValueError("conditional native-Z residuals must be finite and nonnegative")
    if not np.isfinite(feasible_residual_tolerance) or feasible_residual_tolerance <= 0:
        raise ValueError("feasible_residual_tolerance must be positive and finite")
    minimum = min(normalized.values())
    feasible_values = tuple(
        value for value in allowed if normalized[value] <= minimum + float(feasible_residual_tolerance)
    )
    sampled_envelope = None if not feasible_values else (float(min(feasible_values)), float(max(feasible_values)))
    status = INCONCLUSIVE_ASPECT_DEPENDENT_EVIDENCE if aspect_evidence_ambiguous else CONDITIONAL_NATIVE_Z_UNREGISTERED
    boundary_tolerance = 1.0e-12
    reaches_declared_boundary = any(
        abs(value - z_interval.lower_m) <= boundary_tolerance
        or abs(value - z_interval.upper_m) <= boundary_tolerance
        for value in feasible_values
    )
    if not feasible_values or len(feasible_values) < 2 or reaches_declared_boundary:
        status = INCONCLUSIVE_ASPECT_DEPENDENT_EVIDENCE
    return ConditionalNativeZSearchResult(
        status=status,
        conditional_geometric_center_z_m=None,
        near_minimum_geometric_center_z_samples_m=feasible_values,
        sampled_near_minimum_z_envelope_m=sampled_envelope,
        residuals_by_z={f"{value:.17g}": normalized[value] for value in allowed},
        z_interval=z_interval,
        geometry=geometry,
        absolute_height_status=ABSOLUTE_HEIGHT_STATUS_MISSING_DATUM if not datum.available else ABSOLUTE_HEIGHT_STATUS_SOURCE_DATUM_ONLY,
        response_localization_used=True,
    )


@dataclass(frozen=True)
class SyntheticAspectReturnStudy:
    """Synthetic bounded radial nuisance study; not a measured localization."""

    aspects_deg: np.ndarray
    radial_offsets_m: np.ndarray
    observed_peak_xy_m: np.ndarray
    feasible_center_xy_m: np.ndarray
    fixed_point_residuals_m: np.ndarray
    radial_bounds_m: tuple[float, float]
    fixed_point_assumption_fails: bool
    status: str = INCONCLUSIVE_ASPECT_DEPENDENT_EVIDENCE

    def __post_init__(self) -> None:
        aspects = _readonly(self.aspects_deg)
        offsets = _readonly(self.radial_offsets_m)
        peaks = _readonly(self.observed_peak_xy_m)
        feasible = _readonly(self.feasible_center_xy_m)
        residuals = _readonly(self.fixed_point_residuals_m)
        if aspects.ndim != 1 or offsets.shape != aspects.shape or peaks.shape != (aspects.size, 2):
            raise ValueError("synthetic aspect-return arrays have inconsistent shapes")
        if feasible.ndim != 2 or feasible.shape[1] != 2:
            raise ValueError("conditional feasible centers must have shape [candidate,2]")
        if residuals.shape != aspects.shape or aspects.size < 2:
            raise ValueError("synthetic aspect-return study requires at least two records")
        if not np.isfinite(aspects).all() or not np.isfinite(offsets).all() or not np.isfinite(peaks).all():
            raise ValueError("synthetic aspect-return inputs must be finite")
        if self.radial_bounds_m[0] < 0 or self.radial_bounds_m[0] > self.radial_bounds_m[1]:
            raise ValueError("radial bounds must be ordered and nonnegative")
        for array, name in ((aspects, "aspects_deg"), (offsets, "radial_offsets_m"), (peaks, "observed_peak_xy_m"), (feasible, "feasible_center_xy_m"), (residuals, "fixed_point_residuals_m")):
            _finite(array, name)
        object.__setattr__(self, "aspects_deg", aspects)
        object.__setattr__(self, "radial_offsets_m", offsets)
        object.__setattr__(self, "observed_peak_xy_m", peaks)
        object.__setattr__(self, "feasible_center_xy_m", feasible)
        object.__setattr__(self, "fixed_point_residuals_m", residuals)

    @property
    def radial_offset_span_m(self) -> float:
        return float(np.max(self.radial_offsets_m) - np.min(self.radial_offsets_m))

    def as_dict(self) -> dict[str, Any]:
        return {
            "aspects_deg": self.aspects_deg.tolist(),
            "radial_offsets_m": self.radial_offsets_m.tolist(),
            "observed_peak_xy_m": self.observed_peak_xy_m.tolist(),
            "feasible_center_xy_m": self.feasible_center_xy_m.tolist(),
            "feasible_center_count": int(self.feasible_center_xy_m.shape[0]),
            "fixed_point_residuals_m": self.fixed_point_residuals_m.tolist(),
            "fixed_point_assumption_fails": self.fixed_point_assumption_fails,
            "radial_offset_span_m": self.radial_offset_span_m,
            "radial_bounds_m": list(self.radial_bounds_m),
            "status": self.status,
            "interpretation": "roughly-1m return is an aspect-dependent nuisance, not a fixed geometric center",
        }


def _feasible_centers_from_radial_bounds(
    observed_peak_xy_m: np.ndarray,
    aspects_deg: np.ndarray,
    radial_bounds_m: tuple[float, float],
    *,
    center_xy_m: np.ndarray,
    grid_halfwidth_m: float,
    grid_size: int,
    transverse_tolerance_m: float,
) -> np.ndarray:
    if grid_size <= 0 or grid_size > 101 or grid_size % 2 == 0:
        raise ValueError("conditional center grid_size must be odd and in 1..101")
    axes = np.linspace(center_xy_m - grid_halfwidth_m, center_xy_m + grid_halfwidth_m, grid_size)
    gx, gy = np.meshgrid(axes[:, 0], axes[:, 1], indexing="ij")
    candidates = np.stack((gx.reshape(-1), gy.reshape(-1)), axis=1)
    radians = np.deg2rad(aspects_deg)
    directions = np.stack((np.cos(radians), np.sin(radians)), axis=1)
    valid = np.ones(candidates.shape[0], dtype=bool)
    for peak, direction in zip(observed_peak_xy_m, directions):
        vectors = peak[None, :] - candidates
        projection = vectors @ direction
        transverse = np.abs(vectors[:, 0] * direction[1] - vectors[:, 1] * direction[0])
        valid &= (projection >= radial_bounds_m[0] - 1.0e-12)
        valid &= (projection <= radial_bounds_m[1] + 1.0e-12)
        valid &= transverse <= transverse_tolerance_m + 1.0e-12
    return candidates[valid]


def synthetic_aspect_return_study(
    center_xy_m: Sequence[float],
    aspects_deg: Sequence[float],
    radial_offsets_m: Sequence[float],
    *,
    radial_bounds_m: tuple[float, float] = (0.70, 1.15),
    grid_halfwidth_m: float = 0.50,
    grid_size: int = 21,
    transverse_tolerance_m: float = 0.08,
) -> SyntheticAspectReturnStudy:
    center = _readonly(center_xy_m)
    aspects = _readonly(aspects_deg)
    offsets = _readonly(radial_offsets_m)
    if center.shape != (2,):
        raise ValueError("synthetic center_xy_m must have shape [2]")
    if aspects.ndim != 1 or offsets.shape != aspects.shape or aspects.size < 2:
        raise ValueError("synthetic aspect and radial arrays must have matching length >= 2")
    radians = np.deg2rad(aspects)
    directions = np.stack((np.cos(radians), np.sin(radians)), axis=1)
    peaks = center[None, :] + offsets[:, None] * directions
    residuals = np.linalg.norm(peaks - center[None, :], axis=1)
    feasible = _feasible_centers_from_radial_bounds(
        peaks,
        aspects,
        radial_bounds_m,
        center_xy_m=center,
        grid_halfwidth_m=float(grid_halfwidth_m),
        grid_size=int(grid_size),
        transverse_tolerance_m=float(transverse_tolerance_m),
    )
    return SyntheticAspectReturnStudy(
        aspects_deg=aspects,
        radial_offsets_m=offsets,
        observed_peak_xy_m=peaks,
        feasible_center_xy_m=feasible,
        fixed_point_residuals_m=residuals,
        radial_bounds_m=(float(radial_bounds_m[0]), float(radial_bounds_m[1])),
        fixed_point_assumption_fails=bool(np.max(residuals) > transverse_tolerance_m),
    )


def shared_vertical_offset_feasible_center_z(
    scattering_z_m: Sequence[float], offset_bounds_m: tuple[float, float]
) -> tuple[float, float]:
    """Return a scalar offset envelope; joint stratum support uses the named solver below."""

    scattering = _readonly(scattering_z_m)
    lower, upper = (float(offset_bounds_m[0]), float(offset_bounds_m[1]))
    if scattering.ndim != 1 or scattering.size == 0 or not np.isfinite(scattering).all():
        raise ValueError("scattering_z_m must be a nonempty finite vector")
    if not np.isfinite([lower, upper]).all() or lower > upper:
        raise ValueError("vertical offset bounds must be finite and ordered")
    return (float(np.min(scattering) - upper), float(np.max(scattering) - lower))


def _rotate_xy(vector_xy: np.ndarray, angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(float(angle_deg))
    rotation = np.asarray(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
    return rotation @ vector_xy


@dataclass(frozen=True)
class SyntheticNuisanceStratum:
    """One fixed (pass, sector) witness under the q=c+rho R(u,alpha)+zeta ez rule."""

    pass_id: int
    sector_id: int
    antenna_xyz_m: np.ndarray
    look_direction_xy: np.ndarray
    observed_scattering_point_xyz_m: np.ndarray
    rho_m: float
    alpha_deg: float

    def __post_init__(self) -> None:
        antenna = _readonly(self.antenna_xyz_m)
        look = _readonly(self.look_direction_xy)
        observed = _readonly(self.observed_scattering_point_xyz_m)
        if int(self.pass_id) not in TOPHAT_PANEL_PASSES or int(self.sector_id) not in TOPHAT_PANEL_SECTORS:
            raise ValueError("synthetic strata must use the frozen TopHat pass/sector panel")
        if antenna.shape != (3,) or look.shape != (2,) or observed.shape != (3,):
            raise ValueError("synthetic stratum vectors have invalid shapes")
        if not np.isfinite(antenna).all() or not np.isfinite(look).all() or not np.isfinite(observed).all():
            raise ValueError("synthetic stratum geometry must be finite")
        if not np.isclose(np.linalg.norm(look), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("synthetic stratum look direction must be a horizontal unit vector")
        if not np.isfinite([self.rho_m, self.alpha_deg]).all() or self.rho_m < 0:
            raise ValueError("synthetic stratum nuisance values must be finite")
        object.__setattr__(self, "pass_id", int(self.pass_id))
        object.__setattr__(self, "sector_id", int(self.sector_id))
        object.__setattr__(self, "antenna_xyz_m", antenna)
        object.__setattr__(self, "look_direction_xy", look)
        object.__setattr__(self, "observed_scattering_point_xyz_m", observed)
        object.__setattr__(self, "rho_m", float(self.rho_m))
        object.__setattr__(self, "alpha_deg", float(self.alpha_deg))

    @property
    def key(self) -> tuple[int, int]:
        return self.pass_id, self.sector_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "pass_id": self.pass_id,
            "sector_id": self.sector_id,
            "antenna_xyz_m": self.antenna_xyz_m.tolist(),
            "look_direction_xy": self.look_direction_xy.tolist(),
            "observed_scattering_point_xyz_m": self.observed_scattering_point_xyz_m.tolist(),
            "rho_m": self.rho_m,
            "alpha_deg": self.alpha_deg,
            "equation": "q_s=c+rho_s*rotate_xy(u_s,alpha_s)+zeta*e_z",
            "look_direction_definition": "horizontal unit vector from antenna XYZ toward planted synthetic center",
        }


@dataclass(frozen=True)
class SyntheticAspectNuisanceRule:
    """Fixed synthetic tube rule; bands are sensitivity assumptions, not data facts."""

    mode: str = "bounded_shared_zeta"
    rho_bounds_m: tuple[float, float] = SYNTHETIC_RADIAL_BOUNDS_M
    alpha_bounds_deg: tuple[float, float] = SYNTHETIC_ALPHA_BOUNDS_DEG
    shared_zeta_bounds_m: tuple[float, float] = SYNTHETIC_SHARED_ZETA_BOUNDS_M
    known_zeta_m: float | None = None
    horizontal_tolerance_m: float = 1.0e-9
    vertical_tolerance_m: float = 1.0e-9
    quorum_fraction: float = SYNTHETIC_ROBUST_QUORUM_FRACTION
    minimum_quorum: int = SYNTHETIC_MIN_ROBUST_QUORUM

    def __post_init__(self) -> None:
        if self.mode not in {"full_fixed", "bounded_shared_zeta"}:
            raise ValueError("synthetic nuisance mode must be full_fixed or bounded_shared_zeta")
        if tuple(float(v) for v in self.rho_bounds_m) != SYNTHETIC_RADIAL_BOUNDS_M:
            raise ValueError("synthetic rho sensitivity band is fixed at [0.75,1.25] m")
        if tuple(float(v) for v in self.alpha_bounds_deg) != SYNTHETIC_ALPHA_BOUNDS_DEG:
            raise ValueError("synthetic alpha sensitivity band is fixed at [-30,30] degrees")
        if tuple(float(v) for v in self.shared_zeta_bounds_m) != SYNTHETIC_SHARED_ZETA_BOUNDS_M:
            raise ValueError("synthetic shared-zeta sensitivity band is fixed at [-0.5,0.5] m")
        if self.mode == "full_fixed" and self.known_zeta_m is None:
            raise ValueError("full_fixed synthetic rule requires one known zeta")
        if self.known_zeta_m is not None and not (
            self.shared_zeta_bounds_m[0] <= float(self.known_zeta_m) <= self.shared_zeta_bounds_m[1]
        ):
            raise ValueError("known zeta must remain inside the stated shared-zeta band")
        if not np.isfinite([self.horizontal_tolerance_m, self.vertical_tolerance_m]).all() or min(
            self.horizontal_tolerance_m, self.vertical_tolerance_m
        ) <= 0:
            raise ValueError("synthetic nuisance tolerances must be positive and finite")
        if not 0.0 < self.quorum_fraction <= 1.0 or self.minimum_quorum < 1:
            raise ValueError("synthetic robust quorum settings are invalid")

    def quorum_for(self, stratum_count: int) -> int:
        stratum_count = int(stratum_count)
        if stratum_count <= 0:
            raise ValueError("robust quorum requires at least one predeclared stratum")
        return max(self.minimum_quorum, int(np.ceil(self.quorum_fraction * stratum_count)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "equation": "q_s=c+rho_s*rotate_xy(u_s,alpha_s)+zeta*e_z",
            "rho_bounds_m": list(self.rho_bounds_m),
            "alpha_bounds_deg": list(self.alpha_bounds_deg),
            "shared_zeta_bounds_m": list(self.shared_zeta_bounds_m),
            "known_zeta_m": self.known_zeta_m,
            "rho_alpha_scope": "one fixed nuisance pair per predeclared (pass,sector) stratum; never per pulse/frequency",
            "zeta_scope": "one shared variable across all eligible strata" if self.mode == "bounded_shared_zeta" else "one known fixed value",
            "quorum_fraction": self.quorum_fraction,
            "minimum_quorum": self.minimum_quorum,
            "quorum_interpretation": "fixed robustness/sensitivity assumption, not statistical confidence",
            "source_af": "fixed supplied conditioning; cannot independently validate autofocus",
        }


def _horizontal_nuisance_match(
    stratum: SyntheticNuisanceStratum,
    candidate_xy_m: np.ndarray,
    rule: SyntheticAspectNuisanceRule,
) -> bool:
    look_vector = candidate_xy_m - stratum.antenna_xyz_m[:2]
    look_norm = float(np.linalg.norm(look_vector))
    if look_norm == 0.0:
        return False
    look_direction = look_vector / look_norm
    delta = stratum.observed_scattering_point_xyz_m[:2] - candidate_xy_m
    rho = float(np.linalg.norm(delta))
    if rho == 0.0:
        return False
    dot = float(np.dot(look_direction, delta / rho))
    delta_unit = delta / rho
    cross = float(look_direction[0] * delta_unit[1] - look_direction[1] * delta_unit[0])
    alpha = float(np.rad2deg(np.arctan2(cross, dot)))
    if rule.mode == "full_fixed":
        return bool(
            abs(rho - stratum.rho_m) <= rule.horizontal_tolerance_m
            and abs(alpha - stratum.alpha_deg) <= rule.horizontal_tolerance_m
        )
    return bool(
        rule.rho_bounds_m[0] - rule.horizontal_tolerance_m <= rho <= rule.rho_bounds_m[1] + rule.horizontal_tolerance_m
        and rule.alpha_bounds_deg[0] - rule.horizontal_tolerance_m <= alpha <= rule.alpha_bounds_deg[1] + rule.horizontal_tolerance_m
    )


def _shared_zeta_support(
    eligible: Sequence[SyntheticNuisanceStratum],
    candidate_z_m: float,
    rule: SyntheticAspectNuisanceRule,
) -> tuple[int, float | None]:
    if not eligible:
        return 0, None
    if rule.mode == "full_fixed":
        count = sum(
            abs(float(stratum.observed_scattering_point_xyz_m[2]) - float(candidate_z_m) - float(rule.known_zeta_m))
            <= rule.vertical_tolerance_m
            for stratum in eligible
        )
        return int(count), None if count == 0 else float(rule.known_zeta_m)
    # All strata share one zeta.  Maximize support over the finite union of
    # interval endpoints and midpoints; no independent per-stratum zeta exists.
    intervals = []
    for stratum in eligible:
        center = float(stratum.observed_scattering_point_xyz_m[2]) - float(candidate_z_m)
        low = max(rule.shared_zeta_bounds_m[0], center - rule.vertical_tolerance_m)
        high = min(rule.shared_zeta_bounds_m[1], center + rule.vertical_tolerance_m)
        if low <= high:
            intervals.append((low, high))
    if not intervals:
        return 0, None
    witnesses = {float(rule.shared_zeta_bounds_m[0]), float(rule.shared_zeta_bounds_m[1])}
    for low, high in intervals:
        witnesses.update((float(low), float(high), float((low + high) / 2.0)))
    best_count = 0
    best_zeta: float | None = None
    for zeta in sorted(witnesses):
        count = sum(low <= zeta <= high for low, high in intervals)
        if count > best_count:
            best_count = int(count)
            best_zeta = float(zeta)
    return best_count, best_zeta


@dataclass(frozen=True)
class SyntheticRobustFeasibleSet:
    candidate_centers_xyz_m: np.ndarray
    support_counts: np.ndarray
    witness_shared_zeta_m: np.ndarray
    stratum_count: int
    quorum_k: int
    rule: SyntheticAspectNuisanceRule
    status: str = CONDITIONAL_XY_FEASIBLE_SET_UNREGISTERED

    def __post_init__(self) -> None:
        centers = _readonly(self.candidate_centers_xyz_m)
        supports = np.asarray(self.support_counts, dtype=np.int64).copy()
        witnesses = _readonly(self.witness_shared_zeta_m)
        if centers.ndim != 2 or centers.shape[1] != 3 or supports.shape != (centers.shape[0],) or witnesses.shape != (centers.shape[0],):
            raise ValueError("synthetic feasible-set arrays have inconsistent shapes")
        if np.any(supports < 0) or np.any(supports > int(self.stratum_count)):
            raise ValueError("synthetic feasible-set support counts are invalid")
        supports.setflags(write=False)
        object.__setattr__(self, "candidate_centers_xyz_m", centers)
        object.__setattr__(self, "support_counts", supports)
        object.__setattr__(self, "witness_shared_zeta_m", witnesses)

    @property
    def single_supplied_stencil_candidate(self) -> bool:
        """True only when exactly one candidate from the supplied stencil survives."""

        return bool(self.candidate_centers_xyz_m.shape[0] == 1)

    @property
    def z_interval_m(self) -> tuple[float, float] | None:
        if not self.candidate_centers_xyz_m.shape[0]:
            return None
        return (
            float(np.min(self.candidate_centers_xyz_m[:, 2])),
            float(np.max(self.candidate_centers_xyz_m[:, 2])),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_centers_xyz_m": self.candidate_centers_xyz_m.tolist(),
            "candidate_count": int(self.candidate_centers_xyz_m.shape[0]),
            "support_counts": self.support_counts.tolist(),
            "witness_shared_zeta_m": self.witness_shared_zeta_m.tolist(),
            "stratum_count": self.stratum_count,
            "quorum_k": self.quorum_k,
            "single_supplied_stencil_candidate": self.single_supplied_stencil_candidate,
            "uniqueness_interpretation": "stencil cardinality only; not general or physical uniqueness",
            "z_interval_m": None if self.z_interval_m is None else list(self.z_interval_m),
            "rule": self.rule.as_dict(),
            "status": self.status,
            "quorum_interpretation": "fixed robustness/sensitivity assumption, not statistical confidence",
        }


def synthetic_robust_feasible_set(
    strata: Sequence[SyntheticNuisanceStratum],
    rule: SyntheticAspectNuisanceRule,
    candidate_centers_xyz_m: Sequence[Sequence[float]],
) -> SyntheticRobustFeasibleSet:
    """Evaluate fixed-quorum support with exactly one shared zeta variable."""

    records = tuple(sorted(strata, key=lambda item: (item.pass_id, item.sector_id)))
    if not records:
        raise ValueError("synthetic robust feasible set requires predeclared strata")
    keys = [record.key for record in records]
    if len(set(keys)) != len(keys):
        raise ValueError("synthetic strata must have unique (pass,sector) keys")
    candidates = np.asarray(candidate_centers_xyz_m, dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1] != 3 or candidates.shape[0] == 0:
        raise ValueError("synthetic candidate stencil must have shape [candidate,3]")
    if candidates.shape[0] > 4096:
        raise TophatLocalizationBlocked(BLOCKED_RESOURCE_CAP_EXCEEDED, "synthetic candidate stencil exceeds the bounded cap")
    _finite(candidates, "synthetic candidate centers")
    quorum = rule.quorum_for(len(records))
    kept: list[np.ndarray] = []
    supports: list[int] = []
    witnesses: list[float] = []
    for candidate in candidates:
        eligible = [record for record in records if _horizontal_nuisance_match(record, candidate[:2], rule)]
        support, witness = _shared_zeta_support(eligible, float(candidate[2]), rule)
        if support >= quorum:
            kept.append(candidate)
            supports.append(support)
            witnesses.append(np.nan if witness is None else witness)
    status = CONDITIONAL_XY_FEASIBLE_SET_UNREGISTERED if kept else INCONCLUSIVE_ASPECT_DEPENDENT_EVIDENCE
    if rule.mode == "bounded_shared_zeta" and kept:
        kept_z = np.asarray(kept, dtype=np.float64)[:, 2]
        if float(np.max(kept_z) - np.min(kept_z)) <= rule.vertical_tolerance_m:
            status = INCONCLUSIVE_ASPECT_DEPENDENT_EVIDENCE
    return SyntheticRobustFeasibleSet(
        candidate_centers_xyz_m=np.asarray(kept, dtype=np.float64).reshape((-1, 3)) if kept else np.empty((0, 3), dtype=np.float64),
        support_counts=np.asarray(supports, dtype=np.int64),
        witness_shared_zeta_m=np.asarray(witnesses, dtype=np.float64),
        stratum_count=len(records),
        quorum_k=quorum,
        rule=rule,
        status=status,
    )


def joint_shared_zeta_feasible_set(
    strata: Sequence[SyntheticNuisanceStratum],
    *,
    candidate_centers_xyz_m: Sequence[Sequence[float]],
    radial_bounds_m: tuple[float, float] = SYNTHETIC_RADIAL_BOUNDS_M,
    alpha_bounds_deg: tuple[float, float] = SYNTHETIC_ALPHA_BOUNDS_DEG,
    shared_zeta_bounds_m: tuple[float, float] = SYNTHETIC_SHARED_ZETA_BOUNDS_M,
) -> SyntheticRobustFeasibleSet:
    """Project one joint (c,zeta) support relation onto candidate centers."""

    if tuple(float(v) for v in radial_bounds_m) != SYNTHETIC_RADIAL_BOUNDS_M:
        raise ValueError("joint synthetic feasibility requires the fixed radial sensitivity band")
    if tuple(float(v) for v in alpha_bounds_deg) != SYNTHETIC_ALPHA_BOUNDS_DEG:
        raise ValueError("joint synthetic feasibility requires the fixed alpha sensitivity band")
    if tuple(float(v) for v in shared_zeta_bounds_m) != SYNTHETIC_SHARED_ZETA_BOUNDS_M:
        raise ValueError("joint synthetic feasibility requires the fixed shared-zeta sensitivity band")
    return synthetic_robust_feasible_set(
        strata,
        SyntheticAspectNuisanceRule(
            mode="bounded_shared_zeta",
            rho_bounds_m=radial_bounds_m,
            alpha_bounds_deg=alpha_bounds_deg,
            shared_zeta_bounds_m=shared_zeta_bounds_m,
        ),
        candidate_centers_xyz_m,
    )


def synthetic_full_fixed_rule_recovery(
    strata: Sequence[SyntheticNuisanceStratum], *, known_zeta_m: float = 0.0
) -> np.ndarray:
    """Recover a planted center only when rho, alpha, zeta and look mapping are fixed."""

    records = tuple(strata)
    if not records:
        raise ValueError("full-rule synthetic recovery requires strata")
    rule = SyntheticAspectNuisanceRule(mode="full_fixed", known_zeta_m=float(known_zeta_m))
    # Solve the fixed equation with the candidate-dependent horizontal look
    # direction u_s(c), rather than treating the stored direction as a free
    # or ground-truth return field.  The antenna is distant in this fixture,
    # so the deterministic fixed-point iteration is tightly contractive.
    center_xy = np.mean(np.asarray([record.observed_scattering_point_xyz_m[:2] for record in records]), axis=0)
    for _ in range(64):
        updates = []
        for record in records:
            look_vector = center_xy - record.antenna_xyz_m[:2]
            look_vector /= np.linalg.norm(look_vector)
            updates.append(
                record.observed_scattering_point_xyz_m[:2]
                - record.rho_m * _rotate_xy(look_vector, record.alpha_deg)
            )
        update = np.mean(np.asarray(updates), axis=0)
        if np.linalg.norm(update - center_xy) <= 1.0e-13:
            center_xy = update
            break
        center_xy = update
    center_z = float(np.mean([record.observed_scattering_point_xyz_m[2] for record in records]) - known_zeta_m)
    center = np.asarray((*center_xy, center_z), dtype=np.float64)
    if not all(_horizontal_nuisance_match(record, center[:2], rule) for record in records):
        raise ValueError("full fixed synthetic rule is inconsistent across strata")
    support, _ = _shared_zeta_support(records, center_z, rule)
    if support != len(records):
        raise ValueError("full fixed synthetic rule did not recover every stratum")
    return _readonly(center)


def synthetic_nuisance_strata(
    fixture: "SyntheticTophatCylinderRing | None" = None,
    *,
    shared_zeta_m: float = 0.0,
) -> tuple[SyntheticNuisanceStratum, ...]:
    """Build 32 fixed (pass,sector) strata from actual synthetic antenna XYZ."""

    fixture = SyntheticTophatCylinderRing() if fixture is None else fixture
    if not np.isfinite(shared_zeta_m) or not (
        SYNTHETIC_SHARED_ZETA_BOUNDS_M[0] <= shared_zeta_m <= SYNTHETIC_SHARED_ZETA_BOUNDS_M[1]
    ):
        raise ValueError("shared synthetic zeta must remain in the declared band")
    records: list[SyntheticNuisanceStratum] = []
    for pass_id in TOPHAT_PANEL_PASSES:
        elevation = float(fixture.elevations_deg[(pass_id - 1) % fixture.elevations_deg.size])
        for index, sector_id in enumerate(TOPHAT_PANEL_SECTORS):
            azimuth = float(fixture.aspects_deg[index % fixture.aspects_deg.size] + 0.7 * (pass_id - 1))
            angle = np.deg2rad(azimuth)
            antenna = np.asarray(
                [120.0 * np.cos(angle) + 0.4 * pass_id, 120.0 * np.sin(angle) - 0.3 * pass_id, 15.0 + elevation],
                dtype=np.float64,
            )
            look = fixture.geometric_center_xyz_m[:2] - antenna[:2]
            look /= np.linalg.norm(look)
            rho = float(fixture.radial_base_m + fixture.radial_aspect_amplitude_m * np.cos(angle + 0.11 * pass_id))
            alpha = float(12.0 * np.sin(np.deg2rad(azimuth + 13.0 * pass_id)))
            observed = np.asarray(
                [
                    *(fixture.geometric_center_xyz_m[:2] + rho * _rotate_xy(look, alpha)),
                    fixture.geometric_center_xyz_m[2] + shared_zeta_m,
                ],
                dtype=np.float64,
            )
            records.append(
                SyntheticNuisanceStratum(
                    pass_id=pass_id,
                    sector_id=sector_id,
                    antenna_xyz_m=antenna,
                    look_direction_xy=look,
                    observed_scattering_point_xyz_m=observed,
                    rho_m=rho,
                    alpha_deg=alpha,
                )
            )
    return tuple(records)


def synthetic_candidate_stencil(center_xyz_m: Sequence[float], *, halfwidth_m: float = 0.4, grid_size: int = 5) -> np.ndarray:
    """Small fixed 3-D stencil; never the 40^3 TopHat reporting grid."""

    center = np.asarray(center_xyz_m, dtype=np.float64)
    if center.shape != (3,) or grid_size <= 0 or grid_size > 9:
        raise ValueError("synthetic candidate stencil requires a 3-vector and grid_size <= 9")
    xy = np.linspace(-float(halfwidth_m), float(halfwidth_m), int(grid_size))
    z = np.asarray([center[2] - 0.4, center[2] - 0.2, center[2], center[2] + 0.2, center[2] + 0.4])
    points = np.asarray([[center[0] + dx, center[1] + dy, zz] for dx in xy for dy in xy for zz in z], dtype=np.float64)
    return _readonly(points)


def synthetic_full_response_candidate_stencil(
    seed: TophatDiagramXYSeed = DEFAULT_TOPHAT_DIAGRAM_XY_SEED,
) -> np.ndarray:
    """Build the declared 5 x 5 x 7 response stencil before any truth is planted."""

    if not isinstance(seed, TophatDiagramXYSeed):
        raise TypeError("the full response stencil requires a qualified TophatDiagramXYSeed")
    xy = np.asarray(seed.xy_m, dtype=np.float64)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("the qualified response-stencil seed must contain finite XY coordinates")
    points = np.asarray(
        [
            [xy[0] + dx, xy[1] + dy, z]
            for dx in SYNTHETIC_RESPONSE_STENCIL_OFFSETS_XY_M
            for dy in SYNTHETIC_RESPONSE_STENCIL_OFFSETS_XY_M
            for z in SYNTHETIC_RESPONSE_STENCIL_Z_M
        ],
        dtype=np.float64,
    )
    if points.shape != (SYNTHETIC_RESPONSE_CANDIDATE_COUNT, 3):
        raise AssertionError("the declared response stencil must contain exactly 175 candidates")
    return _readonly(points)


@dataclass(frozen=True)
class SyntheticTophatResponseModel:
    """Fixed synthetic response model used only for native-complex mechanics."""

    rho_m: float = 1.0
    alpha_deg: float = 0.0
    zeta_m: float = 0.0
    model_name: str = "synthetic_global_fixed_far_side_response_v1"
    source_af_conditioning: str = "fixed supplied source-AF representation; no response-side refit"
    side_convention: str = (
        "look=candidate-antenna is the antenna-to-candidate horizontal vector; positive rho with alpha=0 "
        "uses that vector; "
        "synthetic far-side sensitivity only, not established measured TopHat physics"
    )

    def __post_init__(self) -> None:
        if float(self.rho_m) != 1.0 or float(self.alpha_deg) != 0.0 or float(self.zeta_m) != 0.0:
            raise ValueError("the native-complex response validator uses the single fixed rho=1, alpha=0, zeta=0 triplet")
        if not np.isfinite(float(self.rho_m)) or not (SYNTHETIC_RADIAL_BOUNDS_M[0] <= float(self.rho_m) <= SYNTHETIC_RADIAL_BOUNDS_M[1]):
            raise ValueError("synthetic global rho must remain inside the declared band")
        if not np.isfinite(float(self.alpha_deg)) or not (SYNTHETIC_ALPHA_BOUNDS_DEG[0] <= float(self.alpha_deg) <= SYNTHETIC_ALPHA_BOUNDS_DEG[1]):
            raise ValueError("synthetic global alpha must remain inside the declared band")
        if not np.isfinite(float(self.zeta_m)) or not (
            SYNTHETIC_SHARED_ZETA_BOUNDS_M[0] <= float(self.zeta_m) <= SYNTHETIC_SHARED_ZETA_BOUNDS_M[1]
        ):
            raise ValueError("synthetic shared zeta must remain inside the declared band")
        object.__setattr__(self, "rho_m", float(self.rho_m))
        object.__setattr__(self, "alpha_deg", float(self.alpha_deg))
        object.__setattr__(self, "zeta_m", float(self.zeta_m))

    @classmethod
    def declared_default(cls) -> "SyntheticTophatResponseModel":
        return cls(rho_m=1.0, alpha_deg=0.0, zeta_m=0.0)

    def rho_for(self, identity: Any) -> float:
        if _identity_key(identity)[0] not in TOPHAT_PANEL_PASSES or _identity_key(identity)[2] not in TOPHAT_PANEL_SECTORS:
            raise ValueError("identity is outside the declared synthetic response panel")
        return self.rho_m

    def alpha_for(self, identity: Any) -> float:
        if _identity_key(identity)[0] not in TOPHAT_PANEL_PASSES or _identity_key(identity)[2] not in TOPHAT_PANEL_SECTORS:
            raise ValueError("identity is outside the declared synthetic response panel")
        return self.alpha_deg

    def q_xyz(
        self,
        candidate_xyz_m: Sequence[float],
        *,
        position_xyz_m: Sequence[float],
        identity: Any,
    ) -> np.ndarray:
        candidate = np.asarray(candidate_xyz_m, dtype=np.float64)
        antenna = np.asarray(position_xyz_m, dtype=np.float64)
        if candidate.shape != (3,) or antenna.shape != (3,) or not np.isfinite(candidate).all() or not np.isfinite(antenna).all():
            raise ValueError("synthetic response geometry requires finite XYZ vectors")
        look = candidate[:2] - antenna[:2]
        norm = float(np.linalg.norm(look))
        if norm <= 0.0:
            raise ValueError("synthetic response candidate cannot coincide horizontally with an antenna")
        look = look / norm
        offset_xy = self.rho_for(identity) * _rotate_xy(look, self.alpha_for(identity))
        return np.asarray(
            [candidate[0] + offset_xy[0], candidate[1] + offset_xy[1], candidate[2] + self.zeta_m],
            dtype=np.float64,
        )

    def kernel_for_geometry(
        self,
        candidate_xyz_m: Sequence[float],
        *,
        position_xyz_m: Sequence[float],
        identity: Any,
        frequencies_hz: Sequence[float],
        effective_r0_m: float,
    ) -> np.ndarray:
        q = self.q_xyz(candidate_xyz_m, position_xyz_m=position_xyz_m, identity=identity)
        frequencies = np.asarray(frequencies_hz, dtype=np.float64)
        if frequencies.ndim != 1 or frequencies.size == 0 or not np.isfinite(frequencies).all():
            raise ValueError("synthetic response frequencies must be a finite nonempty vector")
        distance = float(np.linalg.norm(q - np.asarray(position_xyz_m, dtype=np.float64)))
        phase = -1j * (4.0 * np.pi / SPEED_OF_LIGHT_M_S) * (distance - float(effective_r0_m)) * frequencies
        return np.exp(phase).astype(np.complex128, copy=False)

    def kernel(self, candidate_xyz_m: Sequence[float], record: SourceAFObservation) -> np.ndarray:
        if not isinstance(record, SourceAFObservation):
            raise TypeError("synthetic response kernels require SourceAFObservation records")
        return self.kernel_for_geometry(
            candidate_xyz_m,
            position_xyz_m=record.position_xyz_m,
            identity=record.identity,
            frequencies_hz=record.frequencies_hz,
            effective_r0_m=record.effective_r0_m,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "equation": "q_i(c)=c+rho*R_xy(alpha)u_i(c)+zeta*e_z; u_i(c)=normalize((c_xy-a_i_xy,0))",
            "rho_scope": "one globally fixed rho across all 96 records",
            "alpha_scope": "one globally fixed alpha across all 96 records",
            "rho_m": self.rho_m,
            "alpha_deg": self.alpha_deg,
            "fixed_triplet": "rho=1.0 m, alpha=0 deg, zeta=0.0 m across all 96 records",
            "rho_bounds_m": list(SYNTHETIC_RADIAL_BOUNDS_M),
            "alpha_bounds_deg": list(SYNTHETIC_ALPHA_BOUNDS_DEG),
            "shared_zeta_m": self.zeta_m,
            "shared_zeta_bounds_m": list(SYNTHETIC_SHARED_ZETA_BOUNDS_M),
            "source_af_conditioning": self.source_af_conditioning,
            "source_af_application": "source-AF applied exactly once by the generator before evaluator entry",
            "gain_scope": "one global complex gain across all records and frequencies",
            "side_convention": self.side_convention,
            "measured_physics_claim": False,
        }


def synthetic_declared_response_model() -> SyntheticTophatResponseModel:
    return SyntheticTophatResponseModel.declared_default()


_SYNTHETIC_RESPONSE_PULSE_INDICES_BY_PASS_SECTOR: Mapping[tuple[int, int], tuple[int, int, int]] = MappingProxyType(
    {
        (pass_id, sector_id): (0, (count - 1) // 2, count - 1)
        for pass_id, counts in SYNTHETIC_RESPONSE_SHARD_COUNTS_BY_PASS_SECTOR.items()
        for sector_id, count in counts.items()
    }
)


def _synthetic_response_pulse_indices(pass_id: int, sector_id: int) -> tuple[int, int, int]:
    try:
        return _SYNTHETIC_RESPONSE_PULSE_INDICES_BY_PASS_SECTOR[(int(pass_id), int(sector_id))]
    except KeyError as exc:
        raise ValueError("synthetic response pulse selection is outside the documented pass/sector table") from exc


def _synthetic_response_antenna(pass_id: int, sector_id: int, pulse_index: int) -> np.ndarray:
    angle = np.deg2rad(float(sector_id - 2) + 0.11 * pass_id + 0.019 * pulse_index)
    radius = 112.0 + 0.55 * np.sin(0.4 * pass_id + 0.01 * sector_id) + 0.08 * pulse_index
    return np.asarray(
        [
            radius * np.cos(angle) + 0.15 * pass_id,
            radius * np.sin(angle) - 0.12 * pass_id,
            12.0 + 0.85 * pass_id + 0.035 * pulse_index,
        ],
        dtype=np.float64,
    )


def _synthetic_response_r0_source(pass_id: int, sector_id: int, pulse_index: int) -> float:
    return float(119.0 + 0.31 * pass_id + 0.002 * sector_id + 0.007 * pulse_index)


@dataclass(frozen=True)
class SyntheticSourceAFEnvelope:
    """Ragged 96-row source-AF envelope with no held-out truth fields."""

    records: tuple[SourceAFObservation, ...]
    scenario: str
    frequency_sample_count: int = 41_160
    frequency_endpoints_hz_by_pass: Mapping[int, tuple[float, float]] = field(
        default_factory=lambda: MappingProxyType(dict(SYNTHETIC_RESPONSE_FREQUENCY_ENDPOINTS_BY_PASS_HZ))
    )

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if len(records) != 96:
            raise ValueError("synthetic native-complex response envelope requires exactly 96 rows")
        expected_keys: list[tuple[int, str, int, int]] = []
        for pass_id in TOPHAT_PANEL_PASSES:
            for sector_id in TOPHAT_PANEL_SECTORS:
                expected_keys.extend(
                    (pass_id, TOPHAT_PANEL_POLARIZATION, sector_id, pulse)
                    for pulse in _synthetic_response_pulse_indices(pass_id, sector_id)
                )
        keys = tuple(_identity_key(record.identity) for record in records)
        if keys != tuple(expected_keys):
            raise ValueError("synthetic envelope identities must use the documented canonical 96-row order")
        counts = SYNTHETIC_RESPONSE_FREQUENCY_COUNTS_BY_PASS
        for record in records:
            if not isinstance(record, SourceAFObservation):
                raise TypeError("synthetic envelope records must already be SourceAFObservation views")
            if record.role != "train" or record.identity.polarization != "hh":
                raise ValueError("synthetic envelope is limited to HH TRAIN source-AF rows")
            expected_count = int(counts[int(record.identity.pass_id)])
            frequencies = np.asarray(record.frequencies_hz, dtype=np.float64)
            if frequencies.size != expected_count:
                raise ValueError("synthetic envelope cannot uniformize the documented ragged pass vectors")
            expected_endpoints = self.frequency_endpoints_hz_by_pass[int(record.identity.pass_id)]
            if not np.isclose(frequencies[0], expected_endpoints[0], rtol=0.0, atol=0.0) or not np.isclose(
                frequencies[-1], expected_endpoints[1], rtol=0.0, atol=0.0
            ):
                raise ValueError("synthetic envelope frequency endpoints drifted")
            if int(record.identity.pulse_index) not in _synthetic_response_pulse_indices(
                int(record.identity.pass_id), int(record.identity.sector_id)
            ):
                raise ValueError("synthetic envelope pulse index is not a documented first/middle/last row")
        actual_count = sum(int(record.frequencies_hz.size) for record in records)
        if actual_count != int(self.frequency_sample_count):
            raise ValueError("synthetic envelope frequency sample count drifted")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "scenario", str(self.scenario))
        object.__setattr__(self, "frequency_endpoints_hz_by_pass", MappingProxyType(dict(self.frequency_endpoints_hz_by_pass)))

    @property
    def response_energy(self) -> float:
        return float(sum(np.vdot(record.effective_response, record.effective_response).real for record in self.records))

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "record_count": len(self.records),
            "frequency_sample_count": self.frequency_sample_count,
            "pass_ids": list(TOPHAT_PANEL_PASSES),
            "polarization": TOPHAT_PANEL_POLARIZATION,
            "role": "train",
            "sectors": list(TOPHAT_PANEL_SECTORS),
            "frequency_counts_by_pass": {str(key): value for key, value in SYNTHETIC_RESPONSE_FREQUENCY_COUNTS_BY_PASS.items()},
            "frequency_endpoints_hz_by_pass": {
                str(key): list(value) for key, value in self.frequency_endpoints_hz_by_pass.items()
            },
            "shard_counts_by_pass_sector": {
                str(pass_id): {str(sector_id): count for sector_id, count in counts.items()}
                for pass_id, counts in SYNTHETIC_RESPONSE_SHARD_COUNTS_BY_PASS_SECTOR.items()
            },
            "pulse_indices_by_pass_sector": {
                f"{pass_id}/{sector_id}": list(_synthetic_response_pulse_indices(pass_id, sector_id))
                for pass_id in TOPHAT_PANEL_PASSES
                for sector_id in TOPHAT_PANEL_SECTORS
            },
            "canonical_identity_order": "(pass,polarization,sector,pulse)",
            "source_af_representation": "fixed source_af applied exactly once by the generator",
            "header_shape_status": "SYNTHETIC_HEADER_SHAPED",
            "antenna_and_metadata": "synthetic antenna XYZ/r0/phase metadata; no PACE payload or native target coordinate",
            "frequency_vector_status": "linear synthetic frequency analogues matching documented counts/endpoints; not exact stored native vectors",
            "measured_response_payload_access": False,
            "native_target_claim": False,
            "held_out_truth": "not present in the evaluator envelope",
        }


@dataclass(frozen=True)
class SyntheticHeldOutResponseCase:
    """Harness-owned truth paired with a truth-free evaluator envelope."""

    envelope: SyntheticSourceAFEnvelope
    truth_center_xyz_m: np.ndarray
    global_gain: complex
    scenario: str
    source_af_application_count: int = 1
    truth_forward_sample_terms: int = 41_160
    record_gain_profile: tuple[complex, ...] = ()

    def __post_init__(self) -> None:
        truth = _readonly(self.truth_center_xyz_m, dtype=np.float64)
        if truth.shape != (3,) or not np.isfinite(truth).all():
            raise ValueError("synthetic held-out truth center must be a finite XYZ vector")
        if not np.isfinite([complex(self.global_gain).real, complex(self.global_gain).imag]).all():
            raise ValueError("synthetic global gain must be finite")
        if int(self.source_af_application_count) != 1:
            raise ValueError("synthetic generator must apply source-AF exactly once")
        if int(self.truth_forward_sample_terms) != int(self.envelope.frequency_sample_count):
            raise ValueError("synthetic generator truth-forward ledger must equal F")
        profile = tuple(complex(value) for value in self.record_gain_profile)
        if profile and len(profile) != len(self.envelope.records):
            raise ValueError("synthetic record-gain profile must align to the 96 envelope rows")
        object.__setattr__(self, "truth_center_xyz_m", truth)
        object.__setattr__(self, "global_gain", complex(self.global_gain))
        object.__setattr__(self, "scenario", str(self.scenario))
        object.__setattr__(self, "record_gain_profile", profile)

    def as_dict(self) -> dict[str, Any]:
        return {
            "envelope": self.envelope.as_dict(),
            "scenario": self.scenario,
            "source_af_application_count": self.source_af_application_count,
            "truth_separation": "truth center and gain are harness-only and are omitted from evaluator results",
        }


def synthetic_native_complex_response_case(
    truth_center_xyz_m: Sequence[float],
    *,
    model: SyntheticTophatResponseModel | None = None,
    scenario: str = "on_grid",
    global_gain: complex = 1.17 - 0.63j,
) -> SyntheticHeldOutResponseCase:
    """Generate one ragged raw envelope, then perform exactly one source-AF conversion."""

    model = synthetic_declared_response_model() if model is None else model
    if not isinstance(model, SyntheticTophatResponseModel):
        raise TypeError("synthetic response generation requires the declared response model")
    truth = np.asarray(truth_center_xyz_m, dtype=np.float64)
    if truth.shape != (3,) or not np.isfinite(truth).all():
        raise ValueError("synthetic response truth center must be a finite XYZ vector")
    scenario = str(scenario)
    allowed = {
        "on_grid",
        "off_grid",
        "off_grid_flat",
        "off_grid_mismatch",
        "off_grid_record_gain",
        "off_grid_boundary",
    }
    if scenario not in allowed:
        raise ValueError(f"unknown synthetic response scenario {scenario!r}")
    records: list[NativeObservation] = []
    flat_index = 0
    flat_rng = np.random.default_rng(0)
    record_gain_profile: list[complex] = []
    for pass_id in TOPHAT_PANEL_PASSES:
        frequencies = np.linspace(
            SYNTHETIC_RESPONSE_FREQUENCY_ENDPOINTS_BY_PASS_HZ[pass_id][0],
            SYNTHETIC_RESPONSE_FREQUENCY_ENDPOINTS_BY_PASS_HZ[pass_id][1],
            SYNTHETIC_RESPONSE_FREQUENCY_COUNTS_BY_PASS[pass_id],
            dtype=np.float64,
        )
        for sector_id in TOPHAT_PANEL_SECTORS:
            for pulse_index in _synthetic_response_pulse_indices(pass_id, sector_id):
                identity = NativeObservationId(pass_id, "hh", sector_id, pulse_index)
                position = _synthetic_response_antenna(pass_id, sector_id, pulse_index)
                r_correct = float(0.06 + 0.001 * pass_id + 0.0001 * (sector_id % 10))
                ph_correct = float(0.017 * pass_id + 0.0003 * (sector_id % 10) + 0.00001 * pulse_index)
                effective_r0 = _synthetic_response_r0_source(pass_id, sector_id, pulse_index)
                h_primary = model.kernel_for_geometry(
                    truth,
                    position_xyz_m=position,
                    identity=identity,
                    frequencies_hz=frequencies,
                    effective_r0_m=effective_r0,
                )
                if scenario == "off_grid_flat":
                    deterministic_complex = flat_rng.standard_normal(frequencies.size) + 1j * flat_rng.standard_normal(
                        frequencies.size
                    )
                    deterministic_complex /= np.sqrt(np.mean(np.abs(deterministic_complex) ** 2))
                    source_response = complex(global_gain) * deterministic_complex
                elif scenario == "off_grid_mismatch":
                    secondary_phase = 0.43 * np.arange(frequencies.size, dtype=np.float64) + 0.17 * pass_id
                    secondary = 0.47 * np.exp(1j * secondary_phase)
                    source_response = complex(global_gain) * (h_primary + secondary)
                elif scenario == "off_grid_record_gain":
                    record_gain = 0.55 + 0.40j * np.sin(0.37 * pass_id + 0.013 * sector_id + 0.021 * pulse_index)
                    source_response = complex(global_gain) * record_gain * h_primary
                    record_gain_profile.append(complex(record_gain))
                else:
                    source_response = complex(global_gain) * h_primary
                    record_gain_profile.append(1.0 + 0.0j)
                raw_response = np.asarray(source_response * np.exp(-1j * ph_correct), dtype=np.complex128)
                autofocus = AutofocusProvenance(
                    schema=AUTOFOCUS_PROVENANCE_SCHEMA,
                    mode=AUTOFOCUS_PUBLISHED,
                    official_available=True,
                    applied=False,
                    source_shard_id=f"synthetic_pass{pass_id}_hh",
                    range_field="r_correct",
                    phase_field="ph_correct",
                    range_unit="m",
                    phase_unit="rad",
                )
                records.append(
                    NativeObservation(
                        identity=identity,
                        role="train",
                        response=_readonly(raw_response, dtype=np.complex128),
                        frequencies_hz=_readonly(frequencies, dtype=np.float64),
                        position_xyz_m=_readonly(position, dtype=np.float64),
                        r0_m=float(np.float64(effective_r0 - r_correct)),
                        th_deg=float(sector_id - 2),
                        phi_deg=float(-8.0 + 2.0 * (pass_id - 1) + 0.01 * pulse_index),
                        r_correct_raw=r_correct,
                        ph_correct_raw=ph_correct,
                        phase_reference=PhaseReferenceContract(),
                        autofocus=autofocus,
                    )
                )
                flat_index += 1
    converted = build_multipass_source_af(
        tuple(records),
        expected_count=96,
        scope=MultipassHHTrainSourceAFScope(expected_count=96),
    )
    envelope = SyntheticSourceAFEnvelope(converted, scenario=scenario)
    return SyntheticHeldOutResponseCase(
        envelope=envelope,
        truth_center_xyz_m=truth,
        global_gain=complex(global_gain),
        scenario=scenario,
        record_gain_profile=tuple(record_gain_profile),
    )


@dataclass(frozen=True)
class SyntheticCandidateScore:
    candidate_xyz_m: np.ndarray
    gain_hat: complex
    target_energy: float
    gain_denominator: float
    gain_numerator: complex
    residual_energy: float
    loss: float

    def __post_init__(self) -> None:
        candidate = _readonly(self.candidate_xyz_m, dtype=np.float64)
        if candidate.shape != (3,) or not np.isfinite(candidate).all():
            raise ValueError("candidate score requires a finite XYZ candidate")
        object.__setattr__(self, "candidate_xyz_m", candidate)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_xyz_m": self.candidate_xyz_m.tolist(),
            "gain_hat": {"real": self.gain_hat.real, "imag": self.gain_hat.imag},
            "target_energy": self.target_energy,
            "gain_denominator": self.gain_denominator,
            "gain_numerator": {"real": self.gain_numerator.real, "imag": self.gain_numerator.imag},
            "residual_energy": self.residual_energy,
            "loss": self.loss,
        }


@dataclass(frozen=True)
class SyntheticCandidateScoreReport:
    """Gain-profiled diagnostic; it never claims physical localization."""

    scenario: str
    scores: tuple[SyntheticCandidateScore, ...]
    status: str
    target_energy: float
    best_index: int | None
    second_index: int | None
    relative_gap: float | None
    kernel_cache_entries: int
    kernel_rerender_count: int
    evaluator_truth_separation: str = "records/candidates/model only; no truth, XYZ scattering oracle, or residual map"

    @property
    def best_score(self) -> SyntheticCandidateScore | None:
        return None if self.best_index is None else self.scores[self.best_index]

    @property
    def best_candidate_xyz_m(self) -> np.ndarray | None:
        best = self.best_score
        return None if best is None else best.candidate_xyz_m

    def as_dict(self) -> dict[str, Any]:
        best = self.best_score
        return {
            "scenario": self.scenario,
            "status": self.status,
            "candidate_count": len(self.scores),
            "target_energy": self.target_energy,
            "best_candidate_xyz_m": None if best is None else best.candidate_xyz_m.tolist(),
            "best_loss": None if best is None else best.loss,
            "best_gain_hat": None if best is None else {"real": best.gain_hat.real, "imag": best.gain_hat.imag},
            "second_best_loss": None if self.second_index is None else self.scores[self.second_index].loss,
            "relative_gap": self.relative_gap,
            "kernel_cache_entries": self.kernel_cache_entries,
            "kernel_rerender_count": self.kernel_rerender_count,
            "evaluator_truth_separation": self.evaluator_truth_separation,
            "diagnostic_scope": "synthetic sampled discrimination mechanics only; no physical recovery or placement claim",
            "top_scores": [score.as_dict() for score in self.scores[:3]],
        }


@dataclass(frozen=True)
class SyntheticResponseWorkLedger:
    """Explicit bounded-work ledger, with dot arithmetic separate from kernel terms."""

    frequency_sample_count: int
    candidate_count: int
    candidate_kernel_sample_terms: int
    truth_generator_sample_terms: int
    on_grid_score_sample_terms: int
    off_grid_score_sample_terms: int
    final_recheck_control_terms: int
    mismatch_secondary_generator_sample_terms: int = 0
    target_energy_dot_terms: int = 0
    denominator_dot_terms: int = 0
    gain_numerator_dot_terms: int = 0
    residual_dot_terms: int = 0
    header_gate_record_ops: int = 0
    source_af_metadata_record_ops: int = 0
    source_af_conversion_record_ops: int = 0
    candidate_score_ops: int = 0
    candidate_cache_entries: int = 0
    hidden_adaptive_loops: int = 0
    sample_equivalent_terms: int | None = None
    max_sample_equivalent_terms: int = SYNTHETIC_RESPONSE_WORK_CAP

    def __post_init__(self) -> None:
        integer_fields = (
            "frequency_sample_count",
            "candidate_count",
            "candidate_kernel_sample_terms",
            "truth_generator_sample_terms",
            "on_grid_score_sample_terms",
            "off_grid_score_sample_terms",
            "final_recheck_control_terms",
            "mismatch_secondary_generator_sample_terms",
            "target_energy_dot_terms",
            "denominator_dot_terms",
            "gain_numerator_dot_terms",
            "residual_dot_terms",
            "header_gate_record_ops",
            "source_af_metadata_record_ops",
            "source_af_conversion_record_ops",
            "candidate_score_ops",
            "candidate_cache_entries",
            "hidden_adaptive_loops",
        )
        for name in integer_fields:
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        total = (
            self.candidate_kernel_sample_terms
            + self.truth_generator_sample_terms
            + self.on_grid_score_sample_terms
            + self.off_grid_score_sample_terms
            + self.mismatch_secondary_generator_sample_terms
            + self.final_recheck_control_terms
        )
        supplied = total if self.sample_equivalent_terms is None else int(self.sample_equivalent_terms)
        if supplied != total:
            raise ValueError("sample-equivalent ledger total must reconcile its declared categories")
        if int(self.max_sample_equivalent_terms) <= 0 or supplied > int(self.max_sample_equivalent_terms):
            raise ValueError("synthetic response work ledger exceeds its explicit cap")
        object.__setattr__(self, "sample_equivalent_terms", supplied)
        object.__setattr__(self, "max_sample_equivalent_terms", int(self.max_sample_equivalent_terms))

    @property
    def under_cap(self) -> bool:
        return bool(self.sample_equivalent_terms <= self.max_sample_equivalent_terms)

    def as_dict(self) -> dict[str, Any]:
        return {
            "frequency_sample_count_F": self.frequency_sample_count,
            "candidate_count_C": self.candidate_count,
            "candidate_kernel_cache_terms": self.candidate_kernel_sample_terms,
            "truth_generator_terms": self.truth_generator_sample_terms,
            "on_grid_score_terms": self.on_grid_score_sample_terms,
            "six_off_grid_score_terms": self.off_grid_score_sample_terms,
            "mismatch_secondary_generator_terms": self.mismatch_secondary_generator_sample_terms,
            "final_recheck_control_terms": self.final_recheck_control_terms,
            "sample_equivalent_terms": self.sample_equivalent_terms,
            "max_sample_equivalent_terms": self.max_sample_equivalent_terms,
            "under_cap": self.under_cap,
            "dot_arithmetic_terms_separate": {
                "target_energy": self.target_energy_dot_terms,
                "denominator": self.denominator_dot_terms,
                "gain_numerator": self.gain_numerator_dot_terms,
                "residual": self.residual_dot_terms,
            },
            "metadata_operations_separate": {
                "header_gate_records": self.header_gate_record_ops,
                "source_af_metadata_records": self.source_af_metadata_record_ops,
                "source_af_conversion_records": self.source_af_conversion_record_ops,
                "candidate_score_ops": self.candidate_score_ops,
                "candidate_cache_entries": self.candidate_cache_entries,
            },
            "hidden_adaptive_loops": self.hidden_adaptive_loops,
        }


def _validate_full_response_candidates(candidates: Sequence[Sequence[float]]) -> np.ndarray:
    points = np.asarray(candidates, dtype=np.float64)
    if points.shape != (SYNTHETIC_RESPONSE_CANDIDATE_COUNT, 3) or not np.isfinite(points).all():
        raise ValueError("the native-complex response evaluator requires the full 175-candidate stencil")
    tuples = {tuple(float(value) for value in row) for row in points}
    if len(tuples) != SYNTHETIC_RESPONSE_CANDIDATE_COUNT:
        raise ValueError("response candidates must be unique")
    expected = synthetic_full_response_candidate_stencil()
    if not np.array_equal(points, expected):
        raise ValueError("response evaluator candidates must be the independently declared full LTH1 stencil")
    return points


def _validate_synthetic_response_records(records: Sequence[SourceAFObservation]) -> tuple[SourceAFObservation, ...]:
    raw_records = tuple(records)
    if any(not isinstance(record, SourceAFObservation) for record in raw_records):
        raise TypeError("synthetic evaluator accepts only SourceAFObservation records; raw/duck types are rejected")
    return SyntheticSourceAFEnvelope(raw_records, scenario="evaluator_input").records


def _flatten_source_response(records: Sequence[SourceAFObservation]) -> np.ndarray:
    return np.concatenate([np.asarray(record.effective_response, dtype=np.complex128) for record in records])


def _classify_synthetic_score(
    scores: Sequence[SyntheticCandidateScore],
    *,
    target_energy: float,
) -> tuple[str, int | None, int | None, float | None]:
    if target_energy <= 0.0 or not scores:
        return INCONCLUSIVE_ZERO_RECEIVED_ENERGY, None, None, None
    best = scores[0]
    second = scores[1] if len(scores) > 1 else None
    gap = None if second is None else float((second.loss - best.loss) / max(abs(second.loss), 1.0e-15))
    candidates = np.asarray([score.candidate_xyz_m for score in scores], dtype=np.float64)
    best_candidate = np.asarray(best.candidate_xyz_m, dtype=np.float64)
    lower = candidates.min(axis=0)
    upper = candidates.max(axis=0)
    if bool(np.any(np.isclose(best_candidate, lower, rtol=0.0, atol=0.0)) or np.any(np.isclose(best_candidate, upper, rtol=0.0, atol=0.0))):
        status = INCONCLUSIVE_BOUNDARY_CANDIDATE
    elif best.loss > SYNTHETIC_RESPONSE_MAX_MODEL_MISMATCH_LOSS and (
        gap is None or gap >= SYNTHETIC_RESPONSE_MIN_DISCRIMINATION_RELATIVE_GAP
    ):
        status = INCONCLUSIVE_MODEL_MISMATCH
    elif gap is None or gap < SYNTHETIC_RESPONSE_MIN_DISCRIMINATION_RELATIVE_GAP:
        status = INCONCLUSIVE_FLAT_EVIDENCE
    else:
        status = SYNTHETIC_STENCIL_DISCRIMINATIVE_PROFILE
    return status, 0, 1 if second is not None else None, gap


def score_synthetic_native_complex_response_cases(
    response_record_sets: Sequence[Sequence[SourceAFObservation]],
    candidates: Sequence[Sequence[float]],
    model: SyntheticTophatResponseModel,
) -> tuple[tuple[SyntheticCandidateScoreReport, ...], SyntheticResponseWorkLedger]:
    """Score multiple envelopes with one bounded candidate-kernel cache.

    The evaluator sees only source-AF observations, supplied candidates, and the
    declared model.  Held-out truth, scattering points, and oracle residual maps
    are deliberately not accepted by this API.
    """

    if not isinstance(model, SyntheticTophatResponseModel):
        raise TypeError("synthetic evaluator requires SyntheticTophatResponseModel")
    points = _validate_full_response_candidates(candidates)
    record_sets = tuple(_validate_synthetic_response_records(records) for records in response_record_sets)
    if not record_sets:
        raise ValueError("synthetic evaluator requires at least one response envelope")
    first_headers = tuple(
        (
            _identity_key(record.identity),
            tuple(np.asarray(record.frequencies_hz, dtype=np.float64).tolist()),
            tuple(np.asarray(record.position_xyz_m, dtype=np.float64).tolist()),
            float(record.effective_r0_m),
        )
        for record in record_sets[0]
    )
    for records in record_sets[1:]:
        headers = tuple(
            (
                _identity_key(record.identity),
                tuple(np.asarray(record.frequencies_hz, dtype=np.float64).tolist()),
                tuple(np.asarray(record.position_xyz_m, dtype=np.float64).tolist()),
                float(record.effective_r0_m),
            )
            for record in records
        )
        if headers != first_headers:
            raise ValueError("shared response-kernel caching requires identical synthetic headers")
    flattened = tuple(_flatten_source_response(records) for records in record_sets)
    energies = tuple(float(np.vdot(values, values).real) for values in flattened)
    nonzero = tuple(index for index, energy in enumerate(energies) if energy > SYNTHETIC_RESPONSE_MIN_RECEIVED_ENERGY)
    score_lists: list[list[SyntheticCandidateScore]] = [[] for _ in record_sets]
    kernel_cache_entries = 0
    for candidate in points:
        if not nonzero:
            break
        h = np.concatenate(tuple(model.kernel(candidate, record) for record in record_sets[0]))
        kernel_cache_entries += 1
        denominator = float(np.vdot(h, h).real)
        for case_index in nonzero:
            values = flattened[case_index]
            numerator = np.vdot(h, values)
            gain_hat = complex(numerator / denominator)
            residual = values - gain_hat * h
            residual_energy = float(np.vdot(residual, residual).real)
            score_lists[case_index].append(
                SyntheticCandidateScore(
                    candidate_xyz_m=candidate,
                    gain_hat=gain_hat,
                    target_energy=energies[case_index],
                    gain_denominator=denominator,
                    gain_numerator=complex(numerator),
                    residual_energy=residual_energy,
                    loss=float(residual_energy / energies[case_index]),
                )
            )
    reports: list[SyntheticCandidateScoreReport] = []
    for case_index, scores in enumerate(score_lists):
        ordered = tuple(sorted(scores, key=lambda score: score.loss))
        status, _, _, gap = _classify_synthetic_score(ordered, target_energy=energies[case_index])
        reports.append(
            SyntheticCandidateScoreReport(
                scenario=f"evaluator_case_{case_index}",
                scores=ordered,
                status=status,
                target_energy=energies[case_index],
                best_index=0 if ordered else None,
                second_index=1 if len(ordered) > 1 else None,
                relative_gap=gap,
                kernel_cache_entries=kernel_cache_entries,
                kernel_rerender_count=0,
            )
        )
    F = int(record_sets[0][0].frequencies_hz.size)
    F = int(sum(record.frequencies_hz.size for record in record_sets[0]))
    C = int(points.shape[0])
    nonzero_count = len(nonzero)
    ledger = SyntheticResponseWorkLedger(
        frequency_sample_count=F,
        candidate_count=C,
        candidate_kernel_sample_terms=C * F if nonzero else 0,
        truth_generator_sample_terms=0,
        on_grid_score_sample_terms=0,
        off_grid_score_sample_terms=0,
        final_recheck_control_terms=0,
        target_energy_dot_terms=nonzero_count * F,
        denominator_dot_terms=C * F if nonzero else 0,
        gain_numerator_dot_terms=nonzero_count * C * F,
        residual_dot_terms=nonzero_count * C * F,
        header_gate_record_ops=len(record_sets) * len(record_sets[0]),
        candidate_score_ops=nonzero_count * C,
        candidate_cache_entries=kernel_cache_entries,
    )
    return tuple(reports), ledger


def score_synthetic_native_complex_response(
    response_records: Sequence[SourceAFObservation],
    candidates: Sequence[Sequence[float]],
    model: SyntheticTophatResponseModel,
) -> SyntheticCandidateScoreReport:
    """Evaluate one source-AF envelope without truth or response-side conversion."""

    reports, _ = score_synthetic_native_complex_response_cases((response_records,), candidates, model)
    report = reports[0]
    return SyntheticCandidateScoreReport(
        scenario="single_evaluator_input",
        scores=report.scores,
        status=report.status,
        target_energy=report.target_energy,
        best_index=report.best_index,
        second_index=report.second_index,
        relative_gap=report.relative_gap,
        kernel_cache_entries=report.kernel_cache_entries,
        kernel_rerender_count=report.kernel_rerender_count,
    )


@dataclass(frozen=True)
class SyntheticTophatCylinderRing:
    """Planted nonzero-center cylinder/ring aspect-scattering fixture."""

    geometric_center_xyz_m: np.ndarray = field(default_factory=lambda: _readonly((0.35, -0.22, 0.60)))
    cylinder_radius_m: float = 1.20
    cylinder_half_height_m: float = 0.50
    aspects_deg: np.ndarray = field(default_factory=lambda: _readonly((0.0, 90.0, 180.0, 270.0)))
    elevations_deg: np.ndarray = field(default_factory=lambda: _readonly((-8.0, -3.0, 3.0, 8.0)))
    radial_base_m: float = 0.90
    radial_aspect_amplitude_m: float = 0.14
    shared_vertical_offset_m: float = 0.18
    shared_vertical_offset_bounds_m: tuple[float, float] = (0.10, 0.26)
    radial_rule: str = "rho(aspect)=0.90+0.14*cos(aspect); bounded ring nuisance"
    vertical_rule: str = "shared unknown scattering offset in [0.10,0.26] m inside the cylinder"

    def __post_init__(self) -> None:
        center = _readonly(self.geometric_center_xyz_m)
        aspects = _readonly(self.aspects_deg)
        elevations = _readonly(self.elevations_deg)
        if center.shape != (3,) or np.linalg.norm(center) == 0:
            raise ValueError("synthetic fixture requires a finite nonzero native geometric center")
        if aspects.ndim != 1 or elevations.shape != aspects.shape or aspects.size < 4:
            raise ValueError("synthetic cylinder/ring panel needs matching azimuth/elevation arrays")
        if self.cylinder_radius_m <= 0 or self.cylinder_half_height_m <= 0:
            raise ValueError("synthetic cylinder dimensions must be positive")
        if not np.isfinite([self.radial_base_m, self.radial_aspect_amplitude_m, self.shared_vertical_offset_m]).all():
            raise ValueError("synthetic scattering parameters must be finite")
        radial_min = self.radial_base_m - abs(self.radial_aspect_amplitude_m)
        radial_max = self.radial_base_m + abs(self.radial_aspect_amplitude_m)
        if radial_min < 0 or radial_max > self.cylinder_radius_m:
            raise ValueError("synthetic ring returns must remain within the cylinder radius")
        if not (self.shared_vertical_offset_bounds_m[0] <= self.shared_vertical_offset_m <= self.shared_vertical_offset_bounds_m[1]):
            raise ValueError("planted vertical scattering offset must remain in its declared bounds")
        object.__setattr__(self, "geometric_center_xyz_m", center)
        object.__setattr__(self, "aspects_deg", aspects)
        object.__setattr__(self, "elevations_deg", elevations)

    @property
    def radial_offsets_m(self) -> np.ndarray:
        return _readonly(self.radial_base_m + self.radial_aspect_amplitude_m * np.cos(np.deg2rad(self.aspects_deg)))

    @property
    def scattering_points_xyz_m(self) -> np.ndarray:
        directions = np.stack(
            (np.cos(np.deg2rad(self.aspects_deg)), np.sin(np.deg2rad(self.aspects_deg))), axis=1
        )
        points = np.empty((self.aspects_deg.size, 3), dtype=np.float64)
        points[:, :2] = self.geometric_center_xyz_m[:2][None, :] + self.radial_offsets_m[:, None] * directions
        points[:, 2] = self.geometric_center_xyz_m[2] + self.shared_vertical_offset_m
        return _readonly(points)

    def as_dict(self) -> dict[str, Any]:
        return {
            "geometric_center_xyz_m": self.geometric_center_xyz_m.tolist(),
            "cylinder_radius_m": self.cylinder_radius_m,
            "cylinder_half_height_m": self.cylinder_half_height_m,
            "aspects_deg": self.aspects_deg.tolist(),
            "elevations_deg": self.elevations_deg.tolist(),
            "radial_offsets_m": self.radial_offsets_m.tolist(),
            "shared_vertical_offset_m": self.shared_vertical_offset_m,
            "shared_vertical_offset_bounds_m": list(self.shared_vertical_offset_bounds_m),
            "radial_rule": self.radial_rule,
            "vertical_rule": self.vertical_rule,
            "native_scattering_point_z": "center_z + shared_vertical_offset; not geometric center z",
            "absolute_height_above_ground": "not established by this fixture",
        }


def _synthetic_panel_observations(*, degenerate_elevation: bool = False) -> tuple[NativeObservation, ...]:
    records: list[NativeObservation] = []
    frequencies_by_pass = (
        (9.10e9, 9.16e9, 9.24e9),
        (9.11e9, 9.18e9, 9.29e9, 9.37e9),
        (9.12e9, 9.20e9, 9.31e9, 9.43e9, 9.55e9),
        (9.13e9, 9.22e9, 9.35e9),
        (9.14e9, 9.24e9, 9.39e9, 9.51e9),
        (9.15e9, 9.26e9, 9.42e9, 9.58e9),
        (9.16e9, 9.28e9, 9.45e9),
        (9.17e9, 9.30e9, 9.49e9, 9.62e9),
    )
    for pass_id in TOPHAT_PANEL_PASSES:
        elevation = 2.0 if degenerate_elevation else -8.0 + 2.0 * (pass_id - 1)
        for sector in TOPHAT_PANEL_SECTORS:
            theta = float(sector - 2)
            theta_rad = np.deg2rad(theta)
            radius = 100.0 + 0.5 * np.sin(np.deg2rad(pass_id * 7.0 + sector))
            position = np.asarray(
                [
                    radius * np.cos(theta_rad),
                    radius * np.sin(theta_rad),
                    15.0 if degenerate_elevation else 12.0 + 0.8 * pass_id,
                ],
                dtype=np.float64,
            )
            frequencies = np.asarray(frequencies_by_pass[pass_id - 1], dtype=np.float64)
            response = _readonly(
                (0.5 + 0.1j) * (1.0 + 0.03 * pass_id) * np.exp(1j * np.arange(frequencies.size) * 0.17),
                np.complex128,
            )
            autofocus = AutofocusProvenance(
                schema=AUTOFOCUS_PROVENANCE_SCHEMA,
                mode=AUTOFOCUS_PUBLISHED,
                official_available=True,
                applied=False,
                source_shard_id=f"pass{pass_id}_hh",
                range_field="r_correct",
                phase_field="ph_correct",
            )
            records.append(
                NativeObservation(
                    identity=NativeObservationId(pass_id, "hh", sector, 0),
                    role="train",
                    response=response,
                    frequencies_hz=_readonly(frequencies),
                    position_xyz_m=_readonly(position),
                    r0_m=float(np.linalg.norm(position)),
                    th_deg=theta,
                    phi_deg=elevation,
                    r_correct_raw=0.001 * pass_id,
                    ph_correct_raw=0.03 + 0.001 * pass_id,
                    phase_reference=PhaseReferenceContract(),
                    autofocus=autofocus,
                )
            )
    return tuple(records)


def synthetic_multipass_panel_observations(*, degenerate_elevation: bool = False) -> tuple[NativeObservation, ...]:
    """Create bounded in-memory HH/TRAIN records with ragged native frequencies."""

    return _synthetic_panel_observations(degenerate_elevation=degenerate_elevation)


def synthetic_planted_recovery_case() -> dict[str, Any]:
    """Return distinct fixed-rule recovery and bounded-tube ambiguity mechanics."""

    fixture = SyntheticTophatCylinderRing()
    study = synthetic_aspect_return_study(
        fixture.geometric_center_xyz_m[:2],
        fixture.aspects_deg,
        fixture.radial_offsets_m,
        radial_bounds_m=(0.70, 1.10),
        grid_halfwidth_m=0.50,
        grid_size=21,
    )
    scattering_z = fixture.scattering_points_xyz_m[:, 2]
    fixed_strata = synthetic_nuisance_strata(fixture, shared_zeta_m=fixture.shared_vertical_offset_m)
    fixed_recovered = synthetic_full_fixed_rule_recovery(
        fixed_strata, known_zeta_m=fixture.shared_vertical_offset_m
    )
    bounded_strata = synthetic_nuisance_strata(fixture, shared_zeta_m=fixture.shared_vertical_offset_m)
    bounded_rule = SyntheticAspectNuisanceRule(mode="bounded_shared_zeta")
    bounded_set = synthetic_robust_feasible_set(
        bounded_strata,
        bounded_rule,
        synthetic_candidate_stencil(fixture.geometric_center_xyz_m, halfwidth_m=0.4, grid_size=5),
    )
    return {
        "fixture": fixture,
        "aspect_return_study": study,
        "fixed_rule_strata": fixed_strata,
        "fixed_rule_recovered_geometric_center_xyz_m": fixed_recovered,
        "fixed_rule_mechanics_recovery": True,
        "bounded_tube_strata": bounded_strata,
        "bounded_tube_feasible_set": bounded_set,
        "bounded_tube_single_supplied_stencil_candidate": bounded_set.single_supplied_stencil_candidate,
        "scattering_point_native_z_m": float(np.mean(scattering_z)),
        "geometric_center_native_z_m": float(fixture.geometric_center_xyz_m[2]),
        "shared_zeta_joint_geometric_center_z_interval_m": bounded_set.z_interval_m,
        "scalar_scattering_offset_envelope_m": "not used as the joint result",
        "absolute_height_status": BLOCKED_MISSING_NATIVE_Z_DATUM,
        "mechanics_only": True,
        "status": "SYNTHETIC_CONDITIONAL_MECHANICS_ONLY",
    }


def apply_two_cube_measurement_policy(values: Iterable[Any], *, policy: TwoCubeMeasurementExteriorPolicy = DEFAULT_TWO_CUBE_MEASUREMENT_POLICY) -> tuple[np.ndarray, ...]:
    """Apply the declared identity transform without magnitude/crop/subtraction."""

    policy.validate()
    result = []
    for value in values:
        array = np.asarray(value)
        if not np.iscomplexobj(array):
            raise ValueError("two-cube readiness policy requires native complex values")
        result.append(_readonly(array, np.complex128))
    return tuple(result)


def apply_two_cube_measurement_adjoint(values: Iterable[Any], *, policy: TwoCubeMeasurementExteriorPolicy = DEFAULT_TWO_CUBE_MEASUREMENT_POLICY) -> tuple[np.ndarray, ...]:
    """Adjoint of the frozen identity transform; future spotlights need their own adjoint."""

    return apply_two_cube_measurement_policy(values, policy=policy)


def require_native_tophat_placement() -> None:
    """Keep cube extraction/fit blocked until a reviewed numeric native R/t exists."""

    raise TophatLocalizationBlocked(
        BLOCKED_NATIVE_CUBE_PLACEMENT_UNRESOLVED,
        "TopHat native cube extraction/fit remains blocked: numeric native R/t is unresolved",
    )


def validate_tophat_localization_protocol_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("TopHat localization protocol payload must be a mapping")
    if payload.get("schema") != TOPHAT_LOCALIZATION_SCHEMA:
        raise ValueError("unexpected TopHat localization readiness protocol schema")
    if payload.get("data_free") is not True or payload.get("measured_fit_release") is not False:
        raise ValueError("TopHat localization protocol must remain data-free and unreleased")
    if tuple(payload.get("targets", ())) != ("tophat", "toyota_camry"):
        raise ValueError("TopHat localization protocol must retain separate TopHat/Camry targets")
    seed = payload.get("tophat_seed", {})
    if seed.get("label") != TOPHAT_DIAGRAM_LABEL or tuple(seed.get("xy_m", ())) != TOPHAT_DIAGRAM_XY_M:
        raise ValueError("protocol must retain the qualified LTH1 XY seed")
    if seed.get("native_placement") != "missing_numeric_native_R_t" or seed.get("z_status") != "unresolved_no_native_height_datum":
        raise ValueError("protocol cannot promote the diagram seed into native placement or Z")
    panel = payload.get("future_geometry_panel", {})
    if tuple(panel.get("pass_ids", ())) != TOPHAT_PANEL_PASSES or tuple(panel.get("sector_ids", ())) != TOPHAT_PANEL_SECTORS:
        raise ValueError("protocol must retain the exact prospective pass/sector panel")
    if panel.get("polarization") != TOPHAT_PANEL_POLARIZATION or panel.get("role") != "train":
        raise ValueError("protocol must retain the exact HH/TRAIN panel")
    z_search = payload.get("native_z_search", {})
    if tuple(z_search.get("provisional_interval_m", ())) != PROVISIONAL_NATIVE_Z_SEARCH_INTERVAL_M:
        raise ValueError("protocol must retain the provisional [-2,+3] m native-Z breadth assumption")
    if z_search.get("status") != PROVISIONAL_NATIVE_Z_SEARCH_INTERVAL_STATUS or z_search.get("actual_measured_binding") is not None:
        raise ValueError("provisional native-Z breadth must not be presented as measured binding")
    if "discrete retained grid samples" not in str(z_search.get("near_minimum_samples", "")):
        raise ValueError("protocol must disclose discrete near-minimum native-Z samples")
    if "any retained near-minimum sample" not in str(z_search.get("boundary_rule", "")):
        raise ValueError("protocol must disclose endpoint boundary inconclusive behavior")
    nuisance = payload.get("synthetic_nuisance_tube", {})
    if nuisance.get("rho_bounds_m") != list(SYNTHETIC_RADIAL_BOUNDS_M) or nuisance.get("alpha_bounds_deg") != list(SYNTHETIC_ALPHA_BOUNDS_DEG):
        raise ValueError("protocol synthetic nuisance bands drifted")
    if nuisance.get("zeta_modes", {}).get("shared_zeta_bounds_m") != list(SYNTHETIC_SHARED_ZETA_BOUNDS_M):
        raise ValueError("protocol shared-zeta band drifted")
    if "not established measured TopHat physics" not in str(nuisance.get("synthetic_side_convention", "")):
        raise ValueError("protocol must disclose the synthetic far-side convention")
    source_af = payload.get("source_af", {})
    if source_af.get("supplied_conditioning") != "fixed input; this readiness package cannot independently validate the autofocus representation":
        raise ValueError("protocol must state that source-AF is fixed supplied conditioning")
    required_forbidden = {
        "free per-record range corrections",
        "arbitrary frequency-dependent gains",
        "unconstrained 3-D scatterer offsets",
        "cross-channel autofocus borrowing",
    }
    if not required_forbidden.issubset(set(source_af.get("forbidden", ()) )):
        raise ValueError("protocol must forbid free corrections, gains, 3-D offsets, and AF borrowing")
    measurement = payload.get("measurement_policy", {})
    if measurement.get("representation") != "native_complex" or measurement.get("transform") != "identity":
        raise ValueError("TopHat readiness requires native-complex identity measurement policy")
    if any(measurement.get(name) is not False for name in ("crop", "subtraction", "padding", "magnitude_conversion")):
        raise ValueError("TopHat readiness forbids crop/subtraction/padding/magnitude conversion")
    if measurement.get("exterior_scene_retained") is not True or measurement.get("cube_isolation") is not False:
        raise ValueError("TopHat readiness must retain exterior scene returns")
    resource = payload.get("resource_bounds", {})
    if resource.get("dense_40x40x40_search") != "forbidden and rejected before kernels":
        raise ValueError("TopHat readiness must reject dense 40^3 search before kernels")
    return payload


def load_tophat_localization_protocol(path: str | Any) -> Mapping[str, Any]:
    import json
    from pathlib import Path

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_tophat_localization_protocol_payload(payload)


__all__ = [
    "ABSOLUTE_HEIGHT_STATUS_MISSING_DATUM",
    "ABSOLUTE_HEIGHT_STATUS_SOURCE_DATUM_ONLY",
    "AspectElevationGeometryReport",
    "BLOCKED_MISSING_DECLARED_NATIVE_Z_SEARCH_INTERVAL",
    "BLOCKED_MISSING_NATIVE_Z_DATUM",
    "BLOCKED_NATIVE_CUBE_PLACEMENT_UNRESOLVED",
    "BLOCKED_RESOURCE_CAP_EXCEEDED",
    "CONDITIONAL_NATIVE_Z_UNREGISTERED",
    "CONDITIONAL_XY_FEASIBLE_SET_UNREGISTERED",
    "DEFAULT_MIN_ANTENNA_VERTICAL_SPAN_M",
    "DEFAULT_TOPHAT_DIAGRAM_XY_SEED",
    "DEFAULT_TOPHAT_MULTIPASS_PANEL",
    "DEFAULT_TWO_CUBE_MEASUREMENT_POLICY",
    "INCONCLUSIVE_ASPECT_DEPENDENT_EVIDENCE",
    "INCONCLUSIVE_BOUNDARY_CANDIDATE",
    "INCONCLUSIVE_FLAT_EVIDENCE",
    "INCONCLUSIVE_INSUFFICIENT_ELEVATION_GEOMETRY",
    "INCONCLUSIVE_MEASURED_SCREEN_ASPECT_SUPPORT",
    "INCONCLUSIVE_MEASURED_SCREEN_BOUNDARY_OR_COVERAGE",
    "INCONCLUSIVE_MEASURED_SCREEN_HEADER_INPUT",
    "INCONCLUSIVE_MEASURED_SCREEN_NO_COHERENT_SUPPORT",
    "INCONCLUSIVE_MEASURED_SCREEN_OPERATOR_LIMITATION",
    "INCONCLUSIVE_MEASURED_SCREEN_SCENE_OR_ALIAS_AMBIGUITY",
    "INCONCLUSIVE_MODEL_MISMATCH",
    "INCONCLUSIVE_NO_RESPONSE_LOCALIZATION",
    "INCONCLUSIVE_ZERO_RECEIVED_ENERGY",
    "MAX_CONDITIONAL_POINT_COUNT",
    "MEASURED_RESPONSE_SCREEN_ARCHIVE_SOURCE",
    "MEASURED_RESPONSE_SCREEN_CANDIDATE_COUNT",
    "MEASURED_RESPONSE_SCREEN_COUNT_MATRIX",
    "MEASURED_RESPONSE_SCREEN_FINE_CANDIDATE_COUNT",
    "MEASURED_RESPONSE_SCREEN_STAGE_A_MAP_TERMS",
    "MEASURED_RESPONSE_SCREEN_STAGE_B_FINE_MAP_TERMS",
    "MEASURED_RESPONSE_SCREEN_UNION_FORWARD_PSF_TERMS",
    "MEASURED_RESPONSE_SCREEN_OFFGRID_REFERENCE_TERMS",
    "MEASURED_RESPONSE_SCREEN_FREQUENCY_COUNTS_BY_PASS",
    "MEASURED_RESPONSE_SCREEN_SCHEMA",
    "MEASURED_RESPONSE_SCREEN_SCENE",
    "MEASURED_RESPONSE_SCREEN_STATUS_PENDING_GEOMETRY_RULE",
    "MEASURED_RESPONSE_SCREEN_TOTAL_FREQUENCY_SAMPLES",
    "MEASURED_RESPONSE_SCREEN_TOTAL_RECORD_COUNT",
    "MEASURED_RESPONSE_SCREEN_WORK_CAP",
    "MultipassHHTrainSourceAFScope",
    "PROVISIONAL_NATIVE_Z_SEARCH_INTERVAL_M",
    "PROVISIONAL_NATIVE_Z_SEARCH_INTERVAL_STATUS",
    "ResourcePreflight",
    "SYNTHETIC_RESPONSE_CANDIDATE_COUNT",
    "SYNTHETIC_RESPONSE_FREQUENCY_COUNTS_BY_PASS",
    "SYNTHETIC_RESPONSE_FREQUENCY_ENDPOINTS_BY_PASS_HZ",
    "SYNTHETIC_RESPONSE_MAX_MODEL_MISMATCH_LOSS",
    "SYNTHETIC_RESPONSE_MIN_DISCRIMINATION_RELATIVE_GAP",
    "SYNTHETIC_RESPONSE_MIN_RECEIVED_ENERGY",
    "SYNTHETIC_RESPONSE_SHARD_COUNTS_BY_PASS_SECTOR",
    "SYNTHETIC_RESPONSE_STENCIL_OFFSETS_XY_M",
    "SYNTHETIC_RESPONSE_STENCIL_Z_M",
    "SYNTHETIC_RESPONSE_WORK_CAP",
    "SYNTHETIC_STENCIL_DISCRIMINATIVE_PROFILE",
    "SyntheticAspectReturnStudy",
    "SyntheticAspectNuisanceRule",
    "SyntheticNuisanceStratum",
    "SyntheticRobustFeasibleSet",
    "SyntheticCandidateScore",
    "SyntheticCandidateScoreReport",
    "SyntheticHeldOutResponseCase",
    "SyntheticResponseWorkLedger",
    "SyntheticSourceAFEnvelope",
    "SyntheticTophatResponseModel",
    "SyntheticTophatCylinderRing",
    "TophatDiagramXYSeed",
    "TophatLocalizationBlocked",
    "TophatMeasuredResponseScreenDataset",
    "TophatMeasuredResponseScreenGrid",
    "TophatMeasuredResponseScreenInputError",
    "TophatMeasuredResponseScreenMapSet",
    "TophatMeasuredResponseScreenPanel",
    "TophatMeasuredResponseScreenSelection",
    "TophatMeasuredResponseScreenWorkLedger",
    "TophatMeasuredResponseStageAResult",
    "TophatMeasuredResponseStageBResult",
    "TophatMeasuredResponseSupportComponent",
    "TophatMeasuredResponseCubeAssessment",
    "TophatMeasuredResponsePSFResult",
    "TophatMeasuredResponseOffgridSensitivityResult",
    "TophatMeasuredResponseRawSourceBridgeResult",
    "TophatMultipassPanel",
    "TophatNativeHeightDatum",
    "TophatNativeZSearchInterval",
    "TophatTrainPanelSelection",
    "TwoCubeMeasurementExteriorPolicy",
    "apply_two_cube_measurement_adjoint",
    "apply_two_cube_measurement_policy",
    "assess_aspect_elevation_geometry",
    "conditional_native_z_search",
    "convert_tophat_train_source_af",
    "convert_tophat_measured_response_screen_source_af",
    "build_tophat_measured_response_screen_grid",
    "build_tophat_measured_response_screen_panels",
    "build_tophat_measured_response_fine_grid",
    "choose_tophat_measured_screen_reference",
    "joint_shared_zeta_feasible_set",
    "load_tophat_localization_protocol",
    "preflight_resource_budget",
    "preflight_tophat_measured_response_screen_headers",
    "prepare_tophat_measured_response_screen_dataset",
    "evaluate_tophat_measured_response_stage_a",
    "provisional_tophat_native_z_search_interval",
    "require_native_tophat_placement",
    "select_tophat_train_ids",
    "select_tophat_measured_response_screen_ids",
    "shared_vertical_offset_feasible_center_z",
    "synthetic_aspect_return_study",
    "synthetic_multipass_panel_observations",
    "synthetic_nuisance_strata",
    "synthetic_candidate_stencil",
    "synthetic_declared_response_model",
    "synthetic_full_response_candidate_stencil",
    "synthetic_full_fixed_rule_recovery",
    "synthetic_native_complex_response_case",
    "synthetic_robust_feasible_set",
    "synthetic_planted_recovery_case",
    "score_synthetic_native_complex_response",
    "score_synthetic_native_complex_response_cases",
    "measured_tophat_half_power_components",
    "run_tophat_measured_response_stage_b",
    "stream_tophat_measured_response_offgrid_sensitivity",
    "stream_tophat_measured_response_psfs",
    "stream_tophat_measured_response_raw_source_bridge",
    "stream_tophat_measured_response_screen_maps",
    "validate_tophat_train_headers",
    "validate_tophat_localization_protocol_payload",
]
