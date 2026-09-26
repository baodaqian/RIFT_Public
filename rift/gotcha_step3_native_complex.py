"""Bounded Step-3 native-complex readiness mechanics for the two GOTCHA cubes.

This module is deliberately data-free.  It keeps the exact local reporting grids
separate from the tiny explicit supports used by numerical smoke tests, applies a
declared local-to-native rigid placement, and preserves complex native-frequency
measurements through a shared per-observation transform.

The raw path wraps :class:`NativeRaggedOperator` without changing its kernel.  The
source-AF path is a separate, explicitly named Torch bridge over immutable
``SourceAFObservation`` records.  Source-AF records are never duck-typed as raw
``NativeObservation`` records.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    import torch as _torch
    TORCH_AVAILABLE = True
    _TORCH_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover - exercised only on torch-less hosts.
    _torch = None
    TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = ImportError(
        "Step-3 native-complex readiness requires Torch for the differentiable "
        "float64/complex128 source-AF bridge"
    )
    _TORCH_IMPORT_ERROR.__cause__ = exc

try:  # Package imports are used by normal callers.
    from .gotcha_acquisition import (
        AutofocusProvenance,
        NativeObservation,
        NativeObservationId,
        PhaseReferenceContract,
        AUTOFOCUS_PROVENANCE_SCHEMA,
        AUTOFOCUS_OFFICIAL_ABSENT,
        AUTOFOCUS_PUBLISHED,
        PUBLISHED_GOTCHA_REFERENCE_CANDIDATE,
    )
    from .gotcha_source_af import (
        SOURCE_AF_FORMULA,
        SOURCE_REPRESENTATION,
        SourceAFObservation,
        SourceAFScope,
        _kernel as _source_af_kernel_oracle,
    )
    from .gotcha_step2_controls import (
        NativePredictions,
        NativeRaggedOperator,
        PHASE_HYPOTHESIS_ADJOINT,
        PHASE_HYPOTHESIS_FORWARD,
        PHASE_HYPOTHESIS_NAME,
        SPEED_OF_LIGHT_M_S,
    )
except ImportError:  # Direct file loading used by the local validation scripts.
    try:
        from gotcha_acquisition import (
            AutofocusProvenance,
            NativeObservation,
            NativeObservationId,
            PhaseReferenceContract,
            AUTOFOCUS_PROVENANCE_SCHEMA,
            AUTOFOCUS_OFFICIAL_ABSENT,
            AUTOFOCUS_PUBLISHED,
            PUBLISHED_GOTCHA_REFERENCE_CANDIDATE,
        )
        from gotcha_source_af import (
            SOURCE_AF_FORMULA,
            SOURCE_REPRESENTATION,
            SourceAFObservation,
            SourceAFScope,
            _kernel as _source_af_kernel_oracle,
        )
        from gotcha_step2_controls import (
            NativePredictions,
            NativeRaggedOperator,
            PHASE_HYPOTHESIS_ADJOINT,
            PHASE_HYPOTHESIS_FORWARD,
            PHASE_HYPOTHESIS_NAME,
            SPEED_OF_LIGHT_M_S,
        )
    except ImportError:
        from rift.gotcha_acquisition import (
        AutofocusProvenance,
        NativeObservation,
        NativeObservationId,
        PhaseReferenceContract,
        AUTOFOCUS_PROVENANCE_SCHEMA,
        AUTOFOCUS_OFFICIAL_ABSENT,
        AUTOFOCUS_PUBLISHED,
        PUBLISHED_GOTCHA_REFERENCE_CANDIDATE,
    )
        from rift.gotcha_source_af import (
        SOURCE_AF_FORMULA,
        SOURCE_REPRESENTATION,
        SourceAFObservation,
        SourceAFScope,
        _kernel as _source_af_kernel_oracle,
    )
        from rift.gotcha_step2_controls import (
        NativePredictions,
        NativeRaggedOperator,
        PHASE_HYPOTHESIS_ADJOINT,
        PHASE_HYPOTHESIS_FORWARD,
        PHASE_HYPOTHESIS_NAME,
        SPEED_OF_LIGHT_M_S,
    )


def _require_torch() -> Any:
    if not TORCH_AVAILABLE:
        raise ImportError(str(_TORCH_IMPORT_ERROR)) from _TORCH_IMPORT_ERROR
    return _torch


NATIVE_COMPLEX_PROTOCOL_SCHEMA = "rift_gotcha_step3_two_target_native_complex_protocol_v2"
NATIVE_COMPLEX_READINESS_SCHEMA = "rift_gotcha_step3_native_complex_readiness_v2"
MAX_SMOKE_SUPPORT_POINTS = 5
MIN_SMOKE_SUPPORT_POINTS = 3
MAX_SMOKE_KERNEL_EVALUATIONS = 250_000
GLOBAL_COMPLEX_GAIN = 1.0 + 0.0j
CAMRY_FOOTPRINT_ABSOLUTE_TOLERANCE_M = 0.10
CAMRY_FOOTPRINT_NATIVE_CORNERS_M = np.asarray(
    [
        [21.42, -21.14, 0.03],  # LF
        [21.65, -16.40, 0.03],  # LR
        [19.92, -16.31, 0.01],  # RR
        [19.63, -21.11, 0.02],  # RF
    ],
    dtype=np.float64,
)
CAMRY_NOMINAL_LOCAL_CORNERS_M = np.asarray(
    [
        [2.375, 0.87, 0.0],   # LF
        [-2.375, 0.87, 0.0],  # LR
        [-2.375, -0.87, 0.0], # RR
        [2.375, -0.87, 0.0],  # RF
    ],
    dtype=np.float64,
)
GRADIENT_RELATIVE_TOLERANCE = 5.0e-6
DOT_RELATIVE_TOLERANCE = 5.0e-12

TARGET_IDS = ("tophat", "toyota_camry")
_EXPECTED_GRID = {
    "tophat": {
        "lower_edge_m": (-2.0, -2.0, -2.0),
        "upper_edge_exclusive_m": (2.0, 2.0, 2.0),
        "spacing_m": (0.1, 0.1, 0.1),
        "shape": (40, 40, 40),
        "final_sample_m": (1.9, 1.9, 1.9),
    },
    "toyota_camry": {
        "lower_edge_m": (-5.0, -5.0, -5.0),
        "upper_edge_exclusive_m": (5.0, 5.0, 5.0),
        "spacing_m": (0.1, 0.1, 0.1),
        "shape": (100, 100, 100),
        "final_sample_m": (4.9, 4.9, 4.9),
    },
}

CAMRY_VEHICLE_AXIS_ROTATION = np.asarray(
    [
        [-0.05442654550121675, 0.9985177770800097, 0.0],
        [-0.9985177770800097, -0.05442654550121675, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
CAMRY_TRANSLATION_M = np.asarray((20.66, -18.71, 0.02), dtype=np.float64)


def _finite(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values")
    return array


def _readonly(value: Any, dtype: Any) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _identity_key(identity: Any) -> tuple[int, str, int, int]:
    try:
        return (
            int(identity.pass_id),
            str(identity.polarization).lower(),
            int(identity.sector_id),
            int(identity.pulse_index),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError("canonical observation identity must expose pass/polarization/sector/pulse") from exc


def _validate_ids(ids: Sequence[Any]) -> tuple[Any, ...]:
    result = tuple(ids)
    keys = tuple(_identity_key(identity) for identity in result)
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate canonical observation IDs are not allowed")
    return result


@dataclass(frozen=True)
class ReferenceGrid:
    """Exact local readout grid declaration; it does not materialize a cube."""

    target_id: str
    lower_edge_m: tuple[float, float, float]
    upper_edge_exclusive_m: tuple[float, float, float]
    spacing_m: tuple[float, float, float]
    shape: tuple[int, int, int]
    final_sample_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.target_id not in TARGET_IDS:
            raise ValueError(f"unknown target_id: {self.target_id}")
        for label, value in (
            ("lower_edge_m", self.lower_edge_m),
            ("upper_edge_exclusive_m", self.upper_edge_exclusive_m),
            ("spacing_m", self.spacing_m),
            ("final_sample_m", self.final_sample_m),
        ):
            if len(value) != 3 or not np.isfinite(np.asarray(value, dtype=np.float64)).all():
                raise ValueError(f"{label} must be a finite length-3 vector")
        if len(self.shape) != 3 or any(int(n) <= 0 for n in self.shape):
            raise ValueError("shape must be a positive length-3 tuple")
        expected_last = tuple(
            float(self.lower_edge_m[index] + self.spacing_m[index] * (int(self.shape[index]) - 1))
            for index in range(3)
        )
        if not np.allclose(
            np.asarray(expected_last),
            np.asarray(self.final_sample_m, dtype=np.float64),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("final_sample_m must be lower_edge + spacing*(shape-1), without a half-voxel shift")
        if any(float(self.spacing_m[i]) <= 0 for i in range(3)):
            raise ValueError("spacing_m must be positive")
        if any(float(self.upper_edge_exclusive_m[i]) <= float(self.lower_edge_m[i]) for i in range(3)):
            raise ValueError("upper_edge_exclusive_m must exceed lower_edge_m")

    @property
    def point_count(self) -> int:
        return int(np.prod(np.asarray(self.shape, dtype=np.int64)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "frame": "local_target_frame",
            "lower_edge_m": list(self.lower_edge_m),
            "upper_edge_exclusive_m": list(self.upper_edge_exclusive_m),
            "spacing_m": list(self.spacing_m),
            "shape": list(self.shape),
            "final_sample_m": list(self.final_sample_m),
            "endpoint_inclusion": False,
            "half_voxel_shift": False,
            "merged_box": False,
            "reporting_readout_only": True,
        }


def exact_reference_grids() -> Mapping[str, ReferenceGrid]:
    return {
        target_id: ReferenceGrid(target_id=target_id, **values)
        for target_id, values in _EXPECTED_GRID.items()
    }


@dataclass(frozen=True)
class RigidPlacement:
    """A validated ``p_native = R @ p_local + t`` placement."""

    rotation: np.ndarray
    translation_m: np.ndarray
    name: str = "parameter_injected_synthetic"
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        rotation = _readonly(self.rotation, np.float64)
        translation = _readonly(self.translation_m, np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("native placement R must have shape [3,3]")
        if translation.shape != (3,):
            raise ValueError("native placement t_m must have shape [3]")
        _finite(rotation, "native placement R")
        _finite(translation, "native placement t_m")
        orthogonality_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
        determinant = float(np.linalg.det(rotation))
        if orthogonality_error > 2.0e-12 or abs(determinant - 1.0) > 2.0e-12:
            raise ValueError(
                "native placement R must be an orthonormal proper rotation "
                f"(orthogonality_error={orthogonality_error:.3e}, determinant={determinant:.16g})"
            )
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation_m", translation)
        object.__setattr__(self, "assumptions", tuple(str(value) for value in self.assumptions))

    def apply(self, points_local_m: Any) -> np.ndarray:
        points = np.asarray(points_local_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError("local support points must have shape [point,3]")
        _finite(points, "local support points")
        transformed = points @ self.rotation.T + self.translation_m[None, :]
        return _readonly(transformed, np.float64)

    def as_dict(self) -> dict[str, Any]:
        return {
            "equation": "p_native = R @ p_local + t",
            "R": self.rotation.tolist(),
            "t_m": self.translation_m.tolist(),
            "name": self.name,
            "assumptions": list(self.assumptions),
        }


CAMRY_DECLARED_PLACEMENT = RigidPlacement(
    rotation=CAMRY_VEHICLE_AXIS_ROTATION,
    translation_m=CAMRY_TRANSLATION_M,
    name="camry_point_c_declared_working_native_frame_convention",
    assumptions=(
        "vehicle-forward axis is inferred from the rear-to-front footprint midpoint axis",
        "workbook heading 182.80 degrees is provenance/compass-consistent context, not a direct native-CCW angle",
        "local +x is vehicle heading",
        "local +y is the right-handed xy-frame lateral axis",
        "local z maps native +z",
        "Point C is the local origin",
        "footprint-derived vehicle-axis inference; not independently physically registered",
    ),
)


def camry_footprint_consistency() -> dict[str, Any]:
    """Check the declared nominal rectangle against all four workbook corners."""

    predicted = CAMRY_DECLARED_PLACEMENT.apply(CAMRY_NOMINAL_LOCAL_CORNERS_M)
    errors_m = np.linalg.norm(predicted - CAMRY_FOOTPRINT_NATIVE_CORNERS_M, axis=1)
    inverse_local = (
        CAMRY_FOOTPRINT_NATIVE_CORNERS_M - CAMRY_DECLARED_PLACEMENT.translation_m[None, :]
    ) @ CAMRY_DECLARED_PLACEMENT.rotation
    contained = bool(np.all(inverse_local >= -5.0) and np.all(inverse_local < 5.0))
    return {
        "corner_labels": ("LF", "LR", "RR", "RF"),
        "absolute_errors_m": tuple(float(value) for value in errors_m),
        "max_absolute_error_m": float(np.max(errors_m)),
        "absolute_tolerance_m": CAMRY_FOOTPRINT_ABSOLUTE_TOLERANCE_M,
        "all_corners_within_absolute_tolerance": bool(np.all(errors_m <= CAMRY_FOOTPRINT_ABSOLUTE_TOLERANCE_M)),
        "inverse_local_corners_m": inverse_local,
        "inverse_footprint_contained_in_camry_grid": contained,
    }


class MissingTophatNativeTransformError(RuntimeError):
    """Raised when a real TopHat native extraction is attempted without R/t."""


def placement_for_target(target_id: str, placement: RigidPlacement | None = None) -> RigidPlacement:
    if target_id not in TARGET_IDS:
        raise ValueError(f"unknown target_id: {target_id}")
    if target_id == "tophat":
        if placement is None:
            raise MissingTophatNativeTransformError(
                "Tophat actual extraction/fit is fail-closed: missing numeric native R/t; "
                "pass an explicit parameter-injected synthetic RigidPlacement for readiness only"
            )
        return placement
    return CAMRY_DECLARED_PLACEMENT if placement is None else placement


def transform_smoke_support(
    target_id: str,
    points_local_m: Any,
    *,
    placement: RigidPlacement | None = None,
) -> np.ndarray:
    points = np.asarray(points_local_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("smoke support must have shape [point,3]")
    if not MIN_SMOKE_SUPPORT_POINTS <= points.shape[0] <= MAX_SMOKE_SUPPORT_POINTS:
        raise ValueError(
            "degree-zero smoke support must contain 3-5 explicit points; "
            "the full reference-grid point count is forbidden as smoke quadrature"
        )
    grid = exact_reference_grids()[target_id]
    if points.shape[0] in (grid.point_count,):
        raise ValueError("reference-grid points cannot be used as smoke quadrature/support")
    _finite(points, "local support points")
    lower = np.asarray(grid.lower_edge_m, dtype=np.float64)
    upper = np.asarray(grid.upper_edge_exclusive_m, dtype=np.float64)
    if bool(np.any(points < lower[None, :]) or np.any(points >= upper[None, :])):
        raise ValueError(
            f"local smoke support must lie inside the {target_id} exact half-open cube "
            "[lower_edge, upper_edge_exclusive)"
        )
    return placement_for_target(target_id, placement).apply(points)


@dataclass(frozen=True)
class ComplexFrequencyTransform:
    """Canonical ragged per-observation complex multiplier with Hermitian adjoint."""

    observation_ids: tuple[Any, ...]
    multipliers: tuple[np.ndarray, ...]
    frequency_counts: tuple[int, ...]
    name: str = "identity_native_complex_measurement_transform"

    def __post_init__(self) -> None:
        ids = _validate_ids(self.observation_ids)
        counts = tuple(int(value) for value in self.frequency_counts)
        if len(ids) != len(counts) or len(ids) != len(self.multipliers):
            raise ValueError("multiplier IDs, frequency counts, and values must have matching lengths")
        frozen: list[np.ndarray] = []
        for index, (multiplier, count) in enumerate(zip(self.multipliers, counts)):
            array = np.asarray(multiplier, dtype=np.complex128)
            if array.shape != (count,):
                raise ValueError(
                    f"multiplier shape mismatch at observation {index}: expected ({count},), got {array.shape}"
                )
            _finite(array.real, "multiplier.real")
            _finite(array.imag, "multiplier.imag")
            frozen.append(_readonly(array, np.complex128))
        object.__setattr__(self, "observation_ids", ids)
        object.__setattr__(self, "frequency_counts", counts)
        object.__setattr__(self, "multipliers", tuple(frozen))

    @classmethod
    def identity(cls, observation_ids: Sequence[Any], frequency_counts: Sequence[int]) -> "ComplexFrequencyTransform":
        return cls(
            tuple(observation_ids),
            tuple(np.ones(int(count), dtype=np.complex128) for count in frequency_counts),
            tuple(int(count) for count in frequency_counts),
        )

    @classmethod
    def from_operator(
        cls,
        operator: Any,
        multipliers: Sequence[Any] | np.ndarray | None = None,
        *,
        name: str = "synthetic_complex_unit_modulus_measurement_transform",
    ) -> "ComplexFrequencyTransform":
        ids = tuple(operator.observation_ids)
        counts = tuple(int(value) for value in operator.frequency_counts)
        if multipliers is None:
            return cls.identity(ids, counts)
        if isinstance(multipliers, np.ndarray):
            array = np.asarray(multipliers)
            if array.ndim == 1 and len(counts) == 1:
                values = (array,)
            elif array.ndim == 2 and len(set(counts)) == 1 and array.shape == (len(counts), counts[0]):
                values = tuple(array[index] for index in range(array.shape[0]))
            else:
                raise ValueError("multiplier shape mismatch for ragged native observations")
        else:
            values = tuple(multipliers)
        return cls(ids, values, counts, name=name)

    def _check_predictions(self, predictions: NativePredictions) -> None:
        if not isinstance(predictions, NativePredictions):
            raise TypeError("complex measurement transforms require NativePredictions")
        if tuple(predictions.ids) != self.observation_ids:
            raise ValueError("prediction identities do not match canonical multiplier identities")
        for value, count in zip(predictions.values, self.frequency_counts):
            if np.asarray(value).shape != (count,):
                raise ValueError("prediction shape does not match canonical multiplier shape")
            _finite(np.asarray(value).real, "prediction.real")
            _finite(np.asarray(value).imag, "prediction.imag")

    def apply(self, predictions: NativePredictions) -> NativePredictions:
        self._check_predictions(predictions)
        return NativePredictions(self.observation_ids, self.apply_values(predictions.values))

    def apply_values(self, values: Sequence[Any], *, ids: Sequence[Any] | None = None) -> tuple[np.ndarray, ...]:
        """Apply T to a generic ragged complex payload after an optional ID check."""

        if ids is not None and tuple(ids) != self.observation_ids:
            raise ValueError("payload identities do not match canonical multiplier identities")
        if len(values) != len(self.multipliers):
            raise ValueError("payload count does not match canonical multiplier identities")
        result = []
        for value, multiplier, count in zip(values, self.multipliers, self.frequency_counts):
            array = np.asarray(value, dtype=np.complex128)
            if array.shape != (count,):
                raise ValueError("payload shape does not match canonical multiplier shape")
            _finite(array.real, "payload.real")
            _finite(array.imag, "payload.imag")
            result.append(_readonly(array * multiplier, np.complex128))
        return tuple(result)

    def adjoint_values(self, residuals: NativePredictions) -> NativePredictions:
        self._check_predictions(residuals)
        return NativePredictions(
            self.observation_ids,
            tuple(value * np.conjugate(multiplier) for value, multiplier in zip(residuals.values, self.multipliers)),
        )

    def apply_torch(self, values: Sequence[Any], *, device: Any = None) -> tuple[Any, ...]:
        _require_torch()
        import torch

        if len(values) != len(self.multipliers):
            raise ValueError("Torch ragged measurement values do not match multiplier IDs")
        result = []
        for value, multiplier, count in zip(values, self.multipliers, self.frequency_counts):
            if tuple(value.shape) != (count,):
                raise ValueError("Torch value shape does not match canonical multiplier shape")
            factor = torch.tensor(multiplier, dtype=torch.complex128, device=device)
            result.append(value * factor)
        return tuple(result)

    def adjoint_torch(self, residuals: Sequence[Any], *, device: Any = None) -> tuple[Any, ...]:
        _require_torch()
        import torch

        if len(residuals) != len(self.multipliers):
            raise ValueError("Torch ragged residuals do not match multiplier IDs")
        result = []
        for residual, multiplier, count in zip(residuals, self.multipliers, self.frequency_counts):
            if tuple(residual.shape) != (count,):
                raise ValueError("Torch residual shape does not match canonical multiplier shape")
            factor = torch.tensor(np.conjugate(multiplier), dtype=torch.complex128, device=device)
            result.append(residual * factor)
        return tuple(result)


class NativeComplexMeasurementOperator:
    """Native raw operator plus one shared complex transform for data/predictions."""

    def __init__(
        self,
        operator_or_observations: NativeRaggedOperator | Sequence[Any],
        *,
        multipliers: Sequence[Any] | np.ndarray | None = None,
        point_chunk_size: int = 4096,
        max_kernel_evaluations: int = MAX_SMOKE_KERNEL_EVALUATIONS,
    ) -> None:
        if isinstance(operator_or_observations, NativeRaggedOperator):
            operator = operator_or_observations
        else:
            operator = NativeRaggedOperator(
                operator_or_observations,
                point_chunk_size=point_chunk_size,
                max_kernel_evaluations=max_kernel_evaluations,
            )
        self.operator = operator
        self.transform = ComplexFrequencyTransform.from_operator(operator, multipliers)

    @property
    def observations(self) -> tuple[Any, ...]:
        return self.operator.observations

    @property
    def observation_ids(self) -> tuple[Any, ...]:
        return self.operator.observation_ids

    @property
    def total_frequency_samples(self) -> int:
        return self.operator.total_frequency_samples

    @property
    def global_complex_gain(self) -> complex:
        return GLOBAL_COMPLEX_GAIN

    def forward(self, points_xyz_m: Any, coefficients: Any) -> NativePredictions:
        return self.transform.apply(self.operator.forward(points_xyz_m, coefficients))

    def adjoint(self, residuals: NativePredictions, points_xyz_m: Any) -> np.ndarray:
        return self.operator.adjoint(self.transform.adjoint_values(residuals), points_xyz_m)

    def data(self) -> NativePredictions:
        return self.transform.apply(self.operator.data())

    def _resolve_target(
        self,
        *,
        target_raw: NativePredictions | None,
        target_transformed: NativePredictions | None,
    ) -> NativePredictions:
        if target_raw is not None and target_transformed is not None:
            raise ValueError("pass exactly one of target_raw or target_transformed")
        if target_raw is None and target_transformed is None:
            return self.data()
        if target_raw is not None:
            return self.transform.apply(target_raw)
        if not isinstance(target_transformed, NativePredictions):
            raise TypeError("target_transformed must be NativePredictions with canonical IDs")
        self.transform._check_predictions(target_transformed)
        return target_transformed

    def loss_and_gradient(
        self,
        points_xyz_m: Any,
        coefficients: Any,
        *,
        target_raw: NativePredictions | None = None,
        target_transformed: NativePredictions | None = None,
        ridge: float = 0.0,
    ) -> tuple[float, np.ndarray]:
        coefficients_array = np.asarray(coefficients, dtype=np.complex128)
        if ridge < 0.0 or not np.isfinite(ridge):
            raise ValueError("ridge must be finite and nonnegative")
        prediction = self.forward(points_xyz_m, coefficients_array)
        target = self._resolve_target(target_raw=target_raw, target_transformed=target_transformed)
        residual = NativePredictions(
            self.observation_ids,
            tuple(p - y for p, y in zip(prediction.values, target.values)),
        )
        squared = sum(float(np.vdot(value, value).real) for value in residual.values)
        loss = 0.5 * squared / float(self.total_frequency_samples)
        gradient = self.adjoint(residual, points_xyz_m) / float(self.total_frequency_samples)
        if ridge:
            loss += 0.5 * float(ridge) * float(np.vdot(coefficients_array, coefficients_array).real)
            gradient = gradient + float(ridge) * coefficients_array
        return float(loss), np.asarray(gradient, dtype=np.complex128)

    def finite_difference_vjp_report(
        self,
        points_xyz_m: Any,
        coefficients: Any,
        *,
        target_raw: NativePredictions | None = None,
        target_transformed: NativePredictions | None = None,
        epsilon: float = 1.0e-6,
    ) -> dict[str, float]:
        coefficients_array = np.asarray(coefficients, dtype=np.complex128)
        _, analytic = self.loss_and_gradient(
            points_xyz_m,
            coefficients_array,
            target_raw=target_raw,
            target_transformed=target_transformed,
        )
        direction = np.arange(1, coefficients_array.size + 1, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        real_direction = direction.astype(np.complex128)
        imag_direction = 1j * direction.astype(np.complex128)
        real_plus = self.loss_and_gradient(
            points_xyz_m, coefficients_array + epsilon * real_direction,
            target_raw=target_raw, target_transformed=target_transformed,
        )[0]
        real_minus = self.loss_and_gradient(
            points_xyz_m, coefficients_array - epsilon * real_direction,
            target_raw=target_raw, target_transformed=target_transformed,
        )[0]
        imag_plus = self.loss_and_gradient(
            points_xyz_m, coefficients_array + epsilon * imag_direction,
            target_raw=target_raw, target_transformed=target_transformed,
        )[0]
        imag_minus = self.loss_and_gradient(
            points_xyz_m, coefficients_array - epsilon * imag_direction,
            target_raw=target_raw, target_transformed=target_transformed,
        )[0]
        real_fd = (real_plus - real_minus) / (2.0 * epsilon)
        imag_fd = (imag_plus - imag_minus) / (2.0 * epsilon)
        real_exact = float(np.real(np.vdot(analytic, real_direction)))
        imag_exact = float(np.real(np.vdot(analytic, imag_direction)))
        return {
            "real_relative_error": _relative_error(real_fd, real_exact),
            "imag_relative_error": _relative_error(imag_fd, imag_exact),
            "epsilon": float(epsilon),
        }

    def ridge_cgls_fit(
        self,
        points_xyz_m: Any,
        *,
        target_raw: NativePredictions | None = None,
        target_transformed: NativePredictions | None = None,
        ridge: float = 1.0e-3,
        max_iterations: int = 6,
    ) -> dict[str, Any]:
        """Bounded raw-native ridge-CGLS smoke; never a source-AF fit claim."""

        if any(str(observation.role).lower() != "train" for observation in self.observations):
            raise ValueError("raw-native ridge-CGLS readiness fit requires train observations only")
        points = np.asarray(points_xyz_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not 1 <= points.shape[0] <= MAX_SMOKE_SUPPORT_POINTS:
            raise ValueError("raw-native ridge-CGLS requires a bounded explicit support of at most five points")
        if ridge <= 0.0 or not np.isfinite(ridge):
            raise ValueError("ridge must be positive and finite")
        if int(max_iterations) <= 0 or int(max_iterations) != max_iterations or int(max_iterations) > 8:
            raise ValueError("max_iterations must be a positive integer no larger than eight")
        max_iterations = int(max_iterations)
        estimated = self.operator.guard_cgls_evaluations(points.shape[0], max_iterations)
        target = self._resolve_target(target_raw=target_raw, target_transformed=target_transformed)
        coefficient = np.zeros(points.shape[0], dtype=np.complex128)
        residual = NativePredictions(
            self.observation_ids,
            tuple(-value for value in target.values),
        )
        gradient = self.adjoint(residual, points) / float(self.total_frequency_samples) + ridge * coefficient
        direction = -gradient
        history = [_ridge_objective(residual, coefficient, self.total_frequency_samples, ridge)]
        for _ in range(max_iterations):
            direction_prediction = self.forward(points, direction)
            denominator = (
                sum(float(np.vdot(value, value).real) for value in direction_prediction.values)
                / float(self.total_frequency_samples)
                + ridge * float(np.vdot(direction, direction).real)
            )
            numerator = float(np.vdot(gradient, gradient).real)
            if not np.isfinite(denominator) or denominator <= 0.0:
                raise RuntimeError("raw-native ridge-CGLS encountered an invalid search denominator")
            alpha = numerator / denominator
            coefficient = coefficient + alpha * direction
            residual = NativePredictions(
                self.observation_ids,
                tuple(old + alpha * step for old, step in zip(residual.values, direction_prediction.values)),
            )
            next_gradient = self.adjoint(residual, points) / float(self.total_frequency_samples) + ridge * coefficient
            beta_denominator = max(numerator, np.finfo(np.float64).tiny)
            beta = float(np.vdot(next_gradient, next_gradient).real) / beta_denominator
            direction = -next_gradient + beta * direction
            objective = _ridge_objective(residual, coefficient, self.total_frequency_samples, ridge)
            history.append(objective)
            gradient = next_gradient
        if not np.isfinite(np.asarray(history)).all() or any(
            history[index + 1] > history[index] + 5.0e-12 for index in range(len(history) - 1)
        ):
            raise AssertionError("raw-native ridge-CGLS objective is not finite and monotone")
        return {
            "schema": "rift_gotcha_step3_raw_native_ridge_cgls_smoke_v1",
            "status": "PASS",
            "fit_materialization": "bounded_synthetic_raw_native_oracle_only",
            "coefficients": coefficient,
            "objective_history": tuple(float(value) for value in history),
            "kernel_evaluation_estimate": int(estimated),
            "max_kernel_evaluations": int(self.operator.max_kernel_evaluations),
            "max_iterations": max_iterations,
        }


@dataclass(frozen=True)
class SourceAFPredictions:
    ids: tuple[Any, ...]
    values: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if len(self.ids) != len(self.values):
            raise ValueError("source-AF prediction IDs and values must have equal lengths")
        _validate_ids(self.ids)
        object.__setattr__(self, "values", tuple(_readonly(value, np.complex128) for value in self.values))


@dataclass(frozen=True)
class MeasuredCamryReadinessBinding:
    """Explicit provenance token for the one bounded measured Camry diagnostic.

    The default Torch fit remains a data-free synthetic adapter smoke.  A measured
    label is available only when the caller supplies this binding after loading the
    trusted native shard, selecting the sealed TRAIN sector, and materializing the
    resulting source-AF records.
    """

    archive_path: Path
    selected_ids: tuple[Any, ...]
    expected_count: int = 117
    target_id: str = "toyota_camry"
    source_kind: str = "trusted_native_shard_source_af_records"
    selection_status: str = "verified_from_loaded_archive"
    test_payload_opened: bool = False

    def validate(self, records: Sequence[SourceAFObservation]) -> None:
        path = Path(self.archive_path)
        if path.is_symlink() or not path.is_file():
            raise ValueError("measured Camry binding requires the loaded native archive regular file")
        if self.target_id != "toyota_camry":
            raise ValueError("measured readiness binding is limited to the Toyota Camry")
        if self.source_kind != "trusted_native_shard_source_af_records":
            raise ValueError("measured readiness binding must name the trusted source-AF loader")
        if self.selection_status != "verified_from_loaded_archive":
            raise ValueError("measured readiness binding requires archive-derived selection verification")
        if self.test_payload_opened:
            raise ValueError("measured readiness binding rejects opened TEST payloads")
        if int(self.expected_count) != 117:
            raise ValueError("measured readiness binding requires the reviewed 117-pulse sector-002 scope")
        expected_ids = tuple(self.selected_ids)
        actual_ids = tuple(record.identity for record in records)
        if int(self.expected_count) != len(actual_ids) or len(expected_ids) != len(actual_ids):
            raise ValueError("measured readiness binding count does not match source-AF records")
        if expected_ids != actual_ids:
            raise ValueError("measured readiness binding IDs do not match source-AF record order")
        _validate_ids(actual_ids)
        if any(str(record.role).lower() != "train" for record in records):
            raise ValueError("measured readiness binding requires TRAIN-only source-AF records")
        if any(_identity_key(identity)[:3] != (1, "hh", 2) for identity in actual_ids):
            raise ValueError("measured readiness binding requires P1 HH sector-002 identities")


class SourceAFNativeComplexTorchOperator:
    """Differentiable float64/complex128 adapter for effective source-AF records."""

    def __init__(
        self,
        records: Sequence[SourceAFObservation],
        *,
        transform: ComplexFrequencyTransform | None = None,
        max_kernel_evaluations: int = MAX_SMOKE_KERNEL_EVALUATIONS,
    ) -> None:
        _require_torch()
        records_tuple = tuple(records)
        if not records_tuple:
            raise ValueError("at least one source-AF record is required")
        # Header/role/identity checks precede all effective payload access.
        if any(not isinstance(record, SourceAFObservation) for record in records_tuple):
            raise TypeError(
                "SourceAFNativeComplexTorchOperator requires immutable SourceAFObservation records; "
                "raw NativeObservation records must use NativeComplexMeasurementOperator"
            )
        for record in records_tuple:
            _identity_key(record.identity)
            role = str(record.role).lower()
            if role == "test":
                raise ValueError("sealed test source-AF observations are rejected before payload use")
            if role not in {"train", "validation"}:
                raise ValueError("source-AF adapter accepts train or validation observations only")
        _validate_ids(tuple(record.identity for record in records_tuple))
        frequency_counts = tuple(int(record.frequencies_hz.size) for record in records_tuple)
        if transform is None:
            transform = ComplexFrequencyTransform.identity(
                tuple(record.identity for record in records_tuple), frequency_counts
            )
        if tuple(transform.observation_ids) != tuple(record.identity for record in records_tuple):
            raise ValueError("source-AF transform IDs do not match canonical source-AF record IDs")
        self.records = records_tuple
        self.transform = transform
        self.max_kernel_evaluations = int(max_kernel_evaluations)
        if self.max_kernel_evaluations <= 0:
            raise ValueError("max_kernel_evaluations must be positive")

    @property
    def observation_ids(self) -> tuple[Any, ...]:
        return tuple(record.identity for record in self.records)

    @property
    def frequency_counts(self) -> tuple[int, ...]:
        return tuple(int(record.frequencies_hz.size) for record in self.records)

    @property
    def total_frequency_samples(self) -> int:
        return int(sum(self.frequency_counts))

    @property
    def global_complex_gain(self) -> complex:
        return GLOBAL_COMPLEX_GAIN

    def _validate_points_coefficients(self, points_xyz_m: Any, coefficients: Any) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_xyz_m, dtype=np.float64)
        coefficients_array = np.asarray(coefficients, dtype=np.complex128)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError("points_xyz_m must have shape [point,3]")
        if coefficients_array.shape != (points.shape[0],):
            raise ValueError("coefficients must have one complex value per point")
        _finite(points, "points_xyz_m")
        _finite(coefficients_array.real, "coefficients.real")
        _finite(coefficients_array.imag, "coefficients.imag")
        estimate = int(points.shape[0] * self.total_frequency_samples)
        if estimate > self.max_kernel_evaluations:
            raise RuntimeError(
                f"source-AF Torch kernel-evaluation guard exceeded: estimate={estimate} max={self.max_kernel_evaluations}"
            )
        return points, coefficients_array

    def forward_numpy(self, points_xyz_m: Any, coefficients: Any) -> SourceAFPredictions:
        points, coefficients_array = self._validate_points_coefficients(points_xyz_m, coefficients)
        values = []
        for record in self.records:
            kernel = _source_af_kernel_oracle(points, record, record.effective_r0_m)
            values.append(np.einsum("pf,p->f", kernel, coefficients_array, optimize=True))
        return SourceAFPredictions(self.observation_ids, tuple(values))

    def adjoint_numpy(self, residuals: SourceAFPredictions | Sequence[Any], points_xyz_m: Any) -> np.ndarray:
        points = np.asarray(points_xyz_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError("points_xyz_m must have shape [point,3]")
        _finite(points, "points_xyz_m")
        if isinstance(residuals, SourceAFPredictions):
            if tuple(residuals.ids) != self.observation_ids:
                raise ValueError("source-AF residual identities do not match canonical records")
            values = residuals.values
        else:
            values = tuple(np.asarray(value, dtype=np.complex128) for value in residuals)
        if len(values) != len(self.records):
            raise ValueError("source-AF residual count does not match records")
        result = np.zeros(points.shape[0], dtype=np.complex128)
        for record, residual, count in zip(self.records, values, self.frequency_counts):
            if residual.shape != (count,):
                raise ValueError("source-AF residual shape does not match native frequencies")
            _finite(residual.real, "source-AF residual.real")
            _finite(residual.imag, "source-AF residual.imag")
            kernel = _source_af_kernel_oracle(points, record, record.effective_r0_m)
            result += np.einsum("pf,f->p", np.conjugate(kernel), residual, optimize=True)
        return _readonly(result, np.complex128)

    def data_numpy(self) -> SourceAFPredictions:
        return SourceAFPredictions(self.observation_ids, tuple(record.effective_response for record in self.records))

    @staticmethod
    def _torch_points(points_xyz_m: Any, *, device: Any = None) -> Any:
        import torch

        points = (
            points_xyz_m.to(device=device, dtype=torch.float64)
            if torch.is_tensor(points_xyz_m)
            else torch.tensor(points_xyz_m, dtype=torch.float64, device=device)
        )
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError("Torch points_xyz_m must have shape [point,3]")
        if not bool(torch.isfinite(points).all()):
            raise ValueError("Torch points_xyz_m contains non-finite values")
        return points

    @staticmethod
    def _torch_float64(value: Any, *, device: Any) -> Any:
        import torch

        result = value.to(device=device, dtype=torch.float64) if torch.is_tensor(value) else torch.tensor(
            value, dtype=torch.float64, device=device
        )
        if not bool(torch.isfinite(result).all()):
            raise ValueError("Torch real/imag input contains non-finite values")
        return result

    def _torch_kernel(self, points: Any, record: SourceAFObservation) -> Any:
        import torch

        position = torch.tensor(record.position_xyz_m, dtype=torch.float64, device=points.device)
        frequencies = torch.tensor(record.frequencies_hz, dtype=torch.float64, device=points.device)
        distance = torch.linalg.vector_norm(points - position[None, :], dim=1)
        phase_real = -(4.0 * np.pi / SPEED_OF_LIGHT_M_S) * (
            distance[:, None] - float(record.effective_r0_m)
        ) * frequencies[None, :]
        return torch.exp(torch.complex(torch.zeros_like(phase_real), phase_real))

    def forward_torch(self, points_xyz_m: Any, coefficients_real: Any, coefficients_imag: Any) -> tuple[Any, ...]:
        import torch

        points = self._torch_points(points_xyz_m)
        real = self._torch_float64(coefficients_real, device=points.device)
        imag = self._torch_float64(coefficients_imag, device=points.device)
        if real.shape != (points.shape[0],) or imag.shape != (points.shape[0],):
            raise ValueError("Torch real/imag coefficients must each have shape [point]")
        estimate = int(points.shape[0] * self.total_frequency_samples)
        if estimate > self.max_kernel_evaluations:
            raise RuntimeError("source-AF Torch kernel-evaluation guard exceeded")
        coefficients = torch.complex(real, imag)
        values = []
        for record in self.records:
            values.append(torch.einsum("pf,p->f", self._torch_kernel(points, record), coefficients))
        return self.transform.apply_torch(tuple(values), device=points.device)

    def adjoint_torch(self, residuals: Sequence[Any], points_xyz_m: Any) -> Any:
        import torch

        points = self._torch_points(points_xyz_m)
        transformed = self.transform.adjoint_torch(tuple(residuals), device=points.device)
        result = torch.zeros(points.shape[0], dtype=torch.complex128, device=points.device)
        for record, residual in zip(self.records, transformed):
            result = result + torch.einsum("pf,f->p", torch.conj(self._torch_kernel(points, record)), residual)
        return result

    def data_torch(self, *, device: Any = None) -> tuple[Any, ...]:
        import torch

        values = tuple(torch.tensor(record.effective_response, dtype=torch.complex128, device=device) for record in self.records)
        return self.transform.apply_torch(values, device=device)

    def _resolve_torch_target(
        self,
        *,
        device: Any,
        target_raw: SourceAFPredictions | None,
        target_transformed: SourceAFPredictions | None,
    ) -> tuple[Any, ...]:
        import torch

        if target_raw is not None and target_transformed is not None:
            raise ValueError("pass exactly one of target_raw or target_transformed")
        if target_raw is None and target_transformed is None:
            return self.data_torch(device=device)
        selected = target_raw if target_raw is not None else target_transformed
        if not isinstance(selected, SourceAFPredictions):
            raise TypeError(
                "caller-supplied source-AF targets must be SourceAFPredictions with canonical IDs"
            )
        if tuple(selected.ids) != self.observation_ids:
            raise ValueError("source-AF target IDs do not match canonical record IDs")
        values = selected.values
        if len(values) != len(self.records):
            raise ValueError("Torch target count does not match source-AF observations")
        tensors = tuple(
            value.to(device=device, dtype=torch.complex128)
            if torch.is_tensor(value)
            else torch.tensor(value, dtype=torch.complex128, device=device)
            for value in values
        )
        for tensor, count in zip(tensors, self.frequency_counts):
            if tuple(tensor.shape) != (count,):
                raise ValueError("Torch target shape does not match source-AF frequencies")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("Torch target contains non-finite values")
        if target_raw is not None:
            return self.transform.apply_torch(tensors, device=device)
        return tensors

    def loss_torch(
        self,
        points_xyz_m: Any,
        coefficients_real: Any,
        coefficients_imag: Any,
        *,
        target_raw: SourceAFPredictions | None = None,
        target_transformed: SourceAFPredictions | None = None,
        ridge: float = 0.0,
    ) -> Any:
        import torch

        prediction = self.forward_torch(points_xyz_m, coefficients_real, coefficients_imag)
        target_values = self._resolve_torch_target(
            device=prediction[0].device,
            target_raw=target_raw,
            target_transformed=target_transformed,
        )
        residual = []
        for predicted, observed, count in zip(prediction, target_values, self.frequency_counts):
            if tuple(observed.shape) != (count,):
                raise ValueError("Torch target shape does not match source-AF frequencies")
            residual.append(predicted - observed)
        squared = sum(torch.sum(torch.abs(value) ** 2) for value in residual)
        loss = 0.5 * squared / float(self.total_frequency_samples)
        if ridge:
            if ridge < 0.0 or not np.isfinite(ridge):
                raise ValueError("ridge must be finite and nonnegative")
            real_coefficients = self._torch_float64(coefficients_real, device=prediction[0].device)
            imag_coefficients = self._torch_float64(coefficients_imag, device=prediction[0].device)
            loss = loss + 0.5 * float(ridge) * (
                torch.sum(real_coefficients ** 2)
                + torch.sum(imag_coefficients ** 2)
            )
        if not bool(torch.isfinite(loss)):
            raise ValueError("Torch native-complex loss is non-finite")
        return loss

    def manual_gradient_torch(
        self,
        points_xyz_m: Any,
        coefficients_real: Any,
        coefficients_imag: Any,
        *,
        target_raw: SourceAFPredictions | None = None,
        target_transformed: SourceAFPredictions | None = None,
        ridge: float = 0.0,
    ) -> tuple[Any, Any, Any]:
        import torch

        prediction = self.forward_torch(points_xyz_m, coefficients_real, coefficients_imag)
        target_values = self._resolve_torch_target(
            device=prediction[0].device,
            target_raw=target_raw,
            target_transformed=target_transformed,
        )
        residual = tuple(value - observed for value, observed in zip(prediction, target_values))
        gradient_complex = self.adjoint_torch(residual, points_xyz_m) / float(self.total_frequency_samples)
        real = self._torch_float64(coefficients_real, device=gradient_complex.device)
        imag = self._torch_float64(coefficients_imag, device=gradient_complex.device)
        if ridge:
            gradient_complex = gradient_complex + float(ridge) * torch.complex(real, imag)
        loss_kwargs: dict[str, Any] = {"ridge": ridge}
        if target_raw is not None:
            loss_kwargs["target_raw"] = target_raw
        elif target_transformed is not None:
            loss_kwargs["target_transformed"] = target_transformed
        return self.loss_torch(points_xyz_m, real, imag, **loss_kwargs), gradient_complex.real, gradient_complex.imag

    def finite_difference_vjp_report(
        self,
        points_xyz_m: Any,
        coefficients_real: Any,
        coefficients_imag: Any,
        *,
        target_raw: SourceAFPredictions | None = None,
        target_transformed: SourceAFPredictions | None = None,
        ridge: float = 0.0,
        epsilon: float = 1.0e-6,
    ) -> dict[str, float]:
        """Compare Torch-bridge real/imag VJPs with central finite differences."""

        import torch

        real = np.asarray(coefficients_real, dtype=np.float64)
        imag = np.asarray(coefficients_imag, dtype=np.float64)
        direction = np.arange(1, real.size + 1, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        def evaluate(real_value: np.ndarray, imag_value: np.ndarray) -> float:
            with torch.no_grad():
                value = self.loss_torch(
                    points_xyz_m,
                    torch.as_tensor(real_value, dtype=torch.float64),
                    torch.as_tensor(imag_value, dtype=torch.float64),
                    target_raw=target_raw,
                    target_transformed=target_transformed,
                    ridge=ridge,
                )
            return float(value.detach().cpu())

        _, analytic_real, analytic_imag = self.manual_gradient_torch(
            points_xyz_m,
            torch.as_tensor(real, dtype=torch.float64),
            torch.as_tensor(imag, dtype=torch.float64),
            target_raw=target_raw,
            target_transformed=target_transformed,
            ridge=ridge,
        )
        real_fd = (
            evaluate(real + epsilon * direction, imag) - evaluate(real - epsilon * direction, imag)
        ) / (2.0 * epsilon)
        imag_fd = (
            evaluate(real, imag + epsilon * direction) - evaluate(real, imag - epsilon * direction)
        ) / (2.0 * epsilon)
        return {
            "real_relative_error": _relative_error(real_fd, float(torch.dot(analytic_real, torch.as_tensor(direction, dtype=torch.float64)))),
            "imag_relative_error": _relative_error(imag_fd, float(torch.dot(analytic_imag, torch.as_tensor(direction, dtype=torch.float64)))),
            "epsilon": float(epsilon),
        }

    def bounded_torch_ridge_fit(
        self,
        points_xyz_m: Any,
        *,
        target_raw: SourceAFPredictions | None = None,
        target_transformed: SourceAFPredictions | None = None,
        ridge: float = 1.0e-3,
        max_iterations: int = 6,
        measured_binding: MeasuredCamryReadinessBinding | None = None,
    ) -> dict[str, Any]:
        """Small train-only Torch fit with an explicit synthetic/measured label."""

        if any(str(record.role).lower() != "train" for record in self.records):
            raise ValueError("source-AF Torch ridge fit requires train records only")
        if measured_binding is None:
            fit_materialization = "bounded_synthetic_source_af_adapter_only"
        else:
            if not isinstance(measured_binding, MeasuredCamryReadinessBinding):
                raise TypeError("measured_binding must be MeasuredCamryReadinessBinding")
            measured_binding.validate(self.records)
            fit_materialization = "bounded_train_only_measured_camry_readiness_diagnostic"
        points = self._torch_points(points_xyz_m)
        if points.shape[0] < MIN_SMOKE_SUPPORT_POINTS or points.shape[0] > MAX_SMOKE_SUPPORT_POINTS:
            raise ValueError("source-AF Torch fit requires an explicit 3-5 point smoke support")
        if ridge <= 0.0 or not np.isfinite(ridge):
            raise ValueError("ridge must be positive and finite")
        if int(max_iterations) <= 0 or int(max_iterations) != max_iterations or int(max_iterations) > 8:
            raise ValueError("max_iterations must be a positive integer no larger than eight")
        max_iterations = int(max_iterations)
        kernel_evaluation_estimate = int(
            (1 + 2 * max_iterations) * points.shape[0] * self.total_frequency_samples
        )
        if kernel_evaluation_estimate > self.max_kernel_evaluations:
            raise RuntimeError(
                "source-AF Torch ridge-fit cumulative kernel-evaluation guard exceeded: "
                f"estimate={kernel_evaluation_estimate} max={self.max_kernel_evaluations}"
            )
        import torch

        multiplier_energy = float(
            sum(np.vdot(multiplier, multiplier).real for multiplier in self.transform.multipliers)
        )
        lipschitz_upper_bound = (
            float(points.shape[0]) * multiplier_energy / float(self.total_frequency_samples) + float(ridge)
        )
        if not np.isfinite(lipschitz_upper_bound) or lipschitz_upper_bound <= 0.0:
            raise RuntimeError("source-AF Torch ridge-fit Lipschitz bound is invalid")
        step_size = 0.5 / lipschitz_upper_bound
        real = torch.nn.Parameter(torch.zeros(points.shape[0], dtype=torch.float64, device=points.device))
        imag = torch.nn.Parameter(torch.zeros(points.shape[0], dtype=torch.float64, device=points.device))
        optimizer = torch.optim.SGD((real, imag), lr=step_size)
        history: list[float] = []
        measurement_history: list[float] = []
        ridge_penalty_history: list[float] = []
        gradient_norm_history: list[float] = []
        for _ in range(max_iterations + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = self.loss_torch(
                points,
                real,
                imag,
                target_raw=target_raw,
                target_transformed=target_transformed,
                ridge=ridge,
            )
            loss_value = float(loss.detach().cpu())
            ridge_penalty_value = 0.5 * float(ridge) * float(
                (real.detach() ** 2 + imag.detach() ** 2).sum().cpu()
            )
            measurement_value = loss_value - ridge_penalty_value
            if not np.isclose(
                loss_value,
                measurement_value + ridge_penalty_value,
                rtol=0.0,
                atol=1.0e-14,
            ):
                raise AssertionError("Torch source-AF objective did not separate its ridge penalty")
            if history and loss_value > history[-1] + 1.0e-12:
                raise AssertionError("Torch source-AF readiness objective increased")
            history.append(loss_value)
            measurement_history.append(measurement_value)
            ridge_penalty_history.append(ridge_penalty_value)
            if len(history) == max_iterations + 1:
                break
            loss.backward()
            if real.grad is None or imag.grad is None:
                raise RuntimeError("Torch source-AF readiness fit did not produce real/imag gradients")
            gradient_norm_history.append(
                float(torch.sqrt((real.grad ** 2 + imag.grad ** 2).sum()).detach().cpu())
            )
            optimizer.step()
            if not bool(torch.isfinite(real).all()) or not bool(torch.isfinite(imag).all()):
                raise RuntimeError("Torch source-AF readiness fit produced non-finite parameters")
        return {
            "schema": "rift_gotcha_step3_source_af_torch_ridge_smoke_v1",
            "status": "PASS",
            "fit_materialization": fit_materialization,
            "objective_history": tuple(history),
            "measurement_objective_history": tuple(measurement_history),
            "ridge_penalty_history": tuple(ridge_penalty_history),
            "gradient_norm_history": tuple(gradient_norm_history),
            "coefficients_real": real.detach().cpu().numpy(),
            "coefficients_imag": imag.detach().cpu().numpy(),
            "max_iterations": max_iterations,
            "kernel_evaluation_estimate": kernel_evaluation_estimate,
            "max_kernel_evaluations": int(self.max_kernel_evaluations),
            "kernel_guard": "PASS_cumulative_preflight",
            "lipschitz_upper_bound": float(lipschitz_upper_bound),
            "step_size": float(step_size),
            "step_rule": "0.5 / (point_count * sum_abs_multiplier_squared / native_sample_count + ridge)",
            "dtype": "float64_real_imag_complex128_measurements",
        }


def _ridge_objective(residuals: NativePredictions, coefficients: np.ndarray, count: int, ridge: float) -> float:
    return float(
        0.5 * sum(float(np.vdot(value, value).real) for value in residuals.values) / float(count)
        + 0.5 * float(ridge) * float(np.vdot(coefficients, coefficients).real)
    )


def _relative_error(actual: float, expected: float) -> float:
    return float(abs(float(actual) - float(expected)) / max(1.0, abs(float(actual)), abs(float(expected))))


def smoke_support_local() -> np.ndarray:
    return _readonly(
        [
            [0.0, 0.0, 0.0],
            [0.37, -0.21, 0.13],
            [-0.62, 0.44, -0.18],
            [0.91, 0.28, 0.31],
        ],
        np.float64,
    )


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_native_complex_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle, object_pairs_hook=_reject_duplicate_json_pairs)
    validate_native_complex_protocol(payload)
    return payload


def validate_native_complex_protocol(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != NATIVE_COMPLEX_PROTOCOL_SCHEMA:
        raise ValueError("unexpected Step-3 native-complex protocol schema")
    if payload.get("stage") != "local_readiness":
        raise ValueError("native-complex v2 protocol must remain local-readiness stage")
    if payload.get("data_free") is not True or payload.get("measured_fit_release") is not False:
        raise ValueError("native-complex v2 protocol must be data-free and measured-fit fail-closed")
    if payload.get("test_policy") != "sealed_no_selection":
        raise ValueError("native-complex v2 protocol must keep the sealed TEST policy")
    targets = payload.get("targets")
    if not isinstance(targets, Mapping) or set(targets) != set(TARGET_IDS):
        raise ValueError("native-complex v2 protocol must declare exactly the two selected targets")
    expected_grids = exact_reference_grids()
    for target_id, grid in expected_grids.items():
        actual = targets[target_id].get("grid")
        if not isinstance(actual, Mapping):
            raise ValueError(f"missing exact {target_id} reference grid")
        for key, expected in grid.as_dict().items():
            if key == "target_id":
                continue
            if actual.get(key) != expected:
                raise ValueError(f"{target_id} grid field {key} does not preserve the exact local reporting grid")
    tophat_native = targets["tophat"].get("native_placement")
    if not isinstance(tophat_native, Mapping) or tophat_native.get("R") is not None or tophat_native.get("t_m") is not None:
        raise ValueError("Tophat numeric native R/t must remain absent in the v2 readiness protocol")
    camry_native = targets["toyota_camry"].get("native_placement")
    if not isinstance(camry_native, Mapping):
        raise ValueError("Camry declared native placement is missing")
    if not np.array_equal(np.asarray(camry_native.get("R"), dtype=np.float64), CAMRY_VEHICLE_AXIS_ROTATION):
        raise ValueError("Camry declared native R does not match the Point-C working convention")
    if not np.array_equal(np.asarray(camry_native.get("t_m"), dtype=np.float64), CAMRY_TRANSLATION_M):
        raise ValueError("Camry declared native t does not match the Point-C working convention")
    footprint = camry_footprint_consistency()
    if camry_native.get("nominal_footprint_absolute_corner_tolerance_m") != CAMRY_FOOTPRINT_ABSOLUTE_TOLERANCE_M:
        raise ValueError("Camry footprint consistency must declare the 0.10 m absolute corner tolerance")
    if camry_native.get("inverse_local_footprint_containment") != "all_four_corners_in[-5,5)^3":
        raise ValueError("Camry footprint declaration must check all inverse-local corners inside [-5,5)^3")
    if not footprint["all_corners_within_absolute_tolerance"] or not footprint["inverse_footprint_contained_in_camry_grid"]:
        raise ValueError("declared Camry footprint fails the bounded all-corner geometry check")
    if camry_native.get("actual_extraction_fit") != "not_released_by_this_data_free_protocol_requires_reviewed_measured_package":
        raise ValueError("Camry extraction/fit status must remain data-free and review-gated")
    smoke = payload.get("smoke_support")
    if not isinstance(smoke, Mapping) or smoke.get("reference_grid_allowed") is not False:
        raise ValueError("v2 smoke support must be distinct from the reporting grids")
    points = np.asarray(smoke.get("local_points_m"), dtype=np.float64)
    if points.shape != (4, 3):
        raise ValueError("v2 protocol must carry four explicit local smoke points")
    if smoke.get("point_count") != 4:
        raise ValueError("v2 protocol smoke point_count must be four")
    transform_smoke_support(
        "tophat",
        points,
        placement=RigidPlacement(np.eye(3), np.zeros(3), name="protocol_validation_synthetic"),
    )
    transform_smoke_support("toyota_camry", points)
    if payload.get("measurement_transform", {}).get("data_prediction_application") != "same_complex_multiplier_before_loss_magnitude_or_power":
        raise ValueError("v2 protocol must apply one shared complex transform before loss/magnitude/power")
    if payload.get("source_af_operator", {}).get("actual_fit_materialization") != "not_released_data_free_readiness_only":
        raise ValueError("source-AF actual fit materialization must remain unreleased")
    bounds = payload.get("bounds")
    if not isinstance(bounds, Mapping) or bounds.get("optimizer_step_rule") != "0.5 / (point_count * sum_abs_multiplier_squared / native_sample_count + ridge)":
        raise ValueError("v2 protocol must declare the data-dependent Torch optimizer step bound")
    if bounds.get("optimizer_extra_forward_evaluations") != 0:
        raise ValueError("v2 optimizer accounting must not hide extra forward evaluations")


def synthetic_native_observations(*, official_autofocus: bool = False) -> tuple[NativeObservation, ...]:
    """Create only tiny in-memory native headers/payloads for the validator."""

    positions = (
        np.asarray([34.0, -22.0, 6.0]),
        np.asarray([17.0, 31.0, 8.0]),
        np.asarray([-28.0, 19.0, 11.0]),
    )
    frequencies = (
        np.asarray([9.10e9, 9.17e9, 9.31e9]),
        np.asarray([9.11e9, 9.23e9, 9.37e9, 9.46e9]),
        np.asarray([9.08e9, 9.19e9, 9.34e9, 9.52e9, 9.61e9]),
    )
    records = []
    for index, (position, freq) in enumerate(zip(positions, frequencies)):
        response = (0.4 + 0.15j) * np.exp(1j * np.arange(freq.size) * 0.21) * (index + 1)
        response = _readonly(response, np.complex128)
        if official_autofocus:
            r_correct = 0.0015 + 0.0003 * index
            ph_correct = 0.04 + 0.01 * index
            autofocus = AutofocusProvenance(
                schema=AUTOFOCUS_PROVENANCE_SCHEMA,
                mode=AUTOFOCUS_PUBLISHED,
                official_available=True,
                applied=False,
                source_shard_id="pass1_hh",
                range_field="r_correct",
                phase_field="ph_correct",
            )
        else:
            r_correct = None
            ph_correct = None
            autofocus = AutofocusProvenance(
                schema=AUTOFOCUS_PROVENANCE_SCHEMA,
                mode=AUTOFOCUS_OFFICIAL_ABSENT,
                official_available=False,
                applied=False,
                source_shard_id="pass1_hh",
                range_field=None,
                phase_field=None,
            )
        records.append(
            NativeObservation(
                identity=NativeObservationId(1, "hh", 2 + index, 100 + index),
                role="train",
                response=response,
                frequencies_hz=_readonly(freq, np.float64),
                position_xyz_m=_readonly(position, np.float64),
                r0_m=float(np.linalg.norm(position)),
                th_deg=0.0,
                phi_deg=0.0,
                r_correct_raw=r_correct,
                ph_correct_raw=ph_correct,
                phase_reference=PhaseReferenceContract(),
                autofocus=autofocus,
            )
        )
    return tuple(records)


def source_af_records_from_synthetic() -> tuple[SourceAFObservation, ...]:
    scope = SourceAFScope(
        pass_id=1,
        polarization="hh",
        sector_ids=(2, 3, 4),
        role="train",
        expected_count=3,
        name="bounded_native_complex_synthetic_train",
    )
    return tuple(SourceAFObservation.from_observation(record, scope=scope) for record in synthetic_native_observations(official_autofocus=True))


__all__ = [
    "CAMRY_DECLARED_PLACEMENT",
    "CAMRY_VEHICLE_AXIS_ROTATION",
    "CAMRY_TRANSLATION_M",
    "CAMRY_FOOTPRINT_ABSOLUTE_TOLERANCE_M",
    "CAMRY_FOOTPRINT_NATIVE_CORNERS_M",
    "CAMRY_NOMINAL_LOCAL_CORNERS_M",
    "ComplexFrequencyTransform",
    "DOT_RELATIVE_TOLERANCE",
    "GRADIENT_RELATIVE_TOLERANCE",
    "GLOBAL_COMPLEX_GAIN",
    "MAX_SMOKE_KERNEL_EVALUATIONS",
    "MAX_SMOKE_SUPPORT_POINTS",
    "MIN_SMOKE_SUPPORT_POINTS",
    "MeasuredCamryReadinessBinding",
    "MissingTophatNativeTransformError",
    "NATIVE_COMPLEX_PROTOCOL_SCHEMA",
    "NATIVE_COMPLEX_READINESS_SCHEMA",
    "NativeComplexMeasurementOperator",
    "ReferenceGrid",
    "RigidPlacement",
    "SourceAFNativeComplexTorchOperator",
    "SourceAFPredictions",
    "TARGET_IDS",
    "TORCH_AVAILABLE",
    "exact_reference_grids",
    "camry_footprint_consistency",
    "load_native_complex_protocol",
    "placement_for_target",
    "smoke_support_local",
    "source_af_records_from_synthetic",
    "synthetic_native_observations",
    "transform_smoke_support",
    "validate_native_complex_protocol",
]
