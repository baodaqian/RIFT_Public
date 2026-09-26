"""Bounded, matrix-free GOTCHA Step-2 physical controls.

This module is additive to :mod:`rift.gotcha_acquisition` and intentionally
does not import the ``rift`` package.  Callers pass Batch-A
``NativeShard``/``NativeObservation`` records (or records loaded from that
module by path).  The only real-data phase hypothesis named here is the
published-candidate convention
``exp(-i * 4*pi*f/c * (R-r0))`` and its conjugate adjoint.  No autofocus,
uniform-grid approximation, amplitude weighting, legacy BP, or implicit
support is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SPEED_OF_LIGHT_M_S = 299_792_458.0
PHASE_HYPOTHESIS_NAME = "demanet_2012_gotcha_phase_candidate"
PHASE_HYPOTHESIS_FORWARD = "exp(-i * 4*pi*f/c * (R-r0))"
PHASE_HYPOTHESIS_ADJOINT = "conjugate(exp(-i * 4*pi*f/c * (R-r0)))"
PHASE_HYPOTHESIS_STATUS = "named_candidate_unverified_for_real_archive"
DEFAULT_PREFIX_COUNTS = (1, 8, 32, 400, 1600)
H0_SUPPORT_SCHEMA = "rift_gotcha_native_frame_h0_support_v1"
SUPPORT_STATUS = "conditional_imaging_hypothesis_only"

_PASS_IDS = tuple(range(1, 9))
_POLARIZATIONS = ("hh", "hv", "vh", "vv")
_ROLES = ("train", "validation")


def _readonly(value: Any, *, dtype: Any = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _finite(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _as_identity(observation: Any) -> Any:
    identity = getattr(observation, "identity", None)
    if identity is None:
        raise TypeError("Step-2 controls require Batch-A NativeObservation.identity")
    for name in ("pass_id", "polarization", "sector_id", "pulse_index"):
        if not hasattr(identity, name):
            raise TypeError(f"native identity lacks {name}")
    if int(identity.pass_id) not in _PASS_IDS:
        raise ValueError("observation pass identity is outside the frozen GOTCHA set")
    if str(identity.polarization).lower() not in _POLARIZATIONS:
        raise ValueError("observation polarization is outside the frozen GOTCHA set")
    return identity


def _validate_headers(observations: Sequence[Any]) -> tuple[Any, ...]:
    """Validate metadata before touching response payloads."""

    records = tuple(observations)
    if not records:
        raise ValueError("at least one native observation is required")
    seen: set[tuple[int, str, int, int]] = set()
    for observation in records:
        identity = _as_identity(observation)
        role = str(getattr(observation, "role", "")).lower()
        if role == "test":
            raise ValueError("sealed test observations are rejected before payload use")
        if role not in _ROLES:
            raise ValueError("Step-2 controls accept train or validation observations only")
        key = (
            int(identity.pass_id),
            str(identity.polarization).lower(),
            int(identity.sector_id),
            int(identity.pulse_index),
        )
        if key in seen:
            raise ValueError(f"duplicate native observation identity: {key}")
        seen.add(key)
        position = np.asarray(getattr(observation, "position_xyz_m", None))
        if position.shape != (3,):
            raise ValueError("position_xyz_m must have shape [3]")
        _finite(position, "position_xyz_m")
        r0 = float(getattr(observation, "r0_m"))
        if not np.isfinite(r0):
            raise ValueError("r0_m must be finite")
        phase_reference = getattr(observation, "phase_reference", None)
        if phase_reference is None or getattr(phase_reference, "reference_range_field", None) != "r0":
            raise ValueError("native observations must retain the per-pulse r0 reference field")
        if getattr(phase_reference, "geometry_contract", None) != "paired_monostatic_tx_equals_rx_same_observation":
            raise ValueError("native observations must declare paired Tx=Rx geometry")
        autofocus = getattr(observation, "autofocus", None)
        if autofocus is None or bool(getattr(autofocus, "applied", True)):
            raise ValueError("Step-2 controls require raw/unapplied autofocus provenance")
        frequencies = np.asarray(getattr(observation, "frequencies_hz", None))
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError("frequencies_hz must be a nonempty native vector")
        _finite(frequencies.astype(np.float64), "frequencies_hz")
        if frequencies.size > 1 and not np.all(np.diff(frequencies.astype(np.float64)) > 0):
            raise ValueError("native frequencies must be strictly increasing")
    return records


def _validate_payload_shapes(observations: Sequence[Any]) -> tuple[Any, ...]:
    records = _validate_headers(observations)
    for observation in records:
        response = np.asarray(getattr(observation, "response", None))
        frequencies = np.asarray(observation.frequencies_hz)
        if response.shape != frequencies.shape or not np.iscomplexobj(response):
            raise ValueError("each response must be complex with the native [frequency] shape")
        _finite(response.real, "response.real")
        _finite(response.imag, "response.imag")
    return records


def _require_train_observations(observations: Sequence[Any]) -> tuple[Any, ...]:
    records = _validate_headers(observations)
    if any(str(observation.role).lower() != "train" for observation in records):
        raise ValueError("inverse-fit APIs require train observations only")
    return records


def _shard_id(observation: Any) -> str:
    identity = _as_identity(observation)
    return f"pass{int(identity.pass_id)}_{str(identity.polarization).lower()}"


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): value[key] for key in value})


@dataclass(frozen=True)
class NestedTrainingPrefix:
    """One deterministic train-only prefix, keyed by native shard identity."""

    count_per_shard: int
    seed: int
    ids_by_shard: Mapping[str, tuple[Any, ...]]
    counts_by_pass: Mapping[int, int]

    def __post_init__(self) -> None:
        if self.count_per_shard <= 0:
            raise ValueError("prefix count must be positive")
        if any(str(identity.polarization).lower() not in _POLARIZATIONS for ids in self.ids_by_shard.values() for identity in ids):
            raise ValueError("prefix contains an invalid polarization identity")

    @property
    def observation_count(self) -> int:
        return sum(len(ids) for ids in self.ids_by_shard.values())

    @property
    def ids(self) -> tuple[Any, ...]:
        return tuple(identity for ids in self.ids_by_shard.values() for identity in ids)

    def as_dict(self) -> dict[str, Any]:
        return {
            "count_per_shard": self.count_per_shard,
            "seed": self.seed,
            "ids_by_shard": {
                key: [identity.as_dict() for identity in ids]
                for key, ids in self.ids_by_shard.items()
            },
            "counts_by_pass": dict(self.counts_by_pass),
        }


def select_train_prefixes(
    shards: Sequence[Any],
    *,
    counts: Sequence[int] = DEFAULT_PREFIX_COUNTS,
    seed: int = 42,
    requested_passes: Sequence[int] | None = None,
    requested_polarizations: Sequence[str] | None = None,
) -> Mapping[int, NestedTrainingPrefix]:
    """Select nested, deterministic, balanced train-only native identities.

    Each pass/polarization shard gets the same prefix length.  Identities are
    shuffled independently by a stable ``SeedSequence``; no row number is ever
    used to join channels or passes.
    """

    shards = tuple(shards)
    if not shards:
        raise ValueError("at least one NativeShard is required")
    counts = tuple(int(value) for value in counts)
    if not counts or any(value <= 0 for value in counts) or tuple(sorted(set(counts))) != counts:
        raise ValueError("counts must be strictly increasing positive integers")
    requested_pass_set = (
        {int(value) for value in requested_passes}
        if requested_passes is not None
        else {int(getattr(shard, "pass_id")) for shard in shards}
    )
    requested_pol_set = (
        {str(value).lower() for value in requested_polarizations}
        if requested_polarizations is not None
        else {str(getattr(shard, "polarization")).lower() for shard in shards}
    )
    if not requested_pass_set or not requested_pol_set:
        raise ValueError("requested pass/polarization sets must be nonempty")
    shard_ids: dict[str, tuple[Any, ...]] = {}
    for shard in shards:
        # Inspect the row-role vector before invoking observations(), which
        # keeps sealed-test rejection ahead of response payload use.
        roles = np.asarray(getattr(shard, "role", None))
        if roles.size and np.any(np.asarray([str(value).lower() for value in roles]) == "test"):
            raise ValueError("sealed test rows are rejected before payload use")
        pass_id = int(getattr(shard, "pass_id"))
        polarization = str(getattr(shard, "polarization")).lower()
        if pass_id not in requested_pass_set or polarization not in requested_pol_set:
            continue
        key = str(getattr(shard, "shard_id", f"pass{pass_id}_{polarization}"))
        if key in shard_ids:
            raise ValueError(f"duplicate requested shard: {key}")
        identities = tuple(getattr(shard, "identities_for_role")("train"))
        if len(identities) < max(counts):
            raise ValueError(f"{key} has only {len(identities)} train observations")
        seed_sequence = np.random.SeedSequence(
            [int(seed), pass_id, _POLARIZATIONS.index(polarization)]
        )
        order = np.random.default_rng(seed_sequence).permutation(len(identities))
        shard_ids[key] = tuple(identities[int(index)] for index in order)
    expected_keys = {
        f"pass{pass_id}_{polarization}"
        for pass_id in requested_pass_set
        for polarization in requested_pol_set
    }
    if set(shard_ids) != expected_keys:
        raise ValueError(
            f"requested pass/polarization shards are incomplete: missing={sorted(expected_keys - set(shard_ids))}"
        )
    prefixes: dict[int, NestedTrainingPrefix] = {}
    for count in counts:
        selected = {
            key: tuple(identities[:count]) for key, identities in sorted(shard_ids.items())
        }
        counts_by_pass = {
            pass_id: sum(
                len(ids)
                for key, ids in selected.items()
                if key.startswith(f"pass{pass_id}_")
            )
            for pass_id in sorted(requested_pass_set)
        }
        prefixes[count] = NestedTrainingPrefix(
            count_per_shard=count,
            seed=int(seed),
            ids_by_shard=_freeze_mapping(selected),
            counts_by_pass=MappingProxyType(counts_by_pass),
        )
    return MappingProxyType(prefixes)


def validate_nested_prefixes(prefixes: Mapping[int, NestedTrainingPrefix]) -> dict[str, Any]:
    ordered = sorted((int(key), value) for key, value in prefixes.items())
    if not ordered:
        raise ValueError("at least one prefix is required")
    for (small_count, small), (large_count, large) in zip(ordered, ordered[1:]):
        if small_count >= large_count or set(small.ids_by_shard) != set(large.ids_by_shard):
            raise ValueError("prefixes are not ordered/nested by the same shards")
        for key in small.ids_by_shard:
            if tuple(small.ids_by_shard[key]) != tuple(large.ids_by_shard[key][:small_count]):
                raise ValueError(f"prefix {small_count} is not nested in {large_count} for {key}")
    return {
        "prefix_counts": [count for count, _ in ordered],
        "nested": True,
        "shard_count": len(ordered[0][1].ids_by_shard),
        "counts_by_pass": dict(ordered[-1][1].counts_by_pass),
    }


@dataclass(frozen=True)
class NativePredictions:
    ids: tuple[Any, ...]
    values: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if len(self.ids) != len(self.values):
            raise ValueError("prediction IDs and values must have equal lengths")
        object.__setattr__(
            self,
            "values",
            tuple(_readonly(value, dtype=np.complex128) for value in self.values),
        )

    def by_id(self) -> Mapping[Any, np.ndarray]:
        return MappingProxyType(dict(zip(self.ids, self.values)))


@dataclass(frozen=True)
class BackprojectionResult:
    values: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _readonly(self.values, dtype=np.complex128))
        object.__setattr__(self, "metadata", _freeze_mapping(dict(self.metadata)))


class NativeRaggedOperator:
    """Direct float64/complex128 native-frequency operator without a matrix."""

    def __init__(
        self,
        observations: Sequence[Any],
        *,
        point_chunk_size: int = 4096,
        max_kernel_evaluations: int = 20_000_000,
        phase_hypothesis: str = PHASE_HYPOTHESIS_NAME,
    ) -> None:
        self._observations = _validate_headers(observations)
        polarizations = {str(observation.identity.polarization).lower() for observation in self._observations}
        if len(polarizations) != 1:
            raise ValueError("one native operator must contain exactly one polarization")
        self.polarization = next(iter(polarizations))
        self.point_chunk_size = int(point_chunk_size)
        self.max_kernel_evaluations = int(max_kernel_evaluations)
        self.phase_hypothesis = str(phase_hypothesis)
        if self.point_chunk_size <= 0:
            raise ValueError("point_chunk_size must be positive")
        if self.max_kernel_evaluations <= 0:
            raise ValueError("max_kernel_evaluations must be positive")
        if self.phase_hypothesis != PHASE_HYPOTHESIS_NAME:
            raise ValueError("Step-2 controls expose only the named published phase candidate")

    @property
    def observations(self) -> tuple[Any, ...]:
        return self._observations

    @property
    def observation_ids(self) -> tuple[Any, ...]:
        return tuple(_as_identity(observation) for observation in self._observations)

    @property
    def frequency_counts(self) -> tuple[int, ...]:
        return tuple(int(np.asarray(observation.frequencies_hz).size) for observation in self._observations)

    @property
    def total_frequency_samples(self) -> int:
        return int(sum(self.frequency_counts))

    def estimate_kernel_evaluations(self, point_count: int) -> int:
        point_count = int(point_count)
        if point_count <= 0:
            raise ValueError("point_count must be positive")
        return int(point_count * self.total_frequency_samples)

    def estimate_cgls_kernel_evaluations(self, point_count: int, max_iterations: int) -> int:
        """Estimate one initial adjoint plus two operator calls per iteration."""

        if not np.isfinite(max_iterations) or int(max_iterations) <= 0 or int(max_iterations) != max_iterations:
            raise ValueError("max_iterations must be a positive finite integer")
        max_iterations = int(max_iterations)
        return int(self.estimate_kernel_evaluations(point_count) * (1 + 2 * max_iterations))

    def guard_cgls_evaluations(self, point_count: int, max_iterations: int) -> int:
        estimate = self.estimate_cgls_kernel_evaluations(point_count, max_iterations)
        if estimate > self.max_kernel_evaluations:
            raise RuntimeError(
                "CGLS total kernel-evaluation guard exceeded: "
                f"estimate={estimate} max={self.max_kernel_evaluations}"
            )
        return estimate

    def _guard_evaluations(self, point_count: int) -> None:
        estimate = self.estimate_kernel_evaluations(point_count)
        if estimate > self.max_kernel_evaluations:
            raise RuntimeError(
                f"kernel-evaluation guard exceeded: estimate={estimate} max={self.max_kernel_evaluations}"
            )

    @staticmethod
    def _points(points_xyz_m: Any) -> np.ndarray:
        points = np.asarray(points_xyz_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError("points_xyz_m must have shape [point, 3]")
        _finite(points, "points_xyz_m")
        return points

    @staticmethod
    def _kernel(points: np.ndarray, observation: Any) -> np.ndarray:
        position = np.asarray(observation.position_xyz_m, dtype=np.float64)
        frequencies = np.asarray(observation.frequencies_hz).astype(np.float64, copy=False)
        r0 = float(observation.r0_m)
        one_way_range = np.linalg.norm(points - position[None, :], axis=1)
        phase = (
            -1j
            * (4.0 * np.pi / SPEED_OF_LIGHT_M_S)
            * (one_way_range[:, None] - r0)
            * frequencies[None, :]
        )
        return np.exp(phase).astype(np.complex128, copy=False)

    def forward(self, points_xyz_m: Any, coefficients: Any) -> NativePredictions:
        points = self._points(points_xyz_m)
        coeffs = np.asarray(coefficients, dtype=np.complex128)
        if coeffs.shape != (points.shape[0],):
            raise ValueError("coefficients must have one complex value per point")
        _finite(coeffs.real, "coefficients.real")
        _finite(coeffs.imag, "coefficients.imag")
        self._guard_evaluations(points.shape[0])
        values: list[np.ndarray] = []
        for observation in self._observations:
            frequency_count = int(np.asarray(observation.frequencies_hz).size)
            prediction = np.zeros(frequency_count, dtype=np.complex128)
            for start in range(0, points.shape[0], self.point_chunk_size):
                stop = min(start + self.point_chunk_size, points.shape[0])
                kernel = self._kernel(points[start:stop], observation)
                prediction += np.einsum("pf,p->f", kernel, coeffs[start:stop], optimize=True)
            values.append(prediction)
        return NativePredictions(self.observation_ids, tuple(values))

    def _residual_values(self, residuals: NativePredictions | Sequence[Any]) -> tuple[np.ndarray, ...]:
        if isinstance(residuals, NativePredictions):
            if tuple(residuals.ids) != self.observation_ids:
                raise ValueError("prediction identities do not match native observation identities")
            values = residuals.values
        else:
            values = tuple(np.asarray(value, dtype=np.complex128) for value in residuals)
        if len(values) != len(self._observations):
            raise ValueError("residual count does not match native observations")
        validated: list[np.ndarray] = []
        for value, observation in zip(values, self._observations):
            expected = int(np.asarray(observation.frequencies_hz).size)
            if value.shape != (expected,):
                raise ValueError("ragged residual shape does not match native frequency count")
            _finite(value.real, "residual.real")
            _finite(value.imag, "residual.imag")
            validated.append(np.asarray(value, dtype=np.complex128))
        return tuple(validated)

    def adjoint(self, residuals: NativePredictions | Sequence[Any], points_xyz_m: Any) -> np.ndarray:
        points = self._points(points_xyz_m)
        residual_values = self._residual_values(residuals)
        self._guard_evaluations(points.shape[0])
        result = np.zeros(points.shape[0], dtype=np.complex128)
        for observation, residual in zip(self._observations, residual_values):
            for start in range(0, points.shape[0], self.point_chunk_size):
                stop = min(start + self.point_chunk_size, points.shape[0])
                kernel = self._kernel(points[start:stop], observation)
                result[start:stop] += np.einsum(
                    "pf,f->p", np.conjugate(kernel), residual, optimize=True
                )
        return _readonly(result)

    def data(self) -> NativePredictions:
        records = _validate_payload_shapes(self._observations)
        return NativePredictions(
            self.observation_ids,
            tuple(np.asarray(observation.response, dtype=np.complex128) for observation in records),
        )

    def backproject(
        self,
        points_xyz_m: Any,
        residuals: NativePredictions | Sequence[Any] | None = None,
        *,
        normalization: str = "none",
    ) -> BackprojectionResult:
        """Apply the direct adjoint primitive.

        Measured-data workflows must call :func:`conditional_backproject`,
        which binds the grid and kernel guard to an explicit support
        declaration.  This lower-level method remains available for small
        synthetic/direct-oracle fixtures.
        """

        if residuals is None:
            residuals = self.data()
        raw = np.asarray(self.adjoint(residuals, points_xyz_m), dtype=np.complex128)
        if normalization == "none":
            factor = 1.0
        elif normalization == "mean_native_sample":
            factor = 1.0 / float(self.total_frequency_samples)
        else:
            raise ValueError("normalization must be 'none' or 'mean_native_sample'")
        metadata = {
            "schema": "rift_gotcha_unweighted_direct_backprojection_v1",
            "phase_hypothesis": PHASE_HYPOTHESIS_NAME,
            "forward_formula": PHASE_HYPOTHESIS_FORWARD,
            "adjoint_formula": PHASE_HYPOTHESIS_ADJOINT,
            "normalization": normalization,
            "normalization_factor": factor,
            "amplitude_weighting": False,
            "frequency_policy": "native_ragged_exact",
            "operator": "direct_native_ragged_adjoint",
            "polarization": self.polarization,
            "geometry_precision": "float64",
            "accumulation_precision": "complex128",
            "legacy_bp_or_fft_reuse": False,
        }
        return BackprojectionResult(raw * factor, metadata)

    def dot_test(self, points_xyz_m: Any, coefficients: Any, residuals: Sequence[Any]) -> dict[str, Any]:
        forward = self.forward(points_xyz_m, coefficients)
        adjoint = self.adjoint(residuals, points_xyz_m)
        lhs = sum(np.vdot(prediction, residual) for prediction, residual in zip(forward.values, residuals))
        rhs = np.vdot(np.asarray(coefficients, dtype=np.complex128), adjoint)
        scale = max(1.0, abs(lhs), abs(rhs))
        error = float(abs(lhs - rhs) / scale)
        return {
            "lhs_real": float(lhs.real),
            "lhs_imag": float(lhs.imag),
            "rhs_real": float(rhs.real),
            "rhs_imag": float(rhs.imag),
            "relative_error": error,
            "passed": bool(error <= 2.0e-12),
        }


def _flatten(values: Sequence[np.ndarray]) -> np.ndarray:
    if not values:
        return np.empty(0, dtype=np.complex128)
    return np.concatenate([np.asarray(value, dtype=np.complex128) for value in values])


@dataclass(frozen=True)
class FrozenComplexScale:
    value: complex
    fit_role: str
    source_ids: tuple[Any, ...]
    frozen: bool = True
    polarization: str = ""

    def __post_init__(self) -> None:
        if self.fit_role != "train" or not self.frozen:
            raise ValueError("complex scale must be frozen after train-only fitting")
        if self.polarization not in _POLARIZATIONS:
            raise ValueError("complex scale must record one fitted polarization")
        if not np.isfinite(self.value.real) or not np.isfinite(self.value.imag):
            raise ValueError("complex scale must be finite")


def fit_train_complex_scale(
    observations: Sequence[Any], predictions: NativePredictions
) -> FrozenComplexScale:
    records = _validate_payload_shapes(observations)
    polarizations = {str(observation.identity.polarization).lower() for observation in records}
    if len(polarizations) != 1:
        raise ValueError("complex prediction scale must be fitted separately for each polarization")
    if any(str(observation.role).lower() != "train" for observation in records):
        raise ValueError("complex prediction scale must be fitted from train observations only")
    if tuple(predictions.ids) != tuple(_as_identity(observation) for observation in records):
        raise ValueError("scale-fit predictions do not match native observation identities")
    predicted = _flatten(predictions.values)
    target = _flatten([np.asarray(observation.response, dtype=np.complex128) for observation in records])
    denominator = float(np.vdot(predicted, predicted).real)
    if not np.isfinite(denominator) or denominator <= np.finfo(np.float64).tiny:
        raise ValueError("train-only complex scale is undefined for zero-energy predictions")
    value = np.vdot(predicted, target) / denominator
    return FrozenComplexScale(complex(value), "train", tuple(predictions.ids), True, next(iter(polarizations)))


def _metric_group(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    error = predicted - target
    target_energy = float(np.vdot(target, target).real)
    prediction_energy = float(np.vdot(predicted, predicted).real)
    error_energy = float(np.vdot(error, error).real)
    if target_energy > 0:
        relmse: float | None = float(error_energy / target_energy)
        energy_ratio: float | None = float(prediction_energy / target_energy)
        zero_reference: float | None = 1.0
    else:
        relmse = None
        energy_ratio = None
        zero_reference = None
    if target_energy > 0 and prediction_energy > 0:
        correlation = np.vdot(predicted, target) / np.sqrt(target_energy * prediction_energy)
        correlation_values: tuple[float | None, float | None, float | None] = (
            float(correlation.real),
            float(correlation.imag),
            float(abs(correlation)),
        )
    else:
        correlation_values = (None, None, None)
    return {
        "sample_count": int(target.size),
        "raw_complex_mse": float(error_energy / max(1, target.size)),
        "relmse": relmse,
        "zero_reference_relmse": zero_reference,
        "prediction_energy": prediction_energy,
        "target_energy": target_energy,
        "prediction_energy_over_target": energy_ratio,
        "complex_correlation_real": correlation_values[0],
        "complex_correlation_imag": correlation_values[1],
        "complex_correlation_abs": correlation_values[2],
    }


def evaluate_native_metrics(
    observations: Sequence[Any],
    predictions: NativePredictions,
    *,
    scale: FrozenComplexScale | None = None,
) -> dict[str, Any]:
    records = _validate_payload_shapes(observations)
    if tuple(predictions.ids) != tuple(_as_identity(observation) for observation in records):
        raise ValueError("metric predictions do not match native observation identities")
    if scale is not None:
        polarizations = {str(observation.identity.polarization).lower() for observation in records}
        if polarizations != {scale.polarization}:
            raise ValueError("frozen complex scale polarization does not match metric observations")
    target_values = tuple(np.asarray(observation.response, dtype=np.complex128) for observation in records)
    raw = _metric_group(_flatten(predictions.values), _flatten(target_values))
    result: dict[str, Any] = {"raw": raw, "phase_hypothesis": PHASE_HYPOTHESIS_NAME}
    if scale is not None:
        scaled = tuple(value * scale.value for value in predictions.values)
        result["scaled"] = _metric_group(_flatten(scaled), _flatten(target_values))
        result["scale"] = {
            "real": float(scale.value.real),
            "imag": float(scale.value.imag),
            "fit_role": scale.fit_role,
            "frozen": scale.frozen,
            "polarization": scale.polarization,
        }
    return result


@dataclass(frozen=True)
class NativeFrameH0Support:
    """Explicit conditional native-frame support declaration."""

    bounds_m: Mapping[str, tuple[float, float]]
    spacing_m: tuple[float, float, float]
    grid_shape: tuple[int, int, int]
    max_kernel_evaluations: int
    support_mode: str = "height_capable_volume"
    frame_contract: str = "antenna_xyz_unchanged"
    registration_status: str = "unresolved"
    support_status: str = SUPPORT_STATUS
    hypothesis_label: str = "analyst_selected_conditional_imaging_hypothesis"
    units: Mapping[str, str] = MappingProxyType(
        {"x": "m", "y": "m", "z": "m", "r0": "m", "frequency": "Hz"}
    )

    def __post_init__(self) -> None:
        """Validate and freeze direct instances as strictly as mappings."""

        if self.support_mode not in {"height_capable_volume", "plane_no_height"}:
            raise ValueError("support_mode must explicitly be height_capable_volume or plane_no_height")
        if not isinstance(self.bounds_m, Mapping) or set(self.bounds_m) != {"x", "y", "z"}:
            raise ValueError("support bounds_m must explicitly declare x, y, and z")
        canonical_bounds: dict[str, tuple[float, float]] = {}
        for axis in ("x", "y", "z"):
            value = tuple(float(item) for item in self.bounds_m[axis])
            if len(value) != 2 or not np.isfinite(value).all():
                raise ValueError(f"support {axis} bounds must be finite")
            if axis != "z" and not value[0] < value[1]:
                raise ValueError(f"support {axis} bounds must be finite and increasing")
            canonical_bounds[axis] = value
        if self.support_mode == "height_capable_volume" and not canonical_bounds["z"][0] < canonical_bounds["z"][1]:
            raise ValueError("height_capable_volume requires a nonzero z extent")
        if self.support_mode == "plane_no_height" and canonical_bounds["z"][0] != canonical_bounds["z"][1]:
            raise ValueError("plane_no_height requires a zero-thickness z plane")

        spacing = tuple(float(item) for item in self.spacing_m)
        raw_shape = tuple(self.grid_shape)
        if len(raw_shape) != 3 or any(not np.isfinite(item) or int(item) != item for item in raw_shape):
            raise ValueError("support shape must contain three integer layer counts")
        shape = tuple(int(item) for item in raw_shape)
        if len(spacing) != 3 or any(not np.isfinite(item) or item <= 0 for item in spacing):
            raise ValueError("support spacing_m must contain three positive metre spacings")
        if len(shape) != 3 or any(item <= 0 for item in shape):
            raise ValueError("support shape must contain three positive layer counts")
        expected_shape = tuple(
            int(round((canonical_bounds[axis][1] - canonical_bounds[axis][0]) / spacing[index])) + 1
            for index, axis in enumerate(("x", "y", "z"))
        )
        if shape != expected_shape:
            raise ValueError(f"support shape {shape} does not match bounds/spacing {expected_shape}")
        if self.support_mode == "height_capable_volume" and shape[2] < 2:
            raise ValueError("height-capable Step-2 support requires at least two z layers")
        if self.support_mode == "plane_no_height" and shape[2] != 1:
            raise ValueError("plane_no_height support must contain exactly one z layer")
        max_evaluations_value = float(self.max_kernel_evaluations)
        if not np.isfinite(max_evaluations_value) or int(max_evaluations_value) <= 0 or int(max_evaluations_value) != max_evaluations_value:
            raise ValueError("support declaration requires a positive integer max_kernel_evaluations")
        max_evaluations = int(max_evaluations_value)
        if self.frame_contract != "antenna_xyz_unchanged":
            raise ValueError("support frame must explicitly be native antenna xyz unchanged")
        if self.registration_status != "unresolved":
            raise ValueError("real support registration must remain unresolved in Step 2")
        if self.support_status != SUPPORT_STATUS:
            raise ValueError("support must be labeled a conditional imaging hypothesis")
        if not isinstance(self.units, Mapping) or any(
            self.units.get(key) != value
            for key, value in {"x": "m", "y": "m", "z": "m", "r0": "m", "frequency": "Hz"}.items()
        ):
            raise ValueError("support declaration must explicitly declare metre/Hz units")
        object.__setattr__(self, "bounds_m", MappingProxyType(canonical_bounds))
        object.__setattr__(self, "spacing_m", spacing)
        object.__setattr__(self, "grid_shape", shape)
        object.__setattr__(self, "max_kernel_evaluations", max_evaluations)
        object.__setattr__(self, "units", MappingProxyType({str(key): str(value) for key, value in self.units.items()}))

    @classmethod
    def from_mapping(cls, declaration: Mapping[str, Any]) -> "NativeFrameH0Support":
        if not isinstance(declaration, Mapping):
            raise ValueError("support declaration must be a mapping")
        if declaration.get("schema") != H0_SUPPORT_SCHEMA:
            raise ValueError(f"support schema must be {H0_SUPPORT_SCHEMA}")
        bounds_raw = declaration.get("bounds_m")
        if not isinstance(bounds_raw, Mapping) or set(bounds_raw) != {"x", "y", "z"}:
            raise ValueError("support bounds_m must explicitly declare x, y, and z")
        bounds: dict[str, tuple[float, float]] = {}
        for axis in ("x", "y", "z"):
            value = tuple(float(item) for item in bounds_raw[axis])
            if len(value) != 2 or not np.isfinite(value).all():
                raise ValueError(f"support {axis} bounds must be finite")
            if axis != "z" and not value[0] < value[1]:
                raise ValueError(f"support {axis} bounds must be finite and increasing")
            bounds[axis] = value
        support_mode = str(declaration.get("support_mode", ""))
        if support_mode not in {"height_capable_volume", "plane_no_height"}:
            raise ValueError("support_mode must explicitly be height_capable_volume or plane_no_height")
        if support_mode == "height_capable_volume" and not bounds["z"][0] < bounds["z"][1]:
            raise ValueError("height_capable_volume requires a nonzero z extent")
        if support_mode == "plane_no_height" and bounds["z"][0] != bounds["z"][1]:
            raise ValueError("plane_no_height requires a zero-thickness z plane")
        sampling = declaration.get("sampling")
        if not isinstance(sampling, Mapping):
            raise ValueError("support sampling declaration is required")
        spacing = tuple(float(item) for item in sampling.get("spacing_m", ()))
        shape = tuple(int(item) for item in sampling.get("shape", ()))
        if len(spacing) != 3 or any(not np.isfinite(item) or item <= 0 for item in spacing):
            raise ValueError("support spacing_m must contain three positive metre spacings")
        if len(shape) != 3 or any(item <= 0 for item in shape):
            raise ValueError("support shape must contain three positive layer counts")
        expected_shape = tuple(
            int(round((bounds[axis][1] - bounds[axis][0]) / spacing[index])) + 1
            for index, axis in enumerate(("x", "y", "z"))
        )
        if shape != expected_shape:
            raise ValueError(f"support shape {shape} does not match bounds/spacing {expected_shape}")
        if support_mode == "height_capable_volume" and shape[2] < 2:
            raise ValueError("height-capable Step-2 support requires at least two z layers")
        if support_mode == "plane_no_height" and shape[2] != 1:
            raise ValueError("plane_no_height support must contain exactly one z layer")
        units = declaration.get("units")
        if not isinstance(units, Mapping) or any(units.get(key) != value for key, value in {"x": "m", "y": "m", "z": "m", "r0": "m", "frequency": "Hz"}.items()):
            raise ValueError("support declaration must explicitly declare metre/Hz units")
        max_evaluations_raw = declaration.get("max_kernel_evaluations", 0)
        if not np.isfinite(max_evaluations_raw) or int(max_evaluations_raw) <= 0 or int(max_evaluations_raw) != max_evaluations_raw:
            raise ValueError("support declaration requires a positive integer max_kernel_evaluations")
        max_evaluations = int(max_evaluations_raw)
        if declaration.get("frame_contract") != "antenna_xyz_unchanged":
            raise ValueError("support frame must explicitly be native antenna xyz unchanged")
        if declaration.get("registration_status") != "unresolved":
            raise ValueError("real support registration must remain unresolved in Step 2")
        if declaration.get("support_status") != SUPPORT_STATUS:
            raise ValueError("support must be labeled a conditional imaging hypothesis")
        if declaration.get("phase_hypothesis") != PHASE_HYPOTHESIS_NAME:
            raise ValueError("support must name the published phase candidate")
        if declaration.get("autofocus_status") != "raw_unapplied_channel_owned":
            raise ValueError("support must declare raw/unapplied channel-owned autofocus")
        return cls(
            bounds_m=MappingProxyType(bounds),
            spacing_m=spacing,
            grid_shape=shape,
            max_kernel_evaluations=max_evaluations,
            support_mode=support_mode,
            frame_contract="antenna_xyz_unchanged",
            registration_status="unresolved",
            support_status=SUPPORT_STATUS,
            hypothesis_label=str(declaration.get("hypothesis_label", cls.hypothesis_label)),
            units=MappingProxyType({str(key): str(value) for key, value in units.items()}),
        )

    @property
    def point_count(self) -> int:
        return int(np.prod(self.grid_shape, dtype=np.int64))

    def corners(self) -> np.ndarray:
        return np.asarray(
            [
                [x, y, z]
                for x in self.bounds_m["x"]
                for y in self.bounds_m["y"]
                for z in self.bounds_m["z"]
            ],
            dtype=np.float64,
        )

    def grid_points(self) -> np.ndarray:
        """Return the declared x/y/z lattice in deterministic native-frame order."""

        axes = [
            np.linspace(self.bounds_m[axis][0], self.bounds_m[axis][1], self.grid_shape[index], dtype=np.float64)
            for index, axis in enumerate(("x", "y", "z"))
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.column_stack([axis.reshape(-1) for axis in mesh])

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": H0_SUPPORT_SCHEMA,
            "bounds_m": {axis: list(self.bounds_m[axis]) for axis in ("x", "y", "z")},
            "sampling": {"spacing_m": list(self.spacing_m), "shape": list(self.grid_shape)},
            "max_kernel_evaluations": self.max_kernel_evaluations,
            "support_mode": self.support_mode,
            "frame_contract": self.frame_contract,
            "registration_status": self.registration_status,
            "support_status": self.support_status,
            "hypothesis_label": self.hypothesis_label,
            "units": dict(self.units),
            "phase_hypothesis": PHASE_HYPOTHESIS_NAME,
            "autofocus_status": "raw_unapplied_channel_owned",
        }


def _support_observation_report(observations: Sequence[Any]) -> dict[str, Any]:
    records = _validate_payload_shapes(observations)
    positions = np.asarray([observation.position_xyz_m for observation in records], dtype=np.float64)
    r0 = np.asarray([float(observation.r0_m) for observation in records], dtype=np.float64)
    deltas = np.asarray(
        [
            np.diff(np.asarray(observation.frequencies_hz, dtype=np.float64))
            for observation in records
            if np.asarray(observation.frequencies_hz).size > 1
        ],
        dtype=object,
    )
    spacing_values = (
        np.concatenate([value for value in deltas if value.size]).astype(np.float64, copy=False)
        if deltas.size
        else np.empty(0, dtype=np.float64)
    )
    autofocus_modes = sorted({str(observation.autofocus.mode) for observation in records})
    return {
        "observation_ids": [observation.identity.as_dict() for observation in records],
        "roles": {role: sum(str(observation.role).lower() == role for observation in records) for role in ("train", "validation")},
        "pass_ids": sorted({int(observation.identity.pass_id) for observation in records}),
        "polarizations": sorted({str(observation.identity.polarization).lower() for observation in records}),
        "units": {"xyz": "m", "r0": "m", "frequency": "Hz", "phase": "rad"},
        "position_finite": bool(np.isfinite(positions).all()),
        "r0_finite": bool(np.isfinite(r0).all()),
        "r0_minus_position_norm_m": {
            "minimum": float(np.min(r0 - np.linalg.norm(positions, axis=1))),
            "maximum": float(np.max(r0 - np.linalg.norm(positions, axis=1))),
            "mean": float(np.mean(r0 - np.linalg.norm(positions, axis=1))),
        },
        "response_shape_order": {
            "order": "[frequency] per observation",
            "shapes": [list(np.asarray(observation.response).shape) for observation in records],
            "native_frequency_count_range": [
                int(min(np.asarray(observation.frequencies_hz).size for observation in records)),
                int(max(np.asarray(observation.frequencies_hz).size for observation in records)),
            ],
        },
        "native_frequency_spacing_hz": {
            "minimum": float(np.min(spacing_values)) if spacing_values.size else None,
            "maximum": float(np.max(spacing_values)) if spacing_values.size else None,
            "median": float(np.median(spacing_values)) if spacing_values.size else None,
            "uniform_grid_assumed": False,
        },
        "autofocus": {"modes": autofocus_modes, "all_unapplied": all(not bool(observation.autofocus.applied) for observation in records)},
    }


def preflight_support_declaration(
    declaration: Mapping[str, Any] | NativeFrameH0Support | None,
    observations: Sequence[Any],
    *,
    point_count: int | None = None,
) -> dict[str, Any]:
    """Validate explicit native-frame support and estimate bounded work."""

    if declaration is None:
        raise ValueError("real-data support preflight requires an explicit declaration")
    support = declaration if isinstance(declaration, NativeFrameH0Support) else NativeFrameH0Support.from_mapping(declaration)
    records = _validate_payload_shapes(observations)
    requested_point_count = support.point_count if point_count is None else int(point_count)
    if requested_point_count <= 0:
        raise ValueError("point_count must be positive")
    estimate = int(
        requested_point_count
        * sum(int(np.asarray(observation.frequencies_hz).size) for observation in records)
    )
    if estimate > support.max_kernel_evaluations:
        raise RuntimeError(
            f"support preflight exceeds hard kernel-evaluation guard: estimate={estimate} max={support.max_kernel_evaluations}"
        )
    positions = np.asarray([observation.position_xyz_m for observation in records], dtype=np.float64)
    r0 = np.asarray([float(observation.r0_m) for observation in records], dtype=np.float64)
    corners = support.corners()
    lower_deltas = []
    upper_deltas = []
    for position, value in zip(positions, r0):
        nearest = np.clip(
            position,
            np.asarray([support.bounds_m[axis][0] for axis in ("x", "y", "z")]),
            np.asarray([support.bounds_m[axis][1] for axis in ("x", "y", "z")]),
        )
        lower_deltas.append(float(np.linalg.norm(nearest - position) - value))
        upper_deltas.append(float(np.max(np.linalg.norm(corners - position[None, :], axis=1) - value)))
    spacing_report = _support_observation_report(records)
    spacing = spacing_report["native_frequency_spacing_hz"]
    ambiguity = {
        name: (SPEED_OF_LIGHT_M_S / (2.0 * spacing[name])) if spacing[name] else None
        for name in ("minimum", "maximum", "median")
    }
    return {
        "schema": "rift_gotcha_step2_support_preflight_v1",
        "support": support.as_dict(),
        "status": SUPPORT_STATUS,
        "support_mode": support.support_mode,
        "height_capable": support.support_mode == "height_capable_volume",
        "plane_only_no_height_claim": support.support_mode == "plane_no_height",
        "physical_ground_registration": "unresolved",
        "full_scene_support_claim": False,
        "signed_R_minus_r0_span_m": [float(np.min(lower_deltas)), float(np.max(upper_deltas))],
        "support_boundary_diagnostic_status": "conditional_box_diagnostic_not_support_proof",
        "nominal_one_way_ambiguity_m": ambiguity,
        "phase_hypothesis": {
            "name": PHASE_HYPOTHESIS_NAME,
            "forward": PHASE_HYPOTHESIS_FORWARD,
            "adjoint": PHASE_HYPOTHESIS_ADJOINT,
            "status": PHASE_HYPOTHESIS_STATUS,
        },
        "resource_estimate": {
            "point_count": requested_point_count,
            "total_native_frequency_samples": int(sum(np.asarray(observation.frequencies_hz).size for observation in records)),
            "kernel_evaluations": estimate,
            "max_kernel_evaluations": support.max_kernel_evaluations,
            "guard_passed": True,
        },
        "observations": spacing_report,
    }


@dataclass(frozen=True)
class CGLSResult:
    coefficients: np.ndarray
    status: str
    iterations: tuple[Mapping[str, Any], ...]
    lambda_ridge: float
    max_iterations: int
    rtol: float
    kernel_evaluations_estimate: int = 0
    kernel_evaluations_guarded: bool = True
    support_preflight: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficients", _readonly(self.coefficients, dtype=np.complex128))
        object.__setattr__(self, "iterations", tuple(MappingProxyType(dict(row)) for row in self.iterations))
        if self.support_preflight is not None:
            object.__setattr__(self, "support_preflight", _freeze_mapping(dict(self.support_preflight)))

    @property
    def converged(self) -> bool:
        return self.status == "converged"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "lambda_ridge": self.lambda_ridge,
            "max_iterations": self.max_iterations,
            "rtol": self.rtol,
            "kernel_evaluations_estimate": self.kernel_evaluations_estimate,
            "kernel_evaluations_guarded": self.kernel_evaluations_guarded,
            "support_preflight": None if self.support_preflight is None else dict(self.support_preflight),
            "coefficient_count": int(self.coefficients.size),
            "iterations": [dict(row) for row in self.iterations],
        }


def _residual_norm(values: Sequence[np.ndarray]) -> float:
    return float(np.sqrt(sum(float(np.vdot(value, value).real) for value in values)))


def ridge_cgls(
    operator: NativeRaggedOperator,
    points_xyz_m: Any,
    target: NativePredictions,
    *,
    lambda_ridge: float = 1.0e-3,
    max_iterations: int = 20,
    rtol: float = 1.0e-8,
) -> CGLSResult:
    """Solve ``(AᴴA + lambda I)x=Aᴴb`` without materializing ``A``.

    The point coordinates are explicit because the native operator is a
    matrix-free function of both the observations and the candidate support.
    Measured-data workflows must call :func:`conditional_ridge_cgls`, which
    binds the point set and total-work guard to an explicit support
    declaration.  This primitive remains available for synthetic/direct-
    oracle fixtures.
    """

    return ridge_cgls_on_points(
        operator,
        points_xyz_m,
        target,
        lambda_ridge=lambda_ridge,
        max_iterations=max_iterations,
        rtol=rtol,
    )


def ridge_cgls_on_points(
    operator: NativeRaggedOperator,
    points_xyz_m: Any,
    target: NativePredictions,
    *,
    lambda_ridge: float = 1.0e-3,
    max_iterations: int = 20,
    rtol: float = 1.0e-8,
) -> CGLSResult:
    """Matrix-free ridge CGLS over an explicit bounded point set."""

    if any(str(observation.role).lower() != "train" for observation in operator.observations):
        raise ValueError("inverse-fit APIs require train observations only")
    points = operator._points(points_xyz_m)
    point_count = points.shape[0]
    if not np.isfinite(lambda_ridge) or lambda_ridge <= 0:
        raise ValueError("lambda_ridge must be positive and finite")
    if not np.isfinite(max_iterations) or int(max_iterations) <= 0 or int(max_iterations) != max_iterations:
        raise ValueError("max_iterations must be a positive finite integer")
    if not np.isfinite(rtol) or not 0 < rtol < 1:
        raise ValueError("rtol must be finite and in (0,1)")
    if not isinstance(target, NativePredictions):
        raise TypeError("inverse-fit targets must be NativePredictions with canonical IDs")
    b = operator._residual_values(target)
    total_kernel_evaluations = operator.guard_cgls_evaluations(point_count, max_iterations)
    x = np.zeros(point_count, dtype=np.complex128)
    residual = tuple(np.array(value, dtype=np.complex128, copy=True) for value in b)
    gradient = np.asarray(operator.adjoint(residual, points), dtype=np.complex128)
    direction = gradient.copy()
    gamma = float(np.vdot(gradient, gradient).real)
    initial_normal = float(np.sqrt(max(gamma, 0.0)))
    rows: list[dict[str, Any]] = []

    def record(iteration: int, status: str) -> None:
        rows.append(
            {
                "iteration": int(iteration),
                "data_residual_norm": _residual_norm(residual),
                "normal_residual_norm": float(np.linalg.norm(gradient)),
                "objective": float(0.5 * _residual_norm(residual) ** 2 + 0.5 * lambda_ridge * np.vdot(x, x).real),
                "status": status,
            }
        )

    if initial_normal == 0.0:
        record(0, "converged")
        return CGLSResult(
            x,
            "converged",
            tuple(rows),
            float(lambda_ridge),
            int(max_iterations),
            float(rtol),
            total_kernel_evaluations,
            True,
        )

    status = "max_iterations"
    for iteration in range(1, int(max_iterations) + 1):
        forward_direction = operator.forward(points, direction)
        normal_direction = np.asarray(operator.adjoint(forward_direction, points), dtype=np.complex128) + lambda_ridge * direction
        denominator = float(np.vdot(direction, normal_direction).real)
        if not np.isfinite(denominator) or denominator <= np.finfo(np.float64).tiny:
            record(iteration, "breakdown")
            status = "breakdown"
            break
        alpha = gamma / denominator
        x += alpha * direction
        residual = tuple(value - alpha * direction_value for value, direction_value in zip(residual, forward_direction.values))
        gradient = gradient - alpha * normal_direction
        gamma_new = float(np.vdot(gradient, gradient).real)
        if not np.isfinite(gamma_new):
            record(iteration, "breakdown")
            status = "breakdown"
            break
        normal_norm = float(np.sqrt(max(gamma_new, 0.0)))
        if normal_norm <= initial_normal * rtol:
            record(iteration, "converged")
            status = "converged"
            break
        beta = gamma_new / max(gamma, np.finfo(np.float64).tiny)
        direction = gradient + beta * direction
        gamma = gamma_new
        record(iteration, "running" if iteration < int(max_iterations) else "max_iterations")
    return CGLSResult(
        x,
        status,
        tuple(rows),
        float(lambda_ridge),
        int(max_iterations),
        float(rtol),
        total_kernel_evaluations,
        True,
    )


def conditional_backproject(
    declaration: Mapping[str, Any] | NativeFrameH0Support,
    observations: Sequence[Any],
    *,
    residuals: NativePredictions | Sequence[Any] | None = None,
    normalization: str = "none",
    point_chunk_size: int = 4096,
) -> BackprojectionResult:
    """Backproject only on an explicitly declared native-frame support grid."""

    if residuals is not None and not isinstance(residuals, NativePredictions):
        raise TypeError("conditional backprojection residuals must be NativePredictions with canonical IDs")
    if not np.isfinite(point_chunk_size) or int(point_chunk_size) <= 0 or int(point_chunk_size) != point_chunk_size:
        raise ValueError("point_chunk_size must be a positive finite integer")
    point_chunk_size = int(point_chunk_size)
    support = declaration if isinstance(declaration, NativeFrameH0Support) else NativeFrameH0Support.from_mapping(declaration)
    preflight = preflight_support_declaration(support, observations, point_count=support.point_count)
    preflight["resource_estimate"]["point_chunk_size"] = point_chunk_size
    preflight["resource_estimate"]["point_chunks_per_observation"] = int(
        (support.point_count + point_chunk_size - 1) // point_chunk_size
    )
    operator = NativeRaggedOperator(
        observations,
        point_chunk_size=point_chunk_size,
        max_kernel_evaluations=support.max_kernel_evaluations,
    )
    result = operator.backproject(support.grid_points(), residuals, normalization=normalization)
    metadata = dict(result.metadata)
    metadata["support_preflight"] = preflight
    metadata["support_point_count_bound"] = support.point_count
    metadata["point_chunk_size"] = point_chunk_size
    metadata["point_chunks_per_observation"] = int(
        (support.point_count + point_chunk_size - 1) // point_chunk_size
    )
    return BackprojectionResult(result.values, metadata)


def conditional_ridge_cgls(
    declaration: Mapping[str, Any] | NativeFrameH0Support,
    observations: Sequence[Any],
    target: NativePredictions,
    *,
    lambda_ridge: float = 1.0e-3,
    max_iterations: int = 20,
    rtol: float = 1.0e-8,
) -> CGLSResult:
    """Solve ridge CGLS on the declared support grid with its hard guard."""

    if not isinstance(target, NativePredictions):
        raise TypeError("conditional inverse targets must be NativePredictions with canonical IDs")
    support = declaration if isinstance(declaration, NativeFrameH0Support) else NativeFrameH0Support.from_mapping(declaration)
    train_observations = _require_train_observations(observations)
    preflight = preflight_support_declaration(support, train_observations, point_count=support.point_count)
    operator = NativeRaggedOperator(train_observations, max_kernel_evaluations=support.max_kernel_evaluations)
    result = ridge_cgls(
        operator,
        support.grid_points(),
        target,
        lambda_ridge=lambda_ridge,
        max_iterations=max_iterations,
        rtol=rtol,
    )
    return CGLSResult(
        result.coefficients,
        result.status,
        result.iterations,
        result.lambda_ridge,
        result.max_iterations,
        result.rtol,
        result.kernel_evaluations_estimate,
        result.kernel_evaluations_guarded,
        preflight,
    )


__all__ = [
    "BackprojectionResult",
    "CGLSResult",
    "DEFAULT_PREFIX_COUNTS",
    "FrozenComplexScale",
    "H0_SUPPORT_SCHEMA",
    "NativeFrameH0Support",
    "NativePredictions",
    "NativeRaggedOperator",
    "NestedTrainingPrefix",
    "PHASE_HYPOTHESIS_ADJOINT",
    "PHASE_HYPOTHESIS_FORWARD",
    "PHASE_HYPOTHESIS_NAME",
    "PHASE_HYPOTHESIS_STATUS",
    "SUPPORT_STATUS",
    "conditional_backproject",
    "conditional_ridge_cgls",
    "evaluate_native_metrics",
    "fit_train_complex_scale",
    "preflight_support_declaration",
    "ridge_cgls",
    "ridge_cgls_on_points",
    "select_train_prefixes",
    "validate_nested_prefixes",
]
