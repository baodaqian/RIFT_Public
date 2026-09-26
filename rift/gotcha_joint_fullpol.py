"""Fail-closed contracts for the GOTCHA Joint-8 full-polarization experiment.

This module is intentionally independent of the existing single-pass PublicRadar
trainers.  It defines the immutable pieces that those trainers did not need:

* one sector split shared by all eight passes and all four polarizations;
* one 32-shard manifest contract;
* the fixed, cell-centred 1776 x 1776 planar support;
* four independent complex degree-3 response heads on that shared support;
* coherent and matched range-power metric reductions; and
* the serialized calibration/nuisance contract.

The support convention deserves special emphasis.  ``[-50, +50] m`` are the
*cell edges* of a 100 m domain and the 1776 samples are cell centres.  The
compatible pitch is therefore ``100 / 1776 m``.  It is neither an
endpoint-inclusive ``100 / 1775 m`` grid nor a ``1 / 32 m`` grid.  A true
1/32 m sampling of the same closed interval would require 3201 x 3201 points
and is a different experiment.

The pure NumPy contracts remain importable when Torch is unavailable.  The
small Torch scene class is defined when Torch is installed and otherwise
raises a clear error only when instantiated.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

try:  # Keep manifest/calibration/metric validation usable in CPU-only tools.
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised only without Torch.
    torch = None
    nn = None


SCENE_ID = "gotcha_v1_joint8_fullpol"
MANAGER_TRACK_ID = "rift_publicradar_gotcha_joint_v1"
SCENE_COUNT = 1

PASS_IDS = tuple(range(1, 9))
POLARIZATIONS = ("hh", "hv", "vh", "vv")
CO_POLARIZATIONS = ("hh", "vv")
CROSS_POLARIZATIONS = ("hv", "vh")
SECTOR_IDS = tuple(range(1, 361))
SECTOR_COUNT = len(SECTOR_IDS)
SHARD_IDS = tuple(
    f"pass{pass_id}_{polarization}"
    for pass_id in PASS_IDS
    for polarization in POLARIZATIONS
)
SHARD_COUNT = len(SHARD_IDS)
SOURCE_FILE_COUNT = SHARD_COUNT * SECTOR_COUNT

SPLIT_SEED = 42
TRAIN_SECTOR_COUNT = 288
VALIDATION_SECTOR_COUNT = 36
TEST_SECTOR_COUNT = 36
SEALED_TEST_SECTOR_IDS = tuple(
    sector_id for sector_id in SECTOR_IDS if (sector_id - 1) % 10 == 5
)
PAYLOAD_AUDITED_SECTOR_IDS = tuple(
    sector_id for sector_id in SECTOR_IDS if sector_id not in SEALED_TEST_SECTOR_IDS
)
PAYLOAD_AUDITED_SECTOR_COUNT = len(PAYLOAD_AUDITED_SECTOR_IDS)
SEALED_TEST_SECTOR_COUNT = len(SEALED_TEST_SECTOR_IDS)
INVENTORIED_SOURCE_FILE_COUNT = SOURCE_FILE_COUNT
PAYLOAD_AUDITED_SOURCE_FILE_COUNT = SHARD_COUNT * PAYLOAD_AUDITED_SECTOR_COUNT
SEALED_TEST_SOURCE_FILE_COUNT = SHARD_COUNT * SEALED_TEST_SECTOR_COUNT
SPLIT_STRATEGY = "gotcha_sector_interleaved_8train_1validation_1test_v1"
_ROLE_ORDER = ("train", "validation", "test")
_ROLE_SEED_STRIDE = 0x9E3779B1

MANIFEST_SCHEMA = "rift_gotcha_joint_fullpol_manifest_v1"
CALIBRATION_SCHEMA = "rift_gotcha_joint_fullpol_calibration_v1"

SUPPORT_NX = 1776
SUPPORT_NY = 1776
SUPPORT_NZ = 1
SUPPORT_POINT_COUNT = SUPPORT_NX * SUPPORT_NY * SUPPORT_NZ
SUPPORT_MIN_M = -50.0
SUPPORT_MAX_M = 50.0
SUPPORT_Z_M = 0.0
SUPPORT_PITCH_M = (SUPPORT_MAX_M - SUPPORT_MIN_M) / SUPPORT_NX
SUPPORT_FIRST_CENTER_M = SUPPORT_MIN_M + 0.5 * SUPPORT_PITCH_M
SUPPORT_LAST_CENTER_M = SUPPORT_MAX_M - 0.5 * SUPPORT_PITCH_M
INCORRECT_ONE_OVER_32_PITCH_M = 1.0 / 32.0
SUPPORT_SAMPLING_CONVENTION = "cell_centres_between_fixed_domain_edges_v1"

SH_DEGREE = 3
SH_BASIS_COUNT = (SH_DEGREE + 1) ** 2

# A calibration artifact must select and freeze a tighter bound no larger than
# this fail-closed safety ceiling.  One metre is only 1% of the 100 m support;
# it prevents an unconstrained nuisance from becoming a hidden scene shift.
MAX_RANGE_OFFSET_BOUND_M = 1.0

AUTOFOCUS_APPLICATION_ORDER = ("range", "phase")
AUTOFOCUS_RANGE_FORMULA = (
    "range_after_af_m = range_before_af_m + range_sign * af.r_correct_m"
)
AUTOFOCUS_PHASE_FORMULA = (
    "signal_after_af = signal_before_af * exp(1j * phase_sign * af.ph_correct_rad)"
)
AUTOFOCUS_HELDOUT_METRIC = "coherent_relative_mse"


def canonical_shard_id(pass_id: int, polarization: str) -> str:
    """Return the canonical identifier for one pass/polarization stratum."""

    pass_id = int(pass_id)
    polarization = str(polarization).lower()
    if pass_id not in PASS_IDS:
        raise ValueError(f"pass_id must be one of {PASS_IDS}")
    if polarization not in POLARIZATIONS:
        raise ValueError(f"polarization must be one of {POLARIZATIONS}")
    return f"pass{pass_id}_{polarization}"


@dataclass(frozen=True)
class SectorSplit:
    """The one immutable sector split propagated to every data shard.

    Role membership is set by angular slot, while ``seed`` deterministically
    shuffles sectors *within* each role.  The seed therefore cannot move a
    sector between training, validation, and test.
    """

    seed: int
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    role_by_sector: tuple[str, ...]

    def sectors(self, role: str) -> tuple[int, ...]:
        if role not in _ROLE_ORDER:
            raise ValueError(f"role must be one of {_ROLE_ORDER}")
        return getattr(self, role)

    def role_for(self, sector_id: int) -> str:
        sector_id = int(sector_id)
        if sector_id not in SECTOR_IDS:
            raise ValueError("sector_id must be in 1..360")
        return self.role_by_sector[sector_id - 1]

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": SPLIT_STRATEGY,
            "seed": self.seed,
            "sector_ids": list(SECTOR_IDS),
            "role_by_sector": list(self.role_by_sector),
            "sectors": {
                role: list(self.sectors(role)) for role in _ROLE_ORDER
            },
            "counts": {
                role: len(self.sectors(role)) for role in _ROLE_ORDER
            },
        }


def build_sector_split(seed: int = SPLIT_SEED) -> SectorSplit:
    """Build the sealed 288/36/36 split shared by all 32 shards."""

    seed = int(seed)
    if seed != SPLIT_SEED:
        raise ValueError(f"Joint-8 v1 seals split seed {SPLIT_SEED}, got {seed}")

    role_members = {role: [] for role in _ROLE_ORDER}
    role_by_sector = []
    for slot, sector_id in enumerate(SECTOR_IDS):
        within_decade = slot % 10
        if within_decade == 0:
            role = "validation"
        elif within_decade == 5:
            role = "test"
        else:
            role = "train"
        role_members[role].append(sector_id)
        role_by_sector.append(role)

    shuffled: dict[str, tuple[int, ...]] = {}
    for role_index, role in enumerate(_ROLE_ORDER):
        rng = np.random.default_rng(seed + role_index * _ROLE_SEED_STRIDE)
        shuffled[role] = tuple(
            int(value) for value in rng.permutation(role_members[role])
        )

    split = SectorSplit(
        seed=seed,
        train=shuffled["train"],
        validation=shuffled["validation"],
        test=shuffled["test"],
        role_by_sector=tuple(role_by_sector),
    )
    _validate_sector_split(split)
    return split


def _validate_sector_split(split: SectorSplit) -> None:
    expected_counts = {
        "train": TRAIN_SECTOR_COUNT,
        "validation": VALIDATION_SECTOR_COUNT,
        "test": TEST_SECTOR_COUNT,
    }
    combined = []
    for role in _ROLE_ORDER:
        sectors = split.sectors(role)
        if len(sectors) != expected_counts[role]:
            raise ValueError(f"{role} must contain {expected_counts[role]} sectors")
        if len(set(sectors)) != len(sectors):
            raise ValueError(f"{role} repeats a sector")
        combined.extend(sectors)
        for sector_id in sectors:
            if split.role_for(sector_id) != role:
                raise ValueError(f"sector {sector_id} role mapping is inconsistent")
    if set(combined) != set(SECTOR_IDS) or len(combined) != SECTOR_COUNT:
        raise ValueError("sector roles are not a complete disjoint partition")


def support_contract() -> dict[str, Any]:
    """Return the exact fixed-support description embedded in every artifact."""

    return {
        "shape": [SUPPORT_NX, SUPPORT_NY, SUPPORT_NZ],
        "point_count": SUPPORT_POINT_COUNT,
        "domain_edge_bounds_m": {
            "x": [SUPPORT_MIN_M, SUPPORT_MAX_M],
            "y": [SUPPORT_MIN_M, SUPPORT_MAX_M],
        },
        "z_m": SUPPORT_Z_M,
        "sampling_convention": SUPPORT_SAMPLING_CONVENTION,
        "pitch_xy_m": [SUPPORT_PITCH_M, SUPPORT_PITCH_M],
        "first_center_xy_m": [SUPPORT_FIRST_CENTER_M, SUPPORT_FIRST_CENTER_M],
        "last_center_xy_m": [SUPPORT_LAST_CENTER_M, SUPPORT_LAST_CENTER_M],
        "explicitly_not_one_over_32_m": True,
    }


def _require_sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    return value


def _require_exact_float(actual: Any, expected: float, name: str) -> None:
    try:
        actual_float = float(actual)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(actual_float) or not math.isclose(
        actual_float, float(expected), rel_tol=0.0, abs_tol=1.0e-14
    ):
        raise ValueError(f"{name} must equal {expected!r}")


def validate_support_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact cell-centred 1776-square support contract."""

    if not isinstance(contract, Mapping):
        raise ValueError("support must be a mapping")
    expected = support_contract()
    if list(_require_sequence(contract.get("shape"), "support.shape")) != expected["shape"]:
        raise ValueError("support.shape must be [1776, 1776, 1]")
    if int(contract.get("point_count", -1)) != SUPPORT_POINT_COUNT:
        raise ValueError("support.point_count is not 1776 x 1776")
    bounds = contract.get("domain_edge_bounds_m")
    if not isinstance(bounds, Mapping):
        raise ValueError("support.domain_edge_bounds_m must be a mapping")
    for axis in ("x", "y"):
        values = list(_require_sequence(bounds.get(axis), f"support bounds {axis}"))
        if len(values) != 2:
            raise ValueError(f"support bounds {axis} must contain two values")
        _require_exact_float(values[0], SUPPORT_MIN_M, f"support {axis} minimum")
        _require_exact_float(values[1], SUPPORT_MAX_M, f"support {axis} maximum")
    _require_exact_float(contract.get("z_m"), SUPPORT_Z_M, "support.z_m")
    if contract.get("sampling_convention") != SUPPORT_SAMPLING_CONVENTION:
        raise ValueError("support uses the wrong sampling convention")
    for key, expected_value in (
        ("pitch_xy_m", SUPPORT_PITCH_M),
        ("first_center_xy_m", SUPPORT_FIRST_CENTER_M),
        ("last_center_xy_m", SUPPORT_LAST_CENTER_M),
    ):
        values = list(_require_sequence(contract.get(key), f"support.{key}"))
        if len(values) != 2:
            raise ValueError(f"support.{key} must contain two values")
        for axis, value in zip(("x", "y"), values):
            _require_exact_float(value, expected_value, f"support.{key}.{axis}")
    if contract.get("explicitly_not_one_over_32_m") is not True:
        raise ValueError("support must explicitly reject the 1/32 m documentation error")
    if math.isclose(SUPPORT_PITCH_M, INCORRECT_ONE_OVER_32_PITCH_M):
        raise AssertionError("the fixed support unexpectedly became a 1/32 m grid")
    return expected


def _validate_serialized_split(serialized: Mapping[str, Any]) -> SectorSplit:
    if not isinstance(serialized, Mapping):
        raise ValueError("manifest.split must be a mapping")
    expected = build_sector_split()
    if serialized.get("strategy") != SPLIT_STRATEGY:
        raise ValueError("manifest split strategy is incorrect")
    if int(serialized.get("seed", -1)) != SPLIT_SEED:
        raise ValueError("manifest split seed is incorrect")
    if list(_require_sequence(serialized.get("sector_ids"), "split.sector_ids")) != list(SECTOR_IDS):
        raise ValueError("manifest split must enumerate sectors 1..360")
    if list(_require_sequence(serialized.get("role_by_sector"), "split.role_by_sector")) != list(expected.role_by_sector):
        raise ValueError("manifest split role_by_sector is not the sealed interleave")
    sectors = serialized.get("sectors")
    if not isinstance(sectors, Mapping):
        raise ValueError("manifest split sectors must be a mapping")
    for role in _ROLE_ORDER:
        if list(_require_sequence(sectors.get(role), f"split.sectors.{role}")) != list(expected.sectors(role)):
            raise ValueError(f"manifest split {role} ordering is not seed-42 exact")
    counts = serialized.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("manifest split counts must be a mapping")
    for role in _ROLE_ORDER:
        if int(counts.get(role, -1)) != len(expected.sectors(role)):
            raise ValueError(f"manifest split {role} count is incorrect")
    return expected


def validate_joint_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an exact 32-shard, sector-aligned Joint-8 manifest.

    Each shard must inventory one unique source file for every sector 1..360,
    payload-audit exactly the 324 train/validation sectors, and leave the exact
    36 test sectors sealed.  Repeating the canonical role and seal vectors is
    intentional: it lets a shard be rejected in isolation if any materializer
    accidentally moves one pass or polarization into a different role.
    """

    if not isinstance(manifest, Mapping):
        raise ValueError("joint manifest must be a mapping")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema must be {MANIFEST_SCHEMA}")
    if manifest.get("scene_id") != SCENE_ID:
        raise ValueError(f"manifest scene_id must be {SCENE_ID}")
    if manifest.get("manager_track_id") != MANAGER_TRACK_ID:
        raise ValueError(f"manifest manager_track_id must be {MANAGER_TRACK_ID}")
    if int(manifest.get("scene_count", -1)) != SCENE_COUNT:
        raise ValueError(f"manifest scene_count must be {SCENE_COUNT}")
    if list(_require_sequence(manifest.get("passes"), "manifest.passes")) != list(PASS_IDS):
        raise ValueError("manifest passes must be exactly 1..8")
    if list(_require_sequence(manifest.get("polarizations"), "manifest.polarizations")) != list(POLARIZATIONS):
        raise ValueError("manifest polarizations have the wrong order or membership")
    if int(manifest.get("source_file_count", -1)) != SOURCE_FILE_COUNT:
        raise ValueError(f"manifest source_file_count must be {SOURCE_FILE_COUNT}")
    if (
        int(manifest.get("inventoried_source_file_count", -1))
        != INVENTORIED_SOURCE_FILE_COUNT
    ):
        raise ValueError(
            "manifest inventoried_source_file_count must be "
            f"{INVENTORIED_SOURCE_FILE_COUNT}"
        )
    if (
        int(manifest.get("payload_audited_source_file_count", -1))
        != PAYLOAD_AUDITED_SOURCE_FILE_COUNT
    ):
        raise ValueError(
            "manifest payload_audited_source_file_count must be "
            f"{PAYLOAD_AUDITED_SOURCE_FILE_COUNT}"
        )
    if (
        int(manifest.get("sealed_test_source_file_count", -1))
        != SEALED_TEST_SOURCE_FILE_COUNT
    ):
        raise ValueError(
            "manifest sealed_test_source_file_count must be "
            f"{SEALED_TEST_SOURCE_FILE_COUNT}"
        )
    if manifest.get("corrections_applied") is not False:
        raise ValueError("manifest must record corrections_applied=false")
    if manifest.get("test_opened") is not False:
        raise ValueError("training manifest must record test_opened=false")

    split = _validate_serialized_split(manifest.get("split"))
    validate_support_contract(manifest.get("support"))
    shards = list(_require_sequence(manifest.get("shards"), "manifest.shards"))
    if len(shards) != SHARD_COUNT:
        raise ValueError(f"manifest must contain exactly {SHARD_COUNT} shards")

    seen_shards = set()
    seen_source_files = set()
    expected_roles = list(split.role_by_sector)
    for index, shard in enumerate(shards):
        where = f"manifest.shards[{index}]"
        if not isinstance(shard, Mapping):
            raise ValueError(f"{where} must be a mapping")
        try:
            pass_id = int(shard["pass_id"])
            polarization = str(shard["polarization"]).lower()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{where} lacks a valid pass/polarization") from exc
        expected_id = canonical_shard_id(pass_id, polarization)
        if shard.get("shard_id") != expected_id:
            raise ValueError(f"{where}.shard_id must be {expected_id}")
        if expected_id in seen_shards:
            raise ValueError(f"manifest repeats shard {expected_id}")
        seen_shards.add(expected_id)

        sector_ids = list(_require_sequence(shard.get("sector_ids"), f"{where}.sector_ids"))
        if sector_ids != list(SECTOR_IDS):
            raise ValueError(f"{where} must enumerate sectors 1..360 in order")
        sector_roles = list(_require_sequence(shard.get("sector_roles"), f"{where}.sector_roles"))
        if sector_roles != expected_roles:
            raise ValueError(f"{where} does not share the canonical sector roles")
        payload_audited_sector_ids = list(
            _require_sequence(
                shard.get("payload_audited_sector_ids"),
                f"{where}.payload_audited_sector_ids",
            )
        )
        if payload_audited_sector_ids != list(PAYLOAD_AUDITED_SECTOR_IDS):
            raise ValueError(
                f"{where}.payload_audited_sector_ids must be the exact "
                "324 train+validation sectors in canonical order"
            )
        if (
            int(shard.get("payload_audited_sector_count", -1))
            != PAYLOAD_AUDITED_SECTOR_COUNT
        ):
            raise ValueError(
                f"{where}.payload_audited_sector_count must be "
                f"{PAYLOAD_AUDITED_SECTOR_COUNT}"
            )
        sealed_test_sector_ids = list(
            _require_sequence(
                shard.get("sealed_test_sector_ids"),
                f"{where}.sealed_test_sector_ids",
            )
        )
        if sealed_test_sector_ids != list(SEALED_TEST_SECTOR_IDS):
            raise ValueError(
                f"{where}.sealed_test_sector_ids must be the exact "
                "36 held-out test sectors in canonical order"
            )
        if (
            int(shard.get("sealed_test_sector_count", -1))
            != SEALED_TEST_SECTOR_COUNT
        ):
            raise ValueError(
                f"{where}.sealed_test_sector_count must be "
                f"{SEALED_TEST_SECTOR_COUNT}"
            )
        source_files = list(_require_sequence(shard.get("source_files"), f"{where}.source_files"))
        if len(source_files) != SECTOR_COUNT:
            raise ValueError(f"{where} must contain exactly 360 source files")
        for source_file in source_files:
            if not isinstance(source_file, str) or not source_file.strip():
                raise ValueError(f"{where} contains an invalid source path")
            if source_file in seen_source_files:
                raise ValueError(f"source file is reused across shards: {source_file}")
            seen_source_files.add(source_file)

    if seen_shards != set(SHARD_IDS):
        missing = sorted(set(SHARD_IDS) - seen_shards)
        extra = sorted(seen_shards - set(SHARD_IDS))
        raise ValueError(f"manifest shard coverage mismatch: missing={missing}, extra={extra}")
    if len(seen_source_files) != SOURCE_FILE_COUNT:
        raise ValueError("manifest source paths are not globally unique and complete")
    return {
        "schema": MANIFEST_SCHEMA,
        "scene_id": SCENE_ID,
        "manager_track_id": MANAGER_TRACK_ID,
        "scene_count": SCENE_COUNT,
        "shard_count": SHARD_COUNT,
        "source_file_count": SOURCE_FILE_COUNT,
        "inventoried_source_file_count": INVENTORIED_SOURCE_FILE_COUNT,
        "payload_audited_source_file_count": PAYLOAD_AUDITED_SOURCE_FILE_COUNT,
        "sealed_test_source_file_count": SEALED_TEST_SOURCE_FILE_COUNT,
        "sector_counts": {
            role: len(split.sectors(role)) for role in _ROLE_ORDER
        },
        "support_point_count": SUPPORT_POINT_COUNT,
        "test_opened": False,
    }


@dataclass(frozen=True)
class StratumMetricSums:
    """Additive sufficient statistics for one pass/polarization stratum."""

    coherent_squared_error: float
    coherent_target_energy: float
    range_power_squared_error: float
    range_power_target_energy: float
    complex_sample_count: int

    def validate(self) -> "StratumMetricSums":
        values = (
            self.coherent_squared_error,
            self.coherent_target_energy,
            self.range_power_squared_error,
            self.range_power_target_energy,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("metric sums must be finite")
        if self.coherent_squared_error < 0 or self.range_power_squared_error < 0:
            raise ValueError("metric error energies must be nonnegative")
        if self.coherent_target_energy <= 0 or self.range_power_target_energy <= 0:
            raise ValueError("metric target energies must be positive")
        if int(self.complex_sample_count) <= 0:
            raise ValueError("complex_sample_count must be positive")
        return self

    def normalized(self) -> dict[str, Any]:
        self.validate()
        coherent_mse = self.coherent_squared_error / self.coherent_target_energy
        range_mse = self.range_power_squared_error / self.range_power_target_energy
        return {
            "coherent_relative_mse": float(coherent_mse),
            "coherent_relative_l2": float(math.sqrt(coherent_mse)),
            "range_power_relative_mse": float(range_mse),
            "range_power_relative_l2": float(math.sqrt(range_mse)),
            # Preserve both numerators as well as denominators.  A report can
            # therefore reproduce every normalized value without reopening
            # the prediction tensors or trusting this reducer.
            "coherent_squared_error": float(self.coherent_squared_error),
            "coherent_target_energy": float(self.coherent_target_energy),
            "range_power_squared_error": float(
                self.range_power_squared_error
            ),
            "range_power_target_energy": float(self.range_power_target_energy),
            "complex_sample_count": int(self.complex_sample_count),
        }


def metric_sums(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    frequency_axis: int = -1,
) -> StratumMetricSums:
    """Compute coherent and matched range-power sufficient statistics.

    Prediction and target undergo the same orthonormal inverse FFT before
    squared magnitudes are compared.  Polarizations are never coherently
    summed.  The returned quantities are additive across chunks.
    """

    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if prediction.shape != target.shape or prediction.size == 0:
        raise ValueError("prediction and target must have the same nonempty shape")
    if not np.iscomplexobj(prediction) or not np.iscomplexobj(target):
        raise ValueError("coherent metrics require complex prediction and target")
    if not np.isfinite(prediction.real).all() or not np.isfinite(prediction.imag).all():
        raise ValueError("prediction must be finite")
    if not np.isfinite(target.real).all() or not np.isfinite(target.imag).all():
        raise ValueError("target must be finite")
    axis = int(frequency_axis)
    if axis < 0:
        axis += target.ndim
    if axis < 0 or axis >= target.ndim:
        raise ValueError("frequency_axis is outside the array")

    residual = prediction - target
    coherent_target_energy = float(np.sum(np.abs(target) ** 2, dtype=np.float64))
    coherent_squared_error = float(np.sum(np.abs(residual) ** 2, dtype=np.float64))

    prediction_range = np.fft.ifft(prediction, axis=axis, norm="ortho")
    target_range = np.fft.ifft(target, axis=axis, norm="ortho")
    prediction_power = np.abs(prediction_range) ** 2
    target_power = np.abs(target_range) ** 2
    power_residual = prediction_power - target_power
    range_power_target_energy = float(
        np.sum(target_power**2, dtype=np.float64)
    )
    range_power_squared_error = float(
        np.sum(power_residual**2, dtype=np.float64)
    )

    # Make the zero-predictor reference an exact representable contract rather
    # than merely a numerically close property of two separately reduced sums.
    if np.count_nonzero(prediction) == 0:
        coherent_squared_error = coherent_target_energy
        range_power_squared_error = range_power_target_energy

    return StratumMetricSums(
        coherent_squared_error=coherent_squared_error,
        coherent_target_energy=coherent_target_energy,
        range_power_squared_error=range_power_squared_error,
        range_power_target_energy=range_power_target_energy,
        complex_sample_count=int(target.size),
    ).validate()


_NORMALIZED_METRIC_NAMES = (
    "coherent_relative_mse",
    "coherent_relative_l2",
    "range_power_relative_mse",
    "range_power_relative_l2",
)


def _reduce_metric_group(
    ordered_keys: Sequence[str],
    by_stratum: Mapping[str, StratumMetricSums],
    per_stratum: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Reduce a nonempty ordered group without mixing coherent channels."""

    if not ordered_keys:
        raise ValueError("a metric aggregate must contain at least one stratum")
    sums = [by_stratum[key] for key in ordered_keys]
    macro = {
        name: float(math.fsum(per_stratum[key][name] for key in ordered_keys) / len(ordered_keys))
        for name in _NORMALIZED_METRIC_NAMES
    }
    coherent_error = math.fsum(value.coherent_squared_error for value in sums)
    coherent_energy = math.fsum(value.coherent_target_energy for value in sums)
    range_error = math.fsum(value.range_power_squared_error for value in sums)
    range_energy = math.fsum(value.range_power_target_energy for value in sums)
    coherent_mse = coherent_error / coherent_energy
    range_mse = range_error / range_energy
    pooled = {
        "coherent_relative_mse": float(coherent_mse),
        "coherent_relative_l2": float(math.sqrt(coherent_mse)),
        "range_power_relative_mse": float(range_mse),
        "range_power_relative_l2": float(math.sqrt(range_mse)),
        "coherent_squared_error": float(coherent_error),
        "coherent_target_energy": float(coherent_energy),
        "range_power_squared_error": float(range_error),
        "range_power_target_energy": float(range_energy),
        "complex_sample_count": int(
            sum(value.complex_sample_count for value in sums)
        ),
    }
    return {
        "strata": list(ordered_keys),
        "stratum_count": len(ordered_keys),
        "macro": macro,
        "energy_pooled": pooled,
    }


def reduce_metric_sums(
    by_stratum: Mapping[str, StratumMetricSums],
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Reduce strata globally, per polarization, and per pass.

    Every aggregate has two deliberately different readings.  ``macro`` is
    the unweighted arithmetic mean of independently normalized stratum values
    (including an arithmetic mean of per-stratum relative L2).  In contrast,
    ``energy_pooled`` sums error numerators and target-energy denominators
    before division, and defines relative L2 as the square root of that pooled
    relative MSE.  No aggregate coherently adds different polarizations.
    """

    if not isinstance(by_stratum, Mapping) or not by_stratum:
        raise ValueError("by_stratum must be a nonempty mapping")
    keys = set(by_stratum)
    expected = set(SHARD_IDS)
    if require_complete and keys != expected:
        raise ValueError(
            f"metric strata must be the exact 32 shards; missing={sorted(expected - keys)}, "
            f"extra={sorted(keys - expected)}"
        )
    unknown = keys - expected
    if unknown:
        raise ValueError(f"unknown metric strata: {sorted(unknown)}")
    ordered_keys = [key for key in SHARD_IDS if key in by_stratum]
    per_stratum = {}
    for key in ordered_keys:
        value = by_stratum[key]
        if not isinstance(value, StratumMetricSums):
            raise ValueError(f"metric stratum {key} is not StratumMetricSums")
        value.validate()
        per_stratum[key] = value.normalized()

    global_group = _reduce_metric_group(ordered_keys, by_stratum, per_stratum)
    per_polarization = {
        polarization: _reduce_metric_group(
            [
                canonical_shard_id(pass_id, polarization)
                for pass_id in PASS_IDS
                if canonical_shard_id(pass_id, polarization) in by_stratum
            ],
            by_stratum,
            per_stratum,
        )
        for polarization in POLARIZATIONS
        if any(
            canonical_shard_id(pass_id, polarization) in by_stratum
            for pass_id in PASS_IDS
        )
    }
    per_pass = {
        f"pass{pass_id}": _reduce_metric_group(
            [
                canonical_shard_id(pass_id, polarization)
                for polarization in POLARIZATIONS
                if canonical_shard_id(pass_id, polarization) in by_stratum
            ],
            by_stratum,
            per_stratum,
        )
        for pass_id in PASS_IDS
        if any(
            canonical_shard_id(pass_id, polarization) in by_stratum
            for polarization in POLARIZATIONS
        )
    }
    return {
        "per_stratum": per_stratum,
        "per_polarization": per_polarization,
        "per_pass": per_pass,
        # Retain the original top-level global fields for consumers written
        # against the first contract draft.
        "macro": global_group["macro"],
        "energy_pooled": global_group["energy_pooled"],
        "stratum_count": len(ordered_keys),
        "aggregation_semantics": {
            "stratum": (
                "one pass/polarization; different polarizations are never "
                "coherently summed"
            ),
            "macro": (
                "unweighted arithmetic mean of each stratum's independently "
                "normalized value; relative-L2 values themselves are averaged"
            ),
            "energy_pooled": (
                "sum error numerators and matching target-energy denominators "
                "before division; relative L2 is sqrt(pooled relative MSE)"
            ),
        },
    }


def evaluate_joint_metrics(
    predictions: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    *,
    frequency_axis: int = -1,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Compute and reduce Joint-8 metrics without cross-polarization summing."""

    if set(predictions) != set(targets):
        raise ValueError("prediction and target stratum keys must match")
    sums = {
        key: metric_sums(
            predictions[key], targets[key], frequency_axis=frequency_axis
        )
        for key in predictions
    }
    return reduce_metric_sums(sums, require_complete=require_complete)


def exact_zero_predictor_gate(
    targets: Mapping[str, np.ndarray],
    *,
    frequency_axis: int = -1,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Require the zero predictor to score exactly 1.0 under every metric."""

    zeros = {key: np.zeros_like(value) for key, value in targets.items()}
    result = evaluate_joint_metrics(
        zeros,
        targets,
        frequency_axis=frequency_axis,
        require_complete=require_complete,
    )
    metric_names = (
        "coherent_relative_mse",
        "coherent_relative_l2",
        "range_power_relative_mse",
        "range_power_relative_l2",
    )
    rows = list(result["per_stratum"].items()) + [
        ("macro", result["macro"]),
        ("energy_pooled", result["energy_pooled"]),
    ]
    for group_name in ("per_polarization", "per_pass"):
        for aggregate_name, aggregate in result[group_name].items():
            rows.extend(
                (
                    (f"{group_name}.{aggregate_name}.macro", aggregate["macro"]),
                    (
                        f"{group_name}.{aggregate_name}.energy_pooled",
                        aggregate["energy_pooled"],
                    ),
                )
            )
    for location, row in rows:
        for name in metric_names:
            if row[name] != 1.0:
                raise ValueError(
                    f"zero-predictor gate failed at {location}.{name}: {row[name]!r}"
                )
    result["zero_predictor_exact"] = True
    return result


def validate_calibration_artifact(
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate frozen train-only nuisances and channel-own autofocus provenance.

    HH and VV must name their own official ``af.r_correct`` and
    ``af.ph_correct`` sources and prove those corrections were validated before
    application.  HV and VH must prove the official correction arrays were
    absent and must not borrow another channel's values.
    """

    if not isinstance(artifact, Mapping):
        raise ValueError("calibration artifact must be a mapping")
    if artifact.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError(f"calibration schema must be {CALIBRATION_SCHEMA}")
    if artifact.get("scene_id") != SCENE_ID:
        raise ValueError(f"calibration scene_id must be {SCENE_ID}")
    if int(artifact.get("split_seed", -1)) != SPLIT_SEED:
        raise ValueError("calibration split_seed must be 42")
    if artifact.get("fit_role") != "train":
        raise ValueError("calibration nuisances must be fit on train only")
    if artifact.get("frozen") is not True:
        raise ValueError("calibration artifact must be frozen")
    if artifact.get("test_opened") is not False:
        raise ValueError("calibration must be frozen before test is opened")
    try:
        offset_bound = float(artifact["range_offset_bound_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("calibration requires range_offset_bound_m") from exc
    if (
        not math.isfinite(offset_bound)
        or offset_bound <= 0
        or offset_bound > MAX_RANGE_OFFSET_BOUND_M
    ):
        raise ValueError(
            f"range_offset_bound_m must be in (0, {MAX_RANGE_OFFSET_BOUND_M}]"
        )

    entries = list(_require_sequence(artifact.get("strata"), "calibration.strata"))
    if len(entries) != SHARD_COUNT:
        raise ValueError(f"calibration must contain exactly {SHARD_COUNT} strata")
    expected_train = set(build_sector_split().train)
    seen = set()
    for index, entry in enumerate(entries):
        where = f"calibration.strata[{index}]"
        if not isinstance(entry, Mapping):
            raise ValueError(f"{where} must be a mapping")
        try:
            pass_id = int(entry["pass_id"])
            polarization = str(entry["polarization"]).lower()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{where} lacks a valid pass/polarization") from exc
        shard_id = canonical_shard_id(pass_id, polarization)
        if entry.get("shard_id") != shard_id:
            raise ValueError(f"{where}.shard_id must be {shard_id}")
        if shard_id in seen:
            raise ValueError(f"calibration repeats stratum {shard_id}")
        seen.add(shard_id)
        if entry.get("fit_role") != "train" or entry.get("frozen") is not True:
            raise ValueError(f"{shard_id} nuisance must be train-only and frozen")
        try:
            fit_sector_values = [
                int(value)
                for value in _require_sequence(
                    entry.get("fit_sector_ids"), f"{shard_id}.fit_sector_ids"
                )
            ]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{shard_id} nuisance fit sectors must be integer sector IDs"
            ) from exc
        fit_sectors = set(fit_sector_values)
        if (
            len(fit_sector_values) != TRAIN_SECTOR_COUNT
            or len(fit_sectors) != TRAIN_SECTOR_COUNT
            or fit_sectors != expected_train
        ):
            raise ValueError(
                f"{shard_id} nuisance fit sectors must be the exact training role"
            )
        gains: dict[str, float] = {}
        for name in ("gain_real", "gain_imag"):
            try:
                value = float(entry[name])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{shard_id}.{name} must be numeric") from exc
            if not math.isfinite(value):
                raise ValueError(f"{shard_id}.{name} must be finite")
            gains[name] = value
        try:
            range_offset = float(entry["range_offset_m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{shard_id}.range_offset_m must be numeric") from exc
        if not math.isfinite(range_offset) or abs(range_offset) > offset_bound:
            raise ValueError(
                f"{shard_id}.range_offset_m exceeds the frozen bound {offset_bound}"
            )
        if pass_id == 2:
            gauge_values = {
                "gain_real": gains["gain_real"],
                "gain_imag": gains["gain_imag"],
                "range_offset_m": range_offset,
            }
            gauge_expected = {
                "gain_real": 1.0,
                "gain_imag": 0.0,
                "range_offset_m": 0.0,
            }
            for name, expected_value in gauge_expected.items():
                if gauge_values[name] != expected_value:
                    raise ValueError(
                        f"{shard_id}.{name} violates the pass-2 nuisance gauge; "
                        f"expected {expected_value!r}"
                    )

        autofocus = entry.get("autofocus")
        if not isinstance(autofocus, Mapping):
            raise ValueError(f"{shard_id}.autofocus must be a mapping")
        if autofocus.get("validated") is not True:
            raise ValueError(f"{shard_id} autofocus presence/application was not validated")
        if polarization in CO_POLARIZATIONS:
            required = {
                "official_available": True,
                "applied": True,
                "source_shard_id": shard_id,
                "source_polarization": polarization,
                "range_field": "af.r_correct",
                "phase_field": "af.ph_correct",
                "source_file_count": SECTOR_COUNT,
            }
        else:
            required = {
                "official_available": False,
                "applied": False,
                "source_shard_id": None,
                "source_polarization": None,
                "range_field": None,
                "phase_field": None,
                "source_file_count": 0,
            }
        for key, expected_value in required.items():
            if autofocus.get(key) != expected_value:
                raise ValueError(
                    f"{shard_id}.autofocus.{key} must be {expected_value!r}"
                )

        application = autofocus.get("application_contract")
        heldout = autofocus.get("heldout_validation")
        if polarization in CO_POLARIZATIONS:
            if not isinstance(application, Mapping):
                raise ValueError(
                    f"{shard_id}.autofocus.application_contract must be a mapping"
                )
            for sign_name in ("range_sign", "phase_sign"):
                sign = application.get(sign_name)
                if isinstance(sign, bool) or not isinstance(
                    sign, (int, np.integer)
                ) or int(sign) not in (-1, 1):
                    raise ValueError(
                        f"{shard_id}.autofocus.application_contract.{sign_name} "
                        "must be integer -1 or +1"
                    )
            application_required = {
                "application_order": list(AUTOFOCUS_APPLICATION_ORDER),
                "range_formula": AUTOFOCUS_RANGE_FORMULA,
                "phase_formula": AUTOFOCUS_PHASE_FORMULA,
                "range_correction_units": "m",
                "phase_correction_units": "rad",
            }
            for key, expected_value in application_required.items():
                if application.get(key) != expected_value:
                    raise ValueError(
                        f"{shard_id}.autofocus.application_contract.{key} "
                        f"must be {expected_value!r}"
                    )

            if not isinstance(heldout, Mapping):
                raise ValueError(
                    f"{shard_id}.autofocus.heldout_validation must be a mapping"
                )
            heldout_required = {
                "role": "validation",
                "metric": AUTOFOCUS_HELDOUT_METRIC,
                "improved": True,
            }
            for key, expected_value in heldout_required.items():
                if heldout.get(key) != expected_value:
                    raise ValueError(
                        f"{shard_id}.autofocus.heldout_validation.{key} "
                        f"must be {expected_value!r}"
                    )
            try:
                heldout_sector_values = [
                    int(value)
                    for value in _require_sequence(
                        heldout.get("sector_ids"),
                        f"{shard_id}.autofocus.heldout_validation.sector_ids",
                    )
                ]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{shard_id} held-out sectors must be integer sector IDs"
                ) from exc
            expected_validation = set(build_sector_split().validation)
            if (
                len(heldout_sector_values) != VALIDATION_SECTOR_COUNT
                or len(set(heldout_sector_values)) != VALIDATION_SECTOR_COUNT
                or set(heldout_sector_values) != expected_validation
            ):
                raise ValueError(
                    f"{shard_id} autofocus must be proven on the exact validation role"
                )
            try:
                score_before = float(heldout["score_before"])
                score_after = float(heldout["score_after"])
                heldout_sample_count = int(heldout["complex_sample_count"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{shard_id} autofocus held-out scores/count are invalid"
                ) from exc
            if (
                not math.isfinite(score_before)
                or not math.isfinite(score_after)
                or score_before < 0
                or score_after < 0
                or not score_after < score_before
            ):
                raise ValueError(
                    f"{shard_id} autofocus held-out score must strictly improve"
                )
            if heldout_sample_count <= 0:
                raise ValueError(
                    f"{shard_id} autofocus held-out complex_sample_count must be positive"
                )
        else:
            if application is not None:
                raise ValueError(
                    f"{shard_id}.autofocus.application_contract must be null"
                )
            if heldout is not None:
                raise ValueError(
                    f"{shard_id}.autofocus.heldout_validation must be null"
                )

    if seen != set(SHARD_IDS):
        raise ValueError("calibration strata do not cover all 32 pass/polarization pairs")
    return {
        "schema": CALIBRATION_SCHEMA,
        "scene_id": SCENE_ID,
        "stratum_count": SHARD_COUNT,
        "range_offset_bound_m": offset_bound,
        "fit_role": "train",
        "frozen": True,
        "test_opened": False,
        "autofocus_corrected_strata": len(PASS_IDS) * len(CO_POLARIZATIONS),
        "autofocus_absent_strata": len(PASS_IDS) * len(CROSS_POLARIZATIONS),
    }


if nn is not None:

    class _ComplexDegree3Head(nn.Module):
        """One polarization's independent real/imaginary SH parameters."""

        def __init__(self, shape, *, device, dtype, init_scale) -> None:
            super().__init__()
            self.w_re = nn.Parameter(
                init_scale * torch.randn(shape, device=device, dtype=dtype)
            )
            self.w_im = nn.Parameter(
                init_scale * torch.randn(shape, device=device, dtype=dtype)
            )

    class PolarimetricFixedPlanarSHScene(nn.Module):
        """Four independent complex degree-3 heads on one fixed planar lattice.

        The heads are separate modules rather than slices of one parameter.
        Consequently a batch for one polarization leaves every other head's
        gradients as ``None``; an optimizer cannot accidentally advance an
        idle head using stale momentum.  ``position_chunk`` remains shared by
        all four.  Smaller ``nx``/``ny`` values are accepted for unit and
        allocated-node smoke tests; production must instantiate the constants
        in :func:`support_contract`.
        """

        def __init__(
            self,
            *,
            device: Any,
            nx: int = SUPPORT_NX,
            ny: int = SUPPORT_NY,
            extent_m: float = SUPPORT_MAX_M,
            degree: int = SH_DEGREE,
            init_scale: float = 0.0,
            dtype: Any = None,
        ) -> None:
            super().__init__()
            nx = int(nx)
            ny = int(ny)
            degree = int(degree)
            extent_m = float(extent_m)
            init_scale = float(init_scale)
            if nx < 2 or ny < 2:
                raise ValueError("planar dimensions must both be at least two")
            if degree != SH_DEGREE:
                raise ValueError(f"Joint-8 v1 requires SH degree {SH_DEGREE}")
            if not math.isfinite(extent_m) or extent_m <= 0:
                raise ValueError("extent_m must be finite and positive")
            if not math.isfinite(init_scale) or init_scale < 0:
                raise ValueError("init_scale must be finite and nonnegative")
            if dtype is None:
                dtype = torch.get_default_dtype()
            if not torch.empty((), dtype=dtype).is_floating_point():
                raise ValueError("scene parameter dtype must be floating point")
            self.nx = nx
            self.ny = ny
            self.extent_m = extent_m
            self.max_degree = degree
            self.n_basis = SH_BASIS_COUNT
            self.n_points = nx * ny
            shape = (self.n_points, self.n_basis)
            self.heads = nn.ModuleDict(
                {
                    polarization: _ComplexDegree3Head(
                        shape,
                        device=device,
                        dtype=dtype,
                        init_scale=init_scale,
                    )
                    for polarization in POLARIZATIONS
                }
            )

        @property
        def shape(self) -> tuple[int, int, int]:
            return (self.nx, self.ny, 1)

        @property
        def pitch_xy(self) -> tuple[float, float]:
            return (
                2.0 * self.extent_m / self.nx,
                2.0 * self.extent_m / self.ny,
            )

        @staticmethod
        def polarization_index(polarization: str) -> int:
            polarization = str(polarization).lower()
            if polarization not in POLARIZATIONS:
                raise ValueError(f"polarization must be one of {POLARIZATIONS}")
            return POLARIZATIONS.index(polarization)

        def position_chunk(self, start: int, stop: int):
            start = int(start)
            stop = int(stop)
            if not 0 <= start <= stop <= self.n_points:
                raise ValueError("planar position chunk is out of range")
            reference = self.heads[POLARIZATIONS[0]].w_re
            flat = torch.arange(start, stop, device=reference.device, dtype=torch.int64)
            ix = torch.div(flat, self.ny, rounding_mode="floor")
            iy = torch.remainder(flat, self.ny)
            pitch_x, pitch_y = self.pitch_xy
            x = -self.extent_m + (ix.to(reference.dtype) + 0.5) * pitch_x
            y = -self.extent_m + (iy.to(reference.dtype) + 0.5) * pitch_y
            return torch.stack((x, y, torch.zeros_like(x)), dim=-1)

        def view_basis(self, theta, phi):
            from rift.spherical_harmonics import real_sh_basis

            reference = self.heads[POLARIZATIONS[0]].w_re
            theta = torch.as_tensor(theta, device=reference.device, dtype=reference.dtype)
            phi = torch.as_tensor(phi, device=reference.device, dtype=reference.dtype)
            if theta.ndim != 1 or phi.ndim != 1 or theta.shape != phi.shape:
                raise ValueError("theta and phi must be matching nonempty 1D tensors")
            if theta.numel() == 0:
                raise ValueError("theta and phi must not be empty")
            basis = real_sh_basis(theta, phi, self.max_degree)
            if basis.shape != (self.n_basis, theta.numel()):
                raise RuntimeError("unexpected spherical-harmonic basis shape")
            return basis.to(dtype=reference.dtype)

        def view_weight_chunk(self, polarization: str, basis, start: int, stop: int):
            polarization = str(polarization).lower()
            self.polarization_index(polarization)
            head = self.heads[polarization]
            basis = torch.as_tensor(
                basis, device=head.w_re.device, dtype=head.w_re.dtype
            )
            if basis.ndim != 2 or basis.shape[0] != self.n_basis or basis.shape[1] == 0:
                raise ValueError("basis must have shape [16, nonzero views]")
            start = int(start)
            stop = int(stop)
            if not 0 <= start <= stop <= self.n_points:
                raise ValueError("planar weight chunk is out of range")
            real = head.w_re[start:stop] @ basis
            imag = head.w_im[start:stop] @ basis
            return torch.complex(real, imag)

        def scatterer_basis_chunks(self, polarization: str, basis, chunk_size: int):
            chunk_size = int(chunk_size)
            if chunk_size <= 0:
                raise ValueError("chunk_size must be positive")
            for start in range(0, self.n_points, chunk_size):
                stop = min(start + chunk_size, self.n_points)
                yield (
                    self.position_chunk(start, stop),
                    self.view_weight_chunk(polarization, basis, start, stop),
                )

        def assert_production_support(self) -> None:
            if (
                self.nx != SUPPORT_NX
                or self.ny != SUPPORT_NY
                or not math.isclose(
                    self.extent_m, SUPPORT_MAX_M, rel_tol=0.0, abs_tol=0.0
                )
            ):
                raise ValueError("scene instance does not use the Joint-8 production support")


else:  # pragma: no cover - simple diagnostic path for NumPy-only environments.

    class PolarimetricFixedPlanarSHScene:
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "PolarimetricFixedPlanarSHScene requires PyTorch; pure contract helpers do not"
            )


__all__ = [
    "AUTOFOCUS_APPLICATION_ORDER",
    "AUTOFOCUS_HELDOUT_METRIC",
    "AUTOFOCUS_PHASE_FORMULA",
    "AUTOFOCUS_RANGE_FORMULA",
    "CALIBRATION_SCHEMA",
    "CO_POLARIZATIONS",
    "CROSS_POLARIZATIONS",
    "INCORRECT_ONE_OVER_32_PITCH_M",
    "INVENTORIED_SOURCE_FILE_COUNT",
    "MANAGER_TRACK_ID",
    "MANIFEST_SCHEMA",
    "MAX_RANGE_OFFSET_BOUND_M",
    "PASS_IDS",
    "PAYLOAD_AUDITED_SECTOR_COUNT",
    "PAYLOAD_AUDITED_SECTOR_IDS",
    "PAYLOAD_AUDITED_SOURCE_FILE_COUNT",
    "POLARIZATIONS",
    "PolarimetricFixedPlanarSHScene",
    "SCENE_ID",
    "SCENE_COUNT",
    "SEALED_TEST_SECTOR_COUNT",
    "SEALED_TEST_SECTOR_IDS",
    "SEALED_TEST_SOURCE_FILE_COUNT",
    "SECTOR_COUNT",
    "SECTOR_IDS",
    "SHARD_COUNT",
    "SHARD_IDS",
    "SH_BASIS_COUNT",
    "SH_DEGREE",
    "SOURCE_FILE_COUNT",
    "SPLIT_SEED",
    "SPLIT_STRATEGY",
    "SUPPORT_FIRST_CENTER_M",
    "SUPPORT_LAST_CENTER_M",
    "SUPPORT_MAX_M",
    "SUPPORT_MIN_M",
    "SUPPORT_NX",
    "SUPPORT_NY",
    "SUPPORT_PITCH_M",
    "SUPPORT_POINT_COUNT",
    "SectorSplit",
    "StratumMetricSums",
    "build_sector_split",
    "canonical_shard_id",
    "evaluate_joint_metrics",
    "exact_zero_predictor_gate",
    "metric_sums",
    "reduce_metric_sums",
    "support_contract",
    "validate_calibration_artifact",
    "validate_joint_manifest",
    "validate_support_contract",
]
