"""Sealed, semantic B787 sphere10k protocol for the RadarSplat repair lane.

This module deliberately does not modify the historical RadarSplat source,
its sphere2k cache layout, or its checkpoints.  It describes only the new
3,200-train / 1,000-validation B787 development lane.  The protocol is based
on direct schema, role, acquisition, and grid checks; it intentionally has no
content-digest or source-identity gate.

RadarSplat's native training observable is a real local-polar power image:
``sum_elevation(abs(matched_filter_complex)**2)``.  It is *not* a coherent
complex target, so this protocol never stores or manufactures a phase target.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


B787_3200_CANONICAL_NPZ_PATH = (
    "/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/"
    "b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz"
)
B787_3200_CANONICAL_MANIFEST_PATH = (
    "/storage/scratch1/1/dbao31/rift_round8b_impl_20260810/splits/round8b/"
    "b78710k_interp_seed42_train3200_val1000_test1000_v1.json"
)
B787_3200_MANIFEST_NAME = "b78710k_interp_seed42_train3200_val1000_test1000_v1"
B787_3200_NUM_VIEWS = 10_000
B787_3200_NUM_TRAIN = 3_200
B787_3200_NUM_VALIDATION = 1_000
B787_3200_NUM_TEST = 1_000
B787_3200_NUM_UNUSED = 4_800
B787_3200_SEED = 42
CURRENT_OCCUPANCY_THRESHOLD = 0.001

CACHE_SCHEMA = "rift_radarsplat_b7873200_native_power_cache_v1"
TARGET_SCHEMA = "rift_radarsplat_b7873200_native_power_target_v1"
RECIPE_FILENAME = "radarsplat_b7873200_recipe.json"
MANIFEST_FILENAME = "radarsplat_b7873200_manifest.json"
STATS_FILENAME = "radarsplat_b7873200_stats.json"
TARGET_DIRECTORY = "radarsplat_b7873200_views"
ACQUISITION_SCHEMA = "rift_radarsplat_b7873200_acquisition_v1"
ACQUISITION_FILENAME = "radarsplat_b7873200_acquisition.npz"


def _ordered_ids(
    roles: Mapping[str, object], role: str, count: int
) -> tuple[int, ...]:
    values = roles.get(role)
    if not isinstance(values, list) or len(values) != int(count):
        raise ValueError(
            f"B7873200 role {role!r} must contain exactly {int(count)} ordered IDs"
        )
    result: list[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"B7873200 role {role!r} contains a non-integer ID")
        index = int(value)
        if index < 0 or index >= B787_3200_NUM_VIEWS:
            raise ValueError(f"B7873200 role {role!r} contains out-of-range ID {index}")
        result.append(index)
    if len(set(result)) != len(result):
        raise ValueError(f"B7873200 role {role!r} contains duplicate IDs")
    return tuple(result)


def b7873200_sealed_identity(contract: Mapping[str, object]) -> dict[str, object]:
    """Validate and return the path-free meaning of the B787 sphere10k split.

    The generic NPZ loader establishes the archive header before it allows a
    response row to be read.  This narrower policy then fixes the canonical
    interpolation roles and ensures the reserved test and unused rows remain
    outside this baseline's development cache.
    """

    if isinstance(contract, Mapping) and (
        "dataset_identity" in contract
        or str(contract.get("role_manifest_name", "")).startswith("rift_dataset_")
    ):
        from rift.rift_dataset import collection_contract
        return collection_contract(contract)
    if not isinstance(contract, Mapping):
        raise ValueError("B7873200 sealed protocol contract must be a mapping")
    expected_shape = [B787_3200_NUM_VIEWS, 16, 16, 1, 600]
    if contract.get("schema") != "rift_npz_sealed_protocol_v1" or contract.get("version") != 1:
        raise ValueError("B7873200 requires the established sealed NPZ protocol v1")
    if contract.get("data_format") != "npz":
        raise ValueError("B7873200 sealed protocol must name NPZ data")
    if contract.get("response_shape") != expected_shape or contract.get("response_dtype") != "complex64":
        raise ValueError("B7873200 requires the complete 16x16x1x600 complex acquisition")
    if contract.get("role_manifest_name") != B787_3200_MANIFEST_NAME:
        raise ValueError("B7873200 requires the canonical interpolation role manifest")
    if contract.get("split_strategy") != "fixed_tail_subsampled":
        raise ValueError("B7873200 requires the fixed-tail-subsampled interpolation split")
    roles = contract.get("role_ids")
    if not isinstance(roles, Mapping):
        raise ValueError("B7873200 sealed protocol lacks ordered role IDs")
    train = _ordered_ids(roles, "train", B787_3200_NUM_TRAIN)
    validation = _ordered_ids(roles, "validation", B787_3200_NUM_VALIDATION)
    reserved_test = _ordered_ids(roles, "reserved_test", B787_3200_NUM_TEST)
    unused = _ordered_ids(roles, "unused", B787_3200_NUM_UNUSED)

    permutation = np.random.Generator(np.random.PCG64(B787_3200_SEED)).permutation(
        B787_3200_NUM_VIEWS
    )
    validation_start = B787_3200_NUM_VIEWS - B787_3200_NUM_VALIDATION
    test_start = validation_start - B787_3200_NUM_TEST
    expected_roles = {
        "train": tuple(int(value) for value in permutation[:B787_3200_NUM_TRAIN]),
        "validation": tuple(int(value) for value in permutation[validation_start:]),
        "reserved_test": tuple(int(value) for value in permutation[test_start:validation_start]),
        "unused": tuple(int(value) for value in permutation[B787_3200_NUM_TRAIN:test_start]),
    }
    observed_roles = {
        "train": train,
        "validation": validation,
        "reserved_test": reserved_test,
        "unused": unused,
    }
    if observed_roles != expected_roles:
        raise ValueError("B7873200 role IDs do not match the frozen PCG64(seed=42) partition")
    if len(set(train + validation + reserved_test + unused)) != B787_3200_NUM_VIEWS:
        raise ValueError("B7873200 roles are not a complete disjoint partition")
    expected_access = {
        "train_materialized": True,
        "validation_materialized": True,
        "reserved_test_materialized": False,
        "unused_materialized": False,
    }
    if contract.get("response_access") != expected_access:
        raise ValueError("B7873200 protocol would expose a sealed response role")
    return {
        "schema": "rift_npz_sealed_protocol_v1",
        "version": 1,
        "data_format": "npz",
        "response_shape": expected_shape,
        "response_dtype": "complex64",
        "role_manifest_name": B787_3200_MANIFEST_NAME,
        "split_strategy": "fixed_tail_subsampled",
        "role_ids": {
            role: list(values) for role, values in observed_roles.items()
        },
        "response_access": expected_access,
    }


def load_b7873200_sealed_identity(
    npz_path: str | os.PathLike[str], role_manifest_path: str | os.PathLike[str]
) -> dict[str, object]:
    """Load the authoritative header/roles before any response access."""

    requested = os.path.realpath(os.path.abspath(os.fspath(npz_path)))
    canonical = os.path.realpath(B787_3200_CANONICAL_NPZ_PATH)
    from rift.rift_dataset import collection_manifest, load_object_contract
    if collection_manifest(role_manifest_path):
        _arrays, contract = load_object_contract(npz_path, role_manifest_path)
        return b7873200_sealed_identity(contract)
    if requested != canonical:
        raise ValueError(
            "B7873200 requires the canonical sphere10k archive at "
            f"{B787_3200_CANONICAL_NPZ_PATH}; got {requested}"
        )
    # Imported lazily so pure protocol tests do not require PyTorch.
    from train import _load_sealed_npz_protocol_contract

    _arrays, contract = _load_sealed_npz_protocol_contract(
        npz_path,
        role_manifest_path,
        num_train=B787_3200_NUM_TRAIN,
        num_val=B787_3200_NUM_VALIDATION,
        num_test=B787_3200_NUM_TEST,
    )
    return b7873200_sealed_identity(contract)


def atomic_write_json(path: str | os.PathLike[str], payload: Mapping[str, object]) -> Path:
    """Atomically write ordinary semantic metadata, without an identity digest."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent,
        prefix=destination.name + ".tmp.", delete=False,
    ) as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def atomic_save_npz(path: str | os.PathLike[str], **arrays: object) -> Path:
    """Atomically write one authorized real-power target."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".npz", dir=destination.parent, prefix=destination.stem + ".tmp.", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def target_path(root: str | os.PathLike[str], view_index: int) -> Path:
    return Path(root) / TARGET_DIRECTORY / f"view_{int(view_index):06d}.npz"


def _read_json(path: Path, label: str) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"RadarSplat B7873200 {label} is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"RadarSplat B7873200 {label} must be a JSON object")
    return payload


def _finite_number(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"RadarSplat B7873200 {label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"RadarSplat B7873200 {label} must be finite" + (" and positive" if positive else ""))
    return result


def _direct_grid_spec(spec: Mapping[str, object]) -> dict[str, object]:
    required_ints = ("n_azimuth", "n_elevation", "n_range")
    result: dict[str, object] = {}
    for name in required_ints:
        value = spec.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 2:
            raise ValueError(f"RadarSplat B7873200 grid.{name} must be an integer of at least two")
        result[name] = int(value)
    for name in (
        "scene_extent_m",
        "azimuth_center_deg",
        "output_azimuth_resolution_deg",
        "elevation_sampling_resolution_deg",
        "intermediate_azimuth_resolution_deg",
        "azimuth_beamwidth_deg",
        "spectral_leakage_width_m",
    ):
        result[name] = _finite_number(
            spec.get(name), f"grid.{name}", positive=name != "azimuth_center_deg"
        )
    center = spec.get("scene_center_m")
    if not isinstance(center, Sequence) or isinstance(center, (str, bytes)) or len(center) != 3:
        raise ValueError("RadarSplat B7873200 grid.scene_center_m must have three numeric coordinates")
    result["scene_center_m"] = [
        _finite_number(value, f"grid.scene_center_m[{axis}]")
        for axis, value in enumerate(center)
    ]
    if not math.isclose(float(result["azimuth_center_deg"]), 0.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("RadarSplat B7873200 current native target builder fixes azimuth_center_deg at 0")
    ratio = float(result["output_azimuth_resolution_deg"]) / float(
        result["intermediate_azimuth_resolution_deg"]
    )
    if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("RadarSplat B7873200 output/intermediate azimuth resolution must be integral")
    azimuth_span_deg = int(result["n_azimuth"]) * float(
        result["output_azimuth_resolution_deg"]
    )
    if azimuth_span_deg > 360.0:
        raise ValueError(
            "RadarSplat B7873200 local target azimuth span must not exceed 360 degrees"
        )
    range_resolution_m = 2.0 * float(result["scene_extent_m"]) / int(result["n_range"])
    leakage_sigma_pixels = int(
        ((float(result["spectral_leakage_width_m"]) / 2.0) / range_resolution_m) / 3.0
    )
    if leakage_sigma_pixels < 1:
        raise ValueError(
            "RadarSplat B7873200 target grid would produce an invalid zero-pixel "
            "native spectral-leakage sigma"
        )
    return result


def _materialized_role_ids(
    sealed: Mapping[str, object],
    supplied: Mapping[str, Sequence[int]] | None,
) -> tuple[dict[str, list[int]], bool]:
    roles = sealed["role_ids"]
    assert isinstance(roles, Mapping)
    full = {
        "train": [int(value) for value in roles["train"]],
        "validation": [int(value) for value in roles["validation"]],
    }
    if supplied is None:
        return full, False
    if set(supplied) != {"train", "validation"}:
        raise ValueError("RadarSplat B7873200 materialized roles must name exactly train and validation")
    materialized: dict[str, list[int]] = {}
    for role in ("train", "validation"):
        raw_selected = list(supplied[role])
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in raw_selected
        ):
            raise ValueError(f"RadarSplat B7873200 materialized {role} IDs must be integer IDs")
        selected = [int(value) for value in raw_selected]
        if not selected:
            raise ValueError(f"RadarSplat B7873200 materialized {role} role must be non-empty")
        if len(set(selected)) != len(selected) or any(value not in set(full[role]) for value in selected):
            raise ValueError(f"RadarSplat B7873200 materialized {role} IDs must be unique canonical role IDs")
        if selected != full[role][: len(selected)]:
            raise ValueError(
                f"RadarSplat B7873200 materialized {role} IDs must be a frozen role prefix"
            )
        materialized[role] = selected
    is_subset = materialized != full
    return materialized, is_subset


def expected_cache_recipe(
    identity: Mapping[str, object],
    target_spec: Mapping[str, object],
    *,
    materialized_roles: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, object]:
    """Build the direct, inspectable target-cache recipe.

    Callers supply only scientific target settings.  The immutable split
    meaning and full acquisition shape are filled here, so a cache cannot be
    accidentally attached to the old sphere2k data or a coordinate subset.
    """

    sealed = b7873200_sealed_identity(identity)
    materialized, is_subset = _materialized_role_ids(sealed, materialized_roles)
    if not isinstance(target_spec, Mapping):
        raise ValueError("RadarSplat B7873200 target_spec must be a mapping")
    grid = target_spec.get("grid")
    if not isinstance(grid, Mapping):
        raise ValueError("RadarSplat B7873200 target_spec requires a grid")
    direct_grid = _direct_grid_spec(grid)
    operator = target_spec.get("matched_filter")
    if not isinstance(operator, Mapping):
        raise ValueError("RadarSplat B7873200 target_spec requires matched_filter settings")
    expected_operator = {
        "phase_sign": -1.0,
        "response_layout": "tx_rx_freq",
        "range_model": "none",
        "include_four_pi": False,
    }
    for key, value in expected_operator.items():
        if operator.get(key) != value:
            raise ValueError(f"RadarSplat B7873200 requires matched_filter.{key}={value!r}")
    backend = operator.get("backend")
    if backend not in {"direct", "range_nufft"}:
        raise ValueError("RadarSplat B7873200 target preparation requires direct or range_nufft")
    threshold = _finite_number(
        target_spec.get("occupancy_threshold"), "occupancy_threshold", positive=True
    )
    if not math.isclose(threshold, CURRENT_OCCUPANCY_THRESHOLD, rel_tol=0.0, abs_tol=0.0):
        raise ValueError(
            "RadarSplat B7873200 preserves the current occupancy threshold 0.001; "
            "do not silently retune it in this repair lane"
        )
    return {
        "schema": CACHE_SCHEMA,
        "version": 1,
        "sealed_protocol_identity": sealed,
        "response_roles_materialized": ["train", "validation"],
        "materialized_role_ids": materialized,
        "development_subset": (
            None
            if not is_subset
            else {
                "purpose": "engineering small-fit only; not the 3200-view comparison",
                "train_count": len(materialized["train"]),
                "validation_count": len(materialized["validation"]),
            }
        ),
        **({'single_pair_sensor_adaptation': dict(schema='collection_element_hpbw_v1',
            azimuth_beamwidth_deg=10.0, source='simulator_element_power_hpbw',
            physical_psf_equivalence_verified=False)}
            if sealed.get('antenna_selection') and sealed['response_shape'][1:3] == [1, 1] else {}),
        "acquisition_record": {
            "schema": ACQUISITION_SCHEMA,
            "filename": ACQUISITION_FILENAME,
        },
        "acquisition": {
            "response_shape": list(identity["response_shape"]),
            "response_dtype": "complex64",
            "chirp_policy": "mean over the one stored chirp before native matched filtering",
            "coordinate_coverage": ("selected source Tx/Rx x all 600 frequencies" if identity.get("antenna_selection")
                                    else "all 16 Tx x 16 Rx x 600 frequency samples"),
        },
        "target_spec": {
            "observable": "native real polar power",
            "projection": "sum_elevation(abs(matched_filter_complex)**2) -> [azimuth,range]",
            "normalization": {
                "mode": "linear_peak",
                "fit_split": "train",
                "clip": False,
            },
            "occupancy_threshold": CURRENT_OCCUPANCY_THRESHOLD,
            "occupancy_comparison": "normalized power >= occupancy_threshold",
            "grid": direct_grid,
            "matched_filter": {
                **expected_operator,
                "backend": backend,
                "compute_dtype": str(operator.get("compute_dtype", "float64")),
                "point_chunk": int(operator.get("point_chunk", 1024)),
                "pair_chunk": int(operator.get("pair_chunk", 64)),
                "freq_chunk": int(operator.get("freq_chunk", 64)),
                "nufft_oversample": int(operator.get("nufft_oversample", 2)),
                "nufft_kernel_width": int(operator.get("nufft_kernel_width", 20)),
            },
        },
    }


def _scalar_text(value: object, label: str) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"RadarSplat B7873200 target {label} must be scalar")
    return str(array.item())


def load_target(
    root: str | os.PathLike[str],
    view_index: int,
    role: str,
    *,
    expected_grid: Mapping[str, object],
) -> dict[str, np.ndarray]:
    """Load and structurally validate one allowed native-power target."""

    if role not in {"train", "validation"}:
        raise ValueError("RadarSplat B7873200 only exposes train or validation targets")
    path = target_path(root, view_index)
    if not path.is_file():
        raise FileNotFoundError(f"RadarSplat B7873200 target is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "schema",
            "view_index",
            "role",
            "radarsplat_mf_power",
            "sensor_to_world",
            "range_m",
            "azimuth_rad",
            "elevation_rad",
        }
        observed = set(archive.files)
        if observed != required:
            missing = sorted(required.difference(observed))
            extra = sorted(observed.difference(required))
            raise ValueError(
                f"RadarSplat B7873200 target {path} fields differ; missing={missing}, extra={extra}"
            )
        values = {name: np.asarray(archive[name]) for name in required}
    if _scalar_text(values["schema"], "schema") != TARGET_SCHEMA:
        raise ValueError(f"RadarSplat B7873200 target {path} has the wrong schema")
    if int(np.asarray(values["view_index"]).reshape(())) != int(view_index):
        raise ValueError(f"RadarSplat B7873200 target {path} has a mismatched view index")
    if _scalar_text(values["role"], "role") != role:
        raise ValueError(f"RadarSplat B7873200 target {path} has a mismatched role")
    n_azimuth = int(expected_grid["n_azimuth"])
    n_elevation = int(expected_grid["n_elevation"])
    n_range = int(expected_grid["n_range"])
    power = values["radarsplat_mf_power"]
    if power.shape != (n_azimuth, n_range) or power.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError(f"RadarSplat B7873200 target {path} has an invalid power shape or dtype")
    if not np.isfinite(power).all() or np.any(power < 0.0):
        raise ValueError(f"RadarSplat B7873200 target {path} has invalid native power")
    pose = values["sensor_to_world"]
    if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(
        pose[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1.0e-6
    ):
        raise ValueError(f"RadarSplat B7873200 target {path} has an invalid sensor pose")
    rotation = pose[:3, :3].astype(np.float64, copy=False)
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=2.0e-5):
        raise ValueError(f"RadarSplat B7873200 target {path} sensor axes are not orthonormal")
    if np.linalg.det(rotation) <= 0.0:
        raise ValueError(f"RadarSplat B7873200 target {path} sensor axes are not right handed")
    for name, count in (("range_m", n_range), ("azimuth_rad", n_azimuth), ("elevation_rad", n_elevation)):
        axis = values[name]
        if axis.shape != (count,) or axis.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError(f"RadarSplat B7873200 target {path} has an invalid {name} shape or dtype")
        if not np.isfinite(axis).all() or np.any(np.diff(axis.astype(np.float64)) <= 0.0):
            raise ValueError(f"RadarSplat B7873200 target {path} has a non-monotone {name} axis")
    return values


@dataclass(frozen=True)
class RadarSplatB7873200Cache:
    """A verified train/validation-only local-polar target cache."""

    root: Path
    identity: Mapping[str, object]
    recipe: Mapping[str, object]
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    train_peak_power: float
    occupancy_threshold: float
    is_development_subset: bool
    acquisition_record: Mapping[str, np.ndarray]

    @property
    def grid(self) -> Mapping[str, object]:
        target_spec = self.recipe["target_spec"]
        assert isinstance(target_spec, Mapping)
        value = target_spec["grid"]
        assert isinstance(value, Mapping)
        return value


def load_cache(root: str | os.PathLike[str]) -> RadarSplatB7873200Cache:
    """Open an independently named cache without touching test or unused rows."""

    cache_root = Path(root)
    recipe = _read_json(cache_root / RECIPE_FILENAME, "cache recipe")
    if recipe.get("schema") != CACHE_SCHEMA or recipe.get("version") != 1:
        raise ValueError("RadarSplat B7873200 cache has the wrong schema/version")
    identity = recipe.get("sealed_protocol_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("RadarSplat B7873200 cache recipe lacks its sealed protocol identity")
    sealed_identity = b7873200_sealed_identity(identity)
    if recipe.get("sealed_protocol_identity") != sealed_identity:
        raise ValueError("RadarSplat B7873200 cache recipe disagrees with the canonical sealed identity")
    if recipe.get("response_roles_materialized") != ["train", "validation"]:
        raise ValueError("RadarSplat B7873200 cache may materialize only train and validation roles")
    target_spec = recipe.get("target_spec")
    if not isinstance(target_spec, Mapping):
        raise ValueError("RadarSplat B7873200 cache recipe lacks target_spec")
    materialized = recipe.get("materialized_role_ids")
    if not isinstance(materialized, Mapping):
        raise ValueError("RadarSplat B7873200 cache recipe lacks materialized role IDs")
    reconstructed = expected_cache_recipe(
        sealed_identity, target_spec, materialized_roles=materialized
    )
    if recipe != reconstructed:
        raise ValueError("RadarSplat B7873200 cache recipe disagrees with the native target contract")
    # The trainer opens this response-free record only.  It never needs a raw
    # B787 response accessor after cache preparation.
    from rift.radarsplat_b7873200_acquisition import (
        load_acquisition_record,
        validate_target_calibration,
    )
    manifest = _read_json(cache_root / MANIFEST_FILENAME, "target manifest")
    roles = recipe["materialized_role_ids"]
    assert isinstance(roles, Mapping)
    if (
        manifest.get("schema") != CACHE_SCHEMA
        or manifest.get("version") != 1
        or manifest.get("roles") != {"train": roles["train"], "validation": roles["validation"]}
    ):
        raise ValueError("RadarSplat B7873200 target manifest disagrees with the sealed roles")
    stats = _read_json(cache_root / STATS_FILENAME, "target statistics")
    if stats.get("schema") != CACHE_SCHEMA or stats.get("version") != 1:
        raise ValueError("RadarSplat B7873200 target statistics have the wrong schema/version")
    if stats.get("fit_split") != "train" or stats.get("normalization") != "linear_peak" or stats.get("clip") is not False:
        raise ValueError("RadarSplat B7873200 normalizer must be an unclipped train-only linear peak")
    acquisition_identity = ({key: sealed_identity[key] for key in
        ('antenna_selection', 'source_geometry_sha256', 'source_response_shape')}
        if sealed_identity.get('antenna_selection') else None)
    if stats.get('acquisition_identity') != acquisition_identity:
        raise ValueError('RadarSplat normalization acquisition differs from selected source channels')
    peak = _finite_number(stats.get("train_peak_power"), "train_peak_power", positive=True)
    threshold = _finite_number(stats.get("occupancy_threshold"), "occupancy_threshold", positive=True)
    if not math.isclose(threshold, CURRENT_OCCUPANCY_THRESHOLD, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("RadarSplat B7873200 cache statistics must preserve occupancy threshold 0.001")
    grid = target_spec["grid"]
    assert isinstance(grid, Mapping)
    train = tuple(int(value) for value in roles["train"])
    validation = tuple(int(value) for value in roles["validation"])
    acquisition_record = load_acquisition_record(
        cache_root, expected_view_indices=train + validation
    )
    record_metadata = json.loads(str(acquisition_record['metadata_json'].item()))
    if (record_metadata.get('rift_antenna_selection') != sealed_identity.get('antenna_selection')
            or list(acquisition_record['response_shape']) != sealed_identity['response_shape']):
        raise ValueError('RadarSplat acquisition record disagrees with selected source channels')
    if not np.array_equal(
        np.asarray(acquisition_record["scene_center_m"], dtype=np.float64),
        np.asarray(grid["scene_center_m"], dtype=np.float64),
    ):
        raise ValueError("RadarSplat B7873200 acquisition scene centre disagrees with its target-grid recipe")
    target_root = cache_root / TARGET_DIRECTORY
    allowed = set(train + validation)
    if not target_root.is_dir():
        raise FileNotFoundError(f"RadarSplat B7873200 target directory is missing: {target_root}")
    for candidate in target_root.glob("*.npz"):
        if not candidate.name.startswith("view_"):
            raise ValueError(f"RadarSplat B7873200 cache contains an unexpected target artifact: {candidate}")
        try:
            index = int(candidate.stem.split("_")[-1])
        except ValueError as error:
            raise ValueError(f"RadarSplat B7873200 target has a malformed filename: {candidate}") from error
        if index not in allowed:
            raise ValueError(
                "RadarSplat B7873200 cache contains a response-derived target outside train/validation"
            )
    observed_train_peak = 0.0
    for role, indices in (("train", train), ("validation", validation)):
        for index in indices:
            target = load_target(cache_root, index, role, expected_grid=grid)
            validate_target_calibration(target, acquisition_record, grid)
            if role == "train":
                observed_train_peak = max(
                    observed_train_peak, float(np.max(target["radarsplat_mf_power"]))
                )
    if not math.isclose(observed_train_peak, peak, rel_tol=0.0, abs_tol=max(1.0e-12, abs(peak) * 1.0e-6)):
        raise ValueError("RadarSplat B7873200 stored train peak disagrees with its authorized training targets")
    return RadarSplatB7873200Cache(
        root=cache_root,
        identity=sealed_identity,
        recipe=recipe,
        train_indices=train,
        validation_indices=validation,
        train_peak_power=peak,
        occupancy_threshold=threshold,
        is_development_subset=recipe.get("development_subset") is not None,
        acquisition_record=acquisition_record,
    )
