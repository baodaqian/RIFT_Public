"""Sealed B7873200 development policy for the corrected GeRaF wrapper.

This module intentionally leaves the frozen historical GeRaF model, target
preparer, trainer, cache schema, and checkpoint reader untouched.  It adds a
narrow wrapper-level policy for the new sphere10k comparison only.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping

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
B787_3200_TARGET_CACHE_PROTOCOL_FILENAME = "b7873200_sealed_protocol.json"
B787_3200_CACHE_RECIPE_FILENAME = "geraf_b7873200_recipe.json"
B787_3200_CACHE_MANIFEST_FILENAME = "geraf_b7873200_manifest.json"
B787_3200_CACHE_STATS_FILENAME = "geraf_b7873200_stats.json"
B787_3200_CACHE_VIEW_DIRECTORY = "geraf_b7873200_views"
B787_3200_CACHE_ACQUISITION_FILENAME = "geraf_b7873200_acquisition.npz"
B787_3200_CACHE_SCHEMA = "rift_geraf_b7873200_native_mf_cache_v1"
B787_3200_TARGET_SCHEMA = "rift_geraf_b7873200_native_mf_target_v1"
B787_3200_ACQUISITION_SCHEMA = "rift_geraf_b7873200_acquisition_v1"


def _ordered_role_ids(
    roles: Mapping[str, object], name: str, expected_count: int
) -> tuple[int, ...]:
    """Validate one ordered role without accepting coercions or duplicates."""

    values = roles.get(name)
    if not isinstance(values, list) or len(values) != int(expected_count):
        raise ValueError(
            f"B7873200 sealed protocol role {name!r} must contain "
            f"exactly {int(expected_count)} ordered IDs"
        )
    normalized: list[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"B7873200 sealed protocol role {name!r} contains a non-integer ID")
        index = int(value)
        if index < 0 or index >= B787_3200_NUM_VIEWS:
            raise ValueError(f"B7873200 sealed protocol role {name!r} contains out-of-range ID {index}")
        normalized.append(index)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"B7873200 sealed protocol role {name!r} contains duplicate IDs")
    return tuple(normalized)


def b7873200_sealed_protocol_identity(contract: Mapping[str, object]) -> Dict[str, object]:
    """Return the path-free semantic identity of the canonical B7873200 split.

    The generic sealed NPZ loader owns header-before-response access.  This
    narrow adapter additionally binds GeRaF's corrected B787 lane to the
    verified 3,200/1,000/1,000 PCG64 interpolation partition and refuses the
    4,800 unused or 1,000 reserved-test rows.  File and manifest locations are
    provenance, not identity: the valid archive lives under ``/storage/home``
    even though the historical manifest's path hint names the retired project
    location.
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
        raise ValueError("B7873200 sealed protocol disagrees with the 16x16x1x600 archive contract")
    if contract.get("role_manifest_name") != B787_3200_MANIFEST_NAME:
        raise ValueError("B7873200 requires the canonical interpolation role manifest")
    if contract.get("split_strategy") != "fixed_tail_subsampled":
        raise ValueError("B7873200 requires the fixed-tail-subsampled interpolation split")
    roles = contract.get("role_ids")
    if not isinstance(roles, Mapping):
        raise ValueError("B7873200 sealed protocol lacks ordered role IDs")
    train = _ordered_role_ids(roles, "train", B787_3200_NUM_TRAIN)
    validation = _ordered_role_ids(roles, "validation", B787_3200_NUM_VALIDATION)
    test = _ordered_role_ids(roles, "reserved_test", B787_3200_NUM_TEST)
    unused = _ordered_role_ids(roles, "unused", B787_3200_NUM_UNUSED)
    expected_permutation = np.random.Generator(np.random.PCG64(B787_3200_SEED)).permutation(
        B787_3200_NUM_VIEWS
    )
    validation_start = B787_3200_NUM_VIEWS - B787_3200_NUM_VALIDATION
    test_start = validation_start - B787_3200_NUM_TEST
    expected_train = expected_permutation[:B787_3200_NUM_TRAIN]
    expected_unused = expected_permutation[B787_3200_NUM_TRAIN:test_start]
    expected_test = expected_permutation[test_start:validation_start]
    expected_validation = expected_permutation[validation_start:]
    if (
        train != tuple(int(value) for value in expected_train)
        or validation != tuple(int(value) for value in expected_validation)
        or test != tuple(int(value) for value in expected_test)
        or unused != tuple(int(value) for value in expected_unused)
    ):
        raise ValueError("B7873200 role IDs do not match the frozen PCG64(seed=42) partition")
    if len(set(train + validation + test + unused)) != B787_3200_NUM_VIEWS:
        raise ValueError("B7873200 sealed roles are not a complete disjoint partition")
    expected_access = {
        "train_materialized": True,
        "validation_materialized": True,
        "reserved_test_materialized": False,
        "unused_materialized": False,
    }
    if contract.get("response_access") != expected_access:
        raise ValueError("B7873200 sealed protocol would expose a reserved response role")
    return {
        "schema": "rift_npz_sealed_protocol_v1",
        "version": 1,
        "data_format": "npz",
        "response_shape": expected_shape,
        "response_dtype": "complex64",
        "role_manifest_name": B787_3200_MANIFEST_NAME,
        "split_strategy": "fixed_tail_subsampled",
        "role_ids": {
            "train": list(train),
            "validation": list(validation),
            "reserved_test": list(test),
            "unused": list(unused),
        },
        "response_access": expected_access,
    }


def load_b7873200_sealed_protocol_identity(
    npz_path: str | os.PathLike[str], role_manifest_path: str | os.PathLike[str]
) -> Dict[str, object]:
    """Load the established sealed NPZ contract before any response row.

    Collection manifests use the public object-bound library loader. Historical
    B787 manifests retain the private trainer helper and their original path
    gate; call-time imports avoid a cycle with that legacy trainer.
    """

    requested_path = os.path.realpath(os.path.abspath(os.fspath(npz_path)))
    canonical_path = os.path.realpath(B787_3200_CANONICAL_NPZ_PATH)
    from rift.rift_dataset import collection_manifest, load_object_contract
    if collection_manifest(role_manifest_path):
        _arrays, contract = load_object_contract(npz_path, role_manifest_path)
        return b7873200_sealed_protocol_identity(contract)
    if requested_path != canonical_path:
        raise ValueError(
            "B7873200 requires the canonical sphere10k archive at "
            f"{B787_3200_CANONICAL_NPZ_PATH}; got {requested_path}"
        )

    from train import _load_sealed_npz_protocol_contract

    _arrays, contract = _load_sealed_npz_protocol_contract(
        npz_path,
        role_manifest_path,
        num_train=B787_3200_NUM_TRAIN,
        num_val=B787_3200_NUM_VALIDATION,
        num_test=B787_3200_NUM_TEST,
    )
    return b7873200_sealed_protocol_identity(contract)


def _read_json_mapping(path: Path, label: str) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"B7873200 {label} is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"B7873200 {label} must be a JSON object")
    return payload


def _validated_b7873200_recipe(
    recipe: Mapping[str, object], identity: Mapping[str, object]
) -> Mapping[str, object]:
    """Validate the new direct-equality GeRaF cache recipe."""

    if recipe.get("schema") != B787_3200_CACHE_SCHEMA or recipe.get("version") != 1:
        raise ValueError("B7873200 cache recipe has the wrong schema/version")
    if recipe.get("sealed_protocol_identity") != b7873200_sealed_protocol_identity(identity):
        raise ValueError("B7873200 cache recipe disagrees with the sealed split identity")
    if recipe.get("response_roles_materialized") != ["train", "validation"]:
        raise ValueError("B7873200 cache recipe must materialize exactly train and validation roles")
    if recipe.get("acquisition_record") != {
        "schema": B787_3200_ACQUISITION_SCHEMA,
        "filename": B787_3200_CACHE_ACQUISITION_FILENAME,
    }:
        raise ValueError("B7873200 cache recipe must name its direct acquisition record")
    target_spec = recipe.get("target_spec")
    if not isinstance(target_spec, Mapping):
        raise ValueError("B7873200 cache recipe lacks a target_spec object")
    required_target_spec = {
        "native_readout",
        "phase_sign",
        "grid",
        "backend",
        "compute_dtype",
    }
    missing = sorted(required_target_spec.difference(target_spec))
    if missing:
        raise ValueError(f"B7873200 cache recipe target_spec lacks {missing}")
    if target_spec.get("native_readout") != "complex magnitude |MF|":
        raise ValueError("B7873200 cache recipe must preserve GeRaF's native |MF| readout")
    try:
        phase_sign = float(target_spec.get("phase_sign", 0.0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("B7873200 cache recipe phase sign must be numeric") from exc
    if not np.isfinite(phase_sign) or phase_sign != -1.0:
        raise ValueError("B7873200 cache recipe must use the B787 phase convention -1")
    grid = target_spec.get("grid")
    if not isinstance(grid, Mapping):
        raise ValueError("B7873200 cache recipe target_spec.grid must be an object")
    for field in ("scene_extent_m", "aperture_scale"):
        value = grid.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"B7873200 cache recipe grid.{field} must be numeric")
        if not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"B7873200 cache recipe grid.{field} must be finite and positive")
    for field in ("n_azimuth", "n_elevation", "n_depth"):
        value = grid.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
            raise ValueError(f"B7873200 cache recipe grid.{field} must be a positive integer")
    if target_spec.get("backend") != "range":
        raise ValueError("B7873200 cache recipe must use the validated range target backend")
    if target_spec.get("compute_dtype") not in {"float32", "float64"}:
        raise ValueError("B7873200 cache recipe must use a supported floating compute dtype")
    operator = target_spec.get("operator")
    if not isinstance(operator, Mapping):
        raise ValueError("B7873200 cache recipe target_spec.operator must be an object")
    if operator.get("implementation") != "range_nufft":
        raise ValueError("B7873200 cache recipe must record the range-NUFFT operator")
    if operator.get("range_model") != "none" or operator.get("include_four_pi") is not False:
        raise ValueError("B7873200 cache recipe must use the native phase-only matched-filter convention")
    if operator.get("freq_chunk") is not None:
        raise ValueError("B7873200 range-NUFFT cache recipe must not carry a direct-backend frequency chunk")
    for field in ("kernel_width", "oversample", "point_chunk", "pair_chunk"):
        value = operator.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
            raise ValueError(f"B7873200 cache recipe operator.{field} must be a positive integer")
    return recipe


def write_b7873200_target_cache_protocol(
    cache_root: str | os.PathLike[str], identity: Mapping[str, object]
) -> Path:
    """Write one semantic sidecar after a complete target cache.

    This is not a lock, scheduler marker, or integrity pin.  It records the
    already validated sealed-role policy beside a *complete* versioned cache,
    so the wrapper can reject a legacy count-only invocation before training
    begins.
    """

    expected = b7873200_sealed_protocol_identity(identity)
    root = Path(cache_root)
    destination = root / B787_3200_TARGET_CACHE_PROTOCOL_FILENAME
    payload: Dict[str, object] = {
        "version": 1,
        "kind": "geraf_b7873200_sealed_target_cache",
        "sealed_npz_protocol_contract": expected,
    }
    if destination.exists():
        observed = _read_json_mapping(destination, "target-cache protocol sidecar")
        if observed != payload:
            raise ValueError(
                "B7873200 target-cache protocol sidecar disagrees with the requested sealed roles"
            )
        return destination
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=root, prefix=destination.name + ".tmp.", delete=False
    ) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def validate_b7873200_target_cache(
    cache_root: str | os.PathLike[str], identity: Mapping[str, object]
) -> None:
    """Require a complete train/validation-only versioned GeRaF cache."""

    expected = b7873200_sealed_protocol_identity(identity)
    root = Path(cache_root)
    sidecar = _read_json_mapping(
        root / B787_3200_TARGET_CACHE_PROTOCOL_FILENAME,
        "target-cache protocol sidecar",
    )
    expected_sidecar: Dict[str, object] = {
        "version": 1,
        "kind": "geraf_b7873200_sealed_target_cache",
        "sealed_npz_protocol_contract": expected,
    }
    if sidecar != expected_sidecar:
        raise ValueError("B7873200 target-cache protocol sidecar does not match the supplied manifest")
    legacy_artifacts = (
        root / "target_contract.json",
        root / "target_manifest.json",
        root / "power_stats.json",
        root / "views",
    )
    present_legacy = [str(path.name) for path in legacy_artifacts if path.exists()]
    if present_legacy:
        raise ValueError(
            "B7873200 corrected cache must not share a root with legacy GeRaF artifacts: "
            + ", ".join(present_legacy)
        )
    recipe = _read_json_mapping(root / B787_3200_CACHE_RECIPE_FILENAME, "cache recipe")
    _validated_b7873200_recipe(recipe, expected)
    if not (root / B787_3200_CACHE_ACQUISITION_FILENAME).is_file():
        raise FileNotFoundError("B7873200 target cache is missing its direct acquisition record")
    roles = expected["role_ids"]
    if not isinstance(roles, Mapping):
        raise AssertionError("B7873200 semantic identity unexpectedly lacks role IDs")
    target_manifest = _read_json_mapping(
        root / B787_3200_CACHE_MANIFEST_FILENAME, "target manifest"
    )
    if target_manifest.get("schema") != B787_3200_CACHE_SCHEMA or target_manifest.get("version") != 1:
        raise ValueError("B7873200 target manifest has the wrong schema/version")
    manifest_roles = target_manifest.get("roles")
    if not isinstance(manifest_roles, Mapping) or set(manifest_roles) != {"train", "validation"}:
        raise ValueError("B7873200 target manifest must expose exactly train and validation targets")
    for role in ("train", "validation"):
        observed = manifest_roles.get(role)
        if observed != roles[role]:
            raise ValueError(f"B7873200 target manifest role {role!r} disagrees with the sealed manifest")
    stats = _read_json_mapping(root / B787_3200_CACHE_STATS_FILENAME, "target stats")
    if stats.get("schema") != B787_3200_CACHE_SCHEMA or stats.get("version") != 1:
        raise ValueError("B7873200 target stats have the wrong schema/version")
    if stats.get("fit_split") != "train" or stats.get("clip") is not False:
        raise ValueError("B7873200 target stats must use the unclipped train-only normalizer")
    peak = stats.get("geraf_mf_magnitude_peak")
    if isinstance(peak, bool) or not isinstance(peak, (int, float, np.integer, np.floating)):
        raise ValueError("B7873200 target stats lack a numeric GeRaF peak")
    if not np.isfinite(float(peak)) or float(peak) <= 0.0:
        raise ValueError("B7873200 target stats have an invalid GeRaF peak")
    view_root = root / B787_3200_CACHE_VIEW_DIRECTORY
    for role in ("train", "validation"):
        for index in roles[role]:
            if not (view_root / f"view_{int(index):06d}.npz").is_file():
                raise ValueError(f"B7873200 target cache is missing {role} view {int(index)}")
    forbidden = list(roles["reserved_test"]) + list(roles["unused"])
    for index in forbidden:
        if (view_root / f"view_{int(index):06d}.npz").exists():
            raise ValueError(
                "B7873200 target cache contains a sealed response-derived target; "
                "use a fresh development cache"
            )
