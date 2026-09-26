"""Bounded real-B787 cache contract for the GeRaF 16/4 engineering fit.

This opt-in lane intentionally does not alter the complete 3,200-train /
1,000-validation B787 cache contract.  It has a separate cache schema, a
separate completion sidecar, and a narrowed preparation capability that can
read exactly the approved parent ``train[:16]`` and ``validation[:4]`` rows.
The fit path consumes a metadata-only source and the prepared native ``|MF|``
leaves; it never opens a raw radar response.
"""

from __future__ import annotations

import copy
import json
import math
import os
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import train_geraf as trainer
from rift.geraf_b7873200_acquisition import (
    validate_b7873200_acquisition_record,
    validate_b7873200_operator_frequency_grid,
    write_or_validate_b7873200_acquisition_record,
)
from rift.geraf_b7873200_protocol import (
    B787_3200_CACHE_ACQUISITION_FILENAME,
    B787_3200_CACHE_VIEW_DIRECTORY,
    B787_3200_CANONICAL_MANIFEST_PATH,
    B787_3200_CANONICAL_NPZ_PATH,
    B787_3200_NUM_TRAIN,
    B787_3200_NUM_VALIDATION,
    B787_3200_SEED,
    B787_3200_TARGET_CACHE_PROTOCOL_FILENAME,
    b7873200_sealed_protocol_identity,
)
from rift.geraf_b7873200_source import (
    B7873200DevelopmentSource,
    B7873200MetadataArrays,
    load_b7873200_development_source,
    load_b7873200_metadata_source,
)
from rift.power_baseline_dataset import atomic_write_json, frequency_grid_hz
from scripts import prepare_geraf_b7873200_targets as full_preparer


PREPARATION_ID = "geraf_b7873200_engineering_subset16x4_prepare_v1"
FIT_ID = "geraf_b7873200_engineering_subset16x4_fit_v1"
CACHE_SCHEMA = "rift_geraf_b7873200_engineering_subset16x4_native_mf_cache_v1"
CACHE_VERSION = 1
CACHE_RECIPE_FILENAME = "geraf_b78716_smallfit_recipe.json"
CACHE_MANIFEST_FILENAME = "geraf_b78716_smallfit_manifest.json"
CACHE_STATS_FILENAME = "geraf_b78716_smallfit_stats.json"
CACHE_PROTOCOL_FILENAME = "geraf_b78716_smallfit_protocol.json"
TARGET_DIRECTORY = B787_3200_CACHE_VIEW_DIRECTORY

NUM_TRAIN = 16
NUM_VALIDATION = 4
MAX_UPDATES = 32
SCHEDULER_HORIZON_UPDATES = 50_000
MILESTONE_UPDATES = (0, 16, 32)
TARGET_GRID_SHAPE = (8, 8, 8)


@dataclass(frozen=True)
class B78716SmallfitWorklists:
    """The only response-derived IDs permitted by this engineering cache."""

    train: tuple[int, ...]
    validation: tuple[int, ...]

    @property
    def selected(self) -> tuple[int, ...]:
        return self.train + self.validation

    def as_dict(self) -> dict[str, object]:
        return {
            "parent_train_prefix": NUM_TRAIN,
            "parent_validation_prefix": NUM_VALIDATION,
            "train": [int(index) for index in self.train],
            "validation": [int(index) for index in self.validation],
            "selected_target_count": len(self.selected),
        }


def _ordered_parent_role(
    parent_identity: Mapping[str, object], role: str, expected_count: int
) -> tuple[int, ...]:
    roles = parent_identity.get("role_ids")
    if not isinstance(roles, Mapping):
        raise ValueError("B787 parent identity lacks ordered role IDs")
    raw = roles.get(role)
    if not isinstance(raw, list) or len(raw) != int(expected_count):
        raise ValueError(f"B787 parent identity has invalid {role!r} role length")
    values: list[int] = []
    for value in raw:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"B787 parent identity has a non-integer {role!r} ID")
        values.append(int(value))
    if len(set(values)) != len(values):
        raise ValueError(f"B787 parent identity has duplicate {role!r} IDs")
    return tuple(values)


def bounded_worklists(parent_identity: Mapping[str, object]) -> B78716SmallfitWorklists:
    """Derive the frozen ordered parent prefixes without reading a response."""

    canonical = b7873200_sealed_protocol_identity(parent_identity)
    train = _ordered_parent_role(canonical, "train", B787_3200_NUM_TRAIN)
    validation = _ordered_parent_role(canonical, "validation", B787_3200_NUM_VALIDATION)
    worklists = B78716SmallfitWorklists(
        train=train[:NUM_TRAIN], validation=validation[:NUM_VALIDATION]
    )
    if len(worklists.train) != NUM_TRAIN or len(worklists.validation) != NUM_VALIDATION:
        raise AssertionError("canonical B787 roles no longer support the bounded GeRaF worklists")
    return worklists


def subset_contract(parent_identity: Mapping[str, object]) -> dict[str, object]:
    """Record the full parent policy plus the explicit bounded operation IDs."""

    parent = b7873200_sealed_protocol_identity(parent_identity)
    worklists = bounded_worklists(parent)
    return {
        "schema": CACHE_SCHEMA,
        "version": CACHE_VERSION,
        "preparation_id": PREPARATION_ID,
        "fit_id": FIT_ID,
        "scope": "bounded_engineering_16train_4validation_not_production",
        "parent_sealed_protocol_identity": parent,
        "engineering_subset": worklists.as_dict(),
    }


def _readonly_copy(values: object, *, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


_PREPARATION_READERS: weakref.WeakKeyDictionary[
    object, B7873200DevelopmentSource
] = weakref.WeakKeyDictionary()


@dataclass(frozen=True, eq=False)
class BoundedB78716PreparationSource:
    """Metadata plus a private, selected-row-only raw-response capability."""

    arrays: B7873200MetadataArrays
    parent_identity: dict[str, object]
    worklists: B78716SmallfitWorklists

    @property
    def identity(self) -> dict[str, object]:
        """Return a copy of the complete parent policy, never a forged split."""

        return copy.deepcopy(self.parent_identity)

    @property
    def allowed_response_view_indices(self) -> frozenset[int]:
        return frozenset(self.worklists.selected)

    def response_view(self, index: int) -> np.ndarray:
        """Return exactly one selected chirp-averaged response row."""

        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            raise ValueError("B78716 preparation view ID must be an integer")
        normalized = int(index)
        if normalized not in self.allowed_response_view_indices:
            raise PermissionError(
                "bounded GeRaF preparation may read only parent train[:16] and "
                "validation[:4] response rows"
            )
        source = _PREPARATION_READERS.get(self)
        if source is None:
            raise RuntimeError("bounded GeRaF preparation reader is unavailable")
        return source.response_view(normalized)


def build_bounded_preparation_source(
    source: B7873200DevelopmentSource,
) -> BoundedB78716PreparationSource:
    """Wrap the broad development adapter before target preparation begins."""

    parent_identity = b7873200_sealed_protocol_identity(source.identity)
    worklists = bounded_worklists(parent_identity)
    parent_arrays = source.arrays
    arrays = B7873200MetadataArrays(
        path=str(parent_arrays.path),
        response=parent_arrays.response,
        viewpoint_positions=_readonly_copy(parent_arrays.viewpoint_positions, dtype=np.float64),
        tx_pos=_readonly_copy(parent_arrays.tx_pos, dtype=np.float64),
        rx_pos=_readonly_copy(parent_arrays.rx_pos, dtype=np.float64),
        metadata=copy.deepcopy(dict(parent_arrays.metadata)),
    )
    bounded = BoundedB78716PreparationSource(
        arrays=arrays,
        parent_identity=copy.deepcopy(parent_identity),
        worklists=worklists,
    )
    _PREPARATION_READERS[bounded] = source
    return bounded


def load_bounded_preparation_source(
    npz_path: str | os.PathLike[str] = B787_3200_CANONICAL_NPZ_PATH,
    role_manifest: str | os.PathLike[str] = B787_3200_CANONICAL_MANIFEST_PATH,
) -> BoundedB78716PreparationSource:
    """Preflight the parent archive then expose the narrowed 20-row adapter."""

    return build_bounded_preparation_source(
        load_b7873200_development_source(npz_path, role_manifest)
    )


def target_args(*, device: str) -> SimpleNamespace:
    """Return the frozen 8-cubed native-MF target recipe arguments."""

    return SimpleNamespace(
        npz_path=B787_3200_CANONICAL_NPZ_PATH,
        role_manifest=B787_3200_CANONICAL_MANIFEST_PATH,
        cache_root="",
        scene_extent=0.15,
        n_azimuth=TARGET_GRID_SHAPE[1],
        n_elevation=TARGET_GRID_SHAPE[0],
        n_depth=TARGET_GRID_SHAPE[2],
        aperture_scale=1.0,
        phase_sign=-1.0,
        backend="range",
        device=str(device),
        compute_dtype="float64",
        point_chunk=512,
        pair_chunk=32,
        freq_chunk=75,
        kernel_width=20,
        oversample=2,
        resume=True,
        max_views=0,
    )


def training_args(*, device: str) -> SimpleNamespace:
    """Return the existing GeRaF model/optimizer recipe with an unshrunk clock."""

    args = target_args(device=device)
    args.seed = B787_3200_SEED
    args.sdf_levels = 10
    args.sdf_hidden_dim = 256
    args.sdf_layers = 8
    args.sdf_skip_layer = 4
    args.sdf_softplus_beta = 100.0
    args.reflectivity_levels = 0
    args.reflectivity_hidden_dim = 256
    args.reflectivity_layers = 4
    args.reflectivity_output_activation = "softplus"
    args.reflectivity_softplus_beta = 1.0
    args.init_tx_amplitude = 1.0
    args.init_inv_s = 64.0
    args.learnable_inv_s = True
    args.lensless_correction = True
    args.detach_start_cdf = True
    args.directional_exponent = 1.0
    args.min_distance = 1.0e-6
    # This is the original science scheduler horizon, not the 32-update stop
    # budget.  The bounded driver calls build_geraf_scheduler with this value.
    args.steps = SCHEDULER_HORIZON_UPDATES
    args.sdf_lr = 1.0e-4
    args.other_lr = 1.0e-3
    args.weight_decay = 1.0e-2
    args.adam_beta1 = 0.9
    args.adam_beta2 = 0.999
    args.adam_eps = 1.0e-8
    args.cosine_min_lr = 0.0
    args.gradient_clip_norm = 0.0
    args.mask_high_threshold = 0.1
    args.mask_low_ratio = 0.1
    args.mask_low_threshold = 0.0
    args.validation_every = 16
    args.checkpoint_every = 1
    args.checkpoint_seconds = 300.0
    args.log_every = 1
    args.resume = True
    args.resume_path = None
    args.checkpoint_dir = ""
    return args


def _read_json(path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"bounded GeRaF {label} is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"bounded GeRaF {label} must be a JSON object")
    return payload


def _write_or_require_equal(path: Path, payload: Mapping[str, object], label: str) -> None:
    expected = dict(payload)
    if path.exists():
        if _read_json(path, label) != expected:
            raise ValueError(f"bounded GeRaF {label} disagrees with this exact engineering identity")
        return
    atomic_write_json(path, expected)


def _view_path(cache_root: Path, index: int) -> Path:
    return cache_root / TARGET_DIRECTORY / f"view_{int(index):06d}.npz"


def _target_spec(args: SimpleNamespace) -> dict[str, object]:
    """Reuse the full preparer's exact native target-spec construction."""

    return dict(full_preparer._target_spec(args))


def _cache_recipe(
    parent_identity: Mapping[str, object], target_spec: Mapping[str, object]
) -> dict[str, object]:
    contract = subset_contract(parent_identity)
    return {
        "schema": CACHE_SCHEMA,
        "version": CACHE_VERSION,
        "preparation_id": PREPARATION_ID,
        "fit_id": FIT_ID,
        "scope": contract["scope"],
        "parent_sealed_protocol_identity": contract["parent_sealed_protocol_identity"],
        "engineering_subset": contract["engineering_subset"],
        "response_roles_materialized": ["train", "validation"],
        "acquisition_record": {
            "filename": B787_3200_CACHE_ACQUISITION_FILENAME,
            "kind": "full_calibrated_B787_acquisition_record",
        },
        "target_spec": dict(target_spec),
    }


def _manifest(worklists: B78716SmallfitWorklists) -> dict[str, object]:
    return {
        "schema": CACHE_SCHEMA,
        "version": CACHE_VERSION,
        "kind": "native_mf_targets",
        "roles": {
            "train": [int(index) for index in worklists.train],
            "validation": [int(index) for index in worklists.validation],
        },
        "selected_target_count": len(worklists.selected),
    }


def _stats(train_peak: float) -> dict[str, object]:
    if not math.isfinite(float(train_peak)) or float(train_peak) <= 0.0:
        raise ValueError("bounded GeRaF train-only native |MF| peak must be finite and positive")
    return {
        "schema": CACHE_SCHEMA,
        "version": CACHE_VERSION,
        "kind": "train_normalization",
        "fit_split": "train",
        "selected_train_count": NUM_TRAIN,
        "geraf_mf_magnitude_peak": float(train_peak),
        "clip": False,
    }


def _protocol(parent_identity: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema": CACHE_SCHEMA,
        "version": CACHE_VERSION,
        "kind": "geraf_b78716_smallfit_complete_target_cache",
        "subset_contract": subset_contract(parent_identity),
    }


def _expected_target_paths(worklists: B78716SmallfitWorklists, cache_root: Path) -> set[Path]:
    return {_view_path(cache_root, index) for index in worklists.selected}


def _reject_foreign_or_incomplete_root(
    cache_root: Path, worklists: B78716SmallfitWorklists
) -> None:
    """Reject cache-root mixing before a raw response is requested."""

    forbidden = (
        B787_3200_TARGET_CACHE_PROTOCOL_FILENAME,
        "target_contract.json",
        "target_manifest.json",
        "power_stats.json",
        "views",
    )
    present = [cache_root / name for name in forbidden if (cache_root / name).exists()]
    if present:
        raise ValueError(
            "bounded GeRaF cache must not share a root with the production or legacy lane: "
            f"{present[0]}"
        )
    view_root = cache_root / TARGET_DIRECTORY
    if view_root.exists() and not view_root.is_dir():
        raise ValueError("bounded GeRaF target directory is not a directory")
    if view_root.is_dir():
        expected = _expected_target_paths(worklists, cache_root)
        observed = {path for path in view_root.iterdir() if path.is_file()}
        extras = observed.difference(expected)
        if extras:
            raise ValueError(
                "bounded GeRaF cache contains a target outside the approved 16/4 worklist: "
                f"{sorted(extras)[0]}"
            )


def _validate_target_leaves(
    *,
    cache_root: Path,
    source: B7873200DevelopmentSource,
    recipe: Mapping[str, object],
    target_spec: Mapping[str, object],
    args: SimpleNamespace,
    worklists: B78716SmallfitWorklists,
) -> float:
    """Validate the exact 20 payloads and return the train-only peak."""

    _reject_foreign_or_incomplete_root(cache_root, worklists)
    expected_paths = _expected_target_paths(worklists, cache_root)
    observed_paths = {
        path for path in (cache_root / TARGET_DIRECTORY).iterdir() if path.is_file()
    } if (cache_root / TARGET_DIRECTORY).is_dir() else set()
    if observed_paths != expected_paths:
        missing = expected_paths.difference(observed_paths)
        extra = observed_paths.difference(expected_paths)
        detail = sorted(missing or extra)[0]
        raise ValueError(f"bounded GeRaF target inventory is not exactly 20 leaves: {detail}")

    expected_shape = (int(args.n_elevation), int(args.n_azimuth), int(args.n_depth))
    compatibility_cache = SimpleNamespace(
        root=cache_root,
        recipe=dict(recipe),
        target_recipe_schemas=(CACHE_SCHEMA,),
    )
    peak = 0.0
    for role, indices in (("train", worklists.train), ("validation", worklists.validation)):
        for index in indices:
            # This reuses the leaf-level payload/spec validation from the full
            # preparer, then the compact calibration-frame check from the full
            # trainer.  Neither call opens a raw response.
            magnitude = full_preparer._validate_target(
                _view_path(cache_root, index),
                index=index,
                role=role,
                target_spec=target_spec,
                expected_shape=expected_shape,
            )
            target = trainer._load_target(index, role, compatibility_cache)
            trainer._require_frozen_geometry(
                target=target,
                view_index=index,
                arrays=source.arrays,
                args=args,
            )
            if role == "train":
                peak = max(peak, float(np.max(magnitude)))
    if not math.isfinite(peak) or peak <= 0.0:
        raise ValueError("bounded GeRaF 16-target training peak is invalid")
    return peak


@dataclass(frozen=True)
class PreparedB78716SmallfitCache:
    """Metadata-only compatible facade consumed by existing trainer helpers."""

    source: B7873200DevelopmentSource
    root: Path
    recipe: dict[str, object]
    stats: dict[str, object]
    target_manifest: dict[str, object]
    acquisition_record: dict[str, object]
    sealed_identity: dict[str, object]
    parent_sealed_identity: dict[str, object]
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    grid_shape: tuple[int, int, int]
    geraf_mf_magnitude_peak: float
    effective_pairs_per_plane: int
    target_recipe_schemas: tuple[str, ...]


def _validate_recipe(
    recipe: Mapping[str, object], parent_identity: Mapping[str, object], target_spec: Mapping[str, object]
) -> None:
    expected = _cache_recipe(parent_identity, target_spec)
    if dict(recipe) != expected:
        raise ValueError("bounded GeRaF cache recipe does not match the frozen 16/4 target contract")


def _validate_complete_subset_cache(
    *,
    cache_root: Path,
    source: B7873200DevelopmentSource,
    args: SimpleNamespace,
) -> PreparedB78716SmallfitCache:
    """Validate a completed engineering cache using only source metadata."""

    if not isinstance(source.arrays, B7873200MetadataArrays):
        raise TypeError("bounded GeRaF fit must use a metadata-only B787 source")
    if source.arrays.response_payload_materialized is not False:
        raise RuntimeError("bounded GeRaF fit source unexpectedly exposes raw response payloads")
    parent_identity = b7873200_sealed_protocol_identity(source.identity)
    worklists = bounded_worklists(parent_identity)
    target_spec = _target_spec(args)
    recipe = _read_json(cache_root / CACHE_RECIPE_FILENAME, "cache recipe")
    _validate_recipe(recipe, parent_identity, target_spec)
    protocol = _read_json(cache_root / CACHE_PROTOCOL_FILENAME, "completion protocol")
    if protocol != _protocol(parent_identity):
        raise ValueError("bounded GeRaF completion protocol differs from its parent 16/4 contract")
    manifest = _read_json(cache_root / CACHE_MANIFEST_FILENAME, "target manifest")
    expected_manifest = _manifest(worklists)
    if manifest != expected_manifest:
        raise ValueError("bounded GeRaF target manifest does not name exactly the 16/4 worklists")
    acquisition_record = validate_b7873200_acquisition_record(cache_root, source.arrays)
    peak = _validate_target_leaves(
        cache_root=cache_root,
        source=source,
        recipe=recipe,
        target_spec=target_spec,
        args=args,
        worklists=worklists,
    )
    stats = _read_json(cache_root / CACHE_STATS_FILENAME, "target statistics")
    expected_stats = _stats(peak)
    if stats != expected_stats:
        raise ValueError("bounded GeRaF statistics do not equal the train-only target peak")
    return PreparedB78716SmallfitCache(
        source=source,
        root=cache_root,
        recipe=recipe,
        stats=stats,
        target_manifest=manifest,
        acquisition_record=acquisition_record,
        sealed_identity=subset_contract(parent_identity),
        parent_sealed_identity=parent_identity,
        train_indices=worklists.train,
        validation_indices=worklists.validation,
        grid_shape=TARGET_GRID_SHAPE,
        geraf_mf_magnitude_peak=peak,
        effective_pairs_per_plane=16 * 16,
        target_recipe_schemas=(CACHE_SCHEMA,),
    )


def prepare_complete_subset_cache(
    *,
    cache_root: str | os.PathLike[str],
    npz_path: str | os.PathLike[str] = B787_3200_CANONICAL_NPZ_PATH,
    role_manifest: str | os.PathLike[str] = B787_3200_CANONICAL_MANIFEST_PATH,
    device: str,
) -> PreparedB78716SmallfitCache:
    """Create exactly 20 target leaves, then publish the complete cache contract.

    Only a completed manifest/protocol is passed to the later fit stage.  A
    clean re-invocation may validate and reuse finished leaves, but an
    incomplete cache has no fit-visible completion sidecar.
    """

    # The archive paths are provenance settings; bind their expected canonical
    # values in the target recipe rather than accepting alternate data.
    if os.path.normcase(os.path.abspath(os.fspath(npz_path))) != os.path.normcase(
        os.path.abspath(B787_3200_CANONICAL_NPZ_PATH)
    ):
        raise ValueError("bounded GeRaF preparation accepts only the canonical B787 /storage/home archive")
    if os.path.normcase(os.path.abspath(os.fspath(role_manifest))) != os.path.normcase(
        os.path.abspath(B787_3200_CANONICAL_MANIFEST_PATH)
    ):
        raise ValueError("bounded GeRaF preparation accepts only the canonical B787 role manifest")
    destination = Path(cache_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    preparation_source = load_bounded_preparation_source(npz_path, role_manifest)
    parent_identity = b7873200_sealed_protocol_identity(preparation_source.parent_identity)
    worklists = preparation_source.worklists
    args = target_args(device=device)
    target_spec = _target_spec(args)
    recipe = _cache_recipe(parent_identity, target_spec)
    _reject_foreign_or_incomplete_root(destination, worklists)
    _write_or_require_equal(destination / CACHE_RECIPE_FILENAME, recipe, "cache recipe")
    acquisition_record = write_or_validate_b7873200_acquisition_record(
        destination, preparation_source.arrays
    )
    operator_frequency = frequency_grid_hz(preparation_source.arrays.metadata)
    frequencies = torch.as_tensor(
        validate_b7873200_operator_frequency_grid(acquisition_record, operator_frequency),
        dtype=torch.float64,
        device=torch.device(device),
    )
    expected_shape = TARGET_GRID_SHAPE
    for ordinal, (role, index) in enumerate(
        ((role, index) for role, indices in (("train", worklists.train), ("validation", worklists.validation)) for index in indices),
        start=1,
    ):
        path = _view_path(destination, index)
        if path.exists():
            full_preparer._validate_target(
                path,
                index=index,
                role=role,
                target_spec=target_spec,
                expected_shape=expected_shape,
            )
            print(f"[{ordinal}/{len(worklists.selected)}] GeRaF target {index}: cached", flush=True)
            continue
        full_preparer._prepare_one(
            preparation_source,
            index,
            role,
            cache_root=destination,
            target_spec=target_spec,
            args=args,
            device=torch.device(device),
            frequencies=frequencies,
        )
        print(f"[{ordinal}/{len(worklists.selected)}] GeRaF target {index}: prepared", flush=True)

    # Build a metadata-only source only after target preparation is over.  The
    # complete cache gate therefore cannot accidentally retain a raw responder.
    metadata_source = load_b7873200_metadata_source(npz_path, role_manifest)
    peak = _validate_target_leaves(
        cache_root=destination,
        source=metadata_source,
        recipe=recipe,
        target_spec=target_spec,
        args=args,
        worklists=worklists,
    )
    _write_or_require_equal(destination / CACHE_MANIFEST_FILENAME, _manifest(worklists), "target manifest")
    _write_or_require_equal(destination / CACHE_STATS_FILENAME, _stats(peak), "target statistics")
    _write_or_require_equal(destination / CACHE_PROTOCOL_FILENAME, _protocol(parent_identity), "completion protocol")
    return _validate_complete_subset_cache(
        cache_root=destination, source=metadata_source, args=args
    )


def load_complete_subset_cache(
    *,
    cache_root: str | os.PathLike[str],
    npz_path: str | os.PathLike[str] = B787_3200_CANONICAL_NPZ_PATH,
    role_manifest: str | os.PathLike[str] = B787_3200_CANONICAL_MANIFEST_PATH,
    device: str,
) -> PreparedB78716SmallfitCache:
    """Open a completed bounded cache through the metadata-only fit path."""

    if os.path.normcase(os.path.abspath(os.fspath(npz_path))) != os.path.normcase(
        os.path.abspath(B787_3200_CANONICAL_NPZ_PATH)
    ):
        raise ValueError("bounded GeRaF fit accepts only the canonical B787 /storage/home archive")
    if os.path.normcase(os.path.abspath(os.fspath(role_manifest))) != os.path.normcase(
        os.path.abspath(B787_3200_CANONICAL_MANIFEST_PATH)
    ):
        raise ValueError("bounded GeRaF fit accepts only the canonical B787 role manifest")
    source = load_b7873200_metadata_source(npz_path, role_manifest)
    return _validate_complete_subset_cache(
        cache_root=Path(cache_root).expanduser().resolve(), source=source, args=target_args(device=device)
    )
