"""Safe loaders and cache contract for the public SH-SAS/Reed sonar data."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class RestrictedSASUnpickler(pickle.Unpickler):
    """Allow only the four schema containers and NumPy array machinery."""

    _numpy_allowed = {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
    }

    def find_class(self, module: str, name: str):
        if (module, name) in self._numpy_allowed:
            return super().find_class(module, name)
        if module == "data_schemas" and name in {
            "SASDataSchema",
            "SysParams",
            "WfmParams",
            "WfmCropSettings",
            "Geometry",
        }:
            # The official checkout is intentionally importable only while
            # loading.  Its schema file contains passive dict wrappers.
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"forbidden pickle global {module}.{name}")


def restricted_load_system_data(path: str | Path) -> Dict[str, Any]:
    """Load an official ``system_data.pik`` after the caller exposes data_schemas."""
    with Path(path).open("rb") as handle:
        value = RestrictedSASUnpickler(handle).load()
    raw = value._data if hasattr(value, "_data") else value
    if not isinstance(raw, dict):
        raise TypeError(f"unexpected system-data root {type(value)!r}")
    required = {"tx_coords", "rx_coords", "geometry", "crop_settings", "speed_of_sound"}
    missing = required - set(raw)
    if missing:
        raise KeyError(f"system-data file is missing {sorted(missing)}")
    return raw


def schema_dict(value: Any) -> Dict[str, Any]:
    raw = value._data if hasattr(value, "_data") else value
    if not isinstance(raw, dict):
        raise TypeError(f"expected schema dict, got {type(value)!r}")
    return raw


@dataclass(frozen=True)
class SASCache:
    root: Path
    weights: np.ndarray
    tx_coords: np.ndarray
    rx_coords: np.ndarray
    radii: np.ndarray
    corners: np.ndarray
    voxels: np.ndarray
    source_ids: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    tx_vecs: Optional[np.ndarray]
    manifest: Dict[str, Any]

    @property
    def num_pings(self) -> int:
        return int(self.weights.shape[0])

    @property
    def num_bins(self) -> int:
        return int(self.weights.shape[1])

    @property
    def has_explicit_splits(self) -> bool:
        return bool(self.manifest.get("split_contract", {}).get("explicit", False))


def _split_indices(manifest: Dict[str, Any], arrays, name: str, count: int) -> np.ndarray:
    aliases = {
        "train": ("train_indices", "train_ids"),
        "validation": ("validation_indices", "validation_ids", "val_indices", "val_ids"),
        "test": ("test_indices", "test_ids"),
    }[name]
    for key in aliases:
        if key in arrays:
            values = np.asarray(arrays[key], dtype=np.int64)
            break
    else:
        split = manifest.get("split_contract", {})
        values = None
        for key in aliases:
            if key in split:
                values = np.asarray(split[key], dtype=np.int64)
                break
        if values is None:
            return np.arange(count, dtype=np.int64) if name == "train" else np.empty(0, dtype=np.int64)
    if values.ndim != 1 or bool((values < 0).any()) or bool((values >= count).any()):
        raise ValueError(f"{name} split contains invalid row indices")
    if np.unique(values).size != values.size:
        raise ValueError(f"{name} split contains duplicate row indices")
    return values


def load_sas_cache(root: str | Path, mmap_mode: str = "r") -> SASCache:
    root = Path(root)
    with (root / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("complete") is not True:
        raise RuntimeError(f"SAS cache is not complete: {root}")
    arrays = np.load(root / "geometry.npz", allow_pickle=False)
    weights = np.load(root / "weights.npy", mmap_mode=mmap_mode, allow_pickle=False)
    if weights.ndim != 2 or not np.issubdtype(weights.dtype, np.complexfloating):
        raise ValueError("weights.npy must be a complex [ping,bin] array")
    tx = arrays["tx_coords"]
    rx = arrays["rx_coords"]
    radii = arrays["radii"]
    corners = arrays["corners"]
    voxels = arrays["voxels"]
    if tx.shape != rx.shape or tx.shape != (weights.shape[0], 3):
        raise ValueError("Tx/Rx cache shapes do not match weights")
    if radii.shape != (weights.shape[1],):
        raise ValueError("radius cache does not match weights")
    source_ids = np.asarray(arrays["source_ids"] if "source_ids" in arrays else np.arange(weights.shape[0]), dtype=np.int64)
    if source_ids.shape != (weights.shape[0],):
        raise ValueError("source_ids must have one entry per cached ping")
    tx_vecs = None
    if "tx_vecs" in arrays:
        tx_vecs = np.asarray(arrays["tx_vecs"], dtype=np.float32)
        if tx_vecs.shape != (weights.shape[0], 3) or not np.isfinite(tx_vecs).all():
            raise ValueError("tx_vecs must be finite with shape [num_pings,3]")
    train_indices = _split_indices(manifest, arrays, "train", weights.shape[0])
    validation_indices = _split_indices(manifest, arrays, "validation", weights.shape[0])
    test_indices = _split_indices(manifest, arrays, "test", weights.shape[0])
    roles = [train_indices, validation_indices, test_indices]
    if len(np.unique(np.concatenate(roles))) != sum(values.size for values in roles):
        raise ValueError("train/validation/test row-index splits overlap")
    return SASCache(
        root, weights, tx, rx, radii, corners, voxels, source_ids,
        train_indices, validation_indices, test_indices, tx_vecs, manifest,
    )


def atomic_json(payload: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)
