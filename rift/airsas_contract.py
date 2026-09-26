"""Shared identity and metadata checks for the supported AirSAS5k packages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


AIR_SAS_5K_IDENTITY = "airsas_armadillo_5khz"
EXPECTED_SYSTEM_DATA_NAME = "system_data_arma_5k.pik"
EXPECTED_TX_SHAPE = (43200, 3)
EXPECTED_CROP_BINS = 556
EXPECTED_GEOMETRY_SHAPE = (150, 150, 120)
EXPECTED_BANDWIDTH_KHZ = 5.0
RING_SIZE = 360
NUM_RINGS = 120
EXPECTED_RING_COUNTS = {"train_rings": 96, "validation_rings": 12, "test_rings": 12}
EXPECTED_PING_COUNTS = {"train_indices": 34560, "validation_indices": 4320, "test_indices": 4320}


def scene_contract_5k(scene: str) -> dict[str, Any]:
    """Return the exact configuration contract for one supported AirSAS5k scene."""
    contracts = {
        "armadillo": {
            "dataset_identity": "airsas_armadillo_5khz",
            "system_data_name": "system_data_arma_5k.pik",
            "system_tx_shape": (43200, 3),
            "crop_bins": 556,
            "frontend_asset_identity": "arma_5k/5khz_bw_lfm.npy",
        },
        "bunny": {
            "dataset_identity": "airsas_bunny_5khz",
            "system_data_name": "system_data_bunny_5k.pik",
            "system_tx_shape": (54000, 3),
            "crop_bins": 608,
            "frontend_asset_identity": "bunny_5k/5khz_bw_lfm.npy",
        },
    }
    if scene not in contracts:
        raise ValueError(f"unsupported AirSAS5k scene {scene!r}; expected 'armadillo' or 'bunny'")
    return dict(contracts[scene])


def _schema_dict(value: Any) -> Mapping[str, Any]:
    raw = value._data if hasattr(value, "_data") else value
    if not isinstance(raw, Mapping):
        raise ValueError(f"expected schema dictionary, got {type(value)!r}")
    return raw


def validate_system_data_5k(
    path: str | Path, system: Mapping[str, Any], scene: str = "armadillo"
) -> dict[str, Any]:
    """Validate one supported AirSAS5k system-data identity and return its metadata."""
    scene_info = scene_contract_5k(scene)
    path = Path(path)
    if path.name != scene_info["system_data_name"]:
        raise ValueError(f"expected {scene_info['system_data_name']}, got {path.name}")
    tx = np.asarray(system["tx_coords"])
    rx = np.asarray(system["rx_coords"])
    expected_shape = scene_info["system_tx_shape"]
    if tx.shape != expected_shape or rx.shape != expected_shape:
        raise ValueError(f"expected {scene} AirSAS5k Tx/Rx shape {expected_shape}, got {tx.shape}/{rx.shape}")
    crop = _schema_dict(system["crop_settings"])
    bins = int(crop["num_samples"])
    if bins != scene_info["crop_bins"]:
        raise ValueError(f"expected {scene} AirSAS5k crop bins {scene_info['crop_bins']}, got {bins}")
    geometry = _schema_dict(system["geometry"])
    grid_shape = tuple(int(geometry[key]) for key in ("num_x", "num_y", "num_z"))
    if grid_shape != EXPECTED_GEOMETRY_SHAPE:
        raise ValueError(f"expected AirSAS5k geometry {EXPECTED_GEOMETRY_SHAPE}, got {grid_shape}")
    return {
        "identity": scene_info["dataset_identity"],
        "tx_shape": list(tx.shape),
        "crop_bins": bins,
        "geometry_grid_shape": list(grid_shape),
        "bandwidth_khz": EXPECTED_BANDWIDTH_KHZ,
        "frontend_asset_identity": scene_info["frontend_asset_identity"],
    }


def validate_cache_5k(root: str | Path) -> dict[str, Any]:
    """Validate a complete prepared AirSAS5k cache, including array shapes."""
    root = Path(root)
    with (root / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    scene_info = scene_contract_5k(manifest.get("scene"))
    expected = {
        "complete": True,
        "dataset_identity": scene_info["dataset_identity"],
        "frontend_asset_identity": scene_info["frontend_asset_identity"],
        "scene": manifest.get("scene"),
        "bandwidth_khz": EXPECTED_BANDWIDTH_KHZ,
        "num_pings": EXPECTED_TX_SHAPE[0],
        "original_num_bins": scene_info["crop_bins"],
        "ring_size": RING_SIZE,
        "num_rings": NUM_RINGS,
        "ring_protocol": "first_120_complete_elevation_rings",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"cache manifest {key} must be {value!r}, got {manifest.get(key)!r}")
    split = manifest.get("split_contract")
    if not isinstance(split, Mapping) or split.get("explicit") is not True:
        raise ValueError("AirSAS5k cache must carry explicit train/validation/test splits")
    if split.get("counts") != EXPECTED_RING_COUNTS:
        raise ValueError(f"AirSAS5k ring split counts must be {EXPECTED_RING_COUNTS!r}")
    for key, expected_count in EXPECTED_PING_COUNTS.items():
        values = split.get(key)
        if not isinstance(values, list) or len(values) != expected_count:
            raise ValueError(f"AirSAS5k {key} must contain {expected_count} rows")
    normalization = manifest.get("frontend_normalization")
    if not isinstance(normalization, Mapping) or normalization.get("mode") != "train_only" or normalization.get("strict_train_only") is not True:
        raise ValueError("AirSAS5k comparison cache must use strict train-only normalization")
    arrays = np.load(root / "geometry.npz", allow_pickle=False)
    weights = np.load(root / "weights.npy", mmap_mode="r", allow_pickle=False)
    if weights.ndim != 2 or weights.shape[0] != EXPECTED_TX_SHAPE[0]:
        raise ValueError(f"AirSAS5k weights must have {EXPECTED_TX_SHAPE[0]} pings, got {weights.shape}")
    if arrays["tx_coords"].shape != EXPECTED_TX_SHAPE or arrays["rx_coords"].shape != EXPECTED_TX_SHAPE:
        raise ValueError("AirSAS5k cache Tx/Rx arrays do not match the 43,200-ping contract")
    if tuple(int(value) for value in arrays["grid_shape"]) != EXPECTED_GEOMETRY_SHAPE:
        raise ValueError("AirSAS5k cache geometry grid does not match 150x150x120")
    if arrays["radii"].shape != (weights.shape[1],):
        raise ValueError("AirSAS5k cache radii do not match weights")
    return manifest
