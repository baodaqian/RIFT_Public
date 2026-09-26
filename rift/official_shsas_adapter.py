"""Small adapter for running the official SH-SAS trainer on an RIFT cache."""

from pathlib import Path

import numpy as np
import torch

from rift.sas_dataset import load_sas_cache


def load_training_cache(cache_path: str) -> dict:
    cache = load_sas_cache(cache_path)
    if not cache.has_explicit_splits:
        raise ValueError("official SH-SAS adaptation requires explicit cache splits")
    if cache.manifest.get("dataset_identity") != "airsas_armadillo_5khz":
        raise ValueError("official SH-SAS adaptation requires dataset_identity='airsas_armadillo_5khz'")

    rows = cache.train_indices
    tx_vecs = None if cache.tx_vecs is None else np.asarray(cache.tx_vecs[rows])
    return {
        "weights": np.asarray(cache.weights[rows]),
        "tx_coords": np.asarray(cache.tx_coords[rows]),
        "rx_coords": np.asarray(cache.rx_coords[rows]),
        "tx_vecs": tx_vecs,
        "source_ids": np.asarray(cache.source_ids[rows]),
        "radii": np.asarray(cache.radii),
        "corners": np.asarray(cache.corners),
        "voxels": np.asarray(cache.voxels),
        "grid_shape": tuple(int(value) for value in cache.manifest["geometry_grid_shape"]),
        "sound_speed": float(cache.manifest["sound_speed_mps"]),
        "crop": dict(cache.manifest["crop"]),
        "cache_path": str(Path(cache_path).resolve()),
    }


def receiver_query_coordinates(
    rx_pos: torch.Tensor, scene_scale_factor, normalize_scene_dims: bool
) -> torch.Tensor:
    if normalize_scene_dims:
        return rx_pos * scene_scale_factor
    return rx_pos
