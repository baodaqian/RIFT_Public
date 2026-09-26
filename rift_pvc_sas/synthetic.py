"""Synthetic sonar caches in the on-disk format of ``rift.sas_dataset.load_sas_cache`` (Package G).

Used by the PVC tests, the XPU op probe and the synthetic pre-smoke. The
identity is ``synthetic_sas_rings_v1``; it is *not* an AirSAS identity and
``scripts/validate_airsas_5k_cache.py`` rejects it by design, so a synthetic
cache can never be mistaken for the Armadillo5k comparison input.

The layout mirrors AirSAS: ``num_rings`` elevation rings of ``ring_size``
azimuth pings on a circle of radius ``sensor_radius`` around the scene box, a
small Tx/Rx separation along the ring, ``tx_vecs`` pointing at the box centre,
``radii`` as two-way path lengths covering the box, and ring-residue splits:
rings ``i % 10 < 8`` train, ``== 8`` validation, ``== 9`` test when
``num_rings`` is a multiple of ten (120 rings reproduce the Armadillo5k
96/12/12 ring counts), otherwise ``i % 3`` = 0/1/2. Weights are ``complex64``
pseudo-random values under a smooth radial envelope from
``numpy.random.default_rng(seed)``; nothing is measured or physical.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

IDENTITY = "synthetic_sas_rings_v1"
PRODUCTION_SHAPE = dict(num_rings=120, ring_size=360, num_bins=556, grid_shape=(150, 150, 120))
DEFAULT_BOX = ((-0.2, -0.2, 0.0), (0.2, 0.2, 0.3))


def ring_split(num_rings: int, ring_size: int) -> dict:
    if num_rings < 3:
        raise ValueError("a synthetic cache needs at least three rings (train/validation/test)")
    modulus = 10 if num_rings % 10 == 0 else 3
    train_res = list(range(8)) if modulus == 10 else [0]
    val_res, test_res = ([8], [9]) if modulus == 10 else ([1], [2])
    roles = {"train": [], "validation": [], "test": []}
    ring_counts = {"train_rings": 0, "validation_rings": 0, "test_rings": 0}
    for ring in range(num_rings):
        residue = ring % modulus
        role = "train" if residue in train_res else "validation" if residue in val_res else "test"
        roles[role].extend(range(ring * ring_size, (ring + 1) * ring_size))
        ring_counts[f"{role}_rings"] += 1
    return {
        "explicit": True,
        "train_indices": roles["train"],
        "validation_indices": roles["validation"],
        "test_indices": roles["test"],
        "train_ring_residue": f"ring % {modulus} in {train_res}",
        "validation_ring_residue": f"ring % {modulus} in {val_res}",
        "test_ring_residue": f"ring % {modulus} in {test_res}",
        "counts": ring_counts,
    }


def ring_geometry(num_rings: int, ring_size: int, *, sensor_radius: float = 0.85,
                  box=DEFAULT_BOX, separation: float = 0.05):
    lo, hi = (np.asarray(v, dtype=np.float64) for v in box)
    centre = (lo + hi) / 2.0
    heights = np.linspace(0.05, 0.55, num_rings)
    azimuth = np.linspace(0.0, 2.0 * np.pi, ring_size, endpoint=False)
    tx = np.zeros((num_rings * ring_size, 3)); rx = np.zeros_like(tx)
    for ring, z in enumerate(heights):
        rows = slice(ring * ring_size, (ring + 1) * ring_size)
        tx[rows, 0] = centre[0] + sensor_radius * np.cos(azimuth)
        tx[rows, 1] = centre[1] + sensor_radius * np.sin(azimuth)
        tx[rows, 2] = z
        tangent = np.stack((-np.sin(azimuth), np.cos(azimuth), np.zeros_like(azimuth)), axis=-1)
        rx[rows] = tx[rows] + separation * tangent
    tx_vecs = centre[None, :] - tx
    tx_vecs /= np.linalg.norm(tx_vecs, axis=-1, keepdims=True)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return tx.astype(np.float32), rx.astype(np.float32), tx_vecs.astype(np.float32), corners.astype(np.float32)


def two_way_radii(tx: np.ndarray, rx: np.ndarray, corners: np.ndarray, num_bins: int) -> np.ndarray:
    d_tx = np.linalg.norm(tx[:, None, :] - corners[None, :, :], axis=-1)
    d_rx = np.linalg.norm(rx[:, None, :] - corners[None, :, :], axis=-1)
    path = d_tx + d_rx
    return np.linspace(0.98 * float(path.min()), 1.02 * float(path.max()), num_bins).astype(np.float32)


def voxel_lattice(grid_shape: Sequence[int], box=DEFAULT_BOX) -> np.ndarray:
    lo, hi = (np.asarray(v, dtype=np.float64) for v in box)
    axes = [np.linspace(lo[i], hi[i], int(grid_shape[i])) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid.astype(np.float32)


def write_synthetic_cache(root, *, num_rings: int = 3, ring_size: int = 4, num_bins: int = 8,
                          grid_shape: Sequence[int] = (4, 4, 3), seed: int = 0,
                          sensor_radius: float = 0.85, box=DEFAULT_BOX, overwrite: bool = False) -> Path:
    """Write ``manifest.json``, ``weights.npy`` and ``geometry.npz``; return ``root``."""
    root = Path(root)
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("complete") is True and existing.get("dataset_identity") == IDENTITY:
            return root
        raise RuntimeError(f"refusing to overwrite a non-synthetic or incomplete cache at {root}")
    root.mkdir(parents=True, exist_ok=True)
    tx, rx, tx_vecs, corners = ring_geometry(num_rings, ring_size, sensor_radius=sensor_radius, box=box)
    radii = two_way_radii(tx, rx, corners, num_bins)
    num_pings = num_rings * ring_size
    rng = np.random.default_rng(seed)
    envelope = np.exp(-(((np.arange(num_bins) - num_bins / 2.0) / max(num_bins / 4.0, 1.0)) ** 2))
    weights = (rng.standard_normal((num_pings, num_bins)) + 1j * rng.standard_normal((num_pings, num_bins)))
    weights = (1.0e-2 * weights * envelope[None, :]).astype(np.complex64)
    split = ring_split(num_rings, ring_size)
    rmin, rmax = float(radii[0]), float(radii[-1])
    manifest = {
        "complete": True,
        "cache_contract_version": 2,
        "dataset_identity": IDENTITY,
        "frontend_asset_identity": "synthetic/none",
        "scene": "synthetic",
        "bandwidth_khz": 5.0,
        "num_pings": num_pings,
        "num_bins": num_bins,
        "original_num_bins": num_bins,
        "ring_size": ring_size,
        "num_rings": num_rings,
        "ring_protocol": "synthetic_complete_rings",
        "held_out_protocol": "ring_residue",
        "split_contract": split,
        "frontend_normalization": {"mode": "train_only", "strict_train_only": True,
                                   "supplied_scale": 1.0, "max_transmissions": num_pings},
        "aggressive_crop": {"original_min_dist": rmin, "original_max_dist": rmax, "new_min_sample": 0,
                            "new_max_sample_exclusive": num_bins, "cropped_min_dist": rmin, "cropped_max_dist": rmax},
        "crop": {"min_sample": 0, "min_dist": rmin, "max_dist": rmax, "num_samples": num_bins},
        "sound_speed_mps": 343.0,
        "sample_rate_hz": 100000.0,
        "geometry_grid_shape": [int(v) for v in grid_shape],
        "synthetic": {"seed": seed, "sensor_radius": sensor_radius, "box": [list(map(float, b)) for b in box],
                      "generator": "rift_pvc_sas.synthetic.write_synthetic_cache"},
    }
    np.save(root / "weights.npy", weights)
    np.savez(
        root / "geometry.npz",
        tx_coords=tx, rx_coords=rx, tx_vecs=tx_vecs, radii=radii, corners=corners,
        voxels=voxel_lattice(grid_shape, box), grid_shape=np.asarray(grid_shape, dtype=np.int64),
        source_ids=np.arange(num_pings, dtype=np.int64),
        train_indices=np.asarray(split["train_indices"], dtype=np.int64),
        validation_indices=np.asarray(split["validation_indices"], dtype=np.int64),
        test_indices=np.asarray(split["test_indices"], dtype=np.int64),
    )
    tmp = manifest_path.with_name("manifest.json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(manifest_path)
    return root


def main(argv=None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Write a synthetic sonar cache (see module docstring).")
    parser.add_argument("--output", required=True)
    parser.add_argument("--production-shape", action="store_true",
                        help="120 rings x 360 pings, 556 bins, 150x150x120 voxels (the Armadillo5k shape)")
    parser.add_argument("--num-rings", type=int, default=3)
    parser.add_argument("--ring-size", type=int, default=4)
    parser.add_argument("--num-bins", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    shape = dict(PRODUCTION_SHAPE) if args.production_shape else dict(
        num_rings=args.num_rings, ring_size=args.ring_size, num_bins=args.num_bins)
    root = write_synthetic_cache(args.output, seed=args.seed, **shape)
    print(f"synthetic cache ready: {root}")


if __name__ == "__main__":
    main()
