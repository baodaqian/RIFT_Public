#!/usr/bin/env python
"""Score a sonar density export with the public benchmark's 5-mm protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


SCENE_TO_GT = {
    "armadillo": "armadilo",
    "buddha": "budda",
    "bunny": "bunny",
    "xyz_dragon": "xyz_dragon",
}


def voxel_keys(points: np.ndarray, unit: float = 0.005) -> set[tuple[int, int, int]]:
    minimum = np.asarray([-0.2, -0.2, 0.0])
    index = np.floor((points - minimum) / unit).astype(np.int64)
    shape = np.asarray([int(0.4 / unit), int(0.4 / unit), int(0.3 / unit)])
    valid = ((index >= 0) & (index < shape)).all(axis=1)
    return {tuple(row) for row in index[valid]}


def chamfer_squared(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    da = cKDTree(b).query(a, workers=-1)[0]
    db = cKDTree(a).query(b, workers=-1)[0]
    return float(np.square(da).mean() + np.square(db).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density", required=True)
    parser.add_argument("--geometry", required=True, help="cache geometry.npz")
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--scene", choices=tuple(SCENE_TO_GT), required=True)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    density = np.load(args.density, allow_pickle=False).reshape(-1).astype(np.float64)
    arrays = np.load(args.geometry, allow_pickle=False)
    voxels = arrays["voxels"]
    if density.shape[0] != voxels.shape[0]:
        raise ValueError("density and official voxel lattice lengths differ")
    span = density.max() - density.min()
    normalized = np.zeros_like(density) if span <= 0 else (density - density.min()) / span
    predicted = voxels[normalized > args.threshold]

    stem = SCENE_TO_GT[args.scene]
    gt_root = Path(args.gt_root)
    gt_surface = np.load(gt_root / f"{stem}_surface.npy", allow_pickle=False)
    gt_volume = np.load(gt_root / f"{stem}_volume.npy", allow_pickle=False)
    pred_voxels, gt_voxels = voxel_keys(predicted), voxel_keys(gt_volume)
    intersection = len(pred_voxels & gt_voxels)
    union = len(pred_voxels | gt_voxels)
    precision = intersection / max(len(pred_voxels), 1)
    recall = intersection / max(len(gt_voxels), 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-20)
    result = {
        "scene": args.scene,
        "threshold": args.threshold,
        "predicted_points": len(predicted),
        "chamfer_surface": chamfer_squared(gt_surface, predicted),
        "chamfer_volume": chamfer_squared(gt_volume, predicted),
        "iou": intersection / max(union, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "voxel_unit_m": 0.005,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
