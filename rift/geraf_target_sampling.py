"""Query source-equivalent MF lattice interpolation without a dense volume.

The source uses FP32, zero-padded ``grid_sample`` with ``align_corners=True``.
This helper retains that lattice and interpolation, fetching only the needed
corners. It does not evaluate a continuous matched filter at the query itself.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def sample_lattice(
    points_norm: np.ndarray,
    grid_size: int,
    fetch_values: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Return FP32 interpolated values for normalized ``(Q, 3)`` xyz points.

    ``fetch_values`` receives sorted, unique in-bounds ``int64`` lattice IDs,
    with x outermost and z innermost, and returns a matching FP32 vector. It
    is called at most once and never for queries contributing only padding.
    Temporary storage scales with Q, not ``grid_size ** 3``. Nonfinite values
    are rejected; this matches the target preparation's finite-value gate.
    """
    if (isinstance(grid_size, (bool, np.bool_))
            or not isinstance(grid_size, (int, np.integer))
            or int(grid_size) < 2
            or int(grid_size) ** 3 > np.iinfo(np.int64).max):
        raise ValueError("grid_size must be an integer >= 2 with int64 lattice IDs")
    n = int(grid_size)
    raw = np.asarray(points_norm)
    if raw.ndim != 2 or raw.shape[1] != 3 or raw.dtype.kind not in "fiu":
        raise ValueError("points_norm must be a real numeric array with shape (Q, 3)")
    with np.errstate(over="ignore", invalid="ignore"):
        points = raw.astype(np.float32)
    if not np.isfinite(points).all():
        raise ValueError("points_norm must remain finite after FP32 conversion")
    result = np.zeros(len(points), dtype=np.float32)
    # For every n >= 2, coordinates beyond [-3, 3] contribute only zero
    # padding. Remove them before integer conversion to avoid overflow.
    rows = np.flatnonzero((np.abs(points) <= np.float32(3)).all(axis=1))
    if not len(rows):
        return result
    # Preserve grid_sampler_unnormalize's FP32 operation order.
    positions = ((points[rows] + np.float32(1)) / np.float32(2)) * np.float32(n - 1)
    lower = np.floor(positions).astype(np.int64)
    fraction = positions - lower.astype(np.float32)
    weights = (np.float32(1) - fraction, fraction)
    corners = []
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                ijk = lower + np.array([dx, dy, dz], dtype=np.int64)
                # grid_sample sees z,y,x after the source's axis reversal.
                weight = weights[dz][:, 2] * weights[dy][:, 1] * weights[dx][:, 0]
                valid = ((ijk >= 0) & (ijk < n)).all(axis=1) & (weight != 0)
                local_rows = np.flatnonzero(valid)
                ids = (ijk[valid, 0] * n + ijk[valid, 1]) * n + ijk[valid, 2]
                corners.append((local_rows, ids, weight[valid]))
    all_ids = np.concatenate([item[1] for item in corners])
    if not len(all_ids):
        return result
    unique, inverse = np.unique(all_ids, return_inverse=True)
    values = np.asarray(fetch_values(unique))
    if values.shape != unique.shape or values.dtype != np.dtype(np.float32):
        raise ValueError("fetch_values must return a matching flat float32 array")
    if not np.isfinite(values).all():
        raise ValueError("fetch_values returned nonfinite lattice values")
    sampled = np.zeros(len(rows), dtype=np.float32)
    start = 0
    for local_rows, ids, weight in corners:
        end = start + len(ids)
        sampled[local_rows] += values[inverse[start:end]] * weight
        start = end
    result[rows] = sampled
    return result
