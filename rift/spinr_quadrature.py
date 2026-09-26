"""Torch-free tensor-product cell quadrature fixtures for SpINR-style integration.

The production adapter turns these NumPy arrays into device tensors.  Keeping
the rule construction independent of Torch makes the numerical rule testable
on the development workstation, where the full neural stack is intentionally
not installed.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _validate_rule_inputs(grid_size: int, nodes_per_cell: int, support_m: float) -> tuple[int, int, float]:
    grid_size = int(grid_size)
    nodes_per_cell = int(nodes_per_cell)
    support_m = float(support_m)
    if grid_size <= 1:
        raise ValueError("grid_size must be greater than one")
    if nodes_per_cell <= 0:
        raise ValueError("nodes_per_cell must be positive")
    if not np.isfinite(support_m) or support_m <= 0.0:
        raise ValueError("support_m must be finite and positive")
    return grid_size, nodes_per_cell, support_m


def midpoint_grid_arrays(
    grid_size: int,
    *,
    support_m: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return midpoint nodes and scalar cell weights as NumPy arrays."""

    grid_size, _nodes_per_cell, support_m = _validate_rule_inputs(grid_size, 1, support_m)
    pitch = 2.0 * support_m / grid_size
    axis = -support_m + (np.arange(grid_size, dtype=np.float64) + 0.5) * pitch
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    weights = np.full(points.shape[0], pitch ** 3, dtype=np.float64)
    return np.ascontiguousarray(points), weights


def gauss_legendre_cell_grid_arrays(
    grid_size: int,
    *,
    nodes_per_cell: int = 2,
    support_m: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return tensor Gauss–Legendre nodes/weights inside every support cell.

    ``grid_size`` is the number of parent cells along each axis.  The returned
    point count is ``(grid_size * nodes_per_cell)**3``.  Every returned weight
    is a physical volume in m³; for the two-node rule the weights are uniform,
    while higher-order reference rules retain their tensor-product weights.
    """

    grid_size, nodes_per_cell, support_m = _validate_rule_inputs(
        grid_size, nodes_per_cell, support_m
    )
    nodes, one_dimensional_weights = np.polynomial.legendre.leggauss(nodes_per_cell)
    pitch = 2.0 * support_m / grid_size
    centers = -support_m + (np.arange(grid_size, dtype=np.float64) + 0.5) * pitch
    axis = (centers[:, None] + 0.5 * pitch * nodes[None, :]).reshape(-1)
    axis_weights = np.tile(0.5 * pitch * one_dimensional_weights, grid_size)

    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    weights = (
        axis_weights[:, None, None]
        * axis_weights[None, :, None]
        * axis_weights[None, None, :]
    ).reshape(-1)
    if not np.isfinite(points).all() or not np.isfinite(weights).all() or (weights <= 0.0).any():
        raise FloatingPointError("cell quadrature produced non-finite or non-positive values")
    return np.ascontiguousarray(points), np.ascontiguousarray(weights)
