"""Small shared helpers for the sealed B787 common evaluation contract.

The RF-supported mask is a renderer-coverage mask only.  It is derived from the
same midpoint grid and differentiable bistatic range-cell deposition used by the
Radar Fields readout; it never inspects a learned field or measured target.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from rift.encoding import generate_dynamic_grid
from rift.radar_fields import bistatic_range_cells
from rift.radar_fields_dataset import range_bin_size, scene_range_mask


RF_GRID_GRANULARITY = 48
RF_GRID_EXTENT_M = 0.15
RF_RANGE_MARGIN_M = 0.05
RF_MASK_PAIR_CHUNK = 8
RF_NUM_BINS = 600


def make_rf_midpoint_grid(
    device: torch.device,
    *,
    granularity: int = RF_GRID_GRANULARITY,
    extent_m: float = RF_GRID_EXTENT_M,
) -> torch.Tensor:
    """Create the exact float32 RF midpoint grid used by the native readout."""

    grid = generate_dynamic_grid(
        int(granularity), float(extent_m), device, jitter=False
    ).reshape(-1, 3)
    if grid.dtype != torch.float32:
        grid = grid.to(dtype=torch.float32)
    return grid


def rf_supported_mask(
    grid_xyz: torch.Tensor,
    *,
    ranges: torch.Tensor,
    viewpoint: torch.Tensor,
    tx_pos: torch.Tensor,
    rx_pos: torch.Tensor,
    pair_indices: torch.Tensor,
    metadata: Mapping[str, object],
    num_bins: int = RF_NUM_BINS,
    extent_m: float = RF_GRID_EXTENT_M,
    range_margin_m: float = RF_RANGE_MARGIN_M,
    pair_chunk: int = RF_MASK_PAIR_CHUNK,
    roi: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(supported, roi)`` for one B787 acquisition view.

    ``supported`` has shape ``[pairs, roi_bins]`` and is exactly the RF
    ``cell_mass[:, roi] > 0`` predicate.  The dummy values are deliberately all
    ones: only interpolation mass is used, so learned/model/target values cannot
    affect the mask.
    """

    if grid_xyz.ndim != 2 or grid_xyz.shape[-1] != 3:
        raise ValueError("RF mask grid must have shape [N,3]")
    if grid_xyz.dtype != torch.float32:
        raise ValueError("RF mask grid must use float32 coordinates")
    if tx_pos.dtype != torch.float32 or rx_pos.dtype != torch.float32:
        raise ValueError("RF mask Tx/Rx positions must use float32 coordinates")
    if pair_indices.ndim != 1:
        raise ValueError("RF mask pair indices must be one-dimensional")
    if roi is None:
        roi = scene_range_mask(
            ranges,
            viewpoint,
            float(extent_m),
            margin=float(range_margin_m),
        )
    if roi.ndim != 1 or roi.shape[0] != ranges.shape[0] or roi.dtype != torch.bool:
        raise ValueError("RF mask ROI must be a boolean vector matching ranges")
    if not bool(roi.any().item()):
        raise ValueError("RF mask requires a non-empty scene-range ROI")

    dummy_values = torch.ones(
        (grid_xyz.shape[0], 1), dtype=torch.float32, device=grid_xyz.device
    )
    if int(num_bins) != int(ranges.shape[0]):
        raise ValueError("RF mask range-bin count must match the supplied ranges")
    _, cell_mass = bistatic_range_cells(
        dummy_values,
        grid_xyz,
        tx_pos,
        rx_pos,
        pair_indices,
        bin_size=range_bin_size(metadata),
        num_bins=int(num_bins),
        pair_chunk=int(pair_chunk),
    )
    return cell_mass[:, roi] > 0, roi


def partition_squared_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    supported: torch.Tensor,
) -> dict[str, float | int]:
    """Return additive squared-error/target/count statistics for two partitions."""

    if prediction.shape != target.shape or prediction.shape != supported.shape:
        raise ValueError("prediction, target, and support mask must have matching shapes")
    if prediction.numel() == 0:
        raise ValueError("RF-supported metric requires at least one ROI element")
    if supported.dtype != torch.bool:
        raise ValueError("RF-supported metric mask must be boolean")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("RF-supported metric inputs must be finite")

    squared_error = (prediction - target).square()
    target_squared = target.square()
    padded = ~supported

    def sums(mask: torch.Tensor) -> tuple[float, float, int]:
        return (
            float(squared_error[mask].sum().item()),
            float(target_squared[mask].sum().item()),
            int(mask.sum().item()),
        )

    supported_error, supported_target, supported_count = sums(supported)
    padded_error, padded_target, padded_count = sums(padded)
    if supported_count <= 0:
        raise ValueError("RF-supported metric mask is empty")
    if supported_target <= 0.0:
        raise ValueError("RF-supported metric has a non-positive target norm")
    return {
        "supported_squared_error": supported_error,
        "supported_target_squared_norm": supported_target,
        "supported_element_count": supported_count,
        "padded_squared_error": padded_error,
        "padded_target_squared_norm": padded_target,
        "padded_element_count": padded_count,
    }


def pooled_relative_mse(
    squared_error: float,
    target_squared_norm: float,
) -> float:
    """Derive one pooled RelMSE without fabricating a ratio for an empty norm."""

    if target_squared_norm <= 0.0:
        raise ValueError("pooled relative MSE requires a positive target squared norm")
    return float(squared_error / target_squared_norm)


def diagnostic_relative_mse(
    squared_error: float,
    target_squared_norm: float,
) -> float:
    """Per-view diagnostic ratio; NaN when the partition has no target energy.

    Not for role-pooled reporting: use ``pooled_relative_mse`` there.
    """

    if not math.isfinite(squared_error) or not math.isfinite(target_squared_norm):
        raise ValueError("diagnostic relative MSE requires finite inputs")
    if target_squared_norm <= 0.0:
        return float("nan")
    return float(squared_error / target_squared_norm)
