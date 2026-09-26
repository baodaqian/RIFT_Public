"""Numerical and observable contracts for the independent SpINR v1 adaptation.

The source measurements are passband frequency samples, not dechirped ADC
samples. Their exact coherent operator is retained; an FFT defines the range
observable. No extra leakage or residual-video-phase kernel belongs upstream.
"""
from __future__ import annotations

import math

import torch

from rift.config import cc

PAPER_RECIPE = "rift_dataset_spinr_v1_passband_g96_gl2_scene_bins_v1"
DIRECT_RECIPE = "rift_dataset_spinr_v1_passband_direct_bins_1500_v1"
BUDGET48_RECIPE = "rift_dataset_spinr_v1_passband_direct_bins_g48_midpoint_1500_v1"
BUDGET150_RECIPE = "rift_dataset_spinr_v1_passband_direct_bins_g48_midpoint_150_v1"
PAPER_EPOCHS = 1500
COMPARISON_EPOCHS = 150
PAPER_REFERENCE = "https://arxiv.org/html/2503.23313v2"
PARENT_GRID = 96
NODES_PER_CELL = 2


def recipe_name_from_identity(identity: dict) -> str:
    from rift.spinr_style import SPINR_STYLE_RECIPE_ID
    names = {SPINR_STYLE_RECIPE_ID: "legacy-midpoint", PAPER_RECIPE: "paper-v1",
             DIRECT_RECIPE: "paper-v1-direct", BUDGET48_RECIPE: "budget48-direct-1500",
             BUDGET150_RECIPE: "budget48-direct"}
    try:
        return names[identity.get("recipe_id")]
    except (KeyError, TypeError) as exc:
        raise ValueError("unrecognized SpINR scientific recipe") from exc


def scene_range_bin_mask(frequencies_hz: torch.Tensor, rx_pos_m: torch.Tensor,
                         tx_pos_m: torch.Tensor, *, support_m: float = 0.15,
                         phase_sign: float = -1.0) -> torch.Tensor:
    """Geometry-only, per-pair scene bins in FFT order, shaped [F,Rx,Tx].

    For S_n = exp(sign*i*2*pi*(f0+n*df)*R/c), its FFT peak is at
    k = sign*N*df*R/c modulo N. Bound R over the entire support cube:
    separate point-to-box minima give a conservative lower bound; the maximum
    of the convex sum of distances occurs at a cube vertex. Include floor/ceil
    bracketing bins. The selection includes no response- or mesh-derived input.
    Sidelobes outside these bins remain part of full-response evaluation.
    """
    f = torch.as_tensor(frequencies_hz)
    if f.dtype != torch.float64 or f.ndim != 1 or f.numel() < 2 or not torch.isfinite(f).all():
        raise ValueError("scene-bin selection requires finite float64 frequencies")
    df = f[1] - f[0]
    if df <= 0 or not torch.allclose(f[1:] - f[:-1], df.expand(f.numel()-1), rtol=1e-10, atol=1e-6):
        raise ValueError("SpINR FFT range bins require an increasing uniform frequency grid")
    if not math.isfinite(support_m) or support_m <= 0 or phase_sign not in (-1.0, 1.0):
        raise ValueError("invalid support or phase sign")
    positions = []
    for p in (rx_pos_m, tx_pos_m):
        p = torch.as_tensor(p, device=f.device, dtype=torch.float64)
        if p.ndim != 2 or p.shape[1] != 3 or not p.shape[0] or not torch.isfinite(p).all():
            raise ValueError("antenna positions must be finite [elements,3]")
        positions.append(p)
    rx, tx = positions
    axis = f.new_tensor([-support_m, support_m])
    corners = torch.cartesian_prod(axis, axis, axis)
    r_rx = torch.linalg.vector_norm(corners[:, None] - rx[None], dim=-1)
    r_tx = torch.linalg.vector_norm(corners[:, None] - tx[None], dim=-1)
    upper = (r_rx[:, :, None] + r_tx[:, None, :]).amax(dim=0)
    lower = (torch.linalg.vector_norm((rx.abs()-support_m).clamp_min(0), dim=-1)[:, None]
             + torch.linalg.vector_norm((tx.abs()-support_m).clamp_min(0), dim=-1)[None, :])
    factor = phase_sign * f.numel() * df / cc
    a, b = factor * lower, factor * upper
    first = torch.floor(torch.minimum(a, b))
    last = torch.ceil(torch.maximum(a, b))
    bins = torch.arange(f.numel(), device=f.device, dtype=torch.float64)[:, None, None]
    # The modular difference also handles ranges crossing DC or an alias period.
    return torch.remainder(bins-first, f.numel()) <= (last-first)


def spectral_partition_terms(predicted: torch.Tensor, observed: torch.Tensor,
                             mask: torch.Tensor, *, training_mean_raw_power: float) -> dict[str, float]:
    """Additive scene/remainder loss, energy and response-gradient diagnostics.

    Each loss uses sum over bins / (number of pairs * train mean raw power).
    Adding the two partitions recovers the historical all-bin objective.
    Gradient norms are with respect to one view's *raw complex response*, not
    model parameters; the FFT adjoint contributes the 1/N norm-squared factor.
    """
    from rift.spinr_style import frequency_to_range_bins
    if predicted.shape != observed.shape or mask.shape != predicted.shape or mask.dtype != torch.bool:
        raise ValueError("partition inputs must have matching [F,Rx,Tx] shapes and a boolean mask")
    if predicted.ndim != 3 or not math.isfinite(training_mean_raw_power) or training_mean_raw_power <= 0:
        raise ValueError("partition diagnostics require one view and positive training power")
    p, y = frequency_to_range_bins(predicted), frequency_to_range_bins(observed)
    residual = (p-y).abs().square()
    loss = (p.abs()-y.abs()).square() + 0.5*residual
    scale = 1.0 / (p.shape[1]*p.shape[2]*training_mean_raw_power)
    unit = p / p.abs().clamp_min(torch.finfo(p.real.dtype).tiny)
    cotangent = scale * (2*(p.abs()-y.abs())*unit + p-y)
    result = {}
    for label, selection in (("scene", mask), ("remainder", ~mask)):
        result[f"{label}_objective"] = float(loss[selection].sum()) * scale
        result[f"{label}_squared_error"] = float(residual[selection].sum())
        result[f"{label}_target_energy"] = float(y[selection].abs().square().sum())
        result[f"{label}_bin_pair_count"] = int(selection.sum())
        result[f"{label}_response_gradient_norm_squared"] = float(cotangent[selection].abs().square().sum()) / p.shape[0]
    return result


def field_readout(model, *, grid_size: int, support_m: float,
                  initial_output_scale: float, neural_point_tile: int,
                  device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample signed sigma and its magnitude on declared physical midpoints.

    Geometry uses |sigma|, never a signed occupancy or a zero-level surface.
    Quadrature volume belongs to the signal integral and is excluded here.
    """
    from rift.spinr_style import midpoint_grid
    if neural_point_tile < 1 or not math.isfinite(initial_output_scale) or initial_output_scale <= 0:
        raise ValueError("invalid readout tile or scale")
    points, _ = midpoint_grid(grid_size, support_m=support_m, device=device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            sigma = torch.cat([model(p).to(torch.float64) for p in points.split(neural_point_tile)])
            sigma = sigma * initial_output_scale
    finally:
        model.train(was_training)
    if torch.is_complex(sigma) or not torch.isfinite(sigma).all():
        raise ValueError("SpINR geometry readout must be finite signed real")
    return points, sigma
