"""Efficient pair-dependent GeRaF signal tracing on a uniform frequency grid.

RIFT's standard range operator assumes one complex scatterer weight shared by
all Tx/Rx pairs.  GeRaF's shifted-Lambertian response, visibility correction,
and path decay instead produce one amplitude per ``(pair, sample)``.  This
module is the corresponding type-1 range NUFFT.  It reuses the already gated
gridding primitives in :mod:`rift.range_operator` but deliberately applies no
extra RIFT geometric gain: GeRaF's Eq. 5 amplitude already contains the full
physical factors.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.utils.checkpoint

from rift.config import cc
from rift.range_operator import (
    _complex_dtype,
    _deapodization,
    _default_tau_bins,
    _mode_bins,
    _next_power_of_two,
    _normalize_freq_indices,
    _spread_to_grid,
    _validate_and_reconstruct_grid,
    range_adjoint_operator,
)


FOUR_PI_SQUARED_INV = 1.0 / ((4.0 * math.pi) ** 2)


def bistatic_pair_positions(
    tx_positions: torch.Tensor, rx_positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten calibrated arrays in RIFT's Tx-outer/Rx-inner channel order."""

    if tx_positions.ndim != 2 or tx_positions.shape[-1] != 3:
        raise ValueError("tx_positions must have shape [Tx,3]")
    if rx_positions.ndim != 2 or rx_positions.shape[-1] != 3:
        raise ValueError("rx_positions must have shape [Rx,3]")
    num_tx, num_rx = tx_positions.shape[0], rx_positions.shape[0]
    tx_pair = tx_positions[:, None, :].expand(num_tx, num_rx, 3).reshape(-1, 3)
    rx_pair = rx_positions[None, :, :].expand(num_tx, num_rx, 3).reshape(-1, 3)
    return tx_pair, rx_pair


def _trace_point_chunk(
    positions,
    pair_weights,
    tx_pair,
    rx_pair,
    f_ref,
    df,
    phase_sign,
    eps,
    m_grid,
    kernel_width,
    tau_bins,
    complex_dtype,
):
    """Pure point-chunk body suitable for gradient checkpointing."""

    r_tx = torch.linalg.vector_norm(
        positions[:, None, :] - tx_pair[None, :, :], dim=-1
    ).clamp_min(eps)
    r_rx = torch.linalg.vector_norm(
        positions[:, None, :] - rx_pair[None, :, :], dim=-1
    ).clamp_min(eps)
    r_sum = r_tx + r_rx  # [Nc,Pc]
    phase_ref = phase_sign * (2.0 * torch.pi * f_ref / cc) * r_sum
    carrier = torch.polar(torch.ones_like(phase_ref), phase_ref)
    # pair_weights arrives as [Pc,Nc]; gridding uses [Nc,Pc].
    amplitude = pair_weights.transpose(0, 1).to(complex_dtype) * carrier.to(complex_dtype)
    x = phase_sign * (2.0 * torch.pi * df / cc) * r_sum
    u = torch.remainder(x * (float(m_grid) / (2.0 * torch.pi)), float(m_grid))
    grid = torch.zeros(
        (tx_pair.shape[0], m_grid), dtype=complex_dtype, device=positions.device
    )
    _spread_to_grid(grid, amplitude, u, kernel_width, tau_bins)
    return grid


def pairwise_range_forward_operator(
    freqs_full: torch.Tensor,
    kvector_full: torch.Tensor,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    sample_positions: torch.Tensor,
    pair_sample_amplitudes: torch.Tensor,
    *,
    phase_sign: float = -1.0,
    eps: float = 1.0e-9,
    freq_indices: Optional[torch.Tensor] = None,
    oversample: int = 2,
    kernel_width: int = 20,
    pair_chunk: int = 32,
    point_chunk: int = 4096,
    compute_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Trace GeRaF amplitudes into ``[F,Rx,Tx]`` complex responses.

    ``pair_sample_amplitudes`` is ``[Tx*Rx,N]`` in Tx-outer/Rx-inner order.
    It may be real or complex and remains fully differentiable.
    """

    if oversample < 2:
        raise ValueError("oversample must be >= 2")
    if kernel_width < 4 or pair_chunk <= 0 or point_chunk <= 0:
        raise ValueError("invalid gridding/chunk configuration")
    if sample_positions.ndim != 2 or sample_positions.shape[-1] != 3:
        raise ValueError("sample_positions must have shape [N,3]")
    tx_pair, rx_pair = bistatic_pair_positions(tx_positions, rx_positions)
    n_pairs = tx_pair.shape[0]
    n_points = sample_positions.shape[0]
    if pair_sample_amplitudes.shape != (n_pairs, n_points):
        raise ValueError(
            f"pair_sample_amplitudes must be {(n_pairs, n_points)}, "
            f"got {tuple(pair_sample_amplitudes.shape)}"
        )

    device = sample_positions.device
    real_dtype = compute_dtype
    complex_dtype = _complex_dtype(real_dtype)
    ideal_freqs = _validate_and_reconstruct_grid(
        torch.as_tensor(freqs_full, device=device),
        torch.as_tensor(kvector_full, device=device),
    ).to(real_dtype)
    nf_full = ideal_freqs.shape[0]
    selected = _normalize_freq_indices(freq_indices, nf_full, device)
    df = ideal_freqs[1] - ideal_freqs[0]
    f_ref = ideal_freqs[nf_full // 2]
    m_grid = _next_power_of_two(int(math.ceil(float(oversample) * nf_full)))
    tau_bins = _default_tau_bins(kernel_width)
    deapod = _deapodization(
        nf_full, m_grid, kernel_width, tau_bins, device, real_dtype
    )
    mode_bins = _mode_bins(nf_full, m_grid, device)

    positions = sample_positions.to(real_dtype)
    tx_pair = tx_pair.to(device=device, dtype=real_dtype)
    rx_pair = rx_pair.to(device=device, dtype=real_dtype)
    amplitudes = pair_sample_amplitudes.to(device=device, dtype=complex_dtype)
    rendered_chunks = []
    for pair_start in range(0, n_pairs, pair_chunk):
        pair_end = min(pair_start + pair_chunk, n_pairs)
        pair_grid = torch.zeros(
            (pair_end - pair_start, m_grid), dtype=complex_dtype, device=device
        )
        for point_start in range(0, n_points, point_chunk):
            point_end = min(point_start + point_chunk, n_points)
            chunk_args = (
                positions[point_start:point_end],
                amplitudes[pair_start:pair_end, point_start:point_end],
                tx_pair[pair_start:pair_end],
                rx_pair[pair_start:pair_end],
                f_ref,
                df,
                float(phase_sign),
                float(eps),
                int(m_grid),
                int(kernel_width),
                float(tau_bins),
                complex_dtype,
            )
            if torch.is_grad_enabled() and pair_sample_amplitudes.requires_grad:
                contribution = torch.utils.checkpoint.checkpoint(
                    _trace_point_chunk, *chunk_args, use_reentrant=False
                )
            else:
                contribution = _trace_point_chunk(*chunk_args)
            pair_grid = pair_grid + contribution
        frequency_values = m_grid * torch.fft.ifft(pair_grid, dim=-1)
        rendered_chunks.append(frequency_values[:, mode_bins] / deapod.view(1, nf_full))

    pair_frequency = torch.cat(rendered_chunks, dim=0)[:, selected]
    num_tx, num_rx = tx_positions.shape[0], rx_positions.shape[0]
    return pair_frequency.view(num_tx, num_rx, -1).permute(2, 1, 0).contiguous()


def matched_filter_from_response_range(
    complex_response: torch.Tensor,
    freqs_full: torch.Tensor,
    kvector_full: torch.Tensor,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    query_points: torch.Tensor,
    *,
    phase_sign: float = -1.0,
    oversample: int = 2,
    kernel_width: int = 20,
    pair_chunk: int = 32,
    point_chunk: int = 4096,
    compute_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Fast phase-only matched-filter complex amplitude (GeRaF Eq. 3)."""

    amplitude = range_adjoint_operator(
        freqs_full,
        kvector_full,
        rx_positions,
        tx_positions,
        query_points,
        complex_response,
        phase_sign=phase_sign,
        oversample=oversample,
        kernel_width=kernel_width,
        pair_chunk=pair_chunk,
        point_chunk=point_chunk,
        compute_dtype=compute_dtype,
        range_model="none",
    )
    return amplitude / FOUR_PI_SQUARED_INV


def trace_and_match_magnitude(
    freqs_full: torch.Tensor,
    kvector_full: torch.Tensor,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    sample_positions: torch.Tensor,
    pair_sample_amplitudes: torch.Tensor,
    query_points: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return predicted complex response and native GeRaF ``|MF|``.

    A frequency subset is deliberately rejected here.  The forward tracer can
    expose selected frequencies for standalone diagnostics, but the composed
    reportable readout must not trace a subset and then call an adjoint that
    still assumes the full grid.
    """

    if kwargs.get("freq_indices") is not None:
        raise ValueError(
            "trace_and_match_magnitude does not support freq_indices until the "
            "identical subset is explicitly plumbed into the matched-filter adjoint"
        )

    response = pairwise_range_forward_operator(
        freqs_full,
        kvector_full,
        tx_positions,
        rx_positions,
        sample_positions,
        pair_sample_amplitudes,
        **kwargs,
    )
    mf_amplitude = matched_filter_from_response_range(
        response,
        freqs_full,
        kvector_full,
        tx_positions,
        rx_positions,
        query_points,
        phase_sign=kwargs.get("phase_sign", -1.0),
        oversample=kwargs.get("oversample", 2),
        kernel_width=kwargs.get("kernel_width", 20),
        pair_chunk=kwargs.get("pair_chunk", 32),
        point_chunk=kwargs.get("point_chunk", 4096),
        compute_dtype=kwargs.get("compute_dtype", torch.float64),
    )
    return response, mf_amplitude.abs()


__all__ = [
    "bistatic_pair_positions",
    "pairwise_range_forward_operator",
    "matched_filter_from_response_range",
    "trace_and_match_magnitude",
]
