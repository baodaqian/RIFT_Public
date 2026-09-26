"""Chunk-source wrappers around the frozen PublicRadar range operator.

The frozen operator serializes its arithmetic after receiving complete
position and weight tensors.  These wrappers preserve the exact internal
kernel/FFT while accepting factories that generate those tensors one spatial
chunk at a time, avoiding dense full-scene coordinate/effective-weight
materialization.
"""

from __future__ import annotations

import math

import torch
import torch.utils.checkpoint

from rift.coherent_radar_geometry import (
    BISTATIC_NEAR_FIELD_ABSOLUTE,
    MONOSTATIC_FAR_FIELD_REFERENCE,
    MONOSTATIC_NEAR_FIELD_REFERENCE,
    VALID_PROPAGATION_MODELS,
)
from rift.config import cc
from rift.range_operator import (
    _complex_dtype,
    _deapodization,
    _default_tau_bins,
    _geom_gain,
    _mode_bins,
    _next_power_of_two,
    _normalize_freq_indices,
    _pair_indices,
    _render_adjoint_point_chunk,
    _render_forward_point_chunk,
    _spread_to_grid,
    _validate_and_reconstruct_grid,
)


def _common(
    freqs_full,
    kvector_full,
    device,
    compute_dtype,
    freq_indices,
    oversample,
    kernel_width,
):
    if oversample < 2:
        raise ValueError("oversample must be >= 2")
    freqs_full = torch.as_tensor(freqs_full, device=device)
    kvector_full = torch.as_tensor(kvector_full, device=device)
    ideal_freqs = _validate_and_reconstruct_grid(freqs_full, kvector_full)
    nf_full = ideal_freqs.shape[0]
    freq_idx = _normalize_freq_indices(freq_indices, nf_full, device)
    real_dtype = compute_dtype
    complex_dtype = _complex_dtype(real_dtype)
    ideal_freqs = ideal_freqs.to(real_dtype)
    df = ideal_freqs[1] - ideal_freqs[0]
    f_ref = ideal_freqs[nf_full // 2]
    m_grid = _next_power_of_two(int(math.ceil(float(oversample) * nf_full)))
    tau_bins = _default_tau_bins(kernel_width)
    deapod = _deapodization(
        nf_full, m_grid, kernel_width, tau_bins, device, real_dtype
    )
    return {
        "nf_full": nf_full,
        "freq_idx": freq_idx,
        "real_dtype": real_dtype,
        "complex_dtype": complex_dtype,
        "df": df,
        "f_ref": f_ref,
        "m_grid": m_grid,
        "tau_bins": tau_bins,
        "deapod": deapod,
        "mode_bins": _mode_bins(nf_full, m_grid, device),
    }


def _validate_point_mask(point_mask, positions):
    """Evaluate an optional per-view spatial mask for one point chunk."""
    if point_mask is None:
        return None
    mask = point_mask(positions)
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask, device=positions.device)
    else:
        mask = mask.to(device=positions.device)
    if mask.dtype != torch.bool:
        raise ValueError("point_mask must return a boolean tensor")
    if mask.ndim != 1 or mask.shape[0] != positions.shape[0]:
        raise ValueError("point_mask must return shape [chunk_points]")
    return mask


def _aligned_view_path_lengths(
    positions,
    view_tx_pos,
    view_rx_pos,
    propagation_model,
    reference_range_m,
    scene_center_m,
    eps,
):
    """Return aligned ``[point, view]`` propagation legs and two-way paths."""
    if propagation_model == BISTATIC_NEAR_FIELD_ABSOLUTE:
        r_tx = torch.linalg.vector_norm(
            positions[:, None, :] - view_tx_pos[None, :, :], dim=-1
        ).clamp_min(eps)
        r_rx = torch.linalg.vector_norm(
            positions[:, None, :] - view_rx_pos[None, :, :], dim=-1
        ).clamp_min(eps)
        return r_tx, r_rx, r_tx + r_rx

    platform = 0.5 * (view_tx_pos + view_rx_pos)
    center = torch.as_tensor(
        scene_center_m, device=positions.device, dtype=positions.dtype
    )
    if propagation_model == MONOSTATIC_NEAR_FIELD_REFERENCE:
        center_range = torch.linalg.vector_norm(
            platform - center[None, :], dim=-1
        ).clamp_min(eps)
        one_way = (
            torch.linalg.vector_norm(
                positions[:, None, :] - platform[None, :, :], dim=-1
            )
            - center_range[None, :]
            + float(reference_range_m)
        )
    elif propagation_model == MONOSTATIC_FAR_FIELD_REFERENCE:
        look = platform - center[None, :]
        look = look / torch.linalg.vector_norm(look, dim=-1).clamp_min(eps)[:, None]
        one_way = float(reference_range_m) - (positions - center[None, :]) @ look.T
    else:
        raise ValueError(f"unsupported aligned propagation model {propagation_model!r}")
    if not bool(torch.isfinite(one_way).all()) or bool((one_way <= 0.0).any()):
        raise ValueError("reference range is too small for the requested scene points")
    return one_way, one_way, 2.0 * one_way


def _render_forward_aligned_view_point_chunk(
    positions,
    weights,
    view_tx_pos,
    view_rx_pos,
    f_ref,
    df,
    phase_sign,
    eps,
    g_const,
    m_grid,
    kernel_width,
    tau_bins,
    complex_dtype,
    range_model,
    propagation_model,
    reference_range_m,
    scene_center_m,
):
    """Render one point chunk for strictly aligned, independent views."""
    r_tx, r_rx, r_sum = _aligned_view_path_lengths(
        positions,
        view_tx_pos,
        view_rx_pos,
        propagation_model,
        reference_range_m,
        scene_center_m,
        eps,
    )
    geom = _geom_gain(r_tx, r_rx, r_sum, range_model, g_const, eps)
    phase_ref = phase_sign * (2.0 * torch.pi * f_ref / cc) * r_sum
    amplitude = weights * geom.to(complex_dtype) * torch.polar(
        torch.ones_like(phase_ref), phase_ref
    )
    x = phase_sign * (2.0 * torch.pi * df / cc) * r_sum
    u = torch.remainder(x * (float(m_grid) / (2.0 * torch.pi)), float(m_grid))
    grid = torch.zeros(
        (view_tx_pos.shape[0], m_grid),
        dtype=complex_dtype,
        device=positions.device,
    )
    _spread_to_grid(grid, amplitude, u, kernel_width, tau_bins)
    return grid


def range_forward_operator_aligned_view_chunks(
    freqs_full,
    kvector_full,
    view_rx_pos,
    view_tx_pos,
    scatterer_chunks,
    phase_sign=1.0,
    eps=1.0e-9,
    freq_indices=None,
    oversample=2,
    kernel_width=20,
    compute_dtype=torch.float64,
    range_model="sum2",
    propagation_model=BISTATIC_NEAR_FIELD_ABSOLUTE,
    reference_range_m=None,
    scene_center_m=(0.0, 0.0, 0.0),
    point_mask=None,
):
    """Render independent aligned views without forming cross-view pairs.

    ``view_rx_pos`` and ``view_tx_pos`` must both have shape ``[B, 3]``.
    The replayable chunk factory must yield positions ``[K, 3]`` and complex
    effective weights ``[K, B]``. The returned tensor has shape ``[B, F]``.
    """
    device = torch.as_tensor(freqs_full).device
    state = _common(
        freqs_full,
        kvector_full,
        device,
        compute_dtype,
        freq_indices,
        oversample,
        kernel_width,
    )
    view_rx_pos = torch.as_tensor(
        view_rx_pos, device=device, dtype=state["real_dtype"]
    )
    view_tx_pos = torch.as_tensor(
        view_tx_pos, device=device, dtype=state["real_dtype"]
    )
    if view_rx_pos.ndim != 2 or view_rx_pos.shape[1] != 3:
        raise ValueError("view_rx_pos must have shape [views, 3]")
    if view_tx_pos.ndim != 2 or view_tx_pos.shape[1] != 3:
        raise ValueError("view_tx_pos must have shape [views, 3]")
    if view_rx_pos.shape != view_tx_pos.shape or view_rx_pos.shape[0] == 0:
        raise ValueError("aligned Rx/Tx view batches must have equal nonzero size")
    if not bool(torch.isfinite(view_rx_pos).all()) or not bool(
        torch.isfinite(view_tx_pos).all()
    ):
        raise ValueError("aligned view positions must be finite")
    if propagation_model not in VALID_PROPAGATION_MODELS:
        raise ValueError(f"unknown propagation_model {propagation_model!r}")
    center = torch.as_tensor(
        scene_center_m, device=device, dtype=state["real_dtype"]
    )
    if center.shape != (3,) or not bool(torch.isfinite(center).all()):
        raise ValueError("scene_center_m must contain three finite coordinates")
    if propagation_model != BISTATIC_NEAR_FIELD_ABSOLUTE:
        if reference_range_m is None or not math.isfinite(
            float(reference_range_m)
        ) or float(reference_range_m) <= 0.0:
            raise ValueError("reference_range_m must be finite and positive")
        if not bool(torch.allclose(view_tx_pos, view_rx_pos, rtol=0.0, atol=1.0e-5)):
            raise ValueError("aligned monostatic Tx and Rx phase centres are not co-located")
    center_tuple = tuple(float(value) for value in center.detach().cpu())

    batch_size = int(view_tx_pos.shape[0])
    grid = torch.zeros(
        (batch_size, state["m_grid"]),
        dtype=state["complex_dtype"],
        device=device,
    )
    g_const = 1.0 / ((4.0 * torch.pi) ** 2)
    saw_points = False
    for positions, weights in scatterer_chunks():
        positions = positions.to(device=device, dtype=state["real_dtype"])
        weights = weights.to(device=device, dtype=state["complex_dtype"])
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("position chunks must have shape [K, 3]")
        if weights.ndim != 2 or weights.shape != (positions.shape[0], batch_size):
            raise ValueError("aligned weight chunks must have shape [K, views]")
        if positions.shape[0] == 0:
            continue
        saw_points = True
        mask = _validate_point_mask(point_mask, positions)
        if mask is not None:
            positions = positions[mask]
            weights = weights[mask]
        if positions.shape[0] == 0:
            continue
        chunk_args = (
            positions,
            weights,
            view_tx_pos,
            view_rx_pos,
            state["f_ref"],
            state["df"],
            phase_sign,
            eps,
            g_const,
            state["m_grid"],
            kernel_width,
            state["tau_bins"],
            state["complex_dtype"],
            range_model,
            propagation_model,
            reference_range_m,
            center_tuple,
        )
        if torch.is_grad_enabled():
            grid_chunk = torch.utils.checkpoint.checkpoint(
                _render_forward_aligned_view_point_chunk,
                *chunk_args,
                use_reentrant=False,
            )
        else:
            grid_chunk = _render_forward_aligned_view_point_chunk(*chunk_args)
        grid = grid + grid_chunk
    if not saw_points:
        raise ValueError("scatterer chunk factory yielded no points")
    fft_values = state["m_grid"] * torch.fft.ifft(grid, dim=-1)
    rendered = fft_values[:, state["mode_bins"]] / state["deapod"].view(
        1, state["nf_full"]
    )
    return rendered[:, state["freq_idx"]].contiguous()


def range_forward_operator_chunks(
    freqs_full,
    kvector_full,
    arr_pos_rx,
    arr_pos_tx,
    scatterer_chunks,
    phase_sign=1.0,
    eps=1.0e-9,
    freq_indices=None,
    oversample=2,
    kernel_width=20,
    pair_chunk=64,
    compute_dtype=torch.float64,
    range_model="sum2",
    propagation_model=BISTATIC_NEAR_FIELD_ABSOLUTE,
    reference_range_m=None,
    scene_center_m=(0.0, 0.0, 0.0),
    point_mask=None,
):
    """Render from a replayable ``() -> iterable[(positions, weights)]``.

    ``point_mask`` is an optional callable evaluated independently for every
    spatial chunk and view. Invalid rows are removed before propagation, so
    they contribute exactly zero signal and receive exactly zero gradient.
    """
    if pair_chunk <= 0:
        raise ValueError("pair_chunk must be positive")
    device = torch.as_tensor(freqs_full).device
    state = _common(
        freqs_full,
        kvector_full,
        device,
        compute_dtype,
        freq_indices,
        oversample,
        kernel_width,
    )
    arr_pos_rx = arr_pos_rx.to(device=device, dtype=state["real_dtype"])
    arr_pos_tx = arr_pos_tx.to(device=device, dtype=state["real_dtype"])
    num_rx = arr_pos_rx.shape[0]
    num_tx = arr_pos_tx.shape[0]
    n_pairs = num_rx * num_tx
    tx_all, rx_all = _pair_indices(num_tx, num_rx, device)
    g_const = 1.0 / ((4.0 * torch.pi) ** 2)
    out_pairs = []

    for pair_start in range(0, n_pairs, pair_chunk):
        pair_end = min(pair_start + pair_chunk, n_pairs)
        tx_idx = tx_all[pair_start:pair_end]
        rx_idx = rx_all[pair_start:pair_end]
        grid = torch.zeros(
            (pair_end - pair_start, state["m_grid"]),
            dtype=state["complex_dtype"],
            device=device,
        )
        saw_points = False
        for positions, weights in scatterer_chunks():
            positions = positions.to(device=device, dtype=state["real_dtype"])
            weights = weights.to(device=device, dtype=state["complex_dtype"])
            if positions.ndim != 2 or positions.shape[1] != 3:
                raise ValueError("position chunks must have shape [N, 3]")
            if weights.ndim != 1 or weights.shape[0] != positions.shape[0]:
                raise ValueError("weight chunks must have shape [N]")
            if positions.shape[0] == 0:
                continue
            saw_points = True
            mask = _validate_point_mask(point_mask, positions)
            if mask is not None:
                positions = positions[mask]
                weights = weights[mask]
            if positions.shape[0] == 0:
                continue
            chunk_args = (
                positions,
                weights,
                arr_pos_tx,
                arr_pos_rx,
                tx_idx,
                rx_idx,
                state["f_ref"],
                state["df"],
                phase_sign,
                eps,
                g_const,
                state["m_grid"],
                kernel_width,
                state["tau_bins"],
                state["complex_dtype"],
                range_model,
                propagation_model,
                reference_range_m,
                scene_center_m,
            )
            if torch.is_grad_enabled():
                grid_chunk = torch.utils.checkpoint.checkpoint(
                    _render_forward_point_chunk, *chunk_args, use_reentrant=False
                )
            else:
                grid_chunk = _render_forward_point_chunk(*chunk_args)
            grid = grid + grid_chunk
        if not saw_points:
            raise ValueError("scatterer chunk factory yielded no points")
        fft_vals = state["m_grid"] * torch.fft.ifft(grid, dim=-1)
        rendered = fft_vals[:, state["mode_bins"]] / state["deapod"].view(
            1, state["nf_full"]
        )
        out_pairs.append(rendered[:, state["freq_idx"]])

    pairs_freq = torch.cat(out_pairs, dim=0)
    return pairs_freq.view(num_tx, num_rx, -1).permute(2, 1, 0).contiguous()


def range_adjoint_operator_chunks(
    freqs_full,
    kvector_full,
    arr_pos_rx,
    arr_pos_tx,
    position_chunks,
    residual,
    phase_sign=1.0,
    eps=1.0e-9,
    freq_indices=None,
    oversample=2,
    kernel_width=20,
    pair_chunk=64,
    compute_dtype=torch.float64,
    range_model="sum2",
    propagation_model=BISTATIC_NEAR_FIELD_ABSOLUTE,
    reference_range_m=None,
    scene_center_m=(0.0, 0.0, 0.0),
    point_mask=None,
):
    """Apply the exact adjoint while generating positions by spatial chunk.

    With ``point_mask``, invalid rows are excluded from propagation and
    scattered back as exact zeros in the returned full ordered point vector.
    This preserves the forward/adjoint identity for view-dependent support.
    """
    if pair_chunk <= 0:
        raise ValueError("pair_chunk must be positive")
    device = torch.as_tensor(freqs_full).device
    state = _common(
        freqs_full,
        kvector_full,
        device,
        compute_dtype,
        freq_indices,
        oversample,
        kernel_width,
    )
    arr_pos_rx = arr_pos_rx.to(device=device, dtype=state["real_dtype"])
    arr_pos_tx = arr_pos_tx.to(device=device, dtype=state["real_dtype"])
    residual = residual.to(device=device, dtype=state["complex_dtype"])
    num_rx = arr_pos_rx.shape[0]
    num_tx = arr_pos_tx.shape[0]
    n_pairs = num_rx * num_tx
    tx_all, rx_all = _pair_indices(num_tx, num_rx, device)
    g_const = 1.0 / ((4.0 * torch.pi) ** 2)
    output = None

    for pair_start in range(0, n_pairs, pair_chunk):
        pair_end = min(pair_start + pair_chunk, n_pairs)
        tx_idx = tx_all[pair_start:pair_end]
        rx_idx = rx_all[pair_start:pair_end]
        residual_pairs = residual[:, rx_idx, tx_idx].permute(1, 0).contiguous()
        frequency_grid = torch.zeros(
            (pair_end - pair_start, state["m_grid"]),
            dtype=state["complex_dtype"],
            device=device,
        )
        bins = state["mode_bins"][state["freq_idx"]]
        frequency_grid[:, bins] = residual_pairs / state["deapod"][
            state["freq_idx"]
        ].conj().view(1, -1)
        adjoint_grid = torch.fft.fft(frequency_grid, dim=-1)
        chunks = []
        for positions in position_chunks():
            positions = positions.to(device=device, dtype=state["real_dtype"])
            if positions.ndim != 2 or positions.shape[1] != 3:
                raise ValueError("position chunks must have shape [N, 3]")
            original_count = int(positions.shape[0])
            if original_count == 0:
                continue
            mask = _validate_point_mask(point_mask, positions)
            if mask is not None:
                valid_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                positions = positions[mask]
                if positions.shape[0] == 0:
                    chunks.append(
                        torch.zeros(
                            original_count,
                            dtype=state["complex_dtype"],
                            device=device,
                        )
                    )
                    continue
            chunk_args = (
                positions,
                adjoint_grid,
                tx_idx,
                rx_idx,
                arr_pos_tx,
                arr_pos_rx,
                state["f_ref"],
                state["df"],
                phase_sign,
                eps,
                g_const,
                state["m_grid"],
                kernel_width,
                state["tau_bins"],
                state["complex_dtype"],
                range_model,
                propagation_model,
                reference_range_m,
                scene_center_m,
            )
            if torch.is_grad_enabled():
                value = torch.utils.checkpoint.checkpoint(
                    _render_adjoint_point_chunk, *chunk_args, use_reentrant=False
                )
            else:
                value = _render_adjoint_point_chunk(*chunk_args)
            if mask is not None:
                value = torch.zeros(
                    original_count,
                    dtype=value.dtype,
                    device=value.device,
                ).index_copy(0, valid_indices, value)
            chunks.append(value)
        if not chunks:
            raise ValueError("position chunk factory yielded no points")
        pair_output = torch.cat(chunks, dim=0)
        output = pair_output if output is None else output + pair_output
    return output
