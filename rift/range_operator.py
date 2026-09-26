"""Range-factorized forward/adjoint operators for point scatterers.

This module keeps the exact per-Tx/Rx Euclidean ranges used by
``forward_operator_lessparallel`` for the training configuration (unity
omega scaling, no pulse spectrum; ``range_model`` selectable and matching
that operator's -- see ``_geom_gain``), but
accelerates the frequency axis.  For each Tx/Rx pair the response on the
uniform frequency grid is a type-1 NUFFT:

    S[i] = sum_n b_n exp(1j * x_n * (i - i_ref))

where ``b_n`` contains the complex scatterer weight, geometric gain, and
carrier phase at reference bin ``i_ref = nf_full // 2``.  Centering the
Fourier modes this way is algebraically identical to using the first
frequency as the reference, but it keeps the rendered modes away from the
oversampled FFT's Nyquist edge and is required for the fp64 exactness gate.

The NUFFT is implemented as differentiable Gaussian gridding in pure torch.
The integer support selection uses ``floor(u)`` and is therefore
piecewise-constant; gradients flow through the smooth Gaussian tap weights
and carrier/geometric factors.  The only omitted derivative is the support
jump at a bin boundary, whose contribution is at the kernel tail for the
locked defaults.

Real CSV data reaches train.py with float32 frequencies.  The operator
therefore validates that the input is consistent with a uniform linspace,
then reconstructs and uses that ideal fp64 grid internally.  Brute-force
comparisons that evaluate the raw float32 frequency values can differ at
the 1e-4--1e-2 level from this reconstruction; that is frequency rounding,
not an operator mismatch.

Memory (measured 2026-07-26): with checkpointing on, peak activation memory
is set by ONE point-chunk's backward-recompute graph and is therefore
O(point_chunk * pair_chunk), NOT O(n_points):

    chunk graph  = point_chunk * pair_chunk * 672 B   (fp64; 416 B at fp32)
    retained     = 256 * ceil(n_points/point_chunk) * m_grid * 16 B

measured by summing the saved-tensor storages of one
``_render_forward_point_chunk`` graph.  The only O(n_points) terms are the
second line above and the scene parameters themselves, both small (a g96
scene is 1.4 GB of AdamW state on ONE rank).  Consequence: scene density is
not VRAM-limited on any current PACE card -- g96 at point_chunk=16384 peaks
under 4 GB even unsharded.  Multi-GPU sharding buys THROUGHPUT here, not
capacity.

Gradient checkpointing (2026-07-08): ``point_chunk`` only bounds each
chunk's *forward* peak memory -- PyTorch retains every chunk's
intermediates for backward regardless of chunk size, so total backward
memory still grows as O(n_points) and eventually OOMs at large N (measured:
N=1e6 OOMs even at the smallest tried ``point_chunk``). Both operators'
point-chunk bodies are therefore wrapped in
``torch.utils.checkpoint.checkpoint`` (only when ``torch.is_grad_enabled()``
-- under ``torch.no_grad()``, e.g. backprojection init, there is no graph
to save so checkpointing would be pure overhead) via
``_render_forward_point_chunk``/``_render_adjoint_point_chunk``. Both
helpers are written to be PURELY FUNCTIONAL (return a fresh tensor, never
read/write a tensor from an outer scope): checkpointing recomputes the
wrapped function during backward, and recomputing an in-place accumulation
into a tensor shared across chunks (the original ``_spread_to_grid(grid,
...)``/``out[slice] +=`` pattern, with ``grid``/``out`` allocated once and
mutated by every chunk) would double-accumulate on recompute, or corrupt
the autograd version counter. Accumulation across chunks now happens
strictly OUTSIDE the checkpoint boundary via out-of-place tensor ops
(``grid = grid + chunk_grid``, ``out = out + torch.cat(pair_chunk_outputs)``).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint

from rift.coherent_radar_geometry import (
    BISTATIC_NEAR_FIELD_ABSOLUTE,
    point_pair_path_lengths,
)
from rift.config import cc


_DEAPOD_CACHE = {}


def _ordinary_cached_constant(value: torch.Tensor) -> torch.Tensor:
    """Return ``value`` as a regular no-grad tensor with the same value semantics.

    A tensor made under ``torch.inference_mode()`` cannot be saved by a later
    autograd operation, even when it is a numerical constant.  Range
    deapodization is reused by preparation/evaluation and differentiable
    rendering, so cache entries must never retain that inference-only tensor
    kind.  Copy only the stale entry; ordinary cache hits remain allocation
    free and retain the exact cache key, layout, dtype, device, and values.
    """

    is_inference = getattr(value, "is_inference", None)
    if not callable(is_inference) or not bool(is_inference()):
        return value
    # ``inference_mode(False)`` is essential here: ``torch.no_grad()`` alone
    # nested inside an outer inference context would still create an inference
    # tensor.  The copy is an ordinary no-grad constant on the same device.
    with torch.inference_mode(False), torch.no_grad():
        return value.clone(memory_format=torch.preserve_format)


def _complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
    if real_dtype == torch.float64:
        return torch.complex128
    if real_dtype == torch.float32:
        return torch.complex64
    raise ValueError("compute_dtype must be torch.float32 or torch.float64")


def _next_power_of_two(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


def _default_tau_bins(kernel_width: int) -> float:
    """Gaussian width in grid-bin units.

    A centered-mode sweep on random type-1 NUFFT cases found the fp64 minima
    near J=16 -> 0.86, J=20 -> 1.08, J=24 -> 1.28.  The linear rule below
    matches those values closely.  The public default is J=20 because J=16
    sits around the 1e-8 relative-error floor for this positive-frequency
    workload, while J=20 clears the 1e-9 validation gate.
    """
    if kernel_width < 4:
        raise ValueError("kernel_width must be at least 4")
    return 0.05 * float(kernel_width) + 0.08


def _validate_and_reconstruct_grid(
    freqs_full: torch.Tensor,
    kvector_full: torch.Tensor,
) -> torch.Tensor:
    if freqs_full.ndim != 1 or kvector_full.ndim != 1:
        raise ValueError("freqs_full and kvector_full must be 1D tensors")
    if freqs_full.shape[0] != kvector_full.shape[0]:
        raise ValueError("freqs_full and kvector_full must have matching lengths")
    if freqs_full.shape[0] < 2:
        raise ValueError("range operator requires at least two frequency bins")

    device = freqs_full.device
    nf_full = freqs_full.shape[0]
    f0 = float(freqs_full[0].detach().cpu())
    f1 = float(freqs_full[-1].detach().cpu())
    ideal = torch.linspace(f0, f1, nf_full, dtype=torch.float64, device=device)
    df = ideal[1] - ideal[0]
    if df <= 0:
        raise ValueError("freqs_full must be strictly increasing")

    max_freq_dev = (freqs_full.to(torch.float64) - ideal).abs().max()
    freq_tol = 1e-2 * df.abs()
    if max_freq_dev >= freq_tol:
        ratio = float(max_freq_dev / df.abs())
        raise ValueError(
            "freqs_full is not consistent with a uniform linspace "
            f"(max deviation / df = {ratio:.3e}); see "
            "docs/EXPERIMENT_NOTES.md#plan-range-operator-and-continuous-points section 3.4"
        )

    k_expected = (2.0 * torch.pi * ideal) / cc
    dk = k_expected[1] - k_expected[0]
    max_k_dev = (kvector_full.to(device=device, dtype=torch.float64) - k_expected).abs().max()
    k_tol = 1e-2 * dk.abs()
    if max_k_dev >= k_tol:
        ratio = float(max_k_dev / dk.abs())
        raise ValueError(
            "kvector_full is inconsistent with freqs_full "
            f"(max deviation / dk = {ratio:.3e})"
        )

    return ideal


def _normalize_freq_indices(
    freq_indices: Optional[torch.Tensor],
    nf_full: int,
    device: torch.device,
) -> torch.Tensor:
    if freq_indices is None:
        return torch.arange(nf_full, dtype=torch.long, device=device)
    idx = torch.as_tensor(freq_indices, dtype=torch.long, device=device)
    if idx.ndim != 1:
        raise ValueError("freq_indices must be a 1D tensor")
    if idx.numel() == 0:
        raise ValueError("freq_indices must not be empty")
    if int(idx.min()) < 0 or int(idx.max()) >= nf_full:
        raise ValueError("freq_indices contains bins outside freqs_full")
    return idx


def _mode_bins(nf_full: int, m_grid: int, device: torch.device) -> torch.Tensor:
    center = nf_full // 2
    modes = torch.arange(nf_full, dtype=torch.long, device=device) - center
    return torch.remainder(modes, m_grid)


def _deapodization(
    nf_full: int,
    m_grid: int,
    kernel_width: int,
    tau_bins: float,
    device: torch.device,
    real_dtype: torch.dtype,
) -> torch.Tensor:
    key = (
        nf_full,
        m_grid,
        kernel_width,
        round(float(tau_bins), 12),
        device.type,
        device.index,
        str(real_dtype),
    )
    cached = _DEAPOD_CACHE.get(key)
    if cached is not None:
        regular = _ordinary_cached_constant(cached)
        if regular is not cached:
            _DEAPOD_CACHE[key] = regular
        return regular

    # Preparation may call this inside ``torch.inference_mode()``.  Build the
    # reusable constant in an explicitly ordinary no-grad region so a later
    # differentiable forward or adjoint can safely save it for backward.
    with torch.inference_mode(False), torch.no_grad():
        complex_dtype = _complex_dtype(real_dtype)
        grid = torch.zeros(m_grid, dtype=complex_dtype, device=device)
        u0 = torch.zeros(1, dtype=real_dtype, device=device)
        floor_u0 = torch.floor(u0)

        for j in range(kernel_width):
            m_unwrapped = floor_u0 - kernel_width // 2 + 1 + j
            m = torch.remainder(m_unwrapped.to(torch.long), m_grid)
            phi = torch.exp(-((u0 - m_unwrapped) ** 2) / (4.0 * tau_bins))
            grid.index_add_(0, m, phi.to(complex_dtype))

        d_full = m_grid * torch.fft.ifft(grid, dim=0)
        bins = _mode_bins(nf_full, m_grid, device)
        d = d_full[bins]
    _DEAPOD_CACHE[key] = d
    return d


def _geom_gain(r_tx_pair, r_rx_pair, r_sum, range_model: str, g_const: float, eps: float):
    """Geometric spreading, identical to forward_operator_lessparallel's.

    Only the FREQUENCY dependence has to factor through ``r_sum`` for the
    range factorization to hold -- this gain is evaluated per (point, pair)
    and multiplied into the NUFFT amplitude, so it may be any function of the
    two leg lengths.

    "product" = 1/(R_tx*R_rx) is the physically correct two-way spherical
    spreading and is the kernel SpINR/SpINRv2 use; "sum2" = 1/(R_tx+R_rx)^2
    is the historical RIFT default.  They do *not* coincide in the monostatic
    limit: with R_tx = R_rx = R, product is 1/R^2 whereas sum2 is 1/(2R)^2,
    a factor of four smaller.  At this project's operating point the aperture
    changes the legs only slightly, but that fixed convention factor remains;
    it may have been absorbed by learned gains in historical recipes.  New
    fixed-scale recipes must therefore declare the choice explicitly.
    """
    if range_model == "product":
        return g_const / (r_tx_pair * r_rx_pair + eps)
    if range_model == "sum2":
        return g_const / (r_sum.square() + eps)
    if range_model == "none":
        return g_const * torch.ones_like(r_sum)
    raise ValueError("range_model must be one of {'product','sum2','none'}")


def _pair_indices(num_tx: int, num_rx: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    tx_idx = torch.arange(num_tx, dtype=torch.long, device=device).repeat_interleave(num_rx)
    rx_idx = torch.arange(num_rx, dtype=torch.long, device=device).repeat(num_tx)
    return tx_idx, rx_idx


def _spread_to_grid(
    grid: torch.Tensor,
    values: torch.Tensor,
    u: torch.Tensor,
    kernel_width: int,
    tau_bins: float,
) -> None:
    """Accumulate ``values[Nc,P]`` at nonuniform bin coordinates ``u[Nc,P]``."""
    n_pairs, m_grid = grid.shape
    pair_offsets = (torch.arange(n_pairs, device=grid.device, dtype=torch.long) * m_grid).view(1, n_pairs)
    grid_flat = grid.reshape(-1)
    floor_u = torch.floor(u)

    for j in range(kernel_width):
        m_unwrapped = floor_u - kernel_width // 2 + 1 + j
        m = torch.remainder(m_unwrapped.to(torch.long), m_grid)
        phi = torch.exp(-((u - m_unwrapped) ** 2) / (4.0 * tau_bins))
        idx = pair_offsets + m
        src = (values * phi).to(grid.dtype)
        try:
            grid_flat.index_add_(0, idx.reshape(-1), src.reshape(-1))
        except RuntimeError:
            # Older torch builds lacked complex index_add_ on some devices.
            real = grid_flat.real.contiguous()
            imag = grid_flat.imag.contiguous()
            real.index_add_(0, idx.reshape(-1), src.real.reshape(-1))
            imag.index_add_(0, idx.reshape(-1), src.imag.reshape(-1))
            grid.copy_(torch.complex(real, imag).view_as(grid))


def _gather_from_grid(
    adj_grid: torch.Tensor,
    u: torch.Tensor,
    kernel_width: int,
    tau_bins: float,
) -> torch.Tensor:
    """Adjoint of _spread_to_grid for real Gaussian tap weights."""
    n_pairs, m_grid = adj_grid.shape
    pair_offsets = (torch.arange(n_pairs, device=adj_grid.device, dtype=torch.long) * m_grid).view(1, n_pairs)
    adj_flat = adj_grid.reshape(-1)
    floor_u = torch.floor(u)
    out = torch.zeros_like(u, dtype=adj_grid.dtype)

    for j in range(kernel_width):
        m_unwrapped = floor_u - kernel_width // 2 + 1 + j
        m = torch.remainder(m_unwrapped.to(torch.long), m_grid)
        phi = torch.exp(-((u - m_unwrapped) ** 2) / (4.0 * tau_bins))
        idx = pair_offsets + m
        out = out + adj_flat[idx] * phi.to(adj_grid.dtype)
    return out


def _render_forward_point_chunk(
    pos, weights, arr_pos_tx, arr_pos_rx, tx_idx, rx_idx,
    f_ref, df, phase_sign, eps, g_const, m_grid, kernel_width, tau_bins, complex_dtype,
    range_model="sum2", propagation_model=BISTATIC_NEAR_FIELD_ABSOLUTE,
    reference_range_m=None, scene_center_m=(0.0, 0.0, 0.0),
):
    """One point_chunk's contribution to a pair_chunk's gridded spectrum, for
    range_forward_operator. Purely functional -- returns a FRESH
    [n_pair_chunk, m_grid] tensor, never touches a tensor from an outer
    scope -- so it is safe to wrap in torch.utils.checkpoint.checkpoint
    (see module docstring: checkpointing an in-place accumulation into a
    tensor shared across chunks would double-accumulate on recompute).
    """
    r_tx_pair, r_rx_pair, r_sum = point_pair_path_lengths(
        pos,
        arr_pos_tx,
        arr_pos_rx,
        tx_idx,
        rx_idx,
        propagation_model=propagation_model,
        reference_range_m=reference_range_m,
        scene_center_m=scene_center_m,
        eps=eps,
    )
    geom = _geom_gain(r_tx_pair, r_rx_pair, r_sum, range_model, g_const, eps)
    phase_ref = phase_sign * (2.0 * torch.pi * f_ref / cc) * r_sum
    amp = weights[:, None] * geom.to(complex_dtype) * torch.polar(torch.ones_like(phase_ref), phase_ref)

    x = phase_sign * (2.0 * torch.pi * df / cc) * r_sum
    u = torch.remainder(x * (float(m_grid) / (2.0 * torch.pi)), float(m_grid))

    n_pair_chunk = tx_idx.shape[0]
    grid_chunk = torch.zeros((n_pair_chunk, m_grid), dtype=complex_dtype, device=pos.device)
    _spread_to_grid(grid_chunk, amp, u, kernel_width, tau_bins)  # mutates the FRESH local tensor only
    return grid_chunk


def _render_adjoint_point_chunk(
    pos, adj_grid, tx_idx, rx_idx, arr_pos_tx, arr_pos_rx,
    f_ref, df, phase_sign, eps, g_const, m_grid, kernel_width, tau_bins, complex_dtype,
    range_model="sum2", propagation_model=BISTATIC_NEAR_FIELD_ABSOLUTE,
    reference_range_m=None, scene_center_m=(0.0, 0.0, 0.0),
):
    """One point_chunk's contribution to range_adjoint_operator's output for
    one pair_chunk's adj_grid. Purely functional (returns a fresh
    [chunk_size] tensor) for the same checkpointing-safety reason as
    _render_forward_point_chunk.
    """
    r_tx_pair, r_rx_pair, r_sum = point_pair_path_lengths(
        pos,
        arr_pos_tx,
        arr_pos_rx,
        tx_idx,
        rx_idx,
        propagation_model=propagation_model,
        reference_range_m=reference_range_m,
        scene_center_m=scene_center_m,
        eps=eps,
    )
    geom = _geom_gain(r_tx_pair, r_rx_pair, r_sum, range_model, g_const, eps)
    phase_ref = phase_sign * (2.0 * torch.pi * f_ref / cc) * r_sum
    amp = geom.to(complex_dtype) * torch.polar(torch.ones_like(phase_ref), phase_ref)

    x = phase_sign * (2.0 * torch.pi * df / cc) * r_sum
    u = torch.remainder(x * (float(m_grid) / (2.0 * torch.pi)), float(m_grid))
    gathered = _gather_from_grid(adj_grid, u, kernel_width, tau_bins)
    return (amp.conj() * gathered).sum(dim=1)


def range_forward_operator(
    freqs_full,
    kvector_full,
    arr_pos_rx,
    arr_pos_tx,
    scatterer_pos,
    scatterer_weights,
    phase_sign: float = 1.0,
    eps: float = 1e-9,
    freq_indices=None,
    oversample: int = 2,
    kernel_width: int = 20,
    pair_chunk: int = 64,
    point_chunk: int = 262144,
    compute_dtype=torch.float64,
    range_model: str = "sum2",
    propagation_model: str = BISTATIC_NEAR_FIELD_ABSOLUTE,
    reference_range_m: float = None,
    scene_center_m=(0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Render S-parameters with shape ``[nf_sel or nf_full, Rx, Tx]``.

    ``freqs_full`` must be the full uniform frequency grid.  Pass
    ``freq_indices`` to gather train.py's random frequency subset after the
    NUFFT render; the selected subset itself need not be uniform.
    The output dtype follows ``compute_dtype``: complex128 for fp64 compute,
    complex64 for fp32 compute.
    """
    if oversample < 2:
        raise ValueError("oversample must be >= 2")
    if pair_chunk <= 0 or point_chunk <= 0:
        raise ValueError("pair_chunk and point_chunk must be positive")

    device = scatterer_pos.device
    freqs_full = torch.as_tensor(freqs_full, device=device)
    kvector_full = torch.as_tensor(kvector_full, device=device)
    ideal_freqs = _validate_and_reconstruct_grid(freqs_full, kvector_full)
    nf_full = ideal_freqs.shape[0]
    freq_idx = _normalize_freq_indices(freq_indices, nf_full, device)

    real_dtype = compute_dtype
    complex_dtype = _complex_dtype(real_dtype)
    ideal_freqs = ideal_freqs.to(real_dtype)
    df = ideal_freqs[1] - ideal_freqs[0]
    center = nf_full // 2
    f_ref = ideal_freqs[center]

    m_grid = _next_power_of_two(int(math.ceil(float(oversample) * nf_full)))
    tau_bins = _default_tau_bins(kernel_width)
    deapod = _deapodization(nf_full, m_grid, kernel_width, tau_bins, device, real_dtype)

    arr_pos_rx = arr_pos_rx.to(device=device, dtype=real_dtype)
    arr_pos_tx = arr_pos_tx.to(device=device, dtype=real_dtype)
    scatterer_pos = scatterer_pos.to(device=device, dtype=real_dtype)
    scatterer_weights = scatterer_weights.to(device=device, dtype=complex_dtype)

    n_points = scatterer_pos.shape[0]
    num_rx = arr_pos_rx.shape[0]
    num_tx = arr_pos_tx.shape[0]
    n_pairs = num_rx * num_tx
    tx_all, rx_all = _pair_indices(num_tx, num_rx, device)
    g_const = 1.0 / ((4.0 * torch.pi) ** 2)

    out_pairs = []
    mode_bins = _mode_bins(nf_full, m_grid, device)

    for pair_start in range(0, n_pairs, pair_chunk):
        pair_end = min(pair_start + pair_chunk, n_pairs)
        tx_idx = tx_all[pair_start:pair_end]
        rx_idx = rx_all[pair_start:pair_end]
        n_pair_chunk = pair_end - pair_start
        grid = torch.zeros((n_pair_chunk, m_grid), dtype=complex_dtype, device=device)

        for point_start in range(0, n_points, point_chunk):
            point_end = min(point_start + point_chunk, n_points)
            pos = scatterer_pos[point_start:point_end]
            weights = scatterer_weights[point_start:point_end]

            chunk_args = (
                pos, weights, arr_pos_tx, arr_pos_rx, tx_idx, rx_idx,
                f_ref, df, phase_sign, eps, g_const, m_grid, kernel_width, tau_bins, complex_dtype,
                range_model,
                propagation_model,
                reference_range_m,
                scene_center_m,
            )
            if torch.is_grad_enabled():
                # Checkpointed: only this chunk's own [n_pair_chunk, m_grid]
                # tensor is kept live going into backward (recomputed on
                # demand), instead of every point_chunk's intermediates
                # staying resident for the whole pair_chunk -- see module
                # docstring. use_reentrant=False tolerates the non-tensor
                # (int/float/dtype) args mixed in below.
                grid_chunk = torch.utils.checkpoint.checkpoint(
                    _render_forward_point_chunk, *chunk_args, use_reentrant=False
                )
            else:
                grid_chunk = _render_forward_point_chunk(*chunk_args)
            grid = grid + grid_chunk  # out-of-place: safe across checkpoint recomputation

        fft_vals = m_grid * torch.fft.ifft(grid, dim=-1)
        rendered = fft_vals[:, mode_bins] / deapod.view(1, nf_full)
        out_pairs.append(rendered[:, freq_idx])

    pairs_freq = torch.cat(out_pairs, dim=0)  # [Tx*Rx, nf_sel], tx-major/rx-minor
    out = pairs_freq.view(num_tx, num_rx, -1).permute(2, 1, 0).contiguous()
    return out


def range_adjoint_operator(
    freqs_full,
    kvector_full,
    arr_pos_rx,
    arr_pos_tx,
    scatterer_pos,
    S_resid,
    phase_sign: float = 1.0,
    eps: float = 1e-9,
    freq_indices=None,
    oversample: int = 2,
    kernel_width: int = 20,
    pair_chunk: int = 64,
    point_chunk: int = 262144,
    compute_dtype=torch.float64,
    range_model: str = "sum2",
    propagation_model: str = BISTATIC_NEAR_FIELD_ABSOLUTE,
    reference_range_m: float = None,
    scene_center_m=(0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Adjoint with respect to complex scatterer weights.

    Returns a complex tensor of shape ``[N]``.  This is the conjugate
    transpose of ``range_forward_operator``'s gridded linear map, including
    optional ``freq_indices``.
    """
    if oversample < 2:
        raise ValueError("oversample must be >= 2")
    if pair_chunk <= 0 or point_chunk <= 0:
        raise ValueError("pair_chunk and point_chunk must be positive")

    device = scatterer_pos.device
    freqs_full = torch.as_tensor(freqs_full, device=device)
    kvector_full = torch.as_tensor(kvector_full, device=device)
    ideal_freqs = _validate_and_reconstruct_grid(freqs_full, kvector_full)
    nf_full = ideal_freqs.shape[0]
    freq_idx = _normalize_freq_indices(freq_indices, nf_full, device)

    real_dtype = compute_dtype
    complex_dtype = _complex_dtype(real_dtype)
    ideal_freqs = ideal_freqs.to(real_dtype)
    df = ideal_freqs[1] - ideal_freqs[0]
    center = nf_full // 2
    f_ref = ideal_freqs[center]

    m_grid = _next_power_of_two(int(math.ceil(float(oversample) * nf_full)))
    tau_bins = _default_tau_bins(kernel_width)
    deapod = _deapodization(nf_full, m_grid, kernel_width, tau_bins, device, real_dtype)
    mode_bins = _mode_bins(nf_full, m_grid, device)

    arr_pos_rx = arr_pos_rx.to(device=device, dtype=real_dtype)
    arr_pos_tx = arr_pos_tx.to(device=device, dtype=real_dtype)
    scatterer_pos = scatterer_pos.to(device=device, dtype=real_dtype)
    s_resid = S_resid.to(device=device, dtype=complex_dtype)

    n_points = scatterer_pos.shape[0]
    num_rx = arr_pos_rx.shape[0]
    num_tx = arr_pos_tx.shape[0]
    n_pairs = num_rx * num_tx
    tx_all, rx_all = _pair_indices(num_tx, num_rx, device)
    g_const = 1.0 / ((4.0 * torch.pi) ** 2)

    out = torch.zeros(n_points, dtype=complex_dtype, device=device)

    for pair_start in range(0, n_pairs, pair_chunk):
        pair_end = min(pair_start + pair_chunk, n_pairs)
        tx_idx = tx_all[pair_start:pair_end]
        rx_idx = rx_all[pair_start:pair_end]
        n_pair_chunk = pair_end - pair_start

        resid_pairs = s_resid[:, rx_idx, tx_idx].permute(1, 0).contiguous()  # [P, nf_sel]
        freq_grid = torch.zeros((n_pair_chunk, m_grid), dtype=complex_dtype, device=device)
        freq_grid[:, mode_bins[freq_idx]] = resid_pairs / deapod[freq_idx].conj().view(1, -1)
        adj_grid = torch.fft.fft(freq_grid, dim=-1)

        pair_chunk_outputs = []
        for point_start in range(0, n_points, point_chunk):
            point_end = min(point_start + point_chunk, n_points)
            pos = scatterer_pos[point_start:point_end]

            chunk_args = (
                pos, adj_grid, tx_idx, rx_idx, arr_pos_tx, arr_pos_rx,
                f_ref, df, phase_sign, eps, g_const, m_grid, kernel_width, tau_bins, complex_dtype,
                range_model,
                propagation_model,
                reference_range_m,
                scene_center_m,
            )
            if torch.is_grad_enabled():
                chunk_out = torch.utils.checkpoint.checkpoint(
                    _render_adjoint_point_chunk, *chunk_args, use_reentrant=False
                )
            else:
                chunk_out = _render_adjoint_point_chunk(*chunk_args)
            pair_chunk_outputs.append(chunk_out)

        # This pair_chunk's (disjoint, one-per-point-range) chunk outputs
        # concatenate back to [n_points]; accumulate out-of-place across
        # pair_chunks -- same "no in-place mutation of a tensor spanning a
        # checkpoint boundary" rule as the forward operator above.
        out = out + torch.cat(pair_chunk_outputs, dim=0)

    return out
