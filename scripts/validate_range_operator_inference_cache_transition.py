#!/usr/bin/env python3
"""CUDA regression for the shared range-NUFFT inference-to-autograd cache path.

The bounded GeRaF driver prepares native matched-filter targets under
``torch.inference_mode()`` and then performs a differentiable range render in
the same process.  This focused test uses the exact B787 deapodization shape
(600 frequency bins, oversample 2, width 20, float64) without reading B787
data.  It deliberately never clears the shared cache between those phases.
"""

from __future__ import annotations

import inspect

import torch

from rift import range_operator
from rift import serialized_range_operator
from rift.config import cc
from rift.forward_operator import get_kvector
from rift.geraf_signal_operator import (
    matched_filter_from_response_range,
    trace_and_match_magnitude,
)


def check(condition: bool, detail: str) -> None:
    if not condition:
        raise AssertionError(detail)
    print(f"PASS: {detail}", flush=True)


def _cache_key(
    nf_full: int,
    m_grid: int,
    kernel_width: int,
    tau_bins: float,
    device: torch.device,
) -> tuple[object, ...]:
    """Mirror the stable shared-cache key without mutating the cache."""

    return (
        nf_full,
        m_grid,
        kernel_width,
        round(float(tau_bins), 12),
        device.type,
        device.index,
        str(torch.float64),
    )


def _finite_nonzero(*values: torch.Tensor | None) -> bool:
    return all(
        value is not None
        and bool(torch.isfinite(value).all())
        and bool(value.abs().max() > 0.0)
        for value in values
    )


def check_chunk_helper_arity() -> None:
    """Bind both the legacy and extended helper call shapes without CUDA work."""

    legacy_forward = [object()] * 16
    extended_forward = [object()] * 19
    legacy_adjoint = [object()] * 16
    extended_adjoint = [object()] * 19
    inspect.signature(range_operator._render_forward_point_chunk).bind(*legacy_forward)
    inspect.signature(range_operator._render_forward_point_chunk).bind(*extended_forward)
    inspect.signature(range_operator._render_adjoint_point_chunk).bind(*legacy_adjoint)
    inspect.signature(range_operator._render_adjoint_point_chunk).bind(*extended_adjoint)
    print("GERAF_B78716_SMALLFIT_CHUNK_ARITY_PASS", flush=True)


def main() -> None:
    check_chunk_helper_arity()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("range-NUFFT cache-transition validation requires exactly one CUDA device")
    device = torch.device("cuda:0")
    torch.manual_seed(21_067)

    # These are the B787 GeRaF range-NUFFT cache parameters that failed in
    # 12876733.  The geometry is intentionally tiny; all behavior under test
    # is in the shared frequency-axis cache and its autograd consumers.
    nf_full = 600
    oversample = 2
    kernel_width = 20
    m_grid = range_operator._next_power_of_two(nf_full * oversample)
    tau_bins = range_operator._default_tau_bins(kernel_width)
    cache_key = _cache_key(nf_full, m_grid, kernel_width, tau_bins, device)
    if cache_key in range_operator._DEAPOD_CACHE:
        raise AssertionError("cache-transition validation requires a cold dedicated B787 cache key")

    frequencies = torch.linspace(8.5e9, 11.5e9, nf_full, dtype=torch.float64, device=device)
    kvector = get_kvector(frequencies, cc)
    tx_positions = torch.tensor(
        ((10.0, -0.025, 0.010), (10.0, 0.025, -0.010)),
        dtype=torch.float64,
        device=device,
    )
    rx_positions = torch.tensor(
        ((10.0, -0.015, -0.020), (10.0, 0.015, 0.020)),
        dtype=torch.float64,
        device=device,
    )
    sample_positions = torch.tensor(
        ((-0.050, -0.025, 0.010), (0.020, 0.040, -0.015), (0.060, -0.030, 0.025)),
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    amplitude_re = torch.tensor(
        ((0.11, -0.08, 0.06), (0.07, 0.09, -0.05), (-0.04, 0.03, 0.10), (0.05, -0.07, 0.02)),
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    amplitude_im = torch.tensor(
        ((-0.03, 0.05, 0.08), (0.04, -0.06, 0.01), (0.09, 0.02, -0.07), (-0.05, 0.06, 0.03)),
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    pair_amplitudes = torch.complex(amplitude_re, amplitude_im)
    response_re = torch.linspace(
        -0.08, 0.09, nf_full * 4, dtype=torch.float64, device=device
    ).view(nf_full, 2, 2)
    response_im = torch.linspace(
        0.06, -0.05, nf_full * 4, dtype=torch.float64, device=device
    ).view(nf_full, 2, 2)
    prepared_measurement = torch.complex(response_re, response_im)
    operator_kwargs = {
        "phase_sign": -1.0,
        "oversample": oversample,
        "kernel_width": kernel_width,
        "pair_chunk": 2,
        "point_chunk": 2,
        "compute_dtype": torch.float64,
    }

    # Match target preparation: range adjoint / native matched filter first
    # runs under inference mode.  No cache is cleared after this point.
    with torch.inference_mode():
        prepared_mf = matched_filter_from_response_range(
            prepared_measurement,
            frequencies,
            kvector,
            tx_positions,
            rx_positions,
            sample_positions.detach(),
            **operator_kwargs,
        )
    # Read the populated entry directly rather than calling ``_deapodization``
    # again: a second lookup would exercise legacy promotion and could conceal
    # an incorrectly inference-tagged cold construction.
    cold_deapod = range_operator._DEAPOD_CACHE.get(cache_key)
    if cold_deapod is None:
        raise AssertionError(
            "inference-mode native matched-filter preparation did not populate its deapodization cache entry"
        )
    check(
        bool(torch.isfinite(prepared_mf.real).all())
        and bool(torch.isfinite(prepared_mf.imag).all())
        and not cold_deapod.is_inference()
        and not cold_deapod.requires_grad
        and cold_deapod.dtype == torch.complex128
        and cold_deapod.device == device,
        "cold inference-mode native matched-filter preparation stores an ordinary fp64 CUDA deapodization constant",
    )

    # The actual failed transition: differentiable pairwise trace plus native
    # matched-filter readout immediately follows inference preparation.
    response, matched_magnitude = trace_and_match_magnitude(
        frequencies,
        kvector,
        tx_positions,
        rx_positions,
        sample_positions,
        pair_amplitudes,
        sample_positions,
        **operator_kwargs,
    )
    ge_raf_loss = response.abs().square().mean() + matched_magnitude.square().mean()
    ge_raf_loss.backward()
    check(
        bool(torch.isfinite(ge_raf_loss))
        and _finite_nonzero(sample_positions.grad, amplitude_re.grad, amplitude_im.grad),
        "differentiable GeRaF trace plus native matched-filter backward succeeds after inference preparation",
    )

    # Simulate an old long-lived process containing a pre-fix inference cache
    # entry.  This is test setup for promotion behavior, not cache clearing.
    with torch.inference_mode():
        legacy_inference_deapod = cold_deapod.clone()
    if not legacy_inference_deapod.is_inference():
        raise AssertionError("fixture could not create a legacy inference-tensor cache entry")
    range_operator._DEAPOD_CACHE[cache_key] = legacy_inference_deapod
    normalized_deapod = range_operator._deapodization(
        nf_full, m_grid, kernel_width, tau_bins, device, torch.float64
    )
    warm_deapod = range_operator._deapodization(
        nf_full, m_grid, kernel_width, tau_bins, device, torch.float64
    )
    check(
        not normalized_deapod.is_inference()
        and normalized_deapod is warm_deapod
        and torch.equal(normalized_deapod, cold_deapod),
        "a legacy inference cache entry is promoted once to identical ordinary data and warm cache reuse is stable",
    )

    rift_positions = sample_positions.detach()
    rift_re = torch.tensor(
        (0.09, -0.04, 0.06), dtype=torch.float64, device=device, requires_grad=True
    )
    rift_im = torch.tensor(
        (-0.02, 0.07, -0.05), dtype=torch.float64, device=device, requires_grad=True
    )
    rift_weights = torch.complex(rift_re, rift_im)
    rift_kwargs = {
        "phase_sign": -1.0,
        "oversample": oversample,
        "kernel_width": kernel_width,
        "pair_chunk": 2,
        "compute_dtype": torch.float64,
        "range_model": "none",
        "propagation_model": range_operator.BISTATIC_NEAR_FIELD_ABSOLUTE,
    }
    ordinary = range_operator.range_forward_operator(
        frequencies,
        kvector,
        rx_positions,
        tx_positions,
        rift_positions,
        rift_weights,
        point_chunk=2,
        **rift_kwargs,
    )

    ordinary_reference = ordinary.detach()
    ordinary.abs().square().mean().backward()
    check(
        _finite_nonzero(rift_re.grad, rift_im.grad),
        "ordinary absolute-bistatic RIFT forward retains finite weight gradients after the cache transition",
    )

    # Build the serialized consumer only after the ordinary graph has been
    # released.  The two operators must each backpropagate from a genuinely
    # independent graph; constructing both first can accidentally make a
    # graph-reuse failure look like a cache-transition failure.
    serialized_re = torch.tensor(
        (0.09, -0.04, 0.06), dtype=torch.float64, device=device, requires_grad=True
    )
    serialized_im = torch.tensor(
        (-0.02, 0.07, -0.05), dtype=torch.float64, device=device, requires_grad=True
    )
    serialized_weights = torch.complex(serialized_re, serialized_im)

    def rift_chunks() -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (rift_positions[:2], serialized_weights[:2]),
            (rift_positions[2:], serialized_weights[2:]),
        ]

    serialized = serialized_range_operator.range_forward_operator_chunks(
        frequencies,
        kvector,
        rx_positions,
        tx_positions,
        rift_chunks,
        **rift_kwargs,
    )
    check(
        torch.allclose(serialized, ordinary_reference, rtol=1.0e-11, atol=1.0e-11),
        "ordinary and serialized absolute-bistatic RIFT forwards agree through the repaired shared cache",
    )
    serialized_loss = serialized.abs().square().mean()
    serialized_loss.backward()
    check(
        _finite_nonzero(serialized_re.grad, serialized_im.grad),
        "serialized absolute-bistatic RIFT forward retains finite independent weight gradients",
    )

    residual = prepared_measurement.detach()
    adjoint_positions = rift_positions.detach().clone().requires_grad_(True)
    ordinary_adjoint = range_operator.range_adjoint_operator(
        frequencies,
        kvector,
        rx_positions,
        tx_positions,
        adjoint_positions,
        residual,
        point_chunk=2,
        **rift_kwargs,
    )

    ordinary_adjoint_reference = ordinary_adjoint.detach()
    ordinary_adjoint.abs().square().mean().backward()
    check(
        _finite_nonzero(adjoint_positions.grad),
        "ordinary absolute-bistatic RIFT adjoint retains finite position gradients",
    )

    # As above, replay the serialized adjoint from a fresh position leaf after
    # the ordinary adjoint backward has completed.
    serialized_adjoint_positions = rift_positions.detach().clone().requires_grad_(True)

    def position_chunks() -> list[torch.Tensor]:
        return [
            serialized_adjoint_positions[:2],
            serialized_adjoint_positions[2:],
        ]

    serialized_adjoint = serialized_range_operator.range_adjoint_operator_chunks(
        frequencies,
        kvector,
        rx_positions,
        tx_positions,
        position_chunks,
        residual,
        **rift_kwargs,
    )
    torch.cuda.synchronize(device)
    check(
        torch.allclose(serialized_adjoint, ordinary_adjoint_reference, rtol=1.0e-11, atol=1.0e-11)
        and bool(torch.isfinite(ordinary_adjoint.real).all())
        and bool(torch.isfinite(ordinary_adjoint.imag).all()),
        "ordinary and serialized absolute-bistatic RIFT adjoints agree after the same inference-to-autograd cache transition",
    )
    serialized_adjoint_loss = serialized_adjoint.abs().square().mean()
    serialized_adjoint_loss.backward()
    check(
        _finite_nonzero(serialized_adjoint_positions.grad),
        "serialized absolute-bistatic RIFT adjoint retains finite position gradients",
    )
    print("RANGE_OPERATOR_INFERENCE_CACHE_TRANSITION_PASS", flush=True)


if __name__ == "__main__":
    main()
