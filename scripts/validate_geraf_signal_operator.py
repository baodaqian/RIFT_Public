#!/usr/bin/env python
"""Focused CPU validator for GeRaF's pair-dependent forward + range MF.

The oracle is an independent complex128 implementation of the two equations,
not another call into RIFT's gridding code.  Both phase-sign conventions are
checked for values and gradients.  The test is intentionally small enough for
a login-node CPU validation; it does not train a model or read experiment data.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rift.config import cc  # noqa: E402
from rift.forward_operator import get_kvector  # noqa: E402
from rift.geraf_signal_operator import (  # noqa: E402
    bistatic_pair_positions,
    matched_filter_from_response_range,
    pairwise_range_forward_operator,
    trace_and_match_magnitude,
)


EPS = 1.0e-30


def direct_pairwise_forward(
    frequencies: torch.Tensor,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    sample_positions: torch.Tensor,
    pair_sample_amplitudes: torch.Tensor,
    *,
    phase_sign: float,
) -> torch.Tensor:
    """Direct GeRaF signal trace, returned as [F,Rx,Tx]."""

    frequencies = frequencies.to(dtype=torch.float64, device=sample_positions.device)
    points = sample_positions.to(torch.float64)
    tx_pair, rx_pair = bistatic_pair_positions(
        tx_positions.to(torch.float64), rx_positions.to(torch.float64)
    )
    r_tx = torch.linalg.vector_norm(
        points.unsqueeze(0) - tx_pair.unsqueeze(1), dim=-1
    )
    r_rx = torch.linalg.vector_norm(
        points.unsqueeze(0) - rx_pair.unsqueeze(1), dim=-1
    )
    r_sum = r_tx + r_rx  # [P,N]
    phase = (
        float(phase_sign)
        * (2.0 * torch.pi / cc)
        * frequencies[:, None, None]
        * r_sum[None, :, :]
    )
    kernel = torch.polar(torch.ones_like(phase), phase)
    pair_frequency = torch.einsum(
        "pn,fpn->pf",
        pair_sample_amplitudes.to(torch.complex128),
        kernel.to(torch.complex128),
    )
    num_tx, num_rx = tx_positions.shape[0], rx_positions.shape[0]
    return pair_frequency.reshape(num_tx, num_rx, -1).permute(2, 1, 0).contiguous()


def direct_phase_only_matched_filter(
    response: torch.Tensor,
    frequencies: torch.Tensor,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    query_points: torch.Tensor,
    *,
    phase_sign: float,
) -> torch.Tensor:
    """Direct GeRaF Eq. 3 complex MF amplitude (no range gain/four-pi)."""

    frequencies = frequencies.to(dtype=torch.float64, device=query_points.device)
    query = query_points.to(torch.float64)
    tx_pair, rx_pair = bistatic_pair_positions(
        tx_positions.to(torch.float64), rx_positions.to(torch.float64)
    )
    r_tx = torch.linalg.vector_norm(
        query.unsqueeze(0) - tx_pair.unsqueeze(1), dim=-1
    )
    r_rx = torch.linalg.vector_norm(
        query.unsqueeze(0) - rx_pair.unsqueeze(1), dim=-1
    )
    r_sum = r_tx + r_rx  # [P,Q]
    conjugate_phase = (
        -float(phase_sign)
        * (2.0 * torch.pi / cc)
        * frequencies[:, None, None]
        * r_sum[None, :, :]
    )
    kernel = torch.polar(torch.ones_like(conjugate_phase), conjugate_phase)
    pair_frequency = response.permute(2, 1, 0).reshape(-1, response.shape[0])
    return torch.einsum(
        "pf,fpq->q", pair_frequency.to(torch.complex128), kernel.to(torch.complex128)
    )


def relative_l2(observed: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        (observed - expected).norm().detach().cpu()
        / expected.norm().detach().cpu().clamp_min(EPS)
    )


def maximum_absolute(observed: torch.Tensor, expected: torch.Tensor) -> float:
    return float((observed - expected).abs().max().detach().cpu())


@dataclass
class Gate:
    name: str
    value: float
    threshold: float

    @property
    def passed(self) -> bool:
        return math.isfinite(self.value) and self.value <= self.threshold


def make_geometry(seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Deliberately nonuniform calibrated coordinates expose pair-order errors.
    tx = torch.tensor(
        [[10.012, -0.031, 0.017], [9.991, 0.046, -0.009], [10.018, 0.093, 0.025]],
        dtype=torch.float64,
    )
    rx = torch.tensor(
        [[10.004, -0.052, -0.037], [9.986, 0.011, 0.028]], dtype=torch.float64
    )
    points = (torch.rand(11, 3, generator=generator, dtype=torch.float64) - 0.5) * 0.28
    amplitudes = 0.05 + torch.rand(
        tx.shape[0] * rx.shape[0], points.shape[0], generator=generator, dtype=torch.float64
    )
    frequencies = 9.85e9 + torch.arange(24, dtype=torch.float64) * (300.0e6 / 24.0)
    return frequencies, tx, rx, points, amplitudes


def value_gates(
    frequencies,
    tx,
    rx,
    points,
    amplitudes,
    sign,
    args,
):
    kvector = get_kvector(frequencies, cc)
    direct_response = direct_pairwise_forward(
        frequencies, tx, rx, points, amplitudes, phase_sign=sign
    )
    direct_mf = direct_phase_only_matched_filter(
        direct_response, frequencies, tx, rx, points, phase_sign=sign
    )
    range_response = pairwise_range_forward_operator(
        frequencies,
        kvector,
        tx,
        rx,
        points,
        amplitudes,
        phase_sign=sign,
        oversample=args.oversample,
        kernel_width=args.kernel_width,
        pair_chunk=4,
        point_chunk=5,
        compute_dtype=torch.float64,
    )
    range_mf = matched_filter_from_response_range(
        range_response,
        frequencies,
        kvector,
        tx,
        rx,
        points,
        phase_sign=sign,
        oversample=args.oversample,
        kernel_width=args.kernel_width,
        pair_chunk=4,
        point_chunk=5,
        compute_dtype=torch.float64,
    )
    traced_response, traced_magnitude = trace_and_match_magnitude(
        frequencies,
        kvector,
        tx,
        rx,
        points,
        amplitudes,
        points,
        phase_sign=sign,
        oversample=args.oversample,
        kernel_width=args.kernel_width,
        pair_chunk=4,
        point_chunk=5,
        compute_dtype=torch.float64,
    )
    direct_magnitude = direct_mf.abs()
    gates = [
        Gate(f"sign {sign:+.0f} response rel-L2", relative_l2(range_response, direct_response), args.value_rtol),
        Gate(f"sign {sign:+.0f} MF amplitude rel-L2", relative_l2(range_mf, direct_mf), args.mf_rtol),
        Gate(f"sign {sign:+.0f} MF magnitude rel-L2", relative_l2(range_mf.abs(), direct_magnitude), args.magnitude_rtol),
        Gate(f"sign {sign:+.0f} wrapper response identity", maximum_absolute(traced_response, range_response), 0.0),
        Gate(f"sign {sign:+.0f} wrapper magnitude identity", maximum_absolute(traced_magnitude, range_mf.abs()), 0.0),
    ]
    if tuple(range_response.shape) != (frequencies.numel(), rx.shape[0], tx.shape[0]):
        gates.append(Gate(f"sign {sign:+.0f} [F,Rx,Tx] shape", math.inf, 0.0))
    return gates, range_response.detach(), traced_magnitude.detach()


def gradient_gates(frequencies, tx, rx, base_points, base_amplitudes, sign, args):
    kvector = get_kvector(frequencies, cc)
    probe = torch.linspace(0.3, 1.7, base_points.shape[0], dtype=torch.float64)

    direct_points = base_points.detach().clone().requires_grad_(True)
    direct_amplitudes = base_amplitudes.detach().clone().requires_grad_(True)
    direct_response = direct_pairwise_forward(
        frequencies, tx, rx, direct_points, direct_amplitudes, phase_sign=sign
    )
    direct_mf = direct_phase_only_matched_filter(
        direct_response, frequencies, tx, rx, direct_points, phase_sign=sign
    )
    scale = direct_mf.detach().abs().max().clamp_min(EPS)
    direct_loss = (direct_mf.abs() * probe).sum() / scale
    direct_amp_grad, direct_point_grad = torch.autograd.grad(
        direct_loss, (direct_amplitudes, direct_points)
    )

    range_points = base_points.detach().clone().requires_grad_(True)
    range_amplitudes = base_amplitudes.detach().clone().requires_grad_(True)
    _, range_magnitude = trace_and_match_magnitude(
        frequencies,
        kvector,
        tx,
        rx,
        range_points,
        range_amplitudes,
        range_points,
        phase_sign=sign,
        oversample=args.oversample,
        kernel_width=args.kernel_width,
        pair_chunk=4,
        point_chunk=5,
        compute_dtype=torch.float64,
    )
    range_loss = (range_magnitude * probe).sum() / scale
    range_amp_grad, range_point_grad = torch.autograd.grad(
        range_loss, (range_amplitudes, range_points)
    )
    return [
        Gate(
            f"sign {sign:+.0f} amplitude gradient rel-L2",
            relative_l2(range_amp_grad, direct_amp_grad),
            args.gradient_rtol,
        ),
        Gate(
            f"sign {sign:+.0f} point gradient rel-L2",
            relative_l2(range_point_grad, direct_point_grad),
            args.point_gradient_rtol,
        ),
    ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--oversample", type=int, default=2)
    parser.add_argument("--kernel-width", type=int, default=20)
    parser.add_argument("--value-rtol", type=float, default=2.0e-8)
    parser.add_argument("--mf-rtol", type=float, default=4.0e-8)
    parser.add_argument("--magnitude-rtol", type=float, default=4.0e-8)
    parser.add_argument("--gradient-rtol", type=float, default=3.0e-5)
    parser.add_argument("--point-gradient-rtol", type=float, default=1.0e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.oversample < 2 or args.kernel_width < 4:
        raise ValueError("need oversample>=2 and kernel_width>=4")
    torch.manual_seed(args.seed)
    frequencies, tx, rx, points, amplitudes = make_geometry(args.seed)
    gates = []
    signed_outputs = {}
    for sign in (-1.0, 1.0):
        with torch.no_grad():
            sign_gates, response, magnitude = value_gates(
                frequencies, tx, rx, points, amplitudes, sign, args
            )
        gates.extend(sign_gates)
        gates.extend(
            gradient_gates(frequencies, tx, rx, points, amplitudes, sign, args)
        )
        signed_outputs[sign] = (response, magnitude)

    negative_response, negative_magnitude = signed_outputs[-1.0]
    positive_response, positive_magnitude = signed_outputs[1.0]
    gates.extend(
        [
            Gate(
                "real-amplitude phase-sign conjugacy",
                relative_l2(negative_response, positive_response.conj()),
                args.value_rtol,
            ),
            Gate(
                "phase-sign MF-magnitude invariance",
                relative_l2(negative_magnitude, positive_magnitude),
                args.magnitude_rtol,
            ),
        ]
    )
    kvector = get_kvector(frequencies, cc)
    try:
        trace_and_match_magnitude(
            frequencies,
            kvector,
            tx,
            rx,
            points,
            amplitudes,
            points,
            freq_indices=torch.arange(0, frequencies.numel(), 2),
        )
    except ValueError as error:
        gates.append(
            Gate(
                "composed frequency subset rejected before mismatched adjoint",
                0.0 if "identical subset" in str(error) else math.inf,
                0.0,
            )
        )
    else:
        gates.append(
            Gate("composed frequency subset rejected before mismatched adjoint", math.inf, 0.0)
        )

    for gate in gates:
        print(
            f"[{'PASS' if gate.passed else 'FAIL'}] {gate.name}: "
            f"{gate.value:.6e} <= {gate.threshold:.6e}",
            flush=True,
        )
    passed = sum(gate.passed for gate in gates)
    print(f"GeRaF signal-operator gates: {passed}/{len(gates)} passed", flush=True)
    if passed != len(gates):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
