#!/usr/bin/env python3
"""Data-free operator gate for the frozen B7873200 SE Stage-1 physics.

This is not a claim that a synthetic fixture measures the archive's phase
convention.  The B787 FMCW convention is fixed by the established shared
acquisition specification.  This gate verifies that the exact configured
``phase_sign=-1`` / ``range_model=product`` route is actually wired through
the range forward and adjoint operators, and that it is not silently replaced
by RIFT's historical ``sum2`` default.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch

import train
from rift.config import cc
from rift.forward_operator import forward_operator_lessparallel
from rift.range_operator import range_adjoint_operator, range_forward_operator
from rift.sugavanam_ertin_b7873200_stage1 import _validate_generic_rng_state


def relative_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).norm() / right.norm().clamp_min(1.0e-30))


def main() -> None:
    torch.manual_seed(20260905)
    device = torch.device("cpu")
    frequencies = torch.linspace(8.5e9, 11.5e9, 32, dtype=torch.float64, device=device)
    kvector = 2.0 * torch.pi * frequencies / cc
    # Two exactly monostatic pairs make the product/sum2 amplitude ratio an
    # unambiguous factor four while retaining a nontrivial multi-pair fixture.
    array = torch.tensor([[0.0, 0.0, 10.0], [0.02, -0.01, 10.0]], dtype=torch.float64)
    positions = torch.tensor(
        [[0.02, -0.01, 0.01], [-0.03, 0.02, -0.02], [0.01, 0.04, 0.0]],
        dtype=torch.float64,
    )
    weights = torch.tensor([1.0 + 0.5j, -0.3 + 0.2j, 0.1 - 0.4j], dtype=torch.complex128)
    common = {
        "phase_sign": -1.0,
        "range_model": "product",
        "compute_dtype": torch.float64,
        "point_chunk": 3,
        "pair_chunk": 2,
    }
    rendered = range_forward_operator(
        frequencies, kvector, array, array, positions, weights, **common
    )
    reference = forward_operator_lessparallel(
        frequencies,
        kvector,
        array,
        array,
        positions,
        weights,
        artificial_gain=1.0,
        p_spectrum=None,
        omega_scaling="unity",
        phase_sign=-1.0,
        range_model="product",
    ).to(torch.complex128)
    forward_error = relative_error(rendered, reference)
    if forward_error > 3.0e-6:
        raise AssertionError(f"product/-1 range forward parity failed: {forward_error:.3e}")

    legacy_sum2 = range_forward_operator(
        frequencies,
        kvector,
        array,
        array,
        positions,
        weights,
        phase_sign=-1.0,
        range_model="sum2",
        compute_dtype=torch.float64,
        point_chunk=3,
        pair_chunk=2,
    )
    product_ratio = float((rendered.norm() / legacy_sum2.norm()).item())
    if not math.isclose(product_ratio, 4.0, rel_tol=2.0e-6, abs_tol=2.0e-6):
        raise AssertionError(f"product versus sum2 scaling is not four: {product_ratio:.9g}")

    opposite_sign = range_forward_operator(
        frequencies,
        kvector,
        array,
        array,
        positions,
        weights,
        phase_sign=1.0,
        range_model="product",
        compute_dtype=torch.float64,
        point_chunk=3,
        pair_chunk=2,
    )
    sign_distance = relative_error(rendered, opposite_sign)
    if sign_distance < 1.0e-3:
        raise AssertionError("the -1 and +1 phase routes were indistinguishable on the fixture")

    residual = torch.randn_like(rendered.real) + 1j * torch.randn_like(rendered.real)
    residual = residual.to(torch.complex128)
    adjoint = range_adjoint_operator(
        frequencies,
        kvector,
        array,
        array,
        positions,
        residual,
        **common,
    )
    lhs = (rendered.conj() * residual).sum()
    rhs = (weights.conj() * adjoint).sum()
    adjoint_error = float((lhs - rhs).abs() / (rendered.norm() * residual.norm()).clamp_min(1.0e-30))
    if adjoint_error > 3.0e-8:
        raise AssertionError(f"product/-1 adjoint identity failed: {adjoint_error:.3e}")

    # The sealed Stage-1 wrapper requires this exact generic recovery payload
    # before it permits --resume.  Exercise the real capture/restore route on
    # the allocated Torch runtime, rather than trusting the static shape gate.
    train.set_seed(20260905)
    recovery = train.capture_rng_state()
    _validate_generic_rng_state(recovery)
    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(5)
    expected_frequency = torch.rand(5, generator=train._FREQ_RNG)
    restored = train.restore_rng_state(recovery, require_complete=True)
    if restored is not True:
        raise AssertionError("exact Stage-1 RNG payload did not restore")
    if random.random() != expected_python or float(np.random.random()) != expected_numpy:
        raise AssertionError("Stage-1 Python/NumPy RNG continuation changed")
    if not torch.equal(torch.rand(5), expected_torch):
        raise AssertionError("Stage-1 Torch RNG continuation changed")
    if not torch.equal(torch.rand(5, generator=train._FREQ_RNG), expected_frequency):
        raise AssertionError("Stage-1 frequency RNG continuation changed")
    print(
        "SE_B7873200_STAGE1_OPERATOR_PASS: "
        f"forward={forward_error:.3e} ratio={product_ratio:.7g} "
        f"phase_distance={sign_distance:.3e} adjoint={adjoint_error:.3e} rng=exact",
        flush=True,
    )


if __name__ == "__main__":
    main()
