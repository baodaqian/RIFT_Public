#!/usr/bin/env python3
"""Torch runtime preflight for the isolated G96/Gauss2-to-Gauss3 candidate.

The adapter mapping is exercised with a spy to avoid materializing the full
192^3/288^3 B787 rules locally.  Renderer/autograd/manual-VJP parity is then
checked on tiny physical cell rules.  No B787 data is opened and no training
is performed.
"""

from __future__ import annotations

import copy
import io
import math
from collections.abc import Callable, Mapping

import torch

from rift.config import cc
from rift.forward_operator import get_kvector
from rift.range_operator import range_forward_operator
from rift.spinr_style import (
    SPINR_STYLE_GAUSS2_NODES_PER_CELL,
    SPINR_STYLE_GAUSS3_NODES_PER_CELL,
    SPINR_STYLE_PHASE_SIGN,
    SPINR_STYLE_RANGE_MODEL,
    gauss_legendre_cell_grid,
    scale_field_to_renderer_weights,
    spinr_style_objective,
)
from train_spinr_style import real_field_cotangent_from_response, response_cotangent
import train_spinr_style_smoke as g96


CHECKS = 0


def check(condition: bool, message: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}", flush=True)


def expect_rejection(callback: Callable[[], object], message: str) -> None:
    try:
        callback()
    except (AssertionError, TypeError, ValueError):
        check(True, message)
        return
    raise AssertionError(message + " (invalid identity was accepted)")


def _finite_relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = (actual - expected).abs().max().item()
    scale = max(expected.abs().max().item(), 1.0e-30)
    result = float(difference / scale)
    if not math.isfinite(result):
        raise AssertionError("runtime preflight produced a non-finite relative error")
    return result


def _identity_preflight() -> None:
    base = g96._base
    legacy_format = base.SMALLFIT_CHECKPOINT_FORMAT
    legacy_run_name = base.SMALLFIT_RUN_NAME
    legacy_recipe = base.smallfit_recipe_identity()

    g96._configure()
    recipe = g96._g96_recipe_identity()
    check(base.midpoint_grid is g96._g96_grid,
          "candidate configuration installs the G96 cell adapter")
    check(base.SMALLFIT_TRAIN_GRID_SIZE == 192 and base.SMALLFIT_REFERENCE_GRID_SIZE == 288,
          "configured identity is effective 192^3 training and 288^3 reference")
    check(recipe.get("numerical_recipe_id") == g96.G96_NUMERICAL_RECIPE_ID,
          "recipe records a distinct G96 numerical identity")
    operator = recipe.get("operator")
    check(
        isinstance(operator, Mapping)
        and operator.get("parent_cell_grid_size") == 96
        and operator.get("physical_volume_weights") is True
        and operator.get("reference_is_independent_diagnostic_only") is True,
        "recipe records G=96 physical volumes and diagnostic-only reference semantics",
    )

    original_grid = g96.gauss_legendre_cell_grid
    calls: list[tuple[int, int]] = []

    def spy_grid(parent: int, *, nodes_per_cell: int, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append((int(parent), int(nodes_per_cell)))
        return torch.zeros((8, 3), dtype=kwargs.get("dtype", torch.float64)), torch.ones(8, dtype=kwargs.get("dtype", torch.float64))

    g96.gauss_legendre_cell_grid = spy_grid
    try:
        train_points, train_volume = g96._g96_grid(192, device="cpu", dtype=torch.float64)
        ref_points, ref_volume = g96._g96_grid(288, device="cpu", dtype=torch.float64)
    finally:
        g96.gauss_legendre_cell_grid = original_grid
    check(calls == [(96, 2), (96, 3)],
          "adapter maps exactly G=96/Gauss2 training and G=96/Gauss3 reference")
    check(train_points.shape == (8, 3) and train_volume.shape == (8,)
          and ref_points.shape == (8, 3) and ref_volume.shape == (8,),
          "adapter preserves point/volume vector shapes")
    expect_rejection(lambda: g96._g96_grid(48), "historical midpoint grid is rejected by the adapter")

    header = {
        "format": g96.G96_SMALLFIT_CHECKPOINT_FORMAT,
        "run_name": g96.G96_SMALLFIT_RUN_NAME,
        "recipe": recipe,
        "execution": {
            "completed_updates": 0,
            "phase": "updates",
            "resume_count": 0,
            "production_clock": base._production_clock(0),
        },
    }
    base._validate_smallfit_checkpoint_header(header, recipe=recipe)
    serialized = io.BytesIO()
    torch.save(header, serialized)
    serialized.seek(0)
    roundtrip = torch.load(serialized, map_location="cpu", weights_only=True)
    base._validate_smallfit_checkpoint_header(roundtrip, recipe=recipe)
    check(roundtrip == header, "candidate checkpoint header survives identity-preserving serialization")
    old_format = copy.deepcopy(header)
    old_format["format"] = legacy_format
    expect_rejection(lambda: base._validate_smallfit_checkpoint_header(old_format, recipe=recipe),
                     "old midpoint checkpoint format is rejected")
    old_run_name = copy.deepcopy(header)
    old_run_name["run_name"] = legacy_run_name
    expect_rejection(lambda: base._validate_smallfit_checkpoint_header(old_run_name, recipe=recipe),
                     "old midpoint output identity is rejected")
    old_recipe = copy.deepcopy(header)
    old_recipe["recipe"] = legacy_recipe
    expect_rejection(lambda: base._validate_smallfit_checkpoint_header(old_recipe, recipe=recipe),
                     "old midpoint recipe metadata is rejected")

    observed_energy = float(base.SMALLFIT_QUADRATURE_SIGNAL_ELEMENTS) * 2.0
    gate = base.quadrature_difference_decision(difference_rms=0.0, reference_rms=math.sqrt(2.0), label="frequency_signal")
    gradient_gate = base.quadrature_difference_decision(difference_rms=0.0, reference_rms=1.25, label="directional_gradient")
    record = {
        "frozen_weights": True,
        "source_ids": [101, 102, 103, 104],
        "training_grid_size": g96.G96_TRAINING_EFFECTIVE_GRID_SIZE,
        "reference_grid_size": g96.G96_REFERENCE_EFFECTIVE_GRID_SIZE,
        "training_rule": g96.G96_TRAINING_RULE,
        "reference_rule": g96.G96_REFERENCE_RULE,
        "training": {"grid_size": g96.G96_TRAINING_EFFECTIVE_GRID_SIZE, "observed_energy": observed_energy, "signal_elements": base.SMALLFIT_QUADRATURE_SIGNAL_ELEMENTS, "native_spectral_objective": 2.0, "directional_gradient": 1.25},
        "reference": {"grid_size": g96.G96_REFERENCE_EFFECTIVE_GRID_SIZE, "observed_energy": observed_energy, "signal_elements": base.SMALLFIT_QUADRATURE_SIGNAL_ELEMENTS, "native_spectral_objective": 2.0, "directional_gradient": 1.25},
        "signal_gate": gate,
        "directional_gradient_gate": gradient_gate,
        "gate_pass": True,
    }
    g96._validate_g96_quadrature_record(record, expected_source_ids=[101, 102, 103, 104])
    check(True, "G96/Gauss2-to-Gauss3 diagnostic record validates with fixed four-view gates")
    bad_record = copy.deepcopy(record)
    bad_record["reference_grid_size"] = 96
    expect_rejection(lambda: g96._validate_g96_quadrature_record(bad_record, expected_source_ids=[101, 102, 103, 104]),
                     "diagnostic record with the old reference grid is rejected")


def _direct_operator_vjp_preflight(device: torch.device) -> None:
    frequencies = torch.linspace(9.0e9, 9.7e9, 8, dtype=torch.float64, device=device)
    kvector = get_kvector(frequencies, cc).to(device=device, dtype=torch.float64)
    rx_pos = torch.tensor([[0.00, 0.00, 0.40], [0.00, 0.025, 0.40]], dtype=torch.float64, device=device)
    tx_pos = torch.tensor([[0.025, 0.00, 0.40], [-0.025, 0.00, 0.40]], dtype=torch.float64, device=device)
    common = {
        "freqs_full": frequencies,
        "kvector_full": kvector,
        "arr_pos_rx": rx_pos,
        "arr_pos_tx": tx_pos,
        "phase_sign": SPINR_STYLE_PHASE_SIGN,
        "pair_chunk": 2,
        "point_chunk": 64,
        "compute_dtype": torch.float64,
        "range_model": SPINR_STYLE_RANGE_MODEL,
    }
    for nodes_per_cell, label in (
        (SPINR_STYLE_GAUSS2_NODES_PER_CELL, "G96/Gauss2 training"),
        (SPINR_STYLE_GAUSS3_NODES_PER_CELL, "G96/Gauss3 reference"),
    ):
        points, volume = gauss_legendre_cell_grid(2, nodes_per_cell=nodes_per_cell, support_m=0.15, device=device, dtype=torch.float64)
        check(torch.isfinite(volume).all() and bool((volume > 0.0).all()),
              f"{label} runtime rule returns finite positive physical volumes")
        if nodes_per_cell == SPINR_STYLE_GAUSS3_NODES_PER_CELL:
            check(int(torch.unique(volume).numel()) > 1, "Gauss3 reference weights remain nonuniform")
        field_values = torch.linspace(-0.2, 0.3, points.shape[0], dtype=torch.float64, device=device)
        field = field_values.detach().clone().requires_grad_(True)
        shared_weights = scale_field_to_renderer_weights(field, cell_volume_m3=volume, initial_output_scale=0.37)
        shared_response = range_forward_operator(**common, scatterer_pos=points, scatterer_weights=shared_weights)
        index = torch.arange(shared_response.numel(), dtype=torch.float64, device=device).reshape(shared_response.shape)
        observed = 0.05 * torch.polar(torch.ones_like(index), 0.13 * index)
        shared_loss = spinr_style_objective(shared_response, observed, training_mean_raw_power=0.7)
        shared_autograd = torch.autograd.grad(shared_loss, field)[0].detach()
        _, response_gradient = response_cotangent(shared_response.detach(), observed, training_mean_raw_power=0.7)
        manual_vjp = real_field_cotangent_from_response(
            response_cotangent_frequency=response_gradient,
            frequencies_hz=frequencies,
            kvector=kvector,
            rx_pos_m=rx_pos,
            tx_pos_m=tx_pos,
            points_m=points,
            cell_volume_m3=volume,
            initial_output_scale=0.37,
            renderer_point_tile=64,
            pair_tile=2,
        )
        direct_field = field_values.detach().clone().requires_grad_(True)
        direct_weights = direct_field * volume * ((4.0 * math.pi) ** 2) * 0.37
        direct_response = range_forward_operator(**common, scatterer_pos=points, scatterer_weights=direct_weights)
        direct_loss = spinr_style_objective(direct_response, observed, training_mean_raw_power=0.7)
        direct_autograd = torch.autograd.grad(direct_loss, direct_field)[0].detach()
        check(_finite_relative_error(shared_response, direct_response) <= 1.0e-12,
              f"{label} vector renderer weights match direct construction")
        check(_finite_relative_error(shared_autograd, direct_autograd) <= 1.0e-10,
              f"{label} shared renderer autograd matches direct autograd")
        check(_finite_relative_error(manual_vjp, direct_autograd) <= 1.0e-8,
              f"{label} manual real-field VJP matches direct autograd")


def main() -> int:
    _identity_preflight()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"SPINR_STYLE_B78716_G96_GAUSS2_RUNTIME_DEVICE={device}", flush=True)
    _direct_operator_vjp_preflight(device)
    print(f"SpINR-style G96/Gauss2-to-Gauss3 runtime preflight passed: {CHECKS} checks.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
