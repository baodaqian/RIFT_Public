#!/usr/bin/env python3
"""Run the isolated h-refined SpINR-style B787 quadrature candidate.

This adapter leaves the completed midpoint, Gauss2, and Gauss3 recipes and
their artifacts untouched.  It changes only the fixed integration rule and
artifact identity: a 96^3 parent-cell support with two-node tensor
Gauss--Legendre training (effective 192^3 points), compared at update 0 and
60 against an independent three-node rule (effective 288^3 points).

The candidate is a numerical-integration repair proposal, not a convergence
claim.  It preserves the signed-real 39-feature MLP, product range operator,
full-bin objective, optimizer/schedule, sealed roles, four-view effective
batch, and bounded tiled VJP from the bounded v1 driver.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch

from rift import spinr_smoke_runtime as _base
from rift.spinr_style import (
    SPINR_STYLE_GAUSS2_NODES_PER_CELL,
    SPINR_STYLE_GAUSS3_NODES_PER_CELL,
    gauss_legendre_cell_grid,
)


G96_PARENT_CELL_GRID_SIZE = 96
G96_TRAINING_NODES_PER_CELL = SPINR_STYLE_GAUSS2_NODES_PER_CELL
G96_REFERENCE_NODES_PER_CELL = SPINR_STYLE_GAUSS3_NODES_PER_CELL
G96_TRAINING_EFFECTIVE_GRID_SIZE = G96_PARENT_CELL_GRID_SIZE * G96_TRAINING_NODES_PER_CELL
G96_REFERENCE_EFFECTIVE_GRID_SIZE = G96_PARENT_CELL_GRID_SIZE * G96_REFERENCE_NODES_PER_CELL
G96_NUMERICAL_RECIPE_ID = "b78710k_spinr_style_inr_pm_g96_gauss2_v1"
G96_SMALLFIT_CHECKPOINT_FORMAT = "rift_spinr_style_b78716_smallfit_g96_gauss2_v1"
G96_SMALLFIT_RUN_NAME = "b78710k_spinr_style_inr_pm_g96_gauss2_smallfit16_v1"
G96_TRAINING_RULE = "tensor_gauss_legendre_2_nodes_per_axis_on_96_cubed_cells"
G96_REFERENCE_RULE = "tensor_gauss_legendre_3_nodes_per_axis_on_96_cubed_cells"
G96_READOUT_REPRESENTATION = "signed_real_sigma_g96_gauss2_cell_readout"
_ORIGINAL_SMALLFIT_RECIPE_IDENTITY = _base.smallfit_recipe_identity


def _g96_recipe_identity() -> dict[str, object]:
    recipe = _ORIGINAL_SMALLFIT_RECIPE_IDENTITY()
    operator = dict(recipe["operator"])
    operator.update(
        {
            "training_grid_size": G96_TRAINING_EFFECTIVE_GRID_SIZE,
            "reference_grid_size": G96_REFERENCE_EFFECTIVE_GRID_SIZE,
            "training_quadrature": G96_TRAINING_RULE,
            "reference_quadrature": G96_REFERENCE_RULE,
            "parent_cell_grid_size": G96_PARENT_CELL_GRID_SIZE,
            "physical_volume_weights": True,
            "training_rule_is_proposed_correction": True,
            "reference_is_independent_diagnostic_only": True,
        }
    )
    recipe.update(
        {
            "numerical_recipe_id": G96_NUMERICAL_RECIPE_ID,
            "smallfit_recipe_id": G96_SMALLFIT_RUN_NAME,
            "operator": operator,
            "quadrature_diagnostic": {
                "selected_training_views": 4,
                "comparison": "g96_gauss2_vs_gauss3_fixed_weights_signal_and_directional_gradient",
                "training_rule": G96_TRAINING_RULE,
                "reference_rule": G96_REFERENCE_RULE,
                "relative_limit": _base.QUADRATURE_RELATIVE_LIMIT,
                "near_zero_absolute_tolerance": _base.QUADRATURE_NEAR_ZERO_ABSOLUTE_TOLERANCE,
            },
        }
    )
    return recipe


def _g96_grid(grid_size: int, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
    """Map the two effective labels to the fixed 96^3-cell rules."""

    if int(grid_size) == G96_TRAINING_EFFECTIVE_GRID_SIZE:
        nodes_per_cell = G96_TRAINING_NODES_PER_CELL
    elif int(grid_size) == G96_REFERENCE_EFFECTIVE_GRID_SIZE:
        nodes_per_cell = G96_REFERENCE_NODES_PER_CELL
    else:
        raise ValueError(
            "G96 SpINR small-fit accepts only effective 192 (training) or "
            "288 (independent reference) grids"
        )
    return gauss_legendre_cell_grid(
        G96_PARENT_CELL_GRID_SIZE,
        nodes_per_cell=nodes_per_cell,
        support_m=float(kwargs.get("support_m", _base.SPINR_STYLE_SUPPORT_M)),
        device=kwargs.get("device"),
        dtype=kwargs.get("dtype", torch.float64),
    )


def _validate_g96_quadrature_record(
    record: object, *, expected_source_ids: Sequence[int]
) -> None:
    if not isinstance(record, Mapping):
        raise ValueError("bounded resume G96 quadrature evidence is malformed")
    required = {
        "frozen_weights",
        "source_ids",
        "training_grid_size",
        "reference_grid_size",
        "training_rule",
        "reference_rule",
        "training",
        "reference",
        "signal_gate",
        "directional_gradient_gate",
        "gate_pass",
    }
    if set(record) != required or record.get("frozen_weights") is not True:
        raise ValueError("bounded resume G96 quadrature record changed its frozen-weight protocol")
    if record.get("source_ids") != [int(item) for item in expected_source_ids]:
        raise ValueError("bounded resume G96 quadrature record changed its frozen first-four IDs")
    if (
        record.get("training_grid_size") != G96_TRAINING_EFFECTIVE_GRID_SIZE
        or record.get("reference_grid_size") != G96_REFERENCE_EFFECTIVE_GRID_SIZE
        or record.get("training_rule") != G96_TRAINING_RULE
        or record.get("reference_rule") != G96_REFERENCE_RULE
    ):
        raise ValueError("bounded resume G96 quadrature record changed its rule identity")

    observed_energies: list[float] = []
    directional_gradients: list[float] = []
    for key, expected_grid in (
        ("training", G96_TRAINING_EFFECTIVE_GRID_SIZE),
        ("reference", G96_REFERENCE_EFFECTIVE_GRID_SIZE),
    ):
        observation = record.get(key)
        if not isinstance(observation, Mapping) or set(observation) != {
            "grid_size",
            "observed_energy",
            "signal_elements",
            "native_spectral_objective",
            "directional_gradient",
        }:
            raise ValueError(f"bounded resume {key} observation is malformed")
        if observation.get("grid_size") != expected_grid:
            raise ValueError(f"bounded resume {key} changed its effective grid")
        observed_energies.append(
            _base._finite_nonnegative(
                observation.get("observed_energy"), f"bounded resume {key} observed energy"
            )
        )
        if observation.get("signal_elements") != _base.SMALLFIT_QUADRATURE_SIGNAL_ELEMENTS:
            raise ValueError(f"bounded resume {key} changed the full signal domain")
        _base._finite_nonnegative(
            observation.get("native_spectral_objective"), f"bounded resume {key} objective"
        )
        directional = float(observation.get("directional_gradient"))
        if not math.isfinite(directional):
            raise ValueError(f"bounded resume {key} directional gradient is invalid")
        directional_gradients.append(directional)

    if observed_energies[0] != observed_energies[1]:
        raise ValueError("bounded resume G96/Gauss2/Gauss3 records do not use identical observed data")
    signal_gate = record.get("signal_gate")
    gradient_gate = record.get("directional_gradient_gate")
    signal_pass = _base._validate_quadrature_decision(signal_gate, label="frequency_signal")
    gradient_pass = _base._validate_quadrature_decision(
        gradient_gate, label="directional_gradient"
    )
    assert isinstance(signal_gate, Mapping)
    assert isinstance(gradient_gate, Mapping)
    expected_signal_reference = math.sqrt(
        observed_energies[1] / float(_base.SMALLFIT_QUADRATURE_SIGNAL_ELEMENTS)
    )
    if signal_gate.get("reference_rms") != expected_signal_reference:
        raise ValueError("bounded resume G96 signal gate disagrees with observed energy")
    expected_difference = abs(directional_gradients[0] - directional_gradients[1])
    expected_reference = abs(directional_gradients[1])
    if (
        gradient_gate.get("difference_rms") != expected_difference
        or gradient_gate.get("reference_rms") != expected_reference
    ):
        raise ValueError("bounded resume G96 gradient gate disagrees with recorded gradients")
    if record.get("gate_pass") is not bool(signal_pass and gradient_pass):
        raise ValueError("bounded resume G96 gate does not combine its fixed decisions")


def _run_g96_quadrature_diagnostic(**kwargs: object) -> dict[str, object]:
    views = kwargs["views"]
    source_ids = tuple(views.role_ids("train")[: _base.CANONICAL_VIEW_BATCH])
    common = dict(kwargs)
    common["source_ids"] = source_ids
    training = _base._quadrature_grid_observation(
        **common, grid_size=G96_TRAINING_EFFECTIVE_GRID_SIZE
    )
    reference = _base._quadrature_grid_observation(
        **common, grid_size=G96_REFERENCE_EFFECTIVE_GRID_SIZE
    )
    training_signals = training.pop("signal_frequency")
    reference_signals = reference.pop("signal_frequency")
    if not isinstance(training_signals, list) or not isinstance(reference_signals, list):
        raise AssertionError("G96 quadrature signal snapshots were not retained")
    if len(training_signals) != len(reference_signals) or not training_signals:
        raise AssertionError("G96 quadrature signal snapshots disagree on view count")
    difference_energy = 0.0
    for training_signal, reference_signal in zip(training_signals, reference_signals):
        if not (torch.is_tensor(training_signal) and torch.is_tensor(reference_signal)):
            raise AssertionError("G96 quadrature signal snapshots must be tensors")
        if training_signal.shape != reference_signal.shape:
            raise AssertionError("G96 quadrature signal snapshots have incompatible shapes")
        difference_energy += float((training_signal - reference_signal).abs().square().sum().item())
    signal_elements = int(reference["signal_elements"])
    if (
        signal_elements != _base.SMALLFIT_QUADRATURE_SIGNAL_ELEMENTS
        or training.get("signal_elements") != signal_elements
    ):
        raise AssertionError("G96 quadrature changed its full four-view signal domain")
    observed_energy = float(reference["observed_energy"])
    if not math.isclose(observed_energy, float(training["observed_energy"]), rel_tol=0.0, abs_tol=0.0):
        raise AssertionError("G96 training/reference rules must use the same fixed observed data")
    signal_gate = _base.quadrature_difference_decision(
        difference_rms=math.sqrt(difference_energy / signal_elements),
        reference_rms=math.sqrt(observed_energy / signal_elements),
        label="frequency_signal",
    )
    gradient_gate = _base.quadrature_difference_decision(
        difference_rms=abs(float(training["directional_gradient"]) - float(reference["directional_gradient"])),
        reference_rms=abs(float(reference["directional_gradient"])),
        label="directional_gradient",
    )
    return {
        "frozen_weights": True,
        "source_ids": [int(item) for item in source_ids],
        "training_grid_size": G96_TRAINING_EFFECTIVE_GRID_SIZE,
        "reference_grid_size": G96_REFERENCE_EFFECTIVE_GRID_SIZE,
        "training_rule": G96_TRAINING_RULE,
        "reference_rule": G96_REFERENCE_RULE,
        "training": training,
        "reference": reference,
        "signal_gate": signal_gate,
        "directional_gradient_gate": gradient_gate,
        "gate_pass": bool(signal_gate["pass"] and gradient_gate["pass"]),
    }


@torch.no_grad()
def _g96_field_readout(
    *,
    model: _base.SpinrStyleINR,
    points_m: torch.Tensor,
    cell_volume_m3: float | torch.Tensor,
    initial_output_scale: float,
) -> dict[str, object]:
    network_field = _base.evaluate_neural_field_tiled(
        model, points_m, neural_point_tile=_base.SMALLFIT_NEURAL_POINT_TILE
    )
    sigma = network_field * float(initial_output_scale)
    if not torch.isfinite(sigma).all():
        raise FloatingPointError("G96 field readout is non-finite")
    volume = torch.as_tensor(cell_volume_m3, device=sigma.device, dtype=torch.float64)
    if volume.shape != sigma.shape:
        raise ValueError("G96 field readout requires one physical volume per point")
    return {
        "representation": G96_READOUT_REPRESENTATION,
        "integration_rule": G96_TRAINING_RULE,
        "grid_size": G96_TRAINING_EFFECTIVE_GRID_SIZE,
        "parent_cell_grid_size": G96_PARENT_CELL_GRID_SIZE,
        "field_points": int(points_m.shape[0]),
        "support_min_m": [-_base.SPINR_STYLE_SUPPORT_M] * 3,
        "support_max_m": [_base.SPINR_STYLE_SUPPORT_M] * 3,
        "sigma_abs_mean": float(sigma.abs().mean().item()),
        "sigma_abs_max": float(sigma.abs().max().item()),
        "sigma_squared_mean": float(sigma.square().mean().item()),
        "sigma_squared_integral_m3": float((sigma.square() * volume).sum().item()),
        "interpretation": "native field-amplitude/energy readout; not a reconstruction-quality claim",
    }


def _validate_g96_terminal_geometry_readout(readout: object) -> None:
    if not isinstance(readout, Mapping):
        raise ValueError("G96 terminal report lacks geometry readout")
    if readout.get("representation") != G96_READOUT_REPRESENTATION:
        raise ValueError("G96 terminal report changed its field readout representation")
    if (
        readout.get("integration_rule") != G96_TRAINING_RULE
        or readout.get("grid_size") != G96_TRAINING_EFFECTIVE_GRID_SIZE
        or readout.get("parent_cell_grid_size") != G96_PARENT_CELL_GRID_SIZE
        or readout.get("field_points") != G96_TRAINING_EFFECTIVE_GRID_SIZE**3
    ):
        raise ValueError("G96 terminal report changed its integration grid")
    if (
        readout.get("support_min_m") != [-_base.SPINR_STYLE_SUPPORT_M] * 3
        or readout.get("support_max_m") != [_base.SPINR_STYLE_SUPPORT_M] * 3
    ):
        raise ValueError("G96 terminal report changed its support")
    if readout.get("interpretation") != "native field-amplitude/energy readout; not a reconstruction-quality claim":
        raise ValueError("G96 terminal report changed its readout interpretation")
    for key in ("sigma_abs_mean", "sigma_abs_max", "sigma_squared_mean", "sigma_squared_integral_m3"):
        _base._finite_nonnegative(readout.get(key), f"G96 terminal report {key}")


def _g96_production_clearance_reasons(
    *, initial_rel_mse: float, final_rel_mse: float, finite_nonzero_updates: bool, quadrature_pass: bool
) -> list[str]:
    reasons = [
        "bounded_16_train_16_validation_60_update_engineering_smoke_is_not_a_production_run",
        "no_full_3200_view_convergence_or_sealed_test_evidence",
        "g96_gauss2_training_rule_is_an_opt_in_numerical_repair_candidate",
    ]
    if not final_rel_mse < initial_rel_mse:
        reasons.append("fixed_train_coherent_relative_mse_did_not_fall")
    if not finite_nonzero_updates:
        reasons.append("one_or_more_logical_updates_lacked_finite_nonzero_gradient_or_parameter_change")
    if not quadrature_pass:
        reasons.append("g96_gauss2_gauss3_fixed_weight_quadrature_gate_did_not_pass")
    return reasons


def _configure() -> None:
    """Install only this process's isolated numerical-recipe adapter."""

    _base.SMALLFIT_CHECKPOINT_FORMAT = G96_SMALLFIT_CHECKPOINT_FORMAT
    _base.SMALLFIT_RUN_NAME = G96_SMALLFIT_RUN_NAME
    _base.SMALLFIT_TRAIN_GRID_SIZE = G96_TRAINING_EFFECTIVE_GRID_SIZE
    _base.SMALLFIT_REFERENCE_GRID_SIZE = G96_REFERENCE_EFFECTIVE_GRID_SIZE
    _base.midpoint_grid = _g96_grid
    _base.smallfit_recipe_identity = _g96_recipe_identity
    _base._validate_quadrature_record = _validate_g96_quadrature_record
    _base.run_quadrature_diagnostic = _run_g96_quadrature_diagnostic
    _base._field_readout = _g96_field_readout
    _base._validate_terminal_geometry_readout = _validate_g96_terminal_geometry_readout
    _base._production_clearance_reasons = _g96_production_clearance_reasons


def main(argv: Sequence[str] | None = None) -> int:
    _configure()
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
