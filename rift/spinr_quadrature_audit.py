"""Frozen-field integration checks using full neural-parameter gradients.

This module performs no optimization. Higher-order quadrature is a comparison
rule, not exact truth; both order and spatial refinement must be checked.
"""
from __future__ import annotations

import math
import time

import torch

from rift.config import cc
from rift.forward_operator import get_kvector
from rift.range_operator import range_forward_operator
from rift.spinr_fidelity import PAPER_RECIPE, DIRECT_RECIPE, scene_range_bin_mask
from rift.spinr_style import (gauss_legendre_cell_grid, midpoint_grid,
                              scale_field_to_renderer_weights)


def quadrature_plan(recipe: dict, reference_grid: int | None = None) -> list[dict]:
    """Keep the saved rule immutable and build two declared diagnostic rules."""
    operator = recipe["operator"]
    base = int(operator["grid_size"])
    corrected = recipe["recipe_id"] in (PAPER_RECIPE, DIRECT_RECIPE)
    reference = math.ceil(base*4/3) if reference_grid is None else reference_grid
    if type(reference) is not int or reference <= base:
        raise ValueError("reference grid must be an integer larger than the saved parent grid")
    rules = [
        {"label": "saved_training_rule", "kind": "gauss_legendre" if corrected else "midpoint",
         "parent_grid": base, "nodes_per_cell": int(operator["nodes_per_cell"]) if corrected else 1},
        {"label": "higher_order", "kind": "gauss_legendre", "parent_grid": base, "nodes_per_cell": 3},
        {"label": "spatially_refined", "kind": "gauss_legendre", "parent_grid": reference, "nodes_per_cell": 3},
    ]
    for rule in rules:
        rule["integration_points"] = (rule["parent_grid"]*rule["nodes_per_cell"])**3
    return rules


def observe_rule(*, model, observations, frequencies_hz, rule, support_m,
                 initial_output_scale, training_mean_raw_power, scene_bins,
                 device="cpu", neural_point_tile=4096, renderer_point_tile=65536,
                 pair_tile=16, direct_bins=False) -> dict:
    """Render and differentiate one rule with unchanged parameters and RNG.

    ``observations`` is a small sequence of (source_id, response[F,Rx,Tx],
    rx_positions, tx_positions). Gradients use the actual trainer's tiled VJP
    and average the same per-view objective. Existing model mode and .grad
    references are restored, including on exceptions.
    """
    from train_spinr_style import (evaluate_neural_field_tiled, response_cotangent,
                                   real_field_cotangent_from_response, replay_field_cotangent_tiled,
                                   direct_bin_loss_and_field_cotangent)
    if not observations or min(neural_point_tile, renderer_point_tile, pair_tile) < 1:
        raise ValueError("quadrature audit requires observations and positive tile sizes")
    if rule["kind"] not in ("midpoint", "gauss_legendre"):
        raise ValueError("unknown integration rule")
    ids = [int(item[0]) for item in observations]
    if len(set(ids)) != len(ids):
        raise ValueError("diagnostic source IDs must be distinct")
    device = torch.device(device)
    parameters = list(model.named_parameters())
    if not parameters or any(not p.requires_grad for _, p in parameters):
        raise ValueError("all diagnostic model parameters must permit differentiation")
    previous_gradients = [p.grad for _, p in parameters]
    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    started = time.monotonic()
    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        if rule["kind"] == "midpoint":
            points, volumes = midpoint_grid(rule["parent_grid"], support_m=support_m, device=device)
        else:
            points, volumes = gauss_legendre_cell_grid(
                rule["parent_grid"], nodes_per_cell=rule["nodes_per_cell"], support_m=support_m,
                device=device, dtype=torch.float64)
        frequencies = torch.as_tensor(frequencies_hz, device=device, dtype=torch.float64)
        kvector = get_kvector(frequencies, cc)
        field = evaluate_neural_field_tiled(model, points, neural_point_tile=neural_point_tile)
        weights = scale_field_to_renderer_weights(field, cell_volume_m3=volumes,
                                                  initial_output_scale=initial_output_scale)
        field_gradient = torch.zeros_like(field)
        signals, observed_energies, losses = [], [], []
        for _, observed, rx, tx in observations:
            target = torch.as_tensor(observed, dtype=torch.complex128, device=device)
            rx = torch.as_tensor(rx, dtype=torch.float64, device=device)
            tx = torch.as_tensor(tx, dtype=torch.float64, device=device)
            if (target.shape != (len(frequencies), len(rx), len(tx)) or not torch.isfinite(target).all()):
                raise ValueError("invalid diagnostic response shape or values")
            with torch.no_grad():
                predicted = range_forward_operator(
                    frequencies, kvector, rx, tx, points, weights, phase_sign=-1.,
                    range_model="product", compute_dtype=torch.float64,
                    point_chunk=renderer_point_tile, pair_chunk=pair_tile)
            mask = scene_range_bin_mask(frequencies, rx, tx, support_m=support_m) if scene_bins else None
            if direct_bins:
                if not scene_bins:
                    raise ValueError("direct-bin diagnostics require scene selection")
                loss, gradient = direct_bin_loss_and_field_cotangent(
                    field=field, observed=target, frequencies_hz=frequencies,
                    rx_pos_m=rx, tx_pos_m=tx, points_m=points, cell_volume_m3=volumes,
                    initial_output_scale=initial_output_scale, training_mean_raw_power=training_mean_raw_power,
                    renderer_point_tile=renderer_point_tile, pair_tile=pair_tile, support_m=support_m)
                field_gradient.add_(gradient/len(observations))
                signals.append(predicted.detach().cpu())
                observed_energies.append(float(target.abs().square().sum()))
                losses.append(float(loss))
                continue
            with torch.enable_grad():
                loss, response_gradient = response_cotangent(
                    predicted, target, training_mean_raw_power=training_mean_raw_power, range_bin_mask=mask)
            field_gradient.add_(real_field_cotangent_from_response(
                response_cotangent_frequency=response_gradient/len(observations),
                frequencies_hz=frequencies, kvector=kvector, rx_pos_m=rx, tx_pos_m=tx,
                points_m=points, cell_volume_m3=volumes, initial_output_scale=initial_output_scale,
                renderer_point_tile=renderer_point_tile, pair_tile=pair_tile))
            signals.append(predicted.detach().cpu())
            observed_energies.append(float(target.abs().square().sum()))
            losses.append(float(loss))
        with torch.enable_grad():
            replay_field_cotangent_tiled(model, points, field_gradient, neural_point_tile=neural_point_tile)
        gradients = {}
        for name, parameter in parameters:
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"non-finite or absent diagnostic parameter gradient: {name}")
            gradients[name] = parameter.grad.detach().to(device="cpu", dtype=torch.float64).clone()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return {"rule": dict(rule), "source_ids": ids, "signals": signals,
                "observed_energies": observed_energies, "parameter_gradients": gradients,
                "native_spectral_objective": sum(losses)/len(losses),
                "per_view_objectives": losses, "elapsed_seconds": time.monotonic()-started,
                "peak_cuda_allocated_bytes": (torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)}
    finally:
        for (_, parameter), previous in zip(parameters, previous_gradients):
            parameter.grad = previous
        model.train(was_training)


def difference_check(difference_rms, reference_rms, *, relative_tolerance, absolute_tolerance):
    """An explicit absolute rule near zero, otherwise a relative RMS rule."""
    values = (difference_rms, reference_rms, relative_tolerance, absolute_tolerance)
    if not all(math.isfinite(v) and v >= 0 for v in values) or min(values[2:]) <= 0:
        raise ValueError("comparison norms must be finite/nonnegative and tolerances positive")
    near_zero = reference_rms <= absolute_tolerance
    relative = None if near_zero else difference_rms/reference_rms
    return {"difference_rms": difference_rms, "reference_rms": reference_rms,
            "relative_tolerance": relative_tolerance, "absolute_tolerance": absolute_tolerance,
            "mode": "near_zero_absolute" if near_zero else "relative",
            "relative_difference": relative,
            "pass": difference_rms <= absolute_tolerance if near_zero else relative <= relative_tolerance}


def compare_observations(candidate, reference, *, relative_tolerance=.01,
                         signal_absolute_tolerance=1e-12, gradient_absolute_tolerance=1e-10) -> dict:
    """Compare full responses and all parameter-gradient tensors, per view/layer."""
    if (candidate["source_ids"] != reference["source_ids"]
            or candidate["observed_energies"] != reference["observed_energies"]
            or not candidate["source_ids"]
            or len(reference["observed_energies"]) != len(reference["source_ids"])
            or len(candidate["signals"]) != len(candidate["source_ids"])
            or len(reference["signals"]) != len(reference["source_ids"])):
        raise ValueError("quadrature comparisons must use identical nonempty observations")
    signal_checks = []
    signal_difference, signal_elements = 0., 0
    for source_id, a, b, observed_energy in zip(candidate["source_ids"], candidate["signals"],
                                               reference["signals"], reference["observed_energies"]):
        if a.shape != b.shape or a.numel() == 0:
            raise ValueError("quadrature signal layouts disagree")
        error = float((a-b).abs().square().sum())
        signal_checks.append({"source_id": source_id, **difference_check(
            math.sqrt(error/a.numel()), math.sqrt(observed_energy/a.numel()),
            relative_tolerance=relative_tolerance, absolute_tolerance=signal_absolute_tolerance)})
        signal_difference += error
        signal_elements += a.numel()
    signal = difference_check(math.sqrt(signal_difference/signal_elements),
                              math.sqrt(sum(reference["observed_energies"])/signal_elements),
                              relative_tolerance=relative_tolerance, absolute_tolerance=signal_absolute_tolerance)
    left, right = candidate["parameter_gradients"], reference["parameter_gradients"]
    if set(left) != set(right) or not left:
        raise ValueError("quadrature parameter-gradient layouts disagree")
    gradient_checks, gradient_difference, gradient_energy, parameter_count = {}, 0., 0., 0
    for name in right:
        a, b = left[name], right[name]
        if a.shape != b.shape or a.numel() == 0:
            raise ValueError("quadrature parameter-gradient shapes disagree")
        error, energy = float((a-b).square().sum()), float(b.square().sum())
        gradient_checks[name] = difference_check(math.sqrt(error/a.numel()), math.sqrt(energy/a.numel()),
                                                 relative_tolerance=relative_tolerance,
                                                 absolute_tolerance=gradient_absolute_tolerance)
        gradient_difference += error
        gradient_energy += energy
        parameter_count += a.numel()
    gradient = difference_check(math.sqrt(gradient_difference/parameter_count), math.sqrt(gradient_energy/parameter_count),
                                 relative_tolerance=relative_tolerance, absolute_tolerance=gradient_absolute_tolerance)
    return {"candidate": candidate["rule"], "reference": reference["rule"],
            "signal": signal, "per_view_signal": signal_checks,
            "parameter_gradient": gradient, "per_parameter_tensor": gradient_checks,
            "checks_pass": bool(signal["pass"] and gradient["pass"]
                                and all(c["pass"] for c in signal_checks)
                                and all(c["pass"] for c in gradient_checks.values()))}
