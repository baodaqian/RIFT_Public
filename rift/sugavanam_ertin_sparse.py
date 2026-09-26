"""Streaming complex basis-pursuit denoising for SE Eq. 4.

The paper does not name its solver. We root-find the L1-ball Pareto curve,
solving each least-squares subproblem by projected gradient/backtracking.
This uses only voxel-sized recovery state, not a full radar-sized dual vector.
It is an independent solver, not the authors' code or the SPGL1 package.
"""
from __future__ import annotations

import math
import torch

from .sugavanam_ertin_paper import complex_soft_threshold


def project_complex_l1_ball(x, radius):
    """Euclidean projection onto sum(abs(x)) <= radius, preserving phase."""
    if radius < 0 or not math.isfinite(radius):
        raise ValueError("Invalid complex L1 radius")
    if radius == 0:
        return torch.zeros_like(x)
    if float(x.abs().sum()) <= radius:
        return x.clone()
    values = x.abs().flatten().sort(descending=True).values
    index = torch.arange(1, len(values)+1, device=x.device, dtype=values.dtype)
    thresholds = (values.cumsum(0)-radius)/index
    active = values > thresholds
    threshold = thresholds[torch.nonzero(active)[-1, 0]]
    # sum() and sorted cumsum() can round to opposite sides of the radius
    # for a vector on the ball boundary. The projection multiplier is
    # nonnegative; clamp only this roundoff artifact, not the solver tolerances.
    return complex_soft_threshold(x, float(threshold.clamp_min(0)))


def initial_state(target_loss, lipschitz=1.):
    if not math.isfinite(target_loss) or target_loss <= 0 or lipschitz <= 0:
        raise ValueError("A finite positive residual energy budget is required")
    return dict(target_loss=float(target_loss), radius=0., lipschitz=float(lipschitz),
                lower_radius=0., upper_radius=None, root_updates=0,
                converged=False, stalled=False)


def constrained_step(x, value_and_grad, value, state, *, residual_rtol=1e-3,
                     optimality_rtol=1e-5, max_backtracks=60):
    """One projected step and, at subproblem convergence, a Pareto root update.

    Callbacks return f=0.5*mean(|Ax-y|²) and its complex Euclidean gradient.
    ``target_loss`` is sigma²/(2*N*response_rms²), never a fixed LASSO penalty.
    A Frank-Wolfe gap bounds subproblem suboptimality. Both that gap and the
    residual constraint must pass before a field can supervise Stage 2.
    """
    s = dict(state)
    tau, target, L = s["radius"], s["target_loss"], s["lipschitz"]
    x = project_complex_l1_ball(x, tau)
    f, g = value_and_grad(x)
    if not math.isfinite(f) or not torch.isfinite(g).all():
        raise FloatingPointError("Nonfinite sparse inverse objective")
    for backtracks in range(max_backtracks):
        candidate = project_complex_l1_ball(x-g/L, tau)
        delta = candidate-x
        smooth = float(value(candidate))
        upper = f+float((g.conj()*delta).real.sum())+.5*L*float(delta.abs().square().sum())
        slack = 64*torch.finfo(x.real.dtype).eps*max(abs(f), abs(upper), 1e-30)
        if math.isfinite(smooth) and smooth <= upper+slack:
            break
        L *= 2
    else:
        raise RuntimeError("Constrained sparse inverse line search failed")
    smooth, gradient = value_and_grad(candidate)
    if not math.isfinite(smooth) or not torch.isfinite(gradient).all():
        raise FloatingPointError("Nonfinite sparse inverse candidate")
    dual_norm = float(gradient.abs().max())
    gap = max(0., float((candidate.conj()*gradient).real.sum())+tau*dual_norm)
    stationary = gap <= optimality_rtol*max(smooth, target, 1e-30)
    relative_error = (smooth-target)/target
    # The zero field is the global minimum if it is already feasible.
    s["converged"] = bool(stationary and
        (abs(relative_error) <= residual_rtol or (tau == 0 and smooth <= target)))
    evaluated_radius = tau
    if stationary and not s["converged"]:
        if smooth > target:
            s["lower_radius"] = max(s["lower_radius"], tau)
        else:
            s["upper_radius"] = tau if s["upper_radius"] is None else min(tau, s["upper_radius"])
        if dual_norm == 0:
            s["stalled"] = True  # An infeasible least-squares floor, never success.
        else:
            residual = math.sqrt(2*smooth)
            new_tau = tau+(residual-math.sqrt(2*target))*residual/dual_norm
            lo, hi = s["lower_radius"], s["upper_radius"]
            if hi is not None and not lo < new_tau < hi:
                new_tau = (lo+hi)/2
            s["radius"] = max(0., new_tau)
            s["root_updates"] += 1
    s["lipschitz"] = L
    audit = dict(data_loss=smooth, target_data_loss=target,
        residual_relative_error=relative_error, residual_feasible=smooth <= target*(1+residual_rtol),
        l1_norm=float(candidate.abs().sum()), evaluated_l1_radius=evaluated_radius,
        subproblem_duality_gap=gap, subproblem_stationary=stationary,
        converged=s["converged"], stalled=s["stalled"], backtracks=backtracks,
        root_updates=s["root_updates"])
    return candidate.detach(), s, audit
