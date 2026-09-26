"""PVC copy of the historical strict iso-point refresh.

Copied from rift.sugavanam_ertin_a320_stabilized.refresh_iso_points_strict.
Only the three random-draw APIs use the CPU-generator proxy. Projection,
uniformization, both 60% acceptance gates and all audit fields are unchanged.
The original module and its function globals remain untouched.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn
from rift.sugavanam_ertin_a320_stabilized import (
    ProjectionAcceptanceError, PROJECTION_ITERATIONS, PROJECTION_TOLERANCE,
    PROJECTION_MIN_ACCEPTANCE, project_to_zero_level_strict, uniformize_points,
)
from rift_pvc.sugavanam_ertin_stage2_sampling import CpuGeneratorTorch

_sampling = CpuGeneratorTorch()


def refresh_iso_points_strict(
    model: nn.Module,
    seed_points: torch.Tensor,
    extent: float,
    pitch: float,
    n_points: int,
    generator: torch.Generator,
    oversample: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Two-stage refresh with separate >=60% projection gates."""
    if n_points < 3 or len(seed_points) < 3 or oversample < 1.0:
        raise ValueError("invalid strict refresh counts")
    attempted = max(n_points, int(math.ceil(n_points * oversample)))
    index = _sampling.randint(len(seed_points), (attempted,), generator=generator, device=seed_points.device)
    jitter = _sampling.randn(
        attempted,
        3,
        generator=generator,
        device=seed_points.device,
        dtype=seed_points.dtype,
    ) * pitch
    proposed = seed_points[index] + jitter
    initially_inside = (proposed.abs() < extent).all(-1)
    initial = proposed[initially_inside]
    if len(initial) < 3:
        raise ProjectionAcceptanceError(
            "all proposed iso-points left the ROI",
            {
                "status": "failed",
                "failure_stage": "proposal",
                "attempted": attempted,
                "accepted": 0,
                "rejected": attempted,
                "outside_roi": attempted,
                "boundary_clamped": 0,
                "nonfinite": 0,
                "degenerate_gradient": 0,
                "iterations": PROJECTION_ITERATIONS,
                "tolerance": PROJECTION_TOLERANCE,
                "min_acceptance": PROJECTION_MIN_ACCEPTANCE,
            },
        )
    first = project_to_zero_level_strict(model, initial, extent, 2.0 * pitch)
    first_fraction = first.accepted / attempted
    first_audit = {
        "status": "failed",
        "failure_stage": "initial_projection",
        "attempted": attempted,
        "initially_inside": int(initially_inside.sum()),
        "accepted": int(first.accepted),
        "rejected": int(attempted - first.accepted),
        "outside_roi": int((~initially_inside).sum()),
        "boundary_clamped": int(first.boundary_clamped),
        "nonfinite": int(first.nonfinite),
        "degenerate_gradient": int(first.degenerate_gradient),
        "first_acceptance_fraction": first_fraction,
        "first_projection": first.as_dict(),
        "iterations": PROJECTION_ITERATIONS,
        "tolerance": PROJECTION_TOLERANCE,
        "min_acceptance": PROJECTION_MIN_ACCEPTANCE,
    }
    if first.accepted < 3 or first_fraction < PROJECTION_MIN_ACCEPTANCE:
        raise ProjectionAcceptanceError(
            f"initial projection accepted {first.accepted}/{attempted}", first_audit
        )

    candidates = first.points
    if len(candidates) > n_points:
        order = _sampling.randperm(len(candidates), generator=generator, device=candidates.device)
        candidates = candidates[order[:n_points]]
    moved_all = uniformize_points(candidates, bandwidth=2.0 * pitch)
    moved_inside = (moved_all.abs() < extent).all(-1)
    moved = moved_all[moved_inside]
    if len(moved) < 3:
        failed = dict(first_audit)
        failed.update(
            {
                "failure_stage": "uniformization_domain",
                "accepted": 0,
                "rejected": attempted,
                "outside_roi": int((~initially_inside).sum()) + int((~moved_inside).sum()),
            }
        )
        raise ProjectionAcceptanceError("uniformization left fewer than three in-domain points", failed)
    second = project_to_zero_level_strict(model, moved, extent, 2.0 * pitch)
    second_fraction = second.accepted / max(len(candidates), 1)
    final = second.points
    if len(final) > n_points:
        order = _sampling.randperm(len(final), generator=generator, device=final.device)
        final = final[order[:n_points]]
    audit = {
        "status": "passed",
        "attempted": attempted,
        "accepted": int(len(final)),
        "first_acceptance_fraction": first_fraction,
        "second_acceptance_fraction": second_fraction,
        "policy_acceptance_fraction": min(first_fraction, second_fraction),
        "outside_roi": int((~initially_inside).sum()) + int((~moved_inside).sum()),
        "rejected": int(attempted - len(final)),
        "boundary_clamped": int(first.boundary_clamped + second.boundary_clamped),
        "nonfinite": int(first.nonfinite + second.nonfinite),
        "degenerate_gradient": int(first.degenerate_gradient + second.degenerate_gradient),
        "first_projection": first.as_dict(),
        "second_projection": second.as_dict(),
        "iterations": PROJECTION_ITERATIONS,
        "tolerance": PROJECTION_TOLERANCE,
        "min_acceptance": PROJECTION_MIN_ACCEPTANCE,
    }
    if second_fraction < PROJECTION_MIN_ACCEPTANCE or len(final) < min(n_points, 32):
        audit.update({"status": "failed", "failure_stage": "uniformized_projection"})
        raise ProjectionAcceptanceError(
            f"second projection accepted {second.accepted}/{len(candidates)}", audit
        )
    return final.detach(), audit
