"""Validity-gated SDF utilities for the stabilized Sugavanam--Ertin SE2 arm.

This module is intentionally separate from :mod:`rift.sugavanam_ertin` so the
preserved raw SE1 implementation and the PublicRadar reproduction do not change.
SE2 is a labelled stabilized derivative: it adds a closed geometric warm start,
interior/exterior sign anchors, explicit field-validity gates, and projection
audits.  No geometry truth is accepted by any API in this file.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn

from rift.sugavanam_ertin import uniformize_points


SE2_METHOD_NAME = "Sugavanam--Ertin valid-zero stabilized derivative"
SE2_MANAGER_IDENTITY = "rift_sugertin_validzero_v2"
SE2_ARTIFACT_IDENTITY = "b787_sugavanam_ertin_validzero_v2"
SE2_POLICY = "closed_sphere_init_sign_anchors_valid_projection_v2"


class ProjectionAcceptanceError(RuntimeError):
    """Projection-policy failure carrying the audit that explains it."""

    def __init__(self, message: str, audit: Dict[str, object]) -> None:
        super().__init__(message)
        self.audit = audit


@dataclass(frozen=True)
class ClosedFieldSpec:
    """Stage-1-only closed-field initialization parameters."""

    center: Tuple[float, float, float]
    radius: float
    extent: float
    pitch: float
    radius_quantile: float
    radius_cap_fraction: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "center": list(self.center),
            "radius": self.radius,
            "extent": self.extent,
            "pitch": self.pitch,
            "radius_quantile": self.radius_quantile,
            "radius_cap_fraction": self.radius_cap_fraction,
        }


@dataclass
class ProjectionResult:
    """Accepted projected points plus a complete rejection ledger."""

    points: torch.Tensor
    residuals: torch.Tensor
    attempted: int
    accepted: int
    rejected: int
    boundary_clamped: int
    nonfinite: int
    residual_max: float
    residual_q50: float
    residual_q95: float
    residual_q99: float
    finite_residual_max: float
    finite_residual_q50: float
    finite_residual_q95: float
    finite_residual_q99: float
    gradient_norm_min: float
    gradient_norm_q05: float
    gradient_norm_q50: float
    gradient_norm_q95: float
    accepted_by_iteration: Tuple[int, ...]

    def as_dict(self) -> Dict[str, object]:
        return {
            "attempted": self.attempted,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "acceptance_fraction": self.accepted / max(self.attempted, 1),
            "boundary_clamped": self.boundary_clamped,
            "nonfinite": self.nonfinite,
            "residual_max": self.residual_max,
            "residual_q50": self.residual_q50,
            "residual_q95": self.residual_q95,
            "residual_q99": self.residual_q99,
            "finite_residual_max": self.finite_residual_max,
            "finite_residual_q50": self.finite_residual_q50,
            "finite_residual_q95": self.finite_residual_q95,
            "finite_residual_q99": self.finite_residual_q99,
            "gradient_norm_min": self.gradient_norm_min,
            "gradient_norm_q05": self.gradient_norm_q05,
            "gradient_norm_q50": self.gradient_norm_q50,
            "gradient_norm_q95": self.gradient_norm_q95,
            "accepted_by_iteration": list(self.accepted_by_iteration),
        }


def closed_field_spec(
    points: np.ndarray,
    extent: float,
    pitch: float,
    radius_quantile: float = 0.5,
    radius_cap_fraction: float = 0.65,
) -> ClosedFieldSpec:
    """Derive one deterministic closed sphere from the Stage-1 cloud only."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3:
        raise ValueError("points must have shape [N,3] with N>=3")
    if not np.isfinite(pts).all() or extent <= 0 or pitch <= 0:
        raise ValueError("points, extent, and pitch must be finite and positive")
    if not (0.1 <= radius_quantile <= 0.9):
        raise ValueError("radius_quantile must be in [0.1,0.9]")
    if not (0.25 <= radius_cap_fraction <= 0.9):
        raise ValueError("radius_cap_fraction must be in [0.25,0.9]")
    if np.max(np.abs(pts)) >= extent:
        raise ValueError("Stage-1 points must lie strictly inside the SDF ROI")

    center = np.median(pts, axis=0)
    clearance = float(np.min(extent - np.abs(center)))
    upper = radius_cap_fraction * clearance
    lower = 2.0 * pitch
    if upper <= lower:
        raise ValueError("Stage-1 cloud leaves no room for a closed initialization")
    distances = np.linalg.norm(pts - center[None, :], axis=1)
    radius = float(np.clip(np.quantile(distances, radius_quantile), lower, upper))
    return ClosedFieldSpec(
        center=tuple(float(x) for x in center),
        radius=radius,
        extent=float(extent),
        pitch=float(pitch),
        radius_quantile=float(radius_quantile),
        radius_cap_fraction=float(radius_cap_fraction),
    )


def analytic_sphere_sdf(xyz: torch.Tensor, spec: ClosedFieldSpec) -> torch.Tensor:
    center = xyz.new_tensor(spec.center)
    return (xyz - center).norm(dim=-1) - spec.radius


def sample_roi(
    n: int,
    extent: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if n <= 0:
        raise ValueError("sample count must be positive")
    return (2.0 * torch.rand(n, 3, generator=generator, device=device, dtype=dtype) - 1.0) * extent


def sample_boundary(
    n: int,
    extent: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample exact cube faces, including both exterior orientations."""
    points = sample_roi(n, extent, generator, device, dtype)
    axes = torch.randint(3, (n,), generator=generator, device=device)
    signs = torch.randint(2, (n,), generator=generator, device=device, dtype=torch.int64)
    signs = signs.to(dtype=dtype).mul_(2.0).sub_(1.0)
    points[torch.arange(n, device=device), axes] = signs * extent
    return points


def sample_inner(
    n: int,
    spec: ClosedFieldSpec,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample the fixed inner ball implied by the closed initialization."""
    directions = torch.randn(n, 3, generator=generator, device=device, dtype=dtype)
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    radii = torch.rand(n, 1, generator=generator, device=device, dtype=dtype).pow(1.0 / 3.0)
    center = torch.as_tensor(spec.center, device=device, dtype=dtype)
    return center + directions * radii * (0.25 * spec.radius)


def closed_anchor_losses(
    model: nn.Module,
    boundary_points: torch.Tensor,
    inner_points: torch.Tensor,
    margin: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Hinge losses that preserve positive exterior and negative interior."""
    if margin <= 0:
        raise ValueError("anchor margin must be positive")
    boundary = torch.relu(margin - model(boundary_points)).mean()
    interior = torch.relu(model(inner_points) + margin).mean()
    return boundary, interior


def ramped_weight(step: int, maximum: float, warmup: int, ramp: int) -> float:
    """Monotone zero-to-maximum schedule used by the off-surface term."""
    if step <= warmup:
        return 0.0
    if ramp <= 0:
        return float(maximum)
    fraction = min(max((step - warmup) / float(ramp), 0.0), 1.0)
    return float(maximum) * fraction


@torch.no_grad()
def evaluate_field_grid(
    model: nn.Module,
    extent: float,
    grid: int,
    device: torch.device,
    chunk: int = 65536,
) -> Tuple[np.ndarray, float]:
    """Evaluate an inclusive grid so its outer samples are the exact ROI boundary."""
    if grid < 8 or chunk <= 0:
        raise ValueError("grid must be >=8 and chunk must be positive")
    axis = torch.linspace(-extent, extent, grid, device=device)
    yz = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1).reshape(-1, 2)
    slabs = []
    for x in axis:
        xyz = torch.cat((x.expand(len(yz), 1), yz), dim=-1)
        values = []
        for start in range(0, len(xyz), chunk):
            values.append(model(xyz[start : start + chunk]).detach().cpu())
        slabs.append(torch.cat(values))
    field = torch.stack(slabs).reshape(grid, grid, grid).numpy()
    pitch = 2.0 * extent / (grid - 1)
    return field, pitch


def boundary_values(field: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            field[0, :, :].ravel(), field[-1, :, :].ravel(),
            field[:, 0, :].ravel(), field[:, -1, :].ravel(),
            field[:, :, 0].ravel(), field[:, :, -1].ravel(),
        )
    )


def field_validity_from_array(field: np.ndarray, margin: float) -> Dict[str, object]:
    values = np.asarray(field)
    if values.ndim != 3 or min(values.shape) < 2:
        raise ValueError("field must be a 3D grid")
    finite = bool(np.isfinite(values).all())
    if not finite:
        return {
            "passed": False,
            "finite": False,
            "field_min": float("nan"),
            "field_max": float("nan"),
            "boundary_min": float("nan"),
            "boundary_max": float("nan"),
            "strict_crossing": False,
            "exterior_positive": False,
            "margin": float(margin),
        }
    boundary = boundary_values(values)
    lo, hi = float(values.min()), float(values.max())
    b_lo, b_hi = float(boundary.min()), float(boundary.max())
    crossing = lo < -margin and hi > margin
    exterior = b_lo > margin
    return {
        "passed": bool(crossing and exterior),
        "finite": True,
        "field_min": lo,
        "field_max": hi,
        "boundary_min": b_lo,
        "boundary_max": b_hi,
        "strict_crossing": bool(crossing),
        "exterior_positive": bool(exterior),
        "margin": float(margin),
    }


def field_validity_gate(
    model: nn.Module,
    extent: float,
    grid: int,
    margin: float,
    device: torch.device,
    chunk: int = 65536,
) -> Dict[str, object]:
    field, pitch = evaluate_field_grid(model, extent, grid, device, chunk)
    result = field_validity_from_array(field, margin)
    result.update({"grid": int(grid), "pitch": float(pitch)})
    return result


def _projection_summary(
    q: torch.Tensor,
    residual: torch.Tensor,
    accepted_mask: torch.Tensor,
    clamped_mask: torch.Tensor,
    gradient_norm: torch.Tensor,
    accepted_by_iteration: Tuple[int, ...],
) -> ProjectionResult:
    finite_mask = torch.isfinite(residual) & torch.isfinite(q).all(dim=-1)
    accepted_mask = accepted_mask & finite_mask
    accepted_residual = residual[accepted_mask].detach()
    if accepted_residual.numel():
        quantiles = torch.quantile(accepted_residual, accepted_residual.new_tensor([0.5, 0.95, 0.99]))
        residual_max = float(accepted_residual.max())
        q50, q95, q99 = (float(x) for x in quantiles)
    else:
        residual_max = q50 = q95 = q99 = float("inf")
    finite_residual = residual[finite_mask].detach()
    if finite_residual.numel():
        quantiles = torch.quantile(
            finite_residual, finite_residual.new_tensor([0.5, 0.95, 0.99])
        )
        finite_residual_max = float(finite_residual.max())
        finite_q50, finite_q95, finite_q99 = (float(x) for x in quantiles)
    else:
        finite_residual_max = finite_q50 = finite_q95 = finite_q99 = float("inf")
    finite_gradient = gradient_norm[torch.isfinite(gradient_norm)].detach()
    if finite_gradient.numel():
        quantiles = torch.quantile(
            finite_gradient, finite_gradient.new_tensor([0.05, 0.5, 0.95])
        )
        gradient_min = float(finite_gradient.min())
        gradient_q05, gradient_q50, gradient_q95 = (float(x) for x in quantiles)
    else:
        gradient_min = gradient_q05 = gradient_q50 = gradient_q95 = float("nan")
    attempted = int(len(q))
    accepted = int(accepted_mask.sum())
    return ProjectionResult(
        points=q[accepted_mask].detach(),
        residuals=accepted_residual,
        attempted=attempted,
        accepted=accepted,
        rejected=attempted - accepted,
        boundary_clamped=int(clamped_mask.sum()),
        nonfinite=int((~finite_mask).sum()),
        residual_max=residual_max,
        residual_q50=q50,
        residual_q95=q95,
        residual_q99=q99,
        finite_residual_max=finite_residual_max,
        finite_residual_q50=finite_q50,
        finite_residual_q95=finite_q95,
        finite_residual_q99=finite_q99,
        gradient_norm_min=gradient_min,
        gradient_norm_q05=gradient_q05,
        gradient_norm_q50=gradient_q50,
        gradient_norm_q95=gradient_q95,
        accepted_by_iteration=accepted_by_iteration,
    )


def project_to_zero_level_valid(
    model: nn.Module,
    points: torch.Tensor,
    extent: float,
    max_step: float,
    iterations: int = 12,
    tolerance: float = 1e-4,
) -> ProjectionResult:
    """Newton projection that rejects nonconverged and ROI-clamped points."""
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("points must have nonempty shape [N,3]")
    if extent <= 0 or max_step <= 0 or iterations <= 0 or tolerance <= 0:
        raise ValueError("projection parameters must be positive")
    q = points.detach().clone()
    clamped = torch.zeros(len(q), dtype=torch.bool, device=q.device)
    accepted_by_iteration = []
    for _ in range(iterations):
        q.requires_grad_(True)
        sdf = model(q)
        grad = torch.autograd.grad(sdf.sum(), q, create_graph=False)[0]
        residual = sdf.detach().abs()
        converged = (
            torch.isfinite(residual)
            & torch.isfinite(q).all(dim=-1)
            & (residual <= tolerance)
            & (~clamped)
        )
        accepted_by_iteration.append(int(converged.sum()))
        active = torch.isfinite(residual) & (residual > tolerance)
        if not bool(active.any()):
            q = q.detach()
            break
        update = sdf[:, None] * grad / grad.square().sum(-1, keepdim=True).clamp_min(1e-12)
        norm = update.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        update = update * torch.clamp(max_step / norm, max=1.0)
        update = torch.where(active[:, None], update, torch.zeros_like(update))
        proposed = q.detach() - update.detach()
        hit = (proposed.abs() >= extent).any(dim=-1)
        clamped |= hit
        q = proposed.clamp(-extent, extent)
    q = q.detach().requires_grad_(True)
    sdf = model(q)
    grad = torch.autograd.grad(sdf.sum(), q, create_graph=False)[0]
    residual = sdf.detach().abs()
    gradient_norm = grad.detach().norm(dim=-1)
    accepted = (residual <= tolerance) & (~clamped)
    accepted_by_iteration.append(int(accepted.sum()))
    return _projection_summary(
        q.detach(),
        residual,
        accepted,
        clamped,
        gradient_norm,
        tuple(accepted_by_iteration),
    )


def refresh_iso_points_valid(
    model: nn.Module,
    seed_points: torch.Tensor,
    extent: float,
    pitch: float,
    n_points: int,
    generator: torch.Generator,
    tolerance: float = 1e-4,
    iterations: int = 12,
    min_acceptance: float = 0.5,
    oversample: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Create iso-points while carrying only converged, unclamped projections."""
    if not (0.0 < min_acceptance <= 1.0) or oversample < 1.0:
        raise ValueError("invalid projection acceptance policy")
    if n_points < 3 or len(seed_points) < 3:
        raise ValueError("at least three seed and target points are required")
    attempted = max(n_points, int(math.ceil(n_points * oversample)))
    idx = torch.randint(len(seed_points), (attempted,), generator=generator, device=seed_points.device)
    jitter = torch.randn(
        attempted, 3, generator=generator, device=seed_points.device, dtype=seed_points.dtype
    ) * pitch
    initial = seed_points[idx] + jitter
    initially_inside = (initial.abs() < extent).all(dim=-1)
    initial = initial[initially_inside]
    if len(initial) < 3:
        raise RuntimeError("all proposed iso-points left the ROI")

    first = project_to_zero_level_valid(
        model, initial, extent, max_step=2.0 * pitch, iterations=iterations, tolerance=tolerance
    )
    # Count proposals rejected for leaving the ROI before Newton projection.
    first_fraction = first.accepted / attempted
    first_audit = {
        "status": "failed",
        "failure_stage": "initial_projection",
        "attempted": attempted,
        "initially_inside": int(initially_inside.sum()),
        "accepted": first.accepted,
        "rejected": attempted - first.accepted,
        "policy_acceptance_fraction": first_fraction,
        "min_acceptance": float(min_acceptance),
        "outside_roi": int((~initially_inside).sum()),
        "boundary_clamped": first.boundary_clamped,
        "tolerance": float(tolerance),
        "iterations": int(iterations),
        "first_projection": first.as_dict(),
    }
    if first.accepted < 3 or first_fraction < min_acceptance:
        raise ProjectionAcceptanceError(
            "initial iso projection acceptance below policy: "
            f"accepted={first.accepted}/{attempted} threshold={min_acceptance:g}",
            first_audit,
        )
    candidates = first.points
    if len(candidates) > n_points:
        order = torch.randperm(len(candidates), generator=generator, device=candidates.device)[:n_points]
        candidates = candidates[order]

    moved = uniformize_points(candidates, bandwidth=2.0 * pitch)
    moved_inside = (moved.abs() < extent).all(dim=-1)
    moved = moved[moved_inside]
    if len(moved) < 3:
        raise RuntimeError("uniformization moved every iso-point outside the ROI")
    second = project_to_zero_level_valid(
        model, moved, extent, max_step=2.0 * pitch, iterations=iterations, tolerance=tolerance
    )
    # Likewise count points rejected after uniformization for leaving the ROI.
    second_fraction = second.accepted / max(len(candidates), 1)
    final = second.points
    if len(final) > n_points:
        order = torch.randperm(len(final), generator=generator, device=final.device)[:n_points]
        final = final[order]

    accepted = int(len(final))
    acceptance_fraction = accepted / attempted
    policy_acceptance_fraction = min(first_fraction, second_fraction)
    outside_roi = int((~initially_inside).sum()) + int((~moved_inside).sum())
    boundary_clamped = first.boundary_clamped + second.boundary_clamped
    audit = {
        "status": "passed",
        "attempted": attempted,
        "accepted": accepted,
        "rejected": attempted - accepted,
        "acceptance_fraction": acceptance_fraction,
        "policy_acceptance_fraction": policy_acceptance_fraction,
        "first_acceptance_fraction": first_fraction,
        "second_acceptance_fraction": second_fraction,
        "outside_roi": outside_roi,
        "boundary_clamped": boundary_clamped,
        "first_projection": first.as_dict(),
        "second_projection": second.as_dict(),
        "tolerance": float(tolerance),
        "iterations": int(iterations),
        "min_acceptance": float(min_acceptance),
    }
    if policy_acceptance_fraction < min_acceptance or accepted < min(n_points, 32):
        failed_audit = dict(audit)
        failed_audit.update({"status": "failed", "failure_stage": "uniformized_projection"})
        raise ProjectionAcceptanceError(
            "iso projection acceptance below policy: "
            f"accepted={accepted}/{attempted} threshold={min_acceptance:g}",
            failed_audit,
        )
    return final.detach(), audit


def mesh_topology_audit(vertices: np.ndarray, faces: np.ndarray) -> Dict[str, object]:
    """Audit a triangle mesh for finite, closed two-manifold topology."""
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("mesh must contain [V,3] vertices and [F,3] faces")
    if len(vertices) == 0 or len(faces) == 0 or not np.isfinite(vertices).all():
        return {
            "passed": False,
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
            "boundary_edges": -1,
            "nonmanifold_edges": -1,
            "components": 0,
        }
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    edges = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = int(np.sum(counts == 1))
    nonmanifold_edges = int(np.sum(counts > 2))

    parent = np.arange(len(vertices), dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in unique_edges:
        union(int(a), int(b))
    used = np.unique(faces)
    components = len({find(int(x)) for x in used})
    return {
        "passed": bool(boundary_edges == 0 and nonmanifold_edges == 0 and components == 1),
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "edges": int(len(unique_edges)),
        "boundary_edges": boundary_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "components": int(components),
    }
