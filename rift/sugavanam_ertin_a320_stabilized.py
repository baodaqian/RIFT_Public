"""A320-specific validity machinery for a stabilized Sugavanam--Ertin derivative.

This module is additive.  It does not change the preserved raw reproduction or
the earlier B787 valid-zero derivative.  Every geometric prior here is derived
from the retained Stage-1 scattering cloud and the declared reconstruction ROI;
ground-truth geometry is neither accepted nor loaded.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch import nn

from rift.sugavanam_ertin import METHOD_NAME as RAW_METHOD_NAME
from rift.sugavanam_ertin import uniformize_points
from rift.sugavanam_ertin_validzero import (
    ClosedFieldSpec,
    analytic_sphere_sdf,
    closed_anchor_losses,
    closed_field_spec,
    evaluate_field_grid,
    field_validity_from_array,
    field_validity_gate,
    mesh_topology_audit,
    ramped_weight,
    sample_roi,
)


A320_METHOD_NAME = "Sugavanam--Ertin A320 valid-zero stabilized derivative"
A320_MANAGER_IDENTITY = "rift_mesh10k_a320_se_validzero_smoke_v1"
A320_ARTIFACT_IDENTITY = "a320_sugavanam_ertin_validzero_smoke_v1"
A320_POLICY = "stage1_closed_init_oriented_offsets_valid_projection_v1"
A320_IMPLEMENTATION_KIND = "stabilized derivative; not the raw paper reproduction"
A320_STAGE1_CHECKPOINT = (
    "/storage/scratch1/1/dbao31/rift_mesh10k_baselines_20260830_v1/"
    "a320/se/scatter/checkpoint_final.pth.tar"
)
A320_OUTPUT_DIR = (
    "/storage/scratch1/1/dbao31/"
    "rift_mesh10k_a320_se_validzero_smoke_v1"
)

PROJECTION_ITERATIONS = 24
PROJECTION_TOLERANCE = 1.0e-4
PROJECTION_MIN_ACCEPTANCE = 0.60


class ProjectionAcceptanceError(RuntimeError):
    """Projection-policy failure carrying the complete rejection audit."""

    def __init__(self, message: str, audit: Dict[str, object]) -> None:
        super().__init__(message)
        self.audit = audit


@dataclass
class ProjectionResult:
    points: torch.Tensor
    residuals: torch.Tensor
    attempted: int
    accepted: int
    rejected: int
    boundary_clamped: int
    outside_domain: int
    nonfinite: int
    degenerate_gradient: int
    residual_max: float
    residual_q50: float
    residual_q95: float
    residual_q99: float
    gradient_norm_min: float
    gradient_norm_q05: float
    accepted_by_iteration: Tuple[int, ...]

    def as_dict(self) -> Dict[str, object]:
        return {
            "attempted": self.attempted,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "acceptance_fraction": self.accepted / max(self.attempted, 1),
            "boundary_clamped": self.boundary_clamped,
            "outside_domain": self.outside_domain,
            "nonfinite": self.nonfinite,
            "degenerate_gradient": self.degenerate_gradient,
            "residual_max": self.residual_max,
            "residual_q50": self.residual_q50,
            "residual_q95": self.residual_q95,
            "residual_q99": self.residual_q99,
            "gradient_norm_min": self.gradient_norm_min,
            "gradient_norm_q05": self.gradient_norm_q05,
            "accepted_by_iteration": list(self.accepted_by_iteration),
        }


def orient_normals_outward(
    points: np.ndarray,
    normals: np.ndarray,
    center: Sequence[float],
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Give PCA normals one deterministic Stage-1-centred orientation."""
    pts = np.asarray(points, dtype=np.float64)
    vec = np.asarray(normals, dtype=np.float64)
    ctr = np.asarray(center, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or vec.shape != pts.shape or ctr.shape != (3,):
        raise ValueError("points/normals must be [N,3] and center must be [3]")
    if (
        len(pts) < 3
        or not np.isfinite(pts).all()
        or not np.isfinite(vec).all()
        or not np.isfinite(ctr).all()
    ):
        raise ValueError("normal orientation requires finite samples and center")
    norm = np.linalg.norm(vec, axis=1)
    if np.any(norm <= 1.0e-10) or not np.allclose(norm, 1.0, rtol=1.0e-3, atol=1.0e-4):
        raise ValueError("normal orientation requires finite unit PCA normals")
    vec = vec / norm[:, None]
    # First propagate signs through a deterministic k-nearest-neighbour forest.
    # This removes independent eigensolver sign choices before selecting the
    # aggregate outward sign of each connected component.
    k = min(12, len(pts) - 1)
    _, neighbours = cKDTree(pts).query(pts, k=k + 1)
    adjacency = [set() for _ in range(len(pts))]
    for index, row in enumerate(np.atleast_2d(neighbours)):
        for neighbour in np.atleast_1d(row)[1:]:
            other = int(neighbour)
            adjacency[index].add(other)
            adjacency[other].add(index)
    visited = np.zeros(len(pts), dtype=bool)
    component_count = 0
    propagated_flips = 0
    tree_edges = []
    for seed in range(len(pts)):
        if visited[seed]:
            continue
        component_count += 1
        visited[seed] = True
        component = [seed]
        queue = deque([seed])
        while queue:
            parent = queue.popleft()
            for child in sorted(adjacency[parent]):
                if visited[child]:
                    continue
                if float(np.dot(vec[parent], vec[child])) < 0.0:
                    vec[child] *= -1.0
                    propagated_flips += 1
                visited[child] = True
                component.append(child)
                queue.append(child)
                tree_edges.append((parent, child))
        component_index = np.asarray(component, dtype=np.int64)
        radial_component = pts[component_index] - ctr[None, :]
        aggregate = float(np.einsum("ij,ij->", vec[component_index], radial_component))
        if aggregate < 0.0:
            vec[component_index] *= -1.0
            propagated_flips += len(component)
    radial = pts - ctr[None, :]
    final_alignment = np.einsum("ij,ij->i", vec, radial)
    tree_cosine = np.asarray(
        [float(np.dot(vec[a], vec[b])) for a, b in tree_edges], dtype=np.float64
    )
    tree_consistent = bool(len(tree_cosine) == 0 or np.all(tree_cosine >= -1.0e-8))
    if not tree_consistent:
        raise ValueError("normal-orientation propagation is locally inconsistent")
    audit = {
        "count": int(len(vec)),
        "flipped": int(propagated_flips),
        "components": int(component_count),
        "tree_edges": int(len(tree_edges)),
        "tree_consistent": tree_consistent,
        "tree_cosine_min": float(tree_cosine.min()) if len(tree_cosine) else 1.0,
        "finite": bool(np.isfinite(vec).all()),
        "unit_norm_max_error": float(np.max(np.abs(np.linalg.norm(vec, axis=1) - 1.0))),
        "radial_alignment_min": float(final_alignment.min()),
        "radial_alignment_mean": float(final_alignment.mean()),
        "orientation_policy": "knn_forest_consistency_then_stage1_center_outward",
    }
    return vec.astype(np.float32), audit


def oriented_normal_loss(gradient: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    """Signed cosine loss; a reversed normal is penalized instead of accepted."""
    cosine = F.cosine_similarity(gradient, normal, dim=-1, eps=1.0e-8)
    return (1.0 - cosine).mean()


def signed_offset_samples(
    points: np.ndarray,
    normals: np.ndarray,
    extent: float,
    offset: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Create paired exterior-positive/interior-negative Stage-1 offsets."""
    pts = np.asarray(points, dtype=np.float64)
    vec = np.asarray(normals, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or vec.shape != pts.shape:
        raise ValueError("points and normals must have matching [N,3] shape")
    if not np.isfinite(pts).all() or not np.isfinite(vec).all():
        raise ValueError("signed-offset inputs must be finite")
    normal_norm = np.linalg.norm(vec, axis=1)
    if np.any(normal_norm <= 1.0e-10) or not np.allclose(
        normal_norm, 1.0, rtol=1.0e-3, atol=1.0e-4
    ):
        raise ValueError("signed-offset normals must be unit length")
    if extent <= 0.0 or offset <= 0.0 or offset >= extent:
        raise ValueError("extent and signed offset must be positive with offset<extent")
    exterior = pts + offset * vec
    interior = pts - offset * vec
    paired = np.isfinite(exterior).all(axis=1) & np.isfinite(interior).all(axis=1)
    paired &= (np.abs(exterior) < extent).all(axis=1)
    paired &= (np.abs(interior) < extent).all(axis=1)
    pair_fraction = float(paired.mean())
    if int(paired.sum()) < 3 or pair_fraction < 0.60:
        raise ValueError("fewer than 60% of signed-offset pairs remain in the ROI")
    samples = np.concatenate((exterior[paired], interior[paired]), axis=0).astype(np.float32)
    targets = np.concatenate(
        (
            np.full(int(paired.sum()), offset, dtype=np.float32),
            np.full(int(paired.sum()), -offset, dtype=np.float32),
        )
    )
    audit = {
        "source_count": int(len(pts)),
        "paired_count": int(paired.sum()),
        "paired_fraction": pair_fraction,
        "rejected_pairs": int(len(pts) - paired.sum()),
        "sample_count": int(len(samples)),
        "offset": float(offset),
        "finite": bool(np.isfinite(samples).all() and np.isfinite(targets).all()),
        "strictly_inside_roi": bool((np.abs(samples) < extent).all()),
    }
    return samples, targets, audit


def signed_offset_loss(
    model: nn.Module,
    points: torch.Tensor,
    targets: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    if beta <= 0.0 or len(points) != len(targets):
        raise ValueError("signed-offset beta/count contract failed")
    return F.smooth_l1_loss(model(points), targets, beta=beta)


def deterministic_boundary_shell(
    extent: float,
    pitch: float,
    resolution: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    layers: int = 3,
) -> torch.Tensor:
    """Fixed planes spanning the protected outer two-voxel ROI shell."""
    if extent <= 0.0 or pitch <= 0.0 or resolution < 3 or layers != 3:
        raise ValueError("boundary shell needs extent/pitch>0, resolution>=3, and three planes")
    if 2.0 * pitch >= extent:
        raise ValueError("two-voxel protected shell consumes the ROI")
    axis = torch.linspace(-extent, extent, resolution, device=device, dtype=dtype)
    u, v = torch.meshgrid(axis, axis, indexing="ij")
    faces = []
    for layer in range(layers):
        fixed_value = extent - layer * pitch
        for fixed_axis in range(3):
            free_axes = [axis_index for axis_index in range(3) if axis_index != fixed_axis]
            for sign in (-1.0, 1.0):
                face = torch.empty(resolution, resolution, 3, device=device, dtype=dtype)
                face[..., fixed_axis] = sign * fixed_value
                face[..., free_axes[0]] = u
                face[..., free_axes[1]] = v
                faces.append(face.reshape(-1, 3))
    return torch.cat(faces, dim=0)


def deterministic_inner_anchors(
    spec: ClosedFieldSpec,
    count: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Fixed Fibonacci-ball anchors inside the Stage-1-derived sphere."""
    if count < 3:
        raise ValueError("at least three interior anchors are required")
    index = torch.arange(count, device=device, dtype=dtype)
    fraction = (index + 0.5) / float(count)
    z = 1.0 - 2.0 * fraction
    theta = index * (math.pi * (3.0 - math.sqrt(5.0)))
    radial_xy = torch.sqrt(torch.clamp(1.0 - z.square(), min=0.0))
    direction = torch.stack((radial_xy * torch.cos(theta), radial_xy * torch.sin(theta), z), -1)
    radius = (0.25 * spec.radius) * fraction.pow(1.0 / 3.0)
    center = torch.as_tensor(spec.center, device=device, dtype=dtype)
    return center + radius[:, None] * direction


@torch.no_grad()
def protected_shell_gate(
    model: nn.Module,
    shell: torch.Tensor,
    margin: float,
    chunk: int = 65536,
) -> Dict[str, object]:
    if shell.ndim != 2 or shell.shape[1] != 3 or len(shell) == 0 or chunk <= 0:
        raise ValueError("protected shell must be a nonempty [N,3] tensor")
    values = []
    for start in range(0, len(shell), chunk):
        values.append(model(shell[start : start + chunk]).detach().cpu())
    value = torch.cat(values)
    finite = bool(torch.isfinite(value).all())
    lo = float(value.min()) if finite else float("nan")
    hi = float(value.max()) if finite else float("nan")
    return {
        "passed": bool(finite and lo > margin),
        "finite": finite,
        "count": int(len(shell)),
        "minimum": lo,
        "maximum": hi,
        "margin": float(margin),
    }


def strict_field_gate(
    model: nn.Module,
    extent: float,
    grid: int,
    margin: float,
    shell: torch.Tensor,
    device: torch.device,
    chunk: int = 65536,
) -> Dict[str, object]:
    grid_gate = field_validity_gate(model, extent, grid, margin, device, chunk)
    shell_gate = protected_shell_gate(model, shell, margin, chunk)
    result = dict(grid_gate)
    result["protected_shell"] = shell_gate
    result["passed"] = bool(grid_gate["passed"] and shell_gate["passed"])
    return result


def _quantiles(values: torch.Tensor) -> Tuple[float, float, float, float]:
    if values.numel() == 0:
        return (float("inf"),) * 4
    values = values.detach()
    q = torch.quantile(values, values.new_tensor([0.5, 0.95, 0.99]))
    return float(values.max()), float(q[0]), float(q[1]), float(q[2])


def project_to_zero_level_strict(
    model: nn.Module,
    points: torch.Tensor,
    extent: float,
    max_step: float,
    iterations: int = PROJECTION_ITERATIONS,
    tolerance: float = PROJECTION_TOLERANCE,
) -> ProjectionResult:
    """Newton projection accepting only finite, interior, unclamped roots."""
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("projection points must have nonempty [N,3] shape")
    if (
        min(extent, max_step, tolerance) <= 0.0
        or iterations != PROJECTION_ITERATIONS
        or not math.isclose(tolerance, PROJECTION_TOLERANCE, rel_tol=0.0, abs_tol=1.0e-12)
    ):
        raise ValueError(
            f"A320 projection requires exactly {PROJECTION_ITERATIONS} iterations "
            f"and tolerance {PROJECTION_TOLERANCE:g}"
        )
    q = points.detach().clone()
    initially_inside = torch.isfinite(q).all(-1) & (q.abs() < extent).all(-1)
    clamped = ~initially_inside
    accepted_by_iteration = []
    for _ in range(iterations):
        q.requires_grad_(True)
        sdf = model(q)
        grad = torch.autograd.grad(sdf.sum(), q, create_graph=False)[0]
        residual = sdf.detach().abs()
        gradient_norm = grad.detach().norm(dim=-1)
        finite = torch.isfinite(residual) & torch.isfinite(q).all(-1) & torch.isfinite(grad).all(-1)
        inside = (q.detach().abs() < extent).all(-1)
        converged = finite & inside & (~clamped) & (gradient_norm > 1.0e-10) & (residual <= tolerance)
        accepted_by_iteration.append(int(converged.sum()))
        active = finite & (gradient_norm > 1.0e-10) & (residual > tolerance)
        denominator = grad.square().sum(-1, keepdim=True).clamp_min(1.0e-12)
        update = sdf[:, None] * grad / denominator
        update_norm = update.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
        update = update * torch.clamp(max_step / update_norm, max=1.0)
        update = torch.where(active[:, None], update, torch.zeros_like(update))
        proposed = q.detach() - update.detach()
        hit = (proposed.abs() >= extent).any(-1)
        clamped |= hit
        q = proposed.clamp(-extent, extent)

    q = q.detach().requires_grad_(True)
    sdf = model(q)
    grad = torch.autograd.grad(sdf.sum(), q, create_graph=False)[0]
    residual = sdf.detach().abs()
    gradient_norm = grad.detach().norm(dim=-1)
    finite = torch.isfinite(residual) & torch.isfinite(q).all(-1) & torch.isfinite(grad).all(-1)
    inside = (q.detach().abs() < extent).all(-1)
    good_gradient = gradient_norm > 1.0e-10
    accepted_mask = finite & inside & (~clamped) & good_gradient & (residual <= tolerance)
    accepted_by_iteration.append(int(accepted_mask.sum()))
    maximum, q50, q95, q99 = _quantiles(residual[accepted_mask])
    finite_gradient = gradient_norm[torch.isfinite(gradient_norm)]
    if finite_gradient.numel():
        grad_min = float(finite_gradient.min())
        grad_q05 = float(torch.quantile(finite_gradient, finite_gradient.new_tensor(0.05)))
    else:
        grad_min = grad_q05 = float("nan")
    attempted = int(len(q))
    accepted = int(accepted_mask.sum())
    return ProjectionResult(
        points=q.detach()[accepted_mask],
        residuals=residual[accepted_mask],
        attempted=attempted,
        accepted=accepted,
        rejected=attempted - accepted,
        boundary_clamped=int(clamped.sum()),
        outside_domain=int((~inside).sum()),
        nonfinite=int((~finite).sum()),
        degenerate_gradient=int((~good_gradient).sum()),
        residual_max=maximum,
        residual_q50=q50,
        residual_q95=q95,
        residual_q99=q99,
        gradient_norm_min=grad_min,
        gradient_norm_q05=grad_q05,
        accepted_by_iteration=tuple(accepted_by_iteration),
    )


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
    index = torch.randint(len(seed_points), (attempted,), generator=generator, device=seed_points.device)
    jitter = torch.randn(
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
        order = torch.randperm(len(candidates), generator=generator, device=candidates.device)
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
        order = torch.randperm(len(final), generator=generator, device=final.device)
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


def topology_contract(
    vertices: np.ndarray,
    faces: np.ndarray,
    extent: float,
    protected_shell_depth: float = 0.0,
) -> Dict[str, object]:
    """Expose the final finite/watertight/manifold/component gates explicitly."""
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    shape_valid = bool(
        vertices.ndim == 2
        and vertices.shape[1:] == (3,)
        and faces.ndim == 2
        and faces.shape[1:] == (3,)
    )
    finite = bool(shape_valid and np.isfinite(vertices).all() and np.isfinite(faces).all())
    valid_faces = bool(
        shape_valid
        and len(vertices) > 0
        and len(faces) > 0
        and np.issubdtype(faces.dtype, np.integer)
        and np.all(faces >= 0)
        and np.all(faces < len(vertices))
    )
    if not finite or not valid_faces:
        return {
            "passed": False,
            "finite": finite,
            "valid_faces": valid_faces,
            "watertight": False,
            "manifold": False,
            "single_component": False,
            "nondegenerate": False,
            "inside_roi": False,
            "outside_protected_shell": False,
            "protected_shell_depth": float(protected_shell_depth),
            "vertices": int(len(vertices)) if vertices.ndim else 0,
            "faces": int(len(faces)) if faces.ndim else 0,
        }
    repeated = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
    )
    triangles = vertices[faces]
    twice_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    nondegenerate = bool(not repeated.any() and np.isfinite(twice_area).all() and np.all(twice_area > 1.0e-12))
    base = mesh_topology_audit(vertices, faces)
    clearance_min = float(np.min(extent - np.max(np.abs(vertices), axis=1)))
    inside_roi = bool(clearance_min > 0.0)
    outside_shell = bool(clearance_min > protected_shell_depth)
    result = dict(base)
    result.update(
        {
            "finite": finite,
            "valid_faces": valid_faces,
            "nondegenerate": nondegenerate,
            "watertight": bool(base.get("boundary_edges") == 0),
            "manifold": bool(base.get("nonmanifold_edges") == 0),
            "single_component": bool(base.get("components") == 1),
            "inside_roi": inside_roi,
            "outside_protected_shell": outside_shell,
            "roi_clearance_min": clearance_min,
            "protected_shell_depth": float(protected_shell_depth),
        }
    )
    result["passed"] = bool(
        finite
        and valid_faces
        and nondegenerate
        and result["watertight"]
        and result["manifold"]
        and result["single_component"]
        and inside_roi
        and outside_shell
    )
    return result


def classify_se_artifact(metadata: Mapping[str, object]) -> str:
    """Classify old results without silently promoting derivatives to raw SE."""
    method = str(metadata.get("method", "")).lower()
    policy = str(metadata.get("policy", "")).lower()
    implementation = str(metadata.get("implementation_kind", "")).lower()
    calibration = str(metadata.get("calibration", "")).lower()
    calibration_policy = str(metadata.get("calibration_policy", "")).lower()
    calibrated = bool(
        "calibrat" in method
        or "calibrat" in policy
        or "calibrat" in calibration
        or calibration_policy == "stage1_median_unweighted_l1_v1"
    )
    valid_zero = bool(
        "stabilized derivative" in implementation
        or "valid-zero" in method
        or "validzero" in policy
        or "valid_projection" in policy
    )
    raw = str(metadata.get("method", "")) == RAW_METHOD_NAME
    if calibrated and (valid_zero or not raw):
        return "unknown Sugavanam--Ertin artifact"
    if valid_zero and raw:
        return "unknown Sugavanam--Ertin artifact"
    if raw and policy and not calibrated:
        return "unknown Sugavanam--Ertin artifact"
    if calibrated:
        return "calibrated derivative"
    if valid_zero:
        if method == A320_METHOD_NAME.lower():
            return "A320 valid-zero stabilized derivative"
        return "valid-zero stabilized derivative"
    if raw:
        return "raw reproduction"
    return "unknown Sugavanam--Ertin artifact"


__all__ = [
    "A320_ARTIFACT_IDENTITY",
    "A320_IMPLEMENTATION_KIND",
    "A320_MANAGER_IDENTITY",
    "A320_METHOD_NAME",
    "A320_OUTPUT_DIR",
    "A320_POLICY",
    "A320_STAGE1_CHECKPOINT",
    "PROJECTION_ITERATIONS",
    "PROJECTION_MIN_ACCEPTANCE",
    "PROJECTION_TOLERANCE",
    "ProjectionAcceptanceError",
    "analytic_sphere_sdf",
    "classify_se_artifact",
    "closed_anchor_losses",
    "closed_field_spec",
    "deterministic_boundary_shell",
    "deterministic_inner_anchors",
    "evaluate_field_grid",
    "field_validity_from_array",
    "orient_normals_outward",
    "oriented_normal_loss",
    "project_to_zero_level_strict",
    "protected_shell_gate",
    "ramped_weight",
    "refresh_iso_points_strict",
    "sample_roi",
    "signed_offset_loss",
    "signed_offset_samples",
    "strict_field_gate",
    "topology_contract",
]
