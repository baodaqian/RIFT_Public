#!/usr/bin/env python
"""CPU contract gates for the isolated Sugavanam--Ertin SE2 implementation."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rift.sugavanam_ertin_validzero import (  # noqa: E402
    SE2_ARTIFACT_IDENTITY,
    SE2_MANAGER_IDENTITY,
    SE2_METHOD_NAME,
    SE2_POLICY,
    ProjectionAcceptanceError,
    analytic_sphere_sdf,
    closed_anchor_losses,
    closed_field_spec,
    field_validity_from_array,
    field_validity_gate,
    mesh_topology_audit,
    project_to_zero_level_valid,
    ramped_weight,
    refresh_iso_points_valid,
    sample_boundary,
    sample_inner,
)


def gate(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"FAIL {name}: {detail}")
    print(f"PASS {name}{': ' + detail if detail else ''}")


class SphereSDF(torch.nn.Module):
    def __init__(self, radius: float = 0.06) -> None:
        super().__init__()
        self.radius = radius

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return xyz.norm(dim=-1) - self.radius


class NegativeField(torch.nn.Module):
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return xyz[:, 0] * 0.0 - 0.02


def sphere_cloud(count: int = 512, radius: float = 0.06) -> np.ndarray:
    rng = np.random.default_rng(4)
    direction = rng.normal(size=(count, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return (radius * direction).astype(np.float32)


def main() -> None:
    torch.manual_seed(4)
    np.random.seed(4)
    device = torch.device("cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(4)
    points = sphere_cloud()
    spec = closed_field_spec(points, extent=0.15, pitch=0.00625)

    gate(
        "new isolated identity",
        SE2_MANAGER_IDENTITY.endswith("_v2")
        and SE2_ARTIFACT_IDENTITY.endswith("_v2")
        and "valid-zero" in SE2_METHOD_NAME
        and SE2_POLICY.endswith("_v2"),
    )
    gate(
        "Stage-1-only closed specification",
        np.linalg.norm(np.asarray(spec.center)) < 0.01 and 0.04 < spec.radius < 0.08,
        f"center={spec.center} radius={spec.radius:.5g}",
    )

    xyz = torch.as_tensor(points[:32])
    target = analytic_sphere_sdf(xyz, spec)
    gate("analytic initialization target", float(target.abs().mean()) < 0.01)

    boundary = sample_boundary(512, 0.15, generator, device)
    on_face = torch.isclose(boundary.abs(), torch.tensor(0.15), atol=1e-7).any(dim=-1)
    gate("exact boundary sampling", bool(on_face.all()))
    inner = sample_inner(128, spec, generator, device)
    gate(
        "inner anchors stay inside initialization",
        bool((analytic_sphere_sdf(inner, spec) < -0.5 * spec.radius).all()),
    )

    sphere = SphereSDF()
    boundary_loss, inner_loss = closed_anchor_losses(sphere, boundary, inner, margin=1e-4)
    gate(
        "closed sign anchors are satisfiable",
        float(boundary_loss) == 0.0 and float(inner_loss) == 0.0,
    )

    valid = field_validity_gate(sphere, 0.15, 24, 1e-4, device, chunk=2048)
    invalid = field_validity_gate(NegativeField(), 0.15, 16, 1e-4, device, chunk=2048)
    gate(
        "strict field validity gate",
        valid["passed"] and valid["exterior_positive"] and not invalid["passed"],
        f"sphere={valid} negative={invalid}",
    )

    q_direction = torch.randn(512, 3)
    q_direction = q_direction / q_direction.norm(dim=-1, keepdim=True)
    q_radius = 0.02 + 0.10 * torch.rand(512, 1)
    q0 = q_direction * q_radius
    projected = project_to_zero_level_valid(
        sphere, q0, 0.15, max_step=0.03, iterations=12, tolerance=1e-4
    )
    gate(
        "valid Newton projections only",
        projected.accepted > 450
        and projected.boundary_clamped == 0
        and projected.residual_max <= 1e-4
        and 2 <= len(projected.accepted_by_iteration) <= 13
        and list(projected.accepted_by_iteration) == sorted(projected.accepted_by_iteration)
        and projected.accepted_by_iteration[-1] == projected.accepted
        and np.isfinite(projected.finite_residual_q95)
        and np.isfinite(projected.gradient_norm_q50),
        str(projected.as_dict()),
    )
    rejected = project_to_zero_level_valid(
        NegativeField(), q0[:64], 0.15, max_step=0.03, iterations=4, tolerance=1e-4
    )
    gate("unconverged projections are rejected", rejected.accepted == 0)

    failed_audit = None
    try:
        refresh_iso_points_valid(
            NegativeField(),
            torch.as_tensor(points),
            extent=0.15,
            pitch=0.00625,
            n_points=64,
            generator=generator,
            tolerance=1e-4,
            iterations=4,
            min_acceptance=0.5,
            oversample=2.0,
        )
    except ProjectionAcceptanceError as error:
        failed_audit = error.audit
    gate(
        "failed projection carries diagnostics",
        failed_audit is not None
        and failed_audit["status"] == "failed"
        and failed_audit["failure_stage"] == "initial_projection"
        and "accepted_by_iteration" in failed_audit["first_projection"],
        str(failed_audit),
    )

    seeds = torch.as_tensor(points)
    iso, audit = refresh_iso_points_valid(
        sphere,
        seeds,
        extent=0.15,
        pitch=0.00625,
        n_points=128,
        generator=generator,
        tolerance=1e-4,
        iterations=12,
        min_acceptance=0.5,
        oversample=2.0,
    )
    gate(
        "audited iso refresh",
        len(iso) >= 128
        and audit["policy_acceptance_fraction"] >= 0.5
        and float(sphere(iso).abs().max()) <= 1e-4,
        f"N={len(iso)} audit={audit}",
    )

    gate(
        "monotone off-surface ramp",
        ramped_weight(100, 1.0, 100, 200) == 0.0
        and np.isclose(ramped_weight(200, 1.0, 100, 200), 0.5)
        and ramped_weight(400, 1.0, 100, 200) == 1.0,
    )

    from skimage.measure import marching_cubes

    axis = np.linspace(-0.15, 0.15, 32, dtype=np.float32)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    field = np.sqrt(x * x + y * y + z * z) - 0.06
    array_gate = field_validity_from_array(field, 1e-4)
    vertices, faces, _, _ = marching_cubes(field, 0.0, spacing=(axis[1] - axis[0],) * 3)
    topology = mesh_topology_audit(vertices, faces)
    gate(
        "closed exported topology",
        array_gate["passed"]
        and topology["passed"]
        and topology["boundary_edges"] == 0
        and topology["nonmanifold_edges"] == 0,
        str(topology),
    )

    forbidden = {"stl", "mesh_truth", "lidar", "ground_truth", "validation_geometry"}
    names = (
        set(closed_field_spec.__code__.co_varnames)
        | set(project_to_zero_level_valid.__code__.co_varnames)
        | set(refresh_iso_points_valid.__code__.co_varnames)
    )
    gate("geometry-truth exclusion", forbidden.isdisjoint(names), str(sorted(names & forbidden)))

    failed_cloud = False
    try:
        closed_field_spec(np.asarray([[0.15, 0.0, 0.0]] * 3), 0.15, 0.00625)
    except ValueError:
        failed_cloud = True
    gate("boundary-contaminated Stage-1 cloud fails closed", failed_cloud)

    print("PASS 15/15 Sugavanam--Ertin SE2 CPU contract gates")


if __name__ == "__main__":
    main()
