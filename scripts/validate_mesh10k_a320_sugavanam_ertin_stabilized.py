#!/usr/bin/env python
"""CPU contract gates for the isolated A320 stabilized SE smoke package."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import rift.sugavanam_ertin_a320_stabilized as a320  # noqa: E402
from rift import sugavanam_ertin_stabilized_smoke_runtime as trainer  # noqa: E402
from rift.sugavanam_ertin import METHOD_NAME as RAW_METHOD_NAME  # noqa: E402
from rift.sugavanam_ertin_validzero import (  # noqa: E402
    SE2_METHOD_NAME,
    analytic_sphere_sdf,
    closed_field_spec,
)


PASSED = 0
EXPECTED_BASE_GATES = 45  # Updated deliberately when the fixed contract suite changes.


def gate(name: str, condition: bool, detail: str = "") -> None:
    global PASSED
    if not condition:
        raise AssertionError(f"FAIL {name}: {detail}")
    PASSED += 1
    print(f"PASS {name}{': ' + detail if detail else ''}")


def must_raise(name: str, exception, callback) -> None:
    caught = None
    try:
        callback()
    except exception as error:
        caught = error
    gate(name, caught is not None, "expected a fail-closed exception")


class SphereSDF(torch.nn.Module):
    def __init__(self, radius: float = 0.06) -> None:
        super().__init__()
        self.radius = radius

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return xyz.norm(dim=-1) - self.radius


class ConstantField(torch.nn.Module):
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return xyz[:, 0] * 0.0 + 0.02


class NanField(torch.nn.Module):
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return xyz[:, 0] * float("nan")


class BoundaryPlane(torch.nn.Module):
    def __init__(self, extent: float) -> None:
        super().__init__()
        self.extent = extent

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.extent - xyz[:, 0]


class SecondLayerDefect(torch.nn.Module):
    def __init__(self, extent: float, pitch: float) -> None:
        super().__init__()
        self.level = extent - 1.5 * pitch

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return xyz.abs().amax(dim=-1) - self.level


def sphere_cloud(count: int = 768, radius: float = 0.06) -> tuple[np.ndarray, np.ndarray]:
    index = np.arange(count, dtype=np.float64)
    z = 1.0 - 2.0 * (index + 0.5) / count
    theta = index * (np.pi * (3.0 - np.sqrt(5.0)))
    xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    normal = np.stack((xy * np.cos(theta), xy * np.sin(theta), z), axis=1)
    return (radius * normal).astype(np.float32), normal.astype(np.float32)


def projection_copy_with_zero_acceptance(result: a320.ProjectionResult) -> a320.ProjectionResult:
    return a320.ProjectionResult(
        points=result.points[:0],
        residuals=result.residuals[:0],
        attempted=result.attempted,
        accepted=0,
        rejected=result.attempted,
        boundary_clamped=result.boundary_clamped,
        outside_domain=result.outside_domain,
        nonfinite=result.nonfinite,
        degenerate_gradient=result.degenerate_gradient,
        residual_max=float("inf"),
        residual_q50=float("inf"),
        residual_q95=float("inf"),
        residual_q99=float("inf"),
        gradient_norm_min=result.gradient_norm_min,
        gradient_norm_q05=result.gradient_norm_q05,
        accepted_by_iteration=tuple(0 for _ in result.accepted_by_iteration),
    )


def validate_stage1() -> None:
    from rift.sugavanam_ertin import load_scattering_cloud

    try:
        state = torch.load(a320.A320_STAGE1_CHECKPOINT, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(a320.A320_STAGE1_CHECKPOINT, map_location="cpu")
    weights = state.get("model_state_dict") or {}
    real = weights.get("w_re")
    imag = weights.get("w_im")
    active = weights.get("active_mask")
    positions = weights.get("grid_positions")
    cloud = load_scattering_cloud(
        a320.A320_STAGE1_CHECKPOINT,
        threshold_fraction=0.15,
        max_points=20000,
    )
    condition = (
        state.get("scene_repr") == "grid"
        and int(state.get("epoch", -1)) == 150
        and np.isclose(float(state.get("extent", -1.0)), 0.15)
        and int(state.get("granularity", -1)) == 48
        and isinstance(real, torch.Tensor)
        and isinstance(imag, torch.Tensor)
        and tuple(real.shape) == (48, 48, 48)
        and tuple(imag.shape) == (48, 48, 48)
        and bool(torch.isfinite(real).all())
        and bool(torch.isfinite(imag).all())
        and isinstance(active, torch.Tensor)
        and tuple(active.shape) == (48, 48, 48)
        and active.dtype == torch.bool
        and isinstance(positions, torch.Tensor)
        and positions.shape[-1] == 3
        and positions.numel() == 48 * 48 * 48 * 3
        and bool(torch.isfinite(positions).all())
        and len(cloud.points) == 963
        and np.isclose(cloud.threshold, 0.1031300873, rtol=0.0, atol=1.0e-6)
    )
    gate("exact Stage-1 final metadata", condition, a320.A320_STAGE1_CHECKPOINT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-stage1", action="store_true")
    args = parser.parse_args()
    device = torch.device("cpu")
    torch.manual_seed(7)
    np.random.seed(7)

    gate(
        "isolated stabilized identity",
        a320.A320_MANAGER_IDENTITY == "rift_mesh10k_a320_se_validzero_smoke_v1"
        and a320.A320_ARTIFACT_IDENTITY == "a320_sugavanam_ertin_validzero_smoke_v1"
        and "A320 valid-zero stabilized derivative" in a320.A320_METHOD_NAME
        and a320.A320_IMPLEMENTATION_KIND.startswith("stabilized derivative"),
    )
    gate(
        "sealed disjoint paths",
        a320.A320_STAGE1_CHECKPOINT.endswith("/a320/se/scatter/checkpoint_final.pth.tar")
        and "/a320/se/sdf/" not in a320.A320_STAGE1_CHECKPOINT
        and a320.A320_OUTPUT_DIR
        == "/storage/scratch1/1/dbao31/rift_mesh10k_a320_se_validzero_smoke_v1"
        and not a320.A320_OUTPUT_DIR.startswith(os.path.dirname(a320.A320_STAGE1_CHECKPOINT)),
    )
    gate(
        "strict projection constants",
        a320.PROJECTION_ITERATIONS == 24
        and a320.PROJECTION_TOLERANCE == 1.0e-4
        and a320.PROJECTION_MIN_ACCEPTANCE == 0.60,
    )

    points, true_normals = sphere_cloud()
    rng = np.random.default_rng(8)
    random_signs = rng.choice(np.asarray([-1.0, 1.0]), size=(len(points), 1))
    oriented, normal_audit = a320.orient_normals_outward(
        points, true_normals * random_signs, (0.0, 0.0, 0.0)
    )
    alignment = np.einsum("ij,ij->i", oriented, true_normals)
    gate(
        "deterministic locally consistent orientation",
        normal_audit["tree_consistent"]
        and normal_audit["components"] == 1
        and np.allclose(np.linalg.norm(oriented, axis=1), 1.0, atol=1.0e-5)
        and float(alignment.mean()) > 0.99,
        str(normal_audit),
    )
    oriented_again, audit_again = a320.orient_normals_outward(
        points, true_normals * random_signs, (0.0, 0.0, 0.0)
    )
    gate(
        "normal orientation reproducible",
        np.array_equal(oriented, oriented_again) and normal_audit == audit_again,
    )
    must_raise(
        "nonunit normals fail closed",
        ValueError,
        lambda: a320.orient_normals_outward(points, 2.0 * true_normals, (0.0, 0.0, 0.0)),
    )
    bad_nan = true_normals.copy()
    bad_nan[0, 0] = np.nan
    must_raise(
        "nonfinite normals fail closed",
        ValueError,
        lambda: a320.orient_normals_outward(points, bad_nan, (0.0, 0.0, 0.0)),
    )
    must_raise(
        "nonfinite orientation center fails closed",
        ValueError,
        lambda: a320.orient_normals_outward(points, true_normals, (0.0, np.nan, 0.0)),
    )
    bad_zero = true_normals.copy()
    bad_zero[0] = 0.0
    must_raise(
        "zero normals fail closed",
        ValueError,
        lambda: a320.orient_normals_outward(points, bad_zero, (0.0, 0.0, 0.0)),
    )

    offset = 0.00625
    signed_points, signed_targets, signed_audit = a320.signed_offset_samples(
        points, oriented, 0.15, offset
    )
    signed_prediction = SphereSDF()(torch.as_tensor(signed_points))
    gate(
        "signed offsets encode exact exterior/interior signs",
        signed_audit["paired_fraction"] == 1.0
        and np.all(signed_targets[: len(points)] > 0.0)
        and np.all(signed_targets[len(points) :] < 0.0)
        and torch.allclose(signed_prediction, torch.as_tensor(signed_targets), atol=2.0e-6),
        str(signed_audit),
    )
    good_loss = a320.oriented_normal_loss(
        torch.as_tensor(true_normals), torch.as_tensor(true_normals)
    )
    reversed_loss = a320.oriented_normal_loss(
        torch.as_tensor(true_normals), -torch.as_tensor(true_normals)
    )
    gate(
        "normal loss preserves orientation",
        float(good_loss) < 1.0e-6 and float(reversed_loss) > 1.9,
    )

    spec = closed_field_spec(points, extent=0.15, pitch=offset)
    shell = a320.deterministic_boundary_shell(0.15, offset, 17, device)
    inner = a320.deterministic_inner_anchors(spec, 256, device)
    gate(
        "deterministic protected two-voxel shell",
        tuple(shell.shape) == (3 * 6 * 17 * 17, 3)
        and torch.isclose(shell.abs().amax(dim=1), torch.tensor(0.15)).any()
        and torch.isclose(shell.abs().amax(dim=1), torch.tensor(0.15 - 2 * offset)).any(),
    )
    sphere = SphereSDF()
    boundary_loss, inner_loss = a320.closed_anchor_losses(
        sphere, shell, inner, margin=1.0e-4
    )
    gate(
        "deterministic sign anchors are satisfiable",
        float(boundary_loss) == 0.0 and float(inner_loss) == 0.0,
    )
    sphere_gate = a320.strict_field_gate(
        sphere, 0.15, 24, 1.0e-4, shell, device, chunk=4096
    )
    defect_gate = a320.strict_field_gate(
        SecondLayerDefect(0.15, offset), 0.15, 24, 1.0e-4, shell, device, chunk=4096
    )
    gate(
        "protected shell catches second-layer sign loss",
        sphere_gate["passed"]
        and defect_gate["strict_crossing"]
        and defect_gate["exterior_positive"]
        and not defect_gate["protected_shell"]["passed"]
        and not defect_gate["passed"],
        str(defect_gate),
    )

    direction = torch.randn(512, 3)
    direction /= direction.norm(dim=-1, keepdim=True)
    seeds = direction * (0.02 + 0.09 * torch.rand(512, 1))
    projected = a320.project_to_zero_level_strict(sphere, seeds, 0.15, 2.0 * offset)
    gate(
        "strict 24-step projection acceptance",
        projected.accepted > 500
        and len(projected.accepted_by_iteration) == 25
        and projected.residual_max <= 1.0e-4
        and projected.boundary_clamped == 0
        and projected.outside_domain == 0
        and projected.nonfinite == 0
        and projected.degenerate_gradient == 0,
        str(projected.as_dict()),
    )
    constant = a320.project_to_zero_level_strict(
        ConstantField(), seeds[:32], 0.15, 2.0 * offset
    )
    gate(
        "degenerate-gradient projections rejected",
        constant.accepted == 0
        and constant.rejected == 32
        and constant.degenerate_gradient == 32,
        str(constant.as_dict()),
    )
    nan_result = a320.project_to_zero_level_strict(
        NanField(), seeds[:32], 0.15, 2.0 * offset
    )
    gate(
        "nonfinite projections rejected",
        nan_result.accepted == 0 and nan_result.nonfinite == 32,
        str(nan_result.as_dict()),
    )
    outside = a320.project_to_zero_level_strict(
        sphere, torch.full((16, 3), 0.16), 0.15, 2.0 * offset
    )
    gate(
        "initially outside projections rejected",
        outside.accepted == 0 and outside.boundary_clamped == 16,
        str(outside.as_dict()),
    )
    boundary_seed = torch.zeros(16, 3)
    boundary_seed[:, 0] = 0.14
    clamped = a320.project_to_zero_level_strict(
        BoundaryPlane(0.15), boundary_seed, 0.15, 2.0 * offset
    )
    gate(
        "boundary-clamped roots rejected",
        clamped.accepted == 0 and clamped.boundary_clamped == 16,
        str(clamped.as_dict()),
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(11)
    iso, refresh = a320.refresh_iso_points_strict(
        sphere, torch.as_tensor(points), 0.15, offset, 128, generator
    )
    gate(
        "both refresh stages independently pass 60 percent",
        len(iso) == 128
        and refresh["first_acceptance_fraction"] >= 0.60
        and refresh["second_acceptance_fraction"] >= 0.60
        and refresh["policy_acceptance_fraction"] >= 0.60
        and float(sphere(iso).abs().max()) <= 1.0e-4,
        str(refresh),
    )
    first_failure = None
    generator.manual_seed(12)
    try:
        a320.refresh_iso_points_strict(
            ConstantField(), torch.as_tensor(points), 0.15, offset, 64, generator
        )
    except a320.ProjectionAcceptanceError as error:
        first_failure = error.audit
    ledger = {
        "attempted",
        "accepted",
        "rejected",
        "outside_roi",
        "boundary_clamped",
        "nonfinite",
        "degenerate_gradient",
    }
    gate(
        "first refresh failure has complete ledger",
        first_failure is not None
        and first_failure.get("failure_stage") == "initial_projection"
        and ledger.issubset(first_failure),
        str(first_failure),
    )
    calls = 0
    real_project = a320.project_to_zero_level_strict

    def fail_second(*call_args, **call_kwargs):
        nonlocal calls
        calls += 1
        result = real_project(*call_args, **call_kwargs)
        return result if calls == 1 else projection_copy_with_zero_acceptance(result)

    second_failure = None
    generator.manual_seed(13)
    with mock.patch.object(a320, "project_to_zero_level_strict", side_effect=fail_second):
        try:
            a320.refresh_iso_points_strict(
                sphere, torch.as_tensor(points), 0.15, offset, 64, generator
            )
        except a320.ProjectionAcceptanceError as error:
            second_failure = error.audit
    gate(
        "second refresh stage independently enforces 60 percent",
        second_failure is not None
        and second_failure.get("failure_stage") == "uniformized_projection"
        and second_failure.get("second_acceptance_fraction") == 0.0
        and ledger.issubset(second_failure),
        str(second_failure),
    )

    from skimage.measure import marching_cubes

    axis = np.linspace(-0.15, 0.15, 32, dtype=np.float32)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    field = np.sqrt(x * x + y * y + z * z) - 0.06
    vertices, faces, _, _ = marching_cubes(
        field, level=0.0, spacing=(axis[1] - axis[0],) * 3
    )
    vertices += -0.15
    faces = faces.astype(np.int32)
    valid_topology = a320.topology_contract(vertices, faces, 0.15, 2.0 * offset)
    gate("valid sphere topology", valid_topology["passed"], str(valid_topology))
    two_vertices = np.concatenate((vertices, vertices), axis=0)
    two_faces = np.concatenate((faces, faces + len(vertices)), axis=0)
    gate(
        "two components rejected",
        not a320.topology_contract(two_vertices, two_faces, 0.15, 2.0 * offset)["passed"],
    )
    clipped = vertices.copy()
    clipped[:, 0] += 0.15 - 1.5 * offset - clipped[:, 0].max()
    clipped_audit = a320.topology_contract(clipped, faces, 0.15, 2.0 * offset)
    gate(
        "protected-shell mesh rejected",
        clipped_audit["inside_roi"]
        and not clipped_audit["outside_protected_shell"]
        and not clipped_audit["passed"],
        str(clipped_audit),
    )
    triangle_vertices = np.asarray([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]], np.float32)
    triangle_faces = np.asarray([[0, 1, 2]], np.int32)
    gate(
        "open triangle rejected",
        not a320.topology_contract(triangle_vertices, triangle_faces, 0.15)["passed"],
    )
    nonmanifold_vertices = np.asarray(
        [[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, -0.01, 0], [0, 0, 0.01]],
        np.float32,
    )
    nonmanifold_faces = np.asarray([[0, 1, 2], [1, 0, 3], [0, 1, 4]], np.int32)
    gate(
        "nonmanifold edge rejected",
        not a320.topology_contract(nonmanifold_vertices, nonmanifold_faces, 0.15)["passed"],
    )
    degenerate_faces = faces.copy()
    degenerate_faces[0, 2] = degenerate_faces[0, 1]
    gate(
        "degenerate face rejected",
        not a320.topology_contract(vertices, degenerate_faces, 0.15)["passed"],
    )
    invalid_cases = [
        faces.astype(np.float32),
        np.asarray([[0, 1, len(vertices)]], np.int32),
        np.asarray([[0.0, 1.0, np.nan]], np.float32),
    ]
    invalid_passes = []
    for invalid in invalid_cases:
        invalid_passes.append(a320.topology_contract(vertices, invalid, 0.15)["passed"])
    gate("invalid face arrays fail without raising", not any(invalid_passes))

    classification = {
        "raw": a320.classify_se_artifact({"method": RAW_METHOD_NAME}),
        "calibrated": a320.classify_se_artifact(
            {"method": RAW_METHOD_NAME, "calibration_policy": "stage1_median_unweighted_l1_v1"}
        ),
        "b787_validzero": a320.classify_se_artifact(
            {"method": SE2_METHOD_NAME, "policy": "closed_sphere_init_sign_anchors_valid_projection_v2"}
        ),
        "a320_validzero": a320.classify_se_artifact(
            {"method": a320.A320_METHOD_NAME, "implementation_kind": a320.A320_IMPLEMENTATION_KIND}
        ),
        "unknown": a320.classify_se_artifact({"method": "something else"}),
        "unknown_policy": a320.classify_se_artifact(
            {"method": RAW_METHOD_NAME, "policy": "unrecognized_policy"}
        ),
    }
    gate(
        "raw calibrated valid-zero labels stay distinct",
        classification
        == {
            "raw": "raw reproduction",
            "calibrated": "calibrated derivative",
            "b787_validzero": "valid-zero stabilized derivative",
            "a320_validzero": "A320 valid-zero stabilized derivative",
            "unknown": "unknown Sugavanam--Ertin artifact",
            "unknown_policy": "unknown Sugavanam--Ertin artifact",
        },
        str(classification),
    )
    gate(
        "mixed derivative metadata fails classification",
        a320.classify_se_artifact(
            {
                "method": a320.A320_METHOD_NAME,
                "policy": a320.A320_POLICY,
                "calibration_policy": "stage1_median_unweighted_l1_v1",
            }
        )
        == "unknown Sugavanam--Ertin artifact",
    )

    trainer_args = trainer.build_parser().parse_args([])
    trainer.validate_args(trainer_args, require_stage1=False)
    option_names = {action.dest.lower() for action in trainer.build_parser()._actions}
    forbidden = {"stl", "mesh_truth", "lidar", "ground_truth", "validation_geometry"}
    gate("trainer exposes no geometry-truth input", forbidden.isdisjoint(option_names))
    changed = trainer.build_parser().parse_args(["--output-dir", "/tmp/not-the-identity"])
    must_raise(
        "trainer rejects a different output identity",
        ValueError,
        lambda: trainer.validate_args(changed, require_stage1=False),
    )

    original_excepthook = sys.excepthook
    with tempfile.TemporaryDirectory() as temp_root:
        fresh_args = trainer.build_parser().parse_args([])
        try:
            with mock.patch.object(trainer, "A320_OUTPUT_DIR", temp_root):
                trainer.prepare_output(fresh_args)
                gate("empty fresh output root is accepted", Path(temp_root).is_dir())
        finally:
            sys.excepthook = original_excepthook
    with tempfile.TemporaryDirectory() as temp_root:
        (Path(temp_root) / "unexpected.txt").write_text("preserve", encoding="utf-8")
        with mock.patch.object(trainer, "A320_OUTPUT_DIR", temp_root):
            must_raise(
                "nonempty fresh output root is rejected",
                RuntimeError,
                lambda: trainer.prepare_output(trainer.build_parser().parse_args([])),
            )
    with tempfile.TemporaryDirectory() as temp_root:
        (Path(temp_root) / "terminal_failure.json").write_text("{}", encoding="utf-8")
        resume_args = trainer.build_parser().parse_args(
            ["--resume", str(Path(temp_root) / "checkpoint_latest.pth.tar")]
        )
        (Path(temp_root) / "checkpoint_latest.pth.tar").write_bytes(b"preserve")
        with mock.patch.object(trainer, "A320_OUTPUT_DIR", temp_root):
            must_raise(
                "terminal evidence blocks even an explicit resume",
                RuntimeError,
                lambda: trainer.prepare_output(resume_args),
            )
    with tempfile.TemporaryDirectory() as temp_root:
        resume_path = Path(temp_root) / "checkpoint_latest.pth.tar"
        resume_path.write_bytes(b"preserve")
        resume_args = trainer.build_parser().parse_args(["--resume", str(resume_path)])
        try:
            with mock.patch.object(trainer, "A320_OUTPUT_DIR", temp_root):
                trainer.prepare_output(resume_args)
                gate("own latest checkpoint is an eligible resume input", resume_path.is_file())
        finally:
            sys.excepthook = original_excepthook

    class ResumeContractModel:
        @staticmethod
        def config() -> dict:
            return {"contract_probe": True}

    resume_state = {
        "method": a320.A320_METHOD_NAME,
        "manager_identity": a320.A320_MANAGER_IDENTITY,
        "artifact_identity": a320.A320_ARTIFACT_IDENTITY,
        "policy": a320.A320_POLICY,
        "implementation_kind": a320.A320_IMPLEMENTATION_KIND,
        "ground_truth_geometry_used": False,
        "scatter_checkpoint": a320.A320_STAGE1_CHECKPOINT,
        "model_config": ResumeContractModel.config(),
        "args": vars(trainer_args),
        "resume_allowed": False,
    }
    must_raise(
        "checkpoint without clean-interruption marker cannot resume",
        ValueError,
        lambda: trainer.validate_resume_state(resume_state, trainer_args, ResumeContractModel()),
    )
    resume_state["resume_allowed"] = True
    trainer.validate_resume_state(resume_state, trainer_args, ResumeContractModel())
    gate("clean-interruption marker permits identity-matched resume", True)

    project = Path(__file__).resolve().parents[1]
    launcher_text = (
        project / "slurm" / "train_mesh10k_a320_sugavanam_ertin_stabilized_smoke_v1.sbatch"
    ).read_text(encoding="utf-8")
    validation_launcher_text = (
        project / "slurm" / "validate_mesh10k_a320_sugavanam_ertin_stabilized_v1.sbatch"
    ).read_text(encoding="utf-8")
    trainer_text = (project / "rift" / "sugavanam_ertin_stabilized_smoke_runtime.py").read_text(
        encoding="utf-8"
    )
    compare_text = (project / "scripts" / "compare_sugavanam_ertin_geometry.py").read_text(
        encoding="utf-8"
    )
    evaluator_text = (project / "scripts" / "eval_mesh10k_current_geometry.py").read_text(
        encoding="utf-8"
    )
    gate(
        "launcher uses exact source and disjoint root",
        a320.A320_STAGE1_CHECKPOINT in launcher_text
        and a320.A320_OUTPUT_DIR in launcher_text
        and "/a320/se/sdf/" not in launcher_text
        and "train_sugavanam_ertin_smoke.py --recipe legacy-a320-stabilized" in launcher_text
        and 'state.get("resume_allowed") is True' in launcher_text
        and "except TypeError:" in launcher_text,
    )
    gate(
        "CPU validator launcher cannot invoke training",
        "#SBATCH --partition=cpu-small" in validation_launcher_text
        and "validate_mesh10k_a320_sugavanam_ertin_stabilized.py --require-stage1"
        in validation_launcher_text
        and "python -B -u train_mesh10k" not in validation_launcher_text,
    )
    gate(
        "no raw SDF resume path in trainer",
        "/a320/se/sdf/" not in trainer_text
        and "checkpoint_best.pth.tar" not in trainer_text.split("def main", 1)[0]
        and 'state.get("resume_allowed") is not True' in trainer_text,
    )
    gate(
        "comparator fails closed across SE artifact variants",
        "classify_se_artifact" in compare_text
        and "unknown or mixed Sugavanam--Ertin surface metadata" in compare_text
        and "A320 valid-zero surface cannot enter the B787 comparator" in compare_text,
    )
    gate(
        "Stage-1 proxy cannot masquerade as reportable SE geometry",
        '"artifact_variant": "stage1_scattering_grid_proxy"' in evaluator_text
        and '"reportable_method_result": False' in evaluator_text
        and 'row["diagnostic_oracle"] = oracle' in evaluator_text,
    )
    gate(
        "off-surface warm-up is monotone",
        a320.ramped_weight(20, 1.0, 20, 40) == 0.0
        and np.isclose(a320.ramped_weight(40, 1.0, 20, 40), 0.5)
        and a320.ramped_weight(80, 1.0, 20, 40) == 1.0,
    )
    if args.require_stage1:
        validate_stage1()

    expected = EXPECTED_BASE_GATES + int(args.require_stage1)
    if PASSED != expected:
        raise AssertionError(f"FAIL fixed gate count: observed {PASSED}, expected {expected}")
    print(f"PASS {PASSED}/{expected} A320 stabilized SE contract gates")


if __name__ == "__main__":
    main()
