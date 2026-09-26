"""Compatibility runtime for the historical A320 stabilized-SDF smoke.

Use train_sugavanam_ertin_smoke.py --recipe legacy-a320-stabilized. This is not
a second CLI or the shared dataset's default two-stage engineering recipe.

The Stage-1 source and output identity are sealed.  This program cannot read a
raw SDF checkpoint and exposes no ground-truth geometry argument.  It preserves
the original trainer and checkpoint formats by writing only to a new directory.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

import train_sugavanam_ertin_validzero as base
from rift.sugavanam_ertin import (
    PAPER_ID,
    FourierFeatureSDF,
    estimate_pca_normals,
    load_scattering_cloud,
    spatial_gradient,
)
from rift.sugavanam_ertin_a320_stabilized import (
    A320_ARTIFACT_IDENTITY,
    A320_IMPLEMENTATION_KIND,
    A320_MANAGER_IDENTITY,
    A320_METHOD_NAME,
    A320_OUTPUT_DIR,
    A320_POLICY,
    A320_STAGE1_CHECKPOINT,
    PROJECTION_ITERATIONS,
    PROJECTION_MIN_ACCEPTANCE,
    PROJECTION_TOLERANCE,
    ProjectionAcceptanceError,
    analytic_sphere_sdf,
    closed_anchor_losses,
    closed_field_spec,
    deterministic_boundary_shell,
    deterministic_inner_anchors,
    evaluate_field_grid,
    field_validity_from_array,
    orient_normals_outward,
    oriented_normal_loss,
    ramped_weight,
    refresh_iso_points_strict,
    sample_roi,
    signed_offset_loss,
    signed_offset_samples,
    strict_field_gate,
    topology_contract,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=A320_OUTPUT_DIR)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--init-steps", type=int, default=200)
    parser.add_argument("--init-lr", type=float, default=1.0e-3)
    parser.add_argument("--init-batch", type=int, default=2048)
    parser.add_argument("--init-log-every", type=int, default=25)
    parser.add_argument("--batch-on", type=int, default=512)
    parser.add_argument("--batch-off", type=int, default=512)
    parser.add_argument("--batch-iso", type=int, default=256)
    parser.add_argument("--batch-signed", type=int, default=512)
    parser.add_argument("--batch-boundary", type=int, default=256)
    parser.add_argument("--batch-inner", type=int, default=128)
    parser.add_argument("--n-iso", type=int, default=1024)
    parser.add_argument("--iso-start", type=int, default=60)
    parser.add_argument("--iso-refresh", type=int, default=30)
    parser.add_argument("--scatter-threshold", type=float, default=0.15)
    parser.add_argument("--max-scatter-points", type=int, default=20000)
    parser.add_argument("--normal-radius", type=float, default=0.0)
    parser.add_argument("--signed-offset-pitches", type=float, default=1.0)
    parser.add_argument("--n-fourier", type=int, default=9, choices=(6, 9))
    parser.add_argument("--fourier-scale", type=float, default=2.0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--alpha-off", type=float, default=100.0)
    parser.add_argument("--lambda-on", type=float, default=1.0)
    parser.add_argument("--lambda-normal", type=float, default=1.0)
    parser.add_argument("--lambda-signed", type=float, default=1.0)
    parser.add_argument("--lambda-off", type=float, default=1.0)
    parser.add_argument("--lambda-eik", type=float, default=1.0)
    parser.add_argument("--lambda-iso", type=float, default=1.0)
    parser.add_argument("--lambda-iso-normal", type=float, default=1.0)
    parser.add_argument("--lambda-boundary", type=float, default=1.0)
    parser.add_argument("--lambda-inner", type=float, default=1.0)
    parser.add_argument("--off-warmup", type=int, default=20)
    parser.add_argument("--off-ramp", type=int, default=40)
    parser.add_argument("--radius-quantile", type=float, default=0.5)
    parser.add_argument("--radius-cap-fraction", type=float, default=0.65)
    parser.add_argument("--boundary-margin", type=float, default=1.0e-4)
    parser.add_argument("--boundary-shell-resolution", type=int, default=24)
    parser.add_argument("--inner-anchor-count", type=int, default=1024)
    parser.add_argument("--gate-every", type=int, default=10)
    parser.add_argument("--gate-grid", type=int, default=32)
    parser.add_argument("--grid-chunk", type=int, default=32768)
    parser.add_argument("--projection-oversample", type=float, default=2.0)
    parser.add_argument("--mesh-grid", type=int, default=64)
    parser.add_argument("--mesh-points", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def validate_args(args: argparse.Namespace, require_stage1: bool = True) -> None:
    if os.path.realpath(args.output_dir) != os.path.realpath(A320_OUTPUT_DIR):
        raise ValueError("A320 stabilized smoke output identity is sealed")
    if require_stage1 and not os.path.isfile(A320_STAGE1_CHECKPOINT):
        raise FileNotFoundError(A320_STAGE1_CHECKPOINT)
    if args.resume:
        expected = os.path.join(A320_OUTPUT_DIR, "checkpoint_latest.pth.tar")
        if os.path.realpath(args.resume) != os.path.realpath(expected):
            raise ValueError("resume must be this smoke identity's checkpoint_latest.pth.tar")
    if args.steps < args.iso_start or args.iso_start <= 0 or args.iso_refresh <= 0:
        raise ValueError("smoke must contain at least one positive iso refresh")
    positive_counts = (
        args.init_steps,
        args.init_batch,
        args.batch_on,
        args.batch_off,
        args.batch_iso,
        args.batch_signed,
        args.batch_boundary,
        args.batch_inner,
        args.n_iso,
        args.gate_every,
        args.gate_grid,
        args.mesh_grid,
        args.mesh_points,
        args.boundary_shell_resolution,
        args.inner_anchor_count,
    )
    if min(positive_counts) <= 0 or args.gate_grid < 8 or args.mesh_grid < 8:
        raise ValueError("all counts must be positive and grids must be at least 8")
    if args.boundary_shell_resolution < 3 or args.inner_anchor_count < 3:
        raise ValueError("deterministic anchor sets are too small")
    if args.boundary_margin <= 0.0 or args.signed_offset_pitches <= 0.0:
        raise ValueError("boundary margin and signed offset must be positive")
    if args.projection_oversample < 1.0:
        raise ValueError("projection oversampling must be at least one")
    if args.off_warmup < 0 or args.off_ramp < 0:
        raise ValueError("off-surface schedule must be nonnegative")
    sealed = {
        "steps": 120,
        "init_steps": 200,
        "init_lr": 1.0e-3,
        "init_batch": 2048,
        "batch_on": 512,
        "batch_off": 512,
        "batch_iso": 256,
        "batch_signed": 512,
        "batch_boundary": 256,
        "batch_inner": 128,
        "scatter_threshold": 0.15,
        "max_scatter_points": 20000,
        "normal_radius": 0.0,
        "signed_offset_pitches": 1.0,
        "n_fourier": 9,
        "fourier_scale": 2.0,
        "hidden_dim": 512,
        "n_layers": 8,
        "lr": 1.0e-4,
        "alpha_off": 100.0,
        "lambda_on": 1.0,
        "lambda_normal": 1.0,
        "lambda_signed": 1.0,
        "lambda_off": 1.0,
        "lambda_eik": 1.0,
        "lambda_iso": 1.0,
        "lambda_iso_normal": 1.0,
        "lambda_boundary": 1.0,
        "lambda_inner": 1.0,
        "off_warmup": 20,
        "off_ramp": 40,
        "iso_start": 60,
        "iso_refresh": 30,
        "n_iso": 1024,
        "radius_quantile": 0.5,
        "radius_cap_fraction": 0.65,
        "boundary_margin": 1.0e-4,
        "boundary_shell_resolution": 24,
        "inner_anchor_count": 1024,
        "gate_every": 10,
        "gate_grid": 32,
        "grid_chunk": 32768,
        "projection_oversample": 2.0,
        "mesh_grid": 64,
        "mesh_points": 5000,
        "seed": 42,
    }
    for key, required in sealed.items():
        if not _same_value(getattr(args, key), required):
            raise ValueError(f"A320 stabilized smoke argument is sealed: {key}={required}")


def prepare_output(args: argparse.Namespace) -> None:
    """Refuse ambiguous reuse and retain uncaught failure evidence."""
    terminal_markers = (
        "terminal_failure.json",
        "interruption_without_checkpoint.json",
        "checkpoint_projection_failure.pth.tar",
        "checkpoint_final.pth.tar",
        "surface_reconstruction.npz",
        "run_summary.json",
    )
    present_markers = [
        name for name in terminal_markers if os.path.exists(os.path.join(A320_OUTPUT_DIR, name))
    ]
    if present_markers:
        raise RuntimeError(f"A320 stabilized output has terminal evidence: {present_markers}")
    existing = os.listdir(A320_OUTPUT_DIR) if os.path.isdir(A320_OUTPUT_DIR) else []
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(args.resume)
    elif existing:
        raise RuntimeError("fresh A320 stabilized smoke refuses a nonempty output identity")
    os.makedirs(A320_OUTPUT_DIR, exist_ok=True)

    previous_hook = sys.excepthook

    def retain_failure(error_type, error, traceback) -> None:
        clean_without_checkpoint = bool(base._STOP_REQUESTED)
        name = (
            "interruption_without_checkpoint.json"
            if clean_without_checkpoint
            else "terminal_failure.json"
        )
        path = os.path.join(A320_OUTPUT_DIR, name)
        if not os.path.exists(path):
            try:
                base.atomic_json_dump(
                    {
                        "status": (
                            "clean_interruption_without_resume_checkpoint"
                            if clean_without_checkpoint
                            else "terminal_failure"
                        ),
                        "error_type": error_type.__name__,
                        "error": str(error),
                        "resume_allowed": False,
                    },
                    path,
                )
            except Exception:
                pass
        previous_hook(error_type, error, traceback)

    sys.excepthook = retain_failure


def fit_closed_initialization(
    model: FourierFeatureSDF,
    spec,
    shell: torch.Tensor,
    inner: torch.Tensor,
    args: argparse.Namespace,
    generator: torch.Generator,
    device: torch.device,
) -> Dict[str, object]:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    final_loss = float("inf")
    model.train()
    for init_step in range(1, args.init_steps + 1):
        roi = sample_roi(args.init_batch, spec.extent, generator, device)
        boundary_index = torch.randint(
            len(shell), (max(args.init_batch // 4, 32),), generator=generator, device=device
        )
        inner_index = torch.randint(
            len(inner), (max(args.init_batch // 8, 32),), generator=generator, device=device
        )
        boundary_batch = shell[boundary_index]
        inner_batch = inner[inner_index]
        samples = torch.cat((roi, boundary_batch, inner_batch), dim=0)
        target = analytic_sphere_sdf(samples, spec).clamp(
            -0.9 * spec.extent, 0.9 * spec.extent
        )
        prediction = model(samples)
        fit = F.smooth_l1_loss(prediction, target, beta=max(spec.pitch, 1.0e-6))
        boundary_loss, inner_loss = closed_anchor_losses(
            model, boundary_batch, inner_batch, args.boundary_margin
        )
        total = fit + boundary_loss + inner_loss
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite A320 initialization at step {init_step}")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        final_loss = float(total.detach())
        if init_step == 1 or init_step % args.init_log_every == 0 or init_step == args.init_steps:
            print(
                f"A320 stabilized init [{init_step}/{args.init_steps}] loss={final_loss:.6g} "
                f"fit={float(fit.detach()):.4g} boundary={float(boundary_loss.detach()):.4g} "
                f"inner={float(inner_loss.detach()):.4g}",
                flush=True,
            )
        if base._STOP_REQUESTED:
            raise InterruptedError("stop requested during A320 closed initialization")
    model.eval()
    gate = strict_field_gate(
        model,
        spec.extent,
        args.gate_grid,
        args.boundary_margin,
        shell,
        device,
        args.grid_chunk,
    )
    if not gate["passed"]:
        raise RuntimeError(f"A320 closed initialization failed strict gate: {gate}")
    return {
        "policy": A320_POLICY,
        "steps": int(args.init_steps),
        "learning_rate": float(args.init_lr),
        "batch": int(args.init_batch),
        "final_loss": final_loss,
        "closed_field": spec.as_dict(),
        "initial_gate": gate,
    }


RESUME_ARGUMENTS = (
    "steps",
    "init_steps",
    "init_lr",
    "init_batch",
    "init_log_every",
    "batch_on",
    "batch_off",
    "batch_iso",
    "batch_signed",
    "batch_boundary",
    "batch_inner",
    "n_iso",
    "iso_start",
    "iso_refresh",
    "scatter_threshold",
    "max_scatter_points",
    "normal_radius",
    "signed_offset_pitches",
    "n_fourier",
    "fourier_scale",
    "hidden_dim",
    "n_layers",
    "lr",
    "alpha_off",
    "lambda_on",
    "lambda_normal",
    "lambda_signed",
    "lambda_off",
    "lambda_eik",
    "lambda_iso",
    "lambda_iso_normal",
    "lambda_boundary",
    "lambda_inner",
    "off_warmup",
    "off_ramp",
    "radius_quantile",
    "radius_cap_fraction",
    "boundary_margin",
    "boundary_shell_resolution",
    "inner_anchor_count",
    "gate_every",
    "gate_grid",
    "grid_chunk",
    "projection_oversample",
    "mesh_grid",
    "mesh_points",
    "log_every",
    "save_every",
    "seed",
)


def _same_value(saved: object, current: object) -> bool:
    if isinstance(saved, (int, float)) and isinstance(current, (int, float)):
        return bool(np.isclose(float(saved), float(current), rtol=0.0, atol=1.0e-12))
    return saved == current


def _same_derived_metadata(saved: object, current: object) -> bool:
    """Compare deterministic derived audits while tolerating harmless FP jitter."""
    if isinstance(saved, dict) and isinstance(current, dict):
        return saved.keys() == current.keys() and all(
            _same_derived_metadata(saved[key], current[key]) for key in saved
        )
    if isinstance(saved, (list, tuple)) and isinstance(current, (list, tuple)):
        return len(saved) == len(current) and all(
            _same_derived_metadata(left, right) for left, right in zip(saved, current)
        )
    if (
        isinstance(saved, (int, float, np.integer, np.floating))
        and not isinstance(saved, (bool, np.bool_))
        and isinstance(current, (int, float, np.integer, np.floating))
        and not isinstance(current, (bool, np.bool_))
    ):
        return bool(np.isclose(float(saved), float(current), rtol=1.0e-6, atol=1.0e-9))
    return saved == current


def checkpoint_state(
    args: argparse.Namespace,
    model: FourierFeatureSDF,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    best_loss: float,
    cloud,
    spec,
    initialization: dict,
    normal_audit: dict,
    offset_audit: dict,
    iso_points: Optional[torch.Tensor],
    iso_normals: Optional[torch.Tensor],
    history: list,
    gate_history: list,
    refresh_history: list,
    generator: torch.Generator,
    surface_audit: Optional[dict] = None,
    resume_allowed: bool = False,
) -> dict:
    return {
        "method": A320_METHOD_NAME,
        "paper": PAPER_ID,
        "manager_identity": A320_MANAGER_IDENTITY,
        "artifact_identity": A320_ARTIFACT_IDENTITY,
        "policy": A320_POLICY,
        "implementation_kind": A320_IMPLEMENTATION_KIND,
        "ground_truth_geometry_used": False,
        "novel_view_signal_supported": False,
        "run_kind": "smoke",
        "resume_allowed": bool(resume_allowed),
        "step": int(step),
        "best_loss": float(best_loss),
        "model_config": model.config(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": vars(args),
        "scatter_checkpoint": A320_STAGE1_CHECKPOINT,
        "scatter_source_epoch": int(cloud.source_epoch),
        "scatter_threshold_absolute": float(cloud.threshold),
        "n_scattering_centres": int(len(cloud.points)),
        "extent": float(cloud.extent),
        "granularity": int(cloud.granularity),
        "closed_field": spec.as_dict(),
        "initialization": initialization,
        "normal_orientation_audit": normal_audit,
        "signed_offset_audit": offset_audit,
        "projection_contract": {
            "iterations": PROJECTION_ITERATIONS,
            "tolerance": PROJECTION_TOLERANCE,
            "minimum_acceptance_each_stage": PROJECTION_MIN_ACCEPTANCE,
        },
        "iso_points": None if iso_points is None else iso_points.detach().cpu(),
        "iso_normals": None if iso_normals is None else iso_normals.detach().cpu(),
        "history": history,
        "gate_history": gate_history,
        "iso_refresh_history": refresh_history,
        "surface_audit": surface_audit,
        "sample_rng_state": generator.get_state(),
        "rng_state": {
            "python": __import__("random").getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def validate_resume_state(state: dict, args: argparse.Namespace, model: FourierFeatureSDF) -> None:
    identity = (
        state.get("method") == A320_METHOD_NAME
        and state.get("manager_identity") == A320_MANAGER_IDENTITY
        and state.get("artifact_identity") == A320_ARTIFACT_IDENTITY
        and state.get("policy") == A320_POLICY
    )
    if not identity:
        raise ValueError("resume checkpoint is not the A320 stabilized smoke identity")
    if state.get("implementation_kind") != A320_IMPLEMENTATION_KIND:
        raise ValueError("resume implementation label changed")
    if state.get("ground_truth_geometry_used") is not False:
        raise ValueError("resume violates the no-geometry-truth contract")
    if state.get("scatter_checkpoint") != A320_STAGE1_CHECKPOINT:
        raise ValueError("resume Stage-1 source changed")
    if state.get("model_config") != model.config():
        raise ValueError("resume model configuration changed")
    if state.get("resume_allowed") is not True:
        raise ValueError("resume checkpoint was not written by a clean signal interruption")
    saved = state.get("args") or {}
    for key in RESUME_ARGUMENTS:
        if key not in saved or not _same_value(saved[key], getattr(args, key)):
            raise ValueError(f"resume argument changed: {key}")


def export_surface(
    model: FourierFeatureSDF,
    shell: torch.Tensor,
    args: argparse.Namespace,
    extent: float,
    protected_shell_depth: float,
    device: torch.device,
) -> tuple[str, dict]:
    from skimage.measure import marching_cubes

    field, pitch = evaluate_field_grid(model, extent, args.mesh_grid, device, args.grid_chunk)
    validity = field_validity_from_array(field, args.boundary_margin)
    shell_gate = strict_field_gate(
        model,
        extent,
        args.mesh_grid,
        args.boundary_margin,
        shell,
        device,
        args.grid_chunk,
    )["protected_shell"]
    validity.update(
        {
            "grid": int(args.mesh_grid),
            "pitch": float(pitch),
            "protected_shell": shell_gate,
            "passed": bool(validity["passed"] and shell_gate["passed"]),
        }
    )
    if not validity["passed"]:
        raise RuntimeError(f"A320 final field failed strict crossing/boundary gate: {validity}")
    vertices, faces, vertex_normals, _ = marching_cubes(
        field, level=0.0, spacing=(pitch, pitch, pitch)
    )
    vertices += -extent
    faces = faces.astype(np.int32)
    topology = topology_contract(
        vertices, faces, extent, protected_shell_depth=protected_shell_depth
    )
    if not topology["passed"]:
        raise RuntimeError(f"A320 final mesh failed topology contract: {topology}")
    surface = base.sample_triangles(
        vertices, faces, args.mesh_points, np.random.default_rng(args.seed)
    )
    if not np.isfinite(vertex_normals).all() or not np.isfinite(surface).all():
        raise RuntimeError("A320 exported normals/surface samples are non-finite")
    path = os.path.join(A320_OUTPUT_DIR, "surface_reconstruction.npz")
    base.atomic_npz_save(
        path,
        vertices=vertices.astype(np.float32),
        faces=faces,
        vertex_normals=vertex_normals.astype(np.float32),
        surface_points=surface.astype(np.float32),
        sdf=field.astype(np.float32),
        extent=np.float32(extent),
        pitch=np.float32(pitch),
        manager_identity=np.asarray(A320_MANAGER_IDENTITY),
        artifact_identity=np.asarray(A320_ARTIFACT_IDENTITY),
        method=np.asarray(A320_METHOD_NAME),
        policy=np.asarray(A320_POLICY),
        implementation_kind=np.asarray(A320_IMPLEMENTATION_KIND),
        ground_truth_geometry_used=np.bool_(False),
        validity_json=np.asarray(json.dumps(validity, sort_keys=True)),
        topology_json=np.asarray(json.dumps(topology, sort_keys=True)),
    )
    return path, {"validity": validity, "topology": topology}


def main(argv=None) -> None:
    args = parse_args(argv)
    validate_args(args)
    prepare_output(args)
    signal.signal(signal.SIGTERM, base._request_stop)
    signal.signal(signal.SIGINT, base._request_stop)
    base.set_seed(args.seed)
    device = torch.device(args.device)
    cloud = load_scattering_cloud(
        A320_STAGE1_CHECKPOINT,
        threshold_fraction=args.scatter_threshold,
        max_points=args.max_scatter_points,
        normal_radius=args.normal_radius or None,
    )
    if (
        int(cloud.source_epoch) != 150
        or int(cloud.granularity) != 48
        or not np.isclose(float(cloud.extent), 0.15, rtol=0.0, atol=1.0e-12)
        or int(len(cloud.points)) != 963
        or not np.isclose(float(cloud.threshold), 0.1031300873, rtol=0.0, atol=1.0e-6)
    ):
        raise ValueError(
            "sealed A320 Stage-1 final metadata/cloud no longer matches the audited source"
        )
    pitch = 2.0 * cloud.extent / cloud.granularity
    spec = closed_field_spec(
        cloud.points,
        cloud.extent,
        pitch,
        args.radius_quantile,
        args.radius_cap_fraction,
    )
    oriented_normals, normal_audit = orient_normals_outward(
        cloud.points, cloud.normals, spec.center
    )
    signed_points_np, signed_targets_np, offset_audit = signed_offset_samples(
        cloud.points,
        oriented_normals,
        cloud.extent,
        args.signed_offset_pitches * pitch,
    )
    points = torch.as_tensor(cloud.points, device=device)
    normals = torch.as_tensor(oriented_normals, device=device)
    signed_points = torch.as_tensor(signed_points_np, device=device)
    signed_targets = torch.as_tensor(signed_targets_np, device=device)
    shell = deterministic_boundary_shell(
        cloud.extent, pitch, args.boundary_shell_resolution, device
    )
    inner = deterministic_inner_anchors(spec, args.inner_anchor_count, device)
    print(
        f"{A320_METHOD_NAME}; identity={A320_MANAGER_IDENTITY}; "
        f"Stage-1={A320_STAGE1_CHECKPOINT}; centres={len(points)}; "
        f"signed_pairs={offset_audit['paired_count']}; geometry truth: DISABLED",
        flush=True,
    )

    model = FourierFeatureSDF(
        extent=cloud.extent,
        n_fourier=args.n_fourier,
        fourier_scale=args.fourier_scale,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        seed=args.seed,
    ).to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    start_step = 0
    best_loss = float("inf")
    history: list = []
    gate_history: list = []
    refresh_history: list = []
    iso_points = iso_normals = None

    if args.resume:
        try:
            resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        except TypeError:
            resume = torch.load(args.resume, map_location="cpu")
        validate_resume_state(resume, args, model)
        model.load_state_dict(resume["model_state_dict"])
        initialization = dict(resume["initialization"])
        if not _same_derived_metadata(resume.get("closed_field"), spec.as_dict()):
            raise ValueError("Stage-1-derived closed-field specification changed")
        if not _same_derived_metadata(resume.get("normal_orientation_audit"), normal_audit):
            raise ValueError("Stage-1 normal orientation changed")
        if not _same_derived_metadata(resume.get("signed_offset_audit"), offset_audit):
            raise ValueError("Stage-1 signed-offset construction changed")
    else:
        initialization = fit_closed_initialization(
            model, spec, shell, inner, args, generator, device
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.steps, 1), eta_min=args.lr * 0.01
    )
    if args.resume:
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        start_step = int(resume["step"])
        best_loss = float(resume["best_loss"])
        history = list(resume.get("history", []))
        gate_history = list(resume.get("gate_history", []))
        refresh_history = list(resume.get("iso_refresh_history", []))
        iso_points = None if resume.get("iso_points") is None else resume["iso_points"].to(device)
        iso_normals = None if resume.get("iso_normals") is None else resume["iso_normals"].to(device)
        base.restore_rng(resume.get("rng_state"))
        if resume.get("sample_rng_state") is not None:
            generator.set_state(resume["sample_rng_state"].cpu())
        print(f"Resumed A320 stabilized step {start_step}/{args.steps}", flush=True)
    else:
        gate_history.append(
            {"step": 0, "phase": "post_initialization", **initialization["initial_gate"]}
        )

    history_path = os.path.join(A320_OUTPUT_DIR, "sdf_history.csv")
    gate_path = os.path.join(A320_OUTPUT_DIR, "validity_history.json")
    refresh_path = os.path.join(A320_OUTPUT_DIR, "iso_refresh_history.json")
    started = time.time()
    last_gate = dict(gate_history[-1])

    for step0 in range(start_step, args.steps):
        step = step0 + 1
        refresh_due = step >= args.iso_start and (
            iso_points is None or (step - args.iso_start) % args.iso_refresh == 0
        )
        if refresh_due:
            model.eval()
            pre_gate = strict_field_gate(
                model,
                cloud.extent,
                args.gate_grid,
                args.boundary_margin,
                shell,
                device,
                args.grid_chunk,
            )
            pre_gate.update({"step": step, "phase": "pre_iso_refresh"})
            gate_history.append(pre_gate)
            if not pre_gate["passed"]:
                base.atomic_json_dump({"gate_history": gate_history}, gate_path)
                raise RuntimeError(f"A320 pre-refresh strict gate failed: {pre_gate}")
            try:
                iso_points, refresh = refresh_iso_points_strict(
                    model,
                    points,
                    cloud.extent,
                    pitch,
                    args.n_iso,
                    generator,
                    oversample=args.projection_oversample,
                )
            except ProjectionAcceptanceError as error:
                failure = {"step": step, "pre_gate": pre_gate, **error.audit}
                refresh_history.append(failure)
                base.atomic_json_dump({"gate_history": gate_history}, gate_path)
                base.atomic_json_dump({"iso_refresh_history": refresh_history}, refresh_path)
                failure_state = checkpoint_state(
                    args,
                    model,
                    optimizer,
                    scheduler,
                    step - 1,
                    best_loss,
                    cloud,
                    spec,
                    initialization,
                    normal_audit,
                    offset_audit,
                    iso_points,
                    iso_normals,
                    history,
                    gate_history,
                    refresh_history,
                    generator,
                )
                base.atomic_torch_save(
                    failure_state,
                    os.path.join(A320_OUTPUT_DIR, "checkpoint_projection_failure.pth.tar"),
                )
                raise
            iso_normals_np = estimate_pca_normals(
                iso_points.detach().cpu().numpy(), radius=3.0 * pitch
            )
            iso_normals_np, iso_normal_audit = orient_normals_outward(
                iso_points.detach().cpu().numpy(), iso_normals_np, spec.center
            )
            refresh["iso_normal_orientation"] = iso_normal_audit
            iso_normals = torch.as_tensor(iso_normals_np, device=device)
            post_gate = strict_field_gate(
                model,
                cloud.extent,
                args.gate_grid,
                args.boundary_margin,
                shell,
                device,
                args.grid_chunk,
            )
            post_gate.update({"step": step, "phase": "post_iso_refresh"})
            gate_history.append(post_gate)
            if not post_gate["passed"]:
                base.atomic_json_dump({"gate_history": gate_history}, gate_path)
                raise RuntimeError(f"A320 post-refresh strict gate failed: {post_gate}")
            refresh.update({"step": step, "pre_gate": pre_gate, "post_gate": post_gate})
            refresh_history.append(refresh)
            base.atomic_json_dump({"iso_refresh_history": refresh_history}, refresh_path)
            print(
                f"A320 iso refresh step={step}: first={refresh['first_acceptance_fraction']:.3f} "
                f"second={refresh['second_acceptance_fraction']:.3f}",
                flush=True,
            )
            model.train()

        on_index = torch.randint(
            len(points), (args.batch_on,), generator=generator, device=device
        )
        signed_index = torch.randint(
            len(signed_points), (args.batch_signed,), generator=generator, device=device
        )
        boundary_index = torch.randint(
            len(shell), (args.batch_boundary,), generator=generator, device=device
        )
        inner_index = torch.randint(
            len(inner), (args.batch_inner,), generator=generator, device=device
        )
        p_on = points[on_index].clone()
        n_on = normals[on_index]
        p_off = sample_roi(args.batch_off, cloud.extent, generator, device)
        p_signed = signed_points[signed_index]
        t_signed = signed_targets[signed_index]
        p_boundary = shell[boundary_index]
        p_inner = inner[inner_index]

        f_on, g_on = spatial_gradient(model, p_on, create_graph=True)
        f_off, g_off = spatial_gradient(model, p_off, create_graph=True)
        boundary_loss, inner_loss = closed_anchor_losses(
            model, p_boundary, p_inner, args.boundary_margin
        )
        losses = {
            "on": f_on.abs().mean(),
            "normal": oriented_normal_loss(g_on, n_on),
            "signed": signed_offset_loss(model, p_signed, t_signed, beta=max(pitch, 1.0e-6)),
            "off": torch.exp(-args.alpha_off * f_off.abs()).mean(),
            "boundary": boundary_loss,
            "inner": inner_loss,
        }
        eikonal = [(1.0 - g_off.norm(dim=-1)).abs()]
        if iso_points is not None:
            iso_index = torch.randint(
                len(iso_points), (args.batch_iso,), generator=generator, device=device
            )
            q_iso = iso_points[iso_index].clone()
            n_iso = iso_normals[iso_index]
            f_iso, g_iso = spatial_gradient(model, q_iso, create_graph=True)
            losses["iso"] = f_iso.abs().mean()
            losses["iso_normal"] = oriented_normal_loss(g_iso, n_iso)
            eikonal.append((1.0 - g_iso.norm(dim=-1)).abs())
        else:
            zero = f_on.new_zeros(())
            losses["iso"] = zero
            losses["iso_normal"] = zero
        losses["eik"] = torch.cat(eikonal).mean()
        effective_off = ramped_weight(step, args.lambda_off, args.off_warmup, args.off_ramp)
        total = (
            args.lambda_on * losses["on"]
            + args.lambda_normal * losses["normal"]
            + args.lambda_signed * losses["signed"]
            + effective_off * losses["off"]
            + args.lambda_eik * losses["eik"]
            + args.lambda_iso * losses["iso"]
            + args.lambda_iso_normal * losses["iso_normal"]
            + args.lambda_boundary * losses["boundary"]
            + args.lambda_inner * losses["inner"]
        )
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite A320 stabilized loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()

        gate_ran = False
        if step % args.gate_every == 0 or step == args.steps:
            model.eval()
            last_gate = strict_field_gate(
                model,
                cloud.extent,
                args.gate_grid,
                args.boundary_margin,
                shell,
                device,
                args.grid_chunk,
            )
            last_gate.update({"step": step, "phase": "post_optimizer"})
            gate_history.append(last_gate)
            base.atomic_json_dump({"gate_history": gate_history}, gate_path)
            model.train()
            gate_ran = True
            if not last_gate["passed"]:
                raise RuntimeError(f"A320 periodic strict gate failed at step {step}: {last_gate}")

        row = {
            "step": step,
            "total": float(total.detach()),
            **{key: float(value.detach()) for key, value in losses.items()},
            "effective_lambda_off": effective_off,
            "iso_count": 0 if iso_points is None else int(len(iso_points)),
            "gate_field_min": float(last_gate["field_min"]),
            "gate_boundary_min": float(last_gate["boundary_min"]),
            "gate_shell_min": float(last_gate["protected_shell"]["minimum"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.time() - started,
        }
        history.append(row)
        eligible_best = gate_ran and iso_points is not None and row["total"] < best_loss
        if eligible_best:
            best_loss = row["total"]
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(
                f"A320 stabilized [{step}/{args.steps}] loss={row['total']:.6g} "
                f"on={row['on']:.4g} signed={row['signed']:.4g} "
                f"off={row['off']:.4g} off_w={effective_off:.3g} "
                f"eik={row['eik']:.4g} iso={row['iso']:.4g}",
                flush=True,
            )
            base.append_csv(history_path, row)

        state = None
        if eligible_best or step % args.save_every == 0 or base._STOP_REQUESTED:
            state = checkpoint_state(
                args,
                model,
                optimizer,
                scheduler,
                step,
                best_loss,
                cloud,
                spec,
                initialization,
                normal_audit,
                offset_audit,
                iso_points,
                iso_normals,
                history,
                gate_history,
                refresh_history,
                generator,
                resume_allowed=bool(base._STOP_REQUESTED),
            )
        if eligible_best:
            base.atomic_torch_save(state, os.path.join(A320_OUTPUT_DIR, "checkpoint_best.pth.tar"))
        if step % args.save_every == 0 or base._STOP_REQUESTED:
            base.atomic_torch_save(state, os.path.join(A320_OUTPUT_DIR, "checkpoint_latest.pth.tar"))
        if base._STOP_REQUESTED:
            print("Stopped cleanly; A320 latest checkpoint is current.", flush=True)
            return

    if iso_points is None or not refresh_history:
        raise RuntimeError("A320 smoke ended without an accepted strict iso refresh")
    model.eval()
    final_gate = strict_field_gate(
        model,
        cloud.extent,
        args.gate_grid,
        args.boundary_margin,
        shell,
        device,
        args.grid_chunk,
    )
    final_gate.update({"step": args.steps, "phase": "pre_export"})
    gate_history.append(final_gate)
    base.atomic_json_dump({"gate_history": gate_history}, gate_path)
    if not final_gate["passed"]:
        raise RuntimeError(f"A320 final strict crossing/boundary gate failed: {final_gate}")
    latest = checkpoint_state(
        args,
        model,
        optimizer,
        scheduler,
        args.steps,
        best_loss,
        cloud,
        spec,
        initialization,
        normal_audit,
        offset_audit,
        iso_points,
        iso_normals,
        history,
        gate_history,
        refresh_history,
        generator,
    )
    base.atomic_torch_save(latest, os.path.join(A320_OUTPUT_DIR, "checkpoint_latest.pth.tar"))
    surface_path, surface_audit = export_surface(
        model, shell, args, cloud.extent, 2.0 * pitch, device
    )
    final = checkpoint_state(
        args,
        model,
        optimizer,
        scheduler,
        args.steps,
        best_loss,
        cloud,
        spec,
        initialization,
        normal_audit,
        offset_audit,
        iso_points,
        iso_normals,
        history,
        gate_history,
        refresh_history,
        generator,
        surface_audit=surface_audit,
    )
    final_path = os.path.join(A320_OUTPUT_DIR, "checkpoint_final.pth.tar")
    base.atomic_torch_save(final, final_path)
    summary = {
        "method": A320_METHOD_NAME,
        "paper": PAPER_ID,
        "manager_identity": A320_MANAGER_IDENTITY,
        "artifact_identity": A320_ARTIFACT_IDENTITY,
        "policy": A320_POLICY,
        "implementation_kind": A320_IMPLEMENTATION_KIND,
        "run_kind": "smoke",
        "steps": args.steps,
        "best_loss": best_loss,
        "scatter_checkpoint": A320_STAGE1_CHECKPOINT,
        "normal_orientation_audit": normal_audit,
        "signed_offset_audit": offset_audit,
        "final_gate": final_gate,
        "last_iso_refresh": refresh_history[-1],
        "surface_audit": surface_audit,
        "checkpoint_final": final_path,
        "surface_reconstruction": surface_path,
        "ground_truth_geometry_used": False,
        "novel_view_signal_supported": False,
    }
    base.atomic_json_dump(summary, os.path.join(A320_OUTPUT_DIR, "run_summary.json"))
    print(f"A320 stabilized smoke complete: {final_path}; surface={surface_path}", flush=True)
