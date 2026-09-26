#!/usr/bin/env python
"""Train the validity-gated Sugavanam--Ertin SE2 stabilized derivative.

SE2 starts only from a retained Stage-1 scattering-centre checkpoint.  It does
not resume or calibrate the failed raw SE1 SDF.  A deterministic sphere derived
from the Stage-1 cloud initializes the Fourier-feature network; sign anchors,
field gates, and audited projections then keep a closed zero surface alive.

No STL, mesh truth, LiDAR, validation geometry, or other ground-truth geometry
is accepted by this CLI.  Surface evaluation remains a separate post-training
decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import signal
import tempfile
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from rift.sugavanam_ertin import (
    PAPER_ID,
    FourierFeatureSDF,
    estimate_pca_normals,
    load_scattering_cloud,
    spatial_gradient,
)
from rift.sugavanam_ertin_validzero import (
    SE2_ARTIFACT_IDENTITY,
    SE2_MANAGER_IDENTITY,
    SE2_METHOD_NAME,
    SE2_POLICY,
    ClosedFieldSpec,
    ProjectionAcceptanceError,
    analytic_sphere_sdf,
    closed_anchor_losses,
    closed_field_spec,
    evaluate_field_grid,
    field_validity_from_array,
    field_validity_gate,
    mesh_topology_audit,
    ramped_weight,
    refresh_iso_points_valid,
    sample_boundary,
    sample_inner,
    sample_roi,
)


_STOP_REQUESTED = False


def _request_stop(signum, _frame) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"Received signal {signum}; saving after the current optimizer step.", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch_save(value: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".tmp-se2-", suffix=".pth.tar", dir=os.path.dirname(path) or "."
    )
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json_dump(value: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".tmp-se2-", suffix=".json", dir=os.path.dirname(path) or "."
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_npz_save(path: str, **arrays) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".tmp-se2-", suffix=".npz", dir=os.path.dirname(path) or "."
    )
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_csv(path: str, row: Dict[str, object]) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def restore_rng(state: dict) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def run_identity(
    run_kind: str,
    manager_identity: str = SE2_MANAGER_IDENTITY,
    artifact_identity: str = SE2_ARTIFACT_IDENTITY,
) -> tuple[str, str]:
    if run_kind == "full":
        return manager_identity, artifact_identity
    if run_kind == "smoke":
        return f"{manager_identity}_smoke", f"{artifact_identity}_smoke"
    raise ValueError(f"unknown run kind: {run_kind}")


def cosine_normal_loss(gradient: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    cosine = F.cosine_similarity(gradient, normal, dim=-1, eps=1e-8)
    return (1.0 - cosine.abs()).mean()


def fit_closed_initialization(
    model: FourierFeatureSDF,
    spec: ClosedFieldSpec,
    args: argparse.Namespace,
    generator: torch.Generator,
    device: torch.device,
) -> dict:
    """Fit the actual Fourier-feature model to one analytic closed sphere."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    final_loss = float("inf")
    model.train()
    for init_step in range(1, args.init_steps + 1):
        roi = sample_roi(args.init_batch, spec.extent, generator, device)
        boundary = sample_boundary(max(args.init_batch // 4, 32), spec.extent, generator, device)
        inner = sample_inner(max(args.init_batch // 8, 32), spec, generator, device)
        points = torch.cat((roi, boundary, inner), dim=0)
        target = analytic_sphere_sdf(points, spec).clamp(-0.9 * spec.extent, 0.9 * spec.extent)
        prediction = model(points)
        fit_loss = F.smooth_l1_loss(prediction, target, beta=max(spec.pitch, 1e-6))
        boundary_loss, interior_loss = closed_anchor_losses(
            model, boundary, inner, args.boundary_margin
        )
        loss = fit_loss + boundary_loss + interior_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite geometric initialization at step {init_step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        final_loss = float(loss.detach())
        if init_step == 1 or init_step % args.init_log_every == 0 or init_step == args.init_steps:
            print(
                f"SE2 init [{init_step}/{args.init_steps}] loss={final_loss:.6g} "
                f"fit={float(fit_loss.detach()):.4g} boundary={float(boundary_loss.detach()):.4g} "
                f"interior={float(interior_loss.detach()):.4g}",
                flush=True,
            )
        if _STOP_REQUESTED:
            raise InterruptedError("stop requested during geometric initialization")

    model.eval()
    gate = field_validity_gate(
        model, spec.extent, args.gate_grid, args.boundary_margin, device, args.grid_chunk
    )
    if not gate["passed"]:
        raise RuntimeError(f"closed geometric initialization failed validity gate: {gate}")
    return {
        "policy": SE2_POLICY,
        "steps": int(args.init_steps),
        "learning_rate": float(args.init_lr),
        "batch": int(args.init_batch),
        "final_loss": final_loss,
        "closed_field": spec.as_dict(),
        "initial_gate": gate,
    }


def sample_triangles(
    vertices: np.ndarray,
    faces: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    triangles = vertices[faces]
    area = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    valid = np.isfinite(area) & (area > 0)
    triangles, area = triangles[valid], area[valid]
    if len(triangles) == 0 or float(area.sum()) <= 0:
        raise RuntimeError("SE2 mesh has no finite nonzero-area triangles")
    index = rng.choice(len(triangles), size=count, replace=True, p=area / area.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return (
        triangles[index, 0]
        + u[:, None] * (triangles[index, 1] - triangles[index, 0])
        + v[:, None] * (triangles[index, 2] - triangles[index, 0])
    )


def export_valid_surface(
    model: FourierFeatureSDF,
    output_dir: str,
    extent: float,
    grid: int,
    mesh_points: int,
    boundary_margin: float,
    seed: int,
    device: torch.device,
    grid_chunk: int,
    manager_identity: str,
    artifact_identity: str,
) -> tuple[str, dict]:
    from skimage.measure import marching_cubes

    field, pitch = evaluate_field_grid(model, extent, grid, device, grid_chunk)
    validity = field_validity_from_array(field, boundary_margin)
    validity.update({"grid": int(grid), "pitch": float(pitch)})
    if not validity["passed"]:
        raise RuntimeError(f"final SE2 field failed validity gate: {validity}")
    vertices, faces, normals, _ = marching_cubes(
        field, level=0.0, spacing=(pitch, pitch, pitch)
    )
    vertices += -extent
    topology = mesh_topology_audit(vertices, faces)
    clearance = extent - np.abs(vertices).max(axis=1)
    topology["roi_clearance_min"] = float(clearance.min())
    topology["inside_roi"] = bool(np.all(clearance > 0.0))
    topology["passed"] = bool(topology["passed"] and topology["inside_roi"])
    if not topology["passed"]:
        raise RuntimeError(f"final SE2 mesh failed topology gate: {topology}")
    surface = sample_triangles(vertices, faces, mesh_points, np.random.default_rng(seed))
    path = os.path.join(output_dir, "surface_reconstruction.npz")
    atomic_npz_save(
        path,
        vertices=vertices.astype(np.float32),
        faces=faces.astype(np.int32),
        vertex_normals=normals.astype(np.float32),
        surface_points=surface.astype(np.float32),
        sdf=field.astype(np.float32),
        extent=np.float32(extent),
        pitch=np.float32(pitch),
        manager_identity=np.asarray(manager_identity),
        artifact_identity=np.asarray(artifact_identity),
        method=np.asarray(SE2_METHOD_NAME),
        policy=np.asarray(SE2_POLICY),
        ground_truth_geometry_used=np.bool_(False),
        validity_json=np.asarray(json.dumps(validity, sort_keys=True)),
        topology_json=np.asarray(json.dumps(topology, sort_keys=True)),
    )
    return path, {"validity": validity, "topology": topology}


def checkpoint_state(
    args: argparse.Namespace,
    model: FourierFeatureSDF,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    best_loss: float,
    cloud,
    spec: ClosedFieldSpec,
    initialization: dict,
    iso_points: Optional[torch.Tensor],
    iso_normals: Optional[torch.Tensor],
    history: list,
    gate_history: list,
    iso_refresh_history: list,
    sample_generator: torch.Generator,
    surface_audit: Optional[dict] = None,
) -> dict:
    manager_identity, artifact_identity = run_identity(
        args.run_kind, args.manager_identity, args.artifact_identity
    )
    return {
        "method": SE2_METHOD_NAME,
        "paper": PAPER_ID,
        "manager_identity": manager_identity,
        "artifact_identity": artifact_identity,
        "policy": SE2_POLICY,
        "implementation_kind": "stabilized derivative; not the raw paper reproduction",
        "ground_truth_geometry_used": False,
        "novel_view_signal_supported": False,
        "step": int(step),
        "best_loss": float(best_loss),
        "model_config": model.config(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": vars(args),
        "scatter_checkpoint": os.path.abspath(args.scatter_checkpoint),
        "scatter_source_epoch": int(cloud.source_epoch),
        "scatter_threshold_absolute": float(cloud.threshold),
        "n_scattering_centres": int(len(cloud.points)),
        "extent": float(cloud.extent),
        "granularity": int(cloud.granularity),
        "closed_field": spec.as_dict(),
        "initialization": initialization,
        "iso_points": None if iso_points is None else iso_points.detach().cpu(),
        "iso_normals": None if iso_normals is None else iso_normals.detach().cpu(),
        "history": history,
        "gate_history": gate_history,
        "iso_refresh_history": iso_refresh_history,
        "surface_audit": surface_audit,
        "sample_rng_state": sample_generator.get_state(),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def _same_value(saved: object, current: object) -> bool:
    if isinstance(saved, (float, int)) and isinstance(current, (float, int)):
        return bool(np.isclose(float(saved), float(current), rtol=0.0, atol=1e-12))
    return saved == current


RESUME_ARGUMENTS = (
    "run_kind",
    "manager_identity",
    "artifact_identity",
    "steps",
    "init_steps",
    "init_lr",
    "init_batch",
    "batch_on",
    "batch_off",
    "batch_iso",
    "batch_boundary",
    "batch_inner",
    "n_iso",
    "iso_start",
    "iso_refresh",
    "scatter_threshold",
    "max_scatter_points",
    "normal_radius",
    "n_fourier",
    "fourier_scale",
    "hidden_dim",
    "n_layers",
    "lr",
    "alpha_off",
    "lambda_on",
    "lambda_normal",
    "lambda_off",
    "lambda_eik",
    "lambda_iso",
    "lambda_iso_normal",
    "lambda_boundary",
    "lambda_inner",
    "off_warmup",
    "off_ramp",
    "boundary_margin",
    "gate_every",
    "gate_grid",
    "projection_tolerance",
    "projection_iterations",
    "projection_min_acceptance",
    "projection_oversample",
    "radius_quantile",
    "radius_cap_fraction",
    "seed",
)


def validate_resume_state(state: dict, args: argparse.Namespace, model: FourierFeatureSDF) -> None:
    manager_identity, artifact_identity = run_identity(
        args.run_kind, args.manager_identity, args.artifact_identity
    )
    if (
        state.get("method") != SE2_METHOD_NAME
        or state.get("manager_identity") != manager_identity
        or state.get("artifact_identity") != artifact_identity
        or state.get("policy") != SE2_POLICY
    ):
        raise ValueError("resume checkpoint is not this SE2 identity")
    if state.get("ground_truth_geometry_used") is not False:
        raise ValueError("resume checkpoint violates the no-geometry-truth contract")
    if os.path.realpath(state.get("scatter_checkpoint", "")) != os.path.realpath(args.scatter_checkpoint):
        raise ValueError("resume Stage-1 source path changed")
    if state.get("model_config") != model.config():
        raise ValueError("resume model configuration changed")
    saved_args = state.get("args") or {}
    for key in RESUME_ARGUMENTS:
        if key not in saved_args or not _same_value(saved_args[key], getattr(args, key)):
            raise ValueError(f"resume argument changed: {key}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scatter-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-kind", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--manager-identity", default=SE2_MANAGER_IDENTITY)
    parser.add_argument("--artifact-identity", default=SE2_ARTIFACT_IDENTITY)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-on", type=int, default=4096)
    parser.add_argument("--batch-off", type=int, default=4096)
    parser.add_argument("--batch-iso", type=int, default=4096)
    parser.add_argument("--batch-boundary", type=int, default=2048)
    parser.add_argument("--batch-inner", type=int, default=512)
    parser.add_argument("--n-iso", type=int, default=8192)
    parser.add_argument("--iso-start", type=int, default=750)
    parser.add_argument("--iso-refresh", type=int, default=250)
    parser.add_argument("--scatter-threshold", type=float, default=0.15)
    parser.add_argument("--max-scatter-points", type=int, default=20000)
    parser.add_argument("--normal-radius", type=float, default=0.0)
    parser.add_argument("--n-fourier", type=int, default=9, choices=(6, 9))
    parser.add_argument("--fourier-scale", type=float, default=2.0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha-off", type=float, default=100.0)
    parser.add_argument("--lambda-on", type=float, default=1.0)
    parser.add_argument("--lambda-normal", type=float, default=1.0)
    parser.add_argument("--lambda-off", type=float, default=1.0)
    parser.add_argument("--lambda-eik", type=float, default=1.0)
    parser.add_argument("--lambda-iso", type=float, default=1.0)
    parser.add_argument("--lambda-iso-normal", type=float, default=1.0)
    parser.add_argument("--lambda-boundary", type=float, default=1.0)
    parser.add_argument("--lambda-inner", type=float, default=1.0)
    parser.add_argument("--off-warmup", type=int, default=250)
    parser.add_argument("--off-ramp", type=int, default=500)
    parser.add_argument("--init-steps", type=int, default=500)
    parser.add_argument("--init-lr", type=float, default=1e-3)
    parser.add_argument("--init-batch", type=int, default=8192)
    parser.add_argument("--init-log-every", type=int, default=50)
    parser.add_argument("--radius-quantile", type=float, default=0.5)
    parser.add_argument("--radius-cap-fraction", type=float, default=0.65)
    parser.add_argument("--boundary-margin", type=float, default=1e-4)
    parser.add_argument("--gate-every", type=int, default=25)
    parser.add_argument("--gate-grid", type=int, default=48)
    parser.add_argument("--grid-chunk", type=int, default=65536)
    parser.add_argument("--projection-tolerance", type=float, default=1e-4)
    parser.add_argument("--projection-iterations", type=int, default=12)
    parser.add_argument("--projection-min-acceptance", type=float, default=0.5)
    parser.add_argument("--projection-oversample", type=float, default=2.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--mesh-grid", type=int, default=192)
    parser.add_argument("--mesh-points", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.steps < args.iso_start or args.iso_start <= 0 or args.iso_refresh <= 0:
        raise ValueError("SE2 must run through at least one positive iso refresh")
    if args.init_steps <= 0 or args.init_batch < 32:
        raise ValueError("closed initialization requires positive steps and init_batch>=32")
    if min(
        args.batch_on, args.batch_off, args.batch_iso, args.batch_boundary, args.batch_inner,
        args.n_iso, args.gate_every, args.gate_grid, args.mesh_grid, args.mesh_points,
    ) <= 0:
        raise ValueError("batch, gate, mesh, and iso counts must be positive")
    if args.gate_grid < 8 or args.mesh_grid < 8:
        raise ValueError("gate and mesh grids must be at least 8")
    if args.boundary_margin <= 0 or args.projection_tolerance <= 0:
        raise ValueError("boundary margin and projection tolerance must be positive")
    if not (0.0 < args.projection_min_acceptance <= 1.0):
        raise ValueError("projection_min_acceptance must be in (0,1]")
    if args.projection_oversample < 1.0:
        raise ValueError("projection_oversample must be >=1")
    if args.resume and os.path.basename(args.resume) != "checkpoint_latest.pth.tar":
        raise ValueError("SE2 resumes only its own checkpoint_latest.pth.tar")
    for label, identity in (
        ("manager", args.manager_identity),
        ("artifact", args.artifact_identity),
    ):
        if not identity or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for ch in identity):
            raise ValueError(f"invalid SE2 {label} identity: {identity!r}")
    output = os.path.realpath(args.output_dir)
    scatter_dir = os.path.dirname(os.path.realpath(args.scatter_checkpoint))
    if os.path.commonpath((output, scatter_dir)) == scatter_dir:
        raise ValueError("SE2 output must be outside the retained Stage-1 directory")
    if args.resume and os.path.dirname(os.path.realpath(args.resume)) != output:
        raise ValueError("SE2 resume must come from its own output directory")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not os.path.isfile(args.scatter_checkpoint):
        raise FileNotFoundError(args.scatter_checkpoint)
    os.makedirs(args.output_dir, exist_ok=True)
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    set_seed(args.seed)
    device = torch.device(args.device)
    cloud = load_scattering_cloud(
        args.scatter_checkpoint,
        threshold_fraction=args.scatter_threshold,
        max_points=args.max_scatter_points,
        normal_radius=args.normal_radius or None,
    )
    points = torch.as_tensor(cloud.points, device=device)
    normals = torch.as_tensor(cloud.normals, device=device)
    pitch = 2.0 * cloud.extent / cloud.granularity
    spec = closed_field_spec(
        cloud.points,
        cloud.extent,
        pitch,
        args.radius_quantile,
        args.radius_cap_fraction,
    )
    manager_identity, artifact_identity = run_identity(
        args.run_kind, args.manager_identity, args.artifact_identity
    )
    print(
        f"{SE2_METHOD_NAME}; identity={manager_identity}; centres={len(cloud.points)}; "
        f"closed_radius={spec.radius:.6g}; geometry truth: DISABLED",
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
    sample_generator = torch.Generator(device=device)
    sample_generator.manual_seed(args.seed)
    start_step = 0
    best_loss = float("inf")
    history: list = []
    gate_history: list = []
    iso_refresh_history: list = []
    iso_points = iso_normals = None

    if args.resume:
        try:
            resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        except TypeError:
            resume = torch.load(args.resume, map_location="cpu")
        validate_resume_state(resume, args, model)
        model.load_state_dict(resume["model_state_dict"])
        initialization = dict(resume["initialization"])
        saved_spec = resume.get("closed_field") or {}
        if saved_spec != spec.as_dict():
            raise ValueError("Stage-1-derived closed-field specification changed")
    else:
        initialization = fit_closed_initialization(model, spec, args, sample_generator, device)

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
        iso_refresh_history = list(resume.get("iso_refresh_history", []))
        iso_points = None if resume.get("iso_points") is None else resume["iso_points"].to(device)
        iso_normals = None if resume.get("iso_normals") is None else resume["iso_normals"].to(device)
        restore_rng(resume.get("rng_state"))
        if resume.get("sample_rng_state") is not None:
            sample_generator.set_state(resume["sample_rng_state"].cpu())
        print(f"Resumed SE2 step {start_step}/{args.steps} from {args.resume}", flush=True)
    else:
        gate_history.append({"step": 0, "phase": "post_initialization", **initialization["initial_gate"]})

    history_path = os.path.join(args.output_dir, "sdf_history.csv")
    gate_path = os.path.join(args.output_dir, "validity_history.json")
    refresh_path = os.path.join(args.output_dir, "iso_refresh_history.json")
    t0 = time.time()
    last_gate = dict(gate_history[-1])

    for step0 in range(start_step, args.steps):
        step = step0 + 1
        refresh_due = step >= args.iso_start and (
            iso_points is None or (step - args.iso_start) % args.iso_refresh == 0
        )
        if refresh_due:
            model.eval()
            pre_gate = field_validity_gate(
                model, cloud.extent, args.gate_grid, args.boundary_margin, device, args.grid_chunk
            )
            pre_gate.update({"step": step, "phase": "pre_iso_refresh"})
            gate_history.append(pre_gate)
            if not pre_gate["passed"]:
                atomic_json_dump({"gate_history": gate_history}, gate_path)
                raise RuntimeError(f"SE2 pre-refresh validity failure at step {step}: {pre_gate}")
            try:
                iso_points, refresh = refresh_iso_points_valid(
                    model,
                    points,
                    cloud.extent,
                    pitch,
                    args.n_iso,
                    sample_generator,
                    tolerance=args.projection_tolerance,
                    iterations=args.projection_iterations,
                    min_acceptance=args.projection_min_acceptance,
                    oversample=args.projection_oversample,
                )
            except ProjectionAcceptanceError as error:
                failed_refresh = {
                    "step": step,
                    "pre_gate": pre_gate,
                    **error.audit,
                }
                iso_refresh_history.append(failed_refresh)
                atomic_json_dump({"gate_history": gate_history}, gate_path)
                atomic_json_dump({"iso_refresh_history": iso_refresh_history}, refresh_path)
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
                    iso_points,
                    iso_normals,
                    history,
                    gate_history,
                    iso_refresh_history,
                    sample_generator,
                )
                atomic_torch_save(
                    failure_state,
                    os.path.join(args.output_dir, "checkpoint_projection_failure.pth.tar"),
                )
                print(
                    f"SE2 projection failure audit written at step {step}: "
                    f"{failed_refresh['accepted']}/{failed_refresh['attempted']} accepted",
                    flush=True,
                )
                raise
            iso_normals_np = estimate_pca_normals(
                iso_points.detach().cpu().numpy(), radius=3.0 * pitch
            )
            iso_normals = torch.as_tensor(iso_normals_np, device=device)
            post_gate = field_validity_gate(
                model, cloud.extent, args.gate_grid, args.boundary_margin, device, args.grid_chunk
            )
            post_gate.update({"step": step, "phase": "post_iso_refresh"})
            gate_history.append(post_gate)
            if not post_gate["passed"]:
                atomic_json_dump({"gate_history": gate_history}, gate_path)
                raise RuntimeError(f"SE2 post-refresh validity failure at step {step}: {post_gate}")
            refresh.update({"step": step, "pre_gate": pre_gate, "post_gate": post_gate})
            iso_refresh_history.append(refresh)
            atomic_json_dump({"iso_refresh_history": iso_refresh_history}, refresh_path)
            print(
                f"SE2 iso refresh step={step} accepted={refresh['accepted']}/{refresh['attempted']} "
                f"clamped={refresh['boundary_clamped']}",
                flush=True,
            )
            model.train()

        on_index = torch.randint(
            len(points), (args.batch_on,), generator=sample_generator, device=device
        )
        p_on = points[on_index].clone()
        n_on = normals[on_index]
        p_off = sample_roi(args.batch_off, cloud.extent, sample_generator, device)
        p_boundary = sample_boundary(
            args.batch_boundary, cloud.extent, sample_generator, device
        )
        p_inner = sample_inner(args.batch_inner, spec, sample_generator, device)

        f_on, g_on = spatial_gradient(model, p_on, create_graph=True)
        f_off, g_off = spatial_gradient(model, p_off, create_graph=True)
        boundary_loss, inner_loss = closed_anchor_losses(
            model, p_boundary, p_inner, args.boundary_margin
        )
        losses = {
            "on": f_on.abs().mean(),
            "normal": cosine_normal_loss(g_on, n_on),
            "off": torch.exp(-args.alpha_off * f_off.abs()).mean(),
            "boundary": boundary_loss,
            "inner": inner_loss,
        }
        eikonal_terms = [(1.0 - g_off.norm(dim=-1)).abs()]
        if iso_points is not None:
            iso_index = torch.randint(
                len(iso_points), (args.batch_iso,), generator=sample_generator, device=device
            )
            q_iso = iso_points[iso_index].clone()
            n_iso = iso_normals[iso_index]
            f_iso, g_iso = spatial_gradient(model, q_iso, create_graph=True)
            losses["iso"] = f_iso.abs().mean()
            losses["iso_normal"] = cosine_normal_loss(g_iso, n_iso)
            eikonal_terms.append((1.0 - g_iso.norm(dim=-1)).abs())
        else:
            zero = f_on.new_zeros(())
            losses["iso"] = zero
            losses["iso_normal"] = zero
        losses["eik"] = torch.cat(eikonal_terms).mean()

        effective_off = ramped_weight(
            step, args.lambda_off, args.off_warmup, args.off_ramp
        )
        total = (
            args.lambda_on * losses["on"]
            + args.lambda_normal * losses["normal"]
            + effective_off * losses["off"]
            + args.lambda_eik * losses["eik"]
            + args.lambda_iso * losses["iso"]
            + args.lambda_iso_normal * losses["iso_normal"]
            + args.lambda_boundary * losses["boundary"]
            + args.lambda_inner * losses["inner"]
        )
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite SE2 loss at step {step}: {losses}")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()

        gate_ran = False
        if step % args.gate_every == 0 or step == args.steps:
            model.eval()
            last_gate = field_validity_gate(
                model, cloud.extent, args.gate_grid, args.boundary_margin, device, args.grid_chunk
            )
            last_gate.update({"step": step, "phase": "post_optimizer"})
            gate_history.append(last_gate)
            atomic_json_dump({"gate_history": gate_history}, gate_path)
            model.train()
            gate_ran = True
            if not last_gate["passed"]:
                raise RuntimeError(f"SE2 field validity disappeared at step {step}: {last_gate}")

        values = {key: float(value.detach()) for key, value in losses.items()}
        row = {
            "step": step,
            "total": float(total.detach()),
            **values,
            "effective_lambda_off": effective_off,
            "iso_count": 0 if iso_points is None else int(len(iso_points)),
            "gate_field_min": float(last_gate["field_min"]),
            "gate_boundary_min": float(last_gate["boundary_min"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.time() - t0,
        }
        history.append(row)
        eligible_best = gate_ran and iso_points is not None and row["total"] < best_loss
        if eligible_best:
            best_loss = row["total"]
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            print(
                f"SE2 step [{step}/{args.steps}] loss={row['total']:.6g} "
                f"on={row['on']:.4g} off={row['off']:.4g} off_w={effective_off:.3g} "
                f"eik={row['eik']:.4g} iso={row['iso']:.4g} "
                f"boundary={row['boundary']:.4g} inner={row['inner']:.4g}",
                flush=True,
            )
            append_csv(history_path, row)

        state = None
        if eligible_best or step % args.save_every == 0 or _STOP_REQUESTED:
            state = checkpoint_state(
                args, model, optimizer, scheduler, step, best_loss, cloud, spec, initialization,
                iso_points, iso_normals, history, gate_history, iso_refresh_history,
                sample_generator,
            )
        if eligible_best:
            atomic_torch_save(state, os.path.join(args.output_dir, "checkpoint_best.pth.tar"))
        if step % args.save_every == 0 or _STOP_REQUESTED:
            atomic_torch_save(state, os.path.join(args.output_dir, "checkpoint_latest.pth.tar"))
        if _STOP_REQUESTED:
            print(f"Stopped cleanly after SE2 step {step}; latest checkpoint is current.", flush=True)
            return

    if iso_points is None or not iso_refresh_history:
        raise RuntimeError("SE2 ended without a validity-gated iso refresh")
    model.eval()
    final_gate = field_validity_gate(
        model, cloud.extent, args.gate_grid, args.boundary_margin, device, args.grid_chunk
    )
    final_gate.update({"step": args.steps, "phase": "pre_export"})
    gate_history.append(final_gate)
    atomic_json_dump({"gate_history": gate_history}, gate_path)
    if not final_gate["passed"]:
        raise RuntimeError(f"SE2 final field validity failure: {final_gate}")

    latest = checkpoint_state(
        args, model, optimizer, scheduler, args.steps, best_loss, cloud, spec, initialization,
        iso_points, iso_normals, history, gate_history, iso_refresh_history, sample_generator,
    )
    atomic_torch_save(latest, os.path.join(args.output_dir, "checkpoint_latest.pth.tar"))
    surface_path, surface_audit = export_valid_surface(
        model,
        args.output_dir,
        cloud.extent,
        args.mesh_grid,
        args.mesh_points,
        args.boundary_margin,
        args.seed,
        device,
        args.grid_chunk,
        manager_identity,
        artifact_identity,
    )
    final = checkpoint_state(
        args, model, optimizer, scheduler, args.steps, best_loss, cloud, spec, initialization,
        iso_points, iso_normals, history, gate_history, iso_refresh_history, sample_generator,
        surface_audit=surface_audit,
    )
    final_path = os.path.join(args.output_dir, "checkpoint_final.pth.tar")
    atomic_torch_save(final, final_path)
    summary = {
        "method": SE2_METHOD_NAME,
        "paper": PAPER_ID,
        "manager_identity": manager_identity,
        "artifact_identity": artifact_identity,
        "policy": SE2_POLICY,
        "run_kind": args.run_kind,
        "steps": args.steps,
        "best_loss": best_loss,
        "n_scattering_centres": len(cloud.points),
        "scatter_checkpoint": os.path.abspath(args.scatter_checkpoint),
        "closed_field": spec.as_dict(),
        "initialization": initialization,
        "final_gate": final_gate,
        "last_iso_refresh": iso_refresh_history[-1],
        "surface_audit": surface_audit,
        "checkpoint_final": os.path.abspath(final_path),
        "surface_reconstruction": os.path.abspath(surface_path),
        "ground_truth_geometry_used": False,
        "novel_view_signal_supported": False,
    }
    atomic_json_dump(summary, os.path.join(args.output_dir, "run_summary.json"))
    print(f"SE2 complete: {final_path}; surface={surface_path}", flush=True)


if __name__ == "__main__":
    main()
