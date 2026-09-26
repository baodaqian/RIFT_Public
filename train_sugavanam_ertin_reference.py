#!/usr/bin/env python
"""Fit the Sugavanam--Ertin neural-SDF stage to radar scattering centres.

Stage 1 is an isotropic, L1-regularized coherent inversion produced by RIFT's
validated range operator.  This script loads that checkpoint, thresholds its
joint magnitude into the paper's scattering-centre set P, estimates PCA
normals, and trains the paper's Fourier-feature SDF with on/off-surface,
Eikonal, normal, and resampled iso-point losses.

No STL, mesh, LiDAR, or other ground-truth geometry is accepted by this CLI.
Geometry truth is reserved for post-hoc evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import signal
import tempfile
import time
from typing import Dict, Optional

import numpy as np
import torch

from rift.sugavanam_ertin import (
    METHOD_NAME,
    PAPER_ID,
    FourierFeatureSDF,
    estimate_pca_normals,
    load_scattering_cloud,
    project_to_zero_level,
    refresh_iso_points,
    spatial_gradient,
)


_STOP_REQUESTED = False


def _request_stop(signum, _frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"Received signal {signum}; saving after the current optimizer step.", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_torch_save(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-sdf-", suffix=".pth.tar", dir=os.path.dirname(path) or ".")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def model_checkpoint(
    args: argparse.Namespace,
    model: FourierFeatureSDF,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    best_loss: float,
    cloud,
    cloud_sha256: str,
    iso_points: Optional[torch.Tensor],
    iso_normals: Optional[torch.Tensor],
    history: list,
    sample_generator: torch.Generator,
) -> dict:
    return {
        "method": METHOD_NAME,
        "paper": PAPER_ID,
        "implementation_kind": "independent reimplementation; no official code released",
        "stage1_kind": "L1 isotropic scattering centres via RIFT exact bistatic range operator",
        "stage2_kind": "Fourier-feature neural SDF with projected/uniformized iso-points",
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
        "scatter_checkpoint_sha256": cloud_sha256,
        "scatter_source_epoch": cloud.source_epoch,
        "scatter_threshold_absolute": cloud.threshold,
        "n_scattering_centres": int(len(cloud.points)),
        "extent": cloud.extent,
        "granularity": cloud.granularity,
        "iso_points": None if iso_points is None else iso_points.detach().cpu(),
        "iso_normals": None if iso_normals is None else iso_normals.detach().cpu(),
        "history": history,
        "sample_rng_state": sample_generator.get_state(),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def restore_rng(state: dict) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def cosine_normal_loss(grad: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    cos = torch.nn.functional.cosine_similarity(grad, normal, dim=-1, eps=1e-8)
    return (1.0 - cos.abs()).mean()


def append_history(path: str, row: Dict[str, float]) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


@torch.no_grad()
def evaluate_grid(model: FourierFeatureSDF, extent: float, grid: int, device, chunk: int = 262144):
    pitch = 2.0 * extent / grid
    axis = torch.linspace(-extent + pitch / 2, extent - pitch / 2, grid, device=device)
    values = []
    for ix in range(grid):
        yz = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1).reshape(-1, 2)
        xyz = torch.cat((torch.full((len(yz), 1), axis[ix], device=device), yz), dim=-1)
        for start in range(0, len(xyz), chunk):
            values.append(model(xyz[start : start + chunk]).cpu())
    return torch.cat(values).reshape(grid, grid, grid).numpy(), pitch


def sample_triangles(vertices: np.ndarray, faces: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    tri = vertices[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    valid = area > 0
    tri, area = tri[valid], area[valid]
    idx = rng.choice(len(tri), size=n, replace=True, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return tri[idx, 0] + u[:, None] * (tri[idx, 1] - tri[idx, 0]) + v[:, None] * (tri[idx, 2] - tri[idx, 0])


def export_surface(model, out_dir: str, extent: float, grid: int, n_points: int, seed: int, device) -> str:
    from skimage.measure import marching_cubes

    sdf, pitch = evaluate_grid(model, extent, grid, device)
    if not (float(sdf.min()) < 0.0 < float(sdf.max())):
        raise RuntimeError(
            f"SDF has no zero crossing on {grid}^3 grid: min={sdf.min():.4g}, max={sdf.max():.4g}"
        )
    # Sign is globally ambiguous under the paper's absolute-normal losses.
    corners = sdf[np.ix_([0, -1], [0, -1], [0, -1])]
    sign_flip = bool(float(corners.mean()) < 0)
    if sign_flip:
        sdf = -sdf
    verts, faces, normals, _ = marching_cubes(sdf, level=0.0, spacing=(pitch, pitch, pitch))
    verts += -extent + pitch / 2
    surface = sample_triangles(verts, faces, n_points, np.random.default_rng(seed))
    path = os.path.join(out_dir, "surface_reconstruction.npz")
    np.savez_compressed(
        path,
        vertices=verts.astype(np.float32),
        faces=faces.astype(np.int32),
        vertex_normals=normals.astype(np.float32),
        surface_points=surface.astype(np.float32),
        sdf=sdf.astype(np.float32),
        extent=np.float32(extent),
        pitch=np.float32(pitch),
        sign_flipped=np.bool_(sign_flip),
        method=np.asarray(METHOD_NAME),
    )
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scatter-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resume", default=None)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-on", type=int, default=4096)
    p.add_argument("--batch-off", type=int, default=4096)
    p.add_argument("--batch-iso", type=int, default=4096)
    p.add_argument("--n-iso", type=int, default=8192)
    p.add_argument("--iso-start", type=int, default=500)
    p.add_argument("--iso-refresh", type=int, default=250)
    p.add_argument("--scatter-threshold", type=float, default=0.15)
    p.add_argument("--max-scatter-points", type=int, default=20000)
    p.add_argument("--normal-radius", type=float, default=0.0,
                   help="0 = three stage-1 voxel pitches (paper uses 0.3 m on full-scale vehicles)")
    p.add_argument("--n-fourier", type=int, default=9, choices=[6, 9])
    p.add_argument("--fourier-scale", type=float, default=2.0)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--alpha-off", type=float, default=100.0)
    p.add_argument("--lambda-on", type=float, default=1.0)
    p.add_argument("--lambda-normal", type=float, default=1.0)
    p.add_argument("--lambda-off", type=float, default=1.0)
    p.add_argument("--lambda-eik", type=float, default=1.0)
    p.add_argument("--lambda-iso", type=float, default=1.0)
    p.add_argument("--lambda-iso-normal", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--mesh-grid", type=int, default=192)
    p.add_argument("--mesh-points", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true",
                   help="run a tiny contract check (small MLP/batches/steps/mesh), never a paper result")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.steps = min(args.steps, 3)
        args.batch_on = min(args.batch_on, 32)
        args.batch_off = min(args.batch_off, 32)
        args.batch_iso = min(args.batch_iso, 32)
        args.n_iso = min(args.n_iso, 64)
        args.iso_start = min(args.iso_start, 1)
        args.iso_refresh = 1
        args.hidden_dim = min(args.hidden_dim, 32)
        args.n_layers = max(5, min(args.n_layers, 5))
        args.mesh_grid = min(args.mesh_grid, 24)
        args.mesh_points = min(args.mesh_points, 256)
        args.log_every = 1
        args.save_every = 1

    os.makedirs(args.output_dir, exist_ok=True)
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    set_seed(args.seed)
    device = torch.device(args.device)
    cloud_sha = sha256(args.scatter_checkpoint)
    cloud = load_scattering_cloud(
        args.scatter_checkpoint,
        threshold_fraction=args.scatter_threshold,
        max_points=args.max_scatter_points,
        normal_radius=args.normal_radius or None,
    )
    print(
        f"{METHOD_NAME}; {len(cloud.points)} centres from stage-1 epoch {cloud.source_epoch}; "
        f"tau={args.scatter_threshold:g}*max={cloud.threshold:.4g}; geometry truth: DISABLED",
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.steps, 1), eta_min=args.lr * 0.01)
    start_step, best_loss, history = 0, float("inf"), []
    iso_points = iso_normals = None
    sample_gen = torch.Generator(device=device)
    sample_gen.manual_seed(args.seed)

    if args.resume:
        try:
            resume = torch.load(args.resume, map_location=device, weights_only=False)
        except TypeError:
            resume = torch.load(args.resume, map_location=device)
        if resume.get("method") != METHOD_NAME:
            raise ValueError(f"resume file is not {METHOD_NAME}")
        if resume.get("scatter_checkpoint_sha256") != cloud_sha:
            raise ValueError("stage-1 checkpoint changed since the SDF checkpoint was written")
        if resume.get("model_config") != model.config():
            raise ValueError("resume model configuration does not match current CLI")
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        start_step = int(resume["step"])
        best_loss = float(resume["best_loss"])
        history = list(resume.get("history", []))
        iso_points = None if resume.get("iso_points") is None else resume["iso_points"].to(device)
        iso_normals = None if resume.get("iso_normals") is None else resume["iso_normals"].to(device)
        restore_rng(resume.get("rng_state"))
        if resume.get("sample_rng_state") is not None:
            sample_gen.set_state(resume["sample_rng_state"].cpu())
        print(f"Resumed SDF step {start_step}/{args.steps} from {args.resume}", flush=True)

    points = torch.as_tensor(cloud.points, device=device)
    normals = torch.as_tensor(cloud.normals, device=device)
    pitch = 2.0 * cloud.extent / cloud.granularity
    history_path = os.path.join(args.output_dir, "sdf_history.csv")
    t0 = time.time()

    for step0 in range(start_step, args.steps):
        step = step0 + 1
        if step >= args.iso_start and (
            iso_points is None or (step - args.iso_start) % max(args.iso_refresh, 1) == 0
        ):
            model.eval()
            iso_points = refresh_iso_points(
                model, points, cloud.extent, pitch, args.n_iso, generator=sample_gen
            )
            iso_normals_np = estimate_pca_normals(
                iso_points.detach().cpu().numpy(), radius=3.0 * pitch
            )
            iso_normals = torch.as_tensor(iso_normals_np, device=device)
            model.train()
            print(f"Refreshed {len(iso_points)} projected/uniformized iso-points at step {step}", flush=True)

        on_idx = torch.randint(len(points), (args.batch_on,), generator=sample_gen, device=device)
        p_on = points[on_idx].clone()
        n_on = normals[on_idx]
        p_off = (2.0 * torch.rand(
            args.batch_off, 3, generator=sample_gen, device=device
        ) - 1.0) * cloud.extent

        f_on, g_on = spatial_gradient(model, p_on, create_graph=True)
        f_off, g_off = spatial_gradient(model, p_off, create_graph=True)
        losses = {
            "on": f_on.abs().mean(),
            "normal": cosine_normal_loss(g_on, n_on),
            "off": torch.exp(-args.alpha_off * f_off.abs()).mean(),
            "eik": (1.0 - g_off.norm(dim=-1)).abs().mean(),
        }
        if iso_points is not None:
            iso_idx = torch.randint(len(iso_points), (args.batch_iso,), generator=sample_gen, device=device)
            q_iso = iso_points[iso_idx].clone()
            n_iso = iso_normals[iso_idx]
            f_iso, g_iso = spatial_gradient(model, q_iso, create_graph=True)
            losses["iso"] = f_iso.abs().mean()
            losses["iso_normal"] = cosine_normal_loss(g_iso, n_iso)
        else:
            zero = f_on.new_zeros(())
            losses["iso"] = zero
            losses["iso_normal"] = zero

        total = (
            args.lambda_on * losses["on"]
            + args.lambda_normal * losses["normal"]
            + args.lambda_off * losses["off"]
            + args.lambda_eik * losses["eik"]
            + args.lambda_iso * losses["iso"]
            + args.lambda_iso_normal * losses["iso_normal"]
        )
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite loss at step {step}: {losses}")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        scheduler.step()

        values = {k: float(v.detach()) for k, v in losses.items()}
        row = {
            "step": step,
            "total": float(total.detach()),
            **values,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.time() - t0,
        }
        history.append(row)
        is_best = row["total"] < best_loss
        if is_best:
            best_loss = row["total"]
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            print(
                f"SDF step [{step}/{args.steps}] loss={row['total']:.6g} "
                f"on={row['on']:.4g} off={row['off']:.4g} eik={row['eik']:.4g} "
                f"normal={row['normal']:.4g} iso={row['iso']:.4g} "
                f"[{row['seconds'] / step:.3f}s/step]",
                flush=True,
            )
            append_history(history_path, row)

        state = None
        if is_best or step % args.save_every == 0 or step == args.steps or _STOP_REQUESTED:
            state = model_checkpoint(
                args, model, optimizer, scheduler, step, best_loss, cloud, cloud_sha,
                iso_points, iso_normals, history, sample_gen,
            )
        if is_best:
            atomic_torch_save(state, os.path.join(args.output_dir, "checkpoint_best.pth.tar"))
        if step < args.steps or _STOP_REQUESTED:
            if step % args.save_every == 0 or _STOP_REQUESTED:
                atomic_torch_save(state, os.path.join(args.output_dir, "checkpoint_latest.pth.tar"))
        if _STOP_REQUESTED:
            print(f"Stopped cleanly after SDF step {step}; checkpoint_latest is current.", flush=True)
            return

    final_state = model_checkpoint(
        args, model, optimizer, scheduler, args.steps, best_loss, cloud, cloud_sha,
        iso_points, iso_normals, history, sample_gen,
    )
    final_path = os.path.join(args.output_dir, "checkpoint_final.pth.tar")
    atomic_torch_save(final_state, final_path)
    model.eval()
    surface_path = export_surface(
        model, args.output_dir, cloud.extent, args.mesh_grid, args.mesh_points, args.seed, device
    )
    with open(os.path.join(args.output_dir, "run_summary.json"), "w") as f:
        json.dump(
            {
                "method": METHOD_NAME,
                "paper": PAPER_ID,
                "steps": args.steps,
                "best_loss": best_loss,
                "n_scattering_centres": len(cloud.points),
                "scatter_checkpoint": os.path.abspath(args.scatter_checkpoint),
                "scatter_checkpoint_sha256": cloud_sha,
                "checkpoint_final": os.path.abspath(final_path),
                "surface_reconstruction": os.path.abspath(surface_path),
                "ground_truth_geometry_used": False,
                "novel_view_signal_supported": False,
            },
            f,
            indent=2,
        )
    print(f"SDF training complete: {final_path}; surface: {surface_path}", flush=True)


if __name__ == "__main__":
    main()
