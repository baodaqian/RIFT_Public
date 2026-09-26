#!/usr/bin/env python
"""Export a retained Sugavanam--Ertin final SDF with a data-only level fix.

This is deliberately an export-only recovery tool.  It never constructs an
optimizer, resumes training, or writes a checkpoint/history file.  The scalar
level is the unweighted median of the final SDF at the same thresholded stage-1
scattering-centre set used by the on-surface L1 term (Eq. 20 of
Sugavanam--Ertin, arXiv:2602.17556).  Geometry truth is not accepted anywhere
in this CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile

import numpy as np
import torch

# The launcher invokes this file as ``python scripts/...``.  Put the project
# root on sys.path explicitly so ``rift`` resolves regardless of invocation.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rift.sugavanam_ertin import METHOD_NAME, PAPER_ID, FourierFeatureSDF, load_scattering_cloud


CALIBRATION_POLICY = "stage1_median_unweighted_l1_v1"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_npz_save(path: str, **arrays) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-se-cal-", suffix=".npz", dir=os.path.dirname(path) or ".")
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json_dump(value: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-se-cal-", suffix=".json", dir=os.path.dirname(path) or ".")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _torch_load(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6
        return torch.load(path, map_location="cpu")


def _same_saved_argument(saved: object, current: object) -> bool:
    if isinstance(saved, (float, int)) and isinstance(current, (float, int)):
        return bool(np.isclose(float(saved), float(current), rtol=0.0, atol=1e-12))
    return saved == current


def load_final_sdf(args: argparse.Namespace, cloud_sha: str, cloud, device: torch.device):
    """Validate and load only an immutable, planned terminal SDF checkpoint."""
    if os.path.basename(args.sdf_checkpoint) != "checkpoint_final.pth.tar":
        raise ValueError("calibration accepts only a checkpoint_final.pth.tar source")
    state = _torch_load(args.sdf_checkpoint)
    if state.get("method") != METHOD_NAME:
        raise ValueError("source checkpoint is not a Sugavanam--Ertin SDF state")
    if int(state.get("step", -1)) != args.expected_step:
        raise ValueError(f"source checkpoint step is not the required {args.expected_step}")
    if state.get("scatter_checkpoint_sha256") != cloud_sha:
        raise ValueError("source checkpoint does not bind the supplied stage-1 checkpoint")
    if state.get("ground_truth_geometry_used") is not False:
        raise ValueError("source checkpoint violates the no-geometry-truth contract")
    saved_args = state.get("args") or {}
    for key in ("scatter_threshold", "max_scatter_points", "normal_radius"):
        if key not in saved_args or not _same_saved_argument(saved_args[key], getattr(args, key)):
            raise ValueError(f"source checkpoint {key} does not match this export request")
    model_config = state.get("model_config")
    if not isinstance(model_config, dict) or not np.isclose(float(model_config.get("extent", -1)), cloud.extent):
        raise ValueError("source checkpoint model extent does not match the stage-1 cloud")
    model = FourierFeatureSDF(**model_config).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, state


@torch.no_grad()
def evaluate_points(model: torch.nn.Module, points: np.ndarray, device: torch.device, chunk: int) -> np.ndarray:
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    centres = torch.as_tensor(np.asarray(points, dtype=np.float32))
    if centres.ndim != 2 or centres.shape[1] != 3 or len(centres) < 3:
        raise ValueError("stage-1 scattering centres must have shape [N,3] with N>=3")
    values = []
    for start in range(0, len(centres), chunk):
        values.append(model(centres[start : start + chunk].to(device)).detach().cpu().numpy())
    values = np.concatenate(values).astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("source SDF has non-finite values at stage-1 scattering centres")
    return values


def calibrate_level(model: torch.nn.Module, points: np.ndarray, device: torch.device, chunk: int) -> dict:
    """Return the exact unweighted L1-optimal global level for the source cloud."""
    values = evaluate_points(model, points, device, chunk)
    level = float(np.median(values))
    lo, hi = float(values.min()), float(values.max())
    if not (lo < level < hi):
        raise ValueError(
            "stage-1 centre values have no strict median spread: "
            f"min={lo:.6g}, median={level:.6g}, max={hi:.6g}"
        )
    q05, q95 = (float(x) for x in np.quantile(values, (0.05, 0.95)))
    return {
        "policy": CALIBRATION_POLICY,
        "raw_isolevel": level,
        "n_scattering_centres": int(len(values)),
        "raw_sdf_min": lo,
        "raw_sdf_q05": q05,
        "raw_sdf_median": level,
        "raw_sdf_q95": q95,
        "raw_sdf_max": hi,
    }


@torch.no_grad()
def evaluate_grid(model: torch.nn.Module, extent: float, grid: int, device: torch.device, chunk: int) -> tuple[np.ndarray, float]:
    if grid < 2:
        raise ValueError("mesh grid must be at least 2")
    pitch = 2.0 * extent / grid
    axis = torch.linspace(-extent + pitch / 2, extent - pitch / 2, grid, device=device)
    values = []
    for ix in range(grid):
        yz = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1).reshape(-1, 2)
        xyz = torch.cat((torch.full((len(yz), 1), axis[ix], device=device), yz), dim=-1)
        for start in range(0, len(xyz), chunk):
            values.append(model(xyz[start : start + chunk]).detach().cpu())
    raw = torch.cat(values).reshape(grid, grid, grid).numpy()
    if not np.isfinite(raw).all():
        raise ValueError("source SDF has non-finite values on the export grid")
    return raw, pitch


def boundary_values(field: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            field[0, :, :].ravel(), field[-1, :, :].ravel(),
            field[:, 0, :].ravel(), field[:, -1, :].ravel(),
            field[:, :, 0].ravel(), field[:, :, -1].ravel(),
        )
    )


def sample_triangles(vertices: np.ndarray, faces: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    tri = vertices[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    valid = area > 0
    tri, area = tri[valid], area[valid]
    if len(tri) == 0 or not np.isfinite(area).all() or float(area.sum()) <= 0:
        raise RuntimeError("marching-cubes mesh has no finite nonzero-area triangles")
    idx = rng.choice(len(tri), size=n, replace=True, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return tri[idx, 0] + u[:, None] * (tri[idx, 1] - tri[idx, 0]) + v[:, None] * (tri[idx, 2] - tri[idx, 0])


def export_surface(
    model: torch.nn.Module,
    output_dir: str,
    extent: float,
    grid: int,
    n_points: int,
    seed: int,
    device: torch.device,
    calibration: dict,
    source: dict,
    scatter_sha: str,
    chunk: int,
) -> str:
    from skimage.measure import marching_cubes

    raw, pitch = evaluate_grid(model, extent, grid, device, chunk)
    level = float(calibration["raw_isolevel"])
    raw_lo, raw_hi = float(raw.min()), float(raw.max())
    if not (raw_lo < level < raw_hi):
        raise RuntimeError(
            f"calibration level {level:.6g} does not strictly cross the {grid}^3 raw grid "
            f"[{raw_lo:.6g}, {raw_hi:.6g}]"
        )
    residual = raw - level
    boundary = boundary_values(residual)
    if np.all(boundary > 0.0):
        orientation, sign_flipped = 1, False
    elif np.all(boundary < 0.0):
        orientation, sign_flipped = -1, True
    else:
        raise RuntimeError("calibrated zero level reaches or crosses the ROI boundary; refusing an open mesh")
    sdf = orientation * residual
    oriented_boundary = orientation * boundary
    if not (float(sdf.min()) < 0.0 < float(sdf.max())):
        raise RuntimeError("oriented calibrated field has no strict zero crossing")
    verts, faces, normals, _ = marching_cubes(sdf, level=0.0, spacing=(pitch, pitch, pitch))
    verts += -extent + pitch / 2
    surface = sample_triangles(verts, faces, n_points, np.random.default_rng(seed))
    if not (np.isfinite(verts).all() and np.isfinite(normals).all() and np.isfinite(surface).all()):
        raise RuntimeError("marching-cubes export has non-finite geometry")
    path = os.path.join(output_dir, "surface_reconstruction.npz")
    atomic_npz_save(
        path,
        vertices=verts.astype(np.float32),
        faces=faces.astype(np.int32),
        vertex_normals=normals.astype(np.float32),
        surface_points=surface.astype(np.float32),
        sdf=sdf.astype(np.float32),
        extent=np.float32(extent),
        pitch=np.float32(pitch),
        sign_flipped=np.bool_(sign_flipped),
        orientation=np.int8(orientation),
        raw_isolevel=np.float64(level),
        calibration_policy=np.asarray(CALIBRATION_POLICY),
        calibration_n_scattering_centres=np.int64(calibration["n_scattering_centres"]),
        calibration_raw_sdf_min=np.float64(calibration["raw_sdf_min"]),
        calibration_raw_sdf_q05=np.float64(calibration["raw_sdf_q05"]),
        calibration_raw_sdf_median=np.float64(calibration["raw_sdf_median"]),
        calibration_raw_sdf_q95=np.float64(calibration["raw_sdf_q95"]),
        calibration_raw_sdf_max=np.float64(calibration["raw_sdf_max"]),
        raw_grid_min=np.float64(raw_lo),
        raw_grid_max=np.float64(raw_hi),
        boundary_oriented_min=np.float64(oriented_boundary.min()),
        boundary_oriented_max=np.float64(oriented_boundary.max()),
        source_checkpoint=np.asarray(source["checkpoint"]),
        source_checkpoint_sha256=np.asarray(source["sha256"]),
        source_checkpoint_role=np.asarray("checkpoint_final"),
        source_checkpoint_step=np.int64(source["step"]),
        source_checkpoint_best_loss=np.float64(source["best_loss"]),
        scatter_checkpoint_sha256=np.asarray(scatter_sha),
        method=np.asarray(METHOD_NAME),
    )
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scatter-checkpoint", required=True)
    p.add_argument("--sdf-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--scatter-threshold", type=float, default=0.15)
    p.add_argument("--max-scatter-points", type=int, default=20000)
    p.add_argument("--normal-radius", type=float, default=0.0)
    p.add_argument("--expected-step", type=int, default=5000)
    p.add_argument("--mesh-grid", type=int, default=192)
    p.add_argument("--mesh-points", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk", type=int, default=262144)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def calibrated_output_dir(output_dir: str, *source_checkpoints: str) -> str:
    """Return a safe distinct artifact root, never nested below either source."""
    output_dir = os.path.realpath(output_dir)
    for source_checkpoint in source_checkpoints:
        source_dir = os.path.dirname(os.path.realpath(source_checkpoint))
        if os.path.commonpath((output_dir, source_dir)) == source_dir:
            raise ValueError(
                "calibrated artifacts must be outside both retained source checkpoint directories"
            )
    return output_dir


def main() -> None:
    args = parse_args()
    args.output_dir = calibrated_output_dir(
        args.output_dir, args.sdf_checkpoint, args.scatter_checkpoint
    )
    if not os.path.isfile(args.scatter_checkpoint) or not os.path.isfile(args.sdf_checkpoint):
        raise FileNotFoundError("missing required stage-1 or final SDF checkpoint")
    device = torch.device(args.device)
    scatter_sha = sha256(args.scatter_checkpoint)
    cloud = load_scattering_cloud(
        args.scatter_checkpoint,
        threshold_fraction=args.scatter_threshold,
        max_points=args.max_scatter_points,
        normal_radius=args.normal_radius or None,
    )
    model, state = load_final_sdf(args, scatter_sha, cloud, device)
    calibration = calibrate_level(model, cloud.points, device, args.chunk)
    source = {
        "checkpoint": os.path.abspath(args.sdf_checkpoint),
        "sha256": sha256(args.sdf_checkpoint),
        "step": int(state["step"]),
        "best_loss": float(state["best_loss"]),
    }
    surface = export_surface(
        model, args.output_dir, cloud.extent, args.mesh_grid, args.mesh_points, args.seed,
        device, calibration, source, scatter_sha, args.chunk,
    )
    summary = {
        "method": METHOD_NAME,
        "paper": PAPER_ID,
        "calibration_policy": CALIBRATION_POLICY,
        "source_checkpoint": source["checkpoint"],
        "source_checkpoint_sha256": source["sha256"],
        "source_checkpoint_role": "checkpoint_final",
        "source_checkpoint_step": source["step"],
        "source_checkpoint_best_loss": source["best_loss"],
        "scatter_checkpoint": os.path.abspath(args.scatter_checkpoint),
        "scatter_checkpoint_sha256": scatter_sha,
        "scatter_threshold_fraction": args.scatter_threshold,
        "max_scatter_points": args.max_scatter_points,
        "normal_radius": args.normal_radius,
        "calibration": calibration,
        "mesh_grid": args.mesh_grid,
        "mesh_points": args.mesh_points,
        "surface_reconstruction": os.path.abspath(surface),
        "ground_truth_geometry_used": False,
        "novel_view_signal_supported": False,
    }
    atomic_json_dump(summary, os.path.join(args.output_dir, "run_summary.json"))
    print(
        f"CALIBRATION PASS policy={CALIBRATION_POLICY} source=checkpoint_final "
        f"step={source['step']} centres={calibration['n_scattering_centres']} "
        f"level={calibration['raw_isolevel']:.8g} surface={surface}",
        flush=True,
    )


if __name__ == "__main__":
    main()
