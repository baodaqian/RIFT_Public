#!/usr/bin/env python3
"""Postflight fixed-threshold B787 readout for one selected RIFT checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_b787_geometry_metrics import (  # noqa: E402
    chamfer,
    inside_mask,
    marching_cubes_points,
    prf,
    sample_surface_points,
    sample_volume_points,
    voxel_iou,
)
from scripts.eval_scene_geometry import deposit_points, load_energy_cloud  # noqa: E402
from scripts.render_b787_vs_stl import (  # noqa: E402
    PLANES,
    load_stl_vertices,
    stl_into_scene_frame,
    trilinear_sample_centers,
    trilinear_upsample,
)


FIXED_THRESHOLD = 0.20
EXTENT_M = 0.15
CROP_M = 0.075
BASE_GRID = 48
UPSAMPLE = 4
EXPECTED = {
    "original_rift_epoch40": {"epoch": 40, "scene_repr": "grid_sh"},
    "adaptive_rift_epoch111": {"epoch": 111, "scene_repr": "point_sh"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, choices=sorted(EXPECTED))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--stl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--extent", type=float, required=True)
    parser.add_argument("--crop", type=float, required=True)
    parser.add_argument("--base-grid", type=int, required=True)
    parser.add_argument("--upsample", type=int, required=True)
    parser.add_argument("--gt-grid", type=int, default=240)
    parser.add_argument("--n-surface", type=int, default=20_000)
    parser.add_argument("--n-volume", type=int, default=50_000)
    parser.add_argument("--tau", type=float, default=0.00625)
    parser.add_argument("--iou-unit", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def require_fixed_contract(args: argparse.Namespace) -> None:
    if args.threshold != FIXED_THRESHOLD:
        raise ValueError(f"threshold must be the declared fixed value {FIXED_THRESHOLD}")
    if args.extent != EXTENT_M or args.crop != CROP_M:
        raise ValueError("scene bounds/crop must remain fixed at +/-0.15 m and +/-0.075 m")
    if args.base_grid != BASE_GRID or args.upsample != UPSAMPLE:
        raise ValueError("readout must use a common 48^3 base grid and 4x trilinear interpolation")


def load_selected_field(
    checkpoint_path: str,
    *,
    label: str,
    extent: float,
    base_grid: int,
) -> tuple[np.ndarray, dict]:
    expected = EXPECTED[label]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    epoch = int(checkpoint.get("epoch", -1))
    scene_repr = str(checkpoint.get("scene_repr"))
    if epoch != expected["epoch"] or scene_repr != expected["scene_repr"]:
        raise ValueError(
            f"selected checkpoint mismatch: got epoch={epoch}, scene_repr={scene_repr!r}; "
            f"expected epoch={expected['epoch']}, scene_repr={expected['scene_repr']!r}"
        )
    loaded_repr, positions, energies, native = load_energy_cloud(checkpoint_path, extent)
    if loaded_repr != scene_repr:
        raise AssertionError("checkpoint scene representation changed during field loading")

    if native is not None:
        if tuple(native.shape) != (base_grid, base_grid, base_grid):
            raise ValueError(f"regular selected field is {tuple(native.shape)}, expected 48^3")
        base_energy = native.detach().cpu().numpy().astype(np.float64, copy=False)
        readout = {
            "kind": "native_regular_grid_energy",
            "active_scatterers": int(len(positions)),
        }
    else:
        base_energy = (
            deposit_points(positions, energies, extent, base_grid)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
        readout = {
            "kind": "point_sh_cic_trilinear_splat_then_regular_grid_readout",
            "active_scatterers": int(len(positions)),
            "point_energy_conserved": bool(
                np.isclose(base_energy.sum(), float(energies.sum()), rtol=1e-10, atol=1e-18)
            ),
        }
    return base_energy, {"epoch": epoch, "scene_repr": scene_repr, **readout}


def normalized_dense_magnitude(base_energy: np.ndarray, upsample: int) -> tuple[np.ndarray, dict]:
    dense_energy = trilinear_upsample(base_energy, upsample).astype(np.float64, copy=False)
    magnitude = np.sqrt(np.maximum(dense_energy, 0.0))
    lower = float(magnitude.min())
    upper = float(magnitude.max())
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise RuntimeError("selected field cannot be min-max normalized")
    normalized = (magnitude - lower) / (upper - lower)
    return normalized, {
        "native_energy_min": float(base_energy.min()),
        "native_energy_max": float(base_energy.max()),
        "dense_magnitude_min": lower,
        "dense_magnitude_max": upper,
    }


def render_fixed_threshold_views(
    normalized: np.ndarray,
    verts: np.ndarray,
    *,
    label: str,
    epoch: int,
    extent: float,
    crop: float,
    threshold: float,
    output: Path,
) -> None:
    rng = np.random.default_rng(0)
    if len(verts) > 15_000:
        verts = verts[rng.choice(len(verts), 15_000, replace=False)]
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("#080808")
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.3), constrained_layout=True)
    last_image = None
    for axis, (name, mip_axis, horizontal, vertical, hlabel, vlabel) in zip(axes, PLANES):
        mip = normalized.max(axis=mip_axis)
        remaining = [value for value in range(3) if value != mip_axis]
        image = mip.T if remaining == [horizontal, vertical] else mip
        visible = np.ma.masked_less_equal(image, threshold)
        last_image = axis.imshow(
            visible,
            origin="lower",
            extent=[-extent, extent, -extent, extent],
            cmap=cmap,
            aspect="equal",
            interpolation="bilinear",
            vmin=threshold,
            vmax=1.0,
        )
        axis.scatter(
            verts[:, horizontal],
            verts[:, vertical],
            s=0.45,
            c="cyan",
            alpha=0.13,
            linewidths=0,
            rasterized=True,
        )
        axis.set_xlim(-crop, crop)
        axis.set_ylim(-crop, crop)
        axis.set_title(name)
        axis.set_xlabel(hlabel)
        axis.set_ylabel(vlabel)
    figure.colorbar(last_image, ax=axes, label="min-max normalized |w|; fixed threshold t=0.20")
    figure.suptitle(
        f"{label} selected checkpoint, epoch {epoch}\n"
        "B787 registered support: Plenoxel-style 4x trilinear readout, fixed t=0.20",
        fontsize=13,
    )
    figure.savefig(output, dpi=160)
    plt.close(figure)


def render_fixed_threshold_scatter(
    points: np.ndarray,
    values: np.ndarray,
    verts: np.ndarray,
    *,
    label: str,
    epoch: int,
    crop: float,
    output: Path,
) -> None:
    rng = np.random.default_rng(0)
    point_idx = (
        np.arange(len(points))
        if len(points) <= 50_000
        else rng.choice(len(points), 50_000, replace=False)
    )
    vert_idx = (
        np.arange(len(verts))
        if len(verts) <= 15_000
        else rng.choice(len(verts), 15_000, replace=False)
    )
    p = points[point_idx]
    value = values[point_idx]
    truth = verts[vert_idx]
    figure = plt.figure(figsize=(8, 7))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(truth[:, 0], truth[:, 1], truth[:, 2], s=0.3, c="lightgray", alpha=0.12)
    scatter = axis.scatter(
        p[:, 0], p[:, 1], p[:, 2], s=2.0, c=value, cmap="inferno", vmin=FIXED_THRESHOLD, vmax=1.0
    )
    axis.set_xlim(-crop, crop)
    axis.set_ylim(-crop, crop)
    axis.set_zlim(-crop, crop)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.set_title(f"{label}, epoch {epoch}: fixed t=0.20 support vs registered B787 STL")
    figure.colorbar(scatter, ax=axis, shrink=0.65, label="normalized |w|")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def geometry_metrics(
    normalized: np.ndarray,
    sample_centers: np.ndarray,
    tris: np.ndarray,
    *,
    args: argparse.Namespace,
) -> tuple[dict, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    gt_surface = sample_surface_points(tris, args.n_surface, rng)
    edges = np.linspace(-args.extent, args.extent, args.gt_grid + 1)
    gt_centers = 0.5 * (edges[:-1] + edges[1:])
    occupancy = inside_mask(tris, gt_centers, gt_centers, gt_centers)
    gt_volume = sample_volume_points(
        occupancy, gt_centers, gt_centers, gt_centers, args.n_volume, rng
    )

    gx, gy, gz = np.meshgrid(sample_centers, sample_centers, sample_centers, indexing="ij")
    centers = np.stack((gx, gy, gz), axis=-1).reshape(-1, 3)
    flat = normalized.reshape(-1)
    keep = flat > args.threshold
    voxel_points = centers[keep]
    voxel_values = flat[keep]
    mesh_points = marching_cubes_points(
        normalized, sample_centers, args.threshold, args.n_surface, rng
    )

    def summarize(points: np.ndarray) -> dict:
        surface = chamfer(points, gt_surface)
        volume = chamfer(points, gt_volume)
        fscore = prf(points, gt_surface, args.tau)
        return {
            "point_count": int(len(points)),
            "cd_surface": surface["cham"],
            "cd_volume": volume["cham"],
            "surface_l2_mm": surface["l2_mm"],
            "surface_hausdorff_mm": surface["hausdorff_mm"],
            "surface_hd95_mm": surface["hd95_mm"],
            "iou_solid": voxel_iou(points, gt_volume, args.iou_unit, -args.extent, args.extent),
            "iou_shell": voxel_iou(points, gt_surface, args.iou_unit, -args.extent, args.extent),
            **fscore,
        }

    return {
        "voxel": summarize(voxel_points),
        "mesh": summarize(mesh_points),
        "truth": {
            "triangle_count": int(len(tris)),
            "surface_sample_count": int(len(gt_surface)),
            "volume_sample_count": int(len(gt_volume)),
            "solid_gt_grid": int(args.gt_grid),
            "solid_fill_fraction": float(occupancy.mean()),
        },
    }, voxel_points, voxel_values


def main() -> None:
    args = parse_args()
    require_fixed_contract(args)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"fresh output directory required: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()

    base_energy, checkpoint_info = load_selected_field(
        args.checkpoint, label=args.label, extent=args.extent, base_grid=args.base_grid
    )
    normalized, field_stats = normalized_dense_magnitude(base_energy, args.upsample)
    sample_centers = trilinear_sample_centers(
        args.extent, args.base_grid, normalized.shape[0]
    )

    with np.load(args.npz_path, allow_pickle=True) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
    verts = stl_into_scene_frame(load_stl_vertices(args.stl), metadata)
    tris = verts.reshape(-1, 3, 3)
    metrics, voxel_points, voxel_values = geometry_metrics(
        normalized, sample_centers, tris, args=args
    )

    overlay_path = output / "selected_fixed_t0p20_overlay.png"
    scatter_path = output / "selected_fixed_t0p20_support_3d.png"
    support_path = output / "selected_fixed_t0p20_support.npz"
    metrics_path = output / "fixed_t0p20_geometry_metrics.json"
    report_path = output / "evaluation_report.json"

    render_fixed_threshold_views(
        normalized,
        verts,
        label=args.label,
        epoch=checkpoint_info["epoch"],
        extent=args.extent,
        crop=args.crop,
        threshold=args.threshold,
        output=overlay_path,
    )
    render_fixed_threshold_scatter(
        voxel_points,
        voxel_values,
        verts,
        label=args.label,
        epoch=checkpoint_info["epoch"],
        crop=args.crop,
        output=scatter_path,
    )
    np.savez_compressed(
        support_path,
        points_m=voxel_points.astype(np.float32),
        normalized_magnitude=voxel_values.astype(np.float32),
        threshold=np.float64(args.threshold),
        sample_centers_m=sample_centers.astype(np.float64),
    )
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    report = {
        "schema": "rift_b787_selected_checkpoint_plenoxel_postflight_v1",
        "status": "COMPLETED_POSTFLIGHT_RENDER_AND_FIXED_THRESHOLD_GEOMETRY",
        "training_or_resume_performed": False,
        "label": args.label,
        "checkpoint": os.path.realpath(args.checkpoint),
        "checkpoint_selection": checkpoint_info,
        "readout_contract": {
            "scene_bounds_m": [[-args.extent, args.extent]] * 3,
            "display_crop_m": [[-args.crop, args.crop]] * 3,
            "base_grid": [args.base_grid] * 3,
            "dense_grid": [args.base_grid * args.upsample] * 3,
            "interpolation": "torch trilinear align_corners=True on the common energy grid",
            "normalization": "per-selected-checkpoint min-max normalized magnitude over full dense field",
            "fixed_primary_threshold": args.threshold,
            "threshold_operator": "normalized_magnitude > threshold",
            "threshold_selected_before_rendering": True,
        },
        "registration": {
            "target_position_m": metadata.get("target_position_m"),
            "scale_factor": metadata.get("scale_factor"),
            "scaled_dimensions_m": metadata.get("scaled_dimensions_m"),
            "orientation": "dataset/STL native axes retained; STL bbox centered at origin and scaled by metadata scale_factor",
            "ground_truth_used_only_for_postflight_overlay_and_metrics": True,
        },
        "field_stats": field_stats,
        "metrics": metrics,
        "artifacts": {
            "overlay_png": str(overlay_path),
            "support_3d_png": str(scatter_path),
            "fixed_threshold_support_npz": str(support_path),
            "geometry_metrics_json": str(metrics_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
