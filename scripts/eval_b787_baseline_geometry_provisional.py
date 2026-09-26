#!/usr/bin/env python
"""Provisional common geometry readout for the B787 GeRaF and RadarSplat S0 runs.

The completed GeRaF and RadarSplat S0 baselines do not serialize the regular
``grid_sh`` field consumed by :mod:`eval_b787_geometry_metrics`.  This adapter
therefore makes their *native geometry support* explicit on the same native
48^3 B787 lattice before invoking that script's unchanged threshold sweep and
metric definitions:

* GeRaF: NeuS-style logistic SDF surface density, ``logistic_sdf_pdf(SDF, s)``.
  Reflectivity is intentionally excluded: it is appearance, not geometry.
* RadarSplat S0: an opacity-weighted union of the learned anisotropic 3-D
  Gaussians, ``1 - prod_i(1 - alpha_i exp(-0.5 d_i^T Sigma_i^-1 d_i))``.
  The native noise logits and SH reflectance are intentionally excluded.

This is an interim compatibility adapter, not a claim that either method has a
final shared geometry exporter.  Its JSON manifest records the exact support
definition, checkpoint, and evaluation protocol so slide rows can be clearly
labelled provisional.

Run this on an allocated PACE compute node, never a login node::

    python scripts/eval_b787_baseline_geometry_provisional.py \\
        --out-dir /storage/scratch1/1/dbao31/rift_power_baselines_20260813_v2/\
evaluation/provisional_geometry_20260821
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # ``python scripts/<file>.py`` puts ``scripts/`` rather than the project
    # root on sys.path; the established scorer is imported as ``scripts.*``.
    sys.path.insert(0, str(PROJECT_ROOT))


POWER_ROOT = "/storage/scratch1/1/dbao31/rift_power_baselines_20260813_v2"
GERAF_IMPL_ROOT = "/storage/scratch1/1/dbao31/rift_power_baselines_impl_20260813_v2"
DEFAULT_GERAF_CKPT = os.path.join(
    POWER_ROOT, "g0_geraf", "checkpoints", "checkpoint_final.pth.tar"
)
DEFAULT_RADARSPLAT_CKPT = os.path.join(
    POWER_ROOT, "s0_radarsplat", "checkpoints", "checkpoint_final.pth.tar"
)


def voxel_centers(granularity: int, extent: float) -> np.ndarray:
    """Return the native B787 voxel-centre lattice as ``[G, G, G, 3]``."""
    edges = np.linspace(-extent, extent, granularity + 1, dtype=np.float64)
    axis = 0.5 * (edges[:-1] + edges[1:])
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)


def quaternion_to_rotation_matrix(quaternions: np.ndarray) -> np.ndarray:
    """Return WXYZ quaternion rotations used by the native RadarSplat model.

    This is algebraically the same conversion as
    ``rift.radarsplat.quaternion_to_rotation_matrix`` in the implementation
    tree, kept locally to make the extraction independent of its import path.
    """
    q = np.asarray(quaternions, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"expected [N, 4] WXYZ quaternions, got {q.shape}")
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    if np.any(norm <= 0):
        raise ValueError("RadarSplat checkpoint contains a zero quaternion")
    w, x, y, z = (q / norm).T
    out = np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        axis=1,
    )
    return out.reshape(-1, 3, 3)


def covariance_from_quaternion_scale(
    quaternions: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    """Construct ``R diag(scales**2) R.T`` for each native Gaussian."""
    scales = np.asarray(scales, dtype=np.float64)
    if scales.ndim != 2 or scales.shape[1] != 3:
        raise ValueError(f"expected [N, 3] scales, got {scales.shape}")
    if np.any(scales <= 0):
        raise ValueError("RadarSplat checkpoint contains a non-positive scale")
    rotation = quaternion_to_rotation_matrix(quaternions)
    if rotation.shape[0] != scales.shape[0]:
        raise ValueError("Gaussian quaternion and scale counts differ")
    transform = rotation * scales[:, None, :]
    return transform @ np.swapaxes(transform, 1, 2)


def rasterize_gaussian_occupancy_union(
    means: np.ndarray,
    scales: np.ndarray,
    quaternions: np.ndarray,
    opacity: np.ndarray,
    *,
    granularity: int,
    extent: float,
    sigma_radius: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rasterize the native RadarSplat geometry support on the common lattice.

    Each local stencil is evaluated exactly under its anisotropic covariance;
    restricting evaluation to ``sigma_radius`` only drops a bounded Gaussian
    tail.  Combining ``log(1 - alpha)`` is numerically stable and implements
    the requested occupancy union without allocating a [G^3, N] tensor.
    """
    means = np.asarray(means, dtype=np.float64)
    opacity = np.asarray(opacity, dtype=np.float64).reshape(-1)
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError(f"expected [N, 3] Gaussian means, got {means.shape}")
    if not (len(means) == len(scales) == len(quaternions) == len(opacity)):
        raise ValueError("RadarSplat Gaussian parameter counts differ")
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(opacity)):
        raise ValueError("RadarSplat checkpoint has non-finite geometry parameters")
    if sigma_radius <= 0:
        raise ValueError("sigma_radius must be positive")

    covariance = covariance_from_quaternion_scale(quaternions, scales)
    std_axis = np.sqrt(np.clip(np.diagonal(covariance, axis1=1, axis2=2), 0, None))
    pitch = 2.0 * extent / granularity
    axis = -extent + (np.arange(granularity, dtype=np.float64) + 0.5) * pitch
    log_empty = np.zeros((granularity, granularity, granularity), dtype=np.float64)
    touched_gaussians = 0
    updated_cells = 0

    for index, (mean, cov, radius_axis, alpha) in enumerate(
        zip(means, covariance, sigma_radius * std_axis, opacity)
    ):
        if alpha <= 0:
            continue
        # Coordinates i satisfying -extent + (i + .5) * pitch in [lo, hi].
        lo = np.ceil((mean - radius_axis + extent) / pitch - 0.5).astype(int)
        hi = np.floor((mean + radius_axis + extent) / pitch - 0.5).astype(int)
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, granularity - 1)
        if np.any(lo > hi):
            continue
        precision = np.linalg.inv(cov)
        xx, yy, zz = np.meshgrid(
            axis[lo[0]:hi[0] + 1],
            axis[lo[1]:hi[1] + 1],
            axis[lo[2]:hi[2] + 1],
            indexing="ij",
        )
        delta = np.stack((xx - mean[0], yy - mean[1], zz - mean[2]), axis=-1)
        mahal = np.einsum("...i,ij,...j->...", delta, precision, delta)
        contribution = alpha * np.exp(-0.5 * mahal)
        contribution[mahal > sigma_radius * sigma_radius] = 0.0
        contribution = np.clip(contribution, 0.0, 1.0 - 1.0e-15)
        if not np.any(contribution):
            continue
        region = np.s_[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
        log_empty[region] += np.log1p(-contribution)
        touched_gaussians += 1
        updated_cells += int(np.count_nonzero(contribution))

    field = -np.expm1(log_empty)
    stats = {
        "gaussians_total": int(len(means)),
        "gaussians_with_support_in_box": int(touched_gaussians),
        "stencil_cells_updated_before_union": int(updated_cells),
        "opacity_min": float(opacity.min()),
        "opacity_max": float(opacity.max()),
        "sigma_radius": float(sigma_radius),
        "native_geometry_definition": (
            "1 - product_i(1 - sigmoid(opacity_i) * exp(-0.5 * "
            "(x-mu_i)^T Sigma_i^-1 (x-mu_i)))"
        ),
    }
    return field, stats


def _external_geraf_module(implementation_root: str):
    module_path = Path(implementation_root) / "rift" / "geraf.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"GeRaF implementation module is absent: {module_path}")
    spec = importlib.util.spec_from_file_location("_rift_external_geraf", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load GeRaF implementation from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_geraf_surface_density(
    checkpoint_path: str,
    *,
    implementation_root: str,
    granularity: int,
    extent: float,
    chunk: int,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample GeRaF's SDF-only surface density at the B787 g48 cell centres."""
    import torch

    geraf = _external_geraf_module(implementation_root)
    target_device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=False)
    cfg, state = checkpoint["model_config"], checkpoint["model_state_dict"]
    model = geraf.GeRaFSDFNetwork(
        extent=cfg["extent"],
        n_levels=cfg["sdf_levels"],
        hidden_dim=cfg["sdf_hidden_dim"],
        n_layers=cfg["sdf_layers"],
        skip_layer=cfg["sdf_skip_layer"],
        hidden_activation=cfg["sdf_hidden_activation"],
        softplus_beta=cfg["sdf_softplus_beta"],
        encoding_include_input=cfg["sdf_encoding_include_input"],
        encoding_coordinate_scale=cfg["sdf_encoding_coordinate_scale"],
    ).to(target_device)
    model.load_state_dict(
        {key[len("sdf_network."):]: value for key, value in state.items()
         if key.startswith("sdf_network.")}
    )
    model.eval()
    inv_s = float(torch.exp(state["sdf_sharpness.log_inv_s"]).detach().cpu())
    points = torch.from_numpy(voxel_centers(granularity, extent).reshape(-1, 3)).to(
        device=target_device, dtype=torch.float32
    )
    values = np.empty(len(points), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(points), chunk):
            block = points[start:start + chunk]
            values[start:start + len(block)] = (
                geraf.logistic_sdf_pdf(model(block), inv_s).detach().double().cpu().numpy()
            )
    field = values.reshape(granularity, granularity, granularity)
    stats = {
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "inv_s": inv_s,
        "native_geometry_definition": "logistic_sdf_pdf(SDF(x), inv_s)",
        "reflectivity_used": False,
    }
    return field, stats


def load_radarsplat_occupancy(
    checkpoint_path: str,
    *,
    granularity: int,
    extent: float,
    sigma_radius: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read native RadarSplat Gaussian geometry without its appearance terms."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    field, stats = rasterize_gaussian_occupancy_union(
        state["means"].detach().cpu().numpy(),
        np.exp(state["log_scales"].detach().cpu().numpy()),
        state["quaternions"].detach().cpu().numpy(),
        torch.sigmoid(state["opacity_logits"]).detach().cpu().numpy(),
        granularity=granularity,
        extent=extent,
        sigma_radius=sigma_radius,
    )
    stats["checkpoint_step"] = int(checkpoint.get("step", -1))
    stats["noise_probability_used"] = False
    stats["sh_reflectance_used"] = False
    return field, stats


def _metric_api():
    """Load the established scorer lazily so ``--self-test`` stays lightweight."""
    from scripts.eval_b787_geometry_metrics import (  # pylint: disable=import-outside-toplevel
        DEFAULT_THRESHOLDS,
        chamfer,
        inside_mask,
        load_stl_vertices,
        predicted_points,
        prf,
        sample_surface_points,
        sample_volume_points,
        stl_into_scene_frame,
        voxel_iou,
    )
    return {
        "DEFAULT_THRESHOLDS": DEFAULT_THRESHOLDS,
        "chamfer": chamfer,
        "inside_mask": inside_mask,
        "load_stl_vertices": load_stl_vertices,
        "predicted_points": predicted_points,
        "prf": prf,
        "sample_surface_points": sample_surface_points,
        "sample_volume_points": sample_volume_points,
        "stl_into_scene_frame": stl_into_scene_frame,
        "voxel_iou": voxel_iou,
    }


def make_truth(args: argparse.Namespace, api: dict[str, Any], rng: np.random.Generator):
    metadata = json.loads(str(np.load(args.npz_path, allow_pickle=True, mmap_mode="r")["metadata_json"]))
    tris = api["stl_into_scene_frame"](api["load_stl_vertices"](args.stl), metadata).reshape(-1, 3, 3)
    gt_surface = api["sample_surface_points"](tris, args.n_surface, rng)
    grid_edges = np.linspace(-args.extent, args.extent, args.gt_grid + 1)
    grid_centres = 0.5 * (grid_edges[:-1] + grid_edges[1:])
    occupancy = api["inside_mask"](tris, grid_centres, grid_centres, grid_centres)
    gt_volume = api["sample_volume_points"](
        occupancy, grid_centres, grid_centres, grid_centres, args.n_volume, rng
    )
    return tris, gt_surface, gt_volume, occupancy


def evaluate_field(
    field: np.ndarray,
    *,
    method: str,
    source: str,
    args: argparse.Namespace,
    api: dict[str, Any],
    gt_surface: np.ndarray,
    gt_volume: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """Apply the exact voxel/oracle procedure used on slide 12."""
    field = np.asarray(field, dtype=np.float64)
    if field.shape != (args.granularity,) * 3 or not np.all(np.isfinite(field)):
        raise ValueError(f"{method}: invalid support field {field.shape}")
    span = float(field.max() - field.min())
    if not span > 0:
        raise ValueError(f"{method}: support field is constant; cannot threshold it")
    normalized = (field - field.min()) / span
    centres = voxel_centers(args.granularity, args.extent).reshape(-1, 3)
    rows: list[dict[str, Any]] = []
    for threshold in args.thresholds:
        pred = api["predicted_points"](normalized, centres, threshold)
        surface = api["chamfer"](pred, gt_surface)
        shape = api["prf"](pred, gt_surface, args.tau)
        rows.append({
            "run": method,
            "source": source,
            "variant": "voxel",
            "granularity": int(args.granularity),
            "threshold": float(threshold),
            "n_points": int(len(pred)),
            "cd_surface": float(surface["cham"]),
            "hausdorff_mm": float(surface["hausdorff_mm"]),
            "hd95_mm": float(surface["hd95_mm"]),
            "iou_solid": float(api["voxel_iou"](
                pred, gt_volume, args.iou_unit, -args.extent, args.extent
            )),
            "precision": float(shape["precision"]),
            "recall": float(shape["recall"]),
            "f1": float(shape["f1"]),
        })
    valid = [row for row in rows if row["n_points"] >= args.min_points]
    if not valid:
        raise RuntimeError(f"{method}: no threshold retains {args.min_points} points")
    selectors = {
        "Chamfer": ("cd_surface", min),
        "Hausdorff": ("hausdorff_mm", min),
        "HD95": ("hd95_mm", min),
        "IoU": ("iou_solid", max),
        "F1": ("f1", max),
    }
    oracle = {}
    for name, (metric, select) in selectors.items():
        best = select(valid, key=lambda row: row[metric])
        oracle[name] = {
            "metric": metric,
            "value": float(best[metric]),
            "threshold": float(best["threshold"]),
            "n_points": int(best["n_points"]),
        }
    return rows, oracle


def format_oracle(oracle: dict[str, dict[str, float]]) -> dict[str, str]:
    """Format table cells without obscuring the per-metric selected threshold."""
    return {
        "Chamfer": f"{oracle['Chamfer']['value']:.4e} @{oracle['Chamfer']['threshold']:.2f}",
        "Hausdorff": f"{oracle['Hausdorff']['value']:.3f} @{oracle['Hausdorff']['threshold']:.2f}",
        "HD95": f"{oracle['HD95']['value']:.3f} @{oracle['HD95']['threshold']:.2f}",
        "IoU": f"{oracle['IoU']['value']:.4f} @{oracle['IoU']['threshold']:.2f}",
        "F1": f"{oracle['F1']['value']:.4f} @{oracle['F1']['threshold']:.2f}",
    }


def is_scheduler_log_name(name: str) -> bool:
    """Identify Slurm's two scheduler-owned output filename forms."""
    return name.startswith("slurm-") and Path(name).suffix in {".out", ".err"}


def output_contains_prior_results(output: Path) -> bool:
    """Return whether an output directory holds artifacts beyond Slurm logs.

    Slurm opens the configured ``slurm-<job>.out`` and ``.err`` files before
    Python starts. Those scheduler-owned files are safe to coexist with a new
    evaluation, whereas any other entry means a prior evaluator may have
    produced result artifacts that must not be overwritten accidentally.
    """
    for entry in output.iterdir():
        if entry.is_file() and is_scheduler_log_name(entry.name):
            continue
        return True
    return False


def self_test() -> None:
    """Exercise the native Gaussian rasterizer without checkpoints or SciPy."""
    rotation = quaternion_to_rotation_matrix(np.array([[1.0, 0.0, 0.0, 0.0]]))
    if not np.allclose(rotation[0], np.eye(3)):
        raise AssertionError("identity quaternion conversion failed")
    field, stats = rasterize_gaussian_occupancy_union(
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[0.01, 0.01, 0.01]]),
        np.array([[1.0, 0.0, 0.0, 0.0]]),
        np.array([0.5]),
        granularity=16,
        extent=0.15,
        sigma_radius=3.0,
    )
    if not (field.max() > 0 and np.all((field >= 0) & (field < 1))):
        raise AssertionError("Gaussian occupancy union is out of range")
    if stats["gaussians_with_support_in_box"] != 1:
        raise AssertionError("Gaussian support accounting failed")
    if not is_scheduler_log_name("slurm-123.out"):
        raise AssertionError("Slurm stdout was not recognized")
    if not is_scheduler_log_name("slurm-123.err"):
        raise AssertionError("Slurm stderr was not recognized")
    if is_scheduler_log_name("geometry_provisional_summary.json"):
        raise AssertionError("result artifacts must not be treated as scheduler logs")
    print("self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=False)
    parser.add_argument("--geraf-ckpt", default=DEFAULT_GERAF_CKPT)
    parser.add_argument("--radarsplat-ckpt", default=DEFAULT_RADARSPLAT_CKPT)
    parser.add_argument("--geraf-impl-root", default=GERAF_IMPL_ROOT)
    parser.add_argument("--npz-path", default="data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere2k.npz")
    parser.add_argument("--stl", default="data/B787.stl")
    parser.add_argument("--extent", type=float, default=0.15)
    parser.add_argument("--granularity", type=int, default=48)
    parser.add_argument("--thresholds", type=float, nargs="+", default=None)
    parser.add_argument("--tau", type=float, default=0.00625)
    parser.add_argument("--iou-unit", type=float, default=0.005)
    parser.add_argument("--n-surface", type=int, default=20000)
    parser.add_argument("--n-volume", type=int, default=50000)
    parser.add_argument("--gt-grid", type=int, default=240)
    parser.add_argument("--min-points", type=int, default=20)
    parser.add_argument("--geraf-chunk", type=int, default=16384)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gaussian-sigma-radius", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return args
    if not args.out_dir:
        parser.error("--out-dir is required unless --self-test is used")
    if args.granularity <= 1 or args.gt_grid <= 1:
        parser.error("--granularity and --gt-grid must be greater than one")
    return args


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    api = _metric_api()
    if args.thresholds is None:
        args.thresholds = list(api["DEFAULT_THRESHOLDS"])
    output = Path(args.out_dir)
    if output.exists() and output_contains_prior_results(output) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    tris, gt_surface, gt_volume, gt_occupancy = make_truth(args, api, rng)
    print(
        f"truth: {len(tris)} triangles, solid fill {gt_occupancy.mean() * 100:.2f}% "
        f"of {args.gt_grid}^3; scoring native g{args.granularity}",
        flush=True,
    )
    fields = []
    geraf_field, geraf_stats = load_geraf_surface_density(
        args.geraf_ckpt,
        implementation_root=args.geraf_impl_root,
        granularity=args.granularity,
        extent=args.extent,
        chunk=args.geraf_chunk,
        device=args.device,
    )
    np.save(output / "geraf_sdf_surface_density_g48.npy", geraf_field)
    fields.append(("GeRaF · provisional", "GeRaF SDF surface density", geraf_field, geraf_stats))

    radarsplat_field, radarsplat_stats = load_radarsplat_occupancy(
        args.radarsplat_ckpt,
        granularity=args.granularity,
        extent=args.extent,
        sigma_radius=args.gaussian_sigma_radius,
    )
    np.save(output / "radarsplat_s0_gaussian_occupancy_g48.npy", radarsplat_field)
    fields.append(("RadarSplat S0 · provisional", "RadarSplat Gaussian occupancy", radarsplat_field, radarsplat_stats))

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for name, source, field, stats in fields:
        print(f"\n=== {name} ===", flush=True)
        method_rows, oracle = evaluate_field(
            field, method=name, source=source, args=args, api=api,
            gt_surface=gt_surface, gt_volume=gt_volume,
        )
        rows.extend(method_rows)
        summary[name] = {
            "source": source,
            "field_min": float(field.min()),
            "field_max": float(field.max()),
            "field_stats": stats,
            "oracle": oracle,
            "table_cells": format_oracle(oracle),
        }
        for metric, value in summary[name]["table_cells"].items():
            print(f"  {metric:10s} {value}", flush=True)

    with (output / "geometry_provisional_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "status": "PROVISIONAL — method-native support adapters pending shared-evaluator validation",
        "protocol": {
            "source": "scripts/eval_b787_geometry_metrics.py",
            "prediction": "native g48 voxel-centre support, min-max normalized, threshold sweep",
            "thresholds": args.thresholds,
            "tau_m": args.tau,
            "iou_unit_m": args.iou_unit,
            "n_surface": args.n_surface,
            "n_volume": args.n_volume,
            "gt_grid": args.gt_grid,
            "min_points": args.min_points,
            "seed": args.seed,
        },
        "inputs": {
            "geraf_checkpoint": args.geraf_ckpt,
            "radarsplat_checkpoint": args.radarsplat_ckpt,
            "b787_npz": args.npz_path,
            "b787_stl": args.stl,
        },
        "results": summary,
    }
    (output / "geometry_provisional_summary.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
