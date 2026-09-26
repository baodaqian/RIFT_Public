#!/usr/bin/env python
"""Snapshot and score current Mesh10k baseline geometry for result slides.

The script reads the seven methods that expose a three-dimensional scene
representation, renders top/front maximum-intensity projections on a common
48^3 lattice, and applies the existing per-metric oracle-threshold protocol.
It is evaluation-only: no ground-truth mesh enters any training path.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENT_ROOT = Path(
    "/storage/scratch1/1/dbao31/rift_mesh10k_baselines_20260830_v1"
)
POWER_IMPL_ROOT = Path(
    "/storage/scratch1/1/dbao31/rift_power_baselines_impl_20260813_v2"
)

SCENE_EXPECTATIONS = {
    "a320": {"triangles": 9096, "dimensions": [35.7999992371, 34.0197334290, 11.0133180618]},
    "x59": {"triangles": 175542, "dimensions": [287.9863891602, 89.9183197021, 103.7358093262]},
    "firetruck": {"triangles": 2767, "dimensions": [1.5, 1.7000000477, 3.4000000954]},
    "racecar": {"triangles": 2068, "dimensions": [1.2000000477, 0.8325443268, 2.6597719193]},
    "loader": {"triangles": 2646, "dimensions": [1.6605600119, 1.5120249987, 2.4700620174]},
}

DISPLAY_NAMES = {
    "rift": "RIFT (ours)",
    "sp0": "SpINR-style",
    "rf": "Radar Fields",
    "mfbp": "Matched-filter BP",
    "geraf": "GeRaF",
    "rs": "RadarSplat S0",
    "se": "SE Stage-1 scattering-grid proxy (diagnostic; not SDF surface)",
}


def load_metadata(npz_path: Path) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=True, mmap_mode="r") as archive:
        raw = archive["metadata_json"].reshape(-1)[0]
    return json.loads(str(raw))


def _load_binary_stl(path: Path) -> np.ndarray | None:
    size = path.stat().st_size
    if size < 84:
        return None
    with path.open("rb") as handle:
        header = handle.read(80)
        count_raw = handle.read(4)
        if len(count_raw) != 4:
            return None
        count = struct.unpack("<I", count_raw)[0]
        if 84 + 50 * count != size:
            return None
        records = handle.read(count * 50)
    if len(records) != count * 50:
        raise ValueError(f"truncated binary STL: {path}")
    blocks = np.frombuffer(records, dtype=np.uint8).reshape(count, 50)
    return np.frombuffer(blocks[:, 12:48].tobytes(), dtype="<f4").reshape(-1, 3, 3).astype(np.float64)


def _load_ascii_stl(path: Path) -> np.ndarray:
    vertices: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            fields = line.strip().split()
            if len(fields) == 4 and fields[0].lower() == "vertex":
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
    if not vertices or len(vertices) % 3:
        raise ValueError(f"invalid ASCII STL vertex stream: {path}")
    return np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)


def _load_obj(path: Path) -> np.ndarray:
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                fields = line.split()
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
            elif line.startswith("f "):
                tokens = [item.split("/", 1)[0] for item in line.split()[1:]]
                face = []
                for token in tokens:
                    index = int(token)
                    face.append(index - 1 if index > 0 else len(vertices) + index)
                for offset in range(1, len(face) - 1):
                    triangles.append([face[0], face[offset], face[offset + 1]])
    verts = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"invalid OBJ mesh: {path}")
    return verts[faces]


def load_mesh_triangles(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return _load_obj(path)
    if suffix == ".stl":
        binary = _load_binary_stl(path)
        return binary if binary is not None else _load_ascii_stl(path)
    raise ValueError(f"unsupported mesh type: {path}")


def validate_and_transform_mesh(
    triangles: np.ndarray, metadata: dict[str, Any], scene: str
) -> np.ndarray:
    expected = SCENE_EXPECTATIONS[scene]
    if len(triangles) != expected["triangles"]:
        raise ValueError(
            f"{scene}: parsed {len(triangles)} triangles, expected {expected['triangles']}"
        )
    flat = triangles.reshape(-1, 3)
    dimensions = flat.max(axis=0) - flat.min(axis=0)
    if not np.allclose(dimensions, expected["dimensions"], rtol=2e-6, atol=2e-6):
        raise ValueError(f"{scene}: mesh dimensions {dimensions.tolist()} do not match metadata")
    center = 0.5 * (flat.max(axis=0) + flat.min(axis=0))
    scale = float(metadata["scale_factor"])
    transformed = (triangles - center[None, None, :]) * scale
    scaled_dimensions = np.ptp(transformed.reshape(-1, 3), axis=0)
    if not np.allclose(scaled_dimensions, metadata["scaled_dimensions_m"], rtol=2e-6, atol=2e-7):
        raise ValueError(f"{scene}: transformed mesh does not match archived dimensions")
    return transformed


def load_grid_energy(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    real = state["w_re"].detach().cpu()
    imag = state["w_im"].detach().cpu()
    energy = real.square() + imag.square()
    if energy.ndim == 4:
        energy = energy.sum(dim=-1)
    elif energy.ndim != 3:
        raise ValueError(f"unexpected grid coefficient shape: {tuple(real.shape)}")
    energy = energy.numpy().astype(np.float64)
    return energy, {
        "checkpoint_epoch": int(checkpoint.get("epoch", checkpoint.get("step", -1))),
        "checkpoint_loss": float(checkpoint.get("loss", checkpoint.get("best_val_rel_mse", math.nan))),
    }


def load_mfbp_energy(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path) as archive:
        field = archive["complex_adjoint"]
    grid = int(round(field.size ** (1.0 / 3.0)))
    if grid ** 3 != field.size:
        raise ValueError(f"MF-BP field is not cubic: {field.shape}")
    return np.abs(field.reshape(grid, grid, grid)).astype(np.float64) ** 2, {"checkpoint_epoch": 3200}


def history_last_relative(path: Path) -> tuple[int, float]:
    if not path.is_file():
        return -1, math.nan
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates: list[dict[str, Any]] = []
    for value in payload.values() if isinstance(payload, dict) else []:
        if isinstance(value, list) and value and isinstance(value[-1], dict):
            if "mf_magnitude_relative_mse" in value[-1]:
                candidates = value
                break
    if not candidates:
        return -1, math.nan
    record = candidates[-1]
    return int(record.get("step", -1)), float(record["mf_magnitude_relative_mse"])


def method_status(scene_root: Path, method: str) -> str:
    if method in {"rift", "sp0"}:
        return "finished" if (scene_root / method / "model" / "checkpoint_final.pth.tar").is_file() else "in progress"
    if method == "rf":
        return "finished" if (scene_root / method / "model" / "checkpoint_final.pth.tar").is_file() else "in progress"
    if method == "mfbp":
        status_path = scene_root / method / "status.json"
        if status_path.is_file() and json.loads(status_path.read_text(encoding="utf-8")).get("state") == "complete":
            return "finished"
        return "in progress"
    if method == "rs":
        status_path = scene_root / method / "checkpoints" / "status.json"
        if status_path.is_file() and json.loads(status_path.read_text(encoding="utf-8")).get("done") is True:
            return "finished"
        return "in progress"
    if method == "geraf":
        step, _ = history_last_relative(scene_root / method / "checkpoints" / "history.json")
        return "finished" if step >= 50000 else "in progress"
    if method == "se":
        return "finished" if (scene_root / method / "sdf" / "surface_reconstruction.npz").is_file() else "in progress"
    raise KeyError(method)


def load_fields(scene: str, implementation_root: Path) -> list[dict[str, Any]]:
    from scripts.eval_b787_baseline_geometry_provisional import (
        load_geraf_surface_density,
        load_radarsplat_occupancy,
    )

    scene_root = EXPERIMENT_ROOT / scene
    specs = [
        ("rift", scene_root / "rift/model/checkpoint_best.pth.tar", "grid"),
        ("sp0", scene_root / "sp0/model/checkpoint_best.pth.tar", "grid"),
        ("rf", scene_root / "rf/model/checkpoint_final.pth.tar", "grid"),
        ("mfbp", scene_root / "mfbp/matched_filter.npz", "mfbp"),
        ("geraf", scene_root / "geraf/checkpoints/checkpoint_best.pth.tar", "geraf"),
        ("rs", scene_root / "rs/checkpoints/checkpoint_best.pth.tar", "rs"),
        ("se", scene_root / "se/scatter/checkpoint_best.pth.tar", "grid"),
    ]
    output = []
    for method, path, kind in specs:
        record: dict[str, Any] = {
            "method": method,
            "display_name": DISPLAY_NAMES[method],
            "status": method_status(scene_root, method),
            "source": str(path),
        }
        if method == "se":
            record.update(
                {
                    "status": (
                        "diagnostic Stage-1 available"
                        if path.is_file()
                        else "diagnostic Stage-1 unavailable"
                    ),
                    "artifact_variant": "stage1_scattering_grid_proxy",
                    "reportable_method_result": False,
                    "result_note": (
                        "diagnostic Stage-1 proxy only; not a raw, calibrated, or "
                        "valid-zero Sugavanam–Ertin SDF surface"
                    ),
                }
            )
        else:
            record["reportable_method_result"] = True
        if not path.is_file():
            record["error"] = "checkpoint not available"
            output.append(record)
            continue
        try:
            if kind == "grid":
                field, details = load_grid_energy(path)
            elif kind == "mfbp":
                field, details = load_mfbp_energy(path)
            elif kind == "geraf":
                field, details = load_geraf_surface_density(
                    str(path), implementation_root=str(implementation_root),
                    granularity=48, extent=0.15, chunk=16384, device="cpu"
                )
                step, relative = history_last_relative(scene_root / "geraf/checkpoints/history.json")
                details.update({"checkpoint_epoch": step, "native_relative_mse": relative})
            elif kind == "rs":
                field, details = load_radarsplat_occupancy(
                    str(path), granularity=48, extent=0.15, sigma_radius=3.0
                )
                status = json.loads((scene_root / "rs/checkpoints/status.json").read_text(encoding="utf-8"))
                details["native_relative_mse"] = float(status["best_val_relative_mse"])
            else:
                raise AssertionError(kind)
            if field.shape != (48, 48, 48) or not np.isfinite(field).all():
                raise ValueError(f"invalid field shape/range: {field.shape}")
            record.update(details)
            record["field"] = field
        except Exception as error:  # Preserve other methods if one adapter cannot load.
            record["error"] = f"{type(error).__name__}: {error}"
        output.append(record)
    return output


def render_comparison(
    records: list[dict[str, Any]], triangles: np.ndarray, scene_label: str, output_path: Path
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.render_b787_vs_stl import trilinear_upsample

    flat_truth = triangles.reshape(-1, 3)
    if len(flat_truth) > 18000:
        pick = np.random.default_rng(0).choice(len(flat_truth), 18000, replace=False)
        flat_truth = flat_truth[pick]
    dimensions = np.ptp(flat_truth, axis=0)
    length_axis = int(np.argmax(dimensions))
    vertical_axis = int(np.argmin(dimensions))
    width_axis = int(({0, 1, 2} - {length_axis, vertical_axis}).pop())
    views = [
        ("Top view", vertical_axis, length_axis, width_axis),
        ("Front view", length_axis, width_axis, vertical_axis),
    ]

    figure, axes = plt.subplots(2, len(records), figsize=(18.2, 5.7), squeeze=False)
    figure.patch.set_facecolor("#fcfcfb")
    for column, record in enumerate(records):
        field = record.get("field")
        if field is None:
            for row, (view_name, _mip, _horizontal, _vertical) in enumerate(views):
                axis = axes[row, column]
                axis.set_facecolor("#efeee9")
                axis.text(0.5, 0.58, "IN PROGRESS", ha="center", va="center", fontsize=10,
                          weight="bold", color="#8b1e1e", transform=axis.transAxes)
                axis.text(0.5, 0.39, record.get("error", "checkpoint not available"),
                          ha="center", va="center", fontsize=6.5, color="#55524c",
                          wrap=True, transform=axis.transAxes)
                axis.set_xticks([]); axis.set_yticks([])
                if column == 0:
                    axis.set_ylabel(view_name, fontsize=10, color="#52514e")
            axes[0, column].set_title(record["display_name"], fontsize=8.3, pad=6)
            continue

        dense = trilinear_upsample(field, 4)
        for row, (view_name, mip_axis, horizontal_axis, vertical_axis) in enumerate(views):
            axis = axes[row, column]
            image = dense.max(axis=mip_axis)
            remaining = [index for index in range(3) if index != mip_axis]
            if remaining == [horizontal_axis, vertical_axis]:
                image = image.T
            display = np.sqrt(np.maximum(image, 0.0))
            vmin = float(np.percentile(display, 88.0))
            vmax = float(np.percentile(display, 99.7))
            if not vmax > vmin:
                vmin, vmax = float(display.min()), float(display.max() + 1e-12)
            axis.imshow(display, origin="lower", extent=[-0.15, 0.15, -0.15, 0.15],
                        cmap="inferno", interpolation="bilinear", vmin=vmin, vmax=vmax)
            axis.scatter(flat_truth[:, horizontal_axis], flat_truth[:, vertical_axis],
                         s=0.18, c="#00e6e6", alpha=0.16, linewidths=0, rasterized=True)
            axis.set_xlim(-0.075, 0.075); axis.set_ylim(-0.075, 0.075)
            axis.set_xticks([]); axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(view_name, fontsize=10, color="#52514e")

        status = record["status"]
        epoch = record.get("checkpoint_epoch", -1)
        metric = record.get("checkpoint_loss", record.get("native_relative_mse", math.nan))
        metric_text = f"{100.0 * metric:.3f}%" if math.isfinite(metric) else "geometry readout"
        axes[0, column].set_title(
            f"{record['display_name']}\n{status} · {('ep/step ' + str(epoch)) if epoch >= 0 else metric_text}\n{metric_text}",
            fontsize=7.7, color=("#8b1e1e" if status == "in progress" else "#111111"), pad=5,
        )

    figure.suptitle(
        f"{scene_label} learned 3-D fields — current checkpoints on a common readout vs mesh truth (cyan)",
        fontsize=12.4, color="#111111", y=0.995,
    )
    figure.text(
        0.5, 0.012,
        "Visualization only: 48³ native fields, 4× trilinear display interpolation, method-normalized contrast. "
        "Finite-SH is excluded because it predicts signals but has no 3-D scene field.",
        ha="center", va="bottom", fontsize=7.4, color="#66635e",
    )
    figure.tight_layout(rect=[0, 0.035, 1, 0.925], w_pad=0.35, h_pad=0.35)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def evaluate_geometry(
    records: list[dict[str, Any]], triangles: np.ndarray, args: argparse.Namespace
) -> list[dict[str, Any]]:
    from scripts.eval_b787_geometry_metrics import (
        DEFAULT_THRESHOLDS,
        chamfer,
        inside_mask,
        predicted_points,
        prf,
        sample_surface_points,
        sample_volume_points,
        voxel_iou,
    )

    rng = np.random.default_rng(args.seed)
    gt_surface = sample_surface_points(triangles, args.n_surface, rng)
    grid_edges = np.linspace(-args.extent, args.extent, args.gt_grid + 1)
    grid_centres = 0.5 * (grid_edges[:-1] + grid_edges[1:])
    occupancy = inside_mask(triangles, grid_centres, grid_centres, grid_centres)
    gt_volume = sample_volume_points(occupancy, grid_centres, grid_centres, grid_centres,
                                     args.n_volume, rng)
    centres_1d = np.linspace(
        -args.extent + args.extent / 48.0,
        args.extent - args.extent / 48.0,
        48,
    )
    centres = np.stack(np.meshgrid(centres_1d, centres_1d, centres_1d, indexing="ij"), -1).reshape(-1, 3)

    summary = []
    for record in records:
        row = {
            key: value for key, value in record.items()
            if key not in {"field"}
        }
        field = record.get("field")
        if field is None:
            row["geometry_error"] = record.get("error", "field unavailable")
            summary.append(row)
            continue
        span = float(field.max() - field.min())
        if not span > 0:
            row["geometry_error"] = "field is constant"
            summary.append(row)
            continue
        normalized = (field - field.min()) / span
        candidates = []
        for threshold in DEFAULT_THRESHOLDS:
            predicted = predicted_points(normalized, centres, threshold)
            if len(predicted) < args.min_points:
                continue
            surface = chamfer(predicted, gt_surface)
            shape = prf(predicted, gt_surface, args.tau)
            candidates.append({
                "threshold": float(threshold),
                "n_points": int(len(predicted)),
                "chamfer": float(surface["cham"]),
                "hausdorff_mm": float(surface["hausdorff_mm"]),
                "hd95_mm": float(surface["hd95_mm"]),
                "iou": float(voxel_iou(predicted, gt_volume, args.iou_unit,
                                       -args.extent, args.extent)),
                "f1": float(shape["f1"]),
            })
        if not candidates:
            row["geometry_error"] = "no threshold retained enough points"
            summary.append(row)
            continue
        selectors = {
            "chamfer": min,
            "hausdorff_mm": min,
            "hd95_mm": min,
            "iou": max,
            "f1": max,
        }
        oracle = {}
        for metric, selector in selectors.items():
            best = selector(candidates, key=lambda item, name=metric: item[name])
            oracle[metric] = {
                "value": float(best[metric]),
                "threshold": float(best["threshold"]),
                "n_points": int(best["n_points"]),
            }
        if record.get("reportable_method_result") is False:
            row["diagnostic_oracle"] = oracle
        else:
            row["oracle"] = oracle
        summary.append(row)
    return summary


def self_test(asset_root: Path) -> None:
    asset_map = {
        "a320": asset_root / "A320_OpenVSP.stl",
        "x59": asset_root / "nasa_x59.stl",
        "firetruck": asset_root / "kenney_firetruck.obj",
        "racecar": asset_root / "kenney_race_future.obj",
        "loader": asset_root / "kenney_tractor_shovel.obj",
    }
    for scene, path in asset_map.items():
        triangles = load_mesh_triangles(path)
        expected = SCENE_EXPECTATIONS[scene]
        dimensions = np.ptp(triangles.reshape(-1, 3), axis=0)
        assert len(triangles) == expected["triangles"], (scene, len(triangles))
        assert np.allclose(dimensions, expected["dimensions"], rtol=2e-6, atol=2e-6), (
            scene, dimensions,
        )
        print(scene, len(triangles), dimensions.tolist())
    print("mesh parser self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=sorted(SCENE_EXPECTATIONS))
    parser.add_argument("--scene-label")
    parser.add_argument("--npz-path", type=Path)
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--implementation-root", type=Path, default=POWER_IMPL_ROOT)
    parser.add_argument("--extent", type=float, default=0.15)
    parser.add_argument("--gt-grid", type=int, default=240)
    parser.add_argument("--n-surface", type=int, default=20000)
    parser.add_argument("--n-volume", type=int, default=50000)
    parser.add_argument("--tau", type=float, default=0.00625)
    parser.add_argument("--iou-unit", type=float, default=0.005)
    parser.add_argument("--min-points", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self-test-assets", type=Path)
    args = parser.parse_args()
    if args.self_test_assets:
        return args
    for name in ("scene", "scene_label", "npz_path", "mesh", "out_dir"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    return args


def main() -> None:
    args = parse_args()
    if args.self_test_assets:
        self_test(args.self_test_assets)
        return
    assert args.out_dir is not None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.out_dir / "current_geometry_summary.json"
    output_figure = args.out_dir / "current_fields_top_front.png"
    if output_json.exists() or output_figure.exists():
        raise FileExistsError(f"refusing to overwrite existing evaluation: {args.out_dir}")

    metadata = load_metadata(args.npz_path)
    triangles = validate_and_transform_mesh(load_mesh_triangles(args.mesh), metadata, args.scene)
    records = load_fields(args.scene, args.implementation_root)
    render_comparison(records, triangles, args.scene_label, output_figure)
    metrics = evaluate_geometry(records, triangles, args)
    payload = {
        "schema": "rift_mesh10k_current_geometry_v2",
        "scene": args.scene,
        "scene_label": args.scene_label,
        "dataset": str(args.npz_path),
        "source_mesh": str(args.mesh),
        "ground_truth_used_for_training": False,
        "protocol": {
            "field_grid": 48,
            "extent_m": args.extent,
            "thresholds": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40,
                           0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
            "gt_grid": args.gt_grid,
            "surface_samples": args.n_surface,
            "volume_samples": args.n_volume,
            "f1_tolerance_m": args.tau,
            "iou_voxel_m": args.iou_unit,
            "selection": "each metric chooses its own oracle threshold",
        },
        "methods": metrics,
    }
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output_figure}")
    print(f"wrote {output_json}")


if __name__ == "__main__":
    main()
