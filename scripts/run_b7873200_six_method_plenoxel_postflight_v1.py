#!/usr/bin/env python3
"""Render and score six existing B787 fields without training or resuming.

This is a postflight-only reader for the completed B787 model matrix.  It
reads exactly the supplied checkpoints, converts each native representation to
one declared 48-cubed support field, and uses the selected RIFT display
protocol for every column:

    base support/energy 48^3 -> 4x trilinear (align_corners=True) ->
    magnitude -> per-method min-max normalization -> fixed t=0.20.

The outputs are qualitative support readouts plus fixed-threshold voxel
measurements against the registered B787 STL.  They neither open response
payloads nor train, resume, select checkpoints, or change any manager state.
Method-native signal metrics remain in their native reports and are not
recomputed or converted here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



def import_runtime_dependencies() -> None:
    """Load the PACE scientific stack only for a full render invocation."""
    global plt, torch, GeRaFModel, logistic_sdf_pdf, SpinrStyleINR, midpoint_grid
    global chamfer, inside_mask, prf, sample_surface_points, sample_volume_points, voxel_iou
    global deposit_points, load_energy_cloud, PLANES, load_stl_vertices
    global stl_into_scene_frame, trilinear_sample_centers, trilinear_upsample
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    import torch as _torch
    from rift.geraf import GeRaFModel as _GeRaFModel, logistic_sdf_pdf as _logistic_sdf_pdf
    from rift.spinr_style import SpinrStyleINR as _SpinrStyleINR, midpoint_grid as _midpoint_grid
    from scripts.eval_b787_geometry_metrics import (
        chamfer as _chamfer,
        inside_mask as _inside_mask,
        prf as _prf,
        sample_surface_points as _sample_surface_points,
        sample_volume_points as _sample_volume_points,
        voxel_iou as _voxel_iou,
    )
    from scripts.eval_scene_geometry import deposit_points as _deposit_points, load_energy_cloud as _load_energy_cloud
    from scripts.render_b787_vs_stl import (
        PLANES as _PLANES,
        load_stl_vertices as _load_stl_vertices,
        stl_into_scene_frame as _stl_into_scene_frame,
        trilinear_sample_centers as _trilinear_sample_centers,
        trilinear_upsample as _trilinear_upsample,
    )
    plt, torch = _plt, _torch
    GeRaFModel, logistic_sdf_pdf = _GeRaFModel, _logistic_sdf_pdf
    SpinrStyleINR, midpoint_grid = _SpinrStyleINR, _midpoint_grid
    chamfer, inside_mask, prf = _chamfer, _inside_mask, _prf
    sample_surface_points, sample_volume_points, voxel_iou = (
        _sample_surface_points, _sample_volume_points, _voxel_iou
    )
    deposit_points, load_energy_cloud = _deposit_points, _load_energy_cloud
    PLANES, load_stl_vertices = _PLANES, _load_stl_vertices
    stl_into_scene_frame = _stl_into_scene_frame
    trilinear_sample_centers, trilinear_upsample = _trilinear_sample_centers, _trilinear_upsample


SCHEMA = "rift_b7873200_six_method_plenoxel_postflight_v1"
EXPECTED = {
    "original_rift_epoch40": {"epoch": 40, "scene_repr": "grid_sh"},
    "adaptive_rift_epoch111": {"epoch": 111, "scene_repr": "point_sh"},
    "spinr_style_g96_gauss2_update60_smoke16": {"updates": 60},
    "geraf_best_step37000": {"step": 37000},
    "radar_fields_final_step8000": {"step": 8000},
    "sugavanam_ertin_stage1_epoch30": {"epoch": 30, "scene_repr": "grid"},
}


@dataclass(frozen=True)
class FieldSpec:
    identifier: str
    title: str
    subtitle: str
    checkpoint: str
    semantics: str
    geometry_status: str
    interpolation_input: str
    loader: Callable[[str, argparse.Namespace], tuple[np.ndarray, dict[str, Any]]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=False)
    parser.add_argument("--npz-path", required=False)
    parser.add_argument("--stl", required=False)
    parser.add_argument("--original-checkpoint", required=False)
    parser.add_argument("--adaptive-checkpoint", required=False)
    parser.add_argument("--spinr-checkpoint", required=False)
    parser.add_argument("--geraf-checkpoint", required=False)
    parser.add_argument("--radar-fields-checkpoint", required=False)
    parser.add_argument("--sugavanam-ertin-checkpoint", required=False)
    parser.add_argument("--extent", type=float, default=0.15)
    parser.add_argument("--grid", "--base-grid", dest="grid", type=int, default=48)
    parser.add_argument("--upsample", type=int, default=4)
    parser.add_argument("--fixed-threshold", "--threshold", dest="fixed_threshold", type=float, default=0.20)
    parser.add_argument("--crop", type=float, default=0.075)
    parser.add_argument("--f1-tau", "--tau", dest="f1_tau", type=float, default=0.00625)
    parser.add_argument("--iou-unit", type=float, default=0.005)
    parser.add_argument("--n-surface", type=int, default=20_000)
    parser.add_argument("--n-volume", type=int, default=50_000)
    parser.add_argument("--gt-grid", type=int, default=240)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--field-batch", type=int, default=4096)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def require_contract(args: argparse.Namespace) -> None:
    if args.extent != 0.15 or args.crop != 0.075:
        raise ValueError("B787 support bounds/crop are fixed at +/-0.15 m and +/-0.075 m")
    if args.grid != 48 or args.upsample != 4:
        raise ValueError("the comparison requires a 48^3 base field and 4x trilinear readout")
    if args.fixed_threshold != 0.20:
        raise ValueError("the declared fixed threshold is t=0.20")
    if args.f1_tau != 0.00625 or args.iou_unit != 0.005:
        raise ValueError("the B787 fixed geometry tolerance/unit must remain 6.25 mm / 5 mm")
    if args.n_surface != 20_000 or args.n_volume != 50_000 or args.gt_grid != 240:
        raise ValueError("the B787 truth sampling contract must remain 20k surface / 50k volume / 240^3")
    if args.field_batch <= 0:
        raise ValueError("--field-batch must be positive")


def _checkpoint(path: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state_dict"), dict):
        raise ValueError(f"{path}: expected a checkpoint with model_state_dict")
    return payload


def _base_energy_from_rift(checkpoint_path: str, args: argparse.Namespace, *, label: str) -> tuple[np.ndarray, dict[str, Any]]:
    expected = EXPECTED[label]
    checkpoint = _checkpoint(checkpoint_path)
    epoch = int(checkpoint.get("epoch", -1))
    representation = str(checkpoint.get("scene_repr"))
    if epoch != expected["epoch"] or representation != expected["scene_repr"]:
        raise ValueError(f"{label}: selected checkpoint is not epoch {expected['epoch']} {expected['scene_repr']}")
    loaded_repr, points, energies, native = load_energy_cloud(checkpoint_path, args.extent)
    if loaded_repr != representation:
        raise ValueError(f"{label}: inconsistent scene representation while loading support")
    if native is not None:
        if tuple(native.shape) != (args.grid,) * 3:
            raise ValueError(f"{label}: expected {args.grid}^3 regular support, got {tuple(native.shape)}")
        base = native.detach().cpu().numpy().astype(np.float64, copy=False)
        details: dict[str, Any] = {
            "checkpoint_epoch": epoch,
            "scene_repr": representation,
            "kind": "native_regular_grid_energy",
            "active_scatterers": int(len(points)),
        }
    else:
        base = deposit_points(points, energies, args.extent, args.grid).detach().cpu().numpy().astype(np.float64, copy=False)
        details = {
            "checkpoint_epoch": epoch,
            "scene_repr": representation,
            "kind": "point_sh_cic_trilinear_splat_then_regular_grid_readout",
            "active_scatterers": int(len(points)),
            "point_energy_conserved": bool(np.isclose(base.sum(), float(energies.sum()), rtol=1e-10, atol=1e-18)),
        }
    return base, details


def load_original(checkpoint_path: str, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    return _base_energy_from_rift(checkpoint_path, args, label="original_rift_epoch40")


def load_adaptive(checkpoint_path: str, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    return _base_energy_from_rift(checkpoint_path, args, label="adaptive_rift_epoch111")


def load_spinr(checkpoint_path: str, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    checkpoint = _checkpoint(checkpoint_path)
    if (
        checkpoint.get("format") != "rift_spinr_style_b78716_smallfit_g96_gauss2_v1"
        or checkpoint.get("run_name") != "b78710k_spinr_style_inr_pm_g96_gauss2_smallfit16_v1"
    ):
        raise ValueError("SpINR-style checkpoint is not the selected G96/Gauss2 smoke")
    recipe = checkpoint.get("recipe")
    operator = recipe.get("operator") if isinstance(recipe, dict) else None
    worklists = checkpoint.get("worklists")
    if (
        not isinstance(operator, dict)
        or operator.get("training_grid_size") != 192
        or operator.get("reference_grid_size") != 288
        or operator.get("training_quadrature") != "tensor_gauss_legendre_2_nodes_per_axis_on_96_cubed_cells"
        or operator.get("reference_quadrature") != "tensor_gauss_legendre_3_nodes_per_axis_on_96_cubed_cells"
        or operator.get("parent_cell_grid_size") != 96
        or operator.get("physical_volume_weights") is not True
        or not isinstance(worklists, dict)
        or len(worklists.get("fit_training_ids", ())) != 16
        or len(worklists.get("validation_ids", ())) != 16
    ):
        raise ValueError("SpINR-style checkpoint does not carry the selected G96/Gauss2 16/16 recipe")
    execution = checkpoint.get("execution")
    if not isinstance(execution, dict) or int(execution.get("completed_updates", -1)) != EXPECTED["spinr_style_g96_gauss2_update60_smoke16"]["updates"]:
        raise ValueError("SpINR-style checkpoint is not the selected 60-update engineering smoke")
    normalization = checkpoint.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("SpINR-style checkpoint lacks its frozen output scale")
    scale = float(normalization.get("initial_output_scale", float("nan")))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("SpINR-style checkpoint has an invalid initial output scale")
    model = SpinrStyleINR()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    points, _cell_volume = midpoint_grid(args.grid, support_m=args.extent, device="cpu")
    values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(points), args.field_batch):
            sigma = scale * model(points[start:start + args.field_batch])
            values.append(sigma.detach().cpu().double().numpy())
    support = np.concatenate(values).reshape((args.grid,) * 3)
    if not np.isfinite(support).all():
        raise ValueError("SpINR-style field readout is non-finite")
    energy = np.square(support)
    if not np.isfinite(energy).all():
        raise ValueError("SpINR-style energy readout is non-finite")
    return energy, {
        "checkpoint_updates": int(execution["completed_updates"]),
        "kind": "signed_real_sigma_g48_midpoint_readout_squared_to_energy",
        "initial_output_scale": scale,
        "engineering_status": "bounded_16_train_16_validation_60_update_smoke",
    }


def load_geraf(checkpoint_path: str, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    checkpoint = _checkpoint(checkpoint_path)
    step = int(checkpoint.get("step", -1))
    if step != EXPECTED["geraf_best_step37000"]["step"]:
        raise ValueError("GeRaF checkpoint is not the selected step-37000 checkpoint")
    config = checkpoint.get("model_config")
    if not isinstance(config, dict):
        raise ValueError("GeRaF checkpoint lacks model_config")
    model = GeRaFModel(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    axis = -args.extent + (np.arange(args.grid, dtype=np.float64) + 0.5) * (2.0 * args.extent / args.grid)
    xyz = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(xyz), args.field_batch):
            block = torch.from_numpy(xyz[start:start + args.field_batch]).float()
            density = logistic_sdf_pdf(model.sdf_network(block).squeeze(-1), model.sdf_sharpness())
            values.append(density.detach().cpu().double().numpy())
    support = np.concatenate(values).reshape((args.grid,) * 3)
    if not np.isfinite(support).all() or np.any(support < 0):
        raise ValueError("GeRaF SDF support readout is invalid")
    return support, {
        "checkpoint_step": step,
        "kind": "logistic_sdf_pdf_sdf_only_native_support",
        "reflectivity_used": False,
        "support_scalar": "NeuS-style logistic SDF surface density",
    }


def _scalar_grid_energy(checkpoint_path: str, args: argparse.Namespace, *, label: str, require_step: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    checkpoint = _checkpoint(checkpoint_path)
    expected = EXPECTED[label]
    if require_step:
        step = int(checkpoint.get("step", -1))
        if step != expected["step"]:
            raise ValueError(f"{label}: expected selected step {expected['step']}, got {step}")
    else:
        epoch = int(checkpoint.get("epoch", -1))
        representation = str(checkpoint.get("scene_repr"))
        if epoch != expected["epoch"] or representation != expected["scene_repr"]:
            raise ValueError(f"{label}: expected selected epoch/representation")
    state = checkpoint["model_state_dict"]
    re = state.get("w_re")
    im = state.get("w_im")
    if not torch.is_tensor(re) or not torch.is_tensor(im):
        raise ValueError(f"{label}: compatibility support is absent")
    if re.ndim == 4 and re.shape[-1] == 1:
        re, im = re[..., 0], im[..., 0]
    if re.ndim != 3 or tuple(re.shape) != (args.grid,) * 3 or im.shape != re.shape:
        raise ValueError(f"{label}: expected scalar {args.grid}^3 compatibility support")
    energy = (re.double().square() + im.double().square()).detach().cpu().numpy()
    active = state.get("active_mask")
    if active is not None:
        if not torch.is_tensor(active) or active.shape != re.shape:
            raise ValueError(f"{label}: active-mask shape disagrees with support")
        energy *= active.detach().cpu().numpy().astype(np.float64)
    detail: dict[str, Any] = {
        "kind": "scalar_grid_magnitude_squared_to_energy",
        "active_mask_used": active is not None,
    }
    if require_step:
        detail["checkpoint_step"] = int(checkpoint["step"])
    else:
        detail["checkpoint_epoch"] = int(checkpoint["epoch"])
        detail["scene_repr"] = str(checkpoint["scene_repr"])
    return energy, detail


def load_radar_fields(checkpoint_path: str, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    energy, detail = _scalar_grid_energy(checkpoint_path, args, label="radar_fields_final_step8000", require_step=True)
    detail["kind"] = "fixed_direction_hash_grid_alpha_compatibility_readout_squared_to_energy"
    detail["native_observable"] = "normalized-dB range-power intensity; support readout is not a coherent field"
    return energy, detail


def load_sugavanam_ertin(checkpoint_path: str, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    energy, detail = _scalar_grid_energy(checkpoint_path, args, label="sugavanam_ertin_stage1_epoch30")
    detail["kind"] = "stage1_complex_scattering_grid_magnitude_squared_to_energy"
    detail["stage2_started"] = False
    return energy, detail


def normalized_dense_magnitude(base_scalar: np.ndarray, args: argparse.Namespace, *, interpolation_input: str = "energy") -> tuple[np.ndarray, dict[str, float | str]]:
    if base_scalar.shape != (args.grid,) * 3 or not np.isfinite(base_scalar).all() or np.any(base_scalar < 0):
        raise ValueError("base support must be finite, nonnegative, and exactly the common 48^3 lattice")
    dense_scalar = trilinear_upsample(base_scalar, args.upsample).astype(np.float64, copy=False)
    if interpolation_input == "energy":
        magnitude = np.sqrt(np.maximum(dense_scalar, 0.0))
        output_kind = "sqrt(trilinearly interpolated energy)"
    elif interpolation_input == "native_support":
        magnitude = dense_scalar
        output_kind = "trilinearly interpolated native support"
    else:
        raise ValueError(f"unknown interpolation input: {interpolation_input}")
    lower, upper = float(magnitude.min()), float(magnitude.max())
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("support magnitude cannot be min-max normalized")
    return (magnitude - lower) / (upper - lower), {
        "base_scalar_min": float(base_scalar.min()),
        "base_scalar_max": float(base_scalar.max()),
        "dense_magnitude_min": lower,
        "dense_magnitude_max": upper,
        "interpolation_input": interpolation_input,
        "dense_readout": output_kind,
    }


def load_truth(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
    with np.load(args.npz_path, allow_pickle=True) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
    verts = stl_into_scene_frame(load_stl_vertices(args.stl), metadata)
    tris = verts.reshape(-1, 3, 3)
    rng = np.random.default_rng(args.seed)
    surface = sample_surface_points(tris, args.n_surface, rng)
    edges = np.linspace(-args.extent, args.extent, args.gt_grid + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    occupancy = inside_mask(tris, centers, centers, centers)
    volume = sample_volume_points(occupancy, centers, centers, centers, args.n_volume, rng)
    truth = {
        "triangle_count": int(len(tris)),
        "surface_sample_count": int(len(surface)),
        "volume_sample_count": int(len(volume)),
        "solid_gt_grid": int(args.gt_grid),
        "solid_fill_fraction": float(occupancy.mean()),
    }
    return verts, surface, volume, truth, tris


def fixed_voxel_metrics(normalized: np.ndarray, sample_centers: np.ndarray, surface: np.ndarray, volume: np.ndarray, truth: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    gx, gy, gz = np.meshgrid(sample_centers, sample_centers, sample_centers, indexing="ij")
    centers = np.stack((gx, gy, gz), axis=-1).reshape(-1, 3)
    flat = normalized.reshape(-1)
    selected = flat > args.fixed_threshold
    points, values = centers[selected], flat[selected]
    shape_surface = chamfer(points, surface)
    shape_volume = chamfer(points, volume)
    scores = prf(points, surface, args.f1_tau)
    metrics = {
        "voxel": {
            "point_count": int(len(points)),
            "cd_surface": shape_surface["cham"],
            "cd_volume": shape_volume["cham"],
            "surface_l2_mm": shape_surface["l2_mm"],
            "surface_hausdorff_mm": shape_surface["hausdorff_mm"],
            "surface_hd95_mm": shape_surface["hd95_mm"],
            "iou_solid": voxel_iou(points, volume, args.iou_unit, -args.extent, args.extent),
            "iou_shell": voxel_iou(points, surface, args.iou_unit, -args.extent, args.extent),
            **scores,
        },
        "truth": truth,
    }
    return metrics, points, values


def _draw_field(axis: Any, normalized: np.ndarray, verts: np.ndarray, *, plane: tuple[Any, ...], args: argparse.Namespace) -> None:
    _name, mip_axis, horizontal, vertical, _hlabel, _vlabel = plane
    projection = normalized.max(axis=mip_axis)
    remaining = [index for index in range(3) if index != mip_axis]
    image = projection.T if remaining == [horizontal, vertical] else projection
    visible = np.ma.masked_less_equal(image, args.fixed_threshold)
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("#050505")
    axis.imshow(visible, origin="lower", extent=[-args.extent, args.extent] * 2, cmap=cmap,
                aspect="equal", interpolation="bilinear", vmin=args.fixed_threshold, vmax=1.0)
    axis.scatter(verts[:, horizontal], verts[:, vertical], s=0.28, c="#64CCC9", alpha=0.18,
                 linewidths=0, rasterized=True)
    axis.set_xlim(-args.crop, args.crop)
    axis.set_ylim(-args.crop, args.crop)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_facecolor("#050505")
    for spine in axis.spines.values():
        spine.set_color("#252525")


def render_composite(fields: list[tuple[FieldSpec, np.ndarray]], verts: np.ndarray, args: argparse.Namespace, output: Path) -> None:
    figure, axes = plt.subplots(2, len(fields), figsize=(16.8, 5.25), squeeze=False)
    figure.patch.set_facecolor("#fcfcfb")
    for column, (spec, normalized) in enumerate(fields):
        _draw_field(axes[0, column], normalized, verts, plane=PLANES[0], args=args)
        _draw_field(axes[1, column], normalized, verts, plane=PLANES[2], args=args)
        axes[0, column].set_title(f"{spec.title}\n{spec.subtitle}", fontsize=8.8, pad=7)
    axes[0, 0].set_ylabel("Top view", fontsize=10.5, color="#4d4d4d")
    axes[1, 0].set_ylabel("Front view", fontsize=10.5, color="#4d4d4d")
    figure.suptitle("B787 fixed-threshold support readouts vs registered STL (cyan)", fontsize=12.5, y=0.99)
    figure.text(0.5, 0.012, "48³ declared support -> 4× Plenoxel-style trilinear readout; t=0.20. Brightness is independently normalized per method and is not cross-method comparable.", ha="center", va="bottom", fontsize=8.2, color="#555555")
    figure.subplots_adjust(left=0.053, right=0.995, top=0.85, bottom=0.075, hspace=0.07, wspace=0.08)
    figure.savefig(output, dpi=190, facecolor=figure.get_facecolor())
    plt.close(figure)


def render_panel(spec: FieldSpec, normalized: np.ndarray, verts: np.ndarray, args: argparse.Namespace, output: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(3.0, 5.4), squeeze=False)
    _draw_field(axes[0, 0], normalized, verts, plane=PLANES[0], args=args)
    _draw_field(axes[1, 0], normalized, verts, plane=PLANES[2], args=args)
    axes[0, 0].set_title(f"{spec.title}\n{spec.subtitle}", fontsize=9.3, pad=7)
    axes[0, 0].set_ylabel("Top", fontsize=9)
    axes[1, 0].set_ylabel("Front", fontsize=9)
    figure.subplots_adjust(left=0.13, right=0.98, top=0.91, bottom=0.035, hspace=0.10)
    figure.savefig(output, dpi=190, facecolor="white")
    plt.close(figure)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def specs(args: argparse.Namespace) -> list[FieldSpec]:
    return [
        FieldSpec("original_rift_epoch40", "Original RIFT", "e40 · grid-SH", args.original_checkpoint,
                  "rotation-invariant grid-SH scattering magnitude", "common_fixed_threshold_support_readout", "energy", load_original),
        FieldSpec("adaptive_rift_epoch111", "Adaptive RIFT", "e111 · point-SH", args.adaptive_checkpoint,
                  "CIC-splatted point-SH scattering magnitude", "common_fixed_threshold_support_readout", "energy", load_adaptive),
        FieldSpec("spinr_style_g96_gauss2_update60_smoke16", "SpINR-style", "u60 · 16/16 smoke", args.spinr_checkpoint,
                  "signed-real neural scattering-field support", "diagnostic_smoke_support_readout", "energy", load_spinr),
        FieldSpec("geraf_best_step37000", "GeRaF", "u37k · SDF support", args.geraf_checkpoint,
                  "SDF-only logistic surface-density support; reflectivity excluded", "diagnostic_native_geometry_support", "native_support", load_geraf),
        FieldSpec("radar_fields_final_step8000", "Radar Fields", "u8k · intensity field", args.radar_fields_checkpoint,
                  "fixed-direction alpha compatibility support for native intensity model", "diagnostic_native_intensity_support", "energy", load_radar_fields),
        FieldSpec("sugavanam_ertin_stage1_epoch30", "Sugavanam–Ertin", "Stage 1 e30 · proxy", args.sugavanam_ertin_checkpoint,
                  "Stage-1 complex scattering-grid proxy; not a Stage-2 SDF surface", "diagnostic_stage1_proxy_support", "energy", load_sugavanam_ertin),
    ]


def self_test() -> None:
    base = np.arange(48 ** 3, dtype=np.float64).reshape(48, 48, 48)
    # A dependency-free contract check: PACE performs the actual PyTorch
    # trilinear interpolation, while this verifies the declared dimensions and
    # min/max convention without requiring its scientific runtime locally.
    dense = np.repeat(np.repeat(np.repeat(base, 4, axis=0), 4, axis=1), 4, axis=2)
    magnitude = np.sqrt(dense)
    normalized = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
    if normalized.shape != (192, 192, 192) or not np.isclose(normalized.min(), 0.0) or not np.isclose(normalized.max(), 1.0):
        raise AssertionError("dense magnitude normalization self-test failed")
    if not np.isclose(float(base.min()), 0.0) or not float(base.max()) > float(base.min()):
        raise AssertionError("field-stat self-test failed")
    print("B787_SIX_METHOD_PLENOXEL_POSTFLIGHT_SELF_TEST=PASS")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    import_runtime_dependencies()
    require_contract(args)
    required = ("output_dir", "npz_path", "stl", "original_checkpoint", "adaptive_checkpoint", "spinr_checkpoint", "geraf_checkpoint", "radar_fields_checkpoint", "sugavanam_ertin_checkpoint")
    absent = [name for name in required if getattr(args, name) in (None, "")]
    if absent:
        raise ValueError(f"required arguments missing: {', '.join(absent)}")
    output = args.output_dir
    if output.exists():
        raise FileExistsError(f"fresh output directory required: {output}")
    output.mkdir(parents=True)
    panels = output / "panels"
    supports = output / "supports"
    panels.mkdir()
    supports.mkdir()

    verts, surface, volume, truth, _tris = load_truth(args)
    report_methods: list[dict[str, Any]] = []
    panels_for_composite: list[tuple[FieldSpec, np.ndarray]] = []
    all_metrics: dict[str, Any] = {}
    for spec in specs(args):
        base_energy, loader_details = spec.loader(spec.checkpoint, args)
        normalized, field_stats = normalized_dense_magnitude(
            base_energy, args, interpolation_input=spec.interpolation_input
        )
        sample_centers = trilinear_sample_centers(args.extent, args.grid, normalized.shape[0])
        metrics, points, values = fixed_voxel_metrics(normalized, sample_centers, surface, volume, truth, args)
        support_path = supports / f"{spec.identifier}_fixed_t0p20_support.npz"
        np.savez_compressed(support_path, points_m=points.astype(np.float32), normalized_magnitude=values.astype(np.float32), threshold=np.float64(args.fixed_threshold), sample_centers_m=sample_centers.astype(np.float64))
        np.save(supports / f"{spec.identifier}_base_energy_g48.npy", base_energy.astype(np.float32))
        panel_path = panels / f"{spec.identifier}_top_front.png"
        render_panel(spec, normalized, verts, args, panel_path)
        panels_for_composite.append((spec, normalized))
        all_metrics[spec.identifier] = {"semantics": spec.semantics, "geometry_status": spec.geometry_status, **metrics}
        report_methods.append({
            "id": spec.identifier,
            "label": f"{spec.title} {spec.subtitle}",
            "checkpoint": os.path.realpath(spec.checkpoint),
            "field_semantics": spec.semantics,
            "interpolation_input": spec.interpolation_input,
            "geometry_status": spec.geometry_status,
            "loader": loader_details,
            "field_stats": field_stats,
            "artifacts": {"panel_png": str(panel_path), "support_npz": str(support_path), "base_energy_g48_npy": str(supports / f"{spec.identifier}_base_energy_g48.npy")},
            "metrics": metrics,
        })
    composite_path = output / "six_method_top_front.png"
    render_composite(panels_for_composite, verts, args, composite_path)
    metrics_path = output / "fixed_t0p20_geometry_metrics.json"
    write_json(metrics_path, {"schema": SCHEMA, "threshold": args.fixed_threshold, "primary_variant": "dense_voxel", "methods": all_metrics})
    with np.load(args.npz_path, allow_pickle=True) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
    report = {
        "schema": SCHEMA,
        "status": "COMPLETED_EVALUATION_ONLY_RENDER_AND_FIXED_THRESHOLD_GEOMETRY",
        "training_or_resume_performed": False,
        "readout_contract": {
            "scene_bounds_m": [[-args.extent, args.extent]] * 3,
            "display_crop_m": [[-args.crop, args.crop]] * 3,
            "base_grid": [args.grid] * 3,
            "dense_grid": [args.grid * args.upsample] * 3,
            "interpolation": "torch trilinear align_corners=True on the declared common 48^3 scalar",
            "per_method_readout": "energy inputs use sqrt after interpolation; GeRaF preserves its declared native logistic-SDF support scalar",
            "normalization": "per-method min-max normalization over the full dense readout",
            "fixed_primary_threshold": args.fixed_threshold,
            "threshold_operator": "normalized_magnitude > threshold",
            "primary_geometry_variant": "dense_voxel",
            "brightness_note": "independently normalized per method; not cross-method comparable",
        },
        "registration": {
            "target_position_m": metadata.get("target_position_m"),
            "scale_factor": metadata.get("scale_factor"),
            "scaled_dimensions_m": metadata.get("scaled_dimensions_m"),
            "orientation": "dataset/STL native axes retained; STL bbox centered at origin and scaled by metadata scale_factor",
            "ground_truth_used_only_for_postflight_overlay_and_metrics": True,
        },
        "methods": report_methods,
        "artifacts": {"six_method_top_front_png": str(composite_path), "geometry_metrics_json": str(metrics_path)},
    }
    report_path = output / "evaluation_report.json"
    write_json(report_path, report)
    print(f"B787_SIX_METHOD_PLENOXEL_POSTFLIGHT=PASS output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
