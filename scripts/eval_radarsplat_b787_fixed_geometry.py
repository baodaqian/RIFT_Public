#!/usr/bin/env python3
"""Read out the accepted-budget RadarSplat Gaussian support geometry.

This standalone composition reuses the existing provisional Gaussian support
loader and the fixed six-method geometry protocol.  It does not reconstruct a
watertight surface, use RadarSplat appearance/SH/noise terms, or invoke the
six-method runner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_b787_baseline_geometry_provisional as provisional
from scripts import run_b7873200_six_method_plenoxel_postflight_v1 as fixed_protocol


CHECKPOINT_VERSION = 2
CHECKPOINT_STEP = 15_000
GRANULARITY = 48
EXTENT_M = 0.15
SIGMA_RADIUS = 3.0
METHOD_ID = "radarsplat_step15000_native_occupancy_support"
METHOD_LABEL = "RadarSplat · step 15,000 · Gaussian occupancy support proxy"
GEOMETRY_STATUS = "provisional Gaussian occupancy support proxy; not watertight surface"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--npz-path")
    parser.add_argument("--stl")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def _fixed_args(cli: argparse.Namespace) -> argparse.Namespace:
    """Use the common protocol defaults, changing only requested artifacts."""

    args = fixed_protocol.parse_args([])
    args.npz_path = cli.npz_path
    args.stl = cli.stl
    args.output_dir = cli.output_dir
    return args


def _verify_selected_checkpoint(path: str) -> dict[str, object]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"RadarSplat selected checkpoint is missing: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("RadarSplat selected checkpoint must be a mapping")
    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("RadarSplat geometry requires checkpoint version 2")
    if checkpoint.get("step") != CHECKPOINT_STEP:
        raise ValueError("RadarSplat geometry requires the selected step-15000 checkpoint")
    if checkpoint.get("complete") is not False or checkpoint.get("finalization_pending") is not False:
        raise ValueError("RadarSplat geometry requires the accepted nonterminal budget checkpoint")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("RadarSplat selected checkpoint lacks model_state_dict")
    expected = {"means", "log_scales", "quaternions", "opacity_logits", "noise_probability_logits", "sh0", "shN"}
    if set(state) != expected:
        raise ValueError("RadarSplat selected checkpoint has an unexpected model-state schema")
    means = state["means"]
    shn = state["shN"]
    if not torch.is_tensor(means) or not torch.is_tensor(shn) or tuple(means.shape) != (2046, 3):
        raise ValueError("RadarSplat selected checkpoint is not the reviewed 2046-row topology")
    if shn.ndim != 3 or tuple(shn.shape[1:]) != (15, 3):
        raise ValueError("RadarSplat selected checkpoint is not the reviewed SH-degree-3 topology")
    return {
        "path": str(checkpoint_path.resolve()),
        "checkpoint_version": CHECKPOINT_VERSION,
        "step": CHECKPOINT_STEP,
        "complete": False,
        "finalization_pending": False,
        "training_status": "accepted 24h budget stop; nonterminal training state",
        "gaussian_rows": int(means.shape[0]),
        "sh_degree": int(round((int(shn.shape[1]) + 1) ** 0.5 - 1)),
        "native_signal_observable": "RadarSplat native elevation-integrated real 2-D power; no coherent phase output",
    }


def evaluate_support(
    base_support: np.ndarray,
    args: argparse.Namespace,
    surface: np.ndarray,
    volume: np.ndarray,
    truth: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply the unchanged common interpolation, normalization, and metrics."""

    normalized, field_stats = fixed_protocol.normalized_dense_magnitude(
        np.asarray(base_support, dtype=np.float64), args, interpolation_input="native_support"
    )
    sample_centers = fixed_protocol.trilinear_sample_centers(
        args.extent, args.grid, normalized.shape[0]
    )
    metrics, points, values = fixed_protocol.fixed_voxel_metrics(
        normalized, sample_centers, surface, volume, dict(truth), args
    )
    return normalized, field_stats, sample_centers, points, {"metrics": metrics, "values": values}


def _primary_metrics(metrics: Mapping[str, Any]) -> dict[str, float | int]:
    voxel = metrics.get("voxel")
    if not isinstance(voxel, Mapping):
        raise ValueError("fixed geometry scorer did not return voxel metrics")
    return {
        "point_count": int(voxel["point_count"]),
        "chamfer_squared_m2": float(voxel["cd_surface"]),
        "maximum_hausdorff_mm": float(voxel["surface_hausdorff_mm"]),
        "hd95_mm": float(voxel["surface_hd95_mm"]),
        "solid_iou": float(voxel["iou_solid"]),
        "f1": float(voxel["f1"]),
    }


def _write_support(path: Path, points: np.ndarray, values: np.ndarray, sample_centers: np.ndarray, threshold: float) -> None:
    np.savez_compressed(
        path,
        points_m=np.asarray(points, dtype=np.float32),
        normalized_native_support=np.asarray(values, dtype=np.float32),
        sample_centers_m=np.asarray(sample_centers, dtype=np.float64),
        threshold=np.asarray(float(threshold), dtype=np.float64),
    )


def main(argv: Sequence[str] | None = None) -> int:
    cli = parse_args(argv)
    if cli.self_test:
        fixed_protocol.self_test()
        return 0
    missing = [name for name in ("checkpoint", "npz_path", "stl", "output_dir") if getattr(cli, name) in (None, "")]
    if missing:
        raise ValueError("required arguments missing: " + ", ".join(missing))
    args = _fixed_args(cli)
    fixed_protocol.require_contract(args)
    checkpoint_details = _verify_selected_checkpoint(cli.checkpoint)

    if cli.output_dir.exists():
        raise FileExistsError(f"fresh RadarSplat geometry output directory required: {cli.output_dir}")
    cli.output_dir.mkdir(parents=True)
    supports_dir = cli.output_dir / "supports"
    panels_dir = cli.output_dir / "panels"
    supports_dir.mkdir()
    panels_dir.mkdir()

    # The common runner initializes the scorer, STL registration, and the
    # existing align_corners=True trilinear implementation without running its
    # six-method main or loading any other checkpoint.
    fixed_protocol.import_runtime_dependencies()
    verts, surface, volume, truth, _tris = fixed_protocol.load_truth(args)
    base_support, loader_stats = provisional.load_radarsplat_occupancy(
        cli.checkpoint,
        granularity=GRANULARITY,
        extent=EXTENT_M,
        sigma_radius=SIGMA_RADIUS,
    )
    normalized, field_stats, sample_centers, points, scored = evaluate_support(
        base_support, args, surface, volume, truth
    )
    metrics = scored["metrics"]
    support_path = supports_dir / f"{METHOD_ID}_fixed_t0p20_support.npz"
    base_path = supports_dir / f"{METHOD_ID}_base_native_support_g48.npy"
    _write_support(support_path, points, scored["values"], sample_centers, args.fixed_threshold)
    np.save(base_path, np.asarray(base_support, dtype=np.float32))
    spec = fixed_protocol.FieldSpec(
        identifier=METHOD_ID,
        title="RadarSplat",
        subtitle="step 15,000 · native support",
        checkpoint=str(Path(cli.checkpoint).resolve()),
        semantics="opacity-weighted anisotropic Gaussian occupancy union; SH/noise excluded",
        geometry_status=GEOMETRY_STATUS,
        interpolation_input="native_support",
        loader=lambda _checkpoint, _args: (base_support, loader_stats),
    )
    panel_path = panels_dir / f"{METHOD_ID}_top_front.png"
    fixed_protocol.render_panel(spec, normalized, verts, args, panel_path)

    voxel = metrics["voxel"]
    report = {
        "schema": "rift_b7873200_radarsplat_fixed_geometry_v1",
        "status": "COMPLETED_EVALUATION_ONLY_FIXED_THRESHOLD_GEOMETRY",
        "training_or_resume_performed": False,
        "checkpoint_binding": {
            "signal_checkpoint": str(Path(cli.checkpoint).resolve()),
            "geometry_checkpoint": str(Path(cli.checkpoint).resolve()),
            "same_checkpoint": True,
        },
        "selected_checkpoint": checkpoint_details,
        "method": {
            "id": METHOD_ID,
            "label": METHOD_LABEL,
            "geometry_status": GEOMETRY_STATUS,
            "field_formula": loader_stats["native_geometry_definition"],
            "support_loader": loader_stats,
            "native_signal_observable": checkpoint_details["native_signal_observable"],
            "appearance_terms_used": {"sh_reflectance": False, "noise_probability": False},
        },
        "readout_contract": {
            "scene_bounds_m": [[-args.extent, args.extent]] * 3,
            "base_grid": [args.grid] * 3,
            "dense_grid": [args.grid * args.upsample] * 3,
            "interpolation": "existing torch trilinear align_corners=True from the declared common 48^3 scalar",
            "interpolation_input": "native_support",
            "normalization": "full-volume min-max normalization of trilinearly interpolated native support",
            "fixed_primary_threshold": float(args.fixed_threshold),
            "threshold_operator": "normalized_support > threshold",
            "f1_tau_m": float(args.f1_tau),
            "iou_unit_m": float(args.iou_unit),
            "registration": "existing B787 STL/metadata registration through load_truth",
        },
        "truth": truth,
        "field_stats": field_stats,
        "metrics": metrics,
        "primary_metrics": _primary_metrics(metrics),
        "counts": {
            "gaussians_total": int(loader_stats["gaussians_total"]),
            "gaussians_with_support_in_box": int(loader_stats["gaussians_with_support_in_box"]),
            "support_point_count": int(voxel["point_count"]),
        },
        "artifacts": {
            "panel_png": str(panel_path),
            "support_npz": str(support_path),
            "base_native_support_g48_npy": str(base_path),
        },
    }
    fixed_protocol.write_json(cli.output_dir / "evaluation_report.json", report)
    print(f"RADARSPLAT_B787_FIXED_GEOMETRY=PASS output={cli.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
