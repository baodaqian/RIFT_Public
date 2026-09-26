#!/usr/bin/env python
"""Compare Sugavanam--Ertin SDF and RIFT on the common B787 geometry protocol.

Reported for both methods: symmetric squared Chamfer, readable mean-L2,
Hausdorff/HD95, precision/recall/F1 at tau, shell IoU, and Reed/SH-SAS-style
surface-vs-solid IoU.  The SDF additionally gets a true signed-solid IoU.

RIFT has no canonical zero level set, so its occupancy threshold is selected
by maximum F1 from the same declared sweep used by
``eval_b787_geometry_metrics.py``.  The selected threshold is always written;
it is never silently optimized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.eval_b787_geometry_metrics import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    chamfer,
    inside_mask,
    prf,
    sample_surface_points,
    sample_volume_points,
    voxel_iou,
)
from scripts.render_b787_vs_stl import (  # noqa: E402
    load_energy_field,
    load_stl_vertices,
    stl_into_scene_frame,
)


def load_truth(args, rng):
    meta = json.loads(str(np.load(args.npz_path, allow_pickle=True, mmap_mode="r")["metadata_json"]))
    tris = stl_into_scene_frame(load_stl_vertices(args.stl), meta).reshape(-1, 3, 3)
    gt_surface = sample_surface_points(tris, args.n_surface, rng)
    lin = np.linspace(-args.extent, args.extent, args.gt_grid + 1)
    centres = 0.5 * (lin[:-1] + lin[1:])
    gt_occ = inside_mask(tris, centres, centres, centres)
    gt_volume = sample_volume_points(gt_occ, centres, centres, centres, args.n_volume, rng)
    return tris, gt_surface, gt_occ, gt_volume, centres


def metric_row(method, label, pred, gt_surface, gt_volume, args, selected_threshold="zero"):
    ms = chamfer(pred, gt_surface)
    mv = chamfer(pred, gt_volume)
    pf = prf(pred, gt_surface, args.tau)
    # eval_b787_geometry_metrics.voxel_iou takes scalar cube bounds and
    # broadcasts them over xyz.  Passing length-3 arrays makes its scalar
    # shape construction fail before any metric is evaluated.
    lo = -args.extent
    hi = args.extent
    return {
        "method": method,
        "label": label,
        "selected_threshold": selected_threshold,
        "n_pred": len(pred),
        "chamfer_surface_sq_m2": ms["cham"],
        "chamfer_volume_sq_m2": mv["cham"],
        "mean_l2_mm": ms["l2_mm"],
        "hausdorff_mm": ms["hausdorff_mm"],
        "hd95_mm": ms["hd95_mm"],
        "precision": pf["precision"],
        "recall": pf["recall"],
        "f1": pf["f1"],
        "iou_surface": voxel_iou(pred, gt_surface, args.iou_unit, lo, hi),
        "iou_surface_vs_solid": voxel_iou(pred, gt_volume, args.iou_unit, lo, hi),
        "signed_solid_iou": float("nan"),
    }


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def npz_text(z, key, default=None):
    if key not in z.files:
        return default
    return str(np.asarray(z[key]).item())


def surface_calibration_metadata(z, surface_path):
    """Load and classify the saved surface contract before metric evaluation."""
    from rift.sugavanam_ertin import METHOD_NAME
    from rift.sugavanam_ertin_a320_stabilized import (
        A320_METHOD_NAME,
        classify_se_artifact,
    )
    from rift.sugavanam_ertin_validzero import SE2_METHOD_NAME

    scalar_contract_keys = (
        "method",
        "policy",
        "implementation_kind",
        "calibration",
        "calibration_policy",
    )
    metadata = {
        key: npz_text(z, key) for key in scalar_contract_keys if key in z.files
    }
    variant = classify_se_artifact(metadata)
    if variant == "raw reproduction":
        derivative_keys = {
            "policy",
            "manager_identity",
            "artifact_identity",
            "calibration",
            "calibration_policy",
        }
        if (
            npz_text(z, "method") != METHOD_NAME
            or "sign_flipped" not in z.files
            or not derivative_keys.isdisjoint(z.files)
        ):
            raise ValueError("raw SDF surface has mixed or incomplete metadata")
        sign_flipped = bool(np.asarray(z["sign_flipped"]).item())
        source = os.path.join(os.path.dirname(surface_path), "checkpoint_final.pth.tar")
        if not os.path.isfile(source):
            raise ValueError("raw SDF surface lacks checkpoint_final.pth.tar")
        import torch

        try:
            state = torch.load(source, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(source, map_location="cpu")
        if (
            state.get("method") != METHOD_NAME
            or state.get("ground_truth_geometry_used") is not False
            or bool((state.get("args") or {}).get("smoke"))
            or int(state.get("step", -1)) != 5000
        ):
            raise ValueError("raw surface checkpoint violates the raw final contract")
        return {
            "variant": variant,
            "display_method": "Sugavanam--Ertin raw reproduction",
            "policy": "legacy_zero_isolevel",
            "raw_isolevel": 0.0,
            "orientation": -1 if sign_flipped else 1,
            "source_checkpoint": source,
            "source_checkpoint_sha256": None,
            "source_role": "checkpoint_final",
        }
    if variant in {
        "valid-zero stabilized derivative",
        "A320 valid-zero stabilized derivative",
    }:
        method = npz_text(z, "method")
        policy = npz_text(z, "policy")
        manager_identity = npz_text(z, "manager_identity")
        artifact_identity = npz_text(z, "artifact_identity")
        if method == A320_METHOD_NAME:
            raise ValueError("A320 valid-zero surface cannot enter the B787 comparator")
        if method == SE2_METHOD_NAME:
            accepted_identities = {
                ("rift_sugertin_validzero_v2", "b787_sugavanam_ertin_validzero_v2"),
                ("rift_sugertin_validzero_v2p1", "b787_sugavanam_ertin_validzero_v2p1"),
            }
            identity_ok = (
                policy == "closed_sphere_init_sign_anchors_valid_projection_v2"
                and (manager_identity, artifact_identity) in accepted_identities
            )
            display = "Sugavanam--Ertin valid-zero stabilized derivative"
        else:
            identity_ok = False
            display = ""
        truth_used = bool(np.asarray(z["ground_truth_geometry_used"]).item()) if (
            "ground_truth_geometry_used" in z.files
        ) else True
        try:
            validity = json.loads(npz_text(z, "validity_json", "{}"))
            topology = json.loads(npz_text(z, "topology_json", "{}"))
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("valid-zero surface has malformed audit metadata") from error
        if (
            not identity_ok
            or truth_used
            or validity.get("passed") is not True
            or topology.get("passed") is not True
        ):
            raise ValueError("valid-zero surface failed identity/provenance/engineering gates")
        source = os.path.join(os.path.dirname(surface_path), "checkpoint_final.pth.tar")
        if not os.path.isfile(source):
            raise ValueError("valid-zero surface lacks its own checkpoint_final.pth.tar")
        import torch

        try:
            state = torch.load(source, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(source, map_location="cpu")
        surface_audit = state.get("surface_audit") or {}
        checkpoint_ok = (
            state.get("method") == method
            and state.get("policy") == policy
            and state.get("manager_identity") == manager_identity
            and state.get("artifact_identity") == artifact_identity
            and state.get("ground_truth_geometry_used") is False
            and (surface_audit.get("validity") or {}).get("passed") is True
            and (surface_audit.get("topology") or {}).get("passed") is True
        )
        if not checkpoint_ok:
            raise ValueError("valid-zero checkpoint and surface contracts disagree")
        return {
            "variant": variant,
            "display_method": display,
            "policy": policy,
            "raw_isolevel": 0.0,
            "orientation": 1,
            "source_checkpoint": source,
            "source_checkpoint_sha256": None,
            "source_role": "checkpoint_final",
        }
    if variant != "calibrated derivative":
        raise ValueError("unknown or mixed Sugavanam--Ertin surface metadata")
    if (
        npz_text(z, "method") != METHOD_NAME
        or npz_text(z, "policy") is not None
        or "stabilized" in (npz_text(z, "implementation_kind", "").lower())
    ):
        raise ValueError("calibrated surface has mixed raw/derivative metadata")
    policy = npz_text(z, "calibration_policy")
    source = npz_text(z, "source_checkpoint")
    source_sha = npz_text(z, "source_checkpoint_sha256")
    source_role = npz_text(z, "source_checkpoint_role")
    orientation = int(np.asarray(z["orientation"]).item())
    level = float(np.asarray(z["raw_isolevel"]).item())
    sign_flipped = bool(np.asarray(z["sign_flipped"]).item())
    if policy != "stage1_median_unweighted_l1_v1":
        raise ValueError(f"unexpected calibrated SDF policy: {policy!r}")
    if not source or not source_sha or source_role != "checkpoint_final":
        raise ValueError("calibrated surface lacks checkpoint_final provenance")
    if os.path.basename(source) != "checkpoint_final.pth.tar":
        raise ValueError("calibrated surface source is not checkpoint_final.pth.tar")
    if orientation not in (-1, 1) or sign_flipped != (orientation == -1):
        raise ValueError("calibrated surface has inconsistent sign-orientation metadata")
    if not np.isfinite(level):
        raise ValueError("calibrated surface has a non-finite raw isolevel")
    if not os.path.isfile(source) or sha256(source) != source_sha:
        raise ValueError("calibrated surface source checkpoint hash mismatch")
    scatter_sha = npz_text(z, "scatter_checkpoint_sha256")
    if not scatter_sha or len(scatter_sha) != 64:
        raise ValueError("calibrated surface lacks stage-1 checkpoint provenance")
    import torch

    try:
        state = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6
        state = torch.load(source, map_location="cpu")
    if (state.get("method") != METHOD_NAME or int(state.get("step", -1)) != 5000 or
            state.get("ground_truth_geometry_used") is not False or
            state.get("scatter_checkpoint_sha256") != scatter_sha):
        raise ValueError("calibrated surface source checkpoint violates the retained-final contract")
    return {
        "variant": variant,
        "display_method": "Sugavanam--Ertin calibrated derivative",
        "policy": policy,
        "raw_isolevel": level,
        "orientation": orientation,
        "source_checkpoint": source,
        "source_checkpoint_sha256": source_sha,
        "source_role": source_role,
    }


def sdf_row(path, label, gt_surface, gt_occ, gt_volume, gt_axis, args):
    z = np.load(path, allow_pickle=False)
    pred = np.asarray(z["surface_points"], dtype=np.float64)
    calibration = surface_calibration_metadata(z, path)
    if calibration["variant"] == "raw reproduction":
        selected = "zero (raw reproduction)"
    elif calibration["variant"] == "calibrated derivative":
        selected = f"SDF calibrated raw-isolevel={calibration['raw_isolevel']:.8g}"
    else:
        selected = "zero (valid-zero stabilized derivative)"
    row = metric_row(
        calibration["display_method"], label, pred, gt_surface, gt_volume, args, selected
    )
    row["sdf_variant"] = calibration["variant"]
    row["sdf_calibration_policy"] = calibration["policy"]
    row["sdf_raw_isolevel"] = calibration["raw_isolevel"]
    row["sdf_source_checkpoint_role"] = calibration["source_role"]
    row["sdf_source_checkpoint_sha256"] = calibration["source_checkpoint_sha256"] or (
        "legacy-unpinned"
        if calibration["variant"] == "raw reproduction"
        else "not-recorded"
    )
    sdf = np.asarray(z["sdf"])
    sdf_extent = float(z["extent"])
    if sdf.shape == gt_occ.shape and abs(sdf_extent - args.extent) < 1e-9:
        pred_occ = sdf < 0
        union = np.logical_or(pred_occ, gt_occ).sum()
        row["signed_solid_iou"] = (
            float(np.logical_and(pred_occ, gt_occ).sum() / union) if union else 0.0
        )
    else:
        # Evaluate the saved neural SDF at the GT grid for an exact shared lattice.
        import torch
        from rift.sugavanam_ertin import FourierFeatureSDF

        ck_path = calibration["source_checkpoint"]
        try:
            state = torch.load(ck_path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(ck_path, map_location="cpu")
        if state.get("method") != npz_text(z, "method"):
            raise ValueError("surface/checkpoint method metadata disagree")
        model = FourierFeatureSDF(**state["model_config"])
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        pred_occ = np.zeros_like(gt_occ)
        yz = np.stack(np.meshgrid(gt_axis, gt_axis, indexing="ij"), axis=-1).reshape(-1, 2)
        with torch.no_grad():
            for ix, x in enumerate(gt_axis):
                xyz = np.concatenate((np.full((len(yz), 1), x), yz), axis=1).astype(np.float32)
                raw = model(torch.from_numpy(xyz)).numpy() - calibration["raw_isolevel"]
                raw = calibration["orientation"] * raw
                pred_occ[ix] = (raw < 0).reshape(len(gt_axis), len(gt_axis))
        union = np.logical_or(pred_occ, gt_occ).sum()
        row["signed_solid_iou"] = float(np.logical_and(pred_occ, gt_occ).sum() / union) if union else 0.0
    return row


def rift_row(path, label, gt_surface, gt_volume, args):
    energy, grid, _ = load_energy_field(path)
    mag = np.sqrt(np.asarray(energy))
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-30)
    g = int(grid)
    pitch = 2.0 * args.extent / g
    axis = np.linspace(-args.extent + pitch / 2, args.extent - pitch / 2, g)
    centres = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    candidates = []
    for threshold in args.thresholds:
        pred = centres[mag.reshape(-1) > threshold]
        if len(pred) < args.min_points:
            continue
        row = metric_row("RIFT", label, pred, gt_surface, gt_volume, args, threshold)
        candidates.append(row)
    if not candidates:
        raise RuntimeError(f"no RIFT threshold retained >= {args.min_points} points for {path}")
    return max(candidates, key=lambda r: (r["f1"], -r["chamfer_surface_sq_m2"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdf-surfaces", nargs="+", required=True,
                   help="surface_reconstruction.npz files")
    p.add_argument("--sdf-labels", nargs="+", default=None)
    p.add_argument("--rift-checkpoints", nargs="*", default=[])
    p.add_argument("--rift-labels", nargs="+", default=None)
    p.add_argument("--npz-path", default="data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere2k.npz")
    p.add_argument("--stl", default="data/B787.stl")
    p.add_argument("--extent", type=float, default=0.15)
    p.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    p.add_argument("--tau", type=float, default=0.00625)
    p.add_argument("--iou-unit", type=float, default=0.005)
    p.add_argument("--n-surface", type=int, default=20000)
    p.add_argument("--n-volume", type=int, default=50000)
    p.add_argument("--gt-grid", type=int, default=240)
    p.add_argument("--min-points", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", required=True)
    args = p.parse_args()

    if args.sdf_labels and len(args.sdf_labels) != len(args.sdf_surfaces):
        p.error("--sdf-labels length must match --sdf-surfaces")
    if args.rift_labels and len(args.rift_labels) != len(args.rift_checkpoints):
        p.error("--rift-labels length must match --rift-checkpoints")
    rng = np.random.default_rng(args.seed)
    tris, gt_surface, gt_occ, gt_volume, gt_axis = load_truth(args, rng)
    print(f"B787 truth: {len(tris)} triangles; {gt_occ.sum()} occupied cells on {args.gt_grid}^3")
    rows = []
    sdf_labels = args.sdf_labels or [os.path.basename(os.path.dirname(x)) for x in args.sdf_surfaces]
    for path, label in zip(args.sdf_surfaces, sdf_labels):
        rows.append(sdf_row(path, label, gt_surface, gt_occ, gt_volume, gt_axis, args))
    rift_labels = args.rift_labels or [os.path.basename(os.path.dirname(x)) for x in args.rift_checkpoints]
    for path, label in zip(args.rift_checkpoints, rift_labels):
        rows.append(rift_row(path, label, gt_surface, gt_volume, args))

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['method']:20s} {row['label']:28s} "
            f"CD={row['chamfer_surface_sq_m2']:.6g} F1={row['f1']:.4f} "
            f"IoU(shell)={row['iou_surface']:.4f} IoU(solid)={row['iou_surface_vs_solid']:.4f} "
            f"signed-IoU={row['signed_solid_iou']:.4f} threshold={row['selected_threshold']}"
        )
    print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
