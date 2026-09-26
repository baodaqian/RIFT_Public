#!/usr/bin/env python
"""SH-SAS / Reed-protocol 3D-reconstruction metrics for B787 checkpoints.

SH-SAS's main body evaluates GEOMETRY only (their Tables 1/3: Chamfer, IoU,
Precision, F1), and SpINR/SpINRv2 add Hausdorff. Those tables inherit Reed et
al.'s public evaluation code (`external/reed_sas/evaluate/`), so this script
reimplements THAT protocol rather than inventing one -- read it before changing
any definition here:

  * occupancy comes from the reconstruction's magnitude field, MIN-MAX
    normalized to [0,1] and thresholded (`main_mesh_recon_and_3d_space_loss.py`
    lines "mag = (mag-min)/(max-min); condition = mag > thresh"). Min-max is
    what makes the metric survive our free (gain, scene) gauge -- see CLAUDE.md.
  * two prediction variants: **A** = thresholded voxel centres, **B** = marching
    cubes over the thresholded field, resampled ("mesh_" prefix in their code).
  * two ground truths: **surface** points (20k, area-weighted over the STL) and
    **volume** points (50k, inside the watertight mesh). Chamfer is reported
    against both, exactly as they do.
  * Chamfer follows pytorch3d's `chamfer_distance`: the MEAN of SQUARED nearest
    -neighbour distances, summed over both directions. Their published values
    are in that convention; a mean-L2 value in mm is reported alongside because
    the squared form is not readable.
  * IoU voxelizes both point clouds at `--iou-unit` (their 0.005 m) and takes
    intersection/union. Their GT is the SOLID point cloud, which is why every
    method in SpINRv2's Table 1 scores IoU ~= 0.09: a coherent reconstruction is
    a SHELL and the truth is filled. A shell-vs-shell variant is reported too.
  * they sweep the threshold and publish the OPTIMAL value per metric (their
    `evaluate/example_metrics/*/optimal_value_*.csv`). So does this script.

**Cross-paper comparability, stated plainly:** these numbers are on OUR target,
in metres, with our scene box. SH-SAS/SpINR normalize their meshes and never
state the scale, so the ABSOLUTE values are not portable (the same defect as
their supplementary Table 4). What IS portable is the protocol and the ranking
across our own arms. Values normalized by the target's largest dimension
(D = 0.10 m) are printed for that reason.

With --object/--dataset-root, the selected object's registered mesh and
checkpoint identity are required. --fixed-threshold declares the primary
normalized-magnitude threshold; additional --thresholds are oracle diagnostics.
Adaptive point-SH is read on a fixed --point-grid (default 48) by conservative
cloud-in-cell deposition of active, unlocked SH energy. CSV rows record this
spatial readout separately from the threshold and optional dense interpolation.

    module load anaconda3 && conda activate RIFT
    python scripts/eval_b787_geometry_metrics.py \
        --checkpoints training_checkpoints/b787_r4_capdeg3 ... --csv out.csv
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.render_b787_vs_stl import (  # noqa: E402
    load_energy_field, load_stl_vertices, stl_into_scene_frame,
    sample_indices_to_physical, trilinear_sample_centers, trilinear_upsample,
    collection_geometry_inputs,
)

DEFAULT_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
                      0.60, 0.70, 0.80, 0.90, 0.95)


# ---------------------------------------------------------------- ground truth
def sample_surface_points(tris, n, rng):
    """Area-weighted uniform samples over an STL's triangles -> (n,3)."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    idx = rng.choice(len(tris), size=n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    return a[idx] + u[:, None] * (b[idx] - a[idx]) + v[:, None] * (c[idx] - a[idx])


def inside_mask(tris, gx, gy, gz):
    """Solid voxelization by x-scanline parity.

    For each (y,z) grid line, intersect the +x ray with every triangle whose yz
    bounding box contains it, sort the crossings and fill between pairs. The
    B787 STL is watertight (checked: 292 683 of 292 755 edges are shared by
    exactly two triangles, none by one), so parity is well defined.
    """
    ny, nz = len(gy), len(gz)
    occ = np.zeros((len(gx), ny, nz), dtype=bool)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]

    # bin triangles by their yz bounding box over the (y,z) lattice
    dy = gy[1] - gy[0] if ny > 1 else 1.0
    dz = gz[1] - gz[0] if nz > 1 else 1.0
    ymin = np.minimum(np.minimum(a[:, 1], b[:, 1]), c[:, 1])
    ymax = np.maximum(np.maximum(a[:, 1], b[:, 1]), c[:, 1])
    zmin = np.minimum(np.minimum(a[:, 2], b[:, 2]), c[:, 2])
    zmax = np.maximum(np.maximum(a[:, 2], b[:, 2]), c[:, 2])
    j0 = np.clip(np.ceil((ymin - gy[0]) / dy).astype(int), 0, ny - 1)
    j1 = np.clip(np.floor((ymax - gy[0]) / dy).astype(int), 0, ny - 1)
    k0 = np.clip(np.ceil((zmin - gz[0]) / dz).astype(int), 0, nz - 1)
    k1 = np.clip(np.floor((zmax - gz[0]) / dz).astype(int), 0, nz - 1)
    buckets = [[[] for _ in range(nz)] for _ in range(ny)]
    for t in range(len(tris)):
        if ymax[t] < gy[0] or ymin[t] > gy[-1] or zmax[t] < gz[0] or zmin[t] > gz[-1]:
            continue
        for j in range(j0[t], j1[t] + 1):
            row = buckets[j]
            for k in range(k0[t], k1[t] + 1):
                row[k].append(t)

    # barycentric point-in-triangle in the yz projection, then interpolate x
    ay, az, by, bz, cy, cz = a[:, 1], a[:, 2], b[:, 1], b[:, 2], c[:, 1], c[:, 2]
    det = (by - ay) * (cz - az) - (bz - az) * (cy - ay)
    for j in range(ny):
        for k in range(nz):
            cand = buckets[j][k]
            if not cand:
                continue
            t = np.asarray(cand)
            d = det[t]
            ok = np.abs(d) > 1e-20
            t, d = t[ok], d[ok]
            if t.size == 0:
                continue
            py, pz = gy[j] - a[t, 1], gz[k] - a[t, 2]
            w1 = (py * (cz[t] - az[t]) - pz * (cy[t] - ay[t])) / d
            w2 = (pz * (by[t] - ay[t]) - py * (bz[t] - az[t])) / d
            hit = (w1 >= 0) & (w2 >= 0) & (w1 + w2 <= 1)
            if not hit.any():
                continue
            t, w1, w2 = t[hit], w1[hit], w2[hit]
            xs = a[t, 0] + w1 * (b[t, 0] - a[t, 0]) + w2 * (c[t, 0] - a[t, 0])
            xs = np.sort(xs)
            if len(xs) % 2:  # non-manifold grazing hit -- drop this line
                continue
            for lo, hi in zip(xs[0::2], xs[1::2]):
                occ[(gx >= lo) & (gx <= hi), j, k] = True
    return occ


def sample_volume_points(occ, gx, gy, gz, n, rng):
    """Jittered samples from the occupied cells of a solid voxelization."""
    ii, jj, kk = np.nonzero(occ)
    if len(ii) == 0:
        raise RuntimeError("solid voxelization is empty")
    pick = rng.choice(len(ii), size=n, replace=len(ii) < n)
    hx = 0.5 * (gx[1] - gx[0]), 0.5 * (gy[1] - gy[0]), 0.5 * (gz[1] - gz[0])
    p = np.stack([gx[ii[pick]], gy[jj[pick]], gz[kk[pick]]], axis=1)
    return p + (rng.random((n, 3)) * 2 - 1) * np.asarray(hx)


# -------------------------------------------------------------------- metrics
def chamfer(pred, gt):
    """(pytorch3d convention: mean SQUARED NN distance, both directions,
    summed), plus the symmetric mean L2 and the Hausdorff pair."""
    if len(pred) == 0:
        return dict(cham=float("nan"), l2_mm=float("nan"),
                    hausdorff_mm=float("nan"), hd95_mm=float("nan"))
    d_pg, _ = cKDTree(gt).query(pred, workers=-1)
    d_gp, _ = cKDTree(pred).query(gt, workers=-1)
    return dict(
        cham=float((d_pg ** 2).mean() + (d_gp ** 2).mean()),
        l2_mm=float(1000 * 0.5 * (d_pg.mean() + d_gp.mean())),
        hausdorff_mm=float(1000 * max(d_pg.max(), d_gp.max())),
        hd95_mm=float(1000 * max(np.percentile(d_pg, 95), np.percentile(d_gp, 95))),
    )


def prf(pred, gt_surface, tau):
    """Precision / recall / F1 at tolerance tau, against the SURFACE truth."""
    if len(pred) == 0:
        return dict(precision=0.0, recall=0.0, f1=0.0)
    d_pg, _ = cKDTree(gt_surface).query(pred, workers=-1)
    d_gp, _ = cKDTree(pred).query(gt_surface, workers=-1)
    p = float((d_pg <= tau).mean())
    r = float((d_gp <= tau).mean())
    return dict(precision=p, recall=r, f1=(2 * p * r / (p + r) if p + r > 0 else 0.0))


def voxel_iou(pred, gt, unit, lo, hi):
    """Occupancy IoU of two point clouds voxelized on a shared `unit` lattice."""
    if len(pred) == 0:
        return 0.0
    shape = tuple(int(np.ceil((hi - lo) / unit)) + 1 for _ in range(3))

    def occupy(pts):
        idx = np.floor((pts - lo) / unit).astype(int)
        idx = idx[np.all((idx >= 0) & (idx < np.asarray(shape)), axis=1)]
        v = np.zeros(shape, dtype=bool)
        v[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        return v

    a, b = occupy(pred), occupy(gt)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


# ----------------------------------------------------------------- prediction
def predicted_points(mag_norm, centers, thresh):
    """Variant A: centres of voxels whose min-max-normalized magnitude > thresh."""
    keep = mag_norm.reshape(-1) > thresh
    return centers[keep]


def marching_cubes_points(mag_norm, sample_centers, thresh, n, rng):
    """Variant B: marching cubes at `thresh`, resampled to n surface points."""
    from skimage.measure import marching_cubes
    if not (mag_norm.min() < thresh < mag_norm.max()):
        return np.zeros((0, 3))
    verts, faces, _, _ = marching_cubes(mag_norm, level=thresh)
    g = mag_norm.shape[0]
    # marching_cubes returns sample-index coordinates.  For an
    # align_corners=True dense field those samples retain the native grid's
    # first/last voxel-centre locations rather than spanning the box boundary.
    if len(sample_centers) != g:
        raise ValueError("sample-centre axis does not match the marching-cubes field")
    verts = sample_indices_to_physical(verts, sample_centers)
    return sample_surface_points(verts[faces], n, rng)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--object")
    p.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parents[1] / "data/RIFT_dataset")
    from rift.antenna_selection import add_arguments
    add_arguments(p)
    p.add_argument("--num-train", type=int, default=3200, help="Checkpoint training subset size for --object")
    p.add_argument("--npz-path")
    p.add_argument("--stl")
    p.add_argument("--extent", type=float, default=0.15)
    p.add_argument("--point-grid", type=int, default=48,
                   help="CIC readout grid per axis for adaptive point-SH (default 48); ignored for native grids")
    p.add_argument("--upsample", type=int, default=1,
                   help="trilinear upsample of the energy field before thresholding. "
                        "Default 1 (native grid): interpolation is a VISUALIZATION tool "
                        "in this project, so keep it out of the reported metric.")
    p.add_argument("--fixed-threshold", type=float, help="predeclared primary threshold on min-max normalized magnitude")
    p.add_argument("--thresholds", type=float, nargs="+", help="optional oracle diagnostic sweep")
    p.add_argument("--tau", type=float, default=0.00625,
                   help="F1 tolerance [m]; default = one g48 voxel pitch")
    p.add_argument("--iou-unit", type=float, default=0.005,
                   help="IoU voxel size [m]; Reed et al. use 0.005")
    p.add_argument("--n-surface", type=int, default=20000)
    p.add_argument("--n-volume", type=int, default=50000)
    p.add_argument("--gt-grid", type=int, default=240,
                   help="lattice resolution for the solid voxelization of the STL")
    p.add_argument("--mesh-variant", action="store_true",
                   help="also report the marching-cubes prediction (Reed's 'mesh_' rows)")
    p.add_argument("--min-points", type=int, default=20,
                   help="a threshold keeping fewer voxels than this cannot win the "
                        "'optimal over the sweep' report (guards degenerate level sets)")
    p.add_argument("--csv", default=None)
    p.add_argument("--plot", default=None,
                   help="scatter geometry score vs held-out-view rel-MSE "
                        "(val scraped from the SLURM logs by checkpoint name)")
    p.add_argument("--plot-val-max", type=float, default=50.0,
                   help="exclude collapsed arms from the scatter; the question is "
                        "whether geometry ranks arms that all LOOK like an aircraft")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.point_grid < 2 or args.upsample < 1:
        p.error("--point-grid must be >= 2 and --upsample must be >= 1")
    if args.object is None:
        args.npz_path = args.npz_path or "data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere2k.npz"
        args.stl = args.stl or "data/B787.stl"
        args.thresholds = args.thresholds or list(DEFAULT_THRESHOLDS)
    else:
        from rift.rift_dataset import resolve_object_inputs, object_spec
        resolve_object_inputs(object_name=args.object, dataset_root=args.dataset_root, npz_path=args.npz_path)
        args.object = object_spec(args.object)["object_id"]
        if args.fixed_threshold is None:
            p.error("collection geometry requires an explicit --fixed-threshold")
        if args.plot:
            p.error("--plot scrapes historical B787 logs; collection geometry must use matched-checkpoint signal records")
        args.thresholds = args.thresholds or []
    if args.fixed_threshold is not None:
        args.thresholds = sorted(set([args.fixed_threshold, *args.thresholds]))
    if any(not np.isfinite(t) or not 0 < t < 1 for t in args.thresholds):
        p.error("geometry thresholds must be finite and strictly between 0 and 1")
    if args.labels is not None and len(args.labels) != len(args.checkpoints):
        p.error("--labels must match --checkpoints")
    return args


def main(argv=None):
    args = parse_args(argv)

    rng = np.random.default_rng(args.seed)
    contract = None
    if args.object is not None:
        meta, vertices, contract = collection_geometry_inputs(args)
        # Validate every checkpoint before the expensive truth sampling/metrics.
        for path in args.checkpoints:
            ck = os.path.join(path, "checkpoint_best.pth.tar") if os.path.isdir(path) else path
            load_energy_field(ck, expected_contract=contract, extent=args.extent, point_grid=args.point_grid)
    else:
        with np.load(args.npz_path, allow_pickle=True, mmap_mode="r") as archive:
            meta = json.loads(str(archive["metadata_json"]))
        vertices = stl_into_scene_frame(load_stl_vertices(args.stl), meta)
    tris = vertices.reshape(-1, 3, 3)
    d_max = float((tris.reshape(-1, 3).max(0) - tris.reshape(-1, 3).min(0)).max())

    gt_surface = sample_surface_points(tris, args.n_surface, rng)
    g = args.gt_grid
    lin = np.linspace(-args.extent, args.extent, g + 1)
    ctr = 0.5 * (lin[:-1] + lin[1:])
    occ = inside_mask(tris, ctr, ctr, ctr)
    gt_volume = sample_volume_points(occ, ctr, ctr, ctr, args.n_volume, rng)
    print(f"truth: {len(tris)} triangles, D = {1000 * d_max:.1f} mm, "
          f"solid fill = {occ.mean() * 100:.2f}% of the {g}³ box "
          f"({occ.sum()} cells)\n")

    labels = args.labels or [os.path.basename(c.rstrip("/")) for c in args.checkpoints]
    rows = []
    for path, label in zip(args.checkpoints, labels):
        row_start = len(rows)
        ck = os.path.join(path, "checkpoint_best.pth.tar") if os.path.isdir(path) else path
        readout = {}
        energy, G, epoch = load_energy_field(ck, expected_contract=contract, extent=args.extent,
                                           point_grid=args.point_grid, readout_info=readout)
        energy = trilinear_upsample(energy, args.upsample)
        gg = energy.shape[0]
        mag = np.sqrt(energy)                      # Reed keys on |albedo|
        mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-30)
        c_g = trilinear_sample_centers(args.extent, G, gg)
        centers = np.stack(np.meshgrid(c_g, c_g, c_g, indexing="ij"), -1).reshape(-1, 3)

        print(f"=== {label}  (ep {epoch}, {gg}³) ===")
        print("geometry readout:", json.dumps(readout, sort_keys=True))
        if args.fixed_threshold is not None:
            print(f"primary fixed normalized-magnitude threshold: {args.fixed_threshold:g}")
        print(f"{'thresh':>7s} {'n_pts':>7s} {'CD_surf':>9s} {'CD_vol':>9s} "
              f"{'L2 mm':>7s} {'HD95 mm':>8s} {'IoU_sol':>8s} {'IoU_shell':>9s} "
              f"{'prec':>6s} {'rec':>6s} {'F1':>6s}")
        for th in args.thresholds:
            pred = predicted_points(mag, centers, th)
            m_s = chamfer(pred, gt_surface)
            m_v = chamfer(pred, gt_volume)
            f = prf(pred, gt_surface, args.tau)
            iou_sol = voxel_iou(pred, gt_volume, args.iou_unit, -args.extent, args.extent)
            iou_shl = voxel_iou(pred, gt_surface, args.iou_unit, -args.extent, args.extent)
            rows.append(dict(run=label, epoch=epoch, variant="voxel", thresh=th,
                             n_points=len(pred), cd_surface=m_s["cham"],
                             cd_volume=m_v["cham"], l2_mm=m_s["l2_mm"],
                             hausdorff_mm=m_s["hausdorff_mm"], hd95_mm=m_s["hd95_mm"],
                             iou_solid=iou_sol, iou_shell=iou_shl, **f))
            print(f"{th:7.2f} {len(pred):7d} {m_s['cham']:9.2e} {m_v['cham']:9.2e} "
                  f"{m_s['l2_mm']:7.2f} {m_s['hd95_mm']:8.2f} {iou_sol:8.4f} "
                  f"{iou_shl:9.4f} {f['precision']:6.3f} {f['recall']:6.3f} {f['f1']:6.3f}")

            if args.mesh_variant:
                mp = marching_cubes_points(mag, c_g, th, args.n_surface, rng)
                mm_s, mm_v = chamfer(mp, gt_surface), chamfer(mp, gt_volume)
                mf = prf(mp, gt_surface, args.tau)
                rows.append(dict(run=label, epoch=epoch, variant="mesh", thresh=th,
                                 n_points=len(mp), cd_surface=mm_s["cham"],
                                 cd_volume=mm_v["cham"], l2_mm=mm_s["l2_mm"],
                                 hausdorff_mm=mm_s["hausdorff_mm"],
                                 hd95_mm=mm_s["hd95_mm"],
                                 iou_solid=voxel_iou(mp, gt_volume, args.iou_unit,
                                                     -args.extent, args.extent),
                                 iou_shell=voxel_iou(mp, gt_surface, args.iou_unit,
                                                     -args.extent, args.extent), **mf))

        for result in rows[row_start:]:
            result.update(scene_repr=readout.get("scene_repr", "grid_sh"),
                          spatial_readout=readout.get("spatial_readout", "native_voxel_centers"),
                          readout_grid=G, sampled_grid=gg)
        if contract is not None:
            for result in rows[row_start:]:
                result.update(object_id=contract["dataset_identity"]["object_id"],
                    checkpoint=str(Path(ck).resolve()), extent_m=args.extent,
                    extraction="minmax_sqrt_sh_energy", upsample=args.upsample,
                    threshold_role="fixed_primary" if result["thresh"] == args.fixed_threshold else "oracle_diagnostic")
            if args.thresholds == [args.fixed_threshold]:
                continue
        # Reed publishes the OPTIMAL value over the threshold sweep, per metric.
        # `--min-points` guards the degenerate end of that protocol: a level set
        # holding two voxels has a superb Chamfer distance and describes nothing.
        mine = [r for r in rows[row_start:] if r["variant"] == "voxel"
                and r["n_points"] >= args.min_points]
        if not mine:
            print("  (no threshold keeps >= --min-points voxels)\n")
            continue
        best = {
            "CD_surface": min(mine, key=lambda r: r["cd_surface"]),
            "CD_volume": min(mine, key=lambda r: r["cd_volume"]),
            "IoU_solid": max(mine, key=lambda r: r["iou_solid"]),
            "F1": max(mine, key=lambda r: r["f1"]),
        }
        print("  ground-truth-oracle threshold diagnostic (Reed convention):")
        for k, r in best.items():
            val = {"CD_surface": r["cd_surface"], "CD_volume": r["cd_volume"],
                   "IoU_solid": r["iou_solid"], "F1": r["f1"]}[k]
            print(f"    {k:11s} {val:.4g}   @ thresh {r['thresh']:.2f} "
                  f"({r['n_points']} pts)")
        print()

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote", args.csv)

    if args.plot:
        plot_geometry_vs_val(rows, args, labels)


def plot_geometry_vs_val(rows, args, labels):
    """Does geometry quality predict held-out SIGNAL quality? Two panels: the
    per-arm-optimal F1 (Reed's reporting convention) and F1 at Reed's default
    fixed threshold, each against val rel-MSE scraped from the training logs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    from scripts.scrape_training_logs import scrape

    runs = scrape()
    val = {}
    for path, label in zip(args.checkpoints, labels):
        name = os.path.basename(path.rstrip("/"))
        eps = runs.get(name, {}).get("epochs", [])
        v = [e["val"] for e in eps if "val" in e]
        if v:
            val[label] = min(v)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6))
    fig.patch.set_facecolor("#fcfcfb")
    panels = [
        ("per-arm OPTIMAL threshold  (their reporting convention)", "opt"),
        ("FIXED threshold 0.20  (same level set for every arm)", "fixed"),
    ]
    for ax, (title, mode) in zip(axes, panels):
        xs, ys, names = [], [], []
        for label in labels:
            if label not in val or val[label] > args.plot_val_max:
                continue
            mine = [r for r in rows if r["run"] == label and r["variant"] == "voxel"]
            if mode == "opt":
                cand = [r for r in mine if r["n_points"] >= args.min_points]
                if not cand:
                    continue
                y = max(cand, key=lambda r: r["f1"])["f1"]
            else:
                match = [r for r in mine if abs(r["thresh"] - 0.2) < 1e-9]
                if not match:
                    continue
                y = match[0]["f1"]
            xs.append(val[label]), ys.append(y), names.append(label)
        rho, pv = spearmanr(xs, ys)
        ax.scatter(xs, ys, s=60, color="#2a78d6", zorder=3,
                   edgecolor="#fcfcfb", linewidth=1.2)
        for x, y, n in zip(xs, ys, names):
            ax.annotate(n, (x, y), textcoords="offset points", xytext=(6, 4),
                        fontsize=8.5, color="#52514e")
        ax.set_title(f"{title}\nSpearman ρ = {rho:+.2f}  (p = {pv:.3f}, n = {len(xs)})",
                     fontsize=11.5, color="#0b0b0b", loc="left", pad=10)
        ax.set_xlabel("validation rel-MSE  (held-out viewpoints)  →  worse",
                      fontsize=11, color="#52514e")
        ax.set_ylabel("F1 @ 6.25 mm  →  better", fontsize=11, color="#52514e")
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color="#e6e5e2", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c9c8c4")
        ax.tick_params(colors="#52514e", labelsize=10)

    fig.suptitle("Does geometry fidelity predict held-out SIGNAL fidelity? Only if the "
                 "threshold is held fixed.", fontsize=13.5, color="#0b0b0b",
                 x=0.011, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(args.plot, dpi=170, facecolor=fig.get_facecolor())
    print("wrote", args.plot)


if __name__ == "__main__":
    main()
