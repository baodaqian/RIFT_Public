#!/usr/bin/env python
"""Overlay a grid or adaptive point-SH reconstruction against its object's STL.

Point-SH readout deposits active, unlocked SH energy onto a declared voxel-center
lattice (--point-grid, default 48) using energy-conserving cloud-in-cell weights.
It uses the checkpoint's learned positions and optional immutable support bounds.
This spatial readout and any subsequent interpolation are evaluation-only.

The B787 npz was synthesised by scaling the full-scale STL by
`metadata_json['scale_factor']` (largest dim -> D=0.10 m) and centering its
bounding box at the origin (`target_position_m == [0,0,0]`). This script
replays that exact transform so the STL sits in the SAME frame as the trained
scene box ([-extent, extent]^3), then renders the reconstruction's |w|-energy
support (rotation-invariant SH energy per voxel) as max-intensity projections
with the STL silhouette overlaid, in the three orthogonal planes.

Truth geometry (extended target) has no `target_positions`; the STL IS the
truth, hence this dedicated script rather than eval_scene_geometry (radial /
sphere-specific) or render_dense_scene (sphere/cube/tetra outlines only).

Usage:
    python scripts/render_b787_vs_stl.py \
        --checkpoint training_checkpoints/b787_sphere2k_gridsh6_g48_n1800/checkpoint_final.pth.tar \
        --npz-path data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere2k.npz \
        --stl data/B787.stl --extent 0.15 --out-prefix figures/b787_recon/n1800
"""
import argparse
import json
import os
import struct
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_stl_vertices(path):
    """Read binary or ASCII STL as finite triangle vertices, without rescaling.

    Binary headers may start with ``solid``; decide by the exact record length
    before attempting ASCII. The restored A320 source uses ASCII STL.
    """
    raw = Path(path).read_bytes()
    n = struct.unpack("<I", raw[80:84])[0] if len(raw) >= 84 else 0
    if n and len(raw) == 84 + n * 50:
        tris = np.frombuffer(raw, dtype=np.uint8, offset=84).reshape(n, 50)
        verts = np.frombuffer(tris[:, 12:48].tobytes(), dtype="<f4").reshape(-1, 3)
    else:
        try:
            lines = raw.decode("ascii").splitlines()
            if not lines or not lines[0].strip().lower().startswith("solid"):
                raise ValueError("missing ASCII solid header")
            vertices = [parts[1:] for line in lines
                        if (parts := line.split()) and parts[0].lower() == "vertex"]
            facets = sum(line.strip().lower() == "endfacet" for line in lines)
            if not facets or len(vertices) != 3 * facets or any(len(v) != 3 for v in vertices):
                raise ValueError("incomplete ASCII triangles")
            verts = np.asarray(vertices, dtype=np.float64)
        except (UnicodeError, ValueError) as exc:
            raise ValueError(f"Invalid or truncated STL: {path}") from exc
    verts = verts.astype(np.float64)
    if not np.isfinite(verts).all():
        raise ValueError(f"STL vertices must be finite: {path}")
    return verts


def stl_into_scene_frame(verts, meta):
    """Apply the synthesis transform: center bbox at origin, scale by scale_factor."""
    scale = float(meta["scale_factor"])
    bbox_center = (verts.max(0) + verts.min(0)) / 2.0
    return (verts - bbox_center) * scale


def _point_sh_energy_field(sd, extent, grid):
    """Checkpoint-faithful point positions/SH masks, then conservative CIC readout."""
    from scripts.eval_scene_geometry import deposit_points

    if isinstance(grid, bool) or not isinstance(grid, (int, np.integer)) or grid < 2:
        raise ValueError("point-grid must be an integer >= 2")
    if extent is None or not np.isfinite(extent) or extent <= 0:
        raise ValueError("point-SH readout requires a finite positive extent")
    required = ("w_re", "w_im", "anchors", "cell_half", "delta_raw", "active_mask", "order", "basis_degree")
    if any(key not in sd or not torch.is_tensor(sd[key]) for key in required):
        raise ValueError("point-SH checkpoint lacks position/active/order/SH tensors")
    real, imag = sd["w_re"], sd["w_im"]
    if real.ndim != 2 or imag.shape != real.shape or real.shape[1] < 1:
        raise ValueError("point-SH coefficients must have matching [capacity,basis] shapes")
    capacity, basis_count = real.shape
    degree = int(round(basis_count ** .5)) - 1
    if (degree + 1) ** 2 != basis_count:
        raise ValueError("point-SH basis must contain complete degree bands")
    shapes = {"anchors": (capacity, 3), "cell_half": (capacity, 1),
              "delta_raw": (capacity, 3), "active_mask": (capacity,),
              "order": (capacity,), "basis_degree": (basis_count,)}
    if any(tuple(sd[key].shape) != shape for key, shape in shapes.items()):
        raise ValueError("point-SH position/mask/order tensor shapes disagree")
    if sd["active_mask"].dtype != torch.bool:
        raise ValueError("point-SH active_mask must be boolean")
    if sd["order"].dtype not in (torch.int32, torch.int64) or sd["basis_degree"].dtype not in (torch.int32, torch.int64):
        raise ValueError("point-SH order/basis_degree must be integer tensors")
    expected_basis = torch.arange(degree + 1).repeat_interleave(2 * torch.arange(degree + 1) + 1)
    if not torch.equal(sd["basis_degree"], expected_basis):
        raise ValueError("point-SH basis_degree does not match the SH coefficient ordering")
    active = sd["active_mask"]
    order = sd["order"][active]
    if ((order < 0) | (order > degree)).any():
        raise ValueError("point-SH active order is outside the allocated SH degree")
    for key in ("anchors", "cell_half", "delta_raw", "w_re", "w_im"):
        value = sd[key]
        if not value.is_floating_point() or not torch.isfinite(value[active]).all():
            raise ValueError(f"point-SH active {key} must be finite real values")
    if (sd["cell_half"][active] < 0).any():
        raise ValueError("point-SH active cell_half must be nonnegative")

    # Compute in the saved parameter dtype, exactly as AdaptivePointSHScene.positions.
    positions = sd["anchors"][active] + sd["cell_half"][active] * torch.tanh(sd["delta_raw"][active])
    enabled = sd.get("support_bounds_enabled", torch.tensor(False))
    if not torch.is_tensor(enabled) or enabled.dtype != torch.bool or enabled.ndim != 0:
        raise ValueError("point-SH support_bounds_enabled must be a scalar boolean")
    tolerance = max(1e-12, float(extent) * 1e-6)  # fp32 representation of the scene box
    if enabled.item():
        lower, upper = sd.get("support_min"), sd.get("support_max")
        if (not torch.is_tensor(lower) or not torch.is_tensor(upper)
                or lower.shape != (3,) or upper.shape != (3,)
                or not torch.isfinite(lower).all() or not torch.isfinite(upper).all()
                or not (lower < upper).all()
                or (lower < -extent - tolerance).any() or (upper > extent + tolerance).any()):
            raise ValueError("point-SH immutable support bounds disagree with the scene extent")
        positions = torch.maximum(torch.minimum(positions, upper), lower)
    positions = positions.double()
    if (not torch.isfinite(positions).all() or (positions.abs() > extent + tolerance).any()):
        raise ValueError("point-SH positions lie outside the declared scene extent")
    positions = positions.clamp(-extent, extent)  # remove only fp32 boundary roundoff
    unlocked = sd["basis_degree"][None, :] <= order[:, None]
    squared = real[active].double().square() + imag[active].double().square()
    energy = torch.where(unlocked, squared, 0).sum(-1)
    volume = deposit_points(positions, energy, float(extent), int(grid))
    conserved = torch.isclose(volume.sum(), energy.sum(), rtol=1e-10, atol=0).item()
    if not torch.isfinite(volume).all() or not conserved:
        raise ValueError("point-SH CIC readout failed finite/energy-conservation checks")
    info = dict(scene_repr="point_sh", spatial_readout="point_sh_cic_voxel_centers_v1",
                readout_grid=int(grid), active_scatterers=int(active.sum()),
                sh_energy="active_unlocked_sum_abs_coeff_squared",
                support_bounds_enabled=bool(enabled.item()), energy_conserved=bool(conserved),
                source_energy=float(energy.sum()), readout_energy=float(volume.sum()),
                boundary_rule="clamp_neighbor_indices_at_scene_faces")
    return volume.numpy(), info


def load_energy_field(checkpoint_path, *, expected_contract=None, extent=None,
                      point_grid=48, readout_info=None):
    """Grid or point-SH checkpoint -> [G,G,G] rotation-invariant energy.

    Grid fields retain their native lattice. Point-SH uses a declared CIC grid,
    not a reshape of slots/anchors and not an inference from occupied bounds.
    The optional info dict records the extraction without changing the old tuple API.
    """
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if expected_contract is not None:
        from rift.rift_dataset import validate_checkpoint_object
        from scripts.eval_b787_range_power import resolve_scene_extent
        validate_checkpoint_object(ck, expected_contract)
        saved_contract = ck.get("sealed_npz_protocol_contract", {})
        if "training_selection" in expected_contract or "training_selection" in saved_contract:
            from train import _validate_saved_sealed_npz_protocol_contract
            _validate_saved_sealed_npz_protocol_contract(saved_contract, expected_contract)
        if extent is not None and not np.isclose(resolve_scene_extent(ck), extent, rtol=0, atol=1e-12):
            raise ValueError("Geometry extent disagrees with the selected checkpoint")
    sd = ck["model_state_dict"]
    if ck.get("scene_repr") == "point_sh" or "anchors" in sd:
        from scripts.eval_b787_range_power import resolve_scene_extent
        if ck.get("scene_repr") not in (None, "point_sh"):
            raise ValueError("Checkpoint representation disagrees with point-SH state")
        saved_extent = resolve_scene_extent(ck)
        if extent is not None and not np.isclose(saved_extent, extent, rtol=0, atol=1e-12):
            raise ValueError("Geometry extent disagrees with the selected checkpoint")
        volume, info = _point_sh_energy_field(sd, saved_extent, point_grid)
        if readout_info is not None:
            readout_info.update(info)
        return volume, int(point_grid), ck.get("epoch")
    w_re, w_im = sd["w_re"], sd["w_im"]  # [G,G,G,n_basis]
    if w_re.shape != w_im.shape or w_re.ndim not in (3, 4) or len(set(w_re.shape[:3])) != 1:
        raise ValueError("This geometry readout requires grid, grid_sh, or point_sh state")
    energy = w_re ** 2 + w_im ** 2
    if energy.ndim == 4:
        energy = energy.sum(dim=-1)  # Parseval: direction-integrated power
    if expected_contract is not None and "active_mask" in sd:
        energy = energy * sd["active_mask"].reshape(energy.shape)
    g = w_re.shape[0]  # [G,G,G,n_basis]
    if readout_info is not None:
        readout_info.update(scene_repr=ck.get("scene_repr") or ("grid_sh" if w_re.ndim == 4 else "grid"),
                            spatial_readout="native_voxel_centers", readout_grid=int(g))
    return energy.numpy(), int(g), ck.get("epoch")


def trilinear_upsample(energy_ggg, factor):
    """[G,G,G] -> [G*factor]^3 via trilinear interpolation -- the Plenoxel recipe
    for reading a coarse voxel grid as a dense continuous field. VISUALIZATION
    ONLY: it never enters the training path (see CLAUDE.md)."""
    if factor <= 1:
        return energy_ggg
    x = torch.from_numpy(np.ascontiguousarray(energy_ggg))[None, None].float()
    d = energy_ggg.shape[0] * factor
    dense = torch.nn.functional.interpolate(
        x, size=(d, d, d), mode="trilinear", align_corners=True)
    return dense[0, 0].numpy()


def trilinear_sample_centers(extent, native_grid_size, sampled_grid_size=None):
    """Physical coordinates for native or ``align_corners=True`` dense samples.

    The learned values live at voxel centres, not at the scene-box boundary.
    ``align_corners=True`` preserves the first and last voxel-centre locations,
    so an upsampled field must span that same interval.  Recomputing cell
    centres at the dense resolution would silently expand the reconstruction.
    """
    edges = np.linspace(-extent, extent, native_grid_size + 1)
    native = 0.5 * (edges[:-1] + edges[1:])
    sampled_grid_size = sampled_grid_size or native_grid_size
    if sampled_grid_size == native_grid_size:
        return native
    return np.linspace(native[0], native[-1], sampled_grid_size)


def image_extent_from_centers(centers):
    """imshow bounds whose pixel centers coincide with the sampled field."""
    axis = np.asarray(centers)
    if axis.ndim != 1 or len(axis) < 2 or not np.all(np.diff(axis) > 0):
        raise ValueError("Image coordinates require at least two increasing sample centers")
    half_pitch = (axis[1] - axis[0]) / 2
    return [axis[0] - half_pitch, axis[-1] + half_pitch] * 2


def collection_geometry_inputs(args):
    """Validate identity; center/scale every object's original STL without response reads."""
    from rift.rift_dataset import (resolve_object_inputs, load_object_contract,
        object_identity, validate_checkpoint_object, geometry_reference, transform_mesh_vertices)
    npz_path, manifest = resolve_object_inputs(object_name=args.object,
        dataset_root=args.dataset_root, npz_path=args.npz_path)
    _, contract = load_object_contract(npz_path, manifest, num_train=getattr(args, "num_train", None),
        **{key: getattr(args, key, None) for key in ("num_tx", "num_rx", "tx_indices", "rx_indices")})
    validate_checkpoint_object(object_identity(args.object), contract)
    reference = geometry_reference(args.object, args.dataset_root, mesh_path=args.stl)
    if reference["dataset_identity"] != contract["dataset_identity"]:
        raise ValueError("Geometry reference belongs to another object")
    vertices = transform_mesh_vertices(load_stl_vertices(reference["mesh_path"]), reference["metadata"],
        input_frame=reference["transform"].get("input_frame", "source_model_units"))
    return reference["metadata"], vertices, contract


def sample_indices_to_physical(sample_indices, sample_centers):
    """Map continuous lattice indices onto a known physical sample axis."""
    sample_centers = np.asarray(sample_centers)
    if sample_centers.ndim != 1 or len(sample_centers) < 2:
        raise ValueError("sample_centers must be a one-dimensional axis of length >= 2")
    spacing = (sample_centers[-1] - sample_centers[0]) / (len(sample_centers) - 1)
    return sample_centers[0] + np.asarray(sample_indices) * spacing


PLANES = [
    # (name, mip_axis, horiz_axis, vert_axis, horiz_label, vert_label)
    ("Top-down (planform)", 1, 0, 2, "x  [m]", "z  [m]"),
    ("Side",                2, 0, 1, "x  [m]", "y  [m]"),
    ("Front (nose-on)",     0, 2, 1, "z  [m]", "y  [m]"),
]


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--object")
    p.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parents[1] / "data/RIFT_dataset")
    from rift.antenna_selection import add_arguments
    add_arguments(p)
    p.add_argument("--num-train", type=int, default=3200, help="Checkpoint training subset size for --object")
    p.add_argument("--npz-path")
    p.add_argument("--stl")
    p.add_argument("--extent", type=float, default=0.15)
    p.add_argument("--point-grid", type=int, default=48,
                   help="CIC readout grid per axis for point-SH (default 48); grid scenes keep their native lattice")
    p.add_argument("--gamma", type=float, default=1.0,
                   help="display gamma on MIP energy (0.5 = sqrt, brightens weak support)")
    p.add_argument("--pmin-pct", type=float, default=88.0,
                   help="percentile of MIP pixels mapped to black (kills pedestal haze)")
    p.add_argument("--top-frac", type=float, default=0.03,
                   help="energy fraction kept for the thresholded support scatter")
    p.add_argument("--stl-subsample", type=int, default=15000)
    p.add_argument("--upsample", type=int, default=1,
                   help="Plenoxel-style trilinear upsampling factor per axis on the "
                        "energy field before projecting (1 = raw voxel lattice). "
                        "Visualization only -- never in the training path.")
    p.add_argument("--out-prefix")
    args = p.parse_args(argv)
    if args.point_grid < 2 or args.upsample < 1:
        p.error("--point-grid must be >= 2 and --upsample must be >= 1")
    if args.object is None:
        if args.npz_path is None:
            p.error("--npz-path or --object is required")
        args.stl = args.stl or "data/B787.stl"
    else:
        from rift.rift_dataset import resolve_object_inputs, object_spec
        resolve_object_inputs(object_name=args.object, dataset_root=args.dataset_root, npz_path=args.npz_path)
        args.object = object_spec(args.object)["object_id"]
    args.out_prefix = args.out_prefix or (f"figures/RIFT_dataset/{args.object}/recon" if args.object else "figures/b787_recon/recon")
    return args


def main(argv=None):
    args = parse_args(argv)

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)

    contract = None
    if args.object is not None:
        meta, verts, contract = collection_geometry_inputs(args)
    else:
        with np.load(args.npz_path, allow_pickle=True, mmap_mode="r") as z:
            meta = json.loads(str(z["metadata_json"]))
        verts = stl_into_scene_frame(load_stl_vertices(args.stl), meta)
    object_label = args.object or "B787"
    planes = [(f"{hl[0].upper()}{vl[0].upper()} projection", mip, ha, va, hl, vl)
              for _, mip, ha, va, hl, vl in PLANES] if contract else PLANES
    if verts.shape[0] > args.stl_subsample:
        idx = np.random.default_rng(0).choice(verts.shape[0], args.stl_subsample, replace=False)
        verts_plot = verts[idx]
    else:
        verts_plot = verts

    readout = {}
    energy, G, epoch = load_energy_field(args.checkpoint, expected_contract=contract, extent=args.extent,
                                        point_grid=args.point_grid, readout_info=readout)
    print("geometry readout:", json.dumps(readout, sort_keys=True))
    if args.upsample > 1:
        energy = trilinear_upsample(energy, args.upsample)
        print(f"trilinear-upsampled {G}^3 -> {energy.shape[0]}^3 ({args.upsample}x per axis)")
    ext = args.extent
    centers = trilinear_sample_centers(ext, G, energy.shape[0])

    print(f"checkpoint epoch={epoch} G={G} extent=+/-{ext}m")
    print(f"STL scaled dims (m): {meta['scaled_dimensions_m']}  scale={meta['scale_factor']:.4g}")
    print(f"reconstruction bbox occupied by voxels: energy max={energy.max():.3e}")

    # ---- Row-of-3 overlay: MIP heatmap + STL silhouette ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (name, mip_ax, ha, va, hl, vl) in zip(axes, planes):
        mip = energy.max(axis=mip_ax)          # 2D over the two remaining axes
        # order remaining axes so `ha` is horizontal, `va` is vertical
        remaining = [a for a in range(3) if a != mip_ax]
        # mip has shape indexed by remaining[0], remaining[1]
        if remaining == [ha, va]:
            img = mip.T                        # -> [va, ha]
        else:
            img = mip                          # already [va, ha]
        disp = img ** args.gamma
        # clip display so the low-energy pedestal haze goes black and only
        # genuine support shows (percentiles over the in-crop pixels)
        vmax = np.percentile(disp, 99.7)
        vmin = np.percentile(disp, args.pmin_pct)
        ax.imshow(disp, origin="lower", extent=image_extent_from_centers(centers),
                  cmap="inferno", aspect="equal", interpolation="bilinear",
                  vmin=vmin, vmax=vmax)
        ax.scatter(verts_plot[:, ha], verts_plot[:, va], s=0.4, c="cyan",
                   alpha=0.10, linewidths=0, rasterized=True)
        ax.set_title(name, fontsize=12)
        ax.set_xlabel(hl); ax.set_ylabel(vl)
        # tighten to the object, not the whole 0.15 box
        lim = 0.075
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    fig.suptitle(f"{object_label} reconstruction (|w| energy MIP, inferno) vs STL truth (cyan)  "
                 f"— {os.path.basename(os.path.dirname(args.checkpoint))}, ep{epoch}",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = args.out_prefix + "_overlay.png"
    fig.savefig(out, dpi=140)
    print("wrote", out)

    # ---- Companion: STL-only silhouettes (reference, same crops) ----
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (name, mip_ax, ha, va, hl, vl) in zip(axes2, planes):
        ax.scatter(verts_plot[:, ha], verts_plot[:, va], s=0.5, c="k",
                   alpha=0.15, linewidths=0, rasterized=True)
        ax.set_title(name + " — STL truth", fontsize=12)
        ax.set_xlabel(hl); ax.set_ylabel(vl)
        ax.set_aspect("equal")
        lim = 0.075
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    fig2.tight_layout()
    out2 = args.out_prefix + "_stl_only.png"
    fig2.savefig(out2, dpi=140)
    print("wrote", out2)

    # ---- Thresholded top-energy support scatter (the cleanest shape read) ----
    ee = energy.ravel()
    order = np.argsort(ee)[::-1]
    csum = np.cumsum(ee[order])
    keep = order[csum <= args.top_frac * ee.sum()]
    ii, jj, kk = np.unravel_index(keep, energy.shape)   # x,y,z voxel indices
    vx, vy, vz = centers[ii], centers[jj], centers[kk]
    ven = ee[keep]
    print(f"support scatter: {keep.size} voxels hold top {args.top_frac:.0%} energy")
    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5.2))
    scene_pts = {0: vx, 1: vy, 2: vz}
    for ax, (name, _mip, ha, va, hl, vl) in zip(axes3, planes):
        ax.scatter(verts_plot[:, ha], verts_plot[:, va], s=0.6, c="lightgray",
                   alpha=0.25, linewidths=0, rasterized=True, zorder=1)
        ax.scatter(scene_pts[ha], scene_pts[va], s=14, c=ven, cmap="inferno",
                   alpha=0.9, linewidths=0, zorder=2)
        ax.set_title(name, fontsize=12)
        ax.set_xlabel(hl); ax.set_ylabel(vl)
        ax.set_aspect("equal")
        lim = 0.075
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    fig3.suptitle(f"{object_label} reconstruction — top {args.top_frac:.0%}-energy support (inferno) "
                  f"vs STL truth (gray), ep{epoch}", fontsize=13)
    fig3.tight_layout(rect=[0, 0, 1, 0.96])
    out3 = args.out_prefix + "_support.png"
    fig3.savefig(out3, dpi=140)
    print("wrote", out3)


if __name__ == "__main__":
    main()
