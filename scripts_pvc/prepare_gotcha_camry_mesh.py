#!/usr/bin/env python3
"""Register a Toyota Camry XV20 mesh to the GOTCHA Camry for 3D evaluation.

GOTCHA supplies no 3D truth for its Camry, only four ground-level footprint
corners from its ground-truth workbook (``CAMRY_FOOTPRINT_NATIVE_CORNERS_M``,
+-0.10 m). The user identified the car as an XV20 (1997-2001) Camry
(2026-09-22); its footprint (mean 4.78 x 1.76 m) fits the XV20 (4.765 x 1.785 m)
best. The geometry is a same-generation stand-in, not a scan of the car:

* source: Sketchfab "Toyota Camry (Mk4)(XV20) 1997" by Nieve5677, CC Attribution
  (OBJ export, centimetres, front -y, up +z, wheels on z = 0);
* the cabin interior (group ``camry_inner``) is dropped; every exterior part is kept;
* axes map to the GOTCHA region frame the models reconstruct in (front +x, left +y,
  up +z) and the body is scaled per axis to the XV20 specification (length 4.765,
  width 1.785 without mirrors, height 1.430 m); ``--no-spec-scale`` keeps the
  model's own proportions;
* the pose is the least-squares rigid fit (yaw + translation) of the footprint
  rectangle to the four workbook corners; the wheels rest on their mean height.

``--data-frame-offset`` first moves the workbook corners by the documentation-to-data
offset that the calibration array measured (``DOCUMENTATION_TO_DATA_OFFSET``), for the case
that the workbook shares Table 1's survey frame. It writes a separate variant
(default ``data/meshes/camry_xv20_data_frame``); the workbook-only registration is kept.

Outputs in ``--output-dir``: the surface in the region frame and in the native
GOTCHA frame (binary STL, metres), a solid occupancy of the region frame (surface
rasterized, small gaps closed, enclosed space filled; the scanline-parity voxelizer
of ``scripts/eval_b787_geometry_metrics.py`` assumes a watertight mesh, which a
multi-part car is not), a quick-look figure and ``manifest.json``.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
import struct

import numpy as np

SOURCE = dict(title='Toyota Camry (Mk4)(XV20) 1997', author='Nieve5677', license='CC Attribution (Sketchfab)',
              url='https://sketchfab.com/3d-models/toyota-camry-mk4xv20-1997-196829bf4cbc4b8cb26c50964f35ff27')
XV20_SPEC_M = dict(length=4.765, width=1.785, height=1.430, wheelbase=2.670,
                   source='https://en.wikipedia.org/wiki/Toyota_Camry_(XV20) (sedan infobox)')
EXCLUDED_GROUPS = ('camry_inner',)   # cabin interior; not part of the radar-visible exterior
BODY_GROUP = 'camry_Body'
# Workbook corners, native GOTCHA frame (rift/gotcha_step3_native_complex.py), order LF, LR, RR, RF.
FOOTPRINT_NATIVE = np.array([[21.42, -21.14, 0.03], [21.65, -16.40, 0.03],
                             [19.92, -16.31, 0.01], [19.63, -21.11, 0.02]])
# Table 1 trihedral positions vs their measured phase centres (rift/gotcha_calibration.py, CPU job
# 2156302, gotcha_calibration_hh.json; docs/GOTCHA_FORWARD_MODEL_ALIGNMENT.md, calibration array).
DOCUMENTATION_TO_DATA_OFFSET = dict(
    native_m=[0.06, -0.49, 0.0],
    source='median of 56 HH trihedral detections (7 reflectors x 8 passes, TRAIN sectors), CPU job 2156302, '
           '/scratch/group/p.cis261724.000/RIFT_pvc_runs/gotcha_calibration/gotcha_calibration_hh.json',
    determination=('each reflector repeats to ~0.01 m across passes; between reflectors the offsets scatter by '
                   '0.26 m RMS (Table 1 survey error); a rigid fit adds no significant yaw (-0.085 deg) and '
                   'predicts (-0.07, -0.52) m at the Camry, 69 m from the reflectors; leave-one-out rigid fits give '
                   'native x -0.29..+0.09 and native y -0.70..-0.36 there, so native y (local x) is the well-'
                   'determined component and native x (local y) is uncertain by ~0.2 m'),
    heights='trihedral heights agree with Table 1 within +0.03 m, so z is not shifted',
    assumption='the ground-truth workbook corners share the Table 1 documentation frame',
    independent_check=('rift-4b car-only data footprint (job 2156701 code, +-4.5 m window, z = 1.0/1.5 m): '
                       'centre local (0.4-0.6, 0.1-0.3) m, heading ~175 deg; agrees in local x, only to ~0.2 m in '
                       'local y (an energy centroid on a clutter-dominated scene; the trihedrals are the sharper route)'))


def read_obj(path):
    """Vertices (model units) and fan-triangulated faces with their group names."""
    vertices, triangles, groups = [], [], []
    group = None
    with open(path) as stream:
        for line in stream:
            if line.startswith('v '):
                vertices.append(line.split()[1:4])
            elif line.startswith('g '):
                group = line[2:].strip()
            elif line.startswith('f '):
                ids = [int(token.split('/')[0]) for token in line.split()[1:]]
                ids = [i - 1 if i > 0 else len(vertices) + i for i in ids]
                for k in range(1, len(ids) - 1):
                    triangles.append((ids[0], ids[k], ids[k + 1]))
                    groups.append(group)
    return np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int64), np.asarray(groups)


def model_to_region_axes(v_cm):
    """Model (x left, y rear, z up; cm) -> region frame (x front, y left, z up; m)."""
    return np.stack([-v_cm[:, 1], v_cm[:, 0], v_cm[:, 2]], axis=1) / 100.0


def rigid_fit_2d(source, target):
    """Rotation angle and translation minimizing sum |R s + t - target|^2 (Kabsch in the plane)."""
    ms, mt = source.mean(0), target.mean(0)
    h = (source - ms).T @ (target - mt)
    angle = math.atan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return angle, mt - rotation @ ms, rotation


def write_binary_stl(path, triangles_xyz, header='rift gotcha camry xv20'):
    tris = np.asarray(triangles_xyz, dtype=np.float32)
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.where(length > 0, normals / np.maximum(length, 1e-30), 0).astype(np.float32)
    record = np.zeros(len(tris), dtype=[('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
    record['n'], record['v'] = normals, tris
    with open(path, 'wb') as stream:
        stream.write(header.encode()[:80].ljust(80, b' '))
        stream.write(struct.pack('<I', len(tris)))
        stream.write(record.tobytes())


def edge_statistics(triangles):
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return {str(k): int(v) for k, v in sorted(collections.Counter(counts.tolist()).items())}


def solid_occupancy(tris, voxel, half_extent, closing, seed=0):
    """Rasterize the surface onto a region-frame grid, close small gaps, fill enclosed space."""
    from scipy import ndimage
    n = int(math.ceil(2 * half_extent / voxel))
    axis = (np.arange(n) + 0.5) * voxel - half_extent
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    samples = int(area.sum() / (voxel / 4) ** 2)
    rng = np.random.default_rng(seed)
    surface = np.zeros((n, n, n), dtype=bool)
    for start in range(0, samples, 2_000_000):
        m = min(2_000_000, samples - start)
        idx = rng.choice(len(tris), size=m, p=area / area.sum())
        u, v = rng.random(m), rng.random(m)
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        p = a[idx] + u[:, None] * (b[idx] - a[idx]) + v[:, None] * (c[idx] - a[idx])
        ijk = np.floor((p + half_extent) / voxel).astype(int)
        ok = ((ijk >= 0) & (ijk < n)).all(1)
        surface[tuple(ijk[ok].T)] = True
    closed = ndimage.binary_closing(surface, iterations=closing) if closing else surface
    solid = ndimage.binary_fill_holes(closed)
    return axis, surface, solid, samples


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--obj', type=Path, default=Path('data/Toyota_Camry_(Mk4)_(XV20)_1997.obj'))
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='default data/meshes/camry_xv20, or data/meshes/camry_xv20_data_frame with --data-frame-offset')
    parser.add_argument('--data-frame-offset', action='store_true',
                        help='move the workbook corners by DOCUMENTATION_TO_DATA_OFFSET before the fit')
    parser.add_argument('--no-spec-scale', action='store_true', help="keep the model's own proportions (cm -> m only)")
    parser.add_argument('--voxel', type=float, default=0.02, help='solid-occupancy voxel size in metres')
    parser.add_argument('--closing', type=int, default=2, help='binary-closing iterations before the fill')
    parser.add_argument('--region', default='camry',
                        help='GOTCHA region whose local frame the outputs use (default camry; the tuning box is camry_box_v2)')
    parser.add_argument('--region-config', type=Path, help='extra region placements (e.g. rift_pvc/regions/camry_box_v2.json)')
    args = parser.parse_args(argv)
    if args.output_dir is None and args.region != 'camry':
        raise SystemExit('--output-dir is required for a region other than camry')
    if args.output_dir is None:
        args.output_dir = Path('data/meshes/camry_xv20_data_frame' if args.data_frame_offset else 'data/meshes/camry_xv20')

    from rift.gotcha_dataset import load_region
    region = load_region(args.region, args.region_config)
    vertices_cm, triangles, groups = read_obj(args.obj)
    keep = ~np.isin(groups, EXCLUDED_GROUPS)
    local = model_to_region_axes(vertices_cm)

    body = local[np.unique(triangles[groups == BODY_GROUP])]
    body_lo, body_hi = body.min(0), body.max(0)
    ground = local[:, 2].min()
    model_dims = dict(length=float(body_hi[0] - body_lo[0]), width=float(body_hi[1] - body_lo[1]),
                      height=float(local[:, 2].max() - ground))
    centre = np.array([(body_lo[0] + body_hi[0]) / 2, (body_lo[1] + body_hi[1]) / 2, ground])
    scale = (np.ones(3) if args.no_spec_scale else
             np.array([XV20_SPEC_M['length'] / model_dims['length'], XV20_SPEC_M['width'] / model_dims['width'],
                       XV20_SPEC_M['height'] / model_dims['height']]))
    canonical = (local - centre) * scale      # body centre at the origin, wheels on z = 0
    length, width = model_dims['length'] * scale[0], model_dims['width'] * scale[1]

    offset_native = np.asarray(DOCUMENTATION_TO_DATA_OFFSET['native_m']) if args.data_frame_offset else np.zeros(3)
    workbook_local = region.to_local(FOOTPRINT_NATIVE)
    corners_local = region.to_local(FOOTPRINT_NATIVE + offset_native)
    offset_local = (corners_local - workbook_local).mean(0)
    rectangle = np.array([[length / 2, width / 2], [-length / 2, width / 2],
                          [-length / 2, -width / 2], [length / 2, -width / 2]])      # LF, LR, RR, RF
    yaw, shift, rotation = rigid_fit_2d(rectangle, corners_local[:, :2])
    fitted = rectangle @ rotation.T + shift
    residual = np.linalg.norm(fitted - corners_local[:, :2], axis=1)
    ground_z = float(corners_local[:, 2].mean())
    placed = canonical.copy()
    placed[:, :2] = canonical[:, :2] @ rotation.T + shift
    placed[:, 2] += ground_z
    native = placed @ np.asarray(region.rotation_local_to_native).T + np.asarray(region.translation_m)

    tris_local = placed[triangles[keep]]
    tris_native = native[triangles[keep]]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    write_binary_stl(out/'camry_xv20_region_local.stl', tris_local)
    write_binary_stl(out/'camry_xv20_native.stl', tris_native)
    axis, surface, solid, samples = solid_occupancy(tris_local, args.voxel, region.half_extent_m, args.closing)
    np.savez_compressed(out/'camry_xv20_solid_region_local.npz', axis_m=axis, occupancy=solid, surface=surface,
                        voxel_m=args.voxel, closing_iterations=args.closing)
    solid_volume = float(solid.sum()) * args.voxel ** 3
    bbox = tris_local.reshape(-1, 3)
    placed_dims = (bbox.max(0) - bbox.min(0)).tolist()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, (top, side) = plt.subplots(1, 2, figsize=(12, 5))
    sample = tris_local.reshape(-1, 3)[::7]
    top.scatter(sample[:, 0], sample[:, 1], s=0.05, c='0.4')
    if args.data_frame_offset:
        top.plot(*np.vstack([workbook_local[:, :2], workbook_local[:1, :2]]).T, ':', color='0.5', label='workbook corners (as documented)')
    top.plot(*np.vstack([corners_local[:, :2], corners_local[:1, :2]]).T, 'r-o',
             label='workbook corners + calibration offset' if args.data_frame_offset else 'workbook corners')
    top.plot(*np.vstack([fitted, fitted[:1]]).T, 'b--', label='fitted XV20 footprint')
    for name, corner in zip(('LF', 'LR', 'RR', 'RF'), corners_local):
        top.annotate(name, corner[:2], color='r')
    top.set(aspect='equal', xlabel='region x (front) [m]', ylabel='region y (left) [m]', title='top view')
    top.legend(loc='lower right', fontsize=8)
    side.scatter(sample[:, 0], sample[:, 2], s=0.05, c='0.4')
    side.axhline(ground_z, color='r', lw=0.8)
    side.set(aspect='equal', xlabel='region x (front) [m]', ylabel='z [m]', title='side view')
    figure.suptitle('GOTCHA Camry: XV20 stand-in registered to the workbook footprint'
                    + (' + documentation-to-data offset' if args.data_frame_offset else '') + ' (region frame)')
    figure.tight_layout()
    figure.savefig(out/'camry_xv20_registration.png', dpi=150)

    manifest = dict(
        schema='rift_gotcha_camry_mesh_v1', source=dict(SOURCE, file=str(args.obj)),
        stand_in='same-generation model (XV20, user identification 2026-09-22), not a scan of the GOTCHA car',
        excluded_groups=list(EXCLUDED_GROUPS),
        triangles=dict(total=int(len(triangles)), kept=int(keep.sum())),
        axes='model (x left, y rear, z up, cm) -> region (x front, y left, z up, m)',
        model_dims_m=model_dims, xv20_spec_m=XV20_SPEC_M,
        spec_scale=None if args.no_spec_scale else dict(zip(('x', 'y', 'z'), scale.round(6).tolist())),
        pose=dict(frame=f'GOTCHA region "{region.name}" (region.to_local)', yaw_deg=math.degrees(yaw),
                  translation_m=shift.tolist(), ground_z_m=ground_z,
                  footprint_corner_residual_m=dict(zip(('LF', 'LR', 'RR', 'RF'), residual.round(4).tolist())),
                  footprint_rms_m=float(np.sqrt((residual ** 2).mean())), workbook_tolerance_m=0.10,
                  documentation_to_data_offset=(dict(DOCUMENTATION_TO_DATA_OFFSET, applied=True,
                                                     local_m=offset_local.round(4).tolist())
                                                if args.data_frame_offset else dict(applied=False))),
        placed_bbox_dims_m=placed_dims, region=region.as_dict(),
        edge_sharing_counts=edge_statistics(triangles[keep]),
        solid=dict(voxel_m=args.voxel, closing_iterations=args.closing, surface_samples=samples,
                   volume_m3=solid_volume, surface_voxels=int(surface.sum()), solid_voxels=int(solid.sum()),
                   fraction_of_spec_box=solid_volume / (XV20_SPEC_M['length'] * XV20_SPEC_M['width'] * XV20_SPEC_M['height'])),
        outputs=['camry_xv20_region_local.stl', 'camry_xv20_native.stl', 'camry_xv20_solid_region_local.npz',
                 'camry_xv20_registration.png'])
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: manifest[k] for k in ('model_dims_m', 'spec_scale', 'pose', 'placed_bbox_dims_m', 'triangles',
                                               'edge_sharing_counts', 'solid')}, indent=2))


if __name__ == '__main__':
    main()
