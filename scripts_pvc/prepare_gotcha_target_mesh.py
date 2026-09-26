#!/usr/bin/env python3
"""Register a stand-in mesh to a new GOTCHA target box for 3D evaluation (new targets, docs/RIFT_GOTCHA_Tune.md A76).

A separate copy of the Camry's ``prepare_gotcha_camry_mesh.py`` idea for the new targets (Nissan Sentra, Hyundai Santa
Fe). GOTCHA supplies no 3D truth and, without the target-location workbook, no footprint corners, so the pose comes
from the data: the target region (``rift_pvc/regions/gotcha_new_targets.json``) is centred on the TRAIN-only locate
estimate with local +x along the estimated long axis (``gotcha_target_locate.py``). In that frame (x along the car, y
left, z up) the mesh is:

* read from OBJ (fan-triangulated; one object or many), its model axes named by ``--length-axis``, ``--up-axis`` and
  ``--front-sign`` (front = sign x length axis); left = up x front, so the result is a proper rotation;
* scaled per axis to the stated specification (``--spec-length``, ``--spec-width`` without mirrors, measured on the
  band 20-45% of the height, below the mirrors; ``--spec-height``, the 99.5% height quantile, which drops antennas),
  as the Camry was scaled to its XV20 specification; ``--no-spec-scale`` keeps the model's proportions, scaled
  uniformly to ``--spec-length``;
* centred on the region origin in x and y, wheels (the 0.05% lowest height) on the native ground ``--ground-native-z``
  (the Camry's workbook corners average 0.02 m in the same lot);
* written twice, front toward local +x and toward local -x (``front_plusx`` / ``front_minusx``): a 2D image fixes the
  long axis but not which end is the front, so the orientation is left for a data check.

Outputs per orientation in ``--output-dir``/<orientation>/: the surface in the region frame and the native frame (binary
STL, metres), the solid occupancy of the region frame (surface rasterized, gaps closed, enclosed space filled, as the
Camry's), ``manifest.json`` and a quick-look figure.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def read_obj(path):
    vertices, triangles = [], []
    with open(path) as stream:
        for line in stream:
            if line.startswith('v '):
                vertices.append(line.split()[1:4])
            elif line.startswith('f '):
                ids = [int(token.split('/')[0]) for token in line.split()[1:]]
                ids = [i - 1 if i > 0 else len(vertices) + i for i in ids]
                for k in range(1, len(ids) - 1):
                    triangles.append((ids[0], ids[k], ids[k + 1]))
    return np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int64)


def write_binary_stl(path, triangles_xyz, header):
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


def solid_occupancy(tris, voxel, half_extent, closing, seed=0):
    """Rasterize the surface onto a region-frame grid, close small gaps, fill enclosed space (the Camry recipe)."""
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


def to_car_frame(v, length_axis, up_axis, front_sign):
    """Model -> car frame (x front, y left, z up), model units."""
    front = np.zeros(3); front[length_axis] = front_sign
    up = np.zeros(3); up[up_axis] = 1.0
    left = np.cross(up, front)
    return np.stack([v @ front, v @ left, v @ up], axis=1)


def main(argv=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from rift.gotcha_dataset import load_region
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--obj', type=Path, required=True)
    p.add_argument('--target', required=True, help='file stem, e.g. santafe_2004')
    p.add_argument('--region', required=True)
    p.add_argument('--region-config', type=Path, default=Path('rift_pvc/regions/gotcha_new_targets.json'))
    p.add_argument('--length-axis', type=int, required=True)
    p.add_argument('--up-axis', type=int, required=True)
    p.add_argument('--front-sign', type=int, choices=(-1, 1), required=True)
    p.add_argument('--spec-length', type=float, required=True)
    p.add_argument('--spec-width', type=float, required=True, help='body width without mirrors (m)')
    p.add_argument('--spec-height', type=float, required=True)
    p.add_argument('--spec-source', required=True)
    p.add_argument('--no-spec-scale', action='store_true')
    p.add_argument('--ground-native-z', type=float, default=0.02)
    p.add_argument('--source-json', required=True, help='{"title","author","license","url","file"} of the mesh')
    p.add_argument('--stand-in', required=True, help='what the mesh is relative to the GOTCHA vehicle')
    p.add_argument('--voxel', type=float, default=0.02)
    p.add_argument('--closing', type=int, default=2)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(argv)
    region = load_region(args.region, args.region_config)
    extent = float(region.half_extent_m)
    v, tri = read_obj(args.obj)
    car = to_car_frame(v, args.length_axis, args.up_axis, args.front_sign)
    lo, hi = car.min(0), car.max(0)
    z0 = np.quantile(car[:, 2], 0.0005)
    height_model = float(np.quantile(car[:, 2], 0.995) - z0)
    band = (car[:, 2] > z0 + 0.20 * height_model) & (car[:, 2] < z0 + 0.45 * height_model)
    width_model = float(car[band, 1].max() - car[band, 1].min())
    length_model = float(hi[0] - lo[0])
    if args.no_spec_scale:
        scale = np.full(3, args.spec_length / length_model)
    else:
        scale = np.array([args.spec_length / length_model, args.spec_width / width_model, args.spec_height / height_model])
    centre_xy = np.array([(hi[0] + lo[0]) / 2, (car[band, 1].max() + car[band, 1].min()) / 2])
    local_ground = args.ground_native_z - float(region.translation_m[2])
    base = np.empty_like(car)
    base[:, 0] = (car[:, 0] - centre_xy[0]) * scale[0]
    base[:, 1] = (car[:, 1] - centre_xy[1]) * scale[1]
    base[:, 2] = (car[:, 2] - z0) * scale[2] + local_ground
    report = dict(model_extent_units=dict(length=length_model, width_below_mirrors=width_model, height_995=height_model),
                  scale_per_axis=scale.tolist(), spec=dict(length=args.spec_length, width=args.spec_width,
                  height=args.spec_height, source=args.spec_source, scaled='per axis' if not args.no_spec_scale else 'uniform'))
    R = np.asarray(region.rotation_local_to_native, dtype=np.float64)
    t = np.asarray(region.translation_m, dtype=np.float64)
    for name, yaw in (('front_plusx', 0.0), ('front_minusx', 180.0)):
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        local = base @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        out = args.output_dir / name
        out.mkdir(parents=True, exist_ok=True)
        tris_local = local[tri]
        write_binary_stl(out / f'{args.target}_region_local.stl', tris_local, f'rift gotcha {args.target} {name} region')
        write_binary_stl(out / f'{args.target}_native.stl', local[tri] @ R.T + t, f'rift gotcha {args.target} {name} native')
        axis, surface, solid, samples = solid_occupancy(tris_local, args.voxel, extent, args.closing)
        np.savez_compressed(out / f'{args.target}_solid_region_local.npz', occupancy=solid, surface=surface,
                            axis_m=axis, voxel_m=args.voxel)
        occupied = np.argwhere(solid)
        dims = ((occupied.max(0) - occupied.min(0) + 1) * args.voxel).tolist() if len(occupied) else None
        manifest = dict(schema='rift_gotcha_target_mesh_v1', target=args.target, orientation=name, yaw_deg=yaw,
                        source=json.loads(args.source_json), stand_in=args.stand_in,
                        pose=dict(frame=f'{args.region} region local (x along the located long axis, y left, z up)',
                                  centre='region origin (the TRAIN-only locate estimate)',
                                  front='local +x' if yaw == 0 else 'local -x',
                                  ground_native_z=args.ground_native_z, ground_local_z=local_ground,
                                  provenance='docs/RIFT_GOTCHA_Tune.md A75/A76; the orientation is fixed by a later data check'),
                        region=dict(name=region.name, target_id=region.target_id, translation_m=list(region.translation_m),
                                    rotation_local_to_native=[list(r) for r in region.rotation_local_to_native],
                                    half_extent_m=region.half_extent_m, placement_provenance=region.placement_provenance),
                        triangles=int(len(tri)), vertices=int(len(v)), placed_bbox_dims_m=(local.max(0) - local.min(0)).tolist(),
                        solid=dict(voxel_m=args.voxel, closing=args.closing, surface_samples=samples,
                                   occupied_voxels=int(solid.sum()), occupied_bbox_m=dims),
                        outputs=dict(region_stl=f'{args.target}_region_local.stl', native_stl=f'{args.target}_native.stl',
                                     solid=f'{args.target}_solid_region_local.npz'), **report)
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        pick = np.random.default_rng(0).choice(len(local), min(150000, len(local)), replace=False)
        sample = local[pick]
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, (i, j, title) in zip(axes, [(0, 1, 'top (x, y)'), (0, 2, 'side (x, z)'), (1, 2, 'front (y, z)')]):
            ax.scatter(sample[:, i], sample[:, j], s=0.05, c='k')
            ax.axhline(local_ground if j == 2 else 0, color='tab:red', lw=0.6)
            ax.set_xlim(-extent, extent) if i == 0 else ax.set_xlim(-1.5, 1.5)
            ax.set_aspect('equal')
            ax.set_title(f'{args.target} {name}: {title}', fontsize=9)
        figure.tight_layout()
        figure.savefig(out / f'{args.target}_quicklook.png', dpi=90)
        plt.close(figure)
        print(f"{name}: bbox {np.round(local.max(0) - local.min(0), 3).tolist()} m, solid voxels {int(solid.sum())}", flush=True)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
