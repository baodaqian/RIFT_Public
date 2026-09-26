#!/usr/bin/env python3
"""Flatten a Collada (.dae) scene to one OBJ in metres (new-target mesh intake, docs/RIFT_GOTCHA_Tune.md A76).

SketchUp / 3D Warehouse models arrive as Collada: a node hierarchy of instanced geometries in the file's unit (often
inches, ``<unit meter="0.0254">``) with its own up axis. This walks the visual scene with ``pycollada`` (bound
geometries carry every parent transform), keeps triangle sets and polygon lists (triangulated), drops lines, scales
to metres by the file's unit and writes one OBJ with the file's axes unchanged, plus a small JSON record. The axis
roles (length, up, front) are then given to ``prepare_gotcha_target_mesh.py``.

``pycollada`` is not in the PVC environment; run with it on PYTHONPATH, e.g. ``pip install --target <dir> pycollada``.

    PYTHONPATH=<dir>:. python scripts_pvc/gotcha_target_dae_to_obj.py model.dae --output model_m.obj
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv=None):
    import collada
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('dae', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    mesh = collada.Collada(str(args.dae), ignore=[collada.common.DaeUnsupportedError, collada.common.DaeBrokenRefError])
    unit = float(mesh.assetInfo.unitmeter or 1.0)
    up = str(mesh.assetInfo.upaxis)
    blocks, counts = [], dict(triangles=0, polylists=0, skipped=0, geometries=0)
    for bound in mesh.scene.objects('geometry'):
        counts['geometries'] += 1
        for prim in bound.primitives():
            if isinstance(prim, collada.triangleset.BoundTriangleSet):
                tris = prim
                counts['triangles'] += 1
            elif isinstance(prim, (collada.polylist.BoundPolylist, collada.polygons.BoundPolygons)):
                tris = prim.triangleset()
                counts['polylists'] += 1
            else:
                counts['skipped'] += 1
                continue
            if len(tris.vertex_index) == 0:
                continue
            blocks.append(np.asarray(tris.vertex, dtype=np.float64)[np.asarray(tris.vertex_index)] * unit)
    triangles = np.concatenate(blocks)                                   # [T, 3, 3] metres
    vertices, inverse = np.unique(triangles.reshape(-1, 3).round(7), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as out:
        out.write(f'# flattened from {args.dae.name} by gotcha_target_dae_to_obj.py; metres; source up axis {up}\n')
        np.savetxt(out, vertices, fmt='v %.6f %.6f %.6f')
        np.savetxt(out, faces + 1, fmt='f %d %d %d')
    record = dict(source=str(args.dae), unit_meter=unit, up_axis=up, vertices=int(len(vertices)), triangles=int(len(faces)),
                  bbox_min_m=vertices.min(0).tolist(), bbox_max_m=vertices.max(0).tolist(),
                  extent_m=(vertices.max(0) - vertices.min(0)).tolist(), primitives=counts,
                  authoring_tool=str(getattr(mesh.assetInfo.contributors[0], 'authoring_tool', '')) if mesh.assetInfo.contributors else '')
    args.output.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
