#!/usr/bin/env python3
"""Render matched planform and nose-on panels from the rejected SE2 smoke mesh.

This renderer reads the retained postmortem mesh only.  It deliberately fails
closed unless the mesh audit still identifies the rejected step-120 smoke
checkpoint and exactly reproduces its recorded topology.  The projections use
the same readout axes and inferno-on-black palette as the B787 baseline-field
comparison: x/z for planform and z/y for nose-on.  Colour distinguishes the
largest component from the other 38 components; it does not encode intensity.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection


EXPECTED_TOPOLOGY = {
    "vertices": 4958,
    "faces": 9800,
    "edges": 14700,
    "boundary_edges": 0,
    "nonmanifold_edges": 0,
    "components": 39,
}

# Keep every checkpoint in a fixed, shared screen.  The retained mesh extends
# just beyond -0.075 m on y, so 0.08 m avoids silently clipping valid geometry
# while retaining the compact framing of the existing comparison panels.
COMMON_VIEW_LIMIT_M = 0.08
BACKGROUND = "#000004"
VIEW_SPECS = (
    ("top_down", 0, 2, "x", "z", "project along y"),
    ("front_nose_on", 2, 1, "z", "y", "project along x"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, help="Audited rejected_smoke_mesh.npz input")
    parser.add_argument("--audit", required=True, help="Matching rejected_smoke_mesh_audit.json input")
    parser.add_argument("--output-dir", required=True, help="New diagnostic-only output directory")
    return parser.parse_args()


def largest_component_face_mask(vertex_count: int, faces: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Return the original mesh's largest-component face mask without changing it."""
    parent = np.arange(vertex_count, dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first, second, third in faces:
        union(int(first), int(second))
        union(int(second), int(third))
    roots = np.fromiter((find(index) for index in range(vertex_count)), dtype=np.int64)
    face_roots = roots[faces[:, 0]]
    component_roots, face_counts = np.unique(face_roots, return_counts=True)
    largest_root = int(component_roots[np.argmax(face_counts)])
    largest = face_roots == largest_root
    return largest, int(largest.sum()), int((~largest).sum())


def checked_mesh(mesh_path: Path, audit_path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    if not mesh_path.is_file() or mesh_path.name != "rejected_smoke_mesh.npz":
        raise FileNotFoundError("expected the audited rejected_smoke_mesh.npz input")
    if not audit_path.is_file() or audit_path.name != "rejected_smoke_mesh_audit.json":
        raise FileNotFoundError("expected the matching rejected_smoke_mesh_audit.json input")
    if mesh_path.parent != audit_path.parent:
        raise ValueError("mesh and audit must come from the same postmortem directory")

    with audit_path.open(encoding="utf-8") as handle:
        audit = json.load(handle)
    if not isinstance(audit, Mapping) or audit.get("accepted_surface") is not False:
        raise RuntimeError("input audit does not identify a rejected postmortem surface")
    if audit.get("checkpoint_step") != 120:
        raise RuntimeError("input audit is not the retained step-120 smoke checkpoint")
    topology = audit.get("topology")
    if not isinstance(topology, Mapping):
        raise RuntimeError("input audit lacks topology")
    mismatches = {
        name: {"expected": expected, "actual": topology.get(name)}
        for name, expected in EXPECTED_TOPOLOGY.items()
        if topology.get(name) != expected
    }
    if topology.get("passed") is not False:
        mismatches["passed"] = {"expected": False, "actual": topology.get("passed")}
    if mismatches:
        raise RuntimeError(f"input audit does not match the rejected smoke mesh: {mismatches}")

    with np.load(mesh_path) as stored:
        vertices = np.asarray(stored["vertices"], dtype=np.float32)
        faces = np.asarray(stored["faces"], dtype=np.int32)
        checkpoint_step = int(stored["checkpoint_step"])
    if vertices.shape != (EXPECTED_TOPOLOGY["vertices"], 3):
        raise RuntimeError(f"unexpected vertex array shape: {vertices.shape}")
    if faces.shape != (EXPECTED_TOPOLOGY["faces"], 3):
        raise RuntimeError(f"unexpected face array shape: {faces.shape}")
    if checkpoint_step != 120:
        raise RuntimeError(f"expected step 120 in mesh, found {checkpoint_step}")
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise RuntimeError("mesh faces reference vertices outside the audited mesh")
    return vertices, faces, int(topology["components"]), checkpoint_step


def projected_triangles(vertices: np.ndarray, faces: np.ndarray, horizontal_axis: int, vertical_axis: int) -> np.ndarray:
    """Project mesh faces without rotating or altering the retained geometry."""
    return vertices[faces][:, :, [horizontal_axis, vertical_axis]]


def component_face_colours(largest: np.ndarray) -> np.ndarray:
    """Use the existing inferno visual language without implying a field value."""
    face_color = np.empty((len(largest), 4), dtype=np.float64)
    inferno = plt.get_cmap("inferno")
    face_color[largest] = inferno(0.80)
    face_color[~largest] = inferno(0.48)
    return face_color


def save_projection(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_colours: np.ndarray,
    horizontal_axis: int,
    vertical_axis: int,
) -> None:
    triangles = projected_triangles(vertices, faces, horizontal_axis, vertical_axis)
    figure = plt.figure(figsize=(4.4, 4.4), facecolor=BACKGROUND)
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.set_facecolor(BACKGROUND)
    mesh = PolyCollection(triangles, facecolors=face_colours, edgecolors="none", linewidths=0.0, alpha=1.0)
    axis.add_collection(mesh)
    axis.set_xlim(-COMMON_VIEW_LIMIT_M, COMMON_VIEW_LIMIT_M)
    axis.set_ylim(-COMMON_VIEW_LIMIT_M, COMMON_VIEW_LIMIT_M)
    axis.set_aspect("equal", adjustable="box")
    axis.set_axis_off()
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".png", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        figure.savefig(temporary, dpi=190, facecolor=figure.get_facecolor(), pad_inches=0)
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    mesh_path = Path(args.mesh).resolve()
    audit_path = Path(args.audit).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse diagnostic output directory: {output_dir}")
    if mesh_path.parent == output_dir or mesh_path.parent in output_dir.parents:
        raise ValueError("top-view output must be outside the retained postmortem directory")

    vertices, faces, components, step = checked_mesh(mesh_path, audit_path)
    output_dir.mkdir(parents=True)
    largest, largest_faces, satellite_faces = largest_component_face_mask(len(vertices), faces)
    face_colours = component_face_colours(largest)
    top_path = output_dir / "rejected_smoke_top_down_xz.png"
    front_path = output_dir / "rejected_smoke_front_nose_on_zy.png"
    save_projection(top_path, vertices, faces, face_colours, horizontal_axis=0, vertical_axis=2)
    save_projection(front_path, vertices, faces, face_colours, horizontal_axis=2, vertical_axis=1)
    manifest_path = output_dir / "rejected_smoke_matched_views_audit.json"
    manifest_path.write_text(
        json.dumps(
            {
                "accepted_surface": False,
                "checkpoint_step": step,
                "components": components,
                "largest_component_faces": largest_faces,
                "other_component_faces": satellite_faces,
                "common_view_limit_m": COMMON_VIEW_LIMIT_M,
                "palette": {
                    "background": BACKGROUND,
                    "largest_component": "inferno(0.80)",
                    "other_components": "inferno(0.48)",
                    "meaning": "categorical component role, not field intensity",
                },
                "views": [
                    {
                        "file": str(top_path),
                        "name": "top_down",
                        "horizontal_axis": "x",
                        "vertical_axis": "z",
                        "projection_axis": "y",
                    },
                    {
                        "file": str(front_path),
                        "name": "front_nose_on",
                        "horizontal_axis": "z",
                        "vertical_axis": "y",
                        "projection_axis": "x",
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "accepted_surface": False,
                "checkpoint_step": step,
                "components": components,
                "top_view_file": str(top_path),
                "front_view_file": str(front_path),
                "audit_file": str(manifest_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
