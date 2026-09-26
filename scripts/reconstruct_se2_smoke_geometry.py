#!/usr/bin/env python3
"""Reconstruct the rejected SE2 smoke mesh without touching its run directory.

This is deliberately a postmortem diagnostic rather than an exporter: the
retained step-120 checkpoint is evaluated with the run's serialized mesh
settings, then its marching-cubes mesh must reproduce the topology recorded in
the failed Slurm recovery log.  A mismatch fails closed and produces no plot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

from rift.sugavanam_ertin import FourierFeatureSDF
from rift.sugavanam_ertin_validzero import (
    evaluate_field_grid,
    field_validity_from_array,
    mesh_topology_audit,
)


SMOKE_DIRECTORY = "b787_sugavanam_ertin_validzero_v2p1_smoke"
EXPECTED_TOPOLOGY = {
    "vertices": 4958,
    "faces": 9800,
    "edges": 14700,
    "boundary_edges": 0,
    "nonmanifold_edges": 0,
    "components": 39,
}
EXPECTED_ROI_CLEARANCE = 0.07422332763671874


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Retained smoke checkpoint_latest.pth.tar")
    parser.add_argument("--output-dir", required=True, help="New diagnostic-only output directory")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    return parser.parse_args()


def checkpoint_arg(arguments: object, name: str) -> object:
    if isinstance(arguments, Mapping):
        value = arguments.get(name)
    else:
        value = getattr(arguments, name, None)
    if value is None:
        raise KeyError(f"checkpoint has no serialized argument {name!r}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_dump(value: object, path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_npz_dump(path: Path, **arrays: object) -> None:
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def face_component_mask(vertex_count: int, faces: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Return a face mask for the largest connected component without changing the mesh."""
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


def save_plot(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    extent: float,
    topology: Mapping[str, object],
) -> None:
    largest, largest_faces, satellite_faces = face_component_mask(len(vertices), faces)
    triangles = vertices[faces]
    face_color = np.empty((len(faces), 4), dtype=np.float64)
    face_color[largest] = (0.10, 0.43, 0.69, 1.0)
    face_color[~largest] = (0.90, 0.30, 0.12, 1.0)

    figure = plt.figure(figsize=(14.0, 5.1), facecolor="#fbfbf9")
    views = ((25, -55, "isometric"), (7, -90, "front"), (8, 0, "side"))
    for index, (elevation, azimuth, label) in enumerate(views, start=1):
        axis = figure.add_subplot(1, 3, index, projection="3d")
        mesh = Poly3DCollection(
            triangles,
            facecolors=face_color,
            edgecolors="none",
            linewidths=0.0,
            alpha=1.0,
        )
        axis.add_collection3d(mesh)
        axis.set_xlim(-extent, extent)
        axis.set_ylim(-extent, extent)
        axis.set_zlim(-extent, extent)
        axis.set_box_aspect((1, 1, 1))
        axis.set_proj_type("ortho")
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_axis_off()
        axis.set_title(label, fontsize=10.5, color="#3d3c39", pad=-8)

    figure.suptitle(
        "SE2 baseline smoke — postmortem reconstruction from rejected step-120 checkpoint",
        x=0.02,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color="#171716",
    )
    figure.text(
        0.02,
        0.895,
        "Actual marching-cubes mesh; topology gate failed: "
        f"{topology['components']} disconnected components. Not an accepted or reportable surface.",
        ha="left",
        va="center",
        fontsize=10.5,
        color="#514f4a",
    )
    figure.legend(
        handles=(
            Patch(facecolor="#1a6eaf", label=f"largest component ({largest_faces:,} faces)"),
            Patch(facecolor="#e64d1f", label=f"other 38 components ({satellite_faces:,} faces)"),
        ),
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.02),
    )
    figure.subplots_adjust(left=0.005, right=0.995, top=0.84, bottom=0.15, wspace=0.01)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".png", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        figure.savefig(temporary, dpi=150, facecolor=figure.get_facecolor())
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing retained smoke checkpoint: {checkpoint}")
    if checkpoint.name != "checkpoint_latest.pth.tar" or checkpoint.parent.name != SMOKE_DIRECTORY:
        raise ValueError("refusing a checkpoint other than the retained SE2 v2p1 smoke latest checkpoint")
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse diagnostic output directory: {output_dir}")
    if checkpoint.parent == output_dir or checkpoint.parent in output_dir.parents:
        raise ValueError("diagnostic output must not be created inside the retained smoke directory")
    if args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if args.device == "cpu":
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(1)

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(state.get("step", -1)) != 120:
        raise RuntimeError(f"expected a step-120 smoke checkpoint, got step={state.get('step')!r}")
    checkpoint_arguments = state.get("args")
    if checkpoint_arguments is None or str(checkpoint_arg(checkpoint_arguments, "run_kind")) != "smoke":
        raise RuntimeError("checkpoint is not serialized as the SE2 smoke run")
    model_config = state.get("model_config")
    model_state = state.get("model_state_dict")
    if not isinstance(model_config, Mapping) or not isinstance(model_state, Mapping):
        raise RuntimeError("checkpoint lacks a serialized FourierFeatureSDF model")

    extent = float(state["extent"])
    mesh_grid = int(checkpoint_arg(checkpoint_arguments, "mesh_grid"))
    grid_chunk = int(checkpoint_arg(checkpoint_arguments, "grid_chunk"))
    boundary_margin = float(checkpoint_arg(checkpoint_arguments, "boundary_margin"))
    device = torch.device(args.device)
    model = FourierFeatureSDF(**model_config).to(device)
    model.load_state_dict(model_state)
    model.eval()

    field, pitch = evaluate_field_grid(model, extent, mesh_grid, device, grid_chunk)
    validity = field_validity_from_array(field, boundary_margin)
    validity.update({"grid": mesh_grid, "pitch": float(pitch)})
    if not validity["passed"]:
        raise RuntimeError(f"postmortem field no longer passes the smoke validity gate: {validity}")

    vertices, faces, normals, _ = marching_cubes(field, level=0.0, spacing=(pitch, pitch, pitch))
    vertices += -extent
    topology = mesh_topology_audit(vertices, faces)
    clearance = extent - np.abs(vertices).max(axis=1)
    topology["roi_clearance_min"] = float(clearance.min())
    topology["inside_roi"] = bool(np.all(clearance > 0.0))
    topology["passed"] = bool(topology["passed"] and topology["inside_roi"])

    mismatches = {
        name: {"expected": expected, "actual": topology.get(name)}
        for name, expected in EXPECTED_TOPOLOGY.items()
        if topology.get(name) != expected
    }
    if topology["inside_roi"] is not True:
        mismatches["inside_roi"] = {"expected": True, "actual": topology["inside_roi"]}
    if not np.isclose(topology["roi_clearance_min"], EXPECTED_ROI_CLEARANCE, atol=1e-6, rtol=0.0):
        mismatches["roi_clearance_min"] = {
            "expected": EXPECTED_ROI_CLEARANCE,
            "actual": topology["roi_clearance_min"],
        }
    if topology["passed"]:
        mismatches["passed"] = {"expected": False, "actual": True}
    if mismatches:
        raise RuntimeError(f"postmortem reconstruction did not match the logged failed mesh: {mismatches}")

    output_dir.mkdir(parents=True)
    mesh_path = output_dir / "rejected_smoke_mesh.npz"
    plot_path = output_dir / "rejected_smoke_mesh.png"
    audit_path = output_dir / "rejected_smoke_mesh_audit.json"
    atomic_npz_dump(
        mesh_path,
        vertices=vertices.astype(np.float32),
        faces=faces.astype(np.int32),
        vertex_normals=normals.astype(np.float32),
        extent=np.float32(extent),
        pitch=np.float32(pitch),
        checkpoint_step=np.int32(state["step"]),
    )
    save_plot(plot_path, vertices, faces, extent, topology)
    atomic_json_dump(
        {
            "purpose": "postmortem reconstruction of rejected SE2 v2p1 smoke geometry",
            "accepted_surface": False,
            "ground_truth_geometry_used": False,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "checkpoint_step": int(state["step"]),
            "mesh_grid": mesh_grid,
            "grid_chunk": grid_chunk,
            "device": str(device),
            "validity": validity,
            "topology": topology,
            "mesh_file": str(mesh_path),
            "plot_file": str(plot_path),
        },
        audit_path,
    )
    print(f"wrote rejected mesh: {mesh_path}")
    print(f"wrote rejected mesh plot: {plot_path}")
    print(f"wrote audit: {audit_path}")
    print(json.dumps({"validity": validity, "topology": topology}, sort_keys=True))


if __name__ == "__main__":
    main()
