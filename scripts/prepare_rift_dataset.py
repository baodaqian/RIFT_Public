#!/usr/bin/env python3
"""Assemble the six-object RIFT dataset using links; never copy/read radar payloads."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from rift.npz_dataset import load_npz_arrays
from rift.rift_dataset import (DEFAULT_ROOT, PROJECT_ROOT, catalog, geometry_transform, metadata_object_id,
                               object_paths, role_manifest, validate_metadata)


def _without_geometry(payload: dict) -> dict:
    """Only these generated mesh fields may change during a geometry refresh."""
    return {**payload, "objects": [
        {key: value for key, value in row.items() if key not in ("geometry", "geometry_transform")}
        for row in payload.get("objects", [])]}


def write_new_or_equal(path: Path, payload: dict, *, refresh_geometry: bool = False,
                       check_only: bool = False) -> None:
    if path.exists():
        previous = json.loads(path.read_text())
        if previous == payload:
            return
        if not refresh_geometry or _without_geometry(previous) != _without_geometry(payload):
            raise ValueError(f"Refusing to overwrite a different manifest: {path}")
    if check_only:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if path.exists() else "x") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def link_existing(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise ValueError(f"Refusing to replace existing entry: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source.resolve(), destination.parent.resolve()))


def prepare(source_root: Path, output: Path, *, check_only: bool = False,
            mesh_root: Path | None = None, refresh_geometry: bool = False) -> dict:
    config = catalog()
    mesh_root = mesh_root if mesh_root is not None else source_root / "RIFT_dataset" / "meshes"
    rows, links = [], []
    reference = None
    # Validate every archive before making any links. Never index response.
    for index, spec in enumerate(config["objects"]):
        source = source_root / spec["source"]
        arrays = load_npz_arrays(source, load_response=False)
        if list(arrays["response_shape"]) != config["response_shape"] or str(arrays["response_dtype"]) != "complex64":
            raise ValueError(f"Unexpected response header: {source}")
        with zipfile.ZipFile(source) as archive:
            if archive.getinfo("response.npy").compress_type != zipfile.ZIP_STORED:
                raise ValueError(f"RIFT dataset requires seekable uncompressed responses: {source}")
        meta = arrays["meta"]
        validate_metadata(meta)
        name = spec["object_id"]
        if metadata_object_id(meta) != name:
            raise ValueError(f"Archive object identity mismatch: {source}")
        poses = {key: arrays[key] for key in ("viewpoint_positions", "tx_pos", "rx_pos")}
        with np.load(source, allow_pickle=False) as archive:
            poses["viewpoint_angles"] = archive["viewpoint_angles"]
        if reference is None:
            reference = poses
            if not np.allclose(np.linalg.norm(poses["viewpoint_positions"], axis=1), 10, rtol=0, atol=1e-8):
                raise ValueError("Viewpoints are not on the 10 m sphere")
        elif any(not np.array_equal(value, reference[key]) for key, value in poses.items()):
            raise ValueError(f"View/antenna geometry differs for {name}")
        archive_path, split_path = object_paths(output, name)
        links.append((source, archive_path))
        row = {"object_id": name, "archive": str(archive_path.relative_to(output.absolute())),
               "role_manifest": str(split_path.relative_to(output.absolute())),
               "global_frame_start": index * 10000, "global_frame_stop": (index + 1) * 10000,
               "num_views": 10000, "metadata": meta}
        geometry = mesh_root / spec["geometry_filename"]
        if not geometry.is_file():
            raise FileNotFoundError(geometry)
        if (meta.get("source_stl_filename") is not None
                and meta.get("source_stl_filename") != geometry.name):
            raise ValueError(f"Source mesh filename disagrees with NPZ metadata: {geometry}")
        destination = output / "meshes" / geometry.name
        links.append((geometry, destination))
        row["geometry"] = str(destination.relative_to(output))
        row["geometry_transform"] = geometry_transform(name, meta)
        rows.append(row)
    manifest = {**config, "objects": rows, "total_frames": 60000,
                "objects_are_separate_archives": True, "geometry_arrays_equal_across_objects": True,
                "assembly": "relative_symlinks_to_original_archives; originals_unchanged"}
    # Preflight every existing entry/manifest before any write, including on refresh.
    for source, destination in links:
        if (destination.exists() or destination.is_symlink()) and destination.resolve() != source.resolve():
            raise ValueError(f"Conflicting dataset entry: {destination}")
    for spec in config["objects"]:
        _, split = object_paths(output, spec["object_id"])
        write_new_or_equal(split, role_manifest(spec["object_id"]), check_only=True)
    manifest_path = output / "dataset_manifest.json"
    write_new_or_equal(manifest_path, manifest, refresh_geometry=refresh_geometry, check_only=True)
    if not check_only:
        for source, destination in links:
            link_existing(source, destination)
        for spec in config["objects"]:
            _, split = object_paths(output, spec["object_id"])
            write_new_or_equal(split, role_manifest(spec["object_id"]))
        write_new_or_equal(manifest_path, manifest, refresh_geometry=refresh_geometry)
    return {"name": config["name"], "root": str(output.absolute()), "objects": len(rows),
            "views_per_object": 10000, "total_views": 60000, "check_only": check_only,
            "response_payloads_read": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--mesh-root", type=Path,
                        help="Supplied STL directory (default: SOURCE_ROOT/RIFT_dataset/meshes)")
    parser.add_argument("--refresh-geometry", action="store_true",
                        help="Update only generated mesh paths/transforms; refuse any other manifest change")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_root, args.output, check_only=args.check_only,
                             mesh_root=args.mesh_root, refresh_geometry=args.refresh_geometry), indent=2))


if __name__ == "__main__":
    main()
