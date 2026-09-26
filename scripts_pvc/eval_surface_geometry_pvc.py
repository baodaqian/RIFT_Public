#!/usr/bin/env python3
"""Mesh metrics of a Sugavanam-Ertin Stage-2 surface on the RIFT dataset (PVC lane).

SE's geometry is the zero level set of its Stage-2 SDF, exported by the paper
workflow as ``surface.npz`` (marching cubes, vertices in metres in the scene frame).
It is scored exactly as GeRaF's zero-SDF surface (``scripts/eval_geraf_geometry.py``):
``--surface-samples`` area-weighted samples on the predicted surface and on the
registered object mesh (one numpy generator seeded 42, prediction first), then the
evaluator's ``chamfer`` and ``prf`` at ``--tolerance-m`` (default 5 mm, GeRaF's).
The SE checkpoint beside the surface binds the object: its acquisition identity must
equal the registered object's. No radar response is read.

    python scripts_pvc/eval_surface_geometry_pvc.py --object a320 --dataset-root D \
        --role-manifest RUN/role_manifest.json --surface SE_RUN/surface.npz \
        --checkpoint SE_RUN/checkpoint_latest.pt --output OUT/a320_se_surface.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.eval_b787_geometry_metrics import chamfer, prf, sample_surface_points  # noqa: E402
from scripts.render_b787_vs_stl import load_stl_vertices  # noqa: E402


def truth_mesh(object_name, dataset_root, role_manifest, stl=None):
    """The registered object mesh in the scene frame (as eval_geraf_geometry)."""
    from rift.rift_dataset import geometry_reference, load_object_contract, resolve_object_inputs, transform_mesh_vertices
    inputs = resolve_object_inputs(object_name=object_name, dataset_root=dataset_root, role_manifest_path=role_manifest)
    arrays, contract = load_object_contract(*inputs, response_roles=("validation",))
    reference = geometry_reference(object_name, dataset_root, mesh_path=stl)
    truth = transform_mesh_vertices(load_stl_vertices(reference["mesh_path"]), arrays["meta"],
                                    input_frame=reference["transform"]["input_frame"])
    return truth.reshape(-1, 3, 3), reference, contract, inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--object", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--role-manifest", type=Path, required=True)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="the SE checkpoint beside the surface (identity)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stl", type=Path)
    parser.add_argument("--surface-samples", type=int, default=20000)
    parser.add_argument("--tolerance-m", type=float, default=.005)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.surface_samples < 1 or not np.isfinite(args.tolerance_m) or args.tolerance_m <= 0:
        parser.error("surface samples and metric tolerance must be positive")
    truth, reference, contract, inputs = truth_mesh(args.object, args.dataset_root, args.role_manifest, args.stl)
    from rift.sugavanam_ertin_acquisition import CollectionAcquisition, digest
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    acquisition = CollectionAcquisition(npz_path=inputs[0], manifest=inputs[1])
    if digest(checkpoint["acquisition"]) != digest(acquisition.identity):
        raise ValueError("SE checkpoint belongs to another acquisition/object/split")
    with np.load(args.surface, allow_pickle=False) as saved:
        vertices, faces = saved["vertices"].astype(np.float64), saved["faces"].astype(np.int64)
        provenance = json.loads(str(saved["provenance_json"])) if "provenance_json" in saved.files else None
        extent = float(saved["extent_m"]) if "extent_m" in saved.files else None
    if not len(faces):
        raise ValueError("surface has no faces")
    rng = np.random.default_rng(42)
    prediction_points = sample_surface_points(vertices[faces], args.surface_samples, rng)
    truth_points = sample_surface_points(truth, args.surface_samples, rng)
    report = dict(schema="rift_se_surface_geometry_v1", object=args.object, surface=str(args.surface.resolve()),
                  checkpoint=str(args.checkpoint.resolve()), checkpoint_phase=checkpoint.get("phase"),
                  sdf_step=checkpoint.get("sdf_step"), surface_provenance=provenance, extent_m=extent,
                  dataset_identity=contract["dataset_identity"], geometry_reference=reference["transform"],
                  protocol="scripts/eval_geraf_geometry.py surface-to-mesh (samples, seed 42, chamfer, prf)",
                  surface_samples=args.surface_samples, tolerance_m=args.tolerance_m,
                  metrics={**chamfer(prediction_points, truth_points),
                           **prf(prediction_points, truth_points, args.tolerance_m)},
                  radar_response_reads=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
