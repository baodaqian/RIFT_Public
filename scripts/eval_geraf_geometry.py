#!/usr/bin/env python3
"""Native zero-SDF geometry of a validation-selected, object-bound GeRaF model.

Reads checkpoint and acquisition metadata only; no radar response is accessed.
Uses the same identified checkpoint as the native signal readout. The frozen
historical support-density postflight remains a distinct readout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import train_geraf as trainer
from rift.geraf import GeRaFModel
from rift.geraf_v1 import extract_zero_surface
from rift.rift_dataset import (DEFAULT_ROOT, geometry_reference, load_object, object_paths,
                               transform_mesh_vertices, validate_checkpoint_object)
from rift.geraf_b7873200_source import load_b7873200_metadata_source
from rift.geraf_b7873200_acquisition import build_b7873200_acquisition_record
from scripts.eval_geraf_complex_response import _validate_checkpoint_selection
from scripts.eval_b787_geometry_metrics import sample_surface_points, chamfer, prf
from scripts.render_b787_vs_stl import load_stl_vertices


def checkpoint_model(checkpoint, source, contract, device):
    """Reuse the native evaluator's selection, model, recipe and pose gates."""
    if checkpoint.get("schema") == "rift_geraf_source_v1_checkpoint":
        from rift.geraf_source_data import RIFTSourceData
        from rift.geraf_source_training import load_selected_models
        data = RIFTSourceData(source.arrays.path, contract["role_manifest_path"])
        validate_checkpoint_object(checkpoint, contract)
        models, row = load_selected_models(checkpoint, data, device)
        class MetricSDF(torch.nn.Module):
            def __init__(self, network, extent):
                super().__init__()
                self.network, self.extent = network, extent
            def forward(self, points):
                return self.network(points / self.extent)[..., 0] * self.extent
        return SimpleNamespace(sdf_network=MetricSDF(models['scalar'].sdf_network, data.extent),
                               extent=data.extent, implementation='source_v1'), row
    validate_checkpoint_object(checkpoint, contract)
    args = argparse.Namespace(**checkpoint["cli_args"])
    expected_config = trainer.model_config_from_args(args)
    if checkpoint.get("model_config") != expected_config:
        raise ValueError("GeRaF model configuration disagrees with its declared recipe")
    if not np.isclose(args.scene_extent, .15, rtol=0, atol=1e-12):
        raise ValueError("GeRaF geometry extent disagrees with the collection")
    stats = checkpoint["target_stats"]
    cache = SimpleNamespace(sealed_identity=source.identity, recipe=checkpoint["cache_recipe"],
                            acquisition_record=build_b7873200_acquisition_record(source.arrays),
                            stats=stats, target_manifest=checkpoint["target_manifest"],
                            effective_pairs_per_plane=256,
                            geraf_mf_magnitude_peak=stats["geraf_mf_magnitude_peak"])
    row = _validate_checkpoint_selection(checkpoint, cache, args)
    model = GeRaFModel(**expected_config).to(device=device, dtype=torch.float32)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--role-manifest", type=Path, help="Exact RIFT run manifest for subset-trained checkpoints")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stl", type=Path)
    parser.add_argument("--grid", type=int, default=48)
    parser.add_argument("--chunk", type=int, default=16384)
    parser.add_argument("--surface-samples", type=int, default=20000)
    parser.add_argument("--tolerance-m", type=float, default=.005)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.surface_samples < 1 or not np.isfinite(args.tolerance_m) or args.tolerance_m <= 0:
        parser.error("surface samples and metric tolerance must be positive")
    from rift.rift_dataset import resolve_object_inputs, load_object_contract
    inputs = resolve_object_inputs(object_name=args.object, dataset_root=args.dataset_root,
                                   role_manifest_path=args.role_manifest)
    arrays, contract = load_object_contract(*inputs, response_roles=("validation",))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # Source-v1 already has its public selected-acquisition contract. Keep the
    # frozen compatibility metadata loader exclusive to historical recipes.
    source = (SimpleNamespace(arrays=SimpleNamespace(path=str(inputs[0])))
              if checkpoint.get('schema') == 'rift_geraf_source_v1_checkpoint'
              else load_b7873200_metadata_source(*inputs))
    model, selected = checkpoint_model(checkpoint, source, contract, torch.device(args.device))
    vertices, faces, report = extract_zero_surface(model.sdf_network, model.extent,
                                                   grid=args.grid, chunk=args.chunk)
    reference = geometry_reference(args.object, args.dataset_root, mesh_path=args.stl)
    original = load_stl_vertices(reference["mesh_path"])
    truth = transform_mesh_vertices(original, arrays["meta"],
                                    input_frame=reference["transform"]["input_frame"])
    rng = np.random.default_rng(42)
    prediction_points = sample_surface_points(vertices[faces], args.surface_samples, rng)
    truth_points = sample_surface_points(truth.reshape(-1, 3, 3), args.surface_samples, rng)
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    report.update(schema="rift_geraf_native_geometry_v1", checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_sha256=digest.hexdigest(), checkpoint_step=checkpoint["step"],
                  dataset_identity=contract["dataset_identity"], selected_native_mf_metrics=dict(selected),
                  implementation=model.implementation, tolerance_m=args.tolerance_m,
                  surface_samples=args.surface_samples, geometry_reference=reference["transform"],
                  metrics={**chamfer(prediction_points, truth_points),
                           **prf(prediction_points, truth_points, args.tolerance_m)},
                  radar_response_reads=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "surface.npz").open("xb") as handle:
        np.savez(handle, vertices_m=vertices, faces=faces)
    with (args.output_dir / "metrics.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
