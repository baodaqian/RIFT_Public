#!/usr/bin/env python3
"""Native-power and Gaussian-occupancy readout from one identified checkpoint.

Supports upstream, legacy and audit_v1 recipes without reinterpreting defaults.
Only train/validation roles are exposed. No reserved-test readout is implied.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
import train_radarsplat as trainer
from rift.radarsplat_b7873200 import RadarSplatEffects
from rift.radarsplat_b7873200_adapter import (create_optimizers, export_gaussian_occupancy_geometry,
                                           load_native_view, model_from_checkpoint_state)
from rift.radarsplat_b7873200_protocol import RECIPE_FILENAME, load_cache


def readout(*, checkpoint_path, cache_root, device="cpu", role="validation", object_name=None,
            gaussian_chunk_size=64, max_raster_candidate_pairs=2_000_000, geometry_path=None):
    if role not in {"train", "validation"}:
        raise ValueError("readout permits only train or validation")
    device = torch.device(device)
    checkpoint = trainer._load_checkpoint(Path(checkpoint_path), device)
    from rift.radarsplat_release import SCHEMA
    if checkpoint.get("schema") == SCHEMA:
        from rift.radarsplat_release_training import readout as released_readout
        return released_readout(checkpoint, checkpoint_path=checkpoint_path, cache_root=cache_root,
                                device=device, role=role, object_name=object_name, geometry_path=geometry_path)
    identity = checkpoint.get("run_identity", {})
    methods = {"radarsplat_b7873200_native_power_v1", "radarsplat_native_power_audit_v1"}
    if checkpoint.get("checkpoint_version") != trainer.CHECKPOINT_VERSION or identity.get("method") not in methods:
        raise ValueError("unsupported RadarSplat checkpoint recipe")
    # Reject wrong source/object/cache before loading any target pixels.
    with (Path(cache_root) / RECIPE_FILENAME).open() as handle:
        recipe = json.load(handle)
    if recipe != identity.get("target_recipe") or recipe.get("sealed_protocol_identity") != identity.get("sealed_protocol_identity"):
        raise ValueError("checkpoint and target cache identities differ")
    if object_name is not None:
        from rift.rift_dataset import object_identity, validate_checkpoint_object
        validate_checkpoint_object(checkpoint, object_identity(object_name))
    cache = load_cache(cache_root)
    if cache.train_peak_power != identity.get("train_peak_power"):
        raise ValueError("checkpoint and cache normalization differ")
    effects = RadarSplatEffects(**identity["renderer"]["effects"])
    grid = load_native_view(cache, cache.train_indices[0], "train", "cpu").grid
    if trainer._renderer_identity(grid, effects) != identity["renderer"]:
        raise ValueError("checkpoint renderer and calibrated target grid differ")
    model = model_from_checkpoint_state(checkpoint["model_state_dict"], device=device)
    optimization = identity["optimization"]
    optimizers = create_optimizers(model, optimization["learning_rates"],
                                  betas=tuple(optimization["betas"]), eps=optimization["eps"])
    sampler = trainer.DeterministicViewSampler(cache.train_indices, optimization["seed"])
    trainer._restore_checkpoint(checkpoint, identity, model, optimizers, sampler, cache.acquisition_record)
    step = int(checkpoint["step"])
    active_degree = min((step-1)//identity["model"]["sh_degree_interval"], model.sh_degree)
    if step < 1:
        raise ValueError("readout requires a checkpoint with completed updates")
    args = SimpleNamespace(gaussian_chunk_size=gaussian_chunk_size,
        max_raster_candidate_pairs=max_raster_candidate_pairs,
        occupancy_threshold=identity["objective"]["occupancy_threshold"])
    with torch.no_grad():
        metrics = trainer.evaluate_native_power(model, cache, effects, args, device, active_degree, role=role)
    if geometry_path is not None:
        if Path(geometry_path).exists():
            raise FileExistsError(geometry_path)
        export_gaussian_occupancy_geometry(geometry_path, model, active_sh_degree=active_degree,
                                          train_peak_power=cache.train_peak_power)
    return {"schema": "rift_radarsplat_checkpoint_readout_v1", "checkpoint": str(checkpoint_path),
            "step": step, "role": role, "run_identity": identity, "metrics": metrics,
            "geometry": None if geometry_path is None else str(geometry_path),
            "observable": "native real power; Gaussian occupancy geometry from the same checkpoint"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--object")
    parser.add_argument("--role", choices=("train", "validation"), default="validation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--gaussian-chunk-size", type=int, default=64)
    parser.add_argument("--max-raster-candidate-pairs", type=int, default=2_000_000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = readout(checkpoint_path=args.checkpoint, cache_root=args.cache_root,
        device=args.device, role=args.role, object_name=args.object,
        gaussian_chunk_size=args.gaussian_chunk_size, max_raster_candidate_pairs=args.max_raster_candidate_pairs,
        geometry_path=args.geometry)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
