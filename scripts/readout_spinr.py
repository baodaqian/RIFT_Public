#!/usr/bin/env python3
"""Export object-bound SpINR signed sigma, magnitude, and fixed-threshold support.

This is a full-trainer checkpoint readout. Historical engineering-smoke formats
keep their own evaluators. No radar responses are read or geometry used to tune
the threshold; no claim of occupancy probability or an SDF surface is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rift.rift_dataset import DEFAULT_ROOT, load_object, validate_checkpoint_object
from rift.spinr_fidelity import field_readout, recipe_name_from_identity
from rift.spinr_style import (SpinrStyleINR, build_spinr_style_acquisition_identity,
                              validate_spinr_style_acquisition_identity)
from train import load_tensor_checkpoint, _validate_saved_sealed_npz_protocol_contract
from train_spinr_style import (_recipe_identity, _validate_resume_checkpoint_structure,
                               _validate_normalization_against_sealed_contract)


def validate_readout_checkpoint(checkpoint, arrays, contract):
    """Gate object, scientific recipe, acquisition and scale before field access."""
    validate_checkpoint_object(checkpoint, contract)
    _validate_saved_sealed_npz_protocol_contract(checkpoint.get("sealed_npz_protocol_contract"), contract)
    saved = checkpoint.get("spinr_style_recipe", {})
    recipe = recipe_name_from_identity(saved)
    _validate_resume_checkpoint_structure(checkpoint, recipe_identity=_recipe_identity(recipe, len(contract["role_ids"]["train"]), contract.get("antenna_selection")))
    expected = build_spinr_style_acquisition_identity(arrays, contract)
    validate_spinr_style_acquisition_identity(checkpoint["acquisition_identity"], expected)
    _validate_normalization_against_sealed_contract(checkpoint["normalization"], contract)
    return recipe


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    from rift.antenna_selection import add_arguments
    add_arguments(parser)
    parser.add_argument("--num-train", type=int, default=3200, help="Checkpoint training subset size; use 2400 for the selected Delta run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=48)
    parser.add_argument("--fixed-threshold", type=float, required=True,
                        help="Threshold on |sigma|/max(|sigma|), selected without test geometry")
    parser.add_argument("--neural-point-tile", type=int, default=4096)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not math.isfinite(args.fixed_threshold) or not 0 < args.fixed_threshold <= 1:
        raise ValueError("--fixed-threshold must lie in (0,1]")
    if args.grid_size < 2 or args.neural_point_tile < 1:
        raise ValueError("grid size must exceed one and neural tile must be positive")
    if args.output.suffix != ".npz":
        raise ValueError("--output must end in .npz")
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace existing readout {args.output}")
    # Hash the exact bytes that were loaded, before issuing any long field queries.
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
        handle.seek(0)
        checkpoint = load_tensor_checkpoint(handle, map_location="cpu")
    arrays, contract = load_object(args.object, args.dataset_root, num_train=args.num_train,
        num_tx=args.num_tx, num_rx=args.num_rx, tx_indices=args.tx_indices, rx_indices=args.rx_indices)
    recipe = validate_readout_checkpoint(checkpoint, arrays, contract)
    model = SpinrStyleINR().to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    support = checkpoint["spinr_style_recipe"]["network"]["support_m"]
    scale = checkpoint["normalization"]["initial_output_scale"]
    points, sigma = field_readout(model, grid_size=args.grid_size, support_m=support,
                                 initial_output_scale=scale, neural_point_tile=args.neural_point_tile,
                                 device=args.device)
    sigma = sigma.cpu().numpy().reshape((args.grid_size,)*3)
    magnitude = np.abs(sigma)
    peak = float(magnitude.max())
    normalized = magnitude / peak if peak > 0 else np.zeros_like(magnitude)
    selected = normalized >= args.fixed_threshold
    points = points.cpu().numpy()
    provenance = {
        "schema": "rift_spinr_field_readout_v1", "checkpoint_sha256": digest.hexdigest(),
        "checkpoint": str(args.checkpoint.absolute()), "recipe": recipe,
        "dataset_identity": contract["dataset_identity"],
        "epoch_index": checkpoint["epoch_index"], "selection": checkpoint["selection"],
        "optimization_coverage": checkpoint.get("optimization_coverage"),
        "quantity": "abs(initial_output_scale*signed_real_field)",
        "normalization": "divide_by_max", "peak_magnitude": peak,
        "fixed_threshold": args.fixed_threshold, "grid_size": args.grid_size,
        "support_half_extent_m": support, "sampling": "voxel_centers_xyz_x_outer_z_inner",
        "training_quadrature": checkpoint["spinr_style_recipe"]["operator"],
        "threshold_selection": "user_supplied_no_geometry_optimization",
        "radar_responses_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        np.savez(handle, sigma=sigma, magnitude=magnitude, normalized_magnitude=normalized,
                 support_points_m=points[selected.reshape(-1)],
                 metadata_json=np.asarray(json.dumps(provenance, sort_keys=True)))
    print(json.dumps({"output": str(args.output), "support_points": int(selected.sum()),
                      "checkpoint_sha256": digest.hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
