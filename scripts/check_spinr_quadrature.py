#!/usr/bin/env python3
"""Check a frozen full SpINR checkpoint's quadrature on at most four train views.

--dry-run validates metadata/identity and prints the cost plan without response
reads. Actual full-grid evaluation is expensive and belongs on a compute node.
No optimization, checkpoint change, test read or scheduler action is performed.
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

from rift.npz_dataset import restrict_npz_response_views, iter_npz_response_views
from rift.rift_dataset import DEFAULT_ROOT, load_object
from rift.spinr_style import SpinrStyleINR, metadata_frequency_grid, npz_tx_rx_frequency_to_renderer
from rift.spinr_quadrature_audit import quadrature_plan, observe_rule, compare_observations
from scripts.readout_spinr import validate_readout_checkpoint
from train import load_tensor_checkpoint
from train_spinr_style import _disable_tf32


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--object", required=True)
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    from rift.antenna_selection import add_arguments
    add_arguments(p)
    p.add_argument("--num-train", type=int, default=3200, help="Checkpoint training subset size")
    p.add_argument("--output", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--views", type=int, default=4, help="First 1..4 IDs in the saved training role")
    p.add_argument("--reference-grid", type=int, help="Refined GL3 parent grid; default ceil(4/3 * saved parent grid)")
    p.add_argument("--relative-tolerance", type=float, default=.01)
    p.add_argument("--signal-absolute-tolerance", type=float, default=1e-12)
    p.add_argument("--gradient-absolute-tolerance", type=float, default=1e-10)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--neural-point-tile", type=int, default=4096)
    p.add_argument("--renderer-point-tile", type=int, default=65536)
    p.add_argument("--pair-tile", type=int, default=16)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not 1 <= args.views <= 4 or min(args.neural_point_tile, args.renderer_point_tile, args.pair_tile) < 1:
        raise ValueError("choose 1..4 training views and positive tile sizes")
    for value in (args.relative_tolerance, args.signal_absolute_tolerance, args.gradient_absolute_tolerance):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("diagnostic tolerances must be finite and positive")
    if not args.dry_run and args.output is None:
        raise ValueError("--output is required for an actual quadrature audit")
    if not args.dry_run and args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
        handle.seek(0)
        checkpoint = load_tensor_checkpoint(handle, map_location="cpu")
    arrays, contract = load_object(args.object, args.dataset_root, response_roles=("train",), num_train=args.num_train,
        num_tx=args.num_tx, num_rx=args.num_rx, tx_indices=args.tx_indices, rx_indices=args.rx_indices)
    recipe_name = validate_readout_checkpoint(checkpoint, arrays, contract)
    recipe = checkpoint["spinr_style_recipe"]
    rules = quadrature_plan(recipe, args.reference_grid)
    source_ids = contract["role_ids"]["train"][:args.views]
    report = {"schema": "rift_spinr_full_checkpoint_quadrature_v1",
              "checkpoint": str(args.checkpoint.absolute()), "checkpoint_sha256": digest.hexdigest(),
              "dataset_identity": contract["dataset_identity"], "recipe": recipe,
              "epoch_index": checkpoint["epoch_index"], "source_ids": source_ids, "role": "train",
              "frozen_parameters": True, "optimizer_steps": 0, "rules": rules,
              "execution": {"device": args.device, "neural_point_tile": args.neural_point_tile,
                  "renderer_point_tile": args.renderer_point_tile, "pair_tile": args.pair_tile,
                  "frequencies_per_view": 600, "pairs_per_view": contract["response_shape"][1] * contract["response_shape"][2]},
              "diagnostic_tolerances": {"relative_tolerance": args.relative_tolerance,
                  "signal_absolute_tolerance": args.signal_absolute_tolerance,
                  "gradient_absolute_tolerance": args.gradient_absolute_tolerance},
              "scope": "Finite-view numerical diagnostic, not a convergence proof or a training release",
              "dry_run": args.dry_run, "response_reads": 0}
    if args.dry_run:
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested without an available CUDA device")
    _disable_tf32()
    model = SpinrStyleINR().to(device=args.device, dtype=torch.float32)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    restricted = restrict_npz_response_views(arrays, source_ids)
    observations = []
    for source_id, raw in iter_npz_response_views(restricted, source_ids):
        raw = np.asarray(raw)
        if raw.shape != tuple(contract["response_shape"][1:]) or raw.dtype != np.complex64:
            raise ValueError("unexpected collection response layout")
        observations.append((int(source_id), npz_tx_rx_frequency_to_renderer(raw.mean(axis=2)),
                             arrays["rx_pos"][source_id], arrays["tx_pos"][source_id]))
    if [item[0] for item in observations] != source_ids:
        raise ValueError("diagnostic reader did not return exactly the declared ordered training views")
    report["response_reads"] = len(observations)
    results = []
    for rule in rules:
        print(f"Checking {rule['label']}: {rule['integration_points']:,} integration points", flush=True)
        results.append(observe_rule(
            model=model, observations=observations, frequencies_hz=metadata_frequency_grid(arrays["meta"]),
            rule=rule, support_m=recipe["network"]["support_m"],
            initial_output_scale=checkpoint["normalization"]["initial_output_scale"],
            training_mean_raw_power=checkpoint["normalization"]["training_mean_raw_power"],
            scene_bins=recipe_name != "legacy-midpoint", direct_bins=recipe_name in ("paper-v1-direct", "budget48-direct"), device=args.device,
            neural_point_tile=args.neural_point_tile, renderer_point_tile=args.renderer_point_tile,
            pair_tile=args.pair_tile))
    report["observations"] = [{key: value for key, value in result.items()
                               if key not in ("signals", "parameter_gradients")} for result in results]
    report["comparisons"] = [compare_observations(results[a], results[b], **report["diagnostic_tolerances"])
                             for a, b in ((0, 1), (1, 2), (0, 2))]
    report["checks_pass"] = all(comparison["checks_pass"] for comparison in report["comparisons"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "checks_pass": report["checks_pass"]}))
    return 0 if report["checks_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
