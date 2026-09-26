#!/usr/bin/env python
"""Time the Sugavanam--Ertin Stage-1 inner operator on the active accelerator.

Reports milliseconds per *view* for the Eq.4 data objective (value and
value+gradient) at the production recipe's shapes, plus peak memory, and fails
on any XPU->CPU operator fallback. This is what sizes the PVC job time limits
and gives the per-step number Package C's acceptance asks for.

    python scripts_pvc/se_pvc_timing_probe.py --npz-path X --parent-role-manifest Y \\
        --config protocols/se_g40_readout48.json [--views 3]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rift.sugavanam_ertin_acquisition import CollectionAcquisition, data_objective, training_statistics  # noqa: E402
from rift_pvc import accelerator  # noqa: E402
from rift_pvc import sugavanam_ertin_paper_workflow as workflow  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz-path", required=True)
    p.add_argument("--parent-role-manifest", required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--views", type=int, default=3, help="training views to time")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    device = workflow.resolve_device(args.device)
    config = json.loads(args.config.read_text()) if args.config else {}
    acquisition = CollectionAcquisition(object_name=None, dataset_root=None,
        npz_path=args.npz_path, manifest=args.parent_role_manifest)
    recipe = workflow.make_recipe(acquisition.kind, config)
    partition, _, planning = workflow.plan(acquisition, recipe)
    recipe = planning["recipe"]

    points = workflow.grid_points(acquisition.extent, recipe["granularity"], device)
    statistics = training_statistics(acquisition)
    weights = torch.zeros(len(points), dtype=torch.complex128, device=device)
    # Non-zero field so the kernel does real work rather than multiplying zeros.
    weights = weights + torch.full_like(weights, 1e-6)

    report = {"backend": accelerator.backend(), "device": str(device),
              "grid_points": len(points), "granularity": recipe["granularity"],
              "azimuth_bins": recipe["azimuth_bins"], "groups": len(partition.directions),
              "train_views": len(acquisition.keys["train"]),
              "point_chunk": recipe["point_chunk"], "pair_chunk": recipe["pair_chunk"]}

    accelerator.reset_peak_memory_stats()
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        indices = list(range(min(args.views, len(acquisition.keys["train"]))))
        # Warm-up: first use of a kernel JIT-compiles on XPU.
        warm = time.perf_counter()
        data_objective(acquisition, points, weights, indices[:1], statistics, recipe, gradient=True)
        accelerator.synchronize()
        report["first_call_seconds_includes_jit"] = round(time.perf_counter() - warm, 3)

        for label, gradient in (("value", False), ("value_and_grad", True)):
            start = time.perf_counter()
            data_objective(acquisition, points, weights, indices, statistics, recipe, gradient=gradient)
            accelerator.synchronize()
            elapsed = time.perf_counter() - start
            report[f"{label}_ms_per_view"] = round(1000 * elapsed / len(indices), 2)

    fallbacks = sorted({str(r.message) for r in records if "fallback" in str(r.message).lower()})
    report["xpu_cpu_fallbacks"] = fallbacks
    report["peak_allocated_bytes"] = accelerator.max_memory_allocated()
    report["peak_reserved_bytes"] = accelerator.max_memory_reserved()

    # A full Stage-1 iteration touches every training view once per group pass;
    # each constrained_step evaluates the objective ~3x (step, candidate, grad).
    per_view = report.get("value_and_grad_ms_per_view")
    if per_view:
        report["estimated_seconds_per_stage1_iteration"] = round(
            3 * per_view * report["train_views"] / 1000.0, 1)
    print(json.dumps(report, indent=2), flush=True)
    if fallbacks:
        print("FAIL: XPU->CPU operator fallback detected", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
