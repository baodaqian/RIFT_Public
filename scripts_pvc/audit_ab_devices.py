#!/usr/bin/env python3
"""Read-only source audit of SpINR/GeRaF on synthetic CUDA or XPU fixtures.

Writes numerical evidence to the supplied scratch directory. Does not modify
models, recipes, datasets or source inventories. CUDA saves initial GeRaF
weights so the XPU comparison can reuse them across PyTorch versions.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from rift import geraf_source as original_geraf
from rift_pvc import accelerator
from rift_pvc import geraf_source as pvc_geraf
from tests.test_geraf_source import SMALL, acquisition
from rift_pvc.tests.test_geraf_xpu import on_device
from rift_pvc.tests.test_spinr_xpu import (
    test_direct_training_update_matches_cpu,
    test_quadrature_gradients_and_state_preservation,
)


def difference(actual, expected):
    a, b = actual.detach().cpu(), expected.detach().cpu()
    return {
        "relative_l2": float((a-b).norm() / b.norm().clamp_min(1e-30)),
        "max_absolute": float((a-b).abs().max()),
        "finite": bool(torch.isfinite(a).all()),
    }


def geraf_case(module, state, device):
    recipe = module.recipe_from_config(SMALL, .15)
    model = module.build_model(recipe, "cpu")
    # GeRaF restores its ADC-bank dictionaries by reference. Each case needs
    # private caches and chunk cursors, as well as identical network weights.
    model.load_state_dict(copy.deepcopy(state))
    model.to(device)
    op = on_device(acquisition(), device)
    cube = np.ones((SMALL["mf_grid"],)*3, np.float32)
    with module.fixed_numpy_seed(9):
        frame = module.sample_frame(op, recipe, "audit", cube, cube)
    model.radar_cfg = {"native": op}
    losses = model.loss(frame)
    sum(losses.values()).backward()
    gradients = torch.cat([
        p.grad.detach().flatten().cpu() for p in model.parameters()
        if p.grad is not None
    ])
    prediction = module.predict_native(model, frame, op).detach().cpu()
    if not torch.isfinite(gradients).all() or not torch.isfinite(prediction).all():
        raise AssertionError("Nonfinite GeRaF synthetic gradient or prediction")
    return {"losses": {k: v.detach().cpu() for k, v in losses.items()},
            "gradients": gradients, "prediction": prediction}


def compare_cases(actual, expected):
    return {"losses": {k: difference(actual["losses"][k], v)
                       for k, v in expected["losses"].items()},
            "gradients": difference(actual["gradients"], expected["gradients"]),
            "prediction": difference(actual["prediction"], expected["prediction"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "xpu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--require-cuda-parity", action="store_true",
                        help="Require CUDA reference and fail if loss, prediction or gradient parity exceeds the gate")
    args = parser.parse_args()
    os.environ["RIFT_ACCELERATOR"] = args.backend
    if not os.environ.get("SLURM_JOB_ID") or accelerator.device_count() != 1:
        raise RuntimeError("Requires one allocated accelerator")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"job_id": os.environ["SLURM_JOB_ID"], "device": accelerator.describe(),
              "repository": "https://github.com/baodaqian/RIFT.git",
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                cwd=ROOT, text=True).strip(),
              "fixture": "synthetic only; no dataset responses", "status": "running"}
    reference = None
    if args.reference and args.reference.is_file():
        reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    if args.require_cuda_parity:
        if args.backend != "xpu" or reference is None:
            raise RuntimeError("CUDA/XPU parity requires an XPU allocation and an existing reference")
        origin = json.loads((args.reference.parent/"report.json").read_text())
        if origin["device"]["backend"] != "cuda" or origin.get("geraf_cuda_wrapper_parity") != "passed":
            raise RuntimeError("Reference must come from a passing original/adapted CUDA comparison")
    if reference and reference.get("fixture_version") != 2:
        raise RuntimeError("Reference predates isolated ADC-bank state; regenerate it")
    torch.manual_seed(42)
    state = (reference["initial_state"] if reference else
             copy.deepcopy(original_geraf.build_model(
                 original_geraf.recipe_from_config(SMALL, .15)).state_dict()))
    device = torch.device(args.backend)
    initial_state = copy.deepcopy(state)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        # The existing quadrature gate deliberately expects a null CUDA memory
        # field and is therefore only an XPU/CPU gate. Direct-bin parity works
        # on either accelerator without modifying its tolerance.
        test_direct_training_update_matches_cpu(device)
        report["spinr_direct_update_cpu_parity"] = "passed"
        if args.backend == "xpu":
            test_quadrature_gradients_and_state_preservation(device)
            report["spinr_quadrature_cpu_parity"] = "passed"
        original = geraf_case(original_geraf, state, device)
        adapted = geraf_case(pvc_geraf, state, device)
    from tests.test_geraf_source import compare_nested
    compare_nested(state, initial_state)
    report["fixture_version"] = 2
    report["initial_state_preserved"] = True
    report["warnings"] = sorted(set(str(w.message) for w in captured))
    if any("fallback from XPU to CPU" in w for w in report["warnings"]):
        raise AssertionError("XPU operator fallback detected")
    report["geraf_original_vs_adapted_same_device"] = compare_cases(adapted, original)
    if args.backend == "cuda":
        try:
            for name in ("gradients", "prediction"):
                torch.testing.assert_close(adapted[name], original[name], rtol=1e-6, atol=1e-9)
            for name in original["losses"]:
                torch.testing.assert_close(adapted["losses"][name], original["losses"][name],
                                           rtol=1e-6, atol=1e-9)
        except AssertionError as error:
            report["geraf_cuda_wrapper_parity"] = "failed"
            report["comparison_error"] = str(error)
        else:
            report["geraf_cuda_wrapper_parity"] = "passed"
    if reference:
        report["geraf_xpu_vs_saved_h100"] = compare_cases(adapted, reference["adapted"])
    if args.require_cuda_parity:
        comparison = report["geraf_xpu_vs_saved_h100"]
        metrics = [*comparison["losses"].values(), comparison["gradients"], comparison["prediction"]]
        # Declare the gate before CUDA evidence arrives; never select a
        # tolerance after seeing the result. The loss gate retains Package B's
        # 1% bound and now also covers native predictions and network gradients.
        report["cuda_xpu_parity_gate"] = dict(relative_l2=1e-2, absolute=1e-6,
            passed=all(m["finite"] and (m["relative_l2"] <= 1e-2 or m["max_absolute"] <= 1e-6)
                       for m in metrics))
    torch.save({"fixture_version": 2, "initial_state": state, "original": original, "adapted": adapted},
               args.output/"numerics.pt")
    passed = (report.get("cuda_xpu_parity_gate", {}).get("passed", True)
              and report.get("geraf_cuda_wrapper_parity") != "failed")
    report["status"] = "complete" if passed else "failed"
    (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] == "failed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
