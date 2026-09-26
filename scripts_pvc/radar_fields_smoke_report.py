#!/usr/bin/env python3
"""Validate Radar Fields smoke artifacts without reading radar responses.

Cooperative interruption requires both trainer evidence and a valid checkpoint.
A timeout, a fallback, or a resume without committed progress is not success.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rift_pvc.radar_fields_training import normalize_device_rng_state, validate_shim_identity

FALLBACK = "fallback from XPU to CPU"


def finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def snapshot(path, dataset):
    path = Path(path).resolve()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if dataset == "rift":
        if ck.get("artifact_schema") != "radar_fields_checkpoint_v2":
            raise ValueError("Not a collection Radar Fields checkpoint")
        controls, recipe = ck["args"], ck["radar_fields_recipe"]
        model, optimizer, scheduler = (ck[k] for k in ("radar_fields_state_dict", "optimizer_state_dict", "scheduler_state_dict"))
        rng, cpu_rng = ck.get("xpu_rng_state"), ck.get("torch_rng_state")
        if not ck.get("numpy_rng_state_json"):
            raise ValueError("Missing NumPy RNG state")
        identity = {k: ck[k] for k in ("dataset_provenance", "split_provenance", "sealed_protocol_contract")}
        complete = path.name == "checkpoint_final.pth.tar" and ck["step"] == controls["steps"]
        validated_final = any(row.get("step") == ck["step"] for row in ck["history"])
    else:
        if ck.get("schema") != "rift_gotcha_checkpoint_v1" or ck["recipe"].get("method") != "radar_fields":
            raise ValueError("Not a native GOTCHA Radar Fields checkpoint")
        recipe, controls = ck["recipe"], ck["recipe"]["controls"]
        model, optimizer, scheduler = (ck[k] for k in ("model_state_dict", "optimizer", "scheduler"))
        rng, cpu_rng = ck.get("rng_xpu"), ck.get("rng_torch")
        if not ck.get("rng_numpy") or not ck.get("rng_python"):
            raise ValueError("Missing NumPy/Python RNG state")
        identity = {k: ck[k] for k in ("dataset_contract", "dataset_identity")}
        complete = ck.get("complete") is True
        validated_final = ck.get("pending_validation") is None and any(row.get("step") == ck["step"] for row in ck["history"])
    step, budget = ck["step"], controls["steps"]
    if type(step) is not int or not 0 < step <= budget:
        raise ValueError("Checkpoint has no valid committed training progress")
    if ck.get("accelerator_backend") != "xpu" or controls.get("model_backend") != "upstream-tcnn-torchshim":
        raise ValueError("Smoke did not execute the PVC Radar Fields backend")
    validate_shim_identity(ck, SimpleNamespace(**controls))
    normalize_device_rng_state(rng, expected_device_count=1, require_present=True)
    if not torch.is_tensor(cpu_rng) or cpu_rng.dtype != torch.uint8 or cpu_rng.ndim != 1:
        raise ValueError("Missing or invalid Torch RNG state")
    if not model or not optimizer or not optimizer.get("state") or not scheduler:
        raise ValueError("Missing model/optimizer/scheduler state")
    if not all(finite(v) for v in (model, optimizer, scheduler)):
        raise ValueError("Nonfinite training state")
    if scheduler.get("last_epoch") != step:
        raise ValueError("Scheduler clock disagrees with checkpoint step")
    if any("step" not in state or not 0 < float(state["step"]) <= step for state in optimizer["state"].values()):
        raise ValueError("Invalid optimizer clock")
    if complete and (step != budget or not validated_final):
        raise ValueError("Completion lacks final validation")
    return dict(checkpoint=str(path), output_dir=str(path.parent), dataset=dataset, step=step, budget=budget,
                complete=complete, recipe=recipe, identity=identity, tcnn_shim=ck["tcnn_shim"])


def latest_snapshot(root, dataset):
    names = (("checkpoint_latest.pth.tar", "checkpoint_final.pth.tar") if dataset == "rift"
             else ("checkpoint_latest.pt", "checkpoint_final.pt"))
    candidates = [path for name in names for path in Path(root).rglob(name)]
    if not candidates or len({p.parent for p in candidates}) != 1:
        raise ValueError("Smoke root must contain exactly one checkpoint directory")
    states = [snapshot(p, dataset) for p in candidates]
    return max(states, key=lambda row: (row["step"], row["complete"]))


def final_gotcha_result(text):
    decoder, result, cursor = json.JSONDecoder(), None, 0
    while (start := text.find("{", cursor)) >= 0:
        try:
            value, count = decoder.raw_decode(text[start:])
        except ValueError:
            cursor = start + 1
            continue
        cursor = start + count
        if isinstance(value, dict) and value.get("method") == "radar_fields" and isinstance(value.get("result"), dict):
            result = value["result"]
    return result


def report(root, dataset, log, exit_code, previous=None):
    text = Path(log).read_text(errors="replace")
    if not text.strip() or FALLBACK in text:
        raise ValueError("Missing trainer evidence or XPU-to-CPU fallback")
    current = latest_snapshot(root, dataset)
    if previous is not None:
        for key in ("output_dir", "dataset", "budget", "recipe", "identity"):
            if current[key] != previous[key]:
                raise ValueError(f"Resume changed {key}")
        if current["step"] <= previous["step"]:
            raise ValueError("Resume did not advance committed checkpoint progress")
    if dataset == "rift":
        if exit_code != 0:
            raise ValueError(f"Collection trainer failed or timed out: exit {exit_code}")
        steps = [int(x) for x in re.findall(r"Step \[(\d+)/\d+\]", text)]
        if not steps or max(steps) != current["step"]:
            raise ValueError("Checkpoint progress does not match this trainer execution")
        if previous and f"Resumed Radar Fields from step {previous['step']}" not in text:
            raise ValueError("Missing explicit resume evidence")
        if not current["complete"] and not ("Received signal" in text and "Stopped cleanly after publishing checkpoint_latest." in text):
            raise ValueError("No cooperative interruption or completed recipe")
    else:
        result = final_gotcha_result(text)
        expected = "complete" if current["complete"] else "interrupted"
        expected_exit = 0 if current["complete"] else 143
        if exit_code != expected_exit or not result or result.get("status") != expected or result.get("step") != current["step"]:
            raise ValueError("Native trainer outcome disagrees with checkpoint or exit status")
        if result.get("test_accessed") is not False:
            raise ValueError("Native trainer did not confirm sealed test responses")
    return dict(current, status="passed", recovery_exercised=previous is not None,
                trainer_exit=exit_code, fallback_warnings=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("rift", "gotcha"), required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--snapshot", type=Path, help="Save the selected resume checkpoint's pre-run progress")
    parser.add_argument("--log", type=Path)
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.snapshot:
            result = snapshot(args.snapshot, args.dataset)
        else:
            previous = json.loads(args.previous.read_text()) if args.previous else None
            result = report(args.root, args.dataset, args.log, args.exit_code, previous)
        code = 0
    except Exception as exc:
        result, code = dict(status="failed", error=f"{type(exc).__name__}: {exc}"), 1
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
