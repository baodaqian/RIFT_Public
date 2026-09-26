#!/usr/bin/env python3
"""Fail closed on a Package H interruption/resume smoke and report its cost."""
import argparse
import json
import math
import re
from pathlib import Path

import torch


def checkpoint_summary(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    for group in ("sh_sas_state_dict", "gain_state_dict", "model_state_dict"):
        if not all(torch.isfinite(t).all().item() for t in state[group].values()):
            raise ValueError(f"nonfinite {group}")
    if state["optimizer_state_dict"] is None:
        raise ValueError("missing optimizer state")
    for row in state["optimizer_state_dict"]["state"].values():
        for value in row.values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError("nonfinite optimizer state")
    return {"step": state["step"], "accelerator_backend": state.get("accelerator_backend"),
            "sh_sas_backend": state.get("sh_sas_backend"),
            "xpu_rng_entries": len(state.get("xpu_rng_state") or []),
            "contract": state["sealed_npz_protocol_contract"],
            "args": {k: v for k, v in state["args"].items() if k != "resume"},
            "history": state["history"]}


def report(root):
    root = Path(root)
    stages = [json.loads((root / f"phase{i}.json").read_text()) for i in (1, 2)]
    checkpoints = [s["checkpoint"] for s in stages]
    logs = [(root / f"phase{i}.log").read_text() for i in (1, 2)]
    combined = "\n".join(logs)
    errors = []
    for number, (stage, log, ck) in enumerate(zip(stages, logs, checkpoints), 1):
        if stage["exit_code"] != 0:
            errors.append(f"phase {number} exit {stage['exit_code']}")
        interrupted = "Received signal 15" in log and "Stopped cleanly" in log
        completed = "SH-SAS training complete" in log and ck["step"] == 1000
        if not interrupted and not (number == 2 and completed):
            errors.append(f"phase {number} lacks a cooperative stop/completion")
        if ck["accelerator_backend"] != "xpu" or ck["xpu_rng_entries"] < 1:
            errors.append(f"phase {number} lacks XPU checkpoint/RNG evidence")
    if not 0 < checkpoints[0]["step"] < checkpoints[1]["step"] <= 1000:
        errors.append("checkpoint did not advance")
    for key in ("contract", "args", "sh_sas_backend"):
        if checkpoints[0][key] != checkpoints[1][key]:
            errors.append(f"resume changed {key}")
    if f"Resumed SH-SAS from step {checkpoints[0]['step']}" not in logs[1]:
        errors.append("resume was not acknowledged")
    if stages[1]["command"] != stages[0]["command"] + ["--resume", stages[0]["checkpoint_path"]]:
        errors.append("resume command is not the identical recipe plus --resume")
    args = checkpoints[1]["args"]
    expected = {"steps": 1000, "granularity": 48, "num_train": 2400, "num_val": 1000,
                "num_test": 1000, "seed": 42, "num_freq_wanted": 600, "phase_sign": -1.0,
                "compute_dtype": "fp64", "eval_every": 50, "eval_max_views": 0,
                "views_per_step": 1, "hash_levels": 16, "hash_log2_size": 19,
                "hash_final_resolution": 4096, "hidden_dim": 32, "lr": 1e-3,
                "opacity_scale": 0.1, "lambertian": True, "occlusion": True}
    if any(args.get(k) != v for k, v in expected.items()):
        errors.append("checkpoint differs from production SH-SAS recipe")
    contract = checkpoints[1]["contract"]
    if contract["response_access"]["reserved_test_materialized"] or contract["response_access"]["unused_materialized"]:
        errors.append("sealed response role was materialized")
    if contract.get("dataset_identity", {}).get("object_id") != "b787":
        errors.append("checkpoint is not bound to B787")
    acquisition = contract.get("antenna_selection", {})
    if acquisition.get("tx_indices") != [0] or acquisition.get("rx_indices") != [0]:
        errors.append("checkpoint is not bound to source Tx 0/Rx 0")
    if {k: len(v) for k, v in contract.get("role_ids", {}).items()} != {
            "train": 2400, "validation": 1000, "reserved_test": 1000, "unused": 5600}:
        errors.append("checkpoint has incorrect selected role counts")
    fallback_count = len(re.findall(r"(?:Aten Op fallback|fallback from XPU to CPU)", combined))
    if fallback_count:
        errors.append("XPU-to-CPU fallback")
    seconds = [float(s) for s in re.findall(r"\[([\d.]+)s/step\]", combined)]
    if not seconds or not all(math.isfinite(s) and s > 0 for s in seconds):
        errors.append("missing/nonfinite step timing")
    history = checkpoints[1]["history"]
    if not all(math.isfinite(float(v)) for row in history for v in row.values()):
        errors.append("nonfinite validation history")
    validation_seconds = []
    for stage in stages:
        started = None
        for event in stage["events"]:
            match = re.search(r"Step \[(\d+)/1000\]", event["line"])
            if match and int(match[1]) % 50 == 0:
                started = event["seconds"]
            if "Validation [step " in event["line"] and started is not None:
                validation_seconds.append(event["seconds"] - started)
                started = None
    mean_step = sum(seconds) / len(seconds) if seconds else None
    mean_validation = sum(validation_seconds) / len(validation_seconds) if validation_seconds else None
    return {"passed": not errors, "errors": errors, "steps": [c["step"] for c in checkpoints],
            "fallback_count": fallback_count, "mean_seconds_per_step": mean_step,
            "completed_validations": len(history), "validation_history": history,
            "seconds_per_full_validation": mean_validation,
            "seconds_per_validation_view": mean_validation / 1000 if mean_validation is not None else None,
            "estimated_1000_step_seconds": (1000 * mean_step + 20 * mean_validation
                                             if mean_step is not None and mean_validation is not None else None),
            "validation_sizing_available": mean_validation is not None,
            "qualification": "First PVC execution of this 2400-view recipe; no CUDA trajectory or convergence claim."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args(argv)
    try:
        result = report(args.root)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        result = {"passed": False, "errors": [str(exc)]}
    (args.root / "smoke_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
