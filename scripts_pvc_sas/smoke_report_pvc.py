#!/usr/bin/env python
"""Acceptance report for a bounded ``train_sas_pvc.py`` smoke (Package G).

Reads the run directory, the job log and the two phase outcomes recorded by
the launcher, and enforces the acceptance conditions of
RIFT_SAS_PVC_Adaptation.md section 6.5: a cooperative interruption
(``status.json`` ``reason: signal``, exit 0) with a loadable
``checkpoint_latest.pt`` carrying the backend identity, a resumed phase that
advances the step under the same recipe, no XPU->CPU fallback line, no
non-finite loss, and finite validation rows when an evaluation was reached.
Writes a JSON report; exit 1 on any failed condition. A passing report never
means the production recipe completed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import torch

STEP_RE = re.compile(r"^step (\d+)/(\d+) loss=([-+0-9.eE]+|nan|inf) rel_mse=([-+0-9.eE]+|nan|inf)")


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def finite_tensors(state):
    for key in ("model_state_dict", "calibration_state_dict"):
        for name, value in state[key].items():
            if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
                if not torch.isfinite(value).all():
                    return f"{key}.{name} is non-finite"
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--phase1-exit", type=int, required=True)
    parser.add_argument("--phase2-exit", type=int, required=True)
    parser.add_argument("--phase1-wall", type=float, required=True)
    parser.add_argument("--phase2-wall", type=float, required=True)
    parser.add_argument("--status-phase1", required=True, help="copy of status.json taken after phase 1")
    parser.add_argument("--expected-steps", type=int, default=26000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    run = Path(args.run_dir)
    failures = []
    report = {"run_dir": str(run), "model": args.model, "phase1_exit": args.phase1_exit, "phase2_exit": args.phase2_exit,
              "phase1_wall_s": args.phase1_wall, "phase2_wall_s": args.phase2_wall}

    # phase 1: cooperative interruption
    if args.phase1_exit != 0:
        failures.append(f"phase 1 exit {args.phase1_exit} (a cooperative stop exits 0)")
    try:
        status1 = json.loads(Path(args.status_phase1).read_text())
    except Exception as exc:  # noqa: BLE001
        status1 = None
        failures.append(f"phase 1 status.json unreadable: {exc}")
    if status1 is not None:
        report["status_phase1"] = status1
        if status1.get("model") != args.model:
            failures.append("phase 1 status model mismatch")
        if status1.get("done") is True:
            failures.append("phase 1 completed the whole recipe; a bounded smoke must be interrupted")
        elif status1.get("reason") != "signal" or int(status1.get("step", 0)) < 1:
            failures.append(f"phase 1 did not stop cooperatively: {status1}")
    latest = run / "checkpoint_latest.pt"
    step1 = None
    if not latest.exists():
        failures.append("checkpoint_latest.pt missing after phase 1")
    else:
        try:
            state = load(latest)
            step1 = int(status1["step"]) if status1 else int(state["step"])
            report["checkpoint"] = {"step": int(state["step"]), "model_kind": state.get("model_kind"),
                                    "accelerator_backend": state.get("accelerator_backend"),
                                    "xpu_rng_state_entries": len(state.get("xpu_rng_state") or []),
                                    "cuda_rng_state": state.get("cuda_rng_state"),
                                    "calibration_mode": state.get("calibration_mode"),
                                    "active_points": state.get("parameter_counts", {}).get("active_points"),
                                    "saved_steps": state["args"].get("steps")}
            if state.get("model_kind") != args.model:
                failures.append("checkpoint model_kind mismatch")
            if state.get("accelerator_backend") != "xpu":
                failures.append(f"checkpoint backend {state.get('accelerator_backend')!r}, expected 'xpu'")
            if int(state["args"].get("steps", -1)) != args.expected_steps:
                failures.append(f"saved recipe steps {state['args'].get('steps')} != {args.expected_steps}")
            problem = finite_tensors(state)
            if problem:
                failures.append(problem)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"checkpoint_latest.pt unreadable: {exc}")

    # phase 2: resumed and advanced
    if args.phase2_exit != 0:
        failures.append(f"phase 2 exit {args.phase2_exit}")
    status2 = None
    if (run / "status.json").exists():
        status2 = json.loads((run / "status.json").read_text())
        report["status_phase2"] = status2
        step2 = int(status2.get("step", 0))
        if step1 is not None and not (step2 > step1 or status2.get("done") is True):
            failures.append(f"phase 2 did not advance: {step1} -> {step2}")
        if status2.get("done") is not True and status2.get("reason") != "signal":
            failures.append(f"phase 2 ended without a cooperative stop: {status2}")
    else:
        failures.append("status.json missing after phase 2")

    # log: resume line, step lines, fallbacks, non-finite
    text = Path(args.log).read_text(errors="replace")
    resumed = re.findall(rf"Resumed {re.escape(args.model)} at step (\d+)", text)
    report["resumed_at"] = [int(v) for v in resumed]
    if step1 is not None and (not resumed or int(resumed[-1]) != step1):
        failures.append(f"phase 2 did not report resuming at step {step1}: {resumed}")
    fallbacks = len(re.findall(r"fallback from XPU to CPU", text))
    report["fallback_lines"] = fallbacks
    if fallbacks:
        failures.append(f"{fallbacks} XPU->CPU fallback line(s)")
    steps, nonfinite = [], 0
    for line in text.splitlines():
        m = STEP_RE.match(line.strip())
        if m:
            steps.append(int(m.group(1)))
            if not math.isfinite(float(m.group(3))):
                nonfinite += 1
    report["logged_steps"] = len(steps)
    report["last_logged_step"] = max(steps) if steps else None
    if nonfinite or "non-finite loss" in text:
        failures.append("non-finite loss in the log")
    if steps and step1:
        report["s_per_step_phase1_incl_startup"] = args.phase1_wall / step1
    if status2 and step1 is not None and int(status2.get("step", 0)) > step1:
        report["s_per_step_phase2_incl_startup"] = args.phase2_wall / (int(status2["step"]) - step1)
    refinement = [l for l in text.splitlines() if l.startswith("split ") or "refinement" in l.lower()]
    report["refinement_lines"] = refinement[-5:]
    gains = re.findall(r"warm-started g = ([-+0-9.eEj()]+)", text)
    report["gain_warm_start"] = gains[:2]

    # history rows
    history = run / "history.csv"
    if history.exists():
        with history.open() as handle:
            rows = list(csv.DictReader(handle))
        report["history_rows"] = rows[-3:]
        if any(not math.isfinite(float(r["val_rel_mse"])) for r in rows):
            failures.append("non-finite validation metric in history.csv")
    else:
        report["history_rows"] = []

    report["failures"] = failures
    report["pass"] = not failures
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("history_rows",)}, indent=2, sort_keys=True))
    print("SMOKE_REPORT", "PASS" if not failures else "FAIL", failures, flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
