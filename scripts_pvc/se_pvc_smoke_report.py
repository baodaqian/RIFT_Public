#!/usr/bin/env python
"""Summarize a Package C (Sugavanam--Ertin) PVC smoke.

Two modes:

  --gate FILE          print the initialization_audit from a --check-initialization
                       output, asserting it passed (or the degenerate reference fired)
  --run-root DIR       print the Stage-1 loss curve, per-step wall time taken
                       from the run logs, the checkpoint/resume evidence and a
                       fallback scan

Exit code is nonzero when an expectation fails, so the sbatch surfaces it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

AUDIT_KEYS = ("status", "mean_gradient_norm", "saturated_fraction",
              "nonzero_spatial_gradients", "positive_fraction", "mean_abs_sdf_m",
              "initialization", "initialization_std")


def _last_initialization_audit(text: str):
    """The last ``"initialization_audit": {...}`` value, by brace matching.

    The frontend prints its plan and then the method's own JSON, so a plain
    ``json.loads`` of the file will not do; this picks the final audit whichever
    document it came from.
    """
    marker = '"initialization_audit"'
    position = text.rfind(marker)
    if position < 0:
        return None
    start = text.find("{", position + len(marker))
    if start < 0:
        return None
    depth, in_string, escaped = 0, False, False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:index + 1])
                except json.JSONDecodeError:
                    return None
    return None


def report_gate(path: Path, label: str, expect_degenerate: bool) -> int:
    audit = _last_initialization_audit(path.read_text(errors="replace"))
    if audit is None:
        print(f"{label}: could not parse an initialization_audit from {path}")
        return 1
    print(f"{label}: " + json.dumps({k: audit[k] for k in AUDIT_KEYS if k in audit}, indent=2))
    expected = "initialization_degenerate" if expect_degenerate else "initialization_probe_passed"
    if audit.get("status") != expected:
        print(f"FAIL: expected {expected}, got {audit.get('status')!r}")
        return 1
    if expect_degenerate:
        for key, value in (("mean_gradient_norm", 0.0), ("saturated_fraction", 1.0),
                           ("nonzero_spatial_gradients", 0)):
            if audit.get(key) != value:
                print(f"FAIL: gate reference {key}={audit.get(key)!r}, expected {value!r}")
                return 1
        print("OK: the published-initialization gate still fires on this backend")
    return 0


def _stage1_history_summary(checkpoint: Path) -> dict:
    """The Stage-1 loss curve, taken from the checkpoint's own audit trail.

    With the production recipe ``validation_every`` is 10 full iterations, so a
    bounded smoke prints no validation line. ``stage1_history`` records one
    audited entry per sub-aperture step -- that is the curve.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover
        return {"stage1_history": "torch unavailable"}
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - reporting must not mask the run
        return {"stage1_history_error": f"{type(exc).__name__}: {exc}"}
    history = state.get("stage1_history") or []
    keys = ("iteration", "group", "data_loss", "target_data_loss",
            "residual_relative_error", "residual_feasible", "l1_norm",
            "subproblem_duality_gap", "subproblem_stationary", "converged",
            "backtracks", "root_updates")

    def row(entry):
        return {k: (float(entry[k]) if isinstance(entry.get(k), (int, float))
                    and not isinstance(entry.get(k), bool) else entry.get(k))
                for k in keys if k in entry}

    summary = {
        "stage1_steps_completed": len(history),
        "sdf_steps_completed": state.get("sdf_step", 0),
        "stage1_phase": state.get("phase"),
        "stage1_iteration": state.get("iteration"),
        "stage1_group_cursor": state.get("group_cursor"),
        "stage1_curve_first": [row(e) for e in history[:3]],
        "stage1_curve_last": [row(e) for e in history[-3:]],
        "stage1_views_exposed": int(sum(state.get("view_exposures", []) or [])),
        "stage1_solvers_converged": sum(1 for s in (state.get("sparse_solvers") or [])
                                        if isinstance(s, dict) and s.get("converged")),
        "stage1_solvers_total": len(state.get("sparse_solvers") or []),
    }
    losses = [float(e["data_loss"]) for e in history if isinstance(e.get("data_loss"), (int, float))]
    if losses:
        summary["stage1_data_loss_first"] = losses[0]
        summary["stage1_data_loss_last"] = losses[-1]
        summary["stage1_data_loss_min"] = min(losses)
    return summary


def _last_run_result(text: str):
    """Find the trainer's final result among banners and validation JSON."""
    decoder, result, offset = json.JSONDecoder(), None, 0
    while (start := text.find("{", offset)) >= 0:
        try:
            value, length = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            offset = start + 1
            continue
        offset = start + length
        if isinstance(value, dict) and "status" in value and "output" in value:
            result = value
    return result


def report_run(run_root: Path, logs: list[Path], elapsed_seconds: float | None = None,
               previous_report: Path | None = None) -> int:
    result: dict[str, object] = {"run_root": str(run_root.absolute())}
    errors = []
    status_path = run_root / "status.json"
    if status_path.is_file():
        result["status_json"] = json.loads(status_path.read_text())
    else:
        errors.append("Missing status.json")
    result["artifacts"] = sorted(p.name for p in run_root.glob("*")) if run_root.is_dir() else []
    if not logs:
        errors.append("At least one trainer log is required")

    # Stage-1 validation readouts are printed as one JSON line per validation pass.
    curve = []
    stage_lines = 0
    for log in logs:
        if not log.is_file():
            errors.append(f"Missing trainer log: {log}")
            continue
        log_text = log.read_text(errors="replace")
        outcome = _last_run_result(log_text)
        if (outcome is None or outcome.get("status") not in ("interrupted", "complete")
                or Path(outcome["output"]).absolute() != run_root.absolute()):
            errors.append(f"Missing or unsuccessful trainer result: {log}")
        for line in log_text.splitlines():
            line = line.strip()
            if line.startswith('{"stage": 1'):
                stage_lines += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                curve.append({k: row[k] for k in ("iteration", "global_complex_rel_mse",
                                                  "squared_error", "samples", "views") if k in row})
    result["stage1_validation_curve"] = curve
    result["stage1_validation_readouts"] = stage_lines

    checkpoint = run_root / "checkpoint_latest.pt"
    result["checkpoint_latest_exists"] = checkpoint.is_file()
    if checkpoint.is_file():
        result["checkpoint_latest_bytes"] = checkpoint.stat().st_size
        result.update(_stage1_history_summary(checkpoint))
    else:
        errors.append("Missing checkpoint_latest.pt")
    if result.get("stage1_phase") not in ("stage1", "stage2", "complete"):
        errors.append("Checkpoint has no valid smoke phase (or could not be loaded)")
    if not result.get("stage1_steps_completed", 0) or not result.get("stage1_views_exposed", 0):
        errors.append("Checkpoint shows no completed fitting work")
    if isinstance(result.get("status_json"), dict):
        sj = result["status_json"]
        result["progress"] = {k: sj.get(k) for k in
                              ("phase", "iteration", "group_cursor", "sdf_step", "train_views")}

    fallbacks = []
    for log in logs:
        if log.is_file():
            fallbacks += [l for l in log.read_text(errors="replace").splitlines()
                          if "fallback from xpu to cpu" in l.lower()]
    result["xpu_cpu_fallback_lines"] = fallbacks
    if fallbacks:
        errors.append("XPU to CPU operator fallback detected")
    if previous_report is not None:
        previous = json.loads(previous_report.read_text())
        if previous.get("acceptance_errors") != [] or previous.get("run_root") != result["run_root"]:
            errors.append("Previous report is not a passing report for this run")
        before = (previous.get("stage1_steps_completed", -1), previous.get("sdf_steps_completed", -1))
        after = (result.get("stage1_steps_completed", -1), result.get("sdf_steps_completed", -1))
        if any(a < b for a, b in zip(after, before)) or after <= before:
            errors.append("Resume did not advance the saved fitting progress")
        result["progress_before_resume"] = list(before)
    steps = result.get("stage1_steps_completed")
    if elapsed_seconds and isinstance(steps, int) and steps > 0:
        result["fitting_seconds_budget"] = float(elapsed_seconds)
        result["seconds_per_stage1_subaperture_step"] = round(float(elapsed_seconds) / steps, 2)
        groups = result.get("stage1_solvers_total") or 0
        if groups:
            result["estimated_seconds_per_full_stage1_iteration"] = round(
                float(elapsed_seconds) / steps * groups, 1)
    result["acceptance_errors"] = errors
    print(json.dumps(result, indent=2))
    return 1 if errors else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gate", type=Path)
    p.add_argument("--label", default="gate")
    p.add_argument("--expect-degenerate", action="store_true")
    p.add_argument("--run-root", type=Path)
    p.add_argument("--logs", type=Path, nargs="*", default=[])
    p.add_argument("--previous-report", type=Path,
                   help="Require fitting progress beyond the passing fresh-run report")
    p.add_argument("--elapsed-seconds", type=float, default=None,
                   help="wall seconds of actual fitting, for the per-step average")
    args = p.parse_args(argv)
    if args.gate:
        return report_gate(args.gate, args.label, args.expect_degenerate)
    if args.run_root:
        return report_run(args.run_root, list(args.logs), args.elapsed_seconds, args.previous_report)
    p.error("pass --gate or --run-root")


if __name__ == "__main__":
    raise SystemExit(main())
