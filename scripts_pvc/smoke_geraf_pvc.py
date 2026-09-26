#!/usr/bin/env python3
"""Bound production GeRaF by committed updates, without changing its recipe."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

FALLBACK = "fallback from XPU to CPU"


def records(text):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"(?m)^\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        if isinstance(value, dict):
            yield value


def run_segment(command, log_path, *, stop_step=None, deadline=600, grace=90):
    """Require a normal result or our own cooperative update-bound interruption.

    A watchdog timeout and an operator fallback always fail, even if the child
    subsequently writes a valid recovery checkpoint and exits cleanly.
    """
    started, requested, reason = time.monotonic(), None, None
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=True,
                               env={**os.environ, "PYTHONUNBUFFERED": "1"})
    watcher = selectors.DefaultSelector()
    watcher.register(process.stdout, selectors.EVENT_READ)
    pending = b""
    try:
        with Path(log_path).open("wb") as log:
            while watcher.get_map():
                now = time.monotonic()
                if now-started > deadline and reason != "watchdog":
                    reason = "watchdog"
                    if requested is None:
                        requested = now
                        os.killpg(process.pid, signal.SIGTERM)
                if requested is not None and now-requested > grace:
                    raise TimeoutError("GeRaF did not stop within the checkpoint grace period")
                for key, _ in watcher.select(timeout=.2):
                    block = os.read(key.fileobj.fileno(), 65536)
                    if not block:
                        watcher.unregister(key.fileobj)
                        continue
                    log.write(block)
                    log.flush()
                    pending += block
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        line = line.decode(errors="replace")
                        if FALLBACK in line:
                            reason = "fallback"
                        if requested is None:
                            try:
                                progress = json.loads(line)
                            except ValueError:
                                progress = {}
                            reached = (stop_step is not None and isinstance(progress, dict)
                                       and isinstance(progress.get("step"), int)
                                       and "losses" in progress and progress["step"] >= stop_step)
                            if reached or reason == "fallback":
                                reason = reason or "update_bound"
                                requested = time.monotonic()
                                try:
                                    os.killpg(process.pid, signal.SIGTERM)
                                except ProcessLookupError:
                                    pass
            code = process.wait(timeout=grace)
    finally:
        watcher.close()
        process.stdout.close()
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    text = Path(log_path).read_text(errors="replace")
    if FALLBACK in text:
        raise RuntimeError(f"XPU operator fallback in {log_path}")
    if reason == "watchdog":
        raise TimeoutError(f"GeRaF exceeded its watchdog; see {log_path}")
    expected = 0 if stop_step is None else 143
    if code != expected or (stop_step is not None and reason != "update_bound"):
        raise RuntimeError(f"GeRaF exit {code}, stop reason {reason}; expected {expected}: {log_path}")
    results = [row for row in records(text) if "status" in row]
    status = "prepared" if stop_step is None else "interrupted"
    if not results or results[-1].get("status") != status:
        raise RuntimeError(f"GeRaF did not report {status}: {log_path}")
    return dict(wall_seconds=time.monotonic()-started, returncode=code,
                stop_reason=reason, result=results[-1], fallback_warning=False)


def load_partial(path, *, backend="xpu"):
    if not Path(path).is_file():
        raise RuntimeError(f"Required partial checkpoint is missing: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    step = saved.get("step")
    if (saved.get("schema") != "rift_geraf_source_v1_checkpoint" or type(step) is not int
            or not 0 < step < saved.get("recipe", {}).get("steps", 0)
            or saved.get("complete") is not False):
        raise RuntimeError("Smoke requires an incomplete checkpoint with committed updates")
    if saved["recipe"]["steps"] != 50000:
        raise RuntimeError("Smoke must preserve the production 50000-step cosine clock")
    if backend != "cpu" and not saved.get(f"rng_{backend}"):
        raise RuntimeError("Partial checkpoint lacks accelerator RNG state")
    exposures = saved.get("exposures", {})
    if not exposures or sum(exposures.values()) != step * len(saved["models"]):
        raise RuntimeError("Checkpoint exposure count disagrees with committed updates")
    states = saved.get("optimizer", {}).get("state", {})
    if not states or any(int(state["step"]) != step for state in states.values()):
        raise RuntimeError("Checkpoint lacks the complete optimizer update state")
    def finite(value):
        if torch.is_tensor(value):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(v) for v in value)
        return True
    if not finite(saved["models"]) or not finite(saved["optimizer"]):
        raise RuntimeError("Nonfinite model or optimizer checkpoint")
    return saved


def recover(command, output, *, first_update=100, resume_updates=100, deadline=600, backend="xpu"):
    output = Path(output)
    checkpoint = Path(command[command.index("--checkpoint-dir")+1])/"checkpoint_latest.pth.tar"
    if checkpoint.exists():
        raise FileExistsError(f"Use a new smoke root; existing checkpoint: {checkpoint}")
    fresh = run_segment(command, output/"fresh.log", stop_step=first_update, deadline=deadline)
    before = load_partial(checkpoint, backend=backend)
    if before["step"] < first_update or fresh["result"].get("step") != before["step"]:
        raise RuntimeError("First leg did not commit its requested updates")
    # An explicit path makes a missing checkpoint an error, never a fresh run.
    resumed = ["--resume" if token == "--no-resume" else token for token in command]
    resumed += ["--resume-path", str(checkpoint)]
    # Stop at the unchanged checkpoint/log cadence. A signal during next-view
    # target preparation can legitimately retain only the last periodic save.
    interval = math.lcm(before["recipe"]["checkpoint_every"], before["recipe"]["log_every"])
    resume_target = math.ceil((before["step"]+resume_updates)/interval)*interval
    continuation = run_segment(resumed, output/"resume.log",
                               stop_step=resume_target, deadline=deadline)
    after = load_partial(checkpoint, backend=backend)
    for key in ("contract", "recipe", "target_identity"):
        if before[key] != after[key]:
            raise RuntimeError(f"Resume changed {key}")
    if (after["step"] < before["step"]+resume_updates
            or continuation["result"].get("step") != after["step"]
            or set(before["exposures"]) != set(after["exposures"])
            or any(after["exposures"][k] < v for k, v in before["exposures"].items())):
        raise RuntimeError("Resume did not preserve and advance training progress")
    return dict(status="passed", purpose="bounded production-recipe recovery, not convergence",
                first_step=before["step"], final_step=after["step"],
                production_steps=after["recipe"]["steps"],
                final_learning_rates=[g["lr"] for g in after["optimizer"]["param_groups"]],
                checkpoint=str(checkpoint), fresh=fresh, resume=continuation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "train", "all"), default="all")
    parser.add_argument("--first-update", type=int, default=100)
    parser.add_argument("--resume-updates", type=int, default=100)
    args = parser.parse_args()
    from rift_pvc import accelerator
    from rift_pvc.geraf_source import recipe_from_config
    if not os.environ.get("SLURM_JOB_ID") or accelerator.backend() != "xpu" or accelerator.device_count() != 1:
        raise RuntimeError("Requires exactly one allocated PVC card")
    if min(args.first_update, args.resume_updates) < 100 or args.first_update % 100:
        parser.error("First update must be a multiple of the production checkpoint cadence (100); resume at least 100 updates")
    plan = json.loads(args.plan.read_text())
    command, output = plan["command"], args.plan.parent
    config = json.loads(Path(command[command.index("--source-config")+1]).read_text())
    canonical = json.loads((ROOT/"protocols/geraf_mf48_1t1r.json").read_text())
    if recipe_from_config(config, .15) != recipe_from_config(canonical, .15):
        raise ValueError("Use the unchanged production GeRaF configuration for this smoke")
    report = dict(status="running", job_id=os.environ["SLURM_JOB_ID"], stage=args.stage,
                  accelerator=accelerator.describe(), command=command)
    try:
        if args.stage in ("prepare", "all"):
            report["prepare"] = run_segment(command+["--prepare-only"], output/"prepare.log", deadline=1800)
        if args.stage in ("train", "all"):
            report["recovery"] = recover(command, output, first_update=args.first_update,
                                         resume_updates=args.resume_updates)
        report["status"] = "prepared" if args.stage == "prepare" else "passed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (output/f"report_{os.environ['SLURM_JOB_ID']}.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
