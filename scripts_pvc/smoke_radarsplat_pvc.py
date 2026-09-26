#!/usr/bin/env python3
"""Validate bounded production-recipe RadarSplat recovery on PVC.

Stop at the existing 100-update logging/checkpoint cadence, then continue the
same 2000-update recipe. Deadlines are watchdog failures, not successful smoke
outcomes. Both legs must save valid, advancing checkpoints after our SIGTERM.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

FALLBACK = "fallback from XPU to CPU"


def run_segment(command, log_path, *, stop_step=None, deadline=900, grace=120):
    started = time.monotonic()
    requested = None
    reason = None
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    watcher = selectors.DefaultSelector()
    watcher.register(process.stdout, selectors.EVENT_READ)
    pending = b""

    def stop(why):
        nonlocal requested, reason
        if reason not in ("fallback", "watchdog"):
            reason = why
        if requested is None:
            requested = time.monotonic()
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    try:
        with Path(log_path).open("xb") as log:
            while watcher.get_map():
                now = time.monotonic()
                if requested is None and now - started > deadline:
                    stop("watchdog")
                if requested is not None and now - requested > grace:
                    raise TimeoutError("RadarSplat did not stop within the checkpoint grace period")
                for key, _ in watcher.select(timeout=.2):
                    block = os.read(key.fileobj.fileno(), 65536)
                    if not block:
                        watcher.unregister(key.fileobj)
                        continue
                    log.write(block)
                    log.flush()
                    pending += block
                    # Scan before line splitting so warnings without a newline
                    # also stop the process and fail the acceptance result.
                    if FALLBACK.encode() in pending:
                        stop("fallback")
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        try:
                            progress = json.loads(line)
                        except ValueError:
                            continue
                        if (requested is None and stop_step is not None and isinstance(progress, dict)
                                and type(progress.get("step")) is int and "total" in progress
                                and progress["step"] >= stop_step):
                            stop("update_bound")
            code = process.wait(timeout=grace)
    finally:
        watcher.close()
        process.stdout.close()
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    if FALLBACK in Path(log_path).read_text(errors="replace"):
        raise RuntimeError(f"XPU operator fallback: {log_path}")
    if reason == "watchdog":
        raise TimeoutError(f"RadarSplat exceeded the smoke watchdog: {log_path}")
    expected = 0 if stop_step is None else 143
    if code != expected or (stop_step is not None and reason != "update_bound"):
        raise RuntimeError(f"RadarSplat exit {code}, reason {reason}; expected {expected}: {log_path}")
    return dict(returncode=code, stop_reason=reason, wall_seconds=time.monotonic()-started,
                log=str(log_path), fallback_warning=False)


def load_partial(path, cache, *, profile="budget48", backend="xpu"):
    from rift.radarsplat_release import SCHEMA, create_scene, intensity_mapping_from_identity, position_scheduler
    from rift.radarsplat_release_training import identity_for_cache, restore_state
    from rift_pvc.radarsplat_xpu_backend import check_resume_sidecar
    import train_radarsplat as lifecycle
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"Required checkpoint is missing: {path}")
    record = check_resume_sidecar(path.parent)
    if record is None or record.get("accelerator", {}).get("backend") != backend:
        raise RuntimeError("Checkpoint does not record the required accelerator backend")
    saved = lifecycle._load_checkpoint(path, torch.device("cpu"))
    # The saved run's own image domain (decision 3): identities without the key are linear.
    identity = identity_for_cache(cache, profile, intensity_mapping_from_identity(saved.get("identity") or {}))
    if saved.get("schema") != SCHEMA or saved.get("identity") != identity:
        raise RuntimeError("Smoke checkpoint changed the production recipe or dataset identity")
    if not lifecycle._directly_equal(saved.get("acquisition_record"), cache.acquisition_record):
        raise RuntimeError("Smoke checkpoint changed acquisition calibration")
    if type(saved.get("step")) is not int or not 0 < saved["step"] < 2000:
        raise RuntimeError("Recovery smoke requires an incomplete checkpoint with committed updates")
    splats, optimizers = create_scene(scene_scale=identity["adapter"]["initialization_scene_scale"],
                                     scene_center=np.zeros(3), device="cpu",
                                     num_points=identity["model_recipe"]["init_num_pts"])
    restore_state(saved, splats, optimizers, position_scheduler(optimizers),
                  lifecycle.DeterministicViewSampler(cache.train_indices, 42))
    return saved


def recover(command, output, *, first_step=100, resume_steps=100, deadline=900, resume_deadline=600,
            backend="xpu"):
    from rift.radarsplat_b7873200_protocol import load_cache
    import train_radarsplat as lifecycle
    output = Path(output)
    checkpoint = Path(command[command.index("--checkpoint-dir")+1]) / "checkpoint_latest.pt"
    folder = checkpoint.parent
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Fresh recovery smoke needs an empty checkpoint directory: {folder}")
    if "--no-resume" not in command or "--resume" in command:
        raise ValueError("Fresh smoke command must explicitly select --no-resume")
    profile = command[command.index("--fidelity-profile")+1]
    if profile != "budget48":
        raise ValueError("This smoke preserves the selected budget48 production recipe")
    cache = load_cache(Path(command[command.index("--cache-root")+1]))
    if cache.is_development_subset:
        raise ValueError("Production-argument smoke requires all registered development views")
    fresh = run_segment(command, output/"fresh.log", stop_step=first_step, deadline=deadline)
    before = load_partial(checkpoint, cache, profile=profile, backend=backend)
    if before["step"] < first_step:
        raise RuntimeError("Fresh leg did not commit the requested updates")
    from rift.radarsplat_release import DEFAULT_INTENSITY, intensity_mapping_from_identity
    mapping = intensity_mapping_from_identity(before["identity"])
    if mapping != DEFAULT_INTENSITY:
        raise RuntimeError(f"Fresh leg trained in {mapping}, not the production default {DEFAULT_INTENSITY}")
    target = ((before["step"] + resume_steps + 99)//100)*100
    if target >= 2000:
        raise ValueError("Smoke bound must allow a partial resumed checkpoint before step 2000")
    resumed = ["--resume" if token == "--no-resume" else token for token in command]
    # Keep a durable first-leg snapshot: the native engine resumes latest in
    # the same directory and atomically replaces it at the next checkpoint.
    lifecycle._atomic_torch_save(output/"checkpoint_before_resume.pt", before)
    continuation = run_segment(resumed, output/"resume.log", stop_step=target, deadline=resume_deadline)
    after = load_partial(checkpoint, cache, profile=profile, backend=backend)
    for key in ("identity", "acquisition_record"):
        if not lifecycle._directly_equal(before[key], after[key]):
            raise RuntimeError(f"Resume changed {key}")
    if after["step"] < target or after["step"] < before["step"] + resume_steps:
        raise RuntimeError("Resume did not advance the committed optimizer updates")
    return dict(status="passed", purpose="bounded production-recipe recovery, not convergence",
                intensity_mapping=mapping,
                first_step=before["step"], final_step=after["step"], production_steps=2000,
                checkpoint=str(checkpoint), fresh=fresh, resume=continuation)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--stage", choices=("prepare", "train", "all"), default="all")
    p.add_argument("--first-step", type=int, default=100)
    p.add_argument("--resume-steps", type=int, default=100)
    p.add_argument("--deadline", type=float, default=900)
    p.add_argument("--resume-deadline", type=float, default=600)
    p.add_argument("--prepare-deadline", type=float, default=7200)
    p.add_argument("--per-job-checkpoints", action="store_true",
                   help="Train into <plan checkpoint dir>_smoke_<job id>: a fresh run beside the plan's reused "
                        "target cache, leaving earlier smoke checkpoints untouched")
    args = p.parse_args(argv)
    from rift_pvc import accelerator
    if not os.environ.get("SLURM_JOB_ID") or accelerator.backend() != "xpu" or accelerator.device_count() != 1:
        raise RuntimeError("Requires exactly one allocated PVC card")
    if (args.first_step < 100 or args.first_step % 100 or args.resume_steps < 100
            or args.first_step + args.resume_steps + 100 >= 2000
            or min(args.deadline, args.resume_deadline, args.prepare_deadline) <= 0):
        p.error("Use positive watchdogs and partial update bounds at the 100-step production cadence")
    plan = json.loads(args.plan.read_text())
    commands = plan["commands"]
    if len(commands) != 2:
        raise ValueError("Expected target preparation and training commands")
    if args.per_job_checkpoints:
        train = list(commands[1])
        index = train.index("--checkpoint-dir") + 1
        train[index] = f"{train[index]}_smoke_{os.environ['SLURM_JOB_ID']}"
        commands = [commands[0], train]
    output = args.plan.parent / f"smoke_{os.environ['SLURM_JOB_ID']}"
    output.mkdir(exist_ok=False)
    report = dict(status="running", job_id=os.environ["SLURM_JOB_ID"], stage=args.stage,
                  accelerator=accelerator.describe(), commands=commands)
    try:
        if args.stage in ("prepare", "all"):
            report["prepare"] = run_segment(commands[0], output/"prepare.log", deadline=args.prepare_deadline)
        if args.stage in ("train", "all"):
            report["recovery"] = recover(commands[1], output, first_step=args.first_step,
                                          resume_steps=args.resume_steps, deadline=args.deadline,
                                          resume_deadline=args.resume_deadline)
        report["status"] = "prepared" if args.stage == "prepare" else "passed"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
