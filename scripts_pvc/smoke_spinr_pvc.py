#!/usr/bin/env python3
"""Bounded B787 SpINR: original 2400-view recipe, XPU save/interruption/resume.

Only the terminal epoch budget (3) and device differ from the production
argument list. The cosine clock remains 150 epochs. Output is a separate
development run under Package A group scratch.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

import torch
import train_rift_dataset as planner
from rift.rift_dataset import role_manifest
from rift_pvc import accelerator


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str)+"\n")


def execute(command, log_path, *, interrupt_after_epoch=False):
    started = time.monotonic()
    requested = None
    fallback = False
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=False, bufsize=0)
        watcher = selectors.DefaultSelector()
        watcher.register(process.stdout, selectors.EVENT_READ)
        pending = b""
        try:
            while watcher.get_map():
                elapsed = time.monotonic()-started
                if elapsed > 1500 and requested is None:
                    process.send_signal(signal.SIGTERM)
                    requested = time.monotonic()
                if requested is not None and time.monotonic()-requested > 120:
                    process.kill()
                    raise TimeoutError("SpINR did not stop at a safe checkpoint within 120 seconds")
                for key, _ in watcher.select(timeout=1):
                    block = os.read(key.fileobj.fileno(), 65536)
                    if not block:
                        watcher.unregister(key.fileobj)
                        continue
                    pending += block
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        line = line.decode(errors="replace")
                        log.write(line+"\n")
                        log.flush()
                        print(line, flush=True)
                        if "Aten Op fallback from XPU to CPU" in line:
                            fallback = True
                            if requested is None:
                                process.send_signal(signal.SIGTERM)
                                requested = time.monotonic()
                        if interrupt_after_epoch and line.startswith("Epoch 1/3:") and requested is None:
                            process.send_signal(signal.SIGTERM)
                            requested = time.monotonic()
                if elapsed > 1620:
                    raise TimeoutError("SpINR segment exceeded bounded smoke time")
            if pending:
                last = pending.decode(errors="replace")
                log.write(last)
                fallback |= "Aten Op fallback from XPU to CPU" in last
            code = process.wait(timeout=30)
        finally:
            watcher.close()
            if process.poll() is None:
                process.kill()
                process.wait()
    if fallback:
        raise RuntimeError("XPU-to-CPU operator fallback detected")
    if code:
        raise RuntimeError(f"Trainer exit {code}; see {log_path}")
    if interrupt_after_epoch and requested is None:
        raise RuntimeError("The intended interruption was not exercised")
    if time.monotonic()-started > 1500:
        raise TimeoutError("SpINR segment exhausted its smoke budget")
    return {"wall_seconds":time.monotonic()-started, "interruption_requested":requested is not None,
            "fallback_warning":fallback, "returncode":code}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Requires a real PVC allocation")
    if accelerator.backend() != "xpu" or accelerator.device_count() != 1:
        raise RuntimeError("Requires exactly one PVC card")
    output = args.output.resolve()
    if not output.is_relative_to(Path("/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageA")):
        raise ValueError("Use the separate Package A scratch root")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status":"running", "started_at":datetime.now(timezone.utc).isoformat(),
              "job_id":os.environ["SLURM_JOB_ID"], "node":os.environ.get("SLURMD_NODENAME"),
              "accelerator":accelerator.describe(), "source_commit":subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "purpose":"development smoke; not production convergence", "test_accessed":False}
    save(output/"report.json", report)
    try:
        cli_args = planner.parse_args(["--dataset-root", os.environ["RIFT_DATA_ROOT"],
            "--output-root", str(output/"outputs"), "--object", "b787", "--method", "spinr",
            "--num-train", "2400", "--num-tx", "1", "--num-rx", "1",
            "--host-rss-limit-gib", "64", "--dry-run"])
        plan = planner.make_plan(cli_args)
        save(output/"production-plan.json", plan)
        entry = plan["plans"][0]
        planner.write_selected_manifest(entry["role_manifest_path"],
            role_manifest("b787", 2400, plan["antenna_selection"]))
        command = list(entry["commands"][0])
        assert Path(command[1]).name == "train_spinr_style.py"
        command[1] = str(ROOT/"train_spinr_style_pvc.py")
        command += ["--epochs", "3", "--device", "xpu"]
        checkpoint_dir = Path(command[command.index("--checkpoint-root")+1])/command[command.index("--checkpoint-name")+1]
        report["command"] = command
        report["production_changes"] = {"entrypoint":"train_spinr_style_pvc.py", "device":"xpu", "epochs":3}
        save(output/"report.json", report)
        report["fresh"] = execute(command, output/"fresh.log", interrupt_after_epoch=True)
        latest = checkpoint_dir/"checkpoint_latest.pth.tar"
        partial = torch.load(latest, map_location="cpu", weights_only=True)
        assert partial["execution"]["phase"] != "completed"
        assert "torch_xpu_all" in partial["rng_state"]
        report["interrupted_execution"] = partial["execution"]
        report["interrupted_epoch"] = partial["epoch"] if "epoch" in partial else partial.get("epoch_index")
        save(output/"report.json", report)
        report["resume"] = execute(command+["--resume", str(latest)], output/"resume.log")
        final_path = checkpoint_dir/"checkpoint_final.pth.tar"
        final = torch.load(final_path, map_location="cpu", weights_only=True)
        assert final["execution"]["phase"] == "completed"
        assert "torch_xpu_all" in final["rng_state"]
        assert len(final["history"]) == 3
        report.update(status="passed", final_checkpoint=str(final_path), history=final["history"],
                      final_execution=final["execution"], recipe=final["spinr_style_recipe"],
                      checkpoint_keys=list(final))
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save(output/"report.json", report)
        print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
