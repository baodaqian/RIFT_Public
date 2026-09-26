#!/usr/bin/env python3
"""Run the canonical B787 PVC plan, stop cooperatively, resume, and audit."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import train_rift_dataset_pvc as frontend
from rift.rift_dataset import role_manifest
from rift_pvc import accelerator
from scripts_pvc.sh_sas_pvc_smoke_report import checkpoint_summary, main as report_main


def run_stage(command, seconds, path):
    start = time.monotonic()
    events = []
    with path.open("w") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)

        def consume():
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
                events.append({"seconds": time.monotonic() - start, "line": line.rstrip()})

        reader = threading.Thread(target=consume)
        reader.start()
        try:
            process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=300)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        finally:
            reader.join()
            process.stdout.close()
    return {"command": command, "exit_code": process.returncode,
            "wall_seconds": time.monotonic() - start, "events": events}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--term-after", type=float, default=1500)
    parser.add_argument("--resume-after", type=float, default=900)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or accelerator.backend() != "xpu":
        parser.error("this smoke requires a Slurm PVC allocation")
    if min(args.term_after, args.resume_after) <= 0 or args.term_after + args.resume_after > 2700:
        parser.error("positive stage times totaling at most 2700 seconds are required")
    args.output_root.mkdir(parents=True, exist_ok=False)
    cli = ["--dataset-root", args.dataset_root, "--output-root", str(args.output_root),
           "--object", "b787", "--method", "sh_sas", "--num-train", "2400",
           "--num-tx", "1", "--num-rx", "1", "--pvc-smoke"]
    plan = frontend.make_plan(frontend.parse_args(cli))
    (args.output_root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    entry, = plan["plans"]
    command, = entry["commands"]
    # Use exactly the frontend's selected-manifest materialization and command.
    frontend.base.write_selected_manifest(entry["role_manifest_path"],
                                         role_manifest(entry["object"], 2400, plan["antenna_selection"]))
    checkpoint = Path(entry["output_dir"]) / "checkpoint_latest.pth.tar"
    for number, seconds in ((1, args.term_after), (2, args.resume_after)):
        cmd = command if number == 1 else command + ["--resume", str(checkpoint)]
        stage = run_stage(cmd, seconds, args.output_root / f"phase{number}.log")
        final = checkpoint.with_name("checkpoint_final.pth.tar")
        path = final if number == 2 and final.is_file() else checkpoint
        stage["checkpoint_path"] = str(path)
        try:
            stage["checkpoint"] = checkpoint_summary(path)
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            stage["checkpoint_error"] = str(exc)
        (args.output_root / f"phase{number}.json").write_text(json.dumps(stage, indent=2) + "\n")
        if stage["exit_code"] != 0 or "checkpoint" not in stage:
            print(f"SMOKE_EXIT=1: phase {number} failed", flush=True)
            # Even an early failure publishes a machine-readable failed report.
            report_main([str(args.output_root)])
            return 1
    return report_main([str(args.output_root)])


if __name__ == "__main__":
    raise SystemExit(main())
