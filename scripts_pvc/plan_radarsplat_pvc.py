#!/usr/bin/env python3
"""Plan the two RadarSplat PVC commands and materialize the derived role manifest.

The bounded smoke has to interrupt and resume the fitting command itself, so it
cannot let ``train_rift_dataset_pvc.py`` own the subprocesses. This helper runs
exactly the frontend's planner and its manifest writer, performs the PVC twin of
the CUDA frontend's dependency gate (``load_xpu_reference`` instead of
``load_cuda_reference``), then prints the two commands it would have launched
(target preparation, then fitting) so the smoke executes the production
argument list unchanged.

    python scripts_pvc/plan_radarsplat_pvc.py --dataset-root ... --output-root ... \
        --object b787 --num-train 2400 --num-tx 1 --num-rx 1 --radarsplat-recipe budget48

Read-only apart from the derived role manifest, which is a local run input; the
parent dataset and splits are never rewritten.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_rift_dataset as base  # noqa: E402
import train_rift_dataset_pvc as frontend  # noqa: E402
from rift.rift_dataset import PARENT_NUM_TRAIN, role_manifest  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write-manifest", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight", action=argparse.BooleanOptionalAction, default=True,
                        help="Load the PVC reference (fork inventory, torch mirrors) before printing the plan")
    mine, rest = parser.parse_known_args(argv)
    args = frontend.parse_args([*rest, "--method", "radarsplat"])
    plan = frontend.make_plan(args)
    entries = [e for e in plan["plans"] if e["method"] == "radarsplat"]
    if len(entries) != 1:
        raise SystemExit(f"expected exactly one RadarSplat plan, got {len(entries)}")
    entry = entries[0]
    if len(entry["commands"]) != 2:
        raise SystemExit(f"RadarSplat plans exactly two commands (prepare, train); got {len(entry['commands'])}")
    if mine.preflight and entry.get("recipe") in ("upstream", "budget48"):
        from rift_pvc.radarsplat_xpu_backend import load_xpu_reference
        load_xpu_reference(device=args.device or None)
    if mine.write_manifest and (args.num_train != PARENT_NUM_TRAIN or plan["antenna_selection"]):
        base.write_selected_manifest(entry["role_manifest_path"],
                                     role_manifest(entry["object"], args.num_train, plan["antenna_selection"]))
    print(json.dumps({
        "entrypoint": plan["entrypoint"],
        "object": entry["object"],
        "num_train": plan["num_train"],
        "antenna_selection": plan["antenna_selection"],
        "role_manifest_path": entry["role_manifest_path"],
        "output_dir": entry["output_dir"],
        "recipe": entry.get("recipe"),
        "commands": entry["commands"],
        "prepare_command_line": shlex.join(entry["commands"][0]),
        "train_command_line": shlex.join(entry["commands"][1]),
        "cuda_command_lines": [shlex.join(c) for c in entry["cuda_commands"]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
