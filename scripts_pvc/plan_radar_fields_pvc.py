#!/usr/bin/env python3
"""Plan one Radar Fields PVC command and materialize its derived role manifest.

The bounded smoke interrupts and resumes the trainer itself (source-adapted-v3
fixes ``--steps`` at parse time, so the production argument list is kept and
the run is bounded by wall clock), so it cannot let ``train_rift_dataset_pvc.py``
own the subprocess. This helper runs exactly the frontend's planner (with
``--pvc-smoke``, which only suffixes ``--checkpoint-name``) and its manifest
writer, then prints the command it would have launched.

    python scripts_pvc/plan_radar_fields_pvc.py --dataset-root ... --output-root ... \
        --object b787 --num-train 2400 --num-tx 1 --num-rx 1 [--resume CKPT]

Read-only apart from the derived role manifest, which is a local run input.
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
    parser.add_argument("--production-names", action="store_true",
                        help="omit --pvc-smoke (production checkpoint name); default marks the run _pvcsmoke")
    mine, rest = parser.parse_known_args(argv)
    extra = [] if mine.production_names else ["--pvc-smoke"]
    args = frontend.parse_args([*rest, "--method", "radar_fields", *extra])
    plan = frontend.make_plan(args)
    entries = [e for e in plan["plans"] if e["method"] == "radar_fields"]
    if len(entries) != 1 or len(entries[0]["commands"]) != 1:
        raise SystemExit("expected exactly one Radar Fields plan with one command")
    entry = entries[0]
    if mine.write_manifest and (args.num_train != PARENT_NUM_TRAIN or plan["antenna_selection"]):
        base.write_selected_manifest(entry["role_manifest_path"],
                                     role_manifest(entry["object"], args.num_train, plan["antenna_selection"]))
    print(json.dumps({
        "entrypoint": plan["entrypoint"], "object": entry["object"], "num_train": plan["num_train"],
        "antenna_selection": plan["antenna_selection"], "role_manifest_path": entry["role_manifest_path"],
        "output_dir": entry["output_dir"], "pvc_smoke": plan.get("pvc_smoke"),
        "command": entry["commands"][0], "command_line": shlex.join(entry["commands"][0]),
        "cuda_command_line": shlex.join(entry["cuda_commands"][0]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
