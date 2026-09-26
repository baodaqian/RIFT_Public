#!/usr/bin/env python3
"""PVC (Intel XPU) frontend for the six-object homemade RIFT dataset.

Planning is delegated, unchanged, to ``train_rift_dataset.py`` (same objects,
roles, budgets, recipes, manifests and output layout). This frontend only

1. replaces each maintained trainer script by its ``_pvc`` entry point
   (``train.py`` -> ``train_pvc.py``, ``train_spinr_style.py`` ->
   ``train_spinr_style_pvc.py``, ...), refusing methods whose PVC entry point
   does not exist yet;
2. optionally bounds RIFT/grid/isotropic runs for a smoke test (``--pvc-epochs N``), which also
   suffixes the checkpoint name and execution-contract label so a smoke can
   never be confused with a production run; ``--pvc-smoke`` marks step-budget
   trainers whose recipe fixes the budget (Radar Fields ``source-adapted-v3``
   fixes ``--steps``) by suffixing ``--checkpoint-name`` only, the launcher
   bounding them by wall clock (SIGTERM, then ``--resume``);
2b. rewrites the ``--device cuda`` a CUDA planner emits (Radar Fields) to the
   accelerator device;
3. launches the commands exactly as the CUDA frontend does (subprocess per
   command, inside a Slurm allocation, derived manifests written first).

Use ``--dry-run`` for read-only planning. Nothing here modifies the CUDA
pipeline; ``train_rift_dataset.py`` on disk is untouched.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_rift_dataset as base  # noqa: E402  (unchanged planner)
from rift.rift_dataset import PARENT_NUM_TRAIN, PROJECT_ROOT, role_manifest  # noqa: E402
from rift_pvc import accelerator  # noqa: E402

PVC_ENTRYPOINTS = {
    "train.py": "train_pvc.py",
    "train_spinr_style.py": "train_spinr_style_pvc.py",
    "train_geraf.py": "train_geraf_pvc.py",
    "train_sugavanam_ertin.py": "train_sugavanam_ertin_pvc.py",
    "train_radar_fields.py": "train_radar_fields_pvc.py",
    "train_radarsplat.py": "train_radarsplat_pvc.py",
    "scripts/prepare_radarsplat_b7873200_targets.py": "scripts_pvc/prepare_radarsplat_b7873200_targets_pvc.py",
    "train_sh_sas.py": "train_sh_sas_pvc.py",
    "scripts/eval_rift_dataset_model_free.py": "scripts_pvc/eval_rift_dataset_model_free_pvc.py",
}
SMOKE_SUFFIX = "_pvcsmoke"


def _script_key(token: str) -> str:
    path = Path(token)
    if path.is_absolute():
        try:
            return path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return path.name
    return path.as_posix()


def to_pvc_command(command: list[str]) -> list[str]:
    """Map a planned CUDA command to its PVC twin; raise if not yet available."""
    cmd = list(command)
    index = 1 if cmd and cmd[0] == sys.executable else 0
    key = _script_key(cmd[index])
    if key not in PVC_ENTRYPOINTS:
        raise ValueError(f"No PVC entry point is registered for {key}")
    target = ROOT / PVC_ENTRYPOINTS[key]
    if not target.is_file():
        raise FileNotFoundError(
            f"PVC entry point not available yet: {target.name} (see RIFT_PVC_Adaptation.md)")
    cmd[index] = str(target) if Path(cmd[index]).is_absolute() else PVC_ENTRYPOINTS[key]
    return cmd


def _replace_value(cmd: list[str], flag: str, value: str, *, suffix: bool = False) -> bool:
    if flag not in cmd:
        return False
    i = cmd.index(flag) + 1
    cmd[i] = (cmd[i] + value) if suffix else value
    return True


def bound_for_smoke(cmd: list[str], epochs: int) -> list[str]:
    """Bound a train_pvc.py command to ``epochs`` and mark it as a smoke run."""
    cmd = list(cmd)
    required = ("--epochs", "--checkpoint-name", "--execution-contract-label")
    if any(flag not in cmd for flag in required):
        raise ValueError(f"--pvc-epochs requires a train_pvc.py command with {required}")
    _replace_value(cmd, "--epochs", str(int(epochs)))
    _replace_value(cmd, "--checkpoint-name", SMOKE_SUFFIX, suffix=True)
    _replace_value(cmd, "--execution-contract-label", SMOKE_SUFFIX, suffix=True)
    return cmd


def remap_device(cmd: list[str]) -> list[str]:
    """``--device cuda[:N]`` from the CUDA planner -> the accelerator device."""
    cmd = list(cmd)
    if "--device" in cmd:
        i = cmd.index("--device") + 1
        if i < len(cmd) and cmd[i].startswith("cuda"):
            cmd[i] = str(accelerator.device())
    return cmd


def mark_smoke(cmd: list[str]) -> list[str]:
    """Keep the production argument list; suffix --checkpoint-name for a wall-clock-bounded smoke."""
    cmd = list(cmd)
    if not _replace_value(cmd, "--checkpoint-name", SMOKE_SUFFIX, suffix=True):
        raise ValueError("--pvc-smoke requires a command that carries --checkpoint-name")
    return cmd


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--pvc-epochs", type=int, default=None,
                     help="RIFT/grid/isotropic only: override --epochs of train_pvc.py commands and suffix "
                          f"checkpoint name / contract label with {SMOKE_SUFFIX}")
    pre.add_argument("--pvc-smoke", action="store_true",
                     help="Step-budget trainers (including SH-SAS): keep the production argument list, suffix "
                          f"--checkpoint-name and the output dir with {SMOKE_SUFFIX}; bound the run by wall clock")
    pvc, rest = pre.parse_known_args(argv)
    if pvc.pvc_epochs is not None and pvc.pvc_smoke:
        pre.error("--pvc-epochs and --pvc-smoke are mutually exclusive")
    if pvc.pvc_epochs is not None and pvc.pvc_epochs <= 0:
        pre.error("--pvc-epochs must be positive")
    if "-h" in rest or "--help" in rest:
        pre.print_help()
        print("\nAll other arguments are those of train_rift_dataset.py:\n")
    args = base.parse_args(rest)
    args.pvc_epochs = pvc.pvc_epochs
    args.pvc_smoke = pvc.pvc_smoke
    return args


def make_plan(args):
    plan = base.make_plan(args)
    plan["entrypoint"] = "train_rift_dataset_pvc.py"
    plan["accelerator"] = "xpu"
    plan["pvc_epochs"] = args.pvc_epochs
    plan["pvc_smoke"] = args.pvc_smoke
    for entry in plan["plans"]:
        pvc_commands = []
        for command in entry["commands"]:
            command = remap_device(to_pvc_command(command))
            if args.pvc_smoke:
                command = mark_smoke(command)
            if args.pvc_epochs is not None:
                script = _script_key(command[1 if command[0] == sys.executable else 0])
                if script != "train_pvc.py":
                    raise ValueError(
                        f"--pvc-epochs supports only RIFT/grid/isotropic via train_pvc.py; "
                        f"{entry['method']} uses {script}. Select a supported method or omit --pvc-epochs.")
                command = bound_for_smoke(command, args.pvc_epochs)
            pvc_commands.append(command)
        entry["cuda_commands"] = entry["commands"]
        entry["commands"] = pvc_commands
        if args.pvc_epochs is not None or args.pvc_smoke:
            entry["output_dir"] = entry["output_dir"] + SMOKE_SUFFIX
    return plan


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        print(json.dumps({"dataset": "RIFT dataset", "entrypoint": "train_rift_dataset_pvc.py",
                          "methods": base.METHODS, "pvc_entrypoints": PVC_ENTRYPOINTS,
                          "available": sorted(k for k, v in PVC_ENTRYPOINTS.items() if (ROOT / v).is_file())},
                         indent=2))
        return 0
    plan = make_plan(args)
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return 0
    if not args.check_initialization and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Training/preparation requires an experiment-manager allocation; use --dry-run")
    if not args.check_initialization and not args.resume:
        for entry in plan["plans"]:
            path = Path(entry["output_dir"])
            occupied = (path.exists() and any(p.name != "targets" or entry["method"] in base.MERGED_BASELINES
                                              for p in path.iterdir()))
            if occupied:
                raise ValueError(f"Existing output requires explicit --resume or a new output root (--output-root): {path}")
    if args.num_train != PARENT_NUM_TRAIN or plan["antenna_selection"]:
        for entry in plan["plans"]:
            base.write_selected_manifest(entry["role_manifest_path"],
                                         role_manifest(entry["object"], args.num_train, plan["antenna_selection"]))
    for entry in plan["plans"]:
        for command in entry["commands"]:
            print(shlex.join(command), flush=True)
            result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
            if result.returncode:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
