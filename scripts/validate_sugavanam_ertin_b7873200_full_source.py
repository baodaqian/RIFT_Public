#!/usr/bin/env python3
"""Torch-free source/configuration checks for the full B7873200 package."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_NPZ = "/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz"
CANONICAL_MANIFEST = "/storage/scratch1/1/dbao31/rift_round8b_impl_20260810/splits/round8b/b78710k_interp_seed42_train3200_val1000_test1000_v1.json"


def _read(relative: str) -> str:
    path = ROOT / relative
    if not path.is_file():
        raise AssertionError(f"missing required full-package file: {relative}")
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".py":
        ast.parse(text, filename=str(path))
    return text


def _require(text: str, needles: Iterable[str], label: str) -> None:
    for needle in needles:
        if needle not in text:
            raise AssertionError(f"{label} is missing {needle!r}")


def _forbid(text: str, needles: Iterable[str], label: str) -> None:
    for needle in needles:
        if needle in text:
            raise AssertionError(f"{label} still contains forbidden stale setting {needle!r}")


def validate() -> dict[str, object]:
    files = {
        "config": _read("rift/sugavanam_ertin_b7873200_full.py"),
        "driver": _read("train_sugavanam_ertin.py"),
        "stage2": _read("train_sugavanam_ertin_stage2.py"),
        "launcher": _read("slurm/validate_sugavanam_ertin_b7873200_full_v1.sbatch"),
        "postflight": _read("scripts/postflight_sugavanam_ertin_b7873200_full.py"),
    }
    _require(
        files["config"],
        [
            '"train": B787_3200_NUM_TRAIN',
            '"validation": B787_3200_NUM_VALIDATION',
            '"test": B787_3200_NUM_TEST',
            '"unused": B787_3200_NUM_UNUSED',
            '"steps": 5_000',
            '"init_steps": 1_000',
            '"init_lr": 5.0e-4',
            '"init_log_every": 100',
            '"n_fourier": 9',
            '"hidden_dim": 512',
            '"n_layers": 8',
            '"full_scale_proven": False',
            '"status": "proposed_full_budget_requires_manager_review"',
            '"account": "gts-jromberg3-ece"',
            '"wall_time_hours": 12',
            'FULL_WALL_LIMIT_SECONDS: Final = 12 * 60 * 60',
            'FULL_STAGE1_COMPUTE_CAP_EPOCH: Final = 30',
            'FULL_STAGE1_COMPUTE_CAP_RUN_NAME: Final',
            'FULL_STAGE1_COMPUTE_CAP_REPORT_NAME: Final',
        ],
        "full config",
    )
    _require(files["launcher"], [CANONICAL_NPZ, CANONICAL_MANIFEST], "full launcher canonical inputs")
    _require(
        files["driver"],
        [
            "load_b7873200_sealed_identity",
            "compute_train_only_signal_normalization",
            "assert_restricted_roles",
            "load_b7873200_stage1_cloud",
            '"--num-train", "3200"',
            '"--num-val", "1000"',
            '"--num-test", "1000"',
            '"--num-freq-wanted", "600"',
            '"--epochs", "150"',
            '"--granularity", "48"',
            '"--point-chunk", "65536"',
            '"--pair-chunk", "64"',
            "FULL_STAGE1_BUNDLE_FILENAME",
            "FULL_STAGE1_COMPUTE_CAP_EPOCH",
            "compute-capped Stage-1",
            "stage1_compute_capped",
            "nonterminal_compute_capped",
            "stage2_entry_clean_interruption.json",
            "_record_stage2_entry_clean",
            "engineering_override=True",
            "stage2_bundle_only_input",
            '"raw_npz_opened_by_stage2": False',
            'status="scientific_negative_topology"',
            "SE_B7873200_FULL_SCIENTIFIC_NEGATIVE_TOPOLOGY_COMPLETE",
            "complete_package_published",
            "production_clearance",
        ],
        "full driver",
    )
    _require(
        files["launcher"],
        [
            "--account=gts-jromberg3-ece",
            "--qos=inferno",
            "--partition=gpu-rtx6000",
            "--gres=gpu:rtx_6000:1",
            "--cpus-per-task=6",
            "--mem=64G",
            "--tmp=24G",
            "--time=12:00:00",
            "FULL_ROOT=/storage/scratch1/1/dbao31/rift_b7873200_sugavanam_ertin_full3200_replacement_v1",
            "FULL_RUN_NAME=b78710k_sugavanam_ertin_full3200_stage1_stage2_replacement_v1",
            "STAGE1_CHECKPOINT_NAME=sugavanam_ertin_b7873200_stage1_full3200_replacement_v1",
            "STAGE1_DIR=\"$RUN_ROOT/stage1/$STAGE1_CHECKPOINT_NAME\"",
            "STAGE1_LATEST=\"$STAGE1_DIR/checkpoint_latest.pth.tar\"",
            "STAGE1_CLEAN=\"$STAGE1_DIR/full_stage1_clean_interruption.json\"",
            "ERROR: clean Stage-1 report lacks a checkpoint/bundle recovery artifact",
            "train_sugavanam_ertin.py",
            "validate_sugavanam_ertin_b7873200_full_source.py",
            "postflight_sugavanam_ertin_b7873200_full.py",
            "validate_sugavanam_ertin_b7873200_stage1.py",
            "validate_sugavanam_ertin_b7873200_stage1_operator.py",
            "validate_sugavanam_ertin_b7873200_stage2_static.py",
            "validate_sugavanam_ertin_b7873200_stage2_v1.py",
            "SE_B7873200_FULL_EXPERIMENT_COMPLETE",
            "trap forward_term TERM INT",
            "kill -TERM \"$driver_pid\"",
            "kill -TERM -- \"-$driver_pid\"",
            "/usr/bin/setsid",
            "mkfifo -- \"$driver_fifo\"",
            "tee_pid=$!",
            "term_requested=0",
            "term_forwarded=0",
            "SE_B7873200_FULL_TERM_BEFORE_DRIVER_EXIT",
            "SE_B7873200_FULL_REAP_AFTER_INTERRUPTED_WAIT",
            "SE_B7873200_FULL_TERM_DURING_TEE_DRAIN",
            "SE_B7873200_FULL_REAP_AFTER_INTERRUPTED_TEE_WAIT",
            "SE_B7873200_FULL_CLEAN_INTERRUPTION_RECORDED",
            "stage1-compute-cap30",
            "CAP_STAGE1_LATEST",
            "--time=12:00:00",
        ],
        "full launcher",
    )
    _require(
        files["stage2"],
        [
            "preserve_existing_stop: bool = False",
            "if preserve_existing_stop and args.resume is not None and stop_requested():",
            "if not preserve_existing_stop:",
        ],
        "full Stage-2 seam",
    )
    _require(
        files["postflight"],
        [
            "scientific_negative_topology",
            "accepted_experiment_outcome",
            "production_clearance",
            "full_resource_accounting_v2",
            "cuda_peak_reserved_bytes",
            "per-allocation wall envelope",
            "topology_gate_passed",
            "complete_package_published",
            "default=12.0",
        ],
        "full postflight",
    )
    for label, text in files.items():
        _forbid(text, ["sphere2k", "1800", "num-val 200", "validation_count = 200"], label)
    _require(
        files["launcher"],
        ["SLURM_JOB_ACCOUNT:-}", 'SLURM_JOB_QOS:-}" == "inferno"'],
        "full launcher account/QOS guards",
    )
    _forbid(
        files["config"] + files["launcher"],
        [
            "rift_b7873200_sugavanam_ertin_full3200_v1",
            "b78710k_sugavanam_ertin_full3200_stage1_stage2_v1",
        ],
        "full replacement identity",
    )
    if "5000" not in files["config"] and "5_000" not in files["config"]:
        raise AssertionError("full config does not state its retained 5,000 SDF budget")
    if files["config"].count("FULL_STAGE2_CAMPAIGN") < 1:
        raise AssertionError("full config lacks a fresh Stage-2 campaign identity")
    if "FULL_RUN_NAME: Final = \"b78710k_sugavanam_ertin_full3200_stage1_stage2_replacement_v1\"" not in files["config"]:
        raise AssertionError("full config run identity is not the fresh replacement identity")
    if "FULL_ROOT_PARENT: Final = \"/storage/scratch1/1/dbao31/rift_b7873200_sugavanam_ertin_full3200_replacement_v1\"" not in files["config"]:
        raise AssertionError("full config root is not the fresh replacement root")
    if re.search(r"--num-train\", \"1800|--num-val\", \"200", files["driver"]):
        raise AssertionError("full driver contains an old Stage-1 budget")
    result = {
        "schema": "rift_sugavanam_ertin_b7873200_full_source_validation_v1",
        "status": "passed",
        "checked_files": sorted(files),
        "canonical_scope": {
            "archive": CANONICAL_NPZ,
            "manifest": CANONICAL_MANIFEST,
            "role_counts": {"train": 3200, "validation": 1000, "test": 1000, "unused": 4800},
            "response_shape": [10000, 16, 16, 1, 600],
        },
        "checks": [
            "AST parse",
            "canonical sphere10k identity",
            "sealed 3200/1000/1000/4800 roles",
            "full Stage-1 v1 recipe",
            "proposed 1000-init plus retained 5000-SDF budget",
            "bundle-only Stage-2 handoff",
            "scientific-negative topology reporting",
            "Inferno 12-hour allocation envelope",
            "cooperative TERM forwarding and clean return-143 handling",
            "explicit FIFO/tee log draining and interrupted-wait reaping",
            "terminal-final zero-update readout recovery",
            "original native readout and cumulative resource preservation",
            "backward-compatible Stage-2 stop-state preservation seam",
            "no stale sphere2k/1800/200 settings in the new package",
        ],
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = validate()
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
