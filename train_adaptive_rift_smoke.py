#!/usr/bin/env python3
"""Run the approved bounded B787 adaptive-RIFT action gate.

This is the only supported entrypoint for the 16-train/16-validation,
four-epoch B787 adaptive-capacity engineering smoke.  It derives a sealed
complete child manifest from the canonical parent before a response row is
materialized, then delegates all optimization to the ordinary RIFT trainer.

It is deliberately not a production launcher: no test response is exposed,
and a finite run without real split/unlock preservation and later coefficient
updates exits as an action-gate failure rather than an improvement claim.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

import torch

import train as rift_train
from rift.b7873200_adaptive_action_smoke import (
    ACTION_GATE_CHECKPOINT_NAME,
    ACTION_GATE_EXECUTION_LABEL,
    ACTION_GATE_SCHEMA,
    B787_3200_CANONICAL_MANIFEST_PATH,
    B787_3200_CANONICAL_NPZ_PATH,
    AdaptiveActionGateObserver,
    action_gate_train_argv,
    build_action_gate_manifest,
    load_canonical_parent_manifest,
    require_action_gate_pass,
    validate_action_gate_child_manifest,
    write_action_gate_manifest,
    write_action_gate_report,
)


def _resolved(path: str | os.PathLike[str]) -> str:
    return os.path.realpath(os.path.abspath(os.fspath(path)))


def _absolute_unresolved(path: str | os.PathLike[str]) -> Path:
    """Return a lexical absolute path without following symbolic links."""

    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _reject_symlink(path: Path, label: str) -> None:
    """Keep a fresh action identity from being redirected to another run."""

    if path.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True,
                        help="new engineering-only root; never a production checkpoint root")
    parser.add_argument("--npz-path", default=B787_3200_CANONICAL_NPZ_PATH)
    parser.add_argument("--parent-role-manifest", default=B787_3200_CANONICAL_MANIFEST_PATH)
    parser.add_argument("--resume", default=None,
                        help="only this run's checkpoint_latest.pth.tar after a clean interruption")
    return parser.parse_args(argv)


def _validate_cli(args: argparse.Namespace) -> Path:
    if _resolved(args.npz_path) != _resolved(B787_3200_CANONICAL_NPZ_PATH):
        raise ValueError("the action gate accepts only the canonical /storage/home B787 archive")
    if _resolved(args.parent_role_manifest) != _resolved(B787_3200_CANONICAL_MANIFEST_PATH):
        raise ValueError("the action gate accepts only the canonical parent interpolation manifest")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("the B787 action gate is a single-GPU engineering cell, not a distributed run")
    if not torch.cuda.is_available():
        raise RuntimeError("the B787 action gate requires an allocated CUDA GPU; do not run it on a login node")
    # Do not resolve these identities before checking them: a resolved v2
    # pathname could otherwise make a symlink to a terminal predecessor look
    # like this gate's own checkpoint root.
    checkpoint_root = _absolute_unresolved(args.checkpoint_root)
    run_root = checkpoint_root / ACTION_GATE_CHECKPOINT_NAME
    final_checkpoint = run_root / "checkpoint_final.pth.tar"
    latest_checkpoint = run_root / "checkpoint_latest.pth.tar"
    for path, label in (
        (checkpoint_root, "checkpoint root"),
        (run_root, "action-gate run root"),
        (latest_checkpoint, "action-gate latest checkpoint"),
        (final_checkpoint, "action-gate final checkpoint"),
    ):
        _reject_symlink(path, label)
    if final_checkpoint.exists():
        raise ValueError("the B787 action gate already completed or terminally reached final checkpoint; do not rerun it")
    if args.resume is None:
        if latest_checkpoint.exists():
            raise ValueError(
                "an action-gate latest checkpoint already exists; only an explicitly clean same-run resume is allowed"
            )
    else:
        resume = _absolute_unresolved(args.resume)
        _reject_symlink(resume, "resume checkpoint")
        if resume != latest_checkpoint or not resume.is_file():
            raise ValueError("resume must be this action gate's existing checkpoint_latest.pth.tar")
    return run_root


def _build_observer_factory(args: argparse.Namespace, selected_train: list[int], selected_validation: list[int], holder: dict):
    def factory(**context):
        observer = AdaptiveActionGateObserver(
            model=context["model"],
            optimizer=context["optimizer"],
            gain=context["gain"],
            train_loader=context["train_loader"],
            validation_loader=context["validation_loader"],
            criterion=context["criterion"],
            device=context["device"],
            num_freq_selected=context["num_freq_selected"],
            phase_sign=context["phase_sign"],
            forward_operator_name=context["forward_operator_name"],
            compute_dtype=context["compute_dtype"],
            data_format=context["data_format"],
            op_kwargs=context["op_kwargs"],
            occlusion=context["occlusion"],
            selected_train_ids=selected_train,
            selected_validation_ids=selected_validation,
        )
        if args.resume is not None:
            checkpoint = rift_train.load_tensor_checkpoint(args.resume, map_location="cpu")
            observer_state = checkpoint.get("adaptive_event_observer_state")
            if observer_state is None:
                raise ValueError("resume checkpoint lacks this action gate's observer state")
            observer.restore_checkpoint_state(observer_state)
        holder["observer"] = observer
        return observer
    return factory


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_root = _validate_cli(args)

    # This header-only parent preflight comes before the child manifest and
    # before generic train.main may construct any response dataset.
    parent_manifest, _parent_contract = load_canonical_parent_manifest(
        args.npz_path, args.parent_role_manifest
    )
    child_manifest = build_action_gate_manifest(parent_manifest)
    validate_action_gate_child_manifest(parent_manifest, child_manifest)
    manifest_path = write_action_gate_manifest(run_root / "derived_roles.json", child_manifest)
    split = child_manifest["split"]
    selected_train = list(split["train_indices"])
    selected_validation = list(split["validation_indices"])

    holder: dict[str, AdaptiveActionGateObserver] = {}
    train_argv = action_gate_train_argv(
        npz_path=args.npz_path,
        manifest_path=manifest_path,
        checkpoint_root=Path(args.checkpoint_root).expanduser().resolve(),
        resume=args.resume,
    )
    rift_train.main(
        train_argv,
        adaptive_event_observer_factory=_build_observer_factory(
            args, selected_train, selected_validation, holder
        ),
    )
    observer = holder.get("observer")
    if observer is None:
        raise RuntimeError("RIFT action gate did not construct its observer")

    final_checkpoint = run_root / "checkpoint_final.pth.tar"
    if not final_checkpoint.is_file():
        raise RuntimeError("RIFT action gate returned without its final checkpoint")
    checkpoint = rift_train.load_tensor_checkpoint(final_checkpoint, map_location="cpu")
    observer_state = checkpoint.get("adaptive_event_observer_state")
    if not isinstance(observer_state, dict) or observer_state.get("schema") != ACTION_GATE_SCHEMA:
        raise RuntimeError("final RIFT action-gate checkpoint lacks the required observer state")
    if observer_state.get("finished") is not True:
        raise RuntimeError("final RIFT action-gate checkpoint lacks a completed observer timeline")
    if checkpoint.get("execution_contract", {}).get("label") != ACTION_GATE_EXECUTION_LABEL:
        raise RuntimeError("final RIFT action-gate checkpoint has the wrong execution-contract label")

    report = observer.finalize()
    report.update({
        "input_archive_path": _resolved(args.npz_path),
        "parent_manifest_path": _resolved(args.parent_role_manifest),
        "derived_manifest_path": str(manifest_path.resolve()),
        "checkpoint_root": str(run_root.resolve()),
        "acquisition": {"tx": 16, "rx": 16, "frequencies": 600, "units": "metres"},
        "roles": {
            "train": selected_train,
            "validation": selected_validation,
            "reserved_test_count": len(split["test_indices"]),
            "unused_count": len(split["unused_indices"]),
            "test_and_unused_response_materialized": False,
        },
        "bp_initialization_source_view_ids": selected_train,
        "generic_trainer_argv": train_argv,
        "checkpoint_observer_finished": True,
    })
    report_path = write_action_gate_report(run_root / "action_gate_report.json", report)
    print(f"RIFT_B7873200_ADAPTIVE_ACTION_GATE_REPORT: {report_path}")
    require_action_gate_pass(report)
    print("RIFT_B7873200_ADAPTIVE_ACTION_GATE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
