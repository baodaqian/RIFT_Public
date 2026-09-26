"""Reviewed adaptive reporting/recovery workflow behind train.py.

Invoke through ``train.py --workflow adaptive-fullscale``; this module is not
a separate training entrypoint. Historical artifact identities stay intact.

This is one adaptive comparison lane around the ordinary sealed ``train.py``
entrypoint.  The driver validates the canonical header/roles before training,
installs the adaptive-only observation seam, and writes a technical report.
It never materializes the sealed-test or unused response roles, never claims
production clearance from a finite run, and supports a report-only recovery
when a completed final checkpoint exists but report artifacts do not.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

import torch

import train as rift_train
from rift.b7873200_adaptive_fullscale import (
    B787_3200_CANONICAL_MANIFEST_PATH,
    B787_3200_CANONICAL_NPZ_PATH,
    FULLSCALE_CHECKPOINT_NAME,
    FULLSCALE_EXECUTION_LABEL,
    FULLSCALE_SCHEMA,
    AdaptiveFullScaleObserver,
    fullscale_train_argv,
    write_fullscale_report,
)


def _resolved(path: str | os.PathLike[str]) -> str:
    return os.path.realpath(os.path.abspath(os.fspath(path)))


def _absolute_unresolved(path: str | os.PathLike[str]) -> Path:
    """Return an absolute lexical path without following symbolic links."""

    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _reject_symlink(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True,
                        help="new adaptive full-scale root; never the original RIFT root")
    parser.add_argument("--object", default=None, help="RIFT dataset object or alias")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--npz-path", default=None)
    parser.add_argument("--parent-role-manifest", default=None)
    parser.add_argument("--num-train", type=int, default=None,
                        help="Collection-only nested training count; use 2400 for Delta. Omitted retains the manifest's count.")
    parser.add_argument("--resume", default=None,
                        help="only this run's checkpoint_latest.pth.tar after a clean interruption")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="publish missing report evidence from a completed final checkpoint without retraining",
    )
    args = parser.parse_args(argv)
    if args.object is not None:
        from rift.rift_dataset import DEFAULT_ROOT, resolve_object_inputs
        npz, manifest = resolve_object_inputs(
            object_name=args.object, dataset_root=args.dataset_root or DEFAULT_ROOT,
            npz_path=args.npz_path, role_manifest_path=args.parent_role_manifest)
        args.npz_path, args.parent_role_manifest = str(npz), str(manifest)
    else:
        args.npz_path = args.npz_path or B787_3200_CANONICAL_NPZ_PATH
        args.parent_role_manifest = args.parent_role_manifest or B787_3200_CANONICAL_MANIFEST_PATH
    return args


def _validate_cli(args: argparse.Namespace) -> Path:
    from rift.rift_dataset import collection_manifest, load_object_contract, object_identity
    args.dataset_identity = None
    args.observer_type = AdaptiveFullScaleObserver
    args.execution_label = FULLSCALE_EXECUTION_LABEL
    args.checkpoint_name = FULLSCALE_CHECKPOINT_NAME
    if collection_manifest(args.parent_role_manifest):
        _, contract = load_object_contract(args.npz_path, args.parent_role_manifest, num_train=args.num_train)
        args.dataset_identity = contract["dataset_identity"]
        from rift.collection_adaptive import observer_type, execution_label
        args.observer_type = observer_type(contract)
        args.num_train = len(contract["role_ids"]["train"])
        if args.num_train != 3200:
            args.execution_label = execution_label(args.num_train)
            args.checkpoint_name = args.execution_label
        if contract.get('antenna_selection'):
            from .antenna_selection import acquisition_label
            args.execution_label = execution_label(args.num_train) + '_' + acquisition_label(contract['antenna_selection'])
            args.checkpoint_name = args.execution_label
        if getattr(args, "object", None) is not None and args.dataset_identity != object_identity(args.object):
            raise ValueError("Named adaptive object does not match the archive/manifest")
    else:
        if args.num_train not in (None, 3200):
            raise ValueError("Training subsets require a RIFT collection object manifest; historical B787 gates stay fixed")
        if getattr(args, "object", None) is not None:
            raise ValueError("Named adaptive objects require their RIFT dataset manifest")
        if _resolved(args.npz_path) != _resolved(B787_3200_CANONICAL_NPZ_PATH):
            raise ValueError("adaptive full-scale run requires a RIFT dataset object or canonical B787 archive")
        if _resolved(args.parent_role_manifest) != _resolved(B787_3200_CANONICAL_MANIFEST_PATH):
            raise ValueError("adaptive full-scale run requires a collection or canonical B787 role manifest")
    if args.resume is not None and args.report_only:
        raise ValueError("--report-only cannot be combined with --resume")
    if not args.report_only:
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("adaptive full-scale run is single-GPU; distributed point_sh is unsupported")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("adaptive full-scale run requires exactly one allocated CUDA device")

    checkpoint_root = _absolute_unresolved(args.checkpoint_root)
    if args.dataset_identity:
        checkpoint_root = checkpoint_root / args.dataset_identity["object_id"]
    run_root = checkpoint_root / args.checkpoint_name
    latest_checkpoint = run_root / "checkpoint_latest.pth.tar"
    final_checkpoint = run_root / "checkpoint_final.pth.tar"
    report = run_root / "adaptive_fullscale_report.json"
    postflight = run_root / "adaptive_fullscale_postflight.json"
    for path, label in (
        (checkpoint_root, "checkpoint root"),
        (run_root, "adaptive full-scale run root"),
        (latest_checkpoint, "adaptive full-scale latest checkpoint"),
        (final_checkpoint, "adaptive full-scale final checkpoint"),
        (report, "adaptive full-scale report"),
        (postflight, "adaptive full-scale postflight"),
    ):
        _reject_symlink(path, label)

    if args.report_only:
        if not final_checkpoint.is_file():
            raise ValueError("--report-only requires this run's completed checkpoint_final.pth.tar")
        if postflight.exists():
            raise ValueError("report-only recovery refuses to overwrite existing postflight evidence")
        if report.exists() and not report.is_file():
            raise ValueError("existing adaptive full-scale report is not a regular file")
    elif args.resume is None:
        if latest_checkpoint.exists() or final_checkpoint.exists() or report.exists() or postflight.exists():
            raise ValueError(
                "adaptive full-scale output already exists; only an explicitly clean same-run resume is allowed"
            )
    else:
        resume = _absolute_unresolved(args.resume)
        _reject_symlink(resume, "resume checkpoint")
        if resume != latest_checkpoint or not resume.is_file():
            raise ValueError(
                "resume must be this adaptive full-scale run's existing checkpoint_latest.pth.tar"
            )
        if final_checkpoint.exists() or report.exists() or postflight.exists():
            raise ValueError("a terminal adaptive full-scale artifact exists; it cannot be resumed")
    return run_root


def _validate_parent_header(
    npz_path: str | os.PathLike[str], manifest_path: str | os.PathLike[str], num_train=None
) -> dict[str, Any]:
    """Validate the complete parent partition before response materialization."""

    from rift.rift_dataset import collection_manifest, load_object_contract
    if collection_manifest(manifest_path):
        _arrays, contract = load_object_contract(npz_path, manifest_path, num_train=num_train)
    else:
        _arrays, contract = rift_train._load_sealed_npz_protocol_contract(
            npz_path, manifest_path, num_train=3_200, num_val=1_000, num_test=1_000)
    if contract.get("source_response_shape", contract.get("response_shape")) != [10_000, 16, 16, 1, 600]:
        raise ValueError("adaptive dataset response header is not [10000,16,16,1,600]")
    if contract.get("response_dtype") != "complex64":
        raise ValueError("adaptive dataset response dtype is not complex64")
    role_ids = contract.get("role_ids")
    if not isinstance(role_ids, dict):
        raise ValueError("sealed adaptive dataset contract lacks role IDs")
    expected_counts = {
        "train": len(contract["role_ids"]["train"]) if contract.get("dataset_identity") else 3_200,
        "validation": 1_000,
        "reserved_test": 1_000,
        "unused": 8000-len(contract["role_ids"]["train"]) if contract.get("dataset_identity") else 4_800,
    }
    for role, expected in expected_counts.items():
        values = role_ids.get(role)
        if not isinstance(values, list) or len(values) != expected:
            raise ValueError(f"sealed adaptive dataset {role} role count is not {expected}")
    if contract.get("response_access") != {
        "train_materialized": True,
        "validation_materialized": True,
        "reserved_test_materialized": False,
        "unused_materialized": False,
    }:
        raise ValueError("sealed adaptive dataset response-access policy is not train/validation only")
    return dict(contract)


def _quality_metric_artifacts(
    run_root: Path, *, allow_missing: bool = False
) -> list[dict[str, Any]]:
    """Describe the generic trainer's real per-attempt loss artifacts."""

    paths = sorted(run_root.glob("training_validation_losses_*.csv"))
    if not paths:
        if allow_missing:
            return []
        raise RuntimeError(
            "generic trainer returned without a training_validation_losses_TIMESTAMP.csv artifact"
        )
    artifacts: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        if not rows or rows[0] != ["Epoch", "Training Loss", "Validation Loss"]:
            raise RuntimeError(f"unexpected generic loss artifact header: {path}")
        data_rows = len(rows) - 1
        if data_rows < 1:
            raise RuntimeError(f"generic loss artifact has no epoch rows: {path}")
        artifacts.append({
            "path": str(path.absolute()),
            "data_rows": data_rows,
            "local_epoch_start": 1,
            "local_epoch_end": data_rows,
            "scope": (
                "one trainer attempt; resumed attempts restart their saved history at "
                "local epoch 1 and are not global checkpoint evaluations"
            ),
        })
    return artifacts


def _attach_report_context(
    report: dict[str, Any],
    *,
    args: argparse.Namespace,
    run_root: Path,
    contract: Mapping[str, Any],
    train_argv: Sequence[str],
    report_only: bool,
) -> dict[str, Any]:
    train_ids = list(contract["role_ids"]["train"])
    validation_ids = list(contract["role_ids"]["validation"])
    artifacts = _quality_metric_artifacts(run_root, allow_missing=report_only)
    report.update({
        "input_archive_path": _resolved(args.npz_path),
        "parent_manifest_path": _resolved(args.parent_role_manifest),
        "checkpoint_root": str(run_root.absolute()),
        "acquisition": {"tx": 16, "rx": 16, "frequencies": 600, "units": "metres"},
        "roles": {
            "train_count": len(train_ids),
            "validation_count": len(validation_ids),
            "reserved_test_count": len(contract["role_ids"]["reserved_test"]),
            "unused_count": len(contract["role_ids"]["unused"]),
            "train_validation_response_materialized_only": True,
        },
        "generic_trainer_argv": list(train_argv),
        "geometry_readout_status": (
            "pending_compatible_b787_evaluator; scripts/eval_scene_geometry.py is PEC-sphere-specific"
        ),
        "quality_metric_artifact": artifacts[-1]["path"] if artifacts else None,
        "quality_metric_artifacts": artifacts,
        "quality_metric_artifact_status": (
            "available"
            if artifacts
            else "unavailable: interruption occurred before generic loss-artifact publication"
        ),
        "quality_metric_epoch_coverage": (
            "CSV rows are per-attempt trainer history. A clean resume writes another "
            "timestamped artifact whose rows restart at local epoch 1; these rows are "
            "not a fixed-checkpoint evaluation and must not be relabeled as global epochs."
        ),
        "quality_metric_is_fixed_checkpoint_evaluation": False,
        "report_recovery_only": bool(report_only),
        "observer_finished_in_final_checkpoint": True,
    })
    return report


def _validate_final_checkpoint(
    checkpoint: Mapping[str, Any], final_checkpoint: Path, *,
    observer_schema=FULLSCALE_SCHEMA, execution_label=FULLSCALE_EXECUTION_LABEL
) -> dict[str, Any]:
    observer_state = checkpoint.get("adaptive_event_observer_state")
    if not isinstance(observer_state, dict) or observer_state.get("schema") != observer_schema:
        raise RuntimeError("final checkpoint lacks adaptive full-scale observer state")
    if observer_state.get("finished") is not True:
        raise RuntimeError("final checkpoint lacks a completed adaptive full-scale observer timeline")
    execution_contract = checkpoint.get("execution_contract")
    if not isinstance(execution_contract, Mapping) or execution_contract.get("label") != execution_label:
        raise RuntimeError("final checkpoint has the wrong adaptive full-scale execution label")
    if final_checkpoint.name != "checkpoint_final.pth.tar":
        raise RuntimeError("report-only recovery was not pointed at checkpoint_final.pth.tar")
    return observer_state


def _load_passing_report(report_path: Path, final_checkpoint: Path, *, observer_schema=FULLSCALE_SCHEMA) -> dict[str, Any]:
    """Read, but never rewrite, an already-published passing report."""

    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise RuntimeError("existing adaptive full-scale report is not a JSON object")
    if report.get("schema") != observer_schema:
        raise RuntimeError("existing adaptive full-scale report has the wrong schema")
    if report.get("engineering_status") != "fullscale_candidate_not_production_clearance":
        raise RuntimeError("existing adaptive full-scale report has the wrong status")
    if report.get("pass") is not True:
        raise RuntimeError("existing adaptive full-scale report failed; it is terminal")
    if report.get("production_clearance") is not False:
        raise RuntimeError("existing adaptive full-scale report has an invalid clearance state")
    checkpoint_path = report.get("observer_checkpoint_path")
    if not isinstance(checkpoint_path, str) or _resolved(checkpoint_path) != _resolved(final_checkpoint):
        raise RuntimeError("existing adaptive full-scale report points at another checkpoint")
    for field in (
        "cumulative_peak_process_max_rss_kib",
        "cumulative_peak_torch_allocated_bytes",
        "cumulative_peak_torch_reserved_bytes",
    ):
        if field not in report:
            raise RuntimeError(f"existing adaptive full-scale report lacks {field}")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_root = _validate_cli(args)
    contract = _validate_parent_header(args.npz_path, args.parent_role_manifest, args.num_train)
    train_ids = list(contract["role_ids"]["train"])
    validation_ids = list(contract["role_ids"]["validation"])

    train_argv = fullscale_train_argv(
        npz_path=args.npz_path,
        manifest_path=args.parent_role_manifest,
        checkpoint_root=str(run_root.parent),
        resume=args.resume,
    )
    if args.num_train is not None and args.num_train != 3200:
        for flag, value in (("--num-train", args.num_train), ("--checkpoint-name", args.checkpoint_name),
                            ("--execution-contract-label", args.execution_label)):
            train_argv[train_argv.index(flag)+1] = str(value)
    if contract.get('antenna_selection'):
        for flag, value in (('--num-tx', contract['response_shape'][1]), ('--num-rx', contract['response_shape'][2]),
                            ('--checkpoint-name', args.checkpoint_name), ('--execution-contract-label', args.execution_label)):
            train_argv[train_argv.index(flag)+1] = str(value)
    observer_class = args.observer_type
    final_gates = dict(observer_schema=observer_class.schema, execution_label=args.execution_label)

    if args.report_only:
        final_checkpoint = run_root / "checkpoint_final.pth.tar"
        checkpoint = rift_train.load_tensor_checkpoint(final_checkpoint, map_location="cpu")
        if args.dataset_identity:
            rift_train._validate_saved_sealed_npz_protocol_contract(
                checkpoint.get("sealed_npz_protocol_contract"), contract)
        observer_state = _validate_final_checkpoint(checkpoint, final_checkpoint, **final_gates)
        report_path = run_root / "adaptive_fullscale_report.json"
        if report_path.exists():
            # A prior passing report is terminal evidence.  Reuse it read-only
            # so this route can publish only the missing postflight artifact.
            report = _load_passing_report(report_path, final_checkpoint, observer_schema=observer_class.schema)
            print("RIFT_B7873200_ADAPTIVE_FULLSCALE_REPORT_REUSED_READ_ONLY")
        else:
            report = observer_class.report_from_checkpoint_state(
                observer_state, final_checkpoint
            )
            _attach_report_context(
                report,
                args=args,
                run_root=run_root,
                contract=contract,
                train_argv=train_argv,
                report_only=True,
            )
            report_path = write_fullscale_report(report_path, report)
        print(f"RIFT_B7873200_ADAPTIVE_FULLSCALE_REPORT: {report_path}")
        if report.get("pass") is not True:
            raise RuntimeError("completed checkpoint's adaptive full-scale evidence did not pass")
        print("RIFT_B7873200_ADAPTIVE_FULLSCALE_REPORT_ONLY_PASS")
        return 0

    holder: dict[str, AdaptiveFullScaleObserver] = {}

    def observer_factory(**context: Any) -> AdaptiveFullScaleObserver:
        observer = observer_class(
            model=context["model"],
            optimizer=context["optimizer"],
            gain=context["gain"],
            train_loader=context["train_loader"],
            validation_loader=context["validation_loader"],
            device=context["device"],
            num_freq_selected=context["num_freq_selected"],
            phase_sign=context["phase_sign"],
            forward_operator_name=context["forward_operator_name"],
            compute_dtype=context["compute_dtype"],
            data_format=context["data_format"],
            op_kwargs=context["op_kwargs"],
            occlusion=context["occlusion"],
            expected_train_ids=train_ids,
            expected_validation_ids=validation_ids,
            # The report/recovery contract names the terminal checkpoint file,
            # not merely the run directory.  The file is created by train.py
            # after the observer is constructed, so passing its deterministic
            # path here is intentional.
            checkpoint_path=run_root / "checkpoint_final.pth.tar",
        )
        if args.resume is not None:
            checkpoint = rift_train.load_tensor_checkpoint(args.resume, map_location="cpu")
            observer_state = checkpoint.get("adaptive_event_observer_state")
            if observer_state is None:
                raise ValueError("resume checkpoint lacks adaptive full-scale observer state")
            observer.restore_checkpoint_state(observer_state)
        holder["observer"] = observer
        return observer

    rift_train.main(train_argv, adaptive_event_observer_factory=observer_factory)

    observer = holder.get("observer")
    if observer is None:
        raise RuntimeError("adaptive full-scale trainer did not construct its observer")
    final_checkpoint = run_root / "checkpoint_final.pth.tar"
    if not final_checkpoint.is_file():
        raise RuntimeError("adaptive full-scale trainer returned without checkpoint_final.pth.tar")
    checkpoint = rift_train.load_tensor_checkpoint(final_checkpoint, map_location="cpu")
    _validate_final_checkpoint(checkpoint, final_checkpoint, **final_gates)

    report = observer.finalize()
    _attach_report_context(
        report,
        args=args,
        run_root=run_root,
        contract=contract,
        train_argv=train_argv,
        report_only=False,
    )
    report_path = write_fullscale_report(run_root / "adaptive_fullscale_report.json", report)
    print(f"RIFT_B7873200_ADAPTIVE_FULLSCALE_REPORT: {report_path}")
    if report.get("pass") is not True:
        raise RuntimeError("adaptive full-scale technical observer checks did not pass")
    print("RIFT_B7873200_ADAPTIVE_FULLSCALE_OBSERVER_PASS")
    return 0
