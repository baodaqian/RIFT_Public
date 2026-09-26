#!/usr/bin/env python3
"""Run the proposed full B7873200 Sugavanam--Ertin package.

The driver is deliberately a fresh orchestration identity.  It does not
change the corrected Stage-1/Stage-2 implementations, and it never opens the
raw archive during Stage 2: only the validated terminal Stage-1 bundle crosses
that boundary.  A topology failure after a technically complete trajectory is
reported as a scientific-negative result with a truthful provisional mesh;
the complete-package gate is never weakened or bypassed.

The explicit ``--compute-capped-stage1-epoch30`` lane is a separate,
nonterminal Stage-1 evidence identity.  It preserves the literal 150-epoch
trainer contract, resumes only a validated clean latest, and never enters
bundle publication or Stage 2.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np
import torch

import train as rift_train
import train_sugavanam_ertin_smoke as smoke_driver
import train_sugavanam_ertin_stage2 as stage2
from rift.sugavanam_ertin_b7873200_full import (
    B787_3200_CANONICAL_MANIFEST_PATH,
    B787_3200_CANONICAL_NPZ_PATH,
    FULL_MEMORY_LIMIT_BYTES,
    FULL_REPORT_NAME,
    FULL_RESOURCE_ENVELOPE,
    FULL_ROOT_PARENT,
    FULL_RUN_NAME,
    FULL_SCHEMA,
    FULL_STAGE1_BUNDLE_FILENAME,
    FULL_STAGE1_CHECKPOINT_NAME,
    FULL_STAGE1_COMPUTE_CAP_CHECKPOINT_NAME,
    FULL_STAGE1_COMPUTE_CAP_EPOCH,
    FULL_STAGE1_COMPUTE_CAP_REPORT_NAME,
    FULL_STAGE1_COMPUTE_CAP_RESOURCE_NAME,
    FULL_STAGE1_COMPUTE_CAP_RUN_NAME,
    FULL_STAGE1_CONTRACT_LABEL,
    FULL_STAGE2_ARTIFACT,
    FULL_STAGE2_CAMPAIGN,
    FULL_STAGE2_POLICY,
    FULL_WALL_LIMIT_SECONDS,
    full_stage2_recipe,
)
from rift.sugavanam_ertin_b7873200_stage1 import (
    B787_3200_NUM_TEST,
    B787_3200_NUM_TRAIN,
    B787_3200_NUM_UNUSED,
    B787_3200_NUM_VALIDATION,
    Stage1ContractError,
    atomic_save_stage1_bundle,
    build_b7873200_acquisition_identity,
    build_stage1_final_bundle,
    default_stage1_recipe,
    expected_generic_execution_contract,
    load_b7873200_sealed_identity,
    load_b7873200_stage1_cloud,
    validate_b7873200_stage1_final,
)
from rift.sugavanam_ertin_stage2_runtime_v1 import (
    CLEAN_INTERRUPTION_EXIT_CODE,
    LifecyclePhase,
    atomic_json_dump,
    install_stop_handlers,
    load_json_mapping,
    reset_stop_request,
    stop_requested,
)


GOTCHA_BACKEND = {
    "schema": "rift_gotcha_backend_v1",
    "method": "sugavanam_ertin",
    "callable": "run_gotcha",
    "selection_unit": "pass_sector",
    "joint_passes": True,
    "native_frequency_policy": "ragged_exact",
    "polarizations": ["hh"],
    "metric_domain": "stage1_native_complex_diagnostic_stage2_SDF_geometry",
    "fidelity_status": "published_initialization_unresolved_not_benchmark_ready",
}


def run_gotcha(*, dataset, output_dir, config, device, resume):
    """New equation-based recipe; historical full/smoke contracts are unchanged."""
    from rift.sugavanam_ertin_paper_workflow import run_gotcha as run_paper_gotcha
    return run_paper_gotcha(dataset=dataset, output_dir=output_dir, config=config,
                            device=device, resume=resume)


class FullContractError(ValueError):
    """Raised when the full package identity or lifecycle is ambiguous."""


TIMEOUT_RECOVERY_SOURCE_JOB_ID = "12995517"
TIMEOUT_RECOVERY_SOURCE_EPOCH = 16


def _compute_cap_stop_reason(
    *, epoch: int, cap_epoch: int, signal_requested: bool
) -> str | None:
    """Prioritize the exact durable cap over a coincident Slurm TERM."""

    if int(epoch) == int(cap_epoch):
        return "compute_cap"
    if signal_requested:
        return "signal"
    return None


@dataclass(frozen=True)
class FullStage2Contract:
    """Private Stage-2 output contract for this full-data run only."""

    output_dir: str
    stage1_final_bundle: str

    @property
    def latest_checkpoint(self) -> str:
        return str(Path(self.output_dir) / "checkpoint_latest.pth.tar")

    @property
    def complete_dir(self) -> str:
        return str(Path(self.output_dir) / "complete")

    @property
    def lifecycle_path(self) -> str:
        return str(Path(self.output_dir) / "lifecycle.json")


class FullStage1Observer(smoke_driver.Stage1EngineeringObserver):
    """Readout-only full observer; avoids copying every full-grid update."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.logical_optimizer_updates_final: int | None = None
        self.recovery_mode = False
        self.recovery_stop_requested = False
        self.resume_start_readout: dict[str, object] | None = None
        self.original_readout: dict[str, object] | None = None
        self.original_readout_status = "not_started"
        self.evidence_path = Path(str(kwargs["checkpoint_path"])).parent / "full_stage1_native_readouts.json"

    def _persist_readout_evidence(self, current: Mapping[str, object]) -> None:
        self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        existing: Mapping[str, object] | None = None
        if self.evidence_path.is_file():
            try:
                loaded = json.loads(self.evidence_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise FullContractError("Stage-1 native-readout evidence is unreadable") from exc
            if not isinstance(loaded, Mapping):
                raise FullContractError("Stage-1 native-readout evidence is not an object")
            existing = loaded
        if existing is not None and isinstance(existing.get("original_first_attempt"), Mapping):
            self.original_readout = copy.deepcopy(dict(existing["original_first_attempt"]))
            self.original_readout_status = "preserved_first_attempt"
        else:
            self.original_readout = copy.deepcopy(dict(current))
            self.original_readout_status = (
                "recovered_without_first_attempt_evidence"
                if self.recovery_mode else "captured_first_attempt"
            )
        atomic_json_dump(
            {
                "schema": "rift_sugavanam_ertin_b7873200_full_stage1_native_readouts_v1",
                "original_first_attempt": copy.deepcopy(self.original_readout),
                "latest_resume_start": copy.deepcopy(dict(current)),
                "original_readout_status": self.original_readout_status,
            },
            self.evidence_path,
        )

    def on_training_start(
        self, *, model: object, optimizer: object, gain: object | None,
        start_epoch: int, logical_optimizer_updates: int,
    ) -> None:
        super().on_training_start(
            model=model,
            optimizer=optimizer,
            gain=gain,
            start_epoch=start_epoch,
            logical_optimizer_updates=logical_optimizer_updates,
        )
        if self.initial_readout is None:
            raise FullContractError("Stage-1 native start readout was not produced")
        self.resume_start_readout = copy.deepcopy(self.initial_readout)
        self._persist_readout_evidence(self.resume_start_readout)
        if stop_requested() and not self.recovery_mode:
            raise FullContractError("termination arrived before a Stage-1 recovery boundary")
        if stop_requested() and self.recovery_mode:
            self.recovery_stop_requested = True

    def before_optimizer_step(self, *, model: object, optimizer: object) -> None:
        del model, optimizer

    def on_optimizer_step(
        self, *, model: object, optimizer: object, epoch: int,
        logical_optimizer_updates: int, grad_norm: float,
    ) -> None:
        del model, optimizer, epoch, grad_norm
        self.logical_optimizer_updates_final = int(logical_optimizer_updates)

    def on_epoch_end(
        self, *, model: object, optimizer: object, epoch: int,
        num_epochs: int, val_metrics: Mapping[str, object] | None,
    ) -> None:
        super().on_epoch_end(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            num_epochs=num_epochs,
            val_metrics=val_metrics,
        )
        if stop_requested():
            # train.py has written checkpoint_latest (or checkpoint_final on
            # the terminal epoch) immediately before this observer callback.
            # Raising here converts the Slurm warning into a cooperative,
            # checkpointed Stage-1 boundary without changing optimizer state.
            raise SystemExit(CLEAN_INTERRUPTION_EXIT_CODE)

    def result(self) -> dict[str, object]:
        if self.initial_readout is None or self.final_readout is None:
            raise FullContractError("full Stage-1 observer lacks initial/final native readouts")
        if self.logical_optimizer_updates_final is None and not self.recovery_mode:
            raise FullContractError("full Stage-1 observer saw no optimizer update")
        original = self.original_readout or self.initial_readout
        return {
            "schema": "rift_sugavanam_ertin_b7873200_full_stage1_observation_v1",
            "initial": copy.deepcopy(original),
            "original_first_attempt": copy.deepcopy(original),
            "original_readout_status": self.original_readout_status,
            "resume_start": copy.deepcopy(self.resume_start_readout or self.initial_readout),
            "final": copy.deepcopy(self.final_readout),
            "terminal_readout": copy.deepcopy(self.final_readout),
            "observed_optimizer_steps": int(self.logical_optimizer_updates_final or 0),
            "recovery_without_optimizer_updates": bool(self.recovery_mode),
            "per_step_tensor_sampling": "disabled_for_full_grid; initial_final_readouts_only",
        }


class FullStage1RecoveryObserver(FullStage1Observer):
    """Read a terminal generic final without taking any optimizer update."""

    def on_training_start(
        self, *, model: object, optimizer: object, gain: object | None,
        start_epoch: int, logical_optimizer_updates: int,
    ) -> None:
        self.recovery_mode = True
        super().on_training_start(
            model=model,
            optimizer=optimizer,
            gain=gain,
            start_epoch=start_epoch,
            logical_optimizer_updates=logical_optimizer_updates,
        )
        self.final_readout = copy.deepcopy(self.initial_readout)
        self.logical_optimizer_updates_final = 0


class FullStage1ComputeCapObserver(FullStage1Observer):
    """Stop at a durable nonterminal epoch while preserving the 150-epoch recipe."""

    def __init__(self, *, compute_cap_epoch: int, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.compute_cap_epoch = int(compute_cap_epoch)
        self.cap_reached = False

    def on_epoch_end(
        self, *, model: object, optimizer: object, epoch: int,
        num_epochs: int, val_metrics: Mapping[str, object] | None,
    ) -> None:
        # Call the base engineering observer directly so its bookkeeping is
        # retained, but decide the exact cap before FullStage1Observer's
        # generic stop_requested() branch.  A TERM arriving at epoch 30 is
        # therefore terminal cap evidence, not an impossible epoch-30 resume.
        smoke_driver.Stage1EngineeringObserver.on_epoch_end(
            self,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            num_epochs=num_epochs,
            val_metrics=val_metrics,
        )
        reason = _compute_cap_stop_reason(
            epoch=epoch,
            cap_epoch=self.compute_cap_epoch,
            signal_requested=stop_requested(),
        )
        if reason == "compute_cap":
            # train.py has atomically written checkpoint_latest immediately
            # before this callback.  Capture signal readout evidence, then
            # stop with the existing cooperative interruption code; the
            # 150-epoch command and optimizer/scheduler state remain intact.
            self.final_readout = self._readout(model, self._gain)
            self.cap_reached = True
            raise SystemExit(CLEAN_INTERRUPTION_EXIT_CODE)
        if reason == "signal":
            raise SystemExit(CLEAN_INTERRUPTION_EXIT_CODE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=["legacy-full", "paper-v1"], default="legacy-full",
                        help="paper-v1: sub-aperture reconstruction + equation-based SDF; use --recipe paper-v1 --help")
    parser.add_argument("--checkpoint-root", default=FULL_ROOT_PARENT)
    parser.add_argument("--npz-path", default=B787_3200_CANONICAL_NPZ_PATH)
    parser.add_argument("--parent-role-manifest", default=B787_3200_CANONICAL_MANIFEST_PATH)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--compute-capped-stage1-epoch30",
        action="store_true",
        help="resume the validated Stage-1 latest and stop after epoch 30 without final/bundle/Stage-2",
    )
    parser.add_argument(
        "--compute-capped-stage1-source-checkpoint",
        default=None,
        help="exact pre-existing checkpoint_latest used as the epoch-30 cap source",
    )
    parser.add_argument(
        "--recover-timeout12995517",
        action="store_true",
        help="recover only the exact epoch-16 checkpoint left by Slurm job 12995517",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _same_path(left: str | os.PathLike[str], right: str | os.PathLike[str]) -> bool:
    return os.path.realpath(os.path.abspath(os.fspath(left))) == os.path.realpath(
        os.path.abspath(os.fspath(right))
    )


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    parent = Path(os.path.abspath(os.fspath(args.checkpoint_root)))
    compute_cap = bool(getattr(args, "compute_capped_stage1_epoch30", False))
    run_name = FULL_STAGE1_COMPUTE_CAP_RUN_NAME if compute_cap else FULL_RUN_NAME
    from rift.rift_dataset import collection_manifest
    shared_dataset = collection_manifest(args.parent_role_manifest)
    if shared_dataset:
        if compute_cap or getattr(args, "recover_timeout12995517", False):
            raise FullContractError("Historical job-recovery modes do not apply to the RIFT dataset")
        run_name = "rift_dataset_se_full_v1"
    checkpoint_name = (
        FULL_STAGE1_COMPUTE_CAP_CHECKPOINT_NAME if compute_cap else FULL_STAGE1_CHECKPOINT_NAME
    )
    root = parent / run_name
    stage1_root = root / "stage1"
    stage1_generic = stage1_root / checkpoint_name
    stage2_root = root / "stage2"
    source_stage1_latest = None
    if compute_cap:
        source_arg = getattr(args, "compute_capped_stage1_source_checkpoint", None)
        source_stage1_latest = Path(os.path.abspath(os.fspath(source_arg))) if source_arg else (
            parent / FULL_RUN_NAME / "stage1" / FULL_STAGE1_CHECKPOINT_NAME
            / "checkpoint_latest.pth.tar"
        )
    source_stage1_clean = (
        None if source_stage1_latest is None
        else source_stage1_latest.parent / "full_stage1_clean_interruption.json"
    )
    return {
        "root": root,
        "stage1_root": stage1_root,
        "stage1_generic": stage1_generic,
        "stage1_latest": stage1_generic / "checkpoint_latest.pth.tar",
        "stage1_final": stage1_generic / "checkpoint_final.pth.tar",
        "stage1_clean": stage1_generic / "full_stage1_clean_interruption.json",
        "stage1_evidence": stage1_generic / "full_stage1_native_readouts.json",
        "stage1_bundle": stage1_root / FULL_STAGE1_BUNDLE_FILENAME,
        "stage2_root": stage2_root,
        "stage2_latest": stage2_root / "checkpoint_latest.pth.tar",
        "stage2_entry_clean": root / "stage2_entry_clean_interruption.json",
        "report": root / (FULL_STAGE1_COMPUTE_CAP_REPORT_NAME if compute_cap else FULL_REPORT_NAME),
        "resource": root / (
            FULL_STAGE1_COMPUTE_CAP_RESOURCE_NAME if compute_cap else "resource_accounting.json"
        ),
        "compute_cap": compute_cap,
        "timeout_recovery": bool(getattr(args, "recover_timeout12995517", False)),
        "compute_cap_epoch": FULL_STAGE1_COMPUTE_CAP_EPOCH if compute_cap else None,
        "source_stage1_latest": source_stage1_latest,
        "source_stage1_clean": source_stage1_clean,
        "compute_cap_source": None,
    }


def _validate_stage1_clean_record(paths: Mapping[str, Path]) -> Mapping[str, object]:
    if not paths["stage1_clean"].is_file():
        raise FullContractError("Stage-1 clean resume record is missing")
    try:
        record = json.loads(paths["stage1_clean"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FullContractError("Stage-1 clean resume record is unreadable") from exc
    if not isinstance(record, Mapping) or record.get("state") != "clean_interrupted":
        raise FullContractError("Stage-1 resume record is not a clean interruption")
    if record.get("resume_allowed") is not True:
        raise FullContractError("Stage-1 clean resume record is not resumable")
    checkpoint_path = record.get("checkpoint_path")
    if not isinstance(checkpoint_path, str):
        raise FullContractError("Stage-1 clean resume record lacks its checkpoint path")
    if not (_same_path(checkpoint_path, paths["stage1_latest"]) or _same_path(checkpoint_path, paths["stage1_final"])):
        raise FullContractError("Stage-1 clean resume record names the wrong checkpoint")
    if not Path(checkpoint_path).is_file():
        raise FullContractError("Stage-1 clean resume checkpoint is absent")
    return record


def _validate_compute_cap_source(paths: Mapping[str, Path]) -> Mapping[str, object]:
    """Validate the pre-existing clean latest used by the epoch-30 cap."""

    latest = paths.get("source_stage1_latest")
    clean = paths.get("source_stage1_clean")
    if latest is None or clean is None or not latest.is_file() or not clean.is_file():
        raise FullContractError("compute-capped Stage-1 requires source latest plus clean record")
    try:
        record = json.loads(clean.read_text(encoding="utf-8"))
        checkpoint = torch.load(latest, map_location="cpu", weights_only=True)
    except (OSError, json.JSONDecodeError, RuntimeError, TypeError, ValueError) as exc:
        raise FullContractError("compute-capped Stage-1 source evidence is unreadable") from exc
    if (
        not isinstance(record, Mapping)
        or record.get("state") != "clean_interrupted"
        or record.get("resume_allowed") is not True
        or not isinstance(record.get("checkpoint_path"), str)
        or not _same_path(record["checkpoint_path"], latest)
    ):
        raise FullContractError("compute-capped Stage-1 source latest lacks a matching clean record")
    if not isinstance(checkpoint, Mapping):
        raise FullContractError("compute-capped Stage-1 source checkpoint is not a mapping")
    source_epoch = checkpoint.get("epoch")
    if not isinstance(source_epoch, int) or not (0 < source_epoch < FULL_STAGE1_COMPUTE_CAP_EPOCH):
        raise FullContractError(
            f"compute-capped Stage-1 source must be a completed epoch before {FULL_STAGE1_COMPUTE_CAP_EPOCH}"
        )
    origin_epoch = record.get("origin_epoch", source_epoch)
    if not isinstance(origin_epoch, int) or not (0 < origin_epoch <= source_epoch):
        raise FullContractError("compute-capped source origin epoch is invalid")
    origin_checkpoint = record.get("origin_checkpoint_path", str(latest))
    origin_resource = record.get(
        "origin_resource_reference",
        str(latest.parents[2] / "resource_accounting.json"),
    )
    if not isinstance(origin_checkpoint, str) or not isinstance(origin_resource, str):
        raise FullContractError("compute-capped source origin evidence is invalid")
    return {
        "record": dict(record),
        "source_epoch": int(source_epoch),
        "checkpoint_path": str(latest),
        "origin_epoch": int(origin_epoch),
        "origin_checkpoint_path": origin_checkpoint,
        "origin_resource_reference": origin_resource,
    }


def _validate_timeout_recovery_checkpoint(
    args: argparse.Namespace | None,
    paths: Mapping[str, Path],
    *,
    expected_contract: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Validate the exact source checkpoint from Slurm job 12995517.

    The continuation runs under a new allocation.  This path accepts only the
    durable epoch-16 cap checkpoint and carries the source provenance in
    memory; it never writes or synthesizes the source record.
    """

    if not paths.get("timeout_recovery"):
        raise FullContractError("timeout recovery validation was called outside its explicit mode")
    if os.environ.get("SLURM_ARRAY_TASK_ID"):
        raise FullContractError("timeout recovery does not accept array allocations")
    if not paths.get("compute_cap"):
        raise FullContractError("timeout recovery requires the compute-capped Stage-1 identity")

    latest = paths["stage1_latest"]
    source = paths.get("source_stage1_latest")
    if source is None or not _same_path(source, latest):
        raise FullContractError("timeout recovery source must be the exact cap latest")
    if args is not None and (args.resume is None or not _same_path(args.resume, latest)):
        raise FullContractError("timeout recovery resume must name the exact cap latest")
    if paths["stage1_clean"].exists():
        raise FullContractError("timeout recovery refuses an existing cap clean record")
    if paths["stage1_final"].exists() or paths["stage1_bundle"].exists():
        raise FullContractError("timeout recovery cannot coexist with a final checkpoint or bundle")
    if paths["stage2_root"].exists() and any(paths["stage2_root"].iterdir()):
        raise FullContractError("timeout recovery cannot coexist with Stage-2 artifacts")
    if not latest.is_file():
        raise FullContractError("timeout recovery source checkpoint is absent")

    try:
        checkpoint = torch.load(latest, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise FullContractError("timeout recovery checkpoint is unreadable") from exc
    if not isinstance(checkpoint, Mapping):
        raise FullContractError("timeout recovery checkpoint is not a mapping")
    if checkpoint.get("epoch") != TIMEOUT_RECOVERY_SOURCE_EPOCH:
        raise FullContractError("timeout recovery checkpoint is not exactly epoch 16")
    for key in ("optimizer_state_dict", "scheduler_state_dict", "rng_state"):
        value = checkpoint.get(key)
        if not isinstance(value, Mapping) or not value:
            raise FullContractError(f"timeout recovery checkpoint lacks required {key}")
    rng_state = checkpoint["rng_state"]
    if not {"python", "numpy", "torch_cpu", "freq_cpu"}.issubset(rng_state):
        raise FullContractError("timeout recovery checkpoint RNG state is incomplete")
    contract = checkpoint.get("execution_contract")
    if not isinstance(contract, Mapping):
        raise FullContractError("timeout recovery checkpoint lacks its execution contract")
    if (
        contract.get("schema") != "rift_checkpoint_execution_contract_v1"
        or contract.get("label") != FULL_STAGE1_CONTRACT_LABEL
    ):
        raise FullContractError("timeout recovery checkpoint execution contract identity differs")
    if expected_contract is not None and contract != expected_contract:
        raise FullContractError("timeout recovery checkpoint execution contract differs")
    return {
        "record": None,
        "source_epoch": TIMEOUT_RECOVERY_SOURCE_EPOCH,
        "checkpoint_path": str(latest),
        "origin_epoch": TIMEOUT_RECOVERY_SOURCE_EPOCH,
        "origin_checkpoint_path": str(latest),
        "origin_resource_reference": (
            "slurm_job_12995517_TIMEOUT_after_durable_epoch_16_checkpoint"
        ),
        "source_job_id": TIMEOUT_RECOVERY_SOURCE_JOB_ID,
        "source_reason": "Slurm job 12995517 timed out after durable checkpoint_latest write",
        "clean_record_present": False,
    }


def _validate_stage2_clean_record(paths: Mapping[str, Path]) -> Mapping[str, object]:
    lifecycle_path = paths["stage2_root"] / "lifecycle.json"
    if not paths["stage2_latest"].is_file() or not lifecycle_path.is_file():
        raise FullContractError("Stage-2 clean resume requires latest checkpoint and lifecycle")
    try:
        record = json.loads(lifecycle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FullContractError("Stage-2 lifecycle is unreadable") from exc
    if not isinstance(record, Mapping):
        raise FullContractError("Stage-2 lifecycle is not an object")
    if record.get("state") != "clean_interrupted" or record.get("resume_allowed") is not True:
        raise FullContractError("Stage-2 lifecycle is not a clean resumable state")
    checkpoint_path = record.get("checkpoint_path")
    if not isinstance(checkpoint_path, str) or not _same_path(checkpoint_path, paths["stage2_latest"]):
        raise FullContractError("Stage-2 lifecycle names the wrong resume checkpoint")
    return record


def _validate_stage2_entry_clean_record(paths: Mapping[str, Path]) -> Mapping[str, object]:
    if not paths["stage2_entry_clean"].is_file():
        raise FullContractError("Stage-2 entry clean record is missing")
    try:
        record = json.loads(paths["stage2_entry_clean"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FullContractError("Stage-2 entry clean record is unreadable") from exc
    if not isinstance(record, Mapping):
        raise FullContractError("Stage-2 entry clean record is not an object")
    if (
        record.get("schema") != "rift_sugavanam_ertin_b7873200_full_stage2_entry_v1"
        or record.get("phase") != "stage1_terminal_recovery"
        or record.get("state") != "clean_interrupted"
        or record.get("resume_allowed") is not True
    ):
        raise FullContractError("Stage-2 entry record is not a clean resumable state")
    bundle_path = record.get("resume_target")
    checkpoint_path = record.get("checkpoint_path")
    if not isinstance(bundle_path, str) or not _same_path(bundle_path, paths["stage1_bundle"]):
        raise FullContractError("Stage-2 entry record names the wrong bundle continuation")
    if not isinstance(checkpoint_path, str) or not _same_path(checkpoint_path, paths["stage1_bundle"]):
        raise FullContractError("Stage-2 entry record names the wrong checkpoint")
    if not paths["stage1_bundle"].is_file():
        raise FullContractError("Stage-2 entry record lacks its validated bundle")
    if paths["stage2_root"].exists() and any(paths["stage2_root"].iterdir()):
        raise FullContractError("Stage-2 entry continuation cannot coexist with Stage-2 artifacts")
    return record


def _validate_identity(args: argparse.Namespace, paths: Mapping[str, Path]) -> None:
    from rift.rift_dataset import collection_manifest
    shared_dataset = collection_manifest(args.parent_role_manifest)
    if not shared_dataset and not _same_path(args.npz_path, B787_3200_CANONICAL_NPZ_PATH):
        raise FullContractError("full package accepts only canonical sphere10k NPZ")
    if not shared_dataset and not _same_path(args.parent_role_manifest, B787_3200_CANONICAL_MANIFEST_PATH):
        raise FullContractError("full package accepts only the canonical 3200/1000 manifest")
    if args.device != "cuda":
        raise FullContractError("the full package is an Inferno CUDA lane; use --device cuda")
    if getattr(args, "recover_timeout12995517", False) and not paths.get("compute_cap"):
        raise FullContractError("timeout recovery requires the compute-capped Stage-1 identity")
    if paths.get("compute_cap"):
        if not getattr(args, "compute_capped_stage1_epoch30", False):
            raise FullContractError("compute-capped output identity requires its explicit epoch-30 mode")
        if paths.get("timeout_recovery"):
            paths["compute_cap_source"] = _validate_timeout_recovery_checkpoint(args, paths)
            return
        if args.resume is None or not _same_path(args.resume, paths["source_stage1_latest"]):
            raise FullContractError("compute-capped Stage-1 requires --resume at the source latest checkpoint")
        source_is_cap_output = _same_path(args.resume, paths["stage1_latest"])
        if source_is_cap_output:
            if paths["stage1_final"].exists() or paths["stage1_bundle"].exists():
                raise FullContractError("compute-capped Stage-1 cannot resume a terminal output")
            if paths["stage2_root"].exists() and any(paths["stage2_root"].iterdir()):
                raise FullContractError("compute-capped Stage-1 cannot coexist with Stage-2 artifacts")
            if paths["report"].is_file():
                previous = load_json_mapping(paths["report"], "compute-cap report")
                if previous.get("status") != "clean_interruption":
                    raise FullContractError("only an intermediate clean compute-cap report may be resumed")
        elif paths["root"].exists() and any(paths["root"].iterdir()):
            raise FullContractError("initial compute-capped Stage-1 requires a fresh output identity")
        source = _validate_compute_cap_source(paths)
        paths["compute_cap_source"] = source
        return
    resume = None if args.resume is None else Path(os.path.abspath(os.fspath(args.resume)))
    if resume is None:
        if paths["root"].exists() and any(paths["root"].iterdir()):
            raise FullContractError("fresh full run refuses a nonempty run root")
        return
    allowed_resume_paths = (
        paths["stage1_latest"], paths["stage1_final"], paths["stage1_bundle"], paths["stage2_latest"]
    )
    if not any(_same_path(resume, candidate) for candidate in allowed_resume_paths):
        raise FullContractError("--resume must name this full identity's latest, terminal final, bundle, or Stage-2 latest")
    if not resume.is_file():
        raise FullContractError("--resume was supplied but its latest checkpoint is absent")
    if _same_path(resume, paths["stage1_latest"]):
        if paths["stage1_bundle"].exists() or paths["stage1_final"].exists():
            raise FullContractError("Stage-1 latest resume is not valid after a terminal Stage-1 artifact")
        _validate_stage1_clean_record(paths)
    elif _same_path(resume, paths["stage1_final"]):
        if paths["stage1_bundle"].exists() or not paths["stage1_final"].is_file():
            raise FullContractError("terminal Stage-1 recovery requires the unbundled final checkpoint")
        if paths["stage2_root"].exists() and any(paths["stage2_root"].iterdir()):
            raise FullContractError("terminal Stage-1 recovery cannot coexist with Stage-2 artifacts")
    elif _same_path(resume, paths["stage1_bundle"]):
        if not paths["stage1_bundle"].is_file():
            raise FullContractError("bundle continuation requires the validated Stage-1 bundle")
        if paths["stage2_root"].exists() and any(paths["stage2_root"].iterdir()):
            raise FullContractError("bundle continuation cannot restart an existing Stage-2 root")
        if paths["stage2_entry_clean"].is_file():
            _validate_stage2_entry_clean_record(paths)
    else:
        if not paths["stage1_bundle"].is_file():
            raise FullContractError("Stage-2 resume requires the validated Stage-1 final bundle")
        if (paths["stage2_root"] / "terminal_failure.json").exists():
            raise FullContractError("terminal Stage-2 failure is not an automatic resume state")
        if paths["stage2_root"].joinpath("complete").is_dir():
            raise FullContractError("completed or scientific-negative Stage-2 roots are immutable")
        _validate_stage2_clean_record(paths)

    if paths["report"].is_file():
        try:
            previous = json.loads(paths["report"].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FullContractError("existing full report is unreadable") from exc
        if not isinstance(previous, Mapping):
            raise FullContractError("existing full report is not an object")
        if previous.get("status") != "clean_interruption":
            raise FullContractError("only a reported clean interruption may be resumed")
        phase = previous.get("phase")
        if phase == "stage2":
            _validate_stage2_clean_record(paths)
        elif phase in {"stage1", "stage1_terminal_recovery"}:
            if _same_path(resume, paths["stage1_latest"]):
                _validate_stage1_clean_record(paths)
            elif not (_same_path(resume, paths["stage1_final"]) or _same_path(resume, paths["stage1_bundle"])):
                raise FullContractError("clean Stage-1 report requires a Stage-1 continuation target")
        else:
            raise FullContractError("clean report has no recognized resumable phase")


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows source checks
        return None
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _resource_snapshot() -> dict[str, object]:
    result: dict[str, object] = {"process_peak_rss_bytes": _peak_rss_bytes()}
    if torch.cuda.is_available():
        result.update({
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        })
    else:
        result.update({"cuda_peak_allocated_bytes": None, "cuda_peak_reserved_bytes": None})
    return result


def _phase_resource(start: float, *, execution: str) -> dict[str, object]:
    return {
        "execution": execution,
        "wall_seconds": float(time.monotonic() - start),
        **_resource_snapshot(),
    }


def _enforce_resource(record: Mapping[str, object], label: str) -> None:
    wall = float(record["wall_seconds"])
    rss = record.get("process_peak_rss_bytes")
    if not math.isfinite(wall) or wall < 0.0 or wall > FULL_WALL_LIMIT_SECONDS:
        raise FullContractError(f"{label} exceeded the proposed 12-hour allocation envelope")
    if rss is None or not math.isfinite(float(rss)) or float(rss) <= 0.0:
        raise FullContractError(f"{label} lacks a finite process peak-RSS measurement")
    if float(rss) > FULL_MEMORY_LIMIT_BYTES:
        raise FullContractError(f"{label} exceeded the proposed 64 GiB memory envelope")
    for key in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
        value = record.get(key)
        if value is None or not math.isfinite(float(value)) or float(value) < 0.0:
            raise FullContractError(f"{label} lacks a finite CUDA peak measurement: {key}")


_RESOURCE_ATTEMPT: dict[str, object] | None = None


def _begin_resource_attempt(path: Path, *, started: float, resume: str | None) -> None:
    global _RESOURCE_ATTEMPT
    attempts: list[object] = []
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, Mapping) and isinstance(existing.get("attempts"), list):
            attempts = list(existing["attempts"])
    attempt = {
        "attempt_index": len(attempts) + 1,
        "resume_target": resume,
        "started_wall_time": time.time(),
        "status": "running",
        "phase": "preflight",
        "total_wall_seconds": 0.0,
        "phases": {
            "stage1": {"status": "not_started", "execution": "not_started"},
            "stage2": {"status": "not_started", "execution": "not_started"},
        },
    }
    attempts.append(attempt)
    _RESOURCE_ATTEMPT = {"path": path, "started": started, "attempt": attempt, "attempts": attempts}
    _write_resource(path, started=started, status="running", phase="preflight")


def _phase_work_wall(attempts: Sequence[object]) -> float:
    total = 0.0
    for item in attempts:
        if not isinstance(item, Mapping):
            continue
        phases = item.get("phases")
        if not isinstance(phases, Mapping):
            continue
        for record in phases.values():
            if not isinstance(record, Mapping) or record.get("execution") in {"not_started", "reused_terminal_bundle"}:
                continue
            value = record.get("wall_seconds")
            if value is not None:
                total += float(value)
    return total


def _write_resource(path: Path, *, started: float, status: str, phase: str,
                    stage1: Mapping[str, object] | None = None,
                    stage2: Mapping[str, object] | None = None,
                    error: BaseException | None = None) -> None:
    global _RESOURCE_ATTEMPT
    if _RESOURCE_ATTEMPT is None or _RESOURCE_ATTEMPT.get("path") != path:
        _begin_resource_attempt(path, started=started, resume=None)
    attempt = _RESOURCE_ATTEMPT["attempt"]
    attempts = _RESOURCE_ATTEMPT["attempts"]
    if not isinstance(attempt, dict) or not isinstance(attempts, list):
        raise FullContractError("resource attempt ledger is malformed")
    attempt["status"] = status
    attempt["phase"] = phase
    attempt["total_wall_seconds"] = float(time.monotonic() - float(_RESOURCE_ATTEMPT["started"]))
    phases = attempt["phases"]
    if not isinstance(phases, dict):
        raise FullContractError("resource attempt phase ledger is malformed")
    if stage1 is not None:
        phases["stage1"] = dict(stage1)
    if stage2 is not None:
        phases["stage2"] = dict(stage2)
    if error is not None:
        attempt["error"] = {"type": type(error).__name__, "message": str(error)}
    payload: dict[str, object] = {
        "schema": "rift_sugavanam_ertin_b7873200_full_resource_accounting_v2",
        "status": status,
        "phase": phase,
        "declared_envelope": copy.deepcopy(FULL_RESOURCE_ENVELOPE),
        "per_attempt_wall_limit_seconds": FULL_WALL_LIMIT_SECONDS,
        "attempts": attempts,
        "current_attempt_index": attempt["attempt_index"],
        "cumulative_work_wall_seconds": _phase_work_wall(attempts),
        "current_peak": _resource_snapshot(),
    }
    attempt["current_peak"] = payload["current_peak"]
    attempt["cumulative_work_wall_seconds"] = payload["cumulative_work_wall_seconds"]
    if error is not None:
        payload["error"] = {"type": type(error).__name__, "message": str(error)}
    atomic_json_dump(payload, path)


def _stage1_argv(
    paths: Mapping[str, Path], resume: str | None, *, resume_path: str | None = None
) -> list[str]:
    argv = [
        "--data-format", "npz",
        "--npz-path", str(paths.get("npz_path", B787_3200_CANONICAL_NPZ_PATH)),
        "--npz-sealed-protocol",
        "--npz-role-manifest", str(paths.get("role_manifest", B787_3200_CANONICAL_MANIFEST_PATH)),
        "--checkpoint-name", paths["stage1_generic"].name,
        "--checkpoint-root", str(paths["stage1_root"]),
        "--execution-contract-label", FULL_STAGE1_CONTRACT_LABEL,
        "--require-full-resume-state",
        "--num-train", "3200", "--num-val", "1000", "--num-test", "1000",
        "--num-freq-wanted", "600", "--epochs", "150",
        "--loss", "complex", "--scene-repr", "grid",
        "--forward-operator", "range", "--range-model", "product",
        "--compute-dtype", "float64", "--point-chunk", "65536", "--pair-chunk", "64",
        "--extent", "0.15", "--granularity", "48", "--phase-sign", "-1.0",
        "--bp-init", "400", "--lr", "0.003", "--l1-weight", "3e-7",
        "--adam-eps", "1e-20", "--checkpoint-metric", "val", "--t0", "10",
        "--t-mult", "2", "--seed", "42",
        "--prune-every", "0", "--prune-threshold", "0.0", "--prune-criterion", "energy",
        "--prune-start-epoch", "0", "--prune-mode", "mass", "--prune-target-active", "0",
        "--prune-end-epoch", "0", "--prune-min-active", "0",
    ]
    if resume is not None:
        argv.extend(("--resume", str(resume_path or paths["stage1_latest"])))
    return argv


def _validate_compute_cap_boundary(paths: Mapping[str, Path]) -> None:
    """Require an epoch-30 latest and prove no terminal/Stage-2 artifact exists."""

    if not paths["stage1_latest"].is_file():
        raise FullContractError("compute-capped Stage-1 stopped without checkpoint_latest")
    try:
        checkpoint = torch.load(paths["stage1_latest"], map_location="cpu", weights_only=True)
        clean = load_json_mapping(paths["stage1_clean"], "compute-cap clean record")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise FullContractError("compute-capped Stage-1 boundary evidence is unreadable") from exc
    if not isinstance(checkpoint, Mapping) or checkpoint.get("epoch") != FULL_STAGE1_COMPUTE_CAP_EPOCH:
        raise FullContractError("compute-capped Stage-1 latest is not the exact epoch-30 checkpoint")
    if (
        clean.get("state") != "clean_interrupted"
        or clean.get("resume_allowed") is not False
        or clean.get("terminal") is not True
        or clean.get("target_epoch") != FULL_STAGE1_COMPUTE_CAP_EPOCH
        or not _same_path(clean.get("checkpoint_path", ""), paths["stage1_latest"])
    ):
        raise FullContractError("compute-capped Stage-1 latest lacks terminal epoch-30 evidence")
    if paths["stage1_final"].exists() or paths["stage1_bundle"].exists():
        raise FullContractError("compute-capped Stage-1 must not publish final checkpoint or bundle")
    if paths["stage2_root"].exists() and any(paths["stage2_root"].iterdir()):
        raise FullContractError("compute-capped Stage-1 must not create Stage-2 artifacts")


def _stage1_report_summary(recipe: Mapping[str, object]) -> dict[str, object]:
    observation = recipe["observation"]
    scene = recipe["scene"]
    fit = recipe["fit"]
    return {
        "schema": recipe["schema"],
        "recipe_id": recipe["recipe_id"],
        "observation": {
            "all_coordinates": copy.deepcopy(observation["all_coordinates"]),
            "num_freq_wanted": observation["num_freq_wanted"],
            "frequency_policy": observation["frequency_policy"],
            "pair_policy": observation["pair_policy"],
            "forward_operator": observation["forward_operator"],
            "range_model": observation["range_model"],
            "phase_sign": observation["phase_sign"],
            "compute_dtype": observation["compute_dtype"],
            "operator_options": copy.deepcopy(observation["operator_options"]),
        },
        "scene": copy.deepcopy(scene),
        "fit": copy.deepcopy(fit),
        "stage2_cloud_extraction": copy.deepcopy(recipe["stage2_cloud_extraction"]),
        "source_selection": copy.deepcopy(recipe["source_selection"]),
    }


def _load_or_run_stage1(
    paths: Mapping[str, Path],
    recipe: Mapping[str, object],
    normalization: Mapping[str, object],
    *,
    resume: str | None,
) -> tuple[dict[str, object], dict[str, object]]:
    if paths.get("compute_cap"):
        if resume is None or not _same_path(resume, paths["source_stage1_latest"]):
            raise FullContractError("compute-capped Stage-1 resume must name the validated source latest")
        if paths["stage1_final"].exists() or paths["stage1_bundle"].exists():
            raise FullContractError("compute-capped Stage-1 output identity is already terminal")
        if paths.get("timeout_recovery"):
            paths["compute_cap_source"] = _validate_timeout_recovery_checkpoint(
                None, paths, expected_contract=expected_generic_execution_contract(recipe)
            )
        observer_box: dict[str, FullStage1ComputeCapObserver] = {}

        def cap_observer_factory(**kwargs: object) -> FullStage1ComputeCapObserver:
            observer = FullStage1ComputeCapObserver(
                compute_cap_epoch=FULL_STAGE1_COMPUTE_CAP_EPOCH, **kwargs
            )
            observer_box["observer"] = observer
            return observer

        try:
            rift_train.main(
                _stage1_argv(paths, resume, resume_path=str(paths["source_stage1_latest"])),
                engineering_observer_factory=cap_observer_factory,
            )
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            observer = observer_box.get("observer")
            if code == CLEAN_INTERRUPTION_EXIT_CODE and observer is not None:
                source = paths.get("compute_cap_source") or _validate_compute_cap_source(paths)
                cap_reached = bool(observer.cap_reached)
                native = observer.result() if cap_reached else None
                atomic_json_dump(
                    {
                        "schema": "rift_sugavanam_ertin_b7873200_full_stage1_compute_cap_v1",
                        "phase": "stage1_compute_capped",
                        "state": "clean_interrupted",
                        "resume_allowed": not cap_reached,
                        "checkpoint_path": str(paths["stage1_latest"]),
                        "checkpoint_role": (
                            "compute_capped_latest" if cap_reached else "intermediate_latest"
                        ),
                        "source_checkpoint_path": str(paths["source_stage1_latest"]),
                        "source_epoch": int(source["source_epoch"]),
                        "origin_checkpoint_path": str(source["origin_checkpoint_path"]),
                        "origin_epoch": int(source["origin_epoch"]),
                        "origin_resource_reference": str(source["origin_resource_reference"]),
                        "target_epoch": FULL_STAGE1_COMPUTE_CAP_EPOCH,
                        "underlying_recipe_epochs": 150,
                        "terminal": cap_reached,
                        "stage2_started": False,
                        "geometry_metrics": "N/A",
                        "geometry_status": "not_applicable_no_stage2_compute_cap",
                        "signal_readout_label": (
                            "nonterminal_compute_capped" if cap_reached
                            else "clean_interruption_before_compute_cap"
                        ),
                        "native_metrics": native,
                        "reason": (
                            "compute cap raised after durable checkpoint_latest write"
                            if cap_reached else "driver signal reached a durable intermediate checkpoint"
                        ),
                    },
                    paths["stage1_clean"],
                )
                if cap_reached:
                    _validate_compute_cap_boundary(paths)
            raise
        raise FullContractError("compute-capped Stage-1 returned without reaching epoch 30")

    if paths["stage1_bundle"].is_file():
        validated = validate_b7873200_stage1_final(paths["stage1_bundle"], expected_recipe=recipe)
        record = dict(validated["record"])
        provenance = record.get("provenance")
        native = provenance.get("native_metrics") if isinstance(provenance, Mapping) else None
        if not isinstance(native, Mapping):
            raise FullContractError("existing full Stage-1 bundle lacks native initial/final metrics")
        return validated, dict(native)

    def package_final(native: Mapping[str, object], *, evidence_status: str) -> tuple[dict[str, object], dict[str, object]]:
        from rift.sugavanam_ertin_b7873200_stage1 import _torch_load

        generic_state = _torch_load(paths["stage1_final"])
        bundle = build_stage1_final_bundle(
            generic_state,
            recipe,
            provenance={
                "canonical_npz_path": str(paths.get("npz_path", B787_3200_CANONICAL_NPZ_PATH)),
                "role_manifest_path": str(paths.get("role_manifest", B787_3200_CANONICAL_MANIFEST_PATH)),
                "generic_final_path": str(paths["stage1_final"]),
                "bundle_path": str(paths["stage1_bundle"]),
                "full_package_schema": FULL_SCHEMA,
                "reporting_normalization": copy.deepcopy(dict(normalization)),
                "native_metrics": copy.deepcopy(dict(native)),
                "fit_evidence_status": evidence_status,
            },
        )
        atomic_save_stage1_bundle(bundle, paths["stage1_bundle"])
        validated = validate_b7873200_stage1_final(paths["stage1_bundle"], expected_recipe=recipe)
        return validated, dict(native)

    if paths["stage1_final"].exists():
        if resume is None or not _same_path(resume, paths["stage1_final"]):
            raise FullContractError("generic Stage-1 final requires an explicit terminal readout recovery")
        observer_box: dict[str, FullStage1RecoveryObserver] = {}

        def recovery_factory(**kwargs: object) -> FullStage1RecoveryObserver:
            observer = FullStage1RecoveryObserver(**kwargs)
            observer_box["observer"] = observer
            return observer

        rift_train.main(
            _stage1_argv(paths, paths["stage1_final"], resume_path=str(paths["stage1_final"])),
            engineering_observer_factory=recovery_factory,
        )
        observer = observer_box.get("observer")
        if observer is None:
            raise FullContractError("terminal Stage-1 recovery did not construct its observer")
        native = observer.result()
        if native.get("recovery_without_optimizer_updates") is not True or native.get("observed_optimizer_steps") != 0:
            raise FullContractError("terminal Stage-1 recovery took an optimizer update")
        packaged = package_final(native, evidence_status="terminal_final_readout_recovery_without_optimizer_updates")
        if observer.recovery_stop_requested:
            atomic_json_dump(
                {
                    "schema": "rift_sugavanam_ertin_b7873200_full_stage1_recovery_v1",
                    "phase": "stage1_terminal_recovery",
                    "state": "clean_interrupted",
                    "resume_allowed": True,
                    "checkpoint_path": str(paths["stage1_final"]),
                    "checkpoint_role": "terminal_final_recovery",
                    "bundle_path": str(paths["stage1_bundle"]),
                    "reason": "termination arrived during zero-update terminal readout recovery",
                },
                paths["stage1_clean"],
            )
            raise SystemExit(CLEAN_INTERRUPTION_EXIT_CODE)
        return packaged

    if resume is not None and not _same_path(resume, paths["stage1_latest"]):
        raise FullContractError("Stage-1 training resume must name checkpoint_latest")

    observer_box: dict[str, FullStage1Observer] = {}

    def observer_factory(**kwargs: object) -> FullStage1Observer:
        observer = FullStage1Observer(**kwargs)
        observer_box["observer"] = observer
        return observer

    try:
        rift_train.main(
            _stage1_argv(paths, resume),
            engineering_observer_factory=observer_factory,
        )
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if code == CLEAN_INTERRUPTION_EXIT_CODE:
            checkpoint = paths["stage1_final"] if paths["stage1_final"].is_file() else paths["stage1_latest"]
            if not checkpoint.is_file():
                raise FullContractError("Stage-1 termination arrived without a recovery checkpoint") from exc
            atomic_json_dump(
                {
                    "schema": "rift_sugavanam_ertin_b7873200_full_stage1_recovery_v1",
                    "phase": "stage1",
                    "state": "clean_interrupted",
                    "resume_allowed": True,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_role": "terminal_final_recovery" if checkpoint == paths["stage1_final"] else "latest",
                    "reason": "driver signal handler reached the post-checkpoint observer boundary",
                },
                paths["stage1_clean"],
            )
        raise
    if not paths["stage1_final"].is_file():
        raise FullContractError("full Stage-1 trainer returned without checkpoint_final.pth.tar")
    observer = observer_box.get("observer")
    if observer is None:
        raise FullContractError("full Stage-1 trainer did not construct its observer")
    native = observer.result()
    return package_final(native, evidence_status="native_initial_final_readouts_full_train_and_validation")


def _stage2_provenance(
    source: Mapping[str, object],
    recipe: Mapping[str, object],
    paths: Mapping[str, Path],
    normalization: Mapping[str, object],
) -> dict[str, object]:
    dataset_identity = source["stage1_record"].get("sealed_protocol_identity", {}).get("dataset_identity")
    contract = {
        "campaign_identity": FULL_STAGE2_CAMPAIGN,
        "artifact_identity": FULL_STAGE2_ARTIFACT,
        "method": ("Sugavanam--Ertin RIFT dataset full two-stage package" if dataset_identity
                   else "Sugavanam--Ertin B7873200 proposed full two-stage package"),
        **({"dataset_identity": dataset_identity} if dataset_identity else {}),
        "policy_identity": FULL_STAGE2_POLICY,
        "implementation_kind": "corrected Stage-2 v1 implementation through explicit full-data seam",
        "canonical_npz_path": str(paths.get("npz_path", B787_3200_CANONICAL_NPZ_PATH)),
        "stage1_final_bundle_filename": FULL_STAGE1_BUNDLE_FILENAME,
        "stage1_final_bundle_path": str(paths["stage1_bundle"]),
        "output_dir": str(paths["stage2_root"]),
        "ground_truth_geometry_used": False,
        "novel_view_signal_supported": False,
    }
    return {
        "contract": contract,
        "stage1_record": copy.deepcopy(dict(source["stage1_record"])),
        "stage2_recipe": copy.deepcopy(dict(recipe)),
        "reporting_normalization": copy.deepcopy(dict(normalization)),
        "ground_truth_geometry_used": False,
        "novel_view_signal_supported": False,
    }


def _validate_stage2_returned_clean(
    contract: FullStage2Contract,
    provenance: Mapping[str, object],
    recipe: Mapping[str, object],
) -> None:
    lifecycle_contract = stage2.lifecycle_contract_record(provenance)
    record = stage2.load_json_mapping(contract.lifecycle_path, "full Stage-2 lifecycle")
    checked = stage2.validate_lifecycle_record(
        record,
        expected_contract=lifecycle_contract,
        max_steps=int(recipe["steps"]),
    )
    if checked.get("state") != "clean_interrupted" or checked.get("resume_allowed") is not True:
        raise FullContractError("Stage-2 returned 143 without a clean resumable lifecycle")
    if not _same_path(checked.get("checkpoint_path", ""), contract.latest_checkpoint):
        raise FullContractError("Stage-2 returned 143 with the wrong latest checkpoint path")


def _write_provisional_geometry(
    *,
    source: Mapping[str, object],
    recipe: Mapping[str, object],
    contract: FullStage2Contract,
    device: torch.device,
) -> dict[str, object]:
    latest = Path(contract.latest_checkpoint)
    if not latest.is_file():
        raise FullContractError("topology-negative Stage-2 run lacks its latest checkpoint")
    state = stage2.load_torch_mapping(latest, "topology-negative latest checkpoint")
    geometry = stage2._stage1_geometry(source, recipe, device)
    model = stage2._model(recipe, float(geometry["extent"]), device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    field, pitch = stage2.evaluate_field_grid(
        model, geometry["extent"], int(recipe["mesh_grid"]), device, int(recipe["grid_chunk"])
    )
    validity = stage2.field_validity_from_array(field, float(recipe["boundary_margin"]))
    from skimage.measure import marching_cubes

    vertices, faces, _normals, _values = marching_cubes(
        field, level=0.0, spacing=(pitch, pitch, pitch)
    )
    vertices = vertices.astype(np.float32) - float(geometry["extent"])
    faces = faces.astype(np.int32)
    topology = stage2.topology_contract(
        vertices,
        faces,
        float(geometry["extent"]),
        protected_shell_depth=2.0 * float(geometry["pitch"]),
    )
    destination = Path(contract.output_dir) / "provisional_geometry"
    destination.mkdir(parents=True, exist_ok=False)
    stage2._atomic_npz(
        destination / stage2.SURFACE_NAME,
        vertices=vertices,
        faces=faces,
        field=field.astype(np.float32),
        extent=np.asarray(float(geometry["extent"]), dtype=np.float64),
        validity_json=np.asarray(json.dumps(validity, sort_keys=True, allow_nan=False)),
        topology_json=np.asarray(json.dumps(topology, sort_keys=True, allow_nan=False)),
    )
    audit = {
        "schema": "rift_sugavanam_ertin_b7873200_full_provisional_geometry_audit_v1",
        "status": "scientific_negative_topology",
        "validity": validity,
        "topology": topology,
        "grid_pitch": float(pitch),
        "source_latest_checkpoint": str(latest),
        "complete_package_published": False,
        "production_clearance": False,
    }
    atomic_json_dump(audit, destination / "geometry_audit.json")
    return {
        "directory": str(destination),
        "surface": str(destination / stage2.SURFACE_NAME),
        "audit": audit,
    }


def _report(
    paths: Mapping[str, Path],
    *,
    status: str,
    phase: str,
    started: float,
    normalization: Mapping[str, object] | None = None,
    native_metrics: Mapping[str, object] | None = None,
    stage1_recipe: Mapping[str, object] | None = None,
    stage1_resource: Mapping[str, object] | None = None,
    stage2_recipe: Mapping[str, object] | None = None,
    stage2_resource: Mapping[str, object] | None = None,
    provisional: Mapping[str, object] | None = None,
    terminal_failure: str | None = None,
    error: BaseException | None = None,
) -> dict[str, object]:
    compute_cap = bool(paths.get("compute_cap"))
    report: dict[str, object] = {
        "schema": (
            "rift_sugavanam_ertin_b7873200_stage1_compute_cap_report_v1"
            if compute_cap else "rift_sugavanam_ertin_b7873200_full_report_v1"
        ),
        "run_name": (
            FULL_STAGE1_COMPUTE_CAP_RUN_NAME if compute_cap else FULL_RUN_NAME
        ),
        "status": status,
        "phase": phase,
        "scope": (
            "canonical sphere10k sealed Stage-1 continuation; target epoch 30; "
            "not a terminal full pipeline or production clearance"
            if compute_cap else "canonical sphere10k sealed full3200 package; not production clearance"
        ),
        "execution_identity": (
            {
                "mode": "stage1_compute_capped",
                "target_epoch": FULL_STAGE1_COMPUTE_CAP_EPOCH,
                "final_epoch": (
                    FULL_STAGE1_COMPUTE_CAP_EPOCH if status == "compute_capped_stage1" else None
                ),
                "underlying_recipe_epochs": 150,
                "source_checkpoint": None if paths.get("source_stage1_latest") is None
                else str(paths["source_stage1_latest"]),
                "source_epoch": (
                    None if not isinstance(paths.get("compute_cap_source"), Mapping)
                    else paths["compute_cap_source"].get("source_epoch")
                ),
                "origin_epoch": (
                    None if not isinstance(paths.get("compute_cap_source"), Mapping)
                    else paths["compute_cap_source"].get("origin_epoch")
                ),
                "new_epochs": (
                    None if not isinstance(paths.get("compute_cap_source"), Mapping)
                    else FULL_STAGE1_COMPUTE_CAP_EPOCH - int(paths["compute_cap_source"]["origin_epoch"])
                ),
                "total_epoch_budget": FULL_STAGE1_COMPUTE_CAP_EPOCH,
                "source_resource_reference": (
                    None if not isinstance(paths.get("compute_cap_source"), Mapping)
                    else paths["compute_cap_source"].get("origin_resource_reference")
                ),
                "resource_scope": "resumed_epochs_only; source epochs are pre-existing",
                "terminal": status == "compute_capped_stage1",
                "stage2_started": False,
                "signal_readout_label": (
                    "nonterminal_compute_capped" if status == "compute_capped_stage1"
                    else "clean_interruption_before_compute_cap"
                ),
                "geometry_metrics": "N/A",
                "geometry_status": "not_applicable_no_stage2_compute_cap",
                "auto_resume_normal_full_pipeline": False,
            }
            if compute_cap else None
        ),
        "input_archive": str(paths.get("npz_path", B787_3200_CANONICAL_NPZ_PATH)),
        "parent_manifest": str(paths.get("role_manifest", B787_3200_CANONICAL_MANIFEST_PATH)),
        "roles": {
            "train_count": B787_3200_NUM_TRAIN,
            "validation_count": B787_3200_NUM_VALIDATION,
            "reserved_test_count": B787_3200_NUM_TEST,
            "unused_count": B787_3200_NUM_UNUSED,
            "test_and_unused_response_materialized": False,
        },
        "normalization": None if normalization is None else dict(normalization),
        "stage1": {
            "recipe": None if stage1_recipe is None else dict(stage1_recipe),
            "native_metrics": None if native_metrics is None else dict(native_metrics),
            "bundle": None if compute_cap else str(paths["stage1_bundle"]),
            "checkpoint_latest": str(paths["stage1_latest"]),
            "checkpoint_final": None if compute_cap else str(paths["stage1_final"]),
            "resource": None if stage1_resource is None else dict(stage1_resource),
        },
        "stage1_to_stage2": {
            "source": None if compute_cap else str(paths["stage1_bundle"]),
            "source_role": "not_run" if compute_cap else "checkpoint_final",
            "validated_before_geometry": False if compute_cap else paths["stage1_bundle"].is_file(),
            "stage2_bundle_only_input": False if compute_cap else True,
            "raw_npz_opened_by_stage2": False,
            "ground_truth_geometry_used": False,
            "status": "not_run_compute_cap" if compute_cap else "ready",
        },
        "stage2": {
            "campaign_identity": None if compute_cap else FULL_STAGE2_CAMPAIGN,
            "artifact_identity": None if compute_cap else FULL_STAGE2_ARTIFACT,
            "recipe": None if stage2_recipe is None else dict(stage2_recipe),
            "resource": None if stage2_resource is None else dict(stage2_resource),
            "status": "not_run_compute_cap" if compute_cap else status,
            "geometry_metrics": "N/A" if compute_cap else None,
            "geometry_status": "not_applicable_no_stage2_compute_cap" if compute_cap else None,
        },
        "provisional_geometry": None if provisional is None else dict(provisional),
        "terminal_failure_path": terminal_failure,
        "technical_execution_failure": status == "technical_execution_failure",
        "topology_gate_passed": (
            None if provisional is None
            else bool(provisional.get("audit", {}).get("topology", {}).get("passed", False))
        ),
        "provisional_geometry_exported": provisional is not None,
        "complete_package_published": status == "complete",
        "production_clearance": False,
        "resource_accounting": str(paths["resource"]),
        "total_wall_seconds": float(time.monotonic() - started),
    }
    if error is not None:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    atomic_json_dump(report, paths["report"])
    return report


def _record_stage2_entry_clean(
    paths: Mapping[str, Path],
    *,
    started: float,
    normalization: Mapping[str, object] | None,
    native_metrics: Mapping[str, object] | None,
    stage1_recipe: Mapping[str, object] | None,
    stage1_resource: Mapping[str, object],
    stage2_recipe: Mapping[str, object] | None = None,
) -> int:
    if paths["stage2_root"].exists() and any(paths["stage2_root"].iterdir()):
        raise FullContractError("cannot record Stage-2 entry recovery after Stage-2 artifacts appeared")
    atomic_json_dump(
        {
            "schema": "rift_sugavanam_ertin_b7873200_full_stage2_entry_v1",
            "phase": "stage1_terminal_recovery",
            "state": "clean_interrupted",
            "resume_allowed": True,
            "resume_target": str(paths["stage1_bundle"]),
            "checkpoint_path": str(paths["stage1_bundle"]),
            "reason": "termination arrived after validated Stage-1 bundle handoff and before Stage-2 entry",
        },
        paths["stage2_entry_clean"],
    )
    _write_resource(
        paths["resource"],
        started=started,
        status="clean_interruption",
        phase="stage1_terminal_recovery",
        stage1=stage1_resource,
    )
    _report(
        paths,
        status="clean_interruption",
        phase="stage1_terminal_recovery",
        started=started,
        normalization=normalization,
        native_metrics=native_metrics,
        stage1_recipe=stage1_recipe,
        stage1_resource=stage1_resource,
        stage2_recipe=stage2_recipe,
    )
    print(f"SE_B7873200_FULL_STAGE2_ENTRY_CLEAN_INTERRUPTION: {paths['report']}", flush=True)
    return CLEAN_INTERRUPTION_EXIT_CODE


def _record_stage2_resume_clean(
    paths: Mapping[str, Path],
    *,
    started: float,
    normalization: Mapping[str, object] | None,
    native_metrics: Mapping[str, object] | None,
    stage1_recipe: Mapping[str, object] | None,
    stage1_resource: Mapping[str, object],
    stage2_recipe: Mapping[str, object] | None = None,
) -> int:
    # A repeated Stage-2 interruption has a usable lifecycle/latest pair.
    # Validate it and preserve it; do not create or rewrite a Stage-2 entry
    # record, and do not turn the already-existing Stage-2 root into a fresh
    # bundle-entry continuation.
    _validate_stage2_clean_record(paths)
    _write_resource(
        paths["resource"],
        started=started,
        status="clean_interruption",
        phase="stage2",
        stage1=stage1_resource,
    )
    _report(
        paths,
        status="clean_interruption",
        phase="stage2",
        started=started,
        normalization=normalization,
        native_metrics=native_metrics,
        stage1_recipe=stage1_recipe,
        stage1_resource=stage1_resource,
        stage2_recipe=stage2_recipe,
    )
    print(f"SE_B7873200_FULL_STAGE2_RESUME_CLEAN_INTERRUPTION: {paths['report']}", flush=True)
    return CLEAN_INTERRUPTION_EXIT_CODE


def _record_pending_stage2_stop(
    paths: Mapping[str, Path],
    *,
    stage2_resume: bool,
    started: float,
    normalization: Mapping[str, object] | None,
    native_metrics: Mapping[str, object] | None,
    stage1_recipe: Mapping[str, object] | None,
    stage1_resource: Mapping[str, object],
    stage2_recipe: Mapping[str, object] | None = None,
) -> int:
    if stage2_resume:
        return _record_stage2_resume_clean(
            paths,
            started=started,
            normalization=normalization,
            native_metrics=native_metrics,
            stage1_recipe=stage1_recipe,
            stage1_resource=stage1_resource,
            stage2_recipe=stage2_recipe,
        )
    return _record_stage2_entry_clean(
        paths,
        started=started,
        normalization=normalization,
        native_metrics=native_metrics,
        stage1_recipe=stage1_recipe,
        stage1_resource=stage1_resource,
        stage2_recipe=stage2_recipe,
    )


def _run_full(args: argparse.Namespace, paths: Mapping[str, Path]) -> int:
    _validate_identity(args, paths)
    paths = {**paths, "npz_path": Path(args.npz_path), "role_manifest": Path(args.parent_role_manifest)}
    paths["root"].mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    _begin_resource_attempt(paths["resource"], started=started, resume=args.resume)
    reset_stop_request()
    install_stop_handlers()
    current_phase = "preflight"
    normalization: dict[str, object] | None = None
    native_metrics: dict[str, object] | None = None
    stage1_resource: dict[str, object] | None = None
    stage2_resource: dict[str, object] | None = None
    stage1_recipe_summary: dict[str, object] | None = None
    recipe: dict[str, object] | None = None
    try:
        arrays, sealed = load_b7873200_sealed_identity(
            args.npz_path, args.parent_role_manifest
        )
        acquisition = build_b7873200_acquisition_identity(arrays, sealed)
        roles = sealed["role_ids"]
        smoke_driver.assert_restricted_roles(
            arrays,
            {
                "train": roles["train"],
                "validation": roles["validation"],
                "test": roles["reserved_test"],
                "unused": roles["unused"],
            },
        )
        normalization = smoke_driver.compute_train_only_signal_normalization(
            arrays, roles["train"]
        )
        stage1_recipe = default_stage1_recipe(sealed, acquisition)
        stage1_recipe_summary = _stage1_report_summary(stage1_recipe)
        resume = None if args.resume is None else os.fspath(args.resume)
        stage2_resume = resume is not None and _same_path(resume, paths["stage2_latest"])
        current_phase = "stage1"
        had_stage1_bundle = paths["stage1_bundle"].is_file()
        stage1_start = time.monotonic()
        try:
            validated, native_metrics = _load_or_run_stage1(
                paths,
                stage1_recipe,
                normalization,
                resume=resume if resume is not None and (
                    (paths.get("compute_cap") and _same_path(resume, paths["source_stage1_latest"]))
                    or
                    _same_path(resume, paths["stage1_latest"])
                    or _same_path(resume, paths["stage1_final"])
                    or _same_path(resume, paths["stage1_bundle"])
                ) else None,
            )
        except BaseException as stage1_exc:
            execution = "terminal_readout_recovery" if _same_path(resume or "", paths["stage1_final"]) else "training"
            stage1_resource = _phase_resource(stage1_start, execution=execution)
            if (
                paths.get("compute_cap")
                and isinstance(stage1_exc, SystemExit)
                and isinstance(stage1_exc.code, int)
                and stage1_exc.code == CLEAN_INTERRUPTION_EXIT_CODE
            ):
                record = load_json_mapping(paths["stage1_clean"], "compute-cap clean record")
                if record.get("phase") != "stage1_compute_capped" or record.get("target_epoch") != FULL_STAGE1_COMPUTE_CAP_EPOCH:
                    raise FullContractError("compute-capped Stage-1 returned without its epoch-30 evidence") from stage1_exc
                native = record.get("native_metrics")
                terminal_cap = record.get("terminal") is True and record.get("resume_allowed") is False
                if terminal_cap:
                    if not isinstance(native, Mapping):
                        raise FullContractError("compute-capped Stage-1 evidence lacks native signal readout") from stage1_exc
                    _validate_compute_cap_boundary(paths)
                elif record.get("resume_allowed") is not True:
                    raise FullContractError("intermediate compute-cap evidence is not resumable") from stage1_exc
                stage1_resource = _phase_resource(
                    stage1_start,
                    execution="compute_capped_stage1" if terminal_cap else "compute_capped_intermediate",
                )
                _write_resource(
                    paths["resource"], started=started,
                    status="compute_capped_stage1" if terminal_cap else "clean_interruption",
                    phase="stage1_compute_capped", stage1=stage1_resource,
                )
                resource_record = load_json_mapping(paths["resource"], "compute-cap resource ledger")
                resource_record.update({
                    "accounting_scope": "resumed_epochs_only",
                    "total_epoch_budget": FULL_STAGE1_COMPUTE_CAP_EPOCH,
                    "origin_epoch": record.get("origin_epoch"),
                    "origin_checkpoint_path": record.get("origin_checkpoint_path"),
                    "origin_resource_reference": record.get("origin_resource_reference"),
                    "new_epochs": (
                        FULL_STAGE1_COMPUTE_CAP_EPOCH - int(record["origin_epoch"])
                        if isinstance(record.get("origin_epoch"), int) else None
                    ),
                })
                atomic_json_dump(resource_record, paths["resource"])
                _report(
                    paths,
                    status="compute_capped_stage1" if terminal_cap else "clean_interruption",
                    phase="stage1_compute_capped",
                    started=started,
                    normalization=normalization,
                    native_metrics=native,
                    stage1_recipe=stage1_recipe_summary,
                    stage1_resource=stage1_resource,
                )
                if terminal_cap:
                    print(f"SE_B7873200_FULL_STAGE1_COMPUTE_CAPPED_EPOCH30: {paths['report']}", flush=True)
                    return 0
                print(f"SE_B7873200_FULL_STAGE1_COMPUTE_CAP_CLEAN_INTERRUPTION: {paths['report']}", flush=True)
                return CLEAN_INTERRUPTION_EXIT_CODE
            _write_resource(
                paths["resource"], started=started, status="clean_interruption"
                if isinstance(stage1_exc, SystemExit)
                and isinstance(stage1_exc.code, int)
                and stage1_exc.code == CLEAN_INTERRUPTION_EXIT_CODE
                else "failed", phase="stage1", stage1=stage1_resource, error=stage1_exc,
            )
            raise
        execution = "reused_terminal_bundle" if had_stage1_bundle else (
            "terminal_readout_recovery" if _same_path(resume or "", paths["stage1_final"]) else "training"
        )
        stage1_resource = _phase_resource(stage1_start, execution=execution)
        _enforce_resource(stage1_resource, "full Stage-1")
        _write_resource(
            paths["resource"], started=started, status="running", phase="stage2",
            stage1=stage1_resource,
        )
        if stop_requested():
            return _record_pending_stage2_stop(
                paths,
                stage2_resume=stage2_resume,
                started=started,
                normalization=normalization,
                native_metrics=native_metrics,
                stage1_recipe=stage1_recipe_summary,
                stage1_resource=stage1_resource,
            )

        source = load_b7873200_stage1_cloud(
            paths["stage1_bundle"], expected_recipe=stage1_recipe
        )
        recipe = full_stage2_recipe()
        stage2_contract = FullStage2Contract(
            output_dir=str(paths["stage2_root"]),
            stage1_final_bundle=str(paths["stage1_bundle"]),
        )
        provenance = _stage2_provenance(source, recipe, paths, normalization)
        stage2_args = stage2.parse_args(
            ["--device", args.device]
            + ([] if args.resume is None or not _same_path(args.resume, paths["stage2_latest"])
               else ["--resume", str(paths["stage2_latest"])])
        )
        if stop_requested():
            return _record_pending_stage2_stop(
                paths,
                stage2_resume=stage2_resume,
                started=started,
                normalization=normalization,
                native_metrics=native_metrics,
                stage1_recipe=stage1_recipe_summary,
                stage1_resource=stage1_resource,
                stage2_recipe=recipe,
            )
        current_phase = "stage2"
        stage2_start = time.monotonic()
        try:
            result = stage2._run(
                stage2_args,
                contract=stage2_contract,
                recipe=recipe,
                source=source,
                provenance=provenance,
                engineering_override=True,
                preserve_existing_stop=True,
            )
        except RuntimeError as exc:
            stage2_resource = _phase_resource(stage2_start, execution="stage2_lifecycle")
            _write_resource(
                paths["resource"], started=started, status="scientific_negative_topology"
                if "Stage-2 export topology failed" in str(exc) else "failed",
                phase="stage2", stage1=stage1_resource, stage2=stage2_resource, error=exc,
            )
            if "Stage-2 export topology failed" not in str(exc):
                raise
            provisional = _write_provisional_geometry(
                source=source, recipe=recipe, contract=stage2_contract,
                device=torch.device(args.device),
            )
            stage2_resource = _phase_resource(stage2_start, execution="stage2_lifecycle")
            _enforce_resource(stage2_resource, "full Stage-2 including provisional export")
            _report(
                paths,
                status="scientific_negative_topology",
                phase="stage2",
                started=started,
                normalization=normalization,
                native_metrics=native_metrics,
                stage1_recipe=stage1_recipe_summary,
                stage1_resource=stage1_resource,
                stage2_recipe=recipe,
                stage2_resource=stage2_resource,
                provisional=provisional,
                terminal_failure=str(Path(stage2_contract.output_dir) / "terminal_failure.json"),
                error=exc,
            )
            _write_resource(
                paths["resource"], started=started, status="scientific_negative_topology",
                phase="complete", stage1=stage1_resource, stage2=stage2_resource,
            )
            print(f"SE_B7873200_FULL_SCIENTIFIC_NEGATIVE_TOPOLOGY_COMPLETE: {paths['report']}", flush=True)
            return 0
        except SystemExit as exc:
            stage2_resource = _phase_resource(stage2_start, execution="stage2_lifecycle")
            code = exc.code if isinstance(exc.code, int) else 1
            if code == CLEAN_INTERRUPTION_EXIT_CODE:
                _validate_stage2_returned_clean(stage2_contract, provenance, recipe)
                _write_resource(
                    paths["resource"], started=started, status="clean_interruption",
                    phase="stage2", stage1=stage1_resource, stage2=stage2_resource,
                )
                _report(
                    paths, status="clean_interruption", phase="stage2", started=started,
                    normalization=normalization, native_metrics=native_metrics,
                    stage1_recipe=stage1_recipe_summary, stage1_resource=stage1_resource,
                    stage2_recipe=recipe, stage2_resource=stage2_resource,
                )
                return CLEAN_INTERRUPTION_EXIT_CODE
            raise
        stage2_resource = _phase_resource(stage2_start, execution="stage2_lifecycle")
        _enforce_resource(stage2_resource, "full Stage-2")
        if result == CLEAN_INTERRUPTION_EXIT_CODE:
            _validate_stage2_returned_clean(stage2_contract, provenance, recipe)
            _write_resource(
                paths["resource"], started=started, status="clean_interruption",
                phase="stage2", stage1=stage1_resource, stage2=stage2_resource,
            )
            _report(
                paths, status="clean_interruption", phase="stage2", started=started,
                normalization=normalization, native_metrics=native_metrics,
                stage1_recipe=stage1_recipe_summary, stage1_resource=stage1_resource,
                stage2_recipe=recipe, stage2_resource=stage2_resource,
            )
            return CLEAN_INTERRUPTION_EXIT_CODE
        if result != 0:
            raise FullContractError(f"Stage-2 returned nonzero result {result}")
        if not Path(stage2_contract.complete_dir).is_dir():
            raise FullContractError("Stage-2 returned without a complete package")
        _write_resource(
            paths["resource"], started=started, status="complete", phase="complete",
            stage1=stage1_resource, stage2=stage2_resource,
        )
        _report(
            paths, status="complete", phase="complete", started=started, normalization=normalization,
            native_metrics=native_metrics, stage1_recipe=stage1_recipe_summary,
            stage1_resource=stage1_resource, stage2_recipe=recipe,
            stage2_resource=stage2_resource,
        )
        print(f"SE_B7873200_FULL_LIFECYCLE_COMPLETE: {paths['report']}", flush=True)
        return 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if code == CLEAN_INTERRUPTION_EXIT_CODE:
            _write_resource(
                paths["resource"], started=started, status="clean_interruption", phase="interrupted",
                stage1=stage1_resource, stage2=stage2_resource, error=exc,
            )
            _report(
                paths, status="clean_interruption", phase=current_phase, started=started,
                normalization=normalization, native_metrics=native_metrics,
                stage1_recipe=stage1_recipe_summary, stage1_resource=stage1_resource,
                stage2_recipe=recipe, stage2_resource=stage2_resource, error=exc,
            )
            return CLEAN_INTERRUPTION_EXIT_CODE
        terminal = str(paths["stage2_root"] / "terminal_failure.json")
        _write_resource(
            paths["resource"], started=started, status="failed", phase="failed",
            stage1=stage1_resource, stage2=stage2_resource, error=exc,
        )
        _report(
            paths, status="technical_execution_failure", phase=current_phase, started=started,
            normalization=normalization, native_metrics=native_metrics,
            stage1_recipe=stage1_recipe_summary, stage1_resource=stage1_resource,
            stage2_recipe=recipe, stage2_resource=stage2_resource,
            terminal_failure=terminal if Path(terminal).is_file() else None,
            error=exc,
        )
        raise
    except BaseException as exc:
        terminal = str(paths["stage2_root"] / "terminal_failure.json")
        _write_resource(
            paths["resource"], started=started, status="failed", phase="failed",
            stage1=stage1_resource, stage2=stage2_resource, error=exc,
        )
        _report(
            paths, status="technical_execution_failure", phase=current_phase, started=started,
            normalization=normalization, native_metrics=native_metrics,
            stage1_recipe=stage1_recipe_summary, stage1_resource=stage1_resource,
            stage2_recipe=recipe, stage2_resource=stage2_resource,
            terminal_failure=terminal if Path(terminal).is_file() else None,
            error=exc,
        )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--recipe", choices=["legacy-full", "paper-v1"], default="legacy-full")
    selected, _ = selector.parse_known_args(argv)
    if selected.recipe == "paper-v1":
        from rift.sugavanam_ertin_paper_workflow import main as paper_main
        return paper_main(argv)
    args = parse_args(argv)
    paths = _paths(args)
    return _run_full(args, paths)


if __name__ == "__main__":
    raise SystemExit(main())
