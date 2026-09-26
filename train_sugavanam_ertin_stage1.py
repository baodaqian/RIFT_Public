"""Run an object-bound Sugavanam--Ertin Stage-1 scattering field.

This is an intentionally thin, fixed-recipe wrapper around the project's
existing sealed ``train.py`` route.  It never alters historical trainers or
checkpoints.  Once the generic trainer reaches its terminal checkpoint, this
wrapper emits a separate, semantically validated B7873200 final bundle for the
corrected Stage-2 lane.

Only a manager-safe resume from this identity's generic ``checkpoint_latest``
is accepted.  A completed bundle is a validated no-op; it is never overwritten.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from rift.sugavanam_ertin_b7873200_stage1 import (
    B787_3200_CANONICAL_MANIFEST_PATH,
    B787_3200_CANONICAL_NPZ_PATH,
    GENERIC_FINAL_FILENAME,
    STAGE1_BUNDLE_PATH,
    STAGE1_FINAL_BUNDLE_FILENAME,
    STAGE1_OUTPUT_DIR,
    Stage1ContractError,
    atomic_save_stage1_bundle,
    build_b7873200_acquisition_identity,
    build_stage1_final_bundle,
    default_stage1_recipe,
    load_b7873200_sealed_identity,
    validate_b7873200_stage1_recovery_state,
    validate_b7873200_stage1_final,
)


CHECKPOINT_NAME = "b787_sugavanam_ertin_stage1_v1"
CHECKPOINT_ROOT = str(Path(STAGE1_OUTPUT_DIR).parent)
GENERIC_OUTPUT_DIR = STAGE1_OUTPUT_DIR
GENERIC_FINAL_PATH = f"{GENERIC_OUTPUT_DIR}/{GENERIC_FINAL_FILENAME}"
GENERIC_LATEST_PATH = f"{GENERIC_OUTPUT_DIR}/checkpoint_latest.pth.tar"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        default=None,
        help="Only this identity's generic checkpoint_latest.pth.tar after a clean interruption.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _same_path(left: str | os.PathLike[str], right: str | os.PathLike[str]) -> bool:
    return os.path.realpath(os.path.abspath(os.fspath(left))) == os.path.realpath(
        os.path.abspath(os.fspath(right))
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.resume is not None and not _same_path(args.resume, GENERIC_LATEST_PATH):
        raise Stage1ContractError(
            "B7873200 Stage-1 resume must be its own checkpoint_latest.pth.tar"
        )


def frozen_train_argv(resume: str | None) -> list[str]:
    """Build the explicit generic CLI whose semantics are recorded in v1."""

    arguments = [
        "--data-format", "npz",
        "--npz-path", B787_3200_CANONICAL_NPZ_PATH,
        "--npz-sealed-protocol",
        "--npz-role-manifest", B787_3200_CANONICAL_MANIFEST_PATH,
        "--checkpoint-name", CHECKPOINT_NAME,
        "--checkpoint-root", CHECKPOINT_ROOT,
        "--execution-contract-label", "rift_sugavanam_ertin_b7873200_stage1_v1",
        "--require-full-resume-state",
        "--num-train", "3200",
        "--num-val", "1000",
        "--num-test", "1000",
        "--num-freq-wanted", "600",
        "--epochs", "150",
        "--loss", "complex",
        "--scene-repr", "grid",
        "--forward-operator", "range",
        "--range-model", "product",
        "--compute-dtype", "float64",
        "--point-chunk", "65536",
        "--pair-chunk", "64",
        "--extent", "0.15",
        "--granularity", "48",
        "--phase-sign", "-1.0",
        "--bp-init", "400",
        "--lr", "0.003",
        "--l1-weight", "3e-7",
        "--adam-eps", "1e-20",
        "--checkpoint-metric", "val",
        "--t0", "10",
        "--t-mult", "2",
        "--seed", "42",
        "--prune-every", "0",
        "--prune-threshold", "0.0",
        "--prune-criterion", "energy",
        "--prune-start-epoch", "0",
        "--prune-mode", "mass",
        "--prune-target-active", "0",
        "--prune-end-epoch", "0",
        "--prune-min-active", "0",
    ]
    if resume is not None:
        arguments.extend(("--resume", GENERIC_LATEST_PATH))
    return arguments


def _bundle_existing_final(recipe: dict[str, object]) -> None:
    """Turn an already terminal generic final into the isolated Stage-1 bundle."""

    if os.path.exists(STAGE1_BUNDLE_PATH):
        validate_b7873200_stage1_final(STAGE1_BUNDLE_PATH, expected_recipe=recipe)
        print("B7873200 Stage-1 final bundle already validated; no-op.", flush=True)
        return
    if not os.path.isfile(GENERIC_FINAL_PATH):
        raise FileNotFoundError(GENERIC_FINAL_PATH)
    # Keep the generic state unchanged and add the isolated semantic record in
    # a new artifact.  This cannot mutate a baseline checkpoint in place.
    from rift.sugavanam_ertin_b7873200_stage1 import _torch_load

    generic_state = _torch_load(GENERIC_FINAL_PATH)
    bundle = build_stage1_final_bundle(
        generic_state,
        recipe,
        provenance={
            "generic_final_path": GENERIC_FINAL_PATH,
            "bundle_path": STAGE1_BUNDLE_PATH,
        },
    )
    atomic_save_stage1_bundle(bundle, STAGE1_BUNDLE_PATH)
    validate_b7873200_stage1_final(STAGE1_BUNDLE_PATH, expected_recipe=recipe)
    print(f"Wrote validated B7873200 Stage-1 bundle: {STAGE1_BUNDLE_PATH}", flush=True)


def _validate_manager_selected_resume(recipe: dict[str, object]) -> None:
    """Check a partial generic checkpoint without declaring it clean ourselves.

    The experiment manager owns the scheduler-state decision.  This wrapper
    accepts a recovery command only when the caller explicitly supplies the
    sole allowed latest path and that file already proves the same parsed
    scientific configuration.  A partial directory is never silently started
    over or auto-resumed.
    """

    from rift.sugavanam_ertin_b7873200_stage1 import _torch_load

    state = _torch_load(GENERIC_LATEST_PATH)
    validate_b7873200_stage1_recovery_state(state, recipe)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    arrays, sealed = load_b7873200_sealed_identity(
        B787_3200_CANONICAL_NPZ_PATH, B787_3200_CANONICAL_MANIFEST_PATH
    )
    acquisition = build_b7873200_acquisition_identity(arrays, sealed)
    recipe = default_stage1_recipe(sealed, acquisition)

    if os.path.exists(STAGE1_BUNDLE_PATH):
        validate_b7873200_stage1_final(STAGE1_BUNDLE_PATH, expected_recipe=recipe)
        print("B7873200 Stage-1 bundle is complete and validated; no-op.", flush=True)
        return 0
    if os.path.isfile(GENERIC_FINAL_PATH):
        _bundle_existing_final(recipe)
        return 0
    if os.path.exists(GENERIC_LATEST_PATH):
        if args.resume is None:
            raise Stage1ContractError(
                "a partial B7873200 Stage-1 run exists; the manager must first classify it, "
                "then invoke this wrapper with its exact --resume path"
            )
        _validate_manager_selected_resume(recipe)
    elif args.resume is not None:
        raise Stage1ContractError("--resume was supplied but this Stage-1 latest checkpoint is absent")
    elif Path(GENERIC_OUTPUT_DIR).exists() and any(Path(GENERIC_OUTPUT_DIR).iterdir()):
        raise Stage1ContractError("fresh B7873200 Stage-1 refuses a nonempty generic output directory")

    # The generic trainer owns full compute and its own resume mechanics.  This
    # wrapper only permits it after the header/role/acquisition preflight above.
    import train

    train.main(frozen_train_argv(args.resume))
    if os.path.isfile(GENERIC_FINAL_PATH):
        _bundle_existing_final(recipe)
        return 0
    if os.path.isfile(GENERIC_LATEST_PATH):
        raise RuntimeError(
            "generic Stage-1 returned without a final; its partial checkpoint is preserved, "
            "but only the manager may classify whether a future explicit resume is allowed"
        )
    raise RuntimeError(
        "B7873200 Stage-1 ended without a final or a manager-safe latest checkpoint"
    )


if __name__ == "__main__":
    raise SystemExit(main())
