"""Frozen B787 sphere10k/3200 contract for corrected SE Stage 2 v1.

This is a new lane, intentionally not an addition to the historical six-scene
``stage2_stabilized_v3`` registry.  Its only source is the semantic B7873200
Stage-1 *final bundle* produced by
``train_sugavanam_ertin_stage1.py``.  Stage 2 never opens the raw
radar archive or a response accessor.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import os
from pathlib import Path
from typing import Mapping

from rift.sugavanam_ertin_b7873200_stage1 import (
    B787_3200_CANONICAL_NPZ_PATH,
    STAGE1_BUNDLE_PATH,
    STAGE1_FINAL_BUNDLE_FILENAME,
    Stage1ContractError,
    load_b7873200_stage1_cloud,
    validate_b7873200_stage1_final,
)


CAMPAIGN_IDENTITY = "rift_sugavanam_ertin_b7873200_stage2_v1"
ARTIFACT_IDENTITY = "b787_sugavanam_ertin_b7873200_stage2_v1"
METHOD_NAME = "Sugavanam--Ertin B7873200 stabilized two-stage derivative"
POLICY_IDENTITY = "closed_init_oriented_offsets_strict_projection_atomic_export_v1"
IMPLEMENTATION_KIND = "stabilized derivative; not the raw paper reproduction"
OUTPUT_DIR = (
    "/storage/scratch1/1/dbao31/rift_homemade_baselines_20260905_v1/"
    "b787_sugavanam_ertin_stage2_v1"
)
LATEST_CHECKPOINT = f"{OUTPUT_DIR}/checkpoint_latest.pth.tar"
COMPLETE_DIR = f"{OUTPUT_DIR}/complete"


class Stage2ContractError(ValueError):
    """Raised when an input, artifact, or command breaks the B787 v1 contract."""


@dataclass(frozen=True)
class B7873200Stage2Contract:
    """Immutable source/output identity for this one corrected B787 lane."""

    campaign_identity: str = CAMPAIGN_IDENTITY
    artifact_identity: str = ARTIFACT_IDENTITY
    method: str = METHOD_NAME
    policy_identity: str = POLICY_IDENTITY
    implementation_kind: str = IMPLEMENTATION_KIND
    canonical_npz_path: str = B787_3200_CANONICAL_NPZ_PATH
    stage1_final_bundle: str = STAGE1_BUNDLE_PATH
    output_dir: str = OUTPUT_DIR

    @property
    def latest_checkpoint(self) -> str:
        return f"{self.output_dir}/checkpoint_latest.pth.tar"

    @property
    def lifecycle_path(self) -> str:
        return f"{self.output_dir}/lifecycle.json"

    @property
    def complete_dir(self) -> str:
        return f"{self.output_dir}/complete"

    def identity_record(self) -> dict[str, object]:
        return {
            "campaign_identity": self.campaign_identity,
            "artifact_identity": self.artifact_identity,
            "method": self.method,
            "policy_identity": self.policy_identity,
            "implementation_kind": self.implementation_kind,
            "canonical_npz_path": self.canonical_npz_path,
            "stage1_final_bundle_filename": STAGE1_FINAL_BUNDLE_FILENAME,
            "output_dir": self.output_dir,
            "ground_truth_geometry_used": False,
            "novel_view_signal_supported": False,
        }


def default_stage2_recipe() -> dict[str, object]:
    """The explicit full corrected-SDF recipe; operational scheduling is excluded."""

    return {
        "schema": "rift_sugavanam_ertin_b7873200_stage2_recipe_v1",
        "steps": 5_000,
        "init_steps": 200,
        "init_lr": 1.0e-3,
        "init_batch": 2048,
        "init_log_every": 25,
        "batch_on": 512,
        "batch_off": 512,
        "batch_iso": 256,
        "batch_signed": 512,
        "batch_boundary": 256,
        "batch_inner": 128,
        "n_iso": 1024,
        "iso_start": 60,
        "iso_refresh": 30,
        "scatter_threshold": 0.15,
        "max_scatter_points": 20_000,
        "normal_radius_policy": "three_stage1_voxel_pitches",
        "signed_offset_pitches": 1.0,
        "n_fourier": 9,
        "fourier_scale": 2.0,
        "hidden_dim": 512,
        "n_layers": 8,
        "lr": 1.0e-4,
        "alpha_off": 100.0,
        "lambda_on": 1.0,
        "lambda_normal": 1.0,
        "lambda_signed": 1.0,
        "lambda_off": 1.0,
        "lambda_eik": 1.0,
        "lambda_iso": 1.0,
        "lambda_iso_normal": 1.0,
        "lambda_boundary": 1.0,
        "lambda_inner": 1.0,
        "off_warmup": 20,
        "off_ramp": 40,
        "radius_quantile": 0.5,
        "radius_cap_fraction": 0.65,
        "boundary_margin": 1.0e-4,
        "boundary_shell_resolution": 24,
        "inner_anchor_count": 1024,
        "gate_every": 10,
        "gate_grid": 32,
        "grid_chunk": 32768,
        "projection_oversample": 2.0,
        "mesh_grid": 64,
        "mesh_points": 5000,
        "save_every": 20,
        "seed": 42,
    }


def validate_contract(contract: B7873200Stage2Contract) -> B7873200Stage2Contract:
    if not isinstance(contract, B7873200Stage2Contract):
        raise Stage2ContractError("B787 Stage-2 requires its registered contract object")
    canonical = B7873200Stage2Contract()
    if contract != canonical:
        raise Stage2ContractError("B787 Stage-2 paths or identities are not the frozen v1 values")
    if os.path.basename(contract.stage1_final_bundle) != STAGE1_FINAL_BUNDLE_FILENAME:
        raise Stage2ContractError("B787 Stage-2 source is not the Stage-1 final bundle")
    return canonical


def validate_stage2_recipe(recipe: Mapping[str, object]) -> dict[str, object]:
    expected = default_stage2_recipe()
    if not isinstance(recipe, Mapping) or dict(recipe) != expected:
        raise Stage2ContractError("B787 Stage-2 recipe differs from frozen v1")
    return copy.deepcopy(expected)


def load_validated_stage1_source(
    contract: B7873200Stage2Contract,
    *,
    stage1_path: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Validate the complete Stage-1 bundle before deriving its cloud.

    This function has no raw NPZ argument and never opens an archive.  Its
    isolated Stage-1 loader performs grid/provenance validation before it
    derives the thresholded cloud.
    """

    contract = validate_contract(contract)
    requested = contract.stage1_final_bundle if stage1_path is None else os.fspath(stage1_path)
    if os.path.realpath(os.path.abspath(requested)) != os.path.realpath(
        os.path.abspath(contract.stage1_final_bundle)
    ):
        raise Stage2ContractError("B787 Stage-2 cannot substitute a different Stage-1 source")
    try:
        source = load_b7873200_stage1_cloud(requested)
    except Stage1ContractError as exc:
        raise Stage2ContractError(f"B787 Stage-1 final bundle rejected: {exc}") from exc
    record = source.get("stage1_record")
    if not isinstance(record, Mapping):
        raise Stage2ContractError("validated Stage-1 source lacks its provenance record")
    # Re-run the named-bundle check here to make clear that a cloud helper must
    # never be used as a permissive checkpoint loader.
    try:
        validate_b7873200_stage1_final(requested)
    except Stage1ContractError as exc:
        raise Stage2ContractError(f"B787 Stage-1 final bundle rejected: {exc}") from exc
    return source


def stage2_provenance_record(
    contract: B7873200Stage2Contract,
    stage1_record: Mapping[str, object],
    recipe: Mapping[str, object],
) -> dict[str, object]:
    contract = validate_contract(contract)
    recipe = validate_stage2_recipe(recipe)
    if not isinstance(stage1_record, Mapping):
        raise Stage2ContractError("Stage-2 requires the validated Stage-1 provenance record")
    return {
        "contract": contract.identity_record(),
        "stage1_record": copy.deepcopy(dict(stage1_record)),
        "stage2_recipe": recipe,
        "ground_truth_geometry_used": False,
        "novel_view_signal_supported": False,
    }


def lifecycle_contract_record(provenance: Mapping[str, object]) -> dict[str, object]:
    """Return the finite-JSON contract used by lifecycle/status artifacts.

    The checkpoint keeps the complete validated acquisition tensors.  The
    lifecycle intentionally does not duplicate those arrays: every resume
    revalidates the named Stage-1 final bundle before it reads the latest
    checkpoint.  This JSON record still binds the stage, exact ordered sealed
    roles, frozen non-array recipe, structural audit, and B787 output identity.
    """

    if not isinstance(provenance, Mapping):
        raise Stage2ContractError("Stage-2 provenance must be a mapping")
    contract = provenance.get("contract")
    stage1 = provenance.get("stage1_record")
    recipe = provenance.get("stage2_recipe")
    if not isinstance(contract, Mapping) or not isinstance(stage1, Mapping) or not isinstance(recipe, Mapping):
        raise Stage2ContractError("Stage-2 provenance is incomplete")
    stage1_recipe = stage1.get("stage1_recipe")
    if not isinstance(stage1_recipe, Mapping):
        raise Stage2ContractError("Stage-2 provenance lacks the Stage-1 recipe")
    stage1_recipe_json = copy.deepcopy(dict(stage1_recipe))
    acquisition = stage1_recipe_json.pop("acquisition_identity", None)
    if not isinstance(acquisition, Mapping) or acquisition.get("response_payload_materialized") is not False:
        raise Stage2ContractError("Stage-1 acquisition provenance is invalid")
    return {
        "contract": copy.deepcopy(dict(contract)),
        "stage1": {
            "schema": stage1.get("schema"),
            "recipe_id": stage1.get("recipe_id"),
            "role": stage1.get("role"),
            "sealed_protocol_identity": copy.deepcopy(stage1.get("sealed_protocol_identity")),
            "stage1_recipe_without_acquisition_arrays": stage1_recipe_json,
            "structural_audit": copy.deepcopy(stage1.get("structural_audit")),
        },
        "stage2_recipe": copy.deepcopy(dict(recipe)),
        "ground_truth_geometry_used": False,
        "novel_view_signal_supported": False,
    }


def provenance_output_identity(provenance: Mapping[str, object]) -> dict[str, str]:
    """Return the output identity supplied by the validated provenance."""

    contract = provenance.get("contract")
    if not isinstance(contract, Mapping):
        raise Stage2ContractError("Stage-2 provenance lacks its output identity")
    identity = {}
    for key in ("method", "campaign_identity", "artifact_identity", "policy_identity", "implementation_kind"):
        value = contract.get(key)
        if not isinstance(value, str) or not value:
            raise Stage2ContractError(f"Stage-2 provenance identity field {key!r} is invalid")
        identity[key] = value
    return identity
