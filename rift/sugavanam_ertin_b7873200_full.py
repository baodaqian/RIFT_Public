"""Explicit configuration for the B7873200 Sugavanam--Ertin full package.

This module is intentionally Torch-free.  It gives the launcher, source
validator, driver, and handoff one unambiguous full-data identity without
altering the corrected Stage-1/Stage-2 implementation modules or historical
experiment records.
"""

from __future__ import annotations

from typing import Final

from rift.sugavanam_ertin_b7873200_stage1 import (
    B787_3200_CANONICAL_MANIFEST_PATH,
    B787_3200_CANONICAL_NPZ_PATH,
    B787_3200_NUM_UNUSED,
    B787_3200_NUM_TEST,
    B787_3200_NUM_TRAIN,
    B787_3200_NUM_VALIDATION,
    B787_3200_RESPONSE_SHAPE,
)


FULL_SCHEMA: Final = "rift_sugavanam_ertin_b7873200_full_v1"
FULL_RUN_NAME: Final = "b78710k_sugavanam_ertin_full3200_stage1_stage2_replacement_v1"
FULL_ROOT_PARENT: Final = "/storage/scratch1/1/dbao31/rift_b7873200_sugavanam_ertin_full3200_replacement_v1"
FULL_STAGE1_CHECKPOINT_NAME: Final = "sugavanam_ertin_b7873200_stage1_full3200_replacement_v1"
# Compute-capped Stage-1 continuation has a separate identity so its
# nonterminal evidence cannot overwrite the original full-package report.
FULL_STAGE1_COMPUTE_CAP_EPOCH: Final = 30
FULL_STAGE1_COMPUTE_CAP_RUN_NAME: Final = "b78710k_sugavanam_ertin_stage1_compute_capped_epoch30_v1"
FULL_STAGE1_COMPUTE_CAP_CHECKPOINT_NAME: Final = "sugavanam_ertin_b7873200_stage1_compute_capped_epoch30_v1"
FULL_STAGE1_COMPUTE_CAP_REPORT_NAME: Final = "stage1_compute_capped_epoch30_report.json"
FULL_STAGE1_COMPUTE_CAP_RESOURCE_NAME: Final = "stage1_compute_capped_epoch30_resource_accounting.json"
# The output path/checkpoint name are fresh, but the Stage-1 semantic bundle
# deliberately records the already validated frozen v1 execution contract.
FULL_STAGE1_CONTRACT_LABEL: Final = "rift_sugavanam_ertin_b7873200_stage1_v1"
FULL_STAGE2_CAMPAIGN: Final = "rift_sugavanam_ertin_b7873200_stage2_full3200_replacement_v1"
FULL_STAGE2_ARTIFACT: Final = "b787_sugavanam_ertin_stage2_full3200_replacement_v1"
FULL_STAGE2_RECIPE_SCHEMA: Final = "rift_sugavanam_ertin_b7873200_stage2_full3200_replacement_recipe_v1"
FULL_STAGE2_POLICY: Final = "closed_init_oriented_offsets_strict_projection_atomic_export_full3200_v1"
FULL_STAGE1_BUNDLE_FILENAME: Final = "sugavanam_ertin_b7873200_stage1_final_v1.pth.tar"
FULL_REPORT_NAME: Final = "full3200_report.json"
FULL_RESOURCE_NAME: Final = "resource_accounting.json"

FULL_RESPONSE_SHAPE: Final = tuple(B787_3200_RESPONSE_SHAPE)
FULL_ROLE_COUNTS: Final = {
    "train": B787_3200_NUM_TRAIN,
    "validation": B787_3200_NUM_VALIDATION,
    "test": B787_3200_NUM_TEST,
    "unused": B787_3200_NUM_UNUSED,
}

# These are an operational envelope for manager review, not a measured
# runtime prediction.  In particular, the full Stage-1 fit has not run here.
FULL_RESOURCE_ENVELOPE: Final = {
    "account": "gts-jromberg3-ece",
    "launcher": "inferno",
    "partition": "gpu-rtx6000",
    "gpus": 1,
    "gpu_type": "rtx_6000",
    "cpus": 6,
    "memory_gib": 64,
    "temporary_storage_gib": 24,
    "wall_time_hours": 12,
}
FULL_WALL_LIMIT_SECONDS: Final = 12 * 60 * 60
FULL_MEMORY_LIMIT_BYTES: Final = 64 * 1024**3


def full_stage2_recipe() -> dict[str, object]:
    """Return the proposed full-scale Stage-2 recipe.

    The 1,000-step initialization follows the corrected v4 implementation
    package.  The 5,000 SDF steps and large model are retained from the
    existing full v1 recipe; smoke evidence does not prove convergence at
    full scale, so this remains an explicit review item rather than a claim.
    """

    return {
        "schema": FULL_STAGE2_RECIPE_SCHEMA,
        "recipe_revision": 1,
        "steps": 5_000,
        "init_steps": 1_000,
        "init_lr": 5.0e-4,
        "init_batch": 2_048,
        "init_log_every": 100,
        "batch_on": 512,
        "batch_off": 512,
        "batch_iso": 256,
        "batch_signed": 512,
        "batch_boundary": 256,
        "batch_inner": 128,
        "n_iso": 1_024,
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
        "inner_anchor_count": 1_024,
        "gate_every": 10,
        "gate_grid": 32,
        "grid_chunk": 32_768,
        "projection_oversample": 2.0,
        "mesh_grid": 64,
        "mesh_points": 5_000,
        "save_every": 20,
        "seed": 42,
        "budget_basis": {
            "init_steps": {
                "value": 1_000,
                "basis": "corrected_v4 implementation package",
                "full_scale_proven": False,
            },
            "sdf_steps": {
                "value": 5_000,
                "basis": "retained from the frozen full v1 recipe",
                "full_scale_proven": False,
            },
            "status": "proposed_full_budget_requires_manager_review",
        },
    }
