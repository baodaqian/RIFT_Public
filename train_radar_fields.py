#!/usr/bin/env python
"""Train the radar-only Radar Fields baseline on a RIFT npz dataset.

This is intentionally separate from ``train.py``: Radar Fields optimizes real
range-bin intensity and cannot emit RIFT's coherent Re/Im signal. Checkpoints
contain a compatibility ``model_state_dict`` whose single real coefficient is
the occupancy grid, so the existing B787 geometry evaluator can consume them
without a baseline-specific metric implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from rift.encoding import generate_dynamic_grid
from rift.radar_fields import (
    OFFICIAL_REFERENCE_COMMIT,
    NORMALIZED_DB_INTENSITY_DOMAIN,
    NORMALIZED_DB_INTENSITY_LABEL,
    NORMALIZED_DB_RELMSE_LABEL,
    RadarFieldsModel,
    bistatic_range_cells,
    intensity_metrics,
    padded_roi_objective_diagnostics,
    radar_fields_intensity,
    radar_fields_loss,
)
from rift.radar_fields_dataset import (
    RadarFieldsArrays,
    dataset_provenance,
    load_or_create_stats,
    load_radar_fields_npz,
    load_radar_fields_sealed_split_manifest,
    normalize_power_db,
    range_bin_centers,
    range_bin_size,
    restrict_radar_fields_response_views,
    response_view_to_range_power,
    scene_range_mask,
    split_view_indices,
)
from rift.radar_fields_recipe import (
    AUDITED_RECIPE, LEGACY_RECIPE, SOURCE_RECIPE, native_recipe, recipe_name, recipe_contract, validate_recipe_checkpoint,
)
from rift.radar_fields_native import (
    occupancy_probability, render_bistatic_bins, released_batch_loss,
    prepare_bistatic_bins, render_bistatic_batch,
)
from rift.radar_fields_upstream import original_module, OriginalRadarFieldsModel, check_model_backend


STOP_REQUESTED = False

GOTCHA_BACKEND = {
    "schema": "rift_gotcha_backend_v1", "method": "radar_fields", "callable": "run_gotcha",
    "selection_unit": "pass_sector", "joint_passes": True,
    "native_frequency_policy": "ragged_exact", "polarizations": ["hh", "hv", "vh", "vv"],
    "metric_domain": "normalized_dB_range_power_intensity",
}


def run_gotcha(*, dataset, output_dir, config, device, resume):
    from rift.radar_fields_gotcha import run_gotcha as backend
    return backend(dataset=dataset, output_dir=output_dir, config=config, device=device, resume=resume)


def request_stop(signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"Received signal {signum}; will save checkpoint_latest after this optimizer step.", flush=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_cuda_rng_state(
    saved_state: object,
    *,
    expected_device_count: Optional[int] = None,
    require_present: bool = False,
) -> Optional[list[torch.Tensor]]:
    """Validate and return CUDA RNG states as contiguous CPU byte tensors.

    ``torch.load(..., map_location="cuda")`` relocates every tensor in a
    checkpoint, including the CPU byte tensors returned by
    :func:`torch.cuda.get_rng_state_all`.  ``torch.cuda.set_rng_state_all``
    expects those state tensors on CPU, so a CUDA resume must move them back
    deliberately.  The optional count check is reserved for the strict
    versioned continuation contract; legacy CUDA checkpoints retain their
    previous permissive topology behavior.
    """

    if saved_state is None:
        if require_present:
            raise ValueError("strict CUDA resume checkpoint lacks cuda_rng_state")
        return None
    if not isinstance(saved_state, (list, tuple)):
        raise ValueError("cuda_rng_state must be a list or tuple of RNG tensors")
    if expected_device_count is not None and len(saved_state) != int(expected_device_count):
        raise ValueError(
            "strict CUDA resume checkpoint CUDA RNG topology disagrees with the current "
            f"visible device count: checkpoint={len(saved_state)}, current={expected_device_count}"
        )

    normalized = []
    for index, state in enumerate(saved_state):
        if (not torch.is_tensor(state)
                or state.ndim != 1
                or state.dtype != torch.uint8):
            raise ValueError(
                "cuda_rng_state entry "
                f"{index} must be a one-dimensional torch.uint8 RNG tensor"
            )
        normalized.append(state.detach().to(device="cpu", dtype=torch.uint8).contiguous())
    return normalized


def role_provenance(
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    test_indices: Sequence[int],
    *,
    test_payload_materialized: bool,
) -> Dict[str, object]:
    """Record deterministic role identities without reading their responses."""

    roles = {
        "train": [int(value) for value in train_indices],
        "val": [int(value) for value in val_indices],
        "test": [int(value) for value in test_indices],
    }
    return {
        "role_ids": roles,
        "role_counts": {name: len(values) for name, values in roles.items()},
        "test_payload_materialized": bool(test_payload_materialized),
    }


def resolve_diagnostic_view(
    role: str,
    role_index: int,
    train_indices: Sequence[int],
    val_indices: Sequence[int],
) -> int:
    """Return only an explicitly authorized train/validation source view."""

    allowed = {"train": train_indices, "val": val_indices}
    if role not in allowed:
        raise ValueError(
            "diagnostic response materialization is limited to train or val; "
            f"got role {role!r}"
        )
    indices = allowed[role]
    if not 0 <= int(role_index) < len(indices):
        raise ValueError(
            f"diagnostic role index {role_index} is outside the {role} split of length {len(indices)}"
        )
    return int(indices[int(role_index)])


def grid_support_provenance(args) -> Dict[str, object]:
    extent = float(args.extent)
    granularity = int(args.granularity)
    return {
        "representation": "RadarFields hash-grid queried on an explicit cubic readout grid",
        "support_xyz_m": [[-extent, extent], [-extent, extent], [-extent, extent]],
        "readout_granularity": granularity,
        "readout_point_count": granularity ** 3,
    }


def signal_provenance(stats: Mapping[str, object], args) -> Dict[str, object]:
    return {
        "radar_fields_recipe": recipe_contract(args),
        "target_domain": NORMALIZED_DB_INTENSITY_DOMAIN,
        "target_label": NORMALIZED_DB_INTENSITY_LABEL,
        "reported_rel_mse_label": NORMALIZED_DB_RELMSE_LABEL,
        "source_transform": "coherent frequency response -> IFFT over frequency -> squared magnitude range power",
        "normalization": {
            "peak_power": float(stats["peak_power"]),
            "dynamic_range_db": float(stats["dynamic_range_db"]),
            "clamp_floor_relative_power": 10.0 ** (-float(stats["dynamic_range_db"]) / 10.0),
            "power_stats_train_view_ids": stats.get("train_view_indices"),
            "power_stats_normalization_scan_view_ids": stats.get("normalization_scan_view_indices"),
            "power_stats_train_view_ids_verified": bool(stats.get("train_view_ids_verified", False)),
            "power_stats_normalization_provenance": stats.get("normalization_provenance"),
            "power_stats_normalization_provenance_verified": bool(
                stats.get("normalization_provenance_verified", False)
            ),
            "power_stats_cache_reused": bool(stats.get("normalization_stats_cache_reused", False)),
        },
        "render_intensity": {
            "range_law": str(args.range_law),
            "offset": float(args.intensity_offset),
            "scaler": float(args.intensity_scaler),
        },
    }


def checkpoint_provenance(
    resume_path: Optional[str], checkpoint: Optional[Mapping[str, object]],
) -> Dict[str, object]:
    if not resume_path:
        return {
            "initialization": "random_model_parameters",
            "restored": False,
            "checkpoint_path": None,
            "checkpoint_step": None,
            "checkpoint_epoch": None,
            "resume_contract_version": None,
        }
    source = Path(resume_path).resolve()
    return {
        "initialization": "restored_radar_fields_checkpoint",
        "restored": True,
        "checkpoint_path": str(source),
        "checkpoint_file_size_bytes": int(source.stat().st_size) if source.exists() else None,
        "checkpoint_step": int(checkpoint["step"]) if checkpoint and "step" in checkpoint else None,
        "checkpoint_epoch": int(checkpoint["epoch"]) if checkpoint and "epoch" in checkpoint else None,
        "resume_contract_version": checkpoint.get("resume_contract_version") if checkpoint else None,
    }


# New checkpoints carry this marker and must pass the complete continuation
# contract below.  A missing marker denotes a legacy checkpoint: it remains
# restorable, but is never represented as having an exact verified split.
RESUME_CONTRACT_VERSION = 1
SEALED_PROTOCOL_CONTRACT_VERSION = 1


_RESUME_CONFIG_FIELDS = (
    # Exact split recipe.
    "num_train",
    "num_val",
    "num_test",
    "seed",
    "val_from_tail",
    # Representation and target construction.
    "extent",
    "granularity",
    "hidden_dim",
    "feature_dim",
    "sh_degree",
    "sigmoid_tightness",
    "no_batch_norm",
    "hash_levels",
    "hash_features",
    "hash_base_resolution",
    "hash_final_resolution",
    "hash_log2_size",
    "dynamic_range_db",
    "range_margin",
    "range_law",
    "intensity_offset",
    "intensity_scaler",
    # Optimizer/objective controls that alter the continued trajectory.
    "steps",
    "lr",
    "view_batch",
    "train_pairs",
    "val_pairs",
    # These affect which validation samples may set the persisted best
    # checkpoint, as well as when that comparison occurs.  They are part of
    # an exact continuation contract rather than harmless reporting knobs.
    "eval_every",
    "eval_max_views",
    "weight_fft",
    "weight_occ",
    "weight_bimodal",
    "occupancy_threshold",
    "query_chunk",
    "pair_chunk",
)

_SPLIT_ROLE_NAMES = ("train", "val", "test")


def sealed_protocol_contract(sealed_split, split_info: Mapping[str, object]) -> Dict[str, object]:
    """Record the opt-in response-access policy alongside ordinary split IDs."""

    if split_info.get("test_payload_materialized"):
        raise ValueError("a sealed protocol cannot certify a materialized test response payload")
    contract = dict(sealed_split.protocol_contract())
    contract["version"] = SEALED_PROTOCOL_CONTRACT_VERSION
    return contract


def _compatible_value(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) is bool(right)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1.0e-12, abs_tol=1.0e-15)
    return left == right


def _exact_split_role_ids(provenance: Mapping[str, object], *, label: str) -> Dict[str, list[int]]:
    """Validate and normalize persisted role IDs/counts for a strict resume."""

    if not isinstance(provenance, Mapping):
        raise ValueError(f"{label} split_provenance must be a mapping")
    role_ids = provenance.get("role_ids")
    role_counts = provenance.get("role_counts")
    if not isinstance(role_ids, Mapping) or not isinstance(role_counts, Mapping):
        raise ValueError(
            f"{label} split_provenance must contain role_ids and role_counts mappings"
        )
    normalized: Dict[str, list[int]] = {}
    for role in _SPLIT_ROLE_NAMES:
        if role not in role_ids or role not in role_counts:
            raise ValueError(f"{label} split_provenance lacks the {role!r} role")
        raw_ids = role_ids[role]
        if isinstance(raw_ids, (str, bytes)):
            raise ValueError(f"{label} split role {role!r} must be a sequence of integer IDs")
        try:
            ids = [int(value) for value in raw_ids]
            count = int(role_counts[role])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{label} split role {role!r} has non-integer IDs or count"
            ) from exc
        if count != len(ids):
            raise ValueError(
                f"{label} split role {role!r} count={count} disagrees with {len(ids)} IDs"
            )
        if len(set(ids)) != len(ids):
            raise ValueError(f"{label} split role {role!r} contains duplicate view IDs")
        normalized[role] = ids
    return normalized


def _normalized_sealed_protocol_contract(
    contract: Mapping[str, object], *, label: str,
) -> Dict[str, object]:
    """Validate the semantic response-access policy used by a continuation.

    The manifest path and display name are retained in a checkpoint for
    provenance, but are deliberately not resume pins: an equivalent frozen
    manifest may be relocated without changing the supplied source-role IDs.
    The ordinary strict split provenance below compares those train/validation/
    test IDs exactly; this policy comparison covers the remaining sealed-role
    behavior.
    """

    if not isinstance(contract, Mapping):
        raise ValueError(f"{label} sealed_protocol_contract must be a mapping")
    required = (
        "version",
        "role_binding",
        "manifest_path",
        "manifest_name",
        "manifest_schema_version",
        "response_header_shape",
        "response_header_dtype",
        "test_sealed",
        "unused_sealed",
        "complete_partition",
        "unused_role_ids",
        "authorized_response_roles",
        "test_response_materialized",
        "unused_response_materialized",
    )
    missing = [key for key in required if key not in contract]
    if missing:
        raise ValueError(
            f"{label} sealed_protocol_contract lacks required fields: {', '.join(missing)}"
        )
    if int(contract["version"]) != SEALED_PROTOCOL_CONTRACT_VERSION:
        raise ValueError(
            f"{label} sealed_protocol_contract version={contract['version']!r} is unsupported"
        )
    if contract["role_binding"] != "explicit_manifest_source_view_ids":
        raise ValueError(f"{label} sealed protocol is not bound to explicit manifest IDs")
    if not isinstance(contract["manifest_path"], str) or not contract["manifest_path"]:
        raise ValueError(f"{label} sealed_protocol_contract manifest_path must be nonempty text")
    if contract["manifest_name"] is not None and not isinstance(contract["manifest_name"], str):
        raise ValueError(f"{label} sealed_protocol_contract manifest_name must be text or null")
    try:
        manifest_schema_version = int(contract["manifest_schema_version"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} sealed_protocol_contract manifest_schema_version must be an integer"
        ) from exc
    if manifest_schema_version <= 0:
        raise ValueError(f"{label} sealed_protocol_contract manifest_schema_version must be positive")
    raw_header_shape = contract["response_header_shape"]
    if isinstance(raw_header_shape, (str, bytes)):
        raise ValueError(
            f"{label} sealed_protocol_contract response_header_shape must be an integer sequence"
        )
    try:
        header_shape_values = tuple(raw_header_shape)
    except TypeError as exc:
        raise ValueError(
            f"{label} sealed_protocol_contract response_header_shape must be an integer sequence"
        ) from exc
    header_shape = []
    for value in header_shape_values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(
                f"{label} sealed_protocol_contract response_header_shape must contain integers"
            )
        integer = int(value)
        if integer <= 0:
            raise ValueError(
                f"{label} sealed_protocol_contract response_header_shape must contain positive dimensions"
            )
        header_shape.append(integer)
    if len(header_shape) != 5:
        raise ValueError(
            f"{label} sealed_protocol_contract response_header_shape must have five dimensions"
        )
    raw_header_dtype = contract["response_header_dtype"]
    if not isinstance(raw_header_dtype, str) or not raw_header_dtype:
        raise ValueError(
            f"{label} sealed_protocol_contract response_header_dtype must be nonempty text"
        )
    try:
        header_dtype = str(np.dtype(raw_header_dtype))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} sealed_protocol_contract response_header_dtype must be a NumPy dtype string"
        ) from exc
    if contract["test_sealed"] is not True:
        raise ValueError(f"{label} sealed protocol must keep its test role sealed")
    if contract["unused_sealed"] is not True:
        raise ValueError(f"{label} sealed protocol must keep its unused role sealed")
    if contract["complete_partition"] is not True:
        raise ValueError(f"{label} sealed protocol must use a complete source-view partition")
    raw_unused_ids = contract["unused_role_ids"]
    if isinstance(raw_unused_ids, (str, bytes)):
        raise ValueError(f"{label} sealed_protocol_contract unused_role_ids must be an ID sequence")
    try:
        unused_ids = [int(value) for value in raw_unused_ids]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} sealed_protocol_contract unused_role_ids must be integer IDs"
        ) from exc
    if len(set(unused_ids)) != len(unused_ids):
        raise ValueError(f"{label} sealed_protocol_contract unused_role_ids contains duplicates")
    if list(contract["authorized_response_roles"]) != ["train", "val"]:
        raise ValueError(
            f"{label} sealed protocol may authorize only train and validation response roles"
        )
    if contract["test_response_materialized"] is not False:
        raise ValueError(f"{label} sealed protocol records a materialized test response payload")
    if contract["unused_response_materialized"] is not False:
        raise ValueError(f"{label} sealed protocol records a materialized unused response payload")
    return {
        "version": int(contract["version"]),
        "role_binding": contract["role_binding"],
        # Path/name are descriptive provenance only, but the schema version
        # changes the interpretation of the frozen role record and therefore
        # belongs in an exact sealed continuation identity.
        "manifest_schema_version": manifest_schema_version,
        "response_header_shape": header_shape,
        "response_header_dtype": header_dtype,
        "test_sealed": True,
        "unused_sealed": True,
        "complete_partition": True,
        "unused_role_ids": unused_ids,
        **({"dataset_identity": contract["dataset_identity"]}
           if "dataset_identity" in contract else {}),
        **({"acquisition_identity": contract["acquisition_identity"]} if "acquisition_identity" in contract else {}),
        "authorized_response_roles": ["train", "val"],
        "test_response_materialized": False,
        "unused_response_materialized": False,
    }


def preflight_sealed_resume_checkpoint(
    checkpoint: Mapping[str, object],
    *,
    sealed_protocol_requested: bool = False,
    current_dataset: Optional[Mapping[str, object]] = None,
    current_split: Optional[Mapping[str, object]] = None,
    current_sealed_protocol: Optional[Mapping[str, object]] = None,
) -> None:
    """Reject an unsafe sealed resume before payload, stats, or model work.

    A legacy invocation can still use its historic eager path.  A checkpoint
    that was created under the opt-in sealed policy is different: it must not
    silently fall back to eager response loading just long enough to discover
    the mismatch.  This preflight consumes checkpoint metadata and, when the
    current invocation is sealed, only header/pose provenance already resolved
    through the lazy loader.
    """

    saved_sealed_protocol = checkpoint.get("sealed_protocol_contract")
    if current_sealed_protocol is None:
        if not sealed_protocol_requested and saved_sealed_protocol is not None:
            raise ValueError(
                "resume checkpoint is bound to a sealed response-access protocol, "
                "but the current invocation did not opt into it"
            )
        if sealed_protocol_requested and saved_sealed_protocol is None:
            raise ValueError(
                "sealed-protocol resume requires a checkpoint carrying its protocol contract"
            )
        # A sealed checkpoint with --sealed-protocol has to wait only for the
        # current lazy header/manifest binding below.  It must not be judged
        # as a legacy invocation merely because that binding is not available
        # yet; no response payload, stats cache, or model exists at this point.
        return

    if saved_sealed_protocol is None:
        raise ValueError(
            "sealed-protocol resume requires a checkpoint carrying its protocol contract"
        )
    current_contract = _normalized_sealed_protocol_contract(
        current_sealed_protocol, label="current"
    )
    saved_contract = _normalized_sealed_protocol_contract(
        saved_sealed_protocol, label="resume checkpoint"
    )
    if saved_contract != current_contract:
        raise ValueError(
            "resume checkpoint sealed response-access protocol disagrees with the current invocation"
        )
    if "dataset_identity" in current_contract:
        from rift.radar_fields_dataset import validate_power_stats_identity
        validate_power_stats_identity(checkpoint.get("power_stats"), current_contract["dataset_identity"])
        from rift.radar_fields_dataset import validate_power_stats_acquisition
        validate_power_stats_acquisition(checkpoint.get("power_stats"), current_contract.get("acquisition_identity"))

    try:
        strict_contract = int(checkpoint.get("resume_contract_version")) == RESUME_CONTRACT_VERSION
    except (TypeError, ValueError):
        strict_contract = False
    if not strict_contract:
        raise ValueError(
            "sealed-protocol resume requires an exact versioned split continuation contract"
        )
    saved_split = checkpoint.get("split_provenance")
    if current_split is None:
        raise ValueError("sealed-protocol resume requires current split provenance")
    saved_role_ids = _exact_split_role_ids(saved_split, label="resume checkpoint")
    current_role_ids = _exact_split_role_ids(current_split, label="current")
    if saved_role_ids != current_role_ids:
        raise ValueError(
            "resume checkpoint train/val/test view IDs disagree with the current split"
        )
    for label, provenance in (("resume checkpoint", saved_split), ("current", current_split)):
        if not isinstance(provenance, Mapping) or provenance.get("test_payload_materialized") is not False:
            raise ValueError(
                f"{label} sealed split provenance must certify an unmaterialized test response"
            )

    if current_dataset is None:
        raise ValueError("sealed-protocol resume requires current lazy dataset provenance")
    saved_dataset = checkpoint.get("dataset_provenance")
    if not isinstance(saved_dataset, Mapping):
        raise ValueError("sealed-protocol resume requires dataset provenance")
    expected_authorized = len(current_role_ids["train"]) + len(current_role_ids["val"])
    for label, provenance in (
        ("resume checkpoint", saved_dataset),
        ("current", current_dataset),
    ):
        if provenance.get("response_payload_materialized") is not False:
            raise ValueError(
                f"{label} sealed dataset provenance records a materialized response payload"
            )
        if provenance.get("response_access_restricted") is not True:
            raise ValueError(
                f"{label} sealed dataset provenance lacks a train/validation response restriction"
            )
        if provenance.get("authorized_response_view_count") != expected_authorized:
            raise ValueError(
                f"{label} sealed dataset provenance authorizes an unexpected number of response views"
            )
    for name in (
        "response_shape",
        "response_dtype",
        "viewpoint_count",
        "tx_count",
        "rx_count",
        "frequency_count",
    ):
        if saved_dataset.get(name) != current_dataset.get(name):
            raise ValueError(
                f"resume checkpoint dataset {name}={saved_dataset.get(name)!r} disagrees with "
                f"current dataset {current_dataset.get(name)!r}"
            )


def validate_resume_checkpoint(
    checkpoint: Mapping[str, object],
    args,
    stats: Mapping[str, object],
    current_dataset: Optional[Mapping[str, object]] = None,
    current_split: Optional[Mapping[str, object]] = None,
    *,
    resume_device: Optional[torch.device] = None,
    current_sealed_protocol: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Fail on persisted model/normalization incompatibility before restore."""

    validate_recipe_checkpoint(checkpoint, args)

    if checkpoint.get("radar_fields_reference_commit") != OFFICIAL_REFERENCE_COMMIT:
        raise ValueError("resume checkpoint was produced against a different upstream reference")
    if checkpoint.get("auxiliary_geometry_used", True):
        raise ValueError("resume checkpoint does not certify the radar-only contract")
    if "radar_fields_state_dict" not in checkpoint:
        raise ValueError("resume checkpoint lacks radar_fields_state_dict")

    contract_version = checkpoint.get("resume_contract_version")
    if contract_version is None:
        strict_contract = False
    elif int(contract_version) == RESUME_CONTRACT_VERSION:
        strict_contract = True
    else:
        raise ValueError(
            f"unsupported Radar Fields resume_contract_version={contract_version!r}; "
            f"expected {RESUME_CONTRACT_VERSION}"
        )

    continuation_state_verified = False
    cuda_rng_state_verified = False
    if strict_contract:
        # A versioned continuation must restore every state source that affects
        # subsequent updates.  The legacy branch below deliberately retains
        # its permissive behavior, but must not be mistaken for an exact
        # continuation.
        required_state = (
            "optimizer_state_dict",
            "scheduler_state_dict",
            "numpy_rng_state_json",
            "torch_rng_state",
        )
        missing_state = [name for name in required_state if checkpoint.get(name) is None]
        if missing_state:
            raise ValueError(
                "strict resume checkpoint lacks required continuation state: "
                + ", ".join(missing_state)
            )
        for name in ("optimizer_state_dict", "scheduler_state_dict"):
            if not isinstance(checkpoint[name], Mapping):
                raise ValueError(f"strict resume checkpoint {name} must be a mapping")
        if not isinstance(checkpoint["numpy_rng_state_json"], str):
            raise ValueError("strict resume checkpoint numpy_rng_state_json must be text")
        try:
            decoded_numpy_rng = json.loads(checkpoint["numpy_rng_state_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("strict resume checkpoint has invalid numpy_rng_state_json") from exc
        if not isinstance(decoded_numpy_rng, Mapping):
            raise ValueError("strict resume checkpoint numpy_rng_state_json must encode a mapping")
        torch_rng_state = checkpoint["torch_rng_state"]
        if (not torch.is_tensor(torch_rng_state)
                or torch_rng_state.ndim != 1
                or torch_rng_state.dtype != torch.uint8):
            raise ValueError(
                "strict resume checkpoint torch_rng_state must be a one-dimensional "
                "torch.uint8 RNG tensor"
            )
        if resume_device is not None and torch.device(resume_device).type == "cuda":
            # Exact CUDA continuation requires a state for every currently
            # visible device.  CPU resumes intentionally do not impose this
            # requirement: a saved GPU state is neither used nor needed there.
            normalize_cuda_rng_state(
                checkpoint.get("cuda_rng_state"),
                expected_device_count=torch.cuda.device_count(),
                require_present=True,
            )
            cuda_rng_state_verified = True
        continuation_state_verified = True

    saved_args = checkpoint.get("args")
    checked_args = []
    if strict_contract and saved_args is None:
        raise ValueError("strict resume checkpoint lacks persisted args")
    if saved_args is not None:
        if not isinstance(saved_args, Mapping):
            raise ValueError("resume checkpoint args must be a mapping")
        if strict_contract:
            missing_controls = [
                name
                for name in _RESUME_CONFIG_FIELDS
                if name not in saved_args or not hasattr(args, name)
            ]
            if missing_controls:
                raise ValueError(
                    "strict resume checkpoint lacks required split/objective controls: "
                    + ", ".join(missing_controls)
                )
        mismatches = []
        for name in _RESUME_CONFIG_FIELDS:
            if name in saved_args and hasattr(args, name):
                current = getattr(args, name)
                if not _compatible_value(saved_args[name], current):
                    mismatches.append(f"{name}: checkpoint={saved_args[name]!r}, current={current!r}")
                else:
                    checked_args.append(name)
        if mismatches:
            raise ValueError("resume checkpoint configuration mismatch: " + "; ".join(mismatches))

    split_verified = False
    saved_split = checkpoint.get("split_provenance")
    if saved_split is not None and current_split is not None:
        try:
            saved_role_ids = _exact_split_role_ids(saved_split, label="resume checkpoint")
            current_role_ids = _exact_split_role_ids(current_split, label="current")
        except ValueError:
            if strict_contract:
                raise
        else:
            if saved_role_ids != current_role_ids:
                raise ValueError(
                    "resume checkpoint train/val/test view IDs disagree with the current split"
                )
            split_verified = True
    elif strict_contract:
        raise ValueError("strict resume checkpoint lacks current or persisted split provenance")

    sealed_protocol_verified = False
    saved_sealed_protocol = checkpoint.get("sealed_protocol_contract")
    if current_sealed_protocol is None:
        if saved_sealed_protocol is not None:
            raise ValueError(
                "resume checkpoint is bound to a sealed response-access protocol, "
                "but the current invocation did not opt into it"
            )
    else:
        current_contract = _normalized_sealed_protocol_contract(
            current_sealed_protocol, label="current"
        )
        if saved_sealed_protocol is None:
            raise ValueError(
                "sealed-protocol resume requires a checkpoint carrying its protocol contract"
            )
        saved_contract = _normalized_sealed_protocol_contract(
            saved_sealed_protocol, label="resume checkpoint"
        )
        if saved_contract != current_contract:
            raise ValueError(
                "resume checkpoint sealed response-access protocol disagrees with the current invocation"
            )
        if not strict_contract or not split_verified:
            raise ValueError(
                "sealed-protocol resume requires an exact versioned split continuation contract"
            )
        for label, provenance in (("resume checkpoint", saved_split), ("current", current_split)):
            if not isinstance(provenance, Mapping) or provenance.get("test_payload_materialized") is not False:
                raise ValueError(
                    f"{label} sealed split provenance must certify an unmaterialized test response"
                )
        sealed_protocol_verified = True

    saved_stats = checkpoint.get("power_stats")
    if current_sealed_protocol is not None and "dataset_identity" in current_sealed_protocol:
        from rift.radar_fields_dataset import validate_power_stats_identity
        validate_power_stats_identity(saved_stats, current_sealed_protocol["dataset_identity"])
        validate_power_stats_identity(stats, current_sealed_protocol["dataset_identity"])
        from rift.radar_fields_dataset import validate_power_stats_acquisition
        for value in (saved_stats, stats):
            validate_power_stats_acquisition(value, current_sealed_protocol.get("acquisition_identity"))
    power_verified = False
    if saved_stats is not None:
        if not isinstance(saved_stats, Mapping):
            raise ValueError("resume checkpoint power_stats must be a mapping")
        for name in ("peak_power", "dynamic_range_db"):
            if name not in saved_stats:
                raise ValueError(f"resume checkpoint power_stats lacks {name}")
            if not _compatible_value(saved_stats[name], stats[name]):
                raise ValueError(
                    f"resume checkpoint {name}={saved_stats[name]!r} disagrees with current "
                    f"power stats {stats[name]!r}"
                )
        power_verified = True

    dataset_verified = False
    dataset_identity_verified = False
    saved_dataset = checkpoint.get("dataset_provenance")
    if saved_dataset is not None and current_dataset is not None:
        if not isinstance(saved_dataset, Mapping):
            raise ValueError("resume checkpoint dataset_provenance must be a mapping")
        for name in ("response_shape", "response_dtype", "viewpoint_count", "tx_count", "rx_count", "frequency_count"):
            if name in saved_dataset and saved_dataset[name] != current_dataset.get(name):
                raise ValueError(
                    f"resume checkpoint dataset {name}={saved_dataset[name]!r} disagrees with "
                    f"current dataset {current_dataset.get(name)!r}"
                )
        dataset_verified = True
        # Shape alone is not enough to identify an acquisition in the legacy
        # eager path, so ordinary checkpoints retain the resolved-path/file-
        # size guard.  The sealed path deliberately relies on its validated
        # header plus explicit frozen source-role IDs instead: a corrected or
        # relocated archive/manifest is valid provenance, not a path/hash pin.
        identity_fields = (
            ()
            if current_sealed_protocol is not None
            else ("dataset_path", "dataset_file_size_bytes")
        )
        comparable_identity_fields = [
            name
            for name in identity_fields
            if saved_dataset.get(name) is not None and current_dataset.get(name) is not None
        ]
        for name in comparable_identity_fields:
            if saved_dataset[name] != current_dataset[name]:
                raise ValueError(
                    f"resume checkpoint dataset {name}={saved_dataset[name]!r} disagrees with "
                    f"current dataset {current_dataset[name]!r}"
                )
        dataset_identity_verified = bool(comparable_identity_fields)

    if current_sealed_protocol is not None:
        # A sealed checkpoint must certify the concrete lazy capability, not
        # merely claim a policy string.  The source paths remain provenance
        # only; these checks concern response materialization and role scope.
        expected_authorized = (
            len(current_role_ids["train"]) + len(current_role_ids["val"])
        )
        for label, provenance in (
            ("resume checkpoint", saved_dataset),
            ("current", current_dataset),
        ):
            if not isinstance(provenance, Mapping):
                raise ValueError(
                    f"{label} sealed dataset provenance must record the lazy response capability"
                )
            if provenance.get("response_payload_materialized") is not False:
                raise ValueError(
                    f"{label} sealed dataset provenance records a materialized response payload"
                )
            if provenance.get("response_access_restricted") is not True:
                raise ValueError(
                    f"{label} sealed dataset provenance lacks a train/validation response restriction"
                )
            if provenance.get("authorized_response_view_count") != expected_authorized:
                raise ValueError(
                    f"{label} sealed dataset provenance authorizes an unexpected number of response views"
                )

    return {
        "persisted_args_checked": checked_args,
        "strict_resume_contract": strict_contract,
        "split_provenance_verified": split_verified,
        "legacy_contract_unverified": not strict_contract,
        "continuation_state_verified": continuation_state_verified,
        "cuda_rng_state_verified": cuda_rng_state_verified,
        "power_stats_verified": power_verified,
        "dataset_provenance_verified": dataset_verified,
        "dataset_identity_verified": dataset_identity_verified,
        "sealed_protocol_verified": sealed_protocol_verified,
    }


def evenly_spaced_pairs(total_pairs: int, requested: int) -> np.ndarray:
    if requested <= 0 or requested >= total_pairs:
        return np.arange(total_pairs, dtype=np.int64)
    return np.unique(np.linspace(0, total_pairs - 1, requested).round().astype(np.int64))


def sample_pairs(rng: np.random.Generator, total_pairs: int, requested: int) -> np.ndarray:
    if requested <= 0 or requested >= total_pairs:
        return np.arange(total_pairs, dtype=np.int64)
    return np.sort(rng.choice(total_pairs, size=requested, replace=False).astype(np.int64))


class CoveredViewSampler:
    """Permutation epochs with persisted coverage; normalization is not exposure."""

    def __init__(self, indices, state=None):
        self.indices = [int(x) for x in indices]
        self.order = []
        self.cursor = 0
        self.completed_passes = 0
        self.counts = {str(x): 0 for x in self.indices}
        if not self.indices or len(set(self.indices)) != len(self.indices):
            raise ValueError("coverage sampler needs unique training IDs")
        if state is not None:
            if state.get("indices") != self.indices:
                raise ValueError("coverage sampler training IDs mismatch")
            self.order = list(state["order"])
            self.cursor = int(state["cursor"])
            self.completed_passes = int(state["completed_passes"])
            self.counts = dict(state["counts"])
            if (sorted(self.order) != sorted(self.indices) or not 0 <= self.cursor <= len(self.order)
                    or self.completed_passes < 0 or set(self.counts) != set(map(str, self.indices))):
                raise ValueError("invalid saved coverage sampler")
            visited = set(self.order[:self.cursor])
            if any(value != self.completed_passes + int(int(key) in visited)
                   for key, value in self.counts.items()):
                raise ValueError("coverage counts disagree with saved cursor")

    def next(self, count, rng, *, source_sampling=False):
        selected = []
        for _ in range(count):
            if not self.order or self.cursor == len(self.order):
                if self.order:
                    self.completed_passes += 1
                self.order = (list(torch.utils.data.SubsetRandomSampler(self.indices)) if source_sampling
                              else [int(x) for x in rng.permutation(self.indices)])
                self.cursor = 0
            view = self.order[self.cursor]
            self.cursor += 1
            self.counts[str(view)] += 1
            selected.append(view)
        return selected

    def state_dict(self):
        return {"indices": self.indices, "order": self.order, "cursor": self.cursor,
                "completed_passes": self.completed_passes, "counts": self.counts}


def preflight_audited_continuation(checkpoint, args, train_indices, dataset_info,
                                   split_info, sealed_protocol_info):
    """Check all continuation metadata before a normalization response scan."""
    if "training_view_coverage" not in checkpoint:
        raise ValueError("audited checkpoint lacks optimizer view exposure state")
    sampler = CoveredViewSampler(train_indices, checkpoint["training_view_coverage"])
    if sum(sampler.counts.values()) != int(checkpoint["step"]) * args.view_batch:
        raise ValueError("coverage state disagrees with completed optimizer steps")
    stats = checkpoint.get("power_stats")
    if not isinstance(stats, Mapping):
        raise ValueError("audited checkpoint lacks normalization statistics")
    for key in ("peak_power", "dynamic_range_db"):
        if key not in stats or not math.isfinite(float(stats[key])) or float(stats[key]) <= 0:
            raise ValueError(f"invalid audited normalization {key}")
    from rift.radar_fields_dataset import normalization_scan_indices
    expected_scan = normalization_scan_indices(train_indices, max_views=args.stats_max_views)
    if (stats.get("train_view_indices") != train_indices.tolist()
            or stats.get("normalization_scan_view_indices") != expected_scan
            or not _compatible_value(stats["dynamic_range_db"], args.dynamic_range_db)):
        raise ValueError("audited normalization role/control mismatch")
    validate_resume_checkpoint(checkpoint, args, stats, dataset_info, split_info,
                               resume_device=torch.device(args.device),
                               current_sealed_protocol=sealed_protocol_info)


def audited_view_tensors(model, arrays, view_index, pair_indices_np, ranges, stats, args,
                         mask_progress, device, *, defer_render=False):
    """Shared training/readout frontend for v2/v3 range-power recipes."""
    rotation = (arrays.source_sensor_rotation(view_index, device=device)
                if recipe_name(args) == SOURCE_RECIPE else None)
    target_power = response_view_to_range_power(arrays.response_view(view_index), pair_indices_np, device=device)
    target = normalize_power_db(target_power, stats["peak_power"], stats["dynamic_range_db"])
    occupancy = occupancy_probability(
        target, noise_axis="range", noise_multiplier=args.occupancy_noise_multiplier,
        probability_offset=args.occupancy_probability_offset,
        probability_scale=args.occupancy_probability_scale, decay_bins=args.occupancy_decay_bins,
    )
    if recipe_name(args) == SOURCE_RECIPE:
        target = target * (target > .1525)  # released noise_floor; preprocessing script was not released
    viewpoint = torch.as_tensor(arrays.viewpoint_positions[view_index], dtype=torch.float64, device=device)
    roi = scene_range_mask(ranges, viewpoint, args.extent, margin=args.range_margin)
    if not roi.any():
        raise ValueError("view has no scene range bins")
    pairs = torch.as_tensor(pair_indices_np, device=device, dtype=torch.long)
    tx = torch.as_tensor(arrays.tx_pos[view_index], dtype=torch.float64, device=device)[pairs // arrays.num_rx]
    rx = torch.as_tensor(arrays.rx_pos[view_index], dtype=torch.float64, device=device)[pairs % arrays.num_rx]
    prepared = {"geometry": prepare_bistatic_bins(tx, rx, ranges[roi], extent=args.extent,
                                                 ray_samples=args.ray_samples,
                                                 source_sampling=recipe_name(args) == SOURCE_RECIPE,
                                                 rotation=rotation),
                "target": target[:, roi], "occupancy_target": occupancy[:, roi],
                "roi": roi, "ranges": ranges[roi]}
    if defer_render:
        return prepared
    field = render_bistatic_batch(model, [prepared["geometry"]], query_chunk=args.query_chunk,
                                  mask_progress=mask_progress)[0]
    return finish_audited_view(prepared, field, args)


def finish_audited_view(prepared, field, args):
    bin_ranges = prepared["ranges"][None, :].to(field["rcs"])
    if args.range_law in ("released", "code_r2"):
        prediction = original_module("radarfields.radar").rcs_to_intensity(
            field["rcs"], bin_ranges, args.intensity_offset, args.intensity_scaler,
            args.range_law == "released")
    else:
        prediction = radar_fields_intensity(field["rcs"], bin_ranges, args.intensity_offset,
                                            args.intensity_scaler, args.range_law)
    return {"prediction": prediction, "target": prepared["target"], "occupancy": field["alpha"],
            "occupancy_target": prepared["occupancy_target"], "valid": field["coverage"] > 0,
            "coverage": field["coverage"], "roi": prepared["roi"]}


def view_objective(
    model: RadarFieldsModel,
    arrays: RadarFieldsArrays,
    view_index: int,
    pair_indices_np: np.ndarray,
    xyz: torch.Tensor,
    ranges: torch.Tensor,
    stats: Dict[str, float],
    args,
    mask_progress: Optional[float],
    device: torch.device,
    diagnostic_padded_roi: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, float]]:
    if native_recipe(args):
        record = audited_view_tensors(model, arrays, view_index, pair_indices_np, ranges,
                                      stats, args, mask_progress, device)
        loss, terms = released_batch_loss([record], weight_fft=args.weight_fft,
                                         weight_occ=args.weight_occ, weight_bimodal=args.weight_bimodal,
                                             source_exact=recipe_name(args) == SOURCE_RECIPE)
        metrics = intensity_metrics(record["prediction"].detach(), record["target"].detach())
        metrics.update(loss=float(loss.detach()),
                       occupancy_positive_fraction=float((record["occupancy_target"] > 0.01).float().mean()),
                       supported_fraction=float(record["valid"].float().mean()),
                       mean_ray_coverage=float(record["coverage"].mean()))
        if diagnostic_padded_roi:
            metrics.update(padded_roi_objective_diagnostics(record["prediction"], record["target"],
                           record["valid"], model.parameters(), weight_fft=args.weight_fft))
        return loss, terms, metrics
    pair_indices = torch.as_tensor(pair_indices_np, dtype=torch.long, device=device)
    target_power = response_view_to_range_power(
        arrays.response_view(view_index), pair_indices_np, device=device
    )
    target_intensity = normalize_power_db(
        target_power, stats["peak_power"], stats["dynamic_range_db"]
    )

    viewpoint = torch.as_tensor(
        arrays.viewpoint_positions[view_index], dtype=torch.float32, device=device
    )
    tx_pos = torch.as_tensor(arrays.tx_pos[view_index], dtype=torch.float32, device=device)
    rx_pos = torch.as_tensor(arrays.rx_pos[view_index], dtype=torch.float32, device=device)
    # Match the release's ray-direction convention: direction points from
    # the sensor origin toward each queried scene point.  The exact Tx/Rx
    # coordinates remain reserved for bistatic range-cell construction.
    direction = xyz - viewpoint[None, :]

    field = model.query_chunked(
        xyz, direction, mask_progress=mask_progress, chunk_size=args.query_chunk
    )
    cell_values, cell_mass = bistatic_range_cells(
        torch.stack((field["alpha"], field["rcs"]), dim=-1),
        xyz,
        tx_pos,
        rx_pos,
        pair_indices,
        bin_size=range_bin_size(arrays.metadata),
        num_bins=arrays.num_freq,
        pair_chunk=args.pair_chunk,
    )

    roi = scene_range_mask(ranges, viewpoint, args.extent, margin=args.range_margin)
    if not roi.any():
        raise RuntimeError(f"view {view_index} has no range bins intersecting the scene box")
    valid_cells = cell_mass[:, roi] > 0
    pred_occupancy = cell_values[:, roi, 0] * valid_cells.to(cell_values.dtype)
    pred_rcs = cell_values[:, roi, 1] * valid_cells.to(cell_values.dtype)
    target_intensity = target_intensity[:, roi]
    roi_ranges = ranges[roi][None, :]
    pred_intensity = radar_fields_intensity(
        pred_rcs,
        roi_ranges,
        offset=args.intensity_offset,
        scaler=args.intensity_scaler,
        range_law=args.range_law,
    )
    target_occupancy = (target_intensity >= args.occupancy_threshold).to(pred_occupancy.dtype)
    target_occupancy = target_occupancy * valid_cells.to(target_occupancy.dtype)

    loss, terms = radar_fields_loss(
        pred_intensity,
        target_intensity,
        pred_occupancy,
        target_occupancy,
        weight_fft=args.weight_fft,
        weight_occ=args.weight_occ,
        weight_bimodal=args.weight_bimodal,
    )
    metrics = intensity_metrics(pred_intensity.detach(), target_intensity.detach())
    metrics["loss"] = float(loss.detach())
    if diagnostic_padded_roi:
        metrics.update(
            padded_roi_objective_diagnostics(
                pred_intensity,
                target_intensity,
                valid_cells,
                model.parameters(),
                weight_fft=args.weight_fft,
            )
        )
    return loss, terms, metrics


def combine_metrics(records: Iterable[Dict[str, float]]) -> Dict[str, float]:
    records = list(records)
    sq_error = sum(record["sq_error"] for record in records)
    target_power = sum(record["target_power"] for record in records)
    count = sum(record["count"] for record in records)
    mse = sq_error / max(count, 1.0)
    return {
        "loss": float(np.mean([record["loss"] for record in records])) if records else float("nan"),
        "rel_mse": sq_error / max(target_power, 1.0e-30),
        "rmse": math.sqrt(mse),
        "psnr_db": -10.0 * math.log10(max(mse, 1.0e-30)),
        "sq_error": sq_error,
        "target_power": target_power,
        "count": count,
    }


@torch.no_grad()
def _evaluate_impl(
    model: RadarFieldsModel,
    arrays: RadarFieldsArrays,
    view_indices: Sequence[int],
    pair_indices: np.ndarray,
    xyz: torch.Tensor,
    ranges: torch.Tensor,
    stats: Dict[str, float],
    args,
    device: torch.device,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    records = []
    selected_views = list(int(v) for v in view_indices)
    if args.eval_max_views > 0:
        selected_views = selected_views[: args.eval_max_views]
    for number, view in enumerate(selected_views, start=1):
        if STOP_REQUESTED:
            print(
                f"  validation interrupted after {len(records)}/{len(selected_views)} views; "
                "skipping model selection so recovery can checkpoint promptly",
                flush=True,
            )
            break
        _loss, _terms, metrics = view_objective(
            model,
            arrays,
            view,
            pair_indices,
            xyz,
            ranges,
            stats,
            args,
            mask_progress=1.0,
            device=device,
        )
        records.append(metrics)
        if number % 25 == 0:
            print(f"  validation {number}/{len(selected_views)} views", flush=True)
    if was_training:
        model.train()
    combined = combine_metrics(records)
    combined["complete"] = float(len(records) == len(selected_views))
    return combined


def evaluate(model, arrays, view_indices, pair_indices, xyz, ranges, stats, args, device):
    if recipe_name(args) != SOURCE_RECIPE:
        return _evaluate_impl(model, arrays, view_indices, pair_indices, xyz, ranges, stats, args, device)
    devices = list(range(torch.cuda.device_count())) if torch.device(device).type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(0)
        return _evaluate_impl(model, arrays, view_indices, pair_indices, xyz, ranges, stats, args, device)


@torch.no_grad()
def occupancy_compatibility_state(
    model: RadarFieldsModel,
    xyz: torch.Tensor,
    granularity: int,
    query_chunk: int,
) -> Dict[str, torch.Tensor]:
    """Single-coefficient grid understood by eval_b787_geometry_metrics.py."""

    was_training = model.training
    model.eval()
    direction = torch.tensor([0.0, 0.0, 1.0], device=xyz.device)
    alpha = model.query_chunked(
        xyz, direction, mask_progress=1.0, chunk_size=query_chunk
    )["alpha"]
    alpha = alpha.reshape(granularity, granularity, granularity, 1).cpu()
    if was_training:
        model.train()
    return {"w_re": alpha, "w_im": torch.zeros_like(alpha)}


def checkpoint_payload(
    model,
    optimizer,
    scheduler,
    step,
    best_val,
    xyz,
    stats,
    rng,
    history,
    args,
    dataset_info: Optional[Mapping[str, object]] = None,
    split_info: Optional[Mapping[str, object]] = None,
    support_grid_info: Optional[Mapping[str, object]] = None,
    signal_info: Optional[Mapping[str, object]] = None,
    sealed_protocol_info: Optional[Mapping[str, object]] = None,
    view_sampler=None,
) -> Dict[str, object]:
    payload = {
        "artifact_schema": "radar_fields_checkpoint_v2",
        "resume_contract_version": RESUME_CONTRACT_VERSION,
        "step": int(step),
        "epoch": int(step),  # existing geometry scripts display this field
        "loss": float(best_val),
        "best_val_rel_mse": float(best_val),
        "scene_repr": "radar_fields",
        "extent": float(args.extent),
        "granularity": int(args.granularity),
        "radar_fields_reference_commit": OFFICIAL_REFERENCE_COMMIT,
        "auxiliary_geometry_used": False,
        "model_state_dict": occupancy_compatibility_state(
            model, xyz, args.granularity, args.query_chunk
        ),
        "radar_fields_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state_json": json.dumps(rng.bit_generator.state),
        "power_stats": dict(stats),
        "history": list(history),
        "args": vars(args),
        "radar_fields_recipe": recipe_contract(args),
        "dataset_provenance": dict(dataset_info) if dataset_info is not None else None,
        "split_provenance": dict(split_info) if split_info is not None else None,
        "support_grid_provenance": dict(support_grid_info) if support_grid_info is not None else None,
        "signal_provenance": dict(signal_info) if signal_info is not None else None,
    }
    # Preserve the ordinary checkpoint schema when the protocol is not opted
    # into.  Sealed checkpoints carry this additional contract explicitly.
    if sealed_protocol_info is not None:
        payload["sealed_protocol_contract"] = dict(sealed_protocol_info)
    if view_sampler is not None:
        payload["training_view_coverage"] = view_sampler.state_dict()
    return payload


def atomic_torch_save(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    print(f"Checkpoint saved atomically to {path}", flush=True)


def save_history(history: Sequence[Dict[str, object]], path: Path) -> None:
    if not history:
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        # A resumed legacy checkpoint can contain rows without newly added
        # provenance columns.  Taking the ordered union keeps those rows
        # writable while preserving their original metric fields.
        fieldnames = list(dict.fromkeys(key for row in history for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    os.replace(temporary, path)


def build_model(args, device: torch.device) -> RadarFieldsModel:
    if native_recipe(args) and args.model_backend == "upstream-tcnn":
        return OriginalRadarFieldsModel(args).to(device)
    return RadarFieldsModel(
        extent=args.extent,
        hidden_dim=args.hidden_dim,
        feature_dim=args.feature_dim,
        sh_degree=args.sh_degree,
        sigmoid_tightness=args.sigmoid_tightness,
        batch_norm=not args.no_batch_norm,
        hash_levels=args.hash_levels,
        hash_features=args.hash_features,
        hash_base_resolution=args.hash_base_resolution,
        hash_final_resolution=args.hash_final_resolution,
        hash_log2_size=args.hash_log2_size,
        encoding_layout="tcnn" if native_recipe(args) else "legacy",
    ).to(device)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-path")
    parser.add_argument("--object", help="Registered RIFT dataset object or alias")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--checkpoint-name", default="b787_radar_fields")
    parser.add_argument("--checkpoint-root", default="training_checkpoints")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--recipe", choices=(LEGACY_RECIPE, AUDITED_RECIPE, SOURCE_RECIPE), default=LEGACY_RECIPE,
                        help="source-adapted-v3 uses released settings with acquisition adapters; legacy-v1/audited-v2 retain historical recipes")
    parser.add_argument("--model-backend", choices=("upstream-tcnn", "torch"), default=None,
                        help="audited-v2 defaults to the original TCNN model; torch is an explicit portable adaptation")
    parser.add_argument("--ray-samples", type=int, default=64,
                        help="audited-v2 solid-angle quadrature samples per pair/range bin")
    parser.add_argument("--occupancy-noise-multiplier", type=float, default=1.5)
    parser.add_argument("--occupancy-decay-bins", type=float, default=10.0)
    parser.add_argument("--occupancy-probability-offset", type=float, default=-0.15)
    parser.add_argument("--occupancy-probability-scale", type=float, default=2.0)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--diagnose-padded-roi",
        action="store_true",
        help=(
            "measure the existing padded ROI FFT-loss contribution and parameter gradients "
            "for one authorized train/validation view, then exit without an optimizer step"
        ),
    )
    parser.add_argument(
        "--diagnostic-role",
        choices=("train", "val"),
        default="train",
        help="authorized split role for --diagnose-padded-roi (default: train)",
    )
    parser.add_argument(
        "--diagnostic-role-index",
        type=int,
        default=0,
        help="zero-based index within --diagnostic-role for --diagnose-padded-roi",
    )
    parser.add_argument(
        "--diagnostic-pairs",
        type=int,
        default=0,
        help="evenly spaced Tx/Rx pairs for --diagnose-padded-roi; 0 uses all pairs",
    )
    parser.add_argument(
        "--diagnostic-json",
        default=None,
        help="optional new JSON path for --diagnose-padded-roi; stdout is always emitted",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--num-train", type=int, default=1800)
    parser.add_argument("--num-val", type=int, default=200)
    parser.add_argument("--num-test", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-from-tail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--sealed-protocol",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "opt into manifest-bound train/validation response access; the reserved test and "
            "unused roles remain unmaterialized during training/development"
        ),
    )
    parser.add_argument(
        "--sealed-split-manifest",
        default=None,
        help=(
            "explicit JSON manifest with train_indices, validation_indices, test_indices, and "
            "sealed-role policy; required with --sealed-protocol"
        ),
    )

    parser.add_argument("--extent", type=float, default=0.15)
    parser.add_argument("--granularity", type=int, default=48)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--view-batch", type=int, default=10)
    parser.add_argument("--train-pairs", type=int, default=64)
    parser.add_argument("--val-pairs", type=int, default=0, help="0 = all Tx/Rx pairs")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--eval-max-views", type=int, default=0, help="smoke-test limiter; 0 = all")
    parser.add_argument("--stats-max-views", type=int, default=0, help="smoke-test limiter; 0 = all train views")

    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-fft", type=float, default=0.60)
    parser.add_argument("--weight-occ", type=float, default=0.36)
    parser.add_argument("--weight-bimodal", type=float, default=0.03)
    parser.add_argument("--occupancy-threshold", type=float, default=0.05)
    parser.add_argument("--dynamic-range-db", type=float, default=60.0)
    parser.add_argument("--range-margin", type=float, default=0.05)
    parser.add_argument("--range-law", choices=("released", "code_r2", "paper_r4"), default="released")
    parser.add_argument("--intensity-offset", type=float, default=None,
                        help="default 1.0 in audited-v2 (upstream trainer), 0.05 in legacy-v1")
    parser.add_argument("--intensity-scaler", type=float, default=1.0)

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--sigmoid-tightness", type=float, default=1.0)
    parser.add_argument("--no-batch-norm", action="store_true")
    parser.add_argument("--hash-levels", type=int, default=16)
    parser.add_argument("--hash-features", type=int, default=2)
    parser.add_argument("--hash-base-resolution", type=int, default=16)
    parser.add_argument("--hash-final-resolution", type=int, default=512)
    parser.add_argument("--hash-log2-size", type=int, default=19)
    parser.add_argument("--query-chunk", type=int, default=32768)
    parser.add_argument("--pair-chunk", type=int, default=8)
    args = parser.parse_args(argv)
    explicit = {token.split("=")[0] for token in (sys.argv[1:] if argv is None else argv) if token.startswith("--")}
    if args.object:
        from rift.rift_dataset import resolve_object_inputs, DEFAULT_ROOT, object_spec
        npz, manifest = resolve_object_inputs(object_name=args.object,
            dataset_root=args.dataset_root or DEFAULT_ROOT, npz_path=args.npz_path,
            role_manifest_path=args.sealed_split_manifest)
        args.npz_path, args.sealed_split_manifest = str(npz), str(manifest)
        if "--no-sealed-protocol" in explicit:
            parser.error("--object requires the registered sealed protocol")
        args.sealed_protocol = True
        if "--checkpoint-root" not in explicit:
            args.checkpoint_root = str(Path("training_checkpoints/RIFT_dataset") / object_spec(args.object)["object_id"])
        if "--checkpoint-name" not in explicit:
            args.checkpoint_name = "radar_fields"
        for name, value in dict(num_train=3200, num_val=1000, num_test=1000, extent=.15).items():
            if "--" + name.replace("_", "-") in explicit and getattr(args, name) != value:
                parser.error(f"--object fixes {name}={value}")
            setattr(args, name, value)
        if "--recipe" not in explicit:
            args.recipe = SOURCE_RECIPE
    elif args.dataset_root:
        parser.error("--dataset-root requires --object")
    if not args.npz_path:
        parser.error("Provide --object or --npz-path")
    if args.recipe == SOURCE_RECIPE:
        defaults = dict(view_batch=10, train_pairs=100, ray_samples=10, seed=0)
        for name, value in defaults.items():
            if "--" + name.replace("_", "-") not in explicit:
                setattr(args, name, value)
        batches = math.ceil(args.num_train / args.view_batch)
        epochs = math.ceil(800 / batches)
        if "--steps" not in explicit:
            args.steps = epochs * batches
        expected = dict(view_batch=10, train_pairs=100, ray_samples=10, seed=0,
                        steps=epochs*batches, lr=.001, weight_fft=.6, weight_occ=.36,
                        weight_bimodal=.03, range_law="released")
        for name, value in expected.items():
            if getattr(args, name) != value:
                parser.error(f"source-adapted-v3 fixes {name}={value}; choose an explicit engineering recipe for ablations")
        if epochs < 2 or args.num_train % args.view_batch:
            parser.error("source-adapted-v3 requires >=2 complete epochs and full frame batches")
    if args.model_backend is None:
        args.model_backend = "upstream-tcnn" if native_recipe(args) else "torch"
    if recipe_name(args) == LEGACY_RECIPE and args.model_backend != "torch":
        parser.error("legacy-v1 requires its original Torch implementation")
    if args.intensity_offset is None:
        args.intensity_offset = 1.0 if native_recipe(args) else 0.05
    recipe_contract(args)
    if not math.isfinite(args.intensity_offset) or args.intensity_offset <= 0:
        parser.error("--intensity-offset must be finite and positive")
    if not math.isfinite(args.intensity_scaler) or args.intensity_scaler <= 0:
        parser.error("--intensity-scaler must be finite and positive")
    return args


def main():
    args = parse_args()
    if args.eval_only and not args.resume:
        raise ValueError("--eval-only requires --resume")
    if args.eval_only and args.diagnose_padded_roi:
        raise ValueError("--eval-only and --diagnose-padded-roi are mutually exclusive")
    if args.steps <= 0 or args.view_batch <= 0:
        raise ValueError("--steps and --view-batch must be positive")
    if native_recipe(args) and not args.sealed_protocol:
        raise ValueError("audited-v2 requires --sealed-protocol and an explicit role manifest")
    if native_recipe(args) and not args.resume:
        output = Path(args.checkpoint_root) / args.checkpoint_name
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("audited-v2 requires a fresh output directory or explicit matching resume")
    if args.sealed_protocol:
        if not args.sealed_split_manifest:
            raise ValueError("--sealed-protocol requires --sealed-split-manifest")
        if args.num_train <= 0 or args.num_val <= 0 or args.num_test <= 0:
            raise ValueError(
                "--sealed-protocol requires positive --num-train, --num-val, and --num-test"
            )
    elif args.sealed_split_manifest:
        raise ValueError("--sealed-split-manifest requires --sealed-protocol")

    # Inspect a resume checkpoint before any dataset path can take the legacy
    # eager branch.  This is intentionally CPU-mapped and metadata-only from
    # the training perspective; the normal device-mapped restore still occurs
    # below after all continuation gates pass.  In particular, a sealed
    # checkpoint resumed without --sealed-protocol fails before opening an NPZ
    # response, creating a stats cache, a checkpoint directory, or a model.
    resume_preflight_checkpoint = None
    if args.resume:
        resume_preflight_checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=False
        )
        validate_recipe_checkpoint(resume_preflight_checkpoint, args)
        # Reject changed objective/architecture before normalization response reads.
        if native_recipe(args):
            saved_args = resume_preflight_checkpoint.get("args", {})
            for key in _RESUME_CONFIG_FIELDS:
                if key not in saved_args or not _compatible_value(saved_args[key], getattr(args, key)):
                    raise ValueError(f"audited resume configuration mismatch: {key}")
        preflight_sealed_resume_checkpoint(
            resume_preflight_checkpoint,
            sealed_protocol_requested=args.sealed_protocol,
        )

    if native_recipe(args):
        check_model_backend(args)  # Dependencies must fail before any response scan.

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"Radar Fields reference commit: {OFFICIAL_REFERENCE_COMMIT}")
    print("Auxiliary geometry: disabled; initialization: random model parameters")
    print(f"Using device: {device}")

    # The diagnostic and sealed paths resolve only the response header plus
    # pose metadata first.  In sealed mode, explicit role IDs are validated
    # against that header and then installed as a lazy capability before any
    # response payload (including normalization data) can be streamed.
    if args.object or recipe_name(args) == SOURCE_RECIPE:
        from rift.rift_dataset import load_object_contract
        public, contract = load_object_contract(args.npz_path, args.sealed_split_manifest)
        from rift.radar_fields_dataset import from_collection_arrays
        arrays = from_collection_arrays(public, contract)
    else:
        arrays = load_radar_fields_npz(
            args.npz_path,
            load_response=not (args.diagnose_padded_roi or args.sealed_protocol),
        )
    sealed_protocol_info = None
    if args.sealed_protocol:
        sealed_split = load_radar_fields_sealed_split_manifest(
            args.sealed_split_manifest,
            arrays.num_views,
            response_shape=arrays._response_shape(),
            response_dtype=arrays.response_dtype,
            expected_num_train=args.num_train,
            expected_num_val=args.num_val,
            expected_num_test=args.num_test,
        )
        train_indices = np.asarray(sealed_split.train_indices, dtype=np.int64)
        val_indices = np.asarray(sealed_split.validation_indices, dtype=np.int64)
        test_indices = np.asarray(sealed_split.test_indices, dtype=np.int64)
        arrays = restrict_radar_fields_response_views(
            arrays, np.concatenate((train_indices, val_indices))
        )
    else:
        train_indices, val_indices, test_indices = split_view_indices(
            arrays.num_views,
            args.num_train,
            args.num_val,
            args.num_test,
            args.seed,
            val_from_tail=args.val_from_tail,
        )
    split_info = role_provenance(
        train_indices,
        val_indices,
        test_indices,
        test_payload_materialized=arrays.response_is_materialized,
    )
    if args.sealed_protocol:
        sealed_protocol_info = sealed_protocol_contract(sealed_split, split_info)
        from rift.rift_dataset import validate_manifest_object
        with open(args.sealed_split_manifest, encoding="utf-8") as handle:
            identity = validate_manifest_object(json.load(handle), arrays.metadata)
        if identity is not None:
            sealed_protocol_info["dataset_identity"] = identity
            if arrays.acquisition_identity:
                sealed_protocol_info["acquisition_identity"] = arrays.acquisition_identity
    dataset_info = dataset_provenance(arrays)

    # Both a changed manifest role/policy and a sealed-versus-legacy resume
    # are rejected while the dataset is still a restricted lazy header handle.
    # Do this before a cache lookup/recalibration or any model construction.
    if resume_preflight_checkpoint is not None and args.sealed_protocol:
        preflight_sealed_resume_checkpoint(
            resume_preflight_checkpoint,
            sealed_protocol_requested=True,
            current_dataset=dataset_info,
            current_split=split_info,
            current_sealed_protocol=sealed_protocol_info,
        )
        if native_recipe(args):
            preflight_audited_continuation(resume_preflight_checkpoint, args, train_indices,
                                          dataset_info, split_info, sealed_protocol_info)
    # Avoid retaining a full CPU-mapped checkpoint while normalization and the
    # model are prepared; the existing device-mapped restore below remains the
    # sole checkpoint object used for training continuation.
    del resume_preflight_checkpoint

    total_pairs = arrays.num_tx * arrays.num_rx
    val_pairs = evenly_spaced_pairs(total_pairs, args.val_pairs)
    checkpoint_dir = Path(args.checkpoint_root) / args.checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stats = load_or_create_stats(
        str(checkpoint_dir / "radar_fields_power_stats.json"),
        arrays,
        train_indices,
        dynamic_range_db=args.dynamic_range_db,
        max_views=args.stats_max_views,
        sealed_protocol=args.sealed_protocol,
        dataset_identity=(sealed_protocol_info.get("dataset_identity") if sealed_protocol_info else None),
    )
    support_grid_info = grid_support_provenance(args)
    signal_info = signal_provenance(stats, args)
    print(
        f"Dataset: {arrays.num_views} views, {arrays.num_tx}x{arrays.num_rx} pairs, "
        f"{arrays.num_freq} frequencies; range bin={range_bin_size(arrays.metadata):.6f} m"
    )
    print(
        f"Split: train={len(train_indices)} val={len(val_indices)} test={len(test_indices)}; "
        f"normalized-dB range-power peak={float(stats['peak_power']):.6e}",
    )
    if sealed_protocol_info is not None:
        print(
            "Sealed response protocol: explicit manifest roles bound before payload access; "
            "only train and validation responses are authorized during development.",
            flush=True,
        )

    xyz = generate_dynamic_grid(args.granularity, args.extent, device, jitter=False).reshape(-1, 3)
    ranges = range_bin_centers(arrays.metadata, device=device,
                              dtype=torch.float64 if native_recipe(args) else torch.float32)
    model = build_model(args, device)
    parameters = (model.get_params(args.lr) if recipe_name(args) == SOURCE_RECIPE
                  and hasattr(model, "get_params") else model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr, betas=(0.9, 0.99), eps=1.0e-15)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: 0.1 ** min(step / (800 if recipe_name(args) == SOURCE_RECIPE else max(args.steps, 1)), 1.0)
    )
    rng = np.random.default_rng(args.seed)
    start_step = 0
    best_val = float("inf")
    history = []
    view_sampler = CoveredViewSampler(train_indices) if native_recipe(args) else None
    checkpoint = None
    resume_validation = {
        "persisted_args_checked": [],
        "strict_resume_contract": False,
        "split_provenance_verified": False,
        "legacy_contract_unverified": False,
        "continuation_state_verified": False,
        "cuda_rng_state_verified": False,
        "power_stats_verified": False,
        "dataset_provenance_verified": False,
        "dataset_identity_verified": False,
        "sealed_protocol_verified": False,
    }

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        resume_validation = validate_resume_checkpoint(
            checkpoint,
            args,
            stats,
            dataset_info,
            split_info,
            resume_device=device,
            current_sealed_protocol=sealed_protocol_info,
        )
        model.load_state_dict(checkpoint["radar_fields_state_dict"])
        if checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = int(checkpoint["step"])
        if view_sampler is not None:
            if "training_view_coverage" not in checkpoint:
                raise ValueError("audited checkpoint lacks optimizer view exposure state")
            view_sampler = CoveredViewSampler(train_indices, checkpoint["training_view_coverage"])
            if sum(view_sampler.counts.values()) != start_step * args.view_batch:
                raise ValueError("coverage state disagrees with completed optimizer steps")
        best_val = float(checkpoint.get("best_val_rel_mse", checkpoint.get("loss", float("inf"))))
        history = list(checkpoint.get("history", []))
        if checkpoint.get("numpy_rng_state_json") is not None:
            rng.bit_generator.state = json.loads(checkpoint["numpy_rng_state_json"])
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
            cuda_rng_state = normalize_cuda_rng_state(
                checkpoint["cuda_rng_state"],
                expected_device_count=(
                    torch.cuda.device_count()
                    if resume_validation["strict_resume_contract"]
                    else None
                ),
                require_present=resume_validation["strict_resume_contract"],
            )
            # The branch condition and strict validation above ensure this is
            # a concrete CPU tensor list rather than ``None``.
            torch.cuda.set_rng_state_all(cuda_rng_state)
        if resume_validation["legacy_contract_unverified"]:
            print(
                "WARNING: resumed a legacy checkpoint without a strict split/objective "
                "continuation contract; exact split provenance is not certified.",
                flush=True,
            )
        print(f"Resumed Radar Fields from step {start_step}, best val rel-MSE={best_val:.6e}")

    checkpoint_info = checkpoint_provenance(args.resume, checkpoint)

    if args.diagnose_padded_roi:
        if arrays.response_is_materialized:
            raise RuntimeError("diagnostic mode must not retain the full response payload")
        diagnostic_view = resolve_diagnostic_view(
            args.diagnostic_role,
            args.diagnostic_role_index,
            train_indices,
            val_indices,
        )
        diagnostic_pairs = evenly_spaced_pairs(total_pairs, args.diagnostic_pairs)
        was_training = model.training
        model.eval()  # do not update batch-norm state during a measurement-only probe
        _loss, _terms, metrics = view_objective(
            model,
            arrays,
            diagnostic_view,
            diagnostic_pairs,
            xyz,
            ranges,
            stats,
            args,
            mask_progress=1.0,
            device=device,
            diagnostic_padded_roi=True,
        )
        record = {
            "artifact_schema": "radar_fields_padded_roi_v2",
            "diagnostic": "padded_roi_existing_objective_measurement",
            "selected_role": args.diagnostic_role,
            "selected_role_index": int(args.diagnostic_role_index),
            "selected_view_id": diagnostic_view,
            "pair_count": int(diagnostic_pairs.size),
            "dataset": dataset_info,
            "checkpoint": checkpoint_info,
            "checkpoint_compatibility": resume_validation,
            "split": split_info,
            "support_grid": support_grid_info,
            "signal": signal_info,
            "sealed_protocol": sealed_protocol_info,
            "test_response_payload_materialized": False,
            **metrics,
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        if args.diagnostic_json:
            output = Path(args.diagnostic_json)
            if output.exists():
                raise FileExistsError(
                    f"refusing to overwrite an existing diagnostic artifact: {output}"
                )
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temporary, output)
            print(f"wrote {output}", flush=True)
        if was_training:
            model.train()
        return

    if args.eval_only:
        metrics = evaluate(model, arrays, val_indices, val_pairs, xyz, ranges, stats, args, device)
        print(
            f"Validation ({NORMALIZED_DB_RELMSE_LABEL}): rel-MSE={metrics['rel_mse']:.6%} RMSE={metrics['rmse']:.6f} "
            f"PSNR={metrics['psnr_db']:.3f} dB"
        )
        output = checkpoint_dir / "radar_fields_eval.json"
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "artifact_schema": "radar_fields_evaluation_v2",
                    "metric_domain": NORMALIZED_DB_INTENSITY_DOMAIN,
                    "reported_rel_mse_label": NORMALIZED_DB_RELMSE_LABEL,
                    "dataset": dataset_info,
                    "checkpoint": checkpoint_info,
                    "checkpoint_compatibility": resume_validation,
                    "split": split_info,
                    "support_grid": support_grid_info,
                    "signal": signal_info,
                    "sealed_protocol": sealed_protocol_info,
                    **metrics,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        print(f"wrote {output}")
        return

    model.train()
    for step_zero in range(start_step, args.steps):
        step = step_zero + 1
        started = time.time()
        optimizer.zero_grad(set_to_none=True)
        batch_views = (view_sampler.next(args.view_batch, rng, source_sampling=recipe_name(args) == SOURCE_RECIPE) if view_sampler is not None else
                       rng.choice(train_indices, size=args.view_batch, replace=len(train_indices) < args.view_batch))
        records = []
        term_totals = {"fft": 0.0, "occupancy": 0.0, "bimodal": 0.0}
        mask_progress = min(1.0, 0.05 + math.sin(step / max(args.steps - 1, 1) * math.pi / 2.0))

        if recipe_name(args) == SOURCE_RECIPE:
            batches = len(train_indices) // args.view_batch
            epoch = step_zero // batches + 1
            mask_progress = min(1.0, .05 + math.sin(epoch / (args.steps // batches - 1) * math.pi / 2))
        native_records = []
        for view in batch_views:
            pairs = (original_module("radarfields.sampler").get_azimuths(
                1, args.train_pairs, total_pairs, device).cpu().numpy()[0]
                if recipe_name(args) == SOURCE_RECIPE else sample_pairs(rng, total_pairs, args.train_pairs))
            if view_sampler is not None:
                record = audited_view_tensors(model, arrays, int(view), pairs, ranges, stats,
                                               args, mask_progress, device, defer_render=True)
                native_records.append(record)
                continue
            loss, terms, metrics = view_objective(
                model,
                arrays,
                int(view),
                pairs,
                xyz,
                ranges,
                stats,
                args,
                mask_progress=mask_progress,
                device=device,
            )
            (loss / args.view_batch).backward()
            records.append(metrics)
            for key in term_totals:
                term_totals[key] += float(terms[key].detach()) / args.view_batch

        if native_records:
            fields = render_bistatic_batch(model, [record["geometry"] for record in native_records],
                                           query_chunk=args.query_chunk, mask_progress=mask_progress)
            native_records = [finish_audited_view(record, field, args)
                              for record, field in zip(native_records, fields)]
            records = [intensity_metrics(record["prediction"].detach(), record["target"].detach())
                       for record in native_records]
            loss, terms = released_batch_loss(native_records, weight_fft=args.weight_fft,
                                             weight_occ=args.weight_occ, weight_bimodal=args.weight_bimodal,
                                             source_exact=recipe_name(args) == SOURCE_RECIPE)
            loss.backward()
            for record in records:
                record["loss"] = float(loss.detach())
            term_totals = {key: float(value.detach()) for key, value in terms.items()}

        if recipe_name(args) == SOURCE_RECIPE and any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Nonfinite released RF loss gradient; no silent numerical repair")
        optimizer.step()
        scheduler.step()
        train_metrics = combine_metrics(records)
        if view_sampler is not None:
            print(f"  optimizer coverage: unique={sum(n > 0 for n in view_sampler.counts.values())}/"
                  f"{len(train_indices)} min/max exposures={min(view_sampler.counts.values())}/"
                  f"{max(view_sampler.counts.values())}", flush=True)
        elapsed = time.time() - started
        print(
            f"Step [{step}/{args.steps}] loss={train_metrics['loss']:.6e} "
            f"normalized-dB intensity rel-MSE={train_metrics['rel_mse']:.4%} "
            f"fft={term_totals['fft']:.3e} occ={term_totals['occupancy']:.3e} "
            f"bim={term_totals['bimodal']:.3e} [{elapsed:.1f}s/step]",
            flush=True,
        )

        validation = None
        # A preemption checkpoint takes priority over validation.  The Slurm
        # signal window must never be consumed by a 200-view benchmark pass.
        should_evaluate = (step % args.eval_every == 0 or step == args.steps) and not STOP_REQUESTED
        if should_evaluate:
            validation = evaluate(
                model, arrays, val_indices, val_pairs, xyz, ranges, stats, args, device
            )
            if validation["complete"]:
                print(
                    f"Validation [step {step}] normalized-dB intensity rel-MSE={validation['rel_mse']:.6%} "
                    f"RMSE={validation['rmse']:.6f} PSNR={validation['psnr_db']:.3f} dB",
                    flush=True,
                )
                row = {
                    "step": step,
                    "metric_domain": NORMALIZED_DB_INTENSITY_DOMAIN,
                    "train_rel_mse_label": NORMALIZED_DB_RELMSE_LABEL,
                    "val_rel_mse_label": NORMALIZED_DB_RELMSE_LABEL,
                    "train_loss": train_metrics["loss"],
                    "train_rel_mse": train_metrics["rel_mse"],
                    "val_rel_mse": validation["rel_mse"],
                    "val_rmse": validation["rmse"],
                    "val_psnr_db": validation["psnr_db"],
                }
                history.append(row)
                save_history(history, checkpoint_dir / "radar_fields_history.csv")

                if validation["rel_mse"] < best_val:
                    best_val = validation["rel_mse"]
                    payload = checkpoint_payload(
                        model, optimizer, scheduler, step, best_val, xyz, stats, rng, history, args,
                        dataset_info=dataset_info,
                        split_info=split_info,
                        support_grid_info=support_grid_info,
                        signal_info=signal_info,
                        sealed_protocol_info=sealed_protocol_info,
                        view_sampler=view_sampler,
                    )
                    atomic_torch_save(payload, checkpoint_dir / "checkpoint_best.pth.tar")

        should_checkpoint = step % args.checkpoint_every == 0 or should_evaluate or STOP_REQUESTED
        if should_checkpoint:
            payload = checkpoint_payload(
                model, optimizer, scheduler, step, best_val, xyz, stats, rng, history, args,
                dataset_info=dataset_info,
                split_info=split_info,
                support_grid_info=support_grid_info,
                signal_info=signal_info,
                sealed_protocol_info=sealed_protocol_info,
                view_sampler=view_sampler,
            )
            name = "checkpoint_final.pth.tar" if step == args.steps else "checkpoint_latest.pth.tar"
            atomic_torch_save(payload, checkpoint_dir / name)

        if STOP_REQUESTED:
            print("Stopped cleanly after publishing checkpoint_latest.", flush=True)
            return


if __name__ == "__main__":
    main()
