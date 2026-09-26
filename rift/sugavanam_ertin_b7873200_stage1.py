"""B787 sphere10k/3200 Stage-1 provenance for corrected Sugavanam--Ertin.

This module is deliberately separate from the historical Sugavanam--Ertin
trainers.  It turns the generic, sealed ``train.py`` terminal grid checkpoint
into a *new* semantic Stage-1 bundle that a corrected Stage-2 implementation
can consume.  The bundle records the acquisition and the exact training
recipe, but deliberately contains neither a source hash nor a response
payload.  Consequently a legacy sphere2k checkpoint, a best/latest recovery
checkpoint, or a differently configured grid cannot be presented as the
B7873200 Stage-1 source.

The only B787 archive accepted by this lane lives under ``/storage/home``:
``/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/``.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


B787_3200_CANONICAL_NPZ_PATH = (
    "/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/"
    "b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz"
)
B787_3200_CANONICAL_MANIFEST_PATH = (
    "/storage/scratch1/1/dbao31/rift_round8b_impl_20260810/splits/round8b/"
    "b78710k_interp_seed42_train3200_val1000_test1000_v1.json"
)
B787_3200_MANIFEST_NAME = "b78710k_interp_seed42_train3200_val1000_test1000_v1"
B787_3200_RESPONSE_SHAPE = (10_000, 16, 16, 1, 600)
B787_3200_RESPONSE_DTYPE = "complex64"
B787_3200_SPLIT_SEED = 42
B787_3200_NUM_TRAIN = 3_200
B787_3200_NUM_VALIDATION = 1_000
B787_3200_NUM_TEST = 1_000
B787_3200_NUM_UNUSED = 4_800

STAGE1_SCHEMA = "rift_sugavanam_ertin_b7873200_stage1_v1"
STAGE1_RECIPE_ID = "sugavanam_ertin_b7873200_isotropic_scatter_v1"
STAGE1_EXECUTION_CONTRACT_LABEL = "rift_sugavanam_ertin_b7873200_stage1_v1"
STAGE1_ACQUISITION_SCHEMA = "rift_sugavanam_ertin_b7873200_acquisition_v1"
STAGE1_BUNDLE_FIELD = "sugavanam_ertin_b7873200_stage1"
STAGE1_FINAL_BUNDLE_FILENAME = "sugavanam_ertin_b7873200_stage1_final_v1.pth.tar"
GENERIC_FINAL_FILENAME = "checkpoint_final.pth.tar"
STAGE1_OUTPUT_DIR = (
    "/storage/scratch1/1/dbao31/rift_homemade_baselines_20260905_v1/"
    "b787_sugavanam_ertin_stage1_v1"
)
STAGE1_BUNDLE_PATH = f"{STAGE1_OUTPUT_DIR}/{STAGE1_FINAL_BUNDLE_FILENAME}"


class Stage1ContractError(ValueError):
    """Raised when a Stage-1 source is not the sealed B7873200 terminal run."""


@dataclass(frozen=True)
class Stage1StructuralAudit:
    """Finite, grid-layout, and cloud-survival facts for the final bundle."""

    epoch: int
    loss: float
    granularity: int
    extent_m: float
    active_count: int
    retained_count: int
    threshold_fraction: float
    threshold_absolute: float
    maximum_magnitude: float

    def as_dict(self) -> dict[str, object]:
        return {
            "epoch": int(self.epoch),
            "loss": float(self.loss),
            "granularity": int(self.granularity),
            "extent_m": float(self.extent_m),
            "active_count": int(self.active_count),
            "retained_count": int(self.retained_count),
            "threshold_fraction": float(self.threshold_fraction),
            "threshold_absolute": float(self.threshold_absolute),
            "maximum_magnitude": float(self.maximum_magnitude),
        }


def _ordered_role(
    roles: Mapping[str, object], name: str, expected_count: int
) -> tuple[int, ...]:
    values = roles.get(name)
    if not isinstance(values, list) or len(values) != expected_count:
        raise Stage1ContractError(
            f"B7873200 role {name!r} must contain exactly {expected_count} ordered IDs"
        )
    normalized: list[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise Stage1ContractError(f"B7873200 role {name!r} contains a non-integer ID")
        index = int(value)
        if not 0 <= index < B787_3200_RESPONSE_SHAPE[0]:
            raise Stage1ContractError(f"B7873200 role {name!r} has out-of-range ID {index}")
        normalized.append(index)
    if len(set(normalized)) != len(normalized):
        raise Stage1ContractError(f"B7873200 role {name!r} contains duplicate IDs")
    return tuple(normalized)


def canonical_sealed_identity(contract: Mapping[str, object]) -> dict[str, object]:
    """Return the exact path-free B7873200 sealed development identity.

    Locations remain ordinary provenance.  The semantic identity is the header,
    ordered PCG64 role partition, and access policy; no source checksum or
    lock is used.
    """

    if isinstance(contract, Mapping) and (
        "dataset_identity" in contract
        or str(contract.get("role_manifest_name", "")).startswith("rift_dataset_")
    ):
        from rift.rift_dataset import collection_contract
        return collection_contract(contract)
    if not isinstance(contract, Mapping):
        raise Stage1ContractError("sealed protocol contract must be a mapping")
    if contract.get("schema") != "rift_npz_sealed_protocol_v1" or contract.get("version") != 1:
        raise Stage1ContractError("B7873200 requires sealed NPZ protocol v1")
    if contract.get("data_format") != "npz":
        raise Stage1ContractError("B7873200 requires NPZ observations")
    if tuple(contract.get("response_shape", ())) != B787_3200_RESPONSE_SHAPE:
        raise Stage1ContractError("B7873200 response header shape is not [10000,16,16,1,600]")
    if contract.get("response_dtype") != B787_3200_RESPONSE_DTYPE:
        raise Stage1ContractError("B7873200 response dtype must be complex64")
    if contract.get("role_manifest_name") != B787_3200_MANIFEST_NAME:
        raise Stage1ContractError("B7873200 requires the canonical interpolation manifest")
    if contract.get("split_strategy") != "fixed_tail_subsampled":
        raise Stage1ContractError("B7873200 requires fixed-tail-subsampled interpolation roles")
    roles = contract.get("role_ids")
    if not isinstance(roles, Mapping):
        raise Stage1ContractError("sealed protocol contract lacks role_ids")
    train = _ordered_role(roles, "train", B787_3200_NUM_TRAIN)
    validation = _ordered_role(roles, "validation", B787_3200_NUM_VALIDATION)
    reserved_test = _ordered_role(roles, "reserved_test", B787_3200_NUM_TEST)
    unused = _ordered_role(roles, "unused", B787_3200_NUM_UNUSED)
    permutation = np.random.Generator(np.random.PCG64(B787_3200_SPLIT_SEED)).permutation(
        B787_3200_RESPONSE_SHAPE[0]
    )
    test_start = B787_3200_RESPONSE_SHAPE[0] - B787_3200_NUM_VALIDATION - B787_3200_NUM_TEST
    validation_start = B787_3200_RESPONSE_SHAPE[0] - B787_3200_NUM_VALIDATION
    expected = {
        "train": tuple(int(value) for value in permutation[:B787_3200_NUM_TRAIN]),
        "unused": tuple(int(value) for value in permutation[B787_3200_NUM_TRAIN:test_start]),
        "reserved_test": tuple(int(value) for value in permutation[test_start:validation_start]),
        "validation": tuple(int(value) for value in permutation[validation_start:]),
    }
    observed = {
        "train": train,
        "validation": validation,
        "reserved_test": reserved_test,
        "unused": unused,
    }
    if observed != expected:
        raise Stage1ContractError("B7873200 role order does not match PCG64(seed=42)")
    if len(set(train + validation + reserved_test + unused)) != B787_3200_RESPONSE_SHAPE[0]:
        raise Stage1ContractError("B7873200 roles are not a complete disjoint partition")
    access = {
        "train_materialized": True,
        "validation_materialized": True,
        "reserved_test_materialized": False,
        "unused_materialized": False,
    }
    if contract.get("response_access") != access:
        raise Stage1ContractError("B7873200 must expose only train and validation responses")
    return {
        "schema": "rift_npz_sealed_protocol_v1",
        "version": 1,
        "data_format": "npz",
        "response_shape": list(B787_3200_RESPONSE_SHAPE),
        "response_dtype": B787_3200_RESPONSE_DTYPE,
        "role_manifest_name": B787_3200_MANIFEST_NAME,
        "split_strategy": "fixed_tail_subsampled",
        "role_ids": {name: list(values) for name, values in observed.items()},
        "response_access": access,
    }


def _canonical_path(path: str | os.PathLike[str]) -> str:
    return os.path.realpath(os.path.abspath(os.fspath(path)))


def load_b7873200_sealed_identity(
    npz_path: str | os.PathLike[str], role_manifest_path: str | os.PathLike[str]
) -> tuple[Mapping[str, object], dict[str, object]]:
    """Preflight the correct archive before any response is materialized."""

    from rift.rift_dataset import collection_manifest, load_object_contract
    if collection_manifest(role_manifest_path):
        arrays, contract = load_object_contract(npz_path, role_manifest_path)
        return arrays, canonical_sealed_identity(contract)
    if _canonical_path(npz_path) != _canonical_path(B787_3200_CANONICAL_NPZ_PATH):
        raise Stage1ContractError(
            "B7873200 requires the canonical sphere10k archive at "
            f"{B787_3200_CANONICAL_NPZ_PATH}"
        )
    # Imported only after the path gate.  The generic loader reads metadata and
    # the response header before it constructs the restricted lazy response
    # capability.
    from train import _load_sealed_npz_protocol_contract

    arrays, contract = _load_sealed_npz_protocol_contract(
        npz_path,
        role_manifest_path,
        num_train=B787_3200_NUM_TRAIN,
        num_val=B787_3200_NUM_VALIDATION,
        num_test=B787_3200_NUM_TEST,
    )
    if arrays.get("response") is not None:
        raise Stage1ContractError("B7873200 header preflight must not materialize response payloads")
    return arrays, canonical_sealed_identity(contract)


def _metadata_identity(meta: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(meta, Mapping):
        raise Stage1ContractError("B7873200 acquisition needs decoded metadata")
    try:
        target_type = str(meta["target_type"]).lower()
        experiment = str(meta["experiment"]).lower()
        carrier = float(meta["radar_fc_hz"])
        bandwidth = float(meta["radar_bandwidth_hz"])
        samples = int(meta["num_adc_samples"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage1ContractError("B7873200 metadata is incomplete") from exc
    from rift.rift_dataset import metadata_object_id
    source_object = metadata_object_id(meta)
    if experiment != "sphere10k":
        raise Stage1ContractError("Requires RIFT dataset sphere10k metadata")
    if not (
        math.isclose(carrier, 10.0e9, rel_tol=0.0, abs_tol=1.0)
        and math.isclose(bandwidth, 3.0e9, rel_tol=0.0, abs_tol=1.0)
        and samples == 600
    ):
        raise Stage1ContractError("B7873200 requires the 10 GHz / 3 GHz / 600-bin acquisition")
    return {
        "target_type": target_type,
        **({"target_id": source_object} if source_object != "b787" else {}),
        "experiment": "sphere10k",
        "radar_fc_hz": carrier,
        "radar_bandwidth_hz": bandwidth,
        "num_adc_samples": samples,
    }


def _finite_float64_array(value: object, shape: Sequence[int], label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64) or tuple(array.shape) != tuple(shape):
        raise Stage1ContractError(f"B7873200 {label} must be float64 with shape {tuple(shape)}")
    if not np.isfinite(array).all():
        raise Stage1ContractError(f"B7873200 {label} must be finite")
    return np.ascontiguousarray(array, dtype=np.float64)


def build_b7873200_acquisition_identity(
    arrays: Mapping[str, object], sealed_identity: Mapping[str, object]
) -> dict[str, object]:
    """Capture exact physics coordinates without reading a radar response."""

    if not isinstance(arrays, Mapping) or arrays.get("response") is not None:
        raise Stage1ContractError("acquisition identity must be built before response materialization")
    sealed = canonical_sealed_identity(sealed_identity)
    metadata = _metadata_identity(arrays.get("meta", {}))
    roles = sealed["role_ids"]
    authorized = [*roles["train"], *roles["validation"]]
    rx = _finite_float64_array(arrays.get("rx_pos"), (10_000, 16, 3), "rx_pos")
    tx = _finite_float64_array(arrays.get("tx_pos"), (10_000, 16, 3), "tx_pos")
    frequency = (
        metadata["radar_fc_hz"]
        - metadata["radar_bandwidth_hz"] / 2.0
        + np.arange(metadata["num_adc_samples"], dtype=np.float64)
        * (metadata["radar_bandwidth_hz"] / metadata["num_adc_samples"])
    )
    return {
        "schema": STAGE1_ACQUISITION_SCHEMA,
        "metadata": metadata,
        "frequency_hz": np.ascontiguousarray(frequency, dtype=np.float64),
        "authorized_view_ids": [int(value) for value in authorized],
        "rx_pos_m": np.ascontiguousarray(rx[np.asarray(authorized, dtype=np.int64)]),
        "tx_pos_m": np.ascontiguousarray(tx[np.asarray(authorized, dtype=np.int64)]),
        "response_payload_materialized": False,
    }


def _array_equal(left: object, right: object, label: str) -> None:
    left_array = np.asarray(_to_numpy(left, label))
    right_array = np.asarray(_to_numpy(right, label))
    if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
        raise Stage1ContractError(f"B7873200 {label} shape or dtype changed")
    if not (np.isfinite(left_array).all() and np.array_equal(left_array, right_array)):
        raise Stage1ContractError(f"B7873200 {label} changed or is non-finite")


def validate_b7873200_acquisition_identity(
    saved: Mapping[str, object], expected: Mapping[str, object]
) -> None:
    if not isinstance(saved, Mapping):
        raise Stage1ContractError("B7873200 Stage-1 provenance lacks acquisition identity")
    if saved.get("schema") != STAGE1_ACQUISITION_SCHEMA:
        raise Stage1ContractError("B7873200 acquisition schema changed")
    if saved.get("metadata") != expected.get("metadata"):
        raise Stage1ContractError("B7873200 acquisition metadata changed")
    if saved.get("authorized_view_ids") != expected.get("authorized_view_ids"):
        raise Stage1ContractError("B7873200 authorized view IDs changed")
    if saved.get("response_payload_materialized") is not False:
        raise Stage1ContractError("B7873200 provenance must not contain response payloads")
    for key in ("frequency_hz", "rx_pos_m", "tx_pos_m"):
        _array_equal(saved.get(key), expected.get(key), key)


def default_stage1_recipe(
    sealed_identity: Mapping[str, object], acquisition_identity: Mapping[str, object]
) -> dict[str, object]:
    """Return the declared v1 Stage-1 recipe, not a mutable legacy default.

    These values are intentionally explicit.  A scientific change is a new
    versioned recipe rather than an in-place rewrite of an existing final
    bundle.  Operational launch scheduling is deliberately not part of this
    semantic recipe.
    """

    sealed = canonical_sealed_identity(sealed_identity)
    expected_acquisition = copy.deepcopy(dict(acquisition_identity))
    validate_b7873200_acquisition_identity(expected_acquisition, expected_acquisition)
    return {
        "schema": STAGE1_SCHEMA,
        "recipe_id": STAGE1_RECIPE_ID,
        "method": "Sugavanam--Ertin",
        "stage": "stage1_scattering_field",
        "kind": "isotropic_complex_voxel_grid",
        "ground_truth_geometry_used": False,
        "sealed_protocol_identity": sealed,
        "acquisition_identity": expected_acquisition,
        "observation": {
            "data_format": "npz",
            "all_coordinates": {"tx": 16, "rx": 16, "chirps": 1, "frequencies": 600},
            "num_freq_wanted": 600,
            "frequency_policy": "all_600",
            "pair_policy": "all_16x16",
            "forward_operator": "range",
            "range_model": "product",
            "phase_sign": -1.0,
            "compute_dtype": "float64",
            "operator_options": {"point_chunk": 65536, "pair_chunk": 64},
        },
        "scene": {
            "scene_repr": "grid",
            "complex_scalar_per_cell": True,
            "extent_m": 0.15,
            "granularity": 48,
            "coordinates": "cell_centred_uniform_xyz",
            "sh": "absent",
            "adaptive_capacity": False,
            "position_updates": False,
        },
        "fit": {
            "loss": "complex",
            "l1_weight": 3.0e-7,
            "gain": {"enabled": True, "kind": "global_complex"},
            "epochs": 150,
            "seed": 42,
            "optimizer": {
                "name": "AdamW",
                "lr": 3.0e-3,
                "betas": [0.9, 0.999],
                "eps": 1.0e-20,
                "weight_decay": 0.0,
            },
            "scheduler": {"name": "CosineAnnealingWarmRestarts", "t0": 10, "t_mult": 2, "eta_min": 1.0e-6},
            "bp_init": 400,
            "pruning": {
                "enabled": False,
                "prune_every": 0,
                "prune_mode": "mass",
                "prune_threshold": 0.0,
                "prune_criterion": "energy",
                "prune_start_epoch": 0,
                "prune_target_active": 0,
                "prune_end_epoch": 0,
                "prune_min_active": 0,
            },
            "selection": {"metric": "val", "tie_policy": "earliest"},
        },
        "stage2_cloud_extraction": {
            "threshold_fraction": 0.15,
            "max_points": 20000,
            "normal_radius_policy": "three_stage1_voxel_pitches",
        },
        "source_selection": {
            "required_role": "checkpoint_final",
            "terminal_epoch": 150,
            "stage2_uses": "final_not_best_or_latest",
        },
    }


def expected_generic_execution_contract(recipe: Mapping[str, object]) -> dict[str, object]:
    """The compact generic-trainer record Stage 1 requires in every final.

    ``train.py`` derives this record from the parsed command and stores it in
    the checkpoint.  Keeping the expected projection here lets the isolated
    bundle reject a generic final that merely *looks* like this run by filename
    or grid shape.  It deliberately records no response payload or hash.
    """

    observation = recipe["observation"]
    scene = recipe["scene"]
    fit = recipe["fit"]
    pruning = fit["pruning"]
    optimizer = fit["optimizer"]
    scheduler = fit["scheduler"]
    selection = fit["selection"]
    return {
        "schema": "rift_checkpoint_execution_contract_v1",
        "label": STAGE1_EXECUTION_CONTRACT_LABEL,
        "observation": {
            "data_format": "npz",
            "sealed_npz_protocol": True,
            "num_train": B787_3200_NUM_TRAIN,
            "num_validation": B787_3200_NUM_VALIDATION,
            "num_reserved_test": B787_3200_NUM_TEST,
            "num_freq_wanted": int(observation["num_freq_wanted"]),
            "validation_cap_axis": None,
            "validation_from_tail": False,
        },
        "scene": {
            "representation": scene["scene_repr"],
            "extent_m": float(scene["extent_m"]),
            "granularity": int(scene["granularity"]),
            "initial_scale": 0.0,
            "backprojection_views": int(fit["bp_init"]),
            "shell_init_radius_m": 0.0,
            "normalize_scene_scale": False,
            "sh_max_degree": 10,
            "sh_init_degree": 0,
        },
        "physics": {
            "forward_operator": observation["forward_operator"],
            "range_model": observation["range_model"],
            "compute_dtype": observation["compute_dtype"],
            "phase_sign": float(observation["phase_sign"]),
            "coordinate_source": "npz_per_view_positions",
            "num_rx": 16,
            "num_tx": 16,
            "point_chunk": int(observation["operator_options"]["point_chunk"]),
            "pair_chunk": int(observation["operator_options"]["pair_chunk"]),
        },
        "fit": {
            "epochs": int(fit["epochs"]),
            "loss": fit["loss"],
            "step_every": 1,
            "clip_grad_norm": 0.0,
            "learn_global_gain": bool(fit["gain"]["enabled"]),
            "learning_rate": float(optimizer["lr"]),
            "weight_decay": float(optimizer["weight_decay"]),
            "l1_weight": float(fit["l1_weight"]),
            "sh_smooth_weight": 0.0,
            "regularizer_normalization": "active_mean",
            "adam_eps": float(optimizer["eps"]),
            "checkpoint_metric": selection["metric"],
            "seed": int(fit["seed"]),
            "scheduler": {
                "t0": int(scheduler["t0"]),
                "t_mult": int(scheduler["t_mult"]),
                "eta_min": float(scheduler["eta_min"]),
            },
            "pruning": {
                "every": int(pruning["prune_every"]),
                "threshold": float(pruning["prune_threshold"]),
                "criterion": pruning["prune_criterion"],
                "start_epoch": int(pruning["prune_start_epoch"]),
                "mode": pruning["prune_mode"],
                "target_active": int(pruning["prune_target_active"]),
                "end_epoch": int(pruning["prune_end_epoch"]),
                "min_active": int(pruning["prune_min_active"]),
            },
            "view_weight_alpha": 0.0,
            "view_weight_max_ratio": 0.0,
            "magnitude_weight": 0.0,
            "magnitude_warmup_epochs": 0,
            "resume_requires_full_state": True,
            "optimizer": {
                "name": optimizer["name"],
                "scene_learning_rate": float(optimizer["lr"]),
                "gain_learning_rate": float(optimizer["lr"]),
                "betas": [float(value) for value in optimizer["betas"]],
                "eps": float(optimizer["eps"]),
                "weight_decay": float(optimizer["weight_decay"]),
            },
        },
    }


def validate_stage1_recipe(
    recipe: Mapping[str, object],
    sealed_identity: Mapping[str, object],
    acquisition_identity: Mapping[str, object],
) -> dict[str, object]:
    """Reject a Stage-1 recipe whose scientific semantics differ from v1."""

    expected = default_stage1_recipe(sealed_identity, acquisition_identity)
    if not isinstance(recipe, Mapping):
        raise Stage1ContractError("B7873200 Stage-1 recipe must be a mapping")
    # NumPy tensors in the acquisition record make generic dict equality
    # ambiguous, so validate it separately and compare all remaining mappings.
    normalized = copy.deepcopy(dict(recipe))
    acquisition = normalized.pop("acquisition_identity", None)
    expected_without_acquisition = copy.deepcopy(expected)
    expected_acquisition = expected_without_acquisition.pop("acquisition_identity")
    validate_b7873200_acquisition_identity(acquisition, expected_acquisition)
    if normalized != expected_without_acquisition:
        raise Stage1ContractError("B7873200 Stage-1 recipe differs from the frozen v1 recipe")
    return expected


def _to_numpy(value: object, label: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        return np.asarray(value)
    except Exception as exc:  # pragma: no cover - defensive foreign tensor bridge
        raise Stage1ContractError(f"{label} is not array-like") from exc


def expected_cell_centred_grid(granularity: int, extent_m: float, *, dtype: np.dtype) -> np.ndarray:
    if granularity < 2 or not math.isfinite(extent_m) or extent_m <= 0.0:
        raise Stage1ContractError("invalid B7873200 Stage-1 grid geometry")
    pitch = 2.0 * float(extent_m) / int(granularity)
    axis = np.linspace(
        -float(extent_m) + pitch / 2.0,
        float(extent_m) - pitch / 2.0,
        int(granularity),
        dtype=dtype,
    )
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)


def _finite_scalar(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise Stage1ContractError(f"{label} must be a finite scalar")
    try:
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Stage1ContractError(f"{label} must be a finite scalar") from exc
    if not math.isfinite(scalar) or (positive and scalar <= 0.0):
        raise Stage1ContractError(f"{label} must be finite" + (" and positive" if positive else ""))
    return scalar


def _bool_grid(value: object, shape: tuple[int, int, int], label: str) -> np.ndarray:
    array = _to_numpy(value, label)
    if tuple(array.shape) != shape or array.dtype != np.dtype(bool):
        raise Stage1ContractError(f"{label} must be a boolean {shape} grid")
    return np.asarray(array, dtype=bool)


def _float_grid(value: object, shape: tuple[int, int, int], label: str) -> np.ndarray:
    array = _to_numpy(value, label)
    if tuple(array.shape) != shape or array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise Stage1ContractError(f"{label} must be float32/float64 with shape {shape}")
    if not np.isfinite(array).all():
        raise Stage1ContractError(f"{label} contains non-finite values")
    return np.asarray(array)


def _finite_scalar_tensor(value: object, label: str, *, boolean: bool = False) -> object:
    array = _to_numpy(value, label)
    if array.shape != ():
        raise Stage1ContractError(f"{label} must be scalar")
    if boolean:
        if array.dtype != np.dtype(bool):
            raise Stage1ContractError(f"{label} must be boolean")
        return bool(array.item())
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise Stage1ContractError(f"{label} must be float32 or float64")
    scalar = float(array.item())
    if not math.isfinite(scalar):
        raise Stage1ContractError(f"{label} must be finite")
    return scalar


def _finite_state_tree(value: object, label: str) -> None:
    """Fail closed on non-finite optimizer state without assuming Torch internals."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            _finite_state_tree(child, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_state_tree(child, f"{label}[{index}]")
        return
    if isinstance(value, (bool, str, type(None))):
        return
    if isinstance(value, (int, np.integer)):
        return
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise Stage1ContractError(f"{label} contains a non-finite scalar")
        return
    array = _to_numpy(value, label)
    if array.dtype.kind not in "fci" or not np.isfinite(array).all():
        raise Stage1ContractError(f"{label} contains non-finite or unsupported optimizer state")


def _validate_generic_rng_state(payload: object) -> None:
    """Require the exact v2 recovery payload for this new opt-in lane."""

    if not isinstance(payload, Mapping):
        raise Stage1ContractError("Stage-1 latest lacks its complete RNG payload")
    allowed = {"version", "python", "numpy", "torch_cpu", "freq_cpu", "torch_cuda_all"}
    if set(payload).difference(allowed) or payload.get("version") != 2:
        raise Stage1ContractError("Stage-1 RNG payload schema changed")
    required = {"python", "numpy", "torch_cpu", "freq_cpu"}
    if required.difference(payload):
        raise Stage1ContractError("Stage-1 latest lacks RNG generators needed for exact recovery")
    try:
        random.Random().setstate(payload["python"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise Stage1ContractError("Stage-1 Python RNG state is invalid") from exc
    numpy_state = payload["numpy"]
    expected_numpy_keys = {"format", "algorithm", "keys", "position", "has_gauss", "cached_gaussian"}
    if not isinstance(numpy_state, Mapping) or set(numpy_state) != expected_numpy_keys:
        raise Stage1ContractError("Stage-1 NumPy RNG state is invalid")
    if numpy_state.get("format") != "numpy_randomstate_v1":
        raise Stage1ContractError("Stage-1 NumPy RNG format changed")
    keys = _to_numpy(numpy_state.get("keys"), "Stage-1 NumPy RNG keys")
    if keys.dtype != np.dtype(np.uint32) or keys.shape != (624,):
        raise Stage1ContractError("Stage-1 NumPy RNG keys changed")
    position = numpy_state.get("position")
    has_gauss = numpy_state.get("has_gauss")
    cached = numpy_state.get("cached_gaussian")
    if (
        isinstance(position, bool)
        or not isinstance(position, (int, np.integer))
        or not 0 <= int(position) <= 624
        or isinstance(has_gauss, bool)
        or not isinstance(has_gauss, (int, np.integer))
        or int(has_gauss) not in (0, 1)
    ):
        raise Stage1ContractError("Stage-1 NumPy RNG cursor changed")
    if not isinstance(numpy_state.get("algorithm"), str) or not math.isfinite(_finite_scalar(cached, "Stage-1 NumPy RNG cache")):
        raise Stage1ContractError("Stage-1 NumPy RNG metadata changed")
    try:
        np.random.RandomState().set_state(
            (
                numpy_state["algorithm"],
                keys.copy(),
                int(position),
                int(has_gauss),
                float(cached),
            )
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise Stage1ContractError("Stage-1 NumPy RNG state cannot be restored") from exc
    for name in ("torch_cpu", "freq_cpu"):
        tensor = _to_numpy(payload[name], f"Stage-1 {name} RNG state")
        if tensor.dtype != np.dtype(np.uint8) or tensor.ndim != 1 or tensor.size == 0:
            raise Stage1ContractError(f"Stage-1 {name} RNG state is invalid")
    cuda = payload.get("torch_cuda_all")
    if cuda is not None:
        if not isinstance(cuda, (list, tuple)):
            raise Stage1ContractError("Stage-1 CUDA RNG state is invalid")
        for index, item in enumerate(cuda):
            tensor = _to_numpy(item, f"Stage-1 CUDA RNG state {index}")
            if tensor.dtype != np.dtype(np.uint8) or tensor.ndim != 1 or tensor.size == 0:
                raise Stage1ContractError("Stage-1 CUDA RNG state is invalid")


def _validate_generic_runtime_state(
    state: Mapping[str, object], recipe: Mapping[str, object], *, terminal: bool
) -> None:
    """Validate actual generic state without treating scheduled LR as static."""

    expected_contract = expected_generic_execution_contract(recipe)
    if state.get("execution_contract") != expected_contract:
        raise Stage1ContractError(
            "Stage-1 generic final lacks the exact derived B7873200 execution contract"
        )

    fit = recipe["fit"]
    optimizer_recipe = fit["optimizer"]
    epoch = state.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, (int, np.integer)):
        raise Stage1ContractError("Stage-1 generic checkpoint epoch is invalid")
    epoch = int(epoch)
    terminal_epoch = int(fit["epochs"])
    if terminal:
        if epoch != terminal_epoch:
            raise Stage1ContractError("Stage-1 generic final did not reach the terminal epoch")
    elif not 0 < epoch < terminal_epoch:
        raise Stage1ContractError("Stage-1 latest does not name a valid pre-terminal epoch")
    _finite_scalar(state.get("loss"), "Stage-1 checkpoint selector loss")
    gain_state = state.get("gain_state_dict")
    if not isinstance(gain_state, Mapping) or set(gain_state) != {"log_mag", "phase", "initialized"}:
        raise Stage1ContractError("Stage-1 final must retain the global complex-gain state")
    _finite_scalar_tensor(gain_state["log_mag"], "Stage-1 gain log magnitude")
    _finite_scalar_tensor(gain_state["phase"], "Stage-1 gain phase")
    _finite_scalar_tensor(gain_state["initialized"], "Stage-1 gain initialized", boolean=True)

    optimizer_state = state.get("optimizer_state_dict")
    if not isinstance(optimizer_state, Mapping):
        raise Stage1ContractError("Stage-1 final lacks optimizer recovery state")
    groups = optimizer_state.get("param_groups")
    if not isinstance(groups, list) or len(groups) != 2:
        raise Stage1ContractError("Stage-1 final must retain scene and gain optimizer groups")
    expected_group_ids = ((0, 1), (2, 3))
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            raise Stage1ContractError("Stage-1 optimizer group is malformed")
        parameters = group.get("params")
        if not isinstance(parameters, list) or len(parameters) != 2 or any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            for value in parameters
        ) or tuple(int(value) for value in parameters) != expected_group_ids[index]:
            raise Stage1ContractError("Stage-1 optimizer group parameter layout changed")
        for key, expected in (
            ("eps", float(optimizer_recipe["eps"])),
            ("weight_decay", float(optimizer_recipe["weight_decay"])),
        ):
            actual = _finite_scalar(group.get(key), f"Stage-1 optimizer[{index}].{key}")
            if actual != expected:
                raise Stage1ContractError(f"Stage-1 optimizer {key} changed")
        _finite_scalar(group.get("lr"), f"Stage-1 optimizer[{index}].lr", positive=True)
        if "initial_lr" in group and _finite_scalar(
            group["initial_lr"], f"Stage-1 optimizer[{index}].initial_lr"
        ) != float(optimizer_recipe["lr"]):
            raise Stage1ContractError("Stage-1 optimizer initial learning rate changed")
        betas = group.get("betas")
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise Stage1ContractError("Stage-1 optimizer betas are malformed")
        if tuple(float(value) for value in betas) != tuple(float(value) for value in optimizer_recipe["betas"]):
            raise Stage1ContractError("Stage-1 optimizer betas changed")
    states = optimizer_state.get("state")
    if not isinstance(states, Mapping) or set(states) != {0, 1, 2, 3}:
        raise Stage1ContractError("Stage-1 optimizer moments are incomplete")
    model_state = state.get("model_state_dict")
    if not isinstance(model_state, Mapping):
        raise Stage1ContractError("Stage-1 checkpoint lacks model state for AdamW validation")
    references = (
        _to_numpy(model_state.get("w_re"), "Stage-1 AdamW w_re reference"),
        _to_numpy(model_state.get("w_im"), "Stage-1 AdamW w_im reference"),
        _to_numpy(gain_state["log_mag"], "Stage-1 AdamW gain log magnitude reference"),
        _to_numpy(gain_state["phase"], "Stage-1 AdamW gain phase reference"),
    )
    expected_adam_step = epoch * B787_3200_NUM_TRAIN
    for parameter_id, reference in enumerate(references):
        moment = states[parameter_id]
        if not isinstance(moment, Mapping) or set(moment) != {"step", "exp_avg", "exp_avg_sq"}:
            raise Stage1ContractError("Stage-1 AdamW moment schema changed")
        step = _finite_scalar_tensor(moment["step"], f"Stage-1 AdamW step {parameter_id}")
        if not math.isclose(step, float(expected_adam_step), rel_tol=0.0, abs_tol=1.0e-4):
            raise Stage1ContractError("Stage-1 AdamW moment progress changed")
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = _to_numpy(moment[name], f"Stage-1 AdamW {name} {parameter_id}")
            if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
                raise Stage1ContractError("Stage-1 AdamW moment shape or dtype changed")
            if not np.isfinite(tensor).all():
                raise Stage1ContractError("Stage-1 AdamW moment is non-finite")
    _finite_state_tree(optimizer_state, "Stage-1 optimizer")

    scheduler_state = state.get("scheduler_state_dict")
    if not isinstance(scheduler_state, Mapping):
        raise Stage1ContractError("Stage-1 final lacks scheduler recovery state")
    scheduler = fit["scheduler"]
    for key, expected in (
        ("T_0", int(scheduler["t0"])),
        ("T_mult", int(scheduler["t_mult"])),
        ("eta_min", float(scheduler["eta_min"])),
    ):
        if key not in scheduler_state:
            raise Stage1ContractError(f"Stage-1 scheduler lacks {key}")
        actual = scheduler_state[key]
        if isinstance(expected, int):
            if (
                isinstance(actual, (bool, np.bool_))
                or not isinstance(actual, (int, np.integer))
                or int(actual) != expected
            ):
                raise Stage1ContractError(f"Stage-1 scheduler {key} changed")
        elif _finite_scalar(actual, f"Stage-1 scheduler {key}") != expected:
            raise Stage1ContractError(f"Stage-1 scheduler {key} changed")
    last_epoch = scheduler_state.get("last_epoch")
    if (
        isinstance(last_epoch, (bool, np.bool_))
        or not isinstance(last_epoch, (int, np.integer))
        or int(last_epoch) != epoch
    ):
        raise Stage1ContractError("Stage-1 scheduler progress changed")
    base_lrs = scheduler_state.get("base_lrs")
    if not isinstance(base_lrs, list) or len(base_lrs) != len(groups):
        raise Stage1ContractError("Stage-1 scheduler base learning rates changed")
    last_lrs = scheduler_state.get("_last_lr")
    if not isinstance(last_lrs, list) or len(last_lrs) != len(groups):
        raise Stage1ContractError("Stage-1 scheduler current learning rates changed")
    for index, (group, base_lr, current_lr) in enumerate(zip(groups, base_lrs, last_lrs)):
        if not math.isclose(
            _finite_scalar(base_lr, f"Stage-1 scheduler base lr {index}"),
            float(optimizer_recipe["lr"]),
            rel_tol=0.0,
            abs_tol=1.0e-18,
        ):
            raise Stage1ContractError("Stage-1 scheduler base learning rate changed")
        current = _finite_scalar(current_lr, f"Stage-1 scheduler current lr {index}", positive=True)
        if not math.isclose(
            _finite_scalar(group.get("lr"), f"Stage-1 optimizer current lr {index}", positive=True),
            current,
            rel_tol=0.0,
            abs_tol=1.0e-18,
        ):
            raise Stage1ContractError("Stage-1 optimizer/scheduler learning rate mismatch")
    t_i = scheduler_state.get("T_i")
    t_cur = scheduler_state.get("T_cur")
    if (
        isinstance(t_i, bool)
        or not isinstance(t_i, (int, np.integer))
        or int(t_i) <= 0
        or isinstance(t_cur, bool)
        or not isinstance(t_cur, (int, np.integer))
        or not 0 <= int(t_cur) < int(t_i)
    ):
        raise Stage1ContractError("Stage-1 scheduler cycle state changed")
    expected_t_i = int(scheduler["t0"])
    expected_t_cur = epoch
    while expected_t_cur >= expected_t_i:
        expected_t_cur -= expected_t_i
        expected_t_i *= int(scheduler["t_mult"])
    if int(t_i) != expected_t_i or int(t_cur) != expected_t_cur:
        raise Stage1ContractError("Stage-1 scheduler cycle does not match completed epochs")
    expected_lr = float(scheduler["eta_min"]) + (
        float(optimizer_recipe["lr"]) - float(scheduler["eta_min"])
    ) * (1.0 + math.cos(math.pi * expected_t_cur / expected_t_i)) / 2.0
    for index, current_lr in enumerate(last_lrs):
        if not math.isclose(
            _finite_scalar(current_lr, f"Stage-1 scheduler current lr {index}", positive=True),
            expected_lr,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise Stage1ContractError("Stage-1 scheduler current learning rate changed")
    _finite_state_tree(scheduler_state, "Stage-1 scheduler")


def _validate_generic_model_recovery_state(
    state: Mapping[str, object], recipe: Mapping[str, object]
) -> None:
    """Require the full fixed-grid model payload before a Stage-1 resume.

    The generic trainer can load historical model-only checkpoints.  This new
    lane cannot: its wrapper has already promised an exact continuation.  The
    check deliberately verifies structure and finite values, not a terminal
    pruning/retention threshold; a legitimate in-progress grid may not yet
    satisfy the final bundle's cloud-survival policy.
    """

    scene = recipe["scene"]
    model_state = state.get("model_state_dict")
    if not isinstance(model_state, Mapping) or set(model_state) != {
        "w_re", "w_im", "active_mask", "grid_positions"
    }:
        raise Stage1ContractError("Stage-1 latest lacks the full fixed-grid model state")
    granularity = int(scene["granularity"])
    shape = (granularity, granularity, granularity)
    w_re = _float_grid(model_state.get("w_re"), shape, "Stage-1 latest w_re")
    w_im = _float_grid(model_state.get("w_im"), shape, "Stage-1 latest w_im")
    if w_re.dtype != w_im.dtype:
        raise Stage1ContractError("Stage-1 latest complex grid components have different dtypes")
    active = _bool_grid(model_state.get("active_mask"), shape, "Stage-1 latest active_mask")
    if not active.any():
        raise Stage1ContractError("Stage-1 latest has no active grid cells")
    positions = _to_numpy(model_state.get("grid_positions"), "Stage-1 latest grid_positions")
    if positions.dtype not in (np.dtype(np.float32), np.dtype(np.float64)) or positions.shape != (*shape, 3):
        raise Stage1ContractError("Stage-1 latest grid_positions has the wrong dtype or shape")
    if not np.isfinite(positions).all():
        raise Stage1ContractError("Stage-1 latest grid_positions contains non-finite values")
    expected = expected_cell_centred_grid(granularity, float(scene["extent_m"]), dtype=positions.dtype)
    if not np.allclose(positions, expected, rtol=0.0, atol=1.0e-7):
        raise Stage1ContractError("Stage-1 latest grid_positions changed")


def validate_b7873200_stage1_recovery_state(
    state: Mapping[str, object], recipe: Mapping[str, object]
) -> None:
    """Validate an explicit, exact-continuation Stage-1 latest checkpoint.

    This is intentionally stricter than the generic trainer's historical
    resume path.  The corrected B7873200 lane requires model, AdamW, global
    gain, scheduler, and all RNG generators to be present before the manager
    is allowed to invoke ``--resume``.
    """

    if not isinstance(state, Mapping):
        raise Stage1ContractError("Stage-1 latest checkpoint must be a mapping")
    _validate_generic_runtime_state(state, recipe, terminal=False)
    _validate_generic_model_recovery_state(state, recipe)
    sealed = recipe["sealed_protocol_identity"]
    if canonical_sealed_identity(state.get("sealed_npz_protocol_contract", {})) != sealed:
        raise Stage1ContractError("Stage-1 latest sealed role identity changed")
    _validate_generic_rng_state(state.get("rng_state"))


def _validate_generic_final_state(
    state: Mapping[str, object], recipe: Mapping[str, object]
) -> Stage1StructuralAudit:
    if not isinstance(state, Mapping):
        raise Stage1ContractError("generic Stage-1 final checkpoint must be a mapping")
    fit = recipe["fit"]
    scene = recipe["scene"]
    observation = recipe["observation"]
    source = recipe["source_selection"]
    sealed = recipe["sealed_protocol_identity"]
    _validate_generic_runtime_state(state, recipe, terminal=True)
    expected_epoch = int(fit["epochs"])
    if int(state.get("epoch", -1)) != expected_epoch:
        raise Stage1ContractError("Stage-1 final checkpoint does not reach its terminal epoch")
    loss = _finite_scalar(state.get("loss"), "Stage-1 final loss", positive=True)
    if state.get("scene_repr") != scene["scene_repr"]:
        raise Stage1ContractError("Stage-1 final checkpoint is not an isotropic grid")
    if state.get("range_model") != observation["range_model"]:
        raise Stage1ContractError("Stage-1 final checkpoint range model changed")
    if not math.isclose(_finite_scalar(state.get("extent"), "Stage-1 extent", positive=True), float(scene["extent_m"]), rel_tol=0.0, abs_tol=1.0e-12):
        raise Stage1ContractError("Stage-1 final checkpoint extent changed")
    if int(state.get("granularity", -1)) != int(scene["granularity"]):
        raise Stage1ContractError("Stage-1 final checkpoint granularity changed")
    if not math.isclose(_finite_scalar(state.get("l1_weight"), "Stage-1 L1 weight"), float(fit["l1_weight"]), rel_tol=0.0, abs_tol=1.0e-18):
        raise Stage1ContractError("Stage-1 final checkpoint L1 weight changed")
    if _finite_scalar(state.get("adam_eps"), "Stage-1 Adam epsilon", positive=True) != float(fit["optimizer"]["eps"]):
        raise Stage1ContractError("Stage-1 final checkpoint Adam epsilon changed")
    if state.get("adaptive_capacity_v2") not in (False, None):
        raise Stage1ContractError("B7873200 Stage-1 must not use adaptive capacity")
    if state.get("val_cap_axis") is not None:
        raise Stage1ContractError("B7873200 Stage-1 validation must be interpolation, not a cap")
    if canonical_sealed_identity(state.get("sealed_npz_protocol_contract", {})) != sealed:
        raise Stage1ContractError("Stage-1 final checkpoint sealed role identity changed")
    model_state = state.get("model_state_dict")
    if not isinstance(model_state, Mapping) or set(model_state) != {
        "w_re", "w_im", "active_mask", "grid_positions"
    }:
        raise Stage1ContractError("Stage-1 final checkpoint lacks model_state_dict")
    granularity = int(scene["granularity"])
    shape = (granularity, granularity, granularity)
    w_re = _float_grid(model_state.get("w_re"), shape, "Stage-1 w_re")
    w_im = _float_grid(model_state.get("w_im"), shape, "Stage-1 w_im")
    if w_re.dtype != w_im.dtype:
        raise Stage1ContractError("Stage-1 complex grid components have different dtypes")
    active = _bool_grid(model_state.get("active_mask"), shape, "Stage-1 active_mask")
    if not active.any():
        raise Stage1ContractError("Stage-1 final checkpoint has no active cells")
    positions = _to_numpy(model_state.get("grid_positions"), "Stage-1 grid_positions")
    expected_positions = expected_cell_centred_grid(granularity, float(scene["extent_m"]), dtype=positions.dtype)
    if positions.dtype not in (np.dtype(np.float32), np.dtype(np.float64)) or positions.shape != (*shape, 3):
        raise Stage1ContractError("Stage-1 grid_positions has the wrong dtype or shape")
    if not np.isfinite(positions).all() or not np.allclose(
        positions, expected_positions, rtol=0.0, atol=2.0e-7 if positions.dtype == np.float32 else 1.0e-12
    ):
        raise Stage1ContractError("Stage-1 grid_positions is not the declared cell-centred lattice")
    magnitude = np.hypot(w_re.astype(np.float64), w_im.astype(np.float64))
    active_magnitude = magnitude[active]
    maximum = float(active_magnitude.max())
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise Stage1ContractError("Stage-1 final checkpoint has no positive active scattering magnitude")
    threshold_fraction = _finite_scalar(recipe["stage2_cloud_extraction"]["threshold_fraction"], "cloud threshold", positive=True)
    if threshold_fraction >= 1.0:
        raise Stage1ContractError("cloud threshold fraction must be below one")
    threshold = threshold_fraction * maximum
    retained = int(np.count_nonzero(active & (magnitude >= threshold)))
    if retained < 3:
        raise Stage1ContractError("Stage-1 final checkpoint retains fewer than three scattering centres")
    if source.get("required_role") != "checkpoint_final" or int(source.get("terminal_epoch", -1)) != expected_epoch:
        raise Stage1ContractError("Stage-1 recipe source-selection contract changed")
    return Stage1StructuralAudit(
        epoch=expected_epoch,
        loss=loss,
        granularity=granularity,
        extent_m=float(scene["extent_m"]),
        active_count=int(active.sum()),
        retained_count=retained,
        threshold_fraction=threshold_fraction,
        threshold_absolute=threshold,
        maximum_magnitude=maximum,
    )


def build_stage1_final_bundle(
    generic_final_state: Mapping[str, object],
    recipe: Mapping[str, object],
    *,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Package a validated generic terminal state as the new Stage-1 source."""

    sealed = canonical_sealed_identity(recipe.get("sealed_protocol_identity", {}))
    acquisition = recipe.get("acquisition_identity")
    expected_recipe = validate_stage1_recipe(recipe, sealed, acquisition)
    audit = _validate_generic_final_state(generic_final_state, expected_recipe)
    execution_contract = expected_generic_execution_contract(expected_recipe)
    record = {
        "schema": STAGE1_SCHEMA,
        "recipe_id": STAGE1_RECIPE_ID,
        "role": "checkpoint_final",
        "sealed_protocol_identity": sealed,
        "acquisition_identity": copy.deepcopy(expected_recipe["acquisition_identity"]),
        "stage1_recipe": expected_recipe,
        "generic_execution_contract": copy.deepcopy(execution_contract),
        "structural_audit": audit.as_dict(),
        "provenance": {
            "canonical_npz_path": B787_3200_CANONICAL_NPZ_PATH,
            "role_manifest_path": B787_3200_CANONICAL_MANIFEST_PATH,
            "generic_final_filename": GENERIC_FINAL_FILENAME,
        },
    }
    if provenance is not None:
        if not isinstance(provenance, Mapping):
            raise Stage1ContractError("Stage-1 ordinary provenance must be a mapping")
        record["provenance"].update(copy.deepcopy(dict(provenance)))
    return {"generic_final_state": copy.deepcopy(dict(generic_final_state)), STAGE1_BUNDLE_FIELD: record}


def _torch_load(path: str | os.PathLike[str]) -> Mapping[str, object]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - local static environment has no torch
        raise RuntimeError("loading a Stage-1 bundle requires PyTorch") from exc
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6
        loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, Mapping):
        raise Stage1ContractError("Stage-1 bundle file must contain a mapping")
    return loaded


def validate_b7873200_stage1_final(
    final_path: str | os.PathLike[str],
    *,
    expected_recipe: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Load and fail closed unless ``final_path`` is the new terminal bundle."""

    final_path = os.fspath(final_path)
    if Path(final_path).name != STAGE1_FINAL_BUNDLE_FILENAME:
        raise Stage1ContractError(
            "Stage-2 accepts only the named B7873200 Stage-1 final bundle, never best/latest/generic finals"
        )
    bundle = _torch_load(final_path)
    state = bundle.get("generic_final_state")
    record = bundle.get(STAGE1_BUNDLE_FIELD)
    if not isinstance(state, Mapping) or not isinstance(record, Mapping):
        raise Stage1ContractError("B7873200 Stage-1 bundle is incomplete")
    if record.get("schema") != STAGE1_SCHEMA or record.get("recipe_id") != STAGE1_RECIPE_ID:
        raise Stage1ContractError("B7873200 Stage-1 bundle identity changed")
    if record.get("role") != "checkpoint_final":
        raise Stage1ContractError("B7873200 Stage-1 bundle does not name a terminal final")
    recipe = record.get("stage1_recipe")
    sealed = record.get("sealed_protocol_identity")
    acquisition = record.get("acquisition_identity")
    validated_recipe = validate_stage1_recipe(recipe, sealed, acquisition)
    if expected_recipe is not None:
        expected = validate_stage1_recipe(
            expected_recipe,
            expected_recipe.get("sealed_protocol_identity", {}),
            expected_recipe.get("acquisition_identity", {}),
        )
        if not _recipes_equal(validated_recipe, expected):
            raise Stage1ContractError("Stage-1 final bundle recipe is not the expected B7873200 recipe")
    audit = _validate_generic_final_state(state, validated_recipe)
    if record.get("generic_execution_contract") != expected_generic_execution_contract(validated_recipe):
        raise Stage1ContractError("Stage-1 bundle execution contract changed")
    if state.get("execution_contract") != record["generic_execution_contract"]:
        raise Stage1ContractError("Stage-1 final no longer matches its recorded execution contract")
    saved_audit = record.get("structural_audit")
    if saved_audit != audit.as_dict():
        raise Stage1ContractError("Stage-1 structural audit no longer matches its final grid")
    if canonical_sealed_identity(record.get("sealed_protocol_identity", {})) != validated_recipe["sealed_protocol_identity"]:
        raise Stage1ContractError("Stage-1 bundle sealed identity changed")
    validate_b7873200_acquisition_identity(
        acquisition, validated_recipe["acquisition_identity"]
    )
    return {
        "bundle": bundle,
        "generic_final_state": state,
        "record": record,
        "recipe": validated_recipe,
        "audit": audit.as_dict(),
    }


def _recipes_equal(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    """Compare recipes containing NumPy/Torch acquisition arrays explicitly."""

    left_copy = copy.deepcopy(dict(left))
    right_copy = copy.deepcopy(dict(right))
    left_acquisition = left_copy.pop("acquisition_identity", None)
    right_acquisition = right_copy.pop("acquisition_identity", None)
    if left_copy != right_copy:
        return False
    try:
        validate_b7873200_acquisition_identity(left_acquisition, right_acquisition)
    except Stage1ContractError:
        return False
    return True


def load_b7873200_stage1_cloud(
    final_path: str | os.PathLike[str],
    *,
    expected_recipe: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate first, then derive a cloud without any permissive fallback."""

    validated = validate_b7873200_stage1_final(final_path, expected_recipe=expected_recipe)
    state = validated["generic_final_state"]
    recipe = validated["recipe"]
    model = state["model_state_dict"]
    granularity = int(recipe["scene"]["granularity"])
    shape = (granularity, granularity, granularity)
    w_re = _float_grid(model["w_re"], shape, "Stage-1 w_re")
    w_im = _float_grid(model["w_im"], shape, "Stage-1 w_im")
    active = _bool_grid(model["active_mask"], shape, "Stage-1 active_mask")
    positions = _to_numpy(model["grid_positions"], "Stage-1 grid_positions")
    magnitude = np.hypot(w_re.astype(np.float64), w_im.astype(np.float64))
    threshold = float(validated["audit"]["threshold_absolute"])
    keep = active & (magnitude >= threshold)
    flat = np.flatnonzero(keep.reshape(-1))
    max_points = int(recipe["stage2_cloud_extraction"]["max_points"])
    if max_points > 0 and len(flat) > max_points:
        order = np.argsort(magnitude.reshape(-1)[flat], kind="stable")[-max_points:]
        flat = flat[order]
    if len(flat) < 3:
        raise Stage1ContractError("validated Stage-1 cloud has fewer than three centres")
    return {
        "points": positions.reshape(-1, 3)[flat].astype(np.float32, copy=True),
        "magnitude": magnitude.reshape(-1)[flat].astype(np.float32, copy=True),
        "threshold": threshold,
        "source_epoch": int(validated["audit"]["epoch"]),
        "source_checkpoint": os.fspath(final_path),
        "extent": float(recipe["scene"]["extent_m"]),
        "granularity": granularity,
        "stage1_record": copy.deepcopy(validated["record"]),
    }


def atomic_save_stage1_bundle(bundle: Mapping[str, object], path: str | os.PathLike[str]) -> None:
    """Write a complete bundle atomically; never replace a completed artifact."""

    destination = Path(path)
    if destination.name != STAGE1_FINAL_BUNDLE_FILENAME:
        raise Stage1ContractError("Stage-1 bundle destination has the wrong fixed filename")
    if destination.exists():
        validate_b7873200_stage1_final(destination)
        raise FileExistsError(f"refusing to overwrite an existing Stage-1 final bundle: {destination}")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - local static environment has no torch
        raise RuntimeError("writing a Stage-1 bundle requires PyTorch") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    try:
        torch.save(dict(bundle), temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
