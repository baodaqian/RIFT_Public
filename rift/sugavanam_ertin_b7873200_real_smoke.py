"""Bounded real-data adapter for the B787 Sugavanam--Ertin engineering smoke.

This module is intentionally separate from the frozen 3,200/1,000 Stage-1 and
200+5,000-step Stage-2 production identities.  It derives a complete child
role manifest from the canonical parent, keeps the parent test and unused
payloads sealed, and provides only the small source/contract records needed by
the one combined smoke driver.
"""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


B787_CANONICAL_NPZ = (
    "/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/"
    "b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz"
)
B787_PARENT_MANIFEST = (
    "/storage/scratch1/1/dbao31/rift_round8b_impl_20260810/splits/round8b/"
    "b78710k_interp_seed42_train3200_val1000_test1000_v1.json"
)
B787_PARENT_MANIFEST_NAME = "b78710k_interp_seed42_train3200_val1000_test1000_v1"
B787_RESPONSE_SHAPE = [10_000, 16, 16, 1, 600]
B787_PARENT_COUNTS = {"train": 3_200, "validation": 1_000, "test": 1_000, "unused": 4_800}

SMOKE_SCHEMA = "rift_sugavanam_ertin_b7873200_real_smoke_v3"
SMOKE_MANIFEST_NAME = "b78710k_sugavanam_ertin_stage1_stage2_engineering_subset16x16_v3"
SMOKE_STAGE1_LABEL = "rift_sugavanam_ertin_b7873200_stage1_engineering_smoke_v3"
SMOKE_STAGE1_CHECKPOINT_NAME = "sugavanam_ertin_b7873200_stage1_engineering_smoke_v3"
SMOKE_STAGE1_BUNDLE_FILENAME = "stage1_engineering_smoke_v3_final.pth.tar"
SMOKE_STAGE1_BUNDLE_FIELD = "sugavanam_ertin_b7873200_real_smoke_stage1_v3"
SMOKE_STAGE2_CAMPAIGN = "rift_sugavanam_ertin_b7873200_stage2_engineering_smoke_v3"
SMOKE_STAGE2_ARTIFACT = "b787_sugavanam_ertin_stage2_engineering_smoke_v3"
SMOKE_RUN_NAME = "b78710k_sugavanam_ertin_stage1_stage2_engineering_smoke_v3"
SMOKE_REPORT_NAME = "real_smoke_report.json"
SMOKE_TRAIN_COUNT = 16
SMOKE_VALIDATION_COUNT = 16
SMOKE_TEST_COUNT = 1_000
SMOKE_STAGE1_EPOCHS = 2
SMOKE_STAGE1_EXPECTED_UPDATES = 32
SMOKE_STAGE1_RECIPE_REVISION = 3
SMOKE_STAGE1_EXTENT = 0.15
SMOKE_STAGE1_GRANULARITY = 16
COLLECTION_SMOKE_SCHEMA = "rift_dataset_sugavanam_ertin_smoke_v1"


def resolved(path: str | os.PathLike[str]) -> str:
    return os.path.realpath(os.path.abspath(os.fspath(path)))


def smoke_stage1_checkpoint_dir(checkpoint_root: str | os.PathLike[str]) -> Path:
    """Return the exact directory created by ``train.main`` for Stage 1."""

    return Path(os.fspath(checkpoint_root)) / SMOKE_STAGE1_CHECKPOINT_NAME


def read_json_mapping(path: str | os.PathLike[str], *, label: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise ValueError(f"could not read {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _role_ids(value: object, *, label: str, count: int) -> list[int]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{label} must contain exactly {count} IDs")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < 10_000:
            raise ValueError(f"{label} contains an invalid source-view ID")
        result.append(int(item))
    if len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicate source-view IDs")
    return result


def parent_roles(parent: Mapping[str, object]) -> dict[str, list[int]]:
    dataset = parent.get("dataset")
    split = parent.get("split")
    if not isinstance(dataset, Mapping) or not isinstance(split, Mapping):
        raise ValueError("canonical B787 manifest lacks dataset or split metadata")
    from .rift_dataset import DATASET_ID, _exact_value, role_manifest
    if dataset.get("dataset_id") == DATASET_ID:
        expected = role_manifest(dataset.get("object_id"))
        if any(not _exact_value(parent.get(key), expected[key])
               for key in ("schema_version", "name", "dataset", "split")):
            raise ValueError("collection smoke requires the registered object-bound parent manifest")
    elif parent.get("schema_version") != 1 or parent.get("name") != B787_PARENT_MANIFEST_NAME:
        raise ValueError("the real smoke requires a registered collection or canonical B787 manifest")
    if dataset.get("num_views") != 10_000 or list(dataset.get("response_shape", ())) != B787_RESPONSE_SHAPE:
        raise ValueError("canonical B787 manifest has the wrong response header")
    if dataset.get("response_dtype") != "complex64":
        raise ValueError("canonical B787 manifest response dtype changed")
    if split.get("strategy") != "fixed_tail_subsampled" or split.get("complete_partition") is not True:
        raise ValueError("canonical B787 manifest is not the fixed complete parent partition")
    if split.get("test_sealed") is not True or split.get("unused_sealed") is not True:
        raise ValueError("canonical B787 test and unused roles must remain sealed")
    keys = {
        "train": ("num_train", "train_indices"),
        "validation": ("num_validation", "validation_indices"),
        "test": ("num_test", "test_indices"),
        "unused": ("num_unused", "unused_indices"),
    }
    roles = {}
    for role, (count_key, ids_key) in keys.items():
        if split.get(count_key) != B787_PARENT_COUNTS[role]:
            raise ValueError(f"canonical B787 {role} count changed")
        roles[role] = _role_ids(split.get(ids_key), label=f"canonical {role}", count=B787_PARENT_COUNTS[role])
    flattened = [item for values in roles.values() for item in values]
    if len(set(flattened)) != 10_000 or set(flattened) != set(range(10_000)):
        raise ValueError("canonical B787 parent roles are not a complete disjoint partition")
    return roles


def load_canonical_parent(
    npz_path: str | os.PathLike[str], parent_manifest_path: str | os.PathLike[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Header-preflight the archive and manifest before any response read."""

    from .rift_dataset import collection_manifest, load_object_contract
    if collection_manifest(parent_manifest_path):
        _arrays, contract = load_object_contract(npz_path, parent_manifest_path)
        parent = read_json_mapping(parent_manifest_path, label="RIFT dataset smoke parent")
        parent_roles(parent)
        return parent, contract
    if resolved(npz_path) != resolved(B787_CANONICAL_NPZ):
        raise ValueError(f"real smoke accepts only {B787_CANONICAL_NPZ}")
    if resolved(parent_manifest_path) != resolved(B787_PARENT_MANIFEST):
        raise ValueError(f"real smoke accepts only {B787_PARENT_MANIFEST}")
    from train import _load_sealed_npz_protocol_contract

    arrays, contract = _load_sealed_npz_protocol_contract(
        npz_path,
        parent_manifest_path,
        num_train=B787_PARENT_COUNTS["train"],
        num_val=B787_PARENT_COUNTS["validation"],
        num_test=B787_PARENT_COUNTS["test"],
    )
    if arrays.get("response") is not None:
        raise ValueError("parent preflight materialized the response payload")
    if contract.get("response_shape") != B787_RESPONSE_SHAPE or contract.get("response_dtype") != "complex64":
        raise ValueError("canonical B787 archive header changed")
    parent = read_json_mapping(parent_manifest_path, label="canonical B787 manifest")
    parent_roles(parent)
    return parent, dict(contract)


def _child_manifest_record(parent: Mapping[str, object]) -> dict[str, Any]:
    """Create the complete 16/16 child without relaxing parent role policy."""

    roles = parent_roles(parent)
    train = roles["train"][:SMOKE_TRAIN_COUNT]
    validation = roles["validation"][:SMOKE_VALIDATION_COUNT]
    test = list(roles["test"])
    unused = (
        roles["train"][SMOKE_TRAIN_COUNT:]
        + roles["unused"]
        + roles["validation"][SMOKE_VALIDATION_COUNT:]
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "name": SMOKE_MANIFEST_NAME,
        "dataset": {
            "num_views": 10_000,
            "response_shape": list(B787_RESPONSE_SHAPE),
            "response_dtype": "complex64",
        },
        "split": {
            "strategy": "parent_fixed_tail_prefix_engineering_se_v1",
            "num_train": len(train),
            "num_validation": len(validation),
            "num_test": len(test),
            "num_unused": len(unused),
            "complete_partition": True,
            "test_sealed": True,
            "unused_sealed": True,
            "train_indices": train,
            "validation_indices": validation,
            "test_indices": test,
            "unused_indices": unused,
        },
        "engineering_subset": {
            "schema": SMOKE_SCHEMA,
            "version": SMOKE_STAGE1_RECIPE_REVISION,
            "parent_manifest_name": B787_PARENT_MANIFEST_NAME,
            "selection": {
                "train": "ordered parent train[:16]",
                "validation": "ordered parent validation[:16]",
                "test": "all parent sealed-test IDs",
                "unused": "remaining parent train/validation plus parent unused IDs",
            },
            "reporting_status": "engineering_smoke_not_production_or_comparison",
        },
    }
    from .rift_dataset import DATASET_ID, object_identity
    if parent["dataset"].get("dataset_id") == DATASET_ID:
        identity = object_identity(parent["dataset"]["object_id"])
        payload["name"] = f"rift_dataset_{identity['object_id']}_se_smoke16x16_v1"
        payload["dataset"].update(identity)
        payload["engineering_subset"].update(
            schema=COLLECTION_SMOKE_SCHEMA, parent_manifest_name=parent["name"])
    return payload


def build_child_manifest(parent: Mapping[str, object]) -> dict[str, Any]:
    payload = _child_manifest_record(parent)
    validate_child_manifest(parent, payload)
    return payload


def validate_child_manifest(parent: Mapping[str, object], child: Mapping[str, object]) -> None:
    parent_split = parent_roles(parent)
    from .rift_dataset import DATASET_ID, _exact_value
    if parent["dataset"].get("dataset_id") == DATASET_ID:
        if not _exact_value(child, _child_manifest_record(parent)):
            raise ValueError("collection smoke child changed its object, registered 16/16 roles, or provenance")
        return
    child_split = child.get("split")
    if child.get("schema_version") != 1 or child.get("name") != SMOKE_MANIFEST_NAME:
        raise ValueError("real smoke child manifest identity changed")
    if not isinstance(child_split, Mapping):
        raise ValueError("real smoke child manifest lacks split metadata")
    expected = {
        "train_indices": parent_split["train"][:SMOKE_TRAIN_COUNT],
        "validation_indices": parent_split["validation"][:SMOKE_VALIDATION_COUNT],
        "test_indices": parent_split["test"],
        "unused_indices": (
            parent_split["train"][SMOKE_TRAIN_COUNT:]
            + parent_split["unused"]
            + parent_split["validation"][SMOKE_VALIDATION_COUNT:]
        ),
    }
    for key, expected_ids in expected.items():
        if child_split.get(key) != expected_ids:
            raise ValueError(f"real smoke child {key} does not preserve parent order")
    counts = {"train": 16, "validation": 16, "test": 1_000, "unused": 8_968}
    for role, count in counts.items():
        if child_split.get(f"num_{role}") != count:
            raise ValueError(f"real smoke child {role} count changed")
    if (
        child_split.get("strategy") != "parent_fixed_tail_prefix_engineering_se_v1"
        or child_split.get("complete_partition") is not True
        or child_split.get("test_sealed") is not True
        or child_split.get("unused_sealed") is not True
    ):
        raise ValueError("real smoke child must retain a complete sealed partition")
    dataset = child.get("dataset")
    if not isinstance(dataset, Mapping) or dataset.get("num_views") != 10_000:
        raise ValueError("real smoke child dataset metadata changed")
    if list(dataset.get("response_shape", ())) != B787_RESPONSE_SHAPE or dataset.get("response_dtype") != "complex64":
        raise ValueError("real smoke child response header changed")
    engineering = child.get("engineering_subset")
    if not isinstance(engineering, Mapping) or engineering.get("schema") != SMOKE_SCHEMA:
        raise ValueError("real smoke child engineering provenance is incomplete")


def load_collection_smoke_inputs(npz_path, manifest_path) -> tuple[dict, dict]:
    """Public, metadata-only loader for the exact registered 16/16 SE subset.

    This deliberately does not relax the full 3200/1000 collection contract.
    Generic training recognizes only this named engineering schema and checks
    its requested counts separately before using this already restricted handle.
    """
    from .npz_dataset import load_npz_arrays, restrict_npz_response_views
    from .rift_dataset import metadata_object_id, object_identity, role_manifest, validate_metadata

    child = read_json_mapping(manifest_path, label="RIFT dataset SE smoke manifest")
    declared = child.get("dataset")
    if not isinstance(declared, Mapping):
        raise ValueError("collection smoke manifest requires object-valued dataset metadata")
    identity = object_identity(declared.get("object_id"))
    validate_child_manifest(role_manifest(identity["object_id"]), child)
    arrays = load_npz_arrays(npz_path, load_response=False)
    validate_metadata(arrays["meta"])
    if metadata_object_id(arrays["meta"]) != identity["object_id"]:
        raise ValueError("collection smoke object does not match NPZ metadata")
    if (arrays.get("response") is not None or tuple(arrays["response_shape"]) != tuple(B787_RESPONSE_SHAPE)
            or np.dtype(arrays["response_dtype"]) != np.dtype("complex64")):
        raise ValueError("collection smoke response header changed")
    for key, shape in (("viewpoint_positions", (10000, 3)),
                       ("tx_pos", (10000, 16, 3)), ("rx_pos", (10000, 16, 3))):
        value = np.asarray(arrays[key])
        if value.shape != shape or value.dtype != np.float64 or not np.isfinite(value).all():
            raise ValueError(f"collection smoke {key} geometry changed")
    if not np.allclose(np.linalg.norm(arrays["viewpoint_positions"], axis=1), 10.0, rtol=0, atol=1e-8):
        raise ValueError("collection smoke viewpoints must remain on the 10 m sphere")
    with zipfile.ZipFile(npz_path) as archive:
        if archive.getinfo("response.npy").compress_type != zipfile.ZIP_STORED:
            raise ValueError("collection smoke requires uncompressed response storage")
    roles = {role: child["split"][key] for role, key in (
        ("train", "train_indices"), ("validation", "validation_indices"),
        ("reserved_test", "test_indices"), ("unused", "unused_indices"))}
    contract = {
        "schema": "rift_npz_sealed_protocol_v1", "version": 1, "data_format": "npz",
        "source_path": os.path.abspath(os.fspath(npz_path)),
        "response_shape": list(B787_RESPONSE_SHAPE), "response_dtype": "complex64",
        "role_manifest_path": os.path.abspath(os.fspath(manifest_path)),
        "role_manifest_name": child["name"], "split_strategy": child["split"]["strategy"],
        "role_ids": roles, "dataset_identity": identity,
        "response_access": {"train_materialized": True, "validation_materialized": True,
                            "reserved_test_materialized": False, "unused_materialized": False},
    }
    return restrict_npz_response_views(arrays, roles["train"] + roles["validation"]), contract


def write_child_manifest(path: str | os.PathLike[str], payload: Mapping[str, object]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized = json.loads(json.dumps(payload))
    if destination.exists():
        existing = read_json_mapping(destination, label="existing real smoke child manifest")
        if existing != normalized:
            raise ValueError("real smoke manifest path already contains another configuration")
        return destination
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    os.close(descriptor)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(normalized, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def smoke_stage1_argv(
    *,
    npz_path: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    checkpoint_root: str | os.PathLike[str],
    resume: str | os.PathLike[str] | None = None,
) -> list[str]:
    """Return the explicit altered-budget Stage-1 engineering command."""

    argv = [
        "--data-format", "npz",
        "--npz-path", os.fspath(npz_path),
        "--npz-sealed-protocol",
        "--npz-role-manifest", os.fspath(manifest_path),
        "--checkpoint-name", SMOKE_STAGE1_CHECKPOINT_NAME,
        "--checkpoint-root", os.fspath(checkpoint_root),
        "--execution-contract-label", SMOKE_STAGE1_LABEL,
        "--require-full-resume-state",
        "--num-train", str(SMOKE_TRAIN_COUNT),
        "--num-val", str(SMOKE_VALIDATION_COUNT),
        "--num-test", str(SMOKE_TEST_COUNT),
        "--num-tx", "16",
        "--num-rx", "16",
        "--num-freq-wanted", "600",
        "--epochs", str(SMOKE_STAGE1_EPOCHS),
        "--step-every", "1",
        "--loss", "complex",
        "--scene-repr", "grid",
        "--forward-operator", "range",
        "--range-model", "product",
        "--compute-dtype", "float64",
        "--point-chunk", "4096",
        "--pair-chunk", "32",
        "--extent", str(SMOKE_STAGE1_EXTENT),
        "--granularity", str(SMOKE_STAGE1_GRANULARITY),
        "--phase-sign", "-1",
        "--bp-init", "16",
        "--init-scale", "0",
        "--lr", "0.003",
        "--l1-weight", "3e-7",
        "--adam-eps", "1e-20",
        "--weight-decay", "0",
        "--checkpoint-metric", "val",
        "--t0", "10",
        "--t-mult", "2",
        "--seed", "42",
        "--prune-every", "0",
        "--prune-threshold", "0",
        "--prune-criterion", "energy",
        "--prune-start-epoch", "0",
        "--prune-mode", "mass",
        "--prune-target-active", "0",
        "--prune-end-epoch", "0",
        "--prune-min-active", "0",
    ]
    if resume is not None:
        argv.extend(("--resume", os.fspath(resume)))
    return argv


def complex_signal_statistics(values: object) -> dict[str, float | int]:
    """Compute a finite RMS and zero-reference power from coherent samples."""

    samples = np.asarray(values)
    if samples.size == 0 or not np.issubdtype(samples.dtype, np.complexfloating):
        raise ValueError("coherent samples must be a non-empty complex array")
    magnitude = np.abs(samples).astype(np.float64, copy=False)
    squared_sum = float(np.square(magnitude).sum())
    count = int(magnitude.size)
    if count <= 0 or not math.isfinite(squared_sum) or squared_sum <= 0:
        raise ValueError("coherent samples have no finite positive power")
    mean_square = squared_sum / count
    return {
        "raw_complex_rms": math.sqrt(mean_square),
        "zero_reference_mse": mean_square,
        "sample_count": count,
    }


def analytic_sdf_shell_diagnostic(
    points: object,
    *,
    extent: float,
    granularity: int,
    radius_quantile: float,
    radius_cap_fraction: float,
) -> dict[str, object]:
    """Check that the analytic closed-field target fits below its protected shell."""

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3:
        raise ValueError("points must have shape [N,3] with N>=3")
    if not np.isfinite(pts).all() or not math.isfinite(extent) or extent <= 0:
        raise ValueError("points and extent must be finite and positive")
    if int(granularity) != granularity or granularity < 2:
        raise ValueError("granularity must be an integer >= 2")
    if not (0.1 <= radius_quantile <= 0.9):
        raise ValueError("radius_quantile must be in [0.1,0.9]")
    if not (0.25 <= radius_cap_fraction <= 0.9):
        raise ValueError("radius_cap_fraction must be in [0.25,0.9]")
    if np.max(np.abs(pts)) >= extent:
        raise ValueError("Stage-1 points must lie strictly inside the SDF ROI")

    center = np.median(pts, axis=0)
    boundary_clearance = float(np.min(extent - np.abs(center)))
    pitch = 2.0 * float(extent) / int(granularity)
    radius_floor = 2.0 * pitch
    radius_cap = float(radius_cap_fraction) * boundary_clearance
    distances = np.linalg.norm(pts - center[None, :], axis=1)
    radius = float(np.clip(np.quantile(distances, radius_quantile), radius_floor, radius_cap))
    shell_clearance = boundary_clearance - radius_floor - radius
    feasible = bool(radius_cap > radius_floor and shell_clearance > 0.0)
    return {
        "center": [float(value) for value in center],
        "radius": radius,
        "pitch": pitch,
        "boundary_clearance": boundary_clearance,
        "protected_shell_depth": radius_floor,
        "radius_floor": radius_floor,
        "radius_cap": radius_cap,
        "radius_quantile": float(radius_quantile),
        "radius_cap_fraction": float(radius_cap_fraction),
        "shell_clearance": shell_clearance,
        "feasible": feasible,
        "reason": None
        if feasible
        else "analytic target radius plus protected shell does not fit inside the SDF ROI",
    }


def smoke_stage1_recipe(
    *, normalized_train_rms: float, zero_reference_train_mse: float
) -> dict[str, object]:
    if not math.isfinite(normalized_train_rms) or normalized_train_rms <= 0:
        raise ValueError("train-only normalization must be finite and positive")
    if not math.isfinite(zero_reference_train_mse) or zero_reference_train_mse <= 0:
        raise ValueError("same-domain zero reference must be finite and positive")
    return {
        "schema": SMOKE_SCHEMA,
        "recipe_revision": SMOKE_STAGE1_RECIPE_REVISION,
        "recipe_id": SMOKE_STAGE1_LABEL,
        "production_recipe_id": "sugavanam_ertin_b7873200_isotropic_scatter_v1",
        "scope": "bounded_engineering_smoke_not_production_not_comparison_not_convergence_evidence",
        "ground_truth_geometry_used": False,
        "sealed_parent_counts": dict(B787_PARENT_COUNTS),
        "selected_roles": {"train": 16, "validation": 16, "reserved_test": 1_000, "unused": 8_968},
        "observation": {
            "data_format": "npz",
            "all_coordinates": {"tx": 16, "rx": 16, "chirps": 1, "frequencies": 600},
            "forward_operator": "range",
            "range_model": "product",
            "phase_sign": -1.0,
            "compute_dtype": "float64",
            "point_chunk": 4096,
            "pair_chunk": 32,
        },
        "scene": {
            "representation": "grid",
            "extent_m": SMOKE_STAGE1_EXTENT,
            "granularity": SMOKE_STAGE1_GRANULARITY,
            "voxel_pitch_m": 2.0 * SMOKE_STAGE1_EXTENT / SMOKE_STAGE1_GRANULARITY,
        },
        "acquisition_identity": {
            "schema": "rift_sugavanam_ertin_b7873200_smoke_acquisition_v1",
            "coordinates": {"tx": 16, "rx": 16, "frequencies": 600, "units": "metres"},
            "response_payload_materialized": False,
            "normalization_scope": "selected_parent_train_only",
        },
        "fit": {
            "loss": "complex",
            "epochs": SMOKE_STAGE1_EPOCHS,
            "logical_updates": SMOKE_STAGE1_EXPECTED_UPDATES,
            "backprojection_views": 16,
            "optimizer": "AdamW",
            "learning_rate": 0.003,
            "l1_weight": 3.0e-7,
            "adam_eps": 1.0e-20,
            "weight_decay": 0.0,
            "scheduler_clock": "two altered-budget epochs; not the 150-epoch production identity",
            "seed": 42,
        },
        "normalization": {
            "scope": "selected_parent_train_only",
            "source_count": 16,
            "raw_complex_rms": float(normalized_train_rms),
            "zero_reference_train_mse": float(zero_reference_train_mse),
            "used_for": "reported same-domain normalization only; no validation/test leakage",
        },
    }


def compute_train_only_signal_normalization(arrays: Mapping[str, object], train_ids: Sequence[int]) -> dict[str, object]:
    """Stream selected training responses and compute a reporting-only scale."""

    if arrays.get("response") is not None:
        raise ValueError("normalization requires a lazy pre-payload archive reader")
    from rift.npz_dataset import get_npz_response_view, restrict_npz_response_views

    restricted = restrict_npz_response_views(dict(arrays), train_ids)
    squared_sum = 0.0
    count = 0
    for source_id in train_ids:
        response = np.asarray(get_npz_response_view(restricted, int(source_id)))
        if response.shape != (16, 16, 1, 600) or response.dtype != np.dtype(np.complex64):
            raise ValueError("selected B787 response has an unexpected shape or dtype")
        coherent = response.mean(axis=2).astype(np.complex128, copy=False)
        stats = complex_signal_statistics(coherent)
        squared_sum += float(stats["zero_reference_mse"]) * int(stats["sample_count"])
        count += int(stats["sample_count"])
    if count <= 0 or not math.isfinite(squared_sum) or squared_sum <= 0:
        raise ValueError("selected parent-training signal energy is not finite and positive")
    mean_square = squared_sum / count
    return {
        "raw_complex_rms": math.sqrt(mean_square),
        "zero_reference_train_mse": mean_square,
        "source_count": len(train_ids),
        "payload_materialized": True,
        "scope": "selected_parent_train_only",
    }


def assert_restricted_roles(arrays: Mapping[str, object], roles: Mapping[str, Sequence[int]]) -> None:
    """Exercise the capability boundary without reading test/unused payloads."""

    from rift.npz_dataset import get_npz_response_view, restrict_npz_response_views

    allowed = list(roles["train"]) + list(roles["validation"])
    restricted = restrict_npz_response_views(dict(arrays), allowed)
    for forbidden in (roles["test"][0], roles["unused"][0]):
        try:
            get_npz_response_view(restricted, int(forbidden))
        except PermissionError:
            continue
        raise AssertionError(f"restricted real smoke reader exposed forbidden source-view {forbidden}")


def validate_generic_stage1_final(
    state: Mapping[str, object], *, manifest_contract: Mapping[str, object],
    observation: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate the actual small generic final before making a Stage-2 source."""

    if not isinstance(state, Mapping) or state.get("epoch") != SMOKE_STAGE1_EPOCHS:
        raise ValueError("real smoke Stage-1 did not reach its two-epoch terminal state")
    execution = state.get("execution_contract")
    if not isinstance(execution, Mapping) or execution.get("label") != SMOKE_STAGE1_LABEL:
        raise ValueError("real smoke Stage-1 execution identity changed")
    execution_observation = execution.get("observation")
    scene = execution.get("scene")
    physics = execution.get("physics")
    fit = execution.get("fit")
    if not isinstance(execution_observation, Mapping) or not isinstance(scene, Mapping) or not isinstance(physics, Mapping) or not isinstance(fit, Mapping):
        raise ValueError("real smoke Stage-1 execution contract is incomplete")
    if (
        execution_observation.get("num_train") != 16
        or execution_observation.get("num_validation") != 16
        or execution_observation.get("num_reserved_test") != 1_000
        or execution_observation.get("num_freq_wanted") != 600
        or execution_observation.get("sealed_npz_protocol") is not True
    ):
        raise ValueError("real smoke Stage-1 role/count contract changed")
    if (
        physics.get("forward_operator") != "range"
        or physics.get("range_model") != "product"
        or physics.get("phase_sign") != -1.0
        or physics.get("compute_dtype") != "float64"
        or physics.get("num_rx") != 16
        or physics.get("num_tx") != 16
        or physics.get("point_chunk") != 4096
        or physics.get("pair_chunk") != 32
    ):
        raise ValueError("real smoke Stage-1 coherent operator contract changed")
    if (
        scene.get("representation") != "grid"
        or scene.get("granularity") != SMOKE_STAGE1_GRANULARITY
        or float(scene.get("extent_m", 0.0)) != SMOKE_STAGE1_EXTENT
        or scene.get("initial_scale") != 0.0
        or scene.get("backprojection_views") != 16
        or scene.get("normalize_scene_scale") is not False
    ):
        raise ValueError("real smoke Stage-1 grid contract changed")
    if (
        fit.get("epochs") != SMOKE_STAGE1_EPOCHS
        or fit.get("loss") != "complex"
        or fit.get("step_every") != 1
        or fit.get("learning_rate") != 0.003
        or fit.get("weight_decay") != 0.0
        or fit.get("l1_weight") != 3.0e-7
        or fit.get("adam_eps") != 1.0e-20
        or fit.get("checkpoint_metric") != "val"
        or fit.get("seed") != 42
    ):
        raise ValueError("real smoke Stage-1 fit budget/objective changed")
    if state.get("sealed_npz_protocol_contract") != dict(manifest_contract):
        raise ValueError("real smoke Stage-1 sealed child contract changed")
    model = state.get("model_state_dict")
    gain = state.get("gain_state_dict")
    optimizer = state.get("optimizer_state_dict")
    scheduler = state.get("scheduler_state_dict")
    rng = state.get("rng_state")
    if not isinstance(model, Mapping) or set(model) != {"w_re", "w_im", "active_mask", "grid_positions"}:
        raise ValueError("real smoke Stage-1 final lacks the fixed-grid payload")
    if not isinstance(gain, Mapping) or not isinstance(optimizer, Mapping) or not isinstance(scheduler, Mapping) or not isinstance(rng, Mapping):
        raise ValueError("real smoke Stage-1 final lacks recovery state")
    for name in ("w_re", "w_im", "grid_positions"):
        value = model[name]
        if not hasattr(value, "detach") or not bool(np.isfinite(value.detach().cpu().numpy()).all()):
            raise ValueError(f"real smoke Stage-1 model tensor {name} is non-finite")
    active = model["active_mask"]
    if (
        not hasattr(active, "detach")
        or str(active.dtype) != "torch.bool"
        or not bool(active.detach().cpu().numpy().any())
    ):
        raise ValueError("real smoke Stage-1 final has no active cells")
    moments = optimizer.get("state")
    if not isinstance(moments, Mapping) or not moments:
        raise ValueError("real smoke Stage-1 optimizer has no update state")
    steps = []
    for moment in moments.values():
        if isinstance(moment, Mapping) and hasattr(moment.get("step"), "detach"):
            steps.append(float(moment["step"].detach().cpu().item()))
    if not steps or max(steps) < SMOKE_STAGE1_EXPECTED_UPDATES:
        raise ValueError("real smoke Stage-1 did not record the expected nonzero updates")
    selector_loss = float(state.get("loss", float("nan")))
    if not math.isfinite(selector_loss) or selector_loss < 0:
        raise ValueError("real smoke Stage-1 checkpoint selector loss is invalid")
    if observation is None:
        raise ValueError("real smoke Stage-1 lacks observed fit evidence")
    observation_audit = validate_stage1_observation(observation)
    return {
        "epoch": int(state["epoch"]),
        "checkpoint_selector_loss": selector_loss,
        "optimizer_steps": max(steps),
        "observed_fit": observation_audit,
        "sealed_roles": {"train": 16, "validation": 16, "reserved_test": 1_000, "unused": 8_968},
        "operator": {"forward": "range", "range_model": "product", "phase_sign": -1.0, "pairs": "16x16", "frequencies": 600},
        "scene": {
            "representation": "grid",
            "granularity": SMOKE_STAGE1_GRANULARITY,
            "extent_m": SMOKE_STAGE1_EXTENT,
            "voxel_pitch_m": 2.0 * SMOKE_STAGE1_EXTENT / SMOKE_STAGE1_GRANULARITY,
        },
        "ground_truth_geometry_used": False,
    }


def validate_stage1_observation(observation: Mapping[str, object]) -> dict[str, object]:
    """Validate observed gradients, parameter deltas, and coherent readouts."""

    if observation.get("schema") != "rift_sugavanam_ertin_stage1_engineering_observation_v1":
        raise ValueError("real smoke Stage-1 observation schema changed")
    start_epoch = observation.get("start_epoch")
    logical_start = observation.get("logical_optimizer_updates_at_start")
    observed_steps = observation.get("observed_optimizer_steps")
    nonzero_updates = observation.get("observed_nonzero_parameter_updates")
    logical_final = observation.get("logical_optimizer_updates_final")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (
        start_epoch, logical_start, observed_steps, nonzero_updates, logical_final
    )):
        raise ValueError("real smoke Stage-1 observation counts are invalid")
    steps = observation.get("steps")
    if not isinstance(steps, list) or len(steps) != observed_steps or observed_steps <= 0:
        raise ValueError("real smoke Stage-1 observation step count changed")
    if logical_final != logical_start + observed_steps:
        raise ValueError("real smoke Stage-1 observation update progression changed")
    counted_nonzero = 0
    required_step_fields = {
        "epoch", "logical_optimizer_updates", "reported_grad_norm", "observed_gradient_l2",
        "finite_gradients", "nonzero_gradient_tensors", "parameter_delta_l2",
        "max_abs_parameter_delta", "finite_parameter_delta", "nonzero_parameter_update",
    }
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, Mapping) or set(step) != required_step_fields:
            raise ValueError("real smoke Stage-1 observation step schema changed")
        if step["logical_optimizer_updates"] != logical_start + index:
            raise ValueError("real smoke Stage-1 observation step progression changed")
        for key in ("reported_grad_norm", "observed_gradient_l2", "parameter_delta_l2", "max_abs_parameter_delta"):
            if not math.isfinite(float(step[key])) or float(step[key]) < 0:
                raise ValueError(f"real smoke Stage-1 observation is invalid: {key}")
        if not step["finite_gradients"] or not step["finite_parameter_delta"]:
            raise ValueError("real smoke Stage-1 observation contains non-finite state")
        if float(step["observed_gradient_l2"]) <= 0 or float(step["parameter_delta_l2"]) <= 0:
            raise ValueError("real smoke Stage-1 observation contains a zero update")
        if step["nonzero_parameter_update"]:
            counted_nonzero += 1
    if counted_nonzero != nonzero_updates or nonzero_updates <= 0:
        raise ValueError("real smoke Stage-1 observed nonzero update count changed")
    for phase in ("initial", "final"):
        metrics = observation.get(phase)
        if not isinstance(metrics, Mapping) or set(metrics) != {"train", "validation"}:
            raise ValueError(f"real smoke Stage-1 {phase} readout is incomplete")
        for role in ("train", "validation"):
            values = metrics.get(role)
            if not isinstance(values, Mapping) or set(values) != {
                "loss", "residual_power", "zero_reference_power", "relative_mse", "relative_l2"
            }:
                raise ValueError(f"real smoke Stage-1 {phase}/{role} readout schema changed")
            if any(not math.isfinite(float(values[key])) for key in values):
                raise ValueError(f"real smoke Stage-1 {phase}/{role} readout is non-finite")
            if float(values["zero_reference_power"]) <= 0 or float(values["residual_power"]) < 0:
                raise ValueError(f"real smoke Stage-1 {phase}/{role} coherent power is invalid")
            expected_mse = float(values["residual_power"]) / float(values["zero_reference_power"])
            if not math.isclose(float(values["relative_mse"]), expected_mse, rel_tol=1.0e-12, abs_tol=1.0e-15):
                raise ValueError(f"real smoke Stage-1 {phase}/{role} MSE denominator changed")
            if not math.isclose(float(values["relative_l2"]), math.sqrt(expected_mse), rel_tol=1.0e-12, abs_tol=1.0e-15):
                raise ValueError(f"real smoke Stage-1 {phase}/{role} L2 denominator changed")
    return {
        "start_epoch": int(start_epoch),
        "logical_optimizer_updates_at_start": int(logical_start),
        "observed_optimizer_steps": int(observed_steps),
        "observed_nonzero_parameter_updates": int(nonzero_updates),
        "logical_optimizer_updates_final": int(logical_final),
        "initial_final_readouts": True,
        "initial_metrics": copy.deepcopy(dict(observation["initial"])),
        "final_metrics": copy.deepcopy(dict(observation["final"])),
    }


def smoke_stage1_record(
    *,
    state: Mapping[str, object],
    manifest_contract: Mapping[str, object],
    normalization: Mapping[str, object],
    final_path: str,
    observation: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": SMOKE_SCHEMA,
        "recipe_id": SMOKE_STAGE1_LABEL,
        "production_recipe_id": "sugavanam_ertin_b7873200_isotropic_scatter_v1",
        "role": "checkpoint_final",
        "stage1_recipe": smoke_stage1_recipe(
            normalized_train_rms=float(normalization["raw_complex_rms"]),
            zero_reference_train_mse=float(normalization["zero_reference_train_mse"]),
        ),
        "sealed_protocol_identity": copy.deepcopy(dict(manifest_contract)),
        "generic_execution_contract": copy.deepcopy(dict(state["execution_contract"])),
        "engineering_observation": copy.deepcopy(dict(observation)),
        "structural_audit": {
            "epoch": int(state["epoch"]),
            "checkpoint_selector_loss": float(state["loss"]),
            "final_checkpoint": str(final_path),
            "observed_nonzero_parameter_updates": int(observation["observed_nonzero_parameter_updates"]),
            "observed_optimizer_steps": int(observation["observed_optimizer_steps"]),
            "ground_truth_geometry_used": False,
        },
        "provenance": {
            "canonical_npz_path": B787_CANONICAL_NPZ,
            "parent_manifest_path": B787_PARENT_MANIFEST,
            "child_manifest_name": SMOKE_MANIFEST_NAME,
            "stage1_final_filename": SMOKE_STAGE1_BUNDLE_FILENAME,
            "reporting_status": "engineering_smoke_not_production_or_comparison",
        },
    }


def smoke_cloud_from_final(
    state: Mapping[str, object], *, stage1_record: Mapping[str, object], checkpoint_path: str
) -> dict[str, object]:
    model = state["model_state_dict"]
    if not isinstance(model, Mapping):
        raise ValueError("real smoke Stage-1 model state is missing")
    w_re = model["w_re"].detach().cpu().numpy().astype(np.float64, copy=False)
    w_im = model["w_im"].detach().cpu().numpy().astype(np.float64, copy=False)
    active = model["active_mask"].detach().cpu().numpy().astype(bool, copy=False)
    positions = model["grid_positions"].detach().cpu().numpy().astype(np.float64, copy=False)
    grid_shape = (SMOKE_STAGE1_GRANULARITY,) * 3
    if w_re.shape != grid_shape or w_im.shape != w_re.shape or active.shape != w_re.shape or positions.shape != grid_shape + (3,):
        raise ValueError("real smoke Stage-1 fixed-grid shape changed")
    magnitude = np.hypot(w_re, w_im)
    if not (np.isfinite(magnitude).all() and np.isfinite(positions).all() and active.any()):
        raise ValueError("real smoke Stage-1 cloud is non-finite or empty")
    maximum = float(magnitude[active].max())
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("real smoke Stage-1 cloud has no positive scattering magnitude")
    threshold = 0.15 * maximum
    keep = active & (magnitude >= threshold)
    flat = np.flatnonzero(keep.reshape(-1))
    if len(flat) < 3:
        raise ValueError("real smoke Stage-1 extraction retained fewer than three centres")
    return {
        "points": positions.reshape(-1, 3)[flat].astype(np.float32, copy=True),
        "magnitude": magnitude.reshape(-1)[flat].astype(np.float32, copy=True),
        "threshold": float(threshold),
        "source_epoch": int(state["epoch"]),
        "source_checkpoint": str(checkpoint_path),
        "extent": SMOKE_STAGE1_EXTENT,
        "granularity": SMOKE_STAGE1_GRANULARITY,
        "stage1_record": copy.deepcopy(dict(stage1_record)),
        "extraction": {
            "threshold_fraction": 0.15,
            "retained_count": int(len(flat)),
            "maximum_magnitude": maximum,
            "ground_truth_geometry_used": False,
        },
    }
