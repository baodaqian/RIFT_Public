#!/usr/bin/env python
"""Local no-runtime validation for the isolated RadarSplat B7873200 source.

This intentionally avoids importing the package root, which needs PyTorch in
the local desktop environment.  It compiles every authored file, verifies that
the new lane has no digest/identity helpers, and exercises its NumPy-only
sealed-role and recipe policy.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import py_compile
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NEW_LANE = (
    ROOT / "rift" / "radarsplat_b7873200.py",
    ROOT / "rift" / "radarsplat_b7873200_protocol.py",
    ROOT / "rift" / "radarsplat_b7873200_acquisition.py",
    ROOT / "rift" / "radarsplat_b7873200_adapter.py",
    ROOT / "scripts" / "prepare_radarsplat_b7873200_targets.py",
    ROOT / "train_radarsplat.py",
    ROOT / "scripts" / "validate_radarsplat_b7873200_native_contract.py",
    ROOT / "scripts" / "postflight_radarsplat_b7873200_smallfit.py",
)


def _protocol_module():
    path = ROOT / "rift" / "radarsplat_b7873200_protocol.py"
    specification = importlib.util.spec_from_file_location("radarsplat_b7873200_protocol_static", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _canonical_contract(protocol):
    permutation = np.random.Generator(np.random.PCG64(protocol.B787_3200_SEED)).permutation(
        protocol.B787_3200_NUM_VIEWS
    )
    validation_start = protocol.B787_3200_NUM_VIEWS - protocol.B787_3200_NUM_VALIDATION
    test_start = validation_start - protocol.B787_3200_NUM_TEST
    return {
        "schema": "rift_npz_sealed_protocol_v1",
        "version": 1,
        "data_format": "npz",
        "response_shape": [10_000, 16, 16, 1, 600],
        "response_dtype": "complex64",
        "role_manifest_name": protocol.B787_3200_MANIFEST_NAME,
        "split_strategy": "fixed_tail_subsampled",
        "role_ids": {
            "train": [int(value) for value in permutation[:protocol.B787_3200_NUM_TRAIN]],
            "validation": [int(value) for value in permutation[validation_start:]],
            "reserved_test": [int(value) for value in permutation[test_start:validation_start]],
            "unused": [int(value) for value in permutation[protocol.B787_3200_NUM_TRAIN:test_start]],
        },
        "response_access": {
            "train_materialized": True,
            "validation_materialized": True,
            "reserved_test_materialized": False,
            "unused_materialized": False,
        },
    }


def main() -> None:
    for retired_name in (
        "validate_radarsplat_b7873200_native.py",
        "validate_radarsplat_b7873200_training.py",
    ):
        if (ROOT / "scripts" / retired_name).exists():
            raise AssertionError(f"ambiguous historical RadarSplat validator remains under new-lane name: {retired_name}")
    for path in NEW_LANE:
        py_compile.compile(str(path), doraise=True)
    prohibited = ("hashlib", "sha256", "file_sha256", "canonical_json_sha256")
    for path in NEW_LANE:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        if set(prohibited).intersection(names | attributes):
            raise AssertionError(f"new RadarSplat B7873200 source includes a prohibited digest helper: {path}")
    trainer_text = (ROOT / "train_radarsplat.py").read_text(encoding="utf-8")
    for forbidden in ("--npz-path", "response_view", "load_b787_power_arrays", "matched_filter_complex"):
        if forbidden in trainer_text:
            raise AssertionError(f"cache-only trainer exposes a raw-response route: {forbidden}")
    for required in (
        "\"update_l1\"",
        "cuda_max_memory_allocated_bytes",
        "process_peak_rss_kib",
        "RADARSPLAT_B7873200_TRAIN_RESOURCE_JSON=",
        "phase=\"clean_interruption\"",
    ):
        if required not in trainer_text:
            raise AssertionError("bounded native trainer no longer records required update/resource telemetry")
    postflight_text = (ROOT / "scripts" / "postflight_radarsplat_b7873200_smallfit.py").read_text(encoding="utf-8")
    for required in (
        "expected_renderer = trainer._renderer_identity",
        "expected_train_log = (checkpoint_dir / \"training_trace.jsonl\").resolve()",
        "trainer._directly_equal(trace[-1], final_last_train)",
        "COMPLETION_RESUME_MARKER",
        "PREPARE_RESOURCE_PREFIX",
        "TRAIN_RESOURCE_PREFIX",
        "_read_resource_markers_from_log",
        "_completion_resume_log_path",
        "optimizer_updates_across_clean_interruptions_and_finalization",
    ):
        if required not in postflight_text:
            raise AssertionError("small-fit postflight no longer binds required lifecycle/resource semantics")
    launcher_text = (ROOT / "slurm" / "validate_radarsplat_b7873200_smallfit8x4_v1.sbatch").read_text(encoding="utf-8")
    for required in (
        "#SBATCH --signal=TERM@120",
        "trap terminal_signal TERM INT",
        "run_gpu_step python -B -u train_radarsplat.py",
        "RADARSPLAT_B7873200_SMALLFIT8X4_CLEAN_STOP_RETAINED",
        "RADARSPLAT_B7873200_SMALLFIT8X4_POSTFLIGHT_CLEAN_STOP_RETAINED",
        "mkdir -p -- \"$CHECKPOINT_DIR\"",
        "--launcher-log-root \"$LAUNCHER_LOG_ROOT\"",
        "--completion-resume-log \"$recovery_log\"",
    ):
        if required not in launcher_text:
            raise AssertionError("small-fit launcher no longer preserves required direct-step/recovery semantics")
    if "#SBATCH --signal=TERM@120" not in launcher_text or "trap terminal_signal TERM INT" not in launcher_text:
        raise AssertionError("small-fit launcher no longer delivers the warning signal to the trainer step")
    if "#SBATCH --signal=B:TERM@120" in launcher_text:
        raise AssertionError("small-fit launcher incorrectly restricts the warning signal to the batch shell")
    protocol = _protocol_module()
    expected_archive = (
        "/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/"
        "b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz"
    )
    if protocol.B787_3200_CANONICAL_NPZ_PATH != expected_archive:
        raise AssertionError("B7873200 target preparation does not select the corrected canonical archive")
    if f"NPZ={expected_archive}" not in launcher_text:
        raise AssertionError("B7873200 small-fit launcher does not use the corrected canonical archive")
    contract = _canonical_contract(protocol)
    identity = protocol.b7873200_sealed_identity(contract)
    if identity["role_ids"] != contract["role_ids"]:
        raise AssertionError("sealed identity changed the canonical ordered roles")
    bad = _canonical_contract(protocol)
    bad["role_ids"]["reserved_test"][0] = bad["role_ids"]["train"][0]
    try:
        protocol.b7873200_sealed_identity(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("sealed identity accepted a changed reserved-test role")
    target_spec = {
        "grid": {
            "scene_extent_m": 0.15,
            "scene_center_m": [0.0, 0.0, 0.0],
            "azimuth_center_deg": 0.0,
            "n_azimuth": 32,
            "n_elevation": 32,
            "n_range": 32,
            "output_azimuth_resolution_deg": 0.9,
            "elevation_sampling_resolution_deg": 0.9,
            "intermediate_azimuth_resolution_deg": 0.09,
            "azimuth_beamwidth_deg": 1.8,
            "spectral_leakage_width_m": 0.2,
        },
        "matched_filter": {
            "phase_sign": -1.0,
            "response_layout": "tx_rx_freq",
            "range_model": "none",
            "include_four_pi": False,
            "backend": "range_nufft",
            "compute_dtype": "float64",
        },
        "occupancy_threshold": 0.001,
    }
    recipe = protocol.expected_cache_recipe(identity, target_spec)
    if recipe["response_roles_materialized"] != ["train", "validation"]:
        raise AssertionError("native cache recipe would materialize a sealed response role")
    if recipe["target_spec"]["projection"] != "sum_elevation(abs(matched_filter_complex)**2) -> [azimuth,range]":
        raise AssertionError("native power endpoint changed")
    subset_recipe = protocol.expected_cache_recipe(
        identity,
        target_spec,
        materialized_roles={
            "train": identity["role_ids"]["train"][:8],
            "validation": identity["role_ids"]["validation"][:4],
        },
    )
    if subset_recipe["development_subset"] is None or subset_recipe["materialized_role_ids"]["train"] != identity["role_ids"]["train"][:8]:
        raise AssertionError("engineering subset lost its sealed ordered role meaning")
    try:
        protocol.expected_cache_recipe(
            identity,
            target_spec,
            materialized_roles={
                "train": [identity["role_ids"]["reserved_test"][0]],
                "validation": identity["role_ids"]["validation"][:1],
            },
        )
    except ValueError:
        pass
    else:
        raise AssertionError("engineering subset accepted a reserved-test response ID")
    try:
        protocol.expected_cache_recipe(
            identity,
            target_spec,
            materialized_roles={
                "train": identity["role_ids"]["train"][1:9],
                "validation": identity["role_ids"]["validation"][:4],
            },
        )
    except ValueError:
        pass
    else:
        raise AssertionError("engineering subset accepted a non-prefix role slice")
    bad_spec = dict(target_spec)
    bad_grid = dict(target_spec["grid"])
    bad_grid["intermediate_azimuth_resolution_deg"] = 0.11
    bad_spec["grid"] = bad_grid
    try:
        protocol.expected_cache_recipe(identity, bad_spec)
    except ValueError:
        pass
    else:
        raise AssertionError("non-integral output/intermediate grid was accepted")
    bad_leakage = dict(target_spec)
    bad_leakage_grid = dict(target_spec["grid"])
    bad_leakage_grid["spectral_leakage_width_m"] = 0.01
    bad_leakage["grid"] = bad_leakage_grid
    try:
        protocol.expected_cache_recipe(identity, bad_leakage)
    except ValueError:
        pass
    else:
        raise AssertionError("zero-pixel native spectral leakage was accepted into the cache recipe")
    bad_span = dict(target_spec)
    bad_span_grid = dict(target_spec["grid"])
    bad_span_grid["n_azimuth"] = 401
    bad_span["grid"] = bad_span_grid
    try:
        protocol.expected_cache_recipe(identity, bad_span)
    except ValueError:
        pass
    else:
        raise AssertionError("a local target span above 360 degrees was accepted into the cache recipe")
    bad_threshold = dict(target_spec)
    bad_threshold["occupancy_threshold"] = 0.15
    try:
        protocol.expected_cache_recipe(identity, bad_threshold)
    except ValueError:
        pass
    else:
        raise AssertionError("historical 0.15 occupancy threshold was accepted into the repaired lane")
    print("RadarSplat B7873200 static validation passed: compilation, no-digest source, sealed roles, and native recipe.")


if __name__ == "__main__":
    main()
