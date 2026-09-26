"""Source-only checks for the isolated B7873200 SE Stage-1 contract.

This deliberately exercises semantic provenance and grid validation with a
synthetic header/pose surface.  It does not read the B787 archive, materialize
responses, create a checkpoint on PACE, or run training.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import random
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "sugavanam_ertin_b7873200_stage1_under_test",
    ROOT / "rift" / "sugavanam_ertin_b7873200_stage1.py",
)
assert _SPEC is not None and _SPEC.loader is not None
stage1 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = stage1
_SPEC.loader.exec_module(stage1)


def expect_reject(label: str, callback) -> None:
    try:
        callback()
    except (ValueError, RuntimeError, FileNotFoundError, PermissionError):
        return
    raise AssertionError(f"expected rejection: {label}")


def canonical_contract() -> dict[str, object]:
    permutation = np.random.Generator(np.random.PCG64(42)).permutation(10_000)
    return {
        "schema": "rift_npz_sealed_protocol_v1",
        "version": 1,
        "data_format": "npz",
        "source_path": stage1.B787_3200_CANONICAL_NPZ_PATH,
        "response_shape": [10_000, 16, 16, 1, 600],
        "response_dtype": "complex64",
        "role_manifest_path": stage1.B787_3200_CANONICAL_MANIFEST_PATH,
        "role_manifest_name": stage1.B787_3200_MANIFEST_NAME,
        "split_strategy": "fixed_tail_subsampled",
        "role_ids": {
            "train": [int(item) for item in permutation[:3200]],
            "unused": [int(item) for item in permutation[3200:8000]],
            "reserved_test": [int(item) for item in permutation[8000:9000]],
            "validation": [int(item) for item in permutation[9000:]],
        },
        "response_access": {
            "train_materialized": True,
            "validation_materialized": True,
            "reserved_test_materialized": False,
            "unused_materialized": False,
        },
    }


def synthetic_arrays() -> dict[str, object]:
    views = np.arange(10_000, dtype=np.float64)[:, None, None]
    channels = np.arange(16, dtype=np.float64)[None, :, None]
    xyz = np.arange(3, dtype=np.float64)[None, None, :]
    tx = views * 1.0e-4 + channels * 1.0e-6 + xyz * 1.0e-8
    rx = -views * 1.0e-4 - channels * 1.0e-6 - xyz * 1.0e-8
    return {
        "response": None,
        "meta": {
            "target_type": "B787",
            "experiment": "sphere10k",
            "radar_fc_hz": 10.0e9,
            "radar_bandwidth_hz": 3.0e9,
            "num_adc_samples": 600,
        },
        "tx_pos": np.ascontiguousarray(tx, dtype=np.float64),
        "rx_pos": np.ascontiguousarray(rx, dtype=np.float64),
    }


def synthetic_final(recipe: dict[str, object]) -> dict[str, object]:
    grid = int(recipe["scene"]["granularity"])
    extent = float(recipe["scene"]["extent_m"])
    w_re = np.zeros((grid, grid, grid), dtype=np.float32)
    w_im = np.zeros_like(w_re)
    active = np.zeros((grid, grid, grid), dtype=bool)
    for index, value in (((3, 3, 3), 1.0), ((4, 4, 4), 0.9), ((5, 5, 5), 0.8)):
        w_re[index] = value
        active[index] = True
    gain_log_mag = np.asarray(0.0, dtype=np.float32)
    gain_phase = np.asarray(0.0, dtype=np.float32)
    completed_adam_steps = int(recipe["fit"]["epochs"]) * stage1.B787_3200_NUM_TRAIN
    state = {
        "epoch": int(recipe["fit"]["epochs"]),
        "loss": 1.0,
        "scene_repr": "grid",
        "range_model": recipe["observation"]["range_model"],
        "extent": extent,
        "granularity": grid,
        "l1_weight": float(recipe["fit"]["l1_weight"]),
        "adam_eps": float(recipe["fit"]["optimizer"]["eps"]),
        "adaptive_capacity_v2": False,
        "val_cap_axis": None,
        "sealed_npz_protocol_contract": copy.deepcopy(recipe["sealed_protocol_identity"]),
        "execution_contract": stage1.expected_generic_execution_contract(recipe),
        "gain_state_dict": {
            "log_mag": gain_log_mag,
            "phase": gain_phase,
            "initialized": np.asarray(True, dtype=bool),
        },
        "optimizer_state_dict": {
            "state": {
                0: {
                    "step": np.asarray(completed_adam_steps, dtype=np.float32),
                    "exp_avg": np.zeros_like(w_re),
                    "exp_avg_sq": np.zeros_like(w_re),
                },
                1: {
                    "step": np.asarray(completed_adam_steps, dtype=np.float32),
                    "exp_avg": np.zeros_like(w_im),
                    "exp_avg_sq": np.zeros_like(w_im),
                },
                2: {
                    "step": np.asarray(completed_adam_steps, dtype=np.float32),
                    "exp_avg": np.zeros_like(gain_log_mag),
                    "exp_avg_sq": np.zeros_like(gain_log_mag),
                },
                3: {
                    "step": np.asarray(completed_adam_steps, dtype=np.float32),
                    "exp_avg": np.zeros_like(gain_phase),
                    "exp_avg_sq": np.zeros_like(gain_phase),
                },
            },
            "param_groups": [
                {
                    "params": [0, 1],
                    "lr": float(recipe["fit"]["optimizer"]["lr"]),
                    "eps": float(recipe["fit"]["optimizer"]["eps"]),
                    "weight_decay": float(recipe["fit"]["optimizer"]["weight_decay"]),
                    "betas": tuple(recipe["fit"]["optimizer"]["betas"]),
                },
                {
                    "params": [2, 3],
                    "lr": float(recipe["fit"]["optimizer"]["lr"]),
                    "eps": float(recipe["fit"]["optimizer"]["eps"]),
                    "weight_decay": float(recipe["fit"]["optimizer"]["weight_decay"]),
                    "betas": tuple(recipe["fit"]["optimizer"]["betas"]),
                },
            ],
        },
        "scheduler_state_dict": {
            "T_0": int(recipe["fit"]["scheduler"]["t0"]),
            "T_mult": int(recipe["fit"]["scheduler"]["t_mult"]),
            "eta_min": float(recipe["fit"]["scheduler"]["eta_min"]),
            "last_epoch": int(recipe["fit"]["epochs"]),
            "base_lrs": [float(recipe["fit"]["optimizer"]["lr"])] * 2,
            "_last_lr": [float(recipe["fit"]["optimizer"]["lr"])] * 2,
            "T_i": 160,
            "T_cur": 0,
        },
        "model_state_dict": {
            "w_re": w_re,
            "w_im": w_im,
            "active_mask": active,
            "grid_positions": stage1.expected_cell_centred_grid(grid, extent, dtype=np.dtype(np.float32)),
        },
    }
    return state


def synthetic_rng_payload() -> dict[str, object]:
    numpy_state = np.random.RandomState(7).get_state()
    return {
        "version": 2,
        "python": random.Random(7).getstate(),
        "numpy": {
            "format": "numpy_randomstate_v1",
            "algorithm": numpy_state[0],
            "keys": np.asarray(numpy_state[1], dtype=np.uint32),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": np.arange(16, dtype=np.uint8),
        "freq_cpu": np.arange(16, dtype=np.uint8),
    }


def synthetic_latest(recipe: dict[str, object]) -> dict[str, object]:
    latest = synthetic_final(recipe)
    latest["epoch"] = int(recipe["fit"]["epochs"]) - 1
    scheduler = latest["scheduler_state_dict"]
    # T0=10, Tmult=2: epoch 149 is the final tick of the 80-step cycle.
    scheduler["last_epoch"] = 149
    scheduler["T_i"] = 80
    scheduler["T_cur"] = 79
    eta_min = float(recipe["fit"]["scheduler"]["eta_min"])
    base = float(recipe["fit"]["optimizer"]["lr"])
    current = eta_min + (base - eta_min) * (1.0 + np.cos(np.pi * 79.0 / 80.0)) / 2.0
    scheduler["_last_lr"] = [float(current), float(current)]
    for group in latest["optimizer_state_dict"]["param_groups"]:
        group["lr"] = float(current)
        group["initial_lr"] = base
    for moment in latest["optimizer_state_dict"]["state"].values():
        moment["step"] = np.asarray(149 * stage1.B787_3200_NUM_TRAIN, dtype=np.float32)
    latest["rng_state"] = synthetic_rng_payload()
    return latest


def main() -> None:
    contract = canonical_contract()
    sealed = stage1.canonical_sealed_identity(contract)
    assert sealed["role_ids"]["train"] == contract["role_ids"]["train"]
    assert sealed["role_ids"]["validation"] == contract["role_ids"]["validation"]

    changed = copy.deepcopy(contract)
    changed["role_ids"]["train"][0], changed["role_ids"]["train"][1] = (
        changed["role_ids"]["train"][1],
        changed["role_ids"]["train"][0],
    )
    expect_reject("reordered train role", lambda: stage1.canonical_sealed_identity(changed))
    changed = copy.deepcopy(contract)
    changed["response_access"]["reserved_test_materialized"] = True
    expect_reject("materialized test role", lambda: stage1.canonical_sealed_identity(changed))
    changed = copy.deepcopy(contract)
    changed["response_shape"] = [10_000, 16, 16, 1, 599]
    expect_reject("wrong response header", lambda: stage1.canonical_sealed_identity(changed))
    expect_reject(
        "wrong archive rejected before generic loader",
        lambda: stage1.load_b7873200_sealed_identity("/wrong/sphere10k.npz", "ignored.json"),
    )

    arrays = synthetic_arrays()
    acquisition = stage1.build_b7873200_acquisition_identity(arrays, sealed)
    assert acquisition["response_payload_materialized"] is False
    assert acquisition["frequency_hz"].shape == (600,)
    assert acquisition["rx_pos_m"].shape == (4200, 16, 3)
    assert acquisition["tx_pos_m"].shape == (4200, 16, 3)
    changed_arrays = synthetic_arrays()
    changed_arrays["response"] = np.empty((1,), dtype=np.complex64)
    expect_reject(
        "acquisition after raw response",
        lambda: stage1.build_b7873200_acquisition_identity(changed_arrays, sealed),
    )

    recipe = stage1.default_stage1_recipe(sealed, acquisition)
    accepted = stage1.validate_stage1_recipe(recipe, sealed, acquisition)
    assert accepted["observation"]["num_freq_wanted"] == 600
    assert accepted["observation"]["phase_sign"] == -1.0
    assert accepted["observation"]["range_model"] == "product"
    changed_recipe = copy.deepcopy(recipe)
    changed_recipe["observation"]["phase_sign"] = 1.0
    expect_reject("changed phase", lambda: stage1.validate_stage1_recipe(changed_recipe, sealed, acquisition))
    changed_recipe = copy.deepcopy(recipe)
    changed_recipe["scene"]["granularity"] = 32
    expect_reject("changed grid", lambda: stage1.validate_stage1_recipe(changed_recipe, sealed, acquisition))

    final = synthetic_final(recipe)
    bundle = stage1.build_stage1_final_bundle(final, recipe)
    record = bundle[stage1.STAGE1_BUNDLE_FIELD]
    assert record["role"] == "checkpoint_final"
    assert record["structural_audit"]["retained_count"] == 3
    changed_final = copy.deepcopy(final)
    changed_final["execution_contract"]["physics"]["phase_sign"] = 1.0
    expect_reject("wrong derived phase contract", lambda: stage1.build_stage1_final_bundle(changed_final, recipe))
    changed_final = copy.deepcopy(final)
    changed_final["gain_state_dict"]["phase"] = np.asarray(np.nan, dtype=np.float32)
    expect_reject("nonfinite gain", lambda: stage1.build_stage1_final_bundle(changed_final, recipe))
    changed_final = copy.deepcopy(final)
    changed_final["optimizer_state_dict"]["param_groups"][0]["eps"] = 1.0e-8
    expect_reject("wrong optimizer epsilon", lambda: stage1.build_stage1_final_bundle(changed_final, recipe))
    changed_final = copy.deepcopy(final)
    changed_final["epoch"] -= 1
    expect_reject("nonterminal final", lambda: stage1.build_stage1_final_bundle(changed_final, recipe))
    changed_final = copy.deepcopy(final)
    changed_final["model_state_dict"]["grid_positions"][0, 0, 0, 0] += 0.01
    expect_reject("translated grid", lambda: stage1.build_stage1_final_bundle(changed_final, recipe))
    changed_final = copy.deepcopy(final)
    changed_final["model_state_dict"]["active_mask"] = changed_final["model_state_dict"]["active_mask"].astype(np.uint8)
    expect_reject("nonboolean mask", lambda: stage1.build_stage1_final_bundle(changed_final, recipe))
    changed_final = copy.deepcopy(final)
    changed_final["model_state_dict"]["w_re"][3, 3, 3] = np.nan
    expect_reject("nonfinite grid", lambda: stage1.build_stage1_final_bundle(changed_final, recipe))
    changed_final = copy.deepcopy(final)
    changed_final["model_state_dict"]["active_mask"][4, 4, 4] = False
    changed_final["model_state_dict"]["active_mask"][5, 5, 5] = False
    expect_reject("too few retained centres", lambda: stage1.build_stage1_final_bundle(changed_final, recipe))

    latest = synthetic_latest(recipe)
    stage1.validate_b7873200_stage1_recovery_state(latest, recipe)
    changed_latest = copy.deepcopy(latest)
    changed_latest.pop("rng_state")
    expect_reject(
        "latest without full RNG state",
        lambda: stage1.validate_b7873200_stage1_recovery_state(changed_latest, recipe),
    )
    changed_latest = copy.deepcopy(latest)
    changed_latest.pop("gain_state_dict")
    expect_reject(
        "latest without gain state",
        lambda: stage1.validate_b7873200_stage1_recovery_state(changed_latest, recipe),
    )
    changed_latest = copy.deepcopy(latest)
    changed_latest["optimizer_state_dict"] = None
    expect_reject(
        "latest without AdamW state",
        lambda: stage1.validate_b7873200_stage1_recovery_state(changed_latest, recipe),
    )
    changed_latest = copy.deepcopy(latest)
    changed_latest["scheduler_state_dict"]["T_cur"] = 78
    expect_reject(
        "latest with inconsistent scheduler cycle",
        lambda: stage1.validate_b7873200_stage1_recovery_state(changed_latest, recipe),
    )
    changed_latest = copy.deepcopy(latest)
    changed_latest["optimizer_state_dict"]["param_groups"][0]["eps"] = 0.0
    expect_reject(
        "latest with zero Adam epsilon",
        lambda: stage1.validate_b7873200_stage1_recovery_state(changed_latest, recipe),
    )
    changed_latest = copy.deepcopy(latest)
    changed_latest["scheduler_state_dict"]["T_mult"] = 2.5
    expect_reject(
        "latest with fractional scheduler multiplier",
        lambda: stage1.validate_b7873200_stage1_recovery_state(changed_latest, recipe),
    )
    changed_latest = copy.deepcopy(latest)
    changed_latest["scheduler_state_dict"]["last_epoch"] = 149.9
    expect_reject(
        "latest with fractional scheduler epoch",
        lambda: stage1.validate_b7873200_stage1_recovery_state(changed_latest, recipe),
    )

    source_text = (ROOT / "rift" / "sugavanam_ertin_b7873200_stage1.py").read_text(encoding="utf-8")
    assert "hashlib" not in source_text and "sha256" not in source_text.lower()
    assert "/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/" in source_text
    assert "b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz" in source_text
    wrapper_text = (ROOT / "train_sugavanam_ertin_stage1.py").read_text(encoding="utf-8")
    for required in (
        "--npz-sealed-protocol",
        "--num-train", "3200",
        "--num-val", "1000",
        "--num-test", "1000",
        "--num-freq-wanted", "600",
        "--epochs", "150",
        "--scene-repr", "grid",
        "--forward-operator", "range",
        "--range-model", "product",
        "--phase-sign", "-1.0",
        "--execution-contract-label",
        "--require-full-resume-state",
        "--checkpoint-metric", "val",
        "_bundle_existing_final",
        "STAGE1_FINAL_BUNDLE_FILENAME",
    ):
        assert required in wrapper_text, required
    print("SE_B7873200_STAGE1_STATIC_PASS: 47 semantic provenance checks", flush=True)


if __name__ == "__main__":
    main()
