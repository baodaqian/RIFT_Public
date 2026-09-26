#!/usr/bin/env python3
"""Focused CPU contract checks for the Adaptive B787 signal evaluator.

This validator uses a tiny synthetic response cube and metadata-only checkpoint
contracts.  It does not load a production checkpoint, read the B787 archive, or
write any experiment artifact.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rift.radar_fields_dataset import (  # noqa: E402
    load_radar_fields_npz,
    restrict_radar_fields_response_views,
    split_view_indices,
)
from scripts.eval_b787_range_power import (  # noqa: E402
    _sealed_validation_indices,
    resolve_scene_extent,
    validate_normalization_stats,
)


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(name + (f": {detail}" if detail else ""))
    print(f"PASS: {name}" + (f" -- {detail}" if detail else ""))


def expect_error(name: str, callback, fragment: str) -> None:
    try:
        callback()
    except (TypeError, ValueError, PermissionError) as exc:
        check(name, fragment in str(exc), str(exc))
    else:
        raise AssertionError(f"{name}: expected an exception containing {fragment!r}")


def sealed_checkpoint() -> tuple[dict, np.ndarray, np.ndarray]:
    train, validation, test = split_view_indices(10_000, 3_200, 1_000, 1_000, 42, True)
    unused = np.setdiff1d(
        np.arange(10_000, dtype=np.int64),
        np.concatenate((train, validation, test)),
        assume_unique=False,
    )
    checkpoint = {
        "extent": None,
        "execution_contract": {"scene": {"extent_m": 0.15}},
        "sealed_npz_protocol_contract": {
            "response_shape": [10_000, 16, 16, 1, 600],
            "response_dtype": "complex64",
            "role_ids": {
                "train": train.tolist(),
                "validation": validation.tolist(),
                "reserved_test": test.tolist(),
                "unused": unused.tolist(),
            }
        },
    }
    return checkpoint, train, validation


def main() -> None:
    print("1. extent metadata resolution")
    check("valid top-level extent wins", resolve_scene_extent({"extent": 0.2}) == 0.2)
    check(
        "nested execution contract repairs point-scene None",
        resolve_scene_extent({"extent": None, "execution_contract": {"scene": {"extent_m": 0.15}}})
        == 0.15,
    )
    expect_error(
        "unusable extent metadata fails closed",
        lambda: resolve_scene_extent(
            {"extent": None, "execution_contract": {"scene": {"extent_m": float("nan")}}}
        ),
        "finite positive scene extent",
    )

    print("\n2. sealed validation role binding")
    checkpoint, train, validation = sealed_checkpoint()
    arrays = SimpleNamespace(
        num_views=10_000,
        num_tx=16,
        num_rx=16,
        num_freq=600,
        response_shape=(10_000, 16, 16, 1, 600),
        response_dtype=np.dtype("complex64"),
    )
    args = SimpleNamespace(num_train=3_200, num_val=1_000, seed=42, max_views=0)
    selected = _sealed_validation_indices(checkpoint, arrays, args)
    check("checkpoint validation order matches seed-42 fixed tail", np.array_equal(selected, validation))
    args.max_views = 7
    check("max_views preserves validation prefix", np.array_equal(_sealed_validation_indices(checkpoint, arrays, args), validation[:7]))
    bad_checkpoint = dict(checkpoint)
    bad_contract = dict(checkpoint["sealed_npz_protocol_contract"])
    bad_roles = dict(bad_contract["role_ids"])
    bad_roles["validation"] = list(validation)
    bad_roles["validation"][0], bad_roles["validation"][1] = (
        bad_roles["validation"][1],
        bad_roles["validation"][0],
    )
    bad_contract["role_ids"] = bad_roles
    bad_checkpoint["sealed_npz_protocol_contract"] = bad_contract
    args.max_views = 0
    expect_error(
        "checkpoint validation mismatch is rejected",
        lambda: _sealed_validation_indices(bad_checkpoint, arrays, args),
        "validation IDs disagree",
    )
    bad_header_checkpoint = dict(checkpoint)
    bad_header_contract = dict(checkpoint["sealed_npz_protocol_contract"])
    bad_header_contract["response_shape"] = [10_000, 16, 16, 2, 600]
    bad_header_checkpoint["sealed_npz_protocol_contract"] = bad_header_contract
    expect_error(
        "checkpoint response header mismatch is rejected",
        lambda: _sealed_validation_indices(bad_header_checkpoint, arrays, args),
        "response headers",
    )

    print("\n3. normalization provenance")
    stats = {
        "peak_power": 1.0853136e-7,
        "dynamic_range_db": 60.0,
        "train_view_indices": train.tolist(),
        "train_view_count": len(train),
        "normalization_scan_view_indices": train[:128].tolist(),
        "normalization_provenance": {
            "version": 1,
            "normalization_scan_role": "train",
            "train_view_indices": train.tolist(),
            "normalization_scan_view_indices": train[:128].tolist(),
        },
    }
    peak, dynamic = validate_normalization_stats(stats, checkpoint, num_views=10_000)
    check("finite train-only stats accepted", peak == stats["peak_power"] and dynamic == 60.0)
    mismatched = dict(stats)
    mismatched["train_view_indices"] = train[::-1].tolist()
    expect_error(
        "stats TRAIN IDs must match checkpoint",
        lambda: validate_normalization_stats(mismatched, checkpoint, num_views=10_000),
        "train IDs disagree",
    )
    test_scan = dict(stats)
    test_scan["normalization_scan_view_indices"] = [9_999]
    test_scan["normalization_provenance"] = dict(stats["normalization_provenance"])
    test_scan["normalization_provenance"]["normalization_scan_view_indices"] = [9_999]
    expect_error(
        "stats scan cannot include TEST/unused IDs",
        lambda: validate_normalization_stats(test_scan, checkpoint, num_views=10_000),
        "TRAIN role",
    )
    bad_peak = dict(stats)
    bad_peak["peak_power"] = 0.0
    expect_error(
        "nonpositive peak is rejected",
        lambda: validate_normalization_stats(bad_peak, checkpoint, num_views=10_000),
        "finite and positive",
    )

    print("\n4. lazy restricted response access")
    with tempfile.TemporaryDirectory(prefix="rift_eval_contract_") as temporary:
        path = Path(temporary) / "tiny.npz"
        response = np.arange(5 * 1 * 1 * 1 * 4, dtype=np.float32).reshape(5, 1, 1, 1, 4)
        response = response.astype(np.complex64)
        metadata = {"radar_fc_hz": 10.0, "radar_bandwidth_hz": 4.0, "num_adc_samples": 4}
        np.savez(
            path,
            response=response,
            metadata_json=np.asarray(json.dumps(metadata)),
            viewpoint_positions=np.zeros((5, 3), dtype=np.float32),
            tx_pos=np.zeros((5, 1, 3), dtype=np.float32),
            rx_pos=np.zeros((5, 1, 3), dtype=np.float32),
        )
        lazy = load_radar_fields_npz(str(path), load_response=False)
        restricted = restrict_radar_fields_response_views(lazy, [0, 1])
        check("lazy loader keeps response unmaterialized", not restricted.response_is_materialized)
        check("authorized selected row is readable", restricted.response_view(1).shape == (1, 1, 1, 4))
        expect_error(
            "reserved row is denied before payload access",
            lambda: restricted.response_view(2),
            "denied source-view IDs",
        )

    print("\n5. evaluator reads each selected response view once")
    source = Path(__file__).with_name("eval_b787_range_power.py").read_text(encoding="utf-8")
    check("evaluator no longer indexes eager response", "arrays.response[view_index]" not in source)
    check(
        "pending responses stream through one iterator",
        "arrays.iter_response_views(pending_view_ids)" in source,
    )
    check("evaluator requests lazy response loading", "load_response=False" in source)

    print("EVAL_B787_RANGE_POWER_VALIDATION_PASS")


if __name__ == "__main__":
    main()
