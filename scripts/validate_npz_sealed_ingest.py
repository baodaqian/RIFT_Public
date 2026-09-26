#!/usr/bin/env python
"""Data-free regression gates for opt-in sealed NPZ response ingestion.

This script creates tiny local NPZ archives only.  It never opens a PACE
dataset, trains a model, or writes an experiment artifact.  Run it on an
allocated environment with PyTorch available:

    python scripts/validate_npz_sealed_ingest.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rift.npz_dataset as npz_dataset  # noqa: E402
from rift.npz_dataset import (  # noqa: E402
    PecSphereNPZDataset,
    build_npz_dataloaders,
    decode_metadata_json,
    get_npz_response_view,
    iter_npz_response_views,
    load_npz_arrays,
    npz_response_num_views,
    restrict_npz_response_views,
)


CHECKS = 0


def check(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)
    print(f"  PASS: {message}")


def synthetic_arrays():
    metadata = {
        "radar_fc_hz": 79.0e9,
        "radar_bandwidth_hz": 3.0e9,
        "num_adc_samples": 4,
        "target_radius_m": 0.75,
    }
    # Each source view is visibly distinct, including the response payload
    # that must remain inaccessible in the sealed-test check below.
    values = np.arange(5 * 2 * 1 * 2 * 4, dtype=np.float32).reshape(5, 2, 1, 2, 4)
    response = (values + 1j * (1000.0 + values)).astype(np.complex64)
    angles = np.linspace(0.2, 1.2, response.shape[0], dtype=np.float32)
    viewpoints = np.stack((np.cos(angles), np.sin(angles), 0.25 * np.ones_like(angles)), axis=1)
    tx = np.zeros((response.shape[0], response.shape[1], 3), dtype=np.float32)
    rx = np.zeros((response.shape[0], response.shape[2], 3), dtype=np.float32)
    for view in range(response.shape[0]):
        tx[view, :, 0] = float(view + 1)
        rx[view, :, 1] = float(view + 1)
    return metadata, response, viewpoints, tx, rx


def write_npz(path: Path, *, metadata_json, compressed: bool):
    metadata, response, viewpoints, tx, rx = synthetic_arrays()
    writer = np.savez_compressed if compressed else np.savez
    writer(
        path,
        response=response,
        viewpoint_positions=viewpoints,
        tx_pos=tx,
        rx_pos=rx,
        metadata_json=metadata_json,
    )
    return metadata, response


def item_equal(actual, expected):
    return all(torch.equal(left, right) for left, right in zip(actual, expected))


def stage_metadata_and_lazy_views(root: Path):
    print("Stage A: singleton/bytes metadata and bounded selected-view reads")
    metadata, response, _viewpoints, _tx, _rx = synthetic_arrays()
    encodings = (
        ("scalar", np.asarray(json.dumps(metadata))),
        ("singleton", np.asarray([json.dumps(metadata)])),
        ("bytes_singleton", np.asarray([json.dumps(metadata).encode("utf-8")])),
    )
    for compressed in (False, True):
        for name, encoded in encodings:
            path = root / f"{name}_{'compressed' if compressed else 'stored'}.npz"
            write_npz(path, metadata_json=encoded, compressed=compressed)

            eager = load_npz_arrays(path)
            check(
                set(eager) == {"response", "viewpoint_positions", "tx_pos", "rx_pos", "meta"},
                f"{name}/{compressed} eager dictionary keeps the legacy key contract",
            )
            check(
                eager["meta"] == metadata and np.array_equal(eager["response"], response),
                f"{name}/{compressed} eager loader preserves historical metadata and response values",
            )

            lazy = load_npz_arrays(path, load_response=False)
            check(lazy["response"] is None, f"{name}/{compressed} lazy loader retains no full response")
            check(
                lazy["meta"] == metadata and npz_response_num_views(lazy) == response.shape[0],
                f"{name}/{compressed} lazy loader decodes metadata and reads only the header contract",
            )
            restricted = restrict_npz_response_views(lazy, [1, 3])
            check(
                np.array_equal(get_npz_response_view(restricted, 3), response[3]),
                f"{name}/{compressed} authorized source view streams exactly",
            )
            try:
                get_npz_response_view(restricted, 4)
            except PermissionError:
                denied = True
            else:
                denied = False
            check(denied, f"{name}/{compressed} sealed source view is rejected before payload read")
            try:
                restrict_npz_response_views(restricted, [1, 4])
            except PermissionError:
                widening_denied = True
            else:
                widening_denied = False
            check(widening_denied, f"{name}/{compressed} restriction cannot be widened")
            streamed = list(iter_npz_response_views(restricted, [3, 1, 3]))
            check(
                [view for view, _payload in streamed] == [1, 3]
                and np.array_equal(streamed[0][1], response[1])
                and np.array_equal(streamed[1][1], response[3]),
                f"{name}/{compressed} streaming is unique, ordered, and payload-exact",
            )

            eager_dataset = PecSphereNPZDataset(eager, [3, 1], source_path=path)
            lazy_dataset = PecSphereNPZDataset(restricted, [3, 1], source_path=path)
            check(
                len(lazy_dataset) == 2
                and np.array_equal(lazy_dataset.indices, np.asarray([3, 1]))
                and item_equal(lazy_dataset[0], eager_dataset[0])
                and item_equal(lazy_dataset[1], eager_dataset[1]),
                f"{name}/{compressed} lazy dataset matches eager tensor items in seeded source order",
            )


def stage_role_restricted_builder(root: Path):
    print("Stage B: role-restricted builder excludes sealed test payload")
    metadata, response, _viewpoints, _tx, _rx = synthetic_arrays()
    path = root / "builder.npz"
    write_npz(path, metadata_json=np.asarray(json.dumps(metadata)), compressed=True)

    calls = []
    original = npz_dataset.iter_npz_response_views

    def recording_iterator(arrays, view_indices):
        requested = tuple(int(value) for value in view_indices)
        calls.append(requested)
        yield from original(arrays, requested)

    npz_dataset.iter_npz_response_views = recording_iterator
    try:
        training, validation, test = build_npz_dataloaders(
            path,
            num_train=2,
            num_val=1,
            num_test=2,
            seed=23,
            lazy_response=True,
            include_test=False,
        )
    finally:
        npz_dataset.iter_npz_response_views = original

    train_ids = set(int(value) for value in training.dataset.indices)
    val_ids = set(int(value) for value in validation.dataset.indices)
    requested_ids = set(value for call in calls for value in call)
    expected_test = set(np.random.default_rng(23).permutation(response.shape[0])[3:5])
    check(test is None, "sealed builder returns no test DataLoader when include_test=False")
    check(
        requested_ids == train_ids | val_ids and requested_ids.isdisjoint(expected_test),
        "sealed builder streams only train/validation IDs and never constructs a test payload request",
    )
    # Dataset construction fully consumed the selected response stream; batch
    # iteration below should use cached tensor items and not reopen the archive.
    train_batch = next(iter(training))
    val_batch = next(iter(validation))
    check(
        train_batch[0].shape[0] == 1 and val_batch[0].shape[0] == 1,
        "sealed train/validation DataLoaders remain ordinary batch_size=1 loaders",
    )

    eager_train, eager_val, eager_test = build_npz_dataloaders(
        path,
        num_train=2,
        num_val=1,
        num_test=2,
        seed=23,
    )
    check(
        eager_test is not None
        and np.array_equal(eager_train.dataset.indices, training.dataset.indices)
        and np.array_equal(eager_val.dataset.indices, validation.dataset.indices),
        "default builder remains eager and preserves the same seeded split IDs",
    )


def stage_invalid_metadata():
    print("Stage C: invalid metadata remains rejected")
    for raw in (np.asarray(["{}", "{}"]), np.asarray("[]"), np.asarray(7)):
        try:
            decode_metadata_json(raw)
        except ValueError:
            rejected = True
        else:
            rejected = False
        check(rejected, "invalid metadata representation is rejected")


def main():
    with tempfile.TemporaryDirectory(prefix="rift_npz_sealed_ingest_") as temporary:
        root = Path(temporary)
        stage_metadata_and_lazy_views(root)
        stage_role_restricted_builder(root)
        stage_invalid_metadata()
    print(f"PASS: {CHECKS} NPZ sealed-ingestion checks")


if __name__ == "__main__":
    main()
