#!/usr/bin/env python
"""Materialize a group-safe interpolation-only PublicRadar split.

The source NPZ is never modified.  Measurements, geometry, frequencies, and
stored acquisition groups are copied byte-for-value into a new NPZ; only the
three role-index arrays and split metadata change.  A complete viewpoint table
is written beside the new dataset for auditability.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = PROJECT_ROOT / "rift" / "public_radar_tuning.py"
_SPEC = importlib.util.spec_from_file_location(
    "rift_public_radar_tuning_interleaved_v2", HELPER_PATH
)
_HELPERS = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _HELPERS
_SPEC.loader.exec_module(_HELPERS)
interleaved_group_split = _HELPERS.interleaved_group_split
validate_embedded_partition = _HELPERS.validate_embedded_partition


SCHEMA = "rift.public_radar_interleaved_split_v2"
EXPECTED = {
    "cvdomes_camry": {
        "views": (18432, 2304, 2304),
        "groups": (576, 72, 72),
        "metadata": {"dataset": "cvdomes", "polarization": "hh"},
    },
    "gotcha_pass2_hh": {
        "views": (33939, 4242, 4243),
        "groups": (288, 36, 36),
        "metadata": {
            "dataset": "gotcha",
            "pass_id": "pass2",
            "polarization": "hh",
        },
    },
}
ROLE_NAMES = ("train", "validation", "test")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=tuple(EXPECTED), required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _role_summary(arrays, roles, partition):
    groups = np.asarray(arrays["split_group_id"], dtype=np.int64)
    azimuth = np.asarray(arrays["view_azimuth_deg"], dtype=np.float64)
    elevation = np.asarray(arrays["view_elevation_deg"], dtype=np.float64)
    result = {
        "schema": SCHEMA,
        "partition": {
            key: value
            for key, value in partition.items()
            if key not in ("ordered_group_ids", "ordered_group_azimuth_deg", "groups")
        },
        "roles": {},
    }
    train_group_angles = _group_angles(arrays, roles["train"])
    for role in ROLE_NAMES:
        indices = roles[role]
        role_azimuth = azimuth[indices]
        role_elevation = elevation[indices]
        result["roles"][role] = {
            "views": int(indices.size),
            "groups": int(np.unique(groups[indices]).size),
            "azimuth_min_deg": float(role_azimuth.min()),
            "azimuth_max_deg": float(role_azimuth.max()),
            "elevation_min_deg": float(role_elevation.min()),
            "elevation_max_deg": float(role_elevation.max()),
            "unique_elevations": int(np.unique(np.round(role_elevation, 6)).size),
            "max_group_azimuth_gap_deg": _max_circular_gap_deg(
                _group_angles(arrays, indices)
            ),
            "max_nearest_train_group_deg": _max_nearest_circular_deg(
                _group_angles(arrays, indices), train_group_angles
            ),
        }
    return result


def _group_angles(arrays, indices):
    groups = np.asarray(arrays["split_group_id"], dtype=np.int64)
    azimuth = np.asarray(arrays["view_azimuth_deg"], dtype=np.float64)
    values = []
    for group in np.unique(groups[np.asarray(indices, dtype=np.int64)]):
        radians = np.deg2rad(azimuth[groups == group])
        mean_vector = np.exp(1j * radians).mean()
        if abs(mean_vector) <= 1.0e-12:
            raise ValueError(f"group {int(group)} has undefined circular azimuth")
        values.append(float(np.degrees(np.angle(mean_vector)) % 360.0))
    return np.asarray(values, dtype=np.float64)


def _max_circular_gap_deg(angles):
    angles = np.sort(np.mod(np.asarray(angles, dtype=np.float64), 360.0))
    if angles.size < 2:
        raise ValueError("circular coverage needs at least two groups")
    return float(np.diff(np.concatenate((angles, angles[:1] + 360.0))).max())


def _max_nearest_circular_deg(query, reference):
    query = np.mod(np.asarray(query, dtype=np.float64), 360.0)
    reference = np.mod(np.asarray(reference, dtype=np.float64), 360.0)
    distance = np.abs(((query[:, None] - reference[None, :] + 180.0) % 360.0) - 180.0)
    return float(distance.min(axis=1).max())


def _validate_expected(scene, arrays, roles, partition):
    expected = EXPECTED[scene]
    actual_views = tuple(int(roles[role].size) for role in ROLE_NAMES)
    actual_groups = tuple(
        int(partition["groups_by_role"][role]) for role in ROLE_NAMES
    )
    if actual_views != expected["views"]:
        raise ValueError(f"{scene} role-view counts changed: {actual_views}")
    if actual_groups != expected["groups"]:
        raise ValueError(f"{scene} role-group counts changed: {actual_groups}")
    train_angles = _group_angles(arrays, roles["train"])
    max_role_gap = 6.0 if scene == "cvdomes_camry" else 11.0
    max_heldout_to_train = 1.1 if scene == "cvdomes_camry" else 2.1
    for role in ROLE_NAMES:
        role_angles = _group_angles(arrays, roles[role])
        gap = _max_circular_gap_deg(role_angles)
        if gap > max_role_gap:
            raise ValueError(f"{scene} {role} circular coverage gap is {gap:.6f} degrees")
        nearest = _max_nearest_circular_deg(role_angles, train_angles)
        if nearest > max_heldout_to_train:
            raise ValueError(
                f"{scene} {role} is {nearest:.6f} degrees from its nearest train group"
            )
    if scene == "cvdomes_camry":
        elevation = np.asarray(arrays["view_elevation_deg"], dtype=np.float64)
        for role in ROLE_NAMES:
            levels = np.unique(np.round(elevation[roles[role]], 3))
            if levels.tolist() != [30.0, 40.0, 50.0, 60.0]:
                raise ValueError(f"Camry {role} elevation coverage changed: {levels}")


def _metadata_with_split(source, metadata, scene, seed, roles, partition):
    updated = dict(metadata)
    updated.update(
        {
            "split_parent_npz": str(Path(source).resolve()),
            "split_parent_strategy": metadata.get("split_strategy"),
            "split_schema": SCHEMA,
            "split_strategy": partition["strategy"],
            "split_seed": int(seed),
            "split_role_slots": {
                "validation": 0,
                "test": 5,
                "period": 10,
            },
            "split_view_counts": {
                role: int(roles[role].size) for role in ROLE_NAMES
            },
            "split_group_counts": dict(partition["groups_by_role"]),
            "split_scene": scene,
            "split_interpolation_only": True,
        }
    )
    return updated


def _write_viewpoint_table(path, arrays, roles):
    n_view = int(np.asarray(arrays["split_group_id"]).size)
    role_by_view = np.empty(n_view, dtype="U10")
    role_position = np.full(n_view, -1, dtype=np.int64)
    for role in ROLE_NAMES:
        indices = roles[role]
        role_by_view[indices] = role
        role_position[indices] = np.arange(indices.size, dtype=np.int64)
    if np.any(role_position < 0):
        raise AssertionError("viewpoint table has an unassigned view")

    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("view_index", "azimuth_deg", "elevation_deg", "group_id", "role", "role_position")
        )
        for index in range(n_view):
            writer.writerow(
                (
                    index,
                    format(float(arrays["view_azimuth_deg"][index]), ".12g"),
                    format(float(arrays["view_elevation_deg"][index]), ".12g"),
                    int(arrays["split_group_id"][index]),
                    role_by_view[index],
                    int(role_position[index]),
                )
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_viewpoint_table(path, arrays, roles):
    n_view = int(np.asarray(arrays["split_group_id"]).size)
    role_by_view = np.empty(n_view, dtype="U10")
    role_position = np.full(n_view, -1, dtype=np.int64)
    for role in ROLE_NAMES:
        role_by_view[roles[role]] = role
        role_position[roles[role]] = np.arange(roles[role].size, dtype=np.int64)
    expected_header = [
        "view_index",
        "azimuth_deg",
        "elevation_deg",
        "group_id",
        "role",
        "role_position",
    ]
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != expected_header:
            raise ValueError("existing viewpoint table header changed")
        for expected_index in range(n_view):
            row = next(reader, None)
            if row is None or len(row) != len(expected_header):
                raise ValueError("existing viewpoint table is truncated")
            expected = [
                str(expected_index),
                format(float(arrays["view_azimuth_deg"][expected_index]), ".12g"),
                format(float(arrays["view_elevation_deg"][expected_index]), ".12g"),
                str(int(arrays["split_group_id"][expected_index])),
                str(role_by_view[expected_index]),
                str(int(role_position[expected_index])),
            ]
            if row != expected:
                raise ValueError(
                    f"existing viewpoint table changed at view {expected_index}"
                )
        if next(reader, None) is not None:
            raise ValueError("existing viewpoint table has extra rows")


def _validate_existing(output, source_arrays, roles, metadata):
    with np.load(output, allow_pickle=True) as existing:
        expected_keys = set(source_arrays)
        if set(existing.files) != expected_keys:
            raise ValueError("existing v2 NPZ key set disagrees with its parent")
        for key in expected_keys:
            if key in ("train_indices", "validation_indices", "test_indices", "metadata_json"):
                continue
            if (
                existing[key].shape != source_arrays[key].shape
                or existing[key].dtype != source_arrays[key].dtype
                or not np.array_equal(existing[key], source_arrays[key])
            ):
                raise ValueError(f"existing v2 NPZ changed immutable array {key}")
        for role in ROLE_NAMES:
            values = existing[f"{role}_indices"]
            if (
                values.shape != roles[role].shape
                or values.dtype != roles[role].dtype
                or not np.array_equal(values, roles[role])
            ):
                raise ValueError(f"existing v2 NPZ changed {role} membership/order")
        existing_metadata = json.loads(str(existing["metadata_json"]))
        if existing_metadata != metadata:
            raise ValueError("existing v2 NPZ metadata disagrees with this invocation")


def materialize(scene, source, output, seed=42):
    source = Path(source).resolve()
    output = Path(output).resolve()
    if source == output:
        raise ValueError("v2 output must not overwrite its v1 parent")
    if not source.is_file():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)

    with np.load(source, allow_pickle=True) as raw:
        arrays = {key: raw[key] for key in raw.files}
    required = {
        "response",
        "viewpoint_positions",
        "tx_pos",
        "rx_pos",
        "frequencies_hz",
        "view_azimuth_deg",
        "view_elevation_deg",
        "split_group_id",
        "train_indices",
        "validation_indices",
        "test_indices",
        "metadata_json",
    }
    if set(arrays) != required:
        raise ValueError(f"canonical NPZ key contract changed: {sorted(arrays)}")
    metadata = json.loads(str(arrays["metadata_json"]))
    if metadata.get("schema") != "rift_coherent_radar_v1":
        raise ValueError(f"unexpected parent schema {metadata.get('schema')!r}")
    expected_parent = EXPECTED[scene]
    for key, expected_value in expected_parent["metadata"].items():
        if metadata.get(key) != expected_value:
            raise ValueError(
                f"{scene} parent metadata {key} changed: {metadata.get(key)!r}"
            )
    if source.name != f"{scene}.npz" or source.parent.name != scene:
        raise ValueError(
            f"{scene} parent path must end in {scene}/{scene}.npz, got {source}"
        )

    roles, partition = interleaved_group_split(
        arrays["split_group_id"], arrays["view_azimuth_deg"], seed=seed
    )
    _validate_expected(scene, arrays, roles, partition)
    validate_embedded_partition(
        arrays["response"].shape[0],
        roles["train"],
        roles["validation"],
        roles["test"],
        arrays["split_group_id"],
    )
    metadata = _metadata_with_split(source, metadata, scene, seed, roles, partition)
    arrays["train_indices"] = roles["train"]
    arrays["validation_indices"] = roles["validation"]
    arrays["test_indices"] = roles["test"]
    arrays["metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )

    if output.exists():
        _validate_existing(output, arrays, roles, metadata)
    else:
        temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
        try:
            with temporary.open("wb") as handle:
                np.savez(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                temporary.unlink()
        _validate_existing(output, arrays, roles, metadata)

    table_path = output.with_suffix(".viewpoints.csv")
    summary_path = output.with_suffix(".split.json")
    if not table_path.exists():
        _write_viewpoint_table(table_path, arrays, roles)
    _validate_viewpoint_table(table_path, arrays, roles)
    summary = _role_summary(arrays, roles, partition)
    summary.update(
        {
            "scene": scene,
            "source": str(source),
            "output": str(output),
            "viewpoint_table": str(table_path),
        }
    )
    if summary_path.exists():
        existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing_summary != summary:
            raise ValueError("existing split summary disagrees with this invocation")
    else:
        _atomic_json(summary_path, summary)
    return summary


def main():
    args = parse_args()
    summary = materialize(args.scene, args.source, args.output, seed=args.seed)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
