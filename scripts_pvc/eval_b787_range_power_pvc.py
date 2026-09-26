#!/usr/bin/env python3
"""PVC twin of scripts/eval_b787_range_power.py for the train2400 collection runs.

The original (unchanged on disk) validates the normalization statistics against a
hard-coded 3200-view TRAIN role (the sealed B787 split), so every collection
checkpoint trained on 2400 views stops before any metric. This twin rebinds one
function, a copy of validate_normalization_stats whose expected TRAIN count is
the checkpoint's own training_selection.num_train (3200 when absent, as
before), and then runs the original main unchanged: the same coherent complex
RelMSE, normalized (dB) and linear range-power RelMSE, roles and reserved-test
opt-in. --device accepts cpu or xpu (the original checks CUDA only when asked for it).

    python scripts_pvc/eval_b787_range_power_pvc.py --object a320 --dataset-root D \
        --role-manifest RUN/role_manifest.json --checkpoint RUN/rift/checkpoint_best.pth.tar \
        --stats RF_RUN/radar_fields/radar_fields_power_stats.json --label a320_rift \
        --out-dir OUT --device cpu --role validation   # or: --role test --allow-reserved-test
"""
from __future__ import annotations

import math
import os
import sys
from collections.abc import Mapping

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import scripts.eval_b787_range_power as original  # noqa: E402
from scripts.eval_b787_range_power import _ordered_role_ids  # noqa: E402
from rift.radar_fields_dataset import _sealed_stats_cache_matches_train_role  # noqa: E402


def expected_train_count(checkpoint: Mapping[str, object]) -> int:
    """TRAIN role size recorded by the checkpoint's sealed collection contract; 3200 otherwise."""
    sealed = checkpoint.get("sealed_npz_protocol_contract")
    selection = sealed.get("training_selection") if isinstance(sealed, Mapping) else None
    if isinstance(selection, Mapping) and "num_train" in selection:
        return int(selection["num_train"])
    return 3200


def validate_normalization_stats(
    stats: Mapping[str, object],
    checkpoint: Mapping[str, object],
    *,
    num_views: int,
) -> tuple[float, float]:
    """Validate fixed dB normalization before any metric computation."""

    try:
        peak_power = float(stats["peak_power"])
        dynamic_range_db = float(stats["dynamic_range_db"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("normalization stats must contain numeric peak_power and dynamic_range_db") from exc
    if not math.isfinite(peak_power) or peak_power <= 0.0:
        raise ValueError(f"normalization stats peak_power must be finite and positive, got {peak_power!r}")
    if not math.isfinite(dynamic_range_db) or dynamic_range_db <= 0.0:
        raise ValueError(
            "normalization stats dynamic_range_db must be finite and positive, "
            f"got {dynamic_range_db!r}"
        )

    sealed = checkpoint.get("sealed_npz_protocol_contract")
    if isinstance(sealed, Mapping) and "dataset_identity" in sealed:
        from rift.rift_dataset import validate_checkpoint_object
        validate_checkpoint_object(stats, sealed)
    if sealed is None:
        return peak_power, dynamic_range_db

    # PVC adaptation: the TRAIN count comes from the checkpoint's own contract (train2400
    # collection runs); the sealed B787 split keeps 3200. The stats' TRAIN IDs must still
    # equal the checkpoint's TRAIN role exactly (checked below).
    train_indices = _ordered_role_ids(
        checkpoint, "train", num_views=num_views, expected_count=expected_train_count(checkpoint)
    )
    expected_train = train_indices.tolist()
    raw_stats_train = stats.get("train_view_indices")
    raw_stats_scan = stats.get("normalization_scan_view_indices")
    provenance = stats.get("normalization_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("sealed normalization stats lack normalization_provenance")
    if provenance.get("normalization_scan_role") != "train":
        raise ValueError("sealed normalization scan is not declared as a train-role scan")

    def decode_stats_ids(raw: object, label: str) -> list[int]:
        if isinstance(raw, (str, bytes)):
            raise ValueError(f"sealed normalization stats {label} must be an integer-ID sequence")
        try:
            values = tuple(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"sealed normalization stats {label} must be an integer-ID sequence"
            ) from exc
        decoded = []
        for value in values:
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"sealed normalization stats {label} contains a boolean ID")
            try:
                integer = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"sealed normalization stats {label} contains a non-integer ID") from exc
            if integer != value or not 0 <= integer < int(num_views):
                raise ValueError(
                    f"sealed normalization stats {label} contains invalid source-view ID {value!r}"
                )
            decoded.append(integer)
        if len(set(decoded)) != len(decoded):
            raise ValueError(f"sealed normalization stats {label} contains duplicate source-view IDs")
        return decoded

    stats_train = decode_stats_ids(raw_stats_train, "train_view_indices")
    stats_scan = decode_stats_ids(raw_stats_scan, "normalization_scan_view_indices")
    provenance_train = decode_stats_ids(
        provenance.get("train_view_indices"), "provenance.train_view_indices"
    )
    provenance_scan = decode_stats_ids(
        provenance.get("normalization_scan_view_indices"),
        "provenance.normalization_scan_view_indices",
    )
    if stats_train != expected_train or provenance_train != expected_train:
        raise ValueError("normalization stats train IDs disagree with the checkpoint TRAIN role")
    if stats_scan != provenance_scan:
        raise ValueError("normalization stats top-level and provenance scan IDs disagree")
    if not stats_scan or not set(stats_scan).issubset(set(expected_train)):
        raise ValueError("normalization scan IDs must be a nonempty subset of the TRAIN role")
    if not _sealed_stats_cache_matches_train_role(
        stats,
        requested_train_indices=expected_train,
        requested_scan_indices=stats_scan,
        num_views=num_views,
    ):
        raise ValueError(
            "normalization stats lack the existing sealed train-only provenance contract"
        )
    return peak_power, dynamic_range_db


def main(argv=None) -> None:
    original.validate_normalization_stats = validate_normalization_stats
    original.main(argv)


if __name__ == "__main__":
    main()
