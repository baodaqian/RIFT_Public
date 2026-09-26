"""Check the bounded RF B787 ROI diagnostic, without importing Torch or data payloads."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(record, manifest, role):
    split = manifest["split"]
    roles = {"train": split["train_indices"], "val": split["validation_indices"],
             "test": split["test_indices"]}
    unused = split["unused_indices"]
    require([len(roles[k]) for k in ("train", "val", "test")] == [3200, 1000, 1000]
            and len(unused) == 4800, "incorrect role counts")
    all_ids = roles["train"] + roles["val"] + roles["test"] + unused
    require(len(set(all_ids)) == 10000 and set(all_ids) == set(range(10000)), "invalid partition")
    require(record["artifact_schema"] == "radar_fields_padded_roi_v2"
            and record["diagnostic"] == "padded_roi_existing_objective_measurement", "wrong diagnostic")
    require(record["selected_role"] == role and record["selected_role_index"] == 0
            and record["selected_view_id"] == roles[role][0], "wrong selected response")
    require(record["pair_count"] == 256, "not all channel pairs")
    data = record["dataset"]
    require(data["response_shape"] == [10000, 16, 16, 1, 600]
            and data["response_dtype"] == "complex64", "wrong response header")
    require(data["metadata"]["target_type"].lower() == "b787"
            and data["metadata"]["experiment"].lower() == "sphere10k", "wrong scene")
    require(data["response_payload_materialized"] is False
            and data["response_access_restricted"] is True
            and data["authorized_response_view_count"] == 4200, "invalid lazy capability")
    require(record["split"]["role_ids"] == roles
            and record["split"]["test_payload_materialized"] is False
            and record["test_response_payload_materialized"] is False, "invalid role provenance")
    sealed = record["sealed_protocol"]
    require(sealed["unused_role_ids"] == unused and sealed["complete_partition"] is True
            and sealed["test_sealed"] is True and sealed["unused_sealed"] is True
            and sealed["test_response_materialized"] is False
            and sealed["unused_response_materialized"] is False, "sealed role exposed")
    checkpoint = record["checkpoint"]
    require(checkpoint["initialization"] == "random_model_parameters"
            and checkpoint["restored"] is False
            and all(checkpoint[k] is None for k in ("checkpoint_path", "checkpoint_step", "checkpoint_epoch")),
            "unexpected trained checkpoint")
    grid = record["support_grid"]
    require(grid["readout_granularity"] == 48 and grid["readout_point_count"] == 48**3
            and grid["support_xyz_m"] == [[-0.15, 0.15]] * 3, "wrong support/grid")
    normalization = record["signal"]["normalization"]
    peak = normalization["peak_power"]
    require(math.isfinite(peak) and peak > 0 and normalization["dynamic_range_db"] == 60, "bad normalization")
    require(normalization["power_stats_train_view_ids"] == roles["train"]
            and sorted(normalization["power_stats_normalization_scan_view_ids"]) == sorted(roles["train"])
            and normalization["power_stats_train_view_ids_verified"] is True
            and normalization["power_stats_normalization_provenance_verified"] is True,
            "normalization is not verified all-train-only")
    require(normalization["power_stats_cache_reused"] is (role == "val"), "unexpected cache lifecycle")
    render = record["signal"]["render_intensity"]
    require(render == {"range_law": "released", "offset": 0.05, "scaler": 1.0}, "changed intensity convention")
    names = ("roi_cell_count", "valid_roi_cell_count", "padded_roi_cell_count", "padded_roi_fraction",
             "valid_fft_l1_full_mean", "padded_fft_l1_full_mean", "fft_l1_full_mean",
             "valid_fft_weighted_objective", "padded_fft_weighted_objective",
             "valid_fft_model_gradient_l2", "padded_fft_model_gradient_l2", "loss", "rel_mse", "rmse")
    require(all(math.isfinite(record[k]) and record[k] >= 0 for k in names), "nonfinite/negative measurement")
    total, valid, padded = (record[k] for k in names[:3])
    require(total > 0 and valid > 0 and all(int(v) == v for v in (total, valid, padded)), "empty/nonintegral ROI")
    require(total == valid + padded and math.isclose(record["padded_roi_fraction"], padded / total), "bad ROI partition")
    require(math.isclose(record["fft_l1_full_mean"], record["valid_fft_l1_full_mean"]
                         + record["padded_fft_l1_full_mean"], rel_tol=2e-6, abs_tol=1e-7), "nonadditive FFT loss")
    for part in ("valid", "padded"):
        require(math.isclose(record[f"{part}_fft_weighted_objective"],
                             0.6 * record[f"{part}_fft_l1_full_mean"], rel_tol=2e-6, abs_tol=1e-7),
                "wrong FFT objective weight")
    require(record["valid_fft_model_gradient_l2"] > 0, "degenerate valid-cell gradient")
    require(record["padded_fft_model_gradient_l2"] <= 1e-12, "unexpected padded-cell model gradient")
    return {key: record[key] for key in ("selected_role", "selected_view_id", *names)}


def self_test():
    """Synthetic metadata only: no scientific artifacts, Torch, or NPZ access."""
    roles = {"train": list(range(3200)), "val": list(range(3200, 4200)), "test": list(range(4200, 5200))}
    unused = list(range(5200, 10000))
    manifest = {"split": {"train_indices": roles["train"], "validation_indices": roles["val"],
                          "test_indices": roles["test"], "unused_indices": unused}}
    record = {
        "artifact_schema": "radar_fields_padded_roi_v2", "diagnostic": "padded_roi_existing_objective_measurement",
        "selected_role": "train", "selected_role_index": 0, "selected_view_id": 0, "pair_count": 256,
        "dataset": {"response_shape": [10000, 16, 16, 1, 600], "response_dtype": "complex64",
                    "metadata": {"target_type": "b787", "experiment": "sphere10k"},
                    "response_payload_materialized": False, "response_access_restricted": True,
                    "authorized_response_view_count": 4200},
        "split": {"role_ids": roles, "test_payload_materialized": False},
        "test_response_payload_materialized": False,
        "sealed_protocol": {"unused_role_ids": unused, "complete_partition": True, "test_sealed": True,
                            "unused_sealed": True, "test_response_materialized": False,
                            "unused_response_materialized": False},
        "checkpoint": {"initialization": "random_model_parameters", "restored": False,
                       "checkpoint_path": None, "checkpoint_step": None, "checkpoint_epoch": None},
        "support_grid": {"readout_granularity": 48, "readout_point_count": 48**3,
                         "support_xyz_m": [[-0.15, 0.15]] * 3},
        "signal": {"normalization": {"peak_power": 1.0, "dynamic_range_db": 60,
                   "power_stats_train_view_ids": roles["train"],
                   "power_stats_normalization_scan_view_ids": roles["train"],
                   "power_stats_train_view_ids_verified": True,
                   "power_stats_normalization_provenance_verified": True, "power_stats_cache_reused": False},
                   "render_intensity": {"range_law": "released", "offset": 0.05, "scaler": 1.0}},
        "roi_cell_count": 10., "valid_roi_cell_count": 6., "padded_roi_cell_count": 4., "padded_roi_fraction": .4,
        "valid_fft_l1_full_mean": .2, "padded_fft_l1_full_mean": .3, "fft_l1_full_mean": .5,
        "valid_fft_weighted_objective": .12, "padded_fft_weighted_objective": .18,
        "valid_fft_model_gradient_l2": .1, "padded_fft_model_gradient_l2": 0., "loss": 1., "rel_mse": 20., "rmse": 1.}
    check(record, manifest, "train")
    val = copy.deepcopy(record)
    val.update(selected_role="val", selected_view_id=3200)
    val["signal"]["normalization"]["power_stats_cache_reused"] = True
    check(val, manifest, "val")
    cases = [(('pair_count',), 8), (('selected_view_id',), 4200), (('dataset', 'response_shape'), [10000, 8, 8, 1, 60]),
             (('dataset', 'response_payload_materialized'), True), (('sealed_protocol', 'unused_response_materialized'), True),
             (('checkpoint', 'restored'), True), (('fft_l1_full_mean',), 10.), (('loss',), float('nan')),
             (('valid_fft_model_gradient_l2',), 0.), (('padded_fft_model_gradient_l2',), 1.),
             (('signal', 'normalization', 'power_stats_normalization_scan_view_ids'), roles['train'][:-1] + [3200]),
             (('signal', 'normalization', 'power_stats_cache_reused'), True)]
    for path, value in cases:
        broken = copy.deepcopy(record)
        target = broken
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        try:
            check(broken, manifest, "train")
        except (ValueError, KeyError, TypeError):
            continue
        raise AssertionError(f"accepted invalid record: {path}")
    print(f"RF ROI artifact checker self-test passed: {2 + len(cases)} cases")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--manifest")
    parser.add_argument("--train")
    parser.add_argument("--val")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    require(all((args.manifest, args.train, args.val)), "manifest/train/val paths required")
    manifest, train, val = (json.loads(Path(path).read_text(encoding="utf-8"))
                            for path in (args.manifest, args.train, args.val))
    summaries = [check(train, manifest, "train"), check(val, manifest, "val")]
    require(train["signal"]["normalization"]["peak_power"] == val["signal"]["normalization"]["peak_power"],
            "train/validation normalization differs")
    print(json.dumps({"scope": "initial-model support-ROI diagnostic, not NVS or improvement", "measurements": summaries}, sort_keys=True))
    print("RF_B7873200_PADDED_ROI_ARTIFACTS_PASS")


if __name__ == "__main__":
    main()
