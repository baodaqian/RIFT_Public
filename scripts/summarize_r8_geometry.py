#!/usr/bin/env python
"""Validate and summarize the matched native/4x Round-8 geometry sweep."""

import argparse
import csv
import json
import math
from pathlib import Path


EXPECTED_RUNS = {
    "r8_legacy", "r8_e15_lr3em6", "r8_e15_lr3em5", "r8_e15_lr3em4", "r8_e15_lr3em3",
    "rift_r6_learned_dc", "rift_r7_target20k_shdeg1em9",
}
EXPECTED_THRESHOLDS = {
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40,
    0.50, 0.60, 0.70, 0.80, 0.90, 0.95,
}
METRIC_KEYS = ("cd_surface", "hausdorff_mm", "hd95_mm", "iou_solid", "f1")


def summarize_rows(rows, protocol):
    expected_variants = {"voxel"} if protocol == "native_grid" else {"voxel", "mesh"}
    expected_count = len(EXPECTED_RUNS) * len(EXPECTED_THRESHOLDS) * len(expected_variants)
    if len(rows) != expected_count:
        raise RuntimeError(f"{protocol}: expected {expected_count} rows, found {len(rows)}")

    by_variant_run = {}
    for row in rows:
        run = row["run"]
        variant = row["variant"]
        threshold = float(row["thresh"])
        if (run not in EXPECTED_RUNS or variant not in expected_variants
                or threshold not in EXPECTED_THRESHOLDS):
            raise RuntimeError(f"{protocol}: unexpected row {run}/{variant} @ {threshold}")
        numeric = {key: float(row[key]) for key in METRIC_KEYS}
        if not all(math.isfinite(value) for value in numeric.values()):
            raise RuntimeError(f"{protocol}: non-finite metrics for {run}/{variant} @ {threshold}")
        by_variant_run.setdefault(variant, {}).setdefault(run, []).append({
            "threshold": threshold,
            **numeric,
            "n_points": int(row["n_points"]),
        })
    if set(by_variant_run) != expected_variants:
        raise RuntimeError(f"{protocol}: variant set mismatch")

    protocol_summary = {}
    for variant, by_run in by_variant_run.items():
        if set(by_run) != EXPECTED_RUNS:
            raise RuntimeError(f"{protocol}/{variant}: run set mismatch")
        min_points = 1280 if protocol == "trilinear_4x" and variant == "voxel" else 20
        variant_summary = {}
        for run, run_rows in by_run.items():
            fixed = next(row for row in run_rows if row["threshold"] == 0.20)
            valid = [row for row in run_rows if row["n_points"] >= min_points]
            if not valid:
                raise RuntimeError(f"{protocol}/{variant}/{run}: no nondegenerate threshold")
            variant_summary[run] = {
                "minimum_oracle_points": min_points,
                "fixed_threshold_0p20": fixed,
                "oracle": {
                    "chamfer": min(valid, key=lambda row: row["cd_surface"]),
                    "hausdorff": min(valid, key=lambda row: row["hausdorff_mm"]),
                    "hd95": min(valid, key=lambda row: row["hd95_mm"]),
                    "iou": max(valid, key=lambda row: row["iou_solid"]),
                    "f1": max(valid, key=lambda row: row["f1"]),
                },
            }
        protocol_summary[variant] = variant_summary
    return protocol_summary


def summarize_csv(path, protocol):
    with path.open(newline="") as handle:
        return summarize_rows(list(csv.DictReader(handle)), protocol)


def validate_native_rows(current, reference, reference_name):
    """Prove that the corrected evaluator leaves historical native rows unchanged."""
    controls = {"rift_r6_learned_dc", "rift_r7_target20k_shdeg1em9"}
    current_by_key = {
        (row["run"], row["variant"], float(row["thresh"])): row
        for row in current if row["run"] in controls
    }
    reference_by_key = {
        (row["run"], row["variant"], float(row["thresh"])): row
        for row in reference if row["run"] in controls
    }
    if current_by_key.keys() != reference_by_key.keys():
        raise RuntimeError("native R6/R7 reproduction row keys differ from the preserved CSV")
    numeric_fields = (
        "cd_surface", "cd_volume", "l2_mm", "hausdorff_mm", "hd95_mm",
        "iou_solid", "iou_shell", "precision", "recall", "f1",
    )
    for key, current_row in current_by_key.items():
        reference_row = reference_by_key[key]
        if (int(current_row["epoch"]) != int(reference_row["epoch"])
                or int(current_row["n_points"]) != int(reference_row["n_points"])):
            raise RuntimeError(f"native R6/R7 reproduction metadata differs for {key}")
        for field in numeric_fields:
            if not math.isclose(
                    float(current_row[field]), float(reference_row[field]),
                    rel_tol=1e-11, abs_tol=1e-13):
                raise RuntimeError(f"native R6/R7 reproduction differs for {key}: {field}")
    return {
        "status": "pass",
        "reference": str(reference_name),
        "controls": sorted(controls),
        "rows_compared": len(current_by_key),
    }


def validate_native_reproduction(native_csv, reference_csv):
    with native_csv.open(newline="") as handle:
        current = list(csv.DictReader(handle))
    with reference_csv.open(newline="") as handle:
        reference = list(csv.DictReader(handle))
    return validate_native_rows(current, reference, reference_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--reference-csv", required=True)
    args = parser.parse_args()
    root = Path(args.out_root)
    native_csv = root / "native_grid_rows.csv"
    results = {
        "native_grid": summarize_csv(native_csv, "native_grid"),
        "trilinear_4x": summarize_csv(root / "trilinear_4x_rows.csv", "trilinear_4x"),
    }
    payload = {
        "status": "complete",
        "checkpoint_selector": (
            "Round-8 checkpoint_best (all epoch 150); historical controls reproduce the "
            "preserved R6 checkpoint_best epoch 149 and R7 checkpoint_final epoch 150"
        ),
        "trilinear_contract": (
            "established RIFT visualization readout: interpolate rotation-invariant SH energy "
            "4x per axis with torch mode=trilinear, align_corners=True; physical coordinates "
            "preserve the native g48 voxel-centre endpoints"
        ),
        "metric_protocol": (
            "B787 Reed-style threshold sweep; fixed t=0.20 plus per-metric oracle; "
            "tau=6.25 mm; IoU unit=5 mm; seed=0"
        ),
        "interpretation_boundary": (
            "4x densification adds no learned information. Compare arms only within the same "
            "readout. Native voxel results remain the reportable control; 4x voxel and "
            "marching-cubes results are trilinear-readout diagnostics."
        ),
        "native_reproduction_gate": validate_native_reproduction(
            native_csv, Path(args.reference_csv)
        ),
        "results": results,
    }
    temporary = root / "geometry_summary.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(root / "geometry_summary.json")
    print("validated and wrote", root / "geometry_summary.json")


if __name__ == "__main__":
    main()
