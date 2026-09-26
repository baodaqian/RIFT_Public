#!/usr/bin/env python3
"""Torch-free source-contract preflight for the GeRaF B787 16/4 small fit.

This checks only authored-source structure on the local workspace.  The
Torch fixture and real native-MF preparation are intentionally deferred to
one allocated PACE GPU cell.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Gates:
    def __init__(self) -> None:
        self.count = 0

    def check(self, condition: bool, detail: str) -> None:
        if not condition:
            raise AssertionError(detail)
        self.count += 1
        print(f"PASS {self.count:02d}: {detail}", flush=True)


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _tree(relative: str) -> ast.AST:
    path = ROOT / relative
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _has_function(tree: ast.AST, name: str) -> bool:
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        for node in ast.walk(tree)
    )


def _has_class(tree: ast.AST, name: str) -> bool:
    return any(isinstance(node, ast.ClassDef) and node.name == name for node in ast.walk(tree))


def main() -> None:
    gates = Gates()
    production = _text("train_geraf.py")
    production_tree = _tree("train_geraf.py")
    subset = _text("rift/geraf_b78716_smallfit.py")
    subset_tree = _tree("rift/geraf_b78716_smallfit.py")
    driver = _text("train_geraf_smoke.py")
    driver_tree = _tree("train_geraf_smoke.py")
    launcher = _text("slurm/validate_geraf_b78716_smallfit_v1.sbatch")
    protocol = _text("rift/geraf_b7873200_protocol.py")

    gates.check(
        "B787_3200_CANONICAL_NPZ_PATH" in protocol
        and "/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/" in protocol
        and "b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz" in protocol,
        "the canonical B787 archive remains bound to the corrected /storage/home location",
    )
    gates.check(
        "validate_b7873200_target_cache(root, identity)" in production
        and "B787_3200_NUM_TRAIN" in production
        and "B787_3200_NUM_VALIDATION" in production,
        "the production 3200/1000 cache verifier remains full-only",
    )
    gates.check(
        "engineering_subset16x4_native_mf_cache_v1" in subset
        and "NUM_TRAIN = 16" in subset
        and "NUM_VALIDATION = 4" in subset
        and "TARGET_GRID_SHAPE = (8, 8, 8)" in subset,
        "the opt-in cache has a distinct 16/4 schema and frozen 8-cubed target lattice",
    )
    gates.check(
        _has_class(subset_tree, "BoundedB78716PreparationSource")
        and "normalized not in self.allowed_response_view_indices" in subset
        and "load_b7873200_metadata_source" in subset,
        "preparation has an explicit selected-row guard while fitting reopens metadata only",
    )
    gates.check(
        "B787_3200_TARGET_CACHE_PROTOCOL_FILENAME" in subset
        and "write_b7873200_target_cache_protocol" not in subset
        and "_validate_target_leaves" in subset,
        "the new cache publishes its own completion protocol only after exact target-leaf validation",
    )
    gates.check(
        _has_function(production_tree, "train_one_view_update")
        and _has_function(production_tree, "evaluate_indices")
        and "last_train = train_one_view_update(" in production
        and "return evaluate_indices(" in production,
        "production and bounded code share extracted update/evaluation seams instead of a copied optimizer loop",
    )
    gates.check(
        "MAX_UPDATES = 32" in subset
        and "SCHEDULER_HORIZON_UPDATES = 50_000" in subset
        and "horizon_steps=SCHEDULER_HORIZON_UPDATES" in driver
        and "while step < MAX_UPDATES" in driver,
        "the 32-update engineering stop budget is separate from the 50,000-update cosine horizon",
    )
    gates.check(
        "start_paths = _fresh_paths(checkpoint_dir, args.resume)" in driver
        and "if fresh_fit:" in driver
        and "prepare_complete_subset_cache(" in driver
        and "load_complete_subset_cache(" in driver
        and "MILESTONE_UPDATES = (0, 16, 32)" in subset
        and "validation_dynamic_mask_changed" in driver,
        "fresh fitting prepares before use while resume/recovery has a metadata-only cache path and 0/16/32 held-out milestones",
    )
    gates.check(
        "checkpoint_initial.pth.tar" in driver
        and "checkpoint_latest.pth.tar" in driver
        and "checkpoint_final.pth.tar" in driver
        and "only a clean-interruption bounded GeRaF checkpoint may resume" in driver
        and "last_train step disagrees with completed updates" in driver
        and "last_train names a non-selected training view" in driver,
        "bounded recovery is explicit and records initial/latest/final checkpoint evidence",
    )
    gates.check(
        _has_function(driver_tree, "_parameter_change")
        and "any(count != 2 for count in visit_counts.values())" in driver
        and "peak_torch_allocated_bytes" in driver
        and "peak_torch_reserved_bytes" in driver,
        "the driver gates nonzero parameter change, two visits per train view, and memory telemetry",
    )
    gates.check(
        _has_class(driver_tree, "_CleanStop")
        and _has_class(driver_tree, "_PreFitStop")
        and "CLEAN_STOP_EXIT_CODE = 143" in driver
        and "PRE_FIT_STOP_EXIT_CODE = 75" in driver
        and 'print("GERAF_B78716_SMALLFIT_LIFECYCLE_COMPLETE", flush=True)' in driver
        and "terminal_report" in driver
        and "_recover_terminal_report" in driver,
        "terminal publication embeds recoverable report evidence and distinguishes resumable fit stops from pre-fit stops",
    )
    gates.check(
        "RESOURCE_ENVELOPE_SCHEMA" in driver
        and _has_function(driver_tree, "_merge_resource_envelope")
        and "prior_resource_envelope = _validate_resource_envelope(" in driver
        and "resource_envelope=resource_envelope" in driver
        and "resource_envelope=resources" in driver
        and "embedded terminal report resource evidence differs" in driver,
        "clean continuation and terminal recovery retain a validated cumulative resource envelope rather than only final-attempt telemetry",
    )
    gates.check(
        _has_function(driver_tree, "_pre_fit_stop_outcome")
        and "if fresh_fit and _STOP_REQUESTED:" in driver
        and "_pre_fit_stop_outcome(fresh_fit=fresh_fit)" in driver,
        "a pre-fit signal can stop only fresh preparation, while metadata-only resume/recovery remains able to finish",
    )
    gates.check(
        "grep -Fxq 'GERAF_B78716_SMALLFIT_LIFECYCLE_COMPLETE'" in launcher
        and "driver_status == 143" in launcher
        and "driver_status == 75" in launcher
        and "GERAF_B78716_SMALLFIT_TERMINAL_REPORT_RECOVERY_PREFLIGHT" in launcher
        and "attempt_log_dir=\"$launcher_log_root/${SLURM_JOB_ID}\"" in launcher
        and "time_log=\"$attempt_log_dir/geraf_time_v.txt\"" in launcher,
        "the one-cell launcher checks the exact success marker, handles clean-stop/recovery states, and retains driver evidence durably",
    )
    print(f"GERAF_B78716_SMALLFIT_SOURCE_VALIDATION_PASS gates={gates.count}", flush=True)


if __name__ == "__main__":
    main()
