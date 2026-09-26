#!/usr/bin/env python
"""Run the bounded initial-support AirSAS diagnostic through ``train_sas``."""

from __future__ import annotations

import argparse
import gc
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_sas
from scripts.diagnose_sas_refinement import FIXED_SOURCE_IDS, _saved_recipe_argv


STUDY_NAME = "airsas_armadillo5k_initial32_prefit_smoke_v1"
REFERENCE_STEP = 700
END_STEP = 900
INITIAL_GRANULARITY = 32
EXPECTED_REFERENCE = {
    "model": "adaptive_rift_sas",
    "calibration_mode": "log_polar",
    "initial_granularity": 16,
    "granularity": 64,
    "adaptive_capacity": 65536,
    "max_active": 65536,
    "sh_degree": 3,
    "seed": 42,
    "eval_every": 100,
    "eval_pings": 8,
    "eval_bins": 0,
    "max_pings": 0,
    "require_explicit_splits": True,
    "refine_every": 1000,
}


def _flag_index(argv: Sequence[str], flag: str) -> int:
    positions = [index for index, value in enumerate(argv) if value == flag]
    if len(positions) != 1:
        raise ValueError(f"recipe must contain exactly one {flag}, found {len(positions)}")
    return positions[0]


def _build_fresh_recipe_argv(
    saved_args: Mapping[str, Any],
    *,
    cache: Path,
    reference: Path,
    output: Path,
    device: str,
) -> list[str]:
    argv = _saved_recipe_argv(
        saved_args,
        cache=cache,
        checkpoint=reference,
        output=output,
        device=device,
        end_step=END_STEP,
    )
    resume_index = _flag_index(argv, "--resume")
    if resume_index + 1 >= len(argv):
        raise ValueError("recipe has an incomplete --resume pair")
    del argv[resume_index : resume_index + 2]
    if "--profile" in argv:
        raise ValueError("initial-support recipe must not use a profile")
    initial_index = _flag_index(argv, "--initial-granularity")
    if initial_index + 1 >= len(argv):
        raise ValueError("recipe has an incomplete --initial-granularity pair")
    argv[initial_index + 1] = str(INITIAL_GRANULARITY)
    steps_index = _flag_index(argv, "--steps")
    if argv[steps_index + 1] != str(END_STEP):
        raise ValueError(f"recipe terminal step must be {END_STEP}, got {argv[steps_index + 1]!r}")
    if "--resume" in argv:
        raise ValueError("fresh initial-support recipe still contains --resume")
    return argv


def _validate_reference_state(state: Mapping[str, Any]) -> Mapping[str, Any]:
    required_state = ("step", "model_kind", "calibration_mode", "args")
    missing_state = [key for key in required_state if key not in state]
    if missing_state:
        raise ValueError(f"reference checkpoint is missing required metadata: {missing_state}")
    if int(state["step"]) != REFERENCE_STEP:
        raise ValueError(
            f"reference checkpoint must be at step {REFERENCE_STEP}, got {state['step']!r}"
        )
    saved_args = state["args"]
    if not isinstance(saved_args, Mapping):
        raise ValueError("reference checkpoint args metadata is not a mapping")
    required_args = tuple(EXPECTED_REFERENCE)
    missing_args = [key for key in required_args if key not in saved_args]
    if missing_args:
        raise ValueError(f"reference recipe is missing required fields: {missing_args}")
    differences = []
    if state["model_kind"] != EXPECTED_REFERENCE["model"]:
        differences.append(
            f"state.model_kind={state['model_kind']!r} (expected {EXPECTED_REFERENCE['model']!r})"
        )
    if state["calibration_mode"] != EXPECTED_REFERENCE["calibration_mode"]:
        differences.append(
            f"state.calibration_mode={state['calibration_mode']!r} "
            f"(expected {EXPECTED_REFERENCE['calibration_mode']!r})"
        )
    for field, expected in EXPECTED_REFERENCE.items():
        actual = saved_args[field]
        if actual != expected:
            differences.append(f"args.{field}={actual!r} (expected {expected!r})")
    if int(saved_args["refine_every"]) <= END_STEP:
        differences.append(
            f"args.refine_every={saved_args['refine_every']!r} (must exceed {END_STEP})"
        )
    if differences:
        raise ValueError("reference recipe mismatch: " + "; ".join(differences))
    return saved_args


def _validation_history(history: Any) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(history, list):
        raise ValueError("training history is missing from the diagnostic finish context")
    metric_key: str | None = None
    rows: list[dict[str, Any]] = []
    for row in history:
        if not isinstance(row, Mapping) or "step" not in row:
            continue
        candidates = [
            key for key in row
            if str(key).startswith("val_") and str(key).endswith("rel_mse")
        ]
        if len(candidates) != 1:
            raise ValueError(f"could not identify one validation RelMSE field in history row: {sorted(row)}")
        candidate = str(candidates[0])
        if metric_key is None:
            metric_key = candidate
        elif metric_key != candidate:
            raise ValueError(f"validation history changes metric field from {metric_key!r} to {candidate!r}")
        metric = float(row[candidate])
        if math.isfinite(metric):
            rows.append(dict(row))
    if metric_key is None or not rows:
        raise ValueError("training history contains no finite validation RelMSE rows")
    return metric_key, rows


def _close(first: float, second: float) -> bool:
    return math.isclose(float(first), float(second), rel_tol=0.0, abs_tol=1.0e-12)


class InitialSupportDiagnosticObserver:
    """Read-only observer for the fresh 32-cubed pre-refinement fit."""

    def __init__(self, output_dir: str | Path, reference_path: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.reference_path = str(reference_path)
        self.scene = None
        self.cache = None
        self.args = None
        self.train_source_ids = np.empty(0, dtype=np.int64)
        self.fixed_val_source_ids = np.empty(0, dtype=np.int64)
        self.fixed_bin_ids = np.empty(0, dtype=np.int64)
        self.initial_active_count = None
        self.final_active_count = None
        self.allocated_capacity = None
        self.refinement_events = 0
        self._finished = False

    @staticmethod
    def _scene_state(model) -> tuple[Any, int, int, bool]:
        scene = train_sas._adaptive_scene(model)
        if scene is None:
            raise TypeError("initial-support diagnostic requires adaptive_rift_sas")
        active = int(scene.active_mask.sum().item())
        capacity = int(scene.capacity)
        active_orders = scene.order[scene.active_mask]
        degree_zero = bool(torch.all(active_orders == 0).item())
        return scene, active, capacity, degree_zero

    def on_restore(self, **context: Any) -> None:
        if context.get("state") is not None:
            raise ValueError("initial-support diagnostic requires a fresh run with no restored state")
        if int(context["start"]) != 0:
            raise ValueError(f"initial-support diagnostic must start at step 0, got {context['start']}")
        self.args = context["args"]
        self.cache = context["cache"]
        self.scene, active, capacity, degree_zero = self._scene_state(context["model"])
        if active != INITIAL_GRANULARITY ** 3:
            raise ValueError(f"initial active count must be {INITIAL_GRANULARITY ** 3}, got {active}")
        if capacity != EXPECTED_REFERENCE["adaptive_capacity"]:
            raise ValueError(f"adaptive capacity must be 65536, got {capacity}")
        if not degree_zero:
            raise ValueError("initial active SH orders must all be degree zero")
        self.initial_active_count = active
        self.allocated_capacity = capacity
        train_indices = train_sas.select_eval_indices(self.cache.train_indices, 0)
        val_indices = train_sas.select_eval_indices(
            self.cache.validation_indices, FIXED_SOURCE_IDS.size
        )
        self.train_source_ids = np.asarray(self.cache.source_ids[train_indices], dtype=np.int64)
        self.fixed_val_source_ids = np.asarray(self.cache.source_ids[val_indices], dtype=np.int64)
        if not np.array_equal(self.fixed_val_source_ids, FIXED_SOURCE_IDS):
            raise ValueError(
                f"fixed validation source IDs changed: got {self.fixed_val_source_ids.tolist()}"
            )
        self.fixed_bin_ids = train_sas.select_eval_bins(self.cache, self.args)
        expected_bins = np.arange(326, dtype=np.int64)
        if not np.array_equal(self.fixed_bin_ids, expected_bins):
            raise ValueError("initial-support diagnostic must record all bin IDs 0 through 325")

    def on_before_refinement(self, **_context: Any) -> None:
        self.refinement_events += 1
        raise RuntimeError("initial-support diagnostic reached a refinement event before step 900")

    def on_finish(self, **context: Any) -> None:
        step = int(context["step"])
        if step != END_STEP:
            raise ValueError(f"initial-support diagnostic must finish at step {END_STEP}, got {step}")
        if self.scene is None or self.cache is None or self.args is None:
            raise RuntimeError("initial-support diagnostic did not receive restore metadata")
        _scene, active, capacity, degree_zero = self._scene_state(context["model"])
        self.final_active_count = active
        if active != self.initial_active_count or capacity != self.allocated_capacity:
            raise ValueError("active support or capacity changed without a refinement event")
        if not degree_zero or self.refinement_events:
            raise ValueError("initial-support diagnostic changed SH support or refined")

        metric_key, history = _validation_history(context["history"])
        best_row = min(history, key=lambda row: float(row[metric_key]))
        final_rows = [row for row in history if int(row["step"]) == END_STEP]
        if len(final_rows) != 1:
            raise ValueError(f"expected one step-{END_STEP} validation row, got {len(final_rows)}")
        final_row = final_rows[0]
        best_checkpoint = self.output_dir / "checkpoint_best.pt"
        if not best_checkpoint.exists():
            raise RuntimeError("finite validation history has no checkpoint_best.pt")
        best_state = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
        try:
            if int(best_state.get("step", -1)) != int(best_row["step"]):
                raise RuntimeError("checkpoint_best.pt step disagrees with validation history")
            if not _close(best_state.get("best_val_rel_mse", float("nan")), best_row[metric_key]):
                raise RuntimeError("checkpoint_best.pt metric disagrees with validation history")
        finally:
            del best_state

        summary = {
            "status": "success",
            "complete": True,
            "study": STUDY_NAME,
            "reference_path_used_for_recipe_only": self.reference_path,
            "reference_weights_loaded": False,
            "model_test_evaluated": False,
            "refinement_events": self.refinement_events,
            "updates": END_STEP,
            "effective_args": vars(self.args),
            "initial_active_count": int(self.initial_active_count),
            "final_active_count": int(self.final_active_count),
            "allocated_capacity": int(self.allocated_capacity),
            "raster_granularity": int(self.args.granularity),
            "train_source_ids": self.train_source_ids.tolist(),
            "train_count": int(self.train_source_ids.size),
            "fixed_validation_source_ids": self.fixed_val_source_ids.tolist(),
            "fixed_bin_ids": self.fixed_bin_ids.tolist(),
            "history_metric_key": metric_key,
            "best_validation_rel_mse": float(best_row[metric_key]),
            "best_validation_step": int(best_row["step"]),
            "final_step_validation_rel_mse": float(final_row[metric_key]),
            "validation_history": history,
            "best_checkpoint": best_checkpoint.name,
        }
        payload = context["checkpoint_writer"](
            context["model"],
            context["calibration"],
            context["optimizer"],
            step,
            context["best_val"],
            context["rng"],
            context["history"],
            context["args"],
            self.cache,
        )
        payload["diagnostic_summary"] = summary
        payload["diagnostic_output_contract"] = {
            "initial_support_summary": "initial_support_summary.json",
            "checkpoint_final_diagnostic": "checkpoint_final_diagnostic.pt",
        }
        context["atomic_save"](payload, self.output_dir / "checkpoint_final_diagnostic.pt")
        train_sas.atomic_json(summary, self.output_dir / "initial_support_summary.json")
        self._finished = True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--reference-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    reference = Path(args.reference_checkpoint)
    output = Path(args.output)
    if not reference.exists():
        raise FileNotFoundError(f"reference checkpoint does not exist: {reference}")
    if output.exists():
        raise FileExistsError(f"initial-support output already exists: {output}")
    state = torch.load(reference, map_location="cpu", weights_only=False)
    try:
        saved_args = _validate_reference_state(state)
        recipe_argv = _build_fresh_recipe_argv(
            saved_args,
            cache=Path(args.cache),
            reference=reference,
            output=output,
            device=args.device,
        )
    finally:
        del state
        gc.collect()
    observer = InitialSupportDiagnosticObserver(output, reference)
    train_sas.main(recipe_argv, diagnostic_observer=observer)
    if not observer._finished:
        raise RuntimeError("initial-support observer did not publish its final report")


if __name__ == "__main__":
    main()
