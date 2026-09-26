#!/usr/bin/env python
"""Run the native 150x150x120 fixed-grid RIFT-SAS diagnostic."""

from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_sas
from rift.rift_sas import RIFTSASRectangularGrid
from scripts.diagnose_sas_refinement import FIXED_SOURCE_IDS, _saved_recipe_argv


STUDY_NAME = "airsas_armadillo5k_fullgrid150x150x120_smoke_v1_20260914"
REFERENCE_STEP = 700
END_STEP = 900
GRID_SHAPE = (150, 150, 120)
GRID_SITE_COUNT = math.prod(GRID_SHAPE)
SH_DEGREE = 3
EXPECTED_REFERENCE = {
    "model": "adaptive_rift_sas",
    "calibration_mode": "log_polar",
    "seed": 42,
    "eval_every": 100,
    "eval_pings": 8,
    "eval_bins": 0,
    "max_pings": 0,
    "require_explicit_splits": True,
    "sh_degree": SH_DEGREE,
    "refine_every": 1000,
    "initial_granularity": 16,
    "granularity": 64,
    "adaptive_capacity": 65536,
    "max_active": 65536,
    "num_rays": 4900,
    "max_bins": 110,
    "opacity_scale": 500.0,
    "opacity_normalize": False,
    "normal_step": 0.0032,
    "signal_scale": 10.0,
    "lambertian_ratio": 0.0,
    "beamwidth_deg": 30.0,
    "sh_direction": "rx_to_point",
    "grad_clip": 1.0,
    "coefficient_lr": 0.001,
    "lr": 0.001,
}


def _find_flag(argv: Sequence[str], flag: str) -> int:
    positions = [index for index, value in enumerate(argv) if value == flag]
    if len(positions) != 1:
        raise ValueError(f"recipe must contain exactly one {flag}, found {len(positions)}")
    return positions[0]


def _replace_flag(argv: list[str], flag: str, values: Sequence[str]) -> None:
    index = _find_flag(argv, flag)
    old_end = index + 1
    while old_end < len(argv) and not argv[old_end].startswith("--"):
        old_end += 1
    argv[index + 1 : old_end] = list(values)


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
        effective = False if field == "opacity_normalize" and actual is None else actual
        if effective != expected:
            differences.append(f"args.{field}={actual!r} (expected {expected!r})")
    if int(saved_args["refine_every"]) <= END_STEP:
        differences.append(
            f"args.refine_every={saved_args['refine_every']!r} (must exceed {END_STEP})"
        )
    if differences:
        raise ValueError("reference recipe mismatch: " + "; ".join(differences))
    return saved_args


def _build_recipe(
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
    resume_index = _find_flag(argv, "--resume")
    del argv[resume_index : resume_index + 2]
    if "--profile" in argv:
        raise ValueError("full-grid diagnostic must not use --profile")
    _replace_flag(argv, "--model", ["rift_sas"])
    _replace_flag(argv, "--lr", [str(float(saved_args["coefficient_lr"]))])
    _replace_flag(argv, "--steps", [str(END_STEP)])
    _replace_flag(argv, "--eval-every", ["100"])
    _replace_flag(argv, "--eval-pings", ["8"])
    _replace_flag(argv, "--eval-bins", ["0"])
    _replace_flag(argv, "--checkpoint-every", ["100"])
    _replace_flag(argv, "--log-every", ["10"])
    _replace_flag(argv, "--calibration-mode", ["log_polar"])
    _replace_flag(argv, "--query-chunk", ["65536"])
    if "--grid-shape" in argv:
        index = _find_flag(argv, "--grid-shape")
        del argv[index : index + 4]
    argv.extend(["--grid-shape", *(str(value) for value in GRID_SHAPE)])
    if "--ray-chunk" in argv:
        _replace_flag(argv, "--ray-chunk", ["128"])
    else:
        argv.extend(["--ray-chunk", "128"])
    if "--resume" in argv:
        raise ValueError("full-grid diagnostic recipe still contains --resume")
    return argv


def _history_rows(history: Any) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(history, list):
        raise ValueError("training history is missing")
    metric_key = None
    rows = []
    for row in history:
        if not isinstance(row, Mapping) or "step" not in row:
            continue
        candidates = [
            key for key in row
            if str(key).startswith("val_") and str(key).endswith("rel_mse")
        ]
        if len(candidates) != 1:
            raise ValueError(f"could not identify validation RelMSE in history row: {sorted(row)}")
        candidate = str(candidates[0])
        if metric_key is None:
            metric_key = candidate
        elif metric_key != candidate:
            raise ValueError("validation history metric field changed")
        if math.isfinite(float(row[candidate])):
            rows.append(dict(row))
    if metric_key is None or not rows:
        raise ValueError("training history has no finite validation rows")
    return metric_key, rows


def _close(first: Any, second: Any) -> bool:
    return math.isclose(float(first), float(second), rel_tol=0.0, abs_tol=1.0e-12)


class FullGridDiagnosticObserver:
    """Observer that records learned-site gradient support without changing state."""

    def __init__(
        self,
        output_dir: str | Path,
        reference_path: str | Path,
        *,
        expected_grid_shape: tuple[int, int, int] = GRID_SHAPE,
        expected_source_ids: np.ndarray = FIXED_SOURCE_IDS,
        expected_train_count: int = 34560,
        expected_bin_count: int = 326,
        expected_end_step: int = END_STEP,
        expected_dataset_identity: str = "airsas_armadillo_5khz",
        expected_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] = (
            (-0.125, -0.125, 0.0), (0.125, 0.125, 0.2)
        ),
    ) -> None:
        self.output_dir = Path(output_dir)
        self.reference_path = str(reference_path)
        self.expected_grid_shape = tuple(expected_grid_shape)
        self.expected_source_ids = np.asarray(expected_source_ids, dtype=np.int64)
        self.expected_train_count = int(expected_train_count)
        self.expected_bin_count = int(expected_bin_count)
        self.expected_end_step = int(expected_end_step)
        self.expected_dataset_identity = expected_dataset_identity
        self.expected_bounds = expected_bounds
        self.args = None
        self.cache = None
        self.field = None
        self.gradient_support = None
        self.fixed_val_source_ids = np.empty(0, dtype=np.int64)
        self.fixed_bin_ids = np.empty(0, dtype=np.int64)
        self.train_count = 0
        self.started = time.perf_counter()
        self._finished = False

    def on_restore(self, **context: Any) -> None:
        if context.get("state") is not None:
            raise ValueError("full-grid diagnostic requires fresh initialization")
        if int(context["start"]) != 0:
            raise ValueError(f"full-grid diagnostic must start at step 0, got {context['start']}")
        self.args = context["args"]
        self.cache = context["cache"]
        model = context["model"]
        self.field = getattr(model, "coefficient_field", None)
        if not isinstance(self.field, RIFTSASRectangularGrid):
            raise TypeError("full-grid diagnostic did not build RIFTSASRectangularGrid")
        if tuple(self.field.grid_shape) != self.expected_grid_shape:
            raise ValueError(f"learned grid shape changed: {self.field.grid_shape!r}")
        expected_coeff_shape = (*self.expected_grid_shape, (SH_DEGREE + 1) ** 2)
        if tuple(self.field.w_re.shape) != expected_coeff_shape or tuple(self.field.w_im.shape) != expected_coeff_shape:
            raise ValueError("full-grid coefficient tensors do not contain all degree-3 sites")
        if self.field.w_re.shape[-1] != 16:
            raise ValueError("full-grid smoke requires all 16 degree-3 SH coefficients")
        manifest_shape = tuple(int(value) for value in self.cache.manifest["geometry_grid_shape"])
        if self.cache.manifest.get("dataset_identity") != self.expected_dataset_identity:
            raise ValueError("cache dataset identity is not the Armadillo5k contract")
        if manifest_shape != self.expected_grid_shape:
            raise ValueError(f"cache geometry shape is {manifest_shape}, expected {self.expected_grid_shape}")
        corners = np.asarray(self.cache.corners, dtype=np.float32)
        if not np.allclose(corners.min(axis=0), self.expected_bounds[0]) or not np.allclose(
            corners.max(axis=0), self.expected_bounds[1]
        ):
            raise ValueError("cache scene bounds do not match the native Armadillo box")
        self.train_count = int(np.asarray(self.cache.train_indices).size)
        if self.train_count != self.expected_train_count:
            raise ValueError(f"TRAIN count changed: got {self.train_count}, expected {self.expected_train_count}")
        val_indices = train_sas.select_eval_indices(self.cache.validation_indices, self.args.eval_pings)
        self.fixed_val_source_ids = np.asarray(self.cache.source_ids[val_indices], dtype=np.int64)
        if not np.array_equal(self.fixed_val_source_ids, self.expected_source_ids):
            raise ValueError(f"fixed validation source IDs changed: {self.fixed_val_source_ids.tolist()}")
        self.fixed_bin_ids = train_sas.select_eval_bins(self.cache, self.args)
        if not np.array_equal(self.fixed_bin_ids, np.arange(self.expected_bin_count, dtype=np.int64)):
            raise ValueError(f"full-grid smoke must record all {self.expected_bin_count} bins")
        self.gradient_support = torch.zeros(self.expected_grid_shape, dtype=torch.bool)

    def on_before_optimizer(self, **_context: Any) -> None:
        if self.field is None or self.gradient_support is None:
            raise RuntimeError("full-grid observer did not restore a field")
        real_grad = self.field.w_re.grad
        imag_grad = self.field.w_im.grad
        if real_grad is None or imag_grad is None:
            raise RuntimeError("full-grid coefficient gradients are missing")
        support = (real_grad.detach() != 0).any(dim=-1) | (imag_grad.detach() != 0).any(dim=-1)
        self.gradient_support |= support.cpu()

    def on_finish(self, **context: Any) -> None:
        if int(context["step"]) != self.expected_end_step:
            raise ValueError(f"full-grid diagnostic must finish at step {self.expected_end_step}")
        if self.gradient_support is None or self.field is None or self.args is None:
            raise RuntimeError("full-grid observer did not receive restore state")
        metric_key, history = _history_rows(context["history"])
        best_row = min(history, key=lambda row: float(row[metric_key]))
        final_rows = [row for row in history if int(row["step"]) == self.expected_end_step]
        if len(final_rows) != 1:
            raise ValueError(f"expected one step-{END_STEP} validation row")
        best_path = self.output_dir / "checkpoint_best.pt"
        if not best_path.exists():
            raise RuntimeError("finite validation history has no checkpoint_best.pt")
        best_state = torch.load(best_path, map_location="cpu", weights_only=False)
        try:
            if int(best_state.get("step", -1)) != int(best_row["step"]):
                raise RuntimeError("checkpoint_best.pt step disagrees with validation history")
            if not _close(best_state.get("best_val_rel_mse", float("nan")), best_row[metric_key]):
                raise RuntimeError("checkpoint_best.pt metric disagrees with validation history")
        finally:
            del best_state
        support_count = int(self.gradient_support.sum().item())
        summary = {
            "status": "success",
            "complete": True,
            "study": STUDY_NAME,
            "method": "rift_sas_dense_grid",
            "reference_path_used_for_recipe_only": self.reference_path,
            "reference_weights_loaded": False,
            "fresh_initialization": True,
            "effective_recipe": vars(self.args),
            "grid_shape": list(self.expected_grid_shape),
            "grid_site_count": math.prod(self.expected_grid_shape),
            "sh_degree": SH_DEGREE,
            "fixed_validation_source_ids": self.fixed_val_source_ids.tolist(),
            "fixed_bin_ids": self.fixed_bin_ids.tolist(),
            "train_count": self.train_count,
            "updates": self.expected_end_step,
            "history_metric_key": metric_key,
            "validation_history": history,
            "best_validation_rel_mse": float(best_row[metric_key]),
            "best_validation_step": int(best_row["step"]),
            "final_step_validation_rel_mse": float(final_rows[0][metric_key]),
            "best_checkpoint": best_path.name,
            "gradient_support_count": support_count,
            "gradient_support_fraction": support_count / float(math.prod(self.expected_grid_shape)),
            "peak_cuda_memory_bytes": int(
                torch.cuda.max_memory_allocated(context["device"])
            ) if context["device"].type == "cuda" else 0,
            "elapsed_seconds": time.perf_counter() - self.started,
            "model_test_evaluated": False,
            "refinement_events": 0,
        }
        payload = context["checkpoint_writer"](
            context["model"], context["calibration"], context["optimizer"],
            context["step"], context["best_val"], context["rng"], context["history"],
            context["args"], self.cache,
        )
        payload["diagnostic_summary"] = summary
        payload["diagnostic_output_contract"] = {
            "fullgrid_summary": "fullgrid_summary.json",
            "checkpoint_final_diagnostic": "checkpoint_final_diagnostic.pt",
        }
        context["atomic_save"](payload, self.output_dir / "checkpoint_final_diagnostic.pt")
        train_sas.atomic_json(summary, self.output_dir / "fullgrid_summary.json")
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
        raise FileExistsError(f"full-grid output already exists: {output}")
    state = torch.load(reference, map_location="cpu", weights_only=False)
    try:
        saved_args = _validate_reference_state(state)
        recipe = _build_recipe(
            saved_args,
            cache=Path(args.cache),
            reference=reference,
            output=output,
            device=args.device,
        )
    finally:
        del state
        gc.collect()
    observer = FullGridDiagnosticObserver(output, reference)
    train_sas.main(recipe, diagnostic_observer=observer)
    if not observer._finished:
        raise RuntimeError("full-grid observer did not publish its final report")


if __name__ == "__main__":
    main()
