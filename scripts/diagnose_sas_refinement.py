#!/usr/bin/env python
"""Replay a preserved adaptive AirSAS checkpoint and audit one refinement.

This module deliberately delegates the replay loop to :mod:`train_sas`.  The
observer only records fixed-ping renders, topology transitions, optimizer
updates, and non-additive one-factor interventions around the existing
renderer.  It is a diagnostic, not a second training implementation.
"""

from __future__ import annotations

import argparse
import csv
import copy
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


FIXED_SOURCE_IDS = np.asarray(
    [2520, 6377, 13474, 20571, 24428, 31525, 38622, 42479], dtype=np.int64
)
POST_UPDATE_STEPS = (1001, 1002, 1003, 1005, 1010)
DEFAULT_OUTPUT_DIR = (
    "/storage/project/r-jromberg3-0/dbao31/RIFT/"
    "training_checkpoints/airsas_armadillo5k_adaptive_refinement_replay_v1_20260913"
)
INTERVENTION_LABELS = (
    "freeze_gain_to_post_refinement_baseline",
    "restore_original_row_dc_to_post_refinement_baseline",
    "zero_new_sibling_dc",
    "zero_newly_unlocked_angular_coefficients",
)


def _clone_nested(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _clone_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_nested(item) for item in value)
    return value


def _nested_equal(first: Any, second: Any) -> bool:
    if torch.is_tensor(first) or torch.is_tensor(second):
        return torch.is_tensor(first) and torch.is_tensor(second) and torch.equal(first, second)
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        return isinstance(first, np.ndarray) and isinstance(second, np.ndarray) and np.array_equal(first, second)
    if isinstance(first, dict) or isinstance(second, dict):
        return (
            isinstance(first, dict)
            and isinstance(second, dict)
            and set(first) == set(second)
            and all(_nested_equal(first[key], second[key]) for key in first)
        )
    if isinstance(first, (list, tuple)) or isinstance(second, (list, tuple)):
        return (
            isinstance(first, type(second))
            and len(first) == len(second)
            and all(_nested_equal(a, b) for a, b in zip(first, second))
        )
    return first == second


def _tensor_stats(value: torch.Tensor) -> tuple[float, float]:
    if value.numel() == 0:
        return 0.0, 0.0
    magnitude = value.detach().abs().double().reshape(-1)
    return float(magnitude.square().mean().sqrt().item()), float(magnitude.max().item())


def _safe_list(value: torch.Tensor) -> list[Any]:
    return value.detach().cpu().tolist()


class RefinementDiagnosticObserver:
    """Stateful, read-only observer for the adaptive AirSAS replay."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        expected_fixed_val_rel_mse: float = 1.369373,
        fixed_val_tolerance: float = 1.0e-4,
        expected_historical_counts: Mapping[str, int] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.expected_fixed_val_rel_mse = float(expected_fixed_val_rel_mse)
        self.fixed_val_tolerance = float(fixed_val_tolerance)
        self.expected_historical_counts = {
            "n_split": 205,
            "n_angular": 410,
            "active_after": 5531,
        }
        if expected_historical_counts is not None:
            self.expected_historical_counts.update({
                key: int(value) for key, value in expected_historical_counts.items()
            })
        self.model: torch.nn.Module | None = None
        self.calibration: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.cache = None
        self.args = None
        self.device = None
        self.rng: np.random.Generator | None = None
        self.reference_rng: np.random.Generator | None = None
        self.train_indices = np.empty(0, dtype=np.int64)
        self.validation_indices = np.empty(0, dtype=np.int64)
        self.expected_next_step = 701
        self.scene = None
        self.fixed_source_ids = FIXED_SOURCE_IDS.copy()
        self.fixed_ping_indices: np.ndarray | None = None
        self.fixed_bins = np.empty(0, dtype=np.int64)
        self.original_active_mask: torch.Tensor | None = None
        self.original_order: torch.Tensor | None = None
        self.new_sibling_mask: torch.Tensor | None = None
        self.newly_unlocked_mask: torch.Tensor | None = None
        self.new_sibling_inherited_angular_mask: torch.Tensor | None = None
        self.post_refinement_baseline: dict[str, torch.Tensor] | None = None
        self._before_optimizer: dict[str, Any] | None = None
        self._pending_refinement: dict[str, Any] | None = None
        self.stage_metadata: list[dict[str, Any]] = []
        self.stage_predictions: list[np.ndarray] = []
        self.stage_raw_predictions: list[np.ndarray] = []
        self.stage_targets: list[np.ndarray] = []
        self.intervention_metadata: list[dict[str, Any]] = []
        self.intervention_predictions: list[np.ndarray] = []
        self.intervention_raw_predictions: list[np.ndarray] = []
        self.selection_sequence: list[dict[str, Any]] = []
        self.update_rows: list[dict[str, Any]] = []
        self.refinement_events: list[dict[str, Any]] = []
        self.restore_checks: list[dict[str, Any]] = []
        self.observation_checks: list[dict[str, Any]] = []
        self.replay_divergences: list[str] = []
        self.failure: dict[str, Any] | None = None
        self._observation_label: str | None = None
        self._finished = False

    def on_restore(self, **context: Any) -> None:
        state = context.get("state")
        start = int(context["start"])
        if state is None:
            raise ValueError("refinement replay requires a restored checkpoint")
        if start != 700:
            raise ValueError(f"refinement replay requires checkpoint step 700, got {start}")
        model = context["model"]
        scene = train_sas._adaptive_scene(model)
        if scene is None:
            raise TypeError("refinement diagnostic requires adaptive_rift_sas")
        cache = context["cache"]
        if int(cache.num_bins) != 326:
            raise ValueError(f"refinement diagnostic requires all 326 bins, got {cache.num_bins}")
        source_ids = np.asarray(cache.source_ids, dtype=np.int64)
        positions = {int(source): index for index, source in enumerate(source_ids.tolist())}
        missing = [int(source) for source in self.fixed_source_ids if int(source) not in positions]
        if missing:
            raise ValueError(f"fixed validation source IDs are missing from cache: {missing}")

        self.model = model
        self.calibration = context["calibration"]
        self.optimizer = context["optimizer"]
        self.rng = context["rng"]
        self.cache = cache
        self.args = context["args"]
        self.device = torch.device(context["device"])
        self.scene = scene
        self.train_indices = np.asarray(context["train_indices"], dtype=np.int64).copy()
        self.validation_indices = np.asarray(context["validation_indices"], dtype=np.int64).copy()
        if self.rng is None or not isinstance(self.rng, np.random.Generator):
            raise TypeError("refinement diagnostic requires the trainer NumPy Generator")
        bit_generator = type(self.rng.bit_generator)()
        bit_generator.state = copy.deepcopy(state["rng_state"])
        self.reference_rng = np.random.Generator(bit_generator)
        if not _nested_equal(self.rng.bit_generator.state, self.reference_rng.bit_generator.state):
            raise RuntimeError("restored trainer/reference RNG states disagree")
        validation_probe_indices = train_sas.select_eval_indices(
            self.validation_indices, int(self.args.eval_pings)
        )
        validation_probe_sources = np.asarray(cache.source_ids[validation_probe_indices], dtype=np.int64)
        if not np.array_equal(validation_probe_sources, self.fixed_source_ids):
            raise ValueError(
                "fixed source IDs disagree with select_eval_indices(validation_indices, eval_pings): "
                f"expected {self.fixed_source_ids.tolist()}, got {validation_probe_sources.tolist()}"
            )
        self.fixed_ping_indices = np.asarray(validation_probe_indices, dtype=np.int64)
        self.fixed_bins = np.arange(326, dtype=np.int64)
        if not np.array_equal(self.fixed_bins, np.arange(cache.num_bins, dtype=np.int64)):
            raise ValueError("fixed diagnostic bins must be exactly 0..325")
        self.original_active_mask = scene.active_mask.detach().clone()
        self.original_order = scene.order.detach().clone()
        self.new_sibling_mask = torch.zeros_like(scene.active_mask)
        self.newly_unlocked_mask = torch.zeros_like(scene.w_re, dtype=torch.bool)
        self.new_sibling_inherited_angular_mask = torch.zeros_like(scene.w_re, dtype=torch.bool)
        self._record_stage("restored_step_700", 700)
        restored = self.stage_metadata[-1]
        if not math.isfinite(float(restored["rel_mse"])) or abs(
            float(restored["rel_mse"]) - self.expected_fixed_val_rel_mse
        ) > self.fixed_val_tolerance:
            self._fail(
                "restored fixed-validation RelMSE disagrees with the declared value: "
                f"observed={restored['rel_mse']!r}, expected={self.expected_fixed_val_rel_mse:.6f}, "
                f"tolerance={self.fixed_val_tolerance:.1e}"
            )

    def _require_ready(self) -> None:
        if self.model is None or self.calibration is None or self.optimizer is None:
            raise RuntimeError("refinement diagnostic observer was not restored")
        if self.scene is None or self.cache is None or self.args is None:
            raise RuntimeError("refinement diagnostic observer has incomplete context")

    def _fail(self, message: str) -> None:
        self.failure = {"message": str(message), "type": "scientific_invariant_failure"}
        self._write_report(status="failed", error=str(message))
        raise RuntimeError(message)

    def _parameter_snapshot(self) -> dict[str, torch.Tensor]:
        self._require_ready()
        result = {
            f"model.{name}": value.detach().clone()
            for name, value in self.model.named_parameters()
        }
        result.update({
            f"calibration.{name}": value.detach().clone()
            for name, value in self.calibration.named_parameters()
        })
        return result

    def _gradient_snapshot(self) -> dict[str, torch.Tensor | None]:
        self._require_ready()
        result: dict[str, torch.Tensor | None] = {}
        for name, value in self.model.named_parameters():
            result[f"model.{name}"] = None if value.grad is None else value.grad.detach().clone()
        for name, value in self.calibration.named_parameters():
            result[f"calibration.{name}"] = None if value.grad is None else value.grad.detach().clone()
        return result

    def _buffer_snapshot(self) -> dict[str, torch.Tensor]:
        self._require_ready()
        result = {
            f"model.{name}": value.detach().clone()
            for name, value in self.model.named_buffers()
        }
        result.update({
            f"calibration.{name}": value.detach().clone()
            for name, value in self.calibration.named_buffers()
        })
        return result

    def _optimizer_parameter_snapshot(self) -> dict[str, dict[str, Any]]:
        self._require_ready()
        result: dict[str, dict[str, Any]] = {}
        for name, parameter in self.model.named_parameters():
            result[f"model.{name}"] = _clone_nested(self.optimizer.state.get(parameter, {}))
        for name, parameter in self.calibration.named_parameters():
            result[f"calibration.{name}"] = _clone_nested(self.optimizer.state.get(parameter, {}))
        return result

    def _parameter_snapshot_now(self) -> dict[str, torch.Tensor]:
        return self._parameter_snapshot()

    def _allowed_intervention_masks(self, label: str) -> dict[str, torch.Tensor]:
        self._require_ready()
        zero_for = {name: torch.zeros_like(value, dtype=torch.bool) for name, value in self._parameter_snapshot().items()}
        coefficient_names = {
            "model.coefficient_field.scene.w_re",
            "model.coefficient_field.scene.w_im",
        }
        degree = self.scene.basis_degree.reshape(1, -1)
        if label == INTERVENTION_LABELS[0]:
            for name, value in zero_for.items():
                if name.startswith("calibration."):
                    zero_for[name] = torch.ones_like(value, dtype=torch.bool)
        elif label == INTERVENTION_LABELS[1]:
            allowed = self.original_active_mask.reshape(-1, 1) & (degree == 0)
            for name in coefficient_names:
                zero_for[name] = allowed
        elif label == INTERVENTION_LABELS[2]:
            allowed = self.new_sibling_mask.reshape(-1, 1) & (degree == 0)
            for name in coefficient_names:
                zero_for[name] = allowed
        elif label == INTERVENTION_LABELS[3]:
            allowed = self.newly_unlocked_mask | self.new_sibling_inherited_angular_mask
            for name in coefficient_names:
                zero_for[name] = allowed
        else:
            raise ValueError(f"unknown intervention {label!r}")
        return zero_for

    def _assert_only_allowed_changes(
        self,
        before: Mapping[str, torch.Tensor],
        label: str,
    ) -> None:
        after = self._parameter_snapshot_now()
        allowed = self._allowed_intervention_masks(label)
        for name, value in before.items():
            changed = after[name] != value
            if bool((changed & ~allowed[name]).any()):
                raise RuntimeError(f"intervention {label} changed a non-target parameter: {name}")

    @staticmethod
    def _state_mask_equal(
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        key: str,
        mask: torch.Tensor,
    ) -> bool:
        before_value = before.get(key)
        after_value = after.get(key)
        if before_value is None or after_value is None:
            return before_value is None and after_value is None
        if not torch.is_tensor(before_value) or not torch.is_tensor(after_value):
            return before_value == after_value
        return torch.equal(before_value[mask], after_value[mask])

    @staticmethod
    def _state_mask_zero(state: Mapping[str, Any], key: str, mask: torch.Tensor) -> bool:
        value = state.get(key)
        if value is None:
            return True
        if not torch.is_tensor(value):
            return False
        return bool((value[mask] == 0).all())

    def _restore_parameters(self, snapshot: Mapping[str, torch.Tensor]) -> None:
        self._require_ready()
        with torch.no_grad():
            for name, value in self.model.named_parameters():
                value.copy_(snapshot[f"model.{name}"])
            for name, value in self.calibration.named_parameters():
                value.copy_(snapshot[f"calibration.{name}"])

    def _rng_snapshot(self) -> dict[str, Any]:
        self._require_ready()
        if self.rng is None:
            raise RuntimeError("diagnostic RNG is unavailable")
        return {
            "numpy_generator": copy.deepcopy(self.rng.bit_generator.state),
            "torch_cpu": torch.get_rng_state().clone(),
            "cuda": _clone_nested(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None,
        }

    def _restore_rng(self, snapshot: Mapping[str, Any]) -> None:
        self._require_ready()
        if self.rng is None:
            raise RuntimeError("diagnostic RNG is unavailable")
        self.rng.bit_generator.state = copy.deepcopy(snapshot["numpy_generator"])
        torch.set_rng_state(snapshot["torch_cpu"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(snapshot["cuda"])

    def _gain_metadata(self) -> dict[str, float]:
        self._require_ready()
        with torch.no_grad():
            one = torch.ones((), dtype=torch.complex64, device=self.device)
            gain = self.calibration(one).detach()
        if not bool(torch.isfinite(gain.real)) or not bool(torch.isfinite(gain.imag)):
            raise FloatingPointError("non-finite complex calibration gain")
        return {
            "gain_real": float(gain.real.item()),
            "gain_imag": float(gain.imag.item()),
            "gain_abs": float(gain.abs().item()),
            "gain_phase": float(torch.angle(gain).item()),
        }

    def _render_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._require_ready()
        was_model_training = self.model.training
        was_calibration_training = self.calibration.training
        parameters_before = self._parameter_snapshot()
        gradients_before = self._gradient_snapshot()
        buffers_before = self._buffer_snapshot()
        optimizer_before = _clone_nested(self.optimizer.state_dict())
        rng_before = self._rng_snapshot()
        predicted: list[np.ndarray] = []
        raw: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        self.model.eval()
        self.calibration.eval()
        try:
            with torch.no_grad():
                for ping in self.fixed_ping_indices.tolist():
                    _loss, _metrics, aux = train_sas.render_one(
                        self.model,
                        self.calibration,
                        self.cache,
                        int(ping),
                        self.fixed_bins,
                        self.args,
                        self.device,
                        allow_calibration_init=False,
                    )
                    values = (
                        aux["calibration_predicted"],
                        aux["calibration_raw"],
                        aux["calibration_target"],
                    )
                    if not all(bool(torch.isfinite(value).all()) for value in values):
                        raise FloatingPointError("non-finite fixed-ping diagnostic render")
                    predicted.append(values[0].detach().cpu().numpy().astype(np.complex64, copy=True))
                    raw.append(values[1].detach().cpu().numpy().astype(np.complex64, copy=True))
                    targets.append(values[2].detach().cpu().numpy().astype(np.complex64, copy=True))
        finally:
            self.model.train(was_model_training)
            self.calibration.train(was_calibration_training)
            self._restore_rng(rng_before)
        parameters_after = self._parameter_snapshot()
        gradients_after = self._gradient_snapshot()
        buffers_after = self._buffer_snapshot()
        optimizer_after = self.optimizer.state_dict()
        parameters_exact = all(
            torch.equal(parameters_before[name], parameters_after[name]) for name in parameters_before
        )
        gradients_exact = all(
            (gradients_before[name] is None and gradients_after[name] is None)
            or (
                gradients_before[name] is not None
                and gradients_after[name] is not None
                and torch.equal(gradients_before[name], gradients_after[name])
            )
            for name in gradients_before
        )
        buffers_exact = all(
            torch.equal(buffers_before[name], buffers_after[name]) for name in buffers_before
        )
        optimizer_exact = _nested_equal(optimizer_before, optimizer_after)
        rng_exact = _nested_equal(rng_before, self._rng_snapshot())
        modes_exact = self.model.training == was_model_training and self.calibration.training == was_calibration_training
        observation_check = {
            "label": self._observation_label or "render",
            "parameters_exact": parameters_exact,
            "gradients_exact": gradients_exact,
            "buffers_exact": buffers_exact,
            "optimizer_state_exact": optimizer_exact,
            "rng_exact": rng_exact,
            "modes_exact": modes_exact,
            "all_exact": bool(
                parameters_exact
                and gradients_exact
                and buffers_exact
                and optimizer_exact
                and rng_exact
                and modes_exact
            ),
        }
        self.observation_checks.append(observation_check)
        if not observation_check["all_exact"]:
            raise RuntimeError(
                f"observation mutated training state at {observation_check['label']}"
            )
        return np.stack(predicted), np.stack(raw), np.stack(targets)

    @staticmethod
    def _relative_mse(predicted: np.ndarray, target: np.ndarray) -> float:
        numerator = float(np.square(np.abs(predicted - target).astype(np.float64)).sum())
        denominator = float(np.square(np.abs(target).astype(np.float64)).sum())
        if denominator <= 0.0:
            return float("nan")
        return numerator / denominator

    def _record_stage(self, label: str, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        previous_label = self._observation_label
        self._observation_label = label
        try:
            predicted, raw, target = self._render_arrays()
        finally:
            self._observation_label = previous_label
        self.stage_metadata.append({
            "label": str(label),
            "step": int(step),
            "rel_mse": self._relative_mse(predicted, target),
            **self._gain_metadata(),
        })
        self.stage_predictions.append(predicted)
        self.stage_raw_predictions.append(raw)
        self.stage_targets.append(target)
        return predicted, raw, target

    def on_before_optimizer(self, **context: Any) -> None:
        self._require_ready()
        step = int(context["step"])
        if step != self.expected_next_step or step > 1010:
            self._fail(
                f"replay optimizer step mismatch: expected {self.expected_next_step}, got {step}"
            )
        ping = int(context["ping"])
        bins = np.asarray(context["bins"], dtype=np.int64)
        if self.reference_rng is None:
            raise RuntimeError("reference replay RNG is unavailable")
        expected_ping = int(self.reference_rng.choice(self.train_indices))
        expected_bins = train_sas.select_bins(
            self.reference_rng,
            np.asarray(self.cache.weights[expected_ping]),
            int(self.args.max_bins),
        )
        actual_state = copy.deepcopy(self.rng.bit_generator.state)
        reference_state = copy.deepcopy(self.reference_rng.bit_generator.state)
        if ping != expected_ping or not np.array_equal(bins, expected_bins):
            self._fail(
                f"replay sampling mismatch at step {step}: actual ping/bins disagree with "
                "the restored reference generator"
            )
        if not _nested_equal(actual_state, reference_state):
            self._fail(f"replay RNG state mismatch after selection at step {step}")
        source_id = int(self.cache.source_ids[ping])
        self.selection_sequence.append({
            "step": step,
            "cache_ping_index": ping,
            "source_id": source_id,
            "bin_ids": bins.tolist(),
        })
        self._before_optimizer = {
            "step": step,
            "w_re": self.scene.w_re.detach().clone(),
            "w_im": self.scene.w_im.detach().clone(),
            "delta_raw": self.scene.delta_raw.detach().clone(),
            "w_re_grad": None if self.scene.w_re.grad is None else self.scene.w_re.grad.detach().clone(),
            "w_im_grad": None if self.scene.w_im.grad is None else self.scene.w_im.grad.detach().clone(),
            "delta_raw_grad": None if self.scene.delta_raw.grad is None else self.scene.delta_raw.grad.detach().clone(),
            "calibration": {
                name: {
                    "before": value.detach().clone(),
                    "grad": None if value.grad is None else value.grad.detach().clone(),
                }
                for name, value in self.calibration.named_parameters()
            },
        }
        self.expected_next_step += 1

    def _group_masks(self) -> dict[str, torch.Tensor]:
        self._require_ready()
        degree = self.scene.basis_degree.reshape(1, -1)
        active = self.scene.active_mask.reshape(-1, 1)
        original = self.original_active_mask.reshape(-1, 1)
        new_siblings = self.new_sibling_mask.reshape(-1, 1)
        return {
            "original_active_dc": original & (degree == 0),
            "new_sibling_dc": new_siblings & (degree == 0),
            "newly_unlocked_angular": (
                self.newly_unlocked_mask | self.new_sibling_inherited_angular_mask
            ),
            "new_sibling_inherited_angular": self.new_sibling_inherited_angular_mask.clone(),
            "active_unlocked_coefficients": active & (degree <= self.scene.order.reshape(-1, 1)),
        }

    def _append_coefficient_group_row(
        self,
        step: int,
        label: str,
        mask: torch.Tensor,
        before_re: torch.Tensor,
        before_im: torch.Tensor,
        grad_re: torch.Tensor | None,
        grad_im: torch.Tensor | None,
    ) -> None:
        after_re = self.scene.w_re.detach()
        after_im = self.scene.w_im.detach()
        selected_before = torch.complex(before_re[mask], before_im[mask])
        selected_after = torch.complex(after_re[mask], after_im[mask])
        update = selected_after - selected_before
        if grad_re is None or grad_im is None:
            gradient = torch.zeros_like(selected_after)
        else:
            gradient = torch.complex(grad_re[mask], grad_im[mask])
        coefficient_rms, coefficient_max = _tensor_stats(selected_after)
        gradient_rms, gradient_max = _tensor_stats(gradient)
        update_rms, update_max = _tensor_stats(update)
        state_re = self.optimizer.state.get(self.scene.w_re, {})
        state_im = self.optimizer.state.get(self.scene.w_im, {})
        exp_avg = torch.cat([
            state_re.get("exp_avg", torch.zeros_like(self.scene.w_re))[mask],
            state_im.get("exp_avg", torch.zeros_like(self.scene.w_im))[mask],
        ])
        exp_avg_sq = torch.cat([
            state_re.get("exp_avg_sq", torch.zeros_like(self.scene.w_re))[mask],
            state_im.get("exp_avg_sq", torch.zeros_like(self.scene.w_im))[mask],
        ])
        exp_avg_rms, exp_avg_max = _tensor_stats(exp_avg)
        exp_avg_sq_rms, exp_avg_sq_max = _tensor_stats(exp_avg_sq)
        step_value = state_re.get("step", 0.0)
        if torch.is_tensor(step_value):
            step_value = float(step_value.item())
        self.update_rows.append({
            "step": int(step),
            "group": label,
            "parameter": "scene.w_re+scene.w_im",
            "count": int(mask.sum().item()),
            "coefficient_rms": coefficient_rms,
            "coefficient_max": coefficient_max,
            "clipped_gradient_rms": gradient_rms,
            "clipped_gradient_max": gradient_max,
            "actual_update_rms": update_rms,
            "actual_update_max": update_max,
            "adam_step": float(step_value),
            "exp_avg_rms": exp_avg_rms,
            "exp_avg_max": exp_avg_max,
            "exp_avg_sq_rms": exp_avg_sq_rms,
            "exp_avg_sq_max": exp_avg_sq_max,
        })

    def _append_real_parameter_row(
        self,
        step: int,
        group: str,
        parameter_name: str,
        parameter: torch.Tensor,
        before: torch.Tensor,
        gradient: torch.Tensor | None,
    ) -> None:
        current = parameter.detach()
        grad = torch.zeros_like(current) if gradient is None else gradient
        state = self.optimizer.state.get(parameter, {})
        exp_avg = state.get("exp_avg", torch.zeros_like(parameter))
        exp_avg_sq = state.get("exp_avg_sq", torch.zeros_like(parameter))
        step_value = state.get("step", 0.0)
        if torch.is_tensor(step_value):
            step_value = float(step_value.item())
        coefficient_rms, coefficient_max = _tensor_stats(current)
        gradient_rms, gradient_max = _tensor_stats(grad)
        update_rms, update_max = _tensor_stats(current - before)
        exp_avg_rms, exp_avg_max = _tensor_stats(exp_avg)
        exp_avg_sq_rms, exp_avg_sq_max = _tensor_stats(exp_avg_sq)
        self.update_rows.append({
            "step": int(step),
            "group": group,
            "parameter": parameter_name,
            "count": int(parameter.numel()),
            "coefficient_rms": coefficient_rms,
            "coefficient_max": coefficient_max,
            "clipped_gradient_rms": gradient_rms,
            "clipped_gradient_max": gradient_max,
            "actual_update_rms": update_rms,
            "actual_update_max": update_max,
            "adam_step": float(step_value),
            "exp_avg_rms": exp_avg_rms,
            "exp_avg_max": exp_avg_max,
            "exp_avg_sq_rms": exp_avg_sq_rms,
            "exp_avg_sq_max": exp_avg_sq_max,
        })

    def on_after_optimizer(self, **context: Any) -> None:
        self._require_ready()
        if self._before_optimizer is None:
            raise RuntimeError("optimizer observer lost its before-step snapshot")
        before = self._before_optimizer
        step = int(context["step"])
        for label, mask in self._group_masks().items():
            self._append_coefficient_group_row(
                step,
                label,
                mask,
                before["w_re"],
                before["w_im"],
                before["w_re_grad"],
                before["w_im_grad"],
            )
        self._append_real_parameter_row(
            step,
            "position_all",
            "scene.delta_raw",
            self.scene.delta_raw,
            before["delta_raw"],
            before["delta_raw_grad"],
        )
        for name, parameter in self.calibration.named_parameters():
            saved = before["calibration"][name]
            self._append_real_parameter_row(
                step,
                "calibration_all",
                f"calibration.{name}",
                parameter,
                saved["before"],
                saved["grad"],
            )

        if step in POST_UPDATE_STEPS:
            self._record_stage(f"after_step_{step}", step)
            self._run_interventions(f"after_step_{step}", step)
        self._before_optimizer = None

    def on_before_refinement(self, **context: Any) -> None:
        self._require_ready()
        step = int(context["step"])
        before_predictions, before_raw, before_target = self._record_stage(
            f"step_{step}_pre_refinement", step
        )
        self._pending_refinement = {
            "step": step,
            "before_predictions": before_predictions,
            "before_raw": before_raw,
            "before_target": before_target,
            "parameters_before": self._parameter_snapshot(),
            "optimizer_before": self._optimizer_parameter_snapshot(),
            "active_before": self.scene.active_mask.detach().clone(),
            "order_before": self.scene.order.detach().clone(),
            "level_before": self.scene.level.detach().clone(),
        }

    def on_after_refinement(self, **context: Any) -> None:
        self._require_ready()
        if self._pending_refinement is None:
            raise RuntimeError("refinement observer lost its before-event snapshot")
        pending = self._pending_refinement
        step = int(context["step"])
        after_predictions, after_raw, after_target = self._record_stage(
            f"step_{step}_post_refinement", step
        )
        active_before = pending["active_before"]
        order_before = pending["order_before"]
        level_before = pending["level_before"]
        active_after = self.scene.active_mask.detach().clone()
        order_after = self.scene.order.detach().clone()
        level_after = self.scene.level.detach().clone()
        new_rows = active_after & ~active_before
        inherited_rows = active_before & active_after & (level_after != level_before)
        unlocked_rows = active_before & active_after & (order_after > order_before)
        degree = self.scene.basis_degree.reshape(1, -1)
        unlocked_mask = unlocked_rows.reshape(-1, 1) & (
            (degree > order_before.reshape(-1, 1))
            & (degree <= order_after.reshape(-1, 1))
        )
        inherited_angular_mask = new_rows.reshape(-1, 1) & (
            (degree > 0) & (degree <= order_after.reshape(-1, 1))
        )
        parameters_before = pending["parameters_before"]
        parameters_after = self._parameter_snapshot()
        optimizer_before = pending["optimizer_before"]
        optimizer_after = self._optimizer_parameter_snapshot()
        preexisting_coeff_mask = active_before.reshape(-1, 1) & (
            degree <= order_before.reshape(-1, 1)
        )
        coefficient_names = (
            "model.coefficient_field.scene.w_re",
            "model.coefficient_field.scene.w_im",
        )
        preexisting_coefficients_preserved = all(
            torch.equal(
                parameters_before[name][preexisting_coeff_mask],
                parameters_after[name][preexisting_coeff_mask],
            )
            for name in coefficient_names
        )
        gain_unchanged = all(
            torch.equal(parameters_before[f"calibration.{name}"], parameters_after[f"calibration.{name}"])
            for name, _value in self.calibration.named_parameters()
        )
        coefficient_new_mask = new_rows.reshape(-1, 1).expand_as(self.scene.w_re)
        position_reset_mask = (
            (new_rows | inherited_rows).reshape(-1, 1).expand_as(self.scene.delta_raw)
        )

        inherited_adam_ages_moments_preserved = True
        new_sibling_moments_zero = True
        newly_unlocked_moments_zero = True
        for name in coefficient_names:
            before_state = optimizer_before[name]
            after_state = optimizer_after[name]
            for state_key in ("exp_avg", "exp_avg_sq"):
                inherited_adam_ages_moments_preserved &= self._state_mask_equal(
                    before_state, after_state, state_key, preexisting_coeff_mask
                )
                new_sibling_moments_zero &= self._state_mask_zero(
                    after_state, state_key, coefficient_new_mask
                )
                newly_unlocked_moments_zero &= self._state_mask_zero(
                    after_state, state_key, unlocked_mask
                )
            inherited_adam_ages_moments_preserved &= _nested_equal(
                before_state.get("step"), after_state.get("step")
            )
        delta_after = optimizer_after["model.coefficient_field.scene.delta_raw"]
        position_moments_zero_after_split = all(
            self._state_mask_zero(delta_after, state_key, position_reset_mask)
            for state_key in ("exp_avg", "exp_avg_sq")
        )

        self.new_sibling_mask |= new_rows
        self.newly_unlocked_mask |= unlocked_mask
        self.new_sibling_inherited_angular_mask |= inherited_angular_mask
        difference = np.abs(after_predictions - pending["before_predictions"])
        raw_difference = np.abs(after_raw - pending["before_raw"])
        max_difference = float(difference.max()) if difference.size else 0.0
        max_raw_difference = float(raw_difference.max()) if raw_difference.size else 0.0
        prediction_preserved = bool(
            np.isfinite(difference).all()
            and np.allclose(
                after_predictions,
                pending["before_predictions"],
                atol=1.0e-6,
                rtol=1.0e-5,
            )
        )
        raw_preserved = bool(
            np.isfinite(raw_difference).all()
            and np.allclose(
                after_raw,
                pending["before_raw"],
                atol=1.0e-6,
                rtol=1.0e-5,
            )
        )
        new_sibling_coefficients_zero = bool(
            (self.scene.w_re[new_rows].abs().sum() == 0)
            and (self.scene.w_im[new_rows].abs().sum() == 0)
        )
        newly_unlocked_coefficients_zero = bool(
            (self.scene.w_re[unlocked_mask].abs().sum() == 0)
            and (self.scene.w_im[unlocked_mask].abs().sum() == 0)
        )
        inherited_angular_coefficients_zero = bool(
            (self.scene.w_re[inherited_angular_mask].abs().sum() == 0)
            and (self.scene.w_im[inherited_angular_mask].abs().sum() == 0)
        )
        expected_counts = dict(self.expected_historical_counts)
        actual_counts = {
            "n_split": int(context["n_split"]),
            "n_angular": int(context["n_angular"]),
            "active_after": int(active_after.sum().item()),
        }
        replay_divergence = [
            f"{key}: expected {expected}, observed {actual_counts[key]}"
            for key, expected in expected_counts.items()
            if actual_counts[key] != expected
        ]
        self.replay_divergences.extend(replay_divergence)
        position_parameters_changed = not torch.equal(
            parameters_before["model.coefficient_field.scene.delta_raw"],
            parameters_after["model.coefficient_field.scene.delta_raw"],
        )
        zero_birth_ok = bool(
            new_sibling_coefficients_zero
            and newly_unlocked_coefficients_zero
            and inherited_angular_coefficients_zero
            and new_sibling_moments_zero
            and newly_unlocked_moments_zero
            and position_moments_zero_after_split
        )
        topology_record = {
            "step": step,
            "event": int(context["snapshot"]["event"]),
            "n_split": int(context["n_split"]),
            "n_angular": int(context["n_angular"]),
            "active_mask_before": [bool(value) for value in _safe_list(active_before)],
            "active_mask_after": [bool(value) for value in _safe_list(active_after)],
            "order_before": [int(value) for value in _safe_list(order_before)],
            "order_after": [int(value) for value in _safe_list(order_after)],
            "level_before": [int(value) for value in _safe_list(level_before)],
            "level_after": [int(value) for value in _safe_list(level_after)],
            "new_sibling_rows": [int(value) for value in new_rows.nonzero(as_tuple=True)[0].cpu().tolist()],
            "inherited_rows": [int(value) for value in inherited_rows.nonzero(as_tuple=True)[0].cpu().tolist()],
            "newly_unlocked_rows": [int(value) for value in unlocked_rows.nonzero(as_tuple=True)[0].cpu().tolist()],
            "newly_unlocked_columns_by_row": {
                str(int(row)): [
                    int(column)
                    for column in unlocked_mask[row].nonzero(as_tuple=True)[0].cpu().tolist()
                ]
                for row in unlocked_rows.nonzero(as_tuple=True)[0].cpu().tolist()
            },
            "new_sibling_inherited_angular_columns_by_row": {
                str(int(row)): [
                    int(column)
                    for column in inherited_angular_mask[row].nonzero(as_tuple=True)[0].cpu().tolist()
                ]
                for row in new_rows.nonzero(as_tuple=True)[0].cpu().tolist()
            },
            "new_sibling_coefficients_zero": new_sibling_coefficients_zero,
            "newly_unlocked_coefficients_zero": newly_unlocked_coefficients_zero,
            "new_sibling_inherited_angular_coefficients_zero": inherited_angular_coefficients_zero,
            "new_sibling_moments_zero": new_sibling_moments_zero,
            "newly_unlocked_moments_zero": newly_unlocked_moments_zero,
            "position_moments_zero_after_split": position_moments_zero_after_split,
            "inherited_coefficients_preserved": preexisting_coefficients_preserved,
            "gain_unchanged": gain_unchanged,
            "inherited_adam_ages_moments_preserved": inherited_adam_ages_moments_preserved,
            "position_parameters_changed_by_split": position_parameters_changed,
            "position_state_preservation_asserted": False,
            "zero_birth_invariants_pass": zero_birth_ok,
            "pre_post_prediction_max_abs": max_difference,
            "pre_post_prediction_preserved": prediction_preserved,
            "pre_post_raw_max_abs": max_raw_difference,
            "pre_post_raw_preserved": raw_preserved,
            "pre_post_prediction_atol": 1.0e-6,
            "pre_post_prediction_rtol": 1.0e-5,
            "expected_historical_counts": expected_counts,
            "actual_counts": actual_counts,
            "replay_divergence": replay_divergence,
            "report": str(context["report"]),
        }
        self.refinement_events.append(topology_record)
        self._pending_refinement = None
        if not (
            zero_birth_ok
            and prediction_preserved
            and raw_preserved
            and preexisting_coefficients_preserved
            and gain_unchanged
            and inherited_adam_ages_moments_preserved
        ):
            self._fail(
                f"refinement invariant failed at step {step}; evidence is retained in the partial report"
            )
        self.post_refinement_baseline = self._parameter_snapshot()

    def _run_interventions(self, stage_label: str, step: int) -> None:
        self._require_ready()
        if self.post_refinement_baseline is None:
            raise RuntimeError("post-refinement baseline is unavailable for interventions")
        parameter_before = self._parameter_snapshot()
        gradient_before = self._gradient_snapshot()
        buffer_before = self._buffer_snapshot()
        optimizer_before = _clone_nested(self.optimizer.state_dict())
        numpy_rng_before = copy.deepcopy(self.rng.bit_generator.state)
        torch_rng_before = torch.get_rng_state().clone()
        cuda_rng_before = (
            _clone_nested(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None
        )
        baseline = self.post_refinement_baseline
        interventions = {
            INTERVENTION_LABELS[0]: lambda: self._apply_gain_baseline(baseline),
            INTERVENTION_LABELS[1]: lambda: self._apply_original_dc_baseline(baseline),
            INTERVENTION_LABELS[2]: self._zero_new_sibling_dc,
            INTERVENTION_LABELS[3]: self._zero_newly_unlocked_angular,
        }
        try:
            for label, apply in interventions.items():
                # Each factor is evaluated from the identical current
                # post-update state.  The outer finally below is still needed
                # for exceptions and for the caller's continuation state.
                self._restore_parameters(parameter_before)
                self._assert_only_allowed_changes(parameter_before, label)
                with torch.no_grad():
                    apply()
                self._assert_only_allowed_changes(parameter_before, label)
                previous_label = self._observation_label
                self._observation_label = f"{stage_label}:{label}"
                try:
                    predicted, raw, target = self._render_arrays()
                finally:
                    self._observation_label = previous_label
                self.intervention_metadata.append({
                    "stage": stage_label,
                    "step": int(step),
                    "label": label,
                    "rel_mse": self._relative_mse(predicted, target),
                })
                self.intervention_predictions.append(predicted)
                self.intervention_raw_predictions.append(raw)
        finally:
            self._restore_parameters(parameter_before)
            restored_parameters = self._parameter_snapshot()
            parameters_exact = all(
                torch.equal(restored_parameters[name], value)
                for name, value in parameter_before.items()
            )
            gradient_after = self._gradient_snapshot()
            gradients_exact = all(
                (
                    (gradient_before[name] is None and gradient_after[name] is None)
                    or (
                        gradient_before[name] is not None
                        and gradient_after[name] is not None
                        and torch.equal(gradient_before[name], gradient_after[name])
                    )
                )
                for name in gradient_before
            )
            optimizer_exact = _nested_equal(optimizer_before, self.optimizer.state_dict())
            buffer_after = self._buffer_snapshot()
            buffers_exact = all(
                torch.equal(buffer_after[name], value) for name, value in buffer_before.items()
            )
            numpy_rng_exact = _nested_equal(numpy_rng_before, self.rng.bit_generator.state)
            torch_rng_exact = torch.equal(torch_rng_before, torch.get_rng_state())
            cuda_rng_exact = (
                _nested_equal(cuda_rng_before, torch.cuda.get_rng_state_all())
                if torch.cuda.is_available()
                else True
            )
            check = {
                "stage": stage_label,
                "step": int(step),
                "parameters_exact": parameters_exact,
                "gradients_exact": gradients_exact,
                "buffers_exact": buffers_exact,
                "optimizer_state_exact": optimizer_exact,
                "numpy_rng_exact": numpy_rng_exact,
                "torch_rng_exact": torch_rng_exact,
                "cuda_rng_exact": cuda_rng_exact,
                "all_exact": bool(
                    parameters_exact
                    and gradients_exact
                    and buffers_exact
                    and optimizer_exact
                    and numpy_rng_exact
                    and torch_rng_exact
                    and cuda_rng_exact
                ),
            }
            self.restore_checks.append(check)
            if not check["all_exact"]:
                raise RuntimeError(f"intervention restore was not exact at {stage_label}")

    def _apply_gain_baseline(self, baseline: Mapping[str, torch.Tensor]) -> None:
        for name, parameter in self.calibration.named_parameters():
            parameter.copy_(baseline[f"calibration.{name}"])

    def _apply_original_dc_baseline(self, baseline: Mapping[str, torch.Tensor]) -> None:
        rows = self.original_active_mask
        self.scene.w_re[rows, 0] = baseline["model.coefficient_field.scene.w_re"][rows, 0]
        self.scene.w_im[rows, 0] = baseline["model.coefficient_field.scene.w_im"][rows, 0]

    def _zero_new_sibling_dc(self) -> None:
        self.scene.w_re[self.new_sibling_mask, 0] = 0
        self.scene.w_im[self.new_sibling_mask, 0] = 0

    def _zero_newly_unlocked_angular(self) -> None:
        mask = self.newly_unlocked_mask | self.new_sibling_inherited_angular_mask
        self.scene.w_re[mask] = 0
        self.scene.w_im[mask] = 0

    def _summary(self, args: Any, *, status: str, error: str | None = None) -> dict[str, Any]:
        restored = self.stage_metadata[0] if self.stage_metadata else {}
        restored_value = float(restored.get("rel_mse", float("nan")))
        expected_error = abs(restored_value - self.expected_fixed_val_rel_mse)
        stage_labels = [record["label"] for record in self.stage_metadata]
        expected_labels = ["restored_step_700", "step_1000_pre_refinement", "step_1000_post_refinement"]
        expected_labels.extend(f"after_step_{step}" for step in POST_UPDATE_STEPS)
        sequence_steps = [int(record["step"]) for record in self.selection_sequence]
        invariant_status = {
            "replay_sampling": bool(
                len(self.selection_sequence) == 310 and self.expected_next_step == 1011
            ),
            "restored_fixed_val_rel_mse": bool(
                math.isfinite(restored_value)
                and expected_error <= self.fixed_val_tolerance
            ),
            "stage_order": bool(stage_labels == expected_labels),
            "pre_post_refinement": bool(
                self.refinement_events
                and self.refinement_events[0].get("pre_post_prediction_preserved")
                and self.refinement_events[0].get("pre_post_raw_preserved")
            ),
            "zero_birth_and_inherited_state": bool(
                self.refinement_events
                and self.refinement_events[0].get("zero_birth_invariants_pass")
                and self.refinement_events[0].get("inherited_coefficients_preserved")
                and self.refinement_events[0].get("gain_unchanged")
                and self.refinement_events[0].get("inherited_adam_ages_moments_preserved")
            ),
            "intervention_restoration": bool(
                self.restore_checks and all(record["all_exact"] for record in self.restore_checks)
            ),
            "observation_state_preservation": bool(
                self.observation_checks
                and all(record["all_exact"] for record in self.observation_checks)
            ),
        }
        return {
            "status": status,
            "error": error,
            "diagnostic_contract": "adaptive_airsas_refinement_replay_v1",
            "replay_start_step": 700,
            "replay_end_step": int(args.steps) if args is not None else None,
            "replay_optimizer_updates": len(self.selection_sequence),
            "expected_optimizer_updates": 310,
            "optimizer_update_count_exact": bool(len(self.selection_sequence) == 310),
            "selection_steps_first_last": [sequence_steps[0], sequence_steps[-1]] if sequence_steps else [],
            "training_selection_sequence": self.selection_sequence,
            "fixed_source_ids": self.fixed_source_ids.tolist(),
            "fixed_bin_ids": self.fixed_bins.tolist(),
            "fixed_bin_count": int(self.fixed_bins.size),
            "stage_labels": stage_labels,
            "expected_stage_labels": expected_labels,
            "stage_order_exact": bool(stage_labels == expected_labels),
            "restored_fixed_val_rel_mse": restored_value,
            "expected_restored_fixed_val_rel_mse": self.expected_fixed_val_rel_mse,
            "restored_fixed_val_rel_mse_abs_error": expected_error,
            "restored_fixed_val_rel_mse_within_tolerance": bool(expected_error <= self.fixed_val_tolerance),
            "restored_fixed_val_rel_mse_tolerance": self.fixed_val_tolerance,
            "refinement_events": len(self.refinement_events),
            "intervention_count": len(self.intervention_metadata),
            "all_intervention_restores_exact": invariant_status["intervention_restoration"],
            "stage_metadata": self.stage_metadata,
            "intervention_metadata": self.intervention_metadata,
            "restore_checks": self.restore_checks,
            "observation_checks": self.observation_checks,
            "invariant_status": invariant_status,
            "replay_divergence": self.replay_divergences,
            "recipe": vars(args) if args is not None else {},
        }

    @staticmethod
    def _stack_or_empty(values: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
        if not values:
            return np.empty((0,), dtype=dtype)
        return np.stack(values)

    def _write_report(self, *, status: str, error: str | None = None) -> dict[str, Any]:
        """Publish complete or partial evidence; never create a success checkpoint."""

        args = self.args
        np.savez_compressed(
            self.output_dir / "fixed_ping_predictions.npz",
            predicted=self._stack_or_empty(self.stage_predictions, np.dtype(np.complex64)),
            predicted_raw=self._stack_or_empty(self.stage_raw_predictions, np.dtype(np.complex64)),
            target=self._stack_or_empty(self.stage_targets, np.dtype(np.complex64)),
            stage_labels=np.asarray([record["label"] for record in self.stage_metadata]),
            stage_steps=np.asarray([record["step"] for record in self.stage_metadata], dtype=np.int64),
            source_ids=self.fixed_source_ids,
            ping_indices=(self.fixed_ping_indices if self.fixed_ping_indices is not None else np.empty(0, dtype=np.int64)),
            bin_ids=self.fixed_bins,
            intervention_predicted=self._stack_or_empty(self.intervention_predictions, np.dtype(np.complex64)),
            intervention_predicted_raw=self._stack_or_empty(self.intervention_raw_predictions, np.dtype(np.complex64)),
            intervention_stage_labels=np.asarray([record["stage"] for record in self.intervention_metadata]),
            intervention_labels=np.asarray([record["label"] for record in self.intervention_metadata]),
            intervention_steps=np.asarray([record["step"] for record in self.intervention_metadata], dtype=np.int64),
        )
        train_sas.atomic_json({"events": self.refinement_events}, self.output_dir / "refinement_events.json")
        fields = [
            "step", "group", "parameter", "count", "coefficient_rms", "coefficient_max",
            "clipped_gradient_rms", "clipped_gradient_max", "actual_update_rms", "actual_update_max",
            "adam_step", "exp_avg_rms", "exp_avg_max", "exp_avg_sq_rms", "exp_avg_sq_max",
        ]
        temporary = self.output_dir / "update_groups.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.update_rows)
        temporary.replace(self.output_dir / "update_groups.csv")
        summary = self._summary(args, status=status, error=error)
        train_sas.atomic_json(summary, self.output_dir / "diagnostic_summary.json")
        return summary

    def on_finish(self, **context: Any) -> None:
        self._require_ready()
        step = int(context["step"])
        if step != 1010:
            raise ValueError(f"refinement replay must finish at step 1010, got {step}")
        expected_labels = [
            "restored_step_700",
            "step_1000_pre_refinement",
            "step_1000_post_refinement",
            *(f"after_step_{value}" for value in POST_UPDATE_STEPS),
        ]
        actual_labels = [record["label"] for record in self.stage_metadata]
        if len(self.selection_sequence) != 310 or self.expected_next_step != 1011:
            self._fail("replay did not complete the exact 310-step sampling sequence")
        if actual_labels != expected_labels:
            self._fail(f"diagnostic stage ordering mismatch: expected {expected_labels}, got {actual_labels}")
        if len(self.refinement_events) != 1:
            self._fail(f"expected one refinement event, observed {len(self.refinement_events)}")
        if not self.restore_checks or not all(record["all_exact"] for record in self.restore_checks):
            self._fail("one or more intervention restoration checks failed")
        if not self.observation_checks or not all(record["all_exact"] for record in self.observation_checks):
            self._fail("one or more observation state-preservation checks failed")
        summary = self._write_report(status="success")
        payload = context["checkpoint_writer"](
            self.model,
            self.calibration,
            self.optimizer,
            step,
            context["best_val"],
            context["rng"],
            context["history"],
            context["args"],
            self.cache,
        )
        payload["diagnostic_summary"] = summary
        payload["diagnostic_output_contract"] = {
            "fixed_ping_predictions": "fixed_ping_predictions.npz",
            "refinement_events": "refinement_events.json",
            "update_groups": "update_groups.csv",
            "diagnostic_summary": "diagnostic_summary.json",
        }
        payload["diagnostic_fixed_source_ids"] = self.fixed_source_ids.tolist()
        payload["diagnostic_fixed_bin_ids"] = self.fixed_bins.tolist()
        payload["diagnostic_stage_labels"] = [record["label"] for record in self.stage_metadata]
        payload["diagnostic_stage_steps"] = [int(record["step"]) for record in self.stage_metadata]
        context["atomic_save"](payload, self.output_dir / "checkpoint_final_diagnostic.pt")
        self._finished = True


def _flag_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def _saved_recipe_argv(
    saved_args: Mapping[str, Any],
    *,
    cache: Path,
    checkpoint: Path,
    output: Path,
    device: str,
    end_step: int,
) -> list[str]:
    """Turn checkpoint args back into train_sas CLI values without duplication."""

    argv = [
        "--cache", str(cache),
        "--model", str(saved_args.get("model", "adaptive_rift_sas")),
        "--checkpoint-root", str(output.parent),
        "--checkpoint-name", output.name,
        "--resume", str(checkpoint),
        "--steps", str(int(end_step)),
        "--device", device,
    ]
    skip = {
        "cache", "model", "checkpoint_root", "checkpoint_name", "resume", "steps",
        "eval_only", "evaluation_role", "profile", "device", "opacity_normalize",
    }
    for key, value in saved_args.items():
        if key in skip or value is None:
            continue
        flag = _flag_name(key)
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        argv.extend([flag, str(value)])
    opacity_normalize = saved_args.get("opacity_normalize")
    if opacity_normalize is True:
        argv.append("--opacity-normalize")
    elif opacity_normalize is False:
        argv.append("--no-opacity-normalize")
    return argv


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True, help="preserved step-700 checkpoint")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--start-step", type=int, default=700)
    parser.add_argument("--end-step", type=int, default=1010)
    parser.add_argument("--expected-fixed-val-rel-mse", type=float, default=1.369373)
    parser.add_argument("--fixed-val-tolerance", type=float, default=1.0e-4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.start_step != 700 or args.end_step != 1010:
        raise ValueError("the accepted refinement replay is fixed to steps 700 through 1010")
    checkpoint_path = Path(args.checkpoint)
    output = Path(args.output)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(state.get("step", -1)) != args.start_step:
        raise ValueError(f"checkpoint must be at step {args.start_step}")
    saved_args = state.get("args")
    if not isinstance(saved_args, dict):
        raise ValueError("checkpoint lacks the saved train_sas recipe")
    if saved_args.get("model") != "adaptive_rift_sas":
        raise ValueError("refinement replay requires adaptive_rift_sas")
    observer = RefinementDiagnosticObserver(
        output,
        expected_fixed_val_rel_mse=args.expected_fixed_val_rel_mse,
        fixed_val_tolerance=args.fixed_val_tolerance,
    )
    replay_argv = _saved_recipe_argv(
        saved_args,
        cache=Path(args.cache),
        checkpoint=checkpoint_path,
        output=output,
        device=args.device,
        end_step=args.end_step,
    )
    try:
        train_sas.main(replay_argv, diagnostic_observer=observer)
    except BaseException as exc:
        observer._write_report(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    if not observer._finished:
        raise RuntimeError("diagnostic observer did not publish its final report")


if __name__ == "__main__":
    main()
