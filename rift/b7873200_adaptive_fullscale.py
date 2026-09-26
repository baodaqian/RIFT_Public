"""Adaptive RIFT B787 full-scale recipe and observation seam.

This module is an opt-in adapter around the ordinary ``train.py`` loop.  It
does not implement a second optimizer or training loop.  The observer records
full-run topology, event-time coherent prediction preservation, post-event
learning, and runtime resource measurements without reading the sealed test or
unused response roles.
"""
from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

try:  # ``resource`` is available on PACE, but not on Windows.
    import resource as _resource
except ImportError:  # pragma: no cover - platform capability branch
    _resource = None


B787_3200_CANONICAL_NPZ_PATH = (
    "/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/"
    "b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz"
)
B787_3200_CANONICAL_MANIFEST_PATH = (
    "/storage/scratch1/1/dbao31/rift_round8b_impl_20260810/splits/round8b/"
    "b78710k_interp_seed42_train3200_val1000_test1000_v1.json"
)

FULLSCALE_SCHEMA = "rift_b7873200_adaptive_fullscale_observer_v2"
FULLSCALE_CHECKPOINT_NAME = "b78710k_adaptive_rift_fullscale_v1"
FULLSCALE_EXECUTION_LABEL = "b78710k_adaptive_rift_fullscale_v1"
FULLSCALE_NUM_TRAIN = 3_200
FULLSCALE_NUM_VALIDATION = 1_000
FULLSCALE_NUM_TEST = 1_000
FULLSCALE_NUM_UNUSED = 4_800
FULLSCALE_GRANULARITY = 48
FULLSCALE_INITIAL_ACTIVE = FULLSCALE_GRANULARITY ** 3
FULLSCALE_MAX_POINTS = 262_144
FULLSCALE_MAX_DEGREE = 3
FULLSCALE_EPOCHS = 150
FULLSCALE_STEP_EVERY = 1
FULLSCALE_EXPECTED_UPDATES = FULLSCALE_NUM_TRAIN * FULLSCALE_EPOCHS
FULLSCALE_REFINE_EVERY = 10
FULLSCALE_PROBE_EVERY = 16
FULLSCALE_MIN_SPATIAL_EXPOSURE = 3_200
FULLSCALE_MIN_ANGULAR_EXPOSURE = 200
FULLSCALE_SPATIAL_FRACTION = 1.0 / 512.0
FULLSCALE_ANGULAR_FRACTION = 1.0 / 16.0
FULLSCALE_SPATIAL_FLOOR = 0.0
FULLSCALE_ANGULAR_FLOOR = 0.0
FULLSCALE_COOLDOWN_EVENTS = 1
FULLSCALE_CHILD_MATURITY_EVENTS = 1
FULLSCALE_SPLIT_MAX_LEVEL = 1
FULLSCALE_PREDICTION_PRESERVATION_ATOL = 1.0e-7
FULLSCALE_EVENT_PROBE_COUNT = 4


def _optional_max(first: int | None, second: int | None) -> int | None:
    values = [value for value in (first, second) if value is not None]
    return max(values) if values else None


def fullscale_train_argv(
    *,
    npz_path: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    checkpoint_root: str | os.PathLike[str],
    resume: str | os.PathLike[str] | None = None,
) -> list[str]:
    """Return the complete reviewed adaptive full-scale trainer command."""

    argv = [
        "--checkpoint-name", FULLSCALE_CHECKPOINT_NAME,
        "--checkpoint-root", os.fspath(checkpoint_root),
        "--execution-contract-label", FULLSCALE_EXECUTION_LABEL,
        "--require-full-resume-state",
        "--data-format", "npz",
        "--npz-path", os.fspath(npz_path),
        "--npz-sealed-protocol",
        "--npz-role-manifest", os.fspath(manifest_path),
        "--num-train", str(FULLSCALE_NUM_TRAIN),
        "--num-val", str(FULLSCALE_NUM_VALIDATION),
        "--num-test", str(FULLSCALE_NUM_TEST),
        "--num-tx", "16",
        "--num-rx", "16",
        "--num-freq-wanted", "600",
        "--epochs", str(FULLSCALE_EPOCHS),
        "--step-every", str(FULLSCALE_STEP_EVERY),
        "--loss", "complex",
        "--scene-repr", "point_sh",
        "--forward-operator", "range",
        "--range-model", "sum2",
        "--compute-dtype", "float64",
        "--point-chunk", "65536",
        "--pair-chunk", "64",
        "--granularity", str(FULLSCALE_GRANULARITY),
        "--extent", "0.15",
        "--max-points", str(FULLSCALE_MAX_POINTS),
        "--adaptive-max-active", str(FULLSCALE_MAX_POINTS),
        "--sh-init-degree", "0",
        "--sh-max-degree", str(FULLSCALE_MAX_DEGREE),
        "--bp-init", "100",
        "--init-scale", "0",
        "--phase-sign", "-1",
        "--lr", "0.003",
        "--pos-lr", "0.003",
        "--adam-eps", "1e-8",
        "--weight-decay", "0",
        "--t0", "10",
        "--t-mult", "2",
        "--seed", "42",
        "--adaptive-capacity-v2",
        "--adaptive-refine-every", str(FULLSCALE_REFINE_EVERY),
        "--adaptive-probe-every", str(FULLSCALE_PROBE_EVERY),
        "--adaptive-min-spatial-exposure", str(FULLSCALE_MIN_SPATIAL_EXPOSURE),
        "--adaptive-min-angular-exposure", str(FULLSCALE_MIN_ANGULAR_EXPOSURE),
        "--adaptive-spatial-fraction", str(FULLSCALE_SPATIAL_FRACTION),
        "--adaptive-angular-fraction", str(FULLSCALE_ANGULAR_FRACTION),
        "--adaptive-spatial-floor", str(FULLSCALE_SPATIAL_FLOOR),
        "--adaptive-angular-floor", str(FULLSCALE_ANGULAR_FLOOR),
        "--adaptive-cooldown-events", str(FULLSCALE_COOLDOWN_EVENTS),
        "--adaptive-child-maturity-events", str(FULLSCALE_CHILD_MATURITY_EVENTS),
        "--split-max-level", str(FULLSCALE_SPLIT_MAX_LEVEL),
        "--split-every", "0",
        "--grow-every", "0",
        "--prune-every", "0",
        "--l1-weight", "3e-7",
        "--sh-degree-weight", "1e-9",
        "--regularizer-normalization", "fixed_initial",
        "--mag-weight", "0",
        "--mag-warmup-epochs", "0",
        "--checkpoint-metric", "val",
    ]
    if resume is not None:
        argv.extend(["--resume", os.fspath(resume)])
    return argv


def _finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all())


def _score_summary(scores: torch.Tensor, eligible: torch.Tensor) -> dict[str, Any]:
    values = scores[eligible]
    if values.numel() == 0:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": int(values.numel()),
        "min": float(values.min().item()),
        "median": float(values.median().item()),
        "max": float(values.max().item()),
    }


class AdaptiveFullScaleObserver:
    """Record the adaptive events of one full B787 trajectory.

    The observer intentionally avoids all-role metric recomputation and full
    parameter snapshots.  The generic trainer already writes train/validation
    history and the sealed contract; this seam adds the topology-specific
    evidence that the generic CLI cannot provide on its own.
    """

    schema = FULLSCALE_SCHEMA
    num_train = FULLSCALE_NUM_TRAIN
    expected_updates = FULLSCALE_EXPECTED_UPDATES

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        gain: torch.nn.Module | None,
        train_loader: Any,
        validation_loader: Any,
        device: torch.device | str,
        num_freq_selected: int,
        phase_sign: float,
        forward_operator_name: str,
        compute_dtype: torch.dtype,
        data_format: str,
        op_kwargs: Mapping[str, object],
        occlusion: Mapping[str, object] | None,
        expected_train_ids: Sequence[int],
        expected_validation_ids: Sequence[int],
        checkpoint_path: str | os.PathLike[str],
    ) -> None:
        if forward_operator_name != "range" or data_format != "npz":
            raise ValueError("adaptive full-scale observer requires sealed NPZ range training")
        if float(phase_sign) != -1.0 or op_kwargs.get("range_model") != "sum2":
            raise ValueError("adaptive full-scale observer requires phase -1 and sum2 spreading")
        if int(num_freq_selected) != 600:
            raise ValueError("adaptive full-scale observer requires all 600 frequency bins")
        if len(train_loader.dataset) != self.num_train:
            raise ValueError(f"adaptive full-scale observer requires {self.num_train} training views")
        if len(validation_loader.dataset) != FULLSCALE_NUM_VALIDATION:
            raise ValueError("adaptive full-scale observer requires 1000 validation views")
        if list(train_loader.dataset.indices) != list(expected_train_ids):
            raise ValueError("adaptive full-scale train IDs disagree with the sealed manifest")
        if list(validation_loader.dataset.indices) != list(expected_validation_ids):
            raise ValueError("adaptive full-scale validation IDs disagree with the sealed manifest")
        if not isinstance(model, torch.nn.Module) or not hasattr(model, "active_mask"):
            raise ValueError("adaptive full-scale observer requires an adaptive point scene")
        if int(model.active_mask.numel()) != FULLSCALE_MAX_POINTS:
            raise ValueError("adaptive full-scale scene capacity disagrees with the frozen candidate")
        if int(model.active_mask.sum().item()) != FULLSCALE_INITIAL_ACTIVE:
            raise ValueError("adaptive full-scale scene initial active count disagrees with G48")
        if int(model.max_degree) != FULLSCALE_MAX_DEGREE:
            raise ValueError("adaptive full-scale SH cap disagrees with the frozen candidate")

        self.model = model
        self.optimizer = optimizer
        self.gain = gain
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.device = torch.device(device)
        self.num_freq_selected = int(num_freq_selected)
        self.phase_sign = float(phase_sign)
        self.compute_dtype = compute_dtype
        self.op_kwargs = dict(op_kwargs)
        self.occlusion = occlusion
        self.checkpoint_path = str(Path(checkpoint_path).absolute())
        self.probe_ids = [int(value) for value in expected_train_ids[:FULLSCALE_EVENT_PROBE_COUNT]]
        self._probe_batches = [train_loader.dataset[index] for index in range(FULLSCALE_EVENT_PROBE_COUNT)]
        self.records: list[dict[str, Any]] = []
        self._pending: dict[str, Any] | None = None
        self.optimizer_update_count = 0
        self.nonfinite_grad_norm_count = 0
        self.first_optimizer_step_seconds: float | None = None
        self.last_optimizer_step_seconds: float | None = None
        self._start_time = time.monotonic()
        self._elapsed_before_resume = 0.0
        self._finished = False
        self._attempt_count = 1
        self._attempt_peak_torch_allocated_bytes = 0
        self._attempt_peak_torch_reserved_bytes = 0
        self._attempt_peak_process_max_rss_kib: int | None = None
        self._cumulative_peak_torch_allocated_bytes = 0
        self._cumulative_peak_torch_reserved_bytes = 0
        self._cumulative_peak_process_max_rss_kib: int | None = None
        # Do not reset CUDA's peak counters here: model construction and the
        # initial backprojection/gain allocation happened before this observer
        # was installed and are part of the run's resource envelope.
        self._sample_runtime_peaks()

    def _sample_runtime_peaks(self) -> None:
        """Retain initialization-inclusive peaks for this and resumed attempts."""

        if self.device.type == "cuda":
            allocated = int(torch.cuda.max_memory_allocated(self.device))
            reserved = int(torch.cuda.max_memory_reserved(self.device))
            self._attempt_peak_torch_allocated_bytes = max(
                self._attempt_peak_torch_allocated_bytes, allocated
            )
            self._attempt_peak_torch_reserved_bytes = max(
                self._attempt_peak_torch_reserved_bytes, reserved
            )
            self._cumulative_peak_torch_allocated_bytes = max(
                self._cumulative_peak_torch_allocated_bytes,
                self._attempt_peak_torch_allocated_bytes,
            )
            self._cumulative_peak_torch_reserved_bytes = max(
                self._cumulative_peak_torch_reserved_bytes,
                self._attempt_peak_torch_reserved_bytes,
            )
        if _resource is not None:
            rss = int(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss)
            if rss > 0:
                self._attempt_peak_process_max_rss_kib = _optional_max(
                    self._attempt_peak_process_max_rss_kib, rss
                )
                self._cumulative_peak_process_max_rss_kib = _optional_max(
                    self._cumulative_peak_process_max_rss_kib, rss
                )

    def _scene_summary(self) -> dict[str, Any]:
        active = self.model.active_mask
        positions = self.model.positions()[active]
        orders = self.model.order[active]
        levels = self.model.level[active]
        support_ok = bool(
            (positions >= self.model.support_min - 1.0e-6).all()
            and (positions <= self.model.support_max + 1.0e-6).all()
        )
        return {
            "active_points": int(active.sum().item()),
            "allocated_slots": int(active.numel()),
            "active_parameter_scalars": int(self.model.active_parameter_count()),
            "allocated_parameter_scalars": int(self.model.allocated_parameter_count()),
            "order_min": int(orders.min().item()) if orders.numel() else None,
            "order_max": int(orders.max().item()) if orders.numel() else None,
            "level_min": int(levels.min().item()) if levels.numel() else None,
            "level_max": int(levels.max().item()) if levels.numel() else None,
            "support_bounds_ok": support_ok,
            "degree_bounds_ok": bool(
                (orders >= 0).all() and (orders <= self.model.max_degree).all()
            ),
            "support_min_m": [float(value) for value in self.model.support_min.detach().cpu().tolist()],
            "support_max_m": [float(value) for value in self.model.support_max.detach().cpu().tolist()],
        }

    def _render_item(self, item: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Render one authorized train item through the full coherent operator."""

        from train import apply_occlusion, cc, get_kvector, reshape_measured_cubes
        from rift.range_operator import range_forward_operator

        (
            freqs_tensor,
            dphi_tensor,
            dtheta_tensor,
            magnitude_tensor,
            phase_tensor,
            rx_pos,
            tx_pos,
        ) = item
        magnitude_cube, phase_cube = reshape_measured_cubes(
            magnitude_tensor.unsqueeze(0), phase_tensor.unsqueeze(0), self.device, 16, 16
        )
        freqs_tensor = freqs_tensor.to(self.device)
        if int(freqs_tensor.numel()) != self.num_freq_selected:
            raise ValueError("adaptive full-scale observer requires the full 600-bin spectrum")
        freq_indices = torch.arange(freqs_tensor.shape[0], device=self.device)
        dtheta = dtheta_tensor.unsqueeze(0).to(self.device)
        dphi = dphi_tensor.unsqueeze(0).to(self.device)
        scatterer_pos, scatterer_weights = self.model.active_scatterers(dtheta, dphi)
        weights_view = apply_occlusion(
            self.model, scatterer_weights, rx_pos.to(self.device), tx_pos.to(self.device), self.occlusion
        )
        prediction = range_forward_operator(
            freqs_tensor,
            get_kvector(freqs_tensor, cc),
            rx_pos.to(self.device),
            tx_pos.to(self.device),
            scatterer_pos,
            weights_view,
            phase_sign=self.phase_sign,
            freq_indices=freq_indices,
            compute_dtype=self.compute_dtype,
            **self.op_kwargs,
        )
        if self.gain is not None:
            prediction = self.gain(prediction)
        # Materialize the target only for the already-authorized train probe.
        # No validation/test/unused response is opened by this observer.
        _ = magnitude_cube, phase_cube
        return prediction

    def _event_predictions(self) -> list[torch.Tensor]:
        was_training = self.model.training
        gain_training = self.gain.training if self.gain is not None else None
        self.model.eval()
        if self.gain is not None:
            self.gain.eval()
        try:
            with torch.no_grad():
                return [self._render_item(item).detach().clone() for item in self._probe_batches]
        finally:
            self.model.train(was_training)
            if self.gain is not None and gain_training is not None:
                self.gain.train(gain_training)

    def _optimizer_state_finite(self) -> bool:
        for state in self.optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value) and not _finite_tensor(value):
                    return False
        return True

    @staticmethod
    def _band_norm(model: torch.nn.Module, rows: torch.Tensor, degrees: torch.Tensor) -> float:
        if rows.numel() == 0:
            return 0.0
        row_weights_re = model.w_re[rows]
        row_weights_im = model.w_im[rows]
        band_mask = model.basis_degree[None, :] == degrees[:, None]
        value = ((row_weights_re.square() + row_weights_im.square()) * band_mask).sum()
        return float(value.sqrt().item())

    def on_adaptive_event(self, phase: str, **context: Any) -> None:
        event = int(context["event"])
        if phase == "before":
            snapshot = context["snapshot"]
            if not isinstance(snapshot, Mapping):
                raise ValueError("adaptive full-scale observer received a malformed snapshot")
            spatial = self.model._select_refinement_indices(
                snapshot["spatial_score"], snapshot["spatial_eligible"], FULLSCALE_SPATIAL_FRACTION
            )
            angular = self.model._select_refinement_indices(
                snapshot["angular_score"], snapshot["angular_eligible"], FULLSCALE_ANGULAR_FRACTION
            )
            active_before = self.model.active_mask.detach().clone()
            active_count = int(active_before.sum().item())
            free_parent_budget = int((~active_before).sum().item()) // 7
            active_parent_budget = max(FULLSCALE_MAX_POINTS - active_count, 0) // 7
            spatial = spatial[:min(free_parent_budget, active_parent_budget)]
            record: dict[str, Any] = {
                "event": event,
                "epoch": int(context["epoch"]),
                "logical_optimizer_updates": int(context["logical_optimizer_updates"]),
                "active_before": active_count,
                "allocated_slots": int(active_before.numel()),
                "eligible_spatial_count": int(snapshot["spatial_eligible"].sum().item()),
                "eligible_angular_count": int(snapshot["angular_eligible"].sum().item()),
                "selected_spatial_count": int(spatial.numel()),
                "selected_angular_count": int(angular.numel()),
                "spatial_score_summary": _score_summary(
                    snapshot["spatial_score"], snapshot["spatial_eligible"]
                ),
                "angular_score_summary": _score_summary(
                    snapshot["angular_score"], snapshot["angular_eligible"]
                ),
                "scene_before": self._scene_summary(),
            }
            self._pending = {
                "event": event,
                "record_index": len(self.records),
                "record": record,
                "before_predictions": self._event_predictions(),
                "active_before": active_before,
                "selected_spatial": spatial.detach().clone(),
                "selected_angular": angular.detach().clone(),
                "selected_angular_degrees": (
                    self.model.order[angular].detach().clone() + 1
                    if angular.numel() else torch.empty(0, dtype=torch.long, device=self.model.order.device)
                ),
            }
            self.records.append(record)
            return

        if phase != "after":
            raise ValueError(f"unknown adaptive full-scale observer phase {phase!r}")
        if self._pending is None or int(self._pending["event"]) != event:
            raise RuntimeError("adaptive full-scale observer received an unmatched after-event")
        pending = self._pending
        record = pending["record"]
        before_predictions: list[torch.Tensor] = pending["before_predictions"]
        after_predictions = self._event_predictions()
        delta_sq = 0.0
        reference_sq = 0.0
        max_abs = 0.0
        finite = len(before_predictions) == len(after_predictions)
        if finite:
            for before, after in zip(before_predictions, after_predictions):
                finite = finite and bool(torch.isfinite(before).all()) and bool(torch.isfinite(after).all())
                difference = after - before
                delta_sq += float(difference.abs().square().sum().double().item())
                reference_sq += float(before.abs().square().sum().double().item())
                max_abs = max(max_abs, float(difference.abs().max().item()))
        relative_l2 = math.sqrt(delta_sq / reference_sq) if finite and reference_sq > 0.0 else None
        active_before = pending["active_before"]
        child_indices = ((~active_before) & self.model.active_mask).nonzero(as_tuple=True)[0]
        selected_spatial = pending["selected_spatial"]
        selected_angular = pending["selected_angular"]
        expected_angular_degrees = pending["selected_angular_degrees"]
        actual_unlocked = selected_angular[
            self.model.order[selected_angular] >= expected_angular_degrees
        ] if selected_angular.numel() else selected_angular
        n_split = int(context["n_split"])
        n_grown = int(context["n_grown"])
        record.update({
            "n_split": n_split,
            "n_grown": n_grown,
            "active_after": int(context["n_active"]),
            "children_created": int(child_indices.numel()),
            "prediction_finite": bool(finite),
            "prediction_difference_relative_l2": relative_l2,
            "prediction_difference_max_abs": float(max_abs) if finite else None,
            "prediction_preserved": bool(
                finite and max_abs <= FULLSCALE_PREDICTION_PRESERVATION_ATOL
            ),
            "scene_after": self._scene_summary(),
            "model_parameters_finite": all(_finite_tensor(value) for value in self.model.parameters()),
            "optimizer_state_finite": self._optimizer_state_finite(),
            "spatial_selection_matches_action": n_split <= int(selected_spatial.numel()),
            "angular_selection_matches_action": n_grown <= int(actual_unlocked.numel()),
        })
        if n_split or n_grown:
            self._pending = {
                "event": event,
                "record_index": len(self.records) - 1,
                "record": record,
                "children": child_indices.detach().clone(),
                "unlocked": actual_unlocked.detach().clone(),
                "unlocked_degrees": (
                    self.model.order[actual_unlocked].detach().clone()
                    if actual_unlocked.numel() else torch.empty(
                        0, dtype=torch.long, device=self.model.order.device
                    )
                ),
                "checked_after_update": False,
            }
        else:
            self._pending = None

    def on_optimizer_step(self, **context: Any) -> None:
        self._sample_runtime_peaks()
        self.optimizer_update_count += 1
        seconds = float(context["seconds"])
        self.first_optimizer_step_seconds = (
            seconds if self.first_optimizer_step_seconds is None
            else self.first_optimizer_step_seconds
        )
        self.last_optimizer_step_seconds = seconds
        grad_norm = float(context["grad_norm"])
        if not math.isfinite(grad_norm):
            self.nonfinite_grad_norm_count += 1

        if self._pending is None or self._pending.get("checked_after_update"):
            return
        pending = self._pending
        children = pending["children"]
        unlocked = pending["unlocked"]
        child_norm = 0.0
        if children.numel():
            child_norm = float(torch.sqrt(
                self.model.w_re[children].square().sum()
                + self.model.w_im[children].square().sum()
            ).item())
        band_norm = self._band_norm(self.model, unlocked, pending["unlocked_degrees"])
        record = pending["record"]
        record.update({
            "first_post_event_optimizer_update": int(context["logical_optimizer_updates"]),
            "post_event_child_coefficient_l2": child_norm,
            "post_event_unlocked_band_l2": band_norm,
            "post_event_child_updated": bool(child_norm > 0.0) if children.numel() else True,
            "post_event_band_updated": bool(band_norm > 0.0) if unlocked.numel() else True,
        })
        self._pending = None

    def finish_training(self) -> None:
        self._sample_runtime_peaks()
        if self._pending is not None and self._pending.get("checked_after_update"):
            self._pending = None
        self._finished = True

    def checkpoint_state(self) -> dict[str, Any]:
        self._sample_runtime_peaks()
        pending: dict[str, Any] | None = None
        if self._pending is not None and "children" in self._pending:
            pending = {
                "event": int(self._pending["event"]),
                "record_index": int(self._pending["record_index"]),
                "children": [int(value) for value in self._pending["children"].detach().cpu().tolist()],
                "unlocked": [int(value) for value in self._pending["unlocked"].detach().cpu().tolist()],
                "unlocked_degrees": [
                    int(value) for value in self._pending["unlocked_degrees"].detach().cpu().tolist()
                ],
                "checked_after_update": bool(self._pending.get("checked_after_update", False)),
            }
        return {
            "schema": self.schema,
            "version": 2,
            "probe_source_view_ids": list(self.probe_ids),
            "records": copy.deepcopy(self.records),
            "pending": pending,
            "optimizer_update_count": int(self.optimizer_update_count),
            "nonfinite_grad_norm_count": int(self.nonfinite_grad_norm_count),
            "first_optimizer_step_seconds": self.first_optimizer_step_seconds,
            "last_optimizer_step_seconds": self.last_optimizer_step_seconds,
            "attempt_count": int(self._attempt_count),
            "memory_measurement_scope": (
                "initialization_inclusive_per_attempt; cumulative maxima restored "
                "from observer checkpoints on clean resume"
            ),
            "attempt_peak_torch_allocated_bytes": int(self._attempt_peak_torch_allocated_bytes),
            "attempt_peak_torch_reserved_bytes": int(self._attempt_peak_torch_reserved_bytes),
            "attempt_peak_process_max_rss_kib": self._attempt_peak_process_max_rss_kib,
            "cumulative_peak_torch_allocated_bytes": int(
                self._cumulative_peak_torch_allocated_bytes
            ),
            "cumulative_peak_torch_reserved_bytes": int(
                self._cumulative_peak_torch_reserved_bytes
            ),
            "cumulative_peak_process_max_rss_kib": self._cumulative_peak_process_max_rss_kib,
            "last_scene_summary": self._scene_summary(),
            "elapsed_seconds": float(
                self._elapsed_before_resume + (time.monotonic() - self._start_time)
            ),
            "finished": bool(self._finished),
        }

    def restore_checkpoint_state(self, state: Mapping[str, object]) -> None:
        if state.get("schema") != self.schema or state.get("version") != 2:
            raise ValueError("resume checkpoint lacks adaptive full-scale observer state")
        if list(state.get("probe_source_view_ids", ())) != self.probe_ids:
            raise ValueError("resume checkpoint observer probes disagree with this run")
        if state.get("finished") is True:
            raise ValueError("a completed adaptive full-scale run cannot resume")
        records = state.get("records")
        if not isinstance(records, list):
            raise ValueError("resume checkpoint adaptive full-scale records are malformed")
        elapsed = state.get("elapsed_seconds")
        if not isinstance(elapsed, (int, float)) or not math.isfinite(float(elapsed)) or float(elapsed) < 0.0:
            raise ValueError("resume checkpoint adaptive full-scale elapsed time is invalid")
        self.records = copy.deepcopy(records)
        self.optimizer_update_count = int(state.get("optimizer_update_count", 0))
        self.nonfinite_grad_norm_count = int(state.get("nonfinite_grad_norm_count", 0))
        self.first_optimizer_step_seconds = state.get("first_optimizer_step_seconds")
        self.last_optimizer_step_seconds = state.get("last_optimizer_step_seconds")
        previous_attempt_count = int(state.get("attempt_count", 1))
        if previous_attempt_count < 1:
            raise ValueError("resume checkpoint adaptive full-scale attempt count is invalid")
        self._attempt_count = previous_attempt_count + 1
        self._cumulative_peak_torch_allocated_bytes = max(
            self._cumulative_peak_torch_allocated_bytes,
            int(state.get("cumulative_peak_torch_allocated_bytes", 0)),
        )
        self._cumulative_peak_torch_reserved_bytes = max(
            self._cumulative_peak_torch_reserved_bytes,
            int(state.get("cumulative_peak_torch_reserved_bytes", 0)),
        )
        previous_rss = state.get("cumulative_peak_process_max_rss_kib")
        if previous_rss is not None:
            if isinstance(previous_rss, bool) or not isinstance(previous_rss, int) or previous_rss <= 0:
                raise ValueError("resume checkpoint adaptive full-scale RSS measurement is invalid")
            self._cumulative_peak_process_max_rss_kib = _optional_max(
                self._cumulative_peak_process_max_rss_kib, int(previous_rss)
            )
        self._elapsed_before_resume = float(elapsed)
        self._start_time = time.monotonic()
        self._pending = None
        pending = state.get("pending")
        if pending is not None:
            if not isinstance(pending, Mapping):
                raise ValueError("resume checkpoint adaptive full-scale pending state is malformed")
            try:
                event = int(pending["event"])
                record_index = int(pending["record_index"])
                children = torch.as_tensor(pending["children"], dtype=torch.long, device=self.model.w_re.device)
                unlocked = torch.as_tensor(pending["unlocked"], dtype=torch.long, device=self.model.w_re.device)
                degrees = torch.as_tensor(
                    pending["unlocked_degrees"], dtype=torch.long, device=self.model.w_re.device
                )
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                raise ValueError("resume checkpoint adaptive full-scale pending state is malformed") from exc
            if not (0 <= record_index < len(self.records)):
                raise ValueError("resume checkpoint adaptive full-scale pending record index is invalid")
            if len(unlocked) != len(degrees):
                raise ValueError("resume checkpoint adaptive full-scale pending band state is inconsistent")
            self._pending = {
                "event": event,
                "record_index": record_index,
                "record": self.records[record_index],
                "children": children,
                "unlocked": unlocked,
                "unlocked_degrees": degrees,
                "checked_after_update": bool(pending.get("checked_after_update", False)),
            }

    @classmethod
    def report_from_checkpoint_state(
        cls, state: Mapping[str, object], checkpoint_path: str | os.PathLike[str]
    ) -> dict[str, Any]:
        """Rebuild the missing report from a completed checkpoint only.

        This path deliberately consumes observer state and checkpoint metadata
        without reconstructing a model or entering the trainer.  It is only for
        a clean interruption after the final checkpoint was published but before
        report/postflight publication.
        """

        if state.get("schema") != cls.schema or state.get("version") != 2:
            raise ValueError("completed checkpoint lacks the current adaptive full-scale observer state")
        if state.get("finished") is not True:
            raise ValueError("report-only recovery requires a completed observer checkpoint")
        records = state.get("records")
        final_scene = state.get("last_scene_summary")
        if not isinstance(records, list) or not isinstance(final_scene, Mapping):
            raise ValueError("completed checkpoint lacks report-only observer evidence")
        expected_events = FULLSCALE_EPOCHS // FULLSCALE_REFINE_EVERY
        event_preservation = bool(records) and all(
            bool(record.get("prediction_preserved")) for record in records
        )
        event_finite = bool(records) and all(
            bool(record.get("prediction_finite"))
            and bool(record.get("model_parameters_finite"))
            and bool(record.get("optimizer_state_finite"))
            and bool(record.get("scene_after", {}).get("support_bounds_ok"))
            and bool(record.get("scene_after", {}).get("degree_bounds_ok"))
            for record in records
        )
        optimizer_updates = int(state.get("optimizer_update_count", 0))
        nonfinite_grad_norms = int(state.get("nonfinite_grad_norm_count", 0))
        pending = state.get("pending")
        final_event = int(pending["event"]) if isinstance(pending, Mapping) else None
        checks = {
            "observer_timeline_finished": True,
            "expected_optimizer_update_count": optimizer_updates == cls.expected_updates,
            "finite_optimizer_step_norms": nonfinite_grad_norms == 0,
            "expected_refinement_event_count": len(records) == expected_events,
            "event_prediction_preservation": event_preservation,
            "finite_event_state_and_bounds": event_finite,
            "final_scene_bounds_and_caps": bool(
                final_scene.get("support_bounds_ok") and final_scene.get("degree_bounds_ok")
            ),
        }
        allocated = int(state.get("cumulative_peak_torch_allocated_bytes", 0))
        reserved = int(state.get("cumulative_peak_torch_reserved_bytes", 0))
        cumulative_rss = state.get("cumulative_peak_process_max_rss_kib")
        return {
            "schema": cls.schema,
            "version": 1,
            "engineering_status": "fullscale_candidate_not_production_clearance",
            "production_clearance": False,
            "observer_checkpoint_path": str(Path(checkpoint_path).absolute()),
            "probe_source_view_ids": list(state.get("probe_source_view_ids", ())),
            "expected_optimizer_updates": cls.expected_updates,
            "optimizer_update_count": optimizer_updates,
            "expected_refinement_events": expected_events,
            "refinement_event_count": len(records),
            "total_spatial_splits": sum(int(record.get("n_split", 0)) for record in records),
            "total_angular_unlocks": sum(int(record.get("n_grown", 0)) for record in records),
            "events": copy.deepcopy(records),
            "pending_post_event_check": final_event,
            "final_event_without_later_update": bool(final_event == expected_events),
            "final_scene": copy.deepcopy(dict(final_scene)),
            "checks": checks,
            "pass": bool(all(checks.values())),
            "training_quality_source": (
                "generic trainer per-attempt training_validation_losses_TIMESTAMP.csv; "
                "not a fixed-checkpoint evaluation"
            ),
            "wall_seconds_from_observer_start": float(state.get("elapsed_seconds", 0.0)),
            "first_optimizer_step_seconds": state.get("first_optimizer_step_seconds"),
            "last_optimizer_step_seconds": state.get("last_optimizer_step_seconds"),
            "peak_torch_allocated_bytes": allocated,
            "peak_torch_reserved_bytes": reserved,
            "process_max_rss_kib": cumulative_rss,
            "attempt_peak_torch_allocated_bytes": int(
                state.get("attempt_peak_torch_allocated_bytes", 0)
            ),
            "attempt_peak_torch_reserved_bytes": int(
                state.get("attempt_peak_torch_reserved_bytes", 0)
            ),
            "attempt_peak_process_max_rss_kib": state.get("attempt_peak_process_max_rss_kib"),
            "cumulative_peak_torch_allocated_bytes": allocated,
            "cumulative_peak_torch_reserved_bytes": reserved,
            "cumulative_peak_process_max_rss_kib": cumulative_rss,
            "memory_measurement": {
                "scope": state.get(
                    "memory_measurement_scope",
                    "initialization-inclusive cumulative peaks unavailable in checkpoint",
                ),
                "attempt_count": int(state.get("attempt_count", 1)),
                "cuda_peak_counters_reset_by_observer": False,
                "gnu_time_process_rss_scope": (
                    "launcher-supplied current attempt only; retained in postflight artifact"
                ),
                "whole_job_rss_scope": (
                    "optional launcher-supplied current attempt only; retained verbatim"
                ),
            },
        }

    def finalize(self) -> dict[str, Any]:
        self._sample_runtime_peaks()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self._sample_runtime_peaks()
        elapsed = self._elapsed_before_resume + (time.monotonic() - self._start_time)
        allocated = int(self._cumulative_peak_torch_allocated_bytes)
        reserved = int(self._cumulative_peak_torch_reserved_bytes)
        final_scene = self._scene_summary()
        event_preservation = bool(self.records) and all(
            bool(record.get("prediction_preserved")) for record in self.records
        )
        event_finite = bool(self.records) and all(
            bool(record.get("prediction_finite"))
            and bool(record.get("model_parameters_finite"))
            and bool(record.get("optimizer_state_finite"))
            and bool(record.get("scene_after", {}).get("support_bounds_ok"))
            and bool(record.get("scene_after", {}).get("degree_bounds_ok"))
            for record in self.records
        )
        total_splits = sum(int(record.get("n_split", 0)) for record in self.records)
        total_grown = sum(int(record.get("n_grown", 0)) for record in self.records)
        expected_events = FULLSCALE_EPOCHS // FULLSCALE_REFINE_EVERY
        final_event_without_later_update = (
            None if self._pending is None else int(self._pending["event"])
        )
        checks = {
            "observer_timeline_finished": bool(self._finished),
            "expected_optimizer_update_count": self.optimizer_update_count == self.expected_updates,
            "finite_optimizer_step_norms": self.nonfinite_grad_norm_count == 0,
            "expected_refinement_event_count": len(self.records) == expected_events,
            "event_prediction_preservation": event_preservation,
            "finite_event_state_and_bounds": event_finite,
            "final_scene_bounds_and_caps": bool(
                final_scene["support_bounds_ok"] and final_scene["degree_bounds_ok"]
            ),
        }
        return {
            "schema": self.schema,
            "version": 1,
            "engineering_status": "fullscale_candidate_not_production_clearance",
            "production_clearance": False,
            "observer_checkpoint_path": self.checkpoint_path,
            "probe_source_view_ids": list(self.probe_ids),
            "expected_optimizer_updates": self.expected_updates,
            "optimizer_update_count": int(self.optimizer_update_count),
            "expected_refinement_events": expected_events,
            "refinement_event_count": len(self.records),
            "total_spatial_splits": total_splits,
            "total_angular_unlocks": total_grown,
            "events": copy.deepcopy(self.records),
            "pending_post_event_check": (
                final_event_without_later_update
            ),
            "final_event_without_later_update": bool(
                final_event_without_later_update == expected_events
            ),
            "final_scene": final_scene,
            "checks": checks,
            "pass": bool(all(checks.values())),
            "training_quality_source": (
                "generic trainer per-attempt training_validation_losses_TIMESTAMP.csv; "
                "not a fixed-checkpoint evaluation"
            ),
            "wall_seconds_from_observer_start": float(elapsed),
            "first_optimizer_step_seconds": self.first_optimizer_step_seconds,
            "last_optimizer_step_seconds": self.last_optimizer_step_seconds,
            "peak_torch_allocated_bytes": allocated,
            "peak_torch_reserved_bytes": reserved,
            # Keep the historical field name as a cumulative alias while
            # exposing the attempt and cumulative scopes explicitly.
            "process_max_rss_kib": self._cumulative_peak_process_max_rss_kib,
            "attempt_peak_torch_allocated_bytes": int(
                self._attempt_peak_torch_allocated_bytes
            ),
            "attempt_peak_torch_reserved_bytes": int(
                self._attempt_peak_torch_reserved_bytes
            ),
            "attempt_peak_process_max_rss_kib": self._attempt_peak_process_max_rss_kib,
            "cumulative_peak_torch_allocated_bytes": allocated,
            "cumulative_peak_torch_reserved_bytes": reserved,
            "cumulative_peak_process_max_rss_kib": self._cumulative_peak_process_max_rss_kib,
            "memory_measurement": {
                "scope": (
                    "initialization_inclusive_per_attempt; cumulative maxima restored "
                    "from observer checkpoints on clean resume"
                ),
                "attempt_count": int(self._attempt_count),
                "cuda_peak_counters_reset_by_observer": False,
                "gnu_time_process_rss_scope": (
                    "launcher-supplied current attempt only; retained in postflight artifact"
                ),
                "whole_job_rss_scope": (
                    "optional launcher-supplied current attempt only; retained verbatim"
                ),
            },
        }


def write_fullscale_report(path: str | os.PathLike[str], payload: Mapping[str, object]) -> Path:
    """Atomically publish the full-scale observer report."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return destination
