"""Shared bounded SpINR smoke runtime, not a standalone training entrypoint.

The maintained CLI is train_spinr_style_smoke.py. It configures
the G96/Gauss2 numerical recipe before invoking this runtime. Historical midpoint
defaults remain only as the base recipe/checkpoint contract and test fixtures.

This is deliberately a separate, opt-in driver.  It leaves the canonical
3,200-train/1,000-validation SpINR-style recipe, cache, checkpoints, and
entrypoint untouched.  The bounded smoke keeps the same model, operator,
objective, optimizer, and production scheduler clock, but stops after sixty
logical B=4 updates on the ordered parent ``train[:16]`` rows.

It is engineering evidence only: it never reads parent test or unused
responses, never claims a production fit or comparison result, and never
changes quadrature after a diagnostic failure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from rift.config import cc
from rift.forward_operator import get_kvector
from rift.range_operator import range_forward_operator
from rift.spinr_style import (
    SPINR_STYLE_METHOD_ID,
    SPINR_STYLE_PARAMETER_COUNT,
    SPINR_STYLE_PHASE_SIGN,
    SPINR_STYLE_RANGE_MODEL,
    SPINR_STYLE_RECIPE_ID,
    SPINR_STYLE_SUPPORT_M,
    SpinrStyleINR,
    midpoint_grid,
    scale_field_to_renderer_weights,
    spinr_style_objective,
    validate_spinr_style_acquisition_identity,
)
from rift.spinr_style_b78716_smoke import (
    B78716SmokeWorklists,
    BoundedSealedRawComplexViews,
    SMOKE_FIT_VIEW_COUNT,
    SMOKE_VALIDATION_VIEW_COUNT,
    bounded_b78716_smoke_worklists,
    build_bounded_sealed_raw_complex_views,
)
from train import (
    _validate_saved_sealed_npz_protocol_contract,
    capture_rng_state,
    load_tensor_checkpoint,
    restore_rng_state,
)
from train_spinr_style import (
    CANONICAL_INIT_SCALE_COUNT,
    CANONICAL_MAX_EPOCHS,
    CANONICAL_SEED,
    CANONICAL_UPDATES_PER_EPOCH,
    CANONICAL_VIEW_BATCH,
    DEFAULT_B787_MANIFEST_PATH,
    DEFAULT_B787_NPZ_PATH,
    _atomic_json_save,
    _atomic_torch_save,
    _checkpoint_tree_equal,
    _disable_tf32,
    _enforce_memory_gates,
    _peak_host_rss_bytes,
    evaluate_neural_field_tiled,
    evaluate_role,
    logical_batch_update,
    preflight_b787_development_inputs,
    real_field_cotangent_from_response,
    replay_field_cotangent_tiled,
    response_cotangent,
)


SMALLFIT_CHECKPOINT_FORMAT = "rift_spinr_style_b78716_smallfit_v1"
SMALLFIT_REPORT_SCHEMA = "rift_spinr_style_b78716_smallfit_report_v1"
SMALLFIT_RUN_NAME = "b78710k_spinr_style_inr_pm_smallfit16_v1"
SMALLFIT_ENGINEERING_STATUS = (
    "bounded_engineering_smoke_not_production_not_comparison_not_convergence_evidence"
)
SMALLFIT_MAX_UPDATES = 60
SMALLFIT_MILESTONES = (0, 4, 20, 60)
SMALLFIT_TRAIN_GRID_SIZE = 48
SMALLFIT_REFERENCE_GRID_SIZE = 96
SMALLFIT_NEURAL_POINT_TILE = 4096
SMALLFIT_RENDERER_POINT_TILE = 65536
SMALLFIT_PAIR_TILE = 16
SMALLFIT_HOST_RSS_LIMIT_GIB = 32.0
QUADRATURE_RELATIVE_LIMIT = 0.01
# The source signal and directional derivative use physical units.  This is a
# named engineering convention for the rare zero-reference case, rather than
# silently treating 0/0 as a relative pass.
QUADRATURE_NEAR_ZERO_ABSOLUTE_TOLERANCE = 1.0e-12
SMALLFIT_ACQUISITION_TX = 16
SMALLFIT_ACQUISITION_RX = 16
SMALLFIT_FREQUENCY_BINS = 600
SMALLFIT_QUADRATURE_SIGNAL_ELEMENTS = (
    CANONICAL_VIEW_BATCH
    * SMALLFIT_ACQUISITION_TX
    * SMALLFIT_ACQUISITION_RX
    * SMALLFIT_FREQUENCY_BINS
)
# The streamed raw normalizer is calculated deterministically on the host,
# while the first-32 scale calculation is replayed with FP64 renderer math on
# an allocated GPU.  This narrow tolerance permits a legitimate resume on a
# different compatible GPU while rejecting a changed archive, model seed, or
# gain equation.
FROZEN_NORMALIZATION_RELATIVE_TOLERANCE = 1.0e-8
FROZEN_NORMALIZATION_ABSOLUTE_TOLERANCE = 1.0e-12
_CHECKPOINT_LATEST = "checkpoint_latest.pth.tar"
_CHECKPOINT_FINAL = "checkpoint_final.pth.tar"
_METRICS_HISTORY = "metrics_history.json"
_REPORT_NAME = "smallfit_report.json"


def _finite_positive(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return number


def _finite_nonnegative(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return number


def _require_frozen_positive_scalar_match(
    actual: object, expected: object, label: str
) -> float:
    """Require a persisted positive scalar to agree with its frozen replay."""

    actual_number = _finite_positive(actual, f"saved {label}")
    expected_number = _finite_positive(expected, f"recomputed {label}")
    if not math.isclose(
        actual_number,
        expected_number,
        rel_tol=FROZEN_NORMALIZATION_RELATIVE_TOLERANCE,
        abs_tol=FROZEN_NORMALIZATION_ABSOLUTE_TOLERANCE,
    ):
        raise ValueError(f"bounded resume {label} disagrees with its frozen replay")
    return actual_number


def _validate_initial_scale_equation(normalization: Mapping[str, object]) -> None:
    """Require the persisted gain to be exactly the reviewed first-32 rule."""

    observed_energy = _finite_positive(
        normalization.get("initial_scale_observed_energy"), "saved observed scale energy"
    )
    predicted_energy = _finite_positive(
        normalization.get("initial_scale_predicted_energy"), "saved predicted scale energy"
    )
    expected_scale = 0.1 * math.sqrt(observed_energy / predicted_energy)
    _require_frozen_positive_scalar_match(
        normalization.get("initial_output_scale"), expected_scale, "initial output scale"
    )


def _absolute_unresolved(path: str | os.PathLike[str]) -> Path:
    """Return a lexical absolute path without resolving a possible symlink."""

    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _reject_symlink(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link")


def _same_resolved_path(left: str | os.PathLike[str], right: str | os.PathLike[str]) -> bool:
    return os.path.realpath(os.path.abspath(os.fspath(left))) == os.path.realpath(
        os.path.abspath(os.fspath(right))
    )


def smallfit_recipe_identity() -> dict[str, object]:
    """Return the immutable scientific and engineering semantics for resume.

    This is ordinary explicit checkpoint metadata, not a source/data hash or
    coordination mechanism.  Tile values are fixed here because this bounded
    gate has one reviewed resource envelope.
    """

    return {
        "method_id": SPINR_STYLE_METHOD_ID,
        "production_recipe_id": SPINR_STYLE_RECIPE_ID,
        "smallfit_recipe_id": SMALLFIT_RUN_NAME,
        "scope": SMALLFIT_ENGINEERING_STATUS,
        "network": {
            "input_features": 39,
            "hidden_layers": 6,
            "hidden_width": 840,
            "signed_real_output": True,
            "trainable_real_parameters": SPINR_STYLE_PARAMETER_COUNT,
            "support_m": SPINR_STYLE_SUPPORT_M,
            "learned_gain": False,
        },
        "operator": {
            "training_grid_size": SMALLFIT_TRAIN_GRID_SIZE,
            "reference_grid_size": SMALLFIT_REFERENCE_GRID_SIZE,
            "phase_sign": SPINR_STYLE_PHASE_SIGN,
            "range_model": SPINR_STYLE_RANGE_MODEL,
            "physics_dtype": "float64_complex128",
            "network_dtype": "float32",
            "full_frequency_bins": 600,
            "full_pairs": "16x16",
        },
        "objective": {
            "name": "normalized_range_magnitude_plus_half_complex",
            "regularization": "none",
            "normalization": "streamed_parent_train_raw_mean_abs_squared",
        },
        "optimization": {
            "optimizer": "Adam",
            "lr": 1.0e-4,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 0.0,
            "global_gradient_clip_norm": 1.0,
            "production_scheduler": "CosineAnnealingLR",
            "production_scheduler_t_max_epochs": CANONICAL_MAX_EPOCHS,
            "production_scheduler_eta_min": 1.0e-5,
            "production_updates_per_epoch": CANONICAL_UPDATES_PER_EPOCH,
            "logical_view_batch": CANONICAL_VIEW_BATCH,
            "smallfit_updates": SMALLFIT_MAX_UPDATES,
            "smallfit_milestones": list(SMALLFIT_MILESTONES),
            "seed": CANONICAL_SEED,
        },
        "tiles": {
            "neural_point_tile": SMALLFIT_NEURAL_POINT_TILE,
            "renderer_point_tile": SMALLFIT_RENDERER_POINT_TILE,
            "pair_tile": SMALLFIT_PAIR_TILE,
        },
        "quadrature_diagnostic": {
            "selected_training_views": 4,
            "comparison": "G48_vs_G96_fixed_weights_signal_and_directional_gradient",
            "relative_limit": QUADRATURE_RELATIVE_LIMIT,
            "near_zero_absolute_tolerance": QUADRATURE_NEAR_ZERO_ABSOLUTE_TOLERANCE,
        },
    }


def smallfit_update_batches(
    fit_training_ids: Sequence[int], *, seed: int = CANONICAL_SEED
) -> tuple[tuple[int, ...], ...]:
    """Return fifteen deterministic shuffled B=4 cycles over ordered train[:16]."""

    ids = tuple(int(item) for item in fit_training_ids)
    if len(ids) != SMOKE_FIT_VIEW_COUNT or len(set(ids)) != SMOKE_FIT_VIEW_COUNT:
        raise ValueError("the bounded SpINR-style smoke requires exactly 16 distinct fit IDs")
    if len(ids) % CANONICAL_VIEW_BATCH:
        raise AssertionError("the frozen 16-view worklist must divide into B=4 batches")
    batches: list[tuple[int, ...]] = []
    cycles = SMALLFIT_MAX_UPDATES // (len(ids) // CANONICAL_VIEW_BATCH)
    if cycles != 15:
        raise AssertionError("the frozen small-fit budget must contain exactly fifteen 16-view cycles")
    for cycle in range(cycles):
        order = np.random.default_rng(int(seed) + cycle).permutation(np.asarray(ids, dtype=np.int64))
        for row in order.reshape(-1, CANONICAL_VIEW_BATCH):
            batches.append(tuple(int(item) for item in row))
    if len(batches) != SMALLFIT_MAX_UPDATES:
        raise AssertionError("bounded SpINR-style batch schedule has the wrong logical update count")
    return tuple(batches)


def _worklists_dict(worklists: B78716SmokeWorklists) -> dict[str, list[int]]:
    result = worklists.as_dict()
    if set(result) != {
        "fit_training_ids",
        "validation_ids",
        "normalization_training_ids",
        "initialization_training_ids",
    }:
        raise AssertionError("bounded SpINR-style worklists are incomplete")
    if len(result["validation_ids"]) != SMOKE_VALIDATION_VIEW_COUNT:
        raise AssertionError("bounded SpINR-style validation worklist drifted from the frozen 16 views")
    return result


def _batch_schedule_as_lists(batches: Sequence[Sequence[int]]) -> list[list[int]]:
    return [[int(item) for item in batch] for batch in batches]


def _validate_smallfit_checkpoint_header(
    checkpoint: Mapping[str, object], *, recipe: Mapping[str, object]
) -> None:
    if checkpoint.get("format") != SMALLFIT_CHECKPOINT_FORMAT:
        raise ValueError("resume checkpoint has the wrong bounded SpINR-style format")
    if checkpoint.get("run_name") != SMALLFIT_RUN_NAME:
        raise ValueError("resume checkpoint belongs to a different bounded run identity")
    if checkpoint.get("recipe") != dict(recipe):
        raise ValueError("resume checkpoint would change the bounded SpINR-style recipe")
    execution = checkpoint.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("resume checkpoint lacks execution state")
    completed = execution.get("completed_updates")
    if isinstance(completed, bool) or not isinstance(completed, int):
        raise ValueError("resume checkpoint has an invalid completed update count")
    if not 0 <= completed <= SMALLFIT_MAX_UPDATES:
        raise ValueError("resume checkpoint has an out-of-range completed update count")
    phase = execution.get("phase")
    if phase not in {"updates", "interrupted_clean", "pending_finalization"}:
        raise ValueError("only a recoverable bounded small-fit state may resume")
    resume_count = execution.get("resume_count", 0)
    if isinstance(resume_count, bool) or not isinstance(resume_count, int) or resume_count < 0:
        raise ValueError("resume checkpoint has an invalid clean-resume counter")
    clock = execution.get("production_clock")
    if not isinstance(clock, Mapping) or clock != {
        "updates_per_production_epoch": CANONICAL_UPDATES_PER_EPOCH,
        "completed_production_epochs": 0,
        "updates_into_current_production_epoch": completed,
    }:
        raise ValueError("resume checkpoint does not preserve the unadvanced 800-update production clock")
    if completed == SMALLFIT_MAX_UPDATES and phase not in {"interrupted_clean", "pending_finalization"}:
        raise ValueError("only an update-60 pending-finalization state may resume to write terminal evidence")
    if completed < SMALLFIT_MAX_UPDATES and phase == "pending_finalization":
        raise ValueError("pending finalization is valid only after the bounded 60th update")


def _validate_resume_after_preflight(
    checkpoint: Mapping[str, object],
    *,
    recipe: Mapping[str, object],
    sealed_contract: Mapping[str, object],
    acquisition_identity: Mapping[str, object],
    worklists: B78716SmokeWorklists,
    batches: Sequence[Sequence[int]],
    model_template: SpinrStyleINR,
) -> None:
    """Validate semantic recovery compatibility before any response payload opens."""

    _validate_smallfit_checkpoint_header(checkpoint, recipe=recipe)
    _validate_checkpoint_payload_after_preflight(
        checkpoint,
        sealed_contract=sealed_contract,
        acquisition_identity=acquisition_identity,
        worklists=worklists,
        batches=batches,
        model_template=model_template,
        label="resume checkpoint",
    )


def _validate_checkpoint_payload_after_preflight(
    checkpoint: Mapping[str, object],
    *,
    sealed_contract: Mapping[str, object],
    acquisition_identity: Mapping[str, object],
    worklists: B78716SmokeWorklists,
    batches: Sequence[Sequence[int]],
    model_template: SpinrStyleINR,
    label: str,
) -> None:
    """Validate all data-independent model, optimizer, and evidence semantics."""

    _validate_saved_sealed_npz_protocol_contract(
        checkpoint.get("sealed_npz_protocol_contract"), sealed_contract
    )
    saved_acquisition = checkpoint.get("acquisition_identity")
    if not isinstance(saved_acquisition, Mapping):
        raise ValueError(f"{label} lacks the B787 acquisition identity")
    validate_spinr_style_acquisition_identity(saved_acquisition, acquisition_identity)
    if checkpoint.get("worklists") != _worklists_dict(worklists):
        raise ValueError(f"{label} would change selected, normalization, or initialization rows")
    if checkpoint.get("fit_batch_schedule") != _batch_schedule_as_lists(batches):
        raise ValueError(f"{label} would change the frozen B=4 update schedule")
    normalization = checkpoint.get("normalization")
    if not isinstance(normalization, Mapping):
        raise ValueError(f"{label} lacks frozen normalization state")
    if normalization.get("initial_scale_ids") != list(worklists.initialization_training_ids):
        raise ValueError(f"{label} would change the original first-32 gain initialization rows")
    if normalization.get("normalization_source") != "streamed_all_3200_parent_train_rows":
        raise ValueError(f"{label} would change the all-3,200 parent-training normalization policy")
    _finite_positive(normalization.get("training_mean_raw_power"), "saved training mean raw power")
    _finite_positive(normalization.get("initial_output_scale"), "saved initial output scale")
    _finite_positive(normalization.get("initial_scale_observed_energy"), "saved observed scale energy")
    _finite_positive(normalization.get("initial_scale_predicted_energy"), "saved predicted scale energy")
    _validate_initial_scale_equation(normalization)
    history = checkpoint.get("history")
    updates = checkpoint.get("update_records")
    if not isinstance(history, list) or not isinstance(updates, list):
        raise ValueError(f"{label} lacks bounded metrics history")
    execution = checkpoint["execution"]
    assert isinstance(execution, Mapping)
    if len(updates) != int(execution["completed_updates"]):
        raise ValueError(f"{label} update records disagree with its update cursor")
    saved_scheduler = checkpoint.get("scheduler_state_dict")
    saved_optimizer = checkpoint.get("optimizer_state_dict")
    saved_model = checkpoint.get("model_state_dict")
    saved_rng = checkpoint.get("rng_state")
    if not isinstance(saved_scheduler, Mapping) or not isinstance(saved_optimizer, Mapping):
        raise ValueError(f"{label} lacks optimizer/scheduler state")
    if not isinstance(saved_model, Mapping) or not isinstance(saved_rng, Mapping):
        raise ValueError(f"{label} lacks model or RNG state")
    expected_scheduler_optimizer = _new_smallfit_adam(model_template)
    expected_scheduler = _new_smallfit_scheduler(expected_scheduler_optimizer)
    try:
        scheduler_matches = dict(saved_scheduler) == expected_scheduler.state_dict()
    except (RuntimeError, TypeError, ValueError):
        scheduler_matches = False
    if scheduler_matches is not True:
        raise ValueError(f"{label} changed or advanced the production cosine scheduler")
    _validate_model_state_dict(saved_model, model_template)
    _validate_adam_state(saved_optimizer, model_template, completed_updates=int(execution["completed_updates"]))
    _validate_bounded_evidence(
        history=history,
        update_records=updates,
        quadrature_records=checkpoint.get("quadrature_records"),
        completed_updates=int(execution["completed_updates"]),
        batches=batches,
        quadrature_source_ids=worklists.fit_training_ids[:CANONICAL_VIEW_BATCH],
    )
    _validate_restoreable_rng(saved_rng)


def _validate_model_state_dict(saved: Mapping[str, object], template: SpinrStyleINR) -> None:
    """Validate the exact fixed MLP tensor layout without opening radar data."""

    expected = template.state_dict()
    if set(saved) != set(expected):
        raise ValueError("bounded resume model tensors disagree with the fixed SpINR-style architecture")
    for key, reference in expected.items():
        value = saved.get(key)
        if not torch.is_tensor(value):
            raise ValueError(f"bounded resume model tensor {key} is missing or non-tensor")
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError(f"bounded resume model tensor {key} has incompatible layout")
        if not torch.isfinite(value).all():
            raise ValueError(f"bounded resume model tensor {key} is non-finite")


def _scalar_step(value: object, label: str) -> int:
    if torch.is_tensor(value):
        if value.numel() != 1 or not torch.isfinite(value).all():
            raise ValueError(f"{label} must be one finite scalar")
        number = float(value.detach().cpu().item())
    else:
        number = float(value)
    if not math.isfinite(number) or not number.is_integer() or number < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(number)


def _validate_adam_state(
    saved: Mapping[str, object], template: SpinrStyleINR, *, completed_updates: int
) -> None:
    """Require Adam membership, moments, and step count to match the cursor."""

    groups = saved.get("param_groups")
    state = saved.get("state")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(groups[0], Mapping):
        raise ValueError("bounded resume has incompatible Adam parameter groups")
    if not isinstance(state, Mapping):
        raise ValueError("bounded resume lacks Adam state")
    group = groups[0]
    parameter_tensors = tuple(template.parameters())
    expected_ids = list(range(len(parameter_tensors)))
    # Compare against the actual fixed constructor on this runtime rather than
    # enumerating a historical subset of Adam options.  PyTorch has added
    # behavior-changing flags (for example ``maximize``, ``capturable``,
    # ``foreach``, and ``fused``) across releases; loading even one unchecked
    # flag could alter the exact continuation.
    expected_optimizer = _new_smallfit_adam(template)
    # The fixed cosine constructor attaches ``initial_lr`` to Adam's parameter
    # group.  Check against that scheduler-attached form because it is exactly
    # what every saved bounded checkpoint contains.
    _new_smallfit_scheduler(expected_optimizer)
    expected_group = expected_optimizer.state_dict()["param_groups"][0]
    try:
        fixed_group_match = dict(group) == expected_group
    except (RuntimeError, TypeError, ValueError):
        fixed_group_match = False
    if fixed_group_match is not True:
        raise ValueError("bounded resume would change fixed Adam parameter-group semantics")
    if group.get("params") != expected_ids:
        raise ValueError("bounded resume Adam parameter membership disagrees with fixed MLP order")
    if completed_updates == 0:
        if state:
            raise ValueError("zero-update bounded resume must not contain Adam moments")
        return
    if set(state) != set(expected_ids):
        raise ValueError("bounded resume Adam moments do not cover every fixed MLP parameter")
    for parameter_id, parameter in enumerate(parameter_tensors):
        payload = state.get(parameter_id)
        if not isinstance(payload, Mapping):
            raise ValueError("bounded resume Adam parameter state is malformed")
        if _scalar_step(payload.get("step"), f"Adam step for parameter {parameter_id}") != completed_updates:
            raise ValueError("bounded resume Adam step does not match its logical update cursor")
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = payload.get(moment_name)
            if not torch.is_tensor(moment):
                raise ValueError(f"bounded resume Adam lacks {moment_name}")
            if moment.shape != parameter.shape or moment.dtype != parameter.dtype:
                raise ValueError(f"bounded resume Adam {moment_name} has incompatible layout")
            if not torch.isfinite(moment).all():
                raise ValueError(f"bounded resume Adam {moment_name} is non-finite")
        if "max_exp_avg_sq" in payload:
            raise ValueError("bounded resume Adam unexpectedly contains AMSGrad state")


def _validate_metric_mapping(metrics: object, label: str) -> None:
    if not isinstance(metrics, Mapping):
        raise ValueError(f"bounded resume {label} metrics are missing")
    if metrics.get("views") != SMOKE_FIT_VIEW_COUNT:
        raise ValueError(f"bounded resume {label}.views must remain the frozen 16-view role")
    for key in ("native_spectral_objective", "coherent_relative_mse", "coherent_relative_l2"):
        _finite_nonnegative(metrics.get(key), f"bounded resume {label}.{key}")


def _validate_zero_reference_metrics(metrics: object, label: str) -> None:
    """Check the exact same-domain zero-predictor identity, not just finiteness."""

    _validate_metric_mapping(metrics, label)
    assert isinstance(metrics, Mapping)
    for key in ("coherent_relative_mse", "coherent_relative_l2"):
        if float(metrics.get(key, float("nan"))) != 1.0:
            raise ValueError(f"bounded resume {label}.{key} must equal one for the zero predictor")


def _validate_memory_snapshot(memory: object, label: str) -> None:
    if not isinstance(memory, Mapping) or set(memory) != {
        "process_max_rss_bytes", "peak_torch_allocated_bytes", "peak_torch_reserved_bytes"
    }:
        raise ValueError(f"bounded resume {label} memory evidence is incomplete")
    for key, value in memory.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"bounded resume {label}.{key} is invalid")


def _validate_quadrature_decision(decision: object, *, label: str) -> bool:
    if not isinstance(decision, Mapping) or decision.get("label") != label:
        raise ValueError(f"bounded resume quadrature {label} decision is malformed")
    if decision.get("relative_limit") != QUADRATURE_RELATIVE_LIMIT:
        raise ValueError(f"bounded resume quadrature {label} changed the relative limit")
    if decision.get("near_zero_absolute_tolerance") != QUADRATURE_NEAR_ZERO_ABSOLUTE_TOLERANCE:
        raise ValueError(f"bounded resume quadrature {label} changed the near-zero convention")
    difference = _finite_nonnegative(decision.get("difference_rms"), f"bounded resume {label} difference")
    reference = _finite_nonnegative(decision.get("reference_rms"), f"bounded resume {label} reference")
    mode = decision.get("comparison_mode")
    passed = decision.get("pass")
    if not isinstance(passed, bool):
        raise ValueError(f"bounded resume quadrature {label} lacks a boolean gate")
    if mode == "near_zero_absolute":
        if decision.get("absolute_difference") != difference or reference > QUADRATURE_NEAR_ZERO_ABSOLUTE_TOLERANCE:
            raise ValueError(f"bounded resume quadrature {label} has inconsistent near-zero evidence")
        expected = difference <= QUADRATURE_NEAR_ZERO_ABSOLUTE_TOLERANCE
    elif mode == "relative":
        if reference <= QUADRATURE_NEAR_ZERO_ABSOLUTE_TOLERANCE:
            raise ValueError(f"bounded resume quadrature {label} used a relative gate near zero")
        relative = _finite_nonnegative(decision.get("relative_difference"), f"bounded resume {label} relative difference")
        if relative != difference / reference:
            raise ValueError(f"bounded resume quadrature {label} relative evidence is inconsistent")
        expected = relative <= QUADRATURE_RELATIVE_LIMIT
    else:
        raise ValueError(f"bounded resume quadrature {label} has an unknown comparison mode")
    if passed is not expected:
        raise ValueError(f"bounded resume quadrature {label} gate disagrees with its evidence")
    return passed


def _validate_quadrature_record(record: object, *, expected_source_ids: Sequence[int]) -> None:
    if not isinstance(record, Mapping):
        raise ValueError("bounded resume quadrature evidence is malformed")
    required = {
        "frozen_weights",
        "source_ids",
        "training_grid_size",
        "reference_grid_size",
        "g48",
        "g96",
        "signal_gate",
        "directional_gradient_gate",
        "gate_pass",
    }
    if set(record) != required or record.get("frozen_weights") is not True:
        raise ValueError("bounded resume quadrature record changed its frozen-weight protocol")
    source_ids = record.get("source_ids")
    if source_ids != [int(item) for item in expected_source_ids]:
        raise ValueError("bounded resume quadrature record changed its frozen first-four training IDs")
    if record.get("training_grid_size") != SMALLFIT_TRAIN_GRID_SIZE or record.get("reference_grid_size") != SMALLFIT_REFERENCE_GRID_SIZE:
        raise ValueError("bounded resume quadrature record changed G48/G96")
    observed_energies: list[float] = []
    directional_gradients: list[float] = []
    for grid_key, expected_grid in (("g48", SMALLFIT_TRAIN_GRID_SIZE), ("g96", SMALLFIT_REFERENCE_GRID_SIZE)):
        observation = record.get(grid_key)
        if not isinstance(observation, Mapping) or set(observation) != {
            "grid_size", "observed_energy", "signal_elements", "native_spectral_objective", "directional_gradient"
        }:
            raise ValueError(f"bounded resume quadrature {grid_key} observation is malformed")
        if observation.get("grid_size") != expected_grid:
            raise ValueError(f"bounded resume quadrature {grid_key} changed its grid")
        observed_energies.append(
            _finite_nonnegative(observation.get("observed_energy"), f"bounded resume {grid_key} observed energy")
        )
        if observation.get("signal_elements") != SMALLFIT_QUADRATURE_SIGNAL_ELEMENTS:
            raise ValueError(
                f"bounded resume quadrature {grid_key} signal count must retain all "
                "four views, 16x16 pairs, and 600 frequencies"
            )
        _finite_nonnegative(observation.get("native_spectral_objective"), f"bounded resume {grid_key} objective")
        directional = float(observation.get("directional_gradient"))
        if not math.isfinite(directional):
            raise ValueError(f"bounded resume quadrature {grid_key} directional gradient is invalid")
        directional_gradients.append(directional)
    if observed_energies[0] != observed_energies[1]:
        raise ValueError("bounded resume G48/G96 quadrature records do not use identical observed data")
    signal_gate = record.get("signal_gate")
    signal_pass = _validate_quadrature_decision(signal_gate, label="frequency_signal")
    assert isinstance(signal_gate, Mapping)
    expected_signal_reference = math.sqrt(
        observed_energies[1] / float(SMALLFIT_QUADRATURE_SIGNAL_ELEMENTS)
    )
    if signal_gate.get("reference_rms") != expected_signal_reference:
        raise ValueError(
            "bounded resume signal quadrature gate disagrees with its recorded full-domain observed energy"
        )
    directional_gate = record.get("directional_gradient_gate")
    gradient_pass = _validate_quadrature_decision(directional_gate, label="directional_gradient")
    assert isinstance(directional_gate, Mapping)
    expected_difference = abs(directional_gradients[0] - directional_gradients[1])
    expected_reference = abs(directional_gradients[1])
    if (
        directional_gate.get("difference_rms") != expected_difference
        or directional_gate.get("reference_rms") != expected_reference
    ):
        raise ValueError(
            "bounded resume directional quadrature gate disagrees with its recorded G48/G96 gradients"
        )
    if record.get("gate_pass") is not bool(signal_pass and gradient_pass):
        raise ValueError("bounded resume quadrature gate does not combine its two fixed decisions")


def _validate_bounded_evidence(
    *,
    history: Sequence[object],
    update_records: Sequence[object],
    quadrature_records: object,
    completed_updates: int,
    batches: Sequence[Sequence[int]],
    quadrature_source_ids: Sequence[int],
) -> None:
    """Check the committed bounded cursor, metrics, and fixed work order."""

    expected_milestones = tuple(item for item in SMALLFIT_MILESTONES if item <= completed_updates)
    if len(history) != len(expected_milestones):
        raise ValueError("bounded resume milestone history disagrees with its logical update cursor")
    for expected_update, record in zip(expected_milestones, history):
        if not isinstance(record, Mapping) or record.get("update") != expected_update:
            raise ValueError("bounded resume milestone records are missing or reordered")
        if record.get("production_clock") != _production_clock(expected_update):
            raise ValueError("bounded resume milestone changed the production update clock")
        _finite_nonnegative(record.get("elapsed_seconds"), f"bounded resume milestone {expected_update} elapsed seconds")
        if not math.isclose(float(record.get("learning_rate", float("nan"))), 1.0e-4, rel_tol=0.0, abs_tol=1.0e-15):
            raise ValueError("bounded resume milestone changed the unadvanced optimizer learning rate")
        _validate_memory_snapshot(record.get("memory"), f"milestone {expected_update}")
        _validate_metric_mapping(record.get("fixed_train"), "fixed_train")
        _validate_metric_mapping(record.get("held_out_validation"), "held_out_validation")
        zero = record.get("same_domain_zero_reference")
        if not isinstance(zero, Mapping):
            raise ValueError("bounded resume lacks same-domain zero-reference evidence")
        _validate_zero_reference_metrics(zero.get("fixed_train"), "zero.fixed_train")
        _validate_zero_reference_metrics(zero.get("held_out_validation"), "zero.held_out_validation")
    if len(update_records) != completed_updates:
        raise ValueError("bounded resume update records disagree with its logical update cursor")
    for index, record in enumerate(update_records, start=1):
        if not isinstance(record, Mapping) or record.get("update") != index:
            raise ValueError("bounded resume update records are missing or reordered")
        if record.get("source_ids") != [int(item) for item in batches[index - 1]]:
            raise ValueError("bounded resume update record changed its frozen B=4 source IDs")
        for key in ("native_batch_objective", "gradient_norm_before_clip", "parameter_delta_l2", "update_seconds"):
            _finite_nonnegative(record.get(key), f"bounded resume update {index}.{key}")
        if not isinstance(record.get("gradient_was_clipped"), bool):
            raise ValueError("bounded resume update record lacks its clipping indicator")
    if not isinstance(quadrature_records, Mapping):
        raise ValueError("bounded resume lacks quadrature records")
    expected_quadrature = {"0"} if completed_updates < SMALLFIT_MAX_UPDATES else {"0", "60"}
    if set(quadrature_records) != expected_quadrature:
        raise ValueError("bounded resume quadrature records disagree with committed milestones")
    for key in expected_quadrature:
        record = quadrature_records.get(key)
        _validate_quadrature_record(record, expected_source_ids=quadrature_source_ids)


def _validate_restoreable_rng(saved: Mapping[str, object]) -> None:
    """Prove complete RNG recovery before the expensive response stream."""

    current = capture_rng_state()
    try:
        restore_rng_state(saved, require_complete=True)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("bounded resume RNG payload is not completely restorable") from exc
    finally:
        # Keep this preflight observational: the actual recovery below restores
        # the saved state after the fixed model/optimizer objects are built.
        restore_rng_state(current, require_complete=True)


def _checkpoint_state(
    *,
    model: SpinrStyleINR,
    optimizer: Adam,
    scheduler: CosineAnnealingLR,
    recipe: Mapping[str, object],
    sealed_contract: Mapping[str, object],
    acquisition_identity: Mapping[str, object],
    worklists: B78716SmokeWorklists,
    batches: Sequence[Sequence[int]],
    normalization: Mapping[str, object],
    execution: Mapping[str, object],
    history: Sequence[Mapping[str, object]],
    update_records: Sequence[Mapping[str, object]],
    quadrature_records: Mapping[str, object],
) -> dict[str, object]:
    return {
        "format": SMALLFIT_CHECKPOINT_FORMAT,
        "run_name": SMALLFIT_RUN_NAME,
        "engineering_status": SMALLFIT_ENGINEERING_STATUS,
        "recipe": dict(recipe),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": capture_rng_state(),
        "sealed_npz_protocol_contract": dict(sealed_contract),
        "acquisition_identity": dict(acquisition_identity),
        "worklists": _worklists_dict(worklists),
        "fit_batch_schedule": _batch_schedule_as_lists(batches),
        "normalization": dict(normalization),
        "execution": dict(execution),
        "history": [dict(item) for item in history],
        "update_records": [dict(item) for item in update_records],
        "quadrature_records": dict(quadrature_records),
    }


def _save_latest_and_history(
    checkpoint_dir: Path,
    state: Mapping[str, object],
    history: Sequence[Mapping[str, object]],
) -> None:
    _atomic_torch_save(state, checkpoint_dir / _CHECKPOINT_LATEST)
    _atomic_json_save({"history": [dict(item) for item in history]}, checkpoint_dir / _METRICS_HISTORY)


def _production_clock(completed_updates: int) -> dict[str, int]:
    if not 0 <= int(completed_updates) <= SMALLFIT_MAX_UPDATES:
        raise ValueError("bounded production-clock update count is out of range")
    return {
        "updates_per_production_epoch": CANONICAL_UPDATES_PER_EPOCH,
        "completed_production_epochs": 0,
        "updates_into_current_production_epoch": int(completed_updates),
    }


def _execution_state(
    *, phase: str, completed_updates: int, elapsed_seconds: float, resume_count: int
) -> dict[str, object]:
    if phase not in {"updates", "interrupted_clean", "pending_finalization", "completed"}:
        raise ValueError("unknown bounded small-fit execution phase")
    return {
        "phase": phase,
        "completed_updates": int(completed_updates),
        "elapsed_seconds": _finite_nonnegative(elapsed_seconds, "elapsed seconds"),
        "resume_count": int(resume_count),
        "production_clock": _production_clock(int(completed_updates)),
    }


def _resume_segments_after_restore(*, saved_resume_count: int, completed_updates: int) -> int:
    """Count only resumed segments that can replay at least one update.

    A literal resume at the committed update-60 boundary can only finish a
    missing terminal artifact.  It must not mutate the immutable
    report/checkpoint recovery counter, because that counter is bound into a
    report-only or final-only atomic half created at the same boundary.
    """

    if isinstance(saved_resume_count, bool) or not isinstance(saved_resume_count, int) or saved_resume_count < 0:
        raise ValueError("saved bounded execution has an invalid resume count")
    if isinstance(completed_updates, bool) or not isinstance(completed_updates, int):
        raise ValueError("saved bounded execution has an invalid completed update count")
    if not 0 <= completed_updates <= SMALLFIT_MAX_UPDATES:
        raise ValueError("saved bounded execution has an out-of-range completed update count")
    return saved_resume_count + int(completed_updates < SMALLFIT_MAX_UPDATES)


def _new_smallfit_adam(model: SpinrStyleINR) -> Adam:
    """Build the one approved Adam configuration for training and recovery checks."""

    return Adam(
        model.parameters(),
        lr=1.0e-4,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
    )


def _new_smallfit_scheduler(optimizer: Adam) -> CosineAnnealingLR:
    """Attach the frozen, unadvanced production cosine schedule to Adam."""

    return CosineAnnealingLR(
        optimizer,
        T_max=CANONICAL_MAX_EPOCHS,
        eta_min=1.0e-5,
    )


def _parameter_snapshots(model: SpinrStyleINR) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in model.parameters()]


def _parameter_delta_l2(model: SpinrStyleINR, before: Sequence[torch.Tensor]) -> float:
    if len(before) != sum(1 for _ in model.parameters()):
        raise AssertionError("parameter snapshot membership changed")
    squared = 0.0
    for parameter, previous in zip(model.parameters(), before):
        if not torch.isfinite(parameter).all():
            raise FloatingPointError("SpINR-style parameter became non-finite")
        difference = parameter.detach() - previous
        squared += float(difference.double().square().sum().item())
    result = math.sqrt(squared)
    if not math.isfinite(result):
        raise FloatingPointError("SpINR-style parameter update norm is non-finite")
    return result


def _memory_measurements(device: torch.device) -> dict[str, int | None]:
    result: dict[str, int | None] = {
        "process_max_rss_bytes": _peak_host_rss_bytes(),
        "peak_torch_allocated_bytes": None,
        "peak_torch_reserved_bytes": None,
    }
    if device.type == "cuda":
        result["peak_torch_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        result["peak_torch_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    return result


def _zero_role_metrics(
    *,
    views: BoundedSealedRawComplexViews,
    role: str,
    source_ids: Sequence[int],
    training_mean_raw_power: float,
    device: torch.device,
) -> dict[str, float | int]:
    """Evaluate the exact same-domain zero-predictor reference."""

    allowed = set(views.role_ids(role))
    ids = tuple(int(item) for item in source_ids)
    if not ids or any(item not in allowed for item in ids):
        raise PermissionError("zero baseline must use only the declared bounded role")
    numerator = 0.0
    denominator = 0.0
    objective_sum = 0.0
    for source_id in ids:
        observed, _rx, _tx = views.tensor_view(source_id, device=device)
        zero = torch.zeros_like(observed)
        numerator += float(observed.abs().square().sum().item())
        denominator += float(observed.abs().square().sum().item())
        objective_sum += float(
            spinr_style_objective(
                zero, observed, training_mean_raw_power=training_mean_raw_power
            ).item()
        )
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise FloatingPointError("zero baseline encountered nonpositive observed energy")
    return {
        "views": len(ids),
        "coherent_relative_mse": numerator / denominator,
        "coherent_relative_l2": math.sqrt(numerator / denominator),
        "native_spectral_objective": objective_sum / len(ids),
    }


@torch.no_grad()
def _estimate_initial_output_scale(
    *,
    model: SpinrStyleINR,
    views: BoundedSealedRawComplexViews,
    points_m: torch.Tensor,
    cell_volume_m3: float,
    frequencies_hz: torch.Tensor,
    kvector: torch.Tensor,
    device: torch.device,
) -> tuple[float, tuple[int, ...], float, float]:
    """Use exactly the original parent train[:32] gain-initialization rows."""

    source_ids = views.initialization_ids()
    if len(source_ids) != CANONICAL_INIT_SCALE_COUNT:
        raise AssertionError("bounded initial scale must retain exactly parent train[:32]")
    field = evaluate_neural_field_tiled(
        model, points_m, neural_point_tile=SMALLFIT_NEURAL_POINT_TILE
    )
    weights = scale_field_to_renderer_weights(
        field, cell_volume_m3=cell_volume_m3, initial_output_scale=1.0
    )
    predicted_energy = 0.0
    observed_energy = 0.0
    for source_id in source_ids:
        observed, rx_pos, tx_pos = views.initialization_tensor_view(source_id, device=device)
        predicted = range_forward_operator(
            frequencies_hz,
            kvector,
            rx_pos,
            tx_pos,
            points_m,
            weights,
            phase_sign=SPINR_STYLE_PHASE_SIGN,
            pair_chunk=SMALLFIT_PAIR_TILE,
            point_chunk=SMALLFIT_RENDERER_POINT_TILE,
            compute_dtype=torch.float64,
            range_model=SPINR_STYLE_RANGE_MODEL,
        )
        predicted_energy += float(predicted.abs().square().sum().item())
        observed_energy += float(observed.abs().square().sum().item())
    _finite_positive(predicted_energy, "random-field predicted energy")
    _finite_positive(observed_energy, "initial-scale observed energy")
    scale = 0.1 * math.sqrt(observed_energy / predicted_energy)
    return (
        _finite_positive(scale, "initial output scale"),
        tuple(source_ids),
        observed_energy,
        predicted_energy,
    )


@torch.no_grad()
def _validate_resumed_normalization_against_materialized_b787(
    *,
    normalization: Mapping[str, object],
    model: SpinrStyleINR,
    views: BoundedSealedRawComplexViews,
    points_m: torch.Tensor,
    cell_volume_m3: float,
    frequencies_hz: torch.Tensor,
    kvector: torch.Tensor,
    device: torch.device,
) -> None:
    """Replay only frozen normalization/gain evidence before state restoration.

    The checkpoint header proves the requested protocol before any response
    opens.  Once the capability-limited adapter has materialized its allowed
    rows, a literal resume additionally proves that the saved all-3,200 raw
    normalizer and first-32 gain values belong to this archive and the fixed
    seeded model.  No update, role evaluation, test access, or scheduler step
    occurs here.
    """

    _validate_initial_scale_equation(normalization)
    _require_frozen_positive_scalar_match(
        normalization.get("training_mean_raw_power"),
        views.raw_training_mean_power(),
        "all-3,200 training mean raw power",
    )
    scale, source_ids, observed_energy, predicted_energy = _estimate_initial_output_scale(
        model=model,
        views=views,
        points_m=points_m,
        cell_volume_m3=cell_volume_m3,
        frequencies_hz=frequencies_hz,
        kvector=kvector,
        device=device,
    )
    if list(source_ids) != normalization.get("initial_scale_ids"):
        raise ValueError("bounded resume first-32 gain replay changed its source rows")
    _require_frozen_positive_scalar_match(
        normalization.get("initial_scale_observed_energy"),
        observed_energy,
        "initial scale observed energy",
    )
    _require_frozen_positive_scalar_match(
        normalization.get("initial_scale_predicted_energy"),
        predicted_energy,
        "initial scale predicted energy",
    )
    _require_frozen_positive_scalar_match(
        normalization.get("initial_output_scale"), scale, "initial output scale"
    )


def _directional_derivative(model: SpinrStyleINR) -> float:
    """Project the fixed manual gradient on one deterministic unit direction."""

    dot = 0.0
    norm_squared = 0.0
    offset = 0
    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise FloatingPointError("quadrature diagnostic has a missing or non-finite parameter gradient")
        indices = torch.arange(
            offset,
            offset + parameter.numel(),
            device=parameter.device,
            dtype=torch.float64,
        )
        direction = torch.sin(indices + 1.0)
        gradient64 = gradient.detach().reshape(-1).to(dtype=torch.float64)
        dot += float((gradient64 * direction).sum().item())
        norm_squared += float(direction.square().sum().item())
        offset += parameter.numel()
    norm = math.sqrt(norm_squared)
    if not math.isfinite(dot) or not math.isfinite(norm) or norm <= 0.0:
        raise FloatingPointError("quadrature diagnostic direction is invalid")
    return dot / norm


def _quadrature_grid_observation(
    *,
    model: SpinrStyleINR,
    views: BoundedSealedRawComplexViews,
    source_ids: Sequence[int],
    grid_size: int,
    initial_output_scale: float,
    frequencies_hz: torch.Tensor,
    kvector: torch.Tensor,
    training_mean_raw_power: float,
    device: torch.device,
) -> dict[str, object]:
    """Freeze weights and observe signal plus manual-gradient behavior at one grid."""

    ids = tuple(int(item) for item in source_ids)
    if len(ids) != CANONICAL_VIEW_BATCH or any(item not in views.role_ids("train") for item in ids):
        raise PermissionError("quadrature diagnostics require exactly first four selected train views")
    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    try:
        points_m, cell_volume_m3 = midpoint_grid(grid_size, device=device, dtype=torch.float64)
        field = evaluate_neural_field_tiled(
            model, points_m, neural_point_tile=SMALLFIT_NEURAL_POINT_TILE
        )
        weights = scale_field_to_renderer_weights(
            field, cell_volume_m3=cell_volume_m3, initial_output_scale=initial_output_scale
        )
        field_cotangent = torch.zeros_like(field, dtype=torch.float64)
        predicted_signals: list[torch.Tensor] = []
        observed_energy = 0.0
        objective_sum = 0.0
        for source_id in ids:
            observed, rx_pos, tx_pos = views.tensor_view(source_id, device=device)
            with torch.no_grad():
                predicted = range_forward_operator(
                    frequencies_hz,
                    kvector,
                    rx_pos,
                    tx_pos,
                    points_m,
                    weights,
                    phase_sign=SPINR_STYLE_PHASE_SIGN,
                    pair_chunk=SMALLFIT_PAIR_TILE,
                    point_chunk=SMALLFIT_RENDERER_POINT_TILE,
                    compute_dtype=torch.float64,
                    range_model=SPINR_STYLE_RANGE_MODEL,
                )
            loss, response_gradient = response_cotangent(
                predicted, observed, training_mean_raw_power=training_mean_raw_power
            )
            objective_sum += float(loss.item())
            field_cotangent.add_(real_field_cotangent_from_response(
                response_cotangent_frequency=response_gradient / float(CANONICAL_VIEW_BATCH),
                frequencies_hz=frequencies_hz,
                kvector=kvector,
                rx_pos_m=rx_pos,
                tx_pos_m=tx_pos,
                points_m=points_m,
                cell_volume_m3=cell_volume_m3,
                initial_output_scale=initial_output_scale,
                renderer_point_tile=SMALLFIT_RENDERER_POINT_TILE,
                pair_tile=SMALLFIT_PAIR_TILE,
            ))
            predicted_signals.append(predicted.detach().cpu().contiguous())
            observed_energy += float(observed.abs().square().sum().item())
        replay_field_cotangent_tiled(
            model,
            points_m,
            field_cotangent,
            neural_point_tile=SMALLFIT_NEURAL_POINT_TILE,
        )
        directional = _directional_derivative(model)
        if not math.isfinite(observed_energy) or observed_energy < 0.0:
            raise FloatingPointError("quadrature diagnostic observed energy is invalid")
        signal_elements = sum(int(item.numel()) for item in predicted_signals)
        if signal_elements != SMALLFIT_QUADRATURE_SIGNAL_ELEMENTS:
            raise AssertionError(
                "quadrature diagnostic must retain four full 16x16, 600-frequency signals"
            )
        return {
            "grid_size": int(grid_size),
            "signal_frequency": predicted_signals,
            "observed_energy": observed_energy,
            "signal_elements": signal_elements,
            "native_spectral_objective": objective_sum / float(CANONICAL_VIEW_BATCH),
            "directional_gradient": directional,
        }
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)


def quadrature_difference_decision(
    *,
    difference_rms: float,
    reference_rms: float,
    label: str,
) -> dict[str, object]:
    """Apply the explicit relative-or-near-zero quadrature convention."""

    difference = _finite_nonnegative(difference_rms, f"{label} difference RMS")
    reference = _finite_nonnegative(reference_rms, f"{label} reference RMS")
    result: dict[str, object] = {
        "label": label,
        "relative_limit": QUADRATURE_RELATIVE_LIMIT,
        "near_zero_absolute_tolerance": QUADRATURE_NEAR_ZERO_ABSOLUTE_TOLERANCE,
        "difference_rms": difference,
        "reference_rms": reference,
    }
    if reference <= QUADRATURE_NEAR_ZERO_ABSOLUTE_TOLERANCE:
        result.update({
            "comparison_mode": "near_zero_absolute",
            "absolute_difference": difference,
            "pass": difference <= QUADRATURE_NEAR_ZERO_ABSOLUTE_TOLERANCE,
        })
    else:
        relative = difference / reference
        result.update({
            "comparison_mode": "relative",
            "relative_difference": relative,
            "pass": relative <= QUADRATURE_RELATIVE_LIMIT,
        })
    return result


def run_quadrature_diagnostic(
    *,
    model: SpinrStyleINR,
    views: BoundedSealedRawComplexViews,
    initial_output_scale: float,
    frequencies_hz: torch.Tensor,
    kvector: torch.Tensor,
    training_mean_raw_power: float,
    device: torch.device,
) -> dict[str, object]:
    """Compare frozen G48 and G96 signals/gradients on first train[:4]."""

    source_ids = tuple(views.role_ids("train")[:CANONICAL_VIEW_BATCH])
    low = _quadrature_grid_observation(
        model=model,
        views=views,
        source_ids=source_ids,
        grid_size=SMALLFIT_TRAIN_GRID_SIZE,
        initial_output_scale=initial_output_scale,
        frequencies_hz=frequencies_hz,
        kvector=kvector,
        training_mean_raw_power=training_mean_raw_power,
        device=device,
    )
    high = _quadrature_grid_observation(
        model=model,
        views=views,
        source_ids=source_ids,
        grid_size=SMALLFIT_REFERENCE_GRID_SIZE,
        initial_output_scale=initial_output_scale,
        frequencies_hz=frequencies_hz,
        kvector=kvector,
        training_mean_raw_power=training_mean_raw_power,
        device=device,
    )
    low_signals = low.pop("signal_frequency")
    high_signals = high.pop("signal_frequency")
    if not isinstance(low_signals, list) or not isinstance(high_signals, list):
        raise AssertionError("quadrature signal snapshots were not retained")
    if len(low_signals) != len(high_signals) or not low_signals:
        raise AssertionError("quadrature signal snapshots disagree on view count")
    difference_energy = 0.0
    for low_signal, high_signal in zip(low_signals, high_signals):
        if not (torch.is_tensor(low_signal) and torch.is_tensor(high_signal)):
            raise AssertionError("quadrature signal snapshots must be tensors")
        if low_signal.shape != high_signal.shape:
            raise AssertionError("quadrature signal snapshots have incompatible shapes")
        difference_energy += float((low_signal - high_signal).abs().square().sum().item())
    signal_elements = int(high["signal_elements"])
    if (
        signal_elements != SMALLFIT_QUADRATURE_SIGNAL_ELEMENTS
        or low.get("signal_elements") != signal_elements
    ):
        raise AssertionError("quadrature signal snapshot changed its full four-view signal domain")
    observed_energy = float(high["observed_energy"])
    if not math.isclose(observed_energy, float(low["observed_energy"]), rel_tol=0.0, abs_tol=0.0):
        raise AssertionError("quadrature grids must observe the same fixed data")
    signal_decision = quadrature_difference_decision(
        difference_rms=math.sqrt(difference_energy / signal_elements),
        reference_rms=math.sqrt(observed_energy / signal_elements),
        label="frequency_signal",
    )
    gradient_decision = quadrature_difference_decision(
        difference_rms=abs(float(low["directional_gradient"]) - float(high["directional_gradient"])),
        reference_rms=abs(float(high["directional_gradient"])),
        label="directional_gradient",
    )
    return {
        "frozen_weights": True,
        "source_ids": [int(item) for item in source_ids],
        "training_grid_size": SMALLFIT_TRAIN_GRID_SIZE,
        "reference_grid_size": SMALLFIT_REFERENCE_GRID_SIZE,
        "g48": low,
        "g96": high,
        "signal_gate": signal_decision,
        "directional_gradient_gate": gradient_decision,
        "gate_pass": bool(signal_decision["pass"] and gradient_decision["pass"]),
    }


@torch.no_grad()
def _field_readout(
    *,
    model: SpinrStyleINR,
    points_m: torch.Tensor,
    cell_volume_m3: float,
    initial_output_scale: float,
) -> dict[str, float | int | str | list[float]]:
    """Read out physical signed-real sigma, including the frozen gain."""

    network_field = evaluate_neural_field_tiled(
        model, points_m, neural_point_tile=SMALLFIT_NEURAL_POINT_TILE
    )
    field = network_field * _finite_positive(initial_output_scale, "field readout initial output scale")
    amplitude = field.abs()
    energy = field.square()
    return {
        "representation": "signed_real_sigma_midpoint_grid_readout",
        "grid_size": SMALLFIT_TRAIN_GRID_SIZE,
        "support_min_m": [-SPINR_STYLE_SUPPORT_M] * 3,
        "support_max_m": [SPINR_STYLE_SUPPORT_M] * 3,
        "field_points": int(field.numel()),
        "sigma_abs_mean": float(amplitude.mean().item()),
        "sigma_abs_max": float(amplitude.max().item()),
        "sigma_squared_mean": float(energy.mean().item()),
        "sigma_squared_integral_m3": float(energy.sum().item() * cell_volume_m3),
        "interpretation": "native field-amplitude/energy readout; not a reconstruction-quality claim",
    }


def _milestone_record(
    *,
    update: int,
    model: SpinrStyleINR,
    views: BoundedSealedRawComplexViews,
    worklists: B78716SmokeWorklists,
    points_m: torch.Tensor,
    cell_volume_m3: float,
    initial_output_scale: float,
    frequencies_hz: torch.Tensor,
    kvector: torch.Tensor,
    training_mean_raw_power: float,
    optimizer: Adam,
    device: torch.device,
    elapsed_seconds: float,
) -> dict[str, object]:
    train = evaluate_role(
        model=model,
        views=views,
        role="train",
        source_ids=worklists.fit_training_ids,
        points_m=points_m,
        cell_volume_m3=cell_volume_m3,
        initial_output_scale=initial_output_scale,
        frequencies_hz=frequencies_hz,
        kvector=kvector,
        training_mean_raw_power=training_mean_raw_power,
        neural_point_tile=SMALLFIT_NEURAL_POINT_TILE,
        renderer_point_tile=SMALLFIT_RENDERER_POINT_TILE,
        pair_tile=SMALLFIT_PAIR_TILE,
        device=device,
    )
    validation = evaluate_role(
        model=model,
        views=views,
        role="validation",
        source_ids=worklists.validation_ids,
        points_m=points_m,
        cell_volume_m3=cell_volume_m3,
        initial_output_scale=initial_output_scale,
        frequencies_hz=frequencies_hz,
        kvector=kvector,
        training_mean_raw_power=training_mean_raw_power,
        neural_point_tile=SMALLFIT_NEURAL_POINT_TILE,
        renderer_point_tile=SMALLFIT_RENDERER_POINT_TILE,
        pair_tile=SMALLFIT_PAIR_TILE,
        device=device,
    )
    return {
        "update": int(update),
        "production_clock": _production_clock(int(update)),
        "elapsed_seconds": _finite_nonnegative(elapsed_seconds, "milestone elapsed seconds"),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "fixed_train": train,
        "held_out_validation": validation,
        "same_domain_zero_reference": {
            "fixed_train": _zero_role_metrics(
                views=views,
                role="train",
                source_ids=worklists.fit_training_ids,
                training_mean_raw_power=training_mean_raw_power,
                device=device,
            ),
            "held_out_validation": _zero_role_metrics(
                views=views,
                role="validation",
                source_ids=worklists.validation_ids,
                training_mean_raw_power=training_mean_raw_power,
                device=device,
            ),
        },
        "memory": _memory_measurements(device),
    }


class _StopAfterCurrentUpdate:
    """Defer signals until the current logical B=4 update has committed."""

    def __init__(self) -> None:
        self.requested = False

    def __call__(self, signum: int, _frame: object) -> None:
        self.requested = True
        print(
            f"Received signal {signum}; bounded small fit will checkpoint after its current update.",
            flush=True,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--npz-path", default=DEFAULT_B787_NPZ_PATH)
    parser.add_argument("--parent-role-manifest", default=DEFAULT_B787_MANIFEST_PATH)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--host-rss-limit-gib", type=float, default=SMALLFIT_HOST_RSS_LIMIT_GIB)
    return parser.parse_args(argv)


def _validate_completed_smallfit_checkpoint_header(
    checkpoint: Mapping[str, object], *, recipe: Mapping[str, object]
) -> None:
    """Validate a terminal artifact before preserving it during a clean recovery.

    This is intentionally separate from the resumable-header validator: a
    completed final checkpoint must never be loaded back into the optimizer,
    but a report/finalization interruption may leave it as one half of the
    terminal pair.  We preserve a valid half and write only its missing peer.
    """

    if checkpoint.get("format") != SMALLFIT_CHECKPOINT_FORMAT:
        raise ValueError("bounded terminal checkpoint has the wrong format")
    if checkpoint.get("run_name") != SMALLFIT_RUN_NAME:
        raise ValueError("bounded terminal checkpoint belongs to a different run identity")
    if checkpoint.get("engineering_status") != SMALLFIT_ENGINEERING_STATUS:
        raise ValueError("bounded terminal checkpoint has the wrong engineering status")
    if checkpoint.get("recipe") != dict(recipe):
        raise ValueError("bounded terminal checkpoint would change the frozen recipe")
    execution = checkpoint.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("bounded terminal checkpoint lacks execution state")
    if execution.get("phase") != "completed" or execution.get("completed_updates") != SMALLFIT_MAX_UPDATES:
        raise ValueError("bounded terminal checkpoint is not the completed 60-update artifact")
    if execution.get("production_clock") != _production_clock(SMALLFIT_MAX_UPDATES):
        raise ValueError("bounded terminal checkpoint changed the 800-update production clock")
    resume_count = execution.get("resume_count", 0)
    if isinstance(resume_count, bool) or not isinstance(resume_count, int) or resume_count < 0:
        raise ValueError("bounded terminal checkpoint has an invalid resume count")
    _finite_nonnegative(execution.get("elapsed_seconds"), "bounded terminal elapsed seconds")


def _same_scientific_checkpoint_state(
    expected: Mapping[str, object], candidate: Mapping[str, object]
) -> bool:
    """Compare every committed scientific/recovery field across finalization.

    ``execution`` is intentionally excluded: terminal writing naturally has a
    different phase and elapsed time.  Everything that could change the
    trajectory, data binding, optimizer continuation, or diagnostic evidence
    must be byte-for-byte equivalent.
    """

    fields = (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "rng_state",
        "sealed_npz_protocol_contract",
        "acquisition_identity",
        "worklists",
        "fit_batch_schedule",
        "normalization",
        "history",
        "update_records",
        "quadrature_records",
    )
    return set(expected) == set(candidate) and all(
        _checkpoint_tree_equal(expected.get(key), candidate.get(key)) for key in fields
    )


def _validate_existing_terminal_checkpoint(
    path: Path,
    *,
    recipe: Mapping[str, object],
    expected_checkpoint: Mapping[str, object],
    sealed_contract: Mapping[str, object],
    acquisition_identity: Mapping[str, object],
    worklists: B78716SmokeWorklists,
    batches: Sequence[Sequence[int]],
    model_template: SpinrStyleINR,
) -> None:
    """Accept only a fully valid completed checkpoint matching pending evidence."""

    try:
        payload = load_tensor_checkpoint(str(path), map_location="cpu")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("existing bounded terminal checkpoint cannot be read") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("existing bounded terminal checkpoint must be an object")
    _validate_completed_smallfit_checkpoint_header(payload, recipe=recipe)
    _validate_checkpoint_payload_after_preflight(
        payload,
        sealed_contract=sealed_contract,
        acquisition_identity=acquisition_identity,
        worklists=worklists,
        batches=batches,
        model_template=model_template,
        label="existing bounded terminal checkpoint",
    )
    if not _same_scientific_checkpoint_state(expected_checkpoint, payload):
        raise ValueError("existing bounded terminal checkpoint disagrees with the committed update-60 evidence")


def _finite_nonzero_update_records(update_records: Sequence[object]) -> bool:
    return bool(update_records) and all(
        isinstance(record, Mapping)
        and math.isfinite(float(record.get("gradient_norm_before_clip", float("nan"))))
        and float(record.get("gradient_norm_before_clip", 0.0)) > 0.0
        and math.isfinite(float(record.get("parameter_delta_l2", float("nan"))))
        and float(record.get("parameter_delta_l2", 0.0)) > 0.0
        for record in update_records
    )


def _production_clearance_reasons(
    *, initial_rel_mse: float, final_rel_mse: float, finite_nonzero_updates: bool, quadrature_pass: bool
) -> list[str]:
    reasons = [
        "bounded_16_train_16_validation_60_update_engineering_smoke_is_not_a_production_run",
        "no_full_3200_view_convergence_or_sealed_test_evidence",
    ]
    if not final_rel_mse < initial_rel_mse:
        reasons.append("fixed_train_coherent_relative_mse_did_not_fall")
    if not finite_nonzero_updates:
        reasons.append("one_or_more_logical_updates_lacked_finite_nonzero_gradient_or_parameter_change")
    if not quadrature_pass:
        reasons.append("g48_g96_fixed_weight_quadrature_gate_did_not_pass")
    return reasons


def _validate_terminal_memory_evidence(memory: object) -> None:
    if not isinstance(memory, Mapping) or set(memory) != {
        "process_max_rss_bytes", "peak_torch_allocated_bytes", "peak_torch_reserved_bytes"
    }:
        raise ValueError("existing bounded terminal report lacks complete memory evidence")
    for key, value in memory.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"existing bounded terminal report has invalid {key}")


def _validate_terminal_geometry_readout(readout: object) -> None:
    if not isinstance(readout, Mapping):
        raise ValueError("existing bounded terminal report lacks geometry readout")
    if readout.get("representation") != "signed_real_sigma_midpoint_grid_readout":
        raise ValueError("existing bounded terminal report changed its field readout representation")
    if readout.get("grid_size") != SMALLFIT_TRAIN_GRID_SIZE or readout.get("field_points") != SMALLFIT_TRAIN_GRID_SIZE ** 3:
        raise ValueError("existing bounded terminal report changed its field readout grid")
    if readout.get("support_min_m") != [-SPINR_STYLE_SUPPORT_M] * 3 or readout.get("support_max_m") != [SPINR_STYLE_SUPPORT_M] * 3:
        raise ValueError("existing bounded terminal report changed its field readout support")
    if readout.get("interpretation") != "native field-amplitude/energy readout; not a reconstruction-quality claim":
        raise ValueError("existing bounded terminal report changed its field readout interpretation")
    for key in (
        "sigma_abs_mean",
        "sigma_abs_max",
        "sigma_squared_mean",
        "sigma_squared_integral_m3",
    ):
        _finite_nonnegative(readout.get(key), f"existing bounded terminal report {key}")


def _validate_existing_terminal_report(
    path: Path,
    *,
    expected_checkpoint: Mapping[str, object],
    worklists: B78716SmokeWorklists,
) -> None:
    """Accept only a complete report bound to the committed update-60 evidence."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("existing bounded terminal report cannot be read") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("existing bounded terminal report must be an object")
    required_keys = {
        "schema",
        "engineering_status",
        "lifecycle_complete",
        "production_clearance",
        "production_clearance_reasons",
        "method",
        "acquisition",
        "worklists",
        "normalization",
        "milestones",
        "quadrature_records",
        "fixed_train_error",
        "finite_nonzero_updates",
        "logical_update_count",
        "clean_checkpoint_recovery",
        "production_clock",
        "geometry_readout",
        "memory",
        "elapsed_seconds",
        "result_interpretation",
    }
    if set(payload) != required_keys:
        raise ValueError("existing bounded terminal report has an incomplete or mixed schema")
    if payload.get("schema") != SMALLFIT_REPORT_SCHEMA:
        raise ValueError("existing bounded terminal report has the wrong schema")
    if payload.get("engineering_status") != SMALLFIT_ENGINEERING_STATUS:
        raise ValueError("existing bounded terminal report has the wrong engineering status")
    if payload.get("lifecycle_complete") is not True or payload.get("production_clearance") is not False:
        raise ValueError("existing bounded terminal report does not preserve its non-production lifecycle")
    if payload.get("logical_update_count") != SMALLFIT_MAX_UPDATES:
        raise ValueError("existing bounded terminal report has the wrong logical update count")
    if payload.get("production_clock") != _production_clock(SMALLFIT_MAX_UPDATES):
        raise ValueError("existing bounded terminal report changed the 800-update production clock")
    expected_method = {
        "method_id": SPINR_STYLE_METHOD_ID,
        "display_name": "SpINR-style neural baseline (ours)",
        "production_recipe_id": SPINR_STYLE_RECIPE_ID,
        "smallfit_run_name": SMALLFIT_RUN_NAME,
    }
    if payload.get("method") != expected_method:
        raise ValueError("existing bounded terminal report changed its method identity")
    if payload.get("acquisition") != {"tx": 16, "rx": 16, "frequency_bins": 600, "units": "metres"}:
        raise ValueError("existing bounded terminal report changed its acquisition identity")
    if payload.get("worklists") != _worklists_dict(worklists):
        raise ValueError("existing bounded terminal report changed its frozen worklists")
    history = expected_checkpoint.get("history")
    updates = expected_checkpoint.get("update_records")
    quadrature = expected_checkpoint.get("quadrature_records")
    normalization = expected_checkpoint.get("normalization")
    execution = expected_checkpoint.get("execution")
    if not (
        isinstance(history, list)
        and isinstance(updates, list)
        and isinstance(quadrature, Mapping)
        and isinstance(normalization, Mapping)
        and isinstance(execution, Mapping)
    ):
        raise ValueError("committed update-60 checkpoint lacks report-binding evidence")
    if payload.get("normalization") != dict(normalization):
        raise ValueError("existing bounded terminal report changed normalization evidence")
    if payload.get("milestones") != [dict(item) for item in history]:
        raise ValueError("existing bounded terminal report changed milestone evidence")
    if payload.get("quadrature_records") != dict(quadrature):
        raise ValueError("existing bounded terminal report changed quadrature evidence")
    by_update = {int(item["update"]): item for item in history if isinstance(item, Mapping) and "update" in item}
    if set(by_update) != set(SMALLFIT_MILESTONES):
        raise ValueError("committed update-60 checkpoint lacks all report milestones")
    initial_train = by_update[0].get("fixed_train")
    final_train = by_update[SMALLFIT_MAX_UPDATES].get("fixed_train")
    if not isinstance(initial_train, Mapping) or not isinstance(final_train, Mapping):
        raise ValueError("committed update-60 checkpoint lacks fixed-train report metrics")
    initial_rel_mse = _finite_nonnegative(initial_train.get("coherent_relative_mse"), "committed initial RelMSE")
    final_rel_mse = _finite_nonnegative(final_train.get("coherent_relative_mse"), "committed final RelMSE")
    finite_nonzero_updates = _finite_nonzero_update_records(updates)
    initial_quad = quadrature.get("0")
    final_quad = quadrature.get(str(SMALLFIT_MAX_UPDATES))
    quadrature_pass = bool(
        isinstance(initial_quad, Mapping)
        and isinstance(final_quad, Mapping)
        and initial_quad.get("gate_pass") is True
        and final_quad.get("gate_pass") is True
    )
    expected_fixed_train = {
        "initial_coherent_relative_mse": initial_rel_mse,
        "final_coherent_relative_mse": final_rel_mse,
        "strictly_fell": final_rel_mse < initial_rel_mse,
    }
    if payload.get("fixed_train_error") != expected_fixed_train:
        raise ValueError("existing bounded terminal report changed fixed-train evidence")
    if payload.get("finite_nonzero_updates") is not finite_nonzero_updates:
        raise ValueError("existing bounded terminal report changed finite-update evidence")
    if payload.get("production_clearance_reasons") != _production_clearance_reasons(
        initial_rel_mse=initial_rel_mse,
        final_rel_mse=final_rel_mse,
        finite_nonzero_updates=finite_nonzero_updates,
        quadrature_pass=quadrature_pass,
    ):
        raise ValueError("existing bounded terminal report changed its clearance interpretation")
    expected_recovery = {
        "latest_checkpoint_written_after_each_complete_logical_update": True,
        "resume_segments_completed": execution.get("resume_count"),
        "scheduler_clock_was_not_advanced_by_smallfit_cycles": True,
    }
    if payload.get("clean_checkpoint_recovery") != expected_recovery:
        raise ValueError("existing bounded terminal report changed clean-recovery evidence")
    _validate_terminal_memory_evidence(payload.get("memory"))
    _validate_terminal_geometry_readout(payload.get("geometry_readout"))
    _finite_nonnegative(payload.get("elapsed_seconds"), "existing bounded terminal report elapsed seconds")
    if payload.get("result_interpretation") != (
        "A completed bounded fitting/renderer engineering gate only. It is not a "
        "3D reconstruction, NVS, ranking, convergence, or full-scale baseline result."
    ):
        raise ValueError("existing bounded terminal report changed its scope interpretation")


def _validate_partial_terminal_recovery(
    checkpoint: Mapping[str, object],
    *,
    final_checkpoint: Path,
    report_path: Path,
    recipe: Mapping[str, object],
    sealed_contract: Mapping[str, object],
    acquisition_identity: Mapping[str, object],
    worklists: B78716SmokeWorklists,
    batches: Sequence[Sequence[int]],
    model_template: SpinrStyleINR,
) -> None:
    """Permit a literal resume only to finish one missing terminal artifact.

    ``checkpoint_latest`` remains the authoritative pending-finalization state
    until both terminal outputs are atomically present.  A partial pair is not
    terminal evidence and cannot reopen training; it can only complete the
    missing peer at the already committed update-60 boundary.
    """

    final_exists = final_checkpoint.exists()
    report_exists = report_path.exists()
    if not (final_exists or report_exists):
        return
    if final_exists and report_exists:
        raise ValueError("both bounded terminal artifacts already exist; do not rerun or resume")
    execution = checkpoint.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("partial-terminal recovery lacks resumable execution state")
    if (
        execution.get("completed_updates") != SMALLFIT_MAX_UPDATES
        or execution.get("phase") not in {"interrupted_clean", "pending_finalization"}
    ):
        raise ValueError("partial-terminal recovery may only finish a committed update-60 boundary")
    if final_exists:
        _validate_existing_terminal_checkpoint(
            final_checkpoint,
            recipe=recipe,
            expected_checkpoint=checkpoint,
            sealed_contract=sealed_contract,
            acquisition_identity=acquisition_identity,
            worklists=worklists,
            batches=batches,
            model_template=model_template,
        )
    if report_exists:
        _validate_existing_terminal_report(
            report_path,
            expected_checkpoint=checkpoint,
            worklists=worklists,
        )


def _stale_initial_latest_temp(path: Path) -> bool:
    """Recognize only a non-authoritative first latest-checkpoint temp file.

    The shared atomic writer uses ``.<checkpoint>.tmp.<pid>`` then
    ``os.replace``.  A hard stop before that replace leaves no checkpoint to
    resume, only this private temporary.  It is intentionally preserved, never
    deleted or reused: accepting it for a fresh start is safe only when its PID
    differs from this new process's destination filename.
    """

    if not path.is_file():
        return False
    prefix = f".{_CHECKPOINT_LATEST}.tmp."
    suffix = path.name[len(prefix):] if path.name.startswith(prefix) else ""
    return suffix.isdigit() and suffix != str(os.getpid())


def _empty_run_root(root: Path) -> bool:
    """Return whether a root has no durable state, only a stale first-write temp."""

    if not root.is_dir():
        return False
    return all(_stale_initial_latest_temp(path) for path in root.iterdir())


def _validate_cli(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if not _same_resolved_path(args.npz_path, DEFAULT_B787_NPZ_PATH):
        raise ValueError("bounded SpINR-style smoke accepts only the canonical /storage/home B787 archive")
    if not _same_resolved_path(args.parent_role_manifest, DEFAULT_B787_MANIFEST_PATH):
        raise ValueError("bounded SpINR-style smoke accepts only the canonical parent role manifest")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("bounded SpINR-style smoke is a single-GPU engineering job")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("bounded SpINR-style smoke requires exactly one allocated CUDA device")
    if not math.isclose(
        _finite_positive(args.host_rss_limit_gib, "host RSS limit GiB"),
        SMALLFIT_HOST_RSS_LIMIT_GIB,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("bounded SpINR-style smoke fixes the reviewed 32 GiB host RSS envelope")
    checkpoint_root = _absolute_unresolved(args.checkpoint_root)
    run_root = checkpoint_root / SMALLFIT_RUN_NAME
    latest = run_root / _CHECKPOINT_LATEST
    final = run_root / _CHECKPOINT_FINAL
    report = run_root / _REPORT_NAME
    for path, label in (
        (checkpoint_root, "checkpoint root"),
        (run_root, "bounded run root"),
        (latest, "bounded latest checkpoint"),
        (final, "bounded final checkpoint"),
        (report, "bounded report"),
    ):
        _reject_symlink(path, label)
    for path, label in ((latest, "bounded latest checkpoint"), (final, "bounded final checkpoint"), (report, "bounded report")):
        if path.exists() and not path.is_file():
            raise ValueError(f"{label} must be a regular file when present")
    if args.resume is None:
        # A setup interruption before the first atomic checkpoint leaves no
        # experiment state.  Reusing only an empty root is safe; any durable
        # state still requires the literal same-root resume path.
        if run_root.exists() and not _empty_run_root(run_root):
            raise ValueError("bounded run root already has durable state; only explicit same-run clean resume is permitted")
    else:
        resume = _absolute_unresolved(args.resume)
        _reject_symlink(resume, "resume checkpoint")
        if resume != latest or not latest.is_file():
            raise ValueError("resume must be this bounded run's own checkpoint_latest.pth.tar")
        if not run_root.is_dir():
            raise ValueError("bounded resume root is not a directory")
        if final.exists() and report.exists():
            raise ValueError("bounded SpINR-style run has complete terminal evidence; do not rerun or resume it")
    return run_root, latest, final


def _load_resume_preflight(
    resume: str | None, *, recipe: Mapping[str, object]
) -> Mapping[str, object] | None:
    if resume is None:
        return None
    checkpoint = load_tensor_checkpoint(resume, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("bounded resume checkpoint must be an object")
    _validate_smallfit_checkpoint_header(checkpoint, recipe=recipe)
    return checkpoint


def _restore_checkpoint(
    *,
    checkpoint: Mapping[str, object],
    model: SpinrStyleINR,
    optimizer: Adam,
    scheduler: CosineAnnealingLR,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        restore_rng_state(checkpoint["rng_state"], require_complete=True)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("bounded resume checkpoint cannot restore model/optimizer/scheduler/RNG state") from exc
    if int(scheduler.last_epoch) != 0:
        raise ValueError("bounded resume unexpectedly advanced the production scheduler")
    execution = checkpoint["execution"]
    normalization = checkpoint["normalization"]
    history = checkpoint["history"]
    updates = checkpoint["update_records"]
    quadrature = checkpoint.get("quadrature_records", {})
    if not isinstance(execution, Mapping) or not isinstance(normalization, Mapping):
        raise ValueError("bounded resume checkpoint has malformed execution/normalization state")
    if not isinstance(history, list) or not isinstance(updates, list) or not isinstance(quadrature, Mapping):
        raise ValueError("bounded resume checkpoint has malformed evidence history")
    return (
        dict(normalization),
        [dict(item) for item in history if isinstance(item, Mapping)],
        [dict(item) for item in updates if isinstance(item, Mapping)],
        {str(key): value for key, value in quadrature.items()},
    )


def _build_report(
    *,
    worklists: B78716SmokeWorklists,
    history: Sequence[Mapping[str, object]],
    update_records: Sequence[Mapping[str, object]],
    quadrature_records: Mapping[str, object],
    normalization: Mapping[str, object],
    final_readout: Mapping[str, object],
    memory: Mapping[str, object],
    elapsed_seconds: float,
    resume_count: int,
) -> dict[str, object]:
    by_update = {int(item["update"]): item for item in history if "update" in item}
    if set(by_update) != set(SMALLFIT_MILESTONES):
        raise AssertionError("bounded report requires exactly 0/4/20/60 milestones")
    initial = by_update[0]
    final = by_update[SMALLFIT_MAX_UPDATES]
    initial_train = initial["fixed_train"]
    final_train = final["fixed_train"]
    if not isinstance(initial_train, Mapping) or not isinstance(final_train, Mapping):
        raise AssertionError("bounded report has malformed train metrics")
    initial_rel_mse = _finite_nonnegative(initial_train["coherent_relative_mse"], "initial train RelMSE")
    final_rel_mse = _finite_nonnegative(final_train["coherent_relative_mse"], "final train RelMSE")
    finite_nonzero_updates = _finite_nonzero_update_records(update_records)
    initial_quad = quadrature_records.get("0")
    final_quad = quadrature_records.get(str(SMALLFIT_MAX_UPDATES))
    quadrature_pass = bool(
        isinstance(initial_quad, Mapping)
        and isinstance(final_quad, Mapping)
        and initial_quad.get("gate_pass") is True
        and final_quad.get("gate_pass") is True
    )
    production_clearance_reasons = _production_clearance_reasons(
        initial_rel_mse=initial_rel_mse,
        final_rel_mse=final_rel_mse,
        finite_nonzero_updates=finite_nonzero_updates,
        quadrature_pass=quadrature_pass,
    )
    return {
        "schema": SMALLFIT_REPORT_SCHEMA,
        "engineering_status": SMALLFIT_ENGINEERING_STATUS,
        "lifecycle_complete": True,
        "production_clearance": False,
        "production_clearance_reasons": production_clearance_reasons,
        "method": {
            "method_id": SPINR_STYLE_METHOD_ID,
            "display_name": "SpINR-style neural baseline (ours)",
            "production_recipe_id": SPINR_STYLE_RECIPE_ID,
            "smallfit_run_name": SMALLFIT_RUN_NAME,
        },
        "acquisition": {"tx": 16, "rx": 16, "frequency_bins": 600, "units": "metres"},
        "worklists": _worklists_dict(worklists),
        "normalization": dict(normalization),
        "milestones": [dict(by_update[item]) for item in SMALLFIT_MILESTONES],
        "quadrature_records": dict(quadrature_records),
        "fixed_train_error": {
            "initial_coherent_relative_mse": initial_rel_mse,
            "final_coherent_relative_mse": final_rel_mse,
            "strictly_fell": final_rel_mse < initial_rel_mse,
        },
        "finite_nonzero_updates": finite_nonzero_updates,
        "logical_update_count": len(update_records),
        "clean_checkpoint_recovery": {
            "latest_checkpoint_written_after_each_complete_logical_update": True,
            "resume_segments_completed": int(resume_count),
            "scheduler_clock_was_not_advanced_by_smallfit_cycles": True,
        },
        "production_clock": _production_clock(SMALLFIT_MAX_UPDATES),
        "geometry_readout": dict(final_readout),
        "memory": dict(memory),
        "elapsed_seconds": _finite_nonnegative(elapsed_seconds, "total elapsed seconds"),
        "result_interpretation": (
            "A completed bounded fitting/renderer engineering gate only. It is not a "
            "3D reconstruction, NVS, ranking, convergence, or full-scale baseline result."
        ),
    }


def run(args: argparse.Namespace) -> int:
    run_root, _latest, final_checkpoint = _validate_cli(args)
    recipe = smallfit_recipe_identity()
    resume_checkpoint = _load_resume_preflight(args.resume, recipe=recipe)
    report_path = run_root / _REPORT_NAME
    _disable_tf32()
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    # All resume-relevant archive/role/acquisition identity checks happen
    # before the bounded adapter is allowed to decompress any response row.
    # The production preflight's optional checkpoint hook expects the
    # production-only normalization key name.  This bounded driver has its own
    # explicit recovery schema, so bind the same canonical header/roles and
    # acquisition identity here, then validate its small-fit state below—all
    # before a response payload is opened.
    arrays, sealed_contract, acquisition_identity = preflight_b787_development_inputs(
        npz_path=args.npz_path,
        manifest_path=args.parent_role_manifest,
        resume_checkpoint=None,
    )
    worklists = bounded_b78716_smoke_worklists(sealed_contract)
    batches = smallfit_update_batches(worklists.fit_training_ids)
    if resume_checkpoint is not None:
        # A CPU template makes the resumed tensor/Adam layout check independent
        # of any B787 response payload or GPU allocation state.
        resume_model_template = SpinrStyleINR()
        _validate_resume_after_preflight(
            resume_checkpoint,
            recipe=recipe,
            sealed_contract=sealed_contract,
            acquisition_identity=acquisition_identity,
            worklists=worklists,
            batches=batches,
            model_template=resume_model_template,
        )
        _validate_partial_terminal_recovery(
            resume_checkpoint,
            final_checkpoint=final_checkpoint,
            report_path=report_path,
            recipe=recipe,
            sealed_contract=sealed_contract,
            acquisition_identity=acquisition_identity,
            worklists=worklists,
            batches=batches,
            model_template=resume_model_template,
        )
        del resume_model_template

    # Start deferring TERM/INT before a fresh run creates its durable identity
    # or streams 3,200 parent rows.  If a signal arrives during ingestion or
    # initialization, the handler remains set until the zero-update checkpoint
    # is durable and then exits through the same literal resume route.
    stopper = _StopAfterCurrentUpdate()
    original_term = signal.signal(signal.SIGTERM, stopper)
    original_int = signal.signal(signal.SIGINT, stopper)
    views = build_bounded_sealed_raw_complex_views(arrays, sealed_contract)
    print(
        "Bounded raw-complex cache: "
        f"{views.materialized_response_bytes / (1024.0 ** 2):.2f} MiB for retained fit/validation/init rows; "
        "all 3,200 parent train rows were streamed only for normalization.",
        flush=True,
    )
    _enforce_memory_gates(device=device, host_rss_limit_gib=SMALLFIT_HOST_RSS_LIMIT_GIB)

    # Construct under the fixed seed even for a resume, then restore the full
    # state below.  This avoids a shape/state branch that could change an
    # otherwise identical continuation.
    from train_spinr_style import _set_seed  # keep production helper behavior unchanged

    _set_seed(CANONICAL_SEED)
    model = SpinrStyleINR().to(device=device, dtype=torch.float32)
    if model.trainable_parameter_count() != SPINR_STYLE_PARAMETER_COUNT:
        raise AssertionError("SpINR-style parameter count drifted from the frozen recipe")
    optimizer = _new_smallfit_adam(model)
    scheduler = _new_smallfit_scheduler(optimizer)
    points_m, cell_volume_m3 = midpoint_grid(
        SMALLFIT_TRAIN_GRID_SIZE, device=device, dtype=torch.float64
    )
    frequencies_hz = torch.as_tensor(views.frequencies_hz, device=device, dtype=torch.float64)
    kvector = get_kvector(frequencies_hz, cc).to(device=device, dtype=torch.float64)
    if frequencies_hz.shape != (600,) or not torch.isfinite(frequencies_hz).all():
        raise AssertionError("B787 metadata did not provide the required finite 600-bin frequency grid")

    started_at = time.monotonic()
    if resume_checkpoint is None:
        training_mean_raw_power = views.raw_training_mean_power()
        initial_scale, initial_scale_ids, observed_energy, predicted_energy = _estimate_initial_output_scale(
            model=model,
            views=views,
            points_m=points_m,
            cell_volume_m3=cell_volume_m3,
            frequencies_hz=frequencies_hz,
            kvector=kvector,
            device=device,
        )
        normalization: dict[str, object] = {
            "training_mean_raw_power": training_mean_raw_power,
            "initial_scale_ids": list(initial_scale_ids),
            "initial_output_scale": initial_scale,
            "initial_scale_observed_energy": observed_energy,
            "initial_scale_predicted_energy": predicted_energy,
            "normalization_source": "streamed_all_3200_parent_train_rows",
        }
        history: list[dict[str, object]] = []
        update_records: list[dict[str, object]] = []
        quadrature_records: dict[str, object] = {}
        resume_count = 0
        initial_record = _milestone_record(
            update=0,
            model=model,
            views=views,
            worklists=worklists,
            points_m=points_m,
            cell_volume_m3=cell_volume_m3,
            initial_output_scale=initial_scale,
            frequencies_hz=frequencies_hz,
            kvector=kvector,
            training_mean_raw_power=training_mean_raw_power,
            optimizer=optimizer,
            device=device,
            elapsed_seconds=time.monotonic() - started_at,
        )
        history.append(initial_record)
        quadrature_records["0"] = run_quadrature_diagnostic(
            model=model,
            views=views,
            initial_output_scale=initial_scale,
            frequencies_hz=frequencies_hz,
            kvector=kvector,
            training_mean_raw_power=training_mean_raw_power,
            device=device,
        )
        completed_updates = 0
    else:
        saved_normalization = resume_checkpoint.get("normalization")
        if not isinstance(saved_normalization, Mapping):
            raise ValueError("bounded resume checkpoint has malformed normalization state")
        _validate_resumed_normalization_against_materialized_b787(
            normalization=saved_normalization,
            model=model,
            views=views,
            points_m=points_m,
            cell_volume_m3=cell_volume_m3,
            frequencies_hz=frequencies_hz,
            kvector=kvector,
            device=device,
        )
        normalization, history, update_records, quadrature_records = _restore_checkpoint(
            checkpoint=resume_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        training_mean_raw_power = _finite_positive(
            normalization["training_mean_raw_power"], "saved training mean raw power"
        )
        initial_scale = _finite_positive(
            normalization["initial_output_scale"], "saved initial output scale"
        )
        execution = resume_checkpoint["execution"]
        assert isinstance(execution, Mapping)
        completed_updates = int(execution["completed_updates"])
        saved_resume_count = execution.get("resume_count", 0)
        resume_count = _resume_segments_after_restore(
            saved_resume_count=saved_resume_count,
            completed_updates=completed_updates,
        )
        prior_elapsed = _finite_nonnegative(execution.get("elapsed_seconds", 0.0), "saved elapsed seconds")
        started_at -= prior_elapsed

    try:
        # Do not create the durable identity while the expensive setup has no
        # complete logical-update boundary to save.  If setup fails or is hard
        # killed before this point, a subsequent fresh invocation sees no
        # state; a deferred TERM/INT still reaches the zero-update checkpoint
        # below.  An already-empty root is the only setup remnant accepted by
        # the CLI, so no durable evidence is overwritten.
        if resume_checkpoint is None:
            run_root.mkdir(parents=True, exist_ok=True)

        def save_state(phase: str) -> None:
            elapsed = time.monotonic() - started_at
            state = _checkpoint_state(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                recipe=recipe,
                sealed_contract=sealed_contract,
                acquisition_identity=acquisition_identity,
                worklists=worklists,
                batches=batches,
                normalization=normalization,
                execution=_execution_state(
                    phase=phase,
                    completed_updates=completed_updates,
                    elapsed_seconds=elapsed,
                    resume_count=resume_count,
                ),
                history=history,
                update_records=update_records,
                quadrature_records=quadrature_records,
            )
            _save_latest_and_history(run_root, state, history)

        # A fresh zero-update measurement is recoverable before any optimizer
        # step.  A resume already has an authoritative latest checkpoint.
        if resume_checkpoint is None:
            save_state("updates")
        if stopper.requested:
            save_state("interrupted_clean")
            print("SPINR_STYLE_B78716_SMALLFIT_QUADRATURE_CLEAN_INTERRUPT", flush=True)
            return 143

        for update_index in range(completed_updates, SMALLFIT_MAX_UPDATES):
            parameter_before = _parameter_snapshots(model)
            update_started = time.monotonic()
            loss, gradient_norm = logical_batch_update(
                model=model,
                optimizer=optimizer,
                source_ids=batches[update_index],
                views=views,
                points_m=points_m,
                cell_volume_m3=cell_volume_m3,
                initial_output_scale=initial_scale,
                frequencies_hz=frequencies_hz,
                kvector=kvector,
                training_mean_raw_power=training_mean_raw_power,
                neural_point_tile=SMALLFIT_NEURAL_POINT_TILE,
                renderer_point_tile=SMALLFIT_RENDERER_POINT_TILE,
                pair_tile=SMALLFIT_PAIR_TILE,
                device=device,
            )
            parameter_delta = _parameter_delta_l2(model, parameter_before)
            if not math.isfinite(loss) or not math.isfinite(gradient_norm):
                raise FloatingPointError("bounded SpINR-style update produced non-finite loss or gradient norm")
            completed_updates = update_index + 1
            update_records.append({
                "update": completed_updates,
                "source_ids": [int(item) for item in batches[update_index]],
                "native_batch_objective": float(loss),
                "gradient_norm_before_clip": float(gradient_norm),
                "gradient_was_clipped": bool(gradient_norm > 1.0),
                "parameter_delta_l2": parameter_delta,
                "update_seconds": _finite_nonnegative(
                    time.monotonic() - update_started, "logical update seconds"
                ),
            })
            if completed_updates in SMALLFIT_MILESTONES:
                history.append(_milestone_record(
                    update=completed_updates,
                    model=model,
                    views=views,
                    worklists=worklists,
                    points_m=points_m,
                    cell_volume_m3=cell_volume_m3,
                    initial_output_scale=initial_scale,
                    frequencies_hz=frequencies_hz,
                    kvector=kvector,
                    training_mean_raw_power=training_mean_raw_power,
                    optimizer=optimizer,
                    device=device,
                    elapsed_seconds=time.monotonic() - started_at,
                ))
                if completed_updates == SMALLFIT_MAX_UPDATES:
                    quadrature_records[str(completed_updates)] = run_quadrature_diagnostic(
                        model=model,
                        views=views,
                        initial_output_scale=initial_scale,
                        frequencies_hz=frequencies_hz,
                        kvector=kvector,
                        training_mean_raw_power=training_mean_raw_power,
                        device=device,
                    )
            checkpoint_phase = (
                "interrupted_clean"
                if stopper.requested
                else ("pending_finalization" if completed_updates == SMALLFIT_MAX_UPDATES else "updates")
            )
            save_state(checkpoint_phase)
            print(
                f"SPINR_STYLE_B78716_SMALLFIT_UPDATE={completed_updates}/{SMALLFIT_MAX_UPDATES} "
                f"loss={loss:.6e} grad_norm={gradient_norm:.6e} delta_l2={parameter_delta:.6e}",
                flush=True,
            )
            if stopper.requested:
                print("SPINR_STYLE_B78716_SMALLFIT_QUADRATURE_CLEAN_INTERRUPT", flush=True)
                return 143

        if completed_updates != SMALLFIT_MAX_UPDATES:
            raise AssertionError("bounded small-fit loop ended before its frozen 60-update budget")
        if stopper.requested:
            save_state("interrupted_clean")
            print("SPINR_STYLE_B78716_SMALLFIT_QUADRATURE_CLEAN_INTERRUPT", flush=True)
            return 143
        if {int(item["update"]) for item in history} != set(SMALLFIT_MILESTONES):
            raise AssertionError("bounded small fit missed a required 0/4/20/60 milestone")
        if set(quadrature_records) != {"0", str(SMALLFIT_MAX_UPDATES)}:
            raise AssertionError("bounded small fit missed required initial/final quadrature diagnostics")
        if int(scheduler.last_epoch) != 0:
            raise AssertionError("bounded small fit must not advance the production cosine scheduler")
        final_readout = _field_readout(
            model=model,
            points_m=points_m,
            cell_volume_m3=cell_volume_m3,
            initial_output_scale=initial_scale,
        )
        if stopper.requested:
            save_state("interrupted_clean")
            print("SPINR_STYLE_B78716_SMALLFIT_QUADRATURE_CLEAN_INTERRUPT", flush=True)
            return 143
        _enforce_memory_gates(device=device, host_rss_limit_gib=SMALLFIT_HOST_RSS_LIMIT_GIB)
        memory = _memory_measurements(device)
        report = _build_report(
            worklists=worklists,
            history=history,
            update_records=update_records,
            quadrature_records=quadrature_records,
            normalization=normalization,
            final_readout=final_readout,
            memory=memory,
            elapsed_seconds=time.monotonic() - started_at,
            resume_count=resume_count,
        )
        final_state = _checkpoint_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            recipe=recipe,
            sealed_contract=sealed_contract,
            acquisition_identity=acquisition_identity,
            worklists=worklists,
            batches=batches,
            normalization=normalization,
            execution=_execution_state(
                phase="completed",
                completed_updates=completed_updates,
                elapsed_seconds=time.monotonic() - started_at,
                resume_count=resume_count,
            ),
            history=history,
            update_records=update_records,
            quadrature_records=quadrature_records,
        )
        # Keep ``checkpoint_latest`` in recoverable pending-finalization state
        # until both terminal outputs are present.  Each output is atomic and
        # never overwritten: a clean literal resume after a report/final
        # interruption validates the existing half and writes only its peer.
        terminal_expected_checkpoint = resume_checkpoint if resume_checkpoint is not None else final_state
        if report_path.exists():
            _validate_existing_terminal_report(
                report_path,
                expected_checkpoint=terminal_expected_checkpoint,
                worklists=worklists,
            )
        else:
            _atomic_json_save(report, report_path)
        if final_checkpoint.exists():
            _validate_existing_terminal_checkpoint(
                final_checkpoint,
                recipe=recipe,
                expected_checkpoint=terminal_expected_checkpoint,
                sealed_contract=sealed_contract,
                acquisition_identity=acquisition_identity,
                worklists=worklists,
                batches=batches,
                model_template=model,
            )
        else:
            _atomic_torch_save(final_state, final_checkpoint)
        print(f"SPINR_STYLE_B78716_SMALLFIT_QUADRATURE_REPORT: {report_path}", flush=True)
        print("SPINR_STYLE_B78716_SMALLFIT_QUADRATURE_LIFECYCLE_COMPLETE", flush=True)
        return 0
    finally:
        signal.signal(signal.SIGTERM, original_term)
        signal.signal(signal.SIGINT, original_int)


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))
