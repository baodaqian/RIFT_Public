#!/usr/bin/env python3
"""Validate the isolated B787 SpINR-style small-fit package without radar data.

This fixture intentionally does not open the B787 archive, submit a job, or
write a durable experiment artifact.  The allocated launcher runs it before
the one real-data bounded smoke; the driver then supplies the hardware/runtime
evidence that a source-only fixture cannot.
"""

from __future__ import annotations

import ast
import copy
import math
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rift import spinr_smoke_runtime as smallfit  # noqa: E402
import postflight_spinr_style_b78716_smallfit as postflight  # noqa: E402
from rift.spinr_style import SPINR_STYLE_PARAMETER_COUNT, SpinrStyleINR  # noqa: E402
from rift.spinr_style_b78716_smoke import bounded_b78716_smoke_worklists  # noqa: E402


CHECKS = 0


def check(condition: bool, message: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)
    print(f"PASS {CHECKS:02d}: {message}")


def expect_value_error(action: Callable[[], object], message: str) -> None:
    try:
        action()
    except ValueError:
        rejected = True
    else:
        rejected = False
    check(rejected, message)


def nested_equal(left: object, right: object) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(nested_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(nested_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def synthetic_update(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """A data-free all-parameter Adam update for checkpoint recovery testing."""

    optimizer.zero_grad(set_to_none=True)
    loss = sum(parameter.square().mean() for parameter in model.parameters())
    loss.backward()
    optimizer.step()


def canonical_contract() -> dict[str, Any]:
    return {
        "schema": "rift_npz_sealed_protocol_v1",
        "version": 1,
        "data_format": "npz",
        "response_shape": [10000, 16, 16, 1, 600],
        "response_dtype": "complex64",
        "split_strategy": "fixture_canonical_roles",
        "response_access": {
            "train_materialized": True,
            "validation_materialized": True,
            "reserved_test_materialized": False,
            "unused_materialized": False,
        },
        "role_ids": {
            "train": list(range(3200)),
            "validation": list(range(3200, 4200)),
            "reserved_test": list(range(4200, 5200)),
            "unused": list(range(5200, 10000)),
        },
    }


def canonical_acquisition(contract: dict[str, Any]) -> dict[str, Any]:
    roles = contract["role_ids"]
    authorized = list(roles["train"]) + list(roles["validation"])
    count = len(authorized)
    return {
        "schema": "spinr_style_b787_acquisition_v1",
        "metadata": {
            "target_type": "b787",
            "experiment": "sphere10k",
            "radar_fc_hz": 10.0e9,
            "radar_bandwidth_hz": 3.0e9,
            "num_adc_samples": 600,
        },
        "frequency_hz": torch.linspace(8.5e9, 11.5e9, 600, dtype=torch.float64),
        "authorized_view_ids": authorized,
        "rx_pos_m": torch.zeros((count, 16, 3), dtype=torch.float64),
        "tx_pos_m": torch.ones((count, 16, 3), dtype=torch.float64),
    }


def fixture_metrics(relative_mse: float) -> dict[str, float | int]:
    return {
        "views": 16,
        "native_spectral_objective": float(relative_mse),
        "coherent_relative_mse": float(relative_mse),
        "coherent_relative_l2": float(relative_mse) ** 0.5,
    }


def fixture_zero_metrics() -> dict[str, float | int]:
    """The zero predictor's same-domain coherent relative errors are exact."""

    return {
        "views": 16,
        "native_spectral_objective": 1.0,
        "coherent_relative_mse": 1.0,
        "coherent_relative_l2": 1.0,
    }


def fixture_milestone(update: int, relative_mse: float) -> dict[str, Any]:
    metrics = fixture_metrics(relative_mse)
    zero_metrics = fixture_zero_metrics()
    return {
        "update": int(update),
        "production_clock": smallfit._production_clock(int(update)),
        "elapsed_seconds": float(update),
        "learning_rate": 1.0e-4,
        "fixed_train": dict(metrics),
        "held_out_validation": dict(metrics),
        "same_domain_zero_reference": {
            "fixed_train": dict(zero_metrics),
            "held_out_validation": dict(zero_metrics),
        },
        "memory": {
            "process_max_rss_bytes": 1,
            "peak_torch_allocated_bytes": 1,
            "peak_torch_reserved_bytes": 1,
        },
    }


def fixture_quadrature_record(worklists: Any) -> dict[str, Any]:
    signal_reference = math.sqrt(1.0 / float(smallfit.SMALLFIT_QUADRATURE_SIGNAL_ELEMENTS))
    decision = smallfit.quadrature_difference_decision(
        difference_rms=0.001 * signal_reference,
        reference_rms=signal_reference,
        label="frequency_signal",
    )
    gradient_decision = smallfit.quadrature_difference_decision(
        difference_rms=0.0, reference_rms=1.0, label="directional_gradient"
    )
    observation = {
        "observed_energy": 1.0,
        "signal_elements": smallfit.SMALLFIT_QUADRATURE_SIGNAL_ELEMENTS,
        "native_spectral_objective": 1.0,
        "directional_gradient": 1.0,
    }
    return {
        "frozen_weights": True,
        "source_ids": [int(item) for item in worklists.fit_training_ids[:4]],
        "training_grid_size": smallfit.SMALLFIT_TRAIN_GRID_SIZE,
        "reference_grid_size": smallfit.SMALLFIT_REFERENCE_GRID_SIZE,
        "g48": {"grid_size": smallfit.SMALLFIT_TRAIN_GRID_SIZE, **observation},
        "g96": {"grid_size": smallfit.SMALLFIT_REFERENCE_GRID_SIZE, **observation},
        "signal_gate": decision,
        "directional_gradient_gate": gradient_decision,
        "gate_pass": True,
    }


def stage_worklists_and_clock() -> tuple[dict[str, Any], tuple[tuple[int, ...], ...]]:
    contract = canonical_contract()
    worklists = bounded_b78716_smoke_worklists(contract)
    check(
        worklists.fit_training_ids == tuple(range(16))
        and worklists.validation_ids == tuple(range(3200, 3216)),
        "the bounded adapter selects ordered parent train[:16] and validation[:16]",
    )
    check(
        worklists.normalization_training_ids == tuple(range(3200))
        and worklists.initialization_training_ids == tuple(range(32)),
        "normalization streams all parent training rows while gain initialization retains parent train[:32]",
    )
    batches = smallfit.smallfit_update_batches(worklists.fit_training_ids)
    check(
        len(batches) == 60 and all(len(batch) == 4 for batch in batches),
        "the bounded schedule contains exactly sixty logical B=4 updates",
    )
    check(
        all(set(sum((list(batch) for batch in batches[start:start + 4]), [])) == set(worklists.fit_training_ids)
            for start in range(0, 60, 4)),
        "each of fifteen smoke cycles uses every selected training viewpoint exactly once",
    )
    check(
        batches == smallfit.smallfit_update_batches(worklists.fit_training_ids)
        and batches != smallfit.smallfit_update_batches(worklists.fit_training_ids, seed=43),
        "viewpoint shuffling is deterministic for the frozen seed and changes only under an explicit different seed",
    )
    check(
        smallfit._production_clock(60) == {
            "updates_per_production_epoch": 800,
            "completed_production_epochs": 0,
            "updates_into_current_production_epoch": 60,
        },
        "sixty smoke updates remain inside the unadvanced 800-update production epoch",
    )
    return contract, batches


def stage_model_and_quadrature_conventions() -> None:
    model = SpinrStyleINR()
    check(
        model.trainable_parameter_count() == SPINR_STYLE_PARAMETER_COUNT == 3_566_641,
        "the opt-in driver uses the frozen 39-input six-by-840 signed-real parameter budget",
    )
    points = torch.tensor([[0.0, 0.0, 0.0], [0.05, -0.04, 0.03]], dtype=torch.float64)
    prediction = model(points)
    check(
        prediction.shape == (2,) and not torch.is_complex(prediction) and torch.isfinite(prediction).all(),
        "the bounded package preserves a finite signed-real field output",
    )
    relative = smallfit.quadrature_difference_decision(
        difference_rms=0.009, reference_rms=1.0, label="fixture_relative"
    )
    check(
        relative["comparison_mode"] == "relative" and relative["pass"] is True,
        "the G48/G96 relative signal/gradient gate accepts differences at or below one percent",
    )
    rejected = smallfit.quadrature_difference_decision(
        difference_rms=0.011, reference_rms=1.0, label="fixture_relative"
    )
    check(rejected["pass"] is False, "the relative quadrature gate rejects differences above one percent")
    near_zero = smallfit.quadrature_difference_decision(
        difference_rms=0.5 * smallfit.QUADRATURE_NEAR_ZERO_ABSOLUTE_TOLERANCE,
        reference_rms=0.0,
        label="fixture_zero",
    )
    check(
        near_zero["comparison_mode"] == "near_zero_absolute" and near_zero["pass"] is True,
        "zero/near-zero diagnostics use the explicit absolute convention instead of a false relative pass",
    )


def stage_checkpoint_header(contract: dict[str, Any], batches: tuple[tuple[int, ...], ...]) -> None:
    worklists = bounded_b78716_smoke_worklists(contract)
    recipe = smallfit.smallfit_recipe_identity()
    checkpoint = {
        "format": smallfit.SMALLFIT_CHECKPOINT_FORMAT,
        "run_name": smallfit.SMALLFIT_RUN_NAME,
        "recipe": recipe,
        "execution": {
            "phase": "interrupted_clean",
            "completed_updates": 4,
            "production_clock": smallfit._production_clock(4),
        },
    }
    smallfit._validate_smallfit_checkpoint_header(checkpoint, recipe=recipe)
    check(True, "clean recovery header preserves the bounded identity and unadvanced production clock")
    changed = copy.deepcopy(checkpoint)
    changed["execution"]["production_clock"] = smallfit._production_clock(5)
    expect_value_error(
        lambda: smallfit._validate_smallfit_checkpoint_header(changed, recipe=recipe),
        "resume rejects a checkpoint whose cursor and production clock disagree",
    )
    pending = copy.deepcopy(checkpoint)
    pending["execution"].update({
        "phase": "pending_finalization",
        "completed_updates": 60,
        "production_clock": smallfit._production_clock(60),
    })
    smallfit._validate_smallfit_checkpoint_header(pending, recipe=recipe)
    check(True, "a clean update-60 pending-finalization checkpoint can resume only to write terminal evidence")
    changed = copy.deepcopy(pending)
    changed["execution"]["phase"] = "updates"
    expect_value_error(
        lambda: smallfit._validate_smallfit_checkpoint_header(changed, recipe=recipe),
        "resume rejects an ordinary update phase at 60 instead of reopening a terminal budget",
    )
    check(
        smallfit._worklists_dict(worklists)["initialization_training_ids"] == list(range(32))
        and smallfit._batch_schedule_as_lists(batches)[0] != list(range(4)),
        "checkpoint evidence preserves explicit first-32 initialization IDs and shuffled logical batches",
    )
    check(
        smallfit._resume_segments_after_restore(saved_resume_count=0, completed_updates=59) == 1
        and smallfit._resume_segments_after_restore(saved_resume_count=3, completed_updates=12) == 4
        and smallfit._resume_segments_after_restore(saved_resume_count=0, completed_updates=60) == 0
        and smallfit._resume_segments_after_restore(saved_resume_count=3, completed_updates=60) == 3,
        "a terminal-only resume preserves report-bound recovery evidence while any resumed update segment increments it",
    )


def stage_interrupted_resume_trajectory(contract: dict[str, Any], batches: tuple[tuple[int, ...], ...]) -> None:
    """Exercise the new checkpoint format's model/Adam/scheduler/RNG recovery."""

    worklists = bounded_b78716_smoke_worklists(contract)
    recipe = smallfit.smallfit_recipe_identity()
    acquisition = canonical_acquisition(contract)
    torch.manual_seed(123)
    uninterrupted_model = SpinrStyleINR()
    uninterrupted_optimizer = smallfit._new_smallfit_adam(uninterrupted_model)
    uninterrupted_scheduler = smallfit._new_smallfit_scheduler(uninterrupted_optimizer)
    synthetic_update(uninterrupted_model, uninterrupted_optimizer)
    normalization = {
        "training_mean_raw_power": 1.0,
        "initial_scale_ids": list(worklists.initialization_training_ids),
        "initial_output_scale": 0.1,
        "initial_scale_observed_energy": 1.0,
        "initial_scale_predicted_energy": 1.0,
        "normalization_source": "streamed_all_3200_parent_train_rows",
    }
    history = [fixture_milestone(0, 1.0)]
    update_records = [{
        "update": 1,
        "source_ids": list(batches[0]),
        "native_batch_objective": 1.0,
        "gradient_norm_before_clip": 1.0,
        "gradient_was_clipped": False,
        "parameter_delta_l2": 1.0,
        "update_seconds": 1.0,
    }]
    checkpoint = smallfit._checkpoint_state(
        model=uninterrupted_model,
        optimizer=uninterrupted_optimizer,
        scheduler=uninterrupted_scheduler,
        recipe=recipe,
        sealed_contract=contract,
        acquisition_identity=acquisition,
        worklists=worklists,
        batches=batches,
        normalization=normalization,
        execution=smallfit._execution_state(
            phase="interrupted_clean", completed_updates=1, elapsed_seconds=1.0, resume_count=0
        ),
        history=history,
        update_records=update_records,
        quadrature_records={"0": fixture_quadrature_record(worklists)},
    )
    checkpoint = copy.deepcopy(checkpoint)
    smallfit._validate_model_state_dict(checkpoint["model_state_dict"], SpinrStyleINR())
    smallfit._validate_adam_state(
        checkpoint["optimizer_state_dict"], SpinrStyleINR(), completed_updates=1
    )
    changed_adam = copy.deepcopy(checkpoint)
    changed_adam["optimizer_state_dict"]["param_groups"][0]["maximize"] = True
    expect_value_error(
        lambda: smallfit._validate_adam_state(
            changed_adam["optimizer_state_dict"], SpinrStyleINR(), completed_updates=1
        ),
        "pre-payload recovery rejects a behavior-changing Adam maximize flag instead of loading it",
    )
    smallfit._validate_bounded_evidence(
        history=checkpoint["history"],
        update_records=checkpoint["update_records"],
        quadrature_records=checkpoint["quadrature_records"],
        completed_updates=1,
        batches=batches,
        quadrature_source_ids=worklists.fit_training_ids[:4],
    )
    smallfit._validate_restoreable_rng(checkpoint["rng_state"])
    check(True, "pre-payload small-fit validation accepts finite fixed-layout model, Adam, RNG, cursor, and evidence")
    smallfit.restore_rng_state(checkpoint["rng_state"], require_complete=True)
    uninterrupted_rng_probe = torch.rand(8)
    synthetic_update(uninterrupted_model, uninterrupted_optimizer)
    uninterrupted_model_state = copy.deepcopy(uninterrupted_model.state_dict())
    uninterrupted_optimizer_state = copy.deepcopy(uninterrupted_optimizer.state_dict())
    uninterrupted_scheduler_state = copy.deepcopy(uninterrupted_scheduler.state_dict())

    torch.manual_seed(987)
    resumed_model = SpinrStyleINR()
    resumed_optimizer = smallfit._new_smallfit_adam(resumed_model)
    resumed_scheduler = smallfit._new_smallfit_scheduler(resumed_optimizer)
    smallfit._restore_checkpoint(
        checkpoint=checkpoint,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
    )
    resumed_rng_probe = torch.rand(8)
    synthetic_update(resumed_model, resumed_optimizer)
    check(
        nested_equal(resumed_model.state_dict(), uninterrupted_model_state)
        and nested_equal(resumed_optimizer.state_dict(), uninterrupted_optimizer_state)
        and nested_equal(resumed_scheduler.state_dict(), uninterrupted_scheduler_state)
        and resumed_scheduler.last_epoch == 0
        and torch.equal(resumed_rng_probe, uninterrupted_rng_probe),
        "interrupted/resumed small-fit state exactly preserves model, Adam, unadvanced scheduler, RNG, and trajectory",
    )
    check(
        checkpoint["worklists"] == smallfit._worklists_dict(worklists)
        and checkpoint["fit_batch_schedule"] == smallfit._batch_schedule_as_lists(batches)
        and checkpoint["normalization"] == normalization
        and checkpoint["quadrature_records"] == {"0": fixture_quadrature_record(worklists)},
        "small-fit recovery checkpoint preserves worklists, B=4 schedule, fixed scale, and quadrature evidence",
    )

    # Exercise the terminal-pair recovery route without opening radar data.
    # A pending latest checkpoint is authoritative until both independently
    # atomic terminal files exist; either half alone may be completed by a
    # literal resume, but cannot reopen any update.
    pending_terminal = copy.deepcopy(checkpoint)
    terminal_history = [
        fixture_milestone(0, 1.0),
        fixture_milestone(4, 0.9),
        fixture_milestone(20, 0.7),
        fixture_milestone(60, 0.5),
    ]
    terminal_updates = [
        {
            "update": index,
            "source_ids": list(batches[index - 1]),
            "native_batch_objective": 1.0,
            "gradient_norm_before_clip": 1.0,
            "gradient_was_clipped": False,
            "parameter_delta_l2": 1.0,
            "update_seconds": 1.0,
        }
        for index in range(1, 61)
    ]
    terminal_quadrature = {
        "0": fixture_quadrature_record(worklists),
        "60": fixture_quadrature_record(worklists),
    }
    pending_terminal["history"] = terminal_history
    pending_terminal["update_records"] = terminal_updates
    pending_terminal["quadrature_records"] = terminal_quadrature
    for adam_payload in pending_terminal["optimizer_state_dict"]["state"].values():
        step = adam_payload["step"]
        adam_payload["step"] = torch.full_like(step, 60) if torch.is_tensor(step) else 60
    pending_terminal["execution"] = smallfit._execution_state(
        phase="pending_finalization", completed_updates=60, elapsed_seconds=2.0, resume_count=0
    )
    smallfit._validate_resume_after_preflight(
        pending_terminal,
        recipe=recipe,
        sealed_contract=contract,
        acquisition_identity=acquisition,
        worklists=worklists,
        batches=batches,
        model_template=SpinrStyleINR(),
    )
    zero_observed_quadrature = fixture_quadrature_record(worklists)
    zero_signal_gate = smallfit.quadrature_difference_decision(
        difference_rms=0.0,
        reference_rms=0.0,
        label="frequency_signal",
    )
    zero_gradient_gate = smallfit.quadrature_difference_decision(
        difference_rms=0.0,
        reference_rms=0.0,
        label="directional_gradient",
    )
    for grid_name in ("g48", "g96"):
        zero_observed_quadrature[grid_name]["observed_energy"] = 0.0
        zero_observed_quadrature[grid_name]["directional_gradient"] = 0.0
    zero_observed_quadrature["signal_gate"] = zero_signal_gate
    zero_observed_quadrature["directional_gradient_gate"] = zero_gradient_gate
    zero_observed_quadrature["gate_pass"] = True
    smallfit._validate_quadrature_record(
        zero_observed_quadrature,
        expected_source_ids=worklists.fit_training_ids[:4],
    )
    check(
        zero_signal_gate["comparison_mode"] == "near_zero_absolute",
        "bounded quadrature evidence preserves its explicit zero-observed-signal absolute convention",
    )
    wrong_metric_views = copy.deepcopy(pending_terminal)
    wrong_metric_views["history"][0]["fixed_train"]["views"] = 15
    expect_value_error(
        lambda: smallfit._validate_resume_after_preflight(
            wrong_metric_views,
            recipe=recipe,
            sealed_contract=contract,
            acquisition_identity=acquisition,
            worklists=worklists,
            batches=batches,
            model_template=SpinrStyleINR(),
        ),
        "resume rejects a metric record that silently changes the frozen 16-view role size",
    )
    wrong_zero_reference = copy.deepcopy(pending_terminal)
    wrong_zero_reference["history"][0]["same_domain_zero_reference"]["fixed_train"][
        "coherent_relative_mse"
    ] = 0.5
    expect_value_error(
        lambda: smallfit._validate_resume_after_preflight(
            wrong_zero_reference,
            recipe=recipe,
            sealed_contract=contract,
            acquisition_identity=acquisition,
            worklists=worklists,
            batches=batches,
            model_template=SpinrStyleINR(),
        ),
        "resume rejects a same-domain zero reference that is not exactly unit coherent error",
    )
    wrong_scale = copy.deepcopy(pending_terminal)
    wrong_scale["normalization"]["initial_output_scale"] = 0.2
    expect_value_error(
        lambda: smallfit._validate_resume_after_preflight(
            wrong_scale,
            recipe=recipe,
            sealed_contract=contract,
            acquisition_identity=acquisition,
            worklists=worklists,
            batches=batches,
            model_template=SpinrStyleINR(),
        ),
        "resume rejects a gain that no longer follows the fixed first-32 scale equation",
    )
    wrong_quadrature_count = copy.deepcopy(pending_terminal)
    wrong_quadrature_count["quadrature_records"]["0"]["g48"]["signal_elements"] = 4
    expect_value_error(
        lambda: smallfit._validate_resume_after_preflight(
            wrong_quadrature_count,
            recipe=recipe,
            sealed_contract=contract,
            acquisition_identity=acquisition,
            worklists=worklists,
            batches=batches,
            model_template=SpinrStyleINR(),
        ),
        "resume rejects quadrature evidence that drops full-pair or full-frequency signal elements",
    )
    wrong_quadrature_observed = copy.deepcopy(pending_terminal)
    wrong_quadrature_observed["quadrature_records"]["0"]["g96"]["observed_energy"] = 2.0
    expect_value_error(
        lambda: smallfit._validate_resume_after_preflight(
            wrong_quadrature_observed,
            recipe=recipe,
            sealed_contract=contract,
            acquisition_identity=acquisition,
            worklists=worklists,
            batches=batches,
            model_template=SpinrStyleINR(),
        ),
        "resume rejects G48/G96 evidence with mismatched observed data",
    )
    wrong_signal_reference = copy.deepcopy(pending_terminal)
    wrong_signal_reference["quadrature_records"]["0"]["signal_gate"]["reference_rms"] = 1.0
    expect_value_error(
        lambda: smallfit._validate_resume_after_preflight(
            wrong_signal_reference,
            recipe=recipe,
            sealed_contract=contract,
            acquisition_identity=acquisition,
            worklists=worklists,
            batches=batches,
            model_template=SpinrStyleINR(),
        ),
        "resume rejects a signal quadrature denominator that is not the full-domain observed RMS",
    )
    wrong_directional_gate = copy.deepcopy(pending_terminal)
    wrong_directional_gate["quadrature_records"]["0"]["g96"]["directional_gradient"] = 0.5
    expect_value_error(
        lambda: smallfit._validate_resume_after_preflight(
            wrong_directional_gate,
            recipe=recipe,
            sealed_contract=contract,
            acquisition_identity=acquisition,
            worklists=worklists,
            batches=batches,
            model_template=SpinrStyleINR(),
        ),
        "resume rejects a directional quadrature gate detached from its recorded G48/G96 gradients",
    )
    wrong_quadrature_rows = copy.deepcopy(pending_terminal)
    wrong_quadrature_rows["quadrature_records"]["0"]["source_ids"] = [4, 5, 6, 7]
    expect_value_error(
        lambda: smallfit._validate_resume_after_preflight(
            wrong_quadrature_rows,
            recipe=recipe,
            sealed_contract=contract,
            acquisition_identity=acquisition,
            worklists=worklists,
            batches=batches,
            model_template=SpinrStyleINR(),
        ),
        "resume rejects quadrature evidence that is not the frozen first four selected training rows",
    )
    completed_terminal = copy.deepcopy(checkpoint)
    completed_terminal["history"] = copy.deepcopy(terminal_history)
    completed_terminal["update_records"] = copy.deepcopy(terminal_updates)
    completed_terminal["quadrature_records"] = copy.deepcopy(terminal_quadrature)
    for adam_payload in completed_terminal["optimizer_state_dict"]["state"].values():
        step = adam_payload["step"]
        adam_payload["step"] = torch.full_like(step, 60) if torch.is_tensor(step) else 60
    completed_terminal["execution"] = smallfit._execution_state(
        phase="completed", completed_updates=60, elapsed_seconds=2.0, resume_count=0
    )
    completed_terminal["engineering_status"] = smallfit.SMALLFIT_ENGINEERING_STATUS
    terminal_report = smallfit._build_report(
        worklists=worklists,
        history=terminal_history,
        update_records=terminal_updates,
        quadrature_records=terminal_quadrature,
        normalization=normalization,
        final_readout={
            "representation": "signed_real_sigma_midpoint_grid_readout",
            "grid_size": smallfit.SMALLFIT_TRAIN_GRID_SIZE,
            "support_min_m": [-smallfit.SPINR_STYLE_SUPPORT_M] * 3,
            "support_max_m": [smallfit.SPINR_STYLE_SUPPORT_M] * 3,
            "field_points": smallfit.SMALLFIT_TRAIN_GRID_SIZE ** 3,
            "sigma_abs_mean": 1.0,
            "sigma_abs_max": 1.0,
            "sigma_squared_mean": 1.0,
            "sigma_squared_integral_m3": 1.0,
            "interpretation": "native field-amplitude/energy readout; not a reconstruction-quality claim",
        },
        memory={
            "process_max_rss_bytes": 1,
            "peak_torch_allocated_bytes": 1,
            "peak_torch_reserved_bytes": 1,
        },
        elapsed_seconds=2.0,
        resume_count=0,
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        check(smallfit._empty_run_root(root), "an empty pre-checkpoint run root can safely restart fresh")
        stale_pid = str(10**12)
        stale_temp = root / f".checkpoint_latest.pth.tar.tmp.{stale_pid}"
        stale_temp.write_text("interrupted private atomic temp", encoding="utf-8")
        check(
            smallfit._empty_run_root(root),
            "a stale first latest-checkpoint temp is non-durable and can restart without deleting it",
        )
        final_path = root / "checkpoint_final.pth.tar"
        report_path = root / "smallfit_report.json"
        smallfit._atomic_torch_save(completed_terminal, final_path)
        smallfit._validate_partial_terminal_recovery(
            pending_terminal,
            final_checkpoint=final_path,
            report_path=report_path,
            recipe=recipe,
            sealed_contract=contract,
            acquisition_identity=acquisition,
            worklists=worklists,
            batches=batches,
            model_template=SpinrStyleINR(),
        )
        check(True, "a valid final-only terminal interruption can resume only to write its missing report")
        final_path.unlink()
        malformed_final = copy.deepcopy(completed_terminal)
        first_key = next(iter(malformed_final["model_state_dict"]))
        malformed_final["model_state_dict"][first_key] = malformed_final["model_state_dict"][first_key] + 1.0
        smallfit._atomic_torch_save(malformed_final, final_path)
        expect_value_error(
            lambda: smallfit._validate_partial_terminal_recovery(
                pending_terminal,
                final_checkpoint=final_path,
                report_path=report_path,
                recipe=recipe,
                sealed_contract=contract,
                acquisition_identity=acquisition,
                worklists=worklists,
                batches=batches,
                model_template=SpinrStyleINR(),
            ),
            "a terminal checkpoint with changed model evidence is rejected before pairing",
        )
        final_path.unlink()
        smallfit._atomic_json_save(terminal_report, report_path)
        smallfit._validate_partial_terminal_recovery(
            pending_terminal,
            final_checkpoint=final_path,
            report_path=report_path,
            recipe=recipe,
            sealed_contract=contract,
            acquisition_identity=acquisition,
            worklists=worklists,
            batches=batches,
            model_template=SpinrStyleINR(),
        )
        check(True, "a valid report-only terminal interruption can resume only to write its missing final checkpoint")
        report_path.unlink()
        prior_segment_pending = copy.deepcopy(pending_terminal)
        prior_segment_pending["execution"] = smallfit._execution_state(
            phase="pending_finalization", completed_updates=60, elapsed_seconds=2.0, resume_count=3
        )
        prior_segment_report = copy.deepcopy(terminal_report)
        prior_segment_report["clean_checkpoint_recovery"]["resume_segments_completed"] = 3
        smallfit._atomic_json_save(prior_segment_report, report_path)
        smallfit._validate_partial_terminal_recovery(
            prior_segment_pending,
            final_checkpoint=final_path,
            report_path=report_path,
            recipe=recipe,
            sealed_contract=contract,
            acquisition_identity=acquisition,
            worklists=worklists,
            batches=batches,
            model_template=SpinrStyleINR(),
        )
        check(
            True,
            "a report-only update-60 recovery remains resumable without mutating its prior update-segment counter",
        )
        report_path.unlink()
        malformed_report = copy.deepcopy(terminal_report)
        malformed_report["milestones"] = []
        smallfit._atomic_json_save(malformed_report, report_path)
        expect_value_error(
            lambda: smallfit._validate_partial_terminal_recovery(
                pending_terminal,
                final_checkpoint=final_path,
                report_path=report_path,
                recipe=recipe,
                sealed_contract=contract,
                acquisition_identity=acquisition,
                worklists=worklists,
                batches=batches,
                model_template=SpinrStyleINR(),
            ),
            "a terminal report with missing milestone evidence is rejected before pairing",
        )
        report_path.unlink()
        smallfit._atomic_json_save(terminal_report, report_path)
        smallfit._atomic_torch_save(completed_terminal, final_path)
        expect_value_error(
            lambda: smallfit._validate_partial_terminal_recovery(
                pending_terminal,
                final_checkpoint=final_path,
                report_path=report_path,
                recipe=recipe,
                sealed_contract=contract,
                acquisition_identity=acquisition,
                worklists=worklists,
                batches=batches,
                model_template=SpinrStyleINR(),
            ),
            "both terminal artifacts are terminal evidence and reject another resume",
        )
        before_terminal = copy.deepcopy(pending_terminal)
        before_terminal["execution"] = smallfit._execution_state(
            phase="updates", completed_updates=4, elapsed_seconds=1.0, resume_count=0
        )
        report_path.unlink()
        expect_value_error(
            lambda: smallfit._validate_partial_terminal_recovery(
                before_terminal,
                final_checkpoint=final_path,
                report_path=report_path,
                recipe=recipe,
                sealed_contract=contract,
                acquisition_identity=acquisition,
                worklists=worklists,
                batches=batches,
                model_template=SpinrStyleINR(),
            ),
            "a partial terminal artifact cannot reopen a pre-60 training trajectory",
        )


def stage_source_boundaries() -> None:
    driver_source = (ROOT / "rift" / "spinr_smoke_runtime.py").read_text(encoding="utf-8")
    adapter_source = (ROOT / "rift" / "spinr_style_b78716_smoke.py").read_text(encoding="utf-8")
    production_path = ROOT / "train_spinr_style.py"
    operator_path = ROOT / "rift" / "spinr_style.py"
    production_source = production_path.read_text(encoding="utf-8")
    production_tree = ast.parse(production_source, filename=str(production_path))
    operator_tree = ast.parse(operator_path.read_text(encoding="utf-8"), filename=str(operator_path))
    launcher_sources = [
        (ROOT / "slurm" / "validate_spinr_style_b78716_smallfit_g96_gauss2_v1.sbatch").read_text(
            encoding="utf-8"
        ),
    ]
    check(
        "build_bounded_sealed_raw_complex_views" in driver_source
        and "preflight_b787_development_inputs" in driver_source
        and "materialize_b787_development_views" not in driver_source,
        "the small-fit path preflights the canonical parent before its separate bounded payload adapter",
    )
    check(
        "normalization_training_ids" in adapter_source
        and "expected_energy_count = 3200 * 16 * 16 * 600" in adapter_source
        and "generic access is limited to selected fit and validation responses" in adapter_source
        and "initialization access is limited to parent train[:32]" in adapter_source,
        "the adapter records all worklists, streams the full normalizer, and separates selected versus initialization-only access",
    )
    imports_production_views = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "rift.spinr_style"
        and any(alias.name == "SealedRawComplexViews" for alias in node.names)
        for node in ast.walk(production_tree)
    )
    defines_production_view_class = any(
        isinstance(node, ast.ClassDef) and node.name == "SealedRawComplexViews"
        for node in ast.walk(operator_tree)
    )
    retains_800_update_clock = any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == "CANONICAL_UPDATES_PER_EPOCH"
            for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
        )
        and isinstance(node.value, ast.Constant)
        and node.value.value == 800
        for node in ast.walk(production_tree)
    )
    check(
        imports_production_views and defines_production_view_class and retains_800_update_clock,
        "the production trainer uses the shared sealed-view adapter and retains its 800-update epoch clock",
    )
    check(
        "scheduler.step(" not in driver_source
        and "build_bounded_sealed_raw_complex_views" in driver_source,
        "small-fit cycles do not advance the production cosine scheduler or reuse the full-cache adapter",
    )
    check(
        "_validate_resumed_normalization_against_materialized_b787(" in driver_source
        and "views.raw_training_mean_power()" in driver_source
        and "initial_output_scale=initial_scale" in driver_source
        and "field = network_field * _finite_positive(initial_output_scale" in driver_source,
        "resume replays the allowed all-3200/first-32 normalization evidence and the geometry readout applies its frozen gain",
    )
    check(
        driver_source.index("run_root.mkdir(parents=True, exist_ok=True)")
        > driver_source.index('quadrature_records["0"] = run_quadrature_diagnostic(')
        and "_validate_partial_terminal_recovery(" in driver_source,
        "the durable root is delayed until a zero-update boundary exists and partial terminal recovery stays explicit",
    )
    check(
        all(
            '[[ ! -e "$final_checkpoint" && ! -e "$report" && ! -e "$postflight" ]]' in launcher_source
            and '[[ -f "$report" && -f "$final_checkpoint" ]]' in launcher_source
            and 'resume_args=(--resume "$latest_checkpoint")' in launcher_source
            and launcher_source.index('if (( driver_status != 0 )); then')
            < launcher_source.index('python -B -u scripts/postflight_spinr_style_b78716_smallfit.py')
            for launcher_source in launcher_sources
        ),
        "the retained G96 launcher rejects terminal residue and gates postflight on a clean driver and both terminal files",
    )


def stage_postflight_boundary() -> None:
    report = {
        "schema": postflight.REPORT_SCHEMA,
        "engineering_status": postflight.ENGINEERING_STATUS,
        "lifecycle_complete": True,
        "production_clearance": False,
        "logical_update_count": 60,
        "production_clock": smallfit._production_clock(60),
        "memory": {
            "process_max_rss_bytes": 1024 * 1024,
            "peak_torch_allocated_bytes": 1024 * 1024,
            "peak_torch_reserved_bytes": 2 * 1024 * 1024,
        },
        "quadrature_records": {"0": {"gate_pass": False}, "60": {"gate_pass": False}},
        "fixed_train_error": {"strictly_fell": False},
        "finite_nonzero_updates": True,
    }
    outcome = postflight.build_postflight(
        report,
        time_process_rss_kib=1024,
        host_limit_gib=32,
        gpu_total_mib=48 * 1024,
        whole_job_rss_raw="2M",
    )
    check(
        outcome["overall_lifecycle_pass"] is True
        and outcome["production_clearance"] is False
        and outcome["checks"]["quadrature_gate_pass"] is False,
        "resource postflight can retain a failed quadrature result without falsely turning the bounded smoke into production clearance",
    )
    no_op = copy.deepcopy(report)
    no_op["finite_nonzero_updates"] = False
    rejected = postflight.build_postflight(
        no_op,
        time_process_rss_kib=1024,
        host_limit_gib=32,
        gpu_total_mib=48 * 1024,
        whole_job_rss_raw="2M",
    )
    check(
        rejected["overall_lifecycle_pass"] is False,
        "resource postflight refuses a lifecycle pass when any logical update was finite but ineffective",
    )


def main() -> int:
    contract, batches = stage_worklists_and_clock()
    stage_model_and_quadrature_conventions()
    stage_checkpoint_header(contract, batches)
    stage_interrupted_resume_trajectory(contract, batches)
    stage_source_boundaries()
    stage_postflight_boundary()
    print(f"SpINR-style B78716 small-fit validation passed: {CHECKS} checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
