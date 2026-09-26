#!/usr/bin/env python
"""CPU-only actual-entrypoint TERM/resume regression for GeRaF B7873200.

This deliberately uses a one-voxel synthetic cache and a tiny surrogate model;
it never opens the B787 archive or a radar-response payload.  The real
``train_geraf.main`` loop, AdamW optimizer, cosine scheduler,
dynamic-mask bank, atomic checkpoint writer, and checkpoint restore path all
    run unchanged.  It compares uninterrupted training to cooperative TERM before,
    during, and immediately after a scheduled validation, including the state
    restored before the next optimizer update.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import signal
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _validation_tmp_parent() -> Path:
    configured = os.environ.get("RIFT_VALIDATION_TMPDIR")
    parent = ROOT if configured is None else Path(configured).expanduser().resolve()
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        raise RuntimeError(f"no usable validation temporary parent: {parent}")
    return parent

_TRAINER_SPEC = importlib.util.spec_from_file_location(
    "train_geraf_b7873200_entrypoint_under_test", ROOT / "train_geraf.py"
)
if _TRAINER_SPEC is None or _TRAINER_SPEC.loader is None:
    raise RuntimeError("cannot load the corrected B7873200 trainer")
trainer = importlib.util.module_from_spec(_TRAINER_SPEC)
sys.modules[_TRAINER_SPEC.name] = trainer
_TRAINER_SPEC.loader.exec_module(trainer)


class Gates:
    def __init__(self) -> None:
        self.count = 0

    def check(self, condition: bool, detail: str) -> None:
        if not condition:
            raise AssertionError(detail)
        self.count += 1
        print(f"PASS {self.count:02d}: {detail}", flush=True)


class TinyGeRaF(torch.nn.Module):
    """A two-parameter replacement preserving the trainer's optimizer grouping."""

    def __init__(self, **_kwargs: object) -> None:
        super().__init__()
        self.sdf_network = torch.nn.Linear(1, 1, bias=False)
        self.other_parameter = torch.nn.Parameter(torch.tensor(0.125, dtype=torch.float32))


class Harness:
    def __init__(self, signal_mode: str | None) -> None:
        self.signal_mode = signal_mode
        self.signal_sent = False
        self.train_calls = 0
        self.events: list[tuple[str, int, str]] = []


def _args(checkpoint_dir: Path) -> SimpleNamespace:
    """A valid, tiny CPU invocation of the actual runner."""

    return SimpleNamespace(
        npz_path="synthetic-never-opened.npz",
        role_manifest="synthetic-never-opened.json",
        cache_root=str(checkpoint_dir / "synthetic_cache_never_opened"),
        checkpoint_dir=str(checkpoint_dir),
        seed=trainer.B787_3200_SEED,
        scene_extent=0.15,
        n_azimuth=1,
        n_elevation=1,
        n_depth=1,
        aperture_scale=1.0,
        phase_sign=-1.0,
        sdf_levels=10,
        sdf_hidden_dim=1,
        sdf_layers=1,
        sdf_skip_layer=-1,
        sdf_softplus_beta=100.0,
        reflectivity_levels=0,
        reflectivity_hidden_dim=1,
        reflectivity_layers=1,
        reflectivity_output_activation="softplus",
        reflectivity_softplus_beta=1.0,
        init_tx_amplitude=1.0,
        init_inv_s=64.0,
        learnable_inv_s=True,
        lensless_correction=True,
        detach_start_cdf=True,
        directional_exponent=1.0,
        min_distance=1.0e-6,
        steps=2,
        sdf_lr=1.0e-2,
        other_lr=2.0e-2,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1.0e-8,
        cosine_min_lr=0.0,
        gradient_clip_norm=0.0,
        mask_high_threshold=10.0,
        mask_low_ratio=0.1,
        mask_low_threshold=0.0,
        compute_dtype="float32",
        oversample=2,
        kernel_width=4,
        pair_chunk=1,
        point_chunk=1,
        device="cpu",
        validation_every=1,
        checkpoint_every=100,
        checkpoint_seconds=3600.0,
        log_every=100,
        resume=True,
        resume_path=None,
    )


def _synthetic_acquisition_record(*, changed_tx: bool = False) -> dict[str, object]:
    metadata = {
        "radar_fc_hz": 10.0e9,
        "radar_bandwidth_hz": 3.0e9,
        "num_adc_samples": 600,
        "num_chirps_cpi": 1,
    }
    frequency_hz = (metadata["radar_fc_hz"] - metadata["radar_bandwidth_hz"] / 2.0) + (
        np.arange(600, dtype=np.float64) * metadata["radar_bandwidth_hz"] / 600
    )
    tx_pos = np.zeros((10_000, 16, 3), dtype=np.float64)
    if changed_tx:
        tx_pos[321, 4, 2] = 0.125
    return {
        "schema": trainer.B787_3200_ACQUISITION_SCHEMA,
        "version": 1,
        "response_shape": np.asarray((10_000, 16, 16, 1, 600), dtype=np.int64),
        "response_dtype": "complex64",
        "metadata_json": json.dumps(metadata, sort_keys=True, separators=(",", ":")),
        "frequency_hz": frequency_hz,
        "viewpoint_positions": np.zeros((10_000, 3), dtype=np.float64),
        "tx_pos": tx_pos,
        "rx_pos": np.zeros((10_000, 16, 3), dtype=np.float64),
    }


def _synthetic_cache(*, changed_acquisition: bool = False) -> SimpleNamespace:
    """Only fields consumed by the unmodified entrypoint; no response accessor."""

    arrays = SimpleNamespace(
        metadata={
            "radar_fc_hz": 10.0e9,
            "radar_bandwidth_hz": 3.0e9,
            "num_adc_samples": 600,
        }
    )
    return SimpleNamespace(
        source=SimpleNamespace(arrays=arrays),
        recipe={"schema": "synthetic_entrypoint_recipe_v1", "version": 1},
        stats={"schema": "synthetic_entrypoint_stats_v1", "peak": 1.0},
        target_manifest={"schema": "synthetic_entrypoint_manifest_v1", "roles": [17, 23]},
        acquisition_record=_synthetic_acquisition_record(changed_tx=changed_acquisition),
        sealed_identity={"schema": "synthetic_entrypoint_roles_v1", "train": [17], "validation": [23]},
        train_indices=(17,),
        validation_indices=(23,),
        grid_shape=(1, 1, 1),
        geraf_mf_magnitude_peak=1.0,
        effective_pairs_per_plane=1,
    )


def _fake_load_training_view(
    harness: Harness,
    cache: SimpleNamespace,
    index: int,
    role: str,
    _args: SimpleNamespace,
    device: torch.device,
) -> Any:
    if role != "train" or int(index) not in cache.train_indices:
        raise AssertionError("synthetic entrypoint loader must receive only its training role")
    harness.train_calls += 1
    harness.events.append(("train", harness.train_calls, "start"))
    return trainer.TrainingView(
        int(index),
        torch.full(cache.grid_shape, 0.3, dtype=torch.float32, device=device),
        None,
        torch.empty((0, 3), dtype=torch.float32, device=device),
        torch.empty((0, 3), dtype=torch.float32, device=device),
    )


def _fake_predict(harness: Harness, model: TinyGeRaF, cache: SimpleNamespace) -> torch.Tensor:
    value = torch.nn.functional.softplus(model.sdf_network.weight.reshape(1) + model.other_parameter)
    return value.reshape(cache.grid_shape)


def _fake_validate(harness: Harness, _model: TinyGeRaF, _cache: SimpleNamespace) -> tuple[dict[str, float], bool]:
    if harness.train_calls not in (1, 2):
        raise AssertionError("synthetic validation must follow one of the two expected optimizer updates")
    if (
        harness.signal_mode == "before_validation"
        and not harness.signal_sent
        and harness.train_calls == 1
    ):
        harness.signal_sent = True
        harness.events.append(("validate", harness.train_calls, "before_first_view"))
        trainer._request_stop(signal.SIGTERM, None)
        return {"views": 0.0, "voxels": 0.0}, False
    if (
        harness.signal_mode == "during_validation"
        and not harness.signal_sent
        and harness.train_calls == 1
    ):
        harness.signal_sent = True
        # Model one completed validation view before the cooperative TERM.  The
        # replacement only exercises entrypoint control flow; it deliberately
        # does not claim to test the real validation data-access path.
        harness.events.append(("validate", harness.train_calls, "first_view_complete"))
        trainer._request_stop(signal.SIGTERM, None)
        harness.events.append(("validate", harness.train_calls, "interrupted"))
        return {"views": 1.0, "voxels": 1.0}, False
    mse = 0.25 if harness.train_calls == 1 else 0.5
    metrics = {
        "views": 1.0,
        "voxels": 1.0,
        "mf_magnitude_mse": mse,
        "mf_magnitude_rmse": float(np.sqrt(mse)),
        "mf_magnitude_relative_mse": mse,
        "mf_magnitude_psnr_db": 0.0,
    }
    harness.events.append(("validate", harness.train_calls, "complete"))
    if (
        harness.signal_mode == "after_terminal_validation"
        and not harness.signal_sent
        and harness.train_calls == 2
    ):
        # Model a clean TERM after validation generated complete terminal
        # metrics, but before the runner can materialize checkpoint_final.
        harness.signal_sent = True
        trainer._request_stop(signal.SIGTERM, None)
        harness.events.append(("signal", harness.train_calls, "after_terminal_validation"))
    return metrics, True


def _fake_loss(predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weight = valid.to(dtype=predicted.dtype)
    return ((predicted - target).square() * weight).sum() / weight.sum().clamp_min(1.0)


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _assert_same(left: Any, right: Any, label: str) -> None:
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)):
            raise AssertionError(f"checkpoint mismatch at {label}")
        return
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not (isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)):
            raise AssertionError(f"checkpoint mismatch at {label}")
        return
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not (isinstance(left, Mapping) and isinstance(right, Mapping) and set(left) == set(right)):
            raise AssertionError(f"checkpoint mapping mismatch at {label}")
        for key in left:
            _assert_same(left[key], right[key], f"{label}.{key}")
        return
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        if not (
            isinstance(left, (tuple, list))
            and isinstance(right, (tuple, list))
            and len(left) == len(right)
        ):
            raise AssertionError(f"checkpoint sequence mismatch at {label}")
        for number, (first, second) in enumerate(zip(left, right)):
            _assert_same(first, second, f"{label}[{number}]")
        return
    if left != right:
        raise AssertionError(f"checkpoint mismatch at {label}: {left!r} != {right!r}")


def _require_valid_elapsed_seconds(value: Any, label: str) -> None:
    """Accept timing as telemetry, while still rejecting invalid timing values."""

    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise AssertionError(f"checkpoint timing is not numeric at {label}: {value!r}")
    elapsed = float(value)
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise AssertionError(f"checkpoint timing is invalid at {label}: {value!r}")


def _assert_same_last_train(left: Any, right: Any, label: str) -> None:
    """Compare scientific training state exactly while treating timing as telemetry."""

    if left is None or right is None:
        _assert_same(left, right, label)
        return
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise AssertionError(f"last training record is not a mapping at {label}")
    required = {
        "step",
        "view_index",
        "loss",
        "valid_fraction",
        "gradient_norm_before_clip",
        "sdf_lr",
        "other_lr",
        "seconds",
    }
    if set(left) != required or set(right) != required:
        raise AssertionError(f"last training record has unexpected fields at {label}")
    for key in sorted(required):
        field = f"{label}.{key}"
        if key == "seconds":
            _require_valid_elapsed_seconds(left[key], field)
            _require_valid_elapsed_seconds(right[key], field)
            continue
        if key == "gradient_norm_before_clip":
            try:
                left_value = float(left[key])
                right_value = float(right[key])
            except (TypeError, ValueError) as exc:
                raise AssertionError(f"gradient diagnostic is not numeric at {field}") from exc
            left_nan = math.isnan(left_value)
            right_nan = math.isnan(right_value)
            # With clipping disabled, the trainer intentionally records this
            # diagnostic as NaN.  Permit only that paired field-specific case.
            if left_nan or right_nan:
                if left_nan and right_nan:
                    continue
                raise AssertionError(f"checkpoint mismatch at {field}")
            if not math.isfinite(left_value) or not math.isfinite(right_value):
                raise AssertionError(f"gradient diagnostic is non-finite at {field}")
        _assert_same(left[key], right[key], field)


def _assert_equivalent_checkpoint(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    for field in (
        "step",
        "complete",
        "best_val_mse",
        "history",
        "pending_validation_step",
        "last_train",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "rng_state",
        "dynamic_mask_bank",
    ):
        if field == "last_train":
            _assert_same_last_train(left[field], right[field], field)
        else:
            _assert_same(left[field], right[field], field)


def _assert_last_train_comparator_regression(gates: Gates) -> None:
    """Keep elapsed telemetry out of deterministic-state equality only."""

    baseline = {
        "step": 2,
        "view_index": 17,
        "loss": 0.125,
        "valid_fraction": 1.0,
        "gradient_norm_before_clip": math.nan,
        "sdf_lr": 0.005,
        "other_lr": 0.01,
        "seconds": 0.01,
    }
    telemetry_changed = dict(baseline)
    telemetry_changed["seconds"] = 2.0
    _assert_same_last_train(baseline, telemetry_changed, "last_train")
    gates.check(
        True,
        "last-training comparator ignores finite elapsed telemetry and permits only paired disabled-clipping diagnostic NaN",
    )
    invalid_timing = dict(telemetry_changed)
    invalid_timing["seconds"] = -1.0
    try:
        _assert_same_last_train(baseline, invalid_timing, "last_train")
    except AssertionError:
        gates.check(True, "last-training comparator rejects negative elapsed telemetry")
    else:
        raise AssertionError("last-training comparator must reject negative elapsed telemetry")
    scientific_changed = dict(telemetry_changed)
    scientific_changed["loss"] = 0.5
    try:
        _assert_same_last_train(baseline, scientific_changed, "last_train")
    except AssertionError:
        gates.check(True, "last-training comparator still rejects a changed scientific loss")
    else:
        raise AssertionError("last-training comparator must reject a changed scientific loss")


def _run_main(checkpoint_dir: Path, harness: Harness, *, changed_acquisition: bool = False) -> None:
    args = _args(checkpoint_dir)
    cache = _synthetic_cache(changed_acquisition=changed_acquisition)
    original = {
        "parse_args": trainer.parse_args,
        "verify_cache": trainer.verify_prepared_b7873200_cache,
        "model": trainer.GeRaFModel,
        "load_view": trainer.load_training_view,
        "predict": trainer.predict_normalized_magnitude,
        "validate": trainer.validate,
        "loss": trainer.masked_magnitude_l2,
        "kvector": trainer.get_kvector,
    }
    original_handlers = {kind: signal.getsignal(kind) for kind in (signal.SIGTERM, signal.SIGINT)}
    trainer._STOP_REQUESTED = False
    trainer._STOP_SIGNAL = None
    try:
        trainer.parse_args = lambda: copy.deepcopy(args)
        trainer.verify_prepared_b7873200_cache = lambda _args: cache
        trainer.GeRaFModel = TinyGeRaF
        trainer.load_training_view = lambda cache, index, role, run_args, device: _fake_load_training_view(
            harness, cache, index, role, run_args, device
        )
        trainer.predict_normalized_magnitude = lambda model, view, _frequencies, _kvector, cache, _args, create_graph: _fake_predict(
            harness, model, cache
        )
        trainer.validate = lambda model, cache, _frequencies, _kvector, _args, _device: _fake_validate(
            harness, model, cache
        )
        trainer.masked_magnitude_l2 = _fake_loss
        trainer.get_kvector = lambda frequencies, _cc: torch.zeros_like(frequencies)
        try:
            trainer.main()
        except SystemExit as exc:
            if exc.code != trainer.CLEAN_STOP_EXIT_CODE:
                raise
    finally:
        trainer.parse_args = original["parse_args"]
        trainer.verify_prepared_b7873200_cache = original["verify_cache"]
        trainer.GeRaFModel = original["model"]
        trainer.load_training_view = original["load_view"]
        trainer.predict_normalized_magnitude = original["predict"]
        trainer.validate = original["validate"]
        trainer.masked_magnitude_l2 = original["loss"]
        trainer.get_kvector = original["kvector"]
        for kind, handler in original_handlers.items():
            signal.signal(kind, handler)


def _run_interrupted_and_resumed(root: Path, mode: str) -> tuple[Harness, Mapping[str, Any], Mapping[str, Any]]:
    harness = Harness(mode)
    _run_main(root, harness)
    latest_path = root / "checkpoint_latest.pth.tar"
    interrupted = _load_checkpoint(latest_path)
    if interrupted["step"] != 1 or interrupted["pending_validation_step"] != 1:
        raise AssertionError("TERM run must save step-one pending validation before returning")
    if interrupted["history"]:
        raise AssertionError("incomplete step-one validation must not append history")
    if mode == "before_validation":
        if ("validate", 1, "before_first_view") not in harness.events:
            raise AssertionError("before-validation TERM must arrive after update one, before its first validation view")
        if ("validate", 1, "first_view_complete") in harness.events:
            raise AssertionError("before-validation TERM must not model a completed validation view")
    elif mode == "during_validation":
        if ("validate", 1, "first_view_complete") not in harness.events:
            raise AssertionError("during-validation TERM must model a completed first validation view")
        if ("validate", 1, "interrupted") not in harness.events:
            raise AssertionError("during-validation TERM must stop before validation can commit")
    else:
        raise AssertionError(f"unknown TERM mode {mode!r}")
    events_before_resume = len(harness.events)
    harness.signal_mode = None
    _run_main(root, harness)
    resumed = _load_checkpoint(root / "checkpoint_final.pth.tar")
    resume_events = harness.events[events_before_resume:]
    if not resume_events or resume_events[0] != ("validate", 1, "complete"):
        raise AssertionError("resumed runner must complete pending validation before its next update")
    if ("train", 2, "start") not in resume_events:
        raise AssertionError("resumed runner did not reach the second optimizer update")
    if resume_events.index(("validate", 1, "complete")) > resume_events.index(("train", 2, "start")):
        raise AssertionError("resumed runner advanced before completing step-one validation")
    return harness, interrupted, resumed


def _run_terminal_validation_interrupted_and_repaired(
    root: Path,
) -> tuple[Harness, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Exercise TERM after terminal metrics and idempotent artifact repair."""

    harness = Harness("after_terminal_validation")
    _run_main(root, harness)
    final_path = root / "checkpoint_final.pth.tar"
    latest_path = root / "checkpoint_latest.pth.tar"
    if not final_path.is_file() or not latest_path.is_file():
        raise AssertionError("TERM after terminal validation must materialize final and latest artifacts")
    signaled_final = _load_checkpoint(final_path)
    signaled_latest = _load_checkpoint(latest_path)
    for checkpoint in (signaled_final, signaled_latest):
        if (
            checkpoint["step"] != 2
            or checkpoint["complete"] is not True
            or checkpoint["pending_validation_step"] is not None
            or [row["step"] for row in checkpoint["history"]] != [1, 2]
            or checkpoint["stop_reason"] is not None
        ):
            raise AssertionError("terminal TERM must leave fully finalized terminal artifacts")
    if ("signal", 2, "after_terminal_validation") not in harness.events:
        raise AssertionError("terminal TERM must arrive after complete terminal validation metrics")

    # Model a legacy/interrupted artifact pair that preserved the complete
    # latest checkpoint but lost checkpoint_final.  A completed resume must
    # recreate both artifacts without another training update or validation.
    final_path.unlink()
    events_before_repair = len(harness.events)
    train_calls_before_repair = harness.train_calls
    harness.signal_mode = None
    _run_main(root, harness)
    repaired_final = _load_checkpoint(final_path)
    repaired_latest = _load_checkpoint(latest_path)
    if harness.events[events_before_repair:] or harness.train_calls != train_calls_before_repair:
        raise AssertionError("completed-artifact repair must not run another update or validation")

    events_before_idempotence = len(harness.events)
    _run_main(root, harness)
    if harness.events[events_before_idempotence:] or harness.train_calls != train_calls_before_repair:
        raise AssertionError("a second completed resume must remain artifact-idempotent")
    return harness, signaled_final, repaired_final, repaired_latest


def _assert_resume_mutation_rejected(
    gates: Gates,
    root: Path,
    label: str,
    mutate: object,
    *,
    from_completed_run: bool = False,
) -> None:
    """Prove bad resume control state is rejected before another train-view read."""

    harness = Harness(None if from_completed_run else "before_validation")
    _run_main(root, harness)
    checkpoint_path = (
        root / "checkpoint_final.pth.tar" if from_completed_run else root / "checkpoint_latest.pth.tar"
    )
    checkpoint = dict(_load_checkpoint(checkpoint_path))
    mutate(checkpoint)  # type: ignore[operator]
    torch.save(checkpoint, checkpoint_path)
    train_calls_before_resume = harness.train_calls
    harness.signal_mode = None
    try:
        _run_main(root, harness)
    except ValueError:
        gates.check(
            harness.train_calls == train_calls_before_resume,
            f"{label} is rejected before another training-view load",
        )
    else:
        raise AssertionError(f"{label} must be rejected during checkpoint resume")


def main() -> None:
    gates = Gates()
    _assert_last_train_comparator_regression(gates)
    with tempfile.TemporaryDirectory(
        prefix=".geraf_b7873200_entrypoint_", dir=_validation_tmp_parent()
    ) as temporary:
        root = Path(temporary)
        baseline_root = root / "uninterrupted"
        baseline_harness = Harness(None)
        _run_main(baseline_root, baseline_harness)
        baseline_final = _load_checkpoint(baseline_root / "checkpoint_final.pth.tar")
        baseline_best = _load_checkpoint(baseline_root / "checkpoint_best.pth.tar")
        gates.check(
            baseline_final["complete"] is True
            and baseline_final["pending_validation_step"] is None
            and [row["step"] for row in baseline_final["history"]] == [1, 2]
            and baseline_final["best_val_mse"] == 0.25,
            "the unmodified entrypoint writes complete real checkpoints with two validation rows",
        )
        identity = trainer.run_identity(_args(baseline_root), _synthetic_cache(), trainer.model_config_from_args(_args(baseline_root)))
        cadence_changed = _args(baseline_root)
        cadence_changed.validation_every = 2
        gates.check(
            trainer.run_identity(cadence_changed, _synthetic_cache(), trainer.model_config_from_args(cadence_changed)) != identity,
            "validation cadence is part of the sealed run identity and cannot silently change at resume",
        )

        for mode in ("before_validation", "during_validation"):
            mode_root = root / mode
            harness, interrupted, resumed_final = _run_interrupted_and_resumed(mode_root, mode)
            resumed_best = _load_checkpoint(mode_root / "checkpoint_best.pth.tar")
            gates.check(
                interrupted["complete"] is False
                and interrupted["pending_validation_step"] == 1
                and resumed_final["complete"] is True
                and resumed_final["pending_validation_step"] is None,
                f"TERM {mode} saves pending validation and real checkpoint restoration clears it",
            )
            _assert_equivalent_checkpoint(baseline_final, resumed_final)
            _assert_equivalent_checkpoint(baseline_best, resumed_best)
            gates.check(
                True,
                f"TERM {mode} matches uninterrupted model, optimizer, scheduler, sampler, mask, RNG, history, and best selection",
            )
            gates.check(
                ("validate", 1, "complete") in harness.events
                and harness.events.count(("validate", 1, "complete")) == 1,
                f"TERM {mode} commits the recovered intermediate validation exactly once before update two",
            )
        terminal_harness, signaled_final, repaired_final, repaired_latest = (
            _run_terminal_validation_interrupted_and_repaired(root / "after_terminal_validation")
        )
        terminal_best = _load_checkpoint(
            root / "after_terminal_validation" / "checkpoint_best.pth.tar"
        )
        _assert_equivalent_checkpoint(baseline_final, signaled_final)
        _assert_equivalent_checkpoint(baseline_final, repaired_final)
        _assert_equivalent_checkpoint(baseline_final, repaired_latest)
        _assert_equivalent_checkpoint(baseline_best, terminal_best)
        gates.check(
            terminal_harness.train_calls == 2
            and terminal_harness.events.count(("validate", 1, "complete")) == 1
            and terminal_harness.events.count(("validate", 2, "complete")) == 1,
            "TERM after terminal validation preserves uninterrupted state and needs no extra update or validation",
        )
        gates.check(
            signaled_final["stop_reason"] is None
            and repaired_final["stop_reason"] is None
            and repaired_latest["stop_reason"] is None,
            "terminal artifact materialization and completed-resume repair normalize durable completion state",
        )
        mismatch_root = root / "changed_acquisition"
        mismatch_harness = Harness("before_validation")
        _run_main(mismatch_root, mismatch_harness)
        mismatch_harness.signal_mode = None
        try:
            _run_main(mismatch_root, mismatch_harness, changed_acquisition=True)
        except ValueError as exc:
            gates.check(
                "different calibrated poses, metadata, or frequency grid" in str(exc),
                "checkpoint resume rejects a changed direct acquisition record before another update",
            )
        else:
            raise AssertionError("checkpoint resume must reject a changed direct acquisition record")

        _assert_resume_mutation_rejected(
            gates,
            root / "bad_step",
            "an out-of-range optimizer step",
            lambda checkpoint: checkpoint.__setitem__("step", 3),
        )
        _assert_resume_mutation_rejected(
            gates,
            root / "bad_history",
            "a malformed validation-history row",
            lambda checkpoint: checkpoint.__setitem__("history", [{"step": 1}]),
        )
        _assert_resume_mutation_rejected(
            gates,
            root / "bad_pending",
            "a pending validation bound to a different update",
            lambda checkpoint: checkpoint.__setitem__("pending_validation_step", 0),
        )
        _assert_resume_mutation_rejected(
            gates,
            root / "bad_best",
            "a best metric without committed validation history",
            lambda checkpoint: checkpoint.__setitem__("best_val_mse", 0.25),
        )
        _assert_resume_mutation_rejected(
            gates,
            root / "bad_complete",
            "a complete flag with pending validation",
            lambda checkpoint: checkpoint.__setitem__("complete", True),
        )
        _assert_resume_mutation_rejected(
            gates,
            root / "bad_optimizer",
            "an optimizer group whose epsilon changes",
            lambda checkpoint: checkpoint["optimizer_state_dict"]["param_groups"][0].__setitem__("eps", 1.0e-3),
        )
        _assert_resume_mutation_rejected(
            gates,
            root / "bad_scheduler",
            "a scheduler horizon that differs from the sealed run",
            lambda checkpoint: checkpoint["scheduler_state_dict"].__setitem__("T_max", 1),
        )
        _assert_resume_mutation_rejected(
            gates,
            root / "deleted_due_history",
            "a deleted earlier due-validation row",
            lambda checkpoint: (
                checkpoint.__setitem__("history", checkpoint["history"][1:]),
                checkpoint.__setitem__("complete", False),
            ),
            from_completed_run=True,
        )
    print(f"GeRaF B7873200 actual-entrypoint regression passed: {gates.count} checks.", flush=True)


if __name__ == "__main__":
    main()
