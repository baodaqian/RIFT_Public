#!/usr/bin/env python
"""Focused CPU validation for the adaptive AirSAS refinement diagnostic."""

from __future__ import annotations

import ast
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_sas
from rift.rift_sas import AdaptiveRIFTSASField, ComplexSHSonarField
from scripts.diagnose_sas_refinement import (
    FIXED_SOURCE_IDS,
    INTERVENTION_LABELS,
    POST_UPDATE_STEPS,
    RefinementDiagnosticObserver,
)


torch.set_num_threads(1)


def make_tempdir(prefix: str) -> str:
    root = PROJECT_ROOT / "tmp"
    root.mkdir(parents=True, exist_ok=True)
    del prefix
    return str(root)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def nested_equal(first, second) -> bool:
    if torch.is_tensor(first) or torch.is_tensor(second):
        return torch.is_tensor(first) and torch.is_tensor(second) and torch.equal(first, second)
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        return isinstance(first, np.ndarray) and isinstance(second, np.ndarray) and np.array_equal(first, second)
    if isinstance(first, dict) or isinstance(second, dict):
        return (
            isinstance(first, dict)
            and isinstance(second, dict)
            and set(first) == set(second)
            and all(nested_equal(first[key], second[key]) for key in first)
        )
    if isinstance(first, (list, tuple)) or isinstance(second, (list, tuple)):
        return (
            isinstance(first, type(second))
            and len(first) == len(second)
            and all(nested_equal(a, b) for a, b in zip(first, second))
        )
    return first == second


class WeightTable:
    """Small deterministic weight source; avoids allocating a real cache."""

    def __getitem__(self, _index):
        return np.ones(326, dtype=np.complex64)


def make_model_and_optimizer(max_degree: int = 3):
    scene = train_sas.AdaptivePointSHScene.from_regular_grid(
        1,
        1.0,
        "cpu",
        max_degree=max_degree,
        init_degree=0,
        init_scale=0.0,
        capacity=8,
        compact_sh_eval=True,
    )
    with torch.no_grad():
        scene.w_re[0, 0] = 0.7
        scene.w_im[0, 0] = -0.2
    field = AdaptiveRIFTSASField(scene, raster_granularity=4, extent=1.0, query_chunk=128)
    model = ComplexSHSonarField(
        field,
        torch.tensor([-1.0, -1.0, -1.0]),
        torch.tensor([1.0, 1.0, 1.0]),
        sh_degree=max_degree,
        query_chunk=128,
    )
    calibration = train_sas.LogPolarCalibration()
    with torch.no_grad():
        calibration.initialized.fill_(True)
    optimizer = train_sas._optimizer_for_model(
        model,
        calibration,
        SimpleNamespace(coefficient_lr=1.0e-2, position_lr=1.0e-2, lr=1.0e-2),
    )
    return model, calibration, optimizer


def make_observer_fixture():
    model, calibration, optimizer = make_model_and_optimizer()
    rng = np.random.default_rng(42)
    cache = SimpleNamespace(
        num_bins=326,
        source_ids=np.arange(50_000, dtype=np.int64),
        weights=WeightTable(),
        train_indices=np.arange(16, dtype=np.int64),
        validation_indices=FIXED_SOURCE_IDS.copy(),
        test_indices=np.empty(0, dtype=np.int64),
        has_explicit_splits=True,
        manifest={},
    )
    args = SimpleNamespace(
        model="adaptive_rift_sas",
        steps=1010,
        eval_pings=8,
        max_bins=4,
        num_rays=4,
        opacity_scale=1.0,
        lambertian_ratio=0.0,
        normal_step=0.1,
        signal_scale=1.0,
        beamwidth_deg=30.0,
        opacity_normalize=False,
        sh_direction="rx_to_point",
    )
    state = {"step": 700, "rng_state": copy.deepcopy(rng.bit_generator.state)}
    return SimpleNamespace(
        model=model,
        calibration=calibration,
        optimizer=optimizer,
        rng=rng,
        cache=cache,
        args=args,
        state=state,
    )


def fake_signal_from_tensors(
    w_re: torch.Tensor,
    w_im: torch.Tensor,
    delta_raw: torch.Tensor,
    active: torch.Tensor,
    log_mag: torch.Tensor,
    phase: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = torch.arange(1, w_re.shape[1] + 1, dtype=w_re.dtype, device=w_re.device)
    real = (w_re[active] * weights).sum() + 0.07 * delta_raw[active].sum()
    imag = (w_im[active] * weights).sum() + 0.11 * delta_raw[active].sum()
    raw = torch.complex(real.expand(count), imag.expand(count))
    gain = torch.polar(torch.exp(log_mag), phase)
    return gain * raw, raw


def fixed_fake_render_factory(model, calibration):
    target_holder: dict[str, torch.Tensor] = {}
    expected_rel_mse = 1.369373

    def fake_render(_model, _calibration, _cache, _ping, bins, _args, _device, **_kwargs):
        scene = model.underlying_scene
        predicted, raw = fake_signal_from_tensors(
            scene.w_re,
            scene.w_im,
            scene.delta_raw,
            scene.active_mask,
            calibration.log_mag,
            calibration.phase,
            len(bins),
        )
        if "target" not in target_holder:
            target_holder["target"] = predicted.detach() / (1.0 + expected_rel_mse ** 0.5)
        target = target_holder["target"].to(predicted)
        return torch.zeros((), dtype=torch.float32), train_sas.metric_record(predicted, target), {
            "calibration_raw": raw,
            "calibration_target": target,
            "calibration_predicted": predicted,
        }

    return fake_render


def training_fake_render(_model, _calibration, _cache, _ping, bins, _args, _device, **_kwargs):
    scene = _model.underlying_scene
    predicted, raw = fake_signal_from_tensors(
        scene.w_re,
        scene.w_im,
        scene.delta_raw,
        scene.active_mask,
        _calibration.log_mag,
        _calibration.phase,
        len(bins),
    )
    target = torch.ones_like(predicted)
    loss = torch.nn.functional.mse_loss(predicted.real, target.real) + torch.nn.functional.mse_loss(
        predicted.imag, target.imag
    )
    return loss, train_sas.metric_record(predicted, target), {
        "calibration_raw": raw,
        "calibration_target": target,
        "calibration_predicted": predicted,
        "transmittance": torch.ones((len(bins), 1), dtype=torch.float32),
        "lambertian": torch.ones((len(bins), 1), dtype=torch.float32),
        "actual_rays": 1,
    }


def observer_context(fixture):
    return {
        "model": fixture.model,
        "calibration": fixture.calibration,
        "optimizer": fixture.optimizer,
        "rng": fixture.rng,
        "cache": fixture.cache,
        "args": fixture.args,
        "device": torch.device("cpu"),
        "state": fixture.state,
        "start": 700,
        "best_val": 1.0,
        "history": [],
        "train_indices": fixture.cache.train_indices,
        "validation_indices": fixture.cache.validation_indices,
        "test_indices": fixture.cache.test_indices,
    }


def test_default_path_contract() -> None:
    tree = ast.parse(Path(train_sas.__file__).read_text(encoding="utf-8"))
    main_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    check(
        main_node.args.kwonlyargs[-1].arg == "diagnostic_observer",
        "main must expose the requested keyword-only diagnostic observer",
    )
    train_sas._diagnostic_hook(None, "missing_hook", impossible=True)
    print("PASS default observer dispatch remains opt-in")


def test_stage_order_and_fixed_cohort():
    fixture = make_observer_fixture()
    observer = RefinementDiagnosticObserver(
        make_tempdir("sas_diag_stage_"),
        expected_fixed_val_rel_mse=1.369373,
    )
    with mock.patch.object(
        train_sas, "render_one", side_effect=fixed_fake_render_factory(fixture.model, fixture.calibration)
    ):
        observer.on_restore(**observer_context(fixture))
        check(
            observer.fixed_ping_indices.tolist() == FIXED_SOURCE_IDS.tolist(),
            "fixed cohort mapping changed",
        )
        check(
            np.array_equal(observer.fixed_bins, np.arange(326, dtype=np.int64)),
            "fixed bins are not exactly 0..325",
        )

        with torch.no_grad():
            scene = fixture.model.underlying_scene
            scene.refine_spatial_sum[0] = 1.0
            scene.refine_spatial_exposure[0] = 1.0
            scene.refine_angular_sum[0] = 1.0
            scene.refine_angular_exposure[0] = 1.0
        snapshot = fixture.model.underlying_scene.refinement_snapshot(
            max_level=1,
            min_spatial_exposure=1,
            min_angular_exposure=1,
            cooldown_events=0,
            child_maturity_events=0,
        )
        observer.on_before_refinement(
            **dict(observer_context(fixture), step=1000, snapshot=snapshot, scene=fixture.model.underlying_scene)
        )
        n_split, n_angular, active, report = fixture.model.underlying_scene.apply_refinement_snapshot(
            snapshot,
            spatial_fraction=1.0,
            angular_fraction=1.0,
            max_level=1,
            optimizer=fixture.optimizer,
            max_active=8,
        )
        observer.on_after_refinement(
            **dict(
                observer_context(fixture),
                step=1000,
                snapshot=snapshot,
                scene=fixture.model.underlying_scene,
                n_split=n_split,
                n_angular=n_angular,
                active=active,
                report=report,
            )
        )
    labels = [record["label"] for record in observer.stage_metadata]
    check(
        labels == ["restored_step_700", "step_1000_pre_refinement", "step_1000_post_refinement"],
        "stage order changed",
    )
    event = observer.refinement_events[0]
    check(event["new_sibling_coefficients_zero"], "new sibling coefficients were not zero")
    check(event["newly_unlocked_coefficients_zero"], "new angular coefficients were not zero")
    check(event["pre_post_raw_preserved"], "raw pre/post refinement invariant was not checked")
    check(event["inherited_coefficients_preserved"], "inherited coefficients were not preserved")
    check(event["gain_unchanged"], "gain changed across refinement")
    check(
        observer.observation_checks and all(item["all_exact"] for item in observer.observation_checks),
        "observation changed training state",
    )
    print("PASS stage ordering, fixed cohort, topology, gain, raw/prediction invariants")
    return observer


def _expected_intervention_prediction(observer, current, label):
    scene = observer.scene
    w_re = current["model.coefficient_field.scene.w_re"].clone()
    w_im = current["model.coefficient_field.scene.w_im"].clone()
    delta = current["model.coefficient_field.scene.delta_raw"]
    log_mag = current["calibration.log_mag"]
    phase = current["calibration.phase"]
    baseline = observer.post_refinement_baseline
    if label == INTERVENTION_LABELS[0]:
        log_mag = baseline["calibration.log_mag"]
        phase = baseline["calibration.phase"]
    elif label == INTERVENTION_LABELS[1]:
        rows = observer.original_active_mask
        w_re[rows, 0] = baseline["model.coefficient_field.scene.w_re"][rows, 0]
        w_im[rows, 0] = baseline["model.coefficient_field.scene.w_im"][rows, 0]
    elif label == INTERVENTION_LABELS[2]:
        w_re[observer.new_sibling_mask, 0] = 0
        w_im[observer.new_sibling_mask, 0] = 0
    elif label == INTERVENTION_LABELS[3]:
        mask = observer.newly_unlocked_mask | observer.new_sibling_inherited_angular_mask
        w_re[mask] = 0
        w_im[mask] = 0
    return fake_signal_from_tensors(
        w_re,
        w_im,
        delta,
        scene.active_mask,
        log_mag,
        phase,
        326,
    )


def test_independent_single_factor_interventions(observer) -> None:
    model = observer.model
    calibration = observer.calibration
    optimizer = observer.optimizer
    scene = model.underlying_scene
    with torch.no_grad():
        new_rows = observer.new_sibling_mask.nonzero(as_tuple=True)[0]
        angular_mask = observer.newly_unlocked_mask | observer.new_sibling_inherited_angular_mask
        scene.w_re[0, 0] = 2.0
        scene.w_im[0, 0] = -1.5
        scene.w_re[new_rows, 0] = 3.0
        scene.w_im[new_rows, 0] = 1.25
        scene.w_re[angular_mask] = 4.0
        scene.w_im[angular_mask] = -2.0
        calibration.log_mag.fill_(np.log(2.0))
        calibration.phase.fill_(0.4)

    optimizer.zero_grad(set_to_none=True)
    loss = scene.w_re.square().sum() + scene.w_im.square().sum() + scene.delta_raw.square().sum()
    loss = loss + sum(parameter.square().sum() for parameter in calibration.parameters())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(calibration.parameters()), 1.0)
    actual_ping = int(observer.rng.choice(observer.train_indices))
    actual_bins = train_sas.select_bins(observer.rng, np.ones(326, dtype=np.complex64), observer.args.max_bins)
    observer.on_before_optimizer(
        model=model,
        calibration=calibration,
        optimizer=optimizer,
        cache=observer.cache,
        args=observer.args,
        device=torch.device("cpu"),
        scene=scene,
        step=701,
        ping=actual_ping,
        bins=actual_bins,
        loss=loss,
        metrics={},
        aux={},
        grad_norm=torch.tensor(1.0),
        will_refine=False,
    )
    optimizer.step()
    current = observer._parameter_snapshot()
    with mock.patch.object(
        train_sas, "render_one", side_effect=fixed_fake_render_factory(model, calibration)
    ):
        # Step 701 is deliberately outside the production post-update stage
        # list; invoke the same intervention helper directly so this test
        # isolates factor independence rather than stage scheduling.
        observer._run_interventions("independent_factor_test", 701)
    predictions = observer.intervention_predictions[-len(INTERVENTION_LABELS):]
    raws = observer.intervention_raw_predictions[-len(INTERVENTION_LABELS):]
    for index, label in enumerate(INTERVENTION_LABELS):
        expected_predicted, expected_raw = _expected_intervention_prediction(observer, current, label)
        expected_predicted_np = np.broadcast_to(expected_predicted.detach().numpy(), predictions[index].shape)
        expected_raw_np = np.broadcast_to(expected_raw.detach().numpy(), raws[index].shape)
        check(
            np.array_equal(predictions[index], expected_predicted_np),
            f"intervention {label} was cumulative or otherwise incorrect; "
            f"max prediction delta={np.max(np.abs(predictions[index] - expected_predicted_np))}",
        )
        check(
            np.array_equal(raws[index], expected_raw_np),
            f"raw intervention {label} was cumulative or otherwise incorrect",
        )
    check(observer.restore_checks[-1]["all_exact"], "interventions did not restore exact state")
    check(observer.restore_checks[-1]["numpy_rng_exact"], "interventions changed NumPy generator state")
    print("PASS independent single-factor interventions and exact restoration")


def test_rng_mismatch_stops_without_retry() -> None:
    fixture = make_observer_fixture()
    observer = RefinementDiagnosticObserver(make_tempdir("sas_diag_mismatch_"))
    with mock.patch.object(
        train_sas, "render_one", side_effect=fixed_fake_render_factory(fixture.model, fixture.calibration)
    ):
        observer.on_restore(**observer_context(fixture))
    wrong_ping = int(fixture.cache.train_indices[0])
    try:
        observer.on_before_optimizer(
            **dict(
                observer_context(fixture),
                step=701,
                scene=fixture.model.underlying_scene,
                ping=wrong_ping,
                bins=np.arange(4, dtype=np.int64),
                loss=torch.zeros(()),
                metrics={},
                aux={},
                grad_norm=torch.zeros(()),
                will_refine=False,
            )
        )
    except RuntimeError as exc:
        check("sampling mismatch" in str(exc), "wrong replay mismatch error")
    else:
        raise AssertionError("sampling mismatch was not rejected")
    check(observer.failure is not None, "mismatch did not record failure")
    check((observer.output_dir / "diagnostic_summary.json").exists(), "mismatch did not publish partial report")
    print("PASS replay RNG mismatch stops without retry and writes partial evidence")


def test_exception_restoration(observer) -> None:
    before_parameters = observer._parameter_snapshot()
    before_optimizer = observer.optimizer.state_dict()
    before_rng = copy.deepcopy(observer.rng.bit_generator.state)

    def fail_render(*_args, **_kwargs):
        raise RuntimeError("intentional diagnostic failure")

    with mock.patch.object(train_sas, "render_one", side_effect=fail_render):
        try:
            observer._run_interventions("exception_stage", 1002)
        except RuntimeError as exc:
            check("intentional diagnostic failure" in str(exc), "wrong exception escaped")
        else:
            raise AssertionError("intervention exception was swallowed")
    after_parameters = observer._parameter_snapshot()
    check(all(torch.equal(before_parameters[key], after_parameters[key]) for key in before_parameters), "exception did not restore parameters")
    check(nested_equal(before_optimizer, observer.optimizer.state_dict()), "exception changed optimizer state")
    check(nested_equal(before_rng, observer.rng.bit_generator.state), "exception changed NumPy RNG")
    print("PASS exception path restores state without retry")


def make_training_cache():
    return SimpleNamespace(
        weights=WeightTable(),
        tx_coords=np.zeros((50_000, 3), dtype=np.float32),
        rx_coords=np.zeros((50_000, 3), dtype=np.float32),
        radii=np.ones(4, dtype=np.float32),
        corners=np.zeros((8, 3), dtype=np.float32),
        voxels=np.zeros((1, 3), dtype=np.float32),
        source_ids=np.arange(50_000, dtype=np.int64),
        train_indices=np.arange(43_200, dtype=np.int64),
        validation_indices=FIXED_SOURCE_IDS.copy(),
        test_indices=np.empty(0, dtype=np.int64),
        tx_vecs=None,
        manifest={},
        has_explicit_splits=True,
        num_pings=50_000,
        num_bins=326,
    )


def test_main_observer_on_off_continuation() -> None:
    cache = make_training_cache()
    seed = 123
    train_sas.seed_all(seed)
    initial_model, initial_calibration, initial_optimizer = make_model_and_optimizer(max_degree=3)
    with torch.no_grad():
        initial_calibration.log_mag.fill_(float(np.log(1.2)))
        initial_calibration.phase.fill_(0.25)

    cli_common = [
        "--cache", "unused", "--model", "adaptive_rift_sas", "--device", "cpu",
        "--steps", "1010", "--refine-every", "1000", "--probe-every", "16",
        "--initial-granularity", "1", "--granularity", "4", "--adaptive-capacity", "8",
        "--max-active", "8", "--split-max-level", "3", "--spatial-fraction", "0.05",
        "--angular-fraction", "0.10", "--cooldown-events", "1", "--child-maturity-events", "1",
        "--sh-degree", "3", "--max-bins", "110", "--eval-pings", "8", "--eval-bins", "0",
        "--eval-every", "1000", "--checkpoint-every", "1000", "--log-every", "1000",
        "--num-rays", "4", "--signal-scale", "1", "--coefficient-lr", "0.01",
        "--position-lr", "0.01", "--grad-clip", "1", "--seed", str(seed),
        "--calibration-mode", "log_polar", "--no-opacity-normalize",
    ]
    initial_args = train_sas.parse_args(cli_common + ["--checkpoint-name", "initial"])
    initial_args.calibration_mode = "log_polar"
    initial_args.require_explicit_splits = True
    initial_rng = np.random.default_rng(seed)
    initial_state = train_sas.checkpoint(
        initial_model,
        initial_calibration,
        initial_optimizer,
        0,
        float("inf"),
        initial_rng,
        [],
        initial_args,
        cache,
    )
    initial_state["step"] = 700
    checkpoint_path = Path(make_tempdir("sas_diag_main_ckpt_")) / "step700.pt"
    torch.save(initial_state, checkpoint_path)

    target_full_holder: dict[str, np.ndarray] = {}

    def signal_for(model, calibration, bins, *, probe_next_band=False):
        scene = model.underlying_scene
        order = scene.order
        if probe_next_band:
            order = torch.minimum(order + 1, torch.full_like(order, scene.max_degree))
        degree = scene.basis_degree.reshape(1, -1)
        mask = (degree <= order.reshape(-1, 1)) & scene.active_mask.reshape(-1, 1)
        position = scene.positions()
        position_factor = 1.0 + 0.07 * position[:, 0] - 0.05 * position[:, 1] + 0.03 * position[:, 2]
        weights = torch.arange(
            1, scene.w_re.shape[1] + 1, dtype=scene.w_re.dtype, device=scene.w_re.device
        )
        real = (scene.w_re * mask * position_factor[:, None] * weights).sum()
        imag = (scene.w_im * mask * position_factor[:, None] * weights).sum()
        raw = torch.complex(real.expand(len(bins)), imag.expand(len(bins)))
        return calibration(raw), raw

    with torch.no_grad():
        initial_predicted, _initial_raw = signal_for(
            initial_model, initial_calibration, np.arange(326, dtype=np.int64)
        )
        target_full_holder["target"] = (
            initial_predicted.detach().cpu().numpy().astype(np.complex64)
            / (1.0 + np.sqrt(1.369373))
        )

    def actual_fake_render(model, calibration, _cache, _ping, bins, _args, _device, **kwargs):
        predicted, raw = signal_for(
            model,
            calibration,
            bins,
            probe_next_band=bool(kwargs.get("probe_next_band", False)),
        )
        target = torch.as_tensor(
            target_full_holder["target"][np.asarray(bins, dtype=np.int64)],
            dtype=predicted.dtype,
            device=predicted.device,
        )
        loss = torch.nn.functional.mse_loss(predicted.real, target.real) + torch.nn.functional.mse_loss(
            predicted.imag, target.imag
        )
        return loss, train_sas.metric_record(predicted, target), {
            "calibration_raw": raw,
            "calibration_target": target,
            "calibration_predicted": predicted,
            "transmittance": torch.ones((len(bins), 1), dtype=torch.float32, device=predicted.device),
            "lambertian": torch.ones((len(bins), 1), dtype=torch.float32, device=predicted.device),
            "actual_rays": 1,
        }

    built_models = []
    built_optimizers = []
    built_calibrations = []

    def fake_build_model(args, _cache, _device):
        model, _calibration, _optimizer = make_model_and_optimizer(max_degree=args.sh_degree)
        built_models.append(model)
        return model

    original_optimizer_for_model = train_sas._optimizer_for_model

    def wrapped_optimizer(model, calibration, args):
        optimizer = original_optimizer_for_model(model, calibration, args)
        built_optimizers.append(optimizer)
        built_calibrations.append(calibration)
        return optimizer

    evaluation_count = [0]

    def fake_evaluate(*_args, **_kwargs):
        evaluation_count[0] += 1
        rel_mse = 0.5 if evaluation_count[0] % 2 == 1 else 0.4
        return {
            "rel_mse": rel_mse,
            "l1_real": 0.0,
            "l1_imag": 0.0,
            "l1_mag": 0.0,
            "complete": 1.0,
            "views": 1.0,
        }

    default_dir = Path(make_tempdir("sas_diag_default_"))
    observer_dir = default_dir
    dense_calls = []
    def fake_dense_density(points):
        dense_calls.append(1)
        return torch.zeros(points.shape[0], dtype=torch.float32, device=points.device)

    with mock.patch.object(train_sas, "load_sas_cache", return_value=cache), \
         mock.patch.object(train_sas, "_validate_cache_contract", return_value=None), \
         mock.patch.object(train_sas, "_validate_saved_model_box", return_value=None), \
         mock.patch.object(train_sas, "_validate_best_candidate", return_value=None), \
         mock.patch.object(train_sas, "build_model", side_effect=fake_build_model), \
         mock.patch.object(train_sas, "_optimizer_for_model", side_effect=wrapped_optimizer), \
         mock.patch.object(train_sas, "render_one", side_effect=actual_fake_render), \
         mock.patch.object(train_sas, "evaluate", side_effect=fake_evaluate), \
         mock.patch.object(train_sas.ComplexSHSonarField, "dense_density", side_effect=fake_dense_density):
        train_sas.main(
            cli_common + ["--checkpoint-root", str(default_dir), "--checkpoint-name", ".", "--resume", str(checkpoint_path)]
        )
        default_model = built_models[0]
        # make_model_and_optimizer() itself constructs a throwaway optimizer
        # under this patch; the last entries are the trainer-owned objects.
        default_optimizer = built_optimizers[-1]
        default_calibration = built_calibrations[-1]
        default_model_state = copy.deepcopy(default_model.state_dict())
        default_calibration_state = copy.deepcopy(default_calibration.state_dict())
        default_optimizer_state = copy.deepcopy(default_optimizer.state_dict())
        default_gradients = {
            **{
                f"model.{name}": None if value.grad is None else value.grad.detach().clone()
                for name, value in default_model.named_parameters()
            },
            **{
                f"calibration.{name}": None if value.grad is None else value.grad.detach().clone()
                for name, value in default_calibration.named_parameters()
            },
        }
        default_payload = torch.load(default_dir / "checkpoint_final.pt", map_location="cpu", weights_only=False)
        evaluation_count[0] = 0
        observer = RefinementDiagnosticObserver(
            default_dir,
            expected_fixed_val_rel_mse=1.369373,
            expected_historical_counts={"n_split": 1, "n_angular": 1, "active_after": 8},
        )
        train_sas.main(
            cli_common + ["--checkpoint-root", str(observer_dir), "--checkpoint-name", ".", "--resume", str(checkpoint_path)],
            diagnostic_observer=observer,
        )
        observer_model = observer.model
        observer_optimizer = observer.optimizer
        observer_calibration = observer.calibration
        check(observer_model is not None and observer_optimizer is not None, "observer main did not restore model")
        check(observer_calibration is not None, "observer main did not restore calibration")

    check(nested_equal(default_model.state_dict(), observer_model.state_dict()), "observer changed final model state")
    check(nested_equal(default_calibration.state_dict(), observer_calibration.state_dict()), "observer changed final calibration state")
    check(nested_equal(default_optimizer.state_dict(), observer_optimizer.state_dict()), "observer changed final optimizer state")
    check(nested_equal(default_model_state, observer_model.state_dict()), "observer model state diverged from saved off state")
    check(nested_equal(default_calibration_state, observer_calibration.state_dict()), "observer calibration state diverged from saved off state")
    check(nested_equal(default_optimizer_state, observer_optimizer.state_dict()), "observer optimizer state diverged from saved off state")
    observer_gradients = {
        **{
            f"model.{name}": None if value.grad is None else value.grad.detach().clone()
            for name, value in observer_model.named_parameters()
        },
        **{
            f"calibration.{name}": None if value.grad is None else value.grad.detach().clone()
            for name, value in observer_calibration.named_parameters()
        },
    }
    check(nested_equal(default_gradients, observer_gradients), "observer changed final gradients")
    check(observer._finished, "actual refinement observer did not finish successfully")
    check(len(observer.selection_sequence) == 310, "actual observer missed a 310-step replay")
    check(len(observer.intervention_metadata) == 20, "actual observer missed intervention renders")
    check(observer.refinement_events[0]["actual_counts"] == {"n_split": 1, "n_angular": 1, "active_after": 8}, "tiny fixture topology override was not exercised")
    check(all(item["all_exact"] for item in observer.observation_checks), "observation changed training state")
    check(all(item["all_exact"] for item in observer.restore_checks), "intervention changed training state")
    check(len(dense_calls) == 1, "observer path reached normal geometry footer")
    check(nested_equal(default_payload["rng_state"], observer.rng.bit_generator.state), "observer changed replay NumPy RNG")
    check(torch.equal(default_payload["torch_rng_state"], torch.get_rng_state()), "observer changed replay Torch RNG")
    print("PASS train_sas.main actual observer on/off continuation is exact")


def main() -> None:
    test_default_path_contract()
    observer = test_stage_order_and_fixed_cohort()
    test_independent_single_factor_interventions(observer)
    test_exception_restoration(observer)
    test_rng_mismatch_stops_without_retry()
    test_main_observer_on_off_continuation()
    check(POST_UPDATE_STEPS == (1001, 1002, 1003, 1005, 1010), "post-update stage contract changed")
    print("PASS adaptive AirSAS refinement diagnostic validation")


if __name__ == "__main__":
    main()
