#!/usr/bin/env python3
"""One local-only validation package for the bounded B787 RIFT action gate.

The test uses synthetic, temporary data only.  It never opens the PACE B787
archive, submits work, writes a manager file, or retains a generated artifact.
It checks the derived-role provenance, frozen command, observer noninterference,
and a clean epoch-boundary continuation through the ordinary RIFT entrypoint.
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train as rift_train  # noqa: E402
import train_adaptive_rift_smoke as action_driver  # noqa: E402
from rift import range_operator  # noqa: E402
from rift.b7873200_adaptive_action_smoke import (  # noqa: E402
    ACTION_GATE_CHECKPOINT_NAME,
    ACTION_GATE_EXECUTION_LABEL,
    ACTION_GATE_NUM_TEST,
    ACTION_GATE_NUM_TRAIN,
    ACTION_GATE_NUM_VALIDATION,
    ACTION_GATE_SCHEMA,
    B787_3200_CANONICAL_MANIFEST_NAME,
    B787_3200_CANONICAL_NPZ_PATH,
    B787_3200_PARENT_COUNTS,
    B787_3200_RESPONSE_SHAPE,
    AdaptiveActionGateObserver,
    _validate_action_gate_manifest,
    action_gate_train_argv,
    build_action_gate_manifest,
    validate_action_gate_child_manifest,
    write_action_gate_manifest,
)


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


def nested_equal(left: Any, right: Any) -> bool:
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


def canonical_parent() -> dict[str, Any]:
    train_stop = B787_3200_PARENT_COUNTS[0]
    validation_stop = train_stop + B787_3200_PARENT_COUNTS[1]
    test_stop = validation_stop + B787_3200_PARENT_COUNTS[2]
    return {
        "schema_version": 1,
        "name": B787_3200_CANONICAL_MANIFEST_NAME,
        "dataset": {
            "num_views": 10_000,
            "response_shape": list(B787_3200_RESPONSE_SHAPE),
            "response_dtype": "complex64",
        },
        "split": {
            "strategy": "fixed_tail_subsampled",
            "num_train": B787_3200_PARENT_COUNTS[0],
            "num_validation": B787_3200_PARENT_COUNTS[1],
            "num_test": B787_3200_PARENT_COUNTS[2],
            "num_unused": B787_3200_PARENT_COUNTS[3],
            "complete_partition": True,
            "test_sealed": True,
            "unused_sealed": True,
            "train_indices": list(range(0, train_stop)),
            "validation_indices": list(range(train_stop, validation_stop)),
            "test_indices": list(range(validation_stop, test_stop)),
            "unused_indices": list(range(test_stop, 10_000)),
        },
    }


def stage_manifest_and_argv(root: Path) -> None:
    parent = canonical_parent()
    child = build_action_gate_manifest(parent)
    validate_action_gate_child_manifest(parent, child)
    split = child["split"]
    assert isinstance(split, dict)
    check(
        split["train_indices"] == parent["split"]["train_indices"][:16]
        and split["validation_indices"] == parent["split"]["validation_indices"][:16]
        and split["test_indices"] == parent["split"]["test_indices"],
        "derived action roles keep the exact ordered parent train, validation, and sealed-test IDs",
    )
    expected_unused = (
        parent["split"]["train_indices"][16:]
        + parent["split"]["unused_indices"]
        + parent["split"]["validation_indices"][16:]
    )
    check(
        split["unused_indices"] == expected_unused
        and (split["num_train"], split["num_validation"], split["num_test"], split["num_unused"])
        == (16, 16, 1000, 8968),
        "derived action roles seal every remaining parent-development ID in order",
    )
    for role in ("train_indices", "validation_indices", "test_indices", "unused_indices"):
        reordered = copy.deepcopy(child)
        reordered["split"][role][0], reordered["split"][role][1] = (
            reordered["split"][role][1],
            reordered["split"][role][0],
        )
        _validate_action_gate_manifest(reordered)
        expect_value_error(
            lambda reordered=reordered: validate_action_gate_child_manifest(parent, reordered),
            f"parent provenance rejects same-membership reordered {role}",
        )
    changed_provenance = copy.deepcopy(child)
    changed_provenance["engineering_subset"]["selection"]["train"] = "different role"
    expect_value_error(
        lambda: validate_action_gate_child_manifest(parent, changed_provenance),
        "parent provenance rejects a changed descriptive selection contract",
    )

    manifest_path = root / "derived_roles.json"
    write_action_gate_manifest(manifest_path, child)
    write_action_gate_manifest(manifest_path, child)
    before = manifest_path.read_text(encoding="utf-8")
    divergent = copy.deepcopy(child)
    divergent["engineering_subset"]["reporting_status"] = "different"
    expect_value_error(
        lambda: write_action_gate_manifest(manifest_path, divergent),
        "an existing derived-manifest identity cannot be overwritten with a divergent payload",
    )
    check(manifest_path.read_text(encoding="utf-8") == before, "manifest refusal leaves the existing payload intact")

    argv = action_gate_train_argv(
        npz_path=B787_3200_CANONICAL_NPZ_PATH,
        manifest_path=manifest_path,
        checkpoint_root=root / "checkpoints",
    )
    parsed = rift_train.parse_args(argv)
    check(
        parsed.data_format == "npz"
        and parsed.npz_sealed_protocol
        and parsed.num_train == ACTION_GATE_NUM_TRAIN
        and parsed.num_val == ACTION_GATE_NUM_VALIDATION
        and parsed.num_test == ACTION_GATE_NUM_TEST
        and parsed.num_tx == 16
        and parsed.num_rx == 16
        and parsed.num_freq_wanted == 600,
        "frozen action command uses the sealed 16/16/1000 full 16-by-16-by-600 acquisition",
    )
    check(
        parsed.epochs == 4
        and parsed.step_every == 4
        and parsed.loss == "complex"
        and parsed.scene_repr == "point_sh"
        and parsed.forward_operator == "range"
        and parsed.range_model == "product"
        and parsed.compute_dtype == "float64"
        and parsed.phase_sign == -1.0,
        "frozen action command uses the approved four-epoch coherent range/product negative-phase route",
    )
    check(
        parsed.granularity == 8
        and parsed.extent == 0.15
        and parsed.max_points == 640
        and parsed.adaptive_max_active == 640
        and parsed.sh_init_degree == 0
        and parsed.sh_max_degree == 1
        and parsed.bp_init == 16
        and parsed.init_scale == 0.0,
        "frozen action command uses the approved 8-cubed 512-point initial scene and 640-slot action budget",
    )
    check(
        parsed.adaptive_capacity_v2
        and parsed.adaptive_refine_every == 1
        and parsed.adaptive_probe_every == 1
        and parsed.adaptive_spatial_fraction == 1.0 / 640.0
        and parsed.adaptive_angular_fraction == 1.0 / 640.0
        and parsed.adaptive_child_maturity_events == 1
        and parsed.adaptive_cooldown_events == 0
        and parsed.split_max_level == 1
        and parsed.split_every == 0
        and parsed.grow_every == 0
        and parsed.prune_every == 0,
        "frozen action command uses the approved bounded joint adaptive-v2 controller only",
    )
    check(
        parsed.lr == 0.003
        and parsed.pos_lr == 0.003
        and parsed.adam_eps == 1.0e-20
        and parsed.weight_decay == 0.0
        and parsed.t0 == 10
        and parsed.t_mult == 2
        and parsed.seed == 42
        and parsed.l1_weight == 0.0
        and parsed.sh_smooth_weight == 0.0
        and parsed.mag_weight == 0.0
        and not parsed.no_learn_gain
        and not parsed.normalize_scene_scale
        and parsed.clip_grad_norm == 0.0
        and parsed.execution_contract_label == ACTION_GATE_EXECUTION_LABEL
        and parsed.require_full_resume_state,
        "frozen action command preserves gain-on, unnormalized, unclipped optimizer and recovery settings",
    )
    resumed = action_gate_train_argv(
        npz_path=B787_3200_CANONICAL_NPZ_PATH,
        manifest_path=manifest_path,
        checkpoint_root=root / "checkpoints",
        resume=root / "checkpoints" / ACTION_GATE_CHECKPOINT_NAME / "checkpoint_latest.pth.tar",
    )
    check("--resume" not in argv and resumed[-2] == "--resume", "fresh and clean-resume action commands differ only by the exact resume path")


def stage_driver_identity(root: Path) -> None:
    """Exercise path-only fresh/resume identity checks without touching a GPU."""

    original_available = torch.cuda.is_available
    original_world_size = os.environ.get("WORLD_SIZE")
    torch.cuda.is_available = lambda: True
    os.environ["WORLD_SIZE"] = "1"
    try:
        checkpoint_root = root / "driver_identity"
        fresh_args = action_driver.parse_args(["--checkpoint-root", str(checkpoint_root)])
        run_root = action_driver._validate_cli(fresh_args)
        check(
            run_root == checkpoint_root.resolve() / ACTION_GATE_CHECKPOINT_NAME,
            "driver derives its one fresh engineering identity under the requested checkpoint root",
        )
        run_root.mkdir(parents=True)
        latest = run_root / "checkpoint_latest.pth.tar"
        latest.touch()
        expect_value_error(
            lambda: action_driver._validate_cli(fresh_args),
            "driver refuses a fresh run when an existing recovery checkpoint needs an explicit resume",
        )
        resumed_args = action_driver.parse_args([
            "--checkpoint-root", str(checkpoint_root), "--resume", str(latest),
        ])
        check(
            action_driver._validate_cli(resumed_args) == run_root,
            "driver accepts only the exact same-identity latest checkpoint for a clean resume",
        )
        resume_alias = root / "resume_alias.pth.tar"
        resume_alias.symlink_to(latest)
        aliased_resume_args = action_driver.parse_args([
            "--checkpoint-root", str(checkpoint_root), "--resume", str(resume_alias),
        ])
        expect_value_error(
            lambda: action_driver._validate_cli(aliased_resume_args),
            "driver rejects a symlinked resume alias even when it targets this run's latest checkpoint",
        )
        (run_root / "checkpoint_final.pth.tar").touch()
        expect_value_error(
            lambda: action_driver._validate_cli(resumed_args),
            "driver refuses to rerun or resume an already final action-gate identity",
        )

        foreign_root = root / "terminal_predecessor"
        foreign_root.mkdir()
        symlinked_root = root / "redirected_action_root"
        symlinked_root.symlink_to(foreign_root, target_is_directory=True)
        expect_value_error(
            lambda: action_driver._validate_cli(
                action_driver.parse_args(["--checkpoint-root", str(symlinked_root)])
            ),
            "driver rejects a symlinked action root before it can redirect to another run",
        )

        run_link_root = root / "run_link_root"
        run_link_root.mkdir()
        redirected_run = foreign_root / "redirected_run"
        redirected_run.mkdir()
        (run_link_root / ACTION_GATE_CHECKPOINT_NAME).symlink_to(redirected_run, target_is_directory=True)
        expect_value_error(
            lambda: action_driver._validate_cli(
                action_driver.parse_args(["--checkpoint-root", str(run_link_root)])
            ),
            "driver rejects a symlinked action-gate run root before it can redirect to another run",
        )

        latest_link_root = root / "latest_link_root"
        latest_link_run = latest_link_root / ACTION_GATE_CHECKPOINT_NAME
        latest_link_run.mkdir(parents=True)
        foreign_latest = foreign_root / "checkpoint_latest.pth.tar"
        foreign_latest.touch()
        linked_latest = latest_link_run / "checkpoint_latest.pth.tar"
        linked_latest.symlink_to(foreign_latest)
        expect_value_error(
            lambda: action_driver._validate_cli(
                action_driver.parse_args([
                    "--checkpoint-root", str(latest_link_root), "--resume", str(linked_latest),
                ])
            ),
            "driver rejects a symlinked latest checkpoint during literal resume before resolved-path comparison",
        )

        final_link_root = root / "final_link_root"
        final_link_run = final_link_root / ACTION_GATE_CHECKPOINT_NAME
        final_link_run.mkdir(parents=True)
        foreign_final = foreign_root / "checkpoint_final.pth.tar"
        foreign_final.touch()
        (final_link_run / "checkpoint_final.pth.tar").symlink_to(foreign_final)
        expect_value_error(
            lambda: action_driver._validate_cli(
                action_driver.parse_args(["--checkpoint-root", str(final_link_root)])
            ),
            "driver rejects a symlinked final checkpoint before terminal-run checks",
        )
    finally:
        torch.cuda.is_available = original_available
        if original_world_size is None:
            os.environ.pop("WORLD_SIZE", None)
        else:
            os.environ["WORLD_SIZE"] = original_world_size


class FullSpectrumDataset:
    def __init__(self, indices: list[int]) -> None:
        self.indices = list(indices)
        self.requested_indices: list[int] = []
        self.freqs = torch.linspace(8.5e9, 11.5e9, 600, dtype=torch.float32)
        rx = torch.zeros((16, 3), dtype=torch.float32)
        tx = torch.zeros((16, 3), dtype=torch.float32)
        rx[:, 0] = torch.linspace(-0.02, 0.02, 16)
        tx[:, 1] = torch.linspace(-0.02, 0.02, 16)
        self.item = (
            self.freqs,
            torch.tensor([0.25], dtype=torch.float32),  # dphi, matching NPZ __getitem__
            torch.tensor([0.75], dtype=torch.float32),  # dtheta
            torch.full((600, 16 * 16), 2.0, dtype=torch.float32),
            torch.zeros((600, 16 * 16), dtype=torch.float32),
            rx,
            tx,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        self.requested_indices.append(int(index))
        return self.item


class FullSpectrumProbeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.last_dtheta: torch.Tensor | None = None
        self.last_dphi: torch.Tensor | None = None

    def active_scatterers(self, dtheta: torch.Tensor, dphi: torch.Tensor):
        self.last_dtheta = dtheta.detach().clone()
        self.last_dphi = dphi.detach().clone()
        position = torch.zeros((1, 3), device=dtheta.device, dtype=torch.float32)
        weight = torch.complex(self.weight.reshape(1), torch.zeros(1, device=dtheta.device))
        return position, weight


def make_full_spectrum_fixture():
    model = FullSpectrumProbeModel().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    train_dataset = FullSpectrumDataset(list(range(16)))
    validation_dataset = FullSpectrumDataset(list(range(100, 116)))
    return model, optimizer, train_dataset, validation_dataset


def make_full_spectrum_observer(
    model: FullSpectrumProbeModel,
    optimizer: torch.optim.Optimizer,
    train_dataset: FullSpectrumDataset,
    validation_dataset: FullSpectrumDataset,
    *,
    device: torch.device | str = torch.device("cpu"),
) -> AdaptiveActionGateObserver:
    return AdaptiveActionGateObserver(
        model=model,
        optimizer=optimizer,
        gain=None,
        train_loader=SimpleNamespace(dataset=train_dataset),
        validation_loader=SimpleNamespace(dataset=validation_dataset),
        criterion=nn.MSELoss(),
        device=device,
        num_freq_selected=600,
        phase_sign=-1.0,
        forward_operator_name="range",
        compute_dtype=torch.float64,
        data_format="npz",
        op_kwargs={"range_model": "product", "point_chunk": 512, "pair_chunk": 64},
        occlusion=None,
        selected_train_ids=list(range(16)),
        selected_validation_ids=list(range(100, 116)),
    )


def stage_observer_render_and_state() -> None:
    calls: list[dict[str, Any]] = []
    original_renderer = range_operator.range_forward_operator

    def fake_renderer(
        freqs_full, kvector_full, rx_pos, tx_pos, scatterer_pos, scatterer_weights, **kwargs
    ):
        del kvector_full, scatterer_weights
        calls.append({
            "freq_indices": kwargs["freq_indices"].detach().cpu().clone(),
            "phase_sign": kwargs["phase_sign"],
            "range_model": kwargs["range_model"],
            "compute_dtype": kwargs["compute_dtype"],
            "freq_device": freqs_full.device.type,
            "rx_device": rx_pos.device.type,
            "tx_device": tx_pos.device.type,
            "rx_shape": tuple(rx_pos.shape),
            "tx_shape": tuple(tx_pos.shape),
            "scatterer_device": scatterer_pos.device.type,
        })
        dtype = torch.complex128 if kwargs["compute_dtype"] == torch.float64 else torch.complex64
        return torch.zeros(
            (int(kwargs["freq_indices"].numel()), int(rx_pos.shape[0]), int(tx_pos.shape[0])),
            device=freqs_full.device,
            dtype=dtype,
        )

    range_operator.range_forward_operator = fake_renderer
    try:
        rng_before = rift_train.capture_rng_state()
        model, optimizer, dataset, validation_dataset = make_full_spectrum_fixture()
        source_item_before = tuple(value.detach().clone() for value in dataset.item)
        validation_item_before = tuple(value.detach().clone() for value in validation_dataset.item)
        model_state_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        gradients_before = {
            name: None if value.grad is None else value.grad.detach().clone()
            for name, value in model.named_parameters()
        }
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        training_before = model.training
        observer = make_full_spectrum_observer(model, optimizer, dataset, validation_dataset)
        check(
            isinstance(observer.device, torch.device)
            and observer.device == torch.device("cpu"),
            "observer preserves an explicit torch.device input as its canonical local device",
        )
        probe_predictions = observer._predictions()
        metrics = observer._role_metrics(observer.train_loader, observer.validation_loader)
        prediction, observed = observer._render_item(dataset[0])
        rng_after = rift_train.capture_rng_state()
        check(
            prediction.shape == observed.shape == (600, 16, 16)
            and prediction.device.type == "cpu"
            and observed.device.type == "cpu"
            and prediction.dtype == torch.complex128
            and observed.dtype == torch.complex64,
            "observer consumes the direct seven-tuple NPZ item and produces aligned full-spectrum Tx/Rx cubes",
        )
        check(
            model.last_dtheta is not None
            and model.last_dphi is not None
            and float(model.last_dtheta.squeeze()) == 0.75
            and float(model.last_dphi.squeeze()) == 0.25
            and all(torch.equal(entry["freq_indices"], torch.arange(600)) for entry in calls)
            and all(
                entry["phase_sign"] == -1.0
                and entry["range_model"] == "product"
                and entry["compute_dtype"] == torch.float64
                and entry["freq_device"] == "cpu"
                and entry["rx_device"] == "cpu"
                and entry["tx_device"] == "cpu"
                and entry["scatterer_device"] == "cpu"
                and entry["rx_shape"] == (16, 3)
                and entry["tx_shape"] == (16, 3)
                for entry in calls
            ),
            "observer preserves NPZ dphi/dtheta, all 600 bins, and the fp64 range-product geometry/device contract",
        )
        check(
            nested_equal(rng_before, rng_after)
            and all(torch.equal(before, after) for before, after in zip(source_item_before, dataset.item))
            and all(
                torch.equal(before, after)
                for before, after in zip(validation_item_before, validation_dataset.item)
            )
            and all(torch.equal(model_state_before[name], value) for name, value in model.state_dict().items())
            and all(
                (before is None and after.grad is None)
                or (before is not None and after.grad is not None and torch.equal(before, after.grad))
                for name, after in model.named_parameters()
                for before in [gradients_before[name]]
            )
            and nested_equal(optimizer_before, optimizer.state_dict())
            and model.training == training_before,
            "observer construction, probes, and fixed-role metrics leave captured RNGs, state-dict tensors, gradients, optimizer, and mode unchanged",
        )
        check(
            metrics["train"]["coherent_relative_l2"] == 1.0
            and metrics["validation"]["same_domain_zero_relative_mse"] == 1.0,
            "observer emits named coherent and same-domain zero references without reading a reserved role",
        )
        check(
            set(dataset.requested_indices).issuperset(range(16))
            and set(validation_dataset.requested_indices).issuperset(range(16)),
            "observer evaluates every direct train and validation seven-tuple rather than reusing one indexed item",
        )

        observer.records = [{"event": 0}]
        observer.pending = [{"event": 0, "checked_after_update": False}]
        observer.optimizer_updates = [{"logical_optimizer_updates": 1}]
        state = observer.checkpoint_state()
        restored_state = copy.deepcopy(state)
        restored_state["initial_metrics"]["train"]["coherent_relative_l2"] = 123.456
        restored_state["initial_parameters"]["scene.weight"] = (
            restored_state["initial_parameters"]["scene.weight"] + 1.0
        )
        restored_state["elapsed_seconds"] = 17.25
        restored_model, restored_optimizer, restored_dataset, restored_validation_dataset = make_full_spectrum_fixture()
        restored = make_full_spectrum_observer(
            restored_model, restored_optimizer, restored_dataset, restored_validation_dataset
        )
        restored._start_time = -1.0
        restored.restore_checkpoint_state(restored_state)
        check(
            restored.records == observer.records
            and restored.pending == observer.pending
            and restored.optimizer_updates == observer.optimizer_updates
            and restored._initial_metrics == restored_state["initial_metrics"]
            and torch.equal(
                restored._initial_parameters["scene.weight"],
                restored_state["initial_parameters"]["scene.weight"],
            )
            and restored._elapsed_before_resume == 17.25
            and restored._start_time > 0.0,
            "observer checkpoint recovery overwrites provisional baselines with saved evidence and starts a fresh resume wall-time origin",
        )
        original_load = rift_train.load_tensor_checkpoint
        try:
            rift_train.load_tensor_checkpoint = lambda *args, **kwargs: {
                "adaptive_event_observer_state": copy.deepcopy(restored_state)
            }
            factory_model, factory_optimizer, factory_dataset, factory_validation_dataset = make_full_spectrum_fixture()
            factory_holder: dict[str, AdaptiveActionGateObserver] = {}
            factory = action_driver._build_observer_factory(
                SimpleNamespace(resume="synthetic-clean-latest.pth.tar"),
                list(range(16)),
                list(range(100, 116)),
                factory_holder,
            )
            factory_observer = factory(
                model=factory_model,
                optimizer=factory_optimizer,
                gain=None,
                train_loader=SimpleNamespace(dataset=factory_dataset),
                validation_loader=SimpleNamespace(dataset=factory_validation_dataset),
                criterion=nn.MSELoss(),
                device=torch.device("cpu"),
                num_freq_selected=600,
                phase_sign=-1.0,
                forward_operator_name="range",
                compute_dtype=torch.float64,
                data_format="npz",
                op_kwargs={"range_model": "product", "point_chunk": 512, "pair_chunk": 64},
                occlusion=None,
            )
            check(
                factory_holder.get("observer") is factory_observer
                and factory_observer.records == observer.records
                and factory_observer._initial_metrics == restored_state["initial_metrics"]
                and torch.equal(
                    factory_observer._initial_parameters["scene.weight"],
                    restored_state["initial_parameters"]["scene.weight"],
                )
                and factory_observer._elapsed_before_resume == float(restored_state["elapsed_seconds"])
                and factory_observer._start_time > 0.0,
                "the real action-driver factory restores the real observer before generic resume training continues",
            )
        finally:
            rift_train.load_tensor_checkpoint = original_load
        for mutate, message in (
            (lambda value: value.update({"probe_source_view_ids": [-1]}), "wrong probe IDs"),
            (lambda value: value.pop("initial_metrics"), "missing initial metrics"),
            (lambda value: value.update({"initial_parameters": {}}), "missing initial parameters"),
            (lambda value: value.update({"elapsed_seconds": -1.0}), "negative elapsed time"),
            (lambda value: value.update({"finished": True}), "completed observer state"),
        ):
            invalid = copy.deepcopy(restored_state)
            mutate(invalid)
            target_model, target_optimizer, target_dataset, target_validation_dataset = make_full_spectrum_fixture()
            target = make_full_spectrum_observer(
                target_model, target_optimizer, target_dataset, target_validation_dataset
            )
            expect_value_error(
                lambda invalid=invalid, target=target: target.restore_checkpoint_state(invalid),
                f"observer recovery rejects {message}",
            )
    finally:
        range_operator.range_forward_operator = original_renderer


def stage_nan_probe_rejection() -> None:
    def after_record(prediction: torch.Tensor) -> dict[str, Any]:
        observer = object.__new__(AdaptiveActionGateObserver)
        observer.model = SimpleNamespace(
            active_mask=torch.tensor([True]),
            order=torch.tensor([0]),
            level=torch.tensor([0]),
            w_re=torch.zeros((1, 4)),
            w_im=torch.zeros((1, 4)),
            basis_degree=torch.tensor([0, 1, 1, 1]),
        )
        observer._open_events = {
            0: {
                "record": {
                    "angular_indices": [],
                    "spatial_indices": [],
                    "spatial_parent_levels_before": [],
                },
                "predictions": [torch.zeros(1, dtype=torch.complex64)],
                "active_mask": torch.tensor([True]),
                "orders": torch.tensor([0]),
                "levels": torch.tensor([0]),
            }
        }
        observer._predictions = lambda: [prediction]
        observer._scene_summary = lambda: {"support_bounds_ok": True, "degree_bounds_ok": True}
        observer._model_finite = lambda: True
        observer._optimizer_finite = lambda: True
        observer.records = []
        observer.pending = []
        observer.on_adaptive_event(
            "after", event=0, n_split=0, n_grown=0, n_active=1,
            last_grad_norm=0.0, logical_optimizer_updates=0,
        )
        return observer.records[0]

    record = after_record(torch.tensor([complex(float("nan"), 0.0)], dtype=torch.complex64))
    check(
        record["probe_full_coherent_predictions_finite"] is False
        and record["probe_full_coherent_prediction_preserved"] is False
        and record["probe_full_coherent_prediction_difference_max_abs"] is None,
        "a nonfinite probe render fails closed instead of passing the preservation tolerance",
    )
    finite = after_record(torch.zeros(1, dtype=torch.complex64))
    check(
        finite["probe_full_coherent_predictions_finite"] is True
        and finite["probe_full_coherent_prediction_difference_max_abs"] == 0.0
        and finite["probe_full_coherent_prediction_preserved"] is True,
        "an identical finite probe is positively accepted by the preservation control",
    )


def tiny_npz(path: Path) -> tuple[int, ...]:
    n_views, n_freq = 6, 4
    values = np.arange(n_views * n_freq, dtype=np.float32).reshape(n_views, 1, 1, 1, n_freq)
    response = (0.05 + values + 1j * (0.25 + values)).astype(np.complex64)
    angles = np.linspace(0.3, 1.3, n_views, dtype=np.float32)
    viewpoints = np.stack((10.0 * np.cos(angles), 10.0 * np.sin(angles), np.ones_like(angles)), axis=1)
    tx_pos = viewpoints[:, None, :].copy()
    rx_pos = viewpoints[:, None, :].copy()
    rx_pos[..., 1] += 0.01
    metadata = {
        "radar_fc_hz": 10.0e9,
        "radar_bandwidth_hz": 1.0e9,
        "num_adc_samples": n_freq,
        "target_radius_m": 0.1,
    }
    np.savez_compressed(
        path,
        response=response,
        metadata_json=np.asarray(json.dumps(metadata)),
        viewpoint_positions=viewpoints,
        tx_pos=tx_pos,
        rx_pos=rx_pos,
    )
    return response.shape


def tiny_manifest(shape: tuple[int, ...]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "synthetic_action_observer_v1",
        "dataset": {
            "num_views": int(shape[0]),
            "response_shape": [int(value) for value in shape],
            "response_dtype": "complex64",
        },
        "split": {
            "strategy": "fixed_tail_subsampled",
            "num_train": 2,
            "num_validation": 1,
            "num_test": 1,
            "num_unused": 2,
            "complete_partition": True,
            "test_sealed": True,
            "unused_sealed": True,
            "train_indices": [4, 1],
            "validation_indices": [5],
            "test_indices": [0],
            "unused_indices": [2, 3],
        },
    }


def action_event_npz(path: Path) -> tuple[int, ...]:
    """Write the smallest sealed full-acquisition fixture the real observer accepts."""

    n_train, n_validation, n_test = 16, 16, 1
    n_views = n_train + n_validation + n_test
    n_tx = n_rx = 16
    n_freq = 600
    # The target is deliberately nonzero while the deterministic synthetic
    # renderer below starts from nonzero random scene coefficients.  That
    # produces real data-fit gradients for both position and zero-next-band
    # SH probes without invoking a costly physical range kernel in a unit test.
    view_number = np.arange(n_views, dtype=np.float32)
    target_by_view = (
        np.complex64(0.20 + 0.10j)
        * (1.0 + 0.015 * view_number)
        * np.exp(1j * 0.07 * view_number)
    ).astype(np.complex64)
    # Deliberately vary coherent target by view.  A single initialized global
    # gain can match the first view but cannot erase the remaining data-fit
    # residual, which supplies actual position and next-band probe gradients.
    response = np.broadcast_to(
        target_by_view[:, None, None, None, None],
        (n_views, n_tx, n_rx, 1, n_freq),
    ).copy()
    theta = np.linspace(0.35, 1.25, n_views, dtype=np.float32)
    phi = np.linspace(-0.9, 0.9, n_views, dtype=np.float32)
    viewpoint_positions = np.stack((
        10.0 * np.sin(theta) * np.cos(phi),
        10.0 * np.sin(theta) * np.sin(phi),
        10.0 * np.cos(theta),
    ), axis=1).astype(np.float32)
    element_offsets = np.linspace(-0.025, 0.025, n_tx, dtype=np.float32)
    tx_pos = np.repeat(viewpoint_positions[:, None, :], n_tx, axis=1)
    rx_pos = np.repeat(viewpoint_positions[:, None, :], n_rx, axis=1)
    tx_pos[:, :, 0] += element_offsets[None, :]
    rx_pos[:, :, 1] += element_offsets[None, :]
    metadata = {
        "radar_fc_hz": 10.0e9,
        "radar_bandwidth_hz": 3.0e9,
        "num_adc_samples": n_freq,
        "target_radius_m": 0.15,
    }
    np.savez_compressed(
        path,
        response=response,
        metadata_json=np.asarray(json.dumps(metadata)),
        viewpoint_positions=viewpoint_positions,
        tx_pos=tx_pos,
        rx_pos=rx_pos,
    )
    return response.shape


def action_event_manifest(shape: tuple[int, ...]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "synthetic_b787_action_event_v1",
        "dataset": {
            "num_views": int(shape[0]),
            "response_shape": [int(value) for value in shape],
            "response_dtype": "complex64",
        },
        "split": {
            "strategy": "synthetic_ordered_engineering_roles",
            "num_train": 16,
            "num_validation": 16,
            "num_test": 1,
            "complete_partition": True,
            "test_sealed": True,
            "train_indices": list(range(16)),
            "validation_indices": list(range(16, 32)),
            "test_indices": [32],
        },
    }


def action_event_argv(
    npz_path: Path,
    manifest_path: Path,
    checkpoint_root: Path,
    checkpoint_name: str,
    *,
    epochs: int,
    resume: Path | None = None,
) -> list[str]:
    argv = [
        "--checkpoint-name", checkpoint_name,
        "--checkpoint-root", str(checkpoint_root),
        "--execution-contract-label", "synthetic_b787_action_event_v1",
        "--require-full-resume-state",
        "--data-format", "npz",
        "--npz-path", str(npz_path),
        "--npz-sealed-protocol",
        "--npz-role-manifest", str(manifest_path),
        "--num-train", "16", "--num-val", "16", "--num-test", "1",
        "--num-tx", "16", "--num-rx", "16", "--num-freq-wanted", "600",
        "--epochs", str(epochs), "--step-every", "4", "--loss", "complex",
        "--scene-repr", "point_sh", "--forward-operator", "range",
        "--range-model", "product", "--phase-sign", "-1", "--compute-dtype", "float64",
        # One initial point plus seven spare slots is the smallest actual
        # octant split.  The observer retains its production 1/640 decision,
        # which still selects this one eligible parent deterministically.
        "--granularity", "1", "--extent", "0.15", "--max-points", "8",
        "--adaptive-max-active", "8", "--sh-init-degree", "0", "--sh-max-degree", "1",
        "--init-scale", "0.01", "--bp-init", "0", "--lr", "0.003", "--pos-lr", "0.003",
        "--adam-eps", "1e-20", "--weight-decay", "0", "--t0", "10", "--t-mult", "2",
        "--adaptive-capacity-v2", "--adaptive-refine-every", "1",
        "--adaptive-probe-every", "1", "--adaptive-min-spatial-exposure", "1",
        "--adaptive-min-angular-exposure", "1", "--adaptive-spatial-fraction", str(1.0 / 640.0),
        "--adaptive-angular-fraction", str(1.0 / 640.0), "--adaptive-cooldown-events", "0",
        "--adaptive-child-maturity-events", "1", "--split-max-level", "1",
        "--split-every", "0", "--grow-every", "0", "--prune-every", "0", "--seed", "42",
    ]
    if resume is not None:
        argv.extend(["--resume", str(resume)])
    return argv


@contextmanager
def synthetic_action_range_renderer():
    """Patch both call sites with a pure differentiable range stand-in.

    The scalar depends only on weighted scene positions.  An in-place split
    preserves that scalar exactly in intent: its retained heir keeps the
    parent coefficient/position and every new sibling starts at zero.  It
    still gives delta_raw and the locked SH band data gradients, so this is a
    genuine controller event rather than a hand-authored snapshot.
    """

    original_train = rift_train.range_forward_operator
    original_rift = range_operator.range_forward_operator
    calls: list[dict[str, Any]] = []

    def renderer(
        freqs_full: torch.Tensor,
        kvector_full: torch.Tensor,
        rx_pos: torch.Tensor,
        tx_pos: torch.Tensor,
        scatterer_pos: torch.Tensor,
        scatterer_weights: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kvector_full
        indices = kwargs["freq_indices"]
        compute_dtype = kwargs["compute_dtype"]
        dtype = torch.complex128 if compute_dtype == torch.float64 else torch.complex64
        weights = scatterer_weights.to(dtype)
        weighted_x = weights * scatterer_pos[:, 0].to(dtype)
        scalar = weights.sum() + 0.1 * weighted_x.sum()
        calls.append({
            "freq_count": int(indices.numel()),
            "freq_indices": indices.detach().cpu().clone(),
            "compute_dtype": compute_dtype,
            "range_model": kwargs["range_model"],
            "phase_sign": kwargs["phase_sign"],
        })
        return torch.ones(
            (int(indices.numel()), int(rx_pos.shape[0]), int(tx_pos.shape[0])),
            device=freqs_full.device,
            dtype=dtype,
        ) * scalar

    rift_train.range_forward_operator = renderer
    range_operator.range_forward_operator = renderer
    try:
        yield calls
    finally:
        rift_train.range_forward_operator = original_train
        range_operator.range_forward_operator = original_rift


def _without_wall_time(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_wall_time(item)
            for key, item in value.items()
            if key not in {"elapsed_seconds", "optimizer_step_seconds"}
        }
    if isinstance(value, list):
        return [_without_wall_time(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_without_wall_time(item) for item in value)
    return value


def tiny_argv(
    npz_path: Path, manifest_path: Path, checkpoint_root: Path, checkpoint_name: str, *, epochs: int,
    resume: Path | None = None,
) -> list[str]:
    argv = [
        "--checkpoint-name", checkpoint_name,
        "--checkpoint-root", str(checkpoint_root),
        "--execution-contract-label", "synthetic_action_observer_v1",
        "--require-full-resume-state",
        "--data-format", "npz",
        "--npz-path", str(npz_path),
        "--npz-sealed-protocol",
        "--npz-role-manifest", str(manifest_path),
        "--num-train", "2", "--num-val", "1", "--num-test", "1",
        "--num-tx", "1", "--num-rx", "1", "--num-freq-wanted", "4",
        "--epochs", str(epochs), "--step-every", "1", "--loss", "complex",
        "--scene-repr", "point_sh", "--forward-operator", "brute",
        "--phase-sign", "-1", "--granularity", "2", "--extent", "0.1",
        "--max-points", "8", "--sh-init-degree", "0", "--sh-max-degree", "1",
        "--init-scale", "0.01", "--bp-init", "0", "--lr", "0.001", "--pos-lr", "0.001",
        "--adam-eps", "1e-20", "--t0", "1", "--t-mult", "1",
        "--adaptive-capacity-v2", "--adaptive-refine-every", "1",
        "--adaptive-probe-every", "1", "--adaptive-spatial-fraction", "0",
        "--adaptive-angular-fraction", "0", "--prune-end-epoch", "1", "--seed", "42",
    ]
    if resume is not None:
        argv.extend(["--resume", str(resume)])
    return argv


@contextmanager
def forced_cpu_main(device: torch.device | str = torch.device("cpu")):
    original = rift_train.init_distributed
    rift_train.init_distributed = lambda: (0, 1, device)
    try:
        yield
    finally:
        rift_train.init_distributed = original


class PassiveObserver:
    """Small test-only observer that cannot affect the production trajectory."""

    schema = "synthetic_passive_action_observer_v1"

    def __init__(self) -> None:
        self.events: list[tuple[str, int, int]] = []
        self.updates: list[int] = []
        self.finished = False

    def on_adaptive_event(self, phase: str, **context: Any) -> None:
        self.events.append((phase, int(context["event"]), int(context["logical_optimizer_updates"])))

    def on_optimizer_step(self, **context: Any) -> None:
        self.updates.append(int(context["logical_optimizer_updates"]))

    def finish_training(self) -> None:
        self.finished = True

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "events": list(self.events),
            "updates": list(self.updates),
            "finished": bool(self.finished),
        }

    def restore_checkpoint_state(self, state: dict[str, Any]) -> None:
        if state.get("schema") != self.schema or state.get("finished") is True:
            raise ValueError("synthetic observer cannot restore this state")
        self.events = [tuple(entry) for entry in state["events"]]
        self.updates = [int(value) for value in state["updates"]]


def passive_factory(holder: dict[str, PassiveObserver]):
    def factory(**context: Any) -> PassiveObserver:
        observer = PassiveObserver()
        resume = context["args"].resume
        if resume is not None:
            checkpoint = rift_train.load_tensor_checkpoint(resume, map_location="cpu")
            observer.restore_checkpoint_state(checkpoint["adaptive_event_observer_state"])
        holder["observer"] = observer
        return observer
    return factory


class CleanEpochBoundaryStop(RuntimeError):
    pass


def stage_default_off_and_clean_resume(root: Path) -> None:
    npz_path = root / "tiny.npz"
    shape = tiny_npz(npz_path)
    manifest_path = root / "roles.json"
    manifest_path.write_text(json.dumps(tiny_manifest(shape), indent=2), encoding="utf-8")

    default_root = root / "default_off"
    with forced_cpu_main():
        rift_train.main(tiny_argv(npz_path, manifest_path, default_root, "default", epochs=1))
    default_checkpoint = rift_train.load_tensor_checkpoint(
        default_root / "default" / "checkpoint_final.pth.tar", map_location="cpu"
    )
    check(
        "adaptive_event_observer_state" not in default_checkpoint,
        "ordinary adaptive-capacity training remains default-off and writes no observer payload",
    )

    baseline_root = root / "baseline"
    baseline_holder: dict[str, PassiveObserver] = {}
    with forced_cpu_main():
        rift_train.main(
            tiny_argv(npz_path, manifest_path, baseline_root, "baseline", epochs=2),
            adaptive_event_observer_factory=passive_factory(baseline_holder),
        )
    baseline = rift_train.load_tensor_checkpoint(
        baseline_root / "baseline" / "checkpoint_final.pth.tar", map_location="cpu"
    )

    interrupted_root = root / "interrupted"
    interrupted_holder: dict[str, PassiveObserver] = {}
    original_save = rift_train.save_run_checkpoint

    def stop_after_latest(path: str, *args: Any, **kwargs: Any):
        result = original_save(path, *args, **kwargs)
        if Path(path).name == "checkpoint_latest.pth.tar":
            raise CleanEpochBoundaryStop("test-only stop after durable epoch-boundary checkpoint")
        return result

    rift_train.save_run_checkpoint = stop_after_latest
    try:
        try:
            with forced_cpu_main():
                rift_train.main(
                    tiny_argv(npz_path, manifest_path, interrupted_root, "interrupted", epochs=2),
                    adaptive_event_observer_factory=passive_factory(interrupted_holder),
                )
        except CleanEpochBoundaryStop:
            pass
        else:
            raise AssertionError("test stop did not interrupt after an epoch-boundary latest checkpoint")
    finally:
        rift_train.save_run_checkpoint = original_save

    latest_path = interrupted_root / "interrupted" / "checkpoint_latest.pth.tar"
    latest = rift_train.load_tensor_checkpoint(latest_path, map_location="cpu")
    latest_observer = latest["adaptive_event_observer_state"]
    check(
        latest["epoch"] == 1
        and latest_observer["finished"] is False
        and latest_observer["updates"] == [1, 2]
        and latest.get("rng_state", {}).get("version") == 2
        and isinstance(latest.get("optimizer_state_dict"), dict)
        and isinstance(latest.get("scheduler_state_dict"), dict)
        and isinstance(latest.get("gain_state_dict"), dict)
        and isinstance(latest.get("sealed_npz_protocol_contract"), dict),
        "interrupted action seam writes one complete unfinished epoch-boundary recovery payload",
    )

    resumed_root = root / "resumed"
    resumed_holder: dict[str, PassiveObserver] = {}
    with forced_cpu_main():
        rift_train.main(
            tiny_argv(npz_path, manifest_path, resumed_root, "resumed", epochs=2, resume=latest_path),
            adaptive_event_observer_factory=passive_factory(resumed_holder),
        )
    resumed = rift_train.load_tensor_checkpoint(
        resumed_root / "resumed" / "checkpoint_final.pth.tar", map_location="cpu"
    )
    exact_keys = ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "rng_state")
    check(
        all(nested_equal(baseline[key], resumed[key]) for key in exact_keys)
        and baseline["adaptive_event_observer_state"] == resumed["adaptive_event_observer_state"]
        and resumed["adaptive_event_observer_state"]["updates"] == [1, 2, 3, 4]
        and resumed["adaptive_event_observer_state"]["finished"] is True,
        "observer-assisted clean resume exactly matches an uninterrupted adaptive trajectory without duplicate callbacks",
    )


def actual_observer_factory(
    holder: dict[str, AdaptiveActionGateObserver],
    resume: Path | None,
):
    """Use the production driver factory without its canonical-path launcher guard."""

    return action_driver._build_observer_factory(
        SimpleNamespace(resume=None if resume is None else str(resume)),
        list(range(16)),
        list(range(16, 32)),
        holder,
    )


def stage_actual_observer_event_resume(root: Path) -> None:
    """Exercise the real observer through a data-derived split/unlock and resume.

    This remains a local synthetic fixture.  It validates the observer seam
    around the generic trainer's ordinary immutable controller rather than
    manufacturing a refinement snapshot or using any B787 response row.
    """

    npz_path = root / "full_acquisition.npz"
    shape = action_event_npz(npz_path)
    manifest_path = root / "full_acquisition_roles.json"
    manifest_path.write_text(json.dumps(action_event_manifest(shape), indent=2), encoding="utf-8")

    trajectory_keys = (
        "model_state_dict",
        "optimizer_state_dict",
        "gain_state_dict",
        "scheduler_state_dict",
        "rng_state",
    )
    with synthetic_action_range_renderer() as calls:
        disabled_root = root / "actual_observer_disabled"
        with forced_cpu_main("cpu"):
            rift_train.main(
                action_event_argv(npz_path, manifest_path, disabled_root, "disabled", epochs=4)
            )
        disabled = rift_train.load_tensor_checkpoint(
            disabled_root / "disabled" / "checkpoint_final.pth.tar", map_location="cpu"
        )

        uninterrupted_root = root / "actual_observer_uninterrupted"
        uninterrupted_holder: dict[str, AdaptiveActionGateObserver] = {}
        with forced_cpu_main("cpu"):
            rift_train.main(
                action_event_argv(npz_path, manifest_path, uninterrupted_root, "uninterrupted", epochs=4),
                adaptive_event_observer_factory=actual_observer_factory(uninterrupted_holder, None),
            )
        uninterrupted = rift_train.load_tensor_checkpoint(
            uninterrupted_root / "uninterrupted" / "checkpoint_final.pth.tar", map_location="cpu"
        )
        uninterrupted_observer = uninterrupted["adaptive_event_observer_state"]
        check(
            all(nested_equal(disabled[key], uninterrupted[key]) for key in trajectory_keys)
            and "adaptive_event_observer_state" not in disabled,
            "the real observer is diagnostic-only: it leaves the fixed four-epoch scene, gain, optimizer, scheduler, and RNG trajectory unchanged",
        )

        interrupted_root = root / "actual_observer_interrupted"
        interrupted_holder: dict[str, AdaptiveActionGateObserver] = {}
        original_save = rift_train.save_run_checkpoint

        def stop_after_event_checkpoint(path: str, *args: Any, **kwargs: Any):
            result = original_save(path, *args, **kwargs)
            if Path(path).name == "checkpoint_latest.pth.tar":
                raise CleanEpochBoundaryStop(
                    "test-only stop after the durable post-event epoch-boundary checkpoint"
                )
            return result

        rift_train.save_run_checkpoint = stop_after_event_checkpoint
        try:
            try:
                with forced_cpu_main("cpu"):
                    rift_train.main(
                        action_event_argv(npz_path, manifest_path, interrupted_root, "interrupted", epochs=4),
                        adaptive_event_observer_factory=actual_observer_factory(interrupted_holder, None),
                    )
            except CleanEpochBoundaryStop:
                pass
            else:
                raise AssertionError("actual-observer test did not stop after its post-event checkpoint")
        finally:
            rift_train.save_run_checkpoint = original_save

        latest_path = interrupted_root / "interrupted" / "checkpoint_latest.pth.tar"
        latest = rift_train.load_tensor_checkpoint(latest_path, map_location="cpu")
        latest_observer = latest["adaptive_event_observer_state"]
        first_record = latest_observer["records"][0]
        check(
            latest["epoch"] == 1
            and latest_observer["finished"] is False
            and len(latest_observer["records"]) == 1
            and first_record["n_split"] == 1
            and first_record["n_grown"] == 1
            and first_record["spatial_indices"] == [0]
            and first_record["split_parent_indices"] == [0]
            and math.isfinite(first_record["spatial_scores"][0])
            and first_record["spatial_scores"][0] > 0.0
            and math.isfinite(first_record["angular_scores"][0])
            and first_record["angular_scores"][0] > 0.0
            and first_record["spatial_parent_levels_before"] == [0]
            and first_record["split_parent_levels_after"] == [1]
            and first_record["unlocked_indices"] == [0]
            and len(first_record["children"]) == 7
            and first_record["child_birth_event_ok"]
            and first_record["children_zero_at_birth"]
            and first_record["child_optimizer_rows_zero_at_birth"]
            and first_record["unlocked_band_zero_at_unlock"]
            and first_record["unlocked_band_optimizer_columns_zero"]
            and first_record["probe_full_coherent_prediction_preserved"]
            and isinstance(latest.get("sealed_npz_protocol_contract"), dict),
            "the durable post-event checkpoint records one data-derived spatial split, SH unlock, exact parent provenance, and sealed roles",
        )

        resumed_holder: dict[str, AdaptiveActionGateObserver] = {}
        with forced_cpu_main("cpu"):
            rift_train.main(
                action_event_argv(
                    npz_path, manifest_path, interrupted_root, "interrupted", epochs=4, resume=latest_path
                ),
                adaptive_event_observer_factory=actual_observer_factory(resumed_holder, latest_path),
            )
        resumed = rift_train.load_tensor_checkpoint(
            interrupted_root / "interrupted" / "checkpoint_final.pth.tar", map_location="cpu"
        )
        resumed_observer = resumed["adaptive_event_observer_state"]
        post_event = resumed_observer["pending"][0]
        check(
            all(nested_equal(uninterrupted[key], resumed[key]) for key in trajectory_keys)
            and nested_equal(
                _without_wall_time(uninterrupted_observer), _without_wall_time(resumed_observer)
            )
            and resumed_observer["finished"] is True
            and post_event["checked_after_update"]
            and post_event["first_post_event_optimizer_update"] == 5
            and post_event["child_updated"]
            and post_event["band_updated"]
            and [entry["n_split"] for entry in resumed_observer["records"]] == [1, 0, 0, 0]
            and [entry["n_grown"] for entry in resumed_observer["records"]] == [1, 0, 0, 0]
            and len(resumed_observer["pending"]) == 1,
            "a real observer restores its original baselines and action evidence across the generic clean-resume seam without duplicate updates",
        )
        resumed_observer_instance = resumed_holder["observer"]
        report = resumed_observer_instance.finalize()
        check(
            report["pass"] is True
            and report["checks"]["spatial_action_with_later_child_update"] is True
            and report["checks"]["angular_action_with_later_band_update"] is True
            and report["checks"]["expected_update_count"] is True
            and report["checks"]["completed_observer_timeline"] is True
            and isinstance(resumed_observer_instance.device, torch.device)
            and resumed_observer_instance.device == torch.device("cpu")
            and bool(calls)
            and all(
                call["freq_count"] == 600
                and torch.equal(call["freq_indices"], torch.arange(600))
                and call["compute_dtype"] == torch.float64
                and call["range_model"] == "product"
                and call["phase_sign"] == -1.0
                for call in calls
            ),
            "the real observer normalizes the production-shaped string device and finalizes a passing four-epoch synthetic range-call action audit",
        )

        # The allocated PACE route is CUDA and its postflight resource gate
        # depends on these three calls.  Exercise the real finalize method's
        # CUDA branch without allocating a GPU by replacing only the CUDA
        # counters around an already-complete real observer.
        original_device = resumed_observer_instance.device
        original_role_metrics = resumed_observer_instance._role_metrics
        original_synchronize = torch.cuda.synchronize
        original_allocated = torch.cuda.max_memory_allocated
        original_reserved = torch.cuda.max_memory_reserved
        cuda_calls: list[tuple[str, torch.device]] = []

        def fake_synchronize(device: torch.device) -> None:
            cuda_calls.append(("synchronize", device))

        def fake_allocated(device: torch.device) -> int:
            cuda_calls.append(("allocated", device))
            return 123

        def fake_reserved(device: torch.device) -> int:
            cuda_calls.append(("reserved", device))
            return 456

        try:
            resumed_observer_instance.device = torch.device("cuda:0")
            resumed_observer_instance._role_metrics = lambda *_args, **_kwargs: copy.deepcopy(
                report["final_metrics"]
            )
            torch.cuda.synchronize = fake_synchronize
            torch.cuda.max_memory_allocated = fake_allocated
            torch.cuda.max_memory_reserved = fake_reserved
            cuda_report = resumed_observer_instance.finalize()
        finally:
            resumed_observer_instance.device = original_device
            resumed_observer_instance._role_metrics = original_role_metrics
            torch.cuda.synchronize = original_synchronize
            torch.cuda.max_memory_allocated = original_allocated
            torch.cuda.max_memory_reserved = original_reserved
        expected_cuda_device = torch.device("cuda:0")
        check(
            cuda_report["peak_torch_allocated_bytes"] == 123
            and cuda_report["peak_torch_reserved_bytes"] == 456
            and cuda_calls == [
                ("synchronize", expected_cuda_device),
                ("allocated", expected_cuda_device),
                ("reserved", expected_cuda_device),
            ],
            "the actual observer finalize path retains CUDA synchronization and peak-memory accounting for a normalized cuda:0 device",
        )


def main() -> None:
    check(
        B787_3200_CANONICAL_NPZ_PATH
        == "/storage/home/hcoda1/1/dbao31/r-jromberg3-0/RIFT/data/"
        "b787_fmcw_16t16r_10ghz_bw3ghz_r10m_sphere10k.npz",
        "the action gate binds the user-specified remote B787 archive rather than a local dataset-folder assumption",
    )
    with tempfile.TemporaryDirectory(prefix="rift_b7873200_action_gate_") as temporary:
        root = Path(temporary)
        stage_manifest_and_argv(root)
        stage_driver_identity(root)
        stage_observer_render_and_state()
        stage_nan_probe_rejection()
        stage_default_off_and_clean_resume(root)
        stage_actual_observer_event_resume(root)
    print(f"PASS: {CHECKS} bounded B787 action-gate local checks")


if __name__ == "__main__":
    main()
