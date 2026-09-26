#!/usr/bin/env python3
"""Tiny no-data runtime checks for the combined B787 engineering smoke."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rift.sugavanam_ertin import FourierFeatureSDF
from rift.sugavanam_ertin_b7873200_real_smoke import (
    SMOKE_STAGE1_EXTENT,
    SMOKE_STAGE1_GRANULARITY,
    SMOKE_STAGE1_CHECKPOINT_NAME,
    SMOKE_STAGE1_EXPECTED_UPDATES,
    SMOKE_STAGE1_EPOCHS,
    SMOKE_STAGE1_LABEL,
    complex_signal_statistics,
    analytic_sdf_shell_diagnostic,
    smoke_stage1_argv,
    smoke_stage1_checkpoint_dir,
    smoke_stage1_record,
    validate_generic_stage1_final,
)
from rift.sugavanam_ertin_b7873200_stage2_v1 import lifecycle_contract_record
from rift.sugavanam_ertin_stage2_runtime_v1 import RuntimeContractError
import train_sugavanam_ertin_smoke as smoke_driver
import train_sugavanam_ertin_stage2 as stage2


def _expect_failure(function, message: str) -> None:
    try:
        function()
    except (AssertionError, RuntimeContractError, smoke_driver.RealSmokeContractError):
        return
    raise AssertionError(message)


def test_numeric_normalization() -> None:
    stats = complex_signal_statistics(np.asarray([3.0 + 4.0j, 0.0 + 0.0j], dtype=np.complex64))
    assert stats == {"raw_complex_rms": np.sqrt(12.5), "zero_reference_mse": 12.5, "sample_count": 2}


def test_geometry_feasibility_g8_vs_g16() -> None:
    centered = np.asarray(
        [[0.12, 0.0, 0.0], [-0.12, 0.0, 0.0], [0.0, 0.12, 0.0],
         [0.0, -0.12, 0.0], [0.0, 0.0, 0.12], [0.0, 0.0, -0.12]],
        dtype=np.float64,
    )
    kwargs = {"extent": 0.15, "radius_quantile": 0.5, "radius_cap_fraction": 0.65}
    g8 = analytic_sdf_shell_diagnostic(centered, granularity=8, **kwargs)
    g16 = analytic_sdf_shell_diagnostic(centered, granularity=16, **kwargs)
    assert g8["feasible"] is False and g8["shell_clearance"] <= 0.0
    assert g16["feasible"] is True and g16["shell_clearance"] > 0.0
    assert np.allclose(g16["center"], (0.0, 0.0, 0.0))
    assert SMOKE_STAGE1_EXTENT == 0.15 and SMOKE_STAGE1_GRANULARITY == 16


def test_stage1_path_agreement() -> None:
    root = Path("/tmp/se_b7873200_path_probe")
    argv = smoke_stage1_argv(
        npz_path="/canonical/data.npz",
        manifest_path="/canonical/roles.json",
        checkpoint_root=root,
    )
    values = {argv[index]: argv[index + 1] for index in range(0, len(argv) - 1) if argv[index].startswith("--")}
    expected = smoke_stage1_checkpoint_dir(root)
    actual = Path(values["--checkpoint-root"]) / values["--checkpoint-name"]
    assert values["--checkpoint-name"] == SMOKE_STAGE1_CHECKPOINT_NAME
    assert actual == expected


def test_smoke_stage2_recipe_constructor_and_backward() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise AssertionError("runtime smoke requires exactly one CUDA device")
    recipe = smoke_driver._smoke_stage2_recipe()
    assert recipe["schema"] == "rift_sugavanam_ertin_b7873200_stage2_engineering_smoke_recipe_v3"
    assert recipe["recipe_revision"] == 3
    assert recipe["init_steps"] == 1000
    assert recipe["init_lr"] == 5.0e-4
    assert recipe["init_batch"] == 2048
    assert recipe["init_log_every"] == 100
    assert recipe["n_fourier"] == 5
    assert recipe["hidden_dim"] == 128
    assert recipe["sdf_skip_layer_index"] == 4
    assert recipe["sdf_min_layers"] == 5
    assert recipe["n_layers"] == recipe["sdf_min_layers"] == 5
    device = torch.device("cuda")
    model = FourierFeatureSDF(
        extent=0.15,
        n_fourier=int(recipe["n_fourier"]),
        fourier_scale=float(recipe["fourier_scale"]),
        hidden_dim=int(recipe["hidden_dim"]),
        n_layers=int(recipe["n_layers"]),
        seed=int(recipe["seed"]),
    ).to(device)
    encoded_dim = 3 + 2 * int(recipe["n_fourier"])
    assert len(model.layers) == int(recipe["n_layers"])
    assert model.layers[int(recipe["sdf_skip_layer_index"])].in_features == encoded_dim + int(recipe["hidden_dim"])
    points = torch.zeros((2, 3), device=device, dtype=torch.float32, requires_grad=True)
    values = model(points)
    assert torch.isfinite(values).all()
    values.square().mean().backward()
    assert points.grad is not None and torch.isfinite(points.grad).all()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)


class _FakeTensor:
    def __init__(self, values: object) -> None:
        self.values = np.asarray(values, dtype=np.float64)

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def clone(self) -> "_FakeTensor":
        return _FakeTensor(self.values.copy())

    def numpy(self) -> np.ndarray:
        return self.values

    def __sub__(self, other: "_FakeTensor") -> "_FakeTensor":
        return _FakeTensor(self.values - other.values)


class _FakeBoolTensor(_FakeTensor):
    @property
    def dtype(self):
        return "torch.bool"


class _FakeScalar:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def detach(self) -> "_FakeScalar":
        return self

    def cpu(self) -> "_FakeScalar":
        return self

    def item(self) -> float:
        return self.value


class _FakeParameter(_FakeTensor):
    def __init__(self, values: object, gradient: object) -> None:
        super().__init__(values)
        self.grad = _FakeTensor(gradient)


class _FakeModel:
    def __init__(self, parameter: _FakeParameter) -> None:
        self.parameter = parameter

    def named_parameters(self):
        return [("w", self.parameter)]


def test_observer_runtime_names() -> None:
    observer = smoke_driver.Stage1EngineeringObserver.__new__(smoke_driver.Stage1EngineeringObserver)
    observer._gain = None
    observer.start_epoch = 0
    observer.logical_updates_at_start = 0
    observer.initial_readout = {"train": {}, "validation": {}}
    observer.final_readout = {"train": {}, "validation": {}}
    parameter = _FakeParameter([1.25], [2.0])
    observer._before_parameters = {"model.w": _FakeTensor([1.0])}
    observer._before_gradients = {"model.w": _FakeTensor([2.0])}
    observer.steps = []
    observer.on_optimizer_step(
        model=_FakeModel(parameter),
        optimizer=object(),
        epoch=1,
        logical_optimizer_updates=1,
        grad_norm=2.0,
    )
    result = observer.result()
    assert result["observed_optimizer_steps"] == 1
    assert result["observed_nonzero_parameter_updates"] == 1


def test_validate_generic_stage1_final_composition() -> None:
    execution_observation = {
        "num_train": 16,
        "num_validation": 16,
        "num_reserved_test": 1_000,
        "num_freq_wanted": 600,
        "sealed_npz_protocol": True,
        "mapping_identity": "execution-contract",
    }
    engineering_observation = {
        "schema": "rift_sugavanam_ertin_stage1_engineering_observation_v1",
        "start_epoch": 0,
        "logical_optimizer_updates_at_start": 0,
        "observed_optimizer_steps": 1,
        "observed_nonzero_parameter_updates": 1,
        "logical_optimizer_updates_final": 1,
        "mapping_identity": "engineering-observation",
        "steps": [{
            "epoch": 1,
            "logical_optimizer_updates": 1,
            "reported_grad_norm": 1.0,
            "observed_gradient_l2": 1.0,
            "finite_gradients": True,
            "nonzero_gradient_tensors": 1,
            "parameter_delta_l2": 1.0,
            "max_abs_parameter_delta": 1.0,
            "finite_parameter_delta": True,
            "nonzero_parameter_update": True,
        }],
        "initial": {"train": {"loss": 1.0, "residual_power": 1.0, "zero_reference_power": 4.0, "relative_mse": 0.25, "relative_l2": 0.5}, "validation": {"loss": 1.0, "residual_power": 1.0, "zero_reference_power": 4.0, "relative_mse": 0.25, "relative_l2": 0.5}},
        "final": {"train": {"loss": 1.0, "residual_power": 1.0, "zero_reference_power": 4.0, "relative_mse": 0.25, "relative_l2": 0.5}, "validation": {"loss": 1.0, "residual_power": 1.0, "zero_reference_power": 4.0, "relative_mse": 0.25, "relative_l2": 0.5}},
    }
    manifest_contract = {"schema": "composition-probe"}
    state = {
        "epoch": SMOKE_STAGE1_EPOCHS,
        "execution_contract": {
            "label": SMOKE_STAGE1_LABEL,
            "observation": execution_observation,
            "scene": {"representation": "grid", "granularity": 16, "extent_m": 0.15, "initial_scale": 0.0, "backprojection_views": 16, "normalize_scene_scale": False},
            "physics": {"forward_operator": "range", "range_model": "product", "phase_sign": -1.0, "compute_dtype": "float64", "num_rx": 16, "num_tx": 16, "point_chunk": 4096, "pair_chunk": 32},
            "fit": {"epochs": SMOKE_STAGE1_EPOCHS, "loss": "complex", "step_every": 1, "learning_rate": 0.003, "weight_decay": 0.0, "l1_weight": 3.0e-7, "adam_eps": 1.0e-20, "checkpoint_metric": "val", "seed": 42},
        },
        "sealed_npz_protocol_contract": manifest_contract,
        "model_state_dict": {"w_re": _FakeTensor([1.0]), "w_im": _FakeTensor([1.0]), "active_mask": _FakeBoolTensor([True]), "grid_positions": _FakeTensor([1.0])},
        "gain_state_dict": {},
        "optimizer_state_dict": {"state": {0: {"step": _FakeScalar(SMOKE_STAGE1_EXPECTED_UPDATES)}}},
        "scheduler_state_dict": {},
        "rng_state": {"python": "probe"},
        "loss": 1.0,
    }
    audit = validate_generic_stage1_final(
        state,
        manifest_contract=manifest_contract,
        observation=engineering_observation,
    )
    assert audit["observed_fit"]["observed_optimizer_steps"] == 1
    assert audit["observed_fit"]["initial_metrics"]["train"]["relative_mse"] == 0.25


def test_record_to_lifecycle() -> None:
    record = smoke_stage1_record(
        state={"execution_contract": {"label": SMOKE_STAGE1_LABEL}, "epoch": 2, "loss": 1.0},
        manifest_contract={"schema": "test-contract"},
        normalization={"raw_complex_rms": 2.0, "zero_reference_train_mse": 4.0},
        final_path="/tmp/stage1_engineering_smoke_final.pth.tar",
        observation={"observed_nonzero_parameter_updates": 1, "observed_optimizer_steps": 1},
    )
    assert "acquisition_identity" not in record
    assert "acquisition_identity" in record["stage1_recipe"]
    provenance = {
        "contract": {
            "method": "engineering smoke",
            "campaign_identity": "engineering-campaign",
            "artifact_identity": "engineering-artifact",
            "policy_identity": "strict-policy",
            "implementation_kind": "engineering-only",
        },
        "stage1_record": record,
        "stage2_recipe": {"steps": 1},
    }
    lifecycle = lifecycle_contract_record(provenance)
    stage1 = lifecycle["stage1"]
    assert "acquisition_identity" not in stage1["stage1_recipe_without_acquisition_arrays"]
    assert stage1["sealed_protocol_identity"] == {"schema": "test-contract"}


def test_cuda_rng_contract() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise AssertionError("runtime smoke requires exactly one CUDA device")
    torch.cuda.manual_seed_all(90210)
    payload = stage2._capture_rng(include_cuda=True)
    expected = torch.rand(8, device="cuda")
    torch.rand(8, device="cuda")
    stage2._restore_rng(payload, require_cuda=True)
    actual = torch.rand(8, device="cuda")
    assert torch.equal(expected, actual)
    _expect_failure(
        lambda: stage2._validate_rng_state({key: value for key, value in payload.items() if key != "torch_cuda"}, require_cuda=True),
        "CUDA resume validation accepted a missing CUDA RNG state",
    )
    bad = dict(payload)
    bad["torch_cuda"] = []
    _expect_failure(
        lambda: stage2._restore_rng(bad, require_cuda=True),
        "CUDA RNG restoration accepted the wrong topology",
    )


def test_resource_acceptance() -> None:
    accepted = {
        "wall_seconds": 1.0,
        "process_peak_rss_bytes": 1024,
        "cuda_peak_allocated_bytes": 0,
        "cuda_peak_reserved_bytes": 0,
    }
    smoke_driver._validate_resource_snapshot(accepted, label="runtime probe")
    rejected = dict(accepted)
    rejected["process_peak_rss_bytes"] = smoke_driver.SMOKE_MEMORY_LIMIT_BYTES + 1
    _expect_failure(
        lambda: smoke_driver._validate_resource_snapshot(rejected, label="runtime probe"),
        "resource acceptance ignored the declared memory envelope",
    )


def main() -> None:
    test_numeric_normalization()
    test_geometry_feasibility_g8_vs_g16()
    test_stage1_path_agreement()
    test_smoke_stage2_recipe_constructor_and_backward()
    test_observer_runtime_names()
    test_validate_generic_stage1_final_composition()
    test_record_to_lifecycle()
    test_cuda_rng_contract()
    test_resource_acceptance()
    print("SE_B7873200_REAL_SMOKE_RUNTIME_PASS: numeric, geometry, paths, lifecycle, CUDA RNG, resources", flush=True)


if __name__ == "__main__":
    main()
