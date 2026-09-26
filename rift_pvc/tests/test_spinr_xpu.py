"""SpINR PVC numerical, recipe and recovery gates on synthetic acquisitions."""
import copy

import pytest
import torch

import train_spinr_style as original
import train_spinr_style_pvc as trainer
from rift.config import cc
from rift.forward_operator import get_kvector
from rift.spinr_quadrature_audit import observe_rule as cpu_observe
from rift_pvc import accelerator
from rift_pvc import spinr_gotcha_training as native
from rift_pvc.spinr_quadrature_audit import observe_rule
from tests.test_spinr_fidelity import SmoothField, physics, dense_response
from tests.test_spinr_gotcha import native_dataset, TinyField, config


@pytest.fixture(params=["cpu", "xpu"])
def device(request, monkeypatch):
    if request.param == "xpu" and not torch.xpu.is_available():
        pytest.skip("requires an allocated PVC card")
    monkeypatch.setenv("RIFT_ACCELERATOR", request.param)
    return torch.device(request.param)


def test_cli_and_recipes_preserve_current_and_historical_contracts(monkeypatch):
    monkeypatch.setenv("RIFT_ACCELERATOR", "cpu")
    for name in ("legacy-midpoint", "paper-v1", "paper-v1-direct",
                 "budget48-direct", "budget48-direct-1500"):
        assert trainer._recipe_identity(name) == original._recipe_identity(name)
    assert trainer.GOTCHA_BACKEND == original.GOTCHA_BACKEND
    args = trainer.parse_args(["--checkpoint-name", "smoke", "--host-rss-limit-gib", "32",
                               "--recipe", "budget48-direct", "--device", "xpu"])
    assert args.device == "xpu" and args.epochs == 150 and args.grid_size == 48
    with pytest.raises(RuntimeError, match="active backend"):
        trainer._validate_cli_recipe(args)
    # Importing the PVC modules must not rebind CUDA trainer functions.
    assert original.capture_rng_state.__module__ == "train"
    assert trainer.capture_rng_state.__module__ == "rift_pvc.train_rng"


def test_rng_roundtrip_and_foreign_backend_rejection(device):
    trainer._set_seed(42)
    state = trainer.capture_rng_state()
    expected = torch.rand(17, device=device)
    trainer.restore_rng_state(state, require_complete=True)
    torch.testing.assert_close(torch.rand(17, device=device), expected, rtol=0, atol=0)
    if device.type == "xpu":
        assert "torch_xpu_all" in state and "torch_cuda_all" not in state
    foreign = copy.deepcopy(state)
    foreign.pop("torch_xpu_all", None)
    foreign["torch_cuda_all"] = [state["torch_cpu"].clone()]
    foreign["accelerator_backend"] = "cuda"
    before = trainer.capture_rng_state()
    with pytest.raises(ValueError, match="complete usable RNG"):
        trainer._validate_complete_rng_state_without_changing_runtime(foreign)
    assert trainer._checkpoint_tree_equal(before, trainer.capture_rng_state())


def test_direct_training_update_matches_cpu(device):
    frequencies, rx, tx = physics()
    points = torch.tensor([[.023, -.05, .08], [-.06, .03, -.01]], dtype=torch.float64)
    volumes = torch.tensor([.0002, .0003], dtype=torch.float64)
    cpu_model = SmoothField()
    model = copy.deepcopy(cpu_model).to(device)
    observed = dense_response(frequencies, rx, tx, points+.0002,
                              cpu_model(points).detach()*.7, volumes)
    power = float(observed.abs().square().mean())
    entries = [(observed, rx+i*.0001, tx-i*.0001) for i in range(4)]

    class Views:
        def tensor_view(self, i, *, device):
            return tuple(value.to(device) for value in entries[i])

        def role_ids(self, role):
            return tuple(range(4))

    def update(module, field, target):
        return module.logical_batch_update(
            model=field, optimizer=torch.optim.Adam(field.parameters(), lr=1e-4),
            source_ids=range(4), views=Views(), points_m=points.to(target),
            cell_volume_m3=volumes.to(target), initial_output_scale=1.,
            frequencies_hz=frequencies.to(target), kvector=get_kvector(frequencies, cc).to(target),
            training_mean_raw_power=power, neural_point_tile=1, renderer_point_tile=1,
            pair_tile=2, device=target, scene_bins=True, direct_bins=True)

    reference = update(original, cpu_model, torch.device("cpu"))
    result = update(trainer, model, device)
    assert result == pytest.approx(reference, rel=3e-6, abs=1e-9)
    torch.testing.assert_close(model.coefficients.cpu(), cpu_model.coefficients, rtol=3e-6, atol=1e-9)
    torch.testing.assert_close(model.coefficients.grad.cpu(), cpu_model.coefficients.grad,
                               rtol=3e-6, atol=1e-9)


def test_quadrature_gradients_and_state_preservation(device):
    frequencies, rx, tx = physics()
    points = torch.tensor([[.001, -.002, .003], [-.003, .002, -.001]], dtype=torch.float64)
    reference = SmoothField()
    target = dense_response(frequencies, rx, tx, points, reference(points).detach(), .0001)
    kwargs = dict(observations=[(7, target, rx, tx)], frequencies_hz=frequencies,
                  rule={"label": "test", "kind": "midpoint", "parent_grid": 2, "nodes_per_cell": 1},
                  support_m=.008, initial_output_scale=1.,
                  training_mean_raw_power=float(target.abs().square().mean()), scene_bins=True,
                  direct_bins=True, neural_point_tile=7, renderer_point_tile=7, pair_tile=2)
    expected = cpu_observe(model=reference, **kwargs)
    model = copy.deepcopy(reference).to(device)
    previous_grad = torch.ones_like(model.coefficients)
    model.coefficients.grad = previous_grad
    rng = trainer.capture_rng_state()
    actual = observe_rule(model=model, device=device, **kwargs)
    assert model.coefficients.grad is previous_grad and model.training
    assert trainer._checkpoint_tree_equal(rng, trainer.capture_rng_state())
    assert actual["device_backend"] == device.type
    assert actual["peak_cuda_allocated_bytes"] is None
    for a, b in zip(actual["signals"], expected["signals"]):
        torch.testing.assert_close(a, b, rtol=3e-6, atol=1e-13)
    torch.testing.assert_close(actual["parameter_gradients"]["coefficients"],
                               expected["parameter_gradients"]["coefficients"], rtol=3e-6, atol=1e-8)


@pytest.mark.parametrize("stop_after", [1, 3])
def test_native_gotcha_exact_resume(native_dataset, tmp_path, monkeypatch, device, stop_after):
    monkeypatch.setattr(native, "SpinrStyleINR", TinyField)
    options = config()
    result = trainer.run_gotcha(dataset=native_dataset, output_dir=tmp_path/"full",
                               config=options, device=device, resume=None)
    assert result["status"] == "complete"
    expected = trainer.load_tensor_checkpoint(tmp_path/"full/checkpoint_latest.pt", map_location="cpu")
    count = {"updates": 0}
    original_update = native.batch_update

    def update(*args):
        result = original_update(*args)
        count["updates"] += 1
        return result

    monkeypatch.setattr(native, "batch_update", update)
    result = native.run(native_dataset, tmp_path/"resume", options, device=device,
                        should_stop=lambda: count["updates"] >= stop_after)
    assert result["status"] == "interrupted"
    latest = tmp_path/"resume/checkpoint_latest.pt"
    partial = trainer.load_tensor_checkpoint(latest, map_location="cpu")
    assert partial["epoch"] == 0 and partial["cursor"] == stop_after
    result = trainer.run_gotcha(dataset=native_dataset, output_dir=latest.parent, config=options,
                               device=device, resume=latest)
    assert result["status"] == "complete"
    actual = trainer.load_tensor_checkpoint(latest, map_location="cpu")
    for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "rng_state",
                "history", "optimization_coverage", "head_updates", "initial_scales", "training_statistics"):
        assert trainer._checkpoint_tree_equal(actual[key], expected[key]), key
    assert actual["optimization_coverage"]["pulse_exposures"] == 16

