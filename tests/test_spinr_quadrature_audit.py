"""Bounded checks for frozen-checkpoint diagnostics; no production data/fitting."""
import copy
import hashlib
import json

import numpy as np
import pytest
import torch

import train_spinr_style as trainer
from rift.spinr_fidelity import scene_range_bin_mask
from rift.spinr_quadrature_audit import (quadrature_plan, observe_rule,
                                         compare_observations, difference_check)
from rift.spinr_style import gauss_legendre_cell_grid, spinr_style_objective
from tests.test_spinr_fidelity import SmoothField, physics, dense_response, checkpoint_fixture


def test_plan_preserves_saved_rule_and_refines_both_order_and_space():
    for recipe_name, base, nodes in (("legacy-midpoint", 48, 1), ("paper-v1", 96, 2)):
        rules = quadrature_plan(trainer._recipe_identity(recipe_name))
        assert rules[0]["parent_grid"] == base
        assert rules[0]["nodes_per_cell"] == nodes
        assert rules[0]["integration_points"] == (base*nodes)**3
        assert rules[1]["parent_grid"] == base and rules[1]["nodes_per_cell"] == 3
        assert rules[2]["parent_grid"] > base and rules[2]["nodes_per_cell"] == 3
        with pytest.raises(ValueError, match="larger"):
            quadrature_plan(trainer._recipe_identity(recipe_name), base)


@pytest.mark.parametrize("scene_bins,direct_bins", [(False, False), (True, False), (True, True)])
def test_full_parameter_gradient_matches_independent_autograd_and_restores_state(scene_bins, direct_bins):
    f, rx, tx = physics()
    points, volumes = gauss_legendre_cell_grid(2, nodes_per_cell=2, support_m=.008)
    model = SmoothField()
    original = copy.deepcopy(model.state_dict())
    previous_grad = torch.arange(4, dtype=torch.float64)
    model.coefficients.grad = previous_grad
    reference = copy.deepcopy(model)
    reference.zero_grad()
    target = dense_response(f, rx, tx, points+.0001, model(points).detach()*.7, volumes)
    observations = [(7, target, rx, tx), (23, target*.9, rx+.001, tx-.001)]
    power = float(target.abs().square().mean())
    expected_signals, expected_loss = [], 0.
    for _, observed, r, t in observations:
        prediction = dense_response(f, r, t, points, reference(points), volumes)
        expected_signals.append(prediction.detach())
        mask = scene_range_bin_mask(f, r, t, support_m=.008) if scene_bins else None
        expected_loss = expected_loss + spinr_style_objective(
            prediction, observed, training_mean_raw_power=power, range_bin_mask=mask)/2
    expected_loss.backward()
    rng = torch.get_rng_state().clone()
    actual = observe_rule(
        model=model, observations=observations, frequencies_hz=f,
        rule={"label": "tiny", "kind": "gauss_legendre", "parent_grid": 2, "nodes_per_cell": 2},
        support_m=.008, initial_output_scale=1., training_mean_raw_power=power,
        scene_bins=scene_bins, direct_bins=direct_bins, neural_point_tile=7, renderer_point_tile=19, pair_tile=2)
    for predicted, expected in zip(actual["signals"], expected_signals):
        torch.testing.assert_close(predicted, expected, rtol=2e-7, atol=1e-14)
    torch.testing.assert_close(actual["parameter_gradients"]["coefficients"],
                               reference.coefficients.grad, rtol=3e-6, atol=1e-8)
    assert actual["native_spectral_objective"] == pytest.approx(float(expected_loss.detach()), rel=2e-6)
    assert model.training and model.coefficients.grad is previous_grad
    assert torch.equal(torch.get_rng_state(), rng)
    for name, value in original.items():
        assert torch.equal(model.state_dict()[name], value)


def test_observer_restores_existing_gradient_and_mode_after_failure():
    model = SmoothField().eval()
    previous = torch.ones(4, dtype=torch.float64)
    model.coefficients.grad = previous
    f, rx, tx = physics()
    with pytest.raises(ValueError, match="response shape"):
        observe_rule(model=model, observations=[(1, torch.zeros(3), rx, tx)], frequencies_hz=f,
                     rule={"kind": "midpoint", "parent_grid": 2, "nodes_per_cell": 1},
                     support_m=.008, initial_output_scale=1., training_mean_raw_power=1., scene_bins=False)
    assert not model.training and model.coefficients.grad is previous


def observation_record():
    return {"rule": {"label": "synthetic"}, "source_ids": [7, 23],
            "signals": [torch.ones(3, dtype=torch.complex128) for _ in range(2)],
            "observed_energies": [3., 3.],
            "parameter_gradients": {"large": torch.ones(4, dtype=torch.float64)*1e4,
                                    "small": torch.tensor([1., -1.], dtype=torch.float64)}}


def test_gradient_comparison_detects_cancellation_and_layer_error_hidden_in_global_norm():
    reference = observation_record()
    candidate = copy.deepcopy(reference)
    candidate["parameter_gradients"]["small"] *= -1
    # Both sum projections are zero, and large layers dominate the global norm.
    assert candidate["parameter_gradients"]["small"].sum() == reference["parameter_gradients"]["small"].sum()
    report = compare_observations(candidate, reference)
    assert report["parameter_gradient"]["pass"]
    assert not report["per_parameter_tensor"]["small"]["pass"]
    assert not report["checks_pass"]


def test_signal_check_exposes_failing_view_despite_pooled_pass_and_handles_zero_reference():
    reference = observation_record()
    candidate = copy.deepcopy(reference)
    candidate["signals"][1] += .012
    report = compare_observations(candidate, reference)
    assert report["signal"]["pass"]
    assert not report["per_view_signal"][1]["pass"]
    assert not report["checks_pass"]
    check = difference_check(1e-11, 0., relative_tolerance=.01, absolute_tolerance=1e-10)
    assert check["pass"] and check["relative_difference"] is None
    assert check["mode"] == "near_zero_absolute"
    candidate["source_ids"] = [7, 24]
    with pytest.raises(ValueError, match="identical"):
        compare_observations(candidate, reference)


@pytest.fixture
def saved_collection_checkpoint(tmp_path):
    from tests.test_rift_dataset import contract
    from rift.spinr_style import build_spinr_style_acquisition_identity
    state, arrays, _ = checkpoint_fixture()
    expected = contract("b787")
    state["sealed_npz_protocol_contract"] = expected
    state["acquisition_identity"] = build_spinr_style_acquisition_identity(arrays, expected)
    state["normalization"]["initial_scale_training_ids"] = expected["role_ids"]["train"][:32]
    state["optimization_coverage"] = trainer.optimization_coverage(expected["role_ids"]["train"], 0, 0)
    path = tmp_path/"checkpoint.pt"
    torch.save(state, path)
    return path, arrays, expected


def test_cli_dry_run_validates_checkpoint_without_payload_network_or_output(saved_collection_checkpoint,
                                                                          tmp_path, monkeypatch, capsys):
    from scripts import check_spinr_quadrature as cli
    path, arrays, contract = saved_collection_checkpoint
    def load(*args, **kwargs):
        assert kwargs["response_roles"] == ("train",)
        return arrays, contract
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run accessed payload or network")
    monkeypatch.setattr(cli, "load_object", load)
    monkeypatch.setattr(cli, "iter_npz_response_views", forbidden)
    monkeypatch.setattr(cli, "SpinrStyleINR", forbidden)
    output = tmp_path/"unwritten.json"
    assert cli.main(["--object", "b787", "--checkpoint", str(path), "--output", str(output), "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["response_reads"] == 0 and report["optimizer_steps"] == 0
    assert report["source_ids"] == contract["role_ids"]["train"][:4]
    assert report["checkpoint_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["rules"][2]["integration_points"] == 384**3
    assert not output.exists()


def test_cli_wrong_object_fails_before_response_reads(saved_collection_checkpoint, monkeypatch):
    from scripts import check_spinr_quadrature as cli
    from tests.test_rift_dataset import contract
    path, arrays, _ = saved_collection_checkpoint
    monkeypatch.setattr(cli, "load_object", lambda *a, **k: (arrays, contract("loader")))
    def forbidden(*args, **kwargs):
        raise AssertionError("wrong-object response read")
    monkeypatch.setattr(cli, "iter_npz_response_views", forbidden)
    with pytest.raises(ValueError, match="identity|object"):
        cli.main(["--object", "loader", "--checkpoint", str(path), "--dry-run"])


@pytest.mark.parametrize("passes", [True, False])
def test_cli_report_and_failure_exit_use_only_declared_train_views(saved_collection_checkpoint,
                                                                  tmp_path, monkeypatch, passes):
    from scripts import check_spinr_quadrature as cli
    path, arrays, contract = saved_collection_checkpoint
    ids = contract["role_ids"]["train"][:2]
    monkeypatch.setattr(cli, "load_object", lambda *a, **k: (arrays, contract))
    def restrict(a, requested):
        assert a is arrays and requested == ids
        return "restricted"
    def read(a, requested):
        assert a == "restricted" and requested == ids
        for source_id in requested:
            yield source_id, np.ones((16, 16, 1, 600), dtype=np.complex64)
    # Numerical rendering/gradients are tested independently above; this checks
    # the full CLI's identity, response scope, diagnostics and JSON exit contract.
    def observe(**kwargs):
        assert kwargs["scene_bins"]
        assert len(kwargs["frequencies_hz"]) == 600
        assert [o[0] for o in kwargs["observations"]] == ids
        assert all(o[1].shape == (600, 16, 16) for o in kwargs["observations"])
        result = observation_record()
        result["source_ids"], result["rule"] = ids, kwargs["rule"]
        if not passes and result["rule"]["label"] == "saved_training_rule":
            result["parameter_gradients"]["small"] *= -1
        return result
    monkeypatch.setattr(cli, "restrict_npz_response_views", restrict)
    monkeypatch.setattr(cli, "iter_npz_response_views", read)
    monkeypatch.setattr(cli, "observe_rule", observe)
    output = tmp_path/"report.json"
    status = cli.main(["--object", "b787", "--checkpoint", str(path), "--output", str(output),
                       "--views", "2", "--device", "cpu"])
    assert status == (0 if passes else 2)
    report = json.loads(output.read_text())
    assert report["checks_pass"] == passes and report["response_reads"] == 2
    assert report["source_ids"] == ids and report["role"] == "train"
    assert len(report["comparisons"]) == 3
    with pytest.raises(FileExistsError):
        cli.main(["--object", "b787", "--checkpoint", str(path), "--output", str(output)])
