"""Paper direct-bin rendering and fixed-budget contracts, on synthetic data."""
import copy

import pytest
import torch

import train_spinr_style as trainer
from rift.config import cc
from rift.forward_operator import get_kvector
from rift.spinr_direct import render_selected_bins, selected_bin_field_vjp, selected_bin_objective
from rift.spinr_fidelity import scene_range_bin_mask, DIRECT_RECIPE, BUDGET48_RECIPE, BUDGET150_RECIPE, recipe_name_from_identity
from rift.spinr_style import spinr_style_objective
from tests.test_spinr_fidelity import physics, dense_response, SmoothField, checkpoint_fixture


@pytest.mark.parametrize("sign", [-1., 1.])
@pytest.mark.parametrize("carrier", [0., 8.5e9])
@pytest.mark.parametrize("samples", [24, 600])
def test_direct_bins_and_adjoint_match_independent_dense_dft(sign, carrier, samples):
    f, rx, tx = physics()
    f = torch.arange(samples, dtype=torch.float64)*5e6+carrier
    points = torch.tensor([[.023, -.05, .08], [-.06, .03, -.01], [.01, .03, .06]],
                          dtype=torch.float64, requires_grad=True)
    field = torch.tensor([1.2, -.7, .3], dtype=torch.float64, requires_grad=True)
    volume = torch.tensor([.0002, .0003, .0001], dtype=torch.float64)
    mask = scene_range_bin_mask(f, rx, tx, phase_sign=sign)
    kwargs = dict(frequencies_hz=f, rx_pos_m=rx, tx_pos_m=tx, points_m=points,
                  cell_volume_m3=volume, initial_output_scale=.8, mask=mask, phase_sign=sign,
                  point_tile=2, pair_tile=2)
    direct = render_selected_bins(field=field, **kwargs)
    dense = dense_response(f, rx, tx, points, field, volume, scale=.8, sign=sign)
    expected = torch.where(mask, torch.fft.fft(dense, dim=0, norm="forward"), 0.)
    torch.testing.assert_close(direct, expected, rtol=1e-10, atol=1e-17)
    observed = dense_response(f, rx, tx, points.detach()+.0003, field.detach()*.7, volume, sign=sign)
    power = float(observed.abs().square().mean())
    loss = selected_bin_objective(direct, observed, mask, training_mean_raw_power=power)
    expected_loss = spinr_style_objective(dense, observed, training_mean_raw_power=power, range_bin_mask=mask)
    a = torch.autograd.grad(loss, (points, field), retain_graph=True)
    b = torch.autograd.grad(expected_loss, (points, field))
    for actual, reference in zip(a, b):
        torch.testing.assert_close(actual, reference, rtol=1e-9, atol=1e-9)
    detached = direct.detach().requires_grad_()
    cotangent, = torch.autograd.grad(selected_bin_objective(
        detached, observed, mask, training_mean_raw_power=power), detached)
    manual = selected_bin_field_vjp(bin_cotangent=cotangent, **kwargs)
    torch.testing.assert_close(manual, b[1], rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("offset", [0., 1e-10])
def test_exact_bin_removable_singularity_and_alias_wrap(offset):
    f = torch.arange(16, dtype=torch.float64)*cc/(2*16)
    antennas = torch.tensor([[1., 0., 0.]], dtype=torch.float64)
    point = torch.tensor([[offset, 0., 0.]], dtype=torch.float64, requires_grad=True)
    field = torch.ones(1, dtype=torch.float64)
    mask = torch.ones((16, 1, 1), dtype=torch.bool)
    bins = render_selected_bins(frequencies_hz=f, rx_pos_m=antennas, tx_pos_m=antennas,
        points_m=point, field=field, cell_volume_m3=1., initial_output_scale=1., mask=mask)
    expected = torch.fft.fft(dense_response(f, antennas, antennas, point, field, 1.), dim=0, norm="forward")
    torch.testing.assert_close(bins, expected, rtol=1e-10, atol=1e-14)
    g1, = torch.autograd.grad(bins.real.sum(), point, retain_graph=True)
    g2, = torch.autograd.grad(expected.real.sum(), point)
    torch.testing.assert_close(g1, g2, rtol=1e-10, atol=1e-12)
    assert int(bins.abs().reshape(-1).argmax()) == 15


def test_direct_logical_update_never_uses_predicted_frequency_renderer(monkeypatch):
    f, rx, tx = physics()
    points = torch.tensor([[.023, -.05, .08], [-.06, .03, -.01]], dtype=torch.float64)
    volumes = torch.tensor([.0002, .0003], dtype=torch.float64)
    model = SmoothField()
    reference = copy.deepcopy(model)
    observed = dense_response(f, rx, tx, points+.0002, model(points).detach()*.7, volumes)
    power = float(observed.abs().square().mean())
    entries = [(observed, rx+i*.0001, tx-i*.0001) for i in range(4)]
    expected_loss = 0.
    for y, r, t in entries:
        prediction = dense_response(f, r, t, points, reference(points), volumes)
        expected_loss = expected_loss + spinr_style_objective(prediction, y,
            training_mean_raw_power=power, range_bin_mask=scene_range_bin_mask(f, r, t))/4
    expected_loss.backward()
    expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), max_norm=1.)
    class Views:
        def tensor_view(self, i, *, device):
            return entries[i]
        def role_ids(self, role):
            return tuple(range(4))
    def forbidden(*a, **k):
        raise AssertionError("direct training invoked the frequency renderer/adjoint")
    frequency_renderer = trainer.range_forward_operator
    monkeypatch.setattr(trainer, "range_forward_operator", forbidden)
    monkeypatch.setattr(trainer, "range_adjoint_operator", forbidden)
    loss, norm = trainer.logical_batch_update(
        model=model, optimizer=torch.optim.Adam(model.parameters(), lr=0.), source_ids=range(4),
        views=Views(), points_m=points, cell_volume_m3=volumes, initial_output_scale=1.,
        frequencies_hz=f, kvector=get_kvector(f, cc), training_mean_raw_power=power,
        neural_point_tile=1, renderer_point_tile=1, pair_tile=2, device=torch.device("cpu"),
        scene_bins=True, direct_bins=True)
    assert loss == pytest.approx(float(expected_loss.detach()), rel=1e-9)
    assert norm == pytest.approx(float(expected_norm), rel=1e-9)
    torch.testing.assert_close(model.coefficients.grad, reference.coefficients.grad, rtol=1e-9, atol=1e-10)
    monkeypatch.setattr(trainer, "range_forward_operator", frequency_renderer)
    metrics = trainer.evaluate_role(
        model=model, views=Views(), role="validation", source_ids=None,
        points_m=points, cell_volume_m3=volumes, initial_output_scale=1.,
        frequencies_hz=f, kvector=get_kvector(f, cc), training_mean_raw_power=power,
        neural_point_tile=1, renderer_point_tile=2, pair_tile=2, device=torch.device("cpu"),
        scene_bins=True, direct_bins=True)
    assert metrics["native_spectral_objective"] == pytest.approx(loss, rel=1e-9)
    assert metrics["views"] == 4 and metrics["coherent_relative_mse"] > 0


def test_direct_recipe_defaults_and_checkpoint_gates(tmp_path):
    from train_rift_dataset import commands_for
    command = commands_for("a320", "spinr", dataset_root=tmp_path, output_root=tmp_path)[0]
    args = trainer.parse_args(command[2:])
    assert args.recipe == "budget48-direct" and args.epochs == 150 and args.grid_size == 48
    assert args.checkpoint_name == "budget48-direct-150"
    assert trainer._epoch_budget_stop_reason(149, args.recipe) == "development_epoch_budget_reached"
    assert trainer._epoch_budget_stop_reason(150, args.recipe) == "epoch_budget_reached"
    state, _, _ = checkpoint_fixture(args.recipe)
    identity = trainer._recipe_identity(args.recipe)
    assert identity["recipe_id"] == BUDGET150_RECIPE
    assert identity["optimizer"]["cosine_max_epochs"] == 150
    assert trainer._expected_cosine_learning_rate(150, 150) == pytest.approx(1e-5)
    assert identity["operator"]["quadrature"] == "midpoint"
    assert identity["operator"]["integration_points"] == 48**3
    from rift.spinr_quadrature_audit import quadrature_plan
    assert quadrature_plan(identity)[0]["integration_points"] == 48**3
    assert identity["monitoring"]["plateau"] == "disabled_fixed_comparison_epoch_budget"
    assert identity["fidelity"]["no_outcome_driven_model_changes"]
    trainer._validate_resume_checkpoint_structure(state, recipe_identity=identity)
    with pytest.raises(ValueError, match="recipe"):
        trainer._validate_resume_checkpoint_structure(state, recipe_identity=trainer._recipe_identity("paper-v1"))
    with pytest.raises(ValueError, match="recipe"):
        trainer._validate_resume_checkpoint_structure(state, recipe_identity=trainer._recipe_identity("paper-v1-direct"))
    old_command = commands_for("a320", "spinr", dataset_root=tmp_path, output_root=tmp_path,
                               spinr_recipe="budget48-direct-1500")[0]
    old_args = trainer.parse_args(old_command[2:])
    assert old_args.epochs == 1500 and old_args.checkpoint_name == "budget48-direct"
    old_state, _, _ = checkpoint_fixture("budget48-direct-1500")
    old_identity = trainer._recipe_identity("budget48-direct-1500")
    assert old_identity["recipe_id"] == BUDGET48_RECIPE
    assert recipe_name_from_identity(old_identity) == "budget48-direct-1500"
    trainer._validate_resume_checkpoint_structure(old_state, recipe_identity=old_identity)
    with pytest.raises(ValueError, match="recipe"):
        trainer._validate_resume_checkpoint_structure(old_state, recipe_identity=identity)
    state["scheduler_state_dict"]["T_max"] = 300
    with pytest.raises(ValueError, match="cosine"):
        trainer._validate_resume_checkpoint_structure(state, recipe_identity=identity)


@pytest.mark.parametrize("recipe", ["paper-v1-direct", "budget48-direct", "budget48-direct-1500"])
def test_direct_recipe_readout_gate_and_diagnostic_use_same_recipe(recipe):
    from scripts.readout_spinr import validate_readout_checkpoint
    from rift.spinr_quadrature_audit import quadrature_plan
    from rift.spinr_style import build_spinr_style_acquisition_identity
    from tests.test_rift_dataset import contract as collection_contract
    state, arrays, _ = checkpoint_fixture(recipe)
    contract = collection_contract("b787")
    state["sealed_npz_protocol_contract"] = contract
    state["acquisition_identity"] = build_spinr_style_acquisition_identity(arrays, contract)
    state["normalization"]["initial_scale_training_ids"] = contract["role_ids"]["train"][:32]
    state["optimization_coverage"] = trainer.optimization_coverage(contract["role_ids"]["train"], 0, 0)
    assert validate_readout_checkpoint(state, arrays, contract) == recipe
    rules = quadrature_plan(state["spinr_style_recipe"])
    expected = (2, 96) if recipe == "paper-v1-direct" else (1, 48)
    assert (rules[0]["nodes_per_cell"], rules[0]["parent_grid"]) == expected
