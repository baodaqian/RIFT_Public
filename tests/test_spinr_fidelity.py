"""Independent physics/gradient, spectral-support and recipe regression checks.

Only bounded synthetic tensors are used; no production responses or fitting.
"""
import copy
import math
from unittest.mock import patch

import numpy as np
import pytest
import torch

import train_spinr_style as trainer
from rift.config import cc
from rift.forward_operator import get_kvector
from rift.range_operator import range_forward_operator
from rift.spinr_fidelity import (PAPER_RECIPE, scene_range_bin_mask,
                                  spectral_partition_terms, field_readout)
from rift.spinr_style import (SpinrStyleINR, midpoint_grid, gauss_legendre_cell_grid,
                             spinr_style_objective, scale_field_to_renderer_weights)


def physics():
    f = torch.arange(24, dtype=torch.float64)*5e6 + 8.5e9
    rx = torch.tensor([[1.8, .1, .2], [1.9, -.1, -.1]], dtype=torch.float64)
    tx = torch.tensor([[1.75, .04, .2], [1.83, -.2, .06], [1.91, .2, .1]], dtype=torch.float64)
    return f, rx, tx


def dense_response(f, rx, tx, points, sigma, volumes, scale=1.0, sign=-1):
    rt = torch.linalg.vector_norm(points[:, None]-tx[None], dim=-1)
    rr = torch.linalg.vector_norm(points[:, None]-rx[None], dim=-1)
    distance = rr[:, :, None] + rt[:, None, :]
    kernel = torch.exp(sign*2j*math.pi*f[:, None, None, None]*distance[None]/cc)
    amplitude = (scale*sigma*volumes)[:, None, None]/(rr[:, :, None]*rt[:, None, :])
    return (kernel*amplitude[None]).sum(dim=1)


@pytest.mark.parametrize("sign", [-1., 1.])
@pytest.mark.parametrize("standoff", [0.1, 10., 31.])
def test_scene_bins_contain_bracketing_bins_and_alias_wrap(sign, standoff):
    f = torch.arange(600, dtype=torch.float64)*5e6 + 8.5e9
    rx = torch.tensor([[standoff, .01, .02], [standoff+.02, -.1, .1]], dtype=torch.float64)
    tx = rx[:1] + .01
    mask = scene_range_bin_mask(f, rx, tx, phase_sign=sign)
    rng = torch.Generator().manual_seed(21)
    points = .3*torch.rand((103, 3), generator=rng, dtype=torch.float64)-.15
    rt = torch.linalg.vector_norm(points[:, None]-tx[None], dim=-1)
    rr = torch.linalg.vector_norm(points[:, None]-rx[None], dim=-1)
    centers = sign*600*5e6*(rr[:, :, None]+rt[:, None, :])/cc
    for bins in (centers.floor().long() % 600, centers.ceil().long() % 600):
        assert mask.gather(0, bins).all()
    assert mask.shape == (600, 2, 1)
    assert not mask.all()


def test_scene_bin_mask_rejects_nonuniform_frequencies():
    f, rx, tx = physics()
    f[5] += .01
    with pytest.raises(ValueError, match="uniform"):
        scene_range_bin_mask(f, rx, tx)


def test_selected_objective_partitions_and_response_gradients():
    torch.manual_seed(8)
    f, rx, tx = physics()
    mask = scene_range_bin_mask(f, rx, tx)
    p = torch.randn(24, 2, 3, dtype=torch.complex128, requires_grad=True)
    y = torch.randn_like(p)
    power = float(y.abs().square().mean())
    losses = [spinr_style_objective(p, y, training_mean_raw_power=power, range_bin_mask=m)
              for m in (mask, ~mask)]
    full = spinr_style_objective(p, y, training_mean_raw_power=power)
    torch.testing.assert_close(sum(losses), full, rtol=1e-14, atol=1e-14)
    terms = spectral_partition_terms(p.detach(), y, mask, training_mean_raw_power=power)
    for label, loss in zip(("scene", "remainder"), losses):
        g, = torch.autograd.grad(loss, p, retain_graph=True)
        assert terms[label+"_objective"] == pytest.approx(float(loss.detach()), rel=1e-14)
        assert terms[label+"_response_gradient_norm_squared"] == pytest.approx(float(g.abs().square().sum()), rel=1e-13)
    # A residual supported solely outside the scene has zero selected loss,
    # but the full validation score continues to expose it.
    spectrum = torch.zeros_like(p)
    spectrum[~mask] = 1+2j
    outside = torch.fft.ifft(spectrum, dim=0, norm="forward")
    assert float(spinr_style_objective(outside, torch.zeros_like(p), training_mean_raw_power=1.,
                                     range_bin_mask=mask)) < 1e-28
    assert float(spinr_style_objective(outside, torch.zeros_like(p), training_mean_raw_power=1.)) > 1


@pytest.mark.parametrize("carrier", [0., 8.5e9])
def test_production_renderer_and_masked_vjp_against_dense_sum(carrier):
    f, rx, tx = physics()
    f = f-f[0]+carrier
    torch.manual_seed(92)
    points = .2*torch.rand(5, 3, dtype=torch.float64)-.1
    sigma = torch.randn(5, dtype=torch.float64, requires_grad=True)
    volumes = torch.tensor([.1, .3, .7, .2, .4], dtype=torch.float64)*1e-4
    weights = scale_field_to_renderer_weights(sigma, cell_volume_m3=volumes, initial_output_scale=.8)
    k = get_kvector(f, cc)
    predicted = range_forward_operator(f, k, rx, tx, points, weights, phase_sign=-1,
                                      range_model="product", compute_dtype=torch.float64,
                                      point_chunk=2, pair_chunk=2)
    direct = dense_response(f, rx, tx, points, sigma, volumes, scale=.8)
    torch.testing.assert_close(predicted, direct, rtol=2e-7, atol=1e-13)
    target = dense_response(f, rx, tx, points+.0002, sigma.detach()*.7, volumes, scale=.8)
    power = float(target.abs().square().mean())
    mask = scene_range_bin_mask(f, rx, tx)
    loss = spinr_style_objective(direct, target, training_mean_raw_power=power, range_bin_mask=mask)
    direct_gradient, = torch.autograd.grad(loss, sigma)
    _, g = trainer.response_cotangent(predicted, target, training_mean_raw_power=power, range_bin_mask=mask)
    manual = trainer.real_field_cotangent_from_response(
        response_cotangent_frequency=g, frequencies_hz=f, kvector=k,
        rx_pos_m=rx, tx_pos_m=tx, points_m=points, cell_volume_m3=volumes,
        initial_output_scale=.8, renderer_point_tile=2, pair_tile=2)
    torch.testing.assert_close(manual, direct_gradient, rtol=3e-6, atol=1e-9)
    direction = torch.arange(1., 6., dtype=torch.float64)
    eps = 1e-5
    def objective(offset):
        response = dense_response(f, rx, tx, points, sigma.detach()+offset*direction, volumes, scale=.8)
        return spinr_style_objective(response, target, training_mean_raw_power=power, range_bin_mask=mask)
    derivative = (objective(eps)-objective(-eps))/(2*eps)
    torch.testing.assert_close(manual @ direction, derivative, rtol=3e-6, atol=1e-9)


def test_closed_form_dft_matches_exact_frequency_sum_and_retains_carrier():
    f, rx, tx = physics()
    points = torch.tensor([[.023, -.05, .08], [-.06, .03, -.01]], dtype=torch.float64,
                          requires_grad=True)
    sigma = torch.tensor([1.2, -.7], dtype=torch.float64)
    volumes = torch.tensor([.0002, .0003], dtype=torch.float64)
    raw = dense_response(f, rx, tx, points, sigma, volumes)
    rt = torch.linalg.vector_norm(points[:, None]-tx[None], dim=-1)
    rr = torch.linalg.vector_norm(points[:, None]-rx[None], dim=-1)
    distance = rr[:, :, None]+rt[:, None, :]
    n = len(f)
    alpha = -2*math.pi*(f[1]-f[0])*distance/cc
    beta = 2*math.pi*torch.arange(n, dtype=torch.float64)/n
    delta = alpha[None]-beta[:, None, None, None]
    # Wrap before sinc evaluation: removable 0/0 at bin centers has limit one.
    delta = torch.remainder(delta+math.pi, 2*math.pi)-math.pi
    kernel = (torch.exp(.5j*(n-1)*delta)*torch.sinc(n*delta/(2*math.pi))
              /torch.sinc(delta/(2*math.pi)))
    carrier = torch.exp(-2j*math.pi*f[0]*distance/cc)
    amplitude = (sigma*volumes)[:, None, None]/(rr[:, :, None]*rt[:, None, :])
    direct_bins = (kernel*carrier[None]*amplitude[None]).sum(dim=1)
    transformed = torch.fft.fft(raw, dim=0, norm="forward")
    torch.testing.assert_close(direct_bins, transformed, rtol=1e-10, atol=1e-16)
    g1, = torch.autograd.grad(direct_bins.abs().square().sum(), points, retain_graph=True)
    g2, = torch.autograd.grad(transformed.abs().square().sum(), points)
    torch.testing.assert_close(g1, g2, rtol=1e-9, atol=1e-16)
    without_carrier = (kernel*amplitude[None]).sum(dim=1)
    assert torch.linalg.vector_norm(without_carrier-transformed)/torch.linalg.vector_norm(transformed) > .1


def test_masked_four_view_tiled_update_and_validation_match_autograd():
    torch.manual_seed(31)
    f, rx, tx = physics()
    points = torch.rand(7, 3, dtype=torch.float64)*.2-.1
    volumes = torch.linspace(.0001, .0004, 7, dtype=torch.float64)
    entries = [(torch.randn(24, 2, 3, dtype=torch.complex128), rx+i*.001, tx-i*.001)
               for i in range(4)]

    class Views:
        def role_ids(self, role):
            assert role in ("train", "validation")
            return tuple(range(4))

        def tensor_view(self, i, *, device):
            return entries[i]

    model = SmoothField()
    reference = copy.deepcopy(model)
    loss = 0.
    for observed, r, t in entries:
        prediction = dense_response(f, r, t, points, reference(points), volumes)
        loss = loss + spinr_style_objective(
            prediction, observed, training_mean_raw_power=1.,
            range_bin_mask=scene_range_bin_mask(f, r, t))/4
    loss.backward()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.)
    kwargs = dict(model=model, views=Views(), points_m=points, cell_volume_m3=volumes,
                  initial_output_scale=1., frequencies_hz=f, kvector=get_kvector(f, cc),
                  training_mean_raw_power=1., neural_point_tile=3, renderer_point_tile=3,
                  pair_tile=2, device=torch.device("cpu"), scene_bins=True)
    actual, _ = trainer.logical_batch_update(optimizer=optimizer, source_ids=range(4), **kwargs)
    assert actual == pytest.approx(float(loss.detach()), rel=1e-9)
    torch.testing.assert_close(model.coefficients.grad, reference.coefficients.grad, rtol=1e-6, atol=1e-12)
    metrics = trainer.evaluate_role(role="validation", source_ids=None, **kwargs)
    assert metrics["native_spectral_objective"] == pytest.approx(actual, rel=1e-10)
    assert metrics["full_spectral_objective"] == pytest.approx(metrics["scene_objective"]+metrics["remainder_objective"])
    assert metrics["remainder_squared_error"] > 0
    assert 0 < metrics["scene_target_energy_fraction"] < 1


def test_midpoint_feature_null_is_removed_and_cell_integral_improves():
    # Analytic null from the audit, evaluated at actual one-dimensional rule nodes.
    g = 48
    mid = -1+(2*np.arange(g)+1)/g
    nodes, weights = np.polynomial.legendre.leggauss(2)
    gl = (mid[:, None]+nodes[None]/g).reshape(-1)
    null = lambda x: np.sin(32*np.pi*x)-np.sin(16*np.pi*x)
    assert np.max(np.abs(null(mid))) < 5e-14
    assert np.sqrt(np.mean(null(gl)**2)) > .5
    q = 4*np.pi*11.495e9/cc
    errors = []
    for parent, order in ((48, 1), (48, 2), (96, 2), (96, 3)):
        n, w = np.polynomial.legendre.leggauss(order)
        half_cell = .15/parent
        exact = np.sinc(q*half_cell/np.pi)
        estimate = np.sum(w*np.exp(1j*q*half_cell*n))/2
        errors.append(abs(estimate-exact)/abs(exact))
    assert errors[0] > .5
    assert errors[0] > errors[1] > errors[2] > errors[3]
    assert errors[2] < .002


class SmoothField(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.coefficients = torch.nn.Parameter(torch.tensor([.8, -.3, .4, .2], dtype=torch.float64))

    def forward(self, p):
        return self.coefficients[0] + torch.tanh(p @ self.coefficients[1:]*15)


def test_frozen_field_signal_and_parameter_gradient_quadrature_refinement():
    # Smooth frozen neural field over a small support: actual production operator,
    # progressively refined independent GL rules; no optimizer is used.
    f, rx, tx = physics()
    model = SmoothField()
    responses, derivatives = [], []
    probe = torch.linspace(.2, 1., 24, dtype=torch.float64)[:, None, None]*(1+.4j)
    for grid, order in ((2, 2), (4, 3), (8, 5)):
        points, volume = gauss_legendre_cell_grid(grid, nodes_per_cell=order, support_m=.008)
        sigma = model(points)
        w = scale_field_to_renderer_weights(sigma, cell_volume_m3=volume, initial_output_scale=1.)
        response = range_forward_operator(f, get_kvector(f, cc), rx, tx, points, w,
                                          phase_sign=-1, range_model="product", compute_dtype=torch.float64,
                                          point_chunk=1024, pair_chunk=6)
        derivative, = torch.autograd.grad((response*probe.conj()).real.sum(), model.coefficients)
        responses.append(response.detach())
        derivatives.append(derivative)
    for values in (responses, derivatives):
        coarse = torch.linalg.vector_norm(values[0]-values[2])/torch.linalg.vector_norm(values[2])
        fine = torch.linalg.vector_norm(values[1]-values[2])/torch.linalg.vector_norm(values[2])
        assert fine < coarse
        assert fine < 1e-4


def test_full_view_coverage_and_pending_boundary():
    ids = np.random.default_rng(42).permutation(10000)[:3200].tolist()
    batches = trainer.epoch_view_batches(ids, epoch=3)
    assert sorted(i for b in batches for i in b) == sorted(ids)
    record = trainer.optimization_coverage(ids, 3, 17)
    visited = {i for b in batches[:17] for i in b}
    assert record["exposure_counts"] == [3+int(i in visited) for i in ids]
    assert record["view_exposures"] == 3*3200+17*4
    assert trainer.optimization_coverage(ids, 3, 800) == trainer.optimization_coverage(ids, 4, 0)


def checkpoint_fixture(recipe="paper-v1"):
    from scripts.validate_spinr_style import _canonical_contract_and_header_arrays
    contract, arrays, acquisition = _canonical_contract_and_header_arrays()
    model = SpinrStyleINR()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    identity = trainer._recipe_identity(recipe)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=identity["optimizer"]["cosine_max_epochs"], eta_min=1e-5)
    state = trainer._checkpoint_state(
        model=model, optimizer=optimizer, scheduler=scheduler, sealed_contract=contract,
        acquisition_identity=acquisition, recipe_identity=identity,
        operational_settings={}, training_mean_raw_power=1., initial_output_scale=.1,
        initial_scale_ids=contract["role_ids"]["train"][:32], initial_scale_observed_energy=1.,
        initial_scale_predicted_energy=1., epoch_index=0,
        execution=trainer._execution_state(phase="updates", completed_updates=0, loss_sum=0.,
                                          gradient_norm_sum=0., elapsed_seconds=0.),
        best_validation_rel_mse=math.inf, best_epoch=None, history=[])
    return state, arrays, contract


def test_recipe_and_coverage_rejected_before_payload(tmp_path):
    state, _, _ = checkpoint_fixture()
    recipe = trainer._recipe_identity("paper-v1")
    trainer._validate_resume_checkpoint_structure(state, recipe_identity=recipe)
    with pytest.raises(ValueError, match="recipe"):
        trainer._validate_resume_checkpoint_structure(state, recipe_identity=trainer._recipe_identity())
    state["optimization_coverage"]["exposure_counts"][0] = 1
    path = tmp_path/"checkpoint_latest.pth.tar"
    torch.save(state, path)
    args = trainer.parse_args(["--checkpoint-name", "run", "--host-rss-limit-gib", "4",
                               "--recipe", "paper-v1", "--device", "cpu", "--allow-cpu-validation",
                               "--resume", str(path)])
    with patch.object(trainer, "preflight_b787_development_inputs", side_effect=AssertionError("metadata accessed")):
        with pytest.raises(ValueError, match="coverage"):
            trainer.run(args)


def test_collection_routing_explicitly_preserves_legacy(tmp_path):
    from train_rift_dataset import commands_for
    for recipe, grid in (("paper-v1", 96), ("legacy-midpoint", 48)):
        command = commands_for("a320", "spinr", dataset_root=tmp_path, output_root=tmp_path,
                               spinr_recipe=recipe)[0]
        args = trainer.parse_args(command[2:])
        assert args.recipe == recipe and args.grid_size == grid
    assert trainer._recipe_identity()["operator"]["frequency_transform"] == "fft_norm_forward_all_600"
    assert trainer._recipe_identity("paper-v1")["recipe_id"] == PAPER_RECIPE


def test_geometry_readout_retains_signed_values_scale_and_tile_invariance():
    model = SmoothField()
    a, sigma = field_readout(model, grid_size=6, support_m=.15, initial_output_scale=2.,
                             neural_point_tile=7, device="cpu")
    b, other = field_readout(model, grid_size=6, support_m=.15, initial_output_scale=2.,
                             neural_point_tile=31, device="cpu")
    assert model.training
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(sigma, other)
    torch.testing.assert_close(sigma, 2*model(a))
    # Changing integration weights has no role in the continuous-field readout.
    assert sigma.shape == (216,)


def test_readout_rejects_wrong_object_before_network_or_acquisition():
    from scripts.readout_spinr import validate_readout_checkpoint
    from tests.test_rift_dataset import contract
    state = {"sealed_npz_protocol_contract": contract("loader")}
    with patch("scripts.readout_spinr.build_spinr_style_acquisition_identity",
               side_effect=AssertionError("acquisition accessed")):
        with pytest.raises(ValueError, match="identity|object"):
            validate_readout_checkpoint(state, {}, contract("b787"))


def test_collection_readout_cli_exports_matching_checkpoint_without_responses(tmp_path):
    import hashlib
    import json
    from scripts import readout_spinr
    from tests.test_rift_dataset import contract
    from rift.spinr_style import build_spinr_style_acquisition_identity
    state, arrays, _ = checkpoint_fixture()
    expected_contract = contract("b787")
    state["sealed_npz_protocol_contract"] = expected_contract
    state["acquisition_identity"] = build_spinr_style_acquisition_identity(arrays, expected_contract)
    state["normalization"]["initial_scale_training_ids"] = expected_contract["role_ids"]["train"][:32]
    state["optimization_coverage"] = trainer.optimization_coverage(expected_contract["role_ids"]["train"], 0, 0)
    checkpoint, output = tmp_path/"model.pt", tmp_path/"support.npz"
    torch.save(state, checkpoint)
    with patch.object(readout_spinr, "load_object", return_value=(arrays, expected_contract)) as load:
        readout_spinr.main(["--checkpoint", str(checkpoint), "--object", "b787",
                           "--output", str(output), "--grid-size", "3", "--fixed-threshold", ".2"])
        assert load.call_count == 1
    with np.load(output) as result:
        info = json.loads(str(result["metadata_json"]))
        assert info["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert info["radar_responses_read"] is False
        assert info["dataset_identity"] == expected_contract["dataset_identity"]
        np.testing.assert_array_equal(result["magnitude"], np.abs(result["sigma"]))
        assert result["sigma"].shape == (3, 3, 3)
        assert result["support_points_m"].shape[0] == np.count_nonzero(result["normalized_magnitude"] >= .2)


@pytest.mark.parametrize("recipe,epochs", [("paper-v1", 1), ("paper-v1-direct", 1),
    ("budget48-direct", 1), ("budget48-direct", 150), ("budget48-direct-1500", 1)])
def test_corrected_recipe_interrupted_resume_preserves_exposures_and_optimizer(tmp_path, monkeypatch, recipe, epochs):
    """Real trainer lifecycle with an 8-view/two-update synthetic execution harness."""
    from scripts.validate_spinr_style import _canonical_contract_and_header_arrays, _cpu_run_args
    contract, _, acquisition = _canonical_contract_and_header_arrays()
    contract["role_ids"]["train"] = list(range(8))
    contract["role_ids"]["validation"] = [8, 9]
    for key, value in dict(CANONICAL_TRAIN_COUNT=8, CANONICAL_VALIDATION_COUNT=2,
                           CANONICAL_UPDATES_PER_EPOCH=2, CANONICAL_INIT_SCALE_COUNT=2,
                           CANONICAL_TRAIN_DIAGNOSTIC_COUNT=2, CANONICAL_VALIDATION_EVERY=1).items():
        monkeypatch.setattr(trainer, key, value)

    class HarnessField(SmoothField):
        def trainable_parameter_count(self):
            return 4

    class Views:
        frequencies_hz = np.arange(600)*5e6+8.5e9
        materialized_response_bytes = 0

        def role_ids(self, role):
            return tuple(contract["role_ids"][role])

        def raw_training_mean_power(self):
            return 1.

    mode = {"interrupt": False, "stopper": None, "trace": []}

    class Stopper:
        def __init__(self):
            self.requested = False
            mode["stopper"] = self

        def __call__(self, *_):
            self.requested = True

    def update(*, model, optimizer, source_ids, scene_bins, **kwargs):
        assert scene_bins
        assert kwargs.get("direct_bins", False) == (recipe in ("paper-v1-direct", "budget48-direct", "budget48-direct-1500"))
        optimizer.zero_grad()
        loss = model.coefficients.square().sum()*(1+torch.rand(()))
        loss.backward()
        optimizer.step()
        mode["trace"].extend(source_ids)
        if mode["interrupt"]:
            mode["stopper"].requested = True
        return float(loss.detach()), .5

    def evaluate(*, role, source_ids, scene_bins, **kwargs):
        assert scene_bins
        assert kwargs.get("direct_bins", False) == (recipe in ("paper-v1-direct", "budget48-direct", "budget48-direct-1500"))
        return {"views": 2, "coherent_relative_mse": .25, "coherent_relative_l2": .5,
                "native_spectral_objective": .1}

    monkeypatch.setattr(trainer, "SpinrStyleINR", HarnessField)
    monkeypatch.setattr(trainer, "SPINR_STYLE_PARAMETER_COUNT", 4)
    monkeypatch.setattr(trainer, "_spinr_model_state_layout", lambda: {"coefficients": (4,)})
    monkeypatch.setattr(trainer, "preflight_b787_development_inputs", lambda **_: ({}, contract, acquisition))
    monkeypatch.setattr(trainer, "materialize_b787_development_views", lambda *_: Views())
    monkeypatch.setattr(trainer, "gauss_legendre_cell_grid", lambda *a, **k: (torch.zeros(1, 3), torch.ones(1)))
    monkeypatch.setattr(trainer, "estimate_initial_output_scale", lambda **_: (.1, (0, 1), 1., 1.))
    monkeypatch.setattr(trainer, "logical_batch_update", update)
    monkeypatch.setattr(trainer, "evaluate_role", evaluate)
    monkeypatch.setattr(trainer, "_enforce_memory_gates", lambda **_: {})
    monkeypatch.setattr(trainer, "_StopAfterCurrentUpdate", Stopper)
    if recipe in ("paper-v1-direct", "budget48-direct", "budget48-direct-1500"):
        def forbidden_plateau(*args):
            raise AssertionError("paper budget must not use the historical early stop")
        monkeypatch.setattr(trainer, "plateau_reached", forbidden_plateau)

    def run(name, resume=None):
        args = _cpu_run_args(checkpoint_root=tmp_path, checkpoint_name=name, resume=resume)
        args.recipe, args.grid_size = recipe, 48 if recipe in ("budget48-direct", "budget48-direct-1500") else 96
        args.epochs = epochs
        trainer.run(args)

    run("complete")
    expected = trainer.load_tensor_checkpoint(tmp_path/"complete"/"checkpoint_final.pth.tar", map_location="cpu")
    mode["trace"] = []
    mode["interrupt"] = True
    run("resumed")
    latest = tmp_path/"resumed"/"checkpoint_latest.pth.tar"
    partial = trainer.load_tensor_checkpoint(latest, map_location="cpu")
    assert partial["optimization_coverage"]["view_exposures"] == 4
    mode["interrupt"] = False
    run("resumed", str(latest))
    actual = trainer.load_tensor_checkpoint(tmp_path/"resumed"/"checkpoint_final.pth.tar", map_location="cpu")
    assert sorted(mode["trace"]) == sorted(list(range(8))*epochs)
    assert actual["epoch_index"] == epochs
    assert actual["execution"]["stop_reason"] == trainer._epoch_budget_stop_reason(epochs, recipe)
    assert actual["optimization_coverage"] == expected["optimization_coverage"]
    for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "selection", "rng_state"):
        assert trainer._checkpoint_tree_equal(actual[key], expected[key])
