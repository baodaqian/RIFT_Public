"""Independent numerical and recipe-boundary checks, with synthetic data only."""
import argparse
import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rift.geraf import GeRaFModel, GeRaFSDFNetwork, neus_sdf_to_alpha, sample_primary_rays
from rift.geraf_v1 import (sample_scene_rays, cell_sdf_to_alpha, boundary_start_points,
                           measured_ray_mask, MeasuredMaskReference, MeasuredViewMaskBank,
                           SourceSDFNetwork, extract_zero_surface)
from rift.config import cc
from rift.forward_operator import get_kvector
from rift.geraf_signal_operator import pairwise_range_forward_operator, matched_filter_from_response_range
from scripts.validate_geraf_signal_operator import direct_pairwise_forward, direct_phase_only_matched_filter
import train_geraf as trainer


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def tiny_model():
    torch.manual_seed(42)
    return GeRaFModel(.15, sdf_hidden_dim=96, sdf_layers=3, sdf_skip_layer=1,
                     reflectivity_hidden_dim=12, reflectivity_layers=2,
                     implementation="hardened_v1", sdf_initialization="upstream_geometric",
                     sdf_encoding_include_input=True, sdf_encoding_coordinate_scale=1/.15,
                     sdf_output_scale=.15)


def scene_samples(n=8):
    return sample_scene_rays(torch.tensor([0., 0., 10.], dtype=torch.float64),
                             torch.tensor([0., 0., -1.], dtype=torch.float64), extent=.15,
                             n_azimuth=2, n_elevation=2, n_depth=n)


@pytest.mark.parametrize("direction", [[0., 0., -1.], [1., 2., -3.]])
def test_full_scene_cells_cover_box_and_have_actual_endpoints(direction):
    p = torch.tensor(direction, dtype=torch.float64)
    p = p / p.norm()
    samples = sample_scene_rays(-10*p, p, extent=.15, n_azimuth=8, n_elevation=8, n_depth=16)
    assert samples.points.abs().max() <= .15 + 1e-12
    start = samples.ray_origins + samples.depth_edges[:, :1] * p
    end = samples.ray_origins + samples.depth_edges[:, -1:] * p
    torch.testing.assert_close(start.abs().amax(-1), torch.full((len(start),), .15, dtype=p.dtype))
    torch.testing.assert_close(end.abs().amax(-1), torch.full((len(end),), .15, dtype=p.dtype))
    assert samples.points[..., :2].abs().max() > .07  # Not the 2.85 cm physical aperture.
    torch.testing.assert_close(samples.depth_deltas.sum(-1), samples.depth_edges[:, -1] - samples.depth_edges[:, 0])


@pytest.mark.parametrize("stratified", [False, True])
def test_plane_opacity_matches_analytic_cell_cdf_including_last(stratified):
    samples = sample_primary_rays(torch.tensor([0., 0., .15], dtype=torch.float64),
                                 torch.tensor([0., 0., -1.], dtype=torch.float64),
                                 aperture_width=.1, aperture_height=.1, near=0, far=.3,
                                 num_rays=4, num_depth=8, stratified_depth=stratified,
                                 generator=torch.Generator().manual_seed(42))
    # Plane near the last cell would be discarded by the old repeated tail.
    plane = -.13
    sdf = samples.points[..., 2] - plane
    gradient = torch.zeros_like(samples.points)
    gradient[..., 2] = 1
    alpha = cell_sdf_to_alpha(sdf, gradient, samples.primary_direction,
                              samples.depths, samples.depth_edges, 100.)
    cdf = torch.sigmoid((.15 - samples.depth_edges - plane)*100)
    expected = 1 - cdf[:, 1:] / cdf[:, :-1]
    torch.testing.assert_close(alpha, expected, atol=1e-14, rtol=1e-12)
    assert alpha[:, -1].mean() > .5
    assert torch.count_nonzero(neus_sdf_to_alpha(sdf, 100.)[:, -1]) == 0


def test_constant_and_exiting_fields_have_exactly_zero_opacity():
    samples = scene_samples()
    for gradient in (torch.zeros_like(samples.points), samples.primary_direction.expand_as(samples.points)):
        alpha = cell_sdf_to_alpha(torch.ones(samples.depths.shape, dtype=torch.float64), gradient,
                                  samples.primary_direction, samples.depths, samples.depth_edges, 100.)
        assert torch.count_nonzero(alpha) == 0


def test_boundary_queries_are_on_local_scene_even_for_distant_antennas():
    points = scene_samples().points
    for antenna in [torch.tensor([0., 0., 10.]), torch.tensor([9., 1., 10.])]:
        start = boundary_start_points(antenna.double(), points, .15)
        torch.testing.assert_close(start.abs().amax(-1), torch.full(points.shape[:-1], .15, dtype=torch.float64))
        cross = torch.linalg.cross(start-antenna, points-antenna)
        assert cross.abs().max() < 1e-11


def test_float64_geometry_survives_float32_network_and_no_grad_render():
    model, samples = tiny_model(), scene_samples()
    antennas = torch.tensor([[0., 0., 10.]], dtype=torch.float64)
    queried = []
    hook = model.sdf_network.register_forward_pre_hook(lambda _m, x: queried.append(x[0].detach().clone()))
    with torch.no_grad():
        output = model.render_volume(samples, antennas, antennas, create_graph=False)
    hook.remove()
    assert output.points.dtype == torch.float64
    assert torch.equal(output.points, samples.points)
    assert output.sdf.dtype == torch.float32
    assert torch.isfinite(output.amplitudes).all()
    assert all(x.abs().max() <= .15 + 1e-7 for x in queried)
    assert output.diagnostics["render_samples"] == samples.points.numel()//3


def test_first_order_normals_do_not_free_sdf_graph():
    network = GeRaFSDFNetwork(.15, hidden_dim=16, n_layers=2, skip_layer=1)
    sdf, normals = network.gradient(torch.tensor([[.01, .02, .03]], dtype=torch.float64), create_graph=False)
    (sdf.square().sum() + normals.square().sum()).backward()
    assert network.output.weight.grad is not None
    assert torch.isfinite(network.output.weight.grad).all()


def test_geometric_initialization_has_metric_sign_structure():
    torch.manual_seed(42)
    network = SourceSDFNetwork(.15)
    points = torch.tensor([[0., 0., 0.], [.15, .15, .15], [-.15, -.15, -.15]])
    sdf, gradient = network.gradient(points, create_graph=False)
    assert sdf[0] < 0 and (sdf[1:] > 0).all()
    assert gradient[1:].norm(dim=-1).min() > .1
    assert sdf[1:].abs().min() > .01


def test_model_config_round_trip_and_legacy_construction():
    model = tiny_model()
    restored = GeRaFModel(**model.config())
    restored.load_state_dict(model.state_dict(), strict=True)
    x = torch.tensor([[.01, .03, -.02]])
    torch.testing.assert_close(model.sdf_network(x), restored.sdf_network(x), rtol=0, atol=0)
    legacy = GeRaFModel(.15, sdf_hidden_dim=16, sdf_layers=2, sdf_skip_layer=1,
                        reflectivity_hidden_dim=8, reflectivity_layers=1)
    assert legacy.implementation == "legacy"
    assert "implementation" not in legacy.config()
    with pytest.raises(RuntimeError):
        legacy.load_state_dict(model.state_dict())


def test_measured_mask_matches_reference_and_cannot_hide_prediction_collapse():
    measured = torch.tensor([[1., .2], [.001, .002], [0., 0.]])
    accumulated = torch.tensor([[1., .2], [.8, .7], [0., 0.]])
    valid = measured_ray_mask(measured, accumulated)
    assert valid.tolist() == [[True, True], [False, False], [True, True]]
    prediction = torch.zeros_like(measured, requires_grad=True)
    trainer.masked_magnitude_l2(prediction, measured, valid).backward()
    assert prediction.grad[0, 0] < 0
    with pytest.raises(RuntimeError, match="every ray"):
        measured_ray_mask(torch.zeros((2, 3)), torch.ones((2, 3)))


def test_measured_reference_aligns_world_coordinates_and_rejects_roles():
    ref = MeasuredMaskReference(1., 3, {"train_indices": [7, 11]})
    points = torch.tensor([[-1., 0., 0.], [1., 0., 0.]])
    ref.add(7, points, torch.tensor([2., 8.]))
    with pytest.raises(ValueError, match="exactly once"):
        ref.add(7, points, torch.ones(2))
    with pytest.raises(ValueError, match="training"):
        ref.add(99, points, torch.ones(2))
    with pytest.raises(ValueError, match="coverage"):
        ref.finalize()
    ref.add(11, points.flip(0), torch.tensor([8., 2.]))
    ref.finalize()
    torch.testing.assert_close(ref.sample(points), torch.tensor([2., 8.]))


def test_measured_mask_checkpoint_binds_object_and_values():
    kwargs = dict(train_indices=[7], shape=(1, 2), extent=1., size=3, identity={"object": "a320"})
    bank = MeasuredViewMaskBank(**kwargs)
    bank.reference.add(7, torch.tensor([[-1., 0., 0.], [1., 0., 0.]]), torch.ones(2))
    bank.reference.finalize()
    state = bank.state_dict()
    restored = MeasuredViewMaskBank(**kwargs)
    restored.load_state_dict(state)
    assert restored.reference.digest() == bank.reference.digest()
    with pytest.raises(ValueError, match="identity"):
        MeasuredViewMaskBank(**{**kwargs, "identity": {"object": "loader"}}).load_state_dict(state)
    changed = copy.deepcopy(state)
    changed["volume"][0] += 1
    with pytest.raises(ValueError, match="digest"):
        restored.load_state_dict(changed)


def test_sampler_covers_all_3200_and_resumes_exposure_ledger():
    sampler = trainer.DeterministicViewSampler(list(range(3200)), 42)
    observed = [sampler.next() for _ in range(3200)]
    for index in observed:
        sampler.record_update(index)
    assert set(observed) == set(range(3200))
    assert sampler.coverage()["minimum_exposures"] == 1
    restored = trainer.DeterministicViewSampler(list(range(3200)), 999)
    restored.load_state_dict(sampler.state_dict())
    assert restored.coverage() == sampler.coverage()
    assert [restored.next() for _ in range(25)] == [sampler.next() for _ in range(25)]


def test_hardened_renderer_through_actual_nufft_matches_dense_values_and_gradients():
    model, samples = tiny_model().double(), scene_samples(4)
    tx = torch.tensor([[0., 0., 10.], [.01, -.02, 9.99]], dtype=torch.float64)
    rx = torch.tensor([[.02, .01, 10.01]], dtype=torch.float64)
    f = 8.5e9 + torch.arange(9, dtype=torch.float64) * 5e6
    k = get_kvector(f, cc)
    query = torch.tensor([[.01, .02, .03], [-.03, .01, -.02]], dtype=torch.float64)
    from rift.geraf_signal_operator import bistatic_pair_positions
    outputs, derivatives = [], []
    for dense in (True, False):
        model.zero_grad(set_to_none=True)
        output = model.render_volume(samples, *bistatic_pair_positions(tx, rx), create_graph=True)
        points, amplitude = output.points.reshape(-1, 3), output.amplitudes.reshape(2, -1)
        if dense:
            response = direct_pairwise_forward(f, tx, rx, points, amplitude, phase_sign=-1.)
            mf = direct_phase_only_matched_filter(response, f, tx, rx, query, phase_sign=-1.)
        else:
            response = pairwise_range_forward_operator(f, k, tx, rx, points, amplitude, point_chunk=5, pair_chunk=1)
            mf = matched_filter_from_response_range(response, f, k, tx, rx, query, point_chunk=1, pair_chunk=1)
        loss = mf.abs().square().sum()
        gradient = torch.autograd.grad(loss, model.sdf_network.lin3.weight_v)[0]
        outputs.append(mf.detach())
        derivatives.append(gradient)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=2e-7, atol=1e-12)
    torch.testing.assert_close(derivatives[0], derivatives[1], rtol=2e-6, atol=1e-12)
    assert derivatives[1].norm() > 0


def test_released_sdf_values_normals_and_skip_architecture_are_preserved():
    from rift.vendor.geraf_sens.sdf_network import SDFNetwork
    torch.manual_seed(42)
    network = SourceSDFNetwork(.15).double()
    original = SDFNetwork(d_in=3, d_out=1, d_hidden=256, n_layers=8, skip_in=(4,),
                          multires=10, bias=.5, scale=1., geometric_init=True, weight_norm=True).double()
    original.load_state_dict(network.state_dict(), strict=True)
    xyz = torch.tensor([[0., 0., 0.], [.15, .15, .15], [-.15, -.15, -.15]], dtype=torch.float64)
    normalized = (xyz / .15).requires_grad_(True)
    expected = original(normalized).squeeze(-1)
    expected_gradient = torch.autograd.grad(expected.sum(), normalized)[0]
    actual, gradient = network.gradient(xyz, create_graph=True)
    torch.testing.assert_close(actual, expected * .15, rtol=0, atol=0)
    torch.testing.assert_close(gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    assert actual[0] < 0 and (actual[1:] > 0).all()
    assert network.lin3.out_features == 256 - 63
    actual.square().sum().backward()
    assert network.lin0.weight_v.grad.norm() > 0


def test_explicit_hardened_cli_has_distinct_recipe(monkeypatch):
    monkeypatch.setattr("sys.argv", ["train_geraf.py", "--implementation", "hardened_v1", "--cache-root", "/tmp/geraf-no-read",
                                    "--checkpoint-dir", "/tmp/geraf-no-write"])
    args = trainer.parse_args()
    assert args.implementation == "hardened_v1"
    config = trainer.model_config_from_args(args)
    model = GeRaFModel(**config)
    assert isinstance(model.sdf_network, SourceSDFNetwork)
    restored = GeRaFModel(**model.config())
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        output = model.render_volume(scene_samples(3), torch.tensor([[0., 0., 10.]]),
                                     torch.tensor([[0., 0., 10.]]), create_graph=False)
    assert torch.isfinite(output.amplitudes).all()
    from scripts.validate_geraf_b7873200_entrypoint import _synthetic_cache
    cache = _synthetic_cache()
    hardened = trainer.run_identity(args, cache, config)
    args.implementation = "legacy"
    legacy = trainer.run_identity(args, cache, trainer.model_config_from_args(args))
    assert hardened != legacy and "implementation" not in legacy


def test_fixed_plane_signal_and_surface_gradient_converge_under_depth_refinement():
    from rift.geraf import transmittance_from_alpha
    def evaluate(n):
        plane = torch.tensor(.013, dtype=torch.float64, requires_grad=True)
        edges = torch.linspace(-.15, .15, n+1, dtype=torch.float64)[None]
        depths = (edges[:, :-1] + edges[:, 1:]) * .5
        gradient = torch.tensor([0., 0., -1.], dtype=torch.float64).expand(1, n, 3)
        alpha = cell_sdf_to_alpha(plane-depths, gradient, torch.tensor([0., 0., 1.]),
                                  depths, edges, 100.)
        weights = transmittance_from_alpha(alpha).square() * alpha
        value = (weights * torch.exp(1j * 80 * depths)).sum()
        derivative = torch.autograd.grad(value.real, plane)[0]
        return value.detach(), derivative
    reference, reference_gradient = evaluate(16384)
    errors, gradient_errors = [], []
    for n in (32, 64, 128, 256):
        value, gradient = evaluate(n)
        errors.append(abs(value-reference))
        gradient_errors.append(abs(gradient-reference_gradient))
    assert all(b < a * .6 for a, b in zip(errors, errors[1:]))
    assert all(b < a * .6 for a, b in zip(gradient_errors, gradient_errors[1:]))


class AnalyticSphere(torch.nn.Module):
    def __init__(self, radius=.07):
        super().__init__()
        self.radius = torch.nn.Parameter(torch.tensor(radius))

    def forward(self, points):
        return points.norm(dim=-1) - self.radius


def test_native_zero_surface_has_metric_radius_and_refines_without_threshold_tuning():
    model = AnalyticSphere()
    errors = []
    for grid in (24, 48):
        vertices, faces, report = extract_zero_surface(model, .15, grid=grid, chunk=511)
        errors.append(np.abs(np.linalg.norm(vertices, axis=-1) - .07).max())
        assert faces.ndim == 2 and faces.shape[1] == 3
        assert report["isolevel_m"] == 0 and report["boundary_vertex_fraction"] == 0
        assert not report["components_filtered"] and not report["posthoc_alignment"]
    assert errors[-1] < errors[0] * .4
    assert errors[-1] < .0002


def test_native_zero_surface_rejects_no_crossing():
    with pytest.raises(ValueError, match="no resolved zero crossing"):
        extract_zero_surface(AnalyticSphere(radius=1.), .15, grid=12)


def test_render_sampling_retains_original_target_grid():
    targets = scene_samples(4)
    # Narrow the target footprint to the physical-array-sized frozen grid.
    targets.ray_origins[:, :2] *= .1
    targets.points[..., :2] *= .1
    original = targets.points.clone()
    view = SimpleNamespace(samples=targets)
    args = SimpleNamespace(implementation="hardened_v1", n_azimuth=2, n_elevation=2,
                           n_depth=4, scene_extent=.15)
    integration = trainer.render_samples_for_view(view, args)
    assert integration.points[..., :2].abs().max() > .05
    torch.testing.assert_close(targets.points, original, rtol=0, atol=0)
    args.implementation = "legacy"
    assert trainer.render_samples_for_view(view, args) is targets


@pytest.mark.parametrize("mode", ["before_validation", "during_validation"])
def test_hardened_mask_and_exposures_resume_through_real_entrypoint(monkeypatch, tmp_path, mode):
    from dataclasses import replace
    from scripts import validate_geraf_b7873200_entrypoint as harness
    runtime = harness.trainer
    original_args, original_load = harness._args, harness._fake_load_training_view
    preparations = []
    def args_for(path):
        args = original_args(path)
        args.implementation, args.compute_dtype = "hardened_v1", "float64"
        args.measured_mask_current_fraction = .05
        args.measured_mask_accumulated_fraction = .1
        args.measured_mask_grid = 3
        return args
    def load(*args):
        view = original_load(*args)
        return replace(view, samples=SimpleNamespace(points=torch.zeros(1, 1, 3)))
    def prepare(bank, cache):
        preparations.append(tuple(cache.train_indices))
        for index in cache.train_indices:
            bank.reference.add(index, torch.zeros(1, 3), torch.ones(1))
        bank.reference.finalize()
    def predict(model, view, frequencies, kvector, cache, args, *, create_graph):
        return SimpleNamespace(normalized_magnitude=harness._fake_predict(None, model, cache),
                               diagnostics={"synthetic": True})
    monkeypatch.setattr(harness, "_args", args_for)
    monkeypatch.setattr(harness, "_fake_load_training_view", load)
    monkeypatch.setattr(runtime, "_prepare_measured_mask_reference", prepare)
    monkeypatch.setattr(runtime, "predict_normalized_magnitude_with_response", predict)
    baseline = tmp_path / "baseline"
    harness._run_main(baseline, harness.Harness(None))
    expected = harness._load_checkpoint(baseline / "checkpoint_final.pth.tar")
    _, interrupted, resumed = harness._run_interrupted_and_resumed(tmp_path / mode, mode)
    extra = {"unmasked_mf_magnitude_mse", "excluded_target_energy_fraction",
             "render_diagnostics", "training_coverage"}
    for key in extra:
        harness._assert_same(expected["last_train"][key], resumed["last_train"][key], key)
    legacy_records = []
    for checkpoint in (expected, resumed):
        legacy_records.append({**checkpoint, "last_train": {
            k: v for k, v in checkpoint["last_train"].items() if k not in extra}})
    harness._assert_equivalent_checkpoint(*legacy_records)
    assert preparations == [(17,), (17,)]  # Resumption restores the measured reference.
    assert interrupted["training_coverage"]["per_view_exposures"] == {17: 1}
    assert resumed["training_coverage"]["per_view_exposures"] == {17: 2}
    assert resumed["dynamic_mask_bank"]["reference_digest"] == expected["dynamic_mask_bank"]["reference_digest"]
    broken = copy.deepcopy(resumed)
    broken["rng_state"]["view_sampler"]["exposures"][17] = 0
    with pytest.raises(ValueError, match="coverage"):
        runtime._validate_resume_checkpoint_payload(broken, args_for(baseline), harness._synthetic_cache())


def test_reference_preparation_loads_only_registered_training_targets(monkeypatch):
    calls = []
    bank = MeasuredViewMaskBank([7, 11], (1, 1, 2), extent=.15, size=3, identity={})
    cache = SimpleNamespace(train_indices=(7, 11), geraf_mf_magnitude_peak=2.)
    def load(index, role, seen_cache):
        calls.append((index, role))
        assert seen_cache is cache
        return {"geraf_mf_magnitude": torch.tensor([[[1., 2.]]])}
    monkeypatch.setattr(trainer, "_load_target", load)
    monkeypatch.setattr(trainer, "_samples_from_cached_geometry",
                        lambda target, device: SimpleNamespace(points=torch.tensor([[[0., 0., 0.], [.1, 0., 0.]]])))
    monkeypatch.setattr(trainer, "_STOP_REQUESTED", False)
    trainer._prepare_measured_mask_reference(bank, cache)
    assert calls == [(7, "train"), (11, "train")]
    assert bank.reference.volume.max() <= 1.


def test_native_geometry_uses_same_selected_object_checkpoint(monkeypatch):
    from rift.rift_dataset import _object_contract
    from scripts import eval_geraf_geometry as geometry
    monkeypatch.setattr("sys.argv", ["train_geraf.py", "--implementation", "hardened_v1", "--cache-root", "/tmp/no-cache",
                                    "--checkpoint-dir", "/tmp/no-checkpoint"])
    args = trainer.parse_args()
    config = trainer.model_config_from_args(args)
    model = GeRaFModel(**config)
    contract = _object_contract("loader")
    cache = SimpleNamespace(sealed_identity=contract, recipe={}, stats={"geraf_mf_magnitude_peak": 1.},
                            target_manifest={}, acquisition_record={}, effective_pairs_per_plane=256,
                            geraf_mf_magnitude_peak=1.)
    row = {"step": 1000, "views": 1000, "voxels": 1000*32**3,
           "mf_magnitude_mse": .25, "mf_magnitude_relative_mse": .4}
    checkpoint = {"step": 1000, "best_val_mse": .25, "history": [row], "cli_args": vars(args),
                  "model_config": config, "model_state_dict": model.state_dict(),
                  "run_identity": trainer.run_identity(args, cache, config), "acquisition_record": {},
                  "cache_recipe": {}, "target_manifest": {}, "target_stats": cache.stats}
    source = SimpleNamespace(identity=contract, arrays=None)
    monkeypatch.setattr(geometry, "build_b7873200_acquisition_record", lambda arrays: {})
    monkeypatch.setattr("scripts.eval_geraf_complex_response.acquisition_records_equal", lambda a, b: a == b)
    restored, selected = geometry.checkpoint_model(checkpoint, source, contract, torch.device("cpu"))
    assert selected == row and isinstance(restored.sdf_network, SourceSDFNetwork)
    with pytest.raises(ValueError):
        geometry.checkpoint_model(checkpoint, source, _object_contract("a320"), torch.device("cpu"))
    changed = {**checkpoint, "model_config": {**config, "implementation": "legacy"}}
    with pytest.raises(ValueError, match="configuration"):
        geometry.checkpoint_model(changed, source, contract, torch.device("cpu"))


@pytest.mark.parametrize("name", ["a320", "x59", "firetruck", "racecar", "loader", "b787"])
def test_collection_forwards_explicit_geraf_implementation(name, tmp_path):
    from train_rift_dataset import commands_for
    commands = []
    for implementation in ("hardened_v1", "legacy"):
        plan = commands_for(name, "geraf", dataset_root=tmp_path, output_root=tmp_path / "outputs",
                            geraf_implementation=implementation)
        assert plan[-1][plan[-1].index("--implementation") + 1] == implementation
        commands.append(plan)
    assert commands[0][0] == commands[1][0]  # Measured targets are recipe-independent.
