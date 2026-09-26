"""Equation, acquisition and recovery checks; synthetic inputs only."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from rift.sugavanam_ertin_paper import (Subapertures, aggregate_scattering,
    pca_normals, complex_soft_threshold, proximal_step, PaperSDF, field_gradient,
    project_surface, resample_moves, priority_candidate, refresh_surface, sdf_losses)
from rift.sugavanam_ertin_acquisition import (GOTCHAAcquisition, CollectionAcquisition,
    data_objective, training_statistics, response)
from rift.sugavanam_ertin_paper_workflow import (make_recipe, plan, run, validate_resume,
    extract_cloud, grid_points)


@pytest.fixture(autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_subapertures_ignore_manifest_order_cover_all_views_and_wrap():
    angles = np.deg2rad([359., 1., 31., 41., 180., 190., 271.])
    u = np.stack((np.cos(angles), np.sin(angles), np.zeros(len(angles))), -1)
    p = Subapertures.fit(u, 12, 1)
    permutation = np.array([4, 0, 3, 6, 2, 5, 1])
    q = Subapertures.fit(u[permutation], 12, 1)
    np.testing.assert_array_equal(q.assignments, p.assignments[permutation])
    np.testing.assert_allclose(q.directions, p.directions)
    assert np.bincount(p.assignments).sum() == len(u)
    assignment, fallback = p.assign(np.array([[0., 1., 0.]]))
    assert fallback.tolist() == [True]
    assert 0 <= assignment[0] < len(p.directions)


def test_opposite_angular_phases_do_not_cancel_support():
    coefficients = torch.tensor([[1+0j, 0j], [-1+0j, 2j]], dtype=torch.complex128)
    magnitude, strongest, index = aggregate_scattering(coefficients, [[1, 0, 0], [-1, 0, 0]])
    assert coefficients.sum(0)[0] == 0
    torch.testing.assert_close(magnitude, torch.tensor([2., 2.], dtype=torch.float64))
    assert index.tolist() == [0, 1]
    assert strongest.tolist() == [[1., 0., 0.], [-1., 0., 0.]]


def test_mean_aperture_direction_uses_every_native_pulse_not_equal_sector_weights():
    angles = np.deg2rad([1., 20.])
    counts = np.array([1., 9.])
    directions = np.stack((np.cos(angles), np.sin(angles), np.zeros(2)), -1)
    stats = np.stack((counts*np.cos(angles), counts*np.sin(angles), np.zeros(2), counts), -1)
    partition = Subapertures.fit(directions, 12, 1, direction_statistics=stats)
    mean = np.arctan2((counts*np.sin(angles)).sum(), (counts*np.cos(angles)).sum())
    np.testing.assert_allclose(partition.directions[0], [np.cos(mean), np.sin(mean), 0.])
    assert abs(mean - angles.mean()) > .05


def test_sparse_and_collinear_normals_use_strongest_view_not_distant_knn():
    p = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.]])
    fallback = np.tile([0., 0., -1.], (4, 1))
    for radius in (.01, 4.):
        n, valid, used = pca_normals(p, radius, fallback_directions=fallback)
        np.testing.assert_array_equal(n, fallback)
        assert valid.all() and used.all()
    n, valid, used = pca_normals(p, .01)
    assert not valid.any() and not used.any() and not n.any()


def test_pca_plane_axes_are_geometric_and_no_radial_orientation():
    x, y = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
    p = np.stack((x.ravel(), y.ravel(), np.zeros(x.size)), -1)
    n, valid, used = pca_normals(p, 3.)
    assert valid.all() and not used.any()
    np.testing.assert_allclose(np.abs(n[:, 2]), 1.)


def test_proximal_solver_matches_analytic_complex_lasso_and_descent():
    b = torch.tensor([2+3j, -.3+.1j, 0j], dtype=torch.complex128)
    lam = .4
    value = lambda x: .5*float((x-b).abs().square().sum())
    grad = lambda x: (value(x), x-b)
    x = torch.zeros_like(b)
    x, L, audit = proximal_step(x, grad, value, l1=lam, lipschitz=.125)
    torch.testing.assert_close(x, complex_soft_threshold(b, lam), atol=1e-14, rtol=1e-14)
    assert L == 1 and audit["backtracks"] == 3
    assert audit["data_loss"]+audit["l1_loss"] <= value(torch.zeros_like(b))
    assert x[1] == 0 and x[2] == 0


class Plane(nn.Module):
    def __init__(self, offset=0., scale=1.):
        super().__init__()
        self.offset, self.scale = offset, scale
    def forward(self, x):
        return self.scale*(x[:, 2]-self.offset)


class Constant(nn.Module):
    def forward(self, x):
        return torch.full((len(x),), 5e-5, dtype=x.dtype, device=x.device)


def test_projection_rejects_flat_near_zero_and_missing_surfaces_under_no_grad():
    p = torch.tensor([[0., 0., .01], [.02, -.01, -.02]], dtype=torch.float64)
    with torch.no_grad():
        q, audit = project_surface(Plane(), p, extent=.15, max_step=.04, tolerance=1e-5)
        flat, flat_audit = project_surface(Constant(), p, extent=.15, max_step=.04, tolerance=1e-4)
        outside, _ = project_surface(Plane(offset=1.), p, extent=.15, max_step=.04, tolerance=1e-5)
    assert len(q) == len(p) and audit["accepted"] == 2
    torch.testing.assert_close(q[:, 2], torch.zeros(2, dtype=torch.float64))
    assert len(flat) == len(outside) == 0
    assert flat_audit["degenerate_gradient"] == 2


def test_sdf_query_dtype_and_second_derivatives():
    model = PaperSDF(.15, hidden_dim=4, n_layers=4, n_fourier=2)
    p = torch.rand(7, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(2))*.1
    with torch.no_grad():
        f, g = field_gradient(model, p)
    assert f.dtype == g.dtype == torch.float32
    f, g = field_gradient(model, p, create_graph=True)
    loss = f.square().mean() + (g.norm(dim=-1)-1).square().mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in model.parameters())


def test_standard_gaussian_init_is_explicit_deterministic_and_fourth_layer_skip():
    a = PaperSDF(1., hidden_dim=64, n_layers=4, n_fourier=3, initialization="standard_gaussian")
    b = PaperSDF(1., hidden_dim=64, n_layers=4, n_fourier=3, initialization="standard_gaussian")
    assert a.layers[3].in_features == 64+9
    assert a.layers[2].in_features == 64
    for k, v in a.state_dict().items():
        torch.testing.assert_close(v, b.state_dict()[k], atol=0, rtol=0)
    assert a.layers[0].bias.std() > .5


def test_initialization_scale_changes_all_linear_parameters_but_not_fourier_features():
    settings = dict(hidden_dim=64, n_layers=4, n_fourier=3)
    literal = PaperSDF(.15, **settings)
    selected = PaperSDF(.15, initialization_std=.05, **settings)
    torch.testing.assert_close(selected.bands, literal.bands, atol=0, rtol=0)
    for name, value in selected.named_parameters():
        torch.testing.assert_close(value, dict(literal.named_parameters())[name]*.05, atol=0, rtol=0)
    assert selected.model_config["initialization_std"] == .05
    assert "initialization_std" not in literal.model_config


@pytest.mark.parametrize("kind", ["rift_collection", "gotcha_native"])
def test_training_initialization_default_and_explicit_recipe_identities(kind):
    from rift.sugavanam_ertin_paper_workflow import _model_config
    root = Path(__file__).resolve().parents[1]
    selected = make_recipe(kind)
    assert selected["initialization_std"] == .05
    assert selected["fidelity"] == "user_requested_gaussian_initialization_std_override"
    assert _model_config(selected, .15)["initialization_std"] == .05
    assert make_recipe(kind, json.loads((root/"protocols/se_g40_readout48.json").read_text())) == selected
    assert make_recipe(kind, {"initialization_std": .1})["initialization_std"] == .1
    literal = make_recipe(kind, json.loads((root/"protocols/se_paper_std1.json").read_text()))
    assert "initialization_std" not in literal
    assert literal["fidelity"] == "paper_equations_declared_author_gaps"
    acquisition = TinyAcquisition()
    saved = dict(schema=literal["schema"], acquisition=acquisition.identity, recipe=literal)
    with pytest.raises(ValueError, match="recipe changed"):
        validate_resume(saved, acquisition, selected, None)
    assert not acquisition.reads


@pytest.mark.parametrize("kind,extent", [("rift_collection", .15), ("gotcha_native", 5.)])
def test_selected_initialization_has_spatial_and_parameter_gradients(kind, extent):
    from rift.sugavanam_ertin_paper import initialization_audit
    from rift.sugavanam_ertin_paper_workflow import _model_config
    recipe = make_recipe(kind)
    model = PaperSDF(**_model_config(recipe, extent))
    audit = initialization_audit(model, extent, seed=recipe["seed"])
    assert audit["status"] == "initialization_probe_passed"
    assert audit["nonzero_spatial_gradients"] == audit["samples"]
    assert audit["saturated_fraction"] == 0.
    points = (torch.rand(32, 3, generator=torch.Generator().manual_seed(42))*2-1)*extent
    f, g = field_gradient(model, points, create_graph=True)
    (f.square().mean() + (g.norm(dim=-1)-1).square().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero()
               for p in model.parameters())


def test_resampling_equations_against_independent_small_formula():
    p = np.array([[0., 0., 0.], [.5, 0., .1], [0., .8, -.2]])
    n = np.tile([0., 0., 1.], (3, 1))
    uniform, edge = resample_moves(p, n, radius=2, bandwidth=.6, alpha=.05, max_step=10)
    d = p[1:]-p[0]
    w = np.exp(-np.sum(d*d, axis=1)/.6**2)
    phi = np.exp(-d[:, 2]/.6**2)
    expected_uniform = -.05*np.sum(w[:, None]*d/np.linalg.norm(d, axis=1)[:, None], axis=0)
    expected_edge = -np.sum(phi[:, None]*d, axis=0)/phi.sum()-.5*np.sum(w[:, None]*d, axis=0)/w.sum()
    np.testing.assert_allclose(uniform[0], expected_uniform)
    np.testing.assert_allclose(edge[0], expected_edge)
    with pytest.raises(ValueError, match="Eq. 13"):
        resample_moves(p, n, radius=2, bandwidth=.6, alpha=.05, max_step=10, edge_weight="squared_distance")


def test_priority_insertion_selects_sparse_region_and_asymmetric_third():
    p = np.array([[0., 0., 0.], [.01, 0., 0.], [2., 0., 0.], [3., 0., 0.]])
    candidate, pair = priority_candidate(p, radius=1.1)
    assert pair == (2, 3)
    np.testing.assert_allclose(candidate, [7/3, 0., 0.])
    assert priority_candidate(p, radius=.001) is None


def test_resampling_plane_has_real_roots_and_priority_insertions():
    x, y = torch.meshgrid(torch.linspace(-.04, .04, 5), torch.linspace(-.04, .04, 5), indexing="ij")
    p = torch.stack((x.flatten(), y.flatten(), torch.zeros(25)), -1)
    q, audit = refresh_surface(Plane(), p, extent=.15, pitch=.01, count=32,
                               generator=torch.Generator().manual_seed(5))
    assert len(q) >= 16 and audit["insertions"] > 0
    assert q[:, 2].abs().max() < 1e-6
    assert (q.abs() < .15).all()
    assert len(torch.unique(q, dim=0)) == len(q)


def test_six_losses_follow_raw_metric_equations_and_unsigned_normal_targets():
    on = torch.tensor([[0., 0., 0.], [.02, 0., 0.]])
    normal = torch.tensor([[0., 0., 1.], [0., 0., -1.]])
    off = torch.tensor([[0., 0., .02], [0., 0., -.03]])
    iso = on.clone()
    a = sdf_losses(Plane(), on, normal, off, iso, normal, extent=.15)
    b = sdf_losses(Plane(scale=-1), on, normal, off, iso, normal, extent=15.)
    assert set(a) == {"on", "off", "normal", "iso", "iso_normal", "eik"}
    for k in a:
        torch.testing.assert_close(a[k], b[k])
    torch.testing.assert_close(a["off"], torch.exp(-100*off[:, 2].abs()).mean())
    shifted = sdf_losses(Plane(offset=.01), on, normal, off, iso, normal, extent=.15)
    torch.testing.assert_close(shifted["on"], torch.tensor(.01))
    torch.testing.assert_close(shifted["iso"], torch.tensor(.01))
    assert a["normal"] == a["iso_normal"] == a["eik"] == 0
    wrong_normal = torch.tensor([[1., 0., 0.], [1., 0., 0.]])
    wrong = sdf_losses(Plane(), on, normal, off, iso, wrong_normal, extent=.15)
    assert wrong["iso_normal"] == 1  # Would be zero with tautological gradient targets.


class TinyAcquisition:
    """Small complex linear fixture with two opposing angular responses."""
    kind, extent, kernel_scale = "synthetic", .15, 1.
    def __init__(self):
        self.keys = dict(train=[0, 1, 2, 3], validation=[4, 5])
        self.train_sample_counts = [8]*4
        directions = np.array([[1., .1, 0.], [1., .2, 0.], [-1., -.1, 0.], [-1., -.2, 0.]])
        self.directions = dict(train=directions, validation=directions[[0, 2]])
        self.identity = dict(kind="synthetic_se_unit_fixture", sealed_test=True)
        self.reads = []
    def observations(self, key, *, role):
        if key not in self.keys.get(role, []):
            raise PermissionError("sealed")
        self.reads.append((role, key))
        sign = 1 if key in (0, 1, 4) else -1
        yield dict(response=sign*np.linspace(.5, 1., 8).astype(np.complex128))
    def render(self, points, weights, observation, **kwargs):
        return weights


def tiny_recipe():
    return make_recipe("synthetic", dict(granularity=2, azimuth_bins=2, elevation_bins=1,
        stage1_iterations=3, validation_every=1, checkpoint_every=1, initial_lipschitz=.125,
        stage2_steps=3, hidden_dim=8, n_layers=4, n_fourier=2, batch_on=8,
        batch_off=8, batch_iso=8, iso_count=8, iso_start=1, iso_every=1,
        projection_iterations=4, export_grid=8))


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def compare_nested(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            compare_nested(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            compare_nested(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("stop_after", [1, 7])
def test_both_stages_resume_exactly_and_cover_every_training_view(tmp_path, stop_after):
    recipe = tiny_recipe()
    continuous = tmp_path/"continuous"
    interrupted = tmp_path/"interrupted"
    run(TinyAcquisition(), recipe, continuous, device="cpu")
    calls = []
    def stop():
        calls.append(1)
        return len(calls) == stop_after
    result = run(TinyAcquisition(), recipe, interrupted, device="cpu", should_stop=stop)
    assert result["status"] == "interrupted"
    assert result["phase"] == ("stage1" if stop_after == 1 else "stage2")
    run(TinyAcquisition(), recipe, interrupted, device="cpu", resume=interrupted/"checkpoint_latest.pt")
    a, b = load(continuous/"checkpoint_final.pt"), load(interrupted/"checkpoint_final.pt")
    compare_nested(a, b)
    assert a["view_exposures"] == [3, 3, 3, 3]
    assert a["cloud"]["aggregation"] == "sum_abs_subapertures"
    assert a["recipe"]["sdf_signal"] == "not_defined"


@pytest.mark.parametrize("mutation", ["object", "recipe", "coverage", "legacy", "normalization"])
def test_resume_rejects_changed_contracts_before_responses(tmp_path, mutation):
    recipe, a = tiny_recipe(), TinyAcquisition()
    run(a, recipe, tmp_path, device="cpu", should_stop=lambda: True)
    state = load(tmp_path/"checkpoint_latest.pt")
    if mutation == "object": state["acquisition"]["kind"] = "wrong_object"
    if mutation == "recipe": state["recipe"]["residual_relative_energy"] = .9
    if mutation == "coverage": state["view_exposures"][0] += 1
    if mutation == "legacy": state["schema"] = "old_isotropic_checkpoint"
    if mutation == "normalization": state["statistics"]["identity"] = "wrong"
    torch.save(state, tmp_path/"checkpoint_latest.pt")
    b = TinyAcquisition()
    with pytest.raises(ValueError):
        run(b, recipe, tmp_path, device="cpu", resume=tmp_path/"checkpoint_latest.pt")
    assert b.reads == []


def test_planning_and_unknown_configuration_never_read_responses():
    a = TinyAcquisition()
    p, validation, report = plan(a, tiny_recipe())
    assert not a.reads and not report["response_payload_read"]
    assert sum(report["subaperture_train_counts"]) == 4
    for invalid in ({"stage1_iterations": True}, {"residual_relative_energy": float("nan")}, {"unknown": 3}, {"sdf_signal": "complex"}):
        with pytest.raises(ValueError): make_recipe("synthetic", invalid)


def test_gotcha_fourier_kernel_all_native_frequencies_and_gradient(tmp_path):
    from tests.se_dataset_fixtures import write_shard, tiny_region
    from rift.gotcha_dataset import GOTCHADataset, C
    root = tmp_path/"New_Transfer"/"shards"
    write_shard(root/"pass1_hh.npz", nf=9)
    write_shard(root/"pass2_hh.npz", pass_id=2, nf=11)
    ds = GOTCHADataset(tmp_path, passes=(1, 2), region=tiny_region())
    a = GOTCHAAcquisition(ds)
    recipe = make_recipe(a.kind)
    plan(a, recipe)
    assert sum(s.response_reads for s in ds.shards.values()) == 0
    assert len(a.keys["train"]) == 500
    observation = next(a.observations(a.keys["train"][0], role="train"))
    x = torch.tensor([[.01, -.02, .005], [-.005, 0., .01]], dtype=torch.float64, requires_grad=True)
    w = torch.tensor([.4+.2j, -.2+.1j], dtype=torch.complex128, requires_grad=True)
    actual = a.render(x, w, observation, point_chunk=1, pair_chunk=1)
    antenna = torch.from_numpy(observation.position_m)
    f = torch.from_numpy(observation.frequencies_hz.copy())
    distance = antenna.norm()-x @ (antenna/antenna.norm())-observation.reference_range_m
    expected = (w[:, None]*torch.exp(-4j*torch.pi*f[None, :]/C*distance[:, None])).sum(0)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    ga = torch.autograd.grad(actual.abs().square().sum(), (x, w), retain_graph=True)
    ge = torch.autograd.grad(expected.abs().square().sum(), (x, w))
    for u, v in zip(ga, ge): torch.testing.assert_close(u, v, atol=1e-10, rtol=1e-10)
    with pytest.raises(PermissionError):
        next(a.observations((1, ds.split["test"][0]), role="train"))


def test_collection_renderer_asymmetric_pairs_against_direct_reference():
    from rift.config import cc
    a = CollectionAcquisition.__new__(CollectionAcquisition)
    a.kernel_scale = 1.
    x = torch.tensor([[.01, -.02, .005], [-.005, 0., .01]], dtype=torch.float64)
    w = torch.tensor([.4+.2j, -.2+.1j], dtype=torch.complex128, requires_grad=True)
    tx = np.array([[10., 0., 0.], [10., .013, 0.]])
    rx = np.array([[10., -.02, 0.], [10., .004, .01], [10., -.007, .002]])
    freqs = np.linspace(8.5e9, 11.49e9, 12)
    observation = dict(tx=tx, rx=rx, freqs=freqs)
    result = a.render(x, w, observation, point_chunk=1, pair_chunk=2)
    tx, rx = torch.from_numpy(tx), torch.from_numpy(rx)
    rtx = tx.norm(dim=-1)[None]-x @ (tx/tx.norm(dim=-1)[:, None]).T
    rrx = rx.norm(dim=-1)[None]-x @ (rx/rx.norm(dim=-1)[:, None]).T
    distance = rrx[:, :, None]+rtx[:, None, :]
    origin_distance = rx.norm(dim=-1)[:, None]+tx.norm(dim=-1)[None]
    kernel = torch.exp(-2j*torch.pi/cc*torch.from_numpy(freqs)[None, :, None, None]*distance[:, None])/(4*torch.pi)**2/origin_distance[None, None]**2
    expected = (w[:, None, None, None]*kernel).sum(0)
    assert result.shape == (12, 3, 2)
    torch.testing.assert_close(result, expected, atol=1e-12, rtol=2e-8)
    ga = torch.autograd.grad(result.abs().square().sum(), w, retain_graph=True)[0]
    ge = torch.autograd.grad(expected.abs().square().sum(), w)[0]
    torch.testing.assert_close(ga, ge, atol=1e-15, rtol=2e-8)


def test_cli_dispatch_keeps_legacy_workflow_and_registers_hh_only(monkeypatch):
    import train_sugavanam_ertin as cli
    from rift import sugavanam_ertin_paper_workflow as workflow
    called = []
    monkeypatch.setattr(workflow, "main", lambda argv: called.append(argv) or 0)
    assert cli.main(["--recipe", "paper-v1", "--object", "a320", "--dry-run"]) == 0
    assert len(called) == 1
    assert cli.parse_args([]).recipe == "legacy-full"
    spec = cli.GOTCHA_BACKEND
    assert spec["polarizations"] == ["hh"]
    assert spec["fidelity_status"] == "published_initialization_unresolved_not_benchmark_ready"


def test_se_owned_collection_dispatch_selects_paper_recipe_with_explicit_legacy(tmp_path):
    from train_rift_dataset import commands_for
    for name in ("a320", "x59", "firetruck", "racecar", "loader", "b787"):
        command = commands_for(name, "se", dataset_root=tmp_path, output_root=tmp_path/"runs")[0]
        assert command[command.index("--recipe")+1] == "paper-v1"
    command = commands_for("loader", "se", dataset_root=tmp_path, output_root=tmp_path/"runs",
                           se_recipe="legacy-full")[0]
    assert command[command.index("--recipe")+1] == "legacy-full"
    with pytest.raises(ValueError):
        commands_for("loader", "se", dataset_root=tmp_path, output_root=tmp_path, se_recipe="unknown")


@pytest.mark.parametrize("mutation", ["cloud", "optimizer_step", "optimizer_lr", "iso", "history"])
def test_stage2_recovery_rejects_partial_or_swapped_state_before_data(tmp_path, mutation):
    recipe = tiny_recipe()
    calls = []
    def stop():
        calls.append(1)
        return len(calls) == 7
    run(TinyAcquisition(), recipe, tmp_path, device="cpu", should_stop=stop)
    state = load(tmp_path/"checkpoint_latest.pt")
    assert state["phase"] == "stage2" and state["sdf_step"] == 1
    if mutation == "cloud": state["cloud"]["points"][0, 0] += .001
    if mutation == "optimizer_step":
        next(iter(state["sdf_optimizer"]["state"].values()))["step"] += 1
    if mutation == "optimizer_lr": state["sdf_optimizer"]["param_groups"][0]["lr"] *= 2
    if mutation == "iso":
        state["iso_points"] = torch.full((3, 3), 2.)
        state["iso_normals"] = torch.zeros(3, 3)
        state["iso_normal_valid"] = torch.ones(3, dtype=torch.bool)
    if mutation == "history": state["sdf_history"].clear()
    torch.save(state, tmp_path/"checkpoint_latest.pt")
    a = TinyAcquisition()
    with pytest.raises(ValueError): run(a, recipe, tmp_path, device="cpu", resume=tmp_path/"checkpoint_latest.pt")
    assert not a.reads


def test_geometry_readout_preserves_disconnected_support_and_provenance(tmp_path):
    from rift.sugavanam_ertin_paper_workflow import _surface_export
    class TwoSpheres(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.))
        def forward(self, x):
            a = torch.tensor([.065, 0., 0.]).to(x)
            return self.scale*torch.minimum((x-a).norm(dim=-1)-.025, (x+a).norm(dim=-1)-.025)
    provenance = dict(iso_supervised=True, iso_normals_supervised=True, acquisition_sha256="synthetic")
    report = _surface_export(TwoSpheres(), tmp_path, .15, 40, provenance)
    assert report["status"] == "complete"
    assert report["boundary_edges"] == report["nonmanifold_edges"] == 0
    with np.load(tmp_path/"surface.npz") as surface:
        assert (surface["vertices"][:, 0] > 0).any() and (surface["vertices"][:, 0] < 0).any()
        assert json.loads(str(surface["provenance_json"]))["provenance"] == provenance
    provenance["iso_normals_supervised"] = False
    report = _surface_export(TwoSpheres(), tmp_path, .15, 32, provenance)
    assert report["status"] == "incomplete_iso_supervision"


def test_gotcha_budget_exhaustion_covers_all_pulses_without_claiming_eq4_solved(tmp_path):
    from tests.se_dataset_fixtures import write_shard, tiny_region
    from rift.gotcha_dataset import GOTCHADataset
    write_shard(tmp_path/"data"/"New_Transfer"/"shards"/"pass1_hh.npz", nf=5)
    dataset = GOTCHADataset(tmp_path/"data", passes=(1,), region=tiny_region())
    acquisition = GOTCHAAcquisition(dataset)
    config = dict(granularity=2, azimuth_bins=2, elevation_bins=1, stage1_iterations=1,
        stage2_steps=1, hidden_dim=8, n_layers=4, n_fourier=2, batch_on=8, batch_off=8,
        batch_iso=8, iso_count=8, iso_start=1, projection_iterations=4, export_grid=8)
    recipe = make_recipe("synthetic", config)
    result = run(acquisition, recipe, tmp_path/"run", device="cpu")
    assert result["status"] == "stage1_unconverged"
    checkpoint = load(tmp_path/"run"/"checkpoint_final.pt")
    assert checkpoint["view_exposures"] == [1]*250
    assert checkpoint["statistics"]["per_view_samples"] == [5]*250
    assert checkpoint["best"]["validation"]["samples"] == 55*5
    assert "sdf_step" not in checkpoint
    assert checkpoint["acquisition"]["frequencies"] == "ragged_exact"


def test_terminal_surface_tampering_is_rejected(tmp_path):
    recipe = tiny_recipe()
    run(TinyAcquisition(), recipe, tmp_path, device="cpu")
    ck = load(tmp_path/"checkpoint_final.pt")
    # The tiny random-field trajectory may have no surface; test the invariant
    # with an explicit terminal artifact record without assuming convergence.
    ck["surface"]["artifact_sha256"] = "incorrect_hash"
    torch.save(ck, tmp_path/"checkpoint_final.pt")
    acquisition = TinyAcquisition()
    with pytest.raises(ValueError, match="surface artifact"):
        run(acquisition, recipe, tmp_path, device="cpu", resume=tmp_path/"checkpoint_final.pt")
    assert not acquisition.reads


def test_published_settings_cannot_be_silently_replaced_by_convenient_defaults():
    from rift.sugavanam_ertin_paper import SCHEMA
    r = make_recipe("gotcha_native")
    assert r["azimuth_bins"] == 72 and r["elevation_bins"] == 1
    assert r["normal_radius_m"] == .3 and r["initialization"] == "standard_gaussian"
    assert r["hidden_dim"] == 512 and r["n_layers"] == 8 and r["n_fourier"] == 9
    assert r["iso_start"] == 1 and r["granularity"] == 40 and r["export_grid"] == 48
    assert r["schema"] == SCHEMA and SCHEMA.endswith("_v2")
    for changed in ({"initialization": "variance_scaled"}, {"normal_radius_m": .03},
                    {"hidden_dim": 128}, {"edge_weight": "squared_distance"},
                    {"azimuth_bins": 12}, {"n_fourier": 6}, {"lambda_iso": 0.},
                    {"cloud_max_points": 5000}, {"l1": .001}):
        with pytest.raises(ValueError): make_recipe("gotcha_native", changed)


def test_network_coordinates_and_tanh_are_not_scaled_by_extent():
    a = PaperSDF(.15, hidden_dim=4, n_layers=4, n_fourier=2)
    b = PaperSDF(5., hidden_dim=4, n_layers=4, n_fourier=2)
    points = torch.tensor([[.02, -.01, .04], [0., 0., 0.]])
    torch.testing.assert_close(a(points), b(points), atol=0, rtol=0)
    assert a.layers[0].bias.count_nonzero() == 4


def test_projection_uses_published_residual_not_added_metric_distance_test():
    p = torch.tensor([[0., 0., .01]], dtype=torch.float64)
    q, audit = project_surface(Plane(scale=1e-9), p, extent=.15, max_step=.04)
    # Nonzero gradient and residual already below 1e-4: no extra distance,
    # gradient floor, or sign-bracket rule may alter the paper's stopping test.
    torch.testing.assert_close(p, q, atol=0, rtol=0)
    assert audit["accepted"] == 1


@pytest.mark.parametrize("count", [64, 64000])
def test_l1_projection_boundary_roundoff_preserves_finite_complex_phase(count):
    import math
    from rift.sugavanam_ertin_sparse import project_complex_l1_ball
    x = torch.full((count,), 1e-17j, dtype=torch.complex128)
    x[0] = 1j
    radius = math.nextafter(float(x.abs().sum()), 0.)
    projected = project_complex_l1_ball(x, radius)
    assert torch.isfinite(projected).all()
    tolerance = 8*torch.finfo(x.real.dtype).eps*radius
    assert float(projected.abs().sum()) <= radius+tolerance
    assert not projected.real.count_nonzero() and bool((projected.imag >= 0).all())
    torch.testing.assert_close(projected, x, atol=tolerance, rtol=0)
    torch.testing.assert_close(project_complex_l1_ball(projected, radius), projected,
                               atol=tolerance, rtol=0)


def test_l1_projection_retains_exact_shrinkage_and_rejects_invalid_inputs():
    from rift.sugavanam_ertin_sparse import project_complex_l1_ball
    x = torch.tensor([3j, -4., .5j], dtype=torch.complex128)
    expected = torch.tensor([1.5j, -2.5, 0j], dtype=torch.complex128)
    torch.testing.assert_close(project_complex_l1_ball(x, 4.), expected, atol=1e-15, rtol=0)
    torch.testing.assert_close(project_complex_l1_ball(x, 0.), torch.zeros_like(x))
    torch.testing.assert_close(project_complex_l1_ball(x, 8.), x)
    for radius in (-1., float("nan"), float("inf")):
        with pytest.raises(ValueError, match="Invalid complex L1 radius"):
            project_complex_l1_ball(x, radius)
    for threshold in (-1e-19, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="Invalid L1 threshold"):
            complex_soft_threshold(x, threshold)


def test_constraint_solver_matches_independent_diagonal_complex_kkt_solution():
    from rift.sugavanam_ertin_sparse import constrained_step, initial_state
    d = torch.tensor([.5, 1., 1.5, 2.], dtype=torch.float64)
    y = torch.tensor([1+.7j, -.2+.3j, 1.2-2j, .01j], dtype=torch.complex128)
    target = .5*.04*float(y.abs().square().mean())
    value = lambda x: .5*float((d*x-y).abs().square().mean())
    grad = lambda x: (value(x), d*(d*x-y)/len(y))
    # Independent scalar KKT multiplier bisection, not the L1-radius solver.
    lo, hi = 0., float((d*y.abs()).max())
    for _ in range(80):
        lam = (lo+hi)/2
        residual = torch.minimum(y.abs(), lam/d)
        if .5*float(residual.square().mean()) < target: lo = lam
        else: hi = lam
    expected = complex_soft_threshold(y/d, 0.) * (1-(lo/d)/y.abs()).clamp_min(0)
    state, x = initial_state(target, .01), torch.zeros_like(y)
    for _ in range(2000):
        x, state, audit = constrained_step(x, grad, value, state,
            residual_rtol=1e-7, optimality_rtol=1e-9)
        if state["converged"]: break
    assert state["converged"] and audit["residual_feasible"]
    torch.testing.assert_close(x, expected, atol=2e-6, rtol=2e-6)
    assert abs(value(x)-target)/target < 1e-7


def test_infeasible_constraint_does_not_masquerade_as_converged_sparse_solution():
    from rift.sugavanam_ertin_sparse import constrained_step, initial_state
    x = torch.zeros(4, dtype=torch.complex128)
    candidate, state, audit = constrained_step(x, lambda x: (1., x*0), lambda x: 1., initial_state(.01))
    assert state["stalled"] and not state["converged"] and not audit["residual_feasible"]
    assert not candidate.count_nonzero()


@pytest.mark.parametrize("std", [1., .1])
def test_default_initialization_uncertainty_is_exposed_without_fitting_or_responses(tmp_path, std):
    from rift.sugavanam_ertin_paper import initialization_audit
    model = PaperSDF(.15, initialization_std=std)
    audit = initialization_audit(model, .15)
    assert audit["status"] == "initialization_degenerate"
    assert audit["saturated_fraction"] == 1. and audit["nonzero_spatial_gradients"] == 0
    a = TinyAcquisition()
    recipe = make_recipe("synthetic", {"granularity": 2, "initialization_std": std})
    report = run(a, recipe, tmp_path, device="cpu")
    assert report["status"] == "initialization_degenerate" and not report["benchmark_eligible"]
    assert not a.reads and not (tmp_path/"checkpoint_final.pt").exists()
    assert (tmp_path/"initialization.json").is_file()


def test_resume_rejects_forged_constrained_solver_before_response_access(tmp_path):
    recipe = tiny_recipe()
    run(TinyAcquisition(), recipe, tmp_path, device="cpu", should_stop=lambda: True)
    state = load(tmp_path/"checkpoint_latest.pt")
    state["sparse_solvers"][0]["converged"] = True
    torch.save(state, tmp_path/"checkpoint_latest.pt")
    a = TinyAcquisition()
    with pytest.raises(ValueError, match="last committed"):
        run(a, recipe, tmp_path, device="cpu", resume=tmp_path/"checkpoint_latest.pt")
    assert not a.reads


def test_metadata_grid_tracks_range_resolution_without_an_oversampling_improvement():
    a = TinyAcquisition()
    a.range_resolution_m = .05
    _, _, report = plan(a, make_recipe("synthetic", {"granularity": 0}))
    assert report["recipe"]["granularity"] == 6
    assert report["voxel_pitch_m"] == pytest.approx(.05)
    assert not a.reads


def test_gotcha_subapertures_split_boundary_sector_without_losing_or_repeating_pulses(tmp_path):
    from tests.se_dataset_fixtures import write_shard, tiny_region
    from rift.gotcha_dataset import GOTCHADataset, sector_split
    sector = sector_split()["train"][0]
    def double_boundary(arrays, meta):
        row = int(np.flatnonzero(arrays["sector_id"] == sector)[0])
        for k, v in list(arrays.items()):
            if k != "frequencies_hz": arrays[k] = np.concatenate((v, v[row:row+1]), axis=0)
        for index, angle in ((row, 4.9), (360, 5.1)):
            arrays["x"][index] = 20*np.cos(np.deg2rad(angle))
            arrays["y"][index] = 20*np.sin(np.deg2rad(angle))
        arrays["pulse_index"][-1] = 1
    write_shard(tmp_path/"New_Transfer"/"shards"/"pass1_hh.npz", nf=5, mutate=double_boundary)
    ds = GOTCHADataset(tmp_path, passes=(1,), region=tiny_region())
    a = GOTCHAAcquisition(ds)
    keys = [key for key in a.keys["train"] if key[1] == sector]
    assert len(keys) == 2 and {key[2] for key in keys} == {0, 1}
    partition, _, report = plan(a, make_recipe(a.kind))
    assert report["native_view_counts"]["train"] == 250
    assert sum(a.train_sample_counts) == 251*5
    assert sum(s.response_reads for s in ds.shards.values()) == 0
    pulses = [(key[2], o.pulse_index) for key in keys for o in a.observations(key, role="train")]
    assert pulses == [(0, 0), (1, 1)]
    assert sum(s.response_reads for s in ds.shards.values()) == 2


def test_gotcha_fourier_phase_respects_translated_rotated_working_frame(tmp_path):
    from tests.se_dataset_fixtures import write_shard
    from rift.gotcha_dataset import GOTCHADataset, Region, C
    write_shard(tmp_path/"New_Transfer"/"shards"/"pass1_hh.npz", nf=7)
    rotation = ((0., -1., 0.), (1., 0., 0.), (0., 0., 1.))
    translation = (1., -.5, .2)
    region = Region("offset_fixture", "synthetic", translation, rotation, .03, "synthetic test")
    a = GOTCHAAcquisition(GOTCHADataset(tmp_path, passes=(1,), region=region))
    observation = next(a.observations(a.keys["train"][0], role="train"))
    x = torch.tensor([[.01, -.02, .005]], dtype=torch.float64, requires_grad=True)
    w = torch.tensor([.4+.2j], dtype=torch.complex128, requires_grad=True)
    actual = a.render(x, w, observation, point_chunk=1, pair_chunk=1)
    displacement = x @ torch.tensor(rotation, dtype=torch.float64).T
    antenna_from_origin = torch.from_numpy(observation.position_m)-torch.tensor(translation, dtype=torch.float64)
    r = antenna_from_origin.norm()
    path = r-observation.reference_range_m-(displacement @ (antenna_from_origin/r))
    expected = w[0]*torch.exp(-4j*torch.pi*torch.from_numpy(observation.frequencies_hz.copy())/C*path[0])
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    assert torch.autograd.grad(actual.real.sum(), x)[0].abs().max() > 0


def test_stage1_handoff_does_not_cherry_pick_a_better_validation_iteration(tmp_path, monkeypatch):
    from rift import sugavanam_ertin_paper_workflow as workflow
    metrics = iter([.1, .9, 2.])
    monkeypatch.setattr(workflow, "validation_readout", lambda *args: dict(global_complex_rel_mse=next(metrics)))
    run(TinyAcquisition(), tiny_recipe(), tmp_path, device="cpu")
    checkpoint = load(tmp_path/"checkpoint_final.pt")
    assert checkpoint["stage1_source"]["iteration"] == 3
    assert checkpoint["stage1_source"]["validation"]["global_complex_rel_mse"] == 2.
    assert checkpoint["stage1_source"]["selection"] == "constrained_terminal"


def test_gotcha_hook_cannot_exit_successfully_for_an_ineligible_fit(tmp_path, monkeypatch):
    from rift import sugavanam_ertin_paper_workflow as workflow
    from rift import sugavanam_ertin_stage2_runtime_v1 as runtime
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic_fixture")
    monkeypatch.setattr(workflow, "GOTCHAAcquisition", lambda _: TinyAcquisition())
    monkeypatch.setattr(workflow, "run", lambda *args, **kwargs: dict(status="initialization_degenerate"))
    monkeypatch.setattr(runtime, "install_stop_handlers", lambda: None)
    with pytest.raises(RuntimeError, match="initialization_degenerate"):
        workflow.run_gotcha(dataset=None, output_dir=tmp_path, config={}, device="cpu", resume=None)
