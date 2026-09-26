"""Data-free fidelity, role isolation, renderer and recipe regression checks."""
import ast
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d
import torch

from rift.radarsplat_b7873200 import RadarSplatEffects, RadarSplatGrid, RadarSplatModel, additive_gaussian_rasterization
from rift.radarsplat_b7873200_adapter import native_power_objective, ObjectiveWeights
from rift.radarsplat_fidelity import (OccupancyRecipe, TrainingOccupancy, denoise_power,
                                    polar_world_points, sample_polar_power, release_ssim_index)


def grid(stride=9):
    return RadarSplatGrid(num_range_bins=16, range_resolution_m=0.1, range_start_m=9.2,
        azimuth_start_deg=-7.2, azimuth_span_deg=14.4,
        output_azimuth_resolution_deg=0.9, intermediate_azimuth_resolution_deg=0.9/stride,
        spectral_leakage_width_m=0.7)


@pytest.mark.parametrize("stride", [1, 9, 10])
def test_antenna_ablation_shape_and_centres(stride):
    g = grid(stride)
    row = torch.arange(g.intermediate_azimuth_bins, dtype=torch.float64)
    img = row[None, :, None].expand(1, -1, 16)
    out = RadarSplatModel._process_image(img, g, RadarSplatEffects(
        use_spectral_leakage=False, use_azimuth_antenna_gain=False))
    assert out.shape == (1, 16, 16)
    torch.testing.assert_close(out[0, :, 0], row[::stride])
    first = g.intermediate_azimuth_start_deg + .5*g.intermediate_azimuth_resolution_deg
    assert first == pytest.approx(g.azimuth_start_deg + .5*g.output_azimuth_resolution_deg)


def test_probability_clipping_precedes_filter_and_does_not_cap_power():
    model = RadarSplatModel(torch.tensor([[10., 0., 0.]]).repeat(4, 1),
        initial_scale=.15, initial_opacity=.9, initial_noise_probability=.2,
        initial_reflectance=torch.full((4, 3), .9), sh_degree=0)
    effects = replace(RadarSplatEffects.b787_clean(), probability_ceiling=1.,
                      use_azimuth_antenna_gain=False, use_spectral_leakage=False)
    result = model.render(torch.eye(4), grid(), effects)
    assert result["occupancy"].shape == result["final_power"].shape == (1, 16, 16)
    assert result["occupancy"].max() <= 1
    assert result["final_power"].max() > 1
    assert result["noise_probability"].max() <= 1
    legacy = model.render(torch.eye(4), grid(), replace(effects, probability_ceiling=None))
    assert legacy["occupancy"].max() > 1


def test_cutoff_is_per_product_and_explicit_ablation_restores_gradient():
    g = grid(1)
    means = torch.tensor([[[10., 0.]]], dtype=torch.float64)
    covariance = torch.diag(torch.tensor([.01, .0001], dtype=torch.float64))[None, None]
    attributes = torch.tensor([[[.25, .001]]], dtype=torch.float64, requires_grad=True)
    out = additive_gaussian_rasterization(means, covariance, attributes, g)
    assert out[..., 0].sum() > 0 and out[..., 1].sum() == 0
    out[..., 1].sum().backward()
    assert attributes.grad[0, 0, 1] == 0
    attributes.grad = None
    out = additive_gaussian_rasterization(means, covariance, attributes, g, alpha_cutoff=0.)
    out[..., 1].sum().backward()
    assert out[..., 1].sum() > 0 and attributes.grad[0, 0, 1] > 0


def test_release_ssim_against_independent_scipy_moments_and_gradient():
    from scipy.signal import convolve2d
    rng = np.random.default_rng(10)
    x, y = rng.random((2, 15, 16))*2
    t = np.arange(-5, 6)
    g = np.exp(-t*t/(2*1.5**2)); g /= g.sum()
    w = np.outer(g, g)
    conv = lambda a: convolve2d(a, w, mode="valid")
    def reference(a):
        mx, my = conv(a), conv(y)
        vx, vy, cov = conv(a*a)-mx*mx, conv(y*y)-my*my, conv(a*y)-mx*my
        return np.mean((2*mx*my+.0001)*(2*cov+.0009)/((mx*mx+my*my+.0001)*(vx+vy+.0009)))
    tx = torch.tensor(x, requires_grad=True)
    result = release_ssim_index(tx, torch.tensor(y))
    assert result.item() == pytest.approx(reference(x), abs=1e-12)
    result.backward()
    direction = rng.normal(size=x.shape)
    eps = 1e-5
    finite_difference = (reference(x+eps*direction)-reference(x-eps*direction))/(2*eps)
    assert (tx.grad*torch.tensor(direction)).sum().item() == pytest.approx(finite_difference, abs=1e-8)
    assert release_ssim_index(tx, tx).item() == pytest.approx(1.)
    with pytest.raises(ValueError, match="11 pixels"):
        release_ssim_index(torch.ones(8, 8), torch.ones(8, 8))


def test_separate_occupancy_and_image_mean_loss():
    model = RadarSplatModel(torch.tensor([[10., 0., 0.]]), sh_degree=0)
    power = torch.zeros(12, 12, requires_grad=True)
    occupancy = torch.full((12, 12), .2, requires_grad=True)
    labels = torch.zeros(12, 12); labels[0, 0] = 1
    args = dict(max_scale=1., weights=ObjectiveWeights(ssim=0., occupancy=1., max_size=0., opacity_noise=0.))
    losses = native_power_objective({"final_power": power, "occupancy": occupancy},
        torch.zeros_like(power), model, target_occupancy=labels, fidelity_profile="audit_v1", **args)
    assert losses["occupancy_l1"].item() == pytest.approx((.8+143*.2)/144)
    losses["total"].backward()
    assert occupancy.grad[0, 0] < 0 and occupancy.grad[1, 1] > 0
    with pytest.raises(ValueError, match="independently"):
        native_power_objective({"final_power": power, "occupancy": occupancy},
            torch.zeros_like(power), model, fidelity_profile="audit_v1", **args)


def test_noise_regularizer_has_no_zero_seeking_gradient_below_one():
    model = RadarSplatModel(torch.tensor([[10., 0., 0.]]), initial_opacity=.1,
                            initial_noise_probability=.5, sh_degree=0)
    penalty = model.native_regularizers(1.)["opacity_noise"]
    penalty.backward()
    assert penalty == 0 and model.noise_probability_logits.grad.abs().sum() == 0


def test_denoiser_matches_official_decay_helper_when_reference_available():
    # Source is optional and read-only; never import the upstream GUI/data loader.
    source = Path("/tmp/rs_up_signal.py")
    if not source.exists():
        pytest.skip("optional official source reference not downloaded")
    tree = ast.parse(source.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and
                n.name in {"find_decay_region", "apply_saturation_mask"}]
    ns = {"np": np, "gaussian_filter1d": gaussian_filter1d}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), ns)
    rng = np.random.default_rng(31)
    power = .5 + rng.random((8, 128))*.2
    power[:, 40:50] += 1
    recipe = OccupancyRecipe()
    actual, noisy = denoise_power(power, recipe)
    expected, _ = ns["apply_saturation_mask"](power, noisy, recipe.smoothing_sigma_bins)
    np.testing.assert_array_equal(actual, expected)
    assert np.any(noisy)


def arrays(angle=0.):
    c, s = np.cos(angle), np.sin(angle)
    pose = np.eye(4); pose[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return dict(sensor_to_world=pose, range_m=np.linspace(9., 11., 12),
                azimuth_rad=np.linspace(-.1, .1, 12), elevation_rad=np.array([-.02, .02]),
                radarsplat_mf_power=np.ones((12, 12)))


def test_reprojection_identity_and_visibility():
    a = arrays(.4)
    power = np.arange(144.).reshape(12, 12)
    sampled, valid = sample_polar_power(polar_world_points(a), a, power)
    assert valid.all()
    np.testing.assert_allclose(sampled.reshape(2, 12, 12), np.stack((power, power)), atol=1e-11)
    sampled, valid = sample_polar_power([[0, 100, 50]], a, power)
    assert not valid.any() and sampled.sum() == 0


def test_occupancy_uses_only_training_donors_and_spatial_order(monkeypatch):
    from rift import radarsplat_b7873200_protocol as protocol
    cache = SimpleNamespace(train_indices=(3, 1, 2), train_peak_power=1., root=Path("unused"),
        grid={"scene_center_m": [0, 0, 0], "scene_extent_m": 20}, acquisition_record={"view_indices": [1, 2, 3, 99],
        "viewpoint_positions": [[10, 0, 0], [9.99, .01, 0], [-10, 0, 0], [10, 0, 0]]})
    reads = []
    def load(root, index, role, **kwargs):
        assert index != 99 and role == "train"
        reads.append(index)
        return arrays()
    monkeypatch.setattr(protocol, "load_target", load)
    provider = TrainingOccupancy(cache, OccupancyRecipe(window_views=2))
    mask, report = provider.label(1)
    assert report["donor_view_ids"] == [1, 2]
    assert mask.all() and set(reads) == {1, 2}
    with pytest.raises(ValueError, match="training"):
        provider.label(99)
    with pytest.raises(ValueError, match="training"):
        provider._donor(99)


def test_profile_identity_preserves_legacy_and_separates_corrections(tmp_path):
    import train_radarsplat as trainer
    from scripts.validate_radarsplat_b7873200_native_contract import _make_synthetic_cache
    from rift.radarsplat_b7873200_protocol import load_cache
    _make_synthetic_cache(tmp_path)
    cache = load_cache(tmp_path)
    args = trainer.parse_args(["--cache-root", str(tmp_path), "--checkpoint-dir", str(tmp_path/"run")])
    old = trainer.run_identity(args, cache, trainer.effects_for_args(args))
    assert "probability_ceiling" not in old["renderer"]["effects"]
    assert "fidelity_profile" not in old and old["model"]["planar_initialization"]
    args.fidelity_profile = "audit_v1"
    new = trainer.run_identity(args, cache, trainer.effects_for_args(args))
    assert new != old and not new["model"]["planar_initialization"]
    assert new["renderer"]["effects"]["probability_ceiling"] == 1
    assert new["objective"]["occupancy_target"]["window_views"] == 10
    args.map_power_threshold = .2
    assert trainer.run_identity(args, cache, trainer.effects_for_args(args)) != new


def test_corrected_actual_trainer_resume_and_matching_readout(tmp_path, monkeypatch):
    import signal
    import train_radarsplat as trainer
    from scripts.validate_radarsplat_b7873200_native_contract import _make_synthetic_cache
    from scripts.readout_radarsplat_checkpoint import readout
    from rift.radarsplat_b7873200_protocol import load_cache
    cache_root = tmp_path / "cache"
    _make_synthetic_cache(cache_root)
    base = ["--cache-root", str(cache_root), "--device", "cpu", "--steps", "2",
            "--validation-every", "1", "--checkpoint-every", "1", "--init-num-gaussians", "2",
            "--init-scale-m", ".06", "--max-scale-m", ".2", "--prune-every", "1",
            "--allow-development-subset", "--fidelity-profile", "audit_v1"]
    original = trainer.native_power_objective
    def interrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        trainer._request_stop(signal.SIGTERM, None)
        return result
    resumed_root, full_root = tmp_path/"resumed", tmp_path/"full"
    with monkeypatch.context() as m:
        m.setattr(trainer, "native_power_objective", interrupt)
        with pytest.raises(SystemExit) as exc:
            trainer.main(base + ["--checkpoint-dir", str(resumed_root)])
        assert exc.value.code == 143
    trainer.main(base + ["--checkpoint-dir", str(resumed_root)])
    trainer.main(base + ["--checkpoint-dir", str(full_root)])
    resumed = trainer._load_checkpoint(resumed_root/"checkpoint_final.pt", torch.device("cpu"))
    full = trainer._load_checkpoint(full_root/"checkpoint_final.pt", torch.device("cpu"))
    for key in ("model_state_dict", "optimizer_state_dicts", "sampler_state"):
        assert trainer._directly_equal(resumed[key], full[key])
    assert resumed["last_train"]["occupancy_balance_mode"] == "image_mean"
    assert resumed["last_train"]["occupancy_map"]["donor_view_ids"] == list(load_cache(cache_root).train_indices)
    report = readout(checkpoint_path=resumed_root/"checkpoint_best.pt", cache_root=cache_root,
                     geometry_path=tmp_path/"geometry.npz")
    assert report["step"] >= 1 and report["metrics"]["views"] == 1
    assert (tmp_path/"geometry.npz").exists()
    def forbid_pixels(*args, **kwargs):
        raise AssertionError("mismatched resume/readout must fail before loading target pixels")
    with monkeypatch.context() as m:
        m.setattr(trainer, "load_cache", forbid_pixels)
        with pytest.raises(ValueError, match="scientific run"):
            trainer.main([x if x != "audit_v1" else "legacy" for x in base] +
                         ["--checkpoint-dir", str(resumed_root)])
        # A valid checkpoint pointed at another object's recipe must also fail
        # at the metadata gate, independently of profile compatibility.
        import json
        from rift.radarsplat_b7873200_protocol import RECIPE_FILENAME
        other_root = tmp_path/"other_object"
        other_root.mkdir()
        recipe = copy.deepcopy(resumed["run_identity"]["target_recipe"])
        recipe["sealed_protocol_identity"]["object_id"] = "wrong-object"
        (other_root/RECIPE_FILENAME).write_text(json.dumps(recipe))
        other_args = [str(other_root) if x == str(cache_root) else x for x in base]
        with pytest.raises(ValueError, match="scientific run"):
            trainer.main(other_args + ["--checkpoint-dir", str(resumed_root)])
        import scripts.readout_radarsplat_checkpoint as reader
        m.setattr(reader, "load_cache", forbid_pixels)
        with pytest.raises(ValueError, match="identities differ"):
            readout(checkpoint_path=resumed_root/"checkpoint_best.pt", cache_root=other_root)


def test_collection_corrected_and_legacy_commands(tmp_path):
    from rift.radarsplat_collection import commands_for
    options = dict(dataset_root=tmp_path, output_root=tmp_path/"runs")
    corrected = commands_for("a320", recipe="audit_v1", **options)
    legacy = commands_for("a320", recipe="legacy", **options)
    assert "scene_support" in corrected[0] and "audit_v1" in corrected[1]
    assert "--grid-policy" not in legacy[0] and "legacy" in legacy[1]


def test_scene_support_grid_covers_cube_and_keeps_central_sample():
    from rift.radarsplat_fidelity import scene_support_angular_sampling
    az, el, intermediate = scene_support_angular_sampling([[10., 0, 0], [0, 12., 0]],
        half_extent_m=.15, n_azimuth=33, n_elevation=33)
    assert az == el and az/intermediate == pytest.approx(10.)
    assert az < .1
    assert 16*az == pytest.approx(np.rad2deg(np.arcsin(np.sqrt(3)*.15/10)))
    with pytest.raises(ValueError, match="outside"):
        scene_support_angular_sampling([[0., 0, 0]], half_extent_m=.15, n_azimuth=33, n_elevation=33)


def test_corrected_grid_float32_calibration_and_render():
    from rift.radarsplat_fidelity import scene_support_angular_sampling
    from rift.radarsplat_b7873200_adapter import target_grid_from_arrays
    from rift.radarsplat_b7873200_acquisition import _expected_target_geometry
    from scripts.validate_radarsplat_b7873200_native_contract import _fixture_record, _fixture_target_spec
    record = _fixture_record((1, 2))
    spec = _fixture_target_spec()["grid"]
    az, el, intermediate = scene_support_angular_sampling(record["viewpoint_positions"],
        half_extent_m=.15, n_azimuth=33, n_elevation=33)
    spec.update(n_azimuth=33, n_elevation=33, n_range=33,
                output_azimuth_resolution_deg=az, elevation_sampling_resolution_deg=el,
                intermediate_azimuth_resolution_deg=intermediate)
    pose, ranges, azimuths, elevations = _expected_target_geometry(record, 1, spec)
    g = target_grid_from_arrays(range_m=ranges.astype(np.float32),
                               azimuth_rad=azimuths.astype(np.float32), expected_grid=spec)
    model = RadarSplatModel(torch.zeros(1, 3), initial_scale=.003, sh_degree=0)
    result = model.render(torch.tensor(pose, dtype=torch.float32), g,
                          replace(RadarSplatEffects.b787_clean(), probability_ceiling=1.))
    assert result["final_power"].shape == (1, 33, 33)
    assert torch.isfinite(result["final_power"]).all() and result["final_power"].sum() > 0
    changed = azimuths.copy()
    changed[7] += 1e-6
    with pytest.raises(ValueError, match="declared physical bin centres"):
        target_grid_from_arrays(range_m=ranges, azimuth_rad=changed, expected_grid=spec)
    # Reprojection must retain every own-view bin, including endpoints; a
    # float32 first-bin spacing extrapolation loses some of these samples.
    donor = dict(sensor_to_world=pose, range_m=ranges, azimuth_rad=azimuths,
                 elevation_rad=elevations)
    power = np.arange(33*33, dtype=np.float64).reshape(33, 33)/(33*33)
    sampled, visible = sample_polar_power(polar_world_points(donor), donor, power)
    assert visible.all()
    np.testing.assert_allclose(sampled.reshape(33, 33, 33),
                               np.broadcast_to(power, (33, 33, 33)), atol=1e-11)


def test_independent_operator_probe_documents_interference_limit():
    from scripts.validate_radarsplat_operator import diagnostics
    report = diagnostics()
    for row in report["single_reflectors"]:
        assert row["complex_mf_relative_l2"] < 1e-6
        assert row["complex_gradient_relative_l2"] < 1e-6
        assert row["best_scalar_power_shape_relative_l2"] > .1
    assert report["coincident_equal_reflectors"] == dict(in_phase_power_ratio=4.,
        opposite_phase_power_ratio=0., additive_independent_power_ratio=2.)


def test_occupancy_mask_changes_with_donors_but_not_validation(monkeypatch):
    from rift import radarsplat_b7873200_protocol as protocol
    cache = SimpleNamespace(train_indices=(1, 2), train_peak_power=1., root=Path("unused"),
        grid={"scene_center_m": [0, 0, 0], "scene_extent_m": 20}, acquisition_record={
        "view_indices": [1, 2, 99], "viewpoint_positions": [[10, 0, 0], [10, .01, 0], [10, 0, 0]]})
    def load(root, index, role, **kwargs):
        assert role == "train" and index in {1, 2}
        a = arrays(); a["radarsplat_mf_power"][:] = 1 if index == 1 else 0
        return a
    monkeypatch.setattr(protocol, "load_target", load)
    mask, _ = TrainingOccupancy(cache, OccupancyRecipe(window_views=2, power_threshold=.75)).label(1)
    assert not mask.any()  # mean is .5; thresholding the current image alone gives 1.
    mask, _ = TrainingOccupancy(cache, OccupancyRecipe(window_views=1, power_threshold=.75)).label(1)
    assert mask.all()
