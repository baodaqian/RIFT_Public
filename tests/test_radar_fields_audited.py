"""Data-free source comparisons and independent physical/numerical RF checks."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import train_radar_fields as trainer
from rift.radar_fields import HashGridEncoder, RadarFieldsModel, radar_fields_intensity
from rift.radar_fields_dataset import response_view_to_range_power, range_bin_centers
from rift.radar_fields_native import (occupancy_probability, weighted_ray_mean,
    scene_cap_directions, bistatic_ray_points, render_bistatic_bins, released_batch_loss)
from rift.radar_fields_recipe import recipe_contract, validate_recipe_checkpoint
from rift.radar_fields_upstream import original_module, verify_sources


def tiny_model(bn=False):
    return RadarFieldsModel(extent=0.15, hidden_dim=8, feature_dim=4, batch_norm=bn,
        hash_levels=2, hash_base_resolution=4, hash_final_resolution=8,
        hash_log2_size=8, encoding_layout="tcnn")


def args_v2(*extra):
    return trainer.parse_args(["--npz-path", "synthetic.npz", "--recipe", "audited-v2",
                               "--model-backend", "torch", *extra])


def test_effective_offset_matches_official_training_not_helper_default():
    from rift.radar_fields_upstream import REFERENCE_ROOT
    source = (REFERENCE_ROOT / "parse.py").read_text()
    assert '"--initial_offset", type=float, default=1.0' in source
    args = args_v2()
    zero = torch.zeros(3)
    reference = original_module("radarfields.radar").rcs_to_intensity(zero, torch.ones(3), 1.0, 1.0, True)
    assert torch.equal(radar_fields_intensity(zero, torch.ones(3), args.intensity_offset), reference)
    assert torch.equal(reference, zero)
    assert trainer.parse_args(["--npz-path", "synthetic.npz"]).intensity_offset == 0.05


def test_required_upstream_source_files_are_available():
    verify_sources()


def test_occupancy_matches_original_helpers_and_portable_math():
    rng = torch.Generator().manual_seed(8)
    intensity = torch.rand(9, 31, generator=rng) * 0.15
    intensity[1, 5] = 0.8
    intensity[7, 10] = 0.65
    upstream = original_module("radarfields.radar")
    filtered = upstream.compute_spherical_grid_noise_threshold(intensity, 0, 30)
    expected, _ = upstream.bayesian_polar_occupancy_map(filtered, (1, 31))
    actual = occupancy_probability(intensity, noise_axis="azimuth_range")
    portable = occupancy_probability(intensity, noise_axis="azimuth_range", implementation="torch")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(portable, expected, rtol=2e-6, atol=1e-7)
    assert actual[1, 6] > 0  # occlusion evidence beyond a detected return
    assert actual.min() >= 0 and actual.max() < 1


def test_pairs_are_not_misidentified_as_azimuth_beams():
    profiles = torch.zeros(4, 24)
    profiles[:, 5] = 0.8  # all MIMO pairs observe the same real target
    assert not occupancy_probability(profiles, noise_axis="azimuth_range").any()
    target = occupancy_probability(profiles, noise_axis="range")
    assert torch.all(target[:, 5] > 0.99)
    torch.testing.assert_close(target[[3, 1]], occupancy_probability(profiles[[3, 1]]), rtol=0, atol=0)
    assert torch.count_nonzero(occupancy_probability(torch.zeros_like(profiles))) == 0


def test_loss_matches_original_batchmean_and_sample_std():
    alpha = torch.tensor([[0.1, 0.3, 0.9], [0.6, 0.2, 0.8]], requires_grad=True)
    occ = torch.tensor([[0., 0.2, 0.9], [0., 0., 0.8]])
    prediction = alpha * 0.7
    records = [{"prediction": prediction[i], "target": occ[i], "occupancy": alpha[i],
                "occupancy_target": occ[i]} for i in range(2)]
    loss, terms = released_batch_loss(records)
    expected_kl = torch.nn.KLDivLoss(reduction="batchmean")((alpha / alpha.sum()).log(), occ / occ.sum())
    torch.testing.assert_close(terms["occupancy"], expected_kl * .36)
    mask = occ > .01
    torch.testing.assert_close(terms["bimodal"], (alpha[mask].std() + alpha[~mask].std()) * .03)
    loss.backward()
    assert alpha.grad is not None and torch.isfinite(alpha.grad).all()


@pytest.mark.parametrize("targets", [[0.0], [0.8], [0., 0., 0.], [0., .8]])
def test_empty_singleton_occupancy_groups_have_finite_gradients(targets):
    alpha = torch.full((len(targets),), .5, requires_grad=True)
    record = {"prediction": alpha, "target": torch.zeros_like(alpha), "occupancy": alpha,
              "occupancy_target": torch.tensor(targets)}
    loss, _ = released_batch_loss([record])
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(alpha.grad).all()


def test_dense_hash_level_preserves_linear_field_and_derivative():
    encoder = HashGridEncoder(n_levels=1, n_features_per_level=1, base_resolution=4,
                              final_resolution=4, log2_hashmap_size=8, layout="tcnn").double()
    assert encoder.tables[0].num_embeddings == 64
    ids = torch.arange(64)
    with torch.no_grad():
        encoder.tables[0].weight[:, 0] = ids % 4 + 2 * ((ids // 4) % 4) + 3 * (ids // 16)
    x = torch.tensor([[.21, .37, .54]], dtype=torch.float64, requires_grad=True)
    value = encoder(x)
    expected = ((x * 3 + .5) * torch.tensor([1., 2., 3.])).sum()
    torch.testing.assert_close(value.squeeze(), expected)
    value.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([[3., 6., 9.]], dtype=torch.float64))
    with pytest.raises(ValueError, match="unit cube"):
        encoder(torch.tensor([[1.01, .5, .5]], dtype=torch.float64))


def test_hashed_level_uses_unsigned_coherentprime_index_at_vertices():
    encoder = HashGridEncoder(n_levels=1, n_features_per_level=1, base_resolution=8,
                              final_resolution=8, log2_hashmap_size=5, layout="tcnn").double()
    with torch.no_grad():
        encoder.tables[0].weight[:, 0] = torch.arange(32)
    vertices = [(1, 2, 3), (6, 5, 4)]
    x = (torch.tensor(vertices, dtype=torch.float64) - .5) / 7
    expected = [(i ^ (j * 2654435761) ^ (k * 805459861)) & 31 for i, j, k in vertices]
    torch.testing.assert_close(encoder(x).flatten(), torch.tensor(expected, dtype=torch.float64))


def test_chunk_size_does_not_change_training_bn_or_eval_readout():
    torch.manual_seed(9)
    first = tiny_model(bn=True)
    second = deepcopy(first)
    xyz = torch.rand(19, 3) * .2 - .1
    direction = torch.tensor([.3, .5, .8])
    a = first.query_chunked(xyz, direction, chunk_size=4)
    b = second.query_chunked(xyz, direction, chunk_size=19)
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(first.feature_norm.running_mean, second.feature_norm.running_mean)
    assert first.feature_norm.num_batches_tracked == 1
    first.eval()
    perm = torch.randperm(len(xyz))
    a = first.query_chunked(xyz, direction, chunk_size=1)
    b = first.query_chunked(xyz[perm], direction, chunk_size=7)
    for key in a:
        torch.testing.assert_close(a[key][perm], b[key])


def test_ray_points_hit_exact_bistatic_surfaces_asymmetric_pairs():
    tx = torch.tensor([[2., 1., 8.], [4., -1., 8.]], dtype=torch.float64)
    rx = torch.tensor([[2.1, .8, 8.], [3.7, -.9, 8.1]], dtype=torch.float64)
    directions = scene_cap_directions(tx, .15, 17)
    ranges = .5 * (tx.norm(dim=-1) + rx.norm(dim=-1))[:, None] + torch.tensor([[-.1, 0., .1]])
    points = bistatic_ray_points(tx, rx, directions, ranges)
    measured = .5 * ((points-tx[:, None, None]).norm(dim=-1) + (points-rx[:, None, None]).norm(dim=-1))
    torch.testing.assert_close(measured, ranges[:, :, None].expand_as(measured), atol=3e-15, rtol=1e-15)


def test_weighted_integration_matches_released_lut_rule_and_empty_space_mass():
    reference = original_module("radarfields.radar")
    samples = torch.tensor([[[[.3], [.5]], [[.8], [.1]], [[0.], [0.]]]])
    offsets = torch.tensor([[[0., 0., -1.], [0., 0., 0.], [0., 0., 1.]]])
    azimuth = np.array([[-1., .25], [0., 1.], [1., .25]])
    elevation = np.array([[-1., 1.], [1., 1.]])
    expected = reference.integrate_rays_LUT(samples, offsets, 3, azimuth, elevation, "cpu")
    values = samples[0, ..., 0].T
    actual = weighted_ray_mean(values, torch.tensor([.25, 1., .25]))
    torch.testing.assert_close(actual, expected.flatten())
    assert actual[0] < (.3*.25 + .8) / 1.25  # zero exterior still in denominator


def test_renderer_gradients_empty_bins_and_geometric_readout_resolution_independence():
    model = tiny_model().double().eval()
    tx = torch.tensor([[0., 0., 10.]], dtype=torch.float64)
    ranges = torch.tensor([9., 9.9, 10., 10.1, 11.], dtype=torch.float64)
    out = render_bistatic_bins(model, tx, tx, ranges, extent=.15, ray_samples=64, query_chunk=32)
    assert torch.equal(out["rcs"][:, [0, 4]], torch.zeros(1, 2, dtype=torch.float64))
    assert (out["coverage"][:, 1:4] > 0).all()
    out["rcs"].sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_frontend_finite_band_psf_and_coherent_interference_are_explicit():
    n = 64
    f = np.arange(n)
    for position in (17., 17.35):
        response = np.exp(-2j*np.pi*f*position/n)
        actual = response_view_to_range_power(response.astype(np.complex64).reshape(1, 1, 1, n))
        bins = np.arange(n)
        direct = np.abs(np.exp(2j*np.pi*np.outer(bins-position, f)/n).mean(axis=1))**2
        np.testing.assert_allclose(actual.numpy()[0], direct, atol=1e-7, rtol=3e-5)
        assert actual.argmax().item() == 17
    response = np.exp(-2j*np.pi*f*17.35/n).astype(np.complex64)
    a = response_view_to_range_power(response.reshape(1, 1, 1, n))
    cancelled = response_view_to_range_power((response-response).reshape(1, 1, 1, n))
    constructive = response_view_to_range_power((response+response).reshape(1, 1, 1, n))
    assert cancelled.sum() == 0 and (2*a).sum() > 0
    torch.testing.assert_close(constructive, 4*a)
    # Noncoherent RF receives identical per-scatterer powers in both cases.
    # No antenna average can recover this phase-dependent cross-term.


def test_recipe_rejection_precedes_any_dataset_read(monkeypatch, tmp_path):
    args = args_v2()
    checkpoint = {"args": vars(args), "radar_fields_recipe": recipe_contract(args)}
    validate_recipe_checkpoint(checkpoint, args)
    changed = deepcopy(args)
    changed.ray_samples += 1
    with pytest.raises(ValueError, match="configuration mismatch"):
        validate_recipe_checkpoint(checkpoint, changed)
    with pytest.raises(ValueError, match="recipe mismatch"):
        validate_recipe_checkpoint({"args": {}}, args)
    path = tmp_path / "legacy.pt"
    torch.save({"args": {}}, path)
    monkeypatch.setattr(sys, "argv", ["train_radar_fields.py", "--npz-path", "unread.npz",
        "--recipe", "audited-v2", "--model-backend", "torch", "--resume", str(path),
        "--sealed-protocol", "--sealed-split-manifest", "unread.json", "--num-test", "1"])
    monkeypatch.setattr(trainer, "load_radar_fields_npz", lambda *a, **k: pytest.fail("response loader reached"))
    with pytest.raises(ValueError, match="recipe mismatch"):
        trainer.main()


def test_view_coverage_resume_is_exact_and_visits_all_training_ids():
    rng = np.random.default_rng(9)
    sampler = trainer.CoveredViewSampler([5, 1, 9, 2])
    first = sampler.next(3, rng)
    state, rng_state = deepcopy(sampler.state_dict()), deepcopy(rng.bit_generator.state)
    expected = sampler.next(9, rng)
    restored = trainer.CoveredViewSampler([5, 1, 9, 2], state)
    rng.bit_generator.state = rng_state
    assert restored.next(9, rng) == expected
    assert set(first + expected[:1]) == {5, 1, 9, 2}
    assert set(restored.counts.values()) == {3}


def _fixture(root):
    from scripts.validate_radar_fields import _write_sealed_entrypoint_fixture
    path, manifest, _ = _write_sealed_entrypoint_fixture(root)
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    # A distant single sensor and 64 frequency bins; no benchmark responses.
    arrays["viewpoint_positions"][:, 2] = 1.0
    arrays["tx_pos"][:, 0, 2] = 1.0
    arrays["rx_pos"][:, 0, 2] = 1.0
    phase = np.exp(-2j*np.pi*np.arange(64)*20/64).astype(np.complex64)
    arrays["response"] = np.tile(phase, (5, 1, 1, 1, 1))
    metadata = json.loads(arrays["metadata_json"].item())
    metadata["num_adc_samples"] = 64
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez(path, **arrays)
    document = json.loads(manifest.read_text())
    document["dataset"]["response_shape"] = [5, 1, 1, 1, 64]
    manifest.write_text(json.dumps(document))
    return path, manifest


def test_tiny_entrypoint_sealed_resume_matches_uninterrupted(tmp_path, monkeypatch):
    import rift.radar_fields_dataset as ds
    path, manifest = _fixture(tmp_path)
    old_read = ds._read_response_view_from_npz
    visits = []
    def guarded(path, index, *args, **kwargs):
        assert index in (0, 1, 2)
        visits.append(index)
        return old_read(path, index, *args, **kwargs)
    monkeypatch.setattr(ds, "_read_response_view_from_npz", guarded)
    common = ["--npz-path", str(path), "--sealed-protocol", "--sealed-split-manifest", str(manifest),
              "--num-train", "2", "--num-val", "1", "--num-test", "1", "--steps", "3",
              "--view-batch", "2", "--eval-every", "3", "--checkpoint-every", "1",
              "--device", "cpu", "--recipe", "audited-v2", "--model-backend", "torch",
              "--granularity", "3", "--extent", ".15", "--ray-samples", "8",
              "--hidden-dim", "8", "--feature-dim", "4", "--hash-levels", "2",
              "--hash-base-resolution", "4", "--hash-final-resolution", "8", "--hash-log2-size", "8",
              "--checkpoint-root", str(tmp_path), "--query-chunk", "8"]
    def run(name, stop=False, resume=None):
        argv = common + ["--checkpoint-name", name]
        if resume:
            argv += ["--resume", str(resume)]
        monkeypatch.setattr(sys, "argv", ["train_radar_fields.py", *argv])
        monkeypatch.setattr(trainer, "STOP_REQUESTED", stop)
        trainer.main()
    run("whole")
    run("resumed", stop=True)
    run("resumed", resume=tmp_path / "resumed/checkpoint_latest.pth.tar")
    a = torch.load(tmp_path / "whole/checkpoint_final.pth.tar", weights_only=False)
    b = torch.load(tmp_path / "resumed/checkpoint_final.pth.tar", weights_only=False)
    for key, value in a["radar_fields_state_dict"].items():
        torch.testing.assert_close(value, b["radar_fields_state_dict"][key], rtol=0, atol=0)
    def same(left, right):
        if torch.is_tensor(left):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                same(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for x, y in zip(left, right):
                same(x, y)
        else:
            assert left == right
    for key in ("optimizer_state_dict", "scheduler_state_dict", "torch_rng_state", "numpy_rng_state_json"):
        same(a[key], b[key])
    assert a["training_view_coverage"] == b["training_view_coverage"]
    assert set(a["training_view_coverage"]["counts"].values()) == {3}
    assert a["split_provenance"]["test_payload_materialized"] is False
    assert set(visits) == {0, 1, 2}
    # Corrupted continuation metadata must fail before scanning responses even
    # when the caller supplies a fresh output directory without a stats cache.
    monkeypatch.setattr(trainer, "load_or_create_stats", lambda *a, **k: pytest.fail("normalization scan reached"))
    for key in ("training_view_coverage", "optimizer_state_dict"):
        damaged = deepcopy(a)
        del damaged[key]
        bad = tmp_path / f"missing_{key}.pt"
        torch.save(damaged, bad)
        with pytest.raises(ValueError, match="lacks"):
            run("unscanned", resume=bad)


def test_native_readout_uses_identical_renderer(tmp_path):
    from scripts.readout_radar_fields_b7873200_native import evaluate_view
    from rift.radar_fields_dataset import load_radar_fields_npz
    path, _ = _fixture(tmp_path)
    arrays = load_radar_fields_npz(str(path), load_response=False)
    args = args_v2("--device", "cpu", "--extent", ".15", "--ray-samples", "8")
    model = tiny_model().eval()
    ranges = range_bin_centers(arrays.metadata, dtype=torch.float64)
    stats = {"peak_power": 1., "dynamic_range_db": 60.}
    native = trainer.audited_view_tensors(model, arrays, 0, np.array([0]), ranges, stats, args, 1., torch.device("cpu"))
    result = evaluate_view(model, arrays, 0, xyz=torch.zeros(1, 3), ranges=ranges,
                           stats=stats, args=args, device=torch.device("cpu"))
    expected = (native["prediction"]-native["target"]).square().sum().item()
    assert result["regions"]["whole_roi"]["prediction"]["sq_error"] == expected
    assert result["regions"]["whole_roi"]["zero_reference"]["sq_error"] == native["target"].square().sum().item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires allocated CUDA GPU and installed tinycudann")
@pytest.mark.parametrize("recipe", ["audited-v2", "source-adapted-v3"])
def test_original_cuda_model_forward_gradient_and_readout_parity(recipe):
    from rift.radar_fields_upstream import OriginalRadarFieldsModel
    args = trainer.parse_args(["--recipe", recipe, "--npz-path", "unused.npz", "--device", "cuda"])
    model = OriginalRadarFieldsModel(args).cuda().eval()
    xyz = (torch.rand(257, 3, device="cuda")-.5) * args.extent
    direction = F.normalize(torch.randn_like(xyz), dim=-1)
    direct = model.original((xyz+args.extent)/(2*args.extent),
                            F.normalize(direction, dim=-1) if recipe == "audited-v2" else direction, sin_epoch=.8)
    wrapped = model(xyz, direction, mask_progress=.8)
    for key, original_key in (("alpha", "alpha"), ("reflectance", "rd")):
        expected = direct[original_key].flatten()
        if recipe == "audited-v2":
            expected = expected.float()
        torch.testing.assert_close(wrapped[key], expected, rtol=0, atol=0)
    wrapped["rcs"].square().mean().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().max() > 0 for g in gradients)
    chunked = model.query_chunked(xyz, direction, mask_progress=.8, chunk_size=53)
    for key in wrapped:
        torch.testing.assert_close(chunked[key], wrapped[key], rtol=5e-3, atol=5e-4)


def test_collection_dispatch_binds_new_recipe_and_explicit_legacy(tmp_path):
    from train_rift_dataset import commands_for
    from scripts.readout_radar_fields_b7873200_native import CONFIG_ARG_KEYS
    kwargs = dict(dataset_root=tmp_path, output_root=tmp_path)
    for recipe in ("audited-v2", "legacy-v1"):
        command = commands_for("a320", "radar_fields", radar_fields_recipe=recipe, **kwargs)[0]
        assert command[command.index("--recipe")+1] == recipe
        if recipe == "audited-v2":
            args = trainer.parse_args(command[2:])
            protocol = Path(__file__).resolve().parents[1] / "protocols/radar_fields_rift_audited_v2.json"
            training = json.loads(protocol.read_text())["training"]
            for key in (*CONFIG_ARG_KEYS, "checkpoint_name", "recipe", "model_backend", "ray_samples"):
                assert getattr(args, key) == training[key], key


@pytest.mark.parametrize("num_train", [2400, 3200])
def test_native_readout_accepts_validation_selected_audited_snapshot(tmp_path, num_train):
    from types import SimpleNamespace
    from scripts.readout_radar_fields_b7873200_native import validate_checkpoint
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "protocols/radar_fields_rift_audited_v2.json").read_text())
    config["training"]["num_train"] = num_train
    args = {**config["training"], "npz_path": str(tmp_path / "object.npz"),
            "sealed_split_manifest": str(tmp_path / "roles.json")}
    roles = {"train": list(range(num_train)), "val": list(range(8000, 9000)), "test": list(range(9000, 10000))}
    checkpoint = {"scene_repr": "radar_fields", "step": 2000, "args": args,
                  "radar_fields_recipe": recipe_contract(SimpleNamespace(**args)),
                  "split_provenance": {"role_ids": roles, "test_payload_materialized": False},
                  "power_stats": {"peak_power": 1., "train_view_indices": roles["train"],
                                  "normalization_scan_view_indices": roles["train"]},
                  "radar_fields_state_dict": {},
                  "sealed_protocol_contract": {"test_response_materialized": False}}
    inputs = dict(npz_path=args["npz_path"], manifest_path=args["sealed_split_manifest"])
    validate_checkpoint(checkpoint, config, **inputs)
    checkpoint["step"] = 8001
    with pytest.raises(ValueError, match="outside"):
        validate_checkpoint(checkpoint, config, **inputs)
