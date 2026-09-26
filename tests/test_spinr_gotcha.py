"""Native SpINR physics, root dispatch and recovery, using synthetic shards."""
import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import train_gotcha_dataset as cli
import train_spinr_style as root
from rift.gotcha_dataset import C, GOTCHADataset, Observation, Region, digest
from rift.spinr_native import NativeKernel, bin_objective, loss_and_field_vjp
from rift import spinr_gotcha_training as runtime
from tests.test_gotcha_dataset import write_shard, tiny_region


@pytest.mark.parametrize("nonuniform", [False, True])
@pytest.mark.parametrize("samples", [32, 37])
def test_native_kernel_preserves_frequencies_reference_product_volume_and_gradients(nonuniform, samples):
    f = 9e9+np.arange(samples, dtype=np.float64)*5e6
    if nonuniform:
        f[1::2] += 128
    region = Region("fixture", "synthetic", (20., -7., .2),
                    ((0., -1., 0.), (1., 0., 0.), (0., 0., 1.)), .03, "test")
    obs = Observation(2, "vv", 5, 17, np.array([40., 3., 4.]), f, 19.,
                      np.exp(.2j)*np.ones(samples), "own_published_source_af_once")
    kernel = NativeKernel(obs, region, point_tile=1)
    assert kernel.affine == (not nonuniform)
    np.testing.assert_array_equal(kernel.f.numpy(), f)
    points = torch.tensor([[.01, -.02, .005], [-.01, .002, -.004]], dtype=torch.float64, requires_grad=True)
    field = torch.tensor([.7, -.2], dtype=torch.float64, requires_grad=True)
    volume = torch.tensor([.0001, .0003], dtype=torch.float64)
    antenna = torch.tensor(region.to_local(obs.position_m), dtype=torch.float64)
    distance = torch.linalg.vector_norm(points-antenna, dim=-1)
    dense = (torch.exp((-4j*math.pi/C)*(distance[:, None]-obs.reference_range_m)*torch.tensor(f)[None])
             *(field*volume*1.7/distance.square())[:, None]).sum(0)
    native = kernel.render(points, field, volume, 1.7, selected=False)
    bins = kernel.render(points, field, volume, 1.7)
    expected = torch.fft.fft(dense, norm="forward")[kernel.bin_ids]
    torch.testing.assert_close(native, dense, rtol=1e-11, atol=1e-17)
    torch.testing.assert_close(bins, expected, rtol=1e-10, atol=1e-17)
    target = torch.tensor(obs.response)*1e-6
    y = torch.fft.fft(target, norm="forward")[kernel.bin_ids]
    expected_loss = bin_objective(expected, y, 1e-12)
    reference = torch.autograd.grad(expected_loss, (points, field), retain_graph=True)
    actual = torch.autograd.grad(bin_objective(bins, y, 1e-12), (points, field))
    for a, b in zip(actual, reference):
        torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-10)
    loss, gradient = loss_and_field_vjp(kernel, points.detach(), field.detach(), volume, 1.7,
                                       target.numpy(), 1e-12)
    assert loss == pytest.approx(float(expected_loss), rel=1e-9)
    torch.testing.assert_close(gradient, reference[1], rtol=1e-9, atol=1e-11)
    if nonuniform:
        replaced = copy.copy(obs)
        object.__setattr__(replaced, "frequencies_hz", f[0]+np.arange(samples)*(f[-1]-f[0])/(samples-1))
        wrong = NativeKernel(replaced, region).render(points.detach(), field.detach(), volume, 1.7, selected=False)
        assert float((wrong-native.detach()).norm()/native.detach().norm()) > 1e-7


@pytest.fixture
def native_dataset(tmp_path):
    for p in (1, 2):
        for pol in ("hh", "vv"):
            write_shard(tmp_path/"New_Transfer"/"shards"/f"pass{p}_{pol}.npz", p, pol, nf=32+p)
    ds = GOTCHADataset(tmp_path, passes=(1, 2), polarizations=("hh", "vv"), region=tiny_region())
    original = ds.viewpoints
    # Explicit engineering subset only in this test instance; the production
    # split/ingress are unchanged and the synthetic checkpoint identity says so.
    roles = {role:[(p,s) for p,s in original(role) if s in ds.split[role][:(2 if role == "train" else 1)]]
             for role in ("train", "validation")}
    ds.viewpoints = lambda role: roles[role]
    ds.contract["synthetic_test_view_subset"] = roles
    ds.contract = json.loads(json.dumps(ds.contract))
    ds.identity = digest(ds.contract)
    return ds


def config():
    return dict(epochs=2, grid_size=2, nodes_per_cell=1, pulse_batch_size=3,
                neural_point_tile=3, renderer_point_tile=3, checkpoint_every=1, validation_every=1)


class TinyField(torch.nn.Module):
    """Four-parameter numerical lifecycle fixture, never a production option."""
    def __init__(self, *, support_m):
        super().__init__()
        self.support = support_m
        self.coefficients = torch.nn.Parameter(torch.tensor([.8, -.3, .4, .2], dtype=torch.float64))

    def forward(self, points):
        return self.coefficients[0]+torch.tanh(points.double() @ self.coefficients[1:]/self.support)


def assert_tree_equal(a, b):
    assert root._checkpoint_tree_equal(a, b)


def test_metadata_plan_counts_every_native_train_pulse_and_never_reads(native_dataset, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("metadata plan read a response")
    for shard in native_dataset.shards.values():
        monkeypatch.setattr(shard, "read", forbidden)
    plan = runtime.PulsePlan(native_dataset)
    report = runtime.preflight(native_dataset, {})
    assert len(plan.records) == 8
    assert report["training_pulses"] == 8 and report["recipe"]["epochs"] == 150
    assert report["recipe"]["optimizer"]["cosine_epochs"] == 150
    assert report["recipe"]["budget_scope"] == "comparison_epoch_count"
    assert runtime._expected_lr(150, 150) == pytest.approx(1e-5)
    assert report["recipe"]["grid_size"] == 48 and report["recipe"]["nodes_per_cell"] == 1
    assert report["integration_points"] == 110592
    explicit = json.loads((Path(__file__).resolve().parents[1]/"protocols/gotcha_spinr_g48_midpoint.json").read_text())
    assert runtime.recipe_from_config(explicit["spinr"]) == report["recipe"]
    assert set(report["native_kernel_modes"].values()) == {"exact_native_point_kernel_DFT"}
    assert report["response_reads"] is False
    recipe = runtime.recipe_from_config(config())
    assert plan.coverage(0, plan.batches(recipe), recipe) == plan.coverage(1, 0, recipe)
    assert plan.coverage(2, 0, recipe)["pulse_exposures"] == 16
    assert all(v == 4 for v in plan.coverage(2, 0, recipe)["pulse_exposures_per_view"])


def test_g96_resume_requires_explicit_grid_before_response_or_model_access(native_dataset, tmp_path, monkeypatch):
    from rift.gotcha_dataset import validate_checkpoint
    saved = dict(schema="rift_gotcha_checkpoint_v1", dataset_contract=native_dataset.contract,
                 dataset_identity=native_dataset.identity,
                 recipe=runtime.recipe_from_config({"grid_size": 96, "nodes_per_cell": 2}))
    path = tmp_path/"checkpoint_latest.pt"
    torch.save(saved, path)
    def forbidden(*args, **kwargs):
        raise AssertionError("grid mismatch reached a model or response")
    monkeypatch.setattr(runtime, "SpinrStyleINR", forbidden)
    for shard in native_dataset.shards.values():
        monkeypatch.setattr(shard, "read", forbidden)
    with pytest.raises(ValueError, match="training recipe changed"):
        runtime.run(native_dataset, tmp_path, {}, device="cpu", resume=path)
    # Explicit G96 still satisfies the unchanged scientific identity gate.
    validate_checkpoint(saved, native_dataset, runtime.recipe_from_config({"grid_size": 96, "nodes_per_cell": 2}))
    with pytest.raises(ValueError, match="training recipe changed"):
        validate_checkpoint(saved, native_dataset, runtime.recipe_from_config({"grid_size": 96}))


def test_multi_pulse_sector_coverage_and_statistics_remain_training_only(tmp_path):
    def two_pulses(arrays, metadata):
        for key, value in list(arrays.items()):
            if key != "frequencies_hz" and len(value) == 360:
                arrays[key] = np.repeat(value, 2, axis=0)
        arrays["pulse_index"] = np.tile(np.arange(2, dtype=np.int32), 360)
        arrays["response"][1::2] *= 2
    write_shard(tmp_path/"New_Transfer/shards/pass1_hh.npz", mutate=two_pulses)
    dataset = GOTCHADataset(tmp_path, passes=(1,), region=tiny_region())
    plan = runtime.PulsePlan(dataset)
    assert len(plan.records) == 500 and np.all(plan.view_counts == 2)
    stats = runtime.training_statistics(plan)
    assert stats["hh"]["pulses"] == 500 and stats["hh"]["samples"] == 500*32
    assert stats["hh"]["mean_power"] == pytest.approx(2.5)
    shard = dataset.shards[1,"hh"]
    assert shard.response_reads == 500
    assert all(shard.row_roles[row] == "train" for _, row in plan.records)


def test_root_hook_dispatches_real_native_runtime_and_preserves_channels(native_dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "SpinrStyleINR", TinyField)
    registry = cli.backend_registry()
    spec = registry["spinr"]
    assert spec["status"] == "available" and spec["default_config"]["epochs"] == 150
    assert spec["default_config"]["grid_size"] == 48
    assert set(spec["polarizations"]) == {"hh", "hv", "vh", "vv"}
    assert "spinr" in cli.resolve_methods(["all"], registry, ["hh", "vv"])
    entry = dict(method="spinr", backend=spec, config=config(), output_dir=str(tmp_path/"fit"), resume=None)
    result = cli.dispatch(native_dataset, entry, "cpu")
    assert result["status"] == "complete" and result["budget_scope"] == "development_only"
    saved = root.load_tensor_checkpoint(tmp_path/"fit/checkpoint_latest.pt", map_location="cpu")
    assert set(saved["model_state_dict"]) == {"hh.coefficients", "vv.coefficients"}
    assert saved["optimization_coverage"]["pulse_exposures"] == 16
    assert saved["optimization_coverage"]["by_polarization"] == {"hh":8, "vv":8}
    assert saved["history"][-1]["validation"]["viewpoints"] == 2
    assert saved["training_statistics"]["hh"]["pulses"] == 4
    assert saved["scheduler_state_dict"]["last_epoch"] == 2
    # Completed resume performs no statistics, validation or response reads.
    before = sum(s.response_reads for s in native_dataset.shards.values())
    entry["resume"] = str(tmp_path/"fit/checkpoint_latest.pt")
    assert cli.dispatch(native_dataset, entry, "cpu")["status"] == "complete"
    assert sum(s.response_reads for s in native_dataset.shards.values()) == before


@pytest.mark.parametrize("stop_after", [1, 3])
@pytest.mark.parametrize("cosine_epochs", [150, 1500])
def test_interrupted_resume_matches_uninterrupted_including_pending_finalization(native_dataset, tmp_path, monkeypatch, stop_after, cosine_epochs):
    monkeypatch.setattr(runtime, "SpinrStyleINR", TinyField)
    trace = []
    update = runtime.batch_update
    def tracked(*args):
        trace.extend(int(i) for i in args[3])
        return update(*args)
    monkeypatch.setattr(runtime, "batch_update", tracked)
    options = dict(config(), cosine_epochs=cosine_epochs)
    runtime.run(native_dataset, tmp_path/"full", options, device="cpu")
    expected = root.load_tensor_checkpoint(tmp_path/"full/checkpoint_latest.pt", map_location="cpu")
    expected_trace = list(trace)
    trace.clear()
    counter = {"n":0}
    def interrupted(*args):
        result = tracked(*args)
        counter["n"] += 1
        return result
    monkeypatch.setattr(runtime, "batch_update", interrupted)
    result = runtime.run(native_dataset, tmp_path/"resume", options, device="cpu", should_stop=lambda:counter["n"] >= stop_after)
    assert result["status"] == "interrupted"
    latest = tmp_path/"resume/checkpoint_latest.pt"
    partial = root.load_tensor_checkpoint(latest, map_location="cpu")
    assert partial["cursor"] == stop_after and partial["epoch"] == 0
    runtime.run(native_dataset, tmp_path/"resume", options, device="cpu", resume=latest)
    actual = root.load_tensor_checkpoint(latest, map_location="cpu")
    assert actual["scheduler_state_dict"]["T_max"] == cosine_epochs
    assert trace == expected_trace
    for key in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "rng_state",
                "history", "optimization_coverage", "head_updates", "initial_scales", "training_statistics"):
        assert_tree_equal(actual[key], expected[key])


def test_historical_schedule_rejected_as_new_recipe_before_response(native_dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "SpinrStyleINR", TinyField)
    options = dict(config(), cosine_epochs=1500)
    output = tmp_path/"fit"
    runtime.run(native_dataset, output, options, device="cpu", should_stop=lambda:True)
    path = output/"checkpoint_latest.pt"
    def forbidden(*args, **kwargs):
        raise AssertionError("changed budget accessed a response or model")
    monkeypatch.setattr(runtime, "SpinrStyleINR", forbidden)
    for shard in native_dataset.shards.values():
        monkeypatch.setattr(shard, "read", forbidden)
    with pytest.raises(ValueError, match="training recipe changed"):
        runtime.run(native_dataset, output, config(), device="cpu", resume=path)
    old = runtime.recipe_from_config(dict(epochs=1500))
    assert old["optimizer"]["cosine_epochs"] == 1500
    assert old["budget_scope"] == "paper_epoch_count"
    for invalid in (True, 300, 150.0):
        with pytest.raises(ValueError, match="cosine_epochs"):
            runtime.recipe_from_config(dict(cosine_epochs=invalid))
    with pytest.raises(ValueError, match="exceed"):
        runtime.recipe_from_config(dict(epochs=1500, cosine_epochs=150))


@pytest.mark.parametrize("corruption", ["identity", "recipe", "coverage", "normalization", "optimizer", "scheduler", "model", "clipping"])
def test_bad_resume_rejected_before_any_response(native_dataset, tmp_path, monkeypatch, corruption):
    monkeypatch.setattr(runtime, "SpinrStyleINR", TinyField)
    output = tmp_path/"fit"
    runtime.run(native_dataset, output, config(), device="cpu", should_stop=lambda:True)
    path = output/"checkpoint_latest.pt"
    saved = root.load_tensor_checkpoint(path, map_location="cpu")
    if corruption == "identity": saved["dataset_identity"] = "wrong"
    if corruption == "recipe": saved["recipe"]["epochs"] = 3
    if corruption == "coverage": saved["optimization_coverage"]["pulse_exposures"] += 1
    if corruption == "normalization": saved["initial_scales"]["hh"]["value"] *= 2
    if corruption == "optimizer": saved["optimizer_state_dict"]["param_groups"][0]["betas"] = (.8, .9)
    if corruption == "scheduler": saved["scheduler_state_dict"]["T_max"] = 300
    if corruption == "model": saved["model_state_dict"]["hh.coefficients"][0] = float("nan")
    if corruption == "clipping": saved["partial_clipped_updates"] = 1
    torch.save(saved, path)
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid checkpoint accessed response")
    for shard in native_dataset.shards.values():
        monkeypatch.setattr(shard, "read", forbidden)
    with pytest.raises(ValueError):
        runtime.run(native_dataset, output, config(), device="cpu", resume=path)


def test_terminal_artifact_recovery_never_reopens_responses(native_dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "SpinrStyleINR", TinyField)
    options = dict(config(), epochs=1)
    output = tmp_path/"fit"
    runtime.run(native_dataset, output, options, device="cpu")
    latest = output/"checkpoint_latest.pt"
    saved = root.load_tensor_checkpoint(latest, map_location="cpu")
    (output/"checkpoint_best.pt").unlink()
    (output/"checkpoint_final.pt").unlink()
    def forbidden(*a, **k):
        raise AssertionError("terminal recovery opened a response")
    for shard in native_dataset.shards.values():
        monkeypatch.setattr(shard, "read", forbidden)
    result = runtime.run(native_dataset, output, options, device="cpu", resume=latest)
    assert result["status"] == "complete"
    for name in ("checkpoint_best.pt", "checkpoint_final.pt"):
        assert_tree_equal(root.load_tensor_checkpoint(output/name, map_location="cpu"), saved)


def test_unknown_config_and_cross_dataset_checkpoints_are_rejected(native_dataset, tmp_path):
    with pytest.raises(ValueError, match="configuration"):
        runtime.preflight(native_dataset, {"invented_option":1})
    path = tmp_path/"checkpoint_latest.pt"
    torch.save({"format":"rift_spinr_style_b787_v2"}, path)
    with pytest.raises(ValueError, match="GOTCHA"):
        root.run_gotcha(dataset=native_dataset, output_dir=tmp_path, config={}, device="cpu", resume=path)


def test_real_root_dry_run_uses_native_hook_without_model_or_response(native_dataset, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "GOTCHADataset", lambda *a, **k:native_dataset)
    def forbidden(*a, **k):
        raise AssertionError("dry run executed a model or response read")
    monkeypatch.setattr(runtime, "SpinrStyleINR", forbidden)
    for shard in native_dataset.shards.values():
        monkeypatch.setattr(shard, "read", forbidden)
    result = cli.main(["--method", "spinr", "--polarizations", "hh", "vv", "--dry-run",
                       "--output-root", str(tmp_path/"unwritten")])
    assert result["plans"][0]["backend"]["module"] == "train_spinr_style"
    assert not (tmp_path/"unwritten").exists()
