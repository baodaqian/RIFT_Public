"""Bounded native-frequency, root-dispatch and recovery checks for RF."""
from copy import deepcopy
from pathlib import Path
import json
import signal

import numpy as np
import pytest
import torch

import train_gotcha_dataset as gotcha_cli
import train_radar_fields as rf_cli
from train_rift_dataset import commands_for
from rift.gotcha_dataset import C, GOTCHADataset, Observation, NativeShardReader
from rift.radar_fields_gotcha import (recipe_from_config, range_geometry, matched_range_power,
    run_gotcha, validate_resume, prepare_frame, training_statistics, model_args)
from rift.radar_fields_recipe import SOURCE_RECIPE
from test_gotcha_dataset import write_shard, tiny_region


def portable_config():
    return dict(profile="audited-v2", model_backend="torch", steps=2, view_batch=1,
                seed=7, ray_samples=8, eval_every=2, checkpoint_every=1,
                hidden_dim=8, feature_dim=4, hash_levels=2,
                hash_base_resolution=4, hash_final_resolution=8, hash_log2_size=8)


def fixture_dataset(tmp_path, pols=("hh",), passes=(1,)):
    for p in passes:
        for i, pol in enumerate(pols):
            write_shard(tmp_path / f"New_Transfer/shards/pass{p}_{pol}.npz", p, pol, nf=32+p+i)
    return GOTCHADataset(tmp_path, passes=passes, polarizations=pols, region=tiny_region())


def test_exact_nonuniform_matched_range_power_and_reference_phase():
    f = np.array([9e9, 9.04e9, 9.13e9, 9.6e9, 10e9])
    r0, point = 19.7, 20.13
    response = 1.3*np.exp(-4j*np.pi/C*f*(point-r0))
    obs = Observation(1, "hh", 1, 0, np.array([20.,0.,0.]), f, r0, response, "fixture")
    r = torch.tensor([19.9, point, 20.4], dtype=torch.float64)
    actual = matched_range_power(obs, r, chunk=1)
    expected = np.abs(np.exp(4j*np.pi/C*(r.numpy()[:,None]-r0)*f) @ response / len(f))**2
    np.testing.assert_allclose(actual.numpy(), expected, atol=2e-12, rtol=2e-11)
    assert actual[1] == pytest.approx(1.3**2, abs=1e-13)
    np.testing.assert_array_equal(obs.frequencies_hz, f)


def test_source_defaults_and_explicit_engineering_profile():
    a = recipe_from_config({}, .15, 3200)
    b = recipe_from_config({}, 5., 2000)
    assert (a["controls"]["steps"], b["controls"]["steps"]) == (960, 800)
    for recipe in (a,b):
        c = recipe["controls"]
        assert (c["profile"], c["view_batch"], c["train_profiles"], c["ray_samples"], c["seed"]) == (SOURCE_RECIPE,10,100,10,0)
        assert recipe["metric_domain"] == "normalized_dB_range_power_intensity"
    with pytest.raises(ValueError, match="fixes"):
        recipe_from_config({"steps":8000}, .15)
    with pytest.raises(ValueError, match="Unknown"):
        recipe_from_config({"untracked_change":1}, .15)
    assert recipe_from_config(portable_config(), .15)["controls"]["steps"] == 2


def test_selected_training_sizes_recompute_schedule_and_readout_recipe(tmp_path):
    from scripts.readout_radar_fields_b7873200_native import CONFIG_ARG_KEYS
    collection = recipe_from_config({}, .15, 2400)["controls"]
    gotcha = recipe_from_config({}, 5., 1500)["controls"]
    assert (collection["steps"], collection["eval_every"], collection["checkpoint_every"]) == (960, 240, 240)
    assert (gotcha["steps"], gotcha["eval_every"], gotcha["checkpoint_every"]) == (900, 150, 150)
    for wrong in ({"steps": 800}, {"eval_every": 200}, {"checkpoint_every": 200}):
        with pytest.raises(ValueError, match="fixes"):
            recipe_from_config(wrong, 5., 1500)
    config = json.loads((Path(__file__).resolve().parents[1]/"protocols/radar_fields_rift_train2400_source_adapted_v3.json").read_text())
    cmd = commands_for("a320", "radar_fields", dataset_root=tmp_path, output_root=tmp_path, num_train=2400)[0]
    args = rf_cli.parse_args(cmd[2:])
    for key in (*CONFIG_ARG_KEYS, "recipe", "ray_samples", "model_backend"):
        assert getattr(args, key) == config["training"][key], key


def test_both_root_routes_select_source_profile_and_old_recipes_remain(tmp_path):
    command = commands_for("a320", "radar_fields", dataset_root=tmp_path, output_root=tmp_path)[0]
    args = rf_cli.parse_args(command[2:])
    assert (args.recipe, args.steps, args.view_batch, args.seed, args.train_pairs) == (SOURCE_RECIPE,960,10,0,100)
    direct = rf_cli.parse_args(["--object", "a320", "--device", "cuda"])
    assert direct.recipe == SOURCE_RECIPE and direct.sealed_protocol
    assert (direct.num_train,direct.num_val,direct.num_test) == (3200,1000,1000)
    registry = gotcha_cli.backend_registry()
    assert registry["radar_fields"]["status"] == "available"
    assert set(registry["radar_fields"]["polarizations"]) == {"hh","hv","vh","vv"}
    assert rf_cli.parse_args(["--npz-path", "historical.npz"]).recipe == "legacy-v1"


def test_gotcha_root_planning_never_reads_responses(tmp_path, monkeypatch):
    fixture_dataset(tmp_path)
    monkeypatch.setattr(NativeShardReader, "read", lambda *_: pytest.fail("planning read response"))
    args = gotcha_cli.parse_args(["--dataset-root",str(tmp_path),"--passes","1",
        "--method","radar_fields","--dry-run","--output-root",str(tmp_path/"output")])
    ds, plan = gotcha_cli.make_plan(args)
    assert not ds.summary()["response_payload_read"]
    assert plan["plans"][0]["backend"]["module"] == "train_radar_fields"
    assert not (tmp_path/"output").exists()


def test_root_gotcha_dispatch_calls_maintained_backend(tmp_path, monkeypatch):
    dataset = fixture_dataset(tmp_path)
    import rift.radar_fields_gotcha as backend
    seen = []
    def fake(**kwargs):
        seen.append(kwargs)
        return {"status":"complete"}
    monkeypatch.setattr(backend,"run_gotcha",fake)
    plan = dict(backend=gotcha_cli.backend_registry()["radar_fields"], method="radar_fields",
                output_dir=str(tmp_path/"run"), resume=None, config={})
    assert gotcha_cli.dispatch(dataset,plan,"cpu")["status"] == "complete"
    assert seen[0]["dataset"] is dataset and seen[0]["config"] == {}


def test_native_heads_passes_frequency_grids_and_sealed_test(tmp_path):
    ds = fixture_dataset(tmp_path, ("hh","hv"), (1,2))
    for pol in ds.polarizations:
        for p in ds.passes:
            obs = next(ds.observations(p,ds.split["train"][0],pol))
            antenna, ranges = range_geometry(obs,ds.region)
            assert antenna.dtype == ranges.dtype == torch.float64
            assert len(obs.frequencies_hz) == 32+p+(pol=="hv")
            assert obs.autofocus == ("own_published_source_af_once" if pol=="hh" else "raw_official_af_absent")
            assert torch.isfinite(matched_range_power(obs,ranges)).all()
    with pytest.raises(PermissionError):
        next(ds.observations(1,ds.split["test"][0],"hh"))


def assert_equal(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
    elif isinstance(left,dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        assert len(left) == len(right)
        for a,b in zip(left,right):
            assert_equal(a,b)
    else:
        assert left == right


def test_actual_cpu_updates_resume_and_pre_response_identity_gates(tmp_path, monkeypatch):
    ds = fixture_dataset(tmp_path, ("hh","hv"))
    config = portable_config()
    complete = tmp_path/"complete"
    result = run_gotcha(dataset=ds,output_dir=complete,config=config,device="cpu",resume=None)
    assert result["status"] == "complete" and result["step"] == 2
    full = torch.load(complete/"checkpoint_final.pt",weights_only=False)
    assert set(full["training_statistics"]["peak_power"]) == {"hh","hv"}
    assert all(v==250 for v in full["training_statistics"]["pulse_counts"].values())
    assert full["history"][-1]["metrics"]["by_polarization"]["hv"]["pulses"] == 55
    assert sum(full["training_view_coverage"]["counts"].values()) == 2
    interrupted = tmp_path/"interrupted"
    original_step = torch.optim.Adam.step
    with monkeypatch.context() as m:
        def stop_after_update(opt,*args,**kwargs):
            result = original_step(opt,*args,**kwargs)
            signal.raise_signal(signal.SIGTERM)
            return result
        m.setattr(torch.optim.Adam,"step",stop_after_update)
        partial = run_gotcha(dataset=ds,output_dir=interrupted,config=config,device="cpu",resume=None)
        assert partial["status"] == "interrupted" and partial["step"] == 1
    result = run_gotcha(dataset=ds,output_dir=interrupted,config=config,device="cpu",resume=interrupted/"checkpoint_latest.pt")
    resumed = torch.load(interrupted/"checkpoint_final.pt",weights_only=False)
    for key in ("model_state_dict","optimizer","scheduler","history","training_view_coverage","rng_torch","rng_numpy"):
        assert_equal(full[key],resumed[key])
    assert any(torch.count_nonzero(s["exp_avg"]) for s in full["optimizer"]["state"].values())
    monkeypatch.setattr(NativeShardReader,"read",lambda *_: pytest.fail("read before identity gate"))
    bad = deepcopy(full); bad["training_statistics"]["role"] = "validation"
    bad_path = tmp_path/"bad.pt"; torch.save(bad,bad_path)
    with pytest.raises(ValueError,match="normalization"):
        run_gotcha(dataset=ds,output_dir=tmp_path/"bad-output",config=config,device="cpu",resume=bad_path)
    with pytest.raises(ValueError,match="recipe"):
        run_gotcha(dataset=ds,output_dir=tmp_path/"wrong-recipe",config={**config,"ray_samples":9},device="cpu",resume=complete/"checkpoint_final.pt")


def test_source_readout_config_matches_collection_command(tmp_path):
    from scripts.readout_radar_fields_b7873200_native import CONFIG_ARG_KEYS
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/"protocols/radar_fields_rift_source_adapted_v3.json").read_text())
    cmd = commands_for("a320","radar_fields",dataset_root=tmp_path,output_root=tmp_path)[0]
    args = rf_cli.parse_args(cmd[2:])
    for key in (*CONFIG_ARG_KEYS,"recipe","ray_samples","model_backend"):
        assert getattr(args,key) == config["training"][key], key


def test_source_profile_executes_one_actual_batch_with_original_clock(tmp_path, monkeypatch):
    ds = fixture_dataset(tmp_path)
    # Explicit portable neural probe; source sampling, objective and clocks
    # remain selected. This is not a claim of CUDA network parity, so the probe
    # lifts only the released-network locks of source-adapted-v3 for this test.
    import rift.radar_fields_recipe as recipe_module
    config = dict(model_backend="torch", hidden_dim=8, feature_dim=4, hash_levels=2,
                  hash_base_resolution=4, hash_final_resolution=8, hash_log2_size=8)
    monkeypatch.setattr(recipe_module, "SOURCE_LOCKS",
                        {k: v for k, v in recipe_module.SOURCE_LOCKS.items() if k not in config})
    original = torch.optim.Adam.step
    def stop(opt,*args,**kwargs):
        out = original(opt,*args,**kwargs)
        signal.raise_signal(signal.SIGTERM)
        return out
    monkeypatch.setattr(torch.optim.Adam,"step",stop)
    out = tmp_path/"source"
    result = run_gotcha(dataset=ds,output_dir=out,config=config,device="cpu",resume=None)
    assert result["status"] == "interrupted" and result["step"] == 1
    saved = torch.load(out/"checkpoint_latest.pt",weights_only=False)
    assert sum(saved["training_view_coverage"]["counts"].values()) == 10
    assert saved["scheduler"]["_last_lr"][0] == pytest.approx(.001*.1**(1/800))
    assert any(torch.count_nonzero(s["exp_avg"]) for s in saved["optimizer"]["state"].values())
    validate_resume(saved,ds,recipe_from_config(config,ds.region.half_extent_m,250),"cpu")
