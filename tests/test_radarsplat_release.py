"""Pinned-source parity and comparison-recipe tests; no real dataset fitting."""
import ast
import copy
import json
import math
from pathlib import Path
import signal
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rift import radarsplat_release as release


def test_pinned_source_and_actual_launcher_contract():
    root = release.verify_reference(cuda_dependencies=True)
    shell = (root/"examples/demo_scripts/run_radarsplat.sh").read_text()
    top = (root/"examples/demo_scripts/run_all_radarsplat.sh").read_text()
    for text in ("INIT_NUM_PTS=20000", "INIT_SCALE=0.5"):
        assert text in top
    for text in ("--max_steps 2000", "L1OCCLOSS_LAMBDA=10", "--sh_degree 5",
                 "--sh_degree_interval 200", "--strategy.refine-stop-iter 0", "--radar_map_thres 0.10"):
        assert text in shell
    assert release.reference_contract()["launch_recipe"]["kernel_max_implemented_sh_degree"] == 4


def test_actual_upstream_initializer_and_position_schedule():
    splats, optimizers = release.create_scene(scene_scale=100, scene_center=[0, 0, 0],
                                             device="cpu", num_points=3)
    generator = torch.Generator().manual_seed(42)
    positions = 50*(torch.rand((3, 3), generator=generator)*2-1); positions[:, 2] = 0
    rgb = torch.rand((3, 3), generator=generator)
    quats = torch.rand((3, 4), generator=generator)
    torch.testing.assert_close(splats["means"], positions)
    torch.testing.assert_close(splats["quats"], quats)
    torch.testing.assert_close(splats["sh0"][:, 0], (rgb-.5)/.28209479177387814)
    assert splats["shN"].shape == (3, 35, 3)
    assert torch.all(splats["scales"].exp() == .25)
    assert torch.all(splats["opacities"].sigmoid() == .5)
    assert torch.all(splats["noise_probs"].sigmoid() == .5)
    schedule = release.position_scheduler(optimizers)
    for _ in range(2000):
        optimizers["means"].step(); schedule.step()
    assert optimizers["means"].param_groups[0]["lr"] == pytest.approx(.016*.01, rel=1e-12)


def test_objective_matches_executed_original_trainer_statements():
    splats, _ = release.create_scene(scene_scale=100, scene_center=[0, 0, 0], device="cpu", num_points=2)
    with torch.no_grad():
        splats["scales"].fill_(math.log(1.3))
        splats["noise_probs"].fill_(1.)
    from rift.radarsplat_fidelity import release_ssim_index
    ssim = lambda x,y,padding: release_ssim_index(x[0, 0],y[0, 0])
    rng = torch.Generator().manual_seed(11)
    p = torch.rand(12, 13, generator=rng, requires_grad=True)
    o = torch.rand(12, 13, generator=rng, requires_grad=True)
    target = torch.rand(12, 13, generator=rng)
    labels = (target>.5).float()
    ours = release.release_loss(p, o, target, labels, splats, ssim)
    source = release.REFERENCE_ROOT/"examples/radar_simple_trainer.py"
    tree = ast.parse(source.read_text())
    runner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Runner")
    train = next(n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    loop = next(n for n in train.body if isinstance(n, ast.For))
    start = next(n.lineno for n in loop.body if isinstance(n, ast.Assign) and
                 any(isinstance(t, ast.Name) and t.id == "max_size_loss" for t in n.targets))
    end = next(n.lineno for n in loop.body if isinstance(n, ast.If) and
               "torch.isnan(loss)" in ast.unparse(n.test))
    statements = [n for n in loop.body if start <= n.lineno < end]
    ns = dict(torch=torch, self=SimpleNamespace(splats=splats), cfg=SimpleNamespace(
        init_scale=.5, ssim_lambda=.2, l1occloss_lambda=10., maxsize_lambda=100., opa_noise_reg_loss_lambda=1000.),
        fused_ssim=ssim, out_img=p, pixels=target, l1loss=(p-target).abs().mean(),
        l1_occ_loss=(o-labels).abs().mean())
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(source), "exec"), ns)
    torch.testing.assert_close(ours["total"], ns["loss"])
    left = torch.autograd.grad(ours["total"], (p,o,splats["scales"],splats["noise_probs"]), retain_graph=True)
    right = torch.autograd.grad(ns["loss"], (p,o,splats["scales"],splats["noise_probs"]))
    for a,b in zip(left,right):
        torch.testing.assert_close(a,b)


def test_default_dispatch_uses_original_recipe_without_engineering_overrides(tmp_path):
    from rift.radarsplat_collection import commands_for
    commands = commands_for("a320", dataset_root=tmp_path, output_root=tmp_path/"runs")
    assert "budget48" in commands[1]
    assert not set(("--steps", "--init-num-gaussians", "--prune-every")) & set(commands[1])
    assert "scene_support" in commands[0]
    old = commands_for("a320", dataset_root=tmp_path, output_root=tmp_path/"runs", recipe="audit_v1")
    assert "480000" in old[1] and "2048" in old[1]


def test_budget_profile_is_explicit_and_rejects_upstream_before_model_access(tmp_path, monkeypatch):
    from rift import radarsplat_release_training as engine
    from scripts.validate_radarsplat_b7873200_native_contract import _make_synthetic_cache
    from rift.radarsplat_b7873200_protocol import load_cache
    _make_synthetic_cache(tmp_path/"cache")
    cache = load_cache(tmp_path/"cache")
    old = engine.identity_for_cache(cache, "upstream")
    new = engine.identity_for_cache(cache, "budget48")
    assert old["model_recipe"]["init_num_pts"] == 20000
    assert new["model_recipe"]["init_num_pts"] == 112000 > 48**3
    assert release.reference_contract()["launch_recipe"] == old["model_recipe"]
    assert {k:v for k,v in old["model_recipe"].items() if k != "init_num_pts"} == {
        k:v for k,v in new["model_recipe"].items() if k != "init_num_pts"}
    output = tmp_path/"run"; output.mkdir()
    torch.save(dict(schema=release.SCHEMA, identity=old), output/"checkpoint_latest.pt")
    monkeypatch.setattr(engine, "create_scene", lambda **_: pytest.fail("model accessed before recipe gate"))
    monkeypatch.setattr(engine, "load_cuda_reference", lambda **_: pytest.fail("CUDA accessed before recipe gate"))
    with pytest.raises(ValueError, match="identity mismatch"):
        engine.train(cache, output, device="cpu", profile="budget48")


def test_no_silent_cpu_model_substitution(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU/alternate-model fallback"):
        release.load_cuda_reference()


def test_original_filters_with_calibrated_output_sampling(monkeypatch):
    import torch.nn.functional as F
    from tests.test_radarsplat_fidelity import grid
    ns = release.source_functions("gsplat/rendering.py", ["spectral_leakage", "azimuth_antenna_gain_projection"],
                                  dict(torch=torch,F=F))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda t: t)  # test original pure kernels on CPU
    calls = []
    def raw(**kw):
        calls.append(kw)
        shape = (1,kw["height"],kw["width"],1)
        p = kw["means"].sum()*0 + torch.ones(shape)*.4
        return p,p/2,None,None,None,None,{}
    source = SimpleNamespace(_radar_rasterization=raw, **{k:ns[k] for k in
        ("spectral_leakage", "azimuth_antenna_gain_projection")})
    splats, _ = release.create_scene(scene_scale=100, scene_center=[0,0,0], device="cpu", num_points=2)
    renderer = release.ReleasedRenderer(source, 100.)
    power, occ = renderer(splats,torch.eye(4),grid(),5,torch.ones(16,16)*.1)
    torch.testing.assert_close(power,torch.full((16,16),.46),atol=2e-6,rtol=0)
    torch.testing.assert_close(occ,torch.full((16,16),.2),atol=2e-6,rtol=0)
    assert calls[0]["sh_degree"]==5 and calls[0]["sph"] and calls[0]["camera_model"]=="ortho"
    power.sum().backward()
    assert splats["means"].grad is not None


def test_multipath_conversion_keeps_short_crop_detection_active(tmp_path, monkeypatch):
    from scripts.validate_radarsplat_b7873200_native_contract import _make_synthetic_cache
    from rift.radarsplat_b7873200_protocol import load_cache, load_target
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda t,*a,**kw:
        t if a and isinstance(a[0],str) and a[0]=="cuda" else original_to(t,*a,**kw))
    _make_synthetic_cache(tmp_path/"cache")
    cache = load_cache(tmp_path/"cache")
    provider = release.ReleasedPreprocessing(cache)
    index = cache.train_indices[0]
    arrays = load_target(cache.root, index, "train", expected_grid=cache.grid)
    n = len(arrays["range_m"])
    synthetic = .5 + .3*np.cos(2*np.pi*6*np.arange(n)/n)
    arrays["radarsplat_mf_power"] = np.tile(synthetic, (len(arrays["azimuth_rad"]), 1))*cache.train_peak_power
    provider.occupancy._donor = lambda _: (arrays, synthetic, np.zeros(len(synthetic)))
    background = provider.background(arrays)
    assert provider.fft_peak_threshold == pytest.approx(30*n/int(50/.0596))
    assert np.isfinite(background).all() and background.max() > 0
    assert background.shape == arrays["radarsplat_mf_power"].shape


def test_released_engine_interruption_resume_preserves_source_schedule(tmp_path,monkeypatch):
    from rift import radarsplat_release_training as engine
    from scripts.validate_radarsplat_b7873200_native_contract import _make_synthetic_cache
    from rift.radarsplat_b7873200_protocol import load_cache
    import train_radarsplat as lifecycle
    from rift.radarsplat_fidelity import release_ssim_index
    # Test the original preprocessing arithmetic on CPU without changing its
    # production source/device contract (which deliberately requires CUDA).
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda t,*a,**kw:
        t if a and isinstance(a[0],str) and a[0]=="cuda" else original_to(t,*a,**kw))
    root=tmp_path/"cache"; _make_synthetic_cache(root); cache=load_cache(root)
    original_create=release.create_scene
    monkeypatch.setattr(engine,"create_scene",lambda **kw:original_create(**{**kw,"num_points":2}))
    control=dict(calls=0,stop=1)
    class Renderer:
        def __init__(self,source,units): self.units=units
        def __call__(self,splats,pose,grid,degree,bg):
            control["calls"]+=1
            if control["calls"]==control["stop"]: signal.raise_signal(signal.SIGTERM)
            size=(grid.output_azimuth_bins,grid.num_range_bins)
            return splats["sh0"].mean().sigmoid().expand(size),splats["opacities"].sigmoid().mean().expand(size)
    monkeypatch.setattr(engine,"ReleasedRenderer",Renderer)
    ssim=lambda x,y,padding:release_ssim_index(x[0,0],y[0,0])
    def run(folder,stop):
        control.update(calls=0,stop=stop)
        with pytest.raises(SystemExit) as exc:
            engine.train(cache,folder,device=torch.device("cpu"),rendering=object(),fused_ssim=ssim)
        assert exc.value.code==143
        return lifecycle._load_checkpoint(folder/"checkpoint_latest.pt",torch.device("cpu"))
    run(tmp_path/"resume",1)
    resumed=run(tmp_path/"resume",1)
    full=run(tmp_path/"full",2)
    assert resumed["step"]==full["step"]==2
    for key in ("splats","optimizers","position_scheduler","sampler"):
        assert lifecycle._directly_equal(resumed[key],full[key])
    # Raw state must be checked before Torch could silently convert its dtype
    # or change a supposedly fixed source hyperparameter.
    for kind in ("dtype", "lr", "moment", "scheduler"):
        corrupt=copy.deepcopy(resumed)
        if kind=="dtype": corrupt["splats"]["means"]=corrupt["splats"]["means"].double()
        if kind=="lr": corrupt["optimizers"]["scales"]["param_groups"][0]["lr"]*=2
        if kind=="moment": corrupt["optimizers"]["sh0"]["state"][0]["exp_avg_sq"].fill_(-1)
        if kind=="scheduler": corrupt["position_scheduler"]["gamma"]*=.9
        torch.save(corrupt,tmp_path/"resume/checkpoint_latest.pt")
        with pytest.raises(ValueError):
            engine.train(cache,tmp_path/"resume",device=torch.device("cpu"),rendering=object(),fused_ssim=ssim)
    torch.save(resumed,tmp_path/"resume/checkpoint_latest.pt")
    monkeypatch.setattr(engine,"load_cuda_reference",lambda **_:(object(),ssim))
    control.update(calls=0,stop=-1)
    from scripts.readout_radarsplat_checkpoint import readout
    geometry=tmp_path/"geometry.npz"
    result=readout(checkpoint_path=tmp_path/"resume/checkpoint_latest.pt",cache_root=root,geometry_path=geometry)
    assert result["step"]==2 and result["metrics"]["views"]==len(cache.validation_indices)
    with np.load(geometry) as data:
        np.testing.assert_allclose(data["means"],resumed["splats"]["means"].numpy()/result["identity"]["adapter"]["model_units_per_m"])
        assert str(data["checkpoint_sha256"])==result["checkpoint_sha256"]
        assert data["support"].shape == (48, 48, 48)
        assert np.isfinite(data["support"]).all()
        assert np.all((data["support"] >= 0) & (data["support"] <= 1))
        assert data["sample_centers_m"].shape == (48,)
    # Wrong object is rejected before either target-cache reads or CUDA setup.
    monkeypatch.setattr(engine,"load_cache",lambda *_:pytest.fail("read before identity gate"))
    with pytest.raises(ValueError):
        readout(checkpoint_path=tmp_path/"resume/checkpoint_latest.pt",cache_root=root,object_name="loader")
    wrong=copy.deepcopy(resumed); wrong["identity"]["source_commit"]="wrong"
    torch.save(wrong,tmp_path/"resume/checkpoint_latest.pt")
    with pytest.raises(ValueError,match="identity mismatch"):
        engine.train(cache,tmp_path/"resume",device=torch.device("cpu"),rendering=object(),fused_ssim=ssim)
