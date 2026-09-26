"""Source-native RF checks, intentionally separate from the MIMO adapter."""


import numpy as np
import pytest
import torch

from rift.radar_fields_released import (make_released_trainer, released_arguments,
    released_epoch_count, optimizer_factories, source_fidelity_inventory)
from rift.radar_fields_upstream import original_module


class SyntheticField(torch.nn.Module):
    """Only a numerical probe for executing the original CPU training code."""
    def __init__(self):
        super().__init__()
        self.coefficient = torch.nn.Parameter(torch.tensor([.2, .3, .4]))

    def forward(self, xyz, directions, sin_epoch=None):
        v = (xyz * self.coefficient).sum(-1, keepdim=True)
        return {"alpha": torch.sigmoid(v), "rd": torch.nn.functional.softplus(v + directions[..., :1])}

    def get_params(self, lr):
        return [{"params": self.parameters(), "lr": lr}]


def test_author_config_is_resolved_without_local_replacements():
    a = released_arguments()
    assert (a.iters, a.bs, a.num_rays_radar, a.num_fov_samples, a.seed) == (800, 10, 100, 10, 0)
    assert a.train_thresholded and a.integrate_rays and a.refine_poses
    assert a.ground_occ and a.penalize_above and a.mask and not a.learned_norm
    assert (a.weight_ground_occ, a.weight_above, a.pose_lr) == (.3, .05, .0009)
    assert (a.initial_offset, a.initial_scaler) == (1., 1.)
    assert source_fidelity_inventory()["rift_adapter_source_equivalent"] is False


def test_optimizer_uses_unmodified_source_parameter_groups_and_clock():
    a = released_arguments()
    make_optimizer, make_scheduler = optimizer_factories(a)
    model = SyntheticField()
    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer)
    assert optimizer.defaults["betas"] == (.9, .99)
    assert optimizer.defaults["eps"] == 1e-15
    assert optimizer.defaults["weight_decay"] == 0
    assert scheduler.lr_lambdas[0](800) == .1
    assert scheduler.lr_lambdas[0](960) == .1
    assert released_epoch_count(a, 3200) == 3  # 960 actual updates, not 800 or 8000


def source_engine(tmp_path):
    poses = torch.eye(4)[None].repeat(3, 1, 1)
    poses[:, 2, 3] = .3
    return make_released_trainer(workspace=tmp_path / "reference", all_poses=poses,
                                 heldout_indices=[2], device="cpu", model=SyntheticField())


def test_complete_original_trainer_keeps_pose_model_and_priors(tmp_path):
    engine = source_engine(tmp_path)
    assert type(engine) is original_module("radarfields.train").Trainer
    assert engine.refine_poses and engine.pose_model is not None
    assert engine.pose_optimizer is not None
    assert engine.args.ground_occ and engine.args.penalize_above
    assert engine.pose_lr_scheduler is None  # released schedule_pose=False
    assert engine.model.__class__ is SyntheticField  # explicit test probe, never an automatic CPU fallback


def test_released_epoch_mask_is_not_replaced_by_a_smooth_step_clock(tmp_path):
    engine = source_engine(tmp_path)
    engine.refine_poses = False  # suppress only trajectory plotting in this clock probe
    engine.args.save_loss_plot = False
    seen = []
    engine.train_epoch = lambda loader: seen.append((engine.epoch, engine.sin_epoch))
    engine.save_checkpoint = lambda **kwargs: None
    engine.train([], np.int32(3))
    np.testing.assert_allclose([v for _, v in seen], [.05+np.sqrt(.5), 1., .05+np.sqrt(.5)])
    assert [e for e, _ in seen] == [1, 2, 3]


def test_actual_sampler_keeps_central_ray_and_random_pitch_yaw():
    sampler = original_module("radarfields.sampler")
    poses = torch.eye(4)[None]
    torch.manual_seed(9)
    out = sampler.get_radar_rays(poses, (10., 10.), 1, 2, 10, torch.zeros(1, 2), "cpu")
    offsets = out["fov_samples"].reshape(1, 2, 10, 3)
    assert ((offsets == 0).all(-1).sum(-1) == 1).all()
    assert (offsets[..., 1].diff(dim=-1) >= 0).all()
    assert offsets[..., 1:].abs().max() <= 5
    torch.manual_seed(9)
    again = sampler.get_radar_rays(poses, (10., 10.), 1, 2, 10, torch.zeros(1, 2), "cpu")
    torch.testing.assert_close(out["directions"], again["directions"], rtol=0, atol=0)


def test_source_loss_includes_grounding_and_above_sensor_terms(tmp_path):
    engine = source_engine(tmp_path)
    b, n, s, r = 1, 2, 10, 3
    torch.manual_seed(5)
    alpha = torch.rand(b, n*s, r, requires_grad=True)
    integrated = alpha.reshape(b, n, s, r).mean(2)
    target = torch.rand(b, n, r)
    occupancy = torch.tensor([[[0., .3, .8], [.6, 0., .4]]])
    offsets = torch.zeros(b, n, s, 3)
    offsets[..., 1] = torch.linspace(-5, 5, s)
    data = {"bs": b, "num_rays_radar": n, "num_fov_samples": s,
            "num_range_samples": r, "fft": target, "occ": occupancy}
    points = {"angular_offsets": offsets.reshape(b, n*s, 3)}
    full = engine.compute_loss(data, points, integrated, integrated, integrated, alpha)
    assert "occupancy grounding penalty" in engine.loss_dict
    assert engine.loss_dict["occupancy above penalty"][-1] > 0
    engine.args.ground_occ = engine.args.penalize_above = False
    omitted = engine.compute_loss(data, points, integrated, integrated, integrated, alpha)
    assert full > omitted
    full.backward()
    assert torch.isfinite(alpha.grad).all()


def test_no_existing_adapter_checkpoint_is_silently_loaded(tmp_path):
    workspace = tmp_path / "existing"
    workspace.mkdir()
    (workspace / "checkpoint.txt").write_text("not a source checkpoint")
    with pytest.raises(FileExistsError, match="fresh workspace"):
        make_released_trainer(workspace=workspace, all_poses=torch.eye(4)[None],
                              heldout_indices=[], device="cpu", model=SyntheticField())


def test_adapted_loss_has_original_values_and_gradients_with_only_sensor_priors_disabled(tmp_path):
    from rift.radar_fields_native import released_batch_loss
    engine = source_engine(tmp_path)
    engine.args.ground_occ = engine.args.penalize_above = False
    torch.manual_seed(2)
    b, n, s, r = 2, 3, 10, 4
    raw = torch.rand(b,n*s,r,requires_grad=True)
    integrated = raw.reshape(b,n,s,r).mean(2)
    pred = torch.log10(integrated + 1)
    target = torch.rand(b,n,r)
    occ = torch.rand(b,n,r)
    occ[..., :2] = 0
    data = dict(bs=b,num_rays_radar=n,num_fov_samples=s,num_range_samples=r,fft=target,occ=occ)
    points = dict(angular_offsets=torch.zeros(b,n*s,3))
    reference = engine.compute_loss(data,points,pred,integrated,integrated,raw)
    records = [dict(prediction=pred[i],target=target[i],occupancy=integrated[i],occupancy_target=occ[i]) for i in range(b)]
    adapted, _ = released_batch_loss(records,source_exact=True)
    torch.testing.assert_close(reference,adapted,rtol=1e-6,atol=1e-7)
    a = torch.autograd.grad(reference,raw,retain_graph=True)[0]
    b = torch.autograd.grad(adapted,raw)[0]
    torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)


def test_source_ray_integration_retains_original_half_product_then_float_weights():
    from rift.radar_fields_native import prepare_bistatic_bins, render_bistatic_batch
    class HalfProbe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.3471))
        def query_chunked(self, xyz, view, **kwargs):
            alpha = self.weight.half().expand(len(xyz))
            return dict(alpha=alpha,rcs=alpha*alpha)
    model = HalfProbe()
    tx = torch.tensor([[10.,0.,0.]],dtype=torch.float64)
    request = prepare_bistatic_bins(tx,tx,torch.tensor([10.]),extent=.15,ray_samples=10,source_sampling=True)
    result = render_bistatic_batch(model,[request],query_chunk=8)[0]
    expected = (model.weight.half()*model.weight.half()).float()*request['inside'].float().mean(-1)
    assert result['rcs'].dtype == torch.float32
    torch.testing.assert_close(result['rcs'],expected)


def test_fixed_exterior_kl_gradient_is_source_positive_limit_not_clipped_probability():
    from rift.radar_fields_native import released_batch_loss
    x = torch.tensor([.2,.7,.4,.8],dtype=torch.float64,requires_grad=True)
    a = torch.cat((x, x[:2]*0))
    occ = torch.tensor([.0,.3,.8,.0,.4,.0],dtype=torch.float64)
    record = dict(prediction=a,target=occ,occupancy=a,occupancy_target=occ,
                  coverage=torch.tensor([1,1,1,1,0,0]))
    actual, _ = released_batch_loss([record],source_exact=True,weight_fft=0,weight_bimodal=0,weight_occ=1)
    # Independent original KL with decreasing positive FIXED exterior values.
    expected = torch.cat((x,torch.full((2,),1e-9,dtype=x.dtype)))
    expected = torch.nn.functional.kl_div((expected/expected.sum()).log(),occ/occ.sum(),reduction='sum')
    da = torch.autograd.grad(actual,x,retain_graph=True)[0]
    de = torch.autograd.grad(expected,x)[0]
    assert torch.isfinite(da).all()
    torch.testing.assert_close(da,de,rtol=2e-6,atol=1e-7)


def test_source_roi_aperture_does_not_depend_on_other_minibatch_pairs(monkeypatch):
    from rift.radar_fields_native import released_scene_directions
    sampler = original_module('radarfields.sampler')
    original = sampler.get_radar_rays
    seen = []
    def capture(poses, intrinsics, *args, **kwargs):
        seen.append(intrinsics[0].clone())
        return original(poses,intrinsics,*args,**kwargs)
    monkeypatch.setattr(sampler,'get_radar_rays',capture)
    a = torch.tensor([[10.,0.,0.]],dtype=torch.float64)
    b = torch.tensor([[5.,1.,0.]],dtype=torch.float64)
    released_scene_directions(a,.15,10)
    released_scene_directions(torch.cat((a,b)),.15,10)
    torch.testing.assert_close(seen[0][0],seen[1][0],rtol=0,atol=0)
