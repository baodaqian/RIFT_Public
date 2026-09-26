"""Radar Fields GOTCHA pose refinement (user decision 2, 2026-09-22): the release's PoseOptimizer per pass-sector."""
from copy import deepcopy

import numpy as np
import pytest
import torch
from scipy.interpolate import interp1d

from rift.radar_fields_gotcha import recipe_from_config, run_gotcha
from rift.radar_fields_pose import DISABLED, RELEASE_POSE, RELEASE_SE3, SectorPoses, sector_frame
from rift.radar_fields_upstream import original_module
from test_radar_fields_gotcha import fixture_dataset, portable_config


def test_disabled_is_the_earlier_recipe_and_the_default_is_the_release_optimizer():
    config = portable_config()
    off = recipe_from_config({**config, "pose_refinement": DISABLED}, .03, 250)
    on = recipe_from_config(config, .03, 250)
    assert "pose_refinement" not in off and "pose_refinement" not in off["controls"]
    assert off["model_recipe"]["pose_refinement"] == "disabled_calibrated_dataset_geometry"
    assert on["pose_refinement"] == dict(RELEASE_POSE, name=RELEASE_SE3)
    assert (on["pose_refinement"]["mode"], on["pose_refinement"]["lr"], on["pose_refinement"]["betas"],
            on["pose_refinement"]["eps"]) == ("SE3", 9e-4, [0.9, 0.99], 1e-15)
    assert {k: v for k, v in on.items() if k not in ("pose_refinement", "model_recipe")} == \
           {k: v for k, v in off.items() if k != "model_recipe"}
    with pytest.raises(ValueError, match="pose refinement"):
        recipe_from_config({**config, "pose_refinement": "so3xr3_unscaled"}, .03, 250)


def test_correction_is_the_release_pose_times_exp_adjustment(tmp_path):
    ds = fixture_dataset(tmp_path)
    poses = SectorPoses(ds, "hh")
    view = tuple(ds.viewpoints("train")[3])
    torch.manual_seed(0)
    with torch.no_grad():
        poses.adjustment.normal_(0, 1e-2)
    frame = torch.as_tensor(sector_frame(ds, view, "hh"), dtype=torch.float32)
    release = original_module("radarfields.nn.pose_refinement").PoseOptimizer(frame[None], "SE3", False, "cpu")
    with torch.no_grad():
        release.pose_adjustment.copy_(poses.adjustment[poses.index[view]][None])
    corrected = release.apply_to_poses(torch.tensor([0])).double()[0]
    antenna = torch.tensor([[.4, -.2, .3], [2., 1., -1.]], dtype=torch.float64)
    sensor_local = (antenna - frame[:3, 3].double()) @ frame[:3, :3].double()   # T^-1 a
    expected = sensor_local @ corrected[:3, :3].T + corrected[:3, 3]          # (T exp(adj)) (T^-1 a)
    torch.testing.assert_close(poses.apply(view, antenna).detach(), expected, rtol=0, atol=1e-5)
    with torch.no_grad():
        poses.adjustment.zero_()
    torch.testing.assert_close(poses.apply(view, antenna).detach(), antenna, rtol=0, atol=1e-12)


def test_held_out_sectors_interpolate_training_adjustments_within_their_pass(tmp_path):
    ds = fixture_dataset(tmp_path, passes=(1, 2))
    poses = SectorPoses(ds, "hh")
    with torch.no_grad():
        poses.adjustment.copy_(torch.randn_like(poses.adjustment))
    for pass_id in (1, 2):
        members = sorted((s, i) for (p, s), i in poses.index.items() if p == pass_id)
        spline = interp1d([s for s, _ in members], poses.adjustment.detach().double().numpy()[[i for _, i in members]],
                          axis=0, fill_value="extrapolate")
        for p, sector in ds.viewpoints("validation"):
            if p == pass_id:
                np.testing.assert_allclose(poses.held_out_adjustment((p, sector)).numpy(), spline(sector),
                                           rtol=1e-6, atol=1e-6)
    view = tuple(ds.viewpoints("validation")[0])
    assert not poses.correction(view).requires_grad


def test_training_moves_the_poses_resumes_them_and_gates_the_mode(tmp_path):
    ds = fixture_dataset(tmp_path)
    config = portable_config()
    out = tmp_path / "on"
    assert run_gotcha(dataset=ds, output_dir=out, config=config, device="cpu", resume=None)["status"] == "complete"
    saved = torch.load(out / "checkpoint_final.pt", weights_only=False)
    assert saved["recipe"]["pose_refinement"]["name"] == RELEASE_SE3
    moved = saved["pose_model_state_dict"]["hh.adjustment"]
    assert moved.shape == (len(ds.viewpoints("train")), 6) and torch.count_nonzero(moved) > 0
    state = saved["pose_optimizer"]["state"]
    assert state and all(s["exp_avg"].abs().sum() > 0 for s in state.values())
    off = tmp_path / "off"
    run_gotcha(dataset=ds, output_dir=off, config={**config, "pose_refinement": DISABLED}, device="cpu", resume=None)
    plain = torch.load(off / "checkpoint_final.pt", weights_only=False)
    assert "pose_model_state_dict" not in plain and "pose_refinement" not in plain["recipe"]
    # A saved mode resumes only as itself.
    with pytest.raises(ValueError):
        run_gotcha(dataset=ds, output_dir=tmp_path / "x", config={**config, "pose_refinement": DISABLED},
                   device="cpu", resume=out / "checkpoint_final.pt")
    bad = deepcopy(saved)
    del bad["pose_optimizer"]
    torch.save(bad, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="pose"):
        run_gotcha(dataset=ds, output_dir=tmp_path / "y", config=config, device="cpu", resume=tmp_path / "bad.pt")
