"""Released Radar Fields pose refinement on GOTCHA pass-sectors (user decision 2, 2026-09-22).

Release (external/RadarFields_reference; configs/radarfields.ini ``refine_poses = True``
with parse.py defaults): ``radarfields.nn.pose_refinement.PoseOptimizer`` holds one se(3)
tangent vector per frame, starting at zero, maps it with ``exp_map_SE3`` and applies it in
the sensor frame, ``pose @ exp(adjustment)``. Its optimizer is Adam, lr 9e-4, betas
(0.9, 0.99), eps 1e-15, with no pose regularization, no coplanarity term and no pose
schedule. Held-out frames get ``interp_test_poses``: linear interpolation of the refined
training adjustments over the frame index, extrapolated at the ends.

GOTCHA adaptation (the collection keeps refinement off: exact poses, spiral order):
- A frame is a registered pass-sector. Its sensor pose is the sector frame of our
  RadarSplat GOTCHA adapter (``frame_for_positions``: mean pulse position, x outward
  from the region), built from shard metadata only.
- The correction moves the sector's pulse positions rigidly, a' = T exp(adj) T^-1 a.
  GOTCHA ships no attitude, so ray pointing stays the adapter's look-at-ROI convention,
  recomputed from the corrected positions. The rotation acts through the rigid motion
  of the sector's ~100 m track about its centre, a lever arm comparable to the release's
  ~50 m scenes, so the release's lr applies unchanged.
- The measured range axis stays the measured one (``range_geometry`` of the nominal
  pulse), as the release keeps its FFT bins and moves the sensor.
- Trajectory order within a pass is the sector id: each pass is one continuous circle cut
  into 1-degree sectors. The 360/1 seam is treated as the sequence ends, where the release
  extrapolates.
- Each polarization head has its own corrections, as it has its own field.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

DISABLED = "disabled"
RELEASE_SE3 = "release_se3_v1"
POSE_MODES = (DISABLED, RELEASE_SE3)
RELEASE_POSE = dict(
    schema="rift_radar_fields_gotcha_pose_v1", mode="SE3", lr=9e-4, betas=[0.9, 0.99], eps=1e-15,
    regularization="none (release reg_poses/reg_poses_coplanar off)", lr_schedule="none (release schedule_pose off)",
    initial_adjustment="zero", frame="radarsplat_gotcha_frame_for_positions_mean_pulse_x_outward",
    application="pulse positions a' = T exp_map_SE3(adj) T^-1 a; look-at-ROI pointing from corrected positions",
    trajectory_order="sector id within each pass",
    held_out="scipy interp1d linear over sector id within the pass, extrapolated at the ends (release interp_test_poses)",
    heads="independent per polarization")


def exp_map_se3(tangent):
    """The release's own ``exp_map_SE3`` (pinned source)."""
    from .radar_fields_upstream import original_module
    return original_module("radarfields.nn.pose_refinement").exp_map_SE3(tangent)


def sector_frame(dataset, view, polarization):
    """4x4 region-local sensor frame of one pass-sector, from shard metadata only."""
    from .radarsplat_gotcha import frame_for_positions
    pass_id, sector = view
    shard = dataset.shards[pass_id, polarization]
    rows = shard.sector_rows[sector]
    if not len(rows):
        raise ValueError("Empty native pass-sector")
    xyz = dataset.region.to_local(np.stack([shard.arrays[k][rows] for k in ("x", "y", "z")], -1))
    return frame_for_positions(xyz)


class SectorPoses(nn.Module):
    """One release se(3) adjustment per TRAIN pass-sector of one polarization head."""

    def __init__(self, dataset, polarization, device="cpu"):
        super().__init__()
        self.train_views = [tuple(v) for v in dataset.viewpoints("train")]
        self.index = {v: i for i, v in enumerate(self.train_views)}
        self._dataset, self._polarization = dataset, polarization
        self._frames = {}
        self.adjustment = nn.Parameter(torch.zeros(len(self.train_views), 6, dtype=torch.float32, device=device))

    def frame(self, view):
        view = tuple(view)
        if view not in self._frames:
            self._frames[view] = torch.as_tensor(sector_frame(self._dataset, view, self._polarization),
                                                 dtype=torch.float64, device=self.adjustment.device)
        return self._frames[view]

    def held_out_adjustment(self, view):
        """Release ``interp_test_poses`` for a non-training sector: detached, linear, extrapolated."""
        from scipy.interpolate import interp1d
        pass_id, sector = tuple(view)
        members = sorted((s, i) for (p, s), i in self.index.items() if p == pass_id)
        if not members:
            raise ValueError(f"pass {pass_id} has no training sector to interpolate from")
        values = self.adjustment.detach().double().cpu().numpy()[[i for _, i in members]]
        if len(members) == 1:
            result = values[0]
        else:
            result = interp1d([s for s, _ in members], values, kind="linear", axis=0, bounds_error=False,
                              fill_value="extrapolate")(sector)
        return torch.as_tensor(result, dtype=torch.float32, device=self.adjustment.device)

    def correction(self, view):
        """4x4 local-frame correction T exp(adj) T^-1 for ``view``; differentiable for TRAIN sectors."""
        view = tuple(view)
        adjustment = (self.adjustment[self.index[view]] if view in self.index
                      else self.held_out_adjustment(view))
        delta = torch.eye(4, dtype=torch.float64, device=adjustment.device)
        delta = torch.cat([exp_map_se3(adjustment[None])[0].double(), delta[3:]], 0)
        frame = self.frame(view)
        return frame @ delta @ torch.linalg.inv(frame)

    def apply(self, view, antenna, correction=None):
        """Corrected region-local antenna position(s) [..., 3]."""
        correction = self.correction(view) if correction is None else correction
        return antenna @ correction[:3, :3].T + correction[:3, 3]


def build_pose_heads(dataset, recipe, device):
    """``None`` when the recipe has no pose refinement (collection-style and pre-decision recipes)."""
    if "pose_refinement" not in recipe:
        return None, None
    heads = nn.ModuleDict({pol: SectorPoses(dataset, pol, device) for pol in dataset.polarizations})
    spec = recipe["pose_refinement"]
    optimizer = torch.optim.Adam(heads.parameters(), lr=spec["lr"], betas=tuple(spec["betas"]), eps=spec["eps"])
    return heads, optimizer


def with_pose_refinement(recipe, mode):
    """Record the mode; ``disabled`` leaves the recipe byte-identical to earlier ones."""
    if mode not in POSE_MODES:
        raise ValueError(f"Unknown Radar Fields GOTCHA pose refinement {mode!r}")
    if mode == DISABLED:
        return recipe
    recipe = dict(recipe, pose_refinement=dict(RELEASE_POSE, name=mode))
    recipe["model_recipe"] = dict(recipe["model_recipe"],
                                  pose_refinement="released_SE3_per_pass_sector_adam_9e-4_interpolated_held_out")
    return recipe
