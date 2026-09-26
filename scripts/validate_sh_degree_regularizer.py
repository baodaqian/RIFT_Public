#!/usr/bin/env python
"""Contract gates for RIFT's degree-weighted spherical-harmonic prior.

The experiment exposed by ``train.py --sh-degree-weight`` must tax angular
complexity without taxing the isotropic coefficient, remain invariant to the
exact gain/scene scale gauge, ignore inactive voxels, and provide the expected
degree-weighted gradients.  These CPU checks exercise that contract without
starting a training job.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rift.calibration import GlobalComplexGain  # noqa: E402
from rift.sparse_scene import SHVoxelGridScene  # noqa: E402
from train import regularization_loss  # noqa: E402


def gate(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"PASS: {name}")


def make_scene() -> tuple[SHVoxelGridScene, GlobalComplexGain]:
    scene = SHVoxelGridScene(
        granularity=2,
        extent=1.0,
        device=torch.device("cpu"),
        max_degree=3,
        init_degree=3,
        init_scale=0.0,
    )
    gain = GlobalComplexGain()
    with torch.no_grad():
        gain.log_mag.fill_(math.log(2.5))
    return scene, gain


def main() -> None:
    weight = 3.0e-8
    scene, gain = make_scene()

    # One coefficient in each degree band, repeated identically in all eight
    # voxels so mean_active has a simple closed form.
    with torch.no_grad():
        scene.w_re[..., 0] = 11.0  # l=0: deliberately large, but free
        scene.w_re[..., 1] = 1.0   # first l=1 coefficient
        scene.w_re[..., 4] = 2.0   # first l=2 coefficient
        scene.w_re[..., 9] = 3.0   # first l=3 coefficient
    penalty, terms = regularization_loss(
        scene, 0.0, weight, gain=gain, return_terms=True
    )
    expected_moment = 2.5 ** 2 * (2.0 * 1.0 ** 2 + 6.0 * 2.0 ** 2 + 12.0 * 3.0 ** 2)
    expected = weight * expected_moment
    gate("closed-form l(l+1) degree weighting", torch.allclose(
        penalty, torch.tensor(expected), rtol=1.0e-6, atol=1.0e-14
    ))
    gate("separate sh_degree audit component equals the total", set(terms) == {"sh_degree"}
         and torch.equal(terms["sh_degree"], penalty))

    with torch.no_grad():
        scene.w_re.zero_()
        scene.w_im.zero_()
        scene.w_re[..., 0] = 1000.0
    dc_penalty = regularization_loss(scene, 0.0, weight, gain=gain)
    gate("degree-0 coefficient is unpenalized", float(dc_penalty) == 0.0)

    with torch.no_grad():
        scene.w_re.zero_()
        scene.w_im.zero_()
        scene.w_re[..., 1] = 0.75
        scene.w_im[..., 9] = -1.25
    before = regularization_loss(scene, 0.0, weight, gain=gain).detach()
    scale = 137.0
    with torch.no_grad():
        scene.w_re.mul_(scale)
        scene.w_im.mul_(scale)
        gain.log_mag.sub_(math.log(scale))
    after = regularization_loss(scene, 0.0, weight, gain=gain).detach()
    gate("gain/scene gauge invariance", torch.allclose(before, after, rtol=2.0e-6, atol=1.0e-14))

    # Inactive entries must neither enter the sum nor its global active mean.
    with torch.no_grad():
        scene.w_re.zero_()
        scene.w_im.zero_()
        scene.active_mask.fill_(False)
        scene.active_mask.reshape(-1)[:4] = True
        scene.w_re.reshape(8, 16)[:4, 1] = 1.0
        scene.w_re.reshape(8, 16)[4:, 9] = 1.0e6
        gain.log_mag.zero_()
    inactive_safe = regularization_loss(scene, 0.0, weight, gain=gain)
    gate("inactive voxels are excluded from sum and mean", torch.allclose(
        inactive_safe, torch.tensor(weight * 2.0), rtol=1.0e-6, atol=1.0e-14
    ))

    # Equal l=1 and l=3 coefficients should receive gradient magnitudes in
    # the 12:2 = 6 ratio; the l=0 gradient must remain exactly zero.
    with torch.no_grad():
        scene.active_mask.fill_(True)
        scene.w_re.zero_()
        scene.w_im.zero_()
        scene.w_re[..., 0] = 1.0
        scene.w_re[..., 1] = 1.0
        scene.w_re[..., 9] = 1.0
    scene.zero_grad(set_to_none=True)
    grad_penalty = regularization_loss(scene, 0.0, weight, gain=None)
    grad_penalty.backward()
    grad = scene.w_re.grad[0, 0, 0]
    gate("degree-0 gradient is exactly zero", float(grad[0]) == 0.0)
    gate("higher-degree gradient follows l(l+1)", torch.allclose(
        grad[9] / grad[1], torch.tensor(6.0), rtol=1.0e-6, atol=1.0e-7
    ))
    gate("all parameter gradients are finite", torch.isfinite(scene.w_re.grad).all()
         and torch.isfinite(scene.w_im.grad).all())

    no_prior, no_terms = regularization_loss(
        scene, 0.0, 0.0, gain=gain, return_terms=True
    )
    gate("zero weight preserves the historical objective", no_prior is None and no_terms == {})
    print("SH degree regularizer: 9/9 checks passed")


if __name__ == "__main__":
    main()
