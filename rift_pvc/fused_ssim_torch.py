"""Torch twin of the pinned fused-SSIM CUDA extension (``fused_ssim_torch_v1``).

Package F, decision D2. The release loss calls
``fused_ssim(power[None, None].repeat(1, 3, 1, 1), target[...], padding="valid")``
from `rahul-goel/fused-ssim` at ``1272e21a282342e89537159e4bad508b19b34157``.
That kernel (``ssim.cu``) computes, per channel, the SSIM map with

* an 11-tap Gaussian window (sigma 1.5) applied separably, x then y, on a
  zero-padded image (``get_pix_value`` returns 0 outside the image);
* ``mu = G*img``, ``sigma_sq = G*img**2 - mu**2``, ``sigma12 = G*(img1*img2) - mu1*mu2``;
* ``map = ((2 mu1 mu2 + C1)(2 sigma12 + C2)) / ((mu1^2 + mu2^2 + C1)(sigma1_sq + sigma2_sq + C2))``
  with ``C1 = 0.01**2`` and ``C2 = 0.03**2``;
* ``padding="valid"`` crops the map by 5 on every side; ``fused_ssim`` returns
  the mean of the (cropped) map;
* a hand-written backward that returns a gradient for ``img1`` only.

This module reproduces those statements with the kernel's own tap values
(``G_00`` .. ``G_10`` literals) and torch autograd. ``img2`` is detached so, as
with the extension, no gradient reaches it. It is the batched NCHW form of
``rift/radarsplat_fidelity.py::release_ssim_index`` (which is the valid-crop
special case on one 2-D image).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

IDENTITY = "fused_ssim_torch_v1"
UPSTREAM_COMMIT = "1272e21a282342e89537159e4bad508b19b34157"
# ssim.cu G_00 .. G_10: gaussian(11, sigma=1.5), normalised, as float literals.
GAUSSIAN_TAPS = (
    0.001028380123898387, 0.0075987582094967365, 0.036000773310661316, 0.10936068743467331,
    0.21300552785396576, 0.26601171493530273, 0.21300552785396576, 0.10936068743467331,
    0.036000773310661316, 0.0075987582094967365, 0.001028380123898387,
)
RADIUS = 5
allowed_padding = ["same", "valid"]


def _blur(x: Tensor) -> Tensor:
    """Separable 11x11 Gaussian, x then y, zero padding, per channel (``do_separable_conv_x/y``)."""
    channels = x.shape[1]
    taps = x.new_tensor(GAUSSIAN_TAPS)
    kx = taps.view(1, 1, 1, 11).expand(channels, 1, 1, 11)
    ky = taps.view(1, 1, 11, 1).expand(channels, 1, 11, 1)
    return F.conv2d(F.conv2d(x, kx, padding=(0, RADIUS), groups=channels), ky, padding=(RADIUS, 0), groups=channels)


def ssim_map(C1: float, C2: float, img1: Tensor, img2: Tensor) -> Tensor:
    """The full ("same") SSIM map of ``fusedssimCUDA`` for NCHW inputs."""
    mu1 = _blur(img1)
    sigma1_sq = _blur(img1 * img1) - mu1 * mu1
    mu2 = _blur(img2)
    sigma2_sq = _blur(img2 * img2) - mu2 * mu2
    sigma12 = _blur(img1 * img2) - mu1 * mu2
    numerator_c = 2.0 * mu1 * mu2 + C1
    numerator_d = 2.0 * sigma12 + C2
    denominator_a = mu1 * mu1 + mu2 * mu2 + C1
    denominator_b = sigma1_sq + sigma2_sq + C2
    return (numerator_c * numerator_d) / (denominator_a * denominator_b)


def fused_ssim(img1: Tensor, img2: Tensor, padding: str = "same", train: bool = True) -> Tensor:
    """Same call shape and result as ``fused_ssim.fused_ssim``: the mean of the SSIM map."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    assert padding in allowed_padding
    if img1.dim() != 4 or img2.dim() != 4 or img1.shape != img2.shape:
        raise ValueError("fused_ssim requires two NCHW images of the same shape")
    if torch.is_complex(img1) or torch.is_complex(img2):
        raise TypeError("fused_ssim requires real images")
    if padding == "valid" and min(img1.shape[-2:]) <= 2 * RADIUS:
        raise ValueError("fused_ssim padding='valid' needs images larger than 10 pixels on both axes")
    values = ssim_map(C1, C2, img1, img2.detach().to(img1.dtype))
    if padding == "valid":
        values = values[:, :, RADIUS:-RADIUS, RADIUS:-RADIUS]
    return values.mean()


__all__ = ["IDENTITY", "UPSTREAM_COMMIT", "GAUSSIAN_TAPS", "allowed_padding", "ssim_map", "fused_ssim"]
