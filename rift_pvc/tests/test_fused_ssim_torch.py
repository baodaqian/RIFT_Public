"""fused_ssim_torch_v1: the release SSIM in the CUDA extension's call shape (Package F, D2)."""
from __future__ import annotations

import pytest
import torch

from rift.radarsplat_fidelity import release_ssim_index
from rift_pvc.fused_ssim_torch import GAUSSIAN_TAPS, allowed_padding, fused_ssim, ssim_map


def _pair(h, w, seed=3, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(h, w, generator=g, dtype=dtype, requires_grad=True),
            torch.rand(h, w, generator=g, dtype=dtype))


def test_kernel_taps_are_the_normalised_gaussian_of_the_extension():
    taps = torch.tensor(GAUSSIAN_TAPS, dtype=torch.float64)
    assert len(taps) == 11 and torch.allclose(taps, taps.flip(0))
    assert abs(float(taps.sum()) - 1.0) < 1e-7
    t = torch.arange(-5, 6, dtype=torch.float64)
    g = torch.exp(-0.5 * (t / 1.5).square())
    torch.testing.assert_close(taps, g / g.sum(), atol=1e-8, rtol=0)


def test_valid_padding_matches_release_ssim_index_values_and_gradients():
    p, t = _pair(40, 33)
    ours = fused_ssim(p[None, None].repeat(1, 3, 1, 1), t[None, None].repeat(1, 3, 1, 1), padding="valid")
    reference = release_ssim_index(p, t)
    assert abs(float(ours) - float(reference)) < 1e-6
    g_ours, = torch.autograd.grad(ours, p, retain_graph=True)
    g_ref, = torch.autograd.grad(reference, p)
    assert float((g_ours - g_ref).abs().max()) < 1e-6


def test_same_padding_map_shape_and_identity_images():
    p, t = _pair(20, 17)
    values = ssim_map(0.01 ** 2, 0.03 ** 2, p[None, None], t[None, None])
    assert values.shape == (1, 1, 20, 17)
    same = ssim_map(0.01 ** 2, 0.03 ** 2, t[None, None], t[None, None])
    torch.testing.assert_close(same, torch.ones_like(same), atol=1e-9, rtol=0)
    assert float(fused_ssim(t[None, None], t[None, None], padding="same")) == pytest.approx(1.0, abs=1e-9)


def test_gradient_reaches_img1_only_and_passes_gradcheck():
    p, t = _pair(14, 15)
    t2 = t.clone().requires_grad_(True)
    fused_ssim(p[None, None], t2[None, None], padding="same").backward()
    assert t2.grad is None and torch.isfinite(p.grad).all()
    x, y = _pair(13, 12, seed=5)
    assert torch.autograd.gradcheck(lambda u: fused_ssim(u[None, None], y[None, None], padding="valid"),
                                    (x,), eps=1e-6, atol=1e-6)


def test_rejects_unknown_padding_shapes_and_tiny_valid_images():
    p, t = _pair(12, 12)
    assert allowed_padding == ["same", "valid"]
    with pytest.raises(AssertionError):
        fused_ssim(p[None, None], t[None, None], padding="reflect")
    with pytest.raises(ValueError):
        fused_ssim(p, t)
    small, small_t = _pair(10, 12)
    with pytest.raises(ValueError):
        fused_ssim(small[None, None], small_t[None, None], padding="valid")
