"""Torch mirrors of the five gsplat CUDA ops the RadarSplat fork's radar branch reaches.

Package F of ``RIFT_PVC_Adaptation.md`` (``docs/RADARSPLAT_PVC_ADAPTATION.md``,
decision D1). The production renderer is the authors' fork of gsplat at
``ea9c8f530c708622cc3b1b560436b5557ac6a49b``; its ``_radar_rasterization``
accumulates radar pixels in torch (``gsplat/cuda/_torch_impl_radar.py``) and
reaches CUDA only through five ops. Each function below mirrors the
corresponding kernel line by line and keeps the signature of the fork's Python
wrapper (``gsplat/cuda/_wrapper.py``) so ``rift_pvc.radarsplat_xpu_backend`` can
bind it in the wrapper's place:

    fully_fused_projection_xpu                <- csrc/fully_fused_projection_fwd.cu
                                                 (+ include/proj.cuh, utils.cuh, transform.cuh, quat.cuh)
    isect_tiles_xpu                           <- csrc/isect_tiles.cu (isect_tiles kernel + cub radix sort)
    isect_offset_encode_xpu                   <- csrc/isect_tiles.cu (isect_offset_encode kernel)
    spherical_harmonics_xpu                   <- csrc/compute_sh_fwd.cu (+ include/spherical_harmonics.cuh)
    rasterize_to_indices_in_range_radargs_xpu <- csrc/rasterize_to_indices_in_range_radargs.cu

Backward passes are torch autograd over these forwards; the CUDA backward
kernels are the parity reference (``rift_pvc/tests/test_radarsplat_parity.py``),
not the implementation. Documented differences from the kernels, none of which
changes a value the fork consumes:

1. Entries the kernels leave uninitialised (projection outputs of culled
   Gaussians, ``radii == 0``; SH colours of masked entries) are finite here:
   culled projections carry finite garbage computed with a unit determinant
   and masked SH colours are zero. The fork only ever reads entries with
   ``radii > 0``.
2. The fork compiles its kernels with ``--use_fast_math`` and the radar index
   kernel calls ``__expf``; these mirrors use IEEE arithmetic. A candidate pair
   whose alpha lies within float rounding of the ``1/255`` cutoff can therefore
   be classified differently (gate 1 reports the count).
3. The fork's own Python ``_isect_tiles`` loops in Python over every Gaussian
   and is unusable at 112000 Gaussians; ``isect_tiles_xpu`` is vectorised and
   reproduces the kernel's 64-bit key encoding and stable radix order.
4. Only the unpacked path (``packed=False``) is implemented, which is the one
   the fork's radar branch calls.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

RADAR_ALPHA_CUTOFF = 1.0 / 255.0
RADAR_ALPHA_MAX = 0.999
# (tile, intersection) pairs evaluated per chunk in the radar index mirror;
# each pair expands to tile_size**2 candidate pixels (16384 x 256 = 4.2M).
PAIR_CHUNK = 1 << 14


# --------------------------------------------------------------------------
# quat.cuh / quat_scale_to_covar_preci.cuh / transform.cuh / proj.cuh / utils.cuh
# --------------------------------------------------------------------------
def quat_to_rotmat(quats: Tensor) -> Tensor:
    """``quat.cuh::quat_to_rotmat``: (w, x, y, z), normalised inside; [..., 3, 3] row-major."""
    w, x, y, z = torch.unbind(quats, dim=-1)
    inv_norm = torch.rsqrt(x * x + y * y + z * z + w * w)
    w, x, y, z = w * inv_norm, x * inv_norm, y * inv_norm, z * inv_norm
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack([
        torch.stack([1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
        torch.stack([2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx)], dim=-1),
        torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2)], dim=-1),
    ], dim=-2)


def quat_scale_to_covar(quats: Tensor, scales: Tensor) -> Tensor:
    """``quat_scale_to_covar_preci.cuh``: C = (R S)(R S)^T. [N, 3, 3]."""
    M = quat_to_rotmat(quats) * scales[..., None, :]
    return M @ M.transpose(-1, -2)


def triu_to_full(covars: Tensor) -> Tensor:
    """The kernel's mat3 built from the six upper-triangle values [xx, xy, xz, yy, yz, zz]."""
    c0, c1, c2, c3, c4, c5 = torch.unbind(covars, dim=-1)
    return torch.stack([
        torch.stack([c0, c1, c2], dim=-1),
        torch.stack([c1, c3, c4], dim=-1),
        torch.stack([c2, c4, c5], dim=-1),
    ], dim=-2)


def _ortho_proj(mean_c: Tensor, covar_c: Tensor, fx: Tensor, fy: Tensor, cx: Tensor, cy: Tensor) -> Tuple[Tensor, Tensor]:
    """``proj.cuh::ortho_proj``: J = [[fx, 0, 0], [0, fy, 0]]; cov2d = J C J^T; mean2d = (fx x + cx, fy y + cy)."""
    s = covar_c
    cov2d = torch.stack([
        torch.stack([fx * fx * s[..., 0, 0], fx * fy * s[..., 0, 1]], dim=-1),
        torch.stack([fy * fx * s[..., 1, 0], fy * fy * s[..., 1, 1]], dim=-1),
    ], dim=-2)
    mean2d = torch.stack([fx * mean_c[..., 0] + cx, fy * mean_c[..., 1] + cy], dim=-1)
    return mean2d, cov2d


def _persp_proj(mean_c: Tensor, covar_c: Tensor, fx: Tensor, fy: Tensor, cx: Tensor, cy: Tensor,
                width: int, height: int) -> Tuple[Tensor, Tensor]:
    """``proj.cuh::persp_proj`` (pinhole), as in the fork's ``_persp_proj``."""
    x, y, z = torch.unbind(mean_c, dim=-1)
    tan_fovx = 0.5 * width / fx
    tan_fovy = 0.5 * height / fy
    lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx
    lim_x_neg = cx / fx + 0.3 * tan_fovx
    lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy
    lim_y_neg = cy / fy + 0.3 * tan_fovy
    rz = 1.0 / z
    rz2 = rz * rz
    tx = z * torch.minimum(lim_x_pos, torch.maximum(-lim_x_neg, x * rz))
    ty = z * torch.minimum(lim_y_pos, torch.maximum(-lim_y_neg, y * rz))
    zeros = torch.zeros_like(x)
    J = torch.stack([
        torch.stack([fx * rz, zeros, -fx * tx * rz2], dim=-1),
        torch.stack([zeros, fy * rz, -fy * ty * rz2], dim=-1),
    ], dim=-2)  # [C, N, 2, 3]
    cov2d = J @ covar_c @ J.transpose(-1, -2)
    mean2d = torch.stack([fx * x * rz + cx, fy * y * rz + cy], dim=-1)
    return mean2d, cov2d


def _fisheye_proj(mean_c: Tensor, covar_c: Tensor, fx: Tensor, fy: Tensor, cx: Tensor, cy: Tensor) -> Tuple[Tensor, Tensor]:
    """``proj.cuh::fisheye_proj``, as in the fork's ``_fisheye_proj``."""
    x, y, z = torch.unbind(mean_c, dim=-1)
    eps = 0.0000001
    xy_len = (x * x + y * y) ** 0.5 + eps
    theta = torch.atan2(xy_len, z + eps)
    mean2d = torch.stack([x * fx * theta / xy_len + cx, y * fy * theta / xy_len + cy], dim=-1)
    x2 = x * x + eps
    y2 = y * y
    xy = x * y
    x2y2 = x2 + y2
    x2y2z2_inv = 1.0 / (x2y2 + z * z)
    b = torch.atan2(xy_len, z) / xy_len / x2y2
    a = z * x2y2z2_inv / x2y2
    J = torch.stack([
        torch.stack([fx * (x2 * a + y2 * b), fx * xy * (a - b), -fx * x * x2y2z2_inv], dim=-1),
        torch.stack([fy * xy * (a - b), fy * (y2 * a + x2 * b), -fy * y * x2y2z2_inv], dim=-1),
    ], dim=-2)
    cov2d = J @ covar_c @ J.transpose(-1, -2)
    return mean2d, cov2d


# --------------------------------------------------------------------------
# fully_fused_projection_fwd.cu
# --------------------------------------------------------------------------
def fully_fused_projection_xpu(
    means: Tensor,  # [N, 3]
    covars: Optional[Tensor],  # [N, 6] or None
    quats: Optional[Tensor],  # [N, 4] or None
    scales: Optional[Tensor],  # [N, 3] or None
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    packed: bool = False,
    sparse_grad: bool = False,
    calc_compensations: bool = False,
    camera_model: str = "pinhole",
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """Mirror of ``fully_fused_projection_fwd_kernel`` with the wrapper's signature.

    Returns ``radii`` (int32 [C, N], 0 for culled), ``means2d`` [C, N, 2],
    ``depths`` [C, N], ``conics`` [C, N, 3] and ``compensations`` ([C, N] or
    ``None`` when ``calc_compensations`` is false, as the wrapper returns).
    """
    if packed or sparse_grad:
        raise NotImplementedError("the PVC mirror implements the unpacked path the fork's radar branch calls")
    C = viewmats.size(0)
    N = means.size(0)
    assert means.size() == (N, 3), means.size()
    assert viewmats.size() == (C, 4, 4), viewmats.size()
    assert Ks.size() == (C, 3, 3), Ks.size()
    if covars is not None:
        assert covars.size() == (N, 6), covars.size()
        covar = triu_to_full(covars)
    else:
        assert quats is not None, "covars or quats is required"
        assert scales is not None, "covars or scales is required"
        assert quats.size() == (N, 4), quats.size()
        assert scales.size() == (N, 3), scales.size()
        covar = quat_scale_to_covar(quats, scales)
    # pos_world_to_cam / covar_world_to_cam (transform.cuh); glm reads the
    # row-major viewmat as columns, which is the same matrix.
    R = viewmats[:, :3, :3]
    t = viewmats[:, :3, 3]
    mean_c = torch.einsum("cij,nj->cni", R, means) + t[:, None, :]  # [C, N, 3]
    covar_c = torch.einsum("cij,njk,clk->cnil", R, covar, R)  # [C, N, 3, 3]
    depths = mean_c[..., 2]
    valid = (depths >= near_plane) & (depths <= far_plane)  # kernel: cull if z < near or z > far
    fx, fy = Ks[:, 0, 0, None], Ks[:, 1, 1, None]  # [C, 1]
    cx, cy = Ks[:, 0, 2, None], Ks[:, 1, 2, None]
    if camera_model == "ortho":
        mean2d, cov2d = _ortho_proj(mean_c, covar_c, fx, fy, cx, cy)
    elif camera_model == "pinhole":
        mean2d, cov2d = _persp_proj(mean_c, covar_c, fx, fy, cx, cy, width, height)
    elif camera_model == "fisheye":
        mean2d, cov2d = _fisheye_proj(mean_c, covar_c, fx, fy, cx, cy)
    else:
        raise ValueError(f"unknown camera model {camera_model!r}")
    c00, c01, c10, c11 = cov2d[..., 0, 0], cov2d[..., 0, 1], cov2d[..., 1, 0], cov2d[..., 1, 1]
    # utils.cuh::add_blur
    det_orig = c00 * c11 - c01 * c10
    c00 = c00 + eps2d
    c11 = c11 + eps2d
    det = c00 * c11 - c01 * c10
    valid = valid & (det > 0)  # kernel: if (det <= 0) cull
    det_safe = torch.where(det > 0, det, torch.ones_like(det))  # finite garbage for culled entries (note 1)
    compensation = torch.sqrt(torch.clamp(det_orig / det_safe, min=0.0))
    # utils.cuh::inverse of the blurred 2-D covariance
    inv_det = 1.0 / det_safe
    conics = torch.stack([c11 * inv_det, -c01 * inv_det, c00 * inv_det], dim=-1)  # [C, N, 3]
    # three-sigma radius (non differentiable)
    b = 0.5 * (c00 + c11)
    v1 = b + torch.sqrt(torch.clamp(b * b - det, min=0.01))
    radius = torch.ceil(3.0 * torch.sqrt(v1))
    valid = valid & (radius > radius_clip)
    mx, my = mean2d[..., 0], mean2d[..., 1]
    outside = (mx + radius <= 0) | (mx - radius >= width) | (my + radius <= 0) | (my - radius >= height)
    valid = valid & ~outside
    radii = torch.where(valid, radius, torch.zeros_like(radius)).to(torch.int32)
    return radii, mean2d, depths, conics, (compensation if calc_compensations else None)


# --------------------------------------------------------------------------
# isect_tiles.cu
# --------------------------------------------------------------------------
def _tile_bits(n_tiles: int) -> int:
    return int(math.floor(math.log2(n_tiles))) + 1


def _depth_bits(depths: Tensor) -> Tensor:
    """``(int64_t) *(int32_t *)&depth``: the low 32 bits of each depth, sign-extended."""
    flat = depths.reshape(-1).contiguous()
    if flat.dtype == torch.float32:
        return flat.view(torch.int32).to(torch.int64)
    if flat.dtype == torch.float64:
        # little-endian low word of the double, as the kernel reads it
        return flat.view(torch.int32).reshape(-1, 2)[:, 0].to(torch.int64)
    raise TypeError(f"isect_tiles mirror supports float32/float64 depths, got {flat.dtype}")


@torch.no_grad()
def isect_tiles_xpu(
    means2d: Tensor,  # [C, N, 2]
    radii: Tensor,  # [C, N]
    depths: Tensor,  # [C, N]
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool = True,
    packed: bool = False,
    n_cameras: Optional[int] = None,
    camera_ids: Optional[Tensor] = None,
    gaussian_ids: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Mirror of the ``isect_tiles`` kernel pair and its stable radix sort (unpacked)."""
    if packed:
        raise NotImplementedError("the PVC mirror implements the unpacked path the fork's radar branch calls")
    C, N, _ = means2d.shape
    assert means2d.shape == (C, N, 2), means2d.shape
    assert radii.shape == (C, N), radii.shape
    assert depths.shape == (C, N), depths.shape
    device = means2d.device
    n_tiles = int(tile_width) * int(tile_height)
    tile_n_bits = _tile_bits(n_tiles)
    cam_n_bits = _tile_bits(C)
    if tile_n_bits + cam_n_bits > 32:
        raise ValueError("camera and tile ids do not fit the kernel's 32 id bits")
    radius = radii.to(means2d.dtype)
    tile_radius = radius / float(tile_size)
    tile_x = means2d[..., 0] / float(tile_size)
    tile_y = means2d[..., 1] / float(tile_size)
    # tile_min inclusive, tile_max exclusive; the kernel's float->uint32 casts saturate at 0
    tmin_x = torch.clamp(torch.floor(tile_x - tile_radius), 0, tile_width).to(torch.int64)
    tmin_y = torch.clamp(torch.floor(tile_y - tile_radius), 0, tile_height).to(torch.int64)
    tmax_x = torch.clamp(torch.ceil(tile_x + tile_radius), 0, tile_width).to(torch.int64)
    tmax_y = torch.clamp(torch.ceil(tile_y + tile_radius), 0, tile_height).to(torch.int64)
    visible = radii > 0
    tiles = torch.where(visible, (tmax_y - tmin_y) * (tmax_x - tmin_x), torch.zeros_like(tmin_x))
    tiles_per_gauss = tiles.to(torch.int32)
    counts = tiles.reshape(-1)
    n_isects = int(counts.sum())
    if n_isects == 0:
        return (tiles_per_gauss, torch.empty(0, dtype=torch.int64, device=device),
                torch.empty(0, dtype=torch.int32, device=device))
    idx = torch.repeat_interleave(torch.arange(C * N, device=device), counts)  # flatten index per isect
    starts = torch.cumsum(counts, 0) - counts
    local = torch.arange(n_isects, device=device) - starts[idx]
    span_x = (tmax_x - tmin_x).reshape(-1)[idx]
    ty = tmin_y.reshape(-1)[idx] + local // span_x
    tx = tmin_x.reshape(-1)[idx] + local % span_x
    tile_id = ty * tile_width + tx
    cid = idx // N
    depth_id = _depth_bits(depths)[idx]
    isect_ids = (cid << (32 + tile_n_bits)) | (tile_id << 32) | depth_id
    flatten_ids = idx.to(torch.int32)
    if sort:
        # cub::DeviceRadixSort::SortPairs over bits [0, 32 + tile_n_bits + cam_n_bits): stable
        mask = (1 << (32 + tile_n_bits + cam_n_bits)) - 1
        order = torch.sort(isect_ids & mask, stable=True).indices
        isect_ids = isect_ids[order]
        flatten_ids = flatten_ids[order]
    return tiles_per_gauss, isect_ids, flatten_ids


@torch.no_grad()
def isect_offset_encode_xpu(isect_ids: Tensor, n_cameras: int, tile_width: int, tile_height: int) -> Tensor:
    """Mirror of the ``isect_offset_encode`` kernel: offsets[c, ty, tx] = number of sorted isects before that tile."""
    C = int(n_cameras)
    n_tiles = int(tile_width) * int(tile_height)
    device = isect_ids.device
    if isect_ids.numel() == 0:
        return torch.zeros((C, tile_height, tile_width), dtype=torch.int32, device=device)
    tile_n_bits = _tile_bits(n_tiles)
    ids = isect_ids.to(torch.int64) >> 32
    cid = ids >> tile_n_bits
    tid = ids & ((1 << tile_n_bits) - 1)
    flat = cid * n_tiles + tid
    if bool((flat < 0).any()) or bool((flat >= C * n_tiles).any()):
        raise ValueError("intersection ids address a camera/tile outside the image (negative depth encoding?)")
    counts = torch.bincount(flat, minlength=C * n_tiles)
    offsets = torch.cumsum(counts, 0) - counts
    return offsets.to(torch.int32).reshape(C, tile_height, tile_width)


# --------------------------------------------------------------------------
# compute_sh_fwd.cu / spherical_harmonics.cuh::sh_coeffs_to_color_fast
# --------------------------------------------------------------------------
KERNEL_MAX_SH_DEGREE = 4


def spherical_harmonics_xpu(
    degrees_to_use: int,
    dirs: Tensor,  # [..., 3]
    coeffs: Tensor,  # [..., K, 3]
    masks: Optional[Tensor] = None,
) -> Tensor:
    """Mirror of ``sh_coeffs_to_color_fast`` for every channel at once.

    The kernel evaluates at most degree 4 (25 bases) whatever ``degrees_to_use``
    says; coefficients beyond index 24 never contribute. Masked entries are
    zero (the kernel leaves them uninitialised).
    """
    assert (degrees_to_use + 1) ** 2 <= coeffs.shape[-2], coeffs.shape
    assert dirs.shape[:-1] == coeffs.shape[:-2], (dirs.shape, coeffs.shape)
    assert dirs.shape[-1] == 3, dirs.shape
    assert coeffs.shape[-1] == 3, coeffs.shape
    if masks is not None:
        assert masks.shape == dirs.shape[:-1], masks.shape
    degree = int(degrees_to_use)
    c = coeffs
    result = 0.2820947917738781 * c[..., 0, :]
    if degree >= 1:
        x, y, z = torch.unbind(dirs, dim=-1)
        norm2 = x * x + y * y + z * z
        inorm = torch.rsqrt(torch.where(norm2 > 0, norm2, torch.ones_like(norm2)))  # finite for a zero direction (note 1)
        x = (x * inorm)[..., None]
        y = (y * inorm)[..., None]
        z = (z * inorm)[..., None]
        result = result + 0.48860251190292 * (-y * c[..., 1, :] + z * c[..., 2, :] - x * c[..., 3, :])
        if degree >= 2:
            z2 = z * z
            fTmp0B = -1.092548430592079 * z
            fC1 = x * x - y * y
            fS1 = 2.0 * x * y
            pSH6 = 0.9461746957575601 * z2 - 0.3153915652525201
            pSH7 = fTmp0B * x
            pSH5 = fTmp0B * y
            pSH8 = 0.5462742152960395 * fC1
            pSH4 = 0.5462742152960395 * fS1
            result = result + (pSH4 * c[..., 4, :] + pSH5 * c[..., 5, :] + pSH6 * c[..., 6, :]
                               + pSH7 * c[..., 7, :] + pSH8 * c[..., 8, :])
            if degree >= 3:
                fTmp0C = -2.285228997322329 * z2 + 0.4570457994644658
                fTmp1B = 1.445305721320277 * z
                fC2 = x * fC1 - y * fS1
                fS2 = x * fS1 + y * fC1
                pSH12 = z * (1.865881662950577 * z2 - 1.119528997770346)
                pSH13 = fTmp0C * x
                pSH11 = fTmp0C * y
                pSH14 = fTmp1B * fC1
                pSH10 = fTmp1B * fS1
                pSH15 = -0.5900435899266435 * fC2
                pSH9 = -0.5900435899266435 * fS2
                result = result + (pSH9 * c[..., 9, :] + pSH10 * c[..., 10, :] + pSH11 * c[..., 11, :]
                                   + pSH12 * c[..., 12, :] + pSH13 * c[..., 13, :] + pSH14 * c[..., 14, :]
                                   + pSH15 * c[..., 15, :])
                if degree >= 4:
                    fTmp0D = z * (-4.683325804901025 * z2 + 2.007139630671868)
                    fTmp1C = 3.31161143515146 * z2 - 0.47308734787878
                    fTmp2B = -1.770130769779931 * z
                    fC3 = x * fC2 - y * fS2
                    fS3 = x * fS2 + y * fC2
                    pSH20 = 1.984313483298443 * z * pSH12 - 1.006230589874905 * pSH6
                    pSH21 = fTmp0D * x
                    pSH19 = fTmp0D * y
                    pSH22 = fTmp1C * fC1
                    pSH18 = fTmp1C * fS1
                    pSH23 = fTmp2B * fC2
                    pSH17 = fTmp2B * fS2
                    pSH24 = 0.6258357354491763 * fC3
                    pSH16 = 0.6258357354491763 * fS3
                    result = result + (pSH16 * c[..., 16, :] + pSH17 * c[..., 17, :] + pSH18 * c[..., 18, :]
                                       + pSH19 * c[..., 19, :] + pSH20 * c[..., 20, :] + pSH21 * c[..., 21, :]
                                       + pSH22 * c[..., 22, :] + pSH23 * c[..., 23, :] + pSH24 * c[..., 24, :])
    if masks is not None:
        result = torch.where(masks[..., None], result, torch.zeros_like(result))
    return result


# --------------------------------------------------------------------------
# rasterize_to_indices_in_range_radargs.cu
# --------------------------------------------------------------------------
@torch.no_grad()
def rasterize_to_indices_in_range_radargs_xpu(
    range_start: int,
    range_end: int,
    transmittances: Tensor,  # [C, image_height, image_width]; accepted and unused, as in the kernel
    means2d: Tensor,  # [C, N, 2]
    conics: Tensor,  # [C, N, 3]
    opacities: Tensor,  # [C, N]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [C, tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
) -> Tuple[Tensor, Tensor, Tensor]:
    """Mirror of ``rasterize_to_indices_in_range_kernel`` (radar variant) and its wrapper.

    For every tile the kernel walks the intersection batches ``[range_start,
    min(range_end, num_batches))`` of ``tile_size**2`` entries, and for every
    pixel of the tile emits the Gaussian when ``sigma >= 0`` and
    ``alpha = min(0.999, opacity * exp(-sigma)) >= 1/255``. There is no
    transmittance early stop in the radar kernel. The output is ordered like
    the kernel's chunk layout: camera, then pixel (row-major), then the
    intersection order within the tile. Returns ``(gaussian_ids, pixel_ids,
    camera_ids)``, all int64.
    """
    C, N, _ = means2d.shape
    assert conics.shape == (C, N, 3), conics.shape
    assert opacities.shape == (C, N), opacities.shape
    assert isect_offsets.shape[0] == C, isect_offsets.shape
    tile_height, tile_width = int(isect_offsets.shape[1]), int(isect_offsets.shape[2])
    assert tile_height * tile_size >= image_height, f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert tile_width * tile_size >= image_width, f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"
    device = means2d.device
    n_isects = int(flatten_ids.numel())
    empty = torch.empty(0, dtype=torch.int64, device=device)
    if n_isects == 0 or int(range_end) <= int(range_start):
        return empty, empty.clone(), empty.clone()
    ts = int(tile_size)
    block = ts * ts
    n_tiles = tile_width * tile_height
    offsets = isect_offsets.reshape(-1).to(torch.int64)
    ends = torch.cat([offsets[1:], offsets.new_tensor([n_isects])])
    lo = offsets + block * int(range_start)
    hi = torch.minimum(ends, offsets + block * int(range_end))
    count = torch.clamp(hi - lo, min=0)
    total = int(count.sum())
    if total == 0:
        return empty, empty.clone(), empty.clone()
    tile = torch.repeat_interleave(torch.arange(C * n_tiles, device=device), count)
    starts = torch.cumsum(count, 0) - count
    isect = lo[tile] + (torch.arange(total, device=device) - starts[tile])
    g = flatten_ids[isect].to(torch.int64)  # flatten index in [C * N]
    cam = tile // n_tiles
    local_tile = tile % n_tiles
    ty = local_tile // tile_width
    tx = local_tile % tile_width
    xy = means2d.reshape(-1, 2)[g]
    con = conics.reshape(-1, 3)[g]
    opac = opacities.reshape(-1)[g]
    dy = torch.arange(ts, device=device, dtype=torch.int32).repeat_interleave(ts)  # thread (y, x) -> block index y*ts+x
    dx = torch.arange(ts, device=device, dtype=torch.int32).repeat(ts)
    hw = int(image_height) * int(image_width)
    keys, gauss = [], []
    for start in range(0, total, PAIR_CHUNK):
        sl = slice(start, min(start + PAIR_CHUNK, total))
        i = ty[sl, None].to(torch.int32) * ts + dy[None, :]  # [n, block]
        j = tx[sl, None].to(torch.int32) * ts + dx[None, :]
        inside = (i < image_height) & (j < image_width)
        px = j.to(xy.dtype) + 0.5
        py = i.to(xy.dtype) + 0.5
        delta_x = xy[sl, 0, None] - px
        delta_y = xy[sl, 1, None] - py
        c = con[sl]
        sigma = (0.5 * (c[:, 0, None] * delta_x * delta_x + c[:, 2, None] * delta_y * delta_y)
                 + c[:, 1, None] * delta_x * delta_y)
        alpha = torch.clamp_max(opac[sl, None] * torch.exp(-sigma), RADAR_ALPHA_MAX)
        keep = inside & ~((sigma < 0) | (alpha < RADAR_ALPHA_CUTOFF))
        pair, pix = keep.nonzero(as_tuple=True)
        keys.append(cam[sl][pair] * hw + i[pair, pix].to(torch.int64) * image_width + j[pair, pix].to(torch.int64))
        gauss.append(g[sl][pair] % N)
    key = torch.cat(keys)
    gid = torch.cat(gauss)
    order = torch.sort(key, stable=True).indices  # pairs were generated in intersection order per tile
    key = key[order]
    gid = gid[order]
    return gid, key % hw, key // hw


__all__ = [
    "RADAR_ALPHA_CUTOFF", "RADAR_ALPHA_MAX", "KERNEL_MAX_SH_DEGREE", "PAIR_CHUNK",
    "quat_to_rotmat", "quat_scale_to_covar", "triu_to_full",
    "fully_fused_projection_xpu", "isect_tiles_xpu", "isect_offset_encode_xpu",
    "spherical_harmonics_xpu", "rasterize_to_indices_in_range_radargs_xpu",
]
