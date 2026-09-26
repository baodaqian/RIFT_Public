"""Formula-level parity of the five torch mirrors (Package F, D1) against the fork's own torch
references and against literal transcriptions of the CUDA kernels. CPU; no card needed."""
from __future__ import annotations

import math
import sys

import pytest
import torch

from rift.radarsplat_release import REFERENCE_ROOT, verify_reference
from rift_pvc import gsplat_torch_ops as ops
from rift_pvc.radarsplat_xpu_backend import install_backend_stub

pytestmark = pytest.mark.skipif(not (REFERENCE_ROOT / "gsplat/rendering.py").is_file(),
                                reason="pinned RadarSplat source not staged (scripts/fetch_radarsplat_reference.py)")


@pytest.fixture(scope="module")
def fork():
    root = verify_reference()
    install_backend_stub()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import gsplat.cuda._torch_impl as reference
    from gsplat.utils import upper_triangular_to_matrices
    return reference, upper_triangular_to_matrices


@pytest.fixture
def scene():
    torch.manual_seed(0)
    N = 96
    means = torch.randn(N, 3)
    quats = torch.randn(N, 4)
    scales = torch.rand(N, 3) * 2
    viewmats = torch.eye(4)[None]
    Ks = torch.tensor([[[6.0, 0, 20.0], [0, 6.0, 12.0], [0, 0, 1]]])
    return means, quats, scales, viewmats, Ks, 40, 24


@pytest.mark.parametrize("camera_model", ["ortho", "pinhole"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_projection_matches_fork_torch_reference(fork, scene, camera_model, dtype):
    """float64: the formulas agree to rounding. float32: the fork's einsum orders the products
    differently, and random ill-conditioned covariances amplify that on the conic (both are
    valid fp32 evaluations; the CUDA kernel has its own order and fast math)."""
    reference, to_full = fork
    W, H = scene[5:]
    means, quats, scales, viewmats, Ks = (t.to(dtype) for t in scene[:5])
    covars, _ = reference._quat_scale_to_covar_preci(quats, scales, True, False, triu=True)
    mine = ops.fully_fused_projection_xpu(means, covars, None, None, viewmats, Ks, W, H, eps2d=0.3,
                                          near_plane=-10., far_plane=10., calc_compensations=True,
                                          camera_model=camera_model)
    # ops.triu_to_full keeps the dtype; the fork's upper_triangular_to_matrices builds float32
    theirs = reference._fully_fused_projection(means, ops.triu_to_full(covars), viewmats, Ks, W, H, eps2d=0.3,
                                               near_plane=-10., far_plane=10., calc_compensations=True,
                                               camera_model=camera_model)
    assert int((theirs[0] > 0).sum()) > 0
    visible = (theirs[0] > 0) & (mine[0] > 0)
    tolerance = dict(atol=1e-10, rtol=1e-9) if dtype == torch.float64 else dict(atol=1e-4, rtol=2e-3)
    if dtype == torch.float64:
        assert torch.equal(mine[0], theirs[0])
    else:
        assert int((mine[0] != theirs[0]).sum()) <= 1  # a ceil boundary can flip in fp32
    for a, b in zip(mine[1:], theirs[1:]):
        torch.testing.assert_close(a[visible], b[visible], **tolerance)


def test_projection_quaternion_path_equals_covariance_path(fork, scene):
    reference, _ = fork
    means, quats, scales, viewmats, Ks, W, H = scene
    covars, _ = reference._quat_scale_to_covar_preci(quats, scales, True, False, triu=True)
    common = dict(near_plane=-10., far_plane=10., camera_model="ortho")
    q = ops.fully_fused_projection_xpu(means, None, quats, scales, viewmats, Ks, W, H, **common)
    c = ops.fully_fused_projection_xpu(means, covars, None, None, viewmats, Ks, W, H, **common)
    assert torch.equal(q[0], c[0])
    torch.testing.assert_close(q[3][c[0] > 0], c[3][c[0] > 0], atol=1e-5, rtol=1e-5)
    assert q[4] is None and c[4] is None  # compensations only when requested, as the wrapper returns
    with pytest.raises(NotImplementedError):
        ops.fully_fused_projection_xpu(means, covars, None, None, viewmats, Ks, W, H, packed=True)


def test_projection_culling_rules_of_the_kernel(fork):
    means = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, -5.0], [0.0, 0.0, 50.0], [100.0, 0.0, 0.5], [0.0, 0.0, 0.5]])
    covars = torch.tensor([[1.0, 0, 0, 1.0, 0, 1.0]] * 4 + [[0.0, 0, 0, 0.0, 0, 0.0]])  # last: zero covariance
    viewmats = torch.eye(4)[None]
    Ks = torch.tensor([[[1.0, 0, 8.0], [0, 1.0, 8.0], [0, 0, 1]]])
    radii, _, depths, conics, _ = ops.fully_fused_projection_xpu(
        means, covars, None, None, viewmats, Ks, 16, 16, eps2d=0.3, near_plane=0.0, far_plane=10.0, camera_model="ortho")
    assert radii.tolist() == [[math.ceil(3 * math.sqrt(1.3)), 0, 0, 0, math.ceil(3 * math.sqrt(0.3))]]
    assert torch.isfinite(conics).all()  # culled entries are finite garbage, never NaN (note 1)
    clipped, *_ = ops.fully_fused_projection_xpu(means, covars, None, None, viewmats, Ks, 16, 16, eps2d=0.3,
                                                 near_plane=0.0, far_plane=10.0, radius_clip=4.0, camera_model="ortho")
    assert clipped.tolist() == [[0, 0, 0, 0, 0]]


def test_projection_backward_is_finite_with_degenerate_gaussians(fork):
    torch.manual_seed(2)
    means = torch.randn(8, 3, requires_grad=True)
    scales = torch.rand(8, 3, requires_grad=True)
    quats = torch.randn(8, 4, requires_grad=True)
    with torch.no_grad():
        scales[0] = 0.0  # zero determinant -> culled (det <= 0 before blur is fine, eps2d keeps it valid)
        means[1, 2] = -100.0  # behind the near plane
    viewmats = torch.eye(4)[None]
    Ks = torch.tensor([[[3.0, 0, 8.0], [0, 3.0, 8.0], [0, 0, 1]]])
    radii, means2d, depths, conics, _ = ops.fully_fused_projection_xpu(
        means, None, quats, scales, viewmats, Ks, 16, 16, near_plane=-1.0, far_plane=1e3, camera_model="ortho")
    visible = radii > 0
    assert int(visible.sum()) >= 1 and not bool(visible[0, 1])
    (means2d[visible].sum() + conics[visible].sum() + depths[visible].sum()).backward()
    assert all(torch.isfinite(t.grad).all() for t in (means, scales, quats))


def test_isect_tiles_matches_fork_reference_and_is_stable(fork, scene):
    reference, _ = fork
    means, quats, scales, viewmats, Ks, W, H = scene
    covars, _ = reference._quat_scale_to_covar_preci(quats, scales, True, False, triu=True)
    radii, means2d, depths, _, _ = ops.fully_fused_projection_xpu(
        means, covars, None, None, viewmats, Ks, W, H, near_plane=-10., far_plane=10., camera_model="ortho")
    ts, tw, th = 16, math.ceil(W / 16), math.ceil(H / 16)
    zero = torch.zeros_like(depths)
    mine = ops.isect_tiles_xpu(means2d, radii, zero, ts, tw, th, packed=False, n_cameras=1, sort=True)
    theirs = reference._isect_tiles(means2d, radii, zero, ts, tw, th, sort=True)
    assert torch.equal(mine[0], theirs[0]) and torch.equal(mine[1], theirs[1])
    assert mine[2].dtype == torch.int32 and sorted(mine[2].tolist()) == sorted(theirs[2].tolist())
    ids, flat = mine[1], mine[2].long()
    tie = ids[1:] == ids[:-1]
    assert bool((ids[1:] >= ids[:-1]).all()) and bool((flat[1:][tie] >= flat[:-1][tie]).all())  # radix-sort order
    # negative/random depths: the int32 bit pattern goes into the key exactly like the kernel
    random_depth = torch.randn_like(depths)
    u_mine = ops.isect_tiles_xpu(means2d, radii, random_depth, ts, tw, th, sort=False)
    u_theirs = reference._isect_tiles(means2d, radii, random_depth, ts, tw, th, sort=False)
    assert torch.equal(u_mine[1], u_theirs[1]) and torch.equal(u_mine[2].long(), u_theirs[2].long())
    # nothing visible
    empty = ops.isect_tiles_xpu(means2d, torch.zeros_like(radii), zero, ts, tw, th)
    assert empty[0].sum() == 0 and empty[1].numel() == 0 and empty[2].numel() == 0
    with pytest.raises(NotImplementedError):
        ops.isect_tiles_xpu(means2d, radii, zero, ts, tw, th, packed=True)


def test_isect_offset_encode_matches_fork_reference(fork, scene):
    reference, _ = fork
    means, quats, scales, viewmats, Ks, W, H = scene
    covars, _ = reference._quat_scale_to_covar_preci(quats, scales, True, False, triu=True)
    radii, means2d, depths, _, _ = ops.fully_fused_projection_xpu(
        means, covars, None, None, viewmats, Ks, W, H, near_plane=-10., far_plane=10., camera_model="ortho")
    ts, tw, th = 8, math.ceil(W / 8), math.ceil(H / 8)
    _, ids, _ = ops.isect_tiles_xpu(means2d, radii, torch.zeros_like(depths), ts, tw, th)
    mine = ops.isect_offset_encode_xpu(ids, 1, tw, th)
    theirs = reference._isect_offset_encode(ids, 1, tw, th)
    assert mine.dtype == torch.int32 and torch.equal(mine, theirs)
    zeros = ops.isect_offset_encode_xpu(torch.empty(0, dtype=torch.int64), 2, tw, th)
    assert zeros.shape == (2, th, tw) and int(zeros.abs().sum()) == 0


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 4])
def test_spherical_harmonics_matches_fork_reference(fork, degree):
    reference, _ = fork
    torch.manual_seed(4)
    dirs = torch.randn(1, 50, 3)
    coeffs = torch.randn(1, 50, 36, 3)
    mine = ops.spherical_harmonics_xpu(degree, dirs, coeffs)
    theirs = reference._spherical_harmonics(degree, dirs, coeffs)
    torch.testing.assert_close(mine, theirs, atol=2e-6, rtol=1e-5)


def test_spherical_harmonics_kernel_cap_and_masks():
    torch.manual_seed(5)
    dirs = torch.randn(2, 30, 3)
    coeffs = torch.randn(2, 30, 36, 3)
    assert torch.equal(ops.spherical_harmonics_xpu(5, dirs, coeffs), ops.spherical_harmonics_xpu(4, dirs, coeffs))
    masks = torch.rand(2, 30) > 0.5
    masked = ops.spherical_harmonics_xpu(3, dirs, coeffs, masks=masks)
    assert torch.equal(masked[~masks], torch.zeros_like(masked[~masks]))
    torch.testing.assert_close(masked[masks], ops.spherical_harmonics_xpu(3, dirs, coeffs)[masks])
    dirs[0, 0] = 0.0  # a zero direction stays finite (note 1) and does not poison the gradient
    coeffs.requires_grad_(True)
    ops.spherical_harmonics_xpu(4, dirs, coeffs, masks=masks).sum().backward()
    assert torch.isfinite(coeffs.grad).all()


def kernel_transcription(range_start, range_end, means2d, conics, opacities, W, H, ts, offsets, flatten_ids):
    """Literal Python transcription of rasterize_to_indices_in_range_radargs.cu."""
    C, N = means2d.shape[:2]
    th, tw = offsets.shape[1:]
    n_isects = len(flatten_ids)
    off = offsets.reshape(-1).tolist()
    out = {}
    block = ts * ts
    for cam in range(C):
        for tile in range(th * tw):
            k = cam * th * tw + tile
            start = off[k]
            end = n_isects if k == C * th * tw - 1 else off[k + 1]
            num_batches = (end - start + block - 1) // block
            if range_start >= num_batches:
                continue
            ty, tx = tile // tw, tile % tw
            for b in range(range_start, min(range_end, num_batches)):
                batch_start = start + block * b
                for t in range(min(block, end - batch_start)):
                    g = int(flatten_ids[batch_start + t])
                    xy = means2d.reshape(-1, 2)[g]
                    opac = float(opacities.reshape(-1)[g])
                    cn = conics.reshape(-1, 3)[g]
                    for i in range(ty * ts, ty * ts + ts):
                        for j in range(tx * ts, tx * ts + ts):
                            if i >= H or j >= W:
                                continue
                            dx = float(xy[0]) - (j + 0.5)
                            dy = float(xy[1]) - (i + 0.5)
                            sigma = 0.5 * (float(cn[0]) * dx * dx + float(cn[2]) * dy * dy) + float(cn[1]) * dx * dy
                            alpha = min(0.999, opac * math.exp(-sigma))
                            if sigma < 0 or alpha < 1 / 255:
                                continue
                            out.setdefault((cam, i * W + j), []).append(g % N)
    gs, px, cs = [], [], []
    for (cam, pid) in sorted(out):
        for g in out[(cam, pid)]:
            gs.append(g)
            px.append(pid)
            cs.append(cam)
    return tuple(torch.tensor(v, dtype=torch.long) for v in (gs, px, cs))


@pytest.fixture
def radar_scene():
    torch.manual_seed(1)
    N, C, W, H, ts = 300, 2, 37, 21, 4  # 4x4 tiles (16 slots per batch) so that several batches exist per tile
    means2d = torch.rand(C, N, 2) * torch.tensor([W, H])
    radii = torch.randint(1, 6, (C, N), dtype=torch.int32)
    radii[0, :5] = 0
    conics = torch.rand(C, N, 3) * torch.tensor([1.0, 0.2, 1.0])
    opacities = torch.rand(C, N)
    tw, th = math.ceil(W / ts), math.ceil(H / ts)
    _, ids, flatten = ops.isect_tiles_xpu(means2d, radii, torch.zeros(C, N), ts, tw, th, n_cameras=C)
    offsets = ops.isect_offset_encode_xpu(ids, C, tw, th)
    return means2d, radii, conics, opacities, W, H, ts, tw, th, offsets, flatten


def test_radar_indices_match_kernel_transcription_for_every_batch_range(radar_scene):
    means2d, radii, conics, opacities, W, H, ts, tw, th, offsets, flatten = radar_scene
    C = means2d.shape[0]
    ends = torch.cat([offsets.flatten()[1:], torch.tensor([flatten.numel()])])
    num_batches = int(((ends - offsets.flatten()).max() + ts * ts - 1) // (ts * ts))
    assert num_batches >= 2
    total = 0
    for a, b in [(0, 1), (1, 2), (0, 10 ** 6), (1, num_batches), (num_batches, num_batches + 5)]:
        mine = ops.rasterize_to_indices_in_range_radargs_xpu(a, b, torch.ones(C, H, W), means2d, conics, opacities,
                                                             W, H, ts, offsets, flatten)
        theirs = kernel_transcription(a, b, means2d, conics, opacities, W, H, ts, offsets, flatten)
        assert all(torch.equal(x, y) for x, y in zip(mine, theirs)), (a, b)
        assert all(t.dtype == torch.int64 for t in mine)
        total += mine[0].numel()
    assert total > 0
    empty = ops.rasterize_to_indices_in_range_radargs_xpu(0, 1, torch.ones(C, H, W), means2d, conics, opacities,
                                                          W, H, ts, offsets, torch.empty(0, dtype=torch.int32))
    assert all(t.numel() == 0 for t in empty)


def test_fork_radar_rasterizer_on_the_mirrors_matches_brute_force(fork, radar_scene):
    """The fork's own ``_rasterize_to_radar_pixels`` (unchanged) on the mirrors.

    Note (fork-obs1): that function re-zeroes ``summed_weights`` inside its batch loop, so
    with ``batch_per_iter`` smaller than the number of batches only the last range survives.
    Production uses the default 100 (one loop iteration for any tile with fewer than 25600
    intersections). Here the full result is checked with one iteration, and the last-range
    behaviour is pinned as the fork computes it.
    """
    from gsplat.cuda import _wrapper
    from gsplat.cuda._torch_impl_radar import _rasterize_to_radar_pixels, accumulate
    _wrapper.rasterize_to_indices_in_range_radargs = ops.rasterize_to_indices_in_range_radargs_xpu
    means2d, radii, conics, opacities, W, H, ts, tw, th, offsets, flatten = radar_scene
    C, N = means2d.shape[:2]
    ends = torch.cat([offsets.flatten()[1:], torch.tensor([flatten.numel()])])
    num_batches = int(((ends - offsets.flatten()).max() + ts * ts - 1) // (ts * ts))
    assert num_batches >= 2
    render = _rasterize_to_radar_pixels(means2d, conics, opacities, W, H, ts, offsets, flatten,
                                        batch_per_iter=num_batches)
    assert render.shape == (C, H, W, 1)
    last_only = _rasterize_to_radar_pixels(means2d, conics, opacities, W, H, ts, offsets, flatten, batch_per_iter=1)
    gs, px, cs = ops.rasterize_to_indices_in_range_radargs_xpu(num_batches - 1, num_batches, torch.ones(C, H, W),
                                                               means2d, conics, opacities, W, H, ts, offsets, flatten)
    expected_last = accumulate(means2d, conics, opacities, gs, px, cs, W, H)
    torch.testing.assert_close(last_only, expected_last, atol=1e-6, rtol=1e-5)
    assert float((render - last_only).abs().max()) > 1e-3  # the earlier ranges really are dropped by the fork
    ii, jj = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    for cam in range(C):
        brute = torch.zeros(H, W)
        for n in range(N):
            if radii[cam, n] <= 0:
                continue
            x, y = means2d[cam, n]
            r = float(radii[cam, n]) / ts
            tmin_x, tmax_x = max(0, math.floor(x / ts - r)), min(tw, math.ceil(x / ts + r))
            tmin_y, tmax_y = max(0, math.floor(y / ts - r)), min(th, math.ceil(y / ts + r))
            cover = (jj // ts >= tmin_x) & (jj // ts < tmax_x) & (ii // ts >= tmin_y) & (ii // ts < tmax_y)
            dx, dy = x - (jj + 0.5), y - (ii + 0.5)
            c = conics[cam, n]
            sigma = 0.5 * (c[0] * dx * dx + c[2] * dy * dy) + c[1] * dx * dy
            alpha = torch.clamp_max(opacities[cam, n] * torch.exp(-sigma), 0.999)
            brute += torch.where(cover & ~((sigma < 0) | (alpha < 1 / 255)), alpha, torch.zeros(()))
        torch.testing.assert_close(render[cam, :, :, 0], brute, atol=1e-5, rtol=1e-5)
