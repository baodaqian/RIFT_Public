"""RadarSplat PVC device gates: the mirrors, the released renderer and the engine on a PVC card.

Skipped without an allocated XPU. Every device result is compared with the CPU
result of the same code on the same inputs; the CPU numbers are pinned by
``test_gsplat_torch_ops.py`` / ``test_radarsplat_pvc.py``.
"""
from __future__ import annotations

import math
import signal
import warnings

import pytest
import torch

from rift.radarsplat_release import REFERENCE_ROOT
from rift_pvc import gsplat_torch_ops as ops
from rift_pvc import radarsplat_xpu_backend as backend

pytestmark = pytest.mark.skipif(not (REFERENCE_ROOT / "gsplat/rendering.py").is_file(),
                                reason="pinned RadarSplat source not staged")
FALLBACK = "fallback from XPU to CPU"


@pytest.fixture
def xpu(monkeypatch):
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        pytest.skip("requires an allocated PVC card")
    monkeypatch.setenv("RIFT_ACCELERATOR", "xpu")
    return torch.device("xpu")


@pytest.fixture(scope="module")
def reference():
    return backend.load_xpu_reference(device="cpu")


def no_fallbacks(records):
    return [str(w.message) for w in records if FALLBACK in str(w.message)]


def test_mirrors_agree_between_cpu_and_xpu(xpu):
    torch.manual_seed(0)
    N, C, W, H, ts = 400, 1, 96, 64, 16
    means = torch.randn(N, 3)
    quats = torch.randn(N, 4)
    scales = torch.rand(N, 3) * 3
    viewmats = torch.eye(4)[None]
    Ks = torch.tensor([[[8.0, 0, 48.0], [0, 8.0, 32.0], [0, 0, 1]]])
    tw, th = math.ceil(W / ts), math.ceil(H / ts)
    coeffs = torch.randn(C, N, 36, 3)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        out = {}
        for dev in ("cpu", xpu):
            to = lambda t: t.to(dev)
            radii, means2d, depths, conics, comp = ops.fully_fused_projection_xpu(
                to(means), None, to(quats), to(scales), to(viewmats), to(Ks), W, H, near_plane=-10., far_plane=10.,
                calc_compensations=True, camera_model="ortho")
            tiles, ids, flat = ops.isect_tiles_xpu(means2d, radii, torch.zeros_like(depths), ts, tw, th, n_cameras=C)
            offsets = ops.isect_offset_encode_xpu(ids, C, tw, th)
            opac = torch.rand(C, N, generator=torch.Generator().manual_seed(1)).to(dev)
            gs, px, cs = ops.rasterize_to_indices_in_range_radargs_xpu(
                0, 100, torch.ones(C, H, W, device=dev), means2d, conics, opac, W, H, ts, offsets, flat)
            sh = ops.spherical_harmonics_xpu(4, means2d.new_tensor(means)[None].expand(C, -1, -1), to(coeffs), masks=radii > 0)
            out[str(dev)] = dict(radii=radii, means2d=means2d, depths=depths, conics=conics, comp=comp, tiles=tiles,
                                 ids=ids, flat=flat, offsets=offsets, gs=gs, px=px, cs=cs, sh=sh)
    assert not no_fallbacks(records), no_fallbacks(records)
    a, b = out["cpu"], out["xpu"]
    for key in ("radii", "tiles", "ids", "flat", "offsets", "gs", "px", "cs"):
        assert torch.equal(a[key], b[key].cpu()), key
    visible = a["radii"] > 0
    assert int(visible.sum()) > 0 and a["gs"].numel() > 0
    for key in ("means2d", "depths", "conics", "comp"):
        torch.testing.assert_close(a[key][visible], b[key].cpu()[visible], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(a["sh"], b["sh"].cpu(), atol=1e-5, rtol=1e-5)


def test_fused_ssim_matches_cpu_on_xpu(xpu):
    from rift_pvc.fused_ssim_torch import fused_ssim
    g = torch.Generator().manual_seed(2)
    p = torch.rand(1, 3, 40, 33, generator=g)
    t = torch.rand(1, 3, 40, 33, generator=g)
    cpu = p.clone().requires_grad_(True)
    dev = p.to(xpu).requires_grad_(True)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        a = fused_ssim(cpu, t, padding="valid")
        b = fused_ssim(dev, t.to(xpu), padding="valid")
        a.backward()
        b.backward()
    assert not no_fallbacks(records)
    assert abs(float(a) - float(b)) < 1e-6
    torch.testing.assert_close(cpu.grad, dev.grad.cpu(), atol=1e-6, rtol=1e-5)


def test_released_renderer_and_loss_on_xpu_match_cpu(xpu, reference):
    from rift_pvc.radarsplat_release import ReleasedRenderer, create_scene, release_loss
    from rift.radarsplat_b7873200 import RadarSplatGrid
    rendering, fused_ssim = reference
    grid = RadarSplatGrid(num_range_bins=16, range_resolution_m=0.1, range_start_m=9.2, azimuth_start_deg=-7.2,
                          azimuth_span_deg=14.4, output_azimuth_resolution_deg=0.9,
                          intermediate_azimuth_resolution_deg=0.1, spectral_leakage_width_m=0.7)
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([-10.0, 0.0, 0.0])
    target = torch.rand(16, 16, generator=torch.Generator().manual_seed(7))
    results = {}
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        for dev in ("cpu", xpu):
            splats, _ = create_scene(scene_scale=100, scene_center=[0, 0, 0], device=dev, num_points=2000)
            renderer = ReleasedRenderer(rendering, 100.0)
            power, occupancy = renderer(splats, pose.to(dev), grid, 5, torch.zeros(16, 16, device=dev))
            losses = release_loss(power, occupancy, target.to(dev), (target.to(dev) > .5).float(), splats, fused_ssim)
            losses["total"].backward()
            results[str(dev)] = (power.detach().cpu(), occupancy.detach().cpu(), float(losses["total"]),
                                 {k: v.grad.cpu() for k, v in splats.items()})
    assert not no_fallbacks(records), no_fallbacks(records)
    (p_cpu, o_cpu, l_cpu, g_cpu), (p_xpu, o_xpu, l_xpu, g_xpu) = results["cpu"], results["xpu"]
    assert float(p_cpu.max()) > 1e-3
    torch.testing.assert_close(p_xpu, p_cpu, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(o_xpu, o_cpu, atol=1e-4, rtol=1e-4)
    assert abs(l_xpu - l_cpu) <= 1e-5 * max(1.0, abs(l_cpu))
    largest = max(float(g.abs().max()) for g in g_cpu.values())
    for name in g_cpu:
        scale = float(g_cpu[name].abs().max())
        # 1e-3 relative per parameter group, with an absolute floor of 1e-6 of the largest gradient:
        # the initial scene is isotropic, so the quaternion gradient is float noise (~1e-11) on both devices
        assert float((g_xpu[name] - g_cpu[name]).abs().max()) <= 1e-3 * scale + 1e-6 * largest, name


def test_engine_interrupt_resume_on_xpu(xpu, tmp_path, monkeypatch, reference):
    import rift.radarsplat_release_training as engine
    import rift_pvc.radarsplat_release_training as pvc_engine
    import rift.radarsplat_release as release
    import train_radarsplat as lifecycle
    from scripts.validate_radarsplat_b7873200_native_contract import _make_synthetic_cache
    from rift.radarsplat_b7873200_protocol import load_cache
    _make_synthetic_cache(tmp_path / "cache")
    cache = load_cache(tmp_path / "cache")
    original_create = release.create_scene
    monkeypatch.setattr(engine, "create_scene", lambda **kw: original_create(**{**kw, "num_points": 64}))
    control = dict(calls=0, stop=1)
    Real = engine.ReleasedRenderer

    class Stopper(Real):
        def __call__(self, *args, **kwargs):
            control["calls"] += 1
            if control["calls"] == control["stop"]:
                signal.raise_signal(signal.SIGTERM)
            return super().__call__(*args, **kwargs)

    monkeypatch.setattr(engine, "ReleasedRenderer", Stopper)
    pvc_engine.install()

    def leg(folder, stop, resume):
        control.update(calls=0, stop=stop)
        with pytest.raises(SystemExit) as exc:
            pvc_engine.train(cache, folder, device=xpu, resume=resume, profile="budget48")
        assert exc.value.code == 143
        return lifecycle._load_checkpoint(folder / "checkpoint_latest.pt", torch.device("cpu"))

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        first = leg(tmp_path / "run", 1, False)
        resumed = leg(tmp_path / "run", 1, True)
        full = leg(tmp_path / "full", 2, False)
    assert not no_fallbacks(records), no_fallbacks(records)
    assert first["step"] == 1 and resumed["step"] == full["step"] == 2
    for key in ("splats", "optimizers", "position_scheduler", "sampler"):
        assert lifecycle._directly_equal(resumed[key], full[key]), key
    assert all(torch.isfinite(v).all() for v in resumed["splats"].values())
    record = backend.read_sidecar(tmp_path / "run")
    assert record["accelerator"]["backend"] == "xpu" and record["radarsplat_backend"] == backend.BACKEND_IDENTITY
