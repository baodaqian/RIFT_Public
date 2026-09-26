"""Gate 1 replay (docs/RADARSPLAT_PVC_ADAPTATION.md section 4): the torch mirrors on PVC against the
fork's CUDA ops recorded on an H100 by scripts_pvc/parity_radarsplat_dump.py.

The dump is found through ``RIFT_PVC_RADARSPLAT_PARITY_DUMP`` or as the newest
``radarsplat_parity_h100_*.pt`` under the Package F parity root; without one the
module is skipped. Tolerances: projection/SH/intersections 1e-5 relative;
rasterized products 1e-4 relative per pixel and identical cutoff masks (up to
candidates within fast-math rounding of 1/255, counted and bounded); loss
values 1e-5; gradients 1e-3 relative. A JSON summary is written next to the dump.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import pytest
import torch

from rift.radarsplat_release import REFERENCE_ROOT
from rift_pvc import gsplat_torch_ops as ops
from rift_pvc import radarsplat_xpu_backend as backend

PARITY_ROOT = Path("/scratch/group/p.cis261724.000/RIFT_pvc_runs/packageF/parity")


def _dump_path():
    explicit = os.environ.get("RIFT_PVC_RADARSPLAT_PARITY_DUMP")
    if explicit:
        return Path(explicit)
    candidates = sorted(glob.glob(str(PARITY_ROOT / "radarsplat_parity_h100_*.pt")), key=os.path.getmtime)
    return Path(candidates[-1]) if candidates else None


DUMP = _dump_path()
pytestmark = [
    pytest.mark.skipif(not (REFERENCE_ROOT / "gsplat/rendering.py").is_file(), reason="pinned source not staged"),
    pytest.mark.skipif(DUMP is None or not DUMP.is_file(), reason="no H100 parity dump (scripts_pvc/parity_radarsplat_h100.sbatch)"),
]
SUMMARY = {}


@pytest.fixture(scope="module")
def dump():
    try:
        payload = torch.load(DUMP, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001  (older torch without the allow-list of nested containers)
        payload = torch.load(DUMP, map_location="cpu", weights_only=False)
    assert payload["schema"] == "rift_pvc_radarsplat_parity_dump_v1"
    return payload


@pytest.fixture(scope="module", params=["cpu", "xpu"])
def device(request):
    if request.param == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        pytest.skip("requires an allocated PVC card")
    return torch.device(request.param)


@pytest.fixture(scope="module")
def reference():
    return backend.load_xpu_reference(device="cpu")


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        if "__ones__" in value:
            return torch.ones(value["__ones__"], device=device)
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move(v, device) for v in value)
    return value


def calls(dump, op):
    return [c for c in dump["calls"] if c["op"] == op]


def rel_max(a, b):
    scale = float(b.abs().max())
    return float((a - b).abs().max()) / max(scale, 1e-12), scale


def record(key, value):
    SUMMARY[key] = value


@pytest.fixture(scope="module", autouse=True)
def write_summary():
    yield
    if SUMMARY:
        path = DUMP.with_suffix(".replay.json")
        path.write_text(json.dumps(SUMMARY, indent=2, sort_keys=True) + "\n")


def test_projection_calls(dump, device):
    for n, call in enumerate(calls(dump, "fully_fused_projection")):
        args, kwargs = move(call["args"], device), move(call["kwargs"], device)
        radii, means2d, depths, conics, comp = ops.fully_fused_projection_xpu(*args, **kwargs)
        r_radii, r_means2d, r_depths, r_conics, r_comp = call["outputs"]
        radii, r_radii = radii.cpu(), r_radii.to(torch.int32)
        mismatch = int((radii != r_radii).sum())
        visible = r_radii > 0
        stats = dict(visible=int(visible.sum()), radii_mismatch=mismatch)
        for name, a, b in (("means2d", means2d, r_means2d), ("depths", depths, r_depths), ("conics", conics, r_conics)):
            stats[name] = rel_max(a.cpu()[visible], b[visible])[0]
        record(f"projection[{n}]@{device}", stats)
        assert mismatch <= max(1, int(1e-5 * radii.numel())), stats
        for name in ("means2d", "depths", "conics"):
            assert stats[name] <= 1e-5, stats
        assert (comp is None) == (r_comp is None)


def test_intersection_calls(dump, device):
    for n, call in enumerate(calls(dump, "isect_tiles")):
        args, kwargs = move(call["args"], device), move(call["kwargs"], device)
        tiles, ids, flat = ops.isect_tiles_xpu(*args, **kwargs)
        r_tiles, r_ids, r_flat = call["outputs"]
        stats = dict(n_isects=int(ids.numel()), tiles_equal=bool(torch.equal(tiles.cpu(), r_tiles.to(torch.int32))),
                     ids_equal=bool(torch.equal(ids.cpu(), r_ids.to(torch.int64))),
                     flatten_equal=bool(torch.equal(flat.cpu().to(torch.int64), r_flat.to(torch.int64))))
        record(f"isect_tiles[{n}]@{device}", stats)
        assert stats["tiles_equal"] and stats["ids_equal"] and stats["flatten_equal"], stats
    for n, call in enumerate(calls(dump, "isect_offset_encode")):
        args, kwargs = move(call["args"], device), move(call["kwargs"], device)
        offsets = ops.isect_offset_encode_xpu(*args, **kwargs)
        equal = bool(torch.equal(offsets.cpu(), call["outputs"].to(torch.int32)))
        record(f"isect_offset_encode[{n}]@{device}", dict(equal=equal, shape=list(offsets.shape)))
        assert equal


def test_spherical_harmonics_calls(dump, device):
    for n, call in enumerate(calls(dump, "spherical_harmonics")):
        args, kwargs = move(call["args"], device), move(call["kwargs"], device)
        colors = ops.spherical_harmonics_xpu(*args, **kwargs).cpu()
        masks = kwargs.get("masks", args[3] if len(args) > 3 else None)
        masks = masks.cpu() if masks is not None else torch.ones(colors.shape[:-1], dtype=torch.bool)
        rel, scale = rel_max(colors[masks], call["outputs"][masks])
        record(f"spherical_harmonics[{n}]@{device}", dict(degree=int(args[0]), rel_max=rel, scale=scale, masked=int(masks.sum())))
        assert rel <= 1e-5


def test_radar_index_calls(dump, device):
    for n, call in enumerate(calls(dump, "rasterize_to_indices_in_range_radargs")):
        args = move(call["args"], device)
        gs, px, cs = ops.rasterize_to_indices_in_range_radargs_xpu(*args)
        r_gs, r_px, r_cs = (t.to(torch.int64) for t in call["outputs"])
        means2d, conics, opacities = args[3], args[4], args[5]
        C, N = means2d.shape[:2]
        W, H = int(args[6]), int(args[7])
        key = lambda g, p, c: (c * H * W + p) * N + g  # noqa: E731
        mine = key(gs.cpu(), px.cpu(), cs.cpu())
        theirs = key(r_gs, r_px, r_cs)
        only_mine = mine[~torch.isin(mine, theirs)]  # (camera, pixel, gaussian) keys are unique per call
        only_theirs = theirs[~torch.isin(theirs, mine)]
        differing = torch.cat([only_mine, only_theirs])
        stats = dict(pairs_cuda=int(theirs.numel()), pairs_mirror=int(mine.numel()), only_mirror=int(only_mine.numel()),
                     only_cuda=int(only_theirs.numel()), order_identical=bool(mine.numel() == theirs.numel() and torch.equal(mine, theirs)))
        if differing.numel():
            g = differing % N
            pix = (differing // N) % (H * W)
            cam = (differing // N) // (H * W)
            flat = cam * N + g
            xy = means2d.reshape(-1, 2).cpu()[flat]
            cn = conics.reshape(-1, 3).cpu()[flat]
            op = opacities.reshape(-1).cpu()[flat]
            dx = xy[:, 0] - ((pix % W).float() + 0.5)
            dy = xy[:, 1] - ((pix // W).float() + 0.5)
            sigma = 0.5 * (cn[:, 0] * dx * dx + cn[:, 2] * dy * dy) + cn[:, 1] * dx * dy
            alpha = torch.clamp_max(op * torch.exp(-sigma), 0.999)
            distance = ((alpha - 1 / 255).abs() / (1 / 255))
            stats["max_relative_distance_to_cutoff"] = float(distance.max())
            stats["sigma_min"] = float(sigma.min())
        record(f"radargs[{n}]@{device}", stats)
        assert differing.numel() <= max(2, int(1e-5 * theirs.numel())), stats
        if differing.numel():
            assert stats["max_relative_distance_to_cutoff"] <= 2e-3, stats  # fast-math __expf vs IEEE exp at the cutoff


def test_product_images(dump, device, reference):
    backend.load_xpu_reference(device="cpu")
    from gsplat.cuda._torch_impl_radar import _rasterize_to_radar_pixels
    for n, call in enumerate(calls(dump, "_rasterize_to_radar_pixels")):
        args, kwargs = move(call["args"], device), move(call["kwargs"], device)
        image = _rasterize_to_radar_pixels(*args, **kwargs).cpu()
        expected = call["outputs"]
        diff = (image - expected).abs()
        rel = diff / expected.abs().clamp_min(1e-6)
        violating = ~((diff <= 1e-6) | (rel <= 1e-4))
        # A candidate pair whose alpha sits on the 1/255 cutoff within float rounding can be kept by one
        # side only (IEEE exp vs the kernel's __expf); it changes one pixel's raw sum by about 1/255.
        # Such pixels are counted and bounded, never hidden.
        stats = dict(shape=list(image.shape), max_abs=float(diff.max()), max_rel_floor_1e6=float(rel.max()),
                     nonzero=int((expected > 0).sum()), cutoff_pixels=int(violating.sum()),
                     cutoff_pixels_max_abs=float(diff[violating].max()) if bool(violating.any()) else 0.0)
        record(f"products[{n}]@{device}", stats)
        assert stats["cutoff_pixels"] <= 2 and stats["cutoff_pixels_max_abs"] <= 1 / 255 + 1e-4, stats


def test_full_render_loss_and_gradients(dump, device, reference):
    from rift.radarsplat_b7873200 import RadarSplatGrid
    from rift_pvc.radarsplat_release import ReleasedRenderer, release_loss
    from rift_pvc.fused_ssim_torch import fused_ssim
    rendering, _ = reference
    view = dump["view"]
    grid = RadarSplatGrid(**view["grid"])
    renderer = ReleasedRenderer(rendering, view["units_per_m"], local_azimuth=view["local_azimuth"])
    for pass_ in dump["passes"]:
        state = pass_.get("splats", dump["scene"]["splats"])  # per-pass fixture (the perturbed pass has its own scene)
        splats = torch.nn.ParameterDict({k: torch.nn.Parameter(v.to(device).clone()) for k, v in state.items()})
        power, occupancy = renderer(splats, view["pose"].to(device), grid, pass_["active_degree"], view["background"].to(device))
        losses = release_loss(power, occupancy, view["target_masked"].to(device), view["labels"].to(device), splats, fused_ssim)
        losses["total"].backward()
        stats = {}
        for name, mine, theirs in (("power", power, pass_["power"]), ("occupancy", occupancy, pass_["occupancy"])):
            mine = mine.detach().cpu()
            diff = (mine - theirs).abs()
            rel = diff / theirs.abs().clamp_min(1e-6)
            # final images are unit-range (clamped to [0, 1]); a filtered cutoff flip spreads to ~3e-5 absolute
            stats[name] = dict(max_abs=float(diff.max()), max_rel=float(rel.max()), ok=bool(((diff <= 1e-4) | (rel <= 1e-4)).all()))
        stats["losses"] = {k: dict(mirror=float(v), cuda=pass_["losses"][k],
                                   rel=abs(float(v) - pass_["losses"][k]) / max(abs(pass_["losses"][k]), 1e-12))
                           for k, v in losses.items()}
        stats["grads"] = {}
        largest = max(float(g.abs().max()) for g in pass_["grads"].values())
        for name, parameter in splats.items():
            mine, theirs = parameter.grad.cpu(), pass_["grads"][name]
            rel, scale = rel_max(mine, theirs)
            per_gaussian = (mine - theirs).abs().reshape(len(theirs), -1).max(1).values / max(scale, 1e-12)
            # A group whose reference gradient is below 1e-6 of the largest group is a zero gradient
            # (e.g. quaternions of the isotropic initial scene): both sides hold rounding noise, so it is
            # checked absolutely against that floor instead of relatively.
            zero_group = scale < 1e-6 * largest
            stats["grads"][name] = dict(rel_max=rel, scale=scale, zero_group=bool(zero_group),
                                        mine_max=float(mine.abs().max()),
                                        l2_rel=float((mine - theirs).norm() / theirs.norm()) if float(theirs.norm()) > 0 else 0.0,
                                        outliers_above_1e3=int((per_gaussian > 1e-3).sum()), gaussians=int(len(theirs)))
        record(f"pass[{pass_['pass_id']}]@{device}", stats)
        assert stats["power"]["ok"] and stats["occupancy"]["ok"], stats
        for name, entry in stats["losses"].items():
            assert entry["rel"] <= 1e-5, (name, entry)
        # Gradient gate: 1e-3 relative in L2 over each parameter group. The max norm is recorded too; a
        # handful of Gaussians sitting on one of the fork's own thresholds (alpha clamp 0.999, 1/255
        # cutoff, product clamp) can flip a pixel term under fp32 rounding and are counted, not hidden.
        for name, entry in stats["grads"].items():
            if entry["zero_group"]:
                assert entry["mine_max"] <= 1e-6 * largest, (name, entry)
                continue
            assert entry["l2_rel"] <= 1e-3, (name, entry)
            assert entry["outliers_above_1e3"] <= max(3, entry["gaussians"] // 20000), (name, entry)


def test_fused_ssim_random_pair(dump, device):
    from rift_pvc.fused_ssim_torch import fused_ssim
    for entry in dump["ssim_random"]:
        img1 = entry["img1"].to(device).clone().requires_grad_(True)
        value = fused_ssim(img1, entry["img2"].to(device), padding=entry["padding"])
        value.backward()
        stats = dict(padding=entry["padding"], value_abs=abs(float(value) - entry["value"]),
                     grad_abs=float((img1.grad.cpu() - entry["grad"]).abs().max()))
        record(f"fused_ssim[{entry['padding']}]@{device}", stats)
        assert stats["value_abs"] <= 1e-6 and stats["grad_abs"] <= 1e-6, stats
    for n, entry in enumerate(dump["ssim_in_loss"]):
        value = fused_ssim(entry["img1"].to(device), entry["img2"].to(device), padding=entry["padding"])
        rel = abs(float(value) - entry["value"]) / max(abs(entry["value"]), 1e-12)
        record(f"ssim_in_loss[{n}]@{device}", dict(rel=rel, value=float(value), cuda=entry["value"]))
        assert rel <= 1e-5
