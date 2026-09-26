#!/usr/bin/env python3
"""Gate 1 of docs/RADARSPLAT_PVC_ADAPTATION.md: record the fork's CUDA op inputs/outputs on an H100.

Runs in the CUDA environment with the fork's compiled extension and the pinned
fused-SSIM (see ``scripts_pvc/parity_radarsplat_h100.sbatch``). On one real
``budget48`` B787 target cache (a frozen-prefix development subset prepared with
the production target spec, since the parity of an op does not depend on the
number of views) and the fresh ``create_scene`` state (seed 42, 112000
Gaussians), it renders one training view through the unchanged
``ReleasedRenderer`` twice (SH degree 0, the first production step, and degree
5, the maximum) and records, for every call of

    gsplat.rendering.fully_fused_projection / isect_tiles / isect_offset_encode / spherical_harmonics
    gsplat.cuda._wrapper.rasterize_to_indices_in_range_radargs
    gsplat.cuda._torch_impl_radar._rasterize_to_radar_pixels   (the six products)
    fused_ssim                                                  (inside release_loss)

its inputs and CUDA outputs, plus the final power/occupancy images,
``release_loss`` and its gradients w.r.t. the seven splat parameters, and
``fused_ssim`` value/gradient on a seeded random pair. ``rift_pvc/tests/test_radarsplat_parity.py``
replays the dump on PVC with the ledger's tolerances.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PRODUCTION_TARGET_SPEC = ["--grid-policy", "scene_support", "--n-azimuth", "33", "--n-elevation", "33", "--n-range", "33"]
SCHEMA = "rift_pvc_radarsplat_parity_dump_v1"


def compact(value):
    """Detached CPU copies; int64 id tensors that fit int32 are stored as int32."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().clone()
        if value.dtype == torch.int64 and value.numel() and int(value.abs().max()) < 2 ** 31 - 1:
            value = value.to(torch.int32)
        return value
    if isinstance(value, tuple):
        return tuple(compact(v) for v in value)
    if isinstance(value, list):
        return [compact(v) for v in value]
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()}
    return value


class Recorder:
    def __init__(self):
        self.calls = []
        self.pass_id = None
        self.restore = []

    def wrap(self, module, name, *, ones_arg=None):
        original = getattr(module, name)

        def wrapped(*args, **kwargs):
            started = time.perf_counter()
            out = original(*args, **kwargs)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            stored_args = list(args)
            if ones_arg is not None:  # the radar index kernel ignores transmittances (all ones): keep the shape only
                stored_args[ones_arg] = {"__ones__": list(args[ones_arg].shape)}
            self.calls.append(dict(op=name, pass_id=self.pass_id, args=compact(tuple(stored_args)),
                                   kwargs=compact(kwargs), outputs=compact(out), seconds=seconds))
            return out

        self.restore.append((module, name, original))
        setattr(module, name, wrapped)
        return original


def prepare_subset(args):
    cache_root = Path(args.cache_root)
    if (cache_root / "recipe.json").exists() and (cache_root / "stats.json").exists():
        print(f"parity: reusing target cache {cache_root}", flush=True)
        return None
    command = [sys.executable, str(ROOT / "scripts/prepare_radarsplat_b7873200_targets.py"),
               "--npz-path", args.npz_path, "--role-manifest", args.role_manifest, "--cache-root", str(cache_root),
               *PRODUCTION_TARGET_SPEC, "--max-train", str(args.max_train), "--max-validation", str(args.max_validation),
               "--device", args.device]
    print("parity: " + " ".join(command), flush=True)
    started = time.perf_counter()
    subprocess.run(command, check=True, cwd=ROOT)
    wall = time.perf_counter() - started
    views = args.max_train + args.max_validation
    print(f"parity: prepared {views} views in {wall:.1f} s ({wall / views:.2f} s/view on {args.device})", flush=True)
    return dict(views=views, seconds=wall, seconds_per_view=wall / views)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--npz-path", required=True)
    parser.add_argument("--role-manifest", required=True)
    parser.add_argument("--max-train", type=int, default=24)
    parser.add_argument("--max-validation", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", default="budget48")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    prepare_stats = prepare_subset(args)

    from rift.radarsplat_release import ReleasedPreprocessing, create_scene, load_cuda_reference, release_loss
    from rift.radarsplat_release_training import identity_for_cache, read_view, renderer_for_cache
    from rift.radarsplat_b7873200_protocol import load_cache
    import train_radarsplat as lifecycle
    device = torch.device(args.device)
    rendering, fused_ssim = load_cuda_reference(device=device)
    import gsplat.cuda._wrapper as wrapper
    import gsplat.cuda._torch_impl_radar as radar_impl
    cache = load_cache(args.cache_root)
    identity = identity_for_cache(cache, args.profile)
    started = time.perf_counter()
    splats, optimizers = create_scene(scene_scale=identity["adapter"]["initialization_scene_scale"],
                                      scene_center=np.zeros(3), device=device,
                                      num_points=identity["model_recipe"]["init_num_pts"])
    print(f"parity: scene of {identity['model_recipe']['init_num_pts']} Gaussians in {time.perf_counter() - started:.1f} s", flush=True)
    renderer = renderer_for_cache(rendering, cache, identity)
    preprocessing = ReleasedPreprocessing(cache)
    sampler = lifecycle.DeterministicViewSampler(cache.train_indices, 42)
    index = sampler.next()
    arrays, grid, target = read_view(cache, index, "train", device)
    labels, _ = preprocessing.occupancy.label(index)
    background = torch.as_tensor(preprocessing.background(arrays), device=device)
    pose = torch.as_tensor(arrays["sensor_to_world"], device=device)
    ranges = grid.range_start_m + (torch.arange(grid.num_range_bins, device=device) + .5) * grid.range_resolution_m
    target_masked = target * (ranges >= 2.5 / renderer.units)
    labels_t = torch.as_tensor(labels.copy(), device=device)

    recorder = Recorder()
    for name in ("fully_fused_projection", "isect_tiles", "isect_offset_encode", "spherical_harmonics"):
        recorder.wrap(rendering, name)
    recorder.wrap(wrapper, "rasterize_to_indices_in_range_radargs", ones_arg=2)
    recorder.wrap(radar_impl, "_rasterize_to_radar_pixels")
    ssim_calls = []

    def recorded_ssim(img1, img2, padding="same", train=True):
        out = fused_ssim(img1, img2, padding=padding, train=train)
        ssim_calls.append(dict(pass_id=recorder.pass_id, padding=padding, img1=compact(img1), img2=compact(img2),
                               value=float(out)))
        return out

    def perturbed(source):
        """Nondegenerate fixture: anisotropic log-scales, random rotations, off-plane means, nonzero SH."""
        import copy
        clone = copy.deepcopy(source)
        generator = torch.Generator().manual_seed(4242)
        with torch.no_grad():
            N = clone["means"].shape[0]
            draw = lambda *shape, scale=1.0: (torch.randn(*shape, generator=generator) * scale).to(clone["means"].device)  # noqa: E731
            clone["means"].add_(draw(N, 3, scale=2.0))
            clone["scales"].add_(draw(N, 3, scale=0.5))
            clone["quats"].copy_(draw(N, 4))
            clone["opacities"].add_(draw(N, scale=1.5))
            clone["noise_probs"].add_(draw(N, scale=1.5))
            clone["sh0"].add_(draw(N, 1, 3, scale=0.3))
            clone["shN"].copy_(draw(N, 35, 3, scale=0.1))
        return clone

    fixtures = [("degree0", splats, 0), ("degree5", splats, 5), ("perturbed_degree5", perturbed(splats), 5)]
    passes = []
    for pass_id, splats, degree in fixtures:
        recorder.pass_id = pass_id
        for parameter in splats.values():
            parameter.grad = None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        power, occupancy = renderer(splats, pose, grid, degree, background)
        torch.cuda.synchronize()
        forward = time.perf_counter() - t0
        losses = release_loss(power, occupancy, target_masked, labels_t, splats, recorded_ssim)
        t1 = time.perf_counter()
        losses["total"].backward()
        torch.cuda.synchronize()
        backward = time.perf_counter() - t1
        passes.append(dict(pass_id=recorder.pass_id, active_degree=degree, power=compact(power), occupancy=compact(occupancy),
                           losses={k: float(v) for k, v in losses.items()},
                           grads={k: compact(v.grad) for k, v in splats.items()},
                           splats={k: compact(v) for k, v in splats.items()},
                           forward_seconds=forward, backward_seconds=backward))
        print(f"parity: pass {recorder.pass_id}: forward {forward:.3f} s, backward {backward:.3f} s, "
              f"loss {float(losses['total']):.6f}, calls so far {len(recorder.calls)}", flush=True)
    for module, name, original in recorder.restore:
        setattr(module, name, original)

    # Reference self-check (added after job 2154367 on ac098 GPU de3bb0f6 returned a strict subset of the
    # correct pairs for bit-identical inputs): the CPU mirror, a proven transcription of the kernel, must
    # reproduce the first radar index call exactly, and the per-product pair counts must respect the
    # subset property alpha(opacity*reflectance) <= alpha(opacity) <= ... of the same geometry.
    from rift_pvc import gsplat_torch_ops as mirror_ops
    first = [c for c in recorder.calls if c["op"] == "rasterize_to_indices_in_range_radargs" and c["pass_id"] == "degree0"]
    counts = [int(c["outputs"][0].numel()) for c in first]
    args0 = list(first[0]["args"])
    args0[2] = torch.ones(args0[2]["__ones__"])
    gs, px, cs = mirror_ops.rasterize_to_indices_in_range_radargs_xpu(*[a.cpu() if isinstance(a, torch.Tensor) else a for a in args0])
    N = args0[3].shape[1]
    key_mirror = px * N + gs
    key_cuda = first[0]["outputs"][1].to(torch.int64) * N + first[0]["outputs"][0].to(torch.int64)
    self_check = dict(radargs_pairs_cuda=int(key_cuda.numel()), radargs_pairs_mirror=int(key_mirror.numel()),
                      only_cuda=int((~torch.isin(key_cuda, key_mirror)).sum()), only_mirror=int((~torch.isin(key_mirror, key_cuda)).sum()),
                      product_pair_counts=counts,
                      subset_property_ok=bool(counts[3] <= counts[1] and counts[4] <= counts[2] and counts[3] <= counts[0]),
                      gpu_uuid=str(getattr(torch.cuda.get_device_properties(0), "uuid", "unknown")))
    self_check["ok"] = bool(self_check["only_cuda"] == 0 and self_check["only_mirror"] == 0 and self_check["subset_property_ok"])
    print("PARITY_REFERENCE_SELF_CHECK_JSON=" + json.dumps(self_check, sort_keys=True), flush=True)

    generator = torch.Generator().manual_seed(11)
    x = torch.rand(1, 3, 64, 48, generator=generator).to(device)
    y = torch.rand(1, 3, 64, 48, generator=generator).to(device)
    ssim_random = []
    for padding in ("same", "valid"):
        img1 = x.clone().requires_grad_(True)
        value = fused_ssim(img1, y, padding=padding)
        value.backward()
        ssim_random.append(dict(padding=padding, img1=compact(x), img2=compact(y), value=float(value), grad=compact(img1.grad)))

    payload = dict(
        schema=SCHEMA,
        environment=dict(hostname=platform.node(), torch=torch.__version__, cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(0), gpu_uuid=self_check["gpu_uuid"],
                         reference_self_check=self_check, slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                         fast_math=os.environ.get("NO_FAST_MATH", "0") != "1",
                         cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
                         matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                         nvidia_tf32_override=os.environ.get("NVIDIA_TF32_OVERRIDE"),
                         created=time.strftime("%Y-%m-%d %H:%M:%S %Z")),
        source=dict(commit=identity["source_commit"], fused_ssim_commit="1272e21a282342e89537159e4bad508b19b34157"),
        cache=dict(root=str(Path(args.cache_root).resolve()), view_index=int(index), profile=args.profile,
                   train_views=len(cache.train_indices), validation_views=len(cache.validation_indices),
                   is_development_subset=bool(cache.is_development_subset), prepare=prepare_stats),
        identity=identity,
        scene=dict(seed=42, num_points=identity["model_recipe"]["init_num_pts"],
                   splats={k: compact(v) for k, v in fixtures[0][1].items()},
                   fixtures=[f[0] for f in fixtures]),
        view=dict(grid=asdict(grid), units_per_m=renderer.units, local_azimuth=renderer.local_azimuth,
                  pose=compact(pose), background=compact(background), target=compact(target),
                  target_masked=compact(target_masked), labels=compact(labels_t)),
        calls=recorder.calls,
        ssim_in_loss=ssim_calls,
        ssim_random=ssim_random,
        passes=passes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    size = args.output.stat().st_size / 2 ** 20
    summary = dict(output=str(args.output), size_mib=round(size, 1), calls=[(c["op"], c["pass_id"]) for c in recorder.calls],
                   pair_counts=[int(c["outputs"][0].numel()) for c in recorder.calls if c["op"] == "rasterize_to_indices_in_range_radargs"],
                   passes=[{k: v for k, v in p.items() if k in ("pass_id", "losses", "forward_seconds", "backward_seconds")} for p in passes],
                   environment=payload["environment"], prepare=prepare_stats)
    summary["reference_self_check"] = self_check
    print("PARITY_DUMP_SUMMARY_JSON=" + json.dumps(summary, sort_keys=True), flush=True)
    if not self_check["ok"]:
        print("PARITY_REFERENCE_SELF_CHECK_FAILED: the CUDA radar index kernel on this GPU did not reproduce the "
              "mirror's pair set; the dump is kept for diagnosis but is not a valid reference", flush=True)
        raise SystemExit(3)
    print("RADARSPLAT_PARITY_DUMP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
