"""Fixed released RadarSplat recipe over a sealed, converted power dataset.

CUDA execution uses original source. Synthetic tests may inject a renderer and
SSIM oracle into the engine; the CLI has no portable fallback or tuning flags.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import signal

import numpy as np
import torch

from rift.radarsplat_release import (SCHEMA, ReleasedPreprocessing, ReleasedRenderer,
    create_scene, load_cuda_reference, position_scheduler, recipe, release_loss, verify_reference,
    PROFILES, model_recipe, profile_from_identity, DEFAULT_INTENSITY, LINEAR_INTENSITY, intensity,
    intensity_mapping_from_identity)
from rift.radarsplat_b7873200_protocol import RECIPE_FILENAME, load_cache, load_target
from rift.radarsplat_b7873200_adapter import target_grid_from_arrays


def read_view(cache, index, role, device, intensity_mapping=LINEAR_INTENSITY):
    allowed = cache.train_indices if role == "train" else cache.validation_indices
    if role not in ("train", "validation") or index not in allowed:
        raise PermissionError("Released RadarSplat only accepts registered development roles")
    if hasattr(cache, "read_target"):
        arrays = cache.read_target(index, role)
        grid = cache.renderer_grid(index)
    else:
        arrays = load_target(cache.root, index, role, expected_grid=cache.grid)
        grid = target_grid_from_arrays(range_m=arrays["range_m"], azimuth_rad=arrays["azimuth_rad"],
                                       expected_grid=cache.grid)
    target = intensity(torch.as_tensor(arrays["radarsplat_mf_power"], device=device), cache.train_peak_power,
                       intensity_mapping)
    # Source images are bounded intensities. This is input conversion, separate
    # from the original renderer's own mandatory pre-filter output clipping.
    target = target.clamp(0, 1)
    return arrays, grid, target


def identity_for_cache(cache, profile="upstream", intensity_mapping=LINEAR_INTENSITY):
    identity = recipe(dataset_identity=cache.identity, target_recipe=cache.recipe,
                  train_peak=cache.train_peak_power, half_extent_m=float(cache.grid["scene_extent_m"]), profile=profile,
                  intensity_mapping=intensity_mapping)
    if getattr(cache, "local_azimuth", False):
        identity["adapter"]["image_sampling"] = "crop of original full-circle pixel lattice with full filter halos; unchanged source pixel kernel and stride"
    return identity


def renderer_for_cache(rendering, cache, identity):
    kwargs = {"local_azimuth": True} if getattr(cache, "local_azimuth", False) else {}
    return ReleasedRenderer(rendering, identity["adapter"]["model_units_per_m"], **kwargs)


@torch.no_grad()
def export_geometry(path, splats, identity, *, checkpoint_path, step, active_degree):
    """Same-checkpoint G48 support proxy plus the unmodified metric Gaussians."""
    import hashlib
    from scripts.eval_b787_baseline_geometry_provisional import rasterize_gaussian_occupancy_union
    from rift.radarsplat_b7873200_protocol import atomic_save_npz
    units = identity["adapter"]["model_units_per_m"]
    means = (splats["means"]/units).detach().cpu().numpy()
    scales = (splats["scales"].exp()/units).detach().cpu().numpy()
    quaternions = torch.nn.functional.normalize(splats["quats"], dim=-1).detach().cpu().numpy()
    opacity = splats["opacities"].sigmoid().detach().cpu().numpy()
    extent = identity["adapter"]["half_extent_m"]
    support, stats = rasterize_gaussian_occupancy_union(
        means, scales, quaternions, opacity, granularity=48, extent=extent, sigma_radius=3.)
    digest = hashlib.sha256()
    with Path(checkpoint_path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    atomic_save_npz(path, schema=np.asarray("rift_radarsplat_released_geometry_v1"),
        checkpoint_sha256=np.asarray(digest.hexdigest()), identity_json=np.asarray(json.dumps(identity, sort_keys=True)),
        step=np.asarray(step), active_sh_degree=np.asarray(active_degree),
        effective_sh_degree=np.asarray(min(active_degree, 4)), model_units_per_m=np.asarray(units),
        means=means, scales=scales, quaternions=quaternions, occupancy=opacity,
        noise_probability=splats["noise_probs"].sigmoid().detach().cpu().numpy(),
        support=support.astype(np.float32), grid_size=np.asarray(48), extent_m=np.asarray(extent),
        sample_centers_m=-extent+(np.arange(48)+.5)*(2*extent/48),
        support_metadata_json=np.asarray(json.dumps(stats, sort_keys=True)),
        observable=np.asarray("Gaussian occupancy-union support proxy, 3-sigma cutoff; no surface or coherent phase"))


def restore_state(restored, splats, optimizers, scheduler, sampler):
    """Validate serialized tensors and the fixed recipe before Torch can cast them."""
    import train_radarsplat as lifecycle
    step = restored.get("step")
    if type(step) is not int or not 0 <= step <= 2000:
        raise ValueError("Released-model checkpoint progress is invalid")
    raw_schedule = restored.get("position_scheduler", {})
    lr = restored.get("optimizers", {}).get("means", {}).get("param_groups", [{}])[0].get("lr")
    initial_lr = optimizers["means"].param_groups[0]["initial_lr"]
    if not isinstance(lr, (int, float)) or not np.isclose(lr, initial_lr*(.01**(step/2000)), rtol=1e-12, atol=0):
        raise ValueError("Released-model checkpoint position learning rate mismatch")
    expected_schedule = scheduler.state_dict()
    expected_schedule.update(last_epoch=step, _step_count=step+1, _last_lr=[lr])
    if not lifecycle._directly_equal(raw_schedule, expected_schedule):
        raise ValueError("Released-model checkpoint progress/scheduler mismatch")
    lifecycle._validate_model_state_for_restore(restored["splats"], splats)
    optimizers["means"].param_groups[0]["lr"] = lr
    lifecycle._validate_serialized_optimizer_states(restored["optimizers"], optimizers, checkpoint_step=step)
    # The source kernel normalizes quaternions; zero norm is not a valid scene.
    if (restored["splats"]["quats"].norm(dim=-1) == 0).any():
        raise ValueError("Released-model checkpoint has a zero quaternion")
    if not torch.isfinite(restored["splats"]["scales"].exp()).all():
        raise ValueError("Released-model checkpoint has overflowing Gaussian scales")
    lifecycle._assert_finite_value(restored.get("validation"), "released checkpoint validation")
    sampler.load_state_dict(restored["sampler"])
    splats.load_state_dict(restored["splats"], strict=True)
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(restored["optimizers"][name])
    scheduler.load_state_dict(raw_schedule)
    return step


def train(cache, output, *, device, resume=True, rendering=None, fused_ssim=None, profile="upstream",
          intensity_mapping=None):
    """``intensity_mapping`` None: a fresh run takes DEFAULT_INTENSITY, a resumed run its saved mapping."""
    import train_radarsplat as lifecycle
    output = Path(output)
    latest = output/"checkpoint_latest.pt"
    final = output/"checkpoint_final.pt"
    existing = list(output.iterdir()) if output.exists() else []
    if existing and (not resume or not latest.exists()):
        raise ValueError("Existing released-model output requires its compatible latest checkpoint")
    restored = lifecycle._load_checkpoint(latest, device) if latest.exists() else None
    if intensity_mapping is None:
        intensity_mapping = (DEFAULT_INTENSITY if restored is None else
                             intensity_mapping_from_identity(restored.get("identity") or {}))
    identity = identity_for_cache(cache, profile, intensity_mapping)
    if restored is not None and (restored.get("schema") != SCHEMA or restored.get("identity") != identity):
        raise ValueError("Released-model checkpoint identity mismatch")
    if restored is not None and not lifecycle._directly_equal(restored.get("acquisition_record"), cache.acquisition_record):
        raise ValueError("Released-model checkpoint acquisition calibration mismatch")
    if rendering is None or fused_ssim is None:
        rendering, fused_ssim = load_cuda_reference(device=device)
    splats, optimizers = create_scene(scene_scale=identity["adapter"]["initialization_scene_scale"],
                                     scene_center=np.zeros(3), device=device, num_points=identity["model_recipe"]["init_num_pts"])
    scheduler = position_scheduler(optimizers)
    sampler = lifecycle.DeterministicViewSampler(cache.train_indices, 42)
    step, metrics = 0, None
    if restored is not None:
        step = restore_state(restored, splats, optimizers, scheduler, sampler)
        metrics = restored.get("validation")
    renderer = renderer_for_cache(rendering, cache, identity)
    preprocessing = ReleasedPreprocessing(cache, intensity_mapping=intensity_mapping)
    output.mkdir(parents=True, exist_ok=True)
    lifecycle.atomic_write_json(output/"recipe.json", identity)
    stop = [False]
    old_handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    def request_stop(*_):
        stop[0] = True
    def save(path):
        lifecycle._atomic_torch_save(path, dict(schema=SCHEMA, identity=identity, step=step,
            splats=splats.state_dict(), optimizers={k:v.state_dict() for k,v in optimizers.items()},
            position_scheduler=scheduler.state_dict(), sampler=sampler.state_dict(), validation=metrics,
            acquisition_record=cache.acquisition_record))
    for s in old_handlers:
        signal.signal(s, request_stop)
    try:
        while step < 2000:
            index = sampler.next()
            arrays, grid, target = read_view(cache, index, "train", device, intensity_mapping)
            labels, _ = preprocessing.occupancy.label(index)
            background = torch.as_tensor(preprocessing.background(arrays), device=device)
            power, occupancy = renderer(splats, torch.as_tensor(arrays["sensor_to_world"], device=device),
                                        grid, min(step//200, 5), background)
            ranges = grid.range_start_m+(torch.arange(grid.num_range_bins, device=device)+.5)*grid.range_resolution_m
            target = target*(ranges >= 2.5/renderer.units)
            losses = release_loss(power, occupancy, target, torch.as_tensor(labels.copy(), device=device), splats, fused_ssim)
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("Released RadarSplat loss is nonfinite; no automatic recipe-changing fallback")
            losses["total"].backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in splats.values()):
                raise RuntimeError("Released RadarSplat produced nonfinite gradients")
            for optimizer in optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            if any(not torch.isfinite(p).all() for p in splats.values()):
                raise RuntimeError("Released RadarSplat produced nonfinite parameters")
            # The demo's refine_stop_iter=0 makes strategy updates a no-op;
            # never add our previous independent opacity pruning here.
            if step % 100 == 0 or stop[0]:
                save(latest)
                print(json.dumps(dict(step=step, **{k:float(v.detach()) for k,v in losses.items()})), flush=True)
            if stop[0]:
                raise SystemExit(143)
        metrics = evaluate(cache, splats, renderer, preprocessing, device, intensity_mapping=intensity_mapping)
        save(final)
        save(latest)
        if profile == "budget48":
            export_geometry(output/"geometry_g48.npz", splats, identity,
                            checkpoint_path=final, step=step, active_degree=5)
        lifecycle.atomic_write_json(output/"summary.json", dict(identity=identity, step=step, validation=metrics))
    finally:
        for s, handler in old_handlers.items():
            signal.signal(s, handler)
    return dict(step=step, validation=metrics, checkpoint=str(final))


@torch.no_grad()
def evaluate(cache, splats, renderer, preprocessing, device, *, role="validation", active_degree=5,
             intensity_mapping=LINEAR_INTENSITY):
    if role not in ("train", "validation"):
        raise PermissionError("Reserved test is not exposed by the released-model readout")
    indices = cache.train_indices if role == "train" else cache.validation_indices
    error, energy, pixels = 0., 0., 0
    for index in indices:
        arrays, grid, target = read_view(cache, index, role, device, intensity_mapping)
        power, _ = renderer(splats, torch.as_tensor(arrays["sensor_to_world"], device=device), grid, active_degree,
                            torch.as_tensor(preprocessing.background(arrays), device=device))
        if not torch.isfinite(power).all():
            raise RuntimeError("Nonfinite released-model evaluation output")
        ranges = grid.range_start_m+(torch.arange(grid.num_range_bins, device=device)+.5)*grid.range_resolution_m
        target = target*(ranges >= 2.5/renderer.units)
        error += float((power-target).square().sum())
        energy += float(target.square().sum())
        pixels += target.numel()
    metrics = dict(native_clipped_power_mse=error/max(pixels, 1),
                   native_clipped_power_relative_mse=error/max(energy, 1e-30), views=len(indices))
    if intensity_mapping != LINEAR_INTENSITY:
        # Same keys (GOTCHA validates the summary schema); the errors are in the mapped intensity.
        metrics["intensity_mapping"] = intensity_mapping
    return metrics


@torch.no_grad()
def readout(checkpoint, *, checkpoint_path, cache_root, device, role, object_name=None, geometry_path=None, cache=None):
    """Render the saved source recipe and export the same Gaussians in metres."""
    import hashlib
    import train_radarsplat as lifecycle
    identity = checkpoint.get("identity", {})
    profile = profile_from_identity(identity)
    if identity.get("model_recipe") != model_recipe(profile):
        raise ValueError("Released-model readout model recipe mismatch before target reads")
    target_recipe = json.loads((Path(cache_root)/RECIPE_FILENAME).read_text())
    if (checkpoint.get("schema") != SCHEMA or identity.get("target_recipe") != target_recipe
            or identity.get("dataset_identity") != target_recipe.get("sealed_protocol_identity")):
        raise ValueError("Released-model checkpoint/cache mismatch before target reads")
    if role not in ("train", "validation"):
        raise PermissionError("Reserved test is not exposed by the released-model readout")
    if geometry_path is not None and Path(geometry_path).exists():
        raise FileExistsError(geometry_path)
    if object_name is not None:
        from rift.rift_dataset import object_identity, validate_checkpoint_object
        validate_checkpoint_object({"sealed_protocol_identity": identity["dataset_identity"]}, object_identity(object_name))
    rendering, _ = load_cuda_reference(device=device)
    cache = load_cache(cache_root) if cache is None else cache
    intensity_mapping = intensity_mapping_from_identity(identity)
    if (identity != identity_for_cache(cache, profile, intensity_mapping)
            or not lifecycle._directly_equal(checkpoint.get("acquisition_record"), cache.acquisition_record)):
        raise ValueError("Released-model readout recipe/calibration mismatch")
    splats, optimizers = create_scene(scene_scale=identity["adapter"]["initialization_scene_scale"],
                                     scene_center=np.zeros(3), device=device, num_points=identity["model_recipe"]["init_num_pts"])
    scheduler = position_scheduler(optimizers)
    sampler = lifecycle.DeterministicViewSampler(cache.train_indices, 42)
    step = restore_state(checkpoint, splats, optimizers, scheduler, sampler)
    if step < 1:
        raise ValueError("Readout requires at least one completed update")
    active_degree = min((step-1)//200, 5)
    units = identity["adapter"]["model_units_per_m"]
    metrics = evaluate(cache, splats, renderer_for_cache(rendering, cache, identity),
                       ReleasedPreprocessing(cache, intensity_mapping=intensity_mapping),
                       device, role=role, active_degree=active_degree, intensity_mapping=intensity_mapping)
    # Provenance binds metric and geometry outputs to exactly one saved scene.
    digest = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()
    if geometry_path is not None:
        export_geometry(geometry_path, splats, identity, checkpoint_path=checkpoint_path,
                        step=step, active_degree=active_degree)
    return dict(schema="rift_radarsplat_released_readout_v1", checkpoint=str(checkpoint_path),
                checkpoint_sha256=digest, step=step, role=role, identity=identity, metrics=metrics,
                geometry=None if geometry_path is None else str(geometry_path),
                observable=("clipped normalized real power" if intensity_mapping == LINEAR_INTENSITY else
                            "60 dB log-mapped train-peak-relative power")
                + "; same-checkpoint metric Gaussian occupancy")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fidelity-profile", choices=tuple(PROFILES), default="upstream")
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    # Introspection exposes effective source values, without accepting tuning
    # flags that could silently change the comparison recipe.
    args.steps, args.init_num_gaussians = 2000, PROFILES[args.fidelity_profile]
    return args


def main(argv=None):
    args = parse_args(argv)
    verify_reference()
    latest = args.checkpoint_dir/"checkpoint_latest.pt"
    if latest.exists():
        import train_radarsplat as lifecycle
        saved = lifecycle._load_checkpoint(latest, torch.device("cpu"))
        target_recipe = json.loads((args.cache_root/RECIPE_FILENAME).read_text())
        if (saved.get("schema") != SCHEMA or saved.get("identity", {}).get("target_recipe") != target_recipe
                or saved.get("identity", {}).get("dataset_identity") != target_recipe.get("sealed_protocol_identity")
                or saved.get("identity", {}).get("model_recipe") != model_recipe(args.fidelity_profile)):
            raise ValueError("Released-model checkpoint/cache mismatch before target reads")
    rendering, ssim = load_cuda_reference(device=args.device)
    cache = load_cache(args.cache_root)
    if cache.is_development_subset:
        raise ValueError("The released comparison recipe requires complete registered development roles")
    return train(cache, args.checkpoint_dir, device=torch.device(args.device), resume=args.resume,
                 rendering=rendering, fused_ssim=ssim, profile=args.fidelity_profile)
