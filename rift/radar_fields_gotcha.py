"""Native GOTCHA adapter for the maintained Radar Fields comparison profile.

This shares the source-adapted-v3 model/loss with the RIFT collection, not
the source-format Navtech Trainer in radar_fields_released.py. Each native
pulse is a monostatic range profile; pass/sector identities are optimizer
viewpoints. No pulse is turned into a synthetic MIMO channel or averaged with
another pulse. Frequencies and effective reference ranges remain native.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
import signal
from types import SimpleNamespace

import numpy as np
import torch

from .gotcha_dataset import C, validate_checkpoint as validate_dataset_checkpoint
from .gotcha_training import atomic_json, atomic_save
from .radar_fields import radar_fields_intensity, NORMALIZED_DB_INTENSITY_DOMAIN
from .radar_fields_dataset import normalize_power_db
from .radar_fields_native import occupancy_probability, prepare_bistatic_bins, render_bistatic_batch, released_batch_loss
from .radar_fields_recipe import recipe_contract, SOURCE_RECIPE, AUDITED_RECIPE
from .radar_fields_pose import RELEASE_SE3, build_pose_heads, with_pose_refinement
from .radar_fields_upstream import check_model_backend, original_module


SCHEMA = "gotcha_radar_fields_v3"
DEFAULTS = dict(
    profile=SOURCE_RECIPE, model_backend="upstream-tcnn", steps=800, view_batch=10, seed=0,
    lr=.001, eval_every=200, checkpoint_every=200, ray_samples=10,
    train_profiles=100,
    dynamic_range_db=60., query_chunk=32768, range_guard_cells=2,
    weight_fft=.60, weight_occ=.36, weight_bimodal=.03,
    occupancy_noise_multiplier=1.5, occupancy_decay_bins=10.,
    occupancy_probability_offset=-.15, occupancy_probability_scale=2.,
    hidden_dim=64, feature_dim=32, sh_degree=3, sigmoid_tightness=1.,
    no_batch_norm=False, hash_levels=16, hash_features=2,
    hash_base_resolution=16, hash_final_resolution=512, hash_log2_size=19,
    # User decision 2 (2026-09-22): the release's pose refinement, per pass-sector
    # (rift/radar_fields_pose.py). "disabled" reproduces the earlier recipe exactly.
    pose_refinement=RELEASE_SE3,
)


def recipe_from_config(config, extent, train_view_count=2000):
    if not isinstance(config, dict) or set(config) - set(DEFAULTS):
        raise ValueError("Unknown Radar Fields GOTCHA configuration field")
    controls = {**DEFAULTS, **config}
    pose_refinement = controls.pop("pose_refinement")
    if controls["profile"] not in (SOURCE_RECIPE, AUDITED_RECIPE):
        raise ValueError("Unknown RF GOTCHA profile")
    if controls["profile"] == SOURCE_RECIPE:
        batches = math.ceil(train_view_count / 10)
        epochs = math.ceil(800 / batches)
        fixed = dict(steps=batches*epochs, view_batch=10, seed=0, ray_samples=10,
                     train_profiles=100, lr=.001, weight_fft=.6, weight_occ=.36, weight_bimodal=.03)
        if train_view_count != 2000:
            fixed.update(eval_every=batches, checkpoint_every=batches)
        if epochs < 2 or train_view_count % 10:
            raise ValueError("Released schedule requires >=2 full epochs and complete frame batches")
        for key, value in fixed.items():
            if key in config and config[key] != value:
                raise ValueError(f"source-adapted-v3 fixes {key}={value}; select audited-v2 for engineering checks")
            controls[key] = value
    integer_keys = ("steps", "view_batch", "seed", "eval_every", "checkpoint_every",
                    "ray_samples", "query_chunk", "range_guard_cells", "hidden_dim",
                    "feature_dim", "sh_degree", "hash_levels", "hash_features",
                    "hash_base_resolution", "hash_final_resolution", "hash_log2_size", "train_profiles")
    for key in integer_keys:
        value = controls[key]
        minimum = 0 if key in ("seed", "range_guard_cells", "sh_degree") else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"Invalid Radar Fields {key}")
    if type(controls["no_batch_norm"]) is not bool:
        raise ValueError("no_batch_norm must be boolean")
    for key, value in controls.items():
        if key in integer_keys or key in ("profile", "model_backend", "no_batch_norm"):
            continue
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError(f"Invalid Radar Fields {key}")
        if key != "occupancy_probability_offset" and value <= 0:
            raise ValueError(f"Radar Fields {key} must be positive")
    if not math.isfinite(extent) or extent <= 0:
        raise ValueError("Invalid GOTCHA scene extent")
    args = SimpleNamespace(**controls, recipe=controls["profile"], extent=float(extent))
    return with_pose_refinement(dict(schema=SCHEMA, method="radar_fields", controls=controls,
                extent_m=float(extent), model_recipe=recipe_contract(args),
                target="exact_native_frequency_matched_range_power_per_pulse",
                range_grid="ROI_center_anchored_Rayleigh_spacing_guarded_cube_interval_per_native_pulse",
                occupancy="per_pulse_ROI_range_median_then_released_Bayesian_recurrence",
                normalization="all_train_pulses_peak_per_polarization_fixed_db",
                intensity_offset=1., intensity_scaler=1., range_law="released",
                metric_domain=NORMALIZED_DB_INTENSITY_DOMAIN,
                angular_sampling="unit_gain_scene_cap_NOT_measured_GOTCHA_antenna",
                exposure=("released_100_native_pulse_samples_with_replacement_all_eligible" if controls["profile"] == SOURCE_RECIPE
                          else "all_native_pulses_in_each_selected_pass_sector"),
                fidelity="source_model_and_losses_with_declared_acquisition_adaptations"), pose_refinement)


def model_args(recipe, device):
    return SimpleNamespace(**recipe["controls"], recipe=recipe["controls"]["profile"],
                           extent=recipe["extent_m"], device=str(device))


def range_geometry(observation, region, guard_cells=2, device="cpu"):
    """Metadata-only ranges and local position; no regularized frequency grid."""
    antenna = np.asarray(region.to_local(observation.position_m), dtype=np.float64)
    frequencies = np.asarray(observation.frequencies_hz, dtype=np.float64)
    if (frequencies.ndim != 1 or len(frequencies) < 2
            or not np.isfinite(frequencies).all() or np.any(np.diff(frequencies) <= 0)
            or antenna.shape != (3,) or not np.isfinite(antenna).all()):
        raise ValueError("Invalid native RF pulse geometry/frequencies")
    extent = float(region.half_extent_m)
    step = C / (2 * (frequencies[-1] - frequencies[0]))
    near = float(np.linalg.norm(antenna - np.clip(antenna, -extent, extent)))
    far = float(np.linalg.norm(np.abs(antenna) + extent))
    # Nonuniform-frequency GOTCHA has no existing FFT lattice to inherit.
    # Anchor the readout at the registered ROI center so a small ROI is not
    # represented only by a tangent surface with no interior query samples.
    center = float(np.linalg.norm(antenna))
    first = math.floor((near-center)/step) - guard_cells
    last = math.ceil((far-center)/step) + guard_cells
    start = center + first * step
    count = last - first + 1
    if start <= 0 or count >= len(frequencies):
        raise ValueError("GOTCHA ROI exceeds the native RF range-profile support")
    ranges = start + torch.arange(count, dtype=torch.float64, device=device) * step
    return torch.as_tensor(antenna, dtype=torch.float64, device=device), ranges


def matched_range_power(observation, ranges, *, chunk=128):
    """|sum_f S(f) exp(+i4pi f(r-r0)/c) / Nf|², at exact input frequencies.

    A constant carrier phase can be factored out after taking magnitude, so
    centered native frequencies improve conditioning without changing power.
    The native reader has already applied each channel's own autofocus once.
    """
    f = torch.as_tensor(np.array(observation.frequencies_hz, copy=True),
                        dtype=torch.float64, device=ranges.device)
    y = torch.as_tensor(np.array(observation.response, copy=True),
                        dtype=torch.complex128, device=ranges.device)
    if y.shape != f.shape or not torch.isfinite(y).all():
        raise ValueError("Invalid native GOTCHA response")
    centered = f - f.mean()
    result = []
    for block in ranges.split(chunk):
        phase = (4 * math.pi / C) * (block[:, None] - observation.reference_range_m) * centered[None, :]
        result.append(((torch.exp(1j * phase) @ y) / len(f)).abs().square())
    return torch.cat(result)


def training_statistics(dataset, recipe, stopped=lambda: False):
    """Every selected TRAIN pulse, independently for each polarization head."""
    peaks, counts = {}, {}
    for pol in dataset.polarizations:
        peak, count = 0., 0
        for p, sector in dataset.viewpoints("train"):
            for obs in dataset.observations(p, sector, pol):
                if stopped():
                    return None
                _, ranges = range_geometry(obs, dataset.region, recipe["controls"]["range_guard_cells"])
                peak = max(peak, float(matched_range_power(obs, ranges).max()))
                count += 1
        if peak <= 0 or count == 0:
            raise ValueError("Radar Fields normalization requires nonzero training power")
        peaks[pol], counts[pol] = peak, count
    return dict(dataset_identity=dataset.identity, train_viewpoints=[list(v) for v in dataset.viewpoints("train")],
                pulse_counts=counts, peak_power=peaks,
                dynamic_range_db=recipe["controls"]["dynamic_range_db"], role="train")


def prepare_frame(dataset, view, pol, recipe, stats, device, *, training=False, pose=None):
    """All actual pulses, each with its own geometry and ragged native grid.

    ``pose`` (a SectorPoses head) moves the sensor, never the measured range bins.
    """
    controls = recipe["controls"]
    correction = None if pose is None else pose.correction(view)
    result = []
    observations = dataset.observations(*view, pol)
    if training and controls["profile"] == SOURCE_RECIPE:
        # Keep every native pulse eligible, exactly as original RF keeps every
        # azimuth eligible but samples 100 profiles for a training frame.
        observations = list(observations)
        selected = original_module("radarfields.sampler").get_azimuths(
            1, controls["train_profiles"], len(observations), device).cpu().tolist()[0]
        observations = [observations[i] for i in selected]
    for obs in observations:
        antenna, ranges = range_geometry(obs, dataset.region, controls["range_guard_cells"], device)
        power = matched_range_power(obs, ranges).float()[None, :]
        if correction is not None:
            antenna = pose.apply(view, antenna, correction)
        target = normalize_power_db(power, stats["peak_power"][pol], controls["dynamic_range_db"])
        occ = occupancy_probability(target, noise_multiplier=controls["occupancy_noise_multiplier"],
            decay_bins=controls["occupancy_decay_bins"],
            probability_offset=controls["occupancy_probability_offset"],
            probability_scale=controls["occupancy_probability_scale"])
        if controls["profile"] == SOURCE_RECIPE:
            target = target * (target > .1525)
        geometry = prepare_bistatic_bins(antenna.detach()[None], antenna.detach()[None], ranges,
            extent=recipe["extent_m"], ray_samples=controls["ray_samples"],
            source_sampling=controls["profile"] == SOURCE_RECIPE)
        if antenna.requires_grad:
            # Monostatic samples are antenna + range * direction, so they move rigidly with
            # the corrected sensor. Pointing (look-at-ROI, drawn directions) is recomputed from
            # the corrected position without a gradient: its off-axis acos is singular at 0.
            geometry["xyz"] = geometry["xyz"] + (antenna - antenna.detach())
        result.append(dict(geometry=geometry, ranges=ranges, target=target, occupancy_target=occ))
    if not result:
        raise ValueError("Empty GOTCHA pass-sector")
    return result


def render_frames(model, frames, recipe, mask_progress):
    """Keep one BN batch per polarization, across the selected physical frames."""
    pulses = [pulse for frame in frames for pulse in frame]
    outputs = render_bistatic_batch(model, [p["geometry"] for p in pulses],
        query_chunk=recipe["controls"]["query_chunk"], mask_progress=mask_progress)
    records, offset = [], 0
    for frame in frames:
        pieces = []
        for pulse, out in zip(frame, outputs[offset:offset+len(frame)]):
            pred = radar_fields_intensity(out["rcs"], pulse["ranges"][None],
                offset=recipe["intensity_offset"], scaler=recipe["intensity_scaler"], range_law="released")
            pieces.append(dict(prediction=pred, target=pulse["target"], occupancy=out["alpha"],
                               occupancy_target=pulse["occupancy_target"], coverage=out["coverage"]))
        records.append({k: torch.cat([p[k].flatten() for p in pieces]) for k in pieces[0]})
        offset += len(frame)
    return records


def evaluate(heads, dataset, recipe, stats, device, stopped=lambda: False, poses=None):
    # Validation is an added comparison protocol, so it must not perturb the
    # release's training RNG. A resumed validation repeats the same ray draws.
    devices = list(range(torch.cuda.device_count())) if torch.device(device).type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(0)
        return _evaluate(heads, dataset, recipe, stats, device, stopped, poses)


def _evaluate(heads, dataset, recipe, stats, device, stopped, poses=None):
    totals = {}
    with torch.no_grad():
        heads.eval()
        for pol, model in heads.items():
            sse = energy = 0.
            cells = pulses = unsupported = 0
            for view in dataset.viewpoints("validation"):
                if stopped():
                    return None
                frame = prepare_frame(dataset, view, pol, recipe, stats, device,
                                      pose=None if poses is None else poses[pol])
                record = render_frames(model, [frame], recipe, 1.)[0]
                sse += float((record["prediction"] - record["target"]).square().double().sum())
                energy += float(record["target"].square().double().sum())
                cells += record["target"].numel()
                pulses += len(frame)
                unsupported += int((record["coverage"] == 0).sum())
            totals[pol] = dict(sse=sse, energy=energy, cells=cells, pulses=pulses,
                               unsupported_cells=unsupported, mse=sse/cells,
                               rel_mse=sse/max(energy, 1e-30))
    sse, energy = sum(v["sse"] for v in totals.values()), sum(v["energy"] for v in totals.values())
    return dict(by_polarization=totals, pooled_rel_mse=sse/max(energy, 1e-30),
                metric_domain=NORMALIZED_DB_INTENSITY_DOMAIN, role="validation")


def validate_resume(saved, dataset, recipe, device):
    from train_radar_fields import CoveredViewSampler, normalize_cuda_rng_state
    validate_dataset_checkpoint(saved, dataset, recipe)
    step = saved.get("step")
    controls = recipe["controls"]
    if type(step) is not int or not 0 <= step <= controls["steps"]:
        raise ValueError("Invalid RF GOTCHA checkpoint step")
    sampler = CoveredViewSampler(range(len(dataset.viewpoints("train"))), saved["training_view_coverage"])
    if sum(sampler.counts.values()) != step * controls["view_batch"]:
        raise ValueError("RF GOTCHA checkpoint exposure count mismatch")
    stats = saved["training_statistics"]
    expected_counts = {pol: sum(len(dataset.shards[p, pol].sector_rows[s])
                        for p, s in dataset.viewpoints("train")) for pol in dataset.polarizations}
    if (stats.get("dataset_identity") != dataset.identity or stats.get("role") != "train"
            or stats.get("train_viewpoints") != [list(v) for v in dataset.viewpoints("train")]
            or stats.get("pulse_counts") != expected_counts
            or stats.get("dynamic_range_db") != controls["dynamic_range_db"]
            or set(stats.get("peak_power", {})) != set(dataset.polarizations)
            or any(not math.isfinite(v) or v <= 0 for v in stats["peak_power"].values())):
        raise ValueError("RF GOTCHA normalization provenance mismatch")
    history = saved["history"]
    if (not isinstance(history, list) or any(type(r.get("step")) is not int or
            not 0 < r["step"] <= step or not math.isfinite(r["metrics"]["pooled_rel_mse"])
            or r["metrics"]["pooled_rel_mse"] < 0 for r in history)
            or [r["step"] for r in history] != sorted(set(r["step"] for r in history))):
        raise ValueError("Invalid RF GOTCHA validation history")
    if saved["best_val"] != min((r["metrics"]["pooled_rel_mse"] for r in history), default=None):
        raise ValueError("RF GOTCHA best metric mismatch")
    pending = saved.get("pending_validation")
    if pending not in (None, step) or (pending is not None and history and history[-1]["step"] == pending):
        raise ValueError("Invalid pending RF GOTCHA validation")
    if saved.get("complete") != (step == controls["steps"] and pending is None and bool(history) and history[-1]["step"] == step):
        raise ValueError("RF GOTCHA completion state mismatch")
    for key in ("optimizer", "scheduler", "model_state_dict", "rng_numpy", "rng_python", "rng_torch", "rng_cuda"):
        if key not in saved:
            raise ValueError(f"RF GOTCHA checkpoint lacks {key}")
    if saved["scheduler"].get("last_epoch") != step:
        raise ValueError("RF GOTCHA scheduler clock mismatch")
    def finite_state(value):
        if torch.is_tensor(value):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite_state(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite_state(v) for v in value)
        return not isinstance(value, float) or math.isfinite(value)
    pose_keys = ("pose_model_state_dict", "pose_optimizer") if "pose_refinement" in recipe else ()
    if any(k not in saved for k in pose_keys) or (not pose_keys and "pose_model_state_dict" in saved):
        raise ValueError("RF GOTCHA checkpoint pose-refinement state mismatch")
    if any(not finite_state(saved[k]) for k in ("model_state_dict", "optimizer", "scheduler", *pose_keys)):
        raise ValueError("Nonfinite RF GOTCHA checkpoint state")
    for state in saved["optimizer"].get("state", {}).values():
        if "step" not in state or not 0 <= float(state["step"]) <= step:
            raise ValueError("Invalid RF GOTCHA optimizer step")
    state = saved["rng_torch"]
    if not torch.is_tensor(state) or state.dtype != torch.uint8 or state.ndim != 1:
        raise ValueError("Invalid RF GOTCHA Torch RNG state")
    if torch.device(device).type == "cuda":
        normalize_cuda_rng_state(saved["rng_cuda"], expected_device_count=torch.cuda.device_count(), require_present=True)


def run_gotcha(*, dataset, output_dir, config, device="cuda", resume=None):
    """Allocation-owned training; tests explicitly select the portable backend."""
    from train_radar_fields import build_model, CoveredViewSampler, normalize_cuda_rng_state
    recipe = recipe_from_config(config, dataset.region.half_extent_m, len(dataset.viewpoints("train")))
    controls, output, device = recipe["controls"], Path(output_dir), torch.device(device)
    args = model_args(recipe, device)
    saved = torch.load(resume, map_location="cpu", weights_only=False) if resume else None
    if saved is not None:
        validate_resume(saved, dataset, recipe, device)  # before any response/cache access
    check_model_backend(args)
    identity = dict(dataset_contract=dataset.contract, dataset_identity=dataset.identity, recipe=recipe)
    if output.exists() and any(output.iterdir()):
        path = output / "run.json"
        if saved is None or not path.is_file() or json.loads(path.read_text()) != identity:
            raise ValueError("RF GOTCHA output requires matching resume or a fresh output directory")
    random.seed(controls["seed"]); np.random.seed(controls["seed"]); torch.manual_seed(controls["seed"])
    rng = np.random.default_rng(controls["seed"])
    heads = torch.nn.ModuleDict({pol: build_model(args, device) for pol in dataset.polarizations})
    source_adapted = controls["profile"] == SOURCE_RECIPE
    groups = ([group for head in heads.values() for group in head.get_params(controls["lr"])]
              if source_adapted and all(hasattr(h, "get_params") for h in heads.values()) else heads.parameters())
    optimizer = torch.optim.Adam(groups, lr=controls["lr"], betas=(.9, .99), eps=1e-15)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .1 ** min(step/(800 if source_adapted else controls["steps"]), 1.))
    poses, pose_optimizer = build_pose_heads(dataset, recipe, device)
    sampler = CoveredViewSampler(range(len(dataset.viewpoints("train"))))
    step, history, best, pending = 0, [], None, None
    stats = None
    if saved is not None:
        heads.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        if poses is not None:
            poses.load_state_dict(saved["pose_model_state_dict"], strict=True)
            pose_optimizer.load_state_dict(saved["pose_optimizer"])
        sampler = CoveredViewSampler(sampler.indices, saved["training_view_coverage"])
        step, history, best, pending, stats = (saved[k] for k in
            ("step", "history", "best_val", "pending_validation", "training_statistics"))
        rng.bit_generator.state = saved["rng_numpy"]
        random.setstate(saved["rng_python"]); torch.set_rng_state(saved["rng_torch"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(normalize_cuda_rng_state(saved["rng_cuda"]))
    stopped = [False]
    def request_stop(*_):
        stopped[0] = True
    handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    def save(name):
        atomic_save(output/name, dict(schema="rift_gotcha_checkpoint_v1", **identity,
            model_state_dict=heads.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            **({} if poses is None else dict(pose_model_state_dict=poses.state_dict(),
                                             pose_optimizer=pose_optimizer.state_dict())),
            step=step, history=history, best_val=best, pending_validation=pending,
            complete=step == controls["steps"] and pending is None and bool(history) and history[-1]["step"] == step,
            training_statistics=stats, training_view_coverage=sampler.state_dict(),
            rng_numpy=rng.bit_generator.state, rng_python=random.getstate(), rng_torch=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else []))
    def finish_validation():
        nonlocal pending, best
        metrics = evaluate(heads, dataset, recipe, stats, device, lambda: stopped[0], poses)
        if metrics is None:
            return False
        history.append(dict(step=step, metrics=metrics))
        improved = best is None or metrics["pooled_rel_mse"] < best
        best = metrics["pooled_rel_mse"] if improved else best
        pending = None
        if improved:
            save("checkpoint_best.pt")
        save("checkpoint_latest.pt")
        atomic_json(output/"history.json", history)
        return True
    try:
        if stats is None:
            stats = training_statistics(dataset, recipe, lambda: stopped[0])
            if stats is None:
                return dict(status="interrupted", stage="normalization", test_accessed=False)
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output/"run.json", identity)
        if pending is not None and not finish_validation():
            return dict(status="interrupted", step=step, test_accessed=False)
        views = dataset.viewpoints("train")
        while step < controls["steps"]:
            selected = [views[i] for i in sampler.next(controls["view_batch"], rng, source_sampling=source_adapted)]
            heads.train(); optimizer.zero_grad(set_to_none=True)
            if pose_optimizer is not None:
                pose_optimizer.zero_grad(set_to_none=True)
            mask = min(1., .05 + math.sin((step+1)/max(controls["steps"]-1, 1)*math.pi/2))
            if source_adapted:
                batches = len(views) // controls["view_batch"]
                epoch = step // batches + 1
                mask = min(1., .05 + math.sin(epoch/(controls["steps"]//batches-1)*math.pi/2))
            for pol, model in heads.items():
                frames = [prepare_frame(dataset, view, pol, recipe, stats, device, training=True,
                                        pose=None if poses is None else poses[pol]) for view in selected]
                records = render_frames(model, frames, recipe, mask)
                loss, _ = released_batch_loss(records, weight_fft=controls["weight_fft"],
                    weight_occ=controls["weight_occ"], weight_bimodal=controls["weight_bimodal"], source_exact=source_adapted)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite RF GOTCHA loss")
                (loss if source_adapted else loss/len(heads)).backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all()
                   for p in (*heads.parameters(), *(() if poses is None else poses.parameters()))):
                raise ValueError("Nonfinite RF GOTCHA gradient")
            optimizer.step()
            if pose_optimizer is not None:  # release order: field step, then pose step, then the LR clock
                pose_optimizer.step()
            scheduler.step(); step += 1
            if step % controls["eval_every"] == 0 or step == controls["steps"]:
                pending = step
            if step % controls["checkpoint_every"] == 0 or stopped[0] or pending is not None:
                save("checkpoint_latest.pt")
            if stopped[0]:
                return dict(status="interrupted", step=step, test_accessed=False)
            if pending is not None and not finish_validation():
                return dict(status="interrupted", step=step, test_accessed=False)
            print(json.dumps(dict(method="radar_fields", step=step, best_validation_rel_mse=best)), flush=True)
        save("checkpoint_final.pt")
        return dict(status="complete", step=step, best_validation_rel_mse=best,
                    metric_domain=NORMALIZED_DB_INTENSITY_DOMAIN, test_accessed=False)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
