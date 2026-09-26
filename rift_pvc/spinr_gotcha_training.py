# PVC copy of rift/spinr_gotcha_training.py at b32c134; numerical recipes remain unchanged.
# Execution adaptation (GOTCHA.md section 7): the training update and the validation
# render evaluate all pulses of a shard group / pass-sector at once through
# rift_pvc.spinr_native_batched (same kernels, masks and modes; per-pulse loop kept
# as batch_update_loop for equivalence tests). Disclosed as recipe['pulse_execution'].
"""Native GOTCHA adapter for the maintained signed-real SpINR field.

All selected native TRAIN pulses are optimized; one field per polarization is
shared across passes. Source autofocus belongs exclusively to the lazy ingress.
This runtime does not import synthetic-data loaders or reuse their checkpoints.
"""
from __future__ import annotations

import math
from pathlib import Path
import signal

import numpy as np
import torch
from rift_pvc import accelerator
from rift_pvc.spinr_native_batched import (PULSE_EXECUTION, BatchedNativeKernel, batched_loss_and_field_vjp,
                                           pulse_bin_objectives, shard_groups)

from rift.gotcha_dataset import validate_checkpoint
from rift.gotcha_training import RangeReadout, atomic_json, atomic_save
from rift.spinr_fidelity import PAPER_EPOCHS, COMPARISON_EPOCHS, PAPER_REFERENCE
from rift.spinr_native import NativeKernel, bin_objective, loss_and_field_vjp
from rift.spinr_style import SpinrStyleINR, gauss_legendre_cell_grid

FORMAT = "spinr_gotcha_native_v1"


def recipe_from_config(config, dataset=None):
    from train_spinr_style_pvc import GOTCHA_BACKEND
    defaults = dict(GOTCHA_BACKEND["default_config"])
    if not isinstance(config, dict) or set(config)-set(defaults)-{"cosine_epochs"}:
        raise ValueError("Unknown SpINR GOTCHA configuration keys")
    defaults.update({k:v for k,v in config.items() if k != "cosine_epochs"})
    for name, value in defaults.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"SpINR GOTCHA {name} must be a positive integer")
    if defaults["epochs"] > PAPER_EPOCHS or defaults["grid_size"] < 2 or defaults["nodes_per_cell"] > 8:
        raise ValueError("SpINR GOTCHA epochs must be <=1500, grid >=2 and quadrature order <=8")
    # Preserve exact old recipe dictionaries when an explicit historical
    # budget/schedule is requested; never reinterpret an existing checkpoint.
    cosine_epochs = config.get("cosine_epochs", PAPER_EPOCHS if defaults["epochs"] > COMPARISON_EPOCHS else COMPARISON_EPOCHS)
    if type(cosine_epochs) is not int or cosine_epochs not in (COMPARISON_EPOCHS, PAPER_EPOCHS):
        raise ValueError("SpINR GOTCHA cosine_epochs must be 150 or 1500")
    if defaults["epochs"] > cosine_epochs:
        raise ValueError("SpINR GOTCHA epochs cannot exceed its cosine schedule")
    if defaults["seed"] != 42:
        raise ValueError("SpINR GOTCHA uses the registered seed 42")
    recipe = dict(schema=FORMAT, method="spinr", paper=PAPER_REFERENCE, **defaults,
                network="same_six_width840_relu_fourier_signed_real_field_as_RIFT_collection",
                support="registered_region_cube_in_metric_local_coordinates",
                propagation="exp(-i*4*pi*f/c*(distance-r0_effective))/distance^2",
                observable="selected_native_index_DFT_bins_norm_forward",
                kernel="exact_affine_closed_form_else_exact_native_point_kernel_DFT",
                scene_bins="cube_delay_and_native_frequency_increment_bounds_floor_ceil_modulo",
                frequency_policy="all_native_exact_no_resampling_or_padding",
                pulse_policy=("same_fixed_pulse_cap_train_validation_test"
                              if dataset is not None and dataset.training_pulse_selection else "all_native"),
                autofocus="ingress_only_channel_owned_once", polarization="independent_heads_shared_across_passes",
                batching="seed_plus_epoch_PCG64_permutation_all_selected_TRAIN_pulses",
                loss="sum_bins_magnitude_squared_plus_half_complex_squared_over_per_pol_train_raw_power",
                initialization="fixed_per_pol_0.1_sqrt_first32_train_observed_over_random_field_energy",
                optimizer=dict(name="Adam", lr=1e-4, betas=[.9,.999], eps=1e-8, weight_decay=0.,
                               clip_global_l2=1., cosine_final_lr=1e-5, cosine_epochs=cosine_epochs),
                selection="minimum_pooled_roi_projected_native_complex_validation_earliest_tie",
                roi_readout=RangeReadout.schema, early_stopping=False,
                budget_scope=("paper_epoch_count" if defaults["epochs"] == PAPER_EPOCHS else
                              "comparison_epoch_count" if defaults["epochs"] == cosine_epochs == COMPARISON_EPOCHS else
                              "development_only"),
                local_choices="architecture/encoding/init/Adam/conditioning/quadrature/bin boundaries/readout",
                acquisition_adaptation="native phase reference and exact ragged frequency DFT; not original paper acquisition",
                numerical_convergence_validated=False,
                pulse_execution=PULSE_EXECUTION)
    from rift.gotcha_frequency_selection import bind_recipe
    return recipe if dataset is None else bind_recipe(dataset, recipe)


class PulsePlan:
    """Metadata-only flattened pulse plan; rows retain their native shard identity."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.keys = list(dataset.shards)
        self.views = dataset.viewpoints("train")
        index = {tuple(key): i for i, key in enumerate(self.views)}
        records, view_ids, pol_ids = [], [], []
        for shard_index, (p, pol) in enumerate(self.keys):
            shard = dataset.shards[p, pol]
            allowed = [s for pp, s in self.views if pp == p]
            rows = np.flatnonzero((shard.row_roles == "train") & np.isin(shard.arrays["sector_id"], allowed))
            records.extend((shard_index, int(row)) for row in rows)
            view_ids.extend(index[p, int(shard.arrays["sector_id"][row])] for row in rows)
            pol_ids.extend([dataset.polarizations.index(pol)]*len(rows))
        if not records or not self.views:
            raise ValueError("SpINR GOTCHA requires nonempty TRAIN pulses")
        self.records = np.asarray(records, dtype=np.int64)
        self.view_ids = np.asarray(view_ids, dtype=np.int64)
        self.pol_ids = np.asarray(pol_ids, dtype=np.int64)
        self.view_counts = np.bincount(self.view_ids, minlength=len(self.views))
        self.pol_counts = np.bincount(self.pol_ids, minlength=len(dataset.polarizations))
        if np.any(self.view_counts == 0) or np.any(self.pol_counts == 0):
            raise ValueError("Every selected TRAIN viewpoint/polarization must contain pulses")

    def read(self, index):
        shard_index, row = self.records[index]
        return self.dataset.shards[self.keys[shard_index]].read(int(row))

    def source_id(self, index):
        shard_index, row = self.records[index]
        p, pol = self.keys[shard_index]
        return [p, pol, int(row)]

    def order(self, epoch, seed=42):
        return np.random.Generator(np.random.PCG64(seed+epoch)).permutation(len(self.records))

    def batches(self, recipe):
        return math.ceil(len(self.records)/recipe["pulse_batch_size"])

    def coverage(self, epoch, cursor, recipe):
        prefix = (self.order(epoch, recipe["seed"])[:cursor*recipe["pulse_batch_size"]]
                  if cursor else np.empty(0, dtype=np.int64))
        counts = self.view_counts*epoch+np.bincount(self.view_ids[prefix], minlength=len(self.views))
        pol_counts = self.pol_counts*epoch+np.bincount(self.pol_ids[prefix], minlength=len(self.pol_counts))
        return dict(schema="spinr_native_optimization_coverage_v1", fitting_only=True,
                    normalization_reads_counted=False, pass_sector_ids=[list(v) for v in self.views],
                    pulse_exposures_per_view=counts.tolist(), pulse_exposures=int(counts.sum()),
                    by_polarization=dict(zip(self.dataset.polarizations, pol_counts.tolist())),
                    optimizer_updates=epoch*self.batches(recipe)+cursor)


def preflight(dataset, config):
    recipe = recipe_from_config(config, dataset)
    plan = PulsePlan(dataset)
    modes = {}
    for key, shard in dataset.shards.items():
        f = shard.frequencies_for_role('train')
        affine = np.array_equal(f, f[0]+np.arange(len(f), dtype=np.float64)*(f[1]-f[0]))
        modes[f"pass{key[0]}_{key[1]}"] = "closed_form" if affine else "exact_native_point_kernel_DFT"
    return dict(recipe=recipe, dataset_identity=dataset.identity,
                training_pulses=len(plan.records), updates_per_epoch=plan.batches(recipe),
                integration_points=(recipe["grid_size"]*recipe["nodes_per_cell"])**3,
                parent_cell_pitch_m=2*dataset.region.half_extent_m/recipe["grid_size"],
                maximum_axis_cell_phase_radians=(4*math.pi/299792458.)
                    *max(s.frequencies_hz[-1] for s in dataset.shards.values())
                    *(2*dataset.region.half_extent_m/recipe["grid_size"]),
                native_kernel_modes=modes, response_reads=False, test_accessed=False)


def training_statistics(plan):
    stats = {pol:dict(energy=0., samples=0, pulses=0) for pol in plan.dataset.polarizations}
    for i in range(len(plan.records)):
        obs = plan.read(i)
        value = stats[obs.polarization]
        value["energy"] += float(np.vdot(obs.response, obs.response).real)
        value["samples"] += len(obs.response)
        value["pulses"] += 1
    for value in stats.values():
        if value["samples"] <= 0 or not math.isfinite(value["energy"]) or value["energy"] <= 0:
            raise ValueError("TRAIN signal energy must be finite and positive")
        value["mean_power"] = value["energy"]/value["samples"]
    return stats


@torch.no_grad()
def initialize_scales(heads, plan, points, volumes, recipe, device):
    from train_spinr_style_pvc import evaluate_neural_field_tiled
    scales = {}
    for pol_id, pol in enumerate(plan.dataset.polarizations):
        ids = np.flatnonzero(plan.pol_ids == pol_id)[:32]
        field = evaluate_neural_field_tiled(heads[pol], points, neural_point_tile=recipe["neural_point_tile"])
        observed_energy, predicted_energy = 0., 0.
        for index in ids:
            obs = plan.read(index)
            kernel = NativeKernel(obs, plan.dataset.region, device=device, point_tile=recipe["renderer_point_tile"])
            predicted = kernel.render(points, field, volumes, 1., selected=False)
            observed_energy += float(np.vdot(obs.response, obs.response).real)
            predicted_energy += float(predicted.abs().square().sum())
        if min(observed_energy, predicted_energy) <= 0 or not all(map(math.isfinite, (observed_energy, predicted_energy))):
            raise ValueError("Invalid SpINR native initialization energy")
        scales[pol] = dict(value=.1*math.sqrt(observed_energy/predicted_energy),
                           source_ids=[plan.source_id(i) for i in ids],
                           observed_energy=observed_energy, predicted_energy=predicted_energy)
    return scales


def batch_update(heads, optimizer, plan, ids, points, volumes, stats, scales, recipe, device):
    from train_spinr_style_pvc import evaluate_neural_field_tiled, replay_field_cotangent_tiled
    optimizer.zero_grad(set_to_none=True)
    heads.train()
    total_loss = 0.
    active = []
    for pol_id, pol in enumerate(plan.dataset.polarizations):
        selected = ids[plan.pol_ids[ids] == pol_id]
        if not len(selected):
            continue
        active.append(pol)
        field = evaluate_neural_field_tiled(heads[pol], points, neural_point_tile=recipe["neural_point_tile"])
        gradient = torch.zeros_like(field)
        for group in shard_groups(plan, selected):
            observations = [plan.read(index) for index in group]
            kernel = BatchedNativeKernel(observations, plan.dataset.region, device=device,
                                         point_tile=recipe["renderer_point_tile"])
            losses, g = batched_loss_and_field_vjp(kernel, points, field, volumes, scales[pol]["value"],
                                                   np.stack([o.response for o in observations]),
                                                   stats[pol]["mean_power"], 1./len(ids))
            total_loss += sum(losses.tolist())/len(ids)
            gradient.add_(g)
        replay_field_cotangent_tiled(heads[pol], points, gradient, neural_point_tile=recipe["neural_point_tile"])
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in heads[pol].parameters()):
            raise FloatingPointError("Missing/nonfinite native SpINR neural gradient")
    norm = float(torch.nn.utils.clip_grad_norm_(heads.parameters(), max_norm=1.))
    if not math.isfinite(norm):
        raise FloatingPointError("Nonfinite native SpINR gradient norm")
    optimizer.step()
    for pol in active:
        for p in heads[pol].parameters():
            if not torch.isfinite(p).all() or any(not torch.isfinite(v).all() for v in optimizer.state[p].values()
                                                if torch.is_tensor(v)):
                raise FloatingPointError("Nonfinite native SpINR optimizer result; last good checkpoint retained")
    return total_loss, norm, active


def batch_update_loop(heads, optimizer, plan, ids, points, volumes, stats, scales, recipe, device):
    """The original per-pulse update, retained for equivalence tests only."""
    from train_spinr_style_pvc import evaluate_neural_field_tiled, replay_field_cotangent_tiled
    optimizer.zero_grad(set_to_none=True)
    heads.train()
    total_loss = 0.
    active = []
    for pol_id, pol in enumerate(plan.dataset.polarizations):
        selected = ids[plan.pol_ids[ids] == pol_id]
        if not len(selected):
            continue
        active.append(pol)
        field = evaluate_neural_field_tiled(heads[pol], points, neural_point_tile=recipe["neural_point_tile"])
        gradient = torch.zeros_like(field)
        for index in selected:
            obs = plan.read(index)
            kernel = NativeKernel(obs, plan.dataset.region, device=device, point_tile=recipe["renderer_point_tile"])
            loss, g = loss_and_field_vjp(kernel, points, field, volumes, scales[pol]["value"],
                                        obs.response, stats[pol]["mean_power"])
            total_loss += loss/len(ids)
            gradient.add_(g/len(ids))
        replay_field_cotangent_tiled(heads[pol], points, gradient, neural_point_tile=recipe["neural_point_tile"])
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in heads[pol].parameters()):
            raise FloatingPointError("Missing/nonfinite native SpINR neural gradient")
    norm = float(torch.nn.utils.clip_grad_norm_(heads.parameters(), max_norm=1.))
    if not math.isfinite(norm):
        raise FloatingPointError("Nonfinite native SpINR gradient norm")
    optimizer.step()
    for pol in active:
        for p in heads[pol].parameters():
            if not torch.isfinite(p).all() or any(not torch.isfinite(v).all() for v in optimizer.state[p].values()
                                                if torch.is_tensor(v)):
                raise FloatingPointError("Nonfinite native SpINR optimizer result; last good checkpoint retained")
    return total_loss, norm, active


@torch.no_grad()
def evaluate(heads, dataset, points, volumes, stats, scales, recipe, device):
    """Validation with one native render per pass-sector/channel; per-pulse metrics unchanged."""
    from train_spinr_style_pvc import evaluate_neural_field_tiled
    readout = RangeReadout(dataset.region, device=device)
    totals = {}
    heads.eval()
    for pol in dataset.polarizations:
        field = evaluate_neural_field_tiled(heads[pol], points, neural_point_tile=recipe["neural_point_tile"])
        t = dict(roi_error=0., roi_energy=0., full_error=0., full_energy=0., native_objective=0., pulses=0, samples=0)
        for p, sector in dataset.viewpoints("validation"):
            observations = list(dataset.observations(p, sector, pol))
            kernel = BatchedNativeKernel(observations, dataset.region, device=device, point_tile=recipe["renderer_point_tile"])
            prediction = kernel.render(points, field, volumes, scales[pol]["value"], selected=False)
            target = torch.as_tensor(np.stack([o.response for o in observations]), dtype=torch.complex128, device=device)
            roi_error = torch.zeros((), dtype=torch.float64, device=device)
            roi_energy = torch.zeros((), dtype=torch.float64, device=device)
            for i, obs in enumerate(observations):
                r = readout.for_observation(obs)
                pred_roi, target_roi = readout.project(prediction[i], r), readout.project(target[i], r)
                roi_error = roi_error + (pred_roi-target_roi).abs().square().sum()
                roi_energy = roi_energy + target_roi.abs().square().sum()
            t["roi_error"] += float(roi_error)
            t["roi_energy"] += float(roi_energy)
            t["full_error"] += float((prediction-target).abs().square().sum())
            t["full_energy"] += float(target.abs().square().sum())
            # Metric-only transform; training uses the selected point kernel.
            t["native_objective"] += float(pulse_bin_objectives(
                kernel.select_bins(torch.fft.fft(prediction, dim=1, norm="forward")),
                kernel.target_bins(np.stack([o.response for o in observations])), stats[pol]["mean_power"]).sum())
            t["pulses"] += len(observations)
            t["samples"] += int(target.numel())
        if min(t["roi_energy"], t["full_energy"], t["pulses"]) <= 0 or not all(math.isfinite(v) for v in t.values()):
            raise ValueError("Invalid full-role native validation metrics")
        totals[pol] = t
    def pooled(numerator, denominator):
        return sum(v[numerator] for v in totals.values())/sum(v[denominator] for v in totals.values())
    return dict(metric_domain="roi_projected_native_complex", pooled_rel_mse=pooled("roi_error", "roi_energy"),
                full_native_complex_rel_mse=pooled("full_error", "full_energy"),
                native_spectral_objective=pooled("native_objective", "pulses"),
                viewpoints=len(dataset.viewpoints("validation")), by_polarization=totals,
                test_accessed=False, roi_qualification="range-compatible clutter may remain; not isolated vehicle truth")


@torch.no_grad()
def evaluate_loop(heads, dataset, points, volumes, stats, scales, recipe, device):
    """The original per-pulse validation, retained for equivalence tests only."""
    from train_spinr_style_pvc import evaluate_neural_field_tiled
    readout = RangeReadout(dataset.region, device=device)
    totals = {}
    heads.eval()
    for pol in dataset.polarizations:
        field = evaluate_neural_field_tiled(heads[pol], points, neural_point_tile=recipe["neural_point_tile"])
        t = dict(roi_error=0., roi_energy=0., full_error=0., full_energy=0., native_objective=0., pulses=0, samples=0)
        for p, sector in dataset.viewpoints("validation"):
            for obs in dataset.observations(p, sector, pol):
                kernel = NativeKernel(obs, dataset.region, device=device, point_tile=recipe["renderer_point_tile"])
                prediction = kernel.render(points, field, volumes, scales[pol]["value"], selected=False)
                target = torch.as_tensor(obs.response, dtype=torch.complex128, device=device)
                r = readout.for_observation(obs)
                pred_roi, target_roi = readout.project(prediction, r), readout.project(target, r)
                t["roi_error"] += float((pred_roi-target_roi).abs().square().sum())
                t["roi_energy"] += float(target_roi.abs().square().sum())
                t["full_error"] += float((prediction-target).abs().square().sum())
                t["full_energy"] += float(target.abs().square().sum())
                # Metric-only transform; training uses the selected point kernel.
                t["native_objective"] += float(bin_objective(torch.fft.fft(prediction, norm="forward")[kernel.bin_ids],
                    kernel.target_bins(obs.response), stats[pol]["mean_power"]))
                t["pulses"] += 1
                t["samples"] += len(target)
        if min(t["roi_energy"], t["full_energy"], t["pulses"]) <= 0 or not all(math.isfinite(v) for v in t.values()):
            raise ValueError("Invalid full-role native validation metrics")
        totals[pol] = t
    def pooled(numerator, denominator):
        return sum(v[numerator] for v in totals.values())/sum(v[denominator] for v in totals.values())
    return dict(metric_domain="roi_projected_native_complex", pooled_rel_mse=pooled("roi_error", "roi_energy"),
                full_native_complex_rel_mse=pooled("full_error", "full_energy"),
                native_spectral_objective=pooled("native_objective", "pulses"),
                viewpoints=len(dataset.viewpoints("validation")), by_polarization=totals,
                test_accessed=False, roi_qualification="range-compatible clutter may remain; not isolated vehicle truth")


def _expected_lr(epoch, cosine_epochs=PAPER_EPOCHS):
    return 1e-5+(1e-4-1e-5)*(1+math.cos(math.pi*epoch/cosine_epochs))/2


def validate_resume(saved, dataset, recipe, plan):
    """Validate source, scientific recipe and progress before response access."""
    validate_checkpoint(saved, dataset, recipe)
    if saved.get("method_format") != FORMAT:
        raise ValueError("Not a native SpINR checkpoint")
    epoch, cursor, updates = (saved.get(k) for k in ("epoch", "cursor", "updates"))
    if (any(type(v) is not int for v in (epoch, cursor, updates)) or not 0 <= epoch <= recipe["epochs"]
            or not 0 <= cursor <= plan.batches(recipe) or updates != epoch*plan.batches(recipe)+cursor
            or (epoch == recipe["epochs"] and cursor != 0)):
        raise ValueError("Invalid native SpINR epoch/update cursor")
    if saved.get("optimization_coverage") != plan.coverage(epoch, cursor, recipe):
        raise ValueError("Native SpINR optimization exposure counts changed")
    history = saved.get("history")
    if not isinstance(history, list) or len(history) != epoch:
        raise ValueError("Invalid native SpINR history")
    candidates = []
    for i, record in enumerate(history, 1):
        if record.get("epoch") != i or record.get("updates") != i*plan.batches(recipe):
            raise ValueError("Native SpINR history/cursor mismatch")
        if record.get("optimization_coverage") != plan.coverage(i, 0, recipe):
            raise ValueError("Native SpINR historical exposure mismatch")
        if not math.isfinite(record.get("training_objective", math.nan)) or record["training_objective"] < 0:
            raise ValueError("Invalid native SpINR training history")
        if (not math.isfinite(record.get("mean_unclipped_gradient_norm", math.nan))
                or record["mean_unclipped_gradient_norm"] < 0
                or not 0 <= record.get("fraction_clipped_updates", math.nan) <= 1):
            raise ValueError("Invalid native SpINR historical clipping diagnostics")
        metrics = record.get("validation")
        if i % recipe["validation_every"] == 0 or i == recipe["epochs"]:
            if not isinstance(metrics, dict):
                raise ValueError("Missing full-role native validation")
        if metrics is not None:
            value = metrics.get("pooled_rel_mse")
            if not isinstance(value, (int,float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid native SpINR selection metric")
            candidates.append((value, i))
    best = min(candidates) if candidates else (None, None)
    if (saved.get("best_val"), saved.get("best_epoch")) != best:
        raise ValueError("Native SpINR selection disagrees with validation history")
    partial = saved.get("partial_loss_sum")
    if not isinstance(partial, (int,float)) or not math.isfinite(partial) or partial < 0 or (cursor == 0 and partial != 0):
        raise ValueError("Invalid native SpINR partial objective")
    grad_sum, clipped = saved.get("partial_gradient_norm_sum"), saved.get("partial_clipped_updates")
    if (not isinstance(grad_sum, (int,float)) or not math.isfinite(grad_sum) or grad_sum < 0
            or type(clipped) is not int or not 0 <= clipped <= cursor
            or (cursor == 0 and (grad_sum != 0 or clipped != 0))):
        raise ValueError("Invalid native SpINR partial clipping diagnostics")
    stats, scales, steps = (saved.get(k) for k in ("training_statistics", "initial_scales", "head_updates"))
    if any(not isinstance(v, dict) or set(v) != set(dataset.polarizations) for v in (stats, scales, steps)):
        raise ValueError("Missing native SpINR channel state")
    for j, pol in enumerate(dataset.polarizations):
        ids = np.flatnonzero(plan.pol_ids == j)
        count = sum(len(dataset.shards[plan.keys[s]].frequencies_for_role('train')) for s, _ in plan.records[ids])
        s, scale = stats[pol], scales[pol]
        if (s.get("samples") != count or s.get("pulses") != len(ids)
                or any(not isinstance(s.get(k), (int,float)) or not math.isfinite(s[k]) or s[k] <= 0 for k in ("energy", "mean_power"))
                or not math.isclose(s["mean_power"], s["energy"]/count, rel_tol=1e-12)):
            raise ValueError("Invalid native SpINR TRAIN statistics")
        if (scale.get("source_ids") != [plan.source_id(i) for i in ids[:32]]
                or any(not isinstance(scale.get(k), (int,float)) or not math.isfinite(scale[k]) or scale[k] <= 0
                       for k in ("value", "observed_energy", "predicted_energy"))
                or not math.isclose(scale["value"], .1*math.sqrt(scale["observed_energy"]/scale["predicted_energy"]), rel_tol=1e-12)):
            raise ValueError("Invalid native SpINR initialization provenance")
        if type(steps[pol]) is not int or not 0 <= steps[pol] <= updates:
            raise ValueError("Invalid native SpINR per-head updates")
        exposures = saved["optimization_coverage"]["by_polarization"][pol]
        if not math.ceil(exposures/recipe["pulse_batch_size"]) <= steps[pol] <= min(exposures, updates):
            raise ValueError("Native SpINR head updates disagree with pulse exposure")
    if not updates <= sum(steps.values()) <= len(steps)*updates:
        raise ValueError("Native SpINR head updates disagree with global updates")
    if not isinstance(saved.get("rng_state"), dict):
        raise ValueError("Missing native SpINR RNG state")


def _validate_model_optimizer(saved, heads):
    from train_spinr_style_pvc import _validate_complete_rng_state_without_changing_runtime
    cosine_epochs = saved["recipe"]["optimizer"]["cosine_epochs"]
    params = list(heads.named_parameters())
    state = saved.get("model_state_dict", {})
    if set(state) != set(heads.state_dict()):
        raise ValueError("Native SpINR model layout changed")
    for name, p in params:
        value = state[name]
        if not torch.is_tensor(value) or value.shape != p.shape or value.dtype != p.dtype or not torch.isfinite(value).all():
            raise ValueError("Invalid native SpINR model tensor")
    optimizer = saved.get("optimizer_state_dict", {})
    groups = optimizer.get("param_groups", [])
    if len(groups) != 1 or groups[0].get("params") != list(range(len(params))):
        raise ValueError("Native SpINR optimizer parameter layout changed")
    group = groups[0]
    if (tuple(group.get("betas", ())) != (.9, .999) or group.get("eps") != 1e-8
            or group.get("weight_decay") != 0 or group.get("amsgrad", False)
            or group.get("maximize", False) or group.get("differentiable", False)
            or group.get("capturable", False) or group.get("decoupled_weight_decay", False)
            or group.get("foreach") is not None or group.get("fused") not in (None, False)
            or not math.isclose(group.get("lr", math.nan), _expected_lr(saved["epoch"], cosine_epochs), rel_tol=0, abs_tol=1e-15)):
        raise ValueError("Native SpINR optimizer recipe changed")
    moments = optimizer.get("state", {})
    if set(moments)-set(range(len(params))):
        raise ValueError("Unexpected native SpINR optimizer state")
    for i, (name, p) in enumerate(params):
        steps = saved["head_updates"][name.split(".")[0]]
        item = moments.get(i)
        if steps == 0:
            if item is not None:
                raise ValueError("Unfitted SpINR head has optimizer moments")
            continue
        if not isinstance(item, dict) or set(item) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Missing native SpINR optimizer moments")
        step = item["step"]
        if not torch.is_tensor(step) or step.numel() != 1 or float(step) != steps:
            raise ValueError("Native SpINR optimizer step disagrees with head exposure")
        for key in ("exp_avg", "exp_avg_sq"):
            v = item[key]
            if not torch.is_tensor(v) or v.shape != p.shape or v.dtype != p.dtype or not torch.isfinite(v).all():
                raise ValueError("Invalid native SpINR optimizer moments")
        if bool((item["exp_avg_sq"] < 0).any()):
            raise ValueError("Negative native SpINR second moment")
    scheduler = saved.get("scheduler_state_dict", {})
    if (scheduler.get("T_max") != cosine_epochs or scheduler.get("eta_min") != 1e-5
            or scheduler.get("last_epoch") != saved["epoch"] or scheduler.get("_step_count") != saved["epoch"]+1
            or scheduler.get("base_lrs") != [1e-4] or len(scheduler.get("_last_lr", [])) != 1
            or not math.isclose(scheduler["_last_lr"][0], _expected_lr(saved["epoch"], cosine_epochs), rel_tol=0, abs_tol=1e-15)):
        raise ValueError("Native SpINR scheduler trajectory changed")
    _validate_complete_rng_state_without_changing_runtime(saved["rng_state"])


def run(dataset, output_dir, config, *, device=None, resume=None, should_stop=None):
    from rift_pvc.train_rng import capture_rng_state, restore_rng_state
    from train import load_tensor_checkpoint
    device = accelerator.device() if device is None else torch.device(device)
    from train_spinr_style_pvc import _set_seed, _disable_tf32
    recipe = recipe_from_config(config, dataset)
    plan = PulsePlan(dataset)
    output = Path(output_dir).resolve()
    saved = None
    if resume is not None:
        resume = Path(resume).resolve()
        if resume.parent != output or resume.name != "checkpoint_latest.pt":
            raise ValueError("Resume native SpINR latest checkpoint in its original output directory")
        saved = load_tensor_checkpoint(resume, map_location="cpu")
        validate_resume(saved, dataset, recipe, plan)
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError("Native SpINR requires a fresh output directory or exact resume")
    _disable_tf32()
    _set_seed(recipe["seed"])
    heads = torch.nn.ModuleDict({p:SpinrStyleINR(support_m=dataset.region.half_extent_m)
                                 for p in dataset.polarizations}).to(device)
    optimizer = torch.optim.Adam(heads.parameters(), lr=1e-4, betas=(.9,.999), eps=1e-8, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=recipe["optimizer"]["cosine_epochs"], eta_min=1e-5)
    if saved is not None:
        _validate_model_optimizer(saved, heads)
        heads.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        # A pending epoch may resume immediately before scheduler.step(). The
        # restored Adam trajectory already contains those committed steps.
        optimizer._opt_called = saved["updates"] > 0
        restore_rng_state(saved["rng_state"], require_complete=True)
        epoch, cursor, updates = (saved[k] for k in ("epoch", "cursor", "updates"))
        stats, scales, history = (saved[k] for k in ("training_statistics", "initial_scales", "history"))
        best, best_epoch = saved["best_val"], saved["best_epoch"]
        head_updates, partial_loss = saved["head_updates"], saved["partial_loss_sum"]
        partial_gradient, partial_clipped = saved["partial_gradient_norm_sum"], saved["partial_clipped_updates"]
    else:
        epoch = cursor = updates = 0
        history, best, best_epoch, partial_loss = [], None, None, 0.
        partial_gradient, partial_clipped = 0., 0
        head_updates = {pol:0 for pol in dataset.polarizations}
    def result(status):
        return dict(status=status, completed_epochs=epoch, updates=updates, output=str(output),
                    budget_scope=recipe["budget_scope"], best_validation_rel_mse=best, test_accessed=False)
    if saved is not None:
        best_path = output/"checkpoint_best.pt"
        repair_best = False
        if best is None:
            if best_path.exists():
                raise ValueError("Unexpected selected checkpoint before any validation")
        elif best_path.exists():
            selected = load_tensor_checkpoint(best_path, map_location="cpu")
            validate_resume(selected, dataset, recipe, plan)
            _validate_model_optimizer(selected, heads)
            if selected["epoch"] > epoch or selected["cursor"] != 0:
                raise ValueError("Selected native SpINR checkpoint is ahead of committed progress")
            repair_best = selected["epoch"] != best_epoch or selected["best_val"] != best
        else:
            repair_best = True
        if repair_best:
            if epoch != best_epoch or cursor != 0:
                raise ValueError("Selected native SpINR checkpoint is missing or stale and cannot be recovered")
            atomic_save(best_path, saved)
    if saved is not None and epoch == recipe["epochs"]:
        # Recover a crash after committing the terminal latest checkpoint but
        # before writing the best/final/history derivative artifacts.
        atomic_save(output/"checkpoint_final.pt", saved)
        atomic_json(output/"history.json", history)
        return result("complete")
    points, volumes = gauss_legendre_cell_grid(recipe["grid_size"], nodes_per_cell=recipe["nodes_per_cell"],
        support_m=dataset.region.half_extent_m, device=device, dtype=torch.float64)
    stopped = {"value":False}
    def request_stop(*_):
        stopped["value"] = True
    def stopping():
        return stopped["value"] or (should_stop is not None and should_stop())
    handlers = {sig:signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    def state():
        return dict(schema="rift_gotcha_checkpoint_v1", method_format=FORMAT,
                    dataset_contract=dataset.contract, dataset_identity=dataset.identity, recipe=recipe,
                    model_state_dict=heads.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict(), rng_state=capture_rng_state(),
                    training_statistics=stats, initial_scales=scales, epoch=epoch, cursor=cursor,
                    updates=updates, head_updates=head_updates, partial_loss_sum=partial_loss,
                    partial_gradient_norm_sum=partial_gradient, partial_clipped_updates=partial_clipped,
                    history=history, best_val=best, best_epoch=best_epoch,
                    optimization_coverage=plan.coverage(epoch, cursor, recipe))
    try:
        if saved is None:
            stats = training_statistics(plan)
            scales = initialize_scales(heads, plan, points, volumes, recipe, device)
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output/"dataset.json", dict(contract=dataset.contract, summary=dataset.summary()))
        atomic_json(output/"recipe.json", recipe)
        atomic_save(output/"checkpoint_latest.pt", state())
        while epoch < recipe["epochs"]:
            order = plan.order(epoch, recipe["seed"])
            while cursor < plan.batches(recipe):
                if stopping():
                    atomic_save(output/"checkpoint_latest.pt", state())
                    return result("interrupted")
                ids = order[cursor*recipe["pulse_batch_size"]:(cursor+1)*recipe["pulse_batch_size"]]
                loss, norm, active = batch_update(heads, optimizer, plan, ids, points, volumes, stats, scales, recipe, device)
                partial_loss += loss*len(ids)
                partial_gradient += norm
                partial_clipped += int(norm > 1.)
                for pol in active:
                    head_updates[pol] += 1
                cursor += 1
                updates += 1
                if updates % recipe["checkpoint_every"] == 0 or stopping() or cursor == plan.batches(recipe):
                    atomic_save(output/"checkpoint_latest.pt", state())
                if stopping():
                    return result("interrupted")
            # A saved cursor == batch count is before scheduler/validation;
            # recovery finalizes this epoch once without repeating an update.
            metrics = None
            if (epoch+1) % recipe["validation_every"] == 0 or epoch+1 == recipe["epochs"]:
                metrics = evaluate(heads, dataset, points, volumes, stats, scales, recipe, device)
            scheduler.step()
            epoch += 1
            record = dict(epoch=epoch, updates=updates, training_objective=partial_loss/len(plan.records),
                          mean_unclipped_gradient_norm=partial_gradient/cursor,
                          fraction_clipped_updates=partial_clipped/cursor,
                          optimization_coverage=plan.coverage(epoch, 0, recipe))
            if metrics is not None:
                record["validation"] = metrics
            history.append(record)
            cursor, partial_loss = 0, 0.
            partial_gradient, partial_clipped = 0., 0
            improved = metrics is not None and (best is None or metrics["pooled_rel_mse"] < best)
            if improved:
                best, best_epoch = metrics["pooled_rel_mse"], epoch
            current = state()
            atomic_save(output/"checkpoint_latest.pt", current)
            if improved:
                atomic_save(output/"checkpoint_best.pt", current)
            atomic_json(output/"history.json", history)
            print(f"SpINR GOTCHA epoch {epoch}/{recipe['epochs']}; updates={updates}; best validation={best}", flush=True)
        atomic_save(output/"checkpoint_final.pt", state())
        return result("complete")
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
