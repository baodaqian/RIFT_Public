"""PVC twin of ``rift/gotcha_training.py`` (built-in GOTCHA RIFT/MFBP methods).

Copy with four adaptations: RNG checkpoint payload and restore go through
``rift_pvc.accelerator`` (cross-backend resume is refused, as the original
refuses a missing CUDA payload), the ``device`` defaults resolve to the
active accelerator, and the recipe carries ``sector_execution``: the default
``batched_pulses_one_backward_v1`` renders all selected pulses of a pass-sector
in one kernel per point chunk with one backward pass (``rift_pvc.gotcha_batched``,
5-6x faster on a Max 1100, GOTCHA.md section 7); ``per_pulse_loop`` keeps the
original loop. Everything else is identical to the original, including the
declared amplitude law (``range_model``), the RIFT-dataset start and priors
(backprojection, gain warm start, coefficient gauge; transferred group-L1 and
SH-degree priors) and the B787 optimizer and schedule (``optimizer``), all
imported from the original module.

Original docstring:
Native-frequency GOTCHA training backends, independent of frozen B787 CLIs.

Every learned channel is shared across the selected elevation passes. The ROI
loss uses a geometry-only range subspace, which suppresses out-of-range clutter
but does not isolate objects at the same delay. Full native residuals are also
reported. Baseline model implementations are deliberately outside this shared RIFT runtime.
"""
from __future__ import annotations

from collections import OrderedDict
import json
import math
import os
from pathlib import Path
import signal

import numpy as np
import torch
from torch import nn

from rift_pvc import accelerator
from torch.utils.checkpoint import checkpoint as recompute

from rift.gotcha_dataset import C, validate_checkpoint
from rift.gotcha_training import (B787_OPTIMIZER, BACKPROJECTION, LEARNED, LEGACY_RANGE_MODEL, RANGE_MODELS,
                                  RIFT_DATASET_PRIORS, b787_epoch_end, b787_schedule, build_scheduler,
                                  fixed_view_order, optimizer_recipe, prior_backward, prior_terms, probe_due,
                                  range_amplitude, rift_dataset_initialization, rift_dataset_recipe, with_legacy_keys)
from rift_pvc.gotcha_batched import (BATCHED, LOOP, accumulate_batched_stats, projected_error_sums,
                                    sector_execution_value, sector_forward, sector_update)
from rift_pvc import gotcha_nufft as nufft
from rift_pvc.gotcha_densify import densify_due, scale_learning_rates, trilinear_densify
from rift.sparse_scene import AdaptivePointSHScene, SHVoxelGridScene, VoxelGridScene

# Baseline implementations are owned separately. Only these native backends
# are implemented here; the root dispatcher discovers opt-in baseline hooks.
METHODS = ('rift', 'rift_grid', 'isotropic', 'mfbp')


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def atomic_save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    torch.save(value, temporary)
    os.replace(temporary, path)


def native_forward(points, weights, antenna, frequencies, reference_range, *, point_chunk=4096,
                   range_model='unit'):
    """Exact native monostatic phase, FP64 geometry and complex128 accumulation.

    No synthetic uniform frequencies, virtual reference, FFT-bin aliasing or
    cross-pass padding is introduced. Blocks are recomputed in backward to
    bound saved phase activations by the declared point chunk. The phase uses
    the reference-adjusted range; ``range_amplitude`` uses the physical range.
    """
    if point_chunk <= 0:
        raise ValueError('point_chunk must be positive')
    if range_model not in RANGE_MODELS:
        raise ValueError(f'range_model must be one of {RANGE_MODELS}')
    antenna = torch.as_tensor(antenna, dtype=torch.float64, device=points.device)
    frequencies = torch.as_tensor(frequencies, dtype=torch.float64, device=points.device)
    def block(x, w):
        physical = torch.linalg.vector_norm(x.double() - antenna, dim=-1)
        distance = physical - float(reference_range)
        phase = (-4 * math.pi / C) * distance[:, None] * frequencies[None, :]
        weight = w.to(torch.complex128)
        amplitude = range_amplitude(physical, range_model)
        if amplitude is not None:
            weight = weight * amplitude
        return (torch.exp(1j * phase) * weight[:, None]).sum(0)
    result = torch.zeros(frequencies.shape, dtype=torch.complex128, device=points.device)
    for start in range(0, len(points), point_chunk):
        x, w = points[start:start+point_chunk], weights[start:start+point_chunk]
        if torch.is_grad_enabled() and (x.requires_grad or w.requires_grad):
            result = result + recompute(block, x, w, use_reentrant=False)
        else:
            result = result + block(x, w)
    return result


class RangeReadout:
    """Geometry-only ROI projector and native matched-range readout.

    Half-Rayleigh delay sampling and two guard cells are explicit adapter
    choices. SVD removes nearly dependent columns before orthogonal projection.
    This schema is distinct from the historical Camry range-projector recipe.
    """
    schema = 'gotcha_roi_native_range_subspace_v1'

    def __init__(self, region, *, device='cpu', cache_size=64):
        self.region, self.device = region, torch.device(device)
        self.cache_size = cache_size
        self._bases = OrderedDict()

    def for_observation(self, observation):
        antenna = self.region.to_local(observation.position_m)
        extent = self.region.half_extent_m
        nearest = np.clip(antenna, -extent, extent)
        r_min = float(np.linalg.norm(antenna - nearest))
        r_max = float(np.linalg.norm(np.abs(antenna) + extent))
        f = observation.frequencies_hz
        spacing = C / (2 * (f[-1] - f[0]))
        step = spacing / 2
        low = r_min - observation.reference_range_m - 2 * spacing
        count = int(math.ceil((r_max - r_min + 4 * spacing) / step)) + 1
        if count >= len(f) or (r_max - r_min + 4*spacing) >= C/(2*np.diff(f).max()):
            raise ValueError('ROI range interval exceeds the native resolvable subspace; use a smaller region')
        key = (f.tobytes(), count)
        if key not in self._bases:
            centered = f - f.mean()
            atoms = np.exp((-4j * np.pi / C) * centered[:, None] * (np.arange(count) * step)[None, :]) / np.sqrt(len(f))
            # A data-independent constant, shared by all observations with the
            # same native frequency vector and guarded interval length.
            u, singular, _ = np.linalg.svd(atoms, full_matrices=False)
            q = u[:, singular > singular[0]*1e-10]
            self._bases[key] = torch.as_tensor(q, dtype=torch.complex128, device=self.device)
            if len(self._bases) > self.cache_size:
                self._bases.popitem(last=False)
        self._bases.move_to_end(key)
        q = self._bases[key]
        frequencies = torch.as_tensor(f.copy(), dtype=torch.float64, device=self.device)
        centered = frequencies - frequencies.mean()
        phase = torch.exp((4j*math.pi/C) * centered * low)
        delays = low + torch.arange(count, device=self.device, dtype=torch.float64) * step
        matched = torch.exp((4j*math.pi/C) * delays[:, None] * centered[None, :]) / len(f)
        return dict(q=q, phase=phase, delays=delays, matched=matched,
                    antenna=torch.as_tensor(antenna, dtype=torch.float64, device=self.device),
                    frequencies=frequencies, range_step=step)

    @staticmethod
    def project(value, readout):
        return readout['q'].conj().T @ (readout['phase'] * value)

    @staticmethod
    def lift(value, readout):
        return readout['phase'].conj() * (readout['q'] @ value)


def box_anchors(support, device):
    """Cell centres of a cubic-pitch grid filling a box centred at the region origin (tuning campaign, Phase 0b).

    ``support = dict(kind='box', half_extents=[hx, hy, hz], pitch=p)``: each axis carries round(2 h / p)
    cells, so the box is filled exactly when 2 h is a multiple of the pitch.
    """
    if support.get('kind') != 'box':
        raise ValueError(f"unknown support kind {support.get('kind')!r}")
    pitch = float(support['pitch'])
    axes = []
    for h in support['half_extents']:
        n = int(round(2 * float(h) / pitch))
        if n < 1 or abs(n * pitch - 2 * float(h)) > 1e-9:
            raise ValueError('box support needs half-extents that are whole multiples of half the pitch')
        axes.append((torch.arange(n, dtype=torch.float64, device=device) + .5) * pitch - float(h))
    return torch.cartesian_prod(*axes)


def _grid(granularity, extent, device):
    axis = (torch.arange(granularity, dtype=torch.float64, device=device) + .5) * (2*extent/granularity) - extent
    return torch.cartesian_prod(axis, axis, axis)


class ChannelField(nn.Module):
    """One polarization head fitted jointly to all selected passes."""
    def __init__(self, method, region, recipe, device):
        super().__init__()
        self.method, self.region, self.recipe = method, region, recipe
        g, e = recipe['granularity'], region.half_extent_m
        support = recipe.get('support')
        if support is not None and method != 'rift':
            raise ValueError('box support is implemented for adaptive RIFT only')
        self.register_buffer('points', _grid(g, e, device) if support is None else box_anchors(support, device))
        self.register_buffer('scale_initialized', torch.tensor(False, device=device))
        if method == 'rift':
            # Opt-in keys (absent from default recipes): an initial SH degree above 0 makes bands
            # 1..init_degree trainable from the first update (reviewer B7); the backprojection start
            # still writes degree 0 only, the other bands start at zero.
            init_degree = int(recipe.get('sh_init_degree', 0))
            init_scale = 0.0 if recipe.get('initialization') == BACKPROJECTION else 1e-3
            if support is None:
                self.field = AdaptivePointSHScene.from_regular_grid(
                    g, e, device, max_degree=recipe['sh_degree'], init_degree=init_degree,
                    init_scale=init_scale, capacity=recipe['max_points'],
                    enforce_support_bounds=True, compact_sh_eval=True)
            else:
                self.field = AdaptivePointSHScene(
                    self.points.float(), float(support['pitch']) / 2, device, max_degree=recipe['sh_degree'],
                    init_degree=init_degree, init_scale=init_scale, capacity=recipe['max_points'],
                    enforce_support_bounds=True, compact_sh_eval=True)
        elif method == 'rift_grid':
            self.field = SHVoxelGridScene(g, e, device, max_degree=recipe['sh_degree'],
                                          init_degree=recipe['sh_degree'], init_scale=1e-3)
        elif method == 'isotropic':
            self.field = VoxelGridScene(g, e, device, init_scale=1e-3)
        else:
            raise ValueError(f'No learned field for {method}')
        from rift.calibration import GlobalComplexGain
        self.gain = GlobalComplexGain().to(device)

    def forward(self, observation, readout, *, probe=False):
        antenna = readout['antenna']
        extra = {}
        if self.method in ('rift', 'rift_grid', 'isotropic'):
            direction = antenna / torch.linalg.vector_norm(antenna)
            theta, phi = torch.acos(direction[2].clamp(-1, 1)), torch.atan2(direction[1], direction[0])
            if self.method == 'isotropic':
                points, weights = self.field.active_scatterers()
            elif self.method == 'rift':
                points, weights = self.field.active_scatterers(theta.float().reshape(1,1), phi.float().reshape(1,1), probe_next_band=probe)
            else:
                points, weights = self.field.active_scatterers(theta.float().reshape(1,1), phi.float().reshape(1,1))
        else:
            raise AssertionError(self.method)
        if nufft.forward_evaluation(self.recipe) == nufft.NUFFT:
            # Opt-in control: the RIFT-dataset NUFFT of the same kernel (rift_pvc.gotcha_nufft).
            predicted = nufft.pulse_render(points, weights, antenna, readout['frequencies'], observation, self,
                                           point_chunk=self.recipe['point_chunk'],
                                           range_model=self.recipe.get('range_model', LEGACY_RANGE_MODEL))
        else:
            predicted = native_forward(points, weights, antenna, readout['frequencies'],
                                       observation.reference_range_m, point_chunk=self.recipe['point_chunk'],
                                       range_model=self.recipe.get('range_model', LEGACY_RANGE_MODEL))
        if self.gain is not None:
            predicted = self.gain(predicted)
        return predicted, extra

    @torch.no_grad()
    def initialize_scale(self, observation, readout):
        if bool(self.scale_initialized):
            return
        predicted, _ = self(observation, readout)
        target = torch.as_tensor(observation.response, device=predicted.device)
        if nufft.loss_domain(self.recipe) == nufft.FULL:
            p, y = predicted, target
        else:
            p, y = RangeReadout.project(predicted, readout), RangeReadout.project(target, readout)
        self.gain.maybe_init_scale(p, y)
        self.scale_initialized.fill_(True)


def recipe_from_args(args, method):
    keys = ('epochs', 'seed', 'granularity', 'max_points', 'sh_degree', 'point_chunk', 'checkpoint_every')
    recipe = dict(schema='gotcha_native_method_recipe_v1', method=method,
                roi_readout=RangeReadout.schema, loss_normalization='train_only_mean_projected_power',
                frequency_policy='all_native',
                pulse_policy=('same_fixed_pulse_cap_train_validation_test' if getattr(args, 'pulses_per_sector', 0)
                              else 'all_native'),
                update_unit=('one_pass_sector_selected_native_pulses_and_selected_channels'
                             if getattr(args, 'pulses_per_sector', 0) else
                             'one_pass_sector_all_native_pulses_and_selected_channels'),
                head_sharing='one_field_per_polarization_shared_across_passes',
                **{key:getattr(args, key) for key in keys},
                **optimizer_recipe(args, method),
                **({'range_model': getattr(args, 'range_model', 'sum2')} if method in LEARNED else {}),
                **(rift_dataset_recipe(args) if method == 'rift' else {}),
                sector_execution=sector_execution_value(getattr(args, 'sector_execution', 'batched')))
    # Opt-in NUFFT/full-native control keys (rift_pvc.gotcha_nufft); none for the default recipe.
    recipe.update(nufft.control_recipe(args, method))
    # Opt-in tuning-campaign keys (docs/RIFT_GOTCHA_Tune.md section 7); absent unless requested.
    if getattr(args, 'support_box', None):
        recipe['support'] = dict(kind='box', half_extents=[float(v) for v in args.support_box],
                                 pitch=float(args.support_pitch), frame='region local, centred at the origin')
    if getattr(args, 'sh_init_degree', 0):
        recipe['sh_init_degree'] = int(args.sh_init_degree)
    if method == 'rift' and getattr(args, 'prune_every', 0):
        # train.py's --prune-* in its target mode (the cubic ramp, then held), energy criterion.
        recipe['prune'] = dict(every=int(args.prune_every), mode='target', criterion='energy',
                               target_active=int(args.prune_target_active), start_epoch=int(args.prune_start_epoch),
                               end_epoch=int(args.prune_end_epoch or recipe['epochs']),
                               min_active=int(args.prune_min_active), threshold=0.01,
                               ramp_from='active points after the start (train.py uses active_mask.numel(), '
                                         'which for a point scene is its capacity)',
                               source='train.py --prune-mode target, _prune_target_for_epoch')
    if method == 'rift' and getattr(args, 'mu1_scale', 1.0) != 1.0:
        recipe['mu1_scale'] = float(args.mu1_scale)
    if method == 'rift' and getattr(args, 'step_every', 1) != 1:
        # train.py --step-every N (sum over N pass-sectors per step). The summed gradient and the summed
        # Hessian scale the eps-damped step's stability number (lr/eps)*lambda by N, so an arm keeps it
        # with lr/N (or eps*N); ``updates`` keeps counting pass-sectors.
        recipe['step_every'] = int(args.step_every)
    if getattr(args, 'unit_split_stride', 0):
        # The dataset contract carries the split itself; the recipe names it so a resume gate sees it too.
        recipe['unit_split'] = dict(schema='gotcha_unit_split_pass_heldout_v1', sector_stride=int(args.unit_split_stride),
                                    heldout_pass=int(args.unit_split_heldout_pass),
                                    heldout_fraction=float(args.unit_split_heldout_fraction), seed=42)
    if method == 'rift' and getattr(args, 'densify_epochs', None):
        # Plenoxel-style trilinear densification (rift_pvc/gotcha_densify.py, A41). It replaces the B787
        # spatial split (its fraction is set to zero here); the angular SH growth events are unchanged.
        recipe['densify'] = dict(schema='gotcha_trilinear_densify_v1', epochs=sorted(int(e) for e in args.densify_epochs),
                                 max_active=int(args.densify_max_active or args.max_points),
                                 energy_floor=float(args.densify_energy_floor),
                                 weight_scale=float(args.densify_weight_scale), normalize=args.densify_normalize,
                                 init=args.densify_init,
                                 lr_factor=float(args.densify_lr_factor), max_level=int(args.densify_max_level),
                                 lr_rule=args.densify_lr_rule, lr_target=float(args.densify_lr_target),
                                 curvature_units=int(args.densify_curvature_units),
                                 curvature_iterations=int(args.densify_curvature_iterations),
                                 initialization=('trilinear interpolation of the parent lattice at the child centre, '
                                                 'empty nodes zero; energy: one factor restoring the kept parents\' '
                                                 'coefficient energy, volume: times weight_scale; not render-preserving'
                                                 if args.densify_init == 'trilinear' else
                                                 'inherit: heir child takes the parent coefficients and exact position, '
                                                 'seven zero siblings; render-preserving'))
        if getattr(args, 'densify_curvature_guard', None):
            # Opt-in (A57): re-apply the curvature rule after SH growth and/or warm restarts; absent keeps
            # the recipe (and every earlier run's identity) unchanged.
            recipe['densify']['curvature_guard'] = sorted(set(args.densify_curvature_guard))
        if recipe.get('optimizer_schedule') is not None:
            recipe['optimizer_schedule'] = dict(recipe['optimizer_schedule'], spatial_fraction=0.0,
                                                spatial_fraction_note='zero: trilinear densify replaces the B787 split')
    if method == 'rift' and getattr(args, 'train_eval_every', 0):
        recipe['train_eval_every'] = int(args.train_eval_every)
    # Opt-in gain/start variants (A62, the user's gain-initialization experiment); absent keys keep the
    # single-pulse warm start, the shared optimizer group and the chosen base rate.
    if method == 'rift' and getattr(args, 'gain_warm_start', 'first_pulse') == 'pooled':
        recipe['gain_warm_start'] = dict(schema='gotcha_gain_warm_start_v1', rule='pooled',
                                         note='complex least-squares gain over every backprojection-start pulse '
                                              'in the loss domain, after the single-pulse warm start')
    if method == 'rift' and getattr(args, 'gain_lr', 0.0):
        recipe['gain_optimizer'] = dict(schema='gotcha_gain_group_v1', lr=float(args.gain_lr), eps=float(args.gain_eps),
                                        note='gain log-magnitude and phase in their own AdamW group, excluded from '
                                             'curvature cuts; the cosine schedule applies')
    if method == 'rift' and getattr(args, 'densify_curvature_start', False):
        recipe['densify']['curvature_start'] = True
    return recipe


GAIN_GROUP = 'gain'


def coefficient_base_lr(optimizer, scheduler):
    """The coefficient group's base rate (group 0): the curvature rule's step, never the gain group's."""
    return scheduler.base_lrs[0] if scheduler is not None else optimizer.param_groups[0]['lr']


def scale_rates_except_gain(optimizer, scheduler, factor):
    """``scale_learning_rates`` on every group but an opt-in gain group (A62), whose rate is its own."""
    gain = [i for i, g in enumerate(optimizer.param_groups) if g.get('name') == GAIN_GROUP]
    if not gain:
        return scale_learning_rates(optimizer, scheduler, factor)
    kept = [(optimizer.param_groups[i]['lr'], optimizer.param_groups[i].get('initial_lr'),
             scheduler.base_lrs[i] if scheduler is not None else None) for i in gain]
    eta_min = getattr(scheduler, 'eta_min', None)
    scale_learning_rates(optimizer, scheduler, factor)
    for i, (lr, initial, base) in zip(gain, kept):
        optimizer.param_groups[i]['lr'] = lr
        if initial is not None:
            optimizer.param_groups[i]['initial_lr'] = initial
        if scheduler is not None:
            scheduler.base_lrs[i] = base
    if eta_min is not None:
        scheduler.eta_min = eta_min * factor
    return [float(group['lr']) for group in optimizer.param_groups]


@torch.no_grad()
def pooled_gain_refit(head, dataset, readout, views, polarization, record):
    """Re-fit the warm-started complex gain by least squares over every backprojection-start pulse (A62).

    The single-pulse warm start (``GlobalComplexGain.maybe_init_scale`` on the first view's first pulse) is
    a draw: 2.13 at -6 deg on the 578 split, 0.69 at -44 deg on the full split (A61). The pooled projection
    <S_pred, S_meas> / ||S_pred||^2 over the start's pulses, in the loss domain, multiplies the gain; the
    gauged coefficients are unchanged, and m1, m2 and the prior weights are recomputed at the new gain.
    """
    full = nufft.loss_domain(head.recipe) == nufft.FULL
    cross, energy, pulses = 0j, 0.0, 0
    for p, sector in views:
        for obs in dataset.observations(p, sector, polarization):
            r = readout.for_observation(obs)
            predicted = head(obs, r)[0]
            target = _target(obs, predicted.device)
            if not full:
                predicted, target = RangeReadout.project(predicted, r), RangeReadout.project(target, r)
            cross += complex((predicted.conj() * target).sum().item())
            energy += float(predicted.abs().square().sum())
            pulses += 1
    if not energy > 0:
        raise ValueError('Pooled gain re-fit: the start renders nothing')
    projection = cross / energy
    first = head.gain.gain_value()
    head.gain.log_mag.add_(math.log(abs(projection)))
    head.gain.phase.add_(math.atan2(projection.imag, projection.real))
    field, mask = head.field, head.field.active_mask
    squared = field.w_re[mask].double().square().sum(-1) + field.w_im[mask].double().square().sum(-1)
    g = float(torch.exp(head.gain.log_mag))
    count = int(mask.sum())
    m1, m2 = g * float(squared.sqrt().sum()) / count, g * g * float(squared.sum()) / count
    reference = head.recipe['rift_dataset_reference']
    gauge = record['coefficient_gauge']
    after = head.gain.gain_value()
    return dict(record, warm_start_gain_first_pulse=record['warm_start_gain'],
                warm_start_gain=[after.real * gauge, after.imag * gauge],
                gain_warm_start=dict(rule='pooled', pulses=pulses, projection=[projection.real, projection.imag],
                                     gain_before=[first.real, first.imag], gain_after=[after.real, after.imag]),
                m1_first_pulse=record['m1'], m2_first_pulse=record['m2'], m1=m1, m2=m2,
                l1_weight=reference['mu1'] / m1, sh_degree_weight=reference['mu2'] / m2)


def epoch_end_with_prune(heads, recipe, schedule, optimizer, scheduler, epoch, initialization):
    """``b787_epoch_end`` with train.py's prune between the scheduler step and the refinement event.

    train.py's order at the end of an epoch: scheduler.step(), then a prune when due, then the
    adaptive refinement event when due. Used only when the recipe carries a sparsity key (``prune`` or
    ``mu1_scale``); every epoch's record then also carries the active count and the number of points
    holding 99% of the coefficient energy.
    """
    from train import _prune_target_for_epoch
    prune = recipe.get('prune')
    record = dict(learning_rates=[float(group['lr']) for group in optimizer.param_groups])
    scheduler.step()
    if prune is not None and (epoch + 1) % prune['every'] == 0 and (epoch + 1) >= prune['start_epoch']:
        record['prune'] = {}
        for pol, head in heads.items():
            scene = head.field
            ramp_from = int(initialization[pol]['prune_ramp_from'])
            target = _prune_target_for_epoch(epoch + 1, 'target', prune['target_active'], prune['start_epoch'],
                                             prune['end_epoch'], ramp_from)
            active, total, _ = scene.prune(threshold_fraction=prune['threshold'], criterion=prune['criterion'],
                                           mode='target', target_active=target, min_active=prune['min_active'],
                                           optimizer=optimizer)
            record['prune'][pol] = dict(target=int(target), active=int(active))
    if (epoch + 1) % recipe['refine_every'] == 0:
        record['refinement'] = {}
        for pol, head in heads.items():
            scene = head.field
            snapshot = scene.refinement_snapshot(
                recipe['max_level'], schedule['min_spatial_exposure'], schedule['min_angular_exposure'],
                spatial_floor=schedule['spatial_floor'], angular_floor=schedule['angular_floor'],
                cooldown_events=schedule['cooldown_events'], child_maturity_events=schedule['child_maturity_events'])
            split, grown, active, _ = scene.apply_refinement_snapshot(
                snapshot, schedule['spatial_fraction'], schedule['angular_fraction'], recipe['max_level'],
                optimizer=optimizer, max_active=recipe['max_points'])
            record['refinement'][pol] = dict(split=int(split), grown=int(grown), active=int(active))
    record['active_points'] = {pol: int(head.field.active_mask.sum()) for pol, head in heads.items()}
    record['points_99pct_energy'] = {pol: energy_concentration(head.field) for pol, head in heads.items()}
    return record


@torch.no_grad()
def energy_concentration(scene, fraction=0.99):
    """Fewest active points holding ``fraction`` of the unlocked coefficient energy (the global gain is common)."""
    unlocked = (scene.basis_degree.view(1, -1) <= scene.order[:, None]).to(scene.w_re.dtype)
    energy = ((scene.w_re.double().square() + scene.w_im.double().square()) * unlocked).sum(-1)[scene.active_mask]
    if energy.numel() == 0 or not float(energy.sum()) > 0:
        return 0
    cumulative = torch.cumsum(energy.sort(descending=True).values, 0)
    return int((cumulative < fraction * cumulative[-1]).sum()) + 1


def _target(observation, device):
    return torch.as_tensor(observation.response, dtype=torch.complex128, device=device)


def training_statistics(dataset, readout):
    """One bounded-memory TRAIN-only pass; no normalization from held-out data."""
    sums = {pol:dict(energy=0., count=0, range_peak=0.) for pol in dataset.polarizations}
    for p, sector in dataset.viewpoints('train'):
        for pol in dataset.polarizations:
            s = sums[pol]
            for observation in dataset.observations(p, sector, pol):
                r = readout.for_observation(observation)
                y = _target(observation, readout.device)
                projected = readout.project(y, r)
                s['energy'] += projected.abs().square().sum().item()
                s['count'] += projected.numel()
                s['range_peak'] = max(s['range_peak'], (r['matched']@y).abs().square().max().item())
    for s in sums.values():
        if not s['count'] or not math.isfinite(s['energy']) or s['energy'] <= 0 or s['range_peak'] <= 0:
            raise ValueError('TRAIN normalization is empty, nonfinite or zero')
        s['mean_power'] = s['energy']/s['count']
    return sums


def objective(method, prediction, observation, r, stats):
    y = _target(observation, prediction.device)
    if stats.get('loss_domain') == nufft.FULL:
        # Full-native control: train.py's complex MSE over every selected bin; the projected target is
        # still returned for the recorded ROI metric.
        return (prediction-y).abs().square().mean()/stats['mean_power'], RangeReadout.project(y, r)
    pred, target = RangeReadout.project(prediction, r), RangeReadout.project(y, r)
    diff = (pred-target).abs().square().mean()
    return diff/stats['mean_power'], target


def evaluate(heads, dataset, readout, stats, method, role='validation'):
    totals = {pol:dict(squared_error=0., target_energy=0., full_native_squared_error=0.,
                      full_native_target_energy=0., pulses=0) for pol in dataset.polarizations}
    heads.eval()
    with torch.no_grad():
        for p, sector in dataset.viewpoints(role):
            for pol in dataset.polarizations:
                for obs in dataset.observations(p, sector, pol):
                    r = readout.for_observation(obs)
                    pred, _ = heads[pol](obs, r)
                    _, target = objective(method, pred, obs, r, stats[pol])
                    scored = readout.project(pred, r)
                    t = totals[pol]
                    t['squared_error'] += (scored-target).abs().square().sum().item()
                    t['target_energy'] += target.abs().square().sum().item()
                    y = _target(obs, pred.device)
                    t['full_native_squared_error'] += (pred-y).abs().square().sum().item()
                    t['full_native_target_energy'] += y.abs().square().sum().item()
                    t['pulses'] += 1
    error, energy = sum(t['squared_error'] for t in totals.values()), sum(t['target_energy'] for t in totals.values())
    if not math.isfinite(error) or not math.isfinite(energy) or energy <= 0:
        raise ValueError('Invalid evaluation aggregate')
    domain = 'roi_projected_native_complex'
    full_error = sum(t['full_native_squared_error'] for t in totals.values())
    full_energy = sum(t['full_native_target_energy'] for t in totals.values())
    if not math.isfinite(full_error) or not math.isfinite(full_energy) or full_energy <= 0:
        raise ValueError('Invalid full-native evaluation aggregate')
    return dict(role=role, metric_domain=domain, pooled_rel_mse=error/energy, by_polarization=totals,
                full_native_complex_rel_mse=full_error/full_energy,
                viewpoints=len(dataset.viewpoints(role)), test_accessed=False,
                roi_qualification='range-compatible clutter can remain; not an isolated vehicle response')


def batched_evaluate(heads, dataset, readout, stats, method, role='validation'):
    """``evaluate`` with one forward per pass-sector/channel; identical aggregates."""
    totals = {pol:dict(squared_error=0., target_energy=0., full_native_squared_error=0.,
                      full_native_target_energy=0., pulses=0) for pol in dataset.polarizations}
    heads.eval()
    with torch.no_grad():
        for p, sector in dataset.viewpoints(role):
            for pol in dataset.polarizations:
                head = heads[pol]
                obs = list(dataset.observations(p, sector, pol))
                rs = [readout.for_observation(o) for o in obs]
                pred, _ = sector_forward(head, obs, rs, method=method, point_chunk=head.recipe['point_chunk'])
                y = torch.stack([_target(o, pred.device) for o in obs])
                error, energy = projected_error_sums(pred, y, rs)
                t = totals[pol]
                t['squared_error'] += error.item()
                t['target_energy'] += energy.item()
                t['full_native_squared_error'] += (pred-y).abs().square().sum().item()
                t['full_native_target_energy'] += y.abs().square().sum().item()
                t['pulses'] += len(obs)
    error, energy = sum(t['squared_error'] for t in totals.values()), sum(t['target_energy'] for t in totals.values())
    if not math.isfinite(error) or not math.isfinite(energy) or energy <= 0:
        raise ValueError('Invalid evaluation aggregate')
    domain = 'roi_projected_native_complex'
    full_error = sum(t['full_native_squared_error'] for t in totals.values())
    full_energy = sum(t['full_native_target_energy'] for t in totals.values())
    if not math.isfinite(full_error) or not math.isfinite(full_energy) or full_energy <= 0:
        raise ValueError('Invalid full-native evaluation aggregate')
    return dict(role=role, metric_domain=domain, pooled_rel_mse=error/energy, by_polarization=totals,
                full_native_complex_rel_mse=full_error/full_energy,
                viewpoints=len(dataset.viewpoints(role)), test_accessed=False,
                roi_qualification='range-compatible clutter can remain; not an isolated vehicle response')


@torch.no_grad()
def role_fit(heads, dataset, readout, method, role='train'):
    """Full-native fit of every unit of ``role``: RelMSE, energy ratio e and real correlation rho (A41).

    RelMSE = 1 + e - 2 rho sqrt(e). TRAIN is judged on RelMSE with rho and e/rho^2 printed beside it, so
    a scene that shrinks toward zero is never read as a better fit.
    """
    sums = dict(error=0., target=0., prediction=0., cross=0., units=0, pulses=0)
    was_training = heads.training
    heads.eval()
    for p, sector in dataset.viewpoints(role):
        for pol in dataset.polarizations:
            head = heads[pol]
            obs = list(dataset.observations(p, sector, pol))
            rs = [readout.for_observation(o) for o in obs]
            pred, _ = sector_forward(head, obs, rs, method=method, point_chunk=head.recipe['point_chunk'])
            y = torch.stack([_target(o, pred.device) for o in obs])
            sums['error'] += float((pred - y).abs().square().sum())
            sums['target'] += float(y.abs().square().sum())
            sums['prediction'] += float(pred.abs().square().sum())
            sums['cross'] += float((pred * y.conj()).real.sum())
            sums['pulses'] += len(obs)
        sums['units'] += 1
    heads.train(was_training)
    if not sums['target'] > 0:
        raise ValueError('Empty or zero-energy role for the fit statistics')
    e = sums['prediction'] / sums['target']
    rho = sums['cross'] / math.sqrt(max(sums['prediction'] * sums['target'], 1e-300))
    return dict(role=role, full_native_rel_mse=sums['error'] / sums['target'], energy_ratio=e, correlation=rho,
                e_over_rho2=(e / rho ** 2 if rho != 0 else None), units=sums['units'], pulses=sums['pulses'])


def update_curvature(head, dataset, readout, view, polarization, mean_power, method, *, iterations=10, seed=0):
    """lambda_max of one update's full-native data-loss Hessian in (w_re, w_im), at the current gain.

    ``scripts_pvc/gotcha_rift_fit_check.py``'s power iteration, rendered with the batched sector
    forward: the prediction is linear in the coefficients, so H v = (2 / (n P)) grad_w
    sum_pulses Re<J v, yhat(w)> / n_bins, and the value does not depend on the data.
    """
    field = head.field
    obs = list(dataset.observations(*view, polarization))
    rs = [readout.for_observation(o) for o in obs]
    chunk = head.recipe['point_chunk']
    saved = field.w_re.detach().clone(), field.w_im.detach().clone()
    generator = torch.Generator().manual_seed(seed)
    v = [torch.randn(w.shape, generator=generator).to(w) for w in (field.w_re, field.w_im)]
    estimates = []
    try:
        for _ in range(iterations):
            with torch.no_grad():
                field.w_re.copy_(v[0])
                field.w_im.copy_(v[1])
                jv = sector_forward(head, obs, rs, method=method, point_chunk=chunk)[0]
                field.w_re.copy_(saved[0])
                field.w_im.copy_(saved[1])
            out = sector_forward(head, obs, rs, method=method, point_chunk=chunk)[0]
            f = (jv.conj() * out).real.sum() / out.shape[-1]
            hv = [2 * g.double() / (len(obs) * mean_power) for g in torch.autograd.grad(f, (field.w_re, field.w_im))]
            norm_v = float(sum(x.double().square().sum() for x in v).sqrt())
            norm_hv = float(sum(x.square().sum() for x in hv).sqrt())
            estimates.append(norm_hv / norm_v)
            v = [(x / norm_hv).to(field.w_re) for x in hv]
    finally:
        with torch.no_grad():
            field.w_re.copy_(saved[0])
            field.w_im.copy_(saved[1])
    return dict(view=list(view), lambda_max=estimates[-1], last_iterations=estimates[-3:])


def densify_heads(heads, recipe, optimizer, scheduler, *, dataset=None, readout=None, stats=None):
    """One trilinear densify event on every channel, then the recipe's learning-rate rule (A41/A42).

    ``lr_rule='fixed'`` multiplies every rate by ``lr_factor``. ``lr_rule='curvature'`` measures the
    per-update Hessian's lambda_max on ``curvature_units`` TRAIN units at the current gain and scales
    every rate so the step's stability number (base lr / eps) lambda_max is at most ``lr_target``
    (never raised; the cosine base is used, so a warm restart stays below the target too).
    """
    spec = recipe['densify']
    record = {pol: trilinear_densify(head.field, max_active=spec['max_active'], energy_floor=spec['energy_floor'],
                                     weight_scale=spec['weight_scale'], max_level=spec['max_level'],
                                     normalize=spec.get('normalize', 'volume'), init=spec.get('init', 'trilinear'),
                                     optimizer=optimizer, birth_event=int(head.field.refine_event_count))
              for pol, head in heads.items()}
    densified = any(r.get('status') == 'densified' for r in record.values())
    rule = spec.get('lr_rule', 'fixed')
    if densified and rule == 'fixed' and spec['lr_factor'] != 1.0:
        record['learning_rates_after'] = scale_rates_except_gain(optimizer, scheduler, spec['lr_factor'])
    elif densified and rule == 'curvature':
        record.update(curvature_rule(heads, recipe, optimizer, scheduler, dataset=dataset, readout=readout, stats=stats))
    record['active_points'] = {pol: int(head.field.active_mask.sum()) for pol, head in heads.items()}
    return record


START_ITERATION_FACTOR = 3


def curvature_rule(heads, recipe, optimizer, scheduler, *, dataset, readout, stats, allow_raise=False):
    """The densify curvature rule on the current field: measure lambda_max on the recipe's TRAIN units and
    lower every rate and the cosine base so (base lr / eps) lambda_max <= lr_target (never raised). An opt-in
    gain group keeps its rate.

    ``allow_raise`` is the opt-in start rule (A62): it sets S equal to the target. A raise applies to the
    coefficient group (group 0) only, the block whose Hessian is measured; positions keep their rate, since
    their curvature is not measured (B69 review). A cut still scales every non-gain group. The start rule runs
    START_ITERATION_FACTOR x the power iterations and records convergence, because an underestimated
    lambda_max would be amplified by a raise.
    """
    spec = recipe['densify']
    if nufft.loss_domain(recipe) != nufft.FULL:
        raise ValueError('The curvature learning-rate rule is implemented for the full-native loss')
    views = dataset.viewpoints('train')
    picks = [views[int(i * len(views) / spec['curvature_units'])] for i in range(spec['curvature_units'])]
    iterations = spec['curvature_iterations'] * (START_ITERATION_FACTOR if allow_raise else 1)
    rows = [dict(update_curvature(head, dataset, readout, view, pol, stats[pol]['mean_power'], 'rift',
                                  iterations=iterations), polarization=pol)
            for pol, head in heads.items() for view in picks]
    if allow_raise:
        for r in rows:
            last = r['last_iterations']
            r['converged'] = len(last) >= 2 and abs(last[-1] - last[-2]) <= 1e-3 * last[-1]
    lam = max(r['lambda_max'] for r in rows)
    if allow_raise or any(g.get('name') == GAIN_GROUP for g in optimizer.param_groups):
        base = coefficient_base_lr(optimizer, scheduler)
    else:   # as before the gain group existed (bit for bit)
        base = max(scheduler.base_lrs) if scheduler is not None else max(g['lr'] for g in optimizer.param_groups)
    eps = optimizer.param_groups[0]['eps']
    before = base / eps * lam
    factor = spec['lr_target'] / before if allow_raise else min(1.0, spec['lr_target'] / before)
    record = dict(curvature=dict(units=rows, lambda_max=lam, base_lr=base, eps=eps,
                                 stability_before=before, factor=factor, stability_after=before * factor,
                                 target=spec['lr_target'],
                                 **(dict(allow_raise=True, iterations=iterations,
                                         raised_groups=[0] if factor > 1 else 'all but gain') if allow_raise else {})))
    if factor > 1.0:
        record['learning_rates_after'] = raise_coefficient_rate(optimizer, scheduler, factor)
    elif factor < 1.0:
        record['learning_rates_after'] = scale_rates_except_gain(optimizer, scheduler, factor)
    return record


def raise_coefficient_rate(optimizer, scheduler, factor):
    """Raise only group 0 (the coefficients) and its cosine base by ``factor`` (> 1; the start rule, A62)."""
    if not factor > 1:
        raise ValueError('raise_coefficient_rate raises; cuts go through scale_rates_except_gain')
    group = optimizer.param_groups[0]
    group['lr'] *= factor
    if 'initial_lr' in group:
        group['initial_lr'] *= factor
    if scheduler is not None:
        scheduler.base_lrs[0] *= factor
    return [float(g['lr']) for g in optimizer.param_groups]


def curvature_guard_due(recipe, schedule_record, scheduler):
    """Triggers of the opt-in curvature guard (A57) after one B787 epoch end, or [].

    'growth': the epoch's refinement event unlocked an SH band on some point (grown > 0), which raises
    that point's block curvature by (L+2)^2/(L+1)^2. 'restart': the scheduler step just restarted the
    cosine, so the next epoch runs at the base rate. The densify rule's own measurement covers an epoch
    that also densifies; the caller runs the guard only on epochs without an event.
    """
    guard = (recipe.get('densify') or {}).get('curvature_guard') or []
    triggers = []
    if 'growth' in guard and any(int(r.get('grown', 0)) > 0
                                 for r in ((schedule_record or {}).get('refinement') or {}).values()):
        triggers.append('growth')
    if 'restart' in guard and scheduler is not None and getattr(scheduler, 'T_cur', None) == 0:
        triggers.append('restart')
    return triggers


def guard_heads(heads, recipe, optimizer, scheduler, triggers, *, dataset, readout, stats):
    """One curvature-guard measurement (A57): the densify curvature rule without an event."""
    return dict(triggers=list(triggers), **curvature_rule(heads, recipe, optimizer, scheduler,
                                                           dataset=dataset, readout=readout, stats=stats))


def _rng_state(rng):
    # Accelerator payload keyed by backend ('cuda' key kept for the CUDA layout).
    backend = accelerator.backend()
    device_states = accelerator.get_rng_state_all() if accelerator.is_available() else None
    return dict(numpy=rng.bit_generator.state, torch=torch.get_rng_state(),
                cuda=device_states if backend == 'cuda' else None,
                accelerator_backend=backend if accelerator.is_available() else None,
                accelerator=device_states)


def _restore_rng(payload, rng):
    rng.bit_generator.state = payload['numpy']
    torch.set_rng_state(payload['torch'].cpu())
    if accelerator.is_available():
        backend = accelerator.backend()
        states = payload.get('accelerator') if payload.get('accelerator_backend') == backend else (
            payload.get('cuda') if backend == 'cuda' else None)
        if states is None:
            origin = payload.get('accelerator_backend') or ('cuda' if payload.get('cuda') is not None else 'none')
            raise ValueError(f'{backend.upper()} resume requires saved {backend.upper()} RNG state '
                             f'(checkpoint carries {origin} state)')
        accelerator.set_rng_state_all([value.cpu() for value in states])


def validate_resume_progress(saved, dataset, recipe):
    """Check continuation progress/statistics before creating outputs or reading responses."""
    size = len(dataset.viewpoints('train'))
    epoch, cursor, updates = (saved.get(k) for k in ('epoch','cursor','updates'))
    if any(type(v) is not int for v in (epoch,cursor,updates)) or not 0 <= epoch <= recipe['epochs'] or not 0 <= cursor <= size:
        raise ValueError('Invalid saved epoch/cursor/update state')
    if updates != epoch*size+cursor:
        raise ValueError('Saved updates do not agree with the pass-sector epoch/cursor')
    order = saved.get('order')
    if order is None:
        if cursor != 0:
            raise ValueError('Mid-epoch checkpoint lacks its viewpoint order')
    elif (not isinstance(order,list) or any(type(i) is not int for i in order)
          or sorted(order) != list(range(size))):
        raise ValueError('Saved viewpoint order is not the exact TRAIN permutation')
    if epoch == recipe['epochs'] and (cursor != 0 or order is not None):
        raise ValueError('Completed checkpoint has incomplete-epoch state')
    stats = saved.get('training_statistics')
    if not isinstance(stats,dict) or set(stats) != set(dataset.polarizations):
        raise ValueError('Missing or incompatible TRAIN normalization statistics')
    for values in stats.values():
        if (not isinstance(values,dict) or type(values.get('count')) is not int or values['count'] <= 0
                or any(not isinstance(values.get(k),(int,float)) or not math.isfinite(values[k]) or values[k] <= 0
                       for k in ('energy','mean_power','range_peak'))
                or not math.isclose(values['energy']/values['count'],values['mean_power'],rel_tol=1e-12)):
            raise ValueError('Malformed TRAIN normalization statistics')
    if not isinstance(saved.get('history'),list) or len(saved['history']) != epoch:
        raise ValueError('Checkpoint history does not match completed epochs')
    if saved.get('best_val') is not None and (not math.isfinite(saved['best_val']) or saved['best_val'] < 0):
        raise ValueError('Invalid saved selection metric')
    for name in ('model_state_dict','optimizer_state_dict','rng_state'):
        if not isinstance(saved.get(name),dict):
            raise ValueError(f'Checkpoint lacks {name}')
    if recipe.get('optimizer') == B787_OPTIMIZER:
        if not isinstance(saved.get('scheduler_state_dict'), dict):
            raise ValueError('Checkpoint lacks scheduler_state_dict')
        if order is not None and order != fixed_view_order(recipe['seed'], size):
            raise ValueError('Saved viewpoint order is not the fixed B787 training order')


def train(dataset, method, recipe, output, *, device=None, resume=None):
    device = accelerator.device() if device is None else device
    """Run only when explicitly invoked inside an experiment allocation."""
    from rift.gotcha_frequency_selection import bind_recipe
    recipe = bind_recipe(dataset, recipe)
    execution = recipe.get('sector_execution', LOOP)
    if execution not in (LOOP, BATCHED):
        raise ValueError(f'Unknown sector_execution {execution!r}')
    batched = execution == BATCHED
    output = Path(output)
    torch.manual_seed(recipe['seed'])
    rng = np.random.Generator(np.random.PCG64(recipe['seed']))
    saved = None
    if resume is not None:
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        if isinstance(saved.get('recipe'), dict):
            # Keys added later keep their earlier meaning (unit amplitude, random start, no priors).
            saved['recipe'] = with_legacy_keys(saved['recipe'], method)
        # Gate recipe, selection and region before any response access.
        validate_checkpoint(saved, dataset, recipe)
        validate_resume_progress(saved, dataset, recipe)
    if output.exists() and any(output.iterdir()) and saved is None:
        raise ValueError('Output exists: select a new output root or an explicit matching resume')
    output.mkdir(parents=True, exist_ok=True)
    heads = nn.ModuleDict({pol:ChannelField(method, dataset.region, recipe, device) for pol in dataset.polarizations})
    nufft.attach_grids(heads, dataset, recipe, device)
    positions = [h.field.delta_raw for h in heads.values() if method == 'rift']
    position_ids = {id(p) for p in positions}
    gain_spec = recipe.get('gain_optimizer') if method == 'rift' else None
    gain_params = [p for h in heads.values() for p in h.gain.parameters()] if gain_spec else []
    gain_ids = {id(p) for p in gain_params}
    params = [p for p in heads.parameters() if id(p) not in position_ids and id(p) not in gain_ids]
    groups = [dict(params=params, lr=recipe['lr'])]
    if positions:
        groups.append(dict(params=positions, lr=recipe['pos_lr']))
    if gain_params:
        # Opt-in (A62): the two gain scalars per channel in their own group, last, so group 0 stays the
        # coefficients' (the curvature rule's base and eps).
        groups.append(dict(params=gain_params, lr=gain_spec['lr'], eps=gain_spec['eps'], name=GAIN_GROUP))
    optimizer = torch.optim.AdamW(groups, eps=recipe['adam_eps'], weight_decay=0)
    # The B787 schedule (adaptive RIFT default); None keeps the earlier constant-LR optimizer.
    schedule = b787_schedule(method, recipe)
    scheduler = build_scheduler(optimizer, schedule)
    readout = RangeReadout(dataset.region, device=device)
    train_views = dataset.viewpoints('train')
    epoch, cursor, history, best, updates, order = 0, 0, [], None, 0, None
    if saved is not None:
        heads.load_state_dict(saved['model_state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        for p, state in optimizer.state.items():
            for key, value in state.items():
                if torch.is_tensor(value) and key != 'step':
                    state[key] = value.to(p.device)
        if scheduler is not None:
            scheduler.load_state_dict(saved['scheduler_state_dict'])
        stats = saved['training_statistics']
        initialization = saved.get('initialization') or {}
        epoch, cursor, history, best, updates, order = (saved[k] for k in ('epoch', 'cursor', 'history', 'best_val', 'updates', 'order'))
        _restore_rng(saved['rng_state'], rng)
    else:
        stats, initialization = (nufft.full_native_statistics(dataset, readout) if nufft.loss_domain(recipe) == nufft.FULL
                                 else training_statistics(dataset, readout)), {}
        if method == 'rift' and recipe.get('initialization') == BACKPROJECTION:
            # The epoch-1 order the loop would draw here; its first views seed the start.
            order = rng.permutation(len(train_views)).tolist()
            if schedule is not None and order != fixed_view_order(recipe['seed'], len(train_views)):
                raise AssertionError('The epoch-1 order must be the fixed B787 training order')
            views = [train_views[i] for i in order[:recipe['bp_views']]]
            start = nufft.control_initialization if nufft.control_keys(recipe) else rift_dataset_initialization
            initialization = {pol: start(head, dataset, readout, views, pol) for pol, head in heads.items()}
            if recipe.get('gain_warm_start') is not None:
                initialization = {pol: pooled_gain_refit(head, dataset, readout, views, pol, initialization[pol])
                                  for pol, head in heads.items()}
            if (recipe.get('densify') or {}).get('curvature_start'):
                # Opt-in (A62): one curvature measurement at the start sets the base so S = target (may raise).
                start_rule = curvature_rule(heads, recipe, optimizer, scheduler, dataset=dataset, readout=readout,
                                            stats=stats, allow_raise=True)
                for pol in initialization:
                    initialization[pol]['curvature_start'] = start_rule
            for pol, head in heads.items():
                if recipe.get('mu1_scale') is not None:
                    # Opt-in sparsity lever: the dimensionless mu1 times the scale, so l1_weight scales alike.
                    initialization[pol]['l1_weight_unscaled'] = initialization[pol]['l1_weight']
                    initialization[pol]['l1_weight'] = initialization[pol]['l1_weight'] * recipe['mu1_scale']
                    initialization[pol]['mu1_scale'] = recipe['mu1_scale']
                if recipe.get('prune') is not None:
                    initialization[pol]['prune_ramp_from'] = int(head.field.active_mask.sum())
            atomic_json(output/'initialization.json', initialization)
    priors = method == 'rift' and recipe.get('priors') == RIFT_DATASET_PRIORS
    if priors and set(initialization) != set(dataset.polarizations):
        raise ValueError('RIFT-dataset priors need the backprojection-start record of every channel')
    atomic_json(output/'dataset.json', dict(summary=dataset.summary(), contract=dataset.contract))
    atomic_json(output/'recipe.json', recipe)
    stop = {'requested':False}
    previous_handlers = {}
    def request_stop(*_):
        stop['requested'] = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[sig] = signal.signal(sig, request_stop)
    # The running TRAIN metric (A41) is saved with the cursor, so a resumed epoch reports what an
    # uninterrupted one would.
    running = None
    if saved is not None and saved.get('train_running_state') is not None and saved['cursor'] > 0:
        running = dict(saved['train_running_state'])

    def save(name):
        atomic_save(output/name, dict(schema='rift_gotcha_checkpoint_v1', dataset_contract=dataset.contract,
                    dataset_identity=dataset.identity, recipe=recipe, model_state_dict=heads.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(), training_statistics=stats,
                    initialization=initialization,
                    epoch=epoch, cursor=cursor, history=history, best_val=best, updates=updates,
                    order=order, rng_state=_rng_state(rng),
                    **({'train_running_state': {k: float(v) if torch.is_tensor(v) else v for k, v in running.items()}}
                       if running is not None else {}),
                    **({'scheduler_state_dict': scheduler.state_dict()} if scheduler is not None else {})))
    try:
        if (method == 'rift' and history and densify_due(recipe, epoch) and cursor == 0 and epoch < recipe['epochs']
                and 'densify' not in (history[-1].get('optimizer') or {})):
            # The epoch ended and was saved, but its densify event did not complete: run it now.
            history[-1].setdefault('optimizer', {})['densify'] = densify_heads(heads, recipe, optimizer, scheduler,
                                                                              dataset=dataset, readout=readout,
                                                                              stats=stats)
            save('checkpoint_latest.pt')
            atomic_json(output/'history.json', history)
        elif (method == 'rift' and history and cursor == 0 and epoch < recipe['epochs'] and not densify_due(recipe, epoch)
              and 'curvature_guard' not in (history[-1].get('optimizer') or {})
              and curvature_guard_due(recipe, history[-1].get('optimizer'), scheduler)):
            # The same for an interrupted curvature-guard measurement (A57).
            history[-1]['optimizer']['curvature_guard'] = guard_heads(
                heads, recipe, optimizer, scheduler, curvature_guard_due(recipe, history[-1].get('optimizer'), scheduler),
                dataset=dataset, readout=readout, stats=stats)
            save('checkpoint_latest.pt')
            atomic_json(output/'history.json', history)
        while epoch < recipe['epochs']:
            if order is None:
                order = (fixed_view_order(recipe['seed'], len(train_views)) if schedule is not None
                         else rng.permutation(len(train_views)).tolist())
            heads.train()
            if cursor == 0 or running is None:
                running = (dict(error=0., target=0., units=0) if method == 'rift' and batched
                           and nufft.loss_domain(recipe) == nufft.FULL else None)
            # Opt-in train.py --step-every N: the gradients of N pass-sectors are summed (train.py's
            # sum, not a mean) before one optimizer step; the epoch's last group may be shorter.
            step_every = int(recipe.get('step_every', 1))
            pending = 0
            while cursor < len(order):
                p, sector = train_views[order[cursor]]
                if pending == 0:
                    optimizer.zero_grad(set_to_none=True)
                probing = probe_due(recipe, schedule, updates=updates, cursor=cursor, epoch=epoch)
                # One optimizer step per pass/sector. Ragged channels contribute
                # equally; native pulse counts within a channel are preserved.
                for pol in dataset.polarizations:
                    head = heads[pol]
                    count = len(dataset.shards[p,pol].sector_rows[sector])
                    if batched:
                        # All selected pulses of this sector/channel in one
                        # forward and one backward; per-pulse statistics exact.
                        obs = list(dataset.observations(p, sector, pol))
                        rs = [readout.for_observation(o) for o in obs]
                        if not bool(head.scale_initialized):
                            for o, r in zip(obs, rs):
                                head.initialize_scale(o, r)
                        scale = count*len(dataset.polarizations)
                        targets = torch.stack([_target(o, device) for o in obs])
                        losses, delta_grad, angular = sector_update(
                            head, obs, rs, targets, stats[pol]['mean_power'], scale, method=method,
                            point_chunk=recipe['point_chunk'], probe=probing)
                        if running is not None:
                            # Pooled full-native RelMSE of the epoch's updates, measured as they train (A41).
                            running['error'] = running['error'] + losses.sum()
                            running['target'] = running['target'] + (targets.abs().square().mean(dim=1)
                                                                     / stats[pol]['mean_power']).sum()
                            running['units'] += 1
                        if method == 'rift':
                            accumulate_batched_stats(head.field, delta_grad, angular)
                        if priors:
                            prior_backward(head, initialization[pol], len(dataset.polarizations))
                        continue
                    for obs in dataset.observations(p, sector, pol):
                        r = readout.for_observation(obs)
                        head.initialize_scale(obs, r)
                        pred, extra = head(obs, r)
                        data_loss, _ = objective(method, pred, obs, r, stats[pol])
                        scale = count*len(dataset.polarizations)
                        if not torch.isfinite(data_loss):
                            raise ValueError('Nonfinite training loss')
                        if method == 'rift':
                            scene = head.field
                            pos_grad = torch.autograd.grad(data_loss, scene.delta_raw, retain_graph=True)[0]
                            angular = (None, None)
                            if probing:
                                probe, _ = head(obs, r, probe=True)
                                probe_loss, _ = objective(method, probe, obs, r, stats[pol])
                                angular = torch.autograd.grad(probe_loss, (scene.w_re, scene.w_im))
                            scene.accumulate_refinement_data_stats(pos_grad, *angular)
                        loss = data_loss
                        (loss/scale).backward()
                    if priors:
                        prior_backward(head, initialization[pol], len(dataset.polarizations))
                pending += 1
                if pending == step_every or cursor + 1 == len(order) or stop['requested']:
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in heads.parameters()):
                        raise ValueError('Nonfinite parameter gradient')
                    optimizer.step()
                    pending = 0
                updates += 1
                cursor += 1
                if method == 'rift' and schedule is None and updates % recipe['refine_every'] == 0:
                    for head in heads.values():
                        scene = head.field
                        snap = scene.refinement_snapshot(recipe['max_level'], 2, 2, cooldown_events=1, child_maturity_events=1)
                        scene.apply_refinement_snapshot(snap, recipe['refine_fraction'], recipe['refine_fraction'],
                                                        recipe['max_level'], optimizer=optimizer, max_active=recipe['max_points'])
                if (updates % recipe['checkpoint_every'] == 0 and pending == 0) or stop['requested']:
                    save('checkpoint_latest.pt')
                if stop['requested']:
                    return dict(status='interrupted', completed_epochs=epoch, updates=updates)
            # As train.py: scheduler step and any refinement event precede validation.
            schedule_record = (None if schedule is None else
                               epoch_end_with_prune(heads, recipe, schedule, optimizer, scheduler, epoch, initialization)
                               if recipe.get('prune') is not None or recipe.get('mu1_scale') is not None else
                               b787_epoch_end(heads, recipe, schedule, optimizer, scheduler, epoch))
            metrics = nufft.with_selection((batched_evaluate if batched else evaluate)(heads, dataset, readout, stats, method),
                                           recipe)
            epoch += 1
            cursor, order = 0, None
            extra = {}
            if running is not None and running['units']:
                extra['train_running'] = dict(full_native_rel_mse=float(running['error']) / float(running['target']),
                                              units=running['units'], measured='during the epoch, as each unit trained')
            every = int(recipe.get('train_eval_every', 0))
            if method == 'rift' and (recipe.get('densify') is not None or every) and (
                    epoch == recipe['epochs'] or (every and epoch % every == 0)):
                extra['train'] = role_fit(heads, dataset, readout, method, 'train')
                extra['validation_fit'] = role_fit(heads, dataset, readout, method, 'validation')
            if method == 'rift' and (recipe.get('gain_optimizer') or recipe.get('gain_warm_start')
                                     or (recipe.get('densify') or {}).get('curvature_start')):
                # Opt-in gain/start variants (A62) record the gain trajectory per epoch.
                extra['gain'] = {pol: [head.gain.gain_value().real, head.gain.gain_value().imag]
                                 for pol, head in heads.items()}
            history.append(dict(epoch=epoch, updates=updates, validation=metrics, **extra,
                                **({'priors': prior_terms(heads, initialization)} if priors else {}),
                                **({'optimizer': schedule_record} if schedule is not None else {})))
            if best is None or nufft.selection_value(metrics) < best:
                best = nufft.selection_value(metrics)
                save('checkpoint_best.pt')
            save('checkpoint_latest.pt')
            atomic_json(output/'history.json', history)
            print(json.dumps(dict(method=method, epoch=epoch, validation_rel_mse=metrics['pooled_rel_mse'],
                                  **({'train_running_rel_mse': extra['train_running']['full_native_rel_mse']}
                                     if 'train_running' in extra else {}),
                                  **({'train_rel_mse': extra['train']['full_native_rel_mse'],
                                      'train_rho': extra['train']['correlation']} if 'train' in extra else {}))),
                  flush=True)
            if method == 'rift' and schedule is not None and densify_due(recipe, epoch) and epoch < recipe['epochs']:
                # After the epoch's own validation and saves, so the history keeps the trained, pre-event
                # state; the resume guard above re-runs an event a stop interrupted.
                history[-1]['optimizer']['densify'] = densify_heads(heads, recipe, optimizer, scheduler,
                                                                    dataset=dataset, readout=readout, stats=stats)
                save('checkpoint_latest.pt')
                atomic_json(output/'history.json', history)
                print(json.dumps(dict(method=method, epoch=epoch, densify=history[-1]['optimizer']['densify'])), flush=True)
            elif (method == 'rift' and schedule is not None and epoch < recipe['epochs']
                  and curvature_guard_due(recipe, schedule_record, scheduler)):
                # Opt-in (A57): SH growth or a warm restart without a densify event re-measures the step.
                history[-1]['optimizer']['curvature_guard'] = guard_heads(
                    heads, recipe, optimizer, scheduler, curvature_guard_due(recipe, schedule_record, scheduler),
                    dataset=dataset, readout=readout, stats=stats)
                save('checkpoint_latest.pt')
                atomic_json(output/'history.json', history)
                print(json.dumps(dict(method=method, epoch=epoch,
                                      curvature_guard=history[-1]['optimizer']['curvature_guard'])), flush=True)
        save('checkpoint_final.pt')
        return dict(status='complete', completed_epochs=epoch, updates=updates, best_validation_rel_mse=best, test_accessed=False)
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def backproject(dataset, recipe, output, *, device=None):
    device = accelerator.device() if device is None else device
    """Coherent TRAIN-only matched-filter support, not a learned NVS model."""
    from rift.gotcha_frequency_selection import bind_recipe
    recipe = bind_recipe(dataset, recipe)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Matched-filter output exists; select a new output root')
    output.mkdir(parents=True, exist_ok=True)
    points = _grid(recipe['granularity'], dataset.region.half_extent_m, device)
    readout = RangeReadout(dataset.region, device=device)
    clouds = {pol:torch.zeros(len(points), dtype=torch.complex128, device=device) for pol in dataset.polarizations}
    counts = {pol:0 for pol in dataset.polarizations}
    with torch.no_grad():
        for p, sector in dataset.viewpoints('train'):
            for pol in dataset.polarizations:
                for obs in dataset.observations(p, sector, pol):
                    r = readout.for_observation(obs)
                    y = _target(obs, device)
                    y = readout.lift(readout.project(y, r), r)
                    for start in range(0, len(points), recipe['point_chunk']):
                        x = points[start:start+recipe['point_chunk']]
                        delta = torch.linalg.vector_norm(x-r['antenna'], dim=-1)-obs.reference_range_m
                        kernel = torch.exp((4j*math.pi/C)*delta[:,None]*r['frequencies'][None,:])
                        clouds[pol][start:start+len(x)] += kernel@y
                    counts[pol] += len(y)
    atomic_save(output/'support.pt', dict(schema='gotcha_mfbp_support_v1', dataset_contract=dataset.contract,
                dataset_identity=dataset.identity, recipe=recipe, local_positions_m=points.cpu(),
                coefficients={p:(v/counts[p]).cpu() for p,v in clouds.items()},
                native_samples=counts, test_accessed=False))
    return dict(status='complete', method='mfbp', training_viewpoints=len(dataset.viewpoints('train')),
                geometry_only=True, test_accessed=False)
