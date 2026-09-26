"""Native-frequency GOTCHA training backends, independent of frozen B787 CLIs.

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
from torch.utils.checkpoint import checkpoint as recompute

from rift.b7873200_adaptive_fullscale import (
    FULLSCALE_ANGULAR_FLOOR, FULLSCALE_ANGULAR_FRACTION, FULLSCALE_CHILD_MATURITY_EVENTS, FULLSCALE_COOLDOWN_EVENTS,
    FULLSCALE_MIN_ANGULAR_EXPOSURE, FULLSCALE_MIN_SPATIAL_EXPOSURE, FULLSCALE_PROBE_EVERY, FULLSCALE_REFINE_EVERY,
    FULLSCALE_SPATIAL_FLOOR, FULLSCALE_SPATIAL_FRACTION, FULLSCALE_SPLIT_MAX_LEVEL)
from rift.gotcha_dataset import C, validate_checkpoint
from rift.range_operator import _geom_gain
from rift.sparse_scene import AdaptivePointSHScene, SHVoxelGridScene, VoxelGridScene

# Baseline implementations are owned separately. Only these native backends
# are implemented here; the root dispatcher discovers opt-in baseline hooks.
METHODS = ('rift', 'rift_grid', 'isotropic', 'mfbp')
LEARNED = ('rift', 'rift_grid', 'isotropic')

# Amplitude law of the learned renderer. 'sum2' is the RIFT-dataset operator's
# two-way spreading (rift/range_operator.py: range_model='sum2', its
# g_const=1/(4*pi)^2 and default eps=1e-9) on the physical monostatic legs,
# r_tx=r_rx=|x-a|; the reference range enters the phase only. 'unit' is the
# earlier GOTCHA kernel (amplitude 1) and the meaning of recipes and
# checkpoints written before the key existed. It is not train.py's 'none',
# which keeps g_const.
RANGE_MODELS = ('sum2', 'unit')
LEGACY_RANGE_MODEL = 'unit'
RIFT_G_CONST = 1.0 / ((4.0 * math.pi) ** 2)
RIFT_EPS = 1.0e-9


def range_amplitude(distance, range_model):
    """Per-point amplitude for physical ranges ``distance``; ``None`` means unit."""
    if range_model == 'unit':
        return None
    if range_model == 'sum2':
        r = distance.clamp_min(RIFT_EPS)
        return _geom_gain(r, r, r + r, 'sum2', RIFT_G_CONST, RIFT_EPS)
    raise ValueError(f'range_model must be one of {RANGE_MODELS}')


# The RIFT-dataset recipe's start and priors, measured on its production B787
# command (train_rift_dataset.py -> train.py full-scale adaptive argv; 2400
# TRAIN views, 1 Tx x 1 Rx) at the end of its backprojection initialization and
# first-view gain warm start (scripts_pvc/rift_dataset_prior_reference.py,
# 2026-09-22; reproduces the production log's alpha 327.5 and gain
# 0.7443-0.0566j). sigma2 is its mean per-sample TRAIN power; m1 = |g| mean_i
# ||w_i|| and m2 = |g|^2 mean_i ||w_i||^2 over its initial points. The weights
# are not dimensionless, since the sum2 amplitude and the data units differ
# between datasets, so GOTCHA keeps mu1 = l1_weight m1 / sigma2 and mu2 =
# sh_degree_weight m2 / sigma2: the same prior strength relative to the data and
# to the initial scene. coefficient_norm (m1 / |g|) is B787's starting per-point
# coefficient size, the gauge that makes Adam's absolute learning-rate steps mean
# the same thing.
_B787 = dict(sigma2=6.707213561508497e-09, gain_magnitude=0.7464650869369507,
             m1=0.0025714804223120227, m2=9.767118844419194e-06)
RIFT_DATASET_REFERENCE = dict(
    source='b787_train2400_1t1r_fullscale_adaptive_v1', backprojection_views=100,
    l1_weight=3e-7, sh_degree_weight=1e-9, regularizer_normalization='fixed_initial', **_B787,
    mu1=3e-7 * _B787['m1'] / _B787['sigma2'], mu2=1e-9 * _B787['m2'] / _B787['sigma2'],
    coefficient_norm=_B787['m1'] / _B787['gain_magnitude'])
BACKPROJECTION = 'rift_dataset_backprojection_v1'
RIFT_DATASET_PRIORS = 'rift_dataset_dimensionless_priors_v1'

# The RIFT-dataset optimizer and schedule: the production B787 adaptive-RIFT
# command (rift/b7873200_adaptive_fullscale.py fullscale_train_argv -> train.py),
# adopted in full by user decision (2026-09-22). train.py builds AdamW over the
# scene coefficients, the positions and the gain with one eps and weight decay
# 0; steps CosineAnnealingWarmRestarts once per epoch after the epoch's last
# update; takes one joint adaptive-capacity-v2 refinement event after every
# refine_every-th epoch, with the probe stride rotated by the epoch; and never
# shuffles its loaders, so every epoch repeats one training order. GOTCHA keeps
# its update unit (one pass-sector) and its TRAIN-power normalization
# (sigma^2 = 1), so its gradients stay far above eps and Adam is effectively
# unthrottled, unlike B787 at its native data scale.
B787_OPTIMIZER = 'rift_dataset_b787_optimizer_v1'
# The earlier GOTCHA optimizer: constant learning rates, a fresh permutation per
# epoch, and refinement every refine_every updates at a single fraction.
LEGACY_OPTIMIZER = 'gotcha_adamw_constant_lr_v1'
OPTIMIZER_KEYS = ('lr', 'pos_lr', 'adam_eps', 'refine_every', 'probe_every', 'refine_fraction', 'max_level')
LEGACY_OPTIMIZER_DEFAULTS = dict(lr=3e-3, pos_lr=1e-4, adam_eps=1e-20, refine_every=100, probe_every=10,
                                 refine_fraction=.05, max_level=3)
# refine_every counts epochs and probe_every pass-sectors, as train.py's
# --adaptive-refine-every / --adaptive-probe-every count epochs and views.
B787_OPTIMIZER_DEFAULTS = dict(lr=3e-3, pos_lr=3e-3, adam_eps=1e-8, refine_every=FULLSCALE_REFINE_EVERY,
                               probe_every=FULLSCALE_PROBE_EVERY, refine_fraction=None,
                               max_level=FULLSCALE_SPLIT_MAX_LEVEL)
B787_SCHEDULE = dict(
    source='b78710k_adaptive_rift_fullscale_v1 (rift/b7873200_adaptive_fullscale.py fullscale_train_argv)',
    weight_decay=0.0, scheduler='cosine_warm_restarts', t0=10, t_mult=2, eta_min=1e-6,
    scheduler_step='once_per_epoch_after_its_last_update', refine_every_unit='epochs',
    probe_rule='(pass_sector_index_in_epoch + epoch) % probe_every == 0',
    view_order='one_seed_permutation_repeated_every_epoch',
    spatial_fraction=FULLSCALE_SPATIAL_FRACTION, angular_fraction=FULLSCALE_ANGULAR_FRACTION,
    min_spatial_exposure=FULLSCALE_MIN_SPATIAL_EXPOSURE, min_angular_exposure=FULLSCALE_MIN_ANGULAR_EXPOSURE,
    spatial_floor=FULLSCALE_SPATIAL_FLOOR, angular_floor=FULLSCALE_ANGULAR_FLOOR,
    cooldown_events=FULLSCALE_COOLDOWN_EVENTS, child_maturity_events=FULLSCALE_CHILD_MATURITY_EVENTS)

# Meaning of adaptive-RIFT recipes and checkpoints written before these keys.
LEGACY_RIFT_KEYS = dict(initialization='random_1e-3', priors='none', optimizer=LEGACY_OPTIMIZER)


def rift_dataset_recipe(args):
    """Adaptive-RIFT recipe keys for the RIFT-dataset start and priors."""
    initialization = getattr(args, 'initialization', 'backprojection')
    priors = getattr(args, 'priors', 'rift_dataset')
    if priors == 'rift_dataset' and initialization != 'backprojection':
        raise ValueError('RIFT-dataset priors are defined at the backprojection start; use --initialization backprojection')
    keys = dict(initialization=BACKPROJECTION if initialization == 'backprojection' else LEGACY_RIFT_KEYS['initialization'],
                priors=RIFT_DATASET_PRIORS if priors == 'rift_dataset' else LEGACY_RIFT_KEYS['priors'])
    if initialization == 'backprojection':
        keys.update(bp_views=int(getattr(args, 'bp_views', 100)), rift_dataset_reference=dict(RIFT_DATASET_REFERENCE))
    return keys


def with_legacy_keys(recipe, method):
    """Fill keys absent from recipes written before they existed with their old meaning."""
    recipe = dict(recipe)
    if method in LEARNED:
        recipe.setdefault('range_model', LEGACY_RANGE_MODEL)
    if method == 'rift':
        for key, value in LEGACY_RIFT_KEYS.items():
            recipe.setdefault(key, value)
    return recipe


def optimizer_recipe(args, method):
    """Optimizer and schedule keys: B787's for adaptive RIFT (default), the earlier GOTCHA values otherwise.

    An unset flag takes the selected recipe's value; an explicit flag is recorded
    as given. rift_grid/isotropic and ``--optimizer legacy`` keep the earlier
    recipe keys and values exactly.
    """
    b787 = method == 'rift' and getattr(args, 'optimizer', 'b787') == 'b787'
    defaults = B787_OPTIMIZER_DEFAULTS if b787 else LEGACY_OPTIMIZER_DEFAULTS
    keys = {}
    for key in OPTIMIZER_KEYS:
        value = getattr(args, key, None)
        keys[key] = defaults[key] if value is None else value
    if b787:
        if getattr(args, 'refine_fraction', None) is not None:
            raise ValueError('--refine-fraction is the earlier single-fraction rule; the B787 schedule uses its '
                             'spatial/angular fractions (select --optimizer legacy to set it)')
        keys.update(optimizer=B787_OPTIMIZER, optimizer_schedule=dict(B787_SCHEDULE))
    elif method == 'rift':
        keys['optimizer'] = LEGACY_OPTIMIZER
    return keys


def b787_schedule(method, recipe):
    """The B787 schedule of an adaptive-RIFT recipe, or ``None`` for the earlier GOTCHA optimizer."""
    if method == 'rift' and recipe.get('optimizer', LEGACY_OPTIMIZER) == B787_OPTIMIZER:
        return recipe['optimizer_schedule']
    return None


def build_scheduler(optimizer, schedule):
    """train.py's CosineAnnealingWarmRestarts over every parameter group, or ``None``."""
    if schedule is None:
        return None
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    return CosineAnnealingWarmRestarts(optimizer, T_0=schedule['t0'], T_mult=schedule['t_mult'],
                                       eta_min=schedule['eta_min'])


def fixed_view_order(seed, count):
    """The epoch-1 permutation, which the B787 schedule repeats (train.py's loaders are never shuffled)."""
    return np.random.Generator(np.random.PCG64(seed)).permutation(count).tolist()


def probe_due(recipe, schedule, *, updates, cursor, epoch):
    """Probe the next SH band on this update: every probe_every-th update, or train.py's epoch-rotated stride."""
    if schedule is None:
        return updates % recipe['probe_every'] == 0
    return (cursor + epoch) % recipe['probe_every'] == 0


def b787_epoch_end(heads, recipe, schedule, optimizer, scheduler, epoch):
    """train.py's end of epoch ``epoch`` (0-based): scheduler step, then a refinement event when due.

    Returns the history record: the learning rates the epoch used and, after an
    event, each channel's split/grown/active counts.
    """
    record = dict(learning_rates=[float(group['lr']) for group in optimizer.param_groups])
    scheduler.step()
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
    return record


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


@torch.no_grad()
def native_adjoint(points, values, antenna, frequencies, reference_range, *, point_chunk=4096, range_model='unit'):
    """Adjoint of ``native_forward`` for one pulse: b_i = sum_f conj(A_i e_if) values_f."""
    if point_chunk <= 0:
        raise ValueError('point_chunk must be positive')
    if range_model not in RANGE_MODELS:
        raise ValueError(f'range_model must be one of {RANGE_MODELS}')
    antenna = torch.as_tensor(antenna, dtype=torch.float64, device=points.device)
    frequencies = torch.as_tensor(frequencies, dtype=torch.float64, device=points.device)
    values = torch.as_tensor(values, dtype=torch.complex128, device=points.device)
    result = torch.empty(len(points), dtype=torch.complex128, device=points.device)
    for start in range(0, len(points), point_chunk):
        physical = torch.linalg.vector_norm(points[start:start+point_chunk].double() - antenna, dim=-1)
        phase = (4 * math.pi / C) * (physical - float(reference_range))[:, None] * frequencies[None, :]
        column = torch.exp(1j * phase) @ values
        amplitude = range_amplitude(physical, range_model)
        result[start:start+point_chunk] = column if amplitude is None else column * amplitude
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


def _grid(granularity, extent, device):
    axis = (torch.arange(granularity, dtype=torch.float64, device=device) + .5) * (2*extent/granularity) - extent
    return torch.cartesian_prod(axis, axis, axis)


class ChannelField(nn.Module):
    """One polarization head fitted jointly to all selected passes."""
    def __init__(self, method, region, recipe, device):
        super().__init__()
        self.method, self.region, self.recipe = method, region, recipe
        g, e = recipe['granularity'], region.half_extent_m
        self.register_buffer('points', _grid(g, e, device))
        self.register_buffer('scale_initialized', torch.tensor(False, device=device))
        if method == 'rift':
            # The backprojection start begins from the zero scene, as train.py's --init-scale 0.
            self.field = AdaptivePointSHScene.from_regular_grid(
                g, e, device, max_degree=recipe['sh_degree'], init_degree=0,
                init_scale=0.0 if recipe.get('initialization') == BACKPROJECTION else 1e-3,
                capacity=recipe['max_points'],
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
        p, y = RangeReadout.project(predicted, readout), RangeReadout.project(target, readout)
        self.gain.maybe_init_scale(p, y)
        self.scale_initialized.fill_(True)


def recipe_from_args(args, method):
    keys = ('epochs', 'seed', 'granularity', 'max_points', 'sh_degree', 'point_chunk', 'checkpoint_every')
    return dict(schema='gotcha_native_method_recipe_v1', method=method,
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
                **(rift_dataset_recipe(args) if method == 'rift' else {}))


def _target(observation, device):
    return torch.as_tensor(observation.response, dtype=torch.complex128, device=device)


@torch.no_grad()
def rift_dataset_initialization(head, dataset, readout, views, polarization):
    """The RIFT-dataset start on GOTCHA: backprojection, gain warm start, coefficient gauge.

    As train.py's backprojection_init: b = sum over the first training views of
    A^H s and w = alpha b with alpha = <A b, s> / ||A b||^2, written to the
    degree-0 coefficient of the active points. Here A is the native renderer
    with the recipe's amplitude law, the views are the first pass-sectors of the
    epoch-1 training order (B787's loader order is its seed-42 permutation), and
    s is the ROI-projected measurement, GOTCHA's loss domain. So b is again the
    descent direction of the objective at the empty scene. The gain is then
    warm-started on the first view's first pulse, as the training loop would do.
    Finally a prediction-preserving gauge (w -> c w, g -> g / c) gives the
    coefficients B787's starting size. The measured scale m1/m2 then fixes the
    prior weights (``RIFT_DATASET_REFERENCE``).
    """
    scene, recipe = head.field, head.recipe
    device, mask = scene.w_re.device, scene.active_mask
    points = scene.grid_positions.reshape(-1, 3)[mask].double()
    options = dict(point_chunk=recipe['point_chunk'], range_model=recipe.get('range_model', LEGACY_RANGE_MODEL))
    b = torch.zeros(len(points), dtype=torch.complex128, device=device)
    used = []
    for p, sector in views:
        for observation in dataset.observations(p, sector, polarization):
            r = readout.for_observation(observation)
            projected = readout.lift(readout.project(_target(observation, device), r), r)
            b += native_adjoint(points, projected, r['antenna'], r['frequencies'], observation.reference_range_m, **options)
            used.append(observation)
    numerator = torch.zeros((), dtype=torch.complex128, device=device)
    denominator = torch.zeros((), dtype=torch.float64, device=device)
    for observation in used:
        r = readout.for_observation(observation)
        rendered = readout.project(native_forward(points, b, r['antenna'], r['frequencies'],
                                                  observation.reference_range_m, **options), r)
        numerator += (rendered.conj() * readout.project(_target(observation, device), r)).sum()
        denominator += rendered.abs().square().sum()
    if not torch.isfinite(numerator) or not bool(denominator > 0) or not torch.isfinite(denominator):
        raise ValueError('Backprojection initialization is degenerate')
    alpha = numerator / denominator
    y00 = 0.5 / math.sqrt(math.pi)
    scene.w_re[mask, 0] = ((alpha * b).real / y00).to(scene.w_re.dtype)
    scene.w_im[mask, 0] = ((alpha * b).imag / y00).to(scene.w_im.dtype)
    first = next(iter(dataset.observations(*views[0], polarization)))
    head.initialize_scale(first, readout.for_observation(first))
    warm_gain = head.gain.gain_value()
    squared = scene.w_re[mask].double().square().sum(-1) + scene.w_im[mask].double().square().sum(-1)
    gauge = RIFT_DATASET_REFERENCE['coefficient_norm'] / float(squared.sqrt().mean())
    scene.w_re.mul_(gauge)
    scene.w_im.mul_(gauge)
    head.gain.log_mag.sub_(math.log(gauge))
    squared = scene.w_re[mask].double().square().sum(-1) + scene.w_im[mask].double().square().sum(-1)
    g = float(torch.exp(head.gain.log_mag))
    count = int(mask.sum())
    m1, m2 = g * float(squared.sqrt().sum()) / count, g * g * float(squared.sum()) / count
    reference = recipe['rift_dataset_reference']
    return dict(schema=BACKPROJECTION, views=[[int(p), int(sector)] for p, sector in views], pulses=len(used),
                alpha=[float(alpha.real), float(alpha.imag)], warm_start_gain=[warm_gain.real, warm_gain.imag],
                coefficient_gauge=gauge, initial_points=count, m1=m1, m2=m2,
                l1_weight=reference['mu1'] / m1, sh_degree_weight=reference['mu2'] / m2)


def rift_dataset_prior(head, initialization):
    """train.py's group-L1 and SH-degree prior at the transferred weights, fixed-initial normalized."""
    from train import regularization_loss
    return regularization_loss(head.field, initialization['l1_weight'], initialization['sh_degree_weight'],
                               gain=head.gain, return_terms=True, normalization='fixed_initial',
                               reference_active_count=initialization['initial_points'])


def prior_backward(head, initialization, channels):
    """Add one update's prior for this channel after its data backward (the refinement statistics exclude it)."""
    prior, _ = rift_dataset_prior(head, initialization)
    if prior is None:
        return
    if not torch.isfinite(prior):
        raise ValueError('Nonfinite prior')
    (prior / channels).backward()


@torch.no_grad()
def prior_terms(heads, initialization):
    """Current prior terms per channel (deterministic, so recorded in history across resumes)."""
    result = {}
    for pol, head in heads.items():
        _, terms = rift_dataset_prior(head, initialization[pol])
        result[pol] = {name: float(value) for name, value in terms.items()}
    return result


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


def _rng_state(rng):
    return dict(numpy=rng.bit_generator.state, torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _restore_rng(payload, rng):
    rng.bit_generator.state = payload['numpy']
    torch.set_rng_state(payload['torch'].cpu())
    if torch.cuda.is_available():
        if payload['cuda'] is None:
            raise ValueError('CUDA resume requires saved CUDA RNG state')
        torch.cuda.set_rng_state_all([value.cpu() for value in payload['cuda']])


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


def train(dataset, method, recipe, output, *, device='cuda', resume=None):
    """Run only when explicitly invoked inside an experiment allocation."""
    from rift.gotcha_frequency_selection import bind_recipe
    recipe = bind_recipe(dataset, recipe)
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
    positions = [h.field.delta_raw for h in heads.values() if method == 'rift']
    position_ids = {id(p) for p in positions}
    params = [p for p in heads.parameters() if id(p) not in position_ids]
    groups = [dict(params=params, lr=recipe['lr'])]
    if positions:
        groups.append(dict(params=positions, lr=recipe['pos_lr']))
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
        stats, initialization = training_statistics(dataset, readout), {}
        if method == 'rift' and recipe.get('initialization') == BACKPROJECTION:
            # The epoch-1 order the loop would draw here; its first views seed the start.
            order = rng.permutation(len(train_views)).tolist()
            if schedule is not None and order != fixed_view_order(recipe['seed'], len(train_views)):
                raise AssertionError('The epoch-1 order must be the fixed B787 training order')
            views = [train_views[i] for i in order[:recipe['bp_views']]]
            initialization = {pol: rift_dataset_initialization(head, dataset, readout, views, pol)
                              for pol, head in heads.items()}
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
    def save(name):
        atomic_save(output/name, dict(schema='rift_gotcha_checkpoint_v1', dataset_contract=dataset.contract,
                    dataset_identity=dataset.identity, recipe=recipe, model_state_dict=heads.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(), training_statistics=stats,
                    initialization=initialization,
                    epoch=epoch, cursor=cursor, history=history, best_val=best, updates=updates,
                    order=order, rng_state=_rng_state(rng),
                    **({'scheduler_state_dict': scheduler.state_dict()} if scheduler is not None else {})))
    try:
        while epoch < recipe['epochs']:
            if order is None:
                order = (fixed_view_order(recipe['seed'], len(train_views)) if schedule is not None
                         else rng.permutation(len(train_views)).tolist())
            heads.train()
            while cursor < len(order):
                p, sector = train_views[order[cursor]]
                optimizer.zero_grad(set_to_none=True)
                probing = probe_due(recipe, schedule, updates=updates, cursor=cursor, epoch=epoch)
                # One optimizer step per pass/sector. Ragged channels contribute
                # equally; native pulse counts within a channel are preserved.
                for pol in dataset.polarizations:
                    head = heads[pol]
                    count = len(dataset.shards[p,pol].sector_rows[sector])
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
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in heads.parameters()):
                    raise ValueError('Nonfinite parameter gradient')
                optimizer.step()
                updates += 1
                cursor += 1
                if method == 'rift' and schedule is None and updates % recipe['refine_every'] == 0:
                    for head in heads.values():
                        scene = head.field
                        snap = scene.refinement_snapshot(recipe['max_level'], 2, 2, cooldown_events=1, child_maturity_events=1)
                        scene.apply_refinement_snapshot(snap, recipe['refine_fraction'], recipe['refine_fraction'],
                                                        recipe['max_level'], optimizer=optimizer, max_active=recipe['max_points'])
                if updates % recipe['checkpoint_every'] == 0 or stop['requested']:
                    save('checkpoint_latest.pt')
                if stop['requested']:
                    return dict(status='interrupted', completed_epochs=epoch, updates=updates)
            # As train.py: scheduler step and any refinement event precede validation.
            schedule_record = (b787_epoch_end(heads, recipe, schedule, optimizer, scheduler, epoch)
                               if schedule is not None else None)
            metrics = evaluate(heads, dataset, readout, stats, method)
            epoch += 1
            cursor, order = 0, None
            history.append(dict(epoch=epoch, updates=updates, validation=metrics,
                                **({'priors': prior_terms(heads, initialization)} if priors else {}),
                                **({'optimizer': schedule_record} if schedule is not None else {})))
            if best is None or metrics['pooled_rel_mse'] < best:
                best = metrics['pooled_rel_mse']
                save('checkpoint_best.pt')
            save('checkpoint_latest.pt')
            atomic_json(output/'history.json', history)
            print(json.dumps(dict(method=method, epoch=epoch, validation_rel_mse=metrics['pooled_rel_mse'])), flush=True)
        save('checkpoint_final.pt')
        return dict(status='complete', completed_epochs=epoch, updates=updates, best_validation_rel_mse=best, test_accessed=False)
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def backproject(dataset, recipe, output, *, device='cuda'):
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
