"""RIFT-dataset NUFFT forward evaluation and full-native loss for PVC GOTCHA adaptive RIFT (opt-in control).

The default GOTCHA recipe evaluates the native kernel by exact direct summation
at the selected native frequencies and fits each pulse's ROI range-subspace
projection. This control reproduces the two corresponding choices of the
RIFT-dataset recipe (the production B787 command: ``train.py --forward-operator
range --loss complex --compute-dtype float64``):

* ``forward_evaluation = rift_dataset_nufft_v1``: the same kernel evaluated with
  the RIFT-dataset range operator's type-1 Gaussian-gridding NUFFT
  (``rift/range_operator.py`` at its B787 defaults: oversample 2, kernel width 20,
  float64). Its primitives are imported unchanged. As in ``range_forward_operator``
  the complete uniform grid is rendered and the selected bins are gathered
  afterwards; the grid is each pass's complete native source grid, reconstructed
  as the uniform linspace through its endpoints and checked by the operator's own
  1e-2-bin gate. The path is GOTCHA's two-way reference-relative path
  ``2(|x - a| - r0)`` (phase sign -1), so the kernel is exactly
  ``exp(-i 4 pi f (|x - a| - r0) / c)``; the declared amplitude law stays on the
  physical range ``|x - a|``. Evaluating along the absolute ~10 km path instead
  would turn the float32 storage of the native frequencies (<= 940 Hz from the
  linspace on Camry) into ~0.4 rad of phase error; on the reference-relative path
  (|dr| <= ~30 m over the Camry cube) it is ~1e-3 rad.
* ``loss_domain = full_native_complex``: the complex MSE over every selected
  native bin, as train.py's complex loss, without the ROI range-subspace
  projection, normalized by the TRAIN mean full-native power (the same unit
  convention as the default recipe, applied to its own loss domain). The
  backprojection start then uses the unprojected measurement (the descent
  direction of this objective at the empty scene), the gain warm start fits the
  full response, and checkpoints are selected by validation full-native RelMSE.
  The ROI-projected metric is still recorded every epoch for comparison.

Everything else (acquisition, optimizer, priors, refinement, update unit) is the
default recipe's. The two keys are independent; the control sets both. Recipes
without them mean direct evaluation and the projected loss, so default recipes
and checkpoints are unchanged. PVC-only; the CUDA trainer is unchanged.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as recompute

from rift.config import cc
from rift.gotcha_dataset import C
from rift.gotcha_training import (LEGACY_RANGE_MODEL, RANGE_MODELS, RIFT_DATASET_REFERENCE, BACKPROJECTION,
                                  native_adjoint, native_forward, range_amplitude)
from rift.range_operator import (_deapodization, _default_tau_bins, _gather_from_grid, _mode_bins,
                                 _next_power_of_two, _spread_to_grid, _validate_and_reconstruct_grid)

DIRECT = 'direct_native_v1'
NUFFT = 'rift_dataset_nufft_v1'
FORWARD_EVALUATIONS = {'direct': DIRECT, 'nufft': NUFFT}
PROJECTED = 'roi_projected_native_complex'
FULL = 'full_native_complex'
LOSS_DOMAINS = {'roi_projected': PROJECTED, 'full_native': FULL}
# range_forward_operator's defaults, which the B787 production command uses.
OVERSAMPLE, KERNEL_WIDTH = 2, 20


def forward_evaluation(recipe):
    return recipe.get('forward_evaluation', DIRECT)


def loss_domain(recipe):
    return recipe.get('loss_domain', PROJECTED)


def control_keys(recipe):
    """True for a recipe with a non-default forward evaluation or loss domain."""
    return forward_evaluation(recipe) != DIRECT or loss_domain(recipe) != PROJECTED


def control_recipe(args, method):
    """Recipe keys for a non-default forward evaluation or loss domain; empty for the default recipe."""
    evaluation = FORWARD_EVALUATIONS[getattr(args, 'forward_evaluation', 'direct')]
    domain = LOSS_DOMAINS[getattr(args, 'loss_domain', 'roi_projected')]
    if evaluation == DIRECT and domain == PROJECTED:
        return {}
    if method != 'rift':
        raise ValueError('--forward-evaluation nufft and --loss-domain full_native are defined for adaptive RIFT only')
    keys = {}
    if evaluation == NUFFT:
        keys.update(forward_evaluation=NUFFT, nufft=dict(
            operator='rift.range_operator type-1 Gaussian-gridding NUFFT (train.py --forward-operator range)',
            oversample=OVERSAMPLE, kernel_width=KERNEL_WIDTH, tau_bins=_default_tau_bins(KERNEL_WIDTH),
            compute_dtype='float64',
            frequency_grid=('complete native source grid of each pass, reconstructed as the uniform linspace '
                            'through its endpoints (range_operator gate 1e-2 bin); selected bins gathered'),
            path='two-way reference-relative 2(|x-a|-r0), phase sign -1',
            amplitude='recipe range_model on the physical range |x-a|'))
    if domain == FULL:
        keys.update(loss_domain=FULL, loss_normalization='train_only_mean_full_native_power',
                    checkpoint_selection='validation_full_native_complex_rel_mse')
    return keys


def output_suffix(recipe):
    """Method-directory suffix that keeps a control run apart from the default recipe's."""
    return ('_nufft' if forward_evaluation(recipe) == NUFFT else '') + (
        '_full_native' if loss_domain(recipe) == FULL else '')


class NativeGrid:
    """One pass's complete native frequency grid, prepared as range_forward_operator prepares its grid."""

    def __init__(self, source_hz, selected, *, device, oversample=OVERSAMPLE, kernel_width=KERNEL_WIDTH):
        device = torch.device(device)
        source = torch.as_tensor(np.array(source_hz, dtype=np.float64), device=device)
        ideal = _validate_and_reconstruct_grid(source, (2.0 * math.pi / cc) * source)
        self.nf_full = int(ideal.shape[0])
        self.df = float(ideal[1] - ideal[0])
        self.f_ref = float(ideal[self.nf_full // 2])
        self.m_grid = _next_power_of_two(int(math.ceil(float(oversample) * self.nf_full)))
        self.kernel_width = int(kernel_width)
        self.tau_bins = _default_tau_bins(self.kernel_width)
        self.deapod = _deapodization(self.nf_full, self.m_grid, self.kernel_width, self.tau_bins, device,
                                     torch.float64)
        self.mode_bins = _mode_bins(self.nf_full, self.m_grid, device)
        self.selected = torch.as_tensor(np.array(selected, dtype=np.int64), device=device)
        if self.selected.ndim != 1 or not len(self.selected) or int(self.selected.min()) < 0 \
                or int(self.selected.max()) >= self.nf_full:
            raise ValueError('Selected bins must index the native source grid')
        self.selected_hz = source[self.selected]
        self.max_linspace_deviation_hz = float((source - ideal).abs().max())


def attach_grids(heads, dataset, recipe, device):
    """Give each channel head the NUFFT grid of every selected pass (no-op for direct evaluation)."""
    if forward_evaluation(recipe) != NUFFT:
        return
    for pol, head in heads.items():
        head.nufft_grids = {p: NativeGrid(shard.frequencies_hz, shard.frequency_indices_for_role('train'),
                                          device=device)
                            for (p, q), shard in dataset.shards.items() if q == pol}


def grid_for(head, observations, frequencies):
    """The NUFFT grid of these pulses; their selected frequencies must be exactly its selected bins."""
    passes = {o.pass_id for o in observations}
    if len(passes) != 1:
        raise ValueError('One NUFFT render covers pulses of one pass')
    grid = head.nufft_grids[passes.pop()]
    if not torch.equal(torch.as_tensor(frequencies, dtype=torch.float64, device=grid.selected_hz.device),
                       grid.selected_hz):
        raise ValueError('Pulse frequencies are not the selected bins of their pass grid')
    return grid


def _render_chunk(dist, weights, refs, f_ref, df, m_grid, kernel_width, tau_bins, range_model):
    """One point chunk's gridded spectrum ``[pulses, m_grid]``; purely functional (checkpoint-safe).

    ``range_operator._render_forward_point_chunk`` with the pulses as its pairs,
    ``r_sum = 2 (|x - a| - r0)`` and phase sign -1, and the declared amplitude on
    the physical range instead of on the path legs.
    """
    path = 2.0 * (dist - refs[:, None])
    phase_ref = (-2.0 * math.pi * f_ref / C) * path
    amp = weights.to(torch.complex128) * torch.polar(torch.ones_like(phase_ref), phase_ref)
    amplitude = range_amplitude(dist, range_model)
    if amplitude is not None:
        amp = amp * amplitude
    x = (-2.0 * math.pi * df / C) * path
    u = torch.remainder(x * (float(m_grid) / (2.0 * math.pi)), float(m_grid))
    grid = torch.zeros((dist.shape[0], m_grid), dtype=torch.complex128, device=dist.device)
    _spread_to_grid(grid, amp.T, u.T, kernel_width, tau_bins)
    return grid


def nufft_forward(dist, weights, refs, grid, *, point_chunk, range_model='unit'):
    """Predictions ``[pulses, selected bins]`` from physical ranges ``dist`` and weights ``[pulses, points]``."""
    if point_chunk <= 0:
        raise ValueError('point_chunk must be positive')
    if range_model not in RANGE_MODELS:
        raise ValueError(f'range_model must be one of {RANGE_MODELS}')
    if dist.shape != weights.shape or dist.ndim != 2 or refs.shape != dist.shape[:1]:
        raise ValueError('dist and weights must be [pulses, points] with one reference range per pulse')
    refs = refs.to(torch.float64)
    total = torch.zeros((dist.shape[0], grid.m_grid), dtype=torch.complex128, device=dist.device)
    for start in range(0, dist.shape[1], point_chunk):
        d, w = dist[:, start:start + point_chunk], weights[:, start:start + point_chunk]
        args = (d, w, refs, grid.f_ref, grid.df, grid.m_grid, grid.kernel_width, grid.tau_bins, range_model)
        if torch.is_grad_enabled() and (d.requires_grad or w.requires_grad):
            total = total + recompute(_render_chunk, *args, use_reentrant=False)
        else:
            total = total + _render_chunk(*args)
    spectrum = grid.m_grid * torch.fft.ifft(total, dim=-1)
    return (spectrum[:, grid.mode_bins] / grid.deapod.view(1, -1))[:, grid.selected]


@torch.no_grad()
def nufft_adjoint(points, values, antennas, refs, grid, *, point_chunk, range_model='unit'):
    """Adjoint of ``nufft_forward`` in the weights, summed over pulses: ``[points]`` from ``values [pulses, bins]``.

    ``range_adjoint_operator``'s construction: deapodized selected bins, forward
    FFT, Gaussian gather, conjugate carrier and amplitude.
    """
    if point_chunk <= 0:
        raise ValueError('point_chunk must be positive')
    if range_model not in RANGE_MODELS:
        raise ValueError(f'range_model must be one of {RANGE_MODELS}')
    values = torch.as_tensor(values, dtype=torch.complex128, device=points.device)
    antennas = torch.as_tensor(antennas, dtype=torch.float64, device=points.device)
    refs = torch.as_tensor(refs, dtype=torch.float64, device=points.device)
    spectrum = torch.zeros((values.shape[0], grid.m_grid), dtype=torch.complex128, device=points.device)
    spectrum[:, grid.mode_bins[grid.selected]] = values / grid.deapod[grid.selected].conj().view(1, -1)
    adjoint_grid = torch.fft.fft(spectrum, dim=-1)
    result = torch.empty(len(points), dtype=torch.complex128, device=points.device)
    for start in range(0, len(points), point_chunk):
        dist = torch.linalg.vector_norm(points[start:start + point_chunk].double()[None] - antennas[:, None], dim=-1)
        path = 2.0 * (dist - refs[:, None])
        phase_ref = (-2.0 * math.pi * grid.f_ref / C) * path
        amp = torch.polar(torch.ones_like(phase_ref), phase_ref)
        amplitude = range_amplitude(dist, range_model)
        if amplitude is not None:
            amp = amp * amplitude
        x = (-2.0 * math.pi * grid.df / C) * path
        u = torch.remainder(x * (float(grid.m_grid) / (2.0 * math.pi)), float(grid.m_grid))
        gathered = _gather_from_grid(adjoint_grid, u.T.contiguous(), grid.kernel_width, grid.tau_bins)
        result[start:start + point_chunk] = (amp.T.conj() * gathered).sum(dim=1)
    return result


def sector_render(points, weights, antennas, refs, grid, *, point_chunk, range_model, pulse_grad_d=None):
    """``batched_native_forward``'s contract through the NUFFT.

    ``pulse_grad_d`` receives dL/d|x_i - a_p| per pulse from the same backward
    pass (each range feeds only its own pulse), as the direct kernel's custom
    backward provides it. Like range_operator, the derivative of the Gaussian
    support jump at a bin boundary is omitted.
    """
    dist = torch.linalg.vector_norm(points.double()[None] - antennas[:, None], dim=-1)
    if pulse_grad_d is not None and dist.requires_grad:
        def keep(gradient):
            pulse_grad_d.copy_(gradient)
        dist.register_hook(keep)
    return nufft_forward(dist, weights, refs, grid, point_chunk=point_chunk, range_model=range_model)


def pulse_render(points, weights, antenna, frequencies, observation, head, *, point_chunk, range_model):
    """One pulse through the NUFFT (the per-pulse loop's ``native_forward``)."""
    grid = grid_for(head, [observation], frequencies)
    antennas = torch.as_tensor(antenna, dtype=torch.float64, device=points.device)[None]
    refs = torch.tensor([observation.reference_range_m], dtype=torch.float64, device=points.device)
    return sector_render(points, weights[None], antennas, refs, grid, point_chunk=point_chunk,
                         range_model=range_model)[0]


def full_losses(pred, targets, mean_power):
    """Per-pulse full-native complex MSE over the selected bins (train.py's complex loss), normalized."""
    return (pred - targets).abs().square().mean(dim=1) / mean_power


def domain_values(value, readout, recipe):
    """``value`` in the recipe's loss domain: the ROI projection, or the native response itself."""
    if loss_domain(recipe) == FULL:
        return value
    return readout['q'].conj().T @ (readout['phase'] * value)


@torch.no_grad()
def full_native_statistics(dataset, readout):
    """TRAIN-only normalization of the full-native loss in one bounded-memory pass.

    ``energy``/``count``/``mean_power`` are the loss domain's (full native), as the
    default recipe's are its projected domain's, so every consumer normalizes by
    ``mean_power``; the projected values are kept under ``projected_*``.
    """
    sums = {pol: dict(loss_domain=FULL, energy=0., count=0, range_peak=0., projected_energy=0., projected_count=0)
            for pol in dataset.polarizations}
    for p, sector in dataset.viewpoints('train'):
        for pol in dataset.polarizations:
            s = sums[pol]
            for observation in dataset.observations(p, sector, pol):
                r = readout.for_observation(observation)
                y = torch.as_tensor(observation.response, dtype=torch.complex128, device=readout.device)
                projected = readout.project(y, r)
                s['energy'] += y.abs().square().sum().item()
                s['count'] += y.numel()
                s['projected_energy'] += projected.abs().square().sum().item()
                s['projected_count'] += projected.numel()
                s['range_peak'] = max(s['range_peak'], (r['matched'] @ y).abs().square().max().item())
    for s in sums.values():
        if not s['count'] or not math.isfinite(s['energy']) or s['energy'] <= 0 or s['range_peak'] <= 0 \
                or not s['projected_count'] or not s['projected_energy'] > 0:
            raise ValueError('TRAIN normalization is empty, nonfinite or zero')
        s['mean_power'] = s['energy'] / s['count']
        s['projected_mean_power'] = s['projected_energy'] / s['projected_count']
    return sums


def with_selection(metrics, recipe):
    """Name the checkpoint-selection metric of a full-native recipe (the default selects by ``pooled_rel_mse``)."""
    if loss_domain(recipe) != FULL:
        return metrics
    return dict(metrics, selection_domain=FULL, selection_rel_mse=metrics['full_native_complex_rel_mse'])


def selection_value(metrics):
    return metrics.get('selection_rel_mse', metrics['pooled_rel_mse'])


def _sector_render(head, points, weights, observations, readouts, range_model, point_chunk):
    """Degree-0 weights ``[points]`` rendered for every pulse of one sector, in the head's forward evaluation."""
    if forward_evaluation(head.recipe) == NUFFT:
        antennas = torch.stack([r['antenna'] for r in readouts])
        refs = torch.tensor([o.reference_range_m for o in observations], dtype=torch.float64, device=points.device)
        grid = grid_for(head, observations, readouts[0]['frequencies'])
        return sector_render(points, weights[None].expand(len(observations), -1), antennas, refs, grid,
                             point_chunk=point_chunk, range_model=range_model)
    return torch.stack([native_forward(points, weights, r['antenna'], r['frequencies'], o.reference_range_m,
                                       point_chunk=point_chunk, range_model=range_model)
                        for o, r in zip(observations, readouts)])


def _sector_adjoint(head, points, values, observations, readouts, range_model, point_chunk):
    if forward_evaluation(head.recipe) == NUFFT:
        antennas = torch.stack([r['antenna'] for r in readouts])
        refs = torch.tensor([o.reference_range_m for o in observations], dtype=torch.float64, device=points.device)
        grid = grid_for(head, observations, readouts[0]['frequencies'])
        return nufft_adjoint(points, values, antennas, refs, grid, point_chunk=point_chunk, range_model=range_model)
    return sum(native_adjoint(points, v, r['antenna'], r['frequencies'], o.reference_range_m,
                              point_chunk=point_chunk, range_model=range_model)
               for v, o, r in zip(values, observations, readouts))


@torch.no_grad()
def control_initialization(head, dataset, readout, views, polarization):
    """``rift_dataset_initialization`` in the control's forward evaluation and loss domain.

    As the default start: b = sum over the first training views of A^H s and
    w = alpha b with alpha = <A b, s> / ||A b||^2 in the loss domain, then the
    gain warm start and the B787 coefficient gauge. A is the head's forward
    evaluation (the NUFFT and its exact adjoint for the control); s is the
    measurement in the loss domain lifted back to native bins, so for the full
    domain the unprojected measurement, the descent direction of that objective
    at the empty scene.
    """
    scene, recipe = head.field, head.recipe
    device, mask = scene.w_re.device, scene.active_mask
    points = scene.grid_positions.reshape(-1, 3)[mask].double()
    range_model, chunk = recipe.get('range_model', LEGACY_RANGE_MODEL), recipe['point_chunk']
    b = torch.zeros(len(points), dtype=torch.complex128, device=device)
    sectors, pulses = [], 0
    for p, sector in views:
        observations = list(dataset.observations(p, sector, polarization))
        readouts = [readout.for_observation(o) for o in observations]
        targets = torch.stack([torch.as_tensor(o.response, dtype=torch.complex128, device=device)
                               for o in observations])
        if loss_domain(recipe) == FULL:
            lifted = targets
        else:
            lifted = torch.stack([readout.lift(readout.project(t, r), r) for t, r in zip(targets, readouts)])
        b += _sector_adjoint(head, points, lifted, observations, readouts, range_model, chunk)
        sectors.append((observations, readouts, targets))
        pulses += len(observations)
    numerator = torch.zeros((), dtype=torch.complex128, device=device)
    denominator = torch.zeros((), dtype=torch.float64, device=device)
    for observations, readouts, targets in sectors:
        rendered = _sector_render(head, points, b, observations, readouts, range_model, chunk)
        for value, target, r in zip(rendered, targets, readouts):
            value, target = domain_values(value, r, recipe), domain_values(target, r, recipe)
            numerator += (value.conj() * target).sum()
            denominator += value.abs().square().sum()
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
    return dict(schema=BACKPROJECTION, views=[[int(p), int(sector)] for p, sector in views], pulses=pulses,
                forward_evaluation=forward_evaluation(recipe), loss_domain=loss_domain(recipe),
                alpha=[float(alpha.real), float(alpha.imag)], warm_start_gain=[warm_gain.real, warm_gain.imag],
                coefficient_gauge=gauge, initial_points=count, m1=m1, m2=m2,
                l1_weight=reference['mu1'] / m1, sh_degree_weight=reference['mu2'] / m2)
