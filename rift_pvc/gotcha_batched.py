"""Batched pass-sector update for the PVC GOTCHA built-in backends.

All selected pulses of one pass-sector/channel are rendered in one
``[pulses, chunk, frequencies]`` kernel per point chunk and back-propagated
once. The per-pulse quantities the adaptive recipe consumes (spatial and
next-band refinement statistics) are recovered from that single backward
pass instead of one extra backward pass per pulse and a separate probe pass:
the per-pulse position gradient is the per-pulse distance gradient (exposed
by the custom backward before its sum over pulses) times the unit look
vector, and the probe's next-band gradient is the per-pulse weight gradient
times the next band's spherical-harmonic basis at that pulse's direction,
because the probe leaves the prediction unchanged. The recipe semantics are
therefore those of ``rift/gotcha_training.py``'s per-pulse loop up to
floating-point summation order (measured ≤ 2e-7 relative on CPU; GOTCHA.md
section 7). The CUDA trainer is unchanged; this module is PVC-only.
"""
from __future__ import annotations

import math

import torch

from rift.gotcha_dataset import C
from rift.gotcha_training import LEGACY_RANGE_MODEL, RANGE_MODELS, range_amplitude
from rift.spherical_harmonics import real_sh_basis
from rift_pvc import gotcha_nufft as nufft

LOOP = 'per_pulse_loop'
BATCHED = 'batched_pulses_one_backward_v1'
SECTOR_EXECUTIONS = {'loop': LOOP, 'batched': BATCHED}


def sector_execution_value(choice):
    if choice in SECTOR_EXECUTIONS.values():
        return choice
    if choice not in SECTOR_EXECUTIONS:
        raise ValueError(f'sector execution must be one of {sorted(SECTOR_EXECUTIONS)}')
    return SECTOR_EXECUTIONS[choice]


def _amplitude_and_slope(dist, range_model):
    """Declared amplitude A(|x - a|) and dA/d|x - a| from the shared definition; None for unit."""
    with torch.enable_grad():
        r = dist.detach().requires_grad_(True)
        amplitude = range_amplitude(r, range_model)
        if amplitude is None:
            return None, None
        slope, = torch.autograd.grad(amplitude.sum(), r)
    return amplitude.detach(), slope


class _PulseBlock(torch.autograd.Function):
    """out[p, f] = sum_i w[p, i] A(|x_i - a_p|) exp(-j k_f (|x_i - a_p| - r_p)), recomputed in backward.

    A is the recipe's declared amplitude law on the physical range (1 for
    'unit'); the reference range r_p enters the phase only. Side output:
    dL/dd[p, i], the per-pulse distance gradient, written into the caller's
    buffer so exact per-pulse position gradients need no extra pass.
    """
    @staticmethod
    def forward(ctx, x, w, antennas, refs, freqs, pulse_grad_d, offset, range_model):
        ctx.save_for_backward(x, w, antennas, refs, freqs)
        ctx.pulse_grad_d, ctx.offset, ctx.range_model = pulse_grad_d, offset, range_model
        dist = torch.linalg.vector_norm(x.double()[None] - antennas[:, None], dim=-1)
        d = dist - refs[:, None]
        e = torch.exp((-4 * math.pi / C) * 1j * d[:, :, None] * freqs[None, None, :])
        weight = w.to(torch.complex128)
        amplitude = range_amplitude(dist, range_model)
        if amplitude is not None:
            weight = weight * amplitude
        return torch.einsum('pc,pcf->pf', weight, e)

    @staticmethod
    def backward(ctx, g):
        x, w, antennas, refs, freqs = ctx.saved_tensors
        diff = x.double()[None] - antennas[:, None]
        dist = torch.linalg.vector_norm(diff, dim=-1)
        d = dist - refs[:, None]
        e = torch.exp((-4 * math.pi / C) * 1j * d[:, :, None] * freqs[None, None, :])
        grad_w = torch.einsum('pf,pcf->pc', g, e.conj())
        kf = (4 * math.pi / C) * freqs
        t = torch.einsum('pf,pcf->pc', g.conj() * ((-1j) * kf)[None, :], e)
        amplitude, slope = _amplitude_and_slope(dist, ctx.range_model)
        if amplitude is not None:
            # d out / d dist = w (A' - j k A) e; A is real, so it scales the contractions.
            grad_w = grad_w * amplitude
            t = t * amplitude + torch.einsum('pf,pcf->pc', g.conj(), e) * slope
        grad_d = (t * w.to(torch.complex128)).real
        if ctx.pulse_grad_d is not None:
            ctx.pulse_grad_d[:, ctx.offset:ctx.offset + x.shape[0]] = grad_d
        grad_x = (grad_d[:, :, None] * (diff / dist[:, :, None])).sum(0)
        return grad_x.to(x.dtype), grad_w.to(w.dtype), None, None, None, None, None, None


def batched_native_forward(points, weights, antennas, refs, freqs, *, point_chunk, pulse_grad_d=None,
                           range_model='unit'):
    """Exact native monostatic phase for all pulses of a sector; complex128 accumulation."""
    if point_chunk <= 0:
        raise ValueError('point_chunk must be positive')
    if range_model not in RANGE_MODELS:
        raise ValueError(f'range_model must be one of {RANGE_MODELS}')
    p, k = weights.shape
    result = torch.zeros(p, freqs.shape[0], dtype=torch.complex128, device=points.device)
    for start in range(0, k, point_chunk):
        result = result + _PulseBlock.apply(points[start:start + point_chunk], weights[:, start:start + point_chunk],
                                            antennas, refs, freqs, pulse_grad_d, start, range_model)
    return result


def _look_angles(antennas):
    direction = antennas / torch.linalg.vector_norm(antennas, dim=-1, keepdim=True)
    return torch.acos(direction[:, 2].clamp(-1, 1)), torch.atan2(direction[:, 1], direction[:, 0])


def sector_forward(head, observations, readouts, *, method, point_chunk, pulse_grad_d=None):
    """Prediction ``[pulses, frequencies]`` of one channel for one pass-sector."""
    device = readouts[0]['antenna'].device
    antennas = torch.stack([r['antenna'] for r in readouts])
    refs = torch.tensor([o.reference_range_m for o in observations], dtype=torch.float64, device=device)
    freqs = readouts[0]['frequencies']
    for r in readouts[1:]:
        if not torch.equal(r['frequencies'], freqs):
            raise ValueError('Pulses of one shard must share the selected frequency vector')
    theta, phi = _look_angles(antennas)
    scene = head.field
    if method == 'isotropic':
        points, w = scene.active_scatterers()
        weights = w[None].expand(len(observations), -1)
    elif method in ('rift', 'rift_grid'):
        points, ws = None, []
        for i in range(len(observations)):
            pts, w = scene.active_scatterers(theta[i].float().reshape(1, 1), phi[i].float().reshape(1, 1))
            ws.append(w)
            points = pts if points is None else points
        weights = torch.stack(ws)
    else:
        raise AssertionError(method)
    range_model = head.recipe.get('range_model', LEGACY_RANGE_MODEL)
    if nufft.forward_evaluation(head.recipe) == nufft.NUFFT:
        # Opt-in control: the RIFT-dataset NUFFT, same per-pulse distance-gradient contract.
        raw = nufft.sector_render(points, weights, antennas, refs, nufft.grid_for(head, observations, freqs),
                                  point_chunk=point_chunk, range_model=range_model, pulse_grad_d=pulse_grad_d)
    else:
        raw = batched_native_forward(points, weights, antennas, refs, freqs, point_chunk=point_chunk,
                                     pulse_grad_d=pulse_grad_d, range_model=range_model)
    pred = head.gain(raw) if head.gain is not None else raw
    return pred, dict(points=points, weights=weights, antennas=antennas, theta=theta, phi=phi)


def _basis_groups(readouts):
    groups = {}
    for i, r in enumerate(readouts):
        groups.setdefault(r['q'].data_ptr(), []).append(i)
    return groups.values()


def projected_losses(pred, targets, readouts, mean_power):
    """Per-pulse ROI-projected losses, exactly ``objective`` applied pulse by pulse.

    Pulses are grouped by projector basis: the guarded range interval, hence
    the basis length, can differ by one cell between pulses of one sector.
    """
    losses = torch.zeros(len(readouts), dtype=torch.float64, device=pred.device)
    for ids in _basis_groups(readouts):
        q = readouts[ids[0]]['q']
        phase = torch.stack([readouts[i]['phase'] for i in ids])
        proj_pred = (phase * pred[ids]) @ q.conj()
        proj_target = (phase * targets[ids]) @ q.conj()
        losses[ids] = (proj_pred - proj_target).abs().square().mean(dim=1) / mean_power
    return losses


@torch.no_grad()
def projected_error_sums(pred, targets, readouts):
    """Summed projected squared error and target energy over the pulses."""
    error = torch.zeros((), dtype=torch.float64, device=pred.device)
    energy = torch.zeros((), dtype=torch.float64, device=pred.device)
    for ids in _basis_groups(readouts):
        q = readouts[ids[0]]['q']
        phase = torch.stack([readouts[i]['phase'] for i in ids])
        proj_pred = (phase * pred[ids]) @ q.conj()
        proj_target = (phase * targets[ids]) @ q.conj()
        error = error + (proj_pred - proj_target).abs().square().sum()
        energy = energy + proj_target.abs().square().sum()
    return error, energy


def sector_update(head, observations, readouts, targets, mean_power, scale, *, method, point_chunk, probe):
    """Back-propagate the summed scaled loss of one channel of one pass-sector.

    Returns the per-pulse losses and, for the adaptive method, the per-pulse
    refinement-statistic inputs (``delta_raw``-space position gradients
    ``[pulses, slots, 3]`` and, when probing, next-band scores ``[pulses, slots]``)
    that the per-pulse loop would have passed to
    ``accumulate_refinement_data_stats`` one pulse at a time.
    """
    scene = head.field
    device = readouts[0]['antenna'].device
    adaptive = method == 'rift'
    pulse_grad_d = None
    if adaptive:
        pulse_grad_d = torch.zeros(len(observations), int(scene.active_mask.sum()), dtype=torch.float64, device=device)
    pred, parts = sector_forward(head, observations, readouts, method=method, point_chunk=point_chunk,
                                 pulse_grad_d=pulse_grad_d)
    weights = parts['weights']
    if adaptive and probe:
        weights.retain_grad()
    if nufft.loss_domain(head.recipe) == nufft.FULL:
        losses = nufft.full_losses(pred, targets, mean_power)
    else:
        losses = projected_losses(pred, targets, readouts, mean_power)
    if not torch.isfinite(losses).all():
        raise ValueError('Nonfinite training loss')
    (losses.sum() / scale).backward()
    if not adaptive:
        return losses.detach(), None, None
    with torch.no_grad():
        points, antennas = parts['points'], parts['antennas']
        u = points.double()[None] - antennas[:, None]
        u = u / torch.linalg.vector_norm(u, dim=-1, keepdim=True)
        world = (scale * pulse_grad_d)[:, :, None] * u                                     # dL_p/dx_i, unscaled loss
    with torch.enable_grad():
        pos = scene.positions()
        jac = torch.autograd.grad(pos, scene.delta_raw, grad_outputs=torch.ones_like(pos))[0]   # elementwise map
    with torch.no_grad():
        active = scene.active_mask
        delta_grad = torch.zeros(len(observations), *scene.delta_raw.shape, dtype=scene.delta_raw.dtype, device=device)
        delta_grad[:, active] = (world * jac[active][None]).to(scene.delta_raw.dtype)
        angular = None
        if probe:
            wg = scale * weights.grad                                                       # dL_p/dweight_i, unscaled
            band = scene.basis_degree.view(1, -1) == (scene.order + 1)[:, None]
            basis = torch.stack([real_sh_basis(parts['theta'][i].float(), parts['phi'][i].float(), scene.max_degree)
                                 for i in range(len(observations))])
            n_band = band.sum(-1).clamp_min(1).to(scene.w_re.dtype)
            basis_sq = torch.einsum('pb,kb->pk', basis.square().to(scene.w_re.dtype), band.to(scene.w_re.dtype)) / n_band[None]
            angular = torch.zeros(len(observations), scene.w_re.shape[0], dtype=scene.w_re.dtype, device=device)
            angular[:, active] = wg.abs().to(scene.w_re.dtype) * basis_sq[:, active].sqrt()
    return losses.detach(), delta_grad, angular


@torch.no_grad()
def accumulate_batched_stats(scene, delta_grad, angular):
    """Add per-pulse statistics as P calls of ``accumulate_refinement_data_stats`` would."""
    active = scene.active_mask
    tanh_delta = torch.tanh(scene.delta_raw.detach())
    jac = scene.cell_half * (1.0 - tanh_delta.square())
    finite_delta = torch.isfinite(delta_grad).all(dim=-1)
    clean = torch.where(finite_delta[..., None], delta_grad, torch.zeros_like(delta_grad))
    jac_floor = (scene.cell_half.abs() * 1.0e-6).clamp_min(torch.finfo(jac.dtype).tiny)
    world = clean / torch.maximum(jac.abs(), jac_floor)[None]
    spatial_raw = (world * scene.cell_half[None]).norm(dim=-1)
    accepted = active[None] & finite_delta & torch.isfinite(spatial_raw)
    spatial = torch.where(accepted, spatial_raw, torch.zeros_like(spatial_raw))
    scene.refine_spatial_sum += spatial.sum(0)
    scene.refine_spatial_exposure += accepted.to(scene.w_re.dtype).sum(0)
    if angular is None:
        return
    eligible = active[None] & (scene.order < scene.max_degree)[None] & torch.isfinite(angular)
    scene.refine_angular_sum += torch.where(eligible, angular, torch.zeros_like(angular)).sum(0)
    scene.refine_angular_exposure += eligible.to(scene.w_re.dtype).sum(0)
