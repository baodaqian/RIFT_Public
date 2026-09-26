"""Batched Stage-1 sub-aperture objective for the PVC Sugavanam--Ertin lane.

``rift.sugavanam_ertin_acquisition.data_objective`` streams one native
observation at a time through ``fourier_forward`` (one pulse per call for
GOTCHA, one view for the collection), with about one million kernel elements
per block and one backward pass per observation. On an accelerator that is
launch-bound (GOTCHA Camry: ~24000 pulses per Stage-1 iteration, three or
more passes per sub-aperture step). ``fourier_forward`` already accepts many
antenna directions per call, so this module evaluates every observation of a
sub-aperture group that shares a native frequency vector in one call, in
blocks of ``recipe["pair_chunk"]`` directions and a 2^24-element kernel
budget, and takes one backward pass for the group. Values equal the original
objective up to floating-point summation order; the recipe, the acquisition
identity, the partition and the checkpoint schema are untouched, so existing
Stage-1 checkpoints resume into it. PVC-only; the CUDA lane is unchanged.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from rift.sugavanam_ertin_acquisition import data_objective, response, validation_readout

EXECUTION = 'pvc_batched_subaperture_objective_v1'
ELEMENT_BUDGET = 1 << 24   # complex128 kernel elements per block (256 MiB); an execution bound only
KINDS = ('gotcha_native', 'rift_collection')


def fourier_forward_budget(points, weights, frequencies, directions, reference, amplitude, *,
                           cc, point_chunk, pair_chunk, element_budget=ELEMENT_BUDGET):
    """``rift.sugavanam_ertin_acquisition.fourier_forward`` with a larger block budget.

    Same kernel, same checkpointed blocks; only the point count per block is
    bounded by ``element_budget`` instead of 2^20 elements.
    """
    k = 2*torch.pi*torch.as_tensor(frequencies, device=points.device, dtype=torch.float64)/cc
    blocks = []
    for start in range(0, len(reference), pair_chunk):
        d, r, a = (v[start:start+pair_chunk] for v in (directions, reference, amplitude))
        local_chunk = min(point_chunk, max(1, element_budget//(len(k)*len(r))))
        result = weights.new_zeros((len(k), len(r)))
        def block(p, w, direction, origin_range, scale):
            path = origin_range[None]-p.double() @ direction.T
            kernel = torch.exp(-1j*k[:, None, None]*path[None])*scale[None, None]
            return (kernel*w[None, :, None]).sum(1)
        for p, w in zip(points.split(local_chunk), weights.split(local_chunk)):
            if torch.is_grad_enabled() and (p.requires_grad or w.requires_grad):
                result = result+checkpoint(block, p, w, d, r, a, use_reentrant=False)
            else:
                result = result+block(p, w, d, r, a)
        blocks.append(result)
    return torch.cat(blocks, dim=1)


def observation_pairs(acquisition, observation, device):
    """One observation as (cc, frequencies, directions [P,3], reference [P], amplitude [P], target [F,P]).

    Exactly the quantities the acquisition's own ``render`` builds, so the
    batched forward is that render applied to many observations at once.
    """
    if acquisition.kind == 'gotcha_native':
        from rift.gotcha_dataset import C
        region = acquisition.dataset.region
        rotation = torch.as_tensor(region.rotation_local_to_native, device=device, dtype=torch.float64)
        translation = torch.as_tensor(region.translation_m, device=device, dtype=torch.float64)
        antenna = torch.as_tensor(observation.position_m, device=device, dtype=torch.float64)
        relative = (antenna-translation) @ rotation
        r = relative.norm()
        directions = 2*(relative/r)[None]
        reference = (2*(r-observation.reference_range_m)).reshape(1)
        target = torch.as_tensor(response(observation), device=device, dtype=torch.complex128).reshape(-1, 1)
        return C, np.array(observation.frequencies_hz, copy=True), directions, reference, torch.ones_like(reference), target
    if acquisition.kind == 'rift_collection':
        from rift.config import cc
        tx = torch.as_tensor(observation["tx"], device=device, dtype=torch.float64)
        rx = torch.as_tensor(observation["rx"], device=device, dtype=torch.float64)
        rt, rr = tx.norm(dim=-1), rx.norm(dim=-1)
        directions = (rx[:, None]/rr[:, None, None]+tx[None]/rt[None, :, None]).reshape(-1, 3)
        reference = (rr[:, None]+rt[None, :]).flatten()
        amplitude = (4*torch.pi)**-2/reference.square()/acquisition.kernel_scale
        target = torch.as_tensor(response(observation), device=device, dtype=torch.complex128).reshape(len(observation["freqs"]), -1)
        return cc, np.array(observation["freqs"], copy=True), directions, reference, amplitude, target
    raise TypeError(f'no batched pairs for acquisition kind {acquisition.kind!r}')


def _collect(acquisition, observations, device):
    """Group observations by native frequency vector; concatenate their pairs."""
    groups = {}
    for observation in observations:
        cc, f, d, r, a, t = observation_pairs(acquisition, observation, device)
        g = groups.setdefault(f.tobytes(), dict(cc=cc, f=f, d=[], r=[], a=[], t=[]))
        g['d'].append(d); g['r'].append(r); g['a'].append(a); g['t'].append(t)
    for g in groups.values():
        g['d'], g['r'], g['a'] = torch.cat(g['d']), torch.cat(g['r']), torch.cat(g['a'])
        g['t'] = torch.cat(g['t'], dim=1)
    return list(groups.values())


def batched_data_objective(acquisition, points, weights, indices, statistics, recipe, *, gradient=False,
                           element_budget=ELEMENT_BUDGET):
    """``data_objective`` of a sub-aperture group with one forward (and backward) per frequency vector."""
    if getattr(acquisition, 'kind', None) not in KINDS:
        return data_objective(acquisition, points, weights, indices, statistics, recipe, gradient=gradient)
    denominator = sum(statistics["per_view_samples"][i] for i in indices)
    x = weights.detach().requires_grad_(gradient)
    observations = (o for i in indices for o in acquisition.observations(acquisition.keys["train"][i], role="train"))
    groups = _collect(acquisition, observations, points.device)
    total = None
    with torch.set_grad_enabled(gradient):
        for g in groups:
            pred = fourier_forward_budget(points, x, g['f'], g['d'], g['r'], g['a'], cc=g['cc'],
                                          point_chunk=recipe["point_chunk"], pair_chunk=recipe["pair_chunk"],
                                          element_budget=element_budget)
            term = .5*(pred-g['t']/statistics["rms"]).abs().square().sum()/denominator
            total = term if total is None else total+term
    loss = float(total.detach())
    if not gradient:
        return loss
    return loss, torch.autograd.grad(total, x)[0].detach()


@torch.no_grad()
def batched_validation_readout(acquisition, points, fields, assignments, statistics, recipe, *,
                               element_budget=ELEMENT_BUDGET):
    """``validation_readout`` with one forward per validation view and frequency vector."""
    if getattr(acquisition, 'kind', None) not in KINDS:
        return validation_readout(acquisition, points, fields, assignments, statistics, recipe)
    numerator, denominator, count = 0., 0., 0
    for i, key in enumerate(acquisition.keys["validation"]):
        weights = fields[int(assignments[i])].to(points.device)
        for g in _collect(acquisition, acquisition.observations(key, role="validation"), points.device):
            pred = fourier_forward_budget(points, weights, g['f'], g['d'], g['r'], g['a'], cc=g['cc'],
                                          point_chunk=recipe["point_chunk"], pair_chunk=recipe["pair_chunk"],
                                          element_budget=element_budget)
            target = g['t']/statistics["rms"]
            numerator += float((pred-target).abs().square().sum())
            denominator += float(target.abs().square().sum())
            count += target.numel()
    if denominator <= 0 or not np.isfinite([numerator, denominator]).all():
        raise ValueError("Invalid full-native validation readout")
    return dict(global_complex_rel_mse=numerator/denominator, squared_error=numerator,
                target_energy=denominator, samples=count, views=len(assignments),
                output="stage1_subaperture_diagnostic_not_SDF_NVS")
