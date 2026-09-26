"""Plenoxel-style trilinear densification of the adaptive RIFT point scene on GOTCHA.

Tuning campaign, docs/RIFT_GOTCHA_Tune.md A41 (user decision, 2026-09-23). The scene is an
``AdaptivePointSHScene`` whose active points sit on one cubic lattice (the box support, or the
cube grid). A densify event replaces every surviving point with its 8 half-pitch octant
children, and each child starts from the trilinear interpolation of the parent lattice at the
child's centre, as Plenoxels upsamples its grid before optimizing the finer one. Empty lattice
nodes count as zero. The children are free points afterwards: own coefficients, own bounded
position offsets, fresh Adam rows.

Two conventions are this module's, not Plenoxels':
- Normalization. ``normalize='volume'`` (default, A41) multiplies the interpolated weight by
  ``weight_scale`` (1/8, the child's share of the parent volume; RIFT's point weight is extensive).
  Neighbouring backprojected nodes carry nearly unrelated carrier phases (2k times the 0.125 m
  pitch is about 50 rad), so interpolation cancels most of the energy (2% kept at the first event
  on the dense 23k scene): the start is close to minimum-norm. ``normalize='energy'`` rescales the
  children to the kept parents' coefficient energy instead; since rho is only about 0.13 after an
  event, that leaves an incoherent component whose part in the TRAIN operator's null space the
  gradient never removes (reviewer B52), so it is not the default.
- The coherent render is not preserved: children spread over the parent cell interfere
  differently from the parent point, so the loss moves at each event. Measured on densified
  copies of the dense 23k checkpoint (2157139, A42): TRAIN correlation 0.49 falls to 0.13 after one
  event and 0.035 after two. ``init='inherit'`` is the render-preserving alternative for the same
  every-point upsampling: the heir child (the octant holding the parent's learned position) takes
  the parent's coefficients and, through its bounded offset, the parent's exact position; the other
  seven start at zero. Anchors stay on the lattice either way.

Points whose energy is below ``energy_floor`` times the largest are dropped before an event (the
Plenoxels prune of empty voxels), and when more than ``max_active // 8`` survive, the most
energetic ones are kept so the children fit the active cap. Neither is a sparsity target.

``rift/`` stays unchanged (AGENTS.md): this module edits the scene's tensors in place under
``torch.no_grad()`` through the same slot contract ``split`` uses.
"""
from __future__ import annotations

import itertools

import torch

SCHEMA = 'gotcha_trilinear_densify_v1'
SIGNS = ((-1., -1., -1.), (-1., -1., 1.), (-1., 1., -1.), (-1., 1., 1.),
         (1., -1., -1.), (1., -1., 1.), (1., 1., -1.), (1., 1., 1.))


def point_energy(scene):
    """``prune(criterion='energy')``'s magnitude: coefficient norm over each point's unlocked bands."""
    unlocked = (scene.basis_degree.view(1, -1) <= scene.order[:, None]).to(scene.w_re.dtype)
    return ((scene.w_re.square() + scene.w_im.square()) * unlocked).sum(dim=-1).sqrt()


def _lattice(scene, slots, pitch):
    """Integer cell indices of ``slots`` on the lattice of ``pitch`` anchored at the support minimum."""
    origin = scene.support_min.double()
    coord = (scene.anchors[slots].double() - origin) / pitch - 0.5
    index = coord.round()
    if slots.numel() and float((coord - index).abs().max()) > 1e-3:
        raise ValueError('Active anchors are not on one lattice; trilinear densify needs a regular level')
    dims = ((scene.support_max.double() - origin) / pitch).round().long()
    if bool((dims <= 0).any()):
        raise ValueError('Degenerate support for the lattice')
    return index.long(), dims


@torch.no_grad()
def trilinear_densify(scene, *, max_active, energy_floor=1e-3, weight_scale=0.125, max_level=3,
                      optimizer=None, birth_event=None, normalize='volume', init='trilinear'):
    """Replace every surviving active point by 8 trilinearly initialized half-pitch children.

    Returns a JSON-ready report. Requires every active point at one level and cell size.
    """
    if normalize not in ('energy', 'volume'):
        raise ValueError("normalize must be 'energy' or 'volume'")
    if init not in ('trilinear', 'inherit'):
        raise ValueError("init must be 'trilinear' or 'inherit'")
    active = scene.active_mask.nonzero(as_tuple=True)[0]
    report = dict(schema=SCHEMA, active_before=int(active.numel()), init=init,
                  normalize=(normalize if init == 'trilinear' else None))
    if active.numel() == 0:
        return dict(report, status='empty')
    level = scene.level[active]
    half = scene.cell_half[active, 0].double()
    if bool((level != level[0]).any()) or float((half - half[0]).abs().max()) > 1e-6 * float(half[0]):
        raise ValueError('Trilinear densify needs every active point at one level and cell size')
    level_from = int(level[0])
    pitch = 2.0 * float(half[0])
    report.update(level_from=level_from, pitch_from_m=pitch)
    if level_from >= int(max_level):
        return dict(report, status='at_max_level', active_after=int(active.numel()))
    capacity = int(scene.active_mask.numel())
    limit = min(int(max_active), capacity)
    if limit < 8:
        raise ValueError('The active cap cannot hold one set of children')

    magnitude = point_energy(scene)
    mag = magnitude[active]
    peak = float(mag.max())
    if peak <= 0:
        return dict(report, status='zero_scene', active_after=int(active.numel()))
    above = mag >= energy_floor * peak
    parents = active[above]
    budget = limit // 8
    over_budget = max(int(parents.numel()) - budget, 0)
    if over_budget:
        ranking = torch.argsort(magnitude[parents], descending=True, stable=True)
        parents = parents[ranking[:budget]]
    total_energy = float(mag.square().sum())
    report.update(dropped_below_floor=int((~above).sum()), dropped_over_budget=over_budget,
                  parents=int(parents.numel()),
                  parent_energy_fraction=float(magnitude[parents].square().sum()) / total_energy)

    # Lattice values: every active point before the drop (the coarse field being upsampled);
    # bands above a point's order are zero by the scene's own contract, masked again for safety.
    index, dims = _lattice(scene, active, pitch)
    lookup = torch.full((int(dims.prod()),), -1, dtype=torch.long, device=active.device)
    flat = (index[:, 0] * dims[1] + index[:, 1]) * dims[2] + index[:, 2]
    lookup[flat] = active
    unlocked = (scene.basis_degree.view(1, -1) <= scene.order[:, None]).to(scene.w_re.dtype)
    node_re, node_im = scene.w_re * unlocked, scene.w_im * unlocked

    parent_index, _ = _lattice(scene, parents, pitch)
    signs = torch.tensor(SIGNS, dtype=torch.float64, device=active.device)            # [8, 3]
    n, nb = int(parents.numel()), scene.w_re.shape[1]
    child_re = torch.zeros(n, 8, nb, dtype=scene.w_re.dtype, device=active.device)
    child_im = torch.zeros_like(child_re)
    child_order = torch.zeros(n, 8, dtype=scene.order.dtype, device=active.device)
    parent_anchor = scene.anchors[parents].double()
    child_anchor = (parent_anchor[:, None, :] + 0.25 * pitch * signs[None]).to(scene.anchors.dtype)
    child_delta = torch.zeros(n, 8, 3, dtype=scene.delta_raw.dtype, device=active.device)
    if init == 'inherit':
        # AdaptivePointSHScene.split's heir rule, kept on the lattice: the heir's offset places it at
        # the parent's learned position (inside the heir octant by construction), so the render is kept.
        position = scene.positions()[parents].double()
        bits = (position >= parent_anchor).long()
        heir = bits[:, 0] * 4 + bits[:, 1] * 2 + bits[:, 2]
        rows = torch.arange(n, device=active.device)
        child_re[rows, heir] = node_re[parents]
        child_im[rows, heir] = node_im[parents]
        child_order[:] = scene.order[parents][:, None]
        relative = (position - child_anchor[rows, heir].double()) / (0.25 * pitch)
        child_delta[rows, heir] = torch.atanh(relative.clamp(-0.999999, 0.999999)).to(child_delta.dtype)
    # A child at parent centre + s·pitch/4 lies a quarter pitch from its parent node and three
    # quarters from the neighbour node in direction s on each axis: weights 3/4 and 1/4.
    for corner in (itertools.product((0, 1), repeat=3) if init == 'trilinear' else ()):
        step = torch.tensor(corner, dtype=torch.long, device=active.device)
        weight = 1.0
        for c in corner:
            weight *= 0.25 if c else 0.75
        node = parent_index[:, None, :] + signs.long()[None] * step[None, None]      # [n, 8, 3]
        inside = ((node >= 0) & (node < dims)).all(dim=-1)
        node_flat = (node[..., 0] * dims[1] + node[..., 1]) * dims[2] + node[..., 2]
        slot = torch.where(inside, lookup[node_flat.clamp(0, lookup.numel() - 1)], torch.full_like(node_flat, -1))
        present = slot >= 0
        safe = slot.clamp_min(0)
        mask = present[..., None].to(child_re.dtype)
        child_re += weight * node_re[safe] * mask
        child_im += weight * node_im[safe] * mask
        child_order = torch.where(present, torch.maximum(child_order, scene.order[safe]), child_order)
    child_mask = (scene.basis_degree.view(1, 1, -1) <= child_order[..., None]).to(child_re.dtype)
    child_re *= child_mask
    child_im *= child_mask
    interpolated_energy = float((child_re.square() + child_im.square()).sum())
    kept_energy = float(magnitude[parents].square().sum())
    if init == 'inherit':
        scale = 1.0
    elif normalize == 'energy':
        if not interpolated_energy > 0:
            raise ValueError('The interpolated children carry no energy')
        scale = (kept_energy / interpolated_energy) ** 0.5
    else:
        scale = float(weight_scale)
    child_re *= scale
    child_im *= scale
    report.update(interpolated_to_parent_energy=interpolated_energy / kept_energy, applied_scale=scale)

    # Retire every current point (parents and dropped ones alike), then fill fresh slots. The
    # tombstone clears each retired row's Adam moments; the fresh rows were inactive and are zero.
    scene._tombstone_slots(active, optimizer=optimizer)
    free = (~scene.active_mask).nonzero(as_tuple=True)[0]
    children = free[:8 * n]
    if children.numel() != 8 * n:
        raise AssertionError('Slot accounting failed after the tombstone')
    scene._tombstone_slots(children, optimizer=optimizer)
    scene.anchors[children] = child_anchor.reshape(-1, 3)
    scene.cell_half[children] = 0.25 * pitch
    scene.delta_raw[children] = child_delta.reshape(-1, 3)
    scene.w_re[children] = child_re.reshape(-1, nb)
    scene.w_im[children] = child_im.reshape(-1, nb)
    scene.order[children] = child_order.reshape(-1)
    scene.level[children] = level_from + 1
    scene.active_mask[children] = True
    if birth_event is not None:
        scene.refine_birth_event[children] = int(birth_event)
    scene.refresh_compact_sh_eval_cap()
    child_energy = float((child_re.square() + child_im.square()).sum())
    return dict(report, status='densified', level_to=level_from + 1, pitch_to_m=pitch / 2,
                active_after=int(scene.active_mask.sum()),
                weight_scale=(float(weight_scale) if init == 'trilinear' and normalize == 'volume' else None),
                energy_floor=float(energy_floor), max_active=int(limit),
                child_to_parent_coefficient_energy=child_energy / total_energy)


def scale_learning_rates(optimizer, scheduler, factor):
    """Multiply every group's learning rate, and the cosine schedule's base and floor, by ``factor``."""
    if not factor > 0:
        raise ValueError('The learning-rate factor must be positive')
    for group in optimizer.param_groups:
        group['lr'] *= factor
        if 'initial_lr' in group:
            group['initial_lr'] *= factor
    if scheduler is not None:
        scheduler.base_lrs = [lr * factor for lr in scheduler.base_lrs]
        if hasattr(scheduler, 'eta_min'):
            scheduler.eta_min *= factor
    return [float(group['lr']) for group in optimizer.param_groups]


def densify_due(recipe, completed_epoch):
    """True when the 1-based ``completed_epoch`` is one of the recipe's densify epochs."""
    densify = recipe.get('densify')
    return densify is not None and int(completed_epoch) in set(densify['epochs'])
