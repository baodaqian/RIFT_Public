"""Per-view visibility (two-way transmittance) for the radar forward operator.

Until 2026-08-06 RIFT's operator was a plain sum over scatterers with a
geometric gain and a propagation phase and NO visibility term: every voxel
radiated to every viewpoint, always, including voxels sitting behind an
opaque target. Four competitors already model this (SH-SAS Eq. 7, DART,
RadarSim's squared two-way ``prod(1-alpha)^2``, GeRaF 2.0's ``T(u)^2``); we
are late here, not first, and the citations should say so.

This module adds the missing factor. The rendered weight of scatterer ``n``
under viewpoint ``v`` becomes

    w_n  ->  w_n * exp(-2 * tau(o_v, x_n))
    tau(o, x) = \\int_0^{|x-o|} sigma(o + s*u) ds     (u = (x-o)/|x-o|)

with ``sigma >= 0`` an extinction field derived from the scene itself.


Why this is CHEAP here: the small-aperture approximation
-------------------------------------------------------
Naively transmittance depends on the path from EACH Tx element to the point
and from the point to EACH Rx element, so the per-point factor becomes a
function of the individual (Tx, Rx) pair. That is 256x more work and -- more
importantly -- it is the kind of term that could break the range
factorization ``rift/range_operator.py`` depends on.

It doesn't, because the array is tiny relative to the standoff. Measured on
``pec_sphere_fmcw_16t16r_79ghz_bw3ghz_r10m_2k.npz``: array aperture 2.85 cm,
standoff 10 m, so the array subtends 0.163 deg from a scene point and the
lateral ray walk 1 m into the scene is 2.85 mm -- about 1/22 of a g48 voxel
(62.5 mm). Transmittance is therefore constant across the array far below
grid resolution: compute it ONCE per (point, VIEW) from the array phase
centre, not per (point, Tx, Rx). The factor is then pair-independent AND
frequency-independent, so it multiplies the complex weight before the
operator is called and both operators (``brute`` and ``range``) are
untouched. ``T_tx * T_rx -> T^2`` recovers the squared two-way form the
prior art already uses.

If a future dataset has an aperture that is NOT small compared with the
standoff (AirSAS: 0.2 m turntable), re-derive this before reusing the module.


Where sigma comes from -- and why NOT from the DC coefficient
-------------------------------------------------------------
SH-SAS keys opacity to the isotropic term, ``rho = |sigma_DC| * zeta``, with
normals from ``grad|sigma_DC|`` and a Lambertian lobe. That is a diffuse-
scattering prior stack. It fails in the specular PEC regime for a one-line
reason: a flat conducting plate is perfectly opaque and has almost no
isotropic return, so ``|c_00|`` is small while the true opacity is total --
DC-keying makes occluders transparent exactly where occlusion matters. We
have already paid for this lesson in another guise (DC-based PRUNING deleted
the specular scatterers a PEC target is made of, which is why
``--prune-criterion energy`` is the default).

So the default key here is the same rotation-invariant angular energy the
prune criterion uses, ``e = sqrt(sum_lm |c_lm|^2)`` over a voxel's UNLOCKED
bands. ``key="dc"`` reproduces SH-SAS's choice and exists so the two can be
run as arms of one experiment with everything else held fixed.


Gauge invariance -- this is load-bearing
----------------------------------------
The ``(gain, scene)`` scale is an EXACTLY flat direction of the objective
(``|g*F(w) - S|^2`` is invariant under ``g -> g/a, w -> a*w``), and measured
``|g|`` spans 2e-4 to 1e5 across checkpoints. A FIXED extinction scale
applied to a raw ``|w|`` would therefore mean nothing: the same scene at a
different gauge would get a different optical depth. ``opacity_volume``
consequently normalizes by the mean energy over occupied entries, leaving a
dimensionless field with mean 1. ``zeta`` then has a clean reading:

    zeta = the one-way optical depth contributed by ONE average-energy voxel.

The whole model stays exactly gauge-invariant: ``w -> a*w`` leaves the
normalized field, hence ``T^2``, unchanged. Two consequences worth knowing
before reading any result: ``zeta -> 0`` recovers the no-occlusion operator
exactly, so an occlusion arm can only match or beat its baseline in TRAIN
loss and the learned ``zeta`` is the quantity to report; and ``zeta`` is
learnable in absolute units, so it is subject to the same
"Adam's step is ~lr in absolute units" caveat as everything else here --
it lives in its own parameter group.


Start NEARLY TRANSPARENT, not opaque
------------------------------------
``exp(-2*tau)`` saturates, and a saturated scene has NO gradient: at
``zeta = 1.5`` on a dense scene every voxel past the first layer sits at
``T^2 ~ 4e-11`` and ``d(T^2)/d(sigma) ~ -2*L*T^2`` is numerically zero, so
training can never walk the opacity back down. Measured in
``scripts/validate_occlusion.py`` Stage E. Initialize ``zeta`` small (the
CLI default is 0.1, i.e. an average voxel is ~82% transparent two-way) and
let the fit raise it. Note the field is mean-normalized, so a shell voxel in
a mostly-empty box carries an ``e_hat`` well above 1 and reaches an opaque
optical depth at a ``zeta`` well below 1.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint

from rift.distributed import all_reduce_sum_grad

_TINY = 1e-30


class OcclusionScale(nn.Module):
    """The extinction scale ``zeta``, held in log space so it stays positive.

    Kept as a module (rather than a bare tensor) so it lands in the
    checkpoint's ``model_state_dict`` sibling slots the same way
    ``GlobalComplexGain`` does, and so a run can be resumed with its learned
    opacity intact.
    """

    def __init__(self, init_scale: float = 1.0, learnable: bool = True, device=None):
        super().__init__()
        if init_scale <= 0:
            raise ValueError("occlusion scale must be > 0")
        self.log_zeta = nn.Parameter(
            torch.tensor(math.log(init_scale), dtype=torch.float32, device=device),
            requires_grad=bool(learnable),
        )

    def forward(self) -> torch.Tensor:
        return torch.exp(self.log_zeta)

    @property
    def value(self) -> float:
        return float(torch.exp(self.log_zeta.detach()).item())


def scene_energy_flat(model, key: str = "energy") -> torch.Tensor:
    """Per-entry non-negative opacity proxy, flattened to the FULL grid order.

    Returns ``[granularity**3]`` for every grid representation, already
    all-reduced for the scene-sharded case, so the caller sees one global
    field regardless of world size. Entry order is the flat order of
    ``rift.encoding.generate_dynamic_grid`` (x outer, y mid, z inner), which
    is what ``ray_transmittance`` indexes into.

    The sqrt is masked rather than eps-padded: ``sqrt(x + eps)`` has gradient
    ``1/(2*sqrt(eps))`` at x = 0, which is enormous, and an entry with
    exactly zero energy should contribute exactly zero opacity gradient.
    ``clamp_min`` inside the sqrt keeps the unselected ``torch.where`` branch
    finite (torch.where evaluates both branches in backward).
    """
    from rift.sharded_scene import ShardedSHVoxelGridScene
    from rift.sparse_scene import SHVoxelGridScene, VoxelGridScene

    if key not in ("energy", "dc"):
        raise ValueError(f"opacity key must be 'energy' or 'dc', got {key!r}")

    if isinstance(model, SHVoxelGridScene):
        w_re, w_im = model.w_re, model.w_im                      # [G,G,G,C]
        order = model.order.unsqueeze(-1)
        basis_degree = model.basis_degree.view(1, 1, 1, -1)
        active = model.active_mask.reshape(-1)
        flat_shape = (-1,)
    elif isinstance(model, ShardedSHVoxelGridScene):
        w_re, w_im = model.w_re, model.w_im                      # [n_local,C]
        order = model.order[:, None]
        basis_degree = model.basis_degree.view(1, -1)
        active = model.active_mask
        flat_shape = (-1,)
    elif isinstance(model, VoxelGridScene):
        # isotropic scene: one complex weight per voxel, no bands to sum
        mag = torch.sqrt((model.w_re ** 2 + model.w_im ** 2).clamp_min(_TINY))
        sq = model.w_re ** 2 + model.w_im ** 2
        e = torch.where(sq > 0, mag, torch.zeros_like(sq)).reshape(-1)
        return e * model.active_mask.reshape(-1).to(e.dtype)
    else:
        raise TypeError(
            f"--occlusion supports the voxel-grid scenes only, got {type(model).__name__}. "
            "point_sh has continuous positions, so its extinction field needs a "
            "splatting rule that has not been designed yet; mlp has no explicit grid."
        )

    if key == "dc":
        sq = w_re[..., 0] ** 2 + w_im[..., 0] ** 2
    else:
        unlocked = (basis_degree <= order).to(w_re.dtype)
        sq = ((w_re ** 2 + w_im ** 2) * unlocked).sum(dim=-1)
    e = torch.where(sq > 0, torch.sqrt(sq.clamp_min(_TINY)), torch.zeros_like(sq))
    e = e.reshape(*flat_shape) * active.to(e.dtype)

    if isinstance(model, ShardedSHVoxelGridScene):
        # scatter this rank's strided voxels back into the full grid and sum:
        # the supports are disjoint, so backward through the all-reduce is the
        # identity (same argument as the S-parameter reduction in train.py).
        n_total = model.granularity ** 3
        full = torch.zeros(n_total, dtype=e.dtype, device=e.device)
        full = full.index_add(0, model.voxel_index, e)
        e = all_reduce_sum_grad(full)
    return e


def opacity_volume(model, scale, key: str = "energy") -> torch.Tensor:
    """Extinction field ``sigma`` [1/m], flat ``[granularity**3]``, >= 0.

    ``scale`` is ``zeta`` (a tensor from ``OcclusionScale`` or a float). See
    the module docstring for why the field is mean-normalized: ``zeta`` reads
    as the one-way optical depth of ONE average-energy voxel, and the whole
    construction is invariant to the (gain, scene) gauge.
    """
    e = scene_energy_flat(model, key=key)
    n_occ = int((e > 0).sum().item())
    if n_occ == 0:
        return torch.zeros_like(e)
    # The mean stays IN the graph. Detaching it would leave the objective
    # unchanged (e/mean(e) is scale-invariant either way) but would make the
    # reported gradient not the gradient of that objective -- caught by the
    # finite-difference stage of scripts/validate_occlusion.py, where the
    # detached version read 1e-20 against a true 3e-2. The extra term is one
    # global mean, so it is cheap; only the occupied COUNT is treated as
    # constant, which it is except at the measure-zero set where an entry
    # crosses zero.
    mean_e = (e.sum() / n_occ).clamp_min(_TINY)
    pitch = 2.0 * float(model.extent) / int(model.granularity)
    zeta = scale() if isinstance(scale, nn.Module) else torch.as_tensor(
        scale, dtype=e.dtype, device=e.device)
    return (zeta.to(e.dtype) / pitch) * (e / mean_e)


def _tau_chunk(points, sigma_flat, origin, extent, granularity, n_steps, back_off):
    """Optical depth from ``origin`` to each point. Purely functional so it is
    safe under ``torch.utils.checkpoint`` (recomputed on backward)."""
    device = points.device
    g = int(granularity)
    e = float(extent)
    pitch = 2.0 * e / g

    d = points - origin.view(1, 3)
    length = torch.linalg.norm(d, dim=-1).clamp_min(1e-12)          # [Nc]
    u = d / length[:, None]
    # keep the slab divisions finite for axis-parallel rays; the resulting
    # +-huge t values are still correct for the min/max slab test
    u_safe = torch.where(u.abs() < 1e-12, torch.full_like(u, 1e-12), u)

    t_a = (-e - origin.view(1, 3)) / u_safe
    t_b = (e - origin.view(1, 3)) / u_safe
    s_in = torch.minimum(t_a, t_b).max(dim=-1).values.clamp_min(0.0)  # box entry
    s_out = torch.maximum(t_a, t_b).min(dim=-1).values               # box exit

    # integrate only what is STRICTLY IN FRONT of the point: back off half a
    # voxel so a scatterer never occludes itself
    end = torch.minimum(length - back_off, s_out)
    start = torch.minimum(s_in, end)
    ds = ((end - start) / float(n_steps)).clamp_min(0.0)             # [Nc]

    m = torch.arange(n_steps, device=device, dtype=points.dtype) + 0.5
    s = start[:, None] + ds[:, None] * m.view(1, -1)                 # [Nc, M]
    x = origin.view(1, 1, 3) + s[:, :, None] * u[:, None, :]         # [Nc, M, 3]

    idx3 = torch.floor((x + e) / pitch)
    inside = ((idx3 >= 0) & (idx3 <= g - 1)).all(dim=-1) & (ds[:, None] > 0)
    idx3 = idx3.clamp(0, g - 1).to(torch.long)
    flat = (idx3[..., 0] * g + idx3[..., 1]) * g + idx3[..., 2]      # x outer, z inner

    sig = sigma_flat[flat] * inside.to(sigma_flat.dtype)
    return sig.sum(dim=-1) * ds


def ray_transmittance(
    points: torch.Tensor,
    sigma_flat: torch.Tensor,
    origin: torch.Tensor,
    extent: float,
    granularity: int,
    n_steps: Optional[int] = None,
    step_frac: float = 0.5,
    point_chunk: int = 16384,
    two_way: bool = True,
    march_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """``exp(-2*tau)`` (or ``exp(-tau)``) from ``origin`` to each of ``points``.

    Fixed-step quadrature with nearest-voxel lookup -- i.e. the piecewise-
    constant field the voxel grid actually represents, evaluated at
    ``step_frac`` of a voxel pitch. ``n_steps`` defaults to the number needed
    for the WORST-case chord (the box diagonal) to be sampled that finely;
    the step is per-ray adaptive (``ds = path/n_steps``), so shorter chords
    are sampled finer than requested, never coarser.

    Marching runs in fp32 regardless of the operator's compute dtype: this is
    a real attenuation with no phase in it, and fp32 is ~1e-7 relative --
    orders of magnitude below any scene effect, and half the memory of the
    fp64 the phase path needs. ``march_dtype`` overrides that; it exists so
    ``scripts/validate_occlusion.py`` can run finite differences, which are
    unresolvable at fp32. Trilinear interpolation is deliberately NOT used
    (project rule: trilinear is for visualization only, never in the training
    path).

    Differentiable w.r.t. ``sigma_flat``. NOT differentiable w.r.t. the point
    positions (nearest-voxel lookup is piecewise constant) -- fine for the
    grid representations, whose positions are frozen on the lattice, and the
    reason ``point_sh`` is rejected upstream.
    """
    g = int(granularity)
    if n_steps is None or n_steps <= 0:
        n_steps = max(8, int(math.ceil(math.sqrt(3.0) * g / max(step_frac, 1e-3))))
    pitch = 2.0 * float(extent) / g
    back_off = 0.5 * pitch

    march_dtype = march_dtype or torch.float32
    pts = points.to(march_dtype)
    sig = sigma_flat.to(march_dtype)
    org = origin.to(device=points.device, dtype=march_dtype).reshape(3)

    taus = []
    for start in range(0, pts.shape[0], point_chunk):
        chunk = pts[start:start + point_chunk]
        args = (chunk, sig, org, float(extent), g, int(n_steps), back_off)
        if torch.is_grad_enabled() and sigma_flat.requires_grad:
            taus.append(torch.utils.checkpoint.checkpoint(
                _tau_chunk, *args, use_reentrant=False))
        else:
            taus.append(_tau_chunk(*args))
    tau = torch.cat(taus, dim=0)
    return torch.exp(-(2.0 if two_way else 1.0) * tau).to(points.dtype)


def view_transmittance(
    model,
    scale,
    origin: torch.Tensor,
    key: str = "energy",
    n_steps: Optional[int] = None,
    step_frac: float = 0.5,
    point_chunk: int = 16384,
    active_only: bool = True,
    march_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """``T^2`` at this scene's ACTIVE scatterers, for one viewpoint.

    Returned in the same order as ``model.active_scatterers(...)`` so the
    caller can multiply the two elementwise. For the sharded scene that is
    this rank's own voxels evaluated against the GLOBAL extinction field --
    a rank must see occluders it does not own.
    """
    from rift.sharded_scene import ShardedSHVoxelGridScene

    sigma = opacity_volume(model, scale, key=key)
    if isinstance(model, ShardedSHVoxelGridScene):
        pos = model.grid_positions
        mask = model.active_mask
    else:
        pos = model.grid_positions.reshape(-1, 3)
        mask = model.active_mask.reshape(-1)
    if active_only:
        pos = pos[mask]
    return ray_transmittance(
        pos, sigma, origin, model.extent, model.granularity,
        n_steps=n_steps, step_frac=step_frac, point_chunk=point_chunk,
        march_dtype=march_dtype,
    )


def array_phase_centre(rx_pos: torch.Tensor, tx_pos: torch.Tensor) -> torch.Tensor:
    """The single origin every ray in a view emanates from.

    Justified by the small-aperture measurement in the module docstring: at
    2.85 cm of aperture and 10 m of standoff the Tx-element, Rx-element and
    centre rays agree to ~1/22 of a voxel, so collapsing them to the array
    phase centre is exact at grid resolution and 256x cheaper.
    """
    return 0.5 * (rx_pos.mean(dim=0) + tx_pos.mean(dim=0))
