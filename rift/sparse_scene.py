"""Explicit (Plenoxel-style) voxel-grid scene representations: scattering
weight per voxel is a directly-optimized parameter, not an MLP output.
Two variants: VoxelGridScene (one ISOTROPIC complex scalar per voxel) and
SHVoxelGridScene (real spherical-harmonic coefficients per voxel, so
reflectivity varies with viewing angle). Both support magnitude-based
pruning so that near-zero voxels are excluded from the forward operator
entirely -- since forward_operator_lessparallel's cost scales linearly in
scatterer count, pruning empty space directly shrinks the dominant
per-viewpoint cost, independent of whatever fraction of the training cube
the real target actually occupies.

Drop-in alternatives to rift/model.py's MLP: train.py selects between all
three via --scene-repr. Unlike the MLP, these hold their own positions (a
fixed regular grid, not re-queried) and expose them directly through
active_scatterers() rather than being called with x_scene_input.
"""
import math

import torch
import torch.nn as nn

from rift.distributed import all_reduce_sum
from rift.encoding import generate_dynamic_grid
from rift.spherical_harmonics import basis_degree_index, num_sh_basis, real_sh_basis


# --- grow() threshold selection -------------------------------------------
#
# grow()'s original rule -- unlock the next SH band wherever the accumulated
# gradient magnitude is >= threshold_fraction * the MAX over eligible entries
# -- turned out to be non-selective on the B787 grid: Round 3 (2026-08-01)
# grew 110592/110592 voxels on every check at threshold_fraction 0.1, i.e.
# min(avg_grad) >= 0.1 * max(avg_grad) across the whole grid. A relative-to-max
# test only discriminates when the statistic has wide dynamic range; here it
# spans under a decade, so "top 10% of the max" is "everything".
#
# Two fixes, both here:
#   * mode="quantile" -- threshold_fraction now means "grow the top q FRACTION
#     of eligible entries", which is selective by construction whatever the
#     distribution looks like.
#   * a selectivity report printed at every grow check, mapping each candidate
#     relmax threshold to the fraction it would actually select -- so ONE run
#     measures the whole threshold->selectivity curve instead of needing a
#     separate run per candidate value.
#
# Both resolve the threshold by BISECTION on a globally-reduced count, which
# behaves identically on one GPU and on a scene-sharded run and is exact to
# float precision. (A binned histogram was tried first and rejected: log-spaced
# bins have their worst resolution near relmax = 1, which is precisely where a
# tight gradient distribution like this one puts all of its mass.) Each grow
# check costs ~40 all-reduces of a scalar -- negligible next to an epoch.
_BISECT_ITERS = 50

_REPORT_RELMAX = (0.9, 0.7, 0.5, 0.3, 0.1, 0.03)
_REPORT_RHO = (0.3, 0.1, 0.05, 0.02, 0.01, 0.003)


# `reduce` must be True ONLY for the voxel-sharded scene, where each rank holds
# a disjoint slice. The plain scenes are REPLICATED on every rank under
# torchrun, so all-reducing their counts would multiply every fraction by the
# world size -- caught by validate_distributed_scene.py's E block at world 2/3.
@torch.no_grad()
def _global_frac_ge(vals_local, thresholds, n_total, device, reduce):
    """Fraction of entries >= each threshold. `thresholds` is a 1-D tensor; one
    all-reduce covers all of them. When `reduce`, every rank must call this."""
    if n_total <= 0:
        return torch.zeros_like(thresholds, dtype=torch.float64)
    if vals_local.numel():
        counts = (vals_local.reshape(-1, 1) >= thresholds.reshape(1, -1)).sum(dim=0).double()
    else:
        counts = torch.zeros(thresholds.numel(), dtype=torch.float64, device=device)
    if reduce:
        counts = all_reduce_sum(counts)
    return counts / n_total


@torch.no_grad()
def _threshold_for_fraction(vals_local, hi, fraction, n_total, device, reduce):
    """Bisect for the threshold selecting `fraction` of the entries. Monotone in
    t, so plain bisection on [0, hi] converges; returns the value whose selected
    fraction is closest to the request."""
    lo = torch.zeros((), dtype=torch.float64, device=device)
    hi = hi.double().clone()
    best, best_err = hi.clone(), float("inf")
    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        frac = float(_global_frac_ge(vals_local, mid.reshape(1), n_total, device, reduce)[0])
        err = abs(frac - fraction)
        if err < best_err:
            best_err, best = err, mid.clone()
        if frac > fraction:      # selecting too many -> raise the threshold
            lo = mid
        else:
            hi = mid
    return best


@torch.no_grad()
def _selectivity_report(vals_local, n_total, device, label, points, scale, reduce):
    """`label`/`points` describe which knob is being characterized; `scale` is
    what the report's x-values are multiplied by to become absolute thresholds
    (the max for --grow-threshold relmax, 1 for --grow-tail-ratio's rho)."""
    ts = torch.tensor(points, dtype=torch.float64, device=device) * scale
    fr = _global_frac_ge(vals_local, ts, n_total, device, reduce)
    return label + ", ".join(f"{p:g}->{float(f):.3f}" for p, f in zip(points, fr))


@torch.no_grad()
def _tail_ratio_report(ratio_local, n_total=None, device=None, reduce=False):
    """grow_angular's analogue of the grad selectivity report: what fraction of
    eligible entries each candidate --grow-tail-ratio would unlock. rho is
    already normalized to [0,1], so it is reported directly."""
    device = device if device is not None else ratio_local.device
    if n_total is None:
        n_total = ratio_local.numel()
    if n_total <= 0:
        return ""
    one = torch.ones((), dtype=torch.float64, device=device)
    return _selectivity_report(ratio_local, n_total, device,
                               "tail-ratio selectivity (rho -> frac grown): ",
                               _REPORT_RHO, one, reduce)


@torch.no_grad()
def grow_threshold_and_report(vals_local, global_max, threshold_fraction, mode,
                              n_total=None, reduce=False):
    """Shared by every scene's grow(): resolve the absolute gradient threshold
    and build the selectivity report string.

    mode="relmax"   -- threshold = threshold_fraction * global_max (legacy).
    mode="quantile" -- threshold = the value that selects the top
                       threshold_fraction of eligible entries.

    With reduce=True (sharded scene only) this contains collectives, so EVERY
    rank must call it -- including ranks whose local eligible set is empty.
    """
    device = global_max.device
    if n_total is None:
        n_total = vals_local.numel()
    if mode == "quantile":
        thresh = _threshold_for_fraction(vals_local, global_max, threshold_fraction,
                                         n_total, device, reduce)
    elif mode == "relmax":
        thresh = global_max.double() * threshold_fraction
    else:
        raise ValueError(f"mode must be 'relmax' or 'quantile', got {mode!r}")
    report = _selectivity_report(vals_local, n_total, device,
                                 "grad selectivity (relmax -> frac grown): ",
                                 _REPORT_RELMAX, global_max.double(), reduce)
    return thresh.to(global_max.dtype), report


# --- prune() threshold selection ------------------------------------------
#
# The original rule -- deactivate everything below threshold_fraction * the
# CURRENT max -- is a RATCHET, and Round 4 (2026-08-02) died of it. Measured
# on the R4 arms (--prune-every 10 --prune-threshold 0.01, active voxel
# fraction per check): 45.2% -> 24.3% -> 10.3% -> 3.2% -> 1.8% -> 0.5% ->
# 0.2% -> 22 voxels, at which point the scene cannot fit anything, the
# calibration gain collapses and training sits at the predict-zero floor.
#
# It is the COMPOSITION prune-then-retrain that ratchets, not prune alone:
# applied twice to frozen weights the rule removes nothing the second time
# (the max survives its own cut -- validate_prune_modes.py block 4b asserts
# this). What kills it is the ~10 epochs between checks, in which the
# optimizer pushes the remaining energy onto the surviving scatterers; the
# next check therefore measures a MORE peaked distribution against a
# recomputed max, and the same nominal 1% cut bites deeper. There is no fixed
# point but the empty scene.
#
# Note also that the damage is nearly invisible in the energy metric -- the
# voxels relmax deletes carry very little energy each. A coherent sum needs
# its small terms (the good B787 fit keeps 62% of its energy OFF the
# airframe and breaks when those DOF are removed; see the --extent 0.06 A/B),
# so watch the ACTIVE COUNT and the train loss at each prune check, not the
# discarded energy fraction.
#
# Note the failure is loudest exactly where the representation is BEST. A
# low-degree scene (the R4 winners, --sh-init-degree 0..3) is far more peaked
# than a degree-6 one, so 0.01*max catches half the grid on the very first
# check; the degree-6 arm pruned only 100->84% over the same schedule and
# never ran away. Do not read that as "degree 6 is safer" -- it is the same
# ratchet, just slower.
#
# Two replacement modes, both with a fixed point:
#   * "mass"   -- spend an ENERGY budget: deactivate the smallest-energy
#                 entries whose combined energy is <= threshold_fraction of
#                 the total. The knob is now "how much of the scene's
#                 angular energy am I willing to throw away at this check",
#                 which is bounded, physical, and independent of how peaked
#                 the distribution happens to be.
#   * "target" -- keep the top-K by energy, with K annealed toward an
#                 explicit target active count (train.py owns the schedule).
#                 Reaches K = target and STAYS there -- a real fixed point,
#                 and the direct way to implement the DOF argument (the
#                 k-space support carries ~296 independent complex DOF over
#                 the B787 bbox, ~4000 counting generously, against 110592
#                 voxels).
# "relmax" is kept as the default so every pre-2026-08-03 run reproduces.
#
# --prune-min-active is a floor honored by ALL modes including relmax: a
# prune may never take the scene below it. Pruning is irreversible, so this
# is the guard rail that makes an over-eager setting survivable.
_REPORT_PRUNE_RELMAX = (0.3, 0.1, 0.03, 0.01, 0.003, 0.001)


@torch.no_grad()
def _global_mass_below(vals_local, thresholds, device, reduce):
    """Energy fraction carried by entries strictly BELOW each threshold.

    Mirrors _global_frac_ge but weights by vals^2 (the pruning statistic is
    sqrt(sum_lm |c_lm|^2), so vals^2 IS the voxel's angular energy) instead of
    counting. Returns mass_below / mass_total, so it is scale-free -- which
    matters because the (gain, scene) gauge makes |w| itself meaningless.
    """
    sq = (vals_local.double() ** 2) if vals_local.numel() else None
    if sq is not None:
        below = (vals_local.reshape(-1, 1) < thresholds.reshape(1, -1)).double()
        mass = (below * sq.reshape(-1, 1)).sum(dim=0)
        total = sq.sum().reshape(1)
    else:
        mass = torch.zeros(thresholds.numel(), dtype=torch.float64, device=device)
        total = torch.zeros(1, dtype=torch.float64, device=device)
    if reduce:
        both = all_reduce_sum(torch.cat([mass, total]))
        mass, total = both[:-1], both[-1:]
    if float(total) <= 0.0:
        return torch.zeros_like(mass)
    return mass / total


@torch.no_grad()
def _threshold_for_mass(vals_local, hi, mass_fraction, device, reduce):
    """Bisect for the threshold whose BELOW-set carries `mass_fraction` of the
    total energy. Monotone in t, same contract as _threshold_for_fraction."""
    lo = torch.zeros((), dtype=torch.float64, device=device)
    hi = hi.double().clone()
    best, best_err = lo.clone(), float("inf")
    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        mass = float(_global_mass_below(vals_local, mid.reshape(1), device, reduce)[0])
        err = abs(mass - mass_fraction)
        if err < best_err:
            best_err, best = err, mid.clone()
        if mass > mass_fraction:   # discarding too much energy -> lower the threshold
            hi = mid
        else:
            lo = mid
    return best


@torch.no_grad()
def prune_threshold_and_report(vals_local, global_max, threshold_fraction, mode,
                               target_active=None, min_active=0, n_total=None,
                               reduce=False):
    """Shared by every scene's prune(): resolve the absolute threshold on the
    pruning statistic and build the selectivity report string.

    `vals_local` must be the statistic over CURRENTLY-ACTIVE entries only --
    already-pruned entries are held at exactly zero, so including them would
    put a spike at 0 that drags every quantile and every mass fraction.
    `n_total` is the global active count (defaults to vals_local.numel()).

    mode="relmax" -- threshold = threshold_fraction * global_max (legacy, the
                     ratchet; see the module comment before choosing it).
    mode="mass"   -- threshold = the value whose below-set carries
                     threshold_fraction of the total angular energy.
    mode="target" -- threshold = the value keeping the top `target_active`
                     entries. threshold_fraction is unused.

    `min_active` is the --prune-min-active floor, honored by every mode: if
    the chosen threshold would leave fewer than that many entries active, it
    is replaced by the top-`min_active` threshold. Resolved through the same
    globally-reduced bisection as mode="target" (NOT a local topk) so the
    single-GPU and voxel-sharded paths make bit-identical decisions --
    scripts/validate_distributed_scene.py gates this.

    With reduce=True (sharded scene only) this contains collectives, so EVERY
    rank must call it -- including ranks whose local active set is empty.
    """
    device = global_max.device
    if n_total is None:
        n_total = vals_local.numel()

    def _top_k_threshold(k):
        if n_total <= 0 or k >= n_total:
            return torch.zeros((), dtype=torch.float64, device=device)
        return _threshold_for_fraction(vals_local, global_max, k / n_total,
                                       n_total, device, reduce)

    if mode == "relmax":
        thresh = global_max.double() * threshold_fraction
    elif mode == "mass":
        thresh = _threshold_for_mass(vals_local, global_max, threshold_fraction,
                                     device, reduce)
    elif mode == "target":
        if target_active is None:
            raise ValueError("mode='target' requires target_active")
        thresh = _top_k_threshold(target_active)
    else:
        raise ValueError(f"mode must be 'relmax', 'mass' or 'target', got {mode!r}")

    if min_active > 0:
        n_keep = float(_global_frac_ge(vals_local, thresh.reshape(1), n_total,
                                       device, reduce)[0]) * n_total
        if n_keep < min_active:
            thresh = _top_k_threshold(min(min_active, n_total))

    # Report BOTH curves: what each candidate relmax would keep, and how much
    # energy this check is actually spending. One run then characterizes the
    # whole knob, the same trick that made the grow threshold scannable.
    keep = _global_frac_ge(vals_local, torch.tensor(_REPORT_PRUNE_RELMAX, dtype=torch.float64,
                                                    device=device) * global_max.double(),
                           n_total, device, reduce)
    spent = float(_global_mass_below(vals_local, thresh.reshape(1), device, reduce)[0])
    report = ("prune selectivity (relmax -> frac kept): "
              + ", ".join(f"{p:g}->{float(f):.3f}" for p, f in zip(_REPORT_PRUNE_RELMAX, keep))
              + f" | energy discarded this check: {spent:.4%}")
    return thresh.to(global_max.dtype), report


class VoxelGridScene(nn.Module):
    def __init__(self, granularity, extent, device, init_scale=0.1):
        super().__init__()
        self.granularity = granularity
        self.extent = extent
        self.w_re = nn.Parameter(init_scale * torch.randn(granularity, granularity, granularity, device=device))
        self.w_im = nn.Parameter(init_scale * torch.randn(granularity, granularity, granularity, device=device))
        self.register_buffer(
            "active_mask", torch.ones(granularity, granularity, granularity, dtype=torch.bool, device=device)
        )
        self.register_buffer("grid_positions", generate_dynamic_grid(granularity, extent, device, jitter=False))

    def active_scatterers(self):
        """Returns (positions[N,3], weights[N]) for currently-active voxels.
        N shrinks as prune() deactivates more voxels; gradients still flow
        correctly to the surviving parameters through the boolean mask.
        """
        weights = torch.complex(self.w_re, self.w_im)
        mask_flat = self.active_mask.reshape(-1)
        pos_flat = self.grid_positions.reshape(-1, 3)
        weights_flat = weights.reshape(-1)
        return pos_flat[mask_flat], weights_flat[mask_flat]

    @torch.no_grad()
    def prune(self, threshold_fraction=0.01, mode="relmax", target_active=None,
              min_active=0):
        """Deactivate low-magnitude voxels; see prune_threshold_and_report for
        what `mode` means and why "relmax" (the default, kept for
        reproducibility) is a ratchet. Irreversible (matches Plenoxels'
        pruning: once deactivated, a voxel receives no further gradient via
        active_scatterers() and cannot regrow), so prefer a mode with a fixed
        point and set --prune-min-active as a guard rail.

        Returns (n_active, n_total, report).
        """
        mag = torch.complex(self.w_re, self.w_im).abs()
        vals = mag[self.active_mask]
        if vals.numel() == 0:
            return 0, int(self.active_mask.numel()), ""
        thresh, report = prune_threshold_and_report(
            vals, vals.max(), threshold_fraction, mode, target_active=target_active,
            min_active=min_active)
        self.active_mask &= (mag >= thresh)
        self.w_re[~self.active_mask] = 0.0
        self.w_im[~self.active_mask] = 0.0
        return int(self.active_mask.sum().item()), int(self.active_mask.numel()), report


class SHVoxelGridScene(nn.Module):
    """Plenoxel-style scene representation with per-voxel ANGULAR
    reflectivity and ADAPTIVE per-voxel SH order: instead of one isotropic
    complex scalar per voxel (VoxelGridScene), each voxel stores real
    spherical-harmonic coefficients (up to max_degree) for its real and
    imaginary parts, and the complex reflectivity actually handed to the
    forward operator is the SH basis evaluated at the CURRENT viewpoint's
    (theta, phi) -- so, unlike VoxelGridScene, the returned weights are
    viewpoint-dependent and must be recomputed once per viewpoint (not
    once per epoch); callers must call active_scatterers(dtheta, dphi)
    inside the per-viewpoint loop.

    Coefficients for ALL degrees up to max_degree are allocated (and
    persisted in the checkpoint) from construction, but each voxel starts
    with only its bottom `init_degree` degree-block "unlocked" (nonzero,
    receiving gradient); everything above that is masked to exactly zero
    contribution AND initialized to exactly zero, so growing a voxel's
    order later is seamless -- it starts contributing 0 the instant it
    unlocks and trains up from there, no discontinuity in the loss.

    grow() is this class's answer to Gaussian-Splatting-style adaptive
    density control: since voxel POSITIONS here are a fixed regular grid
    (never split/cloned), "this region has a gradient signal too strong
    for its current capacity" is resolved by unlocking the next SH degree
    block for that voxel instead of adding new primitives. See grow()'s
    docstring for the exact criterion.

    Deliberately a sibling of VoxelGridScene, not a subclass, so
    isinstance(model, VoxelGridScene) checks elsewhere don't accidentally
    match this class too.
    """
    def __init__(self, granularity, extent, device, max_degree=10, init_degree=0, init_scale=0.1):
        super().__init__()
        if not (0 <= init_degree <= max_degree):
            raise ValueError(f"init_degree ({init_degree}) must be in [0, max_degree={max_degree}]")
        self.granularity = granularity
        self.extent = extent
        self.max_degree = max_degree
        n_basis = num_sh_basis(max_degree)

        self.register_buffer("basis_degree", basis_degree_index(max_degree, device=device))

        self.w_re = nn.Parameter(
            init_scale * torch.randn(granularity, granularity, granularity, n_basis, device=device)
        )
        self.w_im = nn.Parameter(
            init_scale * torch.randn(granularity, granularity, granularity, n_basis, device=device)
        )
        with torch.no_grad():
            locked = self.basis_degree > init_degree
            self.w_re[..., locked] = 0.0
            self.w_im[..., locked] = 0.0

        self.register_buffer(
            "order",
            torch.full((granularity, granularity, granularity), init_degree, dtype=torch.int64, device=device),
        )
        self.register_buffer(
            "active_mask", torch.ones(granularity, granularity, granularity, dtype=torch.bool, device=device)
        )
        self.register_buffer("grid_positions", generate_dynamic_grid(granularity, extent, device, jitter=False))
        self.register_buffer(
            "grad_accum", torch.zeros(granularity, granularity, granularity, device=device)
        )
        self.register_buffer("grad_accum_count", torch.tensor(0, dtype=torch.int64, device=device))

    def active_scatterers(self, dtheta, dphi):
        """dtheta/dphi: the same raw per-viewpoint tensors passed to
        forward_operator.get_array_pos -- extracts the scalar viewpoint
        angle the same way get_array_pos does internally, so callers can
        pass dtheta_tensor.to(device), dphi_tensor.to(device) unchanged.
        """
        theta = dtheta.squeeze(0)[0]
        phi = dphi.squeeze(0)[0]
        basis = real_sh_basis(theta, phi, self.max_degree)  # [n_basis]

        order_mask = (self.basis_degree.view(1, 1, 1, -1) <= self.order.unsqueeze(-1)).to(self.w_re.dtype)
        w_re_eff = torch.einsum('xyzc,c->xyz', self.w_re * order_mask, basis)
        w_im_eff = torch.einsum('xyzc,c->xyz', self.w_im * order_mask, basis)
        weights = torch.complex(w_re_eff, w_im_eff)

        mask_flat = self.active_mask.reshape(-1)
        pos_flat = self.grid_positions.reshape(-1, 3)
        weights_flat = weights.reshape(-1)
        return pos_flat[mask_flat], weights_flat[mask_flat]

    @torch.no_grad()
    def prune(self, threshold_fraction=0.01, criterion="energy", mode="relmax",
              target_active=None, min_active=0):
        """Deactivate voxels whose reflectivity clears no threshold.

        `criterion` is WHAT is measured, `mode` is HOW the cut is chosen --
        see prune_threshold_and_report for the modes and for why the legacy
        "relmax" default is a ratchet that killed every Round 4 arm that used
        it.

        criterion="energy" (default): total angular energy over the voxel's
        UNLOCKED bands, sqrt(sum_lm |c_lm|^2). This is the rotation-invariant
        measure every geometry metric already uses
        (scripts/eval_scene_geometry.py), so a voxel survives iff it carries
        energy from SOME direction. Locked (order > self.order) coefficients
        are excluded -- they are held at exactly zero and contribute nothing
        to any render, so including them would only dilute the comparison.

        criterion="dc": the old l=0-only test (direction-averaged
        reflectivity). Kept for reproducing pre-2026-07-30 runs. It prunes a
        voxel whose DC is small even when its l>=1 bands are strong, i.e. it
        deletes exactly the specular scatterers a PEC target is made of --
        which is why it is no longer the default.

        Irreversible, as in Plenoxels: a deactivated voxel receives no
        further gradient via active_scatterers() and cannot come back. Prune
        gently and late (see --prune-start-epoch) rather than aggressively
        early; accuracy cannot recover from an over-eager prune.
        """
        if criterion == "dc":
            mag = torch.complex(self.w_re[..., 0], self.w_im[..., 0]).abs()
        elif criterion == "energy":
            unlocked = (self.basis_degree.view(1, 1, 1, -1) <= self.order.unsqueeze(-1))
            sq = (self.w_re ** 2 + self.w_im ** 2) * unlocked.to(self.w_re.dtype)
            mag = sq.sum(dim=-1).sqrt()
        else:
            raise ValueError(f"criterion must be 'energy' or 'dc', got {criterion!r}")
        vals = mag[self.active_mask]
        if vals.numel() == 0:
            return 0, int(self.active_mask.numel()), ""
        thresh, report = prune_threshold_and_report(
            vals, vals.max(), threshold_fraction, mode, target_active=target_active,
            min_active=min_active)
        self.active_mask &= (mag >= thresh)
        self.w_re[~self.active_mask] = 0.0
        self.w_im[~self.active_mask] = 0.0
        return int(self.active_mask.sum().item()), int(self.active_mask.numel()), report

    @torch.no_grad()
    def accumulate_grad_stats(self):
        """Call once per epoch (or however often the caller backprops),
        after gradients have accumulated into w_re.grad/w_im.grad but
        before optimizer.zero_grad() clears them -- typically right where
        grad-norm clipping already reads .grad. Masked-out (locked)
        coefficients receive exactly zero gradient by construction (see
        class docstring), so summing over ALL basis slots here is
        equivalent to summing over only the currently-active ones -- no
        separate masking needed.
        """
        if self.w_re.grad is None or self.w_im.grad is None:
            return
        per_voxel_grad_sq = (self.w_re.grad ** 2 + self.w_im.grad ** 2).sum(dim=-1)
        self.grad_accum += per_voxel_grad_sq.sqrt()
        self.grad_accum_count += 1

    @torch.no_grad()
    def grow_angular(self, tail_ratio_threshold=0.05):
        """Angular-derivative growth (Plenoxel-spirit "refine where it is
        needed"), the criterion-driven alternative to grow()'s epoch ladder.

        The SH argument here IS the radar viewpoint direction (see
        active_scatterers), so a voxel whose response varies rapidly as the
        radar moves is one whose expansion is straining against its current
        order cap.  Detect that spectrally: the share of the voxel's angular
        energy sitting in its HIGHEST UNLOCKED degree band,

            rho = sum_{l = order} |c_lm|^2  /  sum_{l <= order} |c_lm|^2

        and unlock the next block wherever rho >= tail_ratio_threshold.

        Why plain |c|^2 and not the l(l+1) Laplace-Beltrami weight used by
        --sh-smooth-weight: within one band l(l+1) is a constant, so it
        cancels out of the numerator and only reweights the denominator's
        band mix.  At order=1 that kills the l=0 term (0*(0+1)=0) and rho is
        identically 1 for every voxel with any angular structure -- every
        voxel would grow on the first check.  Plain energy keeps the DC band
        as the reference, so a mostly-isotropic voxel scores low (verified:
        E0=1,E1=0.02 -> 0.020 plain vs 1.000 weighted).

        Self-limiting: a freshly unlocked band starts at zero coefficients,
        so rho drops to 0 right after growing and only climbs back if the
        data actually pushes energy into that band.  Unlike grow(), this
        reads no gradient accumulator and does not reset it.

        Returns (n_grown, n_active).
        """
        n_active = int(self.active_mask.sum().item())
        deg = self.basis_degree.view(1, 1, 1, -1)
        order_e = self.order.unsqueeze(-1)
        sq = self.w_re ** 2 + self.w_im ** 2
        e_keep = (sq * (deg <= order_e).to(sq.dtype)).sum(dim=-1)
        e_top = (sq * (deg == order_e).to(sq.dtype)).sum(dim=-1)
        ratio = e_top / e_keep.clamp_min(torch.finfo(sq.dtype).tiny)
        eligible = self.active_mask & (self.order < self.max_degree) & (e_keep > 0)
        grow_mask = eligible & (ratio >= tail_ratio_threshold)
        self.order[grow_mask] = self.order[grow_mask] + 1
        return int(grow_mask.sum().item()), n_active, _tail_ratio_report(ratio[eligible])

    @torch.no_grad()
    def grow(self, threshold_fraction=0.1, mode="relmax"):
        """The 'increase SH order instead of splitting' rule: among
        active, not-yet-maxed-out voxels, unlock the next SH degree block
        (order += 1) for any voxel whose gradient magnitude -- averaged
        over the accumulation window since the last grow()/construction
        -- clears a threshold set by `mode`:

          "relmax"   (legacy) threshold_fraction * the max among eligible
                     voxels, the same relative convention as prune().
          "quantile" grow the top threshold_fraction FRACTION of eligible
                     voxels. Prefer this: the relmax test is only selective
                     when avg_grad has wide dynamic range, and on the B787
                     grid it does not -- see the module-level comment.

        This is the direct analogue of Gaussian-Splatting densification
        (split/clone where position gradients are large), but since this
        grid's voxel positions never move, the response to 'gradient too
        high here' is added ANGULAR capacity, not a new primitive.
        Always resets the accumulator afterward (whether or not anything
        grew), so the next window's decision isn't diluted by stale
        history. Returns (n_grown, n_active, report).
        """
        n_active = int(self.active_mask.sum().item())
        if self.grad_accum_count.item() == 0:
            return 0, n_active, ""

        avg_grad = self.grad_accum / self.grad_accum_count.clamp_min(1)
        eligible = self.active_mask & (self.order < self.max_degree)

        n_grown = 0
        report = ""
        if eligible.any():
            vals = avg_grad[eligible]
            thresh, report = grow_threshold_and_report(
                vals, vals.max(), threshold_fraction, mode)
            grow_mask = eligible & (avg_grad >= thresh)
            self.order[grow_mask] = self.order[grow_mask] + 1
            n_grown = int(grow_mask.sum().item())

        self.grad_accum.zero_()
        self.grad_accum_count.zero_()
        return n_grown, n_active, report


class AdaptivePointSHScene(nn.Module):
    """Flat point-list scene with bounded continuous positions and SH weights.

    Anchors are fixed reference positions.  Learned offsets live in
    ``delta_raw`` and are mapped through ``tanh`` so every point remains
    inside its anchor cell:

        position = anchor + cell_half * tanh(delta_raw)

    The SH coefficient/order/prune/grow mechanics intentionally mirror
    SHVoxelGridScene, but all tensors are flat ``[K, ...]`` instead of a
    regular ``[G,G,G, ...]`` grid.
    """

    def __init__(self, anchors, cell_half, device,
                 max_degree=10, init_degree=0, init_scale=0.0,
                 learn_positions=True, capacity=None,
                 enforce_support_bounds=False, compact_sh_eval=False):
        super().__init__()
        if not (0 <= init_degree <= max_degree):
            raise ValueError(f"init_degree ({init_degree}) must be in [0, max_degree={max_degree}]")

        anchors = torch.as_tensor(anchors, dtype=torch.float32, device=device)
        if anchors.ndim != 2 or anchors.shape[-1] != 3:
            raise ValueError("anchors must have shape [K, 3]")
        k_init = anchors.shape[0]

        cell_half = torch.as_tensor(cell_half, dtype=torch.float32, device=device)
        if cell_half.ndim == 0:
            cell_half = cell_half.expand(k_init)
        if cell_half.ndim == 1:
            cell_half = cell_half[:, None]
        if cell_half.shape != (k_init, 1):
            raise ValueError("cell_half must be a scalar, [K], or [K,1]")
        # materialize: the scalar path leaves a stride-0 expanded view, which
        # load_state_dict cannot copy_ into when resuming from a checkpoint
        cell_half = cell_half.contiguous()

        # Freeze the physical domain *before* capacity padding or later
        # recenter/split events.  The opt-in v2 controller clamps rendered
        # coordinates to these original cell-union bounds, so a cell recentered
        # near an exterior face cannot grow the scene support merely by being
        # subdivided.  The legacy default deliberately leaves this off: old
        # point-scene checkpoints retain their historical coordinate map.
        support_min = (anchors - cell_half).amin(dim=0)
        support_max = (anchors + cell_half).amax(dim=0)

        # Fixed-capacity slot pool: split() activates spare slots instead of
        # resizing parameters, so tensor/optimizer-state/checkpoint shapes stay
        # constant for the whole run. Inactive slots are excluded from
        # rendering (active_mask), receive exactly zero gradient, and therefore
        # keep zero Adam moments -- activating one later is equivalent to
        # adding a fresh parameter with reset optimizer state.
        capacity = k_init if capacity is None else int(capacity)
        if capacity < k_init:
            raise ValueError(f"capacity ({capacity}) must be >= number of anchors ({k_init})")
        if capacity > k_init:
            pad = capacity - k_init
            anchors = torch.cat([anchors, torch.zeros(pad, 3, dtype=anchors.dtype, device=device)])
            cell_half = torch.cat([cell_half, torch.zeros(pad, 1, dtype=cell_half.dtype, device=device)])
        k_points = capacity

        self.max_degree = max_degree
        n_basis = num_sh_basis(max_degree)

        self.register_buffer("anchors", anchors)
        self.register_buffer("cell_half", cell_half)
        self.register_buffer("basis_degree", basis_degree_index(max_degree, device=device))
        self.register_buffer("support_min", support_min)
        self.register_buffer("support_max", support_max)
        self.register_buffer(
            "support_bounds_enabled",
            torch.tensor(bool(enforce_support_bounds), dtype=torch.bool, device=device),
        )
        self.register_buffer(
            "compact_sh_eval_enabled",
            torch.tensor(bool(compact_sh_eval), dtype=torch.bool, device=device),
        )
        # Keep the opt-in mode bits in Python as well as persisted buffers.
        # ``positions`` and ``active_scatterers`` run once per viewpoint; a
        # ``Tensor.item()`` there would synchronize a CUDA stream for a static
        # flag.  The buffers remain the checkpoint authority, while these
        # mirrors are refreshed after checkpoint load (or deliberately by a
        # research script through ``refresh_runtime_flags``).
        self._support_bounds_runtime_enabled = bool(enforce_support_bounds)
        self._compact_sh_eval_runtime_enabled = bool(compact_sh_eval)
        # Python-side on purpose: reading a GPU scalar every viewpoint would
        # introduce a synchronization.  It is refreshed only after a topology
        # or order mutation (and after loading a checkpoint).  The fixed
        # coefficient tensor remains the compatibility-safe allocation; this
        # optimization reduces rendering work, not checkpoint/Adam allocation.
        self._compact_eval_degree = int(init_degree) if compact_sh_eval else int(max_degree)

        self.delta_raw = nn.Parameter(torch.zeros(k_points, 3, dtype=torch.float32, device=device),
                                      requires_grad=learn_positions)
        self.w_re = nn.Parameter(init_scale * torch.randn(k_points, n_basis, device=device))
        self.w_im = nn.Parameter(init_scale * torch.randn(k_points, n_basis, device=device))
        active = torch.zeros(k_points, dtype=torch.bool, device=device)
        active[:k_init] = True
        with torch.no_grad():
            locked = self.basis_degree > init_degree
            self.w_re[:, locked] = 0.0
            self.w_im[:, locked] = 0.0
            self.w_re[~active] = 0.0
            self.w_im[~active] = 0.0

        self.register_buffer("order", torch.full((k_points,), init_degree, dtype=torch.int64, device=device))
        self.register_buffer("active_mask", active)
        self.register_buffer("grad_accum", torch.zeros(k_points, device=device))
        # ``pos_grad_accum`` is kept as the public compatibility name, but now
        # records a world-coordinate sensitivity rather than the gradient of
        # tanh-bounded raw offsets.  The raw statistic remains available only
        # as an explicit diagnostic/legacy selector.
        self.register_buffer("pos_raw_grad_accum", torch.zeros(k_points, device=device))
        self.register_buffer("pos_grad_accum", torch.zeros(k_points, device=device))
        self.register_buffer("pos_world_grad_sum", torch.zeros(k_points, 3, device=device))
        self.register_buffer("pos_world_grad_sq_accum", torch.zeros(k_points, 3, device=device))
        self.register_buffer("grad_accum_count", torch.tensor(0, dtype=torch.int64, device=device))
        self.register_buffer("split_event_count", torch.tensor(0, dtype=torch.int64, device=device))
        # v2 refinement windows are deliberately separate from the historical
        # grad_accum buffers above.  They receive *incremental, pre-clipping,
        # data-fit* gradients per training view; legacy grow()/split() keeps
        # its old optimizer-step aggregate semantics unless the new recipe is
        # explicitly selected in train.py.
        self.register_buffer("refine_spatial_sum", torch.zeros(k_points, device=device))
        self.register_buffer("refine_spatial_exposure", torch.zeros(k_points, device=device))
        self.register_buffer("refine_angular_sum", torch.zeros(k_points, device=device))
        self.register_buffer("refine_angular_exposure", torch.zeros(k_points, device=device))
        self.register_buffer("refine_event_count", torch.tensor(0, dtype=torch.int64, device=device))
        self.register_buffer("refine_last_spatial_event", torch.full(
            (k_points,), -1_000_000, dtype=torch.int64, device=device))
        self.register_buffer("refine_last_angular_event", torch.full(
            (k_points,), -1_000_000, dtype=torch.int64, device=device))
        # A zero-weight sibling needs evidence from at least one *completed*
        # refinement interval before it can itself grow/split.  This birth
        # clock is independent of the ordinary selected-point cooldown.
        self.register_buffer("refine_birth_event", torch.full(
            (k_points,), -1_000_000, dtype=torch.int64, device=device))
        self.register_buffer("level", torch.zeros(k_points, dtype=torch.int64, device=device))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Migrate only additive adaptive-statistic buffers from old checkpoints.

        All unrelated strict-loading behaviour remains PyTorch's default, so a
        legacy checkpoint can be resumed without silently accepting an
        incompatible model state.
        """
        defaults = {
            "pos_raw_grad_accum": self.pos_raw_grad_accum,
            "pos_world_grad_sum": self.pos_world_grad_sum,
            "pos_world_grad_sq_accum": self.pos_world_grad_sq_accum,
            "split_event_count": self.split_event_count,
            "support_min": self.support_min,
            "support_max": self.support_max,
            "support_bounds_enabled": self.support_bounds_enabled,
            "compact_sh_eval_enabled": self.compact_sh_eval_enabled,
            "refine_spatial_sum": self.refine_spatial_sum,
            "refine_spatial_exposure": self.refine_spatial_exposure,
            "refine_angular_sum": self.refine_angular_sum,
            "refine_angular_exposure": self.refine_angular_exposure,
            "refine_event_count": self.refine_event_count,
            "refine_last_spatial_event": self.refine_last_spatial_event,
            "refine_last_angular_event": self.refine_last_angular_event,
            "refine_birth_event": self.refine_birth_event,
        }
        for suffix, value in defaults.items():
            key = prefix + suffix
            if key not in state_dict:
                # Event clocks use a negative sentinel so a legacy checkpoint
                # can collect fresh evidence immediately; additive sums/counters
                # start at zero as before.
                state_dict[key] = (value.clone() if suffix in {
                    "refine_last_spatial_event", "refine_last_angular_event",
                    "refine_birth_event", "support_min", "support_max",
                    "support_bounds_enabled", "compact_sh_eval_enabled",
                } else torch.zeros_like(value))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        self.refresh_runtime_flags()
        self.refresh_compact_sh_eval_cap()

    @torch.no_grad()
    def refresh_runtime_flags(self):
        """Refresh infrequently-read Python mirrors of persisted mode bits.

        Checkpoint loading invokes this automatically.  A research script that
        intentionally edits either persisted flag under ``torch.no_grad()``
        must call this method before rendering; regular topology operations do
        not alter the flags.
        """
        self._support_bounds_runtime_enabled = bool(self.support_bounds_enabled.item())
        self._compact_sh_eval_runtime_enabled = bool(self.compact_sh_eval_enabled.item())

    @property
    def grid_positions(self):
        """Compatibility source for no-grad backprojection/grid logging paths."""
        return self.positions().detach()

    def positions(self):
        position = self.anchors + self.cell_half * torch.tanh(self.delta_raw)
        if self._support_bounds_runtime_enabled:
            position = torch.maximum(torch.minimum(position, self.support_max), self.support_min)
        return position

    @torch.no_grad()
    def refresh_compact_sh_eval_cap(self):
        """Refresh the opt-in global SH render cap after external order edits.

        Internal lifecycle methods call this automatically.  It exists for
        research scripts that intentionally edit ``order`` under
        ``torch.no_grad()``; the default (disabled) branch always evaluates
        the historical full allocated SH basis.
        """
        if self._compact_sh_eval_runtime_enabled and bool(self.active_mask.any()):
            self._compact_eval_degree = int(self.order[self.active_mask].max().item())
        else:
            self._compact_eval_degree = int(self.max_degree)
        return self._compact_eval_degree

    def allocated_parameter_count(self):
        """Actual coefficient/position allocation, unlike active DOF count."""
        return sum(parameter.numel() for parameter in self.parameters())

    def allocated_parameter_bytes(self):
        """Actual parameter bytes; optimizer state is additional allocation."""
        return sum(parameter.numel() * parameter.element_size() for parameter in self.parameters())

    def active_parameter_count(self):
        """Return active scalar degrees of freedom, distinct from allocation."""
        active_order = self.order[self.active_mask]
        n_coeff = int((2 * (active_order + 1).square()).sum().item())
        n_pos = 3 * int(self.active_mask.sum().item()) if self.delta_raw.requires_grad else 0
        return n_coeff + n_pos

    @staticmethod
    @torch.no_grad()
    def _clear_optimizer_rows(optimizer, parameter, indices):
        """Clear row-wise Adam state while preserving its global step counter."""
        if optimizer is None or indices is None:
            return
        if indices.dtype == torch.bool:
            indices = indices.nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            return
        state = optimizer.state.get(parameter)
        if not state:
            return
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(key)
            if torch.is_tensor(value) and value.shape == parameter.shape:
                value[indices.to(value.device)] = 0

    @torch.no_grad()
    def _tombstone_slots(self, indices, optimizer=None):
        """Canonical all-zero state for inactive/recycled slots."""
        if indices.dtype == torch.bool:
            indices = indices.nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            return
        self.active_mask[indices] = False
        self.anchors[indices] = 0
        self.cell_half[indices] = 0
        self.delta_raw[indices] = 0
        self.w_re[indices] = 0
        self.w_im[indices] = 0
        self.order[indices] = 0
        self.level[indices] = 0
        self.grad_accum[indices] = 0
        self.pos_raw_grad_accum[indices] = 0
        self.pos_grad_accum[indices] = 0
        self.pos_world_grad_sum[indices] = 0
        self.pos_world_grad_sq_accum[indices] = 0
        self.refine_spatial_sum[indices] = 0
        self.refine_spatial_exposure[indices] = 0
        self.refine_angular_sum[indices] = 0
        self.refine_angular_exposure[indices] = 0
        self.refine_last_spatial_event[indices] = -1_000_000
        self.refine_last_angular_event[indices] = -1_000_000
        self.refine_birth_event[indices] = -1_000_000
        self._clear_optimizer_rows(optimizer, self.delta_raw, indices)
        self._clear_optimizer_rows(optimizer, self.w_re, indices)
        self._clear_optimizer_rows(optimizer, self.w_im, indices)

    @torch.no_grad()
    def sanitize_inactive_slots(self, optimizer=None):
        """Clear stale rows defensively, especially following legacy resume."""
        indices = (~self.active_mask).nonzero(as_tuple=True)[0]
        self._tombstone_slots(indices, optimizer=optimizer)
        self.refresh_compact_sh_eval_cap()
        return int(indices.numel())

    def active_scatterers(self, dtheta, dphi, probe_next_band=False):
        """Return active coherent scatterers for one viewpoint.

        ``probe_next_band`` is a non-mutating capacity diagnostic: it exposes
        exactly one currently locked SH degree (where available), whose
        coefficients are zero at a well-formed unlock.  The rendered value is
        therefore unchanged at probe time, but autograd can measure whether
        the training data-fit loss is sensitive to that next band.  The
        default is byte-for-byte the historical order mask.
        """
        theta = dtheta.squeeze(0)[0]
        phi = dphi.squeeze(0)[0]
        if not self._compact_sh_eval_runtime_enabled:
            # Preserve the historical implementation literally in the default
            # branch.  Old recipes therefore retain their prior numerical and
            # allocation behavior exactly.
            basis = real_sh_basis(theta, phi, self.max_degree)
            order = self.order
            if probe_next_band:
                order = torch.minimum(order + 1, torch.full_like(order, self.max_degree))
            order_mask = (self.basis_degree.view(1, -1) <= order[:, None]).to(self.w_re.dtype)
            w_re_eff = torch.einsum('kb,b->k', self.w_re * order_mask, basis)
            w_im_eff = torch.einsum('kb,b->k', self.w_im * order_mask, basis)
            weights = torch.complex(w_re_eff, w_im_eff)
            return self.positions()[self.active_mask], weights[self.active_mask]

        active = self.active_mask
        eval_degree = self._compact_eval_degree
        if probe_next_band:
            eval_degree = min(eval_degree + 1, self.max_degree)
        n_basis = num_sh_basis(eval_degree)
        basis = real_sh_basis(theta, phi, eval_degree)
        order = self.order[active]
        if probe_next_band:
            order = torch.minimum(order + 1, torch.full_like(order, self.max_degree))
        basis_degree = self.basis_degree[:n_basis]
        order_mask = (basis_degree.view(1, -1) <= order[:, None]).to(self.w_re.dtype)
        w_re_eff = torch.einsum('kb,b->k', self.w_re[active, :n_basis] * order_mask, basis)
        w_im_eff = torch.einsum('kb,b->k', self.w_im[active, :n_basis] * order_mask, basis)
        return self.positions()[active], torch.complex(w_re_eff, w_im_eff)

    @torch.no_grad()
    def prune(self, threshold_fraction=0.01, criterion="energy", mode="relmax",
              target_active=None, min_active=0, optimizer=None):
        """Flat-point analogue of SHVoxelGridScene.prune -- see that
        docstring for why "energy" (total angular energy over unlocked
        bands) is the default criterion and "dc" is legacy, and
        prune_threshold_and_report for what `mode` does."""
        if criterion == "dc":
            mag = torch.complex(self.w_re[:, 0], self.w_im[:, 0]).abs()
        elif criterion == "energy":
            unlocked = (self.basis_degree.view(1, -1) <= self.order[:, None])
            sq = (self.w_re ** 2 + self.w_im ** 2) * unlocked.to(self.w_re.dtype)
            mag = sq.sum(dim=-1).sqrt()
        else:
            raise ValueError(f"criterion must be 'energy' or 'dc', got {criterion!r}")
        vals = mag[self.active_mask]
        if vals.numel() == 0:
            return 0, int(self.active_mask.numel()), ""
        thresh, report = prune_threshold_and_report(
            vals, vals.max(), threshold_fraction, mode, target_active=target_active,
            min_active=min_active)
        self.active_mask &= (mag >= thresh)
        # When a slot can later become a sibling, matching its parameter zeros
        # without clearing Adam's row state is not a fresh identity.
        self._tombstone_slots((~self.active_mask).nonzero(as_tuple=True)[0], optimizer=optimizer)
        self.refresh_compact_sh_eval_cap()
        return int(self.active_mask.sum().item()), int(self.active_mask.numel()), report

    @torch.no_grad()
    def accumulate_grad_stats(self):
        if self.w_re.grad is None or self.w_im.grad is None:
            return
        per_point_grad_sq = (self.w_re.grad ** 2 + self.w_im.grad ** 2).sum(dim=-1)
        self.grad_accum += per_point_grad_sq.sqrt()
        if self.delta_raw.grad is not None:
            active_f = self.active_mask[:, None].to(self.delta_raw.grad.dtype)
            self.pos_raw_grad_accum += (
                self.delta_raw.grad.norm(dim=-1) * self.active_mask.to(self.delta_raw.grad.dtype))
            # x = anchor + h*tanh(delta_raw); convert dL/ddelta to a
            # physical dL/dx score.  Cap the inverse near saturation, where
            # the raw coordinate otherwise creates numerical infinities.
            tanh_delta = torch.tanh(self.delta_raw.detach())
            jac = self.cell_half * (1.0 - tanh_delta.square())
            jac_floor = self.cell_half.abs() * 1.0e-6
            jac_floor = jac_floor.clamp_min(torch.finfo(jac.dtype).tiny)
            world_grad = (self.delta_raw.grad * active_f) / torch.maximum(jac.abs(), jac_floor)
            self.pos_grad_accum += world_grad.norm(dim=-1)
            self.pos_world_grad_sum += world_grad
            self.pos_world_grad_sq_accum += world_grad.square()
        self.grad_accum_count += 1

    @torch.no_grad()
    def accumulate_refinement_data_stats(self, data_delta_raw_grad,
                                         next_band_w_re_grad=None,
                                         next_band_w_im_grad=None):
        """Accumulate v2 refinement statistics from *one* data-fit gradient.

        The caller supplies the incremental gradient caused by the current
        training view before clipping or adding a prior.  This avoids the old
        statistic's dependence on optimizer accumulation and makes a window a
        mean over represented training views rather than a norm of their sum.
        ``next_band_*`` must come from a separate zero-coefficient probe with
        :meth:`active_scatterers(..., probe_next_band=True)`; masked bands
        otherwise have identically zero autograd gradients.
        """
        active = self.active_mask

        if data_delta_raw_grad is not None:
            if data_delta_raw_grad.shape != self.delta_raw.shape:
                raise ValueError("data_delta_raw_grad must match delta_raw")
            # dL/dx is the physical-coordinate sensitivity.  We score the
            # displacement that fits in this cell, h*dL/dx, so scores remain
            # comparable through subdivision.  The raw-coordinate gradient
            # alone would shrink merely because tanh is saturated or h changes.
            tanh_delta = torch.tanh(self.delta_raw.detach())
            jac = self.cell_half * (1.0 - tanh_delta.square())
            # Mask and clean before multiplying: NaN * 0 is NaN, so applying
            # the active mask after an invalid derivative would permanently
            # poison an otherwise unrelated refinement window.
            finite_delta = torch.isfinite(data_delta_raw_grad).all(dim=-1)
            spatial_live = active & finite_delta
            clean_delta = torch.where(
                finite_delta[:, None], data_delta_raw_grad.detach(),
                torch.zeros_like(data_delta_raw_grad),
            )
            # Apply the geometric scale before clamping the denominator.  A
            # zero-sized inactive slot must not turn an otherwise finite value
            # into inf through a pre-scale floor.
            jac_floor = (self.cell_half.abs() * 1.0e-6).clamp_min(
                torch.finfo(jac.dtype).tiny)
            world_grad = clean_delta / torch.maximum(jac.abs(), jac_floor)
            local_displacement_grad = world_grad * self.cell_half
            spatial_raw = local_displacement_grad.norm(dim=-1)
            # Finite input gradients can still overflow while converting to a
            # physical score (for example an extreme derivative in float32).
            # Select the accepted value before accumulation: ``inf * 0`` is
            # NaN, and one NaN would poison the entire refinement window.
            spatial_accepted = spatial_live & torch.isfinite(spatial_raw)
            spatial = torch.where(
                spatial_accepted, spatial_raw,
                torch.zeros_like(self.refine_spatial_sum),
            )
            spatial_live_f = spatial_accepted.to(self.w_re.dtype)
            self.refine_spatial_sum += spatial
            self.refine_spatial_exposure += spatial_live_f

        if next_band_w_re_grad is None and next_band_w_im_grad is None:
            return
        if next_band_w_re_grad is None or next_band_w_im_grad is None:
            raise ValueError("next-band real and imaginary gradients must be supplied together")
        if (next_band_w_re_grad.shape != self.w_re.shape
                or next_band_w_im_grad.shape != self.w_im.shape):
            raise ValueError("next-band gradients must match SH coefficient tensors")
        next_order = self.order + 1
        next_band = self.basis_degree.view(1, -1) == next_order[:, None]
        n_band = next_band.sum(dim=-1).clamp_min(1).to(self.w_re.dtype)
        next_re = next_band_w_re_grad.detach()
        next_im = next_band_w_im_grad.detach()
        finite_band = torch.where(
            next_band,
            torch.isfinite(next_re) & torch.isfinite(next_im),
            torch.ones_like(next_band, dtype=torch.bool),
        ).all(dim=-1)
        safe_re = torch.where(next_band & torch.isfinite(next_re), next_re, torch.zeros_like(next_re))
        safe_im = torch.where(next_band & torch.isfinite(next_im), next_im, torch.zeros_like(next_im))
        score_sq = ((safe_re.square() + safe_im.square())
                    * next_band.to(self.w_re.dtype)).sum(dim=-1) / n_band
        angular = score_sq.clamp_min(0).sqrt()
        eligible = (active & (self.order < self.max_degree) & finite_band
                    & torch.isfinite(angular))
        eligible_f = eligible.to(self.w_re.dtype)
        # Do not multiply a rejected overflowing score by zero: IEEE gives
        # ``inf * 0 == nan``.  ``where`` makes rejected observations exact
        # zeros before the persistent accumulator sees them.
        self.refine_angular_sum += torch.where(
            eligible, angular, torch.zeros_like(self.refine_angular_sum))
        self.refine_angular_exposure += eligible_f

    @torch.no_grad()
    def reset_refinement_data_stats(self):
        """Clear only the v2 data-fit window after both decisions are made."""
        self.refine_spatial_sum.zero_()
        self.refine_spatial_exposure.zero_()
        self.refine_angular_sum.zero_()
        self.refine_angular_exposure.zero_()

    @torch.no_grad()
    def refinement_snapshot(self, max_level, min_spatial_exposure,
                            min_angular_exposure, spatial_floor=0.0,
                            angular_floor=0.0, cooldown_events=0,
                            child_maturity_events=0):
        """Freeze the v2 spatial/angular decision inputs for one event.

        ``cooldown_events`` counts intervening refinement events.  A value of
        one therefore prevents a point selected at event *e* from being
        selected again at event *e+1*; it may return at *e+2* after fresh
        evidence.  The returned tensors are clones so applying either action
        cannot starve the other action's decision.
        """
        if max_level < 0:
            raise ValueError("max_level must be >= 0")
        if min_spatial_exposure < 1 or min_angular_exposure < 1:
            raise ValueError("minimum refinement exposure must be >= 1")
        if spatial_floor < 0 or angular_floor < 0:
            raise ValueError("refinement score floors must be >= 0")
        if cooldown_events < 0:
            raise ValueError("cooldown_events must be >= 0")
        if child_maturity_events < 0:
            raise ValueError("child_maturity_events must be >= 0")

        spatial_score = self.refine_spatial_sum / self.refine_spatial_exposure.clamp_min(1)
        angular_score = self.refine_angular_sum / self.refine_angular_exposure.clamp_min(1)
        event = int(self.refine_event_count.item())
        spatial_cool = (event - self.refine_last_spatial_event) > cooldown_events
        angular_cool = (event - self.refine_last_angular_event) > cooldown_events
        mature = (event - self.refine_birth_event) > child_maturity_events
        spatial_eligible = (
            self.active_mask
            & (self.level < max_level)
            & (self.cell_half[:, 0] > 0)
            & (self.refine_spatial_exposure >= min_spatial_exposure)
            & torch.isfinite(spatial_score)
            & (spatial_score > spatial_floor)
            & spatial_cool
            & mature
        )
        angular_eligible = (
            self.active_mask
            & (self.order < self.max_degree)
            & (self.refine_angular_exposure >= min_angular_exposure)
            & torch.isfinite(angular_score)
            & (angular_score > angular_floor)
            & angular_cool
            & mature
        )
        return {
            "event": event,
            "spatial_score": spatial_score.detach().clone(),
            "angular_score": angular_score.detach().clone(),
            "spatial_eligible": spatial_eligible.detach().clone(),
            "angular_eligible": angular_eligible.detach().clone(),
            "spatial_exposure": self.refine_spatial_exposure.detach().clone(),
            "angular_exposure": self.refine_angular_exposure.detach().clone(),
        }

    @staticmethod
    @torch.no_grad()
    def _select_refinement_indices(scores, eligible, fraction):
        """Deterministically select a bounded positive-score top fraction."""
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("refinement fraction must be in [0, 1]")
        idx = eligible.nonzero(as_tuple=True)[0]
        if fraction == 0.0 or idx.numel() == 0:
            return idx[:0]
        # ``eligible`` already excludes non-finite/non-positive values.  Stable
        # sorting makes ties reproducible by original point index.
        ranking = torch.argsort(scores[idx], descending=True, stable=True)
        count = min(int(math.ceil(fraction * idx.numel())), int(idx.numel()))
        return idx[ranking[:count]]

    @staticmethod
    @torch.no_grad()
    def _clear_optimizer_band_rows(optimizer, parameter, row_idx, band_mask):
        """Clear Adam moments only for a newly unlocked SH band."""
        if optimizer is None or row_idx.numel() == 0:
            return
        state = optimizer.state.get(parameter)
        if not state:
            return
        col_idx = band_mask.nonzero(as_tuple=True)[0]
        if col_idx.numel() == 0:
            return
        rows = row_idx.to(parameter.device)
        cols = col_idx.to(parameter.device)
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(key)
            if torch.is_tensor(value) and value.shape == parameter.shape:
                value[rows[:, None], cols[None, :]] = 0

    @torch.no_grad()
    def unlock_next_bands(self, indices, optimizer=None):
        """Unlock one zero-initialized SH degree while preserving older bands."""
        if indices.dtype == torch.bool:
            indices = indices.nonzero(as_tuple=True)[0]
        indices = indices.to(self.order.device)
        if indices.numel() == 0:
            return 0
        valid = self.active_mask[indices] & (self.order[indices] < self.max_degree)
        indices = indices[valid]
        if indices.numel() == 0:
            return 0
        next_order = self.order[indices] + 1
        for degree in torch.unique(next_order, sorted=True).tolist():
            rows = indices[next_order == degree]
            band = self.basis_degree == int(degree)
            cols = band.nonzero(as_tuple=True)[0]
            # A locked band must begin exactly at zero; clearing its optimizer
            # columns prevents stale momentum from a prior slot identity or
            # interrupted attempted unlock from leaking into the new capacity.
            self.w_re[rows[:, None], cols[None, :]] = 0
            self.w_im[rows[:, None], cols[None, :]] = 0
            self._clear_optimizer_band_rows(optimizer, self.w_re, rows, band)
            self._clear_optimizer_band_rows(optimizer, self.w_im, rows, band)
        self.order[indices] += 1
        self.refresh_compact_sh_eval_cap()
        return int(indices.numel())

    @torch.no_grad()
    def apply_refinement_snapshot(self, snapshot, spatial_fraction,
                                  angular_fraction, max_level, optimizer=None,
                                  max_active=None):
        """Apply both v2 decisions from one immutable data-fit snapshot.

        Angular capacity is unlocked first, then spatial capacity is created.
        That lets zero-weight siblings inherit the same selected SH order while
        preserving the parent's current coherent prediction.  Both candidate
        sets are selected before either mutation.  This method is opt-in via
        ``--adaptive-capacity-v2``; historical grow/split behavior is intact.
        """
        event = int(self.refine_event_count.item())
        if int(snapshot.get("event", -1)) != event:
            raise ValueError("stale refinement snapshot")
        spatial = self._select_refinement_indices(
            snapshot["spatial_score"], snapshot["spatial_eligible"], spatial_fraction)
        angular = self._select_refinement_indices(
            snapshot["angular_score"], snapshot["angular_eligible"], angular_fraction)

        n_active_before = int(self.active_mask.sum().item())
        capacity = int(self.active_mask.numel())
        active_limit = capacity if max_active is None or int(max_active) <= 0 else int(max_active)
        if active_limit > capacity:
            raise ValueError(f"max_active ({active_limit}) exceeds capacity ({capacity})")
        if n_active_before > active_limit:
            raise ValueError(
                f"max_active ({active_limit}) is below the current active count "
                f"({n_active_before}); prune explicitly before enabling adaptive capacity")
        free_parent_budget = int((~self.active_mask).sum().item()) // 7
        active_parent_budget = max(active_limit - n_active_before, 0) // 7
        spatial = spatial[:min(free_parent_budget, active_parent_budget)]

        n_angular = self.unlock_next_bands(angular, optimizer=optimizer)
        if n_angular:
            self.refine_last_angular_event[angular] = event

        n_spatial = 0
        split_report = "spatial disabled or no eligible positive-score points"
        if spatial.numel() > 0:
            # Reuse the thoroughly-tested lifecycle operation while feeding it
            # the immutable v2 selection only.  The old buffers are irrelevant
            # to a v2 recipe and split() clears them after the lifecycle event.
            self.pos_grad_accum.zero_()
            self.pos_grad_accum[spatial] = snapshot["spatial_score"][spatial]
            self.grad_accum_count.fill_(1)
            n_spatial, _, split_report = self.split(
                max_level=max_level, criterion="position_world", mode="count",
                count=int(spatial.numel()), max_active=active_limit,
                optimizer=optimizer, return_report=True,
                birth_event=event, in_place_heir=True,
            )
            if n_spatial:
                self.refine_last_spatial_event[spatial] = event

        self.refine_event_count.add_(1)
        self.reset_refinement_data_stats()
        active_after = int(self.active_mask.sum().item())

        def summary(scores, idx):
            if idx.numel() == 0:
                return "none"
            chosen = scores[idx]
            return (f"{float(chosen.min()):.3e}/"
                    f"{float(chosen.median()):.3e}/"
                    f"{float(chosen.max()):.3e}")

        report = (
            f"adaptive-v2 event {event}: spatial {n_spatial}/{int(snapshot['spatial_eligible'].sum())} "
            f"(min/med/max {summary(snapshot['spatial_score'], spatial)}), angular "
            f"{n_angular}/{int(snapshot['angular_eligible'].sum())} "
            f"(min/med/max {summary(snapshot['angular_score'], angular)}), "
            f"active {n_active_before}->{active_after}, allocated {capacity}, "
            f"active DOF {self.active_parameter_count()}/allocated parameter scalars "
            f"{self.allocated_parameter_count()} ({self.allocated_parameter_bytes() / 2**20:.1f} MiB; "
            "optimizer state additional); " + split_report
        )
        return n_spatial, n_angular, active_after, report

    @torch.no_grad()
    def grow_angular(self, tail_ratio_threshold=0.05):
        """Angular-derivative growth (Plenoxel-spirit "refine where it is
        needed"), the criterion-driven alternative to grow()'s epoch ladder.

        The SH argument here IS the radar viewpoint direction (see
        active_scatterers), so a point whose response varies rapidly as the
        radar moves is one whose expansion is straining against its current
        order cap.  Detect that spectrally: the share of the point's angular
        energy sitting in its HIGHEST UNLOCKED degree band,

            rho = sum_{l = order} |c_lm|^2  /  sum_{l <= order} |c_lm|^2

        and unlock the next block wherever rho >= tail_ratio_threshold.

        Why plain |c|^2 and not the l(l+1) Laplace-Beltrami weight used by
        --sh-smooth-weight: within one band l(l+1) is a constant, so it
        cancels out of the numerator and only reweights the denominator's
        band mix.  At order=1 that kills the l=0 term (0*(0+1)=0) and rho is
        identically 1 for every point with any angular structure -- every
        point would grow on the first check.  Plain energy keeps the DC band
        as the reference, so a mostly-isotropic point scores low (verified:
        E0=1,E1=0.02 -> 0.020 plain vs 1.000 weighted).

        Self-limiting: a freshly unlocked band starts at zero coefficients,
        so rho drops to 0 right after growing and only climbs back if the
        data actually pushes energy into that band.  Unlike grow(), this
        reads no gradient accumulator and does not reset it.

        Returns (n_grown, n_active).
        """
        n_active = int(self.active_mask.sum().item())
        deg = self.basis_degree.view(1, -1)
        order_e = self.order.unsqueeze(-1)
        sq = self.w_re ** 2 + self.w_im ** 2
        e_keep = (sq * (deg <= order_e).to(sq.dtype)).sum(dim=-1)
        e_top = (sq * (deg == order_e).to(sq.dtype)).sum(dim=-1)
        ratio = e_top / e_keep.clamp_min(torch.finfo(sq.dtype).tiny)
        eligible = self.active_mask & (self.order < self.max_degree) & (e_keep > 0)
        grow_mask = eligible & (ratio >= tail_ratio_threshold)
        self.order[grow_mask] = self.order[grow_mask] + 1
        self.refresh_compact_sh_eval_cap()
        return int(grow_mask.sum().item()), n_active, _tail_ratio_report(ratio[eligible])

    @torch.no_grad()
    def grow(self, threshold_fraction=0.1, mode="relmax"):
        """See SHVoxelGridScene.grow for the two threshold modes."""
        n_active = int(self.active_mask.sum().item())
        if self.grad_accum_count.item() == 0:
            return 0, n_active, ""

        avg_grad = self.grad_accum / self.grad_accum_count.clamp_min(1)
        eligible = self.active_mask & (self.order < self.max_degree)

        n_grown = 0
        report = ""
        if eligible.any():
            vals = avg_grad[eligible]
            thresh, report = grow_threshold_and_report(
                vals, vals.max(), threshold_fraction, mode)
            grow_mask = eligible & (avg_grad >= thresh)
            self.order[grow_mask] = self.order[grow_mask] + 1
            n_grown = int(grow_mask.sum().item())

        self.grad_accum.zero_()
        self.pos_raw_grad_accum.zero_()
        self.pos_grad_accum.zero_()
        self.pos_world_grad_sum.zero_()
        self.pos_world_grad_sq_accum.zero_()
        self.grad_accum_count.zero_()
        self.refresh_compact_sh_eval_cap()
        return n_grown, n_active, report

    @torch.no_grad()
    def split(self, threshold_fraction=0.1, max_level=2,
              criterion="coefficient", mode="relmax", count=0,
              max_active=None, optimizer=None, return_report=False,
              random_seed=20260810, densify=True, birth_event=None,
              in_place_heir=False):
        """Selectively densify point cells without perturbing their prediction.

        The public default preserves the historical slot contract: a selected
        parent is retired and eight inactive slots become its octant children.
        One child is the render-preserving heir, re-centred on the parent’s
        learned position and given its SH coefficients; the other seven start
        at zero.  Thus each split consumes eight free slots but increases the
        active count by seven.

        ``in_place_heir=True`` is the explicit v2 lifecycle variant.  It
        retains the selected parent slot as the heir (including its coefficient
        Adam moments) and activates only seven zero-weight sibling slots.  The
        v2 controller uses that variant so a topology decision does not move a
        live coefficient into a fresh optimizer row.  Both routes preserve the
        legacy two-value return shape unless ``return_report=True`` is asked
        for; richer selectors and reports remain opt-in.
        """
        valid_criteria = (
            "coefficient", "position_raw", "position_world",
            "position_world_snr", "random",
        )
        valid_modes = ("relmax", "quantile", "count")
        if criterion not in valid_criteria:
            raise ValueError(f"criterion must be one of {valid_criteria}, got {criterion!r}")
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode!r}")
        if mode == "quantile" and not (0.0 <= threshold_fraction <= 1.0):
            raise ValueError("quantile threshold_fraction must be in [0, 1]")
        if mode == "count" and count < 0:
            raise ValueError("count must be >= 0")

        n_active_before = int(self.active_mask.sum().item())
        capacity = int(self.active_mask.numel())
        active_limit = capacity if max_active is None or int(max_active) <= 0 else int(max_active)
        if active_limit > capacity:
            raise ValueError(f"max_active ({active_limit}) exceeds capacity ({capacity})")

        def finish(n_parents, report):
            result = (int(n_parents), int(self.active_mask.sum().item()))
            return result + (report,) if return_report else result

        if self.grad_accum_count.item() == 0 and criterion != "random":
            return finish(0, f"split {criterion}/{mode}: empty gradient window")

        event_index = int(self.split_event_count.item())
        if criterion == "random":
            generator = torch.Generator(device="cpu")
            derived_seed = (int(random_seed) + 0x9E3779B1 * event_index) % (2 ** 63 - 1)
            generator.manual_seed(derived_seed)
            scores = torch.rand(
                int(self.active_mask.numel()), generator=generator, dtype=torch.float64
            ).to(self.active_mask.device)
        elif criterion == "position_world_snr":
            count_f = self.grad_accum_count.clamp_min(1).to(self.pos_world_grad_sum.dtype)
            denom = (self.pos_world_grad_sq_accum * count_f).sqrt().clamp_min(
                torch.finfo(self.pos_world_grad_sum.dtype).tiny)
            consistency = (self.pos_world_grad_sum / denom).norm(dim=-1)
            unlocked = (self.basis_degree.view(1, -1) <= self.order[:, None]).to(self.w_re.dtype)
            amplitude = ((self.w_re.square() + self.w_im.square()) * unlocked).sum(dim=-1).sqrt()
            scores = amplitude * consistency
        else:
            score_accum = {
                "coefficient": self.grad_accum,
                "position_raw": self.pos_raw_grad_accum,
                "position_world": self.pos_grad_accum,
            }[criterion]
            scores = score_accum / self.grad_accum_count.clamp_min(1)

        # Positive, finite filtering prevents a zero/tied statistic from
        # silently consuming all spare capacity.
        eligible = (self.active_mask & (self.level < max_level)
                    & (self.cell_half[:, 0] > 0) & torch.isfinite(scores) & (scores > 0))
        eligible_idx = eligible.nonzero(as_tuple=True)[0]
        n_eligible = int(eligible_idx.numel())
        requested_idx = eligible_idx[:0]
        # The original public split only exposed coefficient/relmax
        # densification.  It retained ``nonzero``'s ascending slot order when
        # every selected parent fit, and only ranked by gradient when its free
        # eight-child budget forced a truncation.  Keep that exact allocation
        # order for the legacy route.  V2's in-place lifecycle and all extended
        # selectors retain the intentional ranked selection below.
        legacy_default_order = (densify and not in_place_heir
                                and criterion == "coefficient" and mode == "relmax")
        if n_eligible > 0:
            if legacy_default_order:
                cutoff = threshold_fraction * scores[eligible_idx].max()
                requested_idx = eligible_idx[scores[eligible_idx] >= cutoff]
            else:
                ranking = torch.argsort(scores[eligible_idx], descending=True, stable=True)
                ranked_idx = eligible_idx[ranking]
                if mode == "relmax":
                    cutoff = threshold_fraction * scores[ranked_idx[0]]
                    requested_idx = ranked_idx[scores[ranked_idx] >= cutoff]
                elif mode == "quantile":
                    requested_idx = ranked_idx[:int(math.ceil(threshold_fraction * n_eligible))]
                else:
                    requested_idx = ranked_idx[:min(int(count), n_eligible)]

        free_idx = (~self.active_mask).nonzero(as_tuple=True)[0]
        # Retired-parent legacy splits require eight new slots; the explicit
        # in-place v2 variant needs only the seven non-heir siblings.  Both
        # increase active count by seven, so the active-cap budget is shared.
        slots_per_parent = 7 if in_place_heir else 8
        free_parent_budget = int(free_idx.numel()) // slots_per_parent
        active_parent_budget = max(active_limit - n_active_before, 0) // 7
        parent_budget = (min(free_parent_budget, active_parent_budget)
                         if densify else int(requested_idx.numel()))
        if legacy_default_order and int(requested_idx.numel()) > parent_budget:
            # Preserve ascending historical slot order only when every
            # selected parent fits.  Any real allocation ceiling (the legacy
            # free-slot budget or an explicit active cap) instead awards the
            # available child blocks to the highest-gradient parents.
            budget_ranking = torch.argsort(
                scores[requested_idx], descending=True, stable=True)
            requested_idx = requested_idx[budget_ranking]
        parent_idx = requested_idx[:parent_budget]
        n_requested = int(requested_idx.numel())
        n_parents = int(parent_idx.numel())

        if n_parents > 0:
            parent_anchor = self.anchors[parent_idx].clone()
            parent_half = self.cell_half[parent_idx, 0].clone()
            parent_delta = self.delta_raw[parent_idx].clone()
            parent_w_re = self.w_re[parent_idx].clone()
            parent_w_im = self.w_im[parent_idx].clone()
            parent_order = self.order[parent_idx].clone()
            parent_level = self.level[parent_idx].clone()
            parent_position = parent_anchor + parent_half[:, None] * torch.tanh(parent_delta)
            child_half = 0.5 * parent_half
            if self._support_bounds_runtime_enabled:
                # Recentered cells must themselves remain inside the immutable
                # original support, not merely have their rendered coordinate
                # clamped after the fact.  At an exterior face this can reduce
                # a child's scalar movement radius to zero; that is preferable
                # to silently growing the physical inverse-problem domain.
                parent_position = torch.maximum(
                    torch.minimum(parent_position, self.support_max), self.support_min)
                heir_margin = torch.minimum(
                    parent_position - self.support_min,
                    self.support_max - parent_position,
                ).amin(dim=-1).clamp_min(0)
                # The in-place heir is centred at the learned parent position,
                # so it alone needs the potentially tiny boundary-limited
                # radius needed to keep that cell in support.  Siblings are
                # centred at their own octant anchors and retain the largest
                # contained half-width available there.
                child_half = torch.minimum(child_half, heir_margin)

            if densify:
                signs = torch.tensor(
                    [[sx, sy, sz] for sx in (-1.0, 1.0)
                     for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
                    dtype=parent_anchor.dtype, device=parent_anchor.device,
                )
                all_octant_anchors = (parent_anchor[:, None, :]
                                       + 0.5 * parent_half[:, None, None] * signs[None])
                octant_half = (0.5 * parent_half)[:, None].expand(-1, 8)
                if self._support_bounds_runtime_enabled:
                    # Clip an unusual legacy/external parent anchor first,
                    # then bound each octant cell against its *own* margin.
                    # Never reuse the heir's boundary margin: doing so could
                    # collapse useful interior siblings to zero width.
                    all_octant_anchors = torch.maximum(
                        torch.minimum(all_octant_anchors, self.support_max),
                        self.support_min,
                    )
                    sibling_margin = torch.minimum(
                        all_octant_anchors - self.support_min,
                        self.support_max - all_octant_anchors,
                    ).amin(dim=-1).clamp_min(0)
                    octant_half = torch.minimum(octant_half, sibling_margin)
                octant_bits = (parent_position >= parent_anchor).long()
                heir_octant = (octant_bits[:, 0] * 4
                               + octant_bits[:, 1] * 2 + octant_bits[:, 2])
                if in_place_heir:
                    # V2: parent remains the live heir.  Exclude the octant
                    # it replaces from the newly allocated siblings.
                    sibling_idx = free_idx[:7 * n_parents]
                    sibling_mask = torch.ones(
                        n_parents, 8, dtype=torch.bool, device=parent_anchor.device)
                    sibling_mask[torch.arange(n_parents, device=parent_anchor.device), heir_octant] = False
                    sibling_anchors = all_octant_anchors[sibling_mask]
                    sibling_cell_half = octant_half[sibling_mask]
                    self._tombstone_slots(sibling_idx, optimizer=optimizer)
                    self.anchors[sibling_idx] = sibling_anchors
                    self.cell_half[sibling_idx] = sibling_cell_half[:, None]
                    self.order[sibling_idx] = parent_order.repeat_interleave(7)
                    self.level[sibling_idx] = (parent_level + 1).repeat_interleave(7)
                    self.active_mask[sibling_idx] = True
                    if birth_event is not None:
                        self.refine_birth_event[sibling_idx] = int(birth_event)

                    # The existing parent is the heir.  Its coefficients and
                    # their optimizer moments survive; its coordinate moments
                    # reset after re-centring.
                    self.anchors[parent_idx] = parent_position
                    self.cell_half[parent_idx] = child_half[:, None]
                    self.level[parent_idx] = parent_level + 1
                    self.delta_raw[parent_idx] = 0
                    self._clear_optimizer_rows(optimizer, self.delta_raw, parent_idx)
                else:
                    # Historical public route: retired parent plus eight fresh
                    # child slots.  The inherited coefficient moves to the
                    # appropriate child but deliberately starts with a fresh
                    # optimizer row, exactly like every newly allocated slot.
                    child_idx = free_idx[:8 * n_parents]
                    child_matrix = child_idx.reshape(n_parents, 8)
                    heir_idx = child_matrix[
                        torch.arange(n_parents, device=parent_anchor.device), heir_octant]
                    self._tombstone_slots(child_idx, optimizer=optimizer)
                    self.anchors[child_idx] = all_octant_anchors.reshape(-1, 3)
                    self.cell_half[child_idx] = octant_half.reshape(-1, 1)
                    self.order[child_idx] = parent_order.repeat_interleave(8)
                    self.level[child_idx] = (parent_level + 1).repeat_interleave(8)
                    self.active_mask[child_idx] = True
                    self.anchors[heir_idx] = parent_position
                    self.cell_half[heir_idx] = child_half[:, None]
                    self.w_re[heir_idx] = parent_w_re
                    self.w_im[heir_idx] = parent_w_im
                    if birth_event is not None:
                        self.refine_birth_event[child_idx] = int(birth_event)

                    # Retired slots are fully tombstoned, including their Adam
                    # history, before they are eventually recycled.
                    self._tombstone_slots(parent_idx, optimizer=optimizer)
            else:
                # Recenter-only is an extended diagnostic/control operation:
                # it never changes the active slot layout, regardless of which
                # densification route was requested.
                self.anchors[parent_idx] = parent_position
                self.delta_raw[parent_idx] = 0
                self._clear_optimizer_rows(optimizer, self.delta_raw, parent_idx)

        self.grad_accum.zero_()
        self.pos_raw_grad_accum.zero_()
        self.pos_grad_accum.zero_()
        self.pos_world_grad_sum.zero_()
        self.pos_world_grad_sq_accum.zero_()
        self.grad_accum_count.zero_()
        self.split_event_count.add_(1)
        self.refresh_compact_sh_eval_cap()

        selected_scores = scores[parent_idx]
        if selected_scores.numel():
            score_summary = (f"score min/median/max {float(selected_scores.min()):.3e}/"
                             f"{float(selected_scores.median()):.3e}/"
                             f"{float(selected_scores.max()):.3e}")
        else:
            score_summary = "no selected scores"
        action = ("in_place_densify" if in_place_heir else "legacy_densify") if densify else "recenter_only"
        report = (f"{action} {criterion}/{mode}: eligible {n_eligible}, requested {n_requested}, "
                  f"selected {n_parents}; budgets free={free_parent_budget}, "
                  f"active={active_parent_budget}; {score_summary}; "
                  f"active DOF {self.active_parameter_count()}")
        return finish(n_parents, report)

    @classmethod
    def from_regular_grid(cls, granularity, extent, device, **kw):
        anchors = generate_dynamic_grid(granularity, extent, device, jitter=False).reshape(-1, 3)
        cell_half = extent / granularity
        return cls(anchors, cell_half, device, **kw)

    @classmethod
    def from_state(cls, state_dict, device, **kw):
        anchors = state_dict["anchors"].to(device)
        cell_half = state_dict["cell_half"].to(device)
        n_basis = state_dict["w_re"].shape[-1]
        inferred_degree = int(round(n_basis ** 0.5)) - 1
        kw.setdefault("max_degree", inferred_degree)
        model = cls(anchors, cell_half, device, **kw)
        model.load_state_dict({k: v.to(device) for k, v in state_dict.items()})
        return model
