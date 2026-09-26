"""Independent equation-based SE implementation, arXiv:2602.17556v1.

This is a new recipe, not a migration of the historical isotropic/stabilized
checkpoints. See docs/SUGAVANAM_ERTIN_PAPER.md for equation ambiguities and
explicit acquisition/numerical adaptations. No mesh supervision is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = "rift_sugavanam_ertin_subaperture_sdf_v2"
PAPER = "https://arxiv.org/html/2602.17556v1"


def unit_vectors(x):
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all() or np.any(norms <= 1e-12):
        raise ValueError("Expected finite nonzero directions [N,3]")
    return x / norms


@dataclass
class Subapertures:
    """Fixed angular bins fitted from TRAIN geometry only, not manifest order."""
    azimuth_bins: int
    elevation_bins: int
    occupied_bins: np.ndarray
    directions: np.ndarray
    assignments: np.ndarray

    @staticmethod
    def bin_ids(directions, azimuth_bins, elevation_bins):
        u = unit_vectors(directions)
        az = np.mod(np.arctan2(u[:, 1], u[:, 0]), 2 * np.pi)
        el = np.arcsin(np.clip(u[:, 2], -1, 1))
        a = np.minimum((az / (2 * np.pi) * azimuth_bins).astype(int), azimuth_bins - 1)
        e = np.minimum(((el + np.pi / 2) / np.pi * elevation_bins).astype(int), elevation_bins - 1)
        return e * azimuth_bins + a

    @classmethod
    def fit(cls, directions, azimuth_bins=72, elevation_bins=1, *, direction_statistics=None):
        if type(azimuth_bins) is not int or type(elevation_bins) is not int or min(azimuth_bins, elevation_bins) < 1:
            raise ValueError("Angular bin counts must be positive integers")
        u = unit_vectors(directions)
        bins = cls.bin_ids(u, azimuth_bins, elevation_bins)
        occupied, assignments = np.unique(bins, return_inverse=True)
        stats = None if direction_statistics is None else np.asarray(direction_statistics, dtype=np.float64)
        if stats is not None and (stats.shape != (len(u), 4) or not np.isfinite(stats).all() or np.any(stats[:, 3] <= 0)):
            raise ValueError("Expected per-view [sum_cos_az, sum_sin_az, sum_elevation, pulse_count]")
        means = []
        for i in range(len(occupied)):
            local = u[assignments == i]
            # Circular mean azimuth, arithmetic mean elevation: paper Eq. 7.
            angle = np.arctan2(local[:, 1], local[:, 0])
            x, y = np.cos(angle).mean(), np.sin(angle).mean()
            el = np.arcsin(np.clip(local[:, 2], -1, 1)).mean()
            if stats is not None:
                sums = stats[assignments == i].sum(0)
                x, y, el = sums[:3]/sums[3]
            if np.hypot(x, y) < 1e-12:
                raise ValueError("Sub-aperture spans cancelling directions; use narrower angular bins")
            az = np.arctan2(y, x)
            means.append([np.cos(az)*np.cos(el), np.sin(az)*np.cos(el), np.sin(el)])
        return cls(azimuth_bins, elevation_bins, occupied, np.asarray(means), assignments)

    def assign(self, directions):
        """Validation-only readout: nearest TRAIN centre for empty angular bins."""
        u = unit_vectors(directions)
        ids = self.bin_ids(u, self.azimuth_bins, self.elevation_bins)
        lookup = {int(b): i for i, b in enumerate(self.occupied_bins)}
        fallback = np.asarray([int(b) not in lookup for b in ids])
        assigned = np.asarray([lookup.get(int(b), int(np.argmax(v @ self.directions.T)))
                               for b, v in zip(ids, u)], dtype=np.int64)
        return assigned, fallback

    def record(self):
        return dict(azimuth_bins=self.azimuth_bins, elevation_bins=self.elevation_bins,
                    occupied_bins=self.occupied_bins.tolist(), directions=self.directions.tolist(),
                    train_assignments=self.assignments.tolist(),
                    validation_empty_bin_policy="nearest_training_subaperture_direction")


def aggregate_scattering(coefficients, directions):
    """Eqs. 5 and 7: sum |S_m| and retain the strongest sub-aperture direction."""
    c = torch.as_tensor(coefficients)
    if c.ndim != 2 or not c.is_complex() or not torch.isfinite(c).all():
        raise ValueError("Expected finite complex [subaperture, voxel] coefficients")
    d = torch.as_tensor(unit_vectors(directions), device=c.device, dtype=c.real.dtype)
    if len(d) != len(c):
        raise ValueError("One mean direction is required per sub-aperture")
    magnitude = c.abs()
    strongest = magnitude.argmax(0)
    return magnitude.sum(0), d[strongest], strongest


def pca_normals(points, radius, *, fallback_directions=None):
    """Radius PCA; sparse/collinear neighbourhoods use Eq. 7 when available.

    Exclude self when counting neighbours. Never expand a radius to k-NN and
    never orient by the cloud median. Iso normals with insufficient geometric
    evidence are masked, not replaced by the very network gradient being scored.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.isfinite(p).all() or radius <= 0:
        raise ValueError("Invalid PCA points/radius")
    fallback = None if fallback_directions is None else unit_vectors(fallback_directions)
    if fallback is not None and fallback.shape != p.shape:
        raise ValueError("Fallback directions must match cloud")
    tree = cKDTree(p)
    normals = np.zeros_like(p)
    valid = np.zeros(len(p), dtype=bool)
    used_fallback = np.zeros(len(p), dtype=bool)
    for i, neighbours in enumerate(tree.query_ball_point(p, radius)):
        neighbours = [j for j in neighbours if j != i]
        if len(neighbours) >= 3:
            local = p[[i, *neighbours]]
            local = local - local.mean(0)
            eigenvalues, eigenvectors = np.linalg.eigh(local.T @ local)
            if eigenvalues[1] > max(eigenvalues[-1] * 1e-10, np.finfo(float).tiny):
                normals[i] = eigenvectors[:, 0]
                valid[i] = True
        if not valid[i] and fallback is not None:
            normals[i] = fallback[i]
            valid[i] = used_fallback[i] = True
    return normals, valid, used_fallback


def complex_soft_threshold(z, threshold):
    """Proximal map of complex-modulus L1, not componentwise real/imag L1."""
    if threshold < 0 or not math.isfinite(float(threshold)):
        raise ValueError("Invalid L1 threshold")
    return z * (1 - threshold / z.abs().clamp_min(torch.finfo(z.real.dtype).tiny)).clamp_min(0)


def proximal_step(x, value_and_grad, value, *, l1, lipschitz=1.0, max_backtracks=60):
    """Monotone proximal gradient with a verified smooth-term majorizer.

    Solves 0.5*mean(|A x-y|^2) + l1*sum(|x|), the explicitly disclosed
    Lagrangian variant of Eq. 4. No free gain can evade the sparsity penalty.
    Callables may stream an entire sub-aperture without retaining radar cubes.
    """
    if l1 < 0 or lipschitz <= 0 or not math.isfinite(lipschitz):
        raise ValueError("Invalid proximal solver parameters")
    f, g = value_and_grad(x)
    if not math.isfinite(float(f)) or not torch.isfinite(g).all():
        raise FloatingPointError("Nonfinite data objective/gradient")
    L = float(lipschitz)
    for backtracks in range(max_backtracks):
        candidate = complex_soft_threshold(x - g / L, l1 / L)
        delta = candidate - x
        smooth = float(value(candidate))
        upper = float(f) + float((g.conj() * delta).real.sum()) + 0.5 * L * float(delta.abs().square().sum())
        tolerance = 64 * torch.finfo(x.real.dtype).eps * max(1.0, abs(float(f)), abs(upper))
        if math.isfinite(smooth) and smooth <= upper + tolerance:
            return candidate.detach(), L, dict(data_loss=smooth,
                l1_loss=l1 * float(candidate.abs().sum()),
                proximal_gradient_norm=L * float(delta.norm()), backtracks=backtracks)
        L *= 2
    raise RuntimeError("Sparse reconstruction line search failed; no update committed")


class PaperSDF(nn.Module):
    """Eight Softplus layers, width 512, input skip into the fourth layer.

    Default standard Gaussian weights AND biases and an unscaled tanh output
    follow Section 3.1. An explicit initialization_std records a user-selected
    Gaussian scale. Coordinates are in metres. No geometric initialization.
    """
    def __init__(self, extent, n_fourier=9, fourier_scale=2., hidden_dim=512,
                 n_layers=8, initialization="standard_gaussian", seed=42,
                 initialization_std=1.):
        super().__init__()
        if extent <= 0 or n_fourier < 1 or hidden_dim < 1 or n_layers < 4 or fourier_scale <= 0:
            raise ValueError("Invalid SDF architecture")
        if initialization != "standard_gaussian":
            raise ValueError("The paper specifies standard Gaussian weights and biases")
        if not math.isfinite(initialization_std) or initialization_std <= 0:
            raise ValueError("initialization_std must be finite and positive")
        self.extent = float(extent)
        self.model_config = dict(extent=extent, n_fourier=n_fourier, fourier_scale=fourier_scale,
            hidden_dim=hidden_dim, n_layers=n_layers, initialization=initialization, seed=seed)
        if initialization_std != 1.:
            self.model_config["initialization_std"] = float(initialization_std)
        gen = torch.Generator().manual_seed(seed)
        self.register_buffer("bands", torch.randn(n_fourier, 3, generator=gen) * fourier_scale)
        encoded = 3 + 2 * n_fourier
        self.layers = nn.ModuleList([nn.Linear(encoded if i == 0 else hidden_dim + (encoded if i == 3 else 0),
                                              hidden_dim) for i in range(n_layers)])
        self.output = nn.Linear(hidden_dim, 1)
        with torch.no_grad():
            for layer in [*self.layers, self.output]:
                layer.weight.copy_(torch.randn(layer.weight.shape, generator=gen) * initialization_std)
                layer.bias.copy_(torch.randn(layer.bias.shape, generator=gen) * initialization_std)

    def forward(self, points):
        x = points
        angle = 2 * math.pi * (x @ self.bands.T)
        encoded = torch.cat((x, angle.sin(), angle.cos()), -1)
        h = encoded
        for i, layer in enumerate(self.layers):
            if i == 3:
                h = torch.cat((h, encoded), -1)
            h = F.softplus(layer(h))
        return torch.tanh(self.output(h)).squeeze(-1)


def field_gradient(model, points, *, create_graph=False):
    """Local spatial autograd also works inside an outer no_grad readout."""
    parameter = next(model.parameters(), None)
    dtype = points.dtype if parameter is None else parameter.dtype
    device = points.device if parameter is None else parameter.device
    with torch.enable_grad():
        q = points.detach().to(device=device, dtype=dtype).requires_grad_(True)
        values = model(q)
        if values.requires_grad:
            gradient = torch.autograd.grad(values.sum(), q, create_graph=create_graph,
                retain_graph=True, allow_unused=True)[0]
        else:
            gradient = None
        if gradient is None:
            gradient = torch.zeros_like(q)
    return values, gradient


def initialization_audit(model, extent, *, seed=42, samples=256):
    """Bounded forward/derivative probe, without fitting or radar observations.

    Saturation is reported, never fixed by secretly rescaling published weights.
    A degenerate probe prevents an ineligible numerical failure becoming a
    geometry comparison. It is not a convergence guarantee when it passes.
    """
    gen = torch.Generator().manual_seed(seed)
    points = (torch.rand(samples, 3, generator=gen)*2-1)*extent
    f, g = field_gradient(model, points)
    finite = bool(torch.isfinite(f).all() and torch.isfinite(g).all())
    norms = g.detach().norm(dim=-1)
    degenerate = not finite or not bool((norms > 0).any())
    return dict(samples=samples, coordinate_extent_m=extent, seed=seed,
        mean_abs_sdf_m=float(f.detach().abs().mean()),
        positive_fraction=float((f.detach() > 0).float().mean()),
        saturated_fraction=float((f.detach().abs() == 1).float().mean()),
        mean_gradient_norm=float(norms.mean()), nonzero_spatial_gradients=int((norms > 0).sum()),
        status="initialization_degenerate" if degenerate else "initialization_probe_passed",
        interpretation="author_initialization_detail_unresolved" if degenerate else "not_a_convergence_check",
        initialization=("standard_gaussian_weights_and_biases"
            if model.model_config.get("initialization_std", 1.) == 1.
            else "gaussian_weights_and_biases"),
        **({"initialization_std": model.model_config["initialization_std"]}
           if "initialization_std" in model.model_config else {}),
        response_payload_read=False)


def project_surface(model, points, *, extent, max_step, tolerance=1e-4, iterations=24):
    """Eqs. 9--10, stopping on |f| <= 1e-4, without extra surface priors.

    The text omits absolute-value bars despite describing a zero-level set;
    interpreting this as residual magnitude is recorded in the fidelity ledger.
    Zero gradients/nonfinite arithmetic are undefined operations, not surfaces.
    Only final points inside the requested ROI enter its sampling set.
    """
    if min(extent, max_step, tolerance) <= 0 or iterations < 1:
        raise ValueError("Invalid projection configuration")
    parameter = next(model.parameters(), None)
    q = points.detach().clone() if parameter is None else points.detach().to(parameter).clone()
    good = torch.isfinite(q).all(-1)
    tiny = torch.finfo(q.dtype).tiny
    for _ in range(iterations):
        f, g = field_gradient(model, q)
        f, g = f.detach(), g.detach()
        norm = g.norm(dim=-1)
        good &= torch.isfinite(f) & torch.isfinite(g).all(-1) & (norm.square() > tiny)
        update = f[:, None] * g / norm.square().clamp_min(tiny)[:, None]
        update *= (max_step / update.norm(dim=-1).clamp_min(1e-30)).clamp_max(1)[:, None]
        active = good & (f.abs() > tolerance)
        q = q - torch.where(active[:, None], update, torch.zeros_like(update))
        good &= torch.isfinite(q).all(-1)
    f, g = field_gradient(model, q)
    f, g = f.detach(), g.detach()
    norm = g.norm(dim=-1)
    good &= (torch.isfinite(f) & torch.isfinite(g).all(-1) & (norm.square() > tiny)
             & (f.abs() <= tolerance) & (q.abs() < extent).all(-1))
    return q[good].detach(), dict(attempted=len(q), accepted=int(good.sum()),
        rejected=int((~good).sum()), degenerate_gradient=int((norm.square() <= tiny).sum()),
        max_residual=float(f[good].abs().max()) if good.any() else None)


def resample_moves(points, normals, *, radius, bandwidth, alpha, max_step,
                   edge_weight="paper_literal"):
    """Eqs. 11--15, using radius neighbours and separately clipped updates.

    Eq. 13 prints a signed, unsquared exponent; preserve it literally.
    Stable exponent shifting preserves ratios without clipping the formula.
    """
    p, n = np.asarray(points, dtype=np.float64), unit_vectors(normals)
    if p.shape != n.shape or min(radius, bandwidth, max_step) <= 0 or alpha < 0:
        raise ValueError("Invalid resampling inputs")
    if edge_weight != "paper_literal":
        raise ValueError("The paper recipe preserves the printed Eq. 13 exponent")
    tree, uniform, edge = cKDTree(p), p.copy(), p.copy()
    def clip(v):
        return v * min(1., max_step / max(np.linalg.norm(v), 1e-30))
    for i, neighbours in enumerate(tree.query_ball_point(p, radius)):
        idx = [j for j in neighbours if j != i and np.linalg.norm(p[j]-p[i]) > 1e-12]
        if not idx:
            continue
        delta = p[idx] - p[i]
        distance = np.linalg.norm(delta, axis=1)
        log_w = -distance**2 / bandwidth**2
        # Eq. 11 is a SUM of weighted unit vectors, not a normalized mean.
        uniform[i] -= alpha * (np.exp(log_w)[:, None] * delta/distance[:, None]).sum(0)
        dot = np.einsum("ij,ij->i", n[idx], delta)
        log_phi = -dot / bandwidth**2
        phi, w = np.exp(log_phi-log_phi.max()), np.exp(log_w-log_w.max())
        edge_delta = (phi[:, None]*delta).sum(0)/phi.sum()
        repel_delta = .5*(w[:, None]*delta).sum(0)/w.sum()
        edge[i] -= clip(edge_delta) + clip(repel_delta)
    return uniform, edge


def priority_candidate(points, *, radius, excluded=()):
    """Eqs. 16--18: global highest priority, farthest local neighbour, 2:1 insertion."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 2:
        return None
    candidates = []
    for i, neighbours in enumerate(cKDTree(p).query_ball_point(p, radius)):
        idx = [j for j in neighbours if j != i]
        if idx:
            distances = np.linalg.norm(p[idx]-p[i], axis=1)
            j = idx[int(np.argmax(distances))]
            if (i, j) not in excluded:
                candidates.append((float(distances.max()), -i, -j))
    if not candidates:
        return None
    _, i, j = max(candidates)
    i, j = -i, -j
    return (2*p[i]+p[j])/3, (i, j)


def refresh_surface(model, seeds, *, extent, pitch, count, generator,
                    edge_weight="paper_literal", projection_iterations=24):
    """Project, redistribute, edge-resample, priority-upsample, and reproject."""
    if count < 4 or len(seeds) < 3:
        raise ValueError("Insufficient iso-point seeds/target count")
    tolerance = 1e-4
    kwargs = dict(extent=extent, max_step=2*pitch, tolerance=tolerance, iterations=projection_iterations)
    initial_count = max(3, count//2)
    index = torch.randint(len(seeds), (initial_count*2,), generator=generator, device=seeds.device)
    proposed = seeds[index] + torch.randn((len(index), 3), generator=generator,
                                         device=seeds.device, dtype=seeds.dtype)*pitch
    q, first = project_surface(model, proposed, **kwargs)
    q = torch.unique(q, dim=0)[:initial_count]
    audit = dict(initial=first, edge_weight=edge_weight, insertions=0, insertion_rejections=0)
    if len(q) < 3:
        return q, {**audit, "status": "insufficient_roots"}
    _, gradients = field_gradient(model, q)
    uniform, _ = resample_moves(q.cpu(), gradients.detach().cpu(), radius=4*pitch,
                               bandwidth=2*pitch, alpha=.1*pitch, max_step=2*pitch,
                               edge_weight=edge_weight)
    q, audit["uniform_projection"] = project_surface(model, torch.as_tensor(uniform).to(seeds), **kwargs)
    if len(q) < 3:
        return q, {**audit, "status": "insufficient_roots"}
    _, gradients = field_gradient(model, q)
    _, edged = resample_moves(q.cpu(), gradients.detach().cpu(), radius=4*pitch,
                              bandwidth=2*pitch, alpha=.1*pitch, max_step=2*pitch,
                              edge_weight=edge_weight)
    q, audit["edge_projection"] = project_surface(model, torch.as_tensor(edged).to(seeds), **kwargs)
    # Explicit bounded insertion ledger prevents duplicate/no-root infinite loops.
    rejected_pairs = set()
    for _ in range(2*count):
        if len(q) >= count:
            break
        candidate = priority_candidate(q.cpu(), radius=4*pitch, excluded=rejected_pairs)
        if candidate is None:
            break
        point, pair = candidate
        new, _ = project_surface(model, torch.as_tensor(point[None]).to(seeds), **kwargs)
        if len(new) and float(torch.cdist(new, q).min()) > tolerance:
            q = torch.cat((q, new))
            audit["insertions"] += 1
        else:
            rejected_pairs.add(pair)
            audit["insertion_rejections"] += 1
    audit.update(status="ready" if len(q) >= 3 else "insufficient_roots", count=len(q), requested=count)
    return q.detach(), audit


def sdf_losses(model, on, normals, background, iso, iso_normals, *, extent,
               alpha_off=100., iso_normal_valid=None):
    """Six terms in Eqs. 19--25; sign-invariant normals, no signed anchors.

    Raw metric f enters the published losses; do not rescale losses by extent.
    """
    f_on, g_on = field_gradient(model, on, create_graph=True)
    f_off, g_off = field_gradient(model, background, create_graph=True)
    def cosine(g, n):
        return 1-F.cosine_similarity(g, n.to(g), dim=-1).abs()
    zero = f_on.sum()*0
    losses = dict(on=f_on.abs().mean(), normal=cosine(g_on, normals).mean(),
                  off=torch.exp(-alpha_off*f_off.abs()).mean(), iso=zero, iso_normal=zero)
    eik = [(1-g_off.norm(dim=-1)).abs()]
    if iso is not None and len(iso):
        f_iso, g_iso = field_gradient(model, iso, create_graph=True)
        losses["iso"] = f_iso.abs().mean()
        valid = torch.ones(len(iso), dtype=torch.bool, device=g_iso.device) if iso_normal_valid is None else iso_normal_valid.to(g_iso.device)
        if valid.any():
            losses["iso_normal"] = cosine(g_iso[valid], iso_normals[valid]).mean()
        eik.append((1-g_iso.norm(dim=-1)).abs())
    losses["eik"] = torch.cat(eik).mean()
    return losses
