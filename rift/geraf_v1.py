"""Numerical helpers for the explicitly versioned GeRaF v1 adaptation.

Equations follow arXiv:2605.29097v2 (GeRaF 1.0). Implementation evidence:
VictorLlu/GeRaF-SENS, 38266cb6e194e2f3dcbead614069a7281ffd21a5,
``rf_rendering.py`` and ``datasets/transforms/sample.py``. The release's
vision supervision, fixed-reflectivity stage, and GeRaF 2.0 losses are not
part of this implementation. See docs/GERAF_V1_HARDENING.md for adaptations.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping

import torch
import torch.nn.functional as F
from rift.vendor.geraf_sens.sdf_network import SDFNetwork


IMPLEMENTATION = "hardened_v1"
UPSTREAM_COMMIT = "38266cb6e194e2f3dcbead614069a7281ffd21a5"


class SourceSDFNetwork(SDFNetwork):
    """Released SDF architecture with the v1 paper dimensions and metric units.

    The source's raw XYZ, geometric initialization, weight normalization, skip
    dimensions and frequency ordering are retained. Its sin(2**l*x) convention
    and normalized inputs are disclosed implementation choices (the paper
    writes pi in its encoding equation). Reflectivity remains the separate
    position-only four-layer v1 network, not the released v2 feature head.
    """
    def __init__(self, extent, *, n_levels=10, hidden_dim=256, n_layers=8, skip_layer=4):
        if n_levels != 10 or not math.isfinite(extent) or extent <= 0:
            raise ValueError("GeRaF v1 source SDF requires ten levels and a positive extent")
        if n_layers < 1 or (skip_layer is not None and not 0 < skip_layer < n_layers):
            raise ValueError("source SDF requires a valid hidden layer/skip configuration")
        if skip_layer is not None and hidden_dim <= 3 + 6 * n_levels:
            raise ValueError("source SDF skip width must exceed its encoding dimension")
        super().__init__(d_in=3, d_out=1, d_hidden=hidden_dim, n_layers=n_layers,
                         skip_in=() if skip_layer is None else (skip_layer,), multires=n_levels,
                         bias=.5, scale=1., geometric_init=True, weight_norm=True)
        self.extent = float(extent)
        self.n_levels, self.hidden_dim, self.n_layers = n_levels, hidden_dim, n_layers
        self.skip_layer = skip_layer
        self.hidden_activation, self.softplus_beta = "softplus", 100.
        self.encoding_include_input, self.encoding_coordinate_scale = True, 1 / extent
        self.initialization, self.output_scale = "upstream_geometric", float(extent)

    def forward(self, xyz):
        return super().forward(xyz / self.extent).squeeze(-1) * self.extent

    def gradient(self, xyz, *, create_graph=True):
        from rift.geraf import GeRaFSDFNetwork
        return GeRaFSDFNetwork.gradient(self, xyz, create_graph=create_graph)


def ray_box_intervals(origins, directions, extent):
    """Slab intersections with the metric scene AABB; zero components are safe."""
    if not math.isfinite(extent) or extent <= 0:
        raise ValueError("extent must be finite and positive")
    parallel = directions.abs() < 1e-14
    divisor = torch.where(parallel, torch.ones_like(directions), directions)
    a, b = (-extent - origins) / divisor, (extent - origins) / divisor
    low = torch.where(parallel, torch.full_like(a, -torch.inf), torch.minimum(a, b))
    high = torch.where(parallel, torch.full_like(a, torch.inf), torch.maximum(a, b))
    near, far = low.amax(-1), high.amin(-1)
    outside_parallel = (parallel & (origins.abs() > extent)).any(-1)
    valid = (~outside_parallel) & (far > near) & (far > 0)
    return near.clamp_min(0), far, valid


def sample_scene_rays(aperture_center, primary_direction, *, extent, n_azimuth,
                      n_elevation, n_depth, azimuth_axis=None, elevation_axis=None):
    """Deterministic midpoint cells covering the whole scene AABB.

    Sampling footprint is independent of the physical antenna aperture. The
    latter is only 2.85 cm wide for a 10 cm RIFT object. Actual Tx/Rx positions
    still determine phase, spreading, and reflection directions. Invalid rays
    are omitted; each retained ray partitions its complete near/far interval.
    """
    from rift.geraf import PrimaryRaySamples, aperture_basis, safe_normalize
    if min(n_azimuth, n_elevation, n_depth) < 1:
        raise ValueError("scene ray dimensions must be positive")
    center = aperture_center.to(dtype=torch.float64)
    primary = safe_normalize(primary_direction.to(center))
    if azimuth_axis is None or elevation_axis is None:
        azimuth_axis, elevation_axis, _ = aperture_basis(primary)
    azimuth_axis = safe_normalize(azimuth_axis.to(center))
    elevation_axis = safe_normalize(elevation_axis.to(center))
    az_half = extent * azimuth_axis.abs().sum()
    el_half = extent * elevation_axis.abs().sum()
    az = ((torch.arange(n_azimuth, device=center.device, dtype=center.dtype) + .5)
          * (2 / n_azimuth) - 1) * az_half
    el = ((torch.arange(n_elevation, device=center.device, dtype=center.dtype) + .5)
          * (2 / n_elevation) - 1) * el_half
    ee, aa = torch.meshgrid(el, az, indexing="ij")
    origins = (center + aa[..., None] * azimuth_axis + ee[..., None] * elevation_axis).reshape(-1, 3)
    near, far, valid = ray_box_intervals(origins, primary, extent)
    origins, near, far = origins[valid], near[valid], far[valid]
    if not len(origins):
        raise ValueError("no primary rays intersect the scene")
    unit = torch.linspace(0, 1, n_depth + 1, device=center.device, dtype=center.dtype)
    edges = near[:, None] + (far - near)[:, None] * unit
    depths = (edges[:, :-1] + edges[:, 1:]) * .5
    points = origins[:, None] + depths[..., None] * primary
    return PrimaryRaySamples(origins, primary, depths, edges.diff(dim=-1), points, edges)


def cell_sdf_to_alpha(sdf, gradient, direction, depths, edges, inv_s, *, cosine_anneal=1.0):
    """NeuS endpoint estimate around each sample, including the terminal cell.

    Uses the published descending-CDF law with stable log-CDF arithmetic. No
    epsilon numerator creates opacity in empty/constant cells. ``edges`` also
    supports off-centre stratified samples, without moving their emission site.
    """
    if edges.shape != (*sdf.shape[:-1], sdf.shape[-1] + 1):
        raise ValueError("cell edges must bracket every depth sample")
    if not 0 <= cosine_anneal <= 1:
        raise ValueError("cosine_anneal must be in [0,1]")
    if (not bool(torch.isfinite(edges).all()) or bool((edges.diff(dim=-1) <= 0).any())
            or bool((depths < edges[..., :-1]).any()) or bool((depths > edges[..., 1:]).any())):
        raise ValueError("depth samples must lie in finite, positive-width cells")
    true_cos = (gradient * direction).sum(-1)
    slope = -(F.relu(.5 - .5 * true_cos) * (1 - cosine_anneal)
              + F.relu(-true_cos) * cosine_anneal)
    previous = sdf + slope * (edges[..., :-1] - depths).to(sdf)
    following = sdf + slope * (edges[..., 1:] - depths).to(sdf)
    log_ratio = F.logsigmoid(following * inv_s) - F.logsigmoid(previous * inv_s)
    return -torch.expm1(log_ratio.clamp_max(0))


def boundary_start_points(antenna, points, extent):
    """Real rays begin at their scene entry, never at a remote SDF query."""
    direction = points - antenna
    near, _, valid = ray_box_intervals(antenna, direction, extent)
    if not bool(valid.all()) or bool((near > 1 + 1e-8).any()):
        raise ValueError("real ray does not enter the scene before its sample")
    return (antenna + near[..., None] * direction).clamp(-extent, extent)


def render_volume(model, samples, tx_positions, rx_positions, *, lensless_correction=True,
                  detach_start_cdf=True, directional_exponent=1.0, create_graph=True,
                  reflectivity_override=None, alpha_override=None,
                  tx_amplitude_override=None, min_distance=1e-6):
    """Finite-scene, mixed-precision GeRaF amplitude rendering.

    Like the public renderer, boundary correction uses the aperture's mean
    antenna position. Here Tx and Rx have separate means and separate entry
    points. Only boundary CDFs use that approximation; phase and amplitudes
    retain every calibrated bistatic pair. No SDF is queried outside the AABB.
    """
    from rift.geraf import (GeRaFVolumeOutput, safe_normalize, transmittance_from_alpha,
                           shifted_lambertian_bistatic, free_space_amplitude_decay)
    points = samples.points
    if samples.depth_edges is None:
        raise ValueError("hardened GeRaF requires explicit primary cell edges")
    if bool((points.abs() > model.extent + 1e-7).any()):
        raise ValueError("hardened GeRaF samples must lie inside the scene AABB")
    parameter = next(model.sdf_network.parameters())
    query = points.reshape(-1, 3).to(parameter)
    sdf_flat, gradient_flat = model.sdf_network.gradient(query, create_graph=create_graph)
    r, z = points.shape[:2]
    sdf, gradient = sdf_flat.reshape(r, z), gradient_flat.reshape(r, z, 3)
    normals = safe_normalize(gradient)
    reflectivity = model.reflectivity_network(query).reshape(r, z)
    reflectivity = model._override_field(reflectivity_override, reflectivity, "reflectivity_override")
    inv_s = model.sdf_sharpness()
    alpha = cell_sdf_to_alpha(sdf, gradient, samples.primary_direction.to(gradient),
                              samples.depths, samples.depth_edges, inv_s)
    alpha = model._override_field(alpha_override, alpha, "alpha_override").clamp(0, 1)
    primary_t = transmittance_from_alpha(alpha)
    diagnostics = {}
    if lensless_correction:
        start = (samples.ray_origins + samples.depth_edges[:, :1] * samples.primary_direction).clamp(-model.extent, model.extent)
        tx_start = boundary_start_points(tx_positions.mean(0), points, model.extent)
        rx_start = boundary_start_points(rx_positions.mean(0), points, model.extent)
        # Boundary queries need no normal graph. Detach inv_s as well as SDF.
        def cdf_at(x):
            values = []
            for block in x.reshape(-1, 3).split(4096):
                values.append(torch.sigmoid(model.sdf_network(block.to(parameter)) * inv_s))
            return torch.cat(values).reshape(x.shape[:-1])
        with torch.set_grad_enabled(torch.is_grad_enabled() and not detach_start_cdf):
            primary_cdf = cdf_at(start)[:, None]
            tx_cdf, rx_cdf = cdf_at(tx_start), cdf_at(rx_start)
        tx_unclamped, rx_unclamped = primary_t - primary_cdf + tx_cdf, primary_t - primary_cdf + rx_cdf
        two_way = (tx_unclamped.clamp(0, 1) * rx_unclamped.clamp(0, 1)).unsqueeze(0)
        diagnostics = {
            "primary_start_cdf_min": float(primary_cdf.detach().min()),
            "primary_start_cdf_max": float(primary_cdf.detach().max()),
            "real_start_cdf_min": float(torch.minimum(tx_cdf.min(), rx_cdf.min()).detach()),
            "real_start_cdf_max": float(torch.maximum(tx_cdf.max(), rx_cdf.max()).detach()),
            "boundary_clamped_fraction": float((((tx_unclamped < 0) | (tx_unclamped > 1)).float().mean()
                 + ((rx_unclamped < 0) | (rx_unclamped > 1)).float().mean()).detach() * .5),
        }
    else:
        two_way = primary_t.square().unsqueeze(0)
    tx, rx = tx_positions.to(points)[:, None, None], rx_positions.to(points)[:, None, None]
    directional = shifted_lambertian_bistatic(points[None], normals.to(points)[None], tx, rx,
                                             exponent=directional_exponent)
    decay = free_space_amplitude_decay(points[None], tx, rx, min_distance=min_distance)
    amplitude = model.tx_amplitude()
    amplitude = model._override_field(tx_amplitude_override, amplitude, "tx_amplitude_override")
    amplitudes = amplitude * reflectivity[None] * directional * decay * two_way * alpha[None]
    diagnostics.update({
        "sdf_abs_mean_m": float(sdf.detach().abs().mean()),
        "sdf_negative_fraction": float((sdf.detach() < 0).float().mean()),
        "sdf_gradient_norm_mean": float(gradient.detach().norm(dim=-1).mean()),
        "alpha_mean": float(alpha.detach().mean()),
        "terminal_alpha_mean": float(alpha[:, -1].detach().mean()),
        "reflection_active_fraction": float((directional.detach() > 0).float().mean()),
        "reflectivity_mean": float(reflectivity.detach().mean()),
        "inverse_s_per_m": float(inv_s.detach()),
        "transmit_amplitude": float(amplitude.detach()),
        "render_samples": int(points.numel() // 3),
    })
    return GeRaFVolumeOutput(points, sdf, gradient, normals, reflectivity, alpha, primary_t,
                             two_way.expand(tx_positions.shape[0], -1, -1), directional,
                             decay, amplitudes, tx_positions, rx_positions, diagnostics)


def measured_ray_mask(current_magnitude, accumulated_magnitude, *, current_fraction=.05,
                      accumulated_fraction=.1):
    """Upstream ray-level mask, driven entirely by measured training data."""
    if current_magnitude.shape != accumulated_magnitude.shape or current_magnitude.ndim < 2:
        raise ValueError("measured mask inputs must have matching ray/depth shapes")
    for value in (current_magnitude, accumulated_magnitude):
        if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
            raise ValueError("measured mask values must be finite and nonnegative")
    if not (0 <= current_fraction <= 1 and 0 <= accumulated_fraction <= 1):
        raise ValueError("measured mask fractions must be in [0,1]")
    current_high = (current_magnitude > current_magnitude.max() * current_fraction).any(-1)
    accumulated_high = (accumulated_magnitude > accumulated_magnitude.max() * accumulated_fraction).any(-1)
    valid = (~(accumulated_high & ~current_high))[..., None].expand_as(current_magnitude)
    if not bool(valid.any()):
        raise RuntimeError("measured loss mask rejected every ray; refuse optimizer update")
    return valid


def _grid_corners(points, extent, size):
    if not bool(torch.isfinite(points).all()):
        raise ValueError("measured reference points must be finite")
    coordinates = (points / extent + 1) * ((size - 1) / 2)
    inside = (points.abs() <= extent + 1e-7).all(-1)
    coordinates = coordinates.clamp(0, size - 1)
    low = coordinates.floor().long().clamp_max(size - 2)
    fraction = coordinates - low
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                offset = low.new_tensor([dx, dy, dz])
                index = low + offset
                weight = torch.where(offset.bool(), fraction, 1 - fraction).prod(-1) * inside
                yield (index[..., 0] * size + index[..., 1]) * size + index[..., 2], weight


class MeasuredMaskReference:
    """Train-only common-world MF accumulation, independent of prediction.

    The local caches contain rotated per-view MF grids, rather than upstream's
    pre-registered accumulated volume. Trilinear deposition and sampling align
    these measured magnitudes in metric world coordinates. This is target
    preprocessing, not interpolation of the learned scattering field.
    """
    def __init__(self, extent, size, identity: Mapping):
        if size < 2 or not math.isfinite(extent) or extent <= 0:
            raise ValueError("invalid measured reference grid")
        self.extent, self.size = float(extent), int(size)
        self.identity = json.loads(json.dumps(identity, sort_keys=True))
        self.total = torch.zeros(size ** 3, dtype=torch.float64)
        self.mass = torch.zeros_like(self.total)
        self.visited = set()

    def add(self, view_index, points, magnitude):
        index = int(view_index)
        if index not in self.identity["train_indices"] or index in self.visited:
            raise ValueError("reference accepts each registered training view exactly once")
        points = points.detach().cpu().double().reshape(-1, 3)
        values = magnitude.detach().cpu().double().reshape(-1)
        if len(points) != len(values) or not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
            raise ValueError("invalid reference magnitude")
        for cell, weight in _grid_corners(points, self.extent, self.size):
            self.total.scatter_add_(0, cell, weight * values)
            self.mass.scatter_add_(0, cell, weight)
        self.visited.add(index)

    def finalize(self):
        if self.visited != set(self.identity["train_indices"]):
            raise ValueError("measured reference lacks full training-role coverage")
        self.volume = (self.total / self.mass.clamp_min(1e-30)).float()
        if not bool(torch.isfinite(self.volume).all()) or not bool(self.volume.max() > 0):
            raise ValueError("measured reference has no finite positive in-scene evidence")
        return self

    def sample(self, points):
        values = torch.zeros(points.shape[:-1], device=points.device, dtype=points.dtype)
        volume = self.volume.to(points)
        for cell, weight in _grid_corners(points, self.extent, self.size):
            values += volume[cell] * weight
        return values

    def digest(self):
        h = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode())
        h.update(str((self.extent, self.size)).encode())
        h.update(self.volume.detach().cpu().numpy().tobytes())
        return h.hexdigest()


class MeasuredViewMaskBank:
    """Immutable measured mask reference plus strict source/sampling identity."""
    def __init__(self, train_indices, shape, *, extent, size, identity,
                 current_fraction=.05, accumulated_fraction=.1):
        self.allowed = frozenset(map(int, train_indices))
        self.shape = tuple(shape)
        self.current_fraction = float(current_fraction)
        self.accumulated_fraction = float(accumulated_fraction)
        self.reference = MeasuredMaskReference(extent, size, {
            "schema": "rift_geraf_train_measured_mask_v1",
            "train_indices": sorted(self.allowed), "source_and_sampling": identity,
        })
        self.histories = {}  # Legacy checkpoint interface; no prediction history.

    def valid_mask(self, view_index, current_magnitude, *, points):
        if int(view_index) not in self.allowed:
            raise ValueError("measured mask may only be used for a training view")
        if tuple(current_magnitude.shape) != self.shape:
            raise ValueError("measured mask target grid changed")
        accumulated = self.reference.sample(points).reshape(self.shape).to(current_magnitude)
        return measured_ray_mask(current_magnitude.detach(), accumulated,
                                 current_fraction=self.current_fraction,
                                 accumulated_fraction=self.accumulated_fraction)

    def state_dict(self):
        return {
            "schema": "rift_geraf_measured_mask_bank_v1", "shape": self.shape,
            "high_threshold": self.current_fraction, "low_ratio": self.accumulated_fraction,
            "low_threshold": 0.0, "histories": {},
            "reference_identity": self.reference.identity, "extent": self.reference.extent,
            "size": self.reference.size, "volume": self.reference.volume.clone(),
            "reference_digest": self.reference.digest(),
        }

    def load_state_dict(self, state):
        expected = ("rift_geraf_measured_mask_bank_v1", self.shape, self.current_fraction,
                    self.accumulated_fraction, self.reference.extent, self.reference.size,
                    self.reference.identity)
        actual = (state.get("schema"), tuple(state.get("shape", ())), state.get("high_threshold"),
                  state.get("low_ratio"), state.get("extent"), state.get("size"),
                  state.get("reference_identity"))
        if actual != expected or state.get("histories") != {}:
            raise ValueError("measured mask reference source/recipe identity mismatch")
        volume = state.get("volume")
        if (not torch.is_tensor(volume) or volume.dtype != torch.float32
                or volume.shape != (self.reference.size ** 3,)
                or not bool(torch.isfinite(volume).all()) or bool((volume < 0).any())
                or not bool(volume.max() > 0)):
            raise ValueError("invalid measured mask reference volume")
        candidate = MeasuredMaskReference(self.reference.extent, self.reference.size, self.reference.identity)
        candidate.volume = volume.detach().cpu().clone()
        if state.get("reference_digest") != candidate.digest():
            raise ValueError("measured mask reference digest mismatch")
        self.reference.volume = candidate.volume


def extract_zero_surface(sdf_network, extent, *, grid=48, chunk=16384):
    """Extract the native metric SDF zero set without an oracle isolevel shift.

    Disconnected components remain valid output. A missing sign change is a
    reported failure, not a normalized proxy surface. Boundary contact is
    reported rather than silently closing/cropping the surface.
    """
    import numpy as np
    from skimage.measure import marching_cubes
    if grid < 3 or chunk < 1 or not math.isfinite(extent) or extent <= 0:
        raise ValueError("invalid SDF extraction grid/chunk/extent")
    parameter = next(sdf_network.parameters())
    axis = torch.linspace(-extent, extent, grid, dtype=parameter.dtype, device=parameter.device)
    points = torch.cartesian_prod(axis, axis, axis)
    values = []
    with torch.no_grad():
        for block in points.split(chunk):
            values.append(sdf_network(block).detach().cpu())
    field = torch.cat(values).reshape(grid, grid, grid).numpy()
    if not np.isfinite(field).all():
        raise ValueError("non-finite SDF values at extraction grid")
    if not field.min() < 0 < field.max():
        raise ValueError("SDF has no resolved zero crossing inside the scene")
    spacing = 2 * extent / (grid - 1)
    vertices, faces, _, _ = marching_cubes(field, level=0., spacing=(spacing,) * 3,
                                          allow_degenerate=False)
    vertices -= extent
    on_boundary = np.any(np.isclose(np.abs(vertices), extent, atol=spacing * 1e-4), axis=1)
    report = {"readout": "native_sdf_zero_surface", "isolevel_m": 0.0,
              "extent_m": extent, "grid": grid, "spacing_m": spacing,
              "sdf_min_m": float(field.min()), "sdf_max_m": float(field.max()),
              "vertices": len(vertices), "faces": len(faces),
              "boundary_vertex_fraction": float(on_boundary.mean()),
              "components_filtered": False, "posthoc_alignment": False}
    return vertices, faces, report
