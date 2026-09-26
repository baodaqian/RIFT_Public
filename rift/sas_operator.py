"""Differentiable bistatic SAS ellipsoid renderer.

This is a clean-room, vectorized implementation of the measurement contract in
Reed et al.'s public ``sampling.py``/``forward_model.py`` release.  It samples
rays inside the transmitter cone, intersects them with constant time-of-flight
ellipsoids, evaluates a complex scattering field, and integrates the rays with
Lambertian and two-way transmittance factors.

The implementation deliberately keeps the renderer independent of the scene
representation.  RIFT-SAS and the independent SH-SAS model therefore see the
same points, directions, visibility, and loss.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.utils.checkpoint


def normalize(v: torch.Tensor, eps: float = 1.0e-9) -> torch.Tensor:
    return v / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(eps)


def _skew(v: torch.Tensor) -> torch.Tensor:
    z = v.new_zeros(())
    return torch.stack(
        (z, -v[2], v[1], v[2], z, -v[0], -v[1], v[0], z)
    ).reshape(3, 3)


def rotation_a_to_b(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Rodrigues rotation taking unit vector ``a`` to unit vector ``b``."""
    a = normalize(a.reshape(1, 3))[0]
    b = normalize(b.reshape(1, 3))[0]
    cosine = torch.dot(a, b)
    if float(cosine.detach()) < -0.9999:
        # Pick a stable axis perpendicular to a for the pi rotation.
        seed = a.new_tensor([1.0, 0.0, 0.0])
        if float(a[0].abs().detach()) > 0.9:
            seed = a.new_tensor([0.0, 1.0, 0.0])
        axis = normalize(torch.cross(a, seed, dim=0).reshape(1, 3))[0]
        return 2.0 * axis[:, None] * axis[None, :] - torch.eye(3, device=a.device, dtype=a.dtype)
    cross = torch.cross(a, b, dim=0)
    cross_matrix = _skew(cross)
    identity = torch.eye(3, device=a.device, dtype=a.dtype)
    return identity + cross_matrix + (cross_matrix @ cross_matrix) / (1.0 + cosine).clamp_min(1.0e-8)


def cone_rays(
    num_rays: int,
    beamwidth_deg: torch.Tensor | float,
    max_distance: torch.Tensor | float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Reed-compatible deterministic disk grid, normalized to unit rays."""
    if num_rays < 1:
        raise ValueError("num_rays must be positive")
    side = int(math.sqrt(num_rays))
    if side * side != num_rays:
        raise ValueError("num_rays must be a perfect square")
    bw = torch.as_tensor(beamwidth_deg, device=device, dtype=dtype)
    distance = torch.as_tensor(max_distance, device=device, dtype=dtype)
    radius = distance * torch.tan(torch.deg2rad(bw) / 2.0)
    steps = int(math.ceil(math.sqrt(4.0 / math.pi) * side))
    x = torch.linspace(-1.0, 1.0, steps, device=device, dtype=dtype) * radius
    y = torch.linspace(-1.0, 1.0, steps, device=device, dtype=dtype) * radius
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    keep = xx.square() + yy.square() <= radius.square()
    rays = torch.stack((xx[keep], yy[keep], torch.ones_like(xx[keep]) * distance), dim=-1)
    # Upstream calls this argument a target count but the circular crop can
    # return a nearby number.  Preserve that behavior rather than re-sampling.
    return normalize(rays)


def _beamwidth_from_bounds(
    tx_pos: torch.Tensor, tx_direction: torch.Tensor, corners: torch.Tensor
) -> torch.Tensor:
    boundary = normalize(corners - tx_pos.reshape(1, 3))
    cosine = (boundary * tx_direction.reshape(1, 3)).sum(dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return 2.0 * torch.rad2deg(torch.acos(cosine).abs().max())


def ellipsoid_samples(
    radii: torch.Tensor,
    tx_pos: torch.Tensor,
    rx_pos: torch.Tensor,
    corners: torch.Tensor,
    *,
    num_rays: int,
    tx_direction: Optional[torch.Tensor] = None,
    beamwidth_deg: Optional[float] = None,
    point_at_center: bool = True,
    transmit_from_tx: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``[num_bins,num_rays_actual,3]`` samples and world ray directions."""
    if radii.ndim != 1 or radii.numel() < 1:
        raise ValueError("radii must be a non-empty vector")
    tx_pos = tx_pos.reshape(3).to(dtype=torch.float32)
    rx_pos = rx_pos.reshape(3).to(device=tx_pos.device, dtype=tx_pos.dtype)
    corners = corners.to(device=tx_pos.device, dtype=tx_pos.dtype)
    radii = radii.to(device=tx_pos.device, dtype=tx_pos.dtype)
    phase_center = (tx_pos + rx_pos) / 2.0
    separation = torch.linalg.vector_norm(tx_pos - rx_pos)
    if float(separation.detach()) < 1.0e-8:
        tx_axis_world = tx_pos.new_tensor([1.0, 0.0, 0.0])
    else:
        tx_axis_world = normalize((tx_pos - phase_center).reshape(1, 3))[0]
    tx_origin = torch.stack((separation / 2.0, separation.new_zeros(()), separation.new_zeros(())))

    if tx_direction is None:
        direction_world = corners.mean(dim=0) - tx_pos
        if not point_at_center:
            direction_world = direction_world.clone()
            direction_world[2] = tx_pos[2]
    else:
        direction_world = tx_direction.reshape(3).to(tx_pos)
    direction_world = normalize(direction_world.reshape(1, 3))[0]

    world_to_origin = rotation_a_to_b(tx_axis_world, tx_pos.new_tensor([1.0, 0.0, 0.0]))
    direction_origin = world_to_origin @ direction_world
    if beamwidth_deg is None:
        beamwidth = _beamwidth_from_bounds(tx_pos, direction_world, corners)
    else:
        beamwidth = tx_pos.new_tensor(float(beamwidth_deg))
    max_distance = radii.max()
    rays = cone_rays(
        num_rays,
        beamwidth,
        max_distance,
        device=tx_pos.device,
        dtype=tx_pos.dtype,
    )
    align_cone = rotation_a_to_b(tx_pos.new_tensor([0.0, 0.0, 1.0]), direction_origin)
    rays_origin = (align_cone @ rays.T).T

    semi_major = radii / 2.0
    semi_minor_sq = (semi_major.square() - separation.square() / 4.0).clamp_min(1.0e-12)
    semi_minor = torch.sqrt(semi_minor_sq)
    ray_origin = tx_origin if transmit_from_tx else torch.zeros_like(tx_origin)
    a = semi_major[:, None]
    b = semi_minor[:, None]
    direction = rays_origin[None, :, :]
    origin = ray_origin.reshape(1, 1, 3)
    alpha = direction[..., 0].square() / a.square() + (
        direction[..., 1].square() + direction[..., 2].square()
    ) / b.square()
    beta = 2.0 * (
        origin[..., 0] * direction[..., 0] / a.square()
        + (origin[..., 1] * direction[..., 1] + origin[..., 2] * direction[..., 2]) / b.square()
    )
    kappa = origin[..., 0].square() / a.square() + (
        origin[..., 1].square() + origin[..., 2].square()
    ) / b.square() - 1.0
    discriminant = (beta.square() - 4.0 * alpha * kappa).clamp_min(0.0)
    distance = (-beta + torch.sqrt(discriminant)) / (2.0 * alpha).clamp_min(1.0e-12)
    points_origin = origin + distance[..., None] * direction

    origin_to_world = rotation_a_to_b(tx_pos.new_tensor([1.0, 0.0, 0.0]), tx_axis_world)
    points_world = (origin_to_world @ points_origin.reshape(-1, 3).T).T
    points_world = points_world.reshape(points_origin.shape) + phase_center
    directions_world = (origin_to_world @ rays_origin.T).T
    return points_world, directions_world


def exclusive_cumprod(values: torch.Tensor, dim: int = 0) -> torch.Tensor:
    cumulative = torch.cumprod(values, dim=dim)
    shifted = torch.roll(cumulative, 1, dims=dim)
    index = [slice(None)] * values.ndim
    index[dim] = 0
    shifted[tuple(index)] = 1.0
    return shifted


def two_way_transmittance(
    radii: torch.Tensor,
    density: torch.Tensor,
    opacity_scale: float,
    *,
    mean_normalize: bool = False,
) -> torch.Tensor:
    """Approximate two-way visibility used by both comparison arms.

    The factor of two is the existing Reed-compatible approximation: it doubles
    the cumulative integral along the Tx-origin sampling sequence rather than
    integrating separate Tx and Rx paths.  ``mean_normalize`` is an explicit
    common adaptation that removes a global scene-scale factor before applying
    the opacity scale; the literal unnormalized path remains available with
    ``False``.
    """
    if density.ndim != 2 or density.shape[0] != radii.numel():
        raise ValueError("density must have shape [num_bins,num_rays]")
    if radii.numel() == 1:
        delta = torch.ones_like(radii)
    else:
        delta = radii[1:] - radii[:-1]
        delta = torch.cat((delta, delta.mean().reshape(1)))
    density = density.abs()
    if mean_normalize:
        density = density / density.mean().clamp_min(1.0e-12)
    alpha = torch.exp(-2.0 * float(opacity_scale) * density * delta[:, None])
    return exclusive_cumprod(alpha + 1.0e-10, dim=0)


def _render_sas_ray_chunk(
    field,
    radii: torch.Tensor,
    points: torch.Tensor,
    directions: torch.Tensor,
    sh_directions: torch.Tensor,
    selected: torch.Tensor,
    *,
    normal_step: float,
    opacity_scale: float,
    lambertian_ratio: float,
    full_ordered_selection: bool,
    probe_next_band: bool,
) -> Tuple[torch.Tensor, ...]:
    """Render one ray slice while retaining the complete radial context."""

    selected_points = points[selected]
    selected_directions = directions[selected]
    selected_sh_directions = sh_directions[selected]
    if full_ordered_selection:
        values = field.query_sas(
            selected_points.reshape(-1, 3),
            selected_sh_directions.reshape(-1, 3),
            normal_step=normal_step,
            probe_next_band=probe_next_band,
        )
        density = values["density"].reshape(points.shape[0], points.shape[1])
        scatterer = values["scatterer"].reshape(points.shape[0], points.shape[1])
        normals = values["normals"].reshape(points.shape[0], points.shape[1], 3)
    else:
        density = field.query_density(points.reshape(-1, 3)).reshape(points.shape[0], points.shape[1])
        values = field.query_sas(
            selected_points.reshape(-1, 3),
            selected_sh_directions.reshape(-1, 3),
            normal_step=normal_step,
            probe_next_band=probe_next_band,
        )
        scatterer = values["scatterer"].reshape(selected.numel(), points.shape[1])
        normals = values["normals"].reshape(selected.numel(), points.shape[1], 3)

    incidence = (normals * (-selected_directions)).sum(dim=-1).clamp_min(0.0)
    lambertian = float(lambertian_ratio) + (1.0 - float(lambertian_ratio)) * incidence
    transmission = two_way_transmittance(radii, density, opacity_scale, mean_normalize=False)
    estimated = (
        scatterer
        * lambertian.to(scatterer.dtype)
        * transmission[selected].to(scatterer.dtype)
    ).sum(dim=-1)
    context_estimated = torch.zeros(
        points.shape[0], device=estimated.device, dtype=estimated.dtype
    )
    context_estimated[selected] = estimated.detach()
    return (
        estimated,
        density.detach(),
        scatterer.detach(),
        normals.detach(),
        lambertian.detach(),
        transmission.detach(),
        context_estimated,
    )


def _render_sas_bins_ray_chunked(
    field,
    radii: torch.Tensor,
    points: torch.Tensor,
    directions: torch.Tensor,
    sh_directions: torch.Tensor,
    selected: torch.Tensor,
    *,
    ray_chunk: int,
    normal_step: float,
    opacity_scale: float,
    lambertian_ratio: float,
    full_ordered_selection: bool,
    probe_next_band: bool,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    predictions = []
    points_aux = []
    directions_aux = []
    scatterer_aux = []
    density_aux = []
    normals_aux = []
    lambertian_aux = []
    transmittance_aux = []
    context_aux = []
    actual_rays = int(points.shape[1])
    for start in range(0, actual_rays, ray_chunk):
        stop = min(start + ray_chunk, actual_rays)
        chunk_args = (
            field,
            radii,
            points[:, start:stop],
            directions[:, start:stop],
            sh_directions[:, start:stop],
            selected,
        )
        kwargs = {
            "normal_step": normal_step,
            "opacity_scale": opacity_scale,
            "lambertian_ratio": lambertian_ratio,
            "full_ordered_selection": full_ordered_selection,
            "probe_next_band": probe_next_band,
        }
        if torch.is_grad_enabled():
            chunk = torch.utils.checkpoint.checkpoint(
                _render_sas_ray_chunk, *chunk_args, use_reentrant=False, **kwargs
            )
        else:
            chunk = _render_sas_ray_chunk(*chunk_args, **kwargs)
        predictions.append(chunk[0])
        points_aux.append(points[:, start:stop].detach())
        directions_aux.append(directions[:, start:stop].detach())
        scatterer_aux.append(chunk[2])
        density_aux.append(chunk[1])
        normals_aux.append(chunk[3])
        lambertian_aux.append(chunk[4])
        transmittance_aux.append(chunk[5])
        context_aux.append(chunk[6])

    prediction = predictions[0]
    for chunk_prediction in predictions[1:]:
        prediction = prediction + chunk_prediction
    return prediction, {
        # Chunked diagnostic tensors are deliberately detached; only the
        # globally summed complex prediction retains an autograd graph.
        "points": torch.cat(points_aux, dim=1),
        "directions": torch.cat(directions_aux, dim=1),
        "scatterer": torch.cat(scatterer_aux, dim=1),
        "density": torch.cat(density_aux, dim=1),
        "normals": torch.cat(normals_aux, dim=1),
        "lambertian": torch.cat(lambertian_aux, dim=1),
        "transmittance": torch.cat(transmittance_aux, dim=1),
        "actual_rays": torch.tensor(actual_rays, device=points.device),
        "context_estimated": torch.stack(context_aux, dim=0).sum(dim=0),
        "output_bin_indices": selected,
        "sh_directions": sh_directions.detach(),
        "transmittance_contract": "approximate_factor2_tx_origin",
    }


def render_sas_bins(
    field,
    radii: torch.Tensor,
    tx_pos: torch.Tensor,
    rx_pos: torch.Tensor,
    corners: torch.Tensor,
    *,
    num_rays: int,
    opacity_scale: float,
    lambertian_ratio: float = 0.0,
    normal_step: float = 0.0032,
    tx_direction: Optional[torch.Tensor] = None,
    beamwidth_deg: Optional[float] = None,
    point_at_center: bool = True,
    transmit_from_tx: bool = True,
    output_bin_indices: Optional[torch.Tensor] = None,
    mean_normalize_opacity: bool = False,
    sh_direction: str = "rx_to_point",
    probe_next_band: bool = False,
    ray_chunk: int = 0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Render selected ToF bins using one full visibility context.

    ``radii`` is the contiguous context vector used for ellipsoid sampling and
    transmittance.  ``output_bin_indices`` selects loss bins *after* the full
    context render, so sparse output evaluation does not alter visibility.
    SH direction is explicit: the new comparison contract uses Rx-to-point;
    ``tx_to_point`` is retained only as an explicit legacy option.
    """
    if sh_direction not in {"rx_to_point", "tx_to_point"}:
        raise ValueError("sh_direction must be 'rx_to_point' or 'tx_to_point'")
    if ray_chunk < 0:
        raise ValueError("ray_chunk must be nonnegative")
    if ray_chunk > 0 and mean_normalize_opacity:
        raise ValueError("ray-chunked rendering does not support mean-normalized opacity")
    points, ray_directions = ellipsoid_samples(
        radii,
        tx_pos,
        rx_pos,
        corners,
        num_rays=num_rays,
        tx_direction=tx_direction,
        beamwidth_deg=beamwidth_deg,
        point_at_center=point_at_center,
        transmit_from_tx=transmit_from_tx,
    )
    num_bins, actual_rays, _ = points.shape
    directions = ray_directions.reshape(1, actual_rays, 3).expand(num_bins, -1, -1)
    if sh_direction == "rx_to_point":
        sh_directions = normalize(points - rx_pos.reshape(1, 1, 3).to(points))
    else:
        sh_directions = directions
    if output_bin_indices is None:
        selected = torch.arange(num_bins, device=points.device, dtype=torch.long)
    else:
        raw_selected = torch.as_tensor(output_bin_indices, device=points.device)
        if raw_selected.ndim != 1 or raw_selected.numel() == 0:
            raise ValueError("output_bin_indices must be a nonempty 1-D subset of the context bins")
        if raw_selected.dtype == torch.bool or torch.is_complex(raw_selected):
            raise TypeError("output_bin_indices must contain integer indices")
        if torch.is_floating_point(raw_selected):
            if not bool(torch.isfinite(raw_selected).all()) or not bool((raw_selected == raw_selected.round()).all()):
                raise TypeError("output_bin_indices must contain integer-valued indices")
        selected = raw_selected.to(dtype=torch.long)
        if bool((selected < 0).any()) or bool((selected >= num_bins).any()):
            raise IndexError("output_bin_indices must be a valid 1-D subset of the context bins")
        if torch.unique(selected).numel() != selected.numel():
            raise ValueError("output_bin_indices must not contain duplicates")

    # Visibility needs all context samples, but SH scattering and finite-
    # difference normals are queried only at requested output bins.  This keeps
    # sparse loss sampling from multiplying the expensive normal work by the
    # full crop length while retaining exactly the same transmittance origin.
    selected_points = points[selected]
    selected_directions = directions[selected]
    selected_sh_directions = sh_directions[selected]
    full_ordered_selection = selected.numel() == num_bins and torch.equal(
        selected, torch.arange(num_bins, device=selected.device, dtype=selected.dtype)
    )
    if ray_chunk > 0:
        return _render_sas_bins_ray_chunked(
            field,
            radii,
            points,
            directions,
            sh_directions,
            selected,
            ray_chunk=ray_chunk,
            normal_step=normal_step,
            opacity_scale=opacity_scale,
            lambertian_ratio=lambertian_ratio,
            full_ordered_selection=full_ordered_selection,
            probe_next_band=probe_next_band,
        )
    if full_ordered_selection:
        values = field.query_sas(
            selected_points.reshape(-1, 3),
            selected_sh_directions.reshape(-1, 3),
            normal_step=normal_step,
            probe_next_band=probe_next_band,
        )
        density = values["density"].reshape(num_bins, actual_rays)
        scatterer = values["scatterer"].reshape(num_bins, actual_rays)
        normals = values["normals"].reshape(num_bins, actual_rays, 3)
    else:
        density = field.query_density(points.reshape(-1, 3)).reshape(num_bins, actual_rays)
        values = field.query_sas(
            selected_points.reshape(-1, 3),
            selected_sh_directions.reshape(-1, 3),
            normal_step=normal_step,
            probe_next_band=probe_next_band,
        )
        scatterer = values["scatterer"].reshape(selected.numel(), actual_rays)
        normals = values["normals"].reshape(selected.numel(), actual_rays, 3)
    incidence = (normals * (-selected_directions)).sum(dim=-1).clamp_min(0.0)
    lambertian = float(lambertian_ratio) + (1.0 - float(lambertian_ratio)) * incidence
    transmission = two_way_transmittance(
        radii, density, opacity_scale, mean_normalize=mean_normalize_opacity
    )
    integrand = scatterer * lambertian.to(scatterer.dtype) * transmission[selected].to(scatterer.dtype)
    estimated = integrand.sum(dim=-1)
    context_estimated = torch.zeros(num_bins, device=estimated.device, dtype=estimated.dtype)
    context_estimated[selected] = estimated
    return estimated, {
        "points": points,
        "directions": directions,
        "scatterer": scatterer,
        "density": density,
        "normals": normals,
        "lambertian": lambertian,
        "transmittance": transmission,
        "actual_rays": torch.tensor(actual_rays, device=points.device),
        "context_estimated": context_estimated,
        "output_bin_indices": selected,
        "sh_directions": sh_directions,
        "transmittance_contract": "approximate_factor2_tx_origin",
    }
