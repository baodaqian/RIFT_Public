"""Audited RF range-bin rendering and radar-only occupancy targets.

The reference computes noncoherent beam averages at sampled range-bin centres.
It does not synthesize coherent radar phase. Bistatic surfaces and a scene-cap
quadrature adapt that operator to RIFT; they are not the Navtech antenna model.
See external/RADAR_FIELDS_REFERENCE.md for the source/adapter boundary.
"""

from __future__ import annotations

import math
import numpy as np

import torch
import torch.nn.functional as F
from rift.radar_fields_upstream import original_module


def occupancy_probability(
    intensity, *, noise_axis="range", noise_multiplier=1.5,
    probability_offset=-0.15, probability_scale=2.0, decay_bins=10.0,
    implementation="upstream",
):
    """Local radar-only estimator, with the released helper's recurrence.

    Input is [profiles, bins] after fixed training-only dB normalization.
    ``azimuth_range`` reproduces the release's two median thresholds for an
    actual azimuth-resolved scan. ``range`` uses only each profile's radial
    median: MIMO pairs (and GOTCHA platform pulses) are NOT azimuth beams.
    Build the full profile before cropping/subsampling, so the target for a
    pair cannot change with the other pairs selected by an optimizer batch.

    The supplement uses 2*max(medians); the released helper uses 1.5 and >.
    Its recurrence decays the running evidence by distance from the last
    stronger return at EVERY bin; this is retained, not silently replaced
    with a different exponential envelope. With prior 0.5, Bayes(p, 0.5)=p.
    """
    if intensity.ndim != 2 or intensity.numel() == 0:
        raise ValueError("occupancy intensity must be nonempty [profiles,bins]")
    if noise_axis not in ("range", "azimuth_range"):
        raise ValueError("noise_axis must declare range or azimuth_range")
    if not all(math.isfinite(float(x)) for x in (noise_multiplier, probability_offset, probability_scale, decay_bins)):
        raise ValueError("occupancy controls must be finite")
    if noise_multiplier <= 0 or probability_scale <= 0 or decay_bins <= 0:
        raise ValueError("occupancy scale/decay controls must be positive")
    if not torch.isfinite(intensity).all() or (intensity < 0).any():
        raise ValueError("occupancy intensity must be finite and nonnegative")
    radial_floor = intensity.median(dim=-1, keepdim=True).values
    floor = radial_floor
    if noise_axis == "azimuth_range":
        floor = torch.maximum(radial_floor, intensity.median(dim=0, keepdim=True).values)
    filtered = intensity * (intensity > float(noise_multiplier) * floor)
    if implementation == "upstream":
        # The released helper uses NumPy and CPU tensors. Preserve it unchanged;
        # targets require no autograd, and only this bounded per-view array moves.
        result, _ = original_module("radarfields.radar").bayesian_polar_occupancy_map(
            filtered.detach().cpu(), (1, filtered.shape[1]), i_offset=probability_offset,
            i_exp=probability_scale, decay_bins=decay_bins,
        )
        return result.to(intensity)
    if implementation != "torch":
        raise ValueError("unknown occupancy implementation")
    # Limit the exponent before exp to keep externally supplied intensities safe.
    exponent = (float(probability_scale) * (filtered - float(probability_offset))).clamp(max=50)
    evidence = (filtered * exponent.exp()).clamp(0.0, 0.999)
    running = torch.zeros_like(evidence[:, 0])
    last_peak = torch.zeros_like(running)
    columns = []
    for index in range(evidence.shape[1]):
        running = running * torch.exp(-(index - last_peak) / float(decay_bins))
        stronger = evidence[:, index] > running
        last_peak = torch.where(stronger, float(index), last_peak)
        running = torch.maximum(running, evidence[:, index])
        columns.append(running)
    return torch.stack(columns, dim=-1)


def scene_cap_directions(origins, extent, sample_count):
    """Equal-solid-angle deterministic rays covering the cube's bounding sphere.

    This is an explicitly uniform-gain ROI aperture, not an inferred vendor
    radiation pattern. The physical cone depends only on acquisition/support
    metadata. Increasing sample_count refines quadrature without moving bins.
    """
    if origins.ndim != 2 or origins.shape[1] != 3 or sample_count < 1 or extent <= 0:
        raise ValueError("invalid scene-cap geometry")
    distance = torch.linalg.vector_norm(origins, dim=-1, keepdim=True)
    radius = math.sqrt(3.0) * float(extent)
    if not torch.isfinite(origins).all() or (distance <= radius).any():
        raise ValueError("scene-cap sensors must be finite and outside the support sphere")
    forward = -origins / distance
    axis_id = forward.abs().argmin(dim=-1)
    axis = F.one_hot(axis_id, num_classes=3).to(forward)
    right = F.normalize(torch.linalg.cross(forward, axis), dim=-1)
    up = torch.linalg.cross(right, forward)
    cos_edge = torch.sqrt(1.0 - (radius / distance).square())
    index = torch.arange(sample_count, device=origins.device, dtype=origins.dtype)
    cos_theta = 1.0 - ((index + 0.5) / sample_count)[None, :] * (1.0 - cos_edge)
    sin_theta = (1.0 - cos_theta.square()).clamp_min(0).sqrt()
    azimuth = index * (math.pi * (3.0 - math.sqrt(5.0)))
    return (forward[:, None, :] * cos_theta[..., None]
            + right[:, None, :] * (sin_theta * azimuth.cos())[..., None]
            + up[:, None, :] * (sin_theta * azimuth.sin())[..., None])


def bistatic_ray_points(tx, rx, directions, ranges):
    """Intersect Tx-origin rays with exact half-path ellipsoids in float64.

    tx/rx [P,3], directions [P,S,3], ranges [R] or [P,R].
    t=(4 R²-|rx-tx|²)/(4 R-2 d·(rx-tx)); x=tx+t*d.
    """
    tx, rx, directions, ranges = [x.to(dtype=torch.float64) for x in (tx, rx, directions, ranges)]
    if tx.shape != rx.shape or tx.ndim != 2 or tx.shape[-1] != 3:
        raise ValueError("tx/rx must have matching [pairs,3] shapes")
    if directions.ndim != 3 or directions.shape[0] != tx.shape[0] or directions.shape[-1] != 3:
        raise ValueError("directions must be [pairs,samples,3]")
    if ranges.ndim == 1:
        ranges = ranges[None, :].expand(tx.shape[0], -1)
    if ranges.ndim != 2 or ranges.shape[0] != tx.shape[0]:
        raise ValueError("ranges must be [bins] or [pairs,bins]")
    if not all(torch.isfinite(x).all() for x in (tx, rx, directions, ranges)):
        raise ValueError("nonfinite ray geometry")
    if not torch.allclose(directions.norm(dim=-1), torch.ones_like(directions[..., 0]), atol=1e-7, rtol=1e-7):
        raise ValueError("ray directions must be unit vectors")
    baseline = rx - tx
    if (ranges * 2 <= baseline.norm(dim=-1, keepdim=True)).any():
        raise ValueError("range surface must exceed the Tx/Rx baseline half-length")
    dot = (directions * baseline[:, None, :]).sum(dim=-1)
    twice_r = 2 * ranges[:, :, None]
    distance = (twice_r.square() - baseline.square().sum(dim=-1)[:, None, None]) / (2 * (twice_r - dot[:, None, :]))
    return tx[:, None, None, :] + distance[..., None] * directions[:, None, :, :]


def weighted_ray_mean(values, weights):
    """Reference beam average over the final (sample) axis.

    Empty-space samples carry zero values but keep their quadrature weight.
    Do not divide by the number of occupied/in-box samples.
    """
    weights = torch.broadcast_to(weights, values.shape)
    if not torch.isfinite(weights).all() or (weights < 0).any() or (weights.sum(dim=-1) <= 0).any():
        raise ValueError("ray weights must be finite, nonnegative and have positive mass")
    return (values * weights).sum(dim=-1) / weights.sum(dim=-1)


def prepare_bistatic_bins(tx, rx, ranges, *, extent, ray_samples, source_sampling=False, rotation=None):
    """Response-free geometry for one physical frame."""
    tx, rx = tx.double(), rx.double()
    directions = (released_scene_directions(tx, extent, ray_samples, rotation=rotation)
                  if source_sampling else scene_cap_directions(tx, extent, ray_samples))
    points = bistatic_ray_points(tx, rx, directions, ranges)
    inside = (points.abs() <= float(extent)).all(dim=-1)
    flat_inside = inside.reshape(-1)
    xyz = points.reshape(-1, 3)[flat_inside]
    view = directions[:, None, :, :].expand_as(points).reshape(-1, 3)[flat_inside]
    return {"inside": inside, "xyz": xyz, "view": view, "source_sampling": source_sampling}


def released_scene_directions(origins, extent, sample_count, *, rotation=None):
    """Original random pitch/yaw sampler; acquisition ROI replaces Navtech FOV.

    This aperture covers the registered support. It is NOT a measured antenna
    cutoff, nor an interpretation of the simulator's 10-degree HPBW as FOV.
    RIFT supplies the actual array axes. GOTCHA lacks attitude/beam metadata;
    a look-at-ROI frame is an explicit adapter convention there.
    """
    origins = origins.double()
    count = len(origins)
    if rotation is None:
        forward = F.normalize(-origins, dim=-1)
        up = torch.zeros_like(forward); up[:, 2] = 1
        alternate = forward[:, 2].abs() > .99
        up[alternate] = torch.tensor([0., 1., 0.], device=up.device, dtype=up.dtype)
        right = F.normalize(torch.linalg.cross(up, forward), dim=-1)
        rotation = torch.stack((forward, right, torch.linalg.cross(forward, right)), -1)
    rotation = torch.broadcast_to(rotation, (count, 3, 3)).double()
    distance = origins.norm(dim=-1)
    radius = math.sqrt(3.) * extent
    if (distance <= radius).any():
        raise ValueError("RF ROI cone requires an exterior sensor")
    # Account for an element displaced from the device's common boresight.
    off_axis = torch.acos((F.normalize(-origins, dim=-1) * rotation[..., 0]).sum(-1).clamp(-1, 1))
    half_angle = torch.asin(radius / distance) + off_axis
    # Each pair keeps its own fixed geometry-derived aperture. A minibatch's
    # other selected pairs must not change this pair's integration domain.
    opening = (2 * half_angle * 180 / math.pi).float()[:, None, None]
    poses = torch.eye(4, device=origins.device)[None].repeat(count, 1, 1)
    poses[:, :3, :3] = rotation.float()
    poses[:, :3, 3] = origins.float()
    rays = original_module("radarfields.sampler").get_radar_rays(
        poses, (opening, opening), 1, 1, sample_count,
        torch.zeros(count, 1, device=origins.device), origins.device)
    # Exact bistatic intersection requires unit vectors; FP32 sampler rotations
    # introduce only floating-point length error, removed in the FP64 geometry.
    return F.normalize(rays["directions"].double(), dim=-1)


def render_bistatic_batch(model, requests, *, query_chunk, mask_progress=1.0):
    """One neural query batch (and BN update) across all physical frames."""
    if not requests:
        raise ValueError("empty RF render batch")
    parameter = next(model.parameters())
    xyz = torch.cat([request["xyz"] for request in requests]).to(parameter)
    view = torch.cat([request["view"] for request in requests]).to(parameter)
    # Preserve a zero-gradient path for empty geometry so all-empty synthetic
    # batches still support backward without manufacturing occupied samples.
    empty = parameter.reshape(-1)[0] * 0.0
    if xyz.shape[0]:
        field = model.query_chunked(xyz, view, mask_progress=mask_progress, chunk_size=query_chunk)
    results, offset = [], 0
    for request in requests:
        inside = request["inside"]
        count = len(request["xyz"])
        result = {}
        for key in ("alpha", "rcs"):
            values = torch.zeros(inside.numel(), device=parameter.device, dtype=parameter.dtype) + empty
            if count:
                values = values.to(field[key]).masked_scatter(inside.reshape(-1), field[key][offset:offset+count])
            samples = values.reshape(inside.shape).transpose(1, 2)[..., None]
            radar = original_module("radarfields.radar")
            if request.get("source_sampling", False):
                # Even a unit-gain fallback uses the released LUT integration:
                # its FP32 weights promote native TCNN half outputs AFTER the
                # alpha*rd product, exactly as in Trainer.predict_waveform.
                offsets = torch.zeros(inside.shape[0], inside.shape[-1], 3)
                unit_lut = np.array([[-180., 1.], [180., 1.]])
                averaged = radar.integrate_rays_LUT(samples, offsets, inside.shape[-1],
                                                   unit_lut, unit_lut, parameter.device)
            else:
                averaged = radar.avg_rays(samples, inside.shape[-1])
            result[key] = averaged.reshape(inside.shape[:2])
        result["coverage"] = inside.to(parameter.dtype).mean(dim=-1)
        results.append(result)
        offset += count
    return results


def render_bistatic_bins(model, tx, rx, ranges, *, extent, ray_samples, query_chunk, mask_progress=1.0):
    """Render [pairs,bins] occupancy/RCS without a voxel-splat denominator."""
    request = prepare_bistatic_bins(tx, rx, ranges, extent=extent, ray_samples=ray_samples)
    return render_bistatic_batch(model, [request], query_chunk=query_chunk, mask_progress=mask_progress)[0]


def released_batch_loss(records, *, weight_fft=0.60, weight_occ=0.36, weight_bimodal=0.03, source_exact=False):
    """Released global distributions, explicit frame batchmean, and sample std.

    Records may have different ROI lengths. They contain prediction, target,
    occupancy, and occupancy_target tensors. There is no padded fake data in
    either the global normalization or mean intensity error. Empty/singleton
    std groups contribute zero instead of upstream's NaN; no NaN gradients.
    """
    if not records:
        raise ValueError("a Radar Fields batch cannot be empty")
    pred, target, alpha, occ = [torch.cat([r[key].reshape(-1) for r in records])
                               for key in ("prediction", "target", "occupancy", "occupancy_target")]
    if source_exact:
        # Same reductions as Trainer.compute_loss for ragged frame lengths.
        # Keep the release's nan_to_num behaviour, including undefined edge
        # gradients: a failure is reported, not silently repaired by this path.
        alpha = alpha.float()
        reference = occ / occ.sum()
        exterior = torch.cat([r.get("coverage", torch.ones_like(r["occupancy"])).reshape(-1) == 0
                              for r in records])
        if exterior.any():
            # Fixed exterior zeros are an acquisition-support constraint; the
            # released sigmoid has no such zeros. Remove only log(a_i=0), an
            # infinite MODEL-INDEPENDENT constant, retaining -log(sum alpha)
            # for every cell. This is the finite-part gradient of the source
            # KL as fixed exterior occupancy tends to zero; no epsilon, no
            # clipping/repair of learned in-support probabilities.
            log_probability = torch.where(exterior, torch.ones_like(alpha), alpha).log() - alpha.sum().log()
        else:
            log_probability = (alpha / alpha.sum()).log()
        kl = F.kl_div(log_probability, reference, reduction="sum") / len(records)
        mask = occ > .01
        bimodal = alpha[mask].std() + alpha[~mask].std()
        terms = {"fft": F.l1_loss(pred, target)*weight_fft,
                 "occupancy": kl*weight_occ, "bimodal": bimodal*weight_bimodal}
        return terms["fft"] + torch.nan_to_num(terms["occupancy"]) + torch.nan_to_num(terms["bimodal"]), terms
    pred_dist = alpha.clamp_min(1e-12)
    pred_dist = pred_dist / pred_dist.sum()
    target_dist = occ.clamp_min(0)
    target_dist = target_dist / target_dist.sum().clamp_min(1e-12)
    kl = F.kl_div(pred_dist.log(), target_dist, reduction="sum") / len(records)
    mask = occ > 0.01
    bimodal = alpha.sum() * 0
    for selected in (alpha[mask], alpha[~mask]):
        if selected.numel() > 1:
            bimodal = bimodal + selected.float().std(correction=1)
    terms = {"fft": F.l1_loss(pred, target) * weight_fft,
             "occupancy": kl * weight_occ, "bimodal": bimodal * weight_bimodal}
    return sum(terms.values()), terms
