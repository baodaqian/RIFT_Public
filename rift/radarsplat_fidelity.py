"""RadarSplat release conventions and disclosed spherical-acquisition adapters.

Reference: umautobots/radarsplat ea9c8f530c708622cc3b1b560436b5557ac6a49b.
See docs/RADARSPLAT_FIDELITY.md for paper/release differences and provenance.
No raw radar access, mesh supervision, or validation donors are used here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math

import numpy as np
from scipy.ndimage import map_coordinates
import torch
from torch import Tensor
import torch.nn.functional as F
from rift.vendor.radarsplat.preprocessing import apply_saturation_mask

PROFILE = "audit_v1"
UPSTREAM_COMMIT = "ea9c8f530c708622cc3b1b560436b5557ac6a49b"


def scene_support_angular_sampling(viewpoints, *, half_extent_m, n_azimuth, n_elevation, q=10):
    """Cover the origin-centred scene cube at every calibrated sensor pose.

    End sample centres cover the circumscribed sphere. This sets sampling,
    not sensor resolution; the acquisition PSF remains a separate contract.
    """
    positions = np.asarray(viewpoints, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions) or not np.isfinite(positions).all():
        raise ValueError("scene-support sampling requires finite sensor positions")
    if not math.isfinite(half_extent_m) or half_extent_m <= 0 or min(n_azimuth, n_elevation) < 2 or q < 1:
        raise ValueError("invalid scene-support sampling configuration")
    radius = math.sqrt(3.) * half_extent_m
    distance = float(np.linalg.norm(positions, axis=1).min())
    if distance <= radius:
        raise ValueError("scene-support sampling requires sensors outside the scene cube")
    span = 2 * math.degrees(math.asin(radius/distance))
    azimuth = span/(n_azimuth-1)
    return azimuth, span/(n_elevation-1), azimuth/q


def release_ssim_index(predicted: Tensor, target: Tensor) -> Tensor:
    """11x11, sigma=1.5, valid SSIM with fixed unit-range stabilizers.

    This is the mathematical CPU/PyTorch equivalent of the released trainer's
    fused_ssim(..., padding='valid'), not the historical adaptive box-window
    objective. Values above one remain unclipped. Small images fail explicitly.
    """
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("release SSIM requires matching 2-D images")
    if min(predicted.shape) < 11:
        raise ValueError("release SSIM requires at least 11 pixels on both axes")
    if torch.is_complex(predicted) or torch.is_complex(target):
        raise TypeError("release SSIM requires real power")
    if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
        raise ValueError("release SSIM requires finite power")
    x = predicted[None, None]
    y = target.to(predicted)[None, None]
    t = torch.arange(-5, 6, dtype=x.dtype, device=x.device)
    g = torch.exp(-0.5 * (t / 1.5).square())
    g = g / g.sum()
    window = (g[:, None] * g[None, :])[None, None]
    moments = F.conv2d(torch.cat((x, y, x*x, y*y, x*y)), window)
    mx, my, xx, yy, xy = moments.unbind(0)
    vx, vy = xx - mx.square(), yy - my.square()
    cov = xy - mx * my
    return (((2*mx*my + 0.01**2) * (2*cov + 0.03**2)) /
            ((mx.square() + my.square() + 0.01**2) * (vx + vy + 0.03**2))).mean()


@dataclass(frozen=True)
class OccupancyRecipe:
    window_views: int = 10
    power_threshold: float = 0.15
    smoothing_sigma_bins: float = 3.0
    saturation_ratio: float = 0.21
    multipath_ratio: float = 0.2
    multipath_fft_peak: float = 30.0
    skip_first_frequencies: int = 3

    def __post_init__(self):
        if not isinstance(self.window_views, int) or self.window_views < 1:
            raise ValueError("occupancy window must be a positive integer")
        if not isinstance(self.skip_first_frequencies, int) or self.skip_first_frequencies < 1:
            raise ValueError("FFT skip must be a positive integer")
        for value in (self.power_threshold, self.smoothing_sigma_bins,
                      self.saturation_ratio, self.multipath_ratio, self.multipath_fft_peak):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("occupancy recipe values must be finite and positive")

    def identity(self):
        return {
            "schema": "rift_radarsplat_train_multiview_occupancy_v1",
            **asdict(self),
            "donors": "nearest training sensor directions about scene centre; ties by view ID",
            "window_policy": "min(window_views, materialized train count); no validation donors",
            "normalization": "existing unclipped training peak; never fit on validation",
            "denoising": "release half-spectrum DC-inclusive detector and Gaussian decay region",
            "mapping": "3-D target samples reprojected into donor polar images; visible power mean",
            "projection": "threshold mean power, then any occupied elevation sample",
            "spatial_support": "registered scene cube; no mesh or fitted geometry",
            "unknown_policy": "outside donor polar/elevation coverage is unknown, not free",
            "upstream": UPSTREAM_COMMIT,
        }


def denoise_power(power: np.ndarray, recipe: OccupancyRecipe) -> tuple[np.ndarray, np.ndarray]:
    """Release detector + decay-mask preprocessing, independent of occupancy.

    Uses the executable's half-spectrum DC-inclusive ratio, threshold 30 and
    sigma 3 (the paper instead describes other constants). The crop/normalizer
    differ from Boreas; these constants are a disclosed development recipe.
    """
    values = np.asarray(power, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("denoising requires finite nonnegative 2-D power")
    spectrum = np.abs(np.fft.fft(values, axis=-1))[:, :values.shape[1]//2]
    ratio = np.divide(spectrum[:, 0], spectrum.sum(axis=1),
                      out=np.zeros(values.shape[0]), where=spectrum.sum(axis=1) > 0)
    remaining = spectrum[:, recipe.skip_first_frequencies:]
    peaks = remaining.max(axis=1) if remaining.shape[1] else np.zeros(values.shape[0])
    noisy = ((ratio > recipe.saturation_ratio) |
             ((ratio > recipe.multipath_ratio) & (peaks > recipe.multipath_fft_peak)))
    filtered, _ = apply_saturation_mask(values, noisy, recipe.smoothing_sigma_bins,
                                        occ_thres=recipe.power_threshold)
    return filtered, noisy


def polar_world_points(arrays):
    """All physical elevation/azimuth/range sample centres in world space."""
    el, az, r = np.meshgrid(arrays["elevation_rad"].astype(np.float64),
                           arrays["azimuth_rad"].astype(np.float64),
                           arrays["range_m"].astype(np.float64), indexing="ij")
    local = np.stack((r*np.cos(el)*np.cos(az), r*np.cos(el)*np.sin(az), r*np.sin(el)), axis=-1)
    pose = arrays["sensor_to_world"].astype(np.float64)
    return local.reshape(-1, 3) @ pose[:3, :3].T + pose[:3, 3]


def sample_polar_power(points, arrays, power):
    """Bilinear physical-bin sampling; no clamping of out-of-domain queries."""
    pose = arrays["sensor_to_world"].astype(np.float64)
    # Invert the stored transform: its float32 rotation is only approximately
    # orthogonal, and transpose-based round trips can move endpoint samples.
    local = np.linalg.solve(pose[:3, :3],
                            (np.asarray(points, dtype=np.float64) - pose[:3, 3]).T).T
    r = np.linalg.norm(local, axis=-1)
    az = np.arctan2(local[:, 1], local[:, 0])
    el = np.arctan2(local[:, 2], np.linalg.norm(local[:, :2], axis=-1))
    ranges = arrays["range_m"].astype(np.float64)
    azimuths = arrays["azimuth_rad"].astype(np.float64)
    elevations = arrays["elevation_rad"].astype(np.float64)
    az = azimuths.mean() + (az-azimuths.mean()+np.pi) % (2*np.pi) - np.pi
    # Invert the complete serialized axes. Extrapolating the first FP32 bin
    # difference accumulates quantization error across the crop (especially
    # millimetre bins at ten metres) and falsely drops the last valid bins.
    ri = np.interp(r, ranges, np.arange(len(ranges)))
    ai = np.interp(az, azimuths, np.arange(len(azimuths)))
    def covered(values, axis):
        tol = 2 * abs(float(np.spacing(np.float32(max(1e-3, np.abs(axis).max())))))
        return (values >= axis[0]-tol) & (values <= axis[-1]+tol)
    valid = covered(r, ranges) & covered(az, azimuths) & covered(el, elevations)
    samples = map_coordinates(power, [ai, ri], order=1, mode="nearest")
    return np.where(valid, samples, 0.0), valid


class TrainingOccupancy:
    """Deterministic train-only multiview labels for an already verified cache.

    The spherical benchmark has no temporal ground-plane sequence. The paper's
    local-window mean is lifted to 3-D samples, using nearest acquisition
    directions. This is explicitly an acquisition adaptation, not Boreas parity.
    Only small derived images are memoized; no raw payload is opened.
    """
    def __init__(self, cache, recipe: OccupancyRecipe = OccupancyRecipe(), *, intensity_mapping=None):
        from rift.radarsplat_release import LINEAR_INTENSITY
        self.cache, self.recipe = cache, recipe
        self.intensity_mapping = LINEAR_INTENSITY if intensity_mapping is None else intensity_mapping
        self.train = tuple(sorted(cache.train_indices))
        if not self.train:
            raise ValueError("occupancy mapping requires training donors")
        record = cache.acquisition_record
        lookup = {int(v): i for i, v in enumerate(record["view_indices"])}
        centre = np.asarray(cache.grid["scene_center_m"], dtype=np.float64)
        positions = np.asarray(record["viewpoint_positions"], dtype=np.float64)
        self.directions = {}
        for index in self.train:
            direction = positions[lookup[index]] - centre
            norm = np.linalg.norm(direction)
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError("occupancy donor lies at the scene centre")
            self.directions[index] = direction / norm
        self._labels = {}
        # Cache lifetime follows this provider, not a process-global bound method.
        self._donor = lru_cache(maxsize=32)(self._read_donor)

    def _read_donor(self, index):
        from rift.radarsplat_b7873200_protocol import load_target
        if index not in self.directions:
            raise ValueError("occupancy donor must belong to the training role")
        arrays = (self.cache.read_target(index, "train") if hasattr(self.cache, "read_target") else
                  load_target(self.cache.root, index, "train", expected_grid=self.cache.grid))
        from rift.radarsplat_release import intensity
        power, noisy = denoise_power(intensity(arrays["radarsplat_mf_power"], self.cache.train_peak_power,
                                               self.intensity_mapping), self.recipe)
        return arrays, power, noisy

    def label(self, index: int):
        if index not in self.directions:
            raise ValueError("training occupancy labels may only be requested for training views")
        if index in self._labels:
            return self._labels[index]
        target, _, _ = self._donor(index)
        donors = sorted(self.train, key=lambda j: (-float(self.directions[index] @ self.directions[j]), j))
        donors = donors[:self.recipe.window_views]
        points = polar_world_points(target)
        centre = np.asarray(self.cache.grid["scene_center_m"], dtype=np.float64)
        support = (np.abs(points-centre) <= float(self.cache.grid["scene_extent_m"]) + 1e-7).all(axis=1)
        total = np.zeros(len(points), dtype=np.float64)
        counts = np.zeros(len(points), dtype=np.int32)
        noisy_count = 0
        for donor in donors:
            arrays, power, noisy = self._donor(donor)
            values, visible = sample_polar_power(points, arrays, power)
            visible &= support
            total += np.where(visible, values, 0.0)
            counts += visible
            noisy_count += int(noisy.sum())
        average = np.divide(total, counts, out=np.zeros_like(total), where=counts > 0)
        shape = (len(target["elevation_rad"]), len(target["azimuth_rad"]), len(target["range_m"]))
        mask = ((counts > 0) & (average >= self.recipe.power_threshold)).reshape(shape).any(axis=0)
        mask.setflags(write=False)
        diagnostics = {"donor_view_ids": donors, "positive_fraction": float(mask.mean()),
                       "empty_mask": not bool(mask.any()), "covered_fraction": float((counts > 0).mean()),
                       "noisy_donor_beams": noisy_count}
        self._labels[index] = (mask, diagnostics)
        return mask, diagnostics


@torch.no_grad()
def branch_diagnostics(rendered, cutoff: float):
    """Per-product cutoff upper bounds and actually rendered branch energies."""
    power = rendered["gaussian_scene_power"]
    occ = rendered["gaussian_opacity"]
    # Peak attribute above cutoff is necessary, not sufficient, for a pixel hit.
    return {
        "power_peak_eligible_occupancy_below_cutoff": int(((power >= cutoff) & (occ < cutoff)).sum()),
        "rendered_clean_power_sum": float(rendered["clean_power"].sum().cpu()),
        "rendered_noise_power_sum": float(rendered["noise_power"].sum().cpu()),
        "rendered_occupancy_max": float(rendered["occupancy"].max().cpu()),
    }


def loss_branch_gradients(losses, model):
    """Separate native data/occupancy gradients without accumulating .grad."""
    names = ("means", "opacity_logits", "noise_probability_logits", "sh0")
    parameters = tuple(getattr(model, name) for name in names)
    result = {}
    for branch in ("power_l1", "occupancy_l1", "opacity_noise"):
        gradients = (torch.autograd.grad(losses[branch], parameters, retain_graph=True, allow_unused=True)
                     if losses[branch].requires_grad else (None,) * len(parameters))
        result[branch] = {name: 0. if grad is None else float(grad.detach().abs().sum().cpu())
                          for name, grad in zip(names, gradients)}
    return result
