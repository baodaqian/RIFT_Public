"""Native-power adapter and optimization helpers for RadarSplat B7873200.

The historical RadarSplat trainer is intentionally left untouched.  This
module is the narrow bridge from a sealed, train/validation-only B787 cache to
the reconciled Gaussian renderer.  It has no raw-response loader and exposes
only real native power, occupancy, and Gaussian geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from rift.radarsplat_b7873200 import RadarSplatEffects, RadarSplatGrid, RadarSplatModel
from rift.radarsplat_b7873200_protocol import (
    RadarSplatB7873200Cache,
    atomic_save_npz,
    load_target,
)


OCCUPANCY_THRESHOLD = 0.001
PRUNE_OPACITY = 0.0005
GEOMETRY_SCHEMA = "rift_radarsplat_b7873200_gaussian_occupancy_v1"
_PARAMETER_NAMES = (
    "means",
    "log_scales",
    "quaternions",
    "opacity_logits",
    "noise_probability_logits",
    "sh0",
    "shN",
)


def _float32_coordinate_tolerance(values: np.ndarray) -> float:
    """Bound serialization error for target centres stored as IEEE float32.

    B787 range centres are near 10 m, where one float32 ulp is about
    ``9.54e-7`` m.  We allow two ulps after a float64 physical grid is stored
    as float32, but still require every stored centre to agree with the
    declared physical grid.  This is deliberately not a generic relaxation of
    the resolution contract.
    """

    magnitude = max(1.0e-3, float(np.max(np.abs(values))))
    return 2.0 * abs(float(np.spacing(np.float32(magnitude))))


def _validate_declared_axis(
    axis: np.ndarray,
    *,
    label: str,
    declared_step: float,
    declared_center: Optional[float] = None,
) -> np.ndarray:
    """Check centres against their declared physical spacing, not FP32 diffs."""

    values = np.asarray(axis, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError(f"RadarSplat B7873200 {label} axis must be finite and have at least two values")
    if not math.isfinite(float(declared_step)) or float(declared_step) <= 0.0:
        raise ValueError(f"RadarSplat B7873200 {label} declared spacing must be positive")
    if declared_center is None:
        expected = values[0] + np.arange(values.size, dtype=np.float64) * float(declared_step)
    else:
        # Use the recipe's exact origin when available. Anchoring at the rounded
        # first stored bin adds its serialization error to every other bin and
        # can falsely reject a correctly calibrated narrow angular crop.
        expected = float(declared_center) + (
            np.arange(values.size, dtype=np.float64) - 0.5 * (values.size - 1)
        ) * float(declared_step)
    tolerance = _float32_coordinate_tolerance(values)
    if np.any(np.diff(values) <= 0.0) or float(np.max(np.abs(values - expected))) > tolerance:
        raise ValueError(
            f"RadarSplat B7873200 {label} axis disagrees with its declared physical bin centres"
        )
    return values


def native_ssim_index(predicted: Tensor, target: Tensor) -> Tensor:
    """A local, real-power SSIM term with a declared dynamic range.

    Validation targets are normalized by a train-only peak and intentionally
    remain unclipped, so their values can exceed one.  The stabilizers scale
    with the detached observed range rather than assuming an image in [0, 1].
    """

    observed = torch.as_tensor(predicted)
    expected = torch.as_tensor(target, dtype=observed.dtype, device=observed.device)
    if observed.shape != expected.shape or observed.ndim not in {2, 3, 4}:
        raise ValueError("RadarSplat B7873200 SSIM inputs must have the same 2-D image shape")
    if torch.is_complex(observed) or torch.is_complex(expected):
        raise TypeError("RadarSplat B7873200 SSIM accepts real power only")
    if not bool(torch.isfinite(observed).all()) or not bool(torch.isfinite(expected).all()):
        raise ValueError("RadarSplat B7873200 SSIM inputs must be finite")
    if observed.ndim == 2:
        observed = observed[None, None]
        expected = expected[None, None]
    elif observed.ndim == 3:
        observed = observed[:, None]
        expected = expected[:, None]
    height, width = observed.shape[-2:]
    kernel = min(11, height, width)
    if kernel % 2 == 0:
        kernel -= 1
    if kernel < 1:
        raise ValueError("RadarSplat B7873200 SSIM needs non-empty images")
    padding = kernel // 2
    mean_x = F.avg_pool2d(observed, kernel, stride=1, padding=padding, count_include_pad=False)
    mean_y = F.avg_pool2d(expected, kernel, stride=1, padding=padding, count_include_pad=False)
    variance_x = F.avg_pool2d(observed.square(), kernel, stride=1, padding=padding, count_include_pad=False) - mean_x.square()
    variance_y = F.avg_pool2d(expected.square(), kernel, stride=1, padding=padding, count_include_pad=False) - mean_y.square()
    covariance = F.avg_pool2d(observed * expected, kernel, stride=1, padding=padding, count_include_pad=False) - mean_x * mean_y
    dynamic_range = torch.stack((observed.detach().abs().amax(), expected.detach().abs().amax())).amax().clamp_min(1.0)
    c1 = (0.01 * dynamic_range).square()
    c2 = (0.03 * dynamic_range).square()
    numerator = (2.0 * mean_x * mean_y + c1) * (2.0 * covariance + c2)
    denominator = (mean_x.square() + mean_y.square() + c1) * (variance_x + variance_y + c2)
    # C1*C2 is about 9e-8 under the deliberately unit-minimum dynamic range.
    # Clamping at machine epsilon would bias an identical dark image below one;
    # only protect an impossible/underflowed denominator instead.
    return (numerator / denominator.clamp_min(torch.finfo(observed.dtype).tiny)).mean()


@dataclass(frozen=True)
class NativeRadarSplatView:
    """One authorized real-power image and its calibrated local-polar frame."""

    view_index: int
    role: str
    target_power: Tensor
    sensor_to_world: Tensor
    grid: RadarSplatGrid
    elevation_count: int


def target_grid_from_arrays(
    *,
    range_m: np.ndarray,
    azimuth_rad: np.ndarray,
    expected_grid: Mapping[str, object],
) -> RadarSplatGrid:
    """Recover exact pixel edges from the stored target-bin centres.

    The renderer's raw pixels are sampled at half-integer coordinates.  The
    target cache stores physical bin centres; subtracting a half bin here is
    therefore essential for range and azimuth alignment rather than a cosmetic
    coordinate convention.
    """

    n_range = int(expected_grid["n_range"])
    n_azimuth = int(expected_grid["n_azimuth"])
    if len(range_m) != n_range or len(azimuth_rad) != n_azimuth:
        raise ValueError("RadarSplat B7873200 target axis counts disagree with its configured grid")
    scene_extent_m = float(expected_grid["scene_extent_m"])
    if not math.isfinite(scene_extent_m) or scene_extent_m <= 0.0:
        raise ValueError("RadarSplat B7873200 target grid needs a positive scene extent")
    range_step = 2.0 * scene_extent_m / n_range
    configured_output = float(expected_grid["output_azimuth_resolution_deg"])
    output_resolution_rad = math.radians(configured_output)
    ranges = _validate_declared_axis(range_m, label="range", declared_step=range_step)
    azimuth_center_deg = float(expected_grid["azimuth_center_deg"])
    if not math.isfinite(azimuth_center_deg):
        raise ValueError("RadarSplat B7873200 target grid needs a finite azimuth centre")
    azimuths = _validate_declared_axis(
        azimuth_rad, label="azimuth", declared_step=output_resolution_rad,
        declared_center=math.radians(azimuth_center_deg),
    )
    expected_azimuth_centres = (
        math.radians(azimuth_center_deg)
        + (-0.5 * n_azimuth + 0.5 + np.arange(n_azimuth, dtype=np.float64))
        * output_resolution_rad
    )
    if float(np.max(np.abs(azimuths - expected_azimuth_centres))) > _float32_coordinate_tolerance(azimuths):
        raise ValueError("RadarSplat B7873200 target azimuth centres disagree with the configured local crop")
    return RadarSplatGrid(
        num_range_bins=n_range,
        range_resolution_m=range_step,
        range_start_m=float(ranges[0]) - 0.5 * range_step,
        azimuth_start_deg=azimuth_center_deg - 0.5 * n_azimuth * configured_output,
        azimuth_span_deg=n_azimuth * configured_output,
        output_azimuth_resolution_deg=configured_output,
        intermediate_azimuth_resolution_deg=float(expected_grid["intermediate_azimuth_resolution_deg"]),
        azimuth_beamwidth_deg=float(expected_grid["azimuth_beamwidth_deg"]),
        spectral_leakage_width_m=float(expected_grid["spectral_leakage_width_m"]),
    )


def load_native_view(
    cache: RadarSplatB7873200Cache,
    view_index: int,
    role: str,
    device: torch.device | str,
) -> NativeRadarSplatView:
    """Read exactly one allowed native target; raw B787 responses stay sealed."""

    allowed = cache.train_indices if role == "train" else cache.validation_indices
    if role not in {"train", "validation"} or int(view_index) not in set(allowed):
        raise ValueError("RadarSplat B7873200 attempted to load a target outside its declared role")
    arrays = load_target(cache.root, int(view_index), role, expected_grid=cache.grid)
    target = torch.as_tensor(arrays["radarsplat_mf_power"], dtype=torch.float32, device=device)
    target = normalize_power(target, cache.train_peak_power)
    pose = torch.as_tensor(arrays["sensor_to_world"], dtype=torch.float32, device=device)
    grid = target_grid_from_arrays(
        range_m=arrays["range_m"], azimuth_rad=arrays["azimuth_rad"], expected_grid=cache.grid
    )
    return NativeRadarSplatView(
        view_index=int(view_index),
        role=role,
        target_power=target,
        sensor_to_world=pose,
        grid=grid,
        elevation_count=int(np.asarray(arrays["elevation_rad"]).size),
    )


def normalize_power(power: Tensor, train_peak_power: float) -> Tensor:
    """Apply the fixed, unclipped normalizer fitted only on training targets."""

    values = torch.as_tensor(power)
    if torch.is_complex(values):
        raise TypeError("RadarSplat B7873200 expects real squared-power targets, never coherent phase")
    if not math.isfinite(float(train_peak_power)) or float(train_peak_power) <= 0.0:
        raise ValueError("RadarSplat B7873200 train peak must be finite and positive")
    if not bool(torch.isfinite(values).all()) or bool((values < 0.0).any()):
        raise ValueError("RadarSplat B7873200 power must be finite and non-negative")
    return values / values.new_tensor(float(train_peak_power))


def native_power_from_matched_filter(
    amplitude: Tensor,
    *,
    n_elevation: int,
    n_azimuth: int,
    n_range: int,
) -> Tensor:
    """Project complex matched-filter samples to RadarSplat's real endpoint.

    The elevation dimension is squared-power integrated; it is never averaged
    coherently.  This one explicit operation prevents an amplitude/power
    nomenclature slip or an invented phase target in the cache preparer.
    """

    values = torch.as_tensor(amplitude)
    if not torch.is_complex(values):
        raise TypeError("RadarSplat B7873200 native target projection requires complex MF amplitude")
    expected = int(n_elevation) * int(n_azimuth) * int(n_range)
    if values.numel() != expected:
        raise ValueError("RadarSplat B7873200 matched-filter sample count disagrees with its polar grid")
    power = values.abs().square().reshape(int(n_elevation), int(n_azimuth), int(n_range)).sum(dim=0)
    if not bool(torch.isfinite(power).all()) or bool((power < 0.0).any()):
        raise ValueError("RadarSplat B7873200 native power projection is invalid")
    return power


def occupancy_mask(target_power: Tensor, threshold: float = OCCUPANCY_THRESHOLD) -> Tensor:
    """Native radar-derived occupancy labels, preserving the current 0.001 cutoff."""

    if not math.isfinite(float(threshold)) or not 0.0 < float(threshold) <= 1.0:
        raise ValueError("RadarSplat B7873200 occupancy threshold must lie in (0,1]")
    values = torch.as_tensor(target_power)
    if torch.is_complex(values) or not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("occupancy targets require finite non-negative real power")
    return values >= float(threshold)


def target_concentration(target_power: Tensor, threshold: float = OCCUPANCY_THRESHOLD) -> dict[str, float | int]:
    """Report, but do not optimize against, target support concentration.

    This makes the possible two-bin-support / 32-bin-grid limitation measurable
    before a future resolution change is proposed.  It has no acceptance
    threshold and does not alter the sensor adapter.
    """

    values = torch.as_tensor(target_power, dtype=torch.float64)
    if values.ndim != 2:
        raise ValueError("RadarSplat B7873200 concentration expects [azimuth,range] power")
    if torch.is_complex(values) or not bool(torch.isfinite(values).all()) or bool((values < 0.0).any()):
        raise ValueError("RadarSplat B7873200 concentration requires finite non-negative real power")
    total = float(values.sum().cpu())
    count = int(values.numel())
    positive = occupancy_mask(values, threshold)
    positive_count = int(positive.sum().cpu())
    nonzero = values > 0.0
    nonzero_count = int(nonzero.sum().cpu())
    if total <= 0.0:
        return {
            "bins": count,
            "positive_bins": positive_count,
            "positive_fraction": positive_count / count,
            "nonzero_bins": nonzero_count,
            "nonzero_fraction": nonzero_count / count,
            "effective_support_bins": 0.0,
            "peak_to_mean": 0.0,
            "azimuth_std_bins": 0.0,
            "range_std_bins": 0.0,
        }
    flat = values.reshape(-1)
    effective = float((flat.sum().square() / flat.square().sum().clamp_min(1.0e-30)).cpu())
    azimuth_coordinate = torch.arange(values.shape[0], dtype=values.dtype, device=values.device)[:, None]
    range_coordinate = torch.arange(values.shape[1], dtype=values.dtype, device=values.device)[None, :]
    azimuth_mean = (values * azimuth_coordinate).sum() / values.sum()
    range_mean = (values * range_coordinate).sum() / values.sum()
    azimuth_std = torch.sqrt(((values * (azimuth_coordinate - azimuth_mean).square()).sum() / values.sum()).clamp_min(0.0))
    range_std = torch.sqrt(((values * (range_coordinate - range_mean).square()).sum() / values.sum()).clamp_min(0.0))
    return {
        "bins": count,
        "positive_bins": positive_count,
        "positive_fraction": positive_count / count,
        "nonzero_bins": nonzero_count,
        "nonzero_fraction": nonzero_count / count,
        "effective_support_bins": effective,
        "peak_to_mean": float((values.max() / values.mean().clamp_min(1.0e-30)).cpu()),
        "azimuth_std_bins": float(azimuth_std.cpu()),
        "range_std_bins": float(range_std.cpu()),
    }


def balanced_occupancy_l1(predicted_occupancy: Tensor, target_mask: Tensor) -> tuple[Tensor, str]:
    """Balance positive and background pixels without changing the threshold.

    A conventional image-wide mean lets an overwhelmingly empty local-polar
    crop dominate the occupancy objective.  When both classes exist, each
    class gets one half of this term.  Degenerate synthetic views use their
    only observed class and record that fact for diagnostics.
    """

    predicted = torch.as_tensor(predicted_occupancy)
    mask = torch.as_tensor(target_mask, dtype=torch.bool, device=predicted.device)
    if predicted.shape != mask.shape:
        raise ValueError("RadarSplat B7873200 occupancy prediction and mask must have the same shape")
    if not bool(torch.isfinite(predicted).all()):
        raise ValueError("RadarSplat B7873200 occupancy prediction is non-finite")
    target = mask.to(dtype=predicted.dtype)
    residual = (predicted - target).abs()
    positive = residual[mask]
    background = residual[~mask]
    if positive.numel() and background.numel():
        return 0.5 * positive.mean() + 0.5 * background.mean(), "balanced"
    if positive.numel():
        return positive.mean(), "positive_only"
    if background.numel():
        return background.mean(), "background_only"
    raise ValueError("RadarSplat B7873200 occupancy image is empty")


@dataclass(frozen=True)
class ObjectiveWeights:
    ssim: float = 0.2
    occupancy: float = 5.0
    max_size: float = 100.0
    opacity_noise: float = 100.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.ssim <= 1.0:
            raise ValueError("RadarSplat B7873200 SSIM weight must lie in [0,1]")
        if min(self.occupancy, self.max_size, self.opacity_noise) < 0.0:
            raise ValueError("RadarSplat B7873200 objective weights must be non-negative")


def native_power_objective(
    rendered: Mapping[str, Tensor],
    target_power: Tensor,
    model: RadarSplatModel,
    *,
    max_scale: float,
    occupancy_threshold_value: float = OCCUPANCY_THRESHOLD,
    weights: ObjectiveWeights = ObjectiveWeights(),
    target_occupancy: Tensor | None = None,
    fidelity_profile: str = "legacy",
) -> dict[str, Tensor | str]:
    """Versioned native loss; historical recipes keep their original objective."""

    predicted = torch.as_tensor(rendered["final_power"])
    occupancy = torch.as_tensor(rendered["occupancy"])
    target = torch.as_tensor(target_power, dtype=predicted.dtype, device=predicted.device)
    if predicted.ndim == 3 and predicted.shape[0] == 1:
        predicted = predicted[0]
    if occupancy.ndim == 3 and occupancy.shape[0] == 1:
        occupancy = occupancy[0]
    if predicted.shape != target.shape or occupancy.shape != target.shape:
        raise ValueError("RadarSplat B7873200 renderer output must match the native target grid")
    if fidelity_profile not in {"legacy", "audit_v1"}:
        raise ValueError("unknown RadarSplat fidelity profile")
    if fidelity_profile == "audit_v1":
        if target_occupancy is None:
            raise ValueError("audit_v1 requires independently prepared training occupancy")
        labels = torch.as_tensor(target_occupancy, device=predicted.device)
        if labels.shape != target.shape or not torch.isfinite(labels).all() or not ((labels == 0) | (labels == 1)).all():
            raise ValueError("training occupancy must be a finite binary image matching power")
        mask = labels.bool()
    else:
        if target_occupancy is not None:
            raise ValueError("legacy loss cannot silently consume a different occupancy target")
        mask = occupancy_mask(target, occupancy_threshold_value)
    power_l1 = F.l1_loss(predicted, target)
    if fidelity_profile == "audit_v1":
        from rift.radarsplat_fidelity import release_ssim_index
        ssim_loss = 1.0 - release_ssim_index(predicted, target)
        occupancy_l1 = F.l1_loss(occupancy, mask.to(occupancy.dtype))
        balance_mode = "image_mean"
    else:
        ssim_loss = 1.0 - native_ssim_index(predicted, target)
        occupancy_l1, balance_mode = balanced_occupancy_l1(occupancy, mask)
    regularizers = model.native_regularizers(max_scale)
    total = (
        (1.0 - weights.ssim) * power_l1
        + weights.ssim * ssim_loss
        + weights.occupancy * occupancy_l1
        + weights.max_size * regularizers["max_size"]
        + weights.opacity_noise * regularizers["opacity_noise"]
    )
    return {
        "total": total,
        "power_l1": power_l1,
        "ssim_loss": ssim_loss,
        "occupancy_l1": occupancy_l1,
        "occupancy_balance_mode": balance_mode,
        "max_size": regularizers["max_size"],
        "opacity_noise": regularizers["opacity_noise"],
    }


def create_optimizers(
    model: RadarSplatModel,
    learning_rates: Mapping[str, float],
    *,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1.0e-15,
) -> dict[str, torch.optim.Adam]:
    """Make one native Adam optimizer per row-shaped Gaussian parameter."""

    if set(learning_rates) != set(_PARAMETER_NAMES):
        raise ValueError("RadarSplat B7873200 needs one learning rate for every Gaussian parameter")
    return {
        name: torch.optim.Adam(
            [{"params": [getattr(model, name)], "lr": float(learning_rates[name]), "name": name}],
            betas=betas,
            eps=float(eps),
        )
        for name in _PARAMETER_NAMES
    }


def model_from_checkpoint_state(
    state: Mapping[str, object],
    *,
    device: torch.device | str,
    seed: int = 0,
) -> RadarSplatModel:
    """Reconstruct the active Gaussian topology before restoring Adam state.

    Pruning changes the first dimension of every learned Gaussian parameter.
    A continuation must therefore instantiate the saved row count first; it
    must never load a pruned state into the original initialization topology.
    """

    required = set(_PARAMETER_NAMES)
    if set(state) != required:
        raise ValueError("RadarSplat B7873200 checkpoint model state has unexpected parameters")
    means = state["means"]
    log_scales = state["log_scales"]
    sh0 = state["sh0"]
    shn = state["shN"]
    if not all(torch.is_tensor(value) for value in state.values()):
        raise ValueError("RadarSplat B7873200 checkpoint model state must contain tensors")
    for name, value in state.items():
        assert torch.is_tensor(value)
        if value.dtype != torch.float32 or value.layout != torch.strided:
            raise ValueError(
                "RadarSplat B7873200 checkpoint model state must retain native float32 dense tensors; "
                f"{name} is incompatible"
            )
    assert torch.is_tensor(means) and torch.is_tensor(log_scales)
    assert torch.is_tensor(sh0) and torch.is_tensor(shn)
    if not all(bool(torch.isfinite(torch.as_tensor(value)).all()) for value in state.values()):
        raise ValueError("RadarSplat B7873200 checkpoint model state contains non-finite values")
    rows = int(means.shape[0]) if means.ndim == 2 else 0
    if rows < 1 or means.shape != (rows, 3) or log_scales.shape != (rows, 3):
        raise ValueError("RadarSplat B7873200 checkpoint Gaussian geometry has invalid row shapes")
    if state["quaternions"].shape != (rows, 4):
        raise ValueError("RadarSplat B7873200 checkpoint quaternion state has invalid row shapes")
    if state["opacity_logits"].shape != (rows,) or state["noise_probability_logits"].shape != (rows,):
        raise ValueError("RadarSplat B7873200 checkpoint probability state has invalid row shapes")
    if sh0.shape != (rows, 1, 3) or shn.ndim != 3 or shn.shape[0] != rows or shn.shape[2] != 3:
        raise ValueError("RadarSplat B7873200 checkpoint SH state has invalid row shapes")
    coefficient_count = 1 + int(shn.shape[1])
    side = math.isqrt(coefficient_count)
    if side * side != coefficient_count or not 1 <= side <= 5:
        raise ValueError("RadarSplat B7873200 checkpoint SH coefficient count is invalid")
    initial_scales = torch.exp(log_scales.detach())
    if not bool(torch.isfinite(initial_scales).all()) or not bool((initial_scales > 0.0).all()):
        raise ValueError("RadarSplat B7873200 checkpoint model scales are invalid")
    model = RadarSplatModel(
        means.detach().to(device=device, dtype=torch.float32),
        initial_scale=initial_scales.to(device=device, dtype=torch.float32),
        initial_opacity=0.1,
        initial_noise_probability=0.1,
        initial_reflectance=torch.full((rows, 3), 0.5, dtype=torch.float32, device=device),
        sh_degree=side - 1,
        seed=int(seed),
    ).to(device)
    # The explicit dtype check above makes this a device move only.  Never
    # silently cast a checkpointed parameter before a continuation update.
    normalized_state = {
        name: value.detach().to(device=device)
        for name, value in state.items()
    }
    model.load_state_dict(normalized_state, strict=True)
    return model


@torch.no_grad()
def _migrate_optimizer_rows(
    optimizers: Mapping[str, torch.optim.Optimizer],
    replacements: Mapping[str, tuple[torch.nn.Parameter, torch.nn.Parameter]],
    row_index: Tensor,
) -> None:
    """Carry retained Adam moments across a Gaussian-row prune operation."""

    for name, (old_parameter, new_parameter) in replacements.items():
        optimizer = optimizers[name]
        group = optimizer.param_groups[0]
        if (
            len(optimizer.param_groups) != 1
            or len(group["params"]) != 1
            or group["params"][0] is not old_parameter
        ):
            raise ValueError(f"RadarSplat B7873200 optimizer {name} does not match the model row state")
        old_state = optimizer.state.pop(old_parameter, {})
        updated_state: dict[str, Any] = {}
        for key, value in old_state.items():
            if torch.is_tensor(value) and value.ndim and value.shape[0] == old_parameter.shape[0]:
                updated_state[key] = value.index_select(0, row_index.to(value.device)).clone()
            else:
                updated_state[key] = value
        group["params"] = [new_parameter]
        optimizer.state[new_parameter] = updated_state


@torch.no_grad()
def prune_low_opacity(
    model: RadarSplatModel,
    optimizers: Mapping[str, torch.optim.Optimizer],
    threshold: float = PRUNE_OPACITY,
) -> dict[str, int | bool]:
    """Prune only rows strictly below the current cutoff, never all rows."""

    if not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) < 1.0:
        raise ValueError("RadarSplat B7873200 prune threshold must lie in [0,1)")
    opacity = model.opacity.detach()
    # Preserve the documented strict physical cutoff while accepting the
    # one-rounding-step error of sigmoid(logit(0.0005)) in float32. This does
    # not retain a materially sub-threshold Gaussian; it makes the specified
    # boundary reproducible across checkpoint round trips.
    boundary_roundoff = max(
        abs(float(threshold)) * 8.0 * torch.finfo(opacity.dtype).eps,
        torch.finfo(opacity.dtype).tiny,
    )
    keep = opacity >= float(threshold) - boundary_roundoff
    fallback_kept = False
    if not bool(keep.any()):
        keep[torch.argmax(opacity)] = True
        fallback_kept = True
    count_before = model.num_gaussians
    if bool(keep.all()):
        return {"before": count_before, "after": count_before, "fallback_kept": fallback_kept}
    indices = torch.where(keep)[0]
    replacements = model.prune_gaussians(keep)
    _migrate_optimizer_rows(optimizers, replacements, indices)
    return {"before": count_before, "after": model.num_gaussians, "fallback_kept": fallback_kept}


@torch.no_grad()
def export_gaussian_occupancy_geometry(
    path: str | Path,
    model: RadarSplatModel,
    *,
    active_sh_degree: int,
    train_peak_power: float,
) -> Path:
    """Export a native Gaussian/occupancy geometry readout, never a phase field."""

    if not 0 <= int(active_sh_degree) <= model.sh_degree:
        raise ValueError("RadarSplat B7873200 export active SH degree is invalid")
    return atomic_save_npz(
        path,
        schema=np.asarray(GEOMETRY_SCHEMA),
        active_sh_degree=np.asarray(int(active_sh_degree), dtype=np.int64),
        train_peak_power=np.asarray(float(train_peak_power), dtype=np.float64),
        observable=np.asarray("native Gaussian occupancy geometry; no coherent phase field"),
        means=model.means.detach().cpu().numpy().astype(np.float32, copy=False),
        scales=model.scales.detach().cpu().numpy().astype(np.float32, copy=False),
        quaternions=F.normalize(model.quaternions.detach(), p=2, dim=-1).cpu().numpy().astype(np.float32, copy=False),
        occupancy=model.opacity.detach().cpu().numpy().astype(np.float32, copy=False),
        noise_probability=model.noise_probability.detach().cpu().numpy().astype(np.float32, copy=False),
    )
