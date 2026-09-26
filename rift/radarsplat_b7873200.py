"""Differentiable RadarSplat core for the RIFT baseline suite.

This is a clean-room adapter of the official RadarSplat release pinned at
``ea9c8f530c708622cc3b1b560436b5557ac6a49b``.  It intentionally preserves
the release's radar-specific semantics instead of turning RadarSplat back into
an optical alpha-composited 3DGS renderer:

* an explicit set of anisotropic 3-D Gaussians (mean, quaternion, scale);
* occupancy probability ``alpha`` and a learned noise probability ``eta``;
* view-dependent spherical-harmonic reflectance ``rho``;
* per-Gaussian power ``rho * min(alpha + eta, 1)`` (paper Eq. 10);
* Cartesian-to-spherical first-order covariance projection;
* additive, depth-order-independent Gaussian splatting in azimuth/range;
* azimuth antenna-gain convolution and range spectral-leakage convolution;
* separate ``rho*alpha`` and ``rho*eta`` inverse-rendering products; and
* an optional externally reconstructed multipath background.

The repaired B7873200 lane uses one reviewed differentiable PyTorch sparse
rasterizer. It preserves the release's three-sigma projected/tile support and
per-render-product ``1/255`` alpha cutoff, but it has no custom-CUDA build,
source-identity, or digest gate. It never evaluates infinite Gaussian tails.

Important provenance details
----------------------------
``eta`` jointly represents receiver saturation and speckle in the paper; the
release does not provide two independently switchable learned branches.  The
``use_noise_probability`` flag therefore switches the *joint* native branch.
The B787 clean-data preset keeps it enabled, as in the original model, and its
native ``relu(alpha + eta - 1)`` regularizer penalizes only excess joint
probability. It does not independently drive noise mass to zero. Only the
external multipath background is disabled by default.

The paper's Eq. 11 contains elevation gain and an ``R^-4`` factor, but the
pinned release explicitly comments that its projection omits elevation gain
and its renderer applies no range law.  Release-faithful defaults therefore
use unity elevation gain and ``range_power_exponent=0``.  A caller may supply
measured per-Gaussian elevation gain and/or select exponent 4 without
silently claiming that these were present in the released implementation.

Two B787 corrections intentionally differ from bugs in the pinned executable.
World covariance is rotated into the sensor frame before spherical propagation,
and SH directions are evaluated in Cartesian world coordinates.  The pinned
code omits the covariance rotation and subtracts a Cartesian translation from
spherical coordinates.  These physically necessary corrections are serialized
as B787 adaptations rather than claimed as executable-release parity.

The historical reference mapping remains archival context only; it is not an
execution dependency of this isolated B7873200 lane.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Dict, Mapping, Optional, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


_C0 = 0.2820947917738781
METHOD_NAME = "RadarSplat (native B7873200 torch sparse reference)"
_GAUSSIAN_ALPHA_CEILING = 0.999
_GAUSSIAN_ALPHA_CUTOFF = 1.0 / 255.0
_RELEASE_TILE_SIZE = 16
_PER_GAUSSIAN_PARAMETERS = (
    "means",
    "log_scales",
    "quaternions",
    "opacity_logits",
    "noise_probability_logits",
    "sh0",
    "shN",
)


@dataclass(frozen=True)
class RadarSplatGrid:
    """The official 2-D scanning-radar azimuth/range image contract.

    RadarSplat renders a high-resolution intermediate azimuth grid and reduces
    it to sensor resolution with a 1-D antenna kernel.  ``azimuth_start_deg``
    is the *output* crop edge.  The intermediate edge is shifted by half the
    output-minus-intermediate resolution so that stride sample zero is exactly
    the first output-bin centre.  Pixel centres then match
    ``_torch_impl_radar.accumulate``.
    """

    num_range_bins: int
    range_resolution_m: float
    intermediate_azimuth_resolution_deg: float = 0.1
    output_azimuth_resolution_deg: float = 0.9
    azimuth_beamwidth_deg: float = 1.8
    azimuth_start_deg: float = 0.0
    azimuth_span_deg: float = 360.0
    range_start_m: float = 0.0
    spectral_leakage_width_m: float = 1.0
    eps2d_pixels: float = 0.3

    def __post_init__(self) -> None:
        if self.num_range_bins < 1:
            raise ValueError("num_range_bins must be positive")
        positive = (
            self.range_resolution_m,
            self.intermediate_azimuth_resolution_deg,
            self.output_azimuth_resolution_deg,
            self.azimuth_beamwidth_deg,
            self.spectral_leakage_width_m,
        )
        if any(not math.isfinite(float(value)) or value <= 0 for value in positive):
            raise ValueError("grid resolutions, beamwidth, and leakage width must be finite and positive")
        if not math.isfinite(float(self.azimuth_start_deg)) or not math.isfinite(float(self.range_start_m)):
            raise ValueError("grid start coordinates must be finite")
        if self.output_azimuth_resolution_deg < self.intermediate_azimuth_resolution_deg:
            raise ValueError("output azimuth resolution cannot be finer than the intermediate grid")
        if not math.isfinite(float(self.eps2d_pixels)) or self.eps2d_pixels <= 0:
            raise ValueError("eps2d_pixels must be finite and positive")
        if not 0 < self.azimuth_span_deg <= 360.0:
            raise ValueError("azimuth_span_deg must lie in (0,360]")
        self._integer_ratio(
            self.output_azimuth_resolution_deg,
            self.intermediate_azimuth_resolution_deg,
            "output/intermediate azimuth resolution",
        )
        self._integer_ratio(
            self.azimuth_span_deg,
            self.intermediate_azimuth_resolution_deg,
            "azimuth span/intermediate resolution",
        )
        self._integer_ratio(
            self.azimuth_span_deg,
            self.output_azimuth_resolution_deg,
            "azimuth span/output resolution",
        )
        sigma_pixels = int(
            ((self.spectral_leakage_width_m / 2.0) / self.range_resolution_m) / 3.0
        )
        if sigma_pixels < 1:
            raise ValueError("official spectral-leakage discretization produced sigma_pixels < 1")

    @staticmethod
    def _integer_ratio(numerator: float, denominator: float, label: str) -> int:
        ratio = numerator / denominator
        rounded = int(round(ratio))
        if rounded < 1 or not math.isclose(ratio, rounded, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError(f"{label} must be an integer, got {ratio:g}")
        return rounded

    @property
    def intermediate_azimuth_bins(self) -> int:
        return self._integer_ratio(
            self.azimuth_span_deg, self.intermediate_azimuth_resolution_deg, "azimuth bins"
        )

    @property
    def output_azimuth_bins(self) -> int:
        return self._integer_ratio(
            self.azimuth_span_deg, self.output_azimuth_resolution_deg, "azimuth bins"
        )

    @property
    def azimuth_stride(self) -> int:
        return self._integer_ratio(
            self.output_azimuth_resolution_deg,
            self.intermediate_azimuth_resolution_deg,
            "azimuth stride",
        )

    @property
    def intermediate_azimuth_start_offset_deg(self) -> float:
        """Derived edge shift that aligns decimated and target bin centres."""

        return 0.5 * (
            self.output_azimuth_resolution_deg
            - self.intermediate_azimuth_resolution_deg
        )

    @property
    def intermediate_azimuth_start_deg(self) -> float:
        return self.azimuth_start_deg + self.intermediate_azimuth_start_offset_deg

    @property
    def intermediate_azimuth_start_rad(self) -> float:
        return math.radians(self.intermediate_azimuth_start_deg)

    @property
    def range_stop_m(self) -> float:
        return self.range_start_m + self.num_range_bins * self.range_resolution_m

    @property
    def is_full_azimuth(self) -> bool:
        return math.isclose(self.azimuth_span_deg, 360.0, rel_tol=0.0, abs_tol=1.0e-6)

    @property
    def azimuth_start_rad(self) -> float:
        return math.radians(self.azimuth_start_deg % 360.0)

    @property
    def azimuth_span_rad(self) -> float:
        return math.radians(self.azimuth_span_deg)


@dataclass(frozen=True)
class RadarSplatEffects:
    """Switches for observable effects in the released RadarSplat model.

    ``use_noise_probability`` is the native joint saturation/speckle ``eta``
    branch.  It is enabled by default, including for noise-free B787 data;
    setting it false is the paper's explicit ``w/o noise probability``
    ablation.  Multipath is a separate external periodic background and is
    disabled for B787 because the synthetic data provide no such source map.
    """

    use_noise_probability: bool = True
    use_spectral_leakage: bool = True
    use_azimuth_antenna_gain: bool = True
    use_multipath: bool = False
    multipath_weight: float = 0.6
    range_power_exponent: float = 0.0
    output_floor: float = 1.0e-6
    output_ceiling: Optional[float] = 1.0
    # None preserves historical shared clipping. Corrected recipes cap
    # probabilities separately while allowing normalized power above one.
    probability_ceiling: Optional[float] = None
    raster_alpha_cutoff: float = _GAUSSIAN_ALPHA_CUTOFF

    def __post_init__(self) -> None:
        if self.multipath_weight < 0:
            raise ValueError("multipath_weight must be nonnegative")
        if self.range_power_exponent < 0:
            raise ValueError("range_power_exponent must be nonnegative")
        if self.output_floor < 0:
            raise ValueError("output_floor must be nonnegative")
        if self.output_ceiling is not None and self.output_ceiling <= self.output_floor:
            raise ValueError("need output_ceiling > output_floor when a ceiling is supplied")
        if self.probability_ceiling is not None and not self.output_floor < self.probability_ceiling <= 1.0:
            raise ValueError("probability_ceiling must exceed the floor and be at most one")
        if not math.isfinite(self.raster_alpha_cutoff) or not 0 <= self.raster_alpha_cutoff < 1:
            raise ValueError("raster_alpha_cutoff must lie in [0,1)")

    @classmethod
    def b787_clean(cls) -> "RadarSplatEffects":
        """Clean B787 preset without silently selecting the no-noise ablation."""

        return cls(
            use_noise_probability=True,
            use_spectral_leakage=True,
            use_azimuth_antenna_gain=True,
            use_multipath=False,
            multipath_weight=0.0,
            range_power_exponent=0.0,
            output_floor=0.0,
            output_ceiling=None,
        )


def _logit(value: Tensor, eps: float = 1.0e-6) -> Tensor:
    value = value.clamp(eps, 1.0 - eps)
    return torch.log(value) - torch.log1p(-value)


def quaternion_to_rotation_matrix(quaternions: Tensor) -> Tensor:
    """WXYZ quaternion conversion copied algebraically from gsplat's torch path."""

    if quaternions.shape[-1] != 4:
        raise ValueError("quaternions must end in four WXYZ components")
    q = F.normalize(quaternions, p=2, dim=-1)
    w, x, y, z = q.unbind(dim=-1)
    matrix = torch.stack(
        (
            1 - 2 * (y.square() + z.square()),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x.square() + z.square()),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x.square() + y.square()),
        ),
        dim=-1,
    )
    return matrix.reshape(quaternions.shape[:-1] + (3, 3))


def covariance_from_quaternion_scale(quaternions: Tensor, scales: Tensor) -> Tensor:
    """Construct ``R diag(scale^2) R^T`` as in ``_quat_scale_to_covar_preci``."""

    if scales.shape != quaternions.shape[:-1] + (3,):
        raise ValueError("scales and quaternions have incompatible shapes")
    rotation = quaternion_to_rotation_matrix(quaternions)
    transform = rotation * scales[..., None, :]
    return transform @ transform.transpose(-1, -2)


def eval_sh_bases(degree: int, directions: Tensor) -> Tensor:
    """Real SH bases in the exact ordering/constants used by pinned gsplat.

    The pinned CUDA implementation evaluates degrees zero through four.  The
    official demo requests degree five, but coefficients above index 24 are
    not consumed by that CUDA evaluator.  This adapter rejects that ambiguous
    request and exposes the effective maximum explicitly.
    """

    if degree < 0 or degree > 4:
        raise ValueError("the pinned RadarSplat/gsplat evaluator supports effective SH degree 0..4")
    if directions.shape[-1] != 3:
        raise ValueError("directions must end in three coordinates")
    xyz = F.normalize(directions, p=2, dim=-1, eps=1.0e-12)
    x, y, z = xyz.unbind(dim=-1)
    basis = [torch.full_like(x, _C0)]
    if degree >= 1:
        basis.extend((-0.48860251190292 * y, 0.48860251190292 * z, -0.48860251190292 * x))
    if degree >= 2:
        z2 = z.square()
        f_c1 = x.square() - y.square()
        f_s1 = 2 * x * y
        basis.extend(
            (
                0.5462742152960395 * f_s1,
                -1.092548430592079 * z * y,
                0.9461746957575601 * z2 - 0.3153915652525201,
                -1.092548430592079 * z * x,
                0.5462742152960395 * f_c1,
            )
        )
    if degree >= 3:
        z2 = z.square()
        f_c1 = x.square() - y.square()
        f_s1 = 2 * x * y
        f_c2 = x * f_c1 - y * f_s1
        f_s2 = x * f_s1 + y * f_c1
        f_tmp_c = -2.285228997322329 * z2 + 0.4570457994644658
        f_tmp_b = 1.445305721320277 * z
        basis.extend(
            (
                -0.5900435899266435 * f_s2,
                f_tmp_b * f_s1,
                f_tmp_c * y,
                z * (1.865881662950577 * z2 - 1.119528997770346),
                f_tmp_c * x,
                f_tmp_b * f_c1,
                -0.5900435899266435 * f_c2,
            )
        )
    if degree >= 4:
        z2 = z.square()
        f_c1 = x.square() - y.square()
        f_s1 = 2 * x * y
        f_c2 = x * f_c1 - y * f_s1
        f_s2 = x * f_s1 + y * f_c1
        f_c3 = x * f_c2 - y * f_s2
        f_s3 = x * f_s2 + y * f_c2
        f_tmp_d = z * (-4.683325804901025 * z2 + 2.007139630671868)
        f_tmp_c = 3.31161143515146 * z2 - 0.47308734787878
        f_tmp_b = -1.770130769779931 * z
        p20 = (
            1.984313483298443 * z2 * (1.865881662950577 * z2 - 1.119528997770346)
            - 1.006230589874905 * (0.9461746957575601 * z2 - 0.3153915652525201)
        )
        basis.extend(
            (
                0.6258357354491763 * f_s3,
                f_tmp_b * f_s2,
                f_tmp_c * f_s1,
                f_tmp_d * y,
                p20,
                f_tmp_d * x,
                f_tmp_c * f_c1,
                f_tmp_b * f_c2,
                0.6258357354491763 * f_c3,
            )
        )
    return torch.stack(basis, dim=-1)


def cartesian_to_spherical_gaussians(
    means_world: Tensor,
    covariances_world: Tensor,
    sensor_to_world: Tensor,
    eps: float = 1.0e-7,
) -> Dict[str, Tensor]:
    """First-order Cartesian-to-spherical Gaussian transform (paper Eqs. 21--23).

    ``sensor_to_world`` follows the release's pose convention.  A single
    ``[4,4]`` pose is promoted to a one-view batch.  Azimuth is returned in
    ``[0, 2*pi)`` and elevation in ``[-pi/2, pi/2]``.  Unlike the pinned
    executable bug, the world covariance is rotated into the sensor frame
    before propagation.  This is an explicit B787 physical correction.
    """

    if means_world.ndim != 2 or means_world.shape[-1] != 3:
        raise ValueError("means_world must have shape [N,3]")
    if covariances_world.shape != (means_world.shape[0], 3, 3):
        raise ValueError("covariances_world must have shape [N,3,3]")
    if sensor_to_world.shape == (4, 4):
        sensor_to_world = sensor_to_world.unsqueeze(0)
    if sensor_to_world.ndim != 3 or sensor_to_world.shape[-2:] != (4, 4):
        raise ValueError("sensor_to_world must have shape [B,4,4] or [4,4]")
    sensor_to_world = sensor_to_world.to(
        device=means_world.device, dtype=means_world.dtype
    )

    rotation = sensor_to_world[:, :3, :3]
    translation = sensor_to_world[:, :3, 3]
    delta_world = means_world.unsqueeze(0) - translation[:, None, :]
    means_sensor = delta_world @ rotation
    covariance_sensor = (
        rotation.transpose(-1, -2)[:, None, :, :]
        @ covariances_world[None, :, :, :]
        @ rotation[:, None, :, :]
    )

    x, y, z = means_sensor.unbind(dim=-1)
    radius = torch.sqrt(x.square() + y.square() + z.square()).clamp_min(eps)
    radius_xy = torch.sqrt(x.square() + y.square()).clamp_min(eps)
    azimuth = torch.remainder(torch.atan2(y, x), 2.0 * math.pi)
    elevation = torch.atan2(z, radius_xy)
    means_spherical = torch.stack((radius, azimuth, elevation), dim=-1)

    zero = torch.zeros_like(radius)
    jacobian = torch.stack(
        (
            x / radius,
            y / radius,
            z / radius,
            -y / radius_xy.square(),
            x / radius_xy.square(),
            zero,
            -x * z / (radius_xy * radius.square()),
            -y * z / (radius_xy * radius.square()),
            radius_xy / radius.square(),
        ),
        dim=-1,
    ).reshape(means_sensor.shape[0], means_sensor.shape[1], 3, 3)
    covariance_spherical = jacobian @ covariance_sensor @ jacobian.transpose(-1, -2)
    return {
        "means_sensor": means_sensor,
        "covariances_sensor": covariance_sensor,
        "means_spherical": means_spherical,
        "covariances_spherical": covariance_spherical,
        "jacobian": jacobian,
    }


def normalized_gaussian_kernel1d(
    sigma_pixels: float,
    kernel_size: Optional[int] = None,
    *,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Return the unit-sum Gaussian kernel used for antenna/leakage filters."""

    if sigma_pixels <= 0:
        raise ValueError("sigma_pixels must be positive")
    if kernel_size is None:
        kernel_size = int(math.ceil(6.0 * sigma_pixels)) + 1
    if kernel_size < 1:
        raise ValueError("kernel_size must be positive")
    if kernel_size % 2 == 0:
        kernel_size += 1
    coordinate = torch.arange(kernel_size, dtype=dtype, device=device) - kernel_size // 2
    kernel = torch.exp(-0.5 * coordinate.square() / float(sigma_pixels) ** 2)
    return kernel / kernel.sum()


def _as_batched_image(image: Tensor) -> Tuple[Tensor, bool]:
    if image.ndim == 2:
        return image.unsqueeze(0), True
    if image.ndim != 3:
        raise ValueError("image must have shape [H,W] or [B,H,W]")
    return image, False


def spectral_leakage(image: Tensor, range_resolution_m: float, sinc_width_m: float = 1.0) -> Tensor:
    """Official normalized Gaussian approximation to Hamming-window leakage.

    The release uses ``sigma=int(((width/2)/range_resolution)/3)`` pixels,
    a six-sigma odd kernel, zero padding, and convolution along range only.
    """

    if range_resolution_m <= 0 or sinc_width_m <= 0:
        raise ValueError("range resolution and sinc width must be positive")
    sigma_pixels = int(((sinc_width_m / 2.0) / range_resolution_m) / 3.0)
    if sigma_pixels < 1:
        raise ValueError("official spectral-leakage discretization produced sigma_pixels < 1")
    kernel_size = int(sigma_pixels * 6) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    batched, squeeze = _as_batched_image(image)
    kernel = normalized_gaussian_kernel1d(
        float(sigma_pixels), kernel_size, dtype=batched.dtype, device=batched.device
    ).view(1, 1, 1, -1)
    output = F.conv2d(
        batched.unsqueeze(1), kernel, stride=(1, 1), padding=(0, kernel_size // 2)
    ).squeeze(1)
    return output.squeeze(0) if squeeze else output


def azimuth_antenna_gain_projection(
    image: Tensor,
    output_resolution_deg: float = 0.9,
    beamwidth_deg: float = 1.8,
    *,
    input_resolution_deg: Optional[float] = None,
    circular: bool = True,
) -> Tensor:
    """Official Gaussian-like azimuth antenna convolution and decimation.

    The release fixes the kernel standard deviation at four intermediate
    pixels, uses circular azimuth padding, and strides by the integer output
    to intermediate resolution ratio.  Sample zero is retained.  A
    :class:`RadarSplatGrid` aligns that sample with the first output centre by
    shifting the intermediate-grid edge, rather than by changing this release
    operation.
    """

    batched, squeeze = _as_batched_image(image)
    height = batched.shape[-2]
    old_resolution = (
        float(input_resolution_deg) if input_resolution_deg is not None else 360.0 / height
    )
    if old_resolution <= 0:
        raise ValueError("input_resolution_deg must be positive")
    stride_ratio = output_resolution_deg / old_resolution
    stride = int(round(stride_ratio))
    if stride < 1 or not math.isclose(stride_ratio, stride, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError("output/intermediate azimuth resolution must be an integer")
    window_size = int(beamwidth_deg / old_resolution)
    window_size = max(window_size, 1)
    if window_size % 2 == 0:
        window_size += 1
    padding = window_size // 2
    kernel = normalized_gaussian_kernel1d(
        4.0, window_size, dtype=batched.dtype, device=batched.device
    ).view(1, 1, -1, 1)
    padding_mode = "circular" if circular else "constant"
    padded = F.pad(batched.unsqueeze(1), (0, 0, padding, padding), mode=padding_mode)
    output = F.conv2d(padded, kernel, stride=(stride, 1)).squeeze(1)
    return output.squeeze(0) if squeeze else output


def azimuth_filter_halo_bins(grid: RadarSplatGrid) -> int:
    """Return a stride-aligned raw halo covering the released antenna kernel."""

    if grid.is_full_azimuth:
        return 0
    window_size = max(
        int(grid.azimuth_beamwidth_deg / grid.intermediate_azimuth_resolution_deg),
        1,
    )
    if window_size % 2 == 0:
        window_size += 1
    padding = window_size // 2
    stride = grid.azimuth_stride
    return int(math.ceil(padding / stride)) * stride


def spectral_filter_halo_bins(grid: RadarSplatGrid) -> int:
    """Range bins needed to make the released leakage crop edge-independent."""

    sigma_pixels = int(
        ((grid.spectral_leakage_width_m / 2.0) / grid.range_resolution_m) / 3.0
    )
    if sigma_pixels < 1:
        raise ValueError("official spectral-leakage discretization produced sigma_pixels < 1")
    return 3 * sigma_pixels


def extended_local_azimuth_grid(
    grid: RadarSplatGrid,
) -> Tuple[RadarSplatGrid, slice, slice]:
    """Render a local crop with physical halo, then select its requested FOV.

    Returns the extended grid, the final-output crop, and the corresponding
    intermediate-image crop.  Full-circle grids are returned unchanged.
    """

    halo = azimuth_filter_halo_bins(grid)
    if halo == 0:
        return (
            grid,
            slice(0, grid.output_azimuth_bins),
            slice(0, grid.intermediate_azimuth_bins),
        )
    input_resolution = grid.intermediate_azimuth_resolution_deg
    extended_span = grid.azimuth_span_deg + 2.0 * halo * input_resolution
    if extended_span > 360.0 + 1.0e-6:
        raise ValueError("local azimuth crop plus antenna halo exceeds 360 degrees")
    extended = replace(
        grid,
        azimuth_start_deg=grid.azimuth_start_deg - halo * input_resolution,
        azimuth_span_deg=extended_span,
    )
    final_halo = halo // grid.azimuth_stride
    return (
        extended,
        slice(final_halo, final_halo + grid.output_azimuth_bins),
        slice(halo, halo + grid.intermediate_azimuth_bins),
    )


def _relative_azimuth_radians(azimuth: Tensor, grid: RadarSplatGrid) -> Tensor:
    """Map angles to the release pixel axis, retaining a crop's left halo."""

    relative = torch.remainder(
        azimuth - grid.intermediate_azimuth_start_rad, 2.0 * math.pi
    )
    if grid.is_full_azimuth:
        return relative
    # Values immediately before the local crop start arrive near 2*pi.  Give
    # those values their equivalent negative coordinate so projected support
    # can intersect a physical halo without identifying the two crop edges.
    return torch.where(
        (relative > grid.azimuth_span_rad) & (relative > math.pi),
        relative - 2.0 * math.pi,
        relative,
    )


def _physical_gaussians_to_pixels(
    means_range_azimuth: Tensor,
    covariance_range_azimuth: Tensor,
    grid: RadarSplatGrid,
) -> Tuple[Tensor, Tensor]:
    azimuth_resolution = math.radians(grid.intermediate_azimuth_resolution_deg)
    means_pixels = torch.stack(
        (
            (means_range_azimuth[..., 0] - grid.range_start_m)
            / grid.range_resolution_m,
            _relative_azimuth_radians(means_range_azimuth[..., 1], grid)
            / azimuth_resolution,
        ),
        dim=-1,
    )
    resolution = means_range_azimuth.new_tensor(
        (grid.range_resolution_m, azimuth_resolution)
    )
    covariance_pixels = covariance_range_azimuth / (
        resolution[None, None, :, None] * resolution[None, None, None, :]
    )
    return means_pixels, covariance_pixels


def _release_projected_radius(covariance_pixels: Tensor) -> Tuple[Tensor, Tensor]:
    """Pinned CUDA three-sigma radius and positive-determinant validity."""

    c00 = covariance_pixels[..., 0, 0]
    c01 = covariance_pixels[..., 0, 1]
    c11 = covariance_pixels[..., 1, 1]
    determinant = c00 * c11 - c01.square()
    midpoint = 0.5 * (c00 + c11)
    largest = midpoint + torch.sqrt(torch.clamp(midpoint.square() - determinant, min=0.01))
    radius = torch.ceil(3.0 * torch.sqrt(largest.clamp_min(0.0)))
    valid = (
        torch.isfinite(covariance_pixels).all(dim=-1).all(dim=-1)
        & torch.isfinite(radius)
        & (determinant > 0.0)
        & (radius > 0.0)
    )
    return radius, valid


def _release_tile_plan(
    means_pixels: Tensor,
    radii: Tensor,
    valid: Tensor,
    *,
    height: int,
    width: int,
    tile_size: int,
) -> Tuple[list, int]:
    """Build detached release tile intersections and count candidate pairs."""

    plan = []
    candidate_pairs = 0
    for batch_index in range(means_pixels.shape[0]):
        batch_rows = []
        detached_means = means_pixels[batch_index].detach()
        detached_radii = radii[batch_index].detach()
        detached_valid = valid[batch_index].detach()
        for tile_y0 in range(0, height, tile_size):
            tile_y1 = min(tile_y0 + tile_size, height)
            row = []
            for tile_x0 in range(0, width, tile_size):
                tile_x1 = min(tile_x0 + tile_size, width)
                intersects = (
                    detached_valid
                    & (detached_means[:, 0] + detached_radii > tile_x0)
                    & (detached_means[:, 0] - detached_radii < tile_x1)
                    & (detached_means[:, 1] + detached_radii > tile_y0)
                    & (detached_means[:, 1] - detached_radii < tile_y1)
                )
                gaussian_ids = torch.where(intersects)[0]
                candidate_pairs += int(gaussian_ids.numel()) * (
                    (tile_y1 - tile_y0) * (tile_x1 - tile_x0)
                )
                row.append((tile_y0, tile_y1, tile_x0, tile_x1, gaussian_ids))
            batch_rows.append(row)
        plan.append(batch_rows)
    return plan, candidate_pairs


@torch.no_grad()
def rasterization_candidate_pairs(
    means_range_azimuth: Tensor,
    covariance_range_azimuth: Tensor,
    grid: RadarSplatGrid,
    *,
    tile_size: int = _RELEASE_TILE_SIZE,
) -> int:
    """Conservative release-tile Gaussian/pixel work estimate without rasterizing."""

    if means_range_azimuth.ndim != 3 or means_range_azimuth.shape[-1] != 2:
        raise ValueError("means_range_azimuth must have shape [B,N,2]")
    if covariance_range_azimuth.shape != (*means_range_azimuth.shape[:2], 2, 2):
        raise ValueError("covariance_range_azimuth must have shape [B,N,2,2]")
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    means_pixels, covariance_pixels = _physical_gaussians_to_pixels(
        means_range_azimuth, covariance_range_azimuth, grid
    )
    radii, valid = _release_projected_radius(covariance_pixels)
    _, candidate_pairs = _release_tile_plan(
        means_pixels,
        radii,
        valid,
        height=grid.intermediate_azimuth_bins,
        width=grid.num_range_bins,
        tile_size=tile_size,
    )
    return candidate_pairs


def additive_gaussian_rasterization(
    means_range_azimuth: Tensor,
    covariance_range_azimuth: Tensor,
    attributes: Tensor,
    grid: RadarSplatGrid,
    gaussian_chunk_size: int = 128,
    *,
    tile_size: int = _RELEASE_TILE_SIZE,
    max_candidate_pairs: Optional[int] = None,
    backend: str = "torch_sparse_reference",
    sort_depths: Optional[Tensor] = None,
    alpha_cutoff: float = _GAUSSIAN_ALPHA_CUTOFF,
) -> Tensor:
    """Rasterize additive 2-D Gaussians with native sparse support rules.

    Every product uses the same projected three-sigma support and independent
    ``1/255`` alpha cutoff. The only supported backend is the reviewed,
    differentiable PyTorch sparse reference implementation.
    """

    if means_range_azimuth.ndim != 3 or means_range_azimuth.shape[-1] != 2:
        raise ValueError("means_range_azimuth must have shape [B,N,2]")
    batch, count, _ = means_range_azimuth.shape
    if covariance_range_azimuth.shape != (batch, count, 2, 2):
        raise ValueError("covariance_range_azimuth must have shape [B,N,2,2]")
    if attributes.ndim != 3 or attributes.shape[:2] != (batch, count):
        raise ValueError("attributes must have shape [B,N,C]")
    if gaussian_chunk_size < 1 or tile_size < 1:
        raise ValueError("gaussian_chunk_size and tile_size must be positive")
    if not math.isfinite(alpha_cutoff) or not 0 <= alpha_cutoff < 1:
        raise ValueError("alpha_cutoff must lie in [0,1)")
    if max_candidate_pairs is not None and max_candidate_pairs < 1:
        raise ValueError("max_candidate_pairs must be positive when provided")
    if means_range_azimuth.dtype != covariance_range_azimuth.dtype:
        raise ValueError("means and covariance must use the same dtype")
    if attributes.dtype != means_range_azimuth.dtype:
        raise ValueError("attributes and geometry must use the same dtype")
    if backend != "torch_sparse_reference":
        raise ValueError(
            "RadarSplat B7873200 supports only the reviewed torch_sparse_reference renderer"
        )
    if sort_depths is not None and torch.as_tensor(sort_depths).shape != (batch, count):
        raise ValueError("sort_depths must have shape [B,N]")

    dtype, device = means_range_azimuth.dtype, means_range_azimuth.device
    height = grid.intermediate_azimuth_bins
    width = grid.num_range_bins
    channels = int(attributes.shape[-1])
    means_pixels, covariance_pixels = _physical_gaussians_to_pixels(
        means_range_azimuth, covariance_range_azimuth, grid
    )
    radii, valid = _release_projected_radius(covariance_pixels)
    tile_plan, candidate_pairs = _release_tile_plan(
        means_pixels,
        radii,
        valid,
        height=height,
        width=width,
        tile_size=tile_size,
    )
    if max_candidate_pairs is not None and candidate_pairs > max_candidate_pairs:
        raise RuntimeError(
            "RadarSplat PyTorch sparse rasterization is fail-closed: "
            f"{candidate_pairs:,} projected tile candidate pairs exceed the "
            f"configured limit {max_candidate_pairs:,}. Reduce the declared "
            "scene/profile before launching a dense fallback."
        )

    batch_images = []
    for batch_index, batch_rows in enumerate(tile_plan):
        row_tiles = []
        for planned_row in batch_rows:
            column_tiles = []
            for tile_y0, tile_y1, tile_x0, tile_x1, gaussian_ids in planned_row:
                tile_height = tile_y1 - tile_y0
                tile_width = tile_x1 - tile_x0
                tile_flat = torch.zeros(
                    (tile_height * tile_width, channels), dtype=dtype, device=device
                )
                if gaussian_ids.numel():
                    pixel_y, pixel_x = torch.meshgrid(
                        torch.arange(tile_y0, tile_y1, dtype=dtype, device=device) + 0.5,
                        torch.arange(tile_x0, tile_x1, dtype=dtype, device=device) + 0.5,
                        indexing="ij",
                    )
                    pixels = torch.stack((pixel_x.reshape(-1), pixel_y.reshape(-1)), dim=-1)
                    for start in range(0, gaussian_ids.numel(), gaussian_chunk_size):
                        ids = gaussian_ids[start : start + gaussian_chunk_size]
                        means = means_pixels[batch_index, ids]
                        precision = torch.linalg.inv(covariance_pixels[batch_index, ids])
                        delta = pixels[None, :, :] - means[:, None, :]
                        sigma = 0.5 * (
                            precision[:, 0, 0, None] * delta[..., 0].square()
                            + 2.0 * precision[:, 0, 1, None] * delta[..., 0] * delta[..., 1]
                            + precision[:, 1, 1, None] * delta[..., 1].square()
                        )
                        alpha = (
                            attributes[batch_index, ids, None, :] * torch.exp(-sigma[..., None])
                        ).clamp_max(_GAUSSIAN_ALPHA_CEILING)
                        keep = (sigma[..., None] >= 0.0) & (alpha >= alpha_cutoff)
                        tile_flat = tile_flat + torch.where(
                            keep, alpha, torch.zeros_like(alpha)
                        ).sum(dim=0)
                column_tiles.append(tile_flat.reshape(tile_height, tile_width, channels))
            row_tiles.append(torch.cat(column_tiles, dim=1))
        batch_images.append(torch.cat(row_tiles, dim=0))
    return torch.stack(batch_images, dim=0)


def official_cuda_build_metadata() -> Dict[str, object]:
    """Compatibility stub for the retired accelerated backend."""

    raise RuntimeError(
        "RadarSplat B7873200 exposes only torch_sparse_reference; "
        "the retired CUDA/provenance branch is not part of this lane"
    )


def validate_official_cuda_backend() -> None:
    """Compatibility stub that prevents accidental retired-backend execution."""

    official_cuda_build_metadata()


def _official_cuda_small_reference_gate() -> Dict[str, object]:
    """Compatibility stub for historical callers of the removed CUDA gate."""

    official_cuda_build_metadata()

class RadarSplatModel(nn.Module):
    """Explicit RadarSplat Gaussian scene and release-faithful renderer."""

    def __init__(
        self,
        means: Tensor,
        initial_scale: Union[float, Tensor] = 0.5,
        initial_opacity: float = 0.1,
        initial_noise_probability: float = 0.1,
        initial_reflectance: Optional[Tensor] = None,
        sh_degree: int = 3,
        seed: int = 42,
        _initialization_generator: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__()
        if means.ndim != 2 or means.shape[-1] != 3 or means.shape[0] < 1:
            raise ValueError("means must have shape [N,3] with N>0")
        if sh_degree < 0 or sh_degree > 4:
            raise ValueError("effective sh_degree must lie in 0..4 for the pinned release")
        if not 0 < initial_opacity < 1 or not 0 < initial_noise_probability < 1:
            raise ValueError("initial probabilities must lie strictly in (0,1)")
        self.sh_degree = int(sh_degree)
        self.seed = int(seed)
        means = means.detach().clone().to(dtype=torch.float32)
        count = means.shape[0]

        scales = torch.as_tensor(initial_scale, dtype=means.dtype, device=means.device)
        if scales.ndim == 0:
            scales = scales.expand(count, 3).clone()
        elif scales.shape == (3,):
            scales = scales[None, :].expand(count, -1).clone()
        elif scales.shape != (count, 3):
            raise ValueError("initial_scale must be scalar, [3], or [N,3]")
        if torch.any(scales <= 0):
            raise ValueError("all Gaussian scales must be positive")

        generator = _initialization_generator
        if generator is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed)
        quaternions = torch.rand((count, 4), generator=generator, dtype=means.dtype).to(means.device)
        if initial_reflectance is None:
            rgb = torch.rand((count, 3), generator=generator, dtype=means.dtype).to(means.device)
        else:
            rgb = torch.as_tensor(initial_reflectance, dtype=means.dtype, device=means.device)
            if rgb.ndim == 0:
                rgb = rgb.expand(count, 3).clone()
            elif rgb.shape == (count,):
                rgb = rgb[:, None].expand(-1, 3).clone()
            elif rgb.shape == (3,):
                rgb = rgb[None, :].expand(count, -1).clone()
            elif rgb.shape != (count, 3):
                raise ValueError("initial_reflectance must be scalar, [3], [N], or [N,3]")
        coefficients = torch.zeros(
            (count, (self.sh_degree + 1) ** 2, 3), dtype=means.dtype, device=means.device
        )
        coefficients[:, 0, :] = (rgb - 0.5) / _C0

        self.means = nn.Parameter(means)
        self.log_scales = nn.Parameter(torch.log(scales))
        self.quaternions = nn.Parameter(quaternions)
        self.opacity_logits = nn.Parameter(
            _logit(torch.full((count,), initial_opacity, dtype=means.dtype, device=means.device))
        )
        self.noise_probability_logits = nn.Parameter(
            _logit(
                torch.full(
                    (count,), initial_noise_probability, dtype=means.dtype, device=means.device
                )
            )
        )
        self.sh0 = nn.Parameter(coefficients[:, :1, :])
        self.shN = nn.Parameter(coefficients[:, 1:, :])

    @classmethod
    def random_scene(
        cls,
        num_gaussians: int = 20_000,
        extent: float = 1.0,
        planar_initialization: bool = True,
        seed: int = 42,
        **kwargs,
    ) -> "RadarSplatModel":
        """Official random initialization (release sets all initial z to zero)."""

        if num_gaussians < 1 or extent <= 0:
            raise ValueError("num_gaussians and extent must be positive")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        means = 0.5 * extent * (
            torch.rand((num_gaussians, 3), generator=generator) * 2.0 - 1.0
        )
        if planar_initialization:
            means[:, 2] = 0.0
        # The pinned release draws random SH colours immediately after the
        # means and only then draws quaternions.  Preserve that single-stream
        # ordering when reflectance was not supplied explicitly.
        if kwargs.get("initial_reflectance") is None:
            kwargs["initial_reflectance"] = torch.rand(
                (num_gaussians, 3), generator=generator
            )
        # Keep the release's single random stream across means, initial SH
        # colours, and rotations.  Re-seeding inside ``__init__`` would correlate
        # the first three quaternion samples with each Gaussian's xyz mean.
        return cls(
            means=means,
            seed=seed,
            _initialization_generator=generator,
            **kwargs,
        )

    @property
    def num_gaussians(self) -> int:
        return int(self.means.shape[0])

    @property
    def scales(self) -> Tensor:
        return torch.exp(self.log_scales)

    @property
    def opacity(self) -> Tensor:
        return torch.sigmoid(self.opacity_logits)

    @property
    def noise_probability(self) -> Tensor:
        return torch.sigmoid(self.noise_probability_logits)

    @property
    def sh_coefficients(self) -> Tensor:
        return torch.cat((self.sh0, self.shN), dim=1)

    def physical_parameters(self) -> Dict[str, Tensor]:
        """Regularizer-ready physical and raw parameter tensors."""

        return {
            "means": self.means,
            "scales": self.scales,
            "log_scales": self.log_scales,
            "quaternions": F.normalize(self.quaternions, p=2, dim=-1),
            "opacity": self.opacity,
            "opacity_logits": self.opacity_logits,
            "noise_probability": self.noise_probability,
            "noise_probability_logits": self.noise_probability_logits,
            "sh_coefficients": self.sh_coefficients,
        }

    def native_regularizers(self, max_scale: float) -> Dict[str, Tensor]:
        """The release's max-size and ``alpha+eta`` regularization terms."""

        if max_scale <= 0:
            raise ValueError("max_scale must be positive")
        return {
            "max_size": torch.relu(self.scales - max_scale).mean(),
            "opacity_noise": torch.relu(self.opacity + self.noise_probability - 1.0).mean(),
        }

    def view_reflectance(self, sensor_to_world: Tensor, active_sh_degree: Optional[int] = None) -> Tensor:
        """Return per-view, per-Gaussian reflectance ``rho`` with release activation.

        Directions are Cartesian sensor-to-Gaussian vectors.  The pinned
        executable instead subtracts a Cartesian translation from already
        spherical means; retaining that coordinate bug would make rotated
        B787 viewpoints physically inconsistent.
        """

        sensor_to_world = torch.as_tensor(
            sensor_to_world, dtype=self.means.dtype, device=self.means.device
        )
        if sensor_to_world.shape == (4, 4):
            sensor_to_world = sensor_to_world.unsqueeze(0)
        degree = self.sh_degree if active_sh_degree is None else int(active_sh_degree)
        if degree < 0 or degree > self.sh_degree:
            raise ValueError("active_sh_degree must be between zero and the model degree")
        sensor_position = sensor_to_world[:, :3, 3]
        direction = self.means[None, :, :] - sensor_position[:, None, :]
        bases = eval_sh_bases(degree, direction)
        coefficients = self.sh_coefficients[:, : (degree + 1) ** 2, 0]
        reflectance = torch.einsum("bnk,nk->bn", bases, coefficients)
        return torch.clamp(reflectance + 0.5, min=1.0e-6, max=1.0)

    def project_gaussians(
        self,
        sensor_to_world: Tensor,
        grid: RadarSplatGrid,
        *,
        retain_projected_mean_grad: bool = False,
    ) -> Dict[str, Tensor]:
        """Expose release-like projection state for rendering and densification.

        Pixel means/radii/visibility are suitable inputs to a trainer-side
        DefaultStrategy schedule.  If ``retain_projected_mean_grad`` is true,
        ``pixel_means.grad`` is populated after backward.  This primitive also
        exposes full sensor/spherical 3-D means and covariances for a separately
        labelled B787 matched-filter-volume adapter.
        """

        covariance_world = covariance_from_quaternion_scale(self.quaternions, self.scales)
        projection = cartesian_to_spherical_gaussians(
            self.means, covariance_world, sensor_to_world
        )
        means2d = projection["means_spherical"][..., :2]
        covariance2d = projection["covariances_spherical"][..., :2, :2]
        resolution = means2d.new_tensor(
            [
                grid.range_resolution_m,
                math.radians(grid.intermediate_azimuth_resolution_deg),
            ]
        )
        covariance_pixels = covariance2d / (
            resolution[None, None, :, None] * resolution[None, None, None, :]
        )
        identity = torch.eye(2, dtype=means2d.dtype, device=means2d.device)
        covariance_pixels = covariance_pixels + grid.eps2d_pixels * identity
        covariance2d = covariance_pixels * (
            resolution[None, None, :, None] * resolution[None, None, None, :]
        )
        pixel_means, _ = _physical_gaussians_to_pixels(
            means2d, covariance2d, grid
        )
        if retain_projected_mean_grad and pixel_means.requires_grad:
            pixel_means.retain_grad()
        # Route rasterization through pixel_means (as gsplat does) so the
        # retained screen-space gradient is meaningful to densification.
        raster_azimuth = (
            grid.intermediate_azimuth_start_rad
            + pixel_means[..., 1]
            * math.radians(grid.intermediate_azimuth_resolution_deg)
        )
        if grid.is_full_azimuth:
            raster_azimuth = torch.remainder(raster_azimuth, 2.0 * math.pi)
        means2d_for_raster = torch.stack(
            (
                grid.range_start_m
                + pixel_means[..., 0] * grid.range_resolution_m,
                raster_azimuth,
            ),
            dim=-1,
        )
        pixel_radii, valid_covariance = _release_projected_radius(covariance_pixels)
        visible = (
            valid_covariance
            & (pixel_means[..., 0] + pixel_radii > 0.0)
            & (pixel_means[..., 0] - pixel_radii < grid.num_range_bins)
        )
        if not grid.is_full_azimuth:
            visible = (
                visible
                & (pixel_means[..., 1] + pixel_radii > 0.0)
                & (
                    pixel_means[..., 1] - pixel_radii
                    < grid.intermediate_azimuth_bins
                )
            )
        projection.update(
            {
                "means_range_azimuth": means2d_for_raster,
                "covariances_range_azimuth": covariance2d,
                "pixel_means": pixel_means,
                "pixel_covariances": covariance_pixels,
                "pixel_radii": pixel_radii,
                "visibility": visible,
            }
        )
        return projection

    def gaussian_view_state(
        self,
        sensor_to_world: Tensor,
        grid: RadarSplatGrid,
        effects: Optional[RadarSplatEffects] = None,
        *,
        active_sh_degree: Optional[int] = None,
        elevation_gain: Optional[Tensor] = None,
        retain_projected_mean_grad: bool = False,
    ) -> Dict[str, Tensor]:
        """Projection plus all per-Gaussian quantities before rasterization."""

        effects = effects or RadarSplatEffects()
        projection = self.project_gaussians(
            sensor_to_world, grid, retain_projected_mean_grad=retain_projected_mean_grad
        )
        reflectance = self.view_reflectance(sensor_to_world, active_sh_degree)
        batch = projection["means_range_azimuth"].shape[0]
        opacity = self.opacity[None, :].expand(batch, -1)
        learned_noise = self.noise_probability[None, :].expand(batch, -1)
        eta = learned_noise if effects.use_noise_probability else torch.zeros_like(learned_noise)
        if elevation_gain is None:
            gain = torch.ones_like(opacity)
        else:
            gain = torch.as_tensor(elevation_gain, dtype=opacity.dtype, device=opacity.device)
            if gain.shape == (self.num_gaussians,):
                gain = gain[None, :].expand(batch, -1)
            if gain.shape != opacity.shape:
                raise ValueError("elevation_gain must have shape [N] or [B,N]")
            if torch.any(gain < 0):
                raise ValueError("elevation_gain must be nonnegative")
        # Eq. 11 uses squared antenna gain.  Unity is release-faithful because
        # the pinned code has a TODO rather than a released elevation profile.
        power_gain = gain.square()
        if effects.range_power_exponent:
            radius = projection["means_spherical"][..., 0].clamp_min(
                0.5 * grid.range_resolution_m
            )
            power_gain = power_gain * radius.pow(-effects.range_power_exponent)
        clean = torch.clamp(opacity * reflectance, min=effects.output_floor, max=1.0)
        noise = torch.clamp(eta * reflectance, min=effects.output_floor, max=1.0)
        combined = torch.clamp(
            opacity + eta, min=effects.output_floor, max=1.0
        ) * reflectance
        projection.update(
            {
                "gaussian_opacity": opacity,
                "gaussian_noise_probability": eta,
                "gaussian_learned_noise_probability": learned_noise,
                "gaussian_reflectance": reflectance,
                "gaussian_power_gain": power_gain,
                "gaussian_clean_power": clean * power_gain,
                "gaussian_noise_power": noise * power_gain,
                "gaussian_scene_power": combined * power_gain,
            }
        )
        return projection

    @staticmethod
    def _process_image(image: Tensor, grid: RadarSplatGrid, effects: RadarSplatEffects) -> Tensor:
        if effects.use_spectral_leakage:
            image = spectral_leakage(
                image, grid.range_resolution_m, grid.spectral_leakage_width_m
            )
        if effects.use_azimuth_antenna_gain:
            image = azimuth_antenna_gain_projection(
                image,
                output_resolution_deg=grid.output_azimuth_resolution_deg,
                beamwidth_deg=grid.azimuth_beamwidth_deg,
                input_resolution_deg=grid.intermediate_azimuth_resolution_deg,
                circular=grid.is_full_azimuth,
            )
        else:
            # Sample zero is already aligned to the first output bin centre.
            # Disabling the beam filter must not change the measurement grid.
            image = image[..., ::grid.azimuth_stride, :]
        return image

    def render(
        self,
        sensor_to_world: Tensor,
        grid: RadarSplatGrid,
        effects: Optional[RadarSplatEffects] = None,
        *,
        active_sh_degree: Optional[int] = None,
        elevation_gain: Optional[Tensor] = None,
        multipath_power: Optional[Tensor] = None,
        gaussian_chunk_size: int = 128,
        max_candidate_pairs: Optional[int] = None,
        raster_backend: str = "torch_sparse_reference",
        retain_projected_mean_grad: bool = False,
    ) -> Dict[str, Tensor]:
        """Render the official 2-D azimuth/range products.

        ``multipath_power`` must already be reconstructed on the final sensor
        grid, matching the released ``get_multipath_model`` boundary.  It is
        never fabricated when absent.
        """

        effects = effects or RadarSplatEffects()
        state = self.gaussian_view_state(
            sensor_to_world,
            grid,
            effects,
            active_sh_degree=active_sh_degree,
            elevation_gain=elevation_gain,
            retain_projected_mean_grad=retain_projected_mean_grad,
        )
        if effects.use_azimuth_antenna_gain and not grid.is_full_azimuth:
            raster_grid, final_azimuth_crop, intermediate_azimuth_crop = (
                extended_local_azimuth_grid(grid)
            )
        else:
            raster_grid = grid
            final_azimuth_crop = slice(0, grid.output_azimuth_bins)
            intermediate_azimuth_crop = slice(0, grid.intermediate_azimuth_bins)
        if effects.use_spectral_leakage:
            range_halo = spectral_filter_halo_bins(grid)
            raster_grid = replace(
                raster_grid,
                num_range_bins=grid.num_range_bins + 2 * range_halo,
                range_start_m=grid.range_start_m
                - range_halo * grid.range_resolution_m,
            )
            final_range_crop = slice(range_halo, range_halo + grid.num_range_bins)
        else:
            range_halo = 0
            final_range_crop = slice(0, grid.num_range_bins)
        attributes = torch.stack(
            (
                state["gaussian_scene_power"],
                state["gaussian_opacity"],
                state["gaussian_noise_probability"],
                state["gaussian_clean_power"],
                state["gaussian_noise_power"],
                state["gaussian_reflectance"],
            ),
            dim=-1,
        )
        raw = additive_gaussian_rasterization(
            state["means_range_azimuth"],
            state["covariances_range_azimuth"],
            attributes,
            raster_grid,
            gaussian_chunk_size=gaussian_chunk_size,
            max_candidate_pairs=max_candidate_pairs,
            backend=raster_backend,
            # The pinned radar path explicitly sets every intersection depth to
            # zero because the additive observable is depth-order independent.
            # Keeping the same key here also freezes floating reduction order.
            sort_depths=None,
            alpha_cutoff=effects.raster_alpha_cutoff,
        )
        names = (
            "scene_power",
            "occupancy",
            "noise_probability",
            "clean_power",
            "noise_power",
            "reflectance",
        )
        output: Dict[str, Tensor] = {}
        for index, name in enumerate(names):
            image = raw[..., index]
            if name != "reflectance":
                ceiling = effects.output_ceiling
                if name in {"occupancy", "noise_probability"} and effects.probability_ceiling is not None:
                    ceiling = effects.probability_ceiling
                image = (
                    image.clamp_min(effects.output_floor)
                    if ceiling is None
                    else image.clamp(effects.output_floor, ceiling)
                )
            image = self._process_image(image, raster_grid, effects)
            image = image[:, final_azimuth_crop, :]
            output[name] = image[:, :, final_range_crop]

        final_power = output["scene_power"]
        if effects.use_multipath:
            if multipath_power is None:
                raise ValueError("use_multipath=True requires an externally reconstructed multipath_power")
            multipath = torch.as_tensor(
                multipath_power, dtype=final_power.dtype, device=final_power.device
            )
            if multipath.shape == final_power.shape[1:] and final_power.shape[0] == 1:
                multipath = multipath.unsqueeze(0)
            if multipath.shape != final_power.shape:
                raise ValueError(
                    f"multipath_power must match final image shape {tuple(final_power.shape)}"
                )
            final_power = final_power + effects.multipath_weight * multipath
        output["final_power"] = (
            final_power.clamp_min(effects.output_floor)
            if effects.output_ceiling is None
            else final_power.clamp(effects.output_floor, effects.output_ceiling)
        )
        output["raw_intermediate"] = raw[
            :, intermediate_azimuth_crop, final_range_crop, :
        ]
        if raster_grid is not grid or range_halo:
            output["raw_intermediate_with_halo"] = raw
        output.update(state)
        # Expose raw/logit parameters directly for loss and checkpoint code.
        output.update(
            {
                f"parameter_{name}": value
                for name, value in self.physical_parameters().items()
            }
        )
        return output

    def forward(self, sensor_to_world: Tensor, grid: RadarSplatGrid, **kwargs) -> Dict[str, Tensor]:
        return self.render(sensor_to_world, grid, **kwargs)

    def _replace_rows(self, rows: Mapping[str, Tensor], append: bool) -> Dict[str, Tuple[nn.Parameter, nn.Parameter]]:
        replacements: Dict[str, Tuple[nn.Parameter, nn.Parameter]] = {}
        expected = set(_PER_GAUSSIAN_PARAMETERS)
        if set(rows) != expected:
            missing, extra = sorted(expected - set(rows)), sorted(set(rows) - expected)
            raise ValueError(f"row update must cover every Gaussian parameter; missing={missing}, extra={extra}")
        lengths = {name: int(value.shape[0]) for name, value in rows.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"all replacement rows need the same first dimension, got {lengths}")
        for name in _PER_GAUSSIAN_PARAMETERS:
            old = getattr(self, name)
            value = rows[name].to(device=old.device, dtype=old.dtype)
            if value.shape[1:] != old.shape[1:]:
                raise ValueError(f"{name} trailing shape must be {tuple(old.shape[1:])}")
            data = torch.cat((old.detach(), value.detach()), dim=0) if append else value.detach()
            new = nn.Parameter(data, requires_grad=old.requires_grad)
            setattr(self, name, new)
            replacements[name] = (old, new)
        return replacements

    @torch.no_grad()
    def append_gaussians(
        self,
        source_indices: Tensor,
        overrides: Optional[Mapping[str, Tensor]] = None,
    ) -> Dict[str, Tuple[nn.Parameter, nn.Parameter]]:
        """Clone rows consistently and return mappings for optimizer migration.

        The helper does not mutate an optimizer.  A densification strategy must
        replace each old parameter reference with the corresponding new one and
        append zero-valued optimizer-state rows.  This explicit boundary avoids
        silently discarding Adam moments.
        """

        indices = torch.as_tensor(source_indices, device=self.means.device, dtype=torch.long)
        if indices.ndim != 1 or indices.numel() < 1:
            raise ValueError("source_indices must be a nonempty 1-D tensor")
        if int(indices.min()) < 0 or int(indices.max()) >= self.num_gaussians:
            raise IndexError("source_indices contain an invalid Gaussian row")
        overrides = dict(overrides or {})
        unknown = set(overrides) - set(_PER_GAUSSIAN_PARAMETERS)
        if unknown:
            raise ValueError(f"unknown Gaussian parameters in overrides: {sorted(unknown)}")
        rows = {
            name: overrides.get(name, getattr(self, name).detach()[indices].clone())
            for name in _PER_GAUSSIAN_PARAMETERS
        }
        return self._replace_rows(rows, append=True)

    @torch.no_grad()
    def prune_gaussians(self, keep_mask: Tensor) -> Dict[str, Tuple[nn.Parameter, nn.Parameter]]:
        """Keep selected rows consistently and return mappings for optimizer migration."""

        keep = torch.as_tensor(keep_mask, device=self.means.device, dtype=torch.bool)
        if keep.shape != (self.num_gaussians,):
            raise ValueError(f"keep_mask must have shape ({self.num_gaussians},)")
        if not torch.any(keep):
            raise ValueError("cannot prune every Gaussian")
        rows = {name: getattr(self, name).detach()[keep].clone() for name in _PER_GAUSSIAN_PARAMETERS}
        return self._replace_rows(rows, append=False)

    def config(self) -> Dict[str, object]:
        return {
            "method": METHOD_NAME,
            "renderer_backend": "torch_sparse_reference",
            "num_gaussians": self.num_gaussians,
            "sh_degree": self.sh_degree,
            "seed": self.seed,
        }


def renderer_config(grid: RadarSplatGrid, effects: RadarSplatEffects) -> Dict[str, object]:
    """Serializable rendering provenance for checkpoints and result manifests."""

    return {
        "method": METHOD_NAME,
        "renderer_backend": "torch_sparse_reference",
        "grid": asdict(grid),
        "effects": asdict(effects),
        "noise_probability_semantics": "joint receiver-saturation/speckle eta branch",
        "multipath_source": "external precomputed map; never synthesized when absent",
        "release_elevation_gain": "not implemented (unity); optional measured hook exposed",
        "release_range_power_exponent": 0.0,
        "raster_support": "ceil(3sigma), release 16x16 projected tiles",
        "per_channel_alpha_cutoff": _GAUSSIAN_ALPHA_CUTOFF,
        "azimuth_phase_convention": "output_bin_center_aligned",
        "intermediate_azimuth_start_offset_deg": (
            grid.intermediate_azimuth_start_offset_deg
        ),
        "local_crop_filtering": "render azimuth/range halo, filter, crop",
        "b787_physical_coordinate_adaptations": {
            "sensor_frame_covariance_rotation": True,
            "cartesian_sh_direction": True,
        },
    }
