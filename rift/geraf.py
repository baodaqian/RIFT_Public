"""GeRaF v1 neural RF surface renderer.

This module implements the model disclosed in Lu, Shanbhag, and Al
Hassanieh, *GeRaF: Neural Geometry Reconstruction from Radio Frequency
Signals*, arXiv:2605.29097v2.  The public paper specifies an
eight-layer width-256 SDF MLP with ten positional-encoding levels, a
four-layer width-256 reflectivity MLP, a learned global transmit amplitude,
NeuS-style logistic SDF opacity, primary-ray (lensless) sampling, a
shifted-Lambertian reflection factor, and free-space path decay.

The historical implementation remains the default library/checkpoint format.
The full trainer selects the versioned ``hardened_v1`` implementation, which
uses the released SDF network and renderer/mask evidence from GeRaF-SENS.
See ``docs/GERAF_V1_HARDENING.md`` for the pinned source and the differences
between the v1 paper, the released stage-1 configuration, and this adaptation.

The differentiable matched filter is deliberately *not* reimplemented here.
GeRaF is trained on matched-filter magnitude ``|MF|`` (called ``P`` or power
in the paper), but the filter must match the acquisition geometry and signal
convention.  ``render_magnitude`` accepts a signal-tracing callback and a
matched-filter callback, while ``apply_matched_filter_magnitude`` also accepts
an already predicted complex response.  Neither API squares the magnitude.

Conventions
-----------
* SDF is positive outside and zero on the surface (the NeuS convention).
* ``omega_i`` points from transmitter to surface and ``omega_r`` from surface
  to receiver.  Their ideal specular alignment has response one.
* ``free_space_amplitude_decay`` is the bistatic extension of GeRaF Eq. 2:
  ``1 / ((4*pi)^2 * R_tx * R_rx)``.  It reduces exactly to
  ``1 / (4*pi*u)^2`` in the co-located case used by the paper.
* Discrete volume weights use ``T_tx * T_rx * alpha``.  For a co-located
  antenna this is GeRaF's ``T^2 rho du`` after alpha discretization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F


PAPER_ID = "arXiv:2605.29097v2"
METHOD_NAME = "GeRaF v1 independent bistatic adaptation"
_EPS = 1.0e-12


def _require_xyz(name: str, value: torch.Tensor) -> None:
    if not torch.is_tensor(value) or value.ndim < 1 or value.shape[-1] != 3:
        shape = getattr(value, "shape", None)
        raise ValueError(f"{name} must be a tensor ending in dimension 3, got {shape}")


def _positive_scalar_like(value: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    out = value if torch.is_tensor(value) else reference.new_tensor(float(value))
    out = out.to(device=reference.device, dtype=reference.dtype)
    if out.numel() != 1 or bool((out <= 0).detach().item()):
        raise ValueError("expected a positive scalar")
    return out


def safe_normalize(value: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Normalize vectors along their last dimension without producing NaNs."""
    _require_xyz("value", value)
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(eps)


class SinusoidalPositionEncoding(nn.Module):
    """NeRF/NeuS sinusoidal encoding with ``2**level`` frequency bands.

    The paper does not specify coordinate normalization. Historical local
    checkpoints use metric coordinates; the hardened recipe explicitly records
    extent normalization and raw XYZ for a generic geometric initialization.
    """

    def __init__(
        self,
        extent: float,
        n_levels: int = 10,
        include_input: bool = False,
        coordinate_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if extent <= 0:
            raise ValueError("extent must be positive")
        if n_levels < 0:
            raise ValueError("n_levels must be non-negative")
        if coordinate_scale <= 0:
            raise ValueError("coordinate_scale must be positive")
        self.extent = float(extent)
        self.n_levels = int(n_levels)
        self.include_input = bool(include_input)
        self.coordinate_scale = float(coordinate_scale)
        frequencies = 2.0 ** torch.arange(self.n_levels, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies)

    @property
    def output_dim(self) -> int:
        return 3 * (2 * self.n_levels + int(self.include_input))

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        _require_xyz("xyz", xyz)
        x = xyz * self.coordinate_scale
        if self.n_levels == 0:
            return x if self.include_input else x[..., :0]
        phase = math.pi * x.unsqueeze(-2) * self.frequencies.to(x).view(
            *((1,) * (x.ndim - 1)), self.n_levels, 1
        )
        encoded = [torch.sin(phase).flatten(-2), torch.cos(phase).flatten(-2)]
        if self.include_input:
            encoded.insert(0, x)
        return torch.cat(encoded, dim=-1)


class GeRaFSDFNetwork(nn.Module):
    """GeRaF SDF MLP (paper default: 8 layers, width 256, 10 PE levels).

    GeRaF cites NeuS for its S-density construction but does not disclose its
    hidden activation or skip wiring.  Softplus(beta=100) and a middle skip
    are the standard NeuS choices; both are exposed in ``config``.
    """

    def __init__(
        self,
        extent: float,
        n_levels: int = 10,
        hidden_dim: int = 256,
        n_layers: int = 8,
        skip_layer: Optional[int] = 4,
        hidden_activation: str = "softplus",
        softplus_beta: float = 100.0,
        encoding_include_input: bool = False,
        encoding_coordinate_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or n_layers <= 0:
            raise ValueError("hidden_dim and n_layers must be positive")
        if n_levels != 10:
            raise ValueError(
                "reportable GeRaF requires exactly 10 PE levels = 60 sin/cos dimensions"
            )
        if skip_layer is not None and not (0 < skip_layer < n_layers):
            raise ValueError("skip_layer must be inside the hidden stack or None")
        if softplus_beta <= 0:
            raise ValueError("softplus_beta must be positive")
        if hidden_activation != "softplus":
            raise ValueError("the reportable GeRaF SDF hidden activation must be softplus")
        if encoding_include_input:
            raise ValueError(
                "the reportable ten-level GeRaF encoding is exactly 60 sin/cos "
                "dimensions and excludes raw xyz"
            )
        if encoding_coordinate_scale != 1.0:
            raise ValueError(
                "reportable GeRaF applies positional encoding directly to metric "
                "coordinates, so encoding_coordinate_scale must equal 1.0"
            )
        self.extent = float(extent)
        self.n_levels = int(n_levels)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.skip_layer = None if skip_layer is None else int(skip_layer)
        self.hidden_activation = hidden_activation
        self.softplus_beta = float(softplus_beta)
        self.encoding_include_input = False
        self.encoding_coordinate_scale = float(encoding_coordinate_scale)

        self.encoding = SinusoidalPositionEncoding(
            extent,
            n_levels,
            include_input=False,
            coordinate_scale=self.encoding_coordinate_scale,
        )
        encoded_dim = self.encoding.output_dim
        layers = []
        for index in range(self.n_layers):
            in_dim = encoded_dim if index == 0 else self.hidden_dim
            if index == self.skip_layer:
                in_dim += encoded_dim
            layer = nn.Linear(in_dim, self.hidden_dim)
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(self.hidden_dim, 1)
        nn.init.normal_(self.output.weight, mean=0.0, std=1.0e-4)
        nn.init.zeros_(self.output.bias)
        self.activation = nn.Softplus(beta=self.softplus_beta)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        encoded = self.encoding(xyz)
        h = encoded
        for index, layer in enumerate(self.layers):
            if index == self.skip_layer:
                h = torch.cat((h, encoded), dim=-1) / math.sqrt(2.0)
            h = self.activation(layer(h))
        return self.output(h).squeeze(-1)

    def gradient(
        self, xyz: torch.Tensor, *, create_graph: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return SDF and its spatial gradient (the outward surface normal)."""
        query = xyz.to(next(self.parameters()))
        # Evaluation normals still require spatial autograd under no_grad().
        # Retain the SDF graph even if callers request first-order normals and
        # subsequently differentiate a loss through the SDF values themselves.
        with torch.enable_grad():
            if not query.requires_grad:
                query = query.detach().requires_grad_(True)
            sdf = self(query)
            gradient = torch.autograd.grad(
                sdf, query, grad_outputs=torch.ones_like(sdf),
                create_graph=create_graph, retain_graph=True, only_inputs=True,
            )[0]
        return sdf, gradient

    def config(self) -> Dict[str, object]:
        return {
            "extent": self.extent,
            "n_levels": self.n_levels,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "skip_layer": self.skip_layer,
            "encoding": "sin_cos_power_of_two",
            "encoding_include_input": self.encoding_include_input,
            "encoding_coordinate_scale": self.encoding_coordinate_scale,
            "encoding_coordinates": "metric_meters_no_extent_normalization",
            "encoding_output_dim": self.encoding.output_dim,
            "hidden_activation": self.hidden_activation,
            "softplus_beta": self.softplus_beta,
        }


class GeRaFReflectivityNetwork(nn.Module):
    """Position-to-reflection-coefficient MLP (paper default: 4 x 256).

    The paper requires a non-negative reflection coefficient but does not
    publish an encoding or output activation for this network.  Direct xyz
    input and ``softplus`` are the conservative defaults; positional encoding,
    ``sigmoid``, and ``none`` remain explicit for exact future source matching.
    """

    def __init__(
        self,
        extent: float,
        n_levels: int = 0,
        hidden_dim: int = 256,
        n_layers: int = 4,
        output_activation: str = "softplus",
        softplus_beta: float = 1.0,
        encoding_coordinate_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or n_layers <= 0:
            raise ValueError("hidden_dim and n_layers must be positive")
        if output_activation not in {"softplus", "sigmoid", "none"}:
            raise ValueError("output_activation must be softplus, sigmoid, or none")
        if softplus_beta <= 0:
            raise ValueError("softplus_beta must be positive")
        if encoding_coordinate_scale != 1.0:
            raise ValueError(
                "reportable GeRaF reflectivity inputs use metric coordinates, so "
                "encoding_coordinate_scale must equal 1.0"
            )
        self.extent = float(extent)
        self.n_levels = int(n_levels)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.output_activation = output_activation
        self.softplus_beta = float(softplus_beta)
        self.encoding_coordinate_scale = float(encoding_coordinate_scale)

        # The default n_levels=0 is the paper-underspecified direct-xyz input.
        # Any enabled PE follows the reportable sin/cos-only convention.
        self.encoding = SinusoidalPositionEncoding(
            extent,
            n_levels,
            include_input=(n_levels == 0),
            coordinate_scale=self.encoding_coordinate_scale,
        )
        layers = []
        in_dim = self.encoding.output_dim
        for _ in range(self.n_layers):
            layer = nn.Linear(in_dim, self.hidden_dim)
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            in_dim = self.hidden_dim
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(self.hidden_dim, 1)
        nn.init.normal_(self.output.weight, mean=0.0, std=1.0e-4)
        nn.init.zeros_(self.output.bias)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        h = self.encoding(xyz)
        for layer in self.layers:
            h = F.relu(layer(h), inplace=False)
        raw = self.output(h).squeeze(-1)
        if self.output_activation == "softplus":
            return F.softplus(raw, beta=self.softplus_beta)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(raw)
        return raw

    def config(self) -> Dict[str, object]:
        return {
            "extent": self.extent,
            "n_levels": self.n_levels,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "encoding_coordinate_scale": self.encoding_coordinate_scale,
            "encoding_coordinates": "metric_meters_no_extent_normalization",
            "hidden_activation": "relu",
            "output_activation": self.output_activation,
            "softplus_beta": self.softplus_beta,
        }


class LearnableTxAmplitude(nn.Module):
    """The paper's single global effective transmit-amplitude parameter."""

    def __init__(self, init_amplitude: float = 1.0, learnable: bool = True) -> None:
        super().__init__()
        if init_amplitude <= 0:
            raise ValueError("init_amplitude must be positive")
        self.log_amplitude = nn.Parameter(
            torch.tensor(math.log(float(init_amplitude)), dtype=torch.float32),
            requires_grad=bool(learnable),
        )

    def forward(self) -> torch.Tensor:
        return torch.exp(self.log_amplitude)

    @property
    def value(self) -> float:
        return float(self().detach().item())


class LearnableSDFSharpness(nn.Module):
    """Positive inverse logistic scale ``s`` used by the NeuS S-density."""

    def __init__(self, init_inv_s: float = 64.0, learnable: bool = True) -> None:
        super().__init__()
        if init_inv_s <= 0:
            raise ValueError("init_inv_s must be positive")
        self.log_inv_s = nn.Parameter(
            torch.tensor(math.log(float(init_inv_s)), dtype=torch.float32),
            requires_grad=bool(learnable),
        )

    def forward(self) -> torch.Tensor:
        return torch.exp(self.log_inv_s)


def logistic_sdf_cdf(sdf: torch.Tensor, inv_s: torch.Tensor | float) -> torch.Tensor:
    """``Phi_s(f)`` from GeRaF Appendix B / NeuS."""
    scale = _positive_scalar_like(inv_s, sdf)
    return torch.sigmoid(scale * sdf)


def logistic_sdf_pdf(sdf: torch.Tensor, inv_s: torch.Tensor | float) -> torch.Tensor:
    """Derivative of ``Phi_s(f)`` with respect to the SDF value ``f``."""
    scale = _positive_scalar_like(inv_s, sdf)
    cdf = torch.sigmoid(scale * sdf)
    return scale * cdf * (1.0 - cdf)


def neus_opaque_density(
    sdf: torch.Tensor,
    sdf_gradient: torch.Tensor,
    ray_direction: torch.Tensor,
    inv_s: torch.Tensor | float,
    *,
    eps: float = _EPS,
) -> torch.Tensor:
    """GeRaF Eq. 15: ``max(-d Phi_s/du / Phi_s, 0)``.

    ``sdf``, ``sdf_gradient`` and ``ray_direction`` must be broadcastable;
    the final dimension of the latter two is xyz.
    """
    _require_xyz("sdf_gradient", sdf_gradient)
    _require_xyz("ray_direction", ray_direction)
    direction = safe_normalize(ray_direction)
    cdf = logistic_sdf_cdf(sdf, inv_s)
    pdf = logistic_sdf_pdf(sdf, inv_s)
    d_sdf_du = (sdf_gradient * direction).sum(dim=-1)
    return (-pdf * d_sdf_du / cdf.clamp_min(eps)).clamp_min(0.0)


def neus_sdf_to_alpha(
    sdf: torch.Tensor,
    inv_s: torch.Tensor | float,
    *,
    terminal_sdf: Optional[torch.Tensor] = None,
    eps: float = _EPS,
) -> torch.Tensor:
    """Discretize the NeuS/GeRaF density into per-sample alpha values.

    Samples are ordered from the radar into the scene along the final axis.
    The returned tensor has the same shape as ``sdf``.  Its final entry is
    zero unless ``terminal_sdf`` supplies the SDF at the far endpoint of the
    last interval.
    """
    if sdf.ndim < 1 or sdf.shape[-1] < 1:
        raise ValueError("sdf must have at least one depth sample")
    cdf = logistic_sdf_cdf(sdf, inv_s)
    if terminal_sdf is None:
        next_cdf = torch.cat((cdf[..., 1:], cdf[..., -1:]), dim=-1)
    else:
        tail = logistic_sdf_cdf(terminal_sdf, inv_s).unsqueeze(-1)
        if tail.shape[:-1] != cdf.shape[:-1]:
            raise ValueError("terminal_sdf must match sdf without its depth dimension")
        next_cdf = torch.cat((cdf[..., 1:], tail), dim=-1)
    return ((cdf - next_cdf).clamp_min(0.0) / cdf.clamp_min(eps)).clamp(0.0, 1.0)


def transmittance_from_alpha(alpha: torch.Tensor) -> torch.Tensor:
    """Exclusive one-way transmittance ``prod_{j<i}(1-alpha_j)``."""
    if alpha.ndim < 1 or alpha.shape[-1] < 1:
        raise ValueError("alpha must have at least one depth sample")
    survival = (1.0 - alpha).clamp(0.0, 1.0)
    ones = torch.ones_like(survival[..., :1])
    return torch.cumprod(torch.cat((ones, survival[..., :-1]), dim=-1), dim=-1)


def lensless_transmittance_correction(
    primary_transmittance: torch.Tensor,
    primary_start_sdf: torch.Tensor,
    real_start_sdf: torch.Tensor,
    inv_s: torch.Tensor | float,
    *,
    detach_start_cdf: bool = True,
    clamp: bool = True,
) -> torch.Tensor:
    """GeRaF Eq. 7 start-CDF correction for real antenna rays.

    Inputs use the renderer's natural shapes: primary transmittance ``[R,Z]``,
    primary start SDF ``[R]``, and real antenna start SDF ``[B]``.  The output
    is ``[B,R,Z]``.  GeRaF explicitly excludes the start-CDF correction from
    backpropagation; that behavior is the default here.
    """
    if primary_transmittance.ndim != 2:
        raise ValueError("primary_transmittance must have shape [num_rays,num_depth]")
    if primary_start_sdf.shape != primary_transmittance.shape[:-1]:
        raise ValueError("primary_start_sdf must have shape [num_rays]")
    if real_start_sdf.ndim != 1:
        raise ValueError("real_start_sdf must have shape [num_real_rays]")
    primary_cdf = logistic_sdf_cdf(primary_start_sdf, inv_s)
    real_cdf = logistic_sdf_cdf(real_start_sdf, inv_s)
    if detach_start_cdf:
        primary_cdf = primary_cdf.detach()
        real_cdf = real_cdf.detach()
    corrected = (
        primary_transmittance.unsqueeze(0)
        - primary_cdf.view(1, -1, 1)
        + real_cdf.view(-1, 1, 1)
    )
    return corrected.clamp(0.0, 1.0) if clamp else corrected


def shifted_lambertian_bistatic(
    points: torch.Tensor,
    normals: torch.Tensor,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
    exponent: float = 1.0,
) -> torch.Tensor:
    """GeRaF Eqs. 11--12 for broadcastable bistatic geometry.

    The paper's published directional factor is simply
    ``max(dot(reflect(omega_i,n), omega_r), 0)``.  ``exponent`` is exposed for
    directive-lobe ablations but defaults to the paper's linear factor.
    """
    for name, value in (
        ("points", points),
        ("normals", normals),
        ("tx_positions", tx_positions),
        ("rx_positions", rx_positions),
    ):
        _require_xyz(name, value)
    if exponent <= 0:
        raise ValueError("exponent must be positive")
    normal = safe_normalize(normals)
    omega_i = safe_normalize(points - tx_positions)
    omega_r = safe_normalize(rx_positions - points)
    omega_o = omega_i - 2.0 * (normal * omega_i).sum(dim=-1, keepdim=True) * normal
    response = (omega_o * omega_r).sum(dim=-1).clamp(0.0, 1.0)
    return response if exponent == 1.0 else response.pow(float(exponent))


def free_space_amplitude_decay(
    points: torch.Tensor,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
    min_distance: float = 1.0e-6,
) -> torch.Tensor:
    """Bistatic amplitude decay; reduces to GeRaF Eq. 2 when Tx=Rx."""
    if min_distance <= 0:
        raise ValueError("min_distance must be positive")
    for name, value in (
        ("points", points),
        ("tx_positions", tx_positions),
        ("rx_positions", rx_positions),
    ):
        _require_xyz(name, value)
    r_tx = torch.linalg.vector_norm(points - tx_positions, dim=-1).clamp_min(min_distance)
    r_rx = torch.linalg.vector_norm(points - rx_positions, dim=-1).clamp_min(min_distance)
    return 1.0 / (((4.0 * math.pi) ** 2) * r_tx * r_rx)


def free_space_power_decay(
    points: torch.Tensor,
    tx_positions: torch.Tensor,
    rx_positions: torch.Tensor,
    *,
    min_distance: float = 1.0e-6,
) -> torch.Tensor:
    """Squared free-space amplitude decay (``1/(4*pi*u)^4`` monostatic)."""
    amplitude = free_space_amplitude_decay(
        points, tx_positions, rx_positions, min_distance=min_distance
    )
    return amplitude.square()


def aperture_basis(
    primary_direction: torch.Tensor, up_hint: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return right, up, and unit primary-direction vectors for an aperture."""
    _require_xyz("primary_direction", primary_direction)
    if primary_direction.shape != (3,):
        raise ValueError("primary_direction must have shape [3]")
    forward = safe_normalize(primary_direction)
    if up_hint is None:
        axis = int(torch.argmin(forward.detach().abs()).item())
        hint = torch.zeros_like(forward)
        hint[axis] = 1.0
    else:
        _require_xyz("up_hint", up_hint)
        if up_hint.shape != (3,):
            raise ValueError("up_hint must have shape [3]")
        hint = safe_normalize(up_hint.to(forward))
        if float((hint * forward).sum().detach().abs()) > 0.999:
            raise ValueError("up_hint must not be parallel to primary_direction")
    right = safe_normalize(torch.linalg.cross(forward, hint))
    up = safe_normalize(torch.linalg.cross(right, forward))
    return right, up, forward


@dataclass(frozen=True)
class PrimaryRaySamples:
    """Shared GeRaF lensless samples for one radar view."""

    ray_origins: torch.Tensor  # [R,3]
    primary_direction: torch.Tensor  # [3]
    depths: torch.Tensor  # [R,Z]
    depth_deltas: torch.Tensor  # [R,Z]
    points: torch.Tensor  # [R,Z,3]
    depth_edges: Optional[torch.Tensor] = None  # [R,Z+1], required by hardened_v1

    @property
    def num_rays(self) -> int:
        return int(self.points.shape[0])

    @property
    def num_depth(self) -> int:
        return int(self.points.shape[1])


def sample_primary_rays(
    aperture_center: torch.Tensor,
    primary_direction: torch.Tensor,
    *,
    aperture_width: float,
    aperture_height: float,
    near: float,
    far: float,
    num_rays: int = 1024,
    num_depth: int = 32,
    aperture_sampling: str = "grid",
    stratified_depth: bool = True,
    up_hint: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> PrimaryRaySamples:
    """Sample the paper's parallel primary rays and their depth points.

    ``grid`` uses a square aperture lattice (the paper default 1024 = 32^2).
    ``random`` draws uniform aperture locations.  Depth samples are stratified
    by default, one in each of ``num_depth`` equal intervals.
    """
    _require_xyz("aperture_center", aperture_center)
    if aperture_center.shape != (3,):
        raise ValueError("aperture_center must have shape [3]")
    if aperture_width <= 0 or aperture_height <= 0:
        raise ValueError("aperture dimensions must be positive")
    if not (0 <= near < far):
        raise ValueError("need 0 <= near < far")
    if num_rays <= 0 or num_depth <= 0:
        raise ValueError("num_rays and num_depth must be positive")
    if aperture_sampling not in {"grid", "random"}:
        raise ValueError("aperture_sampling must be grid or random")
    right, up, forward = aperture_basis(primary_direction.to(aperture_center), up_hint)

    if aperture_sampling == "grid":
        side = math.isqrt(int(num_rays))
        if side * side != num_rays:
            raise ValueError("grid aperture sampling requires num_rays to be a perfect square")
        axis = (torch.arange(side, device=aperture_center.device, dtype=aperture_center.dtype) + 0.5)
        axis = axis / side - 0.5
        uu, vv = torch.meshgrid(axis, axis, indexing="xy")
        uv = torch.stack((uu.reshape(-1), vv.reshape(-1)), dim=-1)
    else:
        uv = torch.rand(
            num_rays,
            2,
            device=aperture_center.device,
            dtype=aperture_center.dtype,
            generator=generator,
        ) - 0.5
    ray_origins = (
        aperture_center.view(1, 3)
        + (uv[:, :1] * aperture_width) * right.view(1, 3)
        + (uv[:, 1:] * aperture_height) * up.view(1, 3)
    )

    edges = torch.linspace(
        near,
        far,
        num_depth + 1,
        device=aperture_center.device,
        dtype=aperture_center.dtype,
    )
    depth_deltas = (edges[1:] - edges[:-1]).view(1, -1).expand(num_rays, -1)
    if stratified_depth:
        unit = torch.rand(
            num_rays,
            num_depth,
            device=aperture_center.device,
            dtype=aperture_center.dtype,
            generator=generator,
        )
        depths = edges[:-1].view(1, -1) + unit * depth_deltas
    else:
        depths = ((edges[:-1] + edges[1:]) * 0.5).view(1, -1).expand(num_rays, -1)
    points = ray_origins[:, None, :] + depths[..., None] * forward.view(1, 1, 3)
    return PrimaryRaySamples(ray_origins, forward, depths, depth_deltas, points,
                             edges.view(1, -1).expand(num_rays, -1))


@dataclass
class GeRaFVolumeOutput:
    """Differentiable physical terms before signal tracing and matched filtering."""

    points: torch.Tensor  # [R,Z,3]
    sdf: torch.Tensor  # [R,Z]
    sdf_gradient: torch.Tensor  # [R,Z,3]
    normals: torch.Tensor  # [R,Z,3]
    reflectivity: torch.Tensor  # [R,Z]
    alpha: torch.Tensor  # [R,Z]
    primary_transmittance: torch.Tensor  # [R,Z]
    two_way_transmittance: torch.Tensor  # [B,R,Z]
    directional_response: torch.Tensor  # [B,R,Z]
    path_decay: torch.Tensor  # [B,R,Z]
    amplitudes: torch.Tensor  # [B,R,Z]
    tx_positions: torch.Tensor  # [B,3]
    rx_positions: torch.Tensor  # [B,3]
    diagnostics: Optional[Dict[str, Any]] = None


@dataclass
class GeRaFMagnitudeOutput:
    volume: GeRaFVolumeOutput
    complex_response: torch.Tensor
    matched_filter_magnitude: torch.Tensor


class GeRaFModel(nn.Module):
    """GeRaF v1 scene model and pre-signal-tracing physical renderer."""

    def __init__(
        self,
        extent: float,
        *,
        sdf_levels: int = 10,
        sdf_hidden_dim: int = 256,
        sdf_layers: int = 8,
        sdf_skip_layer: Optional[int] = 4,
        sdf_hidden_activation: str = "softplus",
        sdf_softplus_beta: float = 100.0,
        sdf_encoding_include_input: bool = False,
        sdf_encoding_coordinate_scale: float = 1.0,
        reflectivity_levels: int = 0,
        reflectivity_hidden_dim: int = 256,
        reflectivity_layers: int = 4,
        reflectivity_output_activation: str = "softplus",
        reflectivity_softplus_beta: float = 1.0,
        reflectivity_encoding_coordinate_scale: float = 1.0,
        init_tx_amplitude: float = 1.0,
        init_inv_s: float = 64.0,
        learnable_inv_s: bool = True,
        implementation: str = "legacy",
        sdf_initialization: str = "legacy",
        sdf_output_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if implementation not in {"legacy", "hardened_v1"}:
            raise ValueError("unknown GeRaF implementation")
        self.implementation = implementation
        self.extent = float(extent)
        if sdf_initialization == "upstream_geometric":
            from rift.geraf_v1 import SourceSDFNetwork
            if (implementation != "hardened_v1" or not sdf_encoding_include_input
                    or sdf_encoding_coordinate_scale != 1 / extent or sdf_output_scale != extent
                    or sdf_hidden_activation != "softplus" or sdf_softplus_beta != 100.):
                raise ValueError("upstream SDF requires its declared normalized metric configuration")
            self.sdf_network = SourceSDFNetwork(extent, n_levels=sdf_levels,
                                                hidden_dim=sdf_hidden_dim, n_layers=sdf_layers,
                                                skip_layer=sdf_skip_layer)
        else:
            if implementation != "legacy" or sdf_initialization != "legacy" or sdf_output_scale != 1.0:
                raise ValueError("hardened_v1 requires the upstream_geometric SDF configuration")
            self.sdf_network = GeRaFSDFNetwork(
                extent,
                n_levels=sdf_levels,
                hidden_dim=sdf_hidden_dim,
                n_layers=sdf_layers,
                skip_layer=sdf_skip_layer,
                hidden_activation=sdf_hidden_activation,
                softplus_beta=sdf_softplus_beta,
                encoding_include_input=sdf_encoding_include_input,
                encoding_coordinate_scale=sdf_encoding_coordinate_scale,
            )
        self.reflectivity_network = GeRaFReflectivityNetwork(
            extent,
            n_levels=reflectivity_levels,
            hidden_dim=reflectivity_hidden_dim,
            n_layers=reflectivity_layers,
            output_activation=reflectivity_output_activation,
            softplus_beta=reflectivity_softplus_beta,
            encoding_coordinate_scale=reflectivity_encoding_coordinate_scale,
        )
        self.tx_amplitude = LearnableTxAmplitude(init_tx_amplitude)
        self.sdf_sharpness = LearnableSDFSharpness(init_inv_s, learnable_inv_s)

    def config(self) -> Dict[str, object]:
        config = {
            "extent": self.extent,
            "sdf_levels": self.sdf_network.n_levels,
            "sdf_hidden_dim": self.sdf_network.hidden_dim,
            "sdf_layers": self.sdf_network.n_layers,
            "sdf_skip_layer": self.sdf_network.skip_layer,
            "sdf_hidden_activation": self.sdf_network.hidden_activation,
            "sdf_softplus_beta": self.sdf_network.softplus_beta,
            "sdf_encoding_include_input": self.sdf_network.encoding_include_input,
            "sdf_encoding_coordinate_scale": self.sdf_network.encoding_coordinate_scale,
            "reflectivity_levels": self.reflectivity_network.n_levels,
            "reflectivity_hidden_dim": self.reflectivity_network.hidden_dim,
            "reflectivity_layers": self.reflectivity_network.n_layers,
            "reflectivity_output_activation": self.reflectivity_network.output_activation,
            "reflectivity_softplus_beta": self.reflectivity_network.softplus_beta,
            "reflectivity_encoding_coordinate_scale": (
                self.reflectivity_network.encoding_coordinate_scale
            ),
            "init_tx_amplitude": self.tx_amplitude.value,
            "init_inv_s": float(self.sdf_sharpness().detach().item()),
            "learnable_inv_s": bool(self.sdf_sharpness.log_inv_s.requires_grad),
        }
        if self.implementation != "legacy":
            config.update(implementation=self.implementation,
                          sdf_initialization=self.sdf_network.initialization,
                          sdf_output_scale=self.sdf_network.output_scale)
        return config

    @staticmethod
    def _override_field(
        override: Optional[torch.Tensor | float], reference: torch.Tensor, name: str
    ) -> torch.Tensor:
        if override is None:
            return reference
        value = override if torch.is_tensor(override) else reference.new_tensor(float(override))
        try:
            return torch.broadcast_to(value.to(reference), reference.shape)
        except RuntimeError as exc:
            raise ValueError(f"{name} is not broadcastable to {tuple(reference.shape)}") from exc

    def render_volume(
        self,
        samples: PrimaryRaySamples,
        tx_positions: torch.Tensor,
        rx_positions: torch.Tensor,
        *,
        lensless_correction: bool = True,
        detach_start_cdf: bool = True,
        directional_exponent: float = 1.0,
        create_graph: bool = True,
        reflectivity_override: Optional[torch.Tensor | float] = None,
        alpha_override: Optional[torch.Tensor | float] = None,
        tx_amplitude_override: Optional[torch.Tensor | float] = None,
        min_distance: float = 1.0e-6,
    ) -> GeRaFVolumeOutput:
        """Evaluate GeRaF's physical amplitude weights for bistatic pairs.

        ``tx_positions`` and ``rx_positions`` are pairwise ``[B,3]`` arrays.
        The result is still before the phase/time signal-tracing operator;
        amplitudes have shape ``[B,R,Z]``.
        """
        _require_xyz("tx_positions", tx_positions)
        _require_xyz("rx_positions", rx_positions)
        if tx_positions.ndim != 2 or rx_positions.ndim != 2:
            raise ValueError("tx_positions and rx_positions must have shape [B,3]")
        tx_positions = tx_positions.to(samples.points)
        rx_positions = rx_positions.to(samples.points)
        try:
            tx_positions, rx_positions = torch.broadcast_tensors(tx_positions, rx_positions)
        except RuntimeError as exc:
            raise ValueError("tx_positions and rx_positions must be pairwise broadcastable") from exc
        if self.implementation == "hardened_v1":
            from rift.geraf_v1 import render_volume
            return render_volume(
                self, samples, tx_positions, rx_positions,
                lensless_correction=lensless_correction, detach_start_cdf=detach_start_cdf,
                directional_exponent=directional_exponent, create_graph=create_graph,
                reflectivity_override=reflectivity_override, alpha_override=alpha_override,
                tx_amplitude_override=tx_amplitude_override, min_distance=min_distance,
            )
        flat = samples.points.reshape(-1, 3).detach().requires_grad_(True)
        sdf_flat, gradient_flat = self.sdf_network.gradient(flat, create_graph=create_graph)
        reflectivity_flat = self.reflectivity_network(flat)
        r, z = samples.points.shape[:2]
        sdf = sdf_flat.reshape(r, z)
        sdf_gradient = gradient_flat.reshape(r, z, 3)
        normals = safe_normalize(sdf_gradient)
        reflectivity = reflectivity_flat.reshape(r, z)
        reflectivity = self._override_field(
            reflectivity_override, reflectivity, "reflectivity_override"
        )

        inv_s = self.sdf_sharpness()
        alpha = neus_sdf_to_alpha(sdf, inv_s)
        alpha = self._override_field(alpha_override, alpha, "alpha_override").clamp(0.0, 1.0)
        primary_t = transmittance_from_alpha(alpha)

        if lensless_correction:
            primary_start_sdf = self.sdf_network(samples.ray_origins.to(flat))
            tx_start_sdf = self.sdf_network(tx_positions.to(flat))
            rx_start_sdf = self.sdf_network(rx_positions.to(flat))
            t_tx = lensless_transmittance_correction(
                primary_t,
                primary_start_sdf,
                tx_start_sdf,
                inv_s,
                detach_start_cdf=detach_start_cdf,
            )
            t_rx = lensless_transmittance_correction(
                primary_t,
                primary_start_sdf,
                rx_start_sdf,
                inv_s,
                detach_start_cdf=detach_start_cdf,
            )
            two_way_t = t_tx * t_rx
        else:
            two_way_t = primary_t.square().unsqueeze(0).expand(tx_positions.shape[0], -1, -1)

        points_b = flat.reshape(r, z, 3).unsqueeze(0)
        normals_b = normals.unsqueeze(0)
        tx_b = tx_positions[:, None, None, :]
        rx_b = rx_positions[:, None, None, :]
        directional = shifted_lambertian_bistatic(
            points_b,
            normals_b,
            tx_b,
            rx_b,
            exponent=directional_exponent,
        )
        decay = free_space_amplitude_decay(
            points_b, tx_b, rx_b, min_distance=min_distance
        )
        tx_amplitude = self.tx_amplitude()
        if tx_amplitude_override is not None:
            tx_amplitude = self._override_field(
                tx_amplitude_override, tx_amplitude, "tx_amplitude_override"
            )
        amplitudes = (
            tx_amplitude.to(reflectivity)
            * reflectivity.unsqueeze(0)
            * directional
            * decay
            * two_way_t
            * alpha.unsqueeze(0)
        )
        return GeRaFVolumeOutput(
            points=flat.reshape(r, z, 3),
            sdf=sdf,
            sdf_gradient=sdf_gradient,
            normals=normals,
            reflectivity=reflectivity,
            alpha=alpha,
            primary_transmittance=primary_t,
            two_way_transmittance=two_way_t,
            directional_response=directional,
            path_decay=decay,
            amplitudes=amplitudes,
            tx_positions=tx_positions,
            rx_positions=rx_positions,
        )

    def render_magnitude(
        self,
        samples: PrimaryRaySamples,
        tx_positions: torch.Tensor,
        rx_positions: torch.Tensor,
        *,
        signal_trace_callback: Callable[..., torch.Tensor],
        matched_filter_callback: Callable[..., torch.Tensor],
        signal_trace_kwargs: Optional[Mapping[str, Any]] = None,
        matched_filter_kwargs: Optional[Mapping[str, Any]] = None,
        **volume_kwargs: Any,
    ) -> GeRaFMagnitudeOutput:
        """Render matched-filter magnitude through acquisition-specific callbacks.

        The signal tracer is called as ``callback(volume, **kwargs)`` and must
        return a complex response.  The matched filter is called as
        ``callback(complex_response, **kwargs)``.  No geometry, phase, channel
        order, or FFT convention is guessed inside this module.
        """
        volume = self.render_volume(
            samples, tx_positions, rx_positions, **volume_kwargs
        )
        response = signal_trace_callback(volume, **dict(signal_trace_kwargs or {}))
        if not torch.is_tensor(response) or not torch.is_complex(response):
            raise TypeError("signal_trace_callback must return a complex torch.Tensor")
        magnitude = apply_matched_filter_magnitude(
            response,
            matched_filter_callback,
            **dict(matched_filter_kwargs or {}),
        )
        return GeRaFMagnitudeOutput(volume, response, magnitude)


def apply_matched_filter_magnitude(
    predicted_complex_response: torch.Tensor,
    matched_filter_callback: Callable[..., torch.Tensor],
    **kwargs: Any,
) -> torch.Tensor:
    """Apply a shared differentiable MF-magnitude callback to a response."""
    if not torch.is_tensor(predicted_complex_response) or not torch.is_complex(
        predicted_complex_response
    ):
        raise TypeError("predicted_complex_response must be a complex torch.Tensor")
    magnitude = matched_filter_callback(predicted_complex_response, **kwargs)
    if not torch.is_tensor(magnitude) or torch.is_complex(magnitude):
        raise TypeError("matched_filter_callback must return a real torch.Tensor")
    if bool((magnitude.detach() < 0).any().item()):
        raise ValueError("matched_filter_callback must return a non-negative magnitude")
    return magnitude


class DynamicLossMask(nn.Module):
    """Stateful implementation of GeRaF's historical-response loss mask.

    A location is omitted when it was at least ``high_threshold`` in an
    earlier iteration but its current response is below the larger of an
    absolute ``low_threshold`` and ``low_ratio`` of its historical maximum.  The
    paper does not publish these thresholds, so they are required experiment
    metadata instead of hard-coded constants.
    """

    def __init__(
        self,
        high_threshold: float,
        *,
        low_ratio: float = 0.1,
        low_threshold: float = 0.0,
        shape: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        if high_threshold < 0 or low_threshold < 0:
            raise ValueError("mask thresholds must be non-negative")
        if not (0.0 <= low_ratio <= 1.0):
            raise ValueError("low_ratio must lie in [0,1]")
        self.high_threshold = float(high_threshold)
        self.low_ratio = float(low_ratio)
        self.low_threshold = float(low_threshold)
        initial = torch.zeros(tuple(shape), dtype=torch.float32) if shape is not None else torch.empty(0)
        self.register_buffer("historical_max", initial)

    def reset(self, shape: Optional[Sequence[int]] = None) -> None:
        """Clear response history, optionally allocating a new map shape."""
        if shape is None:
            self.historical_max = self.historical_max.new_empty(0)
        else:
            self.historical_max = self.historical_max.new_zeros(tuple(shape))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:
        key = prefix + "historical_max"
        if key in state_dict and self.historical_max.shape != state_dict[key].shape:
            self.historical_max = torch.empty_like(state_dict[key])
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, current_power: torch.Tensor, *, update: bool = True) -> torch.Tensor:
        if not torch.is_tensor(current_power) or torch.is_complex(current_power):
            raise TypeError("current_power must be a real torch.Tensor")
        if bool((current_power.detach() < 0).any().item()):
            raise ValueError("current_power must be non-negative")
        if self.historical_max.numel() == 0:
            self.historical_max = torch.zeros_like(current_power.detach())
        if self.historical_max.shape != current_power.shape:
            raise ValueError(
                f"dynamic-mask shape changed from {tuple(self.historical_max.shape)} "
                f"to {tuple(current_power.shape)}; call reset"
            )
        history = self.historical_max.to(current_power)
        was_high = history >= self.high_threshold
        low_limit = torch.maximum(
            history * self.low_ratio, history.new_full((), self.low_threshold)
        )
        valid = ~(was_high & (current_power <= low_limit))
        if not bool(valid.any().item()):
            raise RuntimeError(
                "dynamic loss mask rejected every location; history was not committed"
            )
        if update:
            with torch.no_grad():
                self.historical_max.copy_(
                    torch.maximum(self.historical_max, current_power.detach().to(self.historical_max))
                )
        return valid


def masked_magnitude_l2(
    predicted_magnitude: torch.Tensor,
    target_magnitude: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean squared native ``|MF|`` error over valid locations only."""
    if (
        predicted_magnitude.shape != target_magnitude.shape
        or valid_mask.shape != predicted_magnitude.shape
    ):
        raise ValueError(
            "predicted_magnitude, target_magnitude, and valid_mask must have identical shapes"
        )
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be boolean")
    if not bool(valid_mask.any().item()):
        raise RuntimeError(
            "dynamic loss mask rejected every location; refuse to advance optimizer/scheduler"
        )
    squared = (predicted_magnitude - target_magnitude).square()
    weights = valid_mask.to(squared.dtype)
    return (squared * weights).sum() / weights.sum().clamp_min(1.0)


__all__ = [
    "PAPER_ID",
    "METHOD_NAME",
    "SinusoidalPositionEncoding",
    "GeRaFSDFNetwork",
    "GeRaFReflectivityNetwork",
    "LearnableTxAmplitude",
    "LearnableSDFSharpness",
    "GeRaFModel",
    "PrimaryRaySamples",
    "GeRaFVolumeOutput",
    "GeRaFMagnitudeOutput",
    "DynamicLossMask",
    "safe_normalize",
    "logistic_sdf_cdf",
    "logistic_sdf_pdf",
    "neus_opaque_density",
    "neus_sdf_to_alpha",
    "transmittance_from_alpha",
    "lensless_transmittance_correction",
    "shifted_lambertian_bistatic",
    "free_space_amplitude_decay",
    "free_space_power_decay",
    "aperture_basis",
    "sample_primary_rays",
    "apply_matched_filter_magnitude",
    "masked_magnitude_l2",
]
