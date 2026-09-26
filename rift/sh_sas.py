"""SH-SAS neural scattering field adapted to RIFT's radar operator.

This module is an independent implementation of Vengurlekar, Pediredla and
Jayasuriya, *SH-SAS* (3DV 2026 / arXiv:2509.11087).  The authors have not
released source code.  The paper contract implemented here is:

* a 16-level multi-resolution hash encoding ending at resolution 4096;
* a two-hidden-layer, width-32 ReLU MLP;
* ``2 * (L + 1)**2`` outputs for complex real-SH coefficients, with ``L=3``;
* density ``rho = zeta * abs(c00 / sqrt(4*pi))``;
* normals from the negative normalized spatial gradient of that DC amplitude;
* a transmitter-facing Lambertian cosine;
* accumulated Tx and Rx transmittance multiplied into the complex scatterer.

RIFT's measured data are swept-frequency radar responses rather than
pulse-deconvolved sonar transients.  ``train_sh_sas.py`` therefore feeds the
view weights produced here to RIFT's exact bistatic frequency operator.  The
field is continuous; the integral is evaluated on a fixed voxel-centre
quadrature lattice so the result can also be scored by the common B787
geometry evaluator.

The optional mean normalization of opacity is a documented radar-conditioning
adaptation.  It makes zeta the optical depth of one average-density voxel and
removes RIFT's exact (global gain, scene scale) gauge.  Passing
``opacity_normalize=False`` recovers the paper's literal Eq. (4).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from rift.encoding import generate_dynamic_grid
from rift.occlusion import ray_transmittance
from rift.radar_fields import HashGridEncoder
from rift.spherical_harmonics import num_sh_basis, real_sh_basis


PAPER_SH_DEGREE = 3
PAPER_HASH_LEVELS = 16
PAPER_HASH_BASE_RESOLUTION = 16
PAPER_HASH_FINAL_RESOLUTION = 4096
PAPER_MLP_WIDTH = 32
PAPER_MLP_HIDDEN_LAYERS = 2
Y00 = 1.0 / math.sqrt(4.0 * math.pi)


def _normalise(v: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    return torch.where(norm > eps, v / norm.clamp_min(eps), torch.zeros_like(v))


def _axis_gradient(volume: torch.Tensor, axis: int, spacing: float) -> torch.Tensor:
    """First-order edges and centred interior differences, without in-place ops."""
    if volume.shape[axis] < 2:
        return torch.zeros_like(volume)
    front = [slice(None)] * volume.ndim
    second = [slice(None)] * volume.ndim
    back = [slice(None)] * volume.ndim
    before_back = [slice(None)] * volume.ndim
    interior_hi = [slice(None)] * volume.ndim
    interior_lo = [slice(None)] * volume.ndim
    front[axis] = slice(0, 1)
    second[axis] = slice(1, 2)
    back[axis] = slice(-1, None)
    before_back[axis] = slice(-2, -1)
    interior_hi[axis] = slice(2, None)
    interior_lo[axis] = slice(None, -2)
    first = (volume[tuple(second)] - volume[tuple(front)]) / spacing
    middle = (volume[tuple(interior_hi)] - volume[tuple(interior_lo)]) / (2.0 * spacing)
    last = (volume[tuple(back)] - volume[tuple(before_back)]) / spacing
    return torch.cat((first, middle, last), dim=axis)


def spatial_gradient(volume: torch.Tensor, spacing: float) -> torch.Tensor:
    """Return ``[...,3]`` gradient for a scalar ``[G,G,G]`` lattice."""
    if volume.ndim != 3:
        raise ValueError(f"volume must have shape [G,G,G], got {tuple(volume.shape)}")
    return torch.stack(
        tuple(_axis_gradient(volume, axis, spacing) for axis in range(3)), dim=-1
    )


def real_sh_basis_for_directions(directions: torch.Tensor, degree: int) -> torch.Tensor:
    """Real SH basis for ``[N,3]`` directions, returned as ``[N,(L+1)^2]``."""
    if directions.ndim != 2 or directions.shape[-1] != 3:
        raise ValueError("directions must have shape [N,3]")
    unit = _normalise(directions)
    theta = torch.acos(unit[:, 2].clamp(-1.0, 1.0))
    phi = torch.atan2(unit[:, 1], unit[:, 0])
    # real_sh_basis is written from tensor operations and supports a vector
    # theta/phi even though RIFT's original caller used one scalar direction.
    return real_sh_basis(theta, phi, degree).transpose(0, 1).contiguous()


def lambertian_cosine(
    points: torch.Tensor, normals: torch.Tensor, transmitter_origin: torch.Tensor
) -> torch.Tensor:
    """Paper Eq. (6): ``max(0, n(x) dot (o_T-x)/||o_T-x||)``."""
    to_tx = _normalise(transmitter_origin.reshape(1, 3) - points)
    return (normals * to_tx).sum(dim=-1).clamp_min(0.0)


def _tv_scalar(volume: torch.Tensor) -> torch.Tensor:
    terms = []
    for axis in range(3):
        if volume.shape[axis] < 2:
            continue
        hi = [slice(None)] * volume.ndim
        lo = [slice(None)] * volume.ndim
        hi[axis] = slice(1, None)
        lo[axis] = slice(None, -1)
        terms.append((volume[tuple(hi)] - volume[tuple(lo)]).abs().mean())
    return sum(terms, volume.new_zeros(()))


def _phase_tv(field: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    terms = []
    for axis in range(3):
        if field.shape[axis] < 2:
            continue
        hi = [slice(None)] * field.ndim
        lo = [slice(None)] * field.ndim
        hi[axis] = slice(1, None)
        lo[axis] = slice(None, -1)
        a, b = field[tuple(hi)], field[tuple(lo)]
        cross = a * b.conj()
        valid = ((a.abs() > eps) & (b.abs() > eps)).detach()
        wrapped = torch.atan2(cross.imag, cross.real + eps).abs()
        terms.append((wrapped * valid).sum() / valid.sum().clamp_min(1))
    return sum(terms, field.real.new_zeros(()))


class SHSASField(nn.Module):
    """Continuous complex SH field plus paper-faithful view rendering terms."""

    def __init__(
        self,
        extent: float,
        granularity: int,
        sh_degree: int = PAPER_SH_DEGREE,
        hidden_dim: int = PAPER_MLP_WIDTH,
        hash_levels: int = PAPER_HASH_LEVELS,
        hash_features: int = 2,
        hash_base_resolution: int = PAPER_HASH_BASE_RESOLUTION,
        hash_final_resolution: int = PAPER_HASH_FINAL_RESOLUTION,
        hash_log2_size: int = 19,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if extent <= 0 or granularity < 2:
            raise ValueError("extent must be positive and granularity must be >= 2")
        if sh_degree < 0:
            raise ValueError("sh_degree must be non-negative")
        self.extent = float(extent)
        self.granularity = int(granularity)
        self.sh_degree = int(sh_degree)
        self.n_basis = num_sh_basis(self.sh_degree)

        self.encoder = HashGridEncoder(
            n_levels=hash_levels,
            n_features_per_level=hash_features,
            base_resolution=hash_base_resolution,
            final_resolution=hash_final_resolution,
            log2_hashmap_size=hash_log2_size,
        )
        layers = []
        in_dim = self.encoder.output_dim
        for _ in range(PAPER_MLP_HIDDEN_LAYERS):
            layer = nn.Linear(in_dim, hidden_dim)
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
            nn.init.zeros_(layer.bias)
            layers.extend((layer, nn.ReLU(inplace=False)))
            in_dim = hidden_dim
        output = nn.Linear(in_dim, 2 * self.n_basis)
        nn.init.xavier_uniform_(output.weight, gain=0.1)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.mlp = nn.Sequential(*layers)

        grid = generate_dynamic_grid(
            self.granularity, self.extent, device or torch.device("cpu"), jitter=False
        ).reshape(-1, 3)
        self.register_buffer("grid_positions", grid)

    @property
    def paper_architecture(self) -> Dict[str, int]:
        return {
            "sh_degree": self.sh_degree,
            "hash_levels": self.encoder.n_levels,
            "hash_base_resolution": self.encoder.base_resolution,
            "hash_final_resolution": self.encoder.final_resolution,
            "mlp_width": self.mlp[0].out_features,
            "mlp_hidden_layers": PAPER_MLP_HIDDEN_LAYERS,
            "output_channels": 2 * self.n_basis,
        }

    def query_coefficients(self, points: torch.Tensor, chunk_size: int = 32768) -> torch.Tensor:
        """Complex coefficients at arbitrary metric-space points, shape ``[N,C]``."""
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError("points must have shape [N,3]")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        outputs = []
        for start in range(0, points.shape[0], chunk_size):
            xyz = points[start : start + chunk_size]
            unit = ((xyz / self.extent) + 1.0) * 0.5
            encoded = self.encoder(unit)
            raw = self.mlp(encoded)
            re, im = raw.split(self.n_basis, dim=-1)
            outputs.append(torch.complex(re, im))
        return torch.cat(outputs, dim=0)

    def dense_coefficients(self, chunk_size: int = 32768) -> torch.Tensor:
        coeff = self.query_coefficients(self.grid_positions, chunk_size=chunk_size)
        return coeff.reshape(
            self.granularity, self.granularity, self.granularity, self.n_basis
        )

    @staticmethod
    def dc_amplitude(coefficients: torch.Tensor) -> torch.Tensor:
        """Paper Eqs. (3-4) before zeta: ``abs(c00 / sqrt(4*pi))``."""
        return coefficients[..., 0].abs() * Y00

    def normals_from_dc(self, coefficients: torch.Tensor) -> torch.Tensor:
        density = self.dc_amplitude(coefficients)
        pitch = 2.0 * self.extent / self.granularity
        return _normalise(-spatial_gradient(density, pitch))

    def opacity_proxy(self, coefficients: torch.Tensor, key: str = "dc") -> torch.Tensor:
        if key == "dc":
            return self.dc_amplitude(coefficients)
        if key == "energy":
            return coefficients.abs().square().sum(dim=-1).clamp_min(0.0).sqrt()
        raise ValueError("opacity key must be 'dc' or 'energy'")

    def view_field(
        self,
        tx_pos: torch.Tensor,
        rx_pos: torch.Tensor,
        *,
        opacity_scale: float = 1.0,
        opacity_key: str = "dc",
        opacity_normalize: bool = True,
        use_lambertian: bool = True,
        use_occlusion: bool = True,
        occlusion_steps: Optional[int] = None,
        occlusion_step_frac: float = 0.5,
        query_chunk: int = 32768,
        occlusion_point_chunk: int = 16384,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate SH-SAS view weights on the fixed quadrature lattice.

        The B787 array aperture is only 2.85 cm at 10 m standoff.  We use the
        Tx and Rx phase centres for SH direction/Lambertian/transmittance;
        exact element locations remain in RIFT's bistatic propagation phase.
        This is the same sub-voxel small-aperture approximation validated by
        ``rift/occlusion.py`` and keeps visibility pair-independent.
        """
        coefficients = self.dense_coefficients(chunk_size=query_chunk)
        flat_coeff = coefficients.reshape(-1, self.n_basis)
        points = self.grid_positions
        tx_origin = tx_pos.mean(dim=0)
        rx_origin = rx_pos.mean(dim=0)

        outgoing = points - rx_origin.reshape(1, 3)
        basis = real_sh_basis_for_directions(outgoing, self.sh_degree)
        scattering = (flat_coeff * basis.to(flat_coeff.dtype)).sum(dim=-1)

        normals = self.normals_from_dc(coefficients).reshape(-1, 3)
        if use_lambertian:
            lambertian = lambertian_cosine(points, normals, tx_origin)
        else:
            lambertian = torch.ones(points.shape[0], dtype=points.dtype, device=points.device)

        proxy = self.opacity_proxy(coefficients, key=opacity_key).reshape(-1)
        if use_occlusion and opacity_scale > 0:
            pitch = 2.0 * self.extent / self.granularity
            if opacity_normalize:
                mean_proxy = proxy.mean().clamp_min(1.0e-30)
                sigma = (float(opacity_scale) / pitch) * (proxy / mean_proxy)
            else:
                # Literal paper Eq. (4): zeta carries inverse-length units.
                sigma = float(opacity_scale) * proxy
            t_tx = ray_transmittance(
                points,
                sigma,
                tx_origin,
                self.extent,
                self.granularity,
                n_steps=occlusion_steps,
                step_frac=occlusion_step_frac,
                point_chunk=occlusion_point_chunk,
                two_way=False,
            )
            t_rx = ray_transmittance(
                points,
                sigma,
                rx_origin,
                self.extent,
                self.granularity,
                n_steps=occlusion_steps,
                step_frac=occlusion_step_frac,
                point_chunk=occlusion_point_chunk,
                two_way=False,
            )
            transmittance = t_tx * t_rx
        else:
            sigma = torch.zeros_like(proxy)
            transmittance = torch.ones_like(proxy)

        weights = scattering * lambertian.to(scattering.dtype) * transmittance.to(scattering.dtype)
        return {
            "points": points,
            "coefficients": coefficients,
            "density": self.dc_amplitude(coefficients),
            "opacity": sigma.reshape(self.granularity, self.granularity, self.granularity),
            "normals": normals.reshape(self.granularity, self.granularity, self.granularity, 3),
            "scattering": scattering.reshape(self.granularity, self.granularity, self.granularity),
            "lambertian": lambertian.reshape(self.granularity, self.granularity, self.granularity),
            "transmittance": transmittance.reshape(
                self.granularity, self.granularity, self.granularity
            ),
            "weights": weights,
        }

    @staticmethod
    def regularizers(view: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """The four non-data terms in paper Eq. (8), each unweighted."""
        density = view["density"]
        scattering = view["scattering"]
        return {
            "sparse": density.abs().mean(),
            "density_tv": _tv_scalar(density),
            "scatter_tv": _tv_scalar(scattering.abs()),
            "phase_tv": _phase_tv(scattering),
        }

    @torch.no_grad()
    def geometry_compatibility_state(self, chunk_size: int = 32768) -> Dict[str, torch.Tensor]:
        """Dense coefficient tensors consumed by the common B787 evaluator."""
        was_training = self.training
        self.eval()
        coeff = self.dense_coefficients(chunk_size=chunk_size).cpu()
        if was_training:
            self.train()
        return {"w_re": coeff.real.contiguous(), "w_im": coeff.imag.contiguous()}
