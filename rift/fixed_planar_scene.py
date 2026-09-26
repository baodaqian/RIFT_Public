"""Fixed, non-adaptive planar SH scene used by PublicRadar v2 pilots."""

from __future__ import annotations

import math

import torch
from torch import nn

from rift.spherical_harmonics import real_sh_basis


class FixedPlanarSHScene(nn.Module):
    """A dense ``nx × ny × 1`` z=0 grid with fixed spherical-harmonic order.

    Coordinates are generated per chunk from flat indices, so neither a full
    position tensor nor a full effective-weight tensor is materialized during
    rendering.  There is intentionally no prune, grow, split, learned offset,
    or spare capacity in this representation.
    """

    def __init__(self, nx, ny, extent, degree, device, init_scale=0.0):
        super().__init__()
        nx = int(nx)
        ny = int(ny)
        degree = int(degree)
        extent = float(extent)
        if nx < 2 or ny < 2:
            raise ValueError("planar dimensions must both be at least 2")
        if degree < 0 or extent <= 0 or not math.isfinite(extent):
            raise ValueError("degree and extent must be finite and nonnegative/positive")
        self.nx = nx
        self.ny = ny
        self.extent = extent
        self.max_degree = degree
        self.n_basis = (degree + 1) ** 2
        self.n_points = nx * ny
        self.w_re = nn.Parameter(
            init_scale * torch.randn(self.n_points, self.n_basis, device=device)
        )
        self.w_im = nn.Parameter(
            init_scale * torch.randn(self.n_points, self.n_basis, device=device)
        )

    @property
    def shape(self):
        return (self.nx, self.ny, 1)

    @property
    def pitch_xy(self):
        return (2.0 * self.extent / self.nx, 2.0 * self.extent / self.ny)

    def position_chunk(self, start, stop):
        start = int(start)
        stop = int(stop)
        if not 0 <= start <= stop <= self.n_points:
            raise ValueError("planar position chunk is out of range")
        flat = torch.arange(start, stop, device=self.w_re.device, dtype=torch.int64)
        ix = torch.div(flat, self.ny, rounding_mode="floor")
        iy = torch.remainder(flat, self.ny)
        pitch_x, pitch_y = self.pitch_xy
        x = -self.extent + (ix.to(self.w_re.dtype) + 0.5) * pitch_x
        y = -self.extent + (iy.to(self.w_re.dtype) + 0.5) * pitch_y
        z = torch.zeros_like(x)
        return torch.stack((x, y, z), dim=-1)

    def weight_chunk(self, dtheta, dphi, start, stop):
        theta = dtheta.squeeze(0)[0]
        phi = dphi.squeeze(0)[0]
        basis = real_sh_basis(theta, phi, self.max_degree)
        real = torch.einsum("kb,b->k", self.w_re[start:stop], basis)
        imag = torch.einsum("kb,b->k", self.w_im[start:stop], basis)
        return torch.complex(real, imag)

    def view_basis(self, dtheta, dphi):
        """Evaluate one shared ``[basis, view]`` matrix for an aligned batch."""
        dtheta = torch.as_tensor(dtheta, device=self.w_re.device)
        dphi = torch.as_tensor(dphi, device=self.w_re.device)
        if dtheta.ndim != 1 or dphi.ndim != 1 or dtheta.shape != dphi.shape:
            raise ValueError("batched directions must be matching 1D tensors")
        if dtheta.numel() == 0:
            raise ValueError("batched directions must not be empty")
        basis = real_sh_basis(dtheta, dphi, self.max_degree)
        if basis.shape != (self.n_basis, dtheta.numel()):
            raise RuntimeError("unexpected batched spherical-harmonic basis shape")
        return basis.to(dtype=self.w_re.dtype)

    def view_weight_chunk(self, basis, start, stop):
        """Return effective weights for aligned views with shape ``[K, B]``."""
        basis = torch.as_tensor(
            basis, device=self.w_re.device, dtype=self.w_re.dtype
        )
        if basis.ndim != 2 or basis.shape[0] != self.n_basis or basis.shape[1] == 0:
            raise ValueError("view basis must have shape [n_basis, views]")
        start = int(start)
        stop = int(stop)
        if not 0 <= start <= stop <= self.n_points:
            raise ValueError("planar weight chunk is out of range")
        real = self.w_re[start:stop] @ basis
        imag = self.w_im[start:stop] @ basis
        return torch.complex(real, imag)

    def scatterer_chunks(self, dtheta, dphi, chunk_size):
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        for start in range(0, self.n_points, chunk_size):
            stop = min(start + chunk_size, self.n_points)
            yield (
                self.position_chunk(start, stop),
                self.weight_chunk(dtheta, dphi, start, stop),
            )

    def scatterer_view_chunks(self, dtheta, dphi, chunk_size):
        """Yield fixed positions and ``[chunk_points, views]`` weights."""
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        basis = self.view_basis(dtheta, dphi)
        yield from self.scatterer_basis_chunks(basis, chunk_size)

    def scatterer_basis_chunks(self, basis, chunk_size):
        """Yield aligned weights from a live or cached ``[basis, view]`` matrix."""
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        basis = torch.as_tensor(
            basis, device=self.w_re.device, dtype=self.w_re.dtype
        )
        if basis.ndim != 2 or basis.shape[0] != self.n_basis or basis.shape[1] == 0:
            raise ValueError("view basis must have shape [n_basis, views]")
        for start in range(0, self.n_points, chunk_size):
            stop = min(start + chunk_size, self.n_points)
            yield (
                self.position_chunk(start, stop),
                self.view_weight_chunk(basis, start, stop),
            )

    def position_chunks(self, chunk_size):
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        for start in range(0, self.n_points, chunk_size):
            stop = min(start + chunk_size, self.n_points)
            yield self.position_chunk(start, stop)


@torch.no_grad()
def balance_fixed_scene_and_gain(model, gain):
    squared = model.w_re.square() + model.w_im.square()
    rms = float(torch.sqrt(squared.sum(dim=-1).mean()).item())
    if not math.isfinite(rms) or rms <= 0:
        raise ValueError(f"cannot balance fixed planar scene with RMS {rms}")
    scale = 1.0 / rms
    model.w_re.mul_(scale)
    model.w_im.mul_(scale)
    return complex(gain / scale), rms, scale
