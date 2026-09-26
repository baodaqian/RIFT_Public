"""Continuous SH fields used by the shared sonar benchmark.

``RIFTSASGrid`` is the current RIFT/Plenoxel representation with complex SH
coefficients stored on a regular grid and queried by coefficient-wise
trilinear interpolation.  ``ComplexSHSonarField`` adapts either that grid or
the independent SH-SAS hash field to physical sonar coordinates.
"""

from __future__ import annotations

import math
import operator
from typing import Dict, Optional

import torch
import torch.nn as nn

from rift.sh_sas import real_sh_basis_for_directions
from rift.sparse_scene import AdaptivePointSHScene, SHVoxelGridScene
from rift.spherical_harmonics import basis_degree_index, num_sh_basis


Y00 = 1.0 / math.sqrt(4.0 * math.pi)


class RIFTSASRectangularGrid(nn.Module):
    """Fixed complex-SH field on an independently sized endpoint lattice.

    The storage order is ``[x, y, z, basis]`` with z varying fastest when the
    lattice is flattened.  Points are normalized coordinates in
    ``[-extent, extent]^3``; endpoint samples are retained exactly and values
    outside the box are zeroed after interpolation.
    """

    def __init__(
        self,
        grid_shape,
        extent: float,
        device: torch.device | str,
        max_degree: int = 3,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        try:
            if any(isinstance(value, bool) for value in grid_shape):
                raise ValueError
            shape = tuple(operator.index(value) for value in grid_shape)
        except (TypeError, ValueError):
            raise ValueError("grid_shape must contain exactly three integers >= 2") from None
        if len(shape) != 3 or any(value < 2 for value in shape):
            raise ValueError("grid_shape must contain exactly three integers >= 2")
        if not math.isfinite(float(extent)) or float(extent) <= 0.0:
            raise ValueError("extent must be positive and finite")
        if not isinstance(max_degree, int) or isinstance(max_degree, bool):
            raise ValueError("max_degree must be an integer")
        n_basis = num_sh_basis(max_degree)
        self.grid_shape = shape
        self.extent = float(extent)
        self.max_degree = int(max_degree)
        self.register_buffer("basis_degree", basis_degree_index(max_degree, device=device))
        self.w_re = nn.Parameter(
            init_scale * torch.randn(*shape, n_basis, device=device, dtype=torch.float32)
        )
        self.w_im = nn.Parameter(
            init_scale * torch.randn(*shape, n_basis, device=device, dtype=torch.float32)
        )

    def _validate_points(self, points: torch.Tensor) -> None:
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError("points must have shape [N,3]")

    def _interpolate(self, values: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        self._validate_points(points)
        coordinates = []
        for axis, size in enumerate(self.grid_shape):
            coordinate = (points[:, axis] + self.extent) * ((size - 1) / (2.0 * self.extent))
            coordinates.append(coordinate.clamp(0.0, float(size - 1)))
        coordinate = torch.stack(coordinates, dim=-1)
        low = torch.floor(coordinate).to(torch.long)
        high = torch.stack(
            [torch.minimum(low[:, axis] + 1, low[:, axis].new_full((), size - 1))
             for axis, size in enumerate(self.grid_shape)],
            dim=-1,
        )
        fraction = coordinate - low.to(coordinate.dtype)
        output = torch.zeros(
            (points.shape[0], values.shape[-1]), device=points.device, dtype=values.dtype
        )
        for bx in (0, 1):
            ix = high[:, 0] if bx else low[:, 0]
            wx = fraction[:, 0] if bx else 1.0 - fraction[:, 0]
            for by in (0, 1):
                iy = high[:, 1] if by else low[:, 1]
                wy = fraction[:, 1] if by else 1.0 - fraction[:, 1]
                for bz in (0, 1):
                    iz = high[:, 2] if bz else low[:, 2]
                    wz = fraction[:, 2] if bz else 1.0 - fraction[:, 2]
                    output = output + values[ix, iy, iz] * (wx * wy * wz)[:, None]
        inside = ((points >= -self.extent) & (points <= self.extent)).all(dim=-1)
        return output * inside[:, None].to(output.dtype)

    def _chunk_slices(self, points: torch.Tensor, chunk_size: int):
        if chunk_size < 0:
            raise ValueError("chunk_size must be nonnegative")
        if chunk_size == 0:
            chunk_size = max(int(points.shape[0]), 1)
        for start in range(0, points.shape[0], chunk_size):
            yield points[start : start + chunk_size]

    def query_coefficients(self, points: torch.Tensor, chunk_size: int = 0) -> torch.Tensor:
        outputs = []
        for chunk in self._chunk_slices(points, int(chunk_size)):
            real = self._interpolate(self.w_re, chunk)
            imag = self._interpolate(self.w_im, chunk)
            outputs.append(torch.complex(real, imag))
        if not outputs:
            return torch.empty(
                (0, self.w_re.shape[-1]), device=points.device, dtype=torch.complex64
            )
        return torch.cat(outputs, dim=0)

    def query_dc(self, points: torch.Tensor, chunk_size: int = 0) -> torch.Tensor:
        outputs = []
        for chunk in self._chunk_slices(points, int(chunk_size)):
            real = self._interpolate(self.w_re[..., :1], chunk)
            imag = self._interpolate(self.w_im[..., :1], chunk)
            outputs.append(torch.complex(real, imag)[:, 0])
        if not outputs:
            return torch.empty((0,), device=points.device, dtype=torch.complex64)
        return torch.cat(outputs, dim=0)


class RIFTSASGrid(SHVoxelGridScene):
    """RIFT SH grid with a continuous Plenoxel-style query operation."""

    def _trilinear(self, values: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError("points must have shape [N,3]")
        granularity = self.granularity
        # RIFT's grid stores cell centres at -extent+pitch/2 ... extent-pitch/2,
        # not endpoint samples.  Border replication over the outer half-cell
        # is the Plenoxel convention and keeps the full scene box queryable.
        pitch = 2.0 * self.extent / granularity
        coordinate = (points + self.extent) / pitch - 0.5
        inside = ((points >= -self.extent) & (points <= self.extent)).all(dim=-1)
        coordinate = coordinate.clamp(0.0, float(granularity - 1))
        low = torch.floor(coordinate).to(torch.long)
        high = (low + 1).clamp_max(granularity - 1)
        fraction = coordinate - low.to(coordinate.dtype)

        output = torch.zeros(points.shape[0], values.shape[-1], device=points.device, dtype=values.dtype)
        for bx in (0, 1):
            ix = high[:, 0] if bx else low[:, 0]
            wx = fraction[:, 0] if bx else 1.0 - fraction[:, 0]
            for by in (0, 1):
                iy = high[:, 1] if by else low[:, 1]
                wy = fraction[:, 1] if by else 1.0 - fraction[:, 1]
                for bz in (0, 1):
                    iz = high[:, 2] if bz else low[:, 2]
                    wz = fraction[:, 2] if bz else 1.0 - fraction[:, 2]
                    output = output + values[ix, iy, iz] * (wx * wy * wz)[:, None]
        return output * inside[:, None].to(output.dtype)

    def query_coefficients(self, points: torch.Tensor, chunk_size: int = 0) -> torch.Tensor:
        del chunk_size
        order_mask = (
            self.basis_degree.reshape(1, 1, 1, -1) <= self.order.unsqueeze(-1)
        ).to(self.w_re.dtype)
        active = self.active_mask.unsqueeze(-1).to(self.w_re.dtype)
        coefficients = torch.complex(self.w_re * order_mask * active, self.w_im * order_mask * active)
        return self._trilinear(coefficients, points)

    def query_dc(self, points: torch.Tensor) -> torch.Tensor:
        """Interpolate only c00; used by finite-difference normals to bound memory."""
        active = self.active_mask.to(self.w_re.dtype)
        dc = torch.complex(self.w_re[..., 0] * active, self.w_im[..., 0] * active).unsqueeze(-1)
        return self._trilinear(dc, points)[:, 0]


class AdaptiveRIFTSASField(nn.Module):
    """Continuous coefficient field backed by :class:`AdaptivePointSHScene`.

    Active point coefficients are differentiably splatted to a fixed regular
    raster with a trilinear hat kernel, then queried with the same trilinear
    interpolation used by :class:`RIFTSASGrid`.  The raster spacing (and thus
    the kernel width) is fixed at construction time; it is deliberately
    independent of each adaptive point's ``cell_half`` so an in-place heir
    split with zero siblings leaves the represented field unchanged.

    The scene stores normalized coordinates in ``[-1, 1]^3``.  The enclosing
    :class:`ComplexSHSonarField` performs the physical-to-normalized transform
    and supplies the physical finite-difference step converted per axis.
    """

    def __init__(
        self,
        scene: AdaptivePointSHScene,
        raster_granularity: int = 64,
        extent: float = 1.0,
        query_chunk: int = 65536,
    ) -> None:
        super().__init__()
        if not isinstance(scene, AdaptivePointSHScene):
            raise TypeError("scene must be an AdaptivePointSHScene")
        if raster_granularity < 2 or extent <= 0:
            raise ValueError("raster_granularity must be >= 2 and extent must be positive")
        if query_chunk <= 0:
            raise ValueError("query_chunk must be positive")
        self.scene = scene
        self.raster_granularity = int(raster_granularity)
        self.extent = float(extent)
        self.query_chunk = int(query_chunk)
        raster_spacing = 2.0 * self.extent / float(self.raster_granularity - 1)
        self.register_buffer("raster_kernel_width", torch.tensor(raster_spacing))

    @property
    def underlying_scene(self) -> AdaptivePointSHScene:
        return self.scene

    @property
    def sh_degree(self) -> int:
        return int(self.scene.max_degree)

    def _masked_coefficients(self, probe_next_band: bool = False) -> torch.Tensor:
        order = self.scene.order
        if probe_next_band:
            order = torch.minimum(order + 1, torch.full_like(order, self.scene.max_degree))
        mask = self.scene.basis_degree.reshape(1, -1) <= order.reshape(-1, 1)
        mask = mask & self.scene.active_mask.reshape(-1, 1)
        mask_f = mask.to(self.scene.w_re.dtype)
        return torch.complex(self.scene.w_re * mask_f, self.scene.w_im * mask_f)

    def _rasterize(self, probe_next_band: bool = False):
        """Splat active points once and return real/imag rasters plus weights."""
        g = self.raster_granularity
        dtype = self.scene.w_re.dtype
        device = self.scene.w_re.device
        positions = self.scene.positions()
        active = self.scene.active_mask
        positions = positions[active]
        coefficients = self._masked_coefficients(probe_next_band)[active]
        if positions.numel() == 0:
            empty = torch.zeros(g ** 3, coefficients.shape[-1], device=device, dtype=dtype)
            return empty, empty.clone(), torch.zeros(g ** 3, device=device, dtype=dtype)

        spacing = positions.new_tensor(float(self.raster_kernel_width))
        coordinate = ((positions + self.extent) / spacing).clamp(0.0, float(g - 1))
        low = torch.floor(coordinate).to(torch.long)
        high = (low + 1).clamp_max(g - 1)
        fraction = coordinate - low.to(coordinate.dtype)
        # Normalize each point's fixed trilinear hat independently.  This
        # makes a query at the point recover its coefficient while remaining
        # additive across points; zero-weight siblings therefore cannot alter
        # the inherited heir's field contribution.
        splat_norm = torch.zeros(positions.shape[0], device=device, dtype=dtype)
        for bx in (0, 1):
            wx = fraction[:, 0] if bx else 1.0 - fraction[:, 0]
            for by in (0, 1):
                wy = fraction[:, 1] if by else 1.0 - fraction[:, 1]
                for bz in (0, 1):
                    wz = fraction[:, 2] if bz else 1.0 - fraction[:, 2]
                    splat_norm = splat_norm + (wx * wy * wz).square()
        splat_norm = splat_norm.clamp_min(1.0e-12)
        real = torch.zeros(g ** 3, coefficients.shape[-1], device=device, dtype=dtype)
        imag = torch.zeros_like(real)
        weights = torch.zeros(g ** 3, device=device, dtype=dtype)
        for bx in (0, 1):
            ix = high[:, 0] if bx else low[:, 0]
            wx = fraction[:, 0] if bx else 1.0 - fraction[:, 0]
            for by in (0, 1):
                iy = high[:, 1] if by else low[:, 1]
                wy = fraction[:, 1] if by else 1.0 - fraction[:, 1]
                for bz in (0, 1):
                    iz = high[:, 2] if bz else low[:, 2]
                    wz = fraction[:, 2] if bz else 1.0 - fraction[:, 2]
                    flat = ix * (g * g) + iy * g + iz
                    contribution = (wx * wy * wz) / splat_norm
                    real.index_add_(0, flat, coefficients.real * contribution[:, None])
                    imag.index_add_(0, flat, coefficients.imag * contribution[:, None])
                    weights.index_add_(0, flat, contribution)
        return real, imag, weights

    def _query_raster(self, values: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError("points must have shape [N,3]")
        g = self.raster_granularity
        spacing = points.new_tensor(float(self.raster_kernel_width))
        coordinate = (points + self.extent) / spacing
        inside = ((points >= -self.extent) & (points <= self.extent)).all(dim=-1)
        coordinate = coordinate.clamp(0.0, float(g - 1))
        low = torch.floor(coordinate).to(torch.long)
        high = (low + 1).clamp_max(g - 1)
        fraction = coordinate - low.to(coordinate.dtype)
        output = torch.zeros(points.shape[0], values.shape[-1], device=points.device, dtype=values.dtype)
        values = values.reshape(g, g, g, -1)
        for bx in (0, 1):
            ix = high[:, 0] if bx else low[:, 0]
            wx = fraction[:, 0] if bx else 1.0 - fraction[:, 0]
            for by in (0, 1):
                iy = high[:, 1] if by else low[:, 1]
                wy = fraction[:, 1] if by else 1.0 - fraction[:, 1]
                for bz in (0, 1):
                    iz = high[:, 2] if bz else low[:, 2]
                    wz = fraction[:, 2] if bz else 1.0 - fraction[:, 2]
                    output = output + values[ix, iy, iz] * (wx * wy * wz)[:, None]
        return output * inside[:, None].to(output.dtype)

    def _query_rasterized(self, raster, points: torch.Tensor) -> torch.Tensor:
        real, imag, _weights = raster
        numerator = self._query_raster(torch.complex(real, imag), points)
        return numerator

    def _query_rasterized_dc(self, raster, points: torch.Tensor) -> torch.Tensor:
        real, imag, _weights = raster
        dc = torch.complex(real[:, :1], imag[:, :1])
        return self._query_raster(dc, points)[:, 0]

    def query_coefficients(self, points: torch.Tensor, chunk_size: int = 0) -> torch.Tensor:
        del chunk_size
        return self._query_rasterized(self._rasterize(), points)

    def query_dc(self, points: torch.Tensor) -> torch.Tensor:
        return self._query_rasterized_dc(self._rasterize(), points)

    def query_density(self, points: torch.Tensor, raster=None) -> torch.Tensor:
        if raster is None:
            raster = self._rasterize()
        outputs = []
        for start in range(0, points.shape[0], self.query_chunk):
            outputs.append(
                self._query_rasterized_dc(raster, points[start : start + self.query_chunk]).abs() * Y00
            )
        return torch.cat(outputs, dim=0)

    def query_sas(
        self,
        points: torch.Tensor,
        directions: torch.Tensor,
        *,
        normal_steps: torch.Tensor | float,
        gradient_steps: Optional[torch.Tensor | float] = None,
        probe_next_band: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if points.shape != directions.shape or points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError("points and directions must both have shape [N,3]")
        raster = self._rasterize(probe_next_band=probe_next_band)
        steps = torch.as_tensor(normal_steps, device=points.device, dtype=points.dtype)
        if steps.ndim == 0:
            steps = steps.expand(3)
        if steps.shape != (3,) or bool((steps <= 0).any()):
            raise ValueError("normal_steps must be a positive scalar or [3] vector")
        gradient_steps = steps if gradient_steps is None else torch.as_tensor(
            gradient_steps, device=points.device, dtype=points.dtype
        )
        if gradient_steps.ndim == 0:
            gradient_steps = gradient_steps.expand(3)
        if gradient_steps.shape != (3,) or bool((gradient_steps <= 0).any()):
            raise ValueError("gradient_steps must be a positive scalar or [3] vector")
        offsets = torch.diag(steps)
        plus_points = (points[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
        minus_points = (points[:, None, :] - offsets[None, :, :]).reshape(-1, 3)
        queried = self._query_rasterized(raster, points)
        n = points.shape[0]
        coefficients = queried
        # Density is the magnitude of the complex DC coefficient.  Taking the
        # magnitude before the finite difference keeps the normal field real
        # and matches the representation-level density contract below.
        plus = self._query_rasterized_dc(raster, plus_points).reshape(n, 3).abs() * Y00
        minus = self._query_rasterized_dc(raster, minus_points).reshape(n, 3).abs() * Y00
        density = coefficients[:, 0].abs() * Y00
        gradient = (plus - minus) / (2.0 * gradient_steps[None, :])
        normals = -gradient / torch.linalg.vector_norm(gradient, dim=-1, keepdim=True).clamp_min(1.0e-12)
        normals = torch.where(torch.isfinite(normals), normals, torch.zeros_like(normals))
        basis = real_sh_basis_for_directions(directions, self.scene.max_degree)
        scatterer = (coefficients * basis.to(coefficients.dtype)).sum(dim=-1)
        return {
            "coefficients": coefficients,
            "scatterer": scatterer,
            "density": density,
            "normals": normals,
        }

    @torch.no_grad()
    def dense_density(self, points: torch.Tensor, chunk_size: int = 131072) -> torch.Tensor:
        raster = self._rasterize()
        outputs = []
        for start in range(0, points.shape[0], chunk_size):
            outputs.append(self.query_density(points[start : start + chunk_size], raster=raster).cpu())
        return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def geometry_compatibility_state(self, points: torch.Tensor, chunk_size: int = 131072) -> Dict[str, torch.Tensor]:
        raster = self._rasterize()
        coefficients = []
        for start in range(0, points.shape[0], chunk_size):
            coefficients.append(self._query_rasterized(raster, points[start : start + chunk_size]).cpu())
        value = torch.cat(coefficients, dim=0)
        return {"w_re": value.real.contiguous(), "w_im": value.imag.contiguous()}


class ComplexSHSonarField(nn.Module):
    """Map physical points to a complex-SH coefficient representation."""

    def __init__(
        self,
        coefficient_field: nn.Module,
        scene_min: torch.Tensor,
        scene_max: torch.Tensor,
        sh_degree: int,
        query_chunk: int = 65536,
    ) -> None:
        super().__init__()
        self.coefficient_field = coefficient_field
        self.sh_degree = int(sh_degree)
        self.query_chunk = int(query_chunk)
        scene_min = torch.as_tensor(scene_min, dtype=torch.float32)
        scene_max = torch.as_tensor(scene_max, dtype=torch.float32)
        if scene_min.shape != (3,) or scene_max.shape != (3,) or not torch.all(scene_max > scene_min):
            raise ValueError("scene_min/scene_max must be ordered 3-vectors")
        self.register_buffer("scene_center", (scene_min + scene_max) / 2.0)
        self.register_buffer("scene_half_extent", (scene_max - scene_min) / 2.0)

    @property
    def underlying_scene(self):
        return getattr(self.coefficient_field, "underlying_scene", None)

    def _model_points(self, physical_points: torch.Tensor) -> torch.Tensor:
        # Both coefficient fields are constructed on the normalized cube [-1,1]^3.
        return (physical_points - self.scene_center) / self.scene_half_extent

    def query_coefficients(self, physical_points: torch.Tensor) -> torch.Tensor:
        return self.coefficient_field.query_coefficients(
            self._model_points(physical_points), chunk_size=self.query_chunk
        )

    def query_density(self, physical_points: torch.Tensor) -> torch.Tensor:
        model_points = self._model_points(physical_points)
        if hasattr(self.coefficient_field, "query_density"):
            return self.coefficient_field.query_density(model_points)
        outputs = []
        for start in range(0, physical_points.shape[0], self.query_chunk):
            points = physical_points[start : start + self.query_chunk]
            if hasattr(self.coefficient_field, "query_density"):
                raise AssertionError("representation-level density query should be handled above")
            if hasattr(self.coefficient_field, "query_dc"):
                dc = self.coefficient_field.query_dc(self._model_points(points))
            else:
                dc = self.coefficient_field.query_coefficients(
                    self._model_points(points), chunk_size=self.query_chunk
                )[..., 0]
            outputs.append(dc.abs() * Y00)
        return torch.cat(outputs, dim=0)

    def query_sas(
        self, physical_points: torch.Tensor, directions: torch.Tensor, *, normal_step: float,
        probe_next_band: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if physical_points.shape != directions.shape or physical_points.shape[-1] != 3:
            raise ValueError("physical_points and directions must both have shape [N,3]")
        model_points = self._model_points(physical_points)
        if hasattr(self.coefficient_field, "query_sas"):
            model_steps = physical_points.new_tensor(float(normal_step)) / self.scene_half_extent
            return self.coefficient_field.query_sas(
                model_points,
                directions,
                normal_steps=model_steps,
                gradient_steps=physical_points.new_tensor(float(normal_step)).expand(3),
                probe_next_band=probe_next_band,
            )
        coefficients = self.query_coefficients(physical_points)
        basis = real_sh_basis_for_directions(directions, self.sh_degree)
        scatterer = (coefficients * basis.to(coefficients.dtype)).sum(dim=-1)
        density = coefficients[..., 0].abs() * Y00

        step = float(normal_step)
        if step <= 0:
            raise ValueError("normal_step must be positive")
        offsets = physical_points.new_tensor(
            [[step, 0.0, 0.0], [0.0, step, 0.0], [0.0, 0.0, step]]
        )
        plus = physical_points[:, None, :] + offsets[None, :, :]
        minus = physical_points[:, None, :] - offsets[None, :, :]
        density_plus = self.query_density(plus.reshape(-1, 3)).reshape(-1, 3)
        density_minus = self.query_density(minus.reshape(-1, 3)).reshape(-1, 3)
        gradient = (density_plus - density_minus) / (2.0 * step)
        normals = -gradient / torch.linalg.vector_norm(gradient, dim=-1, keepdim=True).clamp_min(1e-12)
        normals = torch.where(torch.isfinite(normals), normals, torch.zeros_like(normals))
        return {
            "coefficients": coefficients,
            "scatterer": scatterer,
            "density": density,
            "normals": normals,
        }

    @torch.no_grad()
    def dense_density(self, physical_points: torch.Tensor, chunk_size: int = 131072) -> torch.Tensor:
        was_training = self.training
        self.eval()
        result = []
        for start in range(0, physical_points.shape[0], chunk_size):
            result.append(self.query_density(physical_points[start : start + chunk_size]).cpu())
        if was_training:
            self.train()
        return torch.cat(result, dim=0)
