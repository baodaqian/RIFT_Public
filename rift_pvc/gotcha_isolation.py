"""Per-sector box isolation for GOTCHA in range and cross-range (Phase 0c prototype).

Background (docs/RIFT_GOTCHA_Tune.md sections 5, 7 and 10):
- Today's target, ``RangeReadout``, isolates the registered region in range only, per pulse. At
  ~10 km that keeps a strip ~15 m deep across the whole illuminated lot, plus its range aliases.
- A sector's native pulses (~117 over 0.69 degrees) also resolve cross-range (~1.3 m; alias ~150 m).

The operator here projects a sector's (pulse x frequency) data onto the span of responses from a
box. One sector sees from one direction, so it cannot separate points along the layover normal
(perpendicular to the look direction and the flight direction). The box's data footprint is
therefore its projection onto the sector's slant plane (look direction u, cross-range v). A 2D pixel
grid on that plane, covering the projected box plus a declared guard band, spans the box's
responses; a 3D dictionary is not needed.

    D_s  = responses of the slant-plane footprint grid (geometry only: antennas, frequencies, r0)
    Q_s  = leading left singular vectors of D_s (rank chosen by a relative singular-value cutoff)
    F_s  = Q_s Q_s^H, applied identically to data and to every method's predictions

Nothing here reads a response except ``retained_fraction`` on data the caller passes in. The
cutoff and guard are to be chosen by the injection tests (``scripts_pvc/gotcha_isolation_qualify.py``),
not carried over from ``RangeReadout``'s 1e-10.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from rift.gotcha_dataset import C


@dataclass(frozen=True)
class Box:
    """Rectangular box in the region's local frame: centre, heading of its x side, half-extents."""
    centre: tuple
    heading_deg: float
    half_extents: tuple

    def axes(self):
        h = math.radians(self.heading_deg)
        return np.array([[math.cos(h), math.sin(h), 0.], [-math.sin(h), math.cos(h), 0.], [0., 0., 1.]])

    def to_box(self, points):
        return (np.asarray(points, dtype=np.float64) - np.asarray(self.centre)) @ self.axes().T

    def from_box(self, coords):
        return np.asarray(coords, dtype=np.float64) @ self.axes() + np.asarray(self.centre)

    def contains(self, points, margin=0.):
        return (np.abs(self.to_box(points)) <= np.asarray(self.half_extents) + margin).all(-1)

    def distance(self, points):
        """Euclidean distance from the box surface (0 inside)."""
        outside = np.maximum(np.abs(self.to_box(points)) - np.asarray(self.half_extents), 0.)
        return np.linalg.norm(outside, axis=-1)

    def corners(self):
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
        return self.from_box(signs * np.asarray(self.half_extents))

    def sample(self, count, rng):
        return self.from_box(rng.uniform(-1, 1, size=(count, 3)) * np.asarray(self.half_extents))


@dataclass
class SectorGeometry:
    antennas: np.ndarray      # [P, 3] local
    frequencies: np.ndarray   # [F] Hz (one frequency set per sector)
    reference: np.ndarray     # [P] effective r0 (m)

    @classmethod
    def from_observations(cls, observations, region):
        frequencies = np.asarray(observations[0].frequencies_hz, dtype=np.float64)
        if any(len(o.frequencies_hz) != len(frequencies) or not np.array_equal(o.frequencies_hz, frequencies)
               for o in observations):
            raise ValueError('pulses of one sector must share one frequency set')
        return cls(np.stack([region.to_local(o.position_m) for o in observations]), frequencies,
                   np.asarray([o.reference_range_m for o in observations], dtype=np.float64))

    @property
    def samples(self):
        return len(self.antennas) * len(self.frequencies)

    def frame(self, centre):
        """Slant-plane frame at ``centre``: u toward the radar, v cross-range, n the layover normal."""
        look = self.antennas.mean(0) - np.asarray(centre)
        u = look / np.linalg.norm(look)
        track = self.antennas[-1] - self.antennas[0]
        v = track - (track @ u) * u
        v /= np.linalg.norm(v)
        return u, v, np.cross(u, v)

    def responses(self, points, *, dtype=torch.complex64, device='cpu'):
        """Unit-reflectivity responses [P*F, N] of local ``points`` (native kernel, reference-adjusted)."""
        pts = torch.as_tensor(np.asarray(points), dtype=torch.float64, device=device)
        a = torch.as_tensor(self.antennas, dtype=torch.float64, device=device)
        f = torch.as_tensor(self.frequencies, dtype=torch.float64, device=device)
        r0 = torch.as_tensor(self.reference, dtype=torch.float64, device=device)
        distance = torch.cdist(a, pts) - r0[:, None]                             # [P, N]
        phase = (-4 * math.pi / C) * distance[:, None, :] * f[None, :, None]      # [P, F, N]
        return torch.polar(torch.ones_like(phase), phase).reshape(-1, len(pts)).to(dtype)


def footprint_grid(box, geometry, *, range_step, cross_step, range_guard, cross_guard, shape='hull'):
    """Slant-plane pixel grid through the box centre covering the box's projection plus guard bands.

    Guards are separate because a range cell (~0.24 m) and a cross-range cell (~1.3 m over a
    0.69-degree sector) differ five-fold (reviewer B2). ``shape='hull'`` keeps the pixels inside the
    projected corners' convex hull grown by the guard rectangle (a Minkowski sum). ``'rectangle'``
    keeps the hull's bounding rectangle, which at oblique look angles nearly doubles the area and
    reaches exterior features (qualification 2156732: neighbours at y = +-7 m kept whole).
    """
    u, v, _ = geometry.frame(box.centre)
    corners = box.corners() - np.asarray(box.centre)
    ru, rv = corners @ u, corners @ v
    a = np.arange(ru.min() - range_guard, ru.max() + range_guard + range_step / 2, range_step)
    b = np.arange(rv.min() - cross_guard, rv.max() + cross_guard + cross_step / 2, cross_step)
    A, B = np.meshgrid(a, b, indexing='ij')
    A, B = A.reshape(-1), B.reshape(-1)
    if shape == 'hull':
        from scipy.spatial import Delaunay
        grown = np.array([(x + da, y + db) for x, y in zip(ru, rv)
                          for da in (-range_guard, range_guard) for db in (-cross_guard, cross_guard)])
        inside = Delaunay(grown).find_simplex(np.stack([A, B], 1)) >= 0
        A, B = A[inside], B[inside]
    elif shape != 'rectangle':
        raise ValueError(f'unknown footprint shape {shape!r}')
    return np.asarray(box.centre) + A[:, None] * u + B[:, None] * v


def basis(dictionary):
    """Left singular vectors and singular values of the footprint dictionary (economy SVD)."""
    Q, S, _ = torch.linalg.svd(dictionary, full_matrices=False)
    return Q, S


def truncate(Q, S, cutoff):
    rank = int((S >= cutoff * S[0]).sum())
    return Q[:, :rank], rank


def retained_fraction(Q, vectors):
    """||Q^H y||^2 / ||y||^2 per column of ``vectors`` [samples, N]."""
    coefficients = Q.conj().T @ vectors
    return (coefficients.abs().square().sum(0) / vectors.abs().square().sum(0)).real
