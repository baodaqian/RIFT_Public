"""SpINR kernels for native monostatic phase histories with per-pulse r0.

Native frequencies are never resampled. Exactly affine vectors admit the
closed-form finite DFT; other vectors use the finite DFT of each point kernel.
The latter is an acquisition adaptation, not a claim of paper timing parity.
"""
from __future__ import annotations

import math
import numpy as np
import torch

from rift.gotcha_dataset import C
from rift.spinr_direct import _stable_sinc, _volume


class NativeKernel:
    def __init__(self, observation, region, *, device="cpu", point_tile=512):
        if type(point_tile) is not int or point_tile < 1:
            raise ValueError("native point tile must be a positive integer")
        self.f = torch.as_tensor(np.array(observation.frequencies_hz, copy=True), device=device, dtype=torch.float64)
        self.antenna = torch.as_tensor(region.to_local(observation.position_m), device=device, dtype=torch.float64)
        self.r0 = float(observation.reference_range_m)
        self.support = float(region.half_extent_m)
        if (self.f.ndim != 1 or len(self.f) < 2 or not torch.isfinite(self.f).all()
                or not (self.f[1:] > self.f[:-1]).all() or self.f[0] <= 0
                or self.antenna.shape != (3,) or not torch.isfinite(self.antenna).all()
                or not math.isfinite(self.r0)):
            raise ValueError("invalid native SpINR acquisition")
        self.point_tile = min(point_tile, max(1, 1_048_576//len(self.f)))
        # Equality is intentional: an allclose tolerance would discard native
        # frequency residuals, which can change coherent carrier phase.
        ideal = self.f[0]+torch.arange(len(self.f), device=device, dtype=torch.float64)*(self.f[1]-self.f[0])
        self.affine = torch.equal(self.f, ideal)
        r_min = torch.linalg.vector_norm((self.antenna.abs()-self.support).clamp_min(0))
        r_max = torch.linalg.vector_norm(self.antenna.abs()+self.support)
        delays = torch.stack((r_min-self.r0, r_max-self.r0))
        increments = self.f[1:]-self.f[:-1]
        bounds = (-2*len(self.f)/C)*delays[:, None]*torch.stack((increments.min(), increments.max()))[None]
        first, last = bounds.min().floor(), bounds.max().ceil()
        bins = torch.arange(len(self.f), device=device, dtype=torch.float64)
        self.mask = torch.remainder(bins-first, len(self.f)) <= last-first
        self.bin_ids = self.mask.nonzero().flatten()
        if not len(self.bin_ids):
            raise ValueError("native scene bin selection is empty")

    def tiles(self, points, *, selected=True):
        if points.ndim != 2 or points.shape[1] != 3 or not torch.isfinite(points).all():
            raise ValueError("native integration points must be finite [N,3]")
        for start in range(0, len(points), self.point_tile):
            stop = min(start+self.point_tile, len(points))
            distance = torch.linalg.vector_norm(points[start:stop].double()-self.antenna, dim=-1)
            if bool((distance <= 0).any()):
                raise ValueError("native scatterer coincides with the antenna")
            delta_r = distance-self.r0
            if selected and self.affine:
                n = len(self.f)
                beta = (2*math.pi/n)*self.bin_ids.double()
                delta = (-4*math.pi/C)*(self.f[1]-self.f[0])*delta_r[:, None]-beta[None]
                delta = torch.remainder(delta+math.pi, 2*math.pi)-math.pi
                phase = (-4*math.pi/C)*self.f[0]*delta_r[:, None]+.5*(n-1)*delta
                kernel = (_stable_sinc(n*delta/(2*math.pi))/_stable_sinc(delta/(2*math.pi))
                          *torch.exp(1j*phase)/distance[:, None].square())
            else:
                phase = (-4*math.pi/C)*delta_r[:, None]*self.f[None]
                kernel = torch.exp(1j*phase)/distance[:, None].square()
                if selected:
                    kernel = torch.fft.fft(kernel, dim=1, norm="forward")[:, self.bin_ids]
            yield start, stop, kernel

    def render(self, points, field, volume, scale, *, selected=True):
        if field.shape != (len(points),) or torch.is_complex(field) or not torch.isfinite(field).all():
            raise ValueError("native SpINR field must be finite signed real")
        amplitude = field.double()*_volume(points, volume, scale)
        prediction = torch.zeros(len(self.bin_ids) if selected else len(self.f),
                                 dtype=torch.complex128, device=points.device)
        for start, stop, kernel in self.tiles(points, selected=selected):
            prediction = prediction+(amplitude[start:stop, None]*kernel).sum(0)
        return prediction

    @torch.no_grad()
    def field_vjp(self, points, volume, scale, cotangent):
        if cotangent.shape != (len(self.bin_ids),) or not torch.isfinite(cotangent).all():
            raise ValueError("invalid native selected-bin cotangent")
        weights = _volume(points, volume, scale)
        result = torch.zeros(len(points), dtype=torch.float64, device=points.device)
        for start, stop, kernel in self.tiles(points):
            result[start:stop] = (kernel.conj()*cotangent[None]).real.sum(1)*weights[start:stop]
        return result

    def target_bins(self, response):
        target = torch.as_tensor(np.array(response, copy=True), device=self.f.device, dtype=torch.complex128)
        if target.shape != self.f.shape or not torch.isfinite(target).all():
            raise ValueError("invalid native response")
        return torch.fft.fft(target, norm="forward")[self.bin_ids]


def bin_objective(prediction, target, mean_raw_power):
    if not math.isfinite(mean_raw_power) or mean_raw_power <= 0 or prediction.shape != target.shape:
        raise ValueError("invalid native objective shape or TRAIN normalization")
    return ((prediction.abs()-target.abs()).square()+.5*(prediction-target).abs().square()).sum()/mean_raw_power


def loss_and_field_vjp(kernel, points, field, volume, scale, response, mean_raw_power):
    with torch.no_grad():
        prediction = kernel.render(points, field, volume, scale)
        target = kernel.target_bins(response)
    with torch.enable_grad():
        prediction.requires_grad_(True)
        loss = bin_objective(prediction, target, mean_raw_power)
        cotangent, = torch.autograd.grad(loss, prediction)
    if not torch.isfinite(loss) or not torch.isfinite(cotangent).all():
        raise FloatingPointError("nonfinite native SpINR loss/gradient")
    return float(loss.detach()), kernel.field_vjp(points, volume, scale, cotangent.detach())
