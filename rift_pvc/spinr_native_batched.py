"""Batched native SpINR kernel for pulses that share one native frequency vector.

``rift/spinr_native.py``'s ``NativeKernel`` renders one pulse at a time in
tiles of ``renderer_point_tile`` points (512 in the GOTCHA recipe), which on
an accelerator is launch-bound: 1024 pulses × 216 tiles × ~8 kernels per
update. This kernel evaluates all pulses of one shard group at once as
``[pulses, tile, bins]`` tensors with the same formulas, the same per-pulse
scene-bin masks (padded, masked) and the same closed-form (affine) or exact
point-kernel DFT (ragged) modes. Results equal the per-pulse kernel up to
floating-point summation order. PVC-only; the CUDA runtime is unchanged.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from rift.gotcha_dataset import C
from rift.spinr_direct import _stable_sinc, _volume

PULSE_EXECUTION = 'batched_shard_groups_one_vjp_dft_bmm_v2'
# Ragged (non-affine) vectors: the finite index-DFT of the point kernel restricted to
# each pulse's selected bins is evaluated as a batched matrix product with the
# per-pulse DFT matrix (FP64 complex GEMM) instead of an FFT over all n bins followed
# by a gather; the same exact operator, evaluated on the selected bins only.
DFT_MODE = 'bmm'   # 'bmm' | 'fft' (the per-pulse reference's evaluation, kept for tests)
# Exact closed form for a frequency vector that is affine except for one trailing
# sample (the stride-2 selection keeps the last native bin): the finite DFT of the
# point kernel is a 212-term geometric series plus one term, evaluated only on the
# selected bins, so no 213-point FFT (which has the prime factor 71) is needed.
CLOSED_FORM_ENDPOINT = True
ELEMENT_BUDGET = 1 << 24   # complex128 elements per kernel tile (256 MiB), an execution bound only


class BatchedNativeKernel:
    def __init__(self, observations, region, *, device='cpu', point_tile=512, element_budget=ELEMENT_BUDGET):
        if type(point_tile) is not int or point_tile < 1:
            raise ValueError('native point tile must be a positive integer')
        if not observations:
            raise ValueError('batched native kernel needs at least one pulse')
        f0 = np.array(observations[0].frequencies_hz, copy=True)
        for o in observations[1:]:
            if not np.array_equal(np.asarray(o.frequencies_hz), f0):
                raise ValueError('pulses of one batched kernel must share the native frequency vector')
        self.f = torch.as_tensor(f0, device=device, dtype=torch.float64)
        self.antennas = torch.as_tensor(np.stack([region.to_local(o.position_m) for o in observations]),
                                        device=device, dtype=torch.float64)
        self.r0 = torch.tensor([float(o.reference_range_m) for o in observations], device=device, dtype=torch.float64)
        self.support = float(region.half_extent_m)
        n, p = len(self.f), len(observations)
        if (self.f.ndim != 1 or n < 2 or not torch.isfinite(self.f).all()
                or not (self.f[1:] > self.f[:-1]).all() or self.f[0] <= 0
                or self.antennas.shape != (p, 3) or not torch.isfinite(self.antennas).all()
                or not torch.isfinite(self.r0).all()):
            raise ValueError('invalid native SpINR acquisition')
        self.pulses = p
        ideal = self.f[0] + torch.arange(n, device=device, dtype=torch.float64) * (self.f[1] - self.f[0])
        self.affine = torch.equal(self.f, ideal)
        # Affine prefix of n-1 samples plus one trailing sample of any value.
        self.affine_prefix = (not self.affine) and n >= 3 and torch.equal(self.f[:-1], ideal[:-1])
        self.closed_form_endpoint = bool(CLOSED_FORM_ENDPOINT and self.affine_prefix)
        r_min = torch.linalg.vector_norm((self.antennas.abs() - self.support).clamp_min(0), dim=-1)
        r_max = torch.linalg.vector_norm(self.antennas.abs() + self.support, dim=-1)
        delays = torch.stack((r_min - self.r0, r_max - self.r0), dim=1)                     # [P,2]
        increments = self.f[1:] - self.f[:-1]
        bounds = (-2 * n / C) * delays[:, :, None] * torch.stack((increments.min(), increments.max()))[None, None]
        first = bounds.reshape(p, -1).min(dim=1).values.floor()
        last = bounds.reshape(p, -1).max(dim=1).values.ceil()
        bins = torch.arange(n, device=device, dtype=torch.float64)
        self.mask = torch.remainder(bins[None] - first[:, None], n) <= (last - first)[:, None]   # [P,n]
        counts = self.mask.sum(dim=1)
        if bool((counts == 0).any()):
            raise ValueError('native scene bin selection is empty')
        self.width = int(counts.max())
        order = torch.argsort((~self.mask).to(torch.int8), dim=1, stable=True)
        self.valid = torch.arange(self.width, device=device)[None] < counts[:, None]        # [P,B]
        self.bin_ids = torch.where(self.valid, order[:, :self.width], torch.zeros_like(order[:, :self.width]))
        self.counts = counts
        self.point_tile = max(1, min(int(element_budget // max(1, p * n)), 1 << 16))
        self.dft_mode = DFT_MODE
        self.dft = None
        if not self.affine and not self.closed_form_endpoint and self.dft_mode == 'bmm':
            m = torch.arange(n, device=device, dtype=torch.float64)
            self.dft = torch.exp((-2j * math.pi / n) * m[None, :, None] * self.bin_ids.double()[:, None, :]) / n
            self.dft = self.dft * self.valid[:, None, :]                                       # [P,n,B]

    def tiles(self, points, *, selected=True):
        if points.ndim != 2 or points.shape[1] != 3 or not torch.isfinite(points).all():
            raise ValueError('native integration points must be finite [N,3]')
        n = len(self.f)
        for start in range(0, len(points), self.point_tile):
            stop = min(start + self.point_tile, len(points))
            distance = torch.linalg.vector_norm(points[start:stop].double()[None] - self.antennas[:, None], dim=-1)
            if bool((distance <= 0).any()):
                raise ValueError('native scatterer coincides with the antenna')
            delta_r = distance - self.r0[:, None]                                            # [P,t]
            if selected and self.affine:
                beta = (2 * math.pi / n) * self.bin_ids.double()                              # [P,B]
                delta = (-4 * math.pi / C) * (self.f[1] - self.f[0]) * delta_r[:, :, None] - beta[:, None, :]
                delta = torch.remainder(delta + math.pi, 2 * math.pi) - math.pi
                phase = (-4 * math.pi / C) * self.f[0] * delta_r[:, :, None] + .5 * (n - 1) * delta
                kernel = (_stable_sinc(n * delta / (2 * math.pi)) / _stable_sinc(delta / (2 * math.pi))
                          * torch.exp(1j * phase) / distance[:, :, None].square())
                kernel = kernel * self.valid[:, None, :]
            elif selected and self.closed_form_endpoint:
                m = n - 1                                                                    # affine prefix length
                beta = (2 * math.pi / n) * self.bin_ids.double()                              # [P,B]
                delta = (-4 * math.pi / C) * (self.f[1] - self.f[0]) * delta_r[:, :, None] - beta[:, None, :]
                delta = torch.remainder(delta + math.pi, 2 * math.pi) - math.pi
                phase = (-4 * math.pi / C) * self.f[0] * delta_r[:, :, None] + .5 * (m - 1) * delta
                prefix = (_stable_sinc(m * delta / (2 * math.pi)) / _stable_sinc(delta / (2 * math.pi))
                          * (m / n) * torch.exp(1j * phase))
                last = torch.exp(1j * ((-4 * math.pi / C) * self.f[-1] * delta_r[:, :, None] - beta[:, None, :] * m)) / n
                kernel = (prefix + last) / distance[:, :, None].square()
                kernel = kernel * self.valid[:, None, :]
            else:
                phase = (-4 * math.pi / C) * delta_r[:, :, None] * self.f[None, None, :]
                kernel = torch.exp(1j * phase) / distance[:, :, None].square()
                if selected and self.dft is not None:
                    kernel = torch.bmm(kernel, self.dft)                                         # [P,t,B]
                elif selected:
                    kernel = torch.fft.fft(kernel, dim=2, norm='forward')
                    kernel = kernel.gather(2, self.bin_ids[:, None, :].expand(-1, stop - start, -1)) * self.valid[:, None, :]
            yield start, stop, kernel

    def render(self, points, field, volume, scale, *, selected=True):
        if field.shape != (len(points),) or torch.is_complex(field) or not torch.isfinite(field).all():
            raise ValueError('native SpINR field must be finite signed real')
        amplitude = field.double() * _volume(points, volume, scale)
        width = self.width if selected else len(self.f)
        prediction = torch.zeros(self.pulses, width, dtype=torch.complex128, device=points.device)
        for start, stop, kernel in self.tiles(points, selected=selected):
            prediction = prediction + (amplitude[start:stop][None, :, None] * kernel).sum(1)
        return prediction

    @torch.no_grad()
    def field_vjp(self, points, volume, scale, cotangent):
        if cotangent.shape != (self.pulses, self.width) or not torch.isfinite(cotangent).all():
            raise ValueError('invalid native selected-bin cotangent')
        weights = _volume(points, volume, scale)
        result = torch.zeros(len(points), dtype=torch.float64, device=points.device)
        for start, stop, kernel in self.tiles(points):
            result[start:stop] = (kernel.conj() * cotangent[:, None, :]).real.sum(dim=(0, 2)) * weights[start:stop]
        return result

    def target_bins(self, responses):
        target = torch.as_tensor(np.array(responses, copy=True), device=self.f.device, dtype=torch.complex128)
        if target.shape != (self.pulses, len(self.f)) or not torch.isfinite(target).all():
            raise ValueError('invalid native responses')
        return torch.fft.fft(target, dim=1, norm='forward').gather(1, self.bin_ids) * self.valid

    def select_bins(self, spectrum):
        """Gather this kernel's selected bins from full ``[pulses, F]`` spectra."""
        return spectrum.gather(1, self.bin_ids) * self.valid


def pulse_bin_objectives(prediction, target, mean_raw_power):
    """Per-pulse ``bin_objective`` values ``[pulses]`` over masked selected bins."""
    if not math.isfinite(mean_raw_power) or mean_raw_power <= 0 or prediction.shape != target.shape:
        raise ValueError('invalid native objective shape or TRAIN normalization')
    return ((prediction.abs() - target.abs()).square() + .5 * (prediction - target).abs().square()).sum(dim=1) / mean_raw_power


def batched_loss_and_field_vjp(kernel, points, field, volume, scale, responses, mean_raw_power, weight):
    """Per-pulse losses and the field VJP of ``weight * sum_p loss_p`` in one pass."""
    with torch.no_grad():
        prediction = kernel.render(points, field, volume, scale)
        target = kernel.target_bins(responses)
    with torch.enable_grad():
        prediction.requires_grad_(True)
        losses = pulse_bin_objectives(prediction, target, mean_raw_power)
        cotangent, = torch.autograd.grad((losses * weight).sum(), prediction)
    if not torch.isfinite(losses).all() or not torch.isfinite(cotangent).all():
        raise FloatingPointError('nonfinite native SpINR loss/gradient')
    return losses.detach(), kernel.field_vjp(points, volume, scale, cotangent.detach())


def shard_groups(plan, ids):
    """Split pulse-plan indices into shard groups (one native frequency vector each), order preserved."""
    groups = {}
    for i in ids:
        groups.setdefault(int(plan.records[int(i)][0]), []).append(int(i))
    return list(groups.values())
