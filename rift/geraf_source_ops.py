"""Native-acquisition bindings for the unmodified released GeRaFStage1.

The model calls these through the original tracer/MF signatures. Only the
acquisition phase is replaced: RIFT's actual bistatic frequency samples, or
GOTCHA's exact monostatic frequencies and published phase reference. Source
specular gates, sum-path spreading, point-count normalization, and antenna-
mean (frequency-sum) MF normalization are retained. See the vendored CUDA
reference and docs/GERAF_V1_HARDENING.md for provenance and precision limits.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch.utils.checkpoint import checkpoint
from rift.config import cc


def bilinear_ray_sampler(normals, sigmas, z_vals, tgt_z_vals):
    """Torch port of BilinearRaySampleForward/BackwardKernel, including bounds.

    Geometry is detached, as in the source CUDA autograd wrapper. There is no
    extrapolation or terminal-cell repair. The denominator floor is upstream's.
    """
    z, target = z_vals.detach().contiguous(), tgt_z_vals.detach().contiguous()
    if z.shape[-1] < 2:
        raise ValueError("Source ray interpolation requires at least two samples")
    lower = (torch.searchsorted(z, target, right=True) - 1).clamp(0, z.shape[-1] - 2)
    z0, z1 = z.gather(1, lower), z.gather(1, lower + 1)
    inside = (target >= z0) & (target <= z1)
    w = (target - z0) / (z1 - z0).clamp_min(1e-6)
    ni = lower[..., None].expand(-1, -1, 3)
    n = normals.gather(1, ni) * (1 - w[..., None]) + normals.gather(1, ni + 1) * w[..., None]
    si = lower[None].expand(sigmas.shape[0], -1, -1)
    s = sigmas.gather(2, si) * (1 - w) + sigmas.gather(2, si + 1) * w
    return torch.where(inside[..., None], n, 0), torch.where(inside, s, 0), inside


class _SourceNormalize(torch.autograd.Function):
    """The CUDA normal Jacobian, including its epsilon convention."""
    @staticmethod
    def forward(ctx, x):
        scale = x.norm(dim=-1, keepdim=True) + 1e-7
        result = x / scale
        ctx.save_for_backward(result, scale)
        return result

    @staticmethod
    def backward(ctx, grad):
        result, scale = ctx.saved_tensors
        return (grad - result * (grad * result).sum(-1, keepdim=True)) / scale


def source_amplitudes(points, normals, sigmas, tx, rx, inbounds, total_points):
    """Released CUDA forward/normal-gradient law; no extra physical factors."""
    incoming = points[None] - tx[:, None]
    returning = rx[:, None] - points[None]
    path = incoming.norm(dim=-1) + returning.norm(dim=-1)
    incoming = incoming / (incoming.norm(dim=-1, keepdim=True) + 1e-7)
    returning = returning / (returning.norm(dim=-1, keepdim=True) + 1e-7)
    cos_incidence = (incoming * normals).sum(-1)
    reflected = _SourceNormalize.apply(incoming - 2 * cos_incidence[..., None] * normals)
    alignment = (returning * reflected).sum(-1)
    active = (alignment >= 1e-6) & (cos_incidence <= 0) & inbounds[None]
    return torch.where(active, sigmas * alignment / path.square(), 0) / total_points


@dataclass
class NativeAcquisition:
    tx: torch.Tensor                       # actual paired [P,3], float64
    rx: torch.Tensor
    frequencies: torch.Tensor              # exact native [F]
    reference_path: torch.Tensor           # [P], zero for RIFT; 2*r0 for GOTCHA
    phase_sign: float = -1.
    point_chunk: int = 1024
    pair_chunk: int = 16
    uniform_rift: bool = False

    def indices(self, tx, rx):
        # Stage1 permutes antenna groups. Match complete paired geometry and
        # reject ambiguous references instead of silently choosing a pulse.
        values = torch.cat((tx, rx), -1)
        known = torch.cat((self.tx, self.rx), -1)
        matches = (values[:, None] == known[None]).all(-1)
        if not bool((matches.sum(-1) == 1).all()):
            raise ValueError("GeRaF antenna geometry must uniquely identify native channels")
        return matches.to(torch.int64).argmax(-1)

    def phase(self, points, ids):
        distance = ((points[None] - self.tx[ids, None]).norm(dim=-1)
                    + (points[None] - self.rx[ids, None]).norm(dim=-1))
        distance = distance - self.reference_path[ids, None]
        return torch.exp((self.phase_sign * 2j * torch.pi / cc)
                         * distance[..., None] * self.frequencies)

    def trace(self, normals, sigmas, points, tx, rx, inbounds):
        ids = self.indices(tx, rx)
        points = points.detach().to(self.tx)
        total = len(points)
        if total == 0:
            raise ValueError("GeRaF loss mask removed every rendering ray")
        if self.uniform_rift:
            # A source bank contains paired channels. Group by actual Tx so
            # the established NUFFT never invents a Cartesian MIMO product.
            from rift.geraf_signal_operator import pairwise_range_forward_operator
            result = []
            order = []
            for t in torch.unique(tx, dim=0):
                sel = (tx == t).all(-1).nonzero().flatten()
                a = source_amplitudes(points, normals, sigmas[sel], tx[sel], rx[sel], inbounds, total)
                value = pairwise_range_forward_operator(
                    self.frequencies, 2 * torch.pi * self.frequencies / cc,
                    t[None], rx[sel], points, a, phase_sign=self.phase_sign,
                    pair_chunk=self.pair_chunk, point_chunk=self.point_chunk)
                result.append(value[..., 0].T)
                order.append(sel)
            return torch.cat(result)[torch.cat(order).argsort()]
        outputs = []
        for start in range(0, len(ids), self.pair_chunk):
            sel = ids[start:start + self.pair_chunk]
            weight = sigmas[start:start + self.pair_chunk]
            value = torch.zeros((len(sel), len(self.frequencies)), dtype=torch.complex128, device=points.device)
            for p in range(0, total, self.point_chunk):
                def block(n, s, xyz, valid, channels):
                    a = source_amplitudes(xyz, n, s, self.tx[channels], self.rx[channels], valid, total)
                    return (a[..., None] * self.phase(xyz, channels)).sum(1)
                args = (normals[p:p+self.point_chunk], weight[:, p:p+self.point_chunk],
                        points[p:p+self.point_chunk], inbounds[p:p+self.point_chunk], sel)
                value = value + (checkpoint(block, *args, use_reentrant=False)
                                 if torch.is_grad_enabled() else block(*args))
            outputs.append(value)
        return torch.cat(outputs)

    def matched_filter(self, response, points, tx=None, rx=None):
        tx, rx = (self.tx, self.rx) if tx is None else (tx, rx)
        ids = self.indices(tx, rx)
        points = points.detach().to(self.tx)
        if self.uniform_rift:
            from rift.geraf_signal_operator import matched_filter_from_response_range
            out = torch.zeros(len(points), dtype=torch.complex128, device=points.device)
            for t in torch.unique(tx, dim=0):
                sel = (tx == t).all(-1).nonzero().flatten()
                out = out + matched_filter_from_response_range(
                    response[sel].T[..., None], self.frequencies,
                    2 * torch.pi * self.frequencies / cc, t[None], rx[sel], points,
                    phase_sign=self.phase_sign, pair_chunk=self.pair_chunk, point_chunk=self.point_chunk)
            return out / len(ids)
        outputs = []
        for p in range(0, len(points), self.point_chunk):
            xyz = points[p:p+self.point_chunk]
            value = torch.zeros(len(xyz), dtype=torch.complex128, device=points.device)
            for start in range(0, len(ids), self.pair_chunk):
                sel = ids[start:start+self.pair_chunk]
                def block(s, q, channels):
                    return (self.phase(q, channels).conj() * s[:, None]).sum((0, 2))
                args = (response[start:start+self.pair_chunk], xyz, sel)
                value = value + (checkpoint(block, *args, use_reentrant=False)
                                 if torch.is_grad_enabled() else block(*args))
            outputs.append(value / len(ids))
        return torch.cat(outputs)


def lensless_ray_tracer(*, normals, sigmas, points, antenna_t, antenna_r,
                       radar_cfg, backend='native', inbounds=None, use_diff=True, nomralization=True):
    if not use_diff or not nomralization:
        raise ValueError("The released GeRaF comparison retains directional physics and normalization")
    if inbounds is None:
        inbounds = torch.ones(len(points), device=points.device, dtype=torch.bool)
    response = radar_cfg['native'].trace(normals, sigmas, points, antenna_t, antenna_r, inbounds)
    return response.real, response.imag


def base_matched_filter(*, real, image, t_pos, r_pos, points, radar_cfg, backend='native'):
    result = radar_cfg['native'].matched_filter(torch.complex(real, image), points, t_pos, r_pos)
    return result.real, result.imag
