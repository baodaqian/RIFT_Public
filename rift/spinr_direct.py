"""SpINR §3.3: closed-form synthesis of selected finite-DFT bins.

This transforms the exact native frequency-pulse exponential analytically;
it does not add an FMCW leakage correction to the acquisition operator.
Physical volumes, product spreading and carrier phase each enter once.
"""
from __future__ import annotations

import math
import torch

from rift.config import cc


def _stable_sinc(x):
    # torch.sinc's derivative can cancel near zero. The even Taylor branch
    # retains the same removable-singularity limit and its coordinate gradient.
    squared = (math.pi*x).square()
    polynomial = 1-squared/6+squared.square()/120-squared.pow(3)/5040
    return torch.where(x.abs() < 1e-3, polynomial, torch.sinc(x))


def _kernel_tiles(frequencies_hz, rx_pos_m, tx_pos_m, points_m, mask, *,
                  point_tile, pair_tile, phase_sign):
    if min(point_tile, pair_tile) < 1 or phase_sign not in (-1., 1.):
        raise ValueError("invalid direct-bin tiles or phase sign")
    f = torch.as_tensor(frequencies_hz, device=points_m.device, dtype=torch.float64)
    rx = torch.as_tensor(rx_pos_m, device=f.device, dtype=torch.float64)
    tx = torch.as_tensor(tx_pos_m, device=f.device, dtype=torch.float64)
    points = points_m.to(torch.float64)
    if (f.ndim != 1 or len(f) < 2 or not torch.isfinite(f).all()
            or points.ndim != 2 or points.shape[1] != 3 or not torch.isfinite(points).all()
            or rx.ndim != 2 or rx.shape[1] != 3 or tx.ndim != 2 or tx.shape[1] != 3
            or not len(rx) or not len(tx) or not torch.isfinite(rx).all() or not torch.isfinite(tx).all()):
        raise ValueError("direct bins require finite frequency/point/antenna arrays")
    df = f[1]-f[0]
    if df <= 0 or not torch.allclose(f[1:]-f[:-1], df.expand(len(f)-1), rtol=1e-10, atol=1e-6):
        raise ValueError("direct DFT bins require uniform increasing frequencies")
    if mask.shape != (len(f), len(rx), len(tx)) or mask.dtype != torch.bool:
        raise ValueError("direct-bin mask must be boolean [F,Rx,Tx]")
    pairs = len(rx)*len(tx)
    mask = mask.to(f.device).reshape(len(f), pairs)
    for pair_start in range(0, pairs, pair_tile):
        pair_ids = torch.arange(pair_start, min(pair_start+pair_tile, pairs), device=f.device)
        selected = mask[:, pair_ids].nonzero()  # [entry, (bin, local pair)]
        if not len(selected):
            continue
        bins, local_pairs = selected.unbind(dim=1)
        output_indices = bins*pairs+pair_ids[local_pairs]
        # Bound temporary kernel memory independently of scene size/bin count.
        tile = min(point_tile, max(1, 1_048_576//len(selected)))
        for start in range(0, len(points), tile):
            stop = min(start+tile, len(points))
            p = points[start:stop]
            rr = torch.linalg.vector_norm(p[:, None]-rx[pair_ids//len(tx)][None], dim=-1)
            rt = torch.linalg.vector_norm(p[:, None]-tx[pair_ids % len(tx)][None], dim=-1)
            if bool(((rr <= 0) | (rt <= 0)).any()):
                raise ValueError("scatterer coincides with an antenna")
            distance = (rr+rt)[:, local_pairs]
            # Multiplying integer bins by a Python float would otherwise create
            # an FP32 tensor even though the propagation path is FP64.
            beta = (2*math.pi/len(f))*bins.to(torch.float64)
            delta = phase_sign*2*math.pi*df*distance/cc-beta[None]
            delta = torch.remainder(delta+math.pi, 2*math.pi)-math.pi
            # sinc ratio is finite and differentiable at an exact DFT bin.
            envelope = _stable_sinc(len(f)*delta/(2*math.pi))/_stable_sinc(delta/(2*math.pi))
            phase = phase_sign*2*math.pi*f[0]*distance/cc+.5*(len(f)-1)*delta
            kernel = envelope*torch.exp(1j*phase)/(rr*rt)[:, local_pairs]
            yield start, stop, output_indices, kernel


def _volume(points, cell_volume_m3, initial_output_scale):
    volume = torch.as_tensor(cell_volume_m3, device=points.device, dtype=torch.float64)
    if (volume.ndim > 1 or (volume.ndim == 1 and volume.shape != (len(points),))
            or not torch.isfinite(volume).all() or bool((volume <= 0).any())
            or not math.isfinite(initial_output_scale) or initial_output_scale <= 0):
        raise ValueError("direct bins require positive physical volumes and fixed scale")
    return volume.expand(len(points))*initial_output_scale


def render_selected_bins(*, frequencies_hz, rx_pos_m, tx_pos_m, points_m, field,
                         cell_volume_m3, initial_output_scale, mask,
                         point_tile=65536, pair_tile=16, phase_sign=-1.):
    """Return [F,Rx,Tx], zero outside mask; prediction never passes through FFT.

    Autograd is supported for small references. Production uses no-grad forward
    plus the exact real-field adjoint below, followed by tiled neural replay.
    """
    if field.shape != (len(points_m),) or torch.is_complex(field) or not torch.isfinite(field).all():
        raise ValueError("direct-bin field must be finite signed real [points]")
    amplitude = field.to(torch.float64)*_volume(points_m, cell_volume_m3, initial_output_scale)
    result = torch.zeros(mask.numel(), device=points_m.device, dtype=torch.complex128)
    for start, stop, indices, kernel in _kernel_tiles(
            frequencies_hz, rx_pos_m, tx_pos_m, points_m, mask,
            point_tile=point_tile, pair_tile=pair_tile, phase_sign=phase_sign):
        result = result.index_add(0, indices, (amplitude[start:stop, None]*kernel).sum(dim=0))
    return result.reshape(mask.shape)


@torch.no_grad()
def selected_bin_field_vjp(*, bin_cotangent, frequencies_hz, rx_pos_m, tx_pos_m, points_m,
                           cell_volume_m3, initial_output_scale, mask,
                           point_tile=65536, pair_tile=16, phase_sign=-1.):
    """Exact Re(A^H g) for the signed real field, including scale and volume."""
    if bin_cotangent.shape != mask.shape or not torch.isfinite(bin_cotangent).all():
        raise ValueError("invalid direct-bin cotangent")
    volume = _volume(points_m, cell_volume_m3, initial_output_scale)
    result = torch.zeros(len(points_m), device=points_m.device, dtype=torch.float64)
    cotangent = bin_cotangent.to(device=points_m.device, dtype=torch.complex128).reshape(-1)
    for start, stop, indices, kernel in _kernel_tiles(
            frequencies_hz, rx_pos_m, tx_pos_m, points_m, mask,
            point_tile=point_tile, pair_tile=pair_tile, phase_sign=phase_sign):
        result[start:stop].add_((kernel.conj()*cotangent[indices][None]).real.sum(dim=1)*volume[start:stop])
    return result


def selected_bin_objective(predicted_bins, observed_frequency, mask, *, training_mean_raw_power):
    """Paper magnitude + 0.5 complex squared error, summed over selected bins."""
    if (predicted_bins.shape != observed_frequency.shape or mask.shape != predicted_bins.shape
            or mask.dtype != torch.bool or predicted_bins.ndim != 3
            or not math.isfinite(training_mean_raw_power) or training_mean_raw_power <= 0):
        raise ValueError("invalid selected-bin objective inputs")
    target = torch.fft.fft(observed_frequency, dim=0, norm="forward")
    p, y = predicted_bins[mask], target[mask]
    return ((p.abs()-y.abs()).square()+.5*(p-y).abs().square()).sum() / (
        predicted_bins.shape[1]*predicted_bins.shape[2]*training_mean_raw_power)
