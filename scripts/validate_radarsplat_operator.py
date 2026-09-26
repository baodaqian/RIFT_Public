#!/usr/bin/env python3
"""Small synthetic MF/renderer diagnostic; no dataset or checkpoint access.

Operator parity is tested before power detection. Image-model discrepancies
are measured, not forced to pass an equivalence assertion. No fitting occurs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from rift.matched_filter_power import matched_filter_complex
from rift.radarsplat_b7873200 import RadarSplatEffects, RadarSplatModel
from rift.radarsplat_b7873200_adapter import native_power_from_matched_filter, target_grid_from_arrays
from rift.radarsplat_fidelity import polar_world_points


def diagnostics():
    pose = np.diag([-1., 1., -1., 1.]); pose[0, 3] = 10.
    arrays = dict(sensor_to_world=pose, range_m=np.linspace(9.845, 10.155, 32),
                  azimuth_rad=np.deg2rad(np.linspace(-1.55, 1.55, 32)),
                  elevation_rad=np.deg2rad(np.array([-.4, 0., .4])))
    points = torch.tensor(polar_world_points(arrays), dtype=torch.float64)
    frequency = 8.5e9 + torch.arange(32, dtype=torch.float64)*(3e9/32)
    tx = torch.tensor([[10., -.012, 0.], [10., .014, 0.]], dtype=torch.float64)
    rx = torch.tensor([[10., 0., -.013], [10., 0., .003], [10., 0., .015]], dtype=torch.float64)
    def response(position):
        length = torch.linalg.vector_norm(tx-position, dim=1)[:, None] + torch.linalg.vector_norm(rx-position, dim=1)[None, :]
        return torch.exp(-2j*torch.pi*length[..., None]*frequency/299792458.)
    def mf(signal, backend):
        return matched_filter_complex(signal, tx, rx, frequency, points, backend=backend,
            response_layout="tx_rx_freq", phase_sign=-1., range_model="none",
            include_four_pi=False, compute_dtype=torch.float64, point_chunk=256, pair_chunk=6)
    def power(amplitude):
        return native_power_from_matched_filter(amplitude, n_elevation=3, n_azimuth=32, n_range=32)
    grid = target_grid_from_arrays(range_m=arrays["range_m"], azimuth_rad=arrays["azimuth_rad"],
        expected_grid=dict(n_range=32, n_azimuth=32, scene_extent_m=.16, azimuth_center_deg=0.,
                          output_azimuth_resolution_deg=.1, intermediate_azimuth_resolution_deg=.1,
                          azimuth_beamwidth_deg=1.8, spectral_leakage_width_m=.2))
    report = {
        "schema": "rift_radarsplat_operator_diagnostic_v1",
        "synthetic_acquisition": {
            "frequency_hz": frequency.tolist(), "tx_m": tx.tolist(), "rx_m": rx.tolist(),
            "range_m": arrays["range_m"].tolist(),
            "azimuth_rad": arrays["azimuth_rad"].tolist(),
            "elevation_rad": arrays["elevation_rad"].tolist(),
            "phase_sign": -1., "range_model": "none", "gaussian_scale_m": .003,
            "gradient_objective": "mean(abs(complex_MF)**2), differentiated wrt complex observations",
        },
        "single_reflectors": [],
    }
    for position in ([0., 0., 0.], [.035, .018, .021]):
        source = torch.tensor(position, dtype=torch.float64)
        signal = response(source)
        direct, fast = mf(signal, "direct"), mf(signal, "range_nufft")
        relative = float(torch.linalg.vector_norm(fast-direct)/torch.linalg.vector_norm(direct))
        if relative > 1e-6:
            raise AssertionError(f"production MF/direct parity failed: {relative}")
        # Compare complex observation gradients through the same MF-energy loss.
        differentiable = signal.detach().requires_grad_(True)
        direct_loss = mf(differentiable, "direct").abs().square().mean()
        direct_grad, = torch.autograd.grad(direct_loss, differentiable)
        fast_loss = mf(differentiable, "range_nufft").abs().square().mean()
        fast_grad, = torch.autograd.grad(fast_loss, differentiable)
        grad_error = float(torch.linalg.vector_norm(fast_grad-direct_grad)/torch.linalg.vector_norm(direct_grad))
        if grad_error > 1e-6:
            raise AssertionError(f"production MF gradient parity failed: {grad_error}")
        target = power(direct)
        model = RadarSplatModel(source.float()[None], initial_scale=.003, initial_opacity=.9,
                               initial_noise_probability=.01, initial_reflectance=torch.tensor([.9]), sh_degree=0)
        rendered = model.render(torch.tensor(pose, dtype=torch.float32), grid,
                                RadarSplatEffects.b787_clean())["final_power"][0].double().detach()
        gain = (target*rendered).sum()/rendered.square().sum().clamp_min(1e-30)
        shape_error = float(((gain*rendered-target).square().sum()/target.square().sum()).sqrt())
        centre = lambda p: [float((p.sum(0)*torch.tensor(arrays["range_m"])).sum()/p.sum()),
                            float((p.sum(1)*torch.tensor(arrays["azimuth_rad"])).sum()/p.sum())]
        report["single_reflectors"].append(dict(position_m=position, complex_mf_relative_l2=relative,
            complex_gradient_relative_l2=grad_error, best_scalar_power_shape_relative_l2=shape_error,
            target_centroid_range_azimuth=centre(target), renderer_centroid_range_azimuth=centre(rendered),
            elevation_sum_to_middle_slice_power=float(target.sum()/direct.reshape(3, 32, 32)[1].abs().square().sum())))
    single = response(torch.zeros(3, dtype=torch.float64))
    reference = power(mf(single, "direct")).sum()
    report["coincident_equal_reflectors"] = {
        "in_phase_power_ratio": float(power(mf(single+single, "direct")).sum()/reference),
        "opposite_phase_power_ratio": float(power(mf(single-single, "direct")).sum()/reference),
        "additive_independent_power_ratio": 2.,
    }
    report["interpretation"] = ("MF implementation parity is numerical; Gaussian power is an acquisition adaptation. "
                                "A scalar gain cannot remove PSF/elevation/interference differences.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(diagnostics(), indent=2, allow_nan=False)
    if args.output:
        with args.output.open("x") as handle:
            handle.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
