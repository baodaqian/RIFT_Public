#!/usr/bin/env python
"""Linear capacity check of the shared sonar shell renderer (docs/RIFT_SAS_Train.md, B2/B3).

Question: can a static complex field on a regular grid, rendered by the
unchanged ``rift.sas_operator.render_sas_bins``, express the carrier-bearing
AirSAS measurements of a small TRAIN block, and at which grid spacing? No
training recipe is exercised; this measures expressiveness only. The renderer's
best case is used (below), so a pass is necessary for R1, not sufficient.

Linearization: ``lambertian_ratio 1`` (the integrand's Lambertian factor is
exactly 1) and ``opacity_scale 0`` (``alpha = exp(0) = 1``; ``1 + 1e-10`` is 1 in
fp32, so T = 1 and density receives no gradient). The field is
``RIFTSASRectangularGrid`` (endpoint lattice, trilinear, complex SH up to
``--degree``) in ``ComplexSHSonarField``, zero at the CGLS start. With
``--skip-normals`` (default) a wrapper returns zero normals instead of the
finite-difference ones, which carry zero weight at ratio 1; the pre-flight
asserts that both paths render the same prediction. Every other render
argument is the production one (30 degree beam, Rx-to-point SH direction,
signal scale 10, all 326 bins); ``--num-rays`` defaults to the recipe's 4,900.

Block (B3): fit TRAIN rings ``--fit-rings`` (default 4 6) at every
``--azimuth-stride``-th azimuth and hold out ``--heldout-ring`` (default 5, TRAIN)
at the same azimuths; its reference is the average of its two neighbours,
printed at g = 1 and at the TRAIN-analog gain (``--reference-gain-re/-im``;
defaults from ``rift_sas_references.py``). CGLS on the real vector (w_re, w_im);
``A p`` is a no-grad render, ``A^T r`` the autograd gradient of Re<r, A x>.

``--field adaptive`` (B7 ask 2) replaces the rectangular grid by the adaptive
model's own field: ``AdaptivePointSHScene.from_regular_grid(points)`` (anchors
at cell midpoints, the trainer's construction, degree ``--degree``, zero
coefficients, capacity = points^3) splatted by ``AdaptiveRIFTSASField`` onto
its ``raster^3`` endpoint lattice. Positions are frozen (``delta_raw`` excluded
and without gradient), so the map is linear in (w_re, w_im). ``--points``
pairs one point granularity with each ``--grids`` entry (the raster).

Pre-flight per grid: dot-product adjoint test, linearity test, normals-path
equality, finite adjoint. Per iteration: in-sample and held-out rel-MSE, full
band and per band (FFT over the bins), at g = 1 (the least-squares field
carries the scale) and at the best global g fitted on the fitted block.
Verdict (B3, three-valued per B4): EXPRESSIVE if the best in-sample
12.5-27.5 kHz rel-MSE is <= 0.5; NOT EXPRESSIVE only if the iteration cap was
reached or that value fell by less than 5 % (relative) over the last 5
iterations; UNDECIDED otherwise (for instance a run cut by the time budget while
still falling). The held-out value is reported, never gated. The held-out ring
must be TRAIN (VAL only with ``--allow-val-heldout``); reserved-test rows are
never read (asserted). The JSON report is rewritten after every iteration.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rift.rift_sas import Y00, AdaptiveRIFTSASField, ComplexSHSonarField, RIFTSASRectangularGrid  # noqa: E402
from rift.sas_dataset import load_sas_cache  # noqa: E402
from rift.sas_operator import render_sas_bins  # noqa: E402
from rift.sh_sas import real_sh_basis_for_directions  # noqa: E402
from rift.sparse_scene import AdaptivePointSHScene  # noqa: E402
from scripts_pvc_sas.rift_sas_references import band_masks  # noqa: E402

SIGNAL_SCALE = 10.0
BEAMWIDTH_DEG = 30.0
IN_BAND = "12.5..27.5"
EXPRESSIVE_THRESHOLD = 0.5
PLATEAU_WINDOW = 5
PLATEAU_RELATIVE_DROP = 0.05


def grid_shape(granularity, extent_xyz):
    longest = float(max(extent_xyz))
    return tuple(max(2, int(round(granularity * float(e) / longest))) for e in extent_xyz)


class NormalsFreeField:
    """``ComplexSHSonarField.query_sas`` without the finite-difference normals.

    At ``lambertian_ratio 1`` the renderer multiplies the normals by zero; this
    wrapper returns the identical coefficients, scatterer and density and zero
    normals, avoiding six extra DC queries per point.
    """

    def __init__(self, field: ComplexSHSonarField):
        self.field = field

    def query_density(self, physical_points):
        return self.field.query_density(physical_points)

    def query_sas(self, physical_points, directions, *, normal_step, probe_next_band=False):
        del normal_step, probe_next_band
        coefficients = self.field.query_coefficients(physical_points)
        basis = real_sh_basis_for_directions(directions, self.field.sh_degree)
        scatterer = (coefficients * basis.to(coefficients.dtype)).sum(dim=-1)
        return {
            "coefficients": coefficients,
            "scatterer": scatterer,
            "density": coefficients[..., 0].abs() * Y00,
            "normals": torch.zeros_like(physical_points),
        }


class LinearShellOperator:
    def __init__(self, cache, device, shape, degree, num_rays, skip_normals, points=None):
        corners = torch.as_tensor(cache.corners, dtype=torch.float32)
        if points is None:
            coefficients = RIFTSASRectangularGrid(shape, 1.0, device, max_degree=degree, init_scale=0.0)
            self.params = [coefficients.w_re, coefficients.w_im]
        else:
            scene = AdaptivePointSHScene.from_regular_grid(
                int(points), 1.0, device, max_degree=degree, init_degree=degree, init_scale=0.0,
                capacity=int(points) ** 3, compact_sh_eval=True,
            )
            scene.delta_raw.requires_grad_(False)
            coefficients = AdaptiveRIFTSASField(scene, raster_granularity=int(shape[0]), extent=1.0)
            self.params = [scene.w_re, scene.w_im]
        self.full_field = ComplexSHSonarField(coefficients, corners.amin(0), corners.amax(0), degree).to(device)
        self.field = NormalsFreeField(self.full_field) if skip_normals else self.full_field
        self.num_rays = int(num_rays)
        self.radii = torch.as_tensor(cache.radii, dtype=torch.float32, device=device)
        self.corners = corners.to(device)
        self.tx = torch.as_tensor(cache.tx_coords, dtype=torch.float32, device=device)
        self.rx = torch.as_tensor(cache.rx_coords, dtype=torch.float32, device=device)

    def render(self, ping, field=None):
        prediction, _aux = render_sas_bins(
            field or self.field, self.radii, self.tx[ping], self.rx[ping], self.corners,
            num_rays=self.num_rays, opacity_scale=0.0, lambertian_ratio=1.0, normal_step=0.0032,
            tx_direction=None, beamwidth_deg=BEAMWIDTH_DEG, point_at_center=True,
            transmit_from_tx=True, output_bin_indices=None, mean_normalize_opacity=False,
            sh_direction="rx_to_point",
        )
        return prediction * SIGNAL_SCALE

    def set(self, vector):
        with torch.no_grad():
            for param, value in zip(self.params, vector):
                param.copy_(value)

    def forward(self, vector, pings, field=None):
        self.set(vector)
        with torch.no_grad():
            return torch.stack([self.render(int(p), field) for p in pings])

    def adjoint(self, residual, pings):
        for param in self.params:
            param.grad = None
        for index, ping in enumerate(pings):
            (torch.conj(residual[index]) * self.render(int(ping))).real.sum().backward()
        grads = [param.grad.detach().clone() for param in self.params]
        if not all(bool(torch.isfinite(g).all()) for g in grads):
            raise FloatingPointError("non-finite adjoint")
        return grads


def dot(a, b):
    return sum(float((x.double() * y.double()).sum()) for x, y in zip(a, b))


def spectra(values):
    return np.fft.fft(np.asarray(values, dtype=np.complex128), axis=1)


def band_rel(target_spectrum, prediction_spectrum, masks, gain=None):
    """rel-MSE by band at g (None = best global g fitted on these rows, full band)."""
    y, p = target_spectrum, prediction_spectrum
    if gain is None:
        gain = complex(np.sum(np.conj(p) * y) / max(float(np.sum(np.abs(p) ** 2)), 1e-30))
    total = float(np.sum(np.abs(y) ** 2))
    out = {"gain": [gain.real, gain.imag]}
    for band, mask in {"full": slice(None), **masks}.items():
        yb, pb = y[:, mask], p[:, mask]
        yy = float(np.sum(np.abs(yb) ** 2))
        pp = float(np.sum(np.abs(pb) ** 2))
        py = complex(np.sum(np.conj(pb) * yb))
        out[band] = {
            "target_energy_fraction": yy / total,
            "rel_mse": float(np.sum(np.abs(gain * pb - yb) ** 2)) / yy,
            "single_gain_coherence": abs(py) / max(np.sqrt(pp * yy), 1e-30),
        }
    return out


def verdict(history, iteration_cap):
    values = [row["in_sample_best_g"][IN_BAND]["rel_mse"] for row in history]
    if not values:
        return "UNDECIDED"
    if min(values) <= EXPRESSIVE_THRESHOLD:
        return "EXPRESSIVE"
    if len(values) >= iteration_cap:
        return "NOT EXPRESSIVE"
    if len(values) > PLATEAU_WINDOW:
        before, after = values[-PLATEAU_WINDOW - 1], values[-1]
        if (before - after) / before < PLATEAU_RELATIVE_DROP:
            return "NOT EXPRESSIVE"
    return "UNDECIDED"


def fmt(rel):
    return " ".join(f"{band}:{value['rel_mse']:.3f}" for band, value in rel.items() if band != "gain")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--grids", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--field", choices=("grid", "adaptive"), default="grid")
    parser.add_argument("--points", type=int, nargs="+", default=None,
                        help="adaptive only: point granularity paired with each --grids raster")
    parser.add_argument("--degree", type=int, default=0)
    parser.add_argument("--num-rays", type=int, default=4900)
    parser.add_argument("--fit-rings", type=int, nargs="+", default=[4, 6])
    parser.add_argument("--heldout-ring", type=int, default=5)
    parser.add_argument("--allow-val-heldout", action="store_true", help="permit a VAL ring as the held-out ring")
    parser.add_argument("--azimuth-stride", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--time-budget", type=float, default=0.0, help="seconds per grid; 0 = none")
    parser.add_argument("--skip-normals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reference-gain-re", type=float, default=0.6867503810789376)
    parser.add_argument("--reference-gain-im", type=float, default=-0.015011645747122705)
    parser.add_argument("--json", required=True)
    options = parser.parse_args(argv)
    device = torch.device(options.device)
    cache = load_sas_cache(options.cache)
    ring_size = int(cache.manifest["ring_size"])
    train_rows = set(cache.train_indices.tolist())
    test_rows = set(cache.test_indices.tolist())
    azimuths = np.arange(0, ring_size, options.azimuth_stride)
    pings = np.concatenate([r * ring_size + azimuths for r in options.fit_rings])
    heldout = options.heldout_ring * ring_size + azimuths
    if not train_rows.issuperset(pings.tolist()):
        raise ValueError("the fitted block must contain TRAIN pings only")
    if test_rows.intersection(heldout.tolist()):
        raise AssertionError("the held-out ring holds reserved-test rows")
    if train_rows.issuperset(heldout.tolist()):
        heldout_role = "train"
    elif options.allow_val_heldout and set(cache.validation_indices.tolist()).issuperset(heldout.tolist()):
        heldout_role = "validation"
    else:
        raise ValueError("the held-out ring must be TRAIN (VAL only with --allow-val-heldout)")
    masks = band_masks(cache.num_bins, float(cache.manifest["sample_rate_hz"]))
    target_np = np.asarray(cache.weights[pings]).astype(np.complex64)
    target = torch.as_tensor(target_np, device=device)
    held_np = np.asarray(cache.weights[heldout]).astype(np.complex64)
    y_fit, y_held = spectra(target_np), spectra(held_np)
    y_norm = float((target.abs() ** 2).sum())

    reference = None
    neighbours = (options.heldout_ring - 1, options.heldout_ring + 1)
    if all(r in options.fit_rings for r in neighbours):
        rows = [options.fit_rings.index(r) for r in neighbours]
        n = azimuths.size
        average = 0.5 * (target_np[rows[0] * n:(rows[0] + 1) * n] + target_np[rows[1] * n:(rows[1] + 1) * n])
        g_ref = complex(options.reference_gain_re, options.reference_gain_im)
        reference = {
            "predictor": f"average of rings {neighbours[0]} and {neighbours[1]} at the same azimuths",
            "g1": band_rel(y_held, spectra(average), masks, gain=1.0 + 0j),
            "g_train_analog": band_rel(y_held, spectra(average), masks, gain=g_ref),
        }
        print(f"held-out ring {options.heldout_ring} reference g=1: {fmt(reference['g1'])}", flush=True)
        print(f"held-out ring {options.heldout_ring} reference g_train_analog {g_ref:.4f}: {fmt(reference['g_train_analog'])}", flush=True)

    extent = np.asarray(cache.corners).max(0) - np.asarray(cache.corners).min(0)
    report = {
        "cache": str(options.cache), "field": options.field, "fit_rings": options.fit_rings, "heldout_ring": options.heldout_ring,
        "heldout_role": heldout_role,
        "azimuth_stride": options.azimuth_stride, "fit_pings": int(pings.size), "bins": int(cache.num_bins),
        "M_complex_data": int(pings.size * cache.num_bins), "degree": options.degree, "num_rays": options.num_rays,
        "opacity_scale": 0.0, "lambertian_ratio": 1.0, "skip_normals": bool(options.skip_normals),
        "iteration_cap": options.iterations, "time_budget_seconds": options.time_budget,
        "expressive_rule": (f"EXPRESSIVE: best in-sample {IN_BAND} kHz rel-MSE at best global g <= {EXPRESSIVE_THRESHOLD}; "
                            f"NOT EXPRESSIVE: iteration cap reached or < {PLATEAU_RELATIVE_DROP:.0%} relative drop over the last "
                            f"{PLATEAU_WINDOW} iterations; else UNDECIDED"),
        "heldout_reference": reference, "grids": {},
    }
    print("config " + json.dumps({k: v for k, v in report.items() if k not in ("grids", "heldout_reference")}), flush=True)

    if options.field == "adaptive":
        if options.points is None or len(options.points) != len(options.grids):
            raise ValueError("--field adaptive needs one --points value per --grids raster")
        configs = list(zip(options.points, options.grids))
    else:
        if options.points is not None:
            raise ValueError("--points applies to --field adaptive only")
        configs = [(None, g) for g in options.grids]
    for points, granularity in configs:
        # the adaptive raster is cubic over the normalized box; the grid is aspect-shaped
        shape = (granularity,) * 3 if points is not None else grid_shape(granularity, extent)
        label = str(granularity) if points is None else f"points{points}_raster{granularity}"
        started = time.time()
        if device.type == "xpu":
            torch.xpu.reset_peak_memory_stats(device)
        op = LinearShellOperator(cache, device, shape, options.degree, options.num_rays, options.skip_normals, points)
        unknowns = sum(p.numel() for p in op.params)

        # Pre-flight on two pings: adjoint, linearity, normals-path equality.
        generator = torch.Generator().manual_seed(0)
        u = [torch.randn(p.shape, generator=generator).to(device) for p in op.params]
        w = [torch.randn(p.shape, generator=generator).to(device) for p in op.params]
        pair = pings[:2]
        v = torch.complex(torch.randn(2, cache.num_bins, generator=generator),
                          torch.randn(2, cache.num_bins, generator=generator)).to(device)
        au = op.forward(u, pair)
        adjoint_error = abs(float((torch.conj(v) * au).real.double().sum()) - dot(u, op.adjoint(v, pair)))
        adjoint_error /= max(abs(float((torch.conj(v) * au).real.double().sum())), 1e-30)
        aw = op.forward(w, pair)
        combined = op.forward([2.0 * a + 3.0 * b for a, b in zip(u, w)], pair)
        linearity_error = float((combined - (2.0 * au + 3.0 * aw)).abs().norm() / combined.abs().norm())
        normals_error = 0.0
        if options.skip_normals:
            full = op.forward(u, pair, field=op.full_field)
            normals_error = float((full - au).abs().norm() / full.abs().norm())
        preflight = {"adjoint_dot_rel_error": adjoint_error, "linearity_rel_error": linearity_error,
                     "normals_free_vs_full_rel_error": normals_error}
        print(f"grid {label} {tuple(shape)} K={unknowns} real unknowns M={pings.size * cache.num_bins} complex data; "
              f"preflight {json.dumps(preflight)}", flush=True)
        if adjoint_error > 1e-4 or linearity_error > 1e-4 or normals_error > 1e-5:
            raise AssertionError(f"pre-flight failed: {preflight}")

        history = []
        entry = {
            "field": options.field,
            "points_granularity": points,
            "point_pitch_m": None if points is None else (extent / points).tolist(),
            "grid_shape": list(shape),
            "spacing_m": (extent / (np.asarray(shape) - 1)).tolist(),
            "K_real_unknowns": int(unknowns),
            "preflight": preflight,
            "history": history,
        }
        report["grids"][label] = entry

        def update_entry():
            values = [r["in_sample_best_g"][IN_BAND]["rel_mse"] for r in history]
            entry["iterations_run"] = len(history)
            entry["seconds_per_iteration"] = float(np.mean([r["seconds"] for r in history])) if history else None
            entry["best_in_sample_in_band_rel_mse"] = min(values) if values else None
            entry["verdict"] = verdict(history, options.iterations)
            entry["elapsed_seconds"] = time.time() - started
            if device.type == "xpu":
                entry["peak_memory_bytes"] = int(torch.xpu.max_memory_allocated(device))
            temporary = Path(options.json).with_name(Path(options.json).name + ".tmp")
            temporary.write_text(json.dumps(report, indent=2))
            os.replace(temporary, options.json)

        update_entry()
        x = [torch.zeros_like(p) for p in op.params]
        residual = target.clone()
        prediction = torch.zeros_like(target)
        s = op.adjoint(residual, pings)
        direction = [value.clone() for value in s]
        gamma = dot(s, s)
        for iteration in range(1, options.iterations + 1):
            tick = time.time()
            q = op.forward(direction, pings)
            q_norm = float((q.abs() ** 2).sum())
            if q_norm <= 0.0 or gamma <= 0.0:
                break
            alpha = gamma / q_norm
            x = [xi + alpha * di for xi, di in zip(x, direction)]
            residual = residual - alpha * q
            prediction = prediction + alpha * q
            s = op.adjoint(residual, pings)
            gamma_new = dot(s, s)
            direction = [si + (gamma_new / gamma) * di for si, di in zip(s, direction)]
            gamma = gamma_new
            p_fit = spectra(prediction.cpu().numpy())
            p_held = spectra(op.forward(x, heldout).cpu().numpy())
            in_sample = band_rel(y_fit, p_fit, masks)
            g_block = complex(*in_sample["gain"])
            row = {
                "iteration": iteration,
                "cgls_rel_mse": float((residual.abs() ** 2).sum()) / y_norm,
                "in_sample_best_g": in_sample,
                "heldout_g1": band_rel(y_held, p_held, masks, gain=1.0 + 0j),
                "heldout_g_block": band_rel(y_held, p_held, masks, gain=g_block),
                "seconds": time.time() - tick,
            }
            history.append(row)
            update_entry()
            print(f"grid {label} it {iteration} ({row['seconds']:.0f}s) in-sample@best-g {fmt(in_sample)} | "
                  f"held-out@g1 {fmt(row['heldout_g1'])}", flush=True)
            if options.time_budget and time.time() - started > options.time_budget:
                print(f"grid {label}: time budget reached after {iteration} iterations", flush=True)
                break
        update_entry()
        best = entry["best_in_sample_in_band_rel_mse"]
        print(f"grid {label}: best in-sample {IN_BAND} kHz rel-MSE "
              f"{float('nan') if best is None else best:.4f} -> {entry['verdict']} "
              f"(iterations_run {entry['iterations_run']} of {options.iterations}, {entry['elapsed_seconds']:.0f}s)", flush=True)
        del op
    return 0


if __name__ == "__main__":
    sys.exit(main())
