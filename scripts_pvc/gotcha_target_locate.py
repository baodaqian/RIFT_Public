#!/usr/bin/env python3
"""Locate a GOTCHA vehicle for a new target box (new-target step 1; CPU, TRAIN rows only, no model).

Starting from an approximate native (x, y) (for example read off Casteel et al. 2007, Fig. 1, good to about 1 m), this
backprojects the measured TRAIN phase histories onto a small horizontal image around it with the Camry data images'
adjoint, I(v) = sum_f d(f) exp(+i 4 pi f/c (|v - a| - r0)) (antenna positions and r0 as the reader returns them, the
published autofocus applied once), evaluated by FFT: each pulse's uniformly spaced bins are inverse-transformed to a
fine range profile (``--fft-size``) and interpolated at every pixel's differential range. The raw shards hold the whole
scene, so every pulse and every native bin is used by default: one pulse per 1-degree sector aliases the scene into a
patch under a metre across, and every 4th bin folds clutter from 25 m away. Each pass is imaged coherently and the
passes are summed incoherently (|I_p|^2), so per-pass phase offsets cannot defocus the result.

Rows: the full-data unit split (stride 1, 10% of pass 4 held out, seed 42, as the F5 arms), TRAIN units only; VALIDATION
and TEST responses are never read. The image plane is native z = ``--z``.

The estimate: inside ``--fit-radius`` of the start, the cells within ``--fit-db`` of the local peak are the car's
bright support; its energy-weighted centroid is the centre and the principal axis of its second moments the heading
(the car's long axis, reported as the angle of +local-x in native x-y, degrees, sign chosen toward native -y as for
the Camry). Length and width are the support's extents along and across that axis.

    python scripts_pvc/gotcha_target_locate.py --name sentra --x 23.0 --y -29.0 --output-dir OUT
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from rift.gotcha_dataset import GOTCHADataset, load_region  # noqa: E402
from rift_pvc.gotcha_training import _target  # noqa: E402
from rift_pvc.gotcha_unit_split import apply_unit_split  # noqa: E402
C = 299_792_458.0

SCHEMA = 'gotcha_target_locate_v1'


def provisional_region(name, x, y, z):
    """A region whose local frame is the native frame shifted to (x, y, z): no rotation, so local = native - origin."""
    return {'schema': 'rift_gotcha_regions_v1', 'regions': {name: dict(
        target_id=name, translation_m=[float(x), float(y), float(z)],
        rotation_local_to_native=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], half_extent_m=6.0,
        placement_provenance='provisional locate frame (gotcha_target_locate.py): native axes, origin at the start '
                             'position; used only to image TRAIN data around the candidate')}}


def fft_backproject(image, centres, antennas, r0, freqs, responses, fft_size, batch=1024):
    """image += sum_m sum_k d_mk exp(+i 4 pi f_k/c (|v - a_m| - r0_m)) for uniformly spaced f_k, by FFT and linear
    interpolation of each pulse's range profile (bin c / (2 df fft_size), unambiguous c / (2 df))."""
    k = np.arange(len(freqs))
    df, f0 = np.polyfit(k, np.asarray(freqs, dtype=np.float64), 1)   # the stored bins are float32 (about 1 kHz steps)
    if np.abs(f0 + df * k - freqs).max() > 5e3:
        raise ValueError('frequencies are not uniformly spaced')
    scale = 2 * df * fft_size / C
    for start in range(0, len(antennas), batch):
        a = antennas[start:start + batch]
        d = responses[start:start + batch]
        profile = torch.fft.ifft(d, n=fft_size, dim=1) * fft_size                     # [B, N]
        dr = torch.cdist(a, centres) - r0[start:start + batch, None]                  # [B, P]
        u = dr * scale
        n0 = torch.floor(u)
        frac = (u - n0).to(profile.dtype)
        i0 = torch.remainder(n0.long(), fft_size)
        i1 = torch.remainder(i0 + 1, fft_size)
        value = torch.gather(profile, 1, i0) * (1 - frac) + torch.gather(profile, 1, i1) * frac
        image += (value * torch.exp(1j * (4 * math.pi * f0 / C) * dr)).sum(0)


def estimate(energy, axis_x, axis_y, radius, fit_db):
    gx, gy = np.meshgrid(axis_x, axis_y, indexing='ij')
    inside = np.hypot(gx, gy) <= radius
    e = np.where(inside, energy, 0.0)
    cut = e.max() * 10 ** (-fit_db / 10)
    w = np.where(e > cut, e, 0.0)
    total = w.sum()
    cx, cy = (w * gx).sum() / total, (w * gy).sum() / total
    dx, dy = gx - cx, gy - cy
    cov = np.array([[(w * dx * dx).sum(), (w * dx * dy).sum()], [(w * dx * dy).sum(), (w * dy * dy).sum()]]) / total
    values, vectors = np.linalg.eigh(cov)
    major = vectors[:, 1]
    if major[1] > 0:                      # +local-x toward native -y, as the Camry frame (heading about -93 deg)
        major = -major
    minor = np.array([-major[1], major[0]])
    support = w > 0
    along = dx[support] * major[0] + dy[support] * major[1]
    across = dx[support] * minor[0] + dy[support] * minor[1]
    return dict(centre_offset_m=[float(cx), float(cy)], heading_deg=float(math.degrees(math.atan2(major[1], major[0]))),
                axis_ratio=float(math.sqrt(values[1] / max(values[0], 1e-12))),
                length_m=float(along.max() - along.min()), width_m=float(across.max() - across.min()),
                support_cells=int(support.sum()), threshold_db_below_peak=fit_db, fit_radius_m=radius)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--name', required=True)
    p.add_argument('--x', type=float, required=True, help='approximate native x (m)')
    p.add_argument('--y', type=float, required=True, help='approximate native y (m)')
    p.add_argument('--z', type=float, default=0.5, help='native height of the image plane (m)')
    p.add_argument('--half-width', type=float, default=6.0)
    p.add_argument('--pixel', type=float, default=0.1)
    p.add_argument('--passes', type=int, nargs='+', default=list(range(1, 9)))
    p.add_argument('--pulse-stride', type=int, default=1, help='every k-th pulse of each TRAIN unit')
    p.add_argument('--fft-size', type=int, default=8192)
    p.add_argument('--fit-radius', type=float, default=3.5)
    p.add_argument('--fit-db', type=float, default=8.0, help='support: cells within this many dB of the local peak')
    p.add_argument('--dataset-root', type=Path, default=Path('/scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined'))
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = f'locate_{args.name}'
    config = args.output_dir / f'{name}_region.json'
    config.write_text(json.dumps(provisional_region(name, args.x, args.y, args.z), indent=2) + '\n')
    ds = GOTCHADataset(args.dataset_root, passes=tuple(args.passes), polarizations=('hh',),
                       region=load_region(name, config), pulses_per_sector=0, frequency_stride=1)
    split = apply_unit_split(ds, stride=1, heldout_pass=4, heldout_fraction=0.1)
    n = int(round(2 * args.half_width / args.pixel))
    axis = (np.arange(n) + 0.5) * args.pixel - args.half_width
    grid = torch.as_tensor(np.stack(np.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1, 2), dtype=torch.float64)
    centres = torch.cat([grid, torch.zeros(len(grid), 1, dtype=torch.float64)], 1)   # local z 0 = native z args.z
    energy = np.zeros(n * n)
    per_pass, started = {}, time.time()
    views = ds.viewpoints('train')
    with torch.no_grad():
        for pass_id in args.passes:
            antennas, r0, responses, freqs, count = [], [], [], None, 0
            for view_pass, sector in views:
                if view_pass != pass_id:
                    continue
                for o in list(ds.observations(view_pass, sector, 'hh'))[::args.pulse_stride]:
                    if freqs is None:
                        freqs = np.asarray(o.frequencies_hz, dtype=np.float64)
                    elif not np.array_equal(freqs, np.asarray(o.frequencies_hz, dtype=np.float64)):
                        raise ValueError('frequency grid changes within a pass')
                    antennas.append(ds.region.to_local(o.position_m))
                    r0.append(float(o.reference_range_m))
                    responses.append(_target(o, 'cpu').to(torch.complex64))
                count += 1
            image = torch.zeros(len(centres), dtype=torch.complex128)
            fft_backproject(image, centres, torch.as_tensor(np.asarray(antennas), dtype=torch.float64),
                            torch.as_tensor(r0, dtype=torch.float64), freqs,
                            torch.stack(responses).to(torch.complex128), args.fft_size)
            e = image.abs().square().numpy()
            energy += e
            per_pass[pass_id] = dict(units=count, pulses=len(responses), peak=float(e.max()))
            print(f'{args.name}: pass {pass_id} {count} TRAIN units, {len(responses)} pulses, '
                  f'{time.time() - started:.0f} s', flush=True)
    energy = energy.reshape(n, n)
    fit = estimate(energy, axis, axis, args.fit_radius, args.fit_db)
    centre = [args.x + fit['centre_offset_m'][0], args.y + fit['centre_offset_m'][1]]
    result = dict(schema=SCHEMA, name=args.name, start_native_xy=[args.x, args.y], image_plane_native_z=args.z,
                  centre_native_xy=centre, **fit, half_width_m=args.half_width, pixel_m=args.pixel,
                  passes=args.passes, per_pass=per_pass, pulse_stride=args.pulse_stride, fft_size=args.fft_size,
                  rows='full unit split TRAIN, all native bins; VALIDATION/TEST never read', split=split['schema'],
                  dataset_identity=ds.identity, seconds=round(time.time() - started))
    np.savez_compressed(args.output_dir / f'{args.name}_locate.npz', energy=energy, axis_m=axis,
                        meta=json.dumps(result))
    (args.output_dir / f'{args.name}_locate.json').write_text(json.dumps(result, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    db = 10 * np.log10(energy / energy.max() + 1e-12)
    figure, ax = plt.subplots(figsize=(5.2, 4.6))
    extent = [args.x - args.half_width, args.x + args.half_width, args.y - args.half_width, args.y + args.half_width]
    shown = ax.imshow(db.T, origin='lower', extent=extent, cmap='inferno', vmin=-30, vmax=0)
    ax.plot(args.x, args.y, '+', color='cyan', ms=12, label='start (figure)')
    ax.plot(*centre, 'x', color='lime', ms=10, label='estimated centre')
    h = math.radians(fit['heading_deg'])
    for sign in (-1, 1):
        ax.plot([centre[0], centre[0] + sign * 2.4 * math.cos(h)], [centre[1], centre[1] + sign * 2.4 * math.sin(h)],
                color='lime', lw=1)
    ax.set_xlabel('native x (m)')
    ax.set_ylabel('native y (m)')
    ax.set_title(f"{args.name}: centre ({centre[0]:.2f}, {centre[1]:.2f}) m, heading {fit['heading_deg']:.1f} deg", fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
    figure.colorbar(shown, ax=ax, label='dB (incoherent over passes)')
    figure.tight_layout()
    figure.savefig(args.output_dir / f'{args.name}_locate.png', dpi=150)
    print(json.dumps({k: result[k] for k in ('name', 'centre_native_xy', 'heading_deg', 'length_m', 'width_m',
                                             'axis_ratio', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
