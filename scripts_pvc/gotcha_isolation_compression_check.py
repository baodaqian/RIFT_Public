#!/usr/bin/env python3
"""Filter-then-decimate qualification and data acceptance for GOTCHA box isolation (B3/B4, CPU).

Companion of ``scripts_pvc/gotcha_isolation_qualify.py`` (rift-4b), reusing ``rift_pvc.gotcha_isolation``.
For TRAIN sectors spread in azimuth, all native pulses and the native 424 bins, it builds the slant-plane
footprint projector ``Q Q^H`` of a box with separate range and cross-range guards and reports:

1. **Decimation equivalence** (review item B3). The filtered data ``Q Q^H y`` and every in-box prediction are
   footprint-limited, so the residual between them is band-limited in pulse angle and window-limited in delay.
   For synthetic in-box scenes (random points, random complex amplitudes) plus white noise, the residual
   ``r = Q Q^H (A x1 + n) - A x2`` is decimated to every ``k``-th pulse and every ``s``-th frequency bin and the
   ratio ``k s ||r_dec||^2 / ||r||^2`` is reported over all decimation offsets. Ratios near 1 with small spread
   mean a trainer may use the decimated filtered rows with its plain loss and no projection at training time.
   The raw-data residual ``(A x1 + n) - A x2`` is reported alongside as the contrast case.

2. **Data acceptance** (review item B4). The sector's real TRAIN responses are filtered and back-projected,
   with the raw responses, onto the horizontal plane through the box centre over a strip of +-10 m along the
   look direction by +-75 m across it; the box footprint's share of the strip image energy is reported for
   both. The filtered ``+-8 m`` square image is also saved for inspection.

Geometry-only except for item 2, which reads TRAIN responses. TEST stays sealed. Proposes nothing on its own.

    source .local-setup/activate-pvc.sh
    python scripts_pvc/gotcha_isolation_compression_check.py --box-centre X Y Z --box-heading DEG \
        --box-half HX HY HZ --out-dir DIR
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rift.gotcha_dataset import GOTCHADataset, load_region  # noqa: E402
from rift_pvc.gotcha_isolation import (Box, SectorGeometry, basis, footprint_grid, retained_fraction,  # noqa: E402
                                       truncate)


def footprint_grid_guarded(box, geometry, *, range_step, cross_step, guard_range, guard_cross, shape='hull'):
    """rift-4b's footprint: the projected corners' convex hull grown by the guards, or its bounding rectangle."""
    grid = footprint_grid(box, geometry, range_step=range_step, cross_step=cross_step,
                          range_guard=guard_range, cross_guard=guard_cross, shape=shape)
    return grid, (len(grid),)


def decimation_ratios(residual, pulses, bins, ks, strides):
    """k*s*||r[o::k, p::s]||^2 / ||r||^2 over all offsets (o, p); residual is [pulses*bins]."""
    r = residual.reshape(pulses, bins)
    total = float((r.abs() ** 2).sum())
    out = {}
    for k in ks:
        for s in strides:
            ratios = [k * s * float((r[o::k, p::s].abs() ** 2).sum()) / total for o in range(k) for p in range(s)]
            out[f'k{k}_s{s}'] = dict(mean=float(np.mean(ratios)), min=float(np.min(ratios)), max=float(np.max(ratios)),
                                     samples=int(math.ceil(pulses / k) * math.ceil(bins / s)))
    return out


def backproject(geometry, y, points, chunk=512):
    """Adjoint of the native kernel on local ``points`` for one sector: |image|^2 normalized by the pulse count."""
    y = y.reshape(-1)
    image = torch.zeros(len(points), dtype=torch.complex128)
    for start in range(0, len(points), chunk):
        A = geometry.responses(points[start:start + chunk], dtype=torch.complex128)   # [P*F, n]
        image[start:start + chunk] = A.conj().T @ y
    return (image.abs() ** 2 / len(geometry.antennas) ** 2).numpy()


def strip_points(box, geometry, along, across, step):
    """Horizontal plane through the box centre: ``along`` metres on the horizontal look direction, ``across`` on cross-range."""
    u, v, _ = geometry.frame(box.centre)
    uh = np.array([u[0], u[1], 0.]); uh /= np.linalg.norm(uh)
    vh = np.array([v[0], v[1], 0.]); vh /= np.linalg.norm(vh)
    a = np.arange(-along, along + step / 2, step)
    b = np.arange(-across, across + step / 2, step)
    A, B = np.meshgrid(a, b, indexing='ij')
    pts = np.asarray(box.centre) + A.reshape(-1, 1) * uh + B.reshape(-1, 1) * vh
    return pts, (len(a), len(b))


def square_points(box, half, step):
    a = np.arange(-half, half + step / 2, step)
    X, Y = np.meshgrid(a, a, indexing='ij')
    pts = np.stack([X.ravel(), Y.ravel(), np.full(X.size, box.centre[2])], 1)
    return pts, a


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset-root', default=os.environ.get('GOTCHA_DATA_ROOT',
                   '/scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined'))
    p.add_argument('--region', default='camry')
    p.add_argument('--box-centre', type=float, nargs=3, required=True)
    p.add_argument('--box-heading', type=float, required=True)
    p.add_argument('--box-half', type=float, nargs=3, required=True)
    p.add_argument('--sectors', type=int, default=8)
    p.add_argument('--range-step', type=float, default=0.12)
    p.add_argument('--cross-step', type=float, default=0.65)
    p.add_argument('--guard-range', type=float, default=0.5)
    p.add_argument('--guard-cross', type=float, default=1.5)
    p.add_argument('--cutoff', type=float, default=1e-2)
    p.add_argument('--footprint', choices=('hull', 'rectangle'), default='hull')
    p.add_argument('--decimations', type=int, nargs='+', default=[4, 6, 8, 12])
    p.add_argument('--strides', type=int, nargs='+', default=[1, 2, 4])
    p.add_argument('--scene-points', type=int, default=300)
    p.add_argument('--scene-pairs', type=int, default=4)
    p.add_argument('--snr-db', type=float, default=20.0)
    p.add_argument('--strip-step', type=float, default=0.25)
    p.add_argument('--ceiling-half', type=float, nargs=3, metavar=('HX', 'HY', 'HZ'),
                   help='half-extents of a synthetic in-box target (e.g. the workbook car) whose filtered, pooled image '
                        'gives the box-share ceiling of the acceptance metric; centred on --ceiling-centre')
    p.add_argument('--ceiling-centre', type=float, nargs=3, default=None)
    p.add_argument('--ceiling-points', type=int, default=400)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    box = Box(tuple(args.box_centre), args.box_heading, tuple(args.box_half))
    rng = np.random.default_rng(1)

    ds = GOTCHADataset(Path(args.dataset_root), passes=range(1, 9), polarizations=('hh',),
                       region=load_region(args.region), num_train=1500, pulses_per_sector=0, frequency_stride=1)
    train = sorted(ds.viewpoints('train'), key=lambda v: v[1])
    views = [train[int((i + 0.5) * len(train) / args.sectors)] for i in range(args.sectors)]
    report = dict(schema='gotcha_isolation_compression_check_v1', roles_read=['train'], region=args.region,
                  box=dict(centre=args.box_centre, heading_deg=args.box_heading, half_extents=args.box_half),
                  grid=dict(range_step=args.range_step, cross_step=args.cross_step, guard_range=args.guard_range,
                            guard_cross=args.guard_cross, cutoff=args.cutoff, footprint=args.footprint),
                  decimations=args.decimations, strides=args.strides, snr_db=args.snr_db, sectors=[])
    square_images, noise_images, ceiling_images = [], [], []
    ceiling = None
    if args.ceiling_half is not None:
        ceiling = Box(tuple(args.ceiling_centre or args.box_centre), args.box_heading, tuple(args.ceiling_half))
        report['ceiling_target'] = dict(centre=list(ceiling.centre), heading_deg=args.box_heading,
                                        half_extents=list(args.ceiling_half), points=args.ceiling_points)
    for view in views:
        observations = list(ds.observations(view[0], view[1], 'hh'))
        geometry = SectorGeometry.from_observations(observations, ds.region)
        pulses, bins = len(observations), len(geometry.frequencies)
        grid, shape = footprint_grid_guarded(box, geometry, range_step=args.range_step, cross_step=args.cross_step,
                                             guard_range=args.guard_range, guard_cross=args.guard_cross,
                                             shape=args.footprint)
        Q_full, S = basis(geometry.responses(grid, dtype=torch.complex128))
        Q, rank = truncate(Q_full, S, args.cutoff)
        del Q_full
        entry = dict(view=list(view), pulses=pulses, frequencies=bins, dictionary_columns=int(len(grid)),
                     grid_shape=list(shape), rank=rank, rank_over_samples=rank / (pulses * bins))
        # 1. decimation equivalence on synthetic in-box scenes
        filtered_ratios, raw_ratios, retention = [], [], []
        for _ in range(args.scene_pairs):
            x1 = geometry.responses(box.sample(args.scene_points, rng), dtype=torch.complex128) @ torch.as_tensor(
                rng.standard_normal(args.scene_points) + 1j * rng.standard_normal(args.scene_points))
            x2 = geometry.responses(box.sample(args.scene_points, rng), dtype=torch.complex128) @ torch.as_tensor(
                rng.standard_normal(args.scene_points) + 1j * rng.standard_normal(args.scene_points))
            noise_scale = float(x1.abs().square().mean().sqrt()) * 10 ** (-args.snr_db / 20) / math.sqrt(2)
            noise = torch.as_tensor(noise_scale * (rng.standard_normal(len(x1)) + 1j * rng.standard_normal(len(x1))))
            y = x1 + noise
            y_filt = Q @ (Q.conj().T @ y)
            retention.append(float(retained_fraction(Q, x1[:, None])[0]))
            filtered_ratios.append(decimation_ratios(y_filt - x2, pulses, bins, args.decimations, args.strides))
            raw_ratios.append(decimation_ratios(y - x2, pulses, bins, args.decimations, args.strides))
        def pool(list_of_dicts):
            keys = list_of_dicts[0].keys()
            return {k: dict(mean=float(np.mean([d[k]['mean'] for d in list_of_dicts])),
                            min=float(np.min([d[k]['min'] for d in list_of_dicts])),
                            max=float(np.max([d[k]['max'] for d in list_of_dicts])),
                            samples=list_of_dicts[0][k]['samples']) for k in keys}
        entry['synthetic_scene_retention'] = dict(min=float(np.min(retention)), mean=float(np.mean(retention)))
        entry['decimation_ratio_filtered'] = pool(filtered_ratios)
        entry['decimation_ratio_raw'] = pool(raw_ratios)
        # 2. data acceptance on the real TRAIN responses of this sector
        y = torch.as_tensor(np.stack([o.response for o in observations]).reshape(-1), dtype=torch.complex128)
        y_filt = Q @ (Q.conj().T @ y)
        entry['data_energy_kept'] = float((y_filt.abs() ** 2).sum() / (y.abs() ** 2).sum())
        pts, sshape = strip_points(box, geometry, 10.0, 75.0, args.strip_step)
        inside = box.contains(pts)
        img_raw = backproject(geometry, y, pts)
        img_filt = backproject(geometry, y_filt, pts)
        # White-noise baseline (rift-4b's caution): a white vector of the raw data's energy through the same
        # projector, so the box share is read against what an unstructured field would give on this footprint.
        white = torch.as_tensor((rng.standard_normal(len(y)) + 1j * rng.standard_normal(len(y))) / math.sqrt(2),
                                dtype=torch.complex128)
        white = white * (y.abs().square().sum() / white.abs().square().sum()).sqrt()
        white_filt = Q @ (Q.conj().T @ white)
        entry['white_noise_energy_kept'] = float((white_filt.abs() ** 2).sum() / (white.abs() ** 2).sum())
        img_white = backproject(geometry, white_filt, pts)
        entry['strip'] = dict(shape=list(sshape), pixels_in_box=int(inside.sum()),
                              box_share_raw=float(img_raw[inside].sum() / img_raw.sum()),
                              box_share_filtered=float(img_filt[inside].sum() / img_filt.sum()),
                              box_share_white_noise=float(img_white[inside].sum() / img_white.sum()))
        sq, axis = square_points(box, 8.0, 0.1)
        square_images.append(backproject(geometry, y_filt, sq).reshape(len(axis), len(axis)))
        noise_images.append(backproject(geometry, white_filt, sq).reshape(len(axis), len(axis)))
        if ceiling is not None:
            # Metric ceiling: a synthetic target filling the workbook car's extent, filtered and imaged like the
            # data; its box share is what a perfectly isolated car scores, since image sidelobes and layover of
            # in-box scatterers also fall outside the footprint.
            amps = torch.as_tensor(rng.standard_normal(args.ceiling_points) + 1j * rng.standard_normal(args.ceiling_points))
            y_c = geometry.responses(ceiling.sample(args.ceiling_points, rng), dtype=torch.complex128) @ amps
            y_c_filt = Q @ (Q.conj().T @ y_c)
            img_c = backproject(geometry, y_c_filt, pts)
            entry['strip']['box_share_ceiling'] = float(img_c[inside].sum() / img_c.sum())
            ceiling_images.append(backproject(geometry, y_c_filt, sq).reshape(len(axis), len(axis)))
        report['sectors'].append(entry)
        print(json.dumps(dict(view=view, rank=rank, columns=len(grid), retention=entry['synthetic_scene_retention'],
                              k6_s2_filtered=entry['decimation_ratio_filtered'].get('k6_s2'),
                              data_kept=round(entry['data_energy_kept'], 4), strip=entry['strip']), default=float),
              flush=True)
        (args.out_dir / 'compression_check.json').write_text(json.dumps(report, indent=2) + '\n')

    shares = [e['strip']['box_share_filtered'] for e in report['sectors']]
    total = np.sum(square_images, axis=0)
    noise_total = np.sum(noise_images, axis=0)
    X, Y = np.meshgrid(axis, axis, indexing='ij')
    inside_sq = box.contains(np.stack([X.ravel(), Y.ravel(), np.full(X.size, box.centre[2])], 1)).reshape(X.shape)
    report['summary'] = dict(median_box_share_filtered=float(np.median(shares)),
                             median_box_share_raw=float(np.median([e['strip']['box_share_raw'] for e in report['sectors']])),
                             median_box_share_white_noise=float(np.median([e['strip']['box_share_white_noise'] for e in report['sectors']])),
                             median_data_energy_kept=float(np.median([e['data_energy_kept'] for e in report['sectors']])),
                             median_white_noise_energy_kept=float(np.median([e['white_noise_energy_kept'] for e in report['sectors']])),
                             min_synthetic_retention=float(np.min([e['synthetic_scene_retention']['min'] for e in report['sectors']])),
                             pooled_square_box_share_filtered=float(total[inside_sq].sum() / total.sum()),
                             pooled_square_box_share_white_noise=float(noise_total[inside_sq].sum() / noise_total.sum()))
    extra = {}
    if ceiling_images:
        ceiling_total = np.sum(ceiling_images, axis=0)
        report['summary']['pooled_square_box_share_ceiling'] = float(ceiling_total[inside_sq].sum() / ceiling_total.sum())
        report['summary']['median_strip_box_share_ceiling'] = float(np.median([e['strip']['box_share_ceiling'] for e in report['sectors']]))
        extra = dict(ceiling_images=np.stack(ceiling_images), ceiling_total=ceiling_total)
    (args.out_dir / 'compression_check.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(args.out_dir / 'filtered_square_images.npz', axis=axis, images=np.stack(square_images), total=total,
                        noise_images=np.stack(noise_images), noise_total=noise_total, **extra)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4.6), constrained_layout=True)
    db = 10 * np.log10(total / total.max() + 1e-30)
    im = ax.imshow(db.T, origin='lower', extent=(-8, 8, -8, 8), cmap='inferno', vmin=-30, vmax=0)
    c = box.corners()[[0, 2, 6, 4, 0]]
    ax.plot(c[:, 0], c[:, 1], 'w--', lw=1)
    ax.set_title(f'filtered, {len(views)} TRAIN sectors, z = {box.centre[2]:+.1f} m')
    ax.set_xlabel('local x (m)'); ax.set_ylabel('local y (m)')
    fig.colorbar(im, ax=ax, label='dB re peak')
    fig.savefig(args.out_dir / 'filtered_square_image.png', dpi=120)
    print(json.dumps(report['summary'], indent=1), flush=True)


if __name__ == '__main__':
    main()
