#!/usr/bin/env python3
"""Where exactly is the vehicle? Data-driven placement evidence for a smaller GOTCHA box (TRAIN only, CPU).

For TRAIN sectors spread around the circle (one per ``--azimuth-step`` degrees, cycling passes), each
sector forms a coherent image with all its native pulses (the adjoint of the native kernel, the
calibration array's ``coherent_image`` convention) and the images are summed incoherently:

1. horizontal planes at several heights over a window around the registered region, for the
   footprint and heading;
2. a height sweep over a smaller window: with a ~45 degree look around a full circle, a scatterer
   imaged off its true height smears into a ring of radius ~ height error / tan(elevation), so image
   sharpness sum|I|^4 / (sum|I|^2)^2 against height shows where the ground and the body focus.

Coordinates are the registered region's local frame (``Region.to_local``). The script reports the
bright-footprint centroid, principal axis (heading) and extent above an energy threshold, and writes
figures; it proposes nothing on its own. Reads TRAIN responses only.

    source .local-setup/activate-pvc.sh
    python scripts_pvc/gotcha_box_placement.py --out-dir <dir>
"""
import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rift.gotcha_dataset import C, GOTCHADataset, load_region  # noqa: E402

PASSES = tuple(range(1, 9))


def _dataset(root, region_name, stride):
    return GOTCHADataset(Path(root), passes=PASSES, polarizations=('hh',), region=load_region(region_name),
                         num_train=1500, pulses_per_sector=0, frequency_stride=stride)


def choose_sectors(train, step):
    """One TRAIN (pass, sector) per ``step`` degrees of azimuth, cycling through the passes."""
    chosen, used = [], set()
    for bin_start in range(1, 361, step):
        window = [v for v in train if bin_start <= v[1] < bin_start + step]
        if not window:
            continue
        preferred = [v for v in window if v[0] == PASSES[len(chosen) % len(PASSES)]]
        pick = sorted(preferred or window)[0]
        if pick not in used:
            used.add(pick)
            chosen.append(pick)
    return chosen


def sector_energy(job):
    """|coherent image|^2 of one sector on each requested local point set (lists of [N, 3])."""
    root, region_name, stride, view, point_sets, threads = job
    import torch
    torch.set_num_threads(threads)
    ds = _dataset(root, region_name, stride)
    region = ds.region
    out = [np.zeros(len(points)) for points in point_sets]
    observations = list(ds.observations(view[0], view[1], 'hh'))
    for k, points in enumerate(point_sets):
        pts = torch.as_tensor(points, dtype=torch.float64)
        image = torch.zeros(len(pts), dtype=torch.complex128)
        for obs in observations:
            antenna = torch.as_tensor(region.to_local(obs.position_m))
            f = torch.as_tensor(obs.frequencies_hz, dtype=torch.float64)
            y = torch.as_tensor(obs.response, dtype=torch.complex128)
            for start in range(0, len(pts), 8192):
                d = torch.linalg.vector_norm(pts[start:start + 8192] - antenna, dim=-1) - obs.reference_range_m
                image[start:start + 8192] += (torch.exp(1j * (4 * math.pi / C) * d[:, None] * f[None]) * y).sum(1)
        out[k] = (image.abs().square() / len(observations) ** 2).numpy()
    elevation = float(np.degrees(np.mean([math.atan2(region.to_local(o.position_m)[2],
                  np.hypot(*region.to_local(o.position_m)[:2])) for o in observations])))
    return view, out, elevation


def plane(half, step, z):
    axis = np.arange(-half, half + step / 2, step)
    gx, gy = np.meshgrid(axis, axis, indexing='ij')
    return np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], 1), axis


def footprint(energy, axis, fraction):
    """Centroid, principal axis and extents of the brightest pixels holding ``fraction`` of the energy."""
    flat = energy.ravel()
    order = np.argsort(flat)[::-1]
    keep = order[:np.searchsorted(np.cumsum(flat[order]), fraction * flat.sum()) + 1]
    gx, gy = np.meshgrid(axis, axis, indexing='ij')
    xy = np.stack([gx.ravel()[keep], gy.ravel()[keep]], 1)
    w = flat[keep] / flat[keep].sum()
    centre = (w[:, None] * xy).sum(0)
    cov = ((xy - centre).T * w) @ (xy - centre)
    values, vectors = np.linalg.eigh(cov)
    major = vectors[:, 1]
    heading = math.degrees(math.atan2(major[1], major[0])) % 180
    along = (xy - centre) @ major
    across = (xy - centre) @ vectors[:, 0]
    return dict(energy_fraction=fraction, pixels=int(len(keep)), centre_xy_m=[float(v) for v in centre],
                heading_deg_local=float(heading),
                extent_along_m=[float(np.percentile(along, 2)), float(np.percentile(along, 98))],
                extent_across_m=[float(np.percentile(across, 2)), float(np.percentile(across, 98))])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset-root', default=os.environ.get('GOTCHA_DATA_ROOT',
                   '/scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined'))
    p.add_argument('--region', default='camry')
    p.add_argument('--frequency-stride', type=int, default=1)
    p.add_argument('--azimuth-step', type=int, default=5)
    p.add_argument('--half', type=float, default=8.0, help='horizontal window half-width (m)')
    p.add_argument('--step', type=float, default=0.1)
    p.add_argument('--heights', type=float, nargs='+', default=[-0.5, 0.0, 0.5, 1.0, 1.5])
    p.add_argument('--sweep-half', type=float, default=4.0)
    p.add_argument('--sweep-step', type=float, default=0.1)
    p.add_argument('--sweep-z', type=float, nargs=3, default=[-1.5, 2.5, 0.1], metavar=('LOW', 'HIGH', 'STEP'))
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ds = _dataset(args.dataset_root, args.region, args.frequency_stride)
    views = choose_sectors(ds.viewpoints('train'), args.azimuth_step)
    planes = [plane(args.half, args.step, z) for z in args.heights]
    zs = np.arange(args.sweep_z[0], args.sweep_z[1] + args.sweep_z[2] / 2, args.sweep_z[2])
    sweep = [plane(args.sweep_half, args.sweep_step, z)[0] for z in zs]
    point_sets = [pts for pts, _ in planes] + sweep
    totals = [np.zeros(len(ps)) for ps in point_sets]
    elevations = []
    jobs = [(args.dataset_root, args.region, args.frequency_stride, v, point_sets, args.threads) for v in views]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, (view, energies, elevation) in enumerate(pool.map(sector_energy, jobs), 1):
            for t, e in zip(totals, energies):
                t += e
            elevations.append(elevation)
            print(f'{n}/{len(views)} sector {view} elevation {elevation:.2f}', flush=True)
    axis = planes[0][1]
    grid_shape = (len(axis), len(axis))
    images = {z: totals[i].reshape(grid_shape) for i, z in enumerate(args.heights)}
    sharpness = []
    for i, z in enumerate(zs):
        e = totals[len(planes) + i]
        sharpness.append(float((e ** 2).sum() / e.sum() ** 2))
    np.savez_compressed(args.out_dir / 'box_placement_images.npz', axis=axis, heights=np.asarray(args.heights),
                        images=np.stack([images[z] for z in args.heights]), sweep_z=zs,
                        sweep=np.stack([t.reshape(-1) for t in totals[len(planes):]]))
    report = dict(schema='gotcha_box_placement_v1', region=args.region, roles_read=['train'], sectors=len(views),
                  views=[list(v) for v in views], mean_elevation_deg=float(np.mean(elevations)),
                  frequency_stride=args.frequency_stride, pulses='all native pulses per sector',
                  height_sweep=dict(z=[float(z) for z in zs], sharpness=sharpness,
                                    best_z=float(zs[int(np.argmax(sharpness))])),
                  footprints={f'z={z:+.1f}': {str(f): footprint(images[z], axis, f) for f in (0.3, 0.5)}
                              for z in args.heights})
    (args.out_dir / 'box_placement.json').write_text(json.dumps(report, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(args.heights) + 1, figsize=(4.2 * (len(args.heights) + 1), 4.4),
                             constrained_layout=True)
    peak = max(img.max() for img in images.values())
    for ax, z in zip(axes, args.heights):
        db = 10 * np.log10(images[z] / peak + 1e-30)
        im = ax.imshow(db.T, origin='lower', extent=(-args.half, args.half, -args.half, args.half), cmap='inferno',
                       vmin=-30, vmax=0)
        ax.plot([-5, 5, 5, -5, -5], [-5, -5, 5, 5, -5], 'c-', lw=1, label='current 10 m cube')
        ax.plot([-3, 3, 3, -3, -3], [-3, -3, 3, 3, -3], 'w--', lw=1, label='6 m square (reference)')
        ax.set_title(f'z = {z:+.1f} m (local)')
        ax.set_xlabel('local x (m)')
    axes[0].set_ylabel('local y (m)')
    axes[0].legend(fontsize=7, loc='lower left')
    fig.colorbar(im, ax=axes[:-1], shrink=0.8, label='dB re peak over all heights')
    axes[-1].plot(zs, sharpness, 'o-')
    axes[-1].set_xlabel('image plane z (m, local)')
    axes[-1].set_ylabel('sharpness  sum|I|^4 / (sum|I|^2)^2')
    axes[-1].set_title(f'height sweep, +-{args.sweep_half:g} m window')
    fig.savefig(args.out_dir / 'box_placement.png', dpi=120)
    print(json.dumps(dict(best_z=report['height_sweep']['best_z'],
                          footprint_z0=report['footprints'].get('z=+0.0')), indent=1), flush=True)


if __name__ == '__main__':
    main()
