#!/usr/bin/env python3
"""Qualify per-sector box isolation on GOTCHA geometry by injection (Phase 0c; reviewer R5/R6).

For sampled TRAIN sectors (all native pulses; frequency stride as selected) the script builds the
slant-plane footprint dictionary of a box (``rift_pvc.gotcha_isolation``), and for each guard band and
singular-value cutoff it reports the fraction of energy a unit point keeps after the projection:

- interior: random points in the box, its corners and face centres (should be ~1);
- exterior rings at 0.5-75 m from the box surface, at mid-box height (should fall to ~0);
- the same-range strip, along cross-range at +-10..75 m (the clutter today's target keeps);
- range aliases, the box centre moved by +-Lambda and +-2 Lambda along the look direction, where
  Lambda = c / (2 * median frequency step) (about 51 m at stride 2, 102 m at stride 1);
- layover, the centre moved along the sector's layover normal (inherently ambiguous: reported, not
  counted as leakage).

The same points are also passed through today's per-pulse ``RangeReadout`` of the registered region,
for comparison. Real TRAIN data of the same sectors report the energy each operator keeps.
Geometry only, except for that last check (TRAIN responses). TEST stays sealed.

    source .local-setup/activate-pvc.sh
    python scripts_pvc/gotcha_isolation_qualify.py --box-centre X Y Z --box-heading DEG --box-half HX HY HZ --out-dir DIR
"""
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rift.gotcha_dataset import C, GOTCHADataset, load_region  # noqa: E402
from rift_pvc.gotcha_isolation import (Box, SectorGeometry, basis, footprint_grid, retained_fraction,  # noqa: E402
                                       truncate)
from rift_pvc.gotcha_training import RangeReadout  # noqa: E402

DISTANCES = (0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 75.0)


NAMED = dict(  # exterior features seen in box_placement.png (local frame, metres)
    neighbour_plus_y=[(x, 7.0, z) for x in (-2.0, 0.0, 2.0) for z in (0.3, 1.0)],
    neighbour_minus_y=[(x, -7.0, z) for x in (-2.0, 0.0, 2.0) for z in (0.3, 1.0)],
    line_x7=[(7.0, y, 0.0) for y in (-6.0, -3.0, 0.0, 3.0, 6.0)])


def test_points(box, geometry, alias, rng):
    u, v, n = geometry.frame(box.centre)
    centre = np.asarray(box.centre)
    half = np.asarray(box.half_extents)
    faces = np.concatenate([np.diag(half), -np.diag(half)])
    sets = dict(interior=np.concatenate([box.sample(400, rng), box.corners(), box.from_box(faces)]))
    sets.update({name: np.asarray(points, dtype=np.float64) for name, points in NAMED.items()})
    for label, ring_z in (('', centre[2]), ('ground_', 0.0), ('below_', -0.5)):
        _rings(sets, box, centre, ring_z, label)
    sets['strip'] = centre + np.array([s * v for s in (-75, -40, -20, -10, 10, 20, 40, 75)])
    # Range aliases: half the native period is what stride 2 folds; the full native period is the
    # native data's own ambiguity (identical phase ramp), recorded rather than counted as leakage.
    sets['alias_half'] = centre + np.array([k * alias / 2 * u for k in (-1, 1)])
    sets['alias_full'] = centre + np.array([k * alias * u for k in (-1, 1)])
    sets['layover'] = centre + np.array([h * n for h in (-4, -2, -1, 1, 2, 4)])
    grid = np.arange(-20.0, 20.0 + 1e-9, 1.0)
    gx, gy = np.meshgrid(grid, grid, indexing='ij')
    for label, z in (('grid_ground', 0.0), ('grid_body', 1.0)):
        pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], 1)
        sets[label] = pts[~box.contains(pts)]
    return sets


def footprint_distance_cells(box, geometry, points, range_guard, cross_guard, range_cell, cross_cell):
    """Distance of each point, in resolution cells, outside the guarded hull footprint (0 inside).

    Coordinates are the sector's slant-plane (u, v) about the box centre, scaled by the range and
    cross-range cells so that one unit is one cell in either direction (reviewer B9 metric).
    """
    from scipy.spatial import ConvexHull
    u, v, _ = geometry.frame(box.centre)
    corners = box.corners() - np.asarray(box.centre)
    grown = np.array([((c @ u) + da, (c @ v) + db) for c in corners
                      for da in (-range_guard, range_guard) for db in (-cross_guard, cross_guard)])
    scale = np.array([range_cell, cross_cell])
    hull = grown[ConvexHull(grown).vertices] / scale
    rel = np.asarray(points) - np.asarray(box.centre)
    q = np.stack([rel @ u, rel @ v], 1) / scale
    distance = np.full(len(q), np.inf)
    inside = np.ones(len(q), dtype=bool)
    for a, b in zip(hull, np.roll(hull, -1, 0)):
        edge = b - a
        t = np.clip(((q - a) @ edge) / (edge @ edge), 0, 1)
        distance = np.minimum(distance, np.linalg.norm(q - (a + t[:, None] * edge), axis=1))
        inside &= (edge[0] * (q[:, 1] - a[1]) - edge[1] * (q[:, 0] - a[0])) >= 0
    return np.where(inside, 0., distance), q * scale


def _rings(sets, box, centre, ring_z, label):
    for d in DISTANCES:
        # Points on the horizontal plane through the box centre, at distance d from the box surface.
        angles = np.linspace(0, 2 * math.pi, 72, endpoint=False)
        pts = []
        for t in angles:
            direction = np.array([math.cos(t), math.sin(t), 0.])
            lo, hi = 0., 200.
            for _ in range(60):  # bisection on the ray from the centre for the requested surface distance
                mid = (lo + hi) / 2
                if box.distance(centre + mid * direction) < d:
                    lo = mid
                else:
                    hi = mid
            p = centre + lo * direction
            p[2] = ring_z
            pts.append(p)
        sets[f'{label}ring_{d:g}m'] = np.asarray(pts)


def range_readout_retention(readout, observations, geometry, points):
    """Energy kept by today's per-pulse range projection, pooled over the sector's pulses."""
    Y = geometry.responses(points, dtype=torch.complex128).reshape(len(observations), len(geometry.frequencies), -1)
    kept = total = torch.zeros(len(points), dtype=torch.float64)
    for p, obs in enumerate(observations):
        r = readout.for_observation(obs)
        coefficients = r['q'].conj().T.to(torch.complex128) @ (r['phase'].to(torch.complex128)[:, None] * Y[p])
        kept = kept + coefficients.abs().square().sum(0)
        total = total + Y[p].abs().square().sum(0)
    return (kept / total).numpy()


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(min=float(values.min()), p05=float(np.percentile(values, 5)), median=float(np.median(values)),
                max=float(values.max()))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset-root', default=os.environ.get('GOTCHA_DATA_ROOT',
                   '/scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined'))
    p.add_argument('--region', default='camry')
    p.add_argument('--box-centre', type=float, nargs=3, required=True)
    p.add_argument('--box-heading', type=float, required=True)
    p.add_argument('--box-half', type=float, nargs=3, required=True)
    p.add_argument('--frequency-strides', type=int, nargs='+', default=[1, 2])
    p.add_argument('--sectors', type=int, default=4)
    p.add_argument('--range-step', type=float, default=0.10)
    p.add_argument('--cross-step', type=float, default=0.40)
    p.add_argument('--range-guards', type=float, nargs='+', default=[0.25, 0.5])
    p.add_argument('--cross-guards', type=float, nargs='+', default=[1.3, 2.0])
    p.add_argument('--cutoffs', type=float, nargs='+', default=[1e-1, 3e-2, 1e-2, 1e-3, 1e-4])
    p.add_argument('--footprint-shape', choices=('hull', 'rectangle'), default='hull')
    p.add_argument('--range-cell', type=float, default=0.24, help='range resolution cell (m) for distance scaling')
    p.add_argument('--cross-cell', type=float, default=1.3, help='cross-range resolution cell (m) for distance scaling')
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    box = Box(tuple(args.box_centre), args.box_heading, tuple(args.box_half))
    rng = np.random.default_rng(0)
    report = dict(schema='gotcha_isolation_qualification_v1', box=dict(centre=args.box_centre, heading_deg=args.box_heading,
                  half_extents=args.box_half), range_step=args.range_step, cross_step=args.cross_step, roles_read=['train'],
                  footprint_shape=args.footprint_shape,
                  strides={})
    for stride in args.frequency_strides:
        ds = GOTCHADataset(Path(args.dataset_root), passes=range(1, 9), polarizations=('hh',),
                           region=load_region(args.region), num_train=1500, pulses_per_sector=0, frequency_stride=stride)
        readout = RangeReadout(ds.region, device='cpu')
        train = ds.viewpoints('train')
        spaced = sorted(train, key=lambda v: v[1])
        views = [spaced[int(i * len(spaced) / args.sectors)] for i in range(args.sectors)]
        stride_report = []
        for view in views:
            observations = list(ds.observations(view[0], view[1], 'hh'))
            geometry = SectorGeometry.from_observations(observations, ds.region)
            alias = C / (2 * float(np.median(np.diff(geometry.frequencies))))
            sets = test_points(box, geometry, alias, rng)
            data = torch.as_tensor(np.stack([o.response for o in observations]).reshape(-1, 1), dtype=torch.complex128)
            today = {name: summarize(range_readout_retention(readout, observations, geometry, pts))
                     for name, pts in sets.items()}
            today_data = float(range_readout_retention_data(readout, observations))
            sector = dict(view=list(view), pulses=len(observations), frequencies=len(geometry.frequencies),
                          alias_period_m=alias, today_range_readout=dict(points=today, data_energy_kept=today_data),
                          guards={})
            # Responses once per sector; projection coefficients once per guard; each cutoff is then a
            # prefix sum over the singular modes (retention at rank r = sum of the first r |Q^H y|^2).
            responses = {name: geometry.responses(pts, dtype=torch.complex128) for name, pts in sets.items()}
            norms = {name: Y.abs().square().sum(0) for name, Y in responses.items()}
            for range_guard, cross_guard in [(rg, cg) for rg in args.range_guards for cg in args.cross_guards]:
                guard = f'{range_guard:g}x{cross_guard:g}'
                grid = footprint_grid(box, geometry, range_step=args.range_step, cross_step=args.cross_step,
                                      range_guard=range_guard, cross_guard=cross_guard, shape=args.footprint_shape)
                Q_full, S = basis(geometry.responses(grid, dtype=torch.complex128))
                prefix = {name: torch.cumsum((Q_full.conj().T @ Y).abs().square(), 0) / norms[name]
                          for name, Y in responses.items()}
                data_prefix = torch.cumsum((Q_full.conj().T @ data).abs().square(), 0)[:, 0] / data.abs().square().sum()
                entry = dict(dictionary_columns=int(len(grid)),
                             singular_values_rel=[float(x) for x in (S / S[0])[::max(1, len(S) // 64)]], cutoffs={})
                for cutoff in args.cutoffs:
                    rank = int((S >= cutoff * S[0]).sum())
                    entry['cutoffs'][f'{cutoff:g}'] = dict(
                        rank=rank, points={name: summarize(prefix[name][rank - 1].numpy()) for name in sets},
                        data_energy_kept=float(data_prefix[rank - 1]))
                sector['guards'][guard] = entry
                names = list(sets)
                points = np.concatenate([sets[k] for k in names])
                dist, uv = footprint_distance_cells(box, geometry, points, range_guard, cross_guard,
                                                    args.range_cell, args.cross_cell)
                ranks = [entry['cutoffs'][f'{c:g}']['rank'] for c in args.cutoffs]
                np.savez_compressed(args.out_dir / f'points_{view[0]}_{view[1]}_stride{stride}_{guard}.npz',
                                    points=points, uv=uv, distance_cells=dist,
                                    labels=np.concatenate([[k] * len(sets[k]) for k in names]),
                                    cutoffs=np.asarray(args.cutoffs), ranks=np.asarray(ranks),
                                    retention=np.stack([torch.cat([prefix[k][r - 1] for k in names]).numpy() for r in ranks]))
                del Q_full, prefix
                print(stride, view, 'guard', guard, 'columns', len(grid),
                      {c: (e['rank'], round(e['points']['interior']['p05'], 3), round(e['points']['ground_ring_8m']['max'], 4),
                           round(e['points']['strip']['max'], 4), round(e['points']['alias_half']['max'], 4),
                           round(max(e['points'][k]['max'] for k in NAMED), 4), round(e['data_energy_kept'], 4))
                       for c, e in entry['cutoffs'].items()}, flush=True)
            print(stride, view, 'today range readout: interior p05 %.3f, ground ring_8m max %.3f, strip max %.3f, alias_half max %.3f, data kept %.4f'
                  % (today['interior']['p05'], today['ground_ring_8m']['max'], today['strip']['max'], today['alias_half']['max'], today_data),
                  flush=True)
            stride_report.append(sector)
        report['strides'][str(stride)] = stride_report
        (args.out_dir / 'isolation_qualification.json').write_text(json.dumps(report, indent=2) + '\n')


def range_readout_retention_data(readout, observations):
    kept = total = 0.
    for obs in observations:
        r = readout.for_observation(obs)
        y = torch.as_tensor(obs.response, dtype=torch.complex128)
        kept += float(RangeReadout.project(y, r).abs().square().sum())
        total += float(y.abs().square().sum())
    return kept / total


if __name__ == '__main__':
    main()
