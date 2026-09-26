#!/usr/bin/env python3
"""Stride-2 range-alias leak into the GOTCHA ROI range window (TRAIN only, CPU).

Native GOTCHA phase histories hold 424 bins at about 1.47 MHz spacing, so a pulse's range
profile is periodic with period c/(2 df) of about 102 m. Keeping every second bin (the
campaign's frequency stride 2) halves that period to about 51 m: the profile at delay d and at
d + 51 m become indistinguishable, and the range-only ROI projector
(``rift_pvc.gotcha_training.RangeReadout``) keeps whatever lies half a native period beyond the
box exactly as if it were inside it. This script measures from the data how much energy that
alias image carries relative to the ROI window itself (review item B1 in
``docs/RIFT_GOTCHA_Tune.md``).

For TRAIN sectors spread around the circle, every native pulse, all native bins:

* the matched-filter range profile ``sum_f y(f) exp(+i 4 pi f d / c) / N`` on a fine delay grid
  covering one native period (delays are relative to the reader's effective reference range);
* the ROI delay window of a box by the ``RangeReadout`` rule (nearest and farthest box point
  from the antenna, two range cells of guard on each side), for the current 10 m cube and for a
  candidate anisotropic box given in the registered local frame;
* ``E_window``: profile energy inside that window; ``E_alias``: energy in the same window shifted
  by half a native period, i.e. the unique stride-2 alias image; ``E_total``: energy over the
  period.

Reported per sector and pooled: ``E_window / E_total`` (the range-only isolation share) and
``E_alias / E_window`` (the stride-2 leak relative to the retained content). Reads TRAIN
responses only and proposes nothing on its own.

    source .local-setup/activate-pvc.sh
    python scripts_pvc/gotcha_range_alias_check.py --out-dir <dir>
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rift.gotcha_dataset import C, GOTCHADataset, load_region  # noqa: E402

PASSES = tuple(range(1, 9))


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


def window_bounds(antenna_local, centre, half, reference_range, spacing):
    """RangeReadout's guarded delay window for an axis-aligned local box (per-axis half extents)."""
    rel = antenna_local - centre
    nearest = np.clip(rel, -half, half)
    r_min = float(np.linalg.norm(rel - nearest))
    r_max = float(np.linalg.norm(np.abs(rel) + half))
    return r_min - reference_range - 2 * spacing, r_max - reference_range + 2 * spacing


def in_window(delays, low, high, period):
    """Mask of periodic delays inside [low, high] (window shorter than the period)."""
    shifted = np.mod(delays - low, period)
    return shifted <= (high - low)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset-root', default=os.environ.get('GOTCHA_DATA_ROOT',
                   '/scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined'))
    p.add_argument('--region', default='camry')
    p.add_argument('--azimuth-step', type=int, default=15)
    p.add_argument('--delay-step', type=float, default=0.05, help='range-profile sampling (m)')
    p.add_argument('--cube-half', type=float, default=5.0, help='current cube half extent (m)')
    p.add_argument('--box-half', type=float, nargs=3, default=[3.0, 3.0, 2.0], metavar=('HX', 'HY', 'HZ'),
                   help='candidate box half extents in the local frame (m)')
    p.add_argument('--box-centre', type=float, nargs=3, default=[0.0, 0.0, 1.0], metavar=('X', 'Y', 'Z'),
                   help='candidate box centre in the local frame (m)')
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ds = GOTCHADataset(Path(args.dataset_root), passes=PASSES, polarizations=('hh',),
                       region=load_region(args.region), num_train=1500, pulses_per_sector=0, frequency_stride=1)
    region = ds.region
    views = choose_sectors(ds.viewpoints('train'), args.azimuth_step)
    boxes = dict(cube10=(np.zeros(3), np.full(3, args.cube_half)),
                 box=(np.asarray(args.box_centre, dtype=np.float64), np.asarray(args.box_half, dtype=np.float64)))

    per_sector = []
    pooled = {name: dict(window=0.0, alias=0.0, total=0.0) for name in boxes}
    for n, (pass_id, sector) in enumerate(views, 1):
        obs = list(ds.observations(pass_id, sector, 'hh'))
        f = obs[0].frequencies_hz
        df = float(np.median(np.diff(f)))
        period = C / (2 * df)
        spacing = C / (2 * (f[-1] - f[0]))
        delays = np.arange(-period / 2, period / 2, args.delay_step)
        kernel = np.exp((4j * math.pi / C) * np.outer(delays, f)) / len(f)     # [D, F]
        sums = {name: dict(window=0.0, alias=0.0, total=0.0) for name in boxes}
        for o in obs:
            if not np.array_equal(o.frequencies_hz, f):
                raise ValueError('Frequency vector changed within a sector')
            profile = kernel @ o.response
            energy = np.abs(profile) ** 2
            antenna = region.to_local(o.position_m)
            total = float(energy.sum())
            for name, (centre, half) in boxes.items():
                low, high = window_bounds(antenna, centre, half, o.reference_range_m, spacing)
                if high - low >= period / 2:
                    raise ValueError(f'{name}: window {high - low:.1f} m is not shorter than the stride-2 period')
                mask = in_window(delays, low, high, period)
                alias = in_window(delays, low + period / 2, high + period / 2, period)
                sums[name]['window'] += float(energy[mask].sum())
                sums[name]['alias'] += float(energy[alias].sum())
                sums[name]['total'] += total
        record = dict(pass_id=pass_id, sector_id=sector, pulses=len(obs), native_bins=len(f),
                      native_df_hz=df, native_period_m=period, range_cell_m=spacing)
        for name in boxes:
            s = sums[name]
            for k in ('window', 'alias', 'total'):
                pooled[name][k] += s[k]
            record[name] = dict(window_share=s['window'] / s['total'], alias_over_window=s['alias'] / s['window'],
                                alias_share=s['alias'] / s['total'])
        per_sector.append(record)
        print(f"{n}/{len(views)} sector ({pass_id},{sector}) pulses {len(obs)} "
              + ' '.join(f"{name}: window {record[name]['window_share']:.3f} alias/window {record[name]['alias_over_window']:.3f}"
                         for name in boxes), flush=True)

    summary = {}
    for name, (centre, half) in boxes.items():
        s = pooled[name]
        ratios = np.array([r[name]['alias_over_window'] for r in per_sector])
        shares = np.array([r[name]['window_share'] for r in per_sector])
        summary[name] = dict(centre_local_m=[float(v) for v in centre], half_extent_m=[float(v) for v in half],
                             pooled_window_share=s['window'] / s['total'],
                             pooled_alias_over_window=s['alias'] / s['window'],
                             median_alias_over_window=float(np.median(ratios)),
                             range_alias_over_window=[float(ratios.min()), float(ratios.max())],
                             median_window_share=float(np.median(shares)))
    report = dict(schema='gotcha_range_alias_check_v1', region=args.region, roles_read=['train'],
                  sectors=len(views), views=[list(v) for v in views], frequency_stride_of_data=1,
                  method='matched-filter range profile per native pulse; RangeReadout window rule with two guard cells; '
                         'alias window = same window shifted by half the native period (the stride-2 fold)',
                  delay_step_m=args.delay_step, summary=summary, per_sector=per_sector)
    (args.out_dir / 'range_alias_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
