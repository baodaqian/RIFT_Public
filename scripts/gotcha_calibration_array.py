#!/usr/bin/env python3
"""Measure the GOTCHA system constant from the dataset's calibration trihedrals (TRAIN sectors only).

See rift/gotcha_calibration.py for the method and for what is the dataset's versus
ours. Every selected pass is imaged against the production joint split (all eight
passes, 1500 TRAIN sectors), so reserved-test sectors are never read. The output JSON
records every reflector/pass measurement and the summary constant
(median over the clearly detected 15-inch trihedrals; the 27-inch one is the cross-check).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rift.gotcha_calibration import (BORESIGHT_HALF_WIDTH_DEG, SOURCE, TRIHEDRALS, boresight_sectors,  # noqa: E402
                                     measure_trihedral, system_constant, triangular_trihedral_rcs)
from rift.gotcha_dataset import C, GOTCHADataset, load_region  # noqa: E402

SPLIT = dict(passes=list(range(1, 9)), num_train=1500)
DETECTION_RATIO = 10.0


def measure_pass(job):
    root, pass_id, polarization = job
    ds = GOTCHADataset(Path(root), passes=SPLIT['passes'], polarizations=(polarization,),
                       region=load_region('camry', None), num_train=SPLIT['num_train'])
    shard, train = ds.shards[pass_id, polarization], ds.splits_by_pass[pass_id]['train']
    rows = []
    for name, (edge, x, y, z, heading) in TRIHEDRALS.items():
        sectors = boresight_sectors(train, heading)
        observations = [shard.read(int(r)) for s in sectors for r in shard.sector_rows[s]]
        if not observations:
            rows.append(dict(target=name, pass_id=pass_id, sectors=[], status='no_train_sectors_at_boresight'))
            continue
        result = measure_trihedral(observations, (x, y, z))
        frequencies = np.concatenate([o.frequencies_hz for o in observations])
        wavelength = C / float(frequencies.mean())
        rcs = triangular_trihedral_rcs(edge, wavelength)
        up = np.array([o.position_m[2] - z for o in observations])
        ground = np.array([np.hypot(o.position_m[0] - x, o.position_m[1] - y) for o in observations])
        rows.append(dict(target=name, pass_id=pass_id, sectors=sectors, heading_deg=heading, edge_m=edge,
                         wavelength_m=wavelength, rcs_m2=rcs, rcs_dbsm=10 * math.log10(rcs),
                         elevation_deg=float(np.degrees(np.arctan2(up, ground)).mean()),
                         detection_ratio=result['amplitude'] / result['background_amplitude'],
                         system_constant=system_constant(result['amplitude'], result['mean_range_m'], rcs),
                         status='measured', **result))
    return rows


def summarize(rows):
    measured = [r for r in rows if r.get('status') == 'measured']
    detected = [r for r in measured if r['detection_ratio'] >= DETECTION_RATIO]
    small = np.array([r['system_constant'] for r in detected if r['target'].startswith('15TR')])
    large = np.array([r['system_constant'] for r in detected if r['target'].startswith('27TR')])
    median = float(np.median(small))
    return dict(system_constant=median, estimator='median over detected 15-inch trihedral measurements',
                detected_15in=len(small), measured=len(measured),
                spread_15in_mad_relative=float(np.median(np.abs(small - median)) / median),
                cross_check_27in_median=float(np.median(large)) if len(large) else None,
                cross_check_27in_over_15in_db=(float(20 * np.log10(np.median(large) / median)) if len(large) else None),
                offset_median_m=[float(v) for v in np.median([r['offset_m'] for r in detected], axis=0)])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--polarization', default='hh', choices=('hh', 'vv'))
    p.add_argument('--passes', nargs='+', type=int, default=SPLIT['passes'])
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(argv)
    jobs = [(str(args.dataset_root), p_id, args.polarization) for p_id in args.passes]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [row for rows in pool.map(measure_pass, jobs) for row in rows]
    record = dict(schema='gotcha_calibration_array_v1', source=SOURCE, polarization=args.polarization,
                  split=dict(SPLIT, roles_read=['train']), boresight_half_width_deg=BORESIGHT_HALF_WIDTH_DEG,
                  rcs_model='triangular trihedral 4 pi a^4 / (3 lambda^2) at the mean selected frequency',
                  constant_meaning='data / K = sqrt(RCS) / (Rt + Rr)^2, GeRaF amplitude-law units',
                  detection_ratio_threshold=DETECTION_RATIO, summary=summarize(rows), measurements=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
