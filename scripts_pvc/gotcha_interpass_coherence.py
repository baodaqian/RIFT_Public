#!/usr/bin/env python3
"""Are GOTCHA's eight passes mutually phase-coherent? Tested on the dataset's own trihedrals (CPU).

The published autofocus (applied once per pulse by the reader) focuses each pass; RIFT fits one scene
coherently across all passes, which also needs the passes to agree in phase. For each Table 1
trihedral (point-like phase centre), every pass forms the calibration array's coherent image
(``rift.gotcha_calibration.coherent_image``: the mean per-pulse matched response, native kernel,
all native pulses and frequencies of its TRAIN sectors within +-6 degrees of boresight). Then:

1. a common 3D point is found where the eight complex images add most coherently: first along
   height (the passes differ slightly in elevation, so a phase centre off in z alone makes the
   pass phases disagree), then in x/y;
2. at that point it reports each pass's amplitude and phase and the coherence
   |sum_p I_p| / sum_p |I_p| (1 = passes add in phase; about 1/sqrt(8) = 0.35 for random phases).

Per-pass phases that repeat across all trihedrals point to a constant per-pass offset; phases
that vary smoothly with reflector position point to a per-pass geometry error.
Reads TRAIN responses only.

    source .local-setup/activate-pvc.sh
    python scripts_pvc/gotcha_interpass_coherence.py --dataset-root "$GOTCHA_DATA_ROOT" --out interpass.json
"""
import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rift.gotcha_calibration import TRIHEDRALS, boresight_sectors, coherent_image, measure_trihedral  # noqa: E402
from rift.gotcha_dataset import GOTCHADataset, load_region  # noqa: E402

PASSES = tuple(range(1, 9))


def _observations(root, pass_id, heading, polarization):
    ds = GOTCHADataset(Path(root), passes=PASSES, polarizations=(polarization,),
                       region=load_region('camry', None), num_train=1500)
    shard, train = ds.shards[pass_id, polarization], ds.splits_by_pass[pass_id]['train']
    sectors = boresight_sectors(train, heading)
    return [shard.read(int(r)) for s in sectors for r in shard.sector_rows[s]]


def pass_images(job):
    """One pass: its own fine peak, and its complex image on the caller's point set."""
    root, pass_id, heading, position, points, polarization = job
    observations = _observations(root, pass_id, heading, polarization)
    if not observations:
        return pass_id, None, None, 0
    peak = measure_trihedral(observations, position)
    image = coherent_image(observations, np.asarray(points)) if points is not None else None
    elevation = float(np.degrees(np.mean([math.atan2(o.position_m[2] - position[2],
                  math.hypot(o.position_m[0] - position[0], o.position_m[1] - position[1])) for o in observations])))
    return pass_id, peak, image, elevation


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--polarization', default='hh')
    p.add_argument('--z-half', type=float, default=0.6)
    p.add_argument('--z-step', type=float, default=0.01)
    p.add_argument('--xy-half', type=float, default=0.04)
    p.add_argument('--xy-step', type=float, default=0.01)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(argv)
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, (edge, x, y, z, heading) in TRIHEDRALS.items():
            position = np.array([x, y, z], dtype=np.float64)
            # Stage 0: each pass's own peak (as the calibration array measures it).
            first = list(pool.map(pass_images, [(args.dataset_root, q, heading, position, None, args.polarization)
                                                for q in PASSES]))
            peaks = {q: pk for q, pk, _, _ in first if pk is not None}
            if len(peaks) < 2:
                results[name] = dict(status='fewer than two passes with TRAIN sectors at boresight')
                continue
            centre = position + np.median([pk['offset_m'] for pk in peaks.values()], axis=0)
            # Stage 1: height line through the median peak.
            zs = np.arange(-args.z_half, args.z_half + args.z_step / 2, args.z_step)
            line = np.stack([np.full_like(zs, centre[0]), np.full_like(zs, centre[1]), centre[2] + zs], 1)
            stage = {q: img for q, _, img, _ in pool.map(pass_images, [(args.dataset_root, q, heading, position,
                     line, args.polarization) for q in peaks])}
            best_z = line[int(np.abs(sum(stage.values())).argmax()), 2]
            # Stage 2: x/y refinement at that height.
            axis = np.arange(-args.xy_half, args.xy_half + args.xy_step / 2, args.xy_step)
            gx, gy = np.meshgrid(centre[0] + axis, centre[1] + axis, indexing='ij')
            plane = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, best_z)], 1)
            out = list(pool.map(pass_images, [(args.dataset_root, q, heading, position, plane, args.polarization)
                                              for q in peaks]))
            images = {q: img for q, _, img, _ in out}
            elevations = {q: el for q, _, _, el in out}
            coherent = np.abs(sum(images.values()))
            k = int(coherent.argmax())
            values = {q: complex(images[q][k]) for q in images}
            amplitude_sum = sum(abs(v) for v in values.values())
            reference = values[min(values)]
            results[name] = dict(
                status='measured', heading_deg=heading, best_point_m=[float(v) for v in plane[k]],
                best_point_minus_table1_m=[float(v) for v in plane[k] - position],
                coherence=float(abs(sum(values.values())) / amplitude_sum),
                passes={str(q): dict(amplitude=abs(v), phase_rel_first_rad=float(np.angle(v * np.conj(reference))),
                                     amplitude_over_own_peak=abs(v) / peaks[q]['amplitude'],
                                     elevation_deg=elevations[q], own_peak_offset_m=peaks[q]['offset_m'])
                        for q, v in sorted(values.items())})
            print(name, 'coherence %.3f' % results[name]['coherence'], 'best-z %.3f' % best_z,
                  {q: round(d['phase_rel_first_rad'], 2) for q, d in results[name]['passes'].items()}, flush=True)
    measured = [r for r in results.values() if r.get('status') == 'measured']
    summary = dict(targets=len(measured), coherence=[round(r['coherence'], 3) for r in measured],
                   median_coherence=float(np.median([r['coherence'] for r in measured])) if measured else None,
                   random_phase_expectation=1 / math.sqrt(len(PASSES)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(schema='gotcha_interpass_coherence_v1', polarization=args.polarization,
                                        roles_read=['train'], summary=summary, targets=results), indent=2) + '\n')
    print('summary', summary, flush=True)


if __name__ == '__main__':
    main()
