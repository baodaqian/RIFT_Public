#!/usr/bin/env python3
"""How much of the range strip around a GOTCHA region is the region itself? (TRAIN data only, CPU)

Every cube-confined method fits the region's range-subspace projection, which keeps every return at
the cube's ranges: at ~10 km that is a strip a few metres deep across the whole illuminated scene,
turning with the look direction. This check images that strip per TRAIN pass-sector with all native
pulses (the full sector aperture resolves ~1 m in cross-range and aliases only beyond ~150 m) and
reports the share of the strip's image energy whose ground position falls inside the region's
footprint. A small share means the fitted target is mostly clutter that no cube model can explain.

Image: I(x) = sum_pulses sum_f y exp(+i 4 pi f (|x - a| - r0) / c), the adjoint of the native kernel,
on a ground grid at local z = 0, in look-aligned axes (range u, cross-range v). Sidelobes and folded
clutter make the share an estimate, not a decomposition.

    source .local-setup/activate-pvc.sh
    python scripts_pvc/gotcha_strip_clutter_check.py --sectors 24 --output strip.json
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


def sector_image(observations, region, *, half_range, half_cross, step, chunk=4096):
    antennas = np.stack([region.to_local(o.position_m) for o in observations])
    look = antennas.mean(0)
    u = np.array([look[0], look[1], 0.])
    u /= np.linalg.norm(u)                       # ground range, toward the platform
    v = np.array([-u[1], u[0], 0.])               # ground cross-range
    a = np.arange(-half_range, half_range + 1e-9, step)
    b = np.arange(-half_cross, half_cross + 1e-9, step)
    A, B = np.meshgrid(a, b, indexing="ij")
    grid = A[..., None] * u + B[..., None] * v    # local ground points, z = 0
    points = torch.as_tensor(grid.reshape(-1, 3))
    image = torch.zeros(len(points), dtype=torch.complex128)
    for obs, antenna in zip(observations, antennas):
        f = torch.as_tensor(obs.frequencies_hz, dtype=torch.float64)
        y = torch.as_tensor(obs.response, dtype=torch.complex128)
        antenna = torch.as_tensor(antenna)
        for start in range(0, len(points), chunk):
            x = points[start:start + chunk]
            distance = torch.linalg.vector_norm(x - antenna, dim=-1) - obs.reference_range_m
            image[start:start + chunk] += (torch.exp(1j * (4 * math.pi / C) * distance[:, None] * f[None]) * y).sum(1)
    inside = ((np.abs(grid[..., 0]) <= region.half_extent_m) & (np.abs(grid[..., 1]) <= region.half_extent_m))
    return image.abs().square().numpy().reshape(A.shape), inside, a, b


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset-root', default=os.environ.get('GOTCHA_DATA_ROOT',
                   '/scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined'))
    p.add_argument('--region', default='camry')
    p.add_argument('--polarization', default='hh')
    p.add_argument('--sectors', type=int, default=24)
    p.add_argument('--frequency-stride', type=int, default=2)
    p.add_argument('--half-range', type=float, default=7.5, help='strip half-depth (m), about the cube half-diagonal')
    p.add_argument('--half-cross', type=float, default=75.0, help='strip half-length (m) in cross-range')
    p.add_argument('--step', type=float, default=0.25)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--output', type=Path)
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    region = load_region(args.region)
    ds = GOTCHADataset(args.dataset_root, passes=range(1, 9), polarizations=(args.polarization,), region=region,
                       num_train=1500, pulses_per_sector=0, frequency_stride=args.frequency_stride)
    views = random.Random(0).sample(ds.viewpoints('train'), args.sectors)
    rows = []
    for p_id, sector in views:
        observations = list(ds.observations(p_id, sector, args.polarization))
        energy, inside, a, b = sector_image(observations, region, half_range=args.half_range,
                                            half_cross=args.half_cross, step=args.step)
        cross_profile = energy.sum(0)
        rows.append(dict(view=[p_id, sector], pulses=len(observations),
                         cube_share=float(energy[inside].sum() / energy.sum()),
                         cube_share_of_central_20m=float(energy[inside].sum() / energy[:, np.abs(b) <= 10].sum()),
                         footprint_share_of_area=float(inside.mean()),
                         peak_in_cube=bool(inside.reshape(-1)[int(energy.argmax())])))
        print(rows[-1], flush=True)
    shares = np.array([r['cube_share'] for r in rows])
    summary = dict(sectors=len(rows), median_cube_share=float(np.median(shares)), mean_cube_share=float(shares.mean()),
                   min=float(shares.min()), max=float(shares.max()),
                   footprint_share_of_area=rows[0]['footprint_share_of_area'],
                   sectors_with_peak_in_cube=int(sum(r['peak_in_cube'] for r in rows)),
                   strip=dict(half_range_m=args.half_range, half_cross_m=args.half_cross, step_m=args.step,
                              frequency_stride=args.frequency_stride, pulses='all native pulses of each TRAIN sector'))
    print('summary', summary, flush=True)
    if args.output:
        args.output.write_text(json.dumps(dict(summary=summary, sectors=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
