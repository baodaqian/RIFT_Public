#!/usr/bin/env python3
"""Coherent backprojection of the MEASURED TRAIN data onto the geometry scorer's lattice (CPU, no model).

Reviewer check for the per-pass phase correction (docs/RIFT_GOTCHA_Tune.md B56-B58; user decision about 15:10,
carphase_b56 shards): if the per-pass constants are right, backprojecting every TRAIN unit of every pass coherently
should focus the car better in 3D, since the inter-pass phase is what resolves height. The image is scored with
``eval_gotcha_geometry_pvc.py --run field=NPZ:data_energy:LABEL`` against the data-frame mesh, on corrected and
uncorrected shards alike. Same readout as ``gotcha_coherent_readout_pvc.py``'s data image (I = A^H y on the lattice
centres, the unit split's TRAIN units, the middle pulse of each unit, every ``--freq-stride``-th selected bin), with an
option for every pulse. VALIDATION and TEST responses are not read.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
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
from scripts.render_b787_vs_stl import trilinear_sample_centers  # noqa: E402
from gotcha_coherent_readout_pvc import backproject  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset-root', type=Path, default=Path('/scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined'))
    p.add_argument('--shard-root', type=Path, required=True)
    p.add_argument('--region', default='camry_box_v2')
    p.add_argument('--region-config', type=Path, default=Path('rift_pvc/regions/camry_box_v2.json'))
    p.add_argument('--frequency-stride', type=int, default=2, help='dataset bin stride (as trained)')
    p.add_argument('--freq-stride', type=int, default=4, help='further stride of the selected bins in the image')
    p.add_argument('--all-pulses', action='store_true', help='every pulse of each unit instead of the middle one')
    p.add_argument('--passes', type=int, nargs='+', default=list(range(1, 9)))
    p.add_argument('--grid', type=int, default=48)
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--split-stride', type=int, default=4, help='unit split stride (4: the 578-unit split; 1: full data, F5)')
    p.add_argument('--heldout-fraction', type=float, default=0.5, help='0.5 with stride 4; 0.1 for the full-data split')
    p.add_argument('--heldout-pass', type=int, default=4)
    p.add_argument('--label', default='')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    ds = GOTCHADataset(args.dataset_root, shard_root=args.shard_root, passes=tuple(range(1, 9)), polarizations=('hh',),
                       region=load_region(args.region, args.region_config), pulses_per_sector=0,
                       frequency_stride=args.frequency_stride)
    split = apply_unit_split(ds, stride=args.split_stride, heldout_pass=args.heldout_pass,
                             heldout_fraction=args.heldout_fraction)
    extent = float(ds.region.half_extent_m)
    axis = torch.as_tensor(trilinear_sample_centers(extent, args.grid, args.grid), dtype=torch.float64)
    centres = torch.cartesian_prod(axis, axis, axis)
    image = torch.zeros(len(centres), dtype=torch.complex128)
    views = [v for v in ds.viewpoints('train') if v[0] in set(args.passes)]
    started, pulses = time.time(), 0
    with torch.no_grad():
        for n, (pass_id, sector) in enumerate(views):
            obs = list(ds.observations(pass_id, sector, 'hh'))
            chosen = obs if args.all_pulses else [obs[len(obs) // 2]]
            for o in chosen:
                keep = slice(None, None, args.freq_stride)
                o = dataclasses.replace(o, frequencies_hz=np.asarray(o.frequencies_hz)[keep],
                                        response=np.asarray(o.response)[keep])
                antenna = torch.as_tensor(ds.region.to_local(o.position_m), dtype=torch.float64)
                freqs = torch.as_tensor(np.asarray(o.frequencies_hz, dtype=np.float64))
                backproject(image, centres, antenna, float(o.reference_range_m), freqs,
                            _target(o, 'cpu').to(torch.complex128))
                pulses += 1
            if n % 100 == 0:
                print(f'{n + 1}/{len(views)} units, {time.time() - started:.0f} s', flush=True)
    shape = (args.grid,) * 3
    energy = image.abs().square().reshape(shape).numpy()
    meta = dict(schema='gotcha_data_coherent_image_v1', label=args.label, shard_root=str(args.shard_root),
                dataset_identity=ds.identity, region=ds.region.as_dict(), grid=args.grid, extent_m=extent,
                units=len(views), pulses=pulses, passes=args.passes, all_pulses=args.all_pulses,
                freq_stride=args.freq_stride, role='train', split=split['schema'],
                readout='A^H y: coherent backprojection of the measured TRAIN phase histories; energy |I|^2')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, axis_m=axis.numpy(), data_energy=energy, meta=json.dumps(meta))
    flat = np.sort(energy.ravel())[::-1]
    print(json.dumps(dict(output=str(args.output), units=len(views), pulses=pulses, seconds=round(time.time() - started),
                          peak=float(flat[0]), top100_share=float(flat[:100].sum() / flat.sum()),
                          top1000_share=float(flat[:1000].sum() / flat.sum()))), flush=True)


if __name__ == '__main__':
    main()
