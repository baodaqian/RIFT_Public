#!/usr/bin/env python3
"""Coherent (backprojection) 3D readout of an adaptive-RIFT GOTCHA checkpoint (tuning campaign A52, CPU).

``eval_gotcha_geometry_pvc.py`` reads a point scene incoherently: each point's |w|^2 deposited in
its voxel. A dense coherent scene can hold energy that cancels in every look (the TRAIN operator's
null space, reviewer B52), and that readout counts it as scene. This readout images what the model
predicts instead, the way radar images a scene:

    I(v) = sum_looks sum_f  d(look, f) * exp(+i 4 pi f / c * (|v - a_look| - r0_look))

where d is either the model's predicted phase history (``model``) or the measured, isolated target
(``data``, the matched-filter backprojection reference). I = A^H d on the lattice centres, so
content the looks cannot see does not appear. The energy |I|^2 is written on the scorer's
``--grid``^3 lattice over the region cube (the same centres ``eval_gotcha_geometry_pvc.py`` uses)
and can be scored with its ``--run field=NPZ:KEY:`` option.

Looks: every TRAIN unit of the checkpoint's own split (the unit split is rebuilt from the recipe and
the dataset identity checked), one pulse per unit (the middle one), every ``--freq-stride``-th
selected bin. VALIDATION and TEST responses are not read.

    python scripts_pvc/gotcha_coherent_readout_pvc.py CKPT.pt --output OUT.npz \\
        --shard-root <keep6 shards> --region-config rift_pvc/regions/camry_box_v2.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from rift.gotcha_dataset import C, GOTCHADataset, load_region  # noqa: E402
from rift.gotcha_training import with_legacy_keys  # noqa: E402
from rift_pvc.gotcha_batched import sector_forward  # noqa: E402
from rift_pvc.gotcha_training import ChannelField, _target  # noqa: E402
from scripts.render_b787_vs_stl import trilinear_sample_centers  # noqa: E402


def rebuild_dataset(checkpoint, dataset_root, shard_root, region_config):
    contract, recipe = checkpoint['dataset_contract'], checkpoint['recipe']
    stride = int((contract.get('frequency_selection') or {}).get('stride', 1) or 1)
    selection = contract.get('training_pulse_selection') or {}
    cap = int(selection.get('pulses_per_sector', 0) or 0) if isinstance(selection, dict) else 0
    ds = GOTCHADataset(dataset_root, shard_root=shard_root, passes=contract['passes'],
                       polarizations=contract['polarizations'],
                       region=load_region(contract['region']['name'], region_config),
                       pulses_per_sector=cap, frequency_stride=stride)
    split = recipe.get('unit_split')
    if split is not None:
        from rift_pvc.gotcha_unit_split import apply_unit_split
        apply_unit_split(ds, stride=split['sector_stride'], heldout_pass=split['heldout_pass'],
                         heldout_fraction=split['heldout_fraction'], seed=split.get('seed', 42))
    if ds.identity != checkpoint['dataset_identity']:
        raise ValueError('Rebuilt dataset identity differs from the checkpoint')
    return ds


def backproject(image, centres, antenna, r0, freqs, d, chunk=65536):
    """image += sum_f d(f) exp(+i 4 pi f/c (|v - a| - r0)) for every lattice centre v."""
    k = (4 * math.pi / C) * freqs                                          # [F]
    for start in range(0, len(centres), chunk):
        v = centres[start:start + chunk]
        distance = torch.linalg.vector_norm(v - antenna, dim=-1) - r0     # [V]
        image[start:start + chunk] += (torch.exp(1j * distance[:, None] * k[None]) * d[None]).sum(-1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, default=Path('/scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined'))
    p.add_argument('--shard-root', type=Path, required=True)
    p.add_argument('--region-config', type=Path)
    p.add_argument('--grid', type=int, default=48)
    p.add_argument('--freq-stride', type=int, default=4)
    p.add_argument('--units', type=int, default=0, help='limit the TRAIN units (0 = all; for tests)')
    p.add_argument('--no-data', action='store_true', help='skip the measured-data reference image')
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--polarization', default='hh')
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    recipe = with_legacy_keys(checkpoint['recipe'], checkpoint['recipe']['method'])
    ds = rebuild_dataset(checkpoint, args.dataset_root, args.shard_root, args.region_config)
    head = ChannelField(recipe['method'], ds.region, recipe, 'cpu')
    prefix = f'{args.polarization}.'
    head.load_state_dict({k[len(prefix):]: v for k, v in checkpoint['model_state_dict'].items()
                          if k.startswith(prefix)}, strict=True)
    head.eval()
    extent = float(ds.region.half_extent_m)
    axis = torch.as_tensor(trilinear_sample_centers(extent, args.grid, args.grid), dtype=torch.float64)
    centres = torch.cartesian_prod(axis, axis, axis)
    model_image = torch.zeros(len(centres), dtype=torch.complex128)
    data_image = None if args.no_data else torch.zeros_like(model_image)
    views = ds.viewpoints('train')
    if args.units:
        views = views[:args.units]
    started = time.time()
    with torch.no_grad():
        for n, (pass_id, sector) in enumerate(views):
            obs = list(ds.observations(pass_id, sector, args.polarization))
            o = obs[len(obs) // 2]
            keep = slice(None, None, args.freq_stride)
            o = dataclasses.replace(o, frequencies_hz=np.asarray(o.frequencies_hz)[keep],
                                    response=np.asarray(o.response)[keep])
            antenna = torch.as_tensor(ds.region.to_local(o.position_m), dtype=torch.float64)
            freqs = torch.as_tensor(np.asarray(o.frequencies_hz, dtype=np.float64))
            readout = dict(antenna=antenna, frequencies=freqs)
            pred = sector_forward(head, [o], [readout], method=recipe['method'],
                                  point_chunk=recipe['point_chunk'])[0][0].to(torch.complex128)
            backproject(model_image, centres, antenna, float(o.reference_range_m), freqs, pred)
            if data_image is not None:
                backproject(data_image, centres, antenna, float(o.reference_range_m), freqs,
                            _target(o, 'cpu').to(torch.complex128))
            if n % 50 == 0:
                print(f'{n + 1}/{len(views)} units, {time.time() - started:.0f} s', flush=True)
    shape = (args.grid,) * 3
    out = dict(axis_m=axis.numpy(), model_energy=model_image.abs().square().reshape(shape).numpy(),
               meta=json.dumps(dict(schema='gotcha_coherent_readout_v1', checkpoint=str(args.checkpoint.resolve()),
                                    epoch=checkpoint.get('epoch'), region=ds.region.as_dict(), grid=args.grid,
                                    extent_m=extent, units=len(views), pulses_per_unit=1,
                                    freq_stride=args.freq_stride, role='train',
                                    readout='A^H d: coherent backprojection of predicted (model) or measured (data) '
                                            'phase histories onto the lattice centres; energy |I|^2')))
    if data_image is not None:
        out['data_energy'] = data_image.abs().square().reshape(shape).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **out)
    print(f'wrote {args.output} ({len(views)} units, {time.time() - started:.0f} s)')


if __name__ == '__main__':
    main()
