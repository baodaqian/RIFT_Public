#!/usr/bin/env python3
"""Per-update cost of the GOTCHA adaptive-RIFT NUFFT/full-native control against the default recipe.

Builds the pinned Camry production acquisition (HH, passes 1-8, 1500 train,
cap 16, stride 2, point chunk 16384) for both recipes and times the batched
sector update (response read, readouts, forward, backward with the per-pulse
refinement statistics, AdamW step) over the same TRAIN pass-sectors on the
active accelerator. The scene is the G48 start grid with small random
coefficients (the kernels' cost does not depend on the values). Reads TRAIN
responses only; nothing is written.
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time

import torch

import train_gotcha_dataset_pvc as cli
from rift_pvc import accelerator
from rift_pvc import gotcha_nufft as nufft
from rift_pvc import gotcha_training as pvc
from rift_pvc.gotcha_batched import sector_update

DATA = '/scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined'
CONTROL = ['--forward-evaluation', 'nufft', '--loss-domain', 'full_native']


def synchronize():
    if accelerator.is_available():
        accelerator.synchronize()


def bench(extra, updates, device, output):
    args = cli.parse_args(['--dataset-root', DATA, '--output-root', output, '--region', 'camry',
                           '--polarizations', 'hh', '--passes', *map(str, range(1, 9)), '--num-train', '1500',
                           '--num-tx', '1', '--num-rx', '1', '--pulses-per-sector', '16',
                           '--frequency-stride', '2', '--method', 'rift', '--device', device,
                           '--point-chunk', '16384', *extra])
    dataset, plan = cli.make_plan(args)
    recipe = plan['plans'][0]['config']
    head = pvc.ChannelField('rift', dataset.region, recipe, device)
    heads = torch.nn.ModuleDict({'hh': head})
    nufft.attach_grids(heads, dataset, recipe, device)
    with torch.no_grad():
        mask = head.field.active_mask
        head.field.w_re[mask, 0] = 1e-3 * torch.randn(int(mask.sum()), device=head.field.w_re.device)
        head.field.w_im[mask, 0] = 1e-3 * torch.randn(int(mask.sum()), device=head.field.w_im.device)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=1e-3, eps=recipe['adam_eps'], weight_decay=0)
    readout = pvc.RangeReadout(dataset.region, device=device)
    times = []
    for i, (p, sector) in enumerate(dataset.viewpoints('train')[:updates]):
        synchronize()
        start = time.perf_counter()
        obs = list(dataset.observations(p, sector, 'hh'))
        rs = [readout.for_observation(o) for o in obs]
        targets = torch.stack([pvc._target(o, device) for o in obs])
        optimizer.zero_grad(set_to_none=True)
        losses, delta_grad, angular = sector_update(head, obs, rs, targets, 1.0, len(obs), method='rift',
                                                    point_chunk=recipe['point_chunk'], probe=(i % 10 == 0))
        optimizer.step()
        synchronize()
        times.append(time.perf_counter() - start)
    steady = times[3:]
    return dict(forward_evaluation=nufft.forward_evaluation(recipe), loss_domain=nufft.loss_domain(recipe),
                active_points=int(head.field.active_mask.sum()), updates=len(times),
                median_s=statistics.median(steady), mean_s=statistics.fmean(steady), first_s=times[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--updates', type=int, default=40)
    parser.add_argument('--device', default='xpu')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as output:
        results = [bench([], args.updates, args.device, output),
                   bench(CONTROL, args.updates, args.device, output)]
    results.append(dict(ratio_control_over_default=results[1]['median_s'] / results[0]['median_s']))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
