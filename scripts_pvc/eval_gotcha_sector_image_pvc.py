#!/usr/bin/env python3
"""GOTCHA common sector-image column: every method scored in RadarSplat's own target domain (PVC).

RadarSplat's GOTCHA target (``rift/radarsplat_gotcha.py`` ``sector_power``) is a coherent sector
matched-filter image: all pulses of a pass-sector backprojected onto a local polar lattice and
averaged coherently, then |.|^2 summed over elevation. The coherence across pulses cannot be
undone, so no per-pulse range profile can be recovered from it and RadarSplat has no exact place in
the per-pulse MF range-power column (a property of the GOTCHA frontend, not of an adapter). The
common column goes the other way (user decision 2026-09-23): each method's predicted native spectra
for the sector's pulses go through the same ``sector_power`` construction (copied, with the
prediction in place of the response; ``--method observed`` feeds the measured response and must
reproduce the cached targets) and are scored against RadarSplat's cached validation images with its
own ``evaluate`` definition:

* ``log_intensity_rel_mse`` (primary; RadarSplat's ``native_clipped_power_relative_mse``): both images
  through the release ``intensity`` mapping of the RadarSplat run (TRAIN peak of its cache, 60 dB log),
  clipped to [0, 1], target range-masked as in ``evaluate``; sum (p - t)^2 / sum t^2;
* ``linear_image_power_rel_mse``: the same on the MF power images themselves.

It is one baseline's preprocessing domain, not a neutral metric: report it with each method's own
native-domain score beside it. Floors on the same targets are recorded (zero; TRAIN-mean constant
from the cached TRAIN images; oracle constants). RadarSplat itself is scored in this domain by its
own trainer (``summary.json`` validation). Validation only: the cache holds no reserved-test images.

    python scripts_pvc/eval_gotcha_sector_image_pvc.py --method spinr --campaign-root C3 \\
        --task camry-spinr-full --checkpoint CK --radarsplat-targets C8/.../radarsplat/hh/targets \\
        --label camry_spinr_sector_image_val --out-dir OUT
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SCHEMA = 'gotcha_sector_image_scores_v1'
POLARIZATION = 'hh'


def sector_image(region, observations, responses, calibration, *, device, point_chunk=4096, frequency_chunk=256):
    """``rift.radarsplat_gotcha.sector_power`` with ``responses`` in place of the measured response.

    Same points, per-pulse phase (4 pi f (|x - antenna| - r0) / c), mean over native frequencies,
    coherent mean over the sector's pulses, |.|^2 summed over elevation. Only the execution chunks
    differ (summation order). Returns float64 (azimuth, range).
    """
    from rift.gotcha_dataset import C
    from rift.radarsplat_fidelity import polar_world_points
    points = torch.as_tensor(polar_world_points(calibration), dtype=torch.float64, device=device)
    coherent = torch.zeros(len(points), dtype=torch.complex128, device=device)
    if len(observations) != calibration['native_pulse_count'] or len(responses) != len(observations):
        raise ValueError('Sector pulse count differs from the RadarSplat calibration')
    for observation, response in zip(observations, responses):
        antenna = torch.as_tensor(region.to_local(observation.position_m), dtype=torch.float64, device=device)
        f = torch.as_tensor(observation.frequencies_hz.copy(), dtype=torch.float64, device=device)
        y = torch.as_tensor(response, dtype=torch.complex128, device=device)
        if y.shape != f.shape:
            raise ValueError('Predicted spectrum does not match the native frequencies')
        for begin in range(0, len(points), point_chunk):
            distance = (points[begin:begin + point_chunk] - antenna).norm(dim=-1) - observation.reference_range_m
            value = torch.zeros(len(distance), dtype=torch.complex128, device=device)
            for fi in range(0, len(f), frequency_chunk):
                phase = (4 * math.pi / C) * distance[:, None] * f[None, fi:fi + frequency_chunk]
                value += (torch.exp(1j * phase) * y[None, fi:fi + frequency_chunk]).sum(-1)
            coherent[begin:begin + len(value)] += value / len(f)
    shape = (len(calibration['elevation_rad']), len(calibration['azimuth_rad']), len(calibration['range_m']))
    power = (coherent / len(observations)).abs().square().reshape(shape).sum(0)
    if not torch.isfinite(power).all():
        raise ValueError('Nonfinite sector image')
    return power.cpu().numpy()


def read_cached(cache, index, role):
    """The cached target image with its identity checked against the cache's own calibration."""
    with np.load(cache.target_path(index), allow_pickle=False) as source:
        arrays = {k: source[k] for k in source.files}
    if (str(arrays['recipe_digest']) != cache.recipe_digest or str(arrays['role']) != role
            or int(arrays['view_index']) != index):
        raise ValueError(f'view {index}: cached target identity mismatch')
    for key in ('sensor_to_world', 'range_m', 'azimuth_rad', 'elevation_rad'):
        if not np.array_equal(arrays[key], cache.calibration[index][key]):
            raise ValueError(f'view {index}: cached target calibration changed ({key})')
    return arrays


def radarsplat_run(targets_root):
    """The RadarSplat run beside the cache: its config, mapping, model units and logged validation."""
    targets_root = Path(targets_root)
    recipe = json.loads((targets_root/'radarsplat_b7873200_recipe.json').read_text())
    checkpoints = targets_root.parent/'checkpoints'
    run_recipe = json.loads((checkpoints/'recipe.json').read_text())

    def find(node, key):
        if isinstance(node, dict):
            if key in node:
                return node[key]
            for value in node.values():
                found = find(value, key)
                if found is not None:
                    return found
        return None
    summary = json.loads((checkpoints/'summary.json').read_text()) if (checkpoints/'summary.json').exists() else {}
    return dict(config=recipe['config'], mapping=find(run_recipe, 'intensity_mapping'),
                units=float(find(run_recipe, 'model_units_per_m')), logged_validation=summary.get('validation'),
                checkpoints=str(checkpoints.resolve()))


class ObservedAdapter:
    """Transform check: the measured response itself (reproduces the cached targets)."""

    def __init__(self, checkpoint_path, dataset, entry, device):
        if checkpoint_path is not None:
            raise ValueError('the observed check takes no checkpoint')
        self.identity = dict(method='observed_check', prediction='the measured native response')

    def predict(self, view, position, pol, observations, readouts):
        return torch.as_tensor(np.stack([o.response for o in observations]), dtype=torch.complex128)


def parse_args(argv=None):
    from scripts_pvc.eval_gotcha_heldout_pvc import ADAPTERS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--method', required=True, choices=sorted({*ADAPTERS, 'observed'}))
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--task', required=True, help='campaign task key whose command built the run')
    parser.add_argument('--checkpoint', type=Path, help='required except for --method zero/observed')
    parser.add_argument('--radarsplat-targets', type=Path, required=True,
                        help="the RadarSplat GOTCHA run's target cache (.../radarsplat/hh/targets)")
    parser.add_argument('--label', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-sectors', type=int, default=0, help='debug cap; 0 scores every validation sector')
    args = parser.parse_args(argv)
    if (args.checkpoint is None) != (args.method in ('zero', 'observed')):
        parser.error('--checkpoint is required for a method and not accepted for zero/observed')
    return args


@torch.no_grad()
def main(argv=None):
    from rift.radarsplat_gotcha import GOTCHAPowerCache
    from rift.radarsplat_release import intensity
    from rift_pvc.gotcha_training import RangeReadout
    from scripts_pvc.eval_gotcha_heldout_pvc import ADAPTERS, load_run
    args = parse_args(argv)
    device = torch.device(args.device)
    output = args.out_dir/f'{args.label}_metrics.json'
    if output.exists():
        raise FileExistsError(output)
    dataset, entry, task = load_run(args.campaign_root, args.task)
    run = radarsplat_run(args.radarsplat_targets)
    # The constructor rebuilds the RadarSplat recipe from this dataset and refuses a cache built from another.
    cache = GOTCHAPowerCache(dataset, POLARIZATION, args.radarsplat_targets, run['config'])
    if cache.train_peak_power is None:
        raise ValueError('RadarSplat cache has no TRAIN normalization')
    peak, mapping, mask_m = cache.train_peak_power, run['mapping'], 2.5 / run['units']
    adapter = (ObservedAdapter if args.method == 'observed' else ADAPTERS[args.method])(
        args.checkpoint, dataset, entry, device)
    readout = RangeReadout(dataset.region, device=device)

    def mapped(power, ranges):
        return np.clip(intensity(power, peak, mapping), 0, 1) * (ranges >= mask_m)

    # TRAIN-mean levels (achievable constants) from the cached TRAIN images.
    train = dict(log=0.0, linear=0.0, pixels=0)
    for index in cache.train_indices:
        arrays = read_cached(cache, index, 'train')
        target = arrays['radarsplat_mf_power'].astype(np.float64) * (arrays['range_m'] >= mask_m)
        train['log'] += float(mapped(arrays['radarsplat_mf_power'].astype(np.float64), arrays['range_m']).sum())
        train['linear'] += float(target.sum())
        train['pixels'] += target.size
    levels = dict(log=train['log'] / train['pixels'], linear=train['linear'] / train['pixels'])

    names = ('error', 'energy', 'sum', 'sector_spread')
    totals = {domain: dict({k: 0.0 for k in names}, pixels=0) for domain in ('log', 'linear')}
    indices = list(cache.validation_indices)
    indices = indices[:args.max_sectors] if args.max_sectors else indices
    started = time.perf_counter()
    for position, index in enumerate(indices):
        view = cache.view_keys[index]
        observations = list(dataset.observations(*view, POLARIZATION))
        readouts = [readout.for_observation(o) for o in observations]
        prediction = adapter.predict(view, position, POLARIZATION, observations, readouts)
        predicted = sector_image(dataset.region, observations, prediction.detach().cpu(), cache.calibration[index],
                                 device=device)
        arrays = read_cached(cache, index, 'validation')
        ranges = arrays['range_m']
        measured = arrays['radarsplat_mf_power'].astype(np.float64)
        pairs = dict(log=(mapped(predicted, ranges), mapped(measured, ranges)),
                     linear=(predicted * (ranges >= mask_m), measured * (ranges >= mask_m)))
        for domain, (p, t) in pairs.items():
            a = totals[domain]
            a['error'] += float(np.square(p - t).sum())
            a['energy'] += float(np.square(t).sum())
            a['sum'] += float(t.sum())
            a['sector_spread'] += float(np.square(t - t.mean()).sum())
            a['pixels'] += t.size
        if (position + 1) % 20 == 0 or position + 1 == len(indices):
            print(f'[{position + 1:4d}/{len(indices)}] {time.perf_counter() - started:7.1f}s  '
                  f"log {totals['log']['error'] / totals['log']['energy']:.4f}  "
                  f"linear {totals['linear']['error'] / totals['linear']['energy']:.4f}", flush=True)

    def floors(a, level):
        return dict(zero=1.0, train_mean_constant=(a['energy'] - 2 * level * a['sum'] + a['pixels'] * level ** 2) / a['energy'],
                    oracle_constant=(a['energy'] - a['sum'] ** 2 / a['pixels']) / a['energy'],
                    oracle_per_sector_constant=a['sector_spread'] / a['energy'])
    result = dict(schema=SCHEMA, label=args.label, method=adapter.identity,
                  checkpoint=str(args.checkpoint.resolve()) if args.checkpoint else None,
                  campaign_root=str(args.campaign_root.resolve()), task=args.task, dataset_identity=dataset.identity,
                  polarization=POLARIZATION, selected_role='validation', reserved_test_accessed=False,
                  sectors=len(indices), status='complete' if len(indices) == len(cache.validation_indices) else 'incomplete_or_smoke',
                  log_intensity_rel_mse=totals['log']['error'] / totals['log']['energy'],
                  linear_image_power_rel_mse=totals['linear']['error'] / totals['linear']['energy'],
                  floors=dict(log_intensity=floors(totals['log'], levels['log']),
                              linear_image_power=floors(totals['linear'], levels['linear']),
                              train_levels=levels, train_sectors=len(cache.train_indices)),
                  totals=totals,
                  domain=dict(construction='rift.radarsplat_gotcha.sector_power (copied; prediction in place of response)',
                              radarsplat_targets=str(args.radarsplat_targets.resolve()), recipe_digest=cache.recipe_digest,
                              intensity_mapping=mapping, train_peak_power=peak, range_mask_m=mask_m,
                              radarsplat_logged_validation=run['logged_validation'],
                              note="RadarSplat's own preprocessing domain; report beside each method's native-domain score"),
                  elapsed_seconds=time.perf_counter() - started, device=str(device))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, default=str)
        handle.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'totals'}, indent=2, sort_keys=True, default=str))


if __name__ == '__main__':
    main()
