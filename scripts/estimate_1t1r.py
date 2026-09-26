#!/usr/bin/env python3
"""1t1r / 2400-1500 resource scenarios; arithmetic by default, never training.

Exact inventory: seed-42 subset metadata, collection Tx0/Rx0, native HH passes1-8.
Costs/envelopes inherit the old calculators and are unmeasured hypotheses.
Historical scene budgets only: these numbers are stale for current G48 midpoint,
MF48, 112000-Gaussian, G40 and native RIFT G48 settings. RF is unchanged.
GeRaF adds lazy MF corner work and an explicit .5--6 s/update neural/host term.
--verify-metadata requires local data/Torch, forbids response reads, and checks
all six collection geometries plus native pulse/frequency counts.
"""
import argparse
import json
from math import ceil
from pathlib import Path

import estimate_parallelization as old
import estimate_rf_geraf_radarsplat as baseline
import estimate_storage as storage

GIB = 2**30
MIB = 2**20
GPUS = old.GPUS
CARDS = ('V100 16 GB', 'A30 24 GB', 'A100 40 GB', 'A40 48 GB',
         'L40S 48 GB', 'H100 80 GB SXM', 'H200 141 GB SXM')
NATIVE_TILES = (512, 1024, 2048, 4096, 4096, 8192, 15360)
FREQUENCIES = (424, 426, 428, 428, 428, 430, 434, 432)
TRAIN_PULSES = (21922, 22039, 22184, 22213, 22248, 22177, 22374, 22448)
VAL_PULSES = (6444, 6484, 6488, 6504, 6508, 6519, 6579, 6574)
TRAIN_SAMPLES = sum(p*f for p, f in zip(TRAIN_PULSES, FREQUENCIES))
VAL_SAMPLES = sum(p*f for p, f in zip(VAL_PULSES, FREQUENCIES))
TRACE = 807*64
CORNERS = 8*TRACE  # Unmasked, undeduplicated upper-work scenario, not an observed count.
MODELS = ('RIFT', 'SpINR', 'SE Stage 1', 'Radar Fields', 'GeRaF', 'RadarSplat')


def native_blocks(points, pulses):
    return sum(p*ceil(points/min(4096, 1048576//f)) for p, f in zip(pulses, FREQUENCIES))


def time_seconds(gpu, scenario, model, dataset, *, validation=False, capacity=False):
    collection = dataset == 'Collection'
    views = (1000 if collection else 440) if validation else (2400 if collection else 1500)
    pulses = VAL_PULSES if validation else TRAIN_PULSES
    samples = VAL_SAMPLES if validation else TRAIN_SAMPLES
    if model == 'RIFT':
        points = 262144 if capacity else (110592 if collection else 32768)
        if collection:
            return (1 if validation else 3.1875)*old.physics(
                gpu, scenario, 'range', views*points, views)  # One full-scene point tile, one pair.
        return (1 if validation else 5.3)*old.physics(
            gpu, scenario, 'native', points*samples, sum(pulses))
    if model == 'SpINR':
        if collection:
            if validation:
                # Full validation plus exact first-128 TRAIN diagnostic geometry.
                entries, records = 11013+1407, 1128
                return (old.physics(gpu, scenario, 'direct_bins', old.Q*entries, records*108)
                        + old.physics(gpu, scenario, 'range', records*old.Q, records*108)
                        + old.neural(gpu, scenario, 2, 4096, training=False))
            return (2*old.physics(gpu, scenario, 'direct_bins', old.Q*26371, 2400*108)
                    + old.neural(gpu, scenario, 600, 4096))
        points = old.GOTCHA_POINTS
        return ((1 if validation else 2)*old.physics(gpu, scenario,
                    'native' if validation else 'native_fft', points*samples, native_blocks(points, pulses))
                + old.neural(gpu, scenario, 1 if validation else ceil(sum(pulses)/1024),
                             gpu.neural_tile, training=not validation, points=points))
    if model == 'SE Stage 1':
        points = 343 if collection else 74088
        entries = points*(views*600 if collection else samples)
        blocks = views if collection else native_blocks(points, pulses)
        return (1 if validation else 7)*old.physics(gpu, scenario, 'se', entries, blocks)
    if model == 'GeRaF':
        # Trace forward/replay/backward + predicted MF forward/response adjoint.
        # One collection bank refreshes every pair; native's active bank is half.
        multiplier = 2 if validation else (5 if collection else 3.5)
        tile = 65536 if collection else NATIVE_TILES[GPUS.index(gpu)]
        entries = (multiplier*TRACE+CORNERS)*(views if collection else samples)
        blocks = views*(multiplier*ceil(TRACE/tile)+ceil(CORNERS/tile))
        physics = old.physics(gpu, scenario, 'range' if collection else 'native', entries, blocks)
        fixed = .5 if scenario == old.SCENARIOS[0] else 6.
        return physics+views*fixed
    # Preserve old measured-neither service assumptions; no fabricated new-card rate.
    index = next((i for i, name in enumerate(baseline.CARDS)
                  if name.split()[0] == gpu.name.split()[0]), None)
    if index is None:
        return None
    name = 'RIFT' if collection else 'GOTCHA HH'
    endpoint = old.SCENARIOS.index(scenario)
    services = ((baseline.RF_VAL if validation else baseline.RF_TRAIN)
                if model == 'Radar Fields' else (baseline.RS_VAL if validation else baseline.RS_TRAIN))
    rates = services[name][index]
    if rates is None:
        return None
    if model == 'Radar Fields':
        if validation and not collection:
            return sum(pulses)*rates[endpoint]+views*(.001, .015)[endpoint]
        return (views if validation else views//10)*rates[endpoint]
    return views*rates[endpoint]


def times(model, dataset, gpu, **kwargs):
    return [time_seconds(gpu, s, model, dataset, **kwargs) for s in old.SCENARIOS]


def interval(values, *, unit=None):
    if values[0] is None:
        return '—'
    assert 0 < values[0] <= values[1]
    if unit is None:
        unit = 'h' if values[1] >= 7200 else ('min' if values[1] >= 60 else 's')
    divisor = dict(h=3600, min=60, s=1)[unit]
    return '–'.join(f"{float(f'{v/divisor:.2g}'):g}" for v in values)+' '+unit


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
                      '| '+' | '.join(['---']*len(headers))+' |']
                     + ['| '+' | '.join(map(str, row))+' |' for row in rows])


def timing_table(*, validation=False):
    rows = []
    for model in MODELS:
        for dataset in ('Collection', 'GOTCHA'):
            values = [interval(times(model, dataset, gpu, validation=validation)) for gpu in GPUS]
            if model == 'Radar Fields':
                values[0] = 'Blocked'
            rows.append([model+' / '+dataset, *values])
    return table(['Model / dataset', *CARDS], rows)


def capacity_rows():
    return [
        ['RIFT collection point / pair tile', *['262144 / 1']*7],
        ['RIFT collection envelope, GiB', *[f'{1+960*262144/GIB:.3f}']*7],
        ['RIFT native point tile / envelope, GiB', *['262144 / 11.234']*7],
        ['SpINR collection neural / physics / pair tiles', *['4096 / 65536 / 1']*7],
        ['SpINR collection neural envelope, GiB', *['1.625']*7],
        ['SpINR native neural tile', *[f'{g.neural_tile:,}' for g in GPUS]],
        ['SpINR native neural envelope, GiB', *[f'{1.5+32768*g.neural_tile/GIB:.3f}' for g in GPUS]],
        ['GeRaF collection point / pair tile; envelope GiB', *[f'65536 / 1; {8+1024*TRACE/GIB:.3f}']*7],
        ['GeRaF native point tile; pair request 120', *[f'{t:,}' for t in NATIVE_TILES]],
        ['GeRaF native working envelope, decimal GB', *[f'{(8*GIB+128*t*120*434)/1e9:.2f}' for t in NATIVE_TILES]],
        ['RF collection / native envelope, GiB', 'Blocked', *['2.01 / 4.53']*6],
        ['RadarSplat allowed incidences/product, millions',
         *[f'{(.8*c*1e9-GIB)/(256*6)/1e6:.2f}' for c in (16,24,40,48,48,80,141)]],
        ['SE Stage-1 collection / native envelope, GiB', *[f'{1+96*343*600/GIB:.3f} / 1.094']*7],
    ]


def storage_rows():
    rows = []
    for row in storage.rows():
        row = dict(row)
        if row['model'] == 'SpINR' and row['dataset'] == 'GOTCHA':
            # Scale view-index/exposure histories by selected sectors, keeping metadata allowance.
            cp = ceil((storage.SPINR_TENSORS+.75*storage.NATIVE_HISTORY+MIB)/MIB)
            history = ceil((.75*storage.NATIVE_HISTORY_JSON+MIB)/MIB)
            row.update(checkpoint_budget_mib=cp, retained_budget_mib=3*cp+history,
                       atomic_peak_budget_mib=3*cp+2*history)
        ram = 3400*600*8 if row['dataset'] == 'Collection' and row['model'] != 'SE' else 0
        if row['model'] == 'SE':
            ram = 2*72*(7 if row['dataset'] == 'Collection' else 42)**3*16
        rows.append(dict(model=row['model'], dataset=row['dataset'], host_payload_bytes=ram,
            disk_cache_bytes=0, checkpoint_bytes=row['checkpoint_budget_mib']*MIB,
            retained_files=row['retained_checkpoints'], retained_bytes=row['retained_budget_mib']*MIB,
            atomic_peak_bytes=row['atomic_peak_budget_mib']*MIB))
    for model in ('Radar Fields', 'GeRaF', 'RadarSplat'):
        for dataset in ('Collection', 'GOTCHA'):
            collection = dataset == 'Collection'
            if model == 'Radar Fields':
                ram, cache, cp, files = 0, 200000 if collection else 0, 210000000, 3
            elif model == 'GeRaF':
                ram, cache = 16*(2400*600 if collection else TRAIN_SAMPLES), 4*101**3+128
                cp, files = (60000000 if collection else 1240000000), 2
            else:
                # Same raster dimensions/20k Gaussians; allow original calibration/metadata overhead.
                ram, cache, cp, files = (30e6 if collection else 90e6), (29e6 if collection else 47e6), 35e6, 2
            rows.append(dict(model=model, dataset=dataset, host_payload_bytes=ram, disk_cache_bytes=cache,
                             checkpoint_bytes=cp, retained_files=files, retained_bytes=cp*files,
                             atomic_peak_bytes=cache+cp*(files+1)))
    return rows


def storage_table():
    return table(['Model / dataset', 'Host payload MB', 'Disk cache MB', 'Checkpoint MB × files',
                  'Retained MB', 'Cache + atomic peak MB'],
        [[r['model']+' / '+r['dataset'], f"{r['host_payload_bytes']/1e6:.2f}",
          f"{r['disk_cache_bytes']/1e6:.2f}", f"{r['checkpoint_bytes']/1e6:.1f} × {r['retained_files']}",
          f"{r['retained_bytes']/1e6:.1f}", f"{r['atomic_peak_bytes']/1e6:.1f}"] for r in storage_rows()])


def verify_metadata():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import torch
    from rift import npz_dataset
    from rift.rift_dataset import load_object, catalog
    from rift.gotcha_dataset import GOTCHADataset, NativeShardReader
    from rift.spinr_fidelity import scene_range_bin_mask
    def deny(*a, **k):
        raise AssertionError('Resource arithmetic must not read responses')
    npz_dataset.get_npz_response_view = deny
    npz_dataset._LazyResponseReader.response_view = deny
    npz_dataset._LazyResponseReader.iter_response_views = deny
    NativeShardReader.read = deny
    arrays, contract = load_object('a320', num_train=2400, num_tx=1, num_rx=1)
    for spec in catalog()['objects']:
        other, _ = load_object(spec['object_id'], num_train=2400, num_tx=1, num_rx=1)
        for name in ('tx_pos', 'rx_pos'):
            assert np.array_equal(arrays[name], other[name])
    frequencies = torch.as_tensor(npz_dataset.build_freqs(arrays['meta']))
    for role, expected in [('train', 26371), ('validation', 11013)]:
        bins = [int(scene_range_bin_mask(frequencies, torch.as_tensor(arrays['rx_pos'][v]),
                                        torch.as_tensor(arrays['tx_pos'][v])).sum())
                for v in contract['role_ids'][role]]
        assert sum(bins) == expected and min(bins) == 8 and max(bins) == 13
        if role == 'train':
            assert sum(bins[:128]) == 1407
    dataset = GOTCHADataset(num_train=1500)
    for (p, _), shard in dataset.shards.items():
        assert len(shard.frequencies_hz) == FREQUENCIES[p-1]
        assert np.count_nonzero(shard.row_roles == 'train') == TRAIN_PULSES[p-1]
        assert np.count_nonzero(shard.row_roles == 'validation') == VAL_PULSES[p-1]
        assert shard.response_reads == 0
    return 'Verified six-object Tx0/Rx0 geometry and selected native HH counts; zero response reads.'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--verify-metadata', action='store_true')
    args = parser.parse_args()
    verification = verify_metadata() if args.verify_metadata else None
    if args.json:
        cases = [dict(model=m, dataset=d, gpu=g.name, train_seconds=times(m,d,g),
                      validation_seconds=times(m,d,g,validation=True))
                 for m in MODELS for d in ('Collection','GOTCHA') for g in GPUS]
        print(json.dumps(dict(status='historical_scene_budgets_stale_for_current_defaults',
            scene_budget_contract='docs/SCENE_BUDGET.md', current_defaults_supported=False,
            verification=verification,
            collection_train_samples=1440000, native_train_samples=TRAIN_SAMPLES,
            native_train_pulses=sum(TRAIN_PULSES), cases=cases, storage=storage_rows()), indent=2))
    else:
        print('STALE for current scene defaults: historical 1t1r / 2400-1500 scenarios. '
              'See docs/SCENE_BUDGET.md; RF settings are unchanged.\n')
        print(timing_table())
        print('\nOne validation event:\n'+timing_table(validation=True))
        print('\nGPU envelopes:\n'+table(['Setting', *CARDS], capacity_rows()))
        print('\nStorage: decimal MB; payloads/allowances, not total process memory:\n'+storage_table())
        if verification:
            print('\n'+verification)


if __name__ == '__main__':
    main()
