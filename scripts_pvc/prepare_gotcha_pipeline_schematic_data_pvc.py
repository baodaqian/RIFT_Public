#!/usr/bin/env python3
"""Data for the appendix schematic of RIFT on GOTCHA (fig:gotcha-rift-pipeline), CPU only, TRAIN responses only.

Everything comes from the one training run behind the paper's Toyota Camry row (a 12-epoch run extended to 40 by
``gotcha_extend_epochs.py``, which changes ``recipe['epochs']`` only):

* the backprojection initialization, recomputed with the trainer's own functions (``make_plan`` of
  ``train_gotcha_dataset_pvc.py`` with the run's CLI, then ``rift_dataset_initialization`` and ``pooled_gain_refit``
  on the first 100 pass-sectors of the seed-42 order). Its record is compared with the run's ``initialization.json``
  (views, active count, alpha, gauge, m1, m2, prior weights, warm-start gain); the job fails if they disagree;
* three saved states of the same run: before the densifications after epochs 10 and 16 (snapshots) and the selected
  epoch-40 checkpoint (frozen copy). Each is checked against the run's history (epoch, validation RelMSE);
* for every stage: active count, lattice pitch, SH orders, and a plan-view maximum-intensity projection of the point
  energy deposited (CIC, as the paper's geometry readout) on a cubic lattice at that stage's own pitch;
* the acquisition geometry: per-pass elevation and azimuth of the antenna phase centre seen from the region centre
  (TRAIN pulses), and the split's sector roles from the dataset contract (no test payload is read).

    python scripts_pvc/prepare_gotcha_pipeline_schematic_data_pvc.py --out OUT.npz
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

G = Path('/scratch/group/p.cis261724.000/RIFT_pvc_runs')
TAIL = 'camry_box_v2/809186e53fc3042d/frequency_stride2_all_roles_v2/rift_full_native'
RUN12 = G / 'rift_camry_densify/F5gA_pooled_warmstart_ep12' / TAIL
RUN40 = G / 'rift_camry_densify/F5gA_ext40' / TAIL
STAGES = (('epoch10', G / 'rift_camry_densify/snapshots/F5gA_ep10_pre.pt'),
          ('epoch16', G / 'rift_camry_densify/snapshots/F5gA_ext40_ep16_pre.pt'),
          ('epoch40', G / 'camry_paper_provisional_20260924/checkpoints/rift_F5gA_ext40_best_ep40.pt'))
SHARDS = G / 'gotcha_filtered_shards_20260923/camry_box_v2_provisional_carphase_v2/keep6/New_Transfer/shards'
# scripts_pvc/run_gotcha_rift_densify_pvc.sbatch with the run's knobs (log pvc-rift-densify-F5gA-2158832.log)
TRAINER_ARGS = [
    '--dataset-root', '/scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined', '--shard-root', str(SHARDS),
    '--region', 'camry_box_v2', '--region-config', 'rift_pvc/regions/camry_box_v2.json', '--polarizations', 'hh',
    '--passes', '1', '2', '3', '4', '5', '6', '7', '8', '--num-tx', '1', '--num-rx', '1', '--pulses-per-sector', '0',
    '--frequency-stride', '2', '--method', 'rift', '--optimizer', 'b787', '--adam-eps', '1.490932099939119',
    '--lr', '6e-4', '--pos-lr', '6e-4', '--loss-domain', 'full_native', '--support-box', '3.0', '1.5', '1.25',
    '--support-pitch', '0.125', '--sh-degree', '4', '--sh-init-degree', '0', '--max-points', '1048576',
    '--densify-epochs', '4', '10', '16', '--densify-max-active', '1048576', '--densify-lr-target', '1.0',
    '--densify-lr-factor', '1.0', '--densify-init', 'trilinear', '--densify-normalize', 'volume',
    '--densify-max-level', '3', '--train-eval-every', '2', '--gain-warm-start', 'pooled', '--unit-split-stride', '1',
    '--unit-split-heldout-pass', '4', '--unit-split-heldout-fraction', '0.1', '--device', 'cpu',
    '--point-chunk', '16384', '--checkpoint-every', '50', '--epochs', '12']
EXTENT = 3.0                    # the scorer's cube half-width (m)


def normalized(value):
    return json.loads(json.dumps(value))


def close(a, b, rtol):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(np.all(np.abs(a - b) <= rtol * np.maximum(np.abs(b), 1e-300)))


def close_complex(a, b, rtol):
    """A complex value stored as [re, im]: |a - b| <= rtol |b| (its roundoff-level part may change sign)."""
    a, b = complex(*a), complex(*b)
    return abs(a - b) <= rtol * abs(b)


def field_state(model_state_dict, pol='hh'):
    prefix = f'{pol}.field.'
    return {k[len(prefix):]: v for k, v in model_state_dict.items() if k.startswith(prefix)}


def stage_record(sd):
    """Active count, pitch(es), SH orders, higher-band energy share, and the plan-view MIP at the stage's pitch."""
    from scripts.eval_scene_geometry import deposit_points
    active = sd['active_mask']
    half = sd['cell_half'][active].double().reshape(-1)
    pitches = sorted({round(float(2 * h), 6) for h in torch.unique(half)})
    order = sd['order'][active]
    positions = (sd['anchors'][active] + sd['cell_half'][active] * torch.tanh(sd['delta_raw'][active])).double()
    if bool(sd.get('support_bounds_enabled', torch.tensor(False))):
        positions = torch.maximum(torch.minimum(positions, sd['support_max'].double()), sd['support_min'].double())
    positions = positions.clamp(-EXTENT, EXTENT)
    unlocked = sd['basis_degree'][None, :] <= order[:, None]
    squared = (sd['w_re'][active].double().square() + sd['w_im'][active].double().square()) * unlocked
    energy = squared.sum(1)
    higher = squared[:, sd['basis_degree'] >= 1].sum()
    pitch = min(pitches)
    grid = int(round(2 * EXTENT / pitch))
    volume = deposit_points(positions, energy, EXTENT, grid)          # [x, y, z] over the 6 m cube
    plan = volume.max(dim=2).values.numpy().astype(np.float32)        # maximum over z: [x, y]
    axis = (np.arange(grid) + 0.5) * pitch - EXTENT
    keep_y = np.abs(axis) <= 1.5 + 1e-9
    return dict(active=int(active.sum()), pitches_m=pitches, pitch_m=pitch, readout_grid=grid,
                order_counts={int(o): int((order == o).sum()) for o in torch.unique(order)},
                higher_band_energy_fraction=float(higher / max(float(energy.sum()), 1e-300)),
                energy_total=float(energy.sum())), plan[:, keep_y], axis, axis[keep_y]


def recompute_start():
    """The run's backprojection initialization, recomputed and checked against its initialization.json."""
    import torch.nn as nn
    import train_gotcha_dataset_pvc as trainer
    from rift_pvc import gotcha_nufft as nufft
    from rift_pvc.gotcha_training import (BACKPROJECTION, ChannelField, RangeReadout, b787_schedule,
                                          fixed_view_order, pooled_gain_refit, rift_dataset_initialization)
    with tempfile.TemporaryDirectory() as tmp:
        args = trainer.parse_args(TRAINER_ARGS + ['--output-root', tmp])
        dataset, plan = trainer.make_plan(args)
    (entry,) = plan['plans']
    recipe = entry['config']
    saved_recipe = json.loads((RUN12 / 'recipe.json').read_text())
    diff = sorted(k for k in set(recipe) | set(saved_recipe) if normalized(recipe.get(k)) != normalized(saved_recipe.get(k)))
    if diff:
        raise SystemExit(f'recipe differs from the run: {diff}')
    if normalized(dataset.contract) != normalized(json.loads((RUN12 / 'dataset.json').read_text())['contract']):
        raise SystemExit('dataset contract differs from the run')
    assert recipe['initialization'] == BACKPROJECTION
    torch.manual_seed(recipe['seed'])
    rng = np.random.Generator(np.random.PCG64(recipe['seed']))
    heads = nn.ModuleDict({pol: ChannelField('rift', dataset.region, recipe, 'cpu') for pol in dataset.polarizations})
    nufft.attach_grids(heads, dataset, recipe, 'cpu')
    readout = RangeReadout(dataset.region, device='cpu')
    train_views = dataset.viewpoints('train')
    order = rng.permutation(len(train_views)).tolist()
    if b787_schedule('rift', recipe) is not None and order != fixed_view_order(recipe['seed'], len(train_views)):
        raise AssertionError('epoch-1 order differs from the fixed training order')
    views = [train_views[i] for i in order[:recipe['bp_views']]]
    with torch.no_grad():
        record = {pol: rift_dataset_initialization(head, dataset, readout, views, pol) for pol, head in heads.items()}
        record = {pol: pooled_gain_refit(head, dataset, readout, views, pol, record[pol]) for pol, head in heads.items()}
    saved = json.loads((RUN12 / 'initialization.json').read_text())
    checks = {}
    for pol, rec in record.items():
        ref = saved[pol]
        checks[pol] = dict(
            views=rec['views'] == ref['views'], pulses=rec['pulses'] == ref['pulses'],
            initial_points=rec['initial_points'] == ref['initial_points'],
            alpha=close_complex(rec['alpha'], ref['alpha'], 1e-6), coefficient_gauge=close(rec['coefficient_gauge'],
                                                                                   ref['coefficient_gauge'], 1e-6),
            m1=close(rec['m1'], ref['m1'], 1e-6), m2=close(rec['m2'], ref['m2'], 1e-6),
            l1_weight=close(rec['l1_weight'], ref['l1_weight'], 1e-6),
            sh_degree_weight=close(rec['sh_degree_weight'], ref['sh_degree_weight'], 1e-6),
            gain_after=close_complex(rec['gain_warm_start']['gain_after'],
                                     ref['gain_warm_start']['gain_after'], 1e-5))
    report = dict(checks=checks, recomputed={pol: {k: rec[k] for k in ('alpha', 'coefficient_gauge', 'm1', 'm2',
                                                                         'initial_points', 'pulses')}
                                             for pol, rec in record.items()},
                  recorded={pol: {k: saved[pol][k] for k in ('alpha', 'coefficient_gauge', 'm1', 'm2',
                                                               'initial_points', 'pulses')} for pol in saved})
    print(json.dumps(report, indent=1))
    if not all(all(c.values()) for c in checks.values()):
        raise SystemExit('recomputed initialization disagrees with the run record')
    return dataset, heads.state_dict(), report


def acquisition(dataset):
    """Per-pass antenna elevation/azimuth from the region centre (every 8th TRAIN sector, all its pulses)."""
    passes = {}
    for p in dataset.passes:
        sectors = sorted({s for q, s in dataset.viewpoints('train') if q == p})[::8]
        el, az = [], []
        for s in sectors:
            for obs in dataset.observations(p, s, 'hh'):
                a = dataset.region.to_local(obs.position_m)
                el.append(math.degrees(math.atan2(a[2], math.hypot(a[0], a[1]))))
                az.append(math.degrees(math.atan2(a[1], a[0])))
                rng_m = float(np.linalg.norm(a))
        passes[int(p)] = dict(elevation_deg=[float(np.min(el)), float(np.mean(el)), float(np.max(el))],
                              range_m=rng_m, sectors_sampled=len(sectors), pulses_sampled=len(el),
                              azimuth_deg_sample=[float(x) for x in az[::20][:40]])
    split = dataset.contract['split']
    return dict(passes=passes, heldout_pass=split['heldout_pass'], heldout_sector_ids=split['heldout_sector_ids'],
                sealed_test_sector_ids=split['sealed_test_sector_ids'], selected_sector_ids=split['selected_sector_ids'],
                units=split['units'])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    history = {r['epoch']: r for r in json.loads((RUN40 / 'history.json').read_text())}
    dataset, start_state, start_report = recompute_start()
    geometry = acquisition(dataset)
    print(json.dumps({k: v for k, v in geometry.items() if k == 'passes'}, indent=1))
    arrays, meta = {}, dict(start=start_report, acquisition=geometry, stages={})
    record, plan, _, y_axis = stage_record(field_state(start_state))
    arrays['plan_epoch0'], arrays['y_epoch0'] = plan, y_axis
    meta['stages']['epoch0'] = dict(record, source='recomputed backprojection initialization', epoch=0)
    for name, path in STAGES:
        ck = torch.load(path, map_location='cpu', weights_only=False)
        epoch = int(ck['epoch'])
        val = float(ck['history'][-1]['validation']['selection_rel_mse'])
        if ck['cursor'] != 0 or val != history[epoch]['validation']['selection_rel_mse']:
            raise SystemExit(f'{path}: not the run\'s end-of-epoch-{epoch} state')
        if normalized(ck['dataset_contract']) != normalized(dataset.contract):
            raise SystemExit(f'{path}: dataset contract differs')
        record, plan, _, y_axis = stage_record(field_state(ck['model_state_dict']))
        arrays[f'plan_{name}'], arrays[f'y_{name}'] = plan, y_axis
        meta['stages'][name] = dict(record, source=str(path), epoch=epoch, validation_rel_mse=val)
        del ck
    for name, rec in meta['stages'].items():
        print(name, {k: rec[k] for k in ('epoch', 'active', 'pitches_m', 'order_counts', 'higher_band_energy_fraction')})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, meta=json.dumps(meta), **arrays)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
