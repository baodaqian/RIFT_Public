#!/usr/bin/env python3
"""Achievable constant reference for the MF-power scores: the TRAIN-mean level, scored on a held-out role (PVC).

The held-out evaluators' ``reference_floors`` are oracle constants fitted to the scored targets
themselves. The constant a method could actually predict is the TRAIN mean. This script computes
it with the evaluators' own target definitions and scores it on the held-out targets those
evaluators already computed (for a constant c: sum (t - c)^2 / sum t^2 over the scored bins):

* ``rift-dataset``: the TRAIN mean of the normalized intensity and of the linear power over each
  TRAIN view's ROI bins (``response_view_to_range_power``; ``normalize_power_db`` with the stats'
  TRAIN peak; ``scene_range_mask`` with the evaluator's margin), scored on a held-out per-view
  cache of ``eval_rift_dataset_heldout_pvc.py`` (one Tx/Rx pair: the cached target profile must
  reproduce the stored energies; linear targets rebuilt from it, bins below -60 dB count as 0);
* ``gotcha``: per polarization, the TRAIN mean of the MF power on ``range_geometry`` (guard 2),
  normalized with the TRAIN peak, scored from a held-out metrics JSON of
  ``eval_gotcha_heldout_pvc.py`` through its per-polarization target sums (exact).

    python scripts_pvc/eval_train_constant_pvc.py rift-dataset --object a320 --dataset-root D \\
        --role-manifest RUN/role_manifest.json --stats RF/radar_fields_power_stats.json \\
        --heldout-metrics OUT/a320_spinr_val_metrics.json --output OUT/a320_train_constant_val.json
    python scripts_pvc/eval_train_constant_pvc.py gotcha --campaign-root C7 --task camry-rift-full \\
        --mf-stats OUT/gotcha_mf_train_peak.json --heldout-metrics OUT/camry_zero_val_metrics.json \\
        --output OUT/camry_train_constant_val.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SCHEMA = 'rift_train_constant_reference_v1'


def relmse(values, level):
    return float(((values - level) ** 2).sum() / (values ** 2).sum())


# ---------------------------------------------------------------- RIFT dataset
def rift_dataset_train_levels(args):
    import scripts.eval_b787_range_power as original
    from rift.radar_fields_dataset import (from_collection_arrays, normalize_power_db, range_bin_centers,
                                           response_view_to_range_power, restrict_radar_fields_response_views,
                                           scene_range_mask, validate_power_stats_acquisition)
    from rift.rift_dataset import (evaluation_role_indices, load_object_contract, object_identity,
                                   resolve_object_inputs, validate_checkpoint_object)
    from scripts_pvc.eval_b787_range_power_pvc import validate_normalization_stats
    npz_path, manifest = map(str, resolve_object_inputs(object_name=args.object, dataset_root=args.dataset_root,
                                                        npz_path=None, role_manifest_path=args.role_manifest))
    public, contract = load_object_contract(npz_path, manifest, response_roles=('train',))
    validate_checkpoint_object(object_identity(args.object), contract)
    stats = json.loads(Path(args.stats).read_text())
    validate_checkpoint_object(stats, contract)
    arrays = from_collection_arrays(public, contract)
    validate_power_stats_acquisition(stats, arrays.acquisition_identity)
    train = evaluation_role_indices(contract, 'train')
    arrays = restrict_radar_fields_response_views(arrays, train)
    peak, dynamic_range_db = validate_normalization_stats(
        stats, {'sealed_npz_protocol_contract': contract}, num_views=arrays.num_views)
    ranges = range_bin_centers(arrays.metadata, device='cpu', dtype=torch.float32)
    total_db = total_linear = 0.0
    bins = 0
    for view_index, response_view in arrays.iter_response_views(train):
        power = response_view_to_range_power(response_view, device='cpu')
        intensity = normalize_power_db(power, peak, dynamic_range_db)
        viewpoint = torch.as_tensor(arrays.viewpoint_positions[int(view_index)], dtype=torch.float32)
        roi = scene_range_mask(ranges, viewpoint, original.RF_GRID_EXTENT_M, margin=original.RF_RANGE_MARGIN_M)
        total_db += float(intensity[:, roi].sum())
        total_linear += float(power[:, roi].sum())
        bins += int(intensity[:, roi].numel())
    levels = dict(normalized=total_db / bins, linear=total_linear / bins, bins=bins, views=int(len(train)))
    return levels, contract['dataset_identity'], peak, dynamic_range_db


def score_rift_dataset(metrics_path, levels, identity, peak, dynamic_range_db):
    metrics = json.loads(Path(metrics_path).read_text())
    if metrics['dataset_identity'] != identity or metrics['selected_role'] == 'train':
        raise ValueError(f'{metrics_path}: another dataset or not a held-out role')
    if not np.isclose(metrics['peak_power'], peak, rtol=1e-12, atol=0) or metrics['dynamic_range_db'] != dynamic_range_db:
        raise ValueError(f'{metrics_path}: another MF-power normalization')
    cache = np.load(str(metrics_path).replace('_metrics.json', '_per_view.npz'))
    done = np.isfinite(cache['range_power_rel_mse'])
    t = cache['target_profile'][done].astype(np.float64)
    mask = cache['roi_mask'][done].astype(bool)
    if not (np.array_equal(mask.sum(1), cache['count_db'][done])
            and np.allclose(np.where(mask, t * t, 0).sum(1), cache['target_sq_db'][done], rtol=1e-4, atol=0)):
        raise ValueError(f'{metrics_path}: the cached profile does not reproduce the scored targets (multi-pair?)')
    scored = t[mask]
    linear = np.where(scored > 0, peak * 10.0 ** (dynamic_range_db * (scored - 1.0) / 10.0), 0.0)
    return dict(heldout_metrics=str(Path(metrics_path).resolve()), role=metrics['selected_role'],
                status=metrics['status'], views=int(done.sum()), bins=int(mask.sum()),
                normalized_range_power_rel_mse=relmse(scored, levels['normalized']),
                linear_range_power_rel_mse=relmse(linear, levels['linear']),
                oracle_best_constant=dict(normalized=relmse(scored, scored.mean()), linear=relmse(linear, linear.mean())),
                heldout_mean=dict(normalized=float(scored.mean()), linear=float(linear.mean())))


# ---------------------------------------------------------------- GOTCHA
def gotcha_train_levels(args):
    from rift.radar_fields_dataset import normalize_power_db
    from rift_pvc.radar_fields_gotcha import matched_range_power, range_geometry
    from scripts_pvc.eval_gotcha_heldout_pvc import MF_DYNAMIC_RANGE_DB, MF_GUARD_CELLS, load_run, mf_statistics
    dataset, _, _ = load_run(args.campaign_root, args.task)
    stats = mf_statistics(dataset, args.mf_stats)
    levels = {}
    for pol in dataset.polarizations:
        peak = float(stats['peak_power'][pol])
        total_db = total_linear = 0.0
        bins = pulses = 0
        for view in dataset.viewpoints('train'):
            for obs in dataset.observations(*view, pol):
                _, ranges = range_geometry(obs, dataset.region, MF_GUARD_CELLS, 'cpu')
                power = matched_range_power(obs, ranges)
                total_db += float(normalize_power_db(power, peak, MF_DYNAMIC_RANGE_DB).sum())
                total_linear += float(power.sum())
                bins += int(len(ranges))
                pulses += 1
        levels[pol] = dict(normalized=total_db / bins, linear=total_linear / bins, bins=bins, pulses=pulses)
    return levels, dataset.identity, stats['peak_power']


def score_gotcha(metrics_path, levels, identity, peaks):
    metrics = json.loads(Path(metrics_path).read_text())
    if metrics['dataset_identity'] != identity or metrics['selected_role'] == 'train':
        raise ValueError(f'{metrics_path}: another dataset or not a held-out role')
    if metrics['mf_power_definition']['peak_power'] != peaks:
        raise ValueError(f'{metrics_path}: another MF-power normalization')
    scores = {}
    for domain, prefix in (('normalized', 'db'), ('linear', 'linear')):
        error = energy = 0.0
        for pol, t in metrics['by_polarization'].items():
            if f'{prefix}_sum' not in t:
                raise ValueError(f'{metrics_path}: no target sums (scored before reference_floors existed)')
            c = levels[pol][domain]
            error += t[f'{prefix}_energy'] - 2 * c * t[f'{prefix}_sum'] + t['bins'] * c * c
            energy += t[f'{prefix}_energy']
        scores[domain] = error / energy
    return dict(heldout_metrics=str(Path(metrics_path).resolve()), role=metrics['selected_role'],
                status=metrics['status'], sectors=metrics['sectors'],
                normalized_range_power_rel_mse=scores['normalized'], linear_range_power_rel_mse=scores['linear'],
                oracle_floors=metrics.get('reference_floors'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='dataset', required=True)
    rift = sub.add_parser('rift-dataset')
    rift.add_argument('--object', required=True)
    rift.add_argument('--dataset-root', type=Path, required=True)
    rift.add_argument('--role-manifest', required=True)
    rift.add_argument('--stats', required=True, help="the object's Radar Fields power stats (TRAIN peak)")
    gotcha = sub.add_parser('gotcha')
    gotcha.add_argument('--campaign-root', type=Path, required=True)
    gotcha.add_argument('--task', required=True)
    gotcha.add_argument('--mf-stats', type=Path, required=True)
    for p in (rift, gotcha):
        p.add_argument('--heldout-metrics', type=Path, nargs='+', required=True,
                       help='held-out metrics JSON(s) whose targets the TRAIN constant is scored on')
        p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    if args.dataset == 'rift-dataset':
        levels, identity, peak, dynamic_range_db = rift_dataset_train_levels(args)
        scores = [score_rift_dataset(p, levels, identity, peak, dynamic_range_db) for p in args.heldout_metrics]
        extra = dict(object=args.object, peak_power=peak, dynamic_range_db=dynamic_range_db)
    else:
        levels, identity, peaks = gotcha_train_levels(args)
        scores = [score_gotcha(p, levels, identity, peaks) for p in args.heldout_metrics]
        extra = dict(campaign_root=str(args.campaign_root.resolve()), task=args.task, peak_power=peaks)
    result = dict(schema=SCHEMA, dataset=args.dataset, dataset_identity=identity, train_levels=levels,
                  scores=scores, elapsed_seconds=time.perf_counter() - started,
                  definition='TRAIN-mean constant (achievable); oracle_* are constants fitted to the scored targets',
                  **extra)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
