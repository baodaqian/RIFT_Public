#!/usr/bin/env python3
"""Held-out (validation or reserved-test) signal scoring on GOTCHA Camry for the compared methods (PVC).

Scores, pooled over every selected pulse and native bin of the role, per the
RIFT-dataset evaluator's definitions (user decision 2026-09-22), in their GOTCHA form:

* coherent complex RelMSE ``full_native_complex_rel_mse`` = sum |y_pred - y|^2 / sum |y|^2
  over the selected native bins (the RIFT-dataset evaluator's coherent score), plus the
  ROI range-subspace projected RelMSE the GOTCHA trainers select on
  (``roi_projected_complex_rel_mse``), reported alongside;
* MF (range) power RelMSE: the RIFT-dataset evaluator's range power is Radar Fields'
  target conversion; on GOTCHA's nonuniform native frequencies that conversion is
  Radar Fields' ``matched_range_power`` |sum_f y_f exp(+i4pi f(r-r0)/c)/N_f|^2 on its
  Rayleigh range grid over the Camry cube's range interval (2 guard cells). Both sides
  go through ``normalize_power_db`` with the TRAIN peak over every selected TRAIN pulse
  (per polarization; ``radar_fields_gotcha.training_statistics``) and 60 dB:
  ``normalized_range_power_rel_mse``; the linear variant ``linear_range_power_rel_mse``.

The dataset is rebuilt from the run's own campaign command, so its contract equals the
checkpoint's. The reserved test pass-sectors stay sealed unless ``--allow-reserved-test``
is given with ``--role test``: the already-selected test pulses (same 16-pulse cap and
frequency stride as training) are then made readable, and the dataset contract and
identity are unchanged. One adapter per method produces each pass-sector's predicted
native spectra and checks its checkpoint against the dataset.

    python scripts_pvc/eval_gotcha_heldout_pvc.py --method rift --campaign-root ROOT --task camry-rift-full \\
        --checkpoint RUN/checkpoint_best.pt --label camry_rift_val --out-dir OUT [--role test --allow-reserved-test]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SCHEMA = 'gotcha_heldout_signal_scores_v1'
MF_GUARD_CELLS, MF_DYNAMIC_RANGE_DB = 2, 60.0


# ---------------------------------------------------------------- reserved-test access
def open_reserved_test(dataset):
    """Explicit evaluation-time opt-in: make the already-selected TEST pulses readable.

    ``apply_training_selection`` has already applied the fixed pulse cap to every role and
    labelled the selected test rows ``test``; only the shard readers' role gate and
    ``viewpoints`` keep them sealed. The dataset contract and identity are not touched.
    """
    for shard in dataset.shards.values():
        shard.roles = frozenset(set(shard.roles) | {'test'})
    dataset.reserved_test_opened = True


def role_viewpoints(dataset, role):
    if role == 'test':
        if not getattr(dataset, 'reserved_test_opened', False):
            raise PermissionError('Reserved test pass-sectors are sealed; pass --allow-reserved-test')
        return [(p, sector) for p in dataset.passes for sector in dataset.splits_by_pass[p]['test']]
    return dataset.viewpoints(role)


def load_run(campaign_root, task_key):
    """The dataset and plan entry of a recorded campaign task, rebuilt from its command (CPU)."""
    import train_gotcha_dataset_pvc as cli
    campaign = json.loads((Path(campaign_root)/'campaign.json').read_text())
    task = next(t for t in campaign['tasks'] if t['key'] == task_key)
    command = task['command']
    argv = command[command.index(next(a for a in command if a.endswith('train_gotcha_dataset_pvc.py'))) + 1:]
    argv = [a for a in argv]
    if '--device' in argv:
        argv[argv.index('--device') + 1] = 'cpu'
    argv = [a for i, a in enumerate(argv) if a != '--resume' and (i == 0 or argv[i - 1] != '--resume')]
    args = cli.parse_args(argv)
    dataset, plan = cli.make_plan(args)
    return dataset, plan['plans'][0], task


# ---------------------------------------------------------------- MF-power normalization
def mf_statistics(dataset, cache_path):
    """TRAIN-only peak matched-range power per polarization (Radar Fields' GOTCHA normalization)."""
    from rift_pvc.radar_fields_gotcha import training_statistics
    cache_path = Path(cache_path)
    if cache_path.exists():
        stats = json.loads(cache_path.read_text())
        if stats.get('dataset_identity') == dataset.identity and stats.get('guard_cells') == MF_GUARD_CELLS:
            return stats
        raise ValueError(f'{cache_path} belongs to another dataset/definition')
    stats = training_statistics(dataset, dict(controls=dict(range_guard_cells=MF_GUARD_CELLS,
                                                           dynamic_range_db=MF_DYNAMIC_RANGE_DB)))
    stats = dict(stats, guard_cells=MF_GUARD_CELLS, definition='rift_pvc.radar_fields_gotcha.training_statistics')
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + '\n')
    return stats


# ---------------------------------------------------------------- adapters
class RiftAdapter:
    """GOTCHA adaptive RIFT (default recipe or the NUFFT/full-native control): its batched sector render."""

    def __init__(self, checkpoint_path, dataset, entry, device):
        from rift.gotcha_dataset import validate_checkpoint
        from rift.gotcha_training import with_legacy_keys
        from rift_pvc import gotcha_nufft as nufft
        from rift_pvc import gotcha_training as pvc
        from rift_pvc.gotcha_batched import sector_forward
        ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        recipe = with_legacy_keys(ck['recipe'], 'rift')
        ck['recipe'] = recipe
        validate_checkpoint(ck, dataset, recipe)          # dataset contract/identity equal the checkpoint's
        self.heads = torch.nn.ModuleDict({pol: pvc.ChannelField('rift', dataset.region, recipe, device)
                                          for pol in dataset.polarizations})
        self.heads.load_state_dict(ck['model_state_dict'], strict=True)
        nufft.attach_grids(self.heads, dataset, recipe, device)
        self.heads.eval()
        self.forward, self.recipe = sector_forward, recipe
        history = ck.get('history') or []
        self.identity = dict(method='rift', epoch=int(ck['epoch']), updates=int(ck['updates']),
                             forward_evaluation=nufft.forward_evaluation(recipe), loss_domain=nufft.loss_domain(recipe),
                             range_model=recipe.get('range_model'), optimizer=recipe.get('optimizer'),
                             logged_validation=(history[-1]['validation'] if history else None))

    def predict(self, view, position, pol, observations, readouts):
        prediction, _ = self.forward(self.heads[pol], observations, readouts, method='rift',
                                     point_chunk=self.recipe['point_chunk'])
        return prediction


class SpinrAdapter:
    """GOTCHA SpINR: the fixed field per polarization rendered by its batched native kernel (all bins)."""

    def __init__(self, checkpoint_path, dataset, entry, device):
        from rift.spinr_style import SpinrStyleINR, gauss_legendre_cell_grid
        from rift_pvc.spinr_gotcha_training import BatchedNativeKernel
        from train import load_tensor_checkpoint
        from train_spinr_style_pvc import evaluate_neural_field_tiled
        ck = load_tensor_checkpoint(Path(checkpoint_path), map_location='cpu')
        if ck.get('dataset_contract') != dataset.contract or ck.get('dataset_identity') != dataset.identity:
            raise ValueError('SpINR checkpoint belongs to another GOTCHA dataset')
        recipe = ck['recipe']
        heads = torch.nn.ModuleDict({p: SpinrStyleINR(support_m=dataset.region.half_extent_m)
                                     for p in dataset.polarizations}).to(device)
        heads.load_state_dict(ck['model_state_dict'], strict=True)
        heads.eval()
        self.points, self.volumes = gauss_legendre_cell_grid(recipe['grid_size'], nodes_per_cell=recipe['nodes_per_cell'],
                                                             support_m=dataset.region.half_extent_m, device=device,
                                                             dtype=torch.float64)
        self.fields = {pol: evaluate_neural_field_tiled(heads[pol], self.points,
                                                        neural_point_tile=recipe['neural_point_tile'])
                       for pol in dataset.polarizations}
        self.scales, self.kernel, self.recipe = ck['initial_scales'], BatchedNativeKernel, recipe
        self.region, self.device = dataset.region, device
        history = ck.get('history') or []
        self.identity = dict(method='spinr', epoch=int(ck['epoch']), best_epoch=ck.get('best_epoch'),
                             best_val=ck.get('best_val'), grid_size=recipe['grid_size'],
                             nodes_per_cell=recipe['nodes_per_cell'],
                             logged_validation=next((h['validation'] for h in history
                                                     if h.get('epoch') == ck.get('epoch') and h.get('validation')), None))

    def predict(self, view, position, pol, observations, readouts):
        kernel = self.kernel(observations, self.region, device=self.device, point_tile=self.recipe['renderer_point_tile'])
        return kernel.render(self.points, self.fields[pol], self.volumes, self.scales[pol]['value'], selected=False)


class GerafAdapter:
    """GOTCHA GeRaF (source_v1): the selected model's unmasked ``predict_native``, seeded per view as its validation.

    The pass-sector's acquisition is built from metadata alone (``GOTCHASourceData.acquisition``
    without its train/validation role gate); rays come from GeRaF's own samplers, so the
    prediction never reads the held-out response.
    """

    def __init__(self, checkpoint_path, dataset, entry, device):
        from rift.geraf_source import DEFAULTS, LEGACY_LIGHT_POWER_START, LEGACY_RECEIVER_GEOMETRY
        from rift.geraf_source_data import GOTCHASourceData
        from rift_pvc.geraf_source import fixed_numpy_seed, predict_native, recipe_for_data, sample_frame
        from rift_pvc.geraf_source_training import load_selected_models
        self.data = GOTCHASourceData(dataset)
        ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        models, selected = load_selected_models(ck, self.data, device)
        saved = ck['recipe']
        legacy = dict(receiver_geometry=LEGACY_RECEIVER_GEOMETRY, light_power_start=LEGACY_LIGHT_POWER_START)
        config = {k: saved[k] for k in DEFAULTS if k not in legacy}
        config.update({k: saved.get(k, v) for k, v in legacy.items()})
        self.recipe = recipe_for_data(config, self.data)
        self.models, self.dataset, self.device = models, dataset, device
        self.seed, self.sample, self.predict_native = fixed_numpy_seed, sample_frame, predict_native
        self.identity = dict(method='geraf', step=int(ck['step']), selected=selected,
                             trans_power=float(self.recipe['trans_power']), seed=int(self.recipe['seed']))

    def acquisition(self, view, head, observations):
        from rift.geraf_source_ops import NativeAcquisition
        shard = self.dataset.shards[view[0], head]
        rows = shard.sector_rows[view[1]]
        a = shard.arrays
        positions = np.stack([a[k][rows] for k in ('x', 'y', 'z')], -1).astype(np.float64)
        points = torch.as_tensor(self.dataset.region.to_local(positions), device=self.device)
        reference = a['r0'][rows].astype(np.float64)
        if head in ('hh', 'vv'):
            reference = reference + a['r_correct_raw'][rows]
        if [o.pulse_index for o in observations] != [int(i) for i in a['pulse_index'][rows]]:
            raise ValueError('GeRaF acquisition rows differ from the observations')
        frequencies = torch.as_tensor(np.array(observations[0].frequencies_hz, copy=True), dtype=torch.float64,
                                      device=self.device)
        return NativeAcquisition(points, points, frequencies, torch.as_tensor(2 * reference, dtype=torch.float64,
                                 device=self.device), point_chunk=self.recipe['point_chunk'],
                                 pair_chunk=self.recipe['pair_chunk'])

    def predict(self, view, position, pol, observations, readouts):
        acquisition = self.acquisition(view, pol, observations)
        name = f'heldout_{pol}_pass{view[0]}_sector{view[1]:03d}'
        with self.seed(self.recipe['seed'] + position):
            frame = self.sample(acquisition, self.recipe, name, None, None)
        prediction = self.predict_native(self.models[pol], frame, acquisition)
        return prediction * self.recipe['trans_power']                          # [pulses, bins], raw units


class ZeroAdapter:
    """Reference: the zero prediction (no checkpoint). It scores 1 in every domain by construction."""

    def __init__(self, checkpoint_path, dataset, entry, device):
        if checkpoint_path is not None:
            raise ValueError('the zero reference takes no checkpoint')
        self.device = device
        self.identity = dict(method='zero_reference', prediction='all-zero native spectra')

    def predict(self, view, position, pol, observations, readouts):
        return torch.zeros((len(observations), len(observations[0].response)), dtype=torch.complex128,
                           device=self.device)


ADAPTERS = {'rift': RiftAdapter, 'spinr': SpinrAdapter, 'geraf': GerafAdapter, 'zero': ZeroAdapter}


# ---------------------------------------------------------------- harness
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--method', required=True, choices=sorted(ADAPTERS))
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--task', required=True, help='campaign task key whose command built the run')
    parser.add_argument('--checkpoint', type=Path, help='required except for --method zero')
    parser.add_argument('--label', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--role', choices=('validation', 'test'), default='validation')
    parser.add_argument('--allow-reserved-test', action='store_true')
    parser.add_argument('--mf-stats', type=Path, help='TRAIN-peak cache (default OUT/gotcha_mf_train_peak.json)')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-sectors', type=int, default=0, help='debug cap; 0 scores every pass-sector')
    args = parser.parse_args(argv)
    if args.role == 'test' and not args.allow_reserved_test:
        parser.error('reserved-test scoring requires --allow-reserved-test')
    if (args.checkpoint is None) != (args.method == 'zero'):
        parser.error('--checkpoint is required for a method and not accepted for --method zero')
    return args


def constant_floors(totals):
    """Scores of the best constant MF-power predictors on the scored bins (method independent).

    Zero power clamps to normalized 0, so the zero prediction scores exactly 1 in both MF domains.
    ``best_constant``: one level per polarization for every scored bin; ``best_per_sector_constant``:
    one level per pass-sector and polarization. Both are fitted to the scored role's own targets, so
    they lower-bound any constant predictor (a TRAIN-mean level included).
    """
    def floors(prefix):
        energy = sum(t[f'{prefix}_energy'] for t in totals.values())
        if not energy > 0:
            return None
        spread = sum(t[f'{prefix}_energy'] - t[f'{prefix}_sum'] ** 2 / t['bins'] for t in totals.values() if t['bins'])
        return dict(best_constant=spread / energy,
                    best_per_sector_constant=sum(t[f'{prefix}_sector_spread'] for t in totals.values()) / energy)
    return dict(zero_prediction=dict(normalized=1.0, linear=1.0), normalized=floors('db'), linear=floors('linear'),
                definition='oracle constants fitted to the scored targets; lower bounds on any constant predictor')


@torch.no_grad()
def main(argv=None):
    from rift_pvc.gotcha_training import RangeReadout
    from rift_pvc.radar_fields_gotcha import matched_range_power, range_geometry
    from rift.radar_fields_dataset import normalize_power_db
    args = parse_args(argv)
    device = torch.device(args.device)
    dataset, entry, task = load_run(args.campaign_root, args.task)
    stats = mf_statistics(dataset, args.mf_stats or args.out_dir/'gotcha_mf_train_peak.json')
    adapter = ADAPTERS[args.method](args.checkpoint, dataset, entry, device)
    if args.role == 'test':
        open_reserved_test(dataset)
    views = role_viewpoints(dataset, args.role)
    views = views[:args.max_sectors] if args.max_sectors else views
    readout = RangeReadout(dataset.region, device=device)
    names = ('coherent_error', 'coherent_energy', 'projected_error', 'projected_energy', 'db_error', 'db_energy',
             'linear_error', 'linear_energy', 'db_sum', 'db_sector_spread', 'linear_sum', 'linear_sector_spread')
    totals = {pol: dict({k: 0.0 for k in names}, pulses=0, bins=0, sectors=0) for pol in dataset.polarizations}
    started = time.perf_counter()
    for position, view in enumerate(views):
        for pol in dataset.polarizations:
            observations = list(dataset.observations(*view, pol))
            readouts = [readout.for_observation(o) for o in observations]
            prediction = adapter.predict(view, position, pol, observations, readouts).to(torch.complex128)
            target = torch.as_tensor(np.stack([o.response for o in observations]), dtype=torch.complex128, device=device)
            if prediction.shape != target.shape:
                raise ValueError(f'{view} {pol}: prediction {tuple(prediction.shape)} vs target {tuple(target.shape)}')
            t = totals[pol]
            t['coherent_error'] += float((prediction - target).abs().square().sum())
            t['coherent_energy'] += float(target.abs().square().sum())
            peak = float(stats['peak_power'][pol])
            sector_db, sector_linear = [], []
            for i, obs in enumerate(observations):
                r = readouts[i]
                t['projected_error'] += float((readout.project(prediction[i], r) - readout.project(target[i], r)).abs().square().sum())
                t['projected_energy'] += float(readout.project(target[i], r).abs().square().sum())
                _, ranges = range_geometry(obs, dataset.region, MF_GUARD_CELLS, device)
                measured = matched_range_power(obs, ranges)
                predicted = matched_range_power(SimpleNamespace(frequencies_hz=obs.frequencies_hz,
                                                                response=prediction[i].cpu().numpy(),
                                                                reference_range_m=obs.reference_range_m), ranges)
                im = normalize_power_db(measured, peak, MF_DYNAMIC_RANGE_DB)
                ip = normalize_power_db(predicted, peak, MF_DYNAMIC_RANGE_DB)
                t['db_error'] += float((ip - im).square().sum())
                t['db_energy'] += float(im.square().sum())
                t['linear_error'] += float((predicted - measured).square().sum())
                t['linear_energy'] += float(measured.square().sum())
                t['bins'] += int(len(ranges))
                sector_db.append(im.reshape(-1).double())
                sector_linear.append(torch.as_tensor(measured).reshape(-1).double())
            for key, values in (('db', torch.cat(sector_db)), ('linear', torch.cat(sector_linear))):
                t[f'{key}_sum'] += float(values.sum())
                t[f'{key}_sector_spread'] += float((values - values.mean()).square().sum())
            t['pulses'] += len(observations)
            t['sectors'] += 1
        if (position + 1) % 20 == 0 or position + 1 == len(views):
            print(f'[{position + 1:4d}/{len(views)}] {time.perf_counter() - started:7.1f}s', flush=True)

    def pooled(a, b):
        numerator, denominator = sum(t[a] for t in totals.values()), sum(t[b] for t in totals.values())
        return numerator / denominator if denominator > 0 else None
    role_total = len(role_viewpoints(dataset, args.role))
    result = dict(schema=SCHEMA, label=args.label, method=adapter.identity,
                  checkpoint=str(args.checkpoint.resolve()) if args.checkpoint else None,
                  campaign_root=str(args.campaign_root.resolve()), task=args.task, dataset_identity=dataset.identity,
                  selected_role=args.role, reserved_test_accessed=args.role == 'test', sectors=len(views),
                  status='complete' if len(views) == role_total else 'incomplete_or_smoke',
                  full_native_complex_rel_mse=pooled('coherent_error', 'coherent_energy'),
                  roi_projected_complex_rel_mse=pooled('projected_error', 'projected_energy'),
                  normalized_range_power_rel_mse=pooled('db_error', 'db_energy'),
                  linear_range_power_rel_mse=pooled('linear_error', 'linear_energy'),
                  reference_floors=constant_floors(totals),
                  by_polarization=totals, mf_power_definition=dict(
                      conversion='rift_pvc.radar_fields_gotcha.matched_range_power on range_geometry(guard_cells=2)',
                      normalization='normalize_power_db, TRAIN peak per polarization, 60 dB',
                      peak_power=stats['peak_power']),
                  definitions='RIFT-dataset evaluator (scripts/eval_b787_range_power.py) in GOTCHA form',
                  elapsed_seconds=time.perf_counter() - started, device=str(device))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir/f'{args.label}_metrics.json').write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'by_polarization'}, indent=2, sort_keys=True, default=str))


if __name__ == '__main__':
    main()
