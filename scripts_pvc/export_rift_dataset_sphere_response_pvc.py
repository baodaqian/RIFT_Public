#!/usr/bin/env python3
"""Complex responses of the signal-bearing methods on the RIFT-dataset view sphere (TRAIN + VALIDATION views).

For one object this renders, at every TRAIN and VALIDATION view of the registered role manifest, the full
coherent spectrum S(f) (600 frequencies, the 1 Tx x 1 Rx production acquisition) of

* the measurement (chirp mean, as the evaluator's coherent reference),
* RIFT (the evaluator's own renderer: ``scripts/eval_b787_range_power.py`` load_scene + range operator + gain),
* SpINR-style, GeRaF and Sugavanam-Ertin Stage 1 (the held-out evaluator's adapters,
  ``scripts_pvc/eval_rift_dataset_heldout_pvc.py``),

each at the checkpoint that supplies the paper's tables (read from the final evaluation's
``<object>_<method>_val_metrics.json``). Predictions and measurements are saved per view for the sphere figure
(``scripts_pvc/plot_rift_dataset_sphere_response_pvc.py``). The reserved-test and unused roles are never read.

Check: GeRaF draws its rays with the per-view seed of its validation (seed + position in the role), so every
method's VALIDATION views reproduce the final evaluation's per-view coherent RelMSE; the summary reports the
largest relative deviation per method.

``--observable power`` (after the complex readout) scores every TRAIN and VALIDATION view in the common
matched-range power domain of the paper's Table 5 (the evaluator's metric block): P = |IFFT_f S|^2 per range bin,
normalized dB with the object's TRAIN-only Radar Fields statistics (``--stats``: TRAIN peak, 60 dB span, clipped to
[0, 1]), RelMSE over the view's 13 ROI bins. The four complex methods are converted from their saved spectra; the
power-only methods run through the held-out evaluator's adapters: Radar Fields (its normalized-dB intensity, rays
seeded by the view's position in its role) and RadarSplat (its image -> profile transfer, ROI bins inside its crop;
checkpoint and VALIDATION reference from ``--rs-eval-dir``). Writes <object>/<method>_power.npz; VALIDATION views are
checked against the final evaluation's per-view range-power RelMSE.

    python scripts_pvc/export_rift_dataset_sphere_response_pvc.py --object b787 --dataset-root D \\
        --role-manifest RUN/b787/role_manifest.json --eval-signal-dir FINAL_EVAL/signal --out-dir OUT
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

SCHEMA = 'rift_dataset_sphere_response_v1'
ROLES = ('train', 'validation')
METHODS = ('rift', 'spinr', 'geraf', 'sugavanam_ertin')
POWER_ONLY = ('radar_fields', 'radarsplat')
EVAL_LABEL = {'b787': 'b787', 'airliner_a320': 'a320', 'supersonic_x59': 'x59', 'firetruck': 'firetruck',
              'race_car': 'race_car', 'loader': 'loader'}


class RiftAdapter:
    """RIFT through the evaluator's own per-view path (``eval_b787_range_power.main``), prediction only."""

    kind = 'complex'

    def __init__(self, checkpoint_path, contract, arrays, device, args):
        import scripts.eval_b787_range_power as original
        from rift.config import cc
        from rift.forward_operator import get_kvector
        from rift.occlusion import array_phase_centre, view_transmittance
        from rift.radar_fields_dataset import build_frequency_grid
        from rift.range_operator import range_forward_operator
        from rift.rift_dataset import collection_contract, validate_checkpoint_object
        self.o, self.render = original, range_forward_operator
        self.array_phase_centre, self.view_transmittance = array_phase_centre, view_transmittance
        checkpoint, self.model, self.gain, self.occlusion = original.load_scene(checkpoint_path, device)
        validate_checkpoint_object(checkpoint, contract)
        if collection_contract(checkpoint.get('sealed_npz_protocol_contract', {})) != collection_contract(contract):
            raise ValueError('RIFT checkpoint acquisition/role contract disagrees with the registered object')
        self.freqs = torch.as_tensor(build_frequency_grid(arrays.metadata), dtype=torch.float32, device=device)
        self.kvector = get_kvector(self.freqs, cc)
        self.range_model = checkpoint.get('range_model', 'sum2')
        self.occlusion_value = self.occlusion['scale'].value if self.occlusion is not None else None
        self.device = device
        self.identity = dict(method='rift', epoch=int(checkpoint.get('epoch', -1)),
                             scene_repr=str(checkpoint.get('scene_repr', 'grid_sh')),
                             active=int(self.model.active_mask.sum().item()), range_model=self.range_model)

    def predict(self, view_index, position, rx_pos, tx_pos, viewpoint):
        theta, phi = self.o.theta_phi(viewpoint)
        dtheta = torch.tensor([[theta]], dtype=torch.float32, device=self.device)
        dphi = torch.tensor([[phi]], dtype=torch.float32, device=self.device)
        positions, weights = self.o.active_scatterers_for_view(self.model, dtheta, dphi)
        if self.occlusion is not None and self.occlusion_value >= 1.0e-7:
            occ = self.occlusion
            transmittance = self.view_transmittance(self.model, occ['scale'], self.array_phase_centre(rx_pos, tx_pos),
                                                    key=occ['key'], n_steps=occ['n_steps'],
                                                    step_frac=occ['step_frac'], point_chunk=occ['point_chunk'])
            weights = weights * transmittance.to(weights.dtype)
        prediction = self.render(self.freqs, self.kvector, rx_pos, tx_pos, positions, weights, phase_sign=-1.0,
                                 compute_dtype=torch.float64, pair_chunk=64, point_chunk=262144,
                                 range_model=self.range_model)
        return self.gain(prediction) if self.gain is not None else prediction          # [nf, Rx, Tx]


def adapters():
    from scripts_pvc.eval_rift_dataset_heldout_pvc import ADAPTERS
    return dict(rift=RiftAdapter, spinr=ADAPTERS['spinr'], geraf=ADAPTERS['geraf'],
                sugavanam_ertin=ADAPTERS['sugavanam_ertin'], radar_fields=ADAPTERS['radar_fields'],
                radarsplat=ADAPTERS['radarsplat'])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--object', required=True, help='dataset object id, e.g. b787, airliner_a320')
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--role-manifest', required=True)
    p.add_argument('--eval-signal-dir', type=Path, required=True,
                   help='final evaluation signal/ directory; each method checkpoint is read from its val metrics')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--observable', choices=('complex', 'power'), default='complex')
    p.add_argument('--methods', help='default: the four complex methods (complex), all six (power)')
    p.add_argument('--stats', help="the object's Radar Fields power stats (TRAIN peak); required for power")
    p.add_argument('--rs-eval-dir', type=Path,
                   help='RadarSplat VALIDATION readouts <label>_rs_logpower_val_{metrics.json,per_view.npz} (power)')
    p.add_argument('--device', default='cpu')
    p.add_argument('--max-views', type=int, default=0, help='smoke cap per role; 0 renders every view')
    args = p.parse_args(argv)
    allowed = METHODS if args.observable == 'complex' else METHODS + POWER_ONLY
    args.methods = args.methods.split(',') if args.methods else list(allowed)
    if args.observable == 'power' and not args.stats:
        p.error('--observable power needs --stats')
    if args.observable == 'power' and 'radarsplat' in args.methods and not args.rs_eval_dir:
        p.error('RadarSplat needs --rs-eval-dir')
    unknown = sorted(set(args.methods) - set(allowed))
    if unknown:
        p.error(f'unknown methods {unknown}')
    from rift.rift_dataset import resolve_object_inputs
    args.npz_path, args.role_manifest = map(str, resolve_object_inputs(
        object_name=args.object, dataset_root=args.dataset_root, npz_path=None, role_manifest_path=args.role_manifest))
    args.role = 'validation'          # adapters' default role label; set per view below
    return args


def role_views(args):
    """Registered contract, arrays restricted to TRAIN + VALIDATION, and the ordered views of each role."""
    from rift.radar_fields_dataset import from_collection_arrays, restrict_radar_fields_response_views
    from rift.rift_dataset import evaluation_role_indices, load_object_contract, object_identity, \
        validate_checkpoint_object
    public, contract = load_object_contract(args.npz_path, args.role_manifest, response_roles=ROLES)
    validate_checkpoint_object(object_identity(args.object), contract)
    arrays = from_collection_arrays(public, contract)
    roles = {role: evaluation_role_indices(contract, role) for role in ROLES}
    if args.max_views:
        roles = {role: ids[:args.max_views] for role, ids in roles.items()}
    return contract, arrays, roles, restrict_radar_fields_response_views(arrays, np.concatenate(list(roles.values())))


def eval_record(args, method, field='coherent_rel_mse'):
    """The paper checkpoint and the final evaluation's per-view VALIDATION values of one method."""
    label = EVAL_LABEL.get(args.object, args.object)
    stem = (args.rs_eval_dir / f'{label}_rs_logpower_val' if method == 'radarsplat'
            else args.eval_signal_dir / f'{label}_{method}_val')
    metrics = json.loads(Path(f'{stem}_metrics.json').read_text())
    with np.load(f'{stem}_per_view.npz') as cache:
        per_view = dict(zip(cache['view_indices'].tolist(), cache[field].tolist()))
    return metrics['checkpoint'], per_view


def power_readout(args, device, contract, arrays, restricted, order, where, common, out):
    """Per-view common matched-range power RelMSE of every method with a range-power prediction."""
    import scripts.eval_b787_range_power as original
    from rift.radar_fields_dataset import (normalize_power_db, range_bin_centers, response_view_to_range_power,
                                           scene_range_mask)
    from rift.rift_dataset import validate_checkpoint_object
    from scripts_pvc.eval_b787_range_power_pvc import validate_normalization_stats
    stats = json.loads(Path(args.stats).read_text())
    validate_checkpoint_object(stats, contract)
    peak, span = validate_normalization_stats(stats, {'sealed_npz_protocol_contract': contract},
                                              num_views=arrays.num_views)
    ranges = range_bin_centers(arrays.metadata, device=device, dtype=torch.float32)
    with np.load(out / 'measured.npz') as saved:
        if not np.array_equal(saved['view_indices'], order):
            raise ValueError('measured.npz holds other views; rerun the complex readout first')
        measured = saved['spectrum']                       # complex64 chirp mean of the 1 x 1 pair
    targets, rois = [], []
    for slot, view_index in enumerate(order.tolist()):
        power = response_view_to_range_power(measured[slot][None, None, None, :], device=device)
        targets.append(normalize_power_db(power, peak, span))
        viewpoint = torch.as_tensor(arrays.viewpoint_positions[view_index], dtype=torch.float32, device=device)
        rois.append(scene_range_mask(ranges, viewpoint, original.RF_GRID_EXTENT_M, margin=0.05))

    summary_path = out / 'summary.json'
    summary = json.loads(summary_path.read_text())
    summary.setdefault('power_methods', {})
    summary['power_definition'] = ('common matched-range power RelMSE: normalized dB (TRAIN peak '
                                   f'{peak:.6g}, {span:g} dB), ROI bins (RadarSplat: ROI bins inside its crop)')
    table = adapters()
    for method in args.methods:
        path = out / f'{method}_power.npz'
        if path.exists():
            print(f'{method} power: {path} exists, skipped', flush=True)
            continue
        checkpoint, eval_per_view = eval_record(args, method, 'range_power_rel_mse')
        started = time.perf_counter()
        if method in METHODS:          # power of the saved coherent prediction, as the evaluator computes it
            with np.load(out / f'{method}.npz') as saved:
                if not np.array_equal(saved['view_indices'], order):
                    raise ValueError(f'{method}.npz holds other views')
                spectrum, identity = saved['spectrum'], json.loads(str(saved['record']))['identity']
        else:
            adapter = table[method](checkpoint, contract, restricted, device, args)
            adapter.bind(restricted)
            identity = adapter.identity
        rel, count = np.full(len(order), np.nan), np.zeros(len(order), dtype=np.int64)
        sq, tsq = np.zeros(len(order)), np.zeros(len(order))
        for slot, view_index in enumerate(order.tolist()):
            roi, target = rois[slot], targets[slot]
            if method in METHODS:
                pred = torch.as_tensor(spectrum[slot][None], dtype=torch.complex128, device=device)
                pred_intensity = normalize_power_db(torch.fft.ifft(pred, dim=-1).abs().square(), peak, span)
            else:
                role, position = where[view_index]
                rx_pos = torch.as_tensor(arrays.rx_pos[view_index], dtype=torch.float32, device=device)
                tx_pos = torch.as_tensor(arrays.tx_pos[view_index], dtype=torch.float32, device=device)
                prediction = adapter.predict(view_index, position, rx_pos, tx_pos,
                                             arrays.viewpoint_positions[view_index])
                if method == 'radar_fields':
                    if not torch.equal(prediction['roi'].to(roi.device), roi):
                        raise ValueError(f'view {view_index}: the method ROI differs from the evaluator ROI')
                    pred_intensity = prediction['intensity'].to(target.dtype)
                else:
                    pred_intensity = normalize_power_db(prediction['power'].to(target.dtype), peak, span)
                    roi = roi & prediction['mask'].to(roi.device)
            t, q = target[:, roi], pred_intensity[:, roi]
            sq[slot], tsq[slot] = float((q - t).square().sum()), float(t.square().sum())
            rel[slot], count[slot] = sq[slot] / max(tsq[slot], 1.0e-30), int(t.numel())
            if (slot + 1) % 500 == 0:
                print(f'{method} power: {slot + 1}/{len(order)} views, {time.perf_counter() - started:.0f}s', flush=True)
        val = [(rel[i], eval_per_view[int(v)]) for i, v in enumerate(order) if int(v) in eval_per_view]
        deviation = max((abs(a - b) / max(abs(b), 1e-30) for a, b in val), default=None)
        record = dict(checkpoint=checkpoint, identity=identity,
                      range_power_rel_mse={r: float(sq[common['roles'] == r].sum() / tsq[common['roles'] == r].sum())
                                           for r in ROLES},
                      scored_bins_per_view=[int(count.min()), int(count.max())],
                      validation_views_checked=len(val), max_relative_deviation_from_final_eval=deviation,
                      seconds=time.perf_counter() - started)
        np.savez(path, range_power_rel_mse=rel, scored_bins=count, squared_error=sq, target_squared_norm=tsq,
                 schema=SCHEMA, object=args.object, method=method, record=json.dumps(record, default=str),
                 **{k: v for k, v in common.items() if k != 'frequency_hz'})
        summary['power_methods'][method] = record
        print(f'{method} power: {json.dumps(record, default=str)}', flush=True)
        summary_path.write_text(json.dumps(summary, indent=2, default=str) + '\n')


@torch.no_grad()      # not inference_mode: GeRaF's SDF normals use autograd inside enable_grad
def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    from rift.radar_fields_dataset import build_frequency_grid
    contract, arrays, roles, restricted = role_views(args)
    if arrays.num_tx * arrays.num_rx != 1:
        raise ValueError('defined for the 1 Tx x 1 Rx production acquisition')
    order = np.sort(np.concatenate(list(roles.values())))
    where = {int(v): (role, pos) for role, ids in roles.items() for pos, v in enumerate(ids.tolist())}
    out = args.out_dir / args.object
    out.mkdir(parents=True, exist_ok=True)
    common = dict(view_indices=order, roles=np.array([where[int(v)][0] for v in order]),
                  viewpoint_positions=np.asarray(arrays.viewpoint_positions)[order],
                  frequency_hz=np.asarray(build_frequency_grid(arrays.metadata)))

    measured_path = out / 'measured.npz'
    if not measured_path.exists():
        started, rows = time.perf_counter(), {}
        for view_index, response_view in restricted.iter_response_views(order):
            rows[int(view_index)] = np.asarray(response_view).mean(axis=2).reshape(-1, arrays.num_freq)[0]
        spectrum = np.stack([rows[int(v)] for v in order]).astype(np.complex64)
        np.savez(measured_path, spectrum=spectrum, schema=SCHEMA, object=args.object, **common)
        print(f'measured: {len(order)} views in {time.perf_counter() - started:.0f}s -> {measured_path}', flush=True)
    if args.observable == 'power':
        power_readout(args, device, contract, arrays, restricted, order, where, common, out)
        print(f'done -> {out}', flush=True)
        return
    measured = np.load(measured_path)['spectrum'].astype(np.complex128)

    summary_path = out / 'summary.json'
    summary = (json.loads(summary_path.read_text()) if summary_path.exists() else
               dict(schema=SCHEMA, object=args.object, roles={r: int(len(v)) for r, v in roles.items()},
                    role_manifest=args.role_manifest, npz_path=args.npz_path, methods={}))
    table = adapters()
    for method in args.methods:
        path = out / f'{method}.npz'
        if path.exists():
            print(f'{method}: {path} exists, skipped', flush=True)
            continue
        checkpoint, eval_per_view = eval_record(args, method)
        started = time.perf_counter()
        adapter = table[method](checkpoint, contract, restricted, device, args)
        if hasattr(adapter, 'bind'):
            adapter.bind(restricted)
        spectrum = np.zeros((len(order), arrays.num_freq), dtype=np.complex64)
        for slot, view_index in enumerate(order.tolist()):
            role, position = where[view_index]
            adapter.role = role                          # GeRaF names its ray frame by role
            rx_pos = torch.as_tensor(arrays.rx_pos[view_index], dtype=torch.float32, device=device)
            tx_pos = torch.as_tensor(arrays.tx_pos[view_index], dtype=torch.float32, device=device)
            prediction = adapter.predict(view_index, position, rx_pos, tx_pos, arrays.viewpoint_positions[view_index])
            spectrum[slot] = prediction.permute(2, 1, 0).reshape(-1, arrays.num_freq)[0].cpu().numpy()
            if (slot + 1) % 500 == 0:
                print(f'{method}: {slot + 1}/{len(order)} views, {time.perf_counter() - started:.0f}s', flush=True)
        pred = spectrum.astype(np.complex128)
        rel = (np.abs(pred - measured) ** 2).sum(1) / np.maximum((np.abs(measured) ** 2).sum(1), 1e-30)
        val = [(rel[i], eval_per_view[int(v)]) for i, v in enumerate(order) if int(v) in eval_per_view]
        deviation = max((abs(a - b) / max(abs(b), 1e-30) for a, b in val), default=None)
        record = dict(checkpoint=checkpoint, identity=adapter.identity,
                      coherent_rel_mse={r: float(((np.abs(pred - measured) ** 2).sum(1)[common['roles'] == r]).sum()
                                                 / (np.abs(measured) ** 2).sum(1)[common['roles'] == r].sum())
                                        for r in ROLES},
                      validation_views_checked=len(val), max_relative_deviation_from_final_eval=deviation,
                      seconds=time.perf_counter() - started)
        np.savez(path, spectrum=spectrum, coherent_rel_mse=rel, schema=SCHEMA, object=args.object, method=method,
                 record=json.dumps(record, default=str), **common)
        summary['methods'][method] = record
        print(f'{method}: {json.dumps(record, default=str)}', flush=True)
        summary_path.write_text(json.dumps(summary, indent=2, default=str) + '\n')
    print(f'done -> {out}', flush=True)


if __name__ == '__main__':
    main()
