#!/usr/bin/env python3
"""Re-rendering floors for VAL on the pass-held-out unit split (B48 iv, B52 c); CPU, method-free.

docs/RIFT_GOTCHA_Tune.md: B48 pre-registered, and B52 (c) fixed before VAL is read, that every VAL figure
on the unit split (``rift_pvc/gotcha_unit_split.py``, A41: every 4th unsealed sector ID in all 8 passes,
the pass-4 units of a seeded half held out) is printed beside the zero predictor and the one-look
re-rendering from the nearest training pass. This script computes those floors.

For each target unit (4, s) the box is reconstructed from TRAIN units of the same sector ID in other
passes only (never from the target), by CGLS from zero on the RIFT box grid (isotropic anchors, one
complex coefficient each; ``rift_pvc.gotcha_isolation``'s native kernel; the neighbour tool's
``sector_system`` and ``cgls``), and rendered on the target's own rows and selected bins:
- ``nearest_pass`` (the pre-registered floor): the one training pass nearest the target in look
  elevation from the box centre;
- ``bracket``: the nearest training pass above and the nearest below in elevation, fitted jointly;
- ``all_other_passes``: all seven training passes of that ID (a local multi-look reference).

Scores follow ``rift_pvc.gotcha_training.role_fit`` exactly: sums over the units' pulses and bins,
RelMSE = sum|yhat - y|^2 / sum|y|^2, e = sum|yhat|^2 / sum|y|^2 and rho = sum Re(yhat conj y) / sqrt(..),
pooled and by nearest-pass gap bin (<= 0.1, 0.1-0.2, 0.2-0.3, > 0.3 degrees of elevation). No gain is
fitted for these: a reconstruction of the same data renders on the data's own scale. Beside them are the
energy-pooled per-unit score with a fitted complex gain, 1 - |rho_c|^2 (B48's form), and the per-unit
RelMSE median.

Targets: the 38 VAL units, and as TRAIN controls the pass-4 units of the selected IDs that are not held
out (TRAIN rows, each predicted from the other passes of its ID exactly as a VAL unit is). The CGLS
iteration count reported as the floor is chosen per neighbourhood on the TRAIN controls (lowest pooled
RelMSE), never on VAL; every count is recorded, with M (complex measurements fitted), K (anchors) and the
in-sample RelMSE on the fitted units (B48: state M, K and convergence with every floor).

Reads TRAIN and VAL rows only; TEST stays sealed (the dataset adapter refuses test rows).

    # one role per CPU job, then the selection on the login node:
    python scripts_pvc/gotcha_unit_split_val_floor.py --shard-root <keep6 shards> --roles train --output train.json
    python scripts_pvc/gotcha_unit_split_val_floor.py --shard-root <keep6 shards> --roles validation --output val.json
    python scripts_pvc/gotcha_unit_split_val_floor.py --summarize train.json val.json --output floor.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'scripts_pvc'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rift.gotcha_dataset import GOTCHADataset, load_region  # noqa: E402
from rift_pvc.gotcha_training import box_anchors  # noqa: E402
from rift_pvc.gotcha_unit_split import apply_unit_split, unit_split_ids  # noqa: E402
from gotcha_neighbor_prediction import look, sector_system  # noqa: E402
from rift_pvc.gotcha_isolation import SectorGeometry  # noqa: E402

SCHEMA = 'gotcha_unit_split_val_floor_v1'
GAP_EDGES = (0.1, 0.2, 0.3)
GAP_LABELS = ('<=0.1', '0.1-0.2', '0.2-0.3', '>0.3')
SUM_KEYS = ('error', 'target', 'prediction', 'cross')


def gap_bin(gap_deg):
    for edge, label in zip(GAP_EDGES, GAP_LABELS):
        if abs(gap_deg) <= edge:
            return label
    return GAP_LABELS[-1]


def _adjoint(A, r):
    """A^H r as conj(r^H A): the row product avoids the strided conjugate-transpose matvec (5x faster on CPU)."""
    return (r.conj() @ A).conj()


def cgls(systems, iterations):
    """The neighbour tool's CGLS (min sum_s ||A_s x - y_s||^2 from x = 0), same iterates, faster adjoint.

    Yields (iteration, x, residual energy) at the requested counts.
    """
    wanted = set(iterations)
    x = torch.zeros(systems[0][0].shape[1], dtype=torch.complex128)
    r = [y.to(torch.complex128) for _, y in systems]
    s = sum(_adjoint(A, ri.to(torch.complex64)).to(torch.complex128) for (A, _), ri in zip(systems, r))
    p, gamma = s.clone(), float(s.abs().square().sum())
    for k in range(1, max(iterations) + 1):
        q = [(A @ p.to(torch.complex64)).to(torch.complex128) for A, _ in systems]
        alpha = gamma / float(sum(qi.abs().square().sum() for qi in q))
        x += alpha * p
        r = [ri - alpha * qi for ri, qi in zip(r, q)]
        if k in wanted:
            yield k, x.clone(), float(sum(ri.abs().square().sum() for ri in r))
        s = sum(_adjoint(A, ri.to(torch.complex64)).to(torch.complex128) for (A, _), ri in zip(systems, r))
        new = float(s.abs().square().sum())
        p, gamma = s + (new / gamma) * p, new


def unit_sums(prediction, y):
    """role_fit's per-unit sums, plus the complex inner product for the fitted-gain form."""
    p, t = prediction.to(torch.complex128), y.to(torch.complex128)
    inner = complex((p.conj() * t).sum())
    return dict(error=float((p - t).abs().square().sum()), target=float(t.abs().square().sum()),
                prediction=float(p.abs().square().sum()), cross=float((p * t.conj()).real.sum()),
                inner_re=inner.real, inner_im=inner.imag)


def pooled(rows):
    """role_fit's pooled RelMSE, e, rho and e/rho^2 over ``rows`` (dicts of unit sums)."""
    if not rows:
        return None
    s = {k: sum(r[k] for r in rows) for k in SUM_KEYS}
    e = s['prediction'] / s['target']
    rho = s['cross'] / np.sqrt(max(s['prediction'] * s['target'], 1e-300))
    rho_c = [np.hypot(r['inner_re'], r['inner_im']) / np.sqrt(max(r['prediction'] * r['target'], 1e-300))
             for r in rows]
    fitted = sum(r['target'] * (1 - c * c) for r, c in zip(rows, rho_c)) / s['target']
    # One gain for all units, as a single global scene has: the best real scale gives 1 - rho^2 (rho > 0),
    # the best complex gain 1 - |rho_c pooled|^2 with rho_c pooled = sum <yhat, y> / sqrt(sum|yhat|^2 sum|y|^2).
    pooled_c = np.hypot(sum(r['inner_re'] for r in rows), sum(r['inner_im'] for r in rows)) / np.sqrt(
        max(s['prediction'] * s['target'], 1e-300))
    return dict(units=len(rows), rel_mse=s['error'] / s['target'], energy_ratio=e, correlation=rho,
                e_over_rho2=(e / rho ** 2 if rho else None),
                rel_mse_best_real_scale=(1 - rho ** 2 if rho > 0 else 1.0),
                abs_rho_c_pooled=float(pooled_c), rel_mse_best_global_complex_gain=float(1 - pooled_c ** 2),
                rel_mse_fitted_gain_per_unit=fitted,
                median_abs_rho_c=float(np.median(rho_c)),
                per_unit_rel_mse_median=float(np.median([r['error'] / r['target'] for r in rows])),
                per_unit_rel_mse_p90=float(np.percentile([r['error'] / r['target'] for r in rows], 90)))


def neighbourhood_views(spec, others, gaps):
    """TRAIN units of the target's ID used to reconstruct it; ``gaps`` = neighbour minus target elevation."""
    nearest = min(others, key=lambda v: abs(gaps[v]))
    if spec == 'nearest_pass':
        return [nearest]
    if spec == 'bracket':
        above = [v for v in others if gaps[v] > 0]
        below = [v for v in others if gaps[v] < 0]
        return [min(side, key=lambda v: abs(gaps[v])) for side in (below, above) if side]
    if spec == 'all_other_passes':
        return list(others)
    if spec.startswith('all_but:') or spec.startswith('nearest_but:'):
        # B56: the same neighbourhoods without one training pass (is pass 5's cross-pass offset costing VAL?).
        excluded = int(spec.split(':')[1])
        kept = [v for v in others if v[0] != excluded]
        if not kept:
            return []
        return kept if spec.startswith('all_but:') else [min(kept, key=lambda v: abs(gaps[v]))]
    raise ValueError(f'unknown neighbourhood {spec!r}')


def measure(args):
    torch.set_num_threads(args.threads)
    passes = tuple(range(1, 9))
    ds = GOTCHADataset(args.dataset_root, shard_root=args.shard_root, passes=passes,
                       polarizations=(args.polarization,), region=load_region(args.region, args.region_config),
                       pulses_per_sector=0, frequency_stride=args.frequency_stride)
    split = apply_unit_split(ds, stride=args.stride, heldout_pass=args.heldout_pass,
                             heldout_fraction=args.heldout_fraction)
    selected, heldout = unit_split_ids(args.stride, args.heldout_fraction)
    train_views = {tuple(v) for v in ds.viewpoints('train')}
    validation = [tuple(v) for v in ds.viewpoints('validation')]
    if sorted(validation) != sorted((args.heldout_pass, s) for s in heldout):
        raise AssertionError('The VAL units are not the held-out pass units of the unit split')
    controls = [(args.heldout_pass, s) for s in selected if s not in set(heldout)]
    if not all(v in train_views for v in controls):
        raise AssertionError('A TRAIN control is not a TRAIN unit')
    targets = dict(validation=validation, train=controls)
    anchors = box_anchors(dict(kind='box', half_extents=args.half_extents, pitch=args.pitch), 'cpu').numpy()
    report = dict(schema=SCHEMA, split=split, dataset_identity=ds.identity, shard_root=str(args.shard_root),
                  region=args.region, anchors=len(anchors), pitch=args.pitch, half_extents=args.half_extents,
                  iterations=args.iterations, neighbourhoods=args.neighbourhoods, roles=args.roles,
                  estimator='cgls from zero, isotropic box anchors, native isolation kernel', rows=[])
    for role in args.roles:
        for target in targets[role][:args.limit or None]:
            started = time.perf_counter()
            p0, s0 = target
            others = [(q, s0) for q in passes if q != p0 and (q, s0) in train_views]
            if not others:
                raise AssertionError(f'{target} has no TRAIN unit of its ID in another pass')
            # Looks first (geometry only), then kernels for the units the requested neighbourhoods use.
            looks = {v: look(SectorGeometry.from_observations(list(ds.observations(*v, args.polarization)), ds.region))
                     for v in others}
            A_t, y_t, look_t, obs_t = sector_system(ds, target, args.polarization, anchors)
            gaps = {v: looks[v]['elevation_deg'] - look_t['elevation_deg'] for v in others}
            nearest = min(others, key=lambda v: abs(gaps[v]))
            needed = sorted({v for spec in args.neighbourhoods for v in neighbourhood_views(spec, others, gaps)})
            cache = {v: sector_system(ds, v, args.polarization, anchors) for v in needed}
            for spec in args.neighbourhoods:
                views = neighbourhood_views(spec, others, gaps)
                systems = [cache[v][:2] for v in views]
                energy = float(sum(y.abs().square().sum() for _, y in systems))
                row = dict(role=role, target=list(target), neighbourhood=spec, target_look=look_t,
                           target_pulses=len(obs_t), target_samples=int(y_t.numel()),
                           nearest_pass=nearest[0], nearest_gap_deg=float(gaps[nearest]),
                           gap_bin=gap_bin(gaps[nearest]),
                           views=[list(v) for v in views], view_gaps_deg=[float(gaps[v]) for v in views],
                           view_azimuth_offsets_deg=[float((looks[v]['azimuth_deg'] - look_t['azimuth_deg'] + 180) % 360 - 180)
                                                     for v in views],
                           measurements_m=int(sum(y.numel() for _, y in systems)), anchors_k=len(anchors),
                           by_iteration={})
                for k, x, residual in cgls(systems, args.iterations):
                    prediction = A_t @ x.to(torch.complex64)
                    row['by_iteration'][str(k)] = dict(unit_sums(prediction, y_t), in_sample_rel_mse=residual / energy)
                report['rows'].append(row)
                last = row['by_iteration'][str(args.iterations[-1])]
                print(f'{role} {target} {spec}: {len(views)} units, gap {gaps[nearest]:+.3f} deg (pass {nearest[0]}); '
                      'RelMSE by iteration ' + ', '.join(
                          f"{k}:{v['error'] / v['target']:.3f}" for k, v in row['by_iteration'].items())
                      + f"; in-sample at {args.iterations[-1]}: {last['in_sample_rel_mse']:.3g}", flush=True)
            del cache
            print(f'  {target} done in {time.perf_counter() - started:.0f} s', flush=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
    report['summary'] = summarize_rows(report['rows'], args.iterations)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


def summarize_rows(rows, iterations):
    out = {}
    for role in sorted({r['role'] for r in rows}):
        for spec in sorted({r['neighbourhood'] for r in rows}):
            chosen = [r for r in rows if r['role'] == role and r['neighbourhood'] == spec]
            if not chosen:
                continue
            block = {}
            for k in map(str, iterations):
                units = [r['by_iteration'][k] for r in chosen]
                block[k] = dict(pooled=pooled(units),
                                by_gap_bin={label: pooled([r['by_iteration'][k] for r in chosen if r['gap_bin'] == label])
                                            for label in GAP_LABELS},
                                in_sample_rel_mse_median=float(np.median([u['in_sample_rel_mse'] for u in units])))
            out[f'{role} {spec}'] = block
    return out


def summarize(args):
    reports = [json.loads(Path(p).read_text()) for p in args.summarize]
    identities = {r['dataset_identity'] for r in reports}
    if len(identities) != 1:
        raise ValueError('The reports belong to different datasets')
    rows = [row for r in reports for row in r['rows']]
    iterations = reports[0]['iterations']
    if any(r['iterations'] != iterations for r in reports):
        raise ValueError('The reports record different iteration counts')
    summary = summarize_rows(rows, iterations)
    floors = {}
    for spec in sorted({r['neighbourhood'] for r in rows}):
        train = summary.get(f'train {spec}')
        val = summary.get(f'validation {spec}')
        if not train or not val:
            continue
        # The iteration count is chosen on the TRAIN controls only.
        chosen = min(train, key=lambda k: train[k]['pooled']['rel_mse'])
        floors[spec] = dict(iterations_chosen_on_train_controls=int(chosen),
                            train_controls=train[chosen]['pooled'], validation=val[chosen]['pooled'],
                            validation_by_gap_bin=val[chosen]['by_gap_bin'],
                            validation_in_sample_rel_mse_median=val[chosen]['in_sample_rel_mse_median'],
                            validation_best_count_for_reference=min(val, key=lambda k: val[k]['pooled']['rel_mse']))
    result = dict(schema=SCHEMA + '_summary', dataset_identity=identities.pop(), sources=[str(p) for p in args.summarize],
                  zero_predictor=dict(rel_mse=1.0, energy_ratio=0.0, correlation=None),
                  floors=floors, summary=summary,
                  reading='B52 (c): quote a RIFT VAL figure on the unit split beside floors[nearest_pass].validation '
                          '(the pre-registered one-look re-rendering) and the zero predictor, pooled and by gap bin.')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for spec, f in floors.items():
        v = f['validation']
        print(f"{spec}: iterations {f['iterations_chosen_on_train_controls']} (TRAIN controls RelMSE "
              f"{f['train_controls']['rel_mse']:.3f}); VAL RelMSE {v['rel_mse']:.3f}, e {v['energy_ratio']:.3f}, "
              f"rho {v['correlation']:.3f}, fitted-gain {v['rel_mse_fitted_gain_per_unit']:.3f}; by gap bin "
              + ', '.join(f"{b}: {g['rel_mse']:.3f} (n={g['units']})" for b, g in f['validation_by_gap_bin'].items() if g),
              flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset-root', default=os.environ.get('GOTCHA_DATA_ROOT',
                   '/scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined'))
    p.add_argument('--shard-root', type=Path)
    p.add_argument('--region', default='camry_box_v2')
    p.add_argument('--region-config', type=Path, default=ROOT / 'rift_pvc/regions/camry_box_v2.json')
    p.add_argument('--polarization', default='hh')
    p.add_argument('--frequency-stride', type=int, default=2)
    p.add_argument('--stride', type=int, default=4, help='unit split: every Nth unsealed sector ID')
    p.add_argument('--heldout-pass', type=int, default=4)
    p.add_argument('--heldout-fraction', type=float, default=0.5)
    p.add_argument('--half-extents', type=float, nargs=3, default=[3.0, 1.5, 1.25])
    p.add_argument('--pitch', type=float, default=0.125)
    p.add_argument('--iterations', type=int, nargs='+', default=[1, 2, 3, 5, 10, 20],
                   help='CGLS counts recorded; one-look re-renderings overfit past about 3 (smoke 2157875/6)')
    p.add_argument('--neighbourhoods', nargs='+', default=['nearest_pass', 'bracket', 'all_other_passes'],
                   help="nearest_pass, bracket, all_other_passes, all_but:P, nearest_but:P")
    p.add_argument('--roles', nargs='+', default=['train', 'validation'], choices=['train', 'validation'])
    p.add_argument('--limit', type=int, default=0, help='first N targets per role (smoke tests only)')
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--summarize', type=Path, nargs='+', help='combine per-role reports; choose iterations on TRAIN')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    if args.summarize:
        summarize(args)
        return
    if args.shard_root is None:
        p.error('--shard-root is required to measure')
    if sorted(args.iterations) != args.iterations or len(set(args.iterations)) != len(args.iterations):
        p.error('--iterations must be distinct and ascending')
    measure(args)


if __name__ == '__main__':
    main()
