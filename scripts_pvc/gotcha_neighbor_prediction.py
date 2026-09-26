#!/usr/bin/env python3
"""How predictable is a held-out GOTCHA sector from its TRAIN neighbours? A method-free ceiling (CPU).

docs/RIFT_GOTCHA_Tune.md section 10, A28: every RIFT arm on the box-isolated target (and arm B on the
old target) fits TRAIN sectors (rho 0.4-0.5) but predicts VALIDATION sectors with rho ~ 0. This asks
whether any scene on the box support could do better under the same forward model.

For each target sector t (sampled VALIDATION sectors, and TRAIN sectors held out the same way as a
control) the box is reconstructed from selected TRAIN sectors near t only:
- anchors: the RIFT box-support grid (``rift_pvc.gotcha_training.box_anchors``), isotropic, one
  complex coefficient each;
- forward: ``rift_pvc.gotcha_isolation.SectorGeometry.responses`` (the native kernel, per-pulse r0),
  on the dataset's own rows and selected frequencies (the target every method trains on);
- solver: CGLS on the stacked neighbour sectors (iteration 1 is the scaled adjoint, i.e. a
  backprojection start; early stopping is the only regularization), predictions recorded at several
  iteration counts;
- neighbourhoods (the split is by sector id, so a VALIDATION azimuth is held out in every pass; ds = 0
  is excluded for TRAIN targets as well, so both see the same situation):
  ``below`` / ``above``: the single nearest selected TRAIN sector of t's pass on that side, recorded
  with its gap in sectors (rho against the angular gap);
  ``nearest``: both of them; ``nearest_all_passes``: the nearest on each side in every pass (the local
  analogue of what a multi-look model sees); ``window:k``: every selected TRAIN sector of t's pass
  with 0 < |ds| <= k; ``self`` (TRAIN targets only): the target's own rows, the in-sample control;
  ``same_azimuth_other_passes`` (TRAIN targets only; VALIDATION azimuths have none): the same sector id
  in the other passes, reported separately from the same-pass arms. It separates inter-pass
  decorrelation of the car's response (physics) from angular sampling (the split);
  ``same_azimuth_each_pass`` expands to ``same_azimuth_pass:P``, one row per other pass P (pairwise);
  ``azimuth_window_all_passes:m``: every selected TRAIN sector of every pass with 0 < |ds| <= m (the
  pooled looks a multi-look model sees around a held-out azimuth; B27).
- ``--neighbour-pulse-stride`` / ``--neighbour-bin-stride`` subsample the neighbours' rows and bins
  (never the target's) to fit large pooled arms in memory. keep6 rows at pulse stride 2 still sample
  cross-range alias-free for a 6 m box (alias ~12.5 m), and stride-2 bins at bin stride 2 keep a
  ~25 m unambiguous range; a subsampled run carries its own bridge row at a smaller m.
- every row records the look (elevation and azimuth of the sector's mean antenna position from the
  box centre, degrees) of the target and of each neighbour.
- the target side is the training rows exactly (keep6 filtered rows, stride-2 bins). Neighbours use the
  same rows: filter-then-decimate keeps the box content alias-free at keep6, so keep1 would add only
  noise averaging at six times the memory. RIFT's sum2 amplitude varies by < 0.1 % over the box at
  10 km and is absorbed by the gain, so the kernel is the phase kernel.
- iteration counts are chosen on TRAIN targets (never on VALIDATION); every count is reported.

``--magnitude`` also scores every prediction in the magnitude domains (A30):
- the campaign's MF-power metric exactly as ``scripts_pvc/eval_gotcha_heldout_pvc.py`` defines it:
  ``rift_pvc.radar_fields_gotcha.matched_range_power`` on ``range_geometry(guard_cells=2)`` per pulse,
  ``normalize_power_db`` with the TRAIN peak over every selected TRAIN pulse and 60 dB; normalized
  RelMSE sum (p - m)^2 / sum m^2, as reconstructed and with a fitted power scale (log grid), and the
  linear-power RelMSE with a fitted scale;
- plain |.| of the predicted native rows against |.| of the target with a fitted real gain
  (1 - (sum |yhat||y|)^2 / (sum |yhat|^2 sum |y|^2));
- a copy floor per row, no reconstruction: the neighbours' own measured power profiles (MF: range
  offsets from each pulse's ROI centre aligned; |.|: native rows), averaged over the neighbours, pulse
  i of the target matched to the neighbour pulse at the same relative position in its sector; and
- the per-sector constant oracle of the MF metric (the evaluator's ``best_per_sector_constant``).
``--self-all-roles`` adds the in-sector control for VALIDATION targets too.

``--pass-phase-offsets P1 .. P8`` (radians, phase of each pass relative to pass 1, e.g. the TRAIN-only
trihedral offsets of docs/RIFT_GOTCHA_Tune.md section 5b) rotates every row of pass p, target and
neighbours alike, by exp(-i P_p) before fitting and scoring (B38 pre-test: is the single global gain's
penalty the inter-pass phase offset?). Magnitude scores are unaffected.

``--estimator omp`` replaces CGLS by a sparse point fit (A31): greedy orthogonal matching pursuit on the
same anchor grid and kernel (``--omp-batch`` atoms per step by normalized correlation, least-squares
complex amplitudes), predictions recorded at ``--sparsity`` point counts. CGLS iterates stay in the
row space of the neighbours' operator, so their image spectrum is confined to the neighbours' k-space
wedges and renders ~0 at a disjoint wedge (A29-correction); a sparse point model extrapolates across
wedges. Every row also records the complex copy floor (the neighbours' rows, pulse-matched, as the
prediction).

Reported per target and neighbourhood: complex correlation rho_c = <yhat, y> / (|yhat| |y|), the RelMSE
with a fitted complex gain (1 - |rho_c|^2), the raw RelMSE and real correlation (no gain), and the
in-sample fit on the neighbours. Reads TRAIN/VALIDATION responses only; TEST stays sealed.

    source .local-setup/activate-pvc.sh
    python scripts_pvc/gotcha_neighbor_prediction.py --shard-root <keep6 shards> --region camry_box_v2 \\
        --region-config rift_pvc/regions/camry_box_v2.json --targets 10 --output ceiling.json
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rift.gotcha_dataset import GOTCHADataset, load_region  # noqa: E402
from rift_pvc.gotcha_isolation import SectorGeometry  # noqa: E402
from rift_pvc.gotcha_training import box_anchors  # noqa: E402

MF_GUARD_CELLS, MF_DYNAMIC_RANGE_DB = 2, 60.0


PASS_PHASE = {}  # pass id -> radians, set from --pass-phase-offsets


def _rotation(view):
    return complex(np.exp(-1j * PASS_PHASE.get(int(view[0]), 0.0)))


def sector_system(ds, view, polarization, anchors):
    observations = list(ds.observations(*view, polarization))
    for o in observations:
        if len(o.response) != len(o.frequencies_hz):
            raise ValueError('response and frequency selection disagree')
    geometry = SectorGeometry.from_observations(observations, ds.region)
    A = geometry.responses(anchors, dtype=torch.complex64)
    y = torch.as_tensor(np.stack([o.response for o in observations]).reshape(-1), dtype=torch.complex64)
    return A, y * _rotation(view), look(geometry), observations


def neighbour_system(ds, view, polarization, anchors, pulse_stride, bin_stride):
    """``sector_system`` on every ``pulse_stride``-th row and ``bin_stride``-th selected bin."""
    if pulse_stride == 1 and bin_stride == 1:
        return sector_system(ds, view, polarization, anchors)
    observations = list(ds.observations(*view, polarization))[::pulse_stride]
    full = SectorGeometry.from_observations(observations, ds.region)
    geometry = SectorGeometry(full.antennas, full.frequencies[::bin_stride], full.reference)
    A = geometry.responses(anchors, dtype=torch.complex64)
    y = torch.as_tensor(np.stack([o.response[::bin_stride] for o in observations]).reshape(-1), dtype=torch.complex64)
    return A, y * _rotation(view), look(geometry), observations


def look(geometry):
    """Elevation and azimuth (degrees) of the sector's mean antenna position from the box centre."""
    x, y, z = geometry.antennas.mean(0)
    return dict(elevation_deg=float(np.degrees(np.arctan2(z, np.hypot(x, y)))),
                azimuth_deg=float(np.degrees(np.arctan2(y, x))))


def cgls(systems, iterations):
    """min sum_s ||A_s x - y_s||^2 from x = 0; yields (iteration, x) at the requested counts."""
    wanted = set(iterations)
    x = torch.zeros(systems[0][0].shape[1], dtype=torch.complex128)
    r = [y.to(torch.complex128) for _, y in systems]
    s = sum((A.conj().T @ ri.to(torch.complex64)).to(torch.complex128) for (A, _), ri in zip(systems, r))
    p, gamma = s.clone(), float(s.abs().square().sum())
    for k in range(1, max(iterations) + 1):
        q = [(A @ p.to(torch.complex64)).to(torch.complex128) for A, _ in systems]
        alpha = gamma / float(sum(qi.abs().square().sum() for qi in q))
        x += alpha * p
        r = [ri - alpha * qi for ri, qi in zip(r, q)]
        if k in wanted:
            yield k, x.clone(), float(sum(ri.abs().square().sum() for ri in r))
        s = sum((A.conj().T @ ri.to(torch.complex64)).to(torch.complex128) for (A, _), ri in zip(systems, r))
        new = float(s.abs().square().sum())
        p, gamma = s + (new / gamma) * p, new


def omp(systems, counts, batch):
    """Greedy sparse fit: ``batch`` atoms per step by |sum_s A_s^H r_s| / column norm, LS amplitudes.

    Yields (points, x, residual energy) at the requested counts; x is dense with zeros off the support.
    The Gram matrix and right-hand side grow by blocks, so a step costs O(samples * points * batch).
    """
    wanted = sorted(set(counts))
    n = systems[0][0].shape[1]
    norms = torch.sqrt(sum(A.abs().square().sum(0).double() for A, _ in systems)).clamp_min(1e-30)
    ys = [y.to(torch.complex128) for _, y in systems]
    r = [y.clone() for y in ys]
    selected = torch.zeros(n, dtype=torch.bool)
    index = torch.empty(0, dtype=torch.long)
    B = [torch.empty(A.shape[0], 0, dtype=torch.complex128) for A, _ in systems]
    G = torch.empty(0, 0, dtype=torch.complex128)
    rhs = torch.empty(0, dtype=torch.complex128)
    while len(index) < wanted[-1]:
        c = torch.zeros(n, dtype=torch.complex128)
        for (A, _), ri in zip(systems, r):
            c += (A.conj().T @ ri.to(torch.complex64)).to(torch.complex128)
        score = c.abs() / norms
        score[selected] = -1.
        step = min(batch, next(k for k in wanted if k > len(index)) - len(index))
        new = torch.topk(score, step).indices
        selected[new] = True
        index = torch.cat([index, new])
        added = [A[:, new].to(torch.complex128) for A, _ in systems]
        cross = sum(b.conj().T @ a for b, a in zip(B, added))                  # [old, new]
        corner = sum(a.conj().T @ a for a in added)                           # [new, new]
        G = torch.cat([torch.cat([G, cross], 1), torch.cat([cross.conj().T, corner], 1)], 0)
        rhs = torch.cat([rhs, sum(a.conj().T @ y for a, y in zip(added, ys))])
        B = [torch.cat([b, a], 1) for b, a in zip(B, added)]
        ridge = 1e-8 * float(G.diagonal().real.mean())
        z = torch.linalg.solve(G + ridge * torch.eye(len(index), dtype=G.dtype), rhs)
        r = [y - b @ z for b, y in zip(B, ys)]
        if len(index) in wanted:
            x = torch.zeros(n, dtype=torch.complex128)
            x[index] = z
            yield len(index), x, float(sum(ri.abs().square().sum() for ri in r))


def agreement(prediction, y):
    prediction, y = prediction.to(torch.complex128), y.to(torch.complex128)
    pp, yy = float(prediction.abs().square().sum()), float(y.abs().square().sum())
    inner = complex((prediction.conj() * y).sum())
    rho = inner / np.sqrt(pp * yy) if pp > 0 else 0j
    return dict(abs_rho_c=abs(rho), real_rho=rho.real, rel_mse_fitted_gain=1 - abs(rho) ** 2,
                rel_mse_raw=float((prediction - y).abs().square().sum()) / yy, energy_ratio=pp / yy)


def mf_profiles(observations, rows, region, offsets_from=None):
    """Matched range power per pulse on the evaluator's grid; ``rows`` [P, F] complex responses.

    With ``offsets_from`` (the target's observations), each profile is evaluated at the target pulse's
    range offsets from its ROI centre (the copy floor's alignment); otherwise on the pulse's own grid.
    """
    from types import SimpleNamespace
    from rift_pvc.radar_fields_gotcha import matched_range_power, range_geometry
    out = []
    for i, obs in enumerate(offsets_from or observations):
        antenna, ranges = range_geometry(obs, region, MF_GUARD_CELLS)
        if offsets_from is not None:
            j = int(round(i * (len(observations) - 1) / max(len(offsets_from) - 1, 1)))
            source = observations[j]
            centre = float(np.linalg.norm(region.to_local(source.position_m)))
            ranges = ranges - float(antenna.norm()) + centre
            obs, row = source, rows[j]
        else:
            row = rows[i]
        out.append(matched_range_power(SimpleNamespace(frequencies_hz=obs.frequencies_hz, response=np.asarray(row),
                                                       reference_range_m=obs.reference_range_m), ranges))
    return out


def mf_scores(predicted, measured, peak):
    """Evaluator sums for one sector: normalized-dB as given and at the best power scale, linear at its best scale."""
    from rift.radar_fields_dataset import normalize_power_db
    p, m = torch.cat(predicted).double(), torch.cat(measured).double()
    im = normalize_power_db(m, peak, MF_DYNAMIC_RANGE_DB)
    db_energy = float(im.square().sum())

    def db_error(scale):
        return float((normalize_power_db(p * scale, peak, MF_DYNAMIC_RANGE_DB) - im).square().sum())
    scales = 10.0 ** np.linspace(-6, 6, 241)
    errors = [db_error(s) for s in scales]
    best = int(np.argmin(errors))
    fine = scales[best] * 10.0 ** np.linspace(-0.05, 0.05, 21)
    fitted = min(db_error(s) for s in fine)
    pp, pm, mm = float((p * p).sum()), float((p * m).sum()), float((m * m).sum())
    return dict(db_error=db_error(1.0), db_error_fitted=min(fitted, errors[best]), db_energy=db_energy,
                db_constant_spread=float((im - im.mean()).square().sum()),
                linear_error_fitted=mm - (pm * pm / pp if pp > 0 else 0.), linear_energy=mm)


def magnitude_sums(prediction, y):
    a, b = prediction.abs().double(), y.abs().double()
    ab, aa, bb = float((a * b).sum()), float((a * a).sum()), float((b * b).sum())
    return dict(mag_error_fitted=bb - (ab * ab / aa if aa > 0 else 0.), mag_energy=bb)


def ratios(sums):
    return dict(mf_db_rel_mse=sums['db_error'] / sums['db_energy'] if 'db_error' in sums else None,
                mf_db_rel_mse_fitted_scale=sums['db_error_fitted'] / sums['db_energy'],
                mf_linear_rel_mse_fitted_scale=sums['linear_error_fitted'] / sums['linear_energy'],
                magnitude_rel_mse_fitted_gain=sums['mag_error_fitted'] / sums['mag_energy'])


def signed_gap(s, s0):
    return (s - s0 + 180) % 360 - 180


def nearest_side(target, pool, pass_id, side):
    p0, s0 = target
    candidates = [(abs(signed_gap(s, s0)), (p, s)) for p, s in pool
                  if p == pass_id and signed_gap(s, s0) * side > 0]
    return min(candidates)[1] if candidates else None


def neighbours(target, pool, spec, passes):
    """Selected TRAIN views for one neighbourhood; ds = 0 (the target azimuth) is never used."""
    p0, s0 = target
    if spec == 'self':
        return [target]
    if spec in ('below', 'above'):
        view = nearest_side(target, pool, p0, -1 if spec == 'below' else 1)
        return [view] if view else []
    if spec == 'nearest':
        return [v for v in (nearest_side(target, pool, p0, -1), nearest_side(target, pool, p0, 1)) if v]
    if spec == 'nearest_all_passes':
        return [v for p in passes for v in (nearest_side(target, pool, p, -1), nearest_side(target, pool, p, 1)) if v]
    if spec == 'same_azimuth_other_passes':
        return [(p, s) for p, s in pool if s == s0 and p != p0]
    if spec.startswith('same_azimuth_pass:'):
        other = int(spec.split(':')[1])
        return [(p, s) for p, s in pool if s == s0 and p == other != p0]
    if spec.startswith('azimuth_window_all_passes:'):
        m = int(spec.split(':')[1])
        return [(p, s) for p, s in pool if 0 < abs(signed_gap(s, s0)) <= m]
    if spec.startswith('window:'):
        k = int(spec.split(':')[1])
        return [(p, s) for p, s in pool if p == p0 and 0 < abs(signed_gap(s, s0)) <= k]
    raise ValueError(f'unknown neighbourhood {spec!r}')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset-root', default=os.environ.get('GOTCHA_DATA_ROOT',
                   '/scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined'))
    p.add_argument('--shard-root', type=Path)
    p.add_argument('--region', default='camry_box_v2')
    p.add_argument('--region-config', type=Path)
    p.add_argument('--passes', type=int, nargs='+', default=list(range(1, 9)))
    p.add_argument('--polarization', default='hh')
    p.add_argument('--num-train', type=int, default=1500)
    p.add_argument('--pulses-per-sector', type=int, default=0)
    p.add_argument('--frequency-stride', type=int, default=2)
    p.add_argument('--half-extents', type=float, nargs=3, default=[3.0, 1.5, 1.25])
    p.add_argument('--pitch', type=float, default=0.125)
    p.add_argument('--targets', type=int, default=10, help='targets per role (seed 0)')
    p.add_argument('--roles', nargs='+', default=['validation', 'train'], choices=['validation', 'train'])
    p.add_argument('--neighbourhoods', nargs='+', default=['below', 'above', 'nearest', 'nearest_all_passes',
                                                            'window:5', 'self', 'same_azimuth_other_passes'])
    p.add_argument('--iterations', type=int, nargs='+', default=[1, 3, 10, 30])
    p.add_argument('--estimator', choices=('cgls', 'omp'), default='cgls')
    p.add_argument('--sparsity', type=int, nargs='+', default=[50, 200, 1000], help='OMP point counts recorded')
    p.add_argument('--omp-batch', type=int, default=10)
    p.add_argument('--pass-phase-offsets', type=float, nargs='+',
                   help='per-pass phase relative to pass 1 (rad), one value per --passes entry; rows rotated by exp(-i p)')
    p.add_argument('--neighbour-pulse-stride', type=int, default=1)
    p.add_argument('--neighbour-bin-stride', type=int, default=1)
    p.add_argument('--magnitude', action='store_true', help='also score in the MF-power and |.| domains (A30)')
    p.add_argument('--self-all-roles', action='store_true', help='in-sector control for VALIDATION targets too')
    p.add_argument('--mf-stats', type=Path, help='cached TRAIN-peak statistics (computed and written if absent)')
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    if args.pass_phase_offsets:
        if len(args.pass_phase_offsets) != len(args.passes):
            raise SystemExit('--pass-phase-offsets needs one value per --passes entry')
        PASS_PHASE.update({int(q): float(v) for q, v in zip(args.passes, args.pass_phase_offsets)})
    ds = GOTCHADataset(args.dataset_root, shard_root=args.shard_root, passes=tuple(args.passes),
                       polarizations=(args.polarization,), region=load_region(args.region, args.region_config),
                       num_train=args.num_train, pulses_per_sector=args.pulses_per_sector,
                       frequency_stride=args.frequency_stride)
    anchors = box_anchors(dict(kind='box', half_extents=args.half_extents, pitch=args.pitch), 'cpu').numpy()
    peak = None
    if args.magnitude:
        cache_path = args.mf_stats or args.output.with_name('gotcha_mf_train_peak.json')
        if cache_path.exists():
            stats = json.loads(cache_path.read_text())
            if stats.get('dataset_identity') != ds.identity or stats.get('guard_cells') != MF_GUARD_CELLS:
                raise ValueError(f'{cache_path} belongs to another dataset or definition')
        else:
            from rift_pvc.radar_fields_gotcha import training_statistics
            stats = dict(training_statistics(ds, dict(controls=dict(range_guard_cells=MF_GUARD_CELLS,
                                                                    dynamic_range_db=MF_DYNAMIC_RANGE_DB))),
                         guard_cells=MF_GUARD_CELLS, definition='rift_pvc.radar_fields_gotcha.training_statistics')
            cache_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + '\n')
        peak = float(stats['peak_power'][args.polarization])
        print(f'MF TRAIN peak power {peak:.6g} ({cache_path})', flush=True)
    train = [tuple(v) for v in ds.viewpoints('train')]
    rng = random.Random(0)
    sampled = dict(validation=[tuple(v) for v in rng.sample(ds.viewpoints('validation'), args.targets)],
                   train=[tuple(v) for v in rng.sample(train, args.targets)])
    targets = [(role, v) for role in args.roles for v in sampled[role]]
    labels = [str(k) for k in args.iterations] if args.estimator == 'cgls' else [f'K{k}' for k in args.sparsity]
    report = dict(schema='gotcha_neighbor_prediction_v1', estimator=args.estimator, labels=labels,
                  neighbour_pulse_stride=args.neighbour_pulse_stride, neighbour_bin_stride=args.neighbour_bin_stride,
                  pass_phase_offsets=dict(PASS_PHASE) or None,
                  sparsity=args.sparsity if args.estimator == 'omp' else None, omp_batch=args.omp_batch, dataset_identity=ds.identity, region=args.region,
                  shard_root=str(args.shard_root) if args.shard_root else None, anchors=len(anchors),
                  pitch=args.pitch, half_extents=args.half_extents, iterations=args.iterations,
                  neighbourhoods=args.neighbourhoods, rows=[])
    for role, target in targets:
        started = time.perf_counter()
        specs = [e for spec in args.neighbourhoods
                 for e in ([f'same_azimuth_pass:{q}' for q in args.passes] if spec == 'same_azimuth_each_pass' else [spec])]
        wanted = {spec: neighbours(target, train, spec, args.passes) for spec in specs
                  if not (((spec == 'self' and not args.self_all_roles) or spec.startswith('same_azimuth'))
                          and role != 'train')}
        cache = {v: neighbour_system(ds, v, args.polarization, anchors, args.neighbour_pulse_stride,
                                     args.neighbour_bin_stride)
                 for v in sorted({v for vs in wanted.values() for v in vs} - {target})}
        cache[target] = sector_system(ds, target, args.polarization, anchors)
        A_t, y_t, look_t, obs_t = cache[target]
        if args.magnitude:
            rows_t = y_t.reshape(len(obs_t), -1).numpy()
            measured_t = mf_profiles(obs_t, rows_t, ds.region)
        for spec, views in wanted.items():
            row = dict(role=role, target=list(target), neighbourhood=spec, neighbour_sectors=len(views),
                       target_energy=float(y_t.abs().square().sum()),
                       gaps=[[p_, signed_gap(s_, target[1])] for p_, s_ in views], target_look=look_t,
                       neighbour_looks=[cache[v][2] for v in views])
            if not views:
                row['status'] = 'no TRAIN neighbours'
                report['rows'].append(row)
                continue
            systems = [cache[v][:2] for v in views]
            if args.magnitude and spec != 'self':
                # MF profiles align by range offset across passes; native rows (|.|) only on the target's
                # own frequency grid (bin counts differ between passes).
                profiles_sum, magnitude_sum, same_grid = None, None, 0
                for v in views:
                    A_v, y_v, _, obs_v = cache[v]
                    rows_v = y_v.reshape(len(obs_v), -1)
                    profiles = mf_profiles(obs_v, rows_v.numpy(), ds.region, offsets_from=obs_t)
                    profiles_sum = profiles if profiles_sum is None else [a + b for a, b in zip(profiles_sum, profiles)]
                    if np.array_equal(obs_v[0].frequencies_hz, obs_t[0].frequencies_hz):
                        index = [int(round(i * (len(obs_v) - 1) / max(len(obs_t) - 1, 1))) for i in range(len(obs_t))]
                        magnitude = rows_v[index].abs().double()
                        magnitude_sum = magnitude if magnitude_sum is None else magnitude_sum + magnitude
                        same_grid += 1
                sums = mf_scores([c / len(views) for c in profiles_sum], measured_t, peak)
                if magnitude_sum is not None:
                    sums.update(magnitude_sums(magnitude_sum / same_grid, y_t.reshape(len(obs_t), -1)))
                else:
                    sums.update(mag_error_fitted=float('nan'), mag_energy=1.0)
                row['copy_floor'] = dict(ratios(sums), sums=sums, magnitude_copy_neighbours=same_grid)
            if spec != 'self':
                copies = []
                for v in views:
                    A_v, y_v, _, obs_v = cache[v]
                    if (args.neighbour_bin_stride != 1
                            or not np.array_equal(obs_v[0].frequencies_hz, obs_t[0].frequencies_hz)):
                        continue
                    index = [int(round(i * (len(obs_v) - 1) / max(len(obs_t) - 1, 1))) for i in range(len(obs_t))]
                    copies.append(agreement(y_v.reshape(len(obs_v), -1)[index].reshape(-1), y_t)['abs_rho_c'])
                if copies:
                    row['copy_complex_abs_rho_c'] = float(np.mean(copies))
            energy = float(sum(y.abs().square().sum() for _, y in systems))
            row['by_iteration'] = {}
            fits = (((str(k), x, residual) for k, x, residual in cgls(systems, args.iterations))
                    if args.estimator == 'cgls' else
                    ((f'K{k}', x, residual) for k, x, residual in omp(systems, args.sparsity, args.omp_batch)))
            for k, x, residual in fits:
                prediction = A_t @ x.to(torch.complex64)
                row['by_iteration'][k] = dict(agreement(prediction, y_t), in_sample_rel_mse=residual / energy)
                if args.magnitude:
                    rows_p = prediction.reshape(len(obs_t), -1)
                    sums = dict(mf_scores(mf_profiles(obs_t, rows_p.numpy(), ds.region), measured_t, peak),
                                **magnitude_sums(rows_p, y_t.reshape(len(obs_t), -1)))
                    row['by_iteration'][k].update(ratios(sums), magnitude_sums=sums)
                    row['constant_oracle_mf_db_rel_mse'] = sums['db_constant_spread'] / sums['db_energy']
            report['rows'].append(row)
            best = max(v['abs_rho_c'] for v in row['by_iteration'].values())
            print(f'{role} {target} {spec}: {len(views)} sectors, |rho_c| (energy ratio) by {args.estimator} step '
                  + ', '.join(f'{k}:{v["abs_rho_c"]:.3f} ({v["energy_ratio"]:.2g})' for k, v in row['by_iteration'].items())
                  + f'; in-sample RelMSE at {labels[-1]}: {row["by_iteration"][labels[-1]]["in_sample_rel_mse"]:.3f}'
                  + f'; copy |rho_c| {row.get("copy_complex_abs_rho_c", float("nan")):.3f}; best {best:.3f}', flush=True)
        del cache
        print(f'  target done in {time.perf_counter() - started:.0f} s', flush=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    summary = {}
    for role in args.roles:
        for spec in sorted({r['neighbourhood'] for r in report['rows']}):
            for k in labels:
                rows = [r for r in report['rows']
                        if r['role'] == role and r['neighbourhood'] == spec and 'by_iteration' in r]
                if rows:
                    values = [r['by_iteration'][k]['abs_rho_c'] for r in rows]
                    energy = np.array([r['target_energy'] for r in rows])
                    fitted = np.array([r['by_iteration'][k]['rel_mse_fitted_gain'] for r in rows])
                    summary[f'{role} {spec} {k if args.estimator == "omp" else "it" + k}'] = dict(
                        n=len(rows), median_abs_rho_c=float(np.median(values)), mean_abs_rho_c=float(np.mean(values)),
                        pooled_rel_mse_fitted_gain=float((fitted * energy).sum() / energy.sum()),
                        median_energy_ratio=float(np.median([r['by_iteration'][k]['energy_ratio'] for r in rows])),
                        median_copy_abs_rho_c=(float(np.median([r['copy_complex_abs_rho_c'] for r in rows]))
                                               if all('copy_complex_abs_rho_c' in r for r in rows) else None))
                    if args.magnitude:
                        def pooled(source, key_error, key_energy):
                            return float(sum(s[key_error] for s in source) / sum(s[key_energy] for s in source))
                        recon = [r['by_iteration'][k]['magnitude_sums'] for r in rows]
                        block = dict(
                            mf_db=pooled(recon, 'db_error', 'db_energy'),
                            mf_db_fitted_scale=pooled(recon, 'db_error_fitted', 'db_energy'),
                            mf_linear_fitted_scale=pooled(recon, 'linear_error_fitted', 'linear_energy'),
                            magnitude_fitted_gain=pooled(recon, 'mag_error_fitted', 'mag_energy'),
                            constant_oracle_mf_db=pooled(recon, 'db_constant_spread', 'db_energy'))
                        copies = [r['copy_floor']['sums'] for r in rows if 'copy_floor' in r]
                        if copies:
                            same = [c for c in copies if np.isfinite(c['mag_error_fitted'])]
                            block.update(copy_mf_db=pooled(copies, 'db_error', 'db_energy'),
                                         copy_mf_db_fitted_scale=pooled(copies, 'db_error_fitted', 'db_energy'),
                                         copy_mf_linear_fitted_scale=pooled(copies, 'linear_error_fitted', 'linear_energy'),
                                         copy_magnitude_fitted_gain=(pooled(same, 'mag_error_fitted', 'mag_energy')
                                                                     if same else float('nan')))
                            block['margin_mf_db_fitted_scale'] = block['copy_mf_db_fitted_scale'] - block['mf_db_fitted_scale']
                            block['margin_magnitude_fitted_gain'] = (block['copy_magnitude_fitted_gain']
                                                                     - block['magnitude_fitted_gain'])
                        summary[f'{role} {spec} {k if args.estimator == "omp" else "it" + k}']['magnitude_pooled'] = block
    report['summary'] = summary
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    for key, value in summary.items():
        print(key, value, flush=True)


if __name__ == '__main__':
    main()
