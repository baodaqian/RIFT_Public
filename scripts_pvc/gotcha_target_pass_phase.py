#!/usr/bin/env python3
"""New-target step 4a: per-pass phase at a new target from TRAIN units only (a separate copy of
``gotcha_pass_phase_closure.py``, unchanged for the Camry; docs/RIFT_GOTCHA_Tune.md A73-A75).

Differences from the Camry tool: the region and its config are required (no Camry defaults); the split defaults to
the full-data unit split (stride 1, 10% of pass 4 held out), as the F5 arms and the new targets' training use; and
``--id-every k`` keeps every k-th of the all-TRAIN sector IDs, so the leave-one-pass-out measurement spans the full
azimuth circle at about the Camry measurement's size (39 IDs) instead of reading all 275. The original description
follows.

Cross-pass phase at the car from TRAIN units only: pairwise re-rendering and phase closure (CPU).

docs/RIFT_GOTCHA_Tune.md B54: the one-look re-rendering of a pass-4 unit from the same sector ID in another
pass carries a phase that clusters by source pass (about -5, +35, +90 and -10 degrees from passes 2, 3,
5 and 6), unlike the section-5b trihedral constants. This asks, on TRAIN units only, whether those are
per-pass constants at the car (a calibration offset, the same for every sector ID), a per-pass phase that
varies with azimuth (a per-pass position or range error), or an estimator effect.

For every sector ID of the unit split whose units are TRAIN (all 8 passes for the non-held IDs; the 7
non-held-out passes for the held-out IDs, whose pass-4 unit is VAL and is never read), each pass's unit is
backprojected onto the RIFT box grid (CGLS iteration 1: the scaled adjoint, isotropic anchors, the native
isolation kernel; ``scripts_pvc/gotcha_neighbor_prediction.py``'s ``sector_system``) and re-rendered on every
other pass's rows. The complex correlation C[p, q] = <A_p x_q, y_p> / (|A_p x_q| |y_p|) is recorded for
every ordered pair, with each unit's look (elevation, azimuth from the box centre). With the adjoint,
C[q, p] = conj(C[p, q]) up to the positive scales, so information about closure comes from triangles:
per ID the per-pass phases phi_p are the phases of the leading eigenvector of the Hermitian matrix
H = C (the phase-synchronization estimate), and the residual arg(C[p, q] exp(-i (phi_p - phi_q))) is the
non-closure. ``--summarize`` then tests, across IDs, a per-pass constant against a per-pass constant plus
an azimuth sinusoid (a horizontal position offset gives (4 pi / lambda) (dx cos az + dy sin az) cos el),
and compares with the section-5b trihedral offsets.

Reads TRAIN rows only; VAL and TEST rows are never read.

    python scripts_pvc/gotcha_pass_phase_closure.py --shard-root <keep6 shards> --output closure.json
    python scripts_pvc/gotcha_pass_phase_closure.py --summarize closure.json --output closure_summary.json
"""
import argparse
import json
import math
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
from gotcha_neighbor_prediction import sector_system  # noqa: E402

SCHEMA = 'gotcha_pass_phase_closure_v1'
# Section 5b (TRAIN-only trihedrals): median phase of pass p relative to pass 1, radians.
TRIHEDRAL_PHASE = {1: 0.0, 2: -0.22, 3: -0.36, 4: -0.63, 5: -0.34, 6: -0.82, 7: -1.20, 8: -1.20}


def backproject(A, y):
    """CGLS iteration 1 from zero: alpha A^H y with alpha = |A^H y|^2 / |A A^H y|^2 (real, positive)."""
    s = (y.conj() @ A).conj()
    q = A @ s
    return s * (float(s.abs().square().sum()) / float(q.abs().square().sum()))


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
    anchors = box_anchors(dict(kind='box', half_extents=args.half_extents, pitch=args.pitch), 'cpu').numpy()
    ids = selected[:args.limit or None]
    report = dict(schema=SCHEMA, split=split, dataset_identity=ds.identity, shard_root=str(args.shard_root),
                  region=args.region, anchors=len(anchors), pitch=args.pitch, estimator='CGLS iteration 1 (scaled adjoint)',
                  trihedral_phase_rad=TRIHEDRAL_PHASE, ids=[])
    for s in ids:
        started = time.perf_counter()
        views = [(p, s) for p in passes if (p, s) in train_views]
        systems = {v: sector_system(ds, v, args.polarization, anchors) for v in views}
        x = {v: backproject(systems[v][0], systems[v][1]) for v in views}
        n = len(views)
        C = np.zeros((n, n), dtype=np.complex128)
        for i, vp in enumerate(views):
            A_p, y_p = systems[vp][0], systems[vp][1].to(torch.complex128)
            for j, vq in enumerate(views):
                pred = (A_p @ x[vq]).to(torch.complex128)
                inner = complex((pred.conj() * y_p).sum())
                C[i, j] = inner / math.sqrt(float(pred.abs().square().sum()) * float(y_p.abs().square().sum()))
        report['ids'].append(dict(sector_id=int(s), heldout=bool(s in set(heldout)), passes=[v[0] for v in views],
                                  looks=[systems[v][2] for v in views],
                                  energies=[float(systems[v][1].abs().square().sum()) for v in views],
                                  C_re=C.real.tolist(), C_im=C.imag.tolist()))
        diag = np.abs(np.diag(C))
        print(f'sector {s}: {n} passes, self |rho_c| {diag.min():.3f}-{diag.max():.3f}, '
              f'off-diagonal |rho_c| median {np.median(np.abs(C[~np.eye(n, dtype=bool)])):.3f}, '
              f'{time.perf_counter() - started:.0f} s', flush=True)
        del systems, x
        args.output.write_text(json.dumps(report) + '\n')
    args.output.write_text(json.dumps(report) + '\n')


def measure_loo(args):
    """Leave one pass out (B55 next step): fit the other passes of an ID jointly, predict the left-out pass.

    Only IDs whose 8 units are all TRAIN (the non-held IDs). The joint fit has height resolution, so the
    phase of <yhat_p, y_p> is pass p's offset from the consensus of the others, not a one-look layover effect.
    """
    from gotcha_unit_split_val_floor import cgls
    torch.set_num_threads(args.threads)
    passes = tuple(range(1, 9))
    ds = GOTCHADataset(args.dataset_root, shard_root=args.shard_root, passes=passes,
                       polarizations=(args.polarization,), region=load_region(args.region, args.region_config),
                       pulses_per_sector=0, frequency_stride=args.frequency_stride)
    split = apply_unit_split(ds, stride=args.stride, heldout_pass=args.heldout_pass,
                             heldout_fraction=args.heldout_fraction)
    selected, heldout = unit_split_ids(args.stride, args.heldout_fraction)
    train_views = {tuple(v) for v in ds.viewpoints('train')}
    anchors = box_anchors(dict(kind='box', half_extents=args.half_extents, pitch=args.pitch), 'cpu').numpy()
    ids = [s_ for s_ in selected if all((p_, s_) in train_views for p_ in passes)][::args.id_every][:args.limit or None]
    report = dict(schema=SCHEMA + '_loo', split=split, dataset_identity=ds.identity, shard_root=str(args.shard_root),
                  anchors=len(anchors), pitch=args.pitch, iterations=args.loo_iterations,
                  estimator='CGLS from zero on the other passes of the ID, predicting the left-out pass', ids=[])
    for s_ in ids:
        started = time.perf_counter()
        views = [(p_, s_) for p_ in passes]
        systems = {v: sector_system(ds, v, args.polarization, anchors) for v in views}
        rows = []
        for v in views:
            others = [systems[u][:2] for u in views if u != v]
            A_t, y_t = systems[v][0], systems[v][1].to(torch.complex128)
            by_iteration = {}
            for k, x, residual in cgls(others, args.loo_iterations):
                pred = (A_t @ x.to(torch.complex64)).to(torch.complex128)
                inner = complex((pred.conj() * y_t).sum())
                by_iteration[str(k)] = dict(inner_re=inner.real, inner_im=inner.imag,
                                            prediction=float(pred.abs().square().sum()),
                                            target=float(y_t.abs().square().sum()),
                                            in_sample_rel_mse=residual / float(sum(y.abs().square().sum() for _, y in others)))
            rows.append(dict(pass_id=v[0], look=systems[v][2], by_iteration=by_iteration))
        report['ids'].append(dict(sector_id=int(s_), passes=rows))
        k0 = str(args.loo_iterations[0])
        print(f'sector {s_}: LOO phase (deg) at iteration {k0}: ' + ' '.join(
            f"{r['pass_id']}:{math.degrees(math.atan2(r['by_iteration'][k0]['inner_im'], r['by_iteration'][k0]['inner_re'])):+.0f}"
            for r in rows) + f'  ({time.perf_counter() - started:.0f} s)', flush=True)
        del systems
        args.output.write_text(json.dumps(report) + '\n')
    args.output.write_text(json.dumps(report) + '\n')


def summarize_loo(args):
    report = json.loads(Path(args.summarize_loo).read_text())
    out = dict(schema=SCHEMA + '_loo_summary', source=str(args.summarize_loo), by_iteration={})
    for k in map(str, report['iterations']):
        per_pass, pooled = {}, dict(inner=0j, prediction=0.0, target=0.0, inner_corrected=0j)
        phasors = {}
        for item in report['ids']:
            for r in item['passes']:
                u = r['by_iteration'][k]
                c = complex(u['inner_re'], u['inner_im'])
                phasors.setdefault(r['pass_id'], []).append((c, u, math.radians(r['look']['azimuth_deg'])))
        for p_, items in sorted(phasors.items()):
            z = np.asarray([c / abs(c) for c, _, _ in items])
            weights = np.asarray([abs(c) for c, _, _ in items])
            mean = z.mean()
            rho = np.asarray([abs(c) / math.sqrt(u['prediction'] * u['target']) for c, u, _ in items])
            az = np.asarray([a for _, _, a in items])
            ph = np.angle(z)
            J = np.stack([np.ones_like(az), np.cos(az), np.sin(az)], 1)
            theta = np.array([np.angle(mean), 0.0, 0.0])
            for _ in range(50):
                step, *_ = np.linalg.lstsq(J, wrap(ph - J @ theta), rcond=None)
                theta = theta + step
                if np.abs(step).max() < 1e-10:
                    break
            per_pass[str(p_)] = dict(ids=len(items), mean_phase_deg=float(np.degrees(np.angle(mean))), resultant=float(abs(mean)),
                                     weighted_mean_phase_deg=float(np.degrees(np.angle((weights * z).sum()))),
                                     median_abs_rho_c=float(np.median(rho)),
                                     residual_rms_constant_deg=float(np.degrees(np.sqrt((wrap(ph - np.angle(mean)) ** 2).mean()))),
                                     harmonic_amplitude_deg=float(np.degrees(np.hypot(theta[1], theta[2]))),
                                     residual_rms_harmonic_deg=float(np.degrees(np.sqrt((wrap(ph - J @ theta) ** 2).mean()))))
            correction = np.exp(-1j * np.angle((weights * z).sum()))    # one TRAIN-fitted constant per pass
            for c, u, _ in items:
                pooled['inner'] += c
                pooled['inner_corrected'] += c * correction
                pooled['prediction'] += u['prediction']
                pooled['target'] += u['target']
        norm = math.sqrt(pooled['prediction'] * pooled['target'])
        out['by_iteration'][k] = dict(per_pass=per_pass,
                                      pooled_abs_rho_c=abs(pooled['inner']) / norm,
                                      pooled_abs_rho_c_per_pass_constant=abs(pooled['inner_corrected']) / norm,
                                      rel_mse_best_single_gain=1 - (abs(pooled['inner']) / norm) ** 2,
                                      rel_mse_best_single_gain_after_per_pass_constant=1 - (abs(pooled['inner_corrected']) / norm) ** 2)
        b = out['by_iteration'][k]
        print(f"iteration {k}: pooled |rho_c| {b['pooled_abs_rho_c']:.3f} -> {b['pooled_abs_rho_c_per_pass_constant']:.3f} with one "
              f"constant per pass (single-gain RelMSE {b['rel_mse_best_single_gain']:.3f} -> "
              f"{b['rel_mse_best_single_gain_after_per_pass_constant']:.3f})")
        for p_, d in per_pass.items():
            print(f"  pass {p_}: n={d['ids']} mean {d['mean_phase_deg']:+7.1f} deg (R {d['resultant']:.2f}; |rho_c|-weighted "
                  f"{d['weighted_mean_phase_deg']:+7.1f}) median |rho_c| {d['median_abs_rho_c']:.2f} | rms about the constant "
                  f"{d['residual_rms_constant_deg']:.0f} deg, with azimuth harmonic {d['residual_rms_harmonic_deg']:.0f} deg "
                  f"(amplitude {d['harmonic_amplitude_deg']:.0f} deg)")
    Path(args.output).write_text(json.dumps(out, indent=2) + '\n')


def synchronize(C):
    """Per-pass phases from the leading eigenvector of the Hermitian part of C (phase synchronization)."""
    H = 0.5 * (C + C.conj().T)
    np.fill_diagonal(H, 0)
    w, V = np.linalg.eigh(H)
    z = V[:, -1]
    return np.angle(z)


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def summarize(args):
    report = json.loads(Path(args.summarize).read_text())
    lam = 299_792_458.0 / 9.6e9
    rows = []
    for item in report['ids']:
        C = np.asarray(item['C_re']) + 1j * np.asarray(item['C_im'])
        passes = item['passes']
        phi = synchronize(C)
        ref = passes.index(4) if 4 in passes else None
        anchor = passes.index(1) if 1 in passes else 0
        phi = wrap(phi - phi[anchor])                       # relative to pass 1 (the section-5b convention)
        n = len(passes)
        off = ~np.eye(n, dtype=bool)
        model = np.exp(1j * (phi[:, None] - phi[None, :]))
        # C[p, q] = <A_p x_q, y_p>: its phase is arg(y_p) - arg(prediction from q) = phi_p - phi_q.
        residual = wrap(np.angle(C * model.conj()))
        weights = np.abs(C)
        # Implied TRAIN floor for one coherent scene and one gain (B55 b): with the ID-common phase absorbed,
        # 1 - |sum_p E_p exp(i phi_p)|^2 / (sum_p E_p)^2; and after also removing the best phase linear in the
        # pass elevation (what a height shift of the ID's scatterers could absorb; weighted circular LS).
        E = np.asarray(item['energies'])
        el = np.radians([l['elevation_deg'] for l in item['looks']])
        floor_measured = float(1 - abs((E * np.exp(1j * phi)).sum()) ** 2 / E.sum() ** 2)
        best = (floor_measured, 0.0, 0.0)
        for slope in np.linspace(-400.0, 400.0, 1601):          # rad per rad of elevation; |h| <= ~1.4 m
            z = (E * np.exp(1j * (phi - slope * (el - el.mean())))).sum()
            f = float(1 - abs(z) ** 2 / E.sum() ** 2)
            if f < best[0]:
                best = (f, float(slope), float(np.angle(z)))
        rows.append(dict(sector_id=item['sector_id'], heldout=item['heldout'], passes=passes,
                         energy=float(E.sum()), implied_floor_measured=floor_measured,
                         implied_floor_after_height=best[0], height_slope_rad_per_rad=best[1],
                         height_equivalent_m=float(best[1] / (4 * math.pi / lam * math.cos(el.mean()))),
                         azimuth_deg=float(np.mean([l['azimuth_deg'] for l in item['looks']])),
                         elevation_deg={str(p): l['elevation_deg'] for p, l in zip(passes, item['looks'])},
                         phase_rad={str(p): float(v) for p, v in zip(passes, phi)},
                         closure_residual_rms_rad=float(np.sqrt((weights[off] * residual[off] ** 2).sum() / weights[off].sum())),
                         offdiag_abs_rho_c_median=float(np.median(np.abs(C[off])))))
    out = dict(schema=SCHEMA + '_summary', source=str(args.summarize), convention='phase of pass p relative to pass 1 '
               '(section 5b convention: data of pass p ~ exp(i phi_p) x pass-1-consistent data)', ids=rows)
    per_pass = {}
    for p in range(2, 9):
        az, ph = [], []
        for r in rows:
            if str(p) in r['phase_rad'] and '1' in r['phase_rad']:
                az.append(math.radians(r['azimuth_deg']))
                ph.append(r['phase_rad'][str(p)])
        if not ph:
            continue
        az, ph = np.asarray(az), np.asarray(ph)
        z = np.exp(1j * ph)
        constant = float(np.angle(z.mean()))
        resultant = float(np.abs(z.mean()))
        # Per-pass constant plus a first azimuth harmonic, fitted on the circle by Gauss-Newton from the constant.
        theta = np.array([constant, 0.0, 0.0])
        for _ in range(50):
            model = theta[0] + theta[1] * np.cos(az) + theta[2] * np.sin(az)
            r_ = wrap(ph - model)
            J = np.stack([np.ones_like(az), np.cos(az), np.sin(az)], 1)
            step, *_ = np.linalg.lstsq(J, r_, rcond=None)
            theta = theta + step
            if np.abs(step).max() < 1e-10:
                break
        res_const = wrap(ph - constant)
        res_harm = wrap(ph - (theta[0] + theta[1] * np.cos(az) + theta[2] * np.sin(az)))
        amplitude = float(np.hypot(theta[1], theta[2]))
        per_pass[str(p)] = dict(ids=len(ph), circular_mean_rad=constant, resultant=resultant,
                                trihedral_rad=TRIHEDRAL_PHASE[p],
                                residual_rms_constant_rad=float(np.sqrt((res_const ** 2).mean())),
                                harmonic=dict(constant_rad=float(wrap(theta[0])), cos_rad=float(theta[1]),
                                              sin_rad=float(theta[2]), amplitude_rad=amplitude,
                                              horizontal_offset_mm=float(amplitude * lam / (4 * math.pi) / math.cos(math.radians(44.4)) * 1e3),
                                              residual_rms_rad=float(np.sqrt((res_harm ** 2).mean()))))
    out['per_pass'] = per_pass
    out['closure_residual_rms_rad_median'] = float(np.median([r['closure_residual_rms_rad'] for r in rows]))
    energy = np.asarray([r['energy'] for r in rows])
    out['implied_train_floor'] = dict(
        measured=float((energy * np.asarray([r['implied_floor_measured'] for r in rows])).sum() / energy.sum()),
        after_height_absorption=float((energy * np.asarray([r['implied_floor_after_height'] for r in rows])).sum() / energy.sum()),
        median_height_equivalent_m=float(np.median([abs(r['height_equivalent_m']) for r in rows])),
        note='energy-weighted over IDs; one coherent scene, one gain; phase-estimation noise inflates both')
    print('implied TRAIN floor (energy-weighted): measured %.3f, after height absorption %.3f; median |height equivalent| %.2f m'
          % (out['implied_train_floor']['measured'], out['implied_train_floor']['after_height_absorption'],
             out['implied_train_floor']['median_height_equivalent_m']))
    Path(args.output).write_text(json.dumps(out, indent=2) + '\n')
    print(f"closure residual rms (weighted, per ID) median {out['closure_residual_rms_rad_median']:.3f} rad over {len(rows)} IDs")
    for p, d in per_pass.items():
        h = d['harmonic']
        print(f"pass {p}: n={d['ids']:2d} constant {math.degrees(d['circular_mean_rad']):7.1f} deg (resultant {d['resultant']:.2f}, "
              f"rms {math.degrees(d['residual_rms_constant_rad']):5.1f} deg) | trihedral {math.degrees(d['trihedral_rad']):7.1f} deg | "
              f"+harmonic: amplitude {math.degrees(h['amplitude_rad']):6.1f} deg (~{h['horizontal_offset_mm']:.1f} mm), "
              f"rms {math.degrees(h['residual_rms_rad']):5.1f} deg", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset-root', default=os.environ.get('GOTCHA_DATA_ROOT',
                   '/scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined'))
    p.add_argument('--shard-root', type=Path)
    p.add_argument('--region', required=True)
    p.add_argument('--region-config', type=Path, required=True)
    p.add_argument('--polarization', default='hh')
    p.add_argument('--frequency-stride', type=int, default=2)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--heldout-pass', type=int, default=4)
    p.add_argument('--heldout-fraction', type=float, default=0.1)
    p.add_argument('--half-extents', type=float, nargs=3, default=[3.0, 1.5, 1.25])
    p.add_argument('--pitch', type=float, default=0.125)
    p.add_argument('--limit', type=int, default=0, help='first N sector IDs (smoke tests only)')
    p.add_argument('--id-every', type=int, default=7, help='leave-one-pass-out: every k-th all-TRAIN sector ID (spans the circle)')
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--summarize', type=Path, help='analyse a measured report')
    p.add_argument('--loo', action='store_true', help='leave-one-pass-out measurement instead of the pairwise one')
    p.add_argument('--loo-iterations', type=int, nargs='+', default=[1, 2, 3, 5])
    p.add_argument('--summarize-loo', type=Path, help='analyse a leave-one-pass-out report')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    if args.summarize:
        summarize(args)
        return
    if args.summarize_loo:
        summarize_loo(args)
        return
    if args.shard_root is None:
        p.error('--shard-root is required to measure')
    if args.loo:
        measure_loo(args)
    else:
        measure(args)


if __name__ == '__main__':
    main()
