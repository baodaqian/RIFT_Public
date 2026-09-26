#!/usr/bin/env python3
"""Collect the final RIFT-dataset evaluation (six methods x six scenes) into tables (PVC).

Reads, without recomputing anything:

* signal scores per role from ``SIGNAL`` (``<obj>_<method>_<role>_metrics.json``: RIFT from
  ``scripts_pvc/eval_b787_range_power_pvc.py`` -- MF power from its own coherent prediction --,
  the others from ``scripts_pvc/eval_rift_dataset_heldout_pvc.py``); RadarSplat validation falls back
  to ``eval_heldout_checks/<obj>_rs_logpower_val_metrics.json`` (the same C8 final checkpoints);
* floors: TRAIN-mean constant (``<obj>_train_constant_<role>.json``, full ROI; RadarSplat crop bins
  from ``eval_heldout_checks/<obj>_rs_train_constant_val.json``) and the oracle best constant
  (``reference_floors`` of the same-bin runs);
* geometry from ``GEOMETRY/<obj>/geometry_metrics.json`` (six-method postflight: Chamfer, IoU,
  precision, F1 at t = 0.20; 48^3 lattice primary, 192^3 densified alongside).

Writes ``OUT/final_rift_dataset_eval.json`` and ``OUT/final_rift_dataset_eval.md``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path('/scratch/user/u.db364833/RIFT_runs/final_eval_20260923')
CHECKS = Path('/scratch/user/u.db364833/RIFT_runs/eval_heldout_checks')
SCENES = ('a320', 'b787', 'x59', 'firetruck', 'race_car', 'loader')
METHODS = (('rift', 'RIFT'), ('spinr', 'SpINR'), ('geraf', 'GeRaF'), ('radar_fields', 'Radar Fields'),
           ('radarsplat', 'RadarSplat'), ('sugavanam_ertin', 'Sugavanam–Ertin'))
GEOMETRY_KEY = dict(sugavanam_ertin='se')
SUFFIX = dict(validation='val', test='test')


def load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def signal(obj, method, role):
    found = load(ROOT/'signal'/f'{obj}_{method}_{SUFFIX[role]}_metrics.json')
    if found is None and method == 'radarsplat':
        found = load((ROOT/'signal' if role == 'test' else CHECKS)/f'{obj}_rs_logpower_{SUFFIX[role]}_metrics.json')
    if found is None:
        return None
    complete = found.get('status') in ('complete', 'production_complete') or (
        found.get('views_complete') is not None and found.get('views_complete') == found.get('views_total'))
    return dict(coherent=found.get('coherent_complex_rel_mse'), normalized=found.get('normalized_range_power_rel_mse'),
                linear=found.get('linear_range_power_rel_mse'), views=found.get('views_complete'), complete=bool(complete),
                oracle=(found.get('reference_floors') or {}), source=str(Path(found.get('label', ''))))


def floors(obj, role):
    full = load(ROOT/'signal'/f'{obj}_train_constant_{SUFFIX[role]}.json')
    crop = load((ROOT/'signal' if role == 'test' else CHECKS)/f'{obj}_rs_train_constant_{SUFFIX[role]}.json')
    pick = lambda d: None if d is None else dict(normalized=d['scores'][0]['normalized_range_power_rel_mse'],
                                                 linear=d['scores'][0]['linear_range_power_rel_mse'])
    return dict(full_roi=pick(full), radarsplat_crop=pick(crop))


def fmt(value, digits=3):
    if value is None:
        return '—'
    return f'{value:.{digits}g}' if abs(value) < 1e-2 or abs(value) >= 100 else f'{value:.{digits}f}'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'tables')
    parser.add_argument('--roles', nargs='+', default=['validation', 'test'])
    args = parser.parse_args(argv)
    result = dict(signal={}, floors={}, geometry={})
    lines = ['# RIFT dataset: final six-method evaluation', '',
             'Checkpoints: validation-selected best (RIFT, SpINR, GeRaF, Radar Fields), final (RadarSplat C8 log power, '
             'SE Stage 1 selected fields / Stage 2 surface). "—": not applicable or pending. '
             'RIFT MF power is computed from its own predicted coherent spectrum.', '']
    for role in args.roles:
        lines += [f'## Signal, {role}', '',
                  '| Scene | Method | Coherent RelMSE | MF power normalized | MF power linear | views |',
                  '| --- | --- | --- | --- | --- | --- |']
        for obj in SCENES:
            result['floors'].setdefault(role, {})[obj] = f = floors(obj, role)
            for key, title in METHODS:
                s = signal(obj, key, role)
                result['signal'].setdefault(role, {}).setdefault(obj, {})[key] = s
                if s is None:
                    lines.append(f'| {obj} | {title} | pending | pending | pending | |')
                    continue
                note = '' if s['complete'] else ' (partial)'
                linear = '— (dB-derived, not comparable)' if key == 'radar_fields' else fmt(s['linear'])
                lines.append(f"| {obj} | {title} | {fmt(s['coherent'])} | {fmt(s['normalized'])} | {linear} | "
                             f"{s['views']}{note} |")
            for label, floor in (('TRAIN-mean constant (full ROI)', f['full_roi']),
                                 ('TRAIN-mean constant (RadarSplat crop bins)', f['radarsplat_crop'])):
                if floor:
                    lines.append(f"| {obj} | *{label}* | — | {fmt(floor['normalized'])} | {fmt(floor['linear'])} | |")
        lines.append('')
    lines += ['## Geometry (fixed t = 0.20; 48³ lattice primary, 192³ densified in brackets)', '',
              '| Scene | Method | Chamfer (surface, mean L2 mm) | Chamfer (pytorch3d, surface) | IoU solid | IoU shell | '
              'Precision | Recall | F1 |', '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for obj in SCENES:
        g = load(ROOT/'geometry'/obj/'geometry_metrics.json')
        result['geometry'][obj] = g
        for key, title in METHODS:
            entry = (g or {}).get('methods', {}).get(GEOMETRY_KEY.get(key, key))
            if not entry or entry.get('status') != 'scored':
                lines.append(f'| {obj} | {title} | pending | | | | | | |')
                continue
            m = entry['metrics']
            a, b = m['g48_lattice'], next(v for k, v in m.items() if k.startswith('dense_'))
            both = lambda k, d=3: f'{fmt(a.get(k), d)} [{fmt(b.get(k), d)}]'
            lines.append(f"| {obj} | {title} | {both('surface_l2_mm', 3)} | {both('cd_surface')} | {both('iou_solid')} | "
                         f"{both('iou_shell')} | {both('precision')} | {both('recall')} | {both('f1')} |")
    lines += ['', 'Figures: standalone max-intensity projections (template `_draw_field`), one per method and view '
              '(cameras on -x, -y, -z), in `geometry/<scene>/mip_panels/{no_mesh,with_mesh}/<scene>_<method>_mip_{negx,negy,negz}.{png,pdf}`; '
              'one colour scale for every panel (inferno over [t=0.20, 1] of the per-method min-max magnitude, black below t) '
              'with its colour bar at panel height in `geometry/<scene>/mip_panels/<scene>_mip_colorbar.{png,pdf}`.']
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'final_rift_dataset_eval.json').write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + '\n')
    (args.out/'final_rift_dataset_eval.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
