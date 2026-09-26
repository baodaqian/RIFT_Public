#!/usr/bin/env python3
"""Manuscript numbers for the RIFT-dataset tables (per-object signal, per-object geometry, six-scene aggregate).

Reads the final evaluation (``compile_final_rift_dataset_eval_pvc.py``'s sources; nothing is recomputed) and
writes one JSON with every value the manuscript tables use, plus the LaTeX cell strings:

* signal, reserved test: native coherent complex RelMSE and the common range-power RelMSE (the release's
  normalized-dB intensity, TRAIN peak, 60 dB, whole ROI), both in percent. RadarSplat's power image covers
  only its crop bins, so its range-power value is flagged ``crop``; the TRAIN-mean constant floors
  (whole ROI and RadarSplat crop bins) are reported alongside;
* geometry, fixed t = 0.20 on the 48^3 lattice (the RIFT geometry evaluator's primary readout): symmetric
  squared Chamfer against the surface truth (1e-4 m^2), maximum Hausdorff and HD95 (mm), solid IoU, F1;
* aggregate: the unweighted mean over the six scenes, only when all six are available;
* Sugavanam-Ertin appears twice: ``se`` (two-stage; geometry from the Stage-2 surface) and ``se_stage1``
  (Stage 1 only; geometry from the Stage-1 field). Both share the Stage-1 signal.
* Backprojection (``backprojection``, the train-only matched-filter image) has geometry only and is ranked there.

Bold marks the best populated value per scene (aggregate: per column) and metric.

    python scripts_pvc/paper_rift_dataset_numbers_pvc.py --fallback geometry_nogeraf --out numbers.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import scripts_pvc.compile_final_rift_dataset_eval_pvc as final  # noqa: E402

SCENES = ('b787', 'a320', 'x59', 'firetruck', 'race_car', 'loader')        # manuscript column order
SIGNAL_KEY = dict(rift='rift', spinr='spinr', radar_fields='radar_fields', geraf='geraf', radarsplat='radarsplat',
                  se='sugavanam_ertin', se_stage1='sugavanam_ertin')     # SE's signal is its Stage-1 prediction
GEOMETRY = (('chamfer', 'cd_surface', 1e4, False), ('hausdorff', 'surface_hausdorff_mm', 1.0, False),
            ('hd95', 'surface_hd95_mm', 1.0, False), ('iou', 'iou_solid', 1.0, True), ('f1', 'f1', 1.0, True))
NO_COMPLEX = {'radar_fields', 'radarsplat'}
GEOMETRY_ONLY = ('backprojection',)  # ranked geometry rows with no signal observable
DIAGNOSTIC = ()                      # geometry-only diagnostic rows, reported but never ranked (none at present)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=final.ROOT)
    p.add_argument('--fallback', action='append', default=[], help='geometry root used where geometry/<scene> lacks a method')
    p.add_argument('--out', type=Path, required=True)
    return p.parse_args(argv)


def geometry(root, fallbacks, scene, method):
    for name in ['geometry', *fallbacks]:
        found = final.load(root/name/scene/'geometry_metrics.json')
        entry = (found or {}).get('methods', {}).get(method)
        if entry and entry.get('status') == 'scored':
            g = entry['metrics']['g48_lattice']
            values = {key: (None if not math.isfinite(g[field]) else g[field] * scale) for key, field, scale, _ in GEOMETRY}
            return dict(values, points=g['point_count'], source=f'{name}/{scene}')
    return None


def signal(scene, method):
    if method not in SIGNAL_KEY:
        return None
    s = final.signal(scene, SIGNAL_KEY[method], 'test')
    if s is None or not s['complete']:
        return None
    return dict(complex=None if method in NO_COMPLEX else 100 * s['coherent'], power=100 * s['normalized'],
                crop=(method == 'radarsplat'), views=s['views'], source=s['source'])


def sig4(value):
    """Four significant figures in fixed notation (trailing zeros kept); below 1e-4 a \\scriptsize power of ten; exact 0 as '0'."""
    if value == 0:
        return '0'
    if abs(value) < 1e-4:
        mantissa, exponent = f'{value:.3e}'.split('e')
        return f'{{\\scriptsize${mantissa}\\times10^{{{int(exponent)}}}$}}'   # smaller, to keep the column narrow
    rounded = float(f'{value:.4g}')                  # rounding can carry into the next decade (9.99995 -> 10.00)
    digits = max(0, 3 - int(math.floor(math.log10(abs(rounded)))))
    return f'{rounded:.{digits}f}'


def pct(value):
    """Percent with four significant figures, never in exponent form."""
    return f'{sig4(value)}\\%'


def plain(value, key):
    """Four significant figures for every geometry metric (user, 2026-09-23)."""
    return sig4(value)


def best(values, higher):
    numbers = [v for v in values if v is not None]
    if not numbers:
        return None
    return max(numbers) if higher else min(numbers)


def main(argv=None):
    args = parse_args(argv)
    methods = tuple(SIGNAL_KEY) + GEOMETRY_ONLY
    table = dict(signal={}, geometry={}, floors={}, aggregate={})
    for scene in SCENES:
        table['signal'][scene] = {m: signal(scene, m) for m in methods}
        table['geometry'][scene] = {m: geometry(args.root, args.fallback, scene, m) for m in methods + DIAGNOSTIC}
        f = final.floors(scene, 'test')
        table['floors'][scene] = {k: (None if v is None else 100 * v['normalized']) for k, v in f.items()}
    # six-scene macro averages (only when all six scenes are present)
    for m in methods:
        agg = {}
        for key in ('complex', 'power'):
            vals = [(table['signal'][s][m] or {}).get(key) for s in SCENES]
            agg[key] = sum(vals) / len(vals) if all(v is not None for v in vals) else None
        for key, *_ in GEOMETRY:
            vals = [(table['geometry'][s][m] or {}).get(key) for s in SCENES]
            agg[key] = sum(vals) / len(vals) if all(v is not None for v in vals) else None
        table['aggregate'][m] = agg
    for key in ('full_roi', 'radarsplat_crop'):
        vals = [table['floors'][s][key] for s in SCENES]
        table['aggregate'][f'floor_{key}'] = sum(vals) / len(vals) if all(v is not None for v in vals) else None
    # LaTeX cells, with the best populated value in bold
    cells = dict(signal={}, geometry={}, aggregate={})
    for scene in SCENES:
        for key in ('complex', 'power'):
            candidates = {m: (table['signal'][scene][m] or {}).get(key) for m in methods}
            if key == 'power':
                candidates['radarsplat'] = None           # crop-bin score is not ranked against whole-ROI scores
            top = best(candidates.values(), higher=False)
            for m in methods:
                v = (table['signal'][scene][m] or {}).get(key)
                text = None if v is None else pct(v)
                if text and v == top:
                    text = f'$\\mathbf{{{text[:-2]}}}$\\%'
                cells['signal'].setdefault(scene, {}).setdefault(key, {})[m] = text
        for key, _, _, higher in GEOMETRY:
            candidates = [(table['geometry'][scene][m] or {}).get(key) for m in methods]
            top = best(candidates, higher)
            for m in methods + DIAGNOSTIC:
                v = (table['geometry'][scene][m] or {}).get(key)
                text = None if v is None else plain(v, key)
                if text is not None and v == top and m not in DIAGNOSTIC:
                    text = f'$\\mathbf{{{text}}}$'
                cells['geometry'].setdefault(scene, {}).setdefault(key, {})[m] = text
    for key in ('complex', 'power', *[g[0] for g in GEOMETRY]):
        higher = next((g[3] for g in GEOMETRY if g[0] == key), False)
        candidates = {m: table['aggregate'][m][key] for m in methods}
        if key == 'power':
            candidates['radarsplat'] = None
        top = best(candidates.values(), higher)
        for m in methods:
            v = table['aggregate'][m][key]
            text = None if v is None else (pct(v) if key in ('complex', 'power') else plain(v, key))
            if text is not None and v == top:
                text = f'$\\mathbf{{{text[:-2]}}}$\\%' if key in ('complex', 'power') else f'$\\mathbf{{{text}}}$'
            cells['aggregate'].setdefault(key, {})[m] = text
    cells['floors'] = {scene: {k: (None if v is None else pct(v)) for k, v in table['floors'][scene].items()}
                       for scene in SCENES}
    cells['aggregate_floors'] = {k: (None if table['aggregate'][f'floor_{k}'] is None else pct(table['aggregate'][f'floor_{k}']))
                                 for k in ('full_roi', 'radarsplat_crop')}
    table['cells'] = cells
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=2, sort_keys=True) + '\n')
    for scene in SCENES:
        print(scene, {m: (cells['signal'][scene]['complex'][m], cells['signal'][scene]['power'][m]) for m in methods})
    print('aggregate', json.dumps(cells['aggregate'], ensure_ascii=False))
    print(f'NUMBERS=PASS {args.out}')


if __name__ == '__main__':
    main()
