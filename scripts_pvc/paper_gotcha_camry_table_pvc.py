#!/usr/bin/env python3
"""Rows of the manuscript's provisional GOTCHA Camry table (CPU; reads run records only, no radar response).

Every learned row uses one checkpoint for all its cells: the validation-selected checkpoint (user, 2026-09-24).

* TRAIN / VAL: pooled full-native complex RelMSE (RIFT's ``role_fit`` domain) over all 2,410 TRAIN and the 30
  pass-4 VAL units of the full split. VAL from each history's selection metric. TRAIN: RIFT's trainer fit at that
  epoch when it logged one (even epochs), else ``gotcha_rift_role_fit_pvc.py`` (``--rift-role-fit``); SpINR from
  ``gotcha_spinr_role_fit_pvc.py`` (``--spinr-role-fit``). TBD until available; a role fit's VAL must reproduce the
  history's.
* Geometry: rows at the fixed threshold t = 0.20 (the RIFT-dataset readout) from ``eval_gotcha_geometry_pvc.py`` or,
  for the paper's protocol, ``eval_gotcha_geometry_protocol_pvc.py``: Chamfer (m^2), maximum Hausdorff and HD95 (m),
  solid IoU and F1 at tau.
* VAL references: the pre-registered one-look fits of B62 (``floor_validation_F5split_v2.json``, CGLS iteration 1).
* Backprojection: no signal (N/A). Sugavanam--Ertin is not part of the GOTCHA comparison (user, 2026-09-24 14:35).

Writes the rows (between \\midrule and \\bottomrule) into the manuscript in place, between the
``% BEGIN camry-table-rows`` / ``% END camry-table-rows`` markers of ``--tex`` (so the table needs no extra file), and
records them in ``<out-dir>/camry_table_rows.tex`` (not input by the manuscript) and ``camry_table_numbers.json``.

    python scripts_pvc/paper_gotcha_camry_table_pvc.py --out-dir manuscripts/iclr27/figures/gotcha_camry \\
        --rift-history H.json --rift-epoch 24 --spinr-history H.json --spinr-epoch 5 \\
        --geometry camry_paper_provisional.json [--spinr-role-fit fit.json]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts_pvc.paper_rift_dataset_numbers_pvc import pct, sig4  # noqa: E402

FLOORS = Path('/scratch/group/p.cis261724.000/RIFT_pvc_runs/gotcha_unit_split_floor_20260923/floor_validation_F5split_v2.json')
GEOMETRY_KEYS = ('chamfer', 'hausdorff', 'hd95', 'iou', 'f1')
HIGHER = dict(train=False, val=False, chamfer=False, hausdorff=False, hd95=False, iou=True, f1=True)
PENDING, NA = r'\resultpending', r'\resultna'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--rift-history', type=Path, required=True)
    p.add_argument('--rift-epoch', type=int, required=True)
    p.add_argument('--spinr-history', type=Path, required=True)
    p.add_argument('--spinr-epoch', type=int, required=True)
    p.add_argument('--spinr-role-fit', type=Path)
    p.add_argument('--rift-role-fit', type=Path)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--floors', type=Path, default=FLOORS)
    p.add_argument('--threshold', type=float, default=0.20)
    p.add_argument('--tex', type=Path, help='manuscript file whose camry-table-rows marker block receives the rows')
    return p.parse_args(argv)


def splice(path, name, body):
    """Replace the text between the ``% BEGIN <name>`` line and the ``% END <name>`` line of ``path``."""
    text = path.read_text()
    begin, end = text.find(f'% BEGIN {name}'), text.find(f'% END {name}')
    if begin < 0 or end < begin or text.count(f'% BEGIN {name}') != 1:
        raise SystemExit(f'{path}: no single {name} marker block')
    start = text.index('\n', begin) + 1
    path.write_text(text[:start] + body.rstrip('\n') + '\n' + text[end:])


def epoch_entry(path, epoch):
    matches = [row for row in json.loads(path.read_text()) if row['epoch'] == epoch]
    if len(matches) != 1:
        raise SystemExit(f'{path}: {len(matches)} entries for epoch {epoch}')
    return matches[0]


def geometry_rows(path, threshold):
    """method label -> metrics at the fixed threshold; checkpoints recorded for provenance."""
    data = json.loads(path.read_text())
    rows = {}
    for row in data['rows']:
        if abs(row.get('threshold', -1) - threshold) > 1e-12:
            continue
        key = row['method'] if row['method'] in ('rift', 'spinr') else row['method'].split(':')[-1]
        if key in rows:
            raise SystemExit(f'{path}: two rows for {key} at t={threshold}')
        surface = row.get('surface', {})
        rows[key] = dict(chamfer=surface.get('cham'), hausdorff=surface.get('hausdorff_mm', 0) / 1000,
                         hd95=surface.get('hd95_mm', 0) / 1000, iou=row.get('iou_solid'),
                         f1=row.get('prf', {}).get('f1'), points=row['points'], checkpoint=row.get('checkpoint'))
    return data, rows


def main(argv=None):
    args = parse_args(argv)
    rift = epoch_entry(args.rift_history, args.rift_epoch)
    spinr = epoch_entry(args.spinr_history, args.spinr_epoch)
    def role_fit(path, epoch, name):
        fit = json.loads(path.read_text()) if path and path.exists() else None
        if fit is not None and fit['epoch'] != epoch:
            raise SystemExit(f'{name} role fit is epoch {fit["epoch"]}, not {epoch}')
        if fit is not None and 'train' not in fit['roles']:
            fit = None                   # a VAL-only check carries no TRAIN number
        return fit
    fit = role_fit(args.spinr_role_fit, args.spinr_epoch, 'SpINR')
    rift_fit = role_fit(args.rift_role_fit, args.rift_epoch, 'RIFT')
    rift_train = (rift['train']['full_native_rel_mse'] if 'train' in rift
                  else rift_fit['roles']['train']['full_native_rel_mse'] if rift_fit else None)
    geometry_json, geometry = geometry_rows(args.geometry, args.threshold)
    floors = json.loads(args.floors.read_text())['summary']
    values = {
        'rift': dict(train=rift_train, val=rift['validation']['full_native_complex_rel_mse'],
                     **{k: geometry['rift'][k] for k in GEOMETRY_KEYS}),
        'spinr': dict(train=fit['roles']['train']['full_native_rel_mse'] if fit else None,
                      val=spinr['validation']['full_native_complex_rel_mse'],
                      **{k: geometry['spinr'][k] for k in GEOMETRY_KEYS}),
        'backprojection': dict(train=NA, val=NA, **{k: geometry['backprojection_full'][k] for k in GEOMETRY_KEYS}),
    }
    # the trainer's own VAL fit and SpINR's role-fit VAL must agree with the history's selection metric
    if 'validation_fit' in rift:
        assert abs(rift['validation_fit']['full_native_rel_mse'] - values['rift']['val']) < 1e-9
    if rift_fit is not None:
        assert abs(rift_fit['roles']['validation']['full_native_rel_mse'] - values['rift']['val']) < 1e-6, 'RIFT VAL mismatch'
    if fit is not None:
        assert abs(fit['roles']['validation']['full_native_rel_mse'] - values['spinr']['val']) < 1e-6, 'SpINR VAL mismatch'
    references = [(r'\textit{Zero predictor}', 1.0),
                  (r'\textit{One-look nearest-pass fit}', floors['validation nearest_pass']['1']['pooled']['rel_mse']),
                  (r'\textit{One-look seven-pass fit}', floors['validation all_other_passes']['1']['pooled']['rel_mse'])]
    # The VAL references stay in the numbers record but are not typeset (user, 2026-09-24 14:30).
    blocks = [(r'\textit{Full split: 2,410 TRAIN and 30 validation units}',
               [('rift', r'RIFT {\scriptsize (Ours)}'), ('spinr', r"SpINR-style {\scriptsize (arXiv~'25)}"),
                ('backprojection', r"Backprojection {\scriptsize (IEEE TAES~'03)}")], [])]   # split A dropped (user 14:45)
    columns = ('train', 'val', *GEOMETRY_KEYS)
    lines = []
    for title, rows, refs in blocks:
        lines.append(rf'\multicolumn{{8}}{{@{{}}l}}{{{title}}} \\')
        best = {}
        for column in columns:          # bold only where two or more methods in the block have a number
            numbers = [values[key][column] for key, _ in rows if isinstance(values[key][column], float)]
            if len(numbers) >= 2:
                best[column] = max(numbers) if HIGHER[column] else min(numbers)
        for key, label in rows:
            cells = []
            for column in columns:
                value = values[key][column]
                if value is None:
                    cells.append(PENDING)
                elif isinstance(value, str):
                    cells.append(value)
                else:
                    text = pct(100 * value) if column in ('train', 'val') else sig4(value)
                    if best.get(column) == value:
                        text = rf'$\mathbf{{{text}}}$'.replace(r'\%}$', r'}$\%')
                    cells.append(rf'\filled{{{text}}}')
            lines.append(f'{label} & ' + ' & '.join(cells) + r' \\')
        for label, value in refs:
            lines.append(f'{label} & & \\filled{{{pct(100 * value)}}} & & & & & \\\\')
        lines.append(r'\cmidrule(l){1-8}' if (title, rows, refs) != blocks[-1] else '')
    stamp = datetime.datetime.now().astimezone().isoformat(timespec='minutes')
    header = (f'% Generated by scripts_pvc/paper_gotcha_camry_table_pvc.py at {stamp}: RIFT epoch {args.rift_epoch}, '
              f'SpINR epoch {args.spinr_epoch}, geometry {args.geometry.name} at t={args.threshold:.2f}\n')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    body = '\n'.join(line for line in lines if line)
    (args.out_dir/'camry_table_rows.tex').write_text(header + body + '\n')
    if args.tex:
        splice(args.tex, 'camry-table-rows', body)
    record = dict(schema='gotcha_camry_paper_table_v1', generated=stamp, threshold=args.threshold,
                  rift=dict(history=str(args.rift_history), epoch=args.rift_epoch, train_fit=rift.get('train'),
                            validation=rift['validation'], role_fit=str(args.rift_role_fit) if rift_fit else None),
                  spinr=dict(history=str(args.spinr_history), epoch=args.spinr_epoch,
                             role_fit=str(args.spinr_role_fit) if fit else None),
                  geometry=dict(file=str(args.geometry), tau_m=geometry_json['tau_m'],
                                iou_unit_m=geometry_json['iou_unit_m'], mesh=geometry_json['truth']['mesh_dir'],
                                ground_cut=geometry_json.get('ground_cut'), rows=geometry),
                  floors=dict(file=str(args.floors), references=references), values=values)
    (args.out_dir/'camry_table_numbers.json').write_text(json.dumps(record, indent=1, default=str) + '\n')
    print(body)


if __name__ == '__main__':
    main()
