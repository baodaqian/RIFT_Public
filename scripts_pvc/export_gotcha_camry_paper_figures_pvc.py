#!/usr/bin/env python3
"""Manuscript assets for the provisional GOTCHA Camry view figures (PVC twin; CPU only, reads no radar response).

This copies the standalone max-intensity panels of a Camry "best so far" set made by
``scripts_pvc/render_gotcha_camry_mip_panels_pvc.py`` into the manuscript tree under slot names, and draws
one colour bar per view figure whose bar spans exactly that figure's method block (``VIEW_PANEL_HEIGHTS``):

    <paper-dir>/camry_<row>_mip_view{1,2,3}.png   method panels and the reference-surface density (copied unchanged)
    <paper-dir>/camry_mesh_view{1,2,3}.png        registered reference mesh (copied unchanged)
    <paper-dir>/colorbar_mip_view{1,2,3}.pdf      the panels' shared scale, one bar per view figure
                                                  (``--colorbars-only`` redraws just these)
    --tex FILE (camry-run-states marker block)    run states typeset in the row labels and captions
    <paper-dir>/manifest.json                     source files, cameras and run states

Slots follow the RIFT-dataset figures: view1 = front, view2 = side, view3 = plan (the renderer's "top").
Rows are matched by the source's leading index and method name, so a refreshed set whose file names carry
new epochs exports without edits; a row found zero or several times stops the export. ``--row KEY=STEM`` takes a
row from another render instead (files ``STEM_<front|side|top>.png``, e.g. a checkpoint re-rendered by
``render_gotcha_camry_mip_panels_pvc.py``), ``--omit KEY`` removes a row (the manuscript then typesets it as TBD),
and ``--state KEY=TEXT`` sets the run state typeset for a learned row (default: parsed from the source file name).

    python scripts_pvc/export_gotcha_camry_paper_figures_pvc.py --paper-dir manuscripts/iclr27/figures/gotcha_camry
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import shutil
from pathlib import Path

SOURCE = Path('/scratch/group/p.cis261724.000/RIFT_pvc_runs/camry_mip_provisional_20260924_1137/best_so_far')
VIEWS = (('view1', 'front'), ('view2', 'side'), ('view3', 'top'))
# paper row key -> source file pattern (before "_<view>.png")
ROWS = (
    ('mesh', '0_ground_truth_mesh'),
    ('surface', '0_reference_surface_mip'),
    ('rift', '1_RIFT_*_mip'),
    ('spinr', '2_SpINR_*_mip'),
    ('backprojection', '4_Backprojection_fulldata*_mip'),
)
# the method block of the figure: every row except the two ground-truth rows (SE and the 578-unit backprojection
# dropped from GOTCHA, user 2026-09-24 14:35 / 14:45)
METHOD_ROWS = 3
# panel height (in) of each view figure: the third argument of \gotchaviewmatrix in sec/12 (three vehicle columns
# fit 5.5 in); each figure gets its own bar, drawn at its block height so the text prints at --colorbar-font
VIEW_PANEL_HEIGHTS = (('view1', 0.80), ('view2', 0.54), ('view3', 0.65))
# longest first; all three bars take the first that fits the shortest block
COLORBAR_LABELS = ('normalized magnitude (per method, min–max); black: below $t={t:.2f}$',
                   'min–max normalized magnitude; black below $t={t:.2f}$',
                   'min–max normalized magnitude')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--paper-dir', type=Path, required=True)
    p.add_argument('--gap', type=float, default=0.025, help='inches: \\camrygap in the manuscript')
    p.add_argument('--colorbars-only', action='store_true',
                   help='redraw the per-view colour bars only (no panel copy, no run states)')
    p.add_argument('--colorbar-width', type=float, default=0.46)
    p.add_argument('--colorbar-font', type=float, default=6.5)
    p.add_argument('--rift-epochs', type=int, default=40, help='declared epochs of the RIFT run shown')
    p.add_argument('--spinr-epochs', type=int, default=150, help='declared epochs of the SpINR run shown')
    p.add_argument('--row', action='append', default=[], help='KEY=STEM: take row KEY from STEM_<view>.png')
    p.add_argument('--omit', action='append', default=[], help='KEY: leave row KEY out (typeset as TBD)')
    p.add_argument('--state', action='append', default=[], help='KEY=TEXT: run state typeset for rift/spinr')
    p.add_argument('--tex', type=Path, help='manuscript file whose camry-run-states marker block receives the states')
    return p.parse_args(argv)


def find(source, pattern, view):
    matches = sorted(source.glob(f'{pattern}_{view}.png'))
    if len(matches) != 1:
        raise SystemExit(f'{pattern}_{view}.png: {len(matches)} matches in {source}')
    return matches[0]


def colorbar_label(args, threshold, height, margin=0.1):
    """The longest of COLORBAR_LABELS whose printed length fits a bar of ``height`` inches."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts_pvc.paper_figure_font import use_paper_font
    use_paper_font()
    figure = plt.figure()
    renderer = figure.canvas.get_renderer()
    try:
        for label in COLORBAR_LABELS:
            text = figure.text(0, 0, label.format(t=threshold), fontsize=args.colorbar_font)
            if text.get_window_extent(renderer).width / figure.dpi <= height - margin:
                return label.format(t=threshold)
            text.remove()
    finally:
        plt.close(figure)
    raise SystemExit(f'no colour-bar label fits {height:.2f} in')


def render_colorbar(args, threshold, height, path, label):
    """The panels' scale (inferno over [t, 1], black below t) at print size, as the RIFT-dataset exporter draws it."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts_pvc.paper_figure_font import use_paper_font
    use_paper_font()  # the manuscript's Times (TeX Gyre Termes), user 2026-09-24
    width, bar = args.colorbar_width, 0.09
    figure = plt.figure(figsize=(width, height))
    scale = figure.colorbar(ScalarMappable(Normalize(threshold, 1.0), plt.get_cmap('inferno')),
                            cax=figure.add_axes([0, 0, bar / width, 1]))
    ticks = [threshold] + [v for v in (.4, .6, .8) if v > threshold + .05] + [1.0]
    scale.set_ticks(ticks, labels=[f'{v:.1f}' for v in ticks])
    scale.ax.tick_params(labelsize=args.colorbar_font, length=2, width=.5, pad=1.2)
    labels = scale.ax.get_yticklabels()
    labels[0].set_verticalalignment('bottom')
    labels[-1].set_verticalalignment('top')
    scale.outline.set_linewidth(.5)
    scale.set_label(label, fontsize=args.colorbar_font, labelpad=1.5)
    figure.canvas.draw()
    if scale.ax.yaxis.label.get_window_extent().height > figure.bbox.height:
        raise SystemExit(f'{path.name}: colour-bar label overruns the {height:.2f} in bar')
    figure.savefig(path, facecolor='white')
    plt.close(figure)


def render_view_colorbars(args, threshold, out):
    """One bar per view figure, each spanning that figure's method block; the superseded single bar is removed."""
    blocks = {slot: round(METHOD_ROWS*height + (METHOD_ROWS - 1)*args.gap, 4) for slot, height in VIEW_PANEL_HEIGHTS}
    label = colorbar_label(args, threshold, min(blocks.values()))
    for slot, block in blocks.items():
        render_colorbar(args, threshold, block, out/f'colorbar_mip_{slot}.pdf', label)
    (out/'colorbar_mip.pdf').unlink(missing_ok=True)
    return blocks, label


def run_states(sources, args):
    """Run states typeset in the labels: --state overrides, else parsed from the best-so-far names
    (e.g. 1_RIFT_F5gA_ext40_ep22_mip); an omitted row has none."""
    given = dict(item.split('=', 1) for item in args.state)
    patterns = dict(rift=(r'_ep(\d+)_', lambda e: f'epoch {e} of {args.rift_epochs}'),
                    spinr=(r'_ep(\d+)_', lambda e: f'epoch {e} of {args.spinr_epochs}'))
    states = {}
    for key, (regex, text) in patterns.items():
        if key in given:
            states[key] = given[key]
        elif key in sources:
            match = re.search(regex, sources[key].name)
            if match is None:
                raise SystemExit(f'{sources[key].name}: no {regex!r}; pass --state {key}=TEXT')
            states[key] = text(*match.groups())
    return states


def main(argv=None):
    args = parse_args(argv)
    out = args.paper_dir
    if args.colorbars_only:
        manifest = json.loads((out/'manifest.json').read_text())
        blocks, label = render_view_colorbars(args, float(manifest['threshold']), out)
        manifest.pop('colorbar_height_in', None)
        manifest.pop('panel_height_in', None)
        manifest.update(colorbar_heights_in=blocks, panel_heights_in=dict(VIEW_PANEL_HEIGHTS), colorbar_label=label)
        (out/'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
        print(f'COLORBARS=PASS {blocks} label {label!r}')
        return
    source_manifest = json.loads((args.source.parent/'manifest.json').read_text())
    out.mkdir(parents=True, exist_ok=True)
    overrides = dict(item.split('=', 1) for item in args.row)
    unknown = (set(overrides) | set(args.omit)) - {key for key, _ in ROWS}
    if unknown:
        raise SystemExit(f'unknown rows {sorted(unknown)}')
    copied, first_view = [], {}
    for key, pattern in ROWS:
        for slot, view in VIEWS:
            dst = out/(f'camry_mesh_{slot}.png' if key == 'mesh' else f'camry_{key}_mip_{slot}.png')
            if key in args.omit:
                dst.unlink(missing_ok=True)
                continue
            src = Path(f'{overrides[key]}_{view}.png') if key in overrides else find(args.source, pattern, view)
            if not src.is_file():
                raise SystemExit(f'{src}: missing')
            first_view.setdefault(key, src)
            shutil.copy2(src, dst)
            copied.append(dict(row=key, slot=slot, view=view, source=str(src), file=dst.name))
    threshold = float(source_manifest['threshold'])
    blocks, label = render_view_colorbars(args, threshold, out)
    states = run_states(first_view, args)
    stamp = datetime.datetime.now().astimezone().isoformat(timespec='minutes')
    macros = dict(rift='camryriftstate', spinr='camryspinrstate')
    (out/'camry_provenance.tex').unlink(missing_ok=True)          # superseded: the states are written in place
    if args.tex:
        from scripts_pvc.paper_gotcha_camry_table_pvc import splice
        splice(args.tex, 'camry-run-states', ''.join(
            f'\\newcommand{{\\{macro}}}{{{states.get(key, chr(92) + "resultpending")}}}\n' for key, macro in macros.items()))
    manifest = dict(schema='gotcha_camry_paper_figures_v1', exported=stamp, source=str(args.source),
                    threshold=threshold, colorbar_heights_in=blocks, panel_heights_in=dict(VIEW_PANEL_HEIGHTS),
                    colorbar_label=label,
                    rows_overridden=overrides, rows_omitted=args.omit, states=states, views=source_manifest['views'], window_half_m=source_manifest['window_half_m'],
                    lattice=source_manifest['lattice'], mesh_dir=source_manifest['mesh_dir'],
                    scale=source_manifest['scale'], files=copied)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(f'EXPORT=PASS {len(copied)} panels, colour bars {blocks} in, states {states}')


if __name__ == '__main__':
    main()
