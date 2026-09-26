#!/usr/bin/env python3
"""Manuscript assets for the RIFT-dataset view figures (PVC twin; CPU only, reads no radar response).

For every collection scene this copies the six-method postflight's standalone max-intensity panels
(``mip_panels/no_mesh``) into the manuscript tree, renders the scene's registered reference mesh from
the same three cameras with the same crop, orientation and panel size, and draws one colour bar whose
bar spans exactly the figure's panel block (``--colorbar-height``):

    <paper-dir>/<scene>/<scene>_<method>_mip_view{1,2,3}.png   method panels (copied unchanged)
    <paper-dir>/<scene>/<scene>_mesh_view{1,2,3}.png           reference mesh, depth-shaded grey
    <paper-dir>/colorbar_mip.pdf                                the panels' shared scale

Slot k is the scene's k-th camera in the postflight's ``SCENE_VIEWS`` (the same three looks across scenes whose
asset frames differ); ``manifest.json`` records each slot's camera.

The mesh is the RIFT geometry evaluator's registered mesh in the scene frame (the truth every metric is
scored against), sampled densely and drawn with the postflight's orthographic z-buffer and depth shading.
Panels are looked up in ``--eval-root/geometry/<scene>`` first, then in each ``--fallback`` root
(e.g. a five-method run made before one method finished); a panel found in neither is left out, and the
manuscript's ``\\IfFileExists`` guard shows it as pending.

    python scripts_pvc/export_rift_dataset_paper_figures_pvc.py \\
        --paper-dir manuscripts/iclr27/figures/rift_dataset --fallback geometry_nogeraf
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import scripts_pvc.run_rift_dataset_six_method_postflight_pvc as postflight  # noqa: E402

EVAL_ROOT = Path('/scratch/user/u.db364833/RIFT_runs/final_eval_20260923')
SCENES = ('b787', 'a320', 'x59', 'firetruck', 'race_car', 'loader')
METHOD_KEYS = tuple(key for key, _ in postflight.METHODS)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--eval-root', type=Path, default=EVAL_ROOT)
    p.add_argument('--fallback', action='append', default=[], help='extra geometry root under --eval-root')
    p.add_argument('--paper-dir', type=Path, required=True)
    p.add_argument('--scenes', nargs='+', default=list(SCENES))
    p.add_argument('--dataset-root', type=Path, default=postflight.DATASET_ROOT)
    p.add_argument('--mesh-samples', type=int, default=2_000_000)
    p.add_argument('--mesh-px', type=int, default=500)
    p.add_argument('--colorbar-height', type=float, default=4.615, help='inches: the figure panel block (8 panels + 7 gaps)')
    p.add_argument('--colorbar-width', type=float, default=0.46)
    p.add_argument('--colorbar-font', type=float, default=6.5)
    return p.parse_args(argv)


def protocol_of(eval_root, scene, fallbacks):
    for root in ['geometry', *fallbacks]:
        path = eval_root/root/scene/'geometry_metrics.json'
        if path.exists():
            return json.loads(path.read_text())
    return None


def copy_panels(args, scene, out):
    copied, missing = [], []
    for key in METHOD_KEYS:
        for slot, view in enumerate(postflight.views_for(scene), 1):
            name = f'{scene}_{key}_mip_{view[1]}.png'
            for root in ['geometry', *args.fallback]:
                source = args.eval_root/root/scene/'mip_panels'/'no_mesh'/name
                if source.exists():
                    shutil.copy2(source, out/f'{scene}_{key}_mip_view{slot}.png')
                    copied.append(f'{root}/{scene}/mip_panels/no_mesh/{name}')
                    break
            else:
                missing.append(name)
    return copied, missing


def render_mesh_views(args, scene, protocol, out):
    """Reference mesh from the postflight's cameras: same crop, right/up axes and panel size."""
    import matplotlib.pyplot as plt
    from scripts.eval_b787_geometry_metrics import sample_surface_points
    from scripts.render_b787_vs_stl import collection_geometry_inputs
    ns = SimpleNamespace(object=scene, dataset_root=args.dataset_root, npz_path=None, stl=None,
                         num_train=2400, num_tx=1, num_rx=1, tx_indices=None, rx_indices=None)
    _, verts, _ = collection_geometry_inputs(ns)
    tris = np.asarray(verts, dtype=np.float64).reshape(-1, 3, 3)
    points = sample_surface_points(tris, args.mesh_samples, np.random.default_rng(1))
    crop, px = float(protocol['protocol']['crop_m']), args.mesh_px
    pixel = 2 * crop / px
    written = []
    for slot, view in enumerate(postflight.views_for(scene), 1):
        depth = postflight.render_depth(points, view, crop, px, pixel)
        shading = postflight.shade(depth, pixel)

        def draw(axis):
            axis.set_facecolor('#050505')
            cmap = plt.get_cmap('Greys_r').copy()
            cmap.set_bad(alpha=0)
            axis.imshow(shading, origin='lower', extent=[-crop, crop, -crop, crop], cmap=cmap, vmin=-.1, vmax=1.15,
                        interpolation='nearest')
            axis.set_xlim(-crop, crop)
            axis.set_ylim(-crop, crop)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color('#252525')

        stem = out/f'{scene}_mesh_view{slot}'
        written += [str(p) for p in postflight.save_panel(draw, stem) if p.suffix == '.png']
        stem.with_suffix('.pdf').unlink(missing_ok=True)
    return dict(triangles=int(len(tris)), samples=int(len(points)), crop_m=crop, px=px, files=written)


def render_colorbar(args, threshold, path):
    """The panels' scale (template ``_draw_field``: inferno over [t, 1], black below t) at print size."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from scripts_pvc.paper_figure_font import use_paper_font
    use_paper_font()  # the manuscript's Times (TeX Gyre Termes), user 2026-09-24
    width, height, bar = args.colorbar_width, args.colorbar_height, 0.09
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
    scale.set_label(f'normalized magnitude (per method, min–max); black: below $t={threshold:.2f}$',
                    fontsize=args.colorbar_font, labelpad=1.5)
    figure.savefig(path, facecolor='white')
    plt.close(figure)


def main(argv=None):
    import scripts.run_b7873200_six_method_plenoxel_postflight_v1 as template
    template.import_runtime_dependencies()
    args = parse_args(argv)
    args.paper_dir.mkdir(parents=True, exist_ok=True)
    manifest, thresholds = dict(eval_root=str(args.eval_root), fallbacks=args.fallback, scenes={}), set()
    for scene in args.scenes:
        protocol = protocol_of(args.eval_root, scene, args.fallback)
        if protocol is None:
            print(f'{scene}: no postflight output yet, skipped', flush=True)
            continue
        thresholds.add(float(protocol['protocol']['threshold']))
        out = args.paper_dir/scene
        out.mkdir(exist_ok=True)
        copied, missing = copy_panels(args, scene, out)
        mesh = render_mesh_views(args, scene, protocol, out)
        cameras = {f'view{slot}': dict(name=v[0], tag=v[1], forward_axis=v[2], camera_side=v[3], right=v[4], up=v[5])
                   for slot, v in enumerate(postflight.views_for(scene), 1)}
        manifest['scenes'][scene] = dict(cameras=cameras, panels_copied=copied, panels_missing=missing, mesh=mesh,
                                         figure_scale=protocol.get('figure_scale'))
        print(f'{scene}: {len(copied)} panels, {len(missing)} missing, mesh views {len(mesh["files"])}', flush=True)
    if len(thresholds) != 1:
        raise ValueError(f'scenes disagree on the fixed threshold: {sorted(thresholds)}')
    render_colorbar(args, thresholds.pop(), args.paper_dir/'colorbar_mip.pdf')
    (args.paper_dir/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(f'EXPORT=PASS {args.paper_dir}')


if __name__ == '__main__':
    main()
