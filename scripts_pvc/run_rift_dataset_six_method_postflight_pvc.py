#!/usr/bin/env python3
"""Six-method renders and fixed-threshold 3D metrics on one RIFT-dataset scene (PVC twin).

Twin of ``scripts/run_b7873200_six_method_plenoxel_postflight_v1.py`` (unchanged on disk) for the
train2400 production checkpoints of every collection scene. Its readout functions are used
unchanged (``normalized_dense_magnitude``, ``fixed_voxel_metrics``, ``_draw_field``); only the
loaders, the truth frame and the view layout are new:

    each method's native representation -> one declared 48^3 support scalar on the evaluator lattice
    -> 4x Plenoxel-style trilinear (align_corners=True) to 192^3 -> magnitude -> per-method min-max
    -> fixed t = 0.20.

Loaders (energy inputs are square-rooted after interpolation; native supports are not):

* RIFT: adaptive point-SH energy by conservative CIC on the 48^3 lattice (the RIFT evaluator's readout);
* SpINR: its signed field sigma on its own 48^3 midpoint grid, energy sigma^2;
* GeRaF (source_v1): its NeuS logistic SDF density on the lattice (the selected model's SDF and its own
  inverse sharpness; reflectivity unused), as the template's GeRaF readout;
* Radar Fields: its grid readout (the PVC geometry evaluator's loader);
* RadarSplat: its ``geometry_g48.npz`` occupancy-union support, energy support^2;
* Sugavanam-Ertin: its Stage-2 zero surface (``surface.npz``; SE has no native density), area-weighted
  surface samples deposited by CIC onto the lattice (surface density);
* Sugavanam-Ertin, Stage 1 only (``--se-stage1``; reported alongside the two-stage method): the Stage-1 aggregate
  sum_m |S_m| on its own cell-centred grid, squared to energy and trilinearly resampled onto the lattice.
* Backprojection (``--backprojection``): the train-only coherent matched-filter backprojection
  (``scripts_pvc/eval_rift_dataset_model_free_pvc.py --method mfbp``: ``matched_filter.npz``), computed directly on
  the evaluator lattice; energy |A|^2.

Metrics (the RIFT geometry evaluator's functions, fixed threshold 0.20): Chamfer against surface and
volume truth, IoU (solid and shell), precision/recall/F1 at tau = one 48^3 pitch (6.25 mm), on the
48^3 lattice (``upsample 1``, the RIFT evaluator's default, primary) and on the 192^3 densified field
the renders show. Renders: max-intensity projections drawn by the template's ``_draw_field`` from three
axis-aligned orthographic cameras per scene (``SCENE_VIEWS``: -x, -y, -z for A320 and X-59; +z, +x rotated
90 deg counter-clockwise, +y for the scenes whose asset frame is oriented differently, giving the same three
looks), one standalone
square panel per method and view (no composite figure), with and without the registered-mesh overlay. Every
panel uses one colour scale (inferno over [t, 1] of the per-method min-max magnitude; black below t), so one
colour bar per scene, drawn at exactly the panel height, serves all of them. ``--render-3d`` adds standalone
depth-shaded 3D views of the thresholded 192^3 support (reconstruction only).

    python scripts_pvc/run_rift_dataset_six_method_postflight_pvc.py --object b787 --scene b787 \\
        --output-dir OUT/b787 --rift CK --spinr CK --geraf CK --radar-fields CK \\
        --radarsplat GEOMETRY_G48_NPZ --se-surface SURFACE_NPZ --se-checkpoint SE_CK
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

SCHEMA = 'rift_dataset_six_method_postflight_pvc_v1'
DATASET_ROOT = Path('/scratch/user/u.db364833/RIFT_runs/h100_smoke_20260920_6d58645/inputs/RIFT_dataset')
# Cameras: name, file tag, forward axis, camera side (-1: on the negative end, looking toward +),
# (right axis, sign), (up axis, sign). Every view is a proper (unmirrored) camera image.
NEG_AXIS_VIEWS = (
    ('-x', 'negx', 0, -1, (1, -1), (2, 1)),
    ('-y', 'negy', 1, -1, (0, 1), (2, 1)),
    ('-z', 'negz', 2, -1, (0, -1), (1, 1)),
)
# The same three looks for scenes whose asset frame is oriented differently from A320/X-59 (user, 2026-09-23):
# +z for -x, +x rotated 90 deg counter-clockwise for -y, +y for -z. The +y image axes (right +z = the nose,
# up +x) are chosen so the object lies nose-right as in the A320/X-59 -z view.
POS_AXIS_VIEWS = (
    ('+z', 'posz', 2, 1, (0, 1), (1, 1)),
    ('+x, 90° CCW', 'posx_ccw90', 0, 1, (2, -1), (1, 1)),
    ('+y', 'posy', 1, 1, (2, 1), (0, 1)),
)
SCENE_VIEWS = dict(a320=NEG_AXIS_VIEWS, x59=NEG_AXIS_VIEWS, b787=POS_AXIS_VIEWS, firetruck=POS_AXIS_VIEWS,
                   race_car=POS_AXIS_VIEWS, loader=POS_AXIS_VIEWS)


def views_for(obj):
    return SCENE_VIEWS.get(obj, NEG_AXIS_VIEWS)


PANEL_IN, PANEL_DPI = 2.5, 200   # every standalone panel and the colour bar are PANEL_IN tall (500 px)
METHODS = (('rift', 'RIFT'), ('spinr', 'SpINR'), ('geraf', 'GeRaF'), ('radar_fields', 'Radar Fields'),
           ('radarsplat', 'RadarSplat'), ('se', 'Sugavanam–Ertin'), ('se_stage1', 'Sugavanam–Ertin Stage 1'),
           ('backprojection', 'Backprojection'))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--object', required=True)
    p.add_argument('--scene', required=True, help='scene directory name, e.g. airliner_a320')
    p.add_argument('--dataset-root', type=Path, default=DATASET_ROOT)
    p.add_argument('--role-manifest', type=Path, required=True, help="the production run's role_manifest.json")
    p.add_argument('--output-dir', type=Path, required=True)
    for key, _ in METHODS:
        if key not in ('se', 'se_stage1'):
            p.add_argument(f"--{key.replace('_', '-')}")
    p.add_argument('--se-surface')
    p.add_argument('--se-checkpoint')
    p.add_argument('--se-stage1', help='SE Stage-1 selection (stage1_selected.pt): the Stage-1-only variant')
    p.add_argument('--extent', type=float, default=0.15)
    p.add_argument('--grid', type=int, default=48)
    p.add_argument('--upsample', type=int, default=4)
    p.add_argument('--fixed-threshold', type=float, default=0.20)
    p.add_argument('--crop', type=float, default=0.075)
    p.add_argument('--f1-tau', type=float, default=0.00625)
    p.add_argument('--iou-unit', type=float, default=0.005)
    p.add_argument('--n-surface', type=int, default=20_000)
    p.add_argument('--n-volume', type=int, default=50_000)
    p.add_argument('--gt-grid', type=int, default=240)
    p.add_argument('--se-samples', type=int, default=400_000)
    p.add_argument('--image-px', type=int, default=420)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--field-batch', type=int, default=16384)
    p.add_argument('--render-3d', action='store_true', help='also write standalone depth-shaded 3D view panels')
    args = p.parse_args(argv)
    if (args.se_surface is None) != (args.se_checkpoint is None):
        p.error('--se-surface and --se-checkpoint go together')
    return args


# ---------------------------------------------------------------- truth
def load_truth(args):
    """Registered mesh in the scene frame (RIFT geometry evaluator), surface and solid samples as the template."""
    from scripts.eval_b787_geometry_metrics import inside_mask, sample_surface_points, sample_volume_points
    from scripts.render_b787_vs_stl import collection_geometry_inputs
    ns = SimpleNamespace(object=args.object, dataset_root=args.dataset_root, npz_path=None, stl=None,
                         num_train=2400, num_tx=1, num_rx=1, tx_indices=None, rx_indices=None)
    metadata, verts, contract = collection_geometry_inputs(ns)
    tris = np.asarray(verts, dtype=np.float64).reshape(-1, 3, 3)
    rng = np.random.default_rng(args.seed)
    surface = sample_surface_points(tris, args.n_surface, rng)
    edges = np.linspace(-args.extent, args.extent, args.gt_grid + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    occupancy = inside_mask(tris, centers, centers, centers)
    volume = sample_volume_points(occupancy, centers, centers, centers, args.n_volume, rng)
    truth = dict(triangle_count=int(len(tris)), surface_sample_count=int(len(surface)),
                 volume_sample_count=int(len(volume)), solid_gt_grid=int(args.gt_grid),
                 solid_fill_fraction=float(occupancy.mean()))
    return tris, surface, volume, truth, contract


# ---------------------------------------------------------------- loaders -> (base 48^3 scalar, interpolation input, details)
def lattice(args):
    axis = -args.extent + (np.arange(args.grid, dtype=np.float64) + .5) * (2 * args.extent / args.grid)
    return np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)


def via_pvc_evaluator(path, args, contract, label):
    from scripts_pvc.eval_b787_geometry_metrics_pvc import load_any_field
    info = {}
    energy, grid, step = load_any_field(path, expected_contract=contract, extent=args.extent,
                                        point_grid=args.grid, readout_info=info)
    energy = np.asarray(energy, dtype=np.float64)
    if grid != args.grid or energy.shape != (args.grid,) * 3:
        raise ValueError(f'{label}: readout grid {grid} is not the common {args.grid}^3 lattice')
    return energy, 'energy', dict(info, checkpoint_epoch_or_step=step)


def load_geraf(path, args, contract):
    from rift.geraf import logistic_sdf_pdf
    from rift.geraf_source_data import RIFTSourceData
    from rift.rift_dataset import resolve_object_inputs
    from rift_pvc.geraf_source_training import load_selected_models
    npz_path, _ = resolve_object_inputs(object_name=args.object, dataset_root=args.dataset_root,
                                        role_manifest_path=args.role_manifest)
    data = RIFTSourceData(str(npz_path), str(args.role_manifest))
    ck = torch.load(path, map_location='cpu', weights_only=False)
    models, selected = load_selected_models(ck, data, torch.device('cpu'))
    model = models['scalar']
    if not np.isclose(float(data.extent), args.extent, rtol=0, atol=1e-12):
        raise ValueError('GeRaF scene extent disagrees with the evaluator extent')
    points = torch.as_tensor(lattice(args) / args.extent, dtype=next(model.sdf_network.parameters()).dtype)
    values = []
    with torch.no_grad():
        # SingleVarianceNetwork returns one learned value for every point.
        inv_s = model._clip_inverse_s(model.deviation_network(points[:1]))[0, 0]
        for block in points.split(args.field_batch):
            values.append(logistic_sdf_pdf(model.sdf_network(block)[..., 0], inv_s).double().numpy())
    support = np.concatenate(values).reshape((args.grid,) * 3)
    if not np.isfinite(support).all() or (support < 0).any():
        raise ValueError('GeRaF SDF support readout is invalid')
    return support, 'native_support', dict(kind='logistic_sdf_pdf of the selected source_v1 SDF (normalized units, own inv_s)',
                                            reflectivity_used=False, inv_s=float(inv_s), checkpoint_step=int(ck['step']),
                                            selected_step=int(selected['step']))


def load_se(surface_path, checkpoint_path, args, contract):
    from rift.rift_dataset import resolve_object_inputs
    from rift.sugavanam_ertin_acquisition import CollectionAcquisition, digest
    from scripts.eval_b787_geometry_metrics import sample_surface_points
    from scripts.eval_scene_geometry import deposit_points
    inputs = resolve_object_inputs(object_name=args.object, dataset_root=args.dataset_root,
                                   role_manifest_path=args.role_manifest)
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if digest(ck['acquisition']) != digest(CollectionAcquisition(npz_path=inputs[0], manifest=inputs[1]).identity):
        raise ValueError('SE checkpoint belongs to another acquisition/object/split')
    with np.load(surface_path, allow_pickle=False) as saved:
        vertices, faces = saved['vertices'].astype(np.float64), saved['faces'].astype(np.int64)
    if not len(faces):
        raise ValueError('SE surface has no faces')
    samples = sample_surface_points(vertices[faces], args.se_samples, np.random.default_rng(42))
    support = deposit_points(torch.as_tensor(samples), torch.ones(len(samples), dtype=torch.float64),
                             args.extent, args.grid).numpy()
    return support, 'native_support', dict(kind='Stage-2 zero surface, area-weighted samples CIC-deposited (surface density)',
                                            samples=int(args.se_samples), phase=ck.get('phase'), sdf_step=ck.get('sdf_step'),
                                            surface=str(Path(surface_path).resolve()))


def load_se_stage1(checkpoint_path, args):
    """SE with Stage 1 only: its Stage-1 scattering field as the support (the two-stage method uses its Stage-2 surface).

    The method's own Stage-1 aggregation (Eqs. 5/7, ``aggregate_scattering``: sum over sub-apertures of |S_m|, the
    magnitude its Stage-2 cloud is thresholded on) on its cell-centred granularity^3 grid over the same cube,
    squared to energy and trilinearly resampled onto the evaluator lattice."""
    from scipy.ndimage import map_coordinates
    from rift.rift_dataset import resolve_object_inputs
    from rift.sugavanam_ertin_acquisition import CollectionAcquisition, digest
    from rift.sugavanam_ertin_paper import aggregate_scattering
    inputs = resolve_object_inputs(object_name=args.object, dataset_root=args.dataset_root,
                                   role_manifest_path=args.role_manifest)
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if digest(ck['acquisition']) != digest(CollectionAcquisition(npz_path=inputs[0], manifest=inputs[1]).identity):
        raise ValueError('SE Stage-1 selection belongs to another acquisition/object/split')
    fields, n = ck['selected_fields'], int(ck['recipe']['granularity'])
    if fields.shape[1] != n ** 3:
        raise ValueError(f'SE Stage-1 fields are not on a {n}^3 grid')
    magnitude, _, _ = aggregate_scattering(fields, ck['partition']['directions'])
    energy = np.square(magnitude.numpy().reshape(n, n, n))              # grid_points: indexing='ij', x-major
    centres = lattice(args).reshape(args.grid, args.grid, args.grid, 3)[:, 0, 0, 0]
    index = (centres + args.extent) / (2 * args.extent / n) - .5          # evaluator centres in SE cell units
    ii, jj, kk = np.meshgrid(index, index, index, indexing='ij')
    resampled = map_coordinates(energy, [ii, jj, kk], order=1, mode='nearest')
    return np.maximum(resampled, 0.), 'energy', dict(
        kind='SE Stage 1 only: sum_m |S_m| (aggregate_scattering), squared, trilinear to the evaluator lattice',
        subapertures=int(fields.shape[0]), stage1_grid=n, stage1_selection=ck['stage1_source'].get('selection'),
        stage1_validation_complex_rel_mse=ck['stage1_source'].get('validation', {}).get('global_complex_rel_mse'))


def load_backprojection(path, args):
    """Matched-filter backprojection: the coherent adjoint sum over the object's training views
    (``scripts/eval_rift_dataset_model_free.py --method mfbp``), already on the evaluator lattice; energy |A|^2."""
    from rift.rift_dataset import collection_contract, load_object_contract, resolve_object_inputs
    inputs = resolve_object_inputs(object_name=args.object, dataset_root=args.dataset_root,
                                   role_manifest_path=args.role_manifest)
    _, contract = load_object_contract(*inputs, response_roles=('train',))
    expected = json.loads(json.dumps(collection_contract(contract), sort_keys=True))
    with np.load(path, allow_pickle=False) as saved:
        if json.loads(str(saved['identity'].item())) != expected:
            raise ValueError('Backprojection belongs to another acquisition/object/split')
        if not np.array_equal(saved['train_indices'], np.asarray(expected['role_ids']['train'])):
            raise ValueError('Backprojection did not sum exactly the training role')
        if not np.isclose(float(saved['extent']), args.extent, rtol=0, atol=1e-12):
            raise ValueError('Backprojection extent disagrees with the evaluator extent')
        centres = lattice(args).reshape(args.grid, args.grid, args.grid, 3)[:, 0, 0, 0]
        if saved['grid_centers'].shape != centres.shape or not np.allclose(saved['grid_centers'], centres, rtol=0, atol=1e-12):
            raise ValueError(f'Backprojection is not on the common {args.grid}^3 lattice')
        adjoint = saved['complex_adjoint']
        phase_sign = float(saved['phase_sign'])
    if adjoint.shape != (args.grid,) * 3 or not np.isfinite(adjoint).all():
        raise ValueError('Backprojection field is invalid')
    return np.square(np.abs(adjoint)).astype(np.float64), 'energy', dict(
        kind='train-only coherent matched-filter backprojection |A|^2 (phase-only, range_nufft), on the evaluator lattice',
        train_views=int(len(expected['role_ids']['train'])), phase_sign=phase_sign)


# ---------------------------------------------------------------- 3D view renderer
def render_depth(points, view, crop, px, radius_m):
    """Orthographic z-buffer of points seen from the camera's end of ``forward``, each splatted as a disk of
    ``radius_m``; returns depth (inf = empty). A centre-pixel z-buffer followed by a disk minimum filter is
    exactly the disk-splat z-buffer."""
    from scipy.ndimage import minimum_filter
    _, _, forward, side, (ra, rs), (ua, us) = view
    pixel = 2 * crop / px
    u = np.floor((rs * points[:, ra] + crop) / pixel).astype(np.int64)
    v = np.floor((us * points[:, ua] + crop) / pixel).astype(np.int64)
    ok = (u >= 0) & (u < px) & (v >= 0) & (v < px)
    zbuf = np.full(px * px, np.inf)
    np.minimum.at(zbuf, v[ok] * px + u[ok], -side * points[ok, forward])
    zbuf = zbuf.reshape(px, px)
    r = max(0, int(np.ceil(radius_m / pixel)))
    if r:
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        zbuf = minimum_filter(zbuf, footprint=(xx * xx + yy * yy) <= (r + .5) ** 2, mode='constant', cval=np.inf)
    return zbuf


def shade(zbuf, pixel, smooth_px=1.5):
    """Lambert shading from the smoothed depth map's normals (light upper-left, camera side) times a depth cue."""
    from scipy.ndimage import gaussian_filter
    filled = np.isfinite(zbuf)
    if not filled.any():
        return np.full(zbuf.shape, np.nan)
    far = zbuf[filled].max()
    z = np.where(filled, zbuf, far)
    weight = gaussian_filter(filled.astype(float), smooth_px)
    zs = gaussian_filter(np.where(filled, z, 0.), smooth_px) / np.maximum(weight, 1e-9)
    gy, gx = np.gradient(zs, pixel)
    normal = np.stack([gx, gy, -np.ones_like(zs)], -1)          # toward the camera
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    light = np.array([-.45, .55, -1.])
    light /= np.linalg.norm(light)
    lambert = np.clip(normal @ light, 0, 1)
    near = zbuf[filled].min()
    cue = 1 - .45 * (z - near) / max(far - near, 1e-12)
    return np.where(filled, (.18 + .82 * lambert) * cue, np.nan)


def draw_3d(axis, points, view, args, pixel_radius_m):
    """Reconstruction only: the mesh is rendered separately at the same panel size and set beside it."""
    import matplotlib.pyplot as plt
    px, crop = args.image_px, args.crop
    extent = [-crop, crop, -crop, crop]
    axis.set_facecolor('#050505')
    if len(points):
        recon = render_depth(points, view, crop, px, pixel_radius_m)
        cmap = plt.get_cmap('inferno').copy()
        cmap.set_bad(alpha=0)
        axis.imshow(shade(recon, 2 * crop / px), origin='lower', extent=extent, cmap=cmap, vmin=-.15, vmax=1.1,
                    interpolation='nearest')
    axis.set_xlim(-crop, crop)
    axis.set_ylim(-crop, crop)
    axis.set_xticks([])
    axis.set_yticks([])


def draw_mip(axis, postflight, normalized, verts, view, args):
    """The template's ``_draw_field`` along ``forward``; axes mirrored to the camera's right/up."""
    _, _, forward, _, (ra, rs), (ua, us) = view
    postflight._draw_field(axis, normalized, verts, plane=('', forward, ra, ua, '', ''), args=args)
    if rs < 0:
        axis.invert_xaxis()
    if us < 0:
        axis.invert_yaxis()


def save_panel(draw, stem):
    """One square PANEL_IN panel whose axes fill the figure, so every panel has the same pixel size."""
    import matplotlib.pyplot as plt
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(PANEL_IN, PANEL_IN))
    draw(figure.add_axes([0, 0, 1, 1]))
    paths = [stem.with_suffix(f'.{suffix}') for suffix in ('png', 'pdf')]
    for path in paths:
        figure.savefig(path, dpi=PANEL_DPI, facecolor='#050505')
    plt.close(figure)
    return paths


def save_colorbar(stem, args):
    """The panels' colour scale (``_draw_field``: inferno, vmin t, vmax 1), bar spanning exactly PANEL_IN."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    stem.parent.mkdir(parents=True, exist_ok=True)
    width, bar = 0.72, 0.18
    figure = plt.figure(figsize=(width, PANEL_IN))
    scale = figure.colorbar(ScalarMappable(Normalize(args.fixed_threshold, 1.0), plt.get_cmap('inferno')),
                            cax=figure.add_axes([0, 0, bar / width, 1]))
    ticks = [args.fixed_threshold] + [v for v in (.4, .6, .8) if v > args.fixed_threshold + .05] + [1.0]
    scale.set_ticks(ticks, labels=[f'{v:.1f}' for v in ticks])
    scale.ax.tick_params(labelsize=7, length=2.5, width=.6, pad=1.5)
    labels = scale.ax.get_yticklabels()     # keep the end labels inside the panel height
    labels[0].set_verticalalignment('bottom')
    labels[-1].set_verticalalignment('top')
    scale.outline.set_linewidth(.6)
    scale.set_label('min–max normalized magnitude', fontsize=7.5, labelpad=2)
    paths = [stem.with_suffix(f'.{suffix}') for suffix in ('png', 'pdf')]
    for path in paths:
        figure.savefig(path, dpi=PANEL_DPI, facecolor='white')
    plt.close(figure)
    return paths


def render_panels(rows, verts, postflight, args):
    """Standalone MIP panels per method and view, with and without the mesh overlay, plus the scene's colour
    bar; optionally standalone 3D panels. No composite figure."""
    root = args.output_dir/'mip_panels'
    overlays = {'with_mesh': verts, 'no_mesh': verts[:0]}
    voxel_m = 2 * args.extent / (args.grid * args.upsample)
    written = []
    for key, _title, dense, points in rows:
        for view in views_for(args.object):
            tag = view[1]
            for variant, overlay in overlays.items():
                written += save_panel(lambda axis: draw_mip(axis, postflight, dense, overlay, view, args),
                                      root/variant/f'{args.object}_{key}_mip_{tag}')
            if args.render_3d:
                written += save_panel(lambda axis: draw_3d(axis, points, view, args, voxel_m / 2),
                                      args.output_dir/'3d_panels'/f'{args.object}_{key}_3d_{tag}')
    written += save_colorbar(root/f'{args.object}_mip_colorbar', args)
    return [str(path) for path in written]


# ---------------------------------------------------------------- main
def main(argv=None):
    import scripts.run_b7873200_six_method_plenoxel_postflight_v1 as postflight
    postflight.import_runtime_dependencies()
    from scripts.eval_b787_geometry_metrics import sample_surface_points
    from scripts.render_b787_vs_stl import trilinear_sample_centers
    args = parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f'fresh output directory required: {args.output_dir}')
    args.output_dir.mkdir(parents=True)
    tris, surface, volume, truth, contract = load_truth(args)
    truth_points = sample_surface_points(tris, 300_000, np.random.default_rng(1))
    verts = truth_points[::5]          # dense mesh samples for the template's overlay scatter (STL vertices can be sparse)
    loaders = dict(rift=lambda p: via_pvc_evaluator(p, args, contract, 'RIFT'),
                   spinr=lambda p: via_pvc_evaluator(p, args, contract, 'SpINR'),
                   geraf=lambda p: load_geraf(p, args, contract),
                   radar_fields=lambda p: via_pvc_evaluator(p, args, contract, 'Radar Fields'),
                   radarsplat=lambda p: via_pvc_evaluator(p, args, contract, 'RadarSplat'),
                   se=lambda p: load_se(args.se_surface, args.se_checkpoint, args, contract),
                   se_stage1=lambda p: load_se_stage1(p, args),
                   backprojection=lambda p: load_backprojection(p, args))
    sources = dict(rift=args.rift, spinr=args.spinr, geraf=args.geraf, radar_fields=args.radar_fields,
                   radarsplat=args.radarsplat, se=args.se_surface, se_stage1=args.se_stage1,
                   backprojection=args.backprojection)
    rows, results = [], {}
    for key, title in METHODS:
        if sources[key] is None:
            results[key] = dict(status='not_supplied')
            continue
        base, interpolation, details = loaders[key](sources[key])
        np.save(args.output_dir/f'{key}_base_g{args.grid}.npy', base.astype(np.float32))
        variants = {}
        for name, factor in (('g48_lattice', 1), (f'dense_g{args.grid * args.upsample}', args.upsample)):
            local = SimpleNamespace(**{**vars(args), 'upsample': factor})
            normalized, stats = postflight.normalized_dense_magnitude(base, local, interpolation_input=interpolation)
            centers = trilinear_sample_centers(args.extent, args.grid, normalized.shape[0])
            metrics, points, _ = postflight.fixed_voxel_metrics(normalized, centers, surface, volume, truth, local)
            variants[name] = dict(field_stats=stats, **metrics['voxel'])
            if factor == args.upsample:
                dense, dense_points = normalized, points
        results[key] = dict(status='scored', title=title, source=str(Path(sources[key]).resolve()),
                            interpolation_input=interpolation, loader=details, metrics=variants)
        rows.append((key, title, dense, dense_points))
        print(f"{title:16s} g48 F1 {variants['g48_lattice']['f1']:.4f}  dense F1 "
              f"{variants[f'dense_g{args.grid * args.upsample}']['f1']:.4f}", flush=True)

    figures = render_panels(rows, verts, postflight, args)

    report = dict(schema=SCHEMA, object=args.object, scene=args.scene, dataset_identity=contract['dataset_identity'],
                  protocol=dict(base_grid=args.grid, upsample=args.upsample, threshold=args.fixed_threshold,
                                f1_tau_m=args.f1_tau, iou_unit_m=args.iou_unit, n_surface=args.n_surface,
                                n_volume=args.n_volume, gt_grid=args.gt_grid, extent_m=args.extent, crop_m=args.crop,
                                primary='g48_lattice (RIFT geometry evaluator default, upsample 1)',
                                views={v[1]: dict(name=v[0], forward_axis=v[2], camera_side=v[3], right=v[4], up=v[5])
                                       for v in views_for(args.object)}),
                  truth=truth, methods=results, radar_response_reads=0,
                  figures=figures,
                  figure_scale=dict(colormap='inferno', vmin=args.fixed_threshold, vmax=1.0, below_vmin='#050505',
                                    quantity='per-method min-max normalized 192^3 magnitude', panel_in=PANEL_IN,
                                    dpi=PANEL_DPI))
    (args.output_dir/'geometry_metrics.json').write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + '\n')
    print(f'POSTFLIGHT=PASS output={args.output_dir}')


if __name__ == '__main__':
    main()
