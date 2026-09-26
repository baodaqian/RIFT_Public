"""Radar Fields on GOTCHA, presented as relative intensity and K-calibrated RCS.

Relative: the field's own ``rcs`` output, occupancy alpha times reflectance rho
(the quantity the renderer averages along its rays), queried at the G48 cell
centres with the view direction from each TRAIN look's mean antenna position,
averaged over the looks. BatchNorm in evaluation mode, mask progress 1, as in
validation.

Physical (see ``rift_pvc.gotcha_presentation_power``): for each look the
method's own GOTCHA renderer (``range_geometry``, ``prepare_bistatic_bins``,
``render_bistatic_batch``, ``radar_fields_intensity``) renders the per-bin mean
X(r) of alpha*rho over its ray samples, at a ray count raised from the training
10 to ``rays`` (the same released sampler; the training value is a Monte-Carlo
draw of this mean). The rendered intensity I = log10(X + offset) * scaler is
the model's value of the target t = (10 log10(P / peak) + DR) / DR, so the
claimed native matched-range power is P = peak * 10^(DR (I - 1) / 10).
Bins at or below the target floor are 0: t <= 0.1525 for the source profile,
whose targets are zeroed below that (-50.85 dB), and t <= 0 otherwise (the
-DR dB clip). The dB mapping makes a lone cell's render meaningless (a single
cell sits on the floor), so each bin's power is attributed to the ray samples
in proportion to their share of X(r) -- exactly the renderer's own sum -- and
each sample's share to the G48 cell containing it. Energy matching to a 1 m^2
point target at the region centre, formed by the method's own
``matched_range_power`` on the same bins, gives RCS per cell; looks are
averaged. Cells outside every rendered bin above the floor are 0.
"""
from __future__ import annotations

import contextlib
import os

import numpy as np
import torch

from rift.radar_fields import radar_fields_intensity
from rift.radar_fields_gotcha import matched_range_power, model_args, range_geometry
from rift.radar_fields_native import prepare_bistatic_bins, render_bistatic_batch
from rift.radar_fields_recipe import SOURCE_RECIPE
from rift_pvc.gotcha_presentation import PRESENTATION_GRID, MethodPresentation
from rift_pvc.gotcha_presentation_power import (DEFAULT_LOOKS, bin_points, cell_centres, dataset_from_contract,
                                                invert_db, look_metadata, point_target_observation, select_looks)

DEFAULT_RAYS = 2048
SOURCE_TARGET_FLOOR = .1525   # rift/radar_fields_gotcha.prepare_frame: target * (target > .1525)
TORCHSHIM_BACKEND = 'upstream-tcnn-torchshim'


@contextlib.contextmanager
def _shim_allowed_on_cpu():
    """The PVC shim refuses CPU unless RIFT_PVC_TCNN_SHIM=1; allow it for this readout only."""
    previous = os.environ.get('RIFT_PVC_TCNN_SHIM')
    os.environ['RIFT_PVC_TCNN_SHIM'] = '1'
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop('RIFT_PVC_TCNN_SHIM', None)
        else:
            os.environ['RIFT_PVC_TCNN_SHIM'] = previous


def load_head(checkpoint, polarization, device='cpu'):
    """The saved polarization head, rebuilt with the checkpoint's own backend on CPU.

    Extra saved keys (e.g. pose-refinement parameters) are tolerated and
    reported; a missing field parameter is an error.
    """
    recipe = checkpoint['recipe']
    backend = recipe['controls']['model_backend']
    args = model_args(recipe, device)
    if backend == TORCHSHIM_BACKEND:
        # As rift_pvc.radar_fields_training.build_model does for these recipes, without
        # its process-wide trainer rebinding: only the backend check learns the shim id
        # (it delegates every other backend to the original check).
        from rift.radar_fields_upstream import OriginalRadarFieldsModel
        from rift_pvc.radar_fields_upstream import install
        install()
        with _shim_allowed_on_cpu():
            model = OriginalRadarFieldsModel(args).to(device)
    elif backend == 'torch':
        from train_radar_fields import build_model
        model = build_model(args, torch.device(device))
    else:
        raise ValueError(f'Radar Fields backend {backend!r} needs CUDA/tiny-cuda-nn; present it where it was trained')
    prefix = f'{polarization}.'
    state = {k[len(prefix):]: v for k, v in checkpoint['model_state_dict'].items() if k.startswith(prefix)}
    if not state:
        raise ValueError(f'checkpoint has no {polarization} head')
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        raise ValueError(f'Radar Fields head is missing {result.missing_keys[:5]}')
    extra = sorted(set(checkpoint['model_state_dict']) - {prefix + k for k in state} |
                   {prefix + k for k in result.unexpected_keys})
    return model.eval(), extra


def target_floor(recipe):
    return SOURCE_TARGET_FLOOR if recipe['controls']['profile'] == SOURCE_RECIPE else 0.0


@torch.no_grad()
def look_power(model, recipe, peak_power, metadata, *, rays=DEFAULT_RAYS, seed=0, device='cpu'):
    """One look: rendered native power per bin, its per-sample attribution and the point-target energy.

    Returns sample positions, the native power attributed to each sample, the
    point-target energy e1 (1 m^2 at the region centre) and diagnostics.
    """
    controls = recipe['controls']
    mean = metadata['native'].mean(0)
    look = point_target_observation(mean, metadata['local'].mean(0), metadata['frequencies'])[0]
    antenna, ranges = range_geometry(look, metadata['region'], controls['range_guard_cells'], device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        geometry = prepare_bistatic_bins(antenna[None], antenna[None], ranges, extent=recipe['extent_m'],
                                         ray_samples=rays, source_sampling=controls['profile'] == SOURCE_RECIPE)
    rendered = render_bistatic_batch(model, [geometry], query_chunk=controls['query_chunk'], mask_progress=1.)[0]
    X = rendered['rcs'][0].double()
    intensity = radar_fields_intensity(X, ranges, offset=recipe['intensity_offset'],
                                       scaler=recipe['intensity_scaler'], range_law='released')
    power = invert_db(intensity, peak_power, controls['dynamic_range_db'], target_floor(recipe))
    # The renderer's own per-sample values, for the attribution of each bin's power.
    parameter = next(model.parameters())
    samples = model.query_chunked(geometry['xyz'].to(parameter), geometry['view'].to(parameter),
                                  mask_progress=1., chunk_size=controls['query_chunk'])['rcs'].double()
    _, bins, _ = geometry['inside'].nonzero(as_tuple=True)
    S = geometry['inside'].shape[-1]
    check = torch.zeros_like(X).index_add_(0, bins, samples) / S
    if not torch.allclose(check, X, rtol=1e-4, atol=1e-7 * float(X.abs().max().clamp_min(1))):
        raise RuntimeError('Radar Fields attribution does not reproduce the renderer average')
    share = torch.where(X[bins] > 0, samples / (S * X[bins].clamp_min(1e-300)), torch.zeros_like(samples))
    attributed = power[bins] * share
    e1 = float(matched_range_power(look, ranges).sum())
    return dict(xyz=geometry['xyz'].double(), power=attributed, e1=e1, bins_above_floor=int((power > 0).sum()),
                rendered_power=float(power.sum()), attributed_power=float(attributed.sum()),
                look_range_m=float(torch.linalg.vector_norm(antenna)))


@torch.no_grad()
def relative_intensity(model, recipe, looks, extent, grid, device='cpu'):
    """alpha*rho at the G48 cell centres, mean over the looks' view directions."""
    centres = cell_centres(extent, grid)
    parameter = next(model.parameters())
    total = torch.zeros(len(centres), dtype=torch.float64)
    for metadata in looks:
        antenna = torch.as_tensor(metadata['local'].mean(0), dtype=torch.float64)
        view = torch.nn.functional.normalize(centres - antenna, dim=-1)
        total += model.query_chunked(centres.to(parameter), view.to(parameter), mask_progress=1.,
                                     chunk_size=recipe['controls']['query_chunk'])['rcs'].double().clamp_min(0)
    return (total / len(looks)).view(grid, grid, grid).numpy()


def radar_fields_presentation(checkpoint_path, *, shard_root, polarization='hh', grid=PRESENTATION_GRID,
                              looks=DEFAULT_LOOKS, rays=DEFAULT_RAYS):
    """Radar Fields GOTCHA checkpoint -> relative alpha*rho and K-calibrated RCS per G48 cell."""
    from pathlib import Path
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    recipe, contract = checkpoint['recipe'], checkpoint['dataset_contract']
    if recipe.get('method') != 'radar_fields':
        raise ValueError('not a Radar Fields GOTCHA checkpoint')
    extent = float(contract['region']['half_extent_m'])
    model, extra = load_head(checkpoint, polarization)
    dataset = dataset_from_contract(contract, shard_root)
    selected = select_looks(dataset.viewpoints('train'), looks)
    metadata = [dict(look_metadata(dataset, view, polarization), region=dataset.region) for view in selected]
    relative = relative_intensity(model, recipe, metadata, extent, grid)
    stats = checkpoint['training_statistics']
    peak = float(stats['peak_power'][polarization])
    source = dict(checkpoint=str(Path(checkpoint_path).resolve()), step=checkpoint.get('step'),
                  best_validation=checkpoint.get('best_val'), complete=checkpoint.get('complete'),
                  profile=recipe['controls']['profile'], model_backend=recipe['controls']['model_backend'],
                  half_extent_m=extent, looks=len(selected), train_looks=len(dataset.viewpoints('train')),
                  extra_checkpoint_keys=extra[:20], train_peak_power=peak,
                  dynamic_range_db=recipe['controls']['dynamic_range_db'])
    rcs, physical_quantity = None, None
    if polarization == 'hh':
        total = np.zeros((grid,) * 3)
        rendered = attributed = outside = 0.
        for index, look in enumerate(metadata):
            result = look_power(model, recipe, peak, look, rays=rays, seed=index)
            cells, lost = bin_points(result['xyz'], result['power'] / result['e1'], extent, grid)
            total += cells
            outside += lost
            rendered += result['rendered_power'] / result['e1']
            attributed += result['attributed_power'] / result['e1']
        rcs = total / len(metadata)
        physical_quantity = ('RCS per G48 cell (m^2): rendered native matched-range power above the dB floor, '
                             'attributed to cells by their share of the renderer ray average, energy-matched to a '
                             'point target; mean over TRAIN looks')
        source.update(rays=rays, target_floor_intensity=target_floor(recipe),
                      rendered_rcs_m2=rendered / len(metadata), attributed_rcs_m2=attributed / len(metadata),
                      conversion=('P(r) = peak 10^(DR (I(r) - 1) / 10) for I > floor, I = log10(X + offset) * scaler; '
                                  'P_cell = sum_r P(r) * sum_{samples in cell} v_s / (S X(r)); sigma = P_cell / e1, '
                                  'e1 = sum_r |matched filter of K/(2R)^2 point|^2 (= K^2 / (2R)^4 at its bin)'))
    return MethodPresentation('radar_fields', polarization, relative,
                              'field alpha*reflectance (renderer rcs) at G48 cell centres, mean over TRAIN look '
                              'directions (arbitrary units)', rcs, physical_quantity, source)
