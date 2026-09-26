"""GeRaF (native GOTCHA) reconstruction on the presentation grid, relative and K-calibrated.

Plugs into ``rift_pvc.gotcha_presentation`` (see its docstring for the two
directories). Reads the checkpoint and TRAIN acquisition metadata (pulse
positions, r0, frequencies); no response of any role is read.

GeRaF's rendered response of one view is an exact sum over its target ray
samples s (``predict_native`` -> ``NativeAcquisition.trace`` ->
``source_amplitudes``):

    y_p = sum_s a_ps exp(-i 2 pi f (path_ps - 2 r0_p) / c),
    a_ps = sigma_s * spec_ps / path_ps^2 / N   (0 unless specular-active and in bounds),

with sigma_s = reflectivity * exp(light_power) * T^2 * alpha (the released
signal weight: sigmoid reflectivity, trainable log transmit amplitude
light_power, NeuS opacity alpha and transmittance T), spec = cos(2 theta) the
released monostatic specular gate, path = Rt + Rr = 2d, N the view's total
target-sample count (release normalization), trans_power = 1 (native units).
K's definition y / K = sqrt(RCS) / (Rt + Rr)^2 therefore gives each sample exactly

    sqrt(RCS_ps) = a_ps path_ps^2 / K = sigma_s spec_ps / (N K).

- physical: per TRAIN view and pulse, a cell's sqrt(RCS) is the CIC sum of its
  samples' amplitudes (the model's own in-phase sum over the cell, intra-cell
  phase ignored); RCS is its square, averaged over pulses and TRAIN views.
  Occlusion (T) and the specular gate are the model's own, for that view.
- relative: GeRaF's established readout is geometric (the zero-SDF surface,
  ``scripts/eval_geraf_geometry.py``) and carries no intensity, so the relative
  quantity is its renderer's signal weight sigma_s / N (before the specular gate
  and spreading), CIC-summed per cell, squared and averaged over TRAIN views.

CPU numerics: the release's float16 autocast is disabled on the CPU backend
(``rift_pvc.geraf_autocast``), so this readout runs float32 where a PVC card runs
float16 in the SDF/CDF regions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rift.gotcha_dataset import Region
from rift_pvc.gotcha_presentation import K, PRESENTATION_GRID, MethodPresentation, deposit_points

DEFAULT_VIEWS = None     # None = every TRAIN view


def dataset_from_contract(contract, shard_root):
    """The run's GOTCHA dataset (metadata only) rebuilt from its saved native contract."""
    from rift.gotcha_dataset import GOTCHADataset
    from rift.gotcha_frequency_selection import kwargs_from_contract
    from rift.gotcha_pulse_sampling import pulse_limit_from_contract
    r = contract['region']
    region = Region(r['name'], r['target_id'], tuple(r['translation_m']),
                    tuple(map(tuple, r['rotation_local_to_native'])), float(r['half_extent_m']),
                    r['placement_provenance'])
    selection = contract['split'].get('training_selection')
    shard_root = Path(shard_root)
    return GOTCHADataset(shard_root.parent.parent, shard_root=shard_root, region=region,
                         passes=tuple(contract['passes']), polarizations=tuple(contract['polarizations']),
                         num_train=None if selection is None else selection['num_train'],
                         pulses_per_sector=pulse_limit_from_contract(contract), **kwargs_from_contract(contract))


def load_models(checkpoint, data, device='cpu'):
    """Validate the checkpoint against the rebuilt data and load every head, as the readouts do."""
    from rift_pvc.geraf_source import (DEFAULTS, LEGACY_LIGHT_POWER_START, LEGACY_RECEIVER_GEOMETRY,
                                       build_model, recipe_for_data, source_step)
    from rift_pvc.geraf_source_training import validate_checkpoint
    saved = checkpoint['recipe']
    legacy = dict(receiver_geometry=LEGACY_RECEIVER_GEOMETRY, light_power_start=LEGACY_LIGHT_POWER_START)
    config = {k: saved[k] for k in DEFAULTS if k not in legacy}
    config.update({k: saved.get(k, v) for k, v in legacy.items()})
    recipe = recipe_for_data(config, data)
    validate_checkpoint(checkpoint, data, recipe)
    models = {}
    for head in data.heads:
        model = build_model(recipe, device)
        model.load_state_dict(checkpoint['models'][head], strict=True)
        model.update_step(source_step(recipe, checkpoint['step']))
        model.eval()
        models[head] = model
    return models, recipe


@torch.no_grad()
def view_scatterers(model, frame, acquisition):
    """The view's target samples as scatterers: the ``predict_native`` computation, stopped before the sum.

    Returns local points (m), per-pulse sqrt(RCS) [P, N] (m), per-sample signal
    weight sigma / N [N] (in bounds only) and the path lengths [P, N].
    """
    from rift.geraf_source_ops import source_amplitudes
    from rift_pvc.geraf_autocast import autocast_fp16
    from rift_pvc.vendor.geraf_sens.rf_rendering import _compute_transmission
    model.radar_cfg = {'native': acquisition}
    batch = model._extract_batch_inputs(frame)
    keep = torch.ones(len(frame['ray_d']), dtype=torch.bool, device=acquisition.tx.device)
    batch = model._apply_loss_mask(batch, keep)
    ids = torch.arange(len(acquisition.tx), device=acquisition.tx.device)
    prev, current = model._calibrate_transmission(batch['sampled_poses_norm'], ids, batch['r_pos_norm'])
    with autocast_fp16():
        alpha, gradients, features, _, _ = model.compute_sdf_alpha(
            batch['sampled_poses_norm'], batch['dists'], batch['dirs'], model.get_anneal_val(model.step), model.step)
    trans = _compute_transmission(alpha) - prev + current
    sigmas = model.signal_network(batch['sampled_poses_norm'], features, trans) * alpha * trans
    normals, sigmas, inside = model._sample_tgt_normals_and_sigmas(
        gradients, sigmas, batch['z_vals'], batch['tgt_z_vals'], len(ids))
    points = batch['tgt_sampled_poses_glb'].reshape(-1, 3).detach().to(acquisition.tx)
    inside = inside.flatten()
    total = len(points)
    amplitude = source_amplitudes(points, normals.reshape(-1, 3), sigmas.reshape(len(ids), -1).double(),
                                  acquisition.tx, acquisition.rx, inside, total)
    path = ((points[None] - acquisition.tx[:, None]).norm(dim=-1)
            + (acquisition.rx[:, None] - points[None]).norm(dim=-1))
    weight = torch.where(inside, sigmas.reshape(len(ids), -1)[0].double(), 0) / total
    return points, amplitude * path.square() / K, weight, path


def view_cells(points, amplitude, weight, extent, grid):
    """One view's cells: (CIC-summed signal weight)^2 and pulse-mean (CIC-summed sqrt(RCS))^2."""
    relative = deposit_points(points, weight, extent, grid) ** 2
    rcs = sum(deposit_points(points, a, extent, grid) ** 2 for a in amplitude) / len(amplitude)
    return relative, rcs


def geraf_presentation(checkpoint_path, *, shard_root, polarization='hh', grid=PRESENTATION_GRID,
                       max_views=DEFAULT_VIEWS, device='cpu'):
    from rift.geraf_source_data import GOTCHASourceData
    from rift_pvc.geraf_source import fixed_numpy_seed, sample_frame
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    contract = checkpoint['contract']['native_gotcha_contract']
    data = GOTCHASourceData(dataset_from_contract(contract, shard_root))
    if polarization not in data.heads:
        raise ValueError(f'checkpoint has no {polarization} head')
    models, recipe = load_models(checkpoint, data, device)
    model, extent = models[polarization], float(recipe['extent_m'])
    views = list(data.views('train'))
    chosen = (range(len(views)) if max_views is None or max_views >= len(views)
              else np.unique(np.linspace(0, len(views) - 1, max_views).round().astype(int)))
    relative = np.zeros((grid,) * 3)
    rcs = np.zeros((grid,) * 3)
    for index in chosen:
        view = views[index]
        acquisition = data.acquisition('train', view, polarization, recipe, device)
        # Fixed per-view sampling seed, as the validation readout and warm start use.
        with fixed_numpy_seed(recipe['seed'] + int(index)):
            frame = sample_frame(acquisition, recipe, data.key('train', view, polarization))
        points, amplitude, weight, _ = view_scatterers(model, frame, acquisition)
        view_relative, view_rcs = view_cells(points, amplitude, weight, extent, grid)
        relative += view_relative
        rcs += view_rcs
    count = len(chosen)
    history = checkpoint.get('validation_history') or []
    selected = min(history, key=lambda row: row['mf_magnitude_mse']) if history else None
    source = dict(checkpoint=str(Path(checkpoint_path).resolve()), step=checkpoint['step'],
                  best_validation=checkpoint.get('best_mse'),
                  validation_selected=bool(selected and selected['step'] == checkpoint['step']),
                  validation_at_step=next((row for row in history if row['step'] == checkpoint['step']), None),
                  light_power=float(model.signal_network.light_power), trans_power=recipe['trans_power'],
                  train_views=count, train_views_available=len(views), half_extent_m=extent,
                  conversion='sqrt(RCS_ps) = a_ps (Rt+Rr)^2 / K = sigma_s spec_ps / (N K); cell = CIC sum, '
                             'squared, mean over pulses and TRAIN views',
                  numerics='CPU float32 (release float16 autocast disabled on CPU)')
    return MethodPresentation('geraf', polarization, relative / count,
                              'GeRaF render weight sigma/N (reflectivity*exp(light_power)*T^2*alpha / N), '
                              'CIC cell sum squared, mean over TRAIN views (arbitrary units)',
                              rcs / count if polarization == 'hh' else None,
                              'RCS per G48 cell (m^2): model-rendered specular response incl. its occlusion, '
                              'mean over TRAIN views and pulses' if polarization == 'hh' else None,
                              source)
