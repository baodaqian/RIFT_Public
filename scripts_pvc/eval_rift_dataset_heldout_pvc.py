#!/usr/bin/env python3
"""Held-out (validation or reserved-test) signal scoring of the comparison methods on the RIFT dataset (PVC).

Every method is scored with exactly the RIFT-dataset evaluator's definitions
(``scripts/eval_b787_range_power.py``, user decision 2026-09-22): per view, the
prediction's complex spectrum per Tx/Rx pair gives

* coherent complex RelMSE: sum |S_pred - S|^2 / sum |S|^2 against the chirp-mean
  measurement, over every selected pair and native frequency;
* MF (range) power RelMSE: P = |IFFT_f(S)|^2 in range bins, both sides through the
  same ``normalize_power_db`` with the TRAIN peak of the object's Radar Fields power
  statistics (``normalized_range_power_rel_mse``), and the linear-power variant
  (``linear_range_power_rel_mse``), over the ROI range bins of that view;

pooled over the role's views by the original ``aggregate``. The loaders, role
selection (``--role test`` needs ``--allow-reserved-test``), normalization check,
per-view cache and resume come from the original; only the prediction differs,
through one adapter per method, which also checks that its checkpoint belongs to
the registered object and split. Power-only methods have no coherent score
(reported as null, never as a pooled zero).

    python scripts_pvc/eval_rift_dataset_heldout_pvc.py --method spinr --object a320 \\
        --dataset-root D --role-manifest RUN/role_manifest.json \\
        --checkpoint RUN/spinr/budget48-direct-150/checkpoint_best.pth.tar \\
        --stats RF_RUN/radar_fields/radar_fields_power_stats.json --label a320_spinr_test \\
        --out-dir OUT --role test --allow-reserved-test --device cpu

Adapters: spinr, geraf, radar_fields, sugavanam_ertin (Stage-1 fields), radarsplat (image -> profile).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import scripts.eval_b787_range_power as original  # noqa: E402
from rift.radar_fields_dataset import (  # noqa: E402
    build_frequency_grid, normalize_power_db, range_bin_centers, response_view_to_range_power,
    restrict_radar_fields_response_views, scene_range_mask)
from scripts_pvc.eval_b787_range_power_pvc import validate_normalization_stats  # noqa: E402

SCHEMA = 'rift_dataset_heldout_signal_scores_v1'


# ---------------------------------------------------------------- adapters
class SpinrAdapter:
    """SpINR-style INR: the fixed field rendered through its own range operator per view."""

    kind = 'complex'

    def __init__(self, checkpoint_path, contract, arrays, device, args):
        import train_spinr_style_pvc as spinr
        from rift.range_operator import range_forward_operator
        from rift.rift_dataset import collection_contract
        self.spinr, self.render = spinr, range_forward_operator
        ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if ck.get('format') != spinr.CHECKPOINT_FORMAT:
            raise ValueError(f"not a {spinr.CHECKPOINT_FORMAT} checkpoint")
        if collection_contract(ck.get('sealed_npz_protocol_contract', {})) != collection_contract(contract):
            raise ValueError('SpINR checkpoint was not trained on the registered object/split')
        operator = ck['spinr_style_recipe']['operator']
        if (operator['phase_sign'] != spinr.SPINR_STYLE_PHASE_SIGN or operator['range_model'] != spinr.SPINR_STYLE_RANGE_MODEL
                or operator['physics_dtype'] != 'float64_complex128'):
            raise ValueError('SpINR checkpoint operator differs from this code')
        self.frequencies = torch.as_tensor(ck['acquisition_identity']['frequency_hz'], dtype=torch.float64, device=device)
        metadata_grid = torch.as_tensor(build_frequency_grid(arrays.metadata), dtype=torch.float64, device=device)
        if self.frequencies.shape != metadata_grid.shape or not torch.allclose(self.frequencies, metadata_grid, rtol=0, atol=1.0):
            raise ValueError('SpINR checkpoint frequencies differ from the object metadata')
        self.kvector = spinr.get_kvector(self.frequencies, spinr.cc).to(torch.float64)
        if operator['quadrature'] == 'midpoint':
            self.points, cell_volume = spinr.midpoint_grid(operator['grid_size'], device=device, dtype=torch.float64)
        else:
            self.points, cell_volume = spinr.gauss_legendre_cell_grid(
                operator['grid_size'], nodes_per_cell=operator['nodes_per_cell'], device=device, dtype=torch.float64)
        if len(self.points) != operator['integration_points']:
            raise ValueError('SpINR integration grid does not match its recipe')
        model = spinr.SpinrStyleINR().to(device=device, dtype=torch.float32)
        model.load_state_dict(ck['model_state_dict'])
        model.eval()
        self.initial_output_scale = float(ck['normalization']['initial_output_scale'])
        field = spinr.evaluate_neural_field_tiled(model, self.points, neural_point_tile=4096)
        self.weights = spinr.scale_field_to_renderer_weights(field, cell_volume_m3=cell_volume,
                                                             initial_output_scale=self.initial_output_scale)
        self.identity = dict(method='spinr', format=ck['format'], recipe_id=ck['spinr_style_recipe']['recipe_id'],
                             epoch_index=int(ck['epoch_index']), selection=ck.get('selection'),
                             integration_points=len(self.points), initial_output_scale=self.initial_output_scale)

    def predict(self, view_index, position, rx_pos, tx_pos, viewpoint):
        return self.render(self.frequencies, self.kvector, rx_pos.to(torch.float64), tx_pos.to(torch.float64),
                           self.points, self.weights, phase_sign=self.spinr.SPINR_STYLE_PHASE_SIGN,
                           pair_chunk=1, point_chunk=65536, compute_dtype=torch.float64,
                           range_model=self.spinr.SPINR_STYLE_RANGE_MODEL)       # [nf, Rx, Tx]


class GerafAdapter:
    """GeRaF (source_v1): the validation-selected model's unmasked native readout (``predict_native``).

    Rays are drawn by GeRaF's own samplers with the per-view seed its validation uses
    (``recipe.seed`` + position in the role); the view's measured MF volume only
    supplies training targets and loss masks, and ``predict_native`` renders every
    ray unmasked, so the held-out prediction never reads the held-out response. The
    readout is in units of response / ``trans_power``; it is scaled back to raw units.
    """

    kind = 'complex'

    def __init__(self, checkpoint_path, contract, arrays, device, args):
        from rift.geraf_source import DEFAULTS, LEGACY_LIGHT_POWER_START, LEGACY_RECEIVER_GEOMETRY
        from rift.geraf_source_data import RIFTSourceData
        from rift_pvc.geraf_source import fixed_numpy_seed, predict_native, recipe_for_data, sample_frame
        from rift_pvc.geraf_source_training import load_selected_models
        if (arrays.num_tx, arrays.num_rx) != (1, 1):
            raise ValueError('GeRaF held-out adapter is defined for the 1 Tx x 1 Rx production acquisition')
        self.seed, self.sample, self.predict_native = fixed_numpy_seed, sample_frame, predict_native
        # The training data object (train/validation roles) binds the checkpoint's contract unchanged.
        self.data = RIFTSourceData(args.npz_path, args.role_manifest)
        ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        models, selected = load_selected_models(ck, self.data, device)
        saved = ck['recipe']
        legacy = dict(receiver_geometry=LEGACY_RECEIVER_GEOMETRY, light_power_start=LEGACY_LIGHT_POWER_START)
        config = {k: saved[k] for k in DEFAULTS if k not in legacy}
        config.update({k: saved.get(k, v) for k, v in legacy.items()})
        self.recipe = recipe_for_data(config, self.data)      # as load_selected_models
        self.model, self.device, self.role = models['scalar'], device, args.role
        self.num_freq = arrays.num_freq
        self.identity = dict(method='geraf', schema=ck.get('schema'), step=int(ck['step']), selected=selected,
                             trans_power=float(self.recipe['trans_power']), seed=int(self.recipe['seed']),
                             receiver_geometry=self.recipe.get('receiver_geometry'))

    def acquisition(self, view):
        """``RIFTSourceData.acquisition`` for any registered view (geometry metadata only, no response)."""
        from rift.geraf_signal_operator import bistatic_pair_positions
        from rift.geraf_source_ops import NativeAcquisition
        from rift.power_baseline_dataset import frequency_grid_hz
        a, device, recipe = self.data.arrays, self.device, self.recipe
        tx, rx = bistatic_pair_positions(torch.as_tensor(a['tx_pos'][view], device=device),
                                         torch.as_tensor(a['rx_pos'][view], device=device))
        freqs = torch.as_tensor(frequency_grid_hz(a['meta']), device=device)
        return NativeAcquisition(tx, rx, freqs, torch.zeros(len(tx), dtype=torch.float64, device=device),
                                 point_chunk=recipe['point_chunk'], pair_chunk=min(recipe['pair_chunk'], len(tx)),
                                 uniform_rift=True)

    def predict(self, view_index, position, rx_pos, tx_pos, viewpoint):
        acquisition = self.acquisition(view_index)
        with self.seed(self.recipe['seed'] + position):
            frame = self.sample(acquisition, self.recipe, f'{self.role}_scalar_{view_index:05d}', None, None)
        prediction = self.predict_native(self.model, frame, acquisition)              # [pairs, nf]
        return (prediction * self.recipe['trans_power']).T.reshape(self.num_freq, 1, 1)


class RadarFieldsAdapter:
    """Radar Fields (power only): its rendered dB-normalized range-bin intensity over the view's ROI.

    The model is rebuilt from the checkpoint's own training arguments through the PVC
    twin (tcnn torch shim) and rendered by ``audited_view_tensors`` with
    ``mask_progress=1.0`` on every pair, as its own validation. Its random ray samples
    are seeded per view (the view's position in the role), so a score does not depend
    on the evaluator's read order or on resuming; its own validation instead seeds once
    and walks the views in order. Its prediction lives in the normalized
    domain, so the normalized range-power score is direct; the linear score inverts the
    dB mapping (derived). Radar Fields trained against a target zeroed below 0.1525
    (released noise floor); this score uses the evaluator's unmasked target.
    """

    kind = 'intensity'

    def __init__(self, checkpoint_path, contract, arrays, device, args):
        import argparse as _argparse
        import train_radar_fields_pvc as rf_pvc
        from rift_pvc import radar_fields_training as twins
        from scripts_pvc.eval_b787_geometry_metrics_pvc import validate_radar_fields_selection
        self.rf = rf_pvc.install()
        ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if ck.get('artifact_schema') != 'radar_fields_checkpoint_v2':
            raise ValueError('not a radar_fields_checkpoint_v2 checkpoint')
        validate_radar_fields_selection(ck['sealed_protocol_contract'], contract)
        from rift.rift_dataset import validate_checkpoint_object
        validate_checkpoint_object(ck, contract)
        self.args = _argparse.Namespace(**ck['args'])
        self.args.device = str(device)
        stats = json.loads(Path(args.stats).read_text())
        if any(float(ck['power_stats'][k]) != float(stats[k]) for k in ('peak_power', 'dynamic_range_db')):
            raise ValueError('Radar Fields checkpoint power stats differ from --stats')
        self.stats = ck['power_stats']
        self.model = twins.build_model(self.args, device)
        self.model.load_state_dict(ck['radar_fields_state_dict'])
        self.model.eval()
        self.ranges = self.rf.range_bin_centers(arrays.metadata, device=device,
                                                dtype=torch.float64 if self.rf.native_recipe(self.args) else torch.float32)
        self.pairs = self.rf.evenly_spaced_pairs(arrays.num_tx * arrays.num_rx, self.args.val_pairs)
        if len(self.pairs) != arrays.num_tx * arrays.num_rx:
            raise ValueError('Radar Fields validation pairs are a subset; held-out scoring needs every pair')
        self.arrays, self.device = arrays, device
        self.own = dict(squared_error=0.0, target_energy=0.0, views=0)
        self.identity = dict(method='radar_fields', schema=ck['artifact_schema'], step=int(ck['step']),
                             best_val_rel_mse=float(ck['best_val_rel_mse']), recipe=ck['radar_fields_recipe'].get('recipe'),
                             model_backend=self.args.model_backend, tcnn_shim=ck.get('tcnn_shim'),
                             linear_range_power='derived by inverting the normalized dB intensity (power-only method)',
                             training_target='normalized dB intensity zeroed below 0.1525 (released noise floor)')

    def bind(self, arrays):
        self.arrays = arrays

    def extra_summary(self):
        own = self.own
        return dict(method_own_metric=dict(
            definition='its training/validation rel_mse: masked normalized-dB target, ROI bins',
            views_this_invocation=own['views'],
            rel_mse=own['squared_error'] / own['target_energy'] if own['target_energy'] > 0 else None))

    def predict(self, view_index, position, rx_pos, tx_pos, viewpoint):
        torch.manual_seed(int(position))       # per-view ray samples: order- and resume-independent
        out = self.rf.audited_view_tensors(self.model, self.arrays, view_index, self.pairs, self.ranges, self.stats,
                                           self.args, 1.0, self.device)
        roi = out['roi']
        # Its own training metric (masked target) for these views, a reproduction check only.
        self.own['squared_error'] += float((out['prediction'] - out['target']).square().sum())
        self.own['target_energy'] += float(out['target'].square().sum())
        self.own['views'] += 1
        intensity = torch.zeros(len(self.pairs), len(roi), dtype=out['prediction'].dtype, device=self.device)
        intensity[:, roi] = out['prediction']
        return dict(intensity=intensity, roi=roi)


class SugavanamErtinStage1Adapter:
    """Sugavanam-Ertin scored through its Stage-1 sub-aperture fields (user decision 2026-09-23).

    Its Stage-2 SDF has no signal model, so the method's radar prediction is its
    Stage-1 model: each held-out view is assigned to a sub-aperture exactly as its
    validation readout assigns validation views (``Subapertures.assign``: the view's
    occupied angular bin, else the nearest TRAIN sub-aperture direction) and rendered
    with that sub-aperture's field through SE's own first-order Fourier operator
    (``CollectionAcquisition.render``), in units of response / TRAIN RMS, scaled back
    to raw units. The partition is rebuilt from the record saved with the fields (never
    recomputed on the evaluation node). ``--checkpoint`` may be ``stage1_selected.pt``
    (the fields Stage 2 used) or a Stage-1/Stage-2 ``checkpoint_latest.pt``.
    """

    kind = 'complex'

    def __init__(self, checkpoint_path, contract, arrays, device, args):
        from rift.sugavanam_ertin_acquisition import CollectionAcquisition, digest
        from rift.sugavanam_ertin_paper import Subapertures
        from rift.sugavanam_ertin_paper_workflow import grid_points
        if (arrays.num_tx, arrays.num_rx) != (1, 1):
            raise ValueError('SE held-out adapter is defined for the 1 Tx x 1 Rx production acquisition')
        ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.acquisition = CollectionAcquisition(npz_path=args.npz_path, manifest=args.role_manifest)
        if digest(ck['acquisition']) != digest(self.acquisition.identity):
            raise ValueError('SE checkpoint belongs to another acquisition/object/split')
        if 'selected_fields' in ck:
            fields, source = ck['selected_fields'], 'stage1_selected'
        elif isinstance(ck.get('best'), dict) and 'fields' in ck['best']:
            fields, source = ck['best']['fields'], 'best_stage1_fields'
        else:
            raise ValueError('SE checkpoint carries no selected Stage-1 fields')
        record, self.recipe = ck['partition'], ck['recipe']
        self.partition = Subapertures(record['azimuth_bins'], record['elevation_bins'], np.asarray(record['occupied_bins']),
                                      np.asarray(record['directions']), np.asarray(record['train_assignments']))
        if len(fields) != len(self.partition.directions):
            raise ValueError('SE fields do not match the saved partition')
        self.fields = fields.to(device)
        self.rms = float(ck['statistics']['rms'])
        self.points = grid_points(self.acquisition.extent, self.recipe['granularity'], device)
        self.device, self.fallbacks = device, 0
        self.viewpoints = arrays.viewpoint_positions
        self.identity = dict(method='sugavanam_ertin', scored_model='stage1_subaperture_fields', fields_source=source,
                             phase=ck.get('phase'), groups=len(fields), rms=self.rms,
                             granularity=int(self.recipe['granularity']), stage1_solver=self.recipe.get('stage1_solver'),
                             logged_validation=(ck.get('best') or {}).get('validation'),
                             assignment='Subapertures.assign (occupied angular bin, else nearest TRAIN direction)')

    def bind(self, arrays):
        self.arrays = arrays

    def extra_summary(self):
        return dict(sugavanam_ertin_empty_bin_fallback_views=self.fallbacks)

    def predict(self, view_index, position, rx_pos, tx_pos, viewpoint):
        group, fallback = self.partition.assign(np.asarray(viewpoint, dtype=np.float64)[None])
        self.fallbacks += int(fallback[0])
        observation = dict(tx=self.arrays.tx_pos[view_index], rx=self.arrays.rx_pos[view_index],
                           freqs=self.acquisition.freqs)
        prediction = self.acquisition.render(self.points, self.fields[int(group[0])], observation,
                                             point_chunk=self.recipe['point_chunk'], pair_chunk=self.recipe['pair_chunk'])
        return prediction * self.rms                                                  # [nf, Rx, Tx]


def single_pair_profile_map(points, tx, rx, freqs_hz, n_elevation, n_pixels, *, step=0.05):
    """Linear map from a 1 Tx x 1 Rx range-power profile to its polar MF-power image.

    With one antenna pair the matched filter A(x) = sum_f S_f exp(+i 2 pi f R(x)/c)
    (``rift/matched_filter_power.py``, phase_sign -1) depends on x only through the
    bistatic path R = |x - tx| + |x - rx|: |A|^2 = N^2 p(n), with p the continuous
    |IFFT_f S|^2 at bin index n = R B / c. Each image pixel sums |A|^2 over its
    elevation samples. p is parametrized on a uniform grid of n (``step`` bins,
    linear interpolation). Returns (matrix [pixels, nodes], nodes, sample n).
    """
    C = 299792458.0
    n_freq = len(freqs_hz)
    bandwidth = float(freqs_hz[-1] - freqs_hz[0]) * n_freq / (n_freq - 1)
    R = np.linalg.norm(points - tx, axis=1) + np.linalg.norm(points - rx, axis=1)
    n = R * bandwidth / C
    lo, hi = np.floor(n.min()) - 1, np.ceil(n.max()) + 1
    nodes = np.arange(lo, hi + step / 2, step)
    j = np.clip(((n - lo) / step).astype(int), 0, len(nodes) - 2)
    w = (n - nodes[j]) / step
    matrix = np.zeros((n_pixels, len(nodes)))
    pixel = np.tile(np.arange(n_pixels), n_elevation)       # samples ordered (elevation, azimuth, range)
    np.add.at(matrix, (pixel, j), 1 - w)
    np.add.at(matrix, (pixel, j + 1), w)
    return matrix * n_freq ** 2, nodes, n


def nonnegative_profile_fit(matrix, image, n_freq, n_elevation):
    """Non-negative least-squares range-power profile of an image through ``single_pair_profile_map``.

    A power profile is non-negative. Plain least squares reproduces measured (single-pair
    consistent) images exactly but oscillates without bound on images no single pair can
    produce, such as a model's predictions; the bound keeps the projection stable. Rows are
    scaled to elevation-mean interpolation weights and the right-hand side to unit maximum
    (measured on 30 A320 validation views: stored targets reproduced to <= 0.18% of each
    view's peak; predicted profiles <= 4x the image scale, never negative).
    """
    from scipy.optimize import lsq_linear
    columns = np.flatnonzero(matrix.sum(0) > 0)
    weights = matrix[:, columns] / (n_freq ** 2 * n_elevation)
    values = np.asarray(image, dtype=np.float64).reshape(-1) / (n_freq ** 2 * n_elevation)
    scale = float(np.abs(values).max()) or 1.0
    solution = lsq_linear(weights, values / scale, bounds=(0, np.inf), method='bvls').x
    profile = np.zeros(matrix.shape[1])
    profile[columns] = solution * scale
    return profile


class RadarSplatAdapter:
    """RadarSplat (released recipe, power only) through the exact single-pair image -> profile transfer.

    Its prediction is a polar MF-power image (range x azimuth, elevation summed). With the
    production 1 Tx x 1 Rx acquisition that image is a known linear map of the view's
    per-pair range-power profile (``single_pair_profile_map``; checked on stored
    validation targets: forward to 1.7e-5). The profile is the non-negative least-squares
    fit of the predicted image through that map (``nonnegative_profile_fit``), read at the
    evaluator's integer range bins inside RadarSplat's crop; only ROI bins inside the crop
    are scored. The model, renderer and nearest-TRAIN multipath background are its
    released readout's; a held-out view's polar grid is built from its geometry with the
    target builder's own function (checked equal to the cached grid where one exists).
    """

    kind = 'power_profile'

    def __init__(self, checkpoint_path, contract, arrays, device, args):
        import rift_pvc.radarsplat_release_training as rs_pvc
        rs_pvc.install()
        import rift.radarsplat_release_training as engine
        import train_radarsplat as lifecycle
        from rift.radarsplat_b7873200_protocol import RECIPE_FILENAME, load_cache
        from rift.radarsplat_release import (SCHEMA, create_scene, intensity_mapping_from_identity, model_recipe,
                                             position_scheduler, profile_from_identity)
        from rift.rift_dataset import validate_checkpoint_object
        if (arrays.num_tx, arrays.num_rx) != (1, 1):
            raise ValueError('The image -> profile transfer is exact only for one antenna pair')
        self.cache_root = Path(checkpoint_path).resolve().parent.parent/'targets'
        ck = lifecycle._load_checkpoint(Path(checkpoint_path), device)
        identity = ck.get('identity', {})
        profile = profile_from_identity(identity)
        target_recipe = json.loads((self.cache_root/RECIPE_FILENAME).read_text())
        if (ck.get('schema') != SCHEMA or identity.get('model_recipe') != model_recipe(profile)
                or identity.get('target_recipe') != target_recipe
                or identity.get('dataset_identity') != target_recipe.get('sealed_protocol_identity')):
            raise ValueError('RadarSplat checkpoint/cache/recipe mismatch')
        validate_checkpoint_object({'sealed_protocol_identity': identity['dataset_identity']}, contract)
        rendering, _ = engine.load_cuda_reference(device=device)
        self.cache = load_cache(self.cache_root)
        self.mapping = intensity_mapping_from_identity(identity)
        if (identity != engine.identity_for_cache(self.cache, profile, self.mapping)
                or not lifecycle._directly_equal(ck.get('acquisition_record'), self.cache.acquisition_record)):
            raise ValueError('RadarSplat readout recipe/calibration mismatch')
        self.splats, optimizers = create_scene(scene_scale=identity['adapter']['initialization_scene_scale'],
                                               scene_center=np.zeros(3), device=device,
                                               num_points=identity['model_recipe']['init_num_pts'])
        scheduler = position_scheduler(optimizers)
        sampler = lifecycle.DeterministicViewSampler(self.cache.train_indices, 42)
        self.step = engine.restore_state(ck, self.splats, optimizers, scheduler, sampler)
        self.active_degree = min((self.step - 1) // 200, 5)
        self.renderer = engine.renderer_for_cache(rendering, self.cache, identity)
        self.preprocessing = engine.ReleasedPreprocessing(self.cache, intensity_mapping=self.mapping)
        self.units = identity['adapter']['model_units_per_m']
        self.grid_spec = target_recipe['target_spec']['grid']
        self.freqs = build_frequency_grid(arrays.metadata)
        self.device, self.arrays = device, arrays
        self.own = dict(squared_error=0.0, target_energy=0.0, views=0)
        summary_path = Path(checkpoint_path).resolve().parent/'summary.json'
        logged = json.loads(summary_path.read_text()).get('validation') if summary_path.exists() else None
        self.identity = dict(method='radarsplat', schema=ck['schema'], profile=profile, step=int(self.step),
                             intensity_mapping=self.mapping, train_peak_power=float(self.cache.train_peak_power),
                             logged_validation=logged, cache_root=str(self.cache_root),
                             transfer='non-negative least-squares single-pair profile from the predicted polar MF-power image',
                             scored_bins='evaluator ROI bins inside the RadarSplat range crop')

    def bind(self, arrays):
        self.arrays = arrays

    def extra_summary(self):
        own = self.own
        return dict(method_own_metric=dict(
            definition='native_clipped_power_relative_mse on its cached targets (views in its cache only)',
            views_this_invocation=own['views'], max_rebuilt_vs_cached_grid_difference=own.get('max_grid_difference'),
            rel_mse=own['squared_error'] / own['target_energy'] if own['target_energy'] > 0 else None,
            image_linear_power_rel_mse=(own['image_linear_squared_error'] / own['image_linear_target_energy']
                                        if own.get('image_linear_target_energy') else None)))

    def _view_arrays(self, view_index):
        from rift.power_baseline_dataset import build_radarsplat_target_grid
        g = self.grid_spec
        grid = build_radarsplat_target_grid(
            self.arrays.viewpoint_positions[view_index],
            torch.as_tensor(self.arrays.tx_pos[view_index], dtype=torch.float32),
            torch.as_tensor(self.arrays.rx_pos[view_index], dtype=torch.float32),
            scene_center=g['scene_center_m'], scene_extent_m=float(g['scene_extent_m']), n_azimuth=int(g['n_azimuth']),
            n_elevation=int(g['n_elevation']), n_range=int(g['n_range']),
            output_azimuth_resolution_deg=float(g['output_azimuth_resolution_deg']),
            elevation_sampling_resolution_deg=float(g['elevation_sampling_resolution_deg']), device='cpu',
            dtype=torch.float32)
        return {key: getattr(grid, key).detach().to(torch.float32).cpu().numpy()
                for key in ('sensor_to_world', 'range_m', 'azimuth_rad', 'elevation_rad')}

    def predict(self, view_index, position, rx_pos, tx_pos, viewpoint):
        from rift.radarsplat_b7873200_adapter import target_grid_from_arrays
        from rift.radarsplat_fidelity import polar_world_points
        from rift.radarsplat_release import intensity
        view = self._view_arrays(view_index)
        grid = target_grid_from_arrays(range_m=view['range_m'], azimuth_rad=view['azimuth_rad'],
                                       expected_grid=self.cache.grid)
        rendered, _ = self.renderer(self.splats, torch.as_tensor(view['sensor_to_world'], device=self.device), grid,
                                    self.active_degree,
                                    torch.as_tensor(self.preprocessing.background(view), device=self.device))
        rendered = rendered.detach().double().cpu().numpy()
        cached = self.cache_root/'radarsplat_b7873200_views'/f'view_{int(view_index):06d}.npz'
        if cached.exists():
            with np.load(cached) as saved:
                # The cache was built on another device: float32 last-bit rounding only (measured: 1 ulp,
                # 1.9e-6 m in range against a 9 mm range pixel).
                for key in ('sensor_to_world', 'range_m', 'azimuth_rad', 'elevation_rad'):
                    difference = float(np.abs(saved[key].astype(np.float64) - view[key]).max())
                    self.own['max_grid_difference'] = max(self.own.get('max_grid_difference', 0.0), difference)
                    if difference > 1e-5:
                        raise ValueError(f'view {view_index}: rebuilt RadarSplat grid differs from its cache ({key})')
                target = np.clip(intensity(saved['radarsplat_mf_power'].astype(np.float64), self.cache.train_peak_power,
                                           self.mapping), 0, 1)
            ranges = view['range_m'].astype(np.float64)
            target = target * (ranges >= 2.5 / self.units)
            self.own['squared_error'] += float(np.square(rendered - target).sum())
            self.own['target_energy'] += float(np.square(target).sum())
            self.own['views'] += 1
        # Image intensity -> MF power, inverting the recipe's mapping.
        peak = float(self.cache.train_peak_power)
        if self.mapping == 'linear_train_peak_v1':
            image_power = rendered * peak
        else:
            image_power = peak * np.power(10.0, 6.0 * (rendered - 1.0))
        if cached.exists():
            # Linear power in the image domain (no profile transfer): its image vs the cached MF-power image.
            with np.load(cached) as saved:
                target_power = saved['radarsplat_mf_power'].astype(np.float64) * (ranges >= 2.5 / self.units)
            self.own['image_linear_squared_error'] = (self.own.get('image_linear_squared_error', 0.0)
                                                      + float(np.square(image_power - target_power).sum()))
            self.own['image_linear_target_energy'] = (self.own.get('image_linear_target_energy', 0.0)
                                                      + float(np.square(target_power).sum()))
        points = polar_world_points(view)
        tx = np.asarray(self.arrays.tx_pos[view_index], dtype=np.float64)[0]
        rx = np.asarray(self.arrays.rx_pos[view_index], dtype=np.float64)[0]
        matrix, nodes, n = single_pair_profile_map(points, tx, rx, self.freqs, len(view['elevation_rad']),
                                                   image_power.size)
        profile = nonnegative_profile_fit(matrix, image_power, len(self.freqs), len(view['elevation_rad']))
        bins = np.arange(int(np.ceil(n.min())), int(np.floor(n.max())) + 1)
        power = torch.zeros(1, len(self.freqs), dtype=torch.float64, device=self.device)
        mask = torch.zeros(len(self.freqs), dtype=torch.bool, device=self.device)
        power[0, bins] = torch.as_tensor(np.interp(bins, nodes, profile), dtype=torch.float64, device=self.device)
        mask[bins] = True
        return dict(power=power, mask=mask)


ADAPTERS = {'spinr': SpinrAdapter, 'geraf': GerafAdapter, 'radar_fields': RadarFieldsAdapter,
            'sugavanam_ertin': SugavanamErtinStage1Adapter, 'radarsplat': RadarSplatAdapter}


# ---------------------------------------------------------------- harness
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--method', required=True, choices=sorted(ADAPTERS))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--object', required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--role-manifest', required=True)
    parser.add_argument('--stats', required=True, help="the object's Radar Fields power stats (TRAIN peak)")
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--role', choices=('validation', 'test'), default='validation')
    parser.add_argument('--allow-reserved-test', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-views', type=int, default=0, help='debug cap; 0 scores every view of the role')
    parser.add_argument('--save-every', type=int, default=25)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.role == 'test' and not args.allow_reserved_test:
        parser.error('reserved-test scoring requires --allow-reserved-test')
    if args.max_views < 0 or args.save_every <= 0:
        parser.error('--max-views must be nonnegative and --save-every positive')
    from rift.rift_dataset import resolve_object_inputs
    args.npz_path, args.role_manifest = map(str, resolve_object_inputs(
        object_name=args.object, dataset_root=args.dataset_root, npz_path=None, role_manifest_path=args.role_manifest))
    return args


def heldout_inputs(args):
    """Registered contract, role views and arrays; no response row is exposed before these checks."""
    from rift.radar_fields_dataset import from_collection_arrays, validate_power_stats_acquisition
    from rift.rift_dataset import (evaluation_role_indices, load_object_contract, object_identity,
                                   validate_checkpoint_object)
    public, contract = load_object_contract(args.npz_path, args.role_manifest, response_roles=(args.role,),
                                            allow_reserved_test=args.allow_reserved_test)
    validate_checkpoint_object(object_identity(args.object), contract)
    stats = json.loads(Path(args.stats).read_text())
    validate_checkpoint_object(stats, contract)
    arrays = from_collection_arrays(public, contract)
    validate_power_stats_acquisition(stats, arrays.acquisition_identity)
    selected = evaluation_role_indices(contract, args.role, allow_reserved_test=args.allow_reserved_test)
    selected = selected[:args.max_views] if args.max_views else selected
    return contract, stats, arrays, selected


def constant_floors(cache, peak_power, dynamic_range_db):
    """Scores of the best constant predictors on the scored bins, from the per-view cache (method independent).

    A zero prediction scores exactly 1 in both domains (zero power clamps to normalized 0).
    ``best_constant``: one level for every scored bin; ``best_per_view_constant``: one level per
    view. Both are fitted to the scored role's own targets, so they lower-bound any constant
    predictor (a TRAIN-mean level included). The cached profile is the channel mean; the floors
    are exact only when it reproduces the stored target energy (one Tx/Rx pair), else ``None``.
    Linear targets are rebuilt from the cached intensity; bins below the -60 dB clamp count as 0.
    """
    done = np.isfinite(cache['range_power_rel_mse'])
    if not done.any():
        return None
    t = np.asarray(cache['target_profile'][done], dtype=np.float64)
    mask = np.asarray(cache['roi_mask'][done], dtype=bool)
    square, count = np.where(mask, t * t, 0.0).sum(1), mask.sum(1)
    if not (np.array_equal(count, cache['count_db'][done])
            and np.allclose(square, cache['target_sq_db'][done], rtol=1e-4, atol=0)):
        return None
    linear = np.where(t > 0, peak_power * 10.0 ** (dynamic_range_db * (t - 1.0) / 10.0), 0.0)

    def floors(values):
        s, q = np.where(mask, values, 0.0).sum(1), np.where(mask, values * values, 0.0).sum(1)
        energy = q.sum()
        return dict(best_constant=float((energy - s.sum() ** 2 / count.sum()) / energy),
                    best_per_view_constant=float((q - s * s / np.maximum(count, 1)).sum() / energy))
    return dict(zero_prediction=dict(normalized=1.0, linear=1.0), normalized=floors(t), linear=floors(linear),
                bins=int(count.sum()), views=int(done.sum()),
                definition='oracle constants fitted to the scored targets; lower bounds on any constant predictor')


@torch.no_grad()      # not inference_mode: GeRaF's SDF normals use autograd inside enable_grad
def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    contract, stats, arrays, selected = heldout_inputs(args)
    adapter = ADAPTERS[args.method](args.checkpoint, contract, arrays, device, args)
    arrays = restrict_radar_fields_response_views(arrays, selected)
    if hasattr(adapter, 'bind'):
        adapter.bind(arrays)          # adapters that read their own inputs use the role-restricted arrays
    # The stats' TRAIN IDs must equal the registered TRAIN role (method independent).
    peak_power, dynamic_range_db = validate_normalization_stats(
        stats, {'sealed_npz_protocol_contract': contract}, num_views=arrays.num_views)
    extent = original.RF_GRID_EXTENT_M
    ranges = range_bin_centers(arrays.metadata, device=device, dtype=torch.float32)

    out_dir = Path(args.out_dir)
    cache_path, summary_path = out_dir/f'{args.label}_per_view.npz', out_dir/f'{args.label}_metrics.json'
    provenance = dict(contract=contract, checkpoint=str(Path(args.checkpoint).resolve()), stats=stats,
                      method=adapter.identity)
    cache = original.load_or_initialize_cache(cache_path, selected, arrays.viewpoint_positions[selected],
                                              arrays.num_freq, args.resume, role=args.role, provenance=provenance)
    pending_slots = np.flatnonzero(~np.isfinite(cache['range_power_rel_mse']))
    slot_by_view = {int(v): int(s) for s, v in zip(pending_slots, selected[pending_slots])}
    started = time.perf_counter()

    def summary():
        result = original.aggregate(cache)
        if adapter.kind != 'complex':
            result['coherent_complex_rel_mse'] = None          # no coherent prediction; never a pooled zero
        role_total = len(contract['role_ids']['reserved_test' if args.role == 'test' else args.role])
        result.update(schema=SCHEMA, label=args.label, method=adapter.identity, checkpoint=os.path.abspath(args.checkpoint),
                      object=args.object, dataset_identity=contract['dataset_identity'], selected_role=args.role,
                      reserved_test_accessed=args.role == 'test', num_selected=int(len(selected)),
                      status=('complete' if not args.max_views and result['views_complete'] == result['views_total'] == role_total
                              else 'incomplete_or_smoke'),
                      ordered_source_view_ids=selected.tolist(), stats_path=os.path.abspath(args.stats),
                      peak_power=peak_power, dynamic_range_db=dynamic_range_db, range_margin_m=original.RF_RANGE_MARGIN_M,
                      prediction_kind=adapter.kind, definitions='scripts/eval_b787_range_power.py (RIFT-dataset evaluator)',
                      device=str(device), elapsed_seconds_this_invocation=time.perf_counter() - started,
                      reference_floors=constant_floors(cache, peak_power, dynamic_range_db))
        if hasattr(adapter, 'extra_summary'):
            result.update(adapter.extra_summary())
        return result

    def write():
        original.save_cache(cache_path, cache)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary(), indent=2, sort_keys=True) + '\n')

    for number, (view_index, response_view) in enumerate(arrays.iter_response_views(selected[pending_slots]), start=1):
        view_index, view_started = int(view_index), time.perf_counter()
        slot = slot_by_view[view_index]
        viewpoint_np = arrays.viewpoint_positions[view_index]
        rx_pos = torch.as_tensor(arrays.rx_pos[view_index], dtype=torch.float32, device=device)
        tx_pos = torch.as_tensor(arrays.tx_pos[view_index], dtype=torch.float32, device=device)
        prediction = adapter.predict(view_index, slot, rx_pos, tx_pos, viewpoint_np)

        # ---- the original evaluator's per-view metric block (eval_b787_range_power.main) ----
        target_power = response_view_to_range_power(response_view, device=device)
        target_intensity = normalize_power_db(target_power, peak_power, dynamic_range_db)
        viewpoint = torch.as_tensor(viewpoint_np, dtype=torch.float32, device=device)
        roi = scene_range_mask(ranges, viewpoint, extent, margin=0.05)
        if adapter.kind == 'complex':
            pred_flat = prediction.permute(2, 1, 0).reshape(-1, arrays.num_freq)
            measured_np = response_view.mean(axis=2).reshape(-1, arrays.num_freq)
            measured = torch.as_tensor(measured_np, dtype=torch.complex128, device=device)
            pred_power = torch.fft.ifft(pred_flat, dim=-1).abs().square()
            pred_intensity = normalize_power_db(pred_power, peak_power, dynamic_range_db)
        elif adapter.kind == 'intensity':
            # Power-only prediction in the normalized dB domain over the method's own ROI.
            if not torch.equal(prediction['roi'].to(roi.device), roi):
                raise ValueError(f'view {view_index}: the method ROI differs from the evaluator ROI')
            pred_intensity = prediction['intensity'].to(target_intensity.dtype)
            pred_power = peak_power * torch.pow(10.0, dynamic_range_db * (pred_intensity - 1.0) / 10.0)
        else:
            # Range-power profile known only on some bins (RadarSplat's crop): score ROI bins inside it.
            pred_power = prediction['power'].to(target_power.dtype)
            pred_intensity = normalize_power_db(pred_power, peak_power, dynamic_range_db)
            roi = roi & prediction['mask'].to(roi.device)
            if not roi.any():
                raise ValueError(f'view {view_index}: no ROI bin inside the method crop')
        pred_roi, target_roi = pred_intensity[:, roi], target_intensity[:, roi]
        pred_linear_roi, target_linear_roi = pred_power[:, roi], target_power[:, roi]
        sq_db = float((pred_roi - target_roi).square().sum())
        target_sq_db = float(target_roi.square().sum())
        sq_linear = float((pred_linear_roi - target_linear_roi).square().sum())
        target_sq_linear = float(target_linear_roi.square().sum())
        cache['target_signal'][slot] = float(target_roi.mean())
        cache['pred_signal'][slot] = float(pred_roi.mean())
        cache['range_power_rel_mse'][slot] = sq_db / max(target_sq_db, 1.0e-30)
        cache['linear_power_rel_mse'][slot] = sq_linear / max(target_sq_linear, 1.0e-30)
        cache['sq_error_db'][slot] = sq_db
        cache['target_sq_db'][slot] = target_sq_db
        cache['count_db'][slot] = int(pred_roi.numel())
        cache['sq_error_linear'][slot] = sq_linear
        cache['target_sq_linear'][slot] = target_sq_linear
        if adapter.kind == 'complex':
            sq_complex = float((pred_flat - measured).abs().square().sum())
            target_sq_complex = float(measured.abs().square().sum())
            cache['coherent_rel_mse'][slot] = sq_complex / max(target_sq_complex, 1.0e-30)
            cache['sq_error_complex'][slot] = sq_complex
            cache['target_sq_complex'][slot] = target_sq_complex
        cache['target_profile'][slot] = target_intensity.mean(dim=0).float().cpu().numpy()
        cache['pred_profile'][slot] = pred_intensity.mean(dim=0).float().cpu().numpy()
        cache['roi_mask'][slot] = roi.cpu().numpy()
        # ---- end of the original block ----

        done = int(np.isfinite(cache['range_power_rel_mse']).sum())
        coherent = cache['coherent_rel_mse'][slot]
        print(f"[{done:4d}/{len(selected)}] view {view_index:5d}  power={100 * cache['range_power_rel_mse'][slot]:8.3f}%  "
              f"complex={'n/a' if adapter.kind != 'complex' else f'{100 * coherent:7.3f}%'}  "
              f"{time.perf_counter() - view_started:6.2f}s", flush=True)
        if number % args.save_every == 0:
            write()
    write()
    print(json.dumps({k: v for k, v in summary().items() if k != 'ordered_source_view_ids'}, indent=2, sort_keys=True))
    print(f'cache: {cache_path}\nsummary: {summary_path}', flush=True)


if __name__ == '__main__':
    main()
