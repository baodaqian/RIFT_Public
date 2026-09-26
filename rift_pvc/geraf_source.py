"""GeRaF v1 configuration of the released GeRaFStage1, PVC (Intel XPU) twin.

Copy of ``rift/geraf_source.py`` for the PVC port (Package B of
``RIFT_PVC_Adaptation.md``). The recipe, defaults, validation, network
construction and sampling are identical; the only adaptations are

* ``GeRaFStage1`` comes from ``rift_pvc.vendor.geraf_sens.rf_rendering``, whose
  three float16 autocast regions select the active backend, and
* ``predict_native`` uses ``rift_pvc.geraf_autocast.autocast_fp16`` instead of
  the literal ``torch.amp.autocast('cuda', ...)``.

Every other vendored module is imported unchanged from ``rift/``. Network
initialization uses seeded CPU torch RNG; ray sampling uses numpy
(``fixed_numpy_seed`` around ``UniformRaySampler``/``TargetRaySampler``).
Neither draws from the device RNG, so CUDA and XPU use the same host streams.
"""
from __future__ import annotations

import contextlib
import math
import numpy as np
import torch
from rift.vendor.geraf_sens.sdf_network import SDFNetwork
from rift.vendor.geraf_sens.power_network import ReflectivePowerNetwork, SingleVarianceNetwork
from rift_pvc.vendor.geraf_sens.rf_rendering import GeRaFStage1, _compute_transmission
from rift.vendor.geraf_sens.sample import UniformRaySampler, TargetRaySampler, DynamicLossMask
from rift.vendor.geraf_sens.loading import InterpolateMFAtTargets
from rift_pvc.geraf_autocast import autocast_fp16

COMMIT = '38266cb6e194e2f3dcbead614069a7281ffd21a5'
SCHEMA = 'rift_geraf_source_v1'
# User-selected 48^3 target lattice. The released example uses 601^3;
# coarsening is an explicit acquisition recipe change, not source equivalence.
DEFAULTS = dict(steps=50000, n_aperture=32, n_samples=32, n_samples_tgt=64,
                mf_grid=48, sdf_hidden_dim=256, sdf_layers=8, sdf_levels=10,
                sdf_skip=4, variance_init=.3, light_power=0.,
                grad_regression=.1, anneal_end=50000, freeze_inv_s_step=10000,
                model_step_policy='released_runner_zero', bank_size=2,
                mask_current=.04, mask_accumulated=.15,
                sdf_lr=1e-4, other_lr=1e-3, eta_min=5e-4,
                weight_decay=0., gradient_clip=35.,
                trans_power=1., point_chunk=1024, pair_chunk=16,
                checkpoint_every=100, validation_every=1000, log_every=10,
                seed=42, receiver_geometry='float64_intersection', light_power_start='configured')
# Native GOTCHA antennas sit about 2000 scene radii away. The released
# intersect_sphere then forms b^2 - 4c from terms of about 1.6e7, which float32
# cannot resolve (false misses, displaced hits; docs/GOTCHA_FORWARD_MODEL_ALIGNMENT.md
# section 5, G1). 'float64_intersection' runs that released routine on float64
# receiver geometry and casts only its intersection points to the network dtype.
# 'float32_release_cast' is the earlier cast. Recipes written before this key
# existed mean it, and it is omitted from recipes so they stay byte-identical.
RECEIVER_GEOMETRIES = ('float64_intersection', 'float32_release_cast')
LEGACY_RECEIVER_GEOMETRY = 'float32_release_cast'
# Starting value of the released trainable transmit amplitude light_power. The
# release configures it per dataset (its example starts at 7.5644 on its own units)
# rather than learning it up from 0. 'configured' starts at the recipe's light_power
# (class default 0); recipes written before this key mean it, and it is omitted so they
# stay byte-identical. 'train_warm_start_16_views' is for GOTCHA: at 10 km GeRaF's initial
# render sits ~1e10 below the native targets and AdamW's eps throttles light_power
# (~2e-10/step), so the start is the closed-form least-squares scale between the
# initial render and the TRAIN MF targets of the first 16 training views (the render
# is linear in exp(light_power)). Training stays in native units (trans_power=1).
LIGHT_POWER_STARTS = ('configured', 'train_warm_start_16_views')
LEGACY_LIGHT_POWER_START = 'configured'
WARM_START_VIEWS = 16


def recipe_from_config(config, extent):
    if not isinstance(config, dict) or set(config) - set(DEFAULTS):
        raise ValueError(f'Unknown source GeRaF settings: {set(config) - set(DEFAULTS)}')
    recipe = {**DEFAULTS, **config}
    for key, default in DEFAULTS.items():
        value = recipe[key]
        if type(default) is int:
            if type(value) is not int or value < 0:
                raise ValueError(f'{key} must be a nonnegative integer')
        elif isinstance(default, float):
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f'{key} must be finite')
    for key in ('steps', 'n_aperture', 'point_chunk', 'pair_chunk', 'checkpoint_every',
                'validation_every', 'log_every', 'sdf_layers', 'sdf_hidden_dim', 'bank_size'):
        if recipe[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if min(recipe[k] for k in ('mf_grid', 'n_samples', 'n_samples_tgt')) < 2:
        raise ValueError('Source interpolation requires sizes >= 2')
    if not 0 < recipe['sdf_skip'] < recipe['sdf_layers'] or recipe['sdf_hidden_dim'] <= 3 + 6 * recipe['sdf_levels']:
        raise ValueError('Invalid source SDF skip dimensions')
    if any(recipe[k] <= 0 for k in ('sdf_lr', 'other_lr', 'trans_power', 'gradient_clip')):
        raise ValueError('Learning rates, transmit-unit divisor and gradient clip must be positive')
    if any(not 0 <= recipe[k] <= 1 for k in ('mask_current', 'mask_accumulated')):
        raise ValueError('Mask fractions must be in [0,1]')
    if recipe['weight_decay'] < 0 or recipe['eta_min'] < 0 or recipe['grad_regression'] < 0:
        raise ValueError('Optimizer and regularization values must be nonnegative')
    if recipe['model_step_policy'] not in ('released_runner_zero', 'advance'):
        raise ValueError('Unknown model-step policy')
    if recipe['seed'] != 42 or not math.isfinite(extent) or extent <= 0:
        raise ValueError('Registered seed is 42; scene extent must be positive')
    if recipe['receiver_geometry'] not in RECEIVER_GEOMETRIES:
        raise ValueError(f'receiver_geometry must be one of {RECEIVER_GEOMETRIES}')
    if recipe['receiver_geometry'] == LEGACY_RECEIVER_GEOMETRY:
        del recipe['receiver_geometry']
    if recipe['light_power_start'] not in LIGHT_POWER_STARTS:
        raise ValueError(f'light_power_start must be one of {LIGHT_POWER_STARTS}')
    if recipe['light_power_start'] == LEGACY_LIGHT_POWER_START:
        del recipe['light_power_start']
    return dict(**recipe, **({'antenna_bank_adaptation': 'single_nonempty_bank_v1'}
                            if recipe['bank_size'] == 1 else {}), schema=SCHEMA, upstream_commit=COMMIT, extent_m=float(extent),
                implementation='source_v1', loss='source_l2_mf_magnitude',
                reflectivity='source_ReflectivePowerNetwork_feats_dim_0',
                accumulated_mf='sum_train_measured_magnitudes_common_world_grid',
                target_storage='lazy_trilinear_accumulated_only_v1',
                phase='native_exact', trace_normalization='source_total_target_point_count',
                mf_normalization='source_antenna_mean_frequency_sum',
                validation='fresh_all_channels_unmasked_fixed_seed_rays_source_inbounds',
                source_fallbacks=['grad_regression', 'variance_init', 'anneal_end',
                    'freeze_inv_s_step', 'model_step_policy', 'bank_size',
                    'mask_current', 'mask_accumulated', 'eta_min', 'weight_decay',
                    'gradient_clip', 'light_power'],
                unrecovered=['historical_v1_launcher', 'historical_v1_checkpoint',
                    'accumulated_mf_preprocessing', 'native_acquisition_unit_calibration',
                    'native_nvs_evaluation_recipe'])


def recipe_for_data(config, data):
    config = dict(config)
    acquisition = data.contract.get('experiment_contract', {}).get('antenna_selection')
    if acquisition and acquisition['num_tx'] * acquisition['num_rx'] == 1:
        if config.get('bank_size', 1) != 1:
            raise ValueError('Single-pair collection GeRaF requires bank_size=1')
        config['bank_size'] = 1
    return recipe_from_config(config, data.extent)


class NativeGeRaFStage1(GeRaFStage1):
    """Released GeRaFStage1; its sphere intersection runs on float64 receiver geometry."""
    receiver_geometry = LEGACY_RECEIVER_GEOMETRY

    def intersect_sphere(self, r_pos_norm, norm_pts):
        if self.receiver_geometry == LEGACY_RECEIVER_GEOMETRY:
            return super().intersect_sphere(r_pos_norm.float(), norm_pts)
        valid, points = super().intersect_sphere(r_pos_norm.double(), norm_pts.double())
        return valid, points.to(norm_pts.dtype)


class SingleBankGeRaFStage1(NativeGeRaFStage1):
    """Disclosed one-channel extension; the attributed release stays unchanged."""
    def _gather_unselected_adc(self, data_name, ant_indices_groups, current_chunk_id,
                               t_pos, r_pos, sim_real, sim_imag):
        if self.cfg['bank_size'] != 1 or current_chunk_id != 0:
            raise ValueError('Single-bank adaptation requires its modulo-one pointer')
        self.ant_real_list[data_name][0] = sim_real.detach().cpu()
        self.ant_imag_list[data_name][0] = sim_imag.detach().cpu()
        return t_pos[:0], r_pos[:0], sim_real[:0].detach(), sim_imag[:0].detach()


def build_model(recipe, device='cpu'):
    sdf = SDFNetwork(d_in=3, d_out=1, d_hidden=recipe['sdf_hidden_dim'],
                     n_layers=recipe['sdf_layers'], skip_in=(recipe['sdf_skip'],),
                     multires=recipe['sdf_levels'], bias=.5, scale=1.,
                     geometric_init=True, weight_norm=True)
    reflectivity = ReflectivePowerNetwork(
        dict(refelective_act='sigmoid', refelective_exp_max=None,
             light_power=float(recipe['light_power'])), feats_dim=0)
    deviation = SingleVarianceNetwork(float(recipe['variance_init']), activation='exp')
    model_type = SingleBankGeRaFStage1 if recipe['bank_size'] == 1 else NativeGeRaFStage1
    model = model_type(sdf, deviation, reflectivity, rt_backend='native', mf_backend='native',
                      with_grad_regression=recipe['grad_regression'], loss_mode='l2',
                      sample_cfg=dict(bank_size=recipe['bank_size'], anneal_end=recipe['anneal_end'],
                                      freeze_inv_s_step=recipe['freeze_inv_s_step']),
                      radar_cfg={}).to(device)
    model.receiver_geometry = recipe.get('receiver_geometry', LEGACY_RECEIVER_GEOMETRY)
    return model


def source_step(recipe, optimizer_steps):
    # The released tools/train.py never calls update_step / its named StepHook.
    # Advancing it is an explicit alternate recipe, not a silent bug fix.
    return 0 if recipe['model_step_policy'] == 'released_runner_zero' else optimizer_steps


@contextlib.contextmanager
def fixed_numpy_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def sample_frame(acquisition, recipe, name, mf_volume=None, accumulated=None):
    """Adapt only poses/units; execute the authors' sampling and mask transforms."""
    radius = recipe['extent_m']
    center = ((acquisition.tx + acquisition.rx) * .5).mean(0).cpu().numpy()
    direction = -center / np.linalg.norm(center)
    # These sources have no robot rotation.npy. Use the calibrated array axis
    # where available and a deterministic orthogonal basis otherwise.
    rx = acquisition.rx.detach().cpu().numpy()
    candidates = rx - rx[0]
    candidates -= (candidates @ direction)[:, None] * direction
    axis = candidates[np.argmax(np.linalg.norm(candidates, axis=-1))]
    if np.linalg.norm(axis) < 1e-10:
        basis = np.eye(3)[np.argmin(np.abs(direction))]
        axis = basis - np.dot(basis, direction) * direction
    axis /= np.linalg.norm(axis)
    second = np.cross(direction, axis)
    transform = np.diag([radius, radius, radius, 1.])
    frame = dict(data_name=name, sample_cfg=dict(N_aperture=recipe['n_aperture'],
        N_samples=recipe['n_samples'], N_samples_tgt=recipe['n_samples_tgt']),
        ant_coord=(direction, axis, second), ray_d=direction[None], normglb2glb=transform)
    frame = TargetRaySampler()(UniformRaySampler()(frame))
    if mf_volume is not None:
        if callable(getattr(mf_volume, 'sample', None)):
            # Query the same FP32 lattice interpolation, without constructing
            # or retaining an entire per-view MF cube. Source rays/masks remain.
            points = frame['tgt_sampled_poses_norm'].reshape(-1, 3)
            frame['mf_sampled_value'] = mf_volume.sample(points)
            if accumulated is mf_volume:
                frame['accmf_sampled_value'] = frame['mf_sampled_value']
            else:
                frame['accmf_image'] = accumulated
        else:
            frame.update(mf_image=mf_volume, accmf_image=accumulated)
        frame = InterpolateMFAtTargets()(frame)
        frame = DynamicLossMask(recipe['mask_current'], recipe['mask_accumulated'])(frame)
    else:
        frame['mf_sampled_value'] = np.zeros(frame['tgt_sampled_poses_norm'].shape[:2], dtype=np.float32)
        frame['loss_mask'] = np.ones(len(frame['ray_d']), dtype=bool)
    device = acquisition.tx.device
    packed = {}
    for key in ('sampled_poses_norm', 'tgt_sampled_poses_norm', 'z_vals', 'tgt_z_vals',
                'tgt_dists', 'ray_d', 'dists', 'mf_sampled_value', 'loss_mask',
                'sampled_poses_glb', 'tgt_sampled_poses_glb'):
        dtype = torch.bool if key == 'loss_mask' else torch.float64 if key.endswith('_glb') else torch.float32
        packed[key] = torch.as_tensor(frame[key], dtype=dtype, device=device)
    packed.update(data_name=name, t_pos=acquisition.tx, r_pos=acquisition.rx,
                  r_pos_norm=acquisition.rx / radius)   # float64; the model casts per receiver_geometry
    return packed


def predict_native(model, sample, acquisition, *, return_inbounds=False):
    """Fresh, unmasked NVS readout using the same source volume/tracer math.

    The public release exposes geometry prediction, not this acquisition's NVS
    evaluator. This readout does not update training banks or model parameters.
    """
    model.radar_cfg = {'native': acquisition}
    frame = model._extract_batch_inputs(sample)
    keep = torch.ones(len(sample['ray_d']), dtype=torch.bool, device=acquisition.tx.device)
    frame = model._apply_loss_mask(frame, keep)
    ids = torch.arange(len(acquisition.tx), device=acquisition.tx.device)
    with torch.no_grad():
        prev, current = model._calibrate_transmission(frame['sampled_poses_norm'], ids, frame['r_pos_norm'])
        with autocast_fp16():
            alpha, gradients, features, _, _ = model.compute_sdf_alpha(
                frame['sampled_poses_norm'], frame['dists'], frame['dirs'],
                model.get_anneal_val(model.step), model.step)
        trans = _compute_transmission(alpha) - prev + current
        sigmas = model.signal_network(frame['sampled_poses_norm'], features, trans) * alpha * trans
        normals, sigmas, inside = model._sample_tgt_normals_and_sigmas(
            gradients, sigmas, frame['z_vals'], frame['tgt_z_vals'], len(ids))
        response = acquisition.trace(normals.reshape(-1, 3), sigmas.reshape(len(ids), -1),
            frame['tgt_sampled_poses_glb'].reshape(-1, 3), acquisition.tx, acquisition.rx, inside.flatten())
        return (response, inside.flatten()) if return_inbounds else response
