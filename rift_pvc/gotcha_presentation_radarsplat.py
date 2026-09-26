"""RadarSplat on GOTCHA, presented as relative intensity and K-calibrated RCS.

Relative: RadarSplat's own rendered image intensity, attributed to the G48
cell containing each Gaussian's mean (each Gaussian's share of the summed
rendered image, see below), mean over the TRAIN looks; the method's own units
(the recipe's intensity domain). The export's G48 ``support`` proxy (the
occupancy union sampled at cell centres) is not used as the relative map: the
release's Gaussians are ~2.5 cm (0.25 source units) against 21 cm cells, so
it samples between them and is empty or speckled; its statistics are kept in
the provenance.

Physical (see ``rift_pvc.gotcha_presentation_power``): for each look the
method's own renderer (``ReleasedRenderer`` on the fork's rasterizer, through the
PVC torch mirrors on CPU) draws the Gaussians alone -- the per-view multipath
background from training images is not part of the reconstruction and is
left out. The recipe's intensity mapping is undone per output pixel:
``linear_train_peak_v1`` (and a recipe without the key) P = peak * I;
``log_train_peak_60db_v1`` P = peak * 10^(60 (I - 1) / 10) for I > 0, else 0 (at
or below the -60 dB floor); the empty-scene render (the fork clamps every raster
pixel to >= 1e-6) is subtracted. Each pixel's power above the empty scene, as
actually rendered (clamps included), is attributed to the Gaussians in
proportion to their pre-clamp contributions to that pixel. Before its clamps
the rendered image is linear in each Gaussian's rasterized weight w_g =
clamp(o_g + n_g, 1e-6, 1) * min(SH_g(dir) + .5, 1), so the contributions are
gradients, with respect to a unit scale on every w_g, of a render whose power
raster is divided by a constant kappa >= its maximum (both inserted around the
fork's power rasterization call for this readout only; kappa = 1 and the
image unchanged when nothing saturates). Filters, azimuth crop and output
sampling are included; totals are conserved exactly. Energy matching to a 1 m^2 point target at the region centre formed
by the method's own ``sector_power`` (the cache's coherent sector
matched-filter power, summed over elevation samples) gives each Gaussian's
RCS; looks are averaged and each Gaussian's RCS is binned into the G48 cell
containing its mean. Not attributed: the per-Gaussian 0.999 alpha clamp (it
bounds a contribution, it does not saturate a sum) and pixels whose pre-clamp
filtered value is not positive; the per-look ``attributed_power`` /
``rendered_power`` diagnostics show any loss.
"""
from __future__ import annotations

import contextlib
import importlib
import math
from pathlib import Path

import numpy as np
import torch

from rift_pvc.gotcha_presentation import PRESENTATION_GRID, MethodPresentation
from rift_pvc.gotcha_presentation_power import (bin_points, dataset_from_contract, invert_db, look_metadata,
                                                point_target_observation, select_looks)

DEFAULT_LOOKS = 16
LINEAR = 'linear_train_peak_v1'
LOG = 'log_train_peak_60db_v1'
LOG_SPAN_DB = 60.0


def intensity_mapping(identity):
    mapping = identity.get('intensity_mapping', LINEAR)
    if mapping not in (LINEAR, LOG):
        raise ValueError(f'unknown RadarSplat intensity mapping {mapping!r}')
    return mapping


def native_power(image, peak, mapping):
    image = torch.as_tensor(image, dtype=torch.float64)
    if mapping == LINEAR:
        return float(peak) * image
    return invert_db(image, peak, LOG_SPAN_DB, 0.0)


def load_renderer(units_per_m):
    """The released renderer on the fork's rasterizer through the PVC torch mirrors, on CPU."""
    from rift.radarsplat_release import ReleasedRenderer
    from rift_pvc.radarsplat_xpu_backend import load_xpu_reference
    rendering, _ = load_xpu_reference(device='cpu')
    return ReleasedRenderer(rendering, units_per_m, local_azimuth=True)


@contextlib.contextmanager
def per_gaussian_scale(scale, *, unclamp=False):
    """Multiply each Gaussian's rasterized power weight by ``scale`` (ones: unchanged) for one render.

    ``_radar_rasterization`` imports ``_rasterize_to_radar_pixels`` at call time
    and calls it first for the power product (``opacities_w_reflectance``); the
    other products (occupancy, noise, ...) are untouched. With ``unclamp`` the
    power raster is divided by a constant kappa >= its maximum (recorded in the
    yielded state), so neither the rasterizer's [1e-6, 1] clamp nor the
    renderer's [0, 1] clamp binds and the image stays linear in every weight.
    The module attribute is restored on exit; no source file changes.
    """
    radar = importlib.import_module('gsplat.cuda._torch_impl_radar')
    original = radar._rasterize_to_radar_pixels
    state = dict(calls=0, kappa=1.0)

    def scaled(means2d, conics, opacities, *args, **kwargs):
        state['calls'] += 1
        if state['calls'] != 1:
            return original(means2d, conics, opacities, *args, **kwargs)
        out = original(means2d, conics, opacities * scale.to(opacities)[None, :], *args, **kwargs)
        if unclamp:
            state['kappa'] = max(1.0, 1.01 * float(out.detach().max()))
            out = out / state['kappa']
        return out
    radar._rasterize_to_radar_pixels = scaled
    try:
        yield state
    finally:
        radar._rasterize_to_radar_pixels = original


def _empty_splats(splats):
    """One Gaussian whose weight falls under the rasterizer's 1/255 cutoff: the empty scene."""
    return dict(means=torch.zeros(1, 3), quats=torch.tensor([[1., 0., 0., 0.]]),
                scales=torch.full((1, 3), math.log(.25)), opacities=torch.full((1,), -30.),
                noise_probs=torch.full((1,), -30.), sh0=torch.zeros(1, *splats['sh0'].shape[1:]),
                shN=torch.zeros(1, *splats['shN'].shape[1:]))


def calibration_arrays(row):
    arrays = {k: np.asarray(v) for k, v in row.items() if k != 'renderer_grid'}
    arrays['native_pulse_count'] = int(row['native_pulse_count'])
    return arrays


class _PointTargetSector:
    """Stand-in dataset whose observations are a 1 m^2 point at the region centre.

    Only for ``sector_power``: positions and frequencies are the look's own
    metadata; the responses are synthetic (K's definition), never read.
    """
    def __init__(self, region, metadata):
        self.region, self.metadata = region, metadata

    def observations(self, pass_id, sector, polarization):
        for native, local in zip(self.metadata['native'], self.metadata['local']):
            yield point_target_observation(native, local, self.metadata['frequencies'])[0]


def look_rcs(splats, renderer, calibration_row, peak, mapping, active_degree, *, point_energy=None):
    """Per-Gaussian rendered intensity and (with ``point_energy``) RCS for one look.

    Returns (intensity share [N] in image units, RCS [N] in m^2 or None,
    diagnostics). ``point_energy`` is e1 for this look (image energy of a 1 m^2
    point target through the method's own target formation). Each pixel's
    actual rendered value (clamps included) is split among the Gaussians in
    proportion to their pre-clamp contributions, obtained as gradients of an
    unclamped render (``per_gaussian_scale(unclamp=True)``); totals are
    conserved exactly, saturated pixels included.
    """
    from rift.radarsplat_b7873200 import RadarSplatGrid
    grid = RadarSplatGrid(**calibration_row['renderer_grid'])
    pose = torch.as_tensor(np.asarray(calibration_row['sensor_to_world']), dtype=torch.float32)
    multipath = torch.zeros(grid.output_azimuth_bins, grid.num_range_bins)
    splats = {k: v.detach() for k, v in splats.items()}
    with torch.no_grad():
        empty = renderer(_empty_splats(splats), pose, grid, active_degree, multipath)[0].double()
        image = renderer(splats, pose, grid, active_degree, multipath)[0].double()
    drawn = image - empty
    above = native_power(image, peak, mapping) - native_power(empty, peak, mapping)
    scale = torch.ones(len(splats['means']), dtype=torch.float32, requires_grad=True)
    with torch.enable_grad(), per_gaussian_scale(scale, unclamp=True) as state:
        linear = renderer(splats, pose, grid, active_degree, multipath)[0]
        if state['calls'] < 1:
            raise RuntimeError('RadarSplat power rasterization was not reached')
        kappa = state['kappa']
        pre = kappa * (linear.detach().double() - empty)       # pre-clamp Gaussian intensity per pixel
        def share(values):
            weight = torch.where(pre > 0, values / pre.clamp_min(1e-300), torch.zeros_like(pre))
            grad, = torch.autograd.grad((weight.to(linear) * linear).sum(), scale, retain_graph=True)
            return kappa * grad.double()
        intensity, energy = share(drawn), share(above)
    info = dict(rendered_intensity=float(drawn.sum()), attributed_intensity=float(intensity.sum()),
                rendered_power=float(above.sum()), attributed_power=float(energy.sum()), kappa=kappa)
    return intensity, None if point_energy is None else energy / point_energy, info


def point_energy(dataset, metadata, calibration_row, polarization, config):
    """e1: image energy of a 1 m^2 point at the region centre through the method's ``sector_power``."""
    from rift.radarsplat_gotcha import sector_power
    power = sector_power(_PointTargetSector(dataset.region, metadata), *metadata['view'], polarization,
                         calibration_arrays(calibration_row), dict(config, point_chunk=max(config['point_chunk'], 4096)),
                         device='cpu')
    return float(np.asarray(power, dtype=np.float64).sum())


def radarsplat_presentation(checkpoint_path, *, shard_root, polarization='hh', grid=PRESENTATION_GRID,
                            looks=DEFAULT_LOOKS):
    """RadarSplat GOTCHA head checkpoint -> relative occupancy union and K-calibrated RCS per G48 cell."""
    from scripts.eval_b787_baseline_geometry_provisional import rasterize_gaussian_occupancy_union
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    identity = checkpoint['identity']
    sealed = identity['target_recipe']['sealed_protocol_identity']
    if sealed['polarization'] != polarization:
        raise ValueError(f"checkpoint is the {sealed['polarization']} head, not {polarization}")
    contract, adapter = sealed['dataset_contract'], identity['adapter']
    units, extent = float(adapter['model_units_per_m']), float(adapter['half_extent_m'])
    splats = {k: v.detach().float() for k, v in checkpoint['splats'].items()}
    means_m = splats['means'].double().numpy() / units
    support, support_stats = rasterize_gaussian_occupancy_union(
        means_m, (splats['scales'].exp() / units).numpy(), torch.nn.functional.normalize(splats['quats'], dim=-1).numpy(),
        splats['opacities'].sigmoid().numpy(), granularity=grid, extent=extent, sigma_radius=3.)
    step = int(checkpoint['step'])
    mapping = intensity_mapping(identity)
    peak = float(identity['train_peak_power'])
    dataset = dataset_from_contract(contract, shard_root)
    record = checkpoint['acquisition_record']
    train = {tuple(v) for v in dataset.viewpoints('train')}
    by_view = {tuple(key): row for key, row in zip(record['view_keys'], record['calibration']) if tuple(key) in train}
    selected = select_looks(list(by_view), looks)
    renderer = load_renderer(units)
    active_degree = max(0, min((step - 1) // 200, 5))
    config = identity['target_recipe']['config']
    physical = polarization == 'hh'
    intensity = torch.zeros(len(means_m), dtype=torch.float64)
    per_gaussian = torch.zeros(len(means_m), dtype=torch.float64)
    diagnostics = []
    for view in selected:
        e1 = None
        if physical:
            e1 = point_energy(dataset, look_metadata(dataset, view, polarization), by_view[view], polarization, config)
        shares, values, info = look_rcs(splats, renderer, by_view[view], peak, mapping, active_degree, point_energy=e1)
        intensity += shares
        if physical:
            per_gaussian += values
        diagnostics.append(dict(view=list(view), point_energy=e1, **info))
    relative, _ = bin_points(means_m, intensity / len(selected), extent, grid)
    source = dict(checkpoint=str(Path(checkpoint_path).resolve()), step=step, validation=checkpoint.get('validation'),
                  half_extent_m=extent, model_units_per_m=units, gaussians=int(len(means_m)),
                  renderer='fork _radar_rasterization on the PVC torch mirrors (fork_torch_mirror_xpu_v1), CPU',
                  intensity_mapping=mapping, train_peak_power=peak, looks=len(selected), train_looks=len(train),
                  active_sh_degree=active_degree, support_proxy=dict(support_stats,
                      occupied_cells=int((np.asarray(support) > 0).sum()), maximum=float(np.max(support))),
                  looks_detail=diagnostics[:4])
    rcs, physical_quantity = None, None
    if physical:
        rcs, outside = bin_points(means_m, per_gaussian / len(selected), extent, grid)
        physical_quantity = ('RCS per G48 cell (m^2): each Gaussian\'s share of the rendered native image power '
                             '(recipe mapping undone per pixel, empty scene subtracted), energy-matched to a point '
                             'target, summed per cell; mean over TRAIN looks; multipath background excluded')
        ratios = [d['attributed_power'] / max(d['rendered_power'], 1e-300) for d in diagnostics]
        source.update(rcs_outside_cube_m2=outside, attributed_over_rendered=[min(ratios), max(ratios)],
                      conversion=('E_g = d/ds_g sum_pixels ratio * image at s = 1 (s_g scales Gaussian g\'s rasterized '
                                  'weight; ratio = (P(I) - P(I_empty)) / (I - I_empty), P(I) = peak I (linear) or '
                                  'peak 10^(60 (I-1)/10) (log, I > 0)); sigma_g = E_g / e1, '
                                  'e1 = sum_pixels sector_power(point of K/(2R)^2) (= K^2/(2R)^4 per coherent '
                                  'elevation sample at its pixel)'))
    return MethodPresentation('radarsplat', polarization, relative,
                              'rendered image intensity (recipe units) attributed to the G48 cell of each Gaussian '
                              'mean, mean over TRAIN looks; multipath background excluded', rcs, physical_quantity, source)
