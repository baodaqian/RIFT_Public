"""GOTCHA reconstructions presented two ways: relative intensity and K-calibrated RCS.

Presentation only. Nothing here trains, refits or rescales a model, and no
validation or test response is read: the look directions come from the shard
metadata members (x, y, z, sector_id) of TRAIN sectors, never from response.npy.

Every method's reconstruction is read out onto one common grid, the registered
region cube at the scene budget's G48 cells, and written twice with identical
file names:

- ``relative_intensity/``: the method's own reconstructed intensity, in its own
  arbitrary units, normalized to its brightest cell (0 dB). For RIFT this is the
  project's existing point-SH readout (active, unlocked sum |c|^2, CIC-deposited),
  the quantity every collection figure already shows.
- ``physical_K/``: radar cross section per cell in m^2 (shown in dBsm), averaged
  over the TRAIN look directions, through the dataset's system constant
  K = GOTCHA_HH_SYSTEM_CONSTANT (data / K = sqrt(RCS) / (Rt + Rr)^2; HH only).
  One colour scale is shared by all methods in a presentation.

K is the calibration array's measurement (``rift/gotcha_calibration.py``;
alignment doc section 13): median of 48 trihedral detections, per-pass medians
6950-8855, 27-inch cross-check +1.5 dB, so physical values carry about +-1.5 dB.
Cells add incoherently (sum of the scatterers' RCS in the cell).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from rift.gotcha_calibration import GOTCHA_HH_SYSTEM_CONSTANT
from rift.gotcha_dataset import Region
from rift.gotcha_training import RIFT_G_CONST, with_legacy_keys
from rift.spherical_harmonics import real_sh_basis

SCHEMA = 'gotcha_presentation_v1'
RELATIVE_DIR, PHYSICAL_DIR = 'relative_intensity', 'physical_K'
K = GOTCHA_HH_SYSTEM_CONSTANT
K_UNCERTAINTY_DB = 1.5
RELATIVE_FLOOR_DB = -40.0
PHYSICAL_SPAN_DB = 40.0
PRESENTATION_GRID = 48
PLANES = (('top', 2, 'x (m)', 'y (m)'), ('side', 1, 'x (m)', 'z (m)'), ('front', 0, 'y (m)', 'z (m)'))


@dataclass
class MethodPresentation:
    """One method's reconstruction on the presentation grid, in both unit systems."""
    method: str
    polarization: str
    relative: np.ndarray
    relative_quantity: str
    rcs_m2: np.ndarray | None
    physical_quantity: str | None
    source: dict = field(default_factory=dict)

    @property
    def name(self):
        return f'{self.method}_{self.polarization}'


def deposit_points(positions, values, extent, grid):
    """CIC deposit onto [grid]^3 over [-extent, extent]^3; conserves the total.

    Same rule as ``scripts/eval_scene_geometry.deposit_points`` (the collection
    figures' readout), including clamping neighbour indices at the faces.
    """
    positions = torch.as_tensor(positions, dtype=torch.float64)
    values = torch.as_tensor(values, dtype=torch.float64)
    pitch = 2.0 * extent / grid
    f = (positions + extent) / pitch - 0.5
    i0 = torch.floor(f).long()
    t = f - i0.double()
    volume = torch.zeros(grid ** 3, dtype=torch.float64)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                ix, iy, iz = ((i0[:, a] + d).clamp(0, grid - 1) for a, d in enumerate((dx, dy, dz)))
                w = ((t[:, 0] if dx else 1 - t[:, 0]) * (t[:, 1] if dy else 1 - t[:, 1])
                     * (t[:, 2] if dz else 1 - t[:, 2]))
                volume.index_add_(0, (ix * grid + iy) * grid + iz, w * values)
    return volume.view(grid, grid, grid).numpy()


def train_sectors(contract):
    """TRAIN (pass, sector) pairs of a saved GOTCHA dataset contract."""
    split = contract['split']
    if 'sector_ids_by_pass' in split:
        return [(int(p), int(s)) for p, roles in sorted(split['sector_ids_by_pass'].items(), key=lambda i: int(i[0]))
                for s in roles['train']]
    return [(int(p), int(s)) for p in contract['passes'] for s in split['sector_ids']['train']]


def train_look_geometry(contract, shard_root, polarization='hh'):
    """Per-TRAIN-sector mean antenna direction and range in the region's local frame.

    Reads only the metadata members of each shard; response.npy is never opened.
    """
    region = Region(**{k: (tuple(map(tuple, v)) if k == 'rotation_local_to_native' else
                           tuple(v) if k == 'translation_m' else v) for k, v in contract['region'].items()})
    wanted = {}
    for p, s in train_sectors(contract):
        wanted.setdefault(p, set()).add(s)
    directions, ranges, sectors = [], [], []
    for p in sorted(wanted):
        with np.load(Path(shard_root) / f'pass{p}_{polarization}.npz', allow_pickle=False) as shard:
            xyz = np.stack([shard[k] for k in ('x', 'y', 'z')], axis=1).astype(np.float64)
            sector_id = shard['sector_id'].astype(np.int64)
        for s in sorted(wanted[p]):
            rows = sector_id == s
            if not rows.any():
                raise ValueError(f'pass {p} sector {s} has no rows in its shard')
            local = region.to_local(xyz[rows])
            rng = np.linalg.norm(local, axis=1)
            mean = (local / rng[:, None]).mean(0)
            directions.append(mean / np.linalg.norm(mean))
            ranges.append(rng.mean())
            sectors.append((p, s))
    return dict(directions=np.asarray(directions), ranges_m=np.asarray(ranges), sectors=sectors,
                region=region)


def aperture_gram(directions, degree):
    """M = mean over look directions of Y(u) Y(u)^T, so mean |c.Y(u)|^2 = c^H M c.

    Directions map to (theta, phi) exactly as ``ChannelField.forward`` does.
    """
    d = torch.as_tensor(directions, dtype=torch.float64)
    theta, phi = torch.acos(d[:, 2].clamp(-1, 1)), torch.atan2(d[:, 1], d[:, 0])
    basis = real_sh_basis(theta, phi, degree)          # [n_basis, D]
    return basis @ basis.T / basis.shape[1]


def _point_sh_state(checkpoint, polarization):
    prefix = f'{polarization}.field.'
    state = {k[len(prefix):]: v for k, v in checkpoint['model_state_dict'].items() if k.startswith(prefix)}
    if not state:
        raise ValueError(f'checkpoint has no {polarization} field')
    active = state['active_mask']
    positions = state['anchors'][active] + state['cell_half'][active] * torch.tanh(state['delta_raw'][active])
    if bool(state['support_bounds_enabled']):
        positions = torch.maximum(torch.minimum(positions, state['support_max']), state['support_min'])
    unlocked = state['basis_degree'][None, :] <= state['order'][active][:, None]
    coefficients = torch.complex(state['w_re'][active].double(), state['w_im'][active].double()) * unlocked
    degree = int(round(state['w_re'].shape[1] ** .5)) - 1
    return positions.double(), coefficients, degree


def _gain(checkpoint, polarization):
    state = checkpoint['model_state_dict']
    return complex(torch.polar(torch.exp(state[f'{polarization}.gain.log_mag'].double()),
                               state[f'{polarization}.gain.phase'].double()).item())


def rift_point_rcs(coefficients, degree, gain_magnitude, range_model, directions, ranges_m):
    """Per-scatterer RCS (m^2), averaged over ``directions``; see ``rift_presentation``."""
    gram = aperture_gram(directions, degree).to(coefficients.dtype)
    aperture = torch.einsum('kb,bc,kc->k', coefficients, gram, coefficients.conj()).real.clamp_min(0)
    if range_model == 'sum2':
        factor = (gain_magnitude * RIFT_G_CONST / K) ** 2
    elif range_model == 'unit':
        factor = (gain_magnitude / K) ** 2 * float(np.mean((2 * np.asarray(ranges_m)) ** 4))
    else:
        raise ValueError(f'unknown range model {range_model!r}')
    return factor * aperture


def rift_presentation(checkpoint_path, *, shard_root, polarization='hh', grid=PRESENTATION_GRID):
    """Adaptive RIFT (point-SH) checkpoint -> relative energy and K-calibrated RCS.

    The model predicts native data directly: y = D g sum_p rho_p(u) A(r_p) exp(...),
    with A = 1 / ((4 pi)^2 (2r)^2) for ``sum2`` and 1 for ``unit``. K's definition,
    y / K = sqrt(RCS) / (2r)^2, gives sqrt(RCS_p(u)) = g rho_p(u) / ((4 pi)^2 K) for
    ``sum2`` and g rho_p(u) (2r)^2 / K for ``unit`` (r = mean TRAIN range to the
    region centre). RCS is averaged over the TRAIN look directions.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    recipe = with_legacy_keys(checkpoint['recipe'], checkpoint['recipe']['method'])
    if recipe['method'] != 'rift':
        raise NotImplementedError(f"presentation of method {recipe['method']!r} is not implemented")
    contract = checkpoint['dataset_contract']
    extent = float(contract['region']['half_extent_m'])
    positions, coefficients, degree = _point_sh_state(checkpoint, polarization)
    energy = coefficients.abs().square().sum(-1)
    relative = deposit_points(positions, energy, extent, grid)
    source = dict(checkpoint=str(Path(checkpoint_path).resolve()), epoch=checkpoint.get('epoch'),
                  best_validation=checkpoint.get('best_val'), updates=checkpoint.get('updates'),
                  range_model=recipe['range_model'], recipe_schema=recipe.get('schema'),
                  initialization=recipe.get('initialization'), priors=recipe.get('priors'),
                  active_scatterers=int(len(positions)), sh_degree=degree, half_extent_m=extent)
    rcs, physical_quantity = None, None
    if polarization == 'hh':
        geometry = train_look_geometry(contract, shard_root, polarization)
        g = abs(_gain(checkpoint, polarization))
        point_rcs = rift_point_rcs(coefficients, degree, g, recipe['range_model'],
                                   geometry['directions'], geometry['ranges_m'])
        rcs = deposit_points(positions, point_rcs, extent, grid)
        physical_quantity = 'RCS per G48 cell, mean over TRAIN look directions (m^2)'
        source.update(train_look_directions=len(geometry['directions']), gain_magnitude=g,
                      mean_train_range_m=float(geometry['ranges_m'].mean()),
                      conversion=('sqrt(RCS) = g rho / ((4 pi)^2 K)' if recipe['range_model'] == 'sum2'
                                  else 'sqrt(RCS) = g rho (2r)^2 / K'))
    return MethodPresentation('rift', polarization, relative,
                              'point-SH energy, active unlocked sum |c|^2, CIC on G48 (arbitrary units)',
                              rcs, physical_quantity, source)


# Each baseline's presenter lives in its own module and plugs in here by name.
PRESENTERS = {'rift': 'rift_pvc.gotcha_presentation:rift_presentation',
              'spinr': 'rift_pvc.gotcha_presentation_spinr:spinr_presentation',
              'geraf': 'rift_pvc.gotcha_presentation_geraf:geraf_presentation',
              'radar_fields': 'rift_pvc.gotcha_presentation_radar_fields:radar_fields_presentation',
              'radarsplat': 'rift_pvc.gotcha_presentation_radarsplat:radarsplat_presentation'}


def presenter(method):
    """The presentation function for ``method``; raises if its module is not available yet."""
    import importlib
    if method not in PRESENTERS:
        raise ValueError(f'unknown method {method!r}; known: {sorted(PRESENTERS)}')
    module, _, name = PRESENTERS[method].partition(':')
    return getattr(importlib.import_module(module), name)


# ---------------------------------------------------------------- writing --

def _atomic(path, write):
    path = Path(path)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    write(temporary)
    os.replace(temporary, path)


def _db(values, reference):
    with np.errstate(divide='ignore'):
        return 10 * np.log10(np.maximum(values, 0) / reference)


def write_method(root, presentation, extent):
    """Write one method's volumes; figures and summaries are rebuilt by ``render``."""
    root = Path(root)
    peak = float(presentation.relative.max())
    if not math.isfinite(peak) or peak <= 0:
        raise ValueError(f'{presentation.name}: relative intensity has no positive cell')
    meta = dict(schema=SCHEMA, method=presentation.method, polarization=presentation.polarization,
                grid=int(presentation.relative.shape[0]), half_extent_m=float(extent), source=presentation.source)
    _save(root / RELATIVE_DIR / f'{presentation.name}.npz',
          intensity=presentation.relative / peak, native_intensity=presentation.relative,
          metadata=dict(meta, quantity=presentation.relative_quantity, peak_native=peak,
                        normalization="divided by the method's own brightest cell"))
    if presentation.rcs_m2 is not None:
        _save(root / PHYSICAL_DIR / f'{presentation.name}.npz',
              rcs_m2=presentation.rcs_m2, rcs_dbsm=_db(presentation.rcs_m2, 1.0),
              metadata=dict(meta, quantity=presentation.physical_quantity, system_constant_K=K,
                            uncertainty_db=K_UNCERTAINTY_DB))


def _save(path, *, metadata, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    def write(temporary):
        with open(temporary, 'wb') as handle:
            np.savez_compressed(handle, metadata=json.dumps(metadata, sort_keys=True, default=str), **arrays)
    _atomic(path, write)


def _load(directory):
    out = {}
    for path in sorted(Path(directory).glob('*.npz')):
        with np.load(path, allow_pickle=False) as data:
            out[path.stem] = {k: data[k] for k in data.files}
            out[path.stem]['metadata'] = json.loads(str(data['metadata']))
    return out


def _figure(panels, path, *, vmin, vmax, unit, extent):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = len(panels)
    fig, axes = plt.subplots(rows, 3, figsize=(11, 3.3 * rows + 0.4), squeeze=False, constrained_layout=True)
    image = None
    for r, (title, volume_db) in enumerate(panels):
        for c, (plane, axis, xlabel, ylabel) in enumerate(PLANES):
            ax = axes[r][c]
            # Empty cells (-inf dB) take the floor colour rather than rendering as missing.
            projection = np.maximum(np.nan_to_num(volume_db.max(axis=axis), nan=vmin, neginf=vmin), vmin)
            image = ax.imshow(projection.T, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax,
                              extent=(-extent, extent, -extent, extent), interpolation='nearest')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{title}: {plane}' if c == 0 else plane, fontsize=9)
    fig.colorbar(image, ax=axes, shrink=0.8, label=unit)
    _atomic(path, lambda p: fig.savefig(p, dpi=150, format='png'))
    plt.close(fig)


def render(root):
    """Rebuild every figure and summary in both directories from the saved volumes."""
    root = Path(root)
    relative, physical = _load(root / RELATIVE_DIR), _load(root / PHYSICAL_DIR)
    manifest = dict(schema=SCHEMA, grid=PRESENTATION_GRID, directories={
        RELATIVE_DIR: 'each method in its own units, 0 dB = its brightest cell, range '
                      f'{RELATIVE_FLOOR_DB:g}..0 dB; method-native quantity (see each summary)',
        PHYSICAL_DIR: 'RCS per cell in dBsm through K; one colour scale shared by all methods'},
        system_constant_K=K, uncertainty_db=K_UNCERTAINTY_DB,
        K_source='rift/gotcha_calibration.py; docs/GOTCHA_FORWARD_MODEL_ALIGNMENT.md section 13',
        methods={name: v['metadata']['source'] for name, v in relative.items()})
    if relative:
        extent = next(iter(relative.values()))['metadata']['half_extent_m']
        summary = {}
        for name, v in relative.items():
            db = _db(v['intensity'], 1.0)
            _figure([(name, db)], root / RELATIVE_DIR / f'{name}.png', vmin=RELATIVE_FLOOR_DB, vmax=0,
                    unit='dB re brightest cell', extent=extent)
            summary[name] = dict(quantity=v['metadata']['quantity'], peak_native=v['metadata']['peak_native'],
                                 cells_within_20db=int((db >= -20).sum()), source=v['metadata']['source'])
        _figure([(n, _db(v['intensity'], 1.0)) for n, v in relative.items()],
                root / RELATIVE_DIR / 'comparison.png', vmin=RELATIVE_FLOOR_DB, vmax=0,
                unit='dB re each method\'s brightest cell', extent=extent)
        _atomic(root / RELATIVE_DIR / 'summary.json',
                lambda p: Path(p).write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n'))
    if physical:
        extent = next(iter(physical.values()))['metadata']['half_extent_m']
        top = max(float(v['rcs_dbsm'][np.isfinite(v['rcs_dbsm'])].max()) for v in physical.values())
        vmax = 5 * math.ceil(top / 5)
        vmin = vmax - PHYSICAL_SPAN_DB
        summary = dict(system_constant_K=K, uncertainty_db=K_UNCERTAINTY_DB, colour_scale_dbsm=[vmin, vmax])
        for name, v in physical.items():
            # Each method's own figure spans its own 40 dB below its peak, still in absolute dBsm, so
            # a method far weaker than the others stays visible; comparison.png keeps the shared scale.
            rcs = v['rcs_m2']
            own = 5 * math.ceil(float(_db(rcs.max(), 1.0)) / 5) if rcs.max() > 0 else vmax
            _figure([(name, v['rcs_dbsm'])], root / PHYSICAL_DIR / f'{name}.png', vmin=own - PHYSICAL_SPAN_DB,
                    vmax=own, unit='dBsm per cell', extent=extent)
            summary[name] = dict(quantity=v['metadata']['quantity'], figure_scale_dbsm=[own - PHYSICAL_SPAN_DB, own],
                                 peak_cell_dbsm=float(_db(rcs.max(), 1.0)),
                                 total_in_cube_dbsm=float(_db(rcs.sum(), 1.0)),
                                 source=v['metadata']['source'])
        _figure([(n, v['rcs_dbsm']) for n, v in physical.items()], root / PHYSICAL_DIR / 'comparison.png',
                vmin=vmin, vmax=vmax, unit='dBsm per cell', extent=extent)
        _atomic(root / PHYSICAL_DIR / 'summary.json',
                lambda p: Path(p).write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n'))
    _atomic(root / 'manifest.json',
            lambda p: Path(p).write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + '\n'))
    return manifest
