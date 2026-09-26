"""Shared pieces of the GOTCHA presentation for the power-domain baselines.

Radar Fields and RadarSplat fit matched-range POWER normalized by a TRAIN peak
(and, for Radar Fields and RadarSplat's ``log_train_peak_60db_v1``, mapped to
[0, 1] over a dB span). Their RCS readout therefore works through power, never a
coherent amplitude:

1. Render the reconstruction with the method's own renderer and undo the
   method's own intensity mapping, giving the native matched-range power the
   model claims. Rendered values at or below the mapping's floor carry no
   physical meaning and are set to zero.
2. Attribute that rendered power to the G48 cells (method-specific).
3. Divide by ``e1``, the total matched-range power a 1 m^2 point target at the
   region centre produces through the method's OWN target formation: its
   response is data = K sqrt(sigma) / (2R)^2 per sample (the definition of K), so
   its matched-range power is K^2 sigma / (2R)^4 at its own bin. The ratio is
   the RCS of the point target with the same total received power (energy
   matching, independent of the renderer's point-spread width).

Nothing here reads a response: the dataset is rebuilt from the checkpoint's
contract through the metadata-only ``GOTCHADataset`` constructor, and only
TRAIN look geometry and frequencies are used.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from rift.gotcha_dataset import C, GOTCHADataset, Region
from rift_pvc.gotcha_presentation import K

DEFAULT_LOOKS = 64


def region_from_contract(contract):
    value = contract['region']
    return Region(name=value['name'], target_id=value['target_id'], translation_m=tuple(value['translation_m']),
                  rotation_local_to_native=tuple(tuple(r) for r in value['rotation_local_to_native']),
                  half_extent_m=float(value['half_extent_m']), placement_provenance=value['placement_provenance'])


def dataset_from_contract(contract, shard_root):
    """Metadata-only dataset equal to the one a checkpoint was trained on.

    The constructor reads shard metadata (positions, sectors, frequencies,
    roles), never a response payload; the rebuilt contract must equal the saved
    one, so the looks and frequencies are those of the run.
    """
    from rift.gotcha_frequency_selection import kwargs_from_contract
    from rift.gotcha_pulse_sampling import pulse_limit_from_contract
    shard_root = Path(shard_root)
    selection = contract['split'].get('training_selection') or {}
    dataset = GOTCHADataset(shard_root.parent.parent, shard_root=shard_root, passes=contract['passes'],
                            polarizations=contract['polarizations'], region=region_from_contract(contract),
                            num_train=selection.get('num_train'), pulses_per_sector=pulse_limit_from_contract(contract),
                            **kwargs_from_contract(contract))
    if dataset.contract != contract:
        raise ValueError('GOTCHA shards/metadata differ from the checkpoint dataset contract')
    return dataset


def select_looks(views, count=DEFAULT_LOOKS):
    """Evenly spaced subset of the TRAIN (pass, sector) looks, in their stored order."""
    views = list(views)
    if count is None or count >= len(views):
        return views
    index = np.unique(np.round(np.linspace(0, len(views) - 1, int(count))).astype(int))
    return [views[i] for i in index]


def look_metadata(dataset, view, polarization):
    """Native and local pulse positions and TRAIN frequencies of one look (metadata only)."""
    pass_id, sector = view
    shard = dataset.shards[(pass_id, polarization)]
    rows = shard.sector_rows[sector]
    native = np.stack([np.asarray(shard.arrays[k][rows], dtype=np.float64) for k in ('x', 'y', 'z')], axis=-1)
    return dict(view=(int(pass_id), int(sector)), native=native, local=dataset.region.to_local(native),
                frequencies=np.asarray(shard.frequencies_for_role('train'), dtype=np.float64))


def point_target_observation(native_position, local_position, frequencies, point_local=(0., 0., 0.), rcs_m2=1.0):
    """The response of an isolated point of RCS ``rcs_m2`` through K's definition.

    data / K = sqrt(RCS) / (Rt + Rr)^2 per sample, monostatic Rt = Rr = R; the
    reference range is R itself, so the matched filter is coherent at the point.
    """
    distance = float(np.linalg.norm(np.asarray(point_local, dtype=np.float64) - np.asarray(local_position)))
    amplitude = K * math.sqrt(rcs_m2) / (2 * distance) ** 2
    return SimpleNamespace(position_m=np.asarray(native_position, dtype=np.float64),
                           frequencies_hz=np.asarray(frequencies, dtype=np.float64), reference_range_m=distance,
                           response=np.full(len(frequencies), amplitude, dtype=np.complex128)), distance


def point_target_power(rcs_m2, distance):
    """K^2 sigma / (2R)^4: the matched-range power of the point at its own bin."""
    return K ** 2 * rcs_m2 / (2 * distance) ** 4


def invert_db(intensity, peak_power, dynamic_range_db, floor_intensity=0.0):
    """Native power from an intensity in the [0, 1] dB domain; at/below the floor -> 0."""
    intensity = torch.as_tensor(intensity, dtype=torch.float64)
    power = float(peak_power) * torch.pow(10.0, dynamic_range_db * (intensity - 1) / 10)
    return torch.where(intensity > floor_intensity, power, torch.zeros_like(power))


def bin_points(positions, values, extent, grid):
    """Sum point values into the G48 cell that contains each point (points outside are dropped)."""
    positions = torch.as_tensor(positions, dtype=torch.float64)
    values = torch.as_tensor(values, dtype=torch.float64)
    inside = (positions.abs() <= extent).all(-1)
    index = ((positions[inside] + extent) / (2 * extent / grid)).floor().long().clamp(0, grid - 1)
    volume = torch.zeros(grid ** 3, dtype=torch.float64)
    volume.index_add_(0, (index[:, 0] * grid + index[:, 1]) * grid + index[:, 2], values[inside])
    return volume.view(grid, grid, grid).numpy(), float(values[~inside].sum())


def cell_centres(extent, grid):
    axis = (torch.arange(grid, dtype=torch.float64) + .5) * (2 * extent / grid) - extent
    return torch.cartesian_prod(axis, axis, axis)
