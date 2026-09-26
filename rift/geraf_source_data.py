"""Role-restricted RIFT/GOTCHA ingress for the same released GeRaF model."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from rift.geraf_source_ops import NativeAcquisition


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class RIFTSourceData:
    def __init__(self, npz_path, role_manifest):
        from rift.rift_dataset import load_object_contract, catalog, evaluation_role_indices
        self.arrays, contract = load_object_contract(npz_path, role_manifest)
        source = Path(npz_path).resolve()
        stat = source.stat()
        self.contract = dict(dataset_identity=contract['dataset_identity'], experiment_contract=contract, source=dict(path=str(source),
            bytes=stat.st_size, mtime_ns=stat.st_mtime_ns),
            acquisition='native_bistatic_frequency_response',
            heads=['scalar'], reference_path='zero')
        self.identity = identity_hash(self.contract)
        self.extent = float(catalog()['scene_extent_m'])
        self.heads = ('scalar',)
        self.roles = {role: tuple(map(int, evaluation_role_indices(contract, role)))
                      for role in ('train', 'validation')}

    def views(self, role):
        if role not in self.roles:
            raise PermissionError('Only sealed train/validation roles are accepted')
        return self.roles[role]

    def key(self, role, view, head):
        if head not in self.heads or view not in self.views(role):
            raise PermissionError('Unregistered GeRaF object/view/role')
        return f'{role}_{head}_{view:05d}'

    def acquisition(self, role, view, head, recipe, device):
        from rift.geraf_signal_operator import bistatic_pair_positions
        from rift.power_baseline_dataset import frequency_grid_hz
        self.key(role, view, head)
        tx, rx = bistatic_pair_positions(
            torch.as_tensor(self.arrays['tx_pos'][view], device=device),
            torch.as_tensor(self.arrays['rx_pos'][view], device=device))
        freqs = torch.as_tensor(frequency_grid_hz(self.arrays['meta']), device=device)
        return NativeAcquisition(tx, rx, freqs, torch.zeros(len(tx), dtype=torch.float64, device=device),
                                 point_chunk=recipe['point_chunk'], pair_chunk=min(recipe['pair_chunk'], len(tx)),
                                 uniform_rift=True)

    def response(self, role, view, head, device):
        from rift.npz_dataset import get_npz_response_view
        self.key(role, view, head)
        raw = get_npz_response_view(self.arrays, view)
        # Every actual channel/frequency; static chirp repeats are averaged.
        return torch.as_tensor(raw.mean(axis=2).reshape(-1, raw.shape[-1]),
                               dtype=torch.complex128, device=device)


class GOTCHASourceData:
    def __init__(self, dataset):
        self.dataset = dataset
        self.contract = dict(native_gotcha_contract=dataset.contract,
                             source_identity=dataset.identity,
                             acquisition='native_monostatic_pass_sector_frequency_response',
                             heads=list(dataset.polarizations), reference_path='2*r0_effective')
        self.identity = identity_hash(self.contract)
        self.extent = dataset.region.half_extent_m
        self.heads = tuple(dataset.polarizations)

    def views(self, role):
        if role not in ('train', 'validation'):
            raise PermissionError('Only sealed train/validation roles are accepted')
        return tuple(self.dataset.viewpoints(role))

    def key(self, role, view, head):
        if head not in self.heads or tuple(view) not in self.views(role):
            raise PermissionError('Unregistered GeRaF GOTCHA pass-sector/polarization/role')
        return f'{role}_{head}_pass{view[0]}_sector{view[1]:03d}'

    def acquisition(self, role, view, head, recipe, device):
        self.key(role, view, head)
        shard = self.dataset.shards[view[0], head]
        rows = shard.sector_rows[view[1]]
        a = shard.arrays
        positions = np.stack([a[k][rows] for k in ('x', 'y', 'z')], -1).astype(np.float64)
        points = torch.as_tensor(self.dataset.region.to_local(positions), device=device)
        reference = a['r0'][rows].astype(np.float64)
        if head in ('hh', 'vv'):
            reference += a['r_correct_raw'][rows]
        return NativeAcquisition(points, points,
            torch.as_tensor(shard.frequencies_for_role(role).copy(), dtype=torch.float64, device=device),
            torch.as_tensor(2 * reference, dtype=torch.float64, device=device),
            point_chunk=recipe['point_chunk'], pair_chunk=recipe['pair_chunk'])

    def response(self, role, view, head, device):
        self.key(role, view, head)
        # Native reader owns HH/VV autofocus exactly once; cross-pol is raw.
        shard = self.dataset.shards[view[0], head]
        rows = shard.sector_rows[view[1]]
        values = list(self.dataset.observations(view[0], view[1], head))
        if (len(values) != len(rows) or
                tuple(x.pulse_index for x in values) != tuple(map(int, shard.arrays['pulse_index'][rows]))):
            raise ValueError('Native pulse inventory/order changed')
        return torch.as_tensor(np.stack([x.response for x in values]), dtype=torch.complex128, device=device)
