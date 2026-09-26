"""Ordered selection of physical source antennas, shared by dataset adapters.

Stored arrays remain untouched. Selection is independent of viewpoint roles,
frequency sampling, synthetic-aperture pulses and polarization heads.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


def selection(num_tx=None, num_rx=None, tx_indices=None, rx_indices=None, *, source=(16, 16)):
    axes = []
    for count, indices, maximum, label in zip(
            (num_tx, num_rx), (tx_indices, rx_indices), source, ('Tx', 'Rx')):
        if count is not None and (isinstance(count, (bool, np.bool_))
                or not isinstance(count, (int, np.integer)) or not 1 <= count <= maximum):
            raise ValueError(f'{label} count must be an integer in [1,{maximum}]')
        if indices is None:
            indices = list(range(maximum if count is None else int(count)))
        else:
            indices = list(indices)
            if (not indices or any(isinstance(i, (bool, np.bool_))
                    or not isinstance(i, (int, np.integer)) or not 0 <= i < maximum for i in indices)
                    or len(set(indices)) != len(indices)):
                raise ValueError(f'{label} indices must be unique source integers in [0,{maximum})')
            if count is not None and count != len(indices):
                raise ValueError(f'{label} count and explicit indices disagree')
        axes.append(list(map(int, indices)))
    if axes == [list(range(source[0])), list(range(source[1]))]:
        return None  # Exact legacy full acquisition identity is preserved.
    return dict(schema='rift_ordered_source_antennas_v1', source_num_tx=source[0],
                source_num_rx=source[1], num_tx=len(axes[0]), num_rx=len(axes[1]),
                tx_indices=axes[0], rx_indices=axes[1], policy='ordered_source_indices')


def validate_selection(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError('Antenna selection must be an object')
    expected = selection(value.get('num_tx'), value.get('num_rx'),
                         value.get('tx_indices'), value.get('rx_indices'))
    if expected is None or value != expected:
        raise ValueError('Invalid source antenna selection')
    return expected


def acquisition_label(value):
    if value is None:
        return ''
    value = validate_selection(value)
    digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]
    return f"{value['num_tx']}t{value['num_rx']}r_{digest}"


def add_arguments(parser, *, default=None):
    parser.add_argument('--num-tx', type=int, default=default, help='Number of physical source transmitters')
    parser.add_argument('--num-rx', type=int, default=default, help='Number of physical source receivers')
    parser.add_argument('--tx-indices', type=int, nargs='+', help='Ordered zero-based source Tx indices')
    parser.add_argument('--rx-indices', type=int, nargs='+', help='Ordered zero-based source Rx indices')


def from_args(args, *, source=(16, 16), default=None):
    counts = []
    indices = []
    for axis in ('tx', 'rx'):
        count = getattr(args, 'num_' + axis, None)
        ordered = getattr(args, axis + '_indices', None)
        counts.append(default if count is None and ordered is None else count)
        indices.append(ordered)
    return selection(*counts, *indices, source=source)


def geometry_digest(arrays):
    from .npz_dataset import build_freqs
    digest = hashlib.sha256()
    for value in (arrays['tx_pos'], arrays['rx_pos'], build_freqs(arrays['meta'])):
        value = np.ascontiguousarray(value, dtype='<f8')
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


class SelectedResponseReader:
    def __init__(self, reader, tx, rx):
        self.reader, self.tx, self.rx = reader, tuple(tx), tuple(rx)

    def restricted_to(self, indices):
        return type(self)(self.reader.restricted_to(indices), self.tx, self.rx)

    def response_view(self, index):
        return next(self.iter_response_views([index]))[1]

    def iter_response_views(self, indices):
        # Collection archives are ZIP_STORED. Map only after role authorization,
        # then gather the requested channels without scanning other viewpoints.
        from .power_baseline_dataset import _stored_member_memmap
        ordered = sorted(set(self.reader._validate_allowed(indices)))
        if not ordered:
            return
        mapped = _stored_member_memmap(self.reader.source_path, 'response.npy')
        if mapped is None or mapped.shape != self.reader.shape or mapped.dtype != self.reader.dtype:
            raise ValueError('Selected antennas require the unchanged stored source response')
        for index in ordered:
            yield index, np.array(mapped[index][np.ix_(self.tx, self.rx)], copy=True)


def select_arrays(arrays, value):
    value = validate_selection(value)
    if value is None:
        return arrays
    if arrays.get('response') is not None or arrays.get('antenna_selection'):
        raise ValueError('Antenna selection requires an unselected lazy source')
    result = dict(arrays)
    result['source_response_shape'] = arrays['response_shape']
    result['antenna_selection'] = value
    result['source_geometry_sha256'] = geometry_digest(arrays)
    # Sensor attitude belongs to the full source array, independently of which
    # physical channels are selected. These are pose metadata, never responses.
    result['source_tx_pos'] = arrays['tx_pos']
    result['source_rx_pos'] = arrays['rx_pos']
    result['tx_pos'] = arrays['tx_pos'][:, value['tx_indices'], :].copy()
    result['rx_pos'] = arrays['rx_pos'][:, value['rx_indices'], :].copy()
    shape = list(arrays['response_shape'])
    shape[1:3] = [value['num_tx'], value['num_rx']]
    result['response_shape'] = tuple(shape)
    result['_lazy_response_reader'] = SelectedResponseReader(
        arrays['_lazy_response_reader'], value['tx_indices'], value['rx_indices'])
    return result
