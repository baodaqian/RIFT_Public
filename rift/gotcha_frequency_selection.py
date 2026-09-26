"""Shared native-bin selection for training and every declared evaluation role.

A frequency subset changes each method's training objective and normalization.
It is not an unbiased estimate of its full-frequency loss. These helpers do
not read responses or alter source identities, geometry, or autofocus.
"""
from __future__ import annotations

import hashlib
import numpy as np

SCHEMA = 'gotcha_role_native_frequency_selection_v2'
ROLES = ('train', 'validation', 'test')


def selection_config(stride):
    if type(stride) is not int or stride not in (1, 2):
        raise ValueError('frequency-stride must be 1 or 2; larger strides need separate support/rank/aliasing qualification')
    if stride == 1:
        return None
    return dict(schema=SCHEMA, stride=stride, source_bin_policy='zero_based_stride_plus_last_endpoint',
                roles=list(ROLES), validation='same_selected_native_bins',
                test='same_selected_native_bins_payload_sealed',
                normalization='selected_training_acquisition_only',
                objective='changed_training_acquisition_not_unbiased_full_frequency_loss',
                range_projector='role_specific_exact_frequencies_half_Rayleigh_two_guards_SVD_1e-10')


def selected_indices(frequencies, stride):
    selection_config(stride)
    frequencies = np.asarray(frequencies)
    if (frequencies.ndim != 1 or len(frequencies) < 3 or not np.isfinite(frequencies).all()
            or frequencies[0] <= 0 or not (np.diff(frequencies) > 0).all()):
        raise ValueError('Source frequencies must be finite, positive and strictly increasing')
    indices = np.arange(0, len(frequencies), stride, dtype=np.int64)
    if indices[-1] != len(frequencies) - 1:
        indices = np.append(indices, len(frequencies) - 1)
    indices.setflags(write=False)
    return indices


def _frequency_hash(frequencies):
    return hashlib.sha256(np.ascontiguousarray(frequencies, dtype='<f8').tobytes()).hexdigest()


class NativeFrequencySelection:
    """Role-aware metadata for one shard, retaining exact source-bin indices."""
    def __init__(self, frequencies, stride=1):
        self.config = selection_config(stride)
        self.stride = stride
        self.source = np.asarray(frequencies, dtype=np.float64)
        self.source_indices = selected_indices(self.source, 1)
        self.train_indices = selected_indices(self.source, stride)
        self.train = self.source if stride == 1 else self.source[self.train_indices]
        self.train.setflags(write=False)
        self.contract = None if self.config is None else dict(
            source_count=len(self.source), selected_count=len(self.train),
            source_indices=self.train_indices.tolist(),
            source_frequencies_sha256=_frequency_hash(self.source),
            selected_frequencies_sha256=_frequency_hash(self.train),
            endpoints_hz=[float(self.source[0]), float(self.source[-1])])

    def indices(self, role):
        if role not in ROLES:
            raise PermissionError('Frequency metadata access requires a declared acquisition role')
        return self.train_indices

    def frequencies(self, role):
        if role not in ROLES:
            raise PermissionError('Frequency metadata access requires a declared acquisition role')
        return self.train


def selection_contract(shards, stride):
    config = selection_config(stride)
    if config is None:
        return None
    return dict(config, shards={f'pass{p}_{pol}': shard._frequency_selection.contract
                               for (p, pol), shard in sorted(shards.items())})


def kwargs_from_contract(contract):
    """Reconstruct the selection before existing cache/checkpoint identity gates."""
    config = contract.get('frequency_selection')
    if config is None:
        return {}
    if (not isinstance(config, dict) or config.get('stride') != 2
            or {k: v for k, v in config.items() if k != 'shards'} != selection_config(config['stride'])
            or not isinstance(config.get('shards'), dict)):
        raise ValueError('Unsupported saved frequency-selection contract')
    return dict(frequency_stride=config['stride'])


def recipe_fields(dataset):
    """Compose both optional acquisition identities; legacy dictionaries stay exact."""
    fields = {}
    for source, target in (('frequency_selection', 'training_frequency_selection'),
                           ('training_pulse_selection', 'training_pulse_selection')):
        if source in dataset.contract:
            fields[target] = dataset.contract[source]
    return fields


def bind_recipe(dataset, recipe):
    """Describe selected acquisition without changing any model/loss settings."""
    fields = recipe_fields(dataset)
    if not fields:
        return recipe
    result = dict(recipe, **fields)
    if 'training_frequency_selection' in fields and 'frequency_policy' in result:
        result['frequency_policy'] = 'same_selected_native_bins_train_validation_test'
    if 'training_pulse_selection' in fields and 'pulse_policy' in result:
        result['pulse_policy'] = 'same_fixed_shared_pulse_cap_train_validation_test'
        if 'update_unit' in result:
            result['update_unit'] = 'one_pass_sector_selected_TRAIN_pulses_and_selected_channels'
    return result


def preflight(dataset):
    """Check the worst support over all declared roles using the same projector.

    Guarded interval width and basis-column count grow monotonically with cube
    path width. Its maximum covers every selected pulse's support/alias gates;
    the actual SVD establishes retained rank. Only geometry metadata are used.
    """
    if dataset.contract.get('frequency_selection') is None:
        return None
    from rift.gotcha_dataset import C, Observation
    from rift.gotcha_training import RangeReadout
    readout = RangeReadout(dataset.region, device='cpu')
    report = {}
    for (p, pol), shard in sorted(dataset.shards.items()):
        role_rows = {role: np.concatenate([shard.sector_rows[s]
                     for s in dataset.splits_by_pass[p][role]]) for role in ROLES}
        rows = np.concatenate(list(role_rows.values()))
        a = shard.arrays
        positions = np.stack([a[k][rows] for k in ('x', 'y', 'z')], axis=1)
        local = dataset.region.to_local(positions)
        extent = dataset.region.half_extent_m
        widths = (np.linalg.norm(np.abs(local) + extent, axis=1)
                  - np.linalg.norm(local - np.clip(local, -extent, extent), axis=1))
        index = int(np.argmax(widths))
        row = int(rows[index])
        r0 = float(a['r0'][row]) + (float(a['r_correct_raw'][row]) if shard.co_pol else 0.)
        f = shard.frequencies_for_role(str(shard.row_roles[row]))
        observation = Observation(p, pol, int(a['sector_id'][row]), int(a['pulse_index'][row]),
            positions[index], f, r0, np.empty(0, dtype=np.complex128),
            'own_published_source_af_once' if shard.co_pol else 'raw_official_af_absent')
        try:
            r = readout.for_observation(observation)
        except ValueError as exc:
            raise ValueError(f'Frequency selection pass{p}_{pol}: {exc}') from exc
        rank = r['q'].shape[1]
        if not 0 < rank <= len(r['delays']) < len(f):
            raise ValueError('Frequency selection has an invalid retained projector rank')
        spacing = C / (2 * (f[-1] - f[0]))
        report[f'pass{p}_{pol}'] = dict(source_count=len(shard.frequencies_hz),
            selected_count=len(f), max_guarded_basis_columns=len(r['delays']),
            rank_at_max_support=rank, max_guarded_interval_m=float(widths[index] + 4 * spacing),
            conservative_alias_period_m=float(C / (2 * np.diff(f).max())),
            by_role={role: dict(pulses=len(values), native_samples=int(len(values)*len(f)))
                     for role, values in role_rows.items()})
    return dict(schema=SCHEMA, stride=dataset.frequency_stride, shards=report,
                response_payload_read=False, support_rank_alias_checks='passed',
                roles=list(ROLES), validation='same selected native bins and rebuilt projector',
                test='metadata checked; payload sealed')
