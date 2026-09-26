"""Fixed native pulse subsets shared across methods and every scored role.

Selection changes the declared acquisition, not the stored archives.
Native row/pulse IDs remain intact; test payloads remain sealed.
The hash priority is independent of model, call order and polarization, so
identical pulse inventories share the same selection across methods/channels.
"""
from __future__ import annotations

import hashlib
import struct

import numpy as np


SCHEMA = 'gotcha_fixed_role_pulses_v2'
ROLES = ('train', 'validation', 'test')
SEED = 42
POLICY = 'lowest_blake2b64_priority_seed_pass_sector_pulse_index_v1'


def validate_pulse_limit(value):
    if type(value) is not int or value < 0:
        raise ValueError('pulses-per-sector must be a nonnegative integer (0 means all)')
    return value


def select_rows(rows, pulse_indices, limit, pass_id, sector):
    """Nested uniform hash-ranked subset, returned in native pulse order."""
    validate_pulse_limit(limit)
    if not limit or len(rows) <= limit:
        return rows
    priorities = [int.from_bytes(hashlib.blake2b(
        struct.pack('<qqqq', SEED, int(pass_id), int(sector), int(pulse_indices[row])),
        digest_size=8, person=b'RIFTpulsev1').digest(), 'little') for row in rows]
    # Resolve any hash ties by pulse identity; choose positions, then restore
    # the source pulse order. Increasing the cap gives a nested superset.
    chosen = sorted(range(len(rows)), key=lambda i: (priorities[i], int(pulse_indices[rows[i]])))[:limit]
    selected = np.asarray(rows)[sorted(chosen)].copy()
    selected.setflags(write=False)
    return selected


def apply_training_selection(dataset, limit):
    """Restrict both iteration and direct shard reads before response access.

    Called once during GOTCHADataset construction, after native metadata gates.
    Original acquisition headers/arrays/identities stay unchanged. Unselected
    rows in every role become inaccessible through the selected dataset's
    public reader. The function name is retained for API compatibility.
    """
    validate_pulse_limit(limit)
    if limit == 0:
        return None
    inventory = {}
    for (pass_id, pol), shard in dataset.shards.items():
        if shard.response_reads:
            raise ValueError('Pulse selection must precede response access')
        source_rows = shard.sector_rows
        source_roles = shard.row_roles
        rows_by_sector = dict(source_rows)
        roles = source_roles.copy()
        for role in ROLES:
            for sector in dataset.splits_by_pass[pass_id][role]:
                rows = source_rows[sector]
                selected = select_rows(rows, shard.arrays['pulse_index'], limit, pass_id, sector)
                roles[rows] = 'excluded'
                roles[selected] = role
                rows_by_sector[sector] = selected
        roles.setflags(write=False)
        # Keep full metadata for provenance and diagnostics, never as a bypass
        # to the role checks in read(). Source response shape is still native.
        shard.source_sector_rows = source_rows
        shard.source_row_roles = source_roles
        shard.sector_rows = rows_by_sector
        shard.row_roles = roles
        inventory[f'pass{pass_id}_{pol}'] = {}
        for role in ROLES:
            selected_rows = np.flatnonzero(roles == role)
            native_ids = shard.arrays['pulse_index'][selected_rows]
            digest = hashlib.sha256()
            for values in (selected_rows, native_ids):
                digest.update(np.asarray(values, dtype='<i8').tobytes())
            inventory[f'pass{pass_id}_{pol}'][role] = dict(
                source_pulses=int(np.count_nonzero(source_roles == role)),
                selected_pulses=len(selected_rows),
                selected_native_rows_and_pulse_ids_sha256=digest.hexdigest())
    return dict(schema=SCHEMA, pulses_per_sector=limit, seed=SEED, selection=POLICY,
                roles=list(ROLES), normalization='selected_train_pulses',
                validation='same_fixed_pulse_cap', test='same_fixed_pulse_cap_payload_sealed',
                inventory=inventory)


def pulse_limit_from_contract(contract):
    """Recover the declared pulse cap; legacy contracts mean all pulses."""
    selection = contract.get('training_pulse_selection')
    if selection is None:
        return 0
    if (not isinstance(selection, dict) or selection.get('schema') != SCHEMA
            or selection.get('seed') != SEED or selection.get('selection') != POLICY
            or selection.get('roles') != list(ROLES)
            or selection.get('normalization') != 'selected_train_pulses'
            or selection.get('validation') != 'same_fixed_pulse_cap'
            or selection.get('test') != 'same_fixed_pulse_cap_payload_sealed'):
        raise ValueError('Unsupported saved pulse selection; expected matching train/validation/test sampling')
    limit = validate_pulse_limit(selection.get('pulses_per_sector'))
    if not limit:
        raise ValueError('Saved pulse-selection contract requires a positive cap')
    return limit
