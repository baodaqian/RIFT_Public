"""Pass-held-out unit split for RIFT on GOTCHA (tuning campaign A41; reviewer design B48/B49).

The registered split holds out whole sector IDs in every pass, so a VAL look sits 1–4 sectors
from any TRAIN look: extrapolation for a car whose response decorrelates within about 0.18°.
This split keeps every 4th unsealed sector ID (by default) across the full 360° and all passes, and
holds out the units of ``heldout_pass`` for a seeded fraction of those IDs. The same IDs train in
the other passes, which sit 0.07–0.11° away in elevation at the median (gap/coherence 0.4–0.6,
the RIFT dataset's interpolation regime). Each (pass, sector) unit is filtered independently by
the isolation step, so held-out rows share no projector with training rows (B48).

Test sector IDs of the registered split stay sealed in every pass and are never selected. The
split is a new benchmark identity (B49): the dataset contract names it, so its identity, output
directory and resume gate differ from the registered split's.

``rift/`` stays unchanged (AGENTS.md): this module re-labels the rows of an already opened
``GOTCHADataset`` before any response is read.
"""
from __future__ import annotations

import json

import numpy as np

from rift.gotcha_dataset import digest, sector_split

SCHEMA = 'gotcha_unit_split_pass_heldout_v1'


def unit_split_ids(stride=4, heldout_fraction=0.5, seed=42):
    """(selected IDs, held-out IDs): every ``stride``-th unsealed ID in ascending order, seeded subset."""
    if int(stride) != stride or stride < 1:
        raise ValueError('The sector stride must be a positive integer')
    if not 0 < heldout_fraction < 1:
        raise ValueError('The held-out fraction must lie strictly between 0 and 1')
    parent = sector_split()
    unsealed = sorted(parent['train'] + parent['validation'])
    selected = unsealed[::int(stride)]
    count = int(round(heldout_fraction * len(selected)))
    if not 0 < count < len(selected):
        raise ValueError('The held-out fraction leaves no training or no held-out IDs')
    permutation = np.random.Generator(np.random.PCG64(seed)).permutation(len(selected))
    heldout = sorted(int(selected[i]) for i in permutation[:count])
    return [int(s) for s in selected], heldout


def apply_unit_split(dataset, *, stride=4, heldout_pass=4, heldout_fraction=0.5, seed=42):
    """Re-label ``dataset``'s rows to the pass-held-out unit split; returns the split record."""
    if heldout_pass not in dataset.passes:
        raise ValueError(f'Held-out pass {heldout_pass} is not among the selected passes {list(dataset.passes)}')
    if len(dataset.passes) < 2:
        raise ValueError('A pass-held-out split needs at least one training pass beside the held-out pass')
    if any(shard.response_reads for shard in dataset.shards.values()):
        raise RuntimeError('The unit split must be applied before any response is read')
    parent = sector_split()
    sealed = set(parent['test'])
    selected, heldout = unit_split_ids(stride, heldout_fraction, seed)
    if sealed & set(selected):
        raise AssertionError('A sealed test sector was selected')
    held = set(heldout)
    splits = {}
    for p in dataset.passes:
        train = tuple(s for s in selected if not (p == heldout_pass and s in held))
        validation = tuple(heldout) if p == heldout_pass else ()
        splits[p] = dict(train=train, validation=validation, test=tuple(parent['test']))
    for (p, _pol), shard in dataset.shards.items():
        if not np.array_equal(shard.frequency_indices_for_role('train'), shard.frequency_indices_for_role('validation')):
            raise ValueError('The unit split needs the same selected bins in the train and validation roles')
        roles = np.full(len(shard.row_roles), 'excluded', dtype=shard.row_roles.dtype)
        for role in ('train', 'validation', 'test'):
            for sector in splits[p][role]:
                roles[shard.sector_rows[sector]] = role
        roles.setflags(write=False)
        shard.row_roles = roles
    dataset.splits_by_pass = splits
    dataset.split = splits
    dataset.num_train = sum(len(s['train']) for s in splits.values())
    record = dict(schema=SCHEMA, seed=int(seed), sector_stride=int(stride), heldout_pass=int(heldout_pass),
                  heldout_fraction=float(heldout_fraction), selected_sector_ids=selected,
                  heldout_sector_ids=heldout, sealed_test_sector_ids=sorted(sealed),
                  units=dict(train=dataset.num_train, validation=len(heldout)),
                  unit='pass_sector', parent_schema='registered seed-42 sector split',
                  reason='interpolation benchmark for RIFT on GOTCHA (tuning campaign A41, B48/B49); '
                         'the registered VAL role is superseded for this campaign')
    dataset.training_selection = record
    dataset.contract['split'] = dict(record, sector_ids_by_pass={str(p): {k: list(v) for k, v in s.items()}
                                                                  for p, s in splits.items()})
    dataset.contract = json.loads(json.dumps(dataset.contract))
    dataset.identity = digest(dataset.contract)
    return record
