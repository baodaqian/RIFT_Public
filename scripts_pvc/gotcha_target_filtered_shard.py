#!/usr/bin/env python3
"""New-target step 3: a box-isolated, pulse-decimated GOTCHA shard for any named target region (CPU).

A separate copy of ``gotcha_filtered_shard.py`` (the Camry builder, unchanged) for the new GOTCHA targets
(docs/RIFT_GOTCHA_Tune.md A73-A75). The only differences: the region and its config file, and the box, are
arguments (``--region``, ``--region-config``, ``--box-name``, ``--box-centre``, ``--box-heading-deg``,
``--box-half-extents``) instead of the Camry's fixed ``camry`` / ``BOX_V2``. The isolation operating point is the
Camry's frozen one, unchanged. The original description follows.

Write a box-isolated, pulse-decimated GOTCHA shard in the native shard format (Phase 0, reviewer B3/B11).

For one pass and polarization, every TRAIN and VALIDATION sector is read through the unchanged reader
(the channel's published autofocus applied once, all native pulses, all native frequency bins), projected
onto the box's per-sector subspace (``rift_pvc.gotcha_isolation``: hull footprint, geometry only) and
decimated to every ``--keep-every``-th pulse. The result is written as a new ``pass{p}_{pol}.npz``
that ``NativeShardReader`` reads unchanged:

- ``response.npy`` holds the filtered rows (complex64, all native bins) multiplied by
  exp(-i ph_correct_raw), so the reader's single autofocus application restores them; ``r0``,
  ``r_correct_raw`` and ``ph_correct_raw`` stay raw;
- every other member is the source's, restricted to the kept rows; ``pulse_index`` keeps its
  native values, so (sector, pulse) identities stay unique;
- ``metadata_json`` is the source's plus an ``isolation`` block (box, footprint, guards, steps,
  cutoff, per-sector rank and timing, git commit, status).

TEST sectors are never read: their kept rows are written as zeros and the block records
``test_rows: pending_user_decision_B5c``. The training adapter's role gate already refuses TEST.
This shard is for TRAIN/VALIDATION work until the user decides how TEST is filtered.

    source .local-setup/activate-pvc.sh
    python scripts_pvc/gotcha_target_filtered_shard.py --pass-id 1 --out-root <root> --region sentra_box_v1 \
        --region-config rift_pvc/regions/gotcha_new_targets.json --box-name sentra_box_v1 [--verify 4]
"""
import argparse
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rift.gotcha_dataset import NativeShardReader, load_region, sector_split  # noqa: E402
from rift_pvc.gotcha_isolation import Box, SectorGeometry, basis, footprint_grid, truncate  # noqa: E402

# The Camry's box size (camry_box_v2, A24/B12); new target boxes are centred on their region origin by default.
DEFAULT_BOX = dict(centre=(0.0, 0.0, 0.0), heading_deg=0.0, half_extents=(3.0, 1.5, 1.25))
OPERATING_POINT = dict(footprint='hull', range_step=0.10, cross_step=0.65, range_guard=0.25, cross_guard=1.3,
                       cutoff=0.1, frequency='all native bins', pulses='all native pulses of the sector')
MEMBERS = ('frequencies_hz', 'x', 'y', 'z', 'r0', 'th', 'phi', 'sector_id', 'pulse_index', 'pass_id',
           'polarization', 'role', 'r_correct_raw', 'ph_correct_raw')


def git_commit():
    try:
        return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return None


def projector(geometry, box, point):
    grid = footprint_grid(box, geometry, range_step=point['range_step'], cross_step=point['cross_step'],
                          range_guard=point['range_guard'], cross_guard=point['cross_guard'], shape=point['footprint'])
    Q, S = basis(geometry.responses(grid, dtype=torch.complex128))
    Q, rank = truncate(Q, S, point['cutoff'])
    return Q, rank, len(grid)


def filter_sector(reader, rows, region, box, point):
    observations = [reader.read(int(r)) for r in rows]
    if [o.pulse_index for o in observations] != [int(reader.arrays['pulse_index'][r]) for r in rows]:
        raise ValueError('observation order differs from the stored rows')
    geometry = SectorGeometry.from_observations(observations, region)
    started = time.perf_counter()
    Q, rank, columns = projector(geometry, box, point)
    built = time.perf_counter()
    y = torch.as_tensor(np.stack([o.response for o in observations]).reshape(-1), dtype=torch.complex128)
    filtered = (Q @ (Q.conj().T @ y)).reshape(len(observations), -1).numpy()
    applied = time.perf_counter()
    energy_kept = float((np.abs(filtered) ** 2).sum() / (np.abs(y.numpy()) ** 2).sum())
    return filtered, dict(rank=rank, dictionary_columns=columns, pulses=len(observations),
                          projector_build_s=round(built - started, 3), apply_s=round(applied - built, 4),
                          energy_kept=energy_kept)


def write_npz(path, arrays, metadata):
    """Uncompressed members (the reader memory-maps response.npy)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, value in [*arrays.items(), ('metadata_json', np.asarray(json.dumps(metadata, sort_keys=True)))]:
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asanyarray(value), allow_pickle=False)
            archive.writestr(f'{name}.npy', buffer.getvalue())
    with zipfile.ZipFile(temporary) as archive:
        if any(info.compress_type != zipfile.ZIP_STORED for info in archive.infolist()):
            raise ValueError('every member must be stored uncompressed')
    os.replace(temporary, path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset-root', default=os.environ.get('GOTCHA_DATA_ROOT',
                   '/scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined'))
    p.add_argument('--pass-id', type=int, required=True)
    p.add_argument('--polarization', default='hh')
    p.add_argument('--region', required=True)
    p.add_argument('--region-config', type=Path, required=True, help='schema rift_gotcha_regions_v1, new names')
    p.add_argument('--box-name', required=True)
    p.add_argument('--box-centre', type=float, nargs=3, default=list(DEFAULT_BOX['centre']),
                   help='box centre in the region local frame (m)')
    p.add_argument('--box-heading-deg', type=float, default=DEFAULT_BOX['heading_deg'])
    p.add_argument('--box-half-extents', type=float, nargs=3, default=list(DEFAULT_BOX['half_extents']))
    p.add_argument('--keep-every', type=int, nargs='+', default=[1, 6],
                   help='one archive per value, written under <out-root>/keep<k>/ from the same projectors')
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--limit-sectors', type=int, default=0, help='testing only: filter the first N TRAIN/VAL sectors')
    p.add_argument('--verify', type=int, default=4, help='re-read this many sectors through NativeShardReader')
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--status', default='provisional_pending_user_decision')
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    source_path = Path(args.dataset_root) / 'New_Transfer' / 'shards' / f'pass{args.pass_id}_{args.polarization}.npz'
    region = load_region(args.region, args.region_config)
    BOX = dict(name=args.box_name, frame=f'{args.region} region local', centre=tuple(args.box_centre),
               heading_deg=args.box_heading_deg, half_extents=tuple(args.box_half_extents),
               region_config=str(args.region_config))
    box = Box(BOX['centre'], BOX['heading_deg'], BOX['half_extents'])
    reader = NativeShardReader(source_path, args.pass_id, args.polarization, roles=('train', 'validation'))
    with np.load(source_path, allow_pickle=False) as source:
        metadata = json.loads(str(source['metadata_json'].item()))
        members = {k: np.array(source[k]) for k in MEMBERS if k in source.files}
    split = sector_split()
    role_of = {s: r for r, sectors in split.items() for s in sectors}
    outputs = {k: dict(kept=[], responses=[]) for k in args.keep_every}
    per_sector = {}
    developed = 0
    ph = members['ph_correct_raw'] if members['ph_correct_raw'].size else None
    bins = members['frequencies_hz'].size
    for sector in range(1, 361):
        rows = reader.sector_rows[sector]
        role = role_of[sector]
        if role in ('train', 'validation') and (not args.limit_sectors or developed < args.limit_sectors):
            filtered, info = filter_sector(reader, rows, region, box, OPERATING_POINT)
            if ph is not None:  # store AF-unapplied rows: the reader multiplies by exp(+i ph) once
                filtered = filtered * np.exp(-1j * ph[rows].astype(np.float64))[:, None]
            for k, out in outputs.items():
                out['kept'].append(rows[::k])
                out['responses'].append(filtered[::k].astype(np.complex64))
            per_sector[str(sector)] = dict(role=role, **info)
            developed += 1
            print(f'pass {args.pass_id} sector {sector} {role}: rank {info["rank"]}, build {info["projector_build_s"]} s, '
                  f'apply {info["apply_s"]} s, kept {info["energy_kept"]:.4f}', flush=True)
        else:
            # TEST (never read) and, in limited test runs, unprocessed sectors: zero rows.
            for k, out in outputs.items():
                out['kept'].append(rows[::k])
                out['responses'].append(np.zeros((len(rows[::k]), bins), dtype=np.complex64))
            per_sector[str(sector)] = dict(role=role, status='zero_rows_not_read' if role == 'test' else 'not_processed')
    builds = [v['projector_build_s'] for v in per_sector.values() if 'projector_build_s' in v]
    applies = [v['apply_s'] for v in per_sector.values() if 'apply_s' in v]
    written = {}
    for k, out in outputs.items():
        kept = np.concatenate(out['kept'])
        arrays = {name: (v if name == 'frequencies_hz' or v.size == 0 else v[kept]) for name, v in members.items()}
        arrays['response'] = np.concatenate(out['responses'])
        meta = dict(metadata, scene_id=f'{BOX["name"]}_filtered_keep{k}_{args.status}', isolation=dict(
            schema='gotcha_box_isolated_shard_v1', status=args.status, box=BOX, operating_point=OPERATING_POINT,
            keep_every=k, row_selection='rows 0, k, 2k, ... of each sector in native pulse order',
            stored_response='filtered x exp(-i ph_correct_raw): the reader applies the autofocus once',
            test_rows='pending_user_decision_B5c: zeros, source TEST responses never read',
            source=dict(path=str(source_path), identity=reader.identity), git_commit=git_commit(),
            record='docs/RIFT_GOTCHA_Tune.md A73-A75 (new targets); operating point of section 7, A24/B12', limited_sectors=args.limit_sectors,
            timing=dict(sectors=len(builds), projector_build_s_mean=float(np.mean(builds)) if builds else None,
                        projector_build_s_max=float(np.max(builds)) if builds else None,
                        apply_s_mean=float(np.mean(applies)) if applies else None),
            per_sector=per_sector))
        path = args.out_root / f'keep{k}' / 'New_Transfer' / 'shards' / source_path.name
        write_npz(path, arrays, meta)
        written[k] = path
        print(f'wrote {path}: {arrays["response"].shape} rows x bins', flush=True)
    if args.verify:
        done = [int(sec) for sec, v in per_sector.items() if 'rank' in v][:args.verify]
        expected = {sec: filter_sector(reader, reader.sector_rows[sec], region, box, OPERATING_POINT)[0] for sec in done}
        for k, path in written.items():
            check = NativeShardReader(path, args.pass_id, args.polarization, roles=('train', 'validation'))
            worst = 0.
            for sec in done:
                want = expected[sec][::k]
                got = np.stack([check.read(int(r)).response for r in check.sector_rows[sec]])
                worst = max(worst, float(np.abs(got - want).max() / np.abs(want).max()))
            print(f'verify keep{k}: {len(done)} sectors re-read through NativeShardReader, '
                  f'max relative deviation {worst:.2e} (complex64 storage)', flush=True)
            if worst > 1e-5:
                raise SystemExit(f'verification failed for keep{k}')

if __name__ == '__main__':
    main()
