#!/usr/bin/env python3
"""New-target step 4b: apply per-pass phase constants to a new target's box-isolated shards (a separate copy of
``gotcha_rotate_shard_passphase.py``, unchanged for the Camry; docs/RIFT_GOTCHA_Tune.md A73-A75). The only difference:
shared helpers come from the new-target shard builder. The original description follows.

Copy box-isolated GOTCHA shards with a constant per-pass phase correction (docs/RIFT_GOTCHA_Tune.md B38/B39).

Section 5b measured, on TRAIN sectors only, the phase of each pass relative to pass 1 at the dataset's
trihedrals (coherence 0.907; median offsets 0, -0.22, -0.36, -0.63, -0.34, -0.82, -1.20, -1.20 rad).
Its prescription for a common offset is one complex constant per pass, applied to the data for every
method and role alike (the same category as the K = 7988 calibration). This script writes that data:
every stored row of pass p is multiplied by exp(-i phi_p), all other members unchanged, and the
isolation block gains a ``pass_phase_correction`` record (offsets, convention, provenance, status).
TEST rows are zeros in the source and stay zeros. Adopting the corrected shards for the campaign is
the user's decision; until then the status is provisional.

    python scripts_pvc/gotcha_rotate_shard_passphase.py --source-root <.../keep6/New_Transfer/shards> \\
        --out-root <.../camry_box_v2_provisional_passphase/keep6/New_Transfer/shards>
"""
import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts_pvc.gotcha_target_filtered_shard import git_commit, write_npz  # noqa: E402

SECTION_5B_OFFSETS = {1: 0.0, 2: -0.22, 3: -0.36, 4: -0.63, 5: -0.34, 6: -0.82, 7: -1.20, 8: -1.20}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--polarization', default='hh')
    p.add_argument('--passes', type=int, nargs='+', default=list(range(1, 9)))
    p.add_argument('--status', default='provisional_pending_user_decision')
    p.add_argument('--offsets-json', type=Path,
                   help='per-pass offsets (radians) with provenance, {"offsets_rad": {"1": ...}, "convention", "source", '
                        '"tag"}; default the section 5b trihedral constants (A52: the car-level B56 constants)')
    args = p.parse_args(argv)
    offsets, convention, source, tag = (SECTION_5B_OFFSETS,
        'stored rows of pass p multiplied by exp(-i phi_p); phi_p = phase of pass p relative to '
        'pass 1 at the TRAIN-only trihedrals (median over seven reflectors)',
        'docs/RIFT_GOTCHA_Tune.md section 5b (A7), scripts_pvc/gotcha_interpass_coherence.py', 'passphase5b')
    if args.offsets_json is not None:
        spec = json.loads(args.offsets_json.read_text())
        offsets = {int(k): float(v) for k, v in spec['offsets_rad'].items()}
        convention, source, tag = spec['convention'], spec['source'], spec['tag']
        if set(args.passes) - set(offsets):
            raise SystemExit('the offsets file lacks a selected pass')
    for pass_id in args.passes:
        name = f'pass{pass_id}_{args.polarization}.npz'
        with zipfile.ZipFile(args.source_root / name) as archive:
            arrays = {info.filename[:-4]: np.lib.format.read_array(io.BytesIO(archive.read(info.filename)),
                                                                   allow_pickle=False)
                      for info in archive.infolist()}
        metadata = json.loads(str(arrays.pop('metadata_json').item()))
        phi = offsets[pass_id]
        arrays['response'] = (arrays['response'] * np.complex64(np.exp(-1j * phi))).astype(np.complex64)
        isolation = dict(metadata.get('isolation') or {})
        isolation['pass_phase_correction'] = dict(
            offset_rad=phi, offsets_rad={str(k): v for k, v in offsets.items()},
            convention=convention, source=source,
            applies_to='every method and role alike (data calibration, as K = 7988)', status=args.status,
            source_shard=str(args.source_root / name), git_commit=git_commit())
        metadata['isolation'] = isolation
        metadata['scene_id'] = f"{metadata.get('scene_id', 'camry_box_v2')}_{tag}"
        write_npz(args.out_root / name, arrays, metadata)
        print(f'pass {pass_id}: rotated by exp(-i {phi:+.2f}) -> {args.out_root / name}', flush=True)


if __name__ == '__main__':
    main()
