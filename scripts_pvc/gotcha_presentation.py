#!/usr/bin/env python3
"""Present GOTCHA reconstructions as relative intensity and as K-calibrated RCS.

Writes ``<output-root>/relative_intensity/`` and ``<output-root>/physical_K/`` with
identical file names per method, plus ``manifest.json``. Adding a method later to
the same root rewrites every figure, so the shared dBsm colour scale covers all
methods present. CPU only; reads checkpoints and TRAIN shard metadata, never a
response.

    source .local-setup/activate-pvc.sh
    python scripts_pvc/gotcha_presentation.py \\
        --output-root /scratch/group/p.cis261724.000/RIFT_pvc_runs/gotcha_presentation/camry_hh \\
        --run rift=<.../rift/checkpoint_best.pt>
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rift_pvc import gotcha_presentation as presentation  # noqa: E402

DEFAULT_SHARDS = Path(os.environ.get('GOTCHA_DATA_ROOT', '/scratch/user/u.db364833/GOTCHA-CP_Combined/'
                                     'GOTCHA-CP_Combined')) / 'New_Transfer' / 'shards'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--run', action='append', default=[], metavar='METHOD=CHECKPOINT',
                   help=f'one per method; methods: {sorted(presentation.PRESENTERS)}')
    p.add_argument('--polarization', default='hh')
    p.add_argument('--shard-root', type=Path, default=DEFAULT_SHARDS)
    args = p.parse_args(argv)
    for item in args.run:
        method, _, checkpoint = item.partition('=')
        if method not in presentation.PRESENTERS or not checkpoint:
            p.error(f'--run needs METHOD=CHECKPOINT with METHOD in {sorted(presentation.PRESENTERS)}')
        result = presentation.presenter(method)(checkpoint, shard_root=args.shard_root,
                                                polarization=args.polarization)
        presentation.write_method(args.output_root, result, result.source['half_extent_m'])
        print(f'{result.name}: written ({"relative + physical" if result.rcs_m2 is not None else "relative only"})')
    manifest = presentation.render(args.output_root)
    print(f"{args.output_root}: {len(manifest['methods'])} method(s) rendered")


if __name__ == '__main__':
    main()
