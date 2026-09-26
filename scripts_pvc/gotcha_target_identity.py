#!/usr/bin/env python3
"""Expected dataset identity of a GOTCHA target's shards under the full unit split (new targets, docs A77).

Builds the dataset as ``gotcha_data_coherent_image.py`` does (the construction whose identity matched the Camry F5
arms, 809186e5...): all 8 passes, hh, every pulse, frequency stride 2, the full unit split (stride 1, pass 4, 10%).
Reads metadata only. For the campaign seat's launch check.

    python scripts_pvc/gotcha_target_identity.py --shard-root <keep6 shards> --region sentra_box_v1 \\
        --region-config rift_pvc/regions/gotcha_new_targets.json
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from rift.gotcha_dataset import GOTCHADataset, load_region  # noqa: E402
from rift_pvc.gotcha_unit_split import apply_unit_split  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--shard-root', type=Path, required=True)
    p.add_argument('--region', required=True)
    p.add_argument('--region-config', type=Path)
    p.add_argument('--dataset-root', type=Path, default=Path('/scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined'))
    args = p.parse_args(argv)
    ds = GOTCHADataset(args.dataset_root, shard_root=args.shard_root, passes=tuple(range(1, 9)), polarizations=('hh',),
                       region=load_region(args.region, args.region_config), pulses_per_sector=0, frequency_stride=2)
    apply_unit_split(ds, stride=1, heldout_pass=4, heldout_fraction=0.1)
    print(ds.identity)


if __name__ == '__main__':
    main()
