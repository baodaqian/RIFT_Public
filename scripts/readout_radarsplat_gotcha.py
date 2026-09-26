#!/usr/bin/env python3
"""Native GOTCHA RadarSplat power and same-checkpoint metric Gaussian readout."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
import train_radarsplat as lifecycle
from rift.radarsplat_gotcha import cache_from_run
from rift.radarsplat_release_training import readout as source_readout


def readout(*, run_root, polarization, checkpoint_path, device="cuda", role="validation",
            geometry_path=None, dataset_root=None, shard_root=None):
    cache = cache_from_run(run_root, polarization, dataset_root=dataset_root, shard_root=shard_root)
    checkpoint = lifecycle._load_checkpoint(checkpoint_path, torch.device("cpu"))
    return source_readout(checkpoint, checkpoint_path=checkpoint_path, cache_root=cache.root,
        cache=cache, device=torch.device(device), role=role, geometry_path=geometry_path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", required=True, type=Path)
    p.add_argument("--polarization", choices=("hh", "hv", "vh", "vv"), default="hh")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--dataset-root", type=Path)
    p.add_argument("--shard-root", type=Path)
    p.add_argument("--role", choices=("train", "validation"), default="validation")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--geometry", type=Path)
    args = p.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = readout(run_root=args.run_root, polarization=args.polarization, checkpoint_path=args.checkpoint,
        device=args.device, role=args.role, geometry_path=args.geometry, dataset_root=args.dataset_root, shard_root=args.shard_root)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
