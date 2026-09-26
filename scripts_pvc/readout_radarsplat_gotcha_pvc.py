#!/usr/bin/env python3
"""PVC twin of ``scripts/readout_radarsplat_gotcha.py``: native GOTCHA RadarSplat power/geometry readout.

Same CLI; ``--device`` defaults to the accelerator device and the readout runs
on the PVC torch-mirror renderer. The result carries the backend record of the
checkpoint (``backend.json``) and of the readout itself (D5).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
import train_radarsplat as lifecycle  # noqa: E402
from rift_pvc import accelerator  # noqa: E402
from rift_pvc.radarsplat_gotcha import cache_from_run  # noqa: E402
from rift_pvc.radarsplat_release_training import readout as source_readout  # noqa: E402


def readout(*, run_root, polarization, checkpoint_path, device=None, role="validation",
            geometry_path=None, dataset_root=None, shard_root=None):
    device = accelerator.device() if device is None else device
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
    p.add_argument("--device", default=str(accelerator.device()))
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
