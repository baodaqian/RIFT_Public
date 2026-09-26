"""Write a trilinearly densified copy of an adaptive-RIFT GOTCHA checkpoint (tuning campaign A41).

Used to measure the per-update curvature a densify event causes before a run sets its
learning-rate factor: ``gotcha_rift_fit_check.py --curvature`` reads the copy like any checkpoint.
The Hessian of the data loss in the coefficients does not depend on their values (the render is
linear in them), so an untrained copy measures the curvature the trainer meets right after an event.

    python scripts_pvc/gotcha_densify_checkpoint.py SOURCE.pt OUT.pt --events 1 [--max-active N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rift.gotcha_training import with_legacy_keys
from rift.gotcha_dataset import load_region
from rift_pvc.gotcha_densify import trilinear_densify
from rift_pvc.gotcha_training import ChannelField


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--events', type=int, default=1)
    p.add_argument('--max-active', type=int, default=0, help='active cap (default: the checkpoint capacity)')
    p.add_argument('--energy-floor', type=float, default=1e-3)
    p.add_argument('--weight-scale', type=float, default=0.125)
    p.add_argument('--region-config', type=Path)
    args = p.parse_args(argv)
    checkpoint = torch.load(args.source, map_location='cpu', weights_only=False)
    recipe = with_legacy_keys(checkpoint['recipe'], checkpoint['recipe']['method'])
    region = load_region(checkpoint['dataset_contract']['region']['name'], args.region_config)
    reports = {}
    state = dict(checkpoint['model_state_dict'])
    for pol in sorted({k.split('.', 1)[0] for k in state}):
        head = ChannelField(recipe['method'], region, recipe, 'cpu')
        head.load_state_dict({k[len(pol) + 1:]: v for k, v in state.items() if k.startswith(pol + '.')}, strict=True)
        cap = args.max_active or int(head.field.active_mask.numel())
        reports[pol] = [trilinear_densify(head.field, max_active=cap, energy_floor=args.energy_floor,
                                          weight_scale=args.weight_scale, max_level=99) for _ in range(args.events)]
        state.update({f'{pol}.{k}': v for k, v in head.state_dict().items()})
    checkpoint['model_state_dict'] = state
    checkpoint['densified_copy'] = dict(source=str(args.source.resolve()), events=reports,
                                        purpose='curvature measurement only; not a trained state')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(json.dumps(reports, indent=1))


if __name__ == '__main__':
    main()
