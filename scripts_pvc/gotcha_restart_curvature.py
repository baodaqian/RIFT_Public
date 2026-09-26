#!/usr/bin/env python3
"""Stability number at a densify arm's next cosine restart (reviewer check, docs/RIFT_GOTCHA_Tune.md B60).

After each densify event the curvature rule (``densify_heads``) caps S = (base lr / eps) * lambda_max at its target,
with lambda_max the per-update Hessian's top eigenvalue on ``curvature_units`` TRAIN units at the current gain.
Nothing re-measures it between events, and a warm restart returns the rate to the base. Before a run crosses a
restart (``gotcha_extend_epochs.py --past-restart``, or a run declared past epoch 30), this re-measures lambda_max
with the trainer's own ``update_curvature`` on the units the last event used, and reports S at the base and at the
current rate beside that event's record. CPU only. It reads TRAIN units only, and their responses do not enter the
Hessian of the full-native loss.

    python scripts_pvc/gotcha_restart_curvature.py RUN_DIR/checkpoint_final.pt --shard-root <the run's shards> \\
        --output restart_curvature.json [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from rift.gotcha_training import with_legacy_keys  # noqa: E402
from rift_pvc.gotcha_training import ChannelField, RangeReadout, update_curvature  # noqa: E402
from gotcha_coherent_readout_pvc import rebuild_dataset  # noqa: E402


def last_event(history):
    """(epoch, curvature record) of the latest densify event that measured curvature."""
    for entry in reversed(history):
        record = ((entry.get('optimizer') or {}).get('densify') or {}).get('curvature')
        if record:
            return int(entry['epoch']), record
    return None, None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--dataset-root', type=Path, default=Path('/scratch/user/u.db364833/GOTCHA-CP_Combined/GOTCHA-CP_Combined'))
    p.add_argument('--shard-root', type=Path, required=True, help="the run's shard root")
    p.add_argument('--region-config', type=Path, default=Path('rift_pvc/regions/camry_box_v2.json'))
    p.add_argument('--polarization', default='hh')
    p.add_argument('--iterations', type=int, default=0, help="power iterations (0: the recipe's curvature_iterations)")
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--dry-run', action='store_true', help='rebuild, check the unit picks against the event, stop')
    p.add_argument('--output', type=Path)
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    recipe = with_legacy_keys(checkpoint['recipe'], checkpoint['recipe']['method'])
    spec = recipe['densify']
    if spec.get('lr_rule') != 'curvature':
        raise SystemExit('The recipe does not use the curvature learning-rate rule')
    ds = rebuild_dataset(checkpoint, args.dataset_root, args.shard_root, args.region_config)
    views = ds.viewpoints('train')
    units = int(spec['curvature_units'])
    picks = [views[int(i * len(views) / units)] for i in range(units)]
    event_epoch, event = last_event(checkpoint['history'])
    event_views = [list(u['view']) for u in event['units']] if event else None
    if event_views is not None and [list(v) for v in picks] != event_views:
        raise SystemExit(f'unit picks {picks} differ from the last event\'s {event_views}')
    optimizer = checkpoint['optimizer_state_dict']['param_groups']
    scheduler = checkpoint.get('scheduler_state_dict') or {}
    base = max(scheduler['base_lrs']) if scheduler.get('base_lrs') else max(g['lr'] for g in optimizer)
    current = max(g['lr'] for g in optimizer)
    eps = optimizer[0]['eps']
    report = dict(schema='gotcha_restart_curvature_v1', checkpoint=str(args.checkpoint.resolve()),
                  epoch=int(checkpoint['epoch']), shard_root=str(args.shard_root), units=[list(v) for v in picks],
                  base_lr=base, current_lr=current, eps=eps, target=spec.get('lr_target'),
                  scheduler_epoch=scheduler.get('last_epoch'), scheduler_t_i=scheduler.get('T_i'),
                  scheduler_t_cur=scheduler.get('T_cur'),
                  last_event=dict(epoch=event_epoch, lambda_max=event.get('lambda_max'), base_lr=event.get('base_lr'),
                                  stability_after=event.get('stability_after'), factor=event.get('factor'))
                  if event else None)
    print(json.dumps(report, indent=1), flush=True)
    if args.dry_run:
        return
    head = ChannelField(recipe['method'], ds.region, recipe, 'cpu')
    prefix = f'{args.polarization}.'
    head.load_state_dict({k[len(prefix):]: v for k, v in checkpoint['model_state_dict'].items()
                          if k.startswith(prefix)}, strict=True)
    head.eval()
    field = head.field
    readout = RangeReadout(ds.region, device='cpu')
    mean_power = checkpoint['training_statistics'][args.polarization]['mean_power']
    iterations = args.iterations or int(spec['curvature_iterations'])
    rows = []
    for view in picks:
        started = time.time()
        row = update_curvature(head, ds, readout, view, args.polarization, mean_power, 'rift', iterations=iterations)
        row.update(seconds=round(time.time() - started, 1))
        rows.append(row)
        print('unit', row, flush=True)
    lam = max(r['lambda_max'] for r in rows)
    report.update(measured=rows, lambda_max=lam, iterations=iterations,
                  gain=float(torch.exp(head.gain.log_mag.detach())),
                  active_points=int(field.active_mask.sum()), max_order=int(field.order[field.active_mask].max()),
                  stability_at_base=base / eps * lam, stability_at_current=current / eps * lam,
                  lambda_over_last_event=(lam / event['lambda_max']) if event else None)
    print(json.dumps({k: v for k, v in report.items() if k != 'measured'}, indent=1), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
