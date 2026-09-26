"""Extend a completed adaptive-RIFT GOTCHA run past its recipe's epochs (tuning campaign A55, item 4).

The trainer refuses a resume whose recipe differs from the checkpoint's, epochs included. This tool
writes a new output root from which the completed run resumes as if its recipe had named more
epochs. It copies the run directory's records, sets only ``recipe['epochs']`` in the completed
checkpoint and in recipe.json, and appends a declared ``extension`` record to the last history entry.
The trainer carries that entry into every later save and into history.json. Nothing else changes.
The optimizer, scheduler, RNG, statistics, best value and densify record resume as saved.

So epochs 1..E_old are the run that already finished, and epochs E_old+1..E_new are what a run
declared with E_new epochs would have trained. Before its last epoch the trainer reads
``recipe['epochs']`` only in loop bounds, in "epoch < epochs" guards on densify events and guard
measurements, and for the final TRAIN score, which train_eval_every already takes on even epochs.
A densify event after the old final epoch E_old fires on the extension as declared; one at E_old
(skipped then, as the last epoch) is run by the trainer's resume branch before epoch E_old + 1,
where a declared run would have run it (A62: this is what lets short declared runs, e.g. the
12-epoch gain twins, extend to 40).

Refused:
- an incomplete checkpoint (mid-epoch, or epoch != recipe epochs);
- a target not above the old one;
- an extension that trains after a cosine restart or an SH refinement event (either may raise the
  step's stability number, B53/B59) unless the curvature guard covers it (--curvature-guard restart
  and/or growth; recipe key densify.curvature_guard) or --past-restart declares that the curvature
  was re-measured instead.

--curvature-guard adds the opt-in guard (A57) to the recipe; the resumed run then needs the matching
--densify-curvature-guard flag. Unlike an epochs-only extension, a run extended with a newly added
guard is not the run that would have been declared with it (earlier restarts ran unguarded); the
extension record says so.

    python scripts_pvc/gotcha_extend_epochs.py RUN_DIR NEW_OUTPUT_ROOT --epochs 30
    # then resume with the run's own knobs, --output-root NEW_OUTPUT_ROOT --epochs 30 --resume <printed path>
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
from pathlib import Path

import torch

from rift.gotcha_training import atomic_json, atomic_save

SCHEMA = 'gotcha_epoch_extension_v1'
RECORDS = ('dataset.json', 'initialization.json', 'checkpoint_best.pt')


def restart_epochs(t0, t_mult, until):
    """1-based epochs after which CosineAnnealingWarmRestarts restarts (the next epoch runs at the base rate)."""
    out, edge, period = [], int(t0), int(t0)
    while edge <= until:
        out.append(edge)
        period *= int(t_mult)
        edge += period
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dir', type=Path, help='the completed run directory (…/<region>/<id16>/…/rift_full_native)')
    p.add_argument('output_root', type=Path, help='new --output-root for the extension (must not hold this run)')
    p.add_argument('--epochs', type=int, required=True, help='the extended total epoch count')
    p.add_argument('--checkpoint', default='checkpoint_final.pt', help='completed checkpoint in RUN_DIR')
    p.add_argument('--past-restart', action='store_true',
                   help='allow training after restarts or refinement events the guard does not cover '
                        '(declare the curvature measurement, B53/B59)')
    p.add_argument('--curvature-guard', nargs='+', choices=('growth', 'restart'), default=[],
                   help='add the opt-in curvature guard (A57) to the recipe')
    p.add_argument('--note', default='', help='reason recorded with the extension')
    args = p.parse_args(argv)

    run_dir = args.run_dir.resolve()
    # The trainer names its run directory OUTPUT_ROOT/<region>/<id16>[/<subset>]/<method dir>; keep that tail.
    source_root = next((q for q in run_dir.parents if (q / 'train.log').exists()), None)
    if source_root is None:
        raise SystemExit(f'No output root with train.log above {run_dir}')
    tail = run_dir.relative_to(source_root)
    target = args.output_root.resolve() / tail
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f'{target} exists and is not empty; resume it with its own checkpoint_latest.pt instead')

    saved = torch.load(run_dir / args.checkpoint, map_location='cpu', weights_only=False)
    recipe = dict(saved['recipe'])
    old = int(recipe['epochs'])
    if saved['epoch'] != old or saved['cursor'] != 0 or saved.get('order') is not None or len(saved['history']) != old:
        raise SystemExit(f'{args.checkpoint} is not a completed run (epoch {saved["epoch"]} of {old}, '
                         f'cursor {saved["cursor"]})')
    if args.epochs <= old:
        raise SystemExit(f'--epochs {args.epochs} must exceed the completed {old}')
    late = sorted(int(e) for e in (recipe.get('densify') or {}).get('epochs', []) if old <= int(e) < args.epochs)
    densify = dict(recipe.get('densify') or {})
    guard = sorted(set(densify.get('curvature_guard') or []) | set(args.curvature_guard))
    added = sorted(set(args.curvature_guard) - set(densify.get('curvature_guard') or []))
    if added:
        if densify.get('lr_rule') != 'curvature':
            raise SystemExit('--curvature-guard needs a densify recipe with the curvature rule')
        densify['curvature_guard'] = guard
        recipe['densify'] = densify
    schedule = recipe.get('optimizer_schedule') or {}
    crossed = []
    if schedule.get('scheduler') == 'cosine_warm_restarts':
        # A restart after epoch e (e in [old, new)) sets the rate of epoch e + 1 to the base.
        crossed = [e for e in restart_epochs(schedule['t0'], schedule['t_mult'], args.epochs) if old <= e < args.epochs]
    every = int(recipe.get('refine_every') or 0)
    growth = ([e for e in range(every, args.epochs, every) if e >= old]
              if every and int(recipe.get('sh_degree', 0)) > 0 else [])
    unguarded = [f'restart after epoch {e}' for e in crossed if 'restart' not in guard] + \
                [f'refinement event after epoch {e}' for e in growth if 'growth' not in guard]
    if unguarded and not args.past_restart:
        raise SystemExit(f'extending {old} -> {args.epochs} trains after {", ".join(unguarded)}; add '
                         f'--curvature-guard, or pass --past-restart after re-measuring the curvature (B53/B59)')

    record = dict(schema=SCHEMA, from_epochs=old, to_epochs=args.epochs, source_run=str(run_dir),
                  source_checkpoint=args.checkpoint, output_root=str(args.output_root.resolve()),
                  created=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
                  tool='scripts_pvc/gotcha_extend_epochs.py',
                  changed="recipe['epochs'] only" if not added else "recipe['epochs'] and densify.curvature_guard",
                  restarts_crossed=crossed, refinement_events_crossed=growth, curvature_guard_added=added,
                  densify_events_ahead=late,
                  unguarded_acknowledged=unguarded, note=args.note)
    recipe['epochs'] = args.epochs
    saved['recipe'] = recipe
    saved['history'][-1] = dict(saved['history'][-1], extension=record)
    target.mkdir(parents=True, exist_ok=True)
    for name in RECORDS:
        if (run_dir / name).exists():
            shutil.copyfile(run_dir / name, target / name)
    atomic_save(target / 'checkpoint_latest.pt', saved)
    atomic_json(target / 'recipe.json', recipe)
    atomic_json(target / 'history.json', saved['history'])
    atomic_json(target / 'extension.json', record)
    print(json.dumps(dict(record, resume=str(target / 'checkpoint_latest.pt')), indent=1))


if __name__ == '__main__':
    main()
