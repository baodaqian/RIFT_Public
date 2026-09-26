#!/usr/bin/env python3
"""The trainer's own ``role_fit`` numbers for any saved GOTCHA RIFT checkpoint (PVC twin of the SpINR tool).

``rift_pvc.gotcha_training`` measures the TRAIN fit only every ``train_eval_every`` epochs, so a checkpoint saved at
another epoch (e.g. a validation-selected ``checkpoint_best.pt`` at an odd epoch) has none. This tool rebuilds the heads
exactly as ``scripts_pvc/eval_gotcha_heldout_pvc.py``'s RIFT adapter does and calls the same ``role_fit`` (pooled
full-native RelMSE, e, rho, e/rho^2) with the trainer's ``RangeReadout``. The dataset is rebuilt from the training
command's data arguments (after ``--``), and ``validate_checkpoint`` requires its contract to equal the checkpoint's.
TEST stays sealed. The validation role must reproduce the checkpoint's logged selection RelMSE.

    python scripts_pvc/gotcha_rift_role_fit_pvc.py RUN/checkpoint_best.pt --roles validation train \\
        --output fit.json -- <train_gotcha_dataset_pvc.py data arguments>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index('--') if '--' in argv else len(argv)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--roles', nargs='+', default=['validation', 'train'], choices=('validation', 'train'))
    p.add_argument('--device', default='xpu')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv[:split])
    import warnings
    warnings.filterwarnings('error', message='Aten Op fallback from XPU to CPU')
    if args.output.exists():
        raise FileExistsError(args.output)
    import train_gotcha_dataset_pvc as cli
    from rift.gotcha_dataset import validate_checkpoint
    from rift.gotcha_training import with_legacy_keys
    from rift_pvc import gotcha_nufft as nufft
    from rift_pvc import gotcha_training as pvc
    train_args = cli.parse_args(argv[split + 1:] + ['--dry-run'])
    dataset, _ = cli.make_plan(train_args)
    device = torch.device(args.device)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    recipe = with_legacy_keys(ck['recipe'], 'rift')
    ck['recipe'] = recipe
    validate_checkpoint(ck, dataset, recipe)
    heads = torch.nn.ModuleDict({pol: pvc.ChannelField('rift', dataset.region, recipe, device)
                                 for pol in dataset.polarizations})
    heads.load_state_dict(ck['model_state_dict'], strict=True)
    nufft.attach_grids(heads, dataset, recipe, device)
    heads.eval()
    readout = pvc.RangeReadout(dataset.region, device=device)
    history = ck.get('history') or []
    logged = history[-1].get('validation') if history else None
    with torch.no_grad():
        roles = {role: pvc.role_fit(heads, dataset, readout, 'rift', role) for role in args.roles}
    result = dict(schema='gotcha_rift_role_fit_v1', checkpoint=str(args.checkpoint), epoch=int(ck['epoch']),
                  cursor=int(ck.get('cursor') or 0), updates=int(ck['updates']), dataset_identity=dataset.identity,
                  logged_validation_rel_mse=(logged or {}).get('full_native_complex_rel_mse'), test_accessed=False,
                  device=str(device), roles=roles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, default=float) + '\n')
    print(json.dumps(result, indent=1, default=float))


if __name__ == '__main__':
    main()
