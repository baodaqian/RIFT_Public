#!/usr/bin/env python3
"""RIFT's role_fit numbers for a native SpINR GOTCHA checkpoint (BASELINES_CAMRY_BOX.md).

Pooled over every unit of a role (or every Nth with ``--unit-stride``), in the full-native domain RIFT trains and
reads: RelMSE = sum|yhat - y|^2 / sum|y|^2, energy ratio e = sum|yhat|^2 / sum|y|^2 and real correlation
rho = Re<yhat, y> / sqrt(sum|yhat|^2 sum|y|^2), so RelMSE = 1 + e - 2 rho sqrt(e). The render is SpINR's own
validation render (``rift_pvc.spinr_gotcha_training.evaluate``: the tiled field, BatchedNativeKernel, the
checkpoint's initial scale). The dataset is rebuilt from the training command's own arguments (after ``--``),
so its identity must equal the checkpoint's. TEST stays sealed.

    python scripts_pvc/gotcha_spinr_role_fit_pvc.py RUN_DIR/checkpoint_best.pt --roles validation train \\
        --unit-stride 4 --output fit.json -- <train_gotcha_dataset_pvc.py arguments>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@torch.no_grad()
def role_fit(heads, dataset, points, volumes, scales, recipe, device, role, unit_stride):
    from rift_pvc.spinr_native_batched import BatchedNativeKernel
    from train_spinr_style_pvc import evaluate_neural_field_tiled
    out = {}
    for pol in dataset.polarizations:
        field = evaluate_neural_field_tiled(heads[pol], points, neural_point_tile=recipe['neural_point_tile'])
        s = dict(error=0., target=0., prediction=0., cross=0., units=0, pulses=0)
        for view in dataset.viewpoints(role)[::unit_stride]:
            observations = list(dataset.observations(*view, pol))
            kernel = BatchedNativeKernel(observations, dataset.region, device=device, point_tile=recipe['renderer_point_tile'])
            yhat = kernel.render(points, field, volumes, scales[pol]['value'], selected=False)
            y = torch.as_tensor(np.stack([o.response for o in observations]), dtype=torch.complex128, device=device)
            s['error'] += float((yhat - y).abs().square().sum())
            s['target'] += float(y.abs().square().sum())
            s['prediction'] += float(yhat.abs().square().sum())
            s['cross'] += float((yhat * y.conj()).real.sum())
            s['units'] += 1
            s['pulses'] += len(observations)
        e = s['prediction'] / s['target']
        rho = s['cross'] / math.sqrt(max(s['prediction'] * s['target'], 1e-300))
        out[pol] = dict(role=role, full_native_rel_mse=s['error'] / s['target'], energy_ratio=e, correlation=rho,
                        e_over_rho2=e / rho ** 2 if rho else None, units=s['units'], pulses=s['pulses'],
                        unit_stride=unit_stride)
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index('--') if '--' in argv else len(argv)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--roles', nargs='+', default=['validation', 'train'], choices=('validation', 'train'))
    p.add_argument('--unit-stride', type=int, default=1, help='score every Nth unit of each role (1 = all)')
    p.add_argument('--device', default='xpu')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv[:split])
    import train_gotcha_dataset_pvc as cli
    from rift.spinr_style import SpinrStyleINR, gauss_legendre_cell_grid
    from train import load_tensor_checkpoint
    train_args = cli.parse_args(argv[split + 1:] + ['--dry-run'])
    dataset, _ = cli.make_plan(train_args)
    saved = load_tensor_checkpoint(args.checkpoint, map_location='cpu')
    if saved['dataset_identity'] != dataset.identity:
        raise SystemExit('Rebuilt dataset differs from the checkpoint; pass the training command unchanged')
    recipe, device = saved['recipe'], torch.device(args.device)
    heads = torch.nn.ModuleDict({pol: SpinrStyleINR(support_m=dataset.region.half_extent_m)
                                 for pol in dataset.polarizations}).to(device)
    heads.load_state_dict(saved['model_state_dict'], strict=True)
    heads.eval()
    points, volumes = gauss_legendre_cell_grid(recipe['grid_size'], nodes_per_cell=recipe['nodes_per_cell'],
                                               support_m=dataset.region.half_extent_m, device=device, dtype=torch.float64)
    result = dict(schema='gotcha_spinr_role_fit_v1', checkpoint=str(args.checkpoint), epoch=saved['epoch'],
                  best_epoch=saved.get('best_epoch'), dataset_identity=dataset.identity, test_accessed=False,
                  roles={role: role_fit(heads, dataset, points, volumes, saved['initial_scales'], recipe, device, role,
                                        args.unit_stride) for role in args.roles})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result['roles'], indent=1))


if __name__ == '__main__':
    main()
