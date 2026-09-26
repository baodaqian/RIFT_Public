#!/usr/bin/env python3
"""Unmasked native validation readout of a source_v1 GeRaF checkpoint.

Uses the exact recipe saved by either root trainer. Reserved-test evaluation
is deliberately absent from this development readout. RIFT mesh scoring is
available separately through eval_geraf_geometry.py.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
from rift.geraf_source_data import RIFTSourceData, GOTCHASourceData
from rift.geraf_source_training import SourceTargets, load_selected_models, evaluate, file_hash


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('rift', 'gotcha'), required=True)
    p.add_argument('--object')
    p.add_argument('--dataset-root', type=Path)
    p.add_argument('--role-manifest', type=Path, help='RIFT run role_manifest.json; required for a subset-trained checkpoint')
    p.add_argument('--num-train', type=int, help='GOTCHA checkpoint training sectors; omitted preserves the parent contract')
    p.add_argument('--shard-root', type=Path)
    p.add_argument('--region', default='camry')
    p.add_argument('--region-config', type=Path)
    p.add_argument('--passes', type=int, nargs='+', default=list(range(1,9)))
    p.add_argument('--polarizations', nargs='+', default=['hh'])
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--cache-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if args.dataset == 'rift':
        from rift.rift_dataset import DEFAULT_ROOT, resolve_object_inputs
        if not args.object:
            p.error('--dataset rift requires --object')
        if args.num_train is not None:
            p.error('RIFT source readout selects its exact --role-manifest; --num-train is for GOTCHA')
        data = RIFTSourceData(*resolve_object_inputs(object_name=args.object,
            dataset_root=args.dataset_root or DEFAULT_ROOT, role_manifest_path=args.role_manifest))
    else:
        from rift.gotcha_dataset import DEFAULT_ROOT, GOTCHADataset, load_region
        from rift.gotcha_pulse_sampling import pulse_limit_from_contract
        from rift.gotcha_frequency_selection import kwargs_from_contract
        if args.object or args.role_manifest:
            p.error('GOTCHA uses --region rather than a RIFT object')
        data = GOTCHASourceData(GOTCHADataset(args.dataset_root or DEFAULT_ROOT, shard_root=args.shard_root,
            region=load_region(args.region, args.region_config), passes=tuple(args.passes),
            polarizations=tuple(args.polarizations), num_train=args.num_train,
            pulses_per_sector=pulse_limit_from_contract(checkpoint['contract']['native_gotcha_contract']),
            **kwargs_from_contract(checkpoint['contract']['native_gotcha_contract'])))
    models, selected = load_selected_models(checkpoint, data, args.device)
    targets = SourceTargets(args.cache_root, data, checkpoint['recipe'])
    targets.start()
    metrics = evaluate(models, data, checkpoint['recipe'], targets, args.device, lambda: False)
    report = dict(schema='rift_geraf_source_validation_v1', role='validation',
        data_identity=data.identity, recipe=checkpoint['recipe'], checkpoint_step=checkpoint['step'],
        checkpoint_sha256=file_hash(args.checkpoint), selection_record=selected, metrics=metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


if __name__ == '__main__':
    main()
