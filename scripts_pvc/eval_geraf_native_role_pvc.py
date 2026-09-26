#!/usr/bin/env python3
"""GeRaF's native matched-filter amplitude RelMSE on a chosen RIFT-dataset role, reserved TEST included (PVC twin).

For the manuscript's Table 6 "B787 native reduced-observable diagnostics": the same readout the trainer uses for
checkpoint selection (``rift_pvc.geraf_source_training.evaluate``: pooled sum (|MF_pred| - |MF_meas|)^2 over
sum |MF_meas|^2 on the lazily built 48^3 matched-filter targets, the recipe's seeded frame sampling), copied here
with the role as an argument instead of patching the shared function. The data class is RIFTSourceData with the
reserved-test role opened explicitly (``--role test`` requires ``--allow-reserved-test``); the object contract, and
hence the checkpoint's identity check, is unchanged. The checkpoint must be the unmasked validation-selected one
(``load_selected_models``). Run ``--role validation`` first: it must reproduce the selection record's
``mf_relative_mse`` (within the autocast tolerance when the device differs from training).

    python scripts_pvc/eval_geraf_native_role_pvc.py --object b787 --role test --allow-reserved-test \\
        --role-manifest RUN/b787/role_manifest.json --checkpoint RUN/b787/geraf/source_v1/checkpoints/checkpoint_best.pth.tar \\
        --cache-root SCRATCH/targets_test --output out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from rift.geraf_source_data import RIFTSourceData  # noqa: E402
from rift_pvc import accelerator  # noqa: E402
from rift_pvc.geraf_source_training import SourceTargets, file_hash, load_selected_models  # noqa: E402

ROLES = {'validation': 'validation', 'test': 'reserved_test'}


class RIFTSourceDataWithRole(RIFTSourceData):
    """RIFTSourceData with one extra evaluation role opened explicitly; train and validation are unchanged."""

    def __init__(self, npz_path, role_manifest, role, allow_reserved_test):
        from rift.rift_dataset import evaluation_role_indices, load_object_contract
        super().__init__(npz_path, role_manifest)
        if role == 'test':
            if allow_reserved_test is not True:
                raise PermissionError('--role test requires --allow-reserved-test')
            self.arrays, contract = load_object_contract(
                npz_path, role_manifest, response_roles=('train', 'validation', 'reserved_test'),
                allow_reserved_test=True)
            if contract['dataset_identity'] != self.contract['dataset_identity']:
                raise ValueError('Opening the reserved-test role changed the object contract')
            self.roles['test'] = tuple(map(int, evaluation_role_indices(contract, 'test', allow_reserved_test=True)))

    def views(self, role):
        if role not in self.roles:
            raise PermissionError(f'Role {role!r} is not open in this readout')
        return self.roles[role]


def evaluate_role(models, data, recipe, targets, device, role, max_views=None):
    """``rift_pvc.geraf_source_training.evaluate`` with the role as an argument (MF and native-complex totals)."""
    from rift_pvc.geraf_source_training import fixed_numpy_seed, predict_native, sample_frame
    totals = dict(mf_sse=0., mf_energy=0., mf_count=0, native_sse=0., native_energy=0., native_count=0)
    for head in data.heads:
        model = models[head]
        for index, view in enumerate(data.views(role)[:max_views]):
            volume = targets.get(role, view, head, device, lambda: False)
            acquisition = data.acquisition(role, view, head, recipe, device)
            with fixed_numpy_seed(recipe['seed'] + index):
                frame = sample_frame(acquisition, recipe, data.key(role, view, head), volume, volume)
            prediction, inside = predict_native(model, frame, acquisition, return_inbounds=True)
            query = frame['tgt_sampled_poses_glb'].reshape(-1, 3)[inside]
            if not len(query):
                raise ValueError(f'Source GeRaF interpolation has no inbounds {role} queries')
            with torch.no_grad():
                predicted_mf = acquisition.matched_filter(prediction, query).abs()
                measured_mf = frame['mf_sampled_value'].flatten()[inside].double()
                if not bool(torch.isfinite(prediction).all() & torch.isfinite(predicted_mf).all()):
                    raise ValueError(f'Nonfinite GeRaF {role} prediction')
                totals['mf_sse'] += float((predicted_mf - measured_mf).square().sum())
                totals['mf_energy'] += float(measured_mf.square().sum())
                totals['mf_count'] += measured_mf.numel()
                measured = data.response(role, view, head, device) / recipe['trans_power']
                totals['native_sse'] += float((prediction - measured).abs().square().sum())
                totals['native_energy'] += float(measured.abs().square().sum())
                totals['native_count'] += measured.numel()
            if (index + 1) % 100 == 0:
                print(f'{role}: {index + 1}/{len(data.views(role))} views, running MF RelMSE '
                      f'{totals["mf_sse"] / totals["mf_energy"]:.6f}', flush=True)
    if not totals['mf_count'] or totals['mf_energy'] <= 0 or totals['native_energy'] <= 0:
        raise ValueError(f'No nonzero {role} targets')
    return dict(**totals, views=len(data.views(role)[:max_views]), mf_magnitude_mse=totals['mf_sse'] / totals['mf_count'],
                mf_relative_mse=totals['mf_sse'] / totals['mf_energy'],
                native_complex_relative_mse=totals['native_sse'] / totals['native_energy'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--object', required=True)
    p.add_argument('--dataset-root', type=Path)
    p.add_argument('--role-manifest', type=Path, required=True)
    p.add_argument('--role', choices=tuple(ROLES), required=True)
    p.add_argument('--allow-reserved-test', action='store_true')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--cache-root', type=Path, required=True, help='fresh directory for the lazy target manifest')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default=str(accelerator.device()))
    p.add_argument('--max-views', type=int, help='smoke only: score the first N views of the role')
    args = p.parse_args(argv)
    device = torch.device(args.device)
    if device.type in ('cuda', 'xpu') and device.type != accelerator.backend():
        raise RuntimeError(f'Requested {device.type}, active backend is {accelerator.backend()}')
    import warnings
    warnings.filterwarnings('error', message='Aten Op fallback from XPU to CPU')
    if args.output.exists():
        raise FileExistsError(args.output)
    from rift.rift_dataset import DEFAULT_ROOT, resolve_object_inputs
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    data = RIFTSourceDataWithRole(*resolve_object_inputs(object_name=args.object,
                                  dataset_root=args.dataset_root or DEFAULT_ROOT, role_manifest_path=args.role_manifest),
                                  role=args.role, allow_reserved_test=args.allow_reserved_test)
    models, selected = load_selected_models(checkpoint, data, args.device)
    targets = SourceTargets(args.cache_root, data, checkpoint['recipe'])
    targets.start()
    metrics = evaluate_role(models, data, checkpoint['recipe'], targets, args.device, args.role, args.max_views)
    report = dict(schema='rift_geraf_native_role_v1', role=ROLES[args.role], object=args.object,
                  data_identity=data.identity, checkpoint=str(args.checkpoint), checkpoint_step=checkpoint['step'],
                  checkpoint_sha256=file_hash(args.checkpoint), device_backend=device.type,
                  autocast_dtype='float16' if device.type in ('cuda', 'xpu') else None,
                  selection_record=selected, metrics=metrics,
                  status='smoke' if args.max_views else 'complete',
                  validation_reproduction=(metrics['mf_relative_mse'] / selected['mf_relative_mse'] - 1
                                           if args.role == 'validation' and not args.max_views else None))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in report.items() if k != 'selection_record'}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
