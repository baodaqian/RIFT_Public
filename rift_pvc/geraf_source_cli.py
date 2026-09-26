"""Root CLI for the released GeRaF stage, PVC (Intel XPU) twin.

Copy of ``rift/geraf_source_cli.py``. The argument list, its defaults, the
dry-run plan and the ``run`` contract are identical; the two adaptations are the
``--device`` default (the active accelerator instead of the literal ``'cuda'``,
and this is the flag the production command relies on because
``train_rift_dataset.py`` never passes ``--device`` for GeRaF) and the import of
the PVC training module.
"""
import argparse
import json
from pathlib import Path
from rift_pvc.geraf_source import DEFAULTS, recipe_for_data
from rift.geraf_source_data import RIFTSourceData
from rift.rift_dataset import DEFAULT_ROOT, resolve_object_inputs
from rift_pvc import accelerator


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', choices=('source_v1',), default='source_v1')
    p.add_argument('--object', help='Registered RIFT object/alias')
    p.add_argument('--dataset-root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--npz-path', type=Path)
    p.add_argument('--role-manifest', type=Path)
    p.add_argument('--checkpoint-dir', type=Path, required=True)
    p.add_argument('--cache-root', type=Path, help='One accumulated target grid/head and preparation recovery; old caches are incompatible')
    p.add_argument('--source-config', type=Path, help='JSON mapping of explicit source recipe overrides')
    p.add_argument('--device', default=str(accelerator.device()))
    p.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--resume-path', type=Path)
    p.add_argument('--dry-run', action='store_true', help='Metadata and full recipe only; no output writes or response reads')
    p.add_argument('--prepare-only', action='store_true', help='Prepare only the training accumulated grid; no per-view cubes or validation reads')
    for name, default in DEFAULTS.items():
        p.add_argument('--' + name.replace('_', '-'), dest=name, type=type(default), default=None)
    args = p.parse_args(argv)
    try:
        args.npz_path, args.role_manifest = resolve_object_inputs(
            object_name=args.object, dataset_root=args.dataset_root,
            npz_path=args.npz_path, role_manifest_path=args.role_manifest)
    except ValueError as exc:
        p.error(str(exc))
    if args.npz_path is None or args.role_manifest is None:
        p.error('Select --object or explicit registered --npz-path and --role-manifest')
    if args.resume_path and not args.resume:
        p.error('--resume-path conflicts with --no-resume')
    return args


def run(args):
    from rift_pvc.geraf_source_training import train
    config = json.loads(args.source_config.read_text()) if args.source_config else {}
    if not isinstance(config, dict):
        raise ValueError('Source configuration must be a JSON object')
    config.update({k: getattr(args, k) for k in DEFAULTS if getattr(args, k) is not None})
    data = RIFTSourceData(args.npz_path, args.role_manifest)
    recipe = recipe_for_data(config, data)
    if args.dry_run:
        plan = dict(implementation='source_v1', recipe=recipe, data_identity=data.identity,
                    dataset_identity=data.contract['experiment_contract']['dataset_identity'],
                    response_payload_read=False, output_dir=str(args.checkpoint_dir),
                    target_storage=recipe['target_storage'], persistent_per_view_volumes=0,
                    accumulated_volume_bytes_per_head=4 * recipe['mf_grid'] ** 3,
                    target_views={r: len(data.views(r)) for r in ('train', 'validation')})
        print(json.dumps(plan, indent=2))
        return plan
    result = train(data=data, output_dir=args.checkpoint_dir, config=config, device=args.device,
                   resume=str(args.resume_path) if args.resume_path else 'auto' if args.resume else None,
                   cache_root=args.cache_root, prepare_only=args.prepare_only)
    print(json.dumps(result, indent=2))
    if result.get('status') == 'interrupted':
        raise SystemExit(143)
    return result
